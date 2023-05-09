import torch
from torch import nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTEmbeddings

from modules.core import TaskEmbeddings, TaskDecoderBlock, LateralEncoderBlock, TransparentEncoderBlock


class EncDecBUTD(nn.Module):

    def __init__(self, num_tasks: int, num_args: int, num_classes: int, enc_config: ViTConfig, dec_config: ViTConfig,
                 use_butd: bool, use_sinusoidal=False, aux_head_size=None):
        super().__init__()
        self.use_butd = use_butd
        self.num_tasks = num_tasks

        encoder_base = ViTModel(enc_config, add_pooling_layer=False)
        self.encoder = encoder_base.encoder
        self.embeddings = TaskEmbeddings(encoder_base.get_input_embeddings(), enc_config.hidden_size, num_tasks=0,
                                         use_sinusoidal=use_sinusoidal)

        self.decoder_blocks = nn.ModuleList(
            [TaskDecoderBlock(enc_config=enc_config, dec_config=dec_config) for _ in
             range(dec_config.num_hidden_layers)])
        self.classifier = nn.Linear(in_features=dec_config.hidden_size, out_features=num_classes)
        self.bu_classifier = None
        if aux_head_size is not None:
            self.bu_classifier = nn.Linear(in_features=enc_config.hidden_size, out_features=aux_head_size)
            self.final_encoder_layernorm = nn.LayerNorm(enc_config.hidden_size, eps=enc_config.layer_norm_eps)

        self.task_embeddings = nn.Linear(in_features=num_tasks, out_features=dec_config.hidden_size)
        self.argument_embeddings = nn.Linear(in_features=num_args, out_features=dec_config.hidden_size)
        self.task_position_embeddings = nn.Parameter(torch.zeros(1, 2, dec_config.hidden_size))

        self.final_layernorm = nn.LayerNorm(dec_config.hidden_size, eps=dec_config.layer_norm_eps)

    def forward(self, img: torch.Tensor, task: torch.FloatTensor, argument: torch.FloatTensor):
        encoder_hidden_states = self.encoder(self.embeddings(img, [], []),
                                             output_hidden_states=True).hidden_states

        task_tokens = self.task_embeddings(task).unsqueeze(1)
        argument_tokens = self.argument_embeddings(argument).unsqueeze(1)

        decoder_state = torch.cat((task_tokens, argument_tokens), dim=1) + self.task_position_embeddings
        if self.use_butd:
            for encoder_hidden_state, block in zip(encoder_hidden_states[::-1], self.decoder_blocks):
                decoder_state = block(decoder_state, encoder_hidden_state)
        else:
            for block in self.decoder_blocks:
                decoder_state = block(decoder_state, encoder_hidden_states[-1])

        if self.bu_classifier is None:
            return self.classifier(self.final_layernorm(decoder_state[:, 0, :]))
        else:
            return self.classifier(self.final_layernorm(decoder_state[:, 0, :])), self.bu_classifier(
                self.final_encoder_layernorm(encoder_hidden_states[-1][:, 0, :]))


class MixingBUTD(nn.Module):

    def __init__(self, num_tasks: int, num_args: int, num_classes: int, config: ViTConfig, total_token_size: int,
                 use_self_attention: bool, mix_with: str, use_sinusoidal=False, aux_head_size=None):
        super().__init__()

        self.task_encoder_blocks = nn.ModuleList(
            [TransparentEncoderBlock(config, mix_with) for _ in range(config.num_hidden_layers)])
        self.image_embeddings = TaskEmbeddings(ViTEmbeddings(config).patch_embeddings, config.hidden_size, num_tasks=0,
                                               use_sinusoidal=use_sinusoidal,
                                               num_cls_tokens=1 if aux_head_size is None else 2)
        self.image_encoder_blocks = nn.ModuleList(
            [LateralEncoderBlock(config, mix_with, use_self_attention) for _ in range(config.num_hidden_layers)])

        self.task_embeddings = nn.Linear(in_features=num_tasks, out_features=config.hidden_size)
        self.argument_embeddings = nn.Linear(in_features=num_args, out_features=config.hidden_size)
        self.relu = nn.ReLU()

        self.stretch_embedding = nn.Linear(in_features=2 * config.hidden_size, out_features=total_token_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.task_position_embeddings = nn.Parameter(
            torch.zeros(1, total_token_size // config.hidden_size, config.hidden_size))
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=num_classes)

        self.aux_classifier = None
        if aux_head_size is not None:
            self.aux_classifier = nn.Linear(in_features=config.hidden_size, out_features=aux_head_size)

    def forward(self, img: torch.Tensor, task: torch.FloatTensor, argument: torch.FloatTensor):

        embedded_image = self.image_embeddings(img, [], [])

        task_tokens = self.task_embeddings(task)
        argument_tokens = self.argument_embeddings(argument)

        task_input = self.stretch_embedding(self.relu(torch.cat((task_tokens, argument_tokens), dim=1))).view(
            *embedded_image.shape) + self.task_position_embeddings

        mixing_layers = []

        for block in self.task_encoder_blocks:
            task_input, mixing_state = block(task_input)
            mixing_layers.append(mixing_state)

        for mixing_state, block in zip(mixing_layers[::-1], self.image_encoder_blocks):
            embedded_image = block(embedded_image, mixing_state)

        normed_res = self.layernorm(embedded_image)
        if self.aux_classifier is None:
            return self.classifier(normed_res[:, 0, :])
        else:
            return self.classifier(normed_res[:, 0, :]), self.aux_classifier(normed_res[:, 1, :])


class EncoderBUTD(nn.Module):
    def __init__(self, num_tasks: int, num_args: int, num_classes: int, enc_config: ViTConfig, use_sinusoidal=False,
                 aux_head_size=None):
        super().__init__()
        encoder = ViTForImageClassification(enc_config)
        self.embeddings = TaskEmbeddings(encoder.vit.get_input_embeddings(), enc_config.hidden_size,
                                         use_sinusoidal=use_sinusoidal,
                                         num_cls_tokens=1 if aux_head_size is None else 2)
        self.layer_norm = encoder.vit.layernorm
        self.encoder = encoder.vit.encoder
        self.classifier = nn.Linear(in_features=enc_config.hidden_size, out_features=num_classes)
        self.task_embeddings = nn.Linear(in_features=num_tasks, out_features=enc_config.hidden_size)
        self.argument_embeddings = nn.Linear(in_features=num_args, out_features=enc_config.hidden_size)

        self.aux_classifier = None
        if aux_head_size is not None:
            self.aux_classifier = nn.Linear(in_features=enc_config.hidden_size, out_features=aux_head_size)

    def forward(self, img: torch.FloatTensor, task: torch.FloatTensor, argument: torch.FloatTensor):
        task_embed = self.task_embeddings(task).unsqueeze(1)
        argument_embed = self.argument_embeddings(argument).unsqueeze(1)
        image_embed = self.embeddings(img, (task_embed,), (argument_embed,))
        encoder_result = self.encoder(image_embed)[0]
        encoder_result = self.layer_norm(encoder_result)
        if self.aux_classifier is None:
            return self.classifier(encoder_result[:, 0, :])
        else:
            return self.classifier(encoder_result[:, 0, :]), self.aux_classifier(encoder_result[:, 1, :])


class LRBothEncoder(nn.Module):
    def __init__(self, num_classes, enc_config: ViTConfig):
        super().__init__()
        encoder = ViTForImageClassification(enc_config)
        self.embeddings = TaskEmbeddings(encoder.vit.get_input_embeddings(), enc_config.hidden_size, num_tasks=2)
        self.layer_norm = encoder.vit.final_encoder_layernorm
        self.encoder = encoder.vit.encoder
        self.classifier_1 = nn.Linear(in_features=enc_config.hidden_size, out_features=num_classes)
        self.classifier_2 = nn.Linear(in_features=enc_config.hidden_size, out_features=num_classes)
        self.task_embeddings = nn.Linear(in_features=2, out_features=enc_config.hidden_size)
        self.argument_embeddings = nn.Linear(in_features=num_classes, out_features=enc_config.hidden_size)

    def forward(self, img: torch.FloatTensor, task_1: torch.FloatTensor, argument_1: torch.FloatTensor,
                task_2: torch.FloatTensor = None, argument_2: torch.FloatTensor = None):
        task_1_embed = self.task_embeddings(task_1).unsqueeze(1)
        argument_1_embed = self.argument_embeddings(argument_1).unsqueeze(1)
        if task_2 is not None and argument_2 is not None:
            task_2_embed = self.task_embeddings(task_2).unsqueeze(1)
            argument_2_embed = self.argument_embeddings(argument_2).unsqueeze(1)
        else:
            task_2_embed = torch.zeros_like(task_1_embed)
            argument_2_embed = torch.zeros_like(argument_1_embed)
        image_embed = self.embeddings(img, (task_1_embed, task_2_embed), (argument_1_embed, argument_2_embed))
        encoder_result = self.layer_norm(self.encoder(image_embed)[0])
        logits_1 = self.classifier_1(encoder_result[:, 0, :])
        if task_2 is not None:
            logits_2 = self.classifier_2(encoder_result[:, 2, :])
            return logits_1, logits_2
        else:
            return logits_1
