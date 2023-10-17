from dataclasses import dataclass

import torch
from torch import nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTEmbeddings

from modules.core import TaskEmbeddings, TaskDecoderBlock, LateralEncoderBlock, TransparentEncoderBlock
from modules.embeddings import BaseEmbedding


@dataclass
class BUTDResult:
    main_logits: torch.Tensor
    aux_logits: torch.Tensor = None
    task_tokens: torch.Tensor = None
    arg_tokens: torch.Tensor = None
    attentions: tuple[torch.Tensor] = None
    distribution_vectors: tuple[{str: torch.Tensor}] = None
    intermediate_neurons: tuple[torch.Tensor] = None


class EncDecBUTD(nn.Module):

    def __init__(self, num_classes: int, enc_config: ViTConfig, task_embedding: BaseEmbedding,
                 argument_embedding: BaseEmbedding, dec_config: ViTConfig, use_butd: bool, use_sinusoidal=False,
                 aux_head_size=None):
        super().__init__()
        self.use_butd = use_butd

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

        self.task_embeddings = task_embedding
        self.argument_embeddings = argument_embedding
        self.task_position_embeddings = nn.Parameter(torch.zeros(1, 2, dec_config.hidden_size))

        self.final_layernorm = nn.LayerNorm(dec_config.hidden_size, eps=dec_config.layer_norm_eps)

    def forward(self, img: torch.Tensor, task: torch.FloatTensor, argument: torch.FloatTensor,
                output_task_tokens=False) -> BUTDResult:
        encoder_hidden_states = self.encoder(self.embeddings(img, [], []),
                                             output_hidden_states=True).hidden_states

        task_tokens, task_embed_data = self.task_embeddings(task)
        task_tokens = task_tokens.unsqueeze(1)
        argument_tokens, argument_embed_data = self.argument_embeddings(argument)
        argument_tokens = argument_tokens.unsqueeze(1)

        decoder_state = torch.cat((task_tokens, argument_tokens), dim=1) + self.task_position_embeddings
        if self.use_butd:
            for encoder_hidden_state, block in zip(encoder_hidden_states[::-1], self.decoder_blocks):
                decoder_state = block(decoder_state, encoder_hidden_state)
        else:
            for block in self.decoder_blocks:
                decoder_state = block(decoder_state, encoder_hidden_states[-1])

        output = BUTDResult(self.classifier(self.final_layernorm(decoder_state[:, 0, :])))

        if self.bu_classifier is not None:
            output.aux_logits = self.bu_classifier(self.final_encoder_layernorm(encoder_hidden_states[-1][:, 0, :]))

        if output_task_tokens:
            output.task_tokens = decoder_state[:, 1:, :]

        output.distribution_vectors = (task_embed_data, argument_embed_data)
        return output


class MixingBUTD(nn.Module):

    def __init__(self, num_classes: int, config: ViTConfig, task_embedding: BaseEmbedding,
                 argument_embedding: BaseEmbedding, total_token_size: int, use_self_attention: bool, mix_with: str,
                 use_sinusoidal=False, aux_head_size=None, stretch=True):
        super().__init__()
        self.stretch = stretch
        self.task_encoder_blocks = nn.ModuleList(
            [TransparentEncoderBlock(config, mix_with) for _ in range(config.num_hidden_layers)])
        self.image_embeddings = TaskEmbeddings(ViTEmbeddings(config).patch_embeddings, config.hidden_size, num_tasks=0,
                                               use_sinusoidal=use_sinusoidal,
                                               num_cls_tokens=1 if aux_head_size is None else 2)
        self.image_encoder_blocks = nn.ModuleList(
            [LateralEncoderBlock(config, mix_with, use_self_attention) for _ in range(config.num_hidden_layers)])

        self.task_embeddings = task_embedding
        self.argument_embeddings = argument_embedding
        self.relu = nn.ReLU()
        if stretch:
            self.stretch_embedding = nn.Linear(in_features=2 * config.hidden_size, out_features=total_token_size)
        else:
            self.dummy_tokens = nn.Parameter(torch.randn(1, total_token_size // config.hidden_size - 2,
                                                         config.hidden_size))
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.task_position_embeddings = nn.Parameter(
            torch.zeros(1, total_token_size // config.hidden_size, config.hidden_size))
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=num_classes)

        self.aux_classifier = None
        if aux_head_size is not None:
            self.aux_classifier = nn.Linear(in_features=config.hidden_size, out_features=aux_head_size)

    def forward(self, img: torch.Tensor, task: torch.FloatTensor, argument: torch.FloatTensor,
                return_mixing=False, output_task_tokens=False) -> BUTDResult:

        embedded_image = self.image_embeddings(img, [], [])

        task_tokens, task_embed_data = self.task_embeddings(task)
        argument_tokens, argument_embed_data = self.argument_embeddings(argument)

        if self.stretch:
            task_input = self.stretch_embedding(self.relu(torch.cat((task_tokens, argument_tokens), dim=1))).view(
                *embedded_image.shape)
        else:
            task_input = torch.cat((task_tokens.unsqueeze(1), argument_tokens.unsqueeze(1),
                                    self.dummy_tokens.expand(task_tokens.shape[0], -1, -1)), dim=1)

        task_input = task_input + self.task_position_embeddings

        mixing_layers = []

        for block in self.task_encoder_blocks:
            task_input, mixing_state = block(task_input)
            mixing_layers.append(mixing_state)

        for mixing_state, block in zip(mixing_layers[::-1], self.image_encoder_blocks):
            embedded_image = block(embedded_image, mixing_state)

        normed_res = self.layernorm(embedded_image)

        output = BUTDResult(self.classifier(normed_res[:, 0, :]))
        if self.aux_classifier is not None:
            output.aux_logits = self.aux_classifier(normed_res[:, 1, :])
        if return_mixing:
            output.intermediate_neurons = mixing_layers
        if output_task_tokens:
            output.task_tokens = task_input[:, 0, :]
        output.distribution_vectors = (task_embed_data, argument_embed_data)
        return output


class EncoderBUTD(nn.Module):
    def __init__(self, num_classes: int, enc_config: ViTConfig, task_embedding: BaseEmbedding,
                 argument_embedding: BaseEmbedding, use_sinusoidal=False, aux_head_size=None):
        super().__init__()
        encoder = ViTForImageClassification(enc_config)
        self.embeddings = TaskEmbeddings(encoder.vit.get_input_embeddings(), enc_config.hidden_size,
                                         use_sinusoidal=use_sinusoidal,
                                         num_cls_tokens=1 if aux_head_size is None else 2)
        self.layer_norm = encoder.vit.layernorm
        self.encoder = encoder.vit.encoder
        self.classifier = nn.Linear(in_features=enc_config.hidden_size, out_features=num_classes)
        self.task_embeddings = task_embedding
        self.argument_embeddings = argument_embedding

        self.aux_classifier = None
        if aux_head_size is not None:
            self.aux_classifier = nn.Linear(in_features=enc_config.hidden_size, out_features=aux_head_size)

    def forward(self, img: torch.FloatTensor, task: torch.FloatTensor, argument: torch.FloatTensor,
                output_attentions=False, output_task_tokens=False) -> BUTDResult:
        task_tokens, task_embed_data = self.task_embeddings(task)
        task_tokens = task_tokens.unsqueeze(1)
        argument_tokens, argument_embed_data = self.argument_embeddings(argument)
        argument_tokens = argument_tokens.unsqueeze(1)

        image_embed = self.embeddings(img, (task_tokens,), (argument_tokens,))
        encoder_result = self.encoder(image_embed, output_attentions=output_attentions)
        last_state = encoder_result.last_hidden_state
        last_state = self.layer_norm(last_state)

        output = BUTDResult(self.classifier(last_state[:, 0, :]))
        if self.aux_classifier is not None:
            output.aux_logits = self.aux_classifier(last_state[:, 1, :])
        if encoder_result.attentions is not None:
            output.attentions = encoder_result.attentions
        if output_task_tokens:
            output.task_tokens = last_state[:, 2, :]
        output.distribution_vectors = (task_embed_data, argument_embed_data)
        return output


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
