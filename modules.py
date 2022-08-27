import torch
from torch import nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification


class TaskEmbeddings(nn.Module):
    def __init__(self, patch_embeddings: nn.Module, hidden_size: int):
        super().__init__()
        self.patch_embeddings = patch_embeddings
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, hidden_size))

    def forward(self, pixel_values, task_tokens, argument_tokens):
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = torch.cat((task_tokens, argument_tokens, embeddings), dim=1)
        return embeddings + self.position_embeddings


class TaskAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # self.query = nn.Linear()


class LeftRightEncDec(nn.Module):

    def __init__(self, num_classes, enc_hidden_layers, enc_hidden_size, enc_intermediate_size, dec_token_size,
                 dec_layers):
        super().__init__()
        enc_config = ViTConfig(num_hidden_layers=enc_hidden_layers,
                               hidden_size=enc_hidden_size,
                               intermediate_size=enc_intermediate_size)
        vit_model = ViTModel(enc_config)
        self.encoder = vit_model.vit
        self.task_embed = nn.Linear(in_features=2, out_features=dec_token_size)
        self.argument_embed = nn.Linear(in_features=num_classes, out_features=dec_token_size)

    def forward(self, img: torch.Tensor, task: bool, argument: int) -> torch.Tensor:
        return self.encoder(img)


class LeftRightEncoder(nn.Module):
    def __init__(self, num_classes, enc_config: ViTConfig):
        super().__init__()
        encoder = ViTForImageClassification(enc_config)
        self.embeddings = TaskEmbeddings(encoder.vit.get_input_embeddings(), enc_config.hidden_size)
        self.layer_norm = encoder.vit.layernorm
        self.encoder = encoder.vit.encoder
        self.classifier = nn.Linear(in_features=enc_config.hidden_size, out_features=num_classes)
        self.task_embeddings = nn.Linear(in_features=2, out_features=enc_config.hidden_size)
        self.argument_embeddings = nn.Linear(in_features=num_classes, out_features=enc_config.hidden_size)

    def forward(self, img: torch.FloatTensor, task: torch.FloatTensor, argument: torch.FloatTensor):
        task_embed = self.task_embeddings(task).unsqueeze(1)
        argument_embed = self.argument_embeddings(argument).unsqueeze(1)
        image_embed = self.embeddings(img, task_embed, argument_embed)
        encoder_result = self.encoder(image_embed)[0]
        encoder_result = self.layer_norm(encoder_result)
        logits = self.classifier(encoder_result[:, 0, :])
        return logits


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)
