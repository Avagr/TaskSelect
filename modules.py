import math

import torch
from torch import nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTAttention


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


class CrossAttention(nn.Module):
    """
    Partly taken from huggingface: https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
    """

    def __init__(self, enc_config: ViTConfig, dec_config: ViTConfig):
        super().__init__()
        self.num_attention_heads = dec_config.num_attention_heads
        self.attention_head_size = int(dec_config.hidden_size / dec_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(in_features=dec_config.hidden_size, out_features=self.all_head_size)
        self.key = nn.Linear(in_features=enc_config.hidden_size, out_features=self.all_head_size)
        self.value = nn.Linear(in_features=enc_config.hidden_size, out_features=self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, decoder_hidden_states, encoder_hidden_states):
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        query_layer = self.transpose_for_scores(self.query(decoder_hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class TaskDecoderBlock(nn.Module):
    """
        Partly taken from huggingface: https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
        """

    def __init__(self, enc_config: ViTConfig, dec_config: ViTConfig):
        super().__init__()
        self.chunk_size_feed_forward = dec_config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(dec_config)
        self.cross_attention = CrossAttention(enc_config, dec_config)
        self.feedforward = nn.Sequential(nn.Linear(in_features=dec_config.hidden_size,
                                                   out_features=dec_config.intermediate_size),
                                         nn.GELU(),
                                         nn.Linear(in_features=dec_config.intermediate_size,
                                                   out_features=dec_config.hidden_size))
        self.layernorm_before = nn.LayerNorm(dec_config.hidden_size, eps=dec_config.layer_norm_eps)
        self.encoder_layernorm_cross = nn.LayerNorm(enc_config.hidden_size, eps=enc_config.layer_norm_eps)
        self.decoder_layernorm_cross = nn.LayerNorm(dec_config.hidden_size, eps=dec_config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(dec_config.hidden_size, eps=dec_config.layer_norm_eps)

    def forward(self, decoder_hidden_states, encoder_hidden_states):
        self_attention_output = self.attention(self.layernorm_before(decoder_hidden_states))[0]
        decoder_hidden_states = self_attention_output + decoder_hidden_states

        cross_attention_outputs = self.cross_attention(self.decoder_layernorm_cross(decoder_hidden_states),
                                                       self.encoder_layernorm_cross(encoder_hidden_states))
        # First residual connection
        decoder_hidden_states = cross_attention_outputs + decoder_hidden_states

        layer_output = self.layernorm_after(decoder_hidden_states)
        feedforward_output = self.feedforward(layer_output)
        feedforward_output = decoder_hidden_states + feedforward_output

        return feedforward_output


class LeftRightEncDec(nn.Module):

    def __init__(self, num_classes, enc_config: ViTConfig, dec_config: ViTConfig, use_butd):
        super().__init__()
        self.use_butd = use_butd
        self.encoder = ViTModel(enc_config, add_pooling_layer=False)
        self.decoder_blocks = nn.ModuleList(
            [TaskDecoderBlock(enc_config=enc_config, dec_config=dec_config) for _ in
             range(dec_config.num_hidden_layers)])
        self.task_embeddings = nn.Linear(in_features=2, out_features=dec_config.hidden_size)
        self.argument_embeddings = nn.Linear(in_features=num_classes, out_features=dec_config.hidden_size)
        self.layernorm = nn.LayerNorm(enc_config.hidden_size, eps=enc_config.layer_norm_eps)
        self.task_position_embeddings = nn.Parameter(torch.zeros(1, 2, dec_config.hidden_size))
        self.classifier = nn.Linear(in_features=dec_config.hidden_size, out_features=num_classes)

    def forward(self, img: torch.Tensor, task: torch.FloatTensor, argument: torch.FloatTensor) -> torch.Tensor:
        encoder_hidden_states = self.encoder(img, output_hidden_states=True).hidden_states

        task_tokens = self.task_embeddings(task).unsqueeze(1)
        argument_tokens = self.argument_embeddings(argument).unsqueeze(1)

        decoder_state = torch.cat((task_tokens, argument_tokens), dim=1) + self.task_position_embeddings
        if self.use_butd:
            for encoder_hidden_state, block in zip(encoder_hidden_states[::-1], self.decoder_blocks):
                decoder_state = block(decoder_state, self.layernorm(encoder_hidden_state))
        else:
            for block in self.decoder_blocks:
                decoder_state = block(decoder_state, self.layernorm(encoder_hidden_states[-1]))
        return self.classifier(decoder_state[:, 0, :])


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
