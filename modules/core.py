import math

import torch
from torch import nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTAttention


class TaskEmbeddings(nn.Module):
    def __init__(self, patch_embeddings: nn.Module, hidden_size: int, num_tasks=1):
        super().__init__()
        self.patch_embeddings = patch_embeddings
        num_patches = self.patch_embeddings.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + num_patches + 2 * num_tasks, hidden_size))

    def forward(self, pixel_values, task_tokens: list[torch.Tensor], argument_tokens: list[torch.Tensor]):
        embeddings = self.patch_embeddings(pixel_values)
        tasks = list(zip(task_tokens, argument_tokens))
        cls_tokens = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, *[item for pair in tasks for item in pair], embeddings), dim=1)
        return embeddings + self.position_embeddings


class CrossAttention(nn.Module):
    """
    Partly taken from huggingface:
    https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
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
        Partly taken from huggingface:
        https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
        """

    def __init__(self, enc_config: ViTConfig, dec_config: ViTConfig):
        super().__init__()
        self.chunk_size_feed_forward = dec_config.chunk_size_feed_forward

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


class LateralAttention(nn.Module):
    """
    Partly taken from huggingface:
    https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.key = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.value = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, lateral_states, mix_with):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if mix_with == 'q':
            query_layer = query_layer * lateral_states
        elif mix_with == 'k':
            key_layer = key_layer * lateral_states
        elif mix_with == 'v':
            value_layer = value_layer * lateral_states
        else:
            raise ValueError("Only 'q', 'k', 'v' values are permitted for lateral connections")

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


class SelfAttention(nn.Module):
    """
    Partly taken from huggingface:
    https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.key = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.value = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, return_state):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if return_state == 'q':
            return_layer = query_layer
        elif return_state == 'k':
            return_layer = key_layer
        elif return_state == 'v':
            return_layer = value_layer
        else:
            raise ValueError("Only 'q', 'k', 'v' values are permitted for lateral connections")

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, return_layer


class LateralEncoderBlock(nn.Module):
    """
        Partly taken from huggingface:
        https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
        """

    def __init__(self, config: ViTConfig, mix_with: str, use_self_attention: bool):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward

        self.use_self_attention = use_self_attention
        self.mix_with = mix_with
        if use_self_attention:
            self.attention = ViTAttention(config)
        self.lateral_attention = LateralAttention(config)
        self.feedforward = nn.Sequential(nn.Linear(in_features=config.hidden_size,
                                                   out_features=config.intermediate_size),
                                         nn.GELU(),
                                         nn.Linear(in_features=config.intermediate_size,
                                                   out_features=config.hidden_size))
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if use_self_attention:
            self.encoder_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lateral_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, lateral_states):
        hidden_states = self.layernorm_before(hidden_states)
        if self.use_self_attention:
            self_attention_output = self.attention(hidden_states)[0]
            hidden_states = self.encoder_layernorm(self_attention_output + hidden_states)

        lateral_attention_outputs = self.lateral_attention(hidden_states,
                                                           lateral_states,
                                                           self.mix_with)

        hidden_states = lateral_attention_outputs + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        feedforward_output = self.feedforward(layer_output)
        feedforward_output = hidden_states + feedforward_output

        return feedforward_output


class TransparentEncoderBlock(nn.Module):
    """
        Partly taken from huggingface:
        https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/vit/modeling_vit.html
        """

    def __init__(self, config: ViTConfig, return_state: str):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.attention = SelfAttention(config)
        self.return_state = return_state
        self.feedforward = nn.Sequential(nn.Linear(in_features=config.hidden_size,
                                                   out_features=config.intermediate_size),
                                         nn.GELU(),
                                         nn.Linear(in_features=config.intermediate_size,
                                                   out_features=config.hidden_size))
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.layernorm_before(hidden_states)

        lateral_attention_outputs, attention_state = self.attention(hidden_states, self.return_state)

        hidden_states = lateral_attention_outputs + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        feedforward_output = self.feedforward(layer_output)
        feedforward_output = hidden_states + feedforward_output

        return feedforward_output, attention_state


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)
