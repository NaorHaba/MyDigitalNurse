from abc import ABC, abstractmethod

import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import LongformerModel, LongformerConfig, GPT2Config, GPT2Model
import torch.nn.functional as F
from typing import Tuple

from utils import ModelData


class TimeSeries(ABC, nn.Module):
    pass

    @abstractmethod
    def forward(self, x):
        pass


class MSTCN(TimeSeries):

    def __init__(self):
        super().__init__()


class MultiResolutionLSTM(TimeSeries):
    pass


class TemporalTransformer(TimeSeries):
    pass


class TimeSeriesGPT2Model(GPT2Model):

    # TODO - check if more parameters (from GPT2 config) are needed for hyperparameter tuning
    def __init__(self,
                 n_embed=32,
                 n_positions=2 * 60 * 60,  # TODO define in terms of "time" from frame count
                 n_head=2,
                 n_layer=2,
                 pad_token_id=-1,
                 # attention_window=None,
                 n_inner=32,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1):
        self.config = GPT2Config()
        self.config.hidden_size = n_embed
        self.config.n_positions = n_positions
        self.config.n_head = n_head
        self.config.n_layer = n_layer
        self.config.n_inner = n_inner
        self.config.attn_pdrop = attn_pdrop
        self.config.resid_pdrop = resid_pdrop
        # self.config.attention_dilation = [1, ] * num_hidden_layers
        # self.config.attention_window = [256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.pad_token_id = pad_token_id
        super(TimeSeriesGPT2Model, self).__init__(self.config)
        # self.embeddings.word_embeddings = None  # to avoid distributed error of unused parameters

    def forward(self, data: ModelData) -> torch.Tensor:
        # TODO check use_cache option
        x = pad_sequence(torch.split(data.surgeries_data, data.lengths, dim=0), batch_first=True)
        attention_mask = pad_sequence([torch.ones(l) for l in data.lengths], batch_first=True)
        x = super(TimeSeriesGPT2Model, self).forward(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        return x
        # TODO maybe add dropout for the output of the model


# TODO credit for VTN
class VTNLongformerModel(TemporalTransformer, LongformerModel):

    def __init__(self,
                 embed_dim=768,
                 max_position_embeddings=2 * 60 * 60,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 attention_mode='sliding_chunks',
                 pad_token_id=-1,
                 attention_window=None,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):

        self.config = LongformerConfig()
        self.config.attention_mode = attention_mode
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_dilation = [1, ] * num_hidden_layers
        self.config.attention_window = [256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.hidden_size = embed_dim
        super(VTNLongformerModel, self).__init__(self.config, add_pooling_layer=False)
        self.embeddings.word_embeddings = None  # to avoid distributed error of unused parameters


def pad_to_window_size_local(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                             one_sided_window_size: int, pad_token_id: int):
    """A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer self-attention.
    Based on _pad_to_window_size from https://github.com/huggingface/transformers:
    https://github.com/huggingface/transformers/blob/71bdc076dd4ba2f3264283d4bc8617755206dccd/src/transformers/models/longformer/modeling_longformer.py#L1516
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    """
    w = 2 * one_sided_window_size
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids.permute(0, 2, 1), (0, padding_len), value=pad_token_id).permute(0, 2, 1)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    position_ids = F.pad(position_ids, (1, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask, position_ids
