from abc import ABC
from collections import OrderedDict
from functools import partial

import math
import torch
from torch import nn
from torchvision.models.vision_transformer import Encoder
from torchvision import models
from typing import Optional, Callable, NamedTuple, Tuple, List

from utils import ModelData


class MultiViewTransformer(ABC, nn.Module):

    def __init__(self):
        super().__init__()


class MultiViewPretrainedResNet(MultiViewTransformer):
    # use resnet101 as embedder

    def __init__(self, embedding_dim, pretrained=True):
        super().__init__()
        self.encoder = models.resnet101(pretrained=pretrained)
        self.encoder.fc = nn.Linear(2048, embedding_dim)

    def forward(self, data: ModelData) -> ModelData:
        top, side = data.surgeries_data[0], data.surgeries_data[1]
        top = self.encoder(top)
        side = self.encoder(side)

        # sum pooling
        x = torch.stack((top, side), dim=1).sum(dim=1)

        return ModelData(x, data.lengths)


class EncoderParameters(NamedTuple):
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)


class GlobalEncoder(Encoder):
    # same as Encoder but without positional embeddings
    def __init__(self, seq_len, **kw):
        super().__init__(seq_len, **kw)
        # remove the positional embedding parameter from the Encoder, so it won't be optimized during training
        self.pos_embedding.requires_grad = False

    def forward(self, input: torch.Tensor):
        assert input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        return self.ln(self.layers(self.dropout(input)))


class LocalGlobalTransformer(MultiViewTransformer):

    def __init__(
            self,
            image_h: int,
            image_w: int,
            patch_size: int,
            local_parameters: EncoderParameters,
            global_parameters: EncoderParameters
    ):
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.patch_size = patch_size

        seq_length = (image_h // patch_size) * (image_w // patch_size) + 1  # +1 for the early_fusion_classifier token

        self.top_encoder = Encoder(
            seq_length,
            **local_parameters._asdict(),
        )
        self.side_encoder = Encoder(
            seq_length,
            **local_parameters._asdict(),
        )
        self.seq_length = 2 * seq_length

        self.global_encoder = GlobalEncoder(
            seq_length,
            **global_parameters._asdict(),
        )

    def forward(self, data: ModelData) -> ModelData:
        top, side = data.surgeries_data[0], data.surgeries_data[1]
        top_len = top.shape[1]
        x = self.global_encoder(torch.cat([self.top_encoder(top), self.side_encoder(side)], dim=1))

        # Sum-pooling over early_fusion_classifier "token" from each view
        x = x[:, [0, top_len]].sum(dim=1)

        return ModelData(
            x,
            data.lengths)


class Crossformer(MultiViewTransformer):
    pass
