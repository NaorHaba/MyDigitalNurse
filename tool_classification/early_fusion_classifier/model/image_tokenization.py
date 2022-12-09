from abc import ABC

import math
import torch
from torch import nn
from torchvision.models.vision_transformer import ConvStemConfig
from torchvision.ops.misc import ConvNormActivation
from typing import Optional, List, Tuple

from utils import ModelData


class ImageTokenizer(ABC, nn.Module):

    def __init__(self):
        super().__init__()


class PatchTokenizer(ImageTokenizer):

    def __init__(
            self,
            image_h: int,
            image_w: int,
            patch_size: int,
            embedding_dim: int,
            conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        assert image_h % patch_size == 0, "Input height indivisible by patch size!"
        assert image_w % patch_size == 0, "Input width indivisible by patch size!"
        self.image_h = image_h
        self.image_w = image_w
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    ConvNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=embedding_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_h // patch_size) * (image_w // patch_size) + 1

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        seq_length += 1
        self.seq_length = seq_length

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == self.image_h, "Wrong image height!"
        assert w == self.image_w, "Wrong image width!"
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.embedding_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        # Reshape and permute the input tensor
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        return x


class PixelTokenizer(ImageTokenizer):
    pass


class TokenTokenizer(ImageTokenizer):
    pass


tokenizers = {
    "patch": PatchTokenizer,
    "pixel": PixelTokenizer,
    "token": TokenTokenizer,
    "identity": nn.Identity,
}


class TopSideTokenizer(nn.Module):

    def __init__(self, tokenizer, **kw):
        super().__init__()
        assert tokenizer in tokenizers, f"tokenizer should be one of {tokenizer.keys()}"
        self.top_tokenizer = tokenizers[tokenizer](**kw)
        self.side_tokenizer = tokenizers[tokenizer](**kw)

    def forward(self, data: ModelData) -> ModelData:
        return ModelData(
            torch.stack((self.top_tokenizer(data.surgeries_data[0]), self.side_tokenizer(data.surgeries_data[1]))),
            data.lengths)
