from typing import Optional

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from feature_extraction import TopSideExtractor

import torch
from torch import nn as nn

from image_tokenization import TopSideTokenizer
from image_transformers import MultiViewTransformer, LocalGlobalTransformer, EncoderParameters
from time_series_analysis import TimeSeries, MSTCN, TimeSeriesGPT2Model
from utils import ModelData


class VisionModel(nn.Module):

    def __init__(self,
                 feature_extractor: str,
                 tokenizer: str,
                 image_h: int = 1080,
                 image_w: int = 1920,
                 patch_size: int = 120,
                 embedding_dim: int = 256,
                 local_parameters: Optional[EncoderParameters] = None,
                 global_parameters: Optional[EncoderParameters] = None,
                 time_series_layers: int = 1):

        super().__init__()

        self.feature_extractor = TopSideExtractor(feature_extractor)
        self.tokenizer = TopSideTokenizer(tokenizer, image_h=image_h, image_w=image_w, patch_size=patch_size, embedding_dim=embedding_dim)
        if local_parameters is None:
            # default parameters
            local_parameters = EncoderParameters(2, 4, 256, 16)
        if global_parameters is None:
            # default parameters
            global_parameters = EncoderParameters(2, 4, 256, 16)

        self.transformer = LocalGlobalTransformer(image_h, image_w, patch_size, local_parameters, global_parameters)

        self.time_series = TimeSeriesGPT2Model(n_embed=embedding_dim, n_layer=time_series_layers)

        self.predictor = {
            'surgeon_R': nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1)),
            'surgeon_L': nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1)),
            'assistant_R': nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1)),
            'assistant_L': nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1)),
        }

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.tokenizer(x)
        x = self.transformer(x)
        x = self.time_series(x)
        x = {k: self.predictor[k](x) for k in self.predictor}
        return x
