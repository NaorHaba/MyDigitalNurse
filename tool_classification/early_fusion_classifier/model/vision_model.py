from typing import Optional

from feature_extraction import TopSideExtractor

from torch import nn as nn

from image_tokenization import TopSideTokenizer
from image_transformers import LocalGlobalTransformer, EncoderParameters
from time_series_analysis import TimeSeriesGPT2Model


class VisionModel(nn.Module):

    def __init__(self,
                 feature_extractor: str,
                 tokenizer: str,
                 image_h: int = 240,
                 image_w: int = 320,
                 patch_size: int = 20,
                 embedding_dim: int = 1024,
                 local_parameters: Optional[EncoderParameters] = None,
                 global_parameters: Optional[EncoderParameters] = None,
                 time_series_layers: int = 1):

        super().__init__()

        self.feature_extractor = TopSideExtractor(feature_extractor)
        self.tokenizer = TopSideTokenizer(tokenizer, image_h=image_h, image_w=image_w, patch_size=patch_size, embedding_dim=embedding_dim)
        if local_parameters is None:
            # default parameters
            local_parameters = EncoderParameters(3, 8, 1024, 256)
        if global_parameters is None:
            # default parameters
            global_parameters = EncoderParameters(3, 8, 1024, 256)

        self.transformer = LocalGlobalTransformer(image_h, image_w, patch_size, local_parameters, global_parameters)

        self.time_series = TimeSeriesGPT2Model(n_embed=embedding_dim, n_layer=time_series_layers)

        self.predictor_surgeon_R = nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1))
        self.predictor_surgeon_L = nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1))
        self.predictor_assistant_R = nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1))
        self.predictor_assistant_L = nn.Sequential(nn.Linear(embedding_dim, 5), nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.tokenizer(x)
        x = self.transformer(x)
        x = self.time_series(x)
        x1 = self.predictor_surgeon_R(x)
        x2 = self.predictor_surgeon_L(x)
        x3 = self.predictor_assistant_R(x)
        x4 = self.predictor_assistant_L(x)
        return {'surgeon_R': x1, 'surgeon_L': x2, 'assistant_R': x3, 'assistant_L': x4}
