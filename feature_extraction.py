from abc import ABC, abstractmethod

import torch
from torch import nn as nn
from typing import Tuple

from utils import ModelData


class FeatureExtractor(ABC, nn.Module):

    @abstractmethod
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor

    def forward(self, x):
        return self.extractor(x)


class IdentityExtractor(FeatureExtractor):

    def __init__(self):
        extractor = nn.Identity()
        super().__init__(extractor)


class ResNetExtractor(FeatureExtractor):

    def __init__(self):
        extractor = None  # TODO
        super().__init__(extractor)


extractors = {
    "identity": IdentityExtractor,
    "resnet": ResNetExtractor
}


class TopSideExtractor(nn.Module):

    def __init__(self, extractor, **kw):
        super().__init__()
        assert extractor in extractors, f"tokenizer should be one of {extractors.keys()}"
        self.top_extractor = extractors[extractor](**kw)
        self.side_extractor = extractors[extractor](**kw)

    def forward(self, data: ModelData) -> ModelData:
        return ModelData(
            surgeries_data=torch.stack((self.top_extractor(data.surgeries_data[0]), self.side_extractor(data.surgeries_data[1]))),
            lengths=data.lengths)
