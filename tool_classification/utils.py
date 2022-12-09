import torch
from typing import NamedTuple, Tuple

from torch import nn


class ModelData(NamedTuple):
    surgeries_data: torch.Tensor
    lengths: Tuple[int]


SURGERY_GROUPS = {
    'train':
        ['P193_D_N',
         'P122_H_N',
         'P117_A_N',
         'P152_F_N',
         'P137_J_Y',
         'P011_E_Y',
         'P027_I_Y',
         'P024_E_Y',
         'P153_E_N',
         'P163_B_Y',
         'P102_J_N',
         'P041_E_Y',
         'P167_I_N',
         'P129_E_N',
         'P191_J_Y',
         'P016_D_N',
         'P114_H_Y',
         'P095_A_Y',
         'P131_B_N',
         'P172_A_N'],
    'val':
        ['P126_B_Y',
         'P100_D_Y',
         'P030_A_N',
         'P058_H_N',
         'P014_G_N',
         'P038_H_N'],
    'test':
        ['P019_E_Y',
         'P107_J_Y',
         'P009_A_Y',
         'P112_B_N',
         'P015_I_N']
}

LABEL_ID_TO_NAME = {
    0: 'R',
    1: 'L',
    2: 'R2',
    3: 'L2',
    4: 'R+1',
    5: 'L+1',
    6: 'R2+1',
    7: 'L2+1',
    8: 'R+2',
    9: 'L+2',
    10: 'R2+2',
    11: 'L2+2',
    12: 'R+3',
    13: 'L+3',
    14: 'R2+3',
    15: 'L2+3',
    16: 'R+4',
    17: 'L+4',
    18: 'R2+4',
    19: 'L2+4'
}

LABEL_INFO_TO_ID = {
    'surgeon_R_0': 0,
    'surgeon_L_0': 1,
    'assistant_R_0': 2,
    'assistant_L_0': 3,
    'surgeon_R_1': 4,
    'surgeon_L_1': 5,
    'assistant_R_1': 6,
    'assistant_L_1': 7,
    'surgeon_R_2': 8,
    'surgeon_L_2': 9,
    'assistant_R_2': 10,
    'assistant_L_2': 11,
    'surgeon_R_3': 12,
    'surgeon_L_3': 13,
    'assistant_R_3': 14,
    'assistant_L_3': 15,
    'surgeon_R_4': 16,
    'surgeon_L_4': 17,
    'assistant_R_4': 18,
    'assistant_L_4': 19
}

ACTIVATIONS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'lrelu': nn.LeakyReLU, 'gelu': nn.GELU,
               'celu': nn.CELU, 'selu': nn.SELU, 'silu': nn.SiLU}

OPTIMIZERS = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'rmsprop': torch.optim.RMSprop}
