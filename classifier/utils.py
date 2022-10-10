import torch
from typing import NamedTuple, Tuple


class ModelData(NamedTuple):
    surgeries_data: torch.Tensor
    lengths: Tuple[int]


SURGERY_GROUPS = {
    'train': [
        'P219_D_N.Camera1',
        'P171_E_N.Camera1',
        'P191_J_Y.Camera1',
        'P183_D_N.Camera1',
        'P001_G_N.Camera1',
        'P045_E_Y.Camera1',
        'P203_I_N.Camera1',
        'P135_F_N.Camera1',
        'P048_E_Y.Camera1',
        'P051_J_N.Camera1',
        'P206_C_N.Camera1',
        'P116_C_N.Camera1',
        'P047_J_Y.Camera1',
        'P008_C_N.Camera1',
        'P224_F_N.Camera1',
        'P004_H_Y.Camera1',
        'P134_H_N.Camera1',
        'P120_E_Y.Camera1',
    ],  # 19
    'val': [
        'P228_D_N.Camera1',
        'P017_A_Y.Camera1',
        'P003_C_Y.Camera1',
        'P032_J_N.Camera1',
        'P140_B_N.Camera1',
    ],  # 5
    'test': [
        'P209_D_Y.Camera1_new',
        'P201_G_N.Camera1',
        'P030_A_N.Camera1',
        'P033_C_Y.Camera1',
        'P133_A_Y.Camera1',
        'P130_D_N.Camera1',
    ]  # 6
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