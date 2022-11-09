import torch
import torch.nn as nn
import copy
import pandas as pd
from typing import List
import numpy as np
import os
from torch.utils.data import Dataset
from link_utils import process_boris_df
import pickle
import re

ACTIVATIONS = {'relu': torch.nn.ReLU, 'lrelu': torch.nn.LeakyReLU, 'tanh': torch.nn.Tanh, 'sigmoid':torch.nn.Sigmoid}
BORIS = "/strg/C/shared-data/ACS_2019_data/boris/boris_big_gt"


SURGERY_GROUPS_BORIS = {
    'train': ['P193_D_N.Camera1.csv', 'P122_H_N.Camera1.csv', 'P117_A_N.Camera1.csv', 'P152_F_N.Camera1.csv',
              'P137_J_Y.Camera1.csv', 'P011_E_Y.Camera1.csv', 'P027_I_Y.Camera1.csv', 'P024_E_Y.Camera1.csv',
              'P153_E_N.Camera1.csv', 'P163_B_Y.Camera1.csv', 'P102_J_N.Camera1.csv', 'P041_E_Y.Camera1.csv',
              'P167_I_N.Camera1.csv', 'P129_E_N.Camera1.csv', 'P191_J_Y.Camera1.csv', 'P016_D_N.Camera1.csv',
              'P114_H_Y.Camera1.csv', 'P095_A_Y.Camera1.csv', 'P131_B_N.Camera1.csv', 'P172_A_N.Camera1.csv'],
    'val':['P126_B_Y.Camera1.csv', 'P100_D_Y.Camera1.csv', 'P030_A_N.Camera1.csv', 'P058_H_N.Camera1.csv',
           'P014_G_N.Camera1.csv', 'P038_H_N.Camera1.csv'],
    'test': ['P019_E_Y.Camera1.csv', 'P107_J_Y.Camera1.csv', 'P009_A_Y.Camera1.csv', 'P112_B_N.Camera1.csv', 'P015_I_N.Camera1.csv']

}
HANDS = {'both':[ 'R_surgeon','L_surgeon', 'R_assistant', 'L_assistant'],
        'surgeon': [ 'R_surgeon','L_surgeon'],
        'assitant': ['R_assistant', 'L_assistant']}



class RNN_Model(nn.Module):
    """
    RNN model base class
    Supports GRU or LSTM layers
    """
    def __init__(self, rnn_type, input_dim, hidden_dim=64, bidirectional=False, dropout=0.4, num_layers=2, num_classes=25):
        super(RNN_Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                               num_layers=num_layers, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                              num_layers=num_layers, dropout=dropout)
        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, rnn_inputs, lengths, mask):
        rnn_inputs = rnn_inputs.float()

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs, lengths=lengths, batch_first=True,
                                                               enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_input)
        unpacked_rnn_out, unpacked_rnn_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output,
                                                                                            padding_value=-1,
                                                                                            batch_first=True)
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        last_out = torch.stack([unpacked_rnn_out[i, unpacked_rnn_out_lengths[i] - 1, :] for i in range(len(lengths))])
        return self.output(last_out)
        # return self.output(unpacked_rnn_out)


#
class Dataset(Dataset):
    """
    Dataset for surgeries.
    """
    def __init__(self, group: str, graph_worker:str):
        self.surgeries_list = SURGERY_GROUPS_BORIS[group]
        self.hands = HANDS[graph_worker]
        f = open(f"state_to_node_{len(self.hands)}.pkl", "rb")
        self.state_to_node = pickle.load(f)
        self.graph_worker = graph_worker
        self.data_dict()


    def __getitem__(self, item):
        surgery_data = self.data[item]['features']
        label = self.data[item]['label']
        return surgery_data, label

    def __len__(self):
        return len(self.data)

    def data_dict(self):
        self.data = []
        for i,sur in enumerate(self.surgeries_list):
            states, times, stamps = self.create_labels_from_surgery(sur)
            labels = torch.tensor(states[1:])
            # features = [torch.tensor([0.] * state + [1.] + [0.] * (24 - state) + [times[i]] + [stamps[i]]) for i, state in
            #         enumerate(states[:-1])] #1hot+time diff + timestamp
            features = torch.stack([torch.tensor([0.] * state + [1.] + [0.] * (24 - state)) for state in states[:-1]]) #1hot
            for j, label in enumerate(labels):
                self.data+=[{'features':features[:j+1,:], 'label':label}]


    def create_labels_from_surgery(self, surgery):
        df = pd.read_csv(os.path.join(BORIS, surgery), skiprows=15)
        processed_df = process_boris_df(df, self.graph_worker)
        # processed_df['Shift_Time'] = processed_df['Time_Diff'].shift(-1).fillna(1 / processed_df.FPS)
        states = []
        time_diffs = []
        time = []
        for i, row in processed_df.iterrows():
            state = tuple(int(row[x].item()) for x in self.hands)
            states += [self.state_to_node[state]]
            time_diffs += [row.Time_Diff]
            time += [row.Time]
            # states += [state_to_node[state]] * 1 if not withselfloops else [state_to_node[state]] * int(row.FPS*row.Shift_Time)
        return torch.tensor(states), torch.tensor(time_diffs), torch.tensor(time_diffs)


def collate_inputs(batch):
    """
    Collate function to batch several surgeries which may have different lengths.
    :param batch: items from the dataset
    :return: batch which includes: model tensor inputs, labels, lengths, masks and surgery ids for predicting
    """
    input_lengths = []
    batch_features = []
    batch_labels = []
    batch_ids = []
    for sample in batch:
        sample_features = sample[0]
        input_lengths.append(sample_features.shape[0])
        batch_features.append(sample_features)
        batch_labels.append(sample[1])
        # pad
    batch = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
    # labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
    labels = torch.tensor(batch_labels)
    # if len(batch)==1:
    #     labels.unsqueeze(0)
    # compute mask
    input_masks = batch != 0
    return batch.double(), labels.type(torch.LongTensor), torch.tensor(input_lengths), input_masks
