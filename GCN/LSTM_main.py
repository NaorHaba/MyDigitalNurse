import torch
from torch import nn
import numpy as np
from trainer import Trainer
import os
import argparse
from datetime import datetime
import random
import logging
import itertools
import pandas as pd
from torch.utils.data import DataLoader
from LSTM_Model import RNN_Model, Dataset, collate_inputs

#%%


logger = logging.getLogger(__name__)
Imputations = {}
TRAIN_PATH = 'DataFiles/filtered_train_df_0705_LSTM_new.csv'
VAL_PATH = 'DataFiles/filtered_val_df_0705_LSTM_new.csv'
TEST_PATH = 'DataFiles/filtered_test_df_0705_LSTM_new.csv'



def parsing():
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_name', default="LSTMS")

    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="RNN_BORIS", type=str)
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='surgeon')

    # parser.add_argument('--entity', default="surgical_data_science", type=str)

    parser.add_argument('--time_series_model', choices=['LSTM','GRU'], default='LSTM', type=str)

    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--hidden_dim', default=528, type=int)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--input_dim', default=25, type=int)
    parser.add_argument('--early_stop', default=7, type=int)

    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)

    # parser.add_argument('--imputation', choices=['iterative','mean','median'], default='iterative', type=str)
    # parser.add_argument('--window', default=5, type=int)
    # parser.add_argument('--seq_len', default=10, type=int)
    # parser.add_argument('--sample', choices=['over','under','overunder'], default='overunder', type=str)
    # parser.add_argument('--over_sample_rate', default='0.3449', type=float)
    # parser.add_argument('--under_sample_rate', default='0.5034', type=float)

    args = parser.parse_args()
    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



args = parsing()
set_seed()
logger.info(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_train = Dataset('train', args.graph_worker)
dl_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=True)
ds_val = Dataset('val',args.graph_worker)
dl_val = DataLoader(ds_val, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=False)
model = RNN_Model(rnn_type=args.time_series_model,bidirectional=args.bidirectional,
                  input_dim = args.input_dim,hidden_dim = args.hidden_dim,dropout= args.dropout,num_layers =args.num_layers )
trainer = Trainer(model,device=device, early_stop = args.early_stop)
# eval_results, train_results = \
trainer.train(dl_train,dl_val,args.num_epochs, args.lr,args)