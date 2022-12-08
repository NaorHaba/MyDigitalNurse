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


ARGS_DICT = {
    'truesweep': {'time_series_model':'GRU','add_features':'None','num_layers':2,'hidden_dim':128,
             'dropout':0.05378219434578246, 'lr':0.007167994247203912,'batch_size':8},
    'playful': {'time_series_model': 'GRU', 'add_features': 'timestamp', 'num_layers': 2, 'hidden_dim': 128,
             'dropout': 0.06004744594290896, 'lr': 0.003592809947211788, 'batch_size':5},
    'dauntless': {'time_series_model': 'LSTM', 'add_features': 'positional', 'num_layers': 3, 'hidden_dim': 128,
             'dropout': 0.040054728598359395, 'lr': 0.017945374777689625,'batch_size':5},
    'exalted': {'time_series_model': 'LSTM', 'add_features': 'timestamp', 'num_layers': 3, 'hidden_dim': 128,
                  'dropout': 0.19902126894542096, 'lr': 0.007940198975443444, 'batch_size': 1},
    'trim': {'time_series_model': 'GRU', 'add_features': 'timestamp', 'num_layers': 3, 'hidden_dim': 128,
                  'dropout': 0.13817077508581418, 'lr': 0.003321543121275328, 'batch_size': 5},
}

def edit_args(args):
    args_dict = ARGS_DICT[args.model_name]
    args.time_series_model = args_dict['time_series_model']
    args.add_features = args_dict['add_features']
    args.num_layers = args_dict['num_layers']
    args.dropout = args_dict['dropout']
    args.lr = args_dict['lr']
    args.batch_size = args_dict['batch_size']
    return args

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="RNN_BORIS", type=str)
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='surgeon')
    parser.add_argument('--model_name', default='truesweep', type=str)
    parser.add_argument('--starting_point', default=2,type=int)
    # parser.add_argument('--entity', default="surgical_data_science", type=str)

    parser.add_argument('--time_series_model', choices=['LSTM','GRU'], default='GRU', type=str)
    # parser.add_argument('--add_features', default='positional_timestamp', type=str)
    parser.add_argument('--add_features', default='None', type=str)
    parser.add_argument('--features', default='2hands',choices=['1hot','label','2hands'], type=str)

    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--dropout', default=0.05378219434578246, type=float)
    parser.add_argument('--input_dim', default=25, type=int)
    parser.add_argument('--early_stop', default=7, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)

    args = parser.parse_args()
    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



args = parsing()
args = edit_args(args)
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_add_features = len(args.add_features.split('_')) if args.add_features!='None' else 0
args.input_dim =num_add_features + (25 if args.features=='1hot' else 1 if args.features=='label' else 2)
logger.info(args)
num_classes_list = [5,5] if args.features=='2hands' else [25]
ds_train = Dataset('train', args.graph_worker,args.features,args.add_features)
dl_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=True)
# ds_val = Dataset('val',args.graph_worker,args.features, args.add_features)
# dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
ds_test = Dataset('test',args.graph_worker,args.features,args.add_features)
dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
model = RNN_Model(rnn_type=args.time_series_model,bidirectional=args.bidirectional,device=device,
                  input_dim = args.input_dim,hidden_dim = args.hidden_dim,dropout= args.dropout,
                  num_layers =args.num_layers, num_classes_list=num_classes_list)
print(model)
trainer = Trainer(model,device=device, early_stop = args.early_stop)
# eval_results, train_results = \
trainer.train(dl_train,dl_test,args.num_epochs, args.lr,args)