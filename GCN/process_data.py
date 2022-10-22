import pandas as pd
import os
import sys
import argparse
from link_utils import build_Graph_from_dfs

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/shared-data/scalpel/kristina/data/detection/usage/by video')
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='both')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--Project', choices=['MDN-IPG-BORIS', 'MDN-IPG-BORIS-NoSelfLoops'], default='MDN-IPG-BORIS', type=str)

    parser.add_argument('--seed', default=42, type=int)

    ## Embedding Arguments
    parser.add_argument('--embedding_model', choices=['Cora','DeepWalk','torch'], default='Cora')
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--embedding_walk_length', default=5, type=int)
    parser.add_argument('--embedding_walk_number', default=5, type=int)
    parser.add_argument('--embedding_window_size', default=2, type=int)
    parser.add_argument('--embedding_epochs', default=5 , type=int)
    parser.add_argument('--embedding_lr', default=0.07537842758627265, type=float)
    parser.add_argument('--embedding_min_count', default=1, type=int)

    parser.add_argument('--first_conv_f', default=256 , type=int)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--activation', choices=['relu','lrelu','tanh','sigmoid'], default='relu')

    # parser.add_argument('--embedding_walks_per_node', default=20, type=int)
    # parser.add_argument('--embedding_num_negative_samples', default=10, type=int)
    # parser.add_argument('--embedding_p', default=200, type=int)
    # parser.add_argument('--embedding_q', default=1, type=int)
    # parser.add_argument('--embedding_sparse', default=True, type=bool)

    parser.add_argument('--num_epochs', default=15, type=int)
    parser.add_argument('--lr', default=0.06673482330821637, type=float)

    args = parser.parse_args()
    assert args.first_conv_f/2**args.num_layers>=1
    return args

if __name__=='__main__':
    args = parsing()
    data = build_Graph_from_dfs(args, train_type = 'part2')