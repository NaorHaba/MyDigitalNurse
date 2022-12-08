import pandas as pd
import os
import sys
import argparse
from link_utils_new import build_Graph_from_dfs_helper, create_labels_from_surgery,build_graph_from_transitions, create_mappings_2
import pickle
from netrd.distance import Hamming
from scipy.spatial.distance import hamming
import numpy as np

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/shared-data/scalpel/kristina/data/detection/usage/by video')
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='both')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='disabled', type=str)
    parser.add_argument('--Project', choices=['MDN-IPG-BORIS', 'MDN-IPG-BORIS-NoSelfLoops'], default='MDN-IPG-BORIS-NoSelfLoops', type=str)

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

HANDS = {'both':[ 'R_surgeon','L_surgeon', 'R_assistant', 'L_assistant'],
        'surgeon': [ 'R_surgeon','L_surgeon'],
        'assitant': ['R_assistant', 'L_assistant']}


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

# def build_Graph_from_dfs_helper(surgeries, hands,state_to_node,withselfloops,graph_worker):

def load_data(save_path,args):
    surgeries = SURGERY_GROUPS_BORIS['train'] + SURGERY_GROUPS_BORIS['val'] + SURGERY_GROUPS_BORIS['test']
    if os.path.exists(save_path):
        print('load')
        f = open(save_path, "rb")
        data = pickle.load(f)
    else:
        data={}
        hands = HANDS[args.graph_worker]
        if os.path.exists(f"node_to_state_{args.graph_worker}_new.pkl"):
            f = open(f"node_to_state_{args.graph_worker}_new.pkl", "rb")
            node_to_state = pickle.load(f)
            f = open(f"state_to_node_{args.graph_worker}_new.pkl", "rb")
            state_to_node = pickle.load(f)
        else:
            state_to_node, node_to_state=create_mappings_2(surgeries,hands,args.graph_worker)
        for i, surgery in enumerate(surgeries):  ##Leaves 1 Surgery out, for training the network
            cur_surgeries = surgeries[:i] + surgeries[i + 1:]
            all_transitions, all_nodes = build_Graph_from_dfs_helper(cur_surgeries, hands, state_to_node, False,
                                                                     args.graph_worker)
            sur_transitions, sur_nodes = build_Graph_from_dfs_helper([surgery], hands, state_to_node, False,
                                                                     args.graph_worker)
            sur_labels = create_labels_from_surgery(surgery, hands, state_to_node, False, args.graph_worker)
            G_all = build_graph_from_transitions(all_nodes, all_transitions)
            G_sur = build_graph_from_transitions(sur_nodes, sur_transitions)
            data[i] = {'full graph':G_all, 'sur graph': G_sur, 'sur_labels': sur_labels}
        f = open(save_path, "wb")
        pickle.dump(data,f)
    return data

def find_best_seq(l1,l2):
    """
    important: len(L1)<len(L2)
    """
    min_hamming = 1
    n1 = len(l1)
    n2 = len(l2)
    for i in range(0,n2-n1+1):
        tmp_hamming = hamming(l1,l2[i:i+n1])
        if tmp_hamming<min_hamming:
            min_hamming=tmp_hamming
    return min_hamming


def calc_hamming_labels(i,labels, all_data):
    hammings = []
    n= len(labels)
    for j, j_dic in all_data.items():
        if j==i:
            continue
        else:
            l2_labels = j_dic['sur_labels'][0]
            if len(l2_labels)<n:
                hammings.append(find_best_seq(l2_labels,labels))
            else:
                hammings.append(find_best_seq(labels,l2_labels))
    return hammings


if __name__=='__main__':
    args = parsing()
    # path = '/data/home/liory/MyDigitalNurse/GCN/checkgraphs.pkl'
    path = '/data/home/liory/MyDigitalNurse/GCN/checkgraphs_both.pkl'
    data= load_data(path,args)
    graph_hammings = []
    label_hammings = []
    dist_obj = Hamming()
    length = [len(x['sur_labels'][0]) for x in data.values()]
    print(np.min(length))
    print(np.max(length))
    print(np.mean(length))
    # for i, i_dic in data.items():
    #     full_graph = i_dic['full graph']
    #     i_graph =  i_dic['sur graph']
    #     i_graph.add_nodes_from([x for x in full_graph.nodes if x not in i_graph.nodes])
    #     full_graph.add_nodes_from([x for x in i_graph.nodes if x not in full_graph.nodes])
    #     graph_hammings.append(dist_obj(full_graph,i_graph))
    #     label_hammings.append(calc_hamming_labels(i,i_dic['sur_labels'][0],data))
    # print('mean graph hammings: ', np.mean(graph_hammings))
    # print('mean seq hammings: ', np.mean(np.mean([x for x in label_hammings])))
    print('DONE')