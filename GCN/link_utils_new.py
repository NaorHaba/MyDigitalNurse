import sys
import torch
import os
import itertools

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx, negative_sampling

from torch_geometric.datasets import Planetoid
from ge import DeepWalk
import random
import numpy as np
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import pickle
import wandb
import pandas as pd
import random

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


def save_mappings(n_workers = 2):
    state_to_node = {}
    node_to_state= {}
    for i, node in enumerate(itertools.product(list(range(5)), repeat=n_workers)):
        state_to_node[node] = i
        node_to_state[i] = node
    f = open(f"state_to_node_{n_workers}.pkl", "wb")
    pickle.dump(state_to_node, f)
    f = open(f"node_to_state_{n_workers}.pkl", "wb")
    pickle.dump(node_to_state, f)


def build_Graph_from_dfs_new(args, train_type='Link', val_ratio=0.25, test_ratio=0.1):
    '''
    :param surgeries: list of surgery tensors of shape Num_Frames,num_hands
    :return: Graph reoresentation of all surgeries
    '''
    withselfloops = False if 'NoSelfLoops' in args.Project else True
    hands = HANDS[args.graph_worker]
    tuple_len = len(hands)
    f = open(f"state_to_node_{args.graph_worker}_new.pkl", "rb")
    state_to_node=pickle.load(f)
    f = open(f"node_to_state_{args.graph_worker}_new.pkl", "rb")
    node_to_state=pickle.load(f)
    nodes = list(node_to_state.keys())
    if train_type == 'Link':
        surgeries = SURGERY_GROUPS_BORIS['train']+SURGERY_GROUPS_BORIS['val']+SURGERY_GROUPS_BORIS['test']
        all_transitions,all_nodes = build_Graph_from_dfs_helper(surgeries, hands,state_to_node,withselfloops, args.graph_worker)
        train_ratio = 1 - val_ratio - test_ratio
        edges = list(all_transitions.keys())
        set_seed()
        random.shuffle(edges)
        num_edges = len(edges)
        train_edges = edges[:int(train_ratio * num_edges)]
        val_edges = edges[int(train_ratio * num_edges):int((train_ratio + val_ratio) * num_edges)]
        test_edges = edges[int((train_ratio + val_ratio) * num_edges):]
        G_train = build_graph_from_transitions(nodes,all_transitions,train_edges)
        G_val = build_graph_from_transitions(nodes,all_transitions,val_edges)
        G_test = build_graph_from_transitions(nodes,all_transitions,test_edges)
        G_full = build_graph_from_transitions(nodes,all_transitions,edges)
        G_full_data = from_networkx(G_full)
        dataset= create_dataset(G_train,args=args)
        val_data = from_networkx(G_val)
        neg_val_edge_index = negative_sampling(
            edge_index=G_full_data.edge_index, num_nodes=val_data.num_nodes,
            num_neg_samples=val_data.edge_index.size(1)) #Negative
        test_data = from_networkx(G_test)
        neg_test_edge_index = negative_sampling(
            edge_index=G_full_data.edge_index, num_nodes=test_data.num_nodes,
            num_neg_samples=test_data.edge_index.size(1)) #
        dataset.train_pos_edge_index=dataset.edge_index
        if withselfloops:
            dataset.train_weight=dataset.weight.to(torch.float32)
            dataset.val_weight = val_data.weight.to(torch.float32)
            dataset.test_weight = test_data.weight.to(torch.float32)
        else:
            dataset.train_weight=dataset.weight
            dataset.val_weight = val_data.weight
            dataset.test_weight = test_data.weight
        dataset.val_pos_edge_index=val_data.edge_index
        dataset.val_neg_edge_index=neg_val_edge_index
        dataset.test_pos_edge_index=test_data.edge_index
        dataset.test_neg_edge_index=neg_test_edge_index
        return dataset, G_full_data.edge_index
    else:
        if withselfloops:
            save_datasets_path = f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_{args.graph_worker}_new'
        else:
            save_datasets_path = f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_NoSelfLoops_{args.graph_worker}_new'
        train_surgeries = SURGERY_GROUPS_BORIS['train']
        if train_type=='p2':
            os.mkdir(save_datasets_path)
        for i,surgery in enumerate(train_surgeries): ##Leaves 1 Surgery out, for training the network
            cur_path = os.path.join(save_datasets_path,f'{i}')
            if train_type == 'p2':
                os.mkdir(cur_path)
            cur_surgeries =train_surgeries[:i]+train_surgeries[i+1:]
            all_transitions, all_nodes = build_Graph_from_dfs_helper(cur_surgeries,hands,state_to_node,withselfloops,args.graph_worker)
            labels,weights = create_labels_from_surgery(surgery,hands,state_to_node,withselfloops, args.graph_worker)
            G = build_graph_from_transitions(all_nodes, all_transitions)
            if train_type=='check_graphs':
                sample_transitions = build_Graph_from_dfs_helper([surgery],hands,state_to_node,withselfloops,args.graph_worker)
                sample_graph = build_graph_from_transitions(nodes, sample_transitions)
                # train_data_dict[i]={'Graph':G,'sample_graph':sample_graph}
            if train_type == 'p2':
                nx.write_gpickle(G,os.path.join(cur_path,'Graph.gpickle'))
                torch.save(labels, os.path.join(cur_path,'Labels'))
                if len(weights)>0:
                    torch.save(weights, os.path.join(cur_path, 'Weights'))
        # Creates 1 graph for all train surgeries, for testing and validation
        all_transitions = build_Graph_from_dfs_helper(train_surgeries, hands, state_to_node, withselfloops,args.graph_worker)
        G = build_graph_from_transitions(nodes, all_transitions)
    #     print(nodes)
    #     print(all_transitions)
    #     train_data_dict['full_train'] = G
    #     if train_type=='p2':
    #         nx.write_gpickle(G, os.path.join(save_datasets_path, 'Full_Training_Graph.gpickle'))
    #         cur_path = os.path.join(save_datasets_path, 'val')
    #         os.mkdir(cur_path)
    #         labels=[]
    #         weights = []
    #         for surgery in SURGERY_GROUPS_BORIS['val']:
    #             sur_labels, sur_weights = create_labels_from_surgery(surgery,hands,state_to_node,withselfloops,args.graph_worker)
    #             labels.append(sur_labels)
    #             if len(sur_weights)>0:
    #                 weights.append(sur_weights)
    #             # labels.append(create_labels_from_surgery(surgery,hands,state_to_node,withselfloops,args.graph_worker))
    #         f = open(os.path.join(cur_path, 'labels.pkl'), "wb")
    #         pickle.dump(labels, f)
    #         f = open(os.path.join(cur_path, 'weights.pkl'), "wb")
    #         pickle.dump(weights, f)
    #         cur_path = os.path.join(save_datasets_path, 'test')
    #         os.mkdir(cur_path)
    #         labels = []
    #         weights = []
    #         for surgery in SURGERY_GROUPS_BORIS['test']:
    #             sur_labels, sur_weights = create_labels_from_surgery(surgery,hands,state_to_node,withselfloops,args.graph_worker)
    #             labels.append(sur_labels)
    #             if len(sur_weights)>0:
    #                 weights.append(sur_weights)
    #             # labels.append(create_labels_from_surgery(surgery,hands,state_to_node,withselfloops,args.graph_worker))
    #         f = open(os.path.join(cur_path, 'labels.pkl'), "wb")
    #         pickle.dump(labels, f)
    #         f = open(os.path.join(cur_path, 'weights.pkl'), "wb")
    #         pickle.dump(weights, f)
    #     if train_type=='check_graphs':
    #         return train_data_dict


def create_labels_from_surgery(surgery, hands,state_to_node,withselfloops, graph_worker):
    df = pd.read_csv(os.path.join(BORIS, surgery), skiprows=15)
    processed_df = process_boris_df(df, graph_worker)
    processed_df['Shift_Time'] = processed_df['Time_Diff'].shift(-1).fillna(1/processed_df.FPS)
    states = []
    weights = []
    for i, row in processed_df.iterrows():
        state = tuple(int(row[x].item()) for x in hands)
        states += [state_to_node[state]]
        weights += [(row.FPS*row.Shift_Time)]
        # states += [state_to_node[state]] * 1 if not withselfloops else [state_to_node[state]] * int(row.FPS*row.Shift_Time)
    return torch.tensor(states), torch.tensor(weights)



def datasets_for_part_2_overfitcheck(args,g,l):
    train_data = {}
    embedding_dir = os.path.join(args.files_dir, args.load_embeddings)
    for i in range(20):
        data_path = os.path.join(args.videos_path,f'{i}')
        # Graph = nx.read_gpickle(os.path.join(os.path.join(g,f'{i}'),"Graph.gpickle"))
        Graph = nx.DiGraph()
        Graph.add_nodes_from(list(range(25)))
        labels =  torch.load(os.path.join(os.path.join(l,f'{i}'),"Labels"))
        data = create_dataset(G=Graph, args=args, load_embeddings=embedding_dir)
        data.weight = torch.tensor([])
        train_data[i]={'data':data,'labels':labels}
    return train_data, None,None,None
    # return full_dataset,val_labels,test_labels

def read_train_data(i,args):
    data_path = os.path.join(args.videos_path, f'{i}')
    Graph = nx.read_gpickle(os.path.join(data_path, "Graph.gpickle"))
    labels = torch.load(os.path.join(data_path, "Labels"))
    embedding_dir = os.path.join(args.files_dir, args.load_embeddings)
    data = create_dataset(G=Graph, args=args, load_embeddings=embedding_dir)
    return data,labels


def build_Graph_from_dfs_helper(surgeries, hands,state_to_node,withselfloops,graph_worker):
    transitions = {}
    nodes = []
    for sur in surgeries:
        df = pd.read_csv(os.path.join(BORIS,sur), skiprows=15)
        processed_df = process_boris_df(df,graph_worker)
        for i_stage in range(len(processed_df)-1):
            source_row = processed_df.loc[i_stage]
            target_row = processed_df.loc[i_stage+1]
            source = tuple(int(source_row[x].item()) for x in hands)
            target = tuple(int(target_row[x].item()) for x in hands)
            if source==target:
                print('P')
            source_node = state_to_node[source]
            key = (source_node, state_to_node[target])
            if key in transitions.keys():
                transitions[key] += 1
            else:
                transitions[key] = 1
            if source_node not in nodes:
                nodes.append(source_node)
    return transitions, nodes

def build_Graph_from_dfs_helper_partial(surgeries, hands,state_to_node,withselfloops,graph_worker,partial_sur=None,partial_size=None):
    transitions = {}
    nodes = []
    for sur in surgeries:
        df = pd.read_csv(os.path.join(BORIS,sur), skiprows=15)
        processed_df = process_boris_df(df,graph_worker)
        for i_stage in range(len(processed_df)-1):
            source_row = processed_df.loc[i_stage]
            target_row = processed_df.loc[i_stage+1]
            source = tuple(int(source_row[x].item()) for x in hands)
            target = tuple(int(target_row[x].item()) for x in hands)
            if source==target:
                print('P')
            source_node = state_to_node[source]
            key = (source_node, state_to_node[target])
            if key in transitions.keys():
                transitions[key] += 1
            else:
                transitions[key] = 1
            if source_node not in nodes:
                nodes.append(source_node)
    if partial_sur is not None:
        df = pd.read_csv(os.path.join(BORIS,partial_sur), skiprows=15)
        processed_df = process_boris_df(df,graph_worker)
        for i_stage in range(partial_size):
            source_row = processed_df.loc[i_stage]
            target_row = processed_df.loc[i_stage+1]
            source = tuple(int(source_row[x].item()) for x in hands)
            target = tuple(int(target_row[x].item()) for x in hands)
            if source==target:
                print('P')
            source_node = state_to_node[source]
            key = (source_node, state_to_node[target])
            if key in transitions.keys():
                transitions[key] += 1
            else:
                transitions[key] = 1
        labels = []
        for i_stage in range(partial_size, len(processed_df)):
            state_row = processed_df.loc[i_stage]
            state = tuple(int(state_row[x].item()) for x in hands)
            node = state_to_node[state]
            labels.append(node)
    return transitions, nodes, labels

def datasets_for_part_2(args):
    train_data = {}
    embedding_dir = os.path.join(args.files_dir, args.load_embeddings)
    val_weights = test_weights = None
    hands = HANDS[args.graph_worker]
    tuple_len = len(hands)
    f = open(f"state_to_node_{args.graph_worker}_new.pkl", "rb")
    state_to_node=pickle.load(f)
    f = open(f"node_to_state_{args.graph_worker}_new.pkl", "rb")
    node_to_state=pickle.load(f)
    train_surgeries =SURGERY_GROUPS_BORIS['train']+SURGERY_GROUPS_BORIS['val']
    unk_node = max(list(node_to_state.keys()))+1
    for i, surgery in enumerate(train_surgeries):  ##Leaves 1 Surgery out, for training the network
        cur_surgeries = train_surgeries[:i] + train_surgeries[i + 1:]
        transitions, nodes, labels = build_Graph_from_dfs_helper_partial(cur_surgeries, hands, state_to_node, False,
                                                                 args.graph_worker,train_surgeries[i],args.partial_start)
        #TODO: add UNK node+embeddings
        G = build_graph_from_transitions(nodes+[unk_node], transitions)
        data = create_dataset(G=G, args=args, load_embeddings=embedding_dir)
        data.weight = data.weight.to(torch.float32)
        train_data[i]={'data':data,'labels':labels}
    return train_data

def build_graph_from_transitions(nodes,transitions,edges=None):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    if edges==None:
        edges = list(transitions.keys())
    for edge in edges:
        source = edge[0]
        target = edge[1]
        G.add_weighted_edges_from([(source, target, transitions[edge])])
    return G

def create_dataset(G,args=None,load_embeddings=None):
    if args.embedding_model=='Cora':
        data = create_dataset_from_graph_cora(G)
    elif args.embedding_model=='DeepWalk':
        if load_embeddings:
            # print('loading embeddings')
            f = open(load_embeddings, "rb")
            embeddings=pickle.load(f)
        else:
            embeddings = create_embeddings(G,args)
        node_ids = {i:i for i in G.nodes}
        nx.set_node_attributes(G, embeddings, "x")
        nx.set_node_attributes(G, node_ids, "id")
        data = from_networkx(G)
    else:
        n = len(G.nodes)
        embeddings = {i: i for i in range(n)}
        nx.set_node_attributes(G, embeddings, "x")
        data = from_networkx(G)
        # data.x = data.x.to(torch.float32)
    return data




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_info_from_label(label):
    if len(label) == 1:
        worker = 'surgeon'
        side = label
        tool_id = '0'
    elif len(label) == 2:
        worker = 'assistant'
        side = label[0]
        tool_id = '0'
    else:
        if label[1] == '2':
            worker = 'assistant'
        else:
            worker = 'surgeon'
        side = label[0]
        tool_id = label[-1]
    return worker, side, tool_id

def lior_func(args):
    sur = SURGERY_GROUPS_BORIS['train'][19]
    hands = HANDS[args.graph_worker]
    tuple_len = len(hands)
    f = open(f"state_to_node_{tuple_len}.pkl", "rb")
    state_to_node=pickle.load(f)
    labels = create_labels_from_surgery(sur,hands,state_to_node,False)

#TODO
def process_boris_df(df, graph_worker):
    index=0
    first_state = 4 if graph_worker=='both' else 2
    insert_dict = {'Index':index,'Time':0,'Time_Diff':0,'R_surgeon':0,'L_surgeon':0,'R_assistant':0,'L_assistant':0,'FPS':30}
    df = df[['Time', 'FPS', 'Behavior', 'Status']][df.Status == 'START']
    if graph_worker=='surgeon':
        df = df[(~df.Behavior.str.contains('L2'))&(~df.Behavior.str.contains('R2'))]
        insert_dict = {'Index': index, 'Time': 0, 'Time_Diff': 0, 'R_surgeon': 0, 'L_surgeon': 0, 'FPS': 30}
    elif graph_worker=='assistant':
        df = df[(df.Behavior.str.contains('L2'))|(df.Behavior.str.contains('R2'))]
        insert_dict = {'Index': index, 'Time': 0, 'Time_Diff': 0, 'R_assistant':0,'L_assistant':0, 'FPS': 30}
    new_df = pd.DataFrame(insert_dict,index=[0])
    df = df.reset_index()
    for i, row in df.iterrows():
        worker, side, tool_id = get_info_from_label(row.Behavior)
        tool_id = int(tool_id)
        if i<first_state:
            if tool_id!=0:
                new_df.at[0, f'{side}_{worker}'] = tool_id
        else:
            index+=1
            time_diff = row.Time - insert_dict['Time']
            if time_diff==0:
                new_df.at[index-1, f'{side}_{worker}'] = tool_id
                insert_dict[f'{side}_{worker}'] = tool_id
                index-=1
            else:
                insert_dict['Index']=index
                insert_dict['Time_Diff'] = time_diff
                insert_dict['Time'] = row.Time
                insert_dict['FPS'] = row.FPS
                insert_dict[f'{side}_{worker}'] = tool_id
                new_df=new_df.append(insert_dict,ignore_index=True)
    return new_df



def read_surgeries(videos_path:str, group:str='train'):
    """
    :param path: path to surgeries file to read the labels
    :return: list of surgery tensors of shape (Num_Frames,4 (hands))
    """
    surgery_labels = {'surgeon':[],'assistant':[],'both':[]}
    for sur_path in SURGERY_GROUPS[group]:
        sur_dir = os.path.join(videos_path, sur_path)
        sur_data = SurgeryData(sur_dir)
        labels = torch.stack(sur_data.frame_labels)
        surgery_labels['surgeon'].append(labels[:,:2])
        surgery_labels['assistant'].append(labels[:,2:])
        surgery_labels['both'].append(labels)
    return surgery_labels

def build_Graph(surgeries, plot=False, graph_worker='both', create_dicts=False):
    '''
    :param surgeries: list of surgery tensors of shape Num_Frames,num_hands
    :return: Graph reoresentation of all surgeries
    '''
    tuple_len = 4 if graph_worker=='both' else 2
    transitions = {}
    i=0
    state_to_node = {}
    node_to_state={}
    if create_dicts:
        for i,node in enumerate(itertools.product([0, 1, 2, 3, 4], repeat=tuple_len)):
            state_to_node[node] = i
            node_to_state[i] = node
        f = open(f"state_to_node_{graph_worker}.pkl", "wb")
        pickle.dump(state_to_node, f)
        f = open(f"node_to_state_{graph_worker}.pkl", "wb")
        pickle.dump(node_to_state, f)
    else:
        f = open(f"state_to_node_{graph_worker}.pkl", "rb")
        state_to_node=pickle.load(f)
        f = open(f"node_to_state_{graph_worker}.pkl", "rb")
        node_to_state=pickle.load(f)
    for sur in surgeries:
        for i_stage in range(sur.shape[0] - 1):
            source = tuple(x.item() for x in sur[i_stage])
            target = tuple(x.item() for x in sur[i_stage + 1])
            key = (state_to_node[source], state_to_node[target])
            if key in transitions.keys():
                transitions[key] += 1
            else:
                transitions[key] = 1
    G = nx.DiGraph(nodes=list(node_to_state.keys()))
    G.add_nodes_from(list(node_to_state.keys()))
    for pair, weight in transitions.items():
        source = pair[0]
        target = pair[1]
        G.add_weighted_edges_from([(source, target, weight)])
    if plot:
        plot_graph(G)
    return G, state_to_node,node_to_state

def plot_graph(G):
    print("The graph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    pos = nx.spring_layout(G, seed=1)
    plt.figure(figsize=(10, 10))
    nx.draw(G, with_labels=True, pos=pos)
    plt.show()


def create_dataset_from_graph_cora(G):
    """
    :param G: Graph representing surgeries as built in "Build_Graph"
    :return: Dataset for network
    """
    dataset = Planetoid('./tmp/cora', 'Cora')
    # num_nodes= G.number_of_nodes()
    node_ids = {i: i for i in G.nodes}
    nodes = list(G.nodes)
    x = dataset[0].x[nodes]
    nx.set_node_attributes(G, node_ids, "id")
    data = from_networkx(G)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data.x = x
    return data

def create_embeddings(G, args, save_embeddings = True):
    embedding_model = DeepWalk(G, args.embedding_walk_length,args.embedding_walk_number)
    embedding_model.train(window_size=args.embedding_window_size,iter=args.embedding_epochs,embed_size=args.embedding_dim)
    embeddings = embedding_model.get_embeddings()
    if save_embeddings:
        f = open(os.path.join(wandb.run.dir,f"train_embeddings_{args.graph_worker}.pkl"), "wb")
        pickle.dump(embeddings, f)
    return embeddings

class MLP(torch.nn.Module):
    def __init__(self,dataset, args, leaky_slope=0.2, dropout=0.4):
        super(MLP, self).__init__()
        self.input_size = args.embedding_dim
        self.num_classes = 25
        hidden_layers = [args.first_conv_f//(2**i) for i in range(args.num_layers)]
        model_layers = []
        # add hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                model_layers.append(torch.nn.Linear(self.input_size, hidden_layers[i]))
            else:
                model_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            # model_layers.append(torch.nn.BatchNorm1d(hidden_layers[i]))
            model_layers.append(torch.nn.LeakyReLU(leaky_slope))
            # model_layers.append(torch.nn.Dropout(dropout))

        # add output layer
        model_layers.append(torch.nn.Linear(hidden_layers[-1], self.num_classes))
        # model_layers.append(torch.nn.Sigmoid())
        self.classifier = torch.nn.Sequential(*model_layers)

    def encode(self, x=None,edge_index=None, weight=None, last_step=None):
        return self.classifier(x)

    def predict(self, last_step,x=None,edge_index=None,weight=None,removeselfloops=False):
        # cur = torch.cat([x[last_step],torch.tensor([i])])
        output = self.classifier(x.unsqueeze(0))
        # pred = torch.argmax(output)
        return output.squeeze()







class Net(torch.nn.Module):
    """
    Network class from sapir's code
    """
    def __init__(self, dataset,args):
        super(Net, self).__init__()
        self.dataset=dataset
        self.num_layers= args.num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = []
        num_in_features=args.embedding_dim
        num_out_features = args.first_conv_f
        activation = ACTIVATIONS[args.activation]
        for i in range(args.num_layers):
            print(f'in features {num_in_features} out features {num_out_features} ')
            net.append(GCNConv(num_in_features,num_out_features))
            net.append(BatchNorm(num_out_features))
            num_in_features=num_out_features
            num_out_features=num_out_features//2
            if i<args.num_layers-1:
                net.append(activation())
        self.net= torch.nn.Sequential(*net)

    def encode_p2_noselfloops(self,x=None,edge_index=None, last_step=None):
        if x is None and edge_index is None:
            x=self.dataset.x
            x=x.long()
            edge_index = self.dataset.train_pos_edge_index
            weight = self.dataset.train_weight.to(torch.float32)
        for i,layer in enumerate(self.net)[:-1]:
            if (i-1)%3==0:
                x= layer(x,edge_index, weight)
                x=x.to(torch.float32)
            else:
                x= layer(x)
        return(x)

    def encode(self, x=None,edge_index=None, weight=None, last_step=None):
        if x is None and edge_index is None:
            x=self.dataset.x
            edge_index = self.dataset.train_pos_edge_index
            weight = self.dataset.train_weight.to(torch.float32)
        else:
            weight= weight.to(torch.float32)
        for i,layer in enumerate(self.net[:-1]):
            if i%3==0:
                x= layer(x,edge_index, weight)
                x=x.to(torch.float32)
            else:
                x= layer(x)
        return(x)
        # x = self.net[0](x,edge_index )
        # for i in range(1,self.num_layers*2-1,2):
        #     x=self.net[i](x)
        #     x=x.sigmoid() #TODO: add as layer
        #     x=self.net[i+1](x,edge_index)
        # return self.net[-1](x)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def predict(self, last_step,x=None,edge_index=None,weight=None,removeselfloops=False):
        z = self.encode(x,edge_index,weight)
        prob_adj = z @ z.t()
        rel_edges = prob_adj[last_step][:]
        if removeselfloops:
            rel_edges[last_step] = np.float('-inf')
        # pred = torch.argmax(rel_edges).item()
        # return pred, rel_edges
        return rel_edges


class Net_with_embedding(torch.nn.Module):
    """
    Network class from sapir's code
    """
    def __init__(self, dataset, args):
        super(Net_with_embedding, self).__init__()
        self.dataset=dataset
        self.args=args
        self.num_layers= args.num_layers
        net = []
        num_in_features=args.embedding_dim
        num_out_features = args.first_conv_f
        net.append(torch.nn.Embedding(dataset.x.shape[0], args.embedding_dim))
        activation = ACTIVATIONS[args.activation]
        for i in range(args.num_layers):
            net.append(GCNConv(num_in_features,num_out_features))
            net.append(BatchNorm(num_out_features))
            num_in_features=num_out_features
            num_out_features=num_out_features//2
            if i<args.num_layers-1:
                net.append(activation())
        self.net= torch.nn.Sequential(*net)



    def encode(self, x=None,edge_index=None):
        if x is None and edge_index is None:
            x=self.dataset.x
            x=x.long()
            edge_index = self.dataset.train_pos_edge_index
            weight = self.dataset.train_weight.to(torch.float32)
        for i,layer in enumerate(self.net):
            if (i-1)%3==0:
                x= layer(x,edge_index, weight)
                x=x.to(torch.float32)
            else:
                x= layer(x)
        return(x)


    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def predict(self, z,last_step):
        prob_adj = z @ z.t()
        rel_edges = prob_adj[last_step][:]
        return torch.argmax(rel_edges).item(), rel_edges

# sources = dataset.edge_index[0,:]
# source_bool=torch.isin(sources, torch.tensor(last_node))
# source_bool,_ = torch.where(torch.isin(sources, torch.tensor(last_node))==True)
# targets = dataset.edge_index[1,:]
# target_bool=torch.isin(targets, torch.tensor(last_node))
# target_bool,_ = torch.where(target_bool==True)

def find_edge_idx(edge_index, source,target):
    sources = edge_index[0,:]
    source_bool= torch.where(torch.isin(sources, torch.tensor(source))==True)[0]
    targets = edge_index[1,:][source_bool]
    target_index = torch.where(torch.isin(targets, torch.tensor(target))==True)[0]
    if target_index.shape[0]==1:
        cur_edge_index = source_bool[target_index]
        return cur_edge_index
    else:
        return -1