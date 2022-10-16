import sys
import torch
import os
import itertools
sys.path.insert(0, f'{"/".join((os.getcwd()).split("/")[:-1])}/classifier')
from surgery_data import SurgeryData
from utils import SURGERY_GROUPS
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import Planetoid
from ge import DeepWalk
import random
import numpy as np
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import pickle


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

def build_Graph(surgeries, plot=False, graph_worker='both', create_dicts=True):
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
        f = open("state_to_node.pkl", "wb")
        pickle.dump(state_to_node, f)
        f = open("node_to_state.pkl", "wb")
        pickle.dump(node_to_state, f)
    else:
        f = open("state_to_node.pkl", "rb")
        state_to_node=pickle.load(f)
        f = open("node_to_state.pkl", "rb")
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
    nx.draw(G, with_labels=True)
    plt.show()


def create_dataset_from_graph_cora(G):
    """
    :param G: Graph representing surgeries as built in "Build_Graph"
    :return: Dataset for network
    """
    dataset = Planetoid('./tmp/cora', 'Cora')
    num_nodes= G.number_of_nodes()
    x = dataset[0].x[:num_nodes]
    data = from_networkx(G)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data.x = x
    return data

def create_embeddings(G, args, save_embeddings = True):
    embedding_model = DeepWalk(G, args.embedding_walk_length,args.embedding_walk_number)
    embedding_model.train(window_size=args.embedding_window_size,iter=args.embedding_epochs,embed_size=args.embedding_dim)
    embeddings = embedding_model.get_embeddings()
    if save_embeddings:
        f = open(f"train_embeddings_{args.graph_worker}.pkl", "wb")
        pickle.dump(embeddings, f)
    return embeddings

def create_dataset(G,embedding_model='DeepWalk',args=None,load_embeddings=True, train_test_split=True):
    if embedding_model=='cora':
        data = create_dataset_from_graph_cora(G)
    else:
        if load_embeddings:
            # print('loading embeddings')
            f = open(f"train_embeddings_{args.graph_worker}.pkl", "rb")
            embeddings=pickle.load(f)
        else:
            embeddings = create_embeddings(G,args)
        nx.set_node_attributes(G, embeddings, "x")
        data = from_networkx(G)
    set_seed()
    if train_test_split:
        return train_test_split_edges(data, val_ratio=0.25, test_ratio=0.1)
    else:
        return data


class Net(torch.nn.Module):
    """
    Network class from sapir's code
    """
    def __init__(self, dataset,args):
        super(Net, self).__init__()
        self.dataset=dataset
        self.num_layers= args.num_layers
        net = []
        num_in_features=args.embedding_dim
        num_out_features = args.first_conv_f
        for i in range(args.num_layers):
            net.append(GCNConv(num_in_features,num_out_features))
            net.append(BatchNorm(num_out_features))
            num_in_features=num_out_features
            num_out_features=num_out_features//2
        self.net= torch.nn.ModuleList(net)


    def encode(self, x=None,edge_index=None):
        if x is None and edge_index is None:
            x=self.dataset.x
            edge_index = self.dataset.train_pos_edge_index
        x = self.net[0](x,edge_index )
        for i in range(1,self.num_layers*2-1,2):
            x=self.net[i](x)
            x=x.sigmoid()
            x=self.net[i+1](x,edge_index)
        return self.net[-1](x)

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