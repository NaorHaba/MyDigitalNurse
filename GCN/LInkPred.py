##SETUP
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from torch_geometric.utils import negative_sampling, from_networkx
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.utils import train_test_split_edges
import torch_geometric.data
import networkx as nx
import matplotlib.pyplot as plt
from ge import DeepWalk
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd
import os
import datetime
from surgery_data import SurgeryData
from utils import SURGERY_GROUPS
import numpy as np
import argparse
from gensim.models import Word2Vec
from ..walker import RandomWalker
## SAGE ?

# Videos path = '/data/shared-data/scalpel/kristina/data/detection/usage/by video'



def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/shared-data/scalpel/kristina/data/detection/usage/by video')
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='both')

    ## Embedding Arguments
    parser.add_argument('--embedding_model', choices=['Node2Vec','DeepWalk'], default='DeepWalk')
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--embedding_walk_length', default=80, type=int)
    parser.add_argument('--embedding_walk_number', default=10, type=int)
    parser.add_argument('--embedding_window_size', default=5, type=int)
    parser.add_argument('--embedding_epochs', default=10, type=int)
    parser.add_argument('--embedding_lr', default=0.05, type=float)
    parser.add_argument('--embedding_min_count', default=1, type=int)

    # parser.add_argument('--embedding_walks_per_node', default=20, type=int)
    # parser.add_argument('--embedding_num_negative_samples', default=10, type=int)
    # parser.add_argument('--embedding_p', default=200, type=int)
    # parser.add_argument('--embedding_q', default=1, type=int)
    # parser.add_argument('--embedding_sparse', default=True, type=bool)

    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.00876569212062032, type=float)

    args = parser.parse_args()
    return args


def read_surgeries(videos_path:str):
    """
    :param path: path to surgeries file to read the labels
    :return: list of surgery tensors of shape (Num_Frames,4 (hands))
    """
    surgery_labels = {'surgeon':[],'assistant':[],'both':[]}
    for sur_path in SURGERY_GROUPS['train']:
        sur_dir = os.path.join(videos_path, sur_path)
        sur_data = SurgeryData(sur_dir)
        surgery_labels['surgeon'].append(sur_data.frame_labels[:,:2])
        surgery_labels['assistant'].append(sur_data.frame_labels[:,2:])
        surgery_labels['both'].append(sur_data.frame_labels)
    return surgery_labels

def build_Graph(surgeries, plot=False):
    '''
    :param surgeries: list of surgery tensors of shape Num_Frames,num_hands
    :return: Graph reoresentation of all surgeries
    '''
    transitions = {}
    i=0
    state_to_node = {}
    node_to_state={}
    for sur in surgeries:
        for i_stage in range(sur.shape[0] - 1):
            source = tuple(x.item() for x in sur[i_stage])
            target = tuple(x.item() for x in sur[i_stage + 1])
            if source not in state_to_node.keys():
                state_to_node[source]=i
                source_code = i
                node_to_state[i]=source
                i+=1
            else:
                source_code= state_to_node[source]
            if target not in state_to_node.keys():
                state_to_node[target]=i
                target_code=i
                node_to_state[i]=target
                i+=1
            else:
                target_code= state_to_node[source]
            key = (source_code, target_code)
            if key in transitions.keys():
                transitions[key] += 1
            else:
                transitions[key] = 1
    G = nx.DiGraph()
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
    data = train_test_split_edges(data, val_ratio=0.25, test_ratio=0.1)
    return data

def create_dataset_with_embeddings(G, args,embedding_model_name='DeepWalk'):
    if embedding_model_name=='DeepWalk':
        embedding_model = DeepWalk(G, args.embedding_walk_length,args.embedding_walk_number)
    else:
        embedding_model = Node2Vec( **embedding_args).to(device)
    embedding_model.train(window_size=args.embedding_window_size,iter=args.embedding_epochs,embed_size=args.embedding_dim)
    embeddings = embedding_model.get_embeddings()
    nx.set_node_attributes(G, embeddings, "x")
    data = from_networkx(G)
    return train_test_split_edges(data, val_ratio=0.25, test_ratio=0.1)

class Net(torch.nn.Module):
    """
    Network class from sapir's code
    """
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.dataset=dataset
        self.conv1 = GCNConv(dataset.num_features, 1024)
        self.bn1 = BatchNorm(1024)
        self.conv2 = GCNConv(1024, 512)
        self.bn2 = BatchNorm(512)
        self.conv3 = GCNConv(512, 256)
        self.bn3 = BatchNorm(256)
        self.conv4 = GCNConv(256, 128)
        self.bn4 = BatchNorm(128)
        self.conv5 = GCNConv(128, 64)
        self.bn5 = BatchNorm(64)
        self.conv6 = GCNConv(64, 32)

    def encode(self):
        x = self.conv1(self.dataset.x, self.dataset.train_pos_edge_index)
        x = self.bn1(x)
        x = x.sigmoid()
        x = self.conv2(x, self.dataset.train_pos_edge_index)
        x = self.bn2(x)
        x = x.sigmoid()
        x = self.conv3(x, self.dataset.train_pos_edge_index)
        x = self.bn3(x)
        x = x.sigmoid()
        x = self.conv4(x, self.dataset.train_pos_edge_index)
        x = self.bn4(x)
        x = self.conv5(x, self.dataset.train_pos_edge_index)
        return self.bn5(x)

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
        return torch.argmax(prob_adj[0][:]).item()

    # parser.add_argument('--embedding_model', choices=['Node2Vec','DeepWalk'], default='DeepWalk')
    # parser.add_argument('--embedding_dim', default=128, type=int)
    # parser.add_argument('--embedding_walk_length', default=80, type=int)
    # parser.add_argument('--embedding_walk_number', default=10, type=int)
    # parser.add_argument('--embedding_window_size', default=5, type=int)
    # parser.add_argument('--embedding_epochs', default=10, type=int)
    # parser.add_argument('--embedding_lr', default=0.05, type=float)
    # parser.add_argument('--embedding_min_count', default=1, type=int)

class Trainer():
    def __init__(self, G:nx.DiGraph, model:Net,optimizer:torch.optim, dataset:torch_geometric.data.data.Data):
        self.model = model
        self.G=G
        self.data=dataset
        self.optimizer=optimizer

    def get_link_labels(self, pos_edge_index,neg_edge_index):
        """
        :param pos_edge_index:
        :param neg_edge_index:
        :return: tensor with 1 for existing edge and 0 for negative edge
        """
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def train(self):
        """
        :return:
        """
        self.model.train()
        neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index, num_nodes=data.num_nodes,
            num_neg_samples=self.data.train_pos_edge_index.size(1)) #Negative
        optimizer.zero_grad()
        z = self.model.encode()
        link_logits = self.model.decode(z, self.data.train_pos_edge_index, neg_edge_index)
        link_labels = self.get_link_labels(self.data.train_pos_edge_index,neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        self.optimizer.step()
        return loss


    @torch.no_grad()
    def test(self):
        self.model.eval()
        perfs = []
        for prefix in ["val", "test"]:
            pos_edge_index = self.data[f'{prefix}_pos_edge_index']
            neg_edge_index = self.data[f'{prefix}_neg_edge_index']

            z = self.model.encode()
            link_logits = self.model.decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = self.get_link_labels(pos_edge_index,neg_edge_index)
            perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))

        z = self.model.encode()
        link_logits = self.model.decode(z, self.data.val_pos_edge_index, self.data.val_neg_edge_index)
        link_labels = self.get_link_labels(self.data.val_pos_edge_index, self.data.val_neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        perfs.append(loss)
        return perfs


if __name__ == "__main__":
    args = parsing()
    surgeries_data = read_surgeries(args.videos_path)
    G, state_to_node,node_to_state = build_Graph(surgeries_data[args.graph_worker])
    # data = create_dataset_from_graph_cora(G)
    data = create_dataset_with_embeddings(G, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(data).to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    trainer= Trainer(G, model, optimizer,data)
    best_val_perf = test_perf = 0
    train_losses, val_losses = [], []
    val_acc, test_acc = [], []
    for epoch in range(1, args.num_epochs):
        train_loss = trainer.train()
        train_losses.append(train_loss)
        val_perf, tmp_test_perf, val_loss = trainer.test()
        val_losses.append(val_loss)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        val_acc.append(best_val_perf)
        test_acc.append(test_perf)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'
        print(log.format(epoch, train_loss, val_loss, best_val_perf, test_perf))
    z = trainer.model.encode()
    final_edge_index = trainer.model.decode_all(z)
    # torch.save(trainer.model.state_dict(), '/home/liory/GCN/IPG')