import sys
import torch
import os
sys.path.insert(0, f'{"/".join((os.getcwd()).split("/")[:-1])}/classifier')
from surgery_data import SurgeryData
from utils import SURGERY_GROUPS
import numpy as np
import argparse
import random
from link_utils import set_seed, read_surgeries,build_Graph, create_dataset
from link_utils import Net
import networkx as nx
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/shared-data/scalpel/kristina/data/detection/usage/by video')
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='both')

    parser.add_argument('--seed', default=42, type=int)

    ## Embedding Arguments
    parser.add_argument('--embedding_model', choices=['Node2Vec','DeepWalk'], default='DeepWalk')
    parser.add_argument('--embedding_dim', default=256, type=int)
    parser.add_argument('--embedding_walk_length', default=80, type=int)
    parser.add_argument('--embedding_walk_number', default=20, type=int)
    parser.add_argument('--embedding_window_size', default=5, type=int)
    parser.add_argument('--embedding_epochs', default=10, type=int)
    parser.add_argument('--embedding_lr', default=0.05, type=float)
    parser.add_argument('--embedding_min_count', default=1, type=int)

    parser.add_argument('--first_conv_f', default=1024, type=float)
    parser.add_argument('--num_layers', default=5, type=int)

    # parser.add_argument('--embedding_walks_per_node', default=20, type=int)
    # parser.add_argument('--embedding_num_negative_samples', default=10, type=int)
    # parser.add_argument('--embedding_p', default=200, type=int)
    # parser.add_argument('--embedding_q', default=1, type=int)
    # parser.add_argument('--embedding_sparse', default=True, type=bool)

    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.00876569212062032, type=float)

    args = parser.parse_args()
    assert args.first_conv_f/2**args.num_layers>=1
    return args

class Train_next_step_pred():
    def __init__(self,  model:Net,optimizer:torch.optim):
        self.model = model
        self.optimizer=optimizer


    def train(self, obv_surgs:list, cur_surgs_labels:torch.tensor,num_sur_frames=5):
        """
        :return:
        """
        G = build_Graph(obv_surgs+[cur_surgs_labels[0:num_sur_frames]])
        data = create_dataset(G,args,load_embeddings=True, train_test_split=False)
        self.model.train()
        self.optimizer.zero_grad()
        last_state=cur_surgs_labels[num_sur_frames-1]
        all_logits = []
        labels = cur_surgs_labels[num_sur_frames:]
        acc=0
        for state in labels:
            z = self.model.encode(x=data.x, edge_index=data.edge_index)
            prediction, logits=self.model.predict(z,last_state)
            all_logits.append(logits)
            acc+= prediction==state
            if G.has_edge(last_state,state):
                G[last_state][state]['weight']+=1
            else:
                G.add_weighted_edges_from([last_state,state,1])
            last_state=state
            data = create_dataset(G, args, load_embeddings=True, train_test_split=False)
        loss = F.cross_entropy(torch.stack(all_logits), labels)
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
    set_seed(args.seed)
    surgeries_train_data = read_surgeries(args.videos_path,'train')
    surgeries_val_data= read_surgeries(args.videos_path,'val')
    surgeries_test_data= read_surgeries(args.videos_path,'val')
    model = Net(dataset=None,args=args)
    model.load_state_dict(torch.load(f'/home/liory/GCN/MyDigitalNurse/GCN/IPG_both'))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    trainer=Train_next_step_pred(model,optimizer)
    for i,surg in enumerate(surgeries_train_data):
        trainer.train(surgeries_train_data[:i]+surgeries_train_data[i+1:],surg)





