##SETUP
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
import torch_geometric.data
from link_utils import set_seed, read_surgeries,build_Graph, create_dataset
from link_utils import Net
import argparse
import networkx as nx
import copy
import wandb

## SAGE ?

# Videos path = '/data/shared-data/scalpel/kristina/data/detection/usage/by video'



def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/shared-data/scalpel/kristina/data/detection/usage/by video')
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='surgeon')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)

    parser.add_argument('--seed', default=42, type=int)

    ## Embedding Arguments
    parser.add_argument('--embedding_model', choices=['Cora','DeepWalk'], default='DeepWalk')
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

    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.00876569212062032, type=float)

    args = parser.parse_args()
    assert args.first_conv_f/2**args.num_layers>=1
    return args

class Train_link_prediction():
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
        self.optimizer.zero_grad()
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
    set_seed(args.seed)
    # wandb.init(project="MDN_GCN", mode=args.wandb_mode)  # logging to wandb
    # wandb.config.update(args)
    surgeries_data = read_surgeries(args.videos_path,'train')
    G, state_to_node,node_to_state = build_Graph(surgeries_data[args.graph_worker],graph_worker=args.graph_worker)
    data = create_dataset(G,embedding_model=args.embedding_model, args=args,load_embeddings=False, train_test_split=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(data,args).to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    trainer= Train_link_prediction(G, model, optimizer,data)
    best_val_perf = test_perf = best_epoch = 0
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
            best_epoch=epoch
            best_model = copy.deepcopy(trainer.model)
        val_acc.append(best_val_perf)
        test_acc.append(test_perf)
        train_results = {"epoch": epoch, "train loss": train_loss,
                         "Val Loss": val_loss, 'Val Acc': val_perf, 'Test Acc':test_acc}
        # wandb.log(train_results)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'
        print(log.format(epoch, train_loss, val_loss, val_perf, tmp_test_perf))
    print(f'Best Results \n Best Epoch:{best_epoch} Best Val Perf: {best_val_perf}, Best test for best val : {test_perf}')
    z = trainer.model.encode()
    final_edge_index = trainer.model.decode_all(z)
    torch.save(best_model.state_dict(), f'/home/liory/GCN/MyDigitalNurse/GCN/IPG_{args.graph_worker}')