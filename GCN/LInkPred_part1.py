##SETUP
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
import torch_geometric.data
# from link_utils import read_surgeries,build_Graph, create_dataset,build_Graph_from_dfs
# from link_utils import Net, Net_with_embedding,set_seed
from link_utils_new import build_Graph_from_dfs_new, Net, Net_with_embedding,set_seed
import argparse
import networkx as nx
import copy
import wandb
import os

## SAGE ?

# Videos path = '/data/shared-data/scalpel/kristina/data/detection/usage/by video'



def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/shared-data/scalpel/kristina/data/detection/usage/by video')
    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='surgeon')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--Project', choices=['MDN-IPG-BORIS-NEW', 'MDN-IPG-BORIS-NEW-NoSelfLoops'], default='MDN-IPG-BORIS-NEW-NoSelfLoops', type=str)

    parser.add_argument('--seed', default=42, type=int)
    ## Embedding Arguments
    parser.add_argument('--embedding_model', choices=['Cora','DeepWalk','torch'], default='Cora')
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--embedding_walk_length', default=30, type=int)
    parser.add_argument('--embedding_walk_number', default=70, type=int)
    parser.add_argument('--embedding_window_size', default=6, type=int)
    parser.add_argument('--embedding_epochs', default=60 , type=int)
    parser.add_argument('--embedding_lr', default=0.019927752908576343, type=float)
    parser.add_argument('--embedding_min_count', default=1, type=int)

    parser.add_argument('--first_conv_f', default=1024 , type=int)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--activation', choices=['relu','lrelu','tanh','sigmoid'], default='tanh')

    # parser.add_argument('--embedding_walks_per_node', default=20, type=int)
    # parser.add_argument('--embedding_num_negative_samples', default=10, type=int)
    # parser.add_argument('--embedding_p', default=200, type=int)
    # parser.add_argument('--embedding_q', default=1, type=int)
    # parser.add_argument('--embedding_sparse', default=True, type=bool)

    parser.add_argument('--num_epochs', default=23, type=int)
    parser.add_argument('--lr', default=0.030827617049247406, type=float)

    args = parser.parse_args()
    assert args.first_conv_f/2**args.num_layers>=1
    return args

class Train_link_prediction():
    def __init__(self, model:Net,optimizer:torch.optim, dataset:torch_geometric.data.data.Data, all_edges):
        self.model = model
        # self.G=G
        self.data=dataset
        self.optimizer=optimizer
        self.all_edges = all_edges

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
            edge_index=self.all_edges, num_nodes=data.num_nodes,
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
        accs = {'val acc':0, 'test acc': 0}
        roc_auc = {'val':0, 'test':0}
        loss = {'val':0, 'test':0}
        for prefix in ["val", "test"]:
            pos_edge_index = self.data[f'{prefix}_pos_edge_index']
            neg_edge_index = self.data[f'{prefix}_neg_edge_index']
            z = self.model.encode()
            link_logits = self.model.decode(z, pos_edge_index, neg_edge_index)
            link_labels = self.get_link_labels(pos_edge_index,neg_edge_index)
            loss[prefix] = F.binary_cross_entropy_with_logits(link_logits, link_labels).item()
            link_probs = link_logits.sigmoid()
            roc_auc[prefix]=roc_auc_score(link_labels.cpu(), link_probs.cpu())
            # perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
            accs[f'{prefix} acc'] =(((link_probs>0.5)==link_labels).sum() / len(link_labels)).item()
        # z = self.model.encode()
        # link_logits = self.model.decode(z, self.data.val_pos_edge_index, self.data.val_neg_edge_index)
        # link_labels = self.get_link_labels(self.data.val_pos_edge_index, self.data.val_neg_edge_index)
        # loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        # perfs.append(loss.item())
        return roc_auc['val'], roc_auc['test'],loss['val'],loss['test'], accs['val acc'], accs['test acc']


if __name__ == "__main__":
    args = parsing()
    if args.embedding_model == 'Cora':
        args.embedding_dim = 1433
    set_seed(args.seed)
    wandb.init(project=f'{args.Project}_{args.graph_worker}', entity="surgical_data_science", mode=args.wandb_mode)  # logging to wandb
    wandb.config.update(args)
    # surgeries_data = read_surgeries(args.videos_path,'train')
    # G, state_to_node,node_to_state = build_Graph(surgeries_data[args.graph_worker],graph_worker=args.graph_worker,plot=False)
    # data = create_dataset(G,embedding_model=args.embedding_model, args=args,load_embeddings=False, train_test_split=True)
    data, full_data_edges = build_Graph_from_dfs_new(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net_with_embedding(data,args).to(device) if args.embedding_model=='torch' else Net(data,args).to(device)
    print(model)
    data = data.to(device)
    full_data_edges = full_data_edges.to(device)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001) ## all no sel loops
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr) ## all no sel loops

    trainer= Train_link_prediction( model, optimizer,data, full_data_edges)
    best_val_perf = test_perf = best_epoch =best_val_thresh = best_test_threst = 0
    train_losses, val_losses = [], []
    val_acc, test_acc = [], []
    for epoch in range(1, args.num_epochs):
        train_loss = trainer.train()
        train_losses.append(train_loss)
        val_perf, tmp_test_perf, val_loss, test_loss, val_acc_thresh, test_acc_thresh = trainer.test()
        val_losses.append(val_loss)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
            best_epoch=epoch
            best_val_thresh = val_acc_thresh
            best_test_threst = test_acc_thresh
            torch.save(trainer.model.state_dict(), os.path.join(wandb.run.dir, f'IPG_{args.graph_worker}'))
        val_acc.append(best_val_perf)
        test_acc.append(test_perf)
        train_results = {"epoch": epoch, "train loss": train_loss,
                         "Val Loss": val_loss, 'Val Acc': val_perf, 'Test Acc':tmp_test_perf, 'Test Loss':test_loss,
                         'val acc thresh':val_acc_thresh, 'test acc thresh':test_acc_thresh}
        wandb.log(train_results)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f},Test Loss {:.4f},' \
              'val acc thresh:{:.4f},test acc thresh:{:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'
        print(log.format(epoch, train_loss, val_loss,test_loss,val_acc_thresh,test_acc_thresh, val_perf, tmp_test_perf))
    print(f'Best Results \n Best Epoch:{best_epoch} Best Val Perf: {best_val_perf}, Best test for best val :{test_perf}')
    wandb.log({"best epoch": best_epoch, "best val acc": best_val_perf,
                         "best test": test_perf, 'best val thresh':val_acc_thresh,'best test thresh':best_test_threst})
    z = trainer.model.encode()
    final_edge_index = trainer.model.decode_all(z)