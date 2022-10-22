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
from link_utils import Net, Net_with_embedding
import networkx as nx
import torch.nn.functional as F
import wandb
import copy

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/shared-data/scalpel/kristina/data/detection/usage/by video')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)

    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='both')
    parser.add_argument('--loss_calc', choices=['surgery', 'node'], default='surgery')

    parser.add_argument('--files_dir', default='/data/home/liory/GCN/MyDigitalNurse/GCN/wandb/run-20221018_120521-32a3a46u/files')
    parser.add_argument('--model', default='IPG_both' )
    parser.add_argument('--first_conv_f', default=4096 , type=int)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--activation', choices=['relu','lrelu','tanh','sigmoid'], default='lrelu')
    parser.add_argument('--load_embeddings', default='train_embeddings_both.pkl' )
    parser.add_argument('--embedding_model', choices=['Cora','DeepWalk','torch'], default='DeepWalk')
    parser.add_argument('--embedding_dim', default=512, type=int)

    parser.add_argument('--num_epochs', default=13, type=int)
    parser.add_argument('--lr', default=0.0678735678614226, type=float)

    args = parser.parse_args()
    assert args.first_conv_f/2**args.num_layers>=1
    return args


class Train_next_step_pred():
    def __init__(self,  model:Net,optimizer:torch.optim,args, embedding_dir=None):
        self.model = model
        self.optimizer=optimizer
        self.embedding_dir= embedding_dir
        self.args=args
        self.loss_calc = args.loss_calc


    def train(self, obv_surgs:list, cur_surgs_labels:torch.tensor, num_sur_frames=5):
        """
        :return:
        """
        G, state_to_node,node_to_state = build_Graph(obv_surgs+[cur_surgs_labels[0:num_sur_frames]])
        data = create_dataset(G=G,embedding_model=args.embedding_model,args=self.args,load_embeddings=self.embedding_dir, train_test_split=False)
        self.model.train()
        last_node=state_to_node[tuple(x.item() for x in cur_surgs_labels[num_sur_frames-1])]
        all_logits = []
        labels = cur_surgs_labels[num_sur_frames:]
        acc=[]
        node_labels= []
        losses = []
        if self.loss_calc=='surgery':
            self.optimizer.zero_grad()
        for state in labels:
            if self.loss_calc=='node':
                self.optimizer.zero_grad()
            node = state_to_node[tuple(x.item() for x in state)]
            z = self.model.encode(x=data.x, edge_index=data.edge_index)
            prediction, logits=self.model.predict(z,last_node)
            all_logits.append(logits)
            node_labels.append(node)
            acc.append(prediction==node)
            if G.has_edge(last_node,node):
                G[last_node][node]['weight']+=1
            else:
                G.add_edge(last_node,node,weight=1)
            last_node=node
            data = create_dataset(G=G, embedding_model=args.embedding_model, args=self.args,
                                  load_embeddings=self.embedding_dir, train_test_split=False)
            if self.loss_calc=='node':
                loss = F.cross_entropy(logits, torch.tensor(node))
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        if self.loss_calc=='surgery':
            loss = F.cross_entropy(torch.stack(all_logits), torch.tensor(node_labels))
            loss.backward()
            self.optimizer.step()
        final_loss = np.mean(losses) if self.loss_calc=='node' else loss.item()
        return final_loss, np.mean(acc)


    @torch.no_grad()
    def test(self,obv_surgs:list, cur_surgs_labels:torch.tensor,args, num_sur_frames=5):
        self.model.eval()
        G, state_to_node,node_to_state = build_Graph(obv_surgs+[cur_surgs_labels[0:num_sur_frames]])
        data = create_dataset(G=G,embedding_model=args.embedding_model,args=args,load_embeddings=self.embedding_dir, train_test_split=False)
        last_node=state_to_node[tuple(x.item() for x in cur_surgs_labels[num_sur_frames-1])]
        all_logits = []
        labels = cur_surgs_labels[num_sur_frames:]
        acc=[]
        node_labels= []
        losses = []
        for state in labels:
            node = state_to_node[tuple(x.item() for x in state)]
            z = self.model.encode(x=data.x, edge_index=data.edge_index)
            prediction, logits=self.model.predict(z,last_node)
            all_logits.append(logits)
            node_labels.append(node)
            acc.append(prediction==node)
            if G.has_edge(last_node,prediction):
                G[last_node][prediction]['weight']+=1
            else:
                G.add_edge(last_node,prediction,weight=1)
            last_node=prediction
            data = create_dataset(G=G, embedding_model=args.embedding_model, args=args,
                                  load_embeddings=self.embedding_dir, train_test_split=False)
            loss = F.cross_entropy(logits, torch.tensor(node))
            losses.append(loss.item())
        return np.mean(losses), np.mean(acc)


def predictions(data):
    for i, surg in enumerate(data):
        epoch_losses=[]
        epoch_accs = []
        print('Learning Surgery', i)
        mean_surg_loss, mean_surg_acc = trainer.train(
            obv_surgs=data[:i] + data[i + 1:],
            cur_surgs_labels=surg)
        print('Current Surgery Loss: ', mean_surg_loss, 'Mean Accuracy: ', mean_surg_acc)
        epoch_losses.append(mean_surg_loss)
        epoch_accs.append(mean_surg_acc)
    return np.mean(epoch_losses),np.mean(epoch_accs)


if __name__ == "__main__":
    args = parsing()
    set_seed(args.seed)
    wandb.init(project="MDN-IPG-P2", entity="surgical_data_science", mode=args.wandb_mode)  # logging to wandb
    wandb.config.update(args)
    surgeries_train_data = read_surgeries(args.videos_path,'train')
    surgeries_val_data= read_surgeries(args.videos_path,'val')
    surgeries_test_data= read_surgeries(args.videos_path,'test')
    model = Net_with_embedding(dataset=None,args=args) if args.embedding_model=='torch' else Net(dataset=None,args=args)
    model.load_state_dict(torch.load(os.path.join(args.files_dir,args.model)))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    trainer = Train_next_step_pred(model,optimizer,args=args, embedding_dir=os.path.join(args.files_dir,args.load_embeddings))
    train_data = surgeries_train_data[args.graph_worker]
    val_data = surgeries_val_data[args.graph_worker]
    test_data = surgeries_test_data[args.graph_worker]
    best_val_acc = best_test_acc = best_epoch = best_val_loss = best_test_loss = 0
    for epoch in range(args.num_epochs):
        random.shuffle(train_data)
        train_epoch_losses,train_epoch_accs=predictions(train_data)
        print(f'Training Epoch {epoch+1} \n Mean Loss: {train_epoch_losses}, Mean Accuracy: {train_epoch_accs}')
        val_epoch_losses,val_epoch_accs=predictions(val_data,args)
        print(f'Validation Epoch {epoch+1} \n Mean Loss: {val_epoch_losses}, Mean Accuracy: {val_epoch_accs}')
        test_epoch_losses,test_epoch_accs=predictions(test_data,args)
        print(f'Validation Epoch {epoch+1} \n Mean Loss: {test_epoch_losses}, Mean Accuracy: {test_epoch_accs}')
        train_results = {"epoch": epoch, "train loss": train_epoch_losses, "Train Acc": train_epoch_accs,
                         "Val Loss": val_epoch_losses, 'Val Acc': val_epoch_accs,"Test Loss": test_epoch_losses,
                         'Test Acc':test_epoch_accs}
        wandb.log(train_results)
        if val_epoch_accs>best_val_acc:
            best_val_acc=val_epoch_accs
            best_test_acc = test_epoch_accs
            best_val_loss = val_epoch_losses
            best_test_loss = test_epoch_losses
            best_epoch = epoch
            best_model = copy.deepcopy(trainer.model)
    best_results = {"best_epoch": best_epoch, "Best Val Loss": best_val_loss, 'Best Val Acc': best_val_acc,
                    "Best Test Loss": best_test_loss, 'Best Test Acc': best_test_acc}
    wandb.log(best_results)
    torch.save(best_model.state_dict(), os.path.join(wandb.run.dir, f'Part2_{args.graph_worker}'))









