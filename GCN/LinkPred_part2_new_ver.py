import sys
import torch
import os
sys.path.insert(0, f'{"/".join((os.getcwd()).split("/")[:-1])}/classifier')
from surgery_data import SurgeryData
from utils import SURGERY_GROUPS
import numpy as np
import argparse
import random
from link_utils_new import build_Graph_from_dfs_new, Net, Net_with_embedding,set_seed,create_dataset,datasets_for_part_2, find_edge_idx
import networkx as nx
import torch.nn.functional as F
import wandb
import copy
import time
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

MODELS ={
"5nmt5mjz":{'activation': 'sigmoid', 'embedding_dim': 512, 'embedding_model': 'DeepWalk', 'first_conv_f': 128, 'num_layers': 3, 'dir': 'run-20221110_103619-5nmt5mjz', 'p1_model_type': 'MDN-IPG-BORIS-NoSelfLoops'},
"pslhay65":{'activation': 'tanh', 'embedding_dim': 512, 'embedding_model': 'DeepWalk', 'first_conv_f': 256, 'num_layers': 7, 'dir': 'run-20221110_091158-pslhay65', 'p1_model_type': 'MDN-IPG-BORIS-NoSelfLoops'},
"fihmrns0":{'activation': 'tanh', 'embedding_dim': 256, 'embedding_model': 'DeepWalk', 'first_conv_f': 512, 'num_layers': 3, 'dir': 'run-20221110_075818-fihmrns0', 'p1_model_type': 'MDN-IPG-BORIS-NoSelfLoops'},
"elznjvsn":{'activation': 'tanh', 'embedding_dim': 256, 'embedding_model': 'DeepWalk', 'first_conv_f': 256, 'num_layers': 3, 'dir': 'run-20221110_034915-elznjvsn', 'p1_model_type': 'MDN-IPG-BORIS-NoSelfLoops'},
"9xvpfxgs":{'activation': 'lrelu', 'embedding_dim': 1433, 'embedding_model': 'Cora', 'first_conv_f': 128, 'num_layers': 3, 'dir': 'run-20221110_052015-9xvpfxgs', 'p1_model_type': 'MDN-IPG-BORIS-NoSelfLoops'},
"d1ngf5p9":{'activation': 'lrelu', 'embedding_dim': 256, 'embedding_model': 'DeepWalk', 'first_conv_f': 256, 'num_layers': 5, 'dir': 'run-20221110_081727-d1ngf5p9', 'p1_model_type': 'MDN-IPG-BORIS-NoSelfLoops'}
    }

FINAL_HP = {
    'd1ngf5p9':{'loss_calc':'surgery','lr':0.02195912366641873},#charmed-sweep-380
    '5nmt5mjz':{'loss_calc':'surgery','lr':0.049403153223501625},#treasured-sweep-107
    'elznjvsn': {'loss_calc': 'surgery', 'lr': 0.0639265835039942}  # lilac-sweep-481
              }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--Project', default='MDN-NextStep-BORIS_new_1011', type=str)

    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='surgeon')
    parser.add_argument('--p1_run_name', default='5nmt5mjz', type=str)
    parser.add_argument('--p1_model_type', choices=['MDN-IPG-BORIS', 'MDN-IPG-BORIS-NoSelfLoops'], default='MDN-IPG-BORIS')
    parser.add_argument('--train_type', choices=['ForcedNoSelfLoops', 'NoSelfLoops','FullGraph'], default='ForcedNoSelfLoops')
    parser.add_argument('--partial_start', default=1, type=int)

    parser.add_argument('--loss_calc', choices=['surgery', 'node', 'batch'], default='surgery')
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--files_dir', type=str, default='/data/home/liory/MyDigitalNurse/GCN/wandb/run-20221110_103619-5nmt5mjz/files/')
    parser.add_argument('--model', default='IPG_both' )
    parser.add_argument('--first_conv_f', default=256 , type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--activation', choices=['relu','lrelu','tanh','sigmoid'], default='tanh')
    parser.add_argument('--load_embeddings', default='train_embeddings_both.pkl' )
    parser.add_argument('--embedding_model', choices=['Cora','DeepWalk','onehot'], default='DeepWalk')
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--early_stop', default=10, type=float)
    parser.add_argument('--with_scheduler', default=True, type=bool)
    parser.add_argument('--lr', default=0.12381015946468615, type=float)
    args = parser.parse_args()
    args.load_embeddings = f'train_embeddings_{args.graph_worker}.pkl'
    args.model = f'IPG_{args.graph_worker}'
    assert args.first_conv_f/2**args.num_layers>=1
    return args



class Train_next_step_pred():
    def __init__(self,  model:Net,optimizer:torch.optim,train_data,args,unk_node=18, with_scheduler=False):
        self.model = model
        self.optimizer=  torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=2e-10, patience=4, threshold=0.01,
                                      threshold_mode='abs', verbose=True)
        self.args=args
        self.loss_calc = args.loss_calc
        # self.full_training_dataset = full_training_dataset
        self.train_data = train_data
        self.test_data = test_data
        # self.labels = {'val': val_labels, 'test': test_labels}
        # self.videos_path = args.videos_path
        self.embedding_dir = os.path.join(args.files_dir, args.load_embeddings)
        self.remove = True if 'Forced' in args.train_type else False
        self.best_loss = np.float('inf')
        self.num_epochs_not_improved = 0
        self.sched_use = 0
        self.unk_node=unk_node
        self.with_scheduler = with_scheduler

    def find_node_idx_in_data(self,ids,node):
        find_idx = (ids == node).nonzero()
        if find_idx.shape[0] != 0:
            node_idx = find_idx.item()
        else:
            node_idx =(ids == self.unk_node).nonzero().item()
        return node_idx


    def train_surgery(self, data,labels):
        """
        :return:
        """
        last_node=labels[0]
        last_node_idx = self.find_node_idx_in_data(data.id,last_node)
        acc=[]
        node_labels= []
        node_labels_idxs = []
        losses = []
        loss = 0
        self.model.train()
        self.optimizer.zero_grad()
        for i,node in enumerate(labels[1:]):
            logits = self.model.predict(last_step=last_node_idx,x=data.x, edge_index=data.edge_index, weight=data.weight,
                                                    removeselfloops=self.remove)
            node_idx = self.find_node_idx_in_data(data.id,node)
            node_labels.append(node)
            node_labels_idxs.append(node_idx)
            cur_edge_index = find_edge_idx(data.edge_index, last_node_idx, node_idx)
            if cur_edge_index>-1:
                data.weight[cur_edge_index]+=1
            else:
                data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node_idx], [node_idx]])], dim=1)
                data.weight = torch.cat([data.weight, torch.tensor([1])], dim=0)
            last_node=node
            last_node_idx = node_idx
            loss += F.cross_entropy(logits, torch.tensor(last_node_idx))
            prediction = torch.argmax(logits)
            acc.append(prediction==last_node_idx)
        loss.backward()
        self.optimizer.step()
        mean_loss = loss.item()/len(node_labels)
        # self.scheduler.step(mean_loss)
        return mean_loss, np.mean(acc)

    def train_node(self, data,labels):
        """
        :return:
        """
        """
        :return:
        """
        last_node=labels[0]
        last_node_idx = self.find_node_idx_in_data(data.id,last_node)
        acc=[]
        node_labels= []
        node_labels_idxs = []
        losses = []
        loss = 0
        for i,node in enumerate(labels[1:]):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model.predict(last_step=last_node_idx,x=data.x, edge_index=data.edge_index, weight=data.weight,
                                                    removeselfloops=self.remove)
            node_labels.append(node)
            node_idx = self.find_node_idx_in_data(data.id,node)
            node_labels_idxs.append(node_idx)
            cur_edge_index = find_edge_idx(data.edge_index, last_node_idx, node_idx)
            if cur_edge_index>-1:
                data.weight[cur_edge_index]+=1
            else:
                data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node_idx], [node_idx]])], dim=1)
                data.weight = torch.cat([data.weight, torch.tensor([1])], dim=0)
            last_node=node
            last_node_idx = node_idx
            loss = F.cross_entropy(logits, torch.tensor(node_idx))
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            prediction = torch.argmax(logits)
            acc.append(prediction==last_node_idx)
        mean_loss = np.mean(losses)
        # self.scheduler.step(mean_loss)
        return mean_loss, np.mean(acc)
    #
    # @torch.no_grad()
    def test(self):
        self.model.eval()
        all_losses = []
        all_accs = []
        for i, data_dict in self.test_data.items():
            data = data_dict['data']
            labels = data_dict['labels']
            last_node = labels[0]
            last_node_idx = self.find_node_idx_in_data(data.id, last_node)
            sur_acc = []
            node_labels = []
            node_labels_idxs = []
            sur_losses = []
            for i, node in enumerate(labels[1:]):
                node_idx = self.find_node_idx_in_data(data.id, node)
                logits = self.model.predict(last_step=last_node_idx, x=data.x, edge_index=data.edge_index,
                                            weight=data.weight,
                                            removeselfloops=self.remove)
                loss = F.cross_entropy(logits, torch.tensor(node_idx))
                prediction = torch.argmax(logits)
                sur_acc.append(prediction == node_idx)
                last_node_idx = prediction
                sur_losses.append(loss.item())
                cur_edge_index = find_edge_idx(data.edge_index, last_node_idx, prediction)
                if cur_edge_index > -1:
                    data.weight[cur_edge_index] += 1
                else:
                    data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node_idx], [prediction]])], dim=1)
                    data.weight = torch.cat([data.weight, torch.tensor([1])], dim=0)
            mean_loss = np.mean(sur_losses)
            all_losses.append(mean_loss)
            all_accs.append(np.mean(sur_acc))
        return np.mean(all_losses),np.mean(all_accs)

    #

    def predictions(self,samples):
        start = time.time()
        epoch_losses=[]
        epoch_accs = []
        for i in samples:
            # dataset,labels = self.read_train_data(i)
            dataset = self.train_data[i]['data']
            labels = self.train_data[i]['labels']
            # weights = self.train_data[i]['labels_weight']
            print('Learning Surgery', i)
            if self.args.loss_calc=='node':
                mean_surg_loss, mean_surg_acc = self.train_node(dataset,labels)
            elif self.args.loss_calc=='batch':
                mean_surg_loss, mean_surg_acc = self.train_frames_batch(dataset,labels)
            else:
                mean_surg_loss, mean_surg_acc = self.train_surgery(dataset,labels)
            print('Time for surgery:', time.time()-start)
            start = time.time()
            print('Current Surgery Loss: ', mean_surg_loss, 'Mean Accuracy: ', mean_surg_acc)
            epoch_losses.append(mean_surg_loss)
            epoch_accs.append(mean_surg_acc)
        mean_loss = np.mean(epoch_losses)
        if self.with_scheduler:
            self.scheduler.step(mean_loss)
        return mean_loss,np.mean(epoch_accs)


def edit_args(args):
    model_dic = MODELS[args.p1_run_name]
    args.activation = model_dic['activation']
    args.embedding_dim = model_dic['embedding_dim']
    args.embedding_model = model_dic['embedding_model']
    args.first_conv_f = model_dic['first_conv_f']
    args.num_layers = model_dic['num_layers']
    args.files_dir = f"/data/home/liory/MyDigitalNurse/GCN/wandb/{model_dic['dir']}/files/"
    args.p1_model_type = model_dic['p1_model_type']
    args.loss_calc = FINAL_HP[args.p1_run_name]['loss_calc']
    args.lr = FINAL_HP[args.p1_run_name]['lr']
    return args

if __name__ == "__main__":
    args = parsing()
    set_seed(args.seed)
    args = edit_args(args)
    wandb.init(project=args.Project, entity="surgical_data_science", mode=args.wandb_mode)  # logging to wandb
    wandb.config.update(args)
    train_data,test_data, unk_node = datasets_for_part_2(args)
    train_samples =list(train_data.keys())
    model = Net(dataset=None,args=args)
    model.load_state_dict(torch.load(os.path.join(args.files_dir,args.model)))
    trainer = Train_next_step_pred(model,optimizer=None,args=args,train_data=train_data,unk_node=unk_node,
                                   with_scheduler=args.with_scheduler)
    best_test_acc  = best_train_acc  = best_test_loss_epoch=best_test_acc_epoch= 0
    best_train_loss = best_test_loss = np.float('inf')
    steps_no_improve = 0
    set_seed()
    for epoch in range(args.num_epochs):
        random.shuffle(train_samples)
        train_epoch_losses,train_epoch_accs=trainer.predictions(train_samples)
        print(f'Training Epoch {epoch+1} \n Mean Loss: {train_epoch_losses}, Mean Accuracy: {train_epoch_accs}')
        if train_epoch_losses<best_train_loss:
            steps_no_improve=0
            best_train_loss = train_epoch_losses
            best_train_loss_epoch=epoch
        else:
            steps_no_improve+=1
        if best_train_acc<train_epoch_accs:
            best_train_acc = train_epoch_accs
            best_train_acc_epoch = epoch
        test_loss,test_acc=trainer.test()
        print(f'Test Epoch {epoch+1} \n Mean Loss: {test_loss}, Mean Accuracy: {test_acc}')
        if test_loss<best_test_loss:
            best_test_loss = test_loss
            best_test_loss_epoch=epoch
        if best_test_acc<test_acc:
            best_test_acc = test_acc
            best_test_acc_epoch = epoch
        test_loss,test_acc=trainer.test()
        train_results = {"epoch": epoch, "train loss": train_epoch_losses, "Train Acc": train_epoch_accs,
                         "Test Loss": test_loss,'Test Acc':test_acc}
        wandb.log(train_results)
        if steps_no_improve>=args.early_stop:
            break
    best_results = {"best_train_loss": best_train_loss, "best_train_loss_epoch": best_train_loss_epoch,
                    'best_train_acc': best_train_acc,"best_train_acc_epoch": best_train_acc_epoch,
                    'best_test_loss': best_test_loss, 'best_test_loss_epoch':best_test_loss_epoch,
                    'best_test_acc':best_test_acc, "best_test_acc_epoch":best_test_acc_epoch}
    wandb.log(best_results)
    # torch.save(best_model.state_dict(), os.path.join(wandb.run.dir, f'Part2_{args.graph_worker}'))
    print(f'best_train_loss:{best_train_loss} on epoch {best_train_loss_epoch}')
    print(f'best_train_acc:{best_train_acc} on epoch {best_train_acc_epoch}')
    print(f'best_test_loss:{best_test_loss} on epoch {best_test_loss_epoch}')
    print(f'best_test_acc:{best_test_acc} on epoch {best_test_acc_epoch}')
