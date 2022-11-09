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
'6gwrxooj':{'activation': 'tanh', 'embedding_dim': 128, 'embedding_model': 'DeepWalk', 'first_conv_f': 1024,
            'num_layers': 5, 'dir': 'run-20221104_130406-6gwrxooj', 'p1_model_type': 'MDN-IPG-BORIS-NoSelfLoops'},
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
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='disabled', type=str)
    parser.add_argument('--Project', default='MDN-NextStep-BORIS_new', type=str)

    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='surgeon')
    parser.add_argument('--p1_run_name', default='6gwrxooj', type=str)
    parser.add_argument('--p1_model_type', choices=['MDN-IPG-BORIS', 'MDN-IPG-BORIS-NoSelfLoops'], default='MDN-IPG-BORIS')
    parser.add_argument('--train_type', choices=['ForcedNoSelfLoops', 'NoSelfLoops','FullGraph'], default='ForcedNoSelfLoops')
    parser.add_argument('--partial_start', default=5, type=int)

    parser.add_argument('--loss_calc', choices=['surgery', 'node', 'batch'], default='surgery')
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--files_dir', type=str, default='/data/home/liory/MyDigitalNurse/GCN/wandb/run-20221106_084033-j2t5h5ul/files/')
    parser.add_argument('--model', default='IPG_both' )
    parser.add_argument('--first_conv_f', default=256 , type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--activation', choices=['relu','lrelu','tanh','sigmoid'], default='tanh')
    parser.add_argument('--load_embeddings', default='train_embeddings_both.pkl' )
    parser.add_argument('--embedding_model', choices=['Cora','DeepWalk','onehot'], default='onehot')
    parser.add_argument('--embedding_dim', default=25, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    args = parser.parse_args()
    args.load_embeddings = f'train_embeddings_{args.graph_worker}.pkl'
    args.model = f'IPG_{args.graph_worker}'
    assert args.first_conv_f/2**args.num_layers>=1
    return args



class Train_next_step_pred():
    def __init__(self,  model:Net,optimizer:torch.optim,train_data,full_training_dataset, val_labels,test_labels,args, embedding_dir=None):
        self.model = model
        self.optimizer=  torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.005, patience=5, threshold=0.1,
                                      threshold_mode='abs', verbose=True)
        self.args=args
        self.loss_calc = args.loss_calc
        self.full_training_dataset = full_training_dataset
        self.train_data = train_data
        self.labels = {'val': val_labels, 'test': test_labels}
        self.videos_path = args.videos_path
        self.embedding_dir = os.path.join(args.files_dir, args.load_embeddings)
        self.remove = True if 'Forced' in args.train_type else False
        self.best_loss = np.float('inf')
        self.num_epochs_not_improved = 0
        self.sched_use = 0

    def train_frames_batch(self, data,labels):
        last_node=labels[0]
        acc=[]
        losses = []
        batch_index = range(1,len(labels)-1,self.args.batch_size)
        for batch_start in batch_index:
            self.model.train()
            self.optimizer.zero_grad()
            cur_labels = labels[batch_start:batch_start+self.args.batch_size]
            node_labels = []
            all_logits=[]
            for i,node in enumerate(cur_labels):
                # z = self.model.encode(x=data.x, edge_index=data.edge_index, weight=data.weight)
                prediction, logits=self.model.predict(x=data.x, edge_index=data.edge_index, weight=data.weight,
                                                      last_step=last_node,removeselfloops=self.remove, i=i)
                all_logits.append(logits)
                node_labels.append(node)
                acc.append(prediction==node.item())
                cur_edge_index = find_edge_idx(data.edge_index, last_node, node)
                if cur_edge_index>-1:
                    data.weight[cur_edge_index]+=1
                else:
                    data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node], [node]])], dim=1)
                    data.weight = torch.cat([data.weight, torch.tensor([1])], dim=0)
                last_node=node
            loss = F.cross_entropy(torch.stack(all_logits), torch.tensor(node_labels))
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        final_loss = np.mean(losses)
        return final_loss, np.mean(acc)

    def train_surgery(self, data,labels, labels_weight=None):
        """
        :return:
        """
        last_node=labels[0]
        acc=[]
        node_labels= []
        losses = []
        loss = 0
        self.model.train()
        self.optimizer.zero_grad()
        for i,node in enumerate(labels[1:]):
            logits = self.model.predict(last_step=last_node,x=data.x, edge_index=data.edge_index, weight=data.weight,
                                                    removeselfloops=self.remove)
            node_labels.append(node)
            cur_edge_index = find_edge_idx(data.edge_index, last_node, node)
            if cur_edge_index>-1:
                data.weight[cur_edge_index]+=1
            else:
                data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node], [node]])], dim=1)
                data.weight = torch.cat([data.weight, torch.tensor([1])], dim=0)
            if labels_weight is not None:
                self_edge_index = find_edge_idx(data.edge_index, last_node, last_node)
                self_edge_weight = labels_weight[i-1]
                if self_edge_index > -1:
                    data.weight[self_edge_index] += self_edge_weight.item()
                else:
                    data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node], [node]])], dim=1)
                    data.weight = torch.cat([data.weight, torch.tensor([self_edge_weight])], dim=0)
            last_node=node
            loss += F.cross_entropy(logits, torch.tensor(node))
            prediction = torch.argmax(logits)
            acc.append(prediction==node)
        loss.backward()
        self.optimizer.step()
        mean_loss = loss.item()/len(node_labels)
        self.scheduler.step(mean_loss)
        return mean_loss, np.mean(acc)

    def train_node(self, data,labels):
        """
        :return:
        """
        # labels = list(range(10))
        # embeddings = {i: torch.tensor([0.]*i+[1.]+[0.]*(10-i-1)) for i in labels}
        last_node=labels[0]
        acc=[]
        node_labels= []
        losses = []
        for i,node in enumerate(labels[1:]):
            self.model.train()
            self.optimizer.zero_grad()
            # z = self.model.encode(x=data.x, edge_index=data.edge_index, weight=data.weight)
            # prediction, logits=self.model.predict(z,last_node,removeselfloops=self.remove)
            # z = self.model.encode(x=data.x, edge_index=data.edge_index, weight=data.weight)
            # prediction, logits = self.model.predict(x=data.x, edge_index=data.edge_index, weight=data.weight,
            #                                         last_step=last_node, removeselfloops=self.remove)
            # x = torch.cat([data.x[last_node],torch.tensor([i])])
            logits = self.model.predict(last_step=last_node,x=data.x, edge_index=data.edge_index, weight=data.weight,
                                                    removeselfloops=self.remove)
            node_labels.append(node)
            cur_edge_index = find_edge_idx(data.edge_index, last_node, node)
            if cur_edge_index>-1:
                data.weight[cur_edge_index]+=1
            else:
                data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node], [node]])], dim=1)
                data.weight = torch.cat([data.weight, torch.tensor([1])], dim=0)
            last_node=node
            loss = F.cross_entropy(logits, torch.tensor(node))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss.item())
            prediction = torch.argmax(logits)
            acc.append(prediction==node)
            # acc.append(prediction==node.item())
            losses.append(loss.item())
        final_loss = np.mean(losses)
        return final_loss, np.mean(acc)



    def train(self, data,labels):
        """
        :return:
        """
        self.model.train()
        last_node=labels[0]
        acc=[]
        node_labels= []
        losses = []
        if self.loss_calc=='surgery':
            self.optimizer.zero_grad()
        for node in labels[1:]:
            all_logits = []
            z = self.model.encode(x=data.x, edge_index=data.edge_index, weight=data.weight)
            prediction, logits=self.model.predict(z,last_node,removeselfloops=self.remove)
            all_logits.append(logits)
            node_labels.append(node)
            acc.append(prediction==node.item())
            cur_edge_index = find_edge_idx(data.edge_index, last_node, node)
            if cur_edge_index>-1:
                data.weight[cur_edge_index]+=1
            else:
                data.edge_index = torch.cat([data.edge_index, torch.tensor([[last_node], [node]])], dim=1)
                data.weight = torch.cat([data.weight, torch.tensor([1])], dim=0)
            last_node=node
            if self.loss_calc=='node':
                loss = F.cross_entropy(logits, torch.tensor(node))
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                self.optimizer.zero_grad()
            if self.loss_calc=='batch' and len(all_logits)==args.batch_size:
                print('batch loss')
                loss = F.cross_entropy(logits, torch.tensor(node))
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                self.optimizer.zero_grad()
                all_logits=[]
        if self.loss_calc == 'batch' and len(all_logits)>0:
            loss = F.cross_entropy(logits, torch.tensor(node))
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            self.optimizer.zero_grad()
            all_logits = []
        if self.loss_calc=='surgery':
            loss = F.cross_entropy(torch.stack(all_logits), torch.tensor(node_labels))
            loss.backward()
            self.optimizer.step()
        final_loss = np.mean(losses) if self.loss_calc!='surgery' else loss.item()
        return final_loss, np.mean(acc)


    @torch.no_grad()
    def test_batch(self):
        self.model.eval()
        res = {}
        for k, all_labels in self.labels.items():
            losses=[]
            accs = []
            for sur_labels in all_labels:
                full_training_dataset = copy.deepcopy(self.full_training_dataset)
                batch_index = range(1, len(sur_labels) - 1, self.args.batch_size)
                last_node= sur_labels[0]
                sur_losses = []
                for batch_start in batch_index:
                    cur_labels = sur_labels[batch_start:batch_start + self.args.batch_size]
                    all_logits, node_labels,  = [], []
                    for node in cur_labels:
                        # z = self.model.encode(x=full_training_dataset.x, edge_index=full_training_dataset.edge_index,
                        #                       weight=full_training_dataset.weight)
                        # prediction, logits=self.model.predict(z,last_node,removeselfloops=self.remove)
                        prediction, logits = self.model.predict(x=full_training_dataset.x, edge_index=full_training_dataset.edge_index,
                                                                weight=full_training_dataset.weight,
                                                                last_step=last_node, removeselfloops=self.remove)
                        all_logits.append(logits)
                        node_labels.append(node)
                        accs.append(prediction==node)
                        cur_edge_index = find_edge_idx(full_training_dataset.edge_index, last_node, prediction)
                        if cur_edge_index > -1:
                            full_training_dataset.weight[cur_edge_index] += 1
                        else:
                            full_training_dataset.edge_index =\
                                torch.cat([full_training_dataset.edge_index, torch.tensor([[last_node], [prediction]])], dim=1)
                            full_training_dataset.weight = torch.cat([full_training_dataset.weight, torch.tensor([1])], dim=0)
                        last_node=prediction
                    loss = F.cross_entropy(torch.stack(all_logits), torch.tensor(node_labels))
                    sur_losses.append(loss.item())
                losses.append(np.mean(sur_losses))
            res[f'{k}_loss'] = np.mean(losses)
            res[f'{k}_acc'] = np.mean(accs)
        return res['val_loss'], res['val_acc'], res['test_loss'], res['test_acc']


    @torch.no_grad()
    def test(self):
        self.model.eval()
        res = {}
        for k, all_labels in self.labels.items():
            losses=[]
            accs = []
            gt_accs = []
            for sur_labels in all_labels:
                full_training_dataset = copy.deepcopy(self.full_training_dataset)
                last_node= sur_labels[0]
                all_logits, node_labels, sur_losses = [], [], []
                labels = sur_labels[0:]
                for node in labels:
                    # z = self.model.encode(x=full_training_dataset.x, edge_index=full_training_dataset.edge_index,
                    #                       weight=full_training_dataset.weight)
                    # prediction, logits=self.model.predict(z,last_node,removeselfloops=self.remove)
                    prediction, logits = self.model.predict(x=full_training_dataset.x,
                                                            edge_index=full_training_dataset.edge_index,
                                                            weight=full_training_dataset.weight,
                                                            last_step=last_node, removeselfloops=self.remove)
                    all_logits.append(logits)
                    node_labels.append(node)
                    accs.append(prediction==node)
                    gt_accs.append(node==node)
                    cur_edge_index = find_edge_idx(full_training_dataset.edge_index, last_node, prediction)
                    if cur_edge_index > -1:
                        full_training_dataset.weight[cur_edge_index] += 1
                    else:
                        full_training_dataset.edge_index =\
                            torch.cat([full_training_dataset.edge_index, torch.tensor([[last_node], [prediction]])], dim=1)
                        full_training_dataset.weight = torch.cat([full_training_dataset.weight, torch.tensor([1])], dim=0)
                    last_node=prediction
                    if self.loss_calc == 'node':
                        loss = F.cross_entropy(logits, torch.tensor(node))
                        sur_losses.append(loss.item())
                if self.loss_calc == 'surgery':
                    loss = F.cross_entropy(torch.stack(all_logits), torch.tensor(node_labels))
                    losses.append(loss.item())
                else:
                    losses.append(np.mean(sur_losses))
            res[f'{k}_loss'] = np.mean(losses)
            res[f'{k}_acc'] = np.mean(accs)
            res[f'{k}_gt_acc'] = np.mean(gt_accs)
        return res['val_loss'], res['val_acc'], res['test_loss'], res['test_acc'],res['val_gt_acc'], res['test_gt_acc']


    def read_train_data(self,i):
        data_path = os.path.join(self.videos_path, f'{i}')
        Graph = nx.read_gpickle(os.path.join(data_path, "Graph.gpickle"))
        labels = torch.load(os.path.join(data_path, "Labels"))
        data = create_dataset(G=Graph, args=args, load_embeddings=self.embedding_dir)
        return data, labels


    def predictions(self,samples):
        start = time.time()
        for i in samples:
            # dataset,labels = self.read_train_data(i)
            dataset = self.train_data[i]['data']
            labels = self.train_data[i]['labels']
            weights = self.train_data[i]['labels_weight']
            epoch_losses=[]
            epoch_accs = []
            print('Learning Surgery', i)
            if self.args.loss_calc=='node':
                mean_surg_loss, mean_surg_acc = self.train_node(dataset,labels, weights)
            elif self.args.loss_calc=='batch':
                mean_surg_loss, mean_surg_acc = self.train_frames_batch(dataset,labels,weights)
            else:
                mean_surg_loss, mean_surg_acc = self.train_surgery(dataset,labels,weights)
            print('Time for surgery:', time.time()-start)
            start = time.time()
            print('Current Surgery Loss: ', mean_surg_loss, 'Mean Accuracy: ', mean_surg_acc)
            epoch_losses.append(mean_surg_loss)
            epoch_accs.append(mean_surg_acc)
        return np.mean(epoch_losses),np.mean(epoch_accs)

def edit_args(args):
    model_dic = MODELS[args.p1_run_name]
    args.activation = model_dic['activation']
    args.embedding_dim = model_dic['embedding_dim']
    args.embedding_model = model_dic['embedding_model']
    args.first_conv_f = model_dic['first_conv_f']
    args.num_layers = model_dic['num_layers']
    args.files_dir = f"/data/home/liory/MyDigitalNurse/GCN/wandb/{model_dic['dir']}/files/"
    args.p1_model_type = model_dic['p1_model_type']
    return args

if __name__ == "__main__":
    args = parsing()
    set_seed(args.seed)
    if 'NoSelfLoops' in args.p1_model_type:
        args.videos_path = f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_NoSelfLoops_{args.graph_worker}'
        G_path =  f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_NoSelfLoops_{args.graph_worker}'
        L_path =  f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_NoSelfLoops_{args.graph_worker}'
    else:
        args.videos_path = f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_{args.graph_worker}'
        G_path=f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_{args.graph_worker}_new'
        L_path =  f'/data/home/liory/MyDigitalNurse/GCN/Training_datasets_part2_{args.graph_worker}_new'
    args = edit_args(args)
    # wandb.init(project=args.Project, entity="surgical_data_science", mode=args.wandb_mode)  # logging to wandb
    # wandb.config.update(args)
    train_data = datasets_for_part_2(args)
    # full_dataset, val_labels, test_labels = datasets_for_part_2(args)
    # train_samples =list(range(20))
    # train_samples = [0] #OVERFIT
    # model = Net_with_embedding(dataset=None,args=args) if args.embedding_model=='torch' else Net(dataset=None,args=args)
    # model = Net(dataset=None,args=args)
    # model = MLP(dataset=None,args=args)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model)
    # model.load_state_dict(torch.load(os.path.join(args.files_dir,args.model)))
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    # trainer = Train_next_step_pred(model,optimizer=None,args=args,train_data=train_data, full_training_dataset=full_dataset,
    #                                val_labels= val_labels,test_labels=test_labels,
    #                                embedding_dir=os.path.join(args.files_dir,args.load_embeddings))
    # best_val_acc = best_test_acc = best_epoch = best_val_loss = best_test_loss = 0
    # best_loss = np.float('inf')
    # best_acc = best_loss_epoch = best_acc_epoch = 0
    # for epoch in range(args.num_epochs):
    #     random.shuffle(train_samples)
    #     train_epoch_losses,train_epoch_accs=trainer.predictions(train_samples)
    #     print(f'Training Epoch {epoch+1} \n Mean Loss: {train_epoch_losses}, Mean Accuracy: {train_epoch_accs}')
    #     if train_epoch_losses<best_loss:
    #         best_loss = train_epoch_losses
    #         best_loss_epoch=epoch
    #     if best_acc<train_epoch_accs:
    #         best_acc = train_epoch_accs
    #         best_acc_epoch = epoch
        # if args.loss_calc=='batch':
        #     val_epoch_losses, val_epoch_accs, test_epoch_losses, test_epoch_accs = trainer.test_batch()
        # else:
        #     val_epoch_losses,val_epoch_accs,test_epoch_losses,test_epoch_accs, val_gt_acc, test_gt_acc=trainer.test()
    #     print(f'Validation Epoch {epoch+1} \n Mean Loss: {val_epoch_losses}, Mean Accuracy: {val_epoch_accs}')
    #     print(f'Test Epoch {epoch+1} \n Mean Loss: {test_epoch_losses}, Mean Accuracy: {test_epoch_accs}')
    #     # assert test_gt_acc==1
    #     # assert val_gt_acc==1
    #     # print(f'GT CHECK: test: {test_gt_acc}, val: {val_gt_acc}')
    #     train_results = {"epoch": epoch, "train loss": train_epoch_losses, "Train Acc": train_epoch_accs,
    #                      "Val Loss": val_epoch_losses, 'Val Acc': val_epoch_accs,"Test Loss": test_epoch_losses,
    #                      'Test Acc':test_epoch_accs}
    #     wandb.log(train_results)
    #     if val_epoch_accs>best_val_acc:
    #         best_val_acc=val_epoch_accs
    #         best_test_acc = test_epoch_accs
    #         best_val_loss = val_epoch_losses
    #         best_test_loss = test_epoch_losses
    #         best_epoch = epoch
    #         best_model = copy.deepcopy(trainer.model)
    # best_results = {"best_epoch": best_epoch, "Best Val Loss": best_val_loss, 'Best Val Acc': best_val_acc,
    #                 "Best Test Loss": best_test_loss, 'Best Test Acc': best_test_acc}
    # wandb.log(best_results)
    # torch.save(best_model.state_dict(), os.path.join(wandb.run.dir, f'Part2_{args.graph_worker}'))
    print(f'best_loss:{best_loss} on epoch {best_loss_epoch}')
    print(f'best_loss:{best_acc} on epoch {best_acc_epoch}')