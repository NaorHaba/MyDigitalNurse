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
from link_utils import Net, Net_with_embedding, datasets_for_part_2, find_edge_idx
import networkx as nx
import torch.nn.functional as F
import wandb
import copy
import time
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', default='/data/home/liory/GCN/MyDigitalNurse/GCN/Training_datasets_part2_NoSelfLoops')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--Project', choices=['MDN-NextStep-BORIS', 'MDN-NextStep-BORIS-NoSelfLoops'], default='MDN-NextStep-BORIS-NoSelfLoops', type=str)

    parser.add_argument('--graph_worker', choices=['surgeon', 'assistant','both'], default='both')
    parser.add_argument('--loss_calc', choices=['surgery', 'node', 'batch'], default='surgery')
    parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--files_dir', default='/data/home/liory/GCN/MyDigitalNurse/GCN/wandb/run-20221023_085939-wid8ih2u/files')
    # devout-sweep-453
    # --activation = tanh -embedding_dim = 256  -embedding_model = DeepWalk  -first_conv_f = 64 -- -num_layers = 3
    parser.add_argument('--files_dir', default='/data/home/liory/GCN/MyDigitalNurse/GCN/wandb/run-20221022_124019-atyndn4j/files') #With self loops
    #
    # --activation = tanh -embedding_dim = 256  -embedding_model = DeepWalk  -first_conv_f = 4096 -- -num_layers = 8

    parser.add_argument('--model', default='IPG_both' )
    parser.add_argument('--first_conv_f', default=4096 , type=int)
    parser.add_argument('--num_layers', default=8, type=int)
    parser.add_argument('--activation', choices=['relu','lrelu','tanh','sigmoid'], default='tanh')
    parser.add_argument('--load_embeddings', default='train_embeddings_both.pkl' )
    parser.add_argument('--embedding_model', choices=['Cora','DeepWalk','torch'], default='DeepWalk')
    parser.add_argument('--embedding_dim', default=256, type=int)

    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.08427977033491187, type=float)
    args = parser.parse_args()
    assert args.first_conv_f/2**args.num_layers>=1
    return args



class Train_next_step_pred():
    def __init__(self,  model:Net,optimizer:torch.optim,train_data,full_training_dataset, val_labels,test_labels,args, embedding_dir=None):
        self.model = model
        self.optimizer=optimizer
        self.args=args
        self.loss_calc = args.loss_calc
        self.full_training_dataset = full_training_dataset
        self.train_data = train_data
        self.labels = {'val': val_labels, 'test': test_labels}
        self.videos_path = args.videos_path
        self.embedding_dir = os.path.join(args.files_dir, args.load_embeddings)

    def train_frames_batch(self, data,labels):
        self.model.train()
        last_node=labels[0]
        acc=[]
        losses = []
        batch_index = range(1,len(labels)-1,self.args.batch_size)
        for batch_start in batch_index:
            cur_labels = labels[batch_start:batch_start+self.args.batch_size]
            self.optimizer.zero_grad()
            node_labels = []
            all_logits=[]
            for node in cur_labels:
                z = self.model.encode(x=data.x, edge_index=data.edge_index, weight=data.weight)
                prediction, logits=self.model.predict(z,last_node,selfloops=False)
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


    def train_surgery(self, data,labels):
        self.model.train()
        last_node=labels[0]
        acc=[]
        node_labels= []
        losses = []
        all_logits=[]
        self.optimizer.zero_grad()
        for node in labels[1:]:
            z = self.model.encode(x=data.x, edge_index=data.edge_index, weight=data.weight)
            prediction, logits=self.model.predict(z,last_node,selfloops=False)
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
        return loss.item(), np.mean(acc)



    def train_node(self, data,labels):
        """
        :return:
        """
        self.model.train()
        last_node=labels[0]
        acc=[]
        node_labels= []
        losses = []
        for node in labels[1:]:
            z = self.model.encode(x=data.x, edge_index=data.edge_index, weight=data.weight)
            prediction, logits=self.model.predict(z,last_node,selfloops=False)
            node_labels.append(node)
            acc.append(prediction==node.item())
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
            losses.append(loss.item())
            self.optimizer.zero_grad()
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
            prediction, logits=self.model.predict(z,last_node,selfloops=False)
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
        for k, all_labels in self.labels:
            losses=[]
            accs = []
            for sur_labels in all_labels:
                full_training_dataset = copy.deepcopy(self.full_training_dataset)
                batch_index = range(1, len(sur_labels) - 1, self.args.batch_size)
                last_node= sur_labels[0]
                sur_losses = []
                for batch_start in batch_index:
                    cur_labels = sur_labels[batch_start:batch_start + self.args.batch_size]
                    all_logits, node_labels,  = [], [] ,[]
                    for node in cur_labels:
                        z = self.model.encode(x=full_training_dataset.x, edge_index=full_training_dataset.edge_index,
                                              weight=full_training_dataset.weight)
                        prediction, logits=self.model.predict(z,last_node,selfloops=False)
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
                    loss = F.cross_entropy(all_logits, torch.tensor(node_labels))
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
            for sur_labels in all_labels:
                full_training_dataset = copy.deepcopy(self.full_training_dataset)
                last_node= sur_labels[0]
                all_logits, node_labels, sur_losses = [], [], []
                labels = sur_labels[0:]
                for node in labels:
                    z = self.model.encode(x=full_training_dataset.x, edge_index=full_training_dataset.edge_index,
                                          weight=full_training_dataset.weight)
                    prediction, logits=self.model.predict(z,last_node,selfloops=False)
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
        return res['val_loss'], res['val_acc'], res['test_loss'], res['test_acc']


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
            epoch_losses=[]
            epoch_accs = []
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
        return np.mean(epoch_losses),np.mean(epoch_accs)



if __name__ == "__main__":
    args = parsing()
    set_seed(args.seed)
    wandb.init(project=args.Project, entity="surgical_data_science", mode=args.wandb_mode)  # logging to wandb
    wandb.config.update(args)
    train_data, full_dataset, val_labels, test_labels = datasets_for_part_2(args)
    # full_dataset, val_labels, test_labels = datasets_for_part_2(args)
    train_samples =list(range(20))
    model = Net_with_embedding(dataset=None,args=args) if args.embedding_model=='torch' else Net(dataset=None,args=args)
    model.load_state_dict(torch.load(os.path.join(args.files_dir,args.model)))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    trainer = Train_next_step_pred(model,optimizer,args=args,train_data=train_data, full_training_dataset=full_dataset,
                                   val_labels= val_labels,test_labels=test_labels,
                                   embedding_dir=os.path.join(args.files_dir,args.load_embeddings))
    best_val_acc = best_test_acc = best_epoch = best_val_loss = best_test_loss = 0
    for epoch in range(args.num_epochs):
        random.shuffle(train_samples)
        train_epoch_losses,train_epoch_accs=trainer.predictions(train_samples)
        print(f'Training Epoch {epoch+1} \n Mean Loss: {train_epoch_losses}, Mean Accuracy: {train_epoch_accs}')
        if args.loss_calc=='batch':
            val_epoch_losses, val_epoch_accs, test_epoch_losses, test_epoch_accs = trainer.test_batch()
        else:
            val_epoch_losses,val_epoch_accs,test_epoch_losses,test_epoch_accs=trainer.test()
        print(f'Validation Epoch {epoch+1} \n Mean Loss: {val_epoch_losses}, Mean Accuracy: {val_epoch_accs}')
        print(f'Test Epoch {epoch+1} \n Mean Loss: {test_epoch_losses}, Mean Accuracy: {test_epoch_accs}')
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