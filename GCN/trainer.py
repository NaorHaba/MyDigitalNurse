import torch

from LSTM_Model import *
import sys
from torch import optim
import torch.nn as nn
import math
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import wandb
from datetime import datetime
import tqdm


class Trainer:
    """
    RNN Trainer class
    """
    def __init__(self, model, device="cuda",early_stop=15):
        """
        :param model: RNN model
        :param device: in our case, cpu
        """
        # self.model = model.float()
        self.device = device
        self.model= model
        print(device)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100) #loss
        self.early_stop = early_stop

    def train(self, train_data_loader,test_data_loader, num_epochs, learning_rate,args):
        """
        :param train_data_loader:
        :param test_data_loader:
        :param num_epochs: max number of epochs
        :param learning_rate: lr for the optimizer
        :param args:
        :param early_stop: number of epochs with no F1 on train improvement to stop running
        :param eval_rate: validation evaluation rate
        :return: results for each epoch and each dataset
        """
        number_of_seqs = len(train_data_loader.sampler)
        number_of_batches = len(train_data_loader.batch_sampler)
        wandb.init(project="lab1_lstm", entity="labteam",mode=args.wandb_mode) #logging to wandb
        wandb.config.update(args)
        self.model.train()
        best_results = {'val acc': 0, 'val loss': np.float('inf')}
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        steps_no_improve = 0
        min_train_loss = np.float('inf')
        #Train model
        for epoch in range(num_epochs):
            # pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            mean_epoch_acc = 0
            len_batches = 0
            for batch in tqdm.tqdm(train_data_loader):
                batch_input, batch_target, lengths, mask = [x.to(self.device) for x in batch]
                mask = mask.to(self.device)
                optimizer.zero_grad()
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')
                predictions = self.model(batch_input, lengths, mask)
                loss = self.ce(predictions, batch_target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(predictions, 1)
                acc = (predicted==batch_target).sum()
                # acc = np.mean([np.mean([batch_target[j][i] == predicted[j][i] for i in range(lengths[j])]) for j in
                       # range(len(lengths))])
                mean_epoch_acc +=acc.item()
            # pbar.close()
            train_loss = epoch_loss / number_of_batches
            mean_acc = mean_epoch_acc/number_of_seqs
            print(f" \n [epoch {epoch + 1}: train loss = {train_loss},mean acc: {mean_acc}")
                                  # f"train acc = {epoch_acc}, train F1 = {epoch_F1}")
            train_results = {"epoch": epoch, "train loss": train_loss,
                             "train acc": mean_acc}
            results = self.eval(test_data_loader)
            print(f'Val acc: {results["val acc"]}, val loss:{results["val loss"]}')
            if results['val acc'] > best_results['val acc'] + 5e-3:
                best_results['val acc'] = results['val acc']
                best_results['epoch acc'] = epoch
            if results['val loss'] < best_results['val loss'] + 5e-3:
                best_results['val loss'] = results['val loss']
                best_results['epoch loss'] = epoch
            wandb.log(train_results)
            if train_loss < min_train_loss - 5e-3:
                steps_no_improve=0
                min_train_loss=train_loss
            else:
                steps_no_improve+=1
            if steps_no_improve>=self.early_stop:
                break
            # train_results_list.append(train_results)
        #     if (epoch + 1) % eval_rate == 0:
        #         print("epoch: " + str(epoch + 1) + " model evaluation")
        #         results = {"epoch": epoch}
        #         results.update(self.eval(test_data_loader))
        #         eval_results_list.append(results)
        #         print(f"\n  [epoch {epoch + 1}: val acc = {results['val acc']}, val F1 = {results['val F1']}")
        #         wandb.log(results)
        #         if results['val acc'] > best_results['val acc'] + 5e-3:
        #             best_results['val acc'] = results['val acc']
        #             best_results['epoch acc'] = epoch
        #         if results['val F1'] > best_results['val F1'] + 5e-3:
        #             best_results['val F1'] = results['val F1']
        #             best_results['epoch F1'] = epoch
        #             torch.save({"model_state": self.model.state_dict()}, f'{wandb.run.dir}/best_f1.pth') #save model with best Val F1
        #
            # if results['val loss'] > best_results['val loss'] + 1e-2:
            #     best_F1 = epoch_F1
            #     steps_no_improve = 0
            # else:
            #     steps_no_improve += 1
            #     if steps_no_improve >= early_stop:
            #         break
        wandb.log({f'best_{k}': v for k, v in best_results.items()}) #log best results
        wandb.finish()
        print(best_results)
        # return train_results_list,eval_results_list
    #
    # def eval_notforced(self, test_data_loader,name='val'):
    #     results = {}
    #     self.model.eval()
    #     all_preds = []
    #     all_labels = []
    #     epoch_loss=0
    #     mean_epoch_acc=0
    #     loss = 0
    #     with torch.no_grad():
    #         # self.model.to(self.device)
    #         for batch in tqdm.tqdm(test_data_loader):
    #             batch_input, batch_target, lengths, mask = [x.to(self.device) for x in batch]
    #             lengths = lengths.to(dtype=torch.int64).to(device='cpu')
    #             sur_accs = torch.tensor([0] * len(lengths))
    #             batch_predicted_input = batch_input[:,0, :]
    #             first_predictions = self.model(batch_predicted_input, [1] * len(lengths), mask[:, :1, :])
    #             target = batch_target[:, 0]
    #             _, predicted = torch.max(last_pred, dim=1)
    #             sur_accs[lengths > i] += predicted == target
    #             loss += self.ce(first_predictions, target)
    #             for i in range(1,max(lengths)):
    #                 features = [torch.tensor([0.] * state + [1.] + [0.] * (24 - state)) for state in predicted[lengths>i]]
    #                 # batch_predicted_input = torch.stack(batch_predicted_input[length>i,:,:], features)
    #                 batch_samp = batch_input[lengths > i, :i + 1, :]
    #                 mask_samp = mask[lengths > i, :i + 1, :]
    #                 predictions = self.model(batch_samp, [i+1]*sum(lengths>i).item(), mask_samp)
    #                 target = batch_target[lengths > i,i]
    #                 last_pred= predictions[:,-1,:]
    #                 _,predicted = torch.max(last_pred,dim=1)
    #                 sur_accs[lengths>i] += predicted==target
    #                 loss += self.ce(last_pred,target)
    #             epoch_loss += loss.item()
    #             acc = np.mean([sur_accs[i]/lengths[i] for i in range(len(lengths))])
    #             mean_epoch_acc +=acc
    #         results[f'{name} acc'] = (mean_epoch_acc/len(test_data_loader)).item()
    #         results[f'{name} loss'] = epoch_loss/len(test_data_loader)
    #     self.model.train()
    #     return results
    # #
    # def eval(self, test_data_loader,name='val'):
    #     results = {}
    #     self.model.eval()
    #     all_preds = []
    #     all_labels = []
    #     epoch_loss=0
    #     mean_epoch_acc=0
    #     loss = 0
    #     with torch.no_grad():
    #         # self.model.to(self.device)
    #         for batch in test_data_loader:
    #             batch_input, batch_target, lengths, mask,_ = batch
    #             lengths = lengths.to(dtype=torch.int64).to(device='cpu')
    #             sur_accs = torch.tensor([0] * len(lengths))
    #             for i in range(max(lengths)):
    #                 batch_samp = batch_input[lengths > i, :i + 1, :]
    #                 mask_samp = mask[lengths > i, :i + 1, :]
    #                 predictions = self.model(batch_samp, [i+1]*sum(lengths>i).item(), mask_samp)
    #                 target = batch_target[lengths > i,i]
    #                 last_pred= predictions[:,-1,:]
    #                 _,predicted = torch.max(last_pred,dim=1)
    #                 sur_accs[lengths>i] += predicted==target
    #                 loss += self.ce(last_pred,target)
    #             epoch_loss += loss.item()
    #             acc = np.mean([sur_accs[i]/lengths[i] for i in range(len(lengths))])
    #             mean_epoch_acc +=acc
    #         results[f'{name} acc'] = mean_epoch_acc/len(test_data_loader)
    #         results[f'{name} loss'] = epoch_loss/len(test_data_loader)
    #     self.model.train()
    #     return results

    def eval(self, test_data_loader,name='val'):
        results = {}
        self.model.eval()
        all_preds = []
        all_labels = []
        epoch_loss=0
        mean_epoch_acc=0
        loss = 0
        with torch.no_grad():
            # self.model.to(self.device)
            number_of_seqs = len(test_data_loader.sampler)
            for batch in test_data_loader:
                batch_input, batch_target, lengths, mask = [x.to(self.device) for x in batch]
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')
                predictions = self.model(batch_input, lengths, mask)
                loss = self.ce(predictions, batch_target)
                epoch_loss += loss.item()
                _, predicted = torch.max(predictions, 1)
                acc = (predicted == batch_target).sum()
                mean_epoch_acc +=acc.item()
                epoch_loss += loss.item()
            results[f'{name} acc'] = mean_epoch_acc/number_of_seqs
            results[f'{name} loss'] = epoch_loss/len(test_data_loader)
        self.model.train()
        return results

