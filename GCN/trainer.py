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

    def train(self, train_data_loader,test_data_loader,num_epochs, learning_rate,args):
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
        print('num seqs ',number_of_seqs)
        number_of_batches = len(train_data_loader.batch_sampler)
        wandb.init(project="RNN_BORIS_NEW_412", entity="surgical_data_science",mode=args.wandb_mode) #logging to wandb
        wandb.config.update(args)
        best_results = {'val acc': 0, 'val acc per surgery': 0,'val loss': np.float('inf')}
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        steps_no_improve = 0
        min_train_loss = np.float('inf')
        #Train model
        for epoch in range(num_epochs):
            # pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            mean_epoch_acc = 0
            for batch in tqdm.tqdm(train_data_loader):
                batch_input, batch_target, lengths, mask = [x.to(self.device) for x in batch[:-1]]
                batch_ids = batch[-1]
                mask = mask.to(self.device)
                optimizer.zero_grad()
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')
                max_len = torch.max(lengths).item()
                all_predictions = []
                for pos in range(args.starting_point,max_len):
                    cur_features = batch_input[:,:pos+1,:]
                    tmp_lengths = torch.tensor([min([pos+1,cur_sur_len.item()]) for cur_sur_len in lengths])
                    predictions = self.model(cur_features, tmp_lengths, mask)
                    all_predictions.append(predictions)
                tmp_predictions = [[x[i] for x in all_predictions] for i in range(len(all_predictions[0]))]
                all_predictions = [torch.stack(predictions_task).permute(1, 2, 0) for predictions_task in tmp_predictions]
                loss=0
                total_batch_acc = []
                for task_pred,targets in zip(all_predictions,batch_target.permute(2,0,1)):
                    loss += self.ce(task_pred, targets[:,args.starting_point:])
                    _, task_predictions = torch.max(task_pred, 1)
                    task_acc = 0
                    for sur_num, l in enumerate(lengths):
                        task_acc += (targets[sur_num, args.starting_point:l] == task_predictions[sur_num,:l-args.starting_point]).sum().item() / l
                    total_batch_acc.append(task_acc)
                # loss = self.ce(predictions, batch_target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # for sur_num, l in enumerate(lengths):
                #     total_batch_acc += (batch_target[sur_num, :l] == predicted[sur_num, :l]).sum().item() / l
                    # acc = (predicted==batch_target).sum()
                mean_epoch_acc +=np.mean(total_batch_acc)
            # pbar.close()
            train_loss = epoch_loss / number_of_batches
            mean_acc = mean_epoch_acc/number_of_seqs
            print(f" \n [epoch {epoch + 1}: train loss = {train_loss},mean acc: {mean_acc}")
                                  # f"train acc = {epoch_acc}, train F1 = {epoch_F1}")
            train_results = {"epoch": epoch, "train loss": train_loss,
                             "train acc": mean_acc}
            # results = self.eval(val_data_loader,features_type=args.features)
            test_results = self.eval(test_data_loader,starting_point=args.starting_point,name='test',features_type =args.features)
            test_results['epoch'] = epoch
            # print(f'Val acc: {results["val acc"]}, val loss:{results["val loss"]}')
            # if results['val acc'] > best_results['val acc']:
            #     best_results['val acc'] = results['val acc']
            #     best_results['epoch acc'] = epoch
            #     best_results['test acc for best val'] = test_results['test acc']
            # if results['val loss'] < best_results['val loss'] - 5e-3:
            #     best_results['val loss'] = results['val loss']
            #     best_results['epoch loss'] = epoch
            wandb.log(train_results)
            wandb.log(test_results)
            # wandb.log(results)
            if train_loss<min_train_loss-5e-3:
                steps_no_improve=0
            else:
                steps_no_improve+=1
            if train_loss < min_train_loss:
                min_train_loss=train_loss
            if steps_no_improve>=self.early_stop:
                break
        wandb.log({f'best_{k}': v for k, v in best_results.items()}) #log best results
        wandb.finish()
        print(best_results)
        # return train_results_list,eval_results_list
    #

    def eval(self, test_data_loader,features_type,starting_point,name='val'):
        results = {}
        self.model.eval()
        num_s = 5 if name=='test' else 6
        with torch.no_grad():
            epoch_loss = 0
            mean_epoch_acc = 0
            total_batch_acc= []
            for surgery in tqdm.tqdm(test_data_loader):
                sur_accuracy = 0
                sur_predictions = []
                sur_logits = []
                sur_input, sur_target = [x.to(self.device) for x in surgery[:-1]]
                sur_input = sur_input.squeeze(0)
                sur_target = sur_target.squeeze(0)
                num_features = sur_input.shape[1]
                idx_add_features = (25 if features_type=='1hot' else 1 if features_type=='label' else 2)-num_features
                cur_features = sur_input[:starting_point+1,:]
                model_logits = self.model(cur_features.unsqueeze(0), [1])
                preds = [torch.max(x, 1)[1] for x in model_logits]
                # pred = preds[0].item()
                sur_predictions.append(preds)
                sur_logits.append(model_logits)
                for pos,features in enumerate(sur_input[starting_point+1:]):
                    if features_type=='1hot':
                        pred= pred[0].items()
                        pred_features =torch.tensor([0.] * pred + [1.] + [0.] * (24 - pred)).to(self.device)
                    elif features_type=='label':
                        pred = pred[0].item()
                        pred_features =torch.tensor([pred]).to(self.device)
                    else:
                        pred_features =torch.tensor(preds).to(self.device)
                    if idx_add_features<0:
                        pred_features = torch.concat((pred_features,features[idx_add_features:]))
                    pred_features = pred_features.unsqueeze(0)
                    cur_features = torch.concat((cur_features,pred_features))
                    model_logits = self.model(cur_features.unsqueeze(0), [pos+2])
                    preds = [torch.max(x, 1)[1] for x in model_logits]
                    sur_predictions.append(preds)
                    sur_logits.append(model_logits)
                tmp_logits = [[x[i] for x in sur_logits] for i in range(len(sur_logits[0]))]
                all_logits = [torch.stack(predictions_task).float() for predictions_task in tmp_logits]
                tmp_preds = [[x[i] for x in sur_predictions] for i in range(len(sur_predictions[0]))]
                all_preds = [torch.stack(predictions_task).float() for predictions_task in tmp_preds]
                sur_loss= 0
                total_sur_acc = []
                for task_logit,task_pred,targets in zip(all_logits,all_preds,sur_target.permute(1,0)):
                    targets = targets[starting_point:]
                    sur_loss += self.ce(task_logit.squeeze(), targets)
                    total_sur_acc.append((targets == task_pred.squeeze()).sum().item()/targets.shape[0])
                epoch_loss += sur_loss.item()
                mean_epoch_acc += np.mean(total_sur_acc)
        results[f'{name} acc'] = mean_epoch_acc.item()/num_s
        results[f'{name} loss'] = epoch_loss/num_s

        self.model.train()
        return results

