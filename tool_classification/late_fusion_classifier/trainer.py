import os

import torch
import wandb
import torch.nn.functional as F
from tqdm import tqdm
import math


class Trainer:

    def __init__(self, model, optimizer, schedular, label_loss_fn, smoothness_loss_fn, smoothness_factor,
                 early_stop_patience, train_loader, val_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.schedular = schedular
        self.label_loss_fn = label_loss_fn
        self.smoothness_loss_fn = smoothness_loss_fn
        self.smoothness_factor = smoothness_factor
        self.early_stop_patience = early_stop_patience
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

    def _get_loss(self, net_output, ground_truth, lengths) -> torch.Tensor:
        loss = torch.tensor(0.0).to(self.device)
        for p_stage in net_output:
            # p_stage (batch_size, num_classes, seq_len)
            p_stage = p_stage.permute(0, 2, 1)  # (batch_size, seq_len, num_classes)
            for surgery_i in range(p_stage.shape[0]):
                surgery = p_stage[surgery_i]
                surgery = surgery[:lengths[surgery_i]]  # (true_seq_len, num_classes)
                # num_classes = 5 tools * 4 hands = 20
                surgery = surgery.view(-1, 4, 5)  # (true_seq_len, hands, tools)
                surgery = surgery.reshape(-1, 5)  # (true_seq_len * hands, tools)
                label_loss = self.label_loss_fn(surgery, ground_truth[surgery_i][:lengths[surgery_i]].view(-1))

                smoothness_loss = self.smoothness_loss_fn(F.log_softmax(surgery[1:, :], dim=-1),
                                                          F.log_softmax(surgery.detach()[:-1, :], dim=-1))  # TODO create a custom loss function?
                T = 4  # for clamp # FIXME args?
                # label_loss = label_loss.clamp(min=0, max=T)
                smoothness_loss = smoothness_loss.clamp(min=0, max=T)

                loss += label_loss + self.smoothness_factor * smoothness_loss

        return loss

    def _get_predictions(self, net_output, ground_truth, lengths):

        # taking only predictions from the first stage (same as in the paper)
        p_stage = net_output[0]
        # p_stage (batch_size, num_classes, seq_len)
        p_stage = p_stage.permute(0, 2, 1)  # (batch_size, seq_len, num_classes)
        # num_classes = 5 tools * 4 hands = 20
        p_stage = p_stage.view(p_stage.shape[0], -1, 4, 5)  # (batch_size, seq_len, hands, tools)
        predictions = torch.argmax(p_stage, dim=3)  # (batch_size, seq_len, tool_per_hands)

        # slice predictions and ground truth to the actual length of the sequence
        predictions = torch.cat([predictions[i, :lengths[i]] for i in range(len(lengths))])
        ground_truth = torch.cat([ground_truth[i, :lengths[i]] for i in range(len(lengths))])

        return predictions, ground_truth

    def train(self, epochs):
        best_val_acc = float(0)
        early_stop_counter = 0
        print("[START] Training")
        for epoch in range(epochs):
            print(f"[START] Epoch {epoch} / {epochs}")
            wandb.log({"epoch": epoch})
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_gt = []
            train_lengths = 0

            pbar = tqdm(self.train_loader, desc=f"EPOCH {epoch} - Train")
            for batch in pbar:
                batch_input, batch_target, lengths, mask = batch
                batch_input = {input: data.to(self.device) for input, data in batch_input.items()}
                batch_target = batch_target.to(self.device)
                mask = mask.to(self.device)
                mask = mask.permute(0, 2, 1)
                mask = mask[:, 0, :].unsqueeze(1)
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')

                self.optimizer.zero_grad()
                output = self.model(batch_input, lengths, mask)

                loss = self._get_loss(output, batch_target, lengths)
                loss.backward()

                clip_value = 0.25  # FIXME args?
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                self.optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                wandb.log({"train_loss": loss.item()})

                predictions, ground_truth = self._get_predictions(output, batch_target, lengths)
                train_predictions.append(predictions)
                train_gt.append(ground_truth)
                train_lengths += sum(lengths)

            self.schedular.step(train_loss)

            tqdm.write(f"Train avg loss: {train_loss / len(self.train_loader)}")
            train_predictions = torch.cat(train_predictions)
            train_gt = torch.cat(train_gt)
            train_acc = (train_predictions == train_gt).sum().item() / (train_lengths * 4)  # 4 hands
            tqdm.write(f"Train accuracy: {train_acc}")
            wandb.log({"train_accuracy": train_acc})

            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_gt = []
            val_lengths = 0
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f"EPOCH {epoch} - Val")
                for batch in pbar:
                    batch_input, batch_target, lengths, mask = batch
                    batch_input = {input: data.to(self.device) for input, data in batch_input.items()}
                    batch_target = batch_target.to(self.device)
                    mask = mask.to(self.device)
                    mask = mask.permute(0, 2, 1)
                    mask = mask[:, 0, :].unsqueeze(1)
                    lengths = lengths.to(dtype=torch.int64).to(device='cpu')

                    output = self.model(batch_input, lengths, mask)
                    loss = self._get_loss(output, batch_target, lengths)

                    val_loss += loss.item()

                    pbar.set_postfix(loss=loss.item())
                    wandb.log({"val_loss": loss.item()})

                    predictions, ground_truth = self._get_predictions(output, batch_target, lengths)
                    val_predictions.append(predictions)
                    val_gt.append(ground_truth)
                    val_lengths += sum(lengths)

                tqdm.write(f"Val avg loss: {val_loss / len(self.val_loader)}")
                val_predictions = torch.cat(val_predictions)
                val_gt = torch.cat(val_gt)
                val_acc = (val_predictions == val_gt).sum().item() / (val_lengths * 4)  # 4 hands
                tqdm.write(f"Val accuracy: {val_acc}")
                wandb.log({"val_accuracy": val_acc})

            if val_acc > best_val_acc and not math.isclose(val_acc, best_val_acc, rel_tol=1e-4):
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter == self.early_stop_patience:
                    print("Reached early stop patience after epoch", epoch)
                    break

            print(f"[END] Epoch {epoch} / {epochs}")

        print("[END] Training")

    def test(self):
        print("[START] Testing")
        self.model.eval()
        test_loss = 0
        test_predictions = []
        test_gt = []
        test_lengths = 0
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Test")
            for batch in pbar:
                batch_input, batch_target, lengths, mask = batch
                batch_input = {input: data.to(self.device) for input, data in batch_input.items()}
                batch_target = batch_target.to(self.device)
                mask = mask.to(self.device)
                mask = mask.permute(0, 2, 1)
                mask = mask[:, 0, :].unsqueeze(1)
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')

                output = self.model(batch_input, lengths, mask)
                loss = self._get_loss(output, batch_target, lengths)

                test_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                wandb.log({"test_loss": loss.item()})

                predictions, ground_truth = self._get_predictions(output, batch_target, lengths)
                test_predictions.append(predictions)
                test_gt.append(ground_truth)
                test_lengths += sum(lengths)

            tqdm.write(f"Test avg loss: {test_loss / len(self.test_loader)}")
            test_predictions = torch.cat(test_predictions)
            test_gt = torch.cat(test_gt)
            test_acc = (test_predictions == test_gt).sum().item() / (test_lengths * 4)  # 4 hands
            tqdm.write(f"Test accuracy: {test_acc}")
            wandb.log({"test_accuracy": test_acc})

        print("[END] Testing")
