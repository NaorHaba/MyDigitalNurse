import torch
from tqdm import tqdm


class Trainer:

    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

    def get_loss(self, net_output, ground_truth, lengths) -> torch.Tensor:
        total_loss = torch.tensor(0.0, requires_grad=False)
        for i, position in enumerate(['surgeon_R', 'surgeon_L', 'assistant_R', 'assistant_L']):
            cur_output = []
            cur_ground_truth = []
            for j, sur_len in enumerate(lengths):
                cur_output.append(net_output[position][j, :sur_len, :])
                cur_ground_truth.append(ground_truth[j, :sur_len, i])
            cur_output = torch.cat(cur_output)
            cur_ground_truth = torch.cat(cur_ground_truth)
            total_loss += self.loss_fn(cur_output, cur_ground_truth)

        return total_loss

    def get_predictions(self, net_output, ground_truth, lengths):

        predictions = []
        ground_truths = []
        for i, position in enumerate(['surgeon_R', 'surgeon_L', 'assistant_R', 'assistant_L']):
            for j, sur_len in enumerate(lengths):
                predictions.append(net_output[position][j, :sur_len, :].argmax(dim=1))
                ground_truths.append(ground_truth[j, :sur_len, i])

        return torch.cat(predictions), torch.cat(ground_truths)

    def train(self, epochs):
        print("[START] Training")
        for epoch in range(epochs):
            print(f"[START] Epoch {epoch} / {epochs}")
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_gt = []
            train_lengths = 0

            pbar = tqdm(self.train_loader, desc=f"EPOCH {epoch} - Train")
            for batch in pbar:
                self.optimizer.zero_grad()
                x, y = batch
                x.surgeries_data.to(self.device)
                output = self.model(x)
                lengths = x.lengths
                loss = self.get_loss(output, y, lengths)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix(loss=loss.item())

                predictions, ground_truth = self.get_predictions(output, y, lengths)
                train_predictions.append(predictions)
                train_gt.append(ground_truth)
                train_lengths += sum(lengths)

            tqdm.write(f"Train avg loss: {train_loss / len(self.train_loader)}")
            train_predictions = torch.cat(train_predictions)
            train_gt = torch.cat(train_gt)
            tqdm.write(f"Train accuracy: {train_predictions.eq(train_gt).sum().item() / train_lengths}")

            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_gt = []
            val_lengths = 0
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f"EPOCH {epoch} - Val")
                for batch in pbar:
                    x, y = batch
                    x.surgeries_data.to(self.device)
                    output = self.model(x)
                    lengths = x.lengths
                    loss = self.get_loss(output, y, lengths)

                    val_loss += loss.item()

                    pbar.set_postfix(loss=loss.item())

                    predictions, ground_truth = self.get_predictions(output, y, lengths)
                    val_predictions.append(predictions)
                    val_gt.append(ground_truth)
                    val_lengths += sum(lengths)

                tqdm.write(f"Val avg loss: {val_loss / len(self.val_loader)}")
                tqdm.write(f"Val accuracy: {torch.cat(val_predictions).eq(torch.cat(val_gt)).sum().item() / val_lengths}")

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
                x, y = batch
                x.surgeries_data.to(self.device)
                output = self.model(x)
                lengths = x.lengths
                loss = self.get_loss(output, y, lengths)

                test_loss += loss.item()

                pbar.set_postfix(loss=loss.item())

                predictions, ground_truth = self.get_predictions(output, y, lengths)
                test_predictions.append(predictions)
                test_gt.append(ground_truth)
                test_lengths += sum(lengths)

            tqdm.write(f"Test avg loss: {test_loss / len(self.test_loader)}")
            tqdm.write(f"Test accuracy: {torch.cat(test_predictions).eq(torch.cat(test_gt)).sum().item() / test_lengths}")

        print("[END] Testing")


