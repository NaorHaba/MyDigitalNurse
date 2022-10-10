import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler, DataLoader

from datasets import DualVideoDataset
from trainer import Trainer
from utils import ModelData, SURGERY_GROUPS
from vision_model import VisionModel

SURGERIES_DIR = "/data/shared-data/scalpel/kristina/data/detection/usage/by video"


def connect_surgeries_collate_function(data):
    top_lens = tuple(len(x['top_images']) for x in data)
    side_lens = tuple(len(x['side_images']) for x in data)
    assert top_lens == side_lens, "Number of top and side frames must be equal for all surgeries!"
    try:
        top_images = torch.cat([x['top_images'] for x in data], dim=0)
        side_images = torch.cat([x['side_images'] for x in data], dim=0)
    except:
        pass

    # pad y
    labels = pad_sequence([x['labels'] for x in data], batch_first=True)
    return ModelData(
        torch.stack((top_images, side_images)),
        top_lens), labels


if __name__ == '__main__':

    seed = 41
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # create dataloaders
    batch_size = 4
    frames_per_batch = 16

    train_dataset = DualVideoDataset(SURGERIES_DIR, batch_size, frames_per_batch, SURGERY_GROUPS['train'])
    val_dataset = DualVideoDataset(SURGERIES_DIR, batch_size, frames_per_batch, SURGERY_GROUPS['val'])
    test_dataset = DualVideoDataset(SURGERIES_DIR, batch_size, frames_per_batch, SURGERY_GROUPS['test'])

    train_data_loader = DataLoader(train_dataset, batch_size=None, shuffle=False,
                                   sampler=SequentialSampler(train_dataset), collate_fn=connect_surgeries_collate_function)
    val_data_loader = DataLoader(val_dataset, batch_size=None, shuffle=False,
                                 sampler=SequentialSampler(val_dataset), collate_fn=connect_surgeries_collate_function)
    test_data_loader = DataLoader(test_dataset, batch_size=None, shuffle=False,
                                  sampler=SequentialSampler(test_dataset), collate_fn=connect_surgeries_collate_function)

    model = VisionModel('identity', 'patch')  # TODO add parameters
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trainer = Trainer(model, optimizer, loss_fn, train_data_loader, val_data_loader, test_data_loader, device)

    epochs = 200

    trainer.train(epochs)

    trainer.test()

