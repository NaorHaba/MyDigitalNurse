# import colored as colored
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

import utils
from datasets import DualFramesDataset, \
    connect_surgeries_collate_function
from utils import ACTIVATIONS, OPTIMIZERS
from trainer import Trainer
import os
import argparse
import random
import logging
from model.vision_model import VisionModel

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    # names and paths
    parser.add_argument('--surgeries_dir', type=str, default="/data/shared-data/scalpel/kristina/data/detection/usage/by video",
                        help='path to features main directory. complete path depends on the feature_input')

    # wandb
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], default='online',
                        help='Wandb mode')
    parser.add_argument('--project', type=str, default="MDN-EarlyFusion",
                        help='project name for wandb')
    parser.add_argument('--entity', type=str, default="surgical_data_science",
                        help='entity name for wandb')

    # data loading
    parser.add_argument('--device', type=str, default="0", choices=['0', '1', 'cpu'],
                        help='device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for the dataloader')

    # train
    parser.add_argument('--patience', type=int, default=15,
                        help='patience for early stopping')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='number of epochs')

    # model tuning parameters:
    parser.add_argument('--feature_extractor', type=str, choices=['identity'], default='identity',
                        help='feature extractor to use to extract features from the images before the start of the network')  # possible to add more
    parser.add_argument('--tokenizer', type=str, choices=['patch'], default='patch',
                        help='tokenizer to use to tokenize the images')  # possible to add more

    parser.add_argument('--image_h', type=int, default=240,
                        help='image height to convert the images to before feeding them to the network')
    parser.add_argument('--image_w', type=int, default=320,
                        help='image width to convert the images to before feeding them to the network')
    parser.add_argument('--patch_size', type=int, default=20,
                        help='patch size to use for the tokenizer')
    parser.add_argument('--embedding_dim', type=float, default=1024,
                        help='embedding dimension to use for the tokenizer and the vision model')
    parser.add_argument('--time_series_layers', type=int, default=2,
                        help='number of layers to use for the time series model')
    parser.add_argument('--activation', type=str, choices=list(ACTIVATIONS.keys()), default='relu',
                        help='activation to use')

    parser.add_argument('--lr', type=float, default=0.00035,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--optimizer', type=str, choices=list(OPTIMIZERS.keys()), default='adam',
                        help='optimizer to use')
    parser.add_argument('--schedular', type=str, choices=['plateau', 'step'], default='plateau',
                        help='schedular to use')

    args = parser.parse_args()

    args.activation = ACTIVATIONS[args.activation]

    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_model(args):
    model = VisionModel(args.feature_extractor, args.tokenizer, args.image_h, args.image_w, args.patch_size,
                        args.embedding_dim, time_series_layers=args.time_series_layers)
    return model


def train():
    args = get_args()
    logger.info(args)
    set_seed()

    with wandb.init(project=args.project, entity=args.entity, config=args.__dict__, mode=args.wandb_mode):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device if args.device != 'cpu' else '-1'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('using device:', device)

        ds_train = DualFramesDataset(args.surgeries_dir, args.batch_size, args.frames_per_batch, utils.SURGERY_GROUPS['train'])
        ds_val = DualFramesDataset(args.surgeries_dir, args.batch_size, args.frames_per_batch, utils.SURGERY_GROUPS['val'])
        ds_test = DualFramesDataset(args.surgeries_dir, args.batch_size, args.frames_per_batch, utils.SURGERY_GROUPS['test'])

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=None, shuffle=False,
                                               num_workers=args.num_workers, collate_fn=connect_surgeries_collate_function)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=None, shuffle=False,
                                             num_workers=0, collate_fn=connect_surgeries_collate_function)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=None, shuffle=False,
                                              num_workers=0, collate_fn=connect_surgeries_collate_function)

        model = create_model(args)
        model.to(device)
        print(model)

        wandb.watch(model)

        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        # print total trainable parameters
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        optimizer = OPTIMIZERS[args.optimizer](model.parameters(), lr=args.lr)
        if args.schedular == 'plateau':
            schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                                   verbose=True)
        elif args.schedular == 'step':
            schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        else:
            raise NotImplementedError

        label_loss_fn = nn.CrossEntropyLoss()

        # Trainer(model, optimizer, loss_fn, train_data_loader, val_data_loader, test_data_loader, device)
        trainer = Trainer(model, optimizer, schedular, label_loss_fn,
                          args.patience, dl_train, dl_val, dl_test, device)
        trainer.train(args.num_epochs)

        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_model.pth')))

        trainer.test()

    return


if __name__ == '__main__':
    train()
