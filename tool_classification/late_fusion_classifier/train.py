# import colored as colored
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

import utils
from datasets import DualSavedFeaturesFullSurgeryDataset, pad_collate_fn
from utils import ACTIVATIONS, OPTIMIZERS
from trainer import Trainer
import os
import argparse
import random
import logging
from model import MS_TCN, MS_TCN_PP, SeparateFeatureExtractor, SurgeryModel

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    # names and paths
    parser.add_argument('--features_dir_prefix', type=str, default="/data/home/naor-haba/MyDigitalNurse/yolov7/runs/detect/",
                        help='path to features main directory. complete path depends on the feature_input')
    parser.add_argument('--labels_dir', type=str, default="/strg/C/shared-data/ACS_2019_data/boris/boris_big_gt",
                        help='path to labels')

    # wandb
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], default='online',
                        help='Wandb mode')
    parser.add_argument('--project', type=str, default="MDN-MSTCN",
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
    # defaults are selected according to the hyperparameters we found
    parser.add_argument('--time_series_model', type=str, choices=['MSTCN', 'MSTCN++'], default='MSTCN++',
                        help='time series model to use')
    parser.add_argument('--feature_extractor', type=str, choices=['separate_identity'], default='separate_identity',
                        help='feature extractor to use')  # possible to add more

    parser.add_argument('--num_stages', type=int, default=4,
                        help='number of stages in the MSTCN models')
    parser.add_argument('--num_layers', type=int, default=14,
                        help='number of layers in each stage in the MSTCN models')
    parser.add_argument('--num_f_maps', type=int, default=64,
                        help='number of feature maps in each layer in the MSTCN models')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout rate in the MSTCN models')
    parser.add_argument('--activation', type=str, choices=list(ACTIVATIONS.keys()), default='relu',
                        help='activation to use')
    parser.add_argument('--smoothness_factor', type=float, default=0.35,
                        help='smoothness factor for the model losses')

    parser.add_argument('--data_inputs', type=str, nargs='+', default=['top', 'side'],
                        help='data inputs to use')
    parser.add_argument('--feature_input', type=str, choices=['features', 'labels'], default='labels',
                        help='feature input to use')
    parser.add_argument('--frame_rate', type=int, default=1,
                        help='frame rate of the features from the videos')
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
    args.features_per_view = 4500 if args.feature_input == 'features' else 36

    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_model(args):
    if args.feature_extractor == 'separate_identity':
        fe_params = {d + '_fe': nn.Identity() for d in args.data_inputs}
        fe = SeparateFeatureExtractor(**fe_params)
        feature_sizes = {d: args.features_per_view for d in args.data_inputs}
    else:
        raise NotImplementedError

    dims = sum([feature_sizes[d] for d in args.data_inputs])
    if args.time_series_model == 'MSTCN':
        ts = MS_TCN(num_stages=args.num_stages, num_layers=args.num_layers, num_f_maps=args.num_f_maps, dim=dims,
                    num_classes=args.num_classes, activation=args.activation, dropout=args.dropout)
    elif args.time_series_model == 'MSTCN++':
        ts = MS_TCN_PP(num_stages=args.num_stages, num_layers=args.num_layers, num_f_maps=args.num_f_maps, dim=dims,
                       num_classes=args.num_classes, activation=args.activation, dropout=args.dropout)
    else:
        raise NotImplementedError

    sm = SurgeryModel(fe, ts)
    return sm


def train():
    args = get_args()
    logger.info(args)
    set_seed()

    with wandb.init(project=args.project, entity=args.entity, config=args.__dict__, mode=args.wandb_mode):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device if args.device != 'cpu' else '-1'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('using device:', device)

        ds_train = DualSavedFeaturesFullSurgeryDataset(args.features_dir_prefix + args.feature_input, args.labels_dir, utils.SURGERY_GROUPS['train'],
                                                       camera_views=args.data_inputs, features_per_view=args.features_per_view,
                                                       frame_rate=args.frame_rate)
        ds_val = DualSavedFeaturesFullSurgeryDataset(args.features_dir_prefix + args.feature_input, args.labels_dir, utils.SURGERY_GROUPS['val'],
                                                     camera_views=args.data_inputs, features_per_view=args.features_per_view,
                                                     frame_rate=args.frame_rate)
        ds_test = DualSavedFeaturesFullSurgeryDataset(args.features_dir_prefix + args.feature_input, args.labels_dir, utils.SURGERY_GROUPS['test'],
                                                      camera_views=args.data_inputs, features_per_view=args.features_per_view,
                                                      frame_rate=args.frame_rate)

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=pad_collate_fn)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, collate_fn=pad_collate_fn)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                                              num_workers=0, collate_fn=pad_collate_fn)

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
        smooth_loss_fn = nn.MSELoss()

        trainer = Trainer(model, optimizer, schedular, label_loss_fn, smooth_loss_fn, args.smoothness_factor,
                          args.patience, dl_train, dl_val, dl_test, device)
        trainer.train(args.num_epochs)

        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_model.pth')))

        trainer.test()

    return


if __name__ == '__main__':
    train()
