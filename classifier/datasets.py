import os
import random
from typing import List

import cv2
import math
import torch
import torchvision
from torch.utils.data import Dataset

from surgery_data import SurgeryData


class DualVideoDataset(Dataset):

    def __init__(self, videos_dir: str, batch_size: int, frames_per_batch: int, relevant_surgeries: List[str]):
        # check if videos_dir exists
        assert os.path.isdir(videos_dir), f"{videos_dir} does not exist"

        self.videos_dir = videos_dir
        self.batch_size = batch_size
        self.frames_per_batch = frames_per_batch

        self.surgery_data = {}
        self.surgery_amounts = {}
        for _dir in os.listdir(videos_dir):
            if os.path.isdir(os.path.join(videos_dir, _dir)) and self._is_surgery_relevant(_dir, relevant_surgeries):
                self.surgery_data[_dir] = SurgeryData(os.path.join(videos_dir, _dir))
                self.surgery_amounts[_dir] = len(self.surgery_data[_dir])

        # seen surgeries
        self.epoch_groups_of_surgeries = None
        self.number_of_batches_per_group = None
        self._start_new_epoch()

    @staticmethod
    def _is_surgery_relevant(surgery, relevant_surgeries):
        return surgery in relevant_surgeries

    def _start_new_epoch(self):
        # random split all surgeries into groups of size self.batch_size
        self.epoch_groups_of_surgeries = []
        # shuffle surgeries
        surgeries = list(self.surgery_amounts.keys())
        random.shuffle(surgeries)
        for i in range(0, len(surgeries), self.batch_size):
            self.epoch_groups_of_surgeries.append(surgeries[i:i+self.batch_size])
        self.number_of_batches_per_group = [math.ceil(max([self.surgery_amounts[s] for s in group]) / self.frames_per_batch) for group in self.epoch_groups_of_surgeries]

    def __getitem__(self, index):
        # load batch surgeries
        cur_sur_index = self._find_surgery_group_index_for_batch(index)
        cur_sur_group = self.epoch_groups_of_surgeries[cur_sur_index]
        # batch counter for current group
        batch_counter = index - sum(self.number_of_batches_per_group[:cur_sur_index])

        batch = []
        for sur in cur_sur_group:
            start = batch_counter * self.frames_per_batch
            # check if surgery is finished
            if start >= self.surgery_amounts[sur]:
                continue
            end = min(start + self.frames_per_batch, self.surgery_amounts[sur])

            batch.append(self._get_data_from_surgery(sur, start=start, end=end))

        # reset number of batches per group if epoch is finished
        if index + 1 == len(self):
            self._start_new_epoch()

        return batch

    def __len__(self):
        return sum(self.number_of_batches_per_group)

    def _get_data_from_surgery(self, sur, start, end):
        top_images = []
        side_images = []

        labels = []

        file_list = self.surgery_data[sur].image_frames
        for i in range(start, end):
            top_image_path = os.path.join(self.videos_dir, sur, 'images', file_list[i] + '_1.jpg')
            side_image_path = os.path.join(self.videos_dir, sur, 'images', file_list[i] + '_2.jpg')

            # load image as tensor and reshape to (..., 3, H, W)
            top_image = cv2.imread(top_image_path)
            side_image = cv2.imread(side_image_path)

            top_image = torchvision.transforms.ToTensor()(top_image)
            side_image = torchvision.transforms.ToTensor()(side_image)

            top_images.append(top_image)
            side_images.append(side_image)

            labels.append(self.surgery_data[sur].frame_labels[i])

        # return as dict
        return {
            'top_images': torch.stack(top_images),
            'side_images': torch.stack(side_images),
            'labels': torch.stack(labels)
        }

    def _find_surgery_group_index_for_batch(self, index):
        # find group of surgeries for current batch
        group_index = 0
        for i in range(len(self.number_of_batches_per_group)):
            if index < sum(self.number_of_batches_per_group[:i+1]):
                group_index = i
                break
        return group_index