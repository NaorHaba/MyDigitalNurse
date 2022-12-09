import os
import random
from typing import List

import cv2
import math
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils import ModelData
from surgery_data import SurgeryData


# collate function for DataLoader of DualFramesDataset
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


class DualFramesDataset(Dataset):

    def __init__(self, videos_dir: str, batch_size: int, frames_per_batch: int, relevant_surgeries: List[str]):
        # check if videos_dir exists
        assert os.path.isdir(videos_dir), f"{videos_dir} does not exist"

        self.videos_dir = videos_dir
        self.batch_size = batch_size
        self.frames_per_batch = frames_per_batch

        self.preprocess = transforms.Compose([
            transforms.Resize((242, 322)),
            transforms.CenterCrop((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

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

            # load images
            top_image = Image.open(top_image_path).convert('RGB')
            side_image = Image.open(side_image_path).convert('RGB')

            top_image = self.preprocess(top_image)
            side_image = self.preprocess(side_image)

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


# Helper class for DualVideoDataset
class VideoBatchGenerator:
    # read from a video directly and sequentially, to avoid loading all frames into memory
    # return a batch of frames from multiple videos

    def __init__(self, video_path, frames_per_batch, frame_rate):
        assert os.path.isfile(video_path), f"video path {video_path} is not a file"
        assert frames_per_batch > 0, f"frames per batch should be positive, got {frames_per_batch}"
        assert frame_rate is None or frame_rate > 0, f"frame rate should be positive, got {frame_rate}"
        self.video = cv2.VideoCapture(video_path)
        self.frames_per_batch = frames_per_batch
        self.frame_rate = frame_rate

    def __iter__(self):
        return self

    def __next__(self):
        if not self.video.isOpened():
            raise StopIteration
        frames = []
        for i in range(self.frames_per_batch * self.frame_rate):
            ret, frame = self.video.read()
            if ret:
                if i % self.frame_rate == 0:
                    frames.append(torch.tensor(frame).permute(2, 0, 1))
            else:
                self.video.release()
                break
        return self.video.isOpened(), torch.stack(frames)


class DualVideoDataset(Dataset):

    def __init__(self, videos_dir, surgery_batch_amount, frame_batch_amount_in_surgery, frame_rate=None):
        # check if videos_dir exists
        assert os.path.isdir(videos_dir), f"{videos_dir} does not exist"

        self.videos_dir = videos_dir
        self.surgery_batch_amount = surgery_batch_amount
        self.frame_batch_amount_in_surgery = frame_batch_amount_in_surgery
        self.frame_rate = frame_rate

        self.surgeries = list({sur.split('.')[0] for sur in os.listdir(videos_dir)})

        # seen surgeries
        self.epoch_surgery_batch_counter = None
        self.current_surgery_batch = None
        self._start_new_epoch()

    def _start_new_epoch(self):
        # shuffle surgeries
        self.epoch_surgery_batch_counter = 0
        random.shuffle(self.surgeries)
        self._initiate_surgery_batch(self.surgeries[:self.surgery_batch_amount])

    def __getitem__(self, _):

        batch = []
        for sur in self.current_surgery_batch:
            if self.current_surgery_batch[sur] is not None:
                batch.append(self._get_frames_from_surgery(sur))
                should_break = self._update_current_surgery_batch()
                if should_break:
                    break

        return batch, should_break

    def __len__(self):
        return len(self.surgeries)

    def _get_frames_from_surgery(self, sur):
        top, side = self.current_surgery_batch[sur]["top"], self.current_surgery_batch[sur]["side"]
        top_is_opened, top_frames = next(top)
        side_is_opened, side_frames = next(side)
        if not top_is_opened or not side_is_opened:
            print("top and side videos are not the same length for surgery {}".format(sur))  # FIXME
            min_batch_shape = min(top_frames.shape[0], side_frames.shape[0])
            top_frames = top_frames[:min_batch_shape]
            side_frames = side_frames[:min_batch_shape]
            self.current_surgery_batch[sur] = None
        return top_frames, side_frames

    def _update_current_surgery_batch(self):
        if all([self.current_surgery_batch[sur] is None for sur in self.current_surgery_batch]):
            self.epoch_surgery_batch_counter += 1
            if self.epoch_surgery_batch_counter >= len(self.surgeries) // self.surgery_batch_amount:
                self._start_new_epoch()
            else:
                self._initiate_surgery_batch(self.surgeries[self.surgery_batch_amount * self.epoch_surgery_batch_counter:
                                                            self.surgery_batch_amount * (self.epoch_surgery_batch_counter + 1)])

            return True

        return False

    def _initiate_surgery_batch(self, surgeries):
        self.current_surgery_batch = {sur: {"top": VideoBatchGenerator(os.path.join(self.videos_dir, sur + ".Camera1.avi"), self.frame_batch_amount_in_surgery, frame_rate=self.frame_rate),
                                            "side": VideoBatchGenerator(os.path.join(self.videos_dir, sur + ".Camera2.avi"), self.frame_batch_amount_in_surgery, frame_rate=self.frame_rate)}
                                      for sur in surgeries}
