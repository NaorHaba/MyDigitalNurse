import os
import random
import time

import cv2
import torch
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.dataloader import _InfiniteConstantSampler


class VideoBatchGenerator:

    def __init__(self, video_path, frames_per_batch, frame_rate):
        assert os.path.isfile(video_path), f"video path {video_path} is not a file"
        assert frames_per_batch > 0, f"frames per batch should be positive, got {frames_per_batch}"
        assert frame_rate is None or frame_rate > 0, f"frame rate should be positive, got {frame_rate}"
        self.video = cv2.VideoCapture(video_path)
        self.frames_per_batch = frames_per_batch
        self.frame_rate = frame_rate
        if self.frame_rate is None:
            self.frame_rate = 10  # default

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

        return batch

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

def main():
    videos_dir = "data/AVI_files"
    batch_size = 2
    frames_per_batch = 16

    dataset = DualVideoDataset(videos_dir, batch_size, frames_per_batch)

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                             sampler=_InfiniteConstantSampler())

    epoch_steps = 10000
    for i in range(epoch_steps):
        start = time.time()
        batch = next(iter(dataloader))
        print(f"batch {i} took {time.time() - start} seconds")
        pass


if __name__ == "__main__":
    # measure time
    start = time.time()
    main()
    end = time.time()
    print(f"Time: {end - start}")
