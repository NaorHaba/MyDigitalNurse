import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# helper functions:
def get_info_from_label(label: str):
    if len(label) == 1:
        worker = 'surgeon'
        side = label
        tool_id = '0'
    elif len(label) == 2:
        worker = 'assistant'
        side = label[0]
        tool_id = '0'
    else:
        if label[1] == '2':
            worker = 'assistant'
        else:
            worker = 'surgeon'
        side = label[0]
        tool_id = label[-1]
    return worker, side, tool_id


def get_labels_df_from_file(sur_file):
    df = pd.read_csv(sur_file, skiprows=15)
    index = 0
    insert_dict = {'Index': index, 'Time': 0, 'Time_Diff': 0, 'R_surgeon': 0, 'L_surgeon': 0, 'R_assistant': 0,
                   'L_assistant': 0, 'FPS': 30}
    new_df = pd.DataFrame(insert_dict, index=[0])
    df = df[['Time', 'FPS', 'Behavior', 'Status']][df.Status == 'START']
    for i, row in df.iterrows():
        worker, side, tool_id = get_info_from_label(row.Behavior)
        tool_id = int(tool_id)
        if i < 4:
            if tool_id != 0:
                new_df.at[0, f'{side}_{worker}'] = tool_id
        else:
            index += 1
            time_diff = row.Time - insert_dict['Time']
            if time_diff == 0:
                new_df.at[index - 1, f'{side}_{worker}'] = tool_id
                insert_dict[f'{side}_{worker}'] = tool_id
                index -= 1
            else:
                insert_dict['Index'] = index
                insert_dict['Time_Diff'] = time_diff
                insert_dict['Time'] = row.Time
                insert_dict['FPS'] = row.FPS
                insert_dict[f'{side}_{worker}'] = tool_id
                new_df = pd.concat([new_df, pd.DataFrame(insert_dict, index=[index])], ignore_index=True)
    return new_df


def create_labels_from_surgery(sur_file, hands):
    processed_df = get_labels_df_from_file(sur_file)
    processed_df['Shift_Time'] = processed_df['Time_Diff'].shift(-1).fillna(1 / processed_df.FPS)
    states = []
    for i, row in processed_df.iterrows():
        state = tuple(int(row[x].item()) for x in hands)
        states += [state] * int(row.FPS * row.Shift_Time)
    return torch.LongTensor(states)


# dataset implementations:
class DualSavedFeaturesPartialSurgeryDataset(Dataset):
    # a dataset to read a batch of PARTIAL surgeries together and return the features and labels
    # expected to be used with dataloader batch_size=None, which means we return a batch from __getitem__ and not a single sample

    # by PARTIAL, we mean that the surgeries are read sequentially with a given frame amount for each batch
    # meaning, we can use this dataset to read for example 3 surgeries with 100 frames each and keep track of the
    # frame index for each surgery to know where to start reading for the next batch, thus allowing us to read
    # the surgeries in a sequential manner

    def __init__(self, features_dir, labels_dir, surgery_batch_amount, frame_batch_amount_in_surgery, surgeries,
                 selected_hands=('R_surgeon', 'L_surgeon', 'R_assistant', 'L_assistant'), frame_rate=None):
        # check if videos_dir exists
        assert os.path.isdir(features_dir), f"{features_dir} does not exist"

        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.surgery_batch_amount = surgery_batch_amount
        self.frame_batch_amount_in_surgery = frame_batch_amount_in_surgery
        self.selected_hands = selected_hands
        if frame_rate is None:
            frame_rate = 10
        self.frame_rate = frame_rate

        self.surgeries = surgeries

        # seen surgeries
        self.epoch_surgery_batch_counter = None
        self.current_surgery_batch_videos = None
        self.current_surgery_batch_labels = None
        self.current_surgery_batch_batch_counter = None
        self._start_new_epoch()

    def _start_new_epoch(self):
        # shuffle surgeries
        self.epoch_surgery_batch_counter = 0
        random.shuffle(self.surgeries)
        self._initiate_surgery_batch(self.surgeries[:self.surgery_batch_amount])

    def __getitem__(self, _):

        batch_frames = []
        batch_labels = []
        for sur in self.current_surgery_batch_videos:
            if self.current_surgery_batch_videos[sur] is not None:
                batch_frames.append(self._get_frames_from_surgery(sur))
                batch_labels.append(self._get_labels_from_surgery(sur))
        self._update_current_surgery_batch()

        return batch_frames, batch_labels

    def __len__(self):
        return 1500

    def _get_frames_from_surgery(self, sur):
        top_frames, side_frames = self.current_surgery_batch_videos[sur]["top"], self.current_surgery_batch_videos[sur][
            "side"]
        top_frames = top_frames.view(top_frames.shape[0] // 4500, -1)
        side_frames = side_frames.view(side_frames.shape[0] // 4500, -1)
        processed_frames_per_batch = self.frame_batch_amount_in_surgery * self.frame_rate
        frame_indices = range(self.current_surgery_batch_batch_counter[sur] * processed_frames_per_batch,
                              min((self.current_surgery_batch_batch_counter[sur] + 1) * processed_frames_per_batch,
                                  top_frames.shape[0], side_frames.shape[0],
                                  self.current_surgery_batch_labels[sur].shape[0]), self.frame_rate)
        if (self.current_surgery_batch_batch_counter[sur] + 1) * processed_frames_per_batch >= min(len(top_frames),
                                                                                                   len(side_frames)):
            self.current_surgery_batch_videos[sur] = None

        return {"top": top_frames[frame_indices], "side": side_frames[frame_indices]}

    def _update_current_surgery_batch(self):
        if all([self.current_surgery_batch_videos[sur] is None for sur in self.current_surgery_batch_videos]):
            self.epoch_surgery_batch_counter += 1
            if self.epoch_surgery_batch_counter >= len(self.surgeries) // self.surgery_batch_amount:
                self._start_new_epoch()
            else:
                self._initiate_surgery_batch(
                    self.surgeries[self.surgery_batch_amount * self.epoch_surgery_batch_counter:
                                   self.surgery_batch_amount * (self.epoch_surgery_batch_counter + 1)])

            return True

        for sur in self.current_surgery_batch_batch_counter:
            self.current_surgery_batch_batch_counter[sur] += 1

        return False

    def _initiate_surgery_batch(self, surgeries):
        self.current_surgery_batch_videos = {
            sur: {"top": torch.from_numpy(
                np.load(os.path.join(self.features_dir, 'camera1', sur + ".Camera1.npy"))).float(),
                  "side": torch.from_numpy(
                      np.load(os.path.join(self.features_dir, 'camera2', sur + ".Camera2.npy"))).float()}
            for sur in surgeries}
        self.current_surgery_batch_labels = {
            sur: create_labels_from_surgery(os.path.join(self.labels_dir, sur + ".Camera1.csv"), self.selected_hands)
            for sur in surgeries}
        self.current_surgery_batch_batch_counter = {sur: 0 for sur in surgeries}

    def _get_labels_from_surgery(self, sur):
        labels = self.current_surgery_batch_labels[sur]
        processed_frames_per_batch = self.frame_batch_amount_in_surgery * self.frame_rate
        labels_indices = range(self.current_surgery_batch_batch_counter[sur] * processed_frames_per_batch,
                               min((self.current_surgery_batch_batch_counter[sur] + 1) * processed_frames_per_batch,
                                   labels.shape[0]), self.frame_rate)
        if (self.current_surgery_batch_batch_counter[sur] + 1) * processed_frames_per_batch >= len(labels):
            self.current_surgery_batch_videos[sur] = None
        return labels[labels_indices]


class DualSavedFeaturesFullSurgeryDataset(Dataset):
    # Read a WHOLE surgery (features and labels) and return it

    def __init__(self, features_dir, labels_dir, surgeries, camera_views=('top', 'side'), features_per_view=4500,
                 selected_hands=('R_surgeon', 'L_surgeon', 'R_assistant', 'L_assistant'), frame_rate=None):
        # check if videos_dir exists
        assert os.path.isdir(features_dir), f"{features_dir} does not exist"
        assert os.path.isdir(labels_dir), f"{labels_dir} does not exist"
        assert len(camera_views) > 0, "No camera views selected"

        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.camera_views = camera_views
        self.features_per_view = features_per_view
        self.selected_hands = selected_hands
        if frame_rate is None:
            frame_rate = 10
        self.frame_rate = frame_rate

        self.surgeries = surgeries

    def __getitem__(self, i):
        sur = self.surgeries[i]
        surgery_frames = {}

        for view in self.camera_views:
            surgery_frames[view] = torch.from_numpy(
                np.load(os.path.join(self.features_dir, view, sur + ".npy"))).float()
            if 'features' in self.features_dir:
                # a fix for a mistake in the feature extraction - saved as 1 dim instead of 2
                surgery_frames[view] = surgery_frames[view].view(
                    surgery_frames[view].shape[0] // self.features_per_view, -1)
            surgery_frames[view] = surgery_frames[view][::self.frame_rate]

        labels = create_labels_from_surgery(os.path.join(self.labels_dir, sur + ".Camera1.csv"), self.selected_hands)

        min_len = min(min([len(surgery_frames[view]) for view in self.camera_views]), len(labels))
        for view in self.camera_views:
            surgery_frames[view] = surgery_frames[view][:min_len]
        labels = labels[:min_len]

        return surgery_frames, labels

    def __len__(self):
        return len(self.surgeries)


# collate functions for the dataloader
def pad_collate_fn(batch):
    # collate function to pad a batch of surgeries together
    input_lengths = []
    input_masks = []
    batch_features = {}
    batch_labels = []
    input_names = batch[0][0].keys()

    for input_name in input_names:
        batch_features[input_name] = []
        input_lengths_tmp = []
        for sample in batch:
            sample_features = sample[0][input_name]
            input_lengths_tmp.append(sample_features.shape[0])
            batch_features[input_name].append(sample_features)
        # pad
        batch_features[input_name] = torch.nn.utils.rnn.pad_sequence(batch_features[input_name], batch_first=True)
        input_lengths.append(input_lengths_tmp)
        # compute mask
        input_masks.append(batch_features[input_name] != 0)

    input_lengths_tmp = []
    for sample in batch:
        sample_labels = sample[1]
        input_lengths_tmp.append(sample_labels.shape[0])
        batch_labels.append(sample_labels)
    # pad
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, padding_value=-100, batch_first=True)
    input_lengths.append(input_lengths_tmp)

    # sanity check - all inputs (from all feature names of the same surgery) should have the same length
    assert [input_lengths[0]] * len(input_lengths) == input_lengths
    return batch_features, batch_labels, torch.tensor(input_lengths[0]), input_masks[0]


def pad_collate_fn_batch_none(batch):
    # collate function to pad a batch of surgeries together - another case for when batch size is None (for partial surgery readers)
    batch = [[batch[0][i], batch[1][i]] for i in range(len(batch[0]))]
    input_lengths = []
    input_masks = []
    batch_features = {}
    batch_labels = []
    input_names = batch[0][0].keys()

    for input_name in input_names:
        batch_features[input_name] = []
        input_lengths_tmp = []
        for sample in batch:
            sample_features = sample[0][input_name]
            input_lengths_tmp.append(sample_features.shape[0])
            batch_features[input_name].append(sample_features)
        # pad
        batch_features[input_name] = torch.nn.utils.rnn.pad_sequence(batch_features[input_name], batch_first=True)
        input_lengths.append(input_lengths_tmp)
        # compute mask
        input_masks.append(batch_features[input_name] != 0)

    input_lengths_tmp = []
    for sample in batch:
        sample_labels = sample[1]
        input_lengths_tmp.append(sample_labels.shape[0])
        batch_labels.append(sample_labels)
    # pad
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, padding_value=-100, batch_first=True)
    input_lengths.append(input_lengths_tmp)

    # sanity check - all inputs (from all feature names of the same surgery) should have the same length
    assert [input_lengths[0]] * len(input_lengths) == input_lengths
    return batch_features, batch_labels, torch.tensor(input_lengths[0]), input_masks[0]
