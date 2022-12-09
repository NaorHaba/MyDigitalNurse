# The purpose of this file is to convert the labels predicted from the YOLO model to a format that can be used by the late_fusion_classifier (features).
# The labels are in the format: [label, x, y, w, h] for each hand in the frame.
# In order to convert the labels to features with a constant size, we need to do the following:
# 1. Remove duplicates of the same hand (keep the first one)
# 2. Add missing hands (if there are less than 4 hands in the frame)
# 3. One hot encode the labels
# 4. Add zeros to the features of the missing hands
# 5. Concatenate the labels and the features to a single array
# 6. Flatten the array
# 7. Save the array to a file


import os

import numpy as np
import argparse


def process_frame_labels(frame_labels, padding_value):
    tool_labels = (frame_labels // 4).astype(int)  # ranging from 0 to 4
    # 1 hot encode labels
    tool_labels = np.eye(5)[tool_labels]
    # add missing hands
    missing_hands = []
    zeros_pad = np.zeros(5)
    tool_labels_one_hot = []
    tool_labels_index = 0
    for i in range(4):
        if i in frame_labels % 4:
            tool_labels_one_hot.append(tool_labels[tool_labels_index])
            tool_labels_index += 1
        else:
            tool_labels_one_hot.append(zeros_pad)
            missing_hands.append(i)
    tool_labels_one_hot = np.array(tool_labels_one_hot)
    return tool_labels_one_hot, missing_hands


def process_frame_features(frame_features, missing_hands):
    zeros_pad = np.zeros(4)  # 4 coordinates
    features = []
    feature_index = 0
    for i in range(4):
        if i not in missing_hands:
            features.append(frame_features[feature_index])
            feature_index += 1
        else:
            features.append(zeros_pad)
    features = np.array(features)
    return features


def run(surgeries_labels_dir, output_dir, padding_value=0):

    for surgery_name in os.listdir(surgeries_labels_dir):
        print('Processing surgery: {}'.format(surgery_name))
        surgery_path = os.path.join(surgeries_labels_dir, surgery_name)
        output_path = os.path.join(output_dir, surgery_name)
        surgery_files = os.listdir(surgery_path)
        # order by frame number (split by _ and take the last element and remove the extension)
        surgery_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        surgery_features = []
        surgery_duplication_count = 0
        for file in surgery_files:
            # files end with txt
            frame_labels_features = np.loadtxt(os.path.join(surgery_path, file), delimiter=' ')
            # make 2 dimensional array
            frame_labels_features = frame_labels_features.reshape(-1, 5)
            # labels are in the first column
            # order by label % 4 (ordering by hand)
            frame_labels_features = frame_labels_features[(frame_labels_features[:, 0] % 4).argsort()]
            # remove duplicates of the same hand (keep the first tool)
            original_length = len(frame_labels_features)
            frame_labels_features = frame_labels_features[np.unique(frame_labels_features[:, 0] % 4, return_index=True)[1]]
            surgery_duplication_count += original_length - len(frame_labels_features)

            # sanity check
            assert frame_labels_features.shape[0] <= 4, "There should be max 4 labels per frame"

            # process labels to features
            frame_labels = frame_labels_features[:, 0]
            frame_labels, missing_hands = process_frame_labels(frame_labels, padding_value)

            frame_features = frame_labels_features[:, 1:]
            frame_features = process_frame_features(frame_features, missing_hands)

            frame_features = np.concatenate((frame_labels, frame_features), axis=1).flatten()
            surgery_features.append(frame_features)
        surgery_features = np.array(surgery_features)
        print('Surgery {} has {} frames and {} duplications'.format(surgery_name, len(surgery_features), surgery_duplication_count))
        np.save(str(output_path) + '.npy', surgery_features)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--surgeries_labels_dir', type=str, default='../yolov7/runs/detect/exp11/labels')
    parser.add_argument('--output_dir', type=str, default='../yolov7/runs/detect/camera2_labels')
    parser.add_argument('--padding_value', type=int, default=0)
    args = parser.parse_args()

    run(args.surgeries_labels_dir, args.output_dir, args.padding_value)
