import os
import torch
from utils import LABEL_ID_TO_NAME, LABEL_INFO_TO_ID


class SurgeryData:

    def __init__(self, surgery_dir):

        self.surgery_dir = surgery_dir
        self.image_frames = self.parse_surgery_image_frames()
        self.frame_labels = self.parse_surgery_frame_labels()

    def __len__(self):
        return len(self.image_frames)

    def parse_surgery_image_frames(self):
        images_dir = os.path.join(self.surgery_dir, 'images')

        sorted_top_frames = sorted([f for f in os.listdir(images_dir) if self._is_top_frame(f)], key=lambda x: int(x.split('_')[-2]))
        sorted_side_frames = sorted([f for f in os.listdir(images_dir) if self._is_side_frame(f)], key=lambda x: int(x.split('_')[-2]))

        top_frames = set(['_'.join(f.split('_')[:-1]) for f in sorted_top_frames])
        side_frames = set(['_'.join(f.split('_')[:-1]) for f in sorted_side_frames])
        total_frames = set(top_frames).intersection(set(side_frames))

        if total_frames != top_frames or total_frames != side_frames:
            print(f"Warning: top and side frames do not match entirely for {self.surgery_dir}")

        return total_frames

    def parse_surgery_frame_labels(self):
        labels_dir = os.path.join(self.surgery_dir, 'labels')
        top_frames = [f for f in os.listdir(labels_dir) if self._is_top_frame(f)]
        side_frames = [f for f in os.listdir(labels_dir) if self._is_side_frame(f)]
        top_frames = set(['_'.join(f.split('_')[:-1]) for f in top_frames])
        side_frames = set(['_'.join(f.split('_')[:-1]) for f in side_frames])

        all_labels = []
        last_labels = {
            'surgeon': {'R': '0', 'L': '0'},
            'assistant': {'R': '0', 'L': '0'}
        }

        for frame in self.image_frames:
            frame_top_labels = {
                'surgeon': {'R': None, 'L': None},
                'assistant': {'R': None, 'L': None}
            }
            frame_side_labels = {
                'surgeon': {'R': None, 'L': None},
                'assistant': {'R': None, 'L': None}
            }
            if frame not in top_frames:
                print(f"Warning: frame {frame} is missing from top for {self.surgery_dir}")
            if frame not in side_frames:
                print(f"Warning: frame {frame} is missing from side for {self.surgery_dir}")

            top_label_path = os.path.join(labels_dir, f'{frame}_1.txt')
            side_label_path = os.path.join(labels_dir, f'{frame}_2.txt')
            top_labels = [LABEL_ID_TO_NAME[int(l.split(' ')[0])] for l in open(top_label_path).read().splitlines()]
            side_labels = [LABEL_ID_TO_NAME[int(l.split(' ')[0])] for l in open(side_label_path).read().splitlines()]
            for label in top_labels:
                worker, side, tool_id = self.get_info_from_label(label)
                frame_top_labels[worker][side] = tool_id
            for label in side_labels:
                worker, side, tool_id = self.get_info_from_label(label)
                frame_side_labels[worker][side] = tool_id

            parsed_labels = []
            for worker in ['surgeon', 'assistant']:
                for side in ['R', 'L']:
                    if frame_top_labels[worker][side] is None and frame_side_labels[worker][side] is None:
                        last_labels[worker][side] = last_labels[worker][side]  # no change
                    elif frame_top_labels[worker][side] is None:
                        last_labels[worker][side] = frame_side_labels[worker][side]
                    elif frame_side_labels[worker][side] is None:
                        last_labels[worker][side] = frame_top_labels[worker][side]
                    elif frame_top_labels[worker][side] != frame_side_labels[worker][side]:
                        print(f"Warning: Top and side labels do not match for {frame} {worker} {side}")
                        # we decide to take the side label in such cases
                        last_labels[worker][side] = frame_side_labels[worker][side]
                    else:
                        last_labels[worker][side] = frame_top_labels[worker][side]

                    parsed_labels.append(LABEL_INFO_TO_ID['_'.join([worker, side, last_labels[worker][side]])] // 4)

            all_labels.append(torch.LongTensor(parsed_labels))

        return all_labels

    @staticmethod
    def _is_top_frame(path):
        return path.split('_')[-1][0] == '1'

    @staticmethod
    def _is_side_frame(path):
        return path.split('_')[-1][0] == '2'

    @staticmethod
    def get_info_from_label(label):
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
