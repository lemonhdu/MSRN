import os

import numpy as np

from params import *
from torch.utils.data import Dataset

class VideoDataset(Dataset):

    def __init__(self, mode, args):
        super(VideoDataset, self).__init__()
        self.mode = mode
        self.args = args

        # instance_idx is an one-dimension array
        if self.mode == 'train':
            self.instance_idx = np.load(train_index_dir).transpose(1, 0)
        if self.mode == 'test':
            self.instance_idx = np.load(test_index_dir).transpose(1, 0)

        self.overall_scores = self.get_scores(self.instance_idx)
        self.difficulty_level = self.get_difficulty_level(self.instance_idx)
        self.tcn_time = self.get_tcn_time(self.instance_idx)
        self.tcn_time_filling = self.get_tcn_time_filling(self.instance_idx)

    def get_imgs_npy(self, ix):

        index = self.instance_idx[0, ix]

        imgs_npy_dir = os.path.join(frames_npy_dir, str('{:03d}'.format(index))+'.npy')

        imgs_npy = np.load(imgs_npy_dir)

        imgs_npy = imgs_npy.transpose(2, 0, 1).squeeze(axis=2)

        return imgs_npy

    def get_tcn_time_filling(self, instance_idx):

        tcn_time_labels_all = []

        for item in instance_idx[0]:

            load_path = os.path.join(tcn_time_points_filling_save_dir, str("{:03d}".format(item))+".npy")

            tcn_time_labels_single = np.load(load_path)

            tcn_time_labels_all.append(tcn_time_labels_single)

        return tcn_time_labels_all


    def get_scores(self, instance_idx):
        score_array = np.load(overall_score_dir)

        score = score_array[0, instance_idx-1]

        return score

    def get_difficulty_level(self, instance_idx):
        difficulty_array = np.load(difficulty_dir)

        difficulty = difficulty_array[0, instance_idx-1]

        return difficulty


    def get_tcn_time(self, instance_idx):

        np.load.__defaults__ = (None, True, True, 'ASCII')
        tcn_time_array = np.load(tcn_time_dir)
        np.load.__defaults__ = (None, False, True, 'ASCII')

        tcn_time = np.array(tcn_time_array[instance_idx - 1])

        return tcn_time


    def __getitem__(self, ix):
        data = {}
        data['video'] = self.get_imgs_npy(ix)
        data['final_score'] = self.overall_scores[0, ix]
        data['difficulty'] = self.difficulty_level[0, ix]
        data['tcn_time'] = self.tcn_time[0, ix]
        data['tcn_time_labels'] = self.tcn_time_filling[ix]
        return data

    def __len__(self):
        return len(self.tcn_time[0])
