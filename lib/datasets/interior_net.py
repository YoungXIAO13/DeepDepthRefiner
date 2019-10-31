import os
from os.path import join
import cv2
import torch
import numpy as np
import pandas as pd

import torch.utils.data as data


class InteriorNet(data.Dataset):
    def __init__(self, root_dir, split_name, preprocess=None,
                 gt_dir='data', label_dir='label', pred_dir='pred',
                 depth_ext='-depth.png', label_ext='-order-pix.npy'):
        super(InteriorNet, self).__init__()
        self.root_dir = root_dir
        self.split_name = split_name
        self.preprocess = preprocess
        self.gt_dir = gt_dir
        self.label_dir = label_dir
        self.pred_dir = pred_dir
        self.depth_ext = depth_ext
        self.label_ext = label_ext
        self.df = pd.read_csv(join(root_dir, '{}.txt'.format(split_name)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        depth_gt, depth_pred, label = self._fetch_data(index)

        if self.preprocess is not None:
            depth_gt, depth_pred, label = self.preprocess(depth_gt, depth_pred, label)

        depth_gt = torch.from_numpy(np.ascontiguousarray(depth_gt)).float().unsqueeze(0)
        depth_pred = torch.from_numpy(np.ascontiguousarray(depth_pred)).float().unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label)).float().permute(2, 0, 1)

        return depth_gt, depth_pred, label

    def _fetch_data(self, index):
        # fetch depth map and normalize the values into [0, 1)
        depth_gt_path = join(self.root_dir, self.gt_dir, self.df.iloc[index]['scene'],
                             self.df.iloc[index]['image'] + self.depth_ext)
        depth_gt = cv2.imread(depth_gt_path, -1) / 1000 / 50

        depth_pred_path = join(self.root_dir, self.pred_dir, self.df.iloc[index]['scene'],
                               self.df.iloc[index]['image'] + self.depth_ext)
        depth_pred = cv2.imread(depth_pred_path, -1) / 1000 / 50

        # fetch occlusion orientation labels
        label_path = join(self.root_dir, self.label_dir, self.df.iloc[index]['scene'],
                          self.df.iloc[index]['image'] + self.label_ext)
        label = np.load(label_path)

        return depth_gt, depth_pred, label


if __name__ == "__main__":
    root_dir = '/space_sdd/InteriorNet'
    dataset = InteriorNet(root_dir, 'train')
    print(len(dataset))
