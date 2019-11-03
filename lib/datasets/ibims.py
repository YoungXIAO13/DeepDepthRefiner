import os
from os.path import join
import torch
import numpy as np
from scipy import io

import torch.utils.data as data


class Ibims(data.Dataset):
    def __init__(self, root_dir, method_name,
                 gt_dir='gt_depth', label_dir='label', label_ext='-order-pix.npy'):
        super(Ibims, self).__init__()
        self.root_dir = root_dir
        self.gt_dir = gt_dir
        self.method_name = method_name
        self.label_dir = label_dir
        self.label_ext = label_ext
        with open(join(self.root_dir, 'imagelist.txt')) as f:
            image_names = f.readlines()
        self.im_names = [x.strip() for x in image_names]

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, index):
        depth_gt, depth_pred, label = self._fetch_data(index)

        depth_gt = torch.from_numpy(np.ascontiguousarray(depth_gt)).float().unsqueeze(0)
        depth_pred = torch.from_numpy(np.ascontiguousarray(depth_pred)).float().unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label)).float().permute(2, 0, 1)

        return depth_gt, depth_pred, label

    def _fetch_data(self, index):
        # fetch depth map and normalize the values into [0, 1)
        depth_gt_mat = join(self.root_dir, self.gt_dir, '{}.mat'.format(self.im_names[index]))
        depth_pred_mat = join(self.root_dir, self.method_name, '{}_predictions_{}_results.mat'.format(
            self.im_names[index], self.method_name))
        depth_gt, depth_pred = self._load_depths_from_mat(depth_gt_mat, depth_pred_mat)
        depth_gt /= 50
        depth_pred /= 50

        # fetch occlusion orientation labels
        label_path = join(self.root_dir, self.label_dir, self.im_names[index] + self.label_ext)
        label = np.load(label_path)

        return depth_gt, depth_pred, label

    def _load_depths_from_mat(self, gt_mat, pred_mat):
        # load prediction depth
        pred = io.loadmat(pred_mat)['pred_depths']
        pred[np.isnan(pred)] = 0
        pred_invalid = pred.copy()
        pred_invalid[pred_invalid != 0] = 1

        # load ground truth depth
        image_data = io.loadmat(gt_mat)
        data = image_data['data']

        # extract neccessary data
        depth = data['depth'][0][0]  # Raw depth map
        mask_invalid = data['mask_invalid'][0][0]  # Mask for invalid pixels
        mask_transp = data['mask_transp'][0][0]  # Mask for transparent pixels

        mask_missing = depth.copy()  # Mask for further missing depth values in depth map
        mask_missing[mask_missing != 0] = 1

        mask_valid = mask_transp * mask_invalid * mask_missing * pred_invalid  # Combine masks

        depth_valid = depth * mask_valid
        pred_valid = pred * mask_valid
        return depth_valid, pred_valid


if __name__ == "__main__":
    root_dir = '/space_sdd/ibims'
    method_name = 'junli'
    dataset = Ibims(root_dir, method_name)
    print(len(dataset))

    from torch.utils.data import DataLoader
    import sys
    import time
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    begin = time.time()
    for i, data in enumerate(test_loader):
        data = data
        print(time.time() - begin)
        if i == 0:
            print(data[0].shape, data[1].shape, data[2].shape)
            sys.exit()
