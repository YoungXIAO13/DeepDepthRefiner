import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import time

from lib.models.unet import UNet
from lib.datasets.ibims import Ibims
from lib.datasets.interior_net import InteriorNet

from lib.utils.net_utils import kaiming_init, save_checkpoint, load_checkpoint
from lib.utils.evaluate_ibims_error_metrics import compute_distance_related_errors, compute_global_errors

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# network training procedure settings
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

# pth settings
parser.add_argument('--session', type=int, default=0, help='training session')
parser.add_argument('--resume', action='store_true', help='resume checkpoint or not')
parser.add_argument('--checkpoint', type=str, default=None, help='optional reload model path')
parser.add_argument('--save_dir', type=str, default='model', help='save model path')

# dataset settings
parser.add_argument('--val_dir', type=str, default='/space_sdd/ibims', help='testing dataset')
parser.add_argument('--val_method', type=str, default='junli')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# =================CREATE DATASET=========================== #
dataset_val = Ibims(opt.val_dir, opt.val_method)
val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
# ========================================================== #


# ================CREATE NETWORK AND OPTIMIZER============== #
net = UNet()
net.apply(kaiming_init)
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0.0005)

load_checkpoint(net, optimizer, opt.checkpoint)

net.cuda()
# ========================================================== #


# ===================== DEFINE VAL ========================= #
def val(data_loader, net):
    # Initialize global and geometric errors ...
    num_samples = len(data_loader)
    rms = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    thr1 = np.zeros(num_samples, np.float32)
    thr2 = np.zeros(num_samples, np.float32)
    thr3 = np.zeros(num_samples, np.float32)

    net.eval()
    for i, data in enumerate(data_loader):
        # load data and label
        depth_gt, depth_coarse, occlusion = data
        depth_gt, depth_coarse, occlusion = depth_gt.cuda(), depth_coarse.cuda(), occlusion.cuda()

        # forward pass
        depth_pred = net(occlusion, depth_coarse)

        # mask out invalid depth values
        valid_mask = ((depth_gt != 0) * (depth_pred != 0)).float()
        gt_valid = depth_gt * valid_mask
        pred_valid = depth_pred * valid_mask

        # get numpy array from torch tensor
        gt = gt_valid.squeeze().detach().cpu().numpy()
        pred = pred_valid.squeeze().detach().cpu().numpy()

        gt_vec = gt.flatten()
        pred_vec = pred.flatten()

        abs_rel[i], sq_rel[i], rms[i], log10[i], thr1[i], thr2[i], thr3[i] = compute_global_errors(gt_vec, pred_vec)

    return abs_rel, sq_rel, rms, log10, thr1, thr2, thr3
# ========================================================== #


abs_rel, sq_rel, rms, log10, thr1, thr2, thr3 = val(val_loader, net)
print('############ Global Error Metrics #################')
print('rel    = ', np.nanmean(abs_rel))
print('sq_rel = ', np.nanmean(sq_rel))
print('log10  = ', np.nanmean(log10))
print('rms    = ', np.nanmean(rms))
print('thr1   = ', np.nanmean(thr1))
print('thr2   = ', np.nanmean(thr2))
print('thr3   = ', np.nanmean(thr3))