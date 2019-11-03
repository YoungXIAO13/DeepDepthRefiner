import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def kaiming_init(net):
    """Kaiming Init layer parameters."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print('save model at {}'.format(filename))


def load_checkpoint(model, optimizer, pth_file):
    print("loading checkpoint from {}".format(pth_file))
    checkpoint = torch.load(pth_file, map_location=lambda storage, loc: storage.cuda())
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Previous weight loaded')
    return epoch


def log_smooth_l1_loss(pred, target):
    log_pred = pred.log()
    log_target = target.log()
    loss = F.smooth_l1_loss(log_pred, log_target)
    return loss


def neighbor_depth_variation(depth):
    """Compute the variation of depth values in the neighborhood-8 of each pixel"""
    var1 = depth[..., 1:-1, 1:-1] - depth[..., :-2, :-2]
    var2 = depth[..., 1:-1, 1:-1] - depth[..., :-2, 1:-1]
    var3 = depth[..., 1:-1, 1:-1] - depth[..., :-2, 2:]
    var4 = depth[..., 1:-1, 1:-1] - depth[..., 1:-1, :-2]
    var6 = depth[..., 1:-1, 1:-1] - depth[..., 1:-1, 2:]
    var7 = depth[..., 1:-1, 1:-1] - depth[..., 2:, :-2]
    var8 = depth[..., 1:-1, 1:-1] - depth[..., 2:, 1:-1]
    var9 = depth[..., 1:-1, 1:-1] - depth[..., 2:, 2:]
    
    return torch.cat((var1, var2, var3, var4, var6, var7, var8, var9), 1)


def occlusion_aware_loss(depth_gt, depth_pred, occlusion, th=1.):
    """
    Compute a distance between depth maps using the occlusion orientation
    :param depth_gt: (B, 1, H, W)
    :param depth_pred: (B, 1, H, W)
    :param occlusion: (B, 9, H, W)
    """
    # get neighborhood depth variation in (B, 8, H-2, W-2)
    depth_var = neighbor_depth_variation(depth_pred)
    orientation = occlusion[:, 1:, 1:-1, 1:-1]

    fg_mask = (orientation == 1).float()
    bg_mask = (orientation == -1).float()

    fg_loss = (depth_var + th).relu() * fg_mask
    bg_loss = (-depth_var + th).relu() * bg_mask
    loss_avg = fg_loss.sum() / fg_mask.sum() + bg_loss.sum() / bg_mask.sum()

    return loss_avg

