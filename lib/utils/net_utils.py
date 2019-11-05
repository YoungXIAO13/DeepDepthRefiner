import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import atan, tan, pi
import itertools


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


def create_gamma_matrix(H=480, W=640, fx=600, fy=600):
    fov_x = 2 * atan(W / (2 * fx))
    fov_y = 2 * atan(H / (2 * fy))
    gamma = np.zeros((H, W, 2))

    for i, j in itertools.product(range(H), range(W)):
        alpha_x = (pi - fov_x) / 2
        gamma_x = alpha_x + fov_x * ((W - j) / W)

        alpha_y = (pi - fov_y) / 2
        gamma_y = alpha_y + fov_y * ((H - i) / H)

        gamma[i, j, 0] = gamma_x
        gamma[i, j, 1] = gamma_y

    return gamma


def log_smooth_l1_loss(pred, target, log=True):
    valid_mask = (pred != 0) * (target != 0)
    pred = pred[valid_mask]
    target = target[valid_mask]
    if log:
        pred = pred.log()
        target = target.log()
    loss = F.smooth_l1_loss(pred, target)
    return loss


def neighbor_depth_variation(depth):
    """Compute the variation of depth values in the neighborhood-8 of each pixel"""
    var1 = (depth[..., 1:-1, 1:-1] - depth[..., :-2, :-2]) / np.sqrt(2)
    var2 = depth[..., 1:-1, 1:-1] - depth[..., :-2, 1:-1]
    var3 = (depth[..., 1:-1, 1:-1] - depth[..., :-2, 2:]) / np.sqrt(2)
    var4 = depth[..., 1:-1, 1:-1] - depth[..., 1:-1, :-2]
    var6 = depth[..., 1:-1, 1:-1] - depth[..., 1:-1, 2:]
    var7 = (depth[..., 1:-1, 1:-1] - depth[..., 2:, :-2]) / np.sqrt(2)
    var8 = depth[..., 1:-1, 1:-1] - depth[..., 2:, 1:-1]
    var9 = (depth[..., 1:-1, 1:-1] - depth[..., 2:, 2:]) / np.sqrt(2)
    
    return torch.cat((var1, var2, var3, var4, var6, var7, var8, var9), 1)


def compute_tangent_adjusted_depth(depth_p, normal_p, depth_q, normal_q, eps=1e-3):
    # compute the depth map for the middl point
    depth_m = (depth_p + depth_q) / 2

    # compute the tangent-adjusted depth map for p and q
    ratio_p = (depth_p * normal_p).norm(dim=1, keepdim=True) / ((depth_m * normal_p).norm(dim=1, keepdim=True) + eps)
    depth_p_tangent = (depth_m * ratio_p).norm(dim=1, keepdim=True)

    ratio_q = (depth_q * normal_q).norm(dim=1, keepdim=True) / ((depth_m * normal_q).norm(dim=1, keepdim=True) + eps)
    depth_q_tangent = (depth_m * ratio_q).norm(dim=1, keepdim=True)

    return depth_p_tangent - depth_q_tangent


def neighbor_depth_variation_tangent(depth, normal):
    """Compute the variation of tangent-adjusted depth values in the neighborhood-8 of each pixel"""
    depth_crop = depth[..., 1:-1, 1:-1]
    normal_crop = normal[..., 1:-1, 1:-1]
    var1 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., :-2, :-2], normal[..., :-2, :-2]) / np.sqrt(2)
    var2 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., :-2, 1:-1], normal[..., :-2, 1:-1])
    var3 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., :-2, 2:], normal[..., :-2, 2:]) / np.sqrt(2)
    var4 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 1:-1, :-2], normal[..., 1:-1, :-2])
    var6 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 1:-1, 2:], normal[..., 1:-1, 2:])
    var7 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 2:, :-2], normal[..., 2:, :-2]) / np.sqrt(2)
    var8 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 2:, 1:-1], normal[..., 2:, 1:-1])
    var9 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 2:, 2:], normal[..., 2:, 2:]) / np.sqrt(2) 
    
    return torch.cat((var1, var2, var3, var4, var6, var7, var8, var9), 1)


def occlusion_aware_loss(depth_gt, depth_pred, occlusion, normal, gamma, th=1.):
    """
    Compute a distance between depth maps using the occlusion orientation
    :param depth_gt: (B, 1, H, W)
    :param depth_pred: (B, 1, H, W)
    :param occlusion: (B, 9, H, W)
    :param normal: (B, 3, H, W)
    :param gamma: (H, W, 2)
    """
    # change plane2plane depth map to point2point depth map
    delta_x = depth_pred / gamma[:, :, 0].tan()
    delta_y = depth_pred / gamma[:, :, 1].tan()
    depth_point = torch.cat((delta_x, delta_y, depth_pred), 1)

    # get neighborhood depth variation in (B, 8, H-2, W-2)
    depth_point_norm = depth_point.norm(dim=1, keepdim=True)
    depth_var_point = neighbor_depth_variation(depth_point_norm)
    depth_var_tangent = neighbor_depth_variation_tangent(depth_point, normal)
    adjust_mask = (depth_var_tangent > 0).float()
    keep_mask = (depth_var_tangent <= 0).float()
    depth_var_correct = (depth_var_point < depth_var_tangent).float() * depth_var_point + (depth_var_point >= depth_var_tangent).float() * depth_var_tangent
    depth_var = depth_var_correct * adjust_mask + depth_var_point * keep_mask

    # get masks in (B, 8, H-2, W-2)
    orientation = occlusion[:, 1:, 1:-1, 1:-1]
    fn_fg_mask = ((orientation == 1) * (depth_var > -th)).float()
    fn_bg_mask = ((orientation == -1) * (depth_var < th)).float()
    fp_fg_mask = ((orientation != 1) * (depth_var < -th)).float()
    fp_bg_mask = ((orientation != -1) * (depth_var > th)).float()

    # compute the loss for the four situations
    fn_fg_loss = ((depth_var + th).relu() * fn_fg_mask).sum() / (fn_fg_mask.sum() + 1)
    fn_bg_loss = ((-depth_var + th).relu() * fn_bg_mask).sum() / (fn_bg_mask.sum() + 1)
    fp_fg_loss = ((-depth_var - th).relu() * fp_fg_mask).sum() / (fp_fg_mask.sum() + 1)
    fp_bg_loss = ((-depth_var - th).relu() * fp_bg_mask).sum() / (fp_bg_mask.sum() + 1)

    loss_avg = fn_fg_loss + fn_bg_loss + fp_fg_loss + fp_bg_loss
    return loss_avg

