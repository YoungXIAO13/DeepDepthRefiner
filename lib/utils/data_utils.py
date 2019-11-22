import numpy as np


def neighbor_depth_variation(depth, diagonal=np.sqrt(2)):
    """Compute the variation of depth values in the neighborhood-8 of each pixel"""
    depth_crop = depth[1:-1, 1:-1, :]
    var1 = (depth_crop - depth[:-2, :-2, :]) / diagonal
    var2 = depth_crop - depth[:-2, 1:-1, :]
    var3 = (depth_crop - depth[:-2, 2:, :]) / diagonal
    var4 = depth_crop - depth[1:-1, :-2, :]
    var6 = depth_crop - depth[1:-1, 2:, :]
    var7 = (depth_crop - depth[2:, :-2, :]) / diagonal
    var8 = depth_crop - depth[2:, 1:-1, :]
    var9 = (depth_crop - depth[2:, 2:, :]) / diagonal

    return np.concatenate((var1, var2, var3, var4, var6, var7, var8, var9), -1)


def compute_tangent_adjusted_depth(depth_p, normal_p, depth_q, normal_q, eps=1e-3):
    # compute the depth map for the middl point
    depth_m = (depth_p + depth_q) / 2

    # compute the tangent-adjusted depth map for p and q
    ratio_p = np.linalg.norm(depth_p * normal_p, axis=-1, keepdims=True) / (
                np.linalg.norm(depth_m * normal_p, axis=-1, keepdims=True) + eps)
    depth_p_tangent = np.linalg.norm(depth_m * ratio_p, axis=-1, keepdims=True)

    ratio_q = np.linalg.norm(depth_q * normal_q, axis=-1, keepdims=True) / (
                np.linalg.norm(depth_m * normal_q, axis=-1, keepdims=True) + eps)
    depth_q_tangent = np.linalg.norm(depth_m * ratio_q, axis=-1, keepdims=True)

    return depth_p_tangent - depth_q_tangent


def neighbor_depth_variation_tangent(depth, normal, diagonal=np.sqrt(2)):
    """Compute the variation of tangent-adjusted depth values in the neighborhood-8 of each pixel"""
    depth_crop = depth[1:-1, 1:-1, :]
    normal_crop = normal[1:-1, 1:-1, :]
    var1 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[:-2, :-2, :], normal[:-2, :-2, :]) / diagonal
    var2 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[:-2, 1:-1, :], normal[:-2, 1:-1, :])
    var3 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[:-2, 2:, :], normal[:-2, 2:, :]) / diagonal
    var4 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[1:-1, :-2, :], normal[1:-1, :-2, :])
    var6 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[1:-1, 2:, :], normal[1:-1, 2:, :])
    var7 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[2:, :-2, :], normal[2:, :-2, :]) / diagonal
    var8 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[2:, 1:-1, :], normal[2:, 1:-1, :])
    var9 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[2:, 2:, :], normal[2:, 2:, :]) / diagonal

    return np.concatenate((var1, var2, var3, var4, var6, var7, var8, var9), -1)


def normalize_depth_map(depth):
    pred_normalized = depth.copy().astype('f')
    pred_normalized[pred_normalized == 0] = np.nan
    pred_normalized = pred_normalized - np.nanmin(pred_normalized)
    pred_normalized = pred_normalized / np.nanmax(pred_normalized)
    return pred_normalized
