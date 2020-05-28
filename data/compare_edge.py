import os
import numpy as np
import cv2
import math

from scipy import io
from skimage import feature
from scipy import ndimage
from tqdm import tqdm


def canny_edge(depth, th_low=0.15, th_high=0.3):
    # normalize est depth map from 0 to 1
    depth_normalized = depth.copy().astype('f')
    depth_normalized[depth_normalized == 0] = np.nan
    depth_normalized = depth_normalized - np.nanmin(depth_normalized)
    depth_normalized = depth_normalized / np.nanmax(depth_normalized)
    
    edge_est = feature.canny(depth_normalized, sigma=np.sqrt(2), low_threshold=th_low, high_threshold=th_high)
    return edge_est


def load_gt_edge_and_canny(gt_mat):

    # load ground truth depth
    image_data = io.loadmat(gt_mat)
    data = image_data['data']

    # extract neccessary data
    depth = data['depth'][0][0]  # Raw depth map
    mask_invalid = data['mask_invalid'][0][0]  # Mask for invalid pixels
    mask_transp = data['mask_transp'][0][0]  # Mask for transparent pixels
    edge = data['edges'][0][0]
    
    # calculate the canny edge
    edge_est = canny_edge(depth)
    
    return edge, edge_est, mask_invalid * mask_transp


def compute_distance(edges_gt, edges_est):
    # compute distance transform for chamfer metric
    D_gt = ndimage.distance_transform_edt(1 - edges_gt)
    D_est = ndimage.distance_transform_edt(1 - edges_est)

    max_dist_thr = 10.  # Threshold for local neighborhood

    mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

    E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges

    if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be detected in prediction
        dbe_acc = max_dist_thr
        dbe_com = max_dist_thr
    else:
        # accuracy: directed chamfer distance of predicted edges towards gt edges
        dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)

        # completeness: sum of undirected chamfer distances of predicted and gt edges
        ch1 = D_gt * edges_est  # dist(predicted,gt)
        ch1[ch1 > max_dist_thr] = max_dist_thr  # truncate distances
        ch2 = D_est * edges_gt  # dist(gt, predicted)
        ch2[ch2 > max_dist_thr] = max_dist_thr  # truncate distances
        res = ch1 + ch2  # summed distances
        dbe_com = np.nansum(res) / (np.nansum(edges_est) + np.nansum(edges_gt))  # normalized

    return dbe_acc, dbe_com



dbe_acc_canny = []
dbe_com_canny = []
dbe_acc_occ = []
dbe_com_occ = []


## iBims-1
# root_dir = '/space_sdd/ibims'
# gt_dir = 'gt_depth'
# ours_dir = 'our_edge'
# with open(os.path.join(root_dir, 'imagelist.txt')) as f:
#     image_names = f.readlines()
# im_names = [x.strip() for x in image_names]


# for im_name in tqdm(im_names):
#     gt_mat = os.path.join(root_dir, gt_dir, '{}.mat'.format(im_name))
    
#     edge_gt, edge_est, mask = load_gt_edge_and_canny(gt_mat)
    
#     dbe_acc, dbe_com = compute_distance(edge_gt, edge_est)
#     dbe_acc_canny.append(dbe_acc)
#     dbe_com_canny.append(dbe_com)
    
#     edge_ours = cv2.imread(os.path.join(root_dir, ours_dir, '{}-edge_fg.png'.format(im_name)))
#     edge_ours = edge_ours[:, :, 0] / 255.
#     dbe_acc, dbe_com = compute_distance(edge_gt, edge_ours)
#     dbe_acc_occ.append(dbe_acc)
#     dbe_com_occ.append(dbe_com)


## NYUv2
eigen_crop = [21, 461, 25, 617]

gt_depths = np.load('/space_sdd/NYU/nyuv2_depth.npy')[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
gt_edges = np.load('/space_sdd/NYU/nyuv2_boundary.npy')[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]

pred_dir = '/home/xuchong/Projects/occ_edge_order/data/dataset_real/NYUv2/data/val_occ_order_raycasting_DynThesh002'

im_names = sorted([name.split('-')[0] for name in os.listdir(pred_dir) if '-edge_fg.png' in name])

for i in tqdm(range(len(im_names))):
    # load our generated edge
    edge_ours = cv2.imread(os.path.join(pred_dir, '{}-edge_fg.png'.format(im_names[i])))[:, :, 0] / 255
    
    # load gt_depth and generate canny edge
    edge_gt = gt_edges[i]
    depth_gt = gt_depths[i]
    edge_canny = canny_edge(depth_gt, 0.01, 0.1)
    
    dbe_acc, dbe_com = compute_distance(edge_gt, edge_canny)
    dbe_acc_canny.append(dbe_acc)
    dbe_com_canny.append(dbe_com)
    
    dbe_acc, dbe_com = compute_distance(edge_gt, edge_ours)
    dbe_acc_occ.append(dbe_acc)
    dbe_com_occ.append(dbe_com)

    
print("for canny detected edges we have")
print('acc={}, com={}'.format(np.mean(dbe_acc_canny), np.mean(dbe_com_canny)))

print("for occlusion edges we have")
print('acc={}, com={}'.format(np.mean(dbe_acc_occ), np.mean(dbe_com_occ)))

    