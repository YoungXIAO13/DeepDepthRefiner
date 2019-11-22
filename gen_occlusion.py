from lib.utils.net_utils import create_gamma_matrix
from lib.utils.data_utils import neighbor_depth_variation_tangent, neighbor_depth_variation, normalize_depth_map
import numpy as np
import cv2
import os
from skimage import feature

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt


thresh = 200 / 1000
out_dir = '/space_sdd/ibims/yang_contour'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


# load the depth map and the normal map
depth_path = '/space_sdd/ibims/ibims1_core_raw/depth/factory_03.png'
normal_path = '/space_sdd/ibims/ibims1_core_raw_raycasting_15_15/factory_03-normal.png'
depth = cv2.imread(depth_path, -1) / (2 ** 16 - 1) * 50
depth_z = depth[..., np.newaxis]  # [H, W, 1]
normal = cv2.imread(normal_path, -1) / (2 ** 16 - 1) * 2 - 1
normal = normal[:, :, ::-1]  # [H, W ,3]


# create point-to-point depth map from plane-to-plane depth map
gamma = create_gamma_matrix(480, 640, 560, 560)
delta_x = depth_z / np.tan(gamma[:, :, 0][..., np.newaxis])
delta_y = depth_z / np.tan(gamma[:, :, 1][..., np.newaxis])
depth_point = np.concatenate((delta_x, delta_y, depth_z), -1)


# get depth variation for original depth map of size [H-2, W-2, 8]
depth_point_norm = np.linalg.norm(depth_point, axis=-1, keepdims=True)
depth_var_point = neighbor_depth_variation(depth_point_norm)

# get depth variation for tangent-adjuster depth map of size [H-2, W-2, 8]
depth_var_tangent = neighbor_depth_variation_tangent(depth_point, normal)
print(depth_var_tangent.shape)

# normalize est depth map from 0 to 1
depth_normalized = normalize_depth_map(depth)

# apply canny filter to both depth variations
edges_est = feature.canny(depth_normalized, sigma=np.sqrt(2), low_threshold=0.15, high_threshold=0.3)
plt.imsave(os.path.join(out_dir, 'occlusion.png'), edges_est)

# iteration on all pixel neighborhoods
for i in range(8):
    contour = edges_est.copy()

    # get depth variation in shape of [H, W]
    var_point = np.pad(depth_var_point[:, :, i], ((1, 1), (1, 1)), 'edge')
    var_tangent = np.pad(depth_var_tangent[:, :, i], ((1, 1), (1, 1)), 'edge')

    contour_tangent = np.zeros_like(contour)
    # mask = np.logical_and(contour != 0, var_tangent >= thresh)
    mask = np.logical_and(depth >= thresh, var_tangent >= thresh)
    contour_tangent[mask] = 255

    cv2.imwrite(os.path.join(out_dir, 'variation_corrected_{}.png'.format(i)), var_tangent * 30)
    plt.imsave(os.path.join(out_dir, 'occlusion_corrected_{}.png'.format(i)), contour_tangent)
