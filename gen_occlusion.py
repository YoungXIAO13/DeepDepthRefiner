from lib.utils.net_utils import create_gamma_matrix, neighbor_depth_variation_tangent, neighbor_depth_variation
import torch
import numpy as np
import cv2

depth_path = '/space_sdd/ibims/ibims1_core_raw/depth/factory_03.png'
normal_path = '/space_sdd/ibims/ibims1_core_raw/normal/factory_03.png'

depth = cv2.imread(depth_path, -1).astype(np.float) / (2 ** 16 - 1) * 50
normal = cv2.imread(normal_path, -1) / (2 ** 16 - 1) * 2 - 1
normal = normal[:, :, ::-1]

depth = torch.from_numpy(np.ascontiguousarray(depth)).float().unsqueeze(0).unsqueeze(0)
normal = torch.from_numpy(np.ascontiguousarray(normal)).float().permute(2, 0, 1).unsqueeze(0)

print(depth.max())

gamma = create_gamma_matrix(480, 640, 560, 560)
gamma = torch.from_numpy(gamma).float()

delta_x = depth / gamma[:, :, 0].tan()
delta_y = depth / gamma[:, :, 1].tan()
depth_point = torch.cat((delta_x, delta_y, depth), 1)

print(depth_point.shape)

depth_point_norm = depth_point.norm(dim=1, keepdim=True)
depth_var_point = neighbor_depth_variation(depth_point_norm, np.sqrt(2))

depth_var_tangent = neighbor_depth_variation_tangent(depth_point, normal, np.sqrt(2))

print('depth point max = {}'.format(depth_var_point.max()))
print('depth tangent max = {}'.format(depth_var_tangent.max()))

delta = 15
contour_point = depth_var_point.abs() > delta
contour_tangent = depth_var_tangent.abs() > delta

contour = (contour_point * contour_tangent).float().squeeze()

print(contour.shape)

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

for i in range(contour.shape[0]):
    plt.imsave('occlu_{}.png'.format(i), contour[i, :, :].numpy())
