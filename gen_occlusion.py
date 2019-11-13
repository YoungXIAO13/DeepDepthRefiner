from lib.utils.net_utils import create_gamma_matrix, neighbor_depth_variation_tangent, neighbor_depth_variation
import torch
import numpy as np
import cv2

depth_path = '/space_sdd/ibims/ibims1_core_raw/depth/factory_03.png'
normal_path = '/space_sdd/ibims/ibims1_core_raw_raycasting_15_15/factory_03-normal.png'
invalid_path = '/space_sdd/ibims/ibims1_core_raw/zero_depth_mask/factory_03.png'

depth = cv2.imread(depth_path, -1) / (2 ** 16 - 1) * 50
normal = cv2.imread(normal_path, -1) / (2 ** 16 - 1) * 2 - 1
normal = normal[:, :, ::-1]
invalid_mask = cv2.imread(invalid_path, -1)

depth = torch.from_numpy(np.ascontiguousarray(depth)).float().unsqueeze(0).unsqueeze(0)
normal = torch.from_numpy(np.ascontiguousarray(normal)).float().permute(2, 0, 1).unsqueeze(0)

gamma = create_gamma_matrix(480, 640, 560, 560)
gamma = torch.from_numpy(gamma).float()

delta_x = depth / gamma[:, :, 0].tan()
delta_y = depth / gamma[:, :, 1].tan()
depth_point = torch.cat((delta_x, delta_y, depth), 1)

depth_point_norm = depth_point.norm(dim=1, keepdim=True)

depth_var_point = neighbor_depth_variation(depth_point_norm, np.sqrt(2)).squeeze()

depth_var_tangent = neighbor_depth_variation_tangent(depth_point, normal, np.sqrt(2)).squeeze()

print('depth point max = {}'.format(depth_var_point.max()))
print('depth tangent max = {}'.format(depth_var_tangent.max()))

delta = 30 / 1000
contour_fg = (depth_var_point > delta).float() * (depth_var_tangent > delta).float()
contour_bg = (depth_var_point < -delta).float() * (depth_var_tangent < -delta).float()

contour = (contour_fg + contour_bg).numpy().astype('uint8') * 255

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

plt.imsave('depth.png', depth.squeeze().numpy() / 50 * 255)
plt.imsave('depth_point.png', depth_point_norm.squeeze().numpy() / 50 * 255)
for i in range(contour.shape[0]):
    plt.imsave('occlucsion_{}.png'.format(i), contour[i, :, :])
    #plt.imsave('occlucsion_tangent_{}.png'.format(i), contour_tangent[i, :, :].numpy())
    #np.save('depth_var_{}.ply'.format(i), depth_var_point[i, :, :].abs().numpy())
    #np.save('depth_var_tangent_{}.ply'.format(i), depth_var_tangent[i, :, :].abs().numpy())
