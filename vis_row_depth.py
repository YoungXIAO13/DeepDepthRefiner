import argparse
import os
from scipy.io import loadmat
import numpy as np
import cv2

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/xuchong/Projects/occ_edge_order/data/dataset_real/NYUv2/data/val_occ_order_raycasting_woNormal_avgROI_1mm')
parser.add_argument('--gt_depth', type=str, default='/space_sdd/NYU/nyuv2_depth.npy')
parser.add_argument('--refine_dir', type=str,
                    default='/space_sdd/NYU/depth_refine/depth1_grad1_occ0.1_change1_1e-5/eigen/depth_npy')

opt = parser.parse_args()

# load rgb list
img_list = sorted([name for name in os.listdir(opt.data_dir) if name.endswith("-rgb.png")])

# load gt depth
gt_depths = np.load(opt.gt_depth)

# load initial depth map list
def read_eigen():
    ours = loadmat('/space_sdd/NYU/depth_predictions/eigen_nyud_depth_predictions.mat')
    ours = ours['fine_predictions']
    ours = ours.transpose((2, 0, 1))
    out = []
    for line in ours:
        line = cv2.resize(line, (640, 480))
        out.append(line)
    out = np.array(out)
    return out
init_depths = read_eigen()

# load refined depth map list
refine_list = sorted(os.listdir(opt.refine_dir))

eigen_crop = [21, 461, 25, 617]
index = 120
row = 300

img = cv2.imread(os.path.join(opt.data_dir, img_list[index]), -1)
print('img shape is {}'.format(img.shape))

gt_depth = gt_depths[index][21:461, 25:617]
print('gt depth shape is {}'.format(gt_depth.shape))

init_depth = init_depths[index][21:461, 25:617]
print('init depth shape is {}'.format(init_depth.shape))

refine_depth = np.load(os.path.join(opt.refine_dir, refine_list[index]))[21:461, 25:617]
print('refine depth shape is {}'.format(refine_depth.shape))


# draw the figure
fig, (ax1, ax2) = plt.subplots(nrows=2)
img[row - 3: row + 3, :, :] = (img[row - 3: row + 3, :, :] + 255) / 2
ax1.imshow(img)

t = np.arange(592)
ax2.plot(t, gt_depth[row, t], 'r-', t, init_depth[row, t], 'b-', t, refine_depth[row, t], 'g-')

asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
ax2.set_aspect(asp)

fig.savefig('vis_row_depth.eps')
plt.close(fig)

