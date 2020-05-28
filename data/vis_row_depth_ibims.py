import argparse
import os
import numpy as np
import cv2

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='/space_sdd/ibims/ibims1_core_raw/rgb')
parser.add_argument('--result_dir', type=str, default='result/ibims_depth=1_grad=1_occ=0.1_change=1_noContour_useOcc_1e-5/eigen/pred')

opt = parser.parse_args()

# load rgb list
img_list = sorted([name for name in os.listdir(opt.img_dir) if name.endswith(".png")])

# with open('/space_sdd/ibims/imagelist.txt') as f:
#     image_names = f.readlines()
# image_names = [x.strip() for x in image_names]

index = 23
row = 400

img = cv2.imread(os.path.join(opt.img_dir, img_list[index]), -1)[:, :, ::-1]
print('img shape is {}'.format(img.shape))

gt_depth = np.load(os.path.join(opt.result_dir, '{:04}_gt.npy'.format(index)))
print('gt depth shape is {}'.format(gt_depth.shape))

init_depth = np.load(os.path.join(opt.result_dir, '{:04}_init.npy'.format(index)))
print('init depth shape is {}'.format(init_depth.shape))

refine_depth = np.load(os.path.join(opt.result_dir, '{:04}_refine.npy'.format(index)))
print('refine depth shape is {}'.format(refine_depth.shape))


# draw the figure
fig, (ax1, ax2) = plt.subplots(nrows=2)
img[row - 3: row + 3, :, :] = (img[row - 3: row + 3, :, :] + 255) / 2
ax1.imshow(img)

t = np.arange(640)
lines = ax2.plot(t, gt_depth[row, t], 'r-',
                 t, init_depth[row, t], 'b-',
                 t, refine_depth[row, t], 'g-')
plt.setp(lines, linewidth=0.5)

asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
ax2.set_aspect(asp)

fig.savefig('vis_row_depth.eps')
plt.close(fig)

