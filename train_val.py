import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os, sys
import time

sys.path.append('./lib/')UNet
from lib.datasets.ibims import Ibims
from lib.datasets.interiro_net import InteriorNet

from lib.utils.net_utils import kaiming_init, save_checkpoint, load_checkpoint, \
    log_smooth_l1_loss, occlusion_aware_loss
from lib.utils.evaluate_ibims_error_metrics import compute_distance_related_errors, compute_global_errors

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# network training procedure settings
parser.add_argument('--alpha', type=float, default=1., help='weight balance')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')
parser.add_argument('--step', type=int, default=50, help='epoch to decrease')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--print_freq', type=int, default=100, help='frequence of output print')

# pth settings
parser.add_argument('--session', type=int, default=0, help='training session')
parser.add_argument('--resume', action='store_true', help='resume checkpoint or not')
parser.add_argument('--checkpoint', type=str, default=None, help='optional reload model path')
parser.add_argument('--save_dir', type=str, default='model', help='save model path')

# dataset settings
parser.add_argument('--train_dir', type=str, default='/space_sdd/InteriorNet', help='training dataset')
parser.add_argument('--train_method', type=str, default='sharpnet_pred')
parser.add_argument('--val_dir', type=str, default='/space_sdd/ibims', help='testing dataset')
parser.add_argument('--val_method', type=str, default='junli')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# =================CREATE DATASET=========================== #
dataset_train = InteriorNet(opt.train_dir, method_name=opt.train_method)
dataset_val = Ibims(opt.val_dir, opt.val_method)

train_loader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=opt.workers)
# ========================================================== #


# ================CREATE NETWORK AND OPTIMIZER============== #
net = UNet()
net.apply(kaiming_init)

optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0.0005)
lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, [opt.step], gamma=0.1)

if opt.resume:
    load_checkpoint(net, optimizer, opt.checkpoint)

net.cuda()
# ========================================================== #


# =============DEFINE stuff for logs ======================= #
result_path = os.path.join(os.getcwd(), opt.save_dir)
if not os.path.exists(result_path):
    os.makedirs(result_path)
logname = os.path.join(result_path, 'train_log.txt')
with open(logname, 'a') as f:
    f.write(str(opt) + '\n' + '\n')
    f.write('training set: ' + str(len(dataset_train)) + '\n')
    f.write('validation set: ' + str(len(dataset_val)) + '\n')
# ========================================================== #


# =================== DEFINE TRAIN ========================= #
def train(data_loader, net, optimizer):
    net.train()
    end = time.time()
    for i, data in enumerate(data_loader):
        # load data and label
        depth_gt, depth_coarse, occlusion = data
        depth_gt, depth_coarse, occlusion = depth_gt.cuda(), depth_pred.cuda(), occlusion.cuda()

        # forward pass
        depth_pred = net(occlusion, depth_coarse)

        # compute losses and update the meters
        loss_depth_all = log_smooth_l1_loss(depth_pred, depth_gt)
        loss_depth_occ = occlusion_aware_loss(depth_gt, depth_pred, occlusion, 15. / 1000)
        loss = loss_depth_all + opt.alpha * loss_depth_occ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure bacth time
        batch_time = time.time() - end
        end = time.time()

        if (i + 1) % opt.print_freq == 0:
            print("\tEpoch %3d --- Iter [%d/%d] Train loss: %.2f || Batch time: %.2f" %
                  (epoch, i + 1, len(data_loader), loss.item(), batch_time))
# ========================================================== #


def val(data_loader, net, optimizer):
    # Initialize global and geometric errors ...
    num_samples = len(data_loader)
    rms     = np.zeros(num_samples, np.float32)
    log10   = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    thr1    = np.zeros(num_samples, np.float32)
    thr2    = np.zeros(num_samples, np.float32)
    thr3    = np.zeros(num_samples, np.float32)

    net.val()
    for i, data in enumerate(data_loader):
        # load data and label
        depth_gt, depth_coarse, occlusion = data
        depth_gt, depth_coarse, occlusion = depth_gt.cuda(), depth_pred.cuda(), occlusion.cuda()

        # forward pass
        depth_pred = net(occlusion, depth_coarse)

        # get numpy array from torch tensor
        gt = depth_gt.unsqueeze().cpu().numpy()
        pred = depth_pred.unsqueeze().cpu().numpy()

        gt_vec = gt.flatten()
        pred_vec = pred.flatten()

        abs_rel[i], sq_rel[i], rms[i], log10[i], thr1[i], thr2[i], thr3[i] = compute_global_errors(gt_vec,pred_vec)
    
    return abs_rel, sq_rel, rms, log10, thr1, thr2, thr3



# =============BEGIN OF THE LEARNING LOOP=================== #
# initialization
abs_rel, sq_rel, rms, log10, thr1, thr2, thr3 = val(val_loader, net, optimizer)
print('############ Global Error Metrics #################')
print('rel    = ',  np.nanmean(abs_rel))
print('sq_rel = ',  np.nanmean(sq_rel))
print('log10  = ',  np.nanmean(log10))
print('rms    = ',  np.nanmean(rms))
print('thr1   = ',  np.nanmean(thr1))
print('thr2   = ',  np.nanmean(thr2))
print('thr3   = ',  np.nanmean(thr3))
best_rms = np.nanmean(rms)

for epoch in range(opt.n_epoch):
    # update learning rate
    lrScheduler.step()

    # train
    train(train_loader, net, optimizer)    

    # valuate
    abs_rel, sq_rel, rms, log10, thr1, thr2, thr3 = val(val_loader, net, optimizer)

    # log testing reults
    with open(logname, 'a') as f:
        f.write('Results for {} epoch:\n'.format(epoch))
        f.write('rel    =  {}'.format(np.nanmean(abs_rel)))
        f.write('sq_rel =  {}'.format(np.nanmean(sq_rel)))
        f.write('log10  =  {}'.format(np.nanmean(log10)))
        f.write('rms    =  {}'.format(np.nanmean(rms)))
        f.write('thr1   =  {}'.format(np.nanmean(thr1)))
        f.write('thr2   =  {}'.format(np.nanmean(thr2)))
        f.write('thr3   =  {}'.format(np.nanmean(thr3)))

    # update best_rms and save checkpoint
    is_best = np.nanmean(rms) < best_rms
    best_rms = min(np.nanmean(rms), best_rms)
    save_checkpoint({
        'epoch': epoch,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(result_path, 'checkpoint_{}_{:4f}.pth'.format(epoch, best_rms)))
