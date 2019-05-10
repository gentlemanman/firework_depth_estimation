import os
import shutil
import torch
import math
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

class loss_mse(nn.Module):
    def __init__(self):
        super(loss_mse, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, truth):
        loss = self.mse(pred, truth)
        return loss

class loss_l1(nn.Module):
    def __init__(self):
        super(loss_l1, self).__init__()
        self.l1 = nn.L1Loss()
        self.w1 = 0.9
        self.w2 = 0.1

    def forward(self, pred, truth):
        valid_mask = (data > 1e-5)
        loss = self.w1 * self.l1(pred[valid_mask], truth[valid_mask]) + self.w2 * self.l1(pred[1 - valid_mask], truth[1 - valid_mask])
        return loss

# smooth l1
class loss_smooth_l1(nn.Module):
    def __init__(self):
        super(loss_smooth_l1, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred, truth):
        # mask0 = (truth  < 1e-5)
        loss = self.smooth_l1(pred, truth)
        return loss

def update_lr(optimizer, init_lr, epoch, ratio=0.):
    if epoch == 0:
        delta = init_lr - init_lr / 100.
        lr = init_lr / 100. + ratio * delta
    elif epoch > 0 and epoch <= 10:
        lr = init_lr
    elif epoch > 10 and epoch <= 20:
        lr = init_lr / 10.
    else:
        lr = init_lr / 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def write_log(best_txt, result, shape, epoch):
    with open(best_txt, 'w') as txt_file:
        txt_file.write(
            "shape={}\nepoch={}\nrmse={:.3f}\nrel={:.3f}\nlog10={:.3f}\nd1={:.3f}\nd2={:.3f}\ndd31={:.3f}\nt_gpu={:.4f}\n".
            format(shape, epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                   result.delta3,
                   result.gpu_time))

def save_image(rgbs, dpts, pre_dpts, path, idx):
    max_depth = 255.0
    # 原始rgb
    rgb = rgbs[0][0].data.cpu().numpy()

    # Ground Truth
    dpt = dpts[0][0].data.cpu().numpy()

    # dpt = unitization(dpt)
    dpt /= max_depth
    plt.imsave(path + '/' + str(idx) + '_dpt.png', dpt, cmap=matplotlib.cm.jet)

    # 预测值
    pre_dpt = pre_dpts[0][0].data.cpu().numpy()
    u_pre_dpt = pre_dpt / max_depth
    # TODO 剔除无效像素
    plt.imsave(path + '/' + str(idx) + '_dpt_pred.png', u_pre_dpt, cmap=matplotlib.cm.jet)

    # 用于重建的灰度图
    pre_dpt = pre_dpt
    cv.imwrite(path + '/' + str(idx) + '_dpt_pred_cv.png', pre_dpt)

    # 保存原始rgb
    # rgb = rgb.permute(1, 2, 0).numpy()
    plt.imsave(path + '/' + str(idx) + '_rgb.png', rgb)

# ->(0, 1)
def unitization(data):
     max_val = np.max(data)
     min_val = np.min(data)
     tmp = (data - min_val) / (max_val - min_val)
     return tmp

def loss_test():
    loss_fn = loss_mse()
    x = torch.zeros(1, 1, 2, 2)
    y = torch.ones(1, 1, 2, 2)
    r = loss_fn(x, y)
    print(r)

if __name__ == '__main__':
    data = [[1, 2, 2, 3, 7], [1, 2, 3, 10, 11]]
    unitization(data)