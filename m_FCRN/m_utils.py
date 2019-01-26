# 2019/1/20
# get_useful_depth()函数利用输入的rgb图像剔除深度图像中的无效像素
import os
import torch
import torch.nn as nn
import shutil
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

cmap = plt.cm.jet

# 更新学习率
def update_lr(optimiter, init_lr, epoch, ratio):
    if epoch == 0:
        delta = init_lr - init_lr / 100.0
        lr = init_lr / 100.0 + ratio * delta
    if epoch > 0 and epoch <= 10:
        lr = init_lr
    if epoch > 10 and epoch <= 20:
        lr = init_lr / 10.0
    if epoch > 20:
        lr = init_lr / 100.0
    for param_group in optimiter.param_groups:
        param_group['lr'] = lr
    return lr


# 自定义损失函数
class loss_huber(nn.Module):
    def __init__(self):
        super(loss_huber,self).__init__()

    def forward(self, pred, truth):
        c = pred.shape[1] #通道
        h = pred.shape[2] #高
        w = pred.shape[3] #宽
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)
        # 根据当前batch所有像素计算阈值
        t = 0.2 * torch.max(torch.abs(pred - truth))
        # 计算L1范数
        l1 = torch.mean(torch.mean(torch.abs(pred - truth), 1), 0)
        # 计算论文中的L2
        l2 = torch.mean(torch.mean(((pred - truth)**2 + t**2) / t / 2, 1), 0)

        if l1 > t:
            return l2
        else:
            return l1

class loss_mse(nn.Module):
    def __init__(self):
        super(loss_mse, self).__init__()
    def forward(self, pred, truth):
        c = pred.shape[1]
        h = pred.shape[2]
        w = pred.shape[3]
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)

        return torch.mean(torch.mean((pred - truth), 1)**2, 0)

class loss_mse_alone(nn.Module):
    def __init__(self):
        super(loss_mse_alone, self).__init__()
        self.loss = 0.0
    def forward(self, pred, truth):
        c = pred.shape[1]
        h = pred.shape[2]
        w = pred.shape[3]
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)

        mask0 = (truth  > 0.001)
        mask1 = (truth <= 0.001)

        error_0 = pred[mask0] - truth[mask0]
        error_1 = pred[mask1] - truth[mask1]
        if len(error_0) < 1:
            error_0 = error_1

        self.loss = torch.mean(torch.clamp((error_0) ** 2, min=1e-7, max=1e7)) + \
            torch.mean(torch.clamp((error_1) ** 2, min=1e-7, max=1e7))

        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

if __name__ == '__main__':
    loss = loss_huber()
    x = torch.zeros(2, 1, 2, 2)
    y = torch.ones(2, 1, 2, 2)
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    r = loss(x, y)
    print(r)


# 加载数据集的index
def load_split():
    # current_directory = os.getcwd()
    current_directory = '../NYU_Dataset'
    train_lists_path = current_directory + '/trainIdxs.txt'
    test_lists_path = current_directory + '/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    val_start_idx = int(len(train_lists) * 0.8)

    val_lists = train_lists[val_start_idx:-1]
    # train_lists = train_lists[0:val_start_idx]

    #return train_lists, val_lists, test_lists
    return train_lists, test_lists


# 保存检查点
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

# 保存结果图片
def save_image(imgs, dpts, pre_dpts, path, step):
    max_depth = 10.0
    img = imgs[0].data.cpu()

    # Ground Truth
    dpt = dpts[0][0].data.cpu().numpy()
    # dpt /= np.max(dpt)
    dpt /= max_depth
    plt.imsave(path + '/' + str(step) + '_dpt.png', dpt, cmap=matplotlib.cm.jet)

    # 预测值
    pre_dpt = pre_dpts[0][0].data.cpu().numpy()
    # pre_dpt /= np.max(pre_dpt)
    pre_dpt /= max_depth
    pre_dpt = get_useful_depth(img, pre_dpt)# 剔除无效像素
    plt.imsave(path + '/' + str(step) + '_dpt_pred.png', pre_dpt, cmap=matplotlib.cm.jet)

    # 原始RGB图像
    img = img.permute(1, 2, 0).numpy()
    plt.imsave(path + '/' + str(step) + '_img.png', img)

# 输出的深度图与原始RGB图像对比，去除无效的部分
def get_useful_depth(img, dpt):
    img = img[0].squeeze() # 取出rgb第一维的
    # print(dpt.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 1e-3:
                dpt[i][j] = 0.
    return dpt

'''
# 测试网络
def validate(model, val_loader, loss_fn, dtype):
    # validate
    model.eval()
    num_correct, num_samples = 0, 0
    loss_local = 0
    with torch.no_grad():
        for input, depth in val_loader:
            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)
            if num_epochs == epoch + 1:
                # 关于保存的测试图片可以参考 loader 的写法
                # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                input_rgb_image = input[0].data.permute(1, 2, 0)
                input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
                pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

                input_gt_depth_image /= np.max(input_gt_depth_image)
                pred_depth_image /= np.max(pred_depth_image)

                plot.imsave('./result/input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
                plot.imsave('./result/gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image,
                            cmap="viridis")
                plot.imsave('./result/pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image,
                            cmap="viridis")

            loss_local += loss_fn(output, depth_var)

            num_samples += 1

    err = float(loss_local) / num_samples
    print('val_error: %f' % err)
'''
