# 2019/1/20
# 如果想要修改输入图像的size: loader：output_size fcrn：上采样size
import os
import numpy as np
import h5py
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from m_utils import load_split

#output_size = (360, 450)
output_size = (720, 900)
#output_size_dpt =  (720, 900)

# FireWorkDataSet
class FireWork_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type='train'):
        txt_path = ''
        if type == 'train':
            txt_path = data_path + '/shape3_train_path.txt'
        if type == 'test':
            txt_path = data_path + '/shape3_test_path.txt'
        fh = open(txt_path, 'r')
        imgs = []
        dpts = []
        for line in fh:
            if line is not None:
                line = line.rstrip()
                words = line.split()
                imgs.append(words[0])
                dpts.append(words[1])

        self.imgs = imgs
        self.dpts = dpts

    def __getitem__(self, index):
        img_path = self.imgs[index]
        dpt_path = self.dpts[index]

        img = Image.open(img_path).convert('RGB')
        # img = get_binarization_img(img, 3)  # 二值化效果不好反而影响了最后的精度

        dpt = Image.open(dpt_path).convert('L')

        img_transform = transforms.Compose([
            transforms.Resize(output_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        dpt_transform = transforms.Compose([
            transforms.Resize(output_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])


        img = img_transform(img)
        dpt = dpt_transform(dpt)
        dpt = scale(dpt)
        return img, dpt

    def __len__(self):
        return len(self.imgs)

# 返回二值化图像, img:待处理的rgb图像, channel:需要返回图像的通道数
def get_binarization_img(img, channel=1):
    limg = img.convert('L')  # 转换成单通道的灰度图
    threshold = 4
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    bimg = limg.point(table, "1")  # 二值化，自定义阀值写入table
    bimg_L = bimg.convert('L')  # 将二值化后的图像转化成灰度图才能拼接
    if channel == 1:
        return bimg_L
    bimg_3 = Image.merge('RGB', (bimg_L, bimg_L, bimg_L))  # 将二值化图像拼接成三通道
    return bimg_3

# 将深度图缩放10倍,深度的范围就是（0-10m)进一步操作（0.02-10.02）
def scale(depth):
    ratio = torch.FloatTensor([10.0])
    return ratio * depth

# 返回FireWork数据集
def getFireWorkDataset(batch_size=16, data_path='../Firework_Dataset'):
    train_set = FireWork_Dataset(data_path=data_path, type='train')
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True)

    test_set = FireWork_Dataset(data_path=data_path, type='test')
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


class NYU_Dataset(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0) # HWC
        dpt = self.dpts[img_idx].transpose(1, 0)
        img = Image.fromarray(img)
        dpt = Image.fromarray(dpt)
        input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])

        target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])

        img = input_transform(img)
        dpt = target_depth_transform(dpt)
        return img, dpt

    def __len__(self):
        return len(self.lists)

def getNYUDataset(batch_size=16, data_path='../NYU_Dataset/nyu_depth_v2_labeled.mat'):
    train_lists, test_lists = load_split()

    train_set = NYU_Dataset(data_path=data_path, lists=train_lists)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # val_set = NYU_Dataset(data_path=data_path, lists=val_lists)
    # val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

    test_set = NYU_Dataset(data_path=data_path, lists=test_lists)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


# 测试数据加载,OpenCv以彩色图片的形式加载的图片是BGR模式。但是在Matplotlib中,是以RGB的模式加载的。
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
def test_loader():
    max_depth = 10.0
    print("Loading data...")
    # train_loader, test_loader = getNYUDataset()
    train_loader, test_loader = getFireWorkDataset()
    for imgs, dpts in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            dpts = dpts.cuda()

        img = imgs[0].data.cpu().permute(1, 2, 0).numpy()
        plt.imshow(img)
        #plt.show()

        dpt = dpts[0][0].data.cpu().numpy()
        dpt /= max_depth
        # 深度图的可视化
        plt.imshow(dpt, cmap=matplotlib.cm.jet)
        plt.show()

        # 保存RGBA图像也就是原始的深度图
        dpt = dpt * 255.0
        cv.imwrite('./test/dpt.png', dpt)
        print(dpt.shape)

        print(imgs.size())
        print(dpts.size())
        print(len(train_loader))

        #plt.imsave('./data/dpt1.png', dpt)
        #plt.imshow(dpt)
        #plt.show()
        # cv2.waitKey(0)
        break

    '''
    img_cv = imgs[0].data.cpu()
        t0 = img_cv[0].clone()
        t2 = img_cv[2].clone()
        img_cv[0] = t2
        img_cv[2] = t0
        img_cv = img_cv.permute(1, 2, 0).numpy() * 255.0
        print(img_cv.shape)
        cv.imwrite('./test/img_cv.png', img_cv)
    for input, depth in train_loader:
        print(input.size())
        # input_rgb_image = input[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        input_rgb_image = input[0].data.cpu().permute(1, 2, 0)
        input_gt_depth_image = depth[0][0].data.cpu().numpy().astype(np.float32)

        input_gt_depth_image /= np.max(input_gt_depth_image)
        plt.imshow(input_rgb_image)
        plt.show()
        plt.imshow(input_gt_depth_image, cmap="viridis")
        plt.show()
        # plot.imsave('input_rgb_epoch_0.png', input_rgb_image)
        # plot.imsave('gt_depth_epoch_0.png', input_gt_depth_image, cmap="viridis")
        break
    '''

if __name__ == '__main__':
    test_loader()

















