import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
input_size = (480, 480)

class FireWork_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path = '../Firework_Dataset', type='', shape=''):
        txt_path = ''
        if type == 'train':
            txt_path = data_path + '/' + shape +'_train_path.txt'
        if type == 'test':
            txt_path = data_path + '/' + shape +'_test_path.txt'
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
        # img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path).convert('L')
        # 二值化
        imh = img.convert('1')
        img = crop_720x720(img)

        dpt = Image.open(dpt_path).convert('L')
        dpt = crop_720x720(dpt)

        img_transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        dpt_transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        img = img_transform(img)
        dpt = dpt_transform(dpt)
        dpt = scale(dpt)

        return img, dpt

    def __len__(self):
        return len(self.imgs)

def scale(depth):
    ratio = torch.FloatTensor([255.0])
    return ratio * depth

def crop_720x720(data):
    data = data.crop((90, 0, 810, 720)) ## (left, upper, right, lower)
    return data


def getFireWorkDataset(batch_size=1, shape='shape1'):
    train_set = FireWork_Dataset(type='train', shape=shape)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    test_set = FireWork_Dataset(type='test', shape=shape)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader

import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
def loader_test():
    train_loader, test_loader = getFireWorkDataset(shape='18')
    print('加载数据集成功')
    i = 0
    for imgs, dpts in test_loader:

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            dpts = dpts.cuda()

        # img = imgs[0].data.cpu().permute(1, 2, 0).numpy()
        img = imgs[0][0].data.cpu().numpy()
        print(img.shape)
        plt.imshow(img)
        # plt.show()

        dpt = dpts[0][0].data.cpu().numpy()
        dpt /= 255.0
        plt.imshow(dpt, cmap=matplotlib.cm.jet)
        # plt.show()

        dpt *= 255.0
        cv.imwrite('dpt.png', dpt)
        src = cv.imread('dpt.png')
        cv.imshow('dpt', src)
        cv.waitKey(0)


if __name__ == '__main__':
    loader_test()
