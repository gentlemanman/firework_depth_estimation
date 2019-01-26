# 2019/1/18
# 新定义一个数据集的加载， 仅仅加载rgb图像和对应的名字，经过测试模型输出dpt(可视化的效果图)
# 和dpt_cv(通过opencv保存，（0-255）可直接读取出深度的模式)
import os
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
from loader import getNYUDataset, getFireWorkDataset
from metrics import AverageMeter, Result
from m_utils import get_useful_depth

model_path = './run/model_best.pth.tar'
output_dir = './test'
output_size = (720, 900)

# 给定一个待处理的rgb图片文件夹, (rgb,name)
class test_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path='.\\test\\rgb'):
        imgs = []
        labels = []  # 用于记录文件的名称
        for (path, dirnames, filenames) in os.walk(data_path):
            for filename in filenames:
                if filename.endswith('png'):
                    rgb_path = path + '\\' + filename
                    imgs.append(rgb_path)
                    labels.append(int(filename[:-4])) # 获取图片名字

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = self.labels[index]

        img = Image.open(img_path).convert('RGB')

        img_transform = transforms.Compose([
            transforms.Resize(output_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        img = img_transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imgs)

def visualization_test():
    # 数据
    max_depth = 10.0
    batch_size = 2
    data_path = '.\\test\\rgb'
    test_set = test_Dataset(data_path=data_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    # 模型
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    epoch = checkpoint['epoch']
    print('Loading Model Epoch:', epoch)
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()

    for imgs, img_names in test_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            img_names = img_names.cuda()

        # 将tensor数据转化成cpu上
        img = imgs[0].data.cpu()
        img_name = img_names[0].data.cpu().numpy()

        # 预测深度值
        dpts = model(imgs)
        dpt = dpts[0][0].data.cpu().numpy()
        dpt /= max_depth
        dpt = get_useful_depth(img, dpt)
        plt.imsave(output_dir + '/dpt/' + str(img_name) + '.png', dpt, cmap=matplotlib.cm.jet)

        # 保存RGBA图像也就是原始的深度图
        dpt = dpt * 255.0
        cv.imwrite(output_dir + '/dpt_cv/' + str(img_name) + '.png', dpt)

        # 原始RGB图像
        img = img.permute(1, 2, 0).numpy()
        # plt.imsave(output_dir + '/rgb/' + str(names[0]) + '.png', img)


def error_test():
    train_loader, val_loader, test_loader = getNYUDataset()
    # 模型
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    print('模型加载成功')
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    average_meter = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        with torch.no_grad():
            output = model(input)

        gpu_time = time.time() - end

        # 度量
        result = Result()
        result.evaluate(output.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))

        if (i + 1) % 10 == 0:
            print('Validate: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'REL={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(test_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'Rel={average.absrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))


if __name__ == '__main__':
    visualization_test()
    #error_test()



