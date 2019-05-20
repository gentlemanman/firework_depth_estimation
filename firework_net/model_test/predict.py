import os
import torch
from PIL import Image
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from net.archs import NestedUNet, Arg
from metrics import Result, AverageMeter

model_path = './model_best.pth.tar'
data_path = './rgb'
batch_size = 1
img_size = (480, 480)
cur_path = './'

def predict():
    # change original rgb size
    save_new_rgb(data_path=data_path)
    # data
    data_set = predict_Dataset(data_path=data_path)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, drop_last=True)
    # model
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    epoch = checkpoint['epoch']
    result = checkpoint['best_result']
    # shape = checkpoint['shape']
    shape = 'shape1'
    print('Current Shape: ', shape)
    print('Loading Model Epoch:', epoch)
    print_result(result)

    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    # average_meter = AverageMeter()
    # predict
    print('Start Predicting')
    for i, (input, name) in enumerate(data_loader):
        if torch.cuda.is_available():
            input, name = input.cuda(), name.cuda()
            with torch.no_grad():
                output = model(input)
            # save result
            save_images(input, output, name, i+1, len(data_loader))
    print('End')

# 给定一个待处理的rgb图片文件夹
class predict_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path='./rgb'):
        imgs = []
        labels = [] #用于记录文件的名称
        for (path, dirnames, filenames) in os.walk(data_path):
            for filename in filenames:
                if filename.endswith('png'):
                    rgb_path = path + '/' + filename
                    imgs.append(rgb_path)
                    labels.append(int(filename[:-4])) # 获取图片的名字
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = self.labels[index]

        img = Image.open(img_path).convert('L')
        img = crop_720x720(img)

        img_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        img = img_transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imgs)

def crop_720x720(data):
    data = data.crop((90, 0, 810, 720)) ## (left, upper, right, lower)
    return data

def save_images(input, output, name, idx, length):
    # 建立保存结果的文件夹
    dpt_path = os.path.join(cur_path, 'dpt')
    if os.path.exists(dpt_path) is False:
        os.makedirs(dpt_path)
    dpt_cv_path = os.path.join(cur_path, 'dpt_cv')
    if os.path.exists(dpt_cv_path) is False:
        os.makedirs(dpt_cv_path)

    rgb = input[0][0].data.cpu().numpy()
    name = name[0].data.cpu().numpy()

    pre_dpt = output[0][0].data.cpu().numpy()
    # TODO 剔除无效像素
    pre_dpt = exclude_invalid(rgb, pre_dpt)
    # 用于重建的灰度图
    cv.imwrite(dpt_cv_path + '/' + str(name) + '.png', pre_dpt)
    # 可视化的效果图
    pre_dpt /= 255.0
    plt.imsave(dpt_path + '/' + str(name) + '.png', pre_dpt, cmap=matplotlib.cm.jet)
    # 打印处理的进程
    print(str(idx) + '/' + str(length) + ':  ' + str(name) + '.png')

def print_result(result):
    print('Best Result in Test Dataset: '
          'RMSE={result.rmse:.3f} '
          'REL={result.absrel:.3f} '
          'D1={result.delta1:.3f} '
          'D2={result.delta2:.3f} '
          'D3={result.delta3:.3f}'.format(result=result))

# 将rgb图像变换到网络size保存
def save_new_rgb(data_path='./rgb'):
    rgb_interpolation_path = os.path.join(cur_path, 'rgb_' + str(img_size[0]) + 'x' + str(img_size[1]))
    if os.path.exists(rgb_interpolation_path) is False:
        os.makedirs(rgb_interpolation_path)
    for (path, dirnames, filenames) in os.walk(data_path):
        for filename in filenames:
            if filename.endswith('png'):
                rgb_path = path + '/' + filename
                img = Image.open(rgb_path).convert('RGB')
                img = crop_720x720(img)
                trans = transforms.Compose([transforms.Resize(img_size, interpolation=Image.NEAREST)])
                img = trans(img)
                plt.imsave(rgb_interpolation_path + '/' +filename, img)

# 剔除无效的像素
def exclude_invalid(rgb, pre_dpt):
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if rgb[i][j] < 1e-3:
                pre_dpt[i][j] = 0.
    return pre_dpt


if __name__ == '__main__':
    predict()