import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt


# 将ground_truth图像变换到网络size, 并新建文件夹保存
def change_size(cur_path = './model_test', data_path='./model_test/ground_truth', img_size=(480, 480)):
    rgb_interpolation_path = os.path.join(cur_path, 'ground_truth_' + str(img_size[0]) + 'x' + str(img_size[1]))
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

def crop_720x720(data):
    data = data.crop((90, 0, 810, 720)) ## (left, upper, right, lower)
    return data

if __name__ == '__main__':
    change_size(data_path='')