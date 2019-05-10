import os
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import shutil
import numpy as np
import torchvision.transforms as transforms

# 生成训练数据的路径文本
# 注意：1.生成训练还是测试数据的路径; 2.txt往后追加路径; 3.共4个形状,需要改动shape1,2,3,4
def generate_path():
    data_path = '..\\Firework_Dataset\shape1\\test_data\\Output_rgb'
    txt_path = '..\\Firework_Dataset\\shape1_test_path.txt'
    file = open(txt_path, mode='a') # 向txt文件中追加

    for (path, dirnames, filenames) in os.walk(data_path):
        for filename in filenames:
            if filename.endswith('png'):
                rgb_path = path + '\\' + filename
                # print('当前路径 %s' % rgb_path)
                example = int(rgb_path.split('\\')[5][3:]) #当前example
                if  example <= 1:
                    print(example)
                    #print(rgb_path)
                    depth_path = rgb_path.replace('rgb', 'depth')
                    #print(depth_path)
                    line = rgb_path + ' ' + depth_path + '\n'
                    file.write(line)
    #file.write('test\n')
    file.close()

# 给定一张ground_truth图像变成可视化效果
def get_post_dpt():
    max_depth = 10.0
    img_path = "./test/ground_truth/0.png"
    output_path = "./test/ground_truth/_0.png"

    img = Image.open(img_path).convert('L')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img[0].data.numpy()
    img /= max_depth
    print(img.shape)
    # plt.imshow(img, cmap=matplotlib.cm.jet)
    plt.show()
    plt.imsave(output_path, img, cmap=matplotlib.cm.jet)

# 给当前文件夹下的文件进行去重和重命名
def rename_file():
    data_path = '..\\firework_databuffer\\shape4\\depth'
    count = 0
    for(path, dirnames, filenames) in os.walk(data_path):
        for filename in filenames:
            if filename.endswith('png'):
                file_path = path + '\\' + filename
                num = int(filename.split('.')[0])
                if num % 2 == 0: # 保留偶数命名的文件
                    print(file_path)
                    os.rename(file_path, path  + '\\' + str(count) + '.png')
                    count += 1
                else:
                    os.remove(file_path)

import math
if __name__ == '__main__':
    # generate_path()
    # get_post_dpt()
    # rename_file()
    # print(pow(math.e, 1000))
    # a = 1.0 / (pow(math.e, 0) + pow(math.e, 1) + pow(math.e, 3))

    # a = 1 - 0.75 * (0.389975 + 0.5283)
    print(a)

