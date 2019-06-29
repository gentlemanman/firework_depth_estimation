import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def save_img():
    video_path = './video/'
    video_name = '2.mp4'

    folder_name = './video/' + video_name.split('.')[0]
    os.makedirs(folder_name, exist_ok=True)
    vc = cv2.VideoCapture(video_path + video_name) # 读入视频
    c = 0
    rval = vc.isOpened()

    while rval: # 循环读取视频帧
        rval, frame = vc.read()
        rval, frame = vc.read()
        img_path = folder_name + '/'
        if rval:
            cv2.imwrite(img_path + str(c) + '.png', frame) # 保存图片
            c = c + 1
            cv2.waitKey(1)
        else:
            break
    vc.release()
    print('save success')
    print(folder_name)

# 把从视频获得的图片改成合适大小
def change_size():
    img_path = './video/2'
    new_img_path = img_path + '_480'
    os.makedirs(new_img_path, exist_ok=True)
    for (path, dirnames, filenames) in os.walk(img_path):
        for filename in filenames:
            if filename.endswith('png'):
                rgb_path = path + '/' + filename
                img = Image.open(rgb_path).convert('RGB')
                img = crop_720x720(img)
                trans = transforms.Compose([transforms.Resize((480, 480), interpolation=Image.NEAREST)])
                img = trans(img)
                plt.imsave(new_img_path + '/' + filename, img)

def crop_720x720(data):
    data = data.crop((0, 150, 544, 694))  # (left, upper, right, lower)
    return data

# 重命名
def rename_img():
    img_path = './video/2_rename'
    for (path, dirnames, filenames) in os.walk(img_path):
        for filename in filenames:
            if filename.endswith('png'):
                new_name = int(filename.split('.')[0])
                new_name = new_name - 45
                os.rename(img_path + '/' + filename, img_path + '/' + str(new_name) + '.png')

if __name__ == '__main__':
    # save_img()
    # change_size()
    rename_img()