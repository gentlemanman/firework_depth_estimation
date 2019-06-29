import os
import cv2 as cv
from imfill import imfill

# 当前遍历到的像素是否有效
def is_valid(cur_pix, tmp_pix):
    threshold = 5
    if tmp_pix < min_depth and tmp_pix > max_depth:
        return False
    if abs(int(cur_pix) - int(tmp_pix)) > threshold:  #图像像素值是ubyte类型，ubyte类型数据范围为0~255, 强制为整形
        return False
    return True

def filter(img, ks=1, type='avg'):
    print('执行滤波：' + type)
    rows = img.shape[0]
    cols = img.shape[1]
    tmp = img.copy() # 备份一个临时的图片, 一定要用copy操作

    for i in range(rows):
        for j in range(cols):
         # 1.当前像素所表示深度是无效
         if tmp[i][j][0] < min_depth and tmp[i][j][0] > max_depth:
             img[i][j] = [0, 0, 0]
             tmp[i][j] = [0, 0, 0]
         # 2.当前像素深度，进行均值滤波
         else:
             # 设置核边界
             top, bottom, left, right = max(0, i - ks), min(i + ks, rows - 1), max(0, j - ks), min(j + ks,cols - 1)
             cur_pix = tmp[i][j][0]
             count = 0
             useful_pix = []
             for m in range(top, bottom + 1):
                 for n in range(left, right + 1):
                     if is_valid(cur_pix, tmp[m][n][0]) is True:
                         count = count + 1
                         useful_pix.append(tmp[m][n][0])

             if count > 2: # 有效像素多余1个
                 ans = 0
                 if type == 'avg': # 均值滤波
                    for val in useful_pix:
                        ans = ans + val
                    ans = ans // count
                 if type == 'mid': # 中值滤波
                     ans = useful_pix[count//2]
                 img[i][j] = [ans, ans, ans]
             else:
                 img[i][j] = [0, 0, 0]


min_depth = 30
max_depth = 220
ks = 1 # kernel_size
def main():
    input_folder = './data/original/real2'
    output_folder = input_folder.replace('original', 'post')
    os.makedirs(output_folder, exist_ok=True)
    for i in range(30):
        img_path = input_folder + '/'+ str(i) + '.png'
        img = cv.imread(img_path)
        # imfill(img, ks=1, type='mid') # 空洞填充
        filter(img, ks=1, type='avg') # 滤波

        cv.imwrite(img_path.replace('original', 'post'), img)

    cv.namedWindow("img", cv.WINDOW_NORMAL)
    cv.imshow("img", img)
    print('size:', img.shape)
    print('type:', img.dtype)
    print('End')
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
