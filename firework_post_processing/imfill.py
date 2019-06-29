import cv2 as cv

# 当前遍历到的像素是否有效
def is_valid(tmp_pix):
    if min_depth < tmp_pix and tmp_pix < max_depth:
        return True
    return False

def imfill(img, ks=1, type='mid'):
    print('空洞填充：' + type)
    rows = img.shape[0]
    cols = img.shape[1]
    tmp = img.copy() # 备份一个临时的图片

    for i in range(rows):
        for j in range(cols):
         # 1.当前像素所表示深度是无效, 看是否要填补空洞
         if tmp[i][j][0] < min_depth:
             # 设置核边界
             top, bottom, left, right = max(0, i - ks), min(i + ks, rows - 1), max(0, j - ks), min(j + ks,cols - 1)
             count = 0
             useful_pix = []
             for m in range(top, bottom + 1):
                 for n in range(left, right + 1):
                     if is_valid(tmp[m][n][0]) is True:
                         count = count + 1
                         useful_pix.append(tmp[m][n][0])

             if count > 4: # 有效像素多余1个
                 ans = 0
                 if type == 'avg': # 均值填充
                    for val in useful_pix:
                        ans = ans + val
                    ans = ans//count
                 if type == 'mid': # 中值填充
                     ans = useful_pix[count//2]
                 img[i][j] = [ans, ans, ans]
             else:
                 img[i][j] = [0, 0, 0]


min_depth = 30
max_depth = 220
ks = 1 # kernel_size
def main():
    print('空洞填充测试：')
    for i in range(1):
        img_path = './data/original/'+ str(i) + '.png'
        img = cv.imread(img_path)
        imfill(img) # 滤波
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
