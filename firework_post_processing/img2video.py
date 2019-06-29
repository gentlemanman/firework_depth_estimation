import cv2

def saveVideo():
    img_path = './img/real2/'
    img = cv2.imread(img_path + '45.png')
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    print(size)

    videoWrite = cv2.VideoWriter('./img/2.mp4', -1, 15, size)

    for i in range(45, 75):
        fileName = img_path + str(i) +'.png'
        img = cv2.imread(fileName)
        videoWrite.write(img)

    print('end')

saveVideo()