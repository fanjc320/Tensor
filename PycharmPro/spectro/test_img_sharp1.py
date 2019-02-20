import numpy as np, wave,math
# import matplotlib.cbook as cbook
# from matplotlib import docstring
# from matplotlib.path import Path
import matplotlib.pyplot as plt 
# import math
from skimage import io,data,filters
import test_spec
import test_huadongtiao

# img=data.astronaut()
# img_sob1 = sobelEdge(img,sobel_1)
# imshow(img_sob1)


# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

# 从谐波中提取一条线
def OneLine(image):
    image = image[0:200] # 舍弃部分高频部分
    return  image


def imgConvolve(image, kernel):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)
    # print("img_h:",img_h,"img_w:",img_w)
    # print("kernel:", kernel_h,"kernel shape0:", kernel.shape[0],"padding_h:", padding_h,"padding_w:",padding_w)
    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # print("img_padding:",img_padding)
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            # print("i - paddign_h:",i - padding_h,j - padding_w)
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * kernel))
            # print("res:",img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1])
            # print("sum:",np.sum(img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve

def imgYuzhi(image,yuzhi = 20):
    image[image < yuzhi] = 0
    return image

# 检查左值连续性
def imgConv_fjc(image,yuzhi = 40):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    image_convolve = np.zeros(image.shape)
    image_weight = np.zeros([image_convolve.shape[0],image_convolve.shape[1],1])
    image = imgYuzhi(image)

    for i in range(0, img_h):
        for j in range(0, img_w):
            x_size = 50 #取左面多少个像素
            y_size= 20
            curpoint = image[i][j] # 当前点的灰度
            if i<x_size:x_size=i
            if j<y_size:y_size=j
            # curpoint_left = image[i][j-1]
            curpoint_around = np.mean(image[i - x_size:i+x_size , j - y_size:j + y_size])
            print("curpoint:",curpoint,curpoint_around)
            if curpoint_around >= yuzhi :
                image_convolve[i][j]=curpoint
                image_weight[i][j] +=1
                print("i,j:",i,j)

    return image_convolve

y_w = 10
x_w = 10
yuzhi = 40
# 检查峰值 y 方向,即每列的峰值，# x_w 每个采样列的宽度
def imgPeak_fjc(image):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    image_convolve = np.zeros(image.shape)
    image_weight = np.zeros([image_convolve.shape[0],image_convolve.shape[1],1])
    image = imgYuzhi(image)

    # x_size = 50  # 取左面多少个像素
    # y_size = 20
    for j in range(0, img_w,y_w):
        for i in range(0, img_h,x_w):
            curpoint = image[i][j] # 当前点的灰度
            # if i<x_size:x_size=i
            # if j<y_size:y_size=j
            # curpoint_left = image[i][j-1]
            curpoint_around = np.mean(image[i:i+x_w , j:j + y_w])
            # print("curpoint:",curpoint,curpoint_around)
            if curpoint_around >= yuzhi :
                image_convolve[i][j]=curpoint
                image_weight[i][j] +=1
                # print("i,j:",i,j)


    return image_convolve


def Huadong_backCall(x):
    print("Huadong_backCall",x)
    x_w = x

def Huadong_backCall1(x):
    print("Huadong_backCall",x)
    y_w = x

def Huadong_backCall2(x):
    print("Huadong_backCall",x)
    yuzhi = x

# sobel 算子
sobel_1 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

def TestConv(img,sob):
    # img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.createTrackbar('R', 'image', 0, 255, Huadong_backCall)
    cv2.createTrackbar('G', 'image', 0, 255, Huadong_backCall1)
    cv2.createTrackbar('B', 'image', 0, 255, Huadong_backCall2)

    while (True):
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('image', img)
        # 获取滑动条的值
        max_val = cv2.getTrackbarPos('R', 'image')
        min_val = cv2.getTrackbarPos('G', 'image')

        edges = cv2.Canny(img, min_val, max_val)
        print("edges shape:",edges.shape)
        cv2.imshow('window2', edges)

        img_sobel = imgPeak_fjc(img)
        # plt.imshow(img_sobel, cmap="Greys")
        print("img_sobel shape:",img_sobel.shape)
        cv2.imshow('window1', img)

        # 按下ESC键退出
        if cv2.waitKey(30) == 27:
            break

    plt.figure('TestConv', figsize=(8, 8))
    plt.subplot(121)
    # plt.imshow(img, plt.cm.gray)
    plt.imshow(img, cmap="Greys")
    # img_sobel = imgConvolve(img,sob)
    # img_sobel = imgConv_fjc(img)
    img_sobel = imgPeak_fjc(img)
    plt.subplot(122)
    plt.imshow(img_sobel,cmap="Greys")
    plt.show()

# img = data.camera()
img = test_spec.GetWavData()
img = np.clip(img,0,255) # 归一化
# 因为opencv读取文件默认CV_8U类型，在做完卷积后会转化为CV_32FC1类型的矩阵来提高精度或者避免舍入误差。需要clip之后转换为np.uint8。
img = np.array(img,np.uint8)

# 对角线
# img = np.eye(5,4,3)
# img = np.array(np.arange(12).reshape(3,4))

# imgsize = 10
# img = np.zeros([imgsize,imgsize])
# line = np.ones(imgsize)*100
# col = (np.ones(imgsize+1)*100)
# img = np.insert(img,5,line,0)
# img = np.insert(img,5,col,1)

sob = sobel_1
# sob = np.array([[0,0,0],
#        [1,1,1],
#        [0,0,0]])
print("img.shape:",img)

test_huadongtiao.Huadong(Huadong_backCall,Huadong_backCall1,Huadong_backCall2)
TestConv(img,sob)

# while True:
#     if cv2.waitKey(1) == 27:
#         print("---------------------------------------")
#         TestConv(img,sob)



# sob[sob>0]=2
# print(sob)

