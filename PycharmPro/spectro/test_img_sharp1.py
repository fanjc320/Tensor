import numpy as np, wave,math
# import matplotlib.cbook as cbook
# from matplotlib import docstring
# from matplotlib.path import Path
import matplotlib.pyplot as plt 
# import math
from skimage import io,data,filters
import test_spec

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

def imgConv_fjc(image, kernel,yuzhi = 50):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    image_convolve = np.zeros(image.shape)
    image = imgYuzhi(image)

    for i in range(0, img_h):
        for j in range(0, img_w):
            x_size = 1
            y_size= 0 #取左面100个像素
            curpoint = image[i][j] # 当前点的灰度
            if i<x_size:x_size=i
            if j<y_size:y_size=j
            curpoint_left = image[i][j-1]
            # curpoint_left = np.mean(image[i - x_size:i+x_size , j - y_size:j + y_size])
            print("curpoint:",curpoint,curpoint_left)
            if curpoint >= yuzhi and curpoint_left >= yuzhi:
                image_convolve[i][j]=curpoint
                print("i,j:",i,j)

    return image_convolve

# sobel 算子
sobel_1 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

def TestConv(img,sob):
    plt.figure('TestConv', figsize=(8, 8))
    plt.subplot(121)
    # plt.imshow(img, plt.cm.gray)
    plt.imshow(img, cmap="Greys")
    img_sobel = imgConvolve(img,sob)
    # img_sobel = imgConv_fjc(img, sob)
    plt.subplot(122)
    plt.imshow(img_sobel,cmap="Greys")
    plt.show()

# img = data.camera()
img = test_spec.GetWavData()

# 对角线
# img = np.eye(5,4,3)
# img = np.array(np.arange(12).reshape(3,4))

# imgsize = 50
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
TestConv(img,sob)

# sob[sob>0]=2
# print(sob)

