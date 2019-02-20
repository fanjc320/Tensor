import numpy as np, wave,math
# import matplotlib.cbook as cbook
# from matplotlib import docstring
# from matplotlib.path import Path
import matplotlib.pyplot as plt 
# import math
from skimage import io,data,filters
import test_spec
import  cv2


def track_back(x):
    pass

def TestCanny_huadong(img):
    # img = cv2.imread('sudoku.jpg', 0)
    cv2.namedWindow('window')

    # 创建滑动条
    cv2.createTrackbar('maxVal', 'window', 100, 255, track_back)
    cv2.createTrackbar('minVal', 'window', 200, 255, track_back)

    while(True):
        # 获取滑动条的值
        max_val = cv2.getTrackbarPos('maxVal', 'window')
        min_val = cv2.getTrackbarPos('minVal', 'window')

        edges = cv2.Canny(img, min_val, max_val) # 在maxVal~=75 ,minVal=255时效果最好
        # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # edges = cv2.Canny(thresh, min_val, max_val)
        cv2.imshow('window', edges)

        # 按下ESC键退出
        if cv2.waitKey(30) == 27:
            break


# cv2.Canny()进行边缘检测，参数2、3表示最低、高阈值，看完后面的理论就理解了。
# 经验之谈：之前我们用低通滤波的方式模糊了图片，那如果反过来，想得到物体的边缘，就需要用到高通滤波。如果你要理解接下来要说的Canny检测原理，请先阅读：番外篇：图像梯度

def TestCanny(img):
    edges = cv2.Canny(img, 30, 70)
    cv2.imshow('canny', np.hstack((img, edges)))
    cv2.waitKey(0)

    # 2.先阈值，后边缘检测
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 30, 70)

    cv2.imshow('canny', np.hstack((img, thresh, edges)))
    cv2.waitKey(0)

# img = data.camera()
img = test_spec.GetWavData()
# src = cv2.imread(img,cv2.CV_8UC1)
img = np.clip(img,0,255) # 归一化
# 因为opencv读取文件默认CV_8U类型，在做完卷积后会转化为CV_32FC1类型的矩阵来提高精度或者避免舍入误差。需要clip之后转换为np.uint8。
img = np.array(img,np.uint8)
TestCanny(img)
# TestCanny_huadong(img)