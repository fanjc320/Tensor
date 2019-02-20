import numpy as np, wave,math
# import matplotlib.cbook as cbook
# from matplotlib import docstring
# from matplotlib.path import Path
import matplotlib.pyplot as plt 
# import math
from skimage import io,data,filters
import test_spec
import cv2

# 大部分图像处理任务都需要先进行二值化操作，阈值的选取很关键，Otsu阈值法会自动计算阈值。
# Otsu阈值法（日本人大津展之提出的，也可称大津算法）非常适用于双峰图片
# 绘制直方图时，使用了numpy中的ravel()函数，它会将原矩阵压缩成一维数组，便于画直方图。
def TestOtsu(img):
    # 下面这段代码对比了使用固定阈值和Otsu阈值后的不同结果：
    # 另外，对含噪点的图像，先进行滤波操作效果会更好。
    # img = cv2.imread('myplot.png', 0)
    # 固定阈值法
    ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # Otsu阈值法
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 先进行高斯滤波，再使用Otsu阈值法
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 下面我们用Matplotlib把原图、直方图和阈值图都显示出来：
    images = [img, 0, th1, img, 0, th2, blur, 0, th3]
    titles = ['Original', 'Histogram', 'Global(v=100)',
              'Original', 'Histogram', "Otsu's",
              'Gaussian filtered Image', 'Histogram', "Otsu's"]
    for i in range(3):
        # 绘制原图
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3], fontsize=8)
        plt.xticks([]), plt.yticks([])
        # 绘制直方图plt.hist，ravel函数将数组降成一维
        plt.subplot(3, 3, i * 3 + 2)
        plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1], fontsize=8)
        plt.xticks([]), plt.yticks([])
        # 绘制阈值图
        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2], fontsize=8)
        plt.xticks([]), plt.yticks([])
    plt.show()

img = test_spec.GetWavData()
# src = cv2.imread(img,cv2.CV_8UC1)
img = np.clip(img,0,255) # 归一化
# 因为opencv读取文件默认CV_8U类型，在做完卷积后会转化为CV_32FC1类型的矩阵来提高精度或者避免舍入误差。需要clip之后转换为np.uint8。
img = np.array(img,np.uint8)
TestOtsu(img)