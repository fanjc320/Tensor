import numpy as np, wave,math
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import io,data,filters
import scipy.signal as signal
import cv2

import numpy as np
import cv2

# 卷积
def imgConvolve(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:卷积后的矩阵
    '''
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    kernel_h = int(kernel.shape[0])
    kernel_w = int(kernel.shape[1])
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_w = int(img_w + 2 * padding_w)
    print("img_h:",img_h,"img_w:",img_w)
    print("kernel:", kernel_h,"kernel shape0:", kernel.shape[0],"padding_h:", padding_h,"padding_w:",padding_w,
          "convolve_h:", convolve_h,"convolve_w:",convolve_w)
    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_w))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve


# 首先我们把图像卷积函数封装在一个名为imconv的函数中  ( 实际上，scipy库中的signal模块含有一个二维卷积的方法convolve2d()  )
def imconv(image_array, suanzi):
    # image = image_array.copy()# 原图像矩阵的深拷贝,dtype是uint8,不能做运算，会溢出,结果不对
    image_z = np.zeros(image_array.shape)# 原图像矩阵的深拷贝
    dim1, dim2 = image_array.shape


    # 对每个元素与算子进行乘积再求和(忽略最外圈边框像素)
    for i in range(1, dim1 - 1):
        for j in range(1, dim2 - 1):
            image_z[i, j] = (image_array[(i - 1):(i + 2), (j - 1):(j + 2)] * suanzi).sum()

    # 由于卷积后灰度值不一定在0-255之间，统一化成0-255
    image_z = image_z * (255.0 / image_z.max())
    # np.set_printoptions(threshold=np.nan) # 似乎没有此转换，图像更清晰
    return image_z

# 均值滤波
def imgAverageFilter(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:均值滤波后的矩阵
    '''
    return imgConvolve(image, kernel) * (1.0 / kernel.size)


# 高斯滤波
def imgGaussian(sigma):
    '''
    :param sigma: σ标准差
    :return: 高斯滤波器的模板
    '''
    img_h = img_w = 2 * sigma + 1
    gaussian_mat = np.zeros((img_h, img_w))
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            gaussian_mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return gaussian_mat


# Sobel Edge
def sobelEdge(image, sobel):
    '''
    :param image: 图片矩阵
    :param sobel: 滤波窗口
    :return:Sobel处理后的矩阵
    '''
    return imgConvolve(image, sobel)


# Prewitt Edge
def prewittEdge(image, prewitt_x, prewitt_y):
    '''
    :param image: 图片矩阵
    :param prewitt_x: 竖直方向
    :param prewitt_y:  水平方向
    :return:处理后的矩阵
    '''
    img_X = imgConvolve(image, prewitt_x)
    img_Y = imgConvolve(image, prewitt_y)

    img_prediction = np.zeros(img_X.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_X[i][j], img_Y[i][j])
    return img_prediction


######################常量################################
# 滤波3x3
kernel_3x3 = np.ones((3, 3))
# 滤波5x5
kernel_5x5 = np.ones((5, 5))

# sobel 算子
sobel_1 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
# prewitt 算子
prewitt_1 = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_2 = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])



# def TestFunc1():
#     # ######################均值滤波################################
#     # 读图片
#     image = cv2.imread('balloonGrayNoisy.jpg', cv2.IMREAD_GRAYSCALE)
#     # 均值滤波
#     img_k3 = imgAverageFilter(image, kernel_3x3)
#
#     # 写图片
#     cv2.imwrite('average_3x3.jpg', img_k3)
#     # 均值滤波
#     img_k5 = imgAverageFilter(image, kernel_5x5)
#     # 写图片
#     cv2.imwrite('average_5x5.jpg', img_k5)
#     ######################高斯滤波################################
#     image = cv2.imread('balloonGrayNoisy.jpg', cv2.IMREAD_GRAYSCALE)
#     img_gaus1 = imgAverageFilter(image, imgGaussian(1))
#     cv2.imwrite('gaussian1.jpg', img_gaus1)
#     img_gaus2 = imgAverageFilter(image, imgGaussian(2))
#     cv2.imwrite('gaussian2.jpg', img_gaus2)
#     img_gaus3 = imgAverageFilter(image, imgGaussian(3))
#     cv2.imwrite('gaussian3.jpg', img_gaus3)
#
#
#     ######################Sobel算子################################
#     image=cv2.imread('buildingGray.jpg',cv2.IMREAD_GRAYSCALE)
#     img_spbel1 = sobelEdge(image, sobel_1)
#     cv2.imwrite('sobel1.jpg',img_spbel1)
#     img_spbel2 = sobelEdge(image, sobel_2)
#     cv2.imwrite('sobel2.jpg',img_spbel2)
#
#     ######################prewitt算子################################
#     img_prewitt1 = prewittEdge(image, prewitt_1,prewitt_2)
#     cv2.imwrite('prewitt1.jpg',img_prewitt1)

def TestSharp(img,sob):
    plt.figure('TestFunc2', figsize=(8, 8))
    # img = data.camera()
    plt.subplot(121)
    plt.imshow(img, plt.cm.gray)
    img_sobel = sobelEdge(img,sob)
    plt.subplot(122)
    plt.imshow(img_sobel,plt.cm.gray)
    plt.show()


###############################################################
# robert 算子[[-1,-1],[1,1]]
def robert_suanzi(img):
    r, c = img.shape
    r_sunnzi = [[-1, -1], [1, 1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x + 2, y:y + 2]
                list_robert = r_sunnzi * imgChild
                img[x, y] = abs(list_robert.sum())  # 求和加绝对值
    return img


# # sobel算子的实现
def sobel_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                                       new_imageY[i + 1, j + 1]) ** 0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image)  # 无方向算子处理的图像


# Laplace算子
# 常用的Laplace算子模板  [[0,1,0],[1,-4,1],[0,1,0]]   [[1,1,1],[1,-8,1],[1,1,1]]
def Laplace_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    for i in range(r - 2):
        for j in range(c - 2):
            new_image[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * L_sunnzi))
    return np.uint8(new_image)


#python自编程序实现——robert算子、sobel算子、Laplace算子进行图像边缘提取
def TestSuanzi():
    # img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    img = data.camera()
    cv2.imshow('image', img)

    # # robers算子
    out_robert = robert_suanzi(img)
    cv2.imshow('out_robert_image', out_robert)

    # sobel 算子
    out_sobel = sobel_suanzi(img)
    cv2.imshow('out_sobel_image', out_sobel)

    # Laplace算子
    out_laplace = Laplace_suanzi(img)
    cv2.imshow('out_laplace_image', out_laplace)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Testimconv():
    # x方向的Sobel算子
    suanzi_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])


    

    suanzi_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
						 
	# image = cv2.imread('pika.png', 0)
    image = cv2.imread('./resource/wall.png', 0)
    image_array = np.array(image)
    image_x = imconv(image_array, suanzi_x)
    image_y = imconv(image_array, suanzi_y)

    # 为了使看清边缘检测结果，将大于灰度平均值的灰度变成255(白色)
    print("mean:",image_x.mean(),image_y.mean(),image_x.max())
    # image_y[image_y> image_y.mean()] = 255
    image_x[image_x>150]= 255
    image_x[image_x<100]= 255
    # image_y[image_y> 200] = 255
    # image_y[image_y< 100] = 255

    plt.subplot(2, 2, 1)
    plt.imshow(image_array, cmap=cm.gray)
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.hist(image_y.ravel(),256)
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(image_x, cmap=cm.gray)
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(image_y, cmap=cm.gray)
    plt.axis("off")
    plt.show()


# 高斯算子 降噪
def Testimconv_jiangzao(image):
    # 生成高斯算子的函数
    def func(x, y, sigma=1):
        return 100 * (1 / (2 * np.pi * sigma)) * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma ** 2))

    # 生成标准差为5的5*5高斯算子
    suanzi1 = np.fromfunction(func, (5, 5), sigma=5)

    # Laplace扩展算子
    suanzi2 = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]])

    # 打开图像并转化成灰度图像
    # image = data.camera()
    image_array = np.array(image)

    # 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
    image_blur = signal.convolve2d(image_array, suanzi1, mode="same")

    # 对平滑后的图像进行边缘检测
    image2 = signal.convolve2d(image_blur, suanzi2, mode="same")

    # 结果转化到0-255
    image2 = (image2 / float(image2.max())) * 255

    # 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
    image2[image2 > image2.mean()] = 255

    # 显示图像
    plt.subplot(2, 1, 1)
    plt.imshow(image_array, cmap=cm.gray)
    plt.axis("off")
    plt.subplot(2, 1, 2)
    plt.imshow(image2, cmap=cm.gray)
    plt.axis("off")
    plt.show()

# TestSharp(data.camera(),sobel_2)
# TestSuanzi()

Testimconv()

# Testimconv_jiangzao(data.astronaut()[:,:,0])



