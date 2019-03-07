import matplotlib.pyplot as plt
from skimage import io,data,filters
import test_spec
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

# 带连续性的点
class Point_Conti:
    huidu = 0
    i = 0
    j = 0
    LineIndex = 0
    LineV = 0

image_Lines = {(0,0):Point_Conti()}
maxLineVDic = {}
lianXuCnt = 0
# 需要调整的参数 huidu_yuzhi  fangge lianxu_yuzhi,需要有检测最佳结果的方法
# 连续性 这里只要有一个点是段的,就认为连续性中断,以后可以优化为可调节,需要考虑多个点属于一个连续性的问题
def imgLines_fjc(image,huidu_yuzhi=50,fangge =5,lianxu_yuzhi=5):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    image_convolve = np.zeros(image.shape)
    # image_Lines = np.zeros([image_convolve.shape[0],image_convolve.shape[1],1])
    maxLineV = 0
    maxLineV_x_w = 0
    maxLinev_y_w = 0
    maxLinev_yuzhi = 0
    global  maxLineVDic
    cnt_Huidu = 0

    point = Point_Conti()
    test = []
    test.append(point)
    test1 = [Point_Conti()]
    test1.append(point)

    print("point:",point.j,point.i)
    for j in range(fangge, img_w,fangge):
        for i in range(fangge, img_h,fangge):
            cur_huidu = np.sum(image[i:i+fangge, j:j+fangge])
            # # print("cur_huidu:",cur_huidu,np.max(image),np.max(image[10:20,320:340]))
            # iend = i+x_w;jend=j+y_w;
            # if cur_huidu == 0:
            #     continue
            # else:
            #     # image_convolve[i:i+x_w,j:j+y_w] = 200
            #     # continue
            #     tttttt=1
            # print("---------------cur_huidu: i:", cur_huidu,i,j) #, np.max(image), np.max(image[i:iend,j:jend]),i,iend,j,jend)

            # 当前自定义格子的灰度要大于某一阈值
            if cur_huidu < 1:
                print("cur_huidu==0 ",i,j,np.sum(image[i:i+fangge, j:j+fangge]))
                continue
            point_cur = image_Lines.setdefault((i, j), point)
            maxLianxu = 0
            max_x = 1
            max_y = 1
            point = Point_Conti()
            point.i = i
            point.j = j
            # fangge的大小是调节的关键所在，当图像只剩一条线，这个fangge大小就是最佳的
            # 向左面寻找连续性最大的区域
            for x_n in range(-2,2):
                for y_n in range(1,3):
                    if i - x_n * fangge >= 0 and i - x_n * fangge < img_h and j - y_n * fangge >= 0 and j - y_n * fangge < img_w:
                        point_left= image_Lines.setdefault((i-x_n*fangge, j-y_n*fangge),point)
                        lianxu = point_left.LineV
                        if lianxu>maxLianxu:
                            maxLianxu = lianxu
                            max_x = x_n
                            max_y = y_n
            # 左面区域的点的最大连续值
            if maxLianxu>0:
                point_cur.LineV = maxLianxu + 1
                # print("--- LineV: ",point_cur.LineV)
            else:
            # 假如找到的最大连续性的点为0，那么寻找左面灰度最大的区域
                for x_n in range(-2,2):
                    for y_n in range(1,3):
                        if i-x_n*fangge>=0 and i-x_n*fangge<img_h and j-y_n*fangge>=0 and j-y_n*fangge<img_w:
                            point_left = image_Lines.setdefault((i-x_n*fangge, j-y_n*fangge), point)
                            huidu_left = np.sum(image[i-x_n*fangge,j-y_n*fangge])
                            if huidu_left >= huidu_yuzhi:
                                # global lianXuCnt
                                # lianXuCnt=lianXuCnt+1
                                # print("liangxu number:",lianXuCnt)
                                point_cur.LineV = 1 # 左面没有连续值>1的点,那么自己就是连续的初始 # 引用
                                print("---------------- point_left.LineV==:", point_cur.LineV,image_Lines[(i, j)].LineV,"i:",i,"j:",j)


    for (i,j) in image_Lines:
        point = image_Lines.get((i,j),Point_Conti())
        if point.LineV>=lianxu_yuzhi:
            # image_convolve[i:i+fangge,j:j+fangge] = point.huidu
            image_convolve[i:i + fangge, j:j + fangge] = image[i:i+fangge,j:j+fangge]
            # print("i,i+x_w,j,j+j_w huidu: max:",i,j,image[i:i+fangge,j:j+fangge],np.max(image_convolve))
        # else:
        #     print("=== error ================================",point.huidu,point.LineV, i,j)

    return image_convolve,maxLineV ,maxLineVDic.get(maxLineV)

def Huadong_backCall(x):
    pass


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
    cv2.createTrackbar('min', 'image', 0, 255,Huadong_backCall)
    cv2.createTrackbar('max', 'image', 0, 255,Huadong_backCall)
    cv2.createTrackbar('x_w', 'window2', 10, 25,Huadong_backCall)
    cv2.createTrackbar('y_w', 'window2', 10, 25,Huadong_backCall)
    cv2.createTrackbar('yuzhi', 'window2', 0, 55,Huadong_backCall)
    print("max(img):",np.max(img))
    # plt.imshow(img, cmap="Greys")
    # plt.show()
    img[img<100] = 0
    # plt.imshow(img, cmap="Greys")
    # plt.show()
    # plt.hist(img.ravel(), 256)
    # plt.show()
    while (True):
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('image', img)
        # 获取滑动条的值
        max_val = cv2.getTrackbarPos('min', 'image')
        min_val = cv2.getTrackbarPos('max', 'image')
        x_w= cv2.getTrackbarPos('x_w', 'window2')
        y_w = cv2.getTrackbarPos('y_w', 'window2')
        yuzhi = cv2.getTrackbarPos('yuzhi', 'window2')

        edges = cv2.Canny(img, min_val, max_val) # 似乎R255,G0效果挺好
        # print("edges shape:",edges.shape)
        cv2.imshow('window1', edges)
        img_sobel =[]
        maxLineV_final = 0;xw_final =0;yw_final = 0;yuzhi_final=50;
        for huidu_yuzhi in range(15,16):
            for fangge in range(15,16):
                for lianxu_yuzhi in range(2,3):
                    # img_sobel,maxLineV,dic= imgLines_fjc(img,huidu_yuzhi = huidu_yuzhi,fangge=fangge,lianxu_yuzhi=lianxu_yuzhi)
                    img_sobel, maxLineV, dic = imgLines_fjc(img)
                    maxLineV_final = max(maxLineV_final,maxLineV)
                    print("img.shape:",img.shape,type(img),"img_sobel shape:",img_sobel.shape,type(img_sobel),maxLineV)
                    img_sobel = np.clip(img_sobel, 0, 255)  # 归一化
                    img_sobel = np.array(img_sobel, np.uint8)
                    cv2.imshow('window2', img_sobel)

        # img_sobel = np.array(img_sobel)
        # print("img_sobel shape:", img_sobel.shape, type(img_sobel), maxLineV_final)
        plt.imshow(img_sobel,cmap="Greys")
        plt.show()
        break
        # 按下ESC键退出
        if cv2.waitKey(30) == 27:
            break

    # plt.figure('TestConv', figsize=(8, 8))
    # plt.subplot(121)
    # # plt.imshow(img, plt.cm.gray)
    # plt.imshow(img, cmap="Greys")
    # # img_sobel = imgConvolve(img,sob)
    # # img_sobel = imgConv_fjc(img)
    # # img_sobel = imgPeak_fjc(img)
    # img_sobel = imgLines_fjc(img)
    # plt.subplot(122)
    # plt.imshow(img_sobel,cmap="Greys")
    # plt.show()

# img = data.camera()
img = test_spec.GetWavData()
img = np.clip(img,0,255) # 归一化
img = np.array(img,np.uint8)# 因为opencv读取文件默认CV_8U类型，在做完卷积后会转化为CV_32FC1类型的矩阵来提高精度或者避免舍入误差。需要clip之后转换为np.uint8。

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


TestConv(img,sob)


def TestDict():
    dic = {}
    item = dic.get(1,10)
    print(item)
    print(dic.get(1))
    dic["a"] = 100
    print(dic.get("a"))
    item1 = dic.setdefault(2,20)
    item1 = 200
    print(dic.get(2))
    # print(dic["b"])# KeyError: 'b'


def TestSlice():
    test = [[0,1,2,3],
            [4,5,6,7],
            [8,9,10,11]]

    # print("===:",test[2:3,2:3]) # list indices must be integers or slices, not tuple
    print("==:",test[2:3][2:3]) #[]
    # print(test[1:3,1:3]) #list indices must be integers or slices, not tuple
    testa = np.array(test)
    # print("test.shape:",test.shape)#'list' object has no attribute 'shape'
    print("testa.shape:", testa.shape)
    print("===",testa[2:3,2:3])
    print("====",testa[1:3,1:3])

    # a = np.arange(12).reshape([3,4])
    a = np.array(test)
    print(a)
    print("a:",a[1,1])
    print("b:",a[1:,1:])
    # 每个维度可以使用步长跳跃切片
    print("c:",a[::2,1:])
    # 多维数组取步长要用冒号
    print("d:",a[0::2,1:])
    print("e:",a[1:3,1:3])
    print("f:", a[2:3, 2:4],np.sum(a[2:3,2:4]))

# TestSlice()
# TestDict()