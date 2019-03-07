import matplotlib.pyplot as plt
from skimage import io,data,filters
import test_spec
import cv2
import numpy as np
import math


def imgYuzhi(image,yuzhi = 20):
    image[image < yuzhi] = 0
    return image

def Huadong_backCall(x):
    pass


# 带连续性的点
class Point_Conti:
    huidu = 0
    i = 0
    j = 0
    LineIndex = 0
    Beginx =0
    Beginy =0
    LineV = 0

image_Lines = {}
maxLineVDic = {}
lianXuCnt = 0
# 需要调整的参数 huidu_yuzhi  fangge lianxu_yuzhi,需要有检测最佳结果的方法
# 连续性 这里只要有一个点是段的,就认为连续性中断,以后可以优化为可调节,需要考虑多个点属于一个连续性的问题
def imgLines_fjc(image,huidu_yuzhi=50,fangge =5,lianxu_yuzhi=5):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    image_convolve = np.zeros(image.shape)
    image_beginLianXu = np.zeros(image.shape) # 图片连续性开始计数的点
    # image_Lines = np.zeros([image_convolve.shape[0],image_convolve.shape[1],1])
    maxLineV = 0
    lineIndex = 0
    global  maxLineVDic

    point = Point_Conti()
    test = []
    test.append(point)
    test1 = [Point_Conti()]
    test1.append(point)

    print("point:",point.j,point.i)
    for j in range(fangge, img_w,fangge):
        for i in range(fangge, img_h,fangge):
            cur_huidu = np.sum(image[i:i+fangge, j:j+fangge])

            # 当前自定义格子的灰度要大于某一阈值
            if cur_huidu < huidu_yuzhi:
                print("cur_huidu==0 ",i,j,np.sum(image[i:i+fangge, j:j+fangge]))
                continue
            point_cur = image_Lines.setdefault((i, j), point)
            maxLianxu = 0

            beginX = 0
            beginY = 0
            point = Point_Conti()
            point.i = i
            point.j = j
            # fangge的大小是调节的关键所在，当图像只剩一条线，这个fangge大小就是最佳的
            # 向左面寻找连续性最大的区域
            for x_n in range(-2,2):
                for y_n in range(1,2):
                    if i - x_n * fangge >= 0 and i - x_n * fangge < img_h and j - y_n * fangge >= 0 and j - y_n * fangge < img_w:
                        point_left= image_Lines.setdefault((i-x_n*fangge, j-y_n*fangge),point)
                        lianxu = point_left.LineV
                        if lianxu>maxLianxu:
                            maxLianxu = lianxu
                            lineIndex = point_left.LineIndex
                            beginX = point_left.Beginx
                            beginY = point_left.Beginy

            # 左面区域的点的最大连续值
            if maxLianxu>0:
                point_cur.LineIndex =lineIndex
                point_cur.Beginx = beginX
                point_cur.Beginy = beginY
                point_cur.LineV = maxLianxu + 1
                # print("--- LineV: ",point_cur.LineV)
            else:
            # 假如找到的最大连续性的点为0，那么寻找左面灰度最大的区域
                for x_n in range(-1,1):
                    for y_n in range(1,2):
                        if i-x_n*fangge>=0 and i-x_n*fangge<img_h and j-y_n*fangge>=0 and j-y_n*fangge<img_w:
                            point_left = image_Lines.setdefault((i-x_n*fangge, j-y_n*fangge), point)
                            huidu_left = image[i - x_n * fangge, j - y_n * fangge]
                            # huidu_left = np.average(image[i-x_n*fangge:i,j-y_n*fangge:j])
                            if huidu_left >= huidu_yuzhi:
                                # global lianXuCnt
                                # lianXuCnt=lianXuCnt+1
                                # print("liangxu number:",lianXuCnt)
                                point_cur.LineIndex = i
                                point_cur.Beginx = i
                                point_cur.Beginy = j
                                point_cur.LineV = 1 # 左面没有连续值>1的点,那么自己就是连续的初始 # 引用
                                image_beginLianXu[i:i+img_h,j:j+img_w]=200
                                print("---------------- point_left.LineV==:", point_cur.LineV,image_Lines[(i, j)].LineV,"i:",i,"j:",j)

    mapIndex_cnt = {}
    mapXY_cnt = {}
    mostk=0 # 连续性最大的LineIndex
    mostv=0 # 连续性最大的值
    for (i,j) in image_Lines: # 考虑用map函数代替
        point = image_Lines.get((i, j), Point_Conti())
        cnt = mapIndex_cnt.get(point.LineIndex)
        bx = point.Beginx
        by = point.Beginy
        if cnt ==None:
            mapIndex_cnt.setdefault(point.LineIndex, 0)
        else:
            mapIndex_cnt[point.LineIndex]+=1

        xy = mapXY_cnt.get((bx,by))
        if xy ==None:
            mapXY_cnt.setdefault((bx,by),0)
        else:
            mapXY_cnt[(bx,by)]+=1

    for k in mapIndex_cnt:
        print("kkkkkkkkk:",k,mapIndex_cnt[k])
        if mapIndex_cnt[k]>mostv:
            mostk = k
            mostv = mapIndex_cnt[k]

    print("mostk:",mostk)

    for pair in mapXY_cnt:
        print("xyyyyyyyyy:",pair[0],pair[1],mapXY_cnt[(pair[0],pair[1])])

    for (i,j) in image_Lines:
        point = image_Lines.get((i,j),Point_Conti())
        if point.LineIndex==mostk:
            image_convolve[i:i + fangge, j:j + fangge] = 100
            image_convolve[i:i + fangge, j:j + fangge] = image[i:i + fangge, j:j + fangge]
        # if point.LineV>=lianxu_yuzhi:
            # image_convolve[i:i + fangge, j:j + fangge] = image[i:i+fangge,j:j+fangge]
            # image_convolve[i:i + fangge, j:j + fangge] = 100

    return image_convolve,maxLineV ,maxLineVDic.get(maxLineV)
    # return image_beginLianXu,maxLineV ,maxLineVDic.get(maxLineV)



####################################### BEGIN ###################################################
# 连续性竖向上宽度应该是1格，最大连续性的线段应该是那条线段
# 思路1 假设有线段a,竖向宽度是1，必然连续，和已知的图像的相似度最大，就是这条线段，怎么计算相似度是关键，缺点：可能不好计算多条线段的情况,这里的连续指前后或左上左下,右上右下有点，别的地方没有
# 先取所有大于某一阈值的点，然后从做到右，检查左右相邻且竖向上距离最小的点，这个左右相邻不能只有1个点，只有1个点，如果这个点和线相距很远，则是噪点，和左右不连续
# 此例假设只有一条线段，竖向连续阈值需要优化为可调节

#1先假定有连续性的线段，然后计算线段灰度与图像的重合性
#2所有有灰度的点所能构成的线段，然后通过竖向距离，计算连续性，然后得出最优线段,缺点：可能有太多的线段，需要预处理

#3先找到起始点，就是先竖向，找到线段起始点，然后先竖向，按一定连续性生成短线段，这个短线段比如占5格，竖向宽度为1，
# 这个短线段不是一格一格的拟合，而是长度为5的拟合，一格一格的拟合，容易被噪点带偏
# 使用线段去拟合的好处：1，不容易被噪点带偏，2，竖向宽度为1
# #以后可以优化短线段长度，短线段与短线段之间独立，不必以前一个的尾点作为后一个的始点，然后连接这些短线段成总线段（###甚至在短线段的基础上建立稍长线段，比如10格，然后这样迭代出最优结果）
# 或者可以让短线段先尽量连接成长线段，断裂较大的地方，就开始新线段
# 短线段的最大目的是去噪点，假如没有噪点，直接左右相邻的点比较就好了

# 优化:右，右上，右下，找最大灰度值  ## 首先向右，然后右上，然后右下，如果灰度值大于阈值，如果右上比右灰度更大，会被忽略

# 获取右侧所有点
def GetRight_ij(i,j,changdu = 4,shuxiangLian=1):# 根据当前坐标向右获得连续的断线断，竖向宽度为1，长度赞定位5,竖向最大距离1判为连续
    ij_list = []
    for ii in range(i+1,i+changdu):
        juli = ii-i
        for jj in range(j-juli,j+juli+1):
            print("ii:jj",ii,jj)
            ij_list.append((ii,jj))
    return  ij_list


# 不保证坐标是否合法,递归
def GetNextLines(lastLine,linesAll,cnt,changdu=4,fangge=1):
    # for lastLine in lastLines:

    # print("lastLine:",lastLine,lastLine[-1])
    lasti,lastj = lastLine[-1][0],lastLine[-1][1]
    line1 = lastLine + [(lasti-fangge,lastj+fangge)]
    line2 = lastLine + [(lasti , lastj+fangge)]
    line3 = lastLine + [(lasti + fangge, lastj+fangge)]
    # lines.append(line1,lin2,line3)
    if cnt>=4:
        linesAll.append(line1),linesAll.append(line2),linesAll.append(line3)
        return linesAll
    GetNextLines(line1,linesAll,cnt+1,changdu)
    GetNextLines(line2,linesAll,cnt+1,changdu)
    GetNextLines(line3,linesAll,cnt+1,changdu)

# 获取右侧所有短线段
def GetRightShortLines(i,j,changdu = 4):
    ij_list = []
    line0 = [(i,j)]
    linesAll = []
    GetNextLines(line0,linesAll,1,changdu,fangge=5)
    print("==============end====:",linesAll)
    return linesAll

# item = GetRightShortLines(10,10)

#获取图片i,j位置的右侧短线段
def GetRightShortLine_Image(image,i,j,changdu=4,fangge=5):
    linesAll = GetRightShortLines(i,j,changdu)
    result = False
    maxHuidu = 0
    imgw = image.shape[1] # y
    imgh = image.shape[0] # x
    # print("image shape::::::",image.shape,imgh,imgw)
    maxHuiduLine = []
    for line in linesAll:
        for point in line:
            # print("point::::",point)
            x =point[0];y =point[1]
            if x<0 or y<0 or x>= imgh or y>=imgw:
                continue
            curHuidu = np.sum(image[x:x+fangge,y:y+fangge])
            # print("---------:",curHuidu)
            if curHuidu >maxHuidu:
                maxHuidu = curHuidu
                maxHuiduLine = line
                result = True
    print("maxHuidu:::::::::::",maxHuidu)
    return result,maxHuidu,maxHuiduLine

########################################### END #############################################


#img是总图像的一小部分
def GetBestLineFromImg(img,fangge=1,yuzhi=1,lianxuYuzhi=1,curJ_=0,img_New=[]):
    print("img:",img)
    linesAll=[]
    stopJ = 0
    diGuiCnt= 0
    # todo 多个线段并行的情况
    def GetAllLines_New(line=[], curj=curJ_, lianxuAll=0, huiduAll=0, duanlie=0):
        haveBegin = False
        nonlocal stopJ
        # 线段的起始
        if len(line) == 0:
            duanlie = 0  # 初始化断裂的点数为0
            for i in range(img.shape[0]):
                curHuidu = np.sum(img[i:i + fangge, curj:curj + fangge])
                # todo 大于阈值，改为竖向局部极大值
                print("curhuidu i,curj,fangge,",curHuidu,i,curj,fangge)
                if curHuidu >= yuzhi:
                    line = [(i, curj)]
                    haveBegin = True
                    stopJ = curj
                    print("begin i:", i, curj)
                    GetAllLines_New(line, curj + fangge, lianxuAll, huiduAll + curHuidu)
            if haveBegin == False:
                print("errrrrrrrrrrrrrr!!!!!!!!!!!!!!!!")
                GetAllLines_New(line, curj + 1, lianxuAll, huiduAll + curHuidu)
                return
        else:  # 线段的中间和尾点
            # print("curj:",curj,line)
            if curj >= img.shape[1]:  # 超出了img的最大范围，停止迭代
                linesAll.append((line, lianxuAll, huiduAll))
                # print("ok!!!!!!!!!!! line:", line, lianxuAll, huiduAll)
                return

            lasti = line[-1][0]
            top = max(lasti - lianxuYuzhi, 0);
            down = min(lasti + lianxuYuzhi, img.shape[0] - 1)
            # print("lasti top down curj:",lasti,top,down,curj)
            lianXu = False
            for i in range(top, down + 1):
                curHuidu = np.sum(img[i:i + fangge, curj:curj + fangge])
                if curHuidu >= yuzhi:
                    lianxuCha = abs(i - lasti)
                    newline = line + [(i, curj)]  # 假设是引用
                    # print("newLines:",line,i,curj)
                    print("-",curj)
                    GetAllLines_New(newline, curj + fangge, lianxuAll + lianxuCha, huiduAll + curHuidu)
                    lianXu = True
            if lianXu == False:  # 遇到断点
                if duanlie == 0:  # 首次遇到断点，继续连接
                    print("==",curj)
                    GetAllLines_New(line, curj + fangge, lianxuAll, huiduAll, duanlie + 1)  # 遇到断点，继续连接下一个列的点
                else:  # 非首次遇到断点，停止连接,保存线段
                    print("###")
                    linesAll.append((line, lianxuAll, huiduAll))  # 遇到断点，停止连接，保存线段
                # stopJ = max(stopJ,curj)

        # nonlocal diGuiCnt
        # diGuiCnt = diGuiCnt+1
        # print(diGuiCnt)

    GetAllLines_New([])
    print("--------stopJ:",stopJ)
    # print("linesalllll:",linesAll)
    if len(linesAll)==0:
        print("lineaAsll ==0!!!!!!!!!!!!!!!")
        return  np.array([]),np.array([])
    minLianxu = linesAll[0][1]
    maxHuidu = linesAll[0][2]
    bestIndex=0
    # 尾点是连接点，是前一线段的尾点，同时是下一线段的始点,尾点是否应该取权重较大，而不是按连续差值较小
    linesAll = np.array(linesAll)
    print("linesAll shape:",linesAll.shape)
    for index in range(linesAll.shape[0]):
        item = linesAll[index]
        if item[2]>maxHuidu:#取灰度值较大的线
            minLianxu = item[1]
            maxHuidu = item[2]
            bestIndex = index
            # print("bestIndex:",bestIndex)
        elif item[2]==maxHuidu and item[1]<minLianxu:# 灰度值相等的线 取连续差值小的线
            minLianxu = item[1]
            maxHuidu = item[2]
            bestIndex = index
            # print("bbbbbestIndex:", bestIndex)

    if len(img_New)==0:
        img_New = np.zeros(img.shape)

    print("bestIndex:",bestIndex)
    lineBest = linesAll[bestIndex][0]
    for point in lineBest:
        if len(lineBest)>3: # todo 配置或走参数
            ii = point[0];jj=point[1]
            img_New[ii:ii+fangge,jj:jj+fangge]=200

    print("stopJ:------------",stopJ,img.shape[1])
    # if stopJ < img.shape[1] - 1: # 有间隔的线段，会分开成两条线段
    #     GetBestLineFromImg(img,fangge,yuzhi,lianxuYuzhi,stopJ,img_New)

    return linesAll[bestIndex][0],img_New

def Test_GetBestLineFromImg():
    img = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0, 2, 1],
        [0, 0, 0, 0, 1, 0, 0, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 2, 0, 1, 0, 0, 0, 0, 1],
    ]
    img = np.array(img)
    # img = np.tile(img,(2,10))
    plt.imshow(img,cmap="Greys")
    plt.show()
    line,newImg = GetBestLineFromImg(img,fangge=1,yuzhi=1,lianxuYuzhi=2,curJ_=0)
    # if newImg != None: 如果newImg是数组，判断会报错
    if newImg.any():
        plt.imshow(newImg,cmap="Greys")
        plt.show()
    else:
        print("newImg is None!!!!!!")

    print("\n==================================================\n")

# Test_GetBestLineFromImg()


def TestImgLine(img):
    print("======================00")
    plt.imshow(img, cmap="Greys")
    plt.show()
    # plt.hist(img, 256)
    plt.show()
    line, newImg = GetBestLineFromImg(img, fangge=2, yuzhi=300, lianxuYuzhi=2, curJ_=0)
    print("======================11")
    if newImg.any():
        print("======================22")
        plt.imshow(newImg, cmap="Greys")
        plt.show()
    else:
        print("======================None!!!!")

# img = data.camera()
img = test_spec.GetWavData()
img = np.clip(img,0,255) # 归一化
img = np.array(img,np.uint8)# 因为opencv读取文件默认CV_8U类型，在做完卷积后会转化为CV_32FC1类型的矩阵来提高精度或者避免舍入误差。需要clip之后转换为np.uint8。


TestImgLine(img[50:60,50:80])

###########################          END             #####################################

map_Lines = {}
def imgLines_fjc_Line(image,huidu_yuzhi=50,fangge =5,lianxu_yuzhi=5):
    img_h = int(image.shape[0])
    img_w = int(image.shape[1])
    image_convolve = np.zeros(image.shape)
    image_new = np.zeros(image.shape)
    point_original = Point_Conti()
    list = [point_original]*image.shape[0]*image.shape[1]
    Points_test = np.array(list).reshape(image.shape)
    test = []
    testA = np.array(test)
    print("Points_All shape:",Points_test.shape)
    for i in range(len(image)):
        for j in range(len(image[0])):
            # test[i][j] = 10 #list index out of range
            test.append(10)
            point_original = Point_Conti()
            point_original.i=i
            point_original.j=j
            point_original.huidu = image[i,j]
            Points_test[i, j] = point_original
            Points_test[i][j] = point_original

    maxLineV = 0
    global  maxLineVDic

    map_All = {}
    huidu_all =[]
    for i in range(fangge, img_h,fangge):#向下
        for j in range(fangge, img_w, fangge):#向右
            point_original = Point_Conti()
            point_original.i = i
            point_original.j = j
            point_original.huidu = np.sum(image[i:i+fangge,j:j+fangge])
            # print("==huidu:",point_original.huidu)
            map_All.setdefault((i, j), point_original)
            if point_original.huidu!=0:
                huidu_all.append(point_original.huidu)

    plt.hist(huidu_all, 256)
    plt.show()


# todo:开始点,串成线的逻辑
    point = Point_Conti()
    for i in range(fangge, img_h,fangge):#向下
        for j in range(fangge, img_w, fangge):#向右
            cur_huidu = np.sum(image[i:i+fangge, j:j+fangge])
            point_cur = image_Lines.setdefault((i, j), point)
            # 第一条格子开始线段起始灰度要大于某一阈值
            if cur_huidu<1:
                continue

            # result,maxHuiduLine,maxHuidu = GetRightShortLine_Image(image,i,j,changdu=4,fangge=5)
            # if result and maxHuidu>100:
            #     for point in line:
            #         image_convolve[]



            maxLianxu = 0
            beginX = 0
            beginY = 0
            point = Point_Conti()
            point.i = i
            point.j = j
            # fangge的大小是调节的关键所在，当图像只剩一条线，这个fangge大小就是最佳的
            # 向左面寻找连续性最大的区域




    return image_convolve,maxLineV ,maxLineVDic.get(maxLineV)
    # return image_beginLianXu,maxLineV ,maxLineVDic.get(maxLineV)


def TestConv(img):
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
    plt.imshow(img, cmap="Greys")
    plt.show()
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

        for huidu_yuzhi in range(50,51):
            for fangge in range(5,6):
                for lianxu_yuzhi in range(5,6):
                    # line, newImg = GetBestLineFromImg(img, fangge=fangge, yuzhi=huidu_yuzhi, lianxuYuzhi=lianxu_yuzhi, curJ_=0)
                    # plt.imshow(img, cmap="Greys")
                    # plt.show()
                    # print("======================11")
                    # if newImg.any():
                    #     print("======================22")
                    #     plt.imshow(newImg, cmap="Greys")
                    #     plt.show()
                    # else:
                    #     print("*** newImg is None!!!!!!")

                    img_sobel,maxLineV,dic= imgLines_fjc_Line(img,huidu_yuzhi = huidu_yuzhi,fangge=fangge,lianxu_yuzhi=lianxu_yuzhi)
                    # img_sobel, maxLineV, dic = imgLines_fjc(img)
                    maxLineV_final = max(maxLineV_final,maxLineV)
                    print("img.shape:",img.shape,type(img),"img_sobel shape:",img_sobel.shape,type(img_sobel),maxLineV)
                    img_sobel = np.clip(img_sobel, 0, 255)  # 归一化
                    img_sobel = np.array(img_sobel, np.uint8)
                    cv2.imshow('window2', img_sobel)
                    xlen = img_sobel.shape[1]
                    # plt.xticks(np.arange(0,320,5))
                    # plt.yticks(np.arange(0,100,10))

        # img_sobel = np.array(img_sobel)
        # print("img_sobel shape:", img_sobel.shape, type(img_sobel), maxLineV_final)
        plt.imshow(img_sobel,cmap="Greys")
        # plt.grid(ls='--',c='g')
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


# TestConv(img)



