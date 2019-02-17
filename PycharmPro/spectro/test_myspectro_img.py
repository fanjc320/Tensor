
import numpy as np
import matplotlib.pyplot as plt
import wave
import scipy.signal as signal
import os
from scipy.fftpack import fft,ifft
import seaborn
# import math

# 加窗
def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵

def wavread(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    f.close()
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    return waveData


filepath = "./"  # 添加路径
filename = "yusheng.wav"  # 得到文件夹下的所有文件名称
f = wave.open(filepath + filename, 'rb')
data = wavread(filename)
nw = 512
inc = 128
winfunc = signal.hamming(nw)
Frame = enframe(data[0], nw, inc,winfunc)

# for i in range(len(Frame)):
#     for j in range(len(Frame[i])):
#         print(i,np.shape(Frame[i]))

result = []
maxFreq = 0
for i in range(len(Frame)):
    # print(i,Frame[i])
    yy = abs(fft(Frame[i]))
    maxFreq = max(maxFreq,max(yy))
    result.append(yy)




print(np.shape(result))
print(maxFreq)
time = np.arange(1,len(result),1)

print(np.shape(time))


# from skimage import io,data
# img=data.astronaut()
# print(np.shape(img))
# plt.imshow(img[:512][:512][:3])
fig = plt.figure()
#X = [[1,2],[3,4],[5,6],[7,8]]
X = [[-1,-2],[-3,-4],[-15,6],[-7,8]]
#fig.add_subplot(221)
# plt.imshow(X,cmap = plt.cm.gray)
#plt.imshow(X,cmap = plt.cm.summer)
#plt.colorbar()

fig.add_subplot(231)
map = plt.imshow(X, interpolation='nearest', cmap=plt.cm.summer, aspect='auto',vmin=1, vmax=15)
cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
fig.add_subplot(232)
map = plt.imshow(X, interpolation='nearest', cmap=plt.cm.summer, aspect='auto',vmin=3, vmax=5)
cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
fig.add_subplot(233)
map = plt.imshow(X, interpolation='nearest', cmap=plt.cm.summer, aspect='auto', vmin=1, vmax=2)
cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
fig.add_subplot(234)
map = plt.imshow(X, interpolation='nearest', cmap=plt.cm.summer, aspect='auto', vmin=2, vmax=3)
cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
fig.add_subplot(235)
map = plt.imshow(X, interpolation='nearest', cmap=plt.cm.summer, aspect='auto', vmin=3, vmax=4)
cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
plt.show()

# ---------------------------------
# img0 =[]
a0 = np.arange(1,55)
# img0.append(rag)
a1 = a0[::-1]

img0 = []
img1 = []
img0.append(a0)
img1.append(a1)

# ---------------------------------
# img0.append(img1);
img = [img0,img0,img0]
print("img0 shape:",np.shape(img0))
print("img shape:",np.shape(img))
print("img shape:",np.shape(img0))
# plt.imshow(img0*50,origin='lower')





# ---------------------------------

from matplotlib import cm
from matplotlib import axes
def draw_heatmap(data, xlabels, ylabels):
    # cmap=cm.Blues
    cmap = cm.get_cmap('rainbow', 1000)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    #map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    #plt.show()


a = np.random.rand(10, 10)
print(a)
xlabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
ylabels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# draw_heatmap(a, xlabels, ylabels)





