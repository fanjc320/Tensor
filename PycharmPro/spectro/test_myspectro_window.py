import numpy as np
import matplotlib.pyplot as plt
import wave
import scipy.signal as signal
import os
from scipy.fftpack import fft,ifft
import seaborn
# import math

# 不加窗
# def enframe(signal, nw, inc):
#     '''将音频信号转化为帧。
#     参数含义：
#     signal:原始音频型号
#     nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
#     inc:相邻帧的间隔（同上定义）
#     '''
#     signal_length = len(signal)  # 信号总长度
#     if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
#         nf = 1
#     else:  # 否则，计算帧的总长度
#         nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
#     pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
#     zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
#     pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
#     indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
#                                                            (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
#     indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
#     frames = pad_signal[indices]  # 得到帧信号
#     #    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
#     #    return frames*win   #返回帧信号矩阵
#     return frames

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
for i in range(len(result)-13000):
    for j in range(len(result[i])):
        # print(np.shape(result))
        print(i,result[i][j])
        # plt.plot(i,result[i][j])
        # plt.plot(i,i)

# plt.plot(time, result)
plt.imshow(result)
plt.show()

