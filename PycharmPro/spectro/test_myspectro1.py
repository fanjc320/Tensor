# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:15:34 2017

@author: Nobleding
"""

import wave
import matplotlib.pyplot as plt
import numpy as np
import os

filepath = "./"  # 添加路径
filename = "yusheng.wav"  # 得到文件夹下的所有文件名称
f = wave.open(filepath + filename, 'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)  # 读取音频，字符串格式
waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
waveData = np.reshape(waveData, [nframes, nchannels])

print(waveData.shape)
print(waveData.size)
print(np.size(waveData,0))
print(np.size(waveData,1))

weidu = np.size(waveData,1)

f.close()
# plot the wave
time = np.arange(0, nframes) * (1.0 / framerate)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, waveData[:, 0])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Ch-1 wavedata")
plt.grid('on')  # 标尺，on：有，off:无。


if weidu >1:
    plt.subplot(3, 1, 2)
    plt.plot(time, waveData[:, 1])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Ch-2 wavedata")
    plt.grid('on')  # 标尺，on：有，off:无。

if weidu >2:
    plt.subplot(3, 1, 3)
    plt.plot(time, waveData[:, 2])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Ch-3 wavedata")
    plt.grid('on')  # 标尺，on：有，off:无。


plt.show()
