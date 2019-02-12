import wave
import matplotlib.pyplot as plt
import numpy as np
import os

# path = "./"
# name = 'yusheng.wav'
# filename = os.path.join(path, name)
# f = wave.open(filename, 'rb')
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# strData = f.readframes(nframes)  # 读取音频，字符串格式
# waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
# waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
# # plot the wave
# time = np.arange(0, nframes) * (1.0 / framerate)
# plt.plot(time, waveData)
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Single channel wavedata")
# plt.grid('on')  # 标尺，on：有，off:无。

filepath = "./" #添加路径
#filename= os.listdir(filepath) #得到文件夹下的所有文件名称
filename= "yusheng.wav"
f = wave.open(filepath+filename,'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)#读取音频，字符串格式
waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
waveData = np.reshape(waveData,[nframes,nchannels])
f.close()
# plot the wave
time = np.arange(0,nframes)*(1.0 / framerate)
plt.figure()
# plt.subplot(5,1,1)
# plt.plot(time,waveData[:,0])

time= np.reshape(time,[nframes,1]).T
plt.plot(time[0,:nframes],waveData[0,:nframes],c="b")

plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Ch-1 wavedata")
plt.grid('on')#标尺，on：有，off:无。

# plt.subplot(5,1,3)
# plt.plot(time,waveData[:,1])
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Ch-2 wavedata")
# plt.grid('on')#标尺，on：有，off:无。
#
# plt.subplot(5,1,5)
# plt.plot(time,waveData[:,2])
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Ch-3 wavedata")
# plt.grid('on')#标尺，on：有，off:无。
# plt.show()