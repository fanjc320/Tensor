		
import wave
#import matplotlib.pyplot as plt
import numpy as np
import os
import struct
 
#wav文件读取
filepath = "./" #添加路径
filename= os.listdir(filepath) #得到文件夹下的所有文件名称 
print("filename:",filename)
filename = "night.wav"
#f = wave.open(filepath+filename[1],'rb')
f = wave.open(filename,'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)#读取音频，字符串格式
waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
f.close()
#wav文件写入
outData = waveData#待写入wav的数据，这里仍然取waveData数据
outfile = filepath+'out1.wav'
outwave = wave.open(outfile, 'wb')#定义存储路径以及文件名
nchannels = 1
sampwidth = 2
fs = 8000
data_size = len(outData)
framerate = int(fs)
nframes = data_size
comptype = "NONE"
compname = "not compressed"
outwave.setparams((nchannels, sampwidth, framerate, nframes,
    comptype, compname))
	
print("outData:",len(outData))
count = 0;
for v in outData:
        count = count+1;
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))#outData:16位，-32767~32767，注意不要溢出
print("count:",count)
outwave.close()

#pack https://blog.csdn.net/w83761456/article/details/21171085