import os
import wave
import numpy as np
import matplotlib.pyplot as plt

filepath = "./"
filename=os.listdir(filepath)
for file in filename:
	print(filepath+file)
	
file = "./night.wav"
f=wave.open(file,'rb')
params = f.getparams()
print("params.size:",params)
nchannels,sampwidth,framerate,nframes = params[:4]
strData = f.readframes(nframes)#读取音频，字符串格式
#sampwidth,采样宽度，这里是2，代表2个bytes,所以用formatstring用 int16strData = f.readframes(nframes)
waveData = np.fromstring(strData,dtype=np.int16)
waveData = waveData*1.0/(max(abs(waveData)))
#如果是多声道，需打开下面
waveData = np.reshape(waveData,[nframes,nchannels])
print("shape:",len(strData),waveData.shape,nframes)
f.close()

time = np.arange(0,nframes)*(1.0/framerate)
plt.figure()
plt.subplot(5,1,1)
plt.plot(time,waveData[:,0])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Ch-1 wavedata")
plt.grid('on')
plt.subplot(5,1,3)
plt.plot(time,waveData[:,1])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Ch-2 wavedata")
plt.grid('on')
# plt.subplot(5,1,5)
# plt.plot(time,waveData[:,2])
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Ch-3 wavedata")
# plt.grid('on')
plt.show()



