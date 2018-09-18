import numpy as np
import wave
import pyaudio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import fft,ifft
# import common
from Common.common import wavReads,wavRead
import array
plt.tight_layout()

# fft 频谱去掉高频，然后ifft
def Func1():
	fig = plt.figure()

	plt.figure(num=1, figsize=(8, 5), )
	wavdata,wavtime = wavReads("../../res/yusheng1.wav")
	print("wav len",len(wavdata),len(wavtime))
####-----------wavdata-fft-ifft--------------------
	plt.title("yusheng1.wav's Frames")
	plt.subplot(311)
	plt.title("wavdata")
	plt.plot(wavtime, wavdata,color = 'green')

	yf=fft(wavdata)
	xf = np.arange(len(wavdata))
	plt.subplot(312)
	plt.title("fft 412")
	plt.plot(xf, yf, 'r')

	plt.subplot(313)
	yi = ifft(yf)
	plt.title("ifft 413")
	plt.plot(wavtime,yi, 'g')
####--------------fft[:]-ifft[:]-------------------
	plt.figure(num=2,figsize=(8,5),)
	yf_ = yf[100:500]
	plt.title("fft[:] 411")
	plt.subplot(411)
	plt.plot(np.arange(len(yf_)),yf_,'r')
	yi_ = ifft(yf_)
	plt.subplot(412)
	plt.title("fft[:] 412")
	plt.plot(np.arange(len(yi_)),yi_,'b')

	plt.tight_layout()
	plt.show()
	plt.tight_layout()
	
def Func2(InPath = "../../res/yusheng1.wav",OutPath= "../../res/yusheng1_a.wav"):
	# fjc_record(OutFile="test0.wav")
	plt.tight_layout()
	
	wf = wave.open(InPath, "rb")
	nframes = wf.getnframes()
	framerate = wf.getframerate()
	channels = wf.getnchannels()
	sampwidth = wf.getsampwidth()
#读取完整的帧数据到str_data中，这是一个string类型的数据 错，其实是bytes型数据
	str_data = wf.readframes(nframes)
	print("frames: ",nframes,"framerate:",framerate,"sampwidth:",sampwidth,"channels:",channels);
	print("type(strdata)",type(str_data))
	wf.close()
	# A new 1-D array initialized from raw binary or text data in a string.
	wave_data = np.fromstring(str_data, dtype=np.short)
	time = np.arange(0,nframes)*(1.0/framerate)
	time = time[0:int(len(time))] #单声道
	print("len(time):",len(time),"len(wav)",len(wave_data));
	# plt.plot(time, wave_data)
	# plt.show()

#----------------FFT---------------
	plt.figure(num=3,figsize=(8,5),)
	
	yf = fft(wave_data);
	xf = np.arange(0,len(wave_data))
	
	plt.subplot(311)
	# plt.plot(xf,wave_data,'r')
	plt.ylabel("wava_data")
	
	plt.subplot(312)
	plt.plot(xf,yf,'g')
	plt.ylabel("yf")

	# 频谱编辑
	# flag1 = 2000;
	# yf[200:flag1] = 0;
	# yf[flag1:200000] = 0
	# 移频
	shift = 200 # 200 机器声
	tmp = np.zeros(shift)
	print("---", len(tmp), len(yf))
	yf = np.append(tmp,yf)
	print("++++",len(tmp),len(yf))
	plt.subplot(313)
	clip = len(time)
	part = 2000;
	plt.plot(xf[:clip][0:part],yf[:clip][0:part],'b')
	plt.ylabel("yf[:clip]"+InPath)

#--------------IFFT---------------
	plt.figure(num=4,figsize=(8,5),)
	
	yfi_ = ifft(yf[:clip]).real
	yyi_as = yfi_.astype(np.short) #180.9999999,会变成180
	yyi_rint_as = np.rint(yfi_).astype(np.short)#rint 四舍五入

	# plt.subplot(211)
	# plt.plot(time[:clip], wave_data[:clip])
	# plt.subplot(212)
	# plt.plot(time[:clip], yyi_as[:clip],'b-')

#------------------WRITE-----------------
	wf1 = wave.open(OutPath,'wb')
	wf1.setnchannels(channels)
	wf1.setsampwidth(sampwidth)
	wf1.setframerate(framerate)

	# res = wave_data-yyi_as
	# res = res.tolist()
	# print("wave_data type:",type(wave_data[0]))
	# print("wave_data type:", type(res))
	# print("--------res[]!=0---------",np.where(res!=0))#int 默认是int64

	# str = wave_data.tostring()
	# print("wave_data(tostring)->str",str[:Numb]);
	# stri = yyi_as.tostring()
	# print("    yyii(tostring)->stri",stri[:Numb]);

	yyi_rint_as_t = yyi_rint_as-0;# -5000听不出来，-50000才听得出

	Numb = 20;  # 显示前Numb个元素
	print("wave_data[:Numb]", wave_data[:Numb])
	print("yfi_[:Numb]", yfi_[:Numb])
	print("yyi_as", yyi_as[:Numb])
	print("yyi_rint_as", yyi_rint_as[:Numb])
	print("yyi_rint_as_t", yyi_rint_as_t[:Numb])

	wf1.writeframes(yyi_rint_as_t.tostring())
	wf1.close()
	plt.show()

if __name__ == "__main__":
    # Func1()
	# FromToStr()
	Func2()
	# Func2(InPath = "../../res/huimei1.wav",OutPath= "../../res/huimei1_a.wav" )


# ifft Returns:	out : complex ndarray
# The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.
