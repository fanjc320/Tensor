import numpy as np
import wave
import pyaudio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import fft,ifft
# from Common.common import *
import common
import Common.common
import py_audio_block
from common import wavReads,wavWrite
import array
plt.tight_layout()


def Func2():
	plt.tight_layout()
	
	wf = wave.open("./hello11s.wav", "rb")
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
	
	plt.figure(num=1,figsize=(8,5),)
	
	wav_fft = fft(wave_data);
	xf = np.arange(0,len(wave_data))
	
	plt.subplot(311)
	plt.plot(xf,wave_data,'r')
	
	plt.subplot(312)
	plt.plot(xf,wav_fft,'g')
	
	plt.subplot(313)
	clip = len(time)
	plt.plot(xf[:clip],wav_fft[:clip],'b')
	
	plt.figure(num=2,figsize=(8,5),)
	
	wav_ifft = ifft(wav_fft[:clip]).real
	wav_ifft_1 = np.around(wav_ifft).astype(np.short)

	plt.subplot(211)
	plt.plot(time[:clip], wave_data[:clip])
	plt.subplot(212)
	plt.plot(time[:clip], wav_ifft_1[:clip],'b-')
	
	Numb = 20;
	print("len===",len(wave_data),len(wav_ifft_1))
	print("wave_data(fft)->wav_fft(ifft)->wav_ifft",wav_ifft[:Numb])
	print("wave_data(fft)->wav_fft(ifft)->wav_ifft(astypeint)->wav_ifft_1",wav_ifft_1[:Numb])

	wf1 = wave.open("./hello11s_guolv.wav",'wb')
	wf1.setnchannels(channels)
	wf1.setsampwidth(sampwidth)
	wf1.setframerate(framerate)
	
	str = wave_data.tostring()
	stri = wav_ifft_1.tostring()
	print("wave_data(tostring)->str",str[:Numb]);
	print("    wav_ifft_1(tostring)->stri",stri[:Numb]);
	
	wf1.writeframes(wav_ifft_1.tostring())
	wf1.close()
	
	plt.show()
	
	# ll = list(range(1,10,1))
	# print(ll)
	# print("---- ",np.where(ll>5))
	
	
	print(np.where([[True,False],[True,True]],
			[[1,2],[3,4]],
			[[9,8],[7,6]])
	)
	
	x = np.arange(16)
	print("# ",type(x))
	print(x[np.where(x>-1)])
	
	x = np.arange(8).reshape(-1,4)
	print("--",x)
	print(np.where(x>5)) #返回两个数组，第一个，满足条件的所在行，第二个，满足条件的所在列
	
def Func3(inFile,outFile):
	plt.tight_layout()
	
	wf1 = wave.open(inFile, "rb")
	nframes = wf1.getnframes()
	framerate = wf1.getframerate()
	channels = wf1.getnchannels()
	sampwidth = wf1.getsampwidth()
	str_data1 = wf1.readframes(nframes)
	print("frames: ",nframes,"framerate:",framerate,"sampwidth:",sampwidth,"channels:",channels);
	wf1.close()

	wave_data1 = np.fromstring(str_data1, dtype=np.short)
	time1 = np.arange(0,nframes)*(1.0/framerate)
	time1 = time1[0:int(len(time1))] #单声道
	# #########################################################
	plt.figure(1)
	
	wav_fft1 = fft(wave_data1)
	xf1 = np.arange(0,len(wave_data1))
	
	plt.subplot(311)
	plt.title("wave_data1")
	plt.plot(xf1,wave_data1,'r')
	
	plt.subplot(312)
	plt.title("wav_fft1")
	plt.plot(xf1,wav_fft1,'g')
	
	plt.subplot(313)
	clip1 = len(time1)
	plt.title("wav_fft1 clip1:"+str(clip1))
	plt.plot(xf1[:clip1],wav_fft1[:clip1],'b')
	
	# #########################################################
	plt.figure(2)
	
	wav_fft1[1000:25000] = 0,# 去掉高频部分可以降噪!!!!!!!!
	wav_ifft1 = ifft(wav_fft1[:clip1]).real
	wav_ifft_11 = np.around(wav_ifft1).astype(np.short)

	plt.subplot(211)
	plt.title("wave_data1")
	plt.plot(time1, wave_data1)
	plt.subplot(212)
	plt.title("wav_ifft_11 clip1:"+str(clip1))
	plt.plot(time1[:clip1], wav_ifft_11[:clip1],'b-')
	
	# Numb = 20;
	# print("len(wave_data1)",len(wave_data1),"len(wav_ifft_11)",len(wav_ifft_11))
	# print("wav_ifft1[:Numb]  ",wav_ifft1[:Numb])
	# print("wav_ifft_11[:Numb]",wav_ifft_11[:Numb])

	wavWrite(wav_ifft_11,outFile,channels,sampwidth,framerate)
	
	# wave_data_str1 = wave_data1.tostring()
	# wav_ifft_1_str1 = wav_ifft_11.tostring()
	# print("wave_data_str1  ",wave_data_str1[:Numb])
	# print("wav_ifft_1_str1",wav_ifft_1_str1[:Numb])
	
	
	plt.show()
	plt.tight_layout()
	
def TestStr():
	a = "s"
	b = 5;
	print(a+str(b))
	print("dd"+str(b))
	
if __name__ == "__main__":
	# TestStr()
	Func3("../../res/hello11s.wav","../../res/hello11s_guolv.wav")
	