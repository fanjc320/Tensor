import numpy as np
import wave
import pyaudio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import fft,ifft
# import common
from common import wavReads,wavWrite
import array
plt.tight_layout()

def Func3(inFile,outFile,figNumb=1):
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
	plt.figure(figNumb)
	
	wav_fft1 = fft(wave_data1)
	xf1 = np.arange(0,len(wave_data1))
	
	plt.subplot(311)
	plt.title("wave_data1")
	plt.plot(xf1,wave_data1,'r')
	
	plt.subplot(312)
	# clip1 = len(time1)
	clip1 = 800 #去掉500后的高频，声音内容不变，稍显沉闷,300几乎还能听出来hello
	# wav_fft1_clip = wav_fft1[:clip1]
	# arr0 = np.zeros(len(xf1)-5*clip1,dtype=np.short) #减少对称频谱之间的间距 慢慢变得尖锐有噪声
	#等价于 wav_fft1[1000:25000] = 0,# 去掉高频部分可以降噪!!!!!!!!
	# wav_fft1_conc = np.concatenate([wav_fft1[:clip1],arr0,wav_fft1[-clip1:]])
	wav_fft1_conc = wav_fft1[:]
	#带通滤波
	wav_fft1_conc[0:200]=0
	wav_fft1_conc[800:-800]=0
	wav_fft1_conc[-200:]=0
	print("len(wav_fft1_conc)",len(wav_fft1_conc))
	plt.title("wav_fft1_conc clip1:"+str(clip1))
	plt.plot(xf1[:len(wav_fft1_conc)][:clip1],wav_fft1_conc[:clip1],'b')

	wav_ifft1 = ifft(wav_fft1_conc).real
	wav_ifft_as = np.around(wav_ifft1).astype(np.short)

	plt.subplot(313)
	plt.title("wav_ifft_as "+str(clip1))
	plt.plot(time1[:len(wav_ifft_as)], wav_ifft_as,'b-')
	
	# Numb = 20;
	# print("len(wave_data1)",len(wave_data1),"len(wav_ifft_as)",len(wav_ifft_as))
	# print("wav_ifft1[:Numb]  ",wav_ifft1[:Numb])
	# print("wav_ifft_as[:Numb]",wav_ifft_as[:Numb])

	wavWrite(wav_ifft_as,outFile,channels,sampwidth,framerate)
	
	plt.tight_layout()
	
	
if __name__ == "__main__":
	
	Func3("../res/hello11s.wav","../res/hello11s_daitong.wav",1)
	plt.show()