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
	clip1 = 1000
	plt.title("wav_fft1 clip1:"+str(clip1))
	plt.plot(xf1[:clip1],wav_fft1[:clip1],'b')
	
	# #########################################################
	# plt.figure(2)
	
	wav_fft1[1000:25000] = 0,# 去掉高频部分可以降噪!!!!!!!!
	wav_ifft1 = ifft(wav_fft1[:clip1]).real
	wav_ifft_as = np.around(wav_ifft1).astype(np.short)

	plt.subplot(313)
	plt.title("wav_ifft_as clip1:"+str(clip1))
	plt.plot(time1[:clip1], wav_ifft_as[:clip1],'b-')
	
	# Numb = 20;
	# print("len(wave_data1)",len(wave_data1),"len(wav_ifft_as)",len(wav_ifft_as))
	# print("wav_ifft1[:Numb]  ",wav_ifft1[:Numb])
	# print("wav_ifft_as[:Numb]",wav_ifft_as[:Numb])

	# wavWrite(wav_ifft_as,outFile,channels,sampwidth,framerate)
	
	# wave_data_str1 = wave_data1.tostring()
	# wav_ifft_1_str1 = wav_ifft_as.tostring()
	# print("wave_data_str1  ",wave_data_str1[:Numb])
	# print("wav_ifft_1_str1",wav_ifft_1_str1[:Numb])
	
	
	plt.tight_layout()
	
	
if __name__ == "__main__":
	
	Func3("./hello11s.wav","./hello11s_compare.wav",1)
	Func3("./hello21s.wav","./hello21s_compare.wav",2)
	
	plt.show()