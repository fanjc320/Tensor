import numpy as np
import wave
import pyaudio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import fft,ifft
# import common
from common import wavreads
import array
plt.tight_layout()

# fft 频谱去掉高频，然后ifft
def Func1():
	# fjc_record(OutFile="test0.wav")
	plt.tight_layout()
	
	fig = plt.figure()
	
	wavdata,wavtime = wavreads("./hello11s.wav")
	plt.title("hello11.wav's Frames")
	plt.subplot(411)
	plt.plot(wavtime, wavdata,color = 'green')
	# plt.show()
	
	fjc_record_with(wavdata)
	
	yf=fft(wavdata)
	xf = np.arange(len(wavdata))
	plt.subplot(412)
	plt.plot(xf[:1000],yf[:1000],'r')
	# plt.show()
	
	plt.subplot(413)
	yi = ifft(yf)
	plt.plot(wavtime,yi, 'g')
	# plt.show()
	
	plt.figure(num=3,figsize=(8,5),)
	yf_ = yf[100:500]
	plt.subplot(411)
	plt.plot(np.arange(len(yf_)),yf_,'r')
	yi_ = ifft(yf_)
	plt.subplot(412)
	plt.plot(np.arange(len(yi_)),yi_,'b')
	
	
	plt.show()
	plt.tight_layout()
	
def Func2():
	# fjc_record(OutFile="test0.wav")
	plt.tight_layout()
	
	wf = wave.open("./hello11s.wav", "rb")
#创建PyAudio对象
	p = pyaudio.PyAudio()
	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	channels=wf.getnchannels(),
	rate=wf.getframerate(),
	output=True)
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
	plt.plot(time, wave_data)
	# plt.show()
	
	plt.figure(num=3,figsize=(8,5),)
	
	yf = fft(wave_data);
	xf = np.arange(0,len(wave_data))
	
	plt.subplot(311)
	plt.plot(xf,wave_data,'r')
	
	plt.subplot(312)
	plt.plot(xf,yf,'g')
	
	plt.subplot(313)
	clip = len(time)
	plt.plot(xf[:clip],yf[:clip],'b')
	
	plt.figure(num=4,figsize=(8,5),)
	
	yyi = ifft(yf[:clip]).real
	testi = np.ndarray(shape=(2,1),buffer=np.array([181.,488.]),dtype=float)
	
	yyii = yyi.astype(np.int32)
	yy_te = np.around(yyi)
	print("---",yyi[0]," ",int(round(yyi[0]))," yy_te ",yy_te[:20])
	print()
	testii = testi.astype(np.int32)
	yyiif = yyii.astype(float)
	yyii_copy = yyii.copy()
	# plt.plot(time[:clip],yyi)
	print("type(yyi)",type(yyi),"dtype:",yyi.dtype,yyii.dtype,yyiif.dtype)
	print("type(testi)",type(testi),"dtype:",testi.dtype,testii.dtype,"testii",testii)
	print(" yyi",yyi[:20]," yyii",yyii[:20]," yyiif",yyiif[:20])
	print("yyii_copy",yyii_copy);
	
	# data = wave_data.tostring()
	# data = str_data.decode('iso-8859-1')
	Numb = 20;
	str = wave_data.tostring()
	print("str_data",str_data[:Numb])
	print("str_data(fromstring)->wave_data",wave_data[:Numb])
	print("wave_data(tostring)->str",str[:Numb]);
	print("wave_data(fft)->yf(ifft)->yyi",yyi[:Numb])
	print("wave_data(fft)->yf(ifft)->yyi(astypeint)->yyii",yyii[:Numb])
	print("yyistr",(yyi.tostring())[:Numb])
	wf1 = wave.open("./hello11s_w.wav",'wb')
	wf1.setnchannels(channels)
	wf1.setsampwidth(sampwidth)
	wf1.setframerate(framerate)
	# wf1.writeframes(str)
	wf1.writeframes(yyi.tostring())
	wf1.close()
	
	'''
	# 采样点数，修改采样点数和起始位置进行不同位置和长度的音频波形分析
	N=44100
	start=0 #开始采样位置
	df = framerate/(N-1) # 分辨率
	freq = [df*n for n in range(0,N)] #N个元素
	print("--framerate:",framerate," nframes:",nframes," df:",df);
	wave_data2=wave_data[start:start+N]
	c=np.fft.fft(wave_data2)*2/N
	#常规显示采样频率一半的频谱
	d=int(len(c)/2)
	print("max frequency:",d)
	#仅显示频率在4000以下的频谱
	while freq[d]>4000:
		d-=10
	plt.plot(freq[:d-1],abs(c[:d-1]),'r')
	
	'''
	plt.show()
	
	
def FromToStr():
	ali = array.array('i',[1,2,3])
	str = ali.tostring();
	print("ali",ali," tostring()",str);
	arr = np.fromstring(str,dtype=np.int)
	print("arr",arr)
	
if __name__ == "__main__":
    # Func1()
	FromToStr()
	Func2()
	
# Wave_read.readframes(n)
# Reads and returns at most n frames of audio, as a string of bytes.



# ifft Returns:	out : complex ndarray
# The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.
