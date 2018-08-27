import numpy as np
import wave
import pyaudio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import fft,ifft
# import common
from common import wavreads
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
	#读取完整的帧数据到str_data中，这是一个string类型的数据
	str_data = wf.readframes(nframes)
	print("frames: ",nframes,"framerate:",framerate,"sampwidth:",sampwidth,"channels:",channels);
	print("type(strdata)",type(str_data))
	wf.close()
	# A new 1-D array initialized from raw binary or text data in a string.
	wave_data = np.fromstring(str_data, dtype=np.short)
	#将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
	# 单声道的，下面注掉
	# wave_data.shape = -1,2
	# wave_data = wave_data.T
	#time 也是一个数组，与wave_data[0]或wave_data[1]配对形成系列点坐标
	time = np.arange(0,nframes)*(1.0/framerate)
	# time = time[0:int(len(time)/2)] #双声道
	time = time[0:int(len(time))] #单声道
	print("len(time):",len(time),"len(wav)",len(wave_data));
	plt.plot(time, wave_data)
	# plt.show()
	
	# data = wave_data.tostring()
	data = str_data.decode()
	wf1 = wave.open("./hello11s_w.wav",'wb')
	wf1.setnchannels(channels)
	wf1.setsampwidth(sampwidth)
	wf1.setframerate(framerate)
	wf1.writeframes("".join(data)) # ""中间不能有空格，不然语音录入会有很多中断。
	wf1.close()
	
	
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
	plt.show()
	
	
	 
	
	
if __name__ == "__main__":
    # Func1()
	Func2()
	
	
# Wave_read.readframes(n)
# Reads and returns at most n frames of audio, as a string of bytes.