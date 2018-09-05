import numpy as np
import wave
import pyaudio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import fft,ifft
# import common
from Common.common import wavReads
import array
plt.tight_layout()

# fft 频谱去掉高频，然后ifft
def Func1():
	# fjc_record(OutFile="test0.wav")
	plt.tight_layout()
	
	fig = plt.figure()
	
	wavdata,wavtime = wavReads("./res/hello11s.wav")
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
	
	wf = wave.open("./res/hello11s.wav", "rb")
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
	# plt.plot(time, wave_data)
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
	# yyii = yyi.astype(np.int32),180.9999999,会变成180
	yyii = np.around(yyi).astype(np.short)
	print("---",yyi[0]," ",int(round(yyi[0]))," yyii ",(yyii[:20]).astype(np.int32))
	plt.subplot(211)
	plt.plot(time[:clip], wave_data[:clip])
	plt.subplot(212)
	plt.plot(time[:clip], yyii[:clip],'b-')
	
	# data = wave_data.tostring()
	# data = str_data.decode('iso-8859-1')
	Numb = 20;
	
	print("wave_data(fft)->yf(ifft)->yyi",yyi[:Numb])
	print("wave_data(fft)->yf(ifft)->yyi(astypeint)->yyii",yyii[:Numb])

	wf1 = wave.open("./hello11s_w.wav",'wb')
	wf1.setnchannels(channels)
	wf1.setsampwidth(sampwidth)
	wf1.setframerate(framerate)
	# wf1.writeframes(str)
	print("len===",len(wave_data),len(yyii))
	res = wave_data-yyii
	test= wave_data.tolist()
	res = res.tolist()
	print("-----------00-------",np.where(wave_data!=0),"wave_data type:",type(wave_data[0]))
	print("-----------11-------",np.where(yyii!=0),"yyii type:",type(wave_data[0]))
	# print("-----------22-------",res)
	print("--------!=0----------",np.where(res!=0))
	print("--------==0----------",np.where(res==0))
	# print(" np.where ",type(test),np.where(test!=0))
	
	str = wave_data.tostring()
	print("wave_data(tostring)->str",str[:Numb]);
	stri = yyii.tostring()
	print("    yyii(tostring)->stri",stri[:Numb]);
	
	wf1.writeframes(yyii.tostring())
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
	
def TestWhere():
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
	
def FromToStr():
	ali = array.array('i',[1,2,3])
	str = ali.tostring();
	print("ali",ali," tostring()",str);
	arr = np.fromstring(str,dtype=np.int)
	print("arr",arr)
	
if __name__ == "__main__":
    # Func1()
	# FromToStr()
	Func2()
	# TestWhere()
	
# Wave_read.readframes(n)
# Reads and returns at most n frames of audio, as a string of bytes.



# ifft Returns:	out : complex ndarray
# The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.
