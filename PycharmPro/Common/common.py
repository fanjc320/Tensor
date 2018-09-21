import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
import wave

#对双声道抽取左声道
def wavRead(path):
    wavfile =  wave.open(path,"rb")
    params = wavfile.getparams()
    print("params:",params)
    framesra,frameswav= params[2],params[3]
    #readframes()
    #得到每一帧的声音数据，返回的值是二进制数据，在python中用字符串表示二进制数据datawav = wavfile.readframes(frameswav)
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav,dtype = np.short)
    datause.shape = -1,2
    datause = datause.T

    time = np.arange(0, frameswav) * (1.0/framesra)
    print("time:",time.shape)
    return datause[0],time

#用来读取单声道音频
def wavReads(path):
    wavfile = wave.open(path,"rb")
    params = wavfile.getparams()
    print("params:",params)
    framesra,nframes= params[2],params[3]
    #readframes()
    #得到每一帧的声音数据，返回的值是二进制数据，在python中用字符串表示二进制数据datawav = wavfile.readframes(frameswav)
    datawav = wavfile.readframes(nframes)
    wavfile.close()
    data = np.fromstring(datawav,dtype = np.short)

    time = np.arange(0, nframes) * (1.0/framesra)
    print("time:",time.shape)
    return data,time

def ifft_asShort(data):
	wav_ifft1 = ifft(data).real
	wav_ifft_11 = np.around(wav_ifft1).astype(np.short)
	
def wavWrite(wavData,outfile,channels,sampwidth,framerate):
	wf1 = wave.open(outfile,'wb')
	wf1.setnchannels(channels)
	wf1.setsampwidth(sampwidth)
	wf1.setframerate(framerate)
	wavStr = wavData.tostring()
	wf1.writeframes(wavStr)
	wf1.close()
	
def show(ori_func,ft,sampling_period = 5,half=True):
	n = len(ori_func)
	interval = sampling_period/n
	plt.subplot(2,1,1)
	plt.plot(np.arange(0,sampling_period,interval),ori_func,'black')
	plt.xlabel('Time'),plt.ylabel('Amplitude')
	
	plt.subplot(2,1,2)
	
	if(half):
		# frequency = np.arrange(n/2)/(n*interval)
		frequency = np.arange(n/2)/sampling_period #fjc
		# nfft = abs(ft[range(int(n/2))]/n)
		nfft = abs(ft[range(int(n/2))])# fjc
		# print("n:",n,"half range:",range(int(n/2)))
	else:
		frequency = np.arange(n)/sampling_period
		nfft = abs(ft)
		print("ft:",ft)
		# print(" ft[range(n)]:",ft[range(n)]);
		# print("n:",n," frequency:",frequency," range:",range(int(n)))
	
	plt.plot(frequency,nfft,'red')
	plt.xlabel('Freq (Hz)'),plt.ylabel('Amp. Spectrum')
	plt.show()

def show(ori_func, ft, sampling_period):#sampling_period 采样时间总长度
    n = len(ori_func) #n 采样帧数,即采样所有时间点的个数
    interval = sampling_period / n ###### a----采样间隔
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, sampling_period, interval), ori_func, 'black')
    plt.xlabel('Time'), plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    #frequency = np.arange(n / 2) / (n * interval) #等价于frequency = np.arange(n/2)/sampling_period,###### a----
    frequency = np.arange(n) / (n * interval) #等价于frequency = np.arange(n)/sampling_period,###### a----
    print("show n:",n,"interval:",interval);
    print("frequency:",frequency);
    #nfft = abs(ft[range(int(n / 2))] / n) # /2是因为频率的对称性,/n是归一化?
    nfft = abs(ft[range(int(n))] / n) # /2是因为频率的对称性,/n是归一化?
    plt.plot(frequency, nfft, 'red')
    plt.xlabel('Freq(Hz)'), plt.ylabel('Amp. Spectrum')
    plt.show()

# def FromToStr():
# 	ali = array.array('i',[1,2,3])
# 	str = ali.tostring();
# 	print("ali",ali," tostring()",str);
# 	arr = np.fromstring(str,dtype=np.int)
# 	print("arr",arr)

def TestShow():
	time = np.arange(0,5,.05) # f=0.005 就是 1s 采样200个,共5s,就是1000个采样点
	x = np.sin(2*np.pi*1*time)
	# y = np.fft.fft(x)
	# show(x,y)
	# show(x,y,half=True)
	
	x2 = np.sin(2*np.pi*6*time)
	x3 = np.sin(2*np.pi*18*time)
	x += x2+x3
	y = np.fft.fft(x)
	show(x,y)
	
	# 生成方波，振幅是 1，频率为 10Hz
	# 我们的间隔是 0.05s，每秒有 200 个点
	# 所以需要每隔 20 个点设为 1
	# x = np.zeros(len(time))
	# x[::20] = 1
	# print("x:",x);
	# y = np.fft.fft(x)
	# show(x, y)

	# 生成脉冲波
	# x = np.zeros(len(time)) 
	# x[380:400] = np.arange(0, 1, .05) 
	# x[400:420] = np.arange(1, 0, -.05) 
	# y = np.fft.fft(x) 
	# show(x, y) 
	
	# 生成随机数
	# x = np.random.random(100) 
	# y = np.fft.fft(x) 
	# show(x, y) 

# TestShow()