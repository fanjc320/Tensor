import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

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

TestShow()
# range()函数

# 函数说明： range(start, stop[, step]) -> range object，根据start与stop指定的范围以及step设定的步长，生成一个序列。
# 参数含义：start:计数从start开始。默认是从0开始。例如range（5）等价于range（0， 5）;
              # end:技术到end结束，但不包括end.例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
              # scan：每次跳跃的间距，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
# 函数返回的是一个range object
# >>> range(0,5)                 #生成一个range object,而不是[0,1,2,3,4]   
# range(0, 5)     
# >>> c = [i for i in range(0,5)]     #从0 开始到4，不包括5，默认的间隔为1  
# >>> c  
# [0, 1, 2, 3, 4]  

# arrange()函数

# 函数说明：arange([start,] stop[, step,], dtype=None)根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。 dtype : dtype
# >>> np.arange(3)  
  # array([0, 1, 2])  
  # >>> np.arange(3.0)  
  # array([ 0.,  1.,  2.])  
  # >>> np.arange(3,7)  
  # array([3, 4, 5, 6])  
  # >>> np.arange(3,7,2)  
  # array([3, 5])  