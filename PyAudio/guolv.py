import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

def guolv0():
	# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
	x=np.linspace(0,1,1400)      
	
	# 设置需要采样的信号，频率分量有180，390和600
	y=7*np.sin(2*np.pi*180*x) + 2*np.sin(2*np.pi*390*x)+5*np.sin(2*np.pi*600*x)
	
	yy=fft(y)                     #快速傅里叶变换
	yreal = yy.real               # 获取实数部分
	yimag = yy.imag               # 获取虚数部分
	
	
	plt.subplot(421)
	plt.plot(x[0:50],y[0:50])   
	plt.title('Original wave')
	
	xf = np.arange(len(y))        # 频率
	plt.subplot(422)
	plt.plot(xf,yy,'r')
	plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表
	
	out = []
	result = np.where(np.absolute(yy)<200, 0, yy) # 1. np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
	
	plt.subplot(423)
	plt.plot(xf,yy,'g')
	plt.subplot(424)
	plt.plot(xf,result,'y')

def guolv1():

	x=np.linspace(0,1,140)      
	
	#设置需要采样的信号，频率分量有180，390和600
	y=7*np.sin(2*np.pi*18*x) + 2*np.sin(2*np.pi*39*x)+5*np.sin(2*np.pi*60*x)
	
	yy=fft(y)                     #快速傅里叶变换
	yreal = yy.real               # 获取实数部分
	yimag = yy.imag               # 获取虚数部分
	
	
	plt.subplot(421)
	plt.plot(x[0:50],y[0:50])   
	plt.title('Original wave')
	
	xf = np.arange(len(y))        # 频率
	plt.subplot(422)
	plt.plot(xf,yy,'r')
	plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表
	
	out = []
	print("yy before:",yy);
	result = np.where(np.absolute(yy)<200, 0, yy) # 1. np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
	print("yy after:",yy);
	
	plt.subplot(423)
	plt.plot(xf,yy,'g')
	plt.subplot(424)
	plt.plot(xf,result,'y')

def guolv2():

	x=np.linspace(0,1,140)      
	
	#设置需要采样的信号，频率分量有180，390和600
	y=7*np.sin(2*np.pi*18*x) + 2*np.sin(2*np.pi*39*x)+5*np.sin(2*np.pi*60*x)
	
	yy=fft(y)                     #快速傅里叶变换
	yreal = yy.real               # 获取实数部分
	yimag = yy.imag               # 获取虚数部分
	
	
	plt.subplot(421)
	plt.plot(x[0:50],y[0:50])   
	plt.title('Original wave')
	
	xf = np.arange(len(y))        # 频率
	plt.subplot(422)
	plt.plot(xf,yy,'r')
	plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表
	
	out = []
	print("yy before:",yy);
	result = np.where(np.absolute(yy)<200, 0, yy) # 1. np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
	print("yy after:",yy);
	
	plt.subplot(423)
	plt.plot(xf,yy,'g')
	plt.subplot(424)
	plt.plot(xf,result,'y')
	
# guolv0()
# guolv1()
	
plt.show()