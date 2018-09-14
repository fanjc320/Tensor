import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn


#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,14)  #linespace 返回array

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*1*x) + 3*np.sin(2*np.pi*3*x)+5*np.sin(2*np.pi*6*x)
print("sum(y)",sum(abs(y)),sum(y*y))

yy=fft(y)                     #快速傅里叶变换
yreal = yy.real               # 获取实数部分
yimag = yy.imag               # 获取虚数部分

yf=abs(fft(y))                # 取绝对值
print("sum(yf)",sum(yf),sum(yf*yf))
yf1=abs(fft(y))/len(x)           #归一化处理
yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间
yf3 = yf1[range(int(len(x)))]

xf = np.arange(len(y))        # 频率
xf1 = xf
xf2 = xf[range(int(len(x)/2))]  #取一半区间
xf3 = xf[range(int(len(x)))]

plt.subplot(321)
plt.stem(x,y)
plt.title('Original wave')

plt.subplot(322)
plt.stem(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

plt.subplot(323)
plt.stem(xf1,yf1,'g')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

plt.subplot(324)
plt.stem(xf2,yf2,'b')
plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')

plt.subplot(325)
plt.stem(xf3,yf3,'b')
plt.title('FFT of Mixed wave*2)',fontsize=10,color='#F08080')

print(" len(x)",len(x)," len(y) ",len(y),"xf2",xf2);

plt.show()