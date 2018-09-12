import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn


#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,140)  #linespace 返回array

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*10*x) + 3*np.sin(2*np.pi*30*x)+5*np.sin(2*np.pi*60*x)
print("sum(y)",sum(abs(y)),sum(y*y))

yy=fft(y)                     #快速傅里叶变换
# yreal = yy.real               # 获取实数部分
# yimag = yy.imag               # 获取虚数部分

yf=abs(fft(y))                # 取绝对值
print("sum(yf)",sum(yf),sum(yf*yf))
yf3 = yf[range(int(len(x)))]

xf = np.arange(len(y))        # 频率
xf1 = xf
xf2 = xf[range(int(len(x)/2))]  #取一半区间
xf3 = xf[range(int(len(x)))]

plt.subplot(311)
plt.stem(x,y)
plt.title('Original wave')

plt.subplot(312)
plt.stem(xf3,yf3,'b')
plt.title('FFT of Mixed wave*2)',fontsize=10,color='#F08080')

print(" len(x)",len(x)," len(y) ",len(y),"xf2",xf2);

plt.show()