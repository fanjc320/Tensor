import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn


def show(ori_func,ft,sampling_period = 5):
    n = len(ori_func)
    interval = sampling_period/n
    plt.subplot(2,1,1)
    plt.plot(np.arange(0,sampling_period,interval),ori_func,'black')
    plt.xlabel('Time'),plt.ylabel('Amplitude')

    plt.subplot(2,1,2)
    frequency = np.arange(n/2)/(n*interval)
    nfft = abs(ft[range(int(n/2))]/n)
    plt.plot(frequency,nfft,'red')
    plt.xlabel('Freq(Hz)'),plt.ylabel('Amp. Spectrum')
    plt.show()

time = np.arange(0,5,0.005)
x = np.sin(2*np.pi*1*time)
y = np.fft.fft(x)
show(x,y)

#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,14)  #linespace 返回array

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*1*x) + 3*np.sin(2*np.pi*3*x)+5*np.sin(2*np.pi*6*x)

print(" len(x)",len(x)," len(y) ",len(y));

yy=fft(y)                     #快速傅里叶变换

# yf=abs(fft(y))                # 取绝对值
yf=fft(y)                # 取绝对值

xf = np.arange(len(y))        # 频率
xf1 = xf

plt.subplot(321)
plt.stem(x,y)
plt.title('Original wave')

plt.subplot(322)
plt.stem(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

plt.tight_layout()
plt.show()