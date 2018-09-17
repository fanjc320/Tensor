import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import seaborn

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

sampling_period = 2;#采样总时间1s
interval = 0.01#采样间隔0.01s 采样频率 1/interval = 100
n = sampling_period/interval #帧数 ###### a----

time = np.arange(0, sampling_period, interval)#采样时间点分部,第二个参数是采样的总周期，第三个参数是采样间隔，采样频率是1/第三个参数，
x = np.sin(2 * np.pi * 20 * time)
y = np.fft.fft(x)
print("x:",x)
print("y:",y)
show(x, y,sampling_period)

def func1():
    x = np.linspace(0, 1, 14)  # linespace 返回array
    y = 7 * np.sin(2 * np.pi * 1 * x) + 3 * np.sin(2 * np.pi * 3 * x) + 5 * np.sin(2 * np.pi * 6 * x)
    print(" len(x)", len(x), " len(y) ", len(y));
    # yf=abs(fft(y))                # 取绝对值
    yf = fft(y)  # 取绝对值
    xf = np.arange(len(y))  # 频率
    plt.subplot(321)
    plt.stem(x, y)
    plt.title('Original wave')
    plt.subplot(322)
    plt.stem(xf, yf, 'r')
    plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
    plt.tight_layout()
    plt.show()
