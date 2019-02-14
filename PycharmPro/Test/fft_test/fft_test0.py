import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from common import *
import seaborn

sampling_period = 20;#采样总时间1s
interval = 0.001#采样间隔0.01s 采样频率 1/interval = 100
n = sampling_period/interval #帧数 ###### a----

time = np.arange(0, sampling_period, interval)#采样时间点分部,第二个参数是采样的总周期，第三个参数是采样间隔，采样频率是1/第三个参数，
x = 100*np.sin(2 * np.pi * 20 * time)
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

func1()