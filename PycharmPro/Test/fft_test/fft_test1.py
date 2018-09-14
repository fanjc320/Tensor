import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,1400)      

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*200*x) + 3*np.sin(2*np.pi*300*x)+5*np.sin(2*np.pi*600*x)

yf=fft(y)                     #快速傅里叶变换
# yf=abs(fft(y))                # 取绝对值
xf = np.arange(len(y))        # 频率

plt.subplot(221)
plt.stem(x[0:50],y[0:50])
plt.title('Original wave')

plt.subplot(222)
plt.stem(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

plt.subplot(223)
plt.stem(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

_yfi = np.fft.ifft(yf);

plt.subplot(224)
plt.stem(xf[0:50],_yfi[0:50])
print("len:",len(xf),len(_yfi));

plt.show()