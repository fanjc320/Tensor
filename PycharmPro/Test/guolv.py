import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

plt.tight_layout()
#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,1400)      

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)

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
 
plt.subplot(423)
plt.plot(xf[:100],yy[:100],'b')
plt.subplot(424)
plt.plot(xf[:500],yy[:500],'b')

yy_ = ifft(yy)
plt.subplot(425)
plt.plot(xf[:50],yy_[:50],'b')

yy_ = ifft(yy[:500])
plt.subplot(426)
plt.plot(xf[:50],yy_[:50],'g')


plt.show()