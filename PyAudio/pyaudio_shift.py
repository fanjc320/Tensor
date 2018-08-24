# 傅立叶变换(np.fft)

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0,2*np.pi,30)#创建一个包含30个点的余弦波信号
# wave = np.cos(x)
## 傅立叶变换 ###
# transformed = np.fft.fft(wave)
## 逆傅立叶变换 ###
# print(np.all(np.abs(np.fft.ifft(transformed)-wave)<10**(-9)))#对变换后的结果应用ifft函数，应该可以近似地还原初始信号
# plt.plot(x,transformed.real)              # 注意，fft变换结果是复数，要取实部

# x = np.linspace(0,2*np.pi,30)
# wave = np.cos(x)#创建一个包含30个点的余弦波信号
# transformed = np.fft.fft(wave)#使用fft函数对余弦波信号进行傅里叶变换
## 频移 ###
# shifted = np.fft.fftshift(transformed)#使用fftshift函数进行移频操作。
## 逆频移
# print(np.all((np.fft.ifftshift(shifted)-transformed)<10**(-9))) #用ifftshift函数进行逆操作，这将还原移频操作前的信号。
# plt.plot(x,shifted.real)
# plt.show()

# 移频
# numpy.fft模块中的fftshift函数可以将FFT输出中的直流分量移动到频谱的中央。ifftshift函数则是其逆操作。
import numpy as np
from matplotlib.pyplot import plot, show
x = np.linspace(0, 2 * np.pi, 30) 
wave = np.cos(x)  #创建一个包含30个点的余弦波信号。
transformed = np.fft.fft(wave)  #使用fft函数对余弦波信号进行傅里叶变换。
shifted = np.fft.fftshift(transformed) #使用fftshift函数进行移频操作。
print (np.all((np.fft.ifftshift(shifted) - transformed) < 10 ** -9))  #用ifftshift函数进行逆操作，这将还原移频操作前的信号。
plot(transformed, lw=2)
plot(shifted, lw=3)
show()    #使用Matplotlib分别绘制变换和移频处理后的信号。

# numpy.fft.fftshift(x, axes=None)[source]
# Shift the zero-frequency component to the center of the spectrum.

# This function swaps half-spaces for all axes listed (defaults to all). Note that y[0] is the Nyquist component only if len(x) is even.

# Parameters:	
# x : array_like
# Input array.

# axes : int or shape tuple, optional
# Axes over which to shift. Default is None, which shifts all axes.

# Returns:	
# y : ndarray
# The shifted array.

# See also
# ifftshift
# The inverse of fftshift.
# Examples

# >>>
# >>> freqs = np.fft.fftfreq(10, 0.1)
# >>> freqs
# array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
# >>> np.fft.fftshift(freqs)
# array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
# Shift the zero-frequency component only along the second axis:

# >>>
# >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
# >>> freqs
# array([[ 0.,  1.,  2.],
       # [ 3.,  4., -4.],
       # [-3., -2., -1.]])
# >>> np.fft.fftshift(freqs, axes=(1,))
# array([[ 2.,  0.,  1.],
       # [-4.,  3.,  4.],
       # [-1., -3., -2.]])