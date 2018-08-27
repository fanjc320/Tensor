import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real, freq, sp.imag)
plt.show()


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