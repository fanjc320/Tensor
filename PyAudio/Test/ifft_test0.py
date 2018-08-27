import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

cnt = 0;
# @static_vars(counter = 0)
def iplotWithNN(f=10,n=100,number=4):
	# a = np.arange(0.0,1.0,0.02)
	
	t = np.arange(60)
	n = np.zeros((60,),dtype = complex)
	print("n:",n);
	n[40:60] = np.exp(1j*np.random.uniform(0,2*np.pi,(20,))) #元组中只包含一个元素时，需要在元素后面添加逗号
	print("n[40:60]:",n[40:60]);
	s=np.fft.ifft(n)
	plt.plot(t,s.real,'b-',t,s.imag,'r--',t,abs(s),'g')
	
	plt.legend(('real','imaginary','abs(fft)'))
	
# plotWithN(100,0);
# plotWithN(500);

iplotWithNN(10,100,4);
# iplotWithNN(20);

plt.tight_layout()
plt.show();

	

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