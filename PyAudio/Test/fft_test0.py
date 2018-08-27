import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

class Counter:
	counter = 0;


def plotWithN(n=100,number=0):
	# a = np.arange(0.0,1.0,0.02)
	a = np.linspace(0,1,n);
	b = np.sin(2*np.pi*10*a);
	b1 = np.sin(2*np.pi*50*a);
	
	number += 4;
	col = 2;
	cnt = 1;
	
	plt.subplot(number,col,cnt);
	plt.plot(a[0:100],b[0:100]);
	plt.title("-----周长0.1s时域--------");
	
	cnt=cnt+1;
	plt.subplot(number,col,cnt);
	plt.plot(a[0:100],b1[0:100]);
	plt.title("-----周长0.02s时域------");
	
	c = fft(b);
	# print(" 周长0.1s real:",c.real," image:",c.imag);
	cnt=cnt+1;
	plt.subplot(number,col,cnt);
	plt.plot(a,c);
	plt.title("----周长0.1s频域----");
	
	c1 = fft(b1);
	# print(" 周长0.02s real:",c1.real," image:",c1.imag);
	cnt=cnt+1;
	plt.subplot(number,col,cnt);
	plt.plot(a,c1);
	plt.title("-----周长0.02s频域---");

cnt = 0;
# @static_vars(counter = 0)
def plotWithNN(f=10,n=100,number=4):
	# a = np.arange(0.0,1.0,0.02)
	a = np.linspace(0,1,n);
	b = np.sin(2*np.pi*f*a);
	
	col = 2;
	Counter.counter+=1;
	global cnt;
	cnt+=1;
	print("-----------counter:",cnt," ",Counter.counter);
	plt.subplot(number,col,cnt);
	plt.plot(a[0:100],b[0:100]);
	plt.title("----频率:"+str(f));
	
	c = fft(b);
	# print(" 周长0.1s real:",c.real," image:",c.imag);
	cnt+=1;
	plt.subplot(number,col,cnt);
	plt.plot(a,c);
	plt.title("----频率:"+str(f));
	
	number += 4;
	
# plotWithN(100,0);
# plotWithN(500);

plotWithNN(10,100,4);
plotWithNN(20);

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