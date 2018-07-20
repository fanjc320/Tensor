#简单移动平均线是计算与等权重的指示函数的卷积,也可以不等权重. 
#1.用ones函数创建一个元素均为1的数组,然后对整个数组除以N,得到等权重. 
#2.使用权值,调用convolve函数. 
#3.从convolve函数分安徽的数组中取出中间的长度为N的部分(即两者作卷积运算时完全重叠的区域.) 
#4.使用matplotlib绘图


import numpy as np
import matplotlib.pyplot as plt
import sys

N=int(sys.argv[1])
weights = np.ones(N)/N
print("WEIGHTS",weights)

c=np.loadtxt('../data/000001.csv',delimiter=',',
skiprows =(2),usecols=(2,),unpack=True)
c = c[:30]
print("c:",c);
#convolve !!!!
fjc = np.convolve(weights,c);
print("fjc:",fjc)
sam = np.convolve(weights,c)[N-1:-N+1]
t=np.arange(N-1,len(c))
print("sam:",sam)
print("t",t)

plt.plot(t,c[N-1:],lw=1.0)
plt.plot(t,sam,lw=2.0)
plt.show()


# 汉宁窗是一个加权余弦窗函数.numpy.hanning(M) Return the Hanning window.

# Parameters: M : int 
# Number of points in the output window. If zero or 
# Returns: out : ndarray, shape(M,) 
# The window, with the maximum value normalized to one (the value one 
# appears only if M is odd).


#1.调用hanning函数计算权重,生成一个长度为N的窗口,输入参数N

N=int(sys.argv[1])
weights=np.hanning(N)
print(weights)
#2.使用convolve,进行卷积运算.然后绘图.

import numpy as np
import matplotlib.pyplot as plt
import sys

N=int(sys.argv[1])
weights=np.hanning(N)
print("WEIGHTS",weights)

c=np.loadtxt('../data/000001.csv',delimiter=',',
skiprows=(2),usecols=(2,),unpack=True)
c = c[:30]
sam=np.convolve(weights/weights.sum(),c)[N-1:-N+1]
t=np.arange(N-1,len(c))

plt.plot(t,c[N-1:],lw=1.0)
plt.plot(t,sam,lw=2.0)
plt.show()


# 多项式曲线拟合（Polynomial Curve Fitting）
# 标签：监督学习
# https://blog.csdn.net/daunxx/article/details/51588262