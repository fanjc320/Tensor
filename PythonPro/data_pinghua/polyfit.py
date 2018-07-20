
# 多项式拟合
# https://blog.csdn.net/a1212125/article/details/78023550
#一系列的散点可以用函数去拟合,而任何一个连续可微函数都可以展开为一个多次多项式表示(微积分中的泰勒展开式). 

import numpy as np
import sys,os
import matplotlib.pyplot as plt

datas = np.loadtxt('../data/000001.csv',
delimiter=',',skiprows=(2),usecols=(6,),unpack=True)
datas = datas[:30]
t=np.arange(len(datas))
poly = np.polyfit(t,datas,int(sys.argv[1]))
print("polynomial fit",poly)

#下面是输出结果,给出了每一项的系数
#polynomial fit [ -1.35119250e-08   6.01590681e-05  -9.04311066e-02   1.09359604e+02]

#np.polyval()方法是获取poly这个多项式的对应x的值

plt.plot(t,datas)
plt.plot(t,np.polyval(poly,t))
plt.show()