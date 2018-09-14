import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn

class Counter:
	counter = 0;

cnt = 0;
def plotWithN(framerate=100,freq = 10, number=0):
	# a = np.arange(0.0,1.0,0.02)
	a = np.linspace(0,1,framerate);
	b = np.sin(2*np.pi*freq*a);
	b1 = np.sin(2*np.pi*freq*2*a);
	
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

# @freq 频率 @number 图像个数
# 此函数用来画不同频率的sin函数及其fft变换

def iplotWithNN(n=100):

	t = np.arange(60)
	n = np.zeros((60,), dtype=complex)
	print("n:", n);
	n[40:60] = np.exp(1j * np.random.uniform(0, 2 * np.pi, (20,)))  # 元组中只包含一个元素时，需要在元素后面添加逗号
	print("n[40:60]:", n[40:60]);
	s = np.fft.ifft(n)
	plt.plot(t, s.real, 'b-', t, s.imag, 'r--', t, abs(s), 'g')

	plt.legend(('real', 'imaginary', 'abs(fft)'))


# plotWithN(100,0);
# plotWithN(500);

iplotWithNN(20);
iplotWithNN();
	
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


iplotWithNN(10,100,4);
plt.tight_layout();
plt.show();

