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

