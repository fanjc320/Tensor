import numpy as np
import pyaudio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import fft,ifft
# import common
from common import fjc_record,wavread,wavreads,fjc_record_with
plt.tight_layout()

def main():
	# fjc_record(OutFile="test0.wav")
	plt.tight_layout()
	
	fig = plt.figure()
	
	wavdata,wavtime = wavreads("./hello11s.wav")
	plt.title("hello11.wav's Frames")
	plt.subplot(411)
	plt.plot(wavtime, wavdata,color = 'green')
	# plt.show()
	
	fjc_record_with(wavdata)
	
	yf=fft(wavdata)
	xf = np.arange(len(wavdata))
	plt.subplot(412)
	plt.plot(xf[:1000],yf[:1000],'r')
	# plt.show()
	
	plt.subplot(413)
	yi = ifft(yf)
	plt.plot(wavtime,yi, 'g')
	# plt.show()
	
	plt.figure(num=3,figsize=(8,5),)
	yf_ = yf[100:500]
	plt.subplot(411)
	plt.plot(np.arange(len(yf_)),yf_,'r')
	yi_ = ifft(yf_)
	plt.subplot(412)
	plt.plot(np.arange(len(yi_)),yi_,'b')
	
	
	plt.show()
	plt.tight_layout()
	
if __name__ == "__main__":
    main()