'''
播放特定频率 Blocking 范例
'''

import numpy as np
import pyaudio
from matplotlib import pyplot as plt

def sine(frequency,t,sampleRate):
	n = int(t*sampleRate)
	interval = 2*np.pi*frequency/sampleRate
	x = np.arange(n)*interval
	y = np.sin(np.arange(n)*interval)
	plt.scatter(x[0:100],y[0:100])
	plt.show()
	return np.sin(np.arange(n)*interval)
	
def play_tone(stream,frequency=440,t=1,sampleRate = 44100):
	data = sine(frequency,t,sampleRate)
	stream.write(data.astype(np.float32).tostring())
	
if __name__ == '__main__':
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paFloat32,channels=1,rate=44100,output=True)
	play_tone(stream,frequency=1000,t=2)
	stream.close()
	p.terminate()


