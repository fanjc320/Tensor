'''
播放特定频率 NoneBlocking 范例
'''

import numpy as np
import pyaudio
import time

CHUNK = 1024
CH = 1

def sine(frequency,t,sampleRate):
	n = int(t*sampleRate)
	interval = 2*np.pi*frequency/sampleRate
	return np.sin(np.arange(n)*interval)
	
def sliceData(frame_count,channels):
	data = sine(frequency=1000,t=3,sampleRate=44100)
	# 因會再轉換為 np.float32，故無需乘上 sampleBytes
	size = channels * frame_count
	while True:
		dataSlice = data[:size]
		# 此時小數點會用 np.float32 4byte 表示，故資料長度會變為 4 倍
		dataBytes = dataSlice.astype(np.float32).tostring()
		yield dataBytes
		data = np.delete(data,range(size))
		
dataGen = sliceData(CHUNK,CH)
def callback(in_data,frame_count,time_info,status):
	global dataGen
	data = next(dataGen)
	return (data,pyaudio.paContinue)
	
	
	

if __name__ == '__main__':
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paFloat32,channels=CH,rate=44100,output=True,
	frames_per_buffer = CHUNK,stream_callback = callback)
	
	for i in range(1000):
		print(i)
		
	stream.stop_stream()
	stream.close()
	p.terminate()
 

