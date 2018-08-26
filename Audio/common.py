import wave
#import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import pyaudio
import wave as we
 
def wavread(path):
    wavfile =  we.open(path,"rb")
    params = wavfile.getparams()
    print("params:",params)
    framesrate,frames= params[2],params[3]
    #readframes()
    #得到每一帧的声音数据，返回的值是二进制数据，在python中用字符串表示二进制数据datawav = wavfile.readframes(frameswav)
    datawav = wavfile.readframes(frames)
    wavfile.close()
    datause = np.fromstring(datawav,dtype = np.short)
    datause.shape = -1,2
    datause = datause.T

    time = np.arange(0, frames) * (1.0/framesrate)
    print("time:",time.shape,"len data:",len(datause))
    return datause,time
	
# 单声道
def wavreads(path):
	wavfile =  we.open(path,"rb")
	params = wavfile.getparams()
	print("params:",params)
	framesrate,frames= params[2],params[3]
	#readframes()
	#得到每一帧的声音数据，返回的值是二进制数据，在python中用字符串表示二进制数据datawav = wavfile.readframes(frameswav)
	datawav = wavfile.readframes(frames)
	wavfile.close()
	datause = np.fromstring(datawav,dtype = np.short)
	
	time = np.arange(0, frames) * (1.0/framesrate)
	print("time:",time.shape,"len(data):",len(datause),"data:",datause)
	return datause,time 

#wav文件读取
def fjc_record(OutFile = "./fjc.wav",Seconds = 5):
	#参数定义
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT,
	channels=CHANNELS,
	rate=RATE,
	input=True,
	frames_per_buffer=CHUNK)
	
	print("* recording")
	
	frames = []
	for i in range(0, int(RATE / CHUNK * Seconds)):
		data = stream.read(CHUNK)
		frames.append(data)
	
	print("* done recording")
	
	stream.stop_stream()
	stream.close()
	p.terminate()
	wf = wave.open(OutFile, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	
def fjc_record_with(frames,OutFile = "./fjc.wav",Seconds = 5):
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT,
	channels=CHANNELS,
	rate=RATE,
	input=True,
	frames_per_buffer=CHUNK)
	
	stream.stop_stream()
	stream.close()
	p.terminate()
	wf = wave.open(OutFile, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	
# fjc_record(Seconds=15)