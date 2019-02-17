#播放wave文件的实例

#引入库
import wave
import pyaudio
import time
import numpy as np
import pylab as pl

#定义数据流块
chunk = 1024

#只读方式打开wav文件
f = wave.open(r"./心如刀割_single.wav","rb")

p = pyaudio.PyAudio()

#打开数据流
format = p.get_format_from_width(f.getsampwidth())
channels = f.getnchannels()
rate = f.getframerate()
# sampwidth =2 2个字节
print("format,channels,rate,sampwidth",format,channels,rate,f.getsampwidth())

params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
print("nchannels,sampwidth,framerate,nframes",nchannels,sampwidth,framerate,nframes)

stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
channels = f.getnchannels(),
rate = f.getframerate(),
output = True)

# stream = p.open(format,
# channels = f.getnchannels(),
# rate = f.getframerate(),
# output = True)


#写声音输出流到声卡进行播放
count = 0
#b""也行 ""不行
#读取数据,然后放入输出流播放
'''
data = f.readframes(chunk)
while data !=b'':
	stream.write(data)
	data = f.readframes(chunk)
	#time.sleep(1)
	count = count+1
	#print("data",data)
'''	

print("count:",count)
#停止数据流
stream.stop_stream()
stream.close()
print("----------------------")
#关闭 PyAudio
p.terminate()

# 读取波形数据
str_data = f.readframes(nframes)
f.close()

#将波形数据转换为数组
wave_data = np.fromstring(str_data, dtype=np.short)
# wave_data.shape = -1, 2 #数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
# wave_data = wave_data.T
time = np.arange(0, nframes) * (1.0 / framerate)
print("len0 len1 len2 ",len(str_data),len(wave_data),len(time));

# 绘制波形
pl.subplot(211) 
pl.plot(time, wave_data)
# pl.subplot(212) 
# pl.plot(time, wave_data[1], c="g")
pl.xlabel("time (seconds)")
pl.show()

# readframes：读取声音数据，传递一个参数指定需要读取的长度(以取样点为单位)，readframes返回的是二进制数据(一大堆bytes)，在Python中用字符串表示二进制数据：
# str_data = f.readframes(nframes)
# 接下来需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
# wave_data = np.fromstring(str_data, dtype=np.short)