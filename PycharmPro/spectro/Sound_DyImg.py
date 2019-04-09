# coding=utf8
import wave
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import pyaudio
import wave as we
from scipy.fftpack import fft,ifft


def fjc_record(OutFile="../res/fjc.wav", Seconds=1):
    # 参数定义
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 # 通过play函数测试得出结论 channels=1 占用2个字节，channels=2占用4个字节
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
        intdata = np.fromstring(data, dtype=np.short)
        # if i==0:
        #     print(type(data),data)
            # print(data.split(b'|'))
            # for idx in range(0,len(data),2):
            #     chunk_one = data[idx:idx+2]
            #     print("i:",idx,chunk_one,int.from_bytes(chunk_one,byteorder='big'))
        frames.append(intdata)
        # print(intdata)
        spec = fft(intdata)
        print(np.array(spec).shape)

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


fjc_record()


# 读取大小为1，查看输出内容，编码的宽度b'\x00\x00'
def play():
    # 读取wav文件，查看编码方式，只读1或几帧
    wavfile = wave.open(r"./res/yusheng1.wav", "rb")
    params = wavfile.getparams()
    print("params:", params)
    framesra, nframes = params[2], params[3]
    # datawav = wavfile.readframes(nframes)
    datawav = wavfile.readframes(1)
    print("datawav:",type(datawav), datawav)
    return
    wavfile.close()
    data = np.fromstring(datawav, dtype=np.short)
    time = np.arange(0, nframes) * (1.0 / framesra)
    plt.plot(time,data)
    plt.show()
    #############

    # 用文本文件记录wave模块解码每一帧所产生的内容。注意这里不是保存为二进制文件
    dump_buff_file = open(r"./res/fjc.dup", 'w')

    chunk = 4  # 指定WAV文件的大小
    wf = wave.open(r"./res/yusheng1.wav", 'rb')  # 打开WAV文件



    p = pyaudio.PyAudio()  # 初始化PyAudio模块

    # 打开一个数据流对象，解码而成的帧将直接通过它播放出来，我们就能听到声音啦
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    data = wf.readframes(chunk)  # 读取第一帧数据
    print(data)  # 以文本形式打印出第一帧数据，实际上是转义之后的十六进制字符串



    # 播放音频，并使用while循环继续读取并播放后面的帧数
    # 结束的标志为wave模块读到了空的帧
    while data != b'':
        stream.write(data)  # 将帧写入数据流对象中，以此播放之
        data = wf.readframes(chunk)  # 继续读取后面的帧
        dump_buff_file.write(str(data) + "\n---------------------------------------\n")  # 将读出的帧写入文件中，每一个帧用分割线隔开以便阅读

    stream.stop_stream()  # 停止数据流
    stream.close()  # 关闭数据流
    p.terminate()  # 关闭 PyAudio
    print('play函数结束！')


# play()


def TestBytesToInt():
    num1 = int.from_bytes(b'12',byteorder='big')
    num2 = int.from_bytes(b'12',byteorder='little')
    print('%s,'%'num1',num1)
    print('%s,' % 'num2', num2)

    byt1 = (1024).to_bytes(2,byteorder='big')
    byt2 = (1024).to_bytes(10,byteorder='big')
    # byt3 = (-1024).to_bytes(10,byteorder='big')#can't convert negative int to unsigned
    byt3=(3).to_bytes(10,byteorder='big')
    lis1=['byt1','byt2','byt3']
    lis2=[byt1,byt2,byt3]
    lis3=zip(lis1,lis2)
    dic=dict(lis3)
    print(dic)
    print(bytearray(b'\x04\x00')[0])

# TestBytesToInt()