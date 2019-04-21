# coding=utf8
import wave
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import pyaudio
import wave as we
from scipy.fftpack import fft,ifft

def Map(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2;

def fjc_record(OutFile="../res/fjc.wav", Seconds=1):
    # 参数定义
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 # 通过play函数测试得出结论 channels=1 占用2个字节，channels=2占用4个字节
    RATE = 44100

    wavfile = wave.open(r"../res/yusheng1.wav", "rb")
    params = wavfile.getparams()
    print("params:", params)
    framesra, nframes = params[2], params[3]
    print("wavedata:",framesra,nframes)
    spec_arr = np.zeros((1,1024))
    spec_test = np.zeros(201)
    print("spec_arr shpe:",spec_arr.shape)
    for i in range(nframes):
        if i>0:
            break
        datawav = wavfile.readframes(1024)
        data = np.fromstring(datawav, dtype=np.short)#转成np.array
        time = np.arange(0, nframes) * (1.0 / framesra)
        # plt.plot(data)
        spec = fft(data)
        np.set_printoptions(threshold=np.nan)
        print("data:",data.shape,"spec:",type(spec),spec.shape)
        # spec_arr[i]=spec # error
        # spec_arr = np.concatenate(([spec_arr],[spec]),axis=0)
        spec_arr =np.insert(spec_arr,0,[spec],0)
        print("spec_arr spec:",spec_arr.shape)
        spec_v = np.mean(abs(spec))
        print("spec_arr_min:",spec_v)
        # plt.plot(i,np.mean(abs(spec)))
        spec_test[i]=spec_v
        spec_t = spec.transpose()
        print("spec.shape:",spec.shape,spec_t)
        plt.plot(spec_t)

    # plt.plot(range(5),[2,7,4,5,6])
    # plt.plot(range(201),spec_test)
    # plt.plot(range(202),spec_arr)
    # plt.plot(spec_arr,'o')
    plt.show()
    # print("spec_arr:",spec_arr)
'''
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    for i in range(0, int(RATE / CHUNK * Seconds)):
        data = stream.read(CHUNK)
        intdata = np.fromstring(data, dtype=np.short)
        # if i==0:
        #     print(type(data),data)
            # print(data.split(b'|'))
            # for idx in range(0,len(data),2):
            #     chunk_one = data[idx:idx+2]
            #     print("i:",idx,chunk_one,int.from_bytes(chunk_one,byteorder='big'))
        
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
'''

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

def TestAppend():
    a =np.array([[0, 1, 2, 3],
           [4, 5, 6, 7],
           [8, 9, 10, 11]])



    # 内部调用concatenate
    # all the input arrays must have same number of dimensions
    # np.append(a, [1, 1, 1, 1], axis=0)
    b= np.append(a, [[1, 1, 1, 1]], axis=0) # ok
    print("b:", b)
    tmp = np.array([1,2,3,4])
    # b1=np.append(a,tmp,axis=0)# (4,) error
    b1 = np.append(a, [tmp], axis=0)
    print(tmp.shape,"b1:",b1)

    #不好创建空数组,然后添加元素，添加的元素并没有纬度
    a0=np.array([])
    print("a0:",a0)
    a1 = np.append(a0, [[1, 1, 2]])
    print("a0**",a0.shape,a1,np.append(a1,[[1,1,1]]))

    c=np.insert(a,0,[1,1,1,1],0)
    print("c--:",c)
    d=np.delete(a,1,0)
    print("d==:",d)


    spec_arr = np.array((4,3))
    test_arr= np.zeros((4,3))
    # test_arr = np.array(4,3)#data type not understood
    print("spec_arr shape:",spec_arr.shape,spec_arr,test_arr.shape)
    np.insert(spec_arr,0,[1,1,1,1],0);
    test_arr=np.insert(test_arr,0,[1,1,1],0);
    print("spec_arr:",spec_arr)
    print("test_arr:",test_arr)
    x = np.zeros((0,4))
    # y= np.array([[1,1,1,2]])
    z=np.concatenate((x,[[1,1,1,1]]),0)
    print(x.shape);print(z.shape)

    i = np.ones(3);j=np.ones(4);
    print(np.c_[a,i]);
    # print(np.r_[a,i]); #error

# TestAppend()


def TestPlot2DArray():
    arr = np.zeros((0,6));

    for i in range(4):
        item=[]
        for j in range(6):
           item.append(j+i)
        print("item:",item)
        # arr=np.insert(arr,0,[j]*6,0)
        arr=np.insert(arr,0,item,0)

    print("arr:",arr)
    plt.plot(arr,'o')
    # plt.plot(range(5),arr)# 同上
    plt.show()

    farr0=fft(arr)
    print("farr0:",farr0)

    farr=np.zeros((0,6))
    print("farr original:",farr)
    for a in arr:
        fa = fft(a)
        print("fa:",fa)
        farr=np.insert(farr,0,fa,0)
    print("farr:",farr)

    print("farr.shape:",farr.shape)
    farr_t= farr.transpose()
    print("farr_t:",farr_t)
    plt.plot(farr,'o')
    plt.show()

# TestPlot2DArray()
