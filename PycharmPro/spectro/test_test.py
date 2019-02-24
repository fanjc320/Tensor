import numpy as np, wave,math
import cv2
import matplotlib.cbook as cbook
from matplotlib import docstring
from matplotlib.path import Path
import matplotlib.pyplot as plt 
import math
import test_myspectro_source_test
# filename 是文件名
# window_length_ms 是以毫秒为单位的窗长
# window_shift_times 是帧移，是与窗长的比例 例如窗长20ms，帧移0.5就是10毫秒
def getSpectrum(filename, window_length_ms, window_shift_times):
    # 读音频文件
    wav_file = wave.open(filename, 'r')
    # 获取音频文件的各种参数
    params = wav_file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    # 获取音频文件内的数据，不知道为啥获取到的竟然是个字符串，还需要在numpy中转换成short类型的数据
    str_data = wav_file.readframes(wav_length)
    wave_data = numpy.fromstring(str_data, dtype=numpy.short)
    # 将窗长从毫秒转换为点数
    window_length = framerate * window_length_ms / 1000
    window_shift = int(window_length * window_shift_times)
    # 计算总帧数，并创建一个空矩阵
    nframe = (wav_length - (window_length - window_shift)) / window_shift
    nframe = math.floor(nframe)
    window_length = math.floor(window_length)
    a = math.floor(window_length/2)
    # spec = numpy.zeros((window_length/2, nframe))
    spec = numpy.zeros((a, nframe))

    print("max wavedata:",max(wave_data))
    maxlog = 0;
    # 循环计算每一个窗内的fft值
    for i in range(nframe):
        start = i * window_shift
        end = start + window_length
        # [:window_length/2]是指只留下前一半的fft分量
        # spec[:, i] = numpy.log(numpy.abs(numpy.fft.fft(wave_data[start:end])))[:window_length/2]
        r_fft = numpy.abs(numpy.fft.fft(wave_data[start:end]))
        spec[:, i] = 10*numpy.log10(r_fft)[:window_length // 2]
        maxlog = max(maxlog,max(spec[:,i]))

    for i in range(20000):
        print(10*numpy.log10(i))
    print("maxlog:",maxlog)
    print("spec shape:",numpy.shape(spec))
    return spec


 
# 窗长20ms， 窗移时窗长的0.5倍
#speech_spectrum = getSpectrum('./yusheng_ni.wav', 20, 0.5)
#print(numpy.shape(speech_spectrum))
#plt.imshow(speech_spectrum)

x = [0, 20, 25, 31.5, 40, 50, 63, 80, 100, 125,
 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 
 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 
 16000, 20000];
y = [140, 109.51, 104.23, 99.08, 94.18, 89.96, 85.94, 82.05, 78.65, 75.56, 
72.47, 69.86, 67.53, 65.39, 63.45, 62.05, 60.81, 59.89, 60.01, 62.15, 
63.19, 59.96, 57.26, 56.42, 57.57, 60.89, 66.36, 71.66, 73.16, 68.63,
 68.43, 104.92];


    
# def rolling_window(a, window):
    # shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    ##shape:(4,3)  a.shape:(6,)   a.shape[:-1]:()   a.shape[-1]:6
    # print("shape",shape,a.shape,a.shape[:-1],a.shape[-1])
    # strides = a.strides + (a.strides[-1],)
    # print("strides:",a.strides,strides)
    # return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
# a = np.array([1, 12, 3, 4, 5, 6]); 
# b = rolling_window(a, 3)
# print("rolling_window:",b)
# c = np.lib.stride_tricks.as_strided(a, shape=(4,3), strides=(4,4))
# print("strides:",c)


# 数组切片
lst = [
    [-10,-20,-30,-40,-50,-60]
    # [20,30,40,50,60,70],
    # [30,40,50,60,70,80]
    ]

# lst = lst[1:-1]
arr = np.asarray(lst)


fig = plt.figure()
ax = fig.add_subplot(311)
plt.imshow(arr,cmap = 'Greys',vmin=-200,vmax=0)

fig.add_subplot(312)
arr = arr*5
plt.imshow(arr,cmap= 'Greys')




fig.add_subplot(313)
arr0 = [[1,2,3,4,5,6]]
newarr = np.r_[arr0,arr]
print("newarr",newarr)
plt.imshow(newarr,cmap= 'Greys')

plt.show()

# print(arr[0:2, 0:4])
# print(arr[0:2])
# print("arr:",arr)
# b = arr[arr>40]=0
# print("arr>40:",arr)


# lst 没shape,array有shape,

b = np.array([(1,2,3),(2,4,6),(0.5,3,5)],dtype=float)
c = test_myspectro_source_test.maxminnorm(b)
print(c)