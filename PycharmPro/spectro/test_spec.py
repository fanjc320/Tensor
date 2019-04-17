import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import pyaudio
import test_myspectro_source_test

def GetSpecImg():
    path = "..\..\\res"
    name = "fjc.wav"
    filename = os.path.join(path, name)
    # f = wave.open(filename,'rb')
    f = wave.open(r"./res/fjc.wav", 'rb')
    # f = wave.open(r"./res/erquan_part.wav", 'rb')
    # f = wave.open(r"./res/erquan_part_big.wav", 'rb')
    # 得到语音参数
    params = f.getparams()
    nchannels, sampwidth, framerate,nframes = params[:4]
    print("channels:",nchannels,"sampwidth:",sampwidth,"framerate:",framerate,"nframes:",nframes)
    #---------------------------------------------------------------#
    # 将字符串格式的数据转成int型
    print("reading wav file......")
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData,dtype=np.short)
    # 归一化
    waveData = waveData * 1.0/max(abs(waveData))
    #将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
    waveData = np.reshape(waveData,[nframes,nchannels]).T # .T 表示转置
    f.close()#关闭文件
    #----------------------------------------------------------------#
    '''绘制语音波形'''
    time = np.arange(0,nframes) * (1.0 / framerate)#计算时间
    time= np.reshape(time,[nframes,1]).T
    plt.plot(time[0,:nframes],waveData[0,:nframes],c="b")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("Original wave")
    # plt.show()

    # ----------------------------------------------------------------#
    '''绘制语音语谱'''
    framelength = 0.025 #帧长20~30ms
    framesize = framelength*framerate #每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等\
                                        #而NFFT最好取2的整数次方,即framesize最好取的整数次方
    #找到与当前framesize最接近的2的正整数次方
    nfftdict = {}
    lists = [32,64,128,256,512,1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])#按与当前framesize差值升序排列
    framesize = int(sortlist[0][0])#取最接近当前framesize的那个2的正整数次方值为新的framesize

    NFFT = framesize #NFFT必须与时域的点数framsize相等，即不补零的FFT
    print("nfft:",NFFT)
    overlapSize = 1.0/3 * framesize #重叠部分采样点数overlapize约为每帧点数的1/3~1/2
    overlapSize = int(round(overlapSize))#取整
    print("帧长为{},帧叠为{},傅里叶变换点数为{}".format(framesize,overlapSize,NFFT))
    # spectrum,freqs,ts,fig = plt.specgram(waveData[0],NFFT = NFFT,Fs =framerate,window=np.hanning(M = framesize),
    #         noverlap=overlapSize,mode='magnitude',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱图
    #
    spectrum,freqs,ts,im,Z,extent = test_myspectro_source_test.specgram(waveData[0],NFFT = NFFT,Fs =framerate,window=np.hanning(M = framesize),
            noverlap=overlapSize,mode='magnitude',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱图

    # print("GetWavData:", Z.shape, extent, min(freqs), max(freqs))  # freqs是等差数列，是一系列连续的频率数组
    # Z = Z[-200:-1,:]
    plt.imshow(Z, extent=extent)
    plt.axis('auto') # 有这句，extent才会生效
    plt.show()

    # GetWaveData(Z)
    # return Z

GetSpecImg()

def GetWaveData(Z):
    pass
