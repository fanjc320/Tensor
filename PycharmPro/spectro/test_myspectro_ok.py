import numpy, wave,math
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



import numpy, matplotlib.pyplot as plt
# 窗长20ms， 窗移时窗长的0.5倍
speech_spectrum = getSpectrum('./yusheng_ni.wav', 20, 0.5)
print(numpy.shape(speech_spectrum))
plt.imshow(speech_spectrum)


plt.show()