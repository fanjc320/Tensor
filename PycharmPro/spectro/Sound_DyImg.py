import wave
#import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import pyaudio
import wave as we
from scipy.fftpack import fft,ifft


def fjc_record(OutFile="../res/fjc.wav", Seconds=5):
    # 参数定义
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
        # print(type(data))
        frames.append(data)
        intdata = int.from_bytes(data,byteorder='big',signed=False);
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