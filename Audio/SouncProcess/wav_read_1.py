#wave data get  -xlxw

#import
import wave as we
import numpy as np
import matplotlib.pyplot as plt

def wavread(path):
    wavfile =  we.open(path,"rb")
    params = wavfile.getparams()
    print("params:",params)
    framesra,frameswav= params[2],params[3]
    #readframes()
    #得到每一帧的声音数据，返回的值是二进制数据，在python中用字符串表示二进制数据datawav = wavfile.readframes(frameswav)
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav,dtype = np.short)
    datause.shape = -1,2
    datause = datause.T

    time = np.arange(0, frameswav) * (1.0/framesra)
    print("time:",time.shape)
    return datause,time

def main():

    file = open("night.wav", "rb")
    s = file.read(4)
    print(s)
    path = input("The Path is:")
    wavdata,wavtime = wavread(path)
    plt.title("Night.wav's Frames")
    plt.subplot(211)
    plt.plot(wavtime, wavdata[0],color = 'green')
    plt.subplot(212)
    plt.plot(wavtime, wavdata[1])
    plt.show()
    
main()