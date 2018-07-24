"path/to/vad/audio_stream.py"
#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pyaudio import PyAudio,paInt16
from datetime import datetime
import wave
from Tkinter import *
import sys
from ffnn import FFNNVADGeneral

import logging
# import chardet    # 查看编码

# define of params
NUM_SAMPLES =160
FRAMERATE = 16000
CHANNELS = 1
SAMPWIDTH = 2
FORMAT = paInt16
TIME = 125
FRAMESHIFT = 160

def save_wave_file(filename,data):
    '''save the date to the wav file'''
    wf = wave.open(filename,'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPWIDTH)
    wf.setframerate(FRAMERATE)
    wf.writeframes("".join(data)) # ""中间不能有空格，不然语音录入会有很多中断。
    wf.close()

def my_button(root,label_text,button_text,button_stop,button_func,stop_func):
    '''create label and button'''
    label = Label(root,text=label_text,width=30,height=3).pack()
    button = Button(root,text=button_text,command=button_func,anchor='center',width=30,height=3).pack()
    button = Button(root,text=button_stop,command=stop_func,anchor='center',width=30,height=3).pack()

def record_wave():
    '''open the input of wave'''
    pa = PyAudio()
    # 录音
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=FRAMERATE,
                     input=True,
                     frames_per_buffer=NUM_SAMPLES)   #一个buffer存NUM_SAMPLES个字节,作为一帧

    vad = FFNNVADGeneral('/path/to/VAD/alex-master/alex/tools/vad_train/model_voip/vad_nnt_546_hu32_hl1_hla6_pf10_nf10_acf_1.0_mfr20000_mfl20000_mfps0_ts0_usec00_usedelta0_useacc0_mbo1_bs100.tffnn', 
                filter_length=2, sample_rate=16000, framesize=512, frameshift=160,
                usehamming=True, preemcoef=0.97, numchans=26, ceplifter=22, numceps=12, 
                enormalise=True, zmeansource=True, usepower=True, usec0=False, 
                usecmn=False, usedelta=False, useacc=False, n_last_frames=10, 
                n_prev_frames=10, lofreq=125, hifreq=3800, mel_banks_only=True)
    # 语音激活检测神经网络方法的类FFNNVADGeneral.

    save_buffer = []
    count = 0
    # logging设置，用于记录日志
    logging.basicConfig(level=logging.INFO,
                        filename='log.txt',
                        filemode ='w',
                        format='%(message)s')
                        
    while count < TIME*4:
        string_audio_data = stream.read(NUM_SAMPLES)    
        result = vad.decide(string_audio_data)
        frame = count*NUM_SAMPLES/float(FRAMESHIFT)
    time = count*NUM_SAMPLES/float(FRAMERATE)  # time=frame*frameshift/framerate
    logging.info('frame: '+str(frame)+' time: '+str(time)+' prob: '+str(result))  # logging记录字符串，用‘+’连接
        save_buffer.append(string_audio_data)
        count += 1
        #chardet.detect(string_audio_data)  #查看编码类型
        print "."
   
    filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+".wav"
    save_wave_file(filename,save_buffer)
      
    save_buffer = []
    print "filename,saved."

def record_stop():
    # stop record the wave
    sys.exit(0)

def main():
    root = Tk()
    root.geometry('300x200+200+200')
    root.title('record wave')
    my_button(root,"Record a wave","clik to record","stop recording",record_wave,record_stop)
    root.mainloop()
    
if __name__ == "__main__":
    main()



# error
$ bt_audio_service_open: connect() failed: Connection refused (111)
# 解决: 貌似有多余蓝牙库却没有蓝牙设备
$ sudo apt-get purge bluez-alsa

# Warning
$ ALSA lib pcm.c:2239:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
ALSA lib pcm.c:2239:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.center_lfe
ALSA lib pcm.c:2239:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.side
Cannot connect to server socket err = No such file or directory
Cannot connect to server request channel
jack server is not running or cannot be started
# 是由于usr/share/alsa/alsa.conf默认设置