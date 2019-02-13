# python语音信号处理-加窗、分帧、STFT
# 其实这个可以用STFT可以用librosa来进行处理


from __future__ import division
from scikits.talkbox import segment_axis
import numpy as np
import soundfile as sf


def readwav(fn):
    signal, sampleRate = sf.read(fn)
    signal -= np.mean(signal)
    signal /= np.max(np.abs(signal))  # Normalize the amplitude
    nframes = len(signal)
    return signal, sampleRate, nframes


# smoother version of hanning window
def sqrt_hann(M):
    return np.sqrt(np.hanning(M))


# signal: 1D array, returns a 2D complex array
def pro_signal(signal, window='hanning', frame_len=1024, overlap=512):
    if window == 'hanning':
        # w = np.hanning(frame_len)
        w = sqrt_hann(frame_len)
    else:
        w = window
    y = segment_axis(signal, frame_len, overlap=overlap, end='cut')  # use cut instead of pad
    y = w * y
    return y


# signal is a 2d matrix, n_frames * frame_len
# returns a 2d matrix, n_frames * frame_len /2 + 1
def stft(signal):
    out = np.array([np.fft.rfft(signal[i]) for i in xrange(signal.shape[0])])
    return out


# Load the signal
fn = 'lullaby.wav'
signal, sampleRate, nframes = readwav(fn)
print(sampleRate, nframes)
snd = pro_signal(signal, frame_len=1024, overlap=512)
out = stft(snd)
print(out.shape)