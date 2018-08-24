import wave
import pyaudio
import numpy
import pylab
 
#��WAV�ĵ����ļ�·��������Ҫ���޸�
wf = wave.open("E:\\Tensor\\PyAudio\\res\\one.wav", "rb")
#����PyAudio����
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
channels=wf.getnchannels(),
rate=wf.getframerate(),
output=True)
nframes = wf.getnframes()
framerate = wf.getframerate()
#��ȡ������֡���ݵ�str_data�У�����һ��string���͵�����
str_data = wf.readframes(nframes)
wf.close()
#����������ת��Ϊ����
# A new 1-D array initialized from raw binary or text data in a string.
wave_data = numpy.fromstring(str_data, dtype=numpy.short)
#��wave_data�����Ϊ2�У������Զ�ƥ�䡣���޸�shape������ʱ����ʹ��������ܳ��Ȳ��䡣
wave_data.shape = -1,2
#������ת��
wave_data = wave_data.T
#time Ҳ��һ�����飬��wave_data[0]��wave_data[1]����γ�ϵ�е�����
#time = numpy.arange(0,nframes)*(1.0/framerate)
#���Ʋ���ͼ
#pylab.plot(time, wave_data[0])
#pylab.subplot(212)
#pylab.plot(time, wave_data[1], c="g")
#pylab.xlabel("time (seconds)")
#pylab.show()
#
# �����������޸Ĳ�����������ʼλ�ý��в�ͬλ�úͳ��ȵ���Ƶ���η���
N=44100
start=0 #��ʼ����λ��
df = framerate/(N-1) # �ֱ���
freq = [df*n for n in range(0,N)] #N��Ԫ��
wave_data2=wave_data[0][start:start+N]
c=numpy.fft.fft(wave_data2)*2/N
#������ʾ����Ƶ��һ���Ƶ��
d=int(len(c)/2)
#����ʾƵ����4000���µ�Ƶ��
while freq[d]>4000:
d-=10
pylab.plot(freq[:d-1],abs(c[:d-1]),'r')
pylab.show()
