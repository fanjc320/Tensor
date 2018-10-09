%
% pr5_5_1 
clear all; clc; close all;

fp=500; fs=750;                         % �����˲�����ͨ�������Ƶ��
Fs=8000; Fs2=Fs/2;                      % ����Ƶ��
Wp=fp/Fs2; Ws=fs/Fs2;                   % ��ͨ�������Ƶ�ʹ�һ��
Rp=3; Rs=50;                            % ͨ�����ƺ����˥��
[n,Wn]=cheb2ord(Wp,Ws,Rp,Rs);           % ��ȡ�˲�������
[b,a]=cheby2(n,Rs,Wn);                  % �������ѩ��II�͵�ͨ�˲���ϵ��
[db,mag,pha,grd,w]=freqz_m(b,a);        % ���˲�����Ƶ����Ӧ����

filedir=[];                             % ָ���ļ�·��
filename='bluesky3.wav';                % ָ���ļ���
fle=[filedir filename]                  % ����·�����ļ������ַ���
[s,fs]=wavread(fle);                    % ���������ļ�
s=s-mean(s);                            % ����ֱ������
s=s/max(abs(s));                        % ��ֵ��һ��
N=length(s);                            % ����źų���
t=(0:N-1)/fs;                           % ����ʱ��

y=filter(b,a,s);                        % �������ź�ͨ���˲���
wlen=200; inc=80; nfft=512;             % ����֡��,֡�ƺ�nfft��
win=hann(wlen);                         % ���ô�����
d=stftms(s,win,nfft,inc);               % ԭʼ�źŵ�STFT�任
fn=size(d,2);                           % ��ȡ֡��
frameTime=(((1:fn)-1)*inc+nfft/2)/Fs;   % ����ÿ֡��Ӧ��ʱ��--ʱ����̶�
W2=1+nfft/2;                            % ����Ƶ����̶�
n2=1:W2;
freq=(n2-1)*Fs/nfft;
d1=stftms(y,win,nfft,inc);              % �˲����źŵ�STFT�任
% ��ͼ
figure(1)                                  
plot(w/pi*Fs2,db,'k','linewidth',2)
grid; axis([0 4000 -100 5]);
title('��ͨ�˲����ķ�ֵ��Ӧ����')
xlabel('Ƶ��/Hz'); ylabel('��ֵ/dB');
figure(2)                                  
subplot 211; plot(t,s,'k'); 
title('�������źţ����������죬���ơ�')
xlabel(['ʱ��/s' 10 '(a)']); ylabel('��ֵ')
subplot 212; imagesc(frameTime,freq,abs(d(n2,:))); axis xy
title('�������źŵ�����ͼ')
xlabel(['ʱ��/s' 10 '(b)']); ylabel('Ƶ��/Hz')
m = 256;
LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0];
Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors));
figure(3)                                 
subplot 211; plot(t,y,'k'); 
title('�˲���������ź�')
xlabel(['ʱ��/s' 10 '(a)']); ylabel('��ֵ')
subplot 212; imagesc(frameTime,freq,abs(d1(n2,:))); axis xy
title('�˲��������źŵ�����ͼ')
xlabel(['ʱ��/s' 10 '(b)']); ylabel('Ƶ��/Hz')
m = 256;
LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0];
Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors)); ylim([0 1000]);
