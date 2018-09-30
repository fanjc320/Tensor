%
% pr2_4_1
clear all; clc; close all;

filedir=[];                         % ����·��
filename='bluesky3.wav';            % �����ļ���
fle=[filedir filename];             % ����������·�����ļ���
[x,Fs]=wavread(fle);                % ���������ļ�
wlen=200; inc=80; win=hanning(wlen);% ����֡����֡�ƺʹ�����
N=length(x); time=(0:N-1)/Fs;       % ����ʱ��
y=enframe(x,win,inc)';              % ��֡
fn=size(y,2);                       % ֡��
frameTime=(((1:fn)-1)*inc+wlen/2)/Fs; % ����ÿ֡��Ӧ��ʱ��
W2=wlen/2+1; n2=1:W2;
freq=(n2-1)*Fs/wlen;                % ����FFT���Ƶ�ʿ̶�
Y=fft(y);                           % ��ʱ����Ҷ�任
clf                                 % ��ʼ��ͼ��
%=====================================================%
% Plot the STFT result              % ��������ͼ        
%=====================================================%
set(gcf,'Position',[20 100 600 500]);            
axes('Position',[0.1 0.1 0.85 0.5]);  
imagesc(frameTime,freq,abs(Y(n2,:))); % ����Y��ͼ��  
axis xy; ylabel('Ƶ��/Hz');xlabel('ʱ��/s');
title('����ͼ');
m = 64;
LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0];
Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors));

%=====================================================%
% Plot the Speech Waveform          % ���������źŵĲ���  
%=====================================================%
axes('Position',[0.07 0.72 0.9 0.22]);
plot(time,x,'k');
xlim([0 max(time)]);
xlabel('ʱ��/s'); ylabel('��ֵ');
title('�����źŲ���');
