%
% pra_1_3 
clear all; clc; close all;

filedir=[];                               % ���������ļ���·��
filename='deepstep.wav';                  % ���������ļ�������
fle=[filedir filename]                    % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                     % ��ȡ�ļ�
x=xx/max(abs(xx));                        % ��ֵ��һ��
N=length(x);                              % �źų���
time = (0 : N-1)/fs;                      % ����ʱ��̶�
wlen=320;                                 % ֡��
inc=80;                                   % ֡��
nfft=512;                                 % ÿ֡FFT�ĳ���
plot_spectrogram(x,wlen,inc,nfft,fs);     % ��������ͼ
title('����ͼ'); xlabel('ʱ��/s'); ylabel('Ƶ��/Hz');

y=enframe(x,wlen,inc)';                   % ��֡
fn=size(y,2);                             % ��֡��
frameTime=frame2time(fn,wlen,inc,fs);     % ÿ֡��Ӧ��ʱ��
Ef=Ener_entropy(y,fn);                    % �������ر�ֵ
figure(2)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
plot(frameTime,Ef,'k'); grid;
title('���ر�ͼ'); xlabel('ʱ��/s'); ylabel('��ֵ');
