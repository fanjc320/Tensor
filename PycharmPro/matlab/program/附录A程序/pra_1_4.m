%
% pra_1_4    
clear all; clc; close all;

[xx,fs]=wavread('hello28n.wav');        % ���������ļ�
xx=xx-mean(xx);                         % ����ֱ������
signal=xx/max(abs(xx));                 % ��ֵ��һ��

IS=1;                                   % ����ǰ���޻��γ���
wlen=200;                               % ����֡��Ϊ25ms
inc=80;                                 % ����֡��Ϊ10ms
N=length(xx);                           % ֡��
time=(0:N-1)/fs;                        % ����ʱ��
NIS=fix((IS*fs-wlen)/inc +1);           % ��ǰ���޻���֡��

a=4; b=0.001;                           % ���ò���a��b
output=simplesubspec(signal,wlen,inc,NIS,a,b);% �׼�
% ��ͼ
subplot 211; plot(time,signal,'k'); grid; axis tight;
title(['���������ź�']); ylabel('��ֵ')
subplot 212; plot(time,output,'k');grid;  axis tight;
title('�׼����źŲ���'); ylabel('��ֵ'); xlabel('ʱ��/s');



        
