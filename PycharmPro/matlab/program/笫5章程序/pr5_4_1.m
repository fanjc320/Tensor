%
% pr5_4_1 
clear all; clc; close all;

[x,fs,nbit]=wavread('bluesky31.wav');     % ����bluesky31.wav�ļ�
t=(0:length(x)-1)/fs;                     % ����ʱ��
y=detrend(x);                             % ��������������
y=y/max(abs(y));                          % ��ֵ��һ��
subplot 211; plot(t,x,'k');               % ��������������������ź�x
title('��������������ź�');
xlabel('ʱ��/s'); ylabel('��ֵ');
subplot 212; plot(t,y,'k');               % ��������������������ź�y
xlabel('ʱ��/s'); ylabel('��ֵ');
title('����������������ź�');


