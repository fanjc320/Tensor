%
% pr5_4_2  
clear all; clc; close all;

[x,fs,nbit]=wavread('bluesky32.wav');     % ����bluesky32.wav�ļ�

[y,xtrend]=polydetrend(x, fs, 2);         % ����polydetrend����������
t=(0:length(x)-1)/fs;                     % ����ʱ��
subplot 211; plot(t,x,'k');               % ��������������������ź�x
line(t,xtrend,'color',[.6 .6 .6],'linewidth',3); % ��������������
ylim([-1.5 1]);
title('��������������ź�');
legend('��������������ź�','�������ź�',4)
xlabel('ʱ��/s'); ylabel('��ֵ');
subplot 212; plot(t,y,'k');               % ��������������������ź�y
xlabel('ʱ��/s'); ylabel('��ֵ');
title('����������������ź�');


