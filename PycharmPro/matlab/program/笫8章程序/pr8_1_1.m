%
% pr8_1_1 
clear all; clc; close all;

fs=8000; fs2=fs/2;                      % ����Ƶ��
Wp=[60 500]/fs2;                        % �˲���ͨ��
Ws=[20 2000]/fs2;                       % �˲������
Rp=1; Rs=40;                            % ͨ���Ĳ��ƺ������˥��
[n,Wn]=ellipord(Wp,Ws,Rp,Rs);           % �����˲����Ľ���
[b,a]=ellip(n,Rp,Rs,Wn);                % �����˲�����ϵ��
fprintf('b=%5.6f   %5.6f   %5.6f   %5.6f   %5.6f   %5.6f   %5.6f\n',b)
fprintf('a=%5.6f   %5.6f   %5.6f   %5.6f   %5.6f   %5.6f   %5.6f\n',a)

[db, mag, pha, grd,w]=freqz_m(b,a);     % ��ȡƵ����Ӧ����
plot(w/pi*fs/2,db,'k');                 % ��ͼ
grid; ylim([-90 10]);
xlabel('Ƶ��/Hz'); ylabel('��ֵ/dB');
title('��Բ6�״�ͨ�˲���Ƶ����Ӧ����');

