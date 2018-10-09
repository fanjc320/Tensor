%
% pr10_5_2 
clear all; clc; close all;

Fs=8000;                                  % ����Ƶ��
Fs2=Fs/2;

fp=60; fs=20;                             % ͨ�����ƺ����Ƶ��
wp=fp/Fs2; ws=fs/Fs2;                     % ת���ɹ�һ��Ƶ��
Rp=1; Rs=40;                              % ͨ�������˥��
[n,Wn]=cheb2ord(wp,ws,Rp,Rs);             % �����˲����״�

[b1,a1]=cheby2(n,Rs,Wn,'high');           % �����˲���ϵ��
fprintf('b=%5.6f   %5.6f   %5.6f   %5.6f   %5.6f\n',b1);
fprintf('a=%5.6f   %5.6f   %5.6f   %5.6f   %5.6f\n',a1);
fprintf('\n');
[db,mag,pha,grd,w]=freqz_m(b1,a1);        % ����˲���Ƶ����Ӧ
a=[1 -0.99];
db1=freqz_m(1,a);                         % �����ͨ�˲���Ƶ����Ӧ  
A=conv(a,a1);                             % ���㴮���˲���ϵ��
B=b1;
db2=freqz_m(B,A);
fprintf('B=%5.6f   %5.6f   %5.6f   %5.6f   %5.6f\n',B);
fprintf('A=%5.6f   %5.6f   %5.6f   %5.6f   %5.6f   %5.6f\n',A);
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)+100]);
subplot 221; plot(w/pi*Fs2,db1,'k'); 
title('��ͨ�˲�����ֵƵ����Ӧ����')
ylim([-50 0]); ylabel('��ֵ/dB'); xlabel(['Ƶ��/Hz' 10 '(a)']);
subplot 222; plot(w/pi*Fs2,db,'k');
title('��ͨ�˲�����ֵƵ����Ӧ����')
axis([0 500 -50 5]); ylabel('��ֵ/dB'); xlabel(['Ƶ��/Hz' 10 '(b)']);
subplot 212; semilogx(w/pi*Fs2,db2,'k');
title('��ͨ�˲�����ֵƵ����Ӧ����'); ylabel('��ֵ/dB'); 
xlabel(['Ƶ��/Hz' 10 '(c)']); axis([10 3500 -40 5]); grid




