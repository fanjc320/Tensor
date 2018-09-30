%
% pr4_4_2 
clear all; clc; close all;

[x1,fs]=wavread('s1.wav');      % �����ź�s1
x2=wavread('s2.wav');           % �����ź�s2
x3=wavread('a1.wav');           % �����ź�a1
wlen=200;                       % ֡��
inc=80;                         % ֡��
x1=x1/max(abs(x1));             % ��ֵ��һ��
x2=x2/max(abs(x2));
x3=x3/max(abs(x3));
p=12;                           % LPC����

[DIST12,y1lpcc,y2lpcc]=lpcc_dist(x1,x2,wlen,inc,p);% ����x1��x2��LPCC����
[DIST13,y1lpcc,y3lpcc]=lpcc_dist(x1,x3,wlen,inc,p);% ����x1��x3��LPCC����
% ��ͼ
figure(1)
plot(y1lpcc(3,:),y2lpcc(3,:),'k+'); hold on
plot(y1lpcc(7,:),y2lpcc(7,:),'kx'); 
plot(y1lpcc(12,:),y2lpcc(12,:),'k^');
plot(y1lpcc(16,:),y2lpcc(16,:),'kh'); 
legend('��3֡','��7֡','��12֡','��16֡',2)
title('/i1/��/i2/֮���LPCC����ƥ��Ƚ�')
xlabel('�ź�x1');ylabel('�ź�x2')
axis([-6 6 -6 6]);
line([-6 6],[-6 6],'color','k','linestyle','--');

figure(2)
plot(y1lpcc(3,:),y3lpcc(3,:),'k+'); hold on
plot(y1lpcc(7,:),y3lpcc(7,:),'kx'); 
plot(y1lpcc(12,:),y3lpcc(12,:),'k^');
plot(y1lpcc(16,:),y3lpcc(16,:),'kh'); 
legend('��3֡','��7֡','��12֡','��16֡',2)
title('/i1/��/a1/֮���LPCC����ƥ��Ƚ�')
xlabel('�ź�x1');ylabel('�ź�x3')
axis([-6 6 -6 6]);
line([-6 6],[-6 6],'color','k','linestyle','--');

