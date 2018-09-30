%
% pr3_3_2 
clear all; clc; close all;

[x1,fs]=wavread('s1.wav');      % �����ź�s1-\i1\
x2=wavread('s2.wav');           % �����ź�s2-\i2\
x3=wavread('a1.wav');           % �����ź�a1-\a1\
wlen=200;                       % ֡��
inc=80;                         % ֡��
x1=x1/max(abs(x1));             % ��ֵ��һ��
x2=x2/max(abs(x2));
x3=x3/max(abs(x3));
% ����/i1/��/i2/֮���ƥ��Ƚ�
[Dcep,Ccep1,Ccep2]=mel_dist(x1,x2,fs,16,wlen,inc);
figure(1)
plot(Ccep1(3,:),Ccep2(3,:),'k+'); hold on
plot(Ccep1(7,:),Ccep2(7,:),'kx'); 
plot(Ccep1(12,:),Ccep2(12,:),'k^');
plot(Ccep1(16,:),Ccep2(16,:),'kh'); 
legend('��3֡','��7֡','��12֡','��16֡',2)
xlabel('�ź�x1');ylabel('�ź�x2')
axis([-12 12 -12 12]);
line([-12 12],[-12 12],'color','k','linestyle','--');
title('/i1/��/i2/֮���MFCC����ƥ��Ƚ�')

% ����/i1/��/a1/֮���ƥ��Ƚ�
[Dcep,Ccep1,Ccep2]=mel_dist(x1,x3,fs,16,wlen,inc);
figure(2)
plot(Ccep1(3,:),Ccep2(3,:),'k+'); hold on
plot(Ccep1(7,:),Ccep2(7,:),'kx'); 
plot(Ccep1(12,:),Ccep2(12,:),'k^');
plot(Ccep1(16,:),Ccep2(16,:),'kh'); 
legend('��3֡','��7֡','��12֡','��16֡',2)
xlabel('�ź�x1');ylabel('�ź�x3')
axis([-12 12 -12 12]);
line([-12 12],[-12 12],'color','k','linestyle','--');
title('/i1/��/a1/֮���MFCC����ƥ��Ƚ�')
