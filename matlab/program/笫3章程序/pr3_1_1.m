%
% pr3_1_1
clear all; clc; close all;
y=load('su1.txt');                            % ��������
fs=16000; nfft=1024;                          % ����Ƶ�ʺ�FFT�ĳ���
time=(0:nfft-1)/fs;                           % ʱ��̶�
figure(1), subplot 211; plot(time,y,'k');     % �����źŲ���
title('�źŲ���'); axis([0 max(time) -0.7 0.7]);
ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(a)']); grid;
figure(2)
nn=1:nfft/2; ff=(nn-1)*fs/nfft;               % ����Ƶ�ʿ̶�
Y=log(abs(fft(y)));                           % ��ʽ(3-1-8)ȡʵ������
subplot 211; plot(ff,Y(nn),'k'); hold on;     % �����źŵ�Ƶ��ͼ
z=ifft(Y);                                    % ��ʽ(3-1-8)��ȡ����
figure(1), subplot 212; plot(time,z,'k');     % ��������ͼ
title('�źŵ���ͼ'); axis([0 time(512) -0.2 0.2]); grid; 
ylabel('��ֵ'); xlabel(['��Ƶ��/s' 10 '(b)']);
mcep=29;                                      % �������ż�������������弤��Ӧ
zy=z(1:mcep+1);
zy=[zy' zeros(1,1000-2*mcep-1) zy(end:-1:2)']; % ���������弤��Ӧ�ĵ�������
ZY=fft(zy);                                   % ���������弤��Ӧ��Ƶ��
figure(2),                                    % ���������弤��Ӧ��Ƶ�ף��û��߱�ʾ
line(ff,real(ZY(nn)),'color',[.6 .6 .6],'linewidth',3);
grid; hold off; ylim([-4 5]);
title('�ź�Ƶ�ף����ߣ��������弤��Ƶ�ף����ߣ�')
ylabel('��ֵ'); xlabel(['Ƶ��/Hz' 10 '(a)']); 

ft=[zeros(1,mcep+1) z(mcep+2:end-mcep)' zeros(1,mcep)]; % �������ż�������ĵ�������
FT=fft(ft);                                  % �������ż��������Ƶ��
subplot 212; plot(ff,real(FT(nn)),'k'); grid;% �������ż��������Ƶ��
title('���ż�������Ƶ��')
ylabel('��ֵ'); xlabel(['Ƶ��/Hz' 10 '(b)']); 

