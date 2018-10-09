%
% pra_1_2 
clear all; clc; close all;

[xx,fs,nbit]=wavread('digits1_10.wav');
N=length(xx);
time=(0:N-1)/fs;                          % ����ʱ��̶�
x1=xx/max(abs(xx));                       % ��ֵ��һ��
wlen=320;                                 % ֡��
inc=80;                                   % ֡��
yy=enframe(x1,wlen,inc)';                 % ��֡
fn=size(yy,2);                            % ֡��
frameTime=frame2time(fn,wlen,inc,fs);     % ÿ֡��Ӧ��ʱ��
Ef=Ener_entropy(yy,fn);                   % �������ر�ֵ
figure(1)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-100)]);
subplot 211; plot(time,x1,'k'); 
axis([0 max(time) -1 1]);
title('����������������ź�'); xlabel('(a)'); ylabel('��ֵ');
subplot 212; plot(frameTime,Ef,'k'); grid;
title('���ر�ͼ'); xlabel(['ʱ��/s' 10 '(b)']); ylabel('��ֵ');
axis([0 max(time) 0 1]);

xx=xx/max(abs(xx));                       % ��ֵ��һ��
[x,xtrend]=polydetrend(xx, fs, 4);        % ����������
y=enframe(x,wlen,inc)';                   % ��֡
Ef=Ener_entropy(y,fn);                    % �������ر�ֵ
figure(2)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-100)]);
subplot 211; plot(time,x,'k');
line([0 max(time)],[0 0],'color','k');
title('����������������ź�'); xlabel('(a)'); ylabel('��ֵ');
axis([0 max(time) -1 1]);
subplot 212; plot(frameTime,Ef,'k'); grid;
title('���ر�ͼ'); xlabel(['ʱ��/s' 10 '(b)']); ylabel('��ֵ');
axis([0 max(time) 0 1]);
figure(1); subplot 211;
line([time],[xtrend],'color',[.6 .6 .6],'linewidth',2);

