%
% pra_1_1 
clear all; clc; close all;

filedir=[];                               % ���������ļ���·��
filename='colorcloud.wav';                % ���������ļ�������
fle=[filedir filename]                    % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                     % ��ȡ�ļ�
xx=xx/max(abs(xx));                       % ��ֵ��һ��
N=length(xx);                             % �źų���
time = (0 : N-1)/fs;                      % ����ʱ��̶�
wlen=320;                                 % ֡��
inc=80;                                   % ֡��
yy=enframe(xx,wlen,inc)';                 % ����ֱ������ǰ��֡
fn=size(yy,2);                            % ֡��
frameTime=frame2time(fn,wlen,inc,fs);     % ÿ֡��Ӧ��ʱ��
Ef=Ener_entropy(yy,fn);                   % �������ر�ֵ
figure(1)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-100)]);
subplot 211; plot(time,xx,'k');
line([0 max(time)],[0 0],'color','k','linestyle','-.');
title('����ֱ������ǰ���źŲ���ͼ'); xlabel('ʱ��/s'); ylabel('��ֵ');
subplot 212; plot(frameTime,Ef,'k'); grid;
title('���ر�ͼ'); xlabel('ʱ��/s'); ylabel('��ֵ');

xx=xx-mean(xx);                           % ����ֱ������
x=xx/max(abs(xx));                        % ��ֵ��һ��
y=enframe(x,wlen,inc)';                   % ����ֱ���������֡
Ef=Ener_entropy(y,fn);                    % �������ر�ֵ
figure(2)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-100)]);
subplot 211; plot(time,x,'k');
line([0 max(time)],[0 0],'color','k','linestyle','-.');
title('����ֱ����������źŲ���ͼ'); xlabel('ʱ��/s'); ylabel('��ֵ'); 
subplot 212; plot(frameTime,Ef,'k'); grid;
title('���ر�ͼ'); xlabel('ʱ��/s'); ylabel('��ֵ');

