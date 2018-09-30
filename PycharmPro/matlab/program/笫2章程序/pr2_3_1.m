%
% pr2_3_1
clear all; clc; close all;

filedir=[];                % ����·��
filename='bluesky3.wav';   % �����ļ���
fle=[filedir filename];    % ����������·�����ļ���
[x,Fs]=wavread(fle);       % ���������ļ�

wlen=200; inc=80;          % ����֡����֡��
win=hanning(wlen);         % ����������
N=length(x);               % �źų���
X=enframe(x,win,inc)';     % ��֡
fn=size(X,2);              % ���֡��
time=(0:N-1)/Fs;           % ������źŵ�ʱ��̶�
for i=1 : fn
    u=X(:,i);              % ȡ��һ֡
    u2=u.*u;               % �������
    En(i)=sum(u2);         % ��һ֡�ۼ����
end
subplot 211; plot(time,x,'k'); % ����ʱ�䲨�� 
title('��������');
ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(a)']);
frameTime=frame2time(fn,wlen,inc,Fs);   % ���ÿ֡��Ӧ��ʱ��
subplot 212; plot(frameTime,En,'k')     % ������ʱ����ͼ
title('��ʱ����');
 ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(b)']);
