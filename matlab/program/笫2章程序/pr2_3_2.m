%
% pr2_3_2
clear all; clc; close all;

filedir=[];                       % ����·��
filename='bluesky3.wav';          % �����ļ���
fle=[filedir filename];           % ����������·�����ļ���
[xx,Fs]=wavread(fle);             % ���������ļ�
x=xx-mean(xx);                    % ����ֱ������
wlen=200; inc=80;                 % ����֡����֡��
win=hanning(wlen);                % ������
N=length(x);                      % �����ݳ���
X=enframe(x,win,inc)';            % ��֡
fn=size(X,2);                     % ��ȡ֡��
zcr1=zeros(1,fn);                 % ��ʼ��
for i=1:fn
    z=X(:,i);                     % ȡ��һ֡����
    for j=1: (wlen- 1) ;          % ��һ֡��Ѱ�ҹ����
         if z(j)* z(j+1)< 0       % �ж��Ƿ�Ϊ�����
             zcr1(i)=zcr1(i)+1;   % �ǹ���㣬��¼1��
         end
    end
end
time=(0:N-1)/Fs;                  % ����ʱ������
frameTime=frame2time(fn,wlen,inc,Fs);  % ���ÿ֡��Ӧ��ʱ��
% ��ͼ
subplot 211; plot(time,x,'k'); grid;
title('��������');
ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(a)']);
subplot 212; plot(frameTime,zcr1,'k'); grid;
title('��ʱƽ��������');
ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(b)']);
