%
% pr6_2_3 
clear all; clc; close all;

filedir=[];                             % ָ���ļ�·��
filename='bluesky1.wav';                % ָ���ļ���
fle=[filedir filename]                  % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                   % ���������ļ�
x=xx/max(abs(xx));                      % ���ȹ�һ��
N=length(xx);                           % ȡ�źų���
time=(0:N-1)/fs;                        % ����ʱ��̶�

wlen=200; inc=80;                       % ����֡����֡��
IS=0.25; overlap=wlen-inc;              % ����ǰ���޻��γ���
NIS=fix((IS*fs-wlen)/inc +1);           % ����ǰ���޻���֡��
y=enframe(x,wlen,inc)';                 % ��֡
etemp=sum(y.^2);                        % ��ȡ��ʱƽ������
etemp=etemp/max(etemp);                 % ������ֵ��һ��
fn=size(y,2);                           % ֡��
T1=0.002;                               % ������ֵ
T2=0.01;
frameTime=frame2time(fn, wlen, inc, fs);% �����֡��Ӧ��ʱ��
[voiceseg,vsl,SF,NF]=vad_param1D(etemp,T1,T2);% ��һ�������˵���
% ��ͼ
subplot 211; plot(time,x,'k'); hold on
title('���������������죬���ƣ����̵Ĵ󺣡�����');
ylabel('��ֵ'); axis([0 max(time) -1 1]); 
for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
end
subplot 212; plot(frameTime,etemp,'k');
title('������ʱ����ͼ');
ylabel('��ֵ'); axis([0 max(time) 0 1]);
xlabel('ʱ��/s');
line([0 max(time)],[T1 T1],'color','k','LineStyle','-');
line([0 max(time)],[T2 T2],'color','k','LineStyle','--');
