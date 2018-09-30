%
% pr6_2_1 
clear all; clc; close all;

filedir=[];                             % ָ���ļ�·��
filename='bluesky1.wav';                % ָ���ļ���
fle=[filedir filename]                  % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                   % ���������ļ�
xx=xx/max(abs(xx));                     % ���ȹ�һ��
N=length(xx);                           % ȡ�źų���
time=(0:N-1)/fs;                        % ����ʱ��̶�
x=Gnoisegen(xx,20);                     % �Ѱ��������ӵ��ź���

wlen=200; inc=80;                       % ����֡����֡��
IS=0.25; overlap=wlen-inc;              % ����ǰ���޻��γ���
NIS=fix((IS*fs-wlen)/inc +1);           % ����ǰ���޻���֡��
fn=fix((N-wlen)/inc)+1;                 % �����֡��
frameTime=frame2time(fn, wlen, inc, fs);% ����ÿ֡��Ӧ��ʱ��
[voiceseg,vsl,SF,NF]=vad_ezr(x,wlen,inc,NIS); % �˵���
% ��ͼ
subplot 211; plot(time,xx,'k'); hold on
title('���������������죬���ƣ����̵Ĵ󺣡�����');
ylabel('��ֵ'); axis([0 max(time) -1 1]); xlabel('(a)');
for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    line([frameTime(nx1) frameTime(nx1)],[-1.5 1.5],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1.5 1.5],'color','k','LineStyle','--');
end
subplot 212; plot(time,x,'k');
title('������������(�����20dB)');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
xlabel(['ʱ��/s' 10 '(b)']);


