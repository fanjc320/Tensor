%
% pr6_1_1 
clear all; clc; close all;

filedir=[];                             % ָ���ļ�·��
filename='bluesky1.wav';                % ָ���ļ���
fle=[filedir filename]                  % ����·�����ļ������ַ���
[x,fs]=wavread(fle);                    % ���������ļ�
x=x/max(abs(x));                        % ���ȹ�һ��
N=length(x);                            % ȡ�źų���
time=(0:N-1)/fs;                        % ����ʱ��
pos = get(gcf,'Position');              % ��ͼ
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
plot(time,x,'k');         
title('���������죬���ƣ����̵Ĵ󺣡��Ķ˵���');
ylabel('��ֵ'); axis([0 max(time) -1 1]); grid;
xlabel('ʱ��/s');
wlen=200; inc=80;                       % ��֡����
IS=0.1; overlap=wlen-inc;               % ����IS
NIS=fix((IS*fs-wlen)/inc +1);           % ����NIS
fn=fix((N-wlen)/inc)+1;                 % ��֡��
frameTime=frame2time(fn, wlen, inc, fs);% ����ÿ֡��Ӧ��ʱ��
[voiceseg,vsl,SF,NF]=vad_ezm1(x,wlen,inc,NIS);  % �˵���

for k=1 : vsl                           % ������ֹ��λ��
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    nxl=voiceseg(k).duration;
    fprintf('%4d   %4d   %4d   %4d\n',k,nx1,nx2,nxl);
    line([frameTime(nx1) frameTime(nx1)],[-1.5 1.5],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1.5 1.5],'color','k','LineStyle','--');
end
