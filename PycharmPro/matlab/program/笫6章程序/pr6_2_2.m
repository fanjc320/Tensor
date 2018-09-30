%
% pr6_2_2 
clear all; clc; close all;

filedir=[];                             % ָ���ļ�·��
filename='bluesky1.wav';                % ָ���ļ���
fle=[filedir filename]                  % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                   % ���������ļ�
xx=xx/max(abs(xx));                     % ���ȹ�һ��
N=length(xx);                           % ȡ�źų���
time=(0:N-1)/fs;                        % ����ʱ��̶�
SNR=10;                                 % �����
x=Gnoisegen(xx,SNR);                    % �Ѱ��������ӵ��ź���

wlen=200; inc=80;                       % ����֡����֡��
IS=0.25; overlap=wlen-inc;              % ����ǰ���޻��γ���
NIS=fix((IS*fs-wlen)/inc +1);           % ����ǰ���޻���֡��
y=enframe(x,wlen,inc)';                 % ��֡
fn=size(y,2);                           % ֡��
amp=sum(y.^2);                          % ��ȡ��ʱƽ������
zcr=zc2(y,fn);                          % �����ʱƽ��������  
ampm = multimidfilter(amp,5);           % ��ֵ�˲�ƽ������
zcrm = multimidfilter(zcr,5);         
ampth=mean(ampm(1:NIS));                % �����ʼ�޻������������͹����ʵ�ƽ��ֵ 
zcrth=mean(zcrm(1:NIS));
amp2=1.1*ampth; amp1=1.3*ampth;         % ���������͹����ʵ���ֵ
zcr2=0.9*zcrth;

frameTime=frame2time(fn, wlen, inc, fs);% �����֡��Ӧ��ʱ��
[voiceseg,vsl,SF,NF]=vad_param2D_revr(amp,zcr,amp2,amp1,zcr2);% �˵���
% ��ͼ
subplot 211; plot(time,xx,'k');
title('���������������죬���ƣ����̵Ĵ󺣡�����');
ylabel('��ֵ'); axis([0 max(time) -1 1]); 
for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    line([frameTime(nx1) frameTime(nx1)],[-1.5 1.5],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1.5 1.5],'color','k','LineStyle','--');
end
subplot 212; plot(time,x,'k');
title(['������������(�����' num2str(SNR) 'dB)']);
ylabel('��ֵ'); axis([0 max(time) -1 1]);
xlabel('ʱ��/s');

