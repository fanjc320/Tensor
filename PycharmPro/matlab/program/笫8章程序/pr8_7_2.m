%
% pr8_7_2  
clear all; clc; close all;

filedir=[];                             % ���������ļ���·��
filename='tone4.wav';                   % ���������ļ�������
fle=[filedir filename]                  % ����·�����ļ������ַ���
[x,fs]=wavread(fle);                    % ��ȡ�ļ�
x=x-mean(x);                            % ����ֱ������
x=x/max(abs(x));                        % ��ֵ��һ��
SNR=0;                                  % ���������
signal=Gnoisegen(x,SNR);                % ��������
snr1=SNR_singlech(x,signal)             % �����ʼ�����ֵ
N=length(x);                            % �źų���
time = (0 : N-1)/fs;                    % ����ʱ��̶�
wlen=320; inc=80;                       % ֡����֡��
overlap=wlen-inc;                       % ��֡�ص�����  
IS=0.15;                                % ����ǰ���޻��γ���
NIS=fix((IS*fs-wlen)/inc +1);           % ��ǰ���޻���֡��

a=3; b=0.001;                           % ���ò���a��b
output=simplesubspec(signal,wlen,inc,NIS,a,b); % �׼�����
snr2=SNR_singlech(x,output)             % �����׼��������ֵ
y  = enframe(output,wlen,inc)';         % ��֡
fn  = size(y,2);                        % ȡ��֡��
time = (0 : length(x)-1)/fs;            % ����ʱ������
frameTime = frame2time(fn, wlen, inc, fs);% ����ÿһ֡��Ӧ��ʱ��
T1=0.12;                                 % ���û����˵���Ĳ���

[voiceseg,vosl,SF,Ef]=pitch_vad1(y,fn,T1);   % �����Ķ˵���
% 60��500Hz�Ĵ�ͨ�˲���ϵ��
b=[0.012280   -0.039508   0.042177   0.000000   -0.042177   0.039508   -0.012280];
a=[1.000000   -5.527146   12.854342   -16.110307   11.479789   -4.410179   0.713507];
z=filter(b,a,output);                   % ��ͨ�����˲�
yy  = enframe(z,wlen,inc)';             % �˲����źŷ�֡

lmin=floor(fs/500);                     % �������ڵ���Сֵ
lmax=floor(fs/60);                      % �������ڵ����ֵ
period=zeros(1,fn);                     % �������ڳ�ʼ��
period=ACF_corr(yy,fn,voiceseg,vosl,lmax,lmin);  % ������غ�����ȡ��������
tindex=find(period~=0);
F0=zeros(1,fn);                         % ��ʼ�� 
F0(tindex)=fs./period(tindex);          % �������Ƶ��
TT=pitfilterm1(period,voiceseg,vosl);   % �Ի������ڽ���ƽ���˲�
FF=pitfilterm1(F0,voiceseg,vosl);       % �Ի���Ƶ�ʽ���ƽ���˲�
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-150,pos(3),pos(4)+100]);
subplot 611; plot(time,x,'k'); ylabel('��ֵ');
title('ԭʼ�ź�'); axis([0 max(time) -1 1]);
for k=1 : vosl
        line([frameTime(voiceseg(k).begin) frameTime(voiceseg(k).begin)],...
        [-1 1],'color','k');
        line([frameTime(voiceseg(k).end) frameTime(voiceseg(k).end)],...
        [-1 1],'color','k','linestyle','--');
end
subplot 612; plot(time,signal,'k'); ylabel('��ֵ');
title('�����ź�'); axis([0 max(time) -1 1]);
subplot 613; plot(time,output,'k');
title('�����ź�'); axis([0 max(time) -1 1]); ylabel('��ֵ');
subplot 614; plot(frameTime,Ef,'k'); hold on; ylabel('��ֵ');
title('���ر�'); axis([0 max(time) 0 max(Ef)]);
line([0 max(frameTime)],[T1 T1],'color','k','linestyle','--');
for k=1 : vosl
        line([frameTime(voiceseg(k).begin) frameTime(voiceseg(k).begin)],...
        [-1 1],'color','k');
        line([frameTime(voiceseg(k).end) frameTime(voiceseg(k).end)],...
        [-1 1],'color','k','linestyle','--');
end
text(3.2,0.2,'T1');
subplot 615; plot(frameTime,TT,'k');  ylabel('������');
title('��������'); grid; axis([0 max(time) 0 80]);
subplot 616; plot(frameTime,FF,'k'); ylabel('Ƶ��/Hz')
title('����Ƶ��'); grid; axis([0 max(time) 0 450]); xlabel('ʱ��/s'); 
