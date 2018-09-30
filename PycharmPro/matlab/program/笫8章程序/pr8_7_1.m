% 
% pr8_7_1   
clc; close all; clear all;

filedir=[];                             % ���������ļ���·��
filename='tone4.wav';                   % ���������ļ�������
fle=[filedir filename]                  % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                   % ��ȡ�ļ�
xx=xx-mean(xx);                         % ����ֱ������
xx=xx/max(abs(xx));                     % ��ֵ��һ��
SNR=5;                                  % ���������
x=Gnoisegen(xx,SNR);                    % ��������
wlen=320;  inc=80;                      % ֡����֡��
overlap=wlen-inc;                       % ��֡�ص����� 
N=length(x);                            % �źų���
time=(0:N-1)/fs;                        % ����ʱ��

y=enframe(x,wlen,inc)';                 % ��һ�η�֡
fn=size(y,2);                           % ȡ��֡��
frameTime = frame2time(fn, wlen, inc, fs);  % ����ÿһ֡��Ӧ��ʱ��
T1=0.23;                                % ����T1
[voiceseg,vsl,SF,Ef]=pitch_vad1(y,fn,T1); % �����Ķ˵���
% 60��500Hz�Ĵ�ͨ�˲���ϵ��
b=[0.012280   -0.039508   0.042177   0.000000   -0.042177   0.039508   -0.012280];
a=[1.000000   -5.527146   12.854342   -16.110307   11.479789   -4.410179   0.713507];
z=filter(b,a,x);                        % ��ͨ�����˲�
yy  = enframe(z,wlen,inc)';             % �˲����źŷ�֡

lmin=floor(fs/500);                     % �������ڵ���Сֵ
lmax=floor(fs/60);                      % �������ڵ����ֵ
period=zeros(1,fn);                     % �������ڳ�ʼ��
F0=zeros(1,fn);                         % ��ʼ�� 
period=Wavelet_corrm1(yy,fn,voiceseg,vsl,lmax,lmin); % С��-����غ����������
tindex=find(period~=0);
F0(tindex)=fs./period(tindex);          % �������Ƶ��
TT=pitfilterm1(period,voiceseg,vsl);    % ƽ������
FF=pitfilterm1(F0,voiceseg,vsl);        % ƽ������
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-150,pos(3),pos(4)+100]);
subplot 511; plot(time,xx,'k');
axis([0 max(time) -1 1]); title('ԭʼ����'); ylabel('��ֵ');
for k=1 : vsl
    line([frameTime(voiceseg(k).begin) frameTime(voiceseg(k).begin)],...
    [-1 1],'color','k','linestyle','-');
    line([frameTime(voiceseg(k).end) frameTime(voiceseg(k).end)],...
    [-1 1],'color','k','linestyle','--');
end
subplot 512; plot(frameTime,Ef,'k')
line([0 max(frameTime)],[T1 T1],'color','k','linestyle','-.');
text(3.25, T1+0.05,'T1');
axis([0 max(time) 0 1]); title('���ر�'); ylabel('��ֵ')
for k=1 : vsl
    line([frameTime(voiceseg(k).begin) frameTime(voiceseg(k).begin)],...
    [-1 1],'color','k','linestyle','-');
    line([frameTime(voiceseg(k).end) frameTime(voiceseg(k).end)],...
    [-1 1],'color','k','linestyle','--');
end
subplot 513; plot(time,x,'k');  title('��������')
axis([0 max(time) -1 1]); ylabel('��ֵ')
subplot 514; plot(frameTime,TT,'k','linewidth',2);
axis([0 max(time) 0 120]); title('��������'); grid; ylabel('������')
subplot 515; plot(frameTime,FF,'k','linewidth',2);
axis([0 max(time) 0 500]); title('����Ƶ��'); 
grid; xlabel('ʱ��/s');  ylabel('Ƶ��/Hz')
