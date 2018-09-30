%
% pr9_4_1  
clear all; clc; close all;

filedir=[];                                           % ���������ļ���·��
filename='vowels8.wav';                               % ���������ļ�������
fle=[filedir filename]                                % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                                 % ���������ļ�
x=xx-mean(xx);                                        % ���� ֱ������
x=x/max(abs(x));                                      % ��ֵ��һ��
y=filter([1 -.99],1,x);                               % Ԥ����
wlen=200;                                             % ����֡��
inc=80;                                               % ����֡��
xy=enframe(y,wlen,inc)';                              % ��֡
fn=size(xy,2);                                        % ��֡��
Nx=length(y);                                         % ���ݳ���
time=(0:Nx-1)/fs;                                     % ʱ��̶�
frameTime=frame2time(fn,wlen,inc,fs);                 % ÿ֡��Ӧ��ʱ��̶�
T1=0.1;                                               % ������ֵT1��T2�ı�������
miniL=20;                                             % �л��ε���С֡��
p=9; thr1=0.75;                                       % ����Ԥ���������ֵ
[voiceseg,vosl,SF,Ef]=pitch_vad1(xy,fn,T1,miniL);     % �˵���
Msf=repmat(SF',1,3);                                  % ��SF��չΪfn��3������
formant1=Ext_frmnt(xy,p,thr1,fs);                     % ��ȡ�������Ϣ

Fmap1=Msf.*formant1;                                  % ֻȡ�л��ε�����
findex=find(Fmap1==0);                                % �������ֵΪ0 ,��Ϊnan
Fmap=Fmap1;
Fmap(findex)=nan;

nfft=512;                                             % ��������ͼ
d=stftms(y,wlen,nfft,inc);
W2=1+nfft/2;
n2=1:W2;
freq=(n2-1)*fs/nfft;
warning off

% ��ͼ
figure(1)                                             % ���źŵĲ���ͼ�����ر�ͼ
subplot 211; plot(time,x,'k');
title('\a-i-u\����Ԫ���������Ĳ���ͼ');
xlabel('ʱ��/s'); ylabel('��ֵ'); axis([0 max(time) -1.2 1.2]);
subplot 212; plot(frameTime,Ef,'k'); hold on
line([0 max(time)],[T1 T1],'color','k','linestyle','--');
title('��һ�������ر�ͼ'); axis([0 max(time) 0 1.2]);
xlabel('ʱ��/s'); ylabel('��ֵ')
for k=1 : vosl
    in1=voiceseg(k).begin;
    in2=voiceseg(k).end;
    it1=frameTime(in1);
    it2=frameTime(in2);
    line([it1 it1],[0 1.2],'color','k','linestyle','-.');
    line([it2 it2],[0 1.2],'color','k','linestyle','-.');
end

figure(2)                                             % �������źŵ�����ͼ
imagesc(frameTime,freq,abs(d(n2,:)));  axis xy
m = 64; LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0]; Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors)); hold on
plot(frameTime,Fmap,'w');                             % �����Ϲ����Ƶ������
title('������ͼ�ϱ�������Ƶ��');
xlabel('ʱ��/s'); ylabel('Ƶ��/Hz')
