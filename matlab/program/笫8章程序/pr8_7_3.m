%
% pr8_7_3   
clear all; clc; close all;

filedir=[];                               % ���������ļ�·��
filename='tone4.wav';                     % �����ļ���
fle=[filedir filename]                    % ����·�����ļ������ַ���
SNR=0;                                    % ���������
IS=0.15;                                  % ����ǰ���޻��γ���
wlen=240;                                 % ����֡��Ϊ25ms
inc=80;                                   % ����֡��10ms
[xx,fs]=wavread(fle);                     % ��������
xx=xx-mean(xx);                           % ����ֱ������
x=xx/max(abs(xx));                        % ��ֵ��һ��
N=length(x);
time=(0:N-1)/fs;                          % ����ʱ��
lmin=floor(fs/500);                       % ��߻���Ƶ��500Hz��Ӧ�Ļ�������
lmax=floor(fs/60);                        % ��߻���Ƶ��60Hz��Ӧ�Ļ������� 
signal=Gnoisegen(x,SNR);                  % ��������
overlap=wlen-inc;                         % ���ص�������
NIS=fix((IS*fs-wlen)/inc +1);             % ��ǰ���޻���֡��
snr1=SNR_singlech(x,signal)               % �����ʼ�����ֵ
a=3; b=0.01;                              % ����a��b
output=simplesubspec(signal,wlen,inc,NIS,a,b);% �׼�������
snr2=SNR_singlech(x,output)               % �����׼��������ֵ
yy  = enframe(output,wlen,inc)';          % �˲����źŷ�֡
time = (0 : length(x)-1)/fs;              % ����ʱ������
fn=size(yy,2);
frameTime=frame2time(fn,wlen,inc,fs);

Thr1=0.12;                                % ���ö˵�����ֵ
r2=0.5;                                   % ����Ԫ��������ı�������
ThrC=[10 15];                             % �������ڻ������ڼ����ֵ
% ���ڻ������Ķ˵����Ԫ��������
[voiceseg,vosl,vseg,vsl,Thr2,Bth,SF,Ef]=pitch_vads(yy,fn,Thr1,r2,10,8);
% 60��500Hz�Ĵ�ͨ�˲���ϵ��
b=[0.012280   -0.039508   0.042177   0.000000   -0.042177   0.039508   -0.012280];
a=[1.000000   -5.527146   12.854342   -16.110307   11.479789   -4.410179   0.713507];
x=filter(b,a,xx);                         % �����˲�
x=x/max(abs(x));                          % ��ֵ��һ��
y=enframe(x,wlen,inc)';                   % �ڶ��η�֡
lmax=floor(fs/60);                        % �������ڵ���Сֵ
lmin=floor(fs/500);                       % �������ڵ����ֵ
[Extseg,Dlong]=Extoam(voiceseg,vseg,vosl,vsl,Bth);% ����������������쳤��
T1=ACF_corrbpa(y,fn,vseg,vsl,lmax,lmin,ThrC(1));  % ��Ԫ��������л������
% �����������ǰ�������������������        
T0=zeros(1,fn);                           % ��ʼ��
F0=zeros(1,fn);
for k=1 : vsl                             % ����vsl��Ԫ������
    ix1=vseg(k).begin;                    % ��k��Ԫ�����忪ʼλ��
    ix2=vseg(k).end;                      % ��k��Ԫ���������λ��
    in1=Extseg(k).begin;                  % ��k��Ԫ������ǰ�����쿪ʼλ��
    in2=Extseg(k).end;                    % ��k��Ԫ����������������λ��
    ixl1=Dlong(k,1);                      % ǰ�����쳤��
    ixl2=Dlong(k,2);                      % �������쳤��
    if ixl1>0                             % ��Ҫǰ������������
        [Bt,voiceseg]=back_Ext_shtpm1(y,fn,voiceseg,Bth,ix1,...
        ixl1,T1,k,lmax,lmin,ThrC);
    else                                  % ����Ҫǰ������������
        Bt=[];
    end
    if ixl2>0                             % ��Ҫ��������������
        [Ft,voiceseg]=fore_Ext_shtpm1(y,fn,voiceseg,Bth,ix2,...
        ixl2,vsl,T1,k,lmax,lmin,ThrC);
    else                                  % ����Ҫ��������������
        Ft=[];
    end
    T0(in1:in2)=[Bt T1(ix1:ix2) Ft];      % ��k��Ԫ�����������ǰ���������� 
end
tindex=find(T0>lmax);                     % ��ֹ���������������ֵ����Խlmax
T0(tindex)=lmax;
tindex=find(T0<lmin & T0~=0);             % ��ֹ�������С��������ֵ������lmin
T0(tindex)=lmin;
tindex=find(T0~=0);
F0(tindex)=fs./T0(tindex);                 % �������Ƶ��
TT=pitfilterm1(T0,Extseg,vsl);             % T0ƽ���˲�
FF=pitfilterm1(F0,Extseg,vsl);             % F0ƽ���˲�
% STFT����,��������ͼ
nfft=512;
d=stftms(xx,wlen,nfft,inc);
W2=1+nfft/2;
n2=1:W2;
freq=(n2-1)*fs/nfft;
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-150,pos(3),pos(4)+150]);
subplot 611; plot(time,xx,'k'); ylabel('��ֵ');
title('ԭʼ�ź�'); axis([0 max(time) -1 1]);
for k=1 : vosl
        line([frameTime(voiceseg(k).begin) frameTime(voiceseg(k).begin)],...
        [-1 1],'color','k');
        line([frameTime(voiceseg(k).end) frameTime(voiceseg(k).end)],...
        [-1 1],'color','k','linestyle','--');
end
subplot 612; plot(time,signal,'k'); ylabel('��ֵ');
title('�����ź�'); axis([0 max(time) -1 1]);
subplot 613; plot(time,output,'k'); ylabel('��ֵ');
title('�����ź�'); axis([0 max(time) -1 1]);
subplot 614; plot(frameTime,Ef,'k'); hold on; ylabel('��ֵ');
title('���ر�'); axis([0 max(time) 0 max(Ef)]);
line([0 max(frameTime)],[Thr1 Thr1],'color','k','linestyle','--');
plot(frameTime,Thr2,'k','linewidth',2);
for k=1 : vsl
        line([frameTime(vseg(k).begin) frameTime(vseg(k).begin)],...
        [0 max(Ef)],'color','k','linestyle','-.');
        line([frameTime(vseg(k).end) frameTime(vseg(k).end)],...
        [0 max(Ef)],'color','k','linestyle','-.');
end
text(3.2,0.2,'Thr1');
text(2.9,0.55,'Thr2');
subplot 615; plot(frameTime,TT,'k'); ylabel('������'); 
title('��������'); grid; axis([0 max(time) 0 80]);
subplot 616; plot(frameTime,FF,'k'); ylabel('Ƶ��/Hz'); 
title('����Ƶ��'); grid; axis([0 max(time) 0 350]); xlabel('ʱ��/s'); 

figure(2)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-240)]);
imagesc(frameTime,freq,abs(d(n2,:)));  axis xy
m = 64; LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0]; Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors));
hold on
plot(frameTime,FF,'w');
ylim([0 1000]);
title('����ͼ�ϵĻ���Ƶ������');
xlabel('ʱ��/s'); ylabel('Ƶ��/Hz')
