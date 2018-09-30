%
% pr9_5_2   
clear all; clc; close all;

Formant=[800 1200 3000;                          % ����Ԫ����������
    300 2300 3000;               
    350 650 2200];
Bwp=[150 200 250];                               % ����������˲����İ����

filedir=[];                                      % ���������ļ�·��
filename='vowels8.wav';                          % �����ļ���
fle=[filedir filename]                           % ���������ļ�·�����ļ���
[xx, fs, nbits]=wavread(fle);                    % ��������
x=filter([1 -.99],1,xx);                         % Ԥ����
wlen=200;                                        % ֡��
inc=80;                                          % ֡��
y=enframe(x,wlen,inc)';                          % ��֡
fn=size(y,2);                                    % ֡��
Nx=length(x);                                    % ���ݳ���
time=(0:Nx-1)/fs;                                % ʱ��̶�
frameTime=frame2time(fn,wlen,inc,fs);            % ÿ֡��ʱ��̶�
T1=0.15; r2=0.5;                                 % ������ֵ
miniL=10;                                        % �л��ε���С֡��
[voiceseg,vosl,SF,Ef]=pitch_vad1(y,fn,T1,miniL); % �˵���
FRMNT=ones(3,fn)*nan;                            % ��ʼ��

for m=1 : vosl                                   % ��ÿһ�л��δ���
    Frt_cent=Formant(m,:);                       % ȡ���������Ƶ��
    in1=voiceseg(m).begin;                       % �л��ο�ʼ֡��
    in2=voiceseg(m).end;                         % �л��ν���֡��
    ind=in2-in1+1;                               % �л��γ���
    ix1=(in1-1)*inc+1;                           % �л����������еĿ�ʼλ��
    ix2=(in2-1)*inc+wlen;                        % �л����������еĽ���λ��
    ixd=ix2-ix1+1;                               % ���л��γ���
    z=x(ix1:ix2);                                % ��������ȡ�����л���
    for kk=1 : 3                                 % ѭ��3�μ��3�������
        fw=Frt_cent(kk);                         % ȡ����Ӧ������Ƶ��
        fl=fw-Bwp(kk);                           % ����˲����ĵͽ�ֹƵ��
        if fl<200, fl=200;   end
        fh=fw+Bwp(kk);                           % ����˲����ĸ߽�ֹƵ��
        b=fir1(100,[fl fh]*2/fs);                % ��ƴ�ͨ�˲���
        zz=conv(b,z);                            % �����˲�
        zz=zz(51:51+ixd-1);                      % �ӳ�У��
        imp=emd(zz);                             % EMD�任
        impt=hilbert(imp(1,:)');                 % ϣ�����ر任
        fnor=instfreq(impt);                     % ��ȡ˲ʱƵ��              
        f0=[fnor(1); fnor; fnor(end)]*fs;        % ���Ȳ���
        val0=abs(impt);                          % ��ģֵ
        for ii=1 : ind                           % ��ÿ֡����ƽ�������ֵ
            ixb=(ii-1)*inc+1;                    % ��֡�Ŀ�ʼλ��
            ixe=ixb+wlen-1;                      % ��֡�Ľ���λ��
            u0=f0(ixb:ixe);                      % ȡ����֡�е�����
            a0=val0(ixb:ixe);                    % ��ʽ(9-5-17)����
            a2=sum(a0.*a0);
            v0=sum(u0.*a0.*a0)/a2;
            FRMNT(kk,in1+ii-1)=v0;               % ��ֵ��FRMNT
        end
    end
end

%nfft=512;                                        % ��������ͼ
%d=stftms(x,wlen,nfft,inc);
%W2=1+nfft/2;
%n2=1:W2;
%freq=(n2-1)*fs/nfft;

% ��ͼ
figure(1)                                        % ���źŵĲ���ͼ�����ر�ͼ
subplot 211; plot(time,xx,'k');
title('\a-i-u\����Ԫ���������Ĳ���ͼ');
xlabel('ʱ��/s'); ylabel('��ֵ'); xlim([0 max(time)]);
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

figure(2)                                        % ��������ͼ
nfft=512;
d=stftms(x,wlen,nfft,inc);
W2=1+nfft/2;
n2=1:W2;
freq=(n2-1)*fs/nfft;
imagesc(frameTime,freq,abs(d(n2,:)));  axis xy
m = 64; LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0]; Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors));
hold on
plot(frameTime,FRMNT(1,:),'w',frameTime,FRMNT(2,:),'w',frameTime,FRMNT(3,:),'w')
title('����ͼ���ӹ����ֵ');
xlabel('ʱ��/s'); ylabel('Ƶ��/Hz');