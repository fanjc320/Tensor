%
% pr6_9_1 
clear all; clc; close all;

filedir=[];                             % �����ļ�·��
filename='bluesky1.wav';                % �����ļ�����
fle=[filedir filename]                  % �����ļ�·��������
[xx,fs]=wavread(fle);                   % ��������
x=xx/max(abs(xx));                      % ��ֵ��һ��
N=length(x);
time=(0:N-1)/fs;                        % ����ʱ��
IS=0.3;                                 % ����ǰ���޻��γ���
wlen=200;                               % ����֡��Ϊ25ms
inc=80;                                 % ��֡��
SNR=20;                                 % ���������
wnd=hamming(wlen);                      % ���ô�����
overlap=wlen-inc;                       % ���ص�������
NIS=fix((IS*fs-wlen)/inc +1);           % ��ǰ���޻���֡��


noisefile='destroyerops.wav';           % ָ���������ļ���
[signal,noise] = add_noisefile(x,noisefile,SNR,fs);% ��������
y=enframe(signal,wnd,inc)';             % ��֡
fn=size(y,2);                           % ��֡��
frameTime=frame2time(fn, wlen, inc, fs);

Y=fft(y);                               % FFT
Y=abs(Y(1:fix(wlen/2)+1,:));            % ������Ƶ�ʷ�ֵ
N=mean(Y(:,1:NIS),2);                   % ����ǰ���޻���������ƽ��Ƶ��
NoiseCounter=0;
NoiseLength=9;

for i=1:fn, 
    if i<=NIS                           % ��ǰ���޻���������ΪNF=1,SF=0
        SpeechFlag=0;
        NoiseCounter=100;
        SF(i)=0;
        NF(i)=1;
        TNoise(:,i)=N;
    else                                % ���ÿ֡�������Ƶ�׾���
        [NoiseFlag, SpeechFlag, NoiseCounter, Dist]=...
            vad(Y(:,i),N,NoiseCounter,2.5,8); 
        SF(i)=SpeechFlag;
        NF(i)=NoiseFlag;
        D(i)=Dist;
        if SpeechFlag==0                % ����������ζ������׽��и���
            N=(NoiseLength*N+Y(:,i))/(NoiseLength+1); %
            
        end
        TNoise(:,i)=N;
    end
    SN(i)=sum(TNoise(:,i));             % ���������׷�ֵ֮��
end
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)+100]) 
subplot 411; plot(time,x,'k',frameTime,SF,'k--'); 
title('����������');
ylabel('��ֵ'); ylim([-1.2 1.2]);
subplot 412; plot(time,signal,'k');
title(['�������� SNR=' num2str(SNR) '(dB)'])
ylabel('��ֵ'); ylim([-1.5 1.5]);
subplot 413; plot(frameTime,D,'k'); 
ylabel('��ֵ'); grid;
title('����Ƶ�׾���'); ylim([0 25]);
subplot 414; plot(frameTime,SN,'k','linewidth',2); 
xlabel('ʱ��/s'); ylabel('��ֵ'); grid;
title('������ֵ��');  ylim([7 10]);

