%
% pr6_5_1 
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��
Y=fft(y);                               % FFT�任
Y=abs(Y(1:fix(wlen/2)+1,:));            % ������Ƶ�ʷ�ֵ
N=mean(Y(:,1:NIS),2);                   % ����ǰ���޻���������ƽ��Ƶ��
NoiseCounter=0;

for i=1:fn, 
    if i<=NIS                           % ��ǰ���޻���������ΪNF=1,SF=0
        SpeechFlag=0;
        NoiseCounter=100;
        SF(i)=0;
        NF(i)=1;
    else                                % ���ÿ֡�������Ƶ�׾���
        [NoiseFlag, SpeechFlag, NoiseCounter, Dist]=vad(Y(:,i),N,NoiseCounter,2.5,8); 
        SF(i)=SpeechFlag;
        NF(i)=NoiseFlag;
        D(i)=Dist;
    end
end
sindex=find(SF==1);                     % ��SF��Ѱ�ҳ��˵�Ĳ�����ɶ˵���
voiceseg=findSegment(sindex);
vosl=length(voiceseg);
% ��ͼ
subplot 311; plot(time,x,'k'); 
title('����������');
ylabel('��ֵ'); ylim([-1 1]);
subplot 312; plot(time,signal,'k');
title(['�������� SNR=' num2str(SNR) '(dB)'])
ylabel('��ֵ'); ylim([-1.2 1.2]);
subplot 313; plot(frameTime,D,'k'); 
xlabel('ʱ��/s'); ylabel('��ֵ'); 
title('����Ƶ�׾���'); ylim([0 8]);

for k=1 : vosl                           % ��������˵�
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','linestyle','--');
    subplot 313
    line([frameTime(nx1) frameTime(nx1)],[0 8],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 8],'color','k','linestyle','--');
end
