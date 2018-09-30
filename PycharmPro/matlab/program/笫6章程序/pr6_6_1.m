%
% pr6_6_1 
clear all; clc; close all

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��

for i=1:fn
    Sp = abs(fft(y(:,i)));              % FFT�任ȡ��ֵ
    Sp = Sp(1:wlen/2+1);	            % ֻȡ��Ƶ�ʲ���
    Ep=Sp.*Sp;                          % �������
    prob = Ep/(sum(Ep));		        % ����ÿ�����ߵĸ����ܶ�
    H(i) = -sum(prob.*log(prob+eps));  % ��������
end

Enm=multimidfilter(H,10);               % ƽ������
Me=min(Enm);                            % ������ֵ 
eth=mean(Enm(1:NIS));                   
Det=eth-Me;
T1=0.98*Det+Me;
T2=0.93*Det+Me;
[voiceseg,vsl,SF,NF]=vad_param1D_revr(Enm,T1,T2);% ��˫���޷�������˵�
% ��ͼ
subplot 311; 
plot(time,x,'k'); hold on
title('��������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; 
plot(time,signal,'k'); hold on
title(['������������ SNR=' num2str(SNR) 'dB'] );
ylabel('��ֵ'); axis([0 max(time) -1 1]);
top=Det*1.1+Me;
botton=Me-0.1*Det;
subplot 313; plot(frameTime,Enm,'k');  axis([0 max(time) botton top]); 
line([0 fn],[T1 T1],'color','k','LineStyle','--');
line([0 fn],[T2 T2],'color','k','LineStyle','-');
title('��ʱ����'); xlabel('ʱ��/s'); ylabel('����ֵ');
for k=1 : vsl                           % ��������˵�
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
    subplot 313
    line([frameTime(nx1) frameTime(nx1)],[botton top],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[botton top],'color','k','LineStyle','--');
end

