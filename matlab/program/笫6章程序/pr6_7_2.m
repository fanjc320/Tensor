%
% pr6_7_2 
clear all; clc; close all

run Set_I                                    % ��������
run PART_I                                   % �������ݣ���֡��׼��
aparam=2;                                    % ���ò���
for i=1:fn
    Sp = abs(fft(y(:,i)));                   % FFT�任ȡ��ֵ
    Sp = Sp(1:wlen/2+1);	                 % ֻȡ��Ƶ�ʲ���
    Esum(i) = log10(1+sum(Sp.*Sp)/aparam);   % �����������ֵ
    prob = Sp/(sum(Sp));		             % �������
    H(i) = -sum(prob.*log(prob+eps));        % ������ֵ
    Ef(i) = sqrt(1 + abs(Esum(i)/H(i)));     % �������ر�
end   

Enm=multimidfilter(Ef,10);                   % ƽ���˲� 
Me=max(Enm);                                 % Enm���ֵ
eth=mean(Enm(1:NIS));                        % ��ʼ��ֵeth
Det=Me-eth;                                  % ���ֵ��������ֵ
T1=0.05*Det+eth;
T2=0.1*Det+eth;
[voiceseg,vsl,SF,NF]=vad_param1D(Enm,T1,T2); % �����رȷ���˫���޶˵���

% ��ͼ
top=Det*1.1+eth;
botton=eth-0.1*Det;
subplot 311; 
plot(time,x,'k'); hold on
title('��������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; 
plot(time,signal,'k'); hold on
title(['������������ SNR=' num2str(SNR) 'dB'] );
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Enm,'k');  axis([0 max(time) botton top]); 
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
title('��ʱ���ر�'); xlabel('ʱ��/s');
for k=1 : vsl                           % ��������˵�
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','linestyle','--');
    subplot 313
    line([frameTime(nx1) frameTime(nx1)],[botton top],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[botton top],'color','k','linestyle','--');
end

