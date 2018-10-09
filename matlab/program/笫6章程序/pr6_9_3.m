%
% pr6_9_3 
clear all; clc; close all

run Set_I                                   % ��������
SNR=0;                                      % �������������SNR
run PART_I                                  % ��������,��֡��׼��
snr1=SNR_singlech(x,signal)                 % ��������������
alpha=2.8; beta=0.001; c=1;                 % ���ò���alpha,beta��c
% ���öര�׼�����Mtmpsd_ss,ʵ�ּ��봦��
output=Mtmpsd_ssb(signal,wlen,inc,NIS,alpha,beta,c);  
snr2=SNR_singlech(x,output)                 % �������������� 

y=enframe(output,wlen,inc)';                % �Լ������źŷ�֡
aparam=2;                                   % ���ò���
for i=1:fn                                  % �����֡���ر�
    Sp = abs(fft(y(:,i)));                  % FFT�任ȡ��ֵ
    Sp = Sp(1:wlen/2+1);	                % ֻȡ��Ƶ�ʲ���
    Esum(i) = log10(1+sum(Sp.*Sp)/aparam);  % �����������ֵ
    prob = Sp/(sum(Sp));		            % �������
    H(i) = -sum(prob.*log(prob+eps));       % ������ֵ
    Ef(i) = sqrt(1 + abs(Esum(i)/H(i)));    % �������ر�
end   

Enm=multimidfilter(Ef,10);                  % ƽ���˲� 
Me=max(Enm);                                % ȡEnm�����ֵ
eth=mean(Enm(1:NIS));                       % ���ֵeth
Det=Me-eth;                                 % ������ֵ
T1=0.05*Det+eth;
T2=0.1*Det+eth;
[voiceseg,vsl,SF,NF]=vad_param1D(Enm,T1,T2);% ��˫���޷��˵���

% ��ͼ
top=Det*1.1+eth;
botton=eth-0.1*Det;
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-200,pos(3),pos(4)+100])
subplot 411; 
plot(time,x,'k');
title('��������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 412; 
plot(time,signal,'k');
title(['������������ SNR=' num2str(SNR) 'dB'] );
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 413; 
plot(time,output,'k');
title(['�׼����������� SNR=' num2str(round(snr2*100)/100) 'dB'] );
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 414; plot(frameTime,Enm,'k');  axis([0 max(time) botton top]); 
title('��ʱ���ر�'); xlabel('ʱ��/s'); 
line([0 fn],[T1 T1],'color','k','linestyle','--');
line([0 fn],[T2 T2],'color','k','linestyle','-');

for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 411
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','linestyle','--');
    subplot 414
    line([frameTime(nx1) frameTime(nx1)],[botton top],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[botton top],'color','k','linestyle','--');
end


