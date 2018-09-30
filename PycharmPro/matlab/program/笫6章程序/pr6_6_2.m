%
% pr6_6_2 
clear all; clc; close all

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��
df=fs/wlen;                             % ���FFT��Ƶ�ʷֱ���
fx1=fix(250/df)+1; fx2=fix(3500/df)+1;  % �ҳ�250Hz��3500Hz��λ��
km=floor(wlen/8);                       % ������Ӵ�����
K=0.5;                                  % ����K
for i=1:fn
    A=abs(fft(y(:,i)));                 % ȡ��һ֡����FFT��ȡ��ֵ
    E=zeros(wlen/2+1,1);            
    E(fx1+1:fx2-1)=A(fx1+1:fx2-1);      % ֻȡ250��3500Hz֮��ķ���
    E=E.*E;                             % ��������
    P1=E/sum(E);                        % ��ֵ��һ��
    index=find(P1>=0.9);                % Ѱ���Ƿ��з����ĸ��ʴ���0.9
    if ~isempty(index), E(index)=0; end % ����,�÷�����0
    for m=1:km                          % �����Ӵ�����
        Eb(m)=sum(E(4*m-3:4*m));
    end
    prob=(Eb+K)/sum(Eb+K);              % �����Ӵ�����
    Hb(i) = -sum(prob.*log(prob+eps));  % �����Ӵ�����
end   
Enm=multimidfilter(Hb,10);              % ƽ������
Me=min(Enm);                            % ������ֵ
eth=mean(Enm(1:NIS));
Det=eth-Me;
T1=0.99*Det+Me;
T2=0.96*Det+Me;
[voiceseg,vsl,SF,NF]=vad_param1D_revr(Enm,T1,T2);% ��˫���޷�����˵���  
% ��ͼ
top=Det*1.1+Me;
botton=Me-0.1*Det;
subplot 311; 
plot(time,x,'k'); hold on
title('��������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; 
plot(time,signal,'k'); hold on
title(['������������ SNR=' num2str(SNR) 'dB'] );
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Enm,'k');  axis([0 max(time) botton top]); 
line([0 fn],[T1 T1],'color','k','LineStyle','--');
line([0 fn],[T2 T2],'color','k','LineStyle','-');
title('��ʱ�Ľ��Ӵ�����'); xlabel('ʱ��/s'); ylabel('����ֵ');
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

