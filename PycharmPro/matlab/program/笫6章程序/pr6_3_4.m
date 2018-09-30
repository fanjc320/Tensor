%
% pr6_3_4   
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��

for k=1 : fn
    u=y(:,k);                           % ȡһ֡����
    ru=xcorr(u);                        % ��������غ���
    ru0=ru(wlen);                       % ȡ����ֵ
    ru1=max(ru(wlen+17:wlen+133));      % ȡ��һ������ֵ
    R1(k)=ru0/ru1;                      % �����������ֵ
end
Rum=multimidfilter(R1,20);              % ƽ������
Rum=Rum/max(Rum);                       % ��ֵ��һ��

alphath=mean(Rum(1:NIS));               % ������ֵ
T1=0.95*alphath; 
T2=0.75*alphath;
[voiceseg,vsl,SF,NF]=vad_param1D_revr(Rum,T1,T2);% ������˫���޷���˵���

% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title(['������������(�����' num2str(SNR) 'dB)']);
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Rum,'k');
title('��ʱ����غ���������ֵ��'); axis([0 max(time) 0 1.2]);
xlabel('ʱ��/s'); ylabel('��ֵ'); 
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
% ��������˵�
for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
    subplot 313; 
    line([frameTime(nx1) frameTime(nx1)],[0 1.2],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2],'color','k','LineStyle','--');
end

