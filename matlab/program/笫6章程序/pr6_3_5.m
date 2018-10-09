%
% pr6_3_5  
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��

Rw=zeros(2*wlen-1,1);                   % Rw��ʼ��
for k=1 : NIS                           % ��ʽ(6-3-6)����Rw
    u=y(:,k);                           % ȡһ֡����
    ru=xcorr(u);                        % ��������غ���
    Rw=Rw+ru;
end
Rw=Rw/NIS;
Rw2=sum(Rw.*Rw);                        % ����ʽ(6-3-5)�з�ĸ��Rw�Ĳ���

for k=1 : fn
    u=y(:,k);                           % ȡһ֡����
    ru=xcorr(u);                        % ��������غ���
    Cm=sum(ru.*Rw);                     % ����ʽ(6-3-5)�з��Ӳ���
    Cru=sum(ru.*ru);                    % ����ʽ(6-3-5)�з�ĸ��Ry�Ĳ���
    Ru(k)=Cm/sqrt(Rw2*Cru);             % ����ʽ(6-3-5)ÿ֡������غ������Ҽн�
end

Rum=multimidfilter(Ru,10);              % ƽ������
alphath=mean(Rum(1:NIS));               % ������ֵ
T2=0.8*alphath; T1=0.9*alphath;
[voiceseg,vsl,SF,NF]=vad_param1D_revr(Rum,T1,T2);   % ������˫���޷���˵���
% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title(['������������(�����' num2str(SNR) 'dB)']);
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Rum,'k');
title('��ʱ����غ������Ҽн�ֵ'); axis([0 max(time) 0 1]);
xlabel('ʱ��/s'); ylabel('��ֵ'); 
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
for k=1 : vsl                           % ��������˵�
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
    subplot 313; 
    line([frameTime(nx1) frameTime(nx1)],[0 1.2],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2],'color','k','LineStyle','--');
end


