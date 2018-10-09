%
% pr6_3_1 
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��

for k=2 : fn                            % ��������غ���
    u=y(:,k);
    ru=xcorr(u);
    Ru(k)=max(ru);
end
Rum=multimidfilter(Ru,10);              % ƽ������
Rum=Rum/max(Rum);                       % ��һ��
thredth=max(Rum(1:NIS));                % ������ֵ
T1=1.1*thredth;
T2=1.3*thredth;
[voiceseg,vsl,SF,NF]=vad_param1D(Rum,T1,T2);% ����غ����Ķ˵���
% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title(['������������(�����' num2str(SNR) 'dB)']);
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Rum,'k');
title('��ʱ����غ���'); axis([0 max(time) 0 1.2]);
xlabel('ʱ��/s'); ylabel('��ֵ'); 
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
for k=1 : vsl                           % ��������˵�
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
end



