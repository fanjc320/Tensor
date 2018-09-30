%
% pr6_3_3  
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��
[n,Wn]=buttord(300/(fs/2),600/(fs/2),3,20); % �����˲��������ʹ���
[bs,as]=butter(n,Wn);                   % ��ȡ�����˲���ϵ��
for k=1 : fn
    u=y(:,k);                           % ȡһ֡����
    ru=xcorr(u);                        % ��������غ���
    rnu=ru/max(ru);                     % ��һ��
    rpu=filter(bs,as,rnu);              % �����˲�
    Ru(k)=max(rpu);                     % Ѱ�����ֵ
end
Rum=multimidfilter(Ru,10);              % ƽ������
thredth=max(Rum(1:NIS));                % ������ֵ
T1=1.2*thredth;
T2=1.5*thredth;
[voiceseg,vsl,SF,NF]=vad_param1D(Rum,T1,T2);   % ������˫���޶˵���

% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title(['������������(�����' num2str(SNR) 'dB)']);
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Rum,'k');
title('��ʱ��һ������غ���'); grid; ylim([0 1.2]);
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



