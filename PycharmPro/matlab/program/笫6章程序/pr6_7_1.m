% 
% pr6_7_1 
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��

aparam=2; bparam=1;                     % ���ò���
etemp=sum(y.^2);                        % ��������
etemp1=log10(1+etemp/aparam);           % ���������Ķ���ֵ
zcr=zc2(y,fn);                          % ������ֵ
Ecr=etemp1./(zcr+bparam);               % ���������

Ecrm=multimidfilter(Ecr,2);             % ƽ������
dth=mean(Ecrm(1:(NIS)));                % ��ֵ����
T1=1.2*dth;
T2=2*dth;
[voiceseg,vsl,SF,NF]=vad_param1D(Ecrm,T1,T2);% ����ȷ���˫���޶˵���
% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title('������������(�����10dB)');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Ecrm,'k');
title('��ʱ�����ֵ'); grid; ylim([0 1.2*max(Ecrm)]);
xlabel('ʱ��/s'); ylabel('�����ֵ'); 
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
for k=1 : vsl                           % ��������˵�
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
    subplot 313; 
    line([frameTime(nx1) frameTime(nx1)],[0 1.2*max(Ecrm)],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2*max(Ecrm)],'color','k','LineStyle','--');
end

