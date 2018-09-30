%
% pr6_5_2 
clear all; clc; close all;

run Set_I                   % ��������
run PART_I                  % �������ݣ���֡��׼��
for i=1 : fn
    u=y(:,i);               % ȡ��һ֡����
    U(:,i)=rceps(u);        % ��ȡ����
end
C0=mean(U(:,1:5),2);        % �����ǰ5֡����ϵ����ƽ��ֵ��Ϊ������������ϵ���Ĺ���ֵ

for i=6 : fn                % �ӵ�6֡��ʼ����ÿ֡����ϵ���뱳����������ϵ���ľ���
    Cn=U(:,i);                           
    Dst0=(Cn(1)-C0(1)).^2;
    Dstm=0;
    for k=2 :12
        Dstm=Dstm+(Cn(k)-C0(k)).^2;
    end
    Dcep(i)=4.3429*sqrt(Dst0+Dstm);     % ���׾���
end
Dcep(1:5)=Dcep(6);
Dstm=multimidfilter(Dcep,10);           % ƽ������
dth=max(Dstm(1:(NIS)));                 % ��ֵ����
T1=1*dth;
T2=1.5*dth;
[voiceseg,vsl,SF,NF]=vad_param1D(Dstm,T1,T2);% ���׾���˫���޵Ķ˵���
% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title('������������(�����10dB)');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Dstm,'k');
title('��ʱ���׾���ֵ'); axis([0 max(time) 0 1.2*max(Dstm)]);
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
    line([frameTime(nx1) frameTime(nx1)],[0 1.2*max(Dstm)],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2*max(Dstm)],'color','k','LineStyle','--');
end

    