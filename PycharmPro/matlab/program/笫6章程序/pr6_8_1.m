%
% pr6_8_1 
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��
% С���ֽ�����
start=[ 1  8  15  22  29  37  47  60  79  110  165];
send=[ 7  14  21  28  36  46  59  78  109  164  267];
duration=[ 7  7  7  7  8  10  13  19  31  55  103];

for i=1 : fn
    u=y(:,i);                           % ȡһ֡
    [c,l]=wavedec(u,10,'db4');          % ��ĸС��db4����10��ֽ�
    for k=1 : 10
        E(11-k)=mean(abs(c(start(k+1):send(k+1))));% ����ÿ���ƽ����ֵ
    end
    M1=max(E(1:5)); M2=max(E(6:10));    % ��ʽ(6-8-2)��M1��M2
    MD(i)=M1*M2;                        % ��ʽ(6-8-3)����MD
end
MDm=multimidfilter(MD,10);              % ƽ������
MDmth=mean(MDm(1:NIS));                 % ������ֵ
T1=2*MDmth;
T2=3*MDmth;
[voiceseg,vsl,SF,NF]=vad_param1D(MDm,T1,T2);% ��С���ֽ�ϵ��ƽ����ֵ������˫���޶˵���
% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title(['������������ �����=' num2str(SNR) 'dB']);
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,MDm,'k');
title('С���ֽ��ʱϵ��ƽ����ֵ��'); grid; ylim([0 1.2*max(MDm)]); 
xlabel('ʱ��/s'); ylabel('��ֵ');
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
    subplot 313; 
    line([frameTime(nx1) frameTime(nx1)],[0 1.2*max(MDm)],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2*max(MDm)],'color','k','LineStyle','--');
end
