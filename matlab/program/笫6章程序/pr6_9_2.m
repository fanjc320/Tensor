%
% pr6_9_2   
clear all; clc; close all;

run Set_I                               % ��������
SNR=0;                                  % �������������SNR
run PART_I                              % ��������,��֡��׼��
snr1=SNR_singlech(x,signal)             % �����ʼ�����ֵ
a=3; b=0.01;
output=simplesubspec(signal,wlen,inc,NIS,a,b);  % �����׼�
snr2=SNR_singlech(x,output)             % �����׼��������ֵ
y=enframe(output,wlen,inc)';            % �׼���������з�֡
nl2=wlen/2+1;
Y=fft(y);                               % FFTת��Ƶ��
Y_abs=abs(Y(1:nl2,:));                  % ȡ��Ƶ�����ֵ
M=floor(nl2/4);                         % �����Ӵ���
for k=1 : fn
    for i=1 : M                         % ÿ���Ӵ���4���������
        j=(i-1)*4+1;
        SY(i,k)=Y_abs(j,k)+Y_abs(j+1,k)+Y_abs(j+2,k)+Y_abs(j+3,k);
    end
    Dvar(k)=var(SY(:,k));               % ����ÿ֡�Ӵ������Ƶ������
end
Dvarm=multimidfilter(Dvar,10);          % ƽ������
dth=max(Dvarm(1:(NIS)));                % ��ֵ����
T1=1.5*dth;
T2=3*dth;
[voiceseg,vsl,SF,NF]=vad_param1D(Dvarm,T1,T2);% Ƶ�򷽲�˫���޵Ķ˵���

% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-150,pos(3),pos(4)+100]) 
subplot 411; plot(time,x,'k'); axis tight;
title('����������'); ylabel('��ֵ'); xlabel('(a)');
subplot 412; plot(time,signal,'k'); axis tight; xlabel('(b)');
title(['�������� �����=' num2str(SNR) 'dB']); ylabel('��ֵ')
subplot 413; plot(time,output,'k'); xlabel('(c)');
title(['�׼����������� SNR=' num2str(round(snr2*100)/100) 'dB'] ); 
ylabel('��ֵ')
subplot 414; plot(frameTime,Dvarm,'k');
title('�׼���ʱ�����Ӵ�Ƶ������ֵ'); 
xlabel(['ʱ��/s' 10 '(d)']);  ylabel('����ֵ');
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
ylim([0 max(Dvar)]);
for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 411; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
    subplot 414; 
    line([frameTime(nx1) frameTime(nx1)],[0 max(Dvar)],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 max(Dvar)],'color','k','LineStyle','--');
end



        
