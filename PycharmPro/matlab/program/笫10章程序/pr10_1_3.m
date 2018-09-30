%
% pr10_1_3 
clear all; clc; close all;

load Labdata1                 % ����ʵ�����ݺ���������
N=length(xx);                 % ���ݳ�
time=(0:N-1)/fs;              % ʱ����
T=10091-8538+1;               % ȱ����������ĳ���
x1=xx(1:8537);                % ǰ������
x2=xx(10092:29554);           % �������
y1=ydata(:,1);                % ��������1
xx1=[x1; y1; x2];             % ����������1�ϳ�
y2=ydata(:,2);                % ��������2
xx2=[x1; y2; x2];             % ����������2�ϳ�
% ����������1����������2�����Ա����ص���Ӻϳ�
Wt1=(0:T-1)'/T;               % ����б���Ǵ�����w1
Wt2=(T-1:-1:0)'/T;            % ����б���Ǵ�����w2
y=y1.*Wt2+y2.*Wt1;            % ���Ա����ص����
xx3=[x1; y; x2];              % �ϳ�����
%��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)-240]);
plot(time,xx,'k'); axis([0 29.6 -15 10]); 
title('ԭʼ�źŵĲ���'); xlabel('ʱ��/s'); ylabel('��ֵ')
line([8.537 8.537],[-15 10],'color','k','linestyle','-');
line([10.092 10.092],[-15 10],'color','k','linestyle','--');

figure(2)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)+50]);
subplot 221; plot(time,xx1,'k'); axis([9.5 10.5 -10 5]);
line([10.091 10.091],[-15 10],'color','k','linestyle','-.');
title('��һ�������ϳɵĲ���'); xlabel(['ʱ��/s' 10 '(a)']); ylabel('��ֵ')
subplot 222; plot(time,xx2,'k'); axis([8 9.5 -10 5]); 
line([8.538 8.538],[-15 10],'color','k','linestyle','-.');
title('�ڶ��������ϳɵĲ���'); xlabel(['ʱ��/s' 10 '(b)']); ylabel('��ֵ')
subplot 212; plot(time,xx3,'k');  axis([0 29.6 -15 10]); 
line([8.537 8.537],[-15 10],'color','k','linestyle','-');
line([10.092 10.092],[-15 10],'color','k','linestyle','--');
title('���Ա����ص���Ӻ�ϳɵĲ���'); xlabel(['ʱ��/s' 10 '(c)']); ylabel('��ֵ')

