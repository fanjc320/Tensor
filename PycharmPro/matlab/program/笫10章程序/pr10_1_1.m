% pr10_1_1  
% ��conv���ص���Ӿ��conv_ovladd�������Ƚ�
%
clear all; clc; close all;
y=load('data1.txt')';                   % ��������
M=length(y);                            % ���ݳ�
t=0:M-1;                        
h=fir1(100,0.125);                      % ���FIR�˲���
x=conv(h,y);                            % ��conv�������������˲�
x=x(51:1050);                           % ȡ���ӳٵ��˲������
z=conv_ovladd1(y,h,256);                % ͨ���ص���ӷ�������
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-140)]);
plot(t,y,'k');
title('�źŵ�ԭʼ����')
xlabel('����'); ylabel('��ֵ');
%xlabel('n'); ylabel('Am');
figure(2)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-100)]);
hline1 = plot(t,z,'k','LineWidth',1.5); 
hline2 = line(t,x,'LineWidth',5,'Color',[.6 .6 .6]);
set(gca,'Children',[hline1 hline2]);
title('������ص���Ӿ���ıȽ�')
xlabel('����'); ylabel('��ֵ');
legend('���conv','�ص���Ӿ��',2)
ylim([-0.8 1]);