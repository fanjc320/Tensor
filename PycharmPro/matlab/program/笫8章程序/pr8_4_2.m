% 
% pr8_4_2 
clc; close all; clear all;

run Set_II;
run Part_II;

lmin=floor(fs/500);                              % �������ڵ���Сֵ
lmax=floor(fs/60);                               % �������ڵ����ֵ
period=zeros(1,fn);                              % �������ڳ�ʼ��
T0=zeros(1,fn);                                  % ��ʼ��
period=ACFAMDF_corr(y,fn,voiceseg,vosl,lmax,lmin);  % ��ȡ��������
T0=pitfilterm1(period,voiceseg,vosl);            % ��������ƽ������
% ��ͼ
subplot 211, plot(time,x,'k');  title('�����ź�')
axis([0 max(time) -1 1]); grid;  ylabel('��ֵ'); xlabel('ʱ��/s');
subplot 212; hold on
line(frameTime,period,'color',[.6 .6 .6],'linewidth',2); 
xlim([0 max(time)]); title('��������'); 
grid; xlabel('ʱ��/s'); ylabel('������');
plot(frameTime,T0,'k'); hold off
legend('������ֵ','ƽ����ֵ'); box on
