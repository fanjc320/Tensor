% 
% pr8_4_1 
clc; close all; clear all;

run Set_II;                                     % ��������
run Part_II;                                    % �����ļ�,��֡�Ͷ˵���
% �˲���ϵ��
b=[0.012280   -0.039508   0.042177   0.000000   -0.042177   0.039508   -0.012280];
a=[1.000000   -5.527146   12.854342   -16.110307   11.479789   -4.410179   0.713507];
xx=filter(b,a,x);                               % ��ͨ�����˲�
yy  = enframe(xx,wlen,inc)';                    % �˲����źŷ�֡

lmin=floor(fs/500);                             % �������ڵ���Сֵ
lmax=floor(fs/60);                              % �������ڵ����ֵ
period=zeros(1,fn);                             % �������ڳ�ʼ��
period=AMDF_mod(yy,fn,voiceseg,vosl,lmax,lmin); % ��AMDF_mod������ȡ��������
T0=pitfilterm1(period,voiceseg,vosl);           % ��������ƽ������
% ��ͼ
subplot 211, plot(time,x,'k');  title('�����ź�')
axis([0 max(time) -1 1]); grid;  ylabel('��ֵ'); xlabel('ʱ��/s');
subplot 212; hold on
line(frameTime,period,'color',[.6 .6 .6],'linewidth',2); 
axis([0 max(time) 0 120]); title('��������'); 
grid; xlabel('ʱ��/s'); ylabel('������');
subplot 212; plot(frameTime,T0,'k'); hold off
legend('������ֵ','ƽ����ֵ'); box on;
