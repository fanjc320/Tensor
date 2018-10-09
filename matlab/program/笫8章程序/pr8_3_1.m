% 
% pr8_3_1 
clc; close all; clear all;

run Set_II                                      % ��������
run Part_II                                     % �����ļ�����֡�Ͷ˵���
% �˲���ϵ��
b=[0.012280   -0.039508   0.042177   0.000000   -0.042177   0.039508   -0.012280];
a=[1.000000   -5.527146   12.854342   -16.110307   11.479789   -4.410179   0.713507];
xx=filter(b,a,x);                               % ��ͨ�����˲�
yy  = enframe(xx,wlen,inc)';                    % �˲����źŷ�֡

lmin=fix(fs/500);                               % �������ڵ���Сֵ
lmax=fix(fs/60);                                % �������ڵ����ֵ
period=zeros(1,fn);                             % �������ڳ�ʼ��
period=ACF_corr(yy,fn,voiceseg,vosl,lmax,lmin); % ������غ�����ȡ��������
T0=pitfilterm1(period,voiceseg,vosl);           % ƽ������
% ��ͼ
subplot 211, plot(time,x,'k');  title('�����ź�')
axis([0 max(time) -1 1]); grid;  ylabel('��ֵ');
subplot 212; plot(frameTime,T0,'k'); hold on;
xlim([0 max(time)]); title('ƽ����Ļ�������'); 
grid; xlabel('ʱ��/s'); ylabel('������');
for k=1 : vosl
    nx1=voiceseg(k).begin;
    nx2=voiceseg(k).end;
    nxl=voiceseg(k).duration;
    fprintf('%4d   %4d   %4d   %4d\n',k,nx1,nx2,nxl);
    subplot 211
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','linestyle','--');
end
