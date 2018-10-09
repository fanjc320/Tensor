%
% pr3_5_1 
clear all; clc; close all;

fs=5000;                       % ����Ƶ��
N=500;                         % ������
n=1:N;
t1=(n-1)/fs;                   % ����ʱ��
x1=sin(2*pi*50*t1);            % ������1�������ź�
x2=(1/3)*sin(2*pi*150*t1);     % ������2�������ź�
z=x1+x2;                       % �������źŵ���
imp=emd(z);                    % �Ե����źŽ���EMD�ֽ�
[m,n]=size(imp);               % ��ȡEMD�ֽ�ɼ�������
% ��ͼ
subplot(m+1,1,1);              % �������ź�
plot(t1,z,'k');title('ԭʼ�ź�'); ylabel('��ֵ')
subplot 312;                   % ����1�������ź�
line(t1,x2,'color',[.6 .6 .6],'linewidth',5); hold on
subplot 313;                   % ����2�������ź�
line(t1,x1,'color',[.6 .6 .6],'linewidth',5); hold on
for i=1:m
    subplot(m+1,1,i+1);        % ��EMD�ֽ����ź�
    plot(t1,imp(i,:),'k','linewidth',1.5); ylabel('��ֵ')
    title(['imf' num2str(i)]);
end
xlabel('ʱ��/s');

