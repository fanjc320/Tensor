% pr3_3_1 
clear all; clc; close all;

% ����melbankm����,��0-0.5�������24��Mel�˲���,�������δ�����
bank=melbankm(24,256,8000,0,0.5,'t');
bank=full(bank);
bank=bank/max(bank(:));              % ��ֵ��һ��

df=8000/256;                         % ����ֱ���
ff=(0:128)*df;                       % Ƶ������̶�
for k=1 : 24                         % ����24��Mel�˲�����Ӧ����
    plot(ff,bank(k,:),'k','linewidth',2); hold on;
end
hold off; grid;
xlabel('Ƶ��/Hz'); ylabel('��Է�ֵ')
title('Mel�˲������Ƶ����Ӧ����')