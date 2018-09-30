%
% pr4_3_1 
clear all; clc; close all;

filedir=[];                             % ���������ļ���·��
filename='aa.wav';                      % ���������ļ�������
fle=[filedir filename];                 % ����·�����ļ������ַ���
[x,fs]=wavread(fle);                    % ������������
L=240;                                  % ֡��
p=12;                                   % LPC�Ľ���
y=x(8001:8240+p);                       % ȡһ֡����

[EL,alphal,GL,k]=latticem(y,L,p);       % ����Ԥ�ⷨ
ar=alphal(:,p);
a1=lpc(y,p);                            % ��ͨԤ�ⷨ
Y=lpcar2pf(a1,255);                     % ��a1ת�ɹ�����
Y1=lpcar2pf([1; -ar],255);              % ��arת�ɹ�����
fprintf('AR1ϵ��(����Ԥ�ⷨ):\n');
fprintf('%5.4f   %5.4f   %5.4f   %5.4f   %5.4f   %5.4f\n',-ar);
fprintf('AR2ϵ��(��ͨԤ�ⷨ):\n');
fprintf('%5.4f   %5.4f   %5.4f   %5.4f   %5.4f   %5.4f\n',a1(2:p+1));
% ��ͼ
m=1:257;
freq=(m-1)*fs/512;
plot(freq,10*log10(Y),'k'); grid;
line(freq,10*log10(Y1),'color',[.6 .6 .6],'linewidth',2);
legend('��ͨԤ�ⷨ','����Ԥ�ⷨ'); ylabel('��ֵ/dB');
title('��ͨԤ�ⷨ�͸���Ԥ�ⷨ��������Ӧ�ıȽ�'); xlabel('Ƶ��/Hz');
 


