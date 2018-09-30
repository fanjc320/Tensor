%
% pr4_2_1 
clear all; clc; close all;

filedir=[];                             % ���������ļ���·��
filename='aa.wav';                      % ���������ļ�������
fle=[filedir filename]                  % ����·�����ļ������ַ���
[x,fs]=wavread(fle);                    % ������������
L=240;                                  % ֡��
y=x(8001:8000+L);                       % ȡһ֡����
p=12;                                   % LPC�Ľ���
ar=lpc(y,p);                            % ����Ԥ��任
Y=lpcar2ff(ar,255);                     % ��LPC��Ƶ��ֵ
est_x=filter([0 -ar(2:end)],1,y);       % ��LPC��Ԥ�����ֵ
err=y-est_x;                            % ���Ԥ�����
fprintf('LPC:\n');
fprintf('%5.4f   %5.4f   %5.4f   %5.4f   %5.4f   %5.4f   %5.4f\n',ar);
fprintf('\n');
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-200,pos(3),pos(4)+150]);
subplot 311; plot(x,'k'); axis tight;
title('Ԫ��/a/����'); ylabel('��ֵ')
subplot 323; plot(y,'k'); xlim([0 L]); 
title('һ֡����'); ylabel('��ֵ')
subplot 324; plot(est_x,'k'); xlim([0 L]); 
title('Ԥ��ֵ'); ylabel('��ֵ')
subplot 325; plot(abs(Y),'k'); xlim([0 L]); 
title('LPCƵ��'); ylabel('��ֵ'); xlabel('����')
subplot 326; plot(err,'k'); xlim([0 L]); 
title('Ԥ�����'); ylabel('��ֵ'); xlabel('����')









