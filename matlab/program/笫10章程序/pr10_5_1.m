%
%  pr10_5_1 
clear all; clc; close all;

F0 = [700 900 2600];
Bw = [130 70 160];
fs=8000;

[An,Bn]=formant2filter4(F0,Bw,fs);    % ���ú�����ȡ�˲���ϵ��
for k=1 : 4                           % ���ĸ����״�ͨ�˲�����Ƶ������
    A=An(:,k);                        % ȡ���˲���ϵ��
    B=Bn(k);
    fprintf('B=%5.6f   A=%5.6f   %5.6f   %5.6f\n',B,A);
    [H(:,k),w]=freqz(B,A);            % �����Ӧ����
end
freq=w/pi*fs/2;                       % Ƶ����̶�
% ��ͼ
plot(freq,abs(H(:,1)),'k',freq,abs(H(:,2)),'k',freq,abs(H(:,3)),'k',freq,abs(H(:,4)),'k');
axis([0 4000 0 1.05]); grid;
line([F0(1) F0(1)],[0 1.05],'color','k','linestyle','-.');
line([F0(2) F0(2)],[0 1.05],'color','k','linestyle','-.');
line([F0(3) F0(3)],[0 1.05],'color','k','linestyle','-.');
line([3500 3500],[0 1.05],'color','k','linestyle','-.');
title('����������һ���̶�Ƶ�ʵĶ��״�ͨ�˲�����Ӧ����')    
ylabel('��ֵ'); xlabel('Ƶ��/Hz')
    