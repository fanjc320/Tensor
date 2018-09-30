%
% pr5_3_3
clear all; clc; close all;

filedir=[];                         % ָ���ļ�·��
filename='bluesky3.wav';            % ָ���ļ���
fle=[filedir filename];             
[s,fs]=wavread(fle);                % ���������ļ�
s=s/max(abs(s));                    % ��ֵ��һ��
N=length(s);                        % ������ݳ���
time=(0:N-1)/fs;                    % ���ʱ��̶�
subplot 411; plot(time,s,'k');      % �����������źŵĲ���ͼ
title('�������ź�'); ylabel('��ֵ')
filepath_name='factory1.wav';

SNR=[5 0 -5];                       % ����ȵ�ȡֵ����
for k=1 : 3 
    snr=SNR(k);                     % �趨�����
    [x,noise]=add_noisefile(s,filepath_name,snr,fs); % ������ȹ����������ӵ�������
    subplot(4,1,k+1); plot(time,x,'k'); ylabel('��ֵ');  % ��ͼ
    ylim([-2 2]);
    snr1=SNR_singlech(s,x);         % ��������������е������
    fprintf('k=%4d  snr=%5.1f  snr1=%5.4f\n',k,snr,snr1);
    title(['���������ź� �趨�����=' num2str(snr) 'dB  ����������=' ...
        num2str(round(snr1*1e4)/1e4) 'dB']);
end
xlabel('ʱ��/s')
    
    
    