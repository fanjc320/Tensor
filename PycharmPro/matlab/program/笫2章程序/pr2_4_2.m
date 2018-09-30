%
% pr2_4_2 
clear all; clc; close all;

filedir=[];                                    % ����·��
filename='bluesky3.wav';                       % �����ļ���
fle=[filedir filename];                        % ����������·�����ļ���
[wavin0,fs,nbits]=wavread(fle);                % ���������ļ�
nwind=240; noverlap=160; inc=nwind-noverlap;   % ����֡��Ϊ240���ص�Ϊ160��֡��Ϊ80
w_nwind=hanning(200); w_noverlap=195;          % ���öγ�Ϊ200�����ص�Ϊ195
nfft=200;                                      % FFT����Ϊ200
% ��ÿ֡��pwelch_2�����ʱ�������ܶ�
[Pxx] = pwelch_2(wavin0, nwind, noverlap, w_nwind, w_noverlap, nfft);
frameNum=size(Pxx,2);                          % ȡ��֡��
frameTime=frame2time(frameNum,nfft,inc,fs);    % ����ÿ֡��Ӧ��ʱ��
freq=(0:nfft/2)*fs/nfft;                       % ����Ƶ�ʿ̶�
% ��ͼ
imagesc(frameTime,freq,Pxx); axis xy           
ylabel('Ƶ��/Hz');
xlabel('ʱ��/s');
title('��ʱ�������ܶȺ���')
m = 256; LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0]; Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors));
