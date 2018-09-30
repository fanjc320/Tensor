%
% pr3_4_1 
clear all; clc; close all;

[x,fs]=wavread('awav.wav');             % ������������
N=length(x);                            % �źų���
x=x-mean(x);                            % ����ֱ������
J=2;                                    % ��С���任����ΪJ
[C,L] = wavedec(x,J,'db1');             % ��ʱ�����н���һά��ֱ�ֽ�
CaLen=N/2.^J;                           % ���ƽ��Ʋ��ֵ�ϵ������
Ca=C(1:CaLen);                          % ȡ���Ʋ��ֵ�ϵ��
Ca=(Ca-min(Ca))./(max(Ca)-min(Ca));     % �Խ��Ʋ���ϵ������������
for i=1:CaLen                           % �Խ��Ʋ���ϵ��������
    if(Ca(i)<0.8), Ca(i)=0; end
end
[K,V]=findpeaks(Ca,[],6);               % Ѱ�ҷ�ֵλ�ú���ֵ
lk=length(K);
if lk~=0
    for i=2 : lk
        dis(i-1)=K(i)-K(i-1)+1;         % Ѱ�ҷ�ֵ֮��ļ��
    end
    distance=mean(dis);                 % ȡ�����ƽ��ֵ
    pit=fs/2.^J/distance                % ������һ֡�Ļ���Ƶ��
else
    pit=0;
end
% ��ͼ
subplot 211; plot(x,'k'); 
title('һ֡�����ź�')
subplot 212; plot(Ca,'k');
title('��С���ֽ�õ��Ľ���ϵ������������ķ�ֵͼ')

