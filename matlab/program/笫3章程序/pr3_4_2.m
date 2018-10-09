%
% pr3_4_2 
clear all; clc; close all;

filedir=[];                             % ���������ļ�·��
filename='aa.wav';                      % �����ļ���
fle=[filedir filename]                  % ����·�����ļ������ַ���
[xx, fs, nbits]=wavread(fle);           % ���������ļ�
x=xx-mean(xx);                          % ����ֱ������
x=x/max(abs(x));                        % ��ֵ��һ��
N=length(x);                            % ȡ�źų���
T=wpdec(x,5,'db2');                     % ��ʱ�����н���һάС�����ֽ�
% ��ָ���Ľ��,��ʱ�����зֽ��һάС����ϵ���ع�
y(1,:)=wprcoef(T,[5 0]);
y(2,:)=wprcoef(T,[5 1]);
y(3,:)=wprcoef(T,[5 2]);
y(4,:)=wprcoef(T,[5 3]);
y(5,:)=wprcoef(T,[5 4]);
y(6,:)=wprcoef(T,[5 5]);
y(7,:)=wprcoef(T,[5 6]);
y(8,:)=wprcoef(T,[5 7]);

y(9,:)=wprcoef(T,[4 4]);
y(10,:)=wprcoef(T,[4 5]);
y(11,:)=wprcoef(T,[5 11]);
y(12,:)=wprcoef(T,[5 12]);
y(13,:)=wprcoef(T,[4 7]);

y(14,:)=wprcoef(T,[3 4]);
y(15,:)=wprcoef(T,[3 5]);
y(16,:)=wprcoef(T,[3 6]);
y(17,:)=wprcoef(T,[3 7]);
% ��ͼ
subplot 511; plot(x,'k');
ylabel('/a/'); axis tight
for k=1 : 4
    subplot(5,2,k*2+1); plot(y((k-1)*2+1,:),'k');
    ylabel(['y' num2str((k-1)*2+1)]); axis tight;
    subplot(5,2,(k+1)*2); plot(y(k*2,:),'k');
    ylabel(['y' num2str(k*2)]); axis tight;
end
figure    
for k=1 : 4
    subplot(5,2,(k-1)*2+1); plot(y((k-1)*2+9,:),'k');
    ylabel(['y' num2str((k-1)*2+9)]); axis tight;
    subplot(5,2,k*2); plot(y(k*2+8,:),'k');
    ylabel(['y' num2str(k*2+8)]); axis tight;
end
subplot(5,2,9); plot(y(17,:),'k');
ylabel('y17'); axis tight




