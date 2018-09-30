%
% pr5_5_2 
clear all; clc; close all;

As=50;Fs=8000; Fs2=Fs/2;               % �����С˥���Ͳ���Ƶ��
fp=75; fs=60;                          % ͨ�����Ƶ��
df=fp-fs;                              % ��ȡ���ɴ�
M0=round((As-7.95)/(14.36*df/Fs))+2;   % ��ʽ(5-5-4)���󴰳�
M=M0+mod(M0+1,2);                      % ��֤����Ϊ����
wp=fp/Fs2*pi; ws=fs/Fs2*pi;            % תΪԲƵ��
wc=(wp+ws)/2;                          % ��ȡ��ֹƵ��
beta=0.5842*(As-21)^0.4+0.07886*(As-21);% ��ʽ(5-5-5)���betaֵ
fprintf('beta=%5.6f\n',beta);          % ��ʾbeta����ֵ
w_kai=(kaiser(M,beta))';               % ����
hd=ideal_lp(pi,M)-ideal_lp(wc,M);      % �������˲�����������Ӧ(��ͨ�˲��������)
b=hd.*w_kai;                           % ����������Ӧ�봰�������
[h,w]=freqz(b,1,4000);                 % ��Ƶ����Ӧ
db=20*log10(abs(h));

filedir=[];                            % ָ���ļ�·��
filename='bluesky3.wav';                % ָ���ļ���
fle=[filedir filename]                  % ����·�����ļ������ַ���
[s,fs]=wavread(fle);                    % ���������ļ�
s=s-mean(s);                            % ����ֱ������
s=s/max(abs(s));                        % ��ֵ��һ��
N=length(s);                            % ����źų���
t=(0:N-1)/fs;                           % ����ʱ��
ns=0.5*cos(2*pi*50*t);                  % �����50Hz��Ƶ�ź�
x=s+ns';                                % �����źź�50Hz��Ƶ�źŵ���
snr1=SNR_singlech(s,x)                  % �������50Hz��Ƶ�źź�������
y=conv(b,x);                            % FIR�����˲������Ϊy
% ��ͼ
figure(1)
plot(w/pi*Fs2,db,'k','linewidth',2); grid;
axis([0 150 -100 10]);
title('��Ƶ��Ӧ����');
xlabel('Ƶ��/Hz');ylabel('��ֵ/dB');
figure(2)                                 
subplot 311; plot(t,s,'k'); 
title('�������źţ����������죬���ơ�')
xlabel('ʱ��/s'); ylabel('��ֵ')
axis([0 max(t) -1.2 1.2]);
subplot 312; plot(t,x,'k'); 
title('��50Hz��Ƶ�źŵ������ź�')
xlabel('ʱ��/s'); ylabel('��ֵ')
axis([0 max(t) -1.2 1.2]);
z=y(fix(M/2)+1:end-fix(M/2));           % ����conv�������˲�������ӳٵ�Ӱ��
snr2=SNR_singlech(s,z)                  % �����˲��������źŵ������
subplot 313; plot(t,z,'k');
title('����50Hz��Ƶ�źź�������ź�')
xlabel('ʱ��/s'); ylabel('��ֵ')
axis([0 max(t) -1.2 1.2]);
