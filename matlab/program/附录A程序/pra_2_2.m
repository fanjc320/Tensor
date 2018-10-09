%
% pra_2_2    
clear all; clc; close all

filedir=[];                               % ���������ļ���·��
filename='deepstep.wav';                  % ���������ļ�������
fle=[filedir filename]                    % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                     % ��ȡ�ļ�
xx=xx-mean(xx);                           % ����ֱ������
xx=xx/max(abs(xx));                       % ��ֵ��һ��
N=length(xx);                             % �źų���
time = (0 : N-1)/fs;                      % ����ʱ��̶�
wlen=320;                                 % ֡��
inc=80;                                   % ֡��
overlap=wlen-inc;                         % ��֡�ص�����  
lmin=floor(fs/300);                       % ��߻���Ƶ��500Hz��Ӧ�Ļ�������
lmax=floor(fs/60);                        % ��߻���Ƶ��60Hz��Ӧ�Ļ������� 

yy=enframe(xx,wlen,inc)';                 % ��һ�η�֡
fn=size(yy,2);                            % ȡ����֡��
frameTime = frame2time(fn, wlen, inc, fs);% ����ÿһ֡��Ӧ��ʱ��
Thr1=0.1;                                 % ���ö˵�����ֵ
r2=0.5;                                   % ����Ԫ��������ı�������
ThrC=[10 15];                             % �������ڻ������ڼ����ֵ
% ���ڻ������Ķ˵����Ԫ��������
[voiceseg,vosl,vseg,vsl,Thr2,Bth,SF,Ef]=pitch_vads(yy,fn,Thr1,r2,6,5);
figure(1)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)-120]);
subplot 211; plot(time,xx,'k');
title('ԭʼ�źŲ���ͼ'); axis([0 max(time) -1 1]);
xlabel('(a)'); ylabel('��ֵ');
for k=1 : vosl
        line([frameTime(voiceseg(k).begin) frameTime(voiceseg(k).begin)],...
        [-1 1],'color','k');
        line([frameTime(voiceseg(k).end) frameTime(voiceseg(k).end)],...
        [-1 1],'color','k','linestyle','--');
end
subplot 212; plot(frameTime,Ef,'k'); hold on
title('���ر�ͼ'); axis([0 max(time) 0 max(Ef)]);
xlabel(['ʱ��/s' 10 '(b)']); ylabel('��ֵ');
line([0 max(frameTime)],[Thr1 Thr1],'color','k','linestyle','--');
plot(frameTime,Thr2,'k','linewidth',2);
for k=1 : vsl
        line([frameTime(vseg(k).begin) frameTime(vseg(k).begin)],...
        [0 max(Ef)],'color','k','linestyle','-.');
        line([frameTime(vseg(k).end) frameTime(vseg(k).end)],...
        [0 max(Ef)],'color','k','linestyle','-.');
end

% 60��500Hz�Ĵ�ͨ�˲���ϵ��
b=[0.012280   -0.039508   0.042177   0.000000   -0.042177   0.039508   -0.012280];
a=[1.000000   -5.527146   12.854342   -16.110307   11.479789   -4.410179   0.713507];
x=filter(b,a,xx);                         % �����˲�
x=x/max(abs(x));                          % ��ֵ��һ��
y=enframe(x,wlen,inc)';                   % �ʶ��η�֡

m=3;                                      % ȡ��3��Ԫ������
fprintf('m=%4d\n',m);                     % ��ʾ
T1=corrbp_test11(y,fn,vseg,vsl,lmax,lmin,ThrC,m);
