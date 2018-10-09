%
% pra_3_2   
clear all; clc; close all

filedir=[];                               % ���������ļ���·��
filename='tone4.wav';                     % ���������ļ�������
fle=[filedir filename]                    % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                     % ��ȡ�ļ�
xx=xx-mean(xx);                           % ����ֱ������
xx=xx/max(abs(xx));                       % ��ֵ��һ��
N=length(xx);                             % �źų���
time = (0 : N-1)/fs;                      % ����ʱ��̶�
wlen=320;                                 % ֡��
inc=80;                                   % ֡��
overlap=wlen-inc;                         % ��֡�ص�����  
lmin=floor(fs/500);                       % ��߻���Ƶ��500Hz��Ӧ�Ļ�������
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
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)]);
subplot 311; plot(time,xx,'k');
title('ԭʼ�źŲ���'); axis([0 max(time) -1 1]);
xlabel('(a)'); ylabel('��ֵ');
for k=1 : vosl
        line([frameTime(voiceseg(k).begin) frameTime(voiceseg(k).begin)],...
        [-1 1],'color','k');
        line([frameTime(voiceseg(k).end) frameTime(voiceseg(k).end)],...
        [-1 1],'color','k','linestyle','--');
end
subplot 312; plot(frameTime,Ef,'k'); hold on
title('���ر�'); axis([0 max(time) 0 max(Ef)]);
line([0 max(frameTime)],[Thr1 Thr1],'color','k','linestyle','--');
plot(frameTime,Thr2,'k','linewidth',2);
xlabel('(b)'); ylabel('��ֵ');
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
y=enframe(x,wlen,inc)';                   % �ڶ��η�֡

[Extseg,Dlong]=Extoam(voiceseg,vseg,vosl,vsl,Bth);   % ����������������쳤��
T1=ACF_corrbpa(y,fn,vseg,vsl,lmax,lmin,ThrC);     % ��Ԫ��������л������
figure(1)
subplot 313; plot(frameTime,T1,'k'); 
title('�����������'); 
xlabel(['ʱ��/s' 10 '(c)']); ylabel('������');
% ��������ǰ�������������������        
T0=zeros(1,fn);                           % ��ʼ��
F0=zeros(1,fn);
k=5;                                      % ȡ��5��Ԫ������
    ix1=vseg(k).begin;                    % ��k��Ԫ�����忪ʼλ��
    ix2=vseg(k).end;                      % ��k��Ԫ���������λ��
    in1=Extseg(k).begin;                  % ��k��Ԫ������ǰ�����쿪ʼλ��
    in2=Extseg(k).end;                    % ��k��Ԫ����������������λ��
    ixl1=Dlong(k,1);                      % ǰ�����쳤��
    ixl2=Dlong(k,2);                      % �������쳤��
XL=ixl1;
fprintf('k=%4d   XL=%4d\n',k,XL);         % ��ʾ��5��Ԫ�����弰���쳤��
sign=-1;                                  % ǰ������������
TT1=round(T1(ix1));                       % Ԫ���������һ����Ļ�������
ixb=ix1;
    
[Pm,vsegch,vsegchlong]=Ext_corrshtp_test1(y,sign,TT1,XL,ixb,lmax,lmin,ThrC);

