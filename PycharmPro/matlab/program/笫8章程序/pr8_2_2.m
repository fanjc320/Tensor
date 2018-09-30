% 
% pr8_2_2 
clc; close all; clear all;

run Set_II;                                 % ��������
run Part_II;                                % �����ļ�,��֡�Ͷ˵���

lmin=fix(fs/500);                           % ����������ȡ����Сֵ
lmax=fix(fs/60);                            % ����������ȡ�����ֵ
period=zeros(1,fn);                         % �������ڳ�ʼ��
for k=1:fn 
    if SF(k)==1                             % �Ƿ����л�֡��
        y1=y(:,k).*hamming(wlen);           % ȡ��һ֡���ݼӴ�����
        xx=fft(y1);                         % FFT
        a=2*log(abs(xx)+eps);               % ȡģֵ�Ͷ���
        b=ifft(a);                          % ��ȡ���� 
        [R(k),Lc(k)]=max(b(lmin:lmax));     % ��lmin��lmax������Ѱ�����ֵ
        period(k)=Lc(k)+lmin-1;             % ������������
    end
end

T0=zeros(1,fn);                             % ��ʼ��T0��F0
F0=zeros(1,fn);
T0=pitfilterm1(period,voiceseg,vosl);       % ��T0����ƽ�����������������T0
Tindex=find(T0~=0);
F0(Tindex)=fs./T0(Tindex);                  % �������Ƶ��F0
% ��ͼ
subplot 311, plot(time,x,'k');  title('�����ź�')
axis([0 max(time) -1 1]); grid;  ylabel('��ֵ');
subplot 312; line(frameTime,period,'color',[.6 .6 .6],'linewidth',3);
xlim([0 max(time)]); title('��������'); hold on;
ylim([0 150]); ylabel('������'); grid; 
for k=1 : vosl
    nx1=voiceseg(k).begin;
    nx2=voiceseg(k).end;
    nxl=voiceseg(k).duration;
    fprintf('%4d   %4d   %4d   %4d\n',k,nx1,nx2,nxl);
    subplot 311
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','linestyle','--');
end
subplot 312; plot(frameTime,T0,'k'); hold off
legend('ƽ��ǰ','ƽ����');
subplot 313; plot(frameTime,F0,'k'); 
grid; ylim([0 450]);
title('����Ƶ��'); xlabel('ʱ��/s'); ylabel('Ƶ��/Hz');


