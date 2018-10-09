% 
% pr8_2_1  
clc; close all; clear all;

run Set_II;                                 % ��������
run Part_II;                                % �����ļ�,��֡�Ͷ˵���
lmin=fix(fs/500);                           % �������ڵ���Сֵ
lmax=fix(fs/60);                            % �������ڵ����ֵ
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
% ��ͼ
subplot 211, plot(time,x,'k');  title('�����ź�')
axis([0 max(time) -1 1]); ylabel('��ֵ');
subplot 212; plot(frameTime,period,'k');
xlim([0 max(time)]); title('��������'); 
grid; xlabel('ʱ��/s'); ylabel('������');
for k=1 : vosl                              % ����л���
    nx1=voiceseg(k).begin;
    nx2=voiceseg(k).end;
    nxl=voiceseg(k).duration;
    fprintf('%4d   %4d   %4d   %4d\n',k,nx1,nx2,nxl);
    subplot 211
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','linestyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','linestyle','--');
end
