% 
% pr8_5_1 
clc; close all; clear all;

run Set_II;                                 % ��������
run Part_II;                                % �����ļ�,��֡�Ͷ˵���

lmin=fix(fs/500);                           % �������ڵ���Сֵ
lmax=fix(fs/60);                            % �������ڵ����ֵ
period=zeros(1,fn);                         % �������ڳ�ʼ��
p=12;                                       % ��������Ԥ�����
for k=1:fn 
    if SF(k)==1                             % �Ƿ����л�֡��
        u=y(:,k).*hamming(wlen);            % ȡ��һ֡���ݼӴ�����
        ar = lpc(u,p);                      % ����LPCϵ��
        z = filter([0 -ar(2:end)],1,u);     % һ֡����LPC���˲����
        E = u - z;                          % Ԥ�����
        xx=fft(E);                          % FFT
        a=2*log(abs(xx)+eps);               % ȡģֵ�Ͷ���
        b=ifft(a);                          % ��ȡ���� 
        [R(k),Lc(k)]=max(b(lmin:lmax));     % ��Pmin��Pmax����Ѱ�����ֵ
        period(k)=Lc(k)+lmin-1;             % ������������
    end
end
T1=pitfilterm1(period,voiceseg,vosl);       % ��������ƽ������

% ��ͼ
subplot 211, plot(time,x,'k');  title('�����ź�')
axis([0 max(time) -1 1]); grid;  ylabel('��ֵ'); xlabel('ʱ��/s');
subplot 212; hold on
line(frameTime,period,'color',[.6 .6 .6],'linewidth',2); 
axis([0 max(time) 0 150]); title('��������'); 
ylabel('������'); xlabel('ʱ��/s'); grid; 
plot(frameTime,T1,'k'); hold off
legend('������ֵ','ƽ����ֵ'); box on;
