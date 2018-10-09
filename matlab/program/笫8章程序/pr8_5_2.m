% 
% pr8_5_2 
clc; close all; clear all;

run Set_II;                               % ��������
run Part_II;                              % �����ļ�,��֡�Ͷ˵���
% ���ִ�ͨ�˲��������
Rp=1; Rs=50; fs2=fs/2;                    % ͨ������1dB,���˥��50dB 
Wp=[60 500]/fs2;                          % ͨ��Ϊ60��500Hz
Ws=[20 1000]/fs2;                         % ���Ϊ20��1000Hz
[n,Wn]=ellipord(Wp,Ws,Rp,Rs);             % ѡ����Բ�˲���
[b,a]=ellip(n,Rp,Rs,Wn);                  % ����˲���ϵ��
x1=filter(b,a,x);                         % ��ͨ�˲�
x1=x1/max(abs(x1));                       % ��ֵ��һ��
x2=resample(x1,1,4);                      % ��4:1��������

lmin=fix(fs/500);                         % �������ڵ���Сֵ
lmax=fix(fs/60);                          % �������ڵ����ֵ
period=zeros(1,fn);                       % �������ڳ�ʼ��
wind=hanning(wlen/4);                     % ������
y2=enframe(x2,wind,inc/4)';               % ��һ�η�֡
p=4;                                      % LPC����Ϊ4
for i=1 : vosl                            % ֻ���л������ݴ���
    ixb=voiceseg(i).begin;                % ȡһ���л���
    ixe=voiceseg(i).end;                  % ��ȡ���л��ο�ʼ�ͽ���λ�ü�֡��
    ixd=ixe-ixb+1;
    for k=1 : ixd                         % �Ըö��л������ݴ���
        u=y2(:,k+ixb-1);                  % ȡ��һ֡����
        ar = lpc(u,p);                    % ����LPCϵ��
        z = filter([0 -ar(2:end)],1,u);   % һ֡����LPC���˲����
        E = u - z;                        % Ԥ�����
        ru1= xcorr(E, 'coeff');           % �����һ������غ���
        ru1 = ru1(wlen/4:end);            % ȡ�ӳ���Ϊ��ֵ�Ĳ���
        ru=resample(ru1,4,1);             % ��1:4��������
        [tmax,tloc]=max(ru(lmin:lmax));   % ��Pmin��Pmax��Χ��Ѱ�����ֵ
        period(k+ixb-1)=lmin+tloc-1;      % ������Ӧ���ֵ���ӳ���
    end
end
T1=pitfilterm1(period,voiceseg,vosl);     % ��������ƽ������
% ��ͼ
subplot 211, plot(time,x,'k');  title('�����ź�')
axis([0 max(time) -1 1]); grid;  ylabel('��ֵ'); xlabel('ʱ��/s');
subplot 212; hold on
line(frameTime,period,'color',[.6 .6 .6],'linewidth',2); 
xlim([0 max(time)]); title('��������'); grid; 
ylim([0 150]);  ylabel('������'); xlabel('ʱ��/s'); 
plot(frameTime,T1,'k'); hold off
legend('������ֵ','ƽ����ֵ'); box on

