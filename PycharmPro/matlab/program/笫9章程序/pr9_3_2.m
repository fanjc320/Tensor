%
% pr9_3_2  
clear all; clc; close all;

fle='snn27.wav';                            % ָ���ļ���
[xx,fs]=wavread(fle);                       % ����һ֡�����ź�
u=filter([1 -.99],1,xx);                    % Ԥ����
wlen=length(u);                             % ֡��
p=12;                                       % LPC����
a=lpc(u,p);                                 % ���LPCϵ��
U=lpcar2pf(a,255);                          % ��LPCϵ���������������
freq=(0:256)*fs/512;                        % Ƶ�ʿ̶�
df=fs/512;                                  % Ƶ�ʷֱ���
U_log=10*log10(U);                          % �����׷ֱ�ֵ
subplot 211; plot(u,'k');                   % ��ͼ
axis([0 wlen -0.5 0.5]);
title('Ԥ���ز���');
xlabel('������'); ylabel('��ֵ')
subplot 212; plot(freq,U_log,'k');
title('�������ݺ�������������');
xlabel('Ƶ��/Hz'); ylabel('��ֵ/dB');

n_frmnt=4;                                  % ȡ�ĸ������
const=fs/(2*pi);                            % ����  
rts=roots(a);                               % ���
k=1;                                        % ��ʼ��
yf = [];
bandw=[];
for i=1:length(a)-1                     
    re=real(rts(i));                        % ȡ��֮ʵ��
    im=imag(rts(i));                        % ȡ��֮�鲿
    formn=const*atan2(im,re);               % ��(9-3-17)���㹲���Ƶ��
    bw=-2*const*log(abs(rts(i)));           % ��(9-3-18)�������
    
    if formn>150 & bw <700 & formn<fs/2     % �����������ܳɹ����ʹ���
        yf(k)=formn;
        bandw(k)=bw;
        k=k+1;
    end
end

[y, ind]=sort(yf);                          % ����
bw=bandw(ind);
F = [NaN NaN NaN NaN];                      % ��ʼ��
Bw = [NaN NaN NaN NaN];
F(1:min(n_frmnt,length(y))) = y(1:min(n_frmnt,length(y)));   % �������ĸ�
Bw(1:min(n_frmnt,length(y))) = bw(1:min(n_frmnt,length(y))); % �������ĸ�
F0 = F(:);                                  % �������
Bw = Bw(:);
p1=length(F0);                              % �ڹ���崦����
for k=1 : p1
    m=floor(F0(k)/df);
    P(k)=U_log(m+1);
    line([F0(k) F0(k)],[-10 P(k)],'color','k','linestyle','-.');
end
fprintf('F0=%5.2f   %5.2f   %5.2f   %5.2f\n',F0);
fprintf('Bw=%5.2f   %5.2f   %5.2f   %5.2f\n',Bw);

