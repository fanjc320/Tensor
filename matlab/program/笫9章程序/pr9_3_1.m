%
%  pr9_3_1  
clear all; clc; close all;

fle='snn27.wav';                            % ָ���ļ���
[x,fs]=wavread(fle);                        % ����һ֡�����ź� 
u=filter([1 -.99],1,x);                     % Ԥ����
wlen=length(u);                             % ֡��
p=12;                                       % LPC����
a=lpc(u,p);                                 % ���LPCϵ��
U=lpcar2pf(a,255);                          % ��LPCϵ�����Ƶ������
freq=(0:256)*fs/512;                        % Ƶ�ʿ̶�
df=fs/512;                                  % Ƶ�ʷֱ���
U_log=10*log10(U);                          % �����׷ֱ�ֵ
subplot 211; plot(u,'k');                   % ��ͼ
axis([0 wlen -0.5 0.5]);
title('Ԥ���ز���');
xlabel('������'); ylabel('��ֵ')
subplot 212; plot(freq,U,'k');
title('�������ݺ�������������');
xlabel('Ƶ��/Hz'); ylabel('��ֵ');

[Loc,Val]=findpeaks(U);                     % ��U��Ѱ�ҷ�ֵ
ll=length(Loc);                             % �м�����ֵ
for k=1 : ll
    m=Loc(k);                               % ����m-1,m��m+1
    m1=m-1; m2=m+1;
    p=Val(k);                               % ����P(m-1),P(m)��P(m+1)
    p1=U(m1); p2=U(m2);
    aa=(p1+p2)/2-p;                         % ��ʽ(9-3-4)����
    bb=(p2-p1)/2;
    cc=p;
    dm=-bb/2/aa;                            % ��ʽ(9-3-6)����
    pp=-bb*bb/4/aa+cc;                      % ��ʽ(9-3-8)����
    m_new=m+dm;
    bf=-sqrt(bb*bb-4*aa*(cc-pp/2))/aa;      % ��ʽ(9-3-13)����
    F(k)=(m_new-1)*df;                      % ��ʽ(9-3-7)����
    Bw(k)=bf*df;                            % ��ʽ(9-3-14)����
    line([F(k) F(k)],[0 pp],'color','k','linestyle','-.');
end
fprintf('F =%5.2f   %5.2f   %5.2f   %5.2f\n',F)
fprintf('Bw=%5.2f   %5.2f   %5.2f   %5.2f\n',Bw)
