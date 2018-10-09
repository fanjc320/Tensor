%
% pr4_5_1 
clear all; clc; close all;

filedir=[];                   % ���������ļ���·��
filename='aa.wav';            % ���������ļ�������
fle=[filedir filename]        % ����·�����ļ������ַ���
[x,fs]=wavread(fle);          % ��ȡ�����ļ�aa.wav
x=x/max(abs(x));              % ��ֵ��һ��
time=(0:length(x)-1)/fs;      % �����Ӧ��ʱ������
N=200;                        % �趨֡��
M=80;                         % �趨֡�Ƶĳ���  
xn=enframe(x,N,M)';           % ���ղ������з�֡
s=xn(:,100);                  % ȡ��֡�����100֡���з���

p=12;                         % ��Ԥ��״�
num=257;                      % �趨Ƶ�׵ĵ���
a2 =lpc(s,p);                 % �����źŴ��������еĺ���lpc��Ԥ��ϵ��a2
Hw=lpcar2ff(a2,num-2);        % ����lpcar2ff������Ԥ��ϵ��a���LP��Hw
Hw_abs=abs(Hw);               % ȡHw��ģֵ
lsf=ar2lsf(a2);               % ����ar2lsf������arϵ��ת����lsf����
P_w=lsf(1:2:end);             % ��lsf���P��Q��Ӧ��Ƶ�ʣ���λΪ����
Q_w=lsf(2:2:end);
P_f=P_w*fs/2/pi;              % ת���ɵ�λΪHz
Q_f=Q_w*fs/2/pi;
figure(1)
pos = get(gcf,'Position');    % ���û�ͼ��
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-180)]);
plot(time,x,'k');             % �����źŵĲ���
title('�����ź�aa.wav�Ĳ���ͼ ');
xlabel('ʱ��/s'); ylabel('��ֵ')    
xlim([0 max(time)]);
figure(2)
subplot 211; plot(s,'k');     % ����һ֡�źŵĲ���
title('�����ź�aa.wav��һ֡����ͼ ');
xlabel(['����ֵ' 10 '(a)']); ylabel('��ֵ')    
freq=(0:num-1)*fs/512;          % ����Ƶ���Ƶ������
m=1:num;
K=length(Q_w);

ar=lsf2ar(lsf);               % ����lsf2ar������lsfת����Ԥ��ϵ��ar 
Hw1=lpcar2ff(ar,num-2);       % ����lpcar2ff����,��Ԥ��ϵ��ar���LP��Hw1
Hw1_abs=abs(Hw1);
subplot 212;                  % ��Hw��Hw1����һ��ͼ��
hline1 = plot(freq,20*log10(Hw_abs(m)/max(Hw_abs)),'k','LineWidth',2); 
hline2 = line(freq+1,20*log10(Hw1_abs(m)/max(Hw1_abs)),...
    'LineWidth',5,'Color',[.6 .6 .6]);
set(gca,'Children',[hline1 hline2]);
axis([0 fs/2 -35 5]);
title('�����źŵ�LPC�׺����׶Ի�ԭLPC��Ƶ�� ');
xlabel(['Ƶ��/Hz' 10 '(b)']); ylabel('��ֵ')    
for k=1 : K                   % ��P_f��Q_fҲ��ͼ���ô�ֱ�߱��
    line([Q_f(k) Q_f(k)],[-35 5],'color','k','Linestyle','--');
    line([P_f(k) P_f(k)],[-35 5],'color','k','Linestyle','-');
end
for k= 1 : p+1                % ��ʾԤ��ϵ��a2��ar�������߽��бȽ�
    fprintf('%4d   %5.6f   %5.6f\n',k,a2(k),ar(k));
end
