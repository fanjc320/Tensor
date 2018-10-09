%
% pr10_7_1 
clear all; clc; close all;

filedir=[];                                % ���������ļ���·��
filename='bluesky3.wav';                   % ���������ļ�������
fle=[filedir filename]                     % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                      % ��ȡ�ļ�
xx=xx-mean(xx);                            % ȥ��ֱ������
x=xx/max(abs(xx));                         % ��һ��
N=length(x);                               % ���ݳ���
time=(0:N-1)/fs;                           % �����Ӧ��ʱ������

[LowPass] = LowPassFilter(x, fs, 500);     % ��ͨ�˲�
p = PitchEstimation(LowPass, fs);		   % �������Ƶ��
[u, v] = UVSplit(p);                       % ����л��κ��޻�����Ϣ
lu=size(u,1); lv=size(v,1);                % ��ʼ��

pm = [];
ca = [];
for i = 1 : length(v(:,1))
    range = (v(i, 1) : v(i, 2));           % ȡһ���л�����Ϣ
    in = x(range);                         % ȡ�л�������
% ��һ���л���Ѱ�һ��������ע
    [marks, cans] = VoicedSegmentMarking(in, p(range), fs);

    pm = [pm  (marks + range(1))];         % ������������עλ��
    ca = [ca;  (cans + range(1))];         % ������������ע��ѡ����
end
% ��ͼ
figure(1)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-150,pos(3),pos(4)+100]);
subplot 211; plot(time,x,'k'); axis([0 max(time) -1 1.2]);
for k=1 : lv
    line([time(v(k,1)) time(v(k,1))],[-1 1.2 ],'color','k','linestyle','-')
    line([time(v(k,2)) time(v(k,2))],[-1 1.2 ],'color','k','linestyle','--')
end
title('�����źŲ��κͶ˵���');
xlabel(['ʱ��/s' 10 '(a)']); ylabel('��ֵ');
%figure(2)
%pos = get(gcf,'Position');
%set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-160)]);
subplot 212; plot(x,'k'); axis([0 N -1 0.8]);
line([0 N],[0 0],'color','k')
lpm=length(pm);
for k=1 : lpm
    line([pm(k) pm(k)],[0 0.8],'color','k','linestyle','-.')
end
xlim([3000 4000]);
title('���������źŲ��κ���Ӧ���������ע');
xlabel(['����' 10 '(b)']); ylabel('��ֵ');


