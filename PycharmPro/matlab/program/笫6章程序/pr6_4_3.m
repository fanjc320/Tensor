%
% pr6_4_3 
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��

% BARK�Ӵ�������
Fk=[50 20 100; 150 100 200; 250 200 300; 350 300 400; 450 400 510; 570 510 630; 700 630 770;...
    840 770 920; 1000 920 1080; 1170 1080 1270; 1370 1270 1480; 1600 1480 1720; 1850 1720 2000;...
    2150 2000 2320; 2500 2320 2700; 2900 2700 3150; 3400 3150 3700; 4000 3700 4400;...
    4800 4400 5300; 5800 5300 6400; 7000 6400 7700; 8500 7700 9500; 10500 9500 12000;... 
    13500 12000 15500; 18775 15500 22050];

% ��ֵ
fs2=fix(fs/2); 
y=y';
for i=1:fn
    sourfft(i,:)=fft(y(i,:),wlen);                    % FFT�任                    
    sourfft1(i,:)=abs(sourfft(i,1:wlen/2));           % ȡ��Ƶ�ʷ�ֵ
    sourre(i,:)=resample(sourfft1(i,:),fs2,wlen/2);   % �����ڲ�
end
% ����BARK�˲�������
for k=1 : 25
    if Fk(k,3)>fs2
        break
    end
end
num=k-1;

for i=1 : fn
    Sr=sourre(i,:);                     % ȡһ֡��ֵ
    for k=1 : num   
        m1=Fk(k,2); m2=Fk(k,3);         % ���BARK�˲��������½�ֹƵ��
        Srt=Sr(m1:m2);                  % ȡ����Ӧ������
        Dst(k)=var(Srt);                % ����k��BARK�˲����еķ���ֵ
    end
    Dvar(i)=mean(Dst);                  % �����BARK�˲����з���ֵ��ƽ��ֵ
end
Dvarm=multimidfilter(Dvar,10);          % ƽ������
dth=mean(Dvarm(1:(NIS)));               % ��ֵ����
T1=1.5*dth;
T2=3*dth;
[voiceseg,vsl,SF,NF]=vad_param1D(Dvarm,T1,T2);    % BARK�Ӵ���Ƶ������˫���޵Ķ˵���
% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title('������������(�����10dB)');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Dvar,'k');
title('��ʱBARK�Ӵ������Ƶ������ֵ'); axis([0 max(time) 0 1.2*max(Dvar)]);
xlabel('ʱ��/s'); ylabel('��ֵ'); 
line([0,frameTime(fn)], [T1 T1], 'color','k','LineStyle','--');
line([0,frameTime(fn)], [T2 T2], 'color','k','LineStyle','-');
for k=1 : vsl                           % ��������˵�
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    subplot 311; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','LineStyle','--');
    subplot 313; 
    line([frameTime(nx1) frameTime(nx1)],[0 1.2*max(Dvar)],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2*max(Dvar)],'color','k','LineStyle','--');
end
