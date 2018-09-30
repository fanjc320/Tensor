%
% pr6_4_4 
clear all; clc; close all;

run Set_I                                % ��������
run PART_I                               % �������ݣ���֡��׼��
h=waitbar(0,'Running...');               % �������г��������ͼ,��ʼ��
set(h,'name','�˵��� - 0%');            % ���ñ�ͼ������"�˵���"
for i=1 : fn
    u=y(:,i);                            % ȡ��i֡����
    v=wavlet_barkms(u,'db2',fs);         % ����С�����ֽ��ȡ17��BARK�Ӵ�����
    num=size(v,1);
    for k=1 : num   
        Srt=v(k,:);                      % ȡ�õ�k��BARK�Ӵ��е�����
        Dst(k)=var(Srt);                 % ���k��BARK�Ӵ��еķ���ֵ
    end
    Dvar(i)=mean(Dst);                   % ��17��BARK�Ӵ����㷽��ƽ��
    waitbar(i/fn,h)                      % ��ʾ���еİٷֱ�,�ú�����ʾ
% ��ʾ��ͼ������"�˵���",����ʾ���еİٷֱ���,�����ֱ�ʾ
    set(h,'name',['�˵��� - ' sprintf('%2.1f',i/fn*100) '%'])
end
close(h)                                % �رճ��������ͼ
Dvarm=multimidfilter(Dvar,10);          % ƽ������
Dvarm=Dvarm/max(Dvarm);                 % ��ֵ��һ��

dth=mean(Dvarm(1:(NIS)));               % ��ֵ����
T1=1.5*dth;
T2=2.5*dth;
[voiceseg,vsl,SF,NF]=vad_param1D(Dvarm,T1,T2);% С����BARK�Ӵ�ʱ�򷽲�˫���޵Ķ˵���
% ��ͼ
subplot 311; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 312; plot(time,signal,'k');
title('������������(�����10dB)');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 313; plot(frameTime,Dvarm,'k');
title('С������ʱBARK�Ӵ�����ֵ'); axis([0 max(time) 0 1.2*max(Dvarm)]);
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
    line([frameTime(nx1) frameTime(nx1)],[0 1.2*max(Dvarm)],'color','k','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2*max(Dvarm)],'color','k','LineStyle','--');
end
