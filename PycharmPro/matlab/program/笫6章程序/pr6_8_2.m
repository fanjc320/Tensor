%
% pr6_8_2 
clear all; clc; close all;

run Set_I                               % ��������
run PART_I                              % �������ݣ���֡��׼��

imf=emd(signal);                        % EMD�ֽ�
M=size(imf,1);                          % ȡ�÷ֽ��IMF�Ľ���
u=zeros(1,N);
h=waitbar(0,'Running...');              % �������г��������ͼ,��ʼ��
set(h,'name','�˵��� - 0%');           % ���ñ�ͼ�����ơ��˵��⡱
for k=3 : M                             % ����ǰ2��IMF�ع������ź�
    u=u+imf(k,:);
end
z=enframe(u,wnd,inc)';                  % �ع������źŵķ�֡

for k=1 : fn
    v=z(:,k);                           % ȡ��һ֡
    imf=emd(v);                         % EMD�ֽ�
    L=size(imf,1);                      % ȡ�÷ֽ��IMF�Ľ���L
    Etg=zeros(1,wlen);
    for i=1 : L                         % ����ÿ��IMF��ƽ��Teager����
        Etg=Etg+steager(imf(i,:));
    end
    Tg(k,:)=Etg;
    Tgf(k)=mean(Etg);                   % ���㱾֡��ƽ��Teager����
    waitbar(k/fn,h)                     % ��ʾ���еİٷֱ�,�ú�����ʾ
% ��ʾ��ͼ������"�˵���",����ʾ���еİٷֱ���,�����ֱ�ʾ
    set(h,'name',['�˵��� - ' sprintf('%2.1f',k/fn*100) '%'])
end
close(h)                                % �رճ��������
Zcr=zc2(y,fn);                          % ���������
Tgfm=multimidfilter(Tgf,10);            % ƽ������
Zcrm=multimidfilter(Zcr,10);            % ƽ������
Mtg=max(Tgfm);                          % ������ֵ
Tmth=mean(Tgfm(1:NIS));
Zcrth=mean(Zcrm(1:NIS));
T1=1.5*Tmth;
T2=3*Tmth;
T3=0.9*Zcrth;
T4=0.8*Zcrth;
% ˫����˫���޵Ķ˵���
[voiceseg,vsl,SF,NF]=vad_param2D_revr(Tgfm,Zcrm,T1,T2,T3,T4);
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-200,pos(3),pos(4)+150]) 
subplot 511; plot(time,x,'k');
title('����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 512; plot(time,signal,'k');
title(['������������ �����=' num2str(SNR) 'dB']);
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 513; plot(time,u,'k'); ylim([-1 1]);
title('EMD���ع�����������');
ylabel('��ֵ'); axis([0 max(time) -1 1]);
subplot 514; plot(frameTime,Tgfm,'k'); 
title('��ʱEMD�ֽ��Teager����ƽ��ֵ'); ylim([0 1.2*Mtg]);
ylabel('T����ƽ��ֵ'); 
line([0 max(time)], [T1 T1],'color','k','lineStyle','--');
line([0 max(time)], [T2 T2], 'color','k','lineStyle','-');
subplot 515; plot(frameTime,Zcrm,'k');
title('��ʱ������ֵ'); ylim([0 1.2*max(Zcrm)]);
xlabel('ʱ��/s'); ylabel('������ֵ'); 
line([0 max(time)], [T3 T3], 'color','k','lineStyle','--');
line([0 max(time)], [T4 T4], 'color','k','lineStyle','-');

for k=1 : vsl
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    fprintf('%4d   %4d   %4d\n',k,nx1,nx2);
    figure(1); subplot 511; 
    line([frameTime(nx1) frameTime(nx1)],[-1 1],'color','k','lineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1 1],'color','k','lineStyle','--');
    subplot 514; 
    line([frameTime(nx1) frameTime(nx1)],[0 1.2*Mtg],'color','k','lineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2*Mtg],'color','k','lineStyle','--');
    subplot 515;
    line([frameTime(nx1) frameTime(nx1)],[0 1.2*max(Zcrm)],'color','k','lineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[0 1.2*max(Zcrm)],'color','k','lineStyle','--');
end


