%
% pra_4_1 
clear all; clc; close all;

load dpstp_tmpdata1.mat
ixd=length(Pamtmp);
Pam=Pamtmp;
   bpseg=findSegment(pindex);        % �������������ڵ����ݷֶ���Ϣ
   bpl=length(bpseg);                % �����������ڵ����ݷֳɼ��� 
   bdb=bpseg(1).begin;               % ���������ڵ�һ�εĿ�ʼλ��
   if bdb~=1                         % ������������ڵ�һ�ο�ʼλ�ò�Ϊ1
       Ptb=Pamtmp(bdb);              % ���������ڵ�һ�ο�ʼλ�õĻ�������
       Ptbp=Pamtmp(bdb+1);
       pdb=ztcont11(Ptk,bdb,Ptb,Ptbp,c1);% ������ztcont11
       Pam(1:bdb-1)=pdb;             % �Ѵ��������ݷ���Pam��
   end
   if bpl>=2
       for k=1 : bpl-1               % ����м�������������������
           pdb=bpseg(k).end;
           pde=bpseg(k+1).begin;
           Ptb=Pamtmp(pdb);
           Pte=Pamtmp(pde);
           pdm=ztcont21(Ptk,pdb,pde,Ptb,Pte,c1);% ����ztcont21
           Pam(pdb+1:pde-1)=pdm;     % �Ѵ��������ݷ���Pam��
       end
   end
   Pam2=ones(1,ixd)*nan;             % Ϊ���ܻ���������м����ݵ�����
   Pam2(pdb+1:pde-1)=pdm;            % ������Pam2
   Pam2(pdb)=Ptb; Pam2(pde)=Pte;
   bde=bpseg(bpl).end;
   Pte=Pamtmp(bde);
   Pten=Pamtmp(bde-1);
   if bde~=ixd                       % ����������������һ�ο�ʼλ�ò�Ϊkl
% �������ɺ���ztcont31�����ݱ��ƶ�����
       fn=size(Ptk,2);               % ȡ��Ptk�ж�����
       kl=fn-bde;                    % ȡ���������������ж��ٸ����ݵ�
       T0=Pte;                       % �������������һ�����һ�����ݵ�Ļ�������ֵ
       T1=Pten;                      % �������������һ�����ڶ������ݵ�Ļ�������ֵ
       pde=zeros(1,kl);              % ��ʼ��pde
       for k=1:kl                    % ѭ��
           j=k+bde;
           [mv,ml]=min(abs(T0-Ptk(:,j)));  % ��ʽ(8-6-5)Ѱ����С��ֵ
           pde(k)=Ptk(ml,j);               % �ҵ�ml;
           fprintf('k=%4d   %4d   ',k,pde(k));
           TT=Ptk(ml,j);
           if abs(T0-TT)>c1                % ���������ʽ(8-6-6)
               TT=2*T0-T1;                 % �����������
               pde(k)=TT;
               fprintf('����Ϊ %4d',pde(k));
           end
           fprintf('\n');
           T1=T0;
           T0=TT;
       end
% ����ztcont31����
       Pam(bde+1:ixd)=pde;           % �Ѵ��������ݷ���Pam��
       Pam3=ones(1,ixd)*nan;         % Ϊ���ܻ��������β�����ݵ�����
       Pam3(bde+1:ixd)=pde;          % ������Pam3
       Pam3(bde)=Pte;
   end
figure(1)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
plot(1:ixd,Ptk(1,:),'kO','linewidth',3); hold on   
plot(1:ixd,Pamtmp,'k','linewidth',3);
line([0 ixd],[meanx meanx],'color',[.6 .6 .6],'lineStyle','-');
line([0 ixd],[mt1 mt1],'color',[.6 .6 .6],'lineStyle','--');
line([0 ixd],[mt2 mt2],'color',[.6 .6 .6],'lineStyle','--');
line([1:ixd],[Pam2(1:ixd)],'color',[.6 .6 .6],'linewidth',3,'lineStyle','-');
line([1:ixd],[Pam3(1:ixd)],'color',[.6 .6 .6],'linewidth',3,'lineStyle','--');
xlabel('����ֵ'); ylabel('��������');
