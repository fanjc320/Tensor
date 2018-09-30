function period=corrbp_test12(y,fn,vseg,vsl,lmax,lmin,ThrC,tst_i1)
pn=size(y,2);
if pn~=fn, y=y'; end                      % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
period=zeros(1,fn);                       % ��ʼ��
if tst_i1>vsl | tst_i1<1
    disp('error: ѡ���Ԫ�������ҪС��Ԫ���������!'); 
%    period=[];
    return;
end
wlen=size(y,1);                           % ֡��
c1=ThrC(1);                               % �������ڻ������ڼ����ֵ
i=tst_i1;                                 % i=��tst_i1��Ԫ������
    ixb=vseg(i).begin;                    % ��i��Ԫ�����忪ʼλ��
    ixe=vseg(i).end;                      % ��i��Ԫ���������λ��
    ixd=ixe-ixb+1;                        % ��ȡһ���л���֡��֡��
    fprintf('ixd=%4d\n',ixd);             % ��ʾ
    Ptk=zeros(3,ixd);                     % ��ʼ��
    Vtk=zeros(3,ixd);
    Ksl=zeros(1,3);
    for k=1 : ixd                         
        u=y(:,k+ixb-1);                   % ȡ��һ֡�ź�
        ru=xcorr(u,'coeff');              % ��������غ���
        ru=ru(wlen:end);                  % ȡ���ӳ�������
        [Sv,Kv]=findmaxesm3(ru,lmax,lmin);% ��ȡ��������ֵ����ֵ��λ��
        lkv=length(Kv);
        Ptk(1:lkv,k)=Kv';                 % ��ÿ֡��������ֵλ�ô����Ptk������
    end
    Kal=Ptk(1,:);
    figure(51);clf
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
    plot(1:ixd,Ptk(1,1:ixd),'ko-',1:ixd,Ptk(2,1:ixd),'k*-',...
        1:ixd,Ptk(3,1:ixd),'k+-');
    xlabel('������'); ylabel('��������'); hold on
    % �����һ����������
    meanx=mean(Kal);                      % ����Kal��ֵ
    thegma=std(Kal);                      % ����Kal��׼��
    mt1=meanx+thegma;                     % �������������Ͻ�
    mt2=meanx-thegma;                     % �������������½�
    fprintf('meanx=%5.4f   thegma=%5.4f   mt1=%5.4f   mt2=%5.4f\n',...
        meanx,thegma,mt1,mt2);
    line([1 ixd],[meanx meanx],'color','k','linestyle','--');
    line([1 ixd],[meanx+thegma meanx+thegma],'color','k','linestyle','-.');
    line([1 ixd],[meanx-thegma meanx-thegma],'color','k','linestyle','-.');
    if thegma>5, 
        
        Ptemp=zeros(size(Ptk));
        Ptemp(1,(Ptk(1,:)<mt1 & Ptk(1,:)>mt2))=1;% �����������з�����������
        Ptemp(2,(Ptk(2,:)<mt1 & Ptk(2,:)>mt2))=1;
        Ptemp(3,(Ptk(3,:)<mt1 & Ptk(3,:)>mt2))=1;
    
        Pam=zeros(1,ixd);                 % ��Pam�д�����������ڴ����ֵ
        for k=1 : ixd
            if Ptemp(1,k)==1
                Pam(k)=max(Ptk(:,k).*Ptemp(:,k));
            end
        end
        % ����ڶ�����������
        mdex=find(Pam~=0);
        meanx=mean(Pam(mdex));                % ����Pam��ֵ
        thegma=std(Pam(mdex));                % ����Pam��׼��
        if thegma<0.5, thegma=0.5; end
        mt1=meanx+thegma;                     % �������������Ͻ�
        mt2=meanx-thegma;                     % �������������½�
        pindex=find(Pam<mt1 & Pam>mt2);       % Ѱ����������������ݵ�
        fprintf('meanx2=%5.4f   thegma=%5.4f   mt1=%5.4f   mt2=%5.4f\n',...
            meanx,thegma,mt1,mt2);
        line([1 ixd],[meanx meanx],'color',[.6 .6 .6],'linestyle','--');
        line([1 ixd],[meanx+thegma meanx+thegma],'color',[.6 .6 .6],...
            'linestyle','-.');
        line([1 ixd],[meanx-thegma meanx-thegma],'color',[.6 .6 .6],...
            'linestyle','-.');
        Pamtmp=zeros(1,ixd);
        Pamtmp(pindex)=Pam(pindex);           % ����Pamtmp
        line([1:ixd],[Pamtmp(1:ixd)],'color',[.6 .6 .6],'linewidth',3);

        if length(pindex)~=ixd
            bpseg=findSegment(pindex);        % �������������ڵ����ݷֶ���Ϣ
            bpl=length(bpseg);                % ���������ڵ����ݷֳɼ��� 
            bdb=bpseg(1).begin;               % ���������ڵ�һ�εĿ�ʼλ��
            if bdb~=1                         % ������������ڵ�һ�ο�ʼλ�ò�Ϊ1
                Ptb=Pamtmp(bdb);              % ���������ڵ�һ�ο�ʼλ�õĻ�������
                Ptbp=Pamtmp(bdb+1);
                pdb=ztcont11(Ptk,bdb,Ptb,Ptbp,c1);% ������ztcont11
                Pam(1:bdb-1)=pdb;             % �Ѵ��������ݷ���Pam��
            end
            if bpl>=2
                for k=1 : bpl-1               % �����Kal�м�������������������
                    pdb=bpseg(k).end;
                    pde=bpseg(k+1).begin;
                    Ptb=Pamtmp(pdb);
                    Pte=Pamtmp(pde);
                    pdm=ztcont21(Ptk,pdb,pde,Ptb,Pte,c1);% ����ztcont21
                    Pam(pdb+1:pde-1)=pdm;     % �Ѵ��������ݷ���Pam��
                end
            end
            bde=bpseg(bpl).end;
            Pte=Pamtmp(bde);
            Pten=Pamtmp(bde-1);
            if bde~=ixd                 % ����������������һ�εĿ�ʼλ�ò�Ϊixd
                pde=ztcont31(Ptk,bde,Pte,Pten,c1);% ������ztcont31
                Pam(bde+1:ixd)=pde;           % �Ѵ��������ݷ���Pam��
            end
        end
        period(ixb:ixe)=Pam;    
    else    
        period(ixb:ixe)=Kal;
    end
    plot(1:ixd,period(ixb:ixe),'k^-','linewidth',2);
    title('Ԫ������������ڵĳ���ֵ');
    
    
