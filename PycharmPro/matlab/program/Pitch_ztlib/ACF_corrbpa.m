function period=ACF_corrbpa(y,fn,vseg,vsl,lmax,lmin,ThrC)
pn=size(y,2);
if pn~=fn, y=y'; end                      % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
wlen=size(y,1);                           % ֡��
period=zeros(1,fn);                       % ��ʼ��
c1=ThrC(1);                               % ȡ����ֵ
for i=1 : vsl                             % ֻ���л������ݴ���
    ixb=vseg(i).begin;
    ixe=vseg(i).end;
    ixd=ixe-ixb+1;                        % ��ȡһ���л���֡��֡��
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
    meanx=mean(Kal);                      % ����Kal��ֵ
    thegma=std(Kal);                      % ����Kal��׼��
    mt1=meanx+thegma;
    mt2=meanx-thegma;
    if thegma>5, 
        %�жϻ�����ѡ�������ļ����ڵ�һ������������
        Ptemp=zeros(size(Ptk));
        Ptemp(1,(Ptk(1,:)<mt1 & Ptk(1,:)>mt2))=1;
        Ptemp(2,(Ptk(2,:)<mt1 & Ptk(2,:)>mt2))=1;
        Ptemp(3,(Ptk(3,:)<mt1 & Ptk(3,:)>mt2))=1;
        % �����һ���(��)�����鶼��ֵ�ڵ�һ����������,ȡ��ֵ���һ��ֵ����Pam
        Pam=zeros(1,ixd);
        for k=1 : ixd
            if Ptemp(1,k)==1
                Pam(k)=max(Ptk(:,k).*Ptemp(:,k));
            end
        end
        mdex=find(Pam~=0);                    % ��Pam�������ֵ��
        meanx=mean(Pam(mdex));                % ����Pam��ֵ
        thegma=std(Pam(mdex));                % ����Pam��׼��
        if thegma<0.5, thegma=0.5; end
        mt1=meanx+thegma;
        mt2=meanx-thegma;                     % �����˵ڶ���������
        pindex=find(Pam<mt1 & Pam>mt2);       % Ѱ����������������ݵ�
        Pamtmp=zeros(1,ixd);
        Pamtmp(pindex)=Pam(pindex);           % ����Pamtmp

        if length(pindex)~=ixd
            bpseg=findSegment(pindex);        % �������������ڵ����ݷֶ���Ϣ
            bpl=length(bpseg);                % ���������ڵ����ݷֳɼ��� 
            bdb=bpseg(1).begin;               % ���������ڵ�һ�εĿ�ʼλ��
            if bdb~=1                         % ������������ڵ�һ�ο�ʼλ�ò�Ϊ1
                Ptb=Pamtmp(bdb);              % ���������ڵ�һ�ο�ʼλ�õĻ�������
                Ptbp=Pamtmp(bdb+1);
                pdb=ztcont11(Ptk,bdb,Ptb,Ptbp,c1);% ������ztcont11��������
                Pam(1:bdb-1)=pdb;             % �Ѵ��������ݷ���Pam��
            end
            if bpl>=2
                for k=1 : bpl-1               % ������м�����������������
                    pdb=bpseg(k).end;
                    pde=bpseg(k+1).begin;
                    Ptb=Pamtmp(pdb);
                    Pte=Pamtmp(pde);
                    pdm=ztcont21(Ptk,pdb,pde,Ptb,Pte,c1);  % ������ztcont21��������
                    Pam(pdb+1:pde-1)=pdm;     % �Ѵ��������ݷ���Pam��
                end
            end
            bde=bpseg(bpl).end;
            Pte=Pamtmp(bde);
            Pten=Pamtmp(bde-1);
            if bde~=ixd                       % ����������������һ�ν���λ�ò�Ϊixd
                pde=ztcont31(Ptk,bde,Pte,Pten,c1);% ������ztcont31��������
                Pam(bde+1:ixd)=pde;           % �Ѵ��������ݷ���Pam��
            end
        end
        period(ixb:ixe)=Pam;    
    else    
        period(ixb:ixe)=Kal;
    end
end
