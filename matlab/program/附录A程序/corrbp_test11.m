function period=corrbp_test11(y,fn,vseg,vsl,lmax,lmin,ThrC,tst_i1)
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
    plot(1:ixd,Ptk(1,1:ixd),'ko-',1:ixd,Ptk(2,1:ixd),'k*-',1:ixd,...
    Ptk(3,1:ixd),'k+-');
    xlabel('������'); ylabel('��������'); hold on
    % �����һ����������
    meanx=mean(Kal);                      % ����Kal��ֵ
    thegma=std(Kal);                      % ����Kal��׼��
    mt1=meanx+thegma;                     % �������������Ͻ�
    mt2=meanx-thegma;                     % �������������½�
    fprintf('meanx=%5.4f   thegma=%5.4f   mt1=%5.4f   mt2=%5.4f\n',...
        meanx,thegma,mt1,mt2);
    line([1 ixd],[meanx meanx],'color','k','linestyle','--');
    line([1 ixd],[meanx+thegma meanx+thegma],'color','k',...
        'linestyle','-.');
    line([1 ixd],[meanx-thegma meanx-thegma],'color','k',...
        'linestyle','-.');
    if thegma<=5, period=Kal; return; end
    Ptemp=zeros(size(Ptk));
    
    Ptemp(1,(Ptk(1,:)<mt1 & Ptk(1,:)>mt2))=1;% �����������з�������������
    Ptemp(2,(Ptk(2,:)<mt1 & Ptk(2,:)>mt2))=1;
    Ptemp(3,(Ptk(3,:)<mt1 & Ptk(3,:)>mt2))=1;
    Pam=zeros(1,ixd);
    for k=1 : ixd                         % ��Pam�д�����������ڴ����ֵ
        if Ptemp(1,k)==1
            Pam(k)=max(Ptk(:,k).*Ptemp(:,k));
        end
    end
    % ����ڶ�����������
    mdex=find(Pam~=0);
    meanx=mean(Pam(mdex));                % ����Pam����ֵ����ľ�ֵ
    thegma=std(Pam(mdex));                % ����Pam����ֵ����ı�׼��
    mt1=meanx+thegma;                     % �������������Ͻ�                        
    mt2=meanx-thegma;                     % �������������½�
    pindex=find(Pam<mt1 & Pam>mt2);       % Ѱ����������������ݵ�
    fprintf('meanx2=%5.4f   thegma=%5.4f   mt1=%5.4f   mt2=%5.4f\n',...
        meanx,thegma,mt1,mt2);
    line([1 ixd],[meanx meanx],'color',[.6 .6 .6],'linestyle','--');
    line([1 ixd],[meanx+thegma meanx+thegma],'color',...
        [.6 .6 .6],'linestyle','-.');
    line([1 ixd],[meanx-thegma meanx-thegma],'color',...
        [.6 .6 .6],'linestyle','-.');
    Pamtmp=zeros(1,ixd);
    Pamtmp(pindex)=Pam(pindex);           % ����Pamtmp
    period=Pamtmp;
    line([1:ixd],[Pamtmp(1:ixd)],'color',[.6 .6 .6],'linewidth',3);
    title('�ڵڶ�������������Ԫ������������ڵĳ���ֵ');
    

