function period=corrbp_test1(y,fn,vseg,vsl,lmax,lmin,ThrC,tst_i1)
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
    figure(50);
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)]);
    for k=1 : ixd                         
        u=y(:,k+ixb-1);                   % ȡ��һ֡�ź�
        ru=xcorr(u,'coeff');              % ��������غ���
        ru=ru(wlen:end);                  % ȡ���ӳ�������
        subplot 211; plot(u,'k');
        title(['��' num2str(k) '֡����']);
        xlabel('(a)'); ylabel('��ֵ');
        subplot 212;
        plot(ru,'k'); grid; xlim([0 150]);
        title(['��' num2str(k) '֡����غ���R']);
        [Sv,Kv]=findmaxesm3(ru,lmax,lmin);% ��ȡ��������ֵ����ֵ��λ��
        lkv=length(Kv);
        Ptk(1:lkv,k)=Kv';                 % ��ÿ֡����������ֵλ�ô����Ptk������
        fprintf('%4d   %4d   %4d   %4d\n',k,Kv);
        xlabel(['������' 10 '(b)']); ylabel('��ֵ');
        pause
    end
    
    Kal=Ptk(1,:);                      % ����Kal
    meanx=mean(Kal);                   % ����Kal��ֵ
    thegma=std(Kal);                   % ����Kal��׼��
    mt1=meanx+thegma;                  % �������������Ͻ�
    mt2=meanx-thegma;                  % �������������½�
    fprintf('meanx=%5.4f   thegma=%5.4f   mt1=%5.4f   mt2=%5.4f\n',...
        meanx,thegma,mt1,mt2);
    % ����Ԫ�������������ֵ����
    figure(51);clf
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)-200]);
    plot(1:ixd,Ptk(1,1:ixd),'ko-','linewidth',2); hold on
    plot(1:ixd,Ptk(2,1:ixd),'k*-',1:ixd,Ptk(3,1:ixd),'k+-');
    xlabel('������'); ylabel('��������');  
    line([1 ixd],[meanx meanx],'color','k','linestyle','--');
    line([1 ixd],[meanx+thegma meanx+thegma],'color','k',...
        'linestyle','-.');
    line([1 ixd],[meanx-thegma meanx-thegma],'color','k',...
        'linestyle','-.');
    period=Kal;
    return    

