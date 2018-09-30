function [Pm,vsegch,vsegchlong]=Ext_corrshtp_test11(y,sign,TT1,XL,ixb,...
    lmax,lmin,ThrC)
wlen=size(y,1);
Emp=0;                                    % ��ʼ����Emp
c1=ThrC(1); c2=ThrC(2);
% ѭ��XL��ǰ����������������ȡ�������ڳ���ֵ
for k=1 : XL                              
    j=ixb+sign*k;                         % ����֡�ı��
    u=y(:,j);                             % ȡ��һ֡�ź�
    ru=xcorr(u,'coeff');                  % ��������غ���
    ru=ru(wlen:end);                      % ȡ���ӳ�������
    [Sv,Kv]=findmaxesm3(ru,lmax,lmin);    % ��ȡ��������ֵ����ֵ��λ��
    Ptk(:,k)=Kv';
    fprintf('%4d   %4d   %4d   %4d\n',k,Kv);
end
    figure(51)
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
    plot(1:XL,Ptk(1,1:XL),'ko-',1:XL,Ptk(2,1:XL),'k*-',...
        1:XL,Ptk(3,1:XL),'k+-'); grid;
    xlabel('������'); ylabel('��������'); hold on
    Pm=Ptk(1,:);
    vsegch=0; vsegchlong=0;
% ����̾���Ѱ�һ�������
Pkint=zeros(1,XL);
Ts=TT1;                                   % ��ʼ����Ts
Emp=0;                                    % ��ʼ����Emp
for k=1 : XL                              % ѭ��
    Tp=Ptk(:,k);                          % ��Ts�뱾֡��������ֵ��Ѱ����С��ֵ
    Tz=abs(Ts-Tp);
    [tv,tl]=min(Tz);                      % ��С��λ����tl,��ֵΪtv
    if k==1                               % �Ƿ��1֡
        if tv<=c1, Pkint(k)=Tp(tl); Ts=Tp(tl);%��,tvС��c1,����Pkint��Ts
        else Pkint(k)=0; Emp=1; end       % tv����c1,PkintΪ0,Emp=1,Ts����
    else                                  % ���ǵ�1֡
        if Pkint(k-1)==0                  % ��һ֡Pkint�Ƿ�Ϊ0
            if tv<c2, Pkint(k)=Tp(tl); Ts=Tp(tl);%��,tvС��c2,����Pkint��Ts
            else Pkint(k)=0; Emp=1; end   % tv����c2,PkintΪ0,Emp=1,Ts����
        else                              % ��һ֡Pkint��Ϊ0
            if tv<=c1, Pkint(k)=Tp(tl); Ts=Tp(tl);%tvС��c1,����Pkint��Ts
            else Pkint(k)=0; Emp=1; end   % tv����c1,PkintΪ0,Emp=1,Ts����
        end
    end
end
line([1:XL],[Pkint(1:XL)],'color',[.6 .6 .6],'linewidth',3);

% �ڲ崦��
Pm=Pkint;
vsegch=0;
vsegchlong=0;
if Emp==1
    pindexz=find(Pkint==0);             % Ѱ����ֵ�������Ϣ
    pzseg=findSegment(pindexz);
    pzl=length(pzseg);                  % ��ֵ�����м���
    for k1=1 : pzl                      % ȡһ����ֵ����
        zx1=pzseg(k1).begin;            % ��ֵ���俪ʼλ��
        zx2=pzseg(k1).end;              % ��ֵ�������λ��
        zxl=pzseg(k1).duration;         % ��ֵ���䳤��
        if zx1~=1 & zx2~=XL             % ��ֵ�㴦�����������в�
            deltazx=(Pm(zx2+1)-Pm(zx1-1))/(zxl+1);
            for k2=1 : zxl              % �����ڲ�
                Pm(zx1+k2-1)=Pm(zx1-1)+k2*deltazx;
            end
        elseif zx1==1 & zx2~=XL         % ��ֵ�㷢�����������ײ�
            deltazx=(Pm(zx2+1)-TT1)/(zxl+1);
            for k2=1:zxl                % ����TT1�����ڲ�
                Pm(zx1+k2-1)=TT1+k2*deltazx;
            end
        else                            % ��ֵ�㷢����������β����
            vsegch=1;
            vsegchlong=zxl;
        end
    end
end
plot(1:XL,Pm,'k','linewidth',2);
title('��������������ڳ���ֵ');

