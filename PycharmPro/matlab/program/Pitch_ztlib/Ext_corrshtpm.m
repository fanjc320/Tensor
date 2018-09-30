function [Pm,vsegch,vsegchlong]=Ext_corrshtpm(y,sign,TT1,XL,ixb,lmax,lmin,ThrC)
wlen=size(y,1);                           % ȡ��֡��
c1=ThrC(1); c2=ThrC(2);                   % ȡ����ֵ

for k=1 : XL                              % ѭ��XL����������������ȡ�������ڳ���ֵ
    j=ixb+sign*k;                         % ����֡�ı��
    u=y(:,j);                             % ȡ��һ֡�ź�
    ru=xcorr(u,'coeff');                  % ��������غ���
    ru=ru(wlen:end);                      % ȡ���ӳ�������
    [Sv,Kv]=findmaxesm3(ru,lmax,lmin);    % ��ȡ��������ֵ����ֵ��λ��
    Ptk(:,k)=Kv';
end
% ����̾���Ѱ�һ�������
Pkint=zeros(1,XL);
Ts=TT1;                                   % ��ʼ����Ts
Emp=0;                                    % ��ʼ����Emp
for k=1 : XL                              % ѭ��
    Tp=Ptk(:,k);                          % Ts�뱾֡��������ֵ��Ѱ�Ҳ�ֵ��С
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
        elseif zx1==1 & zx2~=XL         % ��ֵ�㷢����������ͷ��
            deltazx=(Pm(zx2+1)-TT1)/(zxl+1);
            for k2=1:zxl                % ����TT1�����ڲ�
                Pm(zx1+k2-1)=TT1+k2*deltazx;
            end
        else
            vsegch=1;
            vsegchlong=zxl;
        end
    end
end

