function period=CAMDF_1(y,fn,vseg,vsl,lmax,lmin)
pn=size(y,2);
if pn~=fn, y=y'; end                      % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
period=zeros(1,fn);                       % ��ʼ��
wlen=size(y,1);                           % ȡ��֡��
for i=1 : vsl                             % ֻ���л������ݴ���
    ixb=vseg(i).begin;
    ixe=vseg(i).end;
    ixd=ixe-ixb+1;                        % ��ȡһ���л��ε�֡��
    for k=1 : ixd                         % �Ըö��л������ݴ���
        u=y(:,k+ixb-1);                   % ȡ��һ֡����
        for m = 0:wlen-1
            R(m+1)=0;
            for n=0:wlen-1
                R(m+1) = R(m+1)+abs(u(mod(m+n,wlen)+1)-u(n+1)); % ����ѭ��ƽ�����Ȳ��
            end
        end 
        [Rmax,Rloc]=max(R(1:lmin));       % ������ֵ
        Rth=0.6*Rmax;                     % ����һ����ֵ
        Rm=find(R(lmin:lmax)<=Rth);       % ��Pmin��Pmax����Ѱ�ҳ�С�ڸ���ֵ������
        if isempty(Rm)                    % ����Ҳ���,T0��Ϊ0
            T0=0;
        else
            m11=Rm(1);                    % �����С����ֵ������
            m22=lmax;
            [Amin,T]=min(R(m11:m22));     % Ѱ����Сֵ�Ĺ�ֵ��
            if isempty(T)
                T0=0;
            else
                T0=T+m11-1;               % ����С��ֵ���λ�ø���T0
            end
            period(k+ixb-1)=T0;           % �����˸�֡�Ļ�������
        end
    end
end


