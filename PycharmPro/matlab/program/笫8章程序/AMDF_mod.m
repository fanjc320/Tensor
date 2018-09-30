function period=AMDF_mod(y,fn,vseg,vsl,lmax,lmin)
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
        for m = 1:wlen
             R0(m) = sum(abs(u(m:wlen)-u(1:wlen-m+1))); % ����ƽ�����Ȳ��
        end 
        [Rmax,Nmax]=max(R0);              % ��ȡAMDF�����ֵ�Ͷ�Ӧλ��
        for i = 1 : wlen                  % �������Ա任
            R(i)=Rmax*(wlen-i)/(wlen-Nmax)-R0(i);
        end
        [Rmax,T]=max(R(lmin:lmax));       % ������ֵ
        T0=T+lmin-1;                      
        period(k+ixb-1)=T0;               % �����˸�֡�Ļ�������
    end
end


