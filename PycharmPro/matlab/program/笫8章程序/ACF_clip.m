function period=ACF_clip(y,fn,vseg,vsl,lmax,lmin)
pn=size(y,2);
if pn~=fn, y=y'; end                      % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
wlen=size(y,1);                           % ȡ��֡��
period=zeros(1,fn);                       % ��ʼ��

for i=1 : vsl                             % ֻ���л������ݴ���
    ixb=vseg(i).begin;
    ixe=vseg(i).end;
    ixd=ixe-ixb+1;                        % ��ȡһ���л��ε�֡��
    for k=1 : ixd                         % �Ըö��л������ݴ���
        u=y(:,k+ixb-1);                   % ȡ��һ֡����
        rate=0.7;                         % ������������ϵ��ȡ0.7
        cl=max(u)*rate;                   % ������������������
        for j=1:wlen                      % ����������������
            if u(j)>cl
                u(j)=u(j)-cl;
            elseif u(j)<=(-cl);
                u(j)=u(j)+cl;
            else
                u(j)=0;
            end
        end
        mx=max(u);                        % ��һ��
        u=u/mx;
        ru= xcorr(u, 'coeff');            % �����һ������غ���
        ru = ru(wlen:end);                % ȡ�ӳ���Ϊ��ֵ�Ĳ���
        [tmax,tloc]=max(ru(lmin:lmax));   % ��Pmin��Pmax��Χ��Ѱ�����ֵ
        period(k+ixb-1)=lmin+tloc-1;      % ������Ӧ���ֵ���ӳ���
    end
end
