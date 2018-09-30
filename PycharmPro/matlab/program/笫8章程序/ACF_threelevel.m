function period=ACF_threelevel(y,fn,vseg,vsl,lmax,lmin)
pn=size(y,2);
if pn~=fn, y=y'; end                      % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
wlen=size(y,1);                           % ȡ��֡��
period=zeros(1,fn);                       % ��ʼ��

for i=1:vsl                               % ֻ���л������ݴ���
    ixb=vseg(i).begin;
    ixe=vseg(i).end;
    ixd=ixe-ixb+1;                        % ��ȡһ���л��ε�֡��
    for k=1 : ixd                         % �Ըö��л������ݴ���
        u=y(:,k+ixb-1);                   % ȡ��һ֡����
        px1=u(1:100);                     % ȡǰ��100������
        px2=u(wlen-99:wlen);              % ȡ��100������
        clm=min(max(px1),max(px2));       % ���������ֵ�н�Сһ��
        cl=clm*0.68;                      % ��0.68������ƽ��������ϵ��
        three=zeros(1,wlen);              % ��ʼ��
        for j=1:wlen;                     % ���������������������ƽ����
            if u(j)>cl;
                u(j)=u(j)-cl;
                three(j)=1;
            elseif   u(j)<-cl;
                u(j)=u(j)+cl;
                three(j)=-1; 
            else
                u(j)=0;
                three(j)=0;
            end
        end
% ���㻥��غ��������������   
        r=xcorr(three,u,'coeff');         % �����һ������غ���
        r=r(wlen:end);                    % ȡ�ӳ���Ϊ��ֵ�Ĳ���
        [v,b]=max(r(lmin:lmax));          % ��Pmin��Pmax��Χ��Ѱ�����ֵ
        period(k+ixb-1)=b+lmin-1;         % ������Ӧ���ֵ���ӳ���
    end
end
