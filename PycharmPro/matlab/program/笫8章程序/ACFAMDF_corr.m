function period=ACFAMDF_corr(y,fn,vseg,vsl,lmax,lmin)
pn=size(y,2);
if pn~=fn, y=y'; end                      % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
period=zeros(1,fn);                       % ��ʼ��
wlen=size(y,1);                           % ȡ��֡��
Acm=zeros(1,lmax);
for i=1 : vsl                             % ֻ���л������ݴ���
    ixb=vseg(i).begin;
    ixe=vseg(i).end;
    ixd=ixe-ixb+1;                        % ��ȡһ���л��ε�֡��
    for k=1 : ixd                         % �Ըö��л������ݴ���
        u=y(:,k+ixb-1);                   % ȡ��һ֡����
        ru= xcorr(u, 'coeff');            % �����һ������غ���
        ru = ru(wlen:end);                % ȡ�ӳ���Ϊ��ֵ�Ĳ���
        for m = 1:wlen
             R(m) = sum(abs(u(m:wlen)-u(1:wlen-m+1))); % ����ƽ�����Ȳ��(AMDF)
        end 
        R=R(1:length(ru));                % ȡ��ru�ȳ� 
        Rindex=find(R~=0);       
        Acm(Rindex)=ru(Rindex)'./R(Rindex);% ����ACF/AMDF
        [tmax,tloc]=max(Acm(lmin:lmax));  % ��Pmin��Pmax��Χ��Ѱ�����ֵ
        period(k+ixb-1)=lmin+tloc-1;      % ������Ӧ���ֵ���ӳ���
        
    end
end


