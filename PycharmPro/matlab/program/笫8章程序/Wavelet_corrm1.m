function period=Wavelet_corrm1(y,fn,vseg,vsl,lmax,lmin)
pn=size(y,2);
if pn~=fn, y=y'; end                      % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
period=zeros(1,fn);                       % ��ʼ��

for i=1 : vsl                             % ֻ���л������ݴ���
    ixb=vseg(i).begin;
    ixe=vseg(i).end;
    ixd=ixe-ixb+1;                        % ��ȡһ���л���֡��֡��
    for k=1 : ixd                         % �Ըö��л������ݴ���
        u=y(:,k+ixb-1);                   % ȡ��һ֡����
        [ca1,cd1] = dwt(u,'db4');         % ��dwt��С���任 
        a1 = upcoef('a',ca1,'db4',1);     % �õ�Ƶϵ���ع��ź�
        ru=xcorr(a1, 'coeff');            % �����һ������غ���
        aL=length(a1);
        ru=ru(aL:end);                    % ȡ�ӳ���Ϊ��ֵ�Ĳ���
        [tmax,tloc]=max(ru(lmin:lmax));   % ��lmin��lmax��Χ��Ѱ�����ֵ 
        period(k+ixb-1)=lmin+tloc-1;      % ������Ӧ���ֵ���ӳ���
    end
end