function formant1=Ext_frmnt(y,p,thr1,fs)
fn=size(y,2);
formant1=zeros(fn,3);
for i=1 : fn
        u=y(:,i);                                   % ȡһ֡����
        a=lpc(u,p);                                 % ���LPCϵ��
        root1=roots(a);                             % ���

        mag_root=abs(root1);                        % ȡ��֮ģֵ
        arg_root=angle(root1);                      % ȡ��֮���
        f_root=arg_root/pi*fs/2;                    % �����ת����Ƶ��
        fmn1=[];                                    % ��ʼ��
        k=1;
        for j=1:p
            if mag_root(j)>thr1                     % �Ƿ���������
                if arg_root(j)>0  & arg_root(j)<pi & f_root(j)>200
                    fmn1(k)=f_root(j);              % ����,���湲���Ƶ��
                    k=k+1;
                end
            end
        end
        if ~isempty(fmn1)                           % ������Ĺ����Ƶ������
            fl=length(fmn1);
            fmnt1=sort(fmn1);
            formant1(i,1:min(fl,3))=fmnt1(1:min(fl,3));% ���ȡ����
        end
end
