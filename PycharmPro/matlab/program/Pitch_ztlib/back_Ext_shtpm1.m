function [Ext_T,voiceseg]=back_Ext_shtpm1(y,fn,voiceseg,Bth,ix1,...
        ixl1,T1,m,lmax,lmin,ThrC)
if nargin<11, ThrC(1)=10; ThrC(2)=15; end
if size(y,2)~=fn, y=y'; end               % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
wlen=size(y,1);                           % ȡ��֡��
TT1=round(T1(ix1));                       % Ԫ�������һ����Ļ�������
XL=ixl1;
sign=-1;                                  % ǰ������������
ixb=ix1;
[Pm,vsegch,vsegchlong]=Ext_corrshtpm(y,sign,TT1,XL,ixb,lmax,lmin,ThrC);

if vsegch==1                              % ���vsegchΪ1,Ҫ��voiceseg�������� 
    j=Bth(m);
    if m~=1
        j1=Bth(m-1);
% �жϱ�Ԫ���������һ��Ԫ�������Ƿ���ͬһ���л�����
        if j~=j1                         % ����ͬһ���л�����,��voiceseg��������
            voiceseg(j).begin=voiceseg(j).begin+vsegchlong;
            voiceseg(j).duration=voiceseg(j).duration-vsegchlong;
        end

    else                                  % �ǵ�һ��Ԫ������,��voiceseg��������
        voiceseg(j).begin=voiceseg(j).begin+vsegchlong;
        voiceseg(j).duration=voiceseg(j).duration-vsegchlong;
    end
end

Pm=Pm(:)';                                % Pm��������
Pmup=fliplr(Pm);                          % ��Pm����
Ext_T=Pmup;                               % ��ֵ��� 







