function [Ext_T,voiceseg]=fore_Ext_shtpm1(y,fn,voiceseg,Bth,ix2,...
        ixl2,vsl,T1,m,lmax,lmin,ThrC)
if nargin<12, ThrC(1)=10; ThrC(2)=15; end
if size(y,2)~=fn, y=y'; end               % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
wlen=size(y,1);                           % ȡ��֡��
XL=ixl2;
sign=1;                                   % ��������������
TT1=round(T1(ix2));                       % Ԫ���������һ����Ļ�������
ixb=ix2;
[Ext_T,vsegch,vsegchlong]=Ext_corrshtpm(y,sign,TT1,XL,ixb,lmax,lmin,ThrC);

if vsegch==1                              % ���vsegchΪ1,Ҫ��voiceseg��������
    j=Bth(m);                            
% �жϱ�Ԫ���������һ��Ԫ�������Ƿ���ͬһ���л�����
    if m~=vsl
        j1=Bth(m+1); 
        if j~=j1                         % ����ͬһ���л�����,��voiceseg��������
            voiceseg(j).end=voiceseg(j).end-vsegchlong;
            voiceseg(j).duration=voiceseg(j).duration-vsegchlong;
        end

    else                                  % �����һ��Ԫ������,��voiceseg��������
        voiceseg(j).end=voiceseg(j).end-vsegchlong;
        voiceseg(j).duration=voiceseg(j).duration-vsegchlong;
    end
end







