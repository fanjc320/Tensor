function y=pitfilterm1(x,vseg,vsl)

y=zeros(size(x));             % ��ʼ��
for i=1 : vsl                 % �ж�����
    ixb=vseg(i).begin;        % �öεĿ�ʼλ��
    ixe=vseg(i).end;          % �öεĽ���λ��
    u0=x(ixb:ixe);            % ȡ��һ������
    y0=medfilt1(u0,5);        % 5�����ֵ�˲�
    v0=linsmoothm(y0,5);      % ����ƽ�� 
    y(ixb:ixe)=v0;            % ��ֵ��y
end
