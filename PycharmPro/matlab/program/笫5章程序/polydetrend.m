function [y,xtrend]=polydetrend(x, fs, m)
x=x(:);                 % �������ź�xת��Ϊ������
N=length(x);            % ���x�ĳ���
t= (0: N-1)'/fs;        % ��x�ĳ��ȺͲ���Ƶ������ʱ������
a=polyfit(t, x, m);     % ����С���˷���������ź�x�Ķ���ʽϵ��a
xtrend=polyval(a, t);   % ��ϵ��a��ʱ������t����������
y=x-xtrend;             % �������ź�x�����������