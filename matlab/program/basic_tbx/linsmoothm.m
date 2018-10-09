function [y] = linsmoothm(x,n)
if nargin< 2
    n=3;
end
win=hanning(n);                        % ��hanning��
win=win/sum(win);                      % ��һ�� 
x=x(:)';                               % ��xת��Ϊ������

len=length(x);
y= zeros(len,1);                       % ��ʼ��y
% ��x����ǰ��n����,�Ա�֤���������x��ͬ��
if mod(n, 2) ==0
    l=n/2;
    x = [ones(1,1)* x(1) x ones(1,l)* x(len)]';
else
    l=(n-1)/2;
    x = [ones(1,1)* x(1) x ones(1,l+1)* x(len)]';
end
% ����ƽ������
for k=1:len
    y(k) = win'* x(k:k+ n- 1);
end


