function tz=steager(z)
N=length(z);                       % ȡ�����ݳ���

for k=2 : N-1                      % ����Teager����
    tz(k)=z(k)^2-z(k-1)*z(k+1);
end
tz(1)=2*tz(2)-tz(3);               % ����������������˵��ֵ
tz(N)=2*tz(N-1)-tz(N-2);
