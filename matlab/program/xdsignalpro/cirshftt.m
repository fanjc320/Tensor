function y=cirshftt(x,m,N);
% Circular shift of m samples wrt size N in sequence x:(time domain)
%
%
if length(x)>N
   error('N must be >= the length of x');
end
x=[x zeros(1, N-length(x))];
n=[0:1:N-1];
n=mod(n-m,N);
y=x(n+1);