function y=conv_ovlsav1(x,h,L)
% ���ص��洢��������
x=x(:)';
Lenx = length (x);          % ����x��
N=length(h);                % ����h�� 
N1=N-1;
M=L-N1;
H=fft(h, L);                % ��h��FFTΪH
x=[zeros(1,N1), x, zeros(1, L-1)]; % ǰ����
K = floor((Lenx+ N1-1)/M);  % ���֡����
Y=zeros(K+1, L);            % ��ʼ��
for k=0 : K                 % ��ÿ֡����
   Xk=fft(x(k*M+1:k*M+L));  % �� FFT
   Y(k+1,:)=real(ifft(Xk.*H));  % ��˽��о��
end
Y=Y(:, N:L)';               % ��ÿ֡��ֻ��������M������
nm=fix(N/2);
y=Y(nm+1:nm+Lenx )';        % �����ӳ���,����2ά���1ά,ȡ���������x�ȳ�
