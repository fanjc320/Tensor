function [A,B]=frmnt2coeff3(f0,bw,fs)
R = exp(-pi*bw/fs);                               % ��ʽ(10-5-17)���㼫ֵ��ģֵ
theta = 2*pi*f0/fs;                               % ��ʽ(10-5-16)���㼫ֵ�����
poles = R.* exp(j*theta);                         % ���ɸ�������
A = real(poly([poles,conj(poles)]));              % ��ʽ(10-5-18)�����ĸϵ��
B=abs(A(1)+A(2)*exp(j*theta)+A(3)*exp(j*2*theta));% ��ʽ(10-5-19)����b0


