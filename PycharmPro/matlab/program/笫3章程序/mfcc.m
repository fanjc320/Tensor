function ccc=mfcc(x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 function ccc=mfcc(x);
%���������������x����MFCC��������ȡ������MFCC������һ��
%���MFCC������Mel�˲����Ľ���Ϊ24
%fft�任�ĳ���Ϊ256������Ƶ��Ϊ8000Hz����x 256���Ϊһ֡
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


bank=melbankm(24,256,8000,0,0.5,'m');
% ��һ��mel�˲�����ϵ��
bank=full(bank);
bank=bank/max(bank(:));

% DCTϵ��,12*24
for k=1:12
  n=0:23;
  dctcoef(k,:)=cos((2*n+1)*k*pi/(2*24));
end

% ��һ��������������
w = 1 + 6 * sin(pi * [1:12] ./ 12);
w = w/max(w);

% Ԥ�����˲���
xx=double(x);
xx=filter([1 -0.9375],1,xx);

% �����źŷ�֡
xx=enframe(xx,256,80);

% ����ÿ֡��MFCC����
for i=1:size(xx,1)
  y = xx(i,:);
  s = y' .* hamming(256);
  t = abs(fft(s));
  t = t.^2;
  c1=dctcoef * log(bank * t(1:129));
  c2 = c1.*w';
  m(i,:)=c2';
end

%���ϵ��
dtm = zeros(size(m));
for i=3:size(m,1)-2
  dtm(i,:) = -2*m(i-2,:) - m(i-1,:) + m(i+1,:) + 2*m(i+2,:);
end
dtm = dtm / 3;

%�ϲ�mfcc������һ�ײ��mfcc����
ccc = [m dtm];
%ȥ����β��֡����Ϊ����֡��һ�ײ�ֲ���Ϊ0
ccc = ccc(3:size(m,1)-2,:);
