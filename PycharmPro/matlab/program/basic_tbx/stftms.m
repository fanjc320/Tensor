function d=stftms(x,win,nfft,inc)
if length(win)==1          % �ж��з����ô�����
    wlen=win;              % ����֡��
    win=hanning(wlen);     % ���ô�����
else
    wlen=length(win);      % ��֡��
end
x=x(:); win=win(:);        % ��x��win����Ϊ������
s = length(x);             % ����x�ĳ���

c = 1;
d = zeros((1+nfft/2),1+fix((s-wlen)/inc));   % ��ʼ���������
 
for b = 0:inc:(s-wlen)           % ����ѭ��
  u = win.*x((b+1):(b+wlen));    % ȡ��һ֡���ݼӴ�
  t = fft(u,nfft);               % ���и���Ҷ�任
  d(:,c) = t(1:(1+nfft/2));      % ȡ1��1+nfft/2֮�����ֵ
  c = c+1;                       % �ı�֡������ȡ��һ֡
end;
