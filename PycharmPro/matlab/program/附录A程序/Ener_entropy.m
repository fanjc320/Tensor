function Ef=Ener_entropy(y,fn)
if size(y,2)~=fn, y=y'; end
wlen=size(y,1);
for i=1:fn
    Sp = abs(fft(y(:,i)));                    % FFTȡ��ֵ
    Sp = Sp(1:wlen/2+1);	              % ֻȡ��Ƶ�ʲ���
    Esum(i) = sum(Sp.*Sp);                    % ��������ֵ
    prob = Sp/(sum(Sp));	              % �������
    H(i) = -sum(prob.*log(prob+eps));         % ������ֵ
end
Ef=sqrt(1 + abs(Esum./H));                    % �������ر�
Ef=Ef/max(Ef);                                % ��һ��

