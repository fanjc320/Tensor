% Part_II
[x,fs]=wavread(fle);                        % ����wav�ļ�
x=x-mean(x);                                % ��ȥֱ������
x=x/max(abs(x));                            % ��ֵ��һ��
y  = enframe(x,wlen,inc)';                  % ��֡
fn  = size(y,2);                            % ȡ��֡��
time = (0 : length(x)-1)/fs;                % ����ʱ������
frameTime = frame2time(fn, wlen, inc, fs);  % �����֡��Ӧ��ʱ������

[voiceseg,vosl,SF,Ef]=pitch_vad1(y,fn,T1);   % �����Ķ˵���
