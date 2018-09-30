function plot_spectrogram(x,wlen,inc,nfft,fs)

if nargin<5, fs=1; end                    % ��û������fs,��fs=1
d=stftms(x,wlen,nfft,inc);                % ��ʱ����Ҷ�任
W2=1+nfft/2;                              % ��Ƶ�ʵĳ���
n2=1:W2;
freq=(n2-1)*fs/nfft;                      % ����Ƶ��
fn=size(d,2);                             % ��֡��
frameTime=frame2time(fn,wlen,inc,fs);     % ����ÿ֡��Ӧ��ʱ��
imagesc(frameTime,freq,abs(d(n2,:)));  axis xy % ��ͼ
m = 64; LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0]; Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors));
