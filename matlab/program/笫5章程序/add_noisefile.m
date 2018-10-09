function [y,noise] = add_noisefile(s,filepath_name,SNR,fs)
s=s(:);                             % ���ź�ת����������
s=s-mean(s);                        % ����ֱ������
[wavin,fs1,nbits]=wavread(filepath_name);   %���������ļ�������
wavin=wavin(:);                     % ����������ת����������
if fs1~=fs                          % �������źŵĲ���Ƶ���������Ĳ���Ƶ�ʲ����
    wavin1=resample(wavin,fs,fs1);  % �������ز�����ʹ��������Ƶ���봿�����źŵĲ���Ƶ����ͬ
else
    wavin1=wavin;
end
wavin1=wavin1-mean(wavin1);         % ����ֱ������

ns=length(s);                       % ���s�ĳ���
noise=wavin1(1:ns);                 % ���������Ƚض�Ϊ��s�ȳ�
noise=noise-mean(noise);            % ����ȥ��ֱ������
signal_power = 1/ns*sum(s.*s);      % ����źŵ�����
noise_power=1/ns*sum(noise.*noise); % �������������
noise_variance = signal_power / ( 10^(SNR/10) );   % ��������趨�ķ���ֵ
noise=sqrt(noise_variance/noise_power)*noise;      % ����������ֵ
y=s+noise;                          % ���ɴ�������

