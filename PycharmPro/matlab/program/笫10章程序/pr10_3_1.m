%
% pr10_3_1 
clear all; clc; close all;

filedir=[];                             % ����·��
filename='colorcloud.wav';              % �����ļ���
fle=[filedir filename];                 % ����������·�����ļ���
[x, fs, bits] = wavread(fle);           % ���������ļ�
x=x-mean(x);                            % ����ֱ������
x=x/max(abs(x));                        % ��ֵ��һ
xl=length(x);                           % ���ݳ���
time=(0:xl-1)/fs;                       % �����ʱ��̶�
p=12;                                   % LPC�Ľ���Ϊ12
wlen=200; inc=80;                       % ֡����֡��
msoverlap = wlen - inc;                 % ÿ֡�ص����ֵĳ���
y=enframe(x,wlen,inc)';                 % ��֡
fn=size(y,2);                           % ȡ֡��
% ��������:��ÿһ֡��LPCϵ����Ԥ�����
for i=1 : fn                            
    u=y(:,i);                           % ȡ��һ֡
    A=lpc(u,p);                         % LPC���ϵ��
    aCoeff(:,i)=A;                      % �����aCoeff������
    errSig = filter(A,1,u);             % ����Ԥ���������
    resid(:,i) = errSig;                % �����resid������
end
% �����ϳ�:��ÿһ֡�ĺϳ��������ӳ����������ź�
for i=1:fn                              
    A = aCoeff(:,i);                    % ȡ�ø�֡��Ԥ��ϵ��
    residFrame = resid(:,i);            % ȡ�ø�֡��Ԥ�����
    synFrame = filter(1, A', residFrame); % Ԥ������,�ϳ�����
    
    outspeech((i-1)*inc+1:i*inc)=synFrame(1:inc);  % �ص��洢���������
% ��������һ֡,��inc������ݲ���
    if i==fn                            
        outspeech(fn*inc+1:(fn-1)*inc+wlen)=synFrame(inc+1:wlen);
    end

end;
ol=length(outspeech);
if ol<xl                                % ��outspeech����,ʹ��x�ȳ�
    outspeech=[outspeech zeros(1,xl-ol)];
end
% ����
wavplay(x,fs);
pause(1)
wavplay(outspeech,fs);
% ��ͼ
subplot 211; plot(time,x,'k');
xlabel(['ʱ��/s' 10 '(a)']); ylabel('��ֵ'); ylim([-1 1.1]);
title('ԭʼ�����ź�')
subplot 212; plot(time,outspeech,'k');
xlabel(['ʱ��/s' 10 '(b)']); ylabel('��ֵ'); ylim([-1 1.1]);
title('�ϳɵ������ź�')


