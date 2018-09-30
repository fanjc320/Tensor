%
% pr9_2_1
clear all; clc; close all;

waveFile='snn27.wav';                            % �����ļ���
[x, fs, nbits]=wavread(waveFile);                % ����һ֡����
u=filter([1 -.99],1,x);                          % Ԥ����
wlen=length(u);                                  % ֡��
cepstL=6;                                        % ��Ƶ���ϴ������Ŀ��
wlen2=wlen/2;               
freq=(0:wlen2-1)*fs/wlen;                        % ����Ƶ���Ƶ�ʿ̶�
u2=u.*hamming(wlen);		                     % �źżӴ�����
U=fft(u2);                                       % ��ʽ(9-2-1)����
U_abs=log(abs(U(1:wlen2)));                      % ��ʽ(9-2-2)����
Cepst=ifft(U_abs);                               % ��ʽ(9-2-3)����
cepst=zeros(1,wlen2);           
cepst(1:cepstL)=Cepst(1:cepstL);                 % ��ʽ(9-2-5)����
cepst(end-cepstL+2:end)=Cepst(end-cepstL+2:end);
spect=real(fft(cepst));                          % ��ʽ(9-2-6)����
[Loc,Val]=findpeaks(spect);                      % Ѱ�ҷ�ֵ
FRMNT=freq(Loc);                                 % ����������Ƶ��
% ��ͼ
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-140)]);
plot(freq,U_abs,'k'); 
hold on; axis([0 4000 -6 2]); grid;
plot(freq,spect,'k','linewidth',2); 
xlabel('Ƶ��/Hz'); ylabel('��ֵ/dB');
title('�ź�Ƶ��,�����ߺ͹����ֵ')
fprintf('%5.2f   %5.2f   %5.2f   %5.2f\n',FRMNT);
for k=1 : 4
    plot(freq(Loc(k)),Val(k),'kO','linewidth',2);
    line([freq(Loc(k)) freq(Loc(k))],[-6 Val(k)],'color','k',...
        'linestyle','-.','linewidth',2);
end
