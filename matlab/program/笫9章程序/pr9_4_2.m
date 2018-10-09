%
% pr9_4_2 
clear all; clc; close all;

filedir=[];                                           % ���������ļ�·��
filename='vowels8.wav';                               % �����ļ���
fle=[filedir filename]
[x, fs, nbits]=wavread(fle);                          % ���������ļ�
y=filter([1 -.99],1,x);                               % Ԥ����
wlen=200;                                             % ����֡��
inc=80;                                               % ����֡��
xy=enframe(y,wlen,inc)';                              % ��֡
fn=size(xy,2);                                        % ��֡��
Nx=length(y);                                         % ���ݳ���
time=(0:Nx-1)/fs;                                     % ʱ��̶�
frameTime=frame2time(fn,wlen,inc,fs);                 % ÿ֡��Ӧ��ʱ��̶�
T1=0.1;                                               % �ж��л��ε����ر���ֵ
miniL=10;                                             % �л��ε���С֡��
[voiceseg,vosl,SF,Ef]=pitch_vad1(xy,fn,T1,miniL);     % �˵���
Msf=repmat(SF',1,3);                                  % ��SF��չΪ3��fn������
Fsamps = 256;                                         % ����Ƶ�򳤶�
Tsamps= fn;                                           % ����ʱ�򳤶�
ct = 0;
warning off
numiter = 10;                                         % ѭ��10��,
iv=2.^(10-10*exp(-linspace(2,10,numiter)/1.9));       % ��0��1024֮������10����
for j=1:numiter
    i=iv(j);                                          
    iy=fix(length(y)/round(i));                       % ����֡��
    [ft1] = seekfmts1(y,iy,fs,10);                    % ��֪֡����ȡ�����
    ct = ct+1;
    ft2(:,:,ct) = interp1(linspace(1,length(y),iy)',...% ��ft1�����ڲ�ΪTsamps��
    Fsamps*ft1',linspace(1,length(y),Tsamps)')';
end
ft3 = squeeze(nanmean(permute(ft2,[3 2 1])));         % �������к�ƽ������
tmap = repmat([1:Tsamps]',1,3);
Fmap=ones(size(tmap))*nan;                            % ��ʼ��
idx = find(~isnan(sum(ft3,2)));                       % Ѱ�ҷ�nan��λ��
fmap = ft3(idx,:);                                    % ��ŷ�nan������

[b,a] = butter(9,0.1);                                % ��Ƶ�ͨ�˲���
fmap1 = round(filtfilt(b,a,fmap));                    % ��ͨ�˲�
fmap2 = (fs/2)*(fmap1/256);                           % �ָ���ʵ��Ƶ��
Ftmp_map(idx,:)=fmap2;                                % �������

Fmap1=Msf.*Ftmp_map;                                  % ֻȡ�л��ε�����
findex=find(Fmap1==0);                                % �������ֵΪ0 ,��Ϊnan
Fmap=Fmap1;
Fmap(findex)=nan;

nfft=512;                                             % ��������ͼ
d=stftms(y,wlen,nfft,inc);
W2=1+nfft/2;
n2=1:W2;
freq=(n2-1)*fs/nfft;
% ��ͼ
figure(1)                                             % ���źŵĲ���ͼ�����ر�ͼ
subplot 211; plot(time,x,'k');
title('\a-i-u\����Ԫ���������Ĳ���ͼ');
xlabel('ʱ��/s'); ylabel('��ֵ'); xlim([0 max(time)]);
subplot 212; plot(frameTime,Ef,'k'); hold on
line([0 max(time)],[T1 T1],'color','k','linestyle','--');
title('��һ�������ر�ͼ'); axis([0 max(time) 0 1.2]);
xlabel('ʱ��/s'); ylabel('��ֵ')
for k=1 : vosl
    in1=voiceseg(k).begin;
    in2=voiceseg(k).end;
    it1=frameTime(in1);
    it2=frameTime(in2);
    line([it1 it1],[0 1.2],'color','k','linestyle','-.');
    line([it2 it2],[0 1.2],'color','k','linestyle','-.');
end

figure(2)                                             % �������źŵ�����ͼ
imagesc(frameTime,freq,abs(d(n2,:)));  axis xy
m = 64; LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0]; Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors)); hold on
plot(frameTime,Fmap,'w');                             % �����Ϲ����Ƶ�ʹ켣
title('������ͼ�ϱ�������Ƶ��');
xlabel('ʱ��/s'); ylabel('Ƶ��/Hz')
