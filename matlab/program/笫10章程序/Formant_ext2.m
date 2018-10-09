function Fmap=Formant_ext2(x,wlen,inc,fs,SF,Doption)

y=filter([1 -.99],1,x);                               % Ԥ����
xy=enframe(y,wlen,inc)';                              % ��֡
fn=size(xy,2);                                        % ��֡��
Nx=length(y);                                         % ���ݳ���
time=(0:Nx-1)/fs;                                     % ʱ��̶�
frameTime=frame2time(fn,wlen,inc,fs);                 % ÿ֡��Ӧ��ʱ��̶�
Msf=repmat(SF',1,3);                                  % ��SF��չΪ3��fn������
Ftmp_map=zeros(fn,3);
warning off
Fsamps = 256;                                         % ����Ƶ�򳤶�
Tsamps= fn;                                           % ����ʱ�򳤶�
ct = 0;

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

Fmap=Ftmp_map';
findex=find(Fmap==nan);
Fmap(findex)=0;
% ��ͼ
if Doption
figure(99)
nfft=512;
d=stftms(y,wlen,nfft,inc);
W2=1+nfft/2;
n2=1:W2;
freq=(n2-1)*fs/nfft;
Fmap1=Msf.*Ftmp_map;                                  % ֻȡ�л��ε�����
findex=find(Fmap1==0);                                % �������ֵΪ0 ,��Ϊnan
Fmapd=Fmap1;
Fmapd(findex)=nan;
imagesc(frameTime,freq,abs(d(n2,:)));  axis xy
m = 64; LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0]; Black = [0.5 0.7 1];
Colors = [LightYellow; MidRed; Black];
colormap(SpecColorMap(m,Colors)); hold on
plot(frameTime,Fmapd,'w');                             % �����Ϲ����Ƶ������
title('������ͼ�ϱ�������Ƶ��');
xlabel('ʱ��/s'); ylabel('Ƶ��/Hz')
end