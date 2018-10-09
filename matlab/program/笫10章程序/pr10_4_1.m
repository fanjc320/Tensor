%
% pr10_4_1  
clear all; clc; close all;

filedir=[];                               % ���������ļ���·��
filename='colorcloud.wav';                % ���������ļ�������
fle=[filedir filename]                    % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                     % ��ȡ�ļ�
xx=xx-mean(xx);                           % ȥ��ֱ������
x=xx/max(abs(xx));                        % ��һ��
N=length(x);                              % ���ݳ���
time=(0:N-1)/fs;                          % ʱ��̶�
wlen=240;                                 % ֡��
inc=80;                                   % ֡��
overlap=wlen-inc;                         % �ص�����
tempr1=(0:overlap-1)'/overlap;            % б���Ǵ�����w1
tempr2=(overlap-1:-1:0)'/overlap;         % б���Ǵ�����w2
n2=1:wlen/2+1;                            % ��Ƶ�ʵ��±�ֵ
wind=hanning(wlen);                       % ������
X=enframe(x,wind,inc)';                   % ��֡
fn=size(X,2);                             % ֡��
T1=0.1; r2=0.5;                           % �˵������
miniL=10;                                 % �л������֡��
mnlong=5;                                 % Ԫ���������֡��
ThrC=[10 15];                             % ��ֵ
p=12;                                     % LPC�״�
frameTime=frame2time(fn,wlen,inc,fs);     % ����ÿ֡��ʱ��̶�
for i=1 : fn                              % ����ÿ֡������Ԥ��ϵ��������
    u=X(:,i);
    [ar,g]=lpc(u,p);
    AR_coeff(:,i)=ar;
    Gain(i)=g;
end
% �������
[Dpitch,Dfreq,Ef,SF,voiceseg,vosl,vseg,vsl,T2]=...
   Ext_F0ztms(xx,fs,wlen,inc,T1,r2,miniL,mnlong,ThrC,0);

tal=0;                                    % ��ʼ��ǰ�����
zint=zeros(p,1);
for i=1:fn; 
    ai=AR_coeff(:,i);                     % ��ȡ��i֡��Ԥ��ϵ��
    sigma_square=Gain(i);                 % ��ȡ��i֡������ϵ��
    sigma=sqrt(sigma_square);
    
    if SF(i)==0                           % �޻�֡
        excitation=randn(wlen,1);         % ����������
        [synt_frame,zint]=filter(sigma,ai,excitation,zint); % �ð������ϳ�����
    else                                  % �л�֡
        PT=round(Dpitch(i));              % ȡ����ֵ
        exc_syn1 =zeros(wlen+tal,1);      % ��ʼ�����巢����
        exc_syn1(mod(1:tal+wlen,PT)==0) = 1;  % �ڻ������ڵ�λ�ò������壬��ֵΪ1
        exc_syn2=exc_syn1(tal+1:tal+inc); % ����֡��inc�������������
        index=find(exc_syn2==1);
        excitation=exc_syn1(tal+1:tal+wlen);% ��һ֡�ļ�������Դ
        
        if isempty(index)                 % ֡��inc������û������
            tal=tal+inc;                  % ������һ֡��ǰ�����
        else                              % ֡��inc������������
            eal=length(index);            % �����м�������
            tal=inc-index(eal);           % ������һ֡��ǰ�����
        end
        gain=sigma/sqrt(1/PT);            % ����
        [synt_frame,zint]=filter(gain, ai,excitation,zint); % ������ϳ�����
    end
        if i==1                           % ��Ϊ��1֡
            output=synt_frame;            % ����Ҫ�ص����,�����ϳ�����
        else
            M=length(output);             % �����Ա����ص���Ӵ���ϳ�����
            output=[output(1:M-overlap); output(M-overlap+1:M).*tempr2+...
                synt_frame(1:overlap).*tempr1; synt_frame(overlap+1:wlen)];
        end
end
ol=length(output);                        % �����output�ӳ����������ź�xx�ȳ�
if ol<N
    output1=[output; zeros(N-ol,1)];
else
    output1=output(1:N);
end
bn=[0.964775   -3.858862   5.788174   -3.858862   0.964775]; % �˲���ϵ��
an=[1.000000   -3.928040   5.786934   -3.789685   0.930791];
output=filter(bn,an,output1);             % ��ͨ�˲�
output=output/max(abs(output));           % ��ֵ��һ

% ͨ����������,�Ƚ�ԭʼ�����ͺϳ�����
wavplay(x,fs);
pause(1)
wavplay(output,fs);
%��ͼ
figure(1)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)+85)])
subplot 411; plot(time,xx,'k'); axis([0 max(time) -1 1.2]);
xlabel('(a)'); 
title('�źŲ���'); ylabel('��ֵ')
subplot 412; plot(frameTime,Ef,'k'); hold on
axis([0 max(time) 0 1.2]); plot(frameTime,T2,'k','linewidth',2);
line([0 max(time)],[T1 T1],'color','k','linestyle','-.');
title('���ر�ͼ'); axis([0 max(time) 0 1.2]); ylabel('��ֵ')
xlabel('(b)'); 
text(3.2,T1+0.05,'T1');
for k=1 : vsl
        line([frameTime(vseg(k).begin) frameTime(vseg(k).begin)],...
        [0 1.2],'color','k','Linestyle','-');
        line([frameTime(vseg(k).end) frameTime(vseg(k).end)],...
        [0 1.2],'color','k','Linestyle','--');
    if k==vsl
        Tx=T2(floor((vseg(k).begin+vseg(k).end)/2));
    end
end
text(2.65,Tx+0.05,'T2');
subplot 413; plot(frameTime,Dpitch,'k'); 
axis([0 max(time) 0 110]);title('��������');ylabel('����ֵ')
xlabel( '(c)'); 
subplot 414; plot(frameTime,Dfreq,'k'); 
axis([0 max(time) 0 250]);title('����Ƶ��');ylabel('Ƶ��/Hz')
xlabel(['ʱ��/s' 10 '(d)']); 

figure(2)
subplot 211; plot(time,x,'k'); title('ԭʼ��������');
axis([0 max(time) -1 1.1]); xlabel('ʱ��/s'); ylabel('��ֵ')
subplot 212; plot(time,output,'k');  title('�ϳ���������');
axis([0 max(time) -1 1.1]); xlabel('ʱ��/s'); ylabel('��ֵ')
