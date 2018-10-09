%
% pr10_5_3 
clear all; clc; close all;

filedir=[];                               % ���������ļ���·��
filename='colorcloud.wav';                % ���������ļ�������
fle=[filedir filename]                    % ����·�����ļ������ַ���
[xx,fs]=wavread(fle);                     % ��ȡ�ļ�
xx=xx-mean(xx);                           % ȥ��ֱ������
x1=xx/max(abs(xx));                       % ��һ��
x=filter([1 -.99],1,x1);                  % Ԥ����
N=length(x);                              % ���ݳ���
time=(0:N-1)/fs;                          % �źŵ�ʱ��̶�
wlen=240;                                 % ֡��
inc=80;                                   % ֡��
overlap=wlen-inc;                         % �ص�����
tempr1=(0:overlap-1)'/overlap;            % б���Ǵ�����w1
tempr2=(overlap-1:-1:0)'/overlap;         % б���Ǵ�����w2
n2=1:wlen/2+1;                            % ��Ƶ�ʵ��±�ֵ
wind=hanning(wlen);                       % ������
X=enframe(x,wlen,inc)';                   % ��֡
fn=size(X,2);                             % ֡��
Etemp=sum(X.*X);                          % ����ÿ֡������
Etemp=Etemp/max(Etemp);                   % ������һ��
T1=0.1; r2=0.5;                           % �˵������
miniL=10;                                 % �л������֡��
mnlong=5;                                 % Ԫ���������֡��
ThrC=[10 15];                             % ��ֵ
p=12;                                     % LPC�״�
frameTime=frame2time(fn,wlen,inc,fs);     % ÿ֡��Ӧ��ʱ��̶�
Doption=0;                                

% ������-���취�������
[Dpitch,Dfreq,Ef,SF,voiceseg,vosl,vseg,vsl,T2]=...
   Ext_F0ztms(x1,fs,wlen,inc,T1,r2,miniL,mnlong,ThrC,Doption);

%% �������ȡ
Frmt=Formant_ext2(x,wlen,inc,fs,SF,Doption);
Bwm=[150 200 250];                        % ���ù̶�����                    
Bw=repmat(Bwm',1,fn);

%% �����ϳ�
zint=zeros(2,4);                          % ��ʼ��
tal=0;
for i=1 : fn
    yf=Frmt(:,i);                         % ȡ��i֡�����������Ƶ�ʺʹ���
    bw=Bw(:,i);
    [an,bn]=formant2filter4(yf,bw,fs);    % ת�����ĸ������˲���ϵ��
    synt_frame=zeros(wlen,1);
    
    if SF(i)==0                           % �޻�֡
        excitation=randn(wlen,1);         % ����������
        for k=1 : 4                       % ���ĸ��˲�����������
            An=an(:,k);
            Bn=bn(k);
            [out(:,k),zint(:,k)]=filter(Bn(1),An,excitation,zint(:,k));
            synt_frame=synt_frame+out(:,k); % �ĸ��˲������������һ��
        end
    else                                  % �л�֡
        PT=round(Dpitch(i));              % ȡ����ֵ
        exc_syn1 =zeros(wlen+tal,1);      % ��ʼ�����巢����
        exc_syn1(mod(1:tal+wlen,PT)==0)=1;% �ڻ������ڵ�λ�ò������壬��ֵΪ1
        exc_syn2=exc_syn1(tal+1:tal+inc); % ����֡��inc�����ڵ��������
        index=find(exc_syn2==1);
        excitation=exc_syn1(tal+1:tal+wlen);% ��һ֡�ļ�������Դ
        
        if isempty(index)                 % ֡��inc������û������
            tal=tal+inc;                  % ������һ֡��ǰ�����
        else                              % ֡��inc������������
            eal=length(index);            % �����м�������
            tal=inc-index(eal);           % ������һ֡��ǰ�����
        end
        for k=1 : 4                       % ���ĸ��˲�����������
            An=an(:,k);
            Bn=bn(k);
            [out(:,k),zint(:,k)]=filter(Bn(1),An,excitation,zint(:,k));
            synt_frame=synt_frame+out(:,k); % �ĸ��˲������������һ��
        end
    end
    Et=sum(synt_frame.*synt_frame);       % �����������ϳ�����
    rt=Etemp(i)/Et;
    synt_frame=sqrt(rt)*synt_frame;
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
    output=[output; zeros(N-ol,1)];
end
% �ϳ�����ͨ����ͨ�˲���
out1=output;
out2=filter(1,[1 -0.99],out1);
b=[0.964775   -3.858862   5.788174   -3.858862   0.964775];
a=[1.000000   -3.928040   5.786934   -3.789685   0.930791];
output=filter(b,a,out2);
output=output/max(abs(output));
% ͨ����������,�Ƚ�ԭʼ�����ͺϳ�����
wavplay(xx,fs);
pause(1)
wavplay(output,fs);
%% ��ͼ
figure(1)
figure(1)
pos = get(gcf,'Position');
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)+85)])
subplot 411; plot(time,x1,'k'); axis([0 max(time) -1 1.1]);
title('�źŲ���'); ylabel('��ֵ')
subplot 412; plot(frameTime,Ef,'k'); hold on
axis([0 max(time) 0 1.2]); plot(frameTime,T2,'k','linewidth',2);
line([0 max(time)],[T1 T1],'color','k','linestyle','-.');
title('���ر�ͼ'); axis([0 max(time) 0 1.2]); ylabel('��ֵ')
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
axis([0 max(time) 0 110]);title('��������'); ylabel('����ֵ')
subplot 414; plot(frameTime,Dfreq,'k'); 
axis([0 max(time) 0 250]);title('����Ƶ��'); ylabel('Ƶ��/Hz')
xlabel('ʱ��/s'); 

figure(2)
subplot 211; plot(time,x1,'k'); title('ԭʼ��������');
axis([0 max(time) -1 1.1]); xlabel('ʱ��/s'); ylabel('��ֵ')
subplot 212; plot(time,output,'k');  title('�ϳ���������');
axis([0 max(time) -1 1.1]); xlabel('ʱ��/s'); ylabel('��ֵ')

figure(3)
out1=out1/max(out1);
subplot 311; plot(time,out1,'k');
title('ͨ���˲���֮ǰ�Ĳ���out1')
ylabel('��ֵ'); ylim([-0.5 1]); xlabel('(a)');
out2=out2/max(out2);
subplot 312; plot(time,out2,'k');
title('ͨ����ͨ�˲�����Ĳ���out2')
ylabel('��ֵ'); ylim([-0.5 1]); xlabel('(b)');
subplot 313; plot(time,output,'k');
title('ͨ����ͨ�˲�����Ĳ���output')
xlabel(['ʱ��/s' 10 '(c)']); ylabel('��ֵ')

