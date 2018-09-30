function [voiceseg,vosl,vseg,vsl,T2,Bth,SF,Ef]=pitch_vads(y,fn,T1,r2,miniL,mnlong)
if nargin<6, mnlong=10; end
if nargin<5, miniL=10; end
if size(y,2)~=fn, y=y'; end                   % ��yת��Ϊÿ�����ݱ�ʾһ֡�����ź�
wlen=size(y,1);                               % ȡ��֡��
for i=1:fn
    Sp = abs(fft(y(:,i)));                    % FFTȡ��ֵ
    Sp = Sp(1:wlen/2+1);	                  % ֻȡ��Ƶ�ʲ���
    Esum(i) = sum(Sp.*Sp);                    % ��������ֵ
    prob = Sp/(sum(Sp));	                  % �������
    H(i) = -sum(prob.*log(prob+eps));         % ������ֵ
end
hindex=find(H<0.1);
H(hindex)=max(H);
Ef=sqrt(1 + abs(Esum./H));                    % �������ر�
Ef=Ef/max(Ef);                                % ��һ��

zindex=find(Ef>=T1);                          % Ѱ��Ef�д���T1�Ĳ���
zseg=findSegment(zindex);                     % �����˵�����ε���Ϣ
zsl=length(zseg);                             % ��������
j=0;
SF=zeros(1,fn);
for k=1 : zsl                                 % �ڴ���T1���޳�С��miniL�Ĳ���
    if zseg(k).duration>=miniL
        j=j+1;
        in1=zseg(k).begin;
        in2=zseg(k).end;
        voiceseg(j).begin=in1;
        voiceseg(j).end=in2;
        voiceseg(j).duration=zseg(k).duration;
        SF(in1:in2)=1;                        % ����SF
    end
end
vosl=length(voiceseg);                        % �л��εĶ���   

T2=zeros(1,fn);
j=0;
for k=1 : vosl                                % ��ÿһ���л�����Ѱ��Ԫ������
    inx1=voiceseg(k).begin;
    inx2=voiceseg(k).end;
    Eff=Ef(inx1:inx2);                        % ȡһ���л��ε����ر�
    Emax=max(Eff);                            % ������л��������رȵ����ֵ
    Et=Emax*r2;                               % ����ڶ�����ֵT2
    if Et<T1, Et=T1; end
    T2(inx1:inx2)=Et;
    zindex=find(Eff>=Et);                     % ���л�����Ѱ��Ef����T2�Ĳ���
    if ~isempty(zindex)
        zseg=findSegment(zindex);
        zsl=length(zseg);
            
        for m=1 : zsl
            if zseg(m).duration>=mnlong       % ֻ�������ȴ���mnlong��Ԫ������
                j=j+1;
                vseg(j).begin=zseg(m).begin+inx1-1;
                vseg(j).end=zseg(m).end+inx1-1;
                vseg(j).duration=zseg(m).duration;
                Bth(j)=k;                     % ���ø�Ԫ������������һ���л���
                
            end
        end
    end
end
vsl=length(vseg);                             % ���Ԫ��������� 




