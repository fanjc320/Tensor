function [Pm,vsegch,vsegchlong]=Ext_corrshtp_test1(y,sign,TT1,XL,ixb,...
    lmax,lmin,ThrC)
wlen=size(y,1);
Emp=0;                                    % ��ʼ����Emp
c1=ThrC(1); c2=ThrC(2);
figure(50);
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1), pos(2)-100,pos(3),pos(4)]);
% ѭ��XL��ǰ����������������ȡ�������ڳ���ֵ
for k=1 : XL                              
    j=ixb+sign*k;                         % ����֡�ı��
    u=y(:,j);                             % ȡ��һ֡�ź�
    ru=xcorr(u,'coeff');                  % ��������غ���
    ru=ru(wlen:end);                      % ȡ���ӳ�������
    figure(50)
    subplot 211; plot(u,'k');
    title(['��' num2str(k) '֡����']);
     xlabel('(a)'); ylabel('��ֵ');
    subplot 212;
    plot(ru,'k'); grid; xlim([0 150]);
    title(['��' num2str(k) '֡����غ���R']);
    xlabel(['������' 10 '(b)']); ylabel('��ֵ');
    [Sv,Kv]=findmaxesm3(ru,lmax,lmin);    % ��ȡ��������ֵ����ֵ��λ��
    Ptk(:,k)=Kv';
    fprintf('%4d   %4d   %4d   %4d\n',k,Kv);
    pause
end

figure(51)
    pos = get(gcf,'Position');
    set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
    plot(1:XL,Ptk(1,1:XL),'ko-',1:XL,Ptk(2,1:XL),'k*-',1:XL,...
        Ptk(3,1:XL),'k+-'); grid;
    xlabel('������'); ylabel('��������')
    title('��������������ں�ѡ����ͼ');
    Pm=Ptk(1,:);
    vsegch=0; vsegchlong=0;

