function pde=ztcont31(Ptk,bde,Pte,Pten,c1)

fn=size(Ptk,2);                     % ȡ��Ptk�ж�����
kl=fn-bde;                          % ȡ�������������ж��ٸ����ݵ�
T0=Pte;                             % �����������һ�����һ�����ݵ�Ļ�������ֵ
T1=Pten;                            % �����������һ�����ڶ������ݵ�Ļ�������ֵ
pde=zeros(1,kl);                    % ��ʼ��pde
for k=1:kl                          % ѭ��
    j=k+bde;
    [mv,ml]=min(abs(T0-Ptk(:,j)));  % ����̾���Ѱ����С��ֵ
    pde(k)=Ptk(ml,j);               % �ҵ�ml;
    TT=Ptk(ml,j);
    if abs(T0-TT)>c1                % ���������ֵ
        TT=2*T0-T1;                 % �����������
        pde(k)=TT;
    end
    T1=T0;
    T0=TT;
end
    
    

