function pdb=ztcont11(Ptk,bdb,Ptb,Ptbp,c1)

kl=bdb-1;                           % ȡ�����������ڵ�һ����ĵ�һ��λ��
T0=Ptb;                             % �������ڵ�һ�ε�һ�����ݵ�Ļ�������ֵ
T1=Ptbp;                            % �������ڵ�һ�εڶ������ݵ�Ļ�������ֵ
pdb=zeros(1,kl);                    % ��ʼ��pdb
for k=kl:-1:1                       % ѭ��
    [mv,ml]=min(abs(T0-Ptk(:,k)));  % ����̾���Ѱ����С��ֵ
    pdb(k)=Ptk(ml,k);               % �ҵ�ml
    TT=Ptk(ml,k);
    if abs(T0-TT)>c1                % ���������ֵ
        TT=2*T0-T1;                 % ��ǰ��������
        pdb(k)=TT;
    end
    T1=T0;
    T0=TT;
end

    

