function pdm=ztcont21(Ptk,bdb,bde,Ptb,Pte,c1)

kl=bde-bdb-1;                       % �������������ж��ٸ����ݵ�
T0=Ptb;                             % ������������ǰһ�ε����һ����Ļ�������
pdm=zeros(1,kl);
Jmp=0; emp=0;
for k=1 : kl                        % ѭ��
    j=k+bdb;
    [mv,ml]=min(abs(T0-Ptk(:,j)));  % ����̾���Ѱ����С��ֵ
    TT=Ptk(ml,j);
    if abs(T0-TT)>c1                % ���������ֵ
        emp=1;                      % ��������
        Jmp=k;
        break                       % ��ֹ��ѭ��
    end
    pdm(k)=Ptk(ml,j);
    T0=Ptk(ml,j);
end

if emp==1                           % ���ڴ�����ֵ���ڲ�
    zxl=kl-Jmp+1;
    deltazx=(Pte-T0)/(zxl+1);
    for k2=1 : zxl                  % �����ڲ�
        pdm(k2)=T0+k2*deltazx;
    end
end

    
    

