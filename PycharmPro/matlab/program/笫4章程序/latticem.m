function [E,alphal,G,k]=latticem(x,L,p)
% ��ʽ(4-3-22)��ʼ��
    e(:,1)=x;
    b(:,1)=x;
    
% i=1ʱ��ʽ(4-3-22)��ʽ(4-3-25)����
    k(1)=sum(e(p+1:p+L,1).*b(p:p+L-1,1))/sqrt((sum(e(p+1:p+L,1).^2)...
        *sum(b(p:p+L-1,1).^2)));
    alphal(1,1)=k(1);
    btemp=[0 b(:,1)']';
    
% i-1=1ʱ��ʽ(4-3-26)����
    e(1:L+p,2)=e(1:L+p,1)-k(1)*btemp(1:L+p);
    b(1:L+p,2)=btemp(1:L+p)-k(1)*e(1:L+p,1);
    
% i=2~p��ʽ(4-3-22)-(4-3-26)����
    for i=2:p
        k(i)=sum(e(p+1:p+L,i).*b(p:p+L-1,i))/sqrt((sum(e(p+1:p+L,i).^2)...
            *sum(b(p:p+L-1,i).^2)));
        alphal(i,i)=k(i);
        for j=1:i-1
            alphal(j,i)=alphal(j,i-1)-k(i)*alphal(i-j,i-1);
        end
        btemp=[0 b(:,i)']';
        e(1:L+p,i+1)=e(1:L+p,i)-k(i)*btemp(1:L+p);
        b(1:L+p,i+1)=btemp(1:L+p)-k(i)*e(1:L+p,i);
    end
    
% ��ʽ(4-2-8)��ʽ(4-2-12)������С�������
    E=sum(x(p+1:p+L).^2);
    for i=1:p
        E=E*(1-k(i).^2);
    end
    
% ��ʽ(4-2-13)��������ϵ��
    G=sqrt(E);
