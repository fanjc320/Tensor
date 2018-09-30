function z=conv_ovladd1(x,h,L)
% ���ص���ӷ������� 
x=x(:)';                                % ��xת����һ��
NN=length(x);                           % ����x��
N=length(h);                            % ����h��
N1=L-N+1;                               % ��x�ֶεĳ��� 
x1=[x zeros(1,L)];
H=fft(h,L);                             % ��h��FFTΪH
J=fix((NN+L)/N1);                       % ��ֿ����
y=zeros(1,NN+2*L);                      % ��ʼ��
for k=1 : J                             % ��ÿ�δ���
    xx=zeros(1,L);
    MI=(k-1)*N1;                        % ��i����x�ϵĿ�ʼλ��-1
    nn=1:N1;
    xx(nn)=x1(nn+MI);                   % ȡһ��xi
    XX=fft(xx,L);                       % �� FFT
    YY=XX.*H;                           % ��˽��о��
    yy=ifft(YY,L);                      % ��FFT��任���yi
% ���ڶμ��ص����
    if k==1                             % ��1�鲻��Ҫ���ص����                          
        for j=1 : L
            y(j)=y(j)+real(yy(j));
        end
    elseif k==J                         % ���һ��ֻ��N1�����ݵ��ص����
        for j=1 : N1
            y(MI+j)=y(MI+j)+real(yy(j));                
        end
    else        
        for j=1 : L                     % �ӵ�2�鿪ʼÿ�鶼Ҫ���ص����
            y(MI+j)=y(MI+j)+real(yy(j));                
        end
    end
end
nn=floor(N/2);
z=y(nn+1:NN+nn);                        % �����ӳ���,ȡ���������x�ȳ�
