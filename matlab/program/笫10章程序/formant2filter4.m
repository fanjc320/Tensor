function [An,Bn]=formant2filter4(F0,Bw,fs)
F0(4)=3500; Bw(4)=100;
for k=1 : 4                        % ��������������һ���̶���ֵ
    f0=F0(k); bw=Bw(k);            % ȡ�������Ƶ�ʺʹ���
    [A,B]=frmnt2coeff3(f0,bw,fs);  % �����ͨ�˲���ϵ��
    An(:,k)=A;                     % �����An��Bn��
    Bn(k)=B;
end
