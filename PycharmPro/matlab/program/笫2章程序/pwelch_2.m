function [Pxx] = pwelch_2(x, nwind, noverlap, w_nwind, w_noverlap, nfft)
% �����ʱ�������ܶȺ���
% x���źţ�nwind��ÿ֡���ȣ�noverlap��ÿ֡�ص���������
% w_nwind��ÿ�εĴ�����������Ӧ�Ķγ���
% w_noverlap��ÿ��֮����ص�����������nfft��FFT�ĳ���

x=x(:);
inc=nwind-noverlap;       % ����֡��
X=enframe(x,nwind,inc)';  % ��֡
frameNum=size(X,2);       % ����֡��
%��pwelch������ÿ֡���㹦�����ܶȺ���
for k=1 : frameNum
    Pxx(:,k)=pwelch(X(:,k),w_nwind,w_noverlap,nfft);
end



