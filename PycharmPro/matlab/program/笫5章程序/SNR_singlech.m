function snr=SNR_singlech(I,In)
% ������������źŵ������
% I �Ǵ������ź�
% In �Ǵ���������ź�
% ����ȼ��㹫ʽ��
% snr=10*log10(Esignal/Enoise)
I=I(:)';                             % ������תΪһ��
In=In(:)';
Ps=sum((I-mean(I)).^2);              % �źŵ�����
Pn=sum((I-In).^2);                   % ����������
snr=10*log10(Ps/Pn);                 % �źŵ�����������������֮�ȣ�����ֱ�ֵ