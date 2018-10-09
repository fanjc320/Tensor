function [NoiseFlag, SpeechFlag, NoiseCounter, Dist]=...
    vadc(signal,noise,NoiseCounter,NoiseMargin,Hangover)
% ����ȱʡֵ
if nargin<4
    NoiseMargin=3;
end
if nargin<5
    Hangover=8;
end
if nargin<3
    NoiseCounter=0;
end
    
FreqResol=length(signal);               % �źų���
% ��֡������ֵ����Ƶ�׺���������Ƶ��֮��ֵ
SpectralDist= 20*(log10(signal)-log10(noise));
SpectralDist(find(SpectralDist<0))=0;   % Ѱ�Ҳ�ֵС��0ֵ��Ϊ0
 
Dist=mean(SpectralDist);                % ��ƽ�����Dist
if (Dist < NoiseMargin)                 % Dist �Ƿ�С�� NoiseMargin
    NoiseFlag=1;                        % �ǣ�NoiseFlag��Ϊ1
    NoiseCounter=NoiseCounter+1;        % NoiseCounter��1
else
    NoiseFlag=0;                        % ��NoiseFlag��Ϊ0
    NoiseCounter=0;                     % NoiseCounter����
end
 
% �Ƿ�NoiseCounter�ѳ����޻�����С����Hangover
if (NoiseCounter > Hangover)            % NoiseCounter����Hangover
    SpeechFlag=0;                       % �ǣ�SpeechFlagΪ0
else 
    SpeechFlag=1;                       % ��SpeechFlagΪ1
end 
