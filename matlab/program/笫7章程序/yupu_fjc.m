%matlab 2016a
%需要先安装语音处理工具箱(matlab_voicebox)


[Y,FS]=audioread('qiannvyouhun_lang_left.wav');
    %Y为读到的双声道数据
    %FS为采样频率

Y1 = Y(:,1);        %Y为双声道数据，取第2通道
% Y2 = Y(:,0);      % error
plot(Y1);           %画Y1波形图
grid on;
specgram(Y1,2048,44100,2048,1536);
    %Y1为波形数据
    %FFT帧长2048点(在44100Hz频率时约为46ms)
    %采样频率44.1KHz
    %加窗长度，一般与帧长相等
    %帧重叠长度，此处取为帧长的3/4