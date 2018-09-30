%--------------------------------------------------------------------------
% main script to do pitch and time scale modification of speech signal
%--------------------------------------------------------------------------
global config;                          % ȫ�ֱ���config
config.pitchScale           = 1.3;	    % ���û�Ƶ�޸�����
config.timeScale            = 0.8;	    % ����ʱ���޸�����
config.resamplingScale      = 1;		% �ز���
config.reconstruct          = 0;		% ���Ϊ����е�ͨ���ع�
config.displayPitchMarks    = 0;		% ���Ϊ�潫��ʾ���������ע
config.playWavOut           = 1;		% ���Ϊ�潫�ڼ�����ϲ��źϳɵ�����
config.cutOffFreq           = 900;	    % ��ͨ�˲����Ľ�ֹƵ��
config.fileIn               = '..\waves\m2.wav';% �����ļ�·�����ļ���
config.fileOut              = '..\waves\syn.wav';% ����ļ�·�����ļ���

global data;            % ȫ�ֱ���config,�ȳ�ʼ��
data.waveOut = [];		% ����Ƶ�޸����Ӻ�ʱ���޸����ӵ�����ĺϳ��������
data.pitchMarks = [];	% ���������źŵĻ��������ע
data.Candidates = [];	% ���������źŻ��������ע�ĺ�ѡ����

[WaveIn, fs] = wavread(config.fileIn);	   % ���������ļ�
WaveIn = WaveIn - mean(WaveIn); 		   % ���ֱ������

[LowPass] = LowPassFilter(WaveIn, fs, config.cutOffFreq); % ���źŽ��е�ͨ�˲�
PitchContour = PitchEstimation(LowPass, fs);% ��������źŵĻ�������
PitchMarking(WaveIn, PitchContour, fs);		% ���л��������ע��PSOLA�ϳ�
wavwrite(data.waveOut, fs, config.fileOut);	% �Ѻϳ�����д��ָ���ļ�

if config.playWavOut
    wavplay(data.waveOut, fs);
end

if config.displayPitchMarks
    PlotPitchMarks(WaveIn, data.candidates, data.pitchMarks, PitchContour); %show the pitch marks
end