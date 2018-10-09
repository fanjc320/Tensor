% EMD ���㾭��ģʽ�ֽ�
%
%
%   �﷨
%
%
% IMF = EMD(X)
% IMF = EMD(X,...,'Option_name',Option_value,...)
% IMF = EMD(X,OPTS)
% [IMF,ORT,NB_ITERATIONS] = EMD(...)
%
%
%   ����
%
%
% IMF = EMD(X) X��һ��ʵʸ�������㷽���ο�[1]��������������IMF�����У�ÿһ�а���һ��IMF������
% ���һ���ǲ��������Ĭ�ϵ�ֹͣ��������[2]��
%
%   ��ÿһ����, mean_amplitude < THRESHOLD2*envelope_amplitude ��ע��ƽ�������������ȵı�ֵС������2��
%   &
%   mean of boolean array {(mean_amplitude)/(envelope_amplitude) > THRESHOLD} < TOLERANCE 
%  ��ע��ƽ�������������ȱ�ֵ�������޵ĵ���ռ�ź��ܵ����еı���С�����ޣ�
%   &
%   |#zeros-#extrema|<=1 ��ע�������ͼ�ֵ�������Ȼ������1��
%
% ���� mean_amplitude = abs(envelope_max+envelope_min)/2 ��ע��ƽ�����ȵ������°����໥������в��һ��ľ���ֵ�������������0��
% �� envelope_amplitude = abs(envelope_max-envelope_min)/2 ��ע��������ȵ������°�����Ծ����һ�룬��������������°��籾��ľ���ֵ��
% 
% IMF = EMD(X) X��һ��ʵʸ�������㷽���ο�[3]��������������IMF�����У�ÿһ�а���һ��IMF������
% ���һ���ǲ��������Ĭ�ϵ�ֹͣ��������[2]��
%
%   ��ÿһ����, mean_amplitude < THRESHOLD2*envelope_amplitude��ע��ƽ�������������ȵı�ֵС������2��
%   &
%   mean of boolean array {(mean_amplitude)/(envelope_amplitude) > THRESHOLD} < TOLERANCE
%  ��ע��ƽ�������������ȱ�ֵ�������޵ĵ���ռ�ź��ܵ����еı���С�����ޣ�
%
% ����ƽ�����ȺͰ�����ȵĶ�����ǰ��ʵ�����������
%
% IMF = EMD(X,...,'Option_name',Option_value,...) �����ض���������ѡ�
%
% IMF = EMD(X,OPTS) ��ǰ��ȼۣ�ֻ������OPTS��һ���ṹ�壬����ÿһ����������Ӧ��ѡ������һ�¡�
%
% [IMF,ORT,NB_ITERATIONS] = EMD(...) ��������ָ��
%                       ________
%         _  |IMF(i,:).*IMF(j,:)|
%   ORT = \ _____________________
%         /
%         -       || X ||^2        i~=j
%
% ����ȡÿһ��IMFʱ���еĵ���������
%
%
%   ѡ��
%
%
%  ֹͣ����ѡ��:
%
% STOP: ֹͣ���� [THRESHOLD,THRESHOLD2,TOLERANCE]
% �������ʸ������С�� 3, ֻ�е�һ��������Ч��������������Ĭ��ֵ
% Ĭ��ֵ: [0.05,0.5,0.05]
%
% FIX (int): ȡ��Ĭ�ϵ�ֹͣ����������  ָ�������ĵ���
%
% FIX_H (int): ȡ��Ĭ�ϵ�ֹͣ����������  ָ�������ĵ������������� |#zeros-#extrema|<=1 ��ֹͣ�������ο� [4]
%
%  �� EMD ѡ��:
%
% COMPLEX_VERSION: ѡ�� EMD �㷨(�ο�[3])
% COMPLEX_VERSION = 1: "algorithm 1"
% COMPLEX_VERSION = 2: "algorithm 2" (default)
% 
% NDIRS: �������ķ������ (Ĭ�� 4)
% rem: ʵ�ʷ������ (���� [3]) �� 2*NDIRS
% 
%  ����ѡ��:
%
% T: ����ʱ�� (����ʸ��) (Ĭ��: 1:length(x))
%
% MAXITERATIONS: ��ȡÿ��IMF�У����õ�������������Ĭ�ϣ�2000��
%
% MAXMODES: ��ȡIMFs�������� (Ĭ��: Inf)
%
% DISPLAY: �������1��ÿ����һ���Զ���ͣ��pause��
% �������2���������̲���ͣ (����ģʽ)
% rem: �������Ǹ�����ʱ����ʾ�����Զ�ȡ��
%
% INTERP: ��ֵ���� 'linear', 'cubic', 'pchip' or 'spline' (Ĭ��)
% ����� interp1 �ĵ�
%
% MASK: ���� masking �źţ��ο� [5]
%
%
%   ����
%
%
% X = rand(1,512);
%
% IMF = emd(X);
%
% IMF = emd(X,'STOP',[0.1,0.5,0.05],'MAXITERATIONS',100);
%
% T = linspace(0,20,1e3);
% X = 2*exp(i*T)+exp(3*i*T)+.5*T;
% IMF = emd(X,'T',T);
%
% OPTIONS.DISLPAY = 1;
% OPTIONS.FIX = 10;
% OPTIONS.MAXMODES = 3;
% [IMF,ORT,NBITS] = emd(X,OPTIONS);
%
%
%   �ο�����
%
%
% [1] N. E. Huang et al., "The empirical mode decomposition and the
% Hilbert spectrum for non-linear and non stationary time series analysis",
% Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
%
% [2] G. Rilling, P. Flandrin and P. Goncalves
% "On Empirical Mode Decomposition and its algorithms",
% IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing
% NSIP-03, Grado (I), June 2003
%
% [3] G. Rilling, P. Flandrin, P. Goncalves and J. M. Lilly.,
% "Bivariate Empirical Mode Decomposition",
% Signal Processing Letters (submitted)
%
% [4] N. E. Huang et al., "A confidence limit for the Empirical Mode
% Decomposition and Hilbert spectral analysis",
% Proc. Royal Soc. London A, Vol. 459, pp. 2317-2345, 2003
%
% [5] R. Deering and J. F. Kaiser, "The use of a masking signal to improve 
% empirical mode decomposition", ICASSP 2005
%
%
% Ҳ���Բο�
%  emd_visu (visualization),
%  emdc, emdc_fix (fast implementations of EMD),
%  cemdc, cemdc_fix, cemdc2, cemdc2_fix (fast implementations of bivariate EMD),
%  hhspectrum (Hilbert-Huang spectrum)
%
%
% G. Rilling, ����޸�: 3.2007
% gabriel.rilling@ens-lyon.fr
% 
% ���룺xray	11.2007

function [imf,ort,nbits] = emd(varargin)
% ���ÿɱ��������

% �����������
[x,t,sd,sd2,tol,MODE_COMPLEX,ndirs,display_sifting,sdt,sd2t,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask] = init(varargin{:});
% ����˵����
% x �ź�
% t ʱ��ʸ��
% sd ����
% sd2 ����2
% tol ����ֵ
% MODE_COMPLEX �Ƿ����ź�
% ndirs �������
% display_sifting �Ƿ���ʾ��������
% sdt ��������չΪ���źų���һ����ʸ��
% sd2t ������2��չΪ���źų���һ����ʸ��
% r ����x
% imf ���ʹ��mask�źţ���ʱIMF�Ѿ��õ���
% k ��¼�Ѿ���ȡ��IMF����
% nbit ��¼��ȡÿһ��IMFʱ�����Ĵ���
% NbIt ��¼�������ܴ���
% MAXITERATIONS ��ȡÿ��IMFʱ���õ�����������
% FIXE ����ָ�������ĵ���
% FIXE_H ����ָ�������ĵ������ұ��� |#zeros-#extrema|<=1 ��ֹͣ����
% MAXMODES ��ȡ�����IMF����
% INTERP ��ֵ����
% mask mask�ź�

% ���Ҫ����ʾ�������̣��� fig_h ���浱ǰͼ�δ��ھ��
if display_sifting
  fig_h = figure;
end

% ��ѭ�� : ����Ҫ�����3����ֵ�㣬�������mask�źţ���������ѭ��
while ~stop_EMD(r,MODE_COMPLEX,ndirs) && (k < MAXMODES+1 || MAXMODES == 0) && ~any(mask)

  % ��ǰģʽ
  m = r;

  % ǰһ�ε�����ģʽ
  mp = m;

  % �����ֵ��ֹͣ����
  if FIXE % ����趨�˵�������
    [stop_sift,moyenne] = stop_sifting_fixe(t,m,INTERP,MODE_COMPLEX,ndirs);
  elseif FIXE_H % ����趨�˵����������ұ���ֹͣ����|#zeros-#extrema|<=1
    stop_count = 0;
    [stop_sift,moyenne] = stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H,MODE_COMPLEX,ndirs);
  else % ����Ĭ��ֹͣ����
    [stop_sift,moyenne] = stop_sifting(m,t,sd,sd2,tol,INTERP,MODE_COMPLEX,ndirs);
  end

  % ��ǰģʽ���ȹ�С���������ȾͿ���������ټ�ֵ��ĳ���
  if (max(abs(m))) < (1e-10)*(max(abs(x)))	% IMF�����ֵС���ź����ֵ��1e-10
    if ~stop_sift % ���ɸ����û��ֹͣ
      warning('emd:warning','forced stop of EMD : too small amplitude')
    else
      disp('forced stop of EMD : too small amplitude')
    end
    break
  end


  % ɸѭ��
  while ~stop_sift && nbitMAXITERATIONS/5 && mod(nbit,floor(MAXITERATIONS/10))==0 && ~FIXE && nbit > 100)
      disp(['mode ',int2str(k),', iteration ',int2str(nbit)])
      if exist('s','var')
        disp(['stop parameter mean value : ',num2str(s)])
      end
      [im,iM] = extr(m);
      disp([int2str(sum(m(im) > 0)),' minima > 0; ',int2str(sum(m(iM) < 0)),' maxima < 0.'])
    end

    % ɸ����
    m = m - moyenne;

    % �����ֵ��ֹͣ����
    if FIXE
      [stop_sift,moyenne] = stop_sifting_fixe(t,m,INTERP,MODE_COMPLEX,ndirs);
    elseif FIXE_H
      [stop_sift,moyenne,stop_count] = stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H,MODE_COMPLEX,ndirs);
    else
      [stop_sift,moyenne,s] = stop_sifting(m,t,sd,sd2,tol,INTERP,MODE_COMPLEX,ndirs);
    end

    % ��ʾ����
    if display_sifting && ~MODE_COMPLEX
      NBSYM = 2;
      [indmin,indmax] = extr(mp);
      [tmin,tmax,mmin,mmax] = boundary_conditions(indmin,indmax,t,mp,mp,NBSYM);
      envminp = interp1(tmin,mmin,t,INTERP);
      envmaxp = interp1(tmax,mmax,t,INTERP);
      envmoyp = (envminp+envmaxp)/2;
      if FIXE || FIXE_H
        display_emd_fixe(t,m,mp,r,envminp,envmaxp,envmoyp,nbit,k,display_sifting)
      else
        sxp = 2*(abs(envmoyp))./(abs(envmaxp-envminp));
        sp = mean(sxp);
        display_emd(t,m,mp,r,envminp,envmaxp,envmoyp,s,sp,sxp,sdt,sd2t,nbit,k,display_sifting,stop_sift)
      end
    end

    mp = m;
    nbit = nbit+1;	% ���ֵ�������
    NbIt = NbIt+1;	% �����������

    if (nbit==(MAXITERATIONS-1) && ~FIXE && nbit > 100)
      if exist('s','var')
        warning('emd:warning',['forced stop of sifting : too many iterations... mode ',int2str(k),'. stop parameter mean value : ',num2str(s)])
      else
        warning('emd:warning',['forced stop of sifting : too many iterations... mode ',int2str(k),'.'])
      end
    end

  end % ɸѭ��
  
  imf(k,:) = m;
  if display_sifting
    disp(['mode ',int2str(k),' stored'])
  end
  nbits(k) = nbit;	% ��¼ÿ��IMF�ĵ�������
  k = k+1;		% IMF����


  r = r - m;		% ��ԭ�ź��м�ȥ��ȡ��IMF
  nbit = 0;		% ���ֵ���������0


end % ��ѭ��

% ��������ź�
if any(r) && ~any(mask)
  imf(k,:) = r;
end

% ��������ָ��
ort = io(x,imf);

% �ر�ͼ��
if display_sifting
  close
end

end

%---------------------------------------------------------------------------------------------------
% �����Ƿ�����㹻�ļ�ֵ�㣨3�������зֽ⣬��ֵ�����С��3���򷵻�1����������ֹͣ����
function stop = stop_EMD(r,MODE_COMPLEX,ndirs)
if MODE_COMPLEX  % ���ź����
  for k = 1:ndirs
    phi = (k-1)*pi/ndirs;
    [indmin,indmax] = extr(real(exp(i*phi)*r));
    ner(k) = length(indmin) + length(indmax);
  end
  stop = any(ner < 3);
else % ʵ�ź����
  [indmin,indmax] = extr(r);
  ner = length(indmin) + length(indmax);
  stop = ner < 3;
end
end

%---------------------------------------------------------------------------------------------------
% ���������ֵ��ģʽ���ȹ���ֵ�����ذ����ֵ
function [envmoy,nem,nzm,amp] = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs)
NBSYM = 2;	% �߽����ص���
if MODE_COMPLEX		% ���ź����
  switch MODE_COMPLEX
    case 1
      for k = 1:ndirs
        phi = (k-1)*pi/ndirs;
        y = real(exp(-i*phi)*m);
        [indmin,indmax,indzer] = extr(y);
        nem(k) = length(indmin)+length(indmax);
        nzm(k) = length(indzer);
        [tmin,tmax,zmin,zmax] = boundary_conditions(indmin,indmax,t,y,m,NBSYM);
        envmin(k,:) = interp1(tmin,zmin,t,INTERP);
        envmax(k,:) = interp1(tmax,zmax,t,INTERP);
      end
      envmoy = mean((envmin+envmax)/2,1);
      if nargout > 3
        amp = mean(abs(envmax-envmin),1)/2;
      end
    case 2
      for k = 1:ndirs
        phi = (k-1)*pi/ndirs;
        y = real(exp(-i*phi)*m);
        [indmin,indmax,indzer] = extr(y);
        nem(k) = length(indmin)+length(indmax);
        nzm(k) = length(indzer);
        [tmin,tmax,zmin,zmax] = boundary_conditions(indmin,indmax,t,y,y,NBSYM);
        envmin(k,:) = exp(i*phi)*interp1(tmin,zmin,t,INTERP);
        envmax(k,:) = exp(i*phi)*interp1(tmax,zmax,t,INTERP);
      end
      envmoy = mean((envmin+envmax),1);
      if nargout > 3
        amp = mean(abs(envmax-envmin),1)/2;
      end
  end
else	% ʵ�ź����
  [indmin,indmax,indzer] = extr(m);	% ������Сֵ�����ֵ�͹����λ��
  nem = length(indmin)+length(indmax);
  nzm = length(indzer);
  [tmin,tmax,mmin,mmax] = boundary_conditions(indmin,indmax,t,m,m,NBSYM);	% �߽�����
  envmin = interp1(tmin,mmin,t,INTERP);
  envmax = interp1(tmax,mmax,t,INTERP);
  envmoy = (envmin+envmax)/2;
  if nargout > 3
    amp = mean(abs(envmax-envmin),1)/2;  	% ����������
  end
end
end

%-------------------------------------------------------------------------------
% Ĭ��ֹͣ���������ǵ��ֵ���ֹͣ����
function [stop,envmoy,s] = stop_sifting(m,t,sd,sd2,tol,INTERP,MODE_COMPLEX,ndirs)
try
  [envmoy,nem,nzm,amp] = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs);
  sx = abs(envmoy)./amp;
  s = mean(sx);
  stop = ~((mean(sx > sd) > tol | any(sx > sd2)) & (all(nem > 2)));  % ֹͣ׼�������˼�ֵ���������2��
  if ~MODE_COMPLEX
    stop = stop && ~(abs(nzm-nem)>1);	% ����ʵ�źţ�Ҫ��ֵ��͹����ĸ������1
  end
catch
  stop = 1;
  envmoy = zeros(1,length(m));
  s = NaN;
end
end

%-------------------------------------------------------------------------------
% ���FIXѡ���ֹͣ����
function [stop,moyenne]= stop_sifting_fixe(t,m,INTERP,MODE_COMPLEX,ndirs)
try
  moyenne = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs);	% ��������²��ᵼ��ֹͣ
  stop = 0;
catch
  moyenne = zeros(1,length(m));
  stop = 1;
end
end

%-------------------------------------------------------------------------------
% ���FIX_Hѡ���ֹͣ����
function [stop,moyenne,stop_count]= stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H,MODE_COMPLEX,ndirs)
try
  [moyenne,nem,nzm] = mean_and_amplitude(m,t,INTERP,MODE_COMPLEX,ndirs);
  if (all(abs(nzm-nem)>1))
    stop = 0;
    stop_count = 0;
  else	% ��ֵ��������������1�󣬻�Ҫ�ﵽָ��������ֹͣ
    stop_count = stop_count+1;
    stop = (stop_count == FIXE_H);
  end
catch
  moyenne = zeros(1,length(m));
  stop = 1;
end
end

%-------------------------------------------------------------------------------
% ��ʾ�ֽ���̣�Ĭ��׼��
function display_emd(t,m,mp,r,envmin,envmax,envmoy,s,sb,sx,sdt,sd2t,nbit,k,display_sifting,stop_sift)

subplot(4,1,1)
plot(t,mp);hold on;
plot(t,envmax,'--k');plot(t,envmin,'--k');plot(t,envmoy,'r');
title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' before sifting']);
set(gca,'XTick',[])
hold  off
subplot(4,1,2)
plot(t,sx)
hold on
plot(t,sdt,'--r')
plot(t,sd2t,':k')
title('stop parameter')
set(gca,'XTick',[])
hold off
subplot(4,1,3)
plot(t,m)
title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' after sifting']);
set(gca,'XTick',[])
subplot(4,1,4);
plot(t,r-m)
title('residue');
disp(['stop parameter mean value : ',num2str(sb),' before sifting and ',num2str(s),' after'])
if stop_sift
  disp('last iteration for this mode')
end
if display_sifting == 2
  pause(0.01)
else
  pause
end
end

%---------------------------------------------------------------------------------------------------
% ��ʾ�ֽ���̣�FIX��FIX_Hֹͣ׼��
function display_emd_fixe(t,m,mp,r,envmin,envmax,envmoy,nbit,k,display_sifting)
subplot(3,1,1)
plot(t,mp);hold on;
plot(t,envmax,'--k');plot(t,envmin,'--k');plot(t,envmoy,'r');
title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' before sifting']);
set(gca,'XTick',[])
hold  off
subplot(3,1,2)
plot(t,m)
title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' after sifting']);
set(gca,'XTick',[])
subplot(3,1,3);
plot(t,r-m)
title('residue');
if display_sifting == 2
  pause(0.01)
else
  pause
end
end

%---------------------------------------------------------------------------------------
% ����߽����������񷨣�
function [tmin,tmax,zmin,zmax] = boundary_conditions(indmin,indmax,t,x,z,nbsym)
% ʵ������£�x = z

lx = length(x);

% �жϼ�ֵ�����
if (length(indmin) + length(indmax) < 3)
  error('not enough extrema')
end

% ��ֵ�ı߽�����
if indmax(1) < indmin(1)	% ��һ����ֵ���Ǽ���ֵ
  if x(1) > x(indmin(1))	% �Ե�һ������ֵΪ�Գ�����
    lmax = fliplr(indmax(2:min(end,nbsym+1)));
    lmin = fliplr(indmin(1:min(end,nbsym)));
    lsym = indmax(1);
  else	% �����һ������ֵС�ڵ�һ����Сֵ������Ϊ��ֵ��һ����Сֵ���Ըõ�Ϊ�Գ�����
    lmax = fliplr(indmax(1:min(end,nbsym)));
    lmin = [fliplr(indmin(1:min(end,nbsym-1))),1];
    lsym = 1;
  end
else
  if x(1) < x(indmax(1))	% �Ե�һ����СֵΪ�Գ�����
    lmax = fliplr(indmax(1:min(end,nbsym)));
    lmin = fliplr(indmin(2:min(end,nbsym+1)));
    lsym = indmin(1);
  else  % �����һ������ֵ���ڵ�һ������ֵ������Ϊ��ֵ��һ������ֵ���Ըõ�Ϊ�Գ�����
    lmax = [fliplr(indmax(1:min(end,nbsym-1))),1];
    lmin = fliplr(indmin(1:min(end,nbsym)));
    lsym = 1;
  end
end

% ����ĩβ��������п�ͷ����
if indmax(end) < indmin(end)
  if x(end) < x(indmax(end))
    rmax = fliplr(indmax(max(end-nbsym+1,1):end));
    rmin = fliplr(indmin(max(end-nbsym,1):end-1));
    rsym = indmin(end);
  else
    rmax = [lx,fliplr(indmax(max(end-nbsym+2,1):end))];
    rmin = fliplr(indmin(max(end-nbsym+1,1):end));
    rsym = lx;
  end
else
  if x(end) > x(indmin(end))
    rmax = fliplr(indmax(max(end-nbsym,1):end-1));
    rmin = fliplr(indmin(max(end-nbsym+1,1):end));
    rsym = indmax(end);
  else
    rmax = fliplr(indmax(max(end-nbsym+1,1):end));
    rmin = [lx,fliplr(indmin(max(end-nbsym+2,1):end))];
    rsym = lx;
  end
end
    
% �����и��ݶԳ����ģ���������
tlmin = 2*t(lsym)-t(lmin);
tlmax = 2*t(lsym)-t(lmax);
trmin = 2*t(rsym)-t(rmin);
trmax = 2*t(rsym)-t(rmax);
    
% ����ԳƵĲ���û���㹻�ļ�ֵ��
if tlmin(1) > t(1) || tlmax(1) > t(1)	% ���ۺ������û�г���ԭ���еķ�Χ
  if lsym == indmax(1)
    lmax = fliplr(indmax(1:min(end,nbsym)));
  else
    lmin = fliplr(indmin(1:min(end,nbsym)));
  end
  if lsym == 1	% ���������Ӧ�ó��֣�����ֱ����ֹ
    error('bug')
  end
  lsym = 1;	% ֱ�ӹ��ڵ�һ������ȡ����
  tlmin = 2*t(lsym)-t(lmin);
  tlmax = 2*t(lsym)-t(lmax);
end   
    
% ����ĩβ��������п�ͷ����
if trmin(end) < t(lx) || trmax(end) < t(lx)
  if rsym == indmax(end)
    rmax = fliplr(indmax(max(end-nbsym+1,1):end));
  else
    rmin = fliplr(indmin(max(end-nbsym+1,1):end));
  end
  if rsym == lx
    error('bug')
  end
  rsym = lx;
  trmin = 2*t(rsym)-t(rmin);
  trmax = 2*t(rsym)-t(rmax);
end 

% ���ص��ϵ�ȡֵ       
zlmax = z(lmax); 
zlmin = z(lmin);
zrmax = z(rmax); 
zrmin = z(rmin);
     
% �������
tmin = [tlmin t(indmin) trmin];
tmax = [tlmax t(indmax) trmax];
zmin = [zlmin z(indmin) zrmin];
zmax = [zlmax z(indmax) zrmax];

end
    
%---------------------------------------------------------------------------------------------------
% ��ֵ��͹����λ����ȡ
function [indmin, indmax, indzer] = extr(x,t)

if(nargin==1)
  t = 1:length(x);
end

m = length(x);

if nargout > 2
  x1 = x(1:m-1);
  x2 = x(2:m);
  indzer = find(x1.*x2<0);	% Ѱ���źŷ��ŷ����仯��λ��

  if any(x == 0)	% �����źŲ�����ǡ��Ϊ0��λ��
    iz = find( x==0 );  % �źŲ�����ǡ��Ϊ0��λ��
    indz = [];
    if any(diff(iz)==1) % ������0�����
      zer = x == 0;	% x=0��Ϊ1�������ط�Ϊ0
      dz = diff([0 zer 0]);	% Ѱ��0���0�Ĺ��ɵ�
      debz = find(dz == 1);	% 0ֵ���
      finz = find(dz == -1)-1;  % 0ֵ�յ�
      indz = round((debz+finz)/2);	% ѡ���м����Ϊ�����
    else
      indz = iz;	% ��û����0��������õ㱾����ǹ����
    end
    indzer = sort([indzer indz]);	% ȫ����������
  end
end

% ��ȡ��ֵ��
d = diff(x);
n = length(d);
d1 = d(1:n-1);
d2 = d(2:n);
indmin = find(d1.*d2<0 & d1<0)+1;	% ��Сֵ
indmax = find(d1.*d2<0 & d1>0)+1;	% ���ֵ


% �������������ֵ��ͬʱ�������м��һ��ֵ��Ϊ��ֵ�㣬����ʽ����0����
if any(d==0)

  imax = [];
  imin = [];

  bad = (d==0);
  dd = diff([0 bad 0]);
  debs = find(dd == 1);
  fins = find(dd == -1);
  if debs(1) == 1	% ����ֵ���������п�ͷ
    if length(debs) > 1
      debs = debs(2:end);
      fins = fins(2:end);
    else
      debs = [];
      fins = [];
    end
  end
  if length(debs) > 0
    if fins(end) == m	% ����ֵ����������ĩβ
      if length(debs) > 1
        debs = debs(1:(end-1));
        fins = fins(1:(end-1));

      else
        debs = [];
        fins = [];
      end
    end
  end
  lc = length(debs);
  if lc > 0
    for k = 1:lc
      if d(debs(k)-1) > 0	% ȡ�м�ֵ
        if d(fins(k)) < 0
          imax = [imax round((fins(k)+debs(k))/2)];
        end
      else
        if d(fins(k)) > 0
          imin = [imin round((fins(k)+debs(k))/2)];
        end
      end
    end
  end

  if length(imax) > 0
    indmax = sort([indmax imax]);
  end

  if length(imin) > 0
    indmin = sort([indmin imin]);
  end

end
end

%---------------------------------------------------------------------------------------------------

function ort = io(x,imf)
% ort = IO(x,imf) ��������ָ��
%
% ���� : - x    : �����ź�
%        - imf  : IMF�ź�

n = size(imf,1);

s = 0;
% ���ݹ�ʽ����
for i = 1:n
  for j = 1:n
    if i ~= j
      s = s + abs(sum(imf(i,:).*conj(imf(j,:)))/sum(x.^2));
    end
  end
end

ort = 0.5*s;
end

%---------------------------------------------------------------------------------------------------
% ������������
function [x,t,sd,sd2,tol,MODE_COMPLEX,ndirs,display_sifting,sdt,sd2t,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask] = init(varargin)

x = varargin{1};
if nargin == 2
  if isstruct(varargin{2})
    inopts = varargin{2};
  else
    error('when using 2 arguments the first one is the analyzed signal X and the second one is a struct object describing the options')
  end
elseif nargin > 2
  try
    inopts = struct(varargin{2:end});
  catch
    error('bad argument syntax')
  end
end

% Ĭ��ֹͣ����
defstop = [0.05,0.5,0.05];

opt_fields = {'t','stop','display','maxiterations','fix','maxmodes','interp','fix_h','mask','ndirs','complex_version'};
% ʱ�����У�ֹͣ�������Ƿ���ʾ��������������ÿһ�ֵ���������IMF��������ֵ������ÿһ�ֵ�������������������mask�źţ����������Ƿ���ø���ģʽ

defopts.stop = defstop;
defopts.display = 0;
defopts.t = 1:max(size(x));
defopts.maxiterations = 2000;
defopts.fix = 0;
defopts.maxmodes = 0;
defopts.interp = 'spline';
defopts.fix_h = 0;
defopts.mask = 0;
defopts.ndirs = 4;
defopts.complex_version = 2;

opts = defopts;

if(nargin==1)
  inopts = defopts;
elseif nargin == 0
  error('not enough arguments')
end

names = fieldnames(inopts);
for nom = names'
  if ~any(strcmpi(char(nom), opt_fields))
    error(['bad option field name: ',char(nom)])
  end
  if ~isempty(eval(['inopts.',char(nom)])) 
    eval(['opts.',lower(char(nom)),' = inopts.',char(nom),';'])
  end
end

t = opts.t;
stop = opts.stop;
display_sifting = opts.display;
MAXITERATIONS = opts.maxiterations;
FIXE = opts.fix;
MAXMODES = opts.maxmodes;
INTERP = opts.interp;
FIXE_H = opts.fix_h;
mask = opts.mask;
ndirs = opts.ndirs;
complex_version = opts.complex_version;

if ~isvector(x)
  error('X must have only one row or one column')
end

if size(x,1) > 1
  x = x.';
end

if ~isvector(t)
  error('option field T must have only one row or one column')
end

if ~isreal(t)
  error('time instants T must be a real vector')
end

if size(t,1) > 1
  t = t';
end

if (length(t)~=length(x))
  error('X and option field T must have the same length')
end

if ~isvector(stop) || length(stop) > 3
  error('option field STOP must have only one row or one column of max three elements')
end

if ~all(isfinite(x))
  error('data elements must be finite')
end

if size(stop,1) > 1
  stop = stop';
end

L = length(stop);
if L < 3
  stop(3) = defstop(3);
end

if L < 2
  stop(2) = defstop(2);
end

if ~ischar(INTERP) || ~any(strcmpi(INTERP,{'linear','cubic','spline'}))
  error('INTERP field must be ''linear'', ''cubic'', ''pchip'' or ''spline''')
end

% ʹ��mask�ź�ʱ�����⴦��
if any(mask)
  if ~isvector(mask) || length(mask) ~= length(x)
    error('masking signal must have the same dimension as the analyzed signal X')
  end

  if size(mask,1) > 1
    mask = mask.';
  end
  opts.mask = 0;
  imf1 = emd(x+mask, opts);
  imf2 = emd(x-mask, opts);
  if size(imf1,1) ~= size(imf2,1)
    warning('emd:warning',['the two sets of IMFs have different sizes: ',int2str(size(imf1,1)),' and ',int2str(size(imf2,1)),' IMFs.'])
  end
  S1 = size(imf1,1);
  S2 = size(imf2,1);
  if S1 ~= S2	% ��������źŷֽ�õ���IMF������һ�£�����˳��
    if S1 < S2
      tmp = imf1;
      imf1 = imf2;
      imf2 = tmp;
    end
    imf2(max(S1,S2),1) = 0;	% ���̵��Ǹ����㣬�ﵽ����һ��
  end
  imf = (imf1+imf2)/2;

end


sd = stop(1);
sd2 = stop(2);
tol = stop(3);

lx = length(x);

sdt = sd*ones(1,lx);
sd2t = sd2*ones(1,lx);

if FIXE
  MAXITERATIONS = FIXE;
  if FIXE_H
    error('cannot use both ''FIX'' and ''FIX_H'' modes')
  end
end

MODE_COMPLEX = ~isreal(x)*complex_version;
if MODE_COMPLEX && complex_version ~= 1 && complex_version ~= 2
  error('COMPLEX_VERSION parameter must equal 1 or 2')
end


% ��ֵ��͹����ĸ���
ner = lx;
nzr = lx;

r = x;

if ~any(mask) % ���ʹ����mask�źţ���ʱimf���Ѿ�����õ���
  imf = [];
end
k = 1;

% ��ȡÿ��ģʽʱ�����Ĵ���
nbit = 0;

% �����������
NbIt = 0;
end
%---------------------------------------------------------------------------------------------------