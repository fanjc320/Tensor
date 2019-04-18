import numpy as np, wave, math
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.pyplot as plt
from Common.common import logFile
# from skimage import data, filters
# import test_img_sharp
# import test_img_sharp1

def mapTo(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


# def mapTo_Arr(arr,stop1,stop2):
#     minv = np.min(arr)
#     maxv = np.max(arr)
#     for item in arr:
#         res = mapTo(item,minv,maxv,stop1,stop2)

# 归一化
def maxminnorm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t


def stride_windows(x, n, noverlap=None, axis=0):
    '''
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.

    .. warning::

        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory,
        so modifying one value may change others.

    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.

    n : integer
        The number of data points in each window.

    noverlap : integer
        The overlap between adjacent windows.
        Default is 0 (no overlap)

    axis : integer
        The axis along which the windows will run.

    References
    ----------
    `stackoverflow: Rolling window for 1D arrays in Numpy?
    <http://stackoverflow.com/a/6811241>`_
    `stackoverflow: Using strides for an efficient moving average filter
    <http://stackoverflow.com/a/4947453>`_
    '''
    if noverlap is None:
        noverlap = 0

    if noverlap >= n:
        raise ValueError('noverlap must be less than n')
    if n < 1:
        raise ValueError('n cannot be less than 1')

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')
    if n == 1 and noverlap == 0:
        if axis == 0:
            print("stride_windows np.newaxis:",np.newaxis)
            return x[np.newaxis]
        else:
            return x[np.newaxis].transpose()
    if n > x.size:
        raise ValueError('n cannot be greater than the length of x')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. noverlap or n. See #3845.
    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    np.set_printoptions(threshold=np.nan)
    if axis == 0:
        shape = (n, (x.shape[-1] - noverlap) // step)
        strides = (x.strides[0], step * x.strides[0])
        print("stride_windows ddd:",n,noverlap,step,x.shape,
              "\nstrides:",x.strides,x.strides[0],
              "\ntype:",type(x[0]),x.dtype)
        res = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        print("stride_windows shape && strides!!:",shape,strides,
              "\nshape:",x.shape,res.shape,
              "\ndtype:",x.dtype,res.dtype)
    else:
        shape = ((x.shape[-1] - noverlap) // step, n)
        strides = (step * x.strides[0], x.strides[0])
    # logFile("./x.txt",x[:1030])
    # logFile("./res.txt",res.transpose().flatten()[:1030])
    # print("x==res",x==res.transpose().flatten())
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)



def stride_repeat(x, n, axis=0):
    '''
    Repeat the values in an array in a memory-efficient manner.  Array x is
    stacked vertically n times.

    .. warning::

        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory, so
        modifying one value may change others.

    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.

    n : integer
        The number of time to repeat the array.

    axis : integer
        The axis along which the data will run.

    References
    ----------
    `stackoverflow: Repeat NumPy array without replicating data?
    <http://stackoverflow.com/a/5568169>`_
    '''
    if axis not in [0, 1]:
        raise ValueError('axis must be 0 or 1')
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')

    if n == 1:
        if axis == 0:
            return np.atleast_2d(x)
        else:
            return np.atleast_2d(x).T
    if n < 1:
        raise ValueError('n cannot be less than 1')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. n. See #3845.
    n = int(n)

    if axis == 0:
        shape = (n, x.size)
        strides = (0, x.strides[0])
    else:
        shape = (x.size, n)
        strides = (x.strides[0], 0)

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def apply_window(x, window, axis=0, return_window=None):
    '''
    Apply the given window to the given 1D or 2D array along the given axis.

    Parameters
    ----------
    x : 1D or 2D array or sequence
        Array or sequence containing the data.

    window : function or array.
        Either a function to generate a window or an array with length
        *x*.shape[*axis*]

    axis : integer
        The axis over which to do the repetition.
        Must be 0 or 1.  The default is 0

    return_window : bool
        If true, also return the 1D values of the window that was applied
    '''
    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError('only 1D or 2D arrays can be used')
    if axis + 1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    xshape = list(x.shape)
    xshapetarg = xshape.pop(axis)

    if cbook.iterable(window):
        if len(window) != xshapetarg:
            raise ValueError('The len(window) must be the same as the shape '
                             'of x for the chosen axis')
        windowVals = window
    else:
        windowVals = window(np.ones(xshapetarg, dtype=x.dtype))

    if x.ndim == 1:
        if return_window:
            return windowVals * x, windowVals
        else:
            return windowVals * x

    xshapeother = xshape.pop()

    otheraxis = (axis + 1) % 2

    windowValsRep = stride_repeat(windowVals, xshapeother, axis=otheraxis)

    if return_window:
        return windowValsRep * x, windowVals
    else:
        return windowValsRep * x


# pyplot.py
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
# @_autogen_docstring(Axes.specgram)
def specgram(x, NFFT=None, Fs=None, Fc=None, detrend=None, window=None,
             noverlap=None, cmap=None, xextent=None, pad_to=None, sides=None,
             scale_by_freq=None, mode=None, scale=None, vmin=None, vmax=None,
             hold=None, data=None, **kwargs):
    # Deprecated: allow callers to override the hold state
    # by passing hold=True|False
    print("======specgram begin======")
    try:
        ret = specgram_ax(x, NFFT=NFFT, Fs=Fs, Fc=Fc, detrend=detrend,
                          window=window, noverlap=noverlap, cmap=cmap,
                          xextent=xextent, pad_to=pad_to, sides=sides,
                          scale_by_freq=scale_by_freq, mode=mode, scale=scale,
                          vmin=vmin, vmax=vmax, data=data, **kwargs)
    finally:
        print("specgram finish fjc")
    # sci(ret[-1])
    print("======specgram======")
    return ret


# _axes.py
# @_preprocess_data(replace_names=["x"], label_namer=None)
# @docstring.dedent_interpd
def specgram_ax(x, NFFT=None, Fs=None, Fc=None, detrend=None,
                window=None, noverlap=None,
                cmap=None, xextent=None, pad_to=None, sides=None,
                scale_by_freq=None, mode=None, scale=None,
                vmin=None, vmax=None, **kwargs):
    """
    Plot a spectrogram.

    Call signature::

      specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=128,
               cmap=None, xextent=None, pad_to=None, sides='default',
               scale_by_freq=None, mode='default', scale='default',
               **kwargs)

    Compute and plot a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the spectrum of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*. The spectrogram is plotted as a colormap
    (using imshow).

    Parameters
    ----------
    x : 1-D array or sequence
        Array or sequence containing the data.

    %(Spectral)s

    %(PSD)s

    mode : [ 'default' | 'psd' | 'magnitude' | 'angle' | 'phase' ]
        What sort of spectrum to use.  Default is 'psd', which takes
        the power spectral density.  'complex' returns the complex-valued
        frequency spectrum.  'magnitude' returns the magnitude spectrum.
        'angle' returns the phase spectrum without unwrapping.  'phase'
        returns the phase spectrum with unwrapping.

    noverlap : integer
        The number of points of overlap between blocks.  The
        default value is 128.

    scale : [ 'default' | 'linear' | 'dB' ]
        The scaling of the values in the *spec*.  'linear' is no scaling.
        'dB' returns the values in dB scale.  When *mode* is 'psd',
        this is dB power (10 * log10).  Otherwise this is dB amplitude
        (20 * log10). 'default' is 'dB' if *mode* is 'psd' or
        'magnitude' and 'linear' otherwise.  This must be 'linear'
        if *mode* is 'angle' or 'phase'.

    Fc : integer
        The center frequency of *x* (defaults to 0), which offsets
        the x extents of the plot to reflect the frequency range used
        when a signal is acquired and then filtered and downsampled to
        baseband.

    cmap :
        A :class:`matplotlib.colors.Colormap` instance; if *None*, use
        default determined by rc

    xextent : [None | (xmin, xmax)]
        The image extent along the x-axis. The default sets *xmin* to the
        left border of the first bin (*spectrum* column) and *xmax* to the
        right border of the last bin. Note that for *noverlap>0* the width
        of the bins is smaller than those of the segments.

    **kwargs :
        Additional kwargs are passed on to imshow which makes the
        specgram image

    Returns
    -------
    spectrum : 2-D array
        Columns are the periodograms of successive segments.

    freqs : 1-D array
        The frequencies corresponding to the rows in *spectrum*.

    t : 1-D array
        The times corresponding to midpoints of segments (i.e., the columns
        in *spectrum*).

    im : instance of class :class:`~matplotlib.image.AxesImage`
        The image created by imshow containing the spectrogram

    See Also
    --------
    :func:`psd`
        :func:`psd` differs in the default overlap; in returning the mean
        of the segment periodograms; in not returning times; and in
        generating a line plot instead of colormap.

    :func:`magnitude_spectrum`
        A single spectrum, similar to having a single segment when *mode*
        is 'magnitude'. Plots a line instead of a colormap.

    :func:`angle_spectrum`
        A single spectrum, similar to having a single segment when *mode*
        is 'angle'. Plots a line instead of a colormap.

    :func:`phase_spectrum`
        A single spectrum, similar to having a single segment when *mode*
        is 'phase'. Plots a line instead of a colormap.

    Notes
    -----
    The parameters *detrend* and *scale_by_freq* do only apply when *mode*
    is set to 'psd'.
    """

    if NFFT is None:
        NFFT = 256  # same default as in mlab.specgram()
    if Fc is None:
        Fc = 0  # same default as in mlab._spectral_helper()
    if noverlap is None:
        noverlap = 128  # same default as in mlab.specgram()

    if mode == 'complex':
        raise ValueError('Cannot plot a complex specgram')

    if scale is None or scale == 'default':
        if mode in ['angle', 'phase']:
            scale = 'linear'
        else:
            scale = 'dB'
    elif mode in ['angle', 'phase'] and scale == 'dB':
        raise ValueError('Cannot use dB scale with angle or phase mode')

    spec, freqs, t = specgram_mlab(x=x, NFFT=NFFT, Fs=Fs,
                                   detrend=detrend, window=window,
                                   noverlap=noverlap, pad_to=pad_to,
                                   sides=sides,
                                   scale_by_freq=scale_by_freq,
                                   mode=mode)

    if scale == 'linear':
        Z = spec
    elif scale == 'dB':
        if mode is None or mode == 'default' or mode == 'psd':
            Z = 10. * np.log10(spec)
            print("==============")
        else:
            Z = 20. * np.log10(spec)
            print("--------------")
    else:
        raise ValueError('Unknown scale %s', scale)

    Z = np.flipud(Z)

    if xextent is None:
        # padding is needed for first and last segment:
        pad_xextent = (NFFT - noverlap) / Fs / 2
        xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
        print("pad_xextent:",pad_xextent,np.min(t),np.max(t))
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]

    print("===z.shape:", Z.shape, extent, min(freqs), max(freqs),cmap)  # freqs是等差数列，是一系列连续的频率数组

    # Z = Z[-200:-1,:]
    # im = plt.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax,
    #                 **kwargs)# vmin,和vmax仅限制了图像颜色深度范围,没有太大意义
    # im = plt.imshow(Z, extent=extent)

    plt.axis('auto')
    im=0
    return spec, freqs, t, im,Z,extent


# mlab.py
# @docstring.dedent_interpd
def specgram_mlab(x, NFFT=None, Fs=None, detrend=None, window=None,
                  noverlap=None, pad_to=None, sides=None, scale_by_freq=None,
                  mode=None):
    """
    Compute a spectrogram.

    Compute and plot a spectrogram of data in x.  Data are split into
    NFFT length segments and the spectrum of each section is
    computed.  The windowing function window is applied to each
    segment, and the amount of overlap of each segment is
    specified with noverlap.

    Parameters
    ----------
    x : array_like
        1-D array or sequence.

    %(Spectral)s

    %(PSD)s

    noverlap : int, optional
        The number of points of overlap between blocks.  The default
        value is 128.
    mode : str, optional
        What sort of spectrum to use, default is 'psd'.
            'psd'
                Returns the power spectral density.

            'complex'
                Returns the complex-valued frequency spectrum.

            'magnitude'
                Returns the magnitude spectrum.

            'angle'
                Returns the phase spectrum without unwrapping.

            'phase'
                Returns the phase spectrum with unwrapping.

    Returns
    -------
    spectrum : array_like
        2-D array, columns are the periodograms of successive segments.

    freqs : array_like
        1-D array, frequencies corresponding to the rows in *spectrum*.

    t : array_like
        1-D array, the times corresponding to midpoints of segments
        (i.e the columns in *spectrum*).

    See Also
    --------
    psd : differs in the overlap and in the return values.
    complex_spectrum : similar, but with complex valued frequencies.
    magnitude_spectrum : similar single segment when mode is 'magnitude'.
    angle_spectrum : similar to single segment when mode is 'angle'.
    phase_spectrum : similar to single segment when mode is 'phase'.

    Notes
    -----
    detrend and scale_by_freq only apply when *mode* is set to 'psd'.

    """
    if noverlap is None:
        noverlap = 128  # default in _spectral_helper() is noverlap = 0
    if NFFT is None:
        NFFT = 256  # same default as in _spectral_helper()
    if len(x) <= NFFT:
        warnings.warn("Only one segment is calculated since parameter NFFT " +
                      "(=%d) >= signal length (=%d)." % (NFFT, len(x)))

    # spec, freqs, t = _spectral_helper(x=x, y=None, NFFT=NFFT, Fs=Fs,
    #                                   detrend_func=detrend, window=window,
    #                                   noverlap=noverlap, pad_to=pad_to,
    #                                   sides=sides,
    #                                   scale_by_freq=scale_by_freq,
    #                                   mode=mode)

    spec, freqs, t = _Unspectral_helper_mag(x=x, y=None, NFFT=NFFT, Fs=Fs,
                                      detrend_func=detrend, window=window,
                                      noverlap=noverlap, pad_to=pad_to,
                                      sides=sides,
                                      scale_by_freq=scale_by_freq,
                                      mode=mode)

    if mode != 'complex':
        spec = spec.real  # Needed since helper implements generically

    return spec, freqs, t


_coh_error = """Coherence is calculated by averaging over *NFFT*
length segments.  Your signal is too short for your choice of *NFFT*.
"""


# mlab.py
def _spectral_helper(x, y=None, NFFT=None, Fs=None, detrend_func=None,
                     window=None, noverlap=None, pad_to=None,
                     sides=None, scale_by_freq=None, mode=None):
    '''
    This is a helper function that implements the commonality between the
    psd, csd, spectrogram and complex, magnitude, angle, and phase spectrums.
    It is *NOT* meant to be used outside of mlab and may change at any time.
    '''
    if y is None:
        # if y is None use x for y
        same_data = True
    else:
        # The checks for if y is x are so that we can use the same function to
        # implement the core of psd(), csd(), and spectrogram() without doing
        # extra calculations.  We return the unaveraged Pxy, freqs, and t.
        same_data = y is x

    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    # if detrend_func is None:
    # detrend_func = detrend_none
    if window is None:
        window = window_hanning

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256

    if mode is None or mode == 'default':
        mode = 'psd'
    elif mode not in ['psd', 'complex', 'magnitude', 'angle', 'phase']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "'default', 'psd', 'complex', "
                         "'magnitude', 'angle', 'phase'" % mode)

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    # Make sure we're dealing with a numpy array. If y and x were the same
    # object to start with, keep them that way
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    if sides is None or sides == 'default':
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
    elif sides not in ['onesided', 'twosided']:
        raise ValueError("Unknown value for sides %s, must be one of: "
                         "'default', 'onesided', or 'twosided'" % sides)

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, (NFFT,))
        x[n:] = 0

    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, (NFFT,))
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if sides == 'twosided':
        numFreqs = pad_to
        if pad_to % 2:
            freqcenter = (pad_to - 1) // 2 + 1
        else:
            freqcenter = pad_to // 2
        scaling_factor = 1.
    elif sides == 'onesided':#true
        if pad_to % 2:
            numFreqs = (pad_to + 1) // 2
        else:
            numFreqs = pad_to // 2 + 1
        scaling_factor = 2.

    result = stride_windows(x, NFFT, noverlap, axis=0)
    result = detrend(result, detrend_func, axis=0)
    result, windowVals = apply_window(result, window, axis=0,
                                      return_window=True)
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(pad_to, 1 / Fs)[:numFreqs]

    if not same_data:
        # if same_data is False, mode must be 'psd'
        resultY = stride_windows(y, NFFT, noverlap)
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = apply_window(resultY, window, axis=0)
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conj(result) * resultY
    elif mode == 'psd':
        result = np.conj(result) * result
        print("-------111-------")
    elif mode == 'magnitude':
        result = np.abs(result) / np.abs(windowVals).sum()
    elif mode == 'angle' or mode == 'phase':
        # we unwrap the phase later to handle the onesided vs. twosided case
        result = np.angle(result)
    elif mode == 'complex':
        result /= np.abs(windowVals).sum()

    print("mode====:",mode,sides)

    if mode == 'psd':

        # Also include scaling factors for one-sided densities and dividing by
        # the sampling frequency, if desired. Scale everything, except the DC
        # component and the NFFT/2 component:

        # if we have a even number of frequencies, don't scale NFFT/2
        if not NFFT % 2:
            slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
        else:
            slc = slice(1, None, None)

        result[slc] *= scaling_factor

        # MATLAB divides by the sampling frequency so that density function
        # has units of dB/Hz and can be integrated by the plotted frequency
        # values. Perform the same scaling here.
        if scale_by_freq: # true
            result /= Fs
            # Scale the spectrum by the norm of the window to compensate for
            # windowing loss; see Bendat & Piersol Sec 11.5.2.
            result /= (np.abs(windowVals) ** 2).sum()
            print("scale_by_freq 00:",scale_by_freq)
        else:
            # In this case, preserve power in the segment, not amplitude
            result /= np.abs(windowVals).sum() ** 2
            print("scale_by_freq 11:", scale_by_freq)

    t = np.arange(NFFT / 2, len(x) - NFFT / 2 + 1, NFFT - noverlap) / Fs

    if sides == 'twosided':
        # center the frequency range at zero
        freqs = np.concatenate((freqs[freqcenter:], freqs[:freqcenter]))
        result = np.concatenate((result[freqcenter:, :],
                                 result[:freqcenter, :]), 0)
        print("not pad_to % 2 00:")
    elif not pad_to % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1
        print("not pad_to % 2 11:")

    # we unwrap the phase here to handle the onesided vs. twosided case
    if mode == 'phase':
        result = np.unwrap(result, axis=0)
    # print("_spectral_helper freqs:",freqs) # 等差数列0-22050, 43 为差,共512个
    return result, freqs, t

# test
# def rolling_window(a, window):
# shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
##shape:(4,3)  a.shape:(6,)   a.shape[:-1]:()   a.shape[-1]:6
# print("shape",shape,a.shape,a.shape[:-1],a.shape[-1])
# strides = a.strides + (a.strides[-1],)
# print("strides:",a.strides,strides)
# return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# a = np.array([1, 12, 3, 4, 5, 6]);
# b = rolling_window(a, 3)
# print("rolling_window:",b)
# c = np.lib.stride_tricks.as_strided(a, shape=(4,3), strides=(4,4))
# print("strides:",c)

def _Unspectral_helper_mag(x, y=None, NFFT=None, Fs=None, detrend_func=None,
                     window=None, noverlap=None, pad_to=None,
                     sides=None, scale_by_freq=None, mode=None):
    '''
    This is a helper function that implements the commonality between the
    psd, csd, spectrogram and complex, magnitude, angle, and phase spectrums.
    It is *NOT* meant to be used outside of mlab and may change at any time.
    '''
    if y is None:
        # if y is None use x for y
        same_data = True
    else:
        # The checks for if y is x are so that we can use the same function to
        # implement the core of psd(), csd(), and spectrogram() without doing
        # extra calculations.  We return the unaveraged Pxy, freqs, and t.
        same_data = y is x

    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    # if detrend_func is None:
    # detrend_func = detrend_none
    if window is None:
        window = window_hanning

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256

    # Make sure we're dealing with a numpy array. If y and x were the same
    # object to start with, keep them that way
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    sides = 'onesided'

    # time = np.arange(0, nframes) * (1.0 / framerate)  # 计算时间
    # time = np.reshape(time, [nframes, 1]).T
    # plt.plot(x, c="b")
    # plt.xlabel("time")
    # plt.ylabel("amplitude-x")
    # plt.show()

    if pad_to is None:
        pad_to = NFFT

    # For real x, ignore the negative frequencies unless told otherwise
    if sides == 'onesided':#true
        if pad_to % 2:
            numFreqs = (pad_to + 1) // 2
        else:
            numFreqs = pad_to // 2 + 1
        scaling_factor = 2.

    result = stride_windows(x, NFFT, noverlap, axis=0)
    print("x shape:",x.shape,result.shape,np.fft.fft(result, n=pad_to, axis=0).shape)
    print("pad_to", pad_to, numFreqs, NFFT, noverlap)
    # result = detrend(result, detrend_func, axis=0)
    # result, windowVals = apply_window(result, window, axis=0,
    #                                   return_window=True)
    plt.plot(result)
    plt.xlabel("result before")
    plt.show()
    oldresult = result
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :] #np.fft.fft(result, n=pad_to, axis=0).shape(1024,215) 只取fft变换结果的一半 numFreqs=513
    temp =np.fft.fft(result, n=pad_to, axis=0)
    res_ifft = np.fft.ifft(temp,n=pad_to,axis=0)
    print("res_ifft==oldresult:",res_ifft==oldresult)
    logFile("./res_ifft.txt",res_ifft)
    logFile("./oldresult.txt",oldresult)
    # x_revert = Unstride_windows(res_ifft,NFFT,noverlap,axis=0)
    # x_revert = result.transpose().flatten() # 和stride_windows是完全可逆的过程
    # print("res_ifft",res_ifft)
    # print("x_revert==x:",x_revert==x)
    plt.plot(res_ifft)
    plt.xlabel("res_ifft")
    plt.show()
    freqs = np.fft.fftfreq(pad_to, 1 / Fs)[:numFreqs]

    if mode == 'magnitude':
        # print("windowssum:",np.abs(windowVals).sum())
        # result = np.abs(result) / np.abs(windowVals).sum()
        result = np.abs(result) / 512

    print("mode====:",mode,sides)

    t = np.arange(NFFT / 2, len(x) - NFFT / 2 + 1, NFFT - noverlap) / Fs

    if not pad_to % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1
        print("not pad_to % 2 11:")

    # print("_spectral_helper freqs:",freqs) # 等差数列0-22050, 43 为差,共512个
    return result, freqs, t


def _Unspectral_helper(x, y=None, NFFT=None, Fs=None, detrend_func=None,
                     window=None, noverlap=None, pad_to=None,
                     sides=None, scale_by_freq=None, mode=None):
    if y is None:
        # if y is None use x for y
        same_data = True
    else:
        same_data = y is x

    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    # if detrend_func is None:
    # detrend_func = detrend_none

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256

    if mode is None or mode == 'default':
        mode = 'psd'

    # Make sure we're dealing with a numpy array. If y and x were the same
    # object to start with, keep them that way
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    sides = 'onesided'

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, (NFFT,))
        x[n:] = 0

    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, (NFFT,))
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if sides == 'onesided':
        if pad_to % 2:
            numFreqs = (pad_to + 1) // 2
        else:
            numFreqs = pad_to // 2 + 1
        scaling_factor = 2.
    # if sides == 'twosided':
    #     numFreqs = pad_to
    #     if pad_to % 2:
    #         freqcenter = (pad_to - 1)//2 + 1
    #     else:
    #         freqcenter = pad_to//2
    #     scaling_factor = 1.

    print("x.shape:",x.shape,noverlap)
    result = stride_windows(x, NFFT, noverlap, axis=0)
    # result = detrend(result, detrend_func, axis=0)
    result, windowVals = apply_window(result, window, axis=0,
                                      return_window=True)
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(pad_to, 1 / Fs)[:numFreqs]

    if mode == 'psd':
        print("result:",result)
        print("conj_result:",np.conj(result))
        result = np.conj(result) * result
        # Also include scaling factors for one-sided densities and dividing by
        # the sampling frequency, if desired. Scale everything, except the DC
        # component and the NFFT/2 component:

        # if we have a even number of frequencies, don't scale NFFT/2
        if not NFFT % 2:
            slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
        else:
            slc = slice(1, None, None)

        result[slc] *= scaling_factor

        # MATLAB divides by the sampling frequency so that density function
        # has units of dB/Hz and can be integrated by the plotted frequency
        # values. Perform the same scaling here.
        if scale_by_freq:
            '''
            result /= Fs
            # Scale the spectrum by the norm of the window to compensate for
            # windowing loss; see Bendat & Piersol Sec 11.5.2.
            result /= (np.abs(windowVals) ** 2).sum()
            print("wwww:",type(windowVals),Fs)
            print("----",(np.asarray([1,2,3]) )**2)
            print("windowVals:",windowVals,np.abs(windowVals),"\n",np.abs(windowVals) ** 2,"\n",(np.abs(windowVals) ** 2).sum())
            '''
        else:
            # In this case, preserve power in the segment, not amplitude
            result /= np.abs(windowVals).sum() ** 2

    t = np.arange(NFFT / 2, len(x) - NFFT / 2 + 1, NFFT - noverlap) / Fs

    # if sides == 'twosided':
    #     # center the frequency range at zero
    #     freqs = np.concatenate((freqs[freqcenter:], freqs[:freqcenter]))
    #     result = np.concatenate((result[freqcenter:, :],
    #                              result[:freqcenter, :]), 0)

    if not pad_to % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1

    # print("_spectral_helper freqs:",freqs) # 等差数列0-22050, 43 为差,共512个
    return result, freqs, t

def TestStrideWindows():
    # x = [1, 2, 3, 4, 5, 6, 7, 8];
    # x = [1,2,3,4,5,6,7,8,9];
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9,10];
    x=np.asarray(x)
    print("x.shape:",x.shape)
    result = stride_windows(x, 3, 0, axis=0)
    print("result:", result);
    window=np.hanning(M=3)
    result, windowVals = apply_Unwindow(result, window, axis=0,
                                      return_window=True)
    print("result apply_window:",result);

def apply_Unwindow(x, window, axis=0, return_window=None):
    '''
    Apply the given window to the given 1D or 2D array along the given axis.

    Parameters
    ----------
    x : 1D or 2D array or sequence
        Array or sequence containing the data.

    window : function or array.
        Either a function to generate a window or an array with length
        *x*.shape[*axis*]

    axis : integer
        The axis over which to do the repetition.
        Must be 0 or 1.  The default is 0

    return_window : bool
        If true, also return the 1D values of the window that was applied
    '''

    result, windowVals = apply_window(x, window, axis=0,
                                      return_window=True)
    print("x",x,x.ndim)
    print("result old:",result)
    print("window:",windowVals)

    # print("x huanyuan:",result/windowVals) # 'NoneType' object is not iterable

    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError('only 1D or 2D arrays can be used')
    if axis + 1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    xshape = list(x.shape)
    xshapetarg = xshape.pop(axis)

    if cbook.iterable(window):
        if len(window) != xshapetarg:
            raise ValueError('The len(window) must be the same as the shape '
                             'of x for the chosen axis')
        windowVals = window
        print("unwindow-----00")
    else:
        windowVals = window(np.ones(xshapetarg, dtype=x.dtype))
        print("unwindow-----11")

    if x.ndim == 1:
        if return_window:
            print("unwindow-----22")
            return windowVals * x, windowVals
        else:
            print("unwindow-----33")
            return windowVals * x

    # xshapeother = xshape.pop()
    #
    # otheraxis = (axis + 1) % 2
    #
    # windowValsRep = stride_repeat(windowVals, xshapeother, axis=otheraxis)
    #
    # if return_window:
    #     print("unwindow ------ return ")
    #     return windowValsRep * x, windowVals
    # else:
    #     print("unwindow-----44")
    #     return windowValsRep * x

def Unstride_windows(strided_res, n, noverlap=None, axis=0):
    '''
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.

    .. warning::

        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory,
        so modifying one value may change others.

    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.

    n : integer
        The number of data points in each window.

    noverlap : integer
        The overlap between adjacent windows.
        Default is 0 (no overlap)

    axis : integer
        The axis along which the windows will run.

    References
    ----------
    `stackoverflow: Rolling window for 1D arrays in Numpy?
    <http://stackoverflow.com/a/6811241>`_
    `stackoverflow: Using strides for an efficient moving average filter
    <http://stackoverflow.com/a/4947453>`_
    '''
    if noverlap is None:
        noverlap = 0

    if noverlap >= n:
        raise ValueError('noverlap must be less than n')
    if n < 1:
        raise ValueError('n cannot be less than 1')



    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. noverlap or n. See #3845.
    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    # np.set_printoptions(threshold=np.nan)
    if axis == 0:
        x = strided_res.transpose().flatten()
    # logFile("./x.txt",x[:1030])
    # logFile("./res.txt",strided_res.transpose().flatten()[:1030])
    print("x==res",x==strided_res)
    return x

# TestStrideWindows()

def TestUnStride():

    sudoku = np.array(
        [2, 8, 7, 1, 6, 5, 9, 4, 3, 9, 5, 4, 7, 3, 2, 1, 6, 8]#,
        # 6, 1, 3, 8, 4, 9, 7, 5, 2,
        # 8, 7, 9, 6, 5, 1, 2, 3, 4,
        # 4, 2, 1, 3, 9, 8, 6, 7, 5,
        # 3, 6, 5, 4, 2, 7, 8, 9, 1,
        # 1, 9, 8, 5, 7, 3, 4, 2, 6,
        # 5, 4, 2, 9, 1, 6, 3, 8, 7]
        ,dtype=np.int8
    )
    # print(sudoku,sudoku.shape)
    # sudoku = sudoku.transpose() #不变
    # print(sudoku,sudoku.shape)

    shape = (6,3)
    strides = sudoku.itemsize * np.array([2,2])
    print("strides:",strides,sudoku.itemsize,sudoku.shape,sudoku.strides)
    # strides=[sudoku.strides[0],1] # 这里是关键
    strides = [1, 3]  # [a,b]这里是关键 每a个元素开始一个节点，每次跳过b个元素，如果b比较大，可能有重复元素，即下一个节点在上一个节点的内部
    # shape = (c,d),而d决定了节点的长度，如果要节点间不重合 则a==d
    squares = np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides)
    print(squares)

    revert_squares = squares.transpose().flatten();
    print(revert_squares)

    shape = (3, 6)
    squares = np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides)
    print(squares)

    # 主要测试flat
    '''
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=np.int8)
    y = np.array([[1,1,1],
                  [2,2,1],
                  [3,3,3]])
    print("x.strides:",x.strides,y.strides)
    # 步伐 大小与dtype有关
    x.strides  # (3, 1)
    print("ddddd ",type(x.strides), x.strides*np.array([1,2]))
    byte_offset = sum(x.strides * np.array([1, 2]))  # 步伐*想要查找的位置
    print("x.flat",x.flat)
    x.flat[byte_offset]  # 6
    x[1, 2]
    '''



TestUnStride()