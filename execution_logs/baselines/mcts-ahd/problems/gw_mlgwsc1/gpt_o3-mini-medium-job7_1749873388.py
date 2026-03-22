import numpy as np
import scipy.signal

def pipeline_v2(strain_h1, strain_l1, times):
    # Estimate sampling frequency and time step.
    fs = 1.0 / np.median(np.diff(times))
    dt = 1.0 / fs

    # Robust detrending using a 3-second median filter.
    kernel_size = int(3.0 * fs)
    if kernel_size % 2 == 0:
        kernel_size += 1
    h1_det = strain_h1 - scipy.signal.medfilt(strain_h1, kernel_size)
    l1_det = strain_l1 - scipy.signal.medfilt(strain_l1, kernel_size)

    # Apply a Chebyshev Type I bandpass filter (passband: 30-425 Hz, 1 dB ripple).
    lowcut, highcut = 30.0, 425.0
    ripple = 1.0  # dB passband ripple
    nyquist = 0.5 * fs
    low_norm = lowcut / nyquist
    high_norm = highcut / nyquist
    order = 4
    b_band, a_band = scipy.signal.cheby1(order, ripple, [low_norm, high_norm], btype='band')
    h1_filt = scipy.signal.filtfilt(b_band, a_band, h1_det)
    l1_filt = scipy.signal.filtfilt(b_band, a_band, l1_det)

    # (Optional) Remove 60 Hz interference using a notch filter (Q=30).
    notch_freq = 60.0
    Q = 30.0
    w0 = notch_freq / nyquist
    b_notch, a_notch = scipy.signal.iirnotch(w0, Q)
    h1_filt = scipy.signal.filtfilt(b_notch, a_notch, h1_filt)
    l1_filt = scipy.signal.filtfilt(b_notch, a_notch, l1_filt)

    # Set sliding window parameters (similar to No.2) and incorporate exponential weighting.
    window_size = 256
    step_size = 128
    n_windows = (len(h1_filt) - window_size) // step_size + 1
    corr_values = np.zeros(n_windows)
    corr_times = np.zeros(n_windows)

    # Precompute exponential weights.
    win_indices = np.arange(window_size)
    exp_weights = np.exp(-win_indices / (0.5 * window_size))

    # Compute exponentially weighted Pearson correlation.
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        seg_h1 = h1_filt[start:end] * exp_weights
        seg_l1 = l1_filt[start:end] * exp_weights
        std_h1 = np.std(seg_h1)
        std_l1 = np.std(seg_l1)
        if std_h1 == 0 or std_l1 == 0:
            corr = 0.0
        else:
            mean_h1 = np.mean(seg_h1)
            mean_l1 = np.mean(seg_l1)
            cov = np.mean((seg_h1 - mean_h1) * (seg_l1 - mean_l1))
            corr = cov / (std_h1 * std_l1)
        corr_values[i] = corr
        # Time stamp at the center of the window.
        corr_times[i] = times[start + window_size // 2]

    # Smooth the correlation time series with a Gaussian kernel.
    kernel_len = 15
    sigma = 3.0
    t_kernel = np.linspace(-int(kernel_len//2), int(kernel_len//2), kernel_len)
    gauss_kernel = np.exp(-t_kernel**2 / (2 * sigma**2))
    gauss_kernel /= np.sum(gauss_kernel)
    corr_smooth = np.convolve(corr_values, gauss_kernel, mode='same')

    # Adaptive thresholding using median and MAD with a slightly lower scaling factor.
    med_corr = np.median(corr_smooth)
    mad_corr = np.median(np.abs(corr_smooth - med_corr))
    thresh = med_corr + 1.8 * mad_corr

    # Detect peaks above the threshold with a minimum separation.
    peaks, properties = scipy.signal.find_peaks(corr_smooth, height=thresh, distance=3)
    peak_times = corr_times[peaks]
    peak_heights = properties["peak_heights"]
    
    # Estimate timing uncertainty via peak widths.
    dt_corr = step_size * dt
    widths, _, _, _ = scipy.signal.peak_widths(corr_smooth, peaks, rel_height=0.5)
    peak_deltat = widths * dt_corr

    result = (np.array(peak_times), np.array(peak_heights), np.array(peak_deltat))
    return result
