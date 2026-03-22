import numpy as np
    import scipy.signal as signal

    # Determine time step and sampling rate.
    if len(times) < 2:
        dt = 1.0
    else:
        dt = times[1] - times[0]
    fs = 1.0 / dt
    nyq = 0.5 * fs

    # 1. Bandpass filtering: 4th-order Butterworth filter (30-450 Hz)
    lowcut, highcut, order = 30.0, 450.0, 4
    low = lowcut / nyq
    high = highcut / nyq
    b_bp, a_bp = signal.butter(order, [low, high], btype='band')
    filt_h1 = signal.filtfilt(b_bp, a_bp, strain_h1)
    filt_l1 = signal.filtfilt(b_bp, a_bp, strain_l1)

    # 2. Notch filtering at 60 Hz (Q=35)
    notch_freq, Q = 60.0, 35.0
    notch_norm = notch_freq / nyq
    b_notch, a_notch = signal.iirnotch(notch_norm, Q)
    filt_h1 = signal.filtfilt(b_notch, a_notch, filt_h1)
    filt_l1 = signal.filtfilt(b_notch, a_notch, filt_l1)

    # 3. Pre-whitening: use a moving median filter to remove local low-frequency trends
    med_window = int(fs * 0.1)  # 100 ms window
    if med_window < 1:
        med_window = 1
    if med_window % 2 == 0:
        med_window += 1
    med_h1 = signal.medfilt(filt_h1, kernel_size=med_window)
    med_l1 = signal.medfilt(filt_l1, kernel_size=med_window)
    whitened_h1 = filt_h1 - med_h1
    whitened_l1 = filt_l1 - med_l1

    # 4. Channel fusion: weighted sum and enhanced square-root product fusion
    weighted_sum = 0.65 * whitened_h1 + 0.35 * whitened_l1
    sqrt_product = np.sign(whitened_h1 * whitened_l1) * np.sqrt(np.abs(whitened_h1 * whitened_l1) + 1e-12)
    combined = (weighted_sum + 0.75 * sqrt_product) / 1.75

    # 5. Dual-stage smoothing: median filter then Gaussian smoothing
    smooth_med = signal.medfilt(combined, kernel_size=9)
    gauss_win_len = 39
    std_gauss = 4.0
    gauss_win = signal.windows.gaussian(gauss_win_len, std=std_gauss)
    gauss_win = gauss_win / np.sum(gauss_win)
    smooth_data = np.convolve(smooth_med, gauss_win, mode='same')

    # 6. Robust normalization: subtract median and scale by MAD-derived std
    median_val = np.median(smooth_data)
    mad_val = np.median(np.abs(smooth_data - median_val))
    robust_std = mad_val * 1.4826 if mad_val > 0 else np.std(smooth_data)
    norm_data = (smooth_data - median_val) / robust_std if robust_std != 0 else smooth_data - median_val

    # 7. Adaptive peak detection: using dynamic threshold and prominence
    threshold = 2.1
    prominence_val = 1.5
    peaks, properties = signal.find_peaks(norm_data, height=threshold, prominence=prominence_val)

    # 8. Compute peak widths at 65% of peak height and convert to seconds
    widths_result = signal.peak_widths(norm_data, peaks, rel_height=0.65)
    peak_deltat = widths_result[0] * dt

    # 9. Extract GPS times and peak heights from the times array
    peak_times = times[peaks]
    peak_heights = properties['peak_heights']

    result = (np.array(peak_times), np.array(peak_heights), np.array(peak_deltat))
    return result
