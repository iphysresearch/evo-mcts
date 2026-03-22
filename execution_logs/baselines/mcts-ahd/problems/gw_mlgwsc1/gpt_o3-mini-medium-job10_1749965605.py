import numpy as np
    from scipy.signal import cheby1, filtfilt, hilbert, find_peaks, peak_widths, wiener
    # Compute sampling parameters
    dt = np.mean(np.diff(times))
    fs = 1.0 / dt
    
    # Detrend and robust normalization using Median Absolute Deviation (MAD)
    def detrend_and_normalize(signal):
        # Remove linear trend
        t = np.arange(len(signal))
        coeffs = np.polyfit(t, signal, 1)
        trend = np.polyval(coeffs, t)
        detrended = signal - trend
        # Robust normalization using MAD
        med = np.median(detrended)
        mad = np.median(np.abs(detrended - med))
        norm_signal = (detrended - med) / (1.4826 * mad + 1e-12)
        return norm_signal
    
    norm_h1 = detrend_and_normalize(strain_h1)
    norm_l1 = detrend_and_normalize(strain_l1)
    
    # Chebyshev Type I bandpass filter (30-500 Hz, order 4, 1 dB ripple)
    lowcut = 30.0
    highcut = 500.0
    order = 4
    ripple = 1  # dB
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, ripple, [low, high], btype='band')
    
    # Filter the normalized signals
    h1_filt = filtfilt(b, a, norm_h1)
    l1_filt = filtfilt(b, a, norm_l1)
    
    # Combine filtered channels using harmonic mean to boost coherent signal components
    eps = 1e-12
    # Avoid division by zero by adding a small constant
    harmonic_mean = 2 * (h1_filt * l1_filt) / (h1_filt + l1_filt + eps)
    # If signs differ, average instead
    mask = np.sign(h1_filt) != np.sign(l1_filt)
    combined_signal = harmonic_mean
    combined_signal[mask] = 0.5 * (h1_filt[mask] + l1_filt[mask])
    
    # Compute analytic signal and envelope, then apply logarithmic compression
    analytic_signal = hilbert(combined_signal)
    envelope = np.abs(analytic_signal)
    log_env = np.log1p(envelope)  # natural log compression
    
    # Apply Wiener filtering for noise suppression
    wiener_smoothed = wiener(log_env, mysize=int(fs*0.1))
    
    # Further smooth using an exponential moving average (EMA)
    alpha = 0.15
    ema_smoothed = np.zeros_like(wiener_smoothed)
    ema_smoothed[0] = wiener_smoothed[0]
    for i in range(1, len(wiener_smoothed)):
        ema_smoothed[i] = alpha * wiener_smoothed[i] + (1 - alpha) * ema_smoothed[i-1]
    
    # Dynamic threshold based on median and 90th percentile
    med_val = np.median(ema_smoothed)
    perc90 = np.percentile(ema_smoothed, 90)
    threshold = med_val + 0.6 * (perc90 - med_val)
    
    # Detect peaks with a minimum spacing of 200 ms
    min_distance = int(fs * 0.2)
    peaks, properties = find_peaks(ema_smoothed, height=threshold, distance=min_distance)
    peak_times = times[peaks]
    peak_heights = properties['peak_heights']
    
    # Estimate timing uncertainty via modified peak width estimation using half maximum criteria
    widths_samples, _, left_ips, right_ips = peak_widths(ema_smoothed, peaks, rel_height=0.5)
    peak_deltat = []
    for i, peak in enumerate(peaks):
        left_bound = int(max(0, np.floor(left_ips[i])))
        right_bound = int(min(len(ema_smoothed)-1, np.ceil(right_ips[i])))
        window_idx = np.arange(left_bound, right_bound + 1)
        window_signal = ema_smoothed[left_bound:right_bound + 1]
        baseline = np.median(window_signal)
        half_level = baseline + 0.5 * (ema_smoothed[peak] - baseline)
        valid = window_signal >= half_level
        if np.sum(valid) < 2:
            eff_width = widths_samples[i]
        else:
            valid_idx = window_idx[valid]
            weights = window_signal[valid] - baseline + 1e-6
            mean_idx = np.sum(valid_idx * weights) / np.sum(weights)
            variance = np.sum(weights * (valid_idx - mean_idx)**2) / np.sum(weights)
            sigma_val = np.sqrt(variance)
            eff_width = 2 * sigma_val / dt
        peak_deltat.append(eff_width * dt)
    peak_deltat = np.array(peak_deltat)
    
    result = (np.array(peak_times), np.array(peak_heights), np.array(peak_deltat))
    return result
