import numpy as np
from scipy import signal

def pipeline_v2(strain_h1, strain_l1, times):
    eps = 1e-6
    # Step 1: Robust combination using squared MAD-based weights (inverse variance style)
    med_h1 = np.median(strain_h1)
    med_l1 = np.median(strain_l1)
    mad_h1 = np.median(np.abs(strain_h1 - med_h1))
    mad_l1 = np.median(np.abs(strain_l1 - med_l1))
    weight_h1 = 1.0 / ((mad_h1 + eps)**2)
    weight_l1 = 1.0 / ((mad_l1 + eps)**2)
    combined = (weight_h1 * strain_h1 + weight_l1 * strain_l1) / (weight_h1 + weight_l1)
    
    # Step 2: Outlier clipping using median and IQR, and detrending with Savitzky-Golay filter
    med_comb = np.median(combined)
    q1 = np.percentile(combined, 25)
    q3 = np.percentile(combined, 75)
    iqr = q3 - q1
    clip_min = med_comb - 4.0 * iqr
    clip_max = med_comb + 4.0 * iqr
    combined = np.clip(combined, clip_min, clip_max)
    
    # Detrend using a Savitzky-Golay filter to extract the slow trend and subtract it
    # Choose window length as ~1 second worth of samples (must be odd) and polynomial order 2.
    dt = np.median(np.diff(times))
    win_length = max(int(1.0 / dt), 3)
    if win_length % 2 == 0:
        win_length += 1
    trend = signal.savgol_filter(combined, window_length=win_length, polyorder=2)
    combined = combined - trend
    
    # Step 3: Sampling parameters
    dt = np.median(np.diff(times))
    fs = 1.0 / dt
    nyq = 0.5 * fs
    
    # Step 4: Chebyshev Type I bandpass filter from 30 to 320 Hz
    lowcut = 30.0
    highcut = 320.0
    low = lowcut / nyq
    high = highcut / nyq
    ripple = 0.5  # dB of ripple in the passband
    order = 4
    b_bp, a_bp = signal.cheby1(order, ripple, [low, high], btype='band')
    filtered = signal.filtfilt(b_bp, a_bp, combined)
    
    # Step 5: Sequential notch filtering at 60, 120, and 180 Hz with Q factor 30
    for notch_freq in [60.0, 120.0, 180.0]:
        w0 = notch_freq / nyq
        Q = 30.0
        b_notch, a_notch = signal.iirnotch(w0, Q)
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
    
    # Step 6: Compute analytic signal envelope using the Hilbert transform
    analytic_signal = signal.hilbert(filtered)
    envelope = np.abs(analytic_signal)
    
    # Step 7: Smooth the envelope using Savitzky-Golay filter
    # Use a window length of ~0.3 sec worth and polynomial order 2.
    win_length_env = max(int(0.3 / dt), 3)
    if win_length_env % 2 == 0:
        win_length_env += 1
    envelope_smooth = signal.savgol_filter(envelope, window_length=win_length_env, polyorder=2)
    
    # Step 8: Two-tier adaptive thresholding
    # First tier: use median + 1.2 * (IQR) as threshold
    med_env = np.median(envelope_smooth)
    q1_env = np.percentile(envelope_smooth, 25)
    q3_env = np.percentile(envelope_smooth, 75)
    iqr_env = q3_env - q1_env
    low_threshold = med_env + 1.2 * iqr_env
    # Second tier: mean + 3.0 * std threshold
    mean_env = np.mean(envelope_smooth)
    std_env = np.std(envelope_smooth)
    high_threshold = mean_env + 3.0 * std_env
    
    # Step 9: Peak detection with enforced minimum separation ~0.15 sec
    min_distance = int(0.15 / dt)
    candidate_peaks, properties = signal.find_peaks(envelope_smooth, height=low_threshold, distance=min_distance)
    if candidate_peaks.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    
    # Retain peaks that also pass the high threshold
    valid_idx = candidate_peaks[properties['peak_heights'] >= high_threshold]
    if valid_idx.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    peak_times = times[valid_idx]
    peak_heights = envelope_smooth[valid_idx]
    
    # Step 10: Compute timing uncertainty using peak widths at 0.6 height relative level, with minimum of 0.5 sec
    widths_result = signal.peak_widths(envelope_smooth, valid_idx, rel_height=0.6)
    widths_sec = widths_result[0] * dt
    # Estimate timing uncertainty inversely proportional to peak height and ensure a minimum duration.
    peak_deltat = np.maximum(widths_sec / (peak_heights + eps), 0.5)
    
    result = (peak_times, peak_heights, peak_deltat)
    return result
