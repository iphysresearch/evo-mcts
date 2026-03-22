import numpy as np
    from scipy.signal import butter, filtfilt, hilbert, detrend, find_peaks, peak_widths, savgol_filter
    from scipy.ndimage import gaussian_filter1d

    # Calculate time resolution and sampling frequency.
    dt = np.median(np.diff(times))
    fs = 1.0 / dt

    # Detrend the input strain data for both detectors.
    proc_h1 = detrend(strain_h1)
    proc_l1 = detrend(strain_l1)
    
    # Remove low-frequency baselines using a Savitzky–Golay filter (~0.5 second window, polyorder=3).
    window_size = int(0.5 / dt)
    if window_size % 2 == 0:
        window_size += 1
    baseline_h1 = savgol_filter(proc_h1, window_length=window_size, polyorder=3)
    baseline_l1 = savgol_filter(proc_l1, window_length=window_size, polyorder=3)
    proc_h1 = proc_h1 - baseline_h1
    proc_l1 = proc_l1 - baseline_l1

    # Design and apply a 6th-order Butterworth bandpass filter (20-350 Hz).
    lowcut = 20.0
    highcut = 350.0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=6, Wn=[low, high], btype='band')
    filt_h1 = filtfilt(b, a, proc_h1)
    filt_l1 = filtfilt(b, a, proc_l1)

    # Compute analytic envelopes via the Hilbert transform.
    envelope_h1 = np.abs(hilbert(filt_h1))
    envelope_l1 = np.abs(hilbert(filt_l1))
    
    # Define sliding-window parameters: 3-second window with 1.5-second overlap.
    window_sec = 3.0
    win_size = int(window_sec / dt)
    if win_size < 1:
        win_size = 1
    step = win_size // 2 if win_size // 2 > 0 else 1

    similarity_values = []
    window_centers = []
    n = len(times)

    # Compute sliding-window Pearson correlation coefficient.
    for start in range(0, n - win_size + 1, step):
        end = start + win_size
        seg1 = envelope_h1[start:end]
        seg2 = envelope_l1[start:end]
        mean1 = np.mean(seg1)
        mean2 = np.mean(seg2)
        std1 = np.std(seg1)
        std2 = np.std(seg2)
        if std1 > 0 and std2 > 0:
            corr = np.sum((seg1 - mean1) * (seg2 - mean2)) / (win_size * std1 * std2)
        else:
            corr = 0.0
        similarity_values.append(corr)
        center_time = times[start + win_size // 2]
        window_centers.append(center_time)

    similarity_values = np.array(similarity_values)
    window_centers = np.array(window_centers)

    # Smooth the similarity time series using a Gaussian filter with sigma=3.
    smooth_similarity = gaussian_filter1d(similarity_values, sigma=3)

    # Define an adaptive threshold using the 90th percentile.
    threshold = np.percentile(smooth_similarity, 90)

    # Identify peaks exceeding the threshold.
    peaks, props = find_peaks(smooth_similarity, height=threshold)
    peak_heights = props['peak_heights']
    peak_times = window_centers[peaks]

    # Estimate timing uncertainty using the widths at 60% relative height.
    widths = peak_widths(smooth_similarity, peaks, rel_height=0.6)[0]
    estimated_dt = step * dt
    peak_deltat = widths * estimated_dt

    return result
