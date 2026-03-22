import numpy as np
    from scipy.signal import hilbert, medfilt, find_peaks, peak_widths, convolve
    
    # Step 1: Compute sampling rate and time step
    dt = np.median(np.diff(times))
    fs = 1.0/dt if dt > 0 else 1.0

    # Step 2: Compute Hilbert envelopes for H1 and L1
    epsilon = 1e-10
    envelope_h1 = np.abs(hilbert(strain_h1))
    envelope_l1 = np.abs(hilbert(strain_l1))
    
    # Step 3: Remove baseline using a 5-second median filter and clip negatives
    med_kernel = int(5 * fs)
    if med_kernel % 2 == 0:
        med_kernel += 1
    baseline_h1 = medfilt(envelope_h1, kernel_size=med_kernel)
    baseline_l1 = medfilt(envelope_l1, kernel_size=med_kernel)
    envelope_h1_corr = np.clip(envelope_h1 - baseline_h1, a_min=0, a_max=None)
    envelope_l1_corr = np.clip(envelope_l1 - baseline_l1, a_min=0, a_max=None)
    
    # Step 4: Apply base-10 logarithmic scaling for dynamic range enhancement
    log_env_h1 = np.log10(envelope_h1_corr + epsilon)
    log_env_l1 = np.log10(envelope_l1_corr + epsilon)
    
    # Step 5: Combine channels using a weighted mixture (60% geometric mean, 40% arithmetic mean)
    # Calculate geometric mean using base-10 logs --> equivalent to 10**((log_env_h1+log_env_l1)/2)
    geo_mean = 10 ** ((log_env_h1 + log_env_l1) / 2.0)
    arith_mean = (envelope_h1_corr + envelope_l1_corr) / 2.0
    combined_signal = 0.6 * geo_mean + 0.4 * arith_mean

    # Step 6: Smooth the combined signal with a Gaussian filter (sigma ~ 1.2 seconds)
    sigma = 1.2 * fs
    win_length = int(6 * sigma)
    if win_length % 2 == 0:
        win_length += 1
    x = np.linspace(-3 * sigma, 3 * sigma, win_length)
    gauss_win = np.exp(-0.5 * (x/sigma)**2)
    gauss_win /= np.sum(gauss_win)
    smooth_signal = convolve(combined_signal, gauss_win, mode='same')
    
    # Step 7: Determine dynamic threshold using median and MAD (median + 4*mad)
    median_val = np.median(smooth_signal)
    mad = np.median(np.abs(smooth_signal - median_val))
    threshold = median_val + 4 * mad

    # Step 8: Detect peaks with dynamic threshold and prominence, and compute FWHM (with 2.0*mad prominence)
    peaks, properties = find_peaks(smooth_signal, height=threshold, prominence=2.0 * mad)
    widths_results = peak_widths(smooth_signal, peaks, rel_height=0.5)
    widths_in_sec = widths_results[0] * dt

    peak_times = times[peaks]
    peak_heights = properties["peak_heights"]
    peak_deltat = widths_in_sec

    return result
