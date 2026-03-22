import numpy as np
from scipy.signal import stft, savgol_filter, hilbert, medfilt
from scipy.ndimage import gaussian_filter1d

def pipeline_v2(strain_h1, strain_l1, times):
    # Sampling parameters
    dt = np.median(np.diff(times))
    fs = 1.0 / dt
    t0 = times[0]
    
    # Combine detector data with inverse‐variance weighting
    std_h1 = np.std(strain_h1) if np.std(strain_h1) > 0 else 1.0
    std_l1 = np.std(strain_l1) if np.std(strain_l1) > 0 else 1.0
    w1 = 1.0 / std_h1
    w2 = 1.0 / std_l1
    combined_strain = (w1 * strain_h1 + w2 * strain_l1) / (w1 + w2)
    
    # Robust centering: subtract median
    norm_signal = combined_strain - np.median(combined_strain)
    
    # Baseline removal: average of median filtering and Savitzky–Golay filtering with new window lengths
    med_window = int(4.0 * fs)
    if med_window % 2 == 0:
        med_window += 1
    if med_window < 5:
        med_window = 5
    baseline_med = medfilt(norm_signal, kernel_size=med_window)
    
    sg_window = int(3.0 * fs)
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window < 5:
        sg_window = 5
    baseline_sg = savgol_filter(norm_signal, window_length=sg_window, polyorder=2)
    
    baseline_combined = 0.55 * baseline_med + 0.45 * baseline_sg
    detrended_strain = norm_signal - baseline_combined
    
    # Enhance transient features using non-linear Hilbert envelope with a changed exponent (1.2)
    analytic_signal = hilbert(detrended_strain)
    envelope = np.abs(analytic_signal)
    enhanced_signal = envelope ** 1.2
    
    # Compute STFT with 2.0-second Hann window and 75% overlap
    nperseg = int(2.0 / dt)
    if nperseg < 8:
        nperseg = 8
    noverlap = int(0.75 * nperseg)
    f, t_spec, Zxx = stft(enhanced_signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    
    # Focus on the frequency band between 30 and 600 Hz and sum the spectral energy
    freq_mask = (f >= 30.0) & (f <= 600.0)
    energy_spectrogram = np.abs(Zxx)**2
    energy_profile = np.sum(energy_spectrogram[freq_mask, :], axis=0)
    
    # Smooth the energy profile using a Gaussian filter with a larger sigma
    energy_smoothed = gaussian_filter1d(energy_profile, sigma=3.0)
    
    # Compute robust statistics for dynamic thresholding: median, IQR, and MAD
    median_val = np.median(energy_smoothed)
    iqr_val = np.percentile(energy_smoothed, 75) - np.percentile(energy_smoothed, 25)
    mad_val = np.median(np.abs(energy_smoothed - median_val))
    threshold = median_val + 1.0 * (iqr_val + mad_val)
    
    # Detect candidate peaks: local maxima above the threshold
    peak_indices = []
    for i in range(1, len(energy_smoothed) - 1):
        if energy_smoothed[i] > energy_smoothed[i - 1] and energy_smoothed[i] > energy_smoothed[i + 1]:
            if energy_smoothed[i] > threshold:
                peak_indices.append(i)
    peak_indices = np.array(peak_indices)
    
    # Refine peaks using valley-boundary analysis to extract trigger parameters
    peak_times = []
    peak_heights = []
    peak_deltat = []
    for idx in peak_indices:
        current_peak_time = t0 + t_spec[idx]
        current_peak_height = energy_smoothed[idx]
        
        # Find left valley boundary
        left = idx
        while left > 0 and energy_smoothed[left] >= energy_smoothed[left - 1]:
            left -= 1
        # Find right valley boundary
        right = idx
        while right < len(energy_smoothed) - 1 and energy_smoothed[right] >= energy_smoothed[right + 1]:
            right += 1
        
        time_window = t_spec[right] - t_spec[left]
        
        peak_times.append(current_peak_time)
        peak_heights.append(current_peak_height)
        peak_deltat.append(time_window)
    
    result = (np.array(peak_times), np.array(peak_heights), np.array(peak_deltat))
    return result
