import numpy as np
import scipy.signal
import scipy.ndimage
from scipy.signal import savgol_filter

def pipeline_v2(strain_h1, strain_l1, times):
    dt = times[1] - times[0]
    nyquist = 0.5 / dt
    
    # Apply 6th order Butterworth bandpass filter between 20-340 Hz
    low_cut = 20 / nyquist
    high_cut = 340 / nyquist
    b, a = scipy.signal.butter(6, [low_cut, high_cut], btype='band')
    filt_h1 = scipy.signal.filtfilt(b, a, strain_h1)
    filt_l1 = scipy.signal.filtfilt(b, a, strain_l1)
    
    # Robust normalization using median and interquartile range with tanh mapping
    def robust_norm_iqr(data):
        med = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25 + 1e-6
        zscore = (data - med) / iqr
        # Use tanh mapping for nonlinearity
        return 0.5 * (1 + np.tanh(zscore))
    
    norm_h1 = robust_norm_iqr(filt_h1)
    norm_l1 = robust_norm_iqr(filt_l1)
    
    # Fuse channels using a weighted average (weights chosen as relative energies)
    energy_h1 = np.sum(norm_h1**2)
    energy_l1 = np.sum(norm_l1**2)
    if energy_h1 + energy_l1 < 1e-6:
        weight_h1 = weight_l1 = 0.5
    else:
        weight_h1 = energy_h1 / (energy_h1 + energy_l1)
        weight_l1 = energy_l1 / (energy_h1 + energy_l1)
    fused = weight_h1 * norm_h1 + weight_l1 * norm_l1
    
    # Extract envelope via a modified Teager-Kaiser Energy Operator (TKEO)
    # TKEO: psi[x(t)] = x(t)^2 - x(t-1)*x(t+1)
    tkeo = np.zeros_like(fused)
    tkeo[1:-1] = fused[1:-1]**2 - fused[:-2]*fused[2:]
    # Rectify and take absolute value to form an envelope estimate:
    envelope = np.abs(tkeo)
    
    # Smooth the envelope using Savitzky-Golay filter.
    # Choose window length based on 0.08 sec duration, must be odd
    win_size = int(np.round(0.08 / dt))
    if win_size % 2 == 0:
        win_size += 1
    if win_size < 5:
        win_size = 5
    smooth_env = savgol_filter(envelope, window_length=win_size, polyorder=2)
    
    # Adaptive threshold: median plus 3.0 times the interquartile range
    med_env = np.median(smooth_env)
    q75_env, q25_env = np.percentile(smooth_env, [75, 25])
    iqr_env = q75_env - q25_env
    threshold = med_env + 3.0 * iqr_env
    
    # Detect peaks above the threshold with a minimum separation of 0.08 sec.
    min_distance = int(0.08 / dt)
    peaks, properties = scipy.signal.find_peaks(smooth_env, height=threshold, distance=min_distance)
    
    refined_times = []
    refined_heights = []
    peak_deltat = []
    half_window_samples = int(0.08 / dt)
    
    for peak in peaks:
        start = max(0, peak - half_window_samples)
        end = min(len(times), peak + half_window_samples + 1)
        local_times = times[start:end]
        local_signal = smooth_env[start:end]
        
        # Use weighted average to refine peak time and compute effective width.
        w = local_signal - np.min(local_signal) + 1e-6  # weights must be positive
        refined_time = np.sum(local_times * w) / np.sum(w)
        refined_height = np.max(local_signal)
        # Compute weighted standard deviation as an estimate of width uncertainty.
        variance = np.sum(w * (local_times - refined_time)**2) / np.sum(w)
        width = 2 * np.sqrt(variance) if variance > 0 else 0.08
        
        refined_times.append(refined_time)
        refined_heights.append(refined_height)
        peak_deltat.append(width)
    
    peak_times = np.array(refined_times)
    peak_heights = np.array(refined_heights)
    peak_deltat = np.array(peak_deltat)
    
    result = (peak_times, peak_heights, peak_deltat)
    return result
