import numpy as np
import pywt
from scipy import signal
import warnings
from sklearn.ensemble import IsolationForest

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhanced gravitational wave detection pipeline (pipeline_v2) that integrates adaptive segmentation,
    multi-scale time-frequency analysis via both STFT and wavelet transforms, inter-detector coherence,
    and lightweight ML-based artifact screening via an IsolationForest.
    
    Processes dual-channel gravitational wave strain data (H1 and L1) and returns candidate trigger information:
    GPS times (peak_times), ranking statistics (peak_heights), and estimated timing uncertainties (peak_deltat).
    
    Parameters:
      strain_h1 : numpy.ndarray
          Raw strain data from the H1 detector.
      strain_l1 : numpy.ndarray
          Raw strain data from the L1 detector.
      times : numpy.ndarray
          Array of GPS timestamps corresponding to strain samples.
    
    Returns:
      peak_times : numpy.ndarray
          Array of candidate GPS trigger times.
      peak_heights : numpy.ndarray
          Ranking statistics (signal significance) for each candidate trigger.
      peak_deltat : numpy.ndarray
          Timing uncertainties (in seconds) for each candidate trigger.
    """
    eps = 1e-8
    dt = np.median(np.diff(times)) if times.size > 1 else 1.0
    fs = 1.0 / dt

    def adaptive_filter_and_whiten(signal_in: np.ndarray) -> np.ndarray:
        # Baseline correction and detrending
        baseline = np.median(signal_in)
        signal_detrended = signal.detrend(signal_in - baseline)
        
        # Estimate noise level using MAD
        mad_val = np.median(np.abs(signal_detrended - np.median(signal_detrended))) + eps
        adapt_factor = np.clip(mad_val, 0.5, 2.5)
        
        # Adaptive bandpass: typical gravitational wave band [30,600] Hz, adjusted by adapt_factor.
        lowcut = np.clip(30.0 * adapt_factor, 25.0, 75.0)
        highcut = np.clip(600.0 / adapt_factor, 400.0, 750.0)
        nyq = 0.5 * fs
        try:
            b_bp, a_bp = signal.butter(4, [lowcut/nyq, highcut/nyq], btype='band')
            filtered = signal.filtfilt(b_bp, a_bp, signal_detrended)
        except Exception as e:
            warnings.warn("Bandpass filtering failed; returning detrended signal. Exception: " + str(e))
            filtered = signal_detrended
        
        # Estimate power spectral density (PSD) using Welch's method on an adaptive segment length.
        nperseg = 2048 if len(filtered) >= 2048 else max(256, len(filtered) // 2)
        try:
            freqs, psd = signal.welch(filtered, fs=fs, nperseg=nperseg, window='hann', noverlap=nperseg // 2)
        except Exception as e:
            warnings.warn("PSD estimation failed with adaptive nperseg, using defaults. Exception: " + str(e))
            freqs, psd = signal.welch(filtered, fs=fs)
        
        # Smooth PSD using a Gaussian window
        win_len = 128
        std_dev = 12
        gauss_win = signal.windows.gaussian(win_len, std=std_dev)
        gauss_win /= np.sum(gauss_win)
        smooth_psd = np.convolve(psd, gauss_win, mode='same')
        smooth_psd = np.maximum(smooth_psd, np.finfo(float).tiny)
        
        # Whitening: FFT normalization using interpolated PSD.
        fft_data = np.fft.rfft(filtered)
        freq_vals = np.fft.rfftfreq(len(filtered), d=dt)
        interp_psd = np.interp(freq_vals, freqs, smooth_psd)
        white_fft = fft_data / np.sqrt(interp_psd)
        whitened = np.fft.irfft(white_fft, n=len(filtered))
        
        # Apply median filtering to remove short-duration artifacts.
        whitened = signal.medfilt(whitened, kernel_size=5)
        return whitened

    # Process both channels.
    white_h1 = adaptive_filter_and_whiten(strain_h1)
    white_l1 = adaptive_filter_and_whiten(strain_l1)

    # Compute dynamic STFT segmentation.
    seg_length = len(white_h1)
    nperseg = int(np.clip(seg_length // 100, 256, 1024))
    noverlap = int(nperseg * 0.75)
    stft_win = signal.windows.hann(nperseg, sym=False)
    
    f_h1, t_spec, Zxx_h1 = signal.stft(white_h1, fs=fs, window=stft_win,
                                       nperseg=nperseg, noverlap=noverlap, detrend=False)
    f_l1, _, Zxx_l1 = signal.stft(white_l1, fs=fs, window=stft_win,
                                  nperseg=nperseg, noverlap=noverlap, detrend=False)
    
    # Compute power spectrograms.
    power_h1 = np.abs(Zxx_h1)**2
    power_l1 = np.abs(Zxx_l1)**2
    
    # Local noise estimation: normalize each frequency bin by its median power.
    med_power_h1 = np.median(power_h1, axis=1, keepdims=True) + eps
    med_power_l1 = np.median(power_l1, axis=1, keepdims=True) + eps
    snr_h1 = power_h1 / med_power_h1
    snr_l1 = power_l1 / med_power_l1

    # Adaptive fusion: weight power from each detector.
    weight_h1 = (snr_h1 / (snr_h1 + snr_l1 + eps)) ** 1.2
    weight_l1 = (snr_l1 / (snr_h1 + snr_l1 + eps)) ** 1.2
    fused_power = power_h1 * weight_h1 + power_l1 * weight_l1

    # Inter-detector coherence: assess phase consistency.
    cross_spec = Zxx_h1 * np.conj(Zxx_l1)
    norm_factor = (np.abs(Zxx_h1) * np.abs(Zxx_l1)) + eps
    coherence = np.clip(np.abs(cross_spec) / norm_factor, 0, 1)
    weight_power = np.maximum(np.median(fused_power, axis=1, keepdims=True), eps)
    coherence_time = np.average(coherence, axis=0, weights=weight_power.flatten())

    # Base metric from STFT.
    base_metric = np.median(fused_power, axis=0)
    stft_metric = base_metric * (0.5 + 0.5 * (coherence_time ** 1.3))
    
    # Wavelet transform based multi-scale analysis.
    scales = np.arange(1, 128)
    coef_h1, _ = pywt.cwt(white_h1, scales, 'morl', sampling_period=dt)
    power_wavelet_h1 = np.mean(np.abs(coef_h1)**2, axis=0)
    coef_l1, _ = pywt.cwt(white_l1, scales, 'morl', sampling_period=dt)
    power_wavelet_l1 = np.mean(np.abs(coef_l1)**2, axis=0)
    
    # Fuse wavelet powers from both detectors.
    fused_wavelet = (power_wavelet_h1 + power_wavelet_l1) / 2.0
    # Resample wavelet metric to match STFT time resolution.
    wavelet_metric = signal.resample(fused_wavelet, len(t_spec))
    
    # Combine STFT and wavelet metrics.
    final_metric = 0.5 * (stft_metric + wavelet_metric)
    
    # Map STFT time bins to absolute GPS times (reference from times[0]).
    metric_times = times[0] + t_spec
    
    # Iterative threshold tuning based on robust median and MAD of final_metric.
    med_val = np.median(final_metric)
    mad_val = np.median(np.abs(final_metric - med_val)) + eps
    robust_std = 1.4826 * mad_val
    k_factor = 3.5
    dynamic_threshold = med_val + k_factor * robust_std

    # Refine threshold iteratively by excluding candidate peaks.
    for _ in range(3):
        candidate_idx, _ = signal.find_peaks(final_metric, height=dynamic_threshold, prominence=robust_std, distance=3)
        if candidate_idx.size > 0 and candidate_idx.size < final_metric.size:
            noise_data = np.delete(final_metric, candidate_idx)
            if noise_data.size > 0:
                med_val = np.median(noise_data)
                mad_val = np.median(np.abs(noise_data - med_val)) + eps
                robust_std = 1.4826 * mad_val
                dynamic_threshold = med_val + k_factor * robust_std
            else:
                break
        else:
            break

    dt_spec = t_spec[1] - t_spec[0] if t_spec.size > 1 else dt
    min_distance = max(1, int(0.5 / dt_spec))
    
    # Detect candidate peaks.
    peaks, peak_props = signal.find_peaks(final_metric, height=dynamic_threshold, prominence=robust_std, distance=min_distance)
    if peaks.size == 0:
        dynamic_threshold = med_val + (k_factor * robust_std * 0.8)
        peaks, peak_props = signal.find_peaks(final_metric, height=dynamic_threshold, prominence=robust_std, distance=min_distance)
    
    # Refine peaks: compute multi-scale widths and initial feature extraction.
    refined_peak_times = []
    refined_peak_heights = []
    refined_peak_widths = []
    for peak in peaks:
        local_widths = []
        for win_scale in [5, 7, 9]:
            win_start = max(peak - win_scale, 0)
            win_end = min(peak + win_scale + 1, len(final_metric))
            # Compute widths using the entire segment (using peak_widths function)
            widths, _, _, _ = signal.peak_widths(final_metric, [peak], rel_height=0.5)
            local_widths.append(widths[0] * dt_spec)
        width_sec = np.mean(local_widths)
        # Extra uncertainty inversely linked to relative prominence.
        peak_val = final_metric[peak]
        local_bg = np.median(final_metric[max(peak - 5, 0):min(peak + 6, len(final_metric))])
        norm_peak = (peak_val - local_bg) / (np.max(final_metric) - local_bg + eps)
        extra_uncert = 8.0 / (norm_peak + eps)
        total_width = np.clip(width_sec + extra_uncert, 1.0, 12.0)
        
        refined_peak_times.append(metric_times[peak])
        refined_peak_heights.append(peak_val)
        refined_peak_widths.append(total_width)
    
    # Assemble candidate features for ML-based artifact screening.
    # Features: [relative time index, peak height, width estimate]
    if len(refined_peak_times) > 0:
        features = np.column_stack([peaks, refined_peak_heights, refined_peak_widths])
        # Use IsolationForest to flag likely artifact triggers.
        clf = IsolationForest(contamination=0.1, random_state=42)
        labels = clf.fit_predict(features)
        # Only retain candidates labeled as normal (1) by the isolation forest.
        keep_mask = labels == 1
        refined_peak_times = np.array(refined_peak_times)[keep_mask]
        refined_peak_heights = np.array(refined_peak_heights)[keep_mask]
        refined_peak_widths = np.array(refined_peak_widths)[keep_mask]
    else:
        refined_peak_times = np.array([])
        refined_peak_heights = np.array([])
        refined_peak_widths = np.array([])
    
    return peak_times, peak_heights, peak_deltat
