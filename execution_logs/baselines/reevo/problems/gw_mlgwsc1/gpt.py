import numpy as np
import warnings
from scipy import signal

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhanced gravitational wave detection pipeline (pipeline_v2) featuring dynamic segmentation,
    advanced local noise estimation, iterative threshold tuning, and improved inter-detector coherence integration.

    Processes dual-channel gravitational wave strain data from H1 and L1 detectors and returns candidate
    trigger information: GPS times, ranking statistics (signal significance), and timing uncertainties.

    Parameters:
      strain_h1 : numpy.ndarray
          Raw strain data from the H1 detector.
      strain_l1 : numpy.ndarray
          Raw strain data from the L1 detector.
      times : numpy.ndarray
          Array of GPS timestamps corresponding to the strain samples.

    Returns:
      peak_times : numpy.ndarray
          Array of candidate GPS trigger times.
      peak_heights : numpy.ndarray
          Ranking statistics for each candidate trigger.
      peak_deltat : numpy.ndarray
          Estimated timing uncertainties (in seconds) for each candidate trigger.
    """
    eps = 1e-8
    dt = np.median(np.diff(times)) if times.size > 1 else 1.0
    fs = 1.0 / dt

    def adaptive_filter_and_whiten(signal_in: np.ndarray) -> np.ndarray:
        # Robust baseline correction and detrending (remove DC component)
        baseline = np.median(signal_in)
        signal_detrended = signal.detrend(signal_in - baseline)
        
        # Estimate noise level using Median Absolute Deviation (MAD)
        mad_val = np.median(np.abs(signal_detrended - np.median(signal_detrended))) + eps
        adapt_factor = np.clip(mad_val, 0.5, 2.5)
        
        # Adaptive bandpass: typical gravitational wave signals are in 30-600 Hz region, adjust bounds by adapt_factor.
        lowcut = np.clip(30.0 * adapt_factor, 25.0, 75.0)
        highcut = np.clip(600.0 / adapt_factor, 400.0, 750.0)
        nyq = 0.5 * fs
        try:
            b_bp, a_bp = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')
            filtered = signal.filtfilt(b_bp, a_bp, signal_detrended)
        except Exception as e:
            warnings.warn("Bandpass filtering failed; returning detrended signal. Exception: " + str(e))
            filtered = signal_detrended
        
        # Estimate the power spectral density (PSD) using Welch's method with adaptive segment length
        nperseg = 2048 if len(filtered) >= 2048 else max(256, len(filtered) // 2)
        try:
            freqs, psd = signal.welch(filtered, fs=fs, nperseg=nperseg, window='hann', noverlap=nperseg // 2)
        except Exception as e:
            warnings.warn("PSD estimation failed with adaptive nperseg, using defaults. Exception: " + str(e))
            freqs, psd = signal.welch(filtered, fs=fs)
        
        # Smooth the PSD using a Gaussian filter to reduce narrowband artifacts
        win_len = 128
        std_dev = 12
        gauss_win = signal.windows.gaussian(win_len, std=std_dev)
        gauss_win /= np.sum(gauss_win)
        smooth_psd = np.convolve(psd, gauss_win, mode='same')
        smooth_psd = np.maximum(smooth_psd, np.finfo(float).tiny)
        
        # Whitening in frequency-domain via FFT normalization using interpolated PSD
        fft_data = np.fft.rfft(filtered)
        freq_vals = np.fft.rfftfreq(len(filtered), d=dt)
        interp_psd = np.interp(freq_vals, freqs, smooth_psd)
        white_fft = fft_data / np.sqrt(interp_psd)
        whitened = np.fft.irfft(white_fft, n=len(filtered))
        
        # Apply median filtering to remove short-duration artifacts
        whitened = signal.medfilt(whitened, kernel_size=5)
        return whitened

    # Process both channels with adaptive filtering and whitening
    white_h1 = adaptive_filter_and_whiten(strain_h1)
    white_l1 = adaptive_filter_and_whiten(strain_l1)

    # Dynamic STFT segmentation: adjust window size based on data length
    seg_length = len(white_h1)
    nperseg = int(np.clip(seg_length // 100, 256, 1024))
    noverlap = int(nperseg * 0.75)
    stft_window = signal.windows.hann(nperseg, sym=False)
    
    # Compute STFT for both channels
    f_h1, t_spec, Zxx_h1 = signal.stft(white_h1, fs=fs, window=stft_window,
                                       nperseg=nperseg, noverlap=noverlap, detrend=False)
    f_l1, _, Zxx_l1 = signal.stft(white_l1, fs=fs, window=stft_window,
                                  nperseg=nperseg, noverlap=noverlap, detrend=False)
    
    # Calculate power spectrograms for each detector
    power_h1 = np.abs(Zxx_h1)**2
    power_l1 = np.abs(Zxx_l1)**2
    
    # Local noise estimation: normalize each frequency bin by its median power
    med_power_h1 = np.median(power_h1, axis=1, keepdims=True) + eps
    med_power_l1 = np.median(power_l1, axis=1, keepdims=True) + eps
    snr_h1 = power_h1 / med_power_h1
    snr_l1 = power_l1 / med_power_l1

    # Adaptive SNR-based fusion: emphasize more significant contributions from each detector
    weight_h1 = (snr_h1 / (snr_h1 + snr_l1 + eps)) ** 1.2
    weight_l1 = (snr_l1 / (snr_h1 + snr_l1 + eps)) ** 1.2
    fused_power = power_h1 * weight_h1 + power_l1 * weight_l1

    # Inter-detector coherence: estimate phase consistency between channels
    cross_spec = Zxx_h1 * np.conj(Zxx_l1)
    norm_factor = (np.abs(Zxx_h1) * np.abs(Zxx_l1)) + eps
    coherence = np.clip(np.abs(cross_spec) / norm_factor, 0, 1)
    # Compute a weighted coherence measure along frequency, weighted by fused power magnitude.
    weight_power = np.maximum(np.median(fused_power, axis=1, keepdims=True), eps)
    coherence_time = np.average(coherence, axis=0, weights=weight_power.flatten())
    
    # Construct an enhanced time-frequency metric that incorporates both power and coherence
    base_metric = np.median(fused_power, axis=0)
    tf_metric = base_metric * (0.5 + 0.5 * (coherence_time ** 1.3))
    
    # Map STFT time bins to absolute GPS times (assumes times[0] as reference)
    metric_times = times[0] + t_spec
    
    # Iterative threshold tuning based on robust median and MAD estimation of tf_metric.
    med_tf = np.median(tf_metric)
    mad_tf = np.median(np.abs(tf_metric - med_tf)) + eps
    robust_std = 1.4826 * mad_tf
    k_factor = 3.5
    dynamic_threshold = med_tf + k_factor * robust_std

    # Iteratively refine threshold by excluding potential signal peaks to better estimate background noise.
    for _ in range(3):
        candidate_peaks, _ = signal.find_peaks(tf_metric, height=dynamic_threshold, prominence=robust_std, distance=3)
        if candidate_peaks.size > 0 and candidate_peaks.size < tf_metric.size:
            noise_data = np.delete(tf_metric, candidate_peaks)
            if noise_data.size > 0:
                med_tf = np.median(noise_data)
                mad_tf = np.median(np.abs(noise_data - med_tf)) + eps
                robust_std = 1.4826 * mad_tf
                dynamic_threshold = med_tf + k_factor * robust_std
            else:
                break
        else:
            break

    # Determine appropriate minimum separation (in time bins) for peak detection.
    dt_spec = t_spec[1] - t_spec[0] if t_spec.size > 1 else dt
    min_distance = max(1, int(0.5 / dt_spec))
    
    # Detect candidate peaks using adaptive threshold and prominence parameters.
    peaks, _ = signal.find_peaks(tf_metric, height=dynamic_threshold, prominence=robust_std, distance=min_distance)
    if peaks.size == 0:
        # Relax threshold if no peaks are found in the first pass.
        dynamic_threshold = med_tf + (k_factor * robust_std * 0.8)
        peaks, _ = signal.find_peaks(tf_metric, height=dynamic_threshold, prominence=robust_std, distance=min_distance)
    
    # Refine detected peaks by evaluating multi-scale local widths to estimate timing uncertainty.
    refined_peak_times = []
    refined_peak_heights = []
    refined_peak_uncertainties = []
    for peak in peaks:
        local_widths = []
        # Evaluate the width for different local scales to average out uncertainties.
        for win_scale in [5, 7, 9]:
            win_start = max(peak - win_scale, 0)
            win_end = min(peak + win_scale + 1, len(tf_metric))
            sub_segment = tf_metric[win_start:win_end]
            # Use peak_widths from the entire segment for consistency.
            widths, _, _, _ = signal.peak_widths(tf_metric, [peak], rel_height=0.5)
            local_widths.append(widths[0] * dt_spec)
        width_sec = np.mean(local_widths)
        # Compute extra uncertainty inversely related to the relative peak prominence
        peak_val = tf_metric[peak]
        local_bg = np.median(tf_metric[max(peak-5, 0):min(peak+6, len(tf_metric))])
        norm_peak = (peak_val - local_bg) / (np.max(tf_metric) - local_bg + eps)
        extra_uncert = 8.0 / (norm_peak + eps)
        total_uncert = np.clip(width_sec + extra_uncert, 1.0, 12.0)

        refined_peak_times.append(metric_times[peak])
        refined_peak_heights.append(peak_val)
        refined_peak_uncertainties.append(total_uncert)
    
    peak_times = np.array(refined_peak_times)
    peak_heights = np.array(refined_peak_heights)
    peak_deltat = np.array(refined_peak_uncertainties)
    
    return peak_times, peak_heights, peak_deltat
