import numpy as np
import pywt
from scipy.signal import detrend, butter, filtfilt, medfilt, stft, find_peaks, peak_widths, welch
from scipy.ndimage import gaussian_filter1d

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pipeline_v2: An advanced gravitational wave detection pipeline that integrates adaptive 
    wavelet denoising, iterative whitening using local noise estimation, multi-scale time-frequency 
    coherence feedback, and dynamic trigger detection with local SNR tuning.
    
    Parameters:
      strain_h1 : np.ndarray
          Raw strain data from the H1 detector.
      strain_l1 : np.ndarray
          Raw strain data from the L1 detector.
      times     : np.ndarray
          Array of GPS timestamps corresponding to the data samples.
      
    Returns:
      tuple[np.ndarray, np.ndarray, np.ndarray]:
          peak_times   : GPS times of detected candidate gravitational wave signals.
          peak_heights : Ranking statistic (significance value) for each detected event.
          peak_deltat  : Estimated timing uncertainties (in seconds) for each event.
    """
    # Determine sampling parameters
    if len(times) > 1:
        dt = times[1] - times[0]
    else:
        dt = 1.0
    fs = 1.0 / dt
    total_samples = len(times)
    segment_start = times[0]

    # ----------------------------------------------------------------------#
    # Step 0: Adaptive Wavelet Denoising with Dynamic Level Selection
    def wavelet_denoise(data: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
        # Select maximum possible level and then choose a level about one third of that maximum, but at least 2
        max_level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
        level = max(2, max_level // 3)
        coeffs = pywt.wavedec(data, wavelet, mode="per", level=level)
        # Robust noise estimate from the finest scale coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        # Dynamic threshold factor based on overall data variability
        dynamic_factor = np.log10(1 + np.std(data)) + 1.0
        thresh = sigma * np.sqrt(2 * np.log(len(data))) * dynamic_factor
        new_coeffs = [coeffs[0]] + [pywt.threshold(c, value=thresh, mode='soft') for c in coeffs[1:]]
        rec = pywt.waverec(new_coeffs, wavelet, mode="per")
        return rec[:len(data)]

    # ----------------------------------------------------------------------#
    # Step 1: Iterative Whitening with Local Noise Feedback and Enhanced Filtering
    def iterative_whitening(strain: np.ndarray, fs: float, dt: float) -> np.ndarray:
        # Start with wavelet-based denoising to suppress large outliers
        denoised = wavelet_denoise(strain)
        proc = detrend(denoised - np.median(denoised))

        # High-pass filtering to remove low frequency drifts (4th order Butterworth, cutoff=20Hz)
        cutoff = 20.0
        nyquist = fs / 2.0
        b, a = butter(4, cutoff / nyquist, btype='highpass')
        filtered = filtfilt(b, a, proc)
        
        # Apply median filtering to suppress transient artifacts
        ksize = 5 if len(filtered) >= 5 else 3
        filtered = medfilt(filtered, kernel_size=ksize)
        
        # Noise floor estimation using Welch's method with adaptive segment size
        nperseg = int(min(4096, max(total_samples // 10, 256)))
        freqs, psd_initial = welch(filtered, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window="hann")
        # Adaptive smoothing on PSD estimate
        kernel_size = 33 if len(psd_initial) >= 33 else (len(psd_initial) // 2) * 2 + 1
        psd_med = medfilt(psd_initial, kernel_size=kernel_size)
        psd_smooth = gaussian_filter1d(psd_med, sigma=2)
        psd_smooth = medfilt(psd_smooth, kernel_size=kernel_size)
        psd_smooth = np.maximum(psd_smooth, np.finfo(float).tiny)
        
        # Whitening in the frequency domain using local PSD estimates
        fft_data = np.fft.rfft(filtered)
        freqs_fft = np.fft.rfftfreq(len(filtered), d=dt)
        psd_interp = np.interp(freqs_fft, freqs, psd_smooth)
        white_fft = fft_data / np.sqrt(psd_interp)
        whitened = np.fft.irfft(white_fft, n=len(filtered))
        return whitened

    white_h1 = iterative_whitening(strain_h1, fs, dt)
    white_l1 = iterative_whitening(strain_l1, fs, dt)

    # ----------------------------------------------------------------------#
    # Step 2: Multi-Scale Time-Frequency Analysis with Coherence Integration
    def multi_scale_metric(data1: np.ndarray, data2: np.ndarray, fs: float, seg_start: float) -> tuple[np.ndarray, np.ndarray]:
        eps = 1e-12
        # Fine-scale STFT for high resolution time-frequency energy mapping
        nperseg_fine = 256
        noverlap_fine = 128
        f_fine, t_fine, Zxx1_fine = stft(data1, fs=fs, nperseg=nperseg_fine, noverlap=noverlap_fine, window="hann")
        _, _, Zxx2_fine = stft(data2, fs=fs, nperseg=nperseg_fine, noverlap=noverlap_fine, window="hann")
        energy_fine = 0.5 * (np.sum(np.abs(Zxx1_fine)**2, axis=0) + np.sum(np.abs(Zxx2_fine)**2, axis=0))
        
        # Calculate cross-spectral coherence between detectors
        cross_spec = np.sum(Zxx1_fine * np.conj(Zxx2_fine), axis=0)
        norm1 = np.sqrt(np.sum(np.abs(Zxx1_fine)**2, axis=0))
        norm2 = np.sqrt(np.sum(np.abs(Zxx2_fine)**2, axis=0))
        coherence = np.clip(np.abs(cross_spec) / (norm1 * norm2 + eps), 0, 1)
        
        # Coarse-scale STFT for robust baseline energy estimation
        nperseg_coarse = 1024
        noverlap_coarse = 512
        _, t_coarse, Zxx1_coarse = stft(data1, fs=fs, nperseg=nperseg_coarse, noverlap=noverlap_coarse, window="hann")
        _, _, Zxx2_coarse = stft(data2, fs=fs, nperseg=nperseg_coarse, noverlap=noverlap_coarse, window="hann")
        energy_coarse = 0.5 * (np.sum(np.abs(Zxx1_coarse)**2, axis=0) + np.sum(np.abs(Zxx2_coarse)**2, axis=0))
        energy_coarse_interp = np.interp(t_fine, t_coarse, energy_coarse)
        
        # Fuse metrics: boost fine-scale energy in regions with high coherence
        metric_fine = energy_fine * (0.5 + 0.5 * np.sqrt(coherence))
        # Weight the combined metric across fine and coarse scales
        metric_combined = 0.5 * metric_fine + 0.5 * energy_coarse_interp
        
        # Map time axis from STFT to GPS time stamps
        metric_times = seg_start + t_fine
        return metric_combined, metric_times

    metric, metric_times = multi_scale_metric(white_h1, white_l1, fs, segment_start)

    # ----------------------------------------------------------------------#
    # Step 3: Adaptive Trigger Detection Using Dynamic Local Thresholding
    def adaptive_trigger_detection(metric: np.ndarray, time_axis: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Estimate global noise statistics using median and MAD
        global_med = np.median(metric)
        global_mad = np.median(np.abs(metric - global_med))
        base_threshold = global_med + 4.0 * global_mad
        
        # Compute local thresholds over a sliding window to account for time-varying noise
        window_size = max(20, len(metric) // 20)
        half_win = window_size // 2
        local_thresholds = np.empty_like(metric)
        for i in range(len(metric)):
            start = max(0, i - half_win)
            end = min(len(metric), i + half_win)
            local_seg = metric[start:end]
            local_med = np.median(local_seg)
            local_mad = np.median(np.abs(local_seg - local_med))
            # Boost factor increases sensitivity in low-variance segments
            boost = 1.0 + (np.std(local_seg) / (np.mean(local_seg) + 1e-6))
            local_thresholds[i] = local_med + 4.0 * local_mad * boost
        # Ensure that the dynamic thresholds do not fall below the global base threshold
        dynamic_threshold = np.maximum(local_thresholds, base_threshold)
        # Use the median dynamic threshold as the overall trigger threshold
        peak_threshold = np.median(dynamic_threshold)
        
        # Enforce a minimum trigger separation (approx. 1 second) based on sampling interval
        min_distance = int(1.0 / dt)
        peaks, properties = find_peaks(metric, height=peak_threshold, distance=min_distance, prominence=0.7 * global_mad)
        if peaks.size == 0:
            return np.array([]), np.array([]), np.array([])
        
        peak_times = time_axis[peaks]
        peak_heights = properties["peak_heights"]
        
        # Estimate timing uncertainty based on the peak widths and local noise characteristics
        widths, _, _, _ = peak_widths(metric, peaks, rel_height=0.5)
        widths_sec = widths * dt
        uncertainties = []
        local_window_bins = 5
        for i, pk in enumerate(peaks):
            local_start = max(pk - local_window_bins, 0)
            local_end = min(pk + local_window_bins + 1, len(metric))
            local_std = np.std(metric[local_start:local_end])
            uncert = 0.5 * widths_sec[i] + 5.0 * (1 + local_std)
            uncertainties.append(np.clip(uncert, 5.0, 20.0))
        peak_deltat = np.array(uncertainties)
        sort_idx = np.argsort(peak_times)
        return peak_times[sort_idx], peak_heights[sort_idx], peak_deltat[sort_idx]

    peak_times, peak_heights, peak_deltat = adaptive_trigger_detection(metric, metric_times, dt)
    return peak_times, peak_heights, peak_deltat
