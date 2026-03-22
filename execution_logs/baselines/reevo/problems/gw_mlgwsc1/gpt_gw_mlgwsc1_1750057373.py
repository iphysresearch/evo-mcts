import numpy as np
import pywt
from scipy.signal import wiener, medfilt, spectrogram, find_peaks, welch, tukey
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import DBSCAN

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gravitational Wave Signal Detection Pipeline (v2 - Modular Adaptive Enhancement)
    
    This version further refines modular adaptivity by:
      - Employing enhanced multi-scale noise filtering via wavelet packet decomposition,
        Wiener filtering, and adaptive spectral whitening.
      - Using dynamic windowing in time-frequency analysis with variance-driven parameter tuning.
      - Integrating cross-detector coherence metrics measured via sliding-window correlation.
      - Performing iterative, uncertainty-aware peak detection and DBSCAN-based clustering.
    
    Args:
        strain_h1 (np.ndarray): 1D raw strain data from H1 detector.
        strain_l1 (np.ndarray): 1D raw strain data from L1 detector.
        times (np.ndarray): 1D array of GPS time stamps corresponding to the strain samples.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - peak_times: GPS times of candidate gravitational wave events.
            - peak_heights: Ranking statistic (significance) for each event.
            - peak_deltat: Estimated timing uncertainty (seconds) for each event.
    """
    
    # Estimate sampling rate from time differences.
    dt = times[1] - times[0]
    fs = 1.0 / dt

    # ---------------- Stage 1: Enhanced Multi-Scale Noise Filtering ----------------
    def adaptive_conditioning(data: np.ndarray, fs: float) -> np.ndarray:
        # Remove DC offset.
        data = data - np.mean(data)
        
        # Wavelet packet decomposition based denoising.
        max_level = 5
        wp = pywt.WaveletPacket(data, 'sym5', maxlevel=max_level, mode='symmetric')
        new_wp = pywt.WaveletPacket(data=None, wavelet='sym5', maxlevel=max_level, mode='symmetric')
        thresh_factor = 1.3
        for node in wp.get_level(max_level, order='freq'):
            coeff = node.data
            sigma = np.median(np.abs(coeff - np.median(coeff))) / 0.6745
            sigma = sigma if sigma > 0 else 1.0
            threshold = thresh_factor * sigma * np.sqrt(2 * np.log(len(coeff)))
            new_wp[node.path] = pywt.threshold(coeff, threshold, mode='soft')
        denoised = new_wp.reconstruct(update=False)[:len(data)]
        
        # Wiener filtering for smoothing residual noise.
        smooth_data = wiener(denoised, mysize=31)
        
        # Adaptive spectral whitening using Welch's method with Tukey window.
        nseg = int(np.clip(len(smooth_data) // 40, 1024, len(smooth_data) // 3))
        win = tukey(nseg, alpha=0.4)
        freqs, psd_est = welch(smooth_data, fs=fs, nperseg=nseg, noverlap=nseg // 2, window=win)
        psd_med = medfilt(psd_est, kernel_size=31)
        psd_med = np.maximum(psd_med, np.finfo(float).eps)
        sig_fft = np.fft.rfft(smooth_data)
        freqs_fft = np.fft.rfftfreq(len(smooth_data), d=1/fs)
        psd_interp = np.interp(freqs_fft, freqs, psd_med)
        white_fft = sig_fft / np.sqrt(psd_interp)
        whitened = np.fft.irfft(white_fft, n=len(smooth_data))
        
        # Final smoothing with median and uniform filters.
        conditioned = medfilt(whitened, kernel_size=7)
        conditioned = uniform_filter1d(conditioned, size=7)
        return conditioned

    # ---------------- Stage 2: Time-Frequency Metric with Coherence ----------------
    def compute_time_frequency_metric(h1: np.ndarray, l1: np.ndarray, times: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Dynamic windowing based on local signal variance.
        base_window = 512
        avg_var = 0.5 * (np.var(h1) + np.var(l1))
        scale_factor = np.clip(1 + (1.0 / (avg_var + 1e-6)) * 0.3, 1, 2.5)
        nperseg = int(np.clip(base_window * scale_factor, 256, 1024))
        noverlap = nperseg // 2
        window_func = tukey(nperseg, alpha=0.4)
        
        # Compute spectrograms for both detectors.
        f, t_spec, Sxx_h1 = spectrogram(h1, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                        window=window_func, mode='magnitude', detrend=False)
        _, _, Sxx_l1 = spectrogram(l1, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                   window=window_func, mode='magnitude', detrend=False)
        
        # Geometric mean to fuse channel information.
        Sxx_combined = np.sqrt(Sxx_h1 * Sxx_l1 + 1e-12)
        
        # Frequency weighting to emphasize 30-300 Hz gravitational wave band.
        freq_weight = np.ones_like(f)
        band_idx = (f >= 30) & (f <= 300)
        freq_weight[band_idx] = np.exp(-0.5 * ((f[band_idx] - 150) / 40) ** 2)
        weighted_Sxx = Sxx_combined * freq_weight[:, None]
        tf_metric = np.mean(weighted_Sxx, axis=0)
        
        # Compute local coherence via sliding-window correlation.
        coherence = np.zeros_like(t_spec)
        half_win = nperseg // 2
        total_samples = len(h1)
        for i, t_val in enumerate(t_spec):
            center = int(t_val * fs)
            start = max(0, center - half_win)
            end = min(total_samples, center + half_win)
            if (end - start) < 10:
                coherence[i] = 0.0
            else:
                seg_h1 = h1[start:end]
                seg_l1 = l1[start:end]
                std1, std2 = np.std(seg_h1), np.std(seg_l1)
                if std1 > 0 and std2 > 0:
                    corr = np.corrcoef(seg_h1, seg_l1)[0, 1]
                    coherence[i] = np.clip(corr, 0.0, 1.0)
                else:
                    coherence[i] = 0.0
        
        # Normalize coherence robustly and merge with tf_metric.
        median_coh = np.median(coherence) + 1e-12
        norm_coherence = coherence / median_coh
        tf_metric_weighted = tf_metric * norm_coherence
        
        # Map spectrogram time bins to absolute GPS times.
        metric_times = times[0] + t_spec
        return tf_metric_weighted, metric_times, norm_coherence

    # ---------------- Stage 3: Iterative, Uncertainty-Aware Peak Detection ----------------
    def iterative_peak_detection(metric: np.ndarray, metric_times: np.ndarray, coherence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Estimate background noise and dispersion.
        background = np.median(metric)
        mad = np.median(np.abs(metric - background)) * 1.4826
        threshold_mult = 3.5
        threshold = background + threshold_mult * mad
        
        dt_metric = np.mean(np.diff(metric_times)) if len(metric_times) > 1 else 1.0
        min_distance = max(1, int(0.5 / dt_metric))
        
        iteration = 0
        max_iter = 6
        peaks, properties = find_peaks(metric, height=threshold, distance=min_distance, prominence=mad)
        # Iteratively relax threshold if necessary.
        while (peaks.size == 0 or (properties.get('prominences') is not None and np.max(properties.get('prominences', [0])) < mad)) and iteration < max_iter:
            threshold = background + (threshold_mult - 0.5*(iteration+1)) * mad
            peaks, properties = find_peaks(metric, height=threshold, distance=min_distance, prominence=mad)
            iteration += 1
        
        if peaks.size == 0:
            return np.array([]), np.array([]), np.array([])
        
        peak_times = metric_times[peaks]
        peak_heights = metric[peaks]
        
        # Estimate timing uncertainty using local variance and inverse coherence weighting.
        epsilon = 1e-3
        local_var = np.array([
            np.var(metric[max(0, idx-3):min(len(metric), idx+4)])
            for idx in peaks
        ])
        peak_deltat = 5.0 * (1.0 / (coherence[peaks] + epsilon)) * (1 + local_var / (background + mad + epsilon))
        
        return peak_times, peak_heights, peak_deltat

    # ---------------- Stage 4: Uncertainty-Aware Clustering of Triggers ----------------
    def cluster_triggers(peak_times: np.ndarray, peak_heights: np.ndarray, peak_deltat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if peak_times.size == 0:
            return peak_times, peak_heights, peak_deltat
        
        # Use DBSCAN on trigger times with a data-driven eps (median uncertainty scaled).
        X = peak_times.reshape(-1, 1)
        base_eps = np.median(peak_deltat) * 1.5
        db = DBSCAN(eps=base_eps, min_samples=1).fit(X)
        labels = db.labels_
        unique_labels = np.unique(labels)
        
        clustered_times = []
        clustered_heights = []
        clustered_deltat = []
        for label in unique_labels:
            idx = (labels == label)
            clustered_times.append(np.mean(peak_times[idx]))
            clustered_heights.append(np.mean(peak_heights[idx]))
            clustered_deltat.append(np.mean(peak_deltat[idx]))
        return np.array(clustered_times), np.array(clustered_heights), np.array(clustered_deltat)
    
    # ---------------- Pipeline Execution ----------------
    # Stage 1: Apply adaptive multi-scale conditioning to raw detector data.
    conditioned_h1 = adaptive_conditioning(strain_h1, fs)
    conditioned_l1 = adaptive_conditioning(strain_l1, fs)
    
    # Stage 2: Compute a time-frequency metric with enhanced cross-channel coherence.
    tf_metric, metric_times, norm_coh = compute_time_frequency_metric(conditioned_h1, conditioned_l1, times, fs)
    
    # Stage 3: Detect candidate peaks using iterative, uncertainty-aware thresholding.
    peak_times, peak_heights, peak_deltat = iterative_peak_detection(tf_metric, metric_times, norm_coh)
    
    # Stage 4: Cluster nearby triggers using DBSCAN with uncertainty-informed parameters.
    clustered_times, clustered_heights, clustered_deltat = cluster_triggers(peak_times, peak_heights, peak_deltat)
    
    # Returning raw detections; replace with clustered_* entries if clustering is desired.
    return peak_times, peak_heights, peak_deltat
