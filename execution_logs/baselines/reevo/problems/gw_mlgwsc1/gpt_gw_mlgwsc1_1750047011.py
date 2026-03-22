import numpy as np
from scipy import signal
from scipy.signal import medfilt, savgol_filter

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline version 2 for gravitational wave signal detection.
    Implements adaptive multi-scale windowing with robust noise estimation,
    cross-detector validation, and additional wavelet-domain fusion.
    
    Inputs:
      strain_h1: 1D numpy array containing H1 detector strain data.
      strain_l1: 1D numpy array containing L1 detector strain data.
      times: 1D numpy array of GPS time stamps corresponding to the strain data.
      
    Returns:
      A tuple (peak_times, peak_heights, peak_deltat) where:
        peak_times: GPS times of identified candidate events.
        peak_heights: Ranking/significance values for each candidate.
        peak_deltat: Estimated timing uncertainty (window) for each candidate.
    """
    dt = times[1] - times[0]
    fs = 1.0 / dt

    # -------------------------------------
    # 1. Adaptive Data Conditioning and Whitening
    # -------------------------------------
    def adaptive_data_conditioning(data: np.ndarray) -> np.ndarray:
        # Robust detrending: combine median filtering and Savitzky-Golay filtering.
        ker_width = int(0.2 * fs)
        if ker_width % 2 == 0:
            ker_width += 1
        # Use median filter to remove outliers/trends
        trend_med = medfilt(data, kernel_size=min(ker_width, len(data) // 2 | 1))
        # Use Savitzky-Golay filter for smooth trend estimation; window must be odd.
        win_length = min(101, len(data)) if len(data) % 2 else min(101, len(data)-1)
        trend_sg = savgol_filter(data, window_length=win_length, polyorder=3)
        # Combine the two estimates
        trend = 0.5 * (trend_med + trend_sg)
        detrended = data - trend
        
        # Remove any residual linear trend (zero-phase detrending)
        t_idx = np.arange(len(detrended))
        poly_coeff = np.polyfit(t_idx, detrended, 1)
        detrended -= np.polyval(poly_coeff, t_idx)
        
        # Adaptive whitening via two-stage PSD estimation (using Welch)
        data_zero = detrended - np.mean(detrended)
        nperseg1 = int(min(4096, len(data_zero) // 8))
        nperseg1 = max(nperseg1, 256)
        nperseg2 = max(256, nperseg1 // 2)
        
        freqs1, psd1 = signal.welch(data_zero, fs=fs, nperseg=nperseg1, window='hann', noverlap=nperseg1 // 2)
        freqs2, psd2 = signal.welch(data_zero, fs=fs, nperseg=nperseg2, window='hann', noverlap=nperseg2 // 2)
        psd2_interp = np.interp(freqs1, freqs2, psd2)
        combined_psd = np.sqrt(psd1 * psd2_interp)
        # Smooth PSD estimate with two-stage moving averages
        smooth_psd = np.convolve(combined_psd, np.ones(32)/32, mode='same')
        smooth_psd = np.convolve(smooth_psd, np.ones(16)/16, mode='same')
        smooth_psd = np.maximum(smooth_psd, np.finfo(float).tiny)
        
        # Perform whitening in frequency domain
        data_fft = np.fft.rfft(data_zero)
        freqs_fft = np.fft.rfftfreq(len(data_zero), d=dt)
        psd_interp = np.interp(freqs_fft, freqs1, smooth_psd)
        white_fft = data_fft / np.sqrt(psd_interp)
        white_data = np.fft.irfft(white_fft, n=len(data_zero))
        return white_data

    whitened_h1 = adaptive_data_conditioning(strain_h1)
    whitened_l1 = adaptive_data_conditioning(strain_l1)
    
    # -------------------------------------
    # 2. Multi-scale Time-Frequency and Wavelet Fusion with Cross-Detector Consistency
    # -------------------------------------
    def compute_fused_metric(h1_data: np.ndarray, l1_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fs_local = fs

        # --- STFT Analysis at two scales ---
        # Scale 1: High time resolution
        nperseg1, noverlap1 = 128, 96
        f1, t1, Sxx_h1_sc1 = signal.spectrogram(h1_data, fs=fs_local, nperseg=nperseg1, noverlap=noverlap1,
                                                window='hann', mode='magnitude', detrend=False)
        _, _, Sxx_l1_sc1 = signal.spectrogram(l1_data, fs=fs_local, nperseg=nperseg1, noverlap=noverlap1,
                                              window='hann', mode='magnitude', detrend=False)

        # Scale 2: Better frequency resolution
        nperseg2, noverlap2 = 512, 384
        f2, t2, Sxx_h1_sc2 = signal.spectrogram(h1_data, fs=fs_local, nperseg=nperseg2, noverlap=noverlap2,
                                                window='hann', mode='magnitude', detrend=False)
        _, _, Sxx_l1_sc2 = signal.spectrogram(l1_data, fs=fs_local, nperseg=nperseg2, noverlap=noverlap2,
                                              window='hann', mode='magnitude', detrend=False)

        # Coherent fusion by geometric mean per scale
        Sxx_coh_sc1 = np.sqrt(Sxx_h1_sc1 * Sxx_l1_sc1)
        Sxx_coh_sc2 = np.sqrt(Sxx_h1_sc2 * Sxx_l1_sc2)
        metric_sc1 = np.mean(Sxx_coh_sc1, axis=0)
        metric_sc2 = np.mean(Sxx_coh_sc2, axis=0)
        # Interpolate scale 2 metric to scale 1 time base
        metric_sc2_interp = np.interp(t1, t2, metric_sc2)
        stft_metric = np.sqrt(metric_sc1 * metric_sc2_interp)
        
        # --- Continuous Wavelet Transform (CWT) Analysis ---
        # Use Ricker (Mexican hat) wavelet with a range of widths
        widths = np.arange(1, 31)
        wave_h1 = signal.cwt(h1_data, signal.ricker, widths)
        wave_l1 = signal.cwt(l1_data, signal.ricker, widths)
        # Collapse the scale dimension with a median (robust measure)
        wave_metric_h1 = np.median(np.abs(wave_h1), axis=0)
        wave_metric_l1 = np.median(np.abs(wave_l1), axis=0)
        wavelet_metric = np.sqrt(wave_metric_h1 * wave_metric_l1)
        # Downsample the wavelet metric to match STFT time base.
        wavelet_metric_interp = np.interp(t1, times, wavelet_metric)
        
        # Final fusion: combine STFT and wavelet metrics using geometric mean.
        fused_metric = np.sqrt(stft_metric * wavelet_metric_interp)
        
        # Map spectrogram times to GPS times (centered on segment mid time)
        segment_mid = times[0] + (times[-1] - times[0]) / 2.0
        metric_times = segment_mid + (t1 - t1[len(t1)//2])
        return fused_metric, metric_times

    tf_metric, metric_times = compute_fused_metric(whitened_h1, whitened_l1)

    # -------------------------------------
    # 3. Candidate Selection with Dynamic Thresholding and Cross-Detector Validation
    # -------------------------------------
    def select_candidates(metric: np.ndarray, metric_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute robust statistics: median and MAD for dynamic thresholding.
        med_val = np.median(metric)
        mad = np.median(np.abs(metric - med_val))
        if mad <= 0:
            mad = np.std(metric) if np.std(metric) > 0 else 1.0
            
        threshold = med_val + 3.5 * mad
        threshold = max(threshold, med_val * 1.2)
        
        # Identify peaks in the fused metric that exceed the dynamic threshold.
        peaks, props = signal.find_peaks(metric, height=threshold, distance=2, prominence=mad)
        candidate_times = metric_times[peaks]
        candidate_heights = metric[peaks]
        
        # Cross-detector consistency check: For each candidate, verify that the whitened signals 
        # in a short window around the candidate time are correlated.
        accepted_times = []
        accepted_heights = []
        half_win = int(0.1 * fs)  # 0.1 sec window on each side
            
        for cand_time, cand_height in zip(candidate_times, candidate_heights):
            # Map candidate GPS time to index in original data using first sample time.
            idx = int((cand_time - times[0]) * fs)
            start_idx = max(0, idx - half_win)
            end_idx = min(len(whitened_h1), idx + half_win)
            if end_idx - start_idx < 10:
                continue  # window too short for reliable correlation
            segment_h1 = whitened_h1[start_idx:end_idx]
            segment_l1 = whitened_l1[start_idx:end_idx]
            # Compute Pearson correlation coefficient (if standard deviations non-zero)
            if np.std(segment_h1) > 0 and np.std(segment_l1) > 0:
                corr_coef = np.corrcoef(segment_h1, segment_l1)[0, 1]
            else:
                corr_coef = 0
            # Accept only if cross-correlation exceeds a minimal threshold (e.g., 0.3)
            if corr_coef > 0.3:
                accepted_times.append(cand_time)
                accepted_heights.append(cand_height)
                
        accepted_times = np.array(accepted_times)
        accepted_heights = np.array(accepted_heights)
        
        # Timing uncertainty: Use a fraction (0.5%) of the segment duration.
        uncertainty = (metric_times[-1] - metric_times[0]) * 0.005
        accepted_deltat = np.full_like(accepted_times, uncertainty)
        return accepted_times, accepted_heights, accepted_deltat

    peak_times, peak_heights, peak_deltat = select_candidates(tf_metric, metric_times)
    
    return peak_times, peak_heights, peak_deltat
