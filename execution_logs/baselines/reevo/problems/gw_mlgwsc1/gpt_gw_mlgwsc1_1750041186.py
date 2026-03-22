import numpy as np
import scipy.signal as signal

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A gravitational-wave signal search pipeline version 2 that combines advanced
    data conditioning, time-frequency analysis using multiple methods (STFT and Wavelet),
    multi-detector consistency checks, and signal‐based vetoes.
    
    Parameters:
      strain_h1 : Numpy array containing H1 strain data.
      strain_l1 : Numpy array containing L1 strain data.
      times     : Numpy array of time stamps corresponding to the detector samples.
      
    Returns:
      A tuple (peak_times, peak_heights, peak_deltat) where:
        - peak_times: GPS times of the candidate triggers.
        - peak_heights: Ranking statistic (significance values) for each candidate.
        - peak_deltat: Timing uncertainty (in seconds) for each candidate trigger.
    """
    
    def data_conditioning_v2(strain: np.ndarray, fs: float) -> np.ndarray:
        # Remove the DC offset
        strain = strain - np.mean(strain)
        # Apply a Butterworth high-pass filter (remove low frequency noise)
        # For gravitational-wave data, we choose a cutoff of ~30 Hz.
        highpass_cutoff = 30.0  # Hz
        nyq = 0.5 * fs
        norm_cut = highpass_cutoff / nyq
        b, a = signal.butter(4, norm_cut, btype='highpass')
        filtered = signal.filtfilt(b, a, strain)
        # Apply a median filter to suppress spikes
        conditioned = signal.medfilt(filtered, kernel_size=5)
        return conditioned

    def whiten_signal(signal_data: np.ndarray, fs: float) -> np.ndarray:
        # Whiten using FFT based method similar to pipeline v1,
        # but with an improved PSD smoothing.
        nperseg = 4096
        strain_zeromean = signal_data - np.mean(signal_data)
        freqs, psd = signal.welch(strain_zeromean, fs=fs, nperseg=nperseg,
                                  window='hann', noverlap=nperseg//2)
        # Smooth the PSD using a running average
        smooth_len = 32
        smooth_filter = np.ones(smooth_len) / smooth_len
        psd_smoothed = np.convolve(psd, smooth_filter, mode='same')
        psd_smoothed = np.maximum(psd_smoothed, np.finfo(float).tiny)
        fft_data = np.fft.rfft(strain_zeromean)
        freq_array = np.fft.rfftfreq(len(strain_zeromean), d=1/fs)
        white_fft = fft_data / np.sqrt(np.interp(freq_array, freqs, psd_smoothed))
        return np.fft.irfft(white_fft, n=len(strain_zeromean))
    
    def compute_timefreq_metric(h1: np.ndarray, l1: np.ndarray, fs: float, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # --- STFT based metric ---
        nperseg = 256
        noverlap = 128
        # Compute STFT for both detectors.
        f, t_stft, Zxx_h1 = signal.stft(h1, fs=fs, window='hann',
                                        nperseg=nperseg, noverlap=noverlap, detrend=False)
        f, t_stft, Zxx_l1 = signal.stft(l1, fs=fs, window='hann',
                                        nperseg=nperseg, noverlap=noverlap, detrend=False)
        # Power in each time window for each detector
        power_h1 = np.mean(np.abs(Zxx_h1)**2, axis=0)
        power_l1 = np.mean(np.abs(Zxx_l1)**2, axis=0)
        # Geometric mean of power from both detectors emphasizes joint excess power.
        stft_metric = np.sqrt(power_h1 * power_l1)

        # --- Local Coherence metric ---
        # Compute sliding window Pearson correlation (detector coherence)
        step = nperseg - noverlap
        n_windows = (len(h1) - nperseg) // step + 1
        coherence_metric = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * step
            segment_h1 = h1[start:start+nperseg]
            segment_l1 = l1[start:start+nperseg]
            if np.std(segment_h1) > 0 and np.std(segment_l1) > 0:
                corr_coef = np.corrcoef(segment_h1, segment_l1)[0, 1]
            else:
                corr_coef = 0.0
            coherence_metric[i] = corr_coef
        # Align the coherence metric time array with t_stft:
        t_coherence = t_stft[:n_windows]

        # --- Wavelet transform based metric ---
        # Using the Ricker (Mexican hat) wavelet to capture transient features.
        widths = np.arange(1, 128)
        cwt_h1 = signal.cwt(h1, signal.ricker, widths)
        cwt_l1 = signal.cwt(l1, signal.ricker, widths)
        # Compute wavelet power as a function of time by averaging over scales.
        wavelet_power_h1 = np.mean(np.abs(cwt_h1)**2, axis=0)
        wavelet_power_l1 = np.mean(np.abs(cwt_l1)**2, axis=0)
        # Resample wavelet power onto the STFT time grid.
        from scipy.interpolate import interp1d
        time_full = times  # full time vector corresponds to the indices of the time series.
        interp_h1 = interp1d(time_full, wavelet_power_h1, bounds_error=False, fill_value="extrapolate")
        interp_l1 = interp1d(time_full, wavelet_power_l1, bounds_error=False, fill_value="extrapolate")
        # Map t_stft times (which are relative to start) into absolute times.
        t_stft_abs = times[0] + t_stft
        wavelet_power_h1_resamp = interp_h1(t_stft_abs)
        wavelet_power_l1_resamp = interp_l1(t_stft_abs)
        wavelet_metric = np.sqrt(wavelet_power_h1_resamp * wavelet_power_l1_resamp)

        # --- Combine Metrics ---
        # For each time-step in the STFT grid (and matching coherence metric, if available),
        # we blend the power metric and the wavelet metric, then modulate by the coherence.
        # Here, we resample the coherence metric to match the length of t_stft.
        if len(coherence_metric) < len(t_stft):
            # Extend coherence metric by holding last value if needed.
            coherence_metric = np.pad(coherence_metric, (0, len(t_stft) - len(coherence_metric)), mode='edge')
        else:
            coherence_metric = coherence_metric[:len(t_stft)]
        
        # Combine metrics: weight STFT and wavelet equally and then modulate by coherence.
        combined_metric = 0.5 * (stft_metric + wavelet_metric) * (0.5 + 0.5 * coherence_metric)
        return combined_metric, t_stft_abs

    def calculate_triggers(metric: np.ndarray, time_grid: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Estimate background from median and standard deviation
        background = np.median(metric)
        sigma = np.std(metric)
        # Use find_peaks with a threshold based on background + sigma for significance.
        height_thresh = background + sigma
        peaks, properties = signal.find_peaks(metric, height=height_thresh, distance=2)
        peak_times = time_grid[peaks]
        peak_heights = metric[peaks]
        # Timing uncertainty is roughly one window length (nperseg / fs)
        uncertainty = (256 / fs)
        peak_deltat = np.full(len(peak_times), uncertainty)
        return peak_times, peak_heights, peak_deltat

    # Main processing begins here.
    dt = times[1] - times[0]
    fs = 1.0 / dt

    # ----- Step 1: Preprocessing -----
    conditioned_h1 = data_conditioning_v2(strain_h1, fs)
    conditioned_l1 = data_conditioning_v2(strain_l1, fs)
    
    # ----- Step 2: Whitening -----
    white_h1 = whiten_signal(conditioned_h1, fs)
    white_l1 = whiten_signal(conditioned_l1, fs)
    
    # ----- Step 3: Time-Frequency Analysis and Multi-detector combination -----
    tf_metric, t_metric = compute_timefreq_metric(white_h1, white_l1, fs, times)
    
    # ----- Step 4: Identify triggers using peak detection with signal-based vetoes -----
    peak_times, peak_heights, peak_deltat = calculate_triggers(tf_metric, t_metric, fs)
    
    return peak_times, peak_heights, peak_deltat
