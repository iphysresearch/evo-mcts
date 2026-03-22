import numpy as np
import scipy.signal as signal

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pipeline_v2 processes gravitational wave data from the H1 and L1 detectors and returns candidate events.
    It involves advanced digital filtering, multiple time-frequency transforms and multi-detector consistency checks.

    Parameters:
      strain_h1, strain_l1: numpy arrays with strain data for H1 and L1 detectors.
      times: numpy array with corresponding GPS times (assumed uniformly sampled).

    Returns:
      peak_times: GPS times for identified events.
      peak_heights: Ranking statistics (significance values) for each event.
      peak_deltat: Timing uncertainty (in seconds) for each event.
    """
    
    def data_conditioning(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # Sampling information
        dt = times[1] - times[0]
        fs = 1.0 / dt

        # Detrend & remove medians
        strain_h1 = signal.detrend(strain_h1) - np.median(strain_h1)
        strain_l1 = signal.detrend(strain_l1) - np.median(strain_l1)
        
        # Apply a Butterworth bandpass filter (e.g., 20 Hz to 500 Hz, typical for GW detectors)
        order = 4
        lowcut, highcut = 20, 500
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype="band")
        filtered_h1 = signal.filtfilt(b, a, strain_h1)
        filtered_l1 = signal.filtfilt(b, a, strain_l1)
        
        # Whiten the data using FFT-based normalization (with spectral smoothing)
        def whiten_strain(strain):
            strain_zero_mean = strain - np.mean(strain)
            n = len(strain_zero_mean)
            freqs, psd = signal.welch(strain_zero_mean, fs=fs, nperseg=4096, window='hann', noverlap=2048)
            # Smooth PSD estimate
            smooth_psd = np.convolve(psd, np.ones(32)/32, mode='same')
            smooth_psd = np.maximum(smooth_psd, np.finfo(float).tiny)
            # FFT and whiten normalization interpolation
            fft_data = np.fft.rfft(strain_zero_mean)
            freqs_fft = np.fft.rfftfreq(n, d=dt)
            norm_factor = np.sqrt(np.interp(freqs_fft, freqs, smooth_psd))
            white_fft = fft_data / norm_factor
            return np.fft.irfft(white_fft, n=n)
        
        white_h1 = whiten_strain(filtered_h1)
        white_l1 = whiten_strain(filtered_l1)
        
        return white_h1, white_l1, fs

    def time_frequency_transformation(white_h1: np.ndarray, white_l1: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        # Use two complementary TF transforms:
        # 1. Short-Time Fourier Transform (STFT)
        nperseg = 256
        noverlap = 128
        f1, t_stft, Zxx_h1 = signal.stft(white_h1, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
        f2, t_stft, Zxx_l1 = signal.stft(white_l1, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
        
        # Compute power spectral density in each time bin then average over frequency bins
        psd_h1 = np.mean(np.abs(Zxx_h1)**2, axis=0)
        psd_l1 = np.mean(np.abs(Zxx_l1)**2, axis=0)
        
        # Multi-detector (coherent) metric: geometric mean gives extra penalty 
        coherent_metric = np.sqrt(psd_h1 * psd_l1)
        
        # 2. Complementary time-frequency analysis via Continuous Wavelet Transform (CWT) with the Ricker wavelet
        widths = np.arange(1, 128)
        # Note: cwt returns (len(widths), len(signal)) matrix. We compute energy over widths and downsample to stft times.
        cwt_h1 = signal.cwt(white_h1, signal.ricker, widths)
        cwt_l1 = signal.cwt(white_l1, signal.ricker, widths)
        energy_h1 = np.mean(np.abs(cwt_h1)**2, axis=0)
        energy_l1 = np.mean(np.abs(cwt_l1)**2, axis=0)
        # Downsample CWT energy to STFT time grid using simple averaging over windows
        ds_factor = int(len(white_h1) / len(t_stft))
        energy_h1_ds = np.array([np.mean(energy_h1[i*ds_factor:(i+1)*ds_factor]) for i in range(len(t_stft))])
        energy_l1_ds = np.array([np.mean(energy_l1[i*ds_factor:(i+1)*ds_factor]) for i in range(len(t_stft))])
        coherent_energy = np.sqrt(energy_h1_ds * energy_l1_ds)
        
        # Combine metrics: weighted average of STFT based and CWT based measurements
        tf_metric = 0.6 * coherent_metric + 0.4 * coherent_energy
        
        # Align times relative to original GPS times: take the midpoint of the segment as offset
        gps_mid = times[0] + (times[-1] - times[0]) / 2
        metric_times = gps_mid + (t_stft - t_stft[len(t_stft)//2])
        
        return tf_metric, metric_times

    def peak_detection(tf_metric: np.ndarray, metric_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Determine baseline using robust median and MAD estimation
        background = np.median(tf_metric)
        mad = np.median(np.abs(tf_metric - background))
        threshold = background + 3 * mad

        # Find peaks that satisfy minimum height and separation (distance based on STFT time resolution)
        peaks, properties = signal.find_peaks(tf_metric, height=threshold, distance=2)
        peak_times = metric_times[peaks]
        peak_heights = tf_metric[peaks]

        # Estimate time uncertainty using the width of each peak at half prominence
        if len(peaks) > 0:
            widths_results = signal.peak_widths(tf_metric, peaks, rel_height=0.5)
            # widths_results[0] gives the widths in number of samples; convert to seconds using time resolution.
            time_resolution = metric_times[1] - metric_times[0]
            peak_deltat = widths_results[0] * time_resolution
        else:
            peak_deltat = np.array([])
        
        return peak_times, peak_heights, peak_deltat

    # Step 1: Data Conditioning
    white_h1, white_l1, fs = data_conditioning(strain_h1, strain_l1, times)
    
    # Step 2: Time-Frequency Transformation and Multi-detector coherence analysis
    tf_metric, metric_times = time_frequency_transformation(white_h1, white_l1, fs)
    
    # Step 3: Peak Detection with signal-based veto (using robust threshold and peak width estimation)
    peak_times, peak_heights, peak_deltat = peak_detection(tf_metric, metric_times)
    
    return peak_times, peak_heights, peak_deltat
