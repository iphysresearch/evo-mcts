import numpy as np
import scipy.signal as signal

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline v2 for gravitational wave detection with enhanced time-frequency analysis,
    coherence check across detectors, and signal-based vetoes.
    
    Steps:
      1. Data Conditioning: Bandpass filtering using a Butterworth filter.
      2. Time-Frequency Transformation: Continuous Wavelet Transform (using Ricker wavelet)
         to produce a metric series that highlights transient energy excess.
      3. Multi-detector Coherence Analysis: Compute a sliding-window Pearson correlation
         to enforce multi-detector consistency.
      4. Metric Combination & Peak Detection: Weight the wavelet-based metric with the
         local coherence and extract peaks using prominence and width criteria.
    
    Returns:
      peak_times: 1D numpy array of GPS times for identified events.
      peak_heights: 1D numpy array representing the significance metric of each event.
      peak_deltat: 1D numpy array with estimated timing uncertainty for each event.
    """
    
    def data_conditioning(strain: np.ndarray, fs: float, lowcut: float = 20.0, highcut: float = 500.0, order: int = 4) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter and remove mean.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        strain_demeaned = strain - np.mean(strain)
        return signal.filtfilt(b, a, strain_demeaned)
    
    def compute_wavelet_metric(data_h1: np.ndarray, data_l1: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Apply continuous wavelet transform (CWT) using the Ricker wavelet on both channels.
        The metric is defined as the average of the mean absolute wavelet coefficients
        (across scales) from both detectors.
        """
        widths = np.arange(1, 31)
        # Compute CWT for each channel
        coef_h1 = signal.cwt(data_h1, signal.ricker, widths)
        coef_l1 = signal.cwt(data_l1, signal.ricker, widths)
        # Average absolute coefficients over scales gives a time series metric.
        metric_h1 = np.mean(np.abs(coef_h1), axis=0)
        metric_l1 = np.mean(np.abs(coef_l1), axis=0)
        # Combine both detector metrics
        return (metric_h1 + metric_l1) / 2.0

    def compute_coherence(data_h1: np.ndarray, data_l1: np.ndarray, times: np.ndarray, window_size: int = 256, step: int = 64) -> np.ndarray:
        """
        Compute a sliding-window Pearson correlation coefficient between the two detectors.
        The output is interpolated to match the original time series length.
        """
        N = len(data_h1)
        win_centers = []
        correlations = []
        for start in range(0, N - window_size + 1, step):
            end = start + window_size
            seg_h1 = data_h1[start:end]
            seg_l1 = data_l1[start:end]
            if np.std(seg_h1) > 0 and np.std(seg_l1) > 0:
                corr = np.corrcoef(seg_h1, seg_l1)[0, 1]
            else:
                corr = 0.0
            win_center = start + window_size // 2
            win_centers.append(times[win_center])
            correlations.append(corr)
        # Interpolate the correlation back to the original time grid.
        win_centers = np.array(win_centers)
        correlations = np.array(correlations)
        coherence_interp = np.interp(times, win_centers, correlations)
        # Ensure coherence is in [0,1]
        coherence_interp = np.clip(coherence_interp, 0, 1)
        return coherence_interp

    def calculate_peaks(metric: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify significant peaks from the metric series. Use signal.find_peaks with 
        a dynamic threshold based on the median background. Then compute the width of
        each detected peak as an estimate for time uncertainty.
        """
        background = np.median(metric)
        # Set minimum height and prominence relative to background.
        min_height = background * 1.2
        min_prominence = background * 0.5
        peaks, properties = signal.find_peaks(metric, height=min_height, prominence=min_prominence, distance=10)
        peak_times = times[peaks]
        peak_heights = metric[peaks]
        # Estimate widths at half prominence as timing uncertainty, convert width to seconds.
        widths_result = signal.peak_widths(metric, peaks, rel_height=0.5)
        # widths_result[0] gives the widths in number of samples; convert using dt.
        dt = times[1] - times[0]
        peak_deltat = widths_result[0] * dt
        return peak_times, peak_heights, peak_deltat

    # Sampling frequency from the time array (assumed uniform).
    dt = times[1] - times[0]
    fs = 1.0 / dt

    # Step 1: Data Conditioning
    conditioned_h1 = data_conditioning(strain_h1, fs)
    conditioned_l1 = data_conditioning(strain_l1, fs)

    # Step 2: Time-Frequency Transformation using Wavelet Transform
    wf_metric = compute_wavelet_metric(conditioned_h1, conditioned_l1, times)

    # Step 3: Multi-detector Coherence Check
    coherence = compute_coherence(conditioned_h1, conditioned_l1, times, window_size=256, step=64)
    
    # Combine the wavelet metric with coherence: emphasize events where detectors agree.
    combined_metric = wf_metric * coherence

    # Optional: Apply a median filter to the combined metric for noise suppression.
    combined_metric = signal.medfilt(combined_metric, kernel_size=9)

    # Step 4: Peak Detection and Uncertainty Estimation
    peak_times, peak_heights, peak_deltat = calculate_peaks(combined_metric, times)
    
    return peak_times, peak_heights, peak_deltat
