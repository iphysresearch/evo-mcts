import numpy as np
import scipy.signal as signal

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline_v2 for gravitational wave signal detection.
    
    Steps:
      1. Data conditioning: Demean and bandpass filter using a Butterworth filter.
      2. Time-frequency analysis: Apply a continuous wavelet transform (CWT)
         with a Ricker (Mexican hat) wavelet and combine energies from two detectors.
      3. Multi-detector coherence and trigger extraction: Find peaks in the combined energy,
         apply multi-detector consistency, and estimate timing uncertainties via peak-widths.
    
    Input:
      strain_h1: np.ndarray with H1 detector strain data.
      strain_l1: np.ndarray with L1 detector strain data.
      times: np.ndarray of time stamps corresponding to the strain samples.
      
    Returns:
      A tuple containing:
        - peak_times: 1D np.ndarray of GPS times for candidate events.
        - peak_heights: 1D np.ndarray with significance (energy values) for each candidate.
        - peak_deltat: 1D np.ndarray with time-uncertainty (duration) estimates for the candidates.
    """
    
    def data_conditioning(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Demean the data
        data_h1 = strain_h1 - np.mean(strain_h1)
        data_l1 = strain_l1 - np.mean(strain_l1)
        
        # Sampling frequency from time array
        dt = times[1] - times[0]
        fs = 1.0 / dt
        
        # Define a Butterworth bandpass filter
        f_low = 30.0   # in Hz: low-frequency cutoff (example)
        f_high = 400.0 # in Hz: high-frequency cutoff (example)
        order = 4
        nyq = 0.5 * fs
        low = f_low / nyq
        high = f_high / nyq
        
        # Create bandpass filter coefficients
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_h1 = signal.filtfilt(b, a, data_h1)
        filtered_l1 = signal.filtfilt(b, a, data_l1)
        
        return filtered_h1, filtered_l1, times
    
    def compute_wavelet_energy(data_h1: np.ndarray, data_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # We'll use a Continuous Wavelet Transform (CWT) with the Ricker wavelet.
        # Define a range of scales. Adjust scale range based on the sampling rate.
        # Typically, scales can be chosen to cover the time-frequency range of interest.
        scales = np.arange(1, 128)
        
        # Compute CWT for H1 and L1 using the Ricker (Mexican hat) wavelet.
        # The cwt function returns an array of shape (len(scales), len(data)).
        cwt_h1 = signal.cwt(data_h1, signal.ricker, scales)
        cwt_l1 = signal.cwt(data_l1, signal.ricker, scales)
        
        # Energy: square of the absolute coefficients.
        energy_h1 = np.abs(cwt_h1)**2
        energy_l1 = np.abs(cwt_l1)**2
        
        # Collapse the scale dimension by taking the average energy over scales.
        metric_h1 = np.mean(energy_h1, axis=0)
        metric_l1 = np.mean(energy_l1, axis=0)
        
        # Coherent combination: simple average of the two detectors' metrics.
        tf_metric = 0.5 * (metric_h1 + metric_l1)
        
        # Normalize the time series metric for improved peak detection.
        tf_metric = (tf_metric - np.mean(tf_metric)) / (np.std(tf_metric) + np.finfo(float).eps)
        
        return tf_metric, times
    
    def calculate_triggers(tf_metric: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Estimate background noise level robustly (e.g., using median absolute deviation)
        background_level = np.median(tf_metric)
        # Set a threshold: require metric to be above a factor relative to background noise,
        # and also above the standard deviation.
        threshold = max(background_level + 1.0, 3.0)  # example threshold
        
        # Detect peaks with a minimum distance (in indices; adjust as needed)
        peaks, properties = signal.find_peaks(tf_metric, height=threshold, distance=fs*0.05, prominence=1.0)
        
        # If no peaks found, return empty arrays
        if len(peaks) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Determine peak times and heights
        peak_times = times[peaks]
        peak_heights = properties['peak_heights']
        
        # Estimate timing uncertainty from the widths at half prominence:
        widths_result = signal.peak_widths(tf_metric, peaks, rel_height=0.5)
        # Convert widths in sample counts to time uncertainty in seconds
        peak_deltat = widths_result[0] * (times[1] - times[0])
        
        return peak_times, peak_heights, peak_deltat
    
    # Step 1: Data Conditioning (Butterworth filtering, demeaning, etc.)
    filtered_h1, filtered_l1, conditioned_times = data_conditioning(strain_h1, strain_l1, times)
    
    # Step 2: Time-Frequency Analysis with Wavelet Transform
    tf_metric, metric_times = compute_wavelet_energy(filtered_h1, filtered_l1, conditioned_times)
    
    # Step 3: Trigger Identification (Multi-detector consistency and peak hunt)
    # Compute sampling rate for peak detection distance parameter (global fs)
    fs = 1.0 / (times[1] - times[0])
    peak_times, peak_heights, peak_deltat = calculate_triggers(tf_metric, metric_times)
    
    return peak_times, peak_heights, peak_deltat
