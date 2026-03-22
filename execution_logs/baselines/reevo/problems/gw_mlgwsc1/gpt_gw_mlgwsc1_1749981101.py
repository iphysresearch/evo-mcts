import numpy as np
import scipy.signal as signal
import pywt

def pipeline_v2(strain_h1: np.ndarray, strain_l1: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Step 1: Data conditioning with bandpass filtering and whitening
    dt = times[1] - times[0]
    fs = 1.0 / dt

    def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def whiten_data(data: np.ndarray, fs: float, nperseg: int = 4096) -> np.ndarray:
        # Subtract mean and estimate PSD via Welch, then whiten in frequency domain
        data = data - np.mean(data)
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, window='hann', noverlap=nperseg//2)
        # Smooth the PSD to avoid spurious dips
        smooth_psd = np.convolve(psd, np.ones(32) / 32, mode='same')
        smooth_psd = np.maximum(smooth_psd, np.finfo(float).tiny)
        data_fft = np.fft.rfft(data)
        freqs_fft = np.fft.rfftfreq(len(data), d=1/fs)
        white_fft = data_fft / np.sqrt(np.interp(freqs_fft, freqs, smooth_psd))
        return np.fft.irfft(white_fft, n=len(data))

    # Apply bandpass filter (e.g., 20-500 Hz typical for gravitational waves)
    lowcut, highcut = 20.0, 500.0
    filtered_h1 = bandpass_filter(strain_h1, lowcut, highcut, fs)
    filtered_l1 = bandpass_filter(strain_l1, lowcut, highcut, fs)
    
    # Whiten the band-passed data to flatten the noise spectrum further
    white_h1 = whiten_data(filtered_h1, fs)
    white_l1 = whiten_data(filtered_l1, fs)

    # Step 2: Time-frequency transformation via Continuous Wavelet Transform (CWT)
    # Define a set of target frequencies and compute corresponding scales for the 'morl' (Morlet) wavelet.
    target_frequencies = np.linspace(lowcut, highcut, num=50)
    central_freq = pywt.central_frequency('morl')
    scales = central_freq / (target_frequencies * dt)

    def compute_wavelet_power(data: np.ndarray, scales: np.ndarray, dt: float) -> np.ndarray:
        coeffs, _ = pywt.cwt(data, scales, 'morl', sampling_period=dt)
        power = np.abs(coeffs)**2
        # Average power over the scales to obtain a time series
        return np.mean(power, axis=0)

    power_h1 = compute_wavelet_power(white_h1, scales, dt)
    power_l1 = compute_wavelet_power(white_l1, scales, dt)

    # Step 3: Coherent network analysis: combine power from both detectors
    # Here we take a geometric mean to favor coincident peaks between detectors
    combined_metric = np.sqrt(power_h1 * power_l1)
    
    # Step 4: Trigger generation with multi-detector consistency and signal-based veto criteria
    # Use a dynamic threshold based on the median background level. Also incorporate a minimal
    # spacing to enforce a time uncertainty (veto nearby triggers).
    background_level = np.median(combined_metric)
    threshold = background_level * 1.2  # slightly above median background
    # Find peaks that exceed the threshold; distance set to enforce a ~0.5 sec separation.
    min_distance = int(0.5 / dt)
    peaks, properties = signal.find_peaks(combined_metric, height=threshold, distance=min_distance, prominence=background_level * 0.3)
    
    # Step 5: Post-trigger signal-based veto (placeholder for further veto analysis)
    # For demonstration, we apply a simple veto: reject triggers with unusually low prominence.
    valid_indices = properties['prominences'] > (background_level * 0.35)
    refined_peaks = peaks[valid_indices]

    # Assign uncertainties (here we assume a fixed uncertainty of 5 seconds per candidate)
    peak_deltat = np.full(len(refined_peaks), 5.0)
    peak_times = times[refined_peaks]
    peak_heights = combined_metric[refined_peaks]

    return peak_times, peak_heights, peak_deltat
