import numpy as np
from scipy.signal import butter, filtfilt, hilbert, find_peaks, peak_widths, savgol_filter
from scipy.signal import medfilt

def pipeline_v2(strain_h1, strain_l1, times):
    # Estimate sampling frequency.
    dt = np.median(np.diff(times))
    fs = 1.0 / dt

    # Detrend the strain data using a linear polynomial fit.
    # This subtracts a best-fit line from each channel.
    def poly_detrend(signal):
        t = np.arange(len(signal))
        p = np.polyfit(t, signal, 1)
        trend = np.polyval(p, t)
        return signal - trend

    strain_h1 = poly_detrend(strain_h1)
    strain_l1 = poly_detrend(strain_l1)

    # Define Butterworth bandpass filter parameters (30-340 Hz, order 2 for a slightly gentler roll-off).
    lowcut = 30.0
    highcut = 340.0
    order = 2
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter with zero-phase filtering.
    filt_h1 = filtfilt(b, a, strain_h1)
    filt_l1 = filtfilt(b, a, strain_l1)

    # Compute analytic signals and get their Hilbert envelopes.
    envelope_h1 = np.abs(hilbert(filt_h1))
    envelope_l1 = np.abs(hilbert(filt_l1))
    
    # Combine envelopes using a weighted average of the arithmetic and geometric means.
    eps = 1e-12
    arithmetic_mean = (envelope_h1 + envelope_l1) / 2.0
    # Compute geometric mean safely using logarithms.
    geometric_mean = np.exp((np.log(envelope_h1 + eps) + np.log(envelope_l1 + eps)) / 2.0)
    combined_envelope = 0.5 * arithmetic_mean + 0.5 * geometric_mean

    # Apply Savitzky–Golay filtering with a window spanning 0.3 seconds (ensuring an odd window length) and polynomial order 2.
    window_len = int(0.3 * fs)
    if window_len % 2 == 0:
        window_len += 1
    if window_len < 3:
        window_len = 3
    sg_filtered = savgol_filter(combined_envelope, window_length=window_len, polyorder=2)
    
    # Further smooth the signal using a median filter over a window corresponding to 0.1-second duration.
    med_window = int(0.1 * fs)
    if med_window % 2 == 0:
        med_window += 1
    if med_window < 3:
        med_window = 3
    smooth_signal = medfilt(sg_filtered, kernel_size=med_window)

    # Dynamic thresholding: compute the interquartile range (IQR) and set threshold as median plus 1.5*IQR.
    med_val = np.median(smooth_signal)
    q25, q75 = np.percentile(smooth_signal, [25, 75])
    iqr = q75 - q25
    threshold = med_val + 1.5 * iqr

    # Detect peaks above the computed dynamic threshold.
    peaks, properties = find_peaks(smooth_signal, height=threshold)
    
    # Estimate timing uncertainty using peak widths at half prominence.
    if peaks.size > 0:
        widths, _, _, _ = peak_widths(smooth_signal, peaks, rel_height=0.5)
        peak_deltat = widths * dt / 2.0
    else:
        peak_deltat = np.array([])

    # Map detected peak indices to GPS times and extract peak heights.
    peak_times = times[peaks]
    peak_heights = properties["peak_heights"]

    result = (peak_times, peak_heights, peak_deltat)
    return result
