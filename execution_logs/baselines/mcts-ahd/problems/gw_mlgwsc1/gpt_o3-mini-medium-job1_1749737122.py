import numpy as np
from scipy.signal import butter, filtfilt, hilbert, find_peaks, peak_widths

def pipeline_v2(strain_h1, strain_l1, times):
    """
    Process gravitational wave data from H1 and L1 detectors to identify candidate signals.

    Inputs:
      strain_h1: numpy array of strain data from H1 detector
      strain_l1: numpy array of strain data from L1 detector
      times: numpy array of time points corresponding to the data samples

    Returns:
      result: a tuple of three numpy arrays (peak_times, peak_heights, peak_deltat) where:
         peak_times: GPS times (from times array) of identified events
         peak_heights: significance values (peak amplitudes) of each event
         peak_deltat: time window uncertainties (peak widths in seconds) of each event
    """
    # Estimate sampling frequency from the times array
    dt = times[1] - times[0]
    fs = 1.0 / dt

    # Design a Butterworth bandpass filter (30-300 Hz typical for GW data)
    lowcut = 30.0
    highcut = 300.0
    order = 4
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Filter the signals
    filtered_h1 = filtfilt(b, a, strain_h1)
    filtered_l1 = filtfilt(b, a, strain_l1)

    # Combine the two channels (simple average)
    combined_signal = 0.5 * (filtered_h1 + filtered_l1)

    # Compute the envelope of the combined signal using the Hilbert transform
    analytic_signal = hilbert(combined_signal)
    envelope = np.abs(analytic_signal)

    # Threshold determination: mean + 3*std
    threshold = envelope.mean() + 3 * envelope.std()

    # Find peaks in the envelope that exceed the threshold
    peaks, properties = find_peaks(envelope, height=threshold)

    # Calculate peak widths at half prominence using scipy.signal.peak_widths
    # This returns widths in number of samples so convert to seconds using dt.
    if len(peaks) > 0:
        widths_result = peak_widths(envelope, peaks, rel_height=0.5)
        widths_in_samples = widths_result[0]
        peak_deltat = widths_in_samples * dt
    else:
        peak_deltat = np.array([])

    # Extract peak properties
    peak_times = times[peaks]
    peak_heights = properties['peak_heights']

    result = (peak_times, peak_heights, peak_deltat)
    return result
