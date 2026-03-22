import numpy as np
from scipy.signal import find_peaks

def pipeline_v2(strain_h1, strain_l1, times):
    dt = np.median(np.diff(times))
    
    # Step 1: Robust scaling using median and MAD (median absolute deviation)
    def robust_zscore(signal):
        med = np.median(signal)
        mad = np.median(np.abs(signal - med))
        # Fallback if mad is zero
        if mad == 0:
            mad = np.mean(np.abs(signal - med))
        return (signal - med) / mad
    scaled_h1 = robust_zscore(strain_h1)
    scaled_l1 = robust_zscore(strain_l1)
    
    # Step 2: Envelope extraction using squared signal smoothing via convolution with a Hann window
    def smooth_envelope(signal):
        squared = signal**2
        # set a smoothing window length corresponding to 2 seconds and ensure it's odd if needed
        win_len = max(5, int(2.0/dt))
        # Create a Hann window for smoothing
        hann_win = np.hanning(win_len)
        hann_win = hann_win / np.sum(hann_win)
        # Perform convolution with mode 'same'
        smooth_sq = np.convolve(squared, hann_win, mode='same')
        # Return RMS-like envelope
        return np.sqrt(np.abs(smooth_sq))
    
    envelope_h1 = smooth_envelope(scaled_h1)
    envelope_l1 = smooth_envelope(scaled_l1)
    
    # Step 3: Combine envelopes using harmonic mean to emphasize simultaneous peaks
    eps = 1e-10
    harmonic_mean = 2.0 / ((1.0/(envelope_h1 + eps)) + (1.0/(envelope_l1 + eps)))
    combined_envelope = harmonic_mean
    
    # Step 4: Adaptive dynamic threshold using median and MAD of the envelope
    med_env = np.median(combined_envelope)
    mad_env = np.median(np.abs(combined_envelope - med_env))
    threshold = med_env + 1.5 * mad_env
    
    # Step 5: Peak detection with minimal distance set by half the window length
    min_distance = max(1, int((2.0/dt) / 2))
    peaks, _ = find_peaks(combined_envelope, height=threshold, distance=min_distance)
    
    refined_peak_times = []
    peak_heights = []
    peak_deltat = []  # uncertainty estimate via FWHM from quadratic fit
    
    # Step 6: Refine peak timing using a 7-point quadratic polynomial fit when possible, fallback to 5 points
    for p in peaks:
        # choose number of points: try 7 points; if not available, use 5 points
        if p >= 3 and p <= len(combined_envelope)-4:
            idx_window = np.arange(p-3, p+4)
        elif p >= 2 and p <= len(combined_envelope)-3:
            idx_window = np.arange(p-2, p+3)
        else:
            refined_peak_times.append(times[p])
            peak_heights.append(combined_envelope[p])
            peak_deltat.append(dt)
            continue
        
        t_window = times[idx_window]
        # Fit quadratic: f(t) = a*t^2 + b*t + c using np.polyfit
        coeffs = np.polyfit(t_window, combined_envelope[idx_window], 2)
        a, b, c = coeffs
        
        # Refined peak time: vertex of the parabola if a != 0
        if a != 0:
            t_refined = -b / (2 * a)
        else:
            t_refined = times[p]
        refined_peak_times.append(t_refined)
        
        peak_val = np.polyval(coeffs, t_refined)
        peak_heights.append(peak_val)
        
        # Estimate FWHM by solving for the points where f(t) = peak_val/2
        half_max = peak_val / 2.0
        # Solve quadratic equation: a*t^2 + b*t + (c-half_max)=0
        disc = b**2 - 4*a*(c-half_max)
        if a != 0 and disc >= 0:
            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            width = np.abs(t2 - t1)
        else:
            width = dt  # fallback uncertainty
        peak_deltat.append(width)
    
    refined_peak_times = np.array(refined_peak_times)
    peak_heights = np.array(peak_heights)
    peak_deltat = np.array(peak_deltat)
    
    result = (refined_peak_times, peak_heights, peak_deltat)
    return result
