from gen_inst import FAR_MIN, FAR_MAX

def calculate_auc(result_data, metric_name='sensitive-distance',
                   far_scaling_factor=31557600/12, far_min=FAR_MIN, far_max=FAR_MAX):
    """
    Calculate Area Under Curve (AUC) for sensitivity curves.
    
    Parameters
    ----------
    result_data : dict
        Dictionary containing 'far' and 'sensitive-distance' data
    metric_name : str
        Name of the metric to calculate AUC for (default: 'sensitive-distance')
    far_scaling_factor : float
        Scaling factor for false alarm rate (default: 1/month)
    far_min : float or None
        Minimum false alarm rate to consider (default: 1/month), if None, use all data
    far_max : float or None
        Maximum false alarm rate to consider (default: 1e3/month), if None, use all data
    """
    import numpy as np
    from scipy.integrate import trapz
    import scipy.interpolate
    
    if min(result_data['far'][:-1] * far_scaling_factor) > far_max:
        # Store AUC in results
        result_data['auc'] = 0
        result_data['far_min'] = far_min
        result_data['far_max'] = far_max
        return result_data

    # Get indices where FAR is between far_min and far_max per month
    if far_min is None and far_max is None:
        mask = np.ones(len(result_data['far']), dtype=bool)
    elif far_min is None:
        mask = result_data['far'] * far_scaling_factor <= far_max
    elif far_max is None:
        mask = result_data['far'] * far_scaling_factor >= far_min
    else:
        mask = (result_data['far'] * far_scaling_factor <= far_max) & (result_data['far'] * far_scaling_factor >= far_min)
    x = np.log10(result_data['far'][mask] * far_scaling_factor)#[1:-1]
    y = result_data[metric_name][mask]#[1:-1]
    
    # Sort x and y to ensure strictly increasing sequence
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # Interpolate to get more points and apply ReLU
    x_interp = np.linspace(x.min(), x.max(), 2000)
    y_interp = scipy.interpolate.CubicSpline(x, y)(x_interp)
    y_interp = np.maximum(0, y_interp)  # Apply ReLU to ensure non-negative values
     
    # Store interpolated results
    result_data['x_interp'] = x_interp
    result_data['y_interp'] = y_interp
    
    # Calculate AUC using interpolated points
    auc = trapz(y_interp, x=x_interp)
    
    # Store AUC in results
    result_data['auc'] = auc
    result_data['far_min'] = far_min
    result_data['far_max'] = far_max
