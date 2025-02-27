import numpy as np

def find_confidence_intervals(sigma):
    from scipy.stats import norm
    if sigma <= 0:
        raise ValueError('sigma must be greater than 0')
    
    p = np.diff(norm.cdf([-sigma, sigma]))[0]
    return np.array([1-p, p]) * 100


def delinearize(x, y):
    A = x**2 + y**2
    B = np.arccos(y / (A ** 0.5)) if A != 0 else 0
    if x < 0:
        B = 2 * np.pi - B
    return np.array([A, B])


def adelinearize(s, c):
    # x sine, y cosine
    A = s**2 + c**2
    B = np.zeros_like(A) if A.all() == 0 else np.arccos(c / (A ** 0.5))
    B[s<0] = 2 * np.pi - B[s<0]

    #where is slower
    #B = np.where(x>0, np.arccos(y / (A ** 0.5)), 2 * np.pi - np.arccos(y[x<0] / (A[x<0] ** 0.5)))
    return np.array([A, B])


def cps(pers, amps, eccs, starmass):
    #sma, minmass = np.zeros(kplanets), np.zeros(kplanets)
    G = 6.674e-11  # m3 / (kg * s2)
    #m2au = 6.685e-12  # au
    #kg2sm = 5.03e-31  # solar masses
    #s2d = 1.157e-5  # days
    #G_ = G * m2au**3 / (kg2sm*s2d)  # au3 / (sm * d2)

    consts = 4*np.pi**2/(G*1.99e30)

    sma = ((pers*24*3600)**2 * starmass / consts)**(1./3) / 1.49598e11
    minmass = amps / ( (28.4329/np.sqrt(1. - eccs**2.)) * (starmass**(-0.5)) * (sma**(-0.5)) )

    return sma, minmass


def hdi_of_samples(samples, cred_mass=0.9):
    sorted_samples = np.sort(samples)
    n_samples = len(sorted_samples)
    interval_idx_inc = int(np.floor(cred_mass * n_samples))
    n_intervals = n_samples - interval_idx_inc
    
    intervals_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    
    min_idx = np.argmin(intervals_width)
    
    hdi_min = sorted_samples[min_idx]
    hdi_max = sorted_samples[min_idx + interval_idx_inc]
    
    return hdi_min, hdi_max


def hdi_of_chain(chain, cred_mass=0.9):
    """
    Calculate the HDI for each dimension in an MCMC chain.

    Parameters:
    chain (numpy.ndarray): The MCMC chain with shape (steps, dimensions).
    cred_mass (float): The credible mass for the HDI (e.g., 0.95 for 95%).

    Returns:
    hdi (numpy.ndarray): Array of shape (dimensions, 2) containing the HDI bounds 
                         for each dimension.
    """
    steps, dimensions = chain.shape
    hdi = np.zeros((dimensions, 2))

    for dim in range(dimensions):
        sorted_samples = np.sort(chain[:, dim])
        n_samples = len(sorted_samples)
        interval_idx_inc = int(np.floor(cred_mass * n_samples))
        n_intervals = n_samples - interval_idx_inc
        
        intervals_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
        
        min_idx = np.argmin(intervals_width)
        
        hdi_min = sorted_samples[min_idx]
        hdi_max = sorted_samples[min_idx + interval_idx_inc]
        
        hdi[dim, 0] = hdi_min
        hdi[dim, 1] = hdi_max

    return hdi


def getExtremePoints(data, typeOfExtreme = None, maxPoints = None):
    """
    from https://towardsdatascience.com/modality-tests-and-kernel-density-estimations-3f349bb9e595
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange == 1)[0]

    if typeOfExtreme == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]

    elif typeOfExtreme == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]

    elif typeOfExtreme is not None:
        idx = idx[::2]

    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data) - 1) in idx:
        idx = np.delete(idx, len(data)-1)

    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx = idx[-maxPoints:]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data)//(maxPoints + 1))

    return idx


def running(arr, window_size=10):
    """Calculate running average of the last n values."""
    averages = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        start_idx = max(0, i - window_size + 1)  # Start of the window (avoid negative index)
        averages[i] = np.mean(arr[start_idx:i + 1], axis=0)  # Average of the last `window_size` points
    return averages


def minmax(x):
    return np.amin(x), np.amax(x)






