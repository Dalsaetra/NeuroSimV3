import numpy as np

def _rank01(arr):
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(arr), endpoint=True)
    return ranks

def bin_counts(spikes_bool, bin_steps):
    """
    spikes_bool: (N, T) boolean
    bin_steps: int (# time steps per bin)
    returns counts: (N, n_bins)
    """
    N, T = spikes_bool.shape
    n_bins = T // bin_steps
    if n_bins == 0:
        return np.zeros((N, 0), dtype=float)
    trimmed = spikes_bool[:, :n_bins * bin_steps]
    return trimmed.reshape(N, n_bins, bin_steps).sum(axis=2)

def power_spectrum_fft(x, fs_hz):
    """
    Simple, windowless PSD estimate via FFT (sufficient for QC dashboards).
    x: 1D array
    fs_hz: sampling rate (Hz)
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    # rfft -> one-sided spectrum
    X = np.fft.rfft(x)
    Pxx = (np.abs(X) ** 2) / (n * fs_hz)
    f = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return f, Pxx

def spectral_entropy(Pxx):
    p = Pxx / (Pxx.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())