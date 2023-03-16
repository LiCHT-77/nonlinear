import numpy as np
from scipy.spatial.distance import pdist


def fractal_gp(state_space, scale_range: tuple[float, float], split=100) -> np.ndarray:
    logr = np.exp(np.linspace(
        np.log(scale_range[0]), np.log(scale_range[1]), split))
    N = len(state_space)

    logCr = (np.expand_dims(pdist(state_space), axis=1) <=
             np.expand_dims(logr, axis=0)).sum(axis=0) / N**2
    logCr[logCr <= 0] = np.nan
    logr = np.log(logr)
    logCr = np.log(logCr)
    slope = np.abs(np.pad(np.diff(logCr), [1, 0], 'constant', constant_values=(
        np.nan, np.nan))) / np.pad(np.diff(logr), [1, 0], 'constant', constant_values=(np.nan, np.nan))
    # columns = 'log r', 'log C^m(r)', 'Slope'
    return np.concatenate([logr.reshape((-1, 1)), logCr.reshape((-1, 1)), slope.reshape((-1, 1))], 1)
