import numpy as np

def project(x, X):
    if x.ndim == 1:
        d = (X - np.expand_dims(x, 0)) ** 2
        d = d.mean(axis=1)
    else:
        d = (X - np.expand_dims(x, 0)) ** 2
        d = d.mean(axis=(1, 2))

    return np.argmin(d)