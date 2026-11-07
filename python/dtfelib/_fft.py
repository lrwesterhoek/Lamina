"""The real FFTs dtfelib uses (pipeline's Hessian / gradient / potential, spectra's P(k)): scipy.fft on every core
(workers=-1) when scipy is installed, else numpy.fft. One shim, so the two modules cannot drift apart."""

import numpy as np

try:
    from scipy import fft as _sfft

    def rfftn(a):
        return _sfft.rfftn(a, workers=-1)

    def irfftn(a, shape):
        return _sfft.irfftn(a, s=shape, workers=-1)
except ImportError:                                     # pragma: no cover
    def rfftn(a):
        return np.fft.rfftn(a)

    def irfftn(a, shape):
        return np.fft.irfftn(a, s=shape)
