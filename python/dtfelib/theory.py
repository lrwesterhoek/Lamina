"""Linear-theory expectations for the void statistics: the void size function (Sheth & van de Weygaert 2004; the
'vdn' volume-conserving count of Jennings, Li & Hu 2013 is the default) and the shapes of the minima of a Gaussian
random field (Bardeen, Bond, Kaiser & Szalay 1986). h-free units; the linear spectrum is spectra.linear_pk's."""

from __future__ import annotations

import math

import numpy as np

import config
from . import spectra

DELTA_V_LINEAR = -2.71        # SvdW's linear shell-crossing threshold (Sheth & van de Weygaert 2004)
DELTA_V_NONLINEAR = -0.8      # the nonlinear density contrast of a void at shell crossing
DELTA_C = 1.686               # the linear collapse threshold (the void-in-cloud barrier)
SMALL_X = 0.1                 # below this x the two-barrier sum takes its one-barrier limit (error < 1e-6)

_trapz = getattr(np, "trapezoid", None) or np.trapz          # NumPy 1.26 (CI's) has only trapz
_K = np.geomspace(1e-5, 1e3, 6000)                            # 1/Mpc: the sigma(R) integration grid (ln k)


def cosmology() -> dict:
    """The cosmology every function here uses by default: config.COSMOLOGY with h."""
    return dict(config.COSMOLOGY, h=config.HUBBLE_H)


def growth(z, cosmo: dict | None = None):
    """D(z), D(0) = 1: spectra.growth_factor (flat LambdaCDM, exact integral)."""
    return spectra.growth_factor(z, cosmo)


def transfer(k_mpc, cosmo: dict | None = None):
    """T(k), k in 1/Mpc: spectra.transfer_eh98 (Eisenstein & Hu 1998, no wiggles)."""
    return spectra.transfer_eh98(k_mpc, cosmo)


_PK0 = {}


def _pk0(cosmo: dict | None = None) -> np.ndarray:
    """P_lin(k, z = 0) on _K, cached per cosmology (the sigma_8 normalisation is a 20000-point integral)."""
    c = cosmo or dict(config.COSMOLOGY)
    key = tuple(sorted(c.items()))
    if key not in _PK0:
        _PK0[key] = spectra.linear_pk(_K, 0.0, c)
    return _PK0[key]


def sigma_tophat(r_mpc, z: float = 0.0, cosmo: dict | None = None) -> np.ndarray:
    """sigma(R, z) of the linear density in top-hat spheres of radius R [Mpc]: sqrt(int P W^2 k^3 dlnk / 2 pi^2)
    times D(z). Vectorised over R; sigma_tophat(8 / h) = sigma_8."""
    r = np.atleast_1d(np.asarray(r_mpc, dtype=np.float64))
    integrand = _pk0(cosmo)[None, :] * spectra._tophat(np.outer(r, _K)) ** 2 * _K[None, :] ** 3 / (2.0 * np.pi ** 2)
    s = np.sqrt(_trapz(integrand, np.log(_K), axis=1)) * float(growth(z, cosmo))
    return s if np.ndim(r_mpc) else float(s[0])


def spectral_moments(sigma_g_mpc: float, z: float = 0.0, cosmo: dict | None = None) -> tuple:
    """(sigma_0, sigma_1, sigma_2) of the linear field smoothed with a Gaussian of 'sigma_g_mpc' (the pipeline's
    smoothing length): sigma_j^2 = int k^(2j) P(k) exp(-k^2 R^2) k^3 dlnk / 2 pi^2 (BBKS eq 4.6a)."""
    base = _pk0(cosmo) * np.exp(-(_K * float(sigma_g_mpc)) ** 2) * _K ** 3 / (2.0 * np.pi ** 2)
    d = float(growth(z, cosmo))
    return tuple(float(np.sqrt(_trapz(base * _K ** (2 * j), np.log(_K)))) * d for j in (0, 1, 2))


def bbks_gamma(sigma_g_mpc: float, cosmo: dict | None = None) -> tuple:
    """(gamma, R_*) of the Gaussian-smoothed linear spectrum: gamma = sigma_1^2 / (sigma_0 sigma_2) (the width of
    the spectrum, 0 < gamma < 1; independent of z in linear theory), R_* = sqrt(3) sigma_1 / sigma_2 [Mpc]."""
    s0, s1, s2 = spectral_moments(sigma_g_mpc, 0.0, cosmo)
    return s1 * s1 / (s0 * s2), math.sqrt(3.0) * s1 / s2


# ---------------------------------------------------------------- the void size function

def svdw_multiplicity(sigma, delta_v: float = DELTA_V_LINEAR, delta_c: float = DELTA_C) -> np.ndarray:
    """f_lnsigma(sigma): the two-barrier first-crossing distribution in ln(1/sigma) (module docstring). delta_c
    = inf (or 0 < delta_c not finite) is the one-barrier limit."""
    s = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    dv = abs(float(delta_v))
    one = np.sqrt(2.0 / np.pi) * (dv / s) * np.exp(-0.5 * (dv / s) ** 2)
    if not np.isfinite(delta_c):
        return one if np.ndim(sigma) else float(one[0])
    dd = dv / (float(delta_c) + dv)
    x = dd * s / dv
    out = one.copy()
    big = x >= SMALL_X
    if big.any():
        xb = x[big]
        jmax = int(np.ceil(12.0 / (np.pi * xb.min()))) + 2           # the last term below exp(-72)
        j = np.arange(1, jmax + 1, dtype=np.float64)[None, :]
        terms = np.exp(-0.5 * (j * np.pi * xb[:, None]) ** 2) * j * np.pi * xb[:, None] ** 2 * np.sin(j * np.pi * dd)
        out[big] = 2.0 * terms.sum(axis=1)
    out = np.clip(out, 0.0, None)                                    # the sum's last-digit noise at large x
    return out if np.ndim(sigma) else float(out[0])


def linear_radius_ratio(delta_v_nl: float = DELTA_V_NONLINEAR) -> float:
    """R / R_L of a void at shell crossing: (1 + delta_v_NL)^(-1/3) (1.71 for -0.8)."""
    return (1.0 + float(delta_v_nl)) ** (-1.0 / 3.0)


def void_size_function(r_mpc, z: float = 0.0, model: str = "vdn", delta_v: float = DELTA_V_LINEAR,
                       delta_c: float = DELTA_C, delta_v_nl: float = DELTA_V_NONLINEAR,
                       cosmo: dict | None = None) -> np.ndarray:
    """dn/dlnR [Mpc^-3] of voids of nonlinear radius R [Mpc] at redshift z: 'vdn' (Jennings, Li & Hu 2013, the
    default) or 'svdw' (Sheth & van de Weygaert 2004); see the module docstring."""
    if model not in ("vdn", "svdw"):
        raise ValueError(f"model must be 'vdn' or 'svdw', got {model!r}")
    r = np.atleast_1d(np.asarray(r_mpc, dtype=np.float64))
    r_l = r / linear_radius_ratio(delta_v_nl)
    eps = 1e-3                                                       # dln(1/sigma)/dlnR_L, centred in ln R
    s = sigma_tophat(r_l, z, cosmo)
    sp, sm = sigma_tophat(r_l * math.exp(eps), z, cosmo), sigma_tophat(r_l * math.exp(-eps), z, cosmo)
    dlninv = -(np.log(sp) - np.log(sm)) / (2.0 * eps)
    vol = 4.0 / 3.0 * np.pi * (r if model == "vdn" else r_l) ** 3
    out = svdw_multiplicity(s, delta_v, delta_c) / vol * dlninv
    return out if np.ndim(r_mpc) else float(out[0])


def void_dn_dlnr_binned(edges_mpc, z: float = 0.0, n: int = 21, **kw) -> np.ndarray:
    """dn/dlnR [Mpc^-3] AVERAGED in ln R over each bin of 'edges_mpc' (len(edges) - 1 values; n points per bin):
    what a histogram of counts / (V dlnR) measures. The model falls by decades across a wide bin, so its
    value at the bin centre is not the comparison."""
    e = np.asarray(edges_mpc, dtype=np.float64)
    return np.array([np.mean(void_size_function(np.geomspace(lo, hi, int(n)), z, **kw))
                     for lo, hi in zip(e[:-1], e[1:])])


def void_number_between(r_lo_mpc: float, r_hi_mpc: float, z: float = 0.0, n: int = 81, **kw) -> float:
    """The model's number density [Mpc^-3] of voids with r_lo <= R < r_hi: dn/dlnR integrated in ln R (trapezoid on
    n points; the fixed-R size function's one-number ratio). 0 for an empty range."""
    lo, hi = float(r_lo_mpc), float(r_hi_mpc)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        return 0.0
    g = np.geomspace(lo, hi, int(n))
    return float(_trapz(void_size_function(g, z, **kw), np.log(g)))


def void_cumulative(r_mpc, z: float = 0.0, r_max_mpc: float | None = None, **kw) -> np.ndarray:
    """n(>R) [Mpc^-3]: dn/dlnR integrated in ln R from each R to r_max_mpc (default 100 R_max(r_mpc), where the
    abundance is exp-small)."""
    r = np.atleast_1d(np.asarray(r_mpc, dtype=np.float64))
    top = float(r_max_mpc) if r_max_mpc else 100.0 * float(r.max())
    grid = np.geomspace(float(r.min()), top, 1200)
    dn = void_size_function(grid, z, **kw)
    lg = np.log(grid)
    seg = 0.5 * (dn[1:] + dn[:-1]) * np.diff(lg)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])       # n(> grid[i])
    out = np.interp(np.log(r), lg, tail)
    return out if np.ndim(r_mpc) else float(out[0])


# ---------------------------------------------------------------- BBKS minima shapes

def bbks_f(x) -> np.ndarray:
    """BBKS eq A15: the curvature factor f(x) of the peak number density (f -> x^3 - 3x at large x)."""
    x = np.asarray(x, dtype=np.float64)
    a = np.sqrt(2.5)
    return ((x ** 3 - 3.0 * x) / 2.0 * (_erf(a * x) + _erf(a * x / 2.0))
            + np.sqrt(2.0 / (5.0 * np.pi)) * ((31.0 * x * x / 4.0 + 1.6) * np.exp(-5.0 * x * x / 8.0)
                                              + (x * x / 2.0 - 1.6) * np.exp(-2.5 * x * x)))


def _erf(x):
    from scipy.special import erf
    return erf(x)


def _ep_grid(n: int = 241):
    """The (e, p) grid of the shape pdfs (cell centres; e in (0, 1/2), p in (-1/2, 1/2)) and BBKS's eigenvalue
    weight e (e^2 - p^2) (1 - 2p) ((1 + p)^2 - 9e^2) on its domain, 0 outside it."""
    e = (np.arange(n) + 0.5) * (0.5 / n)
    p = -0.5 + (np.arange(2 * n) + 0.5) * (0.5 / n)
    E, P = np.meshgrid(e, p, indexing="ij")
    dom = (P <= E) & (np.where(E <= 0.25, P >= -E, P >= 3.0 * E - 1.0))
    w = np.where(dom, E * (E * E - P * P) * (1.0 - 2.0 * P) * ((1.0 + P) ** 2 - 9.0 * E * E), 0.0)
    return e, p, np.clip(w, 0.0, None), (0.5 / n) ** 2


def bbks_ep_pdf(x: float, n: int = 241):
    """P(e, p | x) of BBKS eq 7.6 on the _ep_grid (e, p, pdf), the analytic normalisation (3^2 5^(5/2) /
    sqrt(2 pi) x^8 / f(x)): its grid integral is 1 to the grid's accuracy (the test of f)."""
    e, p, w, _da = _ep_grid(n)
    E, P = np.meshgrid(e, p, indexing="ij")
    pdf = 9.0 * 5.0 ** 2.5 / math.sqrt(2.0 * math.pi) * x ** 8 / float(bbks_f(x)) * w \
        * np.exp(-2.5 * x * x * (3.0 * E * E + P * P))
    return e, p, pdf


_X = np.linspace(0.02, 8.0, 400)        # the curvature grid of the mixtures
_MARG = {}


def _marginals(n: int = 241):
    """Per x of _X: the e- and p-marginals of P(e, p | x), each normalised on its grid (cached)."""
    if n not in _MARG:
        e, p, w, da = _ep_grid(n)
        E2, P2 = np.meshgrid(e, p, indexing="ij")
        q = 3.0 * E2 * E2 + P2 * P2
        me, mp = np.zeros((_X.size, e.size)), np.zeros((_X.size, p.size))
        for i, x in enumerate(_X):
            j = w * np.exp(-2.5 * x * x * q)
            tot = j.sum()
            if tot > 0:
                me[i], mp[i] = j.sum(axis=1) / tot, j.sum(axis=0) / tot
        _MARG[n] = (e, p, me, mp)
    return _MARG[n]


def peak_curvature_pdf(nu=None, gamma: float = 0.5) -> np.ndarray:
    """The curvature x of peaks on _X, normalised: at height nu, f(x) exp(-(x - gamma nu)^2 / 2(1 - gamma^2))
    (BBKS eq A14); nu None: all peaks, f(x) exp(-x^2 / 2). An array of nu gives their MIXTURE (each peak one
    weight), the sample's own distribution of heights."""
    f = np.clip(bbks_f(_X), 0.0, None)
    if nu is None:
        w = f * np.exp(-0.5 * _X ** 2)
    else:
        nus = np.atleast_1d(np.asarray(nu, dtype=np.float64))
        nus = nus[np.isfinite(nus)]
        if not nus.size:
            return np.full(_X.size, np.nan)
        g = float(gamma)
        rows = f[None, :] * np.exp(-0.5 * (_X[None, :] - g * nus[:, None]) ** 2 / (1.0 - g * g))
        rows /= np.maximum(rows.sum(axis=1, keepdims=True), 1e-300)
        w = rows.mean(axis=0)
    return w / w.sum()


def minima_shape_pdf(nu=None, gamma: float = 0.5) -> dict:
    """The pdfs of e and of p of the minima of a Gaussian random field (as minima_shape_percentiles): {'e', 'pdf_e',
    'p', 'pdf_p'}, each pdf a density on its grid (integral 1)."""
    e, p, me, mp = _marginals()
    wx = peak_curvature_pdf(nu, gamma)
    pe, pp = wx @ me, wx @ mp
    return {"e": e, "pdf_e": pe / (e[1] - e[0]), "p": p, "pdf_p": pp / (p[1] - p[0])}


def minima_shape_percentiles(nu=None, gamma: float = 0.5, pct=(16, 50, 84)) -> dict:
    """The percentiles 'pct' of the BBKS ellipticity e and prolateness p of the minima of a Gaussian random field:
    {'e': array, 'p': array, 'e_mean', 'p_mean'}. nu (= -delta_min / sigma_0 per minimum, an array) conditions on the
    sample's heights with the spectrum's gamma; None takes all minima (spectrum-independent, the module docstring)."""
    e, p, me, mp = _marginals()
    wx = peak_curvature_pdf(nu, gamma)
    if not np.isfinite(wx).all():
        nan = np.full(len(pct), np.nan)
        return {"e": nan, "p": nan, "e_mean": np.nan, "p_mean": np.nan}
    pe, pp = wx @ me, wx @ mp

    def quantiles(grid, pdf):
        step = grid[1] - grid[0]
        cdf = np.concatenate([[0.0], np.cumsum(pdf)])
        edges = np.concatenate([[grid[0] - step / 2], grid + step / 2])
        return np.interp(np.asarray(pct, float) / 100.0, cdf / cdf[-1], edges)
    return {"e": quantiles(e, pe), "p": quantiles(p, pp),
            "e_mean": float((e * pe).sum() / pe.sum()), "p_mean": float((p * pp).sum() / pp.sum())}
