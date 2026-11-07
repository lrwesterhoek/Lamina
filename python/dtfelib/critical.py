"""Critical points of the smoothed density contrast: its minima, maxima and the two kinds of saddle, each with
the Hessian eigenvalues at its cell (find_critical_points; SnapshotProducts.critical_points caches them). The
minima filter is config.FOOTPRINT_SIZE wide, a saddle a near-zero gradient (below config.GRADIENT_THRESHOLD of the
largest) at a local maximum of gmax - |grad|. Moved out of dtfelib.pipeline, which re-exports both functions."""

from __future__ import annotations

import numpy as np

import config
from . import fields as dtfe


def eigh_at_coords(hessian: dict, coords) -> tuple[np.ndarray, np.ndarray]:
    """The Hessian's eigenvalues (ascending, (n, 3)) and eigenvectors ((n, 3, 3), columns) at the integer cells
    'coords' ((n, 3)), in float64; empty arrays for no cell."""
    coords = np.asarray(coords, dtype=int)
    if coords.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3, 3))
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
    H = np.empty((len(coords), 3, 3), dtype=np.float64)
    H[:, 0, 0] = hessian['hxx'][i, j, k]
    H[:, 0, 1] = H[:, 1, 0] = hessian['hxy'][i, j, k]
    H[:, 0, 2] = H[:, 2, 0] = hessian['hxz'][i, j, k]
    H[:, 1, 1] = hessian['hyy'][i, j, k]
    H[:, 1, 2] = H[:, 2, 1] = hessian['hyz'][i, j, k]
    H[:, 2, 2] = hessian['hzz'][i, j, k]
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def find_critical_points(delta_s: np.ndarray, hessian: dict, grad_mag: np.ndarray) -> dict:
    """The critical points of the smoothed contrast 'delta_s': {'minima', 'maxima', 'saddle1', 'saddle2'}, each
    {'coords', 'values', 'eigenvalues'}, plus 'field_mean' / 'field_std'. Minima and maxima are the cells equal to
    their config.FOOTPRINT_SIZE box minimum / maximum (periodic); a saddle candidate has a gradient below
    config.GRADIENT_THRESHOLD of the largest at a local maximum of gmax - |grad| and is classed by its number of
    negative eigenvalues (1 or 2)."""
    # the three box filters run slab-threaded (dtfe.box_filter, bit-identical to scipy's serial minimum_filter /
    # maximum_filter): at 512^3 serial filters are ~3 passes of several seconds each
    fp = config.FOOTPRINT_SIZE
    minima_mask = (delta_s == dtfe.box_filter(delta_s, fp, mode='wrap'))
    maxima_mask = (delta_s == dtfe.box_filter(delta_s, fp, mode='wrap', maximum=True))

    gmax = float(np.max(grad_mag))
    gnorm = grad_mag / gmax if gmax > 0 else grad_mag
    cand = gnorm < config.GRADIENT_THRESHOLD
    inv = gmax - grad_mag
    cand &= (inv == dtfe.box_filter(inv, fp, mode='wrap', maximum=True))
    cand &= ~minima_mask
    cand &= ~maxima_mask

    out = {}
    for name, mask in (('minima', minima_mask), ('maxima', maxima_mask)):
        coords = np.argwhere(mask)
        evals, _ = eigh_at_coords(hessian, coords)
        out[name] = {'coords': coords, 'values': delta_s[mask], 'eigenvalues': evals}

    coords = np.argwhere(cand)
    evals, _ = eigh_at_coords(hessian, coords)
    nneg = (evals < 0).sum(axis=1)
    vals = delta_s[cand]
    for name, n in (('saddle1', 1), ('saddle2', 2)):
        sel = nneg == n
        out[name] = {'coords': coords[sel], 'values': vals[sel], 'eigenvalues': evals[sel]}

    out['field_mean'] = float(np.mean(delta_s))
    out['field_std'] = float(np.std(delta_s))
    return out
