"""The measured void radius R_v and its two companions, from ONE matrix of shell means of the smoothed delta about
each minimum (void_radii_cells): the crossing (the first up-crossing of the mean: THE R_v), delta08 and the
compensation ridge; their settings record and cache check; and the distinct voids (distinct_voids)."""

from __future__ import annotations

import numpy as np

import config


R_V_STEP_SIGMA = 0.2      # radial step of the R_v search, in smoothing lengths (the crossing is interpolated)
R_V_MAX_SIGMA = 24.0      # how far out it looks (capped at half the box)
# R_V_MAX_SIGMA = 24: at 15 sigma 17-25% of the underdense minima have no crossing within reach, so the R_v tail
# is cut, not measured (fitted on TNG100-3-Dark and TNG50-3-Dark, z = 0, 1, 3, 5, 10-cell smoothing). 24 is the
# largest whole cap a sigma short of L/2 at 10 cells (512^3). The half-box clause keeps a sigma's margin
# (r_v_reach_cells: min(cap sigma, L/2 - sigma)); the catalogue records the reach it searched (r_v_settings'
# sixth element), so a cache without one is re-measured only where the margin binds (catalog_r_v_ok).
R_V_DIRS = 128            # Fibonacci directions per shell
R_V_DELTA08 = -0.8        # the enclosed contrast of the 'delta08' radius: shell crossing of the spherical model
                          # (nonlinear -0.8 <-> linear -2.71; Sheth & van de Weygaert 2004)
# The R_v ALGORITHM, bumped by hand when the measurement changes (a cache records it and is re-measured when it
# differs): 1 = the first up-crossing alone; 2 = the full shell matrix out to the search range (crossing bit-identical
# to 1, plus delta08 and ridge).
R_V_ALGORITHM = 2


def r_v_settings() -> np.ndarray:
    """What a catalogue's radii were measured with ('r_v_settings'): [step in sigma, reach in sigma, directions,
    algorithm, delta08 threshold]. Read at call time. A catalogue's record carries a sixth element, the reach in
    cells it searched (r_v_record)."""
    return np.array([R_V_STEP_SIGMA, R_V_MAX_SIGMA, R_V_DIRS, R_V_ALGORITHM, R_V_DELTA08], dtype=np.float64)


def r_v_reach_cells(sigma_cells: float, n: int, half_box_margin: bool = True) -> float:
    """How far out the R_v search looks, in cells: min(R_V_MAX_SIGMA sigma, L/2 - sigma) on an n^3 grid (sigma
    floored at a cell, as the step is), never below 0. half_box_margin=False is the margin-free rule,
    min(R_V_MAX_SIGMA sigma, L/2): what a record without a reach was searched with (catalog_r_v_ok)."""
    s = max(float(sigma_cells), 1.0)
    return max(min(R_V_MAX_SIGMA * s, 0.5 * float(n) - (s if half_box_margin else 0.0)), 0.0)


def r_v_record(sigma_cells: float, n: int) -> np.ndarray:
    """The record a newly measured catalogue carries: r_v_settings() plus the reach in cells it searched
    (r_v_reach_cells for its smoothing and grid)."""
    return np.append(r_v_settings(), r_v_reach_cells(sigma_cells, n))


def catalog_r_v_ok(cat: dict, sigma_cells: float | None = None, n: int | None = None) -> bool:
    """Whether a catalogue's radii were measured with the current settings (r_v_settings) out to the current reach
    (r_v_reach_cells) and it holds all three definitions. A cache without a record is re-measured once; a record
    of the five settings alone is ok when the margin-free half-box rule reaches as far as the current one."""
    if 'r_v_settings' not in cat or any(k not in cat for k in ('r_v_cells', 'r_v_delta08_cells', 'r_v_ridge_cells')):
        return False
    have = np.asarray(cat['r_v_settings'], dtype=np.float64).reshape(-1)
    want = r_v_settings()
    if have.shape[0] not in (want.shape[0], want.shape[0] + 1) or not np.array_equal(have[:want.shape[0]], want):
        return False
    sigma_cells = config.SMOOTHING_SIGMA_CELLS if sigma_cells is None else sigma_cells
    n = config.FIELD_RESOLUTION if n is None else n
    reach = r_v_reach_cells(sigma_cells, n)
    if have.shape[0] == want.shape[0]:
        return r_v_reach_cells(sigma_cells, n, half_box_margin=False) == reach
    return float(have[-1]) == reach


def r_v_stale_reason(cat: dict, sigma_cells: float | None = None, n: int | None = None) -> str | None:
    """Why catalog_r_v_ok says no, in a few words for the log (None when it says yes)."""
    if catalog_r_v_ok(cat, sigma_cells, n):
        return None
    if 'r_v_settings' not in cat:
        return "has no record of its R_v settings"
    have = np.asarray(cat['r_v_settings'], dtype=np.float64).reshape(-1)
    want = r_v_settings()
    if have.shape[0] in (want.shape[0], want.shape[0] + 1) and np.array_equal(have[:want.shape[0]], want) \
            and all(k in cat for k in ('r_v_cells', 'r_v_delta08_cells', 'r_v_ridge_cells')):
        sigma_cells = config.SMOOTHING_SIGMA_CELLS if sigma_cells is None else sigma_cells
        n = config.FIELD_RESOLUTION if n is None else n
        old = float(have[-1]) if have.shape[0] > want.shape[0] else r_v_reach_cells(sigma_cells, n, False)
        return (f"was searched out to {old:.4g} cells, today's reach is {r_v_reach_cells(sigma_cells, n):.4g} "
                f"(L/2 less a sigma)")
    if have.shape[0] in (want.shape[0], want.shape[0] + 1) and np.array_equal(have[:want.shape[0]], want):
        return "lacks one of the three radii"
    return f"was measured with R_v settings {have[:want.shape[0]].tolist()}, not today's {want.tolist()}"


def _enclosed_contrast(radii, shells):
    """Delta(<r) at every radius of 'radii' (r_0 = 0 first) from the shell means 'shells' (rows = voids, column 0 =
    the value at r = 0): delta piecewise LINEAR in r between the shells and r^2 dr integrated exactly over each
    piece, so a constant profile gives Delta = that constant."""
    a, b = radii[:-1], radii[1:]
    va, vb = shells[:, :-1], shells[:, 1:]
    d3 = (b ** 3 - a ** 3) / 3.0
    slope = (vb - va) / (b - a)
    piece = va * d3 + slope * ((b ** 4 - a ** 4) / 4.0 - a * d3)          # integral of delta(r) r^2 over [a, b]
    out = np.empty_like(shells)
    out[:, 0] = shells[:, 0]
    out[:, 1:] = 3.0 * np.cumsum(piece, axis=1) / radii[1:] ** 3
    return out


def void_radii_cells(delta_s: np.ndarray, coords, sigma_cells: float) -> dict[str, np.ndarray]:
    """The three measured radii per minimum, in cells, from ONE matrix of shell means of the SMOOTHED delta:
    'r_v_cells' the first up-crossing of 0, 'r_v_delta08_cells' where Delta(<R) first reaches R_V_DELTA08,
    'r_v_ridge_cells' the maximum of the shell mean (NaN at the outermost shell). NaN for a non-underdense minimum."""
    from .profiles import fibonacci_sphere, shell_means          # here: profiles imports catalog, which imports this module
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    out = {k: np.full(len(coords), np.nan) for k in ('r_v_cells', 'r_v_delta08_cells', 'r_v_ridge_cells')}
    if not len(coords):
        return out
    n = delta_s.shape[0]
    step = R_V_STEP_SIGMA * max(float(sigma_cells), 1.0)
    radii = np.arange(1, int(r_v_reach_cells(sigma_cells, n) / step) + 1) * step   # L/2 less a sigma
    v0 = np.asarray(delta_s[coords[:, 0], coords[:, 1], coords[:, 2]], dtype=np.float64)
    todo = np.nonzero(v0 < 0)[0]                                # an overdense 'minimum' has no underdense region
    if not todo.size or not radii.size:
        return out
    dirs = fibonacci_sphere(R_V_DIRS)
    k = radii.size
    centers = np.repeat((coords[todo] + 0.5) / n, k, axis=0)   # (void, shell) pairs, void-major
    shells = np.empty((todo.size, k + 1))
    shells[:, 0] = v0[todo]
    shells[:, 1:] = shell_means(delta_s, centers, np.tile(radii, todo.size), dirs).reshape(todo.size, k)
    r_all = np.concatenate([[0.0], radii])

    def first_up(prof, level):
        # the first column >= level after column 0 (which is below it), interpolated from the column before
        hit = prof[:, 1:] >= level
        rows = np.nonzero(hit.any(axis=1) & (prof[:, 0] < level))[0]
        j = np.argmax(hit[rows], axis=1) + 1
        pr, pv, v = r_all[j - 1], prof[rows, j - 1], prof[rows, j]
        res = np.full(len(prof), np.nan)
        res[rows] = pr + (r_all[j] - pr) * (level - pv) / (v - pv)
        return res

    out['r_v_cells'][todo] = first_up(shells, 0.0)
    out['r_v_delta08_cells'][todo] = first_up(_enclosed_contrast(r_all, shells), R_V_DELTA08)
    j = np.argmax(shells[:, 1:], axis=1) + 1                    # the ridge: a shell, never the minimum itself
    inner = j < k                                               # the outermost shell is no ridge (truncated)
    jj, rows = j[inner], np.nonzero(inner)[0]
    f0, f1, f2 = shells[rows, jj - 1], shells[rows, jj], shells[rows, jj + 1]
    curv = f0 - 2.0 * f1 + f2
    off = np.where(curv < 0, 0.5 * (f0 - f2) / np.where(curv < 0, curv, -1.0), 0.0)
    ridge = np.full(todo.size, np.nan)
    ridge[rows] = r_all[jj] + np.clip(off, -0.5, 0.5) * step
    out['r_v_ridge_cells'][todo] = ridge
    return out


def void_radius_cells(delta_s: np.ndarray, coords, sigma_cells: float) -> np.ndarray:
    """R_v per minimum, in cells (NaN when there is none): the radius at which the spherically averaged SMOOTHED
    density contrast first rises to 0, linearly interpolated from the last negative shell (trilinear shell means,
    R_V_DIRS directions). NaN for a minimum that is not underdense or never reaches 0. The 'crossing' of void_radii_cells."""
    return void_radii_cells(delta_s, coords, sigma_cells)['r_v_cells']


def distinct_voids(coords, delta_values, r_v_cells, n_grid: int, r_eff_cells=None) -> np.ndarray:
    """Which of the given voids are DISTINCT: deepest first, a void whose centre lies inside the region a deeper void
    already kept claims is a sub-minimum of the same region (periodic, cells). With r_eff_cells each void CLAIMS the
    sphere of min(R_v, R_eff); without it the R_v sphere alone. Meant for the WELL-RESOLVED voids."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    r_v = np.asarray(r_v_cells, dtype=np.float64).reshape(-1)
    if r_eff_cells is not None:
        r_v = np.fmin(r_v, np.asarray(r_eff_cells, dtype=np.float64).reshape(-1))   # fmin: a NaN side is ignored
    keep = np.zeros(len(coords), dtype=bool)
    kept_c, kept_r = np.zeros((0, 3)), np.zeros(0)
    for i in np.argsort(np.asarray(delta_values, dtype=np.float64), kind="stable"):
        if not np.isfinite(r_v[i]):
            continue
        if kept_r.size:
            d = coords[i] - kept_c
            d -= n_grid * np.rint(d / n_grid)
            if (np.sqrt((d * d).sum(axis=1)) < kept_r).any():
                continue
        keep[i] = True
        kept_c, kept_r = np.vstack([kept_c, coords[i]]), np.append(kept_r, r_v[i])
    return keep
