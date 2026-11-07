"""The measured void radius R_v and its two companions, from ONE matrix of shell means of the smoothed delta about
each minimum (void_radii_cells): the crossing (the first up-crossing of the mean: THE R_v), delta08 (where the
enclosed contrast reaches -0.8) and the compensation ridge; the record a catalogue keeps of the settings they were
measured with (r_v_settings, r_v_record) and the check a cache is read against (catalog_r_v_ok); the distinct voids
(distinct_voids). Moved out of dtfelib.pipeline, which re-exports every name: setting one of the R_V_* knobs on
pipeline (pipeline.R_V_MAX_SIGMA = ...) sets it here too."""

from __future__ import annotations

import numpy as np

import config


R_V_STEP_SIGMA = 0.2      # radial step of the R_v search, in smoothing lengths (the crossing is interpolated)
R_V_MAX_SIGMA = 24.0      # how far out it looks (capped at half the box)
# Why 24 (15 = 32.4 Mpc on TNG100): at 15 sigma 17-25% of the underdense minima of TNG100-3-Dark and
# TNG50-3-Dark (z = 0, 1, 3, 5; 10-cell smoothing) have NO crossing within reach (NaN, 88-104 per catalogue) or cross
# in its last sigma, 8-17% of the well-resolved ones -- the R_v tail is cut, not measured. Same count at 20 / 24 / 25
# sigma / L/2: 5-8% / 1.1-2.6% / 0.6-1.8% / 0.4-1.2% (well-resolved 4-8% / 0-2.9% / 0-2% / 0-2%). Half the 512^3 box is
# 25.6 sigma at 10 cells; 24 is the largest whole cap a sigma short of it (shells beyond L/2 meet the centre's periodic
# images; within the last sigma the smoothed field already feels them). The distinct-void median crossing R_v
# (Mpc, 15 -> 24 sigma) at z = 0 / 1 / 3 / 5: TNG100 6.94 -> 7.42 / 7.59 -> 7.85 / 6.82 -> 7.64 / 6.27 -> 6.79,
# TNG50 2.82 -> 2.88 / 3.41 -> 3.98 / 3.68 -> 3.97 / 3.13 -> 3.73, and 6-13% more distinct voids (TNG100 z = 0: 77 ->
# 83) as fewer deep voids lack a radius. It is in r_v_settings, so a catalogue measured otherwise is re-measured once
# (~15 s and 2.4 GB per 512^3 snapshot: reading and smoothing the grid; the shells themselves 1-2 s). The half-box
# clause has a sigma's margin of its own (r_v_reach_cells: min(cap sigma, L/2 - sigma); min(cap sigma, L/2) would let
# a large-sigma search run right up to the centre's periodic images: TNG50 at 2.1625 Mpc = 21.4 cells would reach
# exactly 256 cells, not 234.6). The margin changes nothing while cap sigma <= L/2 - sigma, i.e. sigma <=
# 10.24 cells at 512^3 (the production 10 included), so the catalogue records the reach it searched (r_v_settings'
# sixth element) instead of a new algorithm number: a cache without one is served when the current reach equals the
# margin-free rule's and re-measured only where the margin binds (catalog_r_v_ok).
R_V_DIRS = 128            # Fibonacci directions per shell
R_V_DELTA08 = -0.8        # the enclosed contrast of the 'delta08' radius: shell crossing of the spherical model
                          # (nonlinear -0.8 <-> linear -2.71; Sheth & van de Weygaert 2004)
# The R_v ALGORITHM, bumped by hand when the measurement changes (a cache records it, r_v_settings, and is re-measured
# when it differs): 1 = the first up-crossing alone, the loop stopping at it; 2 = the
# full shell matrix out to the search range, giving the crossing (bit-identical to 1), delta08 and ridge.
R_V_ALGORITHM = 2


def r_v_settings() -> np.ndarray:
    """What a catalogue's radii were measured with, as it records them ('r_v_settings'): [step in sigma, reach in
    sigma, directions, algorithm, delta08 threshold]. Read at call time, so a changed module constant counts. A
    catalogue's record carries a sixth element, the reach in cells it searched
    (r_v_record); these five are the settings every other reader keys on (exports, the fate links)."""
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
    (r_v_reach_cells for sigma_cells / n: config's smoothing and grid by default, the namespace the catalogue lives
    in) and it holds all three (catalog.R_V_DEFINITIONS). A cache without a record: not ok,
    re-measured once (else a changed R_V_MAX_SIGMA or algorithm would leave every cache with stale
    radii and NaNs). A record of the five settings alone (no reach) is ok
    when the margin-free half-box rule reaches as far as the current one does -- every sigma <= 10.24 cells at 512^3,
    the production
    smoothing included -- so the margin re-measures only the catalogues it changes."""
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
    the value at r = 0): delta taken piecewise LINEAR in r between the shells and r^2 dr integrated exactly over each
    piece, so a constant profile gives Delta = that constant (a trapezoid on r^2 delta gives 1.5x it at the first
    shell). Column 0 is the central value itself."""
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
    """The three measured radii per minimum, in cells, from ONE matrix of shell means of the SMOOTHED delta
    (trilinear, R_V_DIRS directions about the minimum's cell centre, every R_V_STEP_SIGMA sigma out to R_V_MAX_SIGMA
    sigma or half the box less a sigma, r_v_reach_cells; column 0 the minimum itself at r = 0)
    (catalog.R_V_DEFINITIONS):
      'r_v_cells'          the first up-crossing of 0 (the mean), linearly interpolated from the last negative shell
                           -- void_radius_cells, bit for bit;
      'r_v_delta08_cells'  where the enclosed contrast Delta(<R) (_enclosed_contrast) first reaches R_V_DELTA08 going
                           outward, interpolated likewise; NaN when the minimum itself is not below it (no sphere
                           about it is that empty: on a field smoothed at sigma this is the common case for shallow
                           minima) or it is not reached within reach;
      'r_v_ridge_cells'    the radius of the MAXIMUM of the shell mean over the search range (the compensation ridge),
                           refined by a parabola through it and its neighbours; NaN when the maximum is the outermost
                           shell (the profile still rising: the range truncated it, no ridge was seen).
    All three NaN for a minimum that is not underdense (delta_s >= 0). The question behind delta08 and the ridge:
    is the z < 3 shrinkage of the crossing R_v a property of the voids or the first-crossing rule latching onto an
    inner wall? A void whose crossing R_v falls while delta08 and the ridge hold is the latter."""
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
    density contrast around the minimum first rises to 0 -- the underdense region the void is, MEASURED. The
    ellipsoid's 2 / sqrt|lambda| is where a QUADRATIC rises by a fixed 2 whatever the void's depth, ~1.6x the radius
    where the stacked density reaches the mean, v_r turns to infall and the single-stream fraction bottoms out
    (TNG100 z = 0). Shell means of delta_s (trilinear, R_V_DIRS directions, centred on the minimum's
    cell centre) every R_V_STEP_SIGMA sigma out to R_V_MAX_SIGMA sigma or half the box; the crossing linearly
    interpolated from the last negative shell (from r = 0: the minimum itself). Defined at every redshift (delta_s
    < 0 at a void's minimum); NaN for a minimum that is not underdense or never reaches 0 within reach. The
    'crossing' of void_radii_cells (one shell matrix gives all three definitions)."""
    return void_radii_cells(delta_s, coords, sigma_cells)['r_v_cells']


def distinct_voids(coords, delta_values, r_v_cells, n_grid: int, r_eff_cells=None) -> np.ndarray:
    """Which of the given voids are DISTINCT: deepest first, a void whose centre lies inside the region a deeper void
    already kept claims is a sub-minimum of the same underdense region (periodic distances, cells). With r_eff_cells
    (catalog.distinct_mask always passes it) each void CLAIMS the sphere of min(R_v, R_eff) (a missing one ignored),
    so a void without R_v still takes part with its R_eff, and the tail of the deepest voids, whose R_v reaches 1.2-3
    R_eff (84th percentile R_v/R_eff 1.2-1.5), does not swallow 20-27% of the sample (R_v spheres alone, TNG300 z=0:
    11 voids with R_v > 10 sigma drop 99 of 365). Without it the claim is the R_v sphere alone, and a void without
    R_v is not distinct and claims no region. Meant for the WELL-RESOLVED voids (catalog.distinct_mask runs it over
    all of them, then narrows the result to the caller's sample):
    the smoothed field's minima within an 11-cell footprint often share one void -- 63% of the TNG100 z=0
    resolved voids have another centre within R_eff -- while over ALL minima a few deep ones that fail the cut,
    with R_v up to 14 sigma, would swallow three quarters of the resolved sample (24 of 103 left)."""
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
