"""Contour lines of a map: marching squares in numpy (no matplotlib, no scipy), the levels a map's colour scale
gives and the colour each line is drawn in. Qt-free: the Explore map strokes the pieces with QPainter, a saved figure
with matplotlib, the 3D view through each plane's own transform.

  segments(values, level)         one level's line pieces, (k, 2, 2) in ARRAY coordinates: point (i, j) is the
                                  sample values[i, j]; each piece runs between two crossings on the grid's edges
  plane_segments(values, levels)  every level's pieces, from a block-averaged copy when the plane is wider than
                                  MAX_SIDE (a fine grid stays fluent to scroll), in the full plane's coordinates
  levels(field, vrange, log)      the values the lines are drawn at, in the colour bar's units (log10 on a log map)
  colours(levels, vrange, cmap)   per level black or white, whichever shows on the map's colour at that value
  map_lines(plane, ...)           all of it for one map: the levels, colours and pieces (None: no contours there)
  shell_plan(field, levels, ...)  the 3D view's volume: a few of those levels as thin shells of constant value where
                                  the volume is clear (the density: around the voids' minima), for its transfer
                                  function (volume3d.transfer's 'shells')

The levels follow the owner's figure scripts (dtfelib.figures.delta_contour_levels: 20 levels evenly spaced in the
colour scale's own space; for the density three quarters of them on the void side of the mean, and the mean itself),
on ROUND steps anchored at the mean (0 elsewhere): the map's colour range is the slice's own percentiles unless it is
fixed, so it moves from layer to layer. The steps are 1 and 5 x 10^k (STEPS), each a whole multiple of every finer
one, so a line drawn on two layers is the same value on both: while the range drifts a line appears or goes at the
ends, and when it changes enough for another step the coarser layer's lines are among the finer one's. The level
slider (its position 'detail') multiplies every step by one factor of its own (factor(): 1 at its default), never by
anything of the layer's, so at any position the levels keep that nesting: a line keeps its value from layer to layer.
"""

from __future__ import annotations

import numpy as np

import grids as G

N_LEVELS = 20               # the script's num_contours: the colour range in about this many steps
VOID_SHARE = 0.75           # the density: this share of the steps below the mean (delta_contour_levels' 3/4)
N_OTHER = 12                # any other continuous map: its range in about this many steps
# the round steps, x 10^k: each divides the next (1 | 5 | 10), so the levels of a coarser step are levels of every
# finer one (with 2 and 2.5 the steps 0.02 and 0.025 of two layers would share only the multiples of 0.1)
STEPS = (1.0, 5.0, 10.0)
# the Map group's level slider: a position 'detail' multiplies the rule's steps by factor(), the same on every layer
# of the field: 1 at DETAIL = 0 (the rule's own levels, bit for bit), COARSEST times wider at DETAIL_MIN, FINEST at
# DETAIL_MAX, and the lines per unit of the colour scale (1 / factor) in even steps in between, so each position right
# spaces the lines a little closer (a slider of round steps had four positions, one of counts would draw the same
# lines over most of its travel). The ends hold on ANY range, as _nice_step's step is at most sqrt(5) times its
# share of the range: the coarsest draws two lines or more (the density: a step of up to 4.47 times the rule's leaves
# a line besides the mean, or two where the range misses the mean; elsewhere up to 2.68 times leaves two, a diverging
# map its 0 and one each side) ...
DETAIL, DETAIL_MIN, DETAIL_MAX = 0, -36, 36
COARSEST = {"density": 4.4, "": 2.6}
# ... and the finest at most 1 / FINEST times the default's lines: the reach of the old finest stop (a rung of 2 or 5
# finer) on a smoothed slice, where the lines are worth reading (TNG100-3-Dark 512^2 at 4 cells: 16 lines at the
# default, 40 at the finest; the old stop drew 39). The slice worker's time is about linear in the levels, so the
# finest costs a layer up to 1 / FINEST times the default, on the worker, never on the window's thread
FINEST = 0.4
OLD_STOPS = (-2, 1)         # the level slider's stops before it was continuous (rungs of the ladder: saved states)
MAX_STREAM_LINES = 12       # the stream count: lines at the even counts, thinned to about this many
# a plane wider than this is contoured on a block-averaged copy (whole blocks, the last one shorter): a 512^2 slice
# shown about 700 points wide has cells of 1.4 points, so lines from 2 x 2 blocks lose nothing the eye can follow,
# while a rough unsmoothed slice gives four times fewer pieces to trace and stroke (on the slice worker, which then
# serves the next layer sooner); a 1024^2 slice or a 4096^2 hi-res plane then costs what a 256^2 one does
MAX_SIDE = 256
# maps whose colours are classes or flags, not a scale: no contours (the stream count is a count: see levels())
CLASS_FIELDS = frozenset({"hidden_streams", "caustic", "caustic_class", "tweb", "vweb"})
DISCRETE_CMAPS = frozenset({"tab10"})
ALPHA = 0.7                 # the lines' opacity (his figures: black at 70 %); one point wide on screen
LINE_WIDTH_PT = 0.6         # a saved figure's: his figures' linewidths=0.6
LIGHT = 0.5                 # a colour of this luminance or more takes a black line, a darker one a white line

# the pieces of each marching-squares case: corners a = (i, j), b = (i+1, j), c = (i+1, j+1), d = (i, j+1), the case
# a + 2b + 4c + 8d of the corners at or above the level; edges 0 = a-b, 1 = b-c, 2 = c-d, 3 = d-a. The saddles 5 and
# 10 take one of two pairings by the cell's centre (the mean of its corners): PAIR_X cuts off b and d, PAIR_Y a and c
_FIRST = {1: (3, 0), 2: (0, 1), 3: (3, 1), 4: (1, 2), 6: (0, 2), 7: (3, 2), 8: (2, 3), 9: (0, 2), 11: (1, 2),
          12: (1, 3), 13: (0, 1), 14: (3, 0)}
_PAIR_X = ((0, 1), (2, 3))
_PAIR_Y = ((3, 0), (1, 2))


def why_not(field: str, cmap: str = "", compare: bool = False) -> str:
    """'' when a map of 'field' in colour map 'cmap' can carry contours, else the reason (the checkbox's tooltip): a
    class or flag map, or a colour map of discrete colours. A difference map ('compare') of a class field is a class
    map too."""
    if field in CLASS_FIELDS:
        return "No contours on a class map (T-web or V-web class, caustic flags): its colours are classes"
    if cmap in DISCRETE_CMAPS:
        return f"No contours with a colour map of discrete colours ({cmap}): choose a continuous one"
    return ""


def _nice_step(raw: float) -> float:
    """The round step (STEPS x 10^k: 1 or 5) nearest 'raw' by log distance."""
    if not (np.isfinite(raw) and raw > 0):
        return 0.0
    mag = 10.0 ** np.floor(np.log10(raw))
    return float(min((m * mag for m in STEPS), key=lambda st: abs(np.log(st / raw))))


def factor(detail, field: str = "", compare: bool = False) -> float:
    """The level slider's factor on every step of a map of 'field' at position 'detail' (clamped to DETAIL_MIN..
    DETAIL_MAX): exactly 1 at DETAIL = 0, COARSEST[the density, else ''] at DETAIL_MIN, FINEST at DETAIL_MAX, its
    inverse (the lines per unit of the colour scale) linear in the position on either side."""
    detail = int(np.clip(int(detail), DETAIL_MIN, DETAIL_MAX))
    if detail == DETAIL:
        return 1.0
    end = COARSEST["density" if field == "density" and not compare else ""] if detail < 0 else FINEST
    t = detail / float(DETAIL_MIN if detail < 0 else DETAIL_MAX)
    return 1.0 / (1.0 + t * (1.0 / end - 1.0))


def from_stop(stop, field: str = "density") -> int:
    """The position nearest an old saved stop of the level slider (OLD_STOPS: rungs of the ladder, each 2 or 5 times
    finer or coarser, so about sqrt(10) a rung; -2 exactly 10 times coarser), clamped to the slider's travel."""
    old = 10.0 ** (-float(stop) / 2.0)                  # the old stop's factor on the rule's step
    pos = range(DETAIL_MIN, DETAIL_MAX + 1)
    return min(pos, key=lambda p: (abs(np.log(factor(p, field) / old)), abs(p)))


def _digits(step: float) -> int:
    """Decimals that hold every multiple of a round step exactly (0.025 -> 3): the levels are rounded to them, so a
    level is the same float on every layer however it was reached."""
    return max(0, 3 - int(np.floor(np.log10(step)))) if step > 0 else 6


def _level(round_multiple: float, g: float, anchor: float = 0.0) -> float:
    """anchor + a round multiple of a round step (rounded to _digits(step) already) times the slider's factor 'g':
    the same float on every layer however it was reached (the same multiple, the same factor; the last bits of the
    sum dropped at 12 significant digits, the value alone deciding them), at g = 1 the round value itself."""
    return float(f"{anchor + round_multiple * g:.12g}") + 0.0


def _ladder(anchor: float, step: float, lo: float, hi: float, side: int, g: float = 1.0) -> list[float]:
    """anchor + side * k * step * g for k = 1, 2, ... while strictly inside (lo, hi), 'anchor' inside it ('step' a
    round step, 'g' the slider's factor); a range end itself is not a level (the clipped colour's edge, not a value
    of the field)."""
    if step <= 0:
        return []
    eps = 1e-9 * (hi - lo)
    n = int(np.floor(((anchor - lo) if side < 0 else (hi - anchor)) / (step * g))) + 1
    d = _digits(step)
    vals = (_level(side * round(k * step, d), g, anchor) for k in range(1, min(n, 100000) + 1))
    return [v for v in vals if lo + eps < v < hi - eps]


def level_plan(field: str, vrange, log: bool = False, compare: bool = False, detail: int = DETAIL) -> dict:
    """The contour levels of a map and how they were chosen: {'levels': array in the colour bar's units (log10 on a
    log map), 'mean': the mean's level or None, 'below' / 'above': the spacings below and above it (the density),
    'step': the one spacing elsewhere, 'factor': the level slider's (factor())}. 'vrange' is the map's shown colour
    range (the one colorize returned, in the bar's units).

    The density (rho/rho_bar, not a difference map) with its mean (1; 0 on a log map) inside the range: the script's
    rule -- about three quarters of the levels below the mean, a round step (mean - lo) / 15 apart, the mean itself,
    and above it a round step (hi - mean) / 5 apart but never closer than below (a slice near the mean is mostly noise
    there). The stream count: the even counts inside the range (each line the edge between n - 1 and n + 1 streams,
    '2' the edge of the single-stream regions). Anything else: the multiples of a round step of about (hi - lo) /
    N_OTHER from 0 (a diverging map has its 0); the density with its mean outside the range: of about (hi - lo) /
    N_LEVELS from the mean.

    'detail' (the level slider, DETAIL_MIN..DETAIL_MAX): every one of those steps times factor(detail, field), about
    the same anchor (the mean, else 0), so the density keeps its share below the mean and the mean itself; the stream
    count thins its even counts to about MAX_STREAM_LINES / factor."""
    detail = int(detail)
    g = factor(detail, field, compare)
    plan = {"levels": np.zeros(0), "mean": None, "below": 0.0, "above": 0.0, "step": 0.0, "factor": g}
    try:
        lo, hi = float(vrange[0]), float(vrange[1])
    except (TypeError, ValueError, IndexError):
        return plan
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return plan
    if field == "streams" and not compare:
        top = 10.0 ** hi if log else hi
        bot = 10.0 ** lo if log else lo
        evens = np.arange(2.0, np.floor(top) + 1.0, 2.0)
        evens = evens[(evens > bot) & (evens < top)]
        stride = max(1, int(np.ceil(evens.size * g / MAX_STREAM_LINES)))
        evens = evens[::stride]
        plan["levels"] = np.log10(evens) if log else evens
        plan["step"] = 2.0 * stride
        return plan
    mean = (0.0 if log else 1.0) if field == "density" and not compare else None
    if mean is not None and lo < mean < hi:
        n_void = int(N_LEVELS * VOID_SHARE)
        rule = _nice_step((mean - lo) / n_void)
        rule_above = _nice_step(max((hi - mean) / (N_LEVELS - n_void), rule))
        levs = _ladder(mean, rule, lo, hi, -1, g)[::-1] + [mean] + _ladder(mean, rule_above, lo, hi, +1, g)
        plan.update(levels=np.asarray(levs, dtype=np.float64), mean=mean, below=rule * g, above=rule_above * g)
        return plan
    step = _nice_step((hi - lo) / (N_LEVELS if mean is not None else N_OTHER))
    if step <= 0:
        return plan
    anchor = 0.0 if mean is None else mean              # the density's lines stay on its mean's ladder out here too
    s, d, eps = step * g, _digits(step), 1e-9 * (hi - lo)
    k0, k1 = int(np.ceil((lo - anchor) / s)), int(np.floor((hi - anchor) / s))
    levs = [_level(round(k * step, d), g, anchor) for k in range(k0, min(k1, k0 + 100000) + 1)]
    plan.update(levels=np.asarray([v for v in levs if lo + eps < v < hi - eps], dtype=np.float64), step=s)
    return plan


def levels(field: str, vrange, log: bool = False, compare: bool = False, detail: int = DETAIL) -> np.ndarray:
    """The contour levels of a map of 'field' on the shown colour range 'vrange' (level_plan's rule, at the level
    slider's 'detail'), in the colour bar's units: log10 values on a log map. Empty when the range is empty or not
    finite."""
    return level_plan(field, vrange, log, compare, detail)["levels"]


def colours(levs, vrange, cmap: str, alpha: float = ALPHA) -> np.ndarray:
    """(k, 4) RGBA in 0..1 per level: black where the map's colour at that value is light (luminance >= LIGHT), else
    white -- a line runs where the map has exactly its level's colour, so it shows on every colour map (magma's dark
    voids take white lines, RdBu_r's pale middle black ones); 'alpha' the opacity."""
    levs = np.asarray(levs, dtype=np.float64).reshape(-1)
    lo, hi = float(vrange[0]), float(vrange[1])
    span = hi - lo if hi > lo else 1.0
    idx = np.clip(np.round((levs - lo) / span * 255.0), 0, 255).astype(int)
    rgb = G.lut(cmap)[idx].astype(np.float64) / 255.0
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    out = np.zeros((levs.size, 4))
    out[:, :3] = np.where(lum[:, None] >= LIGHT, 0.0, 1.0)
    out[:, 3] = alpha
    return out


def _edge_points(v: np.ndarray, level: float, edge: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """The crossing of 'level' on edge 'edge' (0..3, per cell) of cells (I, J), (k, 2) in array coordinates. Each grid
    edge is interpolated from its lower-index end whichever cell asks, so the two cells that share an edge get the
    same point to the last bit (the pieces of a line meet exactly)."""
    # edge 0 = the edge along i at (I, J); 2 = along i at (I, J + 1); 3 = along j at (I, J); 1 = along j at (I + 1, J)
    along_i = (edge == 0) | (edge == 2)
    pi = I + (edge == 1)
    pj = J + (edge == 2)
    qi = pi + along_i
    qj = pj + ~along_i
    vp, vq = v[pi, pj], v[qi, qj]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (level - vp) / (vq - vp)
    t = np.where(np.isfinite(t), np.clip(t, 0.0, 1.0), 0.5)
    return np.stack([pi + t * (qi - pi), pj + t * (qj - pj)], axis=-1)


def segments(values, level: float) -> np.ndarray:
    """Marching squares: the pieces of the line where 'values' (a 2D array) equals 'level', as a (k, 2, 2) float
    array -- piece, end, (i, j) in array coordinates (the point (i, j) is the sample values[i, j]; a piece's ends lie
    on the edges between neighbouring samples, linearly interpolated). A sample counts as above when >= level; a
    saddle cell is split by the mean of its four corners; a cell with a non-finite corner (a ratio map's grey) has no
    piece. Empty (0, 2, 2) when the level is outside the values or the array is smaller than 2 x 2."""
    v = np.asarray(values, dtype=np.float64)
    empty = np.zeros((0, 2, 2))
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] < 2 or not np.isfinite(level):
        return empty
    L = float(level)
    fin = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        up = (v >= L) & fin
    case = (up[:-1, :-1].astype(np.uint8) + 2 * up[1:, :-1] + 4 * up[1:, 1:] + 8 * up[:-1, 1:])
    ok = fin[:-1, :-1] & fin[1:, :-1] & fin[1:, 1:] & fin[:-1, 1:]
    I, J = np.nonzero(ok & (case != 0) & (case != 15))
    if I.size == 0:
        return empty
    cs = case[I, J]
    ea, eb = np.zeros(I.size, np.int64), np.zeros(I.size, np.int64)
    for c, (e0, e1) in _FIRST.items():
        m = cs == c
        ea[m], eb[m] = e0, e1
    sad = (cs == 5) | (cs == 10)
    si, sj = I[sad], J[sad]
    second = None
    if si.size:
        centre_up = (v[si, sj] + v[si + 1, sj] + v[si + 1, sj + 1] + v[si, sj + 1]) / 4.0 >= L
        x = (cs[sad] == 5) == centre_up                 # 5 with the centre up, or 10 with it down: cut off b and d
        first = np.where(x[:, None], _PAIR_X[0], _PAIR_Y[0])
        sec = np.where(x[:, None], _PAIR_X[1], _PAIR_Y[1])
        ea[sad], eb[sad] = first[:, 0], first[:, 1]
        second = np.stack([_edge_points(v, L, sec[:, 0], si, sj), _edge_points(v, L, sec[:, 1], si, sj)], axis=1)
    out = np.stack([_edge_points(v, L, ea, I, J), _edge_points(v, L, eb, I, J)], axis=1)
    return out if second is None else np.concatenate([out, second])


def reduced(values, max_side: int = MAX_SIDE):
    """(copy, centres0, centres1): 'values' block-averaged to at most max_side samples a side (blocks of f x f, f the
    smallest whole factor that fits; the last block of an axis takes the rest), with the position of each block's
    centre in the full array's coordinates; the array itself and its own positions when it fits. A block with a
    non-finite sample is non-finite (no line through a ratio map's grey)."""
    v = np.asarray(values, dtype=np.float64)
    n0, n1 = v.shape
    f = max(1, int(np.ceil(max(n0, n1) / float(max_side))))
    if f == 1:
        return v, np.arange(n0, dtype=np.float64), np.arange(n1, dtype=np.float64)
    s0, s1 = np.arange(0, n0, f), np.arange(0, n1, f)
    sums = np.add.reduceat(np.add.reduceat(v, s0, axis=0), s1, axis=1)
    c0 = np.minimum(s0 + f, n0) - s0
    c1 = np.minimum(s1 + f, n1) - s1
    return (sums / np.outer(c0, c1), (s0 + np.minimum(s0 + f, n0) - 1) / 2.0,
            (s1 + np.minimum(s1 + f, n1) - 1) / 2.0)


def plane_segments(values, levs, max_side: int = MAX_SIDE) -> list[np.ndarray]:
    """segments() of every level in 'levs', from reduced(values, max_side), in the FULL array's coordinates (a block's
    centre where its sample sat): one (k, 2, 2) array per level, in the order of 'levs'."""
    small, c0, c1 = reduced(values, max_side)
    full = small.shape == np.shape(values)
    out = []
    for L in np.asarray(levs, dtype=np.float64).reshape(-1):
        s = segments(small, L)
        if not full and s.size:
            s = np.stack([np.interp(s[..., 0], np.arange(c0.size), c0),
                          np.interp(s[..., 1], np.arange(c1.size), c1)], axis=-1)
        out.append(s)
    return out


def scaled(plane, log: bool) -> np.ndarray:
    """The values a map's colours show (grids.colorize's: log10 of the positive ones, the rest at the lowest), so the
    lines are contours of what the colours say."""
    return G._scaled(plane, log)


def map_lines(plane, field: str, vrange, log: bool = False, cmap: str = "viridis", compare: bool = False,
              max_side: int = MAX_SIDE, detail: int = DETAIL) -> dict | None:
    """The contour lines of one map: {'levels' (the bar's units), 'values' (the field's own values: 10^level on a log
    map), 'colours' (k, 4), 'segments' (one (m, 2, 2) per level, array coordinates of 'plane': point (i, j) is
    plane[i, j]), 'shape', 'plan' (level_plan)}; None where contours do not apply (why_not) or no level falls inside
    the range. 'plane' holds the field's values (not log10: 'log' scales them as the colours do); 'vrange' the colour
    range shown, in the bar's units; 'detail' the level slider's (level_plan)."""
    if why_not(field, cmap, compare):
        return None
    plan = level_plan(field, vrange, log, compare, detail)
    levs = plan["levels"]
    if levs.size == 0:
        return None
    segs = plane_segments(scaled(plane, log), levs, max_side)
    return {"levels": levs, "values": 10.0 ** levs if log else levs.copy(), "colours": colours(levs, vrange, cmap),
            "segments": segs, "shape": tuple(np.shape(plane)), "plan": plan, "log": bool(log), "field": field,
            "compare": bool(compare), "detail": int(detail)}


def clip_segments(segs, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """The parts of line pieces ((k, 2, 2): piece, end, (x, y)) inside the rectangle x0..x1, y0..y1 (Liang-Barsky,
    all pieces at once): a piece inside kept as it is, one crossing an edge cut there, one outside dropped. The 3D
    view's planes are drawn in quads (the parts between the other planes, painted in depth order): each quad strokes
    the part of its plane's lines on it."""
    s = np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)
    if not s.size:
        return np.zeros((0, 2, 2))
    p0, d = s[:, 0], s[:, 1] - s[:, 0]
    t0, t1 = np.zeros(len(s)), np.ones(len(s))
    keep = np.ones(len(s), dtype=bool)
    for k, lo_, hi_ in ((0, x0, x1), (1, y0, y1)):
        for p, q in ((-d[:, k], p0[:, k] - lo_), (d[:, k], hi_ - p0[:, k])):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = q / p
            par = p == 0
            keep &= ~(par & (q < 0))                    # parallel to this edge and outside it
            enter, leave = (p < 0) & ~par, (p > 0) & ~par
            t0 = np.where(enter, np.maximum(t0, r), t0)
            t1 = np.where(leave, np.minimum(t1, r), t1)
    keep &= t0 <= t1
    a, b = p0 + t0[:, None] * d, p0 + t1[:, None] * d
    return np.stack([a, b], axis=1)[keep]


def solid_colours(levs, vrange, cmap: str, alpha: float = ALPHA) -> np.ndarray:
    """(k, 3) RGB in 0..1 per level: colours()' line laid at 'alpha' over the map's own colour at that level -- what
    the map shows along its line, as an opaque colour. The 3D view's planes stroke their lines in it (an opaque pen:
    a translucent one darkens every point where two pieces of a line meet)."""
    levs = np.asarray(levs, dtype=np.float64).reshape(-1)
    lo, hi = float(vrange[0]), float(vrange[1])
    span = hi - lo if hi > lo else 1.0
    idx = np.clip(np.round((levs - lo) / span * 255.0), 0, 255).astype(int)
    under = G.lut(cmap)[idx].astype(np.float64) / 255.0
    return alpha * colours(levs, vrange, cmap)[:, :3] + (1.0 - alpha) * under


# ---------------------------------------------------------------------- the 3D view's volume: shells
# The volume draws the levels as thin shells of constant value through its transfer function (volume3d.transfer's
# 'shells': a narrow band of values around each level, in a colour of its own), which QPainter's slices and the GPU's
# shader read alike -- each colours a cell by its level and then blends the colours. Only where the volume is clear
# (below the shown range's low end, the density's voids): there a shell hides nothing, and the voids' minima are what
# the shells are for. The levels are the map's rule (level_plan) on the volume's own range, so a shell has the value
# of a map line of the same field whenever the map's step divides its.
#
# A band of values is a shell only where the field runs through it steeply enough: on a flat void floor near its value
# it fills a whole region (fog), where the field crosses it within less than a cell it breaks into dots. So each band
# is made about SHELL_CELLS cells thick from the field's slope at its level (volume3d.level_slopes), a shell the field
# crosses within less than a cell of the next one is left out, and each shell's opacity follows the share of cells
# its band holds (SHELL_INK): a small shell around a minimum is clear to see, a large one through many voids faint.
SHELL_FIELDS = frozenset({"density"})   # the fields with shells: the voids' minima are what they are for; another
                            # field's clear part is most of the box (the speed's slow 70 %), its shells a fog over it
NO_SHELLS = ("No contour shells in the volume for this field: they are drawn for the density, around its voids' "
             "minima; the planes show any field's lines")
N_SHELLS = 4                # at most this many shells: every k-th level of the rule, k the smallest that keeps so few
SHELL_ENCLOSED = 0.25       # the whole cube: each closing around at most this share of its cells (a level below which
                            # more lie runs through the box as a sheet between the voids: most of a cosmic web's
                            # volume is under the mean), not around their minima ...
SHELL_LOW_PCT = 0.5         # ... and above this percentile of the cells (the deepest few cells, not one empty one)
SHELL_CELLS = 1.5           # a band about this many cells thick where its shell runs, never narrower than a level ...
SHELL_WIDTH = 0.12          # ... nor reaching further than this share of the spacing between two shells to either
                            # side of its value (the share itself without the slopes)
SHELL_INK = 0.4             # a shell's opacity per cell x the cells in its band / the cube's cells across squared
                            # (the optical depth it adds to the picture, spread over it) ...
SHELL_ALPHA = (0.004, 0.3)  # ... within these (per cell, along the line of sight through it: the same look at any
                            # detail and in a close-up as in the whole box), the most for the deepest shell, half of
                            # it for the one nearest the shown range (the largest: the inner ones seen through it)
REF_CELLS = 128             # volume3d.REF_CELLS (this module cannot import volume3d, which imports it; the suite checks
                            # that the two agree): the reference cell the transfer function's opacity is per
SHELL_NEAR, SHELL_DEEP = (205, 225, 255), (60, 140, 255)   # the shells' colours from the one nearest the shown range
                            # (pale) to the deepest (a clear blue): light on the dark view and apart from the map's
                            # colours, the inspected void's amber outline and the voids' grey ones


def _percentile(levels, counts, pct: float) -> float:
    """The level value below which 'pct' percent of the counted cells lie (levels[0], no value, not counted)."""
    c = np.cumsum(np.asarray(counts[1:], dtype=np.float64))
    if not c.size or not c[-1] > 0:
        return float(levels[1])
    return float(levels[1 + min(int(np.searchsorted(c, pct / 100.0 * c[-1])), len(c) - 1)])


def band(shown, value: float, half: float) -> np.ndarray:
    """A shell's band: its weight on each level of a cube ('shown': the levels' values as the opacity reads them) --
    a tent of half-width 2 x half about 'value' (the mass of a band of +-half, with soft edges: no stair-steps where
    it crosses the cells), the level nearest 'value' always whole (a band narrower than a level is not lost); level 0
    (no value) none. volume3d.shell_weights draws it, shell_plan counts its cells."""
    with np.errstate(invalid="ignore"):
        w = np.clip(1.0 - np.abs(shown - value) / max(2.0 * half, 1e-30), 0.0, 1.0)
    w[~np.isfinite(w)] = 0.0
    near = int(np.nanargmin(np.where(np.isfinite(shown[1:]), np.abs(shown[1:] - value), np.inf))) + 1
    w[near] = 1.0
    w[0] = 0.0
    return w


def shell_plan(field: str, levels, counts, lo: float, log: bool = False, symmetric: bool = False,
               cmap: str = "", floor: float | None = None, slopes=None, cells: int | None = None,
               detail: int = DETAIL) -> dict | None:
    """The 3D volume's contour shells: {'levels' (the bar's units: log10 on a log map), 'values' (the field's own),
    'colours' ((k, 3) uint8, nearest the shown range first), 'half' (each band's half-width, the bar's units),
    'cover' (the share of the cube's cells in each band), 'alpha' (each one's opacity per cell), 'opacity' (the same
    per reference cell: volume3d.transfer's), 'spacing', 'step', 'plan', 'field', 'log', 'symmetric'}, or None (no
    shell, a field without shells: SHELL_FIELDS, or one contours do not apply to: why_not). From the prepared cube's
    level values and counts (volume3d.prepare's 'levels', 'counts'), the shown range's low end 'lo' (|value| when
    'symmetric', as the volume shows such a field), the field's slope per level ('slopes': volume3d.level_slopes) and
    the cube's cells along its longest side ('cells': a shell's opacity follows the cells in its band against the
    picture they spread over, SHELL_INK, and is converted to the reference cell's, volume3d.REF_CELLS across the
    window). 'counts' may be of a region only (a close-up's void): the bands' cells are counted there.

    The levels: the map's (level_plan) over the cube's own range; of them those where the volume is clear (below
    'lo'), for the density only its void side (below the mean), and above the cube's SHELL_LOW_PCT percentile closing
    around at most SHELL_ENCLOSED of its cells -- or, given a 'floor' (a close-up: the value at the bottom of the void
    looked at), every one above it, so that the shells nest around that void's own minimum however deep it is.
    Thinned to every k-th of the rule's steps from its anchor (the mean, else 0), at most N_SHELLS, evenly spaced: on
    a log density with steps of 0.1 and a floor at -1.25 the shells sit at -0.2, -0.4, -0.6, -0.8 (k = 2), rho/rho_bar
    = 0.63, 0.40, 0.25, 0.16. Then with the slopes, those the field crosses within less than a cell of the next shell
    are left out (dots, not a surface) and each band is made about SHELL_CELLS cells thick (within one level and
    SHELL_WIDTH of the spacing)."""
    if field not in SHELL_FIELDS or why_not(field, cmap):
        return None
    lv = np.asarray(levels, dtype=np.float64)
    cnt = np.asarray(counts, dtype=np.float64)
    if lv.size < 3 or not cnt[1:].sum() > 0:
        return None
    shown = np.abs(lv) if symmetric else lv            # the values the transfer function's opacity reads
    plan = level_plan(field, (float(lv[1]), float(lv[-1])), log, detail=detail)
    step = float(plan["below"] if plan["mean"] is not None else plan["step"])
    anchor = float(plan["mean"]) if plan["mean"] is not None else 0.0
    cand = plan["levels"]
    if symmetric:                                       # |value|: one shell for each of +L and -L
        cand = np.unique(np.abs(cand[cand != 0.0]))
    if plan["mean"] is not None:
        cand = cand[cand < plan["mean"]]
    if floor is None:                                   # the whole cube: around its deepest minima only
        order = np.argsort(shown[1:], kind="stable")
        cum = np.cumsum(cnt[1:][order]) / cnt[1:].sum()
        under = np.interp(cand, shown[1:][order], cum, left=0.0, right=1.0)    # the share of cells below each level
        cand = cand[(cand > _percentile(shown, cnt, SHELL_LOW_PCT)) & (under <= SHELL_ENCLOSED)]
    else:
        cand = cand[cand > floor]
    cand = cand[cand < lo]
    if not cand.size or not step > 0:
        return None
    k = int(np.ceil(cand.size / N_SHELLS))
    m = np.round(np.abs(anchor - cand) / step).astype(np.int64)
    while k > 1 and not np.any(m % k == 0):
        k -= 1
    keep = cand[m % k == 0]
    keep = keep[np.argsort(np.abs(keep - lo), kind="stable")][:N_SHELLS]   # nearest the shown range first
    spacing = k * step
    dq = float(np.min(np.diff(lv[1:]))) if lv.size > 2 else 0.0
    half = np.full(keep.size, SHELL_WIDTH * spacing)
    if slopes is not None:                              # about SHELL_CELLS cells thick: a tent of mass +-half
        sl = np.asarray(slopes, dtype=np.float64)
        ok = np.isfinite(sl[1:]) & np.isfinite(shown[1:])
        if ok.any():
            o = np.argsort(shown[1:][ok], kind="stable")
            at = np.interp(keep, shown[1:][ok][o], sl[1:][ok][o])
            steep = at > spacing                        # within a cell of the next shell: dots, not a surface
            keep, at, half = keep[~steep], at[~steep], half[~steep]
            half = np.minimum(half, 0.5 * SHELL_CELLS * at)
    half = np.maximum(half, 0.5 * dq)
    below = keep + 2.0 * half + dq < lo                 # the whole band where the volume is clear
    keep, half = keep[below], half[below]
    if not keep.size:
        return None
    t = np.linspace(0.0, 1.0, keep.size) if keep.size > 1 else np.zeros(1)
    rgb = np.round((1.0 - t)[:, None] * np.array(SHELL_NEAR) + t[:, None] * np.array(SHELL_DEEP)).astype(np.uint8)
    held = np.array([float((cnt * band(shown, v, h)).sum()) for v, h in zip(keep, half)])     # cells in each band
    cover = held / cnt[1:].sum()
    across = float(cells) if cells else float(np.cbrt(cnt[1:].sum()))
    alpha = np.clip(SHELL_INK * across ** 2 / np.maximum(held, 1.0), SHELL_ALPHA[0], SHELL_ALPHA[1] * (0.5 + 0.5 * t))
    ratio = across / float(REF_CELLS)                   # a reference cell (REF_CELLS across the window) in cells
    opacity = 1.0 - np.power(1.0 - alpha, ratio)
    return {"levels": keep, "values": 10.0 ** keep if log else keep.copy(), "colours": rgb, "half": half,
            "cover": cover, "alpha": alpha, "opacity": opacity, "spacing": spacing, "step": step, "plan": plan,
            "field": field, "log": bool(log), "symmetric": bool(symmetric)}


def shell_spec(shells: dict | None) -> tuple | None:
    """A shell plan as volume3d.transfer's 'shells' (hashable: it keys the 3D view's pictures): per shell (value,
    r, g, b, opacity, half-width), None for none."""
    if not shells:
        return None
    return tuple((float(v), *(int(c) for c in rgb), float(op), float(h))
                 for v, rgb, op, h in zip(shells["levels"], shells["colours"], shells["opacity"], shells["half"]))


def describe_shells(shells: dict | None) -> str:
    """The 3D volume's colour bar tooltip with shells: how many, at which values (two lines at most)."""
    if not shells:
        return ""
    n = int(shells["levels"].size)
    vals = ", ".join(f"{v:.2g}" for v in shells["values"])
    name = "ρ/ρ̄" if shells["field"] == "density" else "|value|" if shells["symmetric"] else "value"
    return (f"{n} contour shell{'s' if n != 1 else ''} where the volume is clear, at {name} = {vals} (the key at "
            f"the top left); lower the opacity to see them past the walls in front")


def in_saves(opts: dict) -> bool:
    """Whether a saved map or figure grid (figsave's options 'opts') carries the contour lines the screen shows: when
    the options keep an overlay -- the arrows or the void outlines -- so the Bare image preset ('only the map') is the
    map alone and every other preset has them; the options have no tick of their own for the lines (yet)."""
    return bool(opts.get("arrows") or opts.get("voids"))


def image_segments(lines: dict, scale0: float = 1.0, scale1: float = 1.0) -> list[np.ndarray]:
    """map_lines' pieces in the IMAGE pixels of the map (image = plane.T[::-1]: x along the plane's rows, y downward;
    a sample's centre at x = i + 0.5, y = n1 - j - 0.5), times scale0 / scale1 for a picture resampled to other sizes
    (the figure grid's tiles)."""
    n1 = lines["shape"][1]
    out = []
    for s in lines["segments"]:
        if s.size == 0:
            out.append(np.zeros((0, 2, 2)))
            continue
        out.append(np.stack([(s[..., 0] + 0.5) * scale0, (n1 - s[..., 1] - 0.5) * scale1], axis=-1))
    return out


def describe(lines: dict | None) -> str:
    """One or two lines for the colour bar's tooltip: how many contour lines, at which values."""
    if not lines:
        return ""
    plan, levs, log = lines["plan"], lines["levels"], lines["log"]
    n = int(levs.size)
    unit = "in log₁₀ " if log else ""

    def num(v):
        return f"{v:.4g}"
    head = f"{n} contour line{'s' if n != 1 else ''} at the values marked on the bar"
    if lines["field"] == "streams" and not lines["compare"]:
        return head + ": the even stream counts (each the edge between n − 1 and n + 1 streams)"
    if plan["mean"] is not None:
        mean = "ρ/ρ̄ = 1"
        return (f"{head}: every {num(plan['below'])} {unit}below the mean {mean}, the mean itself, every "
                f"{num(plan['above'])} {unit}above it")
    vals = lines["values"]
    return f"{head}: every {num(plan['step'])} {unit}from {num(vals[0])} to {num(vals[-1])}".replace("  ", " ")
