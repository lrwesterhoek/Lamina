"""The set-up that tests/dtfe_sampling_check.sh shares between its snapshot writer and its checks: the linear velocity
field, the box, the regular grid's region, the redshift cone and the user sampling points."""

import numpy as np

N = 4096                    # particles, uniform in the box (an LCG: tests/dtfe_sampling_check.sh)
BOX = 10.0                  # Mpc, not periodic
A = 10.0 * np.array([[1.0, 0.3, -0.2],
                     [0.5, -0.7, 0.4],
                     [-0.6, 0.2, 0.9]])          # v = A x + b: (km/s)/Mpc, a full (non-symmetric) gradient
B = np.array([100.0, -50.0, 25.0])               # km/s
REGION = (2.5, 7.5)         # the regular grids' region, the same on every axis (--region-mpc)
GRID = 8                    # the regular grid (the finer one of method 3's check is 3 GRID)
NS = 20                     # method 2's samples per cell (its default)
CONE = ((0.5, 2.0, 4), (30.0, 150.0, 4), (0.0, 300.0, 6))   # (min, max, bins) of r [Mpc], theta, psi [degrees]
ORIGIN = np.array([5.0, 5.0, 5.0])
SLAB_REGION = ((2.5, 2.5, 0.0), (7.5, 7.5, 10.0))   # S8's grid below the slab of particles: its z-rows start outside
SLAB_GRID = (8, 8, 16)


def cone_centres(origin=ORIGIN):
    """The cone cells' centres in Cartesian coordinates, in the output order (r outermost, psi innermost)."""
    (r0, r1, nr), (t0, t1, nt), (p0, p1, npsi) = CONE
    r = r0 + (np.arange(nr) + 0.5) * (r1 - r0) / nr
    t = np.radians(t0 + (np.arange(nt) + 0.5) * (t1 - t0) / nt)
    p = np.radians(p0 + (np.arange(npsi) + 0.5) * (p1 - p0) / npsi)
    R, T, P = np.meshgrid(r, t, p, indexing="ij")
    x = np.stack([R * np.sin(T) * np.cos(P), R * np.sin(T) * np.sin(P), R * np.cos(T)], axis=-1).reshape(-1, 3)
    return x + origin


def user_samples():
    """64 user sampling points (a 4^3 lattice in [3.5, 6.5]^3, shifted by fixed offsets) and the size of each one's
    sampling cube (0.3-0.8 Mpc per axis): an (64, 6) array of 'x y z dx dy dz' rows."""
    k = np.arange(64)
    i, j, m = k // 16, (k // 4) % 4, k % 4
    pos = 3.5 + np.stack([i, j, m], axis=1) + 0.25 * np.stack([np.sin(k), np.cos(1.7 * k), np.sin(2.3 * k)], axis=1)
    size = 0.3 + 0.5 * np.stack([(k * 7 % 11) / 10.0, (k * 5 % 13) / 12.0, (k * 3 % 7) / 6.0], axis=1)
    return np.hstack([pos, size])


def outside_samples(us):
    """S9's sample list: one sample before and two after the 64 user samples, all three outside REGION."""
    lo, hi = REGION
    pre = np.array([[lo - 1.5, 5.0, 5.0, 0.4, 0.4, 0.4]])
    post = np.array([[hi + 1.0, 5.0, 5.0, 0.4, 0.4, 0.4], [5.0, 5.0, hi + 1.5, 0.3, 0.3, 0.3]])
    return np.vstack([pre, us, post])


def grid_centres(lo, hi, n):
    """The centres of an n[0] x n[1] x n[2] grid over the box lo..hi, in the output order (z innermost)."""
    c = [lo[d] + (np.arange(n[d]) + 0.5) * (hi[d] - lo[d]) / n[d] for d in range(3)]
    g = np.meshgrid(*c, indexing="ij")
    return np.stack([a.ravel() for a in g], axis=1)


def slab(x):
    """S8's particles: the snapshot's positions with z squeezed from [0, BOX] into [0.3 BOX, BOX] (float32, as
    written), so the hull's lower face lies near z = 3 Mpc while the box stays [0, BOX]^3."""
    xs = x.copy()
    xs[:, 2] = 0.3 * BOX + 0.7 * xs[:, 2]
    return xs.astype(np.float32).astype(np.float64)
