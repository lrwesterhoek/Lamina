"""Lagrangian (particle-ID) linking of voids between snapshots: the FATE of every void. The particles inside a
void are found again by ParticleID at the next snapshot; classes SURVIVE / MERGE / SPLIT / UNRESOLVED / VANISH
follow from the shared fractions (thresholds are module constants). Chunked reads, link_stream never holds the pool."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .cli import snapdir_path

SURVIVE_FRAC = 0.5      # shared fraction that makes a descendant THE descendant (survive / merge)
SPLIT_FRAC = 0.25       # shared fraction each of >= 2 descendants needs for a split
# 'unresolved' comes last, so the codes 0-3 mean the same in every census output
CLASSES = ("survive", "merge", "split", "vanish", "unresolved")
CLASS_CODE = {c: i for i, c in enumerate(CLASSES)}
# the cached links' format: 2 = descendants from the whole pool; a version-1 link (descendants = the
# followed sample only) counts the matter in unfollowed minima as vanished, so it is never served
LINK_VERSION = 2
CHUNK = int(os.environ.get("DTFE_FATE_CHUNK", 8_000_000))   # particles per read slice
# The membership sphere's largest radius, in smoothing lengths of the catalogue's own smoothing: members are the
# particles within min(radius, FATE_MAX_SIGMA sigma) of the centre, followed and pool spheres alike. The R_v
# SEARCH reaches 24 sigma, but uncapped the pair count grows with R^3 and the giant overlapping spheres inflate
# MERGE at z >= 3. The measured R_v is not touched.
FATE_MAX_SIGMA = 15.0


# Chunks made at once, in threads. A chunk's tree build, ball queries and ID lookups mostly release the GIL; the
# chunks are handed out in their serial order, so every result is the serial one bit for bit. Each worker holds a
# chunk's working set, so the count is capped by the free memory (_available_bytes): DTFE_FATE_WORKERS overrides
# (1 = serial).
FATE_WORKERS_MAX = 4
_WORKER_BYTES_PER_PARTICLE = {True: 160, False: 400}    # exclusive / overlapping members, per particle of a CHUNK


def _available_bytes():
    """What the other programs leave free now (bytes), or None (the rule of tools/void_outlines.available_bytes):
    DTFE_FREE_GB, else psutil, else macOS vm_stat (free + inactive + speculative + purgeable), else MemAvailable."""
    if os.environ.get("DTFE_FREE_GB"):
        return float(os.environ["DTFE_FREE_GB"]) * 1e9
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        import re
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            page = int(re.search(r"page size of (\d+) bytes", out.stdout).group(1))
            pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages ([\w ]+):\s+(\d+)\.", out.stdout, re.M)}
            return float(page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")))
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def fate_workers(exclusive: bool = False, chunk: int = None) -> int:
    """Chunks to make at once (module note above FATE_WORKERS_MAX): DTFE_FATE_WORKERS when set, else up to
    FATE_WORKERS_MAX and half the cores, as many as fit in half the free memory (one when it cannot be read)."""
    env = os.environ.get("DTFE_FATE_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    w = max(1, min(FATE_WORKERS_MAX, (os.cpu_count() or 2) // 2))
    free = _available_bytes()
    if free is None:
        return 1
    per = _WORKER_BYTES_PER_PARTICLE[bool(exclusive)] * float(CHUNK if chunk is None else chunk)
    return int(max(1, min(w, 0.5 * free // per)))


def combined_path(sim: str, snap: int, root=None) -> Path:
    """The merged snapshot (python/tools/merge_HDF5.py) of 'sim' at 'snap'; FileNotFoundError when absent."""
    p = snapdir_path(sim, int(snap), root) / f"combined_{int(snap):03d}.hdf5"
    if not p.is_file():
        raise FileNotFoundError(f"no combined snapshot {p}")
    return p


def iter_particles(path, chunk: int = CHUNK):
    """(ParticleIDs uint64, positions in FRACTIONAL box units [0,1) float64 (m,3)) per chunk of PartType1. The box is
    the header's BoxSize, in the coordinates' own units (h-free ckpc for a merged file), so the fractions are the
    void catalogue's frame whatever the unit convention."""
    with h5py.File(path, "r") as f:
        box = float(f["Header"].attrs["BoxSize"])
        pos, ids = f["PartType1/Coordinates"], f["PartType1/ParticleIDs"]
        for lo in range(0, ids.shape[0], chunk):
            hi = min(lo + chunk, ids.shape[0])
            frac = np.asarray(pos[lo:hi], dtype=np.float64) / box
            yield np.asarray(ids[lo:hi], dtype=np.uint64), np.mod(frac, 1.0)


def void_sample(cat, n_grid: int) -> np.ndarray:
    """Catalogue rows of the voids that are linked: distinct, well resolved, with a measured R_v (sorted)."""
    from . import catalog                         # pipeline-heavy: imported only when a real catalogue is used
    rv = catalog.r_v_mpc(cat)
    resolved = np.asarray(cat["well_resolved"]).astype(bool) & np.isfinite(rv)
    return np.nonzero(catalog.distinct_mask(cat, resolved, int(n_grid)))[0]


def membership_cap_cells(sigma_cells=None) -> float:
    """The largest membership radius, in cells: FATE_MAX_SIGMA smoothing lengths of 'sigma_cells' (default
    config.SMOOTHING_SIGMA_CELLS; a catalogue records no sigma of its own). A sigma below one cell counts as one,
    as in the R_v search (pipeline.void_radii_cells)."""
    if sigma_cells is None:
        import config
        sigma_cells = config.SMOOTHING_SIGMA_CELLS
    return FATE_MAX_SIGMA * max(float(sigma_cells), 1.0)


def void_pool(cat, n_grid: int, sample_rows, sigma_cells=None) -> tuple[np.ndarray, np.ndarray]:
    """(catalogue rows, radii in CELLS) of the DESCENDANT pool: the followed sample first (pool index == sample
    index), then every other row with a usable radius, ascending: min(R_v, R_eff), R_eff where it has no R_v and
    R_v where it has no R_eff. Every radius is capped at membership_cap_cells(sigma_cells); a NaN stays NaN."""
    from . import catalog, pipeline
    sample_rows = np.asarray(sample_rows, dtype=np.int64)
    rv = catalog.r_v_cells(cat)
    cell = pipeline.catalog_cell_mpc(cat)
    r_eff = catalog.r_eff_mpc(cat) / cell if cell else np.full(rv.size, np.nan)
    claim = np.fmin(rv, r_eff)                      # fmin: the finite one where only one exists
    usable = np.isfinite(claim) & (claim > 0)
    usable[sample_rows] = False
    rest = np.nonzero(usable)[0]
    radii = np.concatenate([np.asarray(cat["r_v_cells"], dtype=np.float64)[sample_rows], claim[rest]])
    return np.concatenate([sample_rows, rest]), np.minimum(radii, membership_cap_cells(sigma_cells))


def void_spheres(cat, n_grid: int, rows, radii_cells=None, sigma_cells=None) -> tuple[np.ndarray, np.ndarray]:
    """(centres (k,3), radii (k,)) of catalogue rows in fractional box units: the minimum's cell CENTRE and the
    membership radius -- R_v capped at membership_cap_cells(sigma_cells) -- or 'radii_cells', in cells,
    as given (void_pool's radii, capped there)."""
    rows = np.asarray(rows, dtype=np.int64)
    centres = (np.asarray(cat["coords"], dtype=np.float64)[rows] + 0.5) / float(n_grid)
    if radii_cells is None:
        radii = np.minimum(np.asarray(cat["r_v_cells"], dtype=np.float64)[rows], membership_cap_cells(sigma_cells))
    else:
        radii = np.asarray(radii_cells, dtype=np.float64)
    return np.mod(centres, 1.0), radii / float(n_grid)


@dataclass
class MemberSet:
    """The member particles of k voids: (ids, void) pairs sorted by id (a particle in two spheres appears twice),
    the member count per void and the particles read (for the mean density inside each sphere). 'n_sample':
    the first n_sample voids are the FOLLOWED ones of a pool (void_pool), None = all of them."""
    ids: np.ndarray          # uint64, sorted
    void: np.ndarray         # int32, the void (0..k-1, in the order of the spheres) of each pair
    n_members: np.ndarray    # int64 (k,)
    n_particles: int
    radii: np.ndarray        # fractional R_v (k,)
    n_sample: int | None = None

    def sample(self) -> "MemberSet":
        """The members of the followed voids alone (the progenitor side of a link): the pairs of voids < n_sample,
        still sorted by id -- the same set void_members gives for the followed spheres alone."""
        k = self.n_members.size if self.n_sample is None else int(self.n_sample)
        if k >= self.n_members.size:
            return self
        m = self.void < k
        return MemberSet(self.ids[m], self.void[m], self.n_members[:k].copy(), self.n_particles, self.radii[:k], k)

    def enclosed_delta(self) -> np.ndarray:
        """Mean density contrast of the matter inside each sphere: members / (N x 4/3 pi R^3) - 1 (a void: < 0)."""
        vol = 4.0 / 3.0 * np.pi * self.radii ** 3
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.n_members / (float(self.n_particles) * vol) - 1.0


class _PairCollector:
    """Member pairs gathered piece by piece and handed back sorted by id. While every id fits 32 bits a pair is ONE
    uint64 key id << 32 | void, sorted IN PLACE; the order is the stable argsort's, since a particle lives in one
    chunk and its pairs arrive in sphere order. Larger ids: separate arrays and the stable argsort."""

    def __init__(self):
        self.packed, self.keys, self.ids, self.void = True, [], [], []

    def add(self, ids, void):
        if not ids.size:
            return
        if self.packed and int(ids.max()) >= 1 << 32:
            self.ids = [k >> np.uint64(32) for k in self.keys]
            self.void = [(k & np.uint64(0xFFFFFFFF)).astype(np.int32) for k in self.keys]
            self.packed, self.keys = False, []
        if self.packed:
            self.keys.append((ids.astype(np.uint64) << np.uint64(32)) | void.astype(np.uint64))
        else:
            self.ids.append(np.asarray(ids, dtype=np.uint64))
            self.void.append(np.asarray(void, dtype=np.int32))

    def sorted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.packed:
            keys = np.concatenate(self.keys) if self.keys else np.zeros(0, np.uint64)
            self.keys = []
            keys.sort()                                   # in place; (id, void) pairs are distinct
            if sys.byteorder == "little":                 # the low words as a view: no 8-byte temporary
                void = keys.view(np.uint32)[0::2].astype(np.int32)
            else:
                void = (keys & np.uint64(0xFFFFFFFF)).astype(np.int32)
            keys >>= np.uint64(32)
            return keys, void
        # one array at a time, its pieces freed before the next
        ids = np.concatenate(self.ids) if self.ids else np.zeros(0, np.uint64)
        self.ids = []
        void = np.concatenate(self.void) if self.void else np.zeros(0, np.int32)
        self.void = []
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        void = void[order]
        return ids, void


def void_members(particles, centres, radii, exclusive: bool = False, n_sample: int | None = None) -> MemberSet:
    """Members (within R_v of the centre, periodic) of every sphere, from an iterable of (ids, frac) chunks.
    exclusive: a particle inside several spheres goes to the one whose centre is nearest in units of its R_v. n_sample:
    the first n_sample spheres are a pool's followed voids; under 'exclusive' a particle goes to the nearest of THOSE."""
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    k = radii.size
    n_s = k if n_sample is None else int(n_sample)
    got, n_part = _PairCollector(), 0
    for c_ids, c_void, n_c in _chunk_pairs(particles, centres, radii, exclusive, n_s):
        n_part += n_c
        for i_p, v_p in zip(c_ids, c_void):
            got.add(i_p, v_p)
        del c_ids, c_void
    ids, void = got.sorted()
    del got
    return MemberSet(ids, void, np.bincount(void, minlength=k).astype(np.int64), n_part, radii,
                     None if n_sample is None else n_s)


def _chunk_pairs(particles, centres, radii, exclusive: bool, n_s: int, with_index: bool = False, workers=None,
                 post=None):
    """Per chunk of 'particles' (a path or an iterable of (ids, frac)): ([ids pieces], [void pieces], particles in
    the chunk), in void_members' order; shared with the streamed link. with_index: also ([chunk-local particle index
    pieces], the chunk's ids). workers: chunks made at once, yielded in chunk order; 'post' runs on each in its worker."""
    if isinstance(particles, (str, Path)):
        particles = iter_particles(particles)
    centres = np.mod(np.asarray(centres, dtype=np.float64).reshape(-1, 3), 1.0)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    w = fate_workers(exclusive) if workers is None else max(1, int(workers))

    def job(ids, frac):
        r = _pairs_of_chunk(ids, frac, centres, radii, exclusive, n_s, with_index)
        return post(r) if post is not None else r

    if w <= 1:
        for ids, frac in particles:
            yield job(ids, frac)
        return
    # at most w chunks in the workers (each holds its slice, tree and pairs: fate_workers' memory cap); the next slice
    # is read while they run, and the oldest is handed out first
    with ThreadPoolExecutor(w) as ex:
        pending = deque()
        for ids, frac in particles:
            pending.append(ex.submit(job, ids, frac))
            del ids, frac
            if len(pending) >= w:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _pairs_of_chunk(ids, frac, centres, radii, exclusive: bool, n_s: int, with_index: bool):
    """One chunk's tuple of _chunk_pairs (centres wrapped, radii float64 there): read-only on its inputs, so chunks
    run in threads (cKDTree's queries and numpy's sorts, gathers and scatters release the GIL)."""
    k = radii.size
    if not k or not ids.size:
        return ([], [], ids.size, [], ids) if with_index else ([], [], ids.size)
    # unbalanced, non-compact: 3-4x faster to build, and each tree serves only k ball queries
    tree = cKDTree(np.mod(frac, 1.0), boxsize=1.0 + 1e-12, balanced_tree=False, compact_nodes=False)
    c_ids, c_void, c_idx = [], [], []
    if exclusive:
        # a particle lives in ONE chunk, so the nearest sphere is chosen here as a RUNNING best per chunk particle
        # (a lexsort of every (id, q) pair is far slower and holds four joined arrays). The spheres come in order
        # and the test is strict ('<'), so an equal q keeps the LOWER sphere: the stable lexsort's choice, bit for
        # bit (tests/py_fate_test.py)
        best_q = np.full(ids.size, np.inf, dtype=np.float32)
        best_v = np.full(ids.size, -1, dtype=np.int32)
    for v in range(k):
        if not radii[v] > 0:
            continue
        idx = np.asarray(tree.query_ball_point(centres[v], radii[v]), dtype=np.int64)
        if not idx.size:
            continue
        if exclusive:                # distance in units of this void's R_v
            d = frac[idx] - centres[v]
            d -= np.rint(d)
            q = np.sqrt(np.einsum("ij,ij->i", d, d)) / radii[v]
            # an unfollowed sphere ranks behind every followed one that holds the particle (q <= 1 there)
            q = (q + (2.0 if v >= n_s else 0.0)).astype(np.float32)
            better = q < best_q[idx]                 # a ball query never returns an index twice
            idx = idx[better]
            best_q[idx] = q[better]
            best_v[idx] = v
            del d, q, better
            continue
        c_ids.append(ids[idx])
        c_void.append(np.full(idx.size, v, dtype=np.int32))
        if with_index:
            c_idx.append(idx)
    del tree
    if exclusive:
        has = np.flatnonzero(best_v >= 0)
        if has.size:
            # by id, as the lexsort gave them (TNG's ParticleIDs are unique: the q key never decides)
            has = has[np.argsort(ids[has], kind="stable")]
            s_ids = ids[has]
            dup = s_ids[1:] == s_ids[:-1]
            if dup.any():
                # an id held by two particles of the chunk (not in a snapshot; a hand-made input): one pair per
                # ID, the nearest sphere over both particles, as the lexsort chose (an exact tie of q in one
                # sphere may then keep the other particle's index, never another void)
                has = has[np.lexsort((best_v[has], best_q[has], s_ids))]
                s_ids = ids[has]
                first = np.ones(has.size, dtype=bool)
                first[1:] = s_ids[1:] != s_ids[:-1]
                has, s_ids = has[first], s_ids[first]
                del first
            c_ids, c_void = [s_ids], [best_v[has]]
            if with_index:
                c_idx = [has]
            del s_ids, dup
        del best_q, best_v, has
    if with_index and len(c_ids) > 1:
        # joined HERE, pieces dropped: this frame holds them until the next chunk, through the consumer's join
        c_ids, c_void, c_idx = [np.concatenate(c_ids)], [np.concatenate(c_void)], [np.concatenate(c_idx)]
    return (c_ids, c_void, ids.size, c_idx, ids) if with_index else (c_ids, c_void, ids.size)


@dataclass
class Link:
    """Shared member counts between the k_a progenitor and k_b descendant voids of one interval (sparse: only
    the pairs that share a particle). The descendants are a pool: 0..n_desc_sample-1 are the followed
    voids of the later snapshot (its sample indices), the rest unfollowed minima; -1 / None = all followed."""
    prog: np.ndarray         # int32, progenitor index (0..k_a-1)
    desc: np.ndarray         # int32, descendant index (0..k_b-1)
    count: np.ndarray        # int64, particles shared
    n_prog_members: np.ndarray
    n_desc_members: np.ndarray
    n_desc_sample: int | None = None

    def n_followed(self) -> int:
        k_b = int(self.n_desc_members.size)
        return k_b if self.n_desc_sample is None or self.n_desc_sample < 0 else min(int(self.n_desc_sample), k_b)


def link(a: MemberSet, b: MemberSet, block: int = 20_000_000) -> Link:
    """The contingency table of two member sets by shared ParticleID (a many-to-many join on the sorted ids): 'a'
    the progenitors (MemberSet.sample() of a pool), 'b' the descendant pool (its n_sample = the followed ones)."""
    k_b = b.n_members.size
    acc = np.zeros(0, np.int64), np.zeros(0, np.int64)
    # in blocks of progenitor pairs: one join over all of them would hold too many index arrays at once
    for lo_i in range(0, a.ids.size, block):
        ids, void = a.ids[lo_i:lo_i + block], a.void[lo_i:lo_i + block]
        lo = np.searchsorted(b.ids, ids, side="left")
        mult = np.searchsorted(b.ids, ids, side="right") - lo
        tot = int(mult.sum())
        if tot == 0:
            continue
        starts = np.repeat(lo, mult) + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(mult) - mult, mult))
        key = np.repeat(void.astype(np.int64), mult) * k_b + b.void[starts]
        del starts
        uk, cnt = np.unique(key, return_counts=True)
        acc = (np.concatenate([acc[0], uk]), np.concatenate([acc[1], cnt]))
    return _link_from_acc(acc, k_b, a.n_members.copy(), b.n_members.copy(), b.n_sample)


def _link_from_acc(acc, k_b, n_prog_members, n_desc_members, n_desc_sample) -> Link:
    """The Link of accumulated (key = prog * k_b + desc, count) pieces: one sum per key, keys ascending."""
    uk, inv = np.unique(acc[0], return_inverse=True)
    cnt = np.bincount(inv.reshape(-1), weights=acc[1], minlength=uk.size).astype(np.int64)
    return Link((uk // k_b).astype(np.int32), (uk % k_b).astype(np.int32), cnt,
                n_prog_members, n_desc_members, n_desc_sample)


# the streamed link counts into a dense (k_a x k_b) table when it is at most this many cells (TNG100: ~250 x 620),
# with np.bincount -- linear time, where np.unique sorts every matched pair; else sparse, as link() does
DENSE_MAX = 50_000_000


def _join_into(acc, a: MemberSet, lo_p, mult_p, idx, desc_void, k_b, block):
    """Add to 'acc' the shared counts of descendant pairs with the progenitor set 'a' (sorted ids): pair i is the
    particle idx[i] of the chunk, in pool void desc_void[i]; lo_p / mult_p are each chunk particle's first position
    and multiplicity in a.ids. 'acc' is a dense int64 table (key = prog * k_b + desc) or a sparse (keys, counts)."""
    dense = not isinstance(acc, tuple)
    for lo_i in range(0, idx.size, block):
        px, bv = idx[lo_i:lo_i + block], desc_void[lo_i:lo_i + block]
        mult = mult_p[px]
        tot = int(mult.sum())
        if tot == 0:
            continue
        lo = lo_p[px]
        del px
        starts = np.repeat(lo, mult) + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(mult) - mult, mult))
        del lo
        key = a.void[starts].astype(np.int64) * k_b + np.repeat(bv.astype(np.int64), mult)
        del starts
        if dense:
            acc += np.bincount(key, minlength=acc.size)
        else:
            uk, cnt = np.unique(key, return_counts=True)
            acc = (np.concatenate([acc[0], uk]), np.concatenate([acc[1], cnt]))
        del key
    if not dense and acc[0].size:
        uk, inv = np.unique(acc[0], return_inverse=True)
        acc = (uk, np.bincount(inv.reshape(-1), weights=acc[1], minlength=uk.size).astype(np.int64))
    return acc


def link_stream(progs, particles, centres, radii, exclusive: bool = False, n_sample: int | None = None,
                keep_sample: bool = True, block: int = 4_000_000, workers=None) -> tuple[list[Link], MemberSet | None]:
    """The links of every progenitor set of 'progs' to the descendant POOL whose spheres are (centres, radii), WITHOUT
    holding the pool's member pairs: chunk by chunk, keeping only the sparse (progenitor, descendant) counts; identical
    to [link(a, void_members(...)) for a in progs]. keep_sample: also return the followed spheres' members."""
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    k = radii.size
    n_s = k if n_sample is None else int(n_sample)
    accs = [np.zeros(int(a.n_members.size) * k, np.int64) if int(a.n_members.size) * k <= DENSE_MAX
            else (np.zeros(0, np.int64), np.zeros(0, np.int64)) for a in progs]
    n_desc = np.zeros(k, dtype=np.int64)
    keep, n_part = _PairCollector(), 0

    def lookup(r):
        """In the chunk's worker: each particle of the chunk looked up once in every progenitor set -- its first
        position and multiplicity in a.ids -- in id order (sorted queries: a cache-friendly binary search)."""
        c_ids, c_void, n_c, c_idx, ids_c = r
        if not c_ids:
            return [], [], n_c, [], []
        o = np.argsort(ids_c)
        s_ids = ids_c[o]
        found = []
        for a in progs:
            lo_p = np.empty(ids_c.size, np.int64)
            mult_p = np.empty(ids_c.size, np.int64)
            lo_p[o] = np.searchsorted(a.ids, s_ids, side="left")
            mult_p[o] = np.searchsorted(a.ids, s_ids, side="right")
            mult_p -= lo_p
            found.append((lo_p, mult_p))
        return c_ids, c_void, n_c, c_idx, found

    for c_ids, c_void, n_c, c_idx, found in _chunk_pairs(particles, centres, radii, exclusive, n_s, with_index=True,
                                                          workers=workers, post=lookup):
        n_part += n_c
        if not c_ids:
            continue
        ci, cv, cx = c_ids[0], c_void[0], c_idx[0]          # one piece each (_chunk_pairs with_index)
        del c_ids, c_void, c_idx
        n_desc += np.bincount(cv, minlength=k)
        for j, a in enumerate(progs):
            lo_p, mult_p = found[j]
            found[j] = None
            accs[j] = _join_into(accs[j], a, lo_p, mult_p, cx, cv, k, block)
            del lo_p, mult_p
        del found, cx
        if keep_sample:
            if n_s >= k:
                keep.add(ci, cv)
            else:
                m = cv < n_s
                keep.add(ci[m], cv[m])
                del m
        del ci, cv
    links = []
    for acc, a in zip(accs, progs):
        if not isinstance(acc, tuple):           # dense: its nonzero cells, keys ascending (as np.unique gives them)
            nz = np.flatnonzero(acc)
            acc = (nz.astype(np.int64), acc[nz])
        links.append(_link_from_acc(acc, k, a.n_members.copy(), n_desc.copy(), None if n_sample is None else n_s))
    if not keep_sample:
        return links, None
    # filter-then-sort == sort-then-filter: the same order as void_members(...).sample()
    ids, void = keep.sorted()
    del keep
    k_s = min(n_s, k)
    return links, MemberSet(ids, void, np.bincount(void, minlength=k_s).astype(np.int64)[:k_s], n_part, radii[:k_s],
                            None if n_sample is None else k_s)


def _top_two(prog, desc, f, k_a):
    """Per progenitor: the descendant with the largest fraction (-1 if none), that fraction and the second largest."""
    order = np.lexsort((-f, prog))                # per progenitor, largest fraction first (ties: lower index first)
    p_s, d_s, f_s = prog[order], desc[order], f[order]
    first = np.ones(p_s.size, dtype=bool)
    first[1:] = p_s[1:] != p_s[:-1]
    second = np.zeros(p_s.size, dtype=bool)
    second[1:] = first[:-1] & ~first[1:]
    best = np.full(k_a, -1, dtype=np.int64)
    f_max, f_second = np.zeros(k_a), np.zeros(k_a)
    best[p_s[first]] = d_s[first]
    f_max[p_s[first]] = f_s[first]
    f_second[p_s[second]] = f_s[second]
    return best, f_max, f_second


def classify(lk: Link, survive_frac: float = SURVIVE_FRAC, split_frac: float = SPLIT_FRAC) -> dict:
    """Per progenitor (over the whole descendant pool): 'cls' (code into CLASSES), 'desc' (main descendant pool index,
    -1 if none), 'in_sample', 'f_desc', 'f_max' / 'f_second' (the two largest shared fractions), 'f_sample', 'n_split'
    (pool voids holding >= split_frac) and 'n_main' (>= 2 is a merge). A progenitor without members vanishes."""
    k_a = lk.n_prog_members.size
    k_s = lk.n_followed()
    f = np.zeros(lk.count.size)
    nm = lk.n_prog_members[lk.prog].astype(np.float64)
    np.divide(lk.count, nm, out=f, where=nm > 0)
    best, f_max, f_second = _top_two(lk.prog, lk.desc, f, k_a)
    s = lk.desc < k_s
    best_s, f_sample, _f2 = _top_two(lk.prog[s], lk.desc[s], f[s], k_a)
    # a followed void holding half the matter is THE descendant even when an overlapping unfollowed minimum (a
    # sub-minimum of the same void, as a rule) holds more: the matter is in the followed void
    pick_s = f_sample >= survive_frac
    desc = np.where(pick_s, best_s, best)
    f_desc = np.where(pick_s, f_sample, f_max)
    n_split = np.bincount(lk.prog[f >= split_frac], minlength=k_a)
    main = f_desc >= survive_frac
    # a descendant reached with >= survive_frac by two or more progenitors: they MERGE
    shared = np.bincount(desc[main], minlength=max(int(lk.n_desc_members.size), 1))
    n_main = np.where(main, shared[np.where(main, desc, 0)], 0)
    desc_out = np.where(f_desc > 0, desc, -1)
    in_sample = (desc_out >= 0) & (desc_out < k_s)
    cls = np.full(k_a, CLASS_CODE["vanish"], dtype=np.int8)
    cls[main] = CLASS_CODE["survive"]
    cls[main & (n_main >= 2)] = CLASS_CODE["merge"]
    cls[~main & (n_split >= 2)] = CLASS_CODE["split"]
    cls[main & ~in_sample] = CLASS_CODE["unresolved"]
    return {"cls": cls, "desc": desc_out, "in_sample": in_sample, "f_desc": f_desc, "f_max": f_max,
            "f_second": f_second, "f_sample": f_sample, "n_split": n_split, "n_main": n_main}


def main_progenitor(lk: Link) -> tuple[np.ndarray, np.ndarray]:
    """Per DESCENDANT (pool index; a followed void's is its sample index): the progenitor (a followed void of the
    earlier snapshot) that gave it the most particles (-1 if none) and that share of the descendant's members --
    the backward link a tracker walks from z = 0 to the past, through followed voids only."""
    k_b = lk.n_desc_members.size
    prog = np.full(k_b, -1, dtype=np.int64)
    frac = np.zeros(k_b)
    if lk.count.size:
        order = np.lexsort((-lk.count, lk.desc))
        d_s = lk.desc[order]
        first = np.ones(d_s.size, dtype=bool)
        first[1:] = d_s[1:] != d_s[:-1]
        prog[d_s[first]] = lk.prog[order][first]
        nm = lk.n_desc_members[d_s[first]].astype(np.float64)
        frac[d_s[first]] = np.where(nm > 0, lk.count[order][first] / np.maximum(nm, 1), 0.0)
    return prog, frac


def chain_survival(fates: list[dict]) -> np.ndarray:
    """Whether each void of the FIRST interval's progenitors still has a main descendant (survive or merge) after
    every interval of 'fates'. The chain follows only FOLLOWED descendants, so it is a lower limit on the matter's
    survival as a void; the direct link (plot_void_fate --ref-snap) checks how much it loses."""
    alive = np.ones(fates[0]["cls"].size, dtype=bool)
    cur = np.arange(fates[0]["cls"].size)
    for fate in fates:
        ok = np.isin(fate["cls"][cur], (CLASS_CODE["survive"], CLASS_CODE["merge"]))
        if "in_sample" in fate:
            ok &= np.asarray(fate["in_sample"])[cur]
        alive &= ok
        cur = np.where(alive, fate["desc"][cur], 0)
    return alive


# ------------------------------------------------------------------ the cached link of one interval
def link_to_arrays(lk: Link) -> dict:
    return {"prog": lk.prog, "desc": lk.desc, "count": lk.count, "n_prog_members": lk.n_prog_members,
            "n_desc_members": lk.n_desc_members, "n_desc_sample": lk.n_followed(), "link_version": LINK_VERSION}


def link_from_arrays(z) -> Link:
    lk = Link(*(np.asarray(z[k]) for k in ("prog", "desc", "count", "n_prog_members", "n_desc_members")))
    lk.n_desc_sample = int(z["n_desc_sample"]) if "n_desc_sample" in z else None
    return lk


# ------------------------------------------------------------------ one snapshot's linked voids, and cached links
class VoidSnapshot:
    """One snapshot's linked voids: the catalogue, the FOLLOWED rows (void_sample) and their spheres ('radii' are the
    MEMBERSHIP radii, R_v capped at FATE_MAX_SIGMA), the descendant POOL (void_pool), the frame, the particle count and
    the catalogue's R_v settings (r_v_record). The pool is read in ONE streamed pass (stream_links)."""

    def __init__(self, sim, snap, method=None, prefix=None, data_root=None, exclusive=False):
        import config
        from . import catalog, pipeline
        self.sim, self.snap, self.exclusive = sim, int(snap), bool(exclusive)
        self.path = combined_path(sim, snap, data_root)
        prod = pipeline.products(int(snap), sim=sim, method=method, prefix=prefix, data_root=data_root)
        try:
            self.cell, self.box, self.n, _src = catalog.frame_of(prod)
            self.cat = prod.voids()
            self.fs, self.method, self.prefix = prod.fs, prod.fs.method, prod.fs.prefix
        finally:
            prod.release()
        with h5py.File(self.path, "r") as f:
            self.n_particles = int(f["PartType1/ParticleIDs"].shape[0])
        # the smoothing the catalogue was served (and its R_v measured) at: the membership cap's unit
        self.sigma_cells = float(config.SMOOTHING_SIGMA_CELLS)
        self.rows = void_sample(self.cat, self.n)
        # the pool first: a followed void's sphere IS its pool sphere (capped, FATE_MAX_SIGMA), the same as
        # progenitor and as descendant; the link caches compare these radii, so a cache made with the uncapped R_v
        # spheres (the 24 sigma reach) is relinked
        self.pool_rows, pool_r = void_pool(self.cat, self.n, self.rows, self.sigma_cells)
        self.centres, self.radii = void_spheres(self.cat, self.n, self.rows, pool_r[:self.rows.size])
        self.pool_centres, self.pool_radii = void_spheres(self.cat, self.n, self.pool_rows, pool_r)
        self.r_v_settings = r_v_record(self.cat)
        self._members = None
        self._prog = None

    def members(self) -> MemberSet:
        """The members of the whole pool, held in memory (the descendant side of link(); the census and the tracker
        stream the pool instead, stream_links: TNG100-3-Dark's pool holds 5-7e8 pairs)."""
        if self._members is None:
            t0 = time.time()
            self._members = void_members(self.path, self.pool_centres, self.pool_radii, self.exclusive,
                                         n_sample=self.rows.size)
            print(f"    snap {self.snap:3d}: {self.rows.size} followed voids of {self.pool_rows.size} in the pool, "
                  f"{self._members.ids.size:,} member particles read in {time.time() - t0:.0f} s", flush=True)
        return self._members

    def prog_members(self) -> MemberSet:
        """The members of the followed voids (the progenitor side of a link): the pool's share when it is read, else
        the followed spheres alone (the same set, void_members' n_sample rule) -- the first snapshot of a census and a
        snapshot whose descendant-side link was cached never need their whole pool."""
        if self._prog is None:
            if self._members is not None:
                self._prog = self._members.sample()
            else:
                t0 = time.time()
                self._prog = void_members(self.path, self.centres, self.radii, self.exclusive)
                print(f"    snap {self.snap:3d}: {self.rows.size} followed voids, {self._prog.ids.size:,} member "
                      f"particles read in {time.time() - t0:.0f} s", flush=True)
        return self._prog

    def stream_links(self, progs, keep_prog: bool = True) -> list[Link]:
        """The links of each snapshot of 'progs' (their followed voids) to this snapshot's whole pool, in ONE pass over
        the particles that never holds the pool's members (link_stream); keep_prog also keeps this
        snapshot's followed voids' members from the same pass (prog_members, for its own link to the next)."""
        pm = [p.prog_members() for p in progs]
        if self._members is not None:                    # already read whole (members()): join in memory
            return [link(a, self._members) for a in pm]
        keep = keep_prog and self._prog is None
        t0 = time.time()
        links, sample = link_stream(pm, self.path, self.pool_centres, self.pool_radii, self.exclusive,
                                    n_sample=self.rows.size, keep_sample=keep)
        if keep:
            self._prog = sample
        print(f"    snap {self.snap:3d}: {self.rows.size} followed voids of {self.pool_rows.size} in the pool streamed "
              f"against {len(pm)} progenitor set(s) in {time.time() - t0:.0f} s"
              + (f" ({sample.ids.size:,} followed member particles kept)" if keep else ""), flush=True)
        return links

    def drop_members(self, keep_prog: bool = False):
        """Free the members; keep_prog keeps the followed voids' (when read) for this snapshot's own link to the next."""
        if keep_prog and self._members is not None and self._prog is None:
            self._prog = self._members.sample()
        self._members = None
        if not keep_prog:
            self._prog = None


def link_tag(method, prefix) -> str:
    """The grid set's tag in link-cache and census names: '' for PS-DTFE's default grids, else '_<method>' or
    '_<prefix>'. One rule for plot_void_fate and plot_void_tracking, so a --prefix run never rewrites the default
    set's link files."""
    default_prefix = {"ps": "ps_output.", "dtfe": "output."}
    g = method if prefix == default_prefix.get(method) else str(prefix).rstrip(".")
    return "" if g == "ps" else f"_{g}"


def link_cache_path(a: VoidSnapshot, b: VoidSnapshot, tag: str = "") -> Path:
    """The pipeline-cache file of the link a -> b (its parameter namespace: smoothing, footprint, criterion);
    '_excl' for exclusive members (a partition: other counts)."""
    from . import pipeline
    excl = "_excl" if getattr(a, "exclusive", False) else ""
    return pipeline.cache_dir() / f"{a.sim}{tag}_{a.method}_fate{excl}_{a.snap:03d}_{b.snap:03d}.npz"


def r_v_record(cat) -> np.ndarray:
    """The R_v settings a catalogue's radii were measured with (its 'r_v_settings' record; the current ones,
    pipeline.r_v_settings, for a catalogue without one -- SnapshotProducts.voids() re-measures such a catalogue
    before serving it, as catalog.provenance assumes). Part of a link cache's key."""
    if "r_v_settings" in cat:
        return np.asarray(cat["r_v_settings"], dtype=np.float64).reshape(-1)
    from . import pipeline
    return np.asarray(pipeline.r_v_settings(), dtype=np.float64).reshape(-1)


def _r_v_of(s) -> np.ndarray:
    rec = getattr(s, "r_v_settings", None)
    return r_v_record(s.cat) if rec is None else np.asarray(rec, dtype=np.float64).reshape(-1)


def _cached_or_none(p: Path, a, b, spheres: dict, r_v: dict, log) -> Link | None:
    """The cached link at 'p' when it is current (LINK_VERSION, both catalogues' R_v settings, the spheres), else None
    (with the reason logged)."""
    if not p.exists():
        return None
    try:
        with np.load(p) as z:
            version = int(z["link_version"]) if "link_version" in z else 1
            if version != LINK_VERSION:
                log(f"    cached {p.name}: link version {version} (descendants = the followed voids only), "
                    f"today's is {LINK_VERSION}: relinking")
                return None
            if any(k not in z for k in r_v):
                log(f"    cached {p.name}: made before the links recorded their catalogues' R_v settings: relinking")
                return None
            for k, v in r_v.items():
                if not (z[k].shape == v.shape and np.array_equal(z[k], v)):
                    log(f"    cached {p.name}: made with other R_v settings ({k[:4]} {z[k].tolist()}, today "
                        f"{v.tolist()}): relinking")
                    return None
            if int(z["n_desc_sample"]) == b.rows.size and all(
                    z[k].shape == v.shape and np.allclose(z[k], v, rtol=0, atol=1e-12) for k, v in spheres.items()):
                return link_from_arrays(z)
            log(f"    cached {p.name}: made for other voids than today's catalogues: relinking")
    except (OSError, ValueError, KeyError) as e:
        log(f"    cached {p.name} unreadable ({e}): relinking")
    return None


def cached_links(progs, b: VoidSnapshot, tag: str = "", log=print, keep_prog: bool = True) -> list[Link]:
    """The shared-particle links p -> b for every p of 'progs' (p's followed voids -> b's whole pool): from the cache
    when it is of the current LINK_VERSION and made with the same R_v settings and spheres, else from the particles
    (then cached atomically; missing links in ONE streamed pass over b). The classes are taken from the counts (classify)."""
    from . import pipeline
    out, todo = [], []
    for a in progs:
        p = link_cache_path(a, b, tag)
        spheres = {"prog_centres": a.centres, "prog_radii": a.radii,
                   "desc_centres": b.pool_centres, "desc_radii": b.pool_radii}
        r_v = {"prog_r_v_settings": _r_v_of(a), "desc_r_v_settings": _r_v_of(b)}
        lk = _cached_or_none(p, a, b, spheres, r_v, log)
        out.append(lk)
        if lk is None:
            todo.append((len(out) - 1, a, p, spheres, r_v))
    if todo:
        made = b.stream_links([a for _i, a, *_r in todo], keep_prog=keep_prog)
        for (i, _a, p, spheres, r_v), lk in zip(todo, made):
            pipeline._save_npz(p, **link_to_arrays(lk), **spheres, **r_v)
            out[i] = lk
    return out


def cached_link(a: VoidSnapshot, b: VoidSnapshot, tag: str = "", log=print, keep_prog: bool = True) -> Link:
    """The link a -> b of cached_links (one progenitor snapshot)."""
    return cached_links([a], b, tag, log, keep_prog)[0]
