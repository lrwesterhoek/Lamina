"""The threads the analysis spreads its large numpy passes over. numpy and scipy release the GIL in those passes,
and every chunk is the same element-wise computation as the whole, so a threaded result is bit-identical to the
serial one."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

THREADS = max(1, min(8, os.cpu_count() or 1))
THREAD_MIN = 1 << 21          # grid elements below which a pass keeps the plain call


def map_chunks(n: int, fn, min_size: int = THREAD_MIN) -> list:
    """[fn(lo, hi)] over [0, n) cut into THREADS contiguous chunks run at once, in chunk order; one call over the
    whole range when n < min_size."""
    if n < min_size or THREADS == 1:
        return [fn(0, n)]
    edges = np.linspace(0, n, THREADS + 1).astype(np.int64)
    pieces = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(lambda ab: fn(*ab), pieces))
