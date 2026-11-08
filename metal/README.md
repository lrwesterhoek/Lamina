# GPU acceleration — the Metal deposit kernels

For whoever builds or maintains the GPU backend. Two things run on the GPU: the PS-DTFE grid **deposit**
and the standard-DTFE volume-averaged `_a` **interpolation**, both with `--gpu`. Both are data-parallel over
the ~10M tetrahedra of a partition and dominate the run time; the Delaunay triangulation (CGAL) and point evaluation
stay on the CPU. Any failure (no device, compile, allocation, a deposit the retries give up on) falls back to the CPU
with a warning, as do scalar fields of more than one component.

The tree is Metal-only (Apple Silicon); elsewhere both programs run on the CPU. The earlier CUDA/HIP port of these
kernels, with its CPU emulation (`CUDAEMU=1`) and its twin checks, was never run on NVIDIA or AMD hardware and is kept
on the branch `archive/cuda-hip-2026-11-08`; `make CUDA=1`, `HIP=1` or `CUDAEMU=1` now stops with a message naming it.

## Build and test

```bash
make DTFE PS-DTFE METAL=1          # Apple Silicon (Metal 3, M-series)
make DTFE PS-DTFE DOUBLE=1 METAL=1 # the double pair: float copies of the tetrahedra go to the same kernels
```
A METAL=1 build defines `-DPS_GPU` / `-DDTFE_GPU`. The mode (metal or off) is stamped in `o/` and
`o_ps/` (`.gpu_mode_*`, `.build_mode` holds its make argument); a switch wipes the objects, so a binary never mixes GPU
and plain objects. `make DTFE $(cat o/.build_mode)` rebuilds in the current mode. The Metal kernels are embedded at build
time (`o/dtfe_deposit_msl.h`, `o_ps/ps_deposit_msl.h`, each with `exact_clip.metal.inc` prepended) and compiled once per
process. Dependency: the header-only **metal-cpp**, vendored at `third_party/metal-cpp`.

Tests (GPU builds; the PS suites use the GPU when the `o_ps/` stamp says metal):
- `tests/dtfe_metal_check.sh` (builds in the `o/.build_mode` mode, METAL=1 when that is empty): standard DTFE
  CPU/GPU parity, a repeated GPU run, the per-tet A/B kernel, the shared grid, the exact average.
- `tests/gpu_scalar_check.sh` (the scalar fields, both binaries), `tests/gpu_mass_check.sh` (the exact mass grid),
  `tests/gpu_retry_check.sh` and `tests/dtfe_gpu_retry_check.sh` (checkpoints, retries by cause, pace recovery).
- The PS suites with a `--gpu` part: `ps_linear_deposit_check.sh`, `ps_vertex_mass_check.sh`,
  `ps_volume_weighted_check.sh`, `ps_halo_release_check.sh`, `ps_hidden_streams_check.sh`, `ps_caustic_class_check.sh`,
  `ps_window_check.sh`, `ps_mass_exact_check.sh`, `ps_fixed_moments_check.sh`, `point_exact_serve_check.sh`,
  `python3 tests/ps_nonperiodic_test.py`, and `ps_double_check.sh` for the double pair.
- The kernel harness (Metal only): `bash metal/build_prototype.sh && ./metal/validate_deposit` (T1-T15 below;
  `BENCH=1` adds a 1M-tet timing, `DUMP_PROFILES=1` per-bin profiles, `VALIDATE_FLOAT_MASS=1` the float mass grid).

CI builds no GPU backend (its runners are Linux): all of these are local.

## Files

| file | role |
|---|---|
| `metal/ps_deposit.metal` | PS-DTFE kernels: `depositFields` (one thread per tetrahedron: mass, velocity moment, dispersion second moment, gradient moment, stream count, scalar; the sampled path, the centroid fallback and the deferrals) and `depositExactItems` (the exact deposit, one thread per block of a tetrahedron's window cells) |
| `metal/dtfe_deposit.metal` | standard-DTFE kernels: `depositAveraged1` (per tetrahedron, A/B), `depositAveraged1Items2/4/8` (the item kernels), `depositExactAverage` (`--exact-average`) |
| `metal/exact_clip.metal.inc` | the exact tetrahedron-cell intersection moments (`sweep_moments`, a register-resident z-sweep) and the face-plane cell skip, prepended to both kernel sources |
| `metal/validate_deposit.cpp`, `metal/build_prototype.sh` | the kernel harness and its build (offline-compiles `ps_deposit.metallib` when Xcode's Metal toolchain is installed, which only the harness loads; git-ignored) |
| `src/CGAL_triangulation/gpu_host.h` | the host interface the callers see (deposit calls, `PSGpuGrids`, the shared-grid API `dtfeGpuShared*`, `psWindowHull`) and the PS host's plain-C++ steps |
| `src/CGAL_triangulation/ps_metal_host.cc`, `dtfe_metal_host.cc` | the Metal hosts: device and pipelines, the chunked dispatch controller, retries |
| `src/CGAL_triangulation/gpu_metal_retry.h` | the two hosts' retry policy (snapshots, causes, pace, fault injection) and its pass loop `gpuRunPass` |
| `src/CGAL_triangulation/ps_deposit_params.h`, `dtfe_gpu_fixed.h` | the per-tet record (`psFillTetRecord`) and the fixed-point units (`psMassFixedScale`, `psMomentFixedScale`, `PSFixedBounds`; the standard grids' units) |
| `src/CGAL_triangulation/ps_exact_inside.h` | the CPU's exact sample-inside test the kernels' certified float test reproduces |

The call sites: `interpolateGrid_phaseSpace` in `ps_interpolation.cc` (PS-DTFE) and `averaged_interpolation_1.cc`
(standard DTFE).

## PS-DTFE deposit

The host extracts flat per-tetrahedron arrays (a 16-float record, vertex velocities, masses) with **exactly the CPU
loop's filters** (ownership, hull, degeneracy), cost-sorts them by grid footprint (a chunk ends with its slowest
thread, so mixing one stretched void tetrahedron with cheap halo ones idles most lanes), dispatches the kernels, copies
the grids back and runs the CPU loop over the deferred tetrahedra only. The kernels honour the partition sub-grid
(`subOrigin/subDims`), including a box that wraps a periodic axis.

**Which samples a tetrahedron contains is decided as on the CPU.** Metal has no double, so the kernel classifies each
sample in float against a rigorous error bound, from a record the host builds in the exact test's canonical frame
(vertex 0 as a cell index plus an in-cell fraction, the edges as differences of the canonical positions;
`psFillTetRecord`). Outside the bound the float decision provably equals the exact one. A tetrahedron with a sample
within the bound of a face is **deferred**: the kernel negates its mass and deposits nothing, and the host deposits it
with the CPU's exact test. So stream counts and `.hidden_streams` are byte-identical CPU against GPU, partitioned or not
(single build). A tetrahedron the kernel's float degeneracy check would reject but the host kept is deferred, not lost.

**The exact deposit** (`depositExactItems`) splits the work below the tetrahedron: one thread per item = (tetrahedron,
block of 16 cells of its bounding-box window; `DTFE_GPU_EXACT_CELLS`). With one thread per tetrahedron the per-thread
cost spans five orders of magnitude once cells are smaller than tetrahedra, and each buffer waits on its few giants
(0.26M particles on 256^3: 30 s with items, more than 17 min without). Phase 0 adds each item's clipped weights into a
per-tet total `sumW`, phase 1 deposits each piece with the share m/sumW, so every tetrahedron deposits exactly its
mass; a tetrahedron whose total stays 0 goes through `depositFields` (sampled path, centroid fallback or deferral). The
host bounds every window from its own copy (the kernel re-derives it in float; surplus items idle) and caps the items
at 32M. They are implicit, as the standard deposit's: a prefix sum over the tetrahedra and a block table (4 B per
tetrahedron plus 4 B per 32 items; `psItemOf` finds an item's tetrahedron by a short binary search), built before the
deposit takes the GPU. `DTFE_GPU_ITEMS=0` keeps the one-thread path for A/B. With the dispersion requested, a tetrahedron whose
velocity covariance float32 cannot resolve (`exact_disp_illcond`: a thin tetrahedron with a steep velocity gradient,
`EX_DISP_KAPPA2`) is left out of phase 0 (only phase 0 decides, so phase 1 never drops a tet whose weights were
summed) and deferred by `depositFields` to the CPU's double deposit; every other piece's mean velocity and covariance
are clamped to bounds a correct value always meets.

**The clipping** (`sweep_moments` in `exact_clip.metal.inc`, shared by `depositExactItems` and `depositExactAverage`)
is not an r3d port: it sweeps the tetrahedron in z. Between breakpoints (the middle vertices' z, the z where a
tetrahedron edge crosses a side plane of the cell, the z where a vertical cell edge enters or leaves the tetrahedron)
the cross-section clipped to the cell's rectangle changes only linearly, so its moments are polynomials of degree <= 4
in z and 3-point Gauss-Legendre integrates them exactly; each cross-section's 2D moments come from a closed Green chain
(the polygon's edges clipped to the rectangle plus the rectangle's chords). Everything stays in registers: Apple GPUs
cannot index registers at run time, and the earlier float32 r3d port (a vertex graph clipped plane by plane) spilled
832 B/thread and was latency-bound. Two rules keep it exact in float: the Green cross term is written so that a
collapsed segment gives exactly 0 under FMA contraction, and close breakpoints are never merged (merging integrated
across kinks and broke thin slivers). Order 1 (pass 0) and order 2 (pass 1) give bit-identical volumes. The whole
PS-DTFE's exact deposit (`--exact-average`) on an M1 Max (one partition): 18.1 -> 8.7 s for 0.26M particles on 256^3, 10.8 -> 6.2 s for
2.1M on 128^3, 35.7 -> 18.1 s for 2.1M on 256^3. The file's header comment has the details.

**The clip's frame.** A piece's moments taken about a far origin cancel (r3d's origin-apex simplices lost the volume
itself). Every GPU exact path clips with the origin at the centre of the overlap of the tetrahedron's bounding box with
the cell and shifts the centroid back; it agrees with the CPU (double r3d, vertex-0 frame) to float rounding at every
resolution (`ps_window_check.sh`'s fine grid, `dtfe_metal_check.sh`'s fine region).

**`--grid-window`.** The sampled kernel visits every cell of a straddling tetrahedron's bounding box and counts the
samples outside the window for its normalization, depositing only those inside. The exact items are built from the
hull of each bounding box intersected with the window's periodic images (`window_hull` in the kernel and `psWindowHull`
on the host: keep them in sync), and the tetrahedron is normalized by its whole weight `wFull`; an empty window piece
then means "no volume inside", so `depositFields` takes no sampled fallback in window mode. On a non-periodic grid a
tetrahedron cut by the grid face keeps only its inside share (`count_beyond_grid` counts the samples beyond the grid
column by column).

## Standard-DTFE `_a` interpolation

Method 1 of `averaged_interpolation_1.cc`: a tetrahedron inside one cell deposits its centroid value times its volume;
a larger one scatters value × V/n at n shared Sobol samples (centred, in the hashed vertex order; up to 10 000). The CPU
precomputes volumes, sample counts and single-cell indices with its own helpers, so both paths make the same decisions.

- **Item kernels** `depositAveraged1Items{2,4,8}` (default 2 slots, `DTFE_GPU_CACHE_K`): one thread per work item =
  (tetrahedron, run of at most 48 samples, `DTFE_GPU_ITEM_SAMPLES`). Items are implicit (a prefix sum and a block table,
  no per-item buffer); a tetrahedron's samples are dealt round-robin to its items, so adjacent lanes read consecutive
  table entries. A per-thread cell cache sums samples that land in the same cell; the host orders tetrahedra along a
  Morton curve of their base cell, and the flush is a SIMD-group reduction by cell (one atomic per field per distinct
  cell per group). The kernels are specialized per field set (function constants `FC_DEN/FC_VEL/FC_GRAD`) and
  register-bound, which is why 2 slots beat 4 and 8 and 48-64 samples per item is the plateau.
- **Evictions go to a threadgroup cell table** (32 KB, open addressing; 1024 entries for density + velocity, 512 with
  the gradient, half that under the exact sums), flushed once per threadgroup; a full probe falls back to device atomics
  (`DTFE_GPU_TABLE=0` turns it off). This matters in voids, where a tetrahedron's samples never repeat a cell within one
  item. On 2.1M particles at 512^3 the largest sub-domain takes 0.7 s on the item kernels and 45 s on the per-tet kernel
  (`depositAveraged1`, `DTFE_GPU_ITEMS=0`), whose 10 000-sample threads trip the watchdog.
- **`--exact-average`** (`depositExactAverage`): the exact deposit's item split (16 cells, `DTFE_GPU_EXACT_CELLS`), each
  piece depositing the interpolant's centroid value and the gradients times its volume, one dispatch. Agrees with the
  CPU to float rounding (density 4e-7, velocity 7e-7, gradient 2e-6 of the peak on the 96^3 test box of
  `dtfe_metal_check.sh`).
- **Sub-domains.** Each OpenMP sub-domain deposits its own sub-grid, serialized on the one queue. The alternative, one
  full grid on the device for the whole call (`DTFE_GPU_SHARED=1`, `DTFE_GPU_SHARED_GB` cap), is tested but off: its
  atomics' footprint on the full grid cost more than the zeroing and read-back it saves (2.1M at 512^3: 12.9 s against
  10.6 s). It keeps float sums and sampled averages only.

The sampled GPU grid agrees with the CPU's except in 0.1-0.2% of the cells, where a sample within float rounding of a
cell face lands in the neighbouring cell (`dtfe_metal_check.sh`: means to 1e-5, cells to 5e-2 outside a 1e-3 fraction).

## Exact sums

Float atomics drop every share below half an ulp of a busy cell's running sum, always downward (0.06% of TNG50-3-Dark
z = 0's box mass), and their order varies, so the grids are fixed-point integer sums instead:
- **PS mass grid**: each share added as rint(share · 2^k) into two `atomic_uint` words per cell with a carry
  (`depositMass`), the partition's mass at 2^61..2^62 units (`psMassFixedScale`; coarser units put a zoom's 3 kpc cells
  5e-4 off).
- **PS moment grids** (velocity, dispersion and its own mean and weight, gradients, scalar): each in its own unit from a
  bound the caller sums (`PSDepositParams.fxMask`, `PSFixedBounds`, `psMomentFixedScale`).
- **Standard `_a` grids**: 64-bit sums with a unit per batch from a bound on its total share (`dtfe_gpu_fixed.h`); the
  item kernel's chunks start at multiples of 256 items (`DTFE_GPU_CHUNK` is rounded up), so the
  SIMD-group sums are fixed too.

The hosts hand the words over unconverted (`PSGpuGrids::massFx`); the CPU adds the deferred tetrahedra and the
`--ps-halo-release` centroid deposits as the same integers and converts once. So every grid is identical from run to
run whatever the order of the atomics (`gpu_mass_check.sh`, `ps_fixed_moments_check.sh`, `dtfe_gpu_retry_check.sh`);
the remaining differences against the CPU are its float arithmetic (~1e-6 of the peak). A share without an integer
value (NaN, infinity, beyond 2^63 units) is left out and counted, and the run warns per grid (`DTFE_TEST_NAN_SHARE`).
A double build converts the words to double once. The float modes stay for A/B: `DTFE_GPU_FLOAT_MASS=1`,
`DTFE_GPU_FLOAT_MOMENTS=1`, `DTFE_GPU_FLOAT_GRIDS=1` (and `DTFE_GPU_SHARED=1`),
all run-to-run variable at the ulp level. Each dispersion share
w·v_i·v_j is a float32 product before the sum, as on the CPU path.

## Pacing and retries

macOS kills command buffers when sustained GPU load starves the display
(`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`); a pipelined submission was killed even with ~0.1 s
buffers, serialized submission with small gaps survives. So the hosts submit serialized chunks that adapt toward ~0.25 s each (PS start 25 000 tetrahedra, 500 for the exact deposit;
`PS_GPU_CHUNK`, `DTFE_GPU_CHUNK`) with an ~8 ms gap. A killed buffer may have deposited part of its atomics, so a chunk is never retried alone: each pass keeps a
host snapshot of its grids at its start and every ~10 s of GPU work (`DTFE_GPU_CHECKPOINT_CHUNKS=n`: every n buffers,
0 = the pass start only), and a failure restores the last snapshot and continues (the standard-DTFE host first
drains its queue with an empty buffer, as a killed buffer's atomics can land late; the PS host restores at once). An "innocent
victim" error (another process hung the GPU) retries at once; a watchdog or memory error makes the process-wide pace
4x shorter buffers with 3x wider gaps, easing back one notch after 90 s of clean GPU work
(`DTFE_GPU_PACE_RECOVER_SEC`). Three failures of one chunk, sixteen in one deposit or a lost device hand the partition
(standard DTFE: the sub-domain) to the CPU. `DTFE_GPU_FAULT_AT=<k>[:innocent|timeout|oom|persist]` makes the k-th buffer fail after it ran (the retry
suites). Running with the display idle avoids the watchdog. Both hosts run their command buffers through ONE loop,
`gpuRunPass` in `gpu_metal_retry.h` (`ps_metal_host.cc` `runChunked` per pass, `dtfe_metal_host.cc` `runBatch` per
batch); the retry suites check it through both.

## Harness tests (`metal/validate_deposit.cpp`)

T1-T15: analytic edge cases (a tetrahedron in one cell, the tiny-tet centroid fallback, periodic-wrap shift
equivalence, the degenerate-tet drop), nSub 1-4 mass conservation, determinism, a random CPU-vs-GPU stress set on every
grid, analytic velocity fields (constant and linear v = Gx), the partition sub-grid (T10), the volume-weighted (T11),
caustic (T12) and exact (T13) paths, a sheared Zel'dovich pancake against an exact piecewise-linear pushforward
reference, `depositExactItems` against the analytic cube and the one-thread path (T14: periodic, wrapped sub-grid,
non-periodic; the items' map with tetrahedra that have no items, run in chunks as the hosts run them) and the inside share of a tetrahedron cut by a non-periodic face (T15). It builds the per-tet records like
the host and deposits deferred tetrahedra with its own CPU reference, whose float inside test with a ±1e-6 tolerance
differs from the kernel's exact decisions in 0.003% of the stream counts (the production CPU deposit agrees exactly,
`point_exact_serve_check.sh` D3/D4). It skips zero-mass tetrahedra, which production deposits with weight 0 into the
stream counts. A test lattice must not be grid-commensurate (samples on shared faces, moiré per slab): the harness
decommensurates the periods, offsets the phases and adds an affine shear while keeping the reference exact.

## Knobs

Environment variables, read by the Metal hosts with the rule of the README's *Environment variables of the binaries*
(an empty value is unset; a switch is off when set to `0`, `false`, `no` or `off`, on otherwise). One name per
concept in both binaries; the Metal spellings of earlier releases (`*_METAL_*`) are no longer read. Two PS-DTFE knobs
keep names of their own because standard DTFE's twins mean something else: `PS_GPU_CHUNK` (tetrahedra; standard
`DTFE_GPU_CHUNK` counts work items) and `PS_GPU_DEBUG_NOATOMIC` (a switch; standard's is a level 1-3).
The older names of the renamed ones are deprecated, read for one release (one warning line at start-up) while the
new name is unset: `PS_EXACT_ITEMS` and `PS_EXACT_ITEM_CELLS` (deprecated: `DTFE_GPU_ITEMS`, `DTFE_GPU_EXACT_CELLS`)
and `PS_GPU_TIMING`, `PS_GPU_FLOAT_MASS`, `PS_GPU_FLOAT_MOMENTS` (deprecated: `DTFE_GPU_TIMING`, `DTFE_GPU_FLOAT_MASS`,
`DTFE_GPU_FLOAT_MOMENTS`).

| variable | effect | default |
|---|---|---|
| `DTFE_GPU_ITEMS` | `0`: standard DTFE runs the per-tet kernel instead of the item kernels, PS-DTFE the one-thread-per-tetrahedron exact deposit (A/B) | on |
| `DTFE_GPU_ITEM_SAMPLES=n` | samples per work item of the item kernels (4-100000) | 48 |
| `DTFE_GPU_CACHE_K=2\|4\|8` | slots of the item kernels' per-thread cell cache | 2 |
| `DTFE_GPU_TABLE` | `0` turns the threadgroup cell table off (evictions go to device atomics) | on |
| `DTFE_GPU_EXACT_CELLS=n` | cells per `--exact-average` item (both binaries) | 16 |
| `DTFE_GPU_SHARED` | one shared full grid on the device for the whole call (sampled, float sums) | off |
| `DTFE_GPU_SHARED_GB` | the shared grid's cap in GB | 25% of RAM |
| `DTFE_GPU_CHUNK` | standard DTFE: the opening chunk (the item kernels round it up to a multiple of 256) | 65536 items; per-tet kernel 25088 tetrahedra, exact 8192 items |
| `PS_GPU_CHUNK` | PS-DTFE: the per-tet kernel's opening chunk in tetrahedra | 25000; exact deposit 500 |
| `DTFE_GPU_TIMING` | standard DTFE: one line per batch, and the Morton order's cell sharing; PS-DTFE: one line per kernel run, the exact items and the checkpoints, and per partition how long its deposit waited for the GPU and held it, and where its time went: before the hold (device setup, plan, the exact items' bounds and map) and in it (uploads, kernels, pacing gaps, snapshots, leftovers, copy-back, releases) | off |
| `DTFE_GPU_FLOAT_MASS` | PS-DTFE: the float mass grid on GPU and CPU, for A/B (see Exact sums) | off |
| `DTFE_GPU_FLOAT_MOMENTS` | PS-DTFE: the float moment sums on GPU and CPU, for A/B | off |
| `DTFE_GPU_FLOAT_GRIDS` | standard DTFE: float atomics for the `_a` grids, for A/B | off |
| `DTFE_GPU_CHECKPOINT_CHUNKS=n` | a grid snapshot every n buffers; 0 = at the pass start only | every ~10 s of GPU work |
| `DTFE_GPU_PACE_RECOVER_SEC=s` | clean GPU work after which an escalated pace eases back one notch | 90 |

For testing and debugging only: `DTFE_GPU_DEBUG_NOWRITE` (a timing experiment: every kernel does all its work but
writes no grid), `DTFE_GPU_DEBUG_NOATOMIC=1|2|3` (timing: 1 issues no atomics in the final reduction, 2 also skips the
gradient's reduction loop, 3 also the gradient's eviction atomics; the grids come out wrong), its PS-DTFE counterpart
`PS_GPU_DEBUG_NOATOMIC=1` (timing, the atomics' share of the sampled kernel: every sample is classified and counted
but none is deposited; the grids come out empty),
`DTFE_GPU_FAULT_AT=k[:innocent|timeout|oom|persist]` (the k-th buffer fails after it ran, for the retry suites),
`DTFE_TEST_NAN_SHARE=k` (every k-th share becomes NaN, for the exact sums' guard), and the harness's `BENCH`, `DUMP_PROFILES`, `VALIDATE_FLOAT_MASS` (Build and
test).
