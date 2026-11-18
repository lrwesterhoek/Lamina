# Design decisions

The choices that shape the code: what it does and why. Change one only with its reason in view. What was considered
and not done closes the page.

## Modes and options

- **Two modes per estimator**, Exact (`--exact`) and Fast (`--sampled`), and no other user option for accuracy or for
  a faster way to the same result. Exact is the default: no sampling noise, and the mass is conserved.
- **PS-DTFE's Fast is one point per cell axis, in 2D as in 3D.** Measured in 3D, the sampled deposit at three points
  per axis took longer than the exact one, so only the cheapest count earns a mode; `--subsamples` keeps the count
  selectable. The 2D programs take the same count: a mode names one deposit rule in both dimensions, and a count of
  their own would be a choice that no measurement supports.
- **Standard DTFE keeps its Fast mode in the launcher.** On a CPU-only build at 512^3, Exact took 615.71 and
  613.76 s against Fast's 423.65 and 426.53 s, so Fast is the quicker way to a standard-DTFE grid set and earns its
  place beside Exact.
- **An A/B switch is scaffolding**: measured once, then hard-wired or removed. What stays for tests and comparisons
  is listed in the developer group of `--full-help`.
- **One name per thing**: lower-case kebab-case options, `ps-` exactly for phase-space concepts, and a driver knob
  that stands for one option is that option in upper snake case (`PS_MERGE`, `GPU_SHARE`, `MAX_CONCURRENT`, ...),
  but for `GRID_SIZE` (`--grid`), `FIELDS` (`--field`) and the knobs that carry the prefix `DTFE_` (`DTFE_RESUME`
  for `--resume`). Renamed options keep no alias and no message (the release notes have the table), and no test
  checks that an old name is refused, since such a test would keep the old names in the tree.
- **The programs' own defaults are the production settings**, so a bare run needs no flag for them: the cell
  average, the web threshold (`--lambda-th` 0.3), the run mode, the device, and standard DTFE's `--padding` of 15
  mean particle spacings (`run_dtfe.sh` passes the same), since 5 leaves the emptiest voids partly outside the
  tessellation. PS-DTFE's own padding default is 5, as it pads in Lagrangian space (at least a tenth of the box) and
  its partitions by `--ps-lagrangian-padding`; `run_ps_dtfe.sh` passes `PADDING` 50.
- **`--ps-vertex-mass` stays an option**: it describes the input (perturbed initial positions), not a quality.
- **Every run names its data**: no default data root or simulation (`DTFE_DATA_ROOT`, `DTFE_SIM`).

## Device and machine

- **The program picks its device** from one table, and nothing else holds a copy of the rule: the drivers pass on
  what the caller set, and the launcher passes `DTFE_GPU=auto` unless its Advanced settings name a device, then
  shows the device the program reports. PS-DTFE deposits on the GPU wherever its deposit can run there, averaged or
  not, so a bare run, a driver run and a launcher run take the same device; standard DTFE's sampled average runs on
  the CPU, where it was faster when measured.
- **Small PS-DTFE Fast runs take the GPU** like every other PS-DTFE deposit that can: the device rule has no size
  threshold. Measured medians on this project's machine, CPU against GPU in seconds: 64^3 particles on a 128^3 grid
  1.45 against 2.80, on 256^3 4.17 against 5.08; 96^3 on 128^3 3.42 against 3.43, on 256^3 5.20 against 7.98;
  128^3 on 128^3 6.69 against 6.75, on 256^3 8.81 against 7.98. The GPU costs at most about 3 s on runs of seconds
  and is ahead in the largest case measured, and a size rule would be one more thing to keep in step.
- **Balanced is the default run mode**, so the machine stays usable; inside a batch job it is performance. The GPU
  share follows the mode (`--gpu-share auto`): only performance mixes the GPU and the free cores.
- **Metal is the only GPU backend**: the CUDA and HIP ports never ran on NVIDIA or AMD hardware and are kept on the
  archive branch the Makefile's error names. Linux builds run on the CPU.
- **A pass of 2^32 cells or more runs on the CPU**, with a warning: the GPU kernels number cells in 32 bits.

## Results and reproducibility

- **The face gradient is the velocity-derivative route** in 3D wherever it applies (standard DTFE always; PS-DTFE
  with the exact deposit and without `--ps-halo-release`): a conservative gradient of the mean velocity, the same
  quantity in both programs; standard DTFE gets the same gradient grids from every average and device.
  `--per-tet-gradient` stays for comparisons and is the only route in 2D.
- **A PS-DTFE run with the face gradient writes both derivatives**, the mean flow's under the plain names and the
  stream average beside them (`*Streams`), since folds make them differ; any other PS-DTFE run writes the stream
  average under the plain names. PS-DTFE's velocity moments are volume-weighted (the standard DTFE convention, which
  the face gradient needs) and its dispersion mass-weighted.
- **Grids are sums of integers.** In single precision PS-DTFE's grids and standard DTFE's exact grids are fixed-point
  sums that repeat to the byte at any thread count; PS-DTFE renumbers each tessellation in a canonical vertex order,
  so this holds under the parallel insert.
- **Standard DTFE's unit spans the whole run**, so its exact grids do not depend on the split. PS-DTFE sets its unit
  per pass and repeats to the byte at a given split only: a run-wide unit with an exact integer merge cost more wall
  time than was accepted.
- **`run_ps_dtfe.sh` and the launcher name `--ps-merge index-order`**, the default: named, it also keeps the memory
  gate on under the auto-tuner's speed split of small runs.
- **A run log without the header's `run mode` line is not read**: the pipeline recomputes such a snapshot rather than
  keep readers of old spellings in the code.
- **Explore draws the void catalogue's distinct sample only**, so a deep minimum outside it gets no ellipse. The
  distinct sample is the one the analysis counts and stacks (`distinct` in `python/dtfelib/catalog.py`: well
  resolved, with an R_v, its centre inside no deeper void's sphere), so the map shows the voids the profiles and
  abundances are made of, and a minimum outside it is no void of the analysis.
- **The deposit loops and the GPU kernels change only with a timing**: every edit there, a removal of unreachable
  code too, is timed against the binary before it, and one that costs time is not made.
- **PS-DTFE's kernel keeps the branches of the removed switches** (the linear deposit, the mass-weighted moments,
  the 128-bit words and a debug flag; [metal/README.md](../metal/README.md) lists them). A kernel without them was
  built and measured. It wrote the same grids only while one never-taken arm stayed as the summing pass's reader of
  the clip's first moments (without that arm the exact GPU grids came out different; the likely reason is that
  Metal's fast math then compiles that pass's volume sum differently from the deposit pass's), and its exact
  deposit pass took about a tenth more device time on the same partition. By the rule above it was not made.
- **The upstream DTFE code is ordinary code**, edited and removed like any other.

## Considered and not done

- A hybrid GPU and CPU deposit by tetrahedron class, or a CPU share of the GPU's tetrahedra: the measured gain was
  small. The split between the devices is by partition (`--gpu-share`).
- Narrowing the GPU lock for sampled runs: a deposit spends little time under the lock beyond the kernel.
- GPU exact-deposit variants, measured and rejected: a column cache, polyhedra in threadgroup memory, one-pass piece
  storage, a full separating-axis rejection, a register port of r3d.
- Skipping the exact deposit's first pass with the analytic tetrahedron volume: it changes bytes, leaves a small mass
  leak per tetrahedron and removes the route for degenerate tetrahedra.
- Shorter GPU pacing gaps: little gain, against the display watchdog.
- Painting the 3D view's picture in threads, processes or tiles: the GIL and the memory bandwidth limit it, and tiles
  change the pixels; the OpenGL renderer covers the on-screen case.
- `logging` instead of `print` in dtfelib and the plot scripts: the launcher parses their output.
- A shared helper for the Python blocks inside the shell tests: boilerplate only, against re-proving every suite.
