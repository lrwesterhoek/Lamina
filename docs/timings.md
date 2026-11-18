# Timings

How long the two modes of each program take on one full-size snapshot, and how much memory they use. Every number
here is one run on one machine, measured in November 2026: a scale to plan with, not a benchmark. The two modes are
described in the [user guide](user-guide.md#5-the-two-modes-exact-and-fast).

## The four runs

The snapshot is TNG100-3-Dark at redshift 0: 94,196,375 dark-matter particles in a periodic box of 75 Mpc/h, put on
a 512^3 grid.

| Program | Mode | Grid work on | Wall time | Peak memory | The plan the program chose | Output |
|---|---|---|---|---|---|---|
| `PS-DTFE` | Exact | GPU | 1549.55 s (25 min 50 s) | 34.08 GB | `--partition 6 6 6 --max-concurrent 4` | 28 files, 35.4 GB |
| `PS-DTFE` | Fast | GPU | 576.01 s (9 min 36 s) | 26.21 GB | `--partition 6 6 6 --max-concurrent 4` | 22 files, 24.7 GB |
| `DTFE` | Exact | GPU | 485.27 s (8 min 5 s) | 41.46 GB | `--partition 3 3 3`, 4 threads per partition | 10 files, 16.1 GB |
| `DTFE` | Fast | CPU | 431.96 s (7 min 12 s) | 34.39 GB | `--partition 2 2 2`, 6 threads per partition | 10 files, 16.1 GB |

The wall time is that of the whole command, from reading the input to the last written grid. The peak memory is the
process's peak memory footprint (`/usr/bin/time -l`). PS-DTFE kept its full grids in files under `--scratch-dir`
while it ran (38.1 GB in Exact and 27.4 GB in Fast, as its `AUTO-TUNE:` line counts them), and memory held in those
files is not part of that footprint.

No command gave `--partition` or `--max-concurrent`: each program chose its plan from the memory budget and printed
it in its `AUTO-TUNE:` line ([user guide, section 11](user-guide.md#11-memory-and-partitions)). PS-DTFE builds
several partitions at once; standard DTFE runs them one after another and splits each over its threads. Standard
DTFE's Exact run took 4 threads per partition where the run mode allows 6, because with 6 the busiest partition and
the next one's look-ahead build would not have fitted the budget; it peaked 0.3 GB above the 41.2 GB it planned
for.

The machine is an Apple M1 Max with 8 performance and 2 efficiency cores and 64 GiB (68.7 GB) of memory, under
macOS 27.3. The programs are a `METAL=1` build and ran in the default run mode (`balanced`, 6 of the 10 threads) with
the default memory budget (60% of the memory, 41.2 GB). The snapshot and the output were on an external SSD, the
scratch files on the internal one. A screen saver and the system's photo analysis were running on the machine while
the runs went.

## The commands

PS-DTFE, with the mode on the last line:

```
./PS-DTFE <snapshot> <output root> --grid 512 --padding 50 --periodic --scratch-dir <folder> \
    --ps-merge index-order --input-format 105 --mpc-unit 1000 \
    --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a tweb_a vweb_a \
    --lambda-th 0.3 --ps-vertex-mass \
    --sample-points <points file> --pts-vel-grad --ps-caustics --ps-lagrangian-input <initial conditions> \
    --gpu --gpu-share auto --run-mode balanced \
    --exact --face-gradient
```

Fast ends in `--sampled --subsamples 1` instead of `--exact --face-gradient`. Beside the grids, both commands
evaluate the fields with their velocity gradients at 16,777,216 sample points (a plane of 4096^2), flag the caustics
and read the particles' initial positions from a second file.

Standard DTFE, with the mode on the last line:

```
./DTFE <snapshot> <output root> --grid 512 --padding 15 --periodic \
    --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a tweb_a vweb_a \
    --lambda-th 0.3 --run-mode balanced \
    --exact --gpu
```

Fast ends in `--sampled` instead of `--exact --gpu`: its Monte-Carlo average runs on the CPU. How the two modes of
standard DTFE compare on a build without the GPU is in the [design decisions](decisions.md).

The commands are printed as they were run, and several of their options only name a default (`--exact`, `--gpu`,
`--run-mode balanced`, `--mpc-unit 1000`, `--lambda-th 0.3`, standard DTFE's `--padding 15`). The times are those of
full production runs, so the two programs' rows are not like for like: the standard DTFE commands evaluate no sample
points and write no dispersion or caustic grids.

## How much the times move

- An earlier build of the programs, from 16 November, ran back to back with the one in the table, with the same
  plans and the same commands (its PS-DTFE also took the option for volume-weighted velocity moments, the only
  behaviour now). It took 1524.78, 570.31, 497.07 and 430.88 s, so the four times in the table differ from them by
  +1.6, +1.0, -2.4 and +0.3%.
- Two runs of one command with one program differ by about as much. Earlier the same night that earlier build ran
  the last three commands twice each: 577.08 and 560.44 s, 492.40 and 480.89 s, 436.38 and 426.24 s, a difference of
  2.4 to 3.0%.
- Other nights have been faster. On 14 November a still earlier build, whose grids are byte for byte those of the
  build of 16 November, made the same four runs with the same plans in 1327.94, 552.77, 450.64 and 407.73 s.

So on this machine PS-DTFE's Exact mode takes 22 to 26 minutes for this snapshot and its Fast mode 9 to 10 minutes;
standard DTFE takes about 8 minutes in Exact on the GPU and about 7 minutes in Fast on the CPU.
