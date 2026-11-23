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
| `DTFE` | Exact | GPU | 429.04 s (7 min 9 s) | 23.83 GB | `--partition 3 3 3`, one tessellation per partition | 10 files, 16.1 GB |
| `DTFE` | Fast | CPU | 353.81 s (5 min 54 s) | 30.44 GB | `--partition 2 2 2`, 6 threads per partition | 10 files, 16.1 GB |

The wall time is that of the whole command, from reading the input to the last written grid. The peak memory is the
process's peak memory footprint (`/usr/bin/time -l`). PS-DTFE kept its full grids in files under `--scratch-dir`
while it ran (38.1 GB in Exact and 27.4 GB in Fast, as its `AUTO-TUNE:` line counts them), and memory held in those
files is not part of that footprint.

No command gave `--partition` or `--max-concurrent`: each program chose its plan from the memory budget and printed
it in its `AUTO-TUNE:` line ([user guide, section 11](user-guide.md#11-memory-and-partitions)). PS-DTFE builds
several partitions at once; standard DTFE runs them one after another. Its Exact run, on the GPU, builds one
tessellation per partition on all of the run mode's threads, the next partition's during this one's deposit; its Fast
run splits each partition over 6 threads. Standard DTFE's Exact run peaked at 23.8 GB against the 29.5 GB it planned
for: its memory model books the next partition's build whole beside the busiest partition's deposit, where the two
overlap only in part (user guide, section 11).

The machine is an Apple M1 Max with 8 performance and 2 efficiency cores and 64 GiB (68.7 GB) of memory, under
macOS 27.3. The programs are a `METAL=1` build and ran in the default run mode (`balanced`, 6 of the 10 threads) with
the default memory budget (60% of the memory, 41.2 GB; for standard DTFE's runs other programs left less free, and
Exact had 37.2 GB, with the plan it makes at 41.2 GB). The snapshot was on an external SSD; PS-DTFE's output was
there too, its scratch files on the internal disk, and standard DTFE's output on the internal disk. A screen saver and
the system's photo analysis were running on the machine while PS-DTFE's runs went.

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

- The standard DTFE rows are from a build of the evening of 23 November, run back to back with the build of the
  morning, the same commands: Exact 429.04 s at 23.83 GB against 499.07 s at 31.84 GB, its grids byte for byte the
  same, and Fast 353.81 s against 357.19 s, the same plan, at a peak of 30.44 GB against 34.17 GB. With the 35.8 GB
  that other programs left it that night, the morning build split each Exact partition over 6 threads without the
  look-ahead; earlier that day, with 41.2 GB, it chose 4 threads with the look-ahead and took 419.04 s at 41.85 GB,
  and a build between the two (one tessellation per partition, before the memory model's refit) took 407.12 s at
  26.10 GB.
- An earlier build of the programs, from 16 November, ran back to back with the build of PS-DTFE's rows, which also
  made the standard rows before 23 November (Exact 485.27 s at 41.46 GB, Fast 431.96 s at 34.39 GB), with the same
  plans and the same commands (its PS-DTFE also took the option for volume-weighted velocity moments, the only
  behaviour now). It took 1524.78, 570.31, 497.07 and 430.88 s, so those four times differ from its by +1.6, +1.0,
  -2.4 and +0.3%.
- Two runs of one command with one program differ by about as much. Earlier the same night that earlier build ran
  the last three commands twice each: 577.08 and 560.44 s, 492.40 and 480.89 s, 436.38 and 426.24 s, a difference of
  2.4 to 3.0%.
- Other nights have been faster. On 14 November a still earlier build, whose grids are byte for byte those of the
  build of 16 November, made the same four runs with the same plans in 1327.94, 552.77, 450.64 and 407.73 s.
- A build of 23 November, whose exact GPU kernels test a run of cells before they clip any, ran the PS-DTFE Exact
  command with its plan given in 1199 s, against 1292 s for the build before it the same night. The table's PS-DTFE
  Exact time is that of the earlier build.

So on this machine PS-DTFE's Exact mode takes 20 to 26 minutes for this snapshot and its Fast mode 9 to 10 minutes;
standard DTFE takes about 7 minutes in Exact on the GPU and about 6 minutes in Fast on the CPU.
