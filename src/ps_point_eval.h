/*
 *  Copyright (c) 2011       Marius Cautun
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *  Copyright (c) 2026       Luuk Westerhoek
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* Arbitrary-point evaluation (--sample-points): CGAL-free entry points shared by BOTH binaries. PS-DTFE evaluates the
   multi-stream fields (one stream per folded tetrahedron containing the point); the standard binary the Eulerian DTFE
   interpolant (one containing tetrahedron). Implemented in CGAL_triangulation/ps_point_eval.cc; the per-triangulation
   workers are declared in triangulation.cpp. Everything is a no-op unless --sample-points is given. */

#ifndef PS_POINT_EVAL_HEADER
#define PS_POINT_EVAL_HEADER

#include <functional>
#include <string>
#include <vector>

struct User_options;
struct Particle_data;

// True once psPointEvalInit() has loaded query points (i.e. --sample-points is active).
bool psPointEvalActive();

// Reads the --sample-points file, wraps the points into the periodic box, and builds the
// point bucket index. Call once, after the run options (region, averageDensity, periodic)
// are final and before any interpolation.
void psPointEvalInit(User_options &userOptions);

// Sorts each point's accumulated stream records (density descending) and reduces them to the
// per-point outputs: total density, mean velocity over the streams, velocity dispersion tensor,
// stream count. Call once, after ALL partitions have been processed.
void psPointEvalFinalize(User_options const &userOptions);

// Writes the finalized point outputs next to the grid outputs ('<root>.pts_*'). See the
// README (PS-DTFE point evaluation) for the exact file layout.
void psPointEvalWriteOutputs(User_options const &userOptions);

// '--resume' of a partitioned '--sample-points' run: the stream records belong to the one partition that owns their
// tetrahedron, so they are kept and checkpointed per partition, as '<dir>/points.<i>' files (temp file, fsync,
// rename). Load fails (with why) when a file is missing or not this run's: keyHash is the checkpoint key's FNV-1a
// hash, so a points file cannot be read into another run's grids.
void   psPointEvalSetPartition(int partition);
double psPointEvalUnsavedBytes(std::vector<char> const &merged, std::vector<char> const &saved);
bool   psPointEvalSavePartitions(std::string const &dir, std::string const &keyHash, std::vector<char> const &merged,
                                 std::vector<char> &saved, std::string &why);
bool   psPointEvalLoadPartitions(std::string const &dir, std::string const &keyHash, std::vector<char> const &merged,
                                 std::string &why);
void   psPointEvalRemoveUnsaved(std::string const &dir, std::vector<char> const &saved);

// --serve: moves the binary protocol onto a private duplicate of stdout and points stdout at stderr, so the protocol
// stream stays clean. Call before ANYTHING is printed (main() does, when '--serve' is on the command line).
void psServeRedirectStdout();

// --serve with --partition: a composite server of partition tessellations (ps_serve.cc). DTFE() calls Begin once,
// AddPartition once per partition (concurrent OpenMP threads are fine: each writes its own slot), then Run.
// selectParticles is called only when the tessellation must be built; ownLo/ownHi are the standard binary's owned
// Eulerian box (null in PS-DTFE).
void psServeCompositeBegin(User_options const &userOptions, int totalPartitions);
void psServeCompositeAddPartition(int index, User_options &partOptions,
                                  std::function<void(std::vector<Particle_data>&)> const &selectParticles,
                                  double const *ownLo, double const *ownHi);
[[noreturn]] void psServeCompositeRun(User_options &userOptions);

// Standard binary, PARTITIONED point evaluation (composite '--serve' and a batch '--sample-points' run with
// '--partition'): a tetrahedron belongs to the partition whose Eulerian sub-box holds its centroid, so a point is
// answered by exactly one partition. psPointEvalOwnBox turns a sub-box into that ownership box: outerLo/outerHi[d] say
// its face on axis d is a face of the run's whole box (decided from the split's GRID indices, never by comparing float
// coordinates). An outer face is periodic as the primary box's face, or non-periodic open to infinity (hull tets keep an owner).
void psPointEvalOwnBox(double const lo[], double const hi[], bool const outerLo[], bool const outerHi[],
                       double ownLo[], double ownHi[]);

// The ownership the standard worker applies to the triangulation(s) evaluated from now on: one partition's box
// (psPointEvalOwnBox) while DTFE()'s SERIAL standard partition loop evaluates it; null restores the single-
// triangulation default (periodic: the primary box; otherwise every tetrahedron). Not for triangulations evaluated
// concurrently -- the composite server passes each partition's box with the partition instead. A no-op in PS-DTFE,
// which owns by the Lagrangian centroid (the partition's lagrangianRegion).
void psPointEvalSetOwnership(double const *ownLo, double const *ownHi);

// --grid-window in a batch partition run: 1 when partition 'partOpts' can reach the window of
// 'mainOpts' by its occupancy map (beside its cached tessellation), 0 when it cannot (skip it),
// -1 when no usable map exists (process it). The map writer, which needs the tessellation type,
// psWriteOccupancyIfMissing(DT&, User_options&), is declared with the workers (triangulation.cpp).
int psOccupancyTouchesWindow(User_options const &partOpts, User_options const &mainOpts);

// Is this run's / partition's tessellation in '--tessellation-cache' (its header matches the full
// descriptor)? The batch partition loop then skips the particle selection.
bool psTessellationCached(User_options const &opts);

// The deposit passes since the last reset that were meant for the GPU and ran on the CPU instead (ps_interpolation.cc);
// reset = true also zeroes the count. DTFE() reports it after the loop (the partition loop's threads run at verbose 0).
int psGpuFallbacksTaken(bool reset = false);

// The deposit passes since the last reset whose face walk (--face-gradient) ran beside the GPU deposit
// (ps_interpolation.cc PSEarlyFaceWalk): walks kept, walks thrown away after a failed device call, the kept walks'
// summed wall time and the part their passes still waited; reset = true zeroes them. DTFE() reports the sums after
// the loop, since the partition loop suppresses the passes' own lines.
void psFaceWalksEarlyTaken(long long &kept, long long &discarded, double &walkSec, double &waitSec, bool reset = false);

// Auto-tune: the mean number of streams at a random point of the box (sum|V_Euler| over sum V_Lagrange), estimated from
// up to 256 random Lagrangian patches before any triangulation exists. Needs Lagrangian positions (phase-space runs).
// Returns < 0 when it cannot tell (too few particles, 2D); *patchesUsed = the patches that contributed. Deterministic.
// regionLo/regionHi (optional): the mean over that EULERIAN box (a '--sample-points' slab) instead; only patches whose
// Eulerian image meets it are triangulated; falls back to the box mean when fewer than 16 patches meet it.
double psEstimateMeanStreams(std::vector<Particle_data> const &particles, double const boxLo[3],
                             double const boxLen[3], bool periodic, int *patchesUsed,
                             double const *regionLo = nullptr, double const *regionHi = nullptr);

#endif
