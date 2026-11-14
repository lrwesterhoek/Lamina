/* The PS-DTFE GPU kernels' parameter blocks: ONE definition for the hosts and the Metal kernels.
   The hosts (ps_metal_host.cc, metal/validate_deposit.cpp) include it through ps_deposit_params.h
   (PSDepositParams); the Makefile PREPENDS this file to metal/ps_deposit.metal, after exact_clip.metal.inc and
   fixed_words.metal.inc, and to metal/ps_face.metal (the face kernel), after fixed_words.metal.inc, when it embeds
   the kernel sources (metal/validate_deposit.cpp and metal/build_prototype.sh do the same), and the kernels name the
   blocks DepositParams, ExactItemParams and FaceParams. So the struct the kernels read cannot drift from the one the
   hosts fill. The file is plain C++ that is also valid MSL: fixed-width scalar types (MSL has int32_t and
   uint32_t), arrays and no vector types (a float3 would be padded to 16 bytes), no includes under MSL. Its
   static_asserts run in both compilers and pin the layout the field offsets below imply.
   Each field's buffer numbers are those of metal/ps_deposit.metal (PSFaceParams: metal/ps_face.metal). */
#ifndef PS_KERNEL_PARAMS_H
#define PS_KERNEL_PARAMS_H

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

// depositFields' and depositExactItems' parameters (Metal buffer 8)
struct PSDepositParams
{
    float    boxLo[3];
    float    dx[3];
    int32_t  nGrid[3];
    int32_t  nSub;
    int32_t  periodic;     // 0 / 1
    // Partition sub-grid (production PS-DTFE stores only the Eulerian bbox each Lagrangian
    // partition touches): output arrays are subDims-sized; a wrapped cell w maps to
    // loc = (w - subOrigin + nGrid) % nGrid -- the box may wrap a periodic axis -- and cells
    // outside [0,subDims) are skipped BEFORE sample counting, exactly like the inSub guard in
    // ps_interpolation.cc. Full grid: subOrigin=0, subDims=nGrid.
    int32_t  subOrigin[3];
    int32_t  subDims[3];
    // Field flags: which moment grids this run actually needs. When a flag is 0 the host binds
    // a 4-byte dummy buffer for that grid and the kernel never touches it (mass and streams are
    // always deposited). fVel = velocity or dispersion selected (mom), fDisp = dispersion (m2,
    // 24 B per cell), fGrad = velocity gradient (grad, 36 B per cell).
    int32_t  fVel;
    int32_t  fDisp;
    int32_t  fGrad;
    // --ps-linear-deposit: weight the interior samples by the DTFE-interpolated linear density
    // (from the per-tet vertex densities in buffer 9), renormalized per tet so the deposited
    // total still equals the tet mass exactly. 0 = uniform equal shares (buffer 9 is a dummy).
    int32_t  fLinear;
    // --ps-volume-weighted: the velocity/dispersion/gradient moments carry equal EULERIAN-VOLUME
    // shares |det(Ax)|/6 / N instead of the mass shares (the mass grid always keeps mass), and
    // their normalizer is deposited into the momw grid (buffer 10; dummy when 0). Mutually
    // exclusive with fLinear (rejected at option parsing).
    int32_t  fVolW;
    // --ps-caustics: atomic-OR the tet's orientation bits (1 = det(Ax)>0, 2 = det<0) into the
    // caustic grid (buffer 11; dummy when 0) at every deposited sample. OR is commutative and
    // idempotent, so the result is chunk-, thread- and partition-order invariant.
    int32_t  fCaustic;
    // --exact: replace the nSub^3 sub-sampled deposit by analytic tet-cell clipping
    // (sweep_moments in exact_clip.metal.inc, see exact_cell_clip in metal/ps_deposit.metal). Ignores nSub; a tet
    // whose whole clip window comes up empty (sliver below float resolution) falls through to
    // the sampled path, whose empty window then drives the centroid fallback -- the same
    // fall-through the CPU exact deposit uses.
    int32_t  fExact;
    uint32_t nTet;
    // The scalar field (builds with ONE scalar component, NO_SCALARS=1; the callers decline the GPU
    // otherwise): the per-tet vertex scalars arrive in buffer 18 (nTet*4; a dummy when both flags are
    // 0) and are deposited exactly like the velocity moment -- the linear profile's value at the
    // sample (or at the piece centroid) times the moment weight -- into buffer 19 (nCell), and the
    // tet's constant scalar gradient times the moment weight into buffer 20 (nCell*3, [flat*3+i] =
    // ds/dx_i, the CPU's scalar_gradient layout for one component).
    int32_t  fScal;
    int32_t  fSGrad;
    // --grid-window: the sub-box (subOrigin/subDims) is a WINDOW of the full grid, not a partition's own
    // box: a tet straddling its edge counts every in-grid sample (pass 0 / phase 0 sumW) exactly as in
    // the full run and deposits only the samples / pieces inside, so the window equals the full run.
    int32_t  windowMode;
    // The mass grid's accumulator. massFixed = 1: every mass share is added as the
    // integer rint(share * massScale) to massFx (buffer 21: two uint words per cell, low then high, the
    // carry done by hand; or four limbs under fxWords 2, below) -- exact, and independent of the order the atomics
    // land in. Float atomics into 'mass' (buffer 3) drop every share below half an ulp of a busy cell's running sum,
    // always downward: 0.06% of the box mass at TNG50-3-Dark z=0, percents in halo-core cells with 1e7 deposits.
    // massFixed = 0 keeps that float grid (DTFE_GPU_FLOAT_MASS=1, for comparison); the unused one is a
    // 4-byte dummy. (The moment grids: fxMask below.)
    int32_t  massFixed;
    float    massScale;    // 2^k (the run's unit, or ps_deposit_params.h psMassFixedScale), so the product is exact
    // The MOMENT grids (ps_deposit_params.h PSFxGrid) as exact fixed-point sums: rint(value * fxScale[g]) adds
    // to the same words in any atomic order, as the mass does. fxMask bit g = grid g is fixed-point: its words (two
    // uint per component, low then high, 8 B per component; or four limbs, 16 B, under fxWords 2) sit in the buffer
    // of its float index, in units of 1/fxScale[g], a power of two (2^K) from the run's units or from a bound the
    // host keeps every cell below.
    // Bit FX_DW = dispw is not deposited (it equals the mass grid's shares; the host reads the mass words).
    // FX_SUMW_BIT = the exact items' per-tet weight totals are fixed-point too (depositExactItems).
    // 0 = every moment grid a float sum (DTFE_GPU_FLOAT_MOMENTS=1 on the host).
    uint32_t fxMask;
    float    fxScale[9];   // [g], g = PS_FX_MOM ... PS_FX_SG: the PS_FX_N grids (PS_FX_DW unused)
    // PS_GPU_DEBUG_NOATOMIC (a timing experiment, the atomics' share of the sampled kernel): 1 = the sampled path
    // classifies and counts every sample as always but deposits none -- no grid atomics, the grids stay zero
    uint32_t debugNoAtomic;
    // The words of the mass grid (buffer 21) and of every fxMask grid (buffers 22..29). 0 = 64-bit words, two uint
    // per component (the default, or a run without its units: the units of the pass). 2 = the run-wide 128-bit words
    // (DTFE_PS_RUN_UNITS=1): four uint limbs per component, little-endian at [4*at .. 4*at+3]
    // (metal/fixed_words.metal.inc), in the run's units (src/fixed_units.h RunUnits) where it has them. The items'
    // sumW words (FX_SUMW_BIT) stay two uint per tet in both. The Metal kernels take the size at compile time
    // (PS_FX_WORDS: the host compiles one library per size and binds the one that matches this field).
    uint32_t fxWords;
    // fxWords 2: the float bits of the share cap 2^capLog2, one for every PS-DTFE grid (RunUnits::capBits): a share
    // at or above it adds nothing and is counted (shareLimbs); 0 otherwise
    uint32_t fxCapBits;
};
static_assert(sizeof(PSDepositParams) == 43 * 4, "PSDepositParams: 43 four-byte fields, no padding (host == kernel)");

// depositExactItems' per-dispatch parameters (Metal buffer 17, set per command buffer): the items of this chunk,
// the pass (== the pipeline's kItemsPhase, 0 or 1; the kernel reads it only in the kItemsPhase-2 pipeline, which runs
// both passes of --ps-linear-deposit), the cells per item, and badAt = the word of the sumW buffer where the
// non-finite guard's counters start, past the per-tet totals (Metal: every one of the 31 buffer slots is taken, so
// they ride in that buffer's tail).
// itemBase = the chunk's first item and nTet = the tets of the items' map (gpu_host.h psBuildExactItemMap): the
// chunk's thread k runs item itemBase + k, which the kernel decodes from the map (psItemOf)
struct PSExactItemParams
{
    uint32_t nItems;
    uint32_t phase;
    uint32_t blockCells;
    uint32_t badAt;
    uint32_t itemBase;
    uint32_t nTet;
};
static_assert(sizeof(PSExactItemParams) == 24, "PSExactItemParams: six 32-bit words (host == kernel)");

// The exact items' map (gpu_host.h psBuildExactItemMap; the kernels' psItemOf): one block-table entry per
// 2^PS_ITEM_BLOCK_SHIFT items, as the standard-DTFE kernel's ITEM_BLOCK_SHIFT
#define PS_ITEM_BLOCK_SHIFT 5

// ---- The GPU face kernel (DTFE_GPU_FACES=1; metal/ps_face.metal faceSweepItems; docs/cleanup-plan.md P11, the GPU
// kernel for the face sweep, step 3): --face-gradient's face sums in face_sweep_f32.h's canonical float arithmetic,
// one thread per item = (record, axis, plane).
// The face record of one kept tetrahedron (the face walk's set: ps_interpolation.cc psFaceWalk's filter), built by
// the host: PS_FACE_REC_FLOATS floats, 112 B, in the order the walk visits the tetrahedra:
//   [E .. E+11]      eulerPos[v][d] at E + 3v + d: the wrapped Eulerian vertices psDepositFaces takes (Real = float)
//   [VG .. VG+8]     the velocity gradient vG[i][j] = d v_j / d x_i at VG + 3i + j, as the walk forms it
//   [V0 .. V0+2]     the velocity of vertex 0
//   [RHO]            the density rho / rho_bar, the walk's double expression rounded once to float
//   [COUNT .. +2]    the planes the tetrahedron owns per axis, fsPlanes' counts, as uint bits (as_type<uint> on the
//                    device; a memcpy of the uint32_t on the host). Their sum is the record's items in the items'
//                    map (gpu_host.h psBuildExactItemMap with blockCells 1); the kernel counts the planes of its
//                    item's axis again and adds nothing where they differ (PS_FACE_CNT_MISMATCH)
// (The design's record had 26 floats, the three counts packed in bytes: a plane count can pass 255 for a stretched
// void tetrahedron on a fine grid, so each count has a word, and 28 floats are seven 16-byte loads.)
#define PS_FACE_REC_E      0
#define PS_FACE_REC_VG     12
#define PS_FACE_REC_V0     21
#define PS_FACE_REC_RHO    24
#define PS_FACE_REC_COUNT  25
#define PS_FACE_REC_FLOATS 28

// The face kernel's counters (Metal buffer 5: PS_FACE_CNT_WORDS uint, the host zeroes them before each command
// buffer and adds them up after one that succeeded): the shares the non-finite guard (or, under fxWords 2, the share
// cap) dropped, per accumulator -- they join ps_deposit_params.h's FX_DROP_FACE_S / N / P, in that order -- and the
// items that added nothing because the device's plane count differed from the record's, or the library's word size
// from fxWords: a mismatch means the words are not the CPU's, so the host must not keep them
#define PS_FACE_CNT_S        0
#define PS_FACE_CNT_N        1
#define PS_FACE_CNT_P        2
#define PS_FACE_CNT_MISMATCH 3
#define PS_FACE_CNT_WORDS    4

// faceSweepItems' parameters (Metal buffer 6, set per command buffer: itemBase and nItems change per chunk). The grid
// is the one the CPU's face walk hands psDepositFaces: lo = the region's lower corner (boxL[2d]), h = the spacing,
// ih = 1.f / h formed ON THE HOST in float (the kernel forms no reciprocal; psDepositFaces' canonical branch forms
// the same float), nGrid, the pass's sub-grid (subOrigin, subDims: a window's box under --grid-window) and the wrap.
// scale[0..2] = the units 2^K of the face words S, N, P (fx.faceS / faceN / faceP.scale); capBits[0..2] = the share
// caps' float bits under fxWords 2 (RunUnits::capBits of RU_PS_FACE_S / N / P), 0 under fxWords 0; fxWords = the
// library's PS_FX_WORDS (0: two uint per int64 word, 2: four limbs), checked by the kernel; nRec = the records of the
// items' map; itemBase = the chunk's first item, nItems = the chunk's items (thread k runs item itemBase + k).
struct PSFaceParams
{
    float    lo[3];
    float    h[3];
    float    ih[3];
    int32_t  nGrid[3];
    int32_t  subOrigin[3];
    int32_t  subDims[3];
    int32_t  periodic;     // 0 / 1
    float    scale[3];
    uint32_t capBits[3];
    uint32_t fxWords;
    uint32_t nRec;
    uint32_t itemBase;
    uint32_t nItems;
};
static_assert(sizeof(PSFaceParams) == 29 * 4, "PSFaceParams: 29 four-byte fields, no padding (host == kernel)");

// The device probe of the face library's arithmetic (metal/ps_face.metal faceProbe, the kernel's design K11): per
// case PS_FACE_PROBE_IN floats in (a, b, c, d) and PS_FACE_PROBE_OUT uint out (float bits; ps_face.metal lists them)
#define PS_FACE_PROBE_IN  4
#define PS_FACE_PROBE_OUT 14

#endif
