/* Metal backend of the PS-DTFE GPU deposit (implements gpu_host.h). Owns the metal-cpp
   implementation symbols for the whole binary; compiled only when METAL=1 (which defines
   PS_GPU for the callers). The kernel source is embedded at build time (o_ps/ps_deposit_msl.h,
   generated from metal/ps_deposit.metal) and runtime-compiled once per word size the process
   uses (pipes(), behind a "#define PS_FX_WORDS 0|2" line); the pipeline states are cached. The GPU face kernel of
   DTFE_GPU_FACES=1 (o_ps/ps_face_msl.h, from metal/ps_face.metal) is a library of its own, compiled once in the safe
   math mode by the first face pass (facePipes()). */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <objc/runtime.h>   // the face library's options: whether this macOS has the math-mode selectors

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <tuple>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>

#include "gpu_host.h"
#include "ps_deposit_params.h"
#include "ps_mass_audit.h"     // DTFE_MASS_AUDIT=1: the fixed-point grid's exact total
#include "../fixed_units.h"    // FxWords + fxFoldLimbs: the run-wide 128-bit words into int64 words and a carry
#include "ps_deposit_msl.h"   // generated: static const char PS_DEPOSIT_MSL[]
#include "ps_face_msl.h"      // generated: static const char PS_FACE_MSL[] (DTFE_GPU_FACES=1)
#include "../thread_count.h"   // CorePool: the core of a thread that waits for the GPU is lent to the pool
#include "../run_mode.h"       // runModeState(): the run mode's pace base (gpu_metal_retry.h) and exact-item cap
#include <atomic>
extern std::atomic<int> psGpuRetries;   // ps_interpolation.cc (run_progress.h)
extern std::atomic<int> psGpuRetriesByCause[4];   // ... by cause: innocent, timeout, oom, other
#include "../message.h"       // ANSI colour helpers only (retry warning below)

namespace {

using DepositParams = PSDepositParams;   // shared host struct (ps_deposit_params.h)

// The pipelines of one word size. The kernel source takes the size as a preprocessor constant (PS_FX_WORDS in
// metal/ps_deposit.metal), so a library carries only its own add path: one runtime compile per size a process uses
// (a run has one size; PSGpuPlan::words).
struct Pipes
{
    MTL::ComputePipelineState* pso  = nullptr;   // depositFields: one thread per tet
    MTL::ComputePipelineState* psoItems[3] = { nullptr, nullptr, nullptr };   // depositExactItems: one per (tet, cell
                                                                     // block), a pipeline per pass (its kItemsPhase),
                                                                     // and [2] both passes of --ps-linear-deposit
    std::string                err;
    std::string                errLinear;   // why psoItems[2] is null: its compile failed (the others stand)
    bool                       ready = false;
};

// The face kernel's pipeline (DTFE_GPU_FACES=1: metal/ps_face.metal faceSweepItems), the 64-bit words only
struct FacePipes
{
    MTL::ComputePipelineState* pso = nullptr;
    std::string                err;
    bool                       ready = false;
};

struct Ctx
{
    MTL::Device*               dev  = nullptr;
    MTL::CommandQueue*         q    = nullptr;
    Pipes                      pipes[2];       // [0] the 64-bit words (PSDepositParams.fxWords 0), [1] the wide (2)
    std::once_flag             pipesOnce[2];   // pipes()
    FacePipes                  facePipes;      // facePipes()
    std::once_flag             faceOnce;
    std::string                err;
    bool                       ready = false;
};

// serializes the dispatches (one queue): a psGpuDepositFields call holds it from its uploads to the end of its passes
std::mutex& ctxMutex() { static std::mutex m; return m; }

// The device and its queue: ctx() runs this once.
void ctxInit(Ctx& c)
{
    c.dev = MTL::CreateSystemDefaultDevice();
    if (!c.dev) { c.err = "no Metal device"; return; }
    c.q = c.dev->newCommandQueue();
    if (!c.q) { c.err = "no Metal command queue"; return; }
    c.ready = true;
}

// The pipelines of one word size (the runtime compile of the kernel source behind its "#define PS_FX_WORDS" line,
// with the compiler's default options for either size): pipes() runs this once per size.
void pipesInit(Ctx& c, uint32_t const words, Pipes& p)
{
    // (the compiler's messages are autoreleased: copied into p.err / p.errLinear before the pool goes)
    struct Pool
    {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        ~Pool() { pool->release(); }
    } const pool;
    auto const t0 = std::chrono::steady_clock::now();
    std::string const src = std::string("#define PS_FX_WORDS ") + (words == 2u ? "2" : "0") + "\n" + PS_DEPOSIT_MSL;
    NS::Error* nsErr = nullptr;
    MTL::Library* lib = c.dev->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding),
        (MTL::CompileOptions*)nullptr, &nsErr);
    if (!lib)
    {
        p.err = std::string("kernel compile failed: ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return;
    }
    for (auto [name, dst, phase] : { std::tuple<const char*, MTL::ComputePipelineState**, int>{"depositFields", &p.pso, -1},
                                     std::tuple<const char*, MTL::ComputePipelineState**, int>{"depositExactItems", &p.psoItems[0], 0},
                                     std::tuple<const char*, MTL::ComputePipelineState**, int>{"depositExactItems", &p.psoItems[1], 1},
                                     std::tuple<const char*, MTL::ComputePipelineState**, int>{"depositExactItems", &p.psoItems[2], 2} })
    {
        MTL::Function* fn = nullptr;
        if (phase < 0) fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
        else
        {
            MTL::FunctionConstantValues* fcv = MTL::FunctionConstantValues::alloc()->init();
            uint32_t const ph = uint32_t(phase);
            fcv->setConstantValue(&ph, MTL::DataTypeUInt, NS::UInteger(0));   // kItemsPhase
            fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding), fcv, &nsErr);
            fcv->release();
        }
        // Pipeline 2 serves only --ps-linear-deposit: when it fails it stays null and the device still serves every
        // other run (runItems sends a linear run's partitions to the CPU with this reason).
        if (!fn)
        {
            // phase 2 is a specialization of a kernel that phases 0 and 1 found: the compiler's reason, not 'not found'
            std::string const why = phase == 2
                ? std::string("kernel '") + name + "' (kItemsPhase 2) did not compile: "
                      + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown")
                : std::string("kernel '") + name + "' not found";
            if (phase == 2) { p.errLinear = why; continue; }
            p.err = why;
            return;
        }
        *dst = c.dev->newComputePipelineState(fn, &nsErr);
        if (!*dst)
        {
            std::string const why = std::string("pipeline state failed: ")
                                  + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
            if (phase == 2) { p.errLinear = why; continue; }
            p.err = why;
            return;
        }
    }
    p.ready = true;
    // DTFE_GPU_TIMING=1: what the compile cost and how the device groups each pipeline's threads
    if (Env::flag({"DTFE_GPU_TIMING"}))
    {
        fprintf(stderr, "PS-DTFE Metal: pipelines for word size %u compiled in %.3f s\n", unsigned(words),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        MTL::ComputePipelineState* const all[4] = { p.pso, p.psoItems[0], p.psoItems[1], p.psoItems[2] };
        char const* const names[4] = { "depositFields", "exact items pass 0", "exact items pass 1",
                                       "exact items linear" };
        for (int i = 0; i < 4; ++i)
            if (all[i])
                fprintf(stderr, "PS-DTFE Metal: pipeline %-18s  %4lu threads per group at most, %3lu wide\n",
                        names[i], static_cast<unsigned long>(all[i]->maxTotalThreadsPerThreadgroup()),
                        static_cast<unsigned long>(all[i]->threadExecutionWidth()));
    }
}

// The process's device context, built by the first caller (std::call_once: the partitions prepare their deposits
// before they take ctxMutex, so several may ask at once; the others wait for the first to finish)
Ctx& ctx()
{
    static Ctx c;
    static std::once_flag once;
    std::call_once(once, ctxInit, std::ref(c));
    return c;
}

// The pipelines of the given word size, compiled by the first caller that asks for that size (as ctx())
Pipes const& pipes(Ctx& c, uint32_t const words)
{
    int const i = words == 2u ? 1 : 0;
    std::call_once(c.pipesOnce[i], pipesInit, std::ref(c), words, std::ref(c.pipes[i]));
    return c.pipes[i];
}

// The face kernel's pipeline (DTFE_GPU_FACES=1): the embedded source (metal/ps_face.metal with fixed_words.metal.inc,
// ps_kernel_params.h, ps_common.metal.inc and src/CGAL_triangulation/face_sweep_f32.h in front) behind
// "#define PS_FX_WORDS 0" -- the 64-bit words, the only ones a face pass takes -- compiled in Metal's SAFE math mode:
// face_sweep_f32.h's float shares are the CPU's bit for bit only when every operation is a correctly rounded IEEE
// single operation and none is reassociated or contracted. The deposit's library keeps the compiler's default fast
// math (pipesInit). fastMathEnabled = false is that mode on every macOS; macOS 15 also names the mode and the
// precise library functions, selectors an older runtime does not have, so each is asked for first. (Which of the
// operations the device rounds as the CPU does is what the kernel's probe, faceProbe, checks: the harness's.)
// facePipes() runs this once.
void facePipesInit(Ctx& c, FacePipes& p)
{
    struct Pool
    {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        ~Pool() { pool->release(); }
    } const pool;
    auto const t0 = std::chrono::steady_clock::now();
    std::string const src = std::string("#define PS_FX_WORDS 0\n") + PS_FACE_MSL;
    MTL::CompileOptions* opt = MTL::CompileOptions::alloc()->init();
    if (!opt) { p.err = "the face kernel's compile options could not be made"; return; }
    opt->setFastMathEnabled(false);
    Class const optClass = object_getClass(reinterpret_cast<id>(opt));
    if (class_respondsToSelector(optClass, sel_registerName("setMathMode:")))
        opt->setMathMode(MTL::MathModeSafe);
    if (class_respondsToSelector(optClass, sel_registerName("setMathFloatingPointFunctions:")))
        opt->setMathFloatingPointFunctions(MTL::MathFloatingPointFunctionsPrecise);
    NS::Error* nsErr = nullptr;
    MTL::Library* lib = c.dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), opt, &nsErr);
    opt->release();
    if (!lib)
    {
        p.err = std::string("face kernel compile failed: ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return;
    }
    MTL::Function* fn = lib->newFunction(NS::String::string("faceSweepItems", NS::UTF8StringEncoding));
    if (!fn)
    {
        p.err = "kernel 'faceSweepItems' not found";
        lib->release();
        return;
    }
    p.pso = c.dev->newComputePipelineState(fn, &nsErr);
    fn->release();
    lib->release();
    if (!p.pso)
    {
        p.err = std::string("face pipeline state failed: ")
              + (nsErr ? nsErr->localizedDescription()->utf8String() : "unknown");
        return;
    }
    p.ready = true;
    if (Env::flag({"DTFE_GPU_TIMING"}))
        fprintf(stderr, "PS-DTFE Metal: face pipeline compiled in %.3f s; faceSweepItems  %4lu threads per group at "
                        "most, %3lu wide\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                static_cast<unsigned long>(p.pso->maxTotalThreadsPerThreadgroup()),
                static_cast<unsigned long>(p.pso->threadExecutionWidth()));
}

// The face kernel's pipeline, compiled by the first face pass that asks (as pipes())
FacePipes const& facePipes(Ctx& c)
{
    std::call_once(c.faceOnce, facePipesInit, std::ref(c), std::ref(c.facePipes));
    return c.facePipes;
}

} // namespace


std::string gpuBackendName() { return "Metal"; }

std::string gpuDeviceName()
{
    Ctx& c = ctx();   // (thread-safe: std::call_once)
    return c.dev ? std::string(c.dev->name()->utf8String()) : std::string();
}


// ---- chunk-level checkpoints and the cause-aware retry policy ----
// Every pass keeps a snapshot of the buffers its kernel writes (the accumulator grids, the exact items' sumW,
// the per-tet masses the kernel negates when it defers a tet), taken at the pass's start and every
// DTFE_GPU_CHECKPOINT_CHUNKS chunks; a failed buffer (which may have deposited part or all of its atomics)
// restores the last snapshot and continues from its chunk, instead of redoing the WHOLE partition deposit (up
// to ~2000 s per TNG50 partition). The policy (GpuFault, the pace, the fault hook, GpuSnapshot) and its pass
// loop, gpuRunPass, live in gpu_metal_retry.h, shared with the standard-DTFE host (dtfe_metal_host.cc);
// psGpuDepositFields' runChunked below runs each pass through it.
namespace {

#include "gpu_metal_retry.h"

// the run's retry counters (ps_interpolation.cc's), by gpuRetryCause's index
void countGpuRetry(GpuFault f, long code, std::string const& desc)
{
    int const k = gpuRetryCause(f, code, desc);
    ++psGpuRetries;
    ++psGpuRetriesByCause[k];
}

} // namespace


namespace {

// The arguments of psHostUploadBuffers: psGpuDepositFields' locals (the record arrays it consumes, by reference).
struct PSHostUploadArgs
{
    Ctx&                 c;
    std::vector<float>  &verts, &vels, &masses, &dens, &scal;
    DepositParams const& P;
    PSGpuPlan const&     pl;
    size_t               nCell;
};

// A call's buffers as the upload leaves them (null: allocation failed), and which optional per-tet arrays are real.
struct PSHostBuffers
{
    MTL::Buffer *bV, *bU, *bM, *bD, *bSc, *bP, *bMass, *bMassFx, *bMom, *bM2, *bGrad, *bStr, *bMomW, *bCaust, *bSV,
                *bDV, *bDW, *bScal, *bSG, *bBad;
    bool haveVels, haveDens, haveScal;
};

// Uploads the per-tet records and the parameter block, freeing each host array at once, and allocates the zeroed grids.
template <class ZeroBuf>
PSHostBuffers psHostUploadBuffers(PSHostUploadArgs const& u, ZeroBuf& zeroBuf)
{
    Ctx& c = u.c;
    std::vector<float> &verts = u.verts, &vels = u.vels, &masses = u.masses, &dens = u.dens, &scal = u.scal;
    DepositParams const& P = u.P;
    PSGpuPlan const& pl = u.pl;
    size_t const nCell = u.nCell;
    MTL::Buffer* bV = c.dev->newBuffer(verts.data(),  verts.size()  * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(verts);
    // vels is empty for density-only runs (kernel never reads it then): bind a dummy. Whether it
    // is real matters at bind time: the per-chunk offset below must NOT be applied to the 4-byte
    // dummy, or every chunk after the first binds past its end -- undefined behaviour that the
    // kernel gets away with only because it never dereferences the pointer in that case.
    bool const haveVels = not vels.empty();
    MTL::Buffer* bU = vels.empty() ? zeroBuf(4)
                    : c.dev->newBuffer(vels.data(), vels.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(vels);
    MTL::Buffer* bM = c.dev->newBuffer(masses.data(), masses.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(masses);
    // per-tet vertex densities (--ps-linear-deposit); dummy when the uniform deposit runs
    // 'dens' carries either the per-tet vertex densities (fLinear) or, when fCaustic runs without
    // fLinear, the per-tet caustic mask in slot 0. Either way the buffer is real and its per-chunk
    // offset must be applied -- keyed on the vector being non-empty, not on fLinear.
    bool const haveDens = not dens.empty();
    MTL::Buffer* bD = (haveDens)
                    ? c.dev->newBuffer(dens.data(), dens.size() * sizeof(float), MTL::ResourceStorageModeShared)
                    : zeroBuf(4);
    std::vector<float>().swap(dens);
    // the per-tet vertex scalars (fScal / fSGrad); dummy otherwise, so its per-chunk offset is keyed
    // on the vector being real, like 'dens'
    bool const haveScal = not scal.empty();
    MTL::Buffer* bSc = haveScal
                     ? c.dev->newBuffer(scal.data(), scal.size() * sizeof(float), MTL::ResourceStorageModeShared)
                     : zeroBuf(4);
    std::vector<float>().swap(scal);
    MTL::Buffer* bP = c.dev->newBuffer(&P, sizeof(P), MTL::ResourceStorageModeShared);
    MTL::Buffer* bMass = zeroBuf(pl.massBytes);
    MTL::Buffer* bMassFx = zeroBuf(pl.massFxBytes);
    MTL::Buffer* bMom  = zeroBuf(pl.momBytes);
    MTL::Buffer* bM2   = zeroBuf(pl.m2Bytes);
    MTL::Buffer* bGrad = zeroBuf(pl.gradBytes);
    MTL::Buffer* bStr  = zeroBuf(nCell * 4);
    MTL::Buffer* bMomW  = zeroBuf(pl.momwBytes);
    MTL::Buffer* bCaust = zeroBuf(pl.caustBytes);
    MTL::Buffer* bSV    = zeroBuf(pl.svBytes);
    MTL::Buffer* bDV    = zeroBuf(pl.dvBytes);
    MTL::Buffer* bDW    = zeroBuf(pl.dwBytes);
    MTL::Buffer* bScal  = zeroBuf(pl.scalBytes);
    MTL::Buffer* bSG    = zeroBuf(pl.sgBytes);
    // the non-finite guard's counters of depositFields (ps_deposit.metal FieldGrids.bad), one word per slot
    MTL::Buffer* bBad   = zeroBuf(FX_DROP_SLOTS * 4);
    return PSHostBuffers{ bV, bU, bM, bD, bSc, bP, bMass, bMassFx, bMom, bM2, bGrad, bStr, bMomW, bCaust, bSV, bDV,
                          bDW, bScal, bSG, bBad, haveVels, haveDens, haveScal };
}

// The arguments of psHostExactItemMap: psGpuDepositFields' locals (by reference those it sets).
struct PSHostItemsArgs
{
    Ctx&                      c;
    std::vector<float> const& verts;
    DepositParams const&      P;
    const size_t*             nGrid;
    size_t                    nTetTotal;
    bool                      periodic, windowMode, timing;
    bool&                     useItems;
    uint32_t&                 blockCells;
    MTL::Buffer*&             bItems;
    size_t&                   nItems;
    PSGpuPhaseTimes&          ph;
};

// Before the hold: bounds the tets' windows from the host records and fills the exact items' map
// (psBuildExactItemMap) into its GPU buffer; the items are off if the buffer fails.
void psHostExactItemMap(PSHostItemsArgs const& x)
{
    if (!x.useItems) return;
    auto tPh = std::chrono::steady_clock::now();
    std::vector<uint64_t> bound;
    double const total = psExactItemBounds(x.verts.data(), x.nTetTotal, x.P, x.nGrid, x.periodic, x.windowMode, bound,
                                           x.blockCells, x.nItems);
    if (x.timing)
        fprintf(stderr, "PS-DTFE Metal: exact items: %zu tets, %zu items of %u cells, window bound %.3g cells\n",
                x.nTetTotal, x.nItems, x.blockCells, total);
    x.ph.bounds += PSGpuPhaseTimes::lap(tPh);
    x.bItems = psExactItemMapFits(x.nTetTotal, x.nItems)
             ? x.c.dev->newBuffer(psExactItemMapWords(x.nTetTotal, x.nItems) * sizeof(uint32_t),
                                  MTL::ResourceStorageModeShared)
             : nullptr;
    if (!x.bItems) x.useItems = false;   // too big (GPU-visible memory, 32-bit item numbers): the one-thread-per-tet kernel still works
    else psBuildExactItemMap(bound, x.blockCells, x.nItems, static_cast<uint32_t*>(x.bItems->contents()));
    x.ph.fill += PSGpuPhaseTimes::lap(tPh);
}

// The arguments of psHostCopyBack: psGpuDepositFields' locals (by reference the call's outputs).
struct PSHostCopyArgs
{
    PSHostBuffers const&   B;
    PSGpuPlan const&       pl;
    size_t                 nCell, nTetTotal;
    bool                   fCaustic;
    PSGpuGrids&            out;
    std::vector<uint32_t>& deferred;
};

// Lists the deferred tets (their negated masses) and copies the kept grids into 'out'.
void psHostCopyBack(PSHostCopyArgs const& k)
{
    PSHostBuffers const& B = k.B;
    MTL::Buffer *bM = B.bM, *bMass = B.bMass, *bMassFx = B.bMassFx, *bMom = B.bMom, *bM2 = B.bM2, *bGrad = B.bGrad,
                *bStr = B.bStr, *bMomW = B.bMomW, *bCaust = B.bCaust, *bSV = B.bSV, *bDV = B.bDV, *bDW = B.bDW,
                *bScal = B.bScal, *bSG = B.bSG;
    PSGpuPlan const& pl = k.pl;
    size_t const nCell = k.nCell, nTetTotal = k.nTetTotal;
    bool const fCaustic = k.fCaustic;
    PSGpuGrids& out = k.out;
    std::vector<uint32_t>& deferred = k.deferred;
    // tets the kernel could not classify exactly (a sample within float rounding of a face):
    // it negated their masses and deposited nothing; the caller deposits them on the CPU
    bool const massAudit = PSMassAudit::on();
    {
        float const* mb = static_cast<float const*>(bM->contents());
        double deferredMass = 0.;
        for (size_t t = 0; t < nTetTotal; ++t)
            if (mb[t] < 0.f) { deferred.push_back(uint32_t(t)); deferredMass -= double(mb[t]); }
        if (massAudit) out.auditDeferredMass = deferredMass;
    }

    if (pl.massFixed && pl.words == 2u)
    {
        // the run-wide 128-bit words: each cell's four limbs folded into its int64 word and the exact carry
        // (fixed_units.h fxFoldLimbs), handed over unconverted, as the 64-bit words below
        FxWords f;
        fxFoldLimbs(static_cast<uint32_t const*>(bMassFx->contents()), nCell, f);
        out.massFx.swap(f.w);
        out.massFxCarry.swap(f.carry);
        if (massAudit)
        {   // the exact integer total in 128 bits (in the run's unit a pass's mass can lie beyond 2^63 units)
            __int128 q = 0;
            for (int64_t const w : out.massFx) q += __int128(w);
            for (auto const& c : out.massFxCarry) q += __int128(c.second) * (__int128(1) << 64);
            out.auditFixedTotal = double(q) / double(pl.massScale);
        }
    }
    else if (pl.massFixed)
    {
        // the words themselves, not their float decode: adding the deferred tets' shares into decoded floats
        // would round away every share below half an ulp of a busy cell, always downward (1.6e-5 of the box
        // mass at TNG50-3-Dark z=0.5). Low word first: on this little-endian host each pair is the int64
        // (ps_deposit_params.h).
        out.massFx.resize(nCell);
        std::memcpy(out.massFx.data(), bMassFx->contents(), nCell * 8);
        if (massAudit)
        {   // the exact integer total (the partition's mass is below 2^62 units: no overflow)
            int64_t q = 0;
            for (size_t i = 0; i < nCell; ++i) q += out.massFx[i];
            out.auditFixedTotal = psMassFixedMass(q, pl.massScale);
        }
    }
    else
        std::memcpy(out.mass.data(), bMass->contents(), nCell * 4);
    // each moment grid: its words (fixed point; each low/high pair IS the int64 on this little-endian
    // host, as for the mass; the run-wide 128-bit words folded with their carry, as the mass) or its floats
    MTL::Buffer* const fxBuf[PS_FX_N] = { bMom, bM2, bGrad, bMomW, bSV, bDV, nullptr, bScal, bSG };
    std::vector<float>* const flt[PS_FX_N] = { &out.mom, &out.m2, &out.grad, &out.momw, &out.streamvol, &out.dispvel,
                                               nullptr, &out.scal, &out.sgrad };
    for (int g = 0; g < PS_FX_N; ++g)
    {
        if (!pl.sel[g] || !fxBuf[g]) continue;
        size_t const n = nCell * PS_FX_COMPS[g];
        if (pl.fxOn[g] && pl.words == 2u)
        {
            FxWords f;
            fxFoldLimbs(static_cast<uint32_t const*>(fxBuf[g]->contents()), n, f);
            out.fx[g].swap(f.w);
            out.fxCarry[g].swap(f.carry);
        }
        else if (pl.fxOn[g]) { out.fx[g].resize(n); std::memcpy(out.fx[g].data(), fxBuf[g]->contents(), n * 8); }
        else std::memcpy(flt[g]->data(), fxBuf[g]->contents(), n * 4);
    }
    std::memcpy(out.streams.data(), bStr->contents(),  nCell * 4);
    if (fCaustic) std::memcpy(out.caustic.data(), bCaust->contents(), nCell * 4);
    if (pl.fDispOwn && !pl.dispwIsMass) std::memcpy(out.dispw.data(), bDW->contents(), nCell * 4);
}

// The core of a thread that sleeps until the GPU is done with its work (the queue, then each buffer's
// waitUntilCompleted), lent to the core pool for that time (thread_count.h: one helper more may be claimed); see
// psGpuDepositFields. A claim on the pool never changes a result.
struct CoreLend
{
    CoreLend()  { CorePool::extra().fetch_sub( 1, std::memory_order_relaxed ); }
    ~CoreLend() { CorePool::extra().fetch_add( 1, std::memory_order_relaxed ); }
    CoreLend(CoreLend const &) = delete;
    CoreLend &operator=(CoreLend const &) = delete;
};

// ---- the GPU face kernel's pass (DTFE_GPU_FACES=1; gpu_host.h PSGpuFaceJob) ----

// Before the hold: the face pipeline (compiled by the first job: facePipes()) and the job's items' map in its GPU
// buffer -- psBuildExactItemMap with blockCells 1, a record's items being the sum of its three plane counts
// (ps_kernel_params.h PS_FACE_REC_COUNT), so an item is (record, its number among the record's items), which the
// kernel turns into (axis, plane). False, with job.err set and no buffer held, when the job cannot run; true with
// bMap and fp null when no record has an item (the pass then has nothing to sweep).
bool psHostFacePrepare(Ctx& c, PSGpuFaceJob& job, FacePipes const*& fp, MTL::Buffer*& bMap)
{
    auto tPh = std::chrono::steady_clock::now();
    fp = nullptr;
    bMap = nullptr;
    job.ok = false;
    job.mismatches = 0;
    for (unsigned long long& d : job.drops) d = 0;
    job.err.clear();
    if (job.P.fxWords != 0u)
    {
        job.err = "the face kernel takes the 64-bit words only";
        return false;
    }
    if (job.recs.size() % size_t(PS_FACE_REC_FLOATS) != 0)
    {
        job.err = "face record size mismatch (PS_FACE_REC_FLOATS floats per record)";
        return false;
    }
    size_t const nRec = job.recs.size() / size_t(PS_FACE_REC_FLOATS);
    std::vector<uint64_t> items(nRec);
    size_t nItems = 0;
    for (size_t r = 0; r < nRec; ++r)
    {
        uint32_t cnt[3];
        std::memcpy(cnt, &job.recs[r * size_t(PS_FACE_REC_FLOATS) + size_t(PS_FACE_REC_COUNT)], sizeof cnt);
        items[r] = uint64_t(cnt[0]) + uint64_t(cnt[1]) + uint64_t(cnt[2]);
        nItems += size_t(items[r]);
    }
    job.nRec = nRec;
    job.nItems = nItems;
    job.mapSec += PSGpuPhaseTimes::lap(tPh);
    if (nItems == 0) return true;
    FacePipes const& p = facePipes(c);   // (its compile is timed on its own: facePipesInit)
    tPh = std::chrono::steady_clock::now();
    if (!p.ready)
    {
        job.err = p.err;
        return false;
    }
    if (!psExactItemMapFits(nRec, nItems))
    {
        job.err = "too many face items for 32-bit item numbers";
        return false;
    }
    bMap = c.dev->newBuffer(psExactItemMapWords(nRec, nItems) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!bMap)
    {
        job.err = "Metal buffer allocation failed (the face items' map)";
        return false;
    }
    psBuildExactItemMap(items, 1u, nItems, static_cast<uint32_t*>(bMap->contents()));
    job.P.nRec = uint32_t(nRec);
    fp = &p;
    job.mapSec += PSGpuPhaseTimes::lap(tPh);
    return true;
}

// In a hold of the GPU (the caller holds ctxMutex): a prepared job's face pass. The records go up (the host's copy
// freed at once), the three word buffers start zeroed (shared storage: on unified memory the copy back is a memcpy;
// the host's words are std::vectors, not page-aligned, so they cannot be wrapped without a copy), faceSweepItems runs
// over the items through gpuRunPass as the exact deposit's items do (the run mode's pace and item cap, the first
// chunk 8192 items, at least 1024; every buffer counted by DTFE_GPU_FAULT_AT's hook), and the words come back into
// the job one accumulator at a time, each device buffer released after its copy. A failed buffer restores the words
// to zero and restarts at item 0 (no periodic checkpoint: the words are known zero at the start, so the snapshot
// holds nothing, and the pass is short); the retry policy's limits count the failed buffers of the deposit in the
// same hold too (nFailed). False -- job.ok false and no word kept -- when the pass gave up, a buffer could not be
// had, or the device counted an item whose plane count differed from its record's (job.mismatches: its words are not
// the CPU's). Takes bMap (released here).
bool psHostFacePass(Ctx& c, PSGpuFaceJob& job, FacePipes const* fp, MTL::Buffer*& bMap, int& nFailed,
                    int& nResumedCk, bool const timing)
{
    auto const tPass = std::chrono::steady_clock::now();
    auto tPh = tPass;
    size_t const nS = job.nCell * 18, nN = job.nCell * 6;
    job.ok = false;
    if (job.nItems == 0)
    {   // no record owns a section: every face word of the pass is zero
        std::vector<float>().swap(job.recs);
        job.wS.assign(nS, 0);
        job.wN.assign(nN, 0);
        job.wP.assign(nN, 0);
        if (bMap) { bMap->release(); bMap = nullptr; }
        job.ok = true;
        job.heldSec += PSGpuPhaseTimes::lap(tPh);
        return true;
    }
    if (!fp || !fp->pso || !bMap)
    {   // (psHostFacePrepare did not return true for this job)
        job.err = "the face pass was not prepared";
        if (bMap) { bMap->release(); bMap = nullptr; }
        return false;
    }
    // (the command buffers are autoreleased and retain the buffers they ran on: drained before the hold ends)
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    auto zeroBuf = [&](size_t bytes) {
        MTL::Buffer* b = c.dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (b) std::memset(b->contents(), 0, bytes);
        return b;
    };
    MTL::Buffer* bR = c.dev->newBuffer(job.recs.data(), job.recs.size() * sizeof(float),
                                       MTL::ResourceStorageModeShared);
    std::vector<float>().swap(job.recs);
    MTL::Buffer* bS = zeroBuf(nS * 8);
    MTL::Buffer* bN = zeroBuf(nN * 8);
    MTL::Buffer* bP = zeroBuf(nN * 8);
    // the guard's and the plane counts' counters (PS_FACE_CNT_*): gpuRunPass zeroes and reads FX_DROP_SLOTS words
    static_assert(PS_FACE_CNT_WORDS <= int(FX_DROP_SLOTS), "the face counters fit gpuRunPass' counter words");
    MTL::Buffer* bC = zeroBuf(size_t(FX_DROP_SLOTS) * 4);
    job.uploadSec += PSGpuPhaseTimes::lap(tPh);
    auto finish = [&](bool const ok) -> bool
    {
        auto tRel = std::chrono::steady_clock::now();
        for (MTL::Buffer* b : { bR, bS, bN, bP, bC, bMap }) if (b) b->release();
        bR = bS = bN = bP = bC = nullptr;
        bMap = nullptr;
        pool->release();
        job.releaseSec += PSGpuPhaseTimes::lap(tRel);
        job.heldSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - tPass).count();
        job.ok = ok;
        return ok;
    };
    if (!bR || !bS || !bN || !bP || !bC)
    {
        job.err = "Metal buffer allocation failed (the face pass: out of GPU-visible memory?)";
        return finish(false);
    }
    GpuSnapshot snap;
    std::string lastErr;
    unsigned long long bad[FX_DROP_SLOTS] = {};
    GpuRetryRun run{ snap, nFailed, nResumedCk, lastErr, bad };
    std::vector<MTL::Buffer*> ckBufs = { bS, bN, bP };
    std::vector<bool> const zeroMask(ckBufs.size(), true);
    GpuPassSpec pass;
    pass.tag = "PS-DTFE Metal";
    pass.scope = "hold";   // (GPU_MAX_FAILS_TOTAL: the deposit's failed buffers in the same hold count too)
    pass.q = c.q;
    pass.pso = fp->pso;
    pass.nThreads = job.nItems;
    pass.minChunk = 1024;
    pass.maxChunk = std::max<size_t>(runModeState().psItemCap, pass.minChunk);
    pass.ckOn = snap.take(ckBufs, zeroMask, 0);   // (all zero: nothing copied)
    pass.periodic = false;
    pass.ckBufs = &ckBufs;
    pass.ckMask = zeroMask;
    pass.ckChunks = 0;
    pass.paceRecoverSec = psHostCheckpointEnv().paceRecoverSec;
    pass.badW = static_cast<uint32_t*>(bC->contents());
    PSFaceParams FP = job.P;
    FP.fxWords = 0u;
    for (uint32_t& cb : FP.capBits) cb = 0u;
    size_t chunk = 8192;
    auto const tRun = std::chrono::steady_clock::now();
    bool const ran = gpuRunPass(pass, run, chunk,
                                [&](MTL::ComputeCommandEncoder* enc, size_t start, uint32_t n)
    {
        FP.itemBase = uint32_t(start);
        FP.nItems = n;
        enc->setBuffer(bR,   0, 0);
        enc->setBuffer(bMap, 0, 1);
        enc->setBuffer(bS,   0, 2);
        enc->setBuffer(bN,   0, 3);
        enc->setBuffer(bP,   0, 4);
        enc->setBuffer(bC,   0, 5);
        enc->setBytes(&FP, sizeof FP, 6);
    },
                                [&] { snap.restore(); return snap.start; },
                                [] {});
    job.gapSec += pass.gaps;
    tPh = std::chrono::steady_clock::now();
    if (!ran)
    {
        job.err = lastErr;
        fprintf(stderr, "%sPS-DTFE Metal: the face pass gave up (%s); the CPU face walk makes this pass's face sums "
                        "instead (the same words)%s\n", MESSAGE::cYellow(), lastErr.c_str(), MESSAGE::cReset());
        return finish(false);
    }
    job.busySec += pass.busy;
    job.buffers += pass.buffers;
    if (timing)
        fprintf(stderr, "PS-DTFE Metal: %-17s %10zu threads  %6zu buffers  %8.3f s busy  %8.3f s wall\n",
                "faceSweepItems", job.nItems, pass.buffers, pass.busy,
                std::chrono::duration<double>(tPh - tRun).count());
    job.mismatches = bad[PS_FACE_CNT_MISMATCH];
    if (job.mismatches > 0)
    {
        job.err = "the device's plane count differed from the records' for "
                + std::to_string(job.mismatches) + " face item(s)";
        fprintf(stderr, "%sPS-DTFE Metal: %llu face item(s) whose plane count the device found other than the host's "
                        "(face_sweep_f32.h fsPlanes: a device rounding the canonical arithmetic does not allow for; "
                        "the kernel's probe faceProbe shows which); the CPU face walk makes this pass's face sums "
                        "instead%s\n", MESSAGE::cYellow(), job.mismatches, MESSAGE::cReset());
        return finish(false);
    }
    for (int k = 0; k < 3; ++k) job.drops[k] = bad[PS_FACE_CNT_S + k];
    // the words back, one accumulator at a time (each pair of uint, low word first, IS the int64 on this
    // little-endian host, as for the deposit's words), each device buffer freed after its copy
    job.wS.resize(nS);
    std::memcpy(job.wS.data(), bS->contents(), nS * 8);
    bS->release();
    bS = nullptr;
    job.wN.resize(nN);
    std::memcpy(job.wN.data(), bN->contents(), nN * 8);
    bN->release();
    bN = nullptr;
    job.wP.resize(nN);
    std::memcpy(job.wP.data(), bP->contents(), nN * 8);
    job.copySec += PSGpuPhaseTimes::lap(tPh);
    return finish(true);
}

// DTFE_GPU_TIMING=1: one line per face pass, in psGpuReportPhases' style
void psHostReportFace(PSGpuFaceJob const& job)
{
    fprintf(stderr, "PS-DTFE Metal: partition %d face pass: %zu records (%.1f MB), %zu items; before the hold: "
                    "items' map %.3f s; in the hold: uploads %.3f s, kernels %.3f s (%zu buffers), pacing gaps "
                    "%.3f s, copy-back %.3f s, releases %.3f s, %.3f s in all\n",
            PSMassAudit::partition, job.nRec, double(job.nRec) * PS_FACE_REC_FLOATS * 4. / 1.e6, job.nItems,
            job.mapSec, job.uploadSec, job.busySec, job.buffers, job.gapSec, job.copySec, job.releaseSec,
            job.heldSec);
}

} // namespace


bool psGpuDepositFields(std::vector<float>& verts,
                          std::vector<float>& vels,
                          std::vector<float>& masses,
                          std::vector<float>& dens,
                          std::vector<float>& scal,
                          const double boxLo[3], const double dx[3],
                          const size_t nGrid[3], const size_t subOrigin[3], const size_t subDims[3],
                          int nSub, bool periodic,
                          bool fVel, bool fDisp, bool fGrad, bool fLinear,
                          bool fVolW, bool fCaustic, bool fExact, bool fScal, bool fSGrad, bool windowMode,
                          double massExtra, PSFixedUnits const& fxUnits,
                          PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err,
                          PSGpuFaceJob* faceJob)
{
    // DTFE_GPU_TIMING=1: one line per kernel run (threads, buffers, GPU-busy and wall seconds),
    // and per partition where its call went, before and in its hold of the GPU (PSGpuPhaseTimes: timed always,
    // printed only then)
    bool const timing = Env::flag({"DTFE_GPU_TIMING"});
    PSGpuPhaseTimes ph;
    auto tPh = std::chrono::steady_clock::now();
    if (faceJob) faceJob->ok = false;   // (DTFE_GPU_FACES=1: set only by a face pass that made the words)
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }
    ph.device += PSGpuPhaseTimes::lap(tPh);

    // Before the hold (ctxMutex below): the plan, the parameter block and the exact items' map, from the host
    // records -- work that needs no queue, done while another partition's passes hold the GPU.
    // The plan (the selected grids, their fixed-point form and device sizes; gpu_host.h).
    // DTFE_GPU_FLOAT_MASS=1 keeps float mass atomics, for comparison.
    bool const floatMass = Env::flag({"DTFE_GPU_FLOAT_MASS"});
    PSGpuPlan const pl = psGpuPlan(subDims, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, fScal, fSGrad, masses,
                                   massExtra, fxUnits, floatMass);
    // The pipelines of the plan's word size: compiled by the first deposit that asks (pipes()), booked as the
    // device's time. The parameter block below carries the same size (P.fxWords = pl.words).
    ph.plan += PSGpuPhaseTimes::lap(tPh);
    Pipes const& pp = pipes(c, pl.words);
    ph.device += PSGpuPhaseTimes::lap(tPh);
    if (!pp.ready) { err = pp.err; return false; }
    size_t const nCell = pl.nCell;
    deferred.clear();
    if (masses.empty())                 // nothing to deposit (empty partition; the caller may still add released tets)
    {
        psGpuInitGrids(out, pl, fxUnits);
        psGpuZeroWords(out, pl);
        if (faceJob) psGpuFaceSums(*faceJob);   // (the face records of massless tetrahedra: their own hold)
        return true;
    }
    if (verts.size() != masses.size() * size_t(PS_TET_STRIDE))
    {
        err = "tet record size mismatch (verts must hold PS_TET_STRIDE floats per tet)";
        return false;
    }

    DepositParams P = psGpuParams(pl, fxUnits, boxLo, dx, nGrid, subOrigin, subDims, nSub, periodic, fLinear,
                                  windowMode, not dens.empty(), not scal.empty(), masses.size());
    size_t const nTetTotal = masses.size();
    ph.plan += PSGpuPhaseTimes::lap(tPh);

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // --exact: split the work below the tet (depositExactItems in ps_deposit.metal), items of
    // blockCells cells of a tet's window (psExactItemBounds in gpu_host.h), as the
    // compact map psBuildExactItemMap. The items follow the caller's cost order, keeping a SIMD group's threads on
    // similar work. DTFE_GPU_ITEMS=0 keeps the one-thread-per-tet kernel (A/B), DTFE_GPU_EXACT_CELLS sets the block.
    bool useItems = fExact;
    useItems = useItems && Env::flag("DTFE_GPU_ITEMS", true);
    MTL::Buffer* bItems = nullptr;
    MTL::Buffer* bSumW  = nullptr;
    size_t nItems = 0;
    uint32_t blockCells = 16;
    if (long const v = Env::integer("DTFE_GPU_EXACT_CELLS", 0); v > 0) blockCells = uint32_t(v);
    psHostExactItemMap(PSHostItemsArgs{ c, verts, P, nGrid, nTetTotal, periodic, windowMode, timing, useItems,
                                        blockCells, bItems, nItems, ph });
    // DTFE_GPU_FACES=1: the face pass's pipeline and items' map, before the hold as well (a job that cannot run
    // leaves faceJob->ok false: the caller's CPU face walk makes the same words)
    FacePipes const* facePipe = nullptr;
    MTL::Buffer* bFaceMap = nullptr;
    bool const faceRun = faceJob && psHostFacePrepare(c, *faceJob, facePipe, bFaceMap);

    auto const tAsk = std::chrono::steady_clock::now();   // DTFE_GPU_TIMING: the wait for the queue, and its hold
    // From here to the return this thread sleeps: first in the queue for the GPU, then in each buffer's
    // waitUntilCompleted (its own work in the hold, the uploads and the copy-back, is a few percent of it). Its core
    // is lent to the core pool for that time (thread_count.h: one helper more may be claimed), so the CPU work that
    // follows the free cores can take it: the sub-grid merge and the normalization, and in a run that deposits a
    // share of its partitions on the CPU ('--gpu-share', DTFE.cpp) their deposit teams. A claim on the pool
    // never changes a result. (CoreLend: above.)
    CoreLend const coreLend;
    std::unique_lock<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue); let go after the releases
    auto const tGot = std::chrono::steady_clock::now();
    tPh = tGot;

    // newBuffer(ptr,...) copies the host array into the MTL buffer; free each host array
    // IMMEDIATELY so at most one array is double-held at a time (instead of all three, a
    // multi-GB difference per partition at production scale). On failure the CPU fallback
    // re-deposits from the triangulation, never from these arrays.
    auto zeroBuf = [&](size_t bytes) {
        MTL::Buffer* b = c.dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (b) std::memset(b->contents(), 0, bytes);
        return b;
    };
    PSHostBuffers const B = psHostUploadBuffers(PSHostUploadArgs{ c, verts, vels, masses, dens, scal, P, pl, nCell }, zeroBuf);
    MTL::Buffer *bV = B.bV, *bU = B.bU, *bM = B.bM, *bD = B.bD, *bSc = B.bSc, *bP = B.bP, *bMass = B.bMass,
                *bMassFx = B.bMassFx, *bMom = B.bMom, *bM2 = B.bM2, *bGrad = B.bGrad, *bStr = B.bStr, *bMomW = B.bMomW,
                *bCaust = B.bCaust, *bSV = B.bSV, *bDV = B.bDV, *bDW = B.bDW, *bScal = B.bScal, *bSG = B.bSG,
                *bBad = B.bBad;
    bool const haveVels = B.haveVels, haveDens = B.haveDens, haveScal = B.haveScal;
    if (useItems)
    {   // the exact items' per-tet weight totals: a float each, or (sumWFixed) two uint words in the tet's own unit
        // (+ the non-finite guard's counter words of the items kernel, at ExactItemParams.badAt), zeroed
        size_t const sumWBytes = nTetTotal * (pl.sumWFixed ? 8 : sizeof(float));
        bSumW = c.dev->newBuffer(std::max<size_t>(sumWBytes, 4) + FX_DROP_SLOTS * 4, MTL::ResourceStorageModeShared);
        if (bSumW) std::memset(bSumW->contents(), 0, sumWBytes);
        else
        {   // too big for GPU-visible memory: the one-thread-per-tet kernel still works
            bItems->release();
            bItems = nullptr;
            useItems = false;
        }
    }
    ph.uploads += PSGpuPhaseTimes::lap(tPh);
    if (!bV || !bU || !bM || !bD || !bSc || !bP || !bMass || !bMassFx || !bMom || !bM2 || !bGrad || !bStr || !bMomW || !bCaust || !bSV || !bDV || !bDW || !bScal || !bSG || !bBad)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        if (pl.words == 2u)
        {   // the run-wide 128-bit words: a grid longer than the device's largest buffer cannot be had at all (the
            // velocity gradient's 144 B per cell of a large sub-grid can reach it); name the first such grid
            size_t const maxLen = size_t(c.dev->maxBufferLength());
            struct { size_t bytes; size_t word; char const* name; } const grids[] = {
                { pl.massFxBytes, 16, "mass" },
                { pl.momBytes,  pl.fxOn[PS_FX_MOM]  ? 16u : 4u, "velocity moment" },
                { pl.m2Bytes,   pl.fxOn[PS_FX_M2]   ? 16u : 4u, "second moment" },
                { pl.gradBytes, pl.fxOn[PS_FX_GRAD] ? 16u : 4u, "velocity gradient" },
                { pl.momwBytes, pl.fxOn[PS_FX_MOMW] ? 16u : 4u, "moment weight" },
                { pl.svBytes,   pl.fxOn[PS_FX_SV]   ? 16u : 4u, "multiplicity" },
                { pl.dvBytes,   pl.fxOn[PS_FX_DV]   ? 16u : 4u, "dispersion mean" },
                { pl.scalBytes, pl.fxOn[PS_FX_SCAL] ? 16u : 4u, "scalar" },
                { pl.sgBytes,   pl.fxOn[PS_FX_SG]   ? 16u : 4u, "scalar gradient" } };
            for (auto const& g : grids)
                if (g.bytes > maxLen)
                {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf), "the %s grid takes %.2f GB in %zu-byte words, over the %.2f GB "
                                  "of the device's largest buffer (maxBufferLength)", g.name, double(g.bytes) / 1e9,
                                  g.word, double(maxLen) / 1e9);
                    err = buf;
                    break;
                }
        }
        for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad,bItems,bSumW}) if (b) b->release();
        if (bFaceMap) bFaceMap->release();
        pool->release();
        return false;
    }

    // Dispatch in CHUNKS, one short command buffer each, fully serialized with a host-side gap between
    // buffers: the macOS GPU watchdog ("Impacting Interactivity") kills a long buffer when the display needs
    // the GPU, and back-to-back buffers too, even ~0.1 s ones -- it reacts to starvation; the gaps let
    // WindowServer in. The chunk size adapts toward pace.targetSec. A killed buffer may have deposited part of
    // its atomics, so a failure resumes from the pass's last snapshot (runChunked below); the caller deposits
    // the partition on the CPU only after a fatal error, 3 failures of one chunk or 16 in the deposit, or
    // when no snapshot can be held. PS_GPU_CHUNK=<n> sets the per-tet kernel's starting
    // chunk size.
    size_t const MIN_CHUNK = fExact ? 50 : 2000;
    size_t const MAX_CHUNK = 500000;
    // The exact items have a cap of their own, the run mode's (run_mode.h psItemCap). Under 500k items a buffer ran
    // 53 ms: the short buffers of gentle and balanced, with which the display never waited longer than a frame.
    // Performance raises it to 4M, which gave byte-identical grids in 1025 s against 1156 s (TNG100-3-Dark z=0 at
    // 512^3) but let the display wait up to 214 ms (at 155 ms buffers) when something else wanted the GPU. The chunk
    // size changes no item's deposit, and the fixed-point sums do not depend on the order.
    size_t const MAX_CHUNK_ITEMS = runModeState().psItemCap;
    GpuPace& pace = gpuPace();        // the run mode's target (0.25 s performance, 0.05 s otherwise), 8 ms gaps, until escalated
    size_t baseChunk = psHostFirstChunk(fExact, pace.targetSec, MIN_CHUNK);
    std::string lastErr;

    PSHostCkEnv const ck = psHostCheckpointEnv();
    long ckChunks = ck.ckChunks;
    double paceRecoverSec = ck.paceRecoverSec;
    std::vector<MTL::Buffer*> const gridBufs = { bMass, bMassFx, bMom, bM2, bGrad, bStr, bMomW, bCaust, bSV, bDV, bDW, bScal, bSG };
    GpuSnapshot snap;
    int nFailed = 0, nResumedCk = 0;   // failed buffers of this deposit, and how many resumed from a periodic checkpoint
    // The shares the non-finite guard dropped in this deposit, by slot (ps_deposit_params.h
    // psFixedShareOk): each pass zeroes its counter words before a command buffer and adds them here after one that
    // succeeded, and a restore rolls these sums back to the held snapshot's -- so a failed buffer (which may have
    // counted part or all of its shares) and the redone ones count once. Handed to the run's counters only when
    // the deposit is kept (a CPU fallback deposits, and counts, the partition itself).
    unsigned long long badRun[FX_DROP_SLOTS] = {};
    GpuRetryRun run{ snap, nFailed, nResumedCk, lastErr, badRun };

    // One paced, chunked run of 'pso' over nThreads threads (gpu_metal_retry.h gpuRunPass); bind(enc, start, n)
    // binds the buffers for the chunk [start, start+n). 'extras' are the non-grid buffers the kernel WRITES (sumW,
    // the per-tet masses it negates); gridsZero: the grids are all zero at the pass's start (snapshotted as such,
    // without a copy); gridsKeepZero: the pass never writes them (the exact items' phase 0), so its periodic
    // checkpoints skip them too. False (lastErr set) when the pass gave up (the caller falls back).
    // badW: this pass's counter words of the non-finite guard (badRun above).
    auto runChunked = [&](MTL::ComputePipelineState* pso, size_t nThreads, size_t firstChunk,
                          size_t minChunk, size_t maxChunk, std::vector<MTL::Buffer*> const& extras, bool gridsZero,
                          bool gridsKeepZero, uint32_t* badW, auto&& bind) -> bool
    {
        auto const tRun = std::chrono::steady_clock::now();
        std::vector<MTL::Buffer*> ckBufs = gridBufs;
        ckBufs.insert(ckBufs.end(), extras.begin(), extras.end());
        std::vector<bool> zeroMask(ckBufs.size(), false);
        if (gridsZero) std::fill(zeroMask.begin(), zeroMask.begin() + gridBufs.size(), true);
        GpuPassSpec pass;
        pass.tag = "PS-DTFE Metal";
        pass.scope = "deposit";
        pass.q = c.q;
        pass.pso = pso;
        pass.nThreads = nThreads;
        pass.minChunk = minChunk;
        pass.maxChunk = maxChunk;
        pass.ckOn = snap.take(ckBufs, zeroMask, 0);
        pass.ckBufs = &ckBufs;
        if (gridsKeepZero) pass.ckMask = zeroMask;
        pass.ckChunks = ckChunks;
        pass.paceRecoverSec = paceRecoverSec;
        pass.badW = badW;
        size_t chunk = std::max(firstChunk, minChunk);
        bool const ok = gpuRunPass(pass, run, chunk, bind,
                                   [&] { snap.restore(); return snap.start; },
                                   [&] { baseChunk = std::max(baseChunk / 2, MIN_CHUNK); });   // the next pass's too
        ph.gaps += pass.gaps;
        if (!ok) return false;
        ph.kernels += pass.busy;
        if (timing)
            fprintf(stderr, "PS-DTFE Metal: %-17s %10zu threads  %6zu buffers  %8.3f s busy  %8.3f s wall\n",
                    pso == pp.pso ? "depositFields" : pso == pp.psoItems[2] ? "exactItems linear" : "depositExactItems",
                    nThreads, pass.buffers, pass.busy,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tRun).count());
        return true;
    };

    // the moment grids' word views: each fixed-point grid's buffer again at 22 + its slot, where the
    // kernel reads it as two uint words per component (a float grid's slot gets the same buffer; unused there)
    auto bindWords = [&](MTL::ComputeCommandEncoder* enc)
    {
        enc->setBuffer(bMom,  0, 22);
        enc->setBuffer(bM2,   0, 23);
        enc->setBuffer(bGrad, 0, 24);
        enc->setBuffer(bMomW, 0, 25);
        enc->setBuffer(bSV,   0, 26);
        enc->setBuffer(bDV,   0, 27);
        enc->setBuffer(bScal, 0, 28);
        enc->setBuffer(bSG,   0, 29);
    };

    // depositFields over the tets [0,nT) of the given records. The grids are bound whole; the
    // per-tet buffers at the chunk's offset (P.nTet = chunk size, rewritten per chunk: safe, the
    // previous chunk completed). The kernel negates the masses of the tets it defers, so mB is
    // part of the snapshot: a redo then runs exactly the same code path.
    auto runPerTet = [&](MTL::Buffer* vB, MTL::Buffer* uB, MTL::Buffer* mB, MTL::Buffer* dB, MTL::Buffer* sB, size_t nT,
                         bool gridsZero) -> bool
    {
        return runChunked(pp.pso, nT, baseChunk, MIN_CHUNK, MAX_CHUNK, { mB }, gridsZero, false, static_cast<uint32_t*>(bBad->contents()),
                          [&](MTL::ComputeCommandEncoder* enc, size_t start, uint32_t n)
        {
            P.nTet = n;
            std::memcpy(bP->contents(), &P, sizeof(P));
            enc->setBuffer(vB,    start * PS_TET_STRIDE * sizeof(float), 0);
            enc->setBuffer(uB,    haveVels ? start * 12 * sizeof(float) : 0, 1);
            enc->setBuffer(mB,    start * sizeof(float),      2);
            enc->setBuffer(bMass, 0, 3);
            enc->setBuffer(bMom,  0, 4);
            enc->setBuffer(bM2,   0, 5);
            enc->setBuffer(bGrad, 0, 6);
            enc->setBuffer(bStr,  0, 7);
            enc->setBuffer(bP,    0, 8);
            enc->setBuffer(dB,    haveDens ? start * 4 * sizeof(float) : 0, 9);
            enc->setBuffer(bMomW, 0, 10);
            enc->setBuffer(bCaust,0, 11);
            enc->setBuffer(bSV,   0, 12);
            enc->setBuffer(bDV,   0, 13);
            enc->setBuffer(bDW,   0, 14);
            enc->setBuffer(sB,    haveScal ? start * 4 * sizeof(float) : 0, 18);
            enc->setBuffer(bScal, 0, 19);
            enc->setBuffer(bSG,   0, 20);
            enc->setBuffer(bMassFx, 0, 21);
            bindWords(enc);
            enc->setBuffer(bBad,  0, 30);
        });
    };

    using ExactItemParams = PSExactItemParams;   // (ps_kernel_params.h: the kernel's own struct)
    uint32_t const badAt = uint32_t(std::max<size_t>(nTetTotal, 1) * (pl.sumWFixed ? 2 : 1));   // past the per-tet totals
    auto runItems = [&](uint32_t phase) -> bool
    {
        // an item clips at most blockCells cells: far cheaper and more uniform than a whole tet.
        // Phase 0 writes only sumW (the grids are still zero), phase 1 only the grids (sumW is read).
        // --ps-linear-deposit runs both phases on the one pipeline that reads the phase from IP (kItemsPhase 2 in
        // metal/ps_deposit.metal): its weights come from the clip's first moments, which two pipelines may round
        // apart, so the summed total would miss the weights deposited by about 1e-8 of a tet's mass
        MTL::ComputePipelineState* const pso = pp.psoItems[P.fLinear != 0 ? 2u : phase];
        if (P.fLinear != 0 && !pp.psoItems[2])
        {   // that pipeline did not compile (pipesInit): the caller deposits this partition on the CPU
            lastErr = "the --ps-linear-deposit item pipeline did not compile: " + pp.errLinear;
            return false;
        }
        std::vector<MTL::Buffer*> extras;
        if (phase == 0) extras.push_back(bSumW);
        return runChunked(pso, nItems, 8192, 1024, MAX_CHUNK_ITEMS, extras, true, phase == 0,
                          static_cast<uint32_t*>(bSumW->contents()) + badAt,
                          [&](MTL::ComputeCommandEncoder* enc, size_t start, uint32_t n)
        {
            ExactItemParams const IP{ n, phase, blockCells, badAt, uint32_t(start), uint32_t(nTetTotal) };
            enc->setBuffer(bV,    0, 0);
            enc->setBuffer(bU,    0, 1);
            enc->setBuffer(bM,    0, 2);
            enc->setBuffer(bMass, 0, 3);
            enc->setBuffer(bMom,  0, 4);
            enc->setBuffer(bM2,   0, 5);
            enc->setBuffer(bGrad, 0, 6);
            enc->setBuffer(bStr,  0, 7);
            enc->setBuffer(bP,    0, 8);
            enc->setBuffer(bD,    0, 9);
            enc->setBuffer(bMomW, 0, 10);
            enc->setBuffer(bCaust,0, 11);
            enc->setBuffer(bSV,   0, 12);
            enc->setBuffer(bDV,   0, 13);
            enc->setBuffer(bDW,   0, 14);
            enc->setBuffer(bItems, 0, 15);   // (the whole map: the chunk starts at IP.itemBase)
            enc->setBuffer(bSumW, 0, 16);
            enc->setBytes(&IP, sizeof IP, 17);
            enc->setBuffer(bSc,   0, 18);
            enc->setBuffer(bScal, 0, 19);
            enc->setBuffer(bSG,   0, 20);
            enc->setBuffer(bMassFx, 0, 21);
            bindWords(enc);
            enc->setBuffer(bSumW, 0, 30);   // the same buffer as 16: its words when sumWFixed
        });
    };

    // The tets the items pass leaves over (sumW still 0: float-degenerate, or a window that clips
    // empty or lies outside the grid) run through depositFields, which falls through to the sampled
    // path / centroid fallback or defers them, exactly as the one-thread path does; their
    // deferrals (negated masses) are copied back into bM. A few leftovers run on compacted copies
    // of their records; many (a --box region cut from a larger cloud leaves most of an outer
    // partition's tets outside) run in place with every other tet's mass masked to 0 (skipped by
    // the kernel), 4 B per tet instead of a ~130 B copy each.
    auto runLeftovers = [&]() -> bool
    {
        auto tL = std::chrono::steady_clock::now();   // (DTFE_GPU_TIMING: this lambda's own host work, not its kernels)
        float* mb = static_cast<float*>(bM->contents());
        float const* sw = static_cast<float const*>(bSumW->contents());
        int64_t const* swx = static_cast<int64_t const*>(bSumW->contents());   // (sumWFixed: the words, little-endian)
        std::vector<uint32_t> left;
        for (size_t t = 0; t < nTetTotal; ++t)
            if (mb[t] > 0.f && !(pl.sumWFixed ? swx[t] > 0 : sw[t] > 0.f)) left.push_back(uint32_t(t));
        if (left.empty()) { ph.leftovers += PSGpuPhaseTimes::lap(tL); return true; }
        size_t const nL = left.size();
        bool const compact = nL <= nTetTotal / 16;
        auto gather = [&](MTL::Buffer* src, size_t stride) {
            MTL::Buffer* b = c.dev->newBuffer(nL * stride * sizeof(float), MTL::ResourceStorageModeShared);
            if (!b) return b;
            float const* s0 = static_cast<float const*>(src->contents());
            float* d0 = static_cast<float*>(b->contents());
            for (size_t k = 0; k < nL; ++k) std::memcpy(d0 + k * stride, s0 + size_t(left[k]) * stride, stride * sizeof(float));
            return b;
        };
        MTL::Buffer* lM = nullptr;
        MTL::Buffer *lV = bV, *lU = bU, *lD = bD, *lS = bSc;
        if (compact)
        {
            lV = gather(bV, PS_TET_STRIDE);
            lU = haveVels ? gather(bU, 12) : zeroBuf(4);
            lM = gather(bM, 1);
            lD = haveDens ? gather(bD, 4) : zeroBuf(4);
            lS = haveScal ? gather(bSc, 4) : zeroBuf(4);
        }
        else if ((lM = zeroBuf(nTetTotal * sizeof(float))))
        {
            float* lm = static_cast<float*>(lM->contents());
            for (uint32_t t : left) lm[t] = mb[t];
        }
        bool ok = lV && lU && lM && lD && lS;
        if (!ok) lastErr = "Metal buffer allocation failed (exact-deposit leftovers)";
        ph.leftovers += PSGpuPhaseTimes::lap(tL);
        if (ok) ok = runPerTet(lV, lU, lM, lD, lS, compact ? nL : nTetTotal, false);
        tL = std::chrono::steady_clock::now();
        if (ok)
        {
            float const* lm = static_cast<float const*>(lM->contents());
            for (size_t k = 0; k < nL; ++k)
            {
                float const v = lm[compact ? k : size_t(left[k])];
                if (v < 0.f) mb[left[k]] = v;
            }
        }
        if (lM) lM->release();
        if (compact) for (MTL::Buffer* b : {lV, lU, lD, lS}) if (b) b->release();
        ph.leftovers += PSGpuPhaseTimes::lap(tL);
        return ok;
    };

    // one pass sequence; each pass heals its own failed buffers from its snapshots (runChunked), so a
    // false here means the GPU gave up on this partition and the caller deposits it on the CPU
    bool const success = useItems ? runItems(0) && runItems(1) && runLeftovers()
                                  : runPerTet(bV, bU, bM, bD, bSc, nTetTotal, true);
    psHostReportPasses("Metal", "buffer", success, nFailed, nResumedCk, snap, timing);
    // the hold ends with the releases: the next partition allocates its whole buffer set as soon
    // as it takes the queue, and the auto-tuner books ONE deposit's buffers (ctxMutex serializes them), so this
    // call's buffers -- and the pool, whose completed command buffers still retain them -- go before the unlock
    tPh = std::chrono::steady_clock::now();
    for (MTL::Buffer* b : {bItems, bSumW}) if (b) b->release();
    ph.releases += PSGpuPhaseTimes::lap(tPh);
    if (!success)
    {
        err = lastErr;
        for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) b->release();
        if (bFaceMap) bFaceMap->release();   // (no face pass after a deposit the GPU gave up: the CPU makes both)
        pool->release();
        return false;
    }

    tPh = std::chrono::steady_clock::now();
    psGpuInitGrids(out, pl, fxUnits);   // (the output grids, allocated only now: a partition waiting for the GPU holds none)
    psHostCopyBack(PSHostCopyArgs{ B, pl, nCell, nTetTotal, fCaustic, out, deferred });
    ph.copyBack += PSGpuPhaseTimes::lap(tPh);

    for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) b->release();
    pool->release();
    // ... and the passes' snapshot, the host copies of the grids: the auto-tuner books one for the run
    // (snapshotBytes), so it goes before the next partition can take the GPU and make its own. No pass restores from
    // it any more; its seconds (snap.sec) stay for the timing line below
    std::vector<GpuSnapshot::Blob>().swap(snap.blobs);
    ph.releases += PSGpuPhaseTimes::lap(tPh);
    auto const tFree = tPh;
    // DTFE_GPU_FACES=1: the face pass in the same hold, now that the deposit's device buffers are gone (what lives
    // through it: the deposit's grids handed back in 'out' and the face words). It never touches the deposit's
    // result: a face pass that gives up leaves faceJob->ok false and the caller's CPU face walk makes the words.
    auto tHoldEnd = tFree;
    if (faceRun)
    {
        psHostFacePass(c, *faceJob, facePipe, bFaceMap, nFailed, nResumedCk, timing);
        tHoldEnd = std::chrono::steady_clock::now();
    }
    lock.unlock();
    // the deposit is kept (its grids are back; a unified-memory copy-back cannot fail): its dropped shares join the
    // run's counts (reported once at the end, DTFE.cpp)
    for (int k = 0; k < FX_DROP_SLOTS; ++k)
        if (badRun[k]) fixedShareDropped(k, badRun[k]);
    if (timing)   // the hold: uploads, passes, copy-back and releases (the GPU idles in all of it but the kernels)
    {
        double const held = std::chrono::duration<double>(tFree - tGot).count();
        psGpuReportPhases("Metal", PSMassAudit::partition, ph, snap.sec, held);
        if (faceRun) psHostReportFace(*faceJob);
        fprintf(stderr, "PS-DTFE Metal: partition %d waited %.3f s for the GPU and held it %.3f s\n",
                PSMassAudit::partition, std::chrono::duration<double>(tGot - tAsk).count(),
                std::chrono::duration<double>(tHoldEnd - tGot).count());
    }
    return true;
}

// The face sums of a job in a hold of their own (gpu_host.h): the pipeline and the items' map before the hold, the
// face pass in it (psHostFacePass), its core lent to the pool while it waits, as psGpuDepositFields does.
bool psGpuFaceSums(PSGpuFaceJob& job)
{
    bool const timing = Env::flag({"DTFE_GPU_TIMING"});
    job.ok = false;
    Ctx& c = ctx();
    if (!c.ready) { job.err = c.err; return false; }
    FacePipes const* fp = nullptr;
    MTL::Buffer* bMap = nullptr;
    if (!psHostFacePrepare(c, job, fp, bMap)) return false;
    if (job.nItems == 0)
    {   // (no record: nothing for the device, so no hold -- the words are zero)
        int nFailed = 0, nResumedCk = 0;
        return psHostFacePass(c, job, fp, bMap, nFailed, nResumedCk, timing);
    }
    auto const tAsk = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point tGot, tEnd;
    bool ok = false;
    {
        CoreLend const coreLend;
        std::unique_lock<std::mutex> lock(ctxMutex());
        tGot = std::chrono::steady_clock::now();
        int nFailed = 0, nResumedCk = 0;   // (the face pass's own: no deposit in this hold)
        ok = psHostFacePass(c, job, fp, bMap, nFailed, nResumedCk, timing);
        tEnd = std::chrono::steady_clock::now();
    }
    if (timing)
    {
        psHostReportFace(job);
        fprintf(stderr, "PS-DTFE Metal: partition %d waited %.3f s for the GPU and held it %.3f s (the face pass)\n",
                PSMassAudit::partition, std::chrono::duration<double>(tGot - tAsk).count(),
                std::chrono::duration<double>(tEnd - tGot).count());
    }
    return ok;
}
