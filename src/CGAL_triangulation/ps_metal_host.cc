/* Metal backend of the PS-DTFE GPU deposit (implements gpu_host.h), compiled only when METAL=1. Owns the metal-cpp symbols
   for the whole binary; the kernel source is embedded at build time (o_ps/ps_deposit_msl.h) and runtime-compiled once, by
   the first deposit (pipes(), behind a "#define PS_FX_WORDS 0" line); the pipeline states are cached. */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

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
#include "ps_deposit_msl.h"   // generated: static const char PS_DEPOSIT_MSL[]
#include "../thread_count.h"   // CorePool: the core of a thread that waits for the GPU is lent to the pool
#include "../run_mode.h"       // runModeState(): the run mode's pace base (gpu_metal_retry.h) and exact-item cap
#include <atomic>
extern std::atomic<int> psGpuRetries;   // ps_interpolation.cc (run_progress.h)
extern std::atomic<int> psGpuRetriesByCause[4];   // ... by cause: innocent, timeout, oom, other
#include "../message.h"       // ANSI colour helpers only (retry warning below)

namespace {

using DepositParams = PSDepositParams;   // shared host struct (ps_deposit_params.h)

// The deposit's pipelines. The kernel source takes the word size as a preprocessor constant (PS_FX_WORDS in
// metal/ps_deposit.metal): 0, the 64-bit words, the one size the host compiles (pipesInit).
struct Pipes
{
    MTL::ComputePipelineState* pso  = nullptr;   // depositFields: one thread per tet
    MTL::ComputePipelineState* psoItems[3] = { nullptr, nullptr, nullptr };   // depositExactItems: one per (tet, cell
                                                                     // block), a pipeline per pass (its kItemsPhase),
                                                                     // and [2] both passes of fLinear (the linear
                                                                     // deposit: no option sets it, the host passes 0)
    std::string                err;
    std::string                errLinear;   // why psoItems[2] is null: its compile failed (the others stand)
    bool                       ready = false;
};

struct Ctx
{
    MTL::Device*               dev  = nullptr;
    MTL::CommandQueue*         q    = nullptr;
    Pipes                      pipes;          // the 64-bit words (PSDepositParams.fxWords 0)
    std::once_flag             pipesOnce;      // pipes()
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

// The deposit's pipelines (the runtime compile of the kernel source behind its "#define PS_FX_WORDS 0" line, with the
// compiler's default options): pipes() runs this once.
void pipesInit(Ctx& c, Pipes& p)
{
    // (the compiler's messages are autoreleased: copied into p.err / p.errLinear before the pool goes)
    struct Pool
    {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        ~Pool() { pool->release(); }
    } const pool;
    auto const t0 = std::chrono::steady_clock::now();
    std::string const src = std::string("#define PS_FX_WORDS 0\n") + PS_DEPOSIT_MSL;
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
        // Pipeline 2 serves only fLinear (the linear deposit, which no option sets): when it fails it stays null
        // and the device still serves every other run (runItems would send a linear run's partitions to the CPU).
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
        fprintf(stderr, "PS-DTFE Metal: pipelines compiled in %.3f s\n",
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

// The deposit's pipelines, compiled by the first caller that asks (as ctx())
Pipes const& pipes(Ctx& c)
{
    std::call_once(c.pipesOnce, pipesInit, std::ref(c), std::ref(c.pipes));
    return c.pipes;
}

} // namespace


std::string gpuBackendName() { return "Metal"; }

std::string gpuDeviceName()
{
    Ctx& c = ctx();   // (thread-safe: std::call_once)
    return c.dev ? std::string(c.dev->name()->utf8String()) : std::string();
}


// ---- chunk-level checkpoints and the cause-aware retry policy ----
// Every pass snapshots the buffers its kernel writes (grids, the exact items' sumW, the per-tet masses the kernel negates
// when it defers a tet) at its start and every DTFE_GPU_CHECKPOINT_CHUNKS chunks; a failed buffer (which may have deposited
// part of its atomics) restores the last snapshot and continues from its chunk. The policy and its pass loop, gpuRunPass, live
// in gpu_metal_retry.h, shared with the standard-DTFE host; psGpuDepositFields' runChunked runs each pass through it.
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
    // vels is empty for density-only runs (the kernel never reads it then): bind a dummy. Whether it is real matters at bind
    // time: the per-chunk offset below must NOT be applied to the 4-byte dummy, or every chunk after the first binds past its
    // end (undefined behaviour the kernel survives only because it never dereferences the pointer).
    bool const haveVels = not vels.empty();
    MTL::Buffer* bU = vels.empty() ? zeroBuf(4)
                    : c.dev->newBuffer(vels.data(), vels.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(vels);
    MTL::Buffer* bM = c.dev->newBuffer(masses.data(), masses.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::vector<float>().swap(masses);
    // 'dens' carries either the per-tet vertex densities (fLinear, which the host does not set) or, when fCaustic runs without
    // fLinear, the per-tet caustic mask in slot 0; dummy when neither. Either way a real buffer needs its per-chunk offset,
    // keyed on the vector being non-empty, not on fLinear.
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

    if (pl.massFixed)
    {
        // the words themselves, not their float decode: adding the deferred tets' shares into decoded floats
        // would round away every share below half an ulp of a busy cell, always downward. Low word first: on this
        // little-endian host each pair is the int64 (ps_deposit_params.h).
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
    // host, as for the mass) or its floats
    MTL::Buffer* const fxBuf[PS_FX_N] = { bMom, bM2, bGrad, bMomW, bSV, bDV, nullptr, bScal, bSG };
    std::vector<float>* const flt[PS_FX_N] = { &out.mom, &out.m2, &out.grad, &out.momw, &out.streamvol, &out.dispvel,
                                               nullptr, &out.scal, &out.sgrad };
    for (int g = 0; g < PS_FX_N; ++g)
    {
        if (!pl.sel[g] || !fxBuf[g]) continue;
        size_t const n = nCell * PS_FX_COMPS[g];
        if (pl.fxOn[g]) { out.fx[g].resize(n); std::memcpy(out.fx[g].data(), fxBuf[g]->contents(), n * 8); }
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
                          PSGpuGrids& out, std::vector<uint32_t>& deferred, std::string& err)
{
    // DTFE_GPU_TIMING=1: one line per kernel run (threads, buffers, GPU-busy and wall seconds),
    // and per partition where its call went, before and in its hold of the GPU (PSGpuPhaseTimes: timed always,
    // printed only then)
    bool const timing = Env::flag({"DTFE_GPU_TIMING"});
    PSGpuPhaseTimes ph;
    auto tPh = std::chrono::steady_clock::now();
    Ctx& c = ctx();
    if (!c.ready) { err = c.err; return false; }
    ph.device += PSGpuPhaseTimes::lap(tPh);

    // Before the hold (ctxMutex below): the plan, the parameter block and the exact items' map, from the host
    // records -- work that needs no queue, done while another partition's passes hold the GPU.
    // The plan (the selected grids, their fixed-point form and device sizes; gpu_host.h).
    PSGpuPlan const pl = psGpuPlan(subDims, fVel, fDisp, fGrad, fVolW, fCaustic, fExact, fScal, fSGrad, masses,
                                   massExtra, fxUnits);
    // The pipelines (the 64-bit words, as the parameter block below says: P.fxWords 0): compiled by the first
    // deposit that asks (pipes()), booked as the device's time.
    ph.plan += PSGpuPhaseTimes::lap(tPh);
    Pipes const& pp = pipes(c);
    ph.device += PSGpuPhaseTimes::lap(tPh);
    if (!pp.ready) { err = pp.err; return false; }
    size_t const nCell = pl.nCell;
    deferred.clear();
    if (masses.empty())                 // nothing to deposit (empty partition; the caller may still add released tets)
    {
        psGpuInitGrids(out, pl, fxUnits);
        psGpuZeroWords(out, pl);
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

    // --exact: split the work below the tet (depositExactItems), items of blockCells cells of a tet's window (psExactItemBounds)
    // in the compact map psBuildExactItemMap, following the caller's cost order so a SIMD group's threads do similar work.
    // DTFE_GPU_ITEMS=0 keeps the one-thread-per-tet kernel (A/B).
    bool useItems = fExact;
    useItems = useItems && Env::flag("DTFE_GPU_ITEMS", true);
    MTL::Buffer* bItems = nullptr;
    MTL::Buffer* bSumW  = nullptr;
    size_t nItems = 0;
    constexpr uint32_t kExactBlockCells = 16;   // the block's cells (psExactItemBounds raises it if too many items)
    uint32_t blockCells = kExactBlockCells;
    psHostExactItemMap(PSHostItemsArgs{ c, verts, P, nGrid, nTetTotal, periodic, windowMode, timing, useItems,
                                        blockCells, bItems, nItems, ph });

    auto const tAsk = std::chrono::steady_clock::now();   // DTFE_GPU_TIMING: the wait for the queue, and its hold
    // From here to the return this thread sleeps: first in the queue for the GPU, then in each buffer's waitUntilCompleted. Its
    // core is lent to the core pool for that time (thread_count.h), so CPU work that follows the free cores (the sub-grid merge,
    // the normalization, the deposit teams of a '--gpu-share' run) can take it. A claim on the pool never changes a result.
    CoreLend const coreLend;
    std::unique_lock<std::mutex> lock(ctxMutex());   // serialize dispatches (single queue); let go after the releases
    auto const tGot = std::chrono::steady_clock::now();
    tPh = tGot;

    // newBuffer(ptr,...) copies the host array into the MTL buffer; each host array is freed IMMEDIATELY so at most one is
    // double-held. On failure the CPU fallback re-deposits from the triangulation, never from these arrays.
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
            releaseBuffer(bItems);
            useItems = false;
        }
    }
    ph.uploads += PSGpuPhaseTimes::lap(tPh);
    if (!bV || !bU || !bM || !bD || !bSc || !bP || !bMass || !bMassFx || !bMom || !bM2 || !bGrad || !bStr || !bMomW || !bCaust || !bSV || !bDV || !bDW || !bScal || !bSG || !bBad)
    {
        err = "Metal buffer allocation failed (out of GPU-visible memory?)";
        for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad,bItems,bSumW}) releaseBuffer(b);
        pool->release();
        return false;
    }

    // Dispatch in CHUNKS, one short command buffer each, serialized with a host-side gap between buffers: the macOS GPU
    // watchdog kills a long buffer when the display needs the GPU, and back-to-back buffers too (it reacts to starvation); the
    // gaps let WindowServer in. The chunk size adapts toward pace.targetSec. A killed buffer may have deposited part of its
    // atomics, so a failure resumes from the pass's last snapshot; the caller deposits on the CPU only after a fatal error, 3
    // failures of one chunk or 16 in the deposit, or when no snapshot can be held. PS_GPU_CHUNK=<n> sets the starting chunk size.
    size_t const MIN_CHUNK = fExact ? 50 : 2000;
    size_t const MAX_CHUNK = 500000;
    // The exact items have a cap of their own, the run mode's (run_mode.h psItemCap): the short buffers of gentle and balanced
    // keep the display responsive, performance raises it at the cost of longer waits. The chunk size changes no item's deposit
    // and the fixed-point sums do not depend on the order.
    size_t const MAX_CHUNK_ITEMS = runModeState().psItemCap;
    GpuPace& pace = gpuPace();        // the run mode's target and gaps, until escalated
    size_t baseChunk = psHostFirstChunk(fExact, pace.targetSec, MIN_CHUNK);
    std::string lastErr;

    PSHostCkEnv const ck = psHostCheckpointEnv();
    long ckChunks = ck.ckChunks;
    double paceRecoverSec = ck.paceRecoverSec;
    std::vector<MTL::Buffer*> const gridBufs = { bMass, bMassFx, bMom, bM2, bGrad, bStr, bMomW, bCaust, bSV, bDV, bDW, bScal, bSG };
    GpuSnapshot snap;
    int nFailed = 0, nResumedCk = 0;   // failed buffers of this deposit, and how many resumed from a periodic checkpoint
    // The shares the non-finite guard dropped in this deposit, by slot (psFixedShareOk): each pass zeroes its counter words
    // before a command buffer and adds them here after one that succeeded, and a restore rolls these sums back to the snapshot's,
    // so failed and redone buffers count once. Handed to the run's counters only when the deposit is kept.
    unsigned long long badRun[FX_DROP_SLOTS] = {};
    GpuRetryRun run{ snap, nFailed, nResumedCk, lastErr, badRun };

    // One paced, chunked run of 'pso' over nThreads threads (gpuRunPass); bind(enc, start, n) binds the buffers for the chunk
    // [start, start+n). 'extras' are the non-grid buffers the kernel WRITES (sumW, the negated per-tet masses); gridsZero: the
    // grids are all zero at the pass's start; gridsKeepZero: the pass never writes them (exact items' phase 0). False (lastErr
    // set) when the pass gave up (the caller falls back). badW: this pass's counter words of the guard (badRun).
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

    // depositFields over the tets [0,nT) of the given records. The grids are bound whole; the per-tet buffers at the chunk's
    // offset. The kernel negates the masses of the tets it defers, so mB is part of the snapshot: a redo runs the same path.
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
        // an item clips at most blockCells cells. Phase 0 writes only sumW, phase 1 only the grids. fLinear (the unused linear
        // deposit) runs both phases on the one pipeline (kItemsPhase 2): its weights come from the clip's first moments, which
        // two pipelines may round apart, so the summed total would miss the weights deposited.
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

    // The tets the items pass leaves over (sumW still 0: float-degenerate, or a window that clips empty or lies outside the grid)
    // run through depositFields, which falls through to the sampled path / centroid fallback or defers them as the one-thread
    // path does; deferrals are copied back into bM. A few run on compacted copies of their records; many (a --box region cut
    // from a larger cloud) run in place with every other tet's mass masked to 0, 4 B per tet instead of a record copy.
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
        releaseBuffer(lM);
        if (compact) for (MTL::Buffer* b : {lV, lU, lD, lS}) releaseBuffer(b);
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
    releaseBuffer(bItems);
    releaseBuffer(bSumW);
    ph.releases += PSGpuPhaseTimes::lap(tPh);
    if (!success)
    {
        err = lastErr;
        for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) releaseBuffer(b);
        pool->release();
        return false;
    }

    tPh = std::chrono::steady_clock::now();
    psGpuInitGrids(out, pl, fxUnits);   // (the output grids, allocated only now: a partition waiting for the GPU holds none)
    psHostCopyBack(PSHostCopyArgs{ B, pl, nCell, nTetTotal, fCaustic, out, deferred });
    ph.copyBack += PSGpuPhaseTimes::lap(tPh);

    for (MTL::Buffer* b : {bV,bU,bM,bD,bSc,bP,bMass,bMassFx,bMom,bM2,bGrad,bStr,bMomW,bCaust,bSV,bDV,bDW,bScal,bSG,bBad}) releaseBuffer(b);
    pool->release();
    // ... and the passes' snapshot, the host copies of the grids: the auto-tuner books one for the run (snapshotBytes), so
    // it goes before the next partition can take the GPU and make its own; its seconds (snap.sec) stay for the timing line below
    std::vector<GpuSnapshot::Blob>().swap(snap.blobs);
    ph.releases += PSGpuPhaseTimes::lap(tPh);
    auto const tFree = tPh;
    lock.unlock();
    // the deposit is kept (its grids are back; a unified-memory copy-back cannot fail): its dropped shares join the
    // run's counts (reported once at the end, DTFE.cpp)
    for (int k = 0; k < FX_DROP_SLOTS; ++k)
        if (badRun[k]) fixedShareDropped(k, badRun[k]);
    if (timing)   // the hold: uploads, passes, copy-back and releases (the GPU idles in all of it but the kernels)
    {
        double const held = std::chrono::duration<double>(tFree - tGot).count();
        psGpuReportPhases("Metal", PSMassAudit::partition, ph, snap.sec, held);
        fprintf(stderr, "PS-DTFE Metal: partition %d waited %.3f s for the GPU and held it %.3f s\n",
                PSMassAudit::partition, std::chrono::duration<double>(tGot - tAsk).count(),
                std::chrono::duration<double>(tFree - tGot).count());
    }
    return true;
}
