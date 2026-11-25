/* The Metal GPU hosts' chunk checkpoints and cause-aware retry policy, shared by ps_metal_host.cc and
   dtfe_metal_host.cc: a failed command buffer restores the last snapshot of the written buffers and resumes there.
   Included by the hosts INSIDE their anonymous namespace, after <Metal/Metal.hpp>, "../message.h" and the std headers;
   the host includes "../run_mode.h" at file scope first. A binary links ONE Metal host, so the process-wide statics
   below (the pace, the fault hook) are that binary's. */

// Releases a buffer whose contents are no longer needed, its pages at once (marked empty first): Metal otherwise
// keeps a released buffer's memory for a while, so the footprint between partitions wanders. Both hosts release
// this way, so the footprint is the live memory the auto-tuner books.
inline void releaseBuffer(MTL::Buffer*& b)
{
    if (!b) return;
    b->setPurgeableState(MTL::PurgeableStateEmpty);
    b->release();
    b = nullptr;
}

// What a failed command buffer says about its cause (both watchdog flavours share Timeout code 2: read the text):
//   Victim    "Innocent Victim": another process hung the GPU -> retry at once at the same pace
//   Watchdog  "Impacting Interactivity" / a GPU timeout -> shorter buffers, wider gaps
//   Resource  out of memory, page fault, invalid resource, internal -> as Watchdog, after 2 s to settle
//   Fatal     device removed / not permitted / access revoked -> straight to the CPU
enum class GpuFault { Victim, Watchdog, Resource, Fatal };

char const* gpuFaultName(GpuFault f)
{
    switch (f)
    {
        case GpuFault::Victim:   return "innocent victim";
        case GpuFault::Watchdog: return "watchdog";
        case GpuFault::Resource: return "resource";
        default:                 return "fatal";
    }
}

GpuFault classifyGpuFault(long code, std::string const& desc)
{
    auto has = [&](char const* s) { return desc.find(s) != std::string::npos; };
    if (has("Innocent Victim") || has("InnocentVictim")) return GpuFault::Victim;
    if (has("Impacting Interactivity") || has("ImpactingInteractivity") || has("Timeout")
        || code == long(MTL::CommandBufferErrorTimeout)) return GpuFault::Watchdog;
    if (code == long(MTL::CommandBufferErrorDeviceRemoved) || code == long(MTL::CommandBufferErrorNotPermitted)
        || code == long(MTL::CommandBufferErrorAccessRevoked)) return GpuFault::Fatal;
    return GpuFault::Resource;
}

// The run's retry counters (run_progress.h: the sidecar's gpu_retries and
// gpu_retries_by_cause, DTFE.cpp's closing 'GPU deposit retries:' line) by the cause a reader can act on:
// 0 innocent victim, 1 timeout (the watchdog), 2 out of memory -- split off the Resource class, which also
// holds page faults and internal errors -- and 3 everything else (those, and the fatal causes). The index
// only: each host bumps its own binary's counters with it (countGpuRetry in the host).
int gpuRetryCause(GpuFault f, long code, std::string const& desc)
{
    int k = 3;
    if (f == GpuFault::Victim) k = 0;
    else if (f == GpuFault::Watchdog) k = 1;
    else if (f == GpuFault::Resource && (code == long(MTL::CommandBufferErrorOutOfMemory)
             || desc.find("out of memory") != std::string::npos || desc.find("OutOfMemory") != std::string::npos
             || desc.find("Insufficient Memory") != std::string::npos)) k = 2;
    return k;
}

// The dispatch pace, kept across partitions once escalated (the batches of dtfe_metal_host.cc share it). It steps
// back one notch (buffers x4, gaps /3, toward the run mode's base, gpuPace) after every DTFE_GPU_PACE_RECOVER_SEC
// (default 90) s of GPU work without a fault of ours, counted in GPU work, not wall time. An innocent victim neither
// escalates nor resets the clean count. Guarded by the host's ctxMutex (touched only inside psGpuDepositFields or
// runBatch, whose callers hold it).
double const GPU_PACE_TARGET0 = 0.25;   // the default pace (performance's base; the state of a caller that skips readOptions)
useconds_t const GPU_PACE_GAP0 = 8000;
struct GpuPace
{
    double targetSec = 0.25; useconds_t gapUs = 8000; double cleanSec = 0.;
    double target0 = GPU_PACE_TARGET0; useconds_t gap0 = GPU_PACE_GAP0;   // the base the recovery returns to (gpuPace)
    // a failure of our own (a watchdog kill, a resource error; not an innocent victim): escalate for the rest of the
    // process -- 4x shorter buffers (at least 20 ms), 3x wider gaps (at most 50 ms) -- and count the clean work afresh
    void onFault()
    {
        targetSec = std::max(targetSec / 4.0, 0.02);
        gapUs     = std::min(gapUs * 3, useconds_t(50000));
        cleanSec  = 0.;
    }
    // a buffer that succeeded after el seconds: a pace escalated past the base (target0, gap0) counts the clean GPU
    // work (the buffer and its gap) and, once that reaches recoverSec, steps back one notch toward the base, the
    // inverse of onFault. True when it stepped back, with 'clean' = the clean seconds that earned it (the count then
    // starts afresh), for the caller's message.
    bool onClean(double const el, double const recoverSec, double& clean)
    {
        if (!(targetSec < target0 || gapUs > gap0)) return false;
        cleanSec += el + gapUs / 1e6;
        if (!(cleanSec >= recoverSec)) return false;
        targetSec = std::min(targetSec * 4.0, target0);
        gapUs     = std::max(gapUs / 3, gap0);
        clean     = cleanSec;
        cleanSec  = 0.;
        return true;
    }
};
// The process's pace, from the run mode's base (run_mode.h); a recovery returns to it. First touched inside a
// deposit, after readOptions applied the mode (a caller that never reads the options, libDTFE, keeps performance's).
GpuPace& gpuPace()
{
    static GpuPace p = [] {
        GpuPace q;
        q.target0 = q.targetSec = runModeState().gpuTargetSec;
        q.gap0    = q.gapUs     = GPU_PACE_GAP0;
        return q;
    }();
    return p;
}

// The test hook DTFE_GPU_FAULT_AT=<k>[:<kind>]: the k-th command buffer of the PROCESS (1-based, over every pass and
// partition; real failures are not counted) runs to completion, its atomics deposited (the worst case), and is then
// reported as failed, so only a correct restore keeps the grids exact. Kinds: 'innocent' (default), 'timeout', 'oom',
// 'persist' (every buffer from the k-th on fails as an innocent victim: the CPU must take the partition). One-shot
// except 'persist'. The binary's one process-wide hook.
struct GpuFaultHook { long at = 0; std::string kind = "innocent"; long count = 0; bool fired = false; };
GpuFaultHook& gpuFaultHook()
{
    static GpuFaultHook h = [] {
        GpuFaultHook r;
        if (const char* env = Env::value("DTFE_GPU_FAULT_AT"))
        {
            r.at = atol(env);
            if (const char* colon = strchr(env, ':')) r.kind = colon + 1;
        }
        return r;
    }();
    return h;
}

// true (and the injected error's description in 'desc', its code in 'code') when this buffer is to fail
bool gpuFaultInjected(std::string& desc, long& code)
{
    GpuFaultHook& h = gpuFaultHook();
    if (h.at <= 0) return false;
    ++h.count;
    bool const persist = h.kind == "persist";
    if (h.count < h.at || (h.fired && !persist)) return false;
    h.fired = true;
    code = 0;
    if (h.kind == "timeout")  { desc = "injected fault (DTFE_GPU_FAULT_AT): Impacting Interactivity"; code = long(MTL::CommandBufferErrorTimeout); }
    else if (h.kind == "oom") { desc = "injected fault (DTFE_GPU_FAULT_AT): out of memory"; code = long(MTL::CommandBufferErrorOutOfMemory); }
    else desc = std::string("injected fault (DTFE_GPU_FAULT_AT") + (persist ? ", persistent" : "") + "): Innocent Victim";
    return true;
}

// One snapshot of a set of buffers, host-side. A buffer known to be all zeros is recorded as such (no copy):
// the first pass's grids at its start, so a short deposit never allocates the spare copy at all.
struct GpuSnapshot
{
    struct Blob { std::unique_ptr<char[]> p; size_t n = 0; bool zero = false; };
    std::vector<MTL::Buffer*> bufs;
    std::vector<Blob> blobs;
    size_t start = 0;      // the pass's first thread NOT yet covered by the snapshot
    bool   valid = false;
    int    taken = 0;      // periodic snapshots (not the pass-start ones)
    double sec   = 0.;     // time spent copying
    size_t bytes = 0;      // largest snapshot held

    // copies every buffer (zeroMask[i]: buffer i is known zero, not copied); false when the host memory for the
    // spare copy is not there (the caller then runs without checkpoints)
    bool take(std::vector<MTL::Buffer*> const& bs, std::vector<bool> const& zeroMask, size_t at)
    {
        auto const t0 = std::chrono::steady_clock::now();
        if (blobs.size() < bs.size()) blobs.resize(bs.size());
        // allocate everything BEFORE copying anything: a failed allocation leaves the previous snapshot whole
        for (size_t i = 0; i < bs.size(); ++i)
        {
            Blob& b = blobs[i];
            size_t const n = bs[i]->length();
            if ((i < zeroMask.size() && zeroMask[i]) || (b.p && b.n >= n)) continue;
            b.p.reset(new (std::nothrow) char[n]);
            b.n = b.p ? n : 0;
            if (!b.p) return false;
        }
        bufs = bs;
        size_t held = 0;
        for (size_t i = 0; i < bs.size(); ++i)
        {
            Blob& b = blobs[i];
            b.zero = i < zeroMask.size() && zeroMask[i];
            if (b.zero) continue;
            size_t const n = bs[i]->length();
            std::memcpy(b.p.get(), bs[i]->contents(), n);
            held += n;
        }
        start = at;
        valid = true;
        bytes = std::max(bytes, held);
        sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
    void restore()
    {
        auto const t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < bufs.size(); ++i)
        {
            size_t const n = bufs[i]->length();
            if (blobs[i].zero) std::memset(bufs[i]->contents(), 0, n);
            else std::memcpy(bufs[i]->contents(), blobs[i].p.get(), n);
        }
        sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------- the pass loop
// Both hosts run their command buffers through gpuRunPass below: ps_metal_host.cc's runChunked (one pass of a
// partition's deposit: the per-tet kernel, or the exact items' two phases and their leftovers) and
// dtfe_metal_host.cc's runBatch (one sub-domain batch), so the policy above is applied by ONE loop.

// each host's own: its binary's retry counters (run_progress.h), bumped once per failed buffer
void countGpuRetry(GpuFault f, long code, std::string const& desc);

// the policy's limits: a chunk failing this often, or this many failed buffers in a deposit (PS: across its
// passes; standard DTFE: in a batch), hands the work to the CPU
int const GPU_MAX_FAILS_SAME_CHUNK = 3, GPU_MAX_FAILS_TOTAL = 16;

size_t gpuChunkAsIs(size_t n) { return n; }

// The state a pass shares with the passes before and after it (PS-DTFE's deposit holds one snapshot, one failure
// count and one error across its passes; a standard-DTFE batch is a single pass).
struct GpuRetryRun
{
    GpuSnapshot&        snap;
    int&                nFailed;      // failed buffers so far (GPU_MAX_FAILS_TOTAL counts them)
    int&                nResumedCk;   // of which resumed from a periodic checkpoint
    std::string&        lastErr;      // the last failure, the caller's fallback message
    unsigned long long* badRun;       // [FX_DROP_SLOTS]: the shares the non-finite guard dropped so far, rolled back
                                      // with every restore so a failed buffer and its redo count once
};

// One pass: what it runs, how it is chunked and checkpointed, and (out) what it cost.
struct GpuPassSpec
{
    char const*                tag = "";         // the log lines' prefix: "PS-DTFE Metal", "DTFE Metal"
    char const*                scope = "";       // what GPU_MAX_FAILS_TOTAL counts in, for the give-up message
    MTL::CommandQueue*         q = nullptr;
    MTL::ComputePipelineState* pso = nullptr;
    size_t                     nThreads = 0, minChunk = 1, maxChunk = 1;
    size_t                   (*align)(size_t) = gpuChunkAsIs;   // a chunk size as the kernel needs it
    bool                       ckOn = false;     // a pass-start snapshot is held (without one the first failure gives up)
    std::vector<MTL::Buffer*> const* ckBufs = nullptr;   // the buffers a periodic checkpoint copies ...
    std::vector<bool>          ckMask;           // ... but those known zero (GpuSnapshot::take)
    long                       ckChunks = -1;    // DTFE_GPU_CHECKPOINT_CHUNKS (-1: every ~10 s of GPU work)
    double                     paceRecoverSec = 90.;   // DTFE_GPU_PACE_RECOVER_SEC
    uint32_t*                  badW = nullptr;   // [FX_DROP_SLOTS]: the guard's counter words, zeroed per buffer
    // out
    size_t submitted = 0, buffers = 0;   // command buffers committed, and those of them that succeeded
    double busy = 0., gaps = 0.;         // the successful buffers' seconds; the idle gaps' and settling sleeps'
};

// The loop: serialized command buffers of 'chunk' threads, each retargeted toward the pace's buffer length and
// followed by its idle gap, with a periodic checkpoint every DTFE_GPU_CHECKPOINT_CHUNKS buffers. A failed buffer (it
// may have deposited part of its atomics) gives up (false, run.lastErr set) on a fatal cause, without a snapshot,
// or after GPU_MAX_FAILS_SAME_CHUNK / GPU_MAX_FAILS_TOTAL failures; else it escalates the pace, shrinks the chunk,
// calls restore() (grids back to the snapshot, returns the first thread to redo) and escalate() (the host's part).
template <class Encode, class Restore, class Escalate>
bool gpuRunPass(GpuPassSpec& p, GpuRetryRun& run, size_t& chunk, Encode&& encode, Restore&& restore,
                Escalate&& escalate)
{
    GpuPace& pace = gpuPace();
    NS::UInteger tg = p.pso->maxTotalThreadsPerThreadgroup();
    if (tg > 256) tg = 256;
    bool ckPeriodic = false;                   // the held snapshot is a periodic checkpoint (not the pass start)
    unsigned long long badCk[FX_DROP_SLOTS];   // run.badRun at the held snapshot
    std::memcpy(badCk, run.badRun, sizeof badCk);
    // failures of the 'same chunk': a failure that comes before the redo got past the previous failed
    // buffer's end (chunk boundaries move with the timing controller, so a start offset is no identity)
    int    failsHere = 0;
    size_t failEnd = 0;
    size_t start = 0, sinceCk = 0;
    while (start < p.nThreads)
    {
        uint32_t const n = uint32_t(std::min(chunk, p.nThreads - start));
        std::memset(p.badW, 0, FX_DROP_SLOTS * 4);   // (the previous buffer completed: waitUntilCompleted below)
        auto const t0 = std::chrono::steady_clock::now();
        MTL::CommandBuffer* cb = p.q->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(p.pso);
        encode(enc, start, n);
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        ++p.submitted;
        double const el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::string desc;
        long code = 0;
        bool failed = false;
        if (cb->status() == MTL::CommandBufferStatusError)
        {
            NS::Error* e = cb->error();
            desc = e ? e->localizedDescription()->utf8String() : "unknown";
            code = e ? long(e->code()) : 0;
            failed = true;
        }
        else failed = gpuFaultInjected(desc, code);   // the test hook (the buffer DID deposit)

        if (failed)
        {
            ++run.nFailed;
            GpuFault const cause = classifyGpuFault(code, desc);
            countGpuRetry(cause, code, desc);   // the run's counters count every failed buffer (run_progress.h)
            run.lastErr = std::string("Metal command buffer failed: ") + desc + " [" + gpuFaultName(cause) + "]";
            failsHere = (failsHere > 0 && start < failEnd) ? failsHere + 1 : 1;
            failEnd = std::max(failEnd, start + n);
            if (cause == GpuFault::Fatal || !p.ckOn || failsHere >= GPU_MAX_FAILS_SAME_CHUNK
                || run.nFailed >= GPU_MAX_FAILS_TOTAL)
            {
                if (!p.ckOn) run.lastErr += " (no checkpoint held)";
                else if (failsHere >= GPU_MAX_FAILS_SAME_CHUNK)
                    run.lastErr += " (" + std::to_string(failsHere) + " failures of the same chunk)";
                else if (run.nFailed >= GPU_MAX_FAILS_TOTAL)
                    run.lastErr += " (" + std::to_string(run.nFailed) + " failed buffers in this " + p.scope + ")";
                return false;
            }
            if (cause != GpuFault::Victim)
            {   // escalate (for the rest of the process): 4x shorter buffers, wider gaps, smaller chunk
                pace.onFault();
                chunk = p.align(std::max(chunk / 4, p.minChunk));
                escalate();
            }
            size_t const resumeAt = restore();
            std::memcpy(run.badRun, badCk, sizeof badCk);
            if (ckPeriodic) ++run.nResumedCk;
            fprintf(stderr, "%s%s: %s -- resuming from the %s at thread %zu of %zu "
                            "(redoing %zu; failure %d of this chunk, target %.0f ms buffers, %.0f ms gaps)%s\n",
                    MESSAGE::cYellow(), p.tag, run.lastErr.c_str(), ckPeriodic ? "checkpoint" : "pass start",
                    resumeAt, p.nThreads, start + n - resumeAt, failsHere,
                    pace.targetSec * 1e3, pace.gapUs / 1e3, MESSAGE::cReset());
            start = resumeAt;
            sinceCk = 0;
            // an innocent victim retries at once; our own buffers' failures let the system settle first
            auto const tGap = std::chrono::steady_clock::now();
            if (cause == GpuFault::Resource) sleep(2);
            else if (cause == GpuFault::Watchdog) sleep(1);
            p.gaps += std::chrono::duration<double>(std::chrono::steady_clock::now() - tGap).count();
            continue;
        }

        start += n;
        ++p.buffers;
        ++sinceCk;
        p.busy += el;
        for (int k = 0; k < FX_DROP_SLOTS; ++k) run.badRun[k] += p.badW[k];
        if (el > 1e-4)   // retarget the next chunk toward targetSec (bounded rescale)
        {
            double scale = pace.targetSec / el;
            if (scale < 0.5) scale = 0.5;
            if (scale > 4.0) scale = 4.0;
            chunk = p.align(std::min(std::max(size_t(double(chunk) * scale), p.minChunk), p.maxChunk));
        }
        double clean = 0.;
        if (pace.onClean(el, p.paceRecoverSec, clean))   // an escalated pace stepped back after a clean stretch
        {   // a step that reached the run mode's base (target0, gap0) says so
            char const* const atBase = pace.targetSec >= pace.target0 && pace.gapUs <= pace.gap0 ? " (the run mode's)" : "";
            fprintf(stderr, "%s: %.0f s of GPU work without a fault -- pace eased to %.0f ms "
                            "buffers, %.0f ms gaps%s\n", p.tag, clean, pace.targetSec * 1e3, pace.gapUs / 1e3, atBase);
        }
        long const every = p.ckChunks >= 0 ? p.ckChunks : std::max(1L, long(std::lround(10.0 / pace.targetSec)));
        if (p.ckOn && every > 0 && long(sinceCk) >= every && start < p.nThreads)
        {   // the queue is drained (waitUntilCompleted above): the buffers hold exactly [0, start)
            if (run.snap.take(*p.ckBufs, p.ckMask, start))
            { ++run.snap.taken; ckPeriodic = true; sinceCk = 0; std::memcpy(badCk, run.badRun, sizeof badCk); }
            // else: no host memory for the spare copy; the last snapshot stays valid
        }
        auto const tGap = std::chrono::steady_clock::now();
        usleep(pace.gapUs);   // deliberate idle gap: lets WindowServer take the GPU
        p.gaps += std::chrono::duration<double>(std::chrono::steady_clock::now() - tGap).count();
    }
    return true;
}
