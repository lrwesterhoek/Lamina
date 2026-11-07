/* The parts of the GPU hosts' retry policy that do not depend on the backend, shared by its two twins,
   gpu_metal_retry.h (Metal) and gpu_cuda_retry.cuh (CUDA/HIP): the classes of a failed command buffer / kernel and
   the DTFE_GPU_FAULT_AT test hook's settings. Each twin classifies its own runtime's errors into GpuFault and
   injects the hook's faults in its own terms (gpuFaultInjected), and keeps its own pace and snapshot.

   Included by the twins, so INSIDE the hosts' anonymous namespace, after "../message.h" (which brings
   env_knobs.h) and <cstdlib> <cstring> <string>; the hook below is that binary's one process-wide hook. */
#ifndef GPU_RETRY_COMMON_H
#define GPU_RETRY_COMMON_H

// What a failed command buffer / kernel says about its cause (each twin says how its runtime's errors map here):
//   Victim    another process's work hung the GPU and ours was collateral: retry at once at the same pace
//   Watchdog  our buffers / kernels ran too long for the display: shorter ones, wider gaps
//   Resource  out of memory and the other non-fatal errors: escalate as for the watchdog, settle 2 s first
//   Fatal     no retry can help (the device is gone, or its context is lost): straight to the CPU
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

// The test hook DTFE_GPU_FAULT_AT=<k>[:<kind>] (each twin's gpuFaultInjected says what it does): the 1-based
// index k of the buffer / kernel of the PROCESS to fail, the kind ('innocent', the default, 'timeout', 'oom' or
// 'persist'), how many have run so far, and whether the one-shot fault has fired.
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

#endif
