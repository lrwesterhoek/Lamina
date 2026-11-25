/* DTFE_MASS_AUDIT=1: the PS-DTFE deposit's mass bookkeeping, a diagnostic (OFF by default: unset, none of it
   runs and every output is unchanged). Prints 'PS mass audit' lines to stderr per pass and per run, all sums in
   double in mass units; the closures it checks are in the audit lines. */

#ifndef PS_MASS_AUDIT_HEADER
#define PS_MASS_AUDIT_HEADER

#include <cstddef>
#include <cstdlib>
#include <mutex>

#include "../env_knobs.h"

namespace PSMassAudit
{
    // read once. 2 adds the per-particle closure on ONE domain only (partition < 0): pass --partition 1 1 1 or --gpu
    inline int level()
    {
        static int const v = int( Env::integer( "DTFE_MASS_AUDIT", 0 ) );
        return v;
    }
    inline bool on() { return level() >= 1; }

    // the partition the calling thread deposits (DTFE.cpp's batch loop sets it; -1 = one domain)
    inline thread_local int partition = -1;

    // run totals over the passes, [0] the unaveraged pass, [1] the averaged one; under 'mutex'
    struct Totals
    {
        double owned = 0., tets = 0., deg0 = 0., dropped = 0., gridPre = 0., gridPost = 0.;
        size_t passes = 0;
    };
    inline std::mutex mutex;
    inline Totals totals[2];
}

#endif
