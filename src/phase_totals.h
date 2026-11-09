/* The run's PHASE TOTALS (docs/cleanup-plan.md P11.2): the seconds every timed computation stage took, summed over
   all threads and sub-domains of the process, by the stage's label -- the figures the device rule of P11.4 and the
   sub-domain pipeline of P11.6 are planned from. printComputationTime (CGAL_triangulation/triangulation_miscellaneous.h)
   adds every stage it times, whatever the verbosity (the sub-domain threads print nothing), and phaseTotalsReport
   prints one line per stage at the end of a run (DTFE.cpp dtfeFinishFields), for both binaries. The sums are wall
   seconds of the stages, so with concurrent sub-domains they exceed the run's wall time: a stage's share of the
   machine, not of the clock. Header-only; one registry per process (an inline function-local static). */
#ifndef DTFE_PHASE_TOTALS_H
#define DTFE_PHASE_TOTALS_H

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "message.h"

namespace PhaseTotals
{
struct Entry { double seconds = 0.; long count = 0; };
inline std::map<std::string, Entry> &registry() { static std::map<std::string, Entry> r; return r; }
inline std::mutex &mutex() { static std::mutex m; return m; }

// adds one timed stage (called by printComputationTime for every stage it reports)
inline void add(std::string const &label, double const seconds)
{
    std::lock_guard<std::mutex> lock( mutex() );
    Entry &e = registry()[label];
    e.seconds += seconds;
    ++e.count;
}

// the totals so far, in descending order of time
inline std::vector< std::pair<std::string, Entry> > snapshot()
{
    std::lock_guard<std::mutex> lock( mutex() );
    std::vector< std::pair<std::string, Entry> > v( registry().begin(), registry().end() );
    std::sort( v.begin(), v.end(), [](auto const &a, auto const &b) { return a.second.seconds > b.second.seconds; } );
    return v;
}

// one line per stage: 'phase totals: <label> <seconds> s over <count> call(s); ...' (nothing when no stage was timed)
inline void report(int const verboseLevel)
{
    auto const v = snapshot();
    if ( v.empty() ) return;
    MESSAGE::Message message( verboseLevel );
    message << "phase totals (wall seconds summed over all threads and sub-domains):\n";
    for (auto const &p : v)
        message << "   " << p.first << ": " << p.second.seconds << " s over " << p.second.count << " call(s)\n";
    message << MESSAGE::Flush;
    std::lock_guard<std::mutex> lock( mutex() );
    registry().clear();   // per DTFE() call ('--interlace' runs two and reports each)
}
}

#endif
