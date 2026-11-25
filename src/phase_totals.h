/* The run's PHASE TOTALS: seconds per timed stage label, summed over all threads and sub-domains (so with
   concurrent sub-domains they exceed the wall time). Fed by printComputationTime, printed by phaseTotalsReport.
   Header-only; one registry per process. */
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
