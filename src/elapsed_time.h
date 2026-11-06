/* printElapsedTime, the step timer every interpolation prints with. Defined once, in subpartition.h (compiled into
   DTFE.cpp); the NGP, SPH and grid-deposit (CIC/TSC/PCS) files call it through this declaration. */
#ifndef ELAPSED_TIME_HEADER
#define ELAPSED_TIME_HEADER

#include <string>
#include <boost/timer.hpp>
#include "user_options.h"

// Prints the elapsed wall-clock time for a step and adds it to userOptions->totalTime.
void printElapsedTime(boost::timer *t, User_options *userOptions,
                      std::string computationQuantityName);

#endif
