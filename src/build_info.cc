/* Build provenance of the binaries (not the library): the Makefile recompiles this file on every link with the git
   revision and UTC link time, so each run log names its code. The fields read 'unknown' outside a git checkout. */

#ifndef DTFE_BUILD_REV
#define DTFE_BUILD_REV "unknown"
#endif
#ifndef DTFE_BUILD_TIME
#define DTFE_BUILD_TIME "unknown"
#endif

char const *dtfeBuildRevision() { return DTFE_BUILD_REV; }
char const *dtfeBuildTime()     { return DTFE_BUILD_TIME; }

// true in the programs, which link this file. partition_checkpoint.h checkpoints only there: the checkpoint folder
// is removed once main.cpp commits the outputs, which a libDTFE host never does. The header's weak fallback is false.
bool dtfeIsProgram() { return true; }
