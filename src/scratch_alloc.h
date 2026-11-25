/* Out-of-core backing for the full-grid accumulators (--scratch-dir): scratch_alloc.cc replaces the global operator
 * new/delete; once armed, an allocation >= the threshold (1 GB, DTFE_SCRATCH_MIN_GB) comes from an mmap'ed, unlinked
 * file (reclaimed at exit, crashes included), so std::vector code and arithmetic are unchanged. Only the binaries
 * link the real hook (libDTFE gets scratch_alloc_stub.o, whose scratchArm() refuses). The directory must be LOCAL. */

#ifndef SCRATCH_ALLOC_HEADER
#define SCRATCH_ALLOC_HEADER

#include <cstddef>

namespace ScratchAlloc
{
    // Arm the scratch route: allocations >= thresholdBytes are served from mmap'ed files in
    // 'dir' from this call on. Returns false when the interposition is not linked into this
    // program (libDTFE stub) -- callers must surface that instead of silently proceeding.
    // 'dir' must already exist and be writable (validated by the caller).
    bool scratchArm(char const *dir, std::size_t thresholdBytes);

    // diagnostics for banners/tests: bytes currently mmap'ed, and mappings served in total
    long scratchLiveBytes();
    long scratchTotalAllocs();

    // ZERO FILLS: std::vector zero-fills what it allocates, and on a scratch mapping those zeros are WRITES (every page
    // goes dirty). Open a ZeroFillScope around a fill of ZEROS into a block nothing else writes (assign(n, 0), or
    // resize(n, 0) of an EMPTY vector): a scratch mapping made in this thread meanwhile is handed out as anonymous
    // memory and moved onto its never-written file at scope end. Thread-local; a no-op where scratch is not armed and
    // off macOS (Linux would keep every zero page resident).
    class ZeroFillScope
    {
    public:
        explicit ZeroFillScope(bool on = true);
        ~ZeroFillScope();
        ZeroFillScope(ZeroFillScope const &) = delete;
        ZeroFillScope &operator=(ZeroFillScope const &) = delete;
    private:
        bool on_;
    };
    long scratchZeroFillMoves();   // diagnostics: mappings moved onto their file at a scope's end

    template <class T> inline bool allZeroBytes(T const &x)
    {
        unsigned char const *b = reinterpret_cast<unsigned char const *>(&x);
        for (std::size_t i = 0; i < sizeof(T); ++i)
            if ( b[i] != 0 ) return false;
        return true;
    }
    // v.assign(n, zero) through a ZeroFillScope (assign drops the old contents: the new block is all zero)
    template <class V, class T> inline void zeroAssign(V &v, std::size_t n, T const &zero)
    {
        ZeroFillScope s( allZeroBytes( zero ) );
        v.assign( n, zero );
    }
    // v.resize(n, zero) through a ZeroFillScope -- only for an EMPTY v: growing a filled vector copies its data
    // into the new block, which must reach the file as written
    template <class V, class T> inline void zeroResize(V &v, std::size_t n, T const &zero)
    {
        ZeroFillScope s( v.empty() && allZeroBytes( zero ) );
        v.resize( n, zero );
    }
}

#endif
