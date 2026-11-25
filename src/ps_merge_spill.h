/* The SPILL of the staged ordered merge (PS-DTFE, DTFE.cpp): a partition waiting for its turn writes its raw
   accumulators to ONE unlinked file (mkstemp + unlink) and frees them; the merge reads them back into the same
   Quantities, so merged grids are byte-identical either way. A failed write keeps the sub-grids; a failed read stops the run. */
#ifndef PS_MERGE_SPILL_H
#define PS_MERGE_SPILL_H

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/vfs.h>   // statfs: a tmpfs default folder is RAM, where a spill saves nothing
#endif

#include "env_knobs.h"
#include "quantities.h"
#include "partition_checkpoint.h"   // forEachAccumulator: the one list of a Quantities' accumulators

namespace PSMergeSpill
{

// where the spill files go: '--scratch-dir' when set (validated at option parsing), else $TMPDIR, else /tmp. 'why' says
// why a folder cannot be used ("" = usable). The disk room is checked before every spill (room()); a Linux tmpfs
// default is refused (it IS memory).
inline std::string folder(std::string const &scratchDir, std::string &why)
{
    why.clear();
    std::string dir = scratchDir;
    bool const fromScratch = not dir.empty();
    if ( dir.empty() )
        dir = Env::text( "TMPDIR", "/tmp" );
    while ( dir.size() > 1 and dir.back() == '/' ) dir.pop_back();
    struct stat st;
    if ( ::stat( dir.c_str(), &st ) != 0 or not S_ISDIR( st.st_mode ) or ::access( dir.c_str(), W_OK ) != 0 )
        why = "'" + dir + "' is not a writable folder";
#ifdef __linux__
    struct statfs fs;
    if ( why.empty() and not fromScratch and ::statfs( dir.c_str(), &fs ) == 0 and fs.f_type == 0x01021994 )   // TMPFS_MAGIC
        why = "'" + dir + "' is a tmpfs (memory); set --scratch-dir to a disk folder";
#endif
    (void)fromScratch;
    return dir;
}

// the bytes the folder's volume can still take (< 0: unknown)
inline double diskFree(std::string const &dir)
{
    struct statvfs vfs;
    if ( ::statvfs( dir.c_str(), &vfs ) != 0 ) return -1.;
    return double( vfs.f_bavail ) * double( vfs.f_frsize );
}

// may 'bytes' be spilled there now? Same 2 GB margin as scratch_alloc.cc: the disk must never fill under the outputs
// still to come. 'reserved': spills being written right now plus, in '--scratch-dir', the sparse scratch mappings
// (statvfs shows their unwritten pages as free; taking them would SIGBUS the grids' write-back)
inline bool room(std::string const &dir, double const bytes, double const reserved = 0., double const margin = 2.e9)
{
    double const f = diskFree( dir );
    return f < 0. or f - margin - reserved >= bytes;
}

// the free space a spill must leave: 2 GB in '--scratch-dir', 16 GB in the system temporary folder (macOS grows its
// swap files on the boot disk under the same memory pressure). DTFE_SPILL_MIN_GB=X sets it for both.
inline double margin(std::string const &scratchDir)
{
    if ( Env::isSet( "DTFE_SPILL_MIN_GB" ) ) return std::max( 0., Env::real( "DTFE_SPILL_MIN_GB", 0. ) ) * 1.e9;
    return scratchDir.empty() ? 16.e9 : 2.e9;
}

// one spilled partition: the open descriptor of its unlinked file (move-only; closing it frees the disk space)
struct File
{
    int fd = -1;
    double bytes = 0.;
    File() = default;
    File(File const &) = delete;
    File &operator=(File const &) = delete;
    File(File &&o) noexcept : fd( o.fd ), bytes( o.bytes ) { o.fd = -1; o.bytes = 0.; }
    File &operator=(File &&o) noexcept
    {
        if ( this != &o ) { close(); fd = o.fd; bytes = o.bytes; o.fd = -1; o.bytes = 0.; }
        return *this;
    }
    ~File() { close(); }
    void close() { if ( fd >= 0 ) ::close( fd ); fd = -1; }
};

namespace detail
{
// the whole buffer, in pieces (one write() call takes at most INT_MAX bytes on macOS); false on any error
inline bool writeAll(int const fd, void const *p, size_t n)
{
    char const *c = static_cast<char const *>( p );
    while ( n > 0 )
    {
        ssize_t const w = ::write( fd, c, std::min( n, size_t( 1 ) << 30 ) );
        if ( w < 0 and errno == EINTR ) continue;
        if ( w <= 0 ) return false;
        c += w; n -= size_t( w );
    }
    return true;
}
inline bool readAll(int const fd, void *p, size_t n)
{
    char *c = static_cast<char *>( p );
    while ( n > 0 )
    {
        ssize_t const r = ::read( fd, c, std::min( n, size_t( 1 ) << 30 ) );
        if ( r < 0 and errno == EINTR ) continue;
        if ( r <= 0 ) return false;
        c += r; n -= size_t( r );
    }
    return true;
}
inline bool writeOne(int const fd, Quantities &q, double &bytes)
{
    bool ok = writeAll( fd, q.ps_subOrigin, sizeof( q.ps_subOrigin ) ) and writeAll( fd, q.ps_subDims, sizeof( q.ps_subDims ) );
    PartitionCheckpoint::forEachAccumulator( q, [&](char const *, auto &v)
    {
        uint64_t const n = v.size();
        ok = ok and writeAll( fd, &n, sizeof( n ) ) and ( n == 0 or writeAll( fd, v.data(), n * sizeof( v[0] ) ) );
        bytes += double( n ) * double( sizeof( v[0] ) );
    } );
    return ok;
}
inline bool readOne(int const fd, Quantities &q)
{
    bool ok = readAll( fd, q.ps_subOrigin, sizeof( q.ps_subOrigin ) ) and readAll( fd, q.ps_subDims, sizeof( q.ps_subDims ) );
    PartitionCheckpoint::forEachAccumulator( q, [&](char const *, auto &v)
    {
        uint64_t n = 0;
        ok = ok and readAll( fd, &n, sizeof( n ) );
        if ( not ok ) return;
        v.resize( size_t( n ) );
        ok = n == 0 or readAll( fd, v.data(), size_t( n ) * sizeof( v[0] ) );
    } );
    return ok;
}
}   // namespace detail

// give every accumulator's memory back (clear() keeps the capacity; the swap does not)
inline void release(Quantities &q)
{
    PartitionCheckpoint::forEachAccumulator( q, [&](char const *, auto &v)
    {
        using V = typename std::decay<decltype( v )>::type;
        V().swap( v );
    } );
}

// write u and a to a new unlinked file in 'dir'; the Quantities are left untouched (the caller frees them once it
// has decided to keep the file). false + 'why' on any failure, and no file is left behind.
inline bool write(Quantities &u, Quantities &a, std::string const &dir, File &out, std::string &why)
{
    std::string path = dir + "/dtfe_merge_spill_" + std::to_string( long( ::getpid() ) ) + "_XXXXXX";
    std::vector<char> name( path.begin(), path.end() );
    name.push_back( '\0' );
    int const fd = ::mkstemp( name.data() );
    if ( fd < 0 ) { why = std::string( "cannot create a file in '" ) + dir + "': " + std::strerror( errno ); return false; }
    ::unlink( name.data() );   // the descriptor keeps it alive; any exit reclaims it
    File f;
    f.fd = fd;
    double bytes = 0.;
    if ( not detail::writeOne( fd, u, bytes ) or not detail::writeOne( fd, a, bytes ) )
    {
        why = std::string( "writing to '" ) + dir + "' failed: " + std::strerror( errno );
        return false;   // f closes: the space comes back
    }
    f.bytes = bytes;
    out = std::move( f );
    return true;
}

// read a spilled partition back into u and a (whose vectors are empty) and close the file; false + 'why' on failure
inline bool read(File &f, Quantities &u, Quantities &a, std::string &why)
{
    bool const ok = ::lseek( f.fd, 0, SEEK_SET ) == 0 and detail::readOne( f.fd, u ) and detail::readOne( f.fd, a );
    if ( not ok ) why = std::string( "reading a spilled partition back failed: " ) + std::strerror( errno );
    f.close();
    return ok;
}

}   // namespace PSMergeSpill

#endif
