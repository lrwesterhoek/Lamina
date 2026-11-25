/* Canonical form of an input path for the cache and globals-record identity keys: realpath(), so one file
   named two ways (relative, symlink, letter case) gives one key. An unresolvable path is kept as given. */

#ifndef DTFE_CANONICAL_PATH_HEADER
#define DTFE_CANONICAL_PATH_HEADER

#include <cstdlib>
#include <string>

inline std::string canonicalPath(std::string const &path)
{
    if ( path.empty() ) return path;
    char *r = ::realpath( path.c_str(), nullptr );
    if ( r == nullptr ) return path;
    std::string const out( r );
    std::free( r );
    return out;
}

#endif
