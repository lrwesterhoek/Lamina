/* The environment variables the binaries read (README.md 'Environment variables of the binaries' and
   metal/README.md 'Knobs' list each one). Every read goes through these helpers, so one rule holds everywhere:
   - an EMPTY value is the same as an unset one (for every kind of value);
   - a flag is off when set to 0, false, no or off (in any case) and on when set to anything else (flag(n, true): a
     switch that is on by default, which only those values turn off);
   - a tristate (a switch with an automatic default) is -1 while unset or set to auto, else the flag;
   - a number is the value read by std::atol / std::atof; the caller checks its range and keeps its default otherwise;
   - a Name may carry an ALIAS, read only when the name itself is unset: a renamed knob kept for a release. The
     renamed ones are listed once, in 'renamed' below (docs/cleanup-plan.md P6.7: one DTFE_ name per concept in both
     binaries), and a Name built from the new name finds its old one there, so a read site names only the new one;
     main() names each old one that is set, once, at start-up (warnRenamed: never from a read, which may happen on a
     worker thread in the middle of a log). An INVERTED alias is the opposite switch (DTFE_NO_PROGRESS=1 is
     DTFE_PROGRESS=0): only flag() reads it. The GPU knobs' Metal spellings (PS_METAL_CHUNK for PS_GPU_CHUNK, ...)
     were dropped once the backend-neutral names had shipped. PS_GPU_CHUNK and PS_GPU_DEBUG_NOATOMIC keep their own
     names: standard DTFE's DTFE_GPU_CHUNK counts work items, not tetrahedra, and its DTFE_GPU_DEBUG_NOATOMIC is a
     level 1-3, not a switch.
   Plain C++17 with no other header of the project: the CUDA/HIP sources (nvcc, hipcc, the CPU emulation) include it
   too. */
#ifndef DTFE_ENV_KNOBS_H
#define DTFE_ENV_KNOBS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

namespace Env
{

// the renamed variables (P6.7), each read under its old name for one release when the new one is unset; both
// binaries read every row (one name per concept), so an old PS-only name now reaches standard DTFE's twin too
struct Renamed
{
    char const *name;    // the new name
    char const *old;     // the deprecated one
    bool inverted;       // the old one is the opposite switch
};
static constexpr Renamed renamed[] = {
    { "DTFE_OVERLAP",                       "DTFE_PS_OVERLAP",          false },
    { "DTFE_CHECKPOINT",                    "DTFE_PS_CHECKPOINT",       false },
    { "DTFE_GPU_TIMING",                    "PS_GPU_TIMING",            false },
    { "DTFE_GPU_FLOAT_MASS",                "PS_GPU_FLOAT_MASS",        false },
    { "DTFE_GPU_FLOAT_MOMENTS",             "PS_GPU_FLOAT_MOMENTS",     false },
    { "DTFE_MASS_AUDIT",                    "PS_MASS_AUDIT",            false },
    { "DTFE_DEBUG_DEGREES",                 "PS_DEBUG_DEGREES",         false },
    { "DTFE_GPU_EMU_WARP",                  "GPU_EMU_WARP",             false },
    { "DTFE_GPU_EXACT_CELLS",               "PS_EXACT_ITEM_CELLS",      false },
    { "DTFE_GPU_ITEMS",                     "PS_EXACT_ITEMS",           false },
    { "DTFE_TRIANGULATION_THREADS",         "DTFE_TBB_THREADS",         false },
    { "DTFE_AUTO_PTS_VEL_GRAD",             "DTFE_AUTO_PTS_VELGRAD",    false },
    { "DTFE_TESSELLATION_CACHE_THREADS",    "DTFE_TESS_CACHE_THREADS",  false },
    { "DTFE_TESSELLATION_CACHE_ZLIB",       "DTFE_TESS_CACHE_ZLIB",     false },
    { "DTFE_TESSELLATION_CACHE_CGAL_IO",    "DTFE_TESS_CACHE_CGAL_IO",  false },
    { "DTFE_TESSELLATION_CACHE_FINGERPRINT", "DTFE_TESS_CACHE_FINGERPRINT", false },
    { "DTFE_TESSELLATION_CACHE_LEGACY",     "DTFE_TESS_CACHE_LEGACY",   false },
    { "DTFE_PROGRESS",                      "DTFE_NO_PROGRESS",         true },
};

// a set, non-empty value of one variable, else nullptr
inline char const *own(char const *k)
{
    if ( k == nullptr ) return nullptr;
    char const *v = std::getenv( k );
    return v != nullptr and v[0] != '\0' ? v : nullptr;
}

// the row of 'renamed' whose new name is n, else nullptr
inline Renamed const *renamedRow(char const *n)
{
    if ( n != nullptr )
        for (Renamed const &r : renamed)
            if ( std::strcmp( r.name, n ) == 0 ) return &r;
    return nullptr;
}

// a variable's name and the alias read when the name is unset (nullptr: none; a renamed one's old name by default)
struct Name
{
    char const *name;
    char const *alias;
    bool inverted;       // the alias is the opposite switch (read by flag() only)
    Name(char const *n, char const *a = nullptr)
        : name( n ), alias( a != nullptr ? a : renamedRow( n ) != nullptr ? renamedRow( n )->old : nullptr ),
          inverted( a == nullptr and renamedRow( n ) != nullptr and renamedRow( n )->inverted ) {}
};

// the value: of the name, else of the alias (not an inverted one); nullptr when neither is set to a non-empty value
inline char const *value(Name const n)
{
    if ( char const *v = own( n.name ) ) return v;
    return n.inverted ? nullptr : own( n.alias );
}

inline bool isSet(Name const n) { return value( n ) != nullptr; }

// the value in lower case (ASCII), for the words below
inline std::string lower(char const *v)
{
    std::string s( v );
    for (char &c : s)
        if ( c >= 'A' and c <= 'Z' ) c = char( c - 'A' + 'a' );
    return s;
}

// true for the values that turn a flag off: 0, false, no, off
inline bool offValue(char const *v)
{
    std::string const s = lower( v );
    return s == "0" or s == "false" or s == "no" or s == "off";
}

// one stderr line per deprecated name that is set ('warning: PS_GPU_TIMING is deprecated, use DTFE_GPU_TIMING'),
// for main() to call once at start-up; an inverted one names the new switch's value that means the same
// (DTFE_NO_PROGRESS=1: 'use DTFE_PROGRESS=0'; DTFE_NO_PROGRESS=0: 'use DTFE_PROGRESS=1')
inline void warnRenamed()
{
    for (Renamed const &r : renamed)
        if ( char const *v = own( r.old ) )
            std::fprintf( stderr, "warning: %s is deprecated, use %s%s\n", r.old, r.name,
                          not r.inverted ? "" : offValue( v ) ? "=1" : "=0" );
}

inline bool flag(Name const n, bool const def = false)
{
    if ( n.inverted and own( n.name ) == nullptr )
        if ( char const *v = own( n.alias ) ) return offValue( v );   // DTFE_NO_PROGRESS=1: DTFE_PROGRESS off
    char const *v = value( n );
    return v == nullptr ? def : not offValue( v );
}

// -1 unset or auto (the switch's automatic choice), 0 an off value, 1 anything else
inline int tristate(Name const n)
{
    char const *v = value( n );
    if ( v == nullptr or lower( v ) == "auto" ) return -1;
    return offValue( v ) ? 0 : 1;
}

inline long integer(Name const n, long const def)
{
    char const *v = value( n );
    return v == nullptr ? def : std::atol( v );
}

inline double real(Name const n, double const def)
{
    char const *v = value( n );
    return v == nullptr ? def : std::atof( v );
}

inline std::string text(Name const n, std::string const &def = std::string())
{
    char const *v = value( n );
    return v == nullptr ? def : std::string( v );
}

}   // namespace Env

#endif
