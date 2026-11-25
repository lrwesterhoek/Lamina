/* The environment variables the binaries read (the README's table lists each one). Every read goes through these
   helpers, so one rule holds: an EMPTY value is an unset one; a flag is off for 0, false, no, off (any case), else on;
   a tristate (DTFE_FACE_GRADIENT, DTFE_EXACT_AVERAGE, DTFE_GPU) is -1 while unset or auto, else the flag (the
   command-line flags beat it); a number is read by std::atol / std::atof and range-checked by the caller.
   Plain C++17 with no other project header. */
#ifndef DTFE_ENV_KNOBS_H
#define DTFE_ENV_KNOBS_H

#include <cstdlib>
#include <string>

namespace Env
{

// a set, non-empty value of one variable, else nullptr
inline char const *own(char const *k)
{
    if ( k == nullptr ) return nullptr;
    char const *v = std::getenv( k );
    return v != nullptr and v[0] != '\0' ? v : nullptr;
}

// a variable's name, made from a 'char const *': a read site passes "DTFE_GPU" or {"DTFE_GPU"}
struct Name
{
    char const *name;
    Name(char const *n) : name( n ) {}
};

// the value; nullptr when it is not set to a non-empty value
inline char const *value(Name const n) { return own( n.name ); }

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

inline bool flag(Name const n, bool const def = false)
{
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
