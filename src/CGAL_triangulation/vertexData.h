/*
 *  Copyright (c) 2011       Marius Cautun
 *
 *                           Kapteyn Astronomical Institute
 *                           University of Groningen, the Netherlands
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef VERTEX_DATA_HEADER
#define VERTEX_DATA_HEADER

#include <cstring>   // memcpy: the star sum's low limb in the double scratch's bytes
#include "../define.h"
#include "../particle_data.h"


/* Per-vertex Delaunay data: the particle properties the vertex represents (see "particle_data.h") plus the dummy flags below. */
struct vertexData : public Data_structure
{
    protected:
    bool   dummy;         // true if vertex is a dummy test point (padding-efficiency test)
    bool   dummyNeighbor; // true if vertex has at least one dummy neighbor (padding test for density)
    double _volAcc;       // vertexDensity's per-vertex volume sum (triangulation.cpp); -1 marks a convex-hull vertex. +8 B/vertex, see auto_tune.h
#ifndef PHASE_SPACE
    uint64_t _volAccHi;   // the high limb of vertexDensity's exact star sum (volumeLimbs); +8 B/vertex, auto_tune.cc
#endif
#ifdef PHASE_SPACE
    Pvector<Real,NO_DIM> _eulerianPos; // Eulerian position (triangulation vertices store Lagrangian coords in PS-DTFE mode)
    uint64_t _particleID;              // snapshot ParticleID (stream identities, --ps-per-stream-ids); +8 B/vertex, see auto_tune.h
    int32_t  _psDeg;                   // --ps-vertex-mass: finite (non-dummy) cells incident to this vertex; +4 B/vertex
#endif


    public:
    vertexData(){ dummy=false; dummyNeighbor=false; _volAcc=0.;
#ifndef PHASE_SPACE
        _volAccHi=0;
#endif
#ifdef PHASE_SPACE
        _particleID=0;
        _psDeg=0;
#endif
    }

    // Returns the scalar field for this vertex; the single hook to customize what "scalar" means.
    inline Pvector<Real,noScalarComp> myScalar()
    {
        return scalar();
    }

    // Copies a particle's fields into this vertex (and its Eulerian position in PS-DTFE mode).
    inline void setData(Particle_data &other)
    {
        weight() = other.weight();
        density() = other.density();
#ifdef VELOCITY
        velocity() = other.velocity();
#endif
#ifdef SCALAR
        scalar() = other.scalar();
#endif
#ifdef PHASE_SPACE
        _eulerianPos = other.position();
        _particleID = other.particleID();
#endif
    }

#ifdef PHASE_SPACE
    inline Pvector<Real,NO_DIM>& eulerianPosition() { return _eulerianPos; }      // full Eulerian position
    inline Real& eulerianPosition(int const i) { return _eulerianPos[i]; }        // one Eulerian component
    inline uint64_t particleID() { return _particleID; }                          // snapshot ParticleID (0 if the reader has none)
    inline void setParticleID(uint64_t const id) { _particleID = id; }             // restoring a cached tessellation (tessellation_cache.h)
    inline int32_t& psDegree() { return _psDeg; }                                 // --ps-vertex-mass incident-cell count (see ps_interpolation.cc)
#endif
    // remaining accessors for 'Data_structure' are in "particle_data.h"

    inline double& volumeAccumulator() { return _volAcc; }                       // vertexDensity's scratch (triangulation.cpp)
#ifndef PHASE_SPACE
    // vertexDensity's exact star sum (triangulation.cpp vertexDensityStarSums), a 128-bit integer: its low 64 bits in
    // the 8 bytes of the double scratch above (copied, never read through the double), its high 64 bits in _volAccHi,
    // which also carries the sum's marks (markVolumeLimbs: a value no sum reaches). Once the sum is converted, the
    // double view volumeAccumulator() is the scratch again.
    inline __int128 volumeLimbs() const
    {
        uint64_t lo;
        std::memcpy( &lo, &_volAcc, sizeof lo );
        return __int128( ( (unsigned __int128)_volAccHi << 64 ) | lo );
    }
    inline void setVolumeLimbs(__int128 const x)
    {
        uint64_t const lo = uint64_t( (unsigned __int128)x );
        std::memcpy( &_volAcc, &lo, sizeof lo );
        _volAccHi = uint64_t( (unsigned __int128)x >> 64 );
    }
    inline int64_t volumeHighLimb() const { return int64_t( _volAccHi ); }
    // the mark 'high' in the high limb, a zero low limb (whose bytes read as the double 0.)
    inline void markVolumeLimbs(int64_t const high)
    {
        _volAcc = 0.;
        _volAccHi = uint64_t( high );
    }
#endif
    inline void setDummy() { dummy=true; dummyNeighbor=true; setDensity(0.); }    // mark as a dummy test point
    inline void setDummyNeighbor() { dummyNeighbor=true; }                        // mark as adjacent to a dummy
    inline bool isDummy() { return dummy; }
    inline bool hasDummyNeighbor() { return dummyNeighbor; }
};

#endif
