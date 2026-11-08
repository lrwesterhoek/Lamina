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



/* Field interpolation evaluated at single sample points (point values, NOT cell volume averages):
   the linear DTFE estimate of the Delaunay cell containing each point. */

#include "triangulation_common.h"
#include "sample_eval.h"




// Interpolates fields at the center of each cell of a regular grid (point value, not cell-averaged).
// 'dt' carries vertex densities; 'quantities' receives the output grid fields.
void interpolateGrid(DT &dt,
                     User_options &userOptions,
                     Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    SampleOutputs out( userOptions.uField, quantities );   // output field vectors


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields at the sampling points given by a regular " <<
            MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region " <<
            boxCoordinates.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;


    // reserve output fields (zero-filled if the triangulation is undefined)
    size_t reserveSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    if ( not out.reserve( reserveSize, dt ) ) return;     // no interpolation possible without cells


    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];

    PaddingReport padding;
    Cell_handle current;    // the previous point's cell: the walk to the next point starts there
    forEachGridCell( boxCoordinates, dx, nGrid, message, [&](Real const *c, size_t i, size_t j, size_t k)
    {
#if NO_DIM==2
        Point samplePoint( c[0], c[1] );
#elif NO_DIM==3
        Point samplePoint( c[0], c[1], c[2] );
#endif
        pushPointAt( dt, out, padding, current, samplePoint, gridCellIndex( i,j,k, nGrid ) );
    } );
    padding.write( out.field, userOptions );  // report cells whose values may be wrong (incomplete tessellation)
}



// Interpolates fields to a redshift-cone grid (for galaxy surveys) at the sample points
// (point value, not cell-averaged). Args as interpolateGrid, on a redshift-cone grid.
void interpolateRedshiftCone(DT &dt,
                             User_options &userOptions,
                             Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    Box boxCoordinates = userOptions.redshiftCone;    // redshift-cone region
    checkAngles( &(boxCoordinates[2]), (NO_DIM-1)*2 );// angles degrees -> radians
    Real *origin = &(userOptions.originPosition[0]);
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    SampleOutputs out( userOptions.uField, quantities );   // output field vectors


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields at the sampling points given by a regular redshift cone grid with " << (NO_DIM==2 ? "r*psi": "r*theta*psi") << " = " <<
            MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " bins in the region " << (NO_DIM==2 ? "{[r_min,r_max],[psi_min,psi_max]}" : "{[r_min,r_max],[theta_min,theta_max],[psi_min,psi_max]}") << " = " << userOptions.redshiftCone.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;


    // reserve output fields (zero-filled if the triangulation is undefined)
    size_t reserveSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    if ( not out.reserve( reserveSize, dt ) ) return;     // no interpolation possible without cells


    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];

    PaddingReport padding;
    Cell_handle current;    // the previous point's cell: the walk to the next point starts there
    forEachGridCell( boxCoordinates, dx, nGrid, message, [&](Real const *c, size_t i, size_t j, size_t k)
    {
        Point samplePoint = conePoint( origin, c );    // c = (r, theta, psi)
        pushPointAt( dt, out, padding, current, samplePoint, gridCellIndex( i,j,k, nGrid ) );
    } );
    padding.write( out.field, userOptions );  // report cells whose values may be wrong (incomplete tessellation)
}




// Interpolates fields at user-defined sampling points (point value, not cell-averaged).
// Args as interpolateGrid, with 'samples' giving the sampling points.
void interpolateUserSampling(DT &dt,
                             vector<Sample_point> &samples,
                             User_options &userOptions,
                             Quantities *quantities)
{
    SampleOutputs out( userOptions.uField, quantities );   // output field vectors


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields at user given sampling points in the region " << userOptions.region.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;


    // reserve output fields (zero-filled if the triangulation is undefined)
    if ( not out.reserve( samples.size(), dt ) ) return;     // no interpolation possible without cells


    PaddingReport padding;
    Cell_handle current;    // the previous point's cell: the walk to the next point starts there
    forEachSample( samples.size(), message, [&](size_t i)
    {
        // a sample outside the region of interest gets the zero record of a point outside the hull, so the
        // outputs keep one entry per sample (no locate: the next sample's hint is the one it had before)
        if ( not userOptions.region.isPointInBox(samples[i].position()) )
        {
            out.pushZero();
            return;
        }
        Point samplePoint = userSamplePoint( samples[i] );
        pushPointAt( dt, out, padding, current, samplePoint, i );
    } );
    padding.write( out.field, userOptions );  // report samples whose values may be wrong (incomplete tessellation)
}
