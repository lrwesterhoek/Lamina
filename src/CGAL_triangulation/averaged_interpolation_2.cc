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


/* Volume-averaged grid interpolation, method 2 (= MC sample points chosen inside the
   grid cell). Method 3 (equidistant sampling) is at the end of the file. */
#include "triangulation_common.h"
#include "sample_eval.h"



// Volume-averaged grid interpolation (method 2): averages the field over NN Monte Carlo points drawn
// inside each grid cell. The only method that can also report the per-cell velocity standard deviation.
void interpolateGrid_averaged_2(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    int const NN = userOptions.noPoints;        // MC samples per grid cell
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    SampleOutputs out( userOptions.aField, quantities, true );   // output field vectors, velocity_std included


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nComputing interpolation of the fields volume averaged over the sampling cell on a regular "
            << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region "
            << boxCoordinates.print()
            << ". The volume average is done using " << NN << " Monte Carlo samples in each grid cell.\n"
            << "\t Done: " << MESSAGE::Flush;


    // reserve output fields (zero-filled if the triangulation is undefined)
    size_t reserveSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    if ( not out.reserve( reserveSize, dt ) ) return;     // no interpolation possible without cells


    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];


    size_t seed = userOptions.randomSeed;
    if ( userOptions.partNo>=0 )
        seed += 1000*userOptions.partNo; // distinct seed per partition
    base_generator_type generator( seed );
    boost::uniform_real<> uni_dist(-0.5,0.5);
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);


    PaddingReport padding;
    Cell_handle previous;   // the cell centre's Delaunay cell: every sample's walk starts there
    forEachGridCell( boxCoordinates, dx, nGrid, message, [&](Real const *c, size_t i, size_t j, size_t k)
    {
#if NO_DIM==2
        previous = locateCell( dt, Point(c[0],c[1]), previous );
#elif NO_DIM==3
        previous = locateCell( dt, Point(c[0],c[1],c[2]), previous );
#endif
        // MC loop over random samples in this grid cell: one draw per statement, x, y, z, because the order of
        // the calls inside one argument list is unspecified (GCC on x86-64 may draw right to left)
        averageOverSamples( dt, out, padding, previous, NN, [&](int, Point &samplePoint)
        {
            double const px = uni()*dx[0] + c[0];
            double const py = uni()*dx[1] + c[1];
#if NO_DIM==2
            samplePoint = Point( px, py );
#elif NO_DIM==3
            double const pz = uni()*dx[2] + c[2];
            samplePoint = Point( px, py, pz );
#endif
        }, gridCellIndex( i,j,k, nGrid ) );
    } );
    padding.write( out.field, userOptions );  // report cells whose values may be wrong (incomplete tessellation)
}




// Volume-averaged redshift-cone interpolation (method 2): MC sampling of NN points
// inside each cone grid cell. Args as interpolateGrid_averaged_2, on a redshift-cone grid.
void interpolateRedshiftCone_averaged_2(DT &dt,
                                        User_options &userOptions,
                                        Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    int const NN = userOptions.noPoints;        // MC samples per grid cell
    Box boxCoordinates = userOptions.redshiftCone;    // redshift-cone region
    checkAngles( &(boxCoordinates[2]), (NO_DIM-1)*2 );// angles degrees -> radians
    Real *origin = &(userOptions.originPosition[0]);
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    SampleOutputs out( userOptions.aField, quantities );   // output field vectors (no velocity_std)


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields volume averaged over the sampling cell on a redshift cone grid with " << (NO_DIM==2 ? "r*psi": "r*theta*psi") << " = " <<
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


    size_t seed = userOptions.randomSeed;
    if ( userOptions.partNo>=0 )
        seed += 1000*userOptions.partNo; // distinct seed per partition
    base_generator_type generator( seed );
    boost::uniform_real<> uni_dist(-0.5,0.5);
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);


    PaddingReport padding;
    Cell_handle previous;   // the cell centre's Delaunay cell: every sample's walk starts there
    forEachGridCell( boxCoordinates, dx, nGrid, message, [&](Real const *c, size_t i, size_t j, size_t k)
    {
        previous = locateCell( dt, conePoint( origin, c ), previous );     // c = (r, theta, psi)
        // MC loop over random samples in this cone grid cell, drawn in (r, theta, psi), then in Cartesian coords
        averageOverSamples( dt, out, padding, previous, NN, [&](int, Point &samplePoint)
        {
            Real s[NO_DIM];
            s[0] = c[0] + uni()*dx[0];
            s[1] = c[1] + uni()*dx[1];
#if NO_DIM==3
            s[2] = c[2] + uni()*dx[2];
#endif
            samplePoint = conePoint( origin, s );
        }, gridCellIndex( i,j,k, nGrid ) );
    } );
    padding.write( out.field, userOptions );  // report cells whose values may be wrong (incomplete tessellation)
}




// Volume-averaged interpolation at user-defined sampling points (method 2): MC sampling
// of NN points inside each sampling cell. Args as interpolateGrid_averaged_2, with 'samples'.
void interpolateUserSampling_averaged_2(DT &dt,
                                        vector<Sample_point> &samples,
                                        User_options &userOptions,
                                        Quantities *quantities)
{
    int const NN = userOptions.noPoints;        // MC samples per grid cell

    SampleOutputs out( userOptions.aField, quantities );   // output field vectors (no velocity_std)


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "Computing interpolation of the fields volume averaged over the sampling cell at user given sampling points in the region " << userOptions.region.print() << ".\n"
            << "\t Done:   " << MESSAGE::Flush;


    // reserve output fields (zero-filled if the triangulation is undefined)
    if ( not out.reserve( samples.size(), dt ) ) return;     // no interpolation possible without cells


    size_t seed = userOptions.randomSeed;
    if ( userOptions.partNo>=0 )
        seed += 1000*userOptions.partNo; // distinct seed per partition
    base_generator_type generator( seed );
    boost::uniform_real<> uni_dist(-0.5,0.5);
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);


    PaddingReport padding;
    Cell_handle previous;   // the sample centre's Delaunay cell: every MC sample's walk starts there
    forEachSample( samples.size(), message, [&](size_t i)
    {
        // a sample outside the region of interest gets the zero record of a point outside the hull, so the
        // outputs keep one entry per sample (no locate and no draws: the next samples keep their hint and realization)
        if ( not userOptions.region.isPointInBox(samples[i].position()) )
        {
            out.pushZero();
            return;
        }
        previous = locateCell( dt, userSamplePoint( samples[i] ), previous );
        // MC loop over random samples in this sampling cell (a cuboid of size delta about the sample)
        averageOverSamples( dt, out, padding, previous, NN, [&](int, Point &samplePoint)
        {
            Real tempX = samples[i].position(0) + uni()*samples[i].delta(0);
            Real tempY = samples[i].position(1) + uni()*samples[i].delta(1);
#if NO_DIM==2
            samplePoint = Point(tempX,tempY);
#elif NO_DIM==3
            Real tempZ = samples[i].position(2) + uni()*samples[i].delta(2);
            samplePoint = Point(tempX,tempY,tempZ);
#endif
        }, i );
    } );
    padding.write( out.field, userOptions );  // report samples whose values may be wrong (incomplete tessellation)
}




// Volume-averaged grid interpolation (method 3): averages the field over NN equidistant (not random)
// points laid out on a regular sub-grid inside each grid cell.
void interpolateGrid_averaged_3(DT &dt,
                                User_options &userOptions,
                                Quantities *quantities)
{
    size_t *nGrid = &(userOptions.gridSize[0]); // grid size along each axis
    int const NN = userOptions.noPoints;        // equidistant samples per grid cell
    int const n = rootN( NN, NO_DIM );          // equidistant samples along each axis
    Box boxCoordinates = userOptions.region;    // box region of interest
    vector<Real> boxLength;                     // box size along each direction
    for (size_t i=0; i<NO_DIM; ++i)
        boxLength.push_back( boxCoordinates[2*i+1]-boxCoordinates[2*i] );

    SampleOutputs out( userOptions.aField, quantities );   // output field vectors (no velocity_std)


    MESSAGE::Message message( userOptions.verboseLevel );
    message << "\nComputing interpolation of the fields volume averaged over the sampling cell on a regular "
            << MESSAGE::printElements( nGrid, NO_DIM, "*" ) << " grid in the region "
            << boxCoordinates.print()
            << ". The volume average is done using " << NN << " equidistant sampling points in each grid cell.\n"
            << "\t Done: " << MESSAGE::Flush;


    // reserve output fields (zero-filled if the triangulation is undefined)
    size_t reserveSize = 1;
    for (size_t i=0; i<NO_DIM; ++i)
        reserveSize *= nGrid[i];
    if ( not out.reserve( reserveSize, dt ) ) return;     // no interpolation possible without cells


    Real dx[NO_DIM];        // grid spacing along each axis
    for (size_t i=0; i<NO_DIM; ++i)
        dx[i] = boxLength[i] / nGrid[i];


    // equidistant sample offsets relative to the grid-cell center
    std::vector<Real> samplePosition_buf(NN * NO_DIM);
    Real (*samplePosition)[NO_DIM] = reinterpret_cast<Real(*)[NO_DIM]>(samplePosition_buf.data());
    for (int i1=0; i1<n; ++i1)
        for (int i2=0; i2<n; ++i2)
#if NO_DIM==2
        {
            int index = i1*n + i2;
#endif
#if NO_DIM==3
            for (int i3=0; i3<n; ++i3)
            {
                int index = i1*n*n + i2*n + i3;
                samplePosition[index][2] = (i3+0.5)*dx[2]/n - dx[2]/2.;
#endif
                samplePosition[index][0] = (i1+0.5)*dx[0]/n - dx[0]/2.;
                samplePosition[index][1] = (i2+0.5)*dx[1]/n - dx[1]/2.;
            }


    PaddingReport padding;
    Cell_handle previous;   // the cell centre's Delaunay cell: every sample's walk starts there
    forEachGridCell( boxCoordinates, dx, nGrid, message, [&](Real const *c, size_t i, size_t j, size_t k)
    {
#if NO_DIM==2
        previous = locateCell( dt, Point(c[0],c[1]), previous );
#elif NO_DIM==3
        previous = locateCell( dt, Point(c[0],c[1],c[2]), previous );
#endif
        // loop over the equidistant sample points in this grid cell
        averageOverSamples( dt, out, padding, previous, NN, [&](int w, Point &samplePoint)
        {
#if NO_DIM==2
            samplePoint = Point( c[0] + samplePosition[w][0], c[1] + samplePosition[w][1] );
#elif NO_DIM==3
            samplePoint = Point( c[0] + samplePosition[w][0], c[1] + samplePosition[w][1], c[2] + samplePosition[w][2] );
#endif
        }, gridCellIndex( i,j,k, nGrid ) );
    } );
    padding.write( out.field, userOptions );  // report cells whose values may be wrong (incomplete tessellation)
}


