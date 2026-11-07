/* The user-sampling path of standard DTFE (interpolateUserSampling and interpolateUserSampling_averaged_2), which
   only the library API reaches: no input reader fills the sample list, so the binaries never take it. This driver
   is the DTFE program with a sample list: it reads the particles as the binary does, the samples from a text file
   (one 'x y z dx dy dz' line per sample: its position and the size of its sampling cell, in the box's units after
   '--MpcUnit'), calls DTFE() and writes the outputs as the binary does. tests/dtfe_sampling_check.sh builds it
   against libDTFE ('make library') with demo/examples/Makefile's flags and checks its outputs.

   Usage: dtfe_user_sampling <samples file> <the DTFE command line: input, output, options> */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/DTFE.h"
#include "../src/io/io.h"

int main(int argc, char *argv[])
{
    if ( argc < 4 )
    {
        std::cerr << "usage: " << argv[0] << " <samples file> <input> <output> [DTFE options ...]\n";
        return 2;
    }
    std::ifstream in( argv[1] );
    if ( not in )
    {
        std::cerr << "cannot read the samples file '" << argv[1] << "'\n";
        return 2;
    }
    std::vector<Sample_point> samples;
    Real pos[NO_DIM], delta[NO_DIM];
    while ( true )
    {
        for (int j=0; j<NO_DIM; ++j) in >> pos[j];
        for (int j=0; j<NO_DIM; ++j) in >> delta[j];
        if ( not in ) break;
        Sample_point s;
        s.setPosition( pos );
        s.setDelta( delta );
        samples.push_back( s );
    }
    if ( samples.empty() )
    {
        std::cerr << "no samples in '" << argv[1] << "'\n";
        return 2;
    }

    argv[1] = argv[0];   // the DTFE command line, as the binary sees it
    User_options userOptions;
    userOptions.readOptions( argc - 1, argv + 1 );
    std::vector<Particle_data> particles;
    std::vector<Sample_point> fromInput;   // (no reader fills it)
    readInputData( &particles, &fromInput, &userOptions );

    Quantities uQuantities, aQuantities;
    DTFE( &particles, samples, userOptions, &uQuantities, &aQuantities );
    writeOutputData( uQuantities, aQuantities, userOptions );
    return 0;
}
