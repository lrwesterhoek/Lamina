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



/* General-purpose utilities: range/option consistency checks and array max. */

#ifndef MISCELLANEOUS_HEADER
#define MISCELLANEOUS_HEADER

#include <string>
#include <cmath>

#include "message.h"



// Error out unless minValue <= target <= maxValue.
template <typename T1>
void intervalCheck(T1 const target,
                   T1 const minValue, T1 const maxValue,
                   std::string errorMessage)
{
    if ( target<minValue or target>maxValue )
    {
        MESSAGE::Error error;
        error << "Some program variable failed a consitency check. The variable " << errorMessage << " has the value " << target << ", but it should be between " << minValue << " to " << maxValue << " ." << MESSAGE::EndError;
    }
}
// Error out unless target >= minValue.
template <typename T1>
void lowerBoundCheck(T1 const target,
                    T1 const minValue,
                    std::string errorMessage)
{
    if ( target<minValue )
    {
        MESSAGE::Error error;
        error << "Some program variable failed a consitency check. The variable " << errorMessage << " has the value " << target << ", but it should be larger or equal than " << minValue << " ." << MESSAGE::EndError;
    }
}


// The next 2 functions require the 'boost/program_options.hpp' library.
// Error out if 'opt1' and 'opt2' are both specified.
template <typename T>
void conflicting_options(const T &vm,
                        const char* opt1, const char* opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted() && vm.count(opt2) && !vm[opt2].defaulted())
    {
        MESSAGE::Error error;
        error << "Conflicting options '" << opt1 << "' and '" << opt2 << "'!\n" << MESSAGE::EndError;
    }
}

// Error out if 'for_what' is specified without 'required_option'.
template <typename T>
void option_dependency(const T &vm,
                        const char* for_what, const char* required_option)
{
    if (vm.count(for_what) && !vm[for_what].defaulted())
        if (vm.count(required_option) == 0 || vm[required_option].defaulted())
        {
            MESSAGE::Error error;
            error << "Option '" << for_what << "' requires option '" << required_option << "'!\n" << MESSAGE::EndError;
        }
}




// Maximum value of an array.
template <typename T, typename T_INT> inline T maximum(T *x, T_INT const size )
{
    T temp = x[0];
    for (T_INT i=1; i<size; ++i)
        if ( temp<x[i] )
            temp = x[i];
    return temp;
}

#endif

