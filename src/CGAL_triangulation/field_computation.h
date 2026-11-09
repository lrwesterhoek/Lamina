#ifndef FIELD_COMPUTATION_HEADER
#define FIELD_COMPUTATION_HEADER

/* Helpers for field values and gradients inside Delaunay cells (shared by the interpolation units).
   Fields vary linearly within a cell, so each gradient is constant and computed once per cell. */


// Inverse of the cell's vertex-difference matrix; left-multiplying vertex differences gives the gradient.
inline void positionMatrix(Cell_handle &cell,
                           Real posMatrixInverse[][NO_DIM])
{
    double Ax[NO_DIM][NO_DIM];
    Vertex_handle base = cell->vertex(0);

    for (int v = 0; v<NO_DIM; ++v)
        for (int i=0; i<NO_DIM; ++i)
            Ax[v][i] = double(cell->vertex(v+1)->point()[i] - base->point()[i]);
    matrixInverse( Ax, posMatrixInverse );
}

// Returns the sample point expressed relative to the cell's base vertex (vertex 0).
inline Point relativeSamplePoint(Vertex_handle const & base,
                                 Point &samplePoint)
{
    Real temp[NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        temp[i] = samplePoint[i] - base->point()[i];
#if NO_DIM==2
    return Point( temp[0], temp[1] );
#elif NO_DIM==3
    return Point( temp[0], temp[1], temp[2] );
#endif
}



// Density gradient inside a Delaunay cell.
template <typename Cell>
void densityGrad(Cell &cell,
                 Real posMatrixInverse[][NO_DIM],
                 Real *densGrad)
{
    Real dens[NO_DIM];  // vertex density differences relative to base
    for (int i=0; i<NO_DIM; ++i)
        dens[i] = cell->vertex(i+1)->info().density() - cell->vertex(0)->info().density();
    matrixMultiplication( posMatrixInverse, dens, densGrad );
}

// Density at a sample point (coordinates relative to the cell's base vertex).
inline Real densityValue(Real *densGrad,
                         Vertex_handle const & base,
                         Point &samplePoint)
{
    Real temp = 0.;
    for (int i=0; i<NO_DIM; ++i)
        temp += samplePoint[i]*densGrad[i];
    return base->info().density() + temp;
}



// Velocity gradient from the cell's vertex position and velocity differences.
template <typename Cell>
void velocityGrad(Cell &cell,
                  Real posMatrixInverse[][NO_DIM],
                  Real velGrad[][noVelComp])
{
    Vertex_handle base = cell->vertex(0);
    Real temp[NO_DIM][noVelComp]; // vertex velocity differences relative to base
    for (size_t v = 0; v<NO_DIM; ++v)
        for (size_t i=0; i<noVelComp; ++i)
            temp[v][i] = cell->vertex(v+1)->info().velocity(i) - base->info().velocity(i);
    matrixMultiplication<noVelComp>( posMatrixInverse, temp, velGrad );
}

// Velocity at a sample point (coordinates relative to the cell's base vertex).
inline Pvector<Real,noVelComp> velocityValue(Real velGrad[][noVelComp],
                                             Vertex_handle const & base,
                                             Point &samplePoint)
{
    Pvector<Real,noVelComp> temp;
    for (int i=0; i<NO_DIM; ++i)
    {
        temp[i] = 0.;
        for (int j=0; j<NO_DIM; ++j)
            temp[i] += velGrad[j][i] * samplePoint[j];
    }

    return temp + base->info().velocity();
}

// Flattens the velocity gradient matrix into the Pvector<Real,noGradComp> output layout.
inline Pvector<Real,noGradComp> velocityGradient(Real velGrad[][noVelComp])
{
    Pvector<Real,noGradComp> temp;
    for (size_t j=0; j<noVelComp; ++j)
        for (size_t i=0; i<NO_DIM; ++i)
            temp[j*NO_DIM+i] = velGrad[i][j];
    return temp;
}



// Scalar gradient from the cell's vertex position and scalar differences.
template <typename Cell>
void scalarGrad(Cell &cell,
                Real posMatrixInverse[][NO_DIM],
                Real sGrad[][noScalarComp])
{
    Real temp[NO_DIM][noScalarComp];    // vertex scalar differences relative to base
    Vertex_handle base = cell->vertex(0);
    for (size_t v = 0; v<NO_DIM; ++v)
        for (size_t i=0; i<noScalarComp; ++i)
            temp[v][i] = cell->vertex(v+1)->info().myScalar()[i] - base->info().myScalar()[i];
    matrixMultiplication<noScalarComp>( posMatrixInverse, temp, sGrad );
}

// Scalar value at a sample point (coordinates relative to the cell's base vertex).
inline Pvector<Real,noScalarComp> scalarValue(Real sGrad[][noScalarComp],
                                              Vertex_handle const & base,
                                              Point &samplePoint)
{
    Pvector<Real,noScalarComp> temp;
    for (size_t i=0; i<noScalarComp; ++i)
    {
        temp[i] = 0.;
        for (size_t j=0; j<NO_DIM; ++j)
            temp[i] += sGrad[j][i] * samplePoint[j];
    }

    return temp + base->info().myScalar();
}

// Flattens the scalar gradient matrix into the Pvector<Real,noScalarGradComp> output layout.
inline Pvector<Real,noScalarGradComp> scalarGradient(Real sGrad[][noScalarComp])
{
    Pvector<Real,noScalarGradComp> temp;
    for (size_t j=0; j<noScalarComp; ++j)
        for (size_t i=0; i<NO_DIM; ++i)
            temp[j*NO_DIM+i] = sGrad[i][j];
    return temp;
}



// The gradients in DOUBLE, for the exact-average deposits (averaged_interpolation_1.cc, plan P10.8): the vertex
// differences are exact in double for float data, the inverse is matrixInverseDouble's, and the products are rounded
// once where the caller narrows them, so the result does not follow the float rounding of CGAL's vertex labelling
// (which differs between the triangulations of two splits: at TNG100 z = 0 the float gradients of the same
// tetrahedra differed by 2.7e-5 of their rms between a 3^3 and a 6^3 plan, up to 7.6e-3 in flat ones).
template <typename Cell>
inline void densityGradDouble(Cell &cell,
                              double posMatrixInverse[][NO_DIM],
                              double *densGrad)
{
    double dens[NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        dens[i] = double( cell->vertex(i+1)->info().density() ) - double( cell->vertex(0)->info().density() );
    for (int i=0; i<NO_DIM; ++i)
    {
        double s = 0.;
        for (int j=0; j<NO_DIM; ++j)
            s += posMatrixInverse[i][j] * dens[j];
        densGrad[i] = s;
    }
}
template <typename Cell>
inline void velocityGradDouble(Cell &cell,
                               double posMatrixInverse[][NO_DIM],
                               double velGrad[][noVelComp])
{
    double temp[NO_DIM][noVelComp];
    for (size_t v = 0; v<NO_DIM; ++v)
        for (size_t i=0; i<noVelComp; ++i)
            temp[v][i] = double( cell->vertex(v+1)->info().velocity(i) ) - double( cell->vertex(0)->info().velocity(i) );
    for (int i=0; i<NO_DIM; ++i)
        for (size_t j=0; j<noVelComp; ++j)
        {
            double s = 0.;
            for (int k=0; k<NO_DIM; ++k)
                s += posMatrixInverse[i][k] * temp[k][j];
            velGrad[i][j] = s;
        }
}
template <typename Cell>
inline void scalarGradDouble(Cell &cell,
                             double posMatrixInverse[][NO_DIM],
                             double sGrad[][noScalarComp])
{
    double temp[NO_DIM][noScalarComp];
    for (size_t v = 0; v<NO_DIM; ++v)
        for (size_t i=0; i<noScalarComp; ++i)
            temp[v][i] = double( cell->vertex(v+1)->info().myScalar()[i] ) - double( cell->vertex(0)->info().myScalar()[i] );
    for (int i=0; i<NO_DIM; ++i)
        for (size_t j=0; j<noScalarComp; ++j)
        {
            double s = 0.;
            for (int k=0; k<NO_DIM; ++k)
                s += posMatrixInverse[i][k] * temp[k][j];
            sGrad[i][j] = s;
        }
}



// Evaluates the user-defined scalar (personalizedFunction) at a sample point, supplying it the
// interpolated density, velocity, and their gradients in the cell.
template <typename Cell>
Pvector<Real,noScalarComp> customScalar(Cell &current,
                                        Real posMatrixInverse[][NO_DIM],
                                        Vertex_handle &base,
                                        Point &samplePoint)
{
    Real densGrad[NO_DIM];
    densityGrad(current, posMatrixInverse, densGrad);
    Real density = densityValue(densGrad,base,samplePoint);

    Real velGrad[NO_DIM][noVelComp];
    velocityGrad( current, posMatrixInverse, velGrad );
    Pvector<Real,noVelComp> velocity = velocityValue(velGrad,base,samplePoint);

    Pvector<Real,noScalarComp> scalar;
    personalizedFunction( samplePoint, density, densGrad, velocity, velGrad, scalar );
    return scalar;
}

#endif
