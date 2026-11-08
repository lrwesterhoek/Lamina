#ifndef SAMPLE_EVAL_HEADER
#define SAMPLE_EVAL_HEADER

/* The per-sample machinery of the sampling interpolations: the point values of unaveraged_interpolation.cc (regular
   grid, redshift cone, user points) and the cell averages of averaged_interpolation_2.cc (methods 2 and 3). Each of
   those functions is a traversal (which centres, in which order) and a sample source; locating a point, the zero
   record outside the hull, the cell's inverse matrix, the field block and the padding report are written here once.
   Everything is inline and keeps no state between calls: the functions run inside the OpenMP sub-domain threads.
   The position expressions keep their operand types (Real temporaries, double random draws) and are never split or
   merged: those set the rounding, and with it every output. Include after triangulation_common.h. */


// The helpers that run once per point or per Monte-Carlo sample are forced inline: left to itself the inliner keeps
// the bigger ones out of line, which made the shared loops measurably slower than the loops written out per function.
#if defined(__GNUC__)
#define SAMPLE_EVAL_INLINE inline __attribute__((always_inline))
#else
#define SAMPLE_EVAL_INLINE inline
#endif


// Locates the Delaunay cell that contains 'p', the walk starting at 'hint' (the 2D and 3D signatures differ).
SAMPLE_EVAL_INLINE Cell_handle locateCell(DT &dt,
                              Point const &p,
                              Cell_handle const &hint)
{
    Locate_type lt;
    int li;
#if NO_DIM==2
    return dt.locate( p, lt, li, hint );
#elif NO_DIM==3
    int lj;
    return dt.locate( p, lt, li, lj, hint );
#endif
}


// The output vectors of one interpolation call and the fields selected for them. 'velocity_std' is set only by the
// variant that reports it (the regular grid's method 2); the others never reserve or fill it.
struct SampleOutputs
{
    Field field;
    std::vector<Real>                             *density;
    std::vector< Pvector<Real,noVelComp> >        *velocity;
    std::vector< Pvector<Real,noGradComp> >       *velocity_gradient;
    std::vector<Real>                             *velocity_std;
    std::vector< Pvector<Real,noScalarComp> >     *scalar;
    std::vector< Pvector<Real,noScalarGradComp> > *scalar_gradient;

    SampleOutputs(Field const &selected,
                  Quantities *quantities,
                  bool const withVelocityStd = false)
        : field( selected ), density( &(quantities->density) ), velocity( &(quantities->velocity) ),
          velocity_gradient( &(quantities->velocity_gradient) ),
          velocity_std( withVelocityStd ? &(quantities->velocity_std) : nullptr ),
          scalar( &(quantities->scalar) ), scalar_gradient( &(quantities->scalar_gradient) )
    {}

    // Reserves the selected outputs for 'size' entries (zero-filled if the triangulation is undefined); false when
    // the triangulation has no cell, so there is nothing to interpolate.
    bool reserve(size_t const size,
                 DT &dt)
    {
        reserveMemory( density, size, field.density, dt, "density" );
        reserveMemory( velocity, size, field.velocity, dt, "velocity" );
        reserveMemory( velocity_gradient, size, field.velocity_gradient, dt, "velocity gradient" );
        if ( velocity_std ) reserveMemory( velocity_std, size, field.velocity_std, dt, "velocity standard deviation" );
        reserveMemory( scalar, size, field.scalar, dt, "scalar" );
        reserveMemory( scalar_gradient, size, field.scalar_gradient, dt, "scalar gradient" );
        return dt.number_of_vertices()>=NO_DIM+1;
    }

    // The record of a point outside the convex hull, or of a user point outside the region: zero in every selected
    // output, so the outputs keep one entry per point.
    SAMPLE_EVAL_INLINE void pushZero()
    {
        if (field.density) density->push_back( Real(0.) );
#ifdef VELOCITY
        if (field.velocity) velocity->push_back( Pvector<Real,noVelComp>::zero() );
        if (field.velocity_gradient) velocity_gradient->push_back( Pvector<Real,noGradComp>::zero() );
        if (velocity_std and field.velocity_std) velocity_std->push_back( Real(0.) );
#endif
#ifdef SCALAR
        if (field.scalar) scalar->push_back( Pvector<Real,noScalarComp>::zero() );
        if (field.scalar_gradient) scalar_gradient->push_back( Pvector<Real,noScalarGradComp>::zero() );
#endif
    }
};


// The grid cells (or user samples) whose value may be off because the tessellation there reaches the padding test's
// dummy points, reported in '<output>_density.badGridPoints' and '<output>_fields.badGridPoints' by 'write'.
struct PaddingReport
{
#ifdef TEST_PADDING
    vector<size_t> incompleteCells_d;   // a dummy neighbour: the density may be off
    vector<size_t> incompleteCells;     // a dummy vertex: the other fields may be off
#endif

    SAMPLE_EVAL_INLINE void flag(bool const dummyNeighbor,
              bool const dummyVertex,
              size_t const index)
    {
#ifdef TEST_PADDING
        if ( dummyNeighbor ) updateDummyGridCells( index, &incompleteCells_d );
        if ( dummyVertex ) updateDummyGridCells( index, &incompleteCells );
#endif
    }
    SAMPLE_EVAL_INLINE void flagCell(Cell_handle &cell,
                  size_t const index)
    {
#ifdef TEST_PADDING
        flag( hasDummyNeighbor( cell ), hasDummyVertex( cell ), index );
#endif
    }
    void write(Field const &field,
               User_options &userOptions)
    {
#ifdef TEST_PADDING
        if (field.density)
            showCellsContainingDummyPoints( &incompleteCells_d, userOptions, userOptions.outputFilename+"_density", "density" );
        if ( field.velocity or field.velocity_gradient or field.scalar or field.scalar_gradient )
            showCellsContainingDummyPoints( &incompleteCells, userOptions, userOptions.outputFilename+"_fields", "fields" );
#endif
    }
};


// The inverse position matrix of the last Delaunay cell a cell average used: consecutive samples in one grid cell
// often land in the same Delaunay cell, and the copy equals a recomputation bit for bit (a pure speed choice).
struct CellInverseCache
{
    Cell_handle cell;
    Real posMatrixInverse[NO_DIM][NO_DIM] = {};     // read only once 'valid'; zeroed so no compiler sees it unset
    Vertex_handle base;
    bool valid = false;

    SAMPLE_EVAL_INLINE void get(Cell_handle &current,
             Real inverse[][NO_DIM],
             Vertex_handle &currentBase)
    {
        if ( valid && current == cell )
        {
            for (int r=0; r<NO_DIM; ++r)
                for (int c=0; c<NO_DIM; ++c)
                    inverse[r][c] = posMatrixInverse[r][c];
            currentBase = base;
        }
        else
        {
            positionMatrix( current, inverse );
            currentBase = current->vertex(0);
            for (int r=0; r<NO_DIM; ++r)
                for (int c=0; c<NO_DIM; ++c)
                    posMatrixInverse[r][c] = inverse[r][c];
            base = currentBase;
            cell = current;
            valid = true;
        }
    }
};


// The point value of the selected fields at 'samplePoint' (absolute coordinates) in the finite cell 'current',
// appended to the outputs. The inverse matrix is computed fresh: consecutive points rarely share a cell.
SAMPLE_EVAL_INLINE void pushPointValues(SampleOutputs &out,
                            Cell_handle &current,
                            Point &samplePoint)
{
    Field const &field = out.field;
    Real posMatrixInverse[NO_DIM][NO_DIM];  // inverse vertex position-difference matrix
    positionMatrix( current, posMatrixInverse );
    Vertex_handle base = current->vertex(0);
    samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex

    if (field.density)
    {
        Real densGrad[NO_DIM];
        densityGrad(current, posMatrixInverse, densGrad);
        out.density->push_back( densityValue(densGrad,base,samplePoint) );
    }
#ifdef VELOCITY
    if (field.velocity or field.velocity_gradient)
    {
        Real velGrad[NO_DIM][noVelComp];
        velocityGrad( current, posMatrixInverse, velGrad );
        if (field.velocity) out.velocity->push_back( velocityValue(velGrad,base,samplePoint) );
        if (field.velocity_gradient) out.velocity_gradient->push_back( velocityGradient(velGrad) );
    }
#endif
#ifdef SCALAR
    if (field.scalar or field.scalar_gradient)
    {
#ifdef MY_SCALAR
        out.scalar->push_back( customScalar(current, posMatrixInverse, base, samplePoint) );
#else
        Real sGrad[NO_DIM][noScalarComp];
        scalarGrad( current, posMatrixInverse, sGrad );
        if (field.scalar) out.scalar->push_back( scalarValue(sGrad,base,samplePoint) );
        if (field.scalar_gradient) out.scalar_gradient->push_back( scalarGradient(sGrad) );
#endif
    }
#endif
}


// One point value: the cell that contains 'samplePoint' is searched from 'hint', which then holds the result (the
// point-value loops chain it through every point, infinite cells included); outside the hull the zero record.
// 'index' is the point's entry in the padding report.
SAMPLE_EVAL_INLINE void pushPointAt(DT &dt,
                        SampleOutputs &out,
                        PaddingReport &padding,
                        Cell_handle &hint,
                        Point &samplePoint,
                        size_t const index)
{
    hint = locateCell( dt, samplePoint, hint );
    if ( dt.is_infinite(hint) )     // infinite cell (outside the convex hull) -> zero field
    {
        out.pushZero();
        return;
    }
    padding.flagCell( hint, index );
    pushPointValues( out, hint, samplePoint );
}


// The sums of the selected fields over the samples of one cell average; 'push' stores sum/NN. A sample outside the
// hull adds nothing but still counts in NN (and as a zero velocity in the standard deviation).
struct CellSums
{
    bool const withVelocityStd;     // the outputs report velocity_std and it is selected
    Real tempD = Real(0.);
#ifdef VELOCITY
    Pvector<Real,noVelComp> tempV = Pvector<Real,noVelComp>::zero();
    Pvector<Real,noGradComp> tempVG = Pvector<Real,noGradComp>::zero();
    std::vector<Pvector<Real,noVelComp>> tempVelocities;   // per sample, for the standard deviation only
#endif
#ifdef SCALAR
    Pvector<Real,noScalarComp> tempS = Pvector<Real,noScalarComp>::zero();
    Pvector<Real,noScalarGradComp> tempSG = Pvector<Real,noScalarGradComp>::zero();
#endif
    bool dummyNeighbors = false, dummyVertices = false;

    SAMPLE_EVAL_INLINE CellSums(SampleOutputs const &out,
             int const NN)
        : withVelocityStd( out.velocity_std and out.field.velocity_std )
    {
#ifdef VELOCITY
        if ( withVelocityStd ) tempVelocities.resize( NN );
#endif
    }

    // adds sample 'w' at 'samplePoint' (relative to 'base') in the finite cell 'current'
    SAMPLE_EVAL_INLINE void add(Field const &field,
             Cell_handle &current,
             Real posMatrixInverse[][NO_DIM],
             Vertex_handle &base,
             Point &samplePoint,
             int const w)
    {
        if (field.density)
        {
            Real densGrad[NO_DIM];
            densityGrad(current, posMatrixInverse, densGrad);
            tempD += densityValue(densGrad,base,samplePoint);
        }
#ifdef VELOCITY
        if (field.velocity or field.velocity_gradient or withVelocityStd)
        {
            Real velGrad[NO_DIM][noVelComp];
            velocityGrad( current, posMatrixInverse, velGrad );
            if (field.velocity) tempV += velocityValue(velGrad,base,samplePoint);
            if (field.velocity_gradient) tempVG += velocityGradient(velGrad);
            if (withVelocityStd) tempVelocities[w] = velocityValue(velGrad,base,samplePoint);
        }
#endif
#ifdef SCALAR
        if (field.scalar or field.scalar_gradient)
        {
#ifdef MY_SCALAR
            tempS += customScalar(current, posMatrixInverse, base, samplePoint);
#else
            Real sGrad[NO_DIM][noScalarComp];
            scalarGrad( current, posMatrixInverse, sGrad );
            if (field.scalar) tempS += scalarValue(sGrad,base,samplePoint);
            if (field.scalar_gradient) tempSG += scalarGradient(sGrad);
#endif
        }
#endif
    }

    // averages and stores the field values
    SAMPLE_EVAL_INLINE void push(SampleOutputs &out,
              int const NN)
    {
        Field const &field = out.field;
        if (field.density) out.density->push_back( tempD/NN );
#ifdef VELOCITY
        if (field.velocity) out.velocity->push_back( tempV/NN );
        if (field.velocity_gradient) out.velocity_gradient->push_back( tempVG/NN );
        if (withVelocityStd) out.velocity_std->push_back( standardDeviation(tempVelocities.data(),NN) );
#endif
#ifdef SCALAR
        if (field.scalar) out.scalar->push_back( tempS/NN );
        if (field.scalar_gradient) out.scalar_gradient->push_back( tempSG/NN );
#endif
    }
};


// One cell average over NN samples, sample w placed by 'sampleAt(w, samplePoint)' (absolute coordinates). Every
// sample's walk starts at 'centreCell', the cell of the sampling cell's centre, which the loop never updates: a
// sample on a face or an edge resolves to the same cell as before. The padding flags of all samples, those outside
// the hull included, mark entry 'index' of the report.
template <class SampleAt>
SAMPLE_EVAL_INLINE void averageOverSamples(DT &dt,
                               SampleOutputs &out,
                               PaddingReport &padding,
                               Cell_handle const &centreCell,
                               int const NN,
                               SampleAt &&sampleAt,
                               size_t const index)
{
    CellSums sums( out, NN );
    CellInverseCache cache;
    Point samplePoint;
    for (int w=0; w<NN; ++w)
    {
        sampleAt( w, samplePoint );
        Cell_handle current = locateCell( dt, samplePoint, centreCell );
#ifdef TEST_PADDING
        if ( hasDummyNeighbor( current ) ) sums.dummyNeighbors = true;
        if ( hasDummyVertex( current ) ) sums.dummyVertices = true;
#endif
        if ( dt.is_infinite(current) )  // infinite cell (sample outside the convex hull) -> skip
            continue;

        Real posMatrixInverse[NO_DIM][NO_DIM];
        Vertex_handle base;
        cache.get( current, posMatrixInverse, base );
        samplePoint = relativeSamplePoint( base, samplePoint ); // relative to base vertex
        sums.add( out.field, current, posMatrixInverse, base, samplePoint, w );
    }
    sums.push( out, NN );
    padding.flag( sums.dummyNeighbors, sums.dummyVertices, index );
}


// Visits the cells of a regular NO_DIM grid over 'box' (spacing dx) in output order, the last axis innermost:
// visit(c, i, j, k) with the cell centre c[] (k = 0 in 2D). The centre is accumulated by repeated addition
// (recomputing it from the index would round differently), and each axis advances in its loop header, after the
// visit returns, so a cell that ends early (outside the hull) cannot freeze the rest of its row. A redshift cone
// uses it with c = (r, theta, psi). Prints the progress and the final '100%'.
template <class Visit>
SAMPLE_EVAL_INLINE void forEachGridCell(Box &box,
                            Real const *dx,
                            size_t const *nGrid,
                            MESSAGE::Message &message,
                            Visit &&visit)
{
    int prev = 0, amount100 = 0;
    Real c[NO_DIM];
    c[0] = box[0] + dx[0]/2.;
    for (size_t i=0; i<nGrid[0]; ++i, c[0] += dx[0])
    {
        amount100 = (100 * i)/ nGrid[0];
        if (prev < amount100)
            message.updateProgress( ++prev );

        c[1] = box[2] + dx[1]/2.;
        for (size_t j=0; j<nGrid[1]; ++j, c[1] += dx[1])
        {
#if NO_DIM==2
            visit( c, i, j, size_t(0) );
#elif NO_DIM==3
            c[2] = box[4] + dx[2]/2.;
            for (size_t k=0; k<nGrid[2]; ++k, c[2] += dx[2])
                visit( c, i, j, k );
#endif
        }
    }
    message << "100%\n" << MESSAGE::Flush;
}


// Visits the user samples 0..n-1 in order: visit(i). Prints the progress and the final '100%'.
template <class Visit>
SAMPLE_EVAL_INLINE void forEachSample(size_t const n,
                          MESSAGE::Message &message,
                          Visit &&visit)
{
    int prev = 0, amount100 = 0;
    for (size_t i=0; i<n; ++i)
    {
        amount100 = (100 * i)/ n;
        if (prev < amount100)
            message.updateProgress( ++prev );
        visit( i );
    }
    message << "100%\n" << MESSAGE::Flush;
}


// The Cartesian position of the redshift-cone point s = (r, theta[, psi]) about 'origin', through Real temporaries
// (they set the rounding of a single-precision build).
SAMPLE_EVAL_INLINE Point conePoint(Real const *origin,
                       Real const *s)
{
    Real const r = s[0], theta = s[1];
#if NO_DIM==2
    Real tempX = origin[0] + r*cos(theta);
    Real tempY = origin[1] + r*sin(theta);
    return Point(tempX,tempY);
#elif NO_DIM==3
    Real const psi = s[2];
    Real tempX = origin[0] + r*sin(theta)*cos(psi);
    Real tempY = origin[1] + r*sin(theta)*sin(psi);
    Real tempZ = origin[2] + r*cos(theta);
    return Point(tempX,tempY,tempZ);
#endif
}

// The position of a user sample.
SAMPLE_EVAL_INLINE Point userSamplePoint(Sample_point &sample)
{
#if NO_DIM==2
    return Point( sample.position(0), sample.position(1) );
#elif NO_DIM==3
    return Point( sample.position(0), sample.position(1), sample.position(2) );
#endif
}

#endif
