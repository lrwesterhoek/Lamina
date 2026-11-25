/* Per-tetrahedron caustic stratification for '--ps-caustics': J = dx/dq is constant in a tetrahedron (J^T = Lag^-1 * Ax);
 * the number k of NEGATIVE real eigenvalues of J is the collapse multiplicity (sign det J = (-1)^k). The A3/A4 bits are
 * fitted across neighbouring tetrahedra, so they are indicators at the stencil's resolution. Per-cell bits are OR-ed. */

#ifndef PS_CAUSTIC_CLASS_HEADER
#define PS_CAUSTIC_CLASS_HEADER

#include <algorithm>
#include <cmath>

#include "../define.h"
#include "../math_functions.h"


namespace PSCausticClass
{

/* Wider than a byte on purpose: bit 8 does not fit in an unsigned char, and a narrowing OR into a byte truncates it with
   no warning, so every accumulator must use this typedef. The '.causticClass' export is float32, exact to 2^24. */
typedef unsigned short CausticMask;

/* Bit layout of the per-cell '.causticClass' mask. Bits 0..1 repeat the parity bits of '.caustic'; bits 2..5 record which
   collapse multiplicities occur among the overlapping tetrahedra; bit 6 is the umbilic indicator; bits 7..8 are the
   Arnold-stratum indicators and need the '--ps-caustic-cusps' stencil. */
enum
{
    BIT_PARITY_POS  = 1 << 0,  // a det(J) > 0 tetrahedron overlaps the cell
    BIT_PARITY_NEG  = 1 << 1,  // a det(J) < 0 tetrahedron overlaps  (both set = fold crossing)
    BIT_COLLAPSE_0  = 1 << 2,  // k = 0 present (no axis collapsed)
    BIT_COLLAPSE_1  = 1 << 3,  // k = 1 present (one axis: wall/pancake)
    BIT_COLLAPSE_2  = 1 << 4,  // k = 2 present (two axes: filament)
    BIT_COLLAPSE_3  = 1 << 5,  // k = 3 present (three axes: node)
    BIT_DEGENERATE  = 1 << 6,  // two eigenvalues coincide (umbilic/D4 INDICATOR, not a D4 label)
    BIT_CUSP        = 1 << 7,  // A3 cusp INDICATOR: D1 = 0, the fold is tangent to its null direction
    BIT_SWALLOWTAIL = 1 << 8,  // A4 INDICATOR: D1 = D2 = 0 as well (see the header note on codim 3)

    /* Everything the GPU deposit can produce: the host builds its per-tet mask without the cusp
       stencil, so bits 7..8 never come back and the merge masks them off. */
    BITS_GPU_MAX    = BIT_PARITY_POS | BIT_PARITY_NEG | BIT_COLLAPSE_0 | BIT_COLLAPSE_1
                    | BIT_COLLAPSE_2 | BIT_COLLAPSE_3 | BIT_DEGENERATE
};

/* D4 (umbilic) thresholds: two eigenvalues of J vanish TOGETHER, so the pair must coincide AND sit at zero (coincidence
 * alone fires on any locally isotropic patch). Both tolerances are relative to the tetrahedron's eigenvalue SPREAD, not to
 * the pair's magnitudes: |l_i - l_j| <= tol*(|l_i| + |l_j|) becomes unsatisfiable exactly where D4 lives. */
static const double DEGENERACY_REL_TOL = 1.e-3;
static const double UMBILIC_ZERO_BAND  = 0.25;

/* A3 (cusp) thresholds. A fold is where the critical eigenvalue lambda_c of J passes through zero; it is an A3 cusp
 * where grad(lambda_c) . v_c = 0. Only near the fold (|lambda_c| small against the eigenvalue spread: FOLD_BAND), with
 * the tangency made scale-free by comparing |grad . v_c| against |grad| (CUSP_REL_TOL). grad is a least-squares estimate
 * from the neighbouring tetrahedra, so this is an INDICATOR at the tessellation's resolution. */
static const double FOLD_BAND    = 0.25;
static const double CUSP_REL_TOL = 0.15;

/* A4 (swallowtail) thresholds on the derivatives of foldDerivatives(). The test is that the Taylor series of lambda_c along
 * v_c STARTS at cubic order: with stencil radius h, c1 = |D1| h, c2 = |D2| h^2/2, c3 = |D3| h^3/6 (A2 is dominated by c1,
 * A3 by c2, A4 by c3); TERM_TOL is how far below c3 the two lower terms must sit. SIGNAL_MIN keeps the ratios from being
 * noise where lambda_c is flat and rejects A5+. Both calibrated on the three 1-D analytic waves (nearest-64 stencil). The
 * flagged band is a fixed fraction of the stencil radius (delta ~ TERM_TOL*h/3), hence proximity, not identification. */
static const double SWALLOWTAIL_TERM_TOL   = 0.50;
static const double SWALLOWTAIL_SIGNAL_MIN = 0.25;


/* Real eigenvalues of a 3x3 matrix in closed form (lambda^3 - p lambda^2 + q lambda - r = 0). Returns the number of real
   roots found (1 or 3) and fills 'root'. */
inline int realEigenvalues3(double const M[3][3], double root[3])
{
    double const p = M[0][0] + M[1][1] + M[2][2];
    double const q = (M[0][0]*M[1][1] - M[0][1]*M[1][0])
                   + (M[0][0]*M[2][2] - M[0][2]*M[2][0])
                   + (M[1][1]*M[2][2] - M[1][2]*M[2][1]);
    double const r = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
                   - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
                   + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);

    // depress to t^3 + a*t + b with lambda = t + p/3
    double const shift = p / 3.;
    double const a = q - p*p/3.;
    double const b = -2.*p*p*p/27. + p*q/3. - r;

    double const disc = -4.*a*a*a - 27.*b*b;      // > 0: three distinct real roots
    double const eps  = 1.e-300;
    if ( std::fabs(a) < eps and std::fabs(b) < eps )
    {                                              // triple root
        root[0] = root[1] = root[2] = shift;
        return 3;
    }
    /* A DOUBLE root sits at disc == 0, where rounding pushes the computed value slightly negative and would divert a real
       triple into the single-root branch, losing the coincident pair the umbilic indicator looks for. Anything within
       rounding of zero is treated as three real roots; the trigonometric branch clamps acos()'s argument. */
    double const discScale = std::max( std::fabs(4.*a*a*a), std::fabs(27.*b*b) );
    if ( disc >= -1.e-9 * discScale )
    {                                              // trigonometric (Viete) form, all roots real
        double const m = 2. * std::sqrt( -a/3. );
        double arg = 3.*b / (a*m);
        if ( arg >  1. ) arg =  1.;                // guard the rounding at a double root
        if ( arg < -1. ) arg = -1.;
        double const theta = std::acos( arg ) / 3.;
        for (int i=0; i<3; ++i)
            root[i] = m * std::cos( theta - 2.*M_PI*double(i)/3. ) + shift;
        return 3;
    }
    // one real root (Cardano)
    double const s = std::sqrt( b*b/4. + a*a*a/27. );
    double const u = std::cbrt( -b/2. + s );
    double const v = std::cbrt( -b/2. - s );
    root[0] = u + v + shift;
    return 1;
}


/* The deformation of one tetrahedron, exposed so the A3 test can reuse the critical eigenvalue and its null direction
   across neighbours. 'valid' is false when the Lagrangian simplex is too flat to define J. */
struct TetDeformation
{
    bool   valid    = false;
    double critical = 0.;            // eigenvalue of J closest to zero (the one that folds)
    double spread   = 0.;            // largest |eigenvalue|, the scale |critical| is judged against
    double vcrit[NO_DIM] = {0.};     // unit right eigenvector of 'critical' (the null direction)
    CausticMask bits = 0;            // parity-free bits: collapse multiplicity + degeneracy
};


// Unit null vector of (M - lambda*I) for a 3x3 M: the row pair whose cross product is longest.
#if NO_DIM==3
inline bool nullVector3(double const M[3][3], double lambda, double v[3])
{
    double A[3][3];
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) A[i][j] = M[i][j] - (i==j ? lambda : 0.);

    double best = 0.;
    for (int a=0; a<3; ++a)
    {
        int const b = (a+1) % 3;
        double const c[3] = { A[a][1]*A[b][2] - A[a][2]*A[b][1],
                              A[a][2]*A[b][0] - A[a][0]*A[b][2],
                              A[a][0]*A[b][1] - A[a][1]*A[b][0] };
        double const n = std::sqrt( c[0]*c[0] + c[1]*c[1] + c[2]*c[2] );
        if ( n > best ) { best = n; v[0]=c[0]/n; v[1]=c[1]/n; v[2]=c[2]/n; }
    }
    return best > 0.;
}
#endif


/* Full deformation analysis of one tetrahedron from its Eulerian (Ax) and Lagrangian (Lag) edge matrices; classifyTet()
   is the thin wrapper for the bits only. Both share two primitives, so the tensor and the cubic are solved once per call. */
inline TetDeformation analyzeTet(double const Ax[NO_DIM][NO_DIM],
                                 double const Lag[NO_DIM][NO_DIM]);
inline bool deformationTensor(double const Ax[NO_DIM][NO_DIM],
                              double const Lag[NO_DIM][NO_DIM],
                              double M[NO_DIM][NO_DIM]);
inline CausticMask bitsFromSpectrum(double const ev[], int const nReal);


/* Classification bits of one tetrahedron from its edge matrices; 0 when the Lagrangian simplex is degenerate (the caller
   keeps its parity bit). The deposit's hot path: it stops at the spectrum and never forms the null vector. */
inline CausticMask classifyTet(double const Ax[NO_DIM][NO_DIM],
                               double const Lag[NO_DIM][NO_DIM])
{
    double M[NO_DIM][NO_DIM];
    if ( not deformationTensor( Ax, Lag, M ) ) return 0;

#if NO_DIM==3
    double ev[3];
    int const nReal = realEigenvalues3( M, ev );
    return bitsFromSpectrum( ev, nReal );
#elif NO_DIM==2
    // 2x2: lambda^2 - tr*lambda + det = 0; a negative discriminant is a complex pair (nReal = 0)
    double const tr  = M[0][0] + M[1][1];
    double const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    double const d2  = tr*tr - 4.*det;
    double ev[2] = {0., 0.};
    int nReal = 0;
    if ( d2 >= 0. )
    {
        double const s = std::sqrt(d2);
        ev[0] = 0.5*(tr + s);
        ev[1] = 0.5*(tr - s);
        nReal = 2;
    }
    return bitsFromSpectrum( ev, nReal );
#endif
}


/* The Lagrangian->Eulerian deformation tensor of one tetrahedron, shared by classifyTet and analyzeTet. */
inline bool deformationTensor(double const Ax[NO_DIM][NO_DIM],
                              double const Lag[NO_DIM][NO_DIM],
                              double M[NO_DIM][NO_DIM])
{
    /* The inverse is formed HERE in double: math_functions.h's matrixInverse() writes a 'Real' and zeroes it below an
       ABSOLUTE |det| < 1e-6, which small Lagrangian simplices can fall below (a zero inverse would read as k = 0).
       The guard below is relative to the matrix scale instead. */
    double scale = 0.;
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j) scale = std::max( scale, std::fabs(Lag[i][j]) );
    if ( scale <= 0. ) return false;

    double lagCopy[NO_DIM][NO_DIM];
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j) lagCopy[i][j] = Lag[i][j];
    double const detLag = determinant( lagCopy );
    // a simplex flatter than this carries no usable deformation tensor
    double detScale = 1.;
    for (int i=0; i<NO_DIM; ++i) detScale *= scale;
    if ( not std::isfinite(detLag) or std::fabs(detLag) < 1.e-12 * detScale ) return false;

    double lagInv[NO_DIM][NO_DIM];
#if NO_DIM==3
    lagInv[0][0] =  (Lag[1][1]*Lag[2][2] - Lag[1][2]*Lag[2][1]) / detLag;
    lagInv[0][1] = -(Lag[0][1]*Lag[2][2] - Lag[0][2]*Lag[2][1]) / detLag;
    lagInv[0][2] =  (Lag[0][1]*Lag[1][2] - Lag[0][2]*Lag[1][1]) / detLag;
    lagInv[1][0] = -(Lag[1][0]*Lag[2][2] - Lag[1][2]*Lag[2][0]) / detLag;
    lagInv[1][1] =  (Lag[0][0]*Lag[2][2] - Lag[0][2]*Lag[2][0]) / detLag;
    lagInv[1][2] = -(Lag[0][0]*Lag[1][2] - Lag[0][2]*Lag[1][0]) / detLag;
    lagInv[2][0] =  (Lag[1][0]*Lag[2][1] - Lag[1][1]*Lag[2][0]) / detLag;
    lagInv[2][1] = -(Lag[0][0]*Lag[2][1] - Lag[0][1]*Lag[2][0]) / detLag;
    lagInv[2][2] =  (Lag[0][0]*Lag[1][1] - Lag[0][1]*Lag[1][0]) / detLag;
#elif NO_DIM==2
    lagInv[0][0] =  Lag[1][1] / detLag;
    lagInv[0][1] = -Lag[0][1] / detLag;
    lagInv[1][0] = -Lag[1][0] / detLag;
    lagInv[1][1] =  Lag[0][0] / detLag;
#endif

    // M = Lag^-1 * Ax  (= J^T; same eigenvalues as J)
    for (int i=0; i<NO_DIM; ++i)
        for (int j=0; j<NO_DIM; ++j)
        {
            double s = 0.;
            for (int k=0; k<NO_DIM; ++k) s += lagInv[i][k] * Ax[k][j];
            M[i][j] = s;
        }
    return true;
}


/* The classification bits that follow from an already-computed spectrum (analyzeTet reuses its eigenvalues). */
inline CausticMask bitsFromSpectrum(double const ev[], int const nReal)
{
    CausticMask bits = 0;
    int nNeg = 0;
    bool degenerate = false;

#if NO_DIM==3
    for (int i=0; i<nReal; ++i)
        if ( ev[i] < 0. ) ++nNeg;
    if ( nReal == 3 )
    {
        double evSpread = 0.;
        for (int i=0; i<3; ++i) evSpread = std::max( evSpread, std::fabs(ev[i]) );
        for (int i=0; i<3 and not degenerate; ++i)
            for (int j=i+1; j<3 and not degenerate; ++j)
            {
                if ( evSpread <= 0. ) continue;
                if ( std::fabs(ev[i]-ev[j]) > DEGENERACY_REL_TOL*evSpread ) continue;
                // ... and the coincident pair must be AT ZERO, or this is just an isotropic patch
                double const pair = std::max( std::fabs(ev[i]), std::fabs(ev[j]) );
                if ( pair <= UMBILIC_ZERO_BAND*evSpread ) degenerate = true;
            }
    }
#elif NO_DIM==2
    /* nReal is 0 for a complex pair (no collapsed axis to report) and 2 otherwise. No umbilic bit in 2D, deliberately:
       the pair would be the whole spectrum, so "both at zero relative to the spread" has no honest scale, and a corank-2
       degeneracy is codimension 3, not generic in a 2-parameter Lagrangian family. */
    for (int i=0; i<nReal; ++i)
        if ( ev[i] < 0. ) ++nNeg;
#endif

    switch ( nNeg )
    {
        case 0:  bits |= BIT_COLLAPSE_0; break;
        case 1:  bits |= BIT_COLLAPSE_1; break;
        case 2:  bits |= BIT_COLLAPSE_2; break;
        default: bits |= BIT_COLLAPSE_3; break;
    }
    if ( degenerate ) bits |= BIT_DEGENERATE;
    return bits;
}


/* Deformation tensor M = Lag^-1 * Ax (= J^T, same eigenvalues), its bits, and the eigenvalue nearest zero with its null
   direction. Shares deformationTensor() + bitsFromSpectrum() with classifyTet, so the cubic is solved once per call. */
inline TetDeformation analyzeTet(double const Ax[NO_DIM][NO_DIM],
                                 double const Lag[NO_DIM][NO_DIM])
{
    TetDeformation td;
#if NO_DIM==3
    double M[3][3];
    if ( not deformationTensor( Ax, Lag, M ) ) return td;   // degenerate simplex: bits stay 0

    double ev[3];
    int const nReal = realEigenvalues3( M, ev );
    if ( nReal < 1 ) return td;
    td.bits = bitsFromSpectrum( ev, nReal );

    int best = 0;
    for (int i=1; i<nReal; ++i)
        if ( std::fabs(ev[i]) < std::fabs(ev[best]) ) best = i;
    td.critical = ev[best];
    for (int i=0; i<nReal; ++i) td.spread = std::max( td.spread, std::fabs(ev[i]) );
    if ( not nullVector3( M, ev[best], td.vcrit ) ) return td;
    td.valid = true;
#else
    // 2D: the characteristic quadratic; a complex pair (a rotation) has no axis to fold along
    double M[2][2];
    if ( not deformationTensor( Ax, Lag, M ) ) return td;
    double const tr  = M[0][0] + M[1][1];
    double const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    double const d2  = tr*tr - 4.*det;
    double ev[2] = {0., 0.};
    int nReal = 0;
    if ( d2 >= 0. )
    {
        double const sq = std::sqrt(d2);
        ev[0] = 0.5*(tr + sq);
        ev[1] = 0.5*(tr - sq);
        nReal = 2;
    }
    td.bits = bitsFromSpectrum( ev, nReal );
    if ( nReal < 2 ) return td;
    int const best = std::fabs(ev[1]) < std::fabs(ev[0]) ? 1 : 0;
    td.critical = ev[best];
    td.spread = std::max( std::fabs(ev[0]), std::fabs(ev[1]) );
    // the null vector of M - lambda I: perpendicular to its longer row
    double const A[2][2] = { { M[0][0] - ev[best], M[0][1] }, { M[1][0], M[1][1] - ev[best] } };
    int const r = ( A[1][0]*A[1][0] + A[1][1]*A[1][1] > A[0][0]*A[0][0] + A[0][1]*A[0][1] ) ? 1 : 0;
    double const n = std::sqrt( A[r][0]*A[r][0] + A[r][1]*A[r][1] );
    if ( n > 0. ) { td.vcrit[0] = -A[r][1] / n; td.vcrit[1] = A[r][0] / n; }
    else          { td.vcrit[0] = 1.; td.vcrit[1] = 0.; }       // M = lambda I: every direction is null
    td.valid = true;
#endif
    return td;
}


/* The A3 (cusp) indicator: 'self' is the tetrahedron's deformation, 'dq' the Lagrangian offsets of the neighbours from its
 * centroid and 'dLambda' their critical eigenvalue minus its own. The gradient g of lambda_c is the least-squares solution of
 * dq . g = dLambda; the cusp condition is |g . v_c| << |g|. False unless the tet is BOTH near its fold and tangent there. */
inline bool cuspIndicator(TetDeformation const &self,
                          double const dq[][NO_DIM], double const dLambda[], int n)
{
#if NO_DIM==3
    if ( not self.valid or n < 3 ) return false;
    if ( self.spread <= 0. ) return false;
    if ( std::fabs(self.critical) > FOLD_BAND * self.spread ) return false;   // not near a fold

    double A[3][3] = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}}, b[3] = {0.,0.,0.};
    for (int k=0; k<n; ++k)
    {
        for (int i=0; i<3; ++i)
        {
            for (int j=0; j<3; ++j) A[i][j] += dq[k][i] * dq[k][j];
            b[i] += dq[k][i] * dLambda[k];
        }
    }
    double Acopy[3][3];
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) Acopy[i][j] = A[i][j];
    double const detA = determinant( Acopy );
    double sc = 0.;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) sc = std::max( sc, std::fabs(A[i][j]) );
    if ( sc <= 0. or not std::isfinite(detA) or std::fabs(detA) < 1.e-12 * sc*sc*sc )
        return false;                       // neighbours too coplanar to fix a gradient

    // Cramer's rule: g = A^-1 b
    double g[3];
    for (int c=0; c<3; ++c)
    {
        double T[3][3];
        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j) T[i][j] = (j==c ? b[i] : A[i][j]);
        g[c] = determinant( T ) / detA;
    }
    /* The yardstick is the largest observed slope |dlambda|/|dq|, not |g|: on an irregular tessellation g's transverse
       components absorb fit noise, so neighbours clustered in a slab give a small |g.v_c| beside a noise-inflated |g| and
       fold cells would be marked as cusps. */
    double slope = 0.;
    for (int k=0; k<n; ++k)
    {
        double const len = std::sqrt( dq[k][0]*dq[k][0] + dq[k][1]*dq[k][1] + dq[k][2]*dq[k][2] );
        if ( len > 0. ) slope = std::max( slope, std::fabs(dLambda[k]) / len );
    }
    double const gn = slope;
    /* A VANISHING variation is the cusp condition in its strongest form (lambda_c is stationary at an A3 point), but ONLY
       when the tetrahedron is essentially ON the fold: neighbours whose critical eigenvalues coincide while lambda_c is
       far from zero would otherwise mark fold cells as cusps. FOLD_BAND is generous so a gradient can be measured on the
       approach; with no gradient to lean on this branch demands lambda_c has actually reached zero. */
    if ( gn <= 0. ) return std::fabs(self.critical) <= 0.02 * self.spread;
    double const along = std::fabs( g[0]*self.vcrit[0] + g[1]*self.vcrit[1] + g[2]*self.vcrit[2] );
    return along <= CUSP_REL_TOL * gn;
#elif NO_DIM==2
    /* 2D, the same test: the generic singularities are folds (A2, lines) and cusps (A3, points). The gradient of lambda_c is
       the 2x2 least-squares fit over the vertex-incident ring, judged against the largest measured slope, as above. */
    if ( not self.valid or n < 2 ) return false;
    if ( self.spread <= 0. ) return false;
    if ( std::fabs(self.critical) > FOLD_BAND * self.spread ) return false;   // not near a fold
    double a00 = 0., a01 = 0., a11 = 0., b0 = 0., b1 = 0.;
    for (int k=0; k<n; ++k)
    {
        a00 += dq[k][0]*dq[k][0]; a01 += dq[k][0]*dq[k][1]; a11 += dq[k][1]*dq[k][1];
        b0  += dq[k][0]*dLambda[k]; b1 += dq[k][1]*dLambda[k];
    }
    double const detA = a00*a11 - a01*a01;
    double const sc = std::max( std::fabs(a00), std::max( std::fabs(a01), std::fabs(a11) ) );
    if ( sc <= 0. or not std::isfinite(detA) or std::fabs(detA) < 1.e-12 * sc*sc )
        return false;                       // neighbours too collinear to fix a gradient
    double const g[2] = { ( a11*b0 - a01*b1) / detA, (-a01*b0 + a00*b1) / detA };
    double slope = 0.;
    for (int k=0; k<n; ++k)
    {
        double const len = std::sqrt( dq[k][0]*dq[k][0] + dq[k][1]*dq[k][1] );
        if ( len > 0. ) slope = std::max( slope, std::fabs(dLambda[k]) / len );
    }
    if ( slope <= 0. ) return std::fabs(self.critical) <= 0.02 * self.spread;
    double const along = std::fabs( g[0]*self.vcrit[0] + g[1]*self.vcrit[1] );
    return along <= CUSP_REL_TOL * slope;
#else
    (void)self; (void)dq; (void)dLambda; (void)n;
    return false;
#endif
}


/* Gaussian elimination with partial pivoting for foldDerivatives()' normal equations. Returns false
   when the system is singular relative to its own scale, i.e. this stencil does not fix the fit. */
inline bool solveSmall(double A[][5], double b[], int n, double x[])
{
    double scale = 0.;
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j) scale = std::max( scale, std::fabs(A[i][j]) );
    if ( scale <= 0. ) return false;

    for (int c=0; c<n; ++c)
    {
        int piv = c;
        for (int r=c+1; r<n; ++r)
            if ( std::fabs(A[r][c]) > std::fabs(A[piv][c]) ) piv = r;
        if ( std::fabs(A[piv][c]) < 1.e-12 * scale ) return false;
        if ( piv != c )
        {
            for (int j=0; j<n; ++j) std::swap( A[c][j], A[piv][j] );
            std::swap( b[c], b[piv] );
        }
        for (int r=c+1; r<n; ++r)
        {
            double const f = A[r][c] / A[c][c];
            if ( f == 0. ) continue;
            for (int j=c; j<n; ++j) A[r][j] -= f * A[c][j];
            b[r] -= f * b[c];
        }
    }
    for (int r=n-1; r>=0; --r)
    {
        double s = b[r];
        for (int j=r+1; j<n; ++j) s -= A[r][j] * x[j];
        x[r] = s / A[r][r];
    }
    for (int i=0; i<n; ++i)
        if ( not std::isfinite(x[i]) ) return false;
    return true;
}


/* The first three derivatives of lambda_c along its null direction v_c, plus the yardsticks the A3/A4 tests judge them by.
 * Model, with t = dq . v_c: dLambda ~= g . dq + c t^2 + d t^3 (5 unknowns, anchored at the centroid); D1 = g.v_c, D2 = 2c,
 * D3 = 6d. g stays fully 3-D (dropping the transverse variation would alias into the axial terms). The cubic term is NOT
 * optional: near a swallowtail a purely linear fit returns g.v_c ~ 0.6 of the slope, above cuspIndicator's threshold, so
 * swallowtailIndicator re-derives D1 from this model and a cell can be flagged A4 but not A3. */
struct FoldDerivatives
{
    bool   valid = false;
    double d1 = 0.;      // (v_c . grad) lambda_c
    double d2 = 0.;      // (v_c . grad)^2 lambda_c
    double d3 = 0.;      // (v_c . grad)^3 lambda_c -- the A4 non-degeneracy witness
    double slope = 0.;   // max |dLambda| / |dq| over the stencil: the measured yardstick for d1
    double span  = 0.;   // max |dq|: the stencil radius the term contributions are weighed at
    double signal = 0.;  // max |dLambda|: the variation the fit has to account for
};


inline FoldDerivatives foldDerivatives(TetDeformation const &self,
                                       double const dq[][NO_DIM], double const dLambda[], int n)
{
    FoldDerivatives fd;
#if NO_DIM==3
    // Five unknowns: five samples would fit exactly (fit the noise); under 10 is not worth a second-derivative claim.
    if ( not self.valid or n < 10 ) return fd;

    double A[5][5] = {{0.}}, b[5] = {0.,0.,0.,0.,0.};
    for (int k=0; k<n; ++k)
    {
        double const t = dq[k][0]*self.vcrit[0] + dq[k][1]*self.vcrit[1] + dq[k][2]*self.vcrit[2];
        double const row[5] = { dq[k][0], dq[k][1], dq[k][2], t*t, t*t*t };
        for (int i=0; i<5; ++i)
        {
            for (int j=0; j<5; ++j) A[i][j] += row[i] * row[j];
            b[i] += row[i] * dLambda[k];
        }
        double const len2 = dq[k][0]*dq[k][0] + dq[k][1]*dq[k][1] + dq[k][2]*dq[k][2];
        if ( len2 <= 0. ) continue;
        double const len = std::sqrt( len2 ), aL = std::fabs( dLambda[k] );
        fd.slope  = std::max( fd.slope,  aL / len );
        fd.span   = std::max( fd.span,   len );
        fd.signal = std::max( fd.signal, aL );
    }

    double x[5];
    if ( not solveSmall( A, b, 5, x ) ) return fd;
    fd.d1 = x[0]*self.vcrit[0] + x[1]*self.vcrit[1] + x[2]*self.vcrit[2];
    fd.d2 = 2. * x[3];
    fd.d3 = 6. * x[4];
    fd.valid = true;
#else
    (void)self; (void)dq; (void)dLambda; (void)n;
#endif
    return fd;
}


/* The A4 (swallowtail) indicator: the fold is tangent to its null direction AND that tangency is itself degenerate.
   Same stencil and arguments as cuspIndicator. Proximity along the cusp curve, not membership of the A4 stratum. */
inline bool swallowtailIndicator(TetDeformation const &self,
                                 double const dq[][NO_DIM], double const dLambda[], int n)
{
    // 2D: never set. A swallowtail is codimension 3, so in 2D it appears only at isolated moments: nothing to flag.
#if NO_DIM==3
    if ( not self.valid or self.spread <= 0. ) return false;
    if ( std::fabs(self.critical) > FOLD_BAND * self.spread ) return false;   // not near a fold

    FoldDerivatives const fd = foldDerivatives( self, dq, dLambda, n );
    if ( not fd.valid ) return false;
    if ( fd.slope <= 0. or fd.span <= 0. or fd.signal <= 0. ) return false;   // nothing varies

    // Tangency keeps its own MEASURED yardstick, as in cuspIndicator, so a large true slope is
    // rejected before any ratio of fitted terms is trusted.
    if ( std::fabs(fd.d1) > CUSP_REL_TOL * fd.slope ) return false;

    double const h  = fd.span;
    double const c1 = std::fabs(fd.d1) * h;
    double const c2 = std::fabs(fd.d2) * h*h / 2.;
    double const c3 = std::fabs(fd.d3) * h*h*h / 6.;

    // The cubic term must account for a real share of the measured variation, or the ratios below
    // pass trivially wherever lambda_c is flat. Also rejects A5+, where D3 vanishes too.
    if ( c3 < SWALLOWTAIL_SIGNAL_MIN * fd.signal ) return false;

    // and the expansion must START at that cubic term: both lower orders negligible beside it
    return c1 <= SWALLOWTAIL_TERM_TOL * c3 and c2 <= SWALLOWTAIL_TERM_TOL * c3;
#else
    (void)self; (void)dq; (void)dLambda; (void)n;
    return false;
#endif
}

}   // namespace PSCausticClass

#endif
