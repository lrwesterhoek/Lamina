/*
 *  Interlacing (FFTW3): F_out(k) = 0.5*(F1(k) + F2(k)*exp(-i*k.dx/2)) then IFFT back; the
 *  phase term on the half-cell-offset grid cancels the leading-order aliasing.
 */

#include "interlacing.h"
#include "message.h"
#include <cmath>
#include <complex>

#include <fftw3.h>
#include "fftw_plans64.h"

// FFTW precision follows the Real type
#ifdef DOUBLE
    #define FFTW_PLAN       fftw_plan
    #define FFTW_COMPLEX    fftw_complex
    #define FFTW_ALLOC_REAL fftw_alloc_real
    #define FFTW_ALLOC_COMPLEX fftw_alloc_complex
    #define FFTW_EXECUTE    fftw_execute
    #define FFTW_DESTROY    fftw_destroy_plan
    #define FFTW_FREE       fftw_free
#else
    #define FFTW_PLAN       fftwf_plan
    #define FFTW_COMPLEX    fftwf_complex
    #define FFTW_ALLOC_REAL fftwf_alloc_real
    #define FFTW_ALLOC_COMPLEX fftwf_alloc_complex
    #define FFTW_EXECUTE    fftwf_execute
    #define FFTW_DESTROY    fftwf_destroy_plan
    #define FFTW_FREE       fftwf_free
#endif


void applyInterlacing(std::vector<Real> &field1,
                      std::vector<Real> &field2,
                      size_t const *nGrid,
                      Real const *dx)
{
    (void)dx;   // the phase per cell of shift is pi*m/n whatever the cell size
    // 64-bit sizes throughout (fftw_plans64.h): the ordinary planners stop at 2^31 cells
    size_t n[NO_DIM];
    size_t N = 1;
    for (int d = 0; d < NO_DIM; ++d) { n[d] = nGrid[d]; N *= n[d]; }
    size_t const nLast = n[NO_DIM-1], nHalf = nLast/2 + 1;      // the last axis is the half spectrum
    size_t const Ncomplex = N / nLast * nHalf;

    Real *in1 = FFTW_ALLOC_REAL(N);
    Real *in2 = FFTW_ALLOC_REAL(N);
    FFTW_COMPLEX *out1 = FFTW_ALLOC_COMPLEX(Ncomplex);
    FFTW_COMPLEX *out2 = FFTW_ALLOC_COMPLEX(Ncomplex);

    for (size_t i = 0; i < N; ++i)
    {
        in1[i] = field1[i];
        in2[i] = field2[i];
    }

    // forward FFT of both grids
    FFTW_PLAN plan_fwd1 = fftwPlans64::r2c(NO_DIM, n, in1, out1, FFTW_ESTIMATE);
    FFTW_PLAN plan_fwd2 = fftwPlans64::r2c(NO_DIM, n, in2, out2, FFTW_ESTIMATE);
    FFTW_EXECUTE(plan_fwd1);
    FFTW_EXECUTE(plan_fwd2);
    FFTW_DESTROY(plan_fwd1);
    FFTW_DESTROY(plan_fwd2);

    /* The second grid samples f(x + dx/2), so its transform carries exp(+i k.dx/2); undo it with
       exp(-i pi sum_d m_d/n_d), m_d the SIGNED frequency (index - n above n/2; the last axis is the half spectrum,
       m >= 0). The raw index would flip the sign of the correction for half the modes (tests/interlacing_check.sh). */
    for (size_t c = 0; c < Ncomplex; ++c)
    {
        size_t rem = c;
        double sum = double(rem % nHalf) / double(nLast);
        rem /= nHalf;
        for (int d = NO_DIM - 2; d >= 0; --d)
        {
            long m = long(rem % n[d]);
            rem /= n[d];
            if ( m > (long(n[d]) - 1) / 2 ) m -= long(n[d]);
            sum += double(m) / double(n[d]);
        }
        double const phase = -M_PI * sum;
        Real const cos_p = Real(std::cos(phase));
        Real const sin_p = Real(std::sin(phase));

        // F2_corrected = F2 * exp(i*phase)
        Real const f2_re = out2[c][0];
        Real const f2_im = out2[c][1];
        Real const f2c_re = f2_re * cos_p - f2_im * sin_p;
        Real const f2c_im = f2_re * sin_p + f2_im * cos_p;

        // Average: F_out = 0.5 * (F1 + F2_corrected)
        out1[c][0] = Real(0.5) * (out1[c][0] + f2c_re);
        out1[c][1] = Real(0.5) * (out1[c][1] + f2c_im);
    }

    // inverse FFT back to real space
    FFTW_PLAN plan_inv = fftwPlans64::c2r(NO_DIM, n, out1, in1, FFTW_ESTIMATE);
    FFTW_EXECUTE(plan_inv);
    FFTW_DESTROY(plan_inv);

    // FFTW c2r does not normalize
    Real norm = Real(1.) / Real(N);
    for (size_t i = 0; i < N; ++i)
        field1[i] = in1[i] * norm;

    FFTW_FREE(in1);
    FFTW_FREE(in2);
    FFTW_FREE(out1);
    FFTW_FREE(out2);
}
