// Part of src/CGAL_triangulation/ps_interpolation.cc, inside its #ifdef PHASE_SPACE: included once, in order, by
// that file and compiled only as part of its object, never alone.
// The pass's close: the fixed-point words to Real, the messages, the moments' normalization, the mass audit's
// line, the stream statistics and export, the phase report (psPhaseReport).

// The fixed-point words become their Real grids, each cell rounded once: the moment grids', then the density.
static inline void psFixedToReal(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, PSFxState &fx)
{
    Quantities *quantities = passCtx.quantities;
    size_t const totalGrid = gridsCtx.totalGrid;
    PSFxWords &massFx = gridsCtx.massFx;
    float &massFxScale = gridsCtx.massFxScale;
    bool &massFxOn = gridsCtx.massFxOn;
    bool const dispNeedsOwnWeight = gridsCtx.dispNeedsOwnWeight;
    std::vector<Real> &dispWeight = gridsCtx.dispWeight;
    // Grid by grid (fxg), each grid's words freed before the next is converted; the mass words last, into the float
    // density (see massFx) that the rho/rho_bar conversion after the moments' normalization reads. Here, after the
    // private copies, the slab lists and (on the final call) the triangulation are freed, the 4 B float grid and the
    // 8 B words briefly coexist.
    for (int g = 0; g < PS_FX_CPU_N; ++g) fx.toReal( g );
    if ( fx.face )   // --face-gradient: the face sums, the same way (normalizePhaseSpace turns them into the gradient)
    {
        psFxToReal( fx.faceS, quantities->face_velocity, totalGrid );
        psFxToReal( fx.faceN, quantities->face_coverage, totalGrid );
        psFxToReal( fx.faceP, quantities->face_momentum, totalGrid );
    }
    if ( massFxOn )
    {
        quantities->density.assign( totalGrid, Real(0.) );
        float const scale = massFxScale;
        psCellRanges( totalGrid, [&](size_t const lo, size_t const hi)
        {
            for (size_t i = lo; i < hi; ++i)
                quantities->density[i] = Real( psMassFixedMass( massFx.w[i], scale ) );
        } );
        massFx.release();
        massFxOn = false;
        // The dispersion's own mass normalizer (dispWeight) sums the very shares this grid sums, at the same
        // sites (the GPU's dispw likewise), so it takes these exact sums too (a float sum of its own would part
        // from them at the last bit).
        // (dispWeight is not even accumulated then -- it may be EMPTY here: dwOffL in the loop)
        if ( dispNeedsOwnWeight )
            dispWeight = quantities->density;
    }
}

// The pass's closing lines: done, the degenerate cells dropped, the halo tetrahedra released.
static inline void psPassMessages(PSPassCtx const &passCtx, size_t const nDegenerateInverse, size_t const nReleased)
{
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
    if ( not userOptions.psSuppressGridStats )
    {
        message << MESSAGE::cGreen() << "Done.\n" << MESSAGE::cReset() << MESSAGE::Flush;
        if ( nDegenerateInverse > 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " dropped "
                    << MESSAGE::cMagenta() << nDegenerateInverse << MESSAGE::cReset()
                    << " degenerate (non-invertible) Eulerian cells.\n" << MESSAGE::Flush;
        if ( nReleased > 0 )
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " released "
                    << MESSAGE::cMagenta() << nReleased << MESSAGE::cReset()
                    << " halo-interior tetrahedra (rho_geo/rho_bar > " << userOptions.psHaloRelease
                    << ") to monolithic centroid deposits.\n" << MESSAGE::Flush;
    }
}

// The moments become means (divided by their weight, the dispersion closed out); deferred: the weights handed on.
static inline void psNormalizeMoments(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx)
{
    Quantities *quantities = passCtx.quantities;
    Field &field = passCtx.field;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<Real> &massWeight = gridsCtx.massWeight;
    std::vector<Real> &dispWeight = gridsCtx.dispWeight;
    std::vector< Pvector<Real,noVelComp> > &dispVel = gridsCtx.dispVel;
    bool const haveVel = gridsCtx.haveVel, needWeight = gridsCtx.needWeight, deferNorm = gridsCtx.deferNorm;
    bool const dispNeedsOwnWeight = gridsCtx.dispNeedsOwnWeight;
    if ( needWeight )
    {
        if ( deferNorm )
        {
            quantities->mass_weight.swap( massWeight );   // sum(V_s); normalized later by caller
            if ( dispNeedsOwnWeight )
            {   // the dispersion's mass-weighted mean/normalizer must survive the partition merge
                quantities->disp_weight.swap( dispWeight );
                quantities->disp_velocity.swap( dispVel );
            }
        }
        else
        {
            // --face-gradient: the mean flow's gradient from the face sums (Quantities::psFaceGradientCell), the
            // regular part's twins kept for the '_a' pass the writer writes (User_options::psStreamsTwins, DTFE_setup)
            bool const face = !quantities->face_velocity.empty() && field.velocity_gradient;
            unsigned const twins = &field == &passCtx.userOptions.aField ? passCtx.userOptions.psStreamsTwins : 0u;
            if ( face ) quantities->psFaceGradientBegin( totalGrid, twins );   // (psFaceDx: set with fx.face)
            for (size_t i = 0; i < totalGrid; ++i)
            {
                Real const w = massWeight[i];
                if ( w > Real(0.) )
                {
                    Real const inv = Real(1.) / w;
                    if (haveVel)                 quantities->velocity[i]          *= inv;   // <v>
                    if (field.velocity_gradient) quantities->velocity_gradient[i] *= inv;
                    if (face)                    quantities->psFaceGradientCell( i );
#ifdef SCALAR
                    if (field.scalar)            quantities->scalar[i]            *= inv;
                    if (field.scalar_gradient)   quantities->scalar_gradient[i]   *= inv;
#endif
                    // (the dispersion is closed out in its own pass below: it is mass-weighted, against its own
                    //  mean and normalizer, while 'velocity' here is the volume-weighted mean)
                }
            }
            if ( face ) quantities->psFaceGradientEnd();
        }

        // serial path: close out the MASS-weighted dispersion against its own mean/normalizer (the loop above
        // normalized 'velocity' by the VOLUME weight, so it is not the right mean here). Mirrors
        // normalizePhaseSpace's second pass.
        if ( !deferNorm && dispNeedsOwnWeight )
            for (size_t i = 0; i < totalGrid; ++i)
            {
                Real const wm = dispWeight[i];
                if ( wm <= Real(0.) ) continue;
                Real const invm = Real(1.) / wm;
                Pvector<Real,noVelComp> const vbar = dispVel[i] * invm;   // <v>_mass
                size_t c = 0;
                for (int a = 0; a < NO_DIM; ++a)
                    for (int b = a; b < NO_DIM; ++b)
                    {
                        Real s = quantities->velocity_dispersion[i][c] * invm - vbar[a]*vbar[b];
                        if (a == b and s < Real(0.)) s = Real(0.);   // variance: clamp FP-noise negatives
                        quantities->velocity_dispersion[i][c] = s;
                        ++c;
                    }
            }
    }
}

// DTFE_MASS_AUDIT: the pass's audit line and the run's totals (a template: the audit and gridSum are the pass's types).
template <class MassAudit, class GridSum>
static inline void psMassAuditLine(PSPassCtx const &passCtx, bool const massAudit, MassAudit &audit,
                                   GridSum const &gridSum, Real const cellVolume, double const auditGridPre)
{
    User_options &userOptions = passCtx.userOptions;
    Field &field = passCtx.field;
    int const nSub = passCtx.nSub;
    if ( massAudit )
    {   // (g) the pass's grid in mass units, before and after the rho/rho_bar conversion; the line (ps_mass_audit.h)
        double const gridPost = !field.density ? -1.
                              : gridSum() * double(cellVolume) * double(userOptions.averageDensity);
        auto rel = [](double x, double ref) { return ref != 0. ? x / ref - 1. : 0.; };
        std::ostringstream s;
        s << std::setprecision(12) << "PS mass audit: partition ";
        if ( PSMassAudit::partition >= 0 ) s << PSMassAudit::partition; else s << "- (one domain)";
        s << " (nSub " << nSub << (&field == &userOptions.aField ? ", averaged" : ", unaveraged") << " pass):"
          << " (a) owned " << audit.owned << " [" << audit.nOwned << " vertices]"
          << " (b) tets " << audit.tets << " [" << audit.nTets << "]"
          << " (c) deg0 " << audit.deg0 << " [" << audit.nDeg0 << " vertices; kept tets touching one " << audit.nTetsDeg0 << "]"
          << " (d) dropped " << audit.nDropped << " owned tets (rho_bar V_lag " << audit.dropped << "), after the filter "
          << audit.after << " [" << audit.nAfter << "]";
        if ( audit.gpuHanded >= 0. )
            s << " (e) gpu handed " << audit.gpuHanded << " deferred " << audit.gpuDeferred << " fixed-point " << audit.gpuFixed
              << " (handed-deferred-fixed " << ( audit.gpuHanded - audit.gpuDeferred - audit.gpuFixed ) << ")"
              << " float grid " << audit.gpuFloat << " (rel " << rel(audit.gpuFloat, audit.gpuFixed) << ")"
              << " released " << audit.released << " gained " << audit.releasedGain;
        s << " (f) cpu tets " << audit.cpuTets << " [" << audit.nCpuTets << "] shares " << audit.cpuShares
          << " gained " << audit.cpuGain << " (shares/tets-1 " << rel(audit.cpuShares, audit.cpuTets)
          << ", gained/shares-1 " << rel(audit.cpuGain, audit.cpuShares) << ")"
          << " (g) grid " << auditGridPre << " post " << gridPost << " (grid/tets-1 " << rel(auditGridPre, audit.tets)
          << ", post/grid-1 " << rel(gridPost, auditGridPre) << ", grid/owned-1 " << rel(auditGridPre, audit.owned) << ")\n";
        std::fprintf( stderr, "%s", s.str().c_str() );
        std::lock_guard<std::mutex> lock( PSMassAudit::mutex );
        PSMassAudit::Totals &T = PSMassAudit::totals[&field == &userOptions.aField ? 1 : 0];
        T.owned += audit.owned; T.tets += audit.tets; T.deg0 += audit.deg0; T.dropped += audit.dropped;
        T.gridPre += auditGridPre; T.gridPost += gridPost; ++T.passes;
    }
}

// The pass's multi-stream and coverage statistics.
static inline void psStreamStats(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, Real const invSamples,
                                 size_t const nSamplesPerCell)
{
    User_options &userOptions = passCtx.userOptions;
    MESSAGE::Message &message = passCtx.message;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<int> &streamCount = gridsCtx.streamCount;
    std::vector<Real> &releasedMult = gridsCtx.releasedMult;
    std::vector<Real> &exactMult = gridsCtx.exactMult;
    bool const psExact = gridsCtx.psExact;
    // multi-stream / coverage statistics. Suppressed in the partition path (each partition covers
    // only part of the grid); DTFE.cpp reports one aggregate from the summed grid.
    if ( not userOptions.psSuppressGridStats )
    {
        Real maxStreams = Real(0.);
        size_t multiStreamCells = 0;
        size_t coveredCells = 0;
        // over the free cores in cell ranges: a maximum (NaN never taken) and two counts, the same in any order
        std::mutex statsMutex;
        psCellRanges( totalGrid, [&](size_t const lo, size_t const hi)
        {
        Real maxStreamsR = Real(0.);
        size_t multiStreamCellsR = 0, coveredCellsR = 0;
        for (size_t i = lo; i < hi; ++i)
        {
            // mean streams over the cell. Under psExact 'streamCount' is the raw tet-TOUCH count
            // ('.tetTouch', up to hundreds), NOT a multiplicity -- the physical one is exactMult,
            // which is what '.streams' exports.
            Real const relFrac = releasedMult.empty() ? Real(0.) : releasedMult[i];
            Real const avg = psExact ? exactMult[i] : Real(streamCount[i]) * invSamples + relFrac;
            if (avg > maxStreamsR) maxStreamsR = avg;
            // '> 1' to a tolerance: exactMult (and a released volume fraction) is a float sum, so
            // single-stream cells sit at 1.0 +/- eps; a pure sample count is an exact integer
            // multiple of invSamples and compares exactly.
            if ((psExact || relFrac > Real(0.)) ? (avg > Real(1.) + PS_STREAM_TOL)
                                            : (streamCount[i] > (int)nSamplesPerCell)) multiStreamCellsR++;
            if (streamCount[i] > 0 || relFrac > Real(0.)) coveredCellsR++;   // a stream reaches the cell
        }
        std::lock_guard<std::mutex> lock( statsMutex );
        if (maxStreamsR > maxStreams) maxStreams = maxStreamsR;
        multiStreamCells += multiStreamCellsR;
        coveredCells += coveredCellsR;
        } );
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Max streams at a grid point: "
                << MESSAGE::cMagenta() << maxStreams << MESSAGE::cReset()
                << ", grid points with multi-stream: " << MESSAGE::cMagenta() << multiStreamCells
                << " (" << (100.*multiStreamCells/totalGrid) << "\%)" << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " grid coverage: "
                << MESSAGE::cMagenta() << coveredCells << "/" << totalGrid
                << " (" << (100.*coveredCells/totalGrid) << "\%)" << MESSAGE::cReset()
                << ( coveredCells < totalGrid ?
                     "  -- uncovered cells suggest insufficient padding or a grid finer than the tessellation" : "" )
                << "\n" << MESSAGE::Flush;
    }
}

// Exports the per-cell stream fields: '.streams', '.hidden_streams', '.tetTouch' and the caustic bits.
static inline void psExportStreams(PSPassCtx const &passCtx, PSGridsCtx const &gridsCtx, Real const invSamples)
{
    User_options &userOptions = passCtx.userOptions;
    Quantities *quantities = passCtx.quantities;
    size_t const totalGrid = gridsCtx.totalGrid;
    std::vector<int> &streamCount = gridsCtx.streamCount;
    std::vector<Real> &releasedMult = gridsCtx.releasedMult;
    std::vector<unsigned char> &subSampleMassFlag = gridsCtx.subSampleMassFlag;
    std::vector<unsigned char> &flippedFlag = gridsCtx.flippedFlag;
    std::vector<Real> &exactMult = gridsCtx.exactMult;
    std::vector<PSCausticClass::CausticMask> &orientBits = gridsCtx.orientBits;
    bool const psExact = gridsCtx.psExact, psCaustics = gridsCtx.psCaustics;
    // Export the per-grid-point stream count: the sampled deposit's sample-mean multiplicity,
    // or -- under --exact -- the ANALYTIC cell-mean multiplicity (1/V_cell) sum V_int,
    // its exact nSub->infinity limit. Both are the same physical quantity on the same scale, so
    // single-stream cells select the same way under either deposit -- but only TO A TOLERANCE
    // (PS_STREAM_TOL / dtfelib.STREAM_TOL): this is a float, never exactly 1.
    if ( psExact )
        quantities->stream_count.swap(exactMult);   // same type and exactMult is dead below: no copy
    else
    {
        quantities->stream_count.resize(totalGrid);
        psCellRanges( totalGrid, [&](size_t const lo, size_t const hi)   // (cell by cell: over the free cores)
        {
            for (size_t i = lo; i < hi; ++i)
                quantities->stream_count[i] = Real(streamCount[i]) * invSamples;
            if ( not releasedMult.empty() )   // released (never sampled) tets: their volume fractions
                for (size_t i = lo; i < hi; ++i)
                    quantities->stream_count[i] += releasedMult[i];
        } );
    }

    // '.hidden_streams' bitmask (quantities.h): bit 1 = the exact multi-stream flag (a flipped tet
    // overlaps the cell; DTFE() clears it where '.streams' already reads multi-stream), bit 2 = the
    // mass flag (a never-sampled tet deposited here: the cell mean includes a centroid-fallback
    // piece). The degree-1 argument behind bit 1 needs the periodic box's complete sheet; a
    // non-periodic sheet has an edge whose image can overlap without folding, so there bit 1
    // keeps the conservative mass flag as well.
    quantities->hidden_streams.resize(totalGrid);
    bool const massIsMulti = !userOptions.periodic;
    psCellRanges( totalGrid, [&](size_t const lo, size_t const hi)
    {
        for (size_t i = lo; i < hi; ++i)
            quantities->hidden_streams[i] = Real( ((flippedFlag[i] || (massIsMulti && subSampleMassFlag[i])) ? 1 : 0)
                                            | (subSampleMassFlag[i] ? 2 : 0) );
    } );

    // --exact: also export the raw integer tet-touch count ('.tetTouch'), the
    // number of tetrahedra with a nonzero intersection with the cell. Kept as a geometric
    // diagnostic (tessellation multiplicity, hundreds in resolved regions); it is NOT a
    // stream count and is not comparable to the sampled deposit's '.streams'.
    if ( psExact )
    {
        quantities->tet_touch.resize(totalGrid);
        psCellRanges( totalGrid, [&](size_t const lo, size_t const hi)
        {
            for (size_t i = lo; i < hi; ++i)
                quantities->tet_touch[i] = Real(streamCount[i]);
        } );
    }

    // Export the caustic orientation bits (0..3, exact in float); OR-merged across partitions, and collapsed to
    // the 0/1 fold flag once, in writeOutputData (input_output.cc), after '.causticClass' is written.
    /* The parity bit 1 (det < 0) is the GEOMETRIC overlap flag.
       orientBits gets its bits only where a tetrahedron DEPOSITS A SAMPLE, so a fold thinner than
       the sub-sample spacing -- every young wall, and the edge of every older one -- sets no
       negative-parity bit in the cells it crosses: per sample, '.caustic' would mark far fewer
       cells than the folded tetrahedra overlap, with a fraction depending on nSub, although the
       '--ps-caustics' help, ps_caustic_class.h and dtfelib promise
       "a det < 0 tetrahedron OVERLAPS the cell". flippedFlag is exactly that
       statement: the separating-axis tet-box test (TetBoxSAT; 2D: the triangle clip) of every
       folded tetrahedron, marked in both deposit paths (the host loop marks the GPU's tets) and
       still RAW here -- Quantities::finalizeHiddenStreams clears it in multi-stream cells only
       after the merge, and only in '.hidden_streams'. OR-ing it in keeps the export partition-,
       thread- and backend-invariant (OR of two OR-merged geometric masks).
       Only bit 1 comes from the geometry. Bit 0 (det > 0) stays per sample: in a periodic box the
       map has degree 1, so every point of the cell lies in a positive tetrahedron and any sample
       of the cell sets it -- '(bits & 3) == 3' is therefore the geometric fold flag in every cell
       a sample reached. The collapse and indicator bits 2..8 are per sample too, NOT geometric:
       a fold cell whose samples all fell outside the folded tetrahedra carries the parity of the
       fold but only the k of the streams its samples saw (typically k = 0) -- nSub-limited, as
       the README notes. */
    if ( psCaustics )
    {
        quantities->caustic_bits.resize(totalGrid);
        psCellRanges( totalGrid, [&](size_t const lo, size_t const hi)
        {
            for (size_t i = lo; i < hi; ++i)
                quantities->caustic_bits[i] = Real( orientBits[i]
                    | ( flippedFlag[i] ? (PSCausticClass::CausticMask)PSCausticClass::BIT_PARITY_NEG
                                       : (PSCausticClass::CausticMask)0 ) );
        } );
    }
}

// The pass's phase split and the run's totals so far (psPhaseTotalUs), one line each.
static inline void psPhaseReport(PSPassCtx const &passCtx, double (&phaseSec)[PS_PH_N], bool const metalDeposited,
                                 std::vector<uint32_t> &deferredOrd)
{
    MESSAGE::Message &message = passCtx.message;
    size_t const nDeferredHere = metalDeposited ? deferredOrd.size() : 0;
    for (int k = 0; k < PS_PH_N; ++k) psPhaseTotalUs[k] += (long long)( phaseSec[k] * 1e6 );
    psPhaseTotalDeferred += (long long)nDeferredHere;
    int const nPasses = ++psPhasePasses;
    {
        std::ostringstream line, total;
        line << std::fixed << std::setprecision(2);
        total << std::fixed << std::setprecision(1);
        bool first = true;
        for (int k = 0; k < PS_PH_N; ++k)
        {
            double const tot = double( psPhaseTotalUs[k].load() ) * 1e-6;
            if ( phaseSec[k] <= 0. && tot <= 0. ) continue;   // a phase this build/run never enters
            line  << (first ? "" : ", ") << psPhaseName[k] << " " << phaseSec[k] << " s";
            total << (first ? "" : ", ") << psPhaseName[k] << " " << tot << " s";
            if ( k == PS_PH_DEFERRED )
            {
                line  << " (" << nDeferredHere << " tets)";
                total << " (" << psPhaseTotalDeferred.load() << " tets)";
            }
            first = false;
        }
        message << MESSAGE::cGreen() << "  >>> field interpolation phases: " << line.str() << "\n"
                << "  >>> ... run total over " << nPasses << " pass(es): " << total.str() << MESSAGE::cReset() << "\n" << MESSAGE::Flush;
    }
}
