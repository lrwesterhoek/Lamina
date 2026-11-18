// Part of src/DTFE.cpp: included once, in order, by that file and compiled only as part of its object, never alone.
// PS-DTFE only (its own #ifdef PHASE_SPACE): the Lagrangian bounding box, the seam report, the Lagrangian
// padding, the partition plan and its costs (PSPartitionPlan, psPartitionCosts) and the GPU share.

#ifdef PHASE_SPACE
// Global Lagrangian bounding box (min/max of lagPos over all particles) with a tiny
// relative margin so boundary particles are strictly inside.
static void computeLagrangianBoundingBox(vector<Particle_data> const &particles,
                                         Box &lagBox)
{
    for (int d=0; d<NO_DIM; ++d)
    {
        lagBox[2*d]   = particles[0].lagPos[d];
        lagBox[2*d+1] = particles[0].lagPos[d];
    }
    for (size_t i=1; i<particles.size(); ++i)
        for (int d=0; d<NO_DIM; ++d)
        {
            if (particles[i].lagPos[d] < lagBox[2*d])   lagBox[2*d]   = particles[i].lagPos[d];
            if (particles[i].lagPos[d] > lagBox[2*d+1]) lagBox[2*d+1] = particles[i].lagPos[d];
        }
    // tiny margin to include boundary particles
    for (int d=0; d<NO_DIM; ++d)
    {
        Real eps = (lagBox[2*d+1] - lagBox[2*d]) * Real(1.e-6);
        lagBox[2*d]   -= eps;
        lagBox[2*d+1] += eps;
    }
}

// The log line of the periodic seam (subpartition.h psPeriodicSeam): the ownership period per axis
// and whether its copies are exact shifts -- the Lagrangian ones are by construction where 'exact'; the Eulerian
// ones are counted when the particles are in memory (an inexact one leaves a rounding-wide sliver at the seam).
static void psReportSeam(MESSAGE::Message &message, Real const seam[], bool const exact[], Real const eulerLen[],
                         Real const pad[], std::vector<Particle_data> const &particles)
{
    message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " periodic ownership period per axis:";
    for (int d=0; d<NO_DIM; ++d)
    {
        message << " [" << seam[d] - eulerLen[d] << ", " << seam[d] << ")";
        if ( exact[d] )
        {
            if ( not particles.empty() )
            {
                size_t const n = psCountInexactCopies( particles, d, double(seam[d]) - double(pad[d]) - 1.e-6 * double(eulerLen[d]), eulerLen[d] );
                if ( n ) message << " (" << n << " Eulerian copies round)";
            }
        }
        else
            message << " (the top particle plane: copies may round, the padding is too wide for an exact seam)";
    }
    message << ".\n" << MESSAGE::Flush;
}


/* The Lagrangian padding of '--ps-lagrangian-padding <k>' (default 8): k mean particle spacings, the partitions'
   (PSPartitionPlan). psLagMeanSpacing: the spacing (V/N)^(1/NO_DIM) of nPart particles in a domain of side lengths
   len[]; 0 = no spacing rule (k = 0, or no particle count or volume to take one from). */
static double psLagMeanSpacing(Real const k, double const nPart, Real const len[])
{
    if ( not ( k > Real(0.) ) ) return 0.;
    double vol = 1.;
    for (int d=0; d<NO_DIM; ++d) vol *= double( len[d] );
    return ( nPart > 0. and vol > 0. ) ? std::pow( vol / nPart, 1. / double(NO_DIM) ) : 0.;
}
/* ... and the padding on one axis: k spacings; periodic: the copies come from the +-1 images only, so more than one
   period of padding would ask for particles no copy provides; one period already holds every neighbour (a set with
   < 8 particles per axis -- tests only). */
static Real psLagSpacingPadding(Real const k, double const lagSpacing, bool const periodic, Real const period)
{
    Real pad = Real( double(k) * lagSpacing );
    if ( periodic and pad > period ) pad = period;
    return pad;
}

// The Lagrangian partition geometry of a partitioned PS-DTFE run, shared by the batch partition
// loop and the composite '--serve' builder: both must select EXACTLY the same particles for
// partition pi (the server even reuses the batch runs' per-partition tessellation caches, whose
// identity is the partition's lagrangianRegion). The arithmetic is the batch loop's, verbatim.
struct PSPartitionPlan
{
    Box    lagBox;
    Real   lagLen[NO_DIM], eulerLen[NO_DIM], lagPadding[NO_DIM];
    size_t grid[NO_DIM];
    int    total = 1;
    bool   periodicCopies = false;
        // periodic: the partitions tile the period [seam - L, seam) per axis (subpartition.h psPeriodicSeam);
        // seamExact = every periodic copy an exact shift (else the fallback anchor lagBox_hi)
    Real   seam[NO_DIM];
    bool   seamExact[NO_DIM];
        // the smallest and the largest lagPos per axis (xMin, xMax, yMin, ...; no margin, unlike lagBox), with
        // which a partition's selection skips the periodic passes that cannot reach it
        // (findParticlesInBoxLagrangianPeriodic); not known without the particles or when a lagPos is not a number
    Real   lagExtent[2*NO_DIM];
    bool   lagExtentKnown = false;
    Real const* selectionExtent() const { return lagExtentKnown ? lagExtent : nullptr; }

    void init(DTFE_State &state, Box &lagBoxGlobal, bool const hasLagrangianPeriodicCopies,
              MESSAGE::Message &message)
    {
        periodicCopies = hasLagrangianPeriodicCopies;
        total = 1;
        for (int d=0; d<NO_DIM; ++d)
        {
            grid[d] = state.options.partition[d];
            total *= int(grid[d]);
        }

        // reuse the Lagrangian bounding box computed before periodic copies, else recompute it here
        if (!lagBoxGlobal.isNullBox())
            lagBox = lagBoxGlobal;
        else
            computeLagrangianBoundingBox( state.particles, lagBox );
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian bounding box = "
                << MESSAGE::cMagenta() << lagBox.print() << MESSAGE::cReset() << "\n" << MESSAGE::Flush;

        for (int d=0; d<NO_DIM; ++d)
            lagLen[d] = lagBox[2*d+1] - lagBox[2*d];

        // the exact extent of the particles as they are selected (nothing moves them between here and the
        // partition loop); a lagPos that is not a number passes the selection's box test in every pass, so
        // the extent is then left unknown and no pass is skipped
        lagExtentKnown = not state.particles.empty();
        if ( lagExtentKnown )
        {
            for (int d=0; d<NO_DIM; ++d)
                lagExtent[2*d] = lagExtent[2*d+1] = state.particles[0].lagPos[d];
            bool ordered = true;
            for (Particle_data const &q : state.particles)
                for (int d=0; d<NO_DIM; ++d)
                {
                    Real const x = q.lagPos[d];
                    if ( x < lagExtent[2*d] )   lagExtent[2*d]   = x;
                    if ( x > lagExtent[2*d+1] ) lagExtent[2*d+1] = x;
                    if ( not ( x == x ) ) ordered = false;
                }
            lagExtentKnown = ordered;
        }

        // Eulerian box length per axis: the period used to shift each partition's deferred copies
        for (int d=0; d<NO_DIM; ++d)
            eulerLen[d] = state.options.boxCoordinates[2*d+1] - state.options.boxCoordinates[2*d];

        // --ps-lagrangian-padding <k> (default 8): k mean particle spacings, (V/N)^(1/NO_DIM) of
        // the Lagrangian domain (the period when periodic, the cloud's bounding box otherwise); N from the recorded
        // globals when the particles are not read yet (ps_globals_record.h)
        double const lagSpacing = psLagMeanSpacing( state.options.psLagPadding,
                state.particles.empty() ? double(state.options.psPresetN) : double(state.particles.size()),
                periodicCopies ? eulerLen : lagLen );
        // Lagrangian padding per axis: k spacings with no floor (--ps-lagrangian-padding, the default) -- or, with
        // '--ps-lagrangian-padding 0' (or no particle count to take a spacing from), the Eulerian fraction floored at 0.30
        // of a partition cell
        for (int d=0; d<NO_DIM; ++d)
        {
            Real padFrac = (state.options.paddingLength[2*d] + state.options.paddingLength[2*d+1]) / (Real(2.) * eulerLen[d]);
            lagPadding[d] = padFrac * lagLen[d];
            if (lagPadding[d] < lagLen[d] / state.options.partition[d] * Real(0.3))
                lagPadding[d] = lagLen[d] / state.options.partition[d] * Real(0.3);
            // (periodic: at most one period -- psLagSpacingPadding)
            if ( lagSpacing > 0. )
                lagPadding[d] = psLagSpacingPadding( state.options.psLagPadding, lagSpacing, periodicCopies,
                                                     eulerLen[d] );
            // --ps-alpha-shape (non-periodic): a kept tetrahedron (circumradius <= a) owned by this
            // partition has its empty circumsphere within 2a of its centroid, so with >= 2a of
            // padding it is in the partition's triangulation exactly when it is in the global one;
            // --ps-vertex-mass also counts the kept tets around each of its vertices, which reach 4a
            if ( state.options.psLagAlphaRadius > 0. )
            {
                Real const alphaPad = Real( 4.2 * state.options.psLagAlphaRadius );
                if (lagPadding[d] < alphaPad) lagPadding[d] = alphaPad;
            }
        }
        message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " Lagrangian partition padding = "
                << MESSAGE::cMagenta() << MESSAGE::printElements( lagPadding, NO_DIM, ", " )
                << MESSAGE::cReset();
        // name the rule in the log: a default of 8 with no particle count to take a spacing from falls back to the
        // Eulerian rule, and the log must not look like k spacings then
        if ( lagSpacing > 0. ) message << " (--ps-lagrangian-padding " << state.options.psLagPadding << " mean spacings of " << lagSpacing << ")";
        else message << " (the Eulerian --padding rule" << ( state.options.psLagPadding > Real(0.) ? ": no particle count for a spacing)" : ", --ps-lagrangian-padding 0)" );
        message << "\n" << MESSAGE::Flush;

        // periodic: the period the partitions tile, per axis -- [seam - L, seam) with every periodic copy an
        // exact shift where the padding allows it (subpartition.h psPeriodicSeam). A function of
        // the bounding box, the period and the padding only, so the composite server and a --grid-window run
        // started from the recorded globals tile the same way as the batch run whose caches they reuse
        for (int d=0; d<NO_DIM; ++d)
        {
            seam[d] = lagBox[2*d+1];
            seamExact[d] = periodicCopies and psPeriodicSeam( lagBox[2*d], lagBox[2*d+1], eulerLen[d], lagPadding[d], seam[d] );
        }
        if ( periodicCopies )
            psReportSeam( message, seam, seamExact, eulerLen, lagPadding, state.particles );
    }

    // partition pi's unpadded Lagrangian region (its ownership box) and the padded one that
    // selects its particles
    void regions(int const pi, Box &lagRegion, Box &lagPadded)      // read-only (Box has no const operator[])
    {
        // unflatten the linear partition index pi into per-axis partition coordinates
        int idx[NO_DIM];
        {
            int rem = pi;
            for (int d=NO_DIM-1; d>=0; --d)
            {
                idx[d] = rem % grid[d];
                rem /= grid[d];
            }
        }

        for (int d=0; d<NO_DIM; ++d)
        {
            // periodic, exact seam: equal slices of the period [seam - L, seam), whose copies are all
            // exact shifts (subpartition.h psPeriodicSeam); the first and last edges are the period's own bits
            if ( periodicCopies and seamExact[d] )
            {
                Real const base = seam[d] - eulerLen[d];   // exact (psPeriodicSeam)
                lagRegion[2*d]   = idx[d] == 0 ? base : base + eulerLen[d] * idx[d] / grid[d];
                lagRegion[2*d+1] = idx[d] + 1 == int(grid[d]) ? seam[d] : base + eulerLen[d] * (idx[d]+1) / grid[d];
                continue;
            }
            lagRegion[2*d]   = lagBox[2*d] + lagLen[d] * idx[d] / grid[d];
            lagRegion[2*d+1] = lagBox[2*d] + lagLen[d] * (idx[d]+1) / grid[d];
            // periodic: the partition regions must tile one full PERIOD, not just the particle bounding box (image
            // tets can have Lagrangian centroids in the gap between lagBox and the periodic box, owned by NO partition:
            // their mass would be dropped). So the first lower edge sits one period below the last upper edge, which
            // is lagBox_hi itself (lagBox_lo + lagLen * grid / grid rounds, and an upper edge an ulp off lagBox_hi
            // leaves a slab of the period owned twice or never).
            if (periodicCopies && idx[d] == 0)
                lagRegion[2*d] = lagBox[2*d+1] - eulerLen[d];
            if (periodicCopies && idx[d] + 1 == int(grid[d]))
                lagRegion[2*d+1] = lagBox[2*d+1];
        }

        lagPadded = lagRegion;
        for (int d=0; d<NO_DIM; ++d)
        {
            lagPadded[2*d]   -= lagPadding[d];
            lagPadded[2*d+1] += lagPadding[d];
            // clamp to the overall Lagrangian box only when no periodic copies exist
            if (!periodicCopies)
            {
                if (lagPadded[2*d]   < lagBox[2*d])   lagPadded[2*d]   = lagBox[2*d];
                if (lagPadded[2*d+1] > lagBox[2*d+1]) lagPadded[2*d+1] = lagBox[2*d+1];
            }
        }
    }

    // the partition index along axis d of an original Lagrangian coordinate x: the slice of regions() it falls in
    // (periodic, exact seam: x wrapped into the period first -- an original above the seam is owned through its -L
    // copy). For psPartitionCosts' ETA proxy only; ownership itself is psFilterCell's.
    int axisIndex(int const d, Real const x) const
    {
        int i;
        if ( periodicCopies and seamExact[d] )
        {
            double const L = double(eulerLen[d]), base = double(seam[d]) - L;
            double f = ( double(x) - base ) / L;
            f -= std::floor( f );
            i = int( std::floor( f * double(grid[d]) ) );
        }
        else
        {
            Real const f = ( x - lagBox.coords[2*d] ) / lagLen[d];   // regions()' anchor-at-lagBox arithmetic
            i = int( std::floor( f * Real(grid[d]) ) );
        }
        return std::max( 0, std::min( int(grid[d]) - 1, i ) );
    }
};

/* Each partition's predicted COST, for the partition loop's ETA. Partitions hold about the same
   number of particles (a Lagrangian split of a lattice), so the particle count says nothing; what makes one
   partition far dearer than another is the collapsed structure in it -- folded tetrahedra inside a halo span
   the halo in Eulerian space, so the deposit visits ~(halo size)^3 cells for each of them. The proxy: each
   particle weighs the number of particles in its cell of a coarse Eulerian count grid with ~1.6 Mpc cells
   (the size of the largest haloes; 8..256 cells per axis), i.e. the sum of the local density over the
   partition's particles -- a pair count that grows with the mass in dense regions. Checked against one
   TNG50-3-Dark z=0 run of 125 partitions: log-correlation 0.86 with the measured partition times; finer cells
   and a squared weight per particle do worse. One O(N) pass. Uniform weights when the particles are not in
   memory. */
static std::vector<double> psPartitionCosts(std::vector<Particle_data> const &particles, PSPartitionPlan &plan,
                                            User_options const &options)
{
    std::vector<double> cost( size_t(plan.total), 1. );
    if ( particles.empty() or plan.total <= 1 ) return cost;
    Real lo[NO_DIM], len[NO_DIM];
    for (int d=0; d<NO_DIM; ++d)
    {
        lo[d]  = options.boxCoordinates.coords[2*d];
        len[d] = options.boxCoordinates.coords[2*d+1] - options.boxCoordinates.coords[2*d];
        if ( not ( len[d] > Real(0.) ) ) return cost;
    }
    size_t nc[NO_DIM];
    size_t cells = 1;
    for (int d=0; d<NO_DIM; ++d)
    {
        nc[d] = size_t( std::max( 8., std::min( 256., std::round( double(len[d]) / 1.6 ) ) ) );   // ~1.6 Mpc cells
        cells *= nc[d];
    }
    auto cellOf = [&](Particle_data const &p) -> size_t
    {
        size_t c = 0;
        for (int d=0; d<NO_DIM; ++d)
        {
            Real x = std::fmod( p.pos[d] - lo[d], len[d] );   // the unwrapped Eulerian position, wrapped
            if ( x < Real(0.) ) x += len[d];
            size_t i = size_t( x / len[d] * Real(nc[d]) );
            if ( i >= nc[d] ) i = nc[d] - 1;
            c = c * nc[d] + i;
        }
        return c;
    };
    std::vector<uint32_t> count( cells, 0 );
    for (auto const &p : particles) ++count[ cellOf( p ) ];
    std::fill( cost.begin(), cost.end(), 0. );
    for (auto const &p : particles)
    {
        int pi = 0;
        for (int d=0; d<NO_DIM; ++d)
        {
            int const i = plan.axisIndex( d, p.lagPos[d] );   // the slice regions() puts it in
            pi = pi * int(plan.grid[d]) + i;
        }
        cost[size_t(pi)] += double( count[ cellOf( p ) ] );
    }
    for (double &c : cost) c = std::max( c, 1. );   // an empty partition still takes a moment
    return cost;
}

#if defined(PS_GPU) && NO_DIM==3
// a resolved GPU share as the run log writes it: 1 (every partition on the GPU), else its fraction to two places
static std::string psGpuShareFraction(long const mille)
{
    if ( mille >= 1000 ) return "1";
    char b[16];
    std::snprintf( b, sizeof(b), "%.2f", double(mille) / 1000. );
    return b;
}
#endif

// the note for a number given as '--gpu-share' where it does not apply ('auto' and 'mixed' say nothing there): why,
// and where every partition deposits instead
static void psGpuShareNote(User_options const &u, char const *cause, char const *device)
{
    MESSAGE::Message message( u.verboseLevel );
    message << "note: '--gpu-share " << u.psGpuShareText << "' does not apply to " << cause
            << ": every partition deposits on " << device << ".\n" << MESSAGE::Flush;
}

/* THE GPU SHARE ('--gpu-share'; the partition loop's rule of the index, psPartitionedRun): of a partitioned GPU
   run's exact deposits, the per mille that run on the GPU, the others on the CPU at the same time. Resolved before the
   plan, since the tuner books a mixed run's partitions by it (auto_tune.cc). First match wins:
     1. where it does not apply -- a build without the 3D GPU deposit, a run on the CPU, the sampled deposit --
        1000, and a number given is told why ('auto' and 'mixed' are silent);
     2. a number given;
     3. 'auto' or 'mixed' with '--resume': the checkpoint's, from its key's 'gpu-share=' line (none: 1000), as the
        split is adopted from it (DTFE_setup); a checkpoint that cannot be read leaves it to the next rule and the
        resume's own refusal;
     4. 'auto' (the default): the run mode's (autoTuneGpuShare); 'mixed': the same model in every run mode, so that
        gentle and balanced use both devices too (every other guard of the model stays).
   One domain has no partition loop, and the plan alone tells: DTFE_setup sets 1000 after it. */
static void psResolveGpuShare(User_options &u, size_t const nParticles)
{
    bool const number = u.psGpuShareSpec >= 0;   // (only a given number: 'auto' is the default)
    u.psGpuShareMille = 1000;
#if defined(PS_GPU) && NO_DIM==3
    if ( not u.psUseMetal or not u.psExactDeposit )
    {
        if ( number )
            psGpuShareNote( u, not u.psUseMetal ? "a run on the CPU" : "the sampled deposit",
                            not u.psUseMetal ? "the CPU" : "the GPU" );
        return;
    }
    if ( number )
    {
        u.psGpuShareMille = u.psGpuShareSpec;
        u.psGpuShareSource = "--gpu-share " + u.psGpuShareText;
        return;
    }
    if ( u.psResume )
    {
        PartitionCheckpoint::Manifest m;
        std::string none;
        if ( PartitionCheckpoint::readManifest( PartitionCheckpoint::directoryFor( u ), m, none ) )
        {
            u.psGpuShareMille = PartitionCheckpoint::keyGpuShareMille( m.key );
            u.psGpuShareSource = "--resume: the checkpoint's";
            MESSAGE::Message message( u.verboseLevel );
            message << MESSAGE::cBold() << "PS-DTFE:" << MESSAGE::cReset() << " --resume: the GPU share "
                    << psGpuShareFraction( u.psGpuShareMille ) << " is the checkpoint's.\n" << MESSAGE::Flush;
            return;
        }
    }
    bool const mixed = u.psGpuShareSpec == -2;   // ('mixed': the model without its run-mode rule)
    std::string why;
    u.psGpuShareMille = autoTuneGpuShare( u, double(nParticles), why, mixed );
    u.psGpuShareSource = std::string( mixed ? "--gpu-share mixed: " : "--gpu-share auto: " ) + why;
#else
    (void)nParticles;
    if ( number ) psGpuShareNote( u, "this CPU-only build", "the CPU" );
#endif
}
#endif
