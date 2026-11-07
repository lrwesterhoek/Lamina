/* The Metal <-> CUDA twin check of the exact tetrahedron-cell clip, compiled and run by tests/gpu_twin_check.sh.

   The GPU exact deposits (--exact-average, --exact-average) clip each tetrahedron against each cell of its
   bounding-box window with a float32 z-sweep that exists twice: sweep_moments, tet_planes and box_outside_tet in
   metal/exact_clip.metal.inc (Metal) and in src/CGAL_triangulation/gpu_exact_clip.cuh (CUDA/HIP, a line-by-line
   twin; the overlap frame's origin is inline in the .metal callers and overlap_origin() there). Nothing else
   compares the two function by function: metal/validate_deposit loads only the Metal source and
   tests/cudaemu_check.sh compares whole runs. This program takes the CUDA header UNCHANGED, as plain C++
   through gpu_cuda_compat.h's CPU emulation (-DGPU_EMU: the device keywords and intrinsics of gpu_cuda_emu.h),
   and runs, per (tetrahedron, cell) pair, the steps of the deposits' exact_cell_clip: the face-plane skip, the
   overlap origin, the order-2 sweep and the order-1 sweep (pass 0 of the PS deposit). The pairs are those of
   metal/validate_deposit.cpp's tests:
     T7   the 20000-tet stress set (normal, thin/caustic and tiny tetrahedra), cells of 10/32, every cell of each
          tetrahedron's bounding-box window;
     T14  the 4000-tet stress sets on cells of 10/64 (the full one and the well-conditioned one);
     T13  the 6-tet Freudenthal cube [2,4]^3 on unit cells, an analytic anchor: the pieces of every cell it covers
          sum to volume 1, the cells around it get nothing.
   (The sets come from a verbatim copy of validate_deposit's makeStress, seed 4242. std::normal_distribution is
   implementation-defined, so a libstdc++ build -- Linux -- draws other tetrahedra of the same classes than the
   Mac's libc++ build; each platform checks its own set.)

   Every pair is compared with
     - the CPU's double-precision r3d (third_party/r3d: the reference of the CPU exact deposits), fed the SAME
       float-rounded overlap-frame coordinates the float sweep gets, so the difference is the sweep's own rounding;
     - on macOS, when built with -DTWIN_METAL (the script does that), the Metal sweep itself: a wrapper kernel
       around exact_clip.metal.inc compiled at run time with default options (fast math), as the hosts compile
       the production kernels.
   Errors are absolute, in units of the cell (volume dx^3, first moments dx^4, second moments dx^5 -- the overlap
   frame keeps a piece's coordinates cell-sized). Tolerances:
     - against r3d (CUDA-emulated, Metal): max <= 1e-4 of the cell -- validate_deposit T13's per-cell tolerance --
       and the MEAN over the set's pairs <= 1e-7, 3e-8, 1.5e-8 of the cell (volume, first, second moments; ~30x the
       measured ~3e-9, 1e-9, 5e-10), which a small systematic slip (a perturbed constant: a Gauss weight, a moment
       normalization) exceeds long before any single pair reaches 1e-4;
     - CUDA-emulated against Metal: max <= 2e-5 of the cell (T14's tolerance between two float paths of the same
       clip) and mean <= 3e-8, 1e-8, 5e-9 (the two differ by about as much as each differs from r3d);
     - order 1 against order 2 (on Metal from two pipelines, as the PS deposit's item phases are compiled): the
       volume bit-identical -- pass 0's weights and pass 1's deposit must see the same volumes -- and the first
       moments within 2e-5 of the cell (not bitwise: under Metal's fast math they differ in a few ulps);
     - the overlap origin bit-identical CUDA against Metal (min, max, add, a halving: nothing to contract);
     - the face-plane skip: a pair either backend skips holds no piece (r3d volume <= 1e-4 of the cell); the two
       backends' skip decisions may differ only on such pairs (their dot/cross products may contract differently).
   It validates the TRANSLITERATION (the same function on the same inputs), not a real NVIDIA/AMD device's numerics:
   the emulation compiles the twin with the host compiler (clang may contract a*b+c into an FMA, GCC in ISO C++
   mode does not; nvcc does). Exit status 0 = every check passed. Metal unavailable at run time: those checks SKIP. */

#if defined(TWIN_METAL)
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/CGAL_triangulation/gpu_cuda_compat.h"   // -DGPU_EMU: gpu_cuda_emu.h's device keywords and intrinsics
namespace cudaclip {   // the backends include it inside their anonymous namespace
#include "../src/CGAL_triangulation/gpu_exact_clip.cuh"
}
#include "../third_party/r3d/r3d.h"

namespace {

int g_pass = 0, g_fail = 0, g_skip = 0;
void verdict(bool ok, const std::string& name, const std::string& detail)
{
    std::printf("  %s  %s: %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    (ok ? g_pass : g_fail)++;
}
void skip(const std::string& name, const std::string& why)
{
    std::printf("  SKIP  %s: %s\n", name.c_str(), why.c_str());
    g_skip++;
}
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...)
{
    char b[1024];
    va_list ap; va_start(ap, f); std::vsnprintf(b, sizeof b, f, ap); va_end(ap);
    return b;
}

// per order (volume, first moments, second moments), in units of the cell; the means sit ~30x (against r3d) and ~9x
// (twin against twin) above the measured ones, which are about the same with clang and GCC
double const TOL_REF_MAX = 1e-4, TOL_REF_MEAN[3] = { 1e-7, 3e-8, 1.5e-8 };    // a twin against double r3d
double const TOL_TWIN_MAX = 2e-5, TOL_TWIN_MEAN[3] = { 3e-8, 1e-8, 5e-9 };    // CUDA-emulated against Metal
double const TOL_SKIP = 1e-4;                            // a skipped pair's r3d volume, of the cell

// ---- the stress sets: validate_deposit.cpp's det3 and makeStress, verbatim (seed 4242) ----------------------------
float det3(const float A[3][3]) {
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
         - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
         + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

void makeStress(uint32_t nTet,float box,float dxc,std::vector<float>& verts,std::vector<float>& vels,std::vector<float>& masses,
                bool wellConditioned=false){
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> U(0.f,box),Um(0.5f,1.5f),Uc(0.f,1.f);
    std::normal_distribution<float> Nrm(0.f,1.f),Nv(0.f,300.f);
    verts.resize(nTet*12); vels.resize(nTet*12); masses.resize(nTet);
    for(uint32_t t=0;t<nTet;++t){
        float b[3]={U(rng),U(rng),U(rng)}; float r=Uc(rng),scale,thin;
        if(wellConditioned){scale=dxc*(1.f+2.f*Uc(rng));thin=1.f;}   // no thin/tiny classes
        else if(r<0.60f){scale=dxc*(1.f+2.f*Uc(rng));thin=1.f;}
        else if(r<0.85f){scale=dxc*(1.f+3.f*Uc(rng));thin=0.02f;}
        else {scale=dxc*0.3f*Uc(rng);thin=1.f;}
        for(;;){
            for(int k=0;k<4;++k){
                verts[t*12+k*3+0]=b[0]+Nrm(rng)*scale;
                verts[t*12+k*3+1]=b[1]+Nrm(rng)*scale;
                verts[t*12+k*3+2]=b[2]+Nrm(rng)*scale*thin;
                for(int i=0;i<3;++i) vels[t*12+k*3+i]=Nv(rng);
            }
            if(!wellConditioned) break;
            // wellConditioned: reject near-coplanar draws (random Gaussian tets have a det->0
            // tail whose float conditioning would dominate any analytic plumbing check)
            float Ax[3][3];
            for(int e=0;e<3;++e)for(int i=0;i<3;++i) Ax[e][i]=verts[t*12+(e+1)*3+i]-verts[t*12+i];
            float d=det3(Ax);
            float a2=0; for(int e=0;e<3;++e){float l2=0;for(int i=0;i<3;++i)l2+=Ax[e][i]*Ax[e][i];a2+=l2;} a2/=3.f;
            if(std::fabs(d)>=0.05f*a2*std::sqrt(a2)) break;
        }
        masses[t]=Um(rng);
    }
}

// T13's Freudenthal cube [2,4]^3: 6 tets {000, path1, path2, 111} along the monotone lattice paths
std::vector<float> freudenthalCube()
{
    float const a = 2.f, b = 4.f;
    auto C = [&](int x, int y, int z, std::vector<float>& v) { v.push_back(x ? b : a); v.push_back(y ? b : a); v.push_back(z ? b : a); };
    int const path[6][2][3] = {{{1,0,0},{1,1,0}},{{1,0,0},{1,0,1}},{{0,1,0},{1,1,0}},
                               {{0,1,0},{0,1,1}},{{0,0,1},{1,0,1}},{{0,0,1},{0,1,1}}};
    std::vector<float> v;
    for (int t = 0; t < 6; ++t) {
        C(0,0,0,v);
        C(path[t][0][0],path[t][0][1],path[t][0][2],v);
        C(path[t][1][0],path[t][1][1],path[t][1][2],v);
        C(1,1,1,v);
    }
    return v;
}

// ---- the pairs ---------------------------------------------------------------------------------------------------
// One (tetrahedron, cell) pair as the deposits' exact_cell_clip sees it, in the tet's vertex-0 frame: the edges
// t1, t2, t3 (t1 and t2 swapped for a negative tet, as the kernels' setup swaps them), the cell centre and the
// half-widths. Outputs per backend: [skip, org x y z, order-2 moments (10), order-1 V and first moments (4)].
constexpr int NIN = 15, NOUT = 18;

struct Batch {
    std::vector<float> in;        // NIN per pair
    std::vector<uint32_t> tet;    // the pair's tetrahedron
    std::vector<int> raw;         // the cell's raw index (3 per pair)
    size_t size() const { return tet.size(); }
    void clear() { in.clear(); tet.clear(); raw.clear(); }
};

// every cell of the bounding-box window of tet t (absolute vertices v, cells of dx from 0) into b
void addPairs(const float* v, uint32_t t, float dx, Batch& b)
{
    float e[3][3];
    for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) e[k][i] = v[(k+1)*3+i] - v[i];
    if (det3(e) < 0.0f) for (int i = 0; i < 3; ++i) std::swap(e[0][i], e[1][i]);   // vertex 0 stays the origin
    int ic0[3], c0[3], c1[3]; float f0[3];
    for (int i = 0; i < 3; ++i) {
        float const q = v[i] / dx;
        ic0[i] = int(std::floor(q)); f0[i] = q - float(ic0[i]);
        float mn = v[i], mx = v[i];
        for (int k = 1; k < 4; ++k) { mn = std::min(mn, v[k*3+i]); mx = std::max(mx, v[k*3+i]); }
        c0[i] = int(std::floor(mn / dx)); c1[i] = int(std::floor(mx / dx));
    }
    for (int x = c0[0]; x <= c1[0]; ++x)
    for (int y = c0[1]; y <= c1[1]; ++y)
    for (int z = c0[2]; z <= c1[2]; ++z) {
        int const r[3] = { x, y, z };
        for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) b.in.push_back(e[k][i]);
        for (int i = 0; i < 3; ++i) b.in.push_back((float(r[i] - ic0[i]) - f0[i] + 0.5f) * dx);   // as the kernels' ccen
        for (int i = 0; i < 3; ++i) b.in.push_back(0.5f * dx);
        b.tet.push_back(t);
        b.raw.insert(b.raw.end(), r, r + 3);
    }
}

// ---- the emulated CUDA twin and the double r3d reference, per pair ------------------------------------------------
void cudaPair(const float* in, float* out, double* ref)
{
    using namespace cudaclip;
    V3 const t1 = v3(in[0], in[1], in[2]), t2 = v3(in[3], in[4], in[5]), t3 = v3(in[6], in[7], in[8]);
    V3 const c = v3(in[9], in[10], in[11]), h = v3(in[12], in[13], in[14]);
    TetPlanes const F = tet_planes(t1, t2, t3);
    out[0] = box_outside_tet(F, c, h) ? 1.0f : 0.0f;
    V3 const clo = sub3(c, h), chi = add3(c, h);
    V3 const org = overlap_origin(c, h, t1, t2, t3);
    out[1] = org.x; out[2] = org.y; out[3] = org.z;
    V3 const p[4] = { neg3(org), sub3(t1, org), sub3(t2, org), sub3(t3, org) };
    V3 const lo = sub3(clo, org), hi = sub3(chi, org);
    float m2[10], m1[10];
    sweep_moments(p[0], p[1], p[2], p[3], lo, hi, true, m2);
    sweep_moments(p[0], p[1], p[2], p[3], lo, hi, false, m1);
    for (int k = 0; k < 10; ++k) out[4+k] = m2[k];
    for (int k = 0; k < 4; ++k) out[14+k] = m1[k];

    // the reference: double r3d on the same float coordinates (the overlap frame), clipped by the cell's 6 planes
    // in the CPU deposits' order (x-lo, x-hi, y-lo, y-hi, z-lo, z-hi)
    r3d_rvec3 tv[4];
    for (int k = 0; k < 4; ++k) { tv[k].x = p[k].x; tv[k].y = p[k].y; tv[k].z = p[k].z; }
    double a[3][3];
    for (int k = 0; k < 3; ++k) for (int i = 0; i < 3; ++i) a[k][i] = tv[k+1].xyz[i] - tv[0].xyz[i];
    double const det = a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                     + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
    for (int k = 0; k < 10; ++k) ref[k] = 0.0;
    if (det == 0.0) return;
    if (det < 0.0) std::swap(tv[1], tv[2]);   // r3d_init_tet wants positive orientation
    r3d_poly poly;
    r3d_init_tet(&poly, tv);
    double const bl[3] = { lo.x, lo.y, lo.z }, bh[3] = { hi.x, hi.y, hi.z };
    r3d_plane pl[6];
    for (int d = 0; d < 3; ++d) {
        for (int q = 0; q < 2; ++q) for (int k = 0; k < 3; ++k) pl[2*d+q].n.xyz[k] = 0.0;
        pl[2*d].n.xyz[d] =  1.0; pl[2*d].d   = -bl[d];
        pl[2*d+1].n.xyz[d] = -1.0; pl[2*d+1].d =  bh[d];
    }
    r3d_clip(&poly, pl, 6);
    r3d_reduce(&poly, ref, 2);
}

void runCpu(const Batch& b, std::vector<float>& out, std::vector<double>& ref, unsigned nThreads)
{
    size_t const n = b.size();
    out.assign(n * NOUT, 0.0f); ref.assign(n * 10, 0.0);
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < nThreads; ++w)
        pool.emplace_back([&, w] {
            for (size_t i = w; i < n; i += nThreads) cudaPair(&b.in[i*NIN], &out[i*NOUT], &ref[i*10]);
        });
    for (auto& th : pool) th.join();
}

// ---- the Metal twin -------------------------------------------------------------------------------------------------
#if defined(TWIN_METAL)
// The wrapper: the per-pair steps of ps_deposit.metal's exact_cell_clip (its overlap origin, inline there), around
// the UNCHANGED exact_clip.metal.inc, which the program prepends as the Makefile does.
const char* const kWrapper = R"MSL(
constant bool kFull [[function_constant(0)]];   // order 2 (the PS deposit's phase 1) or order 1 (phase 0)
kernel void twinClip(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    device const float* r = in + size_t(i) * 15;
    float3 const t1 = float3(r[0], r[1], r[2]), t2 = float3(r[3], r[4], r[5]), t3 = float3(r[6], r[7], r[8]);
    float3 const ccen = float3(r[9], r[10], r[11]), hcell = float3(r[12], r[13], r[14]);
    device float* o = out + size_t(i) * 18;
    TetPlanes const F = tet_planes(t1, t2, t3);
    float3 const clo = ccen - hcell, chi = ccen + hcell;
    float3 const tlo = min(min(float3(0.0f), t1), min(t2, t3));
    float3 const thi = max(max(float3(0.0f), t1), max(t2, t3));
    float3 const org = 0.5f * (max(clo, tlo) + min(chi, thi));
    float mo[10];
    sweep_moments(-org, t1 - org, t2 - org, t3 - org, clo - org, chi - org, kFull, mo);
    if (kFull) {
        o[0] = box_outside_tet(F, ccen, hcell) ? 1.0f : 0.0f;
        o[1] = org.x; o[2] = org.y; o[3] = org.z;
        for (int k = 0; k < 10; ++k) o[4 + k] = mo[k];
    } else
        for (int k = 0; k < 4; ++k) o[14 + k] = mo[k];
}
)MSL";

struct MetalTwin {
    MTL::Device* dev = nullptr; MTL::CommandQueue* q = nullptr;
    MTL::ComputePipelineState* pso[2] = { nullptr, nullptr };   // order 1, order 2 (kFull)
    std::string why;
    bool init(const char* incPath)
    {
        dev = MTL::CreateSystemDefaultDevice();
        if (!dev) { why = "no Metal device"; return false; }
        std::ifstream f(incPath);
        if (!f.good()) { why = std::string("cannot read ") + incPath; return false; }
        std::stringstream ss; ss << f.rdbuf() << "\n" << kWrapper;
        NS::Error* err = nullptr;
        MTL::Library* lib = dev->newLibrary(NS::String::string(ss.str().c_str(), NS::UTF8StringEncoding),
                                            (MTL::CompileOptions*)nullptr, &err);
        if (!lib) { why = std::string("compile failed: ") + (err ? err->localizedDescription()->utf8String() : "?"); return false; }
        for (int full = 0; full < 2; ++full) {   // two pipelines, as the PS deposit's two item phases
            bool const fc = full != 0;
            MTL::FunctionConstantValues* cv = MTL::FunctionConstantValues::alloc()->init();
            cv->setConstantValue(&fc, MTL::DataTypeBool, NS::UInteger(0));
            MTL::Function* fn = lib->newFunction(NS::String::string("twinClip", NS::UTF8StringEncoding), cv, &err);
            cv->release();
            if (fn) pso[full] = dev->newComputePipelineState(fn, &err);
            if (!pso[full]) { why = "no pipeline"; return false; }
        }
        q = dev->newCommandQueue();
        return true;
    }
    // dispatches the batch; finish() waits and copies the outputs (the CPU twin runs in between)
    MTL::CommandBuffer* cb = nullptr; MTL::Buffer *bIn = nullptr, *bOut = nullptr; size_t n = 0;
    void start(const Batch& b)
    {
        n = b.size();
        bIn = dev->newBuffer(b.in.data(), b.in.size() * sizeof(float), MTL::ResourceStorageModeShared);
        bOut = dev->newBuffer(n * NOUT * sizeof(float), MTL::ResourceStorageModeShared);
        uint32_t const nn = uint32_t(n);
        cb = q->commandBuffer();
        MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        for (MTL::ComputePipelineState* p : pso) {
            e->setComputePipelineState(p);
            e->setBuffer(bIn, 0, 0); e->setBuffer(bOut, 0, 1); e->setBytes(&nn, sizeof nn, 2);
            NS::UInteger tg = p->maxTotalThreadsPerThreadgroup(); if (tg > 256) tg = 256;
            e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
        }
        e->endEncoding(); cb->commit();
    }
    bool finish(std::vector<float>& out)
    {
        cb->waitUntilCompleted();
        bool const ok = cb->status() != MTL::CommandBufferStatusError;
        if (!ok) why = std::string("command buffer failed: ") + (cb->error() ? cb->error()->localizedDescription()->utf8String() : "?");
        out.assign((const float*)bOut->contents(), (const float*)bOut->contents() + n * NOUT);
        bIn->release(); bOut->release();
        return ok;
    }
};
#endif

// ---- the comparisons ------------------------------------------------------------------------------------------------
// the moment k's unit: the cell's volume times dx^order
inline int order(int k) { return k == 0 ? 0 : (k < 4 ? 1 : 2); }

struct Diff {   // one comparison over a set: max and mean of the per-pair errors, per order, and the worst pair
    double mx[3] = {0, 0, 0}, sum[3] = {0, 0, 0}, worstVal = -1; size_t n = 0;
    uint32_t worstTet = 0; int worstRaw[3] = {0, 0, 0}; int worstOrder = 0;
    void add(const double* e, uint32_t tet, const int* raw)
    {
        int const o = e[0] >= e[1] ? (e[0] >= e[2] ? 0 : 2) : (e[1] >= e[2] ? 1 : 2);
        if (e[o] > worstVal) { worstVal = e[o]; worstTet = tet; std::memcpy(worstRaw, raw, sizeof worstRaw); worstOrder = o; }
        for (int k = 0; k < 3; ++k) { sum[k] += e[k]; mx[k] = std::max(mx[k], e[k]); }
        ++n;
    }
    double mean(int o) const { return n ? sum[o] / double(n) : 0.0; }
    double maxAll() const { return std::max(mx[0], std::max(mx[1], mx[2])); }
    bool ok(double tolMax, const double tolMean[3]) const
    {
        return maxAll() <= tolMax && mean(0) <= tolMean[0] && mean(1) <= tolMean[1] && mean(2) <= tolMean[2];
    }
    std::string text() const
    {
        return fmt("max |dV| %.1e, |d first| %.1e, |d second| %.1e of the cell; mean %.1e, %.1e, %.1e",
                   mx[0], mx[1], mx[2], mean(0), mean(1), mean(2));
    }
    std::string worstText() const
    {
        return fmt(" (worst: tet %u, cell %d %d %d, order %d)", worstTet, worstRaw[0], worstRaw[1], worstRaw[2], worstOrder);
    }
};

struct SetStats {
    size_t pairs = 0, pieces = 0;
    Diff cudaRef, metalRef, cudaMetal;
    size_t skipC = 0, skipM = 0, skipDisagree = 0, skipLostC = 0, skipLostM = 0;
    size_t orgDiffer = 0;
    // order 1 against order 2: pairs whose volume differs in any bit (the contract: 0), pairs whose first moments do,
    // and the largest first-moment difference in units of the cell -- per backend (0 = CUDA-emulated, 1 = Metal)
    size_t o12V[2] = {0, 0}, o12F[2] = {0, 0};
    double o12Fmax[2] = {0, 0};
};

bool bitEq(float a, float b) { uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y; }

// order-1 outputs [14..17] against the order-2 ones [4..7] of one pair
void order12(const float* o, double unit1, SetStats& S, int b)
{
    S.o12V[b] += !bitEq(o[14], o[4]);
    bool same = true;
    for (int k = 1; k < 4; ++k) {
        same = same && bitEq(o[14+k], o[4+k]);
        S.o12Fmax[b] = std::max(S.o12Fmax[b], std::fabs(double(o[14+k]) - double(o[4+k])) / unit1);
    }
    S.o12F[b] += !same;
}

void accumulate(const Batch& b, float dx, const std::vector<float>& co, const std::vector<double>& ref,
                const std::vector<float>* mo, SetStats& S)
{
    double const vc = double(dx) * dx * dx, unit[3] = { vc, vc * dx, vc * dx * dx };
    for (size_t i = 0; i < b.size(); ++i) {
        const float* c = &co[i*NOUT];
        const double* r = &ref[i*10];
        const int* raw = &b.raw[i*3];
        ++S.pairs;
        bool const piece = r[0] > TOL_SKIP * vc;
        S.pieces += r[0] > 1e-12 * vc;
        double e[3] = {0, 0, 0};
        for (int k = 0; k < 10; ++k) e[order(k)] = std::max(e[order(k)], std::fabs(double(c[4+k]) - r[k]) / unit[order(k)]);
        S.cudaRef.add(e, b.tet[i], raw);
        bool const sc = c[0] != 0.0f;
        S.skipC += sc; S.skipLostC += sc && piece;
        order12(c, unit[1], S, 0);
        if (!mo) continue;
        const float* m = &(*mo)[i*NOUT];
        double em[3] = {0, 0, 0}, ecm[3] = {0, 0, 0};
        for (int k = 0; k < 10; ++k) {
            em[order(k)]  = std::max(em[order(k)],  std::fabs(double(m[4+k]) - r[k]) / unit[order(k)]);
            ecm[order(k)] = std::max(ecm[order(k)], std::fabs(double(c[4+k]) - double(m[4+k])) / unit[order(k)]);
        }
        S.metalRef.add(em, b.tet[i], raw);
        S.cudaMetal.add(ecm, b.tet[i], raw);
        bool const sm = m[0] != 0.0f;
        S.skipM += sm; S.skipLostM += sm && piece; S.skipDisagree += sc != sm;
        S.orgDiffer += !(bitEq(c[1], m[1]) && bitEq(c[2], m[2]) && bitEq(c[3], m[3]));
        order12(m, unit[1], S, 1);
    }
}

unsigned g_threads = 1;
#if defined(TWIN_METAL)
MetalTwin g_metal;
#endif
bool g_metalOk = false;
std::string g_metalWhy = "built without Metal (Linux, or GPU_TWIN_METAL=0)";

// runs every pair of the tets in batches, through both twins and the reference
SetStats runSet(const std::vector<float>& verts, float dx)
{
    SetStats S;
    size_t const nTet = verts.size() / 12, BATCH = size_t(1) << 18;
    Batch b; std::vector<float> co, mo; std::vector<double> ref;
    auto flush = [&] {
        if (!b.size()) return;
#if defined(TWIN_METAL)
        if (g_metalOk) g_metal.start(b);
#endif
        runCpu(b, co, ref, g_threads);
        bool haveMetal = false;
#if defined(TWIN_METAL)
        if (g_metalOk) {
            haveMetal = g_metal.finish(mo);
            if (!haveMetal) { g_metalOk = false; g_metalWhy = g_metal.why; }
        }
#endif
        accumulate(b, dx, co, ref, haveMetal ? &mo : nullptr, S);
        b.clear();
    };
    for (size_t t = 0; t < nTet; ++t) {
        addPairs(&verts[t*12], uint32_t(t), dx, b);
        if (b.size() >= BATCH) flush();
    }
    flush();
    return S;
}

void report(const char* name, size_t nTet, const SetStats& S, bool metal)
{
    std::string const tag = fmt("%s (%zu tets, %zu tet-cell pairs, %zu pieces)", name, nTet, S.pairs, S.pieces);
    auto const& d = S.cudaRef;
    std::string const tolRef = fmt(" (tolerance max %.0e, mean %.0e, %.0e, %.1e)",
                                   TOL_REF_MAX, TOL_REF_MEAN[0], TOL_REF_MEAN[1], TOL_REF_MEAN[2]);
    verdict(d.ok(TOL_REF_MAX, TOL_REF_MEAN), tag + ", CUDA-emulated vs double r3d", d.text() + tolRef + d.worstText());
    verdict(S.o12V[0] == 0 && S.o12Fmax[0] <= TOL_TWIN_MAX && S.skipLostC == 0,
            tag + ", CUDA-emulated order 1 and face-plane skip",
            fmt("order-1 volume bit-identical to order 2 in all but %zu pairs (expect 0), first moments in all but %zu "
                "(max diff %.1e of the cell, tolerance %.0e); %zu pairs skipped, %zu of them holding a piece (expect 0)",
                S.o12V[0], S.o12F[0], S.o12Fmax[0], TOL_TWIN_MAX, S.skipC, S.skipLostC));
    if (!metal) { skip(tag + ", Metal twin", g_metalWhy); return; }
    auto const& m = S.metalRef;
    verdict(m.ok(TOL_REF_MAX, TOL_REF_MEAN), tag + ", Metal vs double r3d", m.text() + tolRef + m.worstText());
    auto const& cm = S.cudaMetal;
    verdict(cm.ok(TOL_TWIN_MAX, TOL_TWIN_MEAN), tag + ", CUDA-emulated vs Metal",
            cm.text() + fmt(" (tolerance max %.0e, mean %.0e, %.0e, %.0e)", TOL_TWIN_MAX, TOL_TWIN_MEAN[0], TOL_TWIN_MEAN[1],
                            TOL_TWIN_MEAN[2]) + cm.worstText());
    verdict(S.o12V[1] == 0 && S.o12Fmax[1] <= TOL_TWIN_MAX && S.skipLostM == 0 && S.orgDiffer == 0,
            tag + ", Metal order 1, skip and overlap origin",
            fmt("order-1 volume (its own pipeline, as the item phases) bit-identical to order 2 in all but %zu pairs, "
                "first moments in all but %zu (max diff %.1e of the cell); %zu pairs skipped (%zu holding a piece; %zu "
                "decided otherwise by the CUDA twin); overlap origin differs from the CUDA twin's in %zu pairs",
                S.o12V[1], S.o12F[1], S.o12Fmax[1], S.skipM, S.skipLostM, S.skipDisagree, S.orgDiffer));
}

// T13: the cube's pieces per cell, summed over its 6 tets, per backend: 1 in each of the 8 cells it covers, 0 around
void cubeAnchor()
{
    std::vector<float> const v = freudenthalCube();
    Batch b;
    for (uint32_t t = 0; t < 6; ++t) addPairs(&v[t*12], t, 1.0f, b);
    std::vector<float> co, mo; std::vector<double> ref;
#if defined(TWIN_METAL)
    if (g_metalOk) g_metal.start(b);
#endif
    runCpu(b, co, ref, g_threads);
    bool haveMetal = false;
#if defined(TWIN_METAL)
    if (g_metalOk) { haveMetal = g_metal.finish(mo); if (!haveMetal) { g_metalOk = false; g_metalWhy = g_metal.why; } }
#endif
    auto check = [&](const char* who, auto vol) {
        double cell[3][3][3] = {};   // cells 2..4 per axis (the window's last row touches the cube's face only)
        for (size_t i = 0; i < b.size(); ++i) {
            const int* r = &b.raw[i*3];
            cell[r[0]-2][r[1]-2][r[2]-2] += vol(i);
        }
        double inErr = 0, outside = 0, total = 0;
        for (int x = 0; x < 3; ++x) for (int y = 0; y < 3; ++y) for (int z = 0; z < 3; ++z) {
            total += cell[x][y][z];
            if (x < 2 && y < 2 && z < 2) inErr = std::max(inErr, std::fabs(cell[x][y][z] - 1.0));
            else outside += std::fabs(cell[x][y][z]);
        }
        verdict(inErr < 1e-4 && outside < 1e-5 && std::fabs(total - 8.0) < 1e-4,
                fmt("T13 analytic cube (6 tets, %zu pairs), %s", b.size(), who),
                fmt("cell volume err %.1e (tolerance 1e-4), outside %.1e (1e-5), total %.6f (expect 8)", inErr, outside, total));
    };
    check("CUDA-emulated", [&](size_t i) { return double(co[i*NOUT+4]); });
    check("double r3d", [&](size_t i) { return ref[i*10]; });
    if (haveMetal) check("Metal", [&](size_t i) { return double(mo[i*NOUT+4]); });
    else skip("T13 analytic cube, Metal", g_metalWhy);
}

}  // namespace

int main(int argc, char** argv)
{
    // argv[1]: metal/exact_clip.metal.inc (the Metal twin's source; default relative to the repository root)
    const char* const inc = argc > 1 ? argv[1] : "metal/exact_clip.metal.inc";
    g_threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
#if defined(TWIN_METAL)
    g_metalOk = g_metal.init(inc);
    if (!g_metalOk) g_metalWhy = g_metal.why;
#else
    (void)inc;
#endif
    std::printf("exact-clip twins: gpu_exact_clip.cuh (%s), %s; reference: double r3d; %u threads\n",
                GPU_BACKEND_STRING, g_metalOk ? "metal/exact_clip.metal.inc (Metal, run-time compiled)" : "no Metal twin",
                g_threads);

    cubeAnchor();
    struct Set { const char* name; uint32_t nTet; float dx; bool wellConditioned; };
    for (Set const& set : { Set{"T7 stress set", 20000, 10.f/32.f, false}, Set{"T14 stress set", 4000, 10.f/64.f, false},
                            Set{"T14 well-conditioned set", 4000, 10.f/64.f, true} }) {
        std::vector<float> v, u, m; makeStress(set.nTet, 10.f, set.dx, v, u, m, set.wellConditioned);
        SetStats const S = runSet(v, set.dx);   // first: a failed Metal batch turns the Metal checks into a SKIP
        report(set.name, set.nTet, S, g_metalOk);
    }
    std::printf("SUMMARY: %d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
    return g_fail ? 1 : 0;
}
