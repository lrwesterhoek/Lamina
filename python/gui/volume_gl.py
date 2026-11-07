"""The 3D view's volume drawn on the GPU (OpenGL 4.1 core, macOS's own): the same picture as explore_3d's QPainter
slice stack, at the speed of the graphics chip -- a 1024^3 cube turns in real time instead of 2 s a settled picture.

How: the cube's levels (volume3d.prepare's 'q', one byte a cell) are one 3D integer texture, the transfer function's
256 RGBA64 entries (volume3d.transfer) a 256 x 1 RGBA16 texture, and a fragment shader casts each pixel's ray
through the cube. It is not a free ray-marcher: it walks the very slices QPainter draws -- the planes through the
cells' centres normal to slice_plan's axis -- and at each it takes the sample QPainter's bilinear drawImage takes
there: the four nearest cells' colours (looked up first, then mixed: the slice images are coloured before they are
resampled) mixed by the ray's place between them, composited front to back with the 'under' operator. So it is the
QPainter picture computed in 32-bit floats instead of 16-bit integers per slice, the same opacity correction (the
reference ray's, the same table) and the same order. While the view turns, a frame may take every stride-th slice
only, the opacity corrected for that longer path (the table's exponent x stride: the same look), and of the cube's
half (2^3 cells averaged in value space, quantised on the same range: mip_layer) rather than of the cube itself:
skipping slices of the cube aliases (a dense knot is hit or missed), the half averages it, and at 1024^3 every slice
of the half (17 ms) is cheaper than every 8th of the cube (23 ms). Blocks of BLOCK^3 cells whose every level is
transparent are skipped when striding (exact: the same pixels).

Drawn offscreen (a QOffscreenSurface and a framebuffer object) and read back into a QImage that View3D draws as it
draws QPainter's picture: the window stays a plain raster widget (a QOpenGLWidget would switch the whole launcher
window to OpenGL composition) and every overlay is drawn as before. No OpenGL under QT_QPA_PLATFORM=offscreen (the
tests), none on a machine without it: renderer() is None (unavailable_reason says why) and View3D keeps QPainter;
Save image and the tests always use QPainter (View3D.volume_picture)."""

from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtGui import QGuiApplication, QImage, QOffscreenSurface, QOpenGLContext, QSurfaceFormat

import volume3d as VOL

# GL enums (no PyOpenGL: Qt's own QOpenGLFunctions take numpy buffers)
GL_TEXTURE_2D, GL_TEXTURE_3D = 0x0DE1, 0x806F
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST = 0x2801, 0x2800, 0x2600
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE = 0x2802, 0x2803, 0x8072, 0x812F
GL_UNPACK_ALIGNMENT, GL_PACK_ALIGNMENT = 0x0CF5, 0x0D05
GL_RED_INTEGER, GL_RGBA, GL_BGRA = 0x8D94, 0x1908, 0x80E1
GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT = 0x1401, 0x1403
GL_R8UI, GL_RG8UI, GL_RGBA16, GL_RGBA8 = 0x8232, 0x8238, 0x805B, 0x8058
GL_RG_INTEGER = 0x8228
GL_TRIANGLES, GL_TEXTURE0, GL_NO_ERROR = 0x0004, 0x84C0, 0
GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_COMPLETE = 0x8D40, 0x8CE0, 0x8CD5
GL_RENDERBUFFER = 0x8D41
GL_MAX_3D_TEXTURE_SIZE = 0x8073

BLOCK = 4                       # empty-space skipping: blocks of BLOCK^3 cells (a 1024^3 cube: 256^3 blocks, 32 MB)
SKIP_FROM_STRIDE = 4            # ... used from this stride on: measured on TNG100-3-Dark 1024^3 (2000 x 1600, the
                                # same pixels either way), stride 4 34 ms against 57 without, stride 8 15 against 22
                                # (1000 x 800); at stride 1 the block test, a fetch each sample waits on, makes it
                                # 2.2x slower (107 -> 240 ms): consecutive samples there hit the cache anyway
MIP_FROM = 64                   # a cube of at least this many cells a side also gets its half (2^3 cells averaged) for
                                # the turning frames: TNG100-3-Dark 1024^3 at 2000 x 1600 every slice of the half cube
                                # 17 ms, against 23 ms for every 8th slice of the cube itself (which aliases: a dense
                                # knot is hit or missed, never averaged)
UPLOAD_SLAB_BYTES = 24 << 20    # the cube goes up in slabs of whole layers of about this much (~18 ms each)
DONE_ALPHA = 1.0 - 1.0 / 1024   # a ray ends once this opaque: what is left behind adds under 1/1024 (half an 8-bit
                                # step); the parity image (16 bits) goes on to 1 - 2^-17

VERTEX = """#version 410 core
void main() {                   // one triangle covering the picture
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(2.0 * p - 1.0, 0.0, 1.0);
}
"""

FRAGMENT = """#version 410 core
uniform usampler3D vol;         // the levels; texel (s, t, r) = cell (k2, k1, k0)
uniform sampler2D lut;          // 256 x 1: each level's premultiplied RGBA (volume3d.transfer)
uniform ivec3 shape;            // cells along x, y, z (the cube's axes 0, 1, 2)
uniform int axis, ra, rb;       // the slicing axis and the two others
uniform ivec3 eA, eRa, eRb;     // those axes' unit steps in texel order
uniform vec3 q0, qx, qy;        // the target plane's point under picture point (x, y): q0 + x qx + y qy (cell units)
uniform vec3 eye, dir;          // perspective: the eye (cell units); orthographic: the rays' direction
uniform int persp;
uniform vec2 pxs;               // the view's pixels per picture pixel
uniform int stride, phase;      // every stride-th slice, those with k = phase (mod stride)
uniform uint clear0, clear1;    // levels clear0..clear1 (and level 0) are transparent: their colours not looked up
uniform float done;             // a ray ends once this opaque
uniform usampler3D occ;         // per block of B^3 cells (texel order as vol): its levels' max and smallest non-zero,
uniform int B;                  // over the block and one cell around it (all a sample in the block can blend)
uniform ivec3 nb;               // blocks along x, y, z
uniform int skip;               // 0: every slice sampled (no block test)

out vec4 frag;

bool clearLevel(uint l) { return l == 0u || (l - clear0) <= (clear1 - clear0); }

void main() {
    vec2 xy = gl_FragCoord.xy * pxs;            // GL's row 0 is the picture's top row: read back in order
    vec3 Q = q0 + xy.x * qx + xy.y * qy;
    vec3 O = persp != 0 ? eye : Q;
    vec3 D = persp != 0 ? Q - eye : dir;
    float t0 = persp != 0 ? 0.0 : -3.0e38, t1 = 3.0e38;
    for (int b = 0; b < 3; ++b) {               // the ray's span inside the cube
        float m = float(shape[b]);
        if (abs(D[b]) < 1e-30) {
            if (O[b] < 0.0 || O[b] > m) { frag = vec4(0.0); return; }
        } else {
            float u = -O[b] / D[b], v = (m - O[b]) / D[b];
            t0 = max(t0, min(u, v));
            t1 = min(t1, max(u, v));
        }
    }
    float Da = D[axis];
    if (!(t0 <= t1) || abs(Da) < 1e-30) { frag = vec4(0.0); return; }
    int ma = shape[axis], mra = shape[ra], mrb = shape[rb];
    float Oa = O[axis], ca = Oa + t0 * Da, cb = Oa + t1 * Da;
    int dk = Da > 0.0 ? 1 : -1;
    int k0 = dk > 0 ? int(ceil(ca - 0.5)) : int(floor(ca - 0.5));     // the first and last slice the ray meets
    int k1 = dk > 0 ? int(floor(cb - 0.5)) : int(ceil(cb - 0.5));
    k0 = clamp(k0, 0, ma - 1);
    k1 = clamp(k1, 0, ma - 1);
    if (stride > 1) {                           // onto the slices kept: k = phase (mod stride)
        int r = ((k0 - phase) % stride + stride) % stride;
        k0 += dk > 0 ? (r == 0 ? 0 : stride - r) : -r;
    }
    float Ora = O[ra], Orb = O[rb], Dra = D[ra], Drb = D[rb], sa = Dra / Da, sb = Drb / Da;
    float fra = float(mra), frb = float(mrb), fB = float(B);
    int nbRa = nb[ra], nbRb = nb[rb];
    vec4 acc = vec4(0.0);
    int kCheck = k0;                            // the slice at which the ray enters a block not yet looked at
    for (int k = k0; dk * (k1 - k) >= 0; k += dk * stride) {
        float kk = float(k) + 0.5 - Oa;
        float u = Ora + kk * sa, v = Orb + kk * sb;          // where the ray crosses slice k
        if (skip != 0 && dk * (k - kCheck) >= 0) {   // a new block: wholly transparent (skipped exactly) or not?
            int bA = k / B;
            int bRa = clamp(int(floor(u / fB)), 0, nbRa - 1), bRb = clamp(int(floor(v / fB)), 0, nbRb - 1);
            uvec2 mm = texelFetch(occ, bA * eA + bRa * eRa + bRb * eRb, 0).rg;
            // where the ray leaves the block, as its coordinate along the slicing axis
            float tx = (float(Da > 0.0 ? min((bA + 1) * B, ma) : bA * B) - Oa) / Da;
            if (abs(Dra) > 1e-30) tx = min(tx, (float(Dra > 0.0 ? min((bRa + 1) * B, mra) : bRa * B) - Ora) / Dra);
            if (abs(Drb) > 1e-30) tx = min(tx, (float(Drb > 0.0 ? min((bRb + 1) * B, mrb) : bRb * B) - Orb) / Drb);
            float cx = Oa + tx * Da;            // the first slice at or past that point (never this one again)
            int kn = dk > 0 ? max(int(ceil(cx - 0.5 - 1e-3)), k + 1) : min(int(floor(cx - 0.5 + 1e-3)), k - 1);
            if (stride > 1) {
                int r = ((kn - phase) % stride + stride) % stride;
                kn += dk > 0 ? (r == 0 ? 0 : stride - r) : -r;
            }
            if (mm.r == 0u || (mm.g >= clear0 && mm.r <= clear1)) {   // every level it can blend is transparent
                k = kn - dk * stride;
                continue;
            }
            kCheck = kn;
        }
        if (u < 0.0 || u > fra || v < 0.0 || v > frb) continue;
        float fu = u - 0.5, fv = v - 0.5;                    // texel centres at i + 1/2, clamped at the edges
        float gu = floor(fu), gv = floor(fv);
        float wu = fu - gu, wv = fv - gv;
        int iu = int(gu), iv = int(gv);
        int u0 = clamp(iu, 0, mra - 1), u1 = clamp(iu + 1, 0, mra - 1);
        int v0 = clamp(iv, 0, mrb - 1), v1 = clamp(iv + 1, 0, mrb - 1);
        ivec3 base = k * eA;
        uint l00 = texelFetch(vol, base + u0 * eRa + v0 * eRb, 0).r;
        uint l01 = texelFetch(vol, base + u0 * eRa + v1 * eRb, 0).r;
        uint l10 = texelFetch(vol, base + u1 * eRa + v0 * eRb, 0).r;
        uint l11 = texelFetch(vol, base + u1 * eRa + v1 * eRb, 0).r;
        if (clearLevel(l00) && clearLevel(l01) && clearLevel(l10) && clearLevel(l11)) continue;
        vec4 c00 = texelFetch(lut, ivec2(int(l00), 0), 0), c01 = texelFetch(lut, ivec2(int(l01), 0), 0);
        vec4 c10 = texelFetch(lut, ivec2(int(l10), 0), 0), c11 = texelFetch(lut, ivec2(int(l11), 0), 0);
        vec4 s = mix(mix(c00, c01, wv), mix(c10, c11, wv), wu);
        acc += (1.0 - acc.a) * s;
        if (acc.a >= done) break;
    }
    frag = acc;
}
"""


def _block_reduce(x: tuple, axis: int, B: int) -> tuple[np.ndarray, np.ndarray]:
    """x = (max, min) arrays -> their (max, min) over blocks of B along 'axis', each block widened by one element at
    either end (all a bilinear sample in the block reads): B + 2 strided maxima in the arrays' own layout (numpy's
    inner loop stays on the contiguous last axis; a reduction over a short axis would not)."""
    mx, mn = x
    m = mx.shape[axis]
    nb = -(-m // B)

    def take(a, sl):
        idx = [slice(None)] * a.ndim
        idx[axis] = sl
        return a[tuple(idx)]
    bmx, bmn = take(mx, slice(0, None, B)).copy(), take(mn, slice(0, None, B)).copy()
    for j in range(1, B):                       # block i's cell j (the last block may be short of some)
        n = len(range(j, m, B))
        np.maximum(take(bmx, slice(0, n)), take(mx, slice(j, None, B)), out=take(bmx, slice(0, n)))
        np.minimum(take(bmn, slice(0, n)), take(mn, slice(j, None, B)), out=take(bmn, slice(0, n)))
    if nb > 1:                                  # the cell before each block (from block 1) and the one after it
        before = take(mx, slice(B - 1, nb * B - 1, B)), take(mn, slice(B - 1, nb * B - 1, B))
        after = take(mx, slice(B, None, B)), take(mn, slice(B, None, B))
        np.maximum(take(bmx, slice(1, None)), before[0], out=take(bmx, slice(1, None)))
        np.minimum(take(bmn, slice(1, None)), before[1], out=take(bmn, slice(1, None)))
        na = after[0].shape[axis]
        np.maximum(take(bmx, slice(0, na)), after[0], out=take(bmx, slice(0, na)))
        np.minimum(take(bmn, slice(0, na)), after[1], out=take(bmn, slice(0, na)))
    return bmx, bmn


def occupancy_rows(q: np.ndarray, B: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """For layers q[i0:i1] ((r, m1, m2) uint8): each layer's (max, smallest non-zero level) over blocks of B along its
    two axes, widened by a cell: ((r, nb1, nb2), (r, nb1, nb2)) uint8 (the min 255 where a block holds only 0s)."""
    mn = q - np.uint8(1)                        # 0 -> 255 (wraps): the min below is then the smallest non-zero - 1
    x = _block_reduce((q, mn), 1, B)            # along whole rows first (contiguous), then the shorter last axis
    bmx, bmn = _block_reduce(x, 2, B)
    return np.ascontiguousarray(bmx), np.ascontiguousarray(bmn)


def occupancy(rows_max: np.ndarray, rows_min: np.ndarray, B: int = BLOCK) -> np.ndarray:
    """occupancy_rows of every layer -> per block of B^3 cells (widened by a cell along every axis) its levels' max
    and smallest non-zero: (nb0, nb1, nb2, 2) uint8, the shader's 'occ'."""
    bmx, bmn = _block_reduce((rows_max, rows_min), 0, B)
    return np.ascontiguousarray(np.stack([bmx, bmn + np.uint8(1)], axis=-1))    # all 0s: 255 + 1 = 0 (max 0 too)


def value_table(levels: np.ndarray, log: bool) -> np.ndarray:
    """Each level's value (volume3d.level_values on the map's scale; a log scale: 10^level), level 0 (no value) NaN."""
    v = np.asarray(levels, dtype=np.float64)
    with np.errstate(over="ignore"):
        vt = np.power(10.0, v) if log else v.copy()
    vt[0] = np.nan
    return vt


def mip_shape(shape) -> tuple:
    return tuple((int(m) + 1) // 2 for m in shape)


def mip_layer(q: np.ndarray, vt: np.ndarray, qrange, log: bool, j: int) -> np.ndarray:
    """Layer j of the cube's half: the mean value over blocks of 2^3 cells (volume3d.block_edges: as equal as whole
    cells allow), taken in value space as volume3d averages the grid (a log scale's levels are 10^level first: a
    dense cell among empty ones keeps its mass), quantised again on the cube's own range (the same transfer table
    colours both); a block holding a cell without a value has none (NaN, level 0), as in the averaging."""
    e = [VOL.block_edges(m, (m + 1) // 2) for m in q.shape]
    blk = vt[q[e[0][j]:e[0][j + 1]]]
    blk = np.add.reduceat(blk, e[2][:-1], axis=2)
    blk = np.add.reduceat(blk, e[1][:-1], axis=1).sum(axis=0)
    n = (e[0][j + 1] - e[0][j]) * np.diff(e[1])[:, None] * np.diff(e[2])[None, :]
    mean = blk / n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log10(mean) if log else mean
    return VOL.quantize(s, qrange)[0]


def mip_levels(q: np.ndarray, levels: np.ndarray, qrange, log: bool) -> np.ndarray:
    """The whole half cube (mip_layer of every layer), at once: the reference the GL upload's threads follow."""
    vt = value_table(levels, log)
    return np.stack([mip_layer(q, vt, qrange, log, j) for j in range(mip_shape(q.shape)[0])])


class GLUnavailable(RuntimeError):
    """No OpenGL here (the offscreen platform, no 4.1 context, a shader the driver refused)."""


_STATE: dict = {}               # 'renderer' (Renderer) or 'why' (str: why there is none), once tried


def disabled_reason() -> str:
    """'' when OpenGL may be tried, else why not (the offscreen platform, DTFE_GUI_NO_GL set, no GUI app)."""
    if os.environ.get("DTFE_GUI_NO_GL", "").strip() not in ("", "0"):
        return "DTFE_GUI_NO_GL is set"
    app = QGuiApplication.instance()
    if app is None:
        return "no Qt application"
    plat = QGuiApplication.platformName()
    if plat in ("offscreen", "minimal", "vnc", "linuxfb", ""):
        return f"the '{plat}' Qt platform has no OpenGL"
    return ""


def renderer() -> "Renderer | None":
    """The one GL renderer (created on first use, on the GUI thread), or None when there is no OpenGL here."""
    if "renderer" in _STATE:
        return _STATE["renderer"]
    if "why" in _STATE:
        return None
    why = disabled_reason()
    if not why:
        try:
            _STATE["renderer"] = Renderer()
            QGuiApplication.instance().aboutToQuit.connect(_at_quit)
            return _STATE["renderer"]
        except GLUnavailable as e:
            why = str(e)
        except Exception as e:                  # noqa: BLE001 -- anything else (a binding, a driver): no GPU here
            why = f"{type(e).__name__}: {e}"
    _STATE["why"] = why
    return None


def current_renderer() -> "Renderer | None":
    """The GL renderer if one was made, without making one (renderer() makes it on first use): what holds the GPU's
    copy of a volume, for freeing it or counting its bytes."""
    return _STATE.get("renderer")


def _at_quit() -> None:
    """The application is closing: the GPU's textures freed while their context is still there (Qt warns of every
    one destroyed later, at the interpreter's exit, without it)."""
    r = _STATE.pop("renderer", None)
    _STATE["why"] = "the application is closing"
    if r is not None:
        try:
            r.release()
            r.ctx.doneCurrent()
        except Exception:                       # noqa: BLE001 -- going anyway
            pass


def unavailable_reason() -> str:
    return _STATE.get("why", "")


def give_up(why: str) -> None:
    """The renderer failed (an error mid-way): QPainter from now on, its resources freed."""
    r = _STATE.pop("renderer", None)
    _STATE["why"] = why
    if r is not None:
        try:
            r.release()
        except Exception:                       # noqa: BLE001 -- going anyway
            pass


def _lut_rgba16(lut: np.ndarray) -> np.ndarray:
    """volume3d.transfer's table (256 uint64: R, G, B, A halfwords in memory order) as (256, 4) uint16."""
    return np.ascontiguousarray(np.asarray(lut, dtype=np.uint64)).view(np.uint16).reshape(-1, 4)


def clear_levels(lut: np.ndarray) -> tuple[int, int]:
    """The longest run of levels (1..255) whose colour is fully transparent, as (first, last); the
    shader skips a sample whose four levels all lie in it (or are level 0) without looking their colours up. None:
    (256, 256), which no level matches (the shader's unsigned range test)."""
    a = _lut_rgba16(lut)[:, 3] == 0
    best, start = (256, 256), None
    for i in range(1, len(a) + 1):
        if i < len(a) and a[i]:
            start = i if start is None else start
        elif start is not None:
            if best == (256, 256) or i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


def geometry(cam, lo, hi, shape) -> dict:
    """The camera in the cube's cell units (cell i spans i..i + 1 along each axis) for the shader: the target plane's
    point under view point (x, y) = q0 + x qx + y qy; perspective: rays from the eye through it, orthographic: rays
    along -toward() through it (view3d.Camera.matrix's projection either way)."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = (hi - lo) / np.asarray(shape, dtype=np.float64)
    R, s, t = cam.R, float(cam.scale), np.asarray(cam.target, dtype=np.float64)
    W, H = 0.5 * cam.width, 0.5 * cam.height
    g = {"q0": (t - W * R[0] / s + H * R[1] / s - lo) / d, "qx": R[0] / (s * d), "qy": -R[1] / (s * d),
         "persp": bool(cam.perspective)}
    if cam.perspective:
        g["eye"], g["dir"] = (cam.eye() - lo) / d, np.zeros(3)
    else:
        g["eye"], g["dir"] = np.zeros(3), -R[2] / d
    return g


class Renderer:
    """The GL side: one offscreen context, the program, the cube's texture (uploaded in slabs: upload_step) and the
    table's, framebuffers sized to the picture. Every call on the GUI thread."""

    def __init__(self):
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.ctx = QOpenGLContext()
        self.ctx.setFormat(fmt)
        if not self.ctx.create():
            raise GLUnavailable("no OpenGL context could be created")
        f = self.ctx.format()
        if (f.majorVersion(), f.minorVersion()) < (4, 1):
            raise GLUnavailable(f"OpenGL {f.majorVersion()}.{f.minorVersion()} (4.1 needed)")
        self.surface = QOffscreenSurface()
        self.surface.setFormat(f)
        self.surface.create()
        if not self.surface.isValid() or not self.ctx.makeCurrent(self.surface):
            raise GLUnavailable("the OpenGL context could not be made current")
        self.f = self.ctx.functions()
        self.ef = self.ctx.extraFunctions()
        from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject
        self.prog = QOpenGLShaderProgram()
        if not (self.prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX)
                and self.prog.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAGMENT) and self.prog.link()):
            raise GLUnavailable(f"the volume shader did not build: {self.prog.log().strip()[:300]}")
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.loc = {n: self.prog.uniformLocation(n) for n in (
            "vol", "lut", "shape", "axis", "ra", "rb", "eA", "eRa", "eRb", "q0", "qx", "qy", "eye", "dir", "persp",
            "pxs", "stride", "phase", "clear0", "clear1", "done", "occ", "B", "nb", "skip")}
        v = np.zeros(1, np.int32)
        self.f.glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, v)
        self.max3d = int(v[0])
        self.renderer_name = str(self.f.glGetString(0x1F01))
        self.lut_tex = self._texture()
        self.levels: list = []                  # the cube (0) and its half (1): _Level each, while held
        self.cube_key = None
        self.mip_jobs = None                    # (array, futures) while the half is averaged
        self.lut_key = None
        self.fbo = {}                           # (w, h, internal format) -> (fbo id, renderbuffer id)
        self.last_ms = 0.0
        self.check("setting up")

    # ------------------------------------------------------------------ GL plumbing
    def current(self):
        if not self.ctx.makeCurrent(self.surface):
            raise GLUnavailable("the OpenGL context could not be made current")

    def check(self, what: str):
        err = self.f.glGetError()
        if err != GL_NO_ERROR:
            raise GLUnavailable(f"OpenGL error 0x{err:04x} {what}")

    def _texture(self) -> int:
        ids = np.zeros(1, np.uint32)
        self.f.glGenTextures(1, ids)
        return int(ids[0])

    def _params(self, target: int):
        for p, val in ((GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                       (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
            self.f.glTexParameteri(target, p, val)
        if target == GL_TEXTURE_3D:
            self.f.glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def release(self):
        """Free the textures and framebuffers (the context stays: a later volume makes new ones)."""
        self.current()
        self.forget_cube()
        self.f.glDeleteTextures(1, np.array([self.lut_tex], np.uint32))
        for fb, rb in self.fbo.values():
            self.ef.glDeleteFramebuffers(1, np.array([fb], np.uint32))
            self.ef.glDeleteRenderbuffers(1, np.array([rb], np.uint32))
        self.fbo = {}
        self.lut_tex = self._texture()
        self.lut_key = None

    # ------------------------------------------------------------------ the cube
    def fits(self, shape) -> bool:
        return max(int(s) for s in shape) <= self.max3d

    def ready(self, key, level: int = 0) -> bool:
        """The cube 'key' (level 1: its half) is wholly on the GPU."""
        return self.cube_key == key and len(self.levels) > level and self.levels[level].done

    def shape(self, level: int = 0) -> tuple:
        return self.levels[level].shape

    def held_bytes(self) -> int:
        """The GPU memory the cube's levels hold (a byte a cell, the blocks a little more)."""
        return sum(int(np.prod(lv.shape)) for lv in self.levels)

    def begin_upload(self, key, q: np.ndarray, levels: np.ndarray | None = None, qrange=None, log: bool = False):
        """Start putting the levels q ((m0, m1, m2) uint8, C order) on the GPU: storage allocated (empty), the layers
        sent by upload_step; with the cube's level values (volume3d.prepare's 'levels', 'qrange', 'log') and at
        least MIP_FROM cells a side its half too, averaged on the GUI's volume threads meanwhile."""
        if self.cube_key == key:
            return
        self.current()
        self.forget_cube()
        q = np.ascontiguousarray(q)
        self.cube_key = key
        self.levels = [_Level(self, q)]
        if levels is not None and min(q.shape) >= MIP_FROM:
            vt = value_table(levels, log)
            half = np.empty(mip_shape(q.shape), np.uint8)

            def one(j):
                half[j] = mip_layer(q, vt, qrange, log, j)
            self.mip_jobs = (half, [VOL.gui_pool().submit(one, j) for j in range(half.shape[0])])

    def upload_step(self, budget_s: float = 0.03) -> bool:
        """Send more layers for about budget_s (all, waiting for the threads: inf): the cube's, then its half's once
        averaged. True when everything is on the GPU (ready(key) may be true before: the cube first)."""
        if not self.levels:
            return True
        self.current()
        end = time.perf_counter() + budget_s
        wait = budget_s == float("inf")
        if not self.levels[0].step(end, wait):
            return False
        if self.mip_jobs is not None and (wait or all(f.done() for f in self.mip_jobs[1])):
            half, futures = self.mip_jobs
            for f in futures:
                f.result()
            self.mip_jobs = None
            self.levels.append(_Level(self, half))
        if self.mip_jobs is not None:
            return False
        return all(lv.step(end, wait) for lv in self.levels[1:])

    def forget_cube(self):
        """The cube's textures freed (the view shows another, or none)."""
        if self.mip_jobs is not None:
            for f in self.mip_jobs[1]:
                f.cancel()
        self.mip_jobs = None
        if self.levels:
            self.current()
        for lv in self.levels:
            lv.free()
        self.levels = []
        self.cube_key = None

    def set_lut(self, lut: np.ndarray):
        key = np.asarray(lut).tobytes()
        if key == self.lut_key:
            return
        self.f.glBindTexture(GL_TEXTURE_2D, self.lut_tex)
        self._params(GL_TEXTURE_2D)
        self.f.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self.f.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, 256, 1, 0, GL_RGBA, GL_UNSIGNED_SHORT, _lut_rgba16(lut))
        self.check("sending the transfer function")
        self.lut_key = key

    # ------------------------------------------------------------------ drawing
    def _framebuffer(self, w: int, h: int, internal: int) -> int:
        k = (w, h, internal)
        if k not in self.fbo:
            if len(self.fbo) >= 3:                # a resized view: the old sizes freed
                for fb, rb in self.fbo.values():
                    self.ef.glDeleteFramebuffers(1, np.array([fb], np.uint32))
                    self.ef.glDeleteRenderbuffers(1, np.array([rb], np.uint32))
                self.fbo = {}
            ids = np.zeros(1, np.uint32)
            self.ef.glGenRenderbuffers(1, ids)
            rb = int(ids[0])
            self.ef.glBindRenderbuffer(GL_RENDERBUFFER, rb)
            self.ef.glRenderbufferStorage(GL_RENDERBUFFER, internal, w, h)
            self.ef.glGenFramebuffers(1, ids)
            fb = int(ids[0])
            self.ef.glBindFramebuffer(GL_FRAMEBUFFER, fb)
            self.ef.glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb)
            if self.ef.glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise GLUnavailable(f"a {w} x {h} framebuffer could not be made")
            self.fbo[k] = (fb, rb)
        return self.fbo[k][0]

    def render(self, cam, lo, hi, plan: dict, lut: np.ndarray, width: int, height: int, stride: int = 1,
               deep: bool = False, level: int = 0) -> np.ndarray:
        """The volume's picture of the cube on the GPU (ready()) as 'cam' sees it: width x height pixels over the
        camera's view (cam.width x cam.height), slices along plan['axis'] (volume3d.slice_plan of the cube's shape),
        coloured by 'lut' (volume3d.transfer, corrected for stride x the plan's path). Returns premultiplied pixels,
        rows from the top: (h, w, 4) uint8 B, G, R, A (QImage.Format_ARGB32_Premultiplied's bytes), or with 'deep'
        (h, w, 4) uint16 R, G, B, A (Format_RGBA64_Premultiplied's). 'level' 1: the cube's half (plan and lut made
        for its shape)."""
        t = time.perf_counter()
        self.current()
        lv = self.levels[level]
        shape = lv.shape
        a = int(plan["axis"])
        r0, r1 = [b for b in range(3) if b != a]
        g = geometry(cam, lo, hi, shape)
        self.set_lut(lut)
        c0, c1 = clear_levels(lut)
        fb = self._framebuffer(width, height, GL_RGBA16 if deep else GL_RGBA8)
        self.ef.glBindFramebuffer(GL_FRAMEBUFFER, fb)
        self.f.glViewport(0, 0, width, height)
        self.f.glDisable(0x0BE2)                # GL_BLEND: the shader writes each pixel once
        self.prog.bind()
        L = self.loc
        unit = [np.array(e) for e in ((0, 0, 1), (0, 1, 0), (1, 0, 0))]   # axis 0 is texel r, axis 2 texel s
        self.ef.glUniform1i(L["vol"], 0)
        self.ef.glUniform1i(L["lut"], 1)
        self.ef.glUniform3i(L["shape"], *(int(s) for s in shape))
        self.ef.glUniform1i(L["axis"], a)
        self.ef.glUniform1i(L["ra"], r0)
        self.ef.glUniform1i(L["rb"], r1)
        for name, ax in (("eA", a), ("eRa", r0), ("eRb", r1)):
            self.ef.glUniform3i(L[name], *(int(v) for v in unit[ax]))
        for name in ("q0", "qx", "qy", "eye", "dir"):
            self.ef.glUniform3f(L[name], *(float(v) for v in g[name]))
        self.ef.glUniform1i(L["persp"], 1 if g["persp"] else 0)
        self.ef.glUniform2f(L["pxs"], float(cam.width) / width, float(cam.height) / height)
        stride = max(1, int(stride))
        self.ef.glUniform1i(L["stride"], stride)
        self.ef.glUniform1i(L["phase"], (stride - 1) // 2)
        self.ef.glUniform1ui(L["clear0"], int(c0))
        self.ef.glUniform1ui(L["clear1"], int(c1))
        self.ef.glUniform1f(L["done"], 1.0 - 2.0 ** -17 if deep else DONE_ALPHA)
        self.ef.glUniform1i(L["occ"], 2)
        self.ef.glUniform1i(L["B"], BLOCK)
        self.ef.glUniform1i(L["skip"], 1 if stride >= SKIP_FROM_STRIDE else 0)
        self.ef.glUniform3i(L["nb"], *(int(v) for v in lv.nb))
        self.f.glActiveTexture(GL_TEXTURE0 + 2)
        self.f.glBindTexture(GL_TEXTURE_3D, lv.occ)
        self.f.glActiveTexture(GL_TEXTURE0)
        self.f.glBindTexture(GL_TEXTURE_3D, lv.tex.textureId())
        self.f.glActiveTexture(GL_TEXTURE0 + 1)
        self.f.glBindTexture(GL_TEXTURE_2D, self.lut_tex)
        self.vao.bind()
        self.f.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.vao.release()
        self.f.glPixelStorei(GL_PACK_ALIGNMENT, 1)
        if deep:
            out = np.empty((height, width, 4), np.uint16)
            self.f.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_SHORT, out)
        else:
            out = np.empty((height, width, 4), np.uint8)
            self.f.glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, out)
        self.check("drawing the volume")
        self.prog.release()
        self.last_ms = 1e3 * (time.perf_counter() - t)
        return out


class _Level:
    """One cube on the GPU: its levels as a 3D integer texture (sent in slabs: step) and its blocks' (max, smallest
    non-zero) texture for skipping empty space (each layer's rows reduced on the GUI's volume threads meanwhile)."""

    def __init__(self, r: Renderer, q: np.ndarray):
        from PySide6.QtOpenGL import QOpenGLTexture
        self.r, self.q, self.k, self.done = r, q, 0, False
        self.shape = m0, m1, m2 = q.shape
        self.occ, self.nb = None, None
        tex = QOpenGLTexture(QOpenGLTexture.Target3D)    # it allocates without data (glTexImage3D's null pointer)
        tex.setFormat(QOpenGLTexture.R8U)
        tex.setSize(m2, m1, m0)
        tex.setMipLevels(1)
        tex.allocateStorage(QOpenGLTexture.Red_Integer, QOpenGLTexture.UInt8)
        r.check("allocating the volume's texture (too large for the GPU?)")
        if not tex.isStorageAllocated():
            raise GLUnavailable(f"no room on the GPU for a {m0} x {m1} x {m2} volume")
        self.tex = tex
        r.f.glBindTexture(GL_TEXTURE_3D, tex.textureId())
        r._params(GL_TEXTURE_3D)
        nb1, nb2 = -(-m1 // BLOCK), -(-m2 // BLOCK)
        rows = (np.empty((m0, nb1, nb2), np.uint8), np.empty((m0, nb1, nb2), np.uint8))

        def one(span):
            i0, i1 = span
            rows[0][i0:i1], rows[1][i0:i1] = occupancy_rows(q[i0:i1])
        self.rows = (rows, [VOL.gui_pool().submit(one, (i, min(i + 16, m0))) for i in range(0, m0, 16)])

    def step(self, end: float, wait: bool) -> bool:
        """Send layers until 'end' (perf_counter); then the blocks once their rows are all there. True when done."""
        if self.done:
            return True
        r, q = self.r, self.q
        m0, m1, m2 = self.shape
        slab = max(1, min(m0, UPLOAD_SLAB_BYTES // max(1, m1 * m2)))
        r.f.glBindTexture(GL_TEXTURE_3D, self.tex.textureId())
        r.f.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        while self.k < m0 and (wait or time.perf_counter() < end):
            e = min(m0, self.k + slab)
            r.ef.glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, self.k, m2, m1, e - self.k, GL_RED_INTEGER, GL_UNSIGNED_BYTE,
                                 q[self.k:e])
            self.k = e
        r.check("sending the volume to the GPU")
        rows, futures = self.rows
        if self.k < m0 or not (wait or all(f.done() for f in futures)):
            return False
        for f in futures:                       # the blocks' texture, from every layer's rows (raises their error)
            f.result()
        occ = occupancy(*rows)
        nb0, nb1, nb2, _ = occ.shape
        self.occ = r._texture()
        r.f.glBindTexture(GL_TEXTURE_3D, self.occ)
        r._params(GL_TEXTURE_3D)
        r.ef.glTexImage3D(GL_TEXTURE_3D, 0, GL_RG8UI, nb2, nb1, nb0, 0, GL_RG_INTEGER, GL_UNSIGNED_BYTE, occ)
        r.check("sending the volume's blocks")
        self.nb = (nb0, nb1, nb2)
        self.q = self.rows = None
        self.done = True
        return True

    def free(self):
        if self.rows is not None:
            for f in self.rows[1]:
                f.cancel()
        self.tex.destroy()
        if self.occ is not None:
            self.r.f.glDeleteTextures(1, np.array([self.occ], np.uint32))
        self.q = self.rows = None
        self.done = False


def to_qimage(px: np.ndarray) -> QImage:
    """render's pixels as a QImage (it keeps the array alive: img.px)."""
    h, w, _ = px.shape
    if px.dtype == np.uint16:
        img = QImage(px.data, w, h, 8 * w, QImage.Format_RGBA64_Premultiplied)
    else:
        img = QImage(px.data, w, h, 4 * w, QImage.Format_ARGB32_Premultiplied)
    img.px = px
    return img

