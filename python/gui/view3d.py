"""The Explore tab's 3D view, its Qt-free part (explore_3d.py draws and wires it): a camera over the box in Mpc,
orthographic or perspective, the three orthogonal slice planes cut into the quads a painter's algorithm can order
exactly, the voids' outlines on the screen, the close-up window around one void, and the planes' pixels read off the
grids.

Orthographic: a parallel projection maps every slice plane to the screen by one affine transform (its texture drawn
exactly, through QPainter's transform), and a sphere of radius R to a circle of radius R x scale, so a void's outline
is its true size against the box at every turn and zoom. Perspective (the eye at a finite distance, the box's near
side drawn larger, as the eye expects; an orthographic box reads as widening toward the back): a plane maps by one
projective transform (a homography: QTransform's m13, m23), an ellipsoid's outline is still an ellipse (the image of
its tangent cone: the dual conic P Q* P^T), but its size depends on its depth; 'scale' holds at the target's depth.

Screen convention: x to the right, y DOWN (Qt's), depth in Mpc growing away from the viewer. A plane's texture is the
map's image of that slice (grids' image = plane.T[::-1]): columns along its first remaining axis, rows along the
second, the top row at the high end."""

from __future__ import annotations

import math

import numpy as np

AXES = ("x", "y", "z")
DEFAULT_YAW = -35.0            # degrees: a turn about the box's z axis ...
DEFAULT_PITCH = 25.0            # ... and the camera raised above the xy plane: an oblique look at three faces
PITCH_LIMIT = 89.0
TURN_PER_PIXEL = 0.4            # degrees of turn per pixel dragged
FIT_MARGIN = 0.9                # a fitted window fills this much of the view's shorter side
MAX_TEXTURE = 1024              # a plane's texture is strided down to at most this many pixels a side
CLOSE_UP_RADII = 2.0            # a close-up shows a cube reaching this many void radii from its centre ...
CLOSE_UP_MIN_CELLS = 8          # ... and at least this many cells ...
CLOSE_UP_FIT = 1.2              # ... and this much past the drawn shape (an ellipsoid's a), so all of it shows
PERSPECTIVE_FOCAL = 1.2         # perspective: the focal length in view-heights (the shorter side): a 45 degree field of
                                # view; at the first fit of a box the eye is some 2.2 of its sides away
NEAR = 0.05                     # ... and never nearer to the drawn window's nearest corner than this much of its side
                                # (along the view): zoomed in further, the eye stops and the view narrows instead

# the twelve edges of a box, as pairs of corner numbers (bit k of a corner = the high end along axis k)
BOX_EDGES = [(a, a | (1 << k)) for a in range(8) for k in range(3) if not a & (1 << k)]


def rest_axes(axis: int) -> tuple[int, int]:
    """The two axes of the plane normal to 'axis', in the map's order."""
    r0, r1 = [b for b in range(3) if b != axis]
    return r0, r1


def rotation(yaw: float, pitch: float) -> np.ndarray:
    """World -> view: the rows are the screen's right, up and toward-the-viewer directions in world coordinates
    (right-handed). A turntable: 'yaw' (degrees) turns the box about its z axis, 'pitch' raises the camera above
    the xy plane; (0, 0) looks along +y with x to the right and z up."""
    p, t = math.radians(yaw), math.radians(pitch)
    right = np.array([math.cos(p), math.sin(p), 0.0])
    toward = np.array([math.sin(p) * math.cos(t), -math.cos(p) * math.cos(t), math.sin(t)])
    return np.stack([right, np.cross(toward, right), toward])


class Camera:
    """A view of the box: 'target' (Mpc) at the centre of a width x height screen, 'scale' pixels per Mpc (at the
    target's depth), turned by yaw and pitch (degrees); orthographic, or with 'perspective' an eye on the line
    through the target toward the viewer, focal/scale away (PERSPECTIVE_FOCAL: a fixed field of view, so zooming
    moves the eye), never nearer to 'bounds' (lo, hi: the window drawn) than NEAR of its side."""

    def __init__(self, target=(0.0, 0.0, 0.0), scale: float = 1.0, yaw: float = DEFAULT_YAW,
                 pitch: float = DEFAULT_PITCH, width: int = 800, height: int = 600, perspective: bool = False):
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.scale, self.yaw, self.pitch = float(scale), float(yaw), float(pitch)
        self.width, self.height = int(width), int(height)
        self.perspective = bool(perspective)
        self.bounds: tuple | None = None                # (lo, hi) Mpc: the eye stays outside (perspective)
        self._R = (None, None)                          # (yaw, pitch), rotation: one per turn, not per projection

    @property
    def R(self) -> np.ndarray:
        if self._R[0] != (self.yaw, self.pitch):
            self._R = ((self.yaw, self.pitch), rotation(self.yaw, self.pitch))
        return self._R[1]

    def copy(self) -> "Camera":
        c = Camera(self.target, self.scale, self.yaw, self.pitch, self.width, self.height, self.perspective)
        c.bounds = self.bounds
        return c

    @property
    def focal(self) -> float:
        """Perspective: the focal length in pixels before the eye stops at the window (distance())."""
        return PERSPECTIVE_FOCAL * max(min(self.width, self.height), 1)

    def distance(self) -> float:
        """Perspective: the eye's distance from the target along toward() (Mpc): focal / scale, but never nearer to
        the bounds' nearest corner than NEAR of their side (inf: orthographic)."""
        if not self.perspective:
            return np.inf
        d = self.focal / self.scale
        if self.bounds is not None:
            lo, hi = (np.asarray(b, dtype=np.float64) for b in self.bounds)
            ahead = float(((box_corners(lo, hi) - self.target) @ self.R[2]).max())
            d = max(d, ahead + NEAR * float(np.max(hi - lo)))
        return float(d)

    def eye(self) -> np.ndarray:
        """Perspective: the eye's position (Mpc)."""
        return self.target + self.distance() * self.toward()

    def matrix(self) -> np.ndarray:
        """(3, 4): a world point (x, y, z, 1) -> its screen position in homogeneous form (X, Y, w), the pixel (X / w,
        Y / w); w is the point's distance in front of the eye along the view (1 everywhere: orthographic). The
        scale at the target's depth is 'scale' either way."""
        R, t, W, H = self.R, self.target, 0.5 * self.width, 0.5 * self.height
        P = np.zeros((3, 4))
        if not self.perspective:
            P[0, :3], P[0, 3] = self.scale * R[0], W - self.scale * R[0] @ t
            P[1, :3], P[1, 3] = -self.scale * R[1], H + self.scale * R[1] @ t
            P[2, 3] = 1.0
            return P
        d = self.distance()
        f = self.scale * d
        P[2, :3], P[2, 3] = -R[2], d + R[2] @ t
        P[0, :3], P[0, 3] = f * R[0], -f * R[0] @ t
        P[1, :3], P[1, 3] = -f * R[1], f * R[1] @ t
        P[0] += W * P[2]
        P[1] += H * P[2]
        return P

    def project(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """World points (n, 3) in Mpc -> screen positions (n, 2) in pixels and depths (n,) in Mpc (along the view,
        growing away from the viewer, 0 at the target). Perspective: a point not in front of the eye is NaN."""
        v = (np.asarray(pts, dtype=np.float64).reshape(-1, 3) - self.target) @ self.R.T
        xy = np.empty((len(v), 2))
        if self.perspective:
            d = self.distance()
            z = d - v[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                k = np.where(z > 1e-9 * d, self.scale * d / z, np.nan)
            xy[:, 0] = 0.5 * self.width + k * v[:, 0]
            xy[:, 1] = 0.5 * self.height - k * v[:, 1]
            return xy, -v[:, 2]
        xy[:, 0] = 0.5 * self.width + self.scale * v[:, 0]
        xy[:, 1] = 0.5 * self.height - self.scale * v[:, 1]
        return xy, -v[:, 2]

    def facing(self, axis: int, pos: float) -> int:
        """+1 when the viewer is on the high side of the plane normal to 'axis' at 'pos', -1 on the low side (an
        orthographic viewer is at infinity toward toward())."""
        if self.perspective:
            return 1 if self.eye()[axis] > pos else -1
        return 1 if self.toward()[axis] > 0 else -1

    def unproject(self, x: float, y: float) -> np.ndarray:
        """The world point at screen (x, y) on the plane through the target facing the viewer."""
        R = self.R
        return self.target + ((x - 0.5 * self.width) * R[0] - (y - 0.5 * self.height) * R[1]) / self.scale

    def turn(self, dx: float, dy: float):
        """A drag of (dx, dy) pixels: the box turns with the mouse."""
        self.yaw -= TURN_PER_PIXEL * dx
        self.pitch = float(np.clip(self.pitch + TURN_PER_PIXEL * dy, -PITCH_LIMIT, PITCH_LIMIT))

    def pan(self, dx: float, dy: float):
        """A drag of (dx, dy) pixels: the box follows the mouse."""
        R = self.R
        self.target -= (dx * R[0] - dy * R[1]) / self.scale

    def zoom(self, factor: float, x: float | None = None, y: float | None = None):
        """Magnify by 'factor' about screen point (x, y) (the centre by default): what is under it stays."""
        if x is None or y is None:
            x, y = 0.5 * self.width, 0.5 * self.height
        keep = self.unproject(x, y)
        self.scale *= float(factor)
        self.target += keep - self.unproject(x, y)

    def fit(self, lo, hi, margin: float = FIT_MARGIN):
        """Centre the window lo..hi (Mpc) and scale it to fill 'margin' of the view as it is turned now (perspective:
        its nearest corners, drawn larger, just inside)."""
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.target = 0.5 * (lo + hi)
        if self.perspective:
            # the eye focal / s away and the focal length focal (unstopped): a corner v (view frame) lands at
            # focal v / (focal / s - v_z) from the centre, inside margin x half the view when
            # 1 / s >= |v_x| / (margin W / 2) + v_z / focal (and so for y): the smallest such s fills the view
            v = (box_corners(lo, hi) - self.target) @ self.R.T
            f = self.focal
            need = np.maximum(np.abs(v[:, 0]) / (0.5 * margin * self.width),
                              np.abs(v[:, 1]) / (0.5 * margin * self.height)) + v[:, 2] / f
            if need.max() > 0:
                self.scale = 1.0 / float(need.max())
            return
        xy, _ = self.project(box_corners(lo, hi))
        span = xy.max(axis=0) - xy.min(axis=0)          # at the current scale
        if np.all(span > 0):
            self.scale *= margin * min(self.width / span[0], self.height / span[1])

    def toward(self) -> np.ndarray:
        """The direction from the box to the viewer (world coordinates)."""
        return self.R[2]


def box_corners(lo, hi) -> np.ndarray:
    """The 8 corners of the box lo..hi, corner c at the high end along axis k when bit k of c is set."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return np.array([[hi[k] if c & (1 << k) else lo[k] for k in range(3)] for c in range(8)])


def hidden_edges(cam: Camera, lo, hi) -> set[int]:
    """The edges (indices into BOX_EDGES) behind the box: orthographic, the three meeting at its corner farthest
    away; perspective, every edge whose two faces both turn away from the eye (four when the eye is level with a
    face's span, as straight in front of a face)."""
    if not cam.perspective:
        _, depth = cam.project(box_corners(lo, hi))
        far = int(np.argmax(depth))
        return {i for i, (a, b) in enumerate(BOX_EDGES) if far in (a, b)}
    lo, hi, e = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), cam.eye()
    out = set()
    for i, (a, b) in enumerate(BOX_EDGES):
        k = (a ^ b).bit_length() - 1                    # the edge runs along axis k
        front = any((e[j] > hi[j]) if a & (1 << j) else (e[j] < lo[j]) for j in range(3) if j != k)
        if not front:
            out.add(i)
    return out


# ---------------------------------------------------------------------- the planes
def plane_quads(lo, hi, pos, shown=(True, True, True)) -> list[dict]:
    """The shown planes inside the window lo..hi (the plane normal to axis a at pos[a]), each cut where the other
    shown planes cross it: [{'axis', 'corners' (4, 3) Mpc, 'u': (u0, u1), 'v': (v0, v1)}] with u, v the quad's
    extent along the plane's rest axes. Cut so, no quad crosses another plane: their order is exact (paint_order)."""
    lo, hi, pos = (np.asarray(a, dtype=np.float64) for a in (lo, hi, pos))
    inside = [bool(shown[a]) and lo[a] <= pos[a] <= hi[a] for a in range(3)]
    out = []
    for a in range(3):
        if not inside[a]:
            continue
        r0, r1 = rest_axes(a)
        cuts = [[lo[r]] + ([pos[r]] if inside[r] and lo[r] < pos[r] < hi[r] else []) + [hi[r]] for r in (r0, r1)]
        for u0, u1 in zip(cuts[0], cuts[0][1:]):
            for v0, v1 in zip(cuts[1], cuts[1][1:]):
                c = np.empty((4, 3))
                c[:, a] = pos[a]
                c[:, r0] = (u0, u1, u1, u0)
                c[:, r1] = (v0, v0, v1, v1)
                out.append({"axis": a, "corners": c, "u": (u0, u1), "v": (v0, v1)})
    return out


def paint_order(quads: list[dict], centres, cam: Camera, pos, lo, hi) -> list[tuple[str, int]]:
    """The order to paint the quads and the spheres (centres (k, 3)) back to front: [('quad', i) | ('sphere', k)].
    The planes split space into octants; a quad lies in one plane and wholly on one side of the other two
    (plane_quads), so ordering by side, plane by plane (a BSP walk: the far side, the plane, the near side), is exact
    for the quads. A sphere goes in its centre's octant, behind or in front of its neighbours by depth there: exact
    unless it crosses a plane."""
    pos = np.asarray(pos, dtype=np.float64)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    splits = {q["axis"] for q in quads}                  # only a drawn plane splits space
    splits = [a for a in range(3) if a in splits and lo[a] < pos[a] < hi[a]]
    viewer = {a: cam.facing(a, pos[a]) for a in splits}  # the viewer's side of each (the eye's, in perspective)

    def side(x, a) -> int:                               # 0 the far side of plane a, 2 the viewer's
        return 2 if (x[a] - pos[a]) * viewer[a] > 0 else 0

    keys = []
    for i, q in enumerate(quads):                        # (a plane on the window's face splits nothing: by depth)
        mid = q["corners"].mean(axis=0)
        _, d = cam.project(mid)
        keys.append((tuple(1 if a == q["axis"] else side(mid, a) for a in splits), -float(d[0]), ("quad", i)))
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if len(c):
        _, depth = cam.project(c)
        for k in range(len(c)):
            keys.append((tuple(side(c[k], a) for a in splits), -float(depth[k]), ("sphere", k)))
    keys.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in keys]


def texture_map(axis: int, pos: float, lo2, hi2, width: int, height: int, cam: Camera) -> tuple[float, ...]:
    """The affine map from a plane's texture pixels (x right, y down; width x height over lo2..hi2 of its rest axes,
    the top row at hi2[1]) to the screen: (m11, m12, m21, m22, dx, dy), QTransform's order (x' = m11 x + m21 y + dx,
    y' = m12 x + m22 y + dy)."""
    r0, r1 = rest_axes(axis)
    p = np.zeros((3, 3))
    p[:, axis] = pos
    p[:, r0] = (lo2[0], hi2[0], lo2[0])                  # the texture's (0, 0), (width, 0), (0, height)
    p[:, r1] = (hi2[1], hi2[1], lo2[1])
    xy, _ = cam.project(p)
    ex, ey = (xy[1] - xy[0]) / width, (xy[2] - xy[0]) / height
    return float(ex[0]), float(ex[1]), float(ey[0]), float(ey[1]), float(xy[0, 0]), float(xy[0, 1])


def plane_transform(cam: Camera, origin, ex, ey) -> tuple[float, ...]:
    """The map from image pixels (x right, y down) of a picture lying in 3D -- pixel (x, y) at origin + x ex + y ey
    (Mpc) -- to the screen, either projection: QTransform's nine numbers (m11, m12, m13, m21, m22, m23, m31, m32,
    m33): x' = (m11 x + m21 y + m31) / w, y' = (m12 x + m22 y + m32) / w, w = m13 x + m23 y + m33 (orthographic:
    m13 = m23 = 0, m33 = 1, texture_map's affine map)."""
    P = cam.matrix()
    hx, hy = P[:, :3] @ np.asarray(ex, dtype=np.float64), P[:, :3] @ np.asarray(ey, dtype=np.float64)
    h0 = P @ np.append(np.asarray(origin, dtype=np.float64), 1.0)
    # divided by |w| at the origin, not w: the same map, but w keeps its sign (positive in front of the eye), so Qt's
    # projective drawing does not take every point in front for one behind (an origin corner behind the eye)
    return tuple(float(v) for v in np.concatenate([hx, hy, h0]) / abs(h0[2]))


def texture_transform(axis: int, pos: float, lo2, hi2, width: int, height: int, cam: Camera) -> tuple[float, ...]:
    """texture_map's map (a plane's texture: width x height pixels over lo2..hi2 of its rest axes, the top row at
    hi2[1]) for either projection, as plane_transform's nine numbers."""
    r0, r1 = rest_axes(axis)
    origin = np.zeros(3)
    origin[axis], origin[r0], origin[r1] = pos, lo2[0], hi2[1]
    ex, ey = np.zeros(3), np.zeros(3)
    ex[r0] = (hi2[0] - lo2[0]) / width
    ey[r1] = -(hi2[1] - lo2[1]) / height
    return plane_transform(cam, origin, ex, ey)


def texture_rect(q: dict, lo2, hi2, width: int, height: int) -> tuple[float, float, float, float]:
    """A quad's part of its plane's texture: (x, y, w, h) in texture pixels."""
    (u0, u1), (v0, v1) = q["u"], q["v"]
    sx, sy = width / (hi2[0] - lo2[0]), height / (hi2[1] - lo2[1])
    return (u0 - lo2[0]) * sx, (hi2[1] - v1) * sy, (u1 - u0) * sx, (v1 - v0) * sy


# ---------------------------------------------------------------------- the voids
# A void is drawn as the shape centre + M u, |u| <= 1: a sphere of radius R (M = R I) or its fitted ellipsoid (M's
# columns: the semi-axes a >= b >= c along their directions). One code path for both: the outline, the principal
# sections and the planes' cuts all follow from M.
SECTIONS = ((0, 1), (1, 2), (0, 2))     # the principal sections, by the two columns of M that span them; the first
                                        # (normal to the third axis: a sphere's equator, an ellipsoid's widest) is
                                        # drawn on every void, all three on the one inspected


def sphere_matrices(radii) -> np.ndarray:
    """(k, 3, 3): spheres of these radii (Mpc)."""
    return np.asarray(radii, dtype=np.float64).reshape(-1, 1, 1) * np.eye(3)


def ellipsoid_matrices(semi_axes, directions) -> np.ndarray:
    """(k, 3, 3): ellipsoids of semi-axes (k, 3) along unit directions (k, 3, 3), direction i in row i (the void
    catalogue's order: a >= b >= c); column i of each matrix is direction i x semi-axis i."""
    d = np.asarray(directions, dtype=np.float64).reshape(-1, 3, 3)
    return np.swapaxes(d, 1, 2) * np.asarray(semi_axes, dtype=np.float64).reshape(-1, 1, 3)


def projection(cam: Camera) -> np.ndarray:
    """(2, 3): a world vector (Mpc) -> its screen vector (pixels; y down)."""
    R = cam.R
    return np.stack([R[0], -R[1]]) * cam.scale


def outline_axes(cam: Camera, M) -> tuple[np.ndarray, np.ndarray]:
    """The shapes' outlines on the screen: (u, v), each (k, 2) pixels, the outline being centre + cos t u + sin t v.
    A parallel projection of an ellipsoid is an ellipse: P M maps the unit ball onto it, so its axes are P M's left
    singular vectors times its singular values (a sphere's: two perpendicular vectors of R x scale)."""
    pm = projection(cam) @ np.asarray(M, dtype=np.float64).reshape(-1, 3, 3)
    U, s, _ = np.linalg.svd(pm)
    L = U * s[:, None, :]
    return L[:, :, 0], L[:, :, 1]


def section_axes(cam: Camera, A, B) -> tuple[np.ndarray, np.ndarray]:
    """The curves centre + cos t A + sin t B (A, B (k, 3) in Mpc: a section of a shape through its centre) on the
    screen: (u, v), each (k, 2) pixels, turned within the curve so that its near half is 0 < t < pi (u has no part
    toward the viewer, v leans toward it)."""
    A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
    t = cam.toward()
    ang = np.arctan2(-(A @ t), B @ t)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    P = projection(cam)
    return (c * A + s * B) @ P.T, (c * B - s * A) @ P.T


def outline_ellipses(cam: Camera, centres, M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shapes' outlines on the screen, either projection: (xy, u, v), each (k, 2) pixels, the outline being
    xy + cos t u + sin t v. Orthographic: xy the centre's projection and outline_axes' u, v. Perspective: the image
    of the cone from the eye tangent to the ellipsoid, an ellipse: with the shape's dual quadric
    Q* = [[M M^T - c c^T, -c], [-c^T, -1]] (the unit sphere's diag(1, 1, 1, -1) carried by [[M, c], [0, 1]]) and the
    camera matrix P, the outline's dual conic is P Q* P^T; scaled to -1 in its corner it is [[N N^T - x x^T, -x],
    [-x^T, -1]] for the ellipse x + N w, |w| = 1. NaN for a shape not wholly in front of the eye (the eye inside it,
    or the shape reaching behind it): no ellipse to draw."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    M = np.asarray(M, dtype=np.float64).reshape(-1, 3, 3)
    if not cam.perspective:
        xy, _ = cam.project(c)
        u, v = outline_axes(cam, M)
        return xy, u, v
    k = len(c)
    Qd = np.zeros((k, 4, 4))
    Qd[:, :3, :3] = M @ np.swapaxes(M, 1, 2) - c[:, :, None] * c[:, None, :]
    Qd[:, :3, 3] = Qd[:, 3, :3] = -c
    Qd[:, 3, 3] = -1.0
    P = cam.matrix()
    C = np.einsum("ij,kjl,ml->kim", P, Qd, P)
    # in front of the eye: the shape's nearest point (along the view) still ahead of it
    w_c = c @ P[2, :3] + P[2, 3]
    reach = np.linalg.norm(np.swapaxes(M, 1, 2) @ P[2, :3], axis=1)
    return _ellipse_from_dual(C, w_c - reach > 1e-6 * np.abs(w_c))


def _ellipse_from_dual(C: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ellipses x + N w, |w| = 1, from dual conics C (k, 3, 3) scaled to -1 in their corner (where ok): x = -C[:2, 2],
    N N^T = C[:2, :2] + x x^T; (x, u, v) with u the longer axis, NaN where not ok or not an ellipse."""
    k = len(C)
    xy, u, v = (np.full((k, 2), np.nan) for _ in range(3))
    ok = ok & (C[:, 2, 2] < 0)
    if ok.any():
        Cn = C[ok] / -C[ok, 2, 2][:, None, None]
        x0 = -Cn[:, :2, 2]
        lam, U = np.linalg.eigh(Cn[:, :2, :2] + x0[:, :, None] * x0[:, None, :])
        L = U * np.sqrt(np.maximum(lam, 0.0))[:, None, :]
        rows = np.nonzero(ok)[0]
        xy[rows], u[rows], v[rows] = x0, L[:, :, 1], L[:, :, 0]
        bad = rows[~np.all(lam > 0, axis=1)]
        xy[bad] = u[bad] = v[bad] = np.nan
    return xy, u, v


def circle_images(M3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The images of the unit circle under projective maps M3 ((k, 3, 3): (cos t, sin t, 1) -> the homogeneous
    screen point): ellipses (x, u, v) as outline_ellipses gives them -- a circle's dual conic diag(1, 1, -1) carried to
    M3 diag(1, 1, -1) M3^T. NaN where the curve is not wholly in front of the eye (w = M3[2] (cos, sin, 1) > 0 for
    every t)."""
    M3 = np.asarray(M3, dtype=np.float64).reshape(-1, 3, 3)
    C = M3 @ np.diag([1.0, 1.0, -1.0]) @ np.swapaxes(M3, 1, 2)
    ok = M3[:, 2, 2] - np.hypot(M3[:, 2, 0], M3[:, 2, 1]) > 1e-9 * np.abs(M3[:, 2, 2])
    return _ellipse_from_dual(C, ok)


def circle_points(M3, t) -> np.ndarray:
    """The screen points of (cos t, sin t, 1) under M3 (k, 3, 3), one angle t (k,) each: (k, 2) pixels."""
    M3 = np.asarray(M3, dtype=np.float64).reshape(-1, 3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    h = M3 @ np.stack([np.cos(t), np.sin(t), np.ones_like(t)], axis=1)[:, :, None]
    return h[:, :2, 0] / h[:, 2:3, 0]


def texture_matrix(T) -> np.ndarray:
    """plane_transform's nine numbers as the 3 x 3 matrix taking (x, y, 1) to the homogeneous screen point."""
    m11, m12, m13, m21, m22, m23, m31, m32, m33 = (float(v) for v in T)
    return np.array([[m11, m21, m31], [m12, m22, m32], [m13, m23, m33]])


def cut_images(T, cuts) -> list:
    """Where the cuts on a plane (plane_cuts' (x, y, L, k): the ellipse (x, y) + L (cos t, sin t) in its texture's
    pixels) land on the screen through the plane's map T (plane_transform's numbers): [(x, u, v) or None], an ellipse
    drawn with an affine map instead of each cut through the projective one (twenty times dearer in Qt). None
    where the cut is not wholly in front of the eye (then draw it through T)."""
    if not cuts:
        return []
    H = texture_matrix(T)
    A = np.zeros((len(cuts), 3, 3))
    for j, (x, y, L, _k) in enumerate(cuts):
        A[j, :2, :2], A[j, :2, 2], A[j, 2, 2] = L, (x, y), 1.0
    xy, u, v = circle_images(H[None] @ A)
    return [None if not np.all(np.isfinite([*xy[j], *u[j], *v[j]])) else (xy[j], u[j], v[j]) for j in range(len(cuts))]


def section_arcs(cam: Camera, centres, A, B) -> tuple[np.ndarray, ...]:
    """The curves centre + cos t A + sin t B (Mpc, (k, 3) each: sections of the shapes through their centres) on the
    screen of either camera, as ellipses with their near half: (x, u, v, start, span), the ellipse x + cos s u +
    sin s v and its near half the arc from s = start over the signed angle span; the far half the rest. Near: on
    the viewer's side of the plane through the centre facing the viewer (the eye, in perspective). A central
    section's image is the unit circle's image under P [[A, B, c], [0, 0, 1]] (circle_images)."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    A, B = np.asarray(A, dtype=np.float64).reshape(-1, 3), np.asarray(B, dtype=np.float64).reshape(-1, 3)
    G = np.zeros((len(c), 4, 3))
    G[:, :3, 0], G[:, :3, 1], G[:, :3, 2], G[:, 3, 2] = A, B, c, 1.0
    M3 = cam.matrix()[None] @ G
    xy, u, v = circle_images(M3)
    w = cam.eye()[None, :] - c if cam.perspective else np.broadcast_to(cam.toward(), c.shape)
    mid = np.arctan2(np.einsum("ij,ij->i", B, w), np.einsum("ij,ij->i", A, w))
    start, span = np.full(len(c), np.nan), np.full(len(c), np.nan)
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    good = np.isfinite(det) & (np.abs(det) > 1e-12)
    if good.any():
        def angle(p):                                   # the point's angle s on its ellipse
            d = p[good] - xy[good]
            a_ = (d[:, 0] * v[good, 1] - d[:, 1] * v[good, 0]) / det[good]
            b_ = (u[good, 0] * d[:, 1] - u[good, 1] * d[:, 0]) / det[good]
            return np.arctan2(b_, a_)
        s0 = angle(circle_points(M3, mid - 0.5 * np.pi))
        s1 = angle(circle_points(M3, mid + 0.5 * np.pi))
        sm = angle(circle_points(M3, mid))
        sp = np.mod(s1 - s0, 2 * np.pi)
        sp = np.where(np.mod(sm - s0, 2 * np.pi) <= sp, sp, sp - 2 * np.pi)   # the way round through the near middle
        start[good], span[good] = s0, sp
    return xy, u, v, start, span


def section_points(cam: Camera, centre, A, B, n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """The curve centre + cos t A + sin t B (Mpc; a section of a shape through its centre) on the screen, either
    projection, as two polylines (n points each, pixels): its near half -- on the viewer's side of the plane through
    the centre facing the viewer (the eye, in perspective) -- and its far half. section_axes' split, for either projection."""
    c = np.asarray(centre, dtype=np.float64)
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    w = cam.eye() - c if cam.perspective else cam.toward()
    mid = np.arctan2(B @ w, A @ w)                       # the near half: t within pi / 2 of mid
    t = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
    near, _ = cam.project(c + np.cos(mid + t)[:, None] * A + np.sin(mid + t)[:, None] * B)
    far, _ = cam.project(c + np.cos(mid + np.pi + t)[:, None] * A + np.sin(mid + np.pi + t)[:, None] * B)
    return near, far


def outline_at(outlines, x: float, y: float):
    """The row of the outline [(cx, cy, u, v, row)] under screen point (x, y): of those containing it, the smallest
    (grids.void_at's rule: a small void drawn over a big one stays reachable). None when none does."""
    best = None
    for cx, cy, u, v, row in outlines or ():
        det = u[0] * v[1] - u[1] * v[0]
        if not abs(det) > 1e-12:
            continue
        dx, dy = x - cx, y - cy
        a, b = (dx * v[1] - dy * v[0]) / det, (u[0] * dy - u[1] * dx) / det
        if a * a + b * b <= 1.0 and (best is None or abs(det) < best[0]):
            best = (abs(det), int(row))
    return None if best is None else best[1]


def plane_cuts(centres, M, box: float, axis: int, pos: float, lo2, hi2, width: int, height: int,
               periodic: bool = True) -> list[tuple[float, float, np.ndarray, int]]:
    """Where the shapes cut the plane normal to 'axis' at 'pos', in its texture's pixels (texture_map's frame:
    width x height over lo2..hi2, the top row at hi2[1]): [(x, y, L, k)], the cut being (x, y) + L (cos t, sin t),
    k the shape's row; each periodic image that reaches the texture kept (grids.void_circles' rule; for spheres
    the same circles). With S = M M^T the cut at distance d from the centre is the conditional ellipse: centre
    shifted by S_ra d / S_aa, shape (S_rr - S_ra S_ar / S_aa)(1 - d^2 / S_aa)."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    M = np.asarray(M, dtype=np.float64).reshape(-1, 3, 3)
    S = M @ np.swapaxes(M, 1, 2)
    r0, r1 = rest_axes(axis)
    rr = [r0, r1]
    d = float(pos) - c[:, axis]
    if periodic and box > 0:
        d -= box * np.round(d / box)
    saa = S[:, axis, axis]
    with np.errstate(invalid="ignore"):
        cut = np.isfinite(saa) & (saa > 0) & (d * d < saa) & np.isfinite(c).all(axis=1)
    idx = np.nonzero(cut)[0]
    if not len(idx):
        return []
    S, d, saa = S[idx], d[idx], saa[idx]
    sra = S[:, rr, axis]
    mid = c[idx][:, rr] + sra * (d / saa)[:, None]
    C = (S[:, rr][:, :, rr] - sra[:, :, None] * sra[:, None, :] / saa[:, None, None]) \
        * (1.0 - d * d / saa)[:, None, None]
    w, V = np.linalg.eigh(C)
    L = V * np.sqrt(np.maximum(w, 0.0))[:, None, :]            # in the plane, Mpc
    (u0, v0), (u1, v1) = (float(lo2[0]), float(lo2[1])), (float(hi2[0]), float(hi2[1]))
    sx, sy = width / (u1 - u0), height / (v1 - v0)
    L = L * np.array([sx, -sy])[None, :, None]                  # texture pixels: x along u, y DOWN along v
    hu, hv = np.sqrt(C[:, 0, 0]), np.sqrt(C[:, 1, 1])           # the cut's half extents (Mpc)
    shifts = (-box, 0.0, box) if periodic and box > 0 else (0.0,)
    out = []
    for j, k in enumerate(idx):
        for du in shifts:
            u = mid[j, 0] + du
            if u + hu[j] < u0 or u - hu[j] > u1:
                continue
            for dv in shifts:
                v = mid[j, 1] + dv
                if v + hv[j] < v0 or v - hv[j] > v1:
                    continue
                out.append(((u - u0) * sx, (v1 - v) * sy, L[j], int(k)))
    return out


def nearest_images(centres, ref, box: float) -> np.ndarray:
    """Each centre moved by whole boxes to its periodic image nearest 'ref' (a close-up near a face shows the voids
    across it where they are)."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    if not box > 0:
        return c.copy()
    return c - box * np.round((c - np.asarray(ref, dtype=np.float64)) / box)


def in_window(centres, radii, lo, hi) -> np.ndarray:
    """Which shapes (centres, and the radius each reaches: a sphere's R, an ellipsoid's a) reach into the window."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    r = np.nan_to_num(np.asarray(radii, dtype=np.float64).reshape(-1, 1), nan=0.0)
    return np.all((c + r >= np.asarray(lo)) & (c - r <= np.asarray(hi)), axis=1)


def close_up_window(centre, radius: float, cell: float, box: float, reach: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The cube a close-up shows around a void of radius 'radius' (Mpc: R_v, or an ellipsoid's R_eff) at 'centre':
    (lo, hi) in Mpc, the box's frame (it may reach past a face: the planes are read periodically). 'reach' is how far
    the drawn shape reaches (a sphere's R_v, an ellipsoid's a), 'cell' the grid's cell, 'box' the periodic box's
    side."""
    # Design: a close-up shows two radii (CLOSE_UP_RADII) around the void. That takes in the compensation ridge just
    # outside R_v and some of the neighbouring walls; a smaller factor fills the view with the void's interior, a
    # larger one shows its neighbours. The cube stays under the box (it must not meet its own periodic image) and
    # keeps CLOSE_UP_MIN_CELLS cells to show structure. Measured on TNG100-3-Dark z = 0: two of an ellipsoid's a made
    # a median void's cube 72 Mpc of the 111; two R_eff, 49 Mpc, near a sphere's two R_v, 42.
    half = max(CLOSE_UP_RADII * float(radius), CLOSE_UP_MIN_CELLS * float(cell), CLOSE_UP_FIT * float(reach))
    if box > 0:
        half = min(half, 0.5 * float(box))
    c = np.asarray(centre, dtype=np.float64)
    return c - half, c + half


# ---------------------------------------------------------------------- the planes' pixels
def cell_window(lo, hi, origin, cell, n: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """The grid cells covering the window lo..hi (Mpc), periodic: per axis the cell indices (wrapped into 0..n-1)
    and the window snapped to whole cells (lo, hi in Mpc, possibly past the box)."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    origin, cell = np.asarray(origin, dtype=np.float64), np.asarray(cell, dtype=np.float64)
    i0 = np.floor((lo - origin) / cell + 1e-9).astype(np.int64)
    i1 = np.maximum(np.ceil((hi - origin) / cell - 1e-9).astype(np.int64), i0 + 1)
    i1 = np.minimum(i1, i0 + n)                          # never more than the box: a cell shown twice
    idx = [np.arange(i0[a], i1[a]) % n for a in range(3)]
    return idx, origin + i0 * cell, origin + i1 * cell


def plane_position(index: int, origin: float, cell: float, lo: float, hi: float, box: float) -> float:
    """Where the plane through cell 'index' lies: its periodic image inside the window lo..hi (Mpc), the highest
    one not past hi (below lo when none is inside: plane_quads then leaves it out)."""
    p = origin + (index + 0.5) * cell
    if box > 0:
        p += box * math.floor((hi - p) / box)
    return float(p)


def stride_for(n0: int, n1: int, limit: int = MAX_TEXTURE) -> int:
    """The step that brings a plane of n0 x n1 cells to at most 'limit' pixels a side."""
    return max(1, -(-max(n0, n1) // limit))
