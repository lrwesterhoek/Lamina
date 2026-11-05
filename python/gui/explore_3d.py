"""The Explore tab's 3D view: View3D, the right pane's picture of the box turned in 3D -- the map's field on three
orthogonal planes, the voids as spheres of their true size, the box's edges, an axis triad and a scale bar -- and
View3DMixin, the part of ExploreControls that fills it (the planes read on the slice worker, planes3d), follows the
map's field, colours and slice, and inspects the voids: hover names one, a click fills the void inspector, a
double-click (or Close up) flies to it with the planes through its centre at the grid's full resolution.

Drawn by QPainter, no OpenGL: an orthographic camera (view3d.py) maps each plane to the screen by one affine
transform and each sphere to a circle, painted back to front in an exact order. It renders the same offscreen (the
tests) as on screen."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen, QTransform
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSpinBox, QVBoxLayout, QWidget)

import grids as G
import view3d as V

VIEW3D_TIP = ("Show the map's field in 3D: three planes through the box (the map's own slice and the middle of the "
              "other two), coloured as the map on one range, with the voids as spheres of radius R_v (R_eff dashed "
              "when ticked above) at their true size against the box. A void near a face is drawn once, at its "
              "catalogue centre, and may reach past the box. Any change to the map's field, colours or slice "
              "follows; the voids need 'voids' ticked.")
CLOSE_TIP = ("Look at the chosen void from near: the camera flies to it and the three planes pass through its centre, "
             "cut to a cube around it at the grid's full resolution, read periodically across the box's faces, with "
             "the neighbouring voids that reach into it")
BG = QColor(26, 27, 31)                     # the 3D view's background: white outlines read on it
SELECTED = QColor(255, 196, 64)
HOVERED = QColor(120, 220, 255)
SHAPE_ITEMS = (("spheres of R_v", "sphere"), ("fitted ellipsoids", "ellipsoid"))
SHAPE_TIP = ("How the voids are drawn. Spheres of R_v: where the smoothed density around the centre rises back to "
             "the mean. Fitted ellipsoids: the catalogue's fit to the density minimum's curvature (its Hessian), "
             "semi-axes a >= b >= c along their own directions, whose (a b c)^⅓ is R_eff; their widest section is "
             "drawn, all three on the void inspected, and the planes show where each ellipsoid cuts them. A close-up "
             "sizes its cube by R_v, or R_eff, and always holds the whole shape.")
EDGE = QColor(12, 12, 14)
DIM = QColor(120, 122, 128)                 # a close-up's neighbours: opaque grey (a translucent line is slow)                   # the dark edge under an outline: it reads over a bright plane
AXIS_COLOURS = (QColor(235, 95, 85), QColor(110, 205, 105), QColor(95, 150, 245))


def planes3d(req: dict) -> dict:
    """The slice worker's part of a 3D request: the shown planes read and smoothed as the map's slices are, cut to
    the window (periodic: a close-up near a face reads across it), strided to at most view3d.MAX_TEXTURE pixels a
    side and coloured on one range (the map's fixed range, else the clip percentiles of all three together)."""
    from explore import _smoothed                   # here: explore imports this module
    o: G.OutputSet = req["output"]
    n = int(o.n)
    cell = o.length / n
    idx, wlo, whi = V.cell_window(req["lo"], req["hi"], o.lo, cell, n)
    raw = [None, None, None]
    for a in range(3):
        if not req["shown"][a]:
            continue
        plane = _smoothed(o.slice(req["field"], a, req["index"][a], req["component"], req["ps_gradient"]),
                              req["smooth"])
        r0, r1 = V.rest_axes(a)
        vals = plane[np.ix_(idx[r0], idx[r1])]
        s = V.stride_for(*vals.shape)
        if s > 1:
            vals = vals[:vals.shape[0] // s * s:s, :vals.shape[1] // s * s:s]
        raw[a] = (np.asarray(vals, dtype=np.float64), s)
    vrange = req.get("vrange")
    if vrange is None:
        both = [v.ravel() for v, _s in (r for r in raw if r is not None)]
        vrange = G.value_range(np.concatenate(both) if both else np.zeros(1), req["log"],
                               (req["clip"], 100 - req["clip"]), req["symmetric"])
    planes = [None, None, None]
    for a, item in enumerate(raw):
        if item is None:
            continue
        vals, s = item
        r0, r1 = V.rest_axes(a)
        img = np.ascontiguousarray(vals.T[::-1])        # the map's image: columns along r0, rows r1 downward
        rgb, _ = G.colorize(img, req["cmap"], log=req["log"], symmetric=req["symmetric"], vrange=vrange)
        lo2 = (float(wlo[r0]), float(wlo[r1]))
        hi2 = (lo2[0] + vals.shape[0] * s * cell[r0], lo2[1] + vals.shape[1] * s * cell[r1])
        pos = V.plane_position(req["index"][a], o.lo[a], cell[a], wlo[a], whi[a], float(o.length[a]))
        planes[a] = {"rgb": np.ascontiguousarray(rgb), "values": vals, "lo2": lo2, "hi2": hi2, "pos": pos,
                     "stride": s}
    return {"request": req, "planes": planes, "lo": wlo, "hi": whi, "vrange": vrange}


def _pen(colour, width: float, style=Qt.SolidLine) -> QPen:
    """A pen whose width is in screen pixels whatever the painter's transform. Opaque whenever wider than one pixel:
    Qt strokes a translucent wide line as a filled path, ten times slower (300 outlines: 116 ms against 12)."""
    pen = QPen(colour, width, style)
    pen.setCosmetic(True)
    return pen


# ==================================================================== the widget
class View3D(QWidget):
    """The box in 3D: planes (per axis None or planes3d's entry), the window lo..hi they fill (Mpc) and the voids
    (set_voids); the mouse turns, moves and zooms it and reports where it points."""
    hovered = Signal(float, float)          # widget position; (-1, -1) when the mouse left
    clicked = Signal(float, float)          # a press and release without a drag
    double_clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(240, 240)
        self.cam = V.Camera()
        self.planes: list | None = None
        self.lo = self.hi = None
        # the voids: centres (k, 3) Mpc, M (k, 3, 3) their shapes (view3d: centre + M u, |u| <= 1), reach (k,) Mpc,
        # rows (k,) catalogue rows, labels (k,), r_eff (k,) or None (dashed spheres), box (the periodic side)
        self.voids: dict | None = None
        self.selected: int | None = None    # the catalogue row of the void inspected (drawn highlighted)
        self.hover_row: int | None = None
        self.focus = False                  # a close-up: the voids other than the one inspected drawn dim
        self.opacity = 0.9                  # the planes': a sphere behind one shows faintly through it
        self.message = "Choose an output on the left."
        self._fast = False                  # turning or zooming: no smooth scaling until the mouse rests
        self._settle = QTimer(self, singleShot=True, interval=180)
        self._settle.timeout.connect(self._settled)
        self._press = self._last = None
        self._moved = False
        self._outlines: list = []           # the last picture's void outlines, [(x, y, u, v, row)] (view3d.outline_at)
        self._drawn: list = []              # the last picture's quads front first: (axis, transform, texture rect)
        self._anim: QVariantAnimation | None = None

    # ---------------------------------------------------------------- the scene
    def set_scene(self, planes, lo, hi):
        self.planes = planes
        for pl in planes or ():
            if pl is not None and "img" not in pl:
                h, w, _ = pl["rgb"].shape
                pl["img"] = QImage(pl["rgb"].data, w, h, 3 * w, QImage.Format_RGB888)   # rgb kept alive in pl
        self.lo = None if lo is None else np.asarray(lo, dtype=np.float64)
        self.hi = None if hi is None else np.asarray(hi, dtype=np.float64)
        self.update()

    def set_voids(self, voids: dict | None):
        self.voids = voids
        self.update()

    def _sized(self):
        self.cam.width, self.cam.height = max(self.width(), 1), max(self.height(), 1)

    def fit(self, lo=None, hi=None):
        """The window (or lo..hi) centred and filling the view, as it is turned now."""
        self._sized()
        lo, hi = (self.lo, self.hi) if lo is None else (lo, hi)
        if lo is not None:
            self.cam.fit(lo, hi)
        self.update()

    def reset_view(self):
        self.cam.yaw, self.cam.pitch = V.DEFAULT_YAW, V.DEFAULT_PITCH
        self.fit()

    def fly_to(self, lo, hi, ms: int = 320):
        """Turn nothing, but move and zoom smoothly until lo..hi fills the view."""
        self._sized()
        goal = self.cam.copy()
        goal.fit(lo, hi)
        if self._anim is not None:
            self._anim.stop()
        t0, s0 = self.cam.target.copy(), self.cam.scale
        if ms <= 0 or not self.isVisible():
            self.cam.target, self.cam.scale = goal.target, goal.scale
            self.update()
            return
        anim = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=ms)

        def step(f):
            f = float(f)
            self.cam.target = t0 + f * (goal.target - t0)
            self.cam.scale = s0 * (goal.scale / s0) ** f
            self._fast = f < 1.0
            self.update()
        anim.valueChanged.connect(step)
        anim.finished.connect(lambda: step(1.0))
        self._anim = anim
        anim.start()

    # ---------------------------------------------------------------- what is under the mouse
    def sphere_at(self, x: float, y: float):
        """The catalogue row of the void outline under (x, y): the smallest one containing it (the map's rule)."""
        return V.outline_at(self._outlines, x, y)

    def point_at(self, x: float, y: float):
        """(axis, position (3,) Mpc, value) of the plane point in front under (x, y), or None."""
        for axis, T, rect in self._drawn:
            inv, ok = T.inverted()
            if not ok:
                continue
            p = inv.map(QPointF(x, y))
            if not rect.contains(p):
                continue
            pl = self.planes[axis]
            vals = pl["values"]
            w, h = vals.shape                            # texture: w columns (r0), h rows (r1, top = high)
            col, row = int(p.x()), int(p.y())
            if not (0 <= col < w and 0 <= row < h):
                continue
            r0, r1 = V.rest_axes(axis)
            xyz = np.zeros(3)
            xyz[axis] = pl["pos"]
            xyz[r0] = pl["lo2"][0] + (col + 0.5) / w * (pl["hi2"][0] - pl["lo2"][0])
            xyz[r1] = pl["hi2"][1] - (row + 0.5) / h * (pl["hi2"][1] - pl["lo2"][1])
            return axis, xyz, float(vals[col, h - 1 - row])
        return None

    # ---------------------------------------------------------------- painting
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        self._outlines, self._drawn = [], []
        if not self.planes or self.lo is None:
            p.setPen(QColor("gray"))
            p.drawText(self.rect().adjusted(20, 20, -20, -20), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            p.end()
            return
        self._sized()
        cam, lo, hi = self.cam, self.lo, self.hi
        pos = np.nan_to_num([pl["pos"] if pl is not None else np.nan for pl in self.planes], nan=-1e300)
        quads = V.plane_quads(lo, hi, pos, [pl is not None for pl in self.planes])
        v = self.voids
        centres = v["centres"] if v else np.zeros((0, 3))
        order = V.paint_order(quads, centres, cam, pos, lo, hi)
        corners, _ = cam.project(V.box_corners(lo, hi))
        hidden = V.hidden_edges(cam, lo, hi)
        self._draw_edges(p, corners, hidden, True)
        transforms, cuts = {}, {}
        for a, pl in enumerate(self.planes):
            if pl is not None:
                img = pl["img"]
                transforms[a] = QTransform(*V.texture_map(a, pl["pos"], pl["lo2"], pl["hi2"], img.width(),
                                                          img.height(), cam))
                cuts[a] = self._plane_cuts(a, pl)
        screen = None
        if v:                                           # every void's outline and sections on the screen, at once
            xy, _ = cam.project(v["centres"])
            u, w = V.outline_axes(cam, v["M"])
            sections = [V.section_axes(cam, v["M"][:, :, i], v["M"][:, :, j]) for i, j in V.SECTIONS]
            screen = (xy, u, w, sections)
            self._outlines = [(float(xy[k, 0]), float(xy[k, 1]), u[k], w[k], int(row))
                              for k, row in enumerate(v["rows"]) if np.all(np.isfinite(u[k]) & np.isfinite(w[k]))]
        for kind, i in order:
            if kind == "quad":
                self._draw_quad(p, quads[i], transforms, cuts)
            else:
                self._draw_void(p, i, screen)
        self._drawn.reverse()                           # front first, for point_at
        self._draw_edges(p, corners, hidden, False)
        self._draw_xray(p, screen)
        self._draw_label(p)
        self._draw_triad(p)
        self._draw_scale(p)
        p.end()

    def _plane_cuts(self, a: int, pl: dict) -> list:
        """Where the voids cut plane a: [((x, y, L, row), dashed)] in its texture's pixels, the cut being (x, y) +
        L (cos t, sin t) (view3d.plane_cuts: for spheres the map's outlines); dashed = the R_eff spheres."""
        v = self.voids
        if not v or self._fast:
            return []
        img = pl["img"]
        out = []
        for M, dashed in ((v["M"], False), (None if v.get("r_eff") is None else V.sphere_matrices(v["r_eff"]), True)):
            if M is None:
                continue
            for x, y, L, k in V.plane_cuts(v["centres"], M, v["box"], a, pl["pos"], pl["lo2"], pl["hi2"],
                                           img.width(), img.height()):
                out.append(((x, y, L, int(v["rows"][k])), dashed))     # k: the row of the arrays; the catalogue's row
        return out

    def _draw_quad(self, p: QPainter, q: dict, transforms: dict, cuts: dict):
        """A quad of its plane's texture, and the voids' cuts on it: a cut inside the quad drawn as is, one crossing
        its edge clipped to it (a clip under a turned transform is a path: slow, so only where needed)."""
        a = q["axis"]
        pl = self.planes[a]
        img = pl["img"]
        rect = QRectF(*V.texture_rect(q, pl["lo2"], pl["hi2"], img.width(), img.height()))
        T = transforms[a]
        p.save()
        p.setTransform(T)
        p.setRenderHint(QPainter.Antialiasing, False)  # no seam where the quads of one plane meet
        # smooth when shrinking a big plane; crisp cells when a small grid is blown up (the map's rule)
        p.setRenderHint(QPainter.SmoothPixmapTransform,
                        bool(not self._fast and max(np.hypot(T.m11(), T.m12()), np.hypot(T.m21(), T.m22())) < 1.0))
        p.setOpacity(self.opacity)
        p.drawImage(rect, img, rect)
        p.setOpacity(1.0)
        inside, crossing = [], []
        for c in cuts.get(a) or ():
            (x, y, L, _row), _dashed = c
            hx, hy = float(np.hypot(L[0, 0], L[0, 1])), float(np.hypot(L[1, 0], L[1, 1]))
            box = QRectF(x - hx, y - hy, 2 * hx, 2 * hy)
            if rect.contains(box):
                inside.append(c)
            elif rect.intersects(box):
                crossing.append(c)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        unit = QRectF(-1.0, -1.0, 2.0, 2.0)
        for group, clip in ((inside, False), (crossing, True)):
            if not group:
                continue
            p.setTransform(T)
            p.setClipRect(rect) if clip else p.setClipping(False)     # the clip is kept in device space
            for (x, y, L, row), dashed in group:
                p.setTransform(QTransform(L[0, 0], L[1, 0], L[0, 1], L[1, 1], x, y) * T)
                style = Qt.DashLine if dashed else Qt.SolidLine
                dim = self.focus and row not in (self.selected, self.hover_row)
                pens = (_pen(DIM, 1.0, style),) if dim else \
                    (_pen(EDGE, 2.2, style), _pen(SELECTED if row == self.selected else QColor("white"), 1.1, style))
                for pen in pens:
                    p.setPen(pen)
                    p.drawEllipse(unit)
        p.restore()
        self._drawn.append((a, transforms[a], rect))

    def _sections(self, p: QPainter, x: float, y: float, k: int, sections, colour, which, near_alpha=210,
                  far_alpha=70):
        """Void k's principal sections 'which' (indices into view3d.SECTIONS): each one's near half bright, its far
        half faint, a unit circle through one transform per section."""
        unit = QRectF(-1.0, -1.0, 2.0, 2.0)
        p.save()
        for s in which:
            u, w = sections[s][0][k], sections[s][1][k]
            p.setTransform(QTransform(u[0], u[1], w[0], w[1], x, y))
            # Qt's angles turn the other way in its y-down frame: t in (0, pi) is the span 0 to -180 degrees
            p.setPen(_pen(QColor(colour.red(), colour.green(), colour.blue(), far_alpha), 1.0))
            p.drawArc(unit, 0, 180 * 16)
            p.setPen(_pen(QColor(colour.red(), colour.green(), colour.blue(), near_alpha), 1.0))
            p.drawArc(unit, 0, -180 * 16)
        p.restore()

    def _draw_void(self, p: QPainter, k: int, screen):
        """Void k: its outline (a dark edge under a white line: it reads over a bright plane), its widest section
        (the inspected void: all three principal sections; none while turning), the hovered and inspected ones
        filled, the R_eff sphere dashed when asked."""
        v = self.voids
        xy, us, ws, sections = screen
        x, y, u, w, row = float(xy[k, 0]), float(xy[k, 1]), us[k], ws[k], int(v["rows"][k])
        hx, hy = float(np.hypot(u[0], w[0])), float(np.hypot(u[1], w[1]))       # the outline's half extents
        if not (np.isfinite(hx) and np.isfinite(hy) and hx > 0) or x + hx < 0 or y + hy < 0 \
                or x - hx > self.width() or y - hy > self.height():
            return
        sel, hov = row == self.selected, row == self.hover_row
        dim = self.focus and not (sel or hov)
        colour = SELECTED if sel else HOVERED if hov else DIM if dim else QColor("white")
        unit = QRectF(-1.0, -1.0, 2.0, 2.0)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.save()
        p.setTransform(QTransform(u[0], u[1], w[0], w[1], x, y))
        if dim:                                         # a neighbour in a close-up: a thin grey outline, no more
            p.setBrush(Qt.NoBrush)
            p.setPen(_pen(DIM, 1.0))
            p.drawEllipse(unit)
            p.restore()
            return
        if sel or hov:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 60 if sel else 45))
            p.drawEllipse(unit)
        p.setBrush(Qt.NoBrush)
        if not self._fast:
            p.setPen(_pen(EDGE, 3.0))
            p.drawEllipse(unit)
        p.setPen(_pen(colour, 2.0 if sel else 1.3))
        p.drawEllipse(unit)
        p.restore()
        if min(np.hypot(*u), np.hypot(*w)) >= 6 and not self._fast:
            self._sections(p, x, y, k, sections, colour, (0, 1, 2) if sel else (0,))
        r_eff = v.get("r_eff")
        if r_eff is not None and np.isfinite(r_eff[k]) and r_eff[k] > 0:
            p.setPen(_pen(colour, 1.0, Qt.DashLine))
            p.drawEllipse(QPointF(x, y), r_eff[k] * self.cam.scale, r_eff[k] * self.cam.scale)

    def _draw_xray(self, p: QPainter, screen):
        """The inspected void seen through the planes: its outline and principal sections over everything (a
        close-up's void sits where the planes cross, behind the ones in front)."""
        if screen is None or self.selected is None:
            return
        hit = np.nonzero(np.asarray(self.voids["rows"]) == self.selected)[0]
        if not len(hit):
            return
        xy, us, ws, sections = screen
        k = int(hit[0])
        x, y, u, w = float(xy[k, 0]), float(xy[k, 1]), us[k], ws[k]
        if not np.all(np.isfinite(u) & np.isfinite(w)):
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        if min(np.hypot(*u), np.hypot(*w)) >= 6 and not self._fast:
            self._sections(p, x, y, k, sections, SELECTED, (0, 1, 2), near_alpha=150, far_alpha=50)
        p.save()
        p.setTransform(QTransform(u[0], u[1], w[0], w[1], x, y))
        p.setPen(_pen(SELECTED, 1.2, Qt.DashLine))
        p.drawEllipse(QRectF(-1.0, -1.0, 2.0, 2.0))
        p.restore()

    def _draw_edges(self, p: QPainter, corners, hidden: set, behind: bool):
        """The window's twelve edges: the three behind it dashed first, the rest over everything."""
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(_pen(QColor(200, 200, 200, 110), 1.0, Qt.DashLine) if behind else _pen(QColor(220, 220, 220), 1.2))
        for i, (a, b) in enumerate(V.BOX_EDGES):
            if (i in hidden) == behind:
                p.drawLine(QPointF(*corners[a]), QPointF(*corners[b]))

    def _draw_label(self, p: QPainter):
        """The inspected void's name beside its outline."""
        hit = next((o for o in self._outlines if o[4] == self.selected), None) if self.selected is not None else None
        if hit is None:
            return
        x, y, u, w, row = hit
        r = float(max(np.hypot(*u), np.hypot(*w)))
        k = int(np.nonzero(np.asarray(self.voids["rows"]) == row)[0][0])
        text = self.voids["labels"][k]
        fm = QFontMetrics(p.font())
        tw, th = fm.horizontalAdvance(text) + 10, fm.height() + 4
        bx = min(max(4.0, x + 0.71 * r + 4), self.width() - tw - 4)
        by = min(max(4.0, y - 0.71 * r - th - 2), self.height() - th - 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(QRectF(bx, by, tw, th), 3, 3)
        p.setPen(SELECTED)
        p.drawText(QRectF(bx + 5, by + 2, tw, th), Qt.AlignLeft | Qt.AlignTop, text)

    def _draw_triad(self, p: QPainter):
        """x, y, z as the box is turned, in the lower-left corner."""
        R = self.cam.R
        ox, oy, L = 36.0, self.height() - 36.0, 24.0
        p.setRenderHint(QPainter.Antialiasing, True)
        ends = []
        for a in range(3):
            dx, dy = R[0, a], -R[1, a]
            ends.append((R[2, a], a, ox + L * dx, oy + L * dy))
        for _t, a, ex, ey in sorted(ends):              # the axis pointing away first
            p.setPen(QPen(AXIS_COLOURS[a], 2.0))
            p.drawLine(QPointF(ox, oy), QPointF(ex, ey))
            p.drawText(QRectF(ex + (ex - ox) * 0.25 - 6, ey + (ey - oy) * 0.25 - 8, 12, 16), Qt.AlignCenter, "xyz"[a])

    def _draw_scale(self, p: QPainter):
        """A bar of a round length in Mpc, in the lower-right corner."""
        span = self.width() / self.cam.scale
        length = G.nice_length(span)
        px = length * self.cam.scale
        if not (8 <= px <= self.width() - 40):
            return
        fm = QFontMetrics(p.font())
        label = f"{length:g} Mpc"
        bx, by = self.width() - 14 - max(px, fm.horizontalAdvance(label)), self.height() - 14
        p.setPen(QPen(QColor("white"), 3, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(QPointF(bx, by), QPointF(bx + px, by))
        p.setPen(QColor("white"))
        p.drawText(QRectF(bx, by - fm.height() - 4, max(px, fm.horizontalAdvance(label)) + 2, fm.height()),
                   Qt.AlignLeft | Qt.AlignBottom, label)

    # ---------------------------------------------------------------- the mouse
    def mousePressEvent(self, e):
        self._press = self._last = e.position()
        self._moved = False

    def mouseMoveEvent(self, e):
        pos = e.position()
        buttons = e.buttons() & (Qt.LeftButton | Qt.RightButton | Qt.MiddleButton)
        if self._press is None or not buttons:
            self.hovered.emit(pos.x(), pos.y())
            return
        if not self._moved and (pos - self._press).manhattanLength() < 4:
            return
        self._moved = True
        d = pos - self._last
        self._last = pos
        if buttons & (Qt.RightButton | Qt.MiddleButton) or e.modifiers() & Qt.ShiftModifier:
            self.cam.pan(d.x(), d.y())
        else:
            self.cam.turn(d.x(), d.y())
        self._interacting()

    def mouseReleaseEvent(self, e):
        if self._press is not None and not self._moved and e.button() == Qt.LeftButton:
            self.clicked.emit(e.position().x(), e.position().y())
        self._press = None

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.double_clicked.emit(e.position().x(), e.position().y())

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if steps:
            self._sized()
            self.cam.zoom(2.0 ** (steps / 4.0), e.position().x(), e.position().y())
            self._interacting()

    def leaveEvent(self, _e):
        self.hovered.emit(-1.0, -1.0)

    def _interacting(self):
        self._fast = True
        self._settle.start()
        self.update()

    def _settled(self):
        self._fast = False
        self.update()


# ==================================================================== the Explore pane's part
class View3DMixin:
    def _build_3d(self, col):
        """The '3D view' group of the left pane."""
        g = QGroupBox("3D view")
        v = QVBoxLayout(g)
        v.addWidget(self._note("The map's field on three planes through the box with the voids as spheres of their "
                               "true size. Drag to turn it, right-drag (or shift-drag) to move it, scroll to zoom. "
                               "Click a void to inspect it; double-click it (or Close up) to look at it from near."))
        row = QHBoxLayout()
        self.view3d_toggle = QPushButton("Show in 3D")
        self.view3d_toggle.setCheckable(True)
        self.view3d_toggle.setToolTip(VIEW3D_TIP)
        self.view3d_toggle.toggled.connect(self.show_3d)
        row.addWidget(self.view3d_toggle)
        self.view3d_reset = QPushButton("Reset view")
        self.view3d_reset.setToolTip("The first oblique look, the whole window in view")
        self.view3d_reset.clicked.connect(lambda: self.view.view3d.reset_view())
        row.addWidget(self.view3d_reset)
        row.addWidget(QLabel("Planes"))
        self.view3d_axes = []
        for a in range(3):
            cb = QCheckBox(V.AXES[a])
            cb.setChecked(True)
            cb.setToolTip(f"The plane normal to {V.AXES[a]}")
            cb.toggled.connect(lambda _on: self._3d_request())
            row.addWidget(cb)
            self.view3d_axes.append(cb)
        self.view3d_opacity = QSpinBox(minimum=10, maximum=100, value=90, singleStep=10)
        self.view3d_opacity.setSuffix(" % opaque")
        self.view3d_opacity.setToolTip("The planes' opacity: below 100 % a sphere behind a plane shows through it")
        self.view3d_opacity.valueChanged.connect(self._3d_opacity)
        row.addWidget(self.view3d_opacity)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Voids as"))
        self.void3d_shape = QComboBox()
        for text, shape in SHAPE_ITEMS:
            self.void3d_shape.addItem(text, shape)
        self.void3d_shape.setToolTip(SHAPE_TIP)
        self.void3d_shape.currentIndexChanged.connect(self._3d_shape_changed)
        row.addWidget(self.void3d_shape)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Void"))
        self.void3d = QComboBox()
        self.void3d.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.void3d.setMinimumContentsLength(18)
        self.void3d.setToolTip("The distinct, well-resolved voids, the largest first (by R_v, or by R_eff when they "
                               "are drawn as ellipsoids): choose one to inspect it")
        self.void3d.activated.connect(self._void3d_chosen)
        row.addWidget(self.void3d, 1)
        self.void3d_close = QPushButton("Close up")
        self.void3d_close.setToolTip(CLOSE_TIP)
        self.void3d_close.clicked.connect(lambda: self.close_up_3d(self._3d_selected()))
        row.addWidget(self.void3d_close)
        self.void3d_back = QPushButton("Whole box")
        self.void3d_back.setToolTip("Back from the close-up to the whole box (its planes where they were)")
        self.void3d_back.clicked.connect(self.whole_box_3d)
        row.addWidget(self.void3d_back)
        v.addLayout(row)
        self.view3d_state = QLabel("")
        self.view3d_state.setWordWrap(True)
        self.view3d_state.setStyleSheet("color: gray")
        v.addWidget(self.view3d_state)
        col.addWidget(g)
        self._3d_shown = False
        self._3d_busy = False
        self._3d_pending = None
        self._3d_res = None                     # the planes shown (planes3d's result)
        self._3d_out = None                     # the output they are of
        self._3d_index = None                   # the planes' cells along x, y, z
        self._3d_box_index = None               # ... in the whole box, kept through a close-up
        self._3d_close = None                   # the close-up: {row, centre, lo, hi}, or None for the whole box
        self._3d_refit = False                  # fit the camera to the next planes
        self._3d_list_key = None                # the catalogue the void menu was filled from
        v3 = self.view.view3d
        v3.hovered.connect(self._3d_hover)
        v3.clicked.connect(self._3d_click)
        v3.double_clicked.connect(self._3d_double)
        self._3d_buttons()

    @staticmethod
    def view3d_applicable(o) -> str:
        """'' when the 3D view can show this output, else why not."""
        if o is None:
            return "no output shown"
        if o.dim != 3:
            return "only for 3D grids"
        return ""

    def _3d_buttons(self):
        why = self.view3d_applicable(self.current)
        self.view3d_toggle.setEnabled(not why or self._3d_shown)
        self.view3d_toggle.setToolTip(VIEW3D_TIP if not why else f"The 3D view: {why}.")
        has = self._3d_shown and self._3d_selected() is not None
        self.void3d_close.setEnabled(has)
        self.void3d_back.setEnabled(self._3d_shown and self._3d_close is not None)
        self.view3d_reset.setEnabled(self._3d_shown)

    def _3d_sync(self):
        """Another output: a 2D one leaves the 3D view."""
        if self._3d_shown and self.view3d_applicable(self.current):
            self.show_3d(False)
            self.status.emit("the 3D view is for 3D grids: back to the map")
        self._3d_buttons()

    def show_3d(self, on: bool):
        """The toggle: the 3D view in place of the map (and of the figure grid), or the map back."""
        btn = self.view3d_toggle
        if on and self.view3d_applicable(self.current):
            on = False
        if on == self._3d_shown:
            btn.blockSignals(True)
            btn.setChecked(on)
            btn.blockSignals(False)
            self._3d_buttons()
            return
        view = self.view
        if on:
            if self._grid_shown:
                self.grid_toggle.setChecked(False)  # the map back first: the 3D view follows the map
            self._3d_shown = True
            self._3d_refit = True
            view.canvas.setVisible(False)
            view.view3d.setVisible(True)
            view.streams.setVisible(False)          # the picture gets the height; a click lists no streams here
            view.answer_title.setText("")
            view.readout.setText(" ")
            view.view3d.message = "reading the planes…"
            if self.voids.isChecked() and not self.voids_applicable(self.current):
                self._voids_request(self.current)
            self._3d_follow()
        else:
            self._3d_shown = False
            view.view3d.setVisible(False)
            view.canvas.setVisible(True)
            view.streams.setVisible(True)
            view.readout.setText(" ")
            self.render()                           # the map's title and colour bar back
        btn.blockSignals(True)
        btn.setChecked(self._3d_shown)
        btn.setText("Back to the map" if self._3d_shown else "Show in 3D")
        btn.blockSignals(False)
        self._3d_buttons()
        self._void_count_note()                     # the map counts by R_v, the 3D ellipsoids by R_eff
        self.changed.emit()

    def _3d_follow(self, req: dict | None = None):
        """The map changed (or the view opened): another output starts at the whole box with the map's slice and the
        middle of the other two axes; the same one moves the map's plane (not in a close-up) and recolours."""
        req = req or self.request()
        if req is None:
            return
        o = req["output"]
        if not (self._same(o, self._3d_out) and self._3d_out.n == o.n) or self._3d_index is None:
            self._3d_out = o
            self._3d_close = None
            self._3d_index = [o.n // 2] * 3
            self._3d_refit = True
        if self._3d_close is None:
            self._3d_index[req["axis"]] = int(req["index"])
        self._3d_request()

    def _3d_window(self, o):
        if self._3d_close is not None:
            return self._3d_close["lo"], self._3d_close["hi"]
        return o.lo, o.lo + o.length

    def _3d_request(self):
        """Read the planes (on the slice worker; only the latest request is served): the map's field, component,
        colours, range clip, fixed range and smoothing, never its comparison or hi-res slice."""
        if not self._3d_shown or self._shut or self.current is None or self._3d_index is None:
            return
        o, name = self.current, self.field.currentData()
        if name is None:
            return
        log = self.log.isChecked()
        lo, hi = self._3d_window(o)
        req = {"output": o, "field": name, "component": self.component.currentText(), "cmap": self.cmap.currentText(),
               "log": log, "clip": self.clip.value(), "smooth": self.smooth.value(),
               "symmetric": name in G.DIVERGING and not log,
               "ps_gradient": self.ps_gradient.currentData() or G.PS_GRADIENT_DEFAULT,
               "index": list(self._3d_index), "shown": [cb.isChecked() for cb in self.view3d_axes],
               "lo": np.asarray(lo, dtype=np.float64), "hi": np.asarray(hi, dtype=np.float64)}
        fixed = self._fixed_range(log)
        if fixed is not None:
            req["vrange"] = fixed
        if self._3d_busy:
            self._3d_pending = req
            return
        self._3d_busy = True
        self._threads()
        self._to_render3d.emit(req)

    def _3d_rendered(self, res: dict):
        self._3d_busy = False
        if self._3d_pending is not None:
            req, self._3d_pending = self._3d_pending, None
            self._3d_busy = True
            self._to_render3d.emit(req)
        req = res["request"]
        if not self._3d_shown or self._shut or not self._same(req["output"], self.current):
            return
        self._3d_res = res
        v3 = self.view.view3d
        v3.set_scene(res["planes"], res["lo"], res["hi"])
        v3.opacity = self.view3d_opacity.value() / 100.0
        if self._3d_refit:
            self._3d_refit = False
            v3.fit()
        o = req["output"]
        label = self._gradient_label(req)
        comp = "" if req["component"] in ("value", "") else f" [{req['component']}]"
        self.view.colorbar.set(req["cmap"], res["vrange"], ("log₁₀ " if req["log"] else "") + label + comp, req["log"])
        unit = "Mpc" if o.box is not None else "cells"
        planes = ", ".join(f"{V.AXES[a]} = {pl['pos']:.2f}" for a, pl in enumerate(res["planes"]) if pl is not None)
        where = f"close-up on void {self._3d_close['row']}" if self._3d_close is not None else "the whole box"
        self.view.title.setText(f"<b>{label}{comp}</b> — {o.title} — 3D, {where}"
                                + (f": planes {planes} {unit}" if planes else ": no plane shown"))
        strides = {pl["stride"] for pl in res["planes"] if pl is not None}
        self.view3d_state.setText((f"planes {planes} {unit}" if planes else "no plane ticked")
                                  + (f" · every {max(strides)}th cell shown" if max(strides, default=1) > 1 else ""))
        self._3d_voids()
        self.changed.emit()

    def _3d_failed(self, msg: str):
        self._3d_busy = False
        if self._3d_pending is not None:
            req, self._3d_pending = self._3d_pending, None
            self._3d_busy = True
            self._to_render3d.emit(req)
            return
        if self._3d_shown:
            self.view.view3d.message = f"Could not read the planes: {msg}"
            self.view.view3d.set_scene(None, None, None)

    def _3d_opacity(self, value: int):
        self.view.view3d.opacity = value / 100.0
        self.view.view3d.update()

    # ---------------------------------------------------------------- the voids
    def _void_result_3d(self):
        """The catalogue of the output shown (the void_outlines.py result), or None: off, not applicable, not found."""
        o = self.current
        if not self.voids.isChecked() or o is None or self.voids_applicable(o):
            return None
        res = self._void_cache.get(self._void_cache_key(o))
        return None if not res or "error" in res else res

    def _3d_voids(self):
        """The spheres over the planes (the distinct, well-resolved voids with an R_v; in a close-up those reaching
        into its cube, at their periodic image nearest the void looked at) and the void menu."""
        v3 = self.view.view3d
        res = self._void_result_3d() if self._3d_shown else None
        self._3d_fill_list(res)
        if res is None or self._3d_res is None:
            v3.selected = v3.hover_row = None
            v3.set_voids(None)
            self._3d_buttons()
            return
        rows = self.void_rows(res, "_r_eff" if self._3d_ellipsoids() else "_r_v")     # the Sizes range, by the size drawn
        rows = rows[np.isfinite(res["_r_v"][rows])]
        if self._3d_close is not None and self._3d_close["row"] not in rows:
            rows = np.append(rows, self._3d_close["row"])        # the void looked at stays, whatever its size
        centres = res["_centres"][rows]
        M, reach, labels = self._3d_shapes(res, rows)
        r_eff = res["_r_eff"][rows] if self.void_reff.isChecked() else None
        box = float(res["box_mpc"])
        if self._3d_close is not None:
            centres = V.nearest_images(centres, self._3d_close["centre"], box)
            keep = V.in_window(centres, reach, self._3d_res["lo"], self._3d_res["hi"])
            rows, centres, M, reach = rows[keep], centres[keep], M[keep], reach[keep]
            labels = [t for t, k in zip(labels, keep) if k]
            r_eff = None if r_eff is None else r_eff[keep]
        v3.selected = self._3d_selected()
        v3.focus = self._3d_close is not None
        v3.set_voids({"centres": centres, "M": M, "reach": reach, "rows": rows, "labels": labels, "r_eff": r_eff,
                      "box": box})
        self._3d_buttons()

    def _3d_ellipsoids(self) -> bool:
        return self.void3d_shape.currentData() == "ellipsoid"

    def _3d_shapes(self, res: dict, rows) -> tuple[np.ndarray, np.ndarray, list]:
        """The voids' shapes as drawn (view3d: centre + M u, |u| <= 1): spheres of R_v, or the fitted ellipsoids
        (a void without a fit keeps its sphere); the radius each reaches (R_v, or a); their labels."""
        rows = np.asarray(rows, dtype=np.int64)
        r_v = res["_r_v"][rows]
        M, reach = V.sphere_matrices(r_v), r_v.copy()
        labels = [f"void {k} · R_v {r:.2f} Mpc" for k, r in zip(rows, r_v)]
        if self._3d_ellipsoids() and "_semi" in res:
            semi = res["_semi"][rows]
            fit = np.all(np.isfinite(semi) & (semi > 0), axis=1)
            M[fit] = V.ellipsoid_matrices(semi[fit], res["_dirs"][rows][fit])
            reach[fit] = semi[fit].max(axis=1)
            labels = [f"void {k} · a, b, c = " + ", ".join(f"{s:.2f}" for s in semi[j]) + " Mpc" if fit[j]
                      else labels[j] for j, k in enumerate(rows)]
        return M, reach, labels

    def _3d_shape_changed(self, *_):
        self._3d_voids()
        self._void_count_note()                     # the Sizes range now counts by the shape's radius

    def _3d_fill_list(self, res):
        """The void menu: the voids drawn, the largest first by the drawn shape's radius (R_v, or the ellipsoids'
        R_eff)."""
        key = None if res is None else (self._void_cache_key(self.current), self._3d_ellipsoids(),
                                        self.void_rmin.value(), self.void_rmax.value())
        if key == self._3d_list_key and self.void3d.count():
            self._3d_list_select()
            return
        self._3d_list_key = key
        self.void3d.blockSignals(True)
        self.void3d.clear()
        if res is None:
            self.void3d.addItem("tick 'voids' above" if not self.voids.isChecked() else "no voids (yet)", None)
        else:
            name, key_r = ("R_eff", "_r_eff") if self._3d_ellipsoids() else ("R_v", "_r_v")
            rows = self.void_rows(res, key_r)
            rows, radius = rows[np.isfinite(res["_r_v"][rows])], res[key_r]
            self.void3d.addItem(f"{len(rows)} voids: choose one" if len(rows) else "no void in the Sizes range", None)
            for k in sorted(rows, key=lambda k: -np.nan_to_num(radius[k], nan=-1.0)):
                delta = res["delta"][k]
                self.void3d.addItem(f"void {k} · {name} {radius[k]:.2f} Mpc"
                                    + ("" if delta is None else f" · δ {float(delta):.2f}"), int(k))
        self.void3d.blockSignals(False)
        self._3d_list_select()

    def _3d_list_select(self):
        j = self.void3d.findData(self._3d_selected()) if self._3d_selected() is not None else 0
        self.void3d.blockSignals(True)
        self.void3d.setCurrentIndex(max(j, 0))
        self.void3d.blockSignals(False)

    def _3d_selected(self):
        """The catalogue row the void inspector shows, when it is of the output shown."""
        picked = self._void_picked
        if picked is None or self.current is None or self._void_result_3d() is None:
            return None
        return picked[1] if picked[0] == self._void_cache_key(self.current) else None

    def select_void_3d(self, row: int):
        """Inspect catalogue row 'row': the void inspector (its whole entry, Go to its centre), the menu, the
        highlight."""
        res = self._void_result_3d()
        if res is None or row is None:
            return
        self._void_picked = (self._void_cache_key(self.current), int(row))
        self.void_info.setText(G.void_info(res, int(row)))
        self.void_goto.setEnabled(not self.is_zoomed())
        self.view.view3d.selected = int(row)
        self.view.view3d.update()
        self._3d_list_select()
        self._3d_buttons()

    def _void3d_chosen(self, j: int):
        row = self.void3d.itemData(j)
        if row is not None:
            self.select_void_3d(int(row))

    def close_up_3d(self, row):
        """Fly to void 'row': the planes through its centre, cut to a cube around it (view3d.close_up_window)."""
        res, o = self._void_result_3d(), self.current
        if res is None or row is None or not self._3d_shown:
            return
        row = int(row)
        c = np.asarray(res["_centres"][row], dtype=np.float64)
        reach = float(self._3d_shapes(res, [row])[1][0])         # the drawn shape's reach: R_v, or the ellipsoid's a
        r = float(res["_r_eff"][row]) if self._3d_ellipsoids() else reach      # its radius: R_v, or R_eff
        if not np.isfinite(r):
            r = reach
        if not np.isfinite(r):
            return
        cell = o.length / o.n
        lo, hi = V.close_up_window(c, r, float(cell.min()), float(res["box_mpc"]), reach if np.isfinite(reach) else 0.0)
        if self._3d_close is None:
            self._3d_box_index = list(self._3d_index)
        self._3d_close = {"row": row, "centre": c, "lo": lo, "hi": hi}
        self._3d_index = [int(np.floor((c[a] - o.lo[a]) / cell[a])) % int(o.n) for a in range(3)]
        self.select_void_3d(row)
        self.view.view3d.fly_to(lo, hi)
        self._3d_request()
        self._3d_buttons()

    def whole_box_3d(self):
        o = self.current
        if self._3d_close is None or o is None:
            return
        self._3d_close = None
        if self._3d_box_index is not None:
            self._3d_index = list(self._3d_box_index)
        self.view.view3d.fly_to(o.lo, o.lo + o.length)
        self._3d_request()
        self._3d_buttons()

    # ---------------------------------------------------------------- the mouse over the view
    def _3d_hover(self, x: float, y: float):
        v3 = self.view.view3d
        row = v3.sphere_at(x, y) if x >= 0 else None
        if row != v3.hover_row:
            v3.hover_row = row
            v3.update()
        if x < 0:
            self.view.readout.setText(" ")
            return
        parts = []
        pt = v3.point_at(x, y)
        if pt is not None:
            _axis, xyz, value = pt
            unit = "Mpc" if self.current is not None and self.current.box is not None else "cells"
            parts.append("   ".join(f"{a} = {c:.2f}" for a, c in zip(V.AXES, xyz)) + f" {unit}")
            parts.append(f"value = {value:.5g}")
        res = self._void_result_3d()
        if row is not None and res is not None:
            parts.append(G.void_info(res, row, short=True))
        self.view.readout.setText("      ".join(parts) or " ")

    def _3d_click(self, x: float, y: float):
        row = self.view.view3d.sphere_at(x, y)
        if row is not None:
            self.select_void_3d(row)

    def _3d_double(self, x: float, y: float):
        row = self.view.view3d.sphere_at(x, y)
        if row is not None:
            self.close_up_3d(row)

    def save_3d_png(self, path) -> str:
        """The 3D view as it is on screen (its pixels at the screen's resolution) to a PNG."""
        if not self.view.view3d.grab().save(str(path)):
            raise OSError(f"could not write {path}")
        return str(path)
