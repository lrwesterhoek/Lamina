"""The Explore tab's 3D view: View3D, the right pane's picture of the box turned in 3D -- the map's field as a
translucent volume (or on three orthogonal planes), the voids as spheres of their true size, the box's edges, an axis
triad and a scale bar -- and View3DMixin, the part of ExploreControls that fills it (the volume or the planes read on
the slice worker, planes3d), follows the map's field, colours and slice, and inspects the voids: hover names one, a
click fills the void inspector, a double-click (or Close up) flies to it, the volume or the planes then read from the
grid at its full resolution around it.

Drawn by QPainter, no OpenGL (Qt's offscreen platform has none: the tests could not draw it): an orthographic camera
(view3d.py) maps each plane, and each slice of the volume, to the screen by one affine transform and each sphere to a
circle. The volume is direct volume rendering by texture slicing (volume3d.py): the averaged cube's slices along the
axis facing the viewer, coloured and made translucent by a transfer function on the map's scale, composited front to
back in 16 bits a channel. It renders the same offscreen (the tests) as on screen."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

import grids as G
import view3d as V
import volume3d as VOL

VIEW3D_TIP = ("Show the map's field in 3D: the whole box as a translucent volume, the field averaged down to the "
              "detail chosen, its low values see-through and its high values opaque in the map's colours (or three "
              "planes through the box: the map's own slice and the middle of the other two), with the voids as "
              "spheres of radius R_v (R_eff dashed when ticked above) at their true size against the box. A void "
              "near a face is drawn once, at its catalogue centre, and may reach past the box. Any change to the "
              "map's field or colours follows (the planes: its slice too); the voids need 'voids' ticked.")
STYLE_ITEMS = (("volume", "volume"), ("planes", "planes"))
STYLE_TIP = ("Volume: the field of the whole box (or a close-up's cube) averaged down to the detail chosen and drawn "
             "as a translucent cloud: cells below the low end of the shown range are transparent, the opacity rises "
             "to the high end on the map's scale (log for a density), coloured with the map's colours over the same "
             "range; its range is its own (the sliders below), the map's smoothing and range clip are the planes'. "
             "Planes: the map's slice and the middle of the other two axes at the grid's own resolution.")
DETAIL_TIP = ("The volume's cells a side, at most: the grid is averaged over blocks of its cells (a 1024³ grid in "
              "blocks of 4³ at 256³, of 2³ at 512³, its own cells at 1024³; a smaller grid or close-up: its own "
              "cells at most). A close-up's cube is read from the grid again at this detail: finer. While the view "
              "turns a coarse copy (160³ at most) is drawn; at rest the full cube at the screen's resolution, a "
              "second or two for 1024³, which also needs some 10 GB of memory while it is read (less free: a smaller "
              "detail, said under the controls).")
PERSPECTIVE_TIP = ("Perspective: the box seen from an eye at a finite distance, its near side drawn larger, as the eye "
                   "expects (without it a box reads as widening toward the back). The voids' outlines are then true "
                   "in shape but their size depends on their depth, and the scale bar holds at the window's centre. "
                   "Off: a parallel projection, every void a circle of its true size against the box.")
RANGE_TIP = ("The shown range on the map's scale: below its low end a cell is transparent (the voids and the "
             "underdense walls of a density from ρ/ρ̄ = 1), from there the opacity rises to its high end; the colour "
             "spans the same range. A diverging field (velocity divergence, eigenvalues) is shown by |value|.")
OPACITY_TIP = ("How opaque the densest cells are: the opacity of a cell at the high end of the range, 1/128 of the "
               "window's side thick. The same look at every detail, close-up and turn.")
FAST_SCALE = 0.5                # while the view turns: the coarse cube on a half-size picture
SHARP_CELL_PX = 3.0             # the settled picture: no finer than this many device pixels a cell
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
EDGE = QColor(12, 12, 14)                   # the dark edge under an outline: it reads over a bright plane
DIM = QColor(120, 122, 128)                 # a close-up's neighbours: opaque grey (a translucent line is slow)
QUIET = QColor(165, 190, 205)               # the voids over the volume: a thin, cool line apart from its warm colours
AXIS_COLOURS = (QColor(235, 95, 85), QColor(110, 205, 105), QColor(95, 150, 245))


def volume_scene(req: dict) -> dict:
    """The slice worker's part of a volume request: the field (its stored grid and the map's component) over the
    window averaged down to req['detail'] cells a side and prepared for drawing (volume3d: levels on the map's
    scale, the three slice stacks). req['progress'] says how far it is and can cancel it."""
    o: G.OutputSet = req["output"]
    prep = VOL.prepared_hit(o, req["field"], req["component"], req["lo"], req["hi"], int(req["detail"]),
                            bool(req["log"]))
    if prep is None:                            # not kept: read, average and colour it
        avg = VOL.average_window(o, req["field"], req["component"], req["lo"], req["hi"], int(req["detail"]),
                                 req.get("progress"))
        prep = VOL.prepare(avg, bool(req["log"]), req.get("progress"))
    return {"request": req, "volume": prep, "planes": [None, None, None], "lo": prep["lo"], "hi": prep["hi"],
            "vrange": None, "vkey": req.get("vkey")}


def planes3d(req: dict) -> dict:
    """The slice worker's part of a 3D request: the shown planes read and smoothed as the map's slices are, cut to
    the window (periodic: a close-up near a face reads across it), strided to at most view3d.MAX_TEXTURE pixels a
    side and coloured on one range (the map's fixed range, else the clip percentiles of all three together). A
    volume request (req['style'] 'volume') goes to volume_scene."""
    if req.get("style") == "volume":
        return volume_scene(req)
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


def _canvas(width: int, height: int, scale: float) -> tuple[QImage, QTransform]:
    """A transparent 16-bit picture of a width x height view at 'scale' of it, and the map from view to its pixels."""
    iw, ih = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    img = QImage(iw, ih, QImage.Format_RGBA64_Premultiplied)
    img.fill(Qt.transparent)
    return img, QTransform.fromScale(iw / max(width, 1), ih / max(height, 1))


def _composite(img: QImage, S: QTransform, plan: dict, slices) -> None:
    """Draw (index, (h, w) uint64 RGBA64 pixels) slices onto img, front first, with the 'under' operator
    (DestinationOver), each by its plan transform (then S) and smoothly (bilinear) resampled."""
    w, h = plan["size"]
    q = QPainter(img)
    q.setCompositionMode(QPainter.CompositionMode_DestinationOver)
    q.setRenderHint(QPainter.SmoothPixmapTransform, True)
    for i, px in slices:
        q.setTransform(QTransform(*plan["transforms"][int(i)]) * S)
        q.drawImage(0, 0, QImage(px.data, w, h, 8 * w, QImage.Format_RGBA64_Premultiplied))
    q.end()


def render_slices(stack: np.ndarray, plan: dict, width: int, height: int, scale: float = 1.0) -> QImage:
    """The volume's picture on a width x height view (drawn at 'scale' of it: smaller while turning, larger on a
    high-density screen): the slices of 'stack' ((m, h, w) uint64, Format_RGBA64_Premultiplied pixels:
    volume3d.transfer of the levels) placed by 'plan' (volume3d.slice_plan) and composited front to back onto a
    transparent 16-bit image. Transparent where nothing is."""
    img, S = _canvas(width, height, scale)
    _composite(img, S, plan, ((i, stack[int(i)]) for i in plan["order"]))
    return img


class SharpPicture:
    """A settled volume picture drawn a few slices at a time (pump), so the window stays responsive while a large
    cube is drawn at the screen's full resolution: each slice's colours gathered from its levels on volume3d's
    threads (numpy lets go of the GIL), a few slices ahead, and composited in order here."""
    AHEAD = 24                                  # slices gathered ahead of the one drawn

    def __init__(self, key, images, lut: np.ndarray, plan: dict, width: int, height: int, ratio: float):
        self.key, self.images, self.lut, self.plan = key, images, lut, plan
        # the picture in device pixels (S maps the view onto them); deliberately no setDevicePixelRatio: QPainter
        # would scale by it again on top of S (a Retina picture twice its size), and _draw_volume draws it into
        # the widget's rect whatever its pixel count
        self.img, self.S = _canvas(width, height, ratio)
        self.order = [int(i) for i in plan["order"]]
        self.next = 0                           # the next slice (in order) to draw
        self.futures: dict = {}
        self.t0 = time.perf_counter()
        self.ms = 0.0

    def _gather(self, i: int) -> np.ndarray:
        w, h = self.plan["size"]
        out = np.empty((h, w), dtype=np.uint64)
        np.take(self.lut, self.images[i], out=out)
        return out

    def done(self) -> bool:
        return self.next >= len(self.order)

    def pump(self, budget_s: float = float("inf")) -> bool:
        """Draw slices for about budget_s seconds (all: inf); True when the picture is complete."""
        end = time.perf_counter() + budget_s
        ready = []
        while not self.done():
            for k in range(self.next, min(self.next + self.AHEAD, len(self.order))):
                i = self.order[k]
                if i not in self.futures:
                    self.futures[i] = VOL.gui_pool().submit(self._gather, i)
            i = self.order[self.next]
            ready.append((i, self.futures.pop(i).result()))
            self.next += 1
            if len(ready) >= 8 or self.done() or time.perf_counter() > end:
                _composite(self.img, self.S, self.plan, ready)
                ready = []
            if time.perf_counter() > end:
                break
        if ready:
            _composite(self.img, self.S, self.plan, ready)
        if self.done():
            self.ms = 1e3 * (time.perf_counter() - self.t0)
        return self.done()

    def cancel(self):
        for f in self.futures.values():
            f.cancel()
        self.futures = {}


def _pen(colour, width: float, style=Qt.SolidLine) -> QPen:
    """A pen whose width is in screen pixels whatever the painter's transform. Opaque whenever wider than one pixel:
    Qt strokes a translucent wide line as a filled path, ten times slower (300 outlines: 116 ms against 12)."""
    pen = QPen(colour, width, style)
    pen.setCosmetic(True)
    return pen


# ==================================================================== the widget
class View3D(QWidget):
    """The box in 3D: a volume (set_volume, drawn through set_transfer's transfer function) or planes (per axis None
    or planes3d's entry), as 'style' says, the window lo..hi they fill (Mpc) and the voids (set_voids); the mouse
    turns, moves and zooms it and reports where it points."""
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
        self.style = "volume"               # 'volume' or 'planes': what is drawn of the field
        self.volume: dict | None = None     # volume3d.prepare's dict
        self.transfer: tuple | None = None  # (lo, hi, opacity, cmap, symmetric, gamma): volume3d.transfer's
        self._stacks: dict = {}             # the coarse cube's slices coloured: (key, stack), while turning
        self._picture = None                # (key, QImage): the quick picture (while turning, or until the sharp one)
        self._sharp = None                  # (key, QImage): the settled picture at the screen's full resolution
        self._job: SharpPicture | None = None   # ... being drawn, a few slices a tick
        self._pump = QTimer(self, interval=0)
        self._pump.timeout.connect(self._pump_tick)
        self.last_volume_ms = 0.0           # the last sharp picture's cost, from start to finish

    # ---------------------------------------------------------------- the scene
    def set_volume(self, prep: dict | None, lo=None, hi=None):
        """The volume to draw (volume3d.prepare's dict, None: none) and its window lo..hi (Mpc)."""
        self.volume = prep
        self._stacks, self._picture, self._sharp = {}, None, None
        self._stop_job()
        if lo is not None:
            self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.update()

    def set_perspective(self, on: bool):
        """Perspective (the eye at a finite distance) or orthographic; the window fitted again, turned as it is."""
        self.cam.perspective = bool(on)
        self.fit()

    def set_transfer(self, lo: float, hi: float, opacity: float, cmap: str, symmetric: bool = False,
                     gamma: float = VOL.DEFAULT_GAMMA):
        """The volume's transfer function: transparent below 'lo' (the map's scale: log10 of a density), the opacity
        rising to 'opacity' at 'hi', coloured with 'cmap' over lo..hi (volume3d.transfer)."""
        self.transfer = (float(lo), float(hi), float(opacity), str(cmap), bool(symmetric), float(gamma))
        self.update()

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
        """The camera to the widget's size and the window drawn (a perspective eye stays outside it)."""
        self.cam.width, self.cam.height = max(self.width(), 1), max(self.height(), 1)
        self.cam.bounds = None if self.lo is None else (self.lo, self.hi)

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
        volume = self.style == "volume"
        if self.lo is None or (self.volume is None or self.transfer is None if volume else not self.planes):
            p.setPen(QColor("gray"))
            p.drawText(self.rect().adjusted(20, 20, -20, -20), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            p.end()
            return
        self._sized()
        cam, lo, hi = self.cam, self.lo, self.hi
        v = self.voids
        corners, _ = cam.project(V.box_corners(lo, hi))
        hidden = V.hidden_edges(cam, lo, hi)
        self._draw_edges(p, corners, hidden, True)
        screen = None
        if v:                                           # every void's outline and sections on the screen, at once
            xy, u, w = V.outline_ellipses(cam, v["centres"], v["M"])
            # the sections: orthographic, ellipses (u, v) about the projected centre; perspective, drawn as polylines
            sections = [V.section_arcs(cam, v["centres"], v["M"][:, :, i], v["M"][:, :, j]) for i, j in V.SECTIONS] \
                if cam.perspective else [V.section_axes(cam, v["M"][:, :, i], v["M"][:, :, j]) for i, j in V.SECTIONS]
            reff = None
            if v.get("r_eff") is not None:              # the dashed R_eff spheres' outlines
                r = np.nan_to_num(np.asarray(v["r_eff"], dtype=np.float64), nan=0.0)
                reff = V.outline_ellipses(cam, v["centres"], V.sphere_matrices(r))
            screen = (xy, u, w, sections, reff)
            self._outlines = [(float(xy[k, 0]), float(xy[k, 1]), u[k], w[k], int(row))
                              for k, row in enumerate(v["rows"])
                              if np.all(np.isfinite(u[k]) & np.isfinite(w[k]) & np.isfinite(xy[k]))]
        if volume:
            self._draw_volume(p)
            if v and len(v["centres"]):                 # over the volume, the farthest first
                _, depth = cam.project(v["centres"])
                for k in np.argsort(-depth, kind="stable"):
                    self._draw_void(p, int(k), screen)
        else:
            pos = np.nan_to_num([pl["pos"] if pl is not None else np.nan for pl in self.planes], nan=-1e300)
            quads = V.plane_quads(lo, hi, pos, [pl is not None for pl in self.planes])
            centres = v["centres"] if v else np.zeros((0, 3))
            order = V.paint_order(quads, centres, cam, pos, lo, hi)
            transforms, cuts = {}, {}
            for a, pl in enumerate(self.planes):
                if pl is not None:
                    img = pl["img"]
                    T9 = V.texture_transform(a, pl["pos"], pl["lo2"], pl["hi2"], img.width(), img.height(), cam)
                    transforms[a] = QTransform(*T9)
                    cuts[a] = self._plane_cuts(a, pl, T9)
            for kind, i in order:
                if kind == "quad":
                    self._draw_quad(p, quads[i], transforms, cuts)
                else:
                    self._draw_void(p, i, screen)
            self._drawn.reverse()                       # front first, for point_at
        self._draw_edges(p, corners, hidden, False)
        if not volume:                                  # the volume hides no outline: nothing to see through
            self._draw_xray(p, screen)
        self._draw_label(p)
        self._draw_triad(p)
        self._draw_scale(p)
        p.end()

    def _view_key(self) -> tuple:
        """What the volume's picture depends on besides the cube: the transfer function and the camera."""
        c = self.cam
        return (id(self.volume), self.transfer, c.yaw, c.pitch, c.scale, tuple(c.target), c.width, c.height,
                c.perspective)

    def _lut(self, plan: dict) -> np.ndarray:
        """The transfer function's table for this plan (its opacity corrected for the path through a slice, to 1 %)."""
        lo, hi, opacity, cmap, symmetric, gamma = self.transfer
        return VOL.transfer(self.volume["levels"], lo, hi, opacity, float(np.round(plan["exponent"], 2)), G.lut(cmap),
                            symmetric, gamma)

    def quick_picture(self) -> QImage | None:
        """The volume's quick picture as the view now stands: the coarse cube (volume3d.COARSE_MAX cells a side at
        most) on a picture FAST_SCALE of the widget, drawn while it turns and until the sharp one is ready; kept
        while only the overlays change. A cube too small for a coarse copy: itself."""
        vol = self.volume
        if vol is None or self.transfer is None or self.lo is None:
            return None
        self._sized()
        key = ("quick",) + self._view_key()
        if self._picture is not None and self._picture[0] == key:
            return self._picture[1]
        src = vol["coarse"] if vol.get("coarse") is not None else vol["fine"]
        plan = VOL.slice_plan(self.cam, self.lo, self.hi, src["shape"])
        skey = (id(src), plan["axis"], round(plan["exponent"], 2), self.transfer)
        hit = self._stacks.get("quick")
        if hit is None or hit[0] != skey:
            self._stacks.pop("quick", None)             # the old one freed before the new one is made
            hit = (skey, VOL.gather(self._lut(plan), VOL.slice_images(src, plan["axis"])))
            self._stacks["quick"] = hit
        img = render_slices(hit[1], plan, self.cam.width, self.cam.height, FAST_SCALE)
        self._picture = (key, img)
        return img

    def _ratio(self) -> float:
        try:
            return max(1.0, float(self.devicePixelRatioF()))
        except (AttributeError, RuntimeError):
            return 1.0

    def _sharp_ratio(self) -> float:
        """The sharp picture's pixels per widget pixel: the screen's own (2 on a Retina display), but no finer than
        SHARP_CELL_PX device pixels a cell (at the target's depth): beyond that a cell is only spread over more
        pixels, at four times the cost a doubling (a 512³ cube zoomed in)."""
        r = self._ratio()
        if self.volume is None or self.lo is None:
            return r
        cell = self.cam.scale * float(np.min((self.hi - self.lo) / np.asarray(self.volume["fine"]["shape"])))
        return float(min(r, max(1.0, SHARP_CELL_PX / max(cell, 1e-9))))

    def _sharp_key(self) -> tuple:
        return ("sharp", round(self._sharp_ratio(), 2)) + self._view_key()

    def _start_job(self) -> SharpPicture:
        """The sharp picture of the full cube at the screen's own resolution, begun (pump draws it)."""
        self._stop_job()
        src = self.volume["fine"]
        plan = VOL.slice_plan(self.cam, self.lo, self.hi, src["shape"])
        self._job = SharpPicture(self._sharp_key(), VOL.slice_images(src, plan["axis"]), self._lut(plan), plan,
                                 self.cam.width, self.cam.height, round(self._sharp_ratio(), 2))
        return self._job

    def _stop_job(self):
        if self._job is not None:
            self._job.cancel()
            self._job = None
        self._pump.stop()

    def _pump_tick(self):
        """A few more slices of the sharp picture (about 25 ms); shown when complete."""
        job = self._job
        if job is None or job.key != self._sharp_key() or self._fast:
            self._stop_job()
            return
        if job.pump(0.025):
            self._sharp = (job.key, job.img)
            self.last_volume_ms = job.ms
            self._job = None
            self._pump.stop()
            self.update()

    def volume_picture(self) -> QImage | None:
        """The volume's picture as the view now stands, at once (the tests, Save image): while turning the quick
        one, else the sharp one at the screen's resolution, its drawing finished here if it was under way."""
        vol = self.volume
        if vol is None or self.transfer is None or self.lo is None:
            return None
        self._sized()
        if self._fast:
            return self.quick_picture()
        key = self._sharp_key()
        if self._sharp is None or self._sharp[0] != key:
            job = self._job if self._job is not None and self._job.key == key else self._start_job()
            job.pump()
            self._sharp, self.last_volume_ms = (key, job.img), job.ms
            self._job = None
            self._pump.stop()
        return self._sharp[1]

    def _draw_volume(self, p: QPainter):
        """The volume: the sharp picture when it is ready for this view; else the quick one, the sharp one begun
        (not while turning) and drawn in ticks, shown when complete."""
        if self.volume is None or self.transfer is None:
            return
        img = None
        if not self._fast:
            key = self._sharp_key()
            if self._sharp is not None and self._sharp[0] == key:
                img = self._sharp[1]
            elif self._job is None or self._job.key != key:
                self._start_job()
                self._pump.start()
        if img is None:
            img = self.quick_picture()
        if img is None:
            return
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(QRectF(0.0, 0.0, float(self.cam.width), float(self.cam.height)), img)
        p.restore()

    def _plane_cuts(self, a: int, pl: dict, T9=None) -> list:
        """Where the voids cut plane a: [((x, y, L, row), dashed, screen)] in its texture's pixels, the cut being
        (x, y) + L (cos t, sin t) (view3d.plane_cuts: for spheres the map's outlines); dashed = the R_eff spheres;
        screen = in perspective the cut's ellipse on the screen through the plane's map T9 (view3d.cut_images: drawn
        with an affine map, not through the homography), else None."""
        v = self.voids
        if not v or self._fast:
            return []
        img = pl["img"]
        out = []
        for M, dashed in ((v["M"], False), (None if v.get("r_eff") is None else V.sphere_matrices(v["r_eff"]), True)):
            if M is None:
                continue
            cuts = V.plane_cuts(v["centres"], M, v["box"], a, pl["pos"], pl["lo2"], pl["hi2"], img.width(),
                                img.height())
            screens = V.cut_images(T9, cuts) if self.cam.perspective and T9 is not None else [None] * len(cuts)
            for (x, y, L, k), scr in zip(cuts, screens):
                out.append(((x, y, L, int(v["rows"][k])), dashed, scr))   # k: the row of the arrays; the catalogue's
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
            (x, y, L, _row), _dashed, _scr = c
            hx, hy = float(np.hypot(L[0, 0], L[0, 1])), float(np.hypot(L[1, 0], L[1, 1]))
            box = QRectF(x - hx, y - hy, 2 * hx, 2 * hy)
            if rect.contains(box):
                inside.append(c)
            elif rect.intersects(box):
                crossing.append(c)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        unit = QRectF(-1.0, -1.0, 2.0, 2.0)
        quad = None
        for group, clip in ((inside, False), (crossing, True)):
            if not group:
                continue
            if self.cam.perspective:                    # the quad's outline on the screen (a homography keeps lines)
                if clip and quad is None:
                    quad = QPainterPath()
                    quad.addPolygon(T.map(QPolygonF(rect)))
                    quad.closeSubpath()
                p.setTransform(QTransform())
                p.setClipPath(quad) if clip else p.setClipping(False)
            else:
                p.setTransform(T)
                p.setClipRect(rect) if clip else p.setClipping(False)     # the clip is kept in device space
            for (x, y, L, row), dashed, scr in group:
                if scr is not None:                     # perspective: the cut's own ellipse on the screen
                    (cx, cy), su, sv = scr
                    p.setTransform(QTransform(su[0], su[1], sv[0], sv[1], cx, cy))
                else:
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
        half faint, a unit circle through one transform per section (perspective: view3d.section_arcs' ellipse and
        its near arc)."""
        unit = QRectF(-1.0, -1.0, 2.0, 2.0)
        if self.cam.perspective:
            p.save()
            for s in which:
                xy, u, w, start, span = (arr[k] for arr in sections[s])
                if not np.all(np.isfinite([*xy, *u, *w, start, span])):
                    continue
                p.setTransform(QTransform(u[0], u[1], w[0], w[1], xy[0], xy[1]))
                far = np.sign(span) * (2 * np.pi - abs(span))
                # Qt's angles turn the other way in its y-down frame: the angle s is -s degrees in drawArc
                for a0, sp, alpha in ((start + span, far, far_alpha), (start, span, near_alpha)):
                    p.setPen(_pen(QColor(colour.red(), colour.green(), colour.blue(), alpha), 1.0))
                    p.drawArc(unit, int(round(-np.degrees(a0) * 16)), int(round(-np.degrees(sp) * 16)))
            p.restore()
            return
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
        xy, us, ws, sections, _reff = screen
        x, y, u, w, row = float(xy[k, 0]), float(xy[k, 1]), us[k], ws[k], int(v["rows"][k])
        hx, hy = float(np.hypot(u[0], w[0])), float(np.hypot(u[1], w[1]))       # the outline's half extents
        if not (np.isfinite(hx) and np.isfinite(hy) and hx > 0) or x + hx < 0 or y + hy < 0 \
                or x - hx > self.width() or y - hy > self.height():
            return
        sel, hov = row == self.selected, row == self.hover_row
        dim = self.focus and not (sel or hov)
        quiet = self.style == "volume" and not (sel or hov or dim)    # over the volume: the field stays in view
        colour = SELECTED if sel else HOVERED if hov else DIM if dim else QUIET if quiet else QColor("white")
        unit = QRectF(-1.0, -1.0, 2.0, 2.0)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.save()
        p.setTransform(QTransform(u[0], u[1], w[0], w[1], x, y))
        if dim or quiet:                                # a close-up's neighbour, or any void over the volume:
            p.setBrush(Qt.NoBrush)                      # a thin opaque outline, no more
            p.setPen(_pen(colour, 1.0))
            p.drawEllipse(unit)
            p.restore()
            if quiet:
                self._draw_reff(p, k, screen, colour)
            return
        if sel or hov:
            p.setPen(Qt.NoPen)
            alpha = (60 if sel else 45) if self.style != "volume" else (24 if sel else 18)   # the volume shows through
            p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), alpha))
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
        self._draw_reff(p, k, screen, colour)

    def _draw_reff(self, p: QPainter, k: int, screen, colour):
        """Void k's sphere of radius R_eff, dashed, when asked (orthographic: a circle of R_eff x scale)."""
        reff = screen[4]
        if reff is None:
            return
        r = self.voids["r_eff"][k]
        x, y = reff[0][k]
        u, w = reff[1][k], reff[2][k]
        if not (np.isfinite(r) and r > 0 and np.all(np.isfinite([x, y, *u, *w]))):
            return
        p.save()
        p.setTransform(QTransform(u[0], u[1], w[0], w[1], x, y))
        p.setBrush(Qt.NoBrush)
        p.setPen(_pen(colour, 1.0, Qt.DashLine))
        p.drawEllipse(QRectF(-1.0, -1.0, 2.0, 2.0))
        p.restore()

    def _draw_xray(self, p: QPainter, screen):
        """The inspected void seen through the planes: its outline and principal sections over everything (a
        close-up's void sits where the planes cross, behind the ones in front)."""
        if screen is None or self.selected is None:
            return
        hit = np.nonzero(np.asarray(self.voids["rows"]) == self.selected)[0]
        if not len(hit):
            return
        xy, us, ws, sections, _reff = screen
        k = int(hit[0])
        x, y, u, w = float(xy[k, 0]), float(xy[k, 1]), us[k], ws[k]
        if not np.all(np.isfinite(u) & np.isfinite(w) & np.isfinite(xy[k])):
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
        """A bar of a round length in Mpc, in the lower-right corner (perspective: true at the target's depth, the
        middle of the window when fitted)."""
        span = self.width() / self.cam.scale
        length = G.nice_length(span)
        px = length * self.cam.scale
        if not (8 <= px <= self.width() - 40):
            return
        fm = QFontMetrics(p.font())
        label = f"{length:g} Mpc" + (" at the centre" if self.cam.perspective else "")
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
        v.addWidget(self._note("The map's field as a translucent volume (or on three planes) with the voids as "
                               "spheres of their true size. Drag to turn it, right-drag (or shift-drag) to move it, "
                               "scroll to zoom. Click a void to inspect it; double-click it (or Close up) to look at "
                               "it from near."))
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
        self.view3d_style = QComboBox()
        for text, style in STYLE_ITEMS:
            self.view3d_style.addItem(text, style)
        self.view3d_style.setToolTip(STYLE_TIP)
        self.view3d_style.currentIndexChanged.connect(self._3d_style_changed)
        row.addWidget(self.view3d_style)
        self.view3d_persp = QCheckBox("perspective")
        self.view3d_persp.setChecked(True)
        self.view3d_persp.setToolTip(PERSPECTIVE_TIP)
        self.view3d_persp.toggled.connect(self._3d_perspective)
        row.addWidget(self.view3d_persp)
        self.planes3d_box = QWidget()                   # the planes' own controls, shown with the planes
        pr = QHBoxLayout(self.planes3d_box)
        pr.setContentsMargins(0, 0, 0, 0)
        self.view3d_axes = []
        for a in range(3):
            cb = QCheckBox(V.AXES[a])
            cb.setChecked(True)
            cb.setToolTip(f"The plane normal to {V.AXES[a]}")
            cb.toggled.connect(lambda _on: self._3d_request())
            pr.addWidget(cb)
            self.view3d_axes.append(cb)
        self.view3d_opacity = QSpinBox(minimum=10, maximum=100, value=90, singleStep=10)
        self.view3d_opacity.setSuffix(" % opaque")
        self.view3d_opacity.setToolTip("The planes' opacity: below 100 % a sphere behind a plane shows through it")
        self.view3d_opacity.valueChanged.connect(self._3d_opacity)
        pr.addWidget(self.view3d_opacity)
        row.addWidget(self.planes3d_box)
        self.view3d_detail = QComboBox()
        for d in VOL.DETAILS:
            self.view3d_detail.addItem(f"{d}³ cells", d)
        self.view3d_detail.setCurrentIndex(self.view3d_detail.findData(VOL.DEFAULT_DETAIL))
        self.view3d_detail.setToolTip(DETAIL_TIP)
        self.view3d_detail.currentIndexChanged.connect(lambda _i: self._3d_request())
        row.addWidget(self.view3d_detail)
        row.addStretch(1)
        v.addLayout(row)
        self.volume3d_box = QWidget()                   # the volume's transfer function, shown with the volume
        vb = QVBoxLayout(self.volume3d_box)
        vb.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Shown from"))
        self.vol3d_lo = QSlider(Qt.Horizontal, minimum=0, maximum=1000)
        self.vol3d_lo_text = QLabel("")
        self.vol3d_hi = QSlider(Qt.Horizontal, minimum=0, maximum=1000, value=1000)
        self.vol3d_hi_text = QLabel("")
        for w_ in (self.vol3d_lo, self.vol3d_hi):
            w_.setToolTip(RANGE_TIP)
            w_.valueChanged.connect(self._3d_tf_moved)
        for lab in (self.vol3d_lo_text, self.vol3d_hi_text):
            lab.setMinimumWidth(QFontMetrics(lab.font()).horizontalAdvance("0.000e+00") + 4)
        row.addWidget(self.vol3d_lo, 1)
        row.addWidget(self.vol3d_lo_text)
        row.addWidget(QLabel("to"))
        row.addWidget(self.vol3d_hi, 1)
        row.addWidget(self.vol3d_hi_text)
        vb.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Opacity"))
        self.vol3d_opacity = QSlider(Qt.Horizontal, minimum=0, maximum=100)
        self.vol3d_opacity.setValue(self._3d_opacity_position(VOL.DEFAULT_OPACITY))
        self.vol3d_opacity.setToolTip(OPACITY_TIP)
        self.vol3d_opacity.valueChanged.connect(self._3d_opacity_moved)
        self.vol3d_opacity_text = QLabel("")
        self.vol3d_opacity_text.setMinimumWidth(QFontMetrics(self.vol3d_opacity_text.font()).horizontalAdvance("100 %")
                                                + 4)
        row.addWidget(self.vol3d_opacity, 1)
        row.addWidget(self.vol3d_opacity_text)
        self.vol3d_default = QPushButton("Default range")
        self.vol3d_default.setToolTip("Back to the range the volume opened with (a density: from ρ/ρ̄ = 1 to its "
                                      "99.9th percentile) and the default opacity")
        self.vol3d_default.clicked.connect(self._3d_tf_default)
        row.addWidget(self.vol3d_default)
        vb.addLayout(row)
        v.addWidget(self.volume3d_box)
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
        self._3d_inflight = None                # the request on the slice worker (its progress, its cancel)
        # the volume's shown range on the map's scale and what it is of: {'of': (field, component, log, symmetric),
        # 'lo', 'hi'}; a volume of another field (or scale) opens at its own default range
        self._3d_tf = None
        self._3d_ticker = QTimer(self, interval=200)   # the averaging's progress, while it runs
        self._3d_ticker.timeout.connect(self._3d_progress)
        v3 = self.view.view3d
        v3.hovered.connect(self._3d_hover)
        v3.clicked.connect(self._3d_click)
        v3.double_clicked.connect(self._3d_double)
        v3.cam.perspective = self.view3d_persp.isChecked()
        self._3d_style_changed()
        self._3d_apply_tf()                     # the opacity's text
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
            view.view3d.message = "averaging the grid…" if self._3d_volume_style() else "reading the planes…"
            if self.voids.isChecked() and not self.voids_applicable(self.current):
                self._voids_request(self.current)
            self._3d_follow()
        else:
            self._3d_shown = False
            self._3d_cancel()
            view.view3d.set_volume(None)            # its coloured slices (128 MB at 256³) freed; the averaged
            view.view3d.setVisible(False)           # cube stays in volume3d's cache: shown again at once
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

    def _3d_volume_style(self) -> bool:
        return self.view3d_style.currentData() == "volume"

    def _3d_request(self):
        """Read the planes or the volume (on the slice worker; only the latest request is served): the map's field,
        component and colours; the planes with its range clip, fixed range and smoothing, the volume averaged down
        to the detail chosen (its range its own: the sliders); never the map's comparison or hi-res slice. A volume
        already shown or on its way is not read again: a change of colours only recolours it."""
        if not self._3d_shown or self._shut or self.current is None or self._3d_index is None:
            return
        o, name = self.current, self.field.currentData()
        if name is None:
            return
        log = self.log.isChecked()
        lo, hi = self._3d_window(o)
        style = self.view3d_style.currentData()
        req = {"output": o, "field": name, "component": self.component.currentText(), "cmap": self.cmap.currentText(),
               "log": log, "clip": self.clip.value(), "smooth": self.smooth.value(),
               "symmetric": name in G.DIVERGING and not log,
               "ps_gradient": self.ps_gradient.currentData() or G.PS_GRADIENT_DEFAULT,
               "index": list(self._3d_index), "shown": [cb.isChecked() for cb in self.view3d_axes],
               "lo": np.asarray(lo, dtype=np.float64), "hi": np.asarray(hi, dtype=np.float64), "style": style}
        if style == "volume":
            req["want"] = int(self.view3d_detail.currentData())
            req["detail"], req["note"] = self._3d_detail(o, req["lo"], req["hi"])
            req["progress"] = {}
            # the files' identity (mtime, size) too: a grid written again by a re-run is read again
            files = tuple(G._cube_key(o.files[s].path, o.dtype, VOL._file_shape(o, s)) for s in VOL._sources(o, name))
            req["vkey"] = (str(o.root), files, name, req["component"], bool(log), tuple(np.round(req["lo"], 9)),
                           tuple(np.round(req["hi"], 9)), req["detail"])
            shown = self._3d_res is not None and self._3d_res.get("vkey") == req["vkey"] \
                and self.view.view3d.volume is not None
            inflight = self._3d_inflight or {}
            flying = self._3d_busy and inflight.get("vkey") == req["vkey"] \
                and not (inflight.get("progress") or {}).get("cancel")      # a cancelled one never arrives
            if shown or flying:
                self._3d_pending = None             # anything queued is older than this: nothing more to read
                if self._3d_busy and not flying:
                    self._3d_cancel()               # back to what is shown: the volume under way is moot
                if shown:
                    self._3d_res["request"].update(cmap=req["cmap"], note=req["note"])
                    self._3d_apply_tf()
                return
        fixed = self._fixed_range(log)
        if fixed is not None:
            req["vrange"] = fixed
        if self._3d_busy:
            self._3d_pending = req
            if (self._3d_inflight or {}).get("vkey") != req.get("vkey"):
                self._3d_cancel()                   # an averaging of another field or window: moot
            return
        self._3d_emit(req)

    def _3d_detail(self, o, lo, hi) -> tuple[int, str]:
        """The detail chosen, and why not when it is not: a cube that would not fit in the free memory less 2 GB (the
        volume shown, given back when this one replaces it, counted as free) falls back to the finest that does; one
        needing more than 1 GB while a DTFE run is going is asked about first (heavy_ok: the run's auto-tuner
        budgeted from the memory free at its start), a 'no' falling back to one under 1 GB. The choice shown now
        for this window is kept as it is: a recolour or a slice step never re-reads it at another detail."""
        want = int(self.view3d_detail.currentData())
        shown = (self._3d_res or {}).get("request") or {}
        if shown.get("style") == "volume" and shown.get("want") == want and self._same(shown.get("output"), o) \
                and np.allclose(shown["lo"], lo) and np.allclose(shown["hi"], hi):
            return int(shown["detail"]), shown.get("note") or ""
        need = VOL.need_bytes(VOL.window_shape(o, lo, hi, want))
        if need <= 1e9:
            return want, ""
        smaller = sorted((d for d in VOL.DETAILS if d < want), reverse=True)

        def fitting(room: float) -> int:
            return next((d for d in smaller if VOL.need_bytes(VOL.window_shape(o, lo, hi, d)) <= room),
                        min(VOL.DETAILS))
        free = G.available_gb()
        prep = (self._3d_res or {}).get("volume")
        held = VOL._prepared_bytes(prep) if prep is not None else 0
        if free is not None and need > (free - 2.0) * 1e9 + held:
            d = fitting((free - 2.0) * 1e9 + held)
            return d, f"{want}³ would need {need / 1e9:.1f} GB with {free:.1f} GB free: {d}³ instead"
        if not self.heavy_ok("3D volume", f"A {want}³ volume in the 3D view", need / 1e9):
            d = fitting(1e9)
            return d, f"a DTFE run is going: {d}³ instead of {want}³"
        return want, ""

    def _3d_perspective(self, on: bool):
        """The perspective box: the view's camera, fitted again as it is turned."""
        self.view.view3d.set_perspective(on)

    def _3d_emit(self, req: dict):
        self._3d_busy = True
        self._3d_inflight = req
        self._threads()
        if req.get("style") == "volume":
            self._3d_ticker.start()
        self._to_render3d.emit(req)

    def _3d_cancel(self):
        """Stop the averaging on the slice worker (its result is not wanted)."""
        prog = (self._3d_inflight or {}).get("progress")
        if self._3d_busy and prog is not None:
            prog["cancel"] = True

    def _3d_progress(self):
        """The averaging's progress in the state line (and on the empty view)."""
        prog = (self._3d_inflight or {}).get("progress") or {}
        if not self._3d_busy:
            self._3d_ticker.stop()
            return
        if not prog.get("total") and prog.get("stage") != "colouring":
            return
        o = self._3d_inflight["output"]
        text = ("colouring the volume…" if prog.get("stage") == "colouring" else
                f"averaging the {o.n}³ grid to the volume: {100.0 * prog.get('done', 0) / prog['total']:.0f} %")
        text += " (the close-up's cube)" if self._3d_close is not None else ""
        self.view3d_state.setText(text)
        v3 = self.view.view3d
        if v3.volume is None:
            v3.message = text
            v3.update()

    def _3d_rendered(self, res: dict):
        self._3d_busy = False
        self._3d_inflight = None
        if self._3d_pending is not None:
            req, self._3d_pending = self._3d_pending, None
            self._3d_emit(req)
        req = res["request"]
        if not self._3d_shown or self._shut or not self._same(req["output"], self.current) \
                or req.get("style", "planes") != self.view3d_style.currentData():
            return
        if (req.get("progress") or {}).get("cancel"):  # made moot while it was coloured: what is shown stays
            if self._3d_res is not None:
                self._3d_state()
            return
        if req.get("style") == "volume":
            self._3d_volume_rendered(res)
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
        failed, self._3d_inflight = self._3d_inflight, None
        if self._3d_pending is not None:
            req, self._3d_pending = self._3d_pending, None
            self._3d_emit(req)
            return
        if msg.startswith(VOL.Cancelled.__name__):     # stopped on purpose: nothing went wrong
            if self._3d_shown and self._3d_res is not None:
                self._3d_state()
            return
        if self._3d_shown:
            volume = (failed or {}).get("style") == "volume"
            self.view.view3d.message = f"Could not read the {'volume' if volume else 'planes'}: {msg}"
            if volume:
                self.view.view3d.set_volume(None)
                self._3d_res = None
            else:
                self.view.view3d.set_scene(None, None, None)
            self.view3d_state.setText(self.view.view3d.message)

    def _3d_opacity(self, value: int):
        self.view.view3d.opacity = value / 100.0
        self.view.view3d.update()

    # ---------------------------------------------------------------- the volume
    def _3d_style_changed(self, *_):
        """Volume or planes: the controls of the one shown, the view's style, a fresh read."""
        volume = self._3d_volume_style()
        self.planes3d_box.setVisible(not volume)
        self.volume3d_box.setVisible(volume)
        self.view3d_detail.setVisible(volume)
        v3 = self.view.view3d
        v3.style = "volume" if volume else "planes"
        if not volume and v3.volume is not None:
            v3.set_volume(None)                     # its coloured slices freed (the averaged cube stays cached)
        if volume:
            v3.planes = None                        # never drawn against another window when the planes come back
        if self._3d_shown:
            if volume and v3.volume is None:
                v3.message = "averaging the grid…"
            elif not volume and not v3.planes:
                v3.message = "reading the planes…"
            self._3d_res = None                     # the other style's result: its title and state no longer hold
            self._3d_request()
        v3.update()

    def _3d_volume_rendered(self, res: dict):
        """A volume read: drawn with the shown range (kept for the same field and scale, else the field's default)
        and the map's colours; the title, the colour bar (the shown range) and the state line say what it is."""
        req, prep = res["request"], res["volume"]
        self._3d_res = res
        v3 = self.view.view3d
        v3.set_volume(prep, res["lo"], res["hi"])
        of = (req["field"], req["component"], bool(req["log"]), bool(req["symmetric"]))
        if self._3d_tf is None or self._3d_tf["of"] != of:
            lo, hi = VOL.default_range(prep, req["field"], req["symmetric"])
            self._3d_tf = {"of": of, "lo": lo, "hi": hi}
        self._3d_sliders()
        self._3d_apply_tf()
        if self._3d_refit:
            self._3d_refit = False
            v3.fit()
        self._3d_voids()
        self.changed.emit()

    def _3d_span(self) -> tuple[float, float]:
        """The sliders' span on the map's scale: the volume's whole range (|value| from 0 when symmetric)."""
        res = self._3d_res or {}
        prep = res.get("volume")
        if prep is None:
            return 0.0, 1.0
        qlo, qhi = prep["qrange"]
        if res["request"]["symmetric"]:
            return 0.0, max(abs(qlo), abs(qhi), 1e-30)
        return float(qlo), float(qhi)

    def _3d_sliders(self):
        """The range sliders at the shown range (no signal: nothing redrawn)."""
        a, b = self._3d_span()
        for sl, val in ((self.vol3d_lo, self._3d_tf["lo"]), (self.vol3d_hi, self._3d_tf["hi"])):
            sl.blockSignals(True)
            sl.setValue(int(round(np.clip((val - a) / (b - a), 0.0, 1.0) * sl.maximum())))
            sl.blockSignals(False)

    @staticmethod
    def _3d_opacity_position(opacity: float) -> int:
        lo, hi = VOL.OPACITY_RANGE
        return int(round(100.0 * np.log(np.clip(opacity, lo, hi) / lo) / np.log(hi / lo)))

    def _3d_opacity_value(self) -> float:
        lo, hi = VOL.OPACITY_RANGE
        return float(lo * (hi / lo) ** (self.vol3d_opacity.value() / 100.0))

    def _3d_value_text(self, s: float) -> str:
        """A value of the shown range as the map's colour bar would say it: a log map's 10^s."""
        req = (self._3d_res or {}).get("request") or {}
        v = 10.0 ** s if req.get("log") else s
        return f"{v:.3g}"

    def _3d_opacity_moved(self, *_):
        """The opacity slider: the opacity only (the range stays exactly as it is); the coarse picture while it
        moves."""
        self._3d_apply_tf()
        if (self._3d_res or {}).get("volume") is not None:
            self.view.view3d._interacting()

    def _3d_tf_moved(self, *_):
        """A range slider moved: its end of the range follows it (the other end only when pushed: the low end never
        past the high one); the coarse picture while it moves."""
        if self._3d_tf is None or (self._3d_res or {}).get("volume") is None:
            self._3d_apply_tf()
            return
        a, b = self._3d_span()
        lo_k, hi_k = self.vol3d_lo.value(), self.vol3d_hi.value()
        mover = self.sender()
        pushed = lo_k >= hi_k
        if pushed:                                  # pushed against each other: the one not moved gives way
            if mover is self.vol3d_hi:
                lo_k = max(0, hi_k - 1)
            else:
                hi_k = min(self.vol3d_hi.maximum(), lo_k + 1)
                lo_k = hi_k - 1
            for sl, k in ((self.vol3d_lo, lo_k), (self.vol3d_hi, hi_k)):
                sl.blockSignals(True)
                sl.setValue(k)
                sl.blockSignals(False)
        value = lambda k, sl: a + k / sl.maximum() * (b - a)       # noqa: E731
        if mover is not self.vol3d_hi or pushed:
            self._3d_tf["lo"] = value(lo_k, self.vol3d_lo)
        if mover is self.vol3d_hi or pushed:
            self._3d_tf["hi"] = value(hi_k, self.vol3d_hi)
        self._3d_apply_tf()
        self.view.view3d._interacting()

    def _3d_tf_default(self):
        res = self._3d_res or {}
        if res.get("volume") is None:
            return
        req = res["request"]
        lo, hi = VOL.default_range(res["volume"], req["field"], req["symmetric"])
        self._3d_tf.update(lo=lo, hi=hi)
        self._3d_sliders()
        self.vol3d_opacity.blockSignals(True)
        self.vol3d_opacity.setValue(self._3d_opacity_position(VOL.DEFAULT_OPACITY))
        self.vol3d_opacity.blockSignals(False)
        self._3d_apply_tf()

    def _3d_apply_tf(self):
        """The shown range, opacity and the map's colours to the view; the slider texts, the colour bar, the title
        and the state line with them."""
        opacity = self._3d_opacity_value()
        pct = 100.0 * opacity
        self.vol3d_opacity_text.setText(f"{pct:.0f} %" if pct >= 10 else f"{pct:.2g} %")
        res = self._3d_res or {}
        if res.get("volume") is None or self._3d_tf is None:
            return
        req, tf = res["request"], self._3d_tf
        cmap = self.cmap.currentText()
        self.view.view3d.set_transfer(tf["lo"], tf["hi"], opacity, cmap, req["symmetric"])
        self.vol3d_lo_text.setText(self._3d_value_text(tf["lo"]))
        self.vol3d_hi_text.setText(self._3d_value_text(tf["hi"]))
        label = self._gradient_label(dict(req, ps_gradient="stored"))      # the volume is of the stored grid
        comp = "" if req["component"] in ("value", "") else f" [{req['component']}]"
        vr = (-tf["hi"], tf["hi"]) if req["symmetric"] else (tf["lo"], tf["hi"])
        self.view.colorbar.set(cmap, vr, ("log₁₀ " if req["log"] else "") + label + comp
                               + (" (|value| shown)" if req["symmetric"] else ""), req["log"])
        self._3d_state()

    def _3d_state(self):
        """The title and the state line of the volume shown: its cells, how many of the grid's each averages, how
        much of it shows."""
        res = self._3d_res or {}
        prep = res.get("volume")
        if prep is None:
            return
        req, o = res["request"], res["request"]["output"]
        label = self._gradient_label(dict(req, ps_gradient="stored"))
        comp = "" if req["component"] in ("value", "") else f" [{req['component']}]"
        where = f"close-up on void {self._3d_close['row']}" if self._3d_close is not None else "the whole box"
        m = prep["shape"]
        cells = "×".join(str(int(k)) for k in m) if len(set(m)) > 1 else f"{int(m[0])}³"
        self.view.title.setText(f"<b>{label}{comp}</b> — {o.title} — 3D, {where}: a volume of {cells} cells")
        block = [c / k for c, k in zip(prep["cells"], m)]
        each = "the grid's own cells" if max(block) <= 1.0 + 1e-9 else \
            ("each the mean of " + (f"{block[0]:.0f}³" if np.allclose(block, np.round(block[0]))
                                    else "×".join(f"{b:.2g}" for b in block)) + " grid cells")
        frac = VOL.visible_fraction(prep, self._3d_tf["lo"], req["symmetric"]) if self._3d_tf else 0.0
        note = req.get("note") or ""
        self.view3d_state.setText(f"volume {cells}, {each} · {100.0 * frac:.3g} % of them above the low end"
                                  + (f" · {note}" if note else ""))

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
        """The 3D view as it is on screen (its pixels at the screen's resolution) to a PNG: the volume's settled
        picture finished first (not the quick one still shown while it is drawn)."""
        v3 = self.view.view3d
        if v3.style == "volume" and v3.volume is not None:
            v3._fast = False
            v3.volume_picture()
        if not v3.grab().save(str(path)):
            raise OSError(f"could not write {path}")
        return str(path)
