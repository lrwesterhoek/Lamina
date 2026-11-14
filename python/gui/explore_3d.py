"""The Explore tab's 3D view: View3D, the right pane's picture of the box turned in 3D -- the map's field as a
translucent volume (or on three orthogonal planes), the voids as spheres of their true size, the box's edges, an axis
triad and a scale bar -- and View3DMixin, the part of ExploreControls that fills it (the volume or the planes read on
the slice worker, planes3d), follows the map's field, colours and slice, and inspects the voids: hover names one, a
click fills the void inspector, a double-click (or Close up) flies to it, the volume or the planes then read from the
grid at its full resolution around it.

Drawn by QPainter: an orthographic camera (view3d.py) maps each plane, and each slice of the volume, to the screen by
one affine transform and each sphere to a circle. The volume is direct volume rendering by texture slicing
(volume3d.py): the averaged cube's slices along the axis facing the viewer, coloured and made translucent by a
transfer function on the map's scale, composited front to back in 16 bits a channel. On screen, where OpenGL is
there, the volume's picture comes from the GPU instead (volume_gl.py: the same slices, samples and compositing, in
real time at 1024^3); offscreen (the tests, which have no OpenGL), for Save image and without OpenGL from QPainter
(volume_picture), the overlays always from QPainter.

With the map's Contours tick the planes carry the map's contour lines (each plane's own, mapped by its transform)
and the volume the levels as thin shells (contours.shell_plan, in its transfer function), with their key."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import (QBitmap, QColor, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPolygonF, QRegion,
                           QTransform)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

import contours as CT
import grids as G
import view3d as V
import volume3d as VOL
import volume_gl as VGL
from app_widgets import button_row, flush, tip, tip_auto
from explore_canvas import segments_path

VIEW3D_TIP = ("Show the map's field in 3D: the whole box as a translucent volume (or on three planes), with the "
              "voids at their true size. The volume: the field averaged down to the detail chosen, its "
              "low values see-through and its high values opaque in the map's colours; the planes: the map's own "
              "slice and the middle of the other two. The voids as the Map group's 'Voids' row draws them: their "
              "ellipsoids at the mean density (the planes show the same sections as the map), or spheres of R_v; "
              "the other outline dashed when ticked there. A void near a face is drawn once, at its catalogue "
              "centre, and may reach past the box. Any change to the map's field or colours follows (the planes: "
              "its slice too); the voids need 'Voids' ticked.")
STYLE_ITEMS = (("volume", "volume"), ("planes", "planes"))
STYLE_TIP = ("Volume: the field as a translucent cloud; planes: the map's slice and the middle of the other two "
             "axes. The volume is the field of the whole box (or a close-up's cube) averaged down to the detail "
             "chosen: cells below the low end of the shown range are transparent, the opacity rises to the high "
             "end on the map's scale (log for a density), coloured with the map's colours over the same range; its "
             "range is its own (the sliders below), the map's smoothing and range clip are the planes'. "
             "The planes are drawn at the grid's own resolution.")
DETAIL_TIP = ("The volume's cells a side, at most: the grid is averaged over blocks of its cells. (A 1024³ grid in "
              "blocks of 4³ at 256³, of 2³ at 512³, its own cells at 1024³; a smaller grid or close-up: its own "
              "cells at most.) A close-up's cube is read from the grid again at this detail: finer. Drawn on the GPU "
              "(where OpenGL is: said under the controls): at rest every slice at the screen's resolution, about "
              "0.1 s for 1024³, while the view turns the cube's half (2³ cells averaged) or every few slices, kept "
              "over 30 frames a second. Without it a coarse copy (160³ at most) while turning, and at rest a second "
              "or two for 1024³. 1024³ also needs some 10 GB of memory while it is read and 1.1 GB on the GPU (less "
              "free: a smaller detail, said under the controls).")
PERSPECTIVE_TIP = ("Perspective: the box seen from an eye at a finite distance, its near side drawn larger, as the eye "
                   "expects (without it a box reads as widening toward the back). The voids' outlines are then true "
                   "in shape but their size depends on their depth, and the scale bar holds at the window's centre. "
                   "Zooming in moves the eye toward what is under the cursor: into the box and on through the "
                   "field, which is then drawn all around from where the eye is (zoom out to back away again). "
                   "Off: a parallel projection, every void a circle of its true size against the box; zooming "
                   "magnifies the box seen from outside.")
RANGE_TIP = ("The shown range on the map's scale: below its low end a cell is transparent, from there the opacity "
             "rises to its high end; the colour spans the same range. A density's range opens at ρ/ρ̄ = 1: its "
             "voids and underdense walls are see-through. A diverging field (velocity divergence, eigenvalues) is "
             "shown by |value|.")
OPACITY_TIP = ("How opaque the densest cells are: the opacity of a cell at the high end of the range, 1/128 of the "
               "window's side thick. The same look at every detail, close-up and turn.")
FAST_SCALE = 0.5                # while the view turns: the coarse cube on a half-size picture
SHARP_CELL_PX = 3.0             # the settled picture: no finer than this many device pixels a cell
PREVIEW_SCALE = 0.5             # a large cube settles in two steps: first at half the settled picture's pixels a side
PREVIEW_CELLS = 384             # (a cube of at least this many cells along an axis: 512^3 and 1024^3), then whole --
                                # at 1024^3 on a Retina screen the first in 1.25 s instead of the whole one's 2.1
TURN_BUDGET_MS = 20.0           # the GPU: a frame's drawing while the view turns, at most (the read-back, the overlays
                                # and the window's flush add some 5-8 ms: over 30 frames a second)
TURN_STEPS = ((0, 1), (0, 2), (1, 1), (1, 2), (1, 4), (1, 8))
                                # ... the turning frames' steps, finer (dearer) first: (cube, stride) = every stride-th
                                # slice (the opacity corrected for the longer path: the same look) of the volume (0)
                                # or of its half, 2^3 cells averaged (1: no knot hit or missed); without the half,
                                # (0, 2 x stride)
CLOSE_TIP = ("Look at the chosen void from near: the camera flies to it and the three planes pass through its centre. "
             "They are cut to a cube around it at the grid's full resolution, read periodically across the box's "
             "faces, with the neighbouring voids that reach into it")
BG = QColor(26, 27, 31)                     # the 3D view's background: white outlines read on it
SELECTED = QColor(255, 196, 64)
HOVERED = QColor(120, 220, 255)
SHAPE_ITEMS = (("as ellipsoids", "ellipsoid"), ("as spheres of R_v", "sphere"))
SHAPE_TIP = ("How the voids are drawn on the map and in the 3D view: as their ellipsoids, or as spheres of R_v. "
             "Ellipsoids: around each density minimum the quadratic fit of its curvature (the Hessian's eigenvalues "
             "and eigenvectors), cut where it returns to the mean density: semi-axes a >= b >= c along their own "
             "directions, R_ell = (a b c)^⅓. The map draws the ellipse where it cuts the slice and a dot at the "
             "minimum; the 3D view the ellipsoid, all three principal sections on the void inspected, and the planes "
             "where each one cuts them. A void without such an ellipsoid (its minimum not below the mean) is its dot "
             "alone. Spheres of R_v: where the spherical average of the smoothed density returns to the mean, drawn "
             "as before. A close-up sizes its cube by R_ell, or R_v, and always holds the whole shape.")
MARK_CELLS = 1.5                # a void without an ellipsoid: a sphere of this many cells at its centre (its mark)
EDGE = QColor(12, 12, 14)                   # the dark edge under an outline: it reads over a bright plane
DIM = QColor(120, 122, 128)                 # a close-up's neighbours: opaque grey (a translucent line is slow)
QUIET = QColor(165, 190, 205)               # the voids over the volume: a thin, cool line apart from its warm colours
# Many voids in one view (the whole box: some 250 at the production smoothing) cover the screen two or three times
# over: as outlines they are a hairball (looked at on TNG300-3-Dark). Above CROWD_VOIDS (and not in a close-up) every
# void but the hovered and the inspected one is a faint fill and a dot at its centre (over the volume, where the fill
# cannot be seen, also its outline, thin and faint): where the voids are, their shapes, and which one the mouse is
# on, read at a glance; the planes keep each void's section, as the map draws it.
CROWD_VOIDS = 40
CROWD_FILL_ALPHA = {"planes": 20, "volume": 8}  # each void's fill (of 255): overlaps add up, a lone void stays faint;
                                                # over the volume fainter (the field's own picture must not wash out)
CROWD_LINE_ALPHA = {"planes": 0, "volume": 90}  # each void's outline (of 255) over the volume, where the fill cannot be
                                                # seen: thin and faint, so the ellipsoids read without a hairball
CROWD_DOT_PX = 1.8
AXIS_COLOURS = (QColor(235, 95, 85), QColor(110, 205, 105), QColor(95, 150, 245))
SHELL_COPY_CELLS = 512 ** 3     # a close-up's cube of up to this many cells gets a copy of its levels with its own
                                # void's shells only (two bytes a cell more); a larger one shows every void's shells
VOID_FLOOR_SHARE = 0.3          # a close-up's contour shells start at the lowest density within this share of the
                                # void's smallest radius from its centre (the catalogue's centre is the minimum of the
                                # smoothed field; the grid's own lies near it)


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
    o: G.OutputSet = req["output"]
    n = int(o.n)
    cell = o.length / n
    idx, wlo, whi = V.cell_window(req["lo"], req["hi"], o.lo, cell, n)
    raw = [None, None, None]
    for a in range(3):
        if not req["shown"][a]:
            continue
        plane = G.map_plane(o, req["field"], a, req["index"][a], req["component"], req["ps_gradient"], req["smooth"],
                            bool(req.get("smooth3d")))          # smoothed as the map is: in 2D, or the cube in 3D
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
    bar = None                                          # the contour levels for the colour bar (shared: one range)
    for a, item in enumerate(raw):
        if item is None:
            continue
        vals, s = item
        r0, r1 = V.rest_axes(a)
        img = np.ascontiguousarray(vals.T[::-1])        # the map's image: columns along r0, rows r1 downward
        rgb, shown = G.colorize(img, req["cmap"], log=req["log"], symmetric=req["symmetric"], vrange=vrange)
        lo2 = (float(wlo[r0]), float(wlo[r1]))
        hi2 = (lo2[0] + vals.shape[0] * s * cell[r0], lo2[1] + vals.shape[1] * s * cell[r1])
        pos = V.plane_position(req["index"][a], o.lo[a], cell[a], wlo[a], whi[a], float(o.length[a]))
        planes[a] = {"rgb": np.ascontiguousarray(rgb), "values": vals, "lo2": lo2, "hi2": hi2, "pos": pos,
                     "stride": s}
        if req.get("contours"):                         # the map's lines on this plane, in its texture's pixels
            lines = CT.map_lines(vals, req["field"], shown, req["log"], req["cmap"],
                                 detail=req.get("contour_detail", CT.DETAIL))
            if lines is not None:
                planes[a]["lines"] = {"segments": CT.image_segments(lines),
                                      "rgb": CT.solid_colours(lines["levels"], shown, req["cmap"])}
                bar = bar or dict(lines, segments=None)
    return {"request": req, "planes": planes, "lo": wlo, "hi": whi, "vrange": vrange, "contours": bar}


def _canvas(width: int, height: int, scale: float) -> tuple[QImage, QTransform]:
    """A transparent 16-bit picture of a width x height view at 'scale' of it, and the map from view to its pixels."""
    iw, ih = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    img = QImage(iw, ih, QImage.Format_RGBA64_Premultiplied)
    img.fill(Qt.transparent)
    return img, QTransform.fromScale(iw / max(width, 1), ih / max(height, 1))


def _region(mask: np.ndarray) -> QRegion:
    """The region of a picture's pixels where the boolean (h, w) 'mask' is set."""
    h, w = mask.shape
    stride = ((w + 31) // 32) * 4                   # a 1-bit image's rows are padded to 32 bits
    bits = np.zeros((h, stride), dtype=np.uint8)
    packed = np.packbits(mask, axis=1)
    bits[:, :packed.shape[1]] = packed
    img = QImage(bits.data, w, h, stride, QImage.Format_Mono)
    img.setColor(0, 0xffffffff)                     # a bitmap's region is its black (1) pixels
    img.setColor(1, 0xff000000)
    return QRegion(QBitmap.fromImage(img))          # (fromImage copies: 'bits' may go)


def part_regions(plan: dict, width: int, height: int) -> list:
    """The parts of a slice plan that draw pixels of a width x height picture of its view, each with the region of
    its pixels: [(part, QRegion)] (None for a part that draws all, as orthographic's one does)."""
    ids = VOL.part_ids(plan, width, height)
    if ids is None:
        return [(q, None) for q in plan["parts"]]
    out = []
    for q in plan["parts"]:
        mask = ids == VOL.part_id(q)
        if len(q["order"]) and mask.any():
            out.append((q, None if mask.all() else _region(mask)))
    return out


def _faded(px: np.ndarray, weight: float) -> np.ndarray:
    """RGBA64 premultiplied pixels ((h, w) uint64) at 'weight' of their colour and opacity (a copy)."""
    c = np.ascontiguousarray(px).view(np.uint16)
    return np.rint(c * np.float32(weight)).astype(np.uint16).view(np.uint64)


def _composite(img: QImage, S: QTransform, part: dict, slices, region: QRegion | None = None) -> None:
    """Draw (index, (h, w) uint64 RGBA64 pixels) slices of a plan's part onto img, front first, with the 'under'
    operator (DestinationOver), each by its transform (then S) and smoothly (bilinear) resampled; only within
    'region' (img's pixels: the part's own, part_regions) when given. The part's nearest slice at its weight
    (slice_plan's 'first')."""
    w, h = part["size"]
    first = part.get("first")
    q = QPainter(img)
    q.setCompositionMode(QPainter.CompositionMode_DestinationOver)
    q.setRenderHint(QPainter.SmoothPixmapTransform, True)
    if region is not None:
        q.setClipRegion(region)                     # set before any transform: in img's pixels, and it stays
    for i, px in slices:
        if first is not None and int(i) == first[0] and first[1] < 1.0:
            px = _faded(px, first[1])
        q.setTransform(QTransform(*part["transforms"][int(i)]) * S)
        q.drawImage(0, 0, QImage(px.data, w, h, 8 * w, QImage.Format_RGBA64_Premultiplied))
    q.end()


def render_slices(stacks, plan: dict, width: int, height: int, scale: float = 1.0) -> QImage:
    """The volume's picture on a width x height view (drawn at 'scale' of it: smaller while turning, larger on a
    high-density screen): the slices of 'stacks' (by slicing axis, a dict or a sequence: (m, h, w) uint64,
    Format_RGBA64_Premultiplied pixels, volume3d.transfer of the levels; or the one array across plan['axis'] when
    that part draws the whole picture, as orthographic) placed by 'plan' (volume3d.slice_plan), each part within its
    own pixels, and composited front to back onto a transparent 16-bit image. Transparent where nothing is."""
    img, S = _canvas(width, height, scale)
    for part, region in part_regions(plan, img.width(), img.height()):
        if not isinstance(stacks, np.ndarray):
            stack = stacks[part["axis"]]
        elif part["axis"] == plan["axis"]:
            stack = stacks
        else:
            raise ValueError("this view draws from more than one slicing axis: render_slices needs every stack")
        _composite(img, S, part, ((i, stack[int(i)]) for i in part["order"]), region)
    return img


class SharpPicture:
    """A settled volume picture drawn a few slices at a time (pump), so the window stays responsive while a large
    cube is drawn at the screen's full resolution: each slice's colours gathered from its levels on volume3d's
    threads (numpy lets go of the GIL), a few slices ahead, and composited in order here, part by part (a
    perspective view draws from up to three slicing axes: volume3d.slice_plan)."""
    AHEAD = 24                                  # slices gathered ahead of the one drawn

    def __init__(self, key, src: dict, lut: np.ndarray, plan: dict, width: int, height: int, ratio: float):
        self.key, self.lut, self.plan = key, lut, plan
        # the picture in device pixels (S maps the view onto them); deliberately no setDevicePixelRatio: QPainter
        # would scale by it again on top of S (a Retina picture twice its size), and _draw_volume draws it into
        # the widget's rect whatever its pixel count
        self.img, self.S = _canvas(width, height, ratio)
        self.parts = part_regions(plan, self.img.width(), self.img.height())
        self.images = {a: VOL.slice_images(src, a) for a in sorted({q["axis"] for q, _ in self.parts})}
        self.order = [(n, int(i)) for n, (q, _) in enumerate(self.parts) for i in q["order"]]   # (part, slice)
        self.next = 0                           # the next slice (in order) to draw
        self.futures: dict = {}
        self.t0 = time.perf_counter()
        self.ms = 0.0

    def _gather(self, n: int, i: int) -> np.ndarray:
        part = self.parts[n][0]
        w, h = part["size"]
        out = np.empty((h, w), dtype=np.uint64)
        # levels < 256 = len(lut): clip never acts, and it skips 'raise''s bounds pass
        np.take(self.lut, self.images[part["axis"]][i], out=out, mode="clip")
        return out

    def done(self) -> bool:
        return self.next >= len(self.order)

    def _draw(self, ready: list):
        """Composite the gathered (part, slice, pixels) in order, each run of one part at once."""
        k = 0
        while k < len(ready):
            n = ready[k][0]
            run = []
            while k < len(ready) and ready[k][0] == n:
                run.append(ready[k][1:])
                k += 1
            part, region = self.parts[n]
            _composite(self.img, self.S, part, run, region)

    def pump(self, budget_s: float = float("inf")) -> bool:
        """Draw slices for about budget_s seconds (all: inf); True when the picture is complete."""
        end = time.perf_counter() + budget_s
        ready = []
        while not self.done():
            for k in range(self.next, min(self.next + self.AHEAD, len(self.order))):
                at = self.order[k]
                if at not in self.futures:
                    self.futures[at] = VOL.gui_pool().submit(self._gather, *at)
            at = self.order[self.next]
            ready.append(at + (self.futures.pop(at).result(),))
            self.next += 1
            if len(ready) >= 8 or self.done() or time.perf_counter() > end:
                self._draw(ready)
                ready = []
            if time.perf_counter() > end:
                break
        if ready:
            self._draw(ready)
        if self.done():
            self.ms = 1e3 * (time.perf_counter() - self.t0)
        return self.done()

    def cancel(self):
        for f in self.futures.values():
            f.cancel()
        self.futures = {}


def next_turn_step(step: int, costs: dict, budget_ms: float = TURN_BUDGET_MS) -> int:
    """The step (an index into TURN_STEPS) of the next GPU frame while the view turns, from the frames drawn so far in
    this turn: 'costs' maps each step tried to its drawing time (ms, the median of its last three frames; a step tried
    fewer times is not in it). Settled, the picture is always every slice of the volume itself."""
    # Design: the step moves one at a time. Coarser while the current step is over budget; finer when the finer one
    # is known to fit or, not tried yet in this turn, when the current one costs under 0.4 of the budget (a step
    # roughly halves the cost, never quite: the cache helps). A step known too dear is never tried again in the same
    # turn, so it cannot flicker between two. Measured on TNG100-3-Dark 1024^3, 2000 x 1600 on a busy M1 Max: (0, 1)
    # 96 ms, (0, 2) 69, (1, 1) 17, (1, 2) 11, (1, 4) ~10. Not chosen: jumping straight to the finest step known to
    # fit, aiming at the mouse's own event rate instead of a fixed budget, keeping the last turn's costs (they hold
    # only while the cube, the screen and the range stay).
    cur = costs.get(step)
    if cur is None:
        return step
    if cur > budget_ms:
        return min(step + 1, len(TURN_STEPS) - 1)
    if step > 0:
        finer = costs.get(step - 1)
        if finer <= budget_ms if finer is not None else cur < 0.4 * budget_ms:
            return step - 1
    return step


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
        # (lo, hi, opacity, cmap, symmetric, gamma, shells): volume3d.transfer's; shells contours.shell_spec or None
        self.transfer: tuple | None = None
        self.shell_key: dict | None = None  # the contour shells' key drawn at the top left: {'head', 'items':
                                            # [(QColor, text)]} (the volume with the Contours tick), else None
        self._line_cuts: dict = {}          # the planes' contour lines cut to each quad: {(lines, rect): pieces}
        self._line_paths: dict = {}         # ... and on the screen: {(lines, rect, map, size): [(QColor, path)]}
        self._stacks: dict = {}             # the coarse cube's slices coloured: (key, stack), while turning
        self._picture = None                # (key, QImage): the quick picture (while turning, or until the sharp one)
        self._sharp = None                  # (key, QImage): the settled picture at the screen's full resolution
        self._preview = None                # (key, QImage): a large cube's first, half-resolution settled picture
        self._job: SharpPicture | None = None   # either being drawn, a few slices a tick
        self._pump = QTimer(self, interval=0)
        self._pump.timeout.connect(self._pump_tick)
        self.last_volume_ms = 0.0           # the last sharp picture's cost, from start to finish
        self.last_preview_ms = 0.0          # ... and the last preview's
        # the GPU's picture (volume_gl), where OpenGL is: the cube goes up in ticks, the picture kept while only the
        # overlays change; QPainter's for Save image (painter_only) and wherever OpenGL is not
        self.painter_only = False
        self.gl_note = ""                   # why the GPU does not draw it ('' when it does, or was never asked)
        self._gl_pic = None                 # (key, QImage)
        self._gl_step: int | None = None    # the turning frames' step (TURN_STEPS), adapted frame by frame ...
        self._gl_costs: dict = {}           # ... from this turn's frames: {step: [ms, ...]}
        self._gl_turning = False            # the last GPU frame was a turning one
        self.last_turn_ms = 0.0             # the last turning frame's GPU time
        self._gl_upload = QTimer(self, interval=10)
        self._gl_upload.timeout.connect(self._gl_upload_tick)

    # ---------------------------------------------------------------- the scene
    def set_volume(self, prep: dict | None, lo=None, hi=None):
        """The volume to draw (volume3d.prepare's dict, None: none) and its window lo..hi (Mpc)."""
        if prep is not self.volume:             # the GPU's copy is of the old one (keyed by its id: freed now)
            self._gl_forget()
        self.volume = prep
        self._stacks, self._picture, self._sharp, self._preview = {}, None, None, None
        self._stop_job()
        if lo is not None:
            self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.update()

    def set_perspective(self, on: bool):
        """Perspective (the eye at a finite distance) or orthographic; the window fitted again, turned as it is."""
        self.cam.perspective = bool(on)
        self.fit()

    def set_transfer(self, lo: float, hi: float, opacity: float, cmap: str, symmetric: bool = False,
                     gamma: float = VOL.DEFAULT_GAMMA, shells: tuple | None = None):
        """The volume's transfer function: transparent below 'lo' (the map's scale: log10 of a density), the opacity
        rising to 'opacity' at 'hi', coloured with 'cmap' over lo..hi (volume3d.transfer); 'shells' the contour
        shells laid into it (contours.shell_spec), None none."""
        self.transfer = (float(lo), float(hi), float(opacity), str(cmap), bool(symmetric), float(gamma),
                         shells or None)
        self._gl_moving()

    def set_scene(self, planes, lo, hi):
        self.planes = planes
        self._line_cuts, self._line_paths = {}, {}      # the contour lines' pieces of the planes shown before
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

    def crowded(self) -> bool:
        """So many voids in the view (the whole box, not a close-up) that each but the hovered and the inspected one
        is drawn as a faint fill and its centre, over the volume with a faint outline (CROWD_VOIDS)."""
        v = self.voids
        return bool(v) and not self.focus and len(v["rows"]) > CROWD_VOIDS

    def _sized(self):
        """The camera to the widget's size and the window drawn (zooming flies a perspective eye into it)."""
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
            w = inv.m13() * x + inv.m23() * y + inv.m33()   # the map backward by hand: QTransform.map holds w at a
            if w <= 0.0:                                 # near limit of its own. Perspective, w <= 0: this line of
                continue                                 # sight meets the plane BEHIND the eye (the eye in the window)
            p = QPointF((inv.m11() * x + inv.m21() * y + inv.m31()) / w,
                        (inv.m12() * x + inv.m22() * y + inv.m32()) / w)
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
        corners = V.box_corners(lo, hi)
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
        if volume:
            self._draw_shell_key(p)
        p.end()

    def _view_key(self) -> tuple:
        """What the volume's picture depends on besides the cube: the transfer function and the camera."""
        c = self.cam
        return (id(self.volume), self.transfer, c.yaw, c.pitch, c.scale, tuple(c.target), c.width, c.height,
                c.perspective)

    def _lut(self, plan: dict, stride: int = 1) -> np.ndarray:
        """The transfer function's table for this plan (its opacity corrected for the path through a slice, to 1 %;
        every stride-th slice drawn: through stride of them)."""
        lo, hi, opacity, cmap, symmetric, gamma, shells = self.transfer
        return VOL.transfer(self.volume["levels"], lo, hi, opacity, float(np.round(plan["exponent"], 2)) * stride,
                            G.lut(cmap), symmetric, gamma, shells)

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
        # the coloured stacks kept from frame to frame, one per slicing axis in use (a perspective view draws from
        # up to three: 33 MB each at 160^3); all remade when the cube, the range or the exponent changes
        skey = (id(src), round(plan["exponent"], 2), self.transfer)
        hit = self._stacks.get("quick")
        if hit is None or hit[0] != skey:
            self._stacks.pop("quick", None)             # the old ones freed before the new ones are made
            hit = (skey, {})
            self._stacks["quick"] = hit
        img, S = _canvas(self.cam.width, self.cam.height, FAST_SCALE)
        for part, region in part_regions(plan, img.width(), img.height()):
            a = part["axis"]
            if a not in hit[1]:
                hit[1][a] = VOL.gather(self._lut(plan), VOL.slice_images(src, a))
            _composite(img, S, part, ((i, hit[1][a][int(i)]) for i in part["order"]), region)
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

    def _preview_key(self) -> tuple | None:
        """A large cube's first settled picture (PREVIEW_SCALE of the sharp one's pixels a side); None for a cube
        small enough to settle in one step (fewer than PREVIEW_CELLS cells along every axis)."""
        if self.volume is None or max(self.volume["fine"]["shape"]) < PREVIEW_CELLS:
            return None
        return ("preview", round(self._sharp_ratio() * PREVIEW_SCALE, 2)) + self._view_key()

    def _next_key(self) -> tuple:
        """The settled picture to draw next: the preview until it is ready for this view, then the sharp one."""
        pkey = self._preview_key()
        if pkey is not None and (self._preview is None or self._preview[0] != pkey):
            return pkey
        return self._sharp_key()

    def _start_job(self, key: tuple | None = None) -> SharpPicture:
        """The settled picture of the full cube begun (pump draws it): 'key' a _sharp_key (the default: the screen's
        own resolution) or a _preview_key (PREVIEW_SCALE of it); the key's second item is the picture's ratio."""
        self._stop_job()
        key = self._sharp_key() if key is None else key
        src = self.volume["fine"]
        plan = VOL.slice_plan(self.cam, self.lo, self.hi, src["shape"])
        self._job = SharpPicture(key, src, self._lut(plan), plan, self.cam.width, self.cam.height, key[1])
        return self._job

    def _stop_job(self):
        if self._job is not None:
            self._job.cancel()
            self._job = None
        self._pump.stop()

    def _pump_tick(self):
        """A few more slices of the settled picture being drawn (about 25 ms); shown when complete. A large cube's
        preview, once complete, is shown and the sharp picture begun at once."""
        job = self._job
        if job is None or self._fast or job.key not in (self._sharp_key(), self._preview_key()):
            self._stop_job()
            return
        if job.pump(0.025):
            self._job = None
            if job.key[0] == "preview":
                self._preview, self.last_preview_ms = (job.key, job.img), job.ms
                self._start_job()
                self._pump.start()
            else:
                self._sharp, self._preview, self.last_volume_ms = (job.key, job.img), None, job.ms
                self._pump.stop()
            self.update()

    def volume_picture(self) -> QImage | None:
        """The volume's picture as the view now stands, at once (the tests, Save image): while turning the quick
        one, else the sharp one at the screen's resolution (never the preview), its drawing finished here if it was
        under way."""
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
            self._sharp, self._preview, self.last_volume_ms = (key, job.img), None, job.ms
            self._job = None
            self._pump.stop()
        return self._sharp[1]

    def _draw_volume(self, p: QPainter):
        """The volume: the sharp picture when it is ready for this view; else a large cube's preview when that is
        ready, else the quick one -- and the next settled picture begun (not while turning) and drawn in ticks: the
        preview, then the sharp one, each shown when complete."""
        if self.volume is None or self.transfer is None:
            return
        if self._draw_gl(p):                    # the GPU drew it (volume_gl), where OpenGL is
            return
        img = None
        if not self._fast:
            key = self._sharp_key()
            if self._sharp is not None and self._sharp[0] == key:
                img = self._sharp[1]
            else:
                pkey = self._preview_key()
                if pkey is not None and self._preview is not None and self._preview[0] == pkey:
                    img = self._preview[1]
                want = self._next_key()
                if self._job is None or self._job.key != want:
                    self._start_job(want)
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
        if pl.get("lines") is not None and not self._fast:  # the contour lines on it, under the voids' cuts (not
            self._draw_lines(p, pl["lines"], rect, T)        # while turning: as the cuts, a rough plane's are slow)
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
                if clip and quad is None:           # (cut in front of the eye first: a quad partly behind it
                    quad = QPainterPath()           # has no outline as a whole, and the clip would hide every cut)
                    outline = V.screen_polygon(
                        (T.m11(), T.m12(), T.m13(), T.m21(), T.m22(), T.m23(), T.m31(), T.m32(), T.m33()),
                        (rect.x(), rect.y(), rect.width(), rect.height()), self.cam.width, self.cam.height)
                    if len(outline):
                        quad.addPolygon(QPolygonF([QPointF(float(u), float(v)) for u, v in outline]))
                        quad.closeSubpath()
                if clip and quad.isEmpty():         # nothing of the quad is in the view
                    continue
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

    def _draw_lines(self, p: QPainter, lines: dict, rect: QRectF, T: QTransform):
        """A plane's contour lines (planes3d's 'lines': per level its pieces in the texture's pixels and its opaque
        colour) on the quad showing 'rect' of the texture: cut to it, mapped by the plane's map T in numpy (no Qt
        clip or projective stroke: view3d.segments_on_screen) and stroked by a cosmetic one-pixel pen, antialiased,
        at the planes' opacity (the painter's). The colour is the map's line laid over its own level's colour
        (contours.solid_colours): what the map shows along it. On a high-dpi screen the stroke is repeated one device
        pixel right and one down, the map's rule (explore_canvas: a wider pen outlines the whole path first, seconds
        for a rough plane): about one point wide, as the map's."""
        T9 = (T.m11(), T.m12(), T.m13(), T.m21(), T.m22(), T.m23(), T.m31(), T.m32(), T.m33())
        box = (rect.left(), rect.top(), rect.right(), rect.bottom())
        key = (id(lines), box, T9, self.cam.width, self.cam.height)
        paths = self._line_paths.get(key)
        if paths is None:                               # cut to the quad (kept while the scene is), then mapped
            cut = self._line_cuts.get((id(lines), box))
            if cut is None:
                cut = [CT.clip_segments(segs, *box) for segs in lines["segments"]]
                self._line_cuts[(id(lines), box)] = cut
            paths = []
            for segs, rgb in zip(cut, lines["rgb"]):
                scr = V.segments_on_screen(segs, T9, self.cam.width, self.cam.height) if len(segs) else segs
                if len(scr):
                    paths.append((QColor.fromRgbF(*(float(c) for c in rgb)), segments_path(scr)))
            if len(self._line_paths) > 64:              # another view: the last one's paths go
                self._line_paths = {}
            self._line_paths[key] = paths
        dpr = self._ratio()
        shifts = ((0.0, 0.0),) if dpr < 1.5 else ((0.0, 0.0), (1.0 / dpr, 0.0), (0.0, 1.0 / dpr))
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        for colour, path in paths:
            p.setPen(_pen(colour, 1.0))
            for dx, dy in shifts:
                p.setTransform(QTransform.fromTranslate(dx, dy))
                p.drawPath(path)
        p.restore()

    def _draw_shell_key(self, p: QPainter):
        """The contour shells' key at the top left: their values in their colours (the volume with Contours ticked),
        so the picture and a saved one say what each shell is."""
        key = self.shell_key
        if not key or not key.get("items"):
            return
        fm = QFontMetrics(p.font())
        h, dot = fm.height(), 0.62 * fm.height()
        widths = [dot + 4 + fm.horizontalAdvance(t) for _c, t in key["items"]]
        head = key["head"]
        bw = 10 + fm.horizontalAdvance(head) + sum(widths) + 10 * len(widths) + 6
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(QRectF(8.0, 8.0, bw, h + 8), 3, 3)
        x, y = 14.0, 12.0
        p.setPen(QColor(225, 225, 230))
        p.drawText(QRectF(x, y, fm.horizontalAdvance(head) + 2, h), Qt.AlignLeft | Qt.AlignVCenter, head)
        x += fm.horizontalAdvance(head) + 10
        for (colour, text), w in zip(key["items"], widths):
            p.setPen(_pen(colour, 2.0))                 # a ring: a shell seen edge on
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(x, y + (h - dot) / 2, dot, dot))
            p.setPen(colour)
            p.drawText(QRectF(x + dot + 4, y, w, h), Qt.AlignLeft | Qt.AlignVCenter, text)
            x += w + 10
        p.restore()

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
        if self.crowded() and not (sel or hov):        # many voids: a faint fill and its centre's dot
            p.save()
            p.setPen(Qt.NoPen)
            p.setTransform(QTransform(u[0], u[1], w[0], w[1], x, y))
            p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), CROWD_FILL_ALPHA.get(self.style, 20)))
            p.drawEllipse(unit)
            line = CROWD_LINE_ALPHA.get(self.style, 0)
            if line:
                p.setBrush(Qt.NoBrush)
                p.setPen(_pen(QColor(colour.red(), colour.green(), colour.blue(), line), 1.0))
                p.drawEllipse(unit)
                p.setPen(Qt.NoPen)
            p.setTransform(QTransform())
            p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 200))
            p.drawEllipse(QPointF(x, y), CROWD_DOT_PX, CROWD_DOT_PX)
            p.restore()
            return
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
        """The window's twelve edges (corners: its eight, Mpc): those behind it dashed first, the rest over
        everything; of an edge passing the eye (the eye inside the window) the part in front of it."""
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(_pen(QColor(200, 200, 200, 110), 1.0, Qt.DashLine) if behind else _pen(QColor(220, 220, 220), 1.2))
        for i, (a, b) in enumerate(V.BOX_EDGES):
            if (i in hidden) == behind:
                ends = V.segment_on_screen(self.cam, corners[a], corners[b])
                if ends is not None:
                    p.drawLine(QPointF(*ends[0]), QPointF(*ends[1]))

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
        middle of the window when fitted; with the eye inside the window, that depth ahead of the eye is said)."""
        span = self.width() / self.cam.scale
        length = G.nice_length(span)
        px = length * self.cam.scale
        if not (8 <= px <= self.width() - 40):
            return
        fm = QFontMetrics(p.font())
        label = f"{length:g} Mpc"
        if self.cam.perspective:
            eye = self.cam.eye()
            inside = self.lo is not None and bool(np.all((eye >= self.lo) & (eye <= self.hi)))
            label += f" at {self.cam.distance():.3g} Mpc ahead" if inside else " at the centre"
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


    # ---------------------------------------------------------------- the GPU's picture
    def _draw_gl(self, p: QPainter) -> bool:
        """The volume drawn from the GPU's picture (the quick one while its cube goes up): True; False when QPainter
        draws it (no OpenGL, Save image, a volume too large, an error)."""
        img = self._gl_frame()
        if img is None:
            return False
        if img is False:                        # the GPU is getting the cube: the quick picture meanwhile
            img = self.quick_picture()
            if img is None:
                return False
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(QRectF(0.0, 0.0, float(self.cam.width), float(self.cam.height)), img)
        p.restore()
        return True

    def _gl_renderer(self):
        """The GPU renderer when it can draw the volume shown (OpenGL here, a cube it holds), else None (QPainter)."""
        if self.painter_only or self.volume is None:
            return None
        r = VGL.renderer()
        if r is None:
            self.gl_note = VGL.unavailable_reason()
            return None
        if not r.fits(self.volume["fine"]["shape"]):
            self.gl_note = f"the volume is larger than the GPU's {r.max3d}³ textures"
            return None
        self.gl_note = ""
        return r

    def _gl_frame(self):
        """The GPU's picture as the view now stands (a QImage), False while its cube goes up, None when QPainter draws
        the volume. Turning: a step of TURN_STEPS (next_turn_step, from this turn's frames); settled: every slice of
        the volume at the screen's own pixels. Kept while only the overlays change."""
        r = self._gl_renderer()
        if r is None:
            return None
        key, vol = id(self.volume), self.volume
        try:
            if not r.ready(key):
                if r.cube_key != key:
                    self._stop_job()                # no QPainter sharp picture: the GPU's comes
                    r.begin_upload(key, vol["fine"]["q"], vol["levels"], vol["qrange"], vol["log"])
                self._gl_upload.start()
                return False
            self._sized()
            scale = self._ratio()
            level, stride, step = 0, 1, None
            if self._fast:
                if not self._gl_turning:            # a new turn: its costs learnt afresh, from the last turn's step
                    self._gl_turning, self._gl_costs = True, {}
                if self._gl_step is None:
                    self._gl_step = TURN_STEPS.index((1, 1))
                step = self._gl_step
                level, stride = TURN_STEPS[step]
                if level == 1 and not r.ready(key, 1):  # the half not there (yet, or a cube too small for one)
                    level, stride = 0, 2 * stride
            else:
                self._gl_turning = False
            vkey = ("gl", level, stride, round(scale, 3)) + self._view_key()
            if self._gl_pic is not None and self._gl_pic[0] == vkey:
                return self._gl_pic[1]
            plan = VOL.slice_plan(self.cam, self.lo, self.hi, r.shape(level))
            w, h = max(1, round(self.cam.width * scale)), max(1, round(self.cam.height * scale))
            t = time.perf_counter()
            px = r.render(self.cam, self.lo, self.hi, plan, self._lut(plan, stride), w, h, stride=stride, level=level)
            ms = 1e3 * (time.perf_counter() - t)
            if step is not None:
                self.last_turn_ms = ms
                seen = self._gl_costs.setdefault(step, [])
                seen.append(ms)
                del seen[:-3]
                self._gl_step = next_turn_step(step, {k: float(np.median(v)) for k, v in self._gl_costs.items()
                                                      if len(v) >= 3})
            else:
                self.last_volume_ms = ms
            self._gl_pic = (vkey, VGL.to_qimage(px))
            return self._gl_pic[1]
        except Exception as e:                      # noqa: BLE001 -- any GL failure: QPainter from now on
            self._gl_failed(e)
            return None

    def _gl_upload_tick(self):
        """A few more layers of the cube to the GPU (about 30 ms); the picture redrawn when it is all there."""
        r = VGL.renderer()
        if r is None or self.volume is None or r.cube_key != id(self.volume):
            self._gl_upload.stop()
            return
        try:
            before = (r.ready(r.cube_key), r.ready(r.cube_key, 1))
            if r.upload_step(0.03):
                self._gl_upload.stop()
            if (r.ready(r.cube_key), r.ready(r.cube_key, 1)) != before:
                self._gl_pic, self._gl_costs = None, {}     # the cube drawn now (or its half for the turning frames)
                self.update()
        except Exception as e:                      # noqa: BLE001
            self._gl_failed(e)

    def _gl_failed(self, e: Exception):
        import sys
        import traceback
        why = f"{type(e).__name__}: {e}" if not isinstance(e, VGL.GLUnavailable) else str(e)
        print(f"3D view: the GPU stopped drawing the volume ({why}); QPainter draws it", file=sys.stderr)
        if not isinstance(e, VGL.GLUnavailable):
            traceback.print_exc()
        VGL.give_up(why)
        self.gl_note = why
        self._gl_upload.stop()
        self._gl_pic = None
        self.update()

    def _gl_forget(self):
        """The GPU's copy of the cube freed (another volume, or none)."""
        self._gl_upload.stop()
        self._gl_pic = None
        r = VGL.current_renderer()
        if r is not None:
            try:
                r.forget_cube()
            except Exception as e:                  # noqa: BLE001
                self._gl_failed(e)

    def gl_bytes(self) -> int:
        """The memory the GPU's copy of the shown volume holds (the cube and its half, a byte a cell: 1.1 GB at
        1024³), 0 without."""
        r = VGL.current_renderer()
        return 0 if r is None else r.held_bytes()

    def _gl_moving(self):
        """A change that may come in a stream (a range or opacity slider dragged, the window resized): with the GPU
        drawing, the turning frames until it rests (each settled 1024³ frame is 0.1-0.2 s on the GUI thread); else
        as before (QPainter draws its quick picture meanwhile by itself)."""
        if not hasattr(self, "_gl_upload"):         # a resize while it is being built
            return
        r = self._gl_renderer() if self.style == "volume" and self.volume is not None else None
        if r is not None and r.ready(id(self.volume)):
            self._interacting()
        else:
            self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if e.oldSize().isValid() and e.oldSize() != e.size():     # dragged larger or smaller (not a first layout,
            self._gl_moving()                                       # nor a hidden widget grabbed: (-1, -1) each time)

# ==================================================================== the Explore pane's part
class View3DMixin:
    def _build_3d(self, col):
        """The '3D view' group of the left pane: the view's row (with the planes' controls), the volume's
        transfer function, the voids, the view's state line; then the 3D state and the view's signals."""
        g = QGroupBox("3D view")
        v = QVBoxLayout(g)
        v.addWidget(self._note("The map's field as a translucent volume (or on three planes) with the voids at "
                               "their true size. Drag to turn it, right-drag (or shift-drag) to move it, "
                               "scroll to zoom. Click a void to inspect it; double-click it (or Close up) to look at "
                               "it from near."))
        self._build_3d_view_row(v)
        self._build_3d_volume(v)
        self._build_3d_voids(v)
        self.view3d_state = self._note("")
        v.addWidget(self.view3d_state)
        col.addWidget(g)
        self._init_3d_state()
        self._wire_3d()

    def _build_3d_view_row(self, v):
        """The view's row: Show in 3D, Reset view, the style, perspective, the planes' controls, the detail. The
        long tooltips are their first sentence, the whole text the What's This (tip_auto)."""
        self.view3d_toggle = QPushButton("Show in 3D")
        self.view3d_toggle.setCheckable(True)
        tip_auto(self.view3d_toggle, VIEW3D_TIP)
        self.view3d_toggle.toggled.connect(self.show_3d)
        self.view3d_reset = tip(QPushButton("Reset view"), "The first oblique look, the whole window in view")
        self.view3d_reset.clicked.connect(lambda: self.view.view3d.reset_view())
        self.view3d_style = QComboBox()
        for text, style in STYLE_ITEMS:
            self.view3d_style.addItem(text, style)
        tip_auto(self.view3d_style, STYLE_TIP)
        self.view3d_style.currentIndexChanged.connect(self._3d_style_changed)
        self.view3d_persp = QCheckBox("perspective")
        self.view3d_persp.setChecked(True)
        tip_auto(self.view3d_persp, PERSPECTIVE_TIP)
        self.view3d_persp.toggled.connect(self._3d_perspective)
        self._build_3d_planes()
        self.view3d_detail = QComboBox()
        for d in VOL.DETAILS:
            self.view3d_detail.addItem(f"{d}³ cells", d)
        self.view3d_detail.setCurrentIndex(self.view3d_detail.findData(VOL.DEFAULT_DETAIL))
        tip_auto(self.view3d_detail, DETAIL_TIP)
        self.view3d_detail.currentIndexChanged.connect(lambda _i: self._3d_request())
        v.addLayout(button_row(self.view3d_toggle, self.view3d_reset, self.view3d_style, self.view3d_persp,
                               self.planes3d_box, self.view3d_detail))

    def _build_3d_planes(self):
        """The planes' own controls for the view's row (shown with the planes): an axis tick each, the opacity."""
        self.planes3d_box = QWidget()
        pr = flush(QHBoxLayout(self.planes3d_box))
        self.view3d_axes = []
        for a in range(3):
            cb = tip(QCheckBox(V.AXES[a]), f"The plane normal to {V.AXES[a]}")
            cb.setChecked(True)
            cb.toggled.connect(lambda _on: self._3d_request())
            pr.addWidget(cb)
            self.view3d_axes.append(cb)
        self.view3d_opacity = QSpinBox(minimum=10, maximum=100, value=90, singleStep=10)
        self.view3d_opacity.setSuffix(" % opaque")
        tip(self.view3d_opacity, "The planes' opacity: below 100 % a sphere behind a plane shows through it")
        self.view3d_opacity.valueChanged.connect(self._3d_opacity)
        pr.addWidget(self.view3d_opacity)

    def _build_3d_volume(self, v):
        """The volume's transfer function (shown with the volume): the shown range, the opacity, Default range."""
        self.volume3d_box = QWidget()
        vb = flush(QVBoxLayout(self.volume3d_box))
        row = QHBoxLayout()
        row.addWidget(QLabel("Shown from"))
        self.vol3d_lo = QSlider(Qt.Horizontal, minimum=0, maximum=1000)
        self.vol3d_lo_text = QLabel("")
        self.vol3d_hi = QSlider(Qt.Horizontal, minimum=0, maximum=1000, value=1000)
        self.vol3d_hi_text = QLabel("")
        for w_ in (self.vol3d_lo, self.vol3d_hi):
            tip_auto(w_, RANGE_TIP)
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
        tip_auto(self.vol3d_opacity, OPACITY_TIP)
        self.vol3d_opacity.valueChanged.connect(self._3d_opacity_moved)
        self.vol3d_opacity_text = QLabel("")
        self.vol3d_opacity_text.setMinimumWidth(QFontMetrics(self.vol3d_opacity_text.font()).horizontalAdvance("100 %")
                                                + 4)
        row.addWidget(self.vol3d_opacity, 1)
        row.addWidget(self.vol3d_opacity_text)
        self.vol3d_default = tip(QPushButton("Default range"), "Back to the range the volume opened with (a density: "
                                 "from ρ/ρ̄ = 1 to its 99.9th percentile) and the default opacity")
        self.vol3d_default.clicked.connect(self._3d_tf_default)
        row.addWidget(self.vol3d_default)
        vb.addLayout(row)
        v.addWidget(self.volume3d_box)

    def _build_3d_voids(self, v):
        """The voids' row: the void menu with Close up and Whole box (how they are drawn is the Map group's choice,
        beside the Voids tick: void3d_shape)."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Void"))
        self.void3d = QComboBox()
        self.void3d.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.void3d.setMinimumContentsLength(18)
        tip(self.void3d, "The distinct, well-resolved voids, the largest first (by the size drawn: R_ell of the "
                         "ellipsoids, or R_v): choose one to inspect it")
        self.void3d.activated.connect(self._void3d_chosen)
        row.addWidget(self.void3d, 1)
        self.void3d_close = tip_auto(QPushButton("Close up"), CLOSE_TIP)
        self.void3d_close.clicked.connect(lambda: self.close_up_3d(self._3d_selected()))
        row.addWidget(self.void3d_close)
        self.void3d_back = tip(QPushButton("Whole box"), "Back from the close-up to the whole box (its planes where "
                               "they were)")
        self.void3d_back.clicked.connect(self.whole_box_3d)
        row.addWidget(self.void3d_back)
        v.addLayout(row)

    def _init_3d_state(self):
        """The 3D view's state: nothing shown yet, no planes, no close-up, no request in flight."""
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
        self._3d_shells = None                  # the volume's contour shells shown (contours.shell_plan), or None
        self._3d_within = None                  # (key, cube): a close-up's cube with its own void's shells only
        self._3d_floor_kept = None              # (key, _3d_void_floor's value or None)
        self._3d_basin_kept = None              # _3d_basin's dict: a close-up's void up to its outer shell
        self._3d_ticker = QTimer(self, interval=200)   # the averaging's progress, while it runs
        self._3d_ticker.timeout.connect(self._3d_progress)

    def _wire_3d(self):
        """The view's signals, its camera's perspective and the controls' first state."""
        v3 = self.view.view3d
        v3.hovered.connect(self._3d_hover)
        v3.clicked.connect(self._3d_click)
        v3.double_clicked.connect(self._3d_double)
        v3.cam.perspective = self.view3d_persp.isChecked()
        self._3d_style_changed()
        self._3d_apply_tf()                     # the opacity's text
        self._3d_buttons()

    def state_3d(self) -> dict:
        """The 3D view's choices a restart keeps (ExploreControls.state): the style, perspective, the planes'
        opacity, the volume's detail and its opacity slider. Not 'Show in 3D' itself, nor a shown range (it belongs
        to one volume)."""
        return {"view3d_style": self.view3d_style.currentData(), "view3d_persp": self.view3d_persp.isChecked(),
                "view3d_opacity": self.view3d_opacity.value(), "view3d_detail": self.view3d_detail.currentData(),
                "vol3d_opacity": self.vol3d_opacity.value()}

    def restore_state_3d(self, d: dict):
        """Back to state_3d()'s choices. A key missing (a state saved before it) or a value not offered leaves
        the control as it is (a fresh window: today's default). Set with signals blocked, then applied once: the
        style's controls, the camera, the planes' opacity, the opacity's text; no read while the view is closed."""
        def number(key):
            try:
                return int(d[key])
            except (KeyError, TypeError, ValueError):
                return None
        style = self.view3d_style.findData(d["view3d_style"]) if isinstance(d.get("view3d_style"), str) else -1
        detail = self.view3d_detail.findData(number("view3d_detail")) if number("view3d_detail") is not None else -1
        persp, opacity, vol_opacity = d.get("view3d_persp"), number("view3d_opacity"), number("vol3d_opacity")
        before = self.view3d_style.currentIndex()
        for w, ok, set_ in ((self.view3d_style, style >= 0, lambda w: w.setCurrentIndex(style)),
                            (self.view3d_persp, isinstance(persp, bool), lambda w: w.setChecked(persp)),
                            (self.view3d_opacity, opacity is not None, lambda w: w.setValue(opacity)),
                            (self.view3d_detail, detail >= 0, lambda w: w.setCurrentIndex(detail)),
                            (self.vol3d_opacity, vol_opacity is not None, lambda w: w.setValue(vol_opacity))):
            if ok:
                w.blockSignals(True)
                set_(w)
                w.blockSignals(False)
        if self.view.view3d.cam.perspective != self.view3d_persp.isChecked():
            self._3d_perspective(self.view3d_persp.isChecked())
        self._3d_opacity(self.view3d_opacity.value())
        if self.view3d_style.currentIndex() != before:
            self._3d_style_changed()            # the style's controls shown (a fresh read when the view is open)
        self._3d_apply_tf()                     # the opacity's text (and the volume's, when one is shown)

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
        tip_auto(self.view3d_toggle, VIEW3D_TIP if not why else f"The 3D view: {why}.")
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
            self._3d_within = self._3d_basin_kept = None    # a close-up's copy of its levels with its shells freed
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
               "smooth3d": self.smooth3d.isChecked() and self.smooth3d.isEnabled(),
               "symmetric": name in G.DIVERGING and not log,
               "ps_gradient": self.ps_gradient.currentData() or G.PS_GRADIENT_DEFAULT,
               "index": list(self._3d_index), "shown": [cb.isChecked() for cb in self.view3d_axes],
               "lo": np.asarray(lo, dtype=np.float64), "hi": np.asarray(hi, dtype=np.float64), "style": style,
               # the map's Contours tick: the planes' lines, the volume's shells (never on a class map: why_not)
               "contours": self.contours.isChecked() and not CT.why_not(name, self.cmap.currentText()),
               "contour_detail": self.contour_detail.value()}
        if style == "volume":
            req["want"] = int(self.view3d_detail.currentData())
            req["detail"], req["note"] = self._3d_detail(o, req["lo"], req["hi"])
            req["progress"] = {}
            # the files' identity (mtime, size) too: a grid written again by a re-run is read again
            files = tuple(G.cube_key(o.files[s].path, o.dtype, VOL.file_shape(o, s)) for s in VOL.sources(o, name))
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
                    self._3d_res["request"].update(cmap=req["cmap"], note=req["note"], contours=req["contours"],
                                                   contour_detail=req["contour_detail"])
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
        # the GPU's copy of the volume shown stays until the new one replaces it
        need = VOL.need_bytes(VOL.window_shape(o, lo, hi, want)) + self.view.view3d.gl_bytes()
        if need <= 1e9:
            return want, ""
        smaller = sorted((d for d in VOL.DETAILS if d < want), reverse=True)

        def fitting(room: float) -> int:
            return next((d for d in smaller if VOL.need_bytes(VOL.window_shape(o, lo, hi, d)) <= room),
                        min(VOL.DETAILS))
        free = G.available_gb()
        prep = (self._3d_res or {}).get("volume")
        held = VOL.prepared_bytes(prep) if prep is not None else 0
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
        comp = G.component_suffix(req["component"])
        lines = res.get("contours")                     # the planes' contour levels: marked, said in the tooltip
        self.view.colorbar.set(req["cmap"], res["vrange"], G.colourbar_label(label, req), req["log"],
                               levels=(lines["levels"], lines["colours"]) if lines else None)
        self.view.colorbar.setToolTip(CT.describe(lines))
        self._contour_count(lines)
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
        prep = res["volume"]
        # the Contours tick: shells of the map's levels where the volume is clear (contours.shell_plan); in a close-up
        # of the density around the void looked at, from its own minimum, and drawn only near it
        shells, cube = None, prep
        if req.get("contours"):
            args = (req["field"], prep["levels"], prep["counts"], tf["lo"], bool(req["log"]), bool(req["symmetric"]),
                    cmap)
            detail = int(req.get("contour_detail", CT.DETAIL))      # the level slider: the map's levels, its count
            floor = self._3d_void_floor(res)
            if floor is None:                           # the whole cube
                shells = CT.shell_plan(*args, slopes=VOL.level_slopes(prep), cells=max(prep["shape"]), detail=detail)
            else:                                       # a close-up: the shells of its void's basin, from its floor
                first = CT.shell_plan(*args, floor=floor, detail=detail)
                basin = None if first is None else \
                    self._3d_basin(prep, float(np.max(first["levels"] + 2.0 * first["half"])))
                if basin is not None:
                    shells = CT.shell_plan(*args[:2], basin["counts"], *args[3:], floor=floor, slopes=basin["slopes"],
                                           cells=max(prep["shape"]), detail=detail)
                    if shells is not None and int(np.prod(prep["shape"])) <= SHELL_COPY_CELLS:
                        cube = self._3d_shell_cube(prep, basin, CT.shell_spec(shells), bool(req["symmetric"]))
        if cube is prep:
            self._3d_within = None                      # a close-up's copy freed once not drawn
        self._3d_shells = shells
        spec = CT.shell_spec(shells)
        v3 = self.view.view3d
        if v3.volume is not cube:
            v3.set_volume(cube, res["lo"], res["hi"])
        v3.shell_key = self._3d_shell_key(shells)
        v3.set_transfer(tf["lo"], tf["hi"], opacity, cmap, req["symmetric"], shells=spec)
        self.vol3d_lo_text.setText(self._3d_value_text(tf["lo"]))
        self.vol3d_hi_text.setText(self._3d_value_text(tf["hi"]))
        label = self._gradient_label(dict(req, ps_gradient="stored"))      # the volume is of the stored grid
        vr = (-tf["hi"], tf["hi"]) if req["symmetric"] else (tf["lo"], tf["hi"])
        marks = None
        if shells:                                      # a shell inside the bar's range is marked there (|value|:
            pos = shells["levels"]                      # both signs); the rest, below it, are in the key
            rgba = np.column_stack([shells["colours"] / 255.0, np.ones(len(pos))])
            if req["symmetric"]:
                pos, rgba = np.concatenate([-pos, pos]), np.concatenate([rgba, rgba])
            marks = (pos, rgba)
        self.view.colorbar.set(cmap, vr, G.colourbar_label(label, req)
                               + (" (|value| shown)" if req["symmetric"] else ""), req["log"], levels=marks)
        self.view.colorbar.setToolTip(CT.describe_shells(shells) if shells else CT.NO_SHELLS
                                      if req.get("contours") and req["field"] not in CT.SHELL_FIELDS else "")
        self._contour_count(shells, "shell")
        self._3d_state()

    def _3d_void_floor(self, res: dict) -> float | None:
        """A density volume's close-up: the lowest value (the map's scale) near the centre of the void looked at --
        within VOID_FLOOR_SHARE of its smallest radius (the fitted ellipsoid's c when drawn so, else R_v), at least
        one cell -- for its contour shells to nest around that void's own minimum (contours.shell_plan's 'floor').
        None for the whole box, another field or a void without a radius. Kept for the cube, the void and its
        shape."""
        close, vres, prep = self._3d_close, self._void_result_3d(), res.get("volume")
        if close is None or vres is None or prep is None or (res.get("request") or {}).get("field") != "density":
            return None
        row = int(close["row"])
        key = (id(prep), row, tuple(np.round(close["centre"], 9)), self._3d_ellipsoids())
        if self._3d_floor_kept is not None and self._3d_floor_kept[0] == key:
            return self._3d_floor_kept[1]
        small = float(vres["_r_v"][row])
        drawn = G.drawn_ellipsoids(vres)
        if self._3d_ellipsoids() and drawn["ok"][row]:
            small = float(np.min(drawn["semi"][row]))       # the drawn ellipsoid's c
        lo, hi = np.asarray(prep["lo"], dtype=np.float64), np.asarray(prep["hi"], dtype=np.float64)
        shape = np.asarray(prep["fine"]["shape"])
        cell = (hi - lo) / shape
        c = (np.asarray(close["centre"], dtype=np.float64) - lo) / cell        # the centre in the cube's cells
        floor = None
        if np.isfinite(small) and small > 0 and np.all((c >= 0) & (c <= shape)):
            rad = np.maximum(VOID_FLOOR_SHARE * small / cell, 1.0)
            a0 = np.clip(np.floor(c - rad).astype(int), 0, shape - 1)
            a1 = np.clip(np.ceil(c + rad).astype(int) + 1, 1, shape)
            q = prep["fine"]["q"][a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
            g = np.ogrid[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
            ball = sum(((g[k] + 0.5 - c[k]) / rad[k]) ** 2 for k in range(3)) <= 1.0
            inner = q[ball & (q > 0)]
            floor = float(prep["levels"][int(inner.min())]) if inner.size else None
        self._3d_floor_kept = (key, floor)
        return floor

    def _3d_basin(self, prep: dict, top: float) -> dict | None:
        """A close-up's void up to its outer contour shell (volume3d.basin from its centre), where its shells are
        drawn: {'key', 'top', 'counts' (its cells' level counts), 'slopes' (the field's slope per level there)};
        None when the centre is not below 'top'. Kept for the cube, the void and the value (a range slider dragged
        does not remake it); the mask itself is made again when a copy of the levels needs it (_3d_shell_cube), not
        held (a byte a cell)."""
        close = self._3d_close
        key = (id(prep), tuple(np.round(close["centre"], 9)), round(top, 9))
        if self._3d_basin_kept is None or self._3d_basin_kept["key"] != key:
            mask = VOL.basin(prep, close["centre"], top)
            self._3d_basin_kept = None if mask is None else {
                "key": key, "top": top, "counts": np.bincount(prep["fine"]["q"][mask], minlength=VOL.LEVELS),
                "slopes": VOL.level_slopes(prep, mask)}
            if self._3d_basin_kept is None:
                return None
        return self._3d_basin_kept

    def _3d_shell_cube(self, prep: dict, basin: dict, spec, symmetric: bool) -> dict:
        """A close-up's cube with its own void's contour shells only (volume3d.shells_within its basin: the
        neighbours' shells left out, the volume's own picture the same); kept while the shells' values and bands,
        the basin and the cube stay (a range slider dragged redraws it, it does not remake it)."""
        key = (tuple((s_[0], s_[5]) for s_ in spec), basin["key"])
        if self._3d_within is None or self._3d_within[0] != key:
            self._3d_within = None                      # the old copy freed before the new one is made
            mask = VOL.basin(prep, self._3d_close["centre"], basin["top"])
            self._3d_within = (key, prep if mask is None else VOL.shells_within(prep, mask, spec, symmetric))
        return self._3d_within[1]

    def _3d_shell_key(self, shells: dict | None) -> dict | None:
        """The view's key of the contour shells (View3D.shell_key): each value in its shell's colour."""
        if not shells:
            return None
        head = ("ρ/ρ̄" if shells["field"] == "density" else "|value|" if shells["symmetric"] else "value") + " ="
        return {"head": "contour shells, " + head,
                "items": [(QColor(*(int(c) for c in rgb)), f"{v:.0f}" if abs(v) >= 100 else f"{v:.2g}")
                          for v, rgb in zip(shells["values"], shells["colours"])]}

    def _3d_state(self):
        """The title and the state line of the volume shown: its cells, how many of the grid's each averages, how
        much of it shows."""
        res = self._3d_res or {}
        prep = res.get("volume")
        if prep is None:
            return
        req, o = res["request"], res["request"]["output"]
        label = self._gradient_label(dict(req, ps_gradient="stored"))
        comp = G.component_suffix(req["component"])
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
        r = VGL.renderer() if not VGL.disabled_reason() else None
        gpu = " · drawn on the GPU" if r is not None and r.fits(m) else ""
        if not VGL.disabled_reason() and not gpu:   # OpenGL was tried here and is not drawing it: say why
            gpu = f" · drawn by QPainter ({self.view.view3d.gl_note or VGL.unavailable_reason() or 'no OpenGL'})"
        self.view3d_state.setText(f"volume {cells}, {each} · {100.0 * frac:.3g} % of them above the low end"
                                  + gpu + (f" · {note}" if note else ""))

    # ---------------------------------------------------------------- the voids
    def _void_result_3d(self):
        """The catalogue of the output shown (the void_outlines.py result), or None: off, not applicable, not found."""
        o = self.current
        if not self.voids.isChecked() or o is None or self.voids_applicable(o):
            return None
        res = self._void_cache.get(self._void_cache_key(o))
        return None if not res or "error" in res else res

    def _3d_rows(self, res: dict) -> np.ndarray:
        """The voids the 3D view draws: the Sizes range by the size drawn (void_rows), with that size defined -- an
        R_v for the spheres; for the ellipsoids every distinct void (one without an ellipsoid is its centre's mark)."""
        rows = self.void_rows(res)
        return rows if self._3d_ellipsoids() else rows[np.isfinite(res["_r_v"][rows])]

    def _3d_voids(self):
        """The voids over the volume or the planes (their ellipsoids or spheres, as the map draws them; in a close-up
        those reaching into its cube, at their periodic image nearest the void looked at) and the void menu."""
        v3 = self.view.view3d
        res = self._void_result_3d() if self._3d_shown else None
        self._3d_fill_list(res)
        if res is None or self._3d_res is None:
            v3.selected = v3.hover_row = None
            v3.set_voids(None)
            self._3d_buttons()
            return
        rows = self._3d_rows(res)                   # the Sizes range, by the size drawn
        if self._3d_close is not None and self._3d_close["row"] not in rows:
            rows = np.append(rows, self._3d_close["row"])        # the void looked at stays, whatever its size
        centres = res["_centres"][rows]
        M, reach, labels = self._3d_shapes(res, rows)
        r_eff = None                        # the other radius, dashed: R_eff beside the spheres, R_v beside ellipsoids
        if self.void_reff.isChecked():
            r_eff = res["_r_v" if self._3d_ellipsoids() else "_r_eff"][rows]
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
        """The voids' shapes as drawn (view3d: centre + M u, |u| <= 1): spheres of R_v, or the ellipsoids at the mean
        density (grids.drawn_ellipsoids: the map's; a void without one is a small sphere at its centre, its mark);
        the radius each reaches (R_v, or a); their labels."""
        rows = np.asarray(rows, dtype=np.int64)
        r_v = res["_r_v"][rows]
        if not self._3d_ellipsoids():
            return V.sphere_matrices(r_v), r_v.copy(), [f"void {k} · R_v {r:.2f} Mpc" for k, r in zip(rows, r_v)]
        d = G.drawn_ellipsoids(res)
        ok = d["ok"][rows]
        mark = MARK_CELLS * float(res.get("cell_mpc") or res["box_mpc"] / res["n_grid"])
        M = V.sphere_matrices(np.full(len(rows), mark))
        M[ok] = d["M"][rows][ok]
        reach = np.where(ok, d["semi"][rows][:, 0], mark)
        labels = [f"void {k} · a, b, c = " + ", ".join(f"{s:.2f}" for s in d["semi"][k]) + " Mpc" if ok[j]
                  else f"void {k} · no ellipsoid (its centre)" for j, k in enumerate(rows)]
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
            self.void3d.addItem("tick 'Voids' above" if not self.voids.isChecked() else "no voids (yet)", None)
        else:
            key_r = self._void_radius_key()
            name = "R_ell" if key_r == "_r_ell" else "R_v"
            rows, radius = self._3d_rows(res), res[key_r]
            self.void3d.addItem(f"{len(rows)} voids: choose one" if len(rows) else "no void in the Sizes range", None)
            for k in sorted(rows, key=lambda k: -np.nan_to_num(radius[k], nan=-1.0)):
                delta = res["delta"][k]
                size = f"{name} {radius[k]:.2f} Mpc" if np.isfinite(radius[k]) else "no ellipsoid"
                self.void3d.addItem(f"void {k} · {size}" + ("" if delta is None else f" · δ {float(delta):.2f}"),
                                    int(k))
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
        r = float(res[self._void_radius_key()][row])            # its size: R_v, or the ellipsoid's R_ell
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
        v3.painter_only = True                  # QPainter's picture, as the tests hold it (not the GPU's)
        try:
            v3._fast = False                    # the settled picture: the planes' lines and cuts too
            if v3.style == "volume" and v3.volume is not None:
                v3.volume_picture()
            if not v3.grab().save(str(path)):
                raise OSError(f"could not write {path}")
        finally:
            v3.painter_only = False
        return str(path)
