"""The GUI's Explore tab: look at any run's grids without writing a plot script, and ask the
snapshot's own tessellation what is at a point.

  ExploreControls   the left pane: which output, field, component, axis and slice, colours,
                    and the point-query server (a dtfelib.Estimator on the snapshot behind the
                    output -- a composite of partitions, out of core, when one tessellation would
                    not fit in memory)
  ExploreView       the right pane: the slice (any size, scaled to fit), hover readout, colour
                    bar, and the answer for a clicked point: its streams, each with its density,
                    velocity and Lagrangian origin q(x)

The data side (grids.py) is Qt-free; this module only draws and wires. Slices load and queries
run on worker threads, so a 1024^3 z-plane or a tessellation build never freezes the window.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QProcess, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QTableWidget,
    QToolButton, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contours as CT  # noqa: E402  (the contour lines: marching squares and the levels)
import figsave as FS  # noqa: E402  (the Save options: presets, units, full resolution)
import grids as G  # noqa: E402
import runspec as rs  # noqa: E402
from app_widgets import ChoiceRadios, tip_auto  # noqa: E402  (tip_auto: a long tooltip, its What's This)
# its parts, one module each (ExploreControls' mixins; the classes and functions re-exported, explore.<name> as
# before): explore_canvas the map widgets, explore_grid the figure grid, explore_save Save image, explore_server the
# point-query server, explore_voids the void outlines, explore_zoom the zoom, explore_3d the 3D view
from explore_3d import SHAPE_ITEMS, SHAPE_TIP, View3D, View3DMixin, planes3d  # noqa: E402
from explore_canvas import ColorBar, ImageCanvas, contour_paths, stroke_layer  # noqa: E402,F401  (re-exported)
from explore_grid import (GridMixin, grid_differences, grid_dpi, grid_figure, grid_full_pixels,  # noqa: E402,F401
                          grid_items, grid_suptitle, grid_tile_circles, grid_tile_contours, grid_tile_marks,
                          grid_tiles, grid_unit, panel_label, panel_names, panel_overlay, panel_void_circles,
                          quantity, type_tag)
from explore_save import SaveMixin  # noqa: E402
from explore_server import ServerMixin, ServerWorker  # noqa: E402,F401  (re-exported)
import explore_voids as XV  # noqa: E402  (the void smoothing box, the search's time)
from explore_voids import VoidsMixin  # noqa: E402
from explore_zoom import (ExactZoomWorker, ZoomMixin, _fields_bytes, _fmt_duration,  # noqa: E402,F401
                          load_exact_planes)

VOID_TOOL = rs.REPO_ROOT / "python" / "tools" / "void_outlines.py"     # the voids of one snapshot, as JSON
VOID_CACHE_DIR = None        # the helper's catalogue cache: None = the pipeline's own (config.CACHE_DIR); tests pin a temp one
VOID_TIP = ("Draw the voids on the map and in the 3D view: each one's ellipsoid where it cuts this slice, or its "
            "sphere of R_v. How they are found: the analysis pipeline smooths the density in 3D (its production "
            "smoothing, or σ beside), takes its local minima, and at each the Hessian (the field's curvature): its "
            "eigenvectors are the ellipsoid's axes, its eigenvalues λ its shape. The ellipsoid drawn is where that "
            "quadratic fit around the minimum, δ_min + ½ Σ λ x², returns to the mean density: semi-axes √(2 |δ_min| "
            "/ λ), its (a b c)^⅓ called R_ell (the catalogue's own fit, 2 cells / √λ, sets only the shape's ratios). "
            "The map shows its section: an ellipse that grows, is largest on the layer of the minimum (marked by a "
            "dot, filled on its own layer) and shrinks as you scroll. R_v is where the spherical average of the "
            "smoothed density returns to the mean. The catalogue is found once per snapshot and smoothing and "
            "cached; one that would not fit in the free memory is not built, and it says so. Point at a void for its "
            "sizes; click it for the whole entry. Saved images and the figure grid carry the outlines, and a saved "
            "PNG gets a CSV table of the voids beside it.")
# the drawn shape (the Map group's 'Voids as', for the map and the 3D view alike; explore_3d keeps the items)
VOID_RV_TIP = ("Also draw each void's sphere of R_v, dashed, around the same centre: where the spherical average of "
               "the smoothed density returns to the mean, the size the catalogue's profiles use")
# a heavy action beside a running DTFE job starts without a question only when it fits in the free memory less
# this (void_outlines.HEADROOM_BYTES, the catalogue build's own margin)
HEAVY_HEADROOM_GB = 4.0
# the choice of a phase-space output's velocity derivatives (grids.py, GRADIENT_FIELDS): its items and tip
PS_GRADIENT_ITEMS = (("mean field ∇⟨v⟩: from the mean velocity, as standard DTFE", "mean-field"),
                     ("stored grids: the stream average ⟨∇v⟩ (an Exact run stores ∇⟨v⟩)", "stored"))
PS_GRADIENT_TIP = ("How a phase-space output's velocity divergence, shear, vorticity, gradient and V-web are "
                   "shown. Mean field (the default, as in the figure scripts): the derivatives of its mean "
                   "velocity grid, ∇⟨v⟩, the quantity standard DTFE stores, computed slice by slice from the "
                   "velocity grid (the V-web at the run's own threshold). Stream average: the grids the program "
                   "stored, the average of each stream's own gradient ⟨∇v⟩, which differs in multi-stream cells "
                   "(its divergence does not average to zero over the box, its V-web finds few filaments): a Fast "
                   "run's, and an Exact run's in 2D or with the halo release. A standard-DTFE output, and a "
                   "phase-space Exact one in 3D (its derivatives from exact face sums), is always shown as "
                   "stored: its grids are the mean field. The hi-res slices and the point zooms are point "
                   "evaluations: the stream average (an exact zoom takes the output's own route). Every title, "
                   "colour bar and figure-grid panel of a phase-space output says which it shows.")
# the contour lines' checkbox (contours.py: the levels, their colours, the line geometry); greyed out on a class map
CONTOURS_TIP = ("Contour lines over the map at the values marked on the colour bar (the density: mostly below the "
                "mean); each line keeps its value from layer to layer")
CONTOUR_DETAIL_TIP = ("Fewer contour lines (left) or more (right): each step spaces them a little closer. The number "
                      "is the lines on the map now")
# while the level slider is dragged, the map (and the figure grid, the 3D view, a zoom) is redrawn at most this often:
# each position is new lines on the slice worker, and the 3D volume's shells and a zoom's lines are made on the GUI
# thread; a click, a key or a programmatic move redraws at once
CONTOUR_DRAG_MS = 120
VOID_REFF_TIP = ("Also draw each void's R_eff = (a b c)^⅓ of the catalogue's fitted ellipsoid, dashed, around the "
                 "same centre: R_eff is about 1.6 R_v, the open question of which radius describes the void")
VOID_RANGE_TIP = ("Draw only the voids of this size ('any' = no bound), by the size drawn: R_ell = (a b c)^⅓ of the "
                  "ellipsoid drawn, or R_v with the spheres. The map, the figure grid, the saved images and their "
                  "CSV, and the 3D view keep the same voids. A few voids have an R_v far larger than their ellipsoid "
                  "(the density never returns to the mean nearby): an upper bound hides them")
VOID_COUNT_TIP = ("The voids drawn are the catalogue's distinct sample: minima whose fit passes its size limits, "
                  "outside every deeper void. Its other minima are not drawn")
VOID_SIGMA_TIP = ("The smoothing the voids are found at: the analysis scripts' own ('production') or one scale in "
                  "Mpc for every box. Production: the scripts' own 10 cells (a different physical scale in every "
                  "box) and their cached catalogue; a value in Mpc smooths every box at that scale (the cross-box "
                  "comparison), with its own cached catalogue, built once per snapshot and smoothing. The arrows "
                  "step from production to its own value, then one cell at a time, down to two cells: a search "
                  "starts when you press Return, leave the box or stop clicking, and the line under the row says "
                  "how long it should take. A small σ finds thousands of small voids, and its first search can take "
                  "minutes: one estimated over two minutes asks first")
SMOOTH3D_TIP = ("Smooth the whole cube in 3D instead of this slice: the slice of the density smoothed in 3D is the "
                "field the voids are found in. The first layer takes a few seconds per field and smoothing; then "
                "scrolling the layers is instant. The contours and the 3D view's planes follow it. A slice smoothed "
                "in 2D is another field: its dark patches are not the sections of the 3D minima")

CMAPS = ["viridis", "magma", "inferno", "cividis", "plasma", "RdBu_r", "coolwarm", "gray", "tab10"]
AXES = ["x", "y", "z"]
# the figure grid's layouts (rows, columns): a field at several redshifts side by side, or several fields of one
# snapshot, composed as one figure
GRID_LAYOUTS = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]
GRID_NONE = ""
GRID_COLS = ("Simulation", "Redshift", "Type", "Field", "Component")      # the composer's table
_C_SIM, _C_SNAP, _C_TYPE, _C_FLD, _C_CMP = range(5)              # the output combo's first entry: an empty panel
GRID_MERGE_TIP = ("The panels touch, with shared axes and their colour bars in one column at the right. No gaps, tick "
                  "labels on the outer edges only, each panel carries a small label with what differs between them; "
                  "one bar when every panel shows the same field, one per field otherwise. Possible only when every "
                  "panel comes from a box of the same size; 'one colour range per field' is implied.")
# the Output menus' first step is a simulation, or one of these two groups (no simulation has a space
# in its name): the custom-snapshot runs, and the grid files opened by hand with the … button
GROUP_CUSTOM = "custom snapshots"
GROUP_OPENED = "opened files"
KEPT_ZOOM_TEXT = "zoom window "     # the zoom line's start when a window carried from another output is not shown
# what a drag on the map computes (the Zoom group's pair): the query server's point values, at once, or the program
# run again on the region for its exact cell averages. Both are exact in their own way: the words are not Fast/Exact
ZOOM_POINT_TEXT = "Point values (instant)"
ZOOM_EXACT_TEXT = "Exact cell averages (recompute)"
ZOOM_POINT_TIP = ("The query server evaluates the dragged region again, point by point, at the resolution below "
                  "(no grid involved); every field of the list is shown from that one evaluation. Start the server "
                  "first.")
ZOOM_EXACT_TIP = ("Runs DTFE / PS-DTFE again on the dragged region, one slice cell thick: every cell the exact "
                  "average over its volume (mass is conserved), on a grid of that many cells across. A whole run, so "
                  "slower, and no server needed; the GPU is used when the program has it. A phase-space run "
                  "tessellates the whole box as the original run did (periodic or not) and deposits only into the "
                  "region, so the zoom shows exactly what a full run at that resolution would show there. It reuses "
                  "the query server's tessellations on disk and skips the partitions whose streams never reach the "
                  "region; a zoom that would take more than a minute asks first, with an estimate.")


def default_cache_dir(near: str) -> Path:
    """Where the query server keeps its tessellations: beside the data they come from (a TNG
    output: the data root; your own file's output: its folder) -- that disk has the room, ~0.15 KB
    per tessellation vertex -- unless it is iCloud Drive or read-only, else the user cache."""
    root = Path(near)
    if str(near).strip() and root.is_dir() and "Mobile Documents" not in str(root.resolve()) \
            and os.access(root, os.W_OK):
        return root / ".dtfe-tessellation-cache"
    base = Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache")
    return base / "DTFE" / "tessellations"


def kept_index(frac: float, n: int) -> int:
    """The cell of an n-cell axis holding the place 'frac' of the box along it (0 = the low face, 1 = the high
    one): the slice another output or redshift is shown at. A place on a cell face (a coarser grid's
    cell centre on a finer one) takes the upper cell, whatever float rounding does; clamped to the grid, so the
    same grid gives back its own index: (i + 0.5) / n -> i."""
    return int(min(n - 1, max(0, int(np.floor(float(frac) * n + 1e-9)))))


# ==================================================================== the difference map
COMPARE_MODES = (("A − B", "diff"), ("log₁₀ A/B", "ratio"))
NAN_GREY = (150, 150, 150)                 # a cell without a value (a ratio where A or B <= 0): not the scale's end


def compare_planes(a, b, mode: str) -> np.ndarray:
    """The difference map of two slices of the same grid: A - B, or log10(A/B) where both are positive
    (NaN elsewhere, drawn grey)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"the two outputs' slices differ in shape: {a.shape} and {b.shape}")
    if mode == "ratio":
        ok = (a > 0) & (b > 0)
        return np.where(ok, np.log10(np.where(ok, a, 1.0) / np.where(ok, b, 1.0)), np.nan)
    return a - b


# Explore field -> dtfelib.io.PointPlane field of a point-evaluated hi-res slice
PTS_FIELDS = {"density": "density", "streams": "streams", "velocity": "velocity", "dispersion": "dispersion",
              "divergence": "velDiv", "shear": "velShear", "vorticity": "velVort", "caustic": "caustic"}
PTS_MAX_SIDE = 4096                        # an 8192^2 plane is shown block-averaged to 4096^2 (screen and memory)
NO_SNAPSHOT_TEXT = "no snapshot known for this output (its input file is not beside it): choose it with Snapshot…"


def pts_plane(o, side, field: str, component: str):
    """(plane[u, v] for the map, the sidecar dict, the block factor) of a point-evaluated hi-res slice: the same
    quantity as the grid field where both exist -- the dispersion as the trace sigma^2 (the grid's '.velDisp'),
    the caustic as the fold flag, the velocity as its norm or a component; the shear and the vorticity as the
    magnitudes PointPlane derives."""
    from dtfelib.io import PointPlane
    pp = PointPlane(o.root, side, redshift=o.redshift)
    name = PTS_FIELDS.get(field)
    if name is None or name not in pp.available():
        raise ValueError(f"this hi-res slice has no {G.FIELD_LABELS.get(field, field)}"
                         + (" (it needs the velocity gradient points)" if field in ("divergence", "shear", "vorticity")
                            else ""))
    arr = pp.field(name)
    if field == "dispersion":
        arr = arr[..., 0] + arr[..., 3] + arr[..., 5]          # xx + yy + zz = sigma^2
    elif field == "caustic":
        arr = ((np.asarray(arr).astype(np.int64) & 3) == 3).astype(np.float64)
    elif arr.ndim == 3:
        comps = G.components("velocity", 3, 3)
        if component in comps[1:]:
            arr = arr[..., comps.index(component) - 1]
        else:
            arr = np.sqrt((np.asarray(arr, dtype=np.float64) ** 2).sum(axis=-1))
    arr = np.asarray(arr, dtype=np.float64)
    f = max(1, int(np.ceil(max(arr.shape) / PTS_MAX_SIDE)))
    if f > 1:                                  # block mean (a mask: any)
        nv, nu = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
        blocks = arr[:nv, :nu].reshape(nv // f, f, nu // f, f)
        arr = blocks.max(axis=(1, 3)) if field == "caustic" else blocks.mean(axis=(1, 3))
    return np.ascontiguousarray(arr.T), pp.side, f       # [u, v]: the slice map's plane[row = first in-plane axis]


# ==================================================================== workers
def map_contours(plane, req: dict, vrange, compare: bool = False) -> dict | None:
    """The contour lines of a map (contours.map_lines on its plane, its colour range 'vrange' as shown, its scale,
    colour map and the level slider's detail) with the canvas's paths of them ('paths', image pixels), their marks for
    the colour bar ('marks') and, when the request names the canvas's layer size ('contour_layer':
    ImageCanvas.layer_size) and this plane's image has the size it was for, the lines already stroked at it ('layer':
    stroke_layer); None where contours do not apply. Run with the slice on the worker's thread, or for a zoom on the
    GUI thread."""
    lines = CT.map_lines(plane, req["field"], vrange, req["log"], req["cmap"], compare=compare,
                         detail=req.get("contour_detail", CT.DETAIL))
    if lines is not None:
        lines["paths"] = contour_paths(CT.image_segments(lines), lines["colours"])
        lines["marks"] = (lines["levels"], lines["colours"])
        size = req.get("contour_layer")
        if size is not None and tuple(size[3:]) == tuple(np.shape(plane)):      # image = plane.T[::-1]: (n0, n1)
            lines["layer"] = (tuple(size), lines["paths"], stroke_layer(lines["paths"], size))
    return lines


def failure_text(e: Exception) -> str:
    """A worker's failure as the view says it: an error's class and text, or a refusal's own sentence (the
    3D smoothing a cube it cannot keep: grids.Smooth3DRefused)."""
    return str(e) if isinstance(e, G.Smooth3DRefused) else f"{type(e).__name__}: {e}"


class SliceWorker(QObject):
    """Reads and colours one slice off the GUI thread; only the latest request is served. The figure
    grid's tiles come the same way (render_grid): coloured straight to pixels, no matplotlib."""
    done = Signal(object)       # dict: rgb, plane, vrange, request
    failed = Signal(str)
    grid_done = Signal(object)  # dict: request, rgb (the tiled composite), labels, bar, shown
    grid_failed = Signal(str)
    done3d = Signal(object)     # explore_3d.planes3d's dict: the 3D view's planes
    failed3d = Signal(str)

    @Slot(object)
    def render3d(self, req: dict):
        try:
            self.done3d.emit(planes3d(req))
        except Exception as e:                  # a partial file, a vanished disk
            self.failed3d.emit(failure_text(e))

    @Slot(object)
    def render_grid(self, req: dict):
        try:
            items = grid_items(req)
            rgb, labels, bar = grid_tiles(req, items)
            circles, dashed, clips = grid_tile_circles(req, items)
            marks, mark_clips = grid_tile_marks(req, items)     # the centres of the voids drawn as ellipsoids
            segs, colours, shared = grid_tile_contours(req, items) if req.get("contours") else ([], [], None)
            paths = contour_paths(segs, colours) if segs else None
            size = req.get("contour_layer")             # stroked here at the size the canvas shows the grid
            layer = (tuple(size), paths, stroke_layer(paths, size)) \
                if paths and size is not None and tuple(size[3:]) == (rgb.shape[1], rgb.shape[0]) else None
            self.grid_done.emit({"request": req, "rgb": rgb, "labels": labels, "bar": bar,
                                 "shown": sum(1 for it in items if it is not None),
                                 "circles": circles, "dashed": dashed, "clips": clips, "marks": marks,
                                 "mark_clips": mark_clips,
                                 "contours": paths, "contour_layer": layer,
                                 "counts": [int(np.size(it["lines"]["levels"])) for it in items
                                            if it is not None and it.get("lines")] if req.get("contours") else None,
                                 "lines": shared if bar is not None else None,
                                 "title": grid_suptitle(req, items)})
        except Exception as e:
            self.grid_failed.emit(failure_text(e))

    @Slot(object)
    def render(self, req: dict):
        try:
            out: G.OutputSet = req["output"]
            pts_side = pts_factor = None
            if req.get("pts"):                      # the point-evaluated hi-res slice in place of the grid's
                plane, pts_side, pts_factor = pts_plane(out, req["pts"], req["field"], req["component"])
                plane = G.smoothed(plane, req.get("smooth", 0))
            else:                                   # (a phase-space output's velocity derivatives: the family
                grad = req.get("ps_gradient", "stored")     # asked for, grids.OutputSet.slice; smoothed in 2D, or
                plane = G.map_plane(out, req["field"], req["axis"], req["index"], req["component"], grad,   # the
                                    req.get("smooth", 0), req.get("smooth3d", False))     # cube's slice in 3D
            planes = None
            if req.get("compare") is not None:      # B read and smoothed the same way; the map is A - B or log10 A/B
                b = G.map_plane(req["compare"], req["field"], req["axis"], req["index"], req["component"],
                                req.get("ps_gradient", "stored"), req.get("smooth", 0), req.get("smooth3d", False))
                planes = (plane, b)
                plane = compare_planes(plane, b, req["compare_mode"])
            # image: horizontal = the first remaining axis, vertical = the second, increasing upward
            img = np.ascontiguousarray(plane.T[::-1])
            rgb, vrange = G.colorize(img, req["cmap"], log=req["log"], clip=(req["clip"], 100 - req["clip"]),
                                     symmetric=req["symmetric"], vrange=req.get("vrange"))
            if planes is not None:
                rgb[~np.isfinite(img)] = NAN_GREY   # colorize paints NaN as vmin: on RdBu_r that would read 'A >> B'
            arrows = out.arrow_field(req["field"], req["axis"], req["index"], smooth=req.get("smooth", 0),
                                     gradient=req.get("ps_gradient", "stored")) \
                if req.get("arrows") and not req.get("pts") else None
            lines = map_contours(plane, req, vrange, compare=planes is not None) if req.get("contours") else None
            self.done.emit({"rgb": np.ascontiguousarray(rgb), "plane": plane, "vrange": vrange, "request": req,
                            "arrows": arrows, "planes": planes, "pts_side": pts_side, "pts_factor": pts_factor,
                            "contours": lines})
        except Exception as e:                  # a partial file, a vanished disk
            self.failed.emit(failure_text(e))


# ==================================================================== canvas

# the finest virtual full grid an exact zoom of a phase-space run asks for along an axis (--ps-window
# allocates the window only; the grid just sets the cell size)
WINDOW_GRID_MAX = 1 << 20

# The time an exact zoom of a phase-space run will take, from measured rates (TNG100-3-Dark,
# 94M particles, from an external SSD, M1 Max): reading snapshot + initial conditions 16 s for 6.0 GB
# (skipped when the snapshot's globals are recorded and every needed partition is cached); per BUILT
# partition 3 s selecting its particles (all 94M scanned; a cached one skips it); 8.4 s loading a
# cached single-stream 6.0M-vertex tessellation (a 0.88 GB file) -- a chunked file loads 1.68x faster
# (measured 1.53 -> 0.91 s on the synthetic 2.1M-particle partitions); ~1.6 s for its exact deposit; building one
# from scratch 26 s per 7M vertices (the production run) plus writing it to the cache. The binary runs
# several partitions at a time (its auto-tuner, from the free memory): the estimate assumes EST_CONCURRENCY.
EST_READ_BPS = 3.8e8
EST_SELECT_S_PER_PARTICLE = 3.2e-8
EST_LOAD_BPS = 1.05e8
EST_CHUNKED_LOAD_SPEEDUP = 1.68
EST_DEPOSIT_S_PER_VERTEX = 2.7e-7
EST_BUILD_S_PER_VERTEX = 3.7e-6
EST_SAVE_BPS = 1.5e8
EST_VERTICES_PER_BYTE = 6.0e6 / 0.884e9
EST_CONCURRENCY = 3
EXACT_CONFIRM_SECONDS = 60      # ask first when an exact zoom will take longer than this

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# zooms kept in memory (the Back button and repeated drags reuse them without a new evaluation)
ZOOM_MEMO_BYTES = 1.5e9
_REQ_LINE = re.compile(r"\[request (\d+)/(\d+)\] partition \d+(?:: (in memory|loading from the cache|building its cell index)| done)")
_REQ_DONE = re.compile(r"\[request done\] (\d+) points, (\d+) of (\d+) partitions, (\d+) loaded from the cache in ([\d.]+) s")


def _void_tool():
    """tools/void_outlines.py as a module (its write_csv), loaded once by path: python/tools is not a package."""
    global _VOID_MOD
    if _VOID_MOD is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("void_outlines", VOID_TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VOID_MOD = mod
    return _VOID_MOD


_VOID_MOD = None


# ==================================================================== the two panes
class ExploreView(QWidget):
    """Right pane: the map, the readout and the query answer."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.title = QLabel("<b>Explore</b>")
        v.addWidget(self.title)
        # one dialog for every save: presets, title, axes, units, colour bar, overlays, margins, full resolution,
        # format; remembered between sessions. The buttons are the view's; ExploreControls places them in the
        # left pane's Map group, beside the controls they save the result of
        self.save_opts_btn = QPushButton("Save options…")
        tip_auto(self.save_opts_btn, "How Save image and Save figure write the picture: a preset or each option. "
                                     "The presets: Standard, Publication, Bare image, Slide, your last choice; the "
                                     "options: title, frame, ticks, axis names and unit, colour bar, arrows and void "
                                     "outlines, margins, dpi, width, full resolution, PNG, PDF, SVG or TIFF")
        self.save_btn = QPushButton("Save image…")
        self.canvas = ImageCanvas()
        v.addWidget(self.canvas, 1)
        self.view3d = View3D()                  # in the canvas's place while 'Show in 3D' is on
        self.view3d.setVisible(False)
        v.addWidget(self.view3d, 1)
        self.colorbar = ColorBar()
        v.addWidget(self.colorbar)
        self.readout = QLabel(" ")
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.readout)
        self.answer_title = QLabel("")
        self.answer_title.setWordWrap(True)
        v.addWidget(self.answer_title)
        self.streams = QTableWidget(0, 6)
        self.streams.setHorizontalHeaderLabels(["stream", "ρ/ρ̄", "v (km/s)", "|v|", "came from q (Mpc)", "value"])
        self.streams.verticalHeader().setVisible(False)
        self.streams.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hdr = self.streams.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)        # 'came from q': three coordinates
        # ResizeToContents measures 'ρ/ρ̄' without its combining bar: a few characters more keep the header whole
        fm = hdr.fontMetrics()
        hdr.setMinimumSectionSize(fm.horizontalAdvance("ρ/ρ̄") + 3 * fm.averageCharWidth())
        self.streams.setMaximumHeight(170)
        v.addWidget(self.streams)


class ExploreControls(View3DMixin, VoidsMixin, SaveMixin, GridMixin, ServerMixin, ZoomMixin, QWidget):
    """Left pane. Owns the workers; talks to an ExploreView."""
    status = Signal(str)
    changed = Signal()          # a map shown or gone, or the server's phase: the menu bar follows
    # queued into the worker threads
    _to_render = Signal(object)
    _to_render_grid = Signal(object)
    _to_render3d = Signal(object)
    _to_start = Signal(object)
    _to_query = Signal(object)
    _to_zoom = Signal(object)
    _to_exact = Signal(object)
    _to_stop = Signal()

    SAVE_KEY = "save_options"                   # the QSettings key of the Save options (JSON)

    def __init__(self, view: ExploreView, data_root_fn, extra_dirs_fn, settings=None):
        super().__init__()
        self.view = view
        # the Save options: remembered in 'settings' (the launcher's QSettings when it remembers; None = this
        # session only, as the tests run); the defaults write the reference pictures of Save image and Save figure
        self._settings = settings
        self.save_options = FS.loads(settings.value(self.SAVE_KEY, "")) if settings is not None else FS.options()
        self._grid_title = ""                   # the figure grid's own title (its worker sends it)
        # the single map's title as a function of the unit of its positions: units = None (the screen's: Mpc, or
        # cells without a box) or ((axis, position) -> (value, unit name)), a saved figure's; None
        # while the figure grid is shown
        self._title_make = None
        self._save_note = ""                    # why a saved figure's axes are not in the unit asked for
        self._data_root = data_root_fn          # () -> current data root
        self._extra_dirs = extra_dirs_fn        # () -> the custom-snapshot output directories
        self.outputs: list[tuple[str, G.OutputSet]] = []
        self.current: G.OutputSet | None = None
        self._plane = None
        self._planes = None                     # (A, B) of a difference map: the hover reads both
        self._arrows = None
        self._contours = None                   # map_contours() of the map on screen: the lines, their levels
        self._req = None
        self._pending = None
        self._busy = False
        self._server_output: G.OutputSet | None = None
        self._server_log: Path | None = None
        self._zoom: dict | None = None          # the zoomed region (see _dragged) once its fields arrived
        self._zoom_fields = None                # dtfelib PointFields on the zoom's plane of points
        self._zoom_pending = False
        self._grid_shown = False                # the figure grid is on the right pane instead of the map
        self._grid_index = None                 # the slice index the shown grid was composed at
        self._grid_busy = False                 # a tile request is with the worker; only the latest is served
        self._grid_pending = None
        self._grid_type_pref: dict = {}         # row -> (prefix, estimator) the user chose; a rebuild's fallback never replaces it
        # the slice kept across the outputs and redshifts of one simulation: the first
        # menu's entry it belongs to, the output last shown (its box, for the custom runs and opened files), the axis,
        # the place along it as a fraction of the box (the user's own, not re-read off a coarser grid on the way) and
        # the zoom window carried along (_carry_zoom). None until the first output is shown
        self._keep: dict | None = None

        col = QVBoxLayout(self)
        # the left pane, one group at a time (each _build_* adds its group to 'col'; the attribute names are
        # the ones the tests and the mixins use)
        self._build_output(col)
        self._build_map(col)
        self._build_3d(col)                     # the 3D view's group (explore_3d)
        self._build_grid(col)
        self._build_streams(col)
        self._build_zoom(col)
        col.addStretch(1)
        self._build_workers()
        self._wire_view(view)

    def _build_output(self, col):
        """The 'Output' group: the simulation, its snapshot by redshift and (several on one) the output."""
        g = QGroupBox("Output")
        f = QFormLayout(g)
        self._out_form = f
        # the output to show, in two steps: the simulation (or the custom snapshots, or the files opened
        # by hand), then its snapshot by redshift (a custom run by name). A third menu appears only when
        # that snapshot has several outputs (prefixes, e.g. the phase-space and the standard run).
        self._by_root: dict[str, G.OutputSet] = {}
        self._pref_prefix = "ps_output"         # the prefix the user chose last: kept across snapshot and simulation changes
        self.sim_combo = QComboBox()
        self.sim_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.sim_combo.setMinimumContentsLength(16)
        self.sim_combo.setToolTip("The simulation whose outputs to show; the custom snapshots and the files opened "
                                  "with the … button have an entry of their own")
        self.sim_combo.currentIndexChanged.connect(self._sim_changed)
        row = QHBoxLayout()
        row.addWidget(self.sim_combo, 1)
        refresh = QToolButton(text="↻")
        refresh.setToolTip("Look for outputs again")
        refresh.clicked.connect(lambda: self.refresh_outputs())
        row.addWidget(refresh)
        open_btn = QToolButton(text="…")
        open_btn.setToolTip("Open any output: choose one of its grid files (e.g. <name>.a_den)")
        open_btn.clicked.connect(self._open_other)
        row.addWidget(open_btn)
        f.addRow("Simulation", row)
        self.snap_combo = QComboBox()
        self.snap_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.snap_combo.setMinimumContentsLength(16)
        self.snap_combo.setToolTip("The snapshot, by redshift (earliest first, today last); a custom run by its name")
        self.snap_combo.currentIndexChanged.connect(self._snap_changed)
        f.addRow("Redshift", self.snap_combo)
        self.prefix_combo = QComboBox()
        self.prefix_combo.setToolTip("This snapshot has several outputs (their file prefixes): which one to show")
        self.prefix_combo.currentIndexChanged.connect(self._prefix_changed)
        f.addRow("Output", self.prefix_combo)
        f.setRowVisible(self.prefix_combo, False)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray")
        f.addRow("", self.info)
        col.addWidget(g)

    def _build_map(self, col):
        """The 'Map' group: field, component, the void outlines and their options, the slice, colours, range, smoothing,
        the comparison and the hi-res slice."""
        g = QGroupBox("Map")
        f = QFormLayout(g)
        self._out_form_map = f
        self.field = QComboBox()
        self.field.currentIndexChanged.connect(self._field_changed)
        f.addRow("Field", self.field)
        self.component = QComboBox()
        self.component.currentIndexChanged.connect(self.render)
        crow = QHBoxLayout()
        crow.addWidget(self.component, 1)
        self.vectors = QCheckBox("vectors")
        self.vectors.setToolTip("Draw the in-plane components as arrows over the map")
        self.vectors.toggled.connect(self.render)
        crow.addWidget(self.vectors)
        f.addRow("Component", crow)
        # the contour lines (the tick is the row's label, as the voids' below) and how many: the level slider, its
        # positions each a slightly closer spacing (contours.DETAIL_MIN..DETAIL_MAX; 0 the rule's own), the lines on
        # the map counted; no tick marks (dozens of positions would make a comb of them)
        self.contours = QCheckBox("Contours")
        self.contours.setToolTip(CONTOURS_TIP)
        self.contours.toggled.connect(self.render)
        lrow = QHBoxLayout()
        self._contour_ends = (QLabel("fewer"), QLabel("more"))     # greyed out with the slider
        lrow.addWidget(self._contour_ends[0])
        self.contour_detail = QSlider(Qt.Horizontal)
        self.contour_detail.setRange(CT.DETAIL_MIN, CT.DETAIL_MAX)
        self.contour_detail.setValue(CT.DETAIL)
        self.contour_detail.setSingleStep(1)
        self.contour_detail.setPageStep(4)
        self.contour_detail.setFixedWidth(140)
        self.contour_detail.setToolTip(CONTOUR_DETAIL_TIP)
        self._contour_drag = QTimer(self, singleShot=True, interval=CONTOUR_DRAG_MS)
        self._contour_drag.timeout.connect(self.render)
        self.contour_detail.valueChanged.connect(self._contour_detail_moved)
        self.contour_detail.sliderReleased.connect(self._contour_detail_released)
        lrow.addWidget(self.contour_detail)
        lrow.addWidget(self._contour_ends[1])
        self.contour_count = QLabel("")
        self.contour_count.setStyleSheet("color: gray")
        lrow.addWidget(self.contour_count)
        lrow.addStretch(1)
        f.addRow(self.contours, lrow)
        for w in (self.contour_detail, *self._contour_ends):
            w.setEnabled(False)
        # a phase-space output's velocity derivatives: the mean field (computed) or the stored stream average
        self.ps_gradient = QComboBox()
        for text, mode in PS_GRADIENT_ITEMS:
            self.ps_gradient.addItem(text, mode)
        self.ps_gradient.setCurrentIndex(self.ps_gradient.findData(G.PS_GRADIENT_DEFAULT))
        tip_auto(self.ps_gradient, PS_GRADIENT_TIP)
        self.ps_gradient.currentIndexChanged.connect(self._ps_gradient_changed)
        f.addRow("PS derivatives", self.ps_gradient)
        self.ps_gradient_note = QLabel("")
        self.ps_gradient_note.setWordWrap(True)
        self.ps_gradient_note.setStyleSheet("color: gray")
        f.addRow("", self.ps_gradient_note)
        f.setRowVisible(self.ps_gradient_note, False)
        # the voids (the tick is the row's label) and their options: how they are drawn (their ellipsoids or spheres
        # of R_v, on the map and in the 3D view alike), the other outline too (dashed), the smoothing they are found
        # at, a jump to the slice through the centre of the void last clicked; under it a grey line (a search running,
        # why there are none, or which field they belong to)
        self.voids = QCheckBox("Voids")
        tip_auto(self.voids, VOID_TIP)
        self.voids.toggled.connect(self._voids_toggled)
        vrow = QHBoxLayout()
        self.void3d_shape = QComboBox()                     # (its old name: the 3D view's tests and state know it)
        for text, shape in SHAPE_ITEMS:
            self.void3d_shape.addItem(text, shape)
        tip_auto(self.void3d_shape, SHAPE_TIP)
        self.void3d_shape.currentIndexChanged.connect(self._void_shape_changed)
        self.void_shape_combo = self.void3d_shape
        vrow.addWidget(self.void3d_shape)
        self.void_reff = QCheckBox("R_v too (dashed)")
        self.void_reff.toggled.connect(self._void_reff_toggled)
        vrow.addWidget(self.void_reff)
        vrow.addWidget(QLabel("σ"))
        self.void_sigma = XV.VoidSigmaBox()
        self.void_sigma.setRange(0.0, 50.0)
        self.void_sigma.setDecimals(3)
        self.void_sigma.setSingleStep(0.1)
        self.void_sigma.setSuffix(" Mpc")
        self.void_sigma.setSpecialValueText("production")
        tip_auto(self.void_sigma, VOID_SIGMA_TIP)
        self.void_sigma.setKeyboardTracking(False)          # one search per value typed, not per keystroke
        self.void_sigma.valueChanged.connect(self._void_sigma_edited)
        self._void_sigma_now = self._void_sigma_back = 0.0  # the smoothing committed (the catalogue's key), the one
        self._void_refused = ""                             # before it; a search declined or stopped (said in a line)
        self._void_sigma_pause = QTimer(self, singleShot=True, interval=XV.SIGMA_PAUSE_MS)
        self._void_sigma_pause.timeout.connect(self._void_sigma_commit)
        vrow.addWidget(self.void_sigma)
        self.void_goto = QPushButton("Go to its centre")
        self.void_goto.setToolTip("Show the slice through the centre of the void last clicked")
        self.void_goto.setEnabled(False)
        self.void_goto.clicked.connect(self._void_goto)
        vrow.addWidget(self.void_goto)
        vrow.addStretch(1)
        f.addRow(self.voids, vrow)
        self.void_note = QLabel("")
        self.void_note.setWordWrap(True)
        self.void_note.setStyleSheet("color: gray")
        f.addRow("", self.void_note)
        f.setRowVisible(self.void_note, False)
        self._void_tick = QTimer(self, interval=1000)       # the search's time so far, in the line, while it runs
        self._void_tick.timeout.connect(self._void_note)
        self._void_t0 = self._void_est = None
        self.confirm_void = None                # asked before a long void search: None = a dialog (tests: a callable)
        self._void_other_text()
        # the voids drawn, by size: R from .. to .. (0 = no bound), and how many that leaves
        srow = QHBoxLayout()
        srow.addWidget(QLabel("R from"))
        self.void_rmin, self.void_rmax = (QDoubleSpinBox(), QDoubleSpinBox())
        for w in (self.void_rmin, self.void_rmax):
            w.setRange(0.0, 1000.0)
            w.setDecimals(1)
            w.setSingleStep(1.0)
            w.setSuffix(" Mpc")
            w.setSpecialValueText("any")
            tip_auto(w, VOID_RANGE_TIP)
            w.setKeyboardTracking(False)            # one redraw per value typed, not per keystroke
            w.valueChanged.connect(self._void_range_changed)
        srow.addWidget(self.void_rmin)
        srow.addWidget(QLabel("to"))
        srow.addWidget(self.void_rmax)
        self.void_count = QLabel("")
        self.void_count.setStyleSheet("color: gray")
        self.void_count.setToolTip(VOID_COUNT_TIP)
        srow.addWidget(self.void_count)
        srow.addStretch(1)
        f.addRow("Sizes", srow)
        self.void_info = QLabel("")
        self.void_info.setWordWrap(True)
        self.void_info.setTextFormat(Qt.RichText)
        self.void_info.setStyleSheet("color: gray")
        f.addRow("", self.void_info)
        self._void_picked = None                # (cache key, catalogue row) of the void last clicked
        ax = QHBoxLayout()
        self.axis = QComboBox()
        self.axis.addItems([f"{a} (normal)" for a in AXES])
        self.axis.setCurrentIndex(2)
        self.axis.currentIndexChanged.connect(self._axis_changed)
        ax.addWidget(self.axis)
        self.index = QSpinBox()
        self.index.valueChanged.connect(self._index_spin)
        ax.addWidget(self.index)
        f.addRow("Slice", ax)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._index_slider)
        f.addRow("", self.slider)
        self.position = QLabel("")
        self.position.setStyleSheet("color: gray")
        f.addRow("", self.position)
        cm = QHBoxLayout()
        self.cmap = QComboBox()
        self.cmap.addItems(CMAPS)
        self.cmap.currentIndexChanged.connect(self.render)
        cm.addWidget(self.cmap)
        self.log = QCheckBox("log₁₀")
        self.log.toggled.connect(self.render)
        cm.addWidget(self.log)
        f.addRow("Colours", cm)
        rg = QHBoxLayout()
        self.clip = QDoubleSpinBox(minimum=0.0, maximum=20.0, decimals=1, singleStep=0.5, value=1.0)
        self.clip.setSuffix(" % clipped at each end")
        self.clip.valueChanged.connect(self.render)
        rg.addWidget(self.clip)
        f.addRow("Range", rg)
        fx = QHBoxLayout()
        self.fixed = QCheckBox("fixed")
        self.fixed.setToolTip("Colour every map on this range (the field's own values; a log map takes positive "
                              "ones): the colours then compare between snapshots, outputs and slices")
        self.fixed.toggled.connect(self._fixed_toggled)
        fx.addWidget(self.fixed)
        self.vmin_edit, self.vmax_edit = QLineEdit(placeholderText="min"), QLineEdit(placeholderText="max")
        self._fixed_for = None              # the field the fixed range was set for: another field drops it
        self._fixed_warned = False
        for e in (self.vmin_edit, self.vmax_edit):
            e.setMaximumWidth(90)
            e.editingFinished.connect(self._fixed_edited)
            fx.addWidget(e)
        take = QToolButton(text="this map's")
        take.setToolTip("Fill in the range of the map on screen and fix it")
        take.clicked.connect(self._take_range)
        fx.addWidget(take)
        self.take_range_btn = take
        fx.addStretch(1)
        f.addRow("", fx)
        self.smooth = QDoubleSpinBox(minimum=0.0, maximum=20.0, decimals=1, singleStep=0.5)
        self.smooth.setSuffix(" cells (Gaussian)")
        self.smooth.valueChanged.connect(self.render)
        smrow = QHBoxLayout()
        smrow.addWidget(self.smooth)
        # the smoothing in 3D: the slice of the whole cube smoothed (the voids' own field), kept per field and value
        self.smooth3d = QCheckBox("3D")
        tip_auto(self.smooth3d, SMOOTH3D_TIP)
        self.smooth3d.toggled.connect(self._smooth3d_toggled)
        smrow.addWidget(self.smooth3d)
        smrow.addStretch(1)
        f.addRow("Smoothing", smrow)
        cmp_row = QHBoxLayout()
        self.compare = QComboBox()
        self.compare.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.compare.setMinimumContentsLength(12)
        tip_auto(self.compare, "Show this map minus another output of the same grid, on a diverging scale centred "
                               "on 0. The same grid: the same cells, from another estimator, prefix, precision or "
                               "deposit of this snapshot, or another snapshot. Slice maps only: a zoom shows A alone")
        self.compare.currentIndexChanged.connect(self._compare_changed)
        cmp_row.addWidget(self.compare, 1)
        self.compare_mode = QComboBox()
        for text, mode in COMPARE_MODES:
            self.compare_mode.addItem(text, mode)
        self.compare_mode.setToolTip("A − B, or log₁₀ A/B where both are positive (grey where not): for density-like fields")
        self.compare_mode.currentIndexChanged.connect(self.render)
        cmp_row.addWidget(self.compare_mode)
        f.addRow("Compare with", cmp_row)
        self.pts_combo = QComboBox()
        self.pts_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.pts_combo.setMinimumContentsLength(12)
        tip_auto(self.pts_combo, "Show the point-evaluated hi-res slice of this output ('.pts_*', the Pipeline's "
                                  "image plane: the field at each pixel centre, no grid) in place of the grid slice. "
                                  "Planes of one size are listed with their axis: the files do not say which they are")
        self.pts_combo.currentIndexChanged.connect(self._pts_changed)
        f.addRow("Hi-res slice", self.pts_combo)
        f.setRowVisible(self.pts_combo, False)
        self._pts_side = None
        save_row = QHBoxLayout()                # the view's save buttons, here beside the controls
        save_row.addWidget(self.view.save_btn)
        save_row.addWidget(self.view.save_opts_btn)
        save_row.addStretch(1)
        f.addRow("Save", save_row)
        col.addWidget(g)

    def _build_grid(self, col):
        """The 'Compare: a figure grid' group: the layout, the panels' table, the fill buttons, show and save."""
        g = QGroupBox("Compare: a figure grid")
        v = QVBoxLayout(g)
        v.addWidget(self._note("Several maps as one figure: a field at several redshifts, or several fields of one "
                               "snapshot. Every panel takes this map's slice (the same axis, the same position across "
                               "the box), its range clip and smoothing; panels of one field share a colour range."))
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Layout"))
        self.grid_layout = QComboBox()
        for r, c in GRID_LAYOUTS:
            self.grid_layout.addItem(f"{r} × {c}", f"{r}x{c}")     # a string: Qt matches it in findData
        self.grid_layout.setToolTip("Rows × columns of the figure")
        self.grid_layout.currentIndexChanged.connect(self._grid_layout_changed)
        lrow.addWidget(self.grid_layout)
        lrow.addWidget(QLabel("rows × columns"))
        lrow.addStretch(1)
        v.addLayout(lrow)
        self.grid_table = QTableWidget(0, len(GRID_COLS))
        self.grid_table.setHorizontalHeaderLabels(list(GRID_COLS))
        self.grid_table.verticalHeader().setVisible(False)
        self.grid_table.verticalHeader().setDefaultSectionSize(26)
        self.grid_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grid_table.horizontalHeader().setSectionResizeMode(_C_CMP, QHeaderView.ResizeToContents)
        self.grid_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.grid_table.setMaximumHeight(9 * 26 + 30)
        tip_auto(self.grid_table, "One row per panel, in reading order: the simulation, its snapshot by redshift, "
                                   "which of that snapshot's outputs (DTFE, PS-DTFE, another prefix), the field and "
                                   "the component. A row without a simulation leaves that panel blank. A row keeps "
                                   "its output type across redshifts when the snapshot has it.")
        v.addWidget(self.grid_table)
        frow = QHBoxLayout()
        self.grid_fill_z = QPushButton("This field across redshifts")
        self.grid_fill_z.setToolTip("Fill the panels with the map's field at this simulation's snapshots: the latest "
                                    "ones that have it, oldest first")
        self.grid_fill_z.clicked.connect(lambda: self.grid_fill("redshifts"))
        frow.addWidget(self.grid_fill_z)
        self.grid_fill_f = QPushButton("This snapshot across fields")
        self.grid_fill_f.setToolTip("Fill the panels with this output's fields, starting at the map's field")
        self.grid_fill_f.clicked.connect(lambda: self.grid_fill("fields"))
        frow.addWidget(self.grid_fill_f)
        self.grid_fill_t = QPushButton("This snapshot across outputs")
        self.grid_fill_t.setToolTip("Fill the panels with the map's field from every output of this snapshot: the "
                                    "standard DTFE and the phase-space runs side by side")
        self.grid_fill_t.clicked.connect(lambda: self.grid_fill("types"))
        frow.addWidget(self.grid_fill_t)
        frow.addStretch(1)
        v.addLayout(frow)
        brow = QHBoxLayout()
        self.grid_toggle = QPushButton("Show the grid")
        self.grid_toggle.setCheckable(True)
        self.grid_toggle.setToolTip("Compose the panels and show the figure in place of the map (any change to the map "
                                    "brings the map back)")
        self.grid_toggle.toggled.connect(self.show_grid)
        brow.addWidget(self.grid_toggle)
        self.grid_save_btn = QPushButton("Save figure…")
        self.grid_save_btn.clicked.connect(self._save_grid_dialog)
        brow.addWidget(self.grid_save_btn)
        self.grid_opts_btn = QPushButton("Options…")
        self.grid_opts_btn.setToolTip("The Save options (the same as the map's): title, panel titles, axes, colour "
                                      "bars, margins, dpi, width, full resolution, format")
        self.grid_opts_btn.clicked.connect(self.open_save_options)
        brow.addWidget(self.grid_opts_btn)
        self.grid_shared = QCheckBox("one colour range per field")
        self.grid_shared.setChecked(True)
        self.grid_shared.setToolTip("Panels showing the same field share vmin/vmax (the union of their ranges), so "
                                    "the colours compare; unticked, every panel is stretched on its own")
        brow.addStretch(1)
        v.addLayout(brow)
        orow = QHBoxLayout()
        self.grid_shared.toggled.connect(lambda _on: self.show_grid(True) if self._grid_shown else None)
        orow.addWidget(self.grid_shared)
        self.grid_merge = QCheckBox("merge the axes: one colour bar")
        tip_auto(self.grid_merge, GRID_MERGE_TIP)
        self.grid_merge.toggled.connect(self._grid_merge_toggled)
        orow.addWidget(self.grid_merge)
        orow.addStretch(1)
        v.addLayout(orow)
        self.grid_state = QLabel("")
        self.grid_state.setWordWrap(True)
        self.grid_state.setStyleSheet("color: gray")
        v.addWidget(self.grid_state)
        col.addWidget(g)
        self._grid_sync_rows()

    def _build_streams(self, col):
        """The 'Streams at a point' group: the query server and its cache folder (how many partitions it keeps in
        memory is the binary's choice, from the free memory)."""
        g = QGroupBox("Streams at a point")
        v = QVBoxLayout(g)
        v.addWidget(self._note("With the server running, click the map."))
        tip_auto(g, "A query server on this output's snapshot: a click lists every stream at that point, with "
                     "its density, velocity and the Lagrangian position its matter came from. A big "
                     "snapshot is served as partitions kept on disk in the cache folder.")
        cf = QFormLayout()
        self.cache_edit = QLineEdit()
        self.cache_edit.textEdited.connect(lambda _t: setattr(self, "_cache_user_set", True))
        self._cache_user_set = False
        crow = QHBoxLayout()
        crow.addWidget(self.cache_edit, 1)
        cb = QToolButton(text="…")
        cb.clicked.connect(self._browse_cache)
        crow.addWidget(cb)
        cf.addRow("Cache folder", crow)
        v.addLayout(cf)
        self._snapshots: dict[str, str] = {}    # output root -> a snapshot file chosen by hand
        brow = QHBoxLayout()
        self.server_btn = QPushButton("Start query server")
        self.server_btn.clicked.connect(self._server_button)
        brow.addWidget(self.server_btn)
        snap_btn = QToolButton(text="Snapshot…")
        snap_btn.setToolTip("The snapshot file this output was computed from, when the launcher cannot find it "
                            "(the input moved, or an output opened by hand): the server and the exact zoom read it")
        snap_btn.clicked.connect(self._choose_snapshot)
        brow.addWidget(snap_btn)
        self.server_state = QLabel("not running")
        self.server_state.setWordWrap(True)
        brow.addWidget(self.server_state, 1)
        v.addLayout(brow)
        col.addWidget(g)

    def _build_zoom(self, col):
        """The 'Zoom' group: what a drag computes (point values or exact cell averages), the resolution, cancel, back
        and out, the progress and the state line."""
        g = QGroupBox("Zoom")
        v = QVBoxLayout(g)
        v.addWidget(self._note("Drag a rectangle on the map: the region again at the resolution below (point values "
                               "need the query server above). Drag again to go deeper."))
        self.zoom_kind = ChoiceRadios([("point", ZOOM_POINT_TEXT, ""), ("exact", ZOOM_EXACT_TEXT, "")])
        tip_auto(self.zoom_kind.button("point"), ZOOM_POINT_TIP)
        tip_auto(self.zoom_kind.button("exact"), ZOOM_EXACT_TIP)
        v.addWidget(self.zoom_kind)
        zrow = QHBoxLayout()
        self.zoom_res = QSpinBox(minimum=64, maximum=4096, value=512, singleStep=64)
        self.zoom_res.setSuffix(" per side")
        self.zoom_res.setToolTip("Points along each side of the zoomed region (exact cell averages: cells)")
        zrow.addWidget(self.zoom_res)
        self.zoom_cancel_btn = QPushButton("Cancel")
        self.zoom_cancel_btn.setToolTip("Stop the exact zoom that is running")
        self.zoom_cancel_btn.setEnabled(False)
        self.zoom_cancel_btn.clicked.connect(self.cancel_exact_zoom)
        zrow.addWidget(self.zoom_cancel_btn)
        self.zoom_back_btn = QPushButton("Back")
        self.zoom_back_btn.setToolTip("Back to the previous zoom level, kept in memory (no new evaluation); "
                                      "at the first level, back to the slice")
        self.zoom_back_btn.setEnabled(False)
        self.zoom_back_btn.clicked.connect(self.zoom_back)
        zrow.addWidget(self.zoom_back_btn)
        self.zoom_out_btn = QPushButton("Zoom out")
        self.zoom_out_btn.setEnabled(False)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zrow.addWidget(self.zoom_out_btn)
        zrow.addStretch(1)
        v.addLayout(zrow)
        prow = QHBoxLayout()
        self.zoom_progress = QProgressBar()
        self.zoom_progress.setTextVisible(False)
        self.zoom_progress.setMaximumHeight(10)
        self.zoom_progress.setVisible(False)
        prow.addWidget(self.zoom_progress, 1)
        self.zoom_time = QLabel("")
        self.zoom_time.setStyleSheet("color: gray")
        self.zoom_time.setToolTip("How long the zoom has been running (the last one: how long it took)")
        prow.addWidget(self.zoom_time)
        v.addLayout(prow)
        self.zoom_state = QLabel("")
        self.zoom_state.setWordWrap(True)
        self.zoom_state.setStyleSheet("color: gray")
        v.addWidget(self.zoom_state)
        col.addWidget(g)

    def _build_workers(self):
        """The slice worker, the query server and the exact zoom's worker, wired (their threads start on first use:
        _threads), and the zoom's timing and memory."""
        # workers: their threads start on first use (a window that never opens Explore owns none)
        self._slice_thread = self._server_thread = None
        self._slice_worker = SliceWorker()
        self._slice_worker.done.connect(self._rendered)
        self._slice_worker.failed.connect(self._render_failed)
        self._to_render.connect(self._slice_worker.render)
        self._to_render_grid.connect(self._slice_worker.render_grid)
        self._slice_worker.grid_done.connect(self._grid_rendered)
        self._slice_worker.grid_failed.connect(self._grid_failed)
        self._to_render3d.connect(self._slice_worker.render3d)
        self._slice_worker.done3d.connect(self._3d_rendered)
        self._slice_worker.failed3d.connect(self._3d_failed)
        self._server = ServerWorker()
        self._exact_worker = ExactZoomWorker()
        self._exact_thread = None
        self._void_cache: dict = {}             # output key -> the void_outlines.py JSON, or {"error": why}
        self._void_proc: QProcess | None = None  # the one catalogue being found
        self._void_key = None                   # ... and the output it is for
        self._void_out = None                   # ... and that output itself (re-queued when the map's search stops it)
        self._void_queue: list = []             # outputs whose catalogue the figure grid waits for, one at a time
        # asked before a memory-heavy action while a DTFE run is going: None = a dialog; tests set a
        # callable(title, text) -> bool
        self.confirm_heavy = None
        self._to_exact.connect(self._exact_worker.run)
        self._exact_worker.done.connect(self._zoomed)
        self._exact_worker.failed.connect(self._exact_failed)
        self._exact_worker.progress.connect(self._exact_progress)
        self._exact_running = False
        # asked before a long exact zoom: None = a dialog; tests set a callable(text) -> bool
        self.confirm_exact = None
        self.server_partition = None              # tests: force the query server's split (a composite)
        # zoom timing and progress: a 0.1 s tick updates the elapsed time and, for a point zoom, follows
        # the server's '[request k/P] ...' lines in its log (--serve-progress)
        self._zoom_t0: float | None = None
        self._zoom_kind = ""
        self._zoom_log_pos = 0
        self._zoom_summary = None                 # the server's '[request done]' figures of the running zoom
        self._zoom_tick = QTimer(self, interval=100)
        self._zoom_tick.timeout.connect(self._tick_zoom)
        # earlier zoom levels (Back) and recent evaluations (a repeated drag), kept in memory
        self._zoom_history: list[tuple[dict, object]] = []
        from collections import OrderedDict
        self._zoom_memo: "OrderedDict[tuple, tuple[dict, object]]" = OrderedDict()
        self._server.started.connect(self._server_started)
        self._server.failed.connect(self._server_failed)
        self._server.answered.connect(self._answered)
        self._server.zoomed.connect(self._zoomed)
        self._server.stopped.connect(lambda: self._set_server_state("stopped"))
        self._to_start.connect(self._server.start)
        self._to_query.connect(self._server.query)
        self._to_zoom.connect(self._server.zoom)
        self._to_stop.connect(self._server.stop)
        self._server_running = self._server_starting = False
        self._shut = False                      # shutdown() ran: nothing may start a thread again
        self._build_timer = QTimer(self, interval=1000)
        self._build_timer.timeout.connect(self._poll_build)

    def _wire_view(self, view):
        """The right pane's signals: the map's hover, click and drag, and the save buttons."""
        view.canvas.hovered.connect(self._hover)
        view.canvas.clicked.connect(self._click)
        view.canvas.dragged.connect(self._dragged)
        view.save_btn.clicked.connect(self._save_image)
        view.save_opts_btn.clicked.connect(self.open_save_options)
        self._save_tips()
        self._refresh_server_button()

    def _threads(self):
        if self._slice_thread is None:
            self._slice_thread, self._server_thread, self._exact_thread = QThread(self), QThread(self), QThread(self)
            self._slice_worker.moveToThread(self._slice_thread)
            self._server.moveToThread(self._server_thread)
            self._exact_worker.moveToThread(self._exact_thread)
            self._slice_thread.start()
            self._server_thread.start()
            self._exact_thread.start()

    @staticmethod
    def _note(text):
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: gray")
        return lab

    # ---------------------------------------------------------------- outputs
    @staticmethod
    def _same(a, b) -> bool:
        """The same output, by its files' root: a refresh replaces the OutputSet objects."""
        return a is not None and b is not None and str(a.root) == str(b.root)

    @staticmethod
    def _group(o) -> str:
        """The first menu's entry an output belongs to: its simulation, else the custom snapshots,
        else the files opened by hand (marked GROUP_OPENED by _open_other)."""
        return o.sim or getattr(o, "group", GROUP_CUSTOM)

    @staticmethod
    def _snap_key(o):
        """The second menu's key of an output: the snapshot number in a simulation, else its root."""
        if o.sim:
            return o.snap if o.snap is not None else -1
        return str(o.root)

    def refresh_outputs(self, select: str | None = None):
        """Look for outputs again and show 'select' (a root), else the one shown now, else the latest
        snapshot of the first simulation. Files opened by hand stay listed."""
        keep = select or (str(self.current.root) if self.current else None)
        found = G.find_outputs(self._data_root(), self._extra_dirs())
        listed = {str(o.root) for _, o in found}
        opened = [(t, o) for t, o in self.outputs if self._group(o) == GROUP_OPENED and str(o.root) not in listed]
        self.outputs = found + opened
        self._by_root = {str(o.root): o for _, o in self.outputs}
        self._rebuild_sims()
        if self._grid_shown:                    # the map comes back below (or the empty state): not a stale picture
            self._leave_grid()
        self._grid_sync_rows()
        if not self.outputs:
            for c in (self.snap_combo, self.prefix_combo):
                c.blockSignals(True)
                c.clear()
                c.blockSignals(False)
            self._out_form.setRowVisible(self.prefix_combo, False)
            self.current = self._plane = self._arrows = None
            self.view.canvas.arrows = None
            self._set_void_canvas(None)
            self._set_contours(None)
            self.info.setText("No grids yet: run the demo (Setup), a Grids job or a Custom snapshot job first.")
            self.view.canvas.set_rgb(None)
            self.view.canvas.setText("No outputs with grids found.")
            self._refresh_server_button()
            return 0
        if not (keep and self._select_output(keep)):
            self._rebuild_snaps()
            self._rebuild_prefixes()
            self._show_current()
        return len(self.outputs)

    def select_root(self, root: str):
        """Show the output whose files are '<root>.*' (the Runs browser's 'Explore')."""
        if not self._select_output(str(root)):
            self.refresh_outputs(select=str(root))

    def refresh_if_vanished(self) -> bool:
        """Look for outputs again when one listed here has lost its grids (the Runs browser's Move to Trash, the
        Finder): neither the menus nor the figure grid may offer it, and its next slice read would fail with
        FileNotFoundError. One stat per output; files opened by hand are the user's. True when it
        looked again."""
        for _, o in self.outputs:
            if self._group(o) != GROUP_OPENED and o.files and not next(iter(o.files.values())).path.is_file():
                self.refresh_outputs()
                return True
        return False

    def _rebuild_sims(self):
        """The first menu: the simulations with grids in natural order (TNG50, TNG100, TNG300), then
        the custom snapshots and the opened files when there are any. Keeps the current choice."""
        outs = [o for _, o in self.outputs]
        c = self.sim_combo
        cur = c.currentData()
        c.blockSignals(True)
        c.clear()
        for sim in sorted({o.sim for o in outs if o.sim}, key=G.sim_sort_key):
            n = len({o.snap for o in outs if o.sim == sim})
            c.addItem(sim, sim)
            c.setItemData(c.count() - 1, f"{n} snapshot{'s' if n != 1 else ''} with grids", Qt.ToolTipRole)
        for group, text in ((GROUP_CUSTOM, "Custom snapshots"), (GROUP_OPENED, "Opened files")):
            if any(self._group(o) == group for o in outs):
                c.addItem(text, group)
        i = c.findData(cur) if cur is not None else -1
        c.setCurrentIndex(i if i >= 0 else 0)
        c.blockSignals(False)

    def _rebuild_snaps(self, select=None, prefer=None):
        """The second menu for the chosen simulation: its snapshots by redshift (earliest first, so
        today is last and the default); for the custom snapshots and opened files, the runs by name.
        'select' (an OutputSet) picks its entry; else 'prefer' (a key: the snapshot shown in another
        simulation) when the menu has it."""
        grp = self.sim_combo.currentData()
        outs = [o for _, o in self.outputs if self._group(o) == grp]
        c = self.snap_combo
        c.blockSignals(True)
        c.clear()
        if grp in (GROUP_CUSTOM, GROUP_OPENED):
            label = "Run"
            for o in sorted(outs, key=lambda o: (o.dir.name.lower(), o.prefix.lower())):
                c.addItem(str(o.root) if grp == GROUP_OPENED else f"{o.prefix}  ({o.dir.name})", str(o.root))
                c.setItemData(c.count() - 1, str(o.root) + ".*", Qt.ToolTipRole)
        else:
            label = "Redshift"
            first = {}
            for o in outs:
                first.setdefault(self._snap_key(o), o)
            for key in sorted(first):
                o = first[key]
                z = f"z = {o.redshift:.2f}   ·   " if o.redshift is not None else ""
                c.addItem(z + (f"snapshot {key:03d}" if key >= 0 else "snapshot ?"), key)
        lab = self._out_form.labelForField(c)
        if lab is not None:
            lab.setText(label)
        want = self._snap_key(select) if select is not None else prefer
        i = c.findData(want) if want is not None else -1
        c.setCurrentIndex(i if i >= 0 else c.count() - 1)
        c.blockSignals(False)

    def _rebuild_prefixes(self, select=None):
        """The third menu: the outputs of the chosen snapshot by prefix, shown only when there are
        several; its current entry is the output on screen. Without 'select': the prefix the user chose
        last, else the production phase-space one (ps_output, then any ps prefix), else the first."""
        grp, key = self.sim_combo.currentData(), self.snap_combo.currentData()
        if grp in (GROUP_CUSTOM, GROUP_OPENED):
            outs = [o for _, o in self.outputs if self._group(o) == grp and str(o.root) == key]
        else:
            outs = [o for _, o in self.outputs if self._group(o) == grp and self._snap_key(o) == key]
        c = self.prefix_combo
        c.blockSignals(True)
        c.clear()
        for o in sorted(outs, key=lambda o: o.prefix):
            c.addItem(o.prefix, str(o.root))
            c.setItemData(c.count() - 1, str(o.root) + ".*", Qt.ToolTipRole)
        i = c.findData(str(select.root)) if select is not None else -1
        if i < 0:
            i = self._preferred_prefix([c.itemText(k) for k in range(c.count())])
        c.setCurrentIndex(i)
        c.blockSignals(False)
        self._out_form.setRowVisible(c, c.count() > 1)

    def _preferred_prefix(self, names: list) -> int:
        """Which of a snapshot's prefixes to show unasked: the one the user chose last, else the
        production phase-space one (ps_output, then any ps prefix), else the first."""
        for want in (self._pref_prefix, G.PS_PREFIX):
            if want in names:
                return names.index(want)
        ps = [k for k, nm in enumerate(names) if nm.startswith("ps")]
        return ps[0] if ps else 0

    def _select_output(self, root: str) -> bool:
        """Set the three menus to the output '<root>.*' and show it; False when it is not listed."""
        o = self._by_root.get(str(root))
        if o is None:
            return False
        i = self.sim_combo.findData(self._group(o))
        if i < 0:
            return False
        self.sim_combo.blockSignals(True)
        self.sim_combo.setCurrentIndex(i)
        self.sim_combo.blockSignals(False)
        self._rebuild_snaps(select=o)
        self._rebuild_prefixes(select=o)
        self._pref_prefix = o.prefix           # an output picked by name (Runs, the demo, a file): its prefix sticks
        self._show_current()
        return True

    def _current_output(self):
        root = self.prefix_combo.currentData()
        return self._by_root.get(root) if root else None

    def _show_current(self):
        o = self._current_output()
        if o is not None:
            self._show_output(o)

    def _sim_changed(self, *_):
        self._rebuild_snaps(prefer=self.snap_combo.currentData())   # the same snapshot when the simulation has it
        self._rebuild_prefixes()
        self._show_current()

    def _snap_changed(self, *_):
        self._rebuild_prefixes()
        self._show_current()

    def _prefix_changed(self, *_):
        if self.prefix_combo.currentIndex() >= 0:
            self._pref_prefix = self.prefix_combo.currentText()     # the user's choice sticks
        self._show_current()

    def _open_other(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open an output: choose one of its grid files",
                                           self._data_root(), "Grid files (*.a_den *.den *.a_vel *.vel);;All files (*)")
        if f:
            self.open_file(f)

    def open_file(self, f) -> bool:
        """Open any output by one of its grid files: an output already listed is shown where it is;
        another is listed under 'Opened files' (once) and shown."""
        root = Path(f).parent / Path(f).stem          # 'snap_z0.5.a_den' -> 'snap_z0.5' (no grid suffix has a dot)
        if str(root) in self._by_root:
            return self._select_output(str(root))
        o = G.OutputSet(root, box=G.custom_box(G.sidecar(root) or {}), title=f"opened · {root}")
        if not o:
            self.status.emit(f"no readable grids next to {f}")
            return False
        o.group = GROUP_OPENED
        self.outputs.append((o.title, o))
        self._by_root[str(o.root)] = o
        self._rebuild_sims()
        self._grid_sync_rows()                  # the composer's menus list it too (as after a refresh)
        return self._select_output(str(o.root))

    @staticmethod
    def _same_box(a, b) -> bool:
        """Two outputs in the same box: both in Mpc with the same corners (the axes both have), or both of unknown
        box on grids of the same size (their positions are cells)."""
        if a is None or b is None or (a.box is None) != (b.box is None):
            return False
        if a.box is None:
            return a.n == b.n
        d = min(a.dim, b.dim)
        return bool(np.allclose(a.lo[:d], b.lo[:d], rtol=1e-6, atol=1e-6)
                    and np.allclose(a.length[:d], b.length[:d], rtol=1e-6, atol=1e-6))

    def _keep_for(self, prev, o) -> dict:
        """The kept slice (self._keep) for showing output 'o' after 'prev'. It is kept for another output
        or redshift of the same simulation (the boxes are comoving: the same place is the same comoving Mpc at every
        redshift), and for another custom run or opened file in the same box; another simulation, or a box of
        another size, starts afresh (the middle, the whole box). On the way out of a 3D output it takes that
        output's slice when something set the index unseen, and the zoom window on screen."""
        m = self._keep
        if m is not None and prev is not None and prev.dim == 3 and self._same(m.get("ref"), prev):
            axis, index = self.axis.currentIndex(), self.index.value()
            if m["frac"] is None or m["axis"] != axis or kept_index(m["frac"], prev.n) != index:
                m.update(axis=axis, frac=(index + 0.5) / prev.n)
        if (m is not None and prev is not None and not self._same(prev, o) and self.is_zoomed()
                and self._same(self._zoom["output"], prev)):
            z = self._zoom
            m["zoom"] = {"axis": int(z["axis"]), "rest": list(z["rest"]), "a0": tuple(z["a0"]), "a1": tuple(z["a1"]),
                         "res": int(z["res"]), "exact": bool(z.get("exact")), "dim": prev.dim,
                         "unit": "Mpc" if prev.box is not None else "cells"}
        if m is None or m["group"] != self._group(o) or not (o.sim or self._same_box(m["ref"], o)):
            self._drop_kept_zoom()
            m = {"group": self._group(o), "axis": None, "frac": None, "zoom": None}
        m["ref"] = o
        self._keep = m
        return m

    def _drop_kept_zoom(self):
        """No zoom window carried to the next output any more (and no line saying one is kept)."""
        if self._keep is not None and self._keep.get("zoom"):
            self._keep["zoom"] = None
            if self.zoom_state.text().startswith(KEPT_ZOOM_TEXT):
                self.zoom_state.setText("")

    def _remember_slice(self):
        """The user chose the slice (the index, the slider, the arrow keys, the axis, a void's centre, a restored
        state): its place in the box is what another output or redshift shows; a zoom window carried
        from another output goes with the old slice."""
        o, m = self.current, self._keep
        if o is None or o.dim == 2 or m is None or not self._same(m.get("ref"), o):
            return
        self._drop_kept_zoom()
        m.update(axis=self.axis.currentIndex(), frac=(self.index.value() + 0.5) / o.n)

    def _show_output(self, o):
        """Show an output: its field list, info line, slice controls, and the first map. The same
        output again (after a refresh: a new object for the same files) keeps the slice. Another output or
        redshift of the same simulation keeps it too: the same axis, the same place in the box at the
        nearest cell of this grid, and the zoom window where it can be shown (_carry_zoom); another simulation, or
        a custom run or opened file in a box of another size, starts at the middle with the whole box (_keep_for).
        A 2D output shows its one plane and leaves the kept slice to the next 3D output."""
        prev = self.current
        same_grid = self._same(prev, o) and prev.n == o.n and prev.dim == o.dim
        slice_now = self.index.value()
        m = self._keep if same_grid and self._keep is not None else self._keep_for(prev, o)
        if prev is not None and not self._same(prev, o) and not self._grid_shown:
            G.drop_smoothed3d(keep=o)       # the output left: its cubes smoothed in 3D go (the grid keeps its own)
        self.current = o
        keep = self.field.currentData()
        self.field.blockSignals(True)
        self.field.clear()
        for name in o.fields():
            self.field.addItem(G.FIELD_LABELS[name], name)
            if name in G.FIELD_HELP:
                self.field.setItemData(self.field.count() - 1, G.FIELD_HELP[name], Qt.ToolTipRole)
        self.field.blockSignals(False)
        j = self.field.findData(keep)
        self.field.setCurrentIndex(j if j >= 0 else 0)
        if not self._cache_user_set:
            near = o.dir if G.sidecar(o.root) is not None else self._data_root()
            self.cache_edit.setText(str(default_cache_dir(str(near))))
        box = "box unknown" if o.box is None else f"box {o.length[0]:.1f} Mpc"
        grid = f"{o.n}² grid (2D)" if o.dim == 2 else f"{o.n}³ grid"
        self.info.setText(f"{grid} · {box}" + (f" · z = {o.redshift:.2f}" if o.redshift is not None else ""))
        self.info.setToolTip(str(o.root) + ".*")
        flat = o.dim == 2               # a 2D grid is its one plane: no axis, no slice to choose
        kept = not flat and not same_grid and m["frac"] is not None     # another output where the slice is kept
        self.axis.blockSignals(True)
        if flat:
            self.axis.setCurrentIndex(2)
        elif kept:
            self.axis.setCurrentIndex(m["axis"])    # (a 2D output on the way set it to z)
        self.axis.setEnabled(not flat)
        self.axis.blockSignals(False)
        index = 0 if flat else slice_now if same_grid else kept_index(m["frac"], o.n) if kept else o.n // 2
        for w in (self.index, self.slider):
            w.blockSignals(True)
            w.setRange(0, 0 if flat else o.n - 1)
            w.setValue(index)
            w.setEnabled(not flat)
            w.blockSignals(False)
        if not flat and m["frac"] is None:          # the first 3D output of a simulation: the middle, remembered
            m.update(axis=self.axis.currentIndex(), frac=(index + 0.5) / o.n)
        self._pts_fill()                    # BEFORE the field's render: it reads the menu (this output's planes)
        self._voids_sync()                  # BEFORE the render too: the outlines are drawn with the map
        self._3d_sync()                     # a 2D output leaves the 3D view
        self._field_changed()
        self._refresh_server_button()
        if not self._same(prev, o):
            self._carry_zoom()              # AFTER the render: it dropped the other output's zoom

    def _field_changed(self, *_):
        o = self.current
        name = self.field.currentData()
        if o is None or name is None:
            return
        if self.fixed.isChecked() and self._fixed_for not in (None, name):    # a range belongs to its field
            self.fixed.blockSignals(True)
            self.fixed.setChecked(False)
            self.fixed.blockSignals(False)
            self.status.emit(f"the fixed range was set for {G.FIELD_LABELS.get(self._fixed_for, self._fixed_for)}: off "
                             "for this field (tick 'fixed' to use it here)")
        ff = o.files[name]
        self.component.blockSignals(True)
        self.component.clear()
        self.component.addItems(G.components(name, ff.ncomp, o.dim))
        self.component.setEnabled(ff.ncomp > 1)
        self.component.blockSignals(False)
        vec = name in G.VECTOR_FIELDS and ff.ncomp == o.dim
        self.vectors.blockSignals(True)
        self.vectors.setProperty("can", vec)
        self.vectors.setEnabled(vec)
        self.vectors.setChecked(vec and name == "velocity")     # the velocity map opens as a vector field
        self.vectors.blockSignals(False)
        for w, on in ((self.log, name in G.LOG_DEFAULT),):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)
        cmap = "RdBu_r" if name in G.DIVERGING else "tab10" if name in ("tweb", "vweb") else \
            "magma" if name in ("density", "dispersion") else "viridis"
        self.cmap.blockSignals(True)
        self.cmap.setCurrentText(cmap)
        self.cmap.blockSignals(False)
        self._compare_fill()
        self._smooth3d_sync()
        self.render()

    def _axis_changed(self, *_):
        self._remember_slice()
        self.render()

    def _index_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self._remember_slice()
        self.render()

    def _index_slider(self, v):
        self.index.blockSignals(True)
        self.index.setValue(v)
        self.index.blockSignals(False)
        self._remember_slice()
        self.render()

    def step_slice(self, delta: int):
        """The arrow keys: the next or previous slice along the current axis (stops at the box's
        edge)."""
        if self.current is not None:
            self.index.setValue(max(0, min(self.index.maximum(), self.index.value() + delta)))

    def step_field(self, delta: int):
        """Up / down: the previous or next field in the list (stops at either end)."""
        n = self.field.count()
        if n:
            self.field.setCurrentIndex(max(0, min(n - 1, self.field.currentIndex() + delta)))

    # ---------------------------------------------------------------- drawing
    def request(self) -> dict | None:
        o, name = self.current, self.field.currentData()
        if o is None or name is None:
            return None
        req = {"output": o, "field": name, "axis": self.axis.currentIndex(), "index": self.index.value(),
               "component": self.component.currentText(), "cmap": self.cmap.currentText(),
               "log": self.log.isChecked(), "clip": self.clip.value(), "smooth": self.smooth.value(),
               "smooth3d": self.smooth3d.isChecked() and self.smooth3d.isEnabled(),
               "symmetric": name in G.DIVERGING and not self.log.isChecked(),
               "arrows": self.vectors.isEnabled() and self.vectors.isChecked(),
               "ps_gradient": self.ps_gradient.currentData() or G.PS_GRADIENT_DEFAULT}
        if self.pts_combo.currentData():
            req.update(pts=self.pts_combo.currentData(), arrows=False)
            if name in ("shear", "vorticity"):  # the hi-res slice holds their magnitudes (>= 0): no component, no
                req.update(component="magnitude", symmetric=False,     # scale centred on 0
                           cmap="viridis" if req["cmap"] == "RdBu_r" else req["cmap"])
        fixed = self._fixed_range(req["log"])
        self._sync_fixed(req["log"])
        if fixed is not None:
            req["vrange"] = fixed
        other = self._compare_output()
        if other is not None:                   # the difference map: its own scale, so Save image and the bar agree
            req.pop("vrange", None)
            req.update(compare=other, compare_mode=self.compare_mode.currentData(), log=False, symmetric=True,
                       cmap="RdBu_r", arrows=False)
        req["contours"] = self._contours_sync(req)
        req["contour_detail"] = self.contour_detail.value()
        if req["contours"]:                     # the worker strokes the lines at the size the canvas shows them
            req["contour_layer"] = self.view.canvas.layer_size()
        return req

    def _contour_detail_moved(self, _value=None):
        """The level slider moved: at once for a click, a key or a programmatic move; while it is dragged at most once
        per CONTOUR_DRAG_MS (each redraw then takes the position of that moment; the workers serve only the latest
        request, as for a scroll), so a drag across many positions does not ask for each of them."""
        if self.contour_detail.isSliderDown():
            if not self._contour_drag.isActive():
                self._contour_drag.start()
            return
        self._contour_drag.stop()
        self.render()

    def _contour_detail_released(self):
        """The drag ended: its last position drawn now if a redraw was still waiting."""
        if self._contour_drag.isActive():
            self._contour_drag.stop()
            self.render()

    def _contours_sync(self, req: dict) -> bool:
        """The Contours tick for the map of request 'req': greyed out, its tooltip saying why, on a class map or a
        colour map of discrete colours (contours.why_not); the user's tick kept either way. The level slider only
        while the map gets lines. True when the map gets its contour lines."""
        why = CT.why_not(req["field"], req["cmap"], req.get("compare") is not None)
        self.contours.setEnabled(not why)
        self.contours.setToolTip(why or CONTOURS_TIP)
        on = not why and self.contours.isChecked()
        for w in (self.contour_detail, *self._contour_ends):
            w.setEnabled(on)
        if not on:
            self.contour_count.setText("")
        return on

    def render(self, *_):
        req = self.request()
        if req is None or self._shut:
            return
        if self._3d_shown:                      # the 3D view follows (the map behind it too: shown again on leaving)
            self._3d_follow(req)
        if self._grid_shown:
            prev = self._req or {}
            same_view = (bool(prev) and self._same(prev.get("output"), req["output"])
                         and prev.get("field") == req["field"] and prev.get("component") == req["component"])
            if same_view:                       # the slice (arrow keys, slider), colours, range, smoothing: the grid follows
                self._req = req
                self._set_position(req)
                self._grid_update_merge()
                self._grid_refresh_soon()
                return
            self._leave_grid()                  # another field or output: the map
            req = self.request()                # ... with the comparison, which the grid left out
        self._grid_update_merge()               # the slice axis decides which box edges the panels share
        o = req["output"]
        self._set_position(req)
        if self._zoom is not None:
            z = self._zoom
            # the stamp LAST (it stats the files): a re-run under the same name gives a new OutputSet of the same
            # root, and the old zoom must not stay beside the new grids' slices
            if (self._same(z["output"], o) and z["axis"] == req["axis"] and z["index"] == req["index"]
                    and ("stamp" not in z or z["stamp"] == self._file_stamp(o))):
                if self._zoom_fields is not None:       # a colour/field change: no new evaluation
                    self._req = req
                    self._render_zoom()
                return
            self._clear_zoom()                          # another slice or output, or rewritten grids: back to its map
            req = self.request()                        # ... with the comparison, which the zoom left out
        if req.get("smooth3d") and req["smooth"] > 0 and not req.get("pts") \
                and not G.smoothed3d_kept(o, req["field"], req["component"], req["smooth"]):
            self.status.emit(f"smoothing the whole {G.FIELD_LABELS.get(req['field'], req['field'])} cube in 3D at "
                             f"{req['smooth']:g} cells (once per field and smoothing: a few seconds)")
        if self._busy:
            self._pending = req             # only the latest request is served
            return
        self._busy = True
        self._req = req
        self._threads()
        self._to_render.emit(req)

    def _fixed_range(self, log: bool):
        """The fixed colour range in the bar's units (log10 on a log map), or None: off, unreadable, empty, or
        a log map with a bound <= 0."""
        if not self.fixed.isChecked():
            return None
        try:
            lo, hi = float(self.vmin_edit.text()), float(self.vmax_edit.text())
        except ValueError:
            return None
        if not hi > lo or (log and lo <= 0):
            return None
        return (float(np.log10(lo)), float(np.log10(hi))) if log else (lo, hi)

    def _fixed_toggled(self, on: bool):
        if on:
            self._fixed_for = self.field.currentData()
        if on and not (self.vmin_edit.text().strip() and self.vmax_edit.text().strip()):
            self._take_range()              # nothing typed yet: start from the map on screen
            return
        self.render()

    def _fixed_edited(self):
        if self.fixed.isChecked():
            self._fixed_for = self.field.currentData()
            self.render()

    def _sync_fixed(self, log: bool) -> None:
        """The clip spin is greyed only while a fixed range is IN USE (not for a range that cannot apply -- a
        log map with a bound <= 0 -- nor while the figure grid shows: the grid uses the clip); a fixed range that
        cannot apply says so once."""
        usable = self.fixed.isChecked() and self._fixed_range(log) is not None
        self.clip.setEnabled(not usable or self._grid_shown)
        if self.fixed.isChecked() and not usable:
            if not self._fixed_warned:
                self.status.emit("the fixed range does not apply to this map (a log map needs min > 0): its own "
                                 "percentile range is shown")
                self._fixed_warned = True
        else:
            self._fixed_warned = False

    def _take_range(self):
        """The map's current colour range into the fields (linear values), fixed from now on."""
        lo, hi = self.view.colorbar.vrange
        if self.view.colorbar.log:
            lo, hi = 10.0 ** lo, 10.0 ** hi
        self._fixed_for = self.field.currentData()
        self.vmin_edit.setText(f"{lo:.4g}")
        self.vmax_edit.setText(f"{hi:.4g}")
        if not self.fixed.isChecked():
            self.fixed.setChecked(True)         # _fixed_toggled renders
        else:
            self.render()

    def state(self) -> dict:
        """What the window remembers of Explore across restarts."""
        return {"root": str(self.current.root) if self.current is not None else "",
                "field": self.field.currentData(), "component": self.component.currentText(),
                "axis": self.axis.currentIndex(), "index": self.index.value(),
                "fixed": self.fixed.isChecked(), "vmin": self.vmin_edit.text(), "vmax": self.vmax_edit.text(),
                "snapshots": dict(self._snapshots),
                "voids": self.voids.isChecked(), "void_reff": self.void_reff.isChecked(),
                "void_sigma": self.void_sigma.value(), "ps_gradient": self.ps_gradient.currentData(),
                "void_rmin": self.void_rmin.value(), "void_rmax": self.void_rmax.value(),
                "void_as": self.void_shape(), "smooth3d": self.smooth3d.isChecked(),
                "contours": self.contours.isChecked(),
                "contour_position": self.contour_detail.value(), **self.state_3d()}

    def restore_state(self, d: dict) -> bool:
        """Back to a remembered state when its output is still listed (selected first by refresh_outputs)."""
        if isinstance(d, dict):                 # the hand-chosen snapshots: for every output (an older state's
            # 'resident', the server's partitions in memory, is the binary's choice now: not read)
            self._snapshots.update({k: v for k, v in (d.get("snapshots") or {}).items() if Path(v).is_file()})
            j = self.ps_gradient.findData(d.get("ps_gradient"))       # for every output too
            if j >= 0 and j != self.ps_gradient.currentIndex():
                self.ps_gradient.setCurrentIndex(j)
            # how the voids are drawn: 'void_as'; a state from before the ellipsoids were the default kept
            # 'void_shape' for the 3D view alone, where spheres were the default everyone saved: only its 'ellipsoid'
            # is read
            old = "ellipsoid" if d.get("void_shape") == "ellipsoid" else None
            shape = d.get("void_as") if "void_as" in d else old
            j = self.void3d_shape.findData(shape)
            if j >= 0:
                self.void3d_shape.setCurrentIndex(j)
            if isinstance(d.get("smooth3d"), bool):                     # the map's smoothing in 3D: for every output
                self.smooth3d.setChecked(d["smooth3d"])
            self.restore_state_3d(d)                                    # its style, perspective, opacity, detail
            if isinstance(d.get("contours"), bool):                     # the contour lines: for every output too
                self.contours.setChecked(d["contours"])
            # ... and how many: the slider's position (clamped to it; read as restore_state_3d reads its numbers), or
            # in a state from before the slider was continuous its old stop ('contour_detail', rungs of the round
            # steps) at the nearest position
            try:
                if "contour_position" in d:
                    self.contour_detail.setValue(int(d["contour_position"]))
                else:
                    self.contour_detail.setValue(CT.from_stop(int(d["contour_detail"])))
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
            self._refresh_server_button()       # refresh_outputs ran first: it judged the server with no snapshots
        if not d or self.current is None or str(self.current.root) != d.get("root"):
            return False
        j = self.field.findData(d.get("field"))
        if j >= 0:
            self.field.setCurrentIndex(j)
        k = self.component.findText(d.get("component", ""))
        if k >= 0:
            self.component.setCurrentIndex(k)
        if self.axis.isEnabled() and d.get("axis") in (0, 1, 2):
            self.axis.setCurrentIndex(int(d["axis"]))
        if self.index.isEnabled():
            self.index.setValue(int(d.get("index", self.index.value())))
        self.vmin_edit.setText(str(d.get("vmin", "")))
        self.vmax_edit.setText(str(d.get("vmax", "")))
        self.fixed.setChecked(bool(d.get("fixed")))
        # the void outlines and their options; the smoothing and the
        # R_eff tick first, so ticking 'voids' asks for the catalogue once, at the remembered smoothing
        for w, v in ((self.void_sigma, float(d.get("void_sigma", 0.0) or 0.0)),
                     (self.void_rmin, float(d.get("void_rmin", 0.0) or 0.0)),
                     (self.void_rmax, float(d.get("void_rmax", 0.0) or 0.0)),
                     (self.void_reff, bool(d.get("void_reff"))), (self.voids, bool(d.get("voids")))):
            w.blockSignals(w is not self.voids)
            if isinstance(w, QDoubleSpinBox):
                w.setValue(v)
            else:
                w.setChecked(v)
            w.blockSignals(False)
            if w is self.void_sigma:
                self._void_sigma_restored()     # committed (raised to two cells) before the tick asks for it
        if self.voids.isChecked():              # ticked already (no 'toggled'): the restored smoothing's catalogue
            self._voids_sync()
            self._voids_redraw()
        return True

    # ---------------------------------------------------------------- memory guard
    def heavy_ok(self, title: str, what: str, need_gb: float | None = None) -> bool:
        """Before a memory-heavy action (a query server, an exact zoom, a void catalogue build): True at once when
        no DTFE run is going (rs.dtfe_processes: never blocks then), or when one is but what this needs fits in what
        is free less HEAVY_HEADROOM_GB (said in the status line); else ask, naming the runs with their resident
        memory, what this needs and what is free now. The run's auto-tuner budgets only from the memory free
        at its own start: a heavy action started later can make the run die of memory pressure."""
        runs = rs.dtfe_processes()["runs"]
        if not runs:
            return True
        free = G.available_gb()
        who = "; ".join(f"{name} pid {pid}{rs._gb_text(gb)}" for pid, name, gb in runs)
        if need_gb is not None and free is not None and need_gb <= free - HEAVY_HEADROOM_GB:
            self.status.emit(f"a DTFE run is going ({who}); this needs about {need_gb:.1f} GB of the {free:.0f} GB "
                             f"free: started")
            return True
        need = f"about {need_gb:.0f} GB" if need_gb is not None else "an unknown amount of memory"
        text = (f"A DTFE run is going ({who}).\n\n{what} needs {need}; "
                + (f"{free:.0f} GB is free now." if free is not None else "the free memory could not be read.")
                + "\n\nStarting it beside the run can push the machine into swap or get the run killed. "
                  "Start it anyway?")
        if self.confirm_heavy is not None:
            return bool(self.confirm_heavy(title, text))
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def memory_status(self) -> str:
        """One line for a status bar: free memory and the DTFE runs and servers alive (their resident GB)."""
        free = G.available_gb()
        d = rs.dtfe_processes()
        parts = [f"{free:.0f} GB free" if free is not None else "free memory unknown"]
        parts += [f"{name} pid {pid}{rs._gb_text(gb)}" for pid, name, gb in d["runs"]]
        parts += [f"server pid {pid}{rs._gb_text(gb)}" for pid, _name, gb in d["servers"]]
        return " · ".join(parts)

    def _canvas_axes(self) -> dict | None:
        """The axes of what the map shows (the slice, or the zoomed region): Mpc, or cells without a box."""
        g = self._geometry()
        if g is None:
            return None
        _axis, _index, rest, lo, hi, _n0, _n1 = g
        return {"x": (float(lo[0]), float(hi[0]), AXES[rest[0]]), "y": (float(lo[1]), float(hi[1]), AXES[rest[1]]),
                "unit": "Mpc" if self._req["output"].box is not None else "cells"}

    @staticmethod
    def comparable(a, b, field: str) -> bool:
        """B can be subtracted from A cell by cell: another output on the same cells (dim, grid, box) holding
        the field with the same components."""
        return (a is not None and b is not None and str(a.root) != str(b.root) and a.dim == b.dim and a.n == b.n
                and (a.box is None) == (b.box is None) and np.allclose(a.lo, b.lo) and np.allclose(a.length, b.length)
                and field in b.files and field in a.files and b.files[field].ncomp == a.files[field].ncomp)

    def _compare_fill(self):
        """The 'Compare with' menu: off, then every comparable output -- this snapshot's other runs first."""
        o, name = self.current, self.field.currentData()
        keep = self.compare.currentData()
        self.compare.blockSignals(True)
        self.compare.clear()
        self.compare.addItem("(off)", None)
        if o is not None and name is not None:
            cands = [b for _, b in self.outputs if self.comparable(o, b, name)]
            cands.sort(key=lambda b: (b.dir != o.dir, b.sim, -1 if b.snap is None else b.snap, str(b.root)))
            for b in cands:
                where = b.sim or b.dir.name
                z = f" · z = {b.redshift:.2f}" if b.redshift is not None else ""
                self.compare.addItem(f"{where}{z} · {G.type_label(b)}", str(b.root))
                self.compare.setItemData(self.compare.count() - 1, str(b.root) + ".*", Qt.ToolTipRole)
        j = self.compare.findData(keep) if keep is not None else 0
        self.compare.setCurrentIndex(max(j, 0))
        self.compare.blockSignals(False)
        signed = name in G.DIVERGING or name in ("velocity", "tweb", "vweb")
        self.compare_mode.model().item(1).setEnabled(not signed)      # log A/B: positive fields only
        if signed:
            self.compare_mode.blockSignals(True)
            self.compare_mode.setCurrentIndex(0)
            self.compare_mode.blockSignals(False)
        self._compare_sync()

    def _compare_output(self):
        """B of the difference map, or None (off, or a zoom / the figure grid on screen)."""
        root = self.compare.currentData()
        if root is None or self.is_zoomed() or self._grid_shown or self.pts_combo.currentData():
            return None
        return self._by_root.get(root)

    def _compare_sync(self):
        """The colour controls do not apply to a difference map (its own diverging scale); the menu itself
        not to a zoom or the figure grid."""
        on = self.compare.currentData() is not None
        free = not self.is_zoomed() and not self._grid_shown and not self.pts_combo.currentData()
        self.compare.setEnabled(free and self.compare.count() > 1)
        self.compare_mode.setEnabled(free and on)
        for w in (self.cmap, self.log, self.fixed, self.take_range_btn):
            w.setEnabled(not (on and free))
        self.vectors.setEnabled(self.vectors.property("can") is not False and not (on and free))

    def _pts_fill(self):
        """The hi-res slice menu for the output shown: off + every matching plane; hidden without one."""
        keep = self.pts_combo.currentData()
        planes = G.find_pts_planes(self.current) if self.current is not None else []
        self.pts_combo.blockSignals(True)
        self.pts_combo.clear()
        self.pts_combo.addItem("off: the grid slice", None)
        for label, side in planes:
            self.pts_combo.addItem(label, str(side))
        j = self.pts_combo.findData(keep) if keep else 0
        self.pts_combo.setCurrentIndex(max(j, 0))
        self.pts_combo.blockSignals(False)
        self._out_form_map.setRowVisible(self.pts_combo, bool(planes))
        self._pts_sync()

    def _pts_sync(self):
        """With the hi-res slice shown the slice axis and position do not apply (the plane is the sidecar's)."""
        on = bool(self.pts_combo.currentData())
        flat = self.current is not None and self.current.dim == 2
        for w in (self.axis, self.index, self.slider):
            w.setEnabled(not on and not flat)

    def _pts_changed(self, *_):
        if self.pts_combo.currentData() and self.is_zoomed():
            self._clear_zoom()
        self._pts_sync()
        self._compare_sync()
        self._smooth3d_sync()
        self.render()

    def _compare_changed(self, *_):
        self._compare_sync()
        self.render()

    def _ps_gradient_changed(self, *_):
        """The velocity-derivative family chosen: the map, its comparison and the figure grid again."""
        self._smooth3d_sync()
        self.render()

    def _smooth3d_sync(self):
        """The Smoothing row's '3D' tick for the map shown: greyed out, its tooltip saying why, where the map is not a
        slice of a stored 3D grid or the cube's smoothing would not fit in the free memory (grids.smooth3d_why); the
        user's tick kept either way. Asked when the output, the field, the hi-res slice or the family changes (the
        free memory is read then, not per layer)."""
        o, name = self.current, self.field.currentData()
        why = G.smooth3d_why(o, name, self.ps_gradient.currentData() or G.PS_GRADIENT_DEFAULT,
                             hires=bool(self.pts_combo.currentData())) if o is not None and name else "no map shown"
        self.smooth3d.setEnabled(not why)
        if why:
            self.smooth3d.setToolTip(f"Smoothing in 3D: {why}.")
        else:
            tip_auto(self.smooth3d, SMOOTH3D_TIP)

    def _smooth3d_toggled(self, *_):
        """The map's smoothing in 3D on or off: the map (its contours), the grid's panels and the 3D view's planes;
        the line under the voids says whether the map now shows their field."""
        self._smooth3d_sync()
        self.render()

    def _gradient_label(self, req: dict) -> str:
        """The field's name on the map for request 'req' (grids.field_label): a phase-space output's velocity
        derivatives say their family -- the mean field, the stored stream average, or (a hi-res slice: a point
        evaluation) the stream average; with a comparison both outputs' families, A's and B's when they differ. The
        note under the choice says when a phase-space map is not what the choice asks, and why."""
        o, name, grad = req["output"], req["field"], req.get("ps_gradient", "stored")
        ps = G.estimator_label(o) == "PS-DTFE"
        fam = o.gradient_family(name, grad)
        note = ""
        if fam and ps and req.get("pts"):
            fam, note = "stream", "a hi-res slice is a point evaluation: the stream average"
        elif fam == "stream" and grad == "mean-field":
            note = f"this output shows the stored stream average: {o.mean_field_reason()}"
        b = req.get("compare")
        fam_b = b.gradient_family(name, grad) if b is not None else fam
        if fam_b == "stream" and grad == "mean-field" and b is not None and not note:
            note = f"B shows the stored stream average: {b.mean_field_reason()}"
        self.ps_gradient_note.setText(note)
        self._out_form_map.setRowVisible(self.ps_gradient_note, bool(note))
        qualify = bool(fam) and (ps or (b is not None and G.estimator_label(b) == "PS-DTFE"))
        if fam and fam_b and fam != fam_b:
            return (f"{G.FIELD_LABELS.get(name, name)} (A {G.GRADIENT_QUALIFIER[fam]}, "
                    f"B {G.GRADIENT_QUALIFIER[fam_b]})")
        return G.field_label(o, name, fam, qualify)

    def _set_position(self, req: dict):
        self.position.setText(self._position_text(req))

    @staticmethod
    def _position_text(req: dict, units=None) -> str:
        """Where the slice of request 'req' lies, 'z = 32.50 Mpc': in Mpc (cells without a box), or with 'units'
        ((axis, position) -> (value, unit name); figsave.position) in a saved figure's unit."""
        o, axis = req["output"], req["axis"]
        if o.dim == 2:
            return "2D: the whole plane"
        pos = o.lo[axis] + (req["index"] + 0.5) * o.length[axis] / o.n
        v, unit = units(axis, pos) if units is not None else (pos, "Mpc" if o.box is not None else "cells")
        return f"{AXES[axis]} = {v:.2f} {unit}"

    def _rendered(self, res: dict):
        self._busy = False
        if self._pending is not None:
            req, self._pending = self._pending, None
            self.render_request(req)
        if self.is_zoomed():                    # asked for before a zoom landed (a field changed meanwhile): the zoom
            return                              # stays; drawn, the slice would take the zoom's geometry
        req = res["request"]
        self._plane = res["plane"]
        self._req = req
        self._arrows = res.get("arrows")
        if self._grid_shown or self._3d_shown:  # a slice in flight when the grid or 3D was shown: keep the picture
            return
        self._planes = res.get("planes")
        self._pts_side = res.get("pts_side")
        self.view.canvas.setStyleSheet("")
        self.view.canvas.marker = None
        self.view.canvas.labels = None
        self.view.canvas.arrows = self._arrows
        self._set_contours(res.get("contours"))
        self._set_void_canvas(self._void_overlay())
        self._void_note()                       # which field the voids belong to: the map may show it now
        self.view.canvas.axes = self._canvas_axes()      # BEFORE the image: its first drawing has the margins
        self.view.canvas.set_rgb(res["rgb"])
        label = self._gradient_label(req)       # the velocity derivatives' family said
        comp = G.component_suffix(req["component"])
        o = req["output"]
        # the title as a function of the positions' unit (a figure saved in Mpc/h or cells says so)
        if req.get("compare") is not None:
            mode = dict((m, t) for t, m in COMPARE_MODES)[req["compare_mode"]]
            self.view.colorbar.set(req["cmap"], res["vrange"], f"{mode}: {label}{comp}", levels=self._contour_marks())
            head = f"<b>{mode}: {label}{comp}</b> — A = {o.title}, B = {req['compare'].title} — "

            def where(units=None):
                return self._position_text(req, units)
        else:
            self.view.colorbar.set(req["cmap"], res["vrange"], G.colourbar_label(label, req), req["log"],
                                   levels=self._contour_marks())
            head = f"<b>{label}{comp}</b> — {o.title} — "
            sd, fct = self._pts_side, res.get("pts_factor") or 1

            def where(units=None):
                if not (req.get("pts") and sd is not None):
                    return self._position_text(req, units)
                c, unit = float(sd.get("center", 0)), "Mpc"            # the plane's own place, in Mpc
                if units is not None and sd.get("axis") in AXES:
                    c, unit = units(AXES.index(sd["axis"]), c)
                return (f"hi-res slice: {sd.get('axis')} = {c:.2f} {unit}, {sd['nu']}×{sd['nv']} points"
                        + (f" (shown {fct}×{fct}-averaged)" if fct > 1 else "") + ", point evaluation")
        self._title_make = lambda units=None: head + where(units)
        self.view.title.setText(self._title_make())
        self.changed.emit()

    def render_request(self, req):
        self._busy = True
        self._to_render.emit(req)

    def _set_contours(self, lines: dict | None):
        """The contour lines of the map on screen (map_contours' dict, or None): kept for Save image, stroked by the
        canvas (with the layer the worker stroked, when it did), said in the colour bar's tooltip (their marks go to
        the bar with its next set()) and counted beside the level slider."""
        self._contours = lines
        self.view.canvas.contours = lines["paths"] if lines else None
        self.view.canvas.contour_layer = lines.get("layer") if lines else None
        self.view.colorbar.setToolTip(CT.describe(lines))
        self._contour_count(lines)

    def _contour_count(self, lines, what: str = "line"):
        """The level slider's count: how many contour levels the picture shown draws -- 'lines' a map_lines dict (or
        None: none), or the figure grid's count per panel (a range when they differ); nothing while the map gets
        none."""
        if not self.contour_detail.isEnabled():
            self.contour_count.setText("")
            return
        counts = lines if isinstance(lines, list) else [0 if not lines else int(np.size(lines["levels"]))]
        lo, hi = (min(counts), max(counts)) if counts else (0, 0)
        n = f"{lo}" if lo == hi else f"{lo}–{hi}"
        self.contour_count.setText(f"{n} {what}{'' if hi == 1 else 's'}")

    def _contour_marks(self):
        """The colour bar's marks of the map's contour levels, or None."""
        return self._contours["marks"] if self._contours else None

    def _render_failed(self, msg):
        self._busy = False
        if self._pending is not None:           # a newer slice is wanted: its result decides, not this one's
            req, self._pending = self._pending, None
            self.render_request(req)
            return
        if self.is_zoomed():                    # a slice asked for before the zoom landed: the zoom stays
            return
        self._plane = self._arrows = None
        self.view.canvas.arrows = None
        self._set_void_canvas(None)
        self._set_contours(None)
        self.view.canvas.set_rgb(None)
        self.view.canvas.setText(msg if msg.startswith("Not smoothed in 3D") else f"Could not read this slice: {msg}")
        self.changed.emit()

    def can_save(self) -> bool:
        return self._plane is not None or self._grid_shown or self._3d_shown

    def is_zoomed(self) -> bool:
        return self._zoom is not None and self._zoom_fields is not None

    def _geometry(self):
        """What the map shows: (axis, index, rest axes, lo and hi of the rest axes, n0, n1) -- the
        whole slice, the zoomed region at its own resolution, or a hi-res slice's plane."""
        req, plane = self._req, self._plane
        if req is None or plane is None:
            return None
        if req.get("pts") and self._pts_side is not None:
            sd = self._pts_side
            rest = [AXES.index(sd["u_axis"]), AXES.index(sd["v_axis"])]
            return (AXES.index(sd["axis"]), 0, rest, np.array([sd["u0"], sd["v0"]], float),
                    np.array([sd["u1"], sd["v1"]], float), plane.shape[0], plane.shape[1])
        o, axis = req["output"], req["axis"]
        rest = [a for a in range(3) if a != axis]
        if self.is_zoomed():
            z = self._zoom
            return (axis, req["index"], rest, np.array([z["a0"][0], z["a1"][0]]),
                    np.array([z["a0"][1], z["a1"][1]]), z["res"], z["res"])
        return axis, req["index"], rest, o.lo[rest], o.lo[rest] + o.length[rest], o.n, o.n

    def _point(self, r: int, c: int):
        """Image pixel -> (3D position, value): the centre of that pixel's cell or zoom point."""
        g = self._geometry()
        if g is None:
            return None
        axis, index, rest, lo, hi, n0, n1 = g
        plane = self._plane
        row, col = c, n1 - 1 - r              # image = plane.T[::-1]; plane rows = first remaining axis
        if not (0 <= row < n0 and 0 <= col < n1):
            return None
        o = self._req["output"]
        x = np.zeros(3)
        x[axis] = o.lo[axis] + (index + 0.5) * o.length[axis] / o.n
        if self._req.get("pts") and self._pts_side is not None:
            x[axis] = float(self._pts_side.get("center", x[axis]))      # the hi-res plane's own position
        x[rest[0]] = lo[0] + (row + 0.5) / n0 * (hi[0] - lo[0])
        x[rest[1]] = lo[1] + (col + 0.5) / n1 * (hi[1] - lo[1])
        return x, float(plane[row, col]), (row, col)

    def _hover(self, r, c):
        if r < 0 or self._grid_shown:
            self.view.readout.setText(" ")
            return
        pt = self._point(r, c)
        if pt is None:
            return
        x, value, (row, col) = pt
        o = self._req["output"]
        unit = "Mpc" if o.box is not None else "cells"
        where = f"zoom point ({row}, {col})" if self.is_zoomed() else \
            f"hi-res point ({row}, {col})" if self._req.get("pts") else \
            f"cell {tuple(int(v) for v in o.plane_point(self._req['axis'], self._req['index'], row, col))}"
        xyz = "   ".join(f"{a} = {v:.2f}" for a, v in zip(AXES[:o.dim], x))
        both = ""
        if self._planes is not None and not self.is_zoomed():
            a, b = self._planes
            both = f"   (A = {float(a[row, col]):.5g}, B = {float(b[row, col]):.5g})"
        k = self._void_pick(r, c)
        void = f"      {G.void_info(self._void_result(), k, short=True)}" if k is not None else ""
        self.view.readout.setText(f"{xyz} {unit}      {where}      value = {value:.5g}{both}{void}")

    def shutdown(self):
        """Stop the server and the threads (the window is closing). Idempotent."""
        self._shut = True
        self._3d_cancel()                               # a 3D volume being read: the slice thread comes back soon
        if self._server_log is not None:
            try:
                self._server_log.unlink(missing_ok=True)
            except OSError:
                pass
        self._zoom_tick.stop()
        self._voids_stop()
        self._server.kill()                            # a build OR a running request: the thread must come back
        self._exact_worker.cancel()                    # a running exact zoom would hold the thread
        if self._slice_thread is None:
            self._server.stop()
            return
        self._to_stop.emit()
        for th in (self._slice_thread, self._server_thread, self._exact_thread):
            th.quit()
            for _ in range(6):                         # up to 30 s: a thread still inside a request is never dropped
                if th.wait(5000):
                    break
        self._slice_thread = self._server_thread = self._exact_thread = None
