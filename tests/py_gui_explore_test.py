#!/usr/bin/env python3
"""The Explore tab's void features, offscreen, on a synthetic snapshot (no simulation data, no binary):

  1. the void inspector: hovering an outline names the void (R_v, R_eff, delta, flags), clicking it fills the
     inspector (every R_v definition the catalogue has, R_eff, central delta, flags, centre), 'Go to its centre'
     moves the slice there, 'R_eff too' adds dashed outlines;
  2. the outlines carried into the saved PNG (matplotlib circles) with a CSV of the voids shown beside it
     (tools/void_outlines.write_csv), and into the figure grid (each panel's own catalogue, its tiles clipped);
  3. the void smoothing: 'production' or a sigma in Mpc, passed to the helper as --smooth-mpc, its own cache key and
     cache namespace;
  4. the memory guard: no question while no DTFE run is going; with one, a question (free memory said), and a
     'no' starts nothing (server, exact zoom, catalogue); a need that fits in the free memory less 4 GB starts
     without a question even then.
  5. the caustic (fold) map of a periodic PS run is the exact fold flag (dtfelib io.caustic_is_fold),
     not the stored per-sample grid of older runs; the caustic-class map stays the stored mask; an
     exact zoom's fold and class maps read its class grid (the 0/1 '.caustic' would give no fold at all).
  6. the Save options (figsave.py): the default save is the reference one pixel for pixel (and byte for byte);
     full resolution = the data's shape and the screen's pixels, with or without the outlines; no axes, title, colour
     bar, units (Mpc, Mpc/h from the header's h, cells), overlays, margins none, width, dpi, PDF/SVG text kept as
     text, TIFF, transparency, font size; the same for the figure grid (separate and merged panels); the presets,
     the dialog (no '--' in its texts) and the options remembered between sessions in QSettings.
  7. the slice kept across outputs and redshifts: another output or redshift of the same simulation
     shows the same axis and place in the box (the nearest cell of its grid: 64^3 and 32^3 outputs, the user's own
     place remembered, not re-read off the coarser grid), and the zoom window (from memory, through the query server,
     or not shown with the reason in the zoom and status lines); another simulation, a custom run or opened file in
     a box of another size start at the middle with the whole box, as the first open does; a 2D output on the way
     leaves the slice and the window to the next 3D output.
  8. the Save options' details: the title's slice position in the axes' unit (map and grid); TIFF
     LZW-compressed, pixel-identical; 'match the data' below 72 dpi shrinks the figure (the map keeps the data's
     pixel count), and with no margins writes the map's own pixels; Mpc/h where h is not known falls back to Mpc,
     said in the status line and the dialog, the choice kept (the grid: every panel); the fold map reads only its
     planes while a job caps the cube cache.
  9. a phase-space output's velocity derivatives as the mean field grad<v> (the default, as the figure
     scripts') or the stored stream average <grad v>: the mean-field planes equal dtfelib's FieldSet.load(name,
     mean_field=True) cut there (the six names, every axis, through slice()'s components and units); standard DTFE
     unchanged; 2D, non-periodic and window outputs keep the stored grid and say why; while a job caps the cube cache
     the velocity is read in slabs, never cached whole, the same numbers; a rewritten velocity grid is not served
     from the plane cache; a face-gradient output (its '.a_velDivStreams', PS-DTFE's default with the exact deposit
     and the volume weighting) shows its stored grids, the mean field, with either choice, and its exact zoom takes
     '--face-gradient' (an output without the marker '--per-tet-gradient'). In the window: the map, its title,
     colour bar and saved figure say the family, the choice
     reaches Compare with (A's and B's families) and the figure grid (a stored PS panel never shares a colour bar
     with a mean-field one), it is remembered, a zoom says the stream average (and its zoom line is text).
 10. the 3D view (view3d.py, explore_3d.py): the camera's rotation is orthonormal and right-handed, a sphere of radius
     R is a circle of R x scale at any turn and an ellipsoid's outline is its projection's, the planes' quads are
     painted in an exact back-to-front order (checked ray by ray), the principal sections' arcs and the textures'
     affine maps land where the 3D points project, an ellipsoid's cut on a plane agrees with the inside test (spheres':
     the map's circles), picking takes the smallest outline, the close-up window and the periodic cell window. In
     perspective (the checks labelled 12): a zoom in moves the eye along the cursor's line of sight, into the box
     and on through the field, and never out of it again; a cursor off the box keeps the plain rule; a zoom out
     backs away; a line or a picture's rectangle that passes the eye is cut to its part in front of it
     (segment_on_screen, screen_polygon); from inside, the void cuts on a plane that passes the eye and the point
     picked under the cursor. In the window: Show in 3D shows the map's slice and the middle of
     the other axes as stored (the same values), the distinct voids as spheres of R_v x scale; the slider moves its
     plane; hover names a void and reads a plane's value; a click inspects the void (the inspector, the menu), a
     double-click closes up on it (the planes through its centre, cut to the window periodically, the neighbours at
     their nearest image); Whole box restores the planes; the fitted ellipsoids reach the launcher ((a b c)^(1/3) =
     R_eff) and 'fitted ellipsoids' draws each as its own, a close-up then sized by R_eff and holding the whole
     ellipsoid, its neighbours dim, the menu by R_eff; the figure grid and the 3D
     view replace each other; the PNG is written; the map comes back. The Sizes range keeps the same voids on the
     map, the grid's panels and the 3D view (by R_v; the ellipsoids by R_eff), says how many, keeps a close-up's void
     and is remembered.
 11. the 3D view's volume (volume3d.py, explore_3d.py): the grid averaged over blocks as equal as whole cells allow
     (each the block's mean, the integral kept), a window across the faces read periodically, a vector's component
     as slice() takes it, plain reads (a cube the map cached used, none added), a grid written again read again,
     an averaging cancelled; the levels within half a step; the transfer function transparent below the range,
     monotone, premultiplied, corrected for the path through a slice, |value| for a diverging field; the slices'
     images placed where their 3D corners project, front first; in perspective each pixel is drawn from the stack
     across the axis its ray crosses the most slices of (slice_plan, part_ids: a part for each axis and sense),
     and with the eye inside the cube QPainter's picture is, pixel by pixel, the march along that pixel's own
     ray. QPainter's 16-bit compositing equals the reference
     'under' compositing; the front half is what is seen from either side; a dense ball is opaque, a ray missing it
     transparent. In the window: the volume is the default (the planes one choice away), of the stored values, a
     density from rho/rho_bar = 1 in the map's colours with the bar on that range; the ray through a void is
     clearer than the densest column, as the reference says; the range sliders keep their order, Default range
     restores; a colour map recolours without a read; the voids are drawn over it; a close-up reads the void's cube
     at the grid's cells, Whole box comes from memory; Save image; the planes back; leaving frees the slices. The
     GPU's picture (volume_gl): offscreen there is no OpenGL and QPainter draws (the state line, the turning policy's
     contract); in a process of its own on the default platform, where an OpenGL context can be made, the GPU's
     picture equals QPainter's within 1.5/255 at every pixel not within 1/32 px of a slice's projected edge (Qt
     rasterises polygons to 1/64 px: a pixel there may take one slice more or less), smooth and spiky fields, both
     projections, three turns, from outside and with the eye in the window; skipping empty blocks changes no
     pixel; the window's own picture against volume_picture(), fitted and zoomed in (else SKIP: no display, no
     OpenGL).
 12. the Explore and 3D-view tooltips take at most two lines (the map, the voids, the 3D view in both styles, the
     Save options dialog); a long text is its first sentence, the whole text the control's What's This.

The pure parts (grids.void_at, void_circles_mpc, void_info, void_outlines.write_csv, view3d, volume3d) run without Qt; the rest needs
PySide6 and reports SKIP (exit 0) without it. Usage: ~/.venvs/dtfe-gui/bin/python tests/py_gui_explore_test.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "gui"))
sys.path.insert(0, str(ROOT / "python" / "tools"))
if "--gl-child" not in sys.argv[1:]:            # the GPU parity child keeps the default platform (OpenGL)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["DTFE_GUI_NO_NOTIFY"] = "1"
TMP = Path(tempfile.mkdtemp(prefix="gui_explore_"))
os.environ["DTFE_FIGURES_ROOT"] = str(TMP / "figures")     # BEFORE runspec is imported: no test figure in the real root

import grids as G  # noqa: E402
import void_outlines as VO  # noqa: E402

PASS, FAIL = [], []
SIM = "SYNEXP-1-Dark"
N, BOX = 64, 64.0


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


def make_snapdir(data: Path, snap: int = 50, z: float = 0.0, seed: int = 7, sim: str = SIM, box: float = BOX) -> Path:
    """data/SIM/snapdir_NNN: a combined header (box, redshift) and an N^3 PS-DTFE set ('ps_output': density,
    velocity, streams) of six broad Gaussian dips on 2% noise -- voids at the production smoothing (10 cells)."""
    import h5py
    d = data / sim / f"snapdir_{snap:03d}"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"combined_{snap:03d}.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = box * 1000.0, float(z), 0.6774, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, N ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
        h["NumPart_Total"] = np.array([0, N ** 3, 0, 0, 0, 0])
    rho = 1.0 + 0.02 * np.random.default_rng(seed).normal(size=(N, N, N))
    ii = np.indices((N, N, N))
    for c, w, depth in (((16, 16, 32), 6.0, 0.9), ((46, 20, 30), 7.0, 0.85), ((20, 48, 34), 5.0, 0.9),
                        ((50, 50, 28), 8.0, 0.8), ((8, 36, 54), 6.0, 0.85), ((36, 6, 10), 5.0, 0.8)):
        r2 = sum((((ii[k] - c[k] + N // 2) % N) - N // 2) ** 2 for k in range(3))
        rho -= depth * np.exp(-0.5 * r2 / w ** 2)
    rho.astype(np.float32).tofile(str(d / "ps_output.a_den"))
    np.zeros((N, N, N, 3), np.float32).tofile(str(d / "ps_output.a_vel"))
    np.ones((N, N, N), np.float32).tofile(str(d / "ps_output.a_streams"))
    return d


# ---------------------------------------------------------------------- pure parts (no Qt)
def pure():
    print("pure parts")
    circles = [(10.0, 10.0, 5.0, 5.0, 3), (11.0, 10.0, 2.0, 2.0, 7), (40.0, 40.0, 4.0, 2.0, 9)]
    check("void_at: the smallest outline containing the point (a small void inside a big one stays reachable), "
          "None outside, ellipses (non-square pixels) respected",
          G.void_at(circles, 11.5, 10.0) == 7 and G.void_at(circles, 6.0, 10.0) == 3 and G.void_at(circles, 30, 30) is None
          and G.void_at(circles, 43.5, 40.0) == 9 and G.void_at(circles, 40.0, 43.0) is None and G.void_at(None, 1, 1) is None)
    cen = np.array([[10.0, 20.0, 30.0], [62.5, 5.0, 31.0]])
    rad = np.array([4.0, 3.0])
    lo, hi = (0.0, 0.0), (64.0, 64.0)
    px = G.void_circles(cen, rad, 64.0, 2, 30.5, [0, 1], lo, hi, 128, 128)
    mp = G.void_circles_mpc(cen, rad, 64.0, 2, 30.5, [0, 1], lo, hi)
    same = len(px) == len(mp) and all(abs(u - x / 2.0) < 1e-9 and abs(v - (64.0 - y / 2.0)) < 1e-9 and abs(r - rx / 2.0) < 1e-9
                                      and k == kk for (x, y, rx, _ry, k), (u, v, r, kk) in zip(px, mp))
    check("void_circles_mpc = void_circles in the data frame (Mpc, y up), periodic images included",
          same and any(abs(u - 62.5) < 1e-9 for u, *_ in mp) and any(u < 0 for u, *_ in mp), f"{px} {mp}")
    res = {"sim": "S", "snap": 5, "method": "ps", "prefix": "", "box_mpc": 64.0, "n_grid": 64, "sigma_mpc": 2.0,
           "centres_mpc": [[1.5, 2.5, 3.5], [10.0, 11.0, 12.0]], "r_v_mpc": [4.0, None], "r_eff_mpc": [6.5, None],
           "delta": [-0.9, -0.5], "resolved": [True, False], "distinct": [True, False],
           "r_v_defs": {"crossing": [4.0, None], "delta08": [3.0, None], "ridge": [5.5, None]}}
    long, short = G.void_info(res, 0), G.void_info(res, 0, short=True)
    check("void_info: R_v of every definition the catalogue has, R_eff, central delta, flags, centre",
          all(t in long for t in ("4.00 Mpc", "3.00 Mpc", "5.50 Mpc", "6.50 Mpc", "-0.900", "resolved", "distinct",
                                  "1.50, 2.50, 3.50", "ridge")) and "R_v 4.00 Mpc" in short and "--" not in long,
          long)
    check("void_info: an undefined radius says 'none', the flags say 'not'",
          "R_v none" in G.void_info(res, 1, short=True) and "not resolved" in G.void_info(res, 1), G.void_info(res, 1))
    path = VO.write_csv(res, TMP / "pure.csv", rows=[1, 0])
    lines = Path(path).read_text().splitlines()
    head = lines[1].split(",")
    check("write_csv: a '#' provenance line, the header with every R_v definition, the rows asked for in order, "
          "empty fields where undefined",
          lines[0].startswith("# S snapshot 5 ps") and head == ["void", "x_mpc", "y_mpc", "z_mpc", "r_v_mpc",
                                                               "r_v_delta08_mpc", "r_v_ridge_mpc", "r_eff_mpc", "delta",
                                                               "resolved", "distinct"]
          and lines[2].startswith("1,10,11,12,,,,,-0.5,0,0") and lines[3].startswith("0,1.5,2.5,3.5,4,3,5.5,6.5,-0.9,1,1"),
          "\n".join(lines))
    # a cached catalogue whose radii voids() would re-measure is not 'cached' to the launcher, and the memory gate
    # applies ('r_v_cells' present is not enough, whatever settings measured it): pipeline.catalog_r_v_ok
    from dtfelib import pipeline
    r3 = {k: np.ones(2) for k in ("r_v_cells", "r_v_delta08_cells", "r_v_ridge_cells")}
    old_set = pipeline.r_v_settings() + np.eye(5)[1]           # another R_V_MAX_SIGMA
    for name, arrays in (("today", dict(r3, r_v_settings=pipeline.r_v_settings())), ("older", dict(r3, r_v_settings=old_set)),
                         ("unrecorded", r3), ("no_rv", {})):
        np.savez(TMP / f"rv_{name}.npz", coords=np.zeros((2, 3)), **arrays)

    class Cached:                                               # a SnapshotProducts whose cache passed the grid stamp
        def __init__(self, path):
            self.path = path

        def _cached(self, _name):
            return self.path
    st = {n: VO.r_v_stale(TMP / f"rv_{n}.npz") for n in ("today", "older", "unrecorded", "no_rv")}
    check("r_v_stale / needs_build: the pipeline's own R_v test; older or unrecorded settings are stale and gated",
          st["today"] is None and all(st[n] for n in ("older", "unrecorded", "no_rv"))
          and "older settings" in st["older"] and not any("--" in str(v) for v in st.values())
          and not VO.needs_build(Cached(TMP / "rv_today.npz")) and VO.needs_build(Cached(TMP / "rv_older.npz"))
          and VO.needs_build(Cached(None)), st)
    custom_redshift()
    save_options_pure()


def custom_redshift():
    """A custom snapshot's velocities in Explore get the sqrt(a) of its Gadget header's redshift, as a
    TNG-layout snapshot's do (redshift None would leave u-units, 1/sqrt(a) too large); the Custom tab
    records the redshift in its sidecar, an older sidecar falls back to the input's header, none = no scaling."""
    import json
    import h5py
    import runspec as rs
    d = TMP / "custom_z"
    d.mkdir()
    n = 4
    vel = np.ones((n, n, n, 3), np.float32)
    snaps = {}
    for name, z in (("z1", 1.0), ("noz", None)):
        f = d / f"{name}.hdf5"
        with h5py.File(f, "w") as h:
            hd = h.create_group("Header").attrs
            hd["BoxSize"] = 10.0
            if z is not None:
                hd["Redshift"] = z
        snaps[name] = f
    side = rs.CustomSpec(input_file=str(snaps["z1"]), input_type=105).settings_sidecar()
    check("settings_sidecar records the header's redshift (None without one)",
          side.get("redshift") == 1.0
          and rs.CustomSpec(input_file=str(snaps["noz"]), input_type=105).settings_sidecar().get("redshift") is None,
          str(side.get("redshift")))
    cases = {"new": {"input_file": str(snaps["z1"]), "input_type": 105, "redshift": 1.0},
             "old": {"input_file": str(snaps["z1"]), "input_type": 105},             # a sidecar from before the key
             "noz": {"input_file": str(snaps["noz"]), "input_type": 105, "redshift": None},
             "ascii": {"input_file": str(snaps["z1"]), "input_type": 101}}            # not a Gadget HDF5 input
    for prefix, sc in cases.items():
        np.ones((n, n, n), np.float32).tofile(str(d / f"{prefix}.a_den"))
        vel.tofile(str(d / f"{prefix}.a_vel"))
        (d / f"{prefix}.gui.json").write_text(json.dumps(sc))
    outs = {o.prefix: o for _, o in G.find_outputs(TMP / "no_root", [d])}
    vmax = {k: float(np.nanmax(o.slice("velocity", 2, 1, "x"))) for k, o in outs.items()}
    want = 0.5 ** 0.5
    check("a custom output's redshift: the sidecar's, else its HDF5 input's header; velocities scaled by sqrt(a) "
          "like a TNG snapshot's (z = 1: 0.707); no Redshift, or no HDF5 input: unscaled",
          outs["new"].redshift == 1.0 and outs["old"].redshift == 1.0 and outs["noz"].redshift is None
          and outs["ascii"].redshift is None and abs(vmax["new"] - want) < 1e-6 and abs(vmax["old"] - want) < 1e-6
          and vmax["noz"] == 1.0 and vmax["ascii"] == 1.0,
          f"{ {k: o.redshift for k, o in outs.items()} } {vmax}")


# ---------------------------------------------------------------------- the Save options
def save_options_pure():
    """figsave's Qt-free parts: the options checked, the presets, the units, the formats, the dpi matching, the box's h."""
    import json
    import h5py
    import figsave as FS
    print("Save options (pure)")
    o = FS.options({"dpi": 5000, "format": "gif", "frame": "no", "width": -3, "font_scale": 1.5, "unit": "mpc_h",
                    "colour": 1, "title_text": "T", "title_for": "auto"})
    check("options(): out-of-range values clipped, unknown formats and wrong types back to the defaults, unknown keys "
          "dropped, the typed title kept for the session",
          o["dpi"] == FS.DPI_MAX and o["format"] == "png" and o["frame"] is True and o["width"] == 0.0
          and o["font_scale"] == 1.5 and o["unit"] == "mpc_h" and "colour" not in o and o["title_text"] == "T", str(o))
    check("the defaults are the reference save's: 200 dpi, a tight crop, PNG, everything on",
          FS.options() == FS.DEFAULTS and FS.save_kwargs(FS.options(), 200) == {"dpi": 200, "bbox_inches": "tight"}
          and FS.rc(FS.options(), "png") == {})
    check("persistent()/dumps(): the typed title is never remembered between sessions; loads() of garbage = defaults",
          "title_text" not in json.loads(FS.dumps(o)) and FS.loads("{not json") == FS.options()
          and FS.loads(None) == FS.options() and FS.loads(json.dumps([1, 2])) == FS.options())
    pub, bare, slide = FS.preset("publication"), FS.preset("bare"), FS.preset("slide")
    check("presets: Publication = everything on, vector PDF or 300 dpi; Bare image = the map alone at full resolution, "
          "no frame, axes, title, bar or overlays; Slide = large fonts; each recognised by which_preset",
          pub["format"] == "pdf" and pub["dpi"] == 300 and all(pub[k] for k in ("title", "frame", "axis_names", "colorbar"))
          and bare["kind"] == "full" and bare["margins"] == "none"
          and not any(bare[k] for k in ("title", "frame", "ticks", "tick_labels", "axis_names", "colorbar", "arrows",
                                        "voids"))
          and slide["font_scale"] > 1.5 and [FS.which_preset(p) for p in (FS.options(), pub, bare, slide)]
          == ["standard", "publication", "bare", "slide"] and FS.which_preset(dict(pub, dpi=301)) is None)
    check("a preset keeps the typed title", FS.preset("bare", {"title_text": "X"})["title_text"] == "X")
    check("title_text(): the typed title only for the map it was typed for",
          FS.title_text({"title_text": "Mine", "title_for": "map A"}, "map A") == "Mine"
          and FS.title_text({"title_text": "Mine", "title_for": "map A"}, "map B") == "map B"
          and FS.title_text({}, "map A") == "map A")
    check("format_of / with_format: the suffix decides (.tif and .tiff are TIFF), a name without one gets the format's",
          [FS.format_of(p) for p in ("a.png", "a.PDF", "a.svg", "a.tif", "a.tiff", "a")] == ["png", "pdf", "svg", "tiff",
                                                                                           "tiff", "png"]
          and FS.with_format("/x/map", "svg").name == "map.svg" and FS.with_format("/x/m.pdf", "png").name == "m.pdf"
          and FS.with_format("/x/m.v2", "tiff").name == "m.v2.tif")

    class Out:
        def __init__(self, box, n=64):
            self.box, self.n = box, n
            self.lo = np.zeros(3) if box is None else np.asarray(box[0], float)
            self.length = np.full(3, float(n)) if box is None else np.asarray(box[1], float) - self.lo
    boxed = Out(((10.0, 20.0, 0.0), (42.0, 52.0, 32.0)))
    um_h, note_h = FS.unit_map(boxed, [0, 1], "mpc_h", 0.7)
    um_c, _ = FS.unit_map(boxed, [0, 1], "cells")
    um_u, note_u = FS.unit_map(boxed, [0, 1], "mpc_h", None)
    um_n, _ = FS.unit_map(Out(None), [0, 1], "mpc")
    check("unit_map: Mpc/h = Mpc x h; cells from the box's low corner (n / length); Mpc/h without a known h is Mpc, "
          "said; no box = cells; Mpc is the identity (the old numbers bit for bit)",
          um_h.name == "Mpc/h" and np.allclose(um_h.ext([10, 42, 20, 52]), [7.0, 29.4, 14.0, 36.4]) and not note_h
          and um_c.name == "cells" and np.allclose(um_c.ext([10, 42, 20, 52]), [0, 64, 0, 64])
          and um_u.name == "Mpc" and "h" in note_u and um_n.name == "cells"
          and FS.unit_map(boxed, [0, 1], "mpc")[0].identity and FS.unit_map(boxed, [0, 1], "mpc")[0].x(1.25) == 1.25)
    # the slice position of a saved title in the axes' unit
    pos_h, pos_c, pos_u, pos_m = (FS.position(boxed, 2, 16.25, "mpc_h", 0.7), FS.position(boxed, 2, 16.25, "cells"),
                                  FS.position(boxed, 2, 16.25, "mpc_h", None), FS.position(boxed, 2, 16.25, "mpc"))
    check("position: along any axis as the axes convert it (Mpc/h = Mpc x h; cells from the box's low corner, the "
          "centre of cell 32 of 64 = 32.5; Mpc/h without h = Mpc; Mpc untouched)",
          pos_h[1] == "Mpc/h" and abs(pos_h[0] - 11.375) < 1e-12 and pos_c == (32.5, "cells")
          and pos_u == (16.25, "Mpc") and pos_m == (16.25, "Mpc"), str((pos_h, pos_c, pos_u, pos_m)))
    check("title_text with the title in the axes' unit: drawn unless a title was typed for this map ('auto' stays its "
          "key, so a unit change keeps a typed title)",
          FS.title_text({}, "z = 1 Mpc", "z = 0.7 Mpc/h") == "z = 0.7 Mpc/h"
          and FS.title_text({"title_text": "Mine", "title_for": "z = 1 Mpc"}, "z = 1 Mpc", "z = 0.7 Mpc/h") == "Mine")
    tw = FS.save_kwargs(FS.options(), 200, "tiff")
    check("save_kwargs: a TIFF is LZW-compressed through Pillow (pil_kwargs); a PNG's arguments unchanged",
          tw.get("pil_kwargs") == {"compression": "tiff_lzw"} and "pil_kwargs" not in FS.save_kwargs(FS.options(), 200)
          and "pil_kwargs" not in FS.save_kwargs(FS.options(), 200, "pdf"), str(tw))
    check("writes_pixels: full resolution, or no margins with 'match the data' and no width (the map's own pixels); "
          "not a borderless figure at a dpi or a width",
          FS.writes_pixels(FS.preset("bare")) and FS.writes_pixels({"margins": "none", "match_data": True})
          and not FS.writes_pixels({"margins": "none"}) and not FS.writes_pixels({"match_data": True})
          and not FS.writes_pixels({"margins": "none", "match_data": True, "width": 640, "width_unit": "px"})
          and "own resolution" in FS.describe({"margins": "none", "match_data": True}))
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 6))
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.5])
    small = Figure(figsize=(7, 6))
    sax = small.add_axes([0.1, 0.1, 0.5, 0.5])
    d_small = FS.match_dpi(small, [(sax, 64)])
    held = sax.get_position().width * small.get_size_inches()[0] * d_small      # the axes' pixels at that dpi
    check("match_dpi: the dpi at which an axes 3.5 in wide holds 700 data pixels is 200, 1200 at most; a 64 pixel "
          "plane (18 dpi wanted) gets 72 dpi and a figure shrunk until its axes hold 64 pixels (a dpi "
          "stopped at 72 alone gives 252 pixels)",
          abs(FS.match_dpi(fig, [(ax, 700)]) - 200.0) < 1e-6 and FS.match_dpi(fig, [(ax, 10 ** 6)]) == FS.DPI_MAX
          and d_small == FS.DPI_MIN and abs(held - 64) < 0.05 and small.get_size_inches()[0] < 2.0
          and tuple(fig.get_size_inches()) == (7.0, 6.0), f"{d_small} {held} {small.get_size_inches()}")
    # the box's h: a custom run's HDF5 input with the h-free mark gives it, one without the mark does not
    d = TMP / "hubble"
    d.mkdir()
    for name, mark in (("hfree", True), ("raw", False)):
        with h5py.File(d / f"{name}.hdf5", "w") as h:
            a = h.create_group("Header").attrs
            a["BoxSize"], a["HubbleParam"] = 10000.0, 0.7
            if mark:
                a["HFreeUnits"] = 0.7
        np.ones((4, 4, 4), np.float32).tofile(str(d / f"{name}.a_den"))
        (d / f"{name}.gui.json").write_text(json.dumps({"input_file": str(d / f"{name}.hdf5"), "input_type": 105,
                                                        "mpc_unit": 1000.0}))
    outs = {o.prefix: o for _, o in G.find_outputs(TMP / "no_root", [d])}
    check("grids.hubble: an h-free custom input gives its h (Mpc/h offered); a raw one (its box may be Mpc/h already) "
          "and an output without a box give None",
          G.hubble(outs["hfree"]) == 0.7 and G.hubble(outs["raw"]) is None
          and G.hubble(G.OutputSet(d / "raw", box=None)) is None, str({k: G.hubble(o) for k, o in outs.items()}))
    texts = ([t for _k, t in FS.PRESET_NAMES] + list(FS.PRESET_TIPS.values()) + [t for _k, t in FS.MARGINS]
             + [t for _k, t in FS.UNITS] + [t for t, _s in FS.FORMATS.values()] + [t for _f, t in FS.file_filters()]
             + [FS.describe(p) for p in (FS.options(), pub, bare, slide, dict(pub, match_data=True))])
    check("the Save options' texts carry no '--'", not [t for t in texts if "--" in t],
          str([t for t in texts if "--" in t]))


def _old_save_png(x, f):
    """ExploreControls.save_png without the Save options, verbatim but for 'self': the reference
    the default save must equal pixel for pixel."""
    import explore as E
    req = x._req
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    o = req["output"]
    data = x._plane.T
    if req["log"]:
        data = np.log10(np.where(data > 0, data, np.nan))
    _, _, rest, glo, ghi, _, _ = x._geometry()
    ext = [glo[0], ghi[0], glo[1], ghi[1]]
    fig, ax = plt.subplots(figsize=(7, 6))
    lo, hi = x.view.colorbar.vrange
    im = ax.imshow(data, origin="lower", extent=ext, cmap=req["cmap"], vmin=lo, vmax=hi)
    unit = "Mpc" if o.box is not None else "cells"
    ax.set_xlabel(f"{E.AXES[rest[0]]} [{unit}]")
    ax.set_ylabel(f"{E.AXES[rest[1]]} [{unit}]")
    ax.set_title(x.view.title.text().replace("<b>", "").replace("</b>", ""), fontsize=8)
    fig.colorbar(im, ax=ax, label=x.view.colorbar.label)
    arrows = getattr(x, "_arrows", None)
    if arrows is not None and arrows["vmax"] > 0:
        h = o.length / o.n
        X, Y = np.meshgrid(o.lo[rest[0]] + arrows["c0"] * h[rest[0]],
                           o.lo[rest[1]] + arrows["c1"] * h[rest[1]], indexing="ij")
        ax.quiver(X, Y, arrows["u"], arrows["v"], color="white", edgecolor="black", linewidth=0.3,
                  angles="xy", scale_units="xy", pivot="middle",
                  scale=arrows["vmax"] / (0.9 * arrows["spacing"] * h[rest[0]]), width=0.003)
    res = x._void_result()
    if res is not None and x.view.canvas.circles is not None:
        axis = x._geometry()[0]
        pos = x._slice_position()
        sel = np.nonzero(res["_shown"])[0]
        circ = [(u, v, r, int(sel[k])) for u, v, r, k in G.void_circles_mpc(
            res["_centres"][sel], res["_r_v"][sel], float(res["box_mpc"]), axis, pos, rest, glo, ghi)]
        dash = [] if not x.void_reff.isChecked() else [(u, v, r, int(sel[k])) for u, v, r, k in G.void_circles_mpc(
            res["_centres"][sel], res["_r_eff"][sel], float(res["box_mpc"]), axis, pos, rest, glo, ghi)]
        G.draw_void_circles_mpl(ax, circ, dash)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _png(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def _n_circles(fig):
    from matplotlib.patches import Circle
    return sum(isinstance(p, Circle) for ax in fig.axes for p in ax.patches)


def saving_map(qa, x, view):
    """The Save options on the single map (the void outlines and their R_eff shown, as part 2 leaves it): the default
    unchanged, full resolution, no axes, title, colour bar, units, overlays, margins, width, dpi, formats, transparency,
    font size."""
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image
    import figsave as FS
    print("Save options: the map")
    d = TMP / "save map"
    d.mkdir(exist_ok=True)
    n0, n1 = x._plane.shape                          # plane rows = the horizontal axis
    _old_save_png(x, d / "old.png")
    w_new = x.save_png(d / "new.png")
    x.save_png(d / "explicit.png", FS.options())
    old, new = _png(d / "old.png"), _png(d / "new.png")
    check("the default save is the reference one: the same pixels as save_png without the Save options (and the same "
          "bytes); the explicit defaults too; the voids' CSV beside it",
          old.shape == new.shape and np.array_equal(old, new) and np.array_equal(new, _png(d / "explicit.png"))
          and (d / "old.png").read_bytes() == (d / "new.png").read_bytes() and len(w_new) == 2,
          f"{old.shape} {new.shape} {w_new}")
    # full resolution
    bare = FS.preset("bare")
    w_full = x.save_png(d / "full.png", bare)
    full = _png(d / "full.png")
    check("Bare image: the map alone, one data pixel per image pixel: the image has the data's shape and the screen's "
          "very pixels; no CSV (no outlines drawn)", full.shape == (n1, n0, 3) and np.array_equal(full, view.canvas._buf)
          and w_full == [str(d / "full.png")], f"{full.shape} vs {(n1, n0)}")
    x.save_png(d / "full_voids.png", dict(bare, voids=True))
    fv = _png(d / "full_voids.png")
    changed = (fv[..., :3] != full).any(axis=-1).mean() if fv.shape[:2] == full.shape[:2] else -1
    check("... with the void outlines over it: the same size, the outlines drawn, the rest of the pixels untouched",
          fv.shape[:2] == full.shape[:2] and 0.0 < changed < 0.5 and (fv.shape[2] == 3 or fv[..., 3].min() == 255),
          f"{fv.shape} {changed}")
    x.save_png(d / "full.tif", bare)
    x.save_png(d / "full.pdf", bare)
    x.save_png(d / "full.svg", bare)
    pdf, svg = (d / "full.pdf").read_bytes(), (d / "full.svg").read_text(errors="replace")
    check("... as TIFF (the same pixels), PDF and SVG (the pixels embedded unresampled)",
          np.array_equal(_png(d / "full.tif")[..., :3], full) and pdf.startswith(b"%PDF") and f"/Width {n0}".encode() in pdf
          and "<image" in svg, f"{len(pdf)} {svg[:80]}")
    # the decorated figure, inspected before saving
    fig, _dpi = x.map_figure(FS.options())
    fig.canvas.draw()
    ax = fig.axes[0]
    base_labels = ax.get_xticklabels() + ax.get_yticklabels()
    base_circles, base_title = _n_circles(fig), ax.get_title()
    plt.close(fig)
    fig, _dpi = x.map_figure(FS.options({"frame": False, "ticks": False, "tick_labels": False, "axis_names": False}))
    fig.canvas.draw()
    ax = fig.axes[0]
    labels = [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible() and t.get_text()]
    check("no axes: no tick labels, no axis names, no frame, ticks of zero length (the default has them all); the "
          "title and the colour bar stay", base_labels and not labels and ax.get_xlabel() == ax.get_ylabel() == ""
          and not any(sp.get_visible() for sp in ax.spines.values())
          and all(t.tick1line.get_markersize() == 0 for t in ax.xaxis.get_major_ticks())
          and ax.get_title() == base_title and len(fig.axes) == 2, f"{labels} {ax.get_xlabel()}")
    plt.close(fig)
    auto = x._map_title()
    titles = []
    for o in ({"title": False}, {"title_text": "A void at z = 0", "title_for": auto},
              {"title_text": "stale", "title_for": "another map"}):
        fig, _ = x.map_figure(FS.options(o))
        titles.append(fig.axes[0].get_title())
        plt.close(fig)
    check("title off: none; a typed title for this map: that title; one typed for another map: the map's own",
          titles == ["", "A void at z = 0", auto] and auto == base_title, str(titles))
    fig, _ = x.map_figure(FS.options({"colorbar": False}))
    n_off = len(fig.axes)
    plt.close(fig)
    fig, _ = x.map_figure(FS.options({"colorbar_label": False}))
    bar_label = fig.axes[1].get_ylabel()
    plt.close(fig)
    check("colour bar off: no bar axes; its label off: the bar without a label", n_off == 1 and bar_label == "",
          f"{n_off} {bar_label!r}")
    _, _, rest, glo, ghi, _, _ = x._geometry()
    xn = "xyz"[rest[0]]
    got = {}
    for unit, extra in (("mpc", {}), ("mpc_h", {}), ("cells", {}), ("mpc", {"units": False})):
        fig, _ = x.map_figure(FS.options(dict(unit=unit, **extra)))
        got[unit + str(extra)] = (fig.axes[0].get_xlabel(), fig.axes[0].get_xlim())
        plt.close(fig)
    h = G.hubble(x.current)
    check("units: Mpc; Mpc/h (the snapshot header's h = 0.6774) scales the axes; cells count the grid's "
          "cells; the unit can be left out of the name",
          got["mpc{}"][0] == f"{xn} [Mpc]" and got["mpc_h{}"][0] == f"{xn} [Mpc/h]" and abs(h - 0.6774) < 1e-12
          and np.allclose(got["mpc_h{}"][1], (glo[0] * h, ghi[0] * h)) and got["cells{}"][0] == f"{xn} [cells]"
          and np.allclose(got["cells{}"][1], (0, n0)) and got["mpc{'units': False}"][0] == xn, str(got))
    # the title's slice position in the axes' unit, not Mpc whatever the axes say
    req = x._req
    an, pos = "xyz"[req["axis"]], float(x._slice_position())
    tt = {}
    for unit in ("mpc", "mpc_h", "cells"):
        fig, _ = x.map_figure(FS.options({"unit": unit}))
        tt[unit] = fig.axes[0].get_title()
        plt.close(fig)
    fig, _ = x.map_figure(FS.options({"unit": "cells", "title_text": "Typed", "title_for": auto}))
    typed = fig.axes[0].get_title()
    plt.close(fig)
    check("the title's slice position in the axes' unit: Mpc as on screen, Mpc/h (times h), cells (the cell's "
          "centre); a title typed for this map stays under another unit",
          tt["mpc"] == auto and f"{an} = {pos * h:.2f} Mpc/h" in tt["mpc_h"] and f"{an} = {pos:.2f} Mpc" in auto
          and f"{an} = {pos:.2f} Mpc" not in tt["mpc_h"] and f"{an} = {req['index'] + 0.5:.2f} cells" in tt["cells"]
          and typed == "Typed", str(tt))
    fig, _ = x.map_figure(FS.options({"voids": False}))
    check("void outlines off: no circles (the default draws them)", base_circles > 0 and _n_circles(fig) == 0,
          f"{base_circles} {_n_circles(fig)}")
    plt.close(fig)
    w = x.save_png(d / "novoids.png", FS.options({"voids": False}))
    check("... and no voids' CSV beside a map saved without them", w == [str(d / "novoids.png")], str(w))
    # sizes: margins, width, dpi
    x.save_png(d / "borderless.png", FS.options({"margins": "none", "dpi": 100}))
    x.save_png(d / "borderless640.png", FS.options({"margins": "none", "width": 640, "width_unit": "px"}))
    x.save_png(d / "w5in.png", FS.options({"width": 5.0, "dpi": 100}))
    x.save_png(d / "w900px.png", FS.options({"width": 900, "width_unit": "px"}))
    x.save_png(d / "dpi100.png", FS.options({"dpi": 100}))
    bl, b640 = _png(d / "borderless.png"), _png(d / "borderless640.png")
    w5, w900, d100 = (_png(d / f).shape[1] for f in ("w5in.png", "w900px.png", "dpi100.png"))
    check("margins none: the map fills the file (7 in at 100 dpi = 700 x 700 for a square box; 640 px asked = 640 x "
          "640), no white edge", bl.shape[:2] == (700, 700) and b640.shape[:2] == (640, 640)
          and bl[0, :, :3].min() < 250, f"{bl.shape} {b640.shape}")
    check("width: the saved file as wide as asked, crop included (5 in at 100 dpi = 500 px, 900 px = 900 px, within "
          "2 px); dpi 100 = half the default's width", abs(w5 - 500) <= 2 and abs(w900 - 900) <= 2
          and abs(d100 - new.shape[1] / 2) <= 3, f"{w5} {w900} {d100} {new.shape}")
    fig, dpi = x.map_figure(FS.options({"match_data": True}))
    fig.set_dpi(dpi)
    ext = fig.axes[0].get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    x.save_png(d / "match.png", FS.options({"match_data": True, "voids": False}))
    mw = _png(d / "match.png").shape[1]
    check("match the data: a 64 px map needs less than 72 dpi: 72 dpi and a smaller figure, the map's axes 64 x 64 "
          "pixels as the data (not 389, resampled), the text around it",
          dpi == FS.DPI_MIN and abs(ext.width - n0) < 0.6 and abs(ext.height - n1) < 0.6 and n0 + 40 < mw < 389,
          f"{dpi} {ext.width} x {ext.height} {mw}")
    # no margins with 'match the data': the map's own pixels, unresampled (as full resolution)
    bm = FS.options({"margins": "none", "match_data": True, "voids": False, "arrows": False})
    w_bm = x.save_png(d / "borderless_match.png", bm)
    blm = _png(d / "borderless_match.png")
    check("no margins with 'match the data': the file is the map's own pixels (the data's shape, the screen's very "
          "pixels, as Full resolution)", blm.shape == (n1, n0, 3) and np.array_equal(blm, view.canvas._buf)
          and w_bm == [str(d / "borderless_match.png")], f"{blm.shape}")
    # formats, transparency, fonts
    x.save_png(d / "map.pdf", FS.preset("publication"))
    x.save_png(d / "map.svg", FS.options({"format": "svg"}))
    x.save_png(d / "map.tif", FS.options())
    x.save_png(d / "transparent.png", FS.options({"transparent": True}))
    pdf, svg = (d / "map.pdf").read_bytes(), (d / "map.svg").read_text(errors="replace")
    tr = _png(d / "transparent.png")
    check("PDF and SVG keep the text as text (TrueType embedded, SVG <text>); TIFF = the PNG's pixels; transparent: "
          "see-through around the figure", pdf.startswith(b"%PDF") and b"/FontFile2" in pdf and "<text" in svg
          and np.array_equal(_png(d / "map.tif")[..., :3], new[..., :3]) and tr.shape[2] == 4 and tr[0, 0, 3] == 0,
          f"{len(pdf)} {tr.shape}")
    # every TIFF LZW-compressed: the figure (matplotlib), full resolution
    # (Pillow) and full resolution with the outlines (matplotlib's figimage); the same pixels as uncompressed
    x.save_png(d / "full_voids.tif", dict(bare, voids=True))
    tifs = {}
    for name in ("map.tif", "full.tif", "full_voids.tif"):
        with Image.open(d / name) as im:
            arr = np.asarray(im)
            tifs[name] = im.info.get("compression")
        Image.fromarray(arr).save(d / ("raw_" + name))              # Pillow's default: uncompressed
        tifs[name] = (tifs[name], np.array_equal(_png(d / ("raw_" + name)), arr),
                      (d / name).stat().st_size < (d / ("raw_" + name)).stat().st_size)
    check("TIFF: LZW-compressed (the figure, full resolution, full resolution with outlines), smaller than the "
          "uncompressed file and pixel-identical to it", all(v == ("tiff_lzw", True, True) for v in tifs.values()),
          str(tifs))
    s = FS.preset("slide")
    with matplotlib.rc_context(FS.rc(s)):
        fig, _ = x.map_figure(s)
        size = (fig.axes[0].title.get_fontsize(), fig.axes[0].xaxis.label.get_fontsize())
        wraps = fig.axes[0].title.get_wrap()
    plt.close(fig)
    check("Slide: every text 1.8 times larger (the title 8 -> 14.4 pt, the axis names 10 -> 18 pt); the long title "
          "wraps instead of running over the colour bar's top label",
          abs(size[0] - 14.4) < 1e-9 and abs(size[1] - 18.0) < 1e-9 and wraps, f"{size} {wraps}")
    # a width or dpi far too large is refused BEFORE matplotlib allocates the picture (50 in at
    # 1200 dpi would be 60000 px across, 12 GB): no file, no figure left open, Save image says why
    import resource
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    refused = []
    for o in ({"width": 50, "width_unit": "in", "dpi": 1200}, {"width": 100000, "width_unit": "px"},
              {"width": 60000, "width_unit": "px", "margins": "none"}):
        try:
            x.save_png(d / "too_big.png", FS.options(o))
        except FS.TooLarge as e:
            refused.append(str(e))
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0     # bytes on macOS, kB on Linux: both small
    msgs = []

    def said(m):
        msgs.append(m)
    x.status.connect(said)
    real_ask, kept = x._ask_image_path, x.save_options
    x._ask_image_path = lambda *a, **k: (str(d / "too_big_dialog.png"), "png")
    x.save_options = FS.options({"width": 100000, "width_unit": "px"})
    try:
        x._save_image()
    finally:
        x._ask_image_path, x.save_options = real_ask, kept
        x.status.disconnect(said)
    check("a width far too large: refused before drawing (no file, no open figure, no memory taken), and Save image "
          "says so instead of writing the screen", len(refused) == 3 and not (d / "too_big.png").exists()
          and not (d / "too_big_dialog.png").exists() and not plt.get_fignums() and grew < 5e8
          and msgs and "could not save" in msgs[-1] and "pixels" in msgs[-1]
          and not [t for t in refused + msgs if "--" in t], f"{refused} {msgs[-1:]} {grew}")
    for f in ("old_voids.csv", "new_voids.csv", "explicit_voids.csv"):
        (d / f).unlink(missing_ok=True)


def saving_grid(qa, x, view):
    """The Save options on the figure grid (two redshifts side by side, each with its own voids, as part 2 leaves it):
    the default unchanged, full resolution, colour bars, panel titles, the title, margins none, merged panels."""
    import matplotlib.pyplot as plt  # noqa: F401  (the Agg backend)
    import explore as E
    import figsave as FS
    print("Save options: the figure grid")
    d = TMP / "save grid"
    d.mkdir(exist_ok=True)
    x.save_grid(d / "default.png")
    x.save_grid(d / "explicit.png", FS.options())
    fig0, _ = x.grid_figure()
    check("the grid's default save: the explicit defaults give the same pixels; the figure keeps its two panels and "
          "two colour bars", np.array_equal(_png(d / "default.png"), _png(d / "explicit.png")) and len(fig0.axes) == 4)
    req = x.grid_request()
    items = E.grid_items(req)
    T = max(it["img"].shape[0] for it in items if it is not None)
    tiles, _l, _b = E.grid_tiles(req, items, max_tile=T)
    x.save_grid(d / "full.png", FS.preset("bare"))
    x.save_grid(d / "full_voids.png", dict(FS.preset("bare"), voids=True))
    full, fv = _png(d / "full.png"), _png(d / "full_voids.png")
    check("full resolution: the panels at their own pixels with the screen's gap (1 x 2 of 64: 64 x 134), the "
          "screen's colours; with each panel's outlines over it", full.shape == (T, 2 * T + 6, 3)
          and np.array_equal(full, tiles) and fv.shape[:2] == full.shape[:2] and (fv[..., :3] != full).any(),
          f"{full.shape}")
    fig, _ = x.grid_figure(FS.options({"colorbar": False, "panel_titles": False, "title": False}))
    titles = [ax.get_title() for ax in fig.axes]
    check("no colour bars, no panel titles, no title: two panel axes, no titles, no suptitle",
          len(fig.axes) == 2 and titles == ["", ""] and (fig._suptitle is None or not fig._suptitle.get_text()),
          f"{len(fig.axes)} {titles}")
    fig, _ = x.grid_figure(FS.options({"title_text": "Two redshifts", "title_for": x._grid_title}))
    check("a title typed for this grid replaces its own (the worker's title)", fig._suptitle is not None
          and fig._suptitle.get_text() == "Two redshifts" and x._grid_title.startswith("slice normal"),
          f"{x._grid_title!r}")
    x.save_grid(d / "borderless.png", FS.options({"margins": "none", "dpi": 100}))
    x.save_grid(d / "grid.svg", FS.options())
    x.save_grid(d / "w1500px.png", FS.options({"width": 1500, "width_unit": "px", "font_scale": 1.3}))
    w1500 = _png(d / "w1500px.png").shape[1]
    check("the grid's width: 1500 px asked (larger fonts), 1500 px written within 2", abs(w1500 - 1500) <= 2, str(w1500))
    bl = _png(d / "borderless.png")
    check("margins none: the panels fill the file (2 x 4.2 in by 4.2 in at 100 dpi = 840 x 420); the SVG keeps its "
          "text", bl.shape[:2] == (420, 840) and "<text" in (d / "grid.svg").read_text(errors="replace"), f"{bl.shape}")
    # ---- the title's position in the axes' unit; one unit in every panel; match the data; TIFF
    import types
    from PIL import Image
    a, pos = req["position_at"]
    h = G.hubble(items[0]["output"])
    fig, _ = x.grid_figure(FS.options({"unit": "mpc_h"}))
    sup_h = fig._suptitle.get_text() if fig._suptitle is not None else ""
    fig, _ = x.grid_figure(FS.options({"unit": "cells"}))
    sup_c = fig._suptitle.get_text() if fig._suptitle is not None else ""
    check("the grid's title: the slice position in the axes' unit (Mpc/h, cells), Mpc on screen",
          f"({'xyz'[a]} = {pos * h:.2f} Mpc/h)" in sup_h and f"({'xyz'[a]} = {req['index'] + 0.5:.2f} cells)" in sup_c
          and f"({'xyz'[a]} = {pos:.2f} Mpc)" in x._grid_title, f"{sup_h!r} {sup_c!r} {x._grid_title!r}")
    known, unknown = types.SimpleNamespace(_hubble=0.7), types.SimpleNamespace(_hubble=None)
    o1 = items[1]["output"]
    keep_h, said = o1._hubble, []

    def say(msg):
        said.append(msg)
    x.status.connect(say)
    o1._hubble = None                               # one panel's h not known (a custom run, an opened file)
    try:
        x.save_grid(d / "noh.png", FS.options({"unit": "mpc_h"}))
        fig, _ = x.grid_figure(FS.options({"unit": "mpc_h"}))
        names = [ax.get_xlabel() for ax in fig.axes[:2]]
    finally:
        o1._hubble = keep_h
        x.status.disconnect(say)
    xa = "xyz"[items[0]["rest"][0]]
    check("Mpc/h asked, one panel's h not known: every panel in Mpc and the status line says so (grid_unit); all "
          "known: Mpc/h", names == [f"{xa} [Mpc]"] * 2 and said and "not known" in said[-1] and "--" not in said[-1]
          and E.grid_unit([{"output": known}, None, {"output": known}], "mpc_h") == ("mpc_h", "")
          and E.grid_unit([{"output": known}, {"output": unknown}], "mpc_h")[0] == "mpc", f"{names} {said[-1:]}")
    m = FS.options({"match_data": True})
    fig, _ = x.grid_figure(m)
    dpi = E.grid_dpi(fig, req, items, m)
    fig.set_dpi(dpi)
    widths = [ax.get_window_extent(fig.canvas.get_renderer()).width for ax in fig.axes[:2]]
    check("the grid, match the data: 64-cell panels at 72 dpi in a smaller figure, each panel 64 pixels across "
          "(not about 210, resampled)", dpi == FS.DPI_MIN and all(abs(w - T) < 0.6 for w in widths),
          f"{dpi} {widths}")
    x.save_grid(d / "grid.tif", FS.options())
    x.save_grid(d / "borderless_match.png", FS.options({"margins": "none", "match_data": True, "voids": False}))
    with Image.open(d / "grid.tif") as im:
        comp = im.info.get("compression")
    check("the grid as TIFF: LZW-compressed; no margins with 'match the data': the panels' own pixels, touching as in "
          "that figure (64 x 128)", comp == "tiff_lzw" and _png(d / "borderless_match.png").shape == (T, 2 * T, 3),
          f"{comp} {_png(d / 'borderless_match.png').shape}")
    x.grid_merge.setChecked(True)
    ok = wait_for(qa, lambda: x._grid_shown and not x._grid_busy and x.grid_request()["merged"], 30)
    fig, _ = x.grid_figure(FS.options({"colorbar": False, "panel_titles": False, "tick_labels": False}))
    fig.canvas.draw()
    labels = [t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible() and t.get_text()]
    check("merged: without the colour bar column, the inside labels and the tick labels: two axes, no texts inside, "
          "no tick labels", ok and len(fig.axes) == 2 and not any(ax.texts for ax in fig.axes) and not labels,
          f"{len(fig.axes)} {labels[:3]}")
    try:                                                    # the grid's size guard
        x.save_grid(d / "too_big.png", FS.options({"width": 30000, "width_unit": "px"}))
        refused = ""
    except FS.TooLarge as e:
        refused = str(e)
    check("the grid: a width far too large is refused before drawing (no file)", "pixels" in refused
          and not (d / "too_big.png").exists(), refused)
    x.save_grid(d / "merged_full.png", FS.preset("bare"))
    check("merged at full resolution: no gap between the panels (64 x 128)", _png(d / "merged_full.png").shape == (T, 2 * T, 3))
    x.grid_merge.setChecked(False)
    wait_for(qa, lambda: x._grid_shown and not x._grid_busy and not x.grid_request()["merged"], 30)


def saving_dialog(qa, x):
    """The Save options dialog: presets fill it, an edit makes it 'Custom', 'Your last choice' goes back, Mpc/h only
    with a known h, no '--' in any text; open_save_options keeps what OK says."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QDialog, QGroupBox, QLabel, QLineEdit
    import figsave as FS
    print("Save options: the dialog")
    ctx = x.save_context()
    dlg = FS.make_dialog(x, x.save_options, ctx)
    first = dlg.preset.currentData()
    dlg.preset.setCurrentIndex(dlg.preset.findData("bare"))
    bare = dlg.options()
    dlg.frame.setChecked(not dlg.frame.isChecked())
    after_edit = dlg.preset.currentData()
    dlg.preset.setCurrentIndex(dlg.preset.findData("last"))
    back = dlg.options()
    mpc_h = dlg.unit.model().item(dlg.unit.findData("mpc_h")).isEnabled()
    check("dialog: opens on the preset the options are, a preset fills every option, an edit makes it Custom, 'Your "
          "last choice' brings the opened options back; Mpc/h offered (the header's h is known); the map's size in "
          "the full-resolution choice", first == "standard" and FS.persistent(bare) == FS.persistent(FS.preset("bare"))
          and after_edit == "custom" and FS.persistent(back) == FS.persistent(x.save_options) and mpc_h
          and f"{x._plane.shape[0]} × {x._plane.shape[1]} px" in dlg.kind_full.text(),
          f"{first} {after_edit} {mpc_h} {dlg.kind_full.text()}")
    dlg.kind_full.setChecked(True)
    greyed = not dlg.dpi.isEnabled() and not dlg.title.isEnabled() and not dlg.frame.isEnabled() and dlg.voids.isEnabled()
    dlg.kind_figure.setChecked(True)
    dlg.margins.setCurrentIndex(dlg.margins.findData("none"))
    greyed2 = not dlg.title.isEnabled() and not dlg.colorbar.isEnabled() and dlg.dpi.isEnabled()
    check("full resolution greys the dpi, the title and the axes (the overlays stay); no margins greys the title, axes "
          "and bar", greyed and greyed2)
    texts = [dlg.windowTitle()]
    for wd in dlg.findChildren(QLabel) + dlg.findChildren(QAbstractButton) + dlg.findChildren(QGroupBox) \
            + dlg.findChildren(QLineEdit) + dlg.findChildren(QComboBox):
        texts += [getattr(wd, "text", lambda: "")() if not isinstance(wd, QGroupBox) else wd.title(), wd.toolTip()]
        if isinstance(wd, QLineEdit):
            texts.append(wd.placeholderText())
        if isinstance(wd, QComboBox):
            texts += [wd.itemText(i) for i in range(wd.count())]
            texts += [str(wd.itemData(i, Qt.ToolTipRole) or "") for i in range(wd.count())]
    for wd in (x.view.save_btn, x.view.save_opts_btn, x.grid_save_btn, x.grid_opts_btn):
        texts += [wd.text(), wd.toolTip()]
    check("the dialog's and the save buttons' texts carry no '--'", not [t for t in texts if "--" in t],
          str([t for t in texts if "--" in t][:3]))
    dlg.deleteLater()
    nohub = FS.make_dialog(x, x.save_options, dict(ctx, hubble=None))
    check("without a known h, Mpc/h is not offered", not nohub.unit.model().item(nohub.unit.findData("mpc_h")).isEnabled())
    nohub.deleteLater()
    # Mpc/h remembered, this output's h not known (an opened file, a custom run without an h-free input;
    # grids.hubble's cached answer faked): the map in Mpc, said in the status line and in the dialog's note, and the
    # remembered choice kept for the outputs that know h
    o, d = x.current, TMP / "save map"
    keep_h, said = o._hubble, []

    def say(msg):
        said.append(msg)
    x.status.connect(say)
    real_ask, kept = x._ask_image_path, x.save_options
    x._ask_image_path = lambda *a, **k: (str(d / "noh.png"), "png")
    x.save_options = FS.options({"unit": "mpc_h", "voids": False})
    o._hubble = None
    try:
        x._save_image()
        import matplotlib.pyplot as plt
        fig, _ = x.map_figure()
        label, title = fig.axes[0].get_xlabel(), fig.axes[0].get_title()
        plt.close(fig)
        dlg = FS.make_dialog(x, x.save_options, x.save_context())
        note, unit_ok = dlg.note.text(), dlg.options()["unit"]
        dlg.deleteLater()
        after = x.save_options["unit"]
    finally:
        o._hubble = keep_h
        x._ask_image_path, x.save_options = real_ask, kept
        x.status.disconnect(say)
    check("Mpc/h remembered where h is not known: the map's axes and title in Mpc, the status line and the dialog's "
          "note say so, Mpc/h stays the choice", label.endswith("[Mpc]") and title == x._map_title()
          and said and "h is not known" in said[-1] and "h is not known" in note and "stays your choice" in note
          and "--" not in note and unit_ok == "mpc_h" and after == "mpc_h", f"{label} {said[-1:]} {note!r}")
    real = FS.make_dialog
    saved = []

    def fake(parent, opts, context=None, can_save=True):
        d_ = real(parent, opts, context, can_save)
        d_.exec = lambda: (d_.preset.setCurrentIndex(d_.preset.findData("publication")), d_.accept(),
                           QDialog.Accepted)[-1]
        return d_
    FS.make_dialog = fake
    real_save = x._save_image
    x._save_image = lambda: saved.append(1)
    try:
        ok = x.open_save_options()
    finally:
        FS.make_dialog = real
        x._save_image = real_save
    check("open_save_options: OK keeps the options chosen (Publication) for every save, without saving now; the save "
          "button's tooltip says so", ok and FS.which_preset(x.save_options) == "publication" and not saved
          and "Publication" in x.view.save_btn.toolTip(), x.view.save_btn.toolTip())
    x.set_save_options(FS.options(), remember=False)


def save_settings():
    """The Save options are remembered between sessions in the launcher's QSettings (without a typed title); none
    without settings (the tests' and remember=False windows)."""
    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (save settings): PySide6 is not importable ({e})")
        return
    import json
    import explore as E
    import figsave as FS
    print("Save options: remembered")
    QApplication.instance() or QApplication([])
    ini = TMP / "launcher.ini"
    made = []

    def controls(settings):
        c = E.ExploreControls(E.ExploreView(), lambda: str(TMP / "no data"), lambda: [], settings=settings)
        made.append(c)
        return c
    try:
        x1 = controls(QSettings(str(ini), QSettings.IniFormat))
        fresh = dict(x1.save_options)
        want = dict(FS.preset("slide"), dpi=150, unit="cells", title_text="mine", title_for="t")
        x1.set_save_options(want)
        x1._settings.sync()
        x2 = controls(QSettings(str(ini), QSettings.IniFormat))
        raw = QSettings(str(ini), QSettings.IniFormat).value(E.ExploreControls.SAVE_KEY, "")
        check("nothing remembered: the defaults; set once, a new session (a new window) has them, the typed title not",
              fresh == FS.options() and x2.save_options == FS.persistent(want) and "title_text" not in x2.save_options
              and "mine" not in str(raw) and json.loads(raw)["dpi"] == 150, str(raw))
        st = QSettings(str(ini), QSettings.IniFormat)
        st.setValue(E.ExploreControls.SAVE_KEY, json.dumps({"dpi": 99999, "format": "gif", "frame": "no"}))
        st.sync()
        x3 = controls(QSettings(str(ini), QSettings.IniFormat))
        st.setValue(E.ExploreControls.SAVE_KEY, "{not json")
        st.sync()
        x4 = controls(QSettings(str(ini), QSettings.IniFormat))
        check("a remembered value out of range or of the wrong kind is checked (dpi 1200 at most, PNG, the frame on); "
              "unreadable settings give the defaults", x3.save_options["dpi"] == FS.DPI_MAX
              and x3.save_options["format"] == "png" and x3.save_options["frame"] is True and x4.save_options == FS.options())
        before = ini.read_bytes()
        x5 = controls(None)
        x5.set_save_options(FS.preset("bare"))
        check("without settings (remember off) the options live for the session only: nothing written",
              ini.read_bytes() == before and x5.save_options["kind"] == "full")
    finally:
        for c in made:
            c.shutdown()


# ---------------------------------------------------------------------- the GUI
def spin(qa, ms):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        qa.processEvents()
        time.sleep(0.01)


def wait_for(qa, cond, limit):
    end = time.time() + limit
    while time.time() < end:
        qa.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def gui():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (GUI part): PySide6 is not importable ({e})")
        return
    import explore as E
    import runspec as rs
    print("GUI")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "data"
    make_snapdir(data, 50, 0.0, 7)
    make_snapdir(data, 99, 0.0, 8)
    E.VOID_CACHE_DIR = TMP / "void cache"
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(900, 800)
    asked = []
    x.confirm_heavy = lambda title, text: asked.append((title, text)) or False
    real_procs = rs.dtfe_processes
    rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}      # no DTFE run going
    try:
        x.refresh_outputs()
        ok = wait_for(qa, lambda: not x._busy and x.current is not None and x._plane is not None, 30)
        check("the synthetic snapshot is shown", ok and x.current.sim == SIM, str(x.current and x.current.root))
        x.voids.setChecked(True)
        ok = wait_for(qa, lambda: x._void_proc is None and bool(x._void_cache), 180)
        res = next(iter(x._void_cache.values()), {})
        check("voids ticked with no run going: the helper runs without a question; its result carries R_eff and "
              "the R_v definitions", ok and "error" not in res and not asked and "r_v_defs" in res
              and "crossing" in res["r_v_defs"] and list(x._void_cache)[0][-1] == 0.0,
              str(res.get("error", ""))[:300])
        c = view.canvas
        # the slice through a void's centre: there the outline is biggest
        sel = [k for k in np.nonzero(res.get("_shown", []))[0] if np.isfinite(res["_r_v"][k])]
        check("the fixture has distinct voids with R_v", len(sel) > 0, f"{len(sel)}")
        if not sel:
            return
        k0 = int(max(sel, key=lambda k: res["_r_v"][k]))
        axis = x.axis.currentIndex()
        i0 = int(res["_centres"][k0][axis] / (BOX / N))
        x.index.setValue((i0 + 5) % N)
        wait_for(qa, lambda: not x._busy and x._req["index"] == (i0 + 5) % N, 30)
        hit = next(((cx, cy, kk) for cx, cy, _rx, _ry, kk in c.circles or [] if kk == k0), None)
        check("an outline of the biggest void on a slice near its centre", hit is not None, f"{c.circles}")
        if hit is None:
            return
        r, col = int(hit[1]), int(hit[0])
        x._hover(r, col)
        check("1. hover over the outline: the readout names the void (R_v, R_eff, delta)",
              f"void {k0}: R_v" in view.readout.text() and "R_eff" in view.readout.text(), view.readout.text())
        x._click(r, col)
        info = x.void_info.text()
        check("1. click: the inspector shows R_v, R_eff, the central delta, the flags and the centre; Go to its "
              "centre enabled", f"void {k0}" in info and "R_eff" in info and "central δ" in info and "distinct" in info
              and "centre (" in info and x.void_goto.isEnabled(), info)
        x._void_goto()
        ok = wait_for(qa, lambda: not x._busy and x._req["index"] == i0, 30)
        check("1. Go to its centre: the slice through the void's centre cell", ok and x.index.value() == i0,
              f"{x.index.value()} vs {i0}")
        x.void_reff.setChecked(True)
        qa.processEvents()
        check("1. R_eff too: dashed outlines over the map, the same voids", bool(c.dashed)
              and c.dashed == x._void_circles("_r_eff") and {k for *_a, k in c.dashed} <= set(sel) | set(
                  np.nonzero(res["_shown"])[0]), f"{c.dashed}")
        c._redraw()                                         # draws both kinds without error
        # ---- 2. the saved PNG and its CSV
        png = TMP / "map.png"
        written = x.save_png(png)
        csvp = TMP / "map_voids.csv"
        rows = x.voids_shown()
        lines = csvp.read_text().splitlines() if csvp.is_file() else []
        ids = [int(ln.split(",")[0]) for ln in lines[2:]]
        check("2. Save image: the PNG and '<name>_voids.csv' with exactly the voids outlined (centre, R_v, R_eff, "
              "delta, flags)", png.is_file() and png.stat().st_size > 10000 and written == [str(png), str(csvp)]
              and ids == rows and k0 in ids and lines[1].startswith("void,x_mpc,y_mpc,z_mpc,r_v_mpc")
              and "r_eff_mpc,delta,resolved,distinct" in lines[1], f"{written} {ids} {rows}")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        orig_close = plt.close
        figs = []
        plt.close = lambda fig=None: figs.append(fig)       # keep the figure to count its circles
        try:
            x.save_png(TMP / "map2.png")
        finally:
            plt.close = orig_close
        fig = figs[0] if figs else None
        n_patch = sum(isinstance(p, Circle) for ax in (fig.axes if fig else []) for p in ax.patches)
        check("2. ... the PNG carries the outlines as on screen (two strokes per circle: R_v and R_eff)",
              fig is not None and n_patch == 2 * (len(c.circles) + len(c.dashed)), f"{n_patch} vs {len(c.circles)}+{len(c.dashed)}")
        if fig is not None:
            orig_close(fig)
        # ---- 2. the figure grid: two snapshots side by side, each with its own catalogue
        x.grid_set_layout(1, 2)
        x.grid_fill("redshifts")
        x.show_grid(True)
        ok = wait_for(qa, lambda: x._void_proc is None and len(x._void_cache) >= 2 and not x._grid_busy
                      and x._grid_pending is None and bool(c.circles), 240)
        keys = [k for k in x._void_cache if "error" not in x._void_cache[k]]
        clips = c.circle_clips or []
        check("2. the figure grid: each panel's own catalogue (found one after the other), its outlines clipped to "
              "its tile", ok and len(keys) == 2 and len(clips) == len(c.circles) + len(c.dashed or [])
              and len({cl[0] for cl in clips}) == 2, f"{len(x._void_cache)} {len(clips)}")
        fig, _n = x.grid_figure()
        n_patch = sum(isinstance(p, Circle) for ax in fig.axes for p in ax.patches)
        check("2. ... and the saved grid figure draws them (matplotlib circles in every panel)",
              n_patch > 0 and all(any(isinstance(p, Circle) for p in ax.patches) for ax in fig.axes[:2]), f"{n_patch}")
        saving_grid(qa, x, view)                            # 6. the Save options on the grid, then on the map
        x.show_grid(False)
        wait_for(qa, lambda: not x._busy and not x._grid_shown, 30)
        saving_map(qa, x, view)
        saving_dialog(qa, x)
        # ---- 3. the smoothing
        n_before = len(x._void_cache)
        x.void_sigma.setValue(6.0)
        ok = wait_for(qa, lambda: x._void_proc is None and len(x._void_cache) > n_before, 180)
        key = x._void_cache_key(x.current)
        r6 = x._void_cache.get(key, {})
        r0 = x._void_cache.get(x._void_cache_key(x.current, 0.0), {})
        check("3. a sigma in Mpc: its own cache key, the helper finds the voids at that sigma under its own namespace",
              ok and key[-1] == 6.0 and "error" not in r6 and abs(r6.get("sigma_mpc", 0) - 6.0) < 1e-6
              and r6.get("namespace") != r0.get("namespace") and r0.get("sigma_asked_mpc") == 0.0
              and any((TMP / "void cache").glob("*s6_*")), f"{key} {str(r6.get('error', ''))[:200]} "
              f"{r6.get('namespace')} {r0.get('namespace')}")
        st = x.state()
        check("3. the voids tick, R_eff and sigma are remembered", st.get("voids") is True and st.get("void_reff") is True
              and st.get("void_sigma") == 6.0, f"{st}")
        x.void_sigma.setValue(0.0)
        qa.processEvents()
        check("3. back to production: the cached catalogue, no new search", x._void_proc is None
              and x._void_cache_key(x.current) in x._void_cache)
        # ---- 4. the memory guard
        rs.dtfe_processes = lambda max_age=2.0: {"runs": [(4242, "PS-DTFE", 37.0)], "servers": []}
        asked.clear()
        x._voids_request(x._by_root[next(r for r in x._by_root if "snapdir_099" in r)])   # already cached: no question
        check("4. a cached catalogue is never asked about", not asked)
        os.environ["DTFE_FREE_GB"] = "2"                    # tight: the 21 MB build does not fit in 2 - 4 GB
        x.void_sigma.setValue(3.0)
        qa.processEvents()
        k3 = x._void_cache_key(x.current)
        check("4. a DTFE run going: before finding the voids the launcher asks (the run, its memory, what is free), "
              "and a 'no' starts nothing", len(asked) == 1 and "PS-DTFE pid 4242, 37.0 GB" in asked[0][1]
              and "free" in asked[0][1] and x._void_proc is None and "error" in x._void_cache.get(k3, {}),
              f"{asked} {x._void_cache.get(k3)}")
        asked.clear()
        os.environ["DTFE_FREE_GB"] = "30"
        check("4. a run going but the need fits in the free memory less the headroom: no question, started (said)",
              x.heavy_ok("Start the query server", "A query server", 12.0) and not asked)
        os.environ["DTFE_FREE_GB"] = "8"
        ok_heavy = x.heavy_ok("Start the query server", "A query server", 12.0)
        check("4. heavy_ok: asked with the need in GB, the answer obeyed", not ok_heavy and len(asked) == 1
              and "about 12 GB" in asked[0][1] and "--" not in asked[0][1], f"{asked}")
        x.confirm_heavy = lambda title, text: True
        check("4. ... a 'yes' lets it start", x.heavy_ok("Exact zoom", "it", None))
        rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": [(1, "PS-DTFE", 5.0)]}
        x.confirm_heavy = lambda title, text: asked.append(text) or False
        asked.clear()
        check("4. no run going (a server alone is not a run): never asks, never blocks",
              x.heavy_ok("Exact zoom", "it", 50.0) and not asked)
        line = x.memory_status()
        check("4. the memory status line: free GB and the server with its memory", "free" in line
              and "server pid 1, 5.0 GB" in line, line)
        check("launcher texts carry no '--'", all("--" not in t for t in (E.VOID_TIP, E.VOID_REFF_TIP, E.VOID_SIGMA_TIP,
                                                                           x.void_goto.toolTip())))
    finally:
        rs.dtfe_processes = real_procs
        os.environ.pop("DTFE_FREE_GB", None)
        x.shutdown() if hasattr(x, "shutdown") else None


def view3d_pure():
    """10. view3d's geometry, no Qt."""
    import view3d as V
    print("3D view (pure)")
    rng = np.random.default_rng(3)
    R0 = V.rotation(0.0, 0.0)
    rots = [V.rotation(*rng.uniform(-180, 180, 2)) for _ in range(20)]
    check("10. rotation: orthonormal and right-handed at any turn; (0, 0) looks along +y with x right and z up",
          all(np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0) for R in rots)
          and np.allclose(R0, [[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    cam = V.Camera(target=(30, 30, 30), scale=7.5, yaw=-50, pitch=33, width=640, height=480)
    c, R = np.array([[12.0, 40.0, 25.0]]), 6.0
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    sphere, ellipsoid = V.sphere_matrices([R]), V.ellipsoid_matrices([[9.0, 6.0, 3.5]], [q.T])
    ball = rng.normal(size=(6000, 3))
    ball /= np.linalg.norm(ball, axis=1)[:, None]
    cxy, cd = cam.project(c)

    def reach(M):                       # how far the projected surface reaches in the outline's own unit (1 = on it)
        u, w = V.outline_axes(cam, M)
        xy, _ = cam.project(c + ball @ M[0].T)
        return u[0], w[0], np.linalg.norm((xy - cxy) @ np.linalg.inv(np.stack([u[0], w[0]], 1)).T, axis=1).max()
    su, sw, s_reach = reach(sphere)
    _u, _w, e_reach = reach(ellipsoid)
    check("10. a sphere of radius R is a circle of R x scale on the screen at any turn (the voids at their true size "
          "against the box); an ellipsoid's outline is the projection's own (its surface reaches it, never past)",
          np.allclose([np.hypot(*su), np.hypot(*sw)], R * 7.5) and abs(su @ sw) < 1e-9
          and abs(s_reach - 1) < 1e-3 and abs(e_reach - 1) < 1e-3, f"{s_reach} {e_reach}")
    check("10. ellipsoid_matrices: column i is direction i (the catalogue's row) times semi-axis i",
          np.allclose(ellipsoid[0], q * [9.0, 6.0, 3.5]) and np.allclose(sphere[0], R * np.eye(3)))
    sec_ok = True
    t = np.linspace(0.05, np.pi - 0.05, 9)
    a = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    for M in (sphere, ellipsoid):
        for i, j in V.SECTIONS:
            su, sw = V.section_axes(cam, M[:, :, i], M[:, :, j])
            curve = c + np.cos(a)[:, None] * M[0][:, i] + np.sin(a)[:, None] * M[0][:, j]
            rxy, rdepth = cam.project(curve)
            pts = cxy + np.cos(t)[:, None] * su[0] + np.sin(t)[:, None] * sw[0]
            near = np.argmin(np.hypot(*(rxy[None, :, :] - pts[:, None, :]).transpose(2, 0, 1)), axis=1)
            sec_ok &= bool(np.hypot(*(rxy[near] - pts).T).max() < 0.2 and np.all(rdepth[near] < cd[0]))
    check("10. the principal sections (a sphere's great circles, an ellipsoid's): centre + cos t u + sin t v is the "
          "projected section, its near half (0 < t < pi) in front of the centre", sec_ok)
    # the cuts on a plane: an ellipsoid's agrees with the inside test point by point; spheres' = the map's circles
    cut = V.plane_cuts(c, ellipsoid, 64.0, 1, 38.0, (0.0, 0.0), (60.0, 60.0), 300, 300)
    uv = rng.uniform(0, 60, (20000, 2))
    p3 = np.stack([uv[:, 0], np.full(len(uv), 38.0), uv[:, 1]], 1)
    S_inv = np.linalg.inv(ellipsoid[0] @ ellipsoid[0].T)
    inside = np.einsum("ni,ij,nj->n", p3 - c, S_inv, p3 - c) <= 1
    tex = np.stack([uv[:, 0] * 5.0, (60.0 - uv[:, 1]) * 5.0], 1)
    in_cut = (np.linalg.norm((tex - cut[0][:2]) @ np.linalg.inv(cut[0][2]).T, axis=1) <= 1) if len(cut) == 1 else None
    cs, rs = rng.uniform(0, 64, (40, 3)), rng.uniform(2, 9, 40)
    ours = V.plane_cuts(cs, V.sphere_matrices(rs), 64.0, 2, 30.5, (0, 0), (64, 64), 128, 128)
    theirs = G.void_circles(cs, rs, 64.0, 2, 30.5, [0, 1], (0, 0), (64, 64), 128, 128)
    check("10. where a shape cuts a plane: an ellipsoid's cut agrees with the inside test at every point; spheres' cuts "
          "are the map's circles (grids.void_circles: the same places, radii and periodic images)",
          in_cut is not None and inside.sum() > 100 and not np.any(inside != in_cut) and len(ours) == len(theirs) > 3
          and all(np.allclose(a_[:2], b_[:2]) and a_[3] == b_[4] and np.allclose(sorted(np.linalg.svd(a_[2])[1]),
                                                                                 sorted(b_[2:4]))
                  for a_, b_ in zip(ours, theirs)), f"{len(cut)} {len(ours)} {len(theirs)}")
    big, small = (100.0, 100.0, np.array([40.0, 0.0]), np.array([0.0, 20.0]), 3), \
        (110.0, 100.0, np.array([6.0, 2.0]), np.array([-1.0, 5.0]), 8)
    check("10. picking an outline: the smallest one containing the point (a small void over a big one stays "
          "reachable), none outside", V.outline_at([big, small], 111.0, 101.0) == 8
          and V.outline_at([big, small], 100.0, 118.0) == 3 and V.outline_at([big, small], 100.0, 125.0) is None)
    # the quads' order: along random rays, the nearest quad is the one painted last
    lo, hi = np.zeros(3), np.full(3, 60.0)
    wrong = overlaps = 0
    for _ in range(60):
        pos, shown = rng.uniform(5, 55, 3), rng.random(3) < 0.85
        cam = V.Camera(scale=8, yaw=rng.uniform(-180, 180), pitch=rng.uniform(-85, 85), width=600, height=600)
        cam.fit(lo, hi)
        quads = V.plane_quads(lo, hi, pos, shown)
        rank = {i: n for n, (_kind, i) in enumerate(V.paint_order(quads, np.zeros((0, 3)), cam, pos, lo, hi))}
        d = -cam.R[2]
        for x, y in rng.uniform(0, 600, (150, 2)):
            p0, hits = cam.unproject(x, y), []
            for i, q in enumerate(quads):
                a = q["axis"]
                if abs(d[a]) < 1e-12:
                    continue
                s = (pos[a] - p0[a]) / d[a]
                pt = p0 + s * d
                r0, r1 = V.rest_axes(a)
                if q["u"][0] <= pt[r0] <= q["u"][1] and q["v"][0] <= pt[r1] <= q["v"][1]:
                    hits.append((s, i))
            if len(hits) > 1:
                overlaps += 1
                wrong += max(hits, key=lambda h: rank[h[1]])[1] != min(hits)[1]
    check("10. the planes cut into quads (12 for three planes inside the window) and painted back to front exactly: "
          "along every ray the nearest quad is painted last", len(V.plane_quads(lo, hi, [30, 30, 30])) == 12
          and overlaps > 1000 and wrong == 0, f"{wrong} wrong of {overlaps}")
    cam = V.Camera(target=(5, 6, 7), scale=3.0, yaw=20, pitch=-40, width=300, height=200)
    m11, m12, m21, m22, dx, dy = V.texture_map(1, 4.5, (1.0, 2.0), (9.0, 12.0), 80, 100, cam)
    tex = np.array([[0, 0], [80, 0], [0, 100], [80, 100]], dtype=float)
    on_screen = np.stack([tex[:, 0] * m11 + tex[:, 1] * m21 + dx, tex[:, 0] * m12 + tex[:, 1] * m22 + dy], 1)
    want, _ = cam.project([[1.0, 4.5, 12.0], [9.0, 4.5, 12.0], [1.0, 4.5, 2.0], [9.0, 4.5, 2.0]])
    check("10. a plane's texture is mapped onto the screen where its corners project (the top row at the high end)",
          np.allclose(on_screen, want))
    # the rule's size is the close-up's choice (view3d.close_up_window): the test holds it to its properties
    centre = np.array([3.0, 50.0, 10.0])
    wins = [V.close_up_window(centre, r, 0.5, 64.0) for r in (0.2, 2.0, 5.0, 12.0, 40.0)]
    half = [0.5 * (hi - lo) for lo, hi in wins]
    lo_r, hi_r = V.close_up_window(centre, 2.0, 0.5, 64.0, reach=9.0)
    check("10. the close-up window: a cube centred on the void, holding its sphere, growing with the radius, a few "
          "cells at least and never more than the box; it holds the whole drawn shape (an ellipsoid's a)",
          np.all(0.5 * (hi_r - lo_r) >= 9.0) and
          all(np.allclose(0.5 * (lo + hi), centre) and np.allclose(h, h[0]) for (lo, hi), h in zip(wins, half))
          and all(h[0] >= r for h, r in zip(half[:4], (0.2, 2.0, 5.0, 12.0)))
          and all(a[0] <= b[0] for a, b in zip(half, half[1:])) and half[0][0] >= 2 * 0.5 and half[-1][0] <= 32.0,
          f"{[float(h[0]) for h in half]}")
    idx, wlo, whi = V.cell_window([-3.2, 10.0, 60.0], [4.1, 20.0, 70.0], np.zeros(3), np.full(3, 1.0), 64)
    pos = V.plane_position(62, 0.0, 1.0, -4.0, 5.0, 64.0)
    check("10. the cell window: whole cells covering it, read periodically across the faces (-4..5 -> cells 60..63, "
          "0..4); a plane's position is its image inside the window",
          list(idx[0]) == [60, 61, 62, 63, 0, 1, 2, 3, 4] and np.allclose(wlo, [-4, 10, 60]) and np.allclose(whi, [5, 20, 70])
          and list(idx[2]) == [60, 61, 62, 63, 0, 1, 2, 3, 4, 5] and pos == -1.5
          and V.plane_position(62, 0.0, 1.0, 0.0, 64.0, 64.0) == 62.5, f"{list(idx[0])} {wlo} {whi} {pos}")
    imgs = V.nearest_images([[62.0, 1.0, 30.0], [10.0, 33.0, 30.0]], [2.0, 63.0, 30.0], 64.0)
    check("10. a close-up near a face shows the voids across it at their nearest image",
          np.allclose(imgs, [[-2.0, 65.0, 30.0], [10.0, 33.0, 30.0]]), str(imgs))


def view3d_gui():
    """10. the 3D view in the window, on gui()'s synthetic snapshot (its catalogue is cached by then)."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (3D view): PySide6 is not importable ({e})")
        return
    from PySide6.QtCore import QAbstractAnimation
    import explore as E
    import explore_3d as X3
    import runspec as rs
    import view3d as V
    print("3D view")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "data"
    if not (data / SIM).is_dir():
        make_snapdir(data, 50, 0.0, 7)
    E.VOID_CACHE_DIR = TMP / "void cache"
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(900, 760)
    view.show()
    real_procs = rs.dtfe_processes
    rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}
    try:
        x.refresh_outputs()
        wait_for(qa, lambda: not x._busy and x._plane is not None, 30)
        x.voids.setChecked(True)
        ok = wait_for(qa, lambda: x._void_proc is None and bool(x._void_cache), 180)
        res = x._void_cache.get(x._void_cache_key(x.current), {})
        check("10. the catalogue for the 3D view", ok and "error" not in res, str(res.get("error", ""))[:200])
        if "error" in res or not res:
            return
        o = x.current
        axis, index = x.axis.currentIndex(), x.index.value()
        check("10. the 3D view draws a volume by default (the planes one choice away), the planes' controls hidden; "
              "in perspective", x.view3d_style.currentData() == "volume" and view.view3d.style == "volume"
              and x.volume3d_box.isVisibleTo(x) and not x.planes3d_box.isVisibleTo(x)
              and x.view3d_persp.isChecked() and view.view3d.cam.perspective)
        x.view3d_style.setCurrentIndex(x.view3d_style.findData("planes"))     # the planes' checks (view3d_volume_gui:
        x.view3d_persp.setChecked(False)                                       # the volume's), orthographic: the
        x.view3d_toggle.setChecked(True)                                       # voids at their true size
        ok = wait_for(qa, lambda: x._3d_res is not None and not x._3d_busy, 30)
        qa.processEvents()
        v3, pl = view.view3d, (x._3d_res or {}).get("planes") or [None] * 3
        want_index = [o.n // 2] * 3
        want_index[axis] = index
        same = all(p is not None and np.array_equal(p["values"], o.slice("density", a, want_index[a]))
                   for a, p in enumerate(pl))
        check("10. Show in 3D: the 3D view in the map's place, the streams table hidden; three planes, the map's slice "
              "and the middle of the other axes, the stored values (the whole box, no stride at 64^3)",
              ok and v3.isVisible() and not view.canvas.isVisible() and not view.streams.isVisible() and same
              and x._3d_index == want_index and "3D, the whole box" in view.title.text(), f"{x._3d_index} {want_index}")
        rows = sorted(int(k) for k in np.nonzero(res["_shown"] & np.isfinite(res["_r_v"]))[0])
        circles = {c[4]: (c[0], c[1], float(np.hypot(*c[2]))) for c in v3._outlines}     # (x, y, radius)
        check("10. the distinct voids with an R_v as spheres, each a circle of R_v x scale; the void menu lists them, "
              "the largest first", sorted(circles) == rows and len(rows) > 0
              and all(abs(circles[k][2] - res["_r_v"][k] * v3.cam.scale) < 1e-6 for k in rows)
              and [x.void3d.itemData(j) for j in range(1, x.void3d.count())]
              == sorted(rows, key=lambda k: -res["_r_v"][k]), f"{sorted(circles)} {rows}")
        # the map's slider moves its plane
        x.index.setValue((index + 7) % o.n)
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_res["request"]["index"][axis] == (index + 7) % o.n, 30)
        moved = x._3d_res["planes"][axis]["pos"]
        check("10. the map's slider moves the 3D view's plane along its axis",
              ok and abs(moved - (o.lo[axis] + ((index + 7) % o.n + 0.5) * o.length[axis] / o.n)) < 1e-9, f"{moved}")
        # hover and click a void: the one whose circle holds a point no other circle does
        k0 = max(rows, key=lambda k: res["_r_v"][k])
        cx, cy, r = circles[k0][:3]
        spot = next(((cx + f * r * np.cos(t), cy + f * r * np.sin(t)) for f in (0.0, 0.5, 0.8)
                     for t in np.linspace(0, 2 * np.pi, 12) if v3.sphere_at(cx + f * r * np.cos(t),
                                                                               cy + f * r * np.sin(t)) == k0), None)
        check("10. the void's outline can be pointed at", spot is not None)
        if spot is None:
            return
        x._3d_hover(*spot)
        check("10. hover: the readout names the void (R_v, R_eff, delta) and the view highlights it",
              f"void {k0}: R_v" in view.readout.text() and v3.hover_row == k0, view.readout.text())
        x._3d_click(*spot)
        check("10. click: the void inspector shows its entry (Go to its centre enabled), the menu selects it, the view "
              "draws it as the one inspected", x._void_picked == (x._void_cache_key(o), k0)
              and f"void {k0}" in x.void_info.text() and x.void_goto.isEnabled()
              and x.void3d.currentData() == k0 and v3.selected == k0 and x.void3d_close.isEnabled())
        # a plane's value under the mouse: the middle of the front-most quad
        _a, T, rect = v3._drawn[0]
        mid = T.map(rect.center())
        pt = v3.point_at(mid.x(), mid.y())
        ok_pt = pt is not None
        if ok_pt:
            a, xyz, value = pt
            ijk = [int(np.floor((xyz[b] - o.lo[b]) / (o.length[b] / o.n))) % o.n for b in range(3)]
            cube = np.fromfile(o.files["density"].path, dtype=o.dtype).reshape((o.n,) * 3)
            ok_pt = abs(value - float(cube[tuple(ijk)])) < 1e-6 and ijk[a] == x._3d_index[a]
        check("10. a plane point under the mouse: its position and the stored value of that cell", ok_pt, f"{pt}")
        # close-up
        scale0 = v3.cam.scale
        x._3d_double(*spot)
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_close is not None
                      and x._3d_res["request"]["lo"][0] == x._3d_close["lo"][0], 30)
        c = res["_centres"][k0]
        cell = o.length / o.n
        lo, hi = V.close_up_window(c, res["_r_v"][k0], float(cell.min()), float(res["box_mpc"]))
        through = [int(np.floor((c[b] - o.lo[b]) / cell[b])) % o.n for b in range(3)]
        idx, wlo, _whi = V.cell_window(lo, hi, o.lo, cell, o.n)
        crop_ok = True
        for a, p in enumerate(x._3d_res["planes"]):
            r0, r1 = V.rest_axes(a)
            crop_ok &= p is not None and np.array_equal(p["values"], o.slice("density", a, through[a])[np.ix_(idx[r0], idx[r1])])
        near = V.nearest_images(res["_centres"][rows], c, float(res["box_mpc"]))
        inside = [k for k, keep in zip(rows, V.in_window(near, res["_r_v"][rows], x._3d_res["lo"], x._3d_res["hi"])) if keep]
        check("10. double-click: a close-up on the void, the planes through its centre's cell, cut to its window "
              "read periodically across the faces, the voids reaching into it at their nearest image",
              ok and x._3d_close["row"] == k0 and x._3d_index == through and crop_ok
              and np.allclose(x._3d_res["lo"], wlo) and sorted(int(r) for r in v3.voids["rows"]) == sorted(inside)
              and f"close-up on void {k0}" in view.title.text() and x.void3d_back.isEnabled(),
              f"{x._3d_index} {through} {crop_ok} {list(v3.voids['rows'])} {inside}")
        wait_for(qa, lambda: v3._anim is None or v3._anim.state() == QAbstractAnimation.Stopped, 10)   # the flight
        qa.processEvents()                                  # (320 ms; longer on a busy machine: never a fixed wait)
        seen = next(((cc[0], cc[1], float(np.hypot(*cc[2]))) for cc in v3._outlines if cc[4] == k0), None)
        check("10. ... the camera flown to the void's window: the void in the middle of the view, still a circle of "
              "R_v x the new scale", np.allclose(v3.cam.target, 0.5 * (lo + hi)) and seen is not None
              and abs(seen[0] - v3.width() / 2) < 1 and abs(seen[1] - v3.height() / 2) < 1
              and abs(seen[2] - res["_r_v"][k0] * v3.cam.scale) < 1e-6 and v3.cam.scale > scale0, f"{seen}")
        png = TMP / "view3d.png"
        written = x.save_3d_png(png)
        from PySide6.QtGui import QImage
        im = QImage(written)
        check("10. Save image in 3D: a PNG of the view as on screen", png.is_file() and im.width() >= v3.width(),
              f"{im.width()} {v3.width()}")
        x.whole_box_3d()
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_close is None and x._3d_res["request"]["lo"][0] == o.lo[0], 30)
        check("10. Whole box: the box's planes where they were", ok and x._3d_index[axis] == (index + 7) % o.n
              and np.allclose(x._3d_res["lo"], o.lo) and not x.void3d_back.isEnabled(), f"{x._3d_index}")
        # the fitted ellipsoids: the helper's JSON carries them (R_eff = (a b c)^(1/3) of them), the view draws them
        semi, dirs = res["_semi"][rows], res["_dirs"][rows]
        check("10. the catalogue's fitted ellipsoids reach the launcher: a >= b >= c in Mpc with (a b c)^(1/3) = R_eff, "
              "their directions orthonormal; the inspector says a, b, c",
              np.all(np.isfinite(semi)) and np.all(np.diff(semi, axis=1) <= 1e-9)
              and np.allclose(np.prod(semi, axis=1) ** (1 / 3), res["_r_eff"][rows], rtol=1e-4)
              and all(np.allclose(d @ d.T, np.eye(3), atol=1e-4) for d in dirs) and "a, b, c = " in x.void_info.text(),
              f"{semi[:2]} {res['_r_eff'][rows][:2]}")
        x.void3d_shape.setCurrentIndex(x.void3d_shape.findData("ellipsoid"))
        qa.processEvents()
        v3.repaint()
        M = V.ellipsoid_matrices(semi, dirs)
        u, w = V.outline_axes(v3.cam, M)
        drawn = {o[4]: o for o in v3._outlines}
        check("10. 'fitted ellipsoids': every void drawn as its own ellipsoid (the outline its projection's), labelled "
              "with its semi-axes; remembered", sorted(drawn) == rows
              and all(np.allclose(drawn[k][2], u[j]) and np.allclose(drawn[k][3], w[j]) for j, k in enumerate(rows))
              and v3.voids["labels"][0].startswith(f"void {rows[0]} · a, b, c = ")
              and x.state().get("void_shape") == "ellipsoid", f"{sorted(drawn)} {rows}")
        x.close_up_3d(k0)
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_close is not None
                      and x._3d_res["request"]["lo"][0] == x._3d_close["lo"][0], 30)
        j0 = rows.index(k0)
        lo_e, hi_e = V.close_up_window(c, res["_r_eff"][k0], float(cell.min()), float(res["box_mpc"]), semi[j0].max())
        check("10. ... a close-up sized by R_eff, holding the whole ellipsoid; its neighbours drawn dim; the menu "
              "sorted by R_eff", ok and np.allclose(x._3d_close["lo"], lo_e) and np.allclose(x._3d_close["hi"], hi_e)
              and v3.focus and v3.selected == k0
              and [x.void3d.itemData(j) for j in range(1, x.void3d.count())]
              == sorted(rows, key=lambda k: -res["_r_eff"][k]) and "R_eff" in x.void3d.itemText(1))
        x.whole_box_3d()
        x.void3d_shape.setCurrentIndex(x.void3d_shape.findData("sphere"))
        wait_for(qa, lambda: not x._3d_busy and x._3d_close is None, 30)
        x.grid_set_layout(1, 2)
        x.grid_fill("redshifts")
        x.show_grid(True)
        wait_for(qa, lambda: not x._grid_busy, 60)
        g_ok = x._grid_shown and not x._3d_shown and not v3.isVisible() and not x.view3d_toggle.isChecked()
        x.view3d_toggle.setChecked(True)
        wait_for(qa, lambda: not x._3d_busy and not x._busy, 30)
        check("10. the figure grid and the 3D view replace each other", g_ok and x._3d_shown and not x._grid_shown
              and v3.isVisible())
        x.view3d_toggle.setChecked(False)
        ok = wait_for(qa, lambda: not x._busy and x._plane is not None, 30)
        qa.processEvents()
        check("10. back to the map: the canvas, the streams table and the map's title", ok and view.canvas.isVisible()
              and view.streams.isVisible() and not v3.isVisible() and "3D" not in view.title.text()
              and x.view3d_toggle.text() == "Show in 3D")
        # ---- the Sizes range: the same voids on the map, the grid's panels, the saves and the 3D view
        r_v, r_eff = res["_r_v"], res["_r_eff"]
        x.void_rmax.setValue(float(np.mean(sorted(r_v[rows])[:2])))     # between the two smallest R_v
        qa.processEvents()
        hi_r = x.void_rmax.value()
        keep_v = [k for k in rows if r_v[k] <= hi_r]
        circ_rows = {k for *_a, k in (x._void_circles() or [])}
        check("10. Sizes: an upper bound keeps the voids of R_v up to it on the map (outlines, the figure grid's panels), "
              "the count says how many", list(x.void_rows(res)) == keep_v and 0 < len(keep_v) < len(rows)
              and circ_rows <= set(keep_v) and list(x._panel_voids(o)["rows"]) == keep_v
              and x.void_count.text() == f"{len(keep_v)} of {len(rows)} voids", f"{keep_v} {circ_rows} {x.void_count.text()}")
        x.view3d_toggle.setChecked(True)
        wait_for(qa, lambda: not x._3d_busy and x._3d_res is not None, 30)
        qa.processEvents()
        menu = [x.void3d.itemData(j) for j in range(1, x.void3d.count())]
        check("10. ... and in the 3D view: the spheres and the menu", sorted(int(k) for k in v3.voids["rows"]) == keep_v
              and sorted(menu) == keep_v, f"{list(v3.voids['rows'])} {menu}")
        x.void3d_shape.setCurrentIndex(x.void3d_shape.findData("ellipsoid"))
        qa.processEvents()
        keep_e = [k for k in rows if r_eff[k] <= hi_r]
        check("10. ... the ellipsoids by their R_eff (the size drawn), said in the count",
              sorted(int(k) for k in v3.voids["rows"]) == keep_e and "(by R_eff)" in x.void_count.text(),
              f"{list(v3.voids['rows'])} {keep_e} {x.void_count.text()}")
        x.void3d_shape.setCurrentIndex(x.void3d_shape.findData("sphere"))
        x.void_rmax.setValue(0.0)
        x.void_rmin.setValue(1000.0)
        qa.processEvents()
        v3.repaint()
        check("10. ... a range no void is in: none drawn, the menu says so, the view still paints",
              len(v3.voids["rows"]) == 0 and x.void3d.itemText(0) == "no void in the Sizes range" and v3._outlines == []
              and x.void_count.text() == f"0 of {len(rows)} voids")
        x.close_up_3d(k0)
        wait_for(qa, lambda: not x._3d_busy and x._3d_close is not None, 30)
        qa.processEvents()
        st = x.state()
        check("10. ... a close-up keeps the void it looks at whatever the range; the range is remembered",
              list(v3.voids["rows"]) == [k0] and st.get("void_rmin") == 1000.0 and st.get("void_rmax") == 0.0)
        x.whole_box_3d()
        x.void_rmin.setValue(0.0)
        wait_for(qa, lambda: not x._3d_busy and x._3d_close is None, 30)
        x.view3d_toggle.setChecked(False)
        wait_for(qa, lambda: not x._busy, 30)
        check("10. ... no range: every distinct void again", list(x.void_rows(res)) == rows
              and x.void_count.text() == f"all {len(rows)} voids")
        check("10. the 3D texts carry no '--'; a 2D grid is refused with the reason",
              all("--" not in t for t in (X3.VIEW3D_TIP, X3.CLOSE_TIP))
              and X3.View3DMixin.view3d_applicable(type("Flat", (), {"dim": 2})()) == "only for 3D grids"
              and X3.View3DMixin.view3d_applicable(o) == "")
    finally:
        rs.dtfe_processes = real_procs
        x.shutdown()


# ---------------------------------------------------------------------- 11. the volume
def _vol_set(d: Path, n: int = 20, seed: int = 11):
    """A small grid set of a side no power of two (density lognormal, velocity of three components) in a box of n
    Mpc: one cell a Mpc."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rho = np.exp(rng.normal(size=(n, n, n))).astype(np.float32)
    vel = rng.normal(size=(n, n, n, 3)).astype(np.float32)
    rho.tofile(str(d / "vol.a_den"))
    vel.tofile(str(d / "vol.a_vel"))
    return G.OutputSet(d / "vol", box=((0.0,) * 3, (float(n),) * 3), redshift=0.0), rho, vel


def _block_mean_ref(a, edges) -> np.ndarray:
    """The block means of a, block by block (the reference for volume3d.average_window)."""
    m = [len(e) - 1 for e in edges]
    out = np.empty(m)
    for i in range(m[0]):
        for j in range(m[1]):
            for k in range(m[2]):
                out[i, j, k] = a[edges[0][i]:edges[0][i + 1], edges[1][j]:edges[1][j + 1],
                                 edges[2][k]:edges[2][k + 1]].mean()
    return out


def _rgba64(img) -> np.ndarray:
    """A Format_RGBA64_Premultiplied QImage's pixels: (h, w, 4) floats in 0..1, premultiplied."""
    h, w = img.height(), img.width()
    raw = np.frombuffer(img.constBits(), dtype=np.uint16, count=img.bytesPerLine() * h // 2)
    return raw.reshape(h, img.bytesPerLine() // 2)[:, :4 * w].reshape(h, w, 4).astype(np.float64) / 65535.0


def perspective_pure():
    """12. the perspective camera's geometry (view3d), no Qt."""
    import view3d as V
    import volume3d as VOL
    print("3D perspective (pure)")
    rng = np.random.default_rng(12)
    lo, hi = np.zeros(3), np.full(3, 100.0)
    cam = V.Camera(width=800, height=700, yaw=-35, pitch=25, perspective=True)
    cam.bounds = (lo, hi)
    cam.fit(lo, hi)
    ortho = cam.copy()
    ortho.perspective = False
    # on the plane through the target facing the viewer the scale is 'scale' (as orthographic); nearer is larger
    R = cam.R
    flat = cam.target + rng.uniform(-40, 40, (20, 1)) * R[0] + rng.uniform(-40, 40, (20, 1)) * R[1]
    corners = V.box_corners(lo, hi)
    xy, depth = cam.project(corners)
    lens = {k: sorted((np.hypot(*(xy[a | (1 << k)] - xy[a])), depth[a] + depth[a | (1 << k)])
                      for a in range(8) if not a & (1 << k)) for k in range(3)}
    k_across = int(np.argmin(np.abs(cam.toward())))       # the axis most across the view (no foreshortening)
    near_longer = [e_[0] for e_ in sorted(lens[k_across], key=lambda e: e[1])]
    near_longer = all(a_ > b_ for a_, b_ in zip(near_longer, near_longer[1:]))
    seg = rng.uniform(0, 100, (2, 3))
    mid, _ = cam.project(0.5 * (seg[0] + seg[1]))
    ends, _ = cam.project(seg)
    cross = (ends[1] - ends[0])[0] * (mid[0] - ends[0])[1] - (ends[1] - ends[0])[1] * (mid[0] - ends[0])[0]
    m = V.FIT_MARGIN
    inside = np.all(np.abs(xy - [400, 350]) <= [m * 400 + 1e-6, m * 350 + 1e-6]) and \
        np.isclose(max(np.abs(xy[:, 0] - 400).max() / 400, np.abs(xy[:, 1] - 350).max() / 350), m)
    check("12. perspective: the plane through the target at 'scale' (as orthographic), a box's near edges drawn "
          "longer than its far ones (those across the view), a straight line straight; fitted, its corners inside the margin, one on it; the "
          "eye some 2.2 sides away", np.allclose(cam.project(flat)[0], ortho.project(flat)[0]) and near_longer
          and abs(cross) < 1e-6 * np.hypot(*(ends[1] - ends[0])) ** 2 and inside
          and 1.8 < cam.distance() / 100.0 < 2.8, f"{near_longer} {cross} {inside} {cam.distance()}")
    pts = rng.uniform(-20, 120, (50, 3))
    ok_m = True
    for c in (cam, ortho):
        h = np.c_[pts, np.ones(len(pts))] @ c.matrix().T
        ok_m &= bool(np.allclose(h[:, :2] / h[:, 2:], c.project(pts)[0]))
    check("12. the camera matrix gives project()'s pixels (either projection)", ok_m)
    # zooming in is a dolly along the cursor's ray: into the box, on through the field, up to its far side
    z, step = cam.copy(), 2.0 ** 0.25
    cx, cy = 520.0, 300.0                                 # the cursor, off the centre
    ray = z.unproject(cx, cy) - z.eye()
    ray /= np.linalg.norm(ray)
    eyes, dist, targ, under = [z.eye()], [z.distance()], [z.target.copy()], []
    for _ in range(60):
        keep = z.unproject(cx, cy)
        z.zoom(step, cx, cy)
        eyes.append(z.eye())
        dist.append(z.distance())
        targ.append(z.target.copy())
        under.append(float(np.hypot(*(z.project(keep)[0][0] - [cx, cy]))))
    eyes, dist, targ = np.array(eyes), np.array(dist), np.array(targ)
    inside = np.all((eyes > lo) & (eyes < hi), axis=1)
    first_in = int(np.argmax(inside))
    ahead = V.DOLLY_MIN * 100.0
    fly = np.flatnonzero(np.isclose(dist, ahead))         # the steps flown: the target kept 'ahead' of the eye
    wall = int(fly[-1]) + 1                               # from here the target is at the far side: closing in
    off = eyes - eyes[0]
    on_ray = np.abs(off - (off @ ray)[:, None] * ray).max(axis=1)
    moved = np.linalg.norm(np.diff(eyes[fly], axis=0), axis=1)
    plain = np.flatnonzero(np.array(dist[1:]) > ahead)    # the first steps: the plain rule (scale x step)
    check("12. zooming in (perspective) dollies the eye along the cursor's ray, into the box (inside it within a "
          "dozen steps of the fitted view and from then on); far from the box it is the plain rule (the distance "
          "/ step, the point under the cursor stays); with the target DOLLY_MIN of the box ahead it flies on, the "
          "same distance every step, the target inside the box; at the far side it closes in on the wall",
          not inside[0] and 0 < first_in <= 12 and np.all(inside[first_in:]) and len(plain) >= 5
          and np.allclose(dist[plain + 1], dist[plain] / step) and max(under[i] for i in plain) < 1e-6
          and len(fly) >= 5 and np.all(np.diff(fly) == 1) and on_ray[:wall].max() < 1e-6
          and np.allclose(moved, moved[0]) and moved[0] > 0.1 * ahead * (1.0 - 1.0 / step)
          and np.all((targ >= lo - 1e-9) & (targ <= hi + 1e-9)) and wall < 60
          and np.all(np.diff(dist[wall:]) < 0) and np.all(np.array(dist[wall:]) < ahead) and dist[-1] > 0,
          f"in at {first_in}, plain {len(plain)}, flown {len(fly)}, wall {wall}, off ray {on_ray[:wall].max():.2g}, "
          f"moved {moved.min():.4g}..{moved.max():.4g}, last distance {dist[-1]:.3g}")
    back = z.copy()
    for _ in range(80):
        back.zoom(1.0 / step)
    out_d = back.distance()
    back2 = back.copy()
    keep = back2.unproject(cx, cy)
    back2.zoom(1.0 / step, cx, cy)
    ortho_z = ortho.copy()
    ortho_z.zoom(80.0)
    check("12. zooming out backs the eye away again (out of the box, the plain rule: the distance x step, the "
          "point under the cursor stays); orthographic, a zoom only magnifies: no eye, the target at the centre",
          not np.all((back.eye() > lo) & (back.eye() < hi)) and out_d > 300.0
          and np.isclose(back2.distance(), out_d * step)
          and np.hypot(*(back2.project(keep)[0][0] - [cx, cy])) < 1e-6
          and np.isinf(ortho_z.distance()) and np.isclose(ortho_z.scale, 80.0 * ortho.scale)
          and np.allclose(ortho_z.target, ortho.target), f"{out_d} {back2.distance()}")
    # what can be drawn of a line that passes the eye: its part in front, cut to the neighbourhood of the view
    zin = cam.copy()
    for _ in range(14):
        zin.zoom(step)
    ein = zin.eye()
    in_box = bool(np.all((ein > lo) & (ein < hi)))
    fwd = -zin.toward()
    a_pt, b_pt = ein + 30.0 * fwd + 4.0 * zin.R[0], ein - 25.0 * fwd + 9.0 * zin.R[1]     # ahead, behind
    seg_ab = V.segment_on_screen(zin, a_pt, b_pt)
    seg_ba = V.segment_on_screen(zin, b_pt, a_pt)
    mid_pt = a_pt + 0.3 * (b_pt - a_pt)                   # still ahead of the eye
    pa, pm = zin.project(a_pt)[0][0], zin.project(mid_pt)[0][0]
    cross_s = (pm - pa)[0] * (np.array(seg_ab[1]) - pa)[1] - (pm - pa)[1] * (np.array(seg_ab[1]) - pa)[0]
    grown = lambda q_: -800 - 1e-6 <= q_[0] <= 1600 + 1e-6 and -700 - 1e-6 <= q_[1] <= 1400 + 1e-6   # noqa: E731
    whole = V.segment_on_screen(cam, corners[0], corners[7])
    check("12. segment_on_screen: a segment wholly in view is its projected ends; one passing the eye (inside the "
          "box) is drawn from its end ahead along its own projected line to the view's neighbourhood, either way "
          "round; one behind the eye not at all; project() has no pixel for a point behind the eye",
          in_box and np.allclose(whole, cam.project(corners[[0, 7]])[0]) and seg_ab is not None
          and np.allclose(seg_ab[0], pa) and abs(cross_s) < 1e-6 * max(1.0, np.hypot(*(pm - pa))) * 2400.0
          and grown(seg_ab[1]) and np.hypot(*(np.array(seg_ab[1]) - pa)) > 50.0
          and np.allclose(seg_ba[0], seg_ab[1]) and np.allclose(seg_ba[1], seg_ab[0])
          and V.segment_on_screen(zin, b_pt, ein + 5.0 * zin.toward()) is None
          and np.all(np.isnan(zin.project(b_pt)[0])) and len(V.hidden_edges(zin, lo, hi)) == 12,
          f"{in_box} {seg_ab} {pa} {cross_s}")
    # a plane's texture through a homography: corners and inside land where they project; orthographic = texture_map
    tex = np.array([[0, 0], [80, 0], [0, 100], [80, 100], [23.0, 61.0]], dtype=float)
    want, _ = cam.project([[10.0, 45.0, 90.0], [90.0, 45.0, 90.0], [10.0, 45.0, 5.0], [90.0, 45.0, 5.0],
                           [10.0 + 23.0, 45.0, 90.0 - 61.0 * 0.85]])
    T = V.texture_transform(1, 45.0, (10.0, 5.0), (90.0, 90.0), 80, 100, cam)
    wt = T[2] * tex[:, 0] + T[5] * tex[:, 1] + T[8]
    got = np.stack([(T[0] * tex[:, 0] + T[3] * tex[:, 1] + T[6]) / wt, (T[1] * tex[:, 0] + T[4] * tex[:, 1] + T[7]) / wt], 1)
    To = V.texture_transform(1, 45.0, (10.0, 5.0), (90.0, 90.0), 80, 100, ortho)
    Ta = V.texture_map(1, 45.0, (10.0, 5.0), (90.0, 90.0), 80, 100, ortho)
    check("12. texture_transform: a plane's texture lands where its points project through the perspective "
          "homography (corners and inside); orthographic it is texture_map's affine map",
          np.allclose(got, want) and np.allclose([To[0], To[1], To[3], To[4], To[6], To[7]], Ta)
          and np.allclose([To[2], To[5], To[8]], [0, 0, 1]), f"{np.abs(got - want).max()}")
    # the outlines: the image of the tangent cone, an ellipse the projected surface reaches all round, never past
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    shapes = [(np.array([[30.0, 60.0, 40.0]]), V.ellipsoid_matrices([[20.0, 12.0, 6.0]], [q.T])),
              (np.array([[80.0, 15.0, 85.0]]), V.sphere_matrices([9.0]))]
    ball = rng.normal(size=(20000, 3))
    ball /= np.linalg.norm(ball, axis=1)[:, None]
    reach = []
    for c, M in shapes:
        x0, u, v = V.outline_ellipses(cam, c, M)
        loc = (cam.project(c + ball @ M[0].T)[0] - x0[0]) @ np.linalg.inv(np.stack([u[0], v[0]], 1)).T
        rho, ang = np.linalg.norm(loc, axis=1), np.arctan2(loc[:, 1], loc[:, 0])
        reach.append((rho.max(), min(rho[(ang >= a) & (ang < a + np.pi / 18)].max()
                                     for a in np.linspace(-np.pi, np.pi, 36, endpoint=False))))
    ox, ou, ov = V.outline_ellipses(ortho, shapes[0][0], shapes[0][1])
    au, av = V.outline_axes(ortho, shapes[0][1])
    around = V.outline_ellipses(cam, cam.eye()[None, :], V.sphere_matrices([5.0]))
    check("12. outline_ellipses in perspective: the projected ellipsoid (and sphere) reaches its outline all round "
          "and never past it; orthographic it is the projected centre with outline_axes; a shape around the eye "
          "has none", all(abs(a_ - 1) < 1e-3 and b_ > 0.99 for a_, b_ in reach)
          and np.allclose(ox, ortho.project(shapes[0][0])[0]) and np.allclose(ou, au) and np.allclose(ov, av)
          and np.all(np.isnan(around[1])), f"{reach}")
    near, far = V.section_points(cam, shapes[0][0][0], shapes[0][1][0][:, 0], shapes[0][1][0][:, 1], n=64)
    e = cam.eye()
    A, B, c0 = shapes[0][1][0][:, 0], shapes[0][1][0][:, 1], shapes[0][0][0]
    t = np.linspace(0, 2 * np.pi, 4000)
    curve = c0 + np.cos(t)[:, None] * A + np.sin(t)[:, None] * B
    cxy, _ = cam.project(curve)
    side = (curve - c0) @ (e - c0)
    on = lambda pts_: np.hypot(*(cxy[None, :, :] - pts_[:, None, :]).transpose(2, 0, 1)).min(axis=1)   # noqa: E731
    near_i = np.argmin(np.hypot(*(cxy[None, :, :] - near[1:-1, None, :]).transpose(2, 0, 1)), axis=1)
    far_i = np.argmin(np.hypot(*(cxy[None, :, :] - far[1:-1, None, :]).transpose(2, 0, 1)), axis=1)
    check("12. section_points: both halves lie on the projected section, the near one on the eye's side",
          on(near).max() < 0.3 and on(far).max() < 0.3 and np.all(side[near_i] > 0) and np.all(side[far_i] < 0))
    # a planar curve's image is an ellipse: the voids' cuts through a plane's homography, their central sections
    T = V.texture_transform(1, 45.0, (10.0, 5.0), (90.0, 90.0), 160, 170, cam)
    H = V.texture_matrix(T)
    tt = np.linspace(0, 2 * np.pi, 400)
    cut_err, cut_ok = 0.0, True
    for x0_, y0_, L_ in ((50.0, 60.0, np.array([[12.0, 3.0], [-2.0, 7.0]])), (130.0, 20.0, np.diag([30.0, 9.0]))):
        (img,) = V.cut_images(T, [(x0_, y0_, L_, 0)])
        cut_ok &= img is not None
        hp = np.stack([x0_ + L_[0, 0] * np.cos(tt) + L_[0, 1] * np.sin(tt), y0_ + L_[1, 0] * np.cos(tt)
                       + L_[1, 1] * np.sin(tt), np.ones_like(tt)], 1) @ H.T
        loc = (hp[:, :2] / hp[:, 2:] - img[0]) @ np.linalg.inv(np.stack([img[1], img[2]], 1)).T
        cut_err = max(cut_err, float(np.abs(np.linalg.norm(loc, axis=1) - 1).max()))
    cs = rng.uniform(15, 85, (6, 3))
    As, Bs = rng.normal(size=(6, 3)) * 9, rng.normal(size=(6, 3)) * 6
    sec_err, near_ok = 0.0, True
    for c_ in (cam, ortho):
        xy_, u_, v_, st_, sp_ = V.section_arcs(c_, cs, As, Bs)
        for k in range(6):
            curve = cs[k] + np.cos(tt)[:, None] * As[k] + np.sin(tt)[:, None] * Bs[k]
            pts_, _ = c_.project(curve)
            loc = (pts_ - xy_[k]) @ np.linalg.inv(np.stack([u_[k], v_[k]], 1)).T
            sec_err = max(sec_err, float(np.abs(np.linalg.norm(loc, axis=1) - 1).max()))
            wv = (c_.eye() - cs[k]) if c_.perspective else c_.toward()
            ang = np.arctan2(loc[:, 1], loc[:, 0])
            inside_arc = np.mod(ang - st_[k] if sp_[k] > 0 else st_[k] - ang, 2 * np.pi) < abs(sp_[k])
            front = (curve - cs[k]) @ wv > 1e-6 * np.linalg.norm(wv)
            near_ok &= bool(np.mean(inside_arc[front]) > 0.98 and np.mean(inside_arc[~front]) < 0.02)
    check("12. the image of a planar curve in perspective is an ellipse: a void's cut through its plane's homography "
          "and a central section land on the ellipses drawn (to 1e-9), the near arc holding exactly the section's "
          "half on the viewer's side (either camera)", cut_ok and cut_err < 1e-9 and sec_err < 1e-9 and near_ok,
          f"{cut_err} {sec_err} {near_ok}")
    # a plane's origin corner behind the eye: the map keeps w > 0 for the points in front
    c4 = cam.copy()
    c4.bounds = (np.full(3, 40.0), np.full(3, 60.0))
    c4.fit(*c4.bounds)
    c4.zoom(4.0)
    e4 = c4.eye()
    T4 = V.texture_transform(1, 50.0, (0.0, 0.0), (100.0, 100.0), 100, 100, c4)
    H4 = V.texture_matrix(T4)
    w_origin = (c4.matrix() @ np.array([0.0, 50.0, 100.0, 1.0]))[2]
    w_in = (H4 @ np.array([50.0, 50.0, 1.0]))[2]                      # the plane's middle, inside the window
    check("12. plane_transform divides by |w|: with the texture's origin corner behind the eye (a whole-box plane "
          "drawn against a close-up) the points in front keep w > 0 (Qt draws them)", w_origin < 0 < w_in,
          f"{w_origin} {w_in} {e4}")
    # the slicing axis in perspective is each pixel's own: the axis its ray crosses the most slices of
    shape5 = (40, 52, 34)
    cell5 = (hi - lo) / np.array(shape5, dtype=np.float64)
    W5, H5 = 61, 49                                       # odd: a pixel's centre in the middle of the view
    xs5, ys5 = (v.ravel() for v in np.meshgrid(np.arange(W5) + 0.5, np.arange(H5) + 0.5))
    bad_axis = bad_part = bad_mid = bad_fly = n_in = n_px = 0
    for _ in range(300):
        c5 = V.Camera(width=W5, height=H5, yaw=rng.uniform(-180, 180), pitch=rng.uniform(-85, 85), perspective=True)
        c5.bounds = (lo, hi)
        c5.fit(lo, hi)
        for _n in range(int(rng.integers(0, 30))):
            c5.zoom(2.0 ** 0.25, *rng.uniform((0, 0), (W5, H5)))
        c5.pan(*rng.uniform(-20, 20, 2))
        e5 = c5.eye()
        n_in += bool(np.all((e5 > lo) & (e5 < hi)))
        plan5 = VOL.slice_plan(c5, lo, hi, shape5)
        ids = VOL.part_ids(plan5, W5, H5).ravel()
        pts = c5.target + ((xs5 - 0.5 * W5)[:, None] * c5.R[0] - (ys5 - 0.5 * H5)[:, None] * c5.R[1]) / c5.scale
        d5 = (pts - e5) / cell5                           # each pixel's ray, in cells
        A5 = np.sort(np.abs(d5), axis=1)
        clear = A5[:, 2] > A5[:, 1] * (1.0 + 1e-9)        # not on a seam between two parts
        ax5 = np.argmax(np.abs(d5), axis=1)
        want = 2 * ax5 + (d5[np.arange(len(d5)), ax5] < 0)
        bad_axis += int(np.count_nonzero(clear & (ids != want)))
        mid = (H5 // 2) * W5 + W5 // 2
        bad_mid += bool(clear[mid] and ids[mid] // 2 != plan5["axis"])
        parts5 = {VOL.part_id(q_): q_ for q_ in plan5["parts"]}
        eye_c = (e5 - lo) / cell5
        for pid in np.unique(ids[clear]):
            sel = clear & (ids == pid)
            ax, sg = int(pid) // 2, (1 if pid % 2 == 0 else -1)
            r0, r1 = V.rest_axes(ax)
            k = np.arange(shape5[ax]) + 0.5               # the slices' positions along the axis, in cells
            t = (k[None, :] - eye_c[ax]) / d5[sel, ax][:, None]
            hit = (t > 0)
            for r in (r0, r1):
                at = eye_c[r] + t * d5[sel, r][:, None]
                hit &= (at > 1e-9) & (at < shape5[r] - 1e-9)
            crossed = np.flatnonzero(hit.any(axis=0))     # the slices some pixel of the part looks at
            n_px += int(np.count_nonzero(hit.any(axis=1)))
            q_ = parts5.get(int(pid))
            bad_part += bool(crossed.size and (q_ is None or q_["sign"] != sg
                                              or not set(crossed.tolist()) <= set(int(i) for i in q_["order"])))
        # the parts follow the turn alone: flying on (and panning) leaves every pixel with its part
        c6 = c5.copy()
        c6.zoom(2.0 ** 0.25, *rng.uniform((0, 0), (W5, H5)))
        c6.pan(*rng.uniform(-10, 10, 2))
        ids6 = VOL.part_ids(VOL.slice_plan(c6, lo, hi, shape5), W5, H5).ravel()
        bad_fly += int(np.count_nonzero(clear & (ids6 != ids)))
    ortho5 = V.Camera(width=W5, height=H5, yaw=20, pitch=30)
    ortho5.fit(lo, hi)
    plan_o = VOL.slice_plan(ortho5, lo, hi, shape5)
    check("12. the slicing axis in perspective is each pixel's own (part_ids): the axis its ray crosses the most "
          "slices of and the sense it runs along it, the middle pixel's the plan's 'axis'; every slice a pixel "
          "looks at inside the window is in its part's order (300 turns, zooms from far outside to inside the "
          "window, pans; cells not cubic); a part follows the turn alone, so flying on leaves every pixel with "
          "its own; orthographic: one part, no part_ids",
          bad_axis == 0 and bad_part == 0 and bad_mid == 0 and bad_fly == 0 and 60 < n_in < 280 and n_px > 100000
          and VOL.part_ids(plan_o, W5, H5) is None and len(plan_o["parts"]) == 1
          and plan_o["parts"][0]["first"] is None,
          f"{bad_axis} {bad_part} {bad_mid} {bad_fly}, {n_in} inside, {n_px} px")
    # the edges behind the box: hidden exactly when the line of sight to them crosses the box
    bad, tried = 0, 0
    for _ in range(40):
        c2 = V.Camera(width=600, height=600, yaw=rng.uniform(-180, 180), pitch=rng.uniform(-80, 80), perspective=True)
        c2.bounds = (lo, hi)
        c2.fit(lo, hi)
        c2.zoom(rng.uniform(0.6, 3.0))
        c2.pan(*rng.uniform(-150, 150, 2))
        e2, hid = c2.eye(), V.hidden_edges(c2, lo, hi)
        for i, (a_, b_) in enumerate(V.BOX_EDGES):
            pt = 0.5 * (corners[a_] + corners[b_])
            f = np.linspace(0.0, 0.995, 400)[:, None]
            ray = e2 + f * (pt - e2)
            through = np.any(np.all((ray > lo + 1e-6) & (ray < hi - 1e-6), axis=1))
            bad += (i in hid) != through
            tried += 1
    check("12. hidden_edges in perspective: an edge is dashed exactly when the line of sight to it crosses the box "
          "(any turn, zoom and pan)", bad == 0 and tried == 480, f"{bad} of {tried}")
    # the planes' order from the eye: along every ray from it the nearest quad is painted last
    wrong = overlaps = 0
    for _ in range(40):
        pos, shown = rng.uniform(5, 95, 3), rng.random(3) < 0.85
        c3 = V.Camera(width=600, height=600, yaw=rng.uniform(-180, 180), pitch=rng.uniform(-85, 85), perspective=True)
        c3.bounds = (lo, hi)
        c3.fit(lo, hi)
        quads = V.plane_quads(lo, hi, pos, shown)
        rank = {i: n for n, (_k, i) in enumerate(V.paint_order(quads, np.zeros((0, 3)), c3, pos, lo, hi))}
        e3 = c3.eye()
        for x, y in rng.uniform(0, 600, (150, 2)):
            dvec = c3.unproject(x, y) - e3
            hits = []
            for i, qd in enumerate(quads):
                a_ = qd["axis"]
                if abs(dvec[a_]) < 1e-12:
                    continue
                s_ = (pos[a_] - e3[a_]) / dvec[a_]
                pt = e3 + s_ * dvec
                r0, r1 = V.rest_axes(a_)
                if s_ > 0 and qd["u"][0] <= pt[r0] <= qd["u"][1] and qd["v"][0] <= pt[r1] <= qd["v"][1]:
                    hits.append((s_, i))
            if len(hits) > 1:
                overlaps += 1
                wrong += max(hits, key=lambda h: rank[h[1]])[1] != min(hits)[1]
    check("12. the planes' quads from the eye: along every ray from it the nearest quad is painted last",
          overlaps > 300 and wrong == 0, f"{wrong} wrong of {overlaps}")
    # the zoom from any view: a cursor off the box, any cursor once inside, after a pan and a turn, at the far wall
    off_line = off_plain = 0.0
    never_in = True
    for cx_, cy_ in ((100.0, 600.0), (780.0, 20.0)):          # lines of sight that miss the box (the fitted view)
        z2 = cam.copy()
        e_ = [z2.eye()]
        r_ = z2.unproject(cx_, cy_) - z2.eye()
        r_ /= np.linalg.norm(r_)
        for _ in range(60):
            d_ = z2.distance()
            z2.zoom(step, cx_, cy_)
            off_plain = max(off_plain, abs(z2.distance() * step / d_ - 1.0))
            e_.append(z2.eye())
        e_ = np.array(e_)
        never_in &= not np.any(np.all((e_ > lo) & (e_ < hi), axis=1))
        o_ = e_ - e_[0]
        off_line = max(off_line, float(np.abs(o_ - (o_ @ r_)[:, None] * r_).max()))
    check("12. ... a cursor off the box (its line of sight misses the box): the plain rule every step, the eye on "
          "that line all the way (no jump onto the box)", never_in and off_line < 1e-6 and off_plain < 1e-9,
          f"off the line {off_line:.3g}, off the rule {off_plain:.3g}, inside {not never_in}")
    left = entered = t_out = 0
    for case in range(120):
        hi_b = np.full(3, 100.0) if case % 2 else np.array([100.0, 60.0, 30.0])
        c4 = V.Camera(width=640, height=480, yaw=rng.uniform(0, 360), pitch=rng.uniform(-80, 80), perspective=True)
        c4.bounds = (lo, hi_b)
        c4.fit(lo, hi_b)
        was_in = False
        x_, y_ = rng.uniform(0, 640), rng.uniform(0, 480)
        for notch in range(60):
            if notch % 7 == 6:                                # the cursor moves now and then
                x_, y_ = rng.uniform(0, 640), rng.uniform(0, 480)
            c4.zoom(step, x_, y_)
            e_ = c4.eye()
            now_in = bool(np.all((e_ >= lo - 1e-9) & (e_ <= hi_b + 1e-9)))
            left += was_in and not now_in
            t_out += now_in and not np.all((c4.target >= lo - 1e-3) & (c4.target <= hi_b + 1e-3))
            was_in = was_in or now_in
        entered += was_in
    check("12. ... once the eye is inside the box no zoom in carries it out again, and the target stays in the box "
          "with it (any turn, any cursor, a box of unequal sides)", left == 0 and t_out == 0 and entered >= 60,
          f"{left} steps out, {t_out} targets out, {entered} of 120 entered")
    tried = never = went_out = off = backward = past = 0
    slowest = 0
    for case in range(150):
        hi_b = np.full(3, 100.0) if case % 2 else np.array([100.0, 60.0, 30.0])
        c4 = V.Camera(width=800, height=700, yaw=rng.uniform(-180, 180), pitch=rng.uniform(-80, 80), perspective=True)
        c4.bounds = (lo, hi_b)
        c4.fit(lo, hi_b)
        c4.pan(rng.uniform(-300, 300), rng.uniform(-300, 300))
        c4.turn(rng.uniform(-200, 200), rng.uniform(-100, 100))
        c4.target = c4.target + rng.uniform(-150, 150) * c4.toward()    # its plane in front of or beyond the box
        e0 = c4.eye()
        p_ = lo + rng.uniform(0.15, 0.85, 3) * (hi_b - lo)              # a point of the box, under the cursor
        (x_, y_), = c4.project(p_[None])[0]
        if np.all((e0 > lo) & (e0 < hi_b)) or not (0 < x_ < 800 and 0 < y_ < 700) or (e0 - p_) @ c4.toward() <= 0:
            continue
        tried += 1
        r_ = (p_ - e0) / np.linalg.norm(p_ - e0)
        with np.errstate(divide="ignore"):
            leave = float(np.where(r_ > 0, (hi_b - e0) / r_, np.where(r_ < 0, (lo - e0) / r_, np.inf)).min())
        e_ = [e0]
        for _ in range(80):
            c4.zoom(step, x_, y_)
            e_.append(c4.eye())
        e_ = np.array(e_)
        is_in = np.all((e_ >= lo - 1e-7) & (e_ <= hi_b + 1e-7), axis=1)
        along = (e_ - e0) @ r_
        first = int(np.argmax(is_in))
        never += not is_in.any()
        went_out += bool(is_in.any()) and not is_in[first:].all()
        off += np.linalg.norm((e_ - e0) - along[:, None] * r_, axis=1).max() > 1e-6
        backward += bool(np.any(np.diff(along) <= 0))
        past += along.max() > leave + 1e-6
        slowest = max(slowest, first)
    check("12. ... after a pan and a turn (the target's plane in front of the box or beyond it) a zoom in with the "
          "cursor on the box still enters it, within 30 steps: forward every step along the cursor's line of sight, "
          "never past the far side", tried >= 80 and never == went_out == off == backward == past == 0
          and slowest <= 30, f"{tried} tried: never in {never}, out again {went_out}, off the line {off}, backward "
          f"{backward}, past the far side {past}, slowest entry {slowest}")
    at_wall = cam.copy()
    for _ in range(60):
        at_wall.zoom(step)
    e_w, d_w = at_wall.eye(), at_wall.distance()
    out1 = at_wall.copy()
    out1.zoom(1.0 / step)
    step_out = float(np.linalg.norm(out1.eye() - e_w))
    at_wall.turn(180.0 / V.TURN_PER_PIXEL, 0.0)                         # turned round: the whole box ahead again
    moves = []
    for _ in range(6):
        a_ = at_wall.eye()
        at_wall.zoom(step)
        moves.append(float(np.linalg.norm(at_wall.eye() - a_)))
    flight = (1.0 - 1.0 / step) * ahead
    check("12. ... at the far wall (the target a small distance ahead): one step out backs the eye away as from a "
          "target DOLLY_MIN of the box ahead, and turned round the eye flies at the full step again",
          0 < d_w < 0.05 * ahead and np.isclose(step_out, (step - 1.0) * ahead)
          and min(moves) > flight * (1.0 - 1e-9) and np.allclose(moves[1:], flight)
          and np.isclose(at_wall.distance(), ahead),
          f"distance {d_w:.3g}, step out {step_out:.4g}, moves {np.round(moves, 3)}, flight {flight:.4g}")
    # screen_polygon: a picture's rectangle cut to what of it is in front of the eye and near the view, against
    # the map taken backward on a grid of screen points
    Wp, Hp, mp = 640, 480, 64
    gx, gy = np.meshgrid(np.linspace(-Wp + 3.0, 2 * Wp - 3.0, 97), np.linspace(-Hp + 3.0, 2 * Hp - 3.0, 73))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    kinds = {"whole": 0, "behind": 0, "passes the eye": 0, "past the view": 0}
    bad_poly, judged = [], 0
    for case in range(240):
        c5 = V.Camera(width=Wp, height=Hp, yaw=rng.uniform(-180, 180), pitch=rng.uniform(-85, 85),
                      perspective=case % 4 != 3)
        c5.bounds = (lo, hi)
        c5.fit(lo, hi)
        c5.target = rng.uniform(10.0, 90.0, 3)                          # the eye put by hand: mostly inside the box
        c5.scale = (c5.focal / rng.uniform(2.0, 60.0)) if c5.perspective else c5.scale * rng.uniform(0.5, 12.0)
        axis_ = int(rng.integers(3))
        T9 = V.texture_transform(axis_, float(rng.uniform(5, 95)), (0.0, 0.0), (100.0, 100.0), mp, mp, c5)
        x0_, y0_ = rng.uniform(0, 0.5 * mp, 2)
        rect = (0.0, 0.0, float(mp), float(mp)) if case % 3 else (x0_, y0_, *rng.uniform(4.0, 0.5 * mp, 2))
        m11, m12, m13, m21, m22, m23, m31, m32, m33 = T9
        A_ = np.array([[m11, m21, m31], [m12, m22, m32], [m13, m23, m33]])
        cs = np.array([(rect[0], rect[1]), (rect[0] + rect[2], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                       (rect[0], rect[1] + rect[3])])
        hc = np.column_stack([cs, np.ones(4)]) @ A_.T
        poly = V.screen_polygon(T9, rect, Wp, Hp)
        if np.all(hc[:, 2] <= 0.0):
            kinds["behind"] += 1
            if len(poly):
                bad_poly.append((case, "behind"))
            continue
        on_grid = np.all(hc[:, 2] > 0.0) and bool(np.all(
            (np.abs(hc[:, 0] / hc[:, 2] - 0.5 * Wp) < 1.5 * Wp) & (np.abs(hc[:, 1] / hc[:, 2] - 0.5 * Hp) < 1.5 * Hp)))
        if on_grid:
            kinds["whole"] += 1
            if poly.shape != (4, 2) or not np.allclose(poly, hc[:, :2] / hc[:, 2:3]):
                bad_poly.append((case, "whole"))
            continue
        kinds["passes the eye" if np.any(hc[:, 2] <= 0.0) else "past the view"] += 1
        back = grid @ np.linalg.inv(A_).T                               # (x, y, 1) / w of the picture's point there
        with np.errstate(divide="ignore", invalid="ignore"):
            tx, ty = back[:, 0] / back[:, 2], back[:, 1] / back[:, 2]
        ref_in = (back[:, 2] > 0.0) & (tx >= rect[0]) & (tx <= rect[0] + rect[2]) & (ty >= rect[1]) \
            & (ty <= rect[1] + rect[3])
        if len(poly) < 3:
            if ref_in.any():
                bad_poly.append((case, "empty"))
            continue
        nxt = np.roll(poly, -1, axis=0)
        edge = nxt - poly
        length = np.hypot(edge[:, 0], edge[:, 1])
        turn_ = np.sign(np.sum(poly[:, 0] * nxt[:, 1] - nxt[:, 0] * poly[:, 1]))
        real = length > 1e-9
        side = turn_ * (edge[real, 0][None] * (grid[:, 1][:, None] - poly[real, 1][None])
                        - edge[real, 1][None] * (grid[:, 0][:, None] - poly[real, 0][None])) / length[real][None]
        deep, clear = np.all(side >= 2.0, axis=1), np.any(side <= -2.0, axis=1)     # 2 px from its outline
        judged += int(deep.sum() + clear.sum())
        if (not np.all(np.isfinite(poly)) or np.any(poly[:, 0] < -Wp - 1e-6) or np.any(poly[:, 0] > 2 * Wp + 1e-6)
                or np.any(poly[:, 1] < -Hp - 1e-6) or np.any(poly[:, 1] > 2 * Hp + 1e-6)
                or np.any(deep & ~ref_in) or np.any(clear & ref_in)):
            bad_poly.append((case, "cut", int((deep & ~ref_in).sum()), int((clear & ref_in).sum())))
    check("12. screen_polygon: a rectangle wholly in front and near the view is its four mapped corners, one "
          "wholly behind the eye is nothing, one that passes the eye or runs past the view grown by its own size "
          "is cut there (finite corners; against the map taken backward on a grid of screen points; either "
          "projection)", not bad_poly and min(kinds.values()) >= 4 and kinds["whole"] >= 20
          and kinds["passes the eye"] >= 30 and kinds["past the view"] >= 30 and judged > 100000,
          f"{bad_poly[:5]} {kinds} judged {judged}")


def volume_pure():
    """11. the volume's averaging, levels, transfer function and slice geometry, no Qt."""
    import view3d as V
    import volume3d as VOL
    print("3D volume (pure)")
    e = VOL.block_edges(20, 7)
    sizes = np.diff(e)
    check("11. block_edges: every cell in exactly one block, the blocks as equal as whole cells allow, never more "
          "blocks than cells", e[0] == 0 and e[-1] == 20 and len(e) == 8 and sizes.min() >= 1
          and sizes.max() - sizes.min() <= 1 and len(VOL.block_edges(5, 9)) == 6, f"{e}")
    n = 20
    o, rho, vel = _vol_set(TMP / "vol", n)
    VOL.drop_volumes()
    prog = {}
    avg = VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 7, prog)
    ed = [VOL.block_edges(n, 7)] * 3
    ref = _block_mean_ref(rho.astype(np.float64), ed)
    weight = np.einsum("i,j,k->ijk", *[np.diff(x).astype(np.float64) for x in ed])
    check("11. the whole box averaged to 7 cells a side (20 is no multiple of 7): each cell its block's mean, the "
          "integral kept (a density's mass); the progress counted every block",
          avg["values"].shape == (7, 7, 7) and np.allclose(avg["values"], ref, rtol=1e-6)
          and np.isclose((avg["values"] * weight).sum(), rho.astype(np.float64).sum(), rtol=1e-6)
          and prog == {"done": 7, "total": 7} and np.allclose(avg["lo"], 0.0) and np.allclose(avg["hi"], n)
          and avg["cells"] == (n, n, n), f"{prog} {np.abs(avg['values'] - ref).max()}")
    check("11. ... kept: the same request again reads nothing", VOL.average_window(o, "density", "value", o.lo,
                                                                                   o.lo + o.length, 7) is avg)
    # the fast paths (blocks of one size, the window at its own cells) are the summing path's own sums, bit for bit:
    # reduceat adds a block as a0 + (0 + a1 + a2 + ...), which rounds apart from left to right under cancellation
    m = 12                                          # 12^3: no plane of a square side (12^3 is no square)
    g = np.ones((m, m, m), dtype=np.float32)
    g[0, 0, 0:4] = [16777216.0, 1.0, 2.0 ** -29, 2.0 ** -29]
    g[1, 0:3, 2] = [1.0, 1e20, -1e20]
    g[2, 3, 3] = -0.0
    d = TMP / "vol_exact"
    d.mkdir(exist_ok=True)
    g.tofile(str(d / "ex.a_den"))
    oe = G.OutputSet(d / "ex", box=((0.0,) * 3, (float(m),) * 3), redshift=0.0)

    def summed(arr, k):                             # the general path: float64, reduceat along z then y, sum over x
        e = VOL.block_edges(m, k)
        a = np.add.reduceat(np.add.reduceat(arr.astype(np.float64), e[:-1], axis=2), e[:-1], axis=1)
        s = np.diff(e).astype(np.float64)
        return np.stack([a[e[j]:e[j + 1]].sum(axis=0) / ((e[j + 1] - e[j]) * s[:, None] * s[None, :])
                         for j in range(k)]).astype(np.float32)
    exact = []
    for k in (6, 4, 3, m):                          # blocks of 2, 3, 4, and the window at its own cells
        VOL.drop_volumes()
        exact.append(VOL.average_window(oe, "density", "value", oe.lo, oe.lo + oe.length, k)["values"].tobytes()
                     == summed(g, k).tobytes())
    VOL.drop_volumes()
    own = VOL.average_window(oe, "density", "value", oe.lo, oe.lo + oe.length, 256)["values"]
    check("11. blocks of one size (2, 3, 4 cells) summed as reduceat sums them, bit for bit, under heavy cancellation; "
          "the window at its own cells the grid's values, a -0.0 cell +0.0 as the sums made it",
          all(exact) and not np.signbit(own[2, 3, 3]) and own[0, 0, 0] == np.float32(16777216.0), f"{exact}")
    lo, hi = np.array([-3.2, 15.5, 4.0]), np.array([6.1, 24.9, 12.0])
    idx, wlo, _whi = V.cell_window(lo, hi, o.lo, o.length / n, n)
    win = rho[np.ix_(*idx)].astype(np.float64)
    avg2 = VOL.average_window(o, "density", "value", lo, hi, 4)
    check("11. a window across the box's faces (a close-up near one): its cells read periodically, then averaged",
          int(idx[0][0]) == 16 and int(idx[1][-1]) == 4 and np.allclose(avg2["lo"], wlo)
          and np.allclose(avg2["values"], _block_mean_ref(win, [VOL.block_edges(len(i), 4) for i in idx]), rtol=1e-6))
    nrm = np.sqrt((vel.astype(np.float64) ** 2).sum(-1)) * o._velocity_factor("velocity")
    comps = G.components("velocity", 3)
    avg3 = VOL.average_window(o, "velocity", "norm", o.lo, o.lo + o.length, 7)
    avg4 = VOL.average_window(o, "velocity", comps[2], o.lo, o.lo + o.length, 7)
    check("11. a vector field: the component the map shows (the norm, one by name: slice()'s rule), then averaged",
          np.allclose(avg3["values"], _block_mean_ref(nrm, ed), rtol=1e-6)
          and np.allclose(o.slice("velocity", 0, 3, "norm"), nrm[3])
          and np.allclose(avg4["values"], _block_mean_ref(vel[..., 1].astype(np.float64), ed), rtol=1e-5, atol=1e-6)
          and np.allclose(o.slice("velocity", 0, 3, comps[2]), vel[3, :, :, 1]), f"{comps}")
    # a cube the map has cached already is used, never a new one read into the cache
    seen, real = [], VOL.read_rows

    def spy(path, dtype, shape, rows, cube=None):
        seen.append(cube is not None)
        return real(path, dtype, shape, rows, cube)
    VOL.read_rows = spy
    try:
        VOL.drop_volumes()
        G.drop_cubes()
        VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 7)
        cold = list(seen)
        G._cached_cube(o.files["density"].path, o.dtype, (n, n, n))
        VOL.drop_volumes()
        seen.clear()
        warm = VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 7)
    finally:
        VOL.read_rows = real
        G.drop_cubes()
    check("11. the grid read by plain reads (no memmap) and none of it put in the cube cache; a cube the map cached "
          "is used instead, the same numbers", cold and not any(cold) and seen and all(seen)
          and np.array_equal(warm["values"], avg["values"]), f"{cold} {seen}")
    time.sleep(0.01)
    (2.0 * rho).tofile(str(o.files["density"].path))            # the grid written again by a re-run
    again = VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 7)
    rho.tofile(str(o.files["density"].path))
    stop = {"cancel": True}
    try:
        VOL.drop_volumes()
        VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 7, stop)
        cancelled = False
    except VOL.Cancelled:
        cancelled = True
    check("11. a grid written again is read again (its mtime and size in the key); 'cancel' stops an averaging",
          np.allclose(again["values"], 2.0 * ref, rtol=1e-6) and cancelled)
    # levels
    s = VOL.scaled(avg["values"], True)
    q, qr = VOL.quantize(s)
    lv = VOL.level_values(qr)
    step = (qr[1] - qr[0]) / (VOL.LEVELS - 2)
    check("11. levels: 1..255 over the values' range (log10 of a density), each value within half a level of its "
          "own; 0 for no value (NaN)", int(q.min()) == 1 and int(q.max()) == 255
          and np.all(np.abs(lv[q] - s) <= 0.5 * step + 1e-6)
          and VOL.quantize(np.array([np.nan, 0.0, 1.0], np.float32))[0][0] == 0)
    prep = VOL.prepare(avg, True)
    fine = prep["fine"]
    oriented_ok = all(np.array_equal(VOL.slice_images(fine, a)[i], np.moveaxis(q, a, 0)[i])
                      for a in range(3) for i in range(7)) and np.array_equal(fine["qT"], np.moveaxis(q, 2, 0)) \
        and fine["qT"].flags.c_contiguous and np.array_equal(fine["q"], q) and prep["coarse"] is None
    lo_d, hi_d = VOL.default_range(prep, "density")
    frac = VOL.visible_fraction(prep, lo_d)
    check("11. prepare (in chunks on the pool): the same levels as quantising the whole cube; a slice's image has "
          "its rows along the first remaining axis, its columns along the second, on every axis (z from the "
          "contiguous transposed copy); no coarse copy under 32 cells; kept; a density opens from rho/rho_bar = 1 to "
          "its 99.9th percentile, the cells above 1 counted",
          oriented_ok and VOL.prepare(avg, True) is prep and lo_d == 0.0
          and abs(hi_d - np.percentile(s, 99.9, method="inverted_cdf")) <= step
          and abs(frac - float((avg["values"] >= 1.0).mean())) < 0.02,
          f"{oriented_ok} {lo_d} {hi_d} {np.percentile(s, 99.9, method='inverted_cdf')} {frac}")
    big = np.exp(np.random.default_rng(2).normal(size=(70, 66, 40))).astype(np.float32)
    big[3, 4, 5] = np.nan
    pb = VOL.prepare({"values": big, "lo": np.zeros(3), "hi": np.array([70.0, 66.0, 40.0]), "cells": big.shape,
                      "key": None}, True)
    ce = [VOL.block_edges(m_, (m_ + 1) // 2) for m_ in big.shape]
    with np.errstate(invalid="ignore"):
        coarse_ref = _block_mean_ref(big.astype(np.float64), ce)
    qc_ref = VOL.quantize(VOL.scaled(coarse_ref, True, float(np.nanmin(big[big > 0]))), pb["qrange"])[0]
    check("11. ... the coarse copy drawn while turning: block means to at most 256 cells a side and at least halved, "
          "on the same levels; a NaN cell is level 0 (transparent) in both",
          pb["coarse"]["shape"] == (35, 33, 20) and np.array_equal(pb["coarse"]["stacks"][0], qc_ref)
          and pb["fine"]["q"][3, 4, 5] == 0 and pb["coarse"]["stacks"][0][1, 2, 2] == 0
          and int(pb["counts"].sum()) == big.size, f"{pb['coarse']['shape']}")
    stop = {"cancel": True}
    try:
        VOL.prepare(dict(avg, key=("a key of its own",)), True, stop)
        coloured = True
    except VOL.Cancelled:
        coloured = False
    now = VOL.prepare(VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, 7), True)
    kept = VOL.prepared_hit(o, "density", "value", o.lo, o.lo + o.length, 7, True)
    check("11. a cancel stops the colouring too (between its chunks), nothing of it kept; a prepared cube is found "
          "again by its window, field and detail without averaging (prepared_hit), not another detail's",
          not coloured and ("a key of its own",) not in {k[0] for k in VOL._PREPARED} and kept is now
          and VOL.prepared_hit(o, "density", "value", o.lo, o.lo + o.length, 5, True) is None, f"{coloured}")
    # the transfer function
    colours = G.lut("inferno")
    lv = VOL.level_values((-2.0, 3.0))
    px = VOL.transfer(lv, 0.0, 2.0, 0.4, 1.0, colours).view(np.uint16).reshape(-1, 4) / 65535.0
    a1 = px[:, 3]
    check("11. transfer: transparent below the low end and for no value, the opacity rising (never falling) to the "
          "one asked at the high end and staying there; premultiplied (no channel above the opacity)",
          np.all(a1[lv < 0.0] == 0.0) and a1[0] == 0.0 and np.all(np.diff(a1[1:]) >= -1e-12) and a1[lv > 0.0].min() > 0
          and np.allclose(a1[lv >= 2.0], 0.4, atol=1e-4) and np.all(px[:, :3] <= a1[:, None] + 1e-9))
    a2 = VOL.transfer(lv, 0.0, 2.0, 0.4, 2.5, colours).view(np.uint16).reshape(-1, 4)[:, 3] / 65535.0
    sym = VOL.transfer(np.array([-np.inf, -3.0, -1.0, 0.0, 1.0, 3.0]), 0.5, 2.0, 0.4, 1.0, G.lut("RdBu_r"),
                       symmetric=True).view(np.uint16).reshape(-1, 4)
    check("11. ... a slice's opacity corrected for the path through it: 1 - (1 - alpha)^(path / reference); a "
          "diverging field by |value|: both tails alike, the middle transparent, the colours apart",
          np.allclose(a2, 1.0 - (1.0 - a1) ** 2.5, atol=1e-4) and sym[0, 3] == 0 and sym[3, 3] == 0
          and sym[1, 3] == sym[5, 3] > 0 and sym[2, 3] == sym[4, 3] > 0 and not np.array_equal(sym[1, :3], sym[5, :3]))
    # the slices on the screen, either projection; in perspective a part for each axis and sense the rays run along
    rng = np.random.default_rng(5)
    lo3, hi3, shape = np.zeros(3), np.array([12.0, 16.0, 10.0]), (6, 8, 5)
    d = (hi3 - lo3) / np.array(shape)
    plans_ok, axes_ok, order_ok, first_ok = True, True, True, True
    rays, deep_rays, pts, faded, inside, most = 0, 0, 0, 0, 0, 0
    for persp, deep in ((False, False), (True, False), (True, True)):
        for yaw, pitch in [(-50.0, 33.0)] + [tuple(rng.uniform(-89, 89, 2)) for _ in range(12)]:
            cam = V.Camera(target=(6, 8, 5), scale=9.0, yaw=yaw, pitch=pitch, width=640, height=480,
                           perspective=persp)
            cam.bounds = (lo3, hi3)
            cam.fit(lo3, hi3)
            for _ in range(int(rng.integers(10, 26)) if deep else 0):     # zoomed in: the eye in or at the window
                cam.zoom(2.0 ** 0.25, *rng.uniform((0.0, 0.0), (640.0, 480.0)))
            plan = VOL.slice_plan(cam, lo3, hi3, shape)
            a, t = plan["axis"], cam.toward()
            ax, ray = VOL.slice_axis(cam, lo3, hi3, shape)
            parts = {(int(q_["axis"]), int(q_["sign"])): q_ for q_ in plan["parts"]}
            mid = parts.get((a, 1 if ray[a] > 0 else -1))
            ec = (cam.eye() - lo3) / d if persp else None           # the eye, in cells from the window's corner
            inside += int(persp and bool(np.all((ec > 0) & (ec < np.array(shape)))))
            most = max(most, len(parts))
            axes_ok &= a == ax and len(parts) == len(plan["parts"]) \
                and bool(np.isclose(plan["exponent"], d[a] / abs(ray[a]) / (16.0 / VOL.REF_CELLS))) \
                and (a == int(np.argmax(np.abs(t) / d)) and 1 <= len(parts) <= 6 if persp
                     else a == int(np.argmax(np.abs(t))) and len(parts) == 1) \
                and (plan["order"] is mid["order"] and plan["transforms"] is mid["transforms"]
                     and plan["size"] == mid["size"] if mid is not None else deep and len(plan["order"]) == 0)
            for (b, sign), part in parts.items():
                r0, r1 = V.rest_axes(b)
                w, h = part["size"]
                order = [int(i) for i in part["order"]]
                axes_ok &= (w, h) == (shape[r1], shape[r0]) and len(part["transforms"]) == shape[b]
                # a slice's image lands where its 3D points project (those in front of the eye)
                for i in sorted({order[0], order[len(order) // 2], order[-1]}) if order else ():
                    T = part["transforms"][i]
                    for x, y in ((0, 0), (w, 0), (0, h), (w, h), (0.3 * w, 0.7 * h)):
                        P = np.zeros(3)
                        P[b] = lo3[b] + (i + 0.5) * d[b]
                        P[r1] = lo3[r1] + x * d[r1]         # columns along the second remaining axis
                        P[r0] = lo3[r0] + y * d[r0]         # rows along the first
                        xy, _ = cam.project(P)
                        if not np.all(np.isfinite(xy[0])):  # behind the eye: Qt cuts the image there
                            continue
                        wt = T[2] * x + T[5] * y + T[8]
                        got = ((T[0] * x + T[3] * y + T[6]) / wt, (T[1] * x + T[4] * y + T[7]) / wt)
                        plans_ok &= bool(wt > 0 and np.allclose(got, xy[0], rtol=1e-5, atol=1e-6))
                        pts += 1
                # the nearest slice ahead of the eye, by its distance in slices; the others in order beyond it
                if not persp:
                    first_ok &= part["first"] is None
                    continue
                k0, wgt = part["first"]
                gaps = sign * (np.arange(shape[b]) + 0.5 - ec[b])
                gap = float(gaps[gaps > 0].min())
                first_ok &= bool(np.isclose(gaps[k0], gap) and np.isclose(wgt, min(1.0, gap))
                                 and (not order or sign * (order[0] - k0) >= 0)
                                 and all(sign * (j - i) > 0 for i, j in zip(order, order[1:])))
                faded += int(deep and wgt < 1.0)
            # front first: a ray (from the eye, or along the view) meets every slice of its part that it crosses
            # in the window, in the part's order
            for X in rng.uniform(lo3, hi3, (200 if deep else 25, 3)):   # few of them are in a close view
                if persp:
                    xy, _ = cam.project(X)
                    if not (np.all(np.isfinite(xy[0])) and 0 <= xy[0][0] <= 640 and 0 <= xy[0][1] <= 480):
                        continue                            # not in the view
                    o_ = cam.eye()
                    dvec = X - o_
                    b = int(np.argmax(np.abs(dvec / d)))    # the axis this ray crosses most slices of
                    part = parts.get((b, 1 if dvec[b] > 0 else -1))
                else:
                    o_ = X + 1e3 * t
                    dvec, b, part = X - o_, a, plan["parts"][0]
                if abs(dvec[b]) < 1e-9:
                    continue
                r0, r1 = V.rest_axes(b)
                rank = {} if part is None else {int(i): n for n, i in enumerate(part["order"])}
                s_ = (lo3[b] + (np.arange(shape[b]) + 0.5) * d[b] - o_[b]) / dvec[b]
                hit = [int(i) for i in np.argsort(s_) if s_[i] > 0
                       and all(lo3[r] <= o_[r] + s_[i] * dvec[r] <= hi3[r] for r in (r0, r1))]
                order_ok &= all(i in rank for i in hit) and all(rank[i] < rank[j] for i, j in zip(hit, hit[1:]))
                rays += bool(hit)
                deep_rays += bool(hit) and deep
    check("11. slice_plan: orthographic, one part across the axis facing the viewer most; perspective, a part for "
          "each axis and sense with slices ahead of the eye (all six with the eye in the window; a pixel draws from "
          "the one its ray runs along most), the middle of the view's across the axis its ray crosses most slices "
          "of; a slice's image (rows along the first remaining "
          "axis) lands where its 3D points in front of the eye project, corners and inside (homographies); every ray "
          "meets all the slices of its part that it crosses, front first; the nearest slice ahead of the eye counts "
          "by its distance in slices; the path through a slice its spacing over |cos|, over the reference",
          plans_ok and axes_ok and order_ok and first_ok and rays > 800 and deep_rays > 150 and pts > 800
          and faded >= 10 and inside >= 5 and most == 6,
          f"{plans_ok} {axes_ok} {order_ok} {first_ok} rays {rays} ({deep_rays} zoomed in) points {pts} "
          f"faded {faded} inside {inside} parts {most}")
    check("11. the reference compositing ('under', front first): an opaque front hides the back; translucent "
          "layers add as c_front + (1 - a_front) c_back",
          np.allclose(VOL.composite([[1, 0, 0, 1], [0, 0, 1, 1]]), [1, 0, 0, 1])
          and np.allclose(VOL.composite([[0.2, 0, 0, 0.5], [0, 0, 0.4, 0.4]]), [0.2, 0, 0.2, 0.7]))


def volume_render():
    """11. the slice stack composited by QPainter, 16 bits a channel, against the reference compositing."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (3D volume rendering): PySide6 is not importable ({e})")
        return
    import explore_3d as X3
    import view3d as V
    import volume3d as VOL
    print("3D volume (rendering)")
    QApplication.instance() or QApplication([])
    m, L, W = 16, 16.0, 400
    # values changing along y only: every ray of a view along y crosses the same samples
    prof = np.linspace(-1.0, 2.0, m)
    vals = np.ascontiguousarray(np.broadcast_to(prof[None, :, None], (m, m, m))).astype(np.float32)
    q, qr = VOL.quantize(VOL.scaled(vals, False))
    stacks, levels = VOL.oriented(q), VOL.level_values(qr)
    cam = V.Camera(target=(8, 8, 8), scale=20.0, yaw=0.0, pitch=0.0, width=W, height=W)       # looking along +y
    plan = VOL.slice_plan(cam, np.zeros(3), np.full(3, L), q.shape)
    lut = VOL.transfer(levels, 0.0, 2.0, 0.5, plan["exponent"], G.lut("viridis"))
    pix = _rgba64(X3.render_slices(lut[stacks[plan["axis"]]], plan, W, W))
    ref = VOL.composite(lut.view(np.uint16).reshape(-1, 4)[q[0, :, 0]].astype(np.float64) / 65535.0)
    check("11. render_slices: a pixel over the cube is the reference compositing of its ray's samples front first "
          "(16-bit channels: within 0.002), a pixel off it transparent",
          plan["axis"] == 1 and np.allclose(pix[200, 200], ref, atol=2e-3) and np.allclose(pix[120, 290], ref, atol=2e-3)
          and pix[5, 5, 3] == 0.0 and ref[3] > 0.3, f"{pix[200, 200]} {ref}")
    # an opaque red half in front of an opaque blue one: the turn decides which is seen
    two = np.where(np.arange(m)[None, :, None] < m // 2, 1.0, 2.0) * np.ones((m, m, m))
    q2, qr2 = VOL.quantize(VOL.scaled(two.astype(np.float32), False))
    rb = np.zeros((256, 3), np.uint8)
    rb[:128] = (255, 0, 0)
    rb[128:] = (0, 0, 255)
    seen = []
    for yaw in (0.0, 180.0):
        cam.yaw = yaw
        plan = VOL.slice_plan(cam, np.zeros(3), np.full(3, L), q2.shape)
        lut = VOL.transfer(VOL.level_values(qr2), 0.5, 2.0, 1.0, plan["exponent"], rb, gamma=1e-9)
        seen.append(_rgba64(X3.render_slices(lut[VOL.oriented(q2)[plan["axis"]]], plan, W, W))[200, 200])
    check("11. ... the front is what is seen: from -y the red half (low y), turned half round the blue one",
          np.allclose(seen[0], [1, 0, 0, 1], atol=2e-3) and np.allclose(seen[1], [0, 0, 1, 1], atol=2e-3), f"{seen}")
    # a dense ball in an empty box: opaque through its middle, transparent where a ray misses it
    m, L = 32, 32.0
    ii = np.indices((m, m, m)) + 0.5
    ball = np.where(((ii - 16.0) ** 2).sum(0) < 8.0 ** 2, 100.0, 0.01).astype(np.float32)
    prep = VOL.prepare({"values": ball, "lo": np.zeros(3), "hi": np.full(3, L), "cells": (m,) * 3, "key": None}, True)
    v3 = X3.View3D()
    v3.resize(W, W)
    v3.set_volume(prep, np.zeros(3), np.full(3, L))
    v3.set_transfer(0.0, 2.0, 1.0, "inferno")
    v3.fit()
    pix = _rgba64(v3.volume_picture())
    cx, cy = (int(c) for c in v3.cam.project([16.0, 16.0, 16.0])[0][0])
    t = v3.cam.toward()
    misses = []
    for c in V.box_corners(np.full(3, 2.5), np.full(3, L - 2.5)):
        r = c - 16.0
        if np.linalg.norm(r - (r @ t) * t) > 11.0:     # its ray passes the ball 3 cells wide of it
            misses.append(tuple(int(k) for k in v3.cam.project(c)[0][0]))
    v3._fast = True
    small = v3.volume_picture()
    v3._fast = False
    check("11. a dense ball in an empty box: opaque through its middle, a ray missing it transparent (the empty "
          "cells below the range), off the box nothing; while turning a half-size picture of the coarse cube",
          pix[cy, cx, 3] > 0.99 and len(misses) >= 2 and all(pix[y, x, 3] == 0.0 for x, y in misses)
          and pix[2, 2, 3] == 0.0 and prep["coarse"]["shape"] == (16, 16, 16) and "quick" in v3._stacks
          and (small.width(), small.height()) == (round(W * X3.FAST_SCALE), round(W * X3.FAST_SCALE)),
          f"{pix[cy, cx]} {misses} {[pix[y, x, 3] for x, y in misses]}")
    # the same through a perspective camera: rays from the eye
    v3.set_perspective(True)
    v3._sharp = None
    pix = _rgba64(v3.volume_picture())
    cx, cy = (int(c) for c in v3.cam.project([16.0, 16.0, 16.0])[0][0])
    e = v3.cam.eye()
    misses = []
    for c in V.box_corners(np.full(3, 2.5), np.full(3, L - 2.5)):
        r, dvec = 16.0 - e, (c - e) / np.linalg.norm(c - e)
        if np.linalg.norm(r - (r @ dvec) * dvec) > 11.0:
            misses.append(tuple(int(k) for k in v3.cam.project(c)[0][0]))
    check("11. ... through a perspective camera: opaque through the ball's middle, a ray from the eye missing it "
          "transparent", pix[cy, cx, 3] > 0.99 and len(misses) >= 2 and all(pix[y, x, 3] == 0.0 for x, y in misses),
          f"{pix[cy, cx]} {misses}")
    # the front half seen in perspective too
    cam = V.Camera(target=(8, 8, 8), scale=20.0, yaw=0.0, pitch=0.0, width=W, height=W, perspective=True)
    cam.bounds = (np.zeros(3), np.full(3, 16.0))
    seen = []
    for yaw in (0.0, 180.0):
        cam.yaw = yaw
        plan = VOL.slice_plan(cam, np.zeros(3), np.full(3, 16.0), q2.shape)
        lut = VOL.transfer(VOL.level_values(qr2), 0.5, 2.0, 1.0, plan["exponent"], rb, gamma=1e-9)
        seen.append(_rgba64(X3.render_slices([lut[s_] for s_ in VOL.oriented(q2)], plan, W, W))[200, 200])
    check("11. ... and the front half is what is seen in perspective as well (red from -y, blue from +y)",
          np.allclose(seen[0], [1, 0, 0, 1], atol=2e-3) and np.allclose(seen[1], [0, 0, 1, 1], atol=2e-3), f"{seen}")
    # the eye inside the cube: each pixel against a march along its own ray (no Qt), a translucent picture
    q3, qr3 = VOL.quantize(VOL.scaled(_gl_field(24, False), True))
    st3, lev3 = VOL.oriented(q3), VOL.level_values(qr3)
    lo3, hi3, W3, H3 = np.zeros(3), np.full(3, 24.0), 320, 240
    worst, n_in, drawn, alphas, kept = 0.0, 0, [], [], 1.0
    for tgt, yaw, pitch, dist in (((12, 12, 12), -35, 25, 6.0), ((9.3, 14.1, 11.2), 60, -40, 4.0),
                                  ((15.2, 8.7, 13.9), -120, 70, 3.0), ((3.1, 20.2, 5.5), 100, -15, 2.0),
                                  ((21.0, 3.3, 18.8), -70, 50, 5.0)):
        cam3 = V.Camera(target=tgt, scale=1.0, yaw=yaw, pitch=pitch, width=W3, height=H3, perspective=True)
        cam3.bounds = (lo3, hi3)
        cam3.scale = cam3.focal / dist                    # the eye 'dist' Mpc behind the target
        plan = VOL.slice_plan(cam3, lo3, hi3, q3.shape)
        lut = VOL.transfer(lev3, float(lev3[40]), float(lev3[230]), 0.04, plan["exponent"], G.lut("inferno"))
        col = [lut[s_] for s_ in st3]
        pix = _rgba64(X3.render_slices(col, plan, W3, H3))
        ref = _ray_march(cam3, lo3, hi3, [c_.view(np.uint16).reshape(c_.shape + (4,)).astype(np.float64) / 65535.0
                                          for c_ in col], W3, H3)
        ok = ~_part_seams(plan, W3, H3) & (_slice_edge_px(cam3, lo3, hi3, q3.shape, plan, W3, H3) >= 1.0 / 32)
        worst = max(worst, float(np.abs(pix - ref).max(axis=2)[ok].max()) * 255.0)
        e3 = cam3.eye()
        n_in += bool(np.all((e3 > lo3) & (e3 < hi3)))
        drawn.append(len(X3.part_regions(plan, W3, H3)))
        alphas.append(float(ref[..., 3].mean()))
        kept = min(kept, float(ok.mean()))
    check("11. the eye inside the cube: QPainter's picture is, pixel by pixel, the march along that pixel's own ray "
          "over the slices of its own part (across the axis the ray runs along most), nearest first, bilinear in "
          "each slice, the nearest by its distance in slices: within 0.25/255 at every pixel not beside another "
          "part's or within 1/32 px of a slice's edge (five eyes in the cube, translucent pictures of one part and "
          "of three)",
          worst <= 0.25 and n_in == 5 and min(drawn) == 1 and max(drawn) == 3 and min(alphas) > 0.15
          and max(alphas) < 0.9 and kept > 0.9,
          f"worst {worst:.4f}/255, {n_in} inside, parts {drawn}, alpha {np.round(alphas, 3)}, kept {kept:.3f}")
    # the sharp picture drawn a few slices at a time is the one drawn at once, at the screen's own resolution
    v3._sharp, v3._job = None, None
    whole = _rgba64(v3.volume_picture())
    job = v3._start_job()
    names, real_gather = set(), job._gather

    def spy_gather(*at):
        names.add(threading.current_thread().name)
        return real_gather(*at)
    job._gather = spy_gather
    ticks = 0
    while not job.pump(0.0005):
        ticks += 1
    check("11. the settled picture drawn in ticks (the window answering between them) equals the one drawn at once; "
          "its colours gathered on the GUI's own volume threads (never queued behind a cube being prepared)",
          ticks >= 1 and np.array_equal(_rgba64(job.img), whole) and names
          and all(n.startswith("volume3d-gui") for n in names), f"{ticks} {names}")
    v3._stop_job()
    # a high-density screen (device ratio 2): the sharp picture has twice the pixels and lands where the box does
    v3.set_perspective(False)
    real_cap = X3.SHARP_CELL_PX
    X3.SHARP_CELL_PX = 1e9                              # no cap: the screen's own ratio
    v3._ratio = lambda: 2.0
    v3._sharp = None
    try:
        sharp = v3.volume_picture()
        ratio = v3._sharp_ratio()
    finally:
        del v3._ratio
        X3.SHARP_CELL_PX = real_cap
    ys, xs = np.nonzero(_rgba64(sharp)[..., 3] > 0.5)
    got = np.array([xs.min(), xs.max() + 1, ys.min(), ys.max() + 1]) / 2.0
    rim = v3.cam.project(16.0 + 8.0 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]))[0]
    want = np.array([rim[:, 0].min(), rim[:, 0].max(), rim[:, 1].min(), rim[:, 1].max()])
    cell = v3.cam.scale * 1.0                            # a cell's size on the screen (the ball's rim is in cells)
    check("11. on a screen of device ratio 2 the sharp picture is drawn at 2x and lands where the projected ball "
          "does, centred on it and within two cells (opaque cells' bilinear edges add up over the slices; drawn "
          "twice its size it would be off by half the ball)", ratio == 2.0 and (sharp.width(), sharp.height()) == (2 * W, 2 * W)
          and np.all(np.abs(got - want) < 2 * cell + 1.0) and np.allclose(got.reshape(2, 2).mean(1), [W / 2, W / 2],
                                                                          atol=2.0), f"{got} {want} cell {cell:.1f}")
    # a large cube (PREVIEW_CELLS along an axis) settles in two steps: half the pixels a side first, then whole
    from PySide6.QtGui import QImage, QPainter
    nb = X3.PREVIEW_CELLS
    rod = np.ones((nb, 32, 32), np.float32)
    rod[:, 8:24, 8:24] = 50.0
    hib = np.array([float(nb), 32.0, 32.0])
    prepb = VOL.prepare({"values": rod, "lo": np.zeros(3), "hi": hib, "cells": rod.shape, "key": None}, True)
    pv = X3.View3D()
    pv.resize(W, W)
    pv._ratio = lambda: 2.0
    pv.set_volume(prepb, np.zeros(3), hib)
    pv.set_transfer(0.0, 2.0, 1.0, "inferno")
    pv.fit()
    screen = QImage(W, W, QImage.Format_RGB32)

    def paint():                                        # what paintEvent does for the volume
        pv._sized()
        p = QPainter(screen)
        pv._draw_volume(p)
        p.end()
    paint()
    first = None if pv._job is None else pv._job.key[0]
    while pv._job is not None and pv._job.key[0] == "preview":
        pv._pump_tick()
    preview, then = pv._preview, pv._job
    paint()                                             # the preview on the screen, the sharp one going on
    kept = pv._job is then and then is not None and then.key[0] == "sharp"
    while pv._job is not None:
        pv._pump_tick()
    full = pv._sharp
    a_prev = _rgba64(preview[1])[..., 3].mean() if preview is not None else -1.0
    a_full = _rgba64(full[1])[..., 3].mean() if full is not None else -2.0
    pv._sharp, pv._preview = None, preview
    saved = pv.volume_picture()
    check("11. a large cube settles in two steps: a picture of half the pixels a side first (shown while the sharp "
          "one is drawn on, which painting does not restart), then the sharp one, the preview then let go; the same "
          "picture at both sizes; a small cube in one step; Save image never the preview",
          first == "preview" and preview is not None and preview[1].width() == W and kept and full is not None
          and full[1].width() == 2 * W and abs(a_prev - a_full) < 0.01 and pv.last_preview_ms > 0
          and saved.width() == 2 * W and pv._preview is None and v3._preview_key() is None,
          f"{first} {None if preview is None else preview[1].width()} {kept} "
          f"{None if full is None else full[1].width()} alpha {a_prev:.4f} {a_full:.4f}")


def _slice_edge_px(cam, lo, hi, shape, plan, W, H) -> np.ndarray:
    """Each pixel centre's distance (px) to the nearest projected edge of a slice the plan draws (the quads of
    every part's slices, exactly: a projective map takes a quad's edges to segments; with the eye in the window,
    the stretch of an edge in front of the eye): where QPainter's 1/64-pixel polygon rasterisation and an exact
    inside test (the GPU's, a ray march's) may disagree on one slice."""
    import view3d as V
    d = (hi - lo) / np.asarray(shape, dtype=np.float64)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    P = np.stack([xx.ravel(), yy.ravel()], 1)
    best = np.full(len(P), np.inf)
    for part in plan["parts"]:
        a = int(part["axis"])
        r0, r1 = V.rest_axes(a)
        for k in part["order"]:
            c = np.zeros((4, 3))
            c[:, a] = lo[a] + (int(k) + 0.5) * d[a]
            c[:, r0] = [lo[r0], hi[r0], hi[r0], lo[r0]]
            c[:, r1] = [lo[r1], lo[r1], hi[r1], hi[r1]]
            for i in range(4):
                seg = V.segment_on_screen(cam, c[i], c[(i + 1) % 4])
                if seg is None:
                    continue
                p0 = np.asarray(seg[0], dtype=np.float64)
                e = np.asarray(seg[1], dtype=np.float64) - p0
                t = np.clip(((P - p0) @ e) / max(float(e @ e), 1e-12), 0.0, 1.0)
                best = np.minimum(best, np.hypot(*(P - (p0 + t[:, None] * e)).T))
    return best.reshape(H, W)


def _part_seams(plan, W, H) -> np.ndarray:
    """True at the pixels beside a pixel of another part of the plan (volume3d.part_ids): where a rounding of a
    ray's direction may give the pixel to either. None are in an orthographic picture."""
    import volume3d as VOL
    ids = VOL.part_ids(plan, W, H)
    out = np.zeros((H, W), dtype=bool)
    if ids is not None:
        dx, dy = ids[:, 1:] != ids[:, :-1], ids[1:] != ids[:-1]
        out[:, 1:] |= dx
        out[:, :-1] |= dx
        out[1:] |= dy
        out[:-1] |= dy
    return out


def _ray_march(cam, lo, hi, rgba, W, H) -> np.ndarray:
    """A perspective view's volume picture ((H, W, 4), premultiplied, 0..1) by the rule alone, without Qt: each
    pixel's ray from the eye through the pixel's centre; the stack across the axis the ray runs along most, in
    cells (ties to the lower axis); its slices ahead of the eye, nearest first, each sampled bilinearly where the
    ray crosses it (nothing outside the image), the nearest one at its distance in slices; composited 'under'.
    rgba: per axis, the (m, h, w, 4) premultiplied colours of volume3d's slice images."""
    import view3d as V
    shape = np.array([rgba[b].shape[0] for b in range(3)])
    cell = (hi - lo) / shape
    e = (cam.eye() - lo) / cell
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    c = cam.target + ((xs - 0.5 * W)[..., None] * cam.R[0] - (ys - 0.5 * H)[..., None] * cam.R[1]) / cam.scale
    D = (c - lo) / cell - e                                 # the rays, in cells
    ax = np.argmax(np.abs(D), axis=2)
    out = np.zeros((H, W, 4))
    for b in range(3):
        r0, r1 = V.rest_axes(b)
        tex = rgba[b]
        m, h, w = tex.shape[:3]
        gap = np.arange(m) + 0.5 - e[b]
        for sign in (1, -1):
            sel = (ax == b) & (sign * D[..., b] > 0)
            if not sel.any():
                continue
            dv, acc = D[sel], np.zeros((int(sel.sum()), 4))
            for n, i in enumerate(np.nonzero(sign * gap > 0)[0][::sign]):
                s = gap[i] / dv[:, b]
                u, v = e[r1] + s * dv[:, r1] - 0.5, e[r0] + s * dv[:, r0] - 0.5      # columns along r1, rows r0
                on = (u > -0.5) & (u < w - 0.5) & (v > -0.5) & (v < h - 0.5)
                u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
                fu, fv = (u - u0)[:, None], (v - v0)[:, None]
                ua, ub = np.clip(u0, 0, w - 1), np.clip(u0 + 1, 0, w - 1)
                va, vb = np.clip(v0, 0, h - 1), np.clip(v0 + 1, 0, h - 1)
                px = ((tex[i][va, ua] * (1 - fu) + tex[i][va, ub] * fu) * (1 - fv)
                      + (tex[i][vb, ua] * (1 - fu) + tex[i][vb, ub] * fu) * fv)
                px = np.where(on[:, None], px, 0.0)
                if n == 0:
                    px = px * min(1.0, sign * gap[i])
                acc += (1.0 - acc[:, 3:4]) * px
            out[sel] = acc
    return out


def _strided_plan(plan, stride) -> dict:
    """A slice plan with every stride-th slice only, as volume_gl takes them while the view turns: those with
    k = (stride - 1) // 2 (mod stride); in perspective the nearest of them ahead of the eye at its distance in
    strides (the stretch it stands for)."""
    phase = (stride - 1) // 2
    parts = []
    for part in plan["parts"]:
        keep = np.array([int(i) for i in part["order"] if int(i) % stride == phase], dtype=np.intp)
        first = part["first"]
        if first is not None:
            gap = part["sign"] * (np.arange(len(part["transforms"])) + 0.5 - plan["view"][3][part["axis"]])
            ahead = [int(k) for k in np.nonzero(gap > 0)[0] if int(k) % stride == phase]
            first = None
            if ahead:
                k = min(ahead, key=lambda j: gap[j])
                first = (k, float(min(1.0, gap[k] / stride)))
        parts.append(dict(part, order=keep, first=first))
    return dict(plan, parts=parts)


def _gl_field(m, spiky, seed=3) -> np.ndarray:
    """A lognormal field: smooth (a Gaussian spectrum) or spiky (a power law to the cell scale)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(m)
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    amp = (kk + 1e-3) ** -1.0 if spiky else np.exp(-(kk * m / 6.0) ** 2)
    g = np.real(np.fft.ifftn(np.fft.fftn(rng.normal(size=(m, m, m))) * amp))
    return np.exp(1.5 * (g - g.mean()) / g.std()).astype(np.float32)


def gl_child():
    """Run by volume_gl_parity in a process of its own on the default Qt platform (the suite's offscreen one has no
    OpenGL): the GPU's volume picture against QPainter's. Prints GL-APP once Qt started, GL-SKIP: <why> without an
    OpenGL context, a GL-PARITY {json} line per case and GL-DONE at its end."""
    from PySide6.QtWidgets import QApplication
    app = QApplication([])                      # noqa: F841 -- kept alive
    print("GL-APP", flush=True)
    import explore_3d as X3
    import view3d as V
    import volume3d as VOL
    import volume_gl as VGL
    r = VGL.renderer()
    if r is None:
        print(f"GL-SKIP: {VGL.unavailable_reason()}", flush=True)
        return
    W, H = 320, 280
    turns = ((-35.0, 25.0), (60.0, -40.0), (170.0, 10.0))
    # orthographic; perspective from the fitted view in through the window's face to an eye within it (wheel notches
    # about the middle of the view, then about a point off it: one on the window at every turn, or the eye would
    # pass beside it)
    views = [(False, yaw, pitch, 0, None) for yaw, pitch in turns] \
        + [(True, yaw, pitch, n, at) for yaw, pitch in turns
           for n, at in ((0, None), (8, None), (14, (100.0, 90.0)), (22, (205.0, 175.0)))]

    def off_by(got, ref, cam, lo, hi, shape, plan, drawn):
        """|got - ref| in 1/255 per pixel; the pixels off the edges of the slices 'drawn' and off the parts' seams;
        the seam pixels off the edges."""
        d = np.abs(got - ref).max(axis=2) * 255.0
        tie = _slice_edge_px(cam, lo, hi, shape, drawn, W, H) < 1.0 / 32
        seam = _part_seams(plan, W, H)
        return d, ~tie & ~seam, seam & ~tie, tie

    for spiky in (False, True):
        shape = (32, 36, 26)                    # no two sides alike: the axes' order is tested too
        vals = np.ascontiguousarray(_gl_field(40, spiky)[:shape[0], :shape[1], :shape[2]])
        lo, hi = np.zeros(3), np.array([32.0, 36.0, 26.0])
        prep = VOL.prepare({"values": vals, "lo": lo, "hi": hi, "cells": shape, "key": None}, True)
        q = prep["fine"]["q"]
        st3 = [VOL.slice_images(prep["fine"], a_) for a_ in range(3)]
        top = 0.8 * float(prep["levels"][-1])
        r.begin_upload(("parity", spiky), q)
        r.upload_step(float("inf"))
        for persp, yaw, pitch, notches, at in views:
            cam = V.Camera(yaw=yaw, pitch=pitch, width=W, height=H, perspective=persp)
            cam.bounds = (lo, hi)
            cam.fit(lo, hi)
            for _ in range(notches):
                cam.zoom(2.0 ** 0.25, *(at or ()))
            plan = VOL.slice_plan(cam, lo, hi, q.shape)
            expo = float(np.round(plan["exponent"], 2))
            lut = VOL.transfer(prep["levels"], 0.0, top, 0.6, expo, G.lut("inferno"))
            ref = _rgba64(X3.render_slices([np.ascontiguousarray(lut[s_]) for s_ in st3], plan, W, H, 1.0))
            gl = r.render(cam, lo, hi, plan, lut, W, H, deep=True).astype(np.float64) / 65535.0
            d, off, seam, tie = off_by(gl, ref, cam, lo, hi, q.shape, plan, plan)
            # every 4th slice (a turning frame): QPainter's picture of the same slices; and the same with and
            # without skipping the empty blocks: not a pixel apart
            lut4 = VOL.transfer(prep["levels"], 0.0, top, 0.6, expo * 4, G.lut("inferno"))
            a4 = r.render(cam, lo, hi, plan, lut4, W, H, stride=4, deep=True)
            keep, VGL.SKIP_FROM_STRIDE = VGL.SKIP_FROM_STRIDE, 10 ** 6
            try:
                b4 = r.render(cam, lo, hi, plan, lut4, W, H, stride=4, deep=True)
            finally:
                VGL.SKIP_FROM_STRIDE = keep
            plan4 = _strided_plan(plan, 4)
            ref4 = _rgba64(X3.render_slices([np.ascontiguousarray(lut4[s_]) for s_ in st3], plan4, W, H, 1.0))
            d4, off4, seam4, _ = off_by(a4.astype(np.float64) / 65535.0, ref4, cam, lo, hi, q.shape, plan, plan4)
            eye = cam.eye()
            drawn = len(X3.part_regions(plan, W, H))            # the parts with pixels in this picture
            print("GL-PARITY " + json.dumps({
                "case": f"{'spiky' if spiky else 'smooth'} {'perspective' if persp else 'orthographic'} "
                        f"yaw {yaw:g} pitch {pitch:g} zoom {notches} ({drawn} parts)",
                "inside": bool(persp and np.all((eye > lo) & (eye < hi))), "parts": drawn,
                "max_off_ties": float(d[off].max()), "p99": float(np.percentile(d, 99)), "mean": float(d.mean()),
                "ties": float(tie.mean()), "covered": float((ref[..., 3] > 0).mean()),
                "seams": int(seam.sum()), "seam_over": int((d[seam] > 1.5).sum()),
                "max4_off_ties": float(d4[off4].max()), "mean4": float(d4.mean()),
                "seam4_over": int((d4[seam4] > 1.5).sum()), "seams4": int(seam4.sum()),
                "skip_same": bool(np.array_equal(a4, b4)), "skipped_any": bool((a4[..., 3] > 0).any())}),
                flush=True)
    # the half cube (level 1, the turning frames'): the GPU's picture of it equals QPainter's of the same half cube
    # and slices (every 2nd), from outside and with the eye in the window
    m = 72                                      # MIP_FROM or more a side: the half is made
    vals = np.ascontiguousarray(_gl_field(m, True))
    lo, hi = np.zeros(3), np.full(3, float(m))
    prep2 = VOL.prepare({"values": vals, "lo": lo, "hi": hi, "cells": (m,) * 3, "key": None}, True)
    r.begin_upload("half", prep2["fine"]["q"], prep2["levels"], prep2["qrange"], prep2["log"])
    r.upload_step(float("inf"))
    half = VGL.mip_levels(prep2["fine"]["q"], prep2["levels"], prep2["qrange"], True)
    sth = VOL.oriented(half)
    res = {"same_levels": bool(r.ready("half", 1) and r.shape(1) == half.shape)}
    for name, notches in (("", 0), ("inside_", 14)):
        cam = V.Camera(yaw=-35.0, pitch=25.0, width=W, height=H, perspective=True)
        cam.bounds = (lo, hi)
        cam.fit(lo, hi)
        for _ in range(notches):
            cam.zoom(2.0 ** 0.25, 100.0, 90.0)
        plan = VOL.slice_plan(cam, lo, hi, half.shape)
        lut = VOL.transfer(prep2["levels"], 0.0, 0.8 * float(prep2["levels"][-1]), 0.6,
                           float(np.round(plan["exponent"], 2)) * 2, G.lut("inferno"))
        gl = r.render(cam, lo, hi, plan, lut, W, H, stride=2, deep=True, level=1).astype(np.float64) / 65535.0
        plan2 = _strided_plan(plan, 2)          # QPainter's every 2nd slice, the shader's
        ref_s2 = _rgba64(X3.render_slices([np.ascontiguousarray(lut[s_]) for s_ in sth], plan2, W, H, 1.0))
        d, off, seam, _ = off_by(gl, ref_s2, cam, lo, hi, half.shape, plan, plan2)
        eye = cam.eye()
        res.update({name + "max_off_ties": float(d[off].max()), name + "mean": float(d.mean()),
                    name + "covered": float((ref_s2[..., 3] > 0).mean()),
                    name + "seam_over": int((d[seam] > 1.5).sum()), name + "seams": int(seam.sum()),
                    name + "eye_in": bool(np.all((eye > lo) & (eye < hi))),
                    name + "parts": len(X3.part_regions(plan, W, H))})
    print("GL-HALF " + json.dumps(res), flush=True)
    r.forget_cube()
    # the window's own picture: View3D's GPU frame against its volume_picture() (8 bits a channel on screen), the
    # fitted view and one zoomed into the window
    lo, hi = np.zeros(3), np.array([32.0, 36.0, 26.0])
    v3 = X3.View3D()
    v3.resize(W, H)
    v3._ratio = lambda: 1.0
    keep_cap, X3.SHARP_CELL_PX = X3.SHARP_CELL_PX, 1e9
    try:
        v3.set_volume(prep, lo, hi)
        v3.set_transfer(0.0, 0.8 * float(prep["levels"][-1]), 0.6, "viridis")
        v3.set_perspective(True)

        def window_off():
            """The window's GPU frame against its volume_picture(), as the view stands."""
            tries = 0
            while v3._gl_frame() is False and tries < 1000:
                VGL.renderer().upload_step(float("inf"))
                tries += 1
            img = v3._gl_frame()
            px = VGL.to_qimage(np.ascontiguousarray(img.px)).convertToFormat(X3.QImage.Format_RGBA64_Premultiplied)
            plan = VOL.slice_plan(v3.cam, lo, hi, prep["fine"]["shape"])
            d, off, seam, _ = off_by(_rgba64(px), _rgba64(v3.volume_picture()), v3.cam, lo, hi,
                                     prep["fine"]["shape"], plan, plan)
            return img, d, off, seam, plan

        img, d, off, seam, _ = window_off()
        win = {"size": [img.width(), img.height()], "max_off_ties": float(d[off].max()), "mean": float(d.mean()),
               "note": v3.gl_note}
        for _ in range(14):                     # the wheel, into the window
            v3.cam.zoom(2.0 ** 0.25, 100.0, 90.0)
        v3._fast = False
        img, d, off, seam, plan = window_off()
        eye = v3.cam.eye()
        win.update({"inside_max_off_ties": float(d[off].max()), "inside_mean": float(d.mean()),
                    "inside_seam_over": int((d[seam] > 1.5).sum()), "inside_seams": int(seam.sum()),
                    "eye_in": bool(np.all((eye > lo) & (eye < hi))),
                    "inside_parts": len(X3.part_regions(plan, W, H)),
                    "inside_note": v3.gl_note})
        # a range or opacity slider dragged: the turning frames until it rests (a settled frame each move is slow)
        v3._fast = False
        v3.set_transfer(0.1, 0.8 * float(prep["levels"][-1]), 0.5, "viridis")
        slider = v3._fast and v3._settle.isActive() and v3._gl_frame() is not None and v3._gl_step is not None
        win["slider_turns"] = bool(slider)
        print("GL-WINDOW " + json.dumps(win), flush=True)
    finally:
        X3.SHARP_CELL_PX = keep_cap
        v3.set_volume(None)
    print("GL-DONE", flush=True)


def volume_gl_parity():
    """11. the GPU's picture (volume_gl) against QPainter's, in a child process on the default platform (gl_child);
    SKIP where Qt does not start there (no display) or no OpenGL context can be made."""
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    env.pop("DTFE_GUI_NO_GL", None)
    env["QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM"] = "1"   # no Dock icon for the child on macOS
    try:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--gl-child"], env=env,
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        check("11. GPU parity: the child process ends", False, "timed out after 600 s")
        return
    out = p.stdout
    if "GL-APP" not in out:
        print(f"SKIP (GPU volume parity): Qt's default platform did not start here (no display?): "
              f"{(p.stderr.strip().splitlines() or [''])[-1][:160]}")
        return
    skip = next((ln for ln in out.splitlines() if ln.startswith("GL-SKIP")), None)
    if skip is not None:
        print(f"SKIP (GPU volume parity): {skip[len('GL-SKIP: '):]}")
        return
    print("3D volume on the GPU (a child process on the default platform)")
    cases = [json.loads(ln[len("GL-PARITY "):]) for ln in out.splitlines() if ln.startswith("GL-PARITY ")]
    win = next((json.loads(ln[len("GL-WINDOW "):]) for ln in out.splitlines() if ln.startswith("GL-WINDOW ")), None)
    done = "GL-DONE" in out and p.returncode == 0
    check("11. GPU parity: the child ran every case to its end", done and len(cases) == 30 and win is not None
          and "GL-HALF " in out,
          f"exit {p.returncode}, {len(cases)} cases; {p.stderr.strip()[-600:]}")
    if not cases:
        return
    worst = max(cases, key=lambda c: c["max_off_ties"])
    n_in = sum(c["inside"] for c in cases)
    check("11. the GPU's picture equals QPainter's within 1.5/255 at every pixel not within 1/32 px of a slice's "
          "projected edge nor beside another part's pixels (30 cases: smooth and spiky fields; orthographic, three "
          "turns; perspective, three turns, each from the fitted view in through the window's face to an eye inside "
          "it, one to three parts in a picture); the pixels beside another part's as well, but for 1 in 100 of them; "
          "over all "
          "pixels p99 <= 1/255 and the mean <= 0.1/255",
          all(c["max_off_ties"] <= 1.5 and c["p99"] <= 1.0 and c["mean"] <= 0.1 and c["covered"] > 0.2
              and c["ties"] < 0.1 and c["seam_over"] <= 0.01 * c["seams"] + 2 for c in cases)
          and n_in == 18 and max(c["parts"] for c in cases) == 3
          and sum(c["seams"] > 100 for c in cases) >= 12,
          f"{n_in} inside; worst {worst}; " + "; ".join(f"{c['case']}: {c['max_off_ties']:.2f}" for c in cases))
    worst4 = max(cases, key=lambda c: c["max4_off_ties"])
    check("11. ... every 4th slice (a turning frame's stride; in perspective the nearest of them ahead of the eye "
          "by its distance in strides) is QPainter's picture of the same slices within 1.5/255 off the slices' "
          "edges and the parts' seams, and skipping the empty blocks changes not a pixel",
          all(c["skip_same"] and c["skipped_any"] and c["max4_off_ties"] <= 1.5 and c["mean4"] <= 0.1
              and c["seam4_over"] <= 0.01 * c["seams4"] + 2 for c in cases),
          f"worst {worst4}; skip differs: {[c['case'] for c in cases if not c['skip_same']]}")
    half = next((json.loads(ln[len("GL-HALF "):]) for ln in out.splitlines() if ln.startswith("GL-HALF ")), None)
    check("11. ... the cube's half (drawn while turning, every 2nd slice of it) is the GPU's too: QPainter's picture "
          "of the same half cube and slices within 1.5/255 off the slices' edges, from outside and with the eye in "
          "the window",
          half is not None and half["same_levels"] and half["max_off_ties"] <= 1.5 and half["covered"] > 0.2
          and half["inside_max_off_ties"] <= 1.5 and half["inside_eye_in"] and half["inside_covered"] > 0.9
          and half["inside_seam_over"] <= 0.01 * half["inside_seams"] + 2,
          str(half))
    if win is not None:
        check("11. ... and in the window: View3D's GPU frame (8 bits) is its volume_picture() within 1.5/255 off the "
              "slices' edges, the fitted view and one zoomed into the window (the eye inside)",
              win["max_off_ties"] <= 1.5 and win["mean"] <= 0.2 and win["note"] == ""
              and win["size"] == [320, 280] and win["eye_in"] and win["inside_max_off_ties"] <= 1.5
              and win["inside_mean"] <= 0.45 and win["inside_note"] == ""
              and win["inside_seam_over"] <= 0.01 * win["inside_seams"] + 2, str(win))
        check("11. ... a slider moved while the GPU draws: its turning frames until the view rests, not a settled "
              "frame per move", win.get("slider_turns") is True, str(win))


def volume_gl_offscreen():
    """11. offscreen (this suite's platform) there is no OpenGL: QPainter draws the volume, the state line does not
    claim the GPU, and the turning policy keeps its contract."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (3D volume, GPU fallback): PySide6 is not importable ({e})")
        return
    import explore_3d as X3
    import volume3d as VOL
    import volume_gl as VGL
    print("3D volume: the GPU's fallback")
    QApplication.instance() or QApplication([])
    m = 16
    vals = np.exp(np.random.default_rng(5).normal(size=(m, m, m))).astype(np.float32)
    prep = VOL.prepare({"values": vals, "lo": np.zeros(3), "hi": np.full(3, 16.0), "cells": (m,) * 3, "key": None},
                       True)
    v3 = X3.View3D()
    v3.resize(200, 200)
    v3.set_volume(prep, np.zeros(3), np.full(3, 16.0))
    v3.set_transfer(0.0, 1.0, 0.5, "viridis")
    v3.fit()
    frame = v3._gl_frame()
    pic = v3.volume_picture()
    check("11. offscreen: no OpenGL (the renderer is None, the reason names the platform), View3D's GPU frame None "
          "and QPainter's picture drawn instead",
          VGL.renderer() is None and "offscreen" in VGL.unavailable_reason() and frame is None
          and "offscreen" in v3.gl_note and pic is not None and v3.gl_bytes() == 0,
          f"{VGL.unavailable_reason()!r} {frame} {v3.gl_note!r}")
    os.environ["DTFE_GUI_NO_GL"] = "1"
    try:
        check("11. DTFE_GUI_NO_GL set: OpenGL is not even tried", "DTFE_GUI_NO_GL" in VGL.disabled_reason())
    finally:
        del os.environ["DTFE_GUI_NO_GL"]
    # any failure setting OpenGL up (not only a refused context: a missing binding, a driver's error) leaves QPainter

    class Broken:
        def __init__(self):
            raise RuntimeError("a driver's error")
    keep = (VGL.disabled_reason, VGL.Renderer, dict(VGL._STATE))
    VGL.disabled_reason, VGL.Renderer = (lambda: ""), Broken
    VGL._STATE.clear()
    try:
        frame = v3._gl_frame()
        check("11. an error while OpenGL is set up (not only a refused context) leaves QPainter drawing, the reason "
              "said", VGL.renderer() is None and frame is None and "RuntimeError" in v3.gl_note
              and v3.volume_picture() is not None, f"{frame} {v3.gl_note!r}")
    finally:
        VGL.disabled_reason, VGL.Renderer = keep[0], keep[1]
        VGL._STATE.clear()
        VGL._STATE.update(keep[2])
    v3.set_volume(None)
    # the turning policy's contract (whatever its rule): a step of TURN_STEPS; a slow frame never makes it finer, a
    # fast one never coarser; a finer step known too dear is not tried again; nothing known, nothing changes
    n, NT = len(X3.TURN_STEPS), X3.next_turn_step
    ok, seen = True, []
    for step in range(n):
        slow, fast = NT(step, {step: 999.0}), NT(step, {step: 0.01})
        ok &= 0 <= slow < n and 0 <= fast < n and slow >= step and fast <= step and NT(step, {}) == step
        if step > 0:
            ok &= NT(step, {step: 0.01, step - 1: 999.0}) >= step
        seen.append((step, slow, fast))
    check("11. the turning policy (next_turn_step): a step of TURN_STEPS, finer first; slow frames never make it "
          "finer, fast ones never coarser, a finer step known too dear is not tried again; coarser from the finest "
          "when slow, finer from the coarsest when fast; the ladder's depth between samples grows",
          ok and NT(0, {0: 999.0}) > 0 and NT(n - 1, {n - 1: 0.01}) < n - 1
          and all((2 if a else 1) * b < (2 if c else 1) * d or (a, b) == (0, 2) for (a, b), (c, d)
                  in zip(X3.TURN_STEPS, X3.TURN_STEPS[1:])), str(seen))
    # the cube's half (the turning frames' cube): 2^3 cells averaged in value space, quantised on the same range
    rng = np.random.default_rng(2)
    good = True
    for log in (True, False):
        vals = np.exp(2.0 * rng.normal(size=(9, 12, 7))).astype(np.float32) - (0.0 if log else 1.0)
        vals[3, 4, 5] = np.nan
        prep = VOL.prepare({"values": vals.astype(np.float32), "lo": np.zeros(3), "hi": np.ones(3),
                            "cells": vals.shape, "key": None}, log)
        q, lv, qr = prep["fine"]["q"], prep["levels"], prep["qrange"]
        got = VGL.mip_levels(q, lv, qr, log)
        vt = VGL.value_table(lv, log)
        e = [VOL.block_edges(mm, (mm + 1) // 2) for mm in q.shape]
        ref = np.zeros(VGL.mip_shape(q.shape))
        for i, j, k in np.ndindex(ref.shape):
            ref[i, j, k] = vt[q[e[0][i]:e[0][i + 1], e[1][j]:e[1][j + 1], e[2][k]:e[2][k + 1]]].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            want = VOL.quantize(np.log10(ref) if log else ref, qr)[0]
        good &= got.shape == (5, 6, 4) and np.array_equal(got, want) and int((got == 0).sum()) == 1
    check("11. the cube's half drawn while turning: each 2^3 block's mean value (a log scale's 10^level: a dense "
          "cell keeps its mass), quantised on the cube's own range; a block with a cell without value has none",
          good)


def view3d_volume_gui():
    """11. the volume in the window, on gui()'s synthetic snapshot (its catalogue is cached by then)."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (3D volume): PySide6 is not importable ({e})")
        return
    import explore as E
    import explore_3d as X3
    import runspec as rs
    import view3d as V
    import volume3d as VOL
    print("3D volume")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "data"
    if not (data / SIM).is_dir():
        make_snapdir(data, 50, 0.0, 7)
    E.VOID_CACHE_DIR = TMP / "void cache"
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(900, 760)
    view.show()
    real_procs = rs.dtfe_processes
    rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}
    try:
        x.refresh_outputs()
        wait_for(qa, lambda: not x._busy and x._plane is not None, 30)
        x.voids.setChecked(True)
        wait_for(qa, lambda: x._void_proc is None and bool(x._void_cache), 180)
        res = x._void_cache.get(x._void_cache_key(x.current), {})
        if "error" in res or not res:
            check("11. the catalogue for the 3D volume", False, str(res.get("error", ""))[:200])
            return
        o, v3 = x.current, view.view3d
        x.view3d_style.setCurrentIndex(x.view3d_style.findData("volume"))
        VOL.drop_volumes()
        x.view3d_toggle.setChecked(True)
        ok = wait_for(qa, lambda: (x._3d_res or {}).get("volume") is not None and not x._3d_busy, 60)
        qa.processEvents()
        v3.repaint()
        prep = x._3d_res["volume"]
        cube = np.fromfile(o.files["density"].path, dtype=o.dtype).reshape((o.n,) * 3)
        avg = VOL.average_window(o, "density", "value", o.lo, o.lo + o.length, VOL.DEFAULT_DETAIL)
        cached = G._cached_cube(o.files["density"].path, o.dtype, (o.n,) * 3, read=False)
        check("11. Show in 3D: a volume of the whole box at the grid's own 64³ cells (the detail is more), the stored "
              "values (the map's cached cube itself, no copy); the title and the state line say so; the range's "
              "controls shown, the planes' hidden",
              ok and prep["shape"] == (64, 64, 64) and np.array_equal(avg["values"], cube)
              and (cached is None or avg["values"] is cached) and "a volume of 64³ cells" in view.title.text()
              and "the grid's own cells" in x.view3d_state.text() and x.volume3d_box.isVisibleTo(x)
              and not x.planes3d_box.isVisibleTo(x) and v3.style == "volume" and v3.volume is prep,
              f"{prep['shape']} {view.title.text()} {x.view3d_state.text()}")
        tf = x._3d_tf
        check("11. ... a density opens from rho/rho_bar = 1 to its 99.9th percentile, in the map's colours; the colour "
              "bar shows that range, the sliders' texts its ends", tf["lo"] == 0.0 and tf["hi"] > 0.0
              and x.vol3d_lo_text.text() == "1" and view.colorbar.vrange == (tf["lo"], tf["hi"]) and view.colorbar.log
              and v3.transfer[:2] == (tf["lo"], tf["hi"]) and v3.transfer[3] == x.cmap.currentText(), f"{tf}")
        # the voids see-through: looking along y, the ray through the biggest void against the densest column
        rows = sorted(int(k) for k in np.nonzero(res["_shown"] & np.isfinite(res["_r_v"]))[0])
        k0 = max(rows, key=lambda k: res["_r_v"][k])
        a, b = x._3d_span()
        x.vol3d_lo.setValue(int(round((np.log10(0.7) - a) / (b - a) * x.vol3d_lo.maximum())))
        x.vol3d_opacity.setValue(x._3d_opacity_position(0.01))
        x.view3d_persp.setChecked(False)                    # parallel rays: one grid column a pixel
        v3.cam.yaw, v3.cam.pitch = 0.0, 0.0
        v3.fit()
        wait_for(qa, lambda: not v3._fast, 10)              # settled: the full cube on the full picture
        pic = _rgba64(v3.volume_picture())
        tfv = v3.transfer
        plan = VOL.slice_plan(v3.cam, v3.lo, v3.hi, prep["shape"])
        lut = VOL.transfer(prep["levels"], tfv[0], tfv[1], tfv[2], round(plan["exponent"], 2), G.lut(tfv[3]))
        rgba = lut.view(np.uint16).reshape(-1, 4).astype(np.float64) / 65535.0
        qcube = prep["fine"]["q"]                           # the levels, (x, y, z)
        visible = (cube >= 0.7).sum(axis=1)                 # per (x, z) column
        c = res["_centres"][k0]
        void_col = (int(c[0]) % o.n, int(c[2]) % o.n)
        bg_col = np.unravel_index(int(np.argmax(visible)), visible.shape)

        def ray(col):
            xy = v3.cam.project([col[0] + 0.5, 32.0, col[1] + 0.5])[0][0]
            return pic[int(xy[1]), int(xy[0]), 3], VOL.composite(rgba[qcube[col[0], :, col[1]]])[3]
        (a_void, r_void), (a_bg, r_bg) = ray(void_col), ray(bg_col)
        check("11. the voids see-through: looking along y, the ray through the biggest void's centre is clearer than "
              "the densest column (its cells below the range transparent), each as the reference compositing says",
              visible[void_col] + 8 < visible[bg_col] and a_void < a_bg - 0.05 and abs(a_void - r_void) < 0.06
              and abs(a_bg - r_bg) < 0.06,
              f"void {void_col} {visible[void_col]} cells: {a_void:.3f} (ref {r_void:.3f}); "
              f"column {bg_col} {visible[bg_col]}: {a_bg:.3f} (ref {r_bg:.3f})")
        x.view3d_persp.setChecked(True)
        # the sharp picture after a turn: drawn in ticks once the view rests, at the screen's resolution
        v3.last_volume_ms = 0.0
        v3.cam.yaw += 7.0
        v3._interacting()
        v3.repaint()
        quick_first = v3._picture is not None and v3._picture[0][0] == "quick" \
            and (v3._sharp is None or v3._sharp[0] != v3._sharp_key())
        ok = wait_for(qa, lambda: not v3._fast and v3._job is None and v3._sharp is not None
                      and v3._sharp[0] == v3._sharp_key(), 30)
        check("11. after a turn the quick picture first, then the sharp one drawn in ticks once the view rests (at the "
              "sharp ratio: the screen's, no finer than 3 device pixels a cell); perspective back on",
              ok and quick_first and v3._sharp[1].width() == int(round(v3.width() * round(v3._sharp_ratio(), 2)))
              and v3.cam.perspective and v3.last_volume_ms > 0, f"{quick_first} {v3.last_volume_ms}")
        # the opacity slider moves the opacity only: a range between the sliders' steps stays exactly as it is
        x._3d_tf.update(lo=0.0123456, hi=0.0234567)
        x._3d_apply_tf()
        x.vol3d_opacity.setValue(x.vol3d_opacity.value() - 3)
        check("11. the opacity slider leaves the shown range exactly as it is (only a range slider moves its end)",
              x._3d_tf["lo"] == 0.0123456 and x._3d_tf["hi"] == 0.0234567 and v3.transfer[:2] == (0.0123456, 0.0234567))
        x._3d_tf_default()
        # the requests: a cancelled one in flight is no longer counted as coming; a moot result is not shown
        shown = x._3d_res
        x._3d_request()                                    # nothing new: the shown volume stays, nothing read
        same = x._3d_res is shown and not x._3d_busy
        vkey = shown["vkey"]
        x._3d_busy, x._3d_inflight, x._3d_pending = True, {"vkey": ("another",), "progress": {}, "style": "volume"}, None
        x._3d_res = None                                   # as if nothing were shown yet
        x._3d_inflight = {"vkey": vkey, "progress": {"cancel": True}, "style": "volume"}
        x._3d_request()
        queued = x._3d_pending is not None and x._3d_pending["vkey"] == vkey
        x._3d_busy, x._3d_inflight, x._3d_pending, x._3d_res = False, None, None, shown
        moot = dict(shown, request=dict(shown["request"], progress={"cancel": True}), volume={"moot": True})
        x._3d_rendered(moot)
        check("11. requests: the volume shown is not read again; asked again while its cancelled reading still runs, it "
              "is queued (a cancelled one never arrives); a result made moot while coloured is not shown",
              same and queued and x._3d_res is shown and v3.volume is shown["volume"])
        # the colouring stage says so even with nothing averaged
        x._3d_busy, x._3d_inflight = True, {"vkey": ("x",), "progress": {"stage": "colouring"}, "style": "volume",
                                            "output": o}
        x._3d_progress()
        colouring = x.view3d_state.text().startswith("colouring the volume")
        x._3d_busy, x._3d_inflight = False, None
        x._3d_state()
        check("11. the state line says 'colouring the volume' while a cube is coloured (also when nothing was "
              "averaged: a cached or prepared cube)", colouring)
        # the sliders: the low end never past the high one; Default range back
        x.vol3d_lo.setValue(x.vol3d_hi.value() + 10)
        pushed = x.vol3d_hi.value() == x.vol3d_lo.value() + 1 and x._3d_tf["lo"] < x._3d_tf["hi"] \
            and v3.transfer[:2] == (x._3d_tf["lo"], x._3d_tf["hi"])
        x.vol3d_default.click()
        check("11. the range's low end pushed past the high end moves it along; Default range restores the range and "
              "the opacity", pushed and x._3d_tf["lo"] == 0.0 and abs(v3.transfer[2] - VOL.DEFAULT_OPACITY) < 0.02)
        # another colour map recolours, reads nothing
        shown = x._3d_res
        x.cmap.setCurrentText("viridis")
        wait_for(qa, lambda: not x._busy and not x._3d_busy, 30)
        check("11. another colour map recolours the volume without reading it again",
              x._3d_res is shown and v3.transfer[3] == "viridis" and view.colorbar.cmap == "viridis")
        # hover still names the voids, drawn over the volume
        wait_for(qa, lambda: not v3._fast, 10)
        v3.repaint()
        circ = {cc[4]: (cc[0], cc[1], float(np.hypot(*cc[2]))) for cc in v3._outlines}
        cx, cy, r = circ[k0]
        spot = next(((cx + f * r * np.cos(t), cy + f * r * np.sin(t)) for f in (0.0, 0.5, 0.8)
                     for t in np.linspace(0, 2 * np.pi, 12) if v3.sphere_at(cx + f * r * np.cos(t),
                                                                               cy + f * r * np.sin(t)) == k0), None)
        if spot is not None:
            x._3d_hover(*spot)
        check("11. the voids drawn over the volume (every distinct one); hover over one names it",
              sorted(circ) == rows and spot is not None and f"void {k0}: R_v" in view.readout.text()
              and v3.hover_row == k0, view.readout.text())
        # close-up: the cube around the void, read again at the grid's resolution
        x.close_up_3d(k0)
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_close is not None and (x._3d_res or {}).get("volume")
                      is not None and np.allclose(x._3d_res["request"]["lo"], x._3d_close["lo"]), 60)
        idx, wlo, _whi = V.cell_window(x._3d_close["lo"], x._3d_close["hi"], o.lo, o.length / o.n, o.n)
        crop = cube[np.ix_(*idx)]
        cu = VOL.average_window(o, "density", "value", x._3d_close["lo"], x._3d_close["hi"], VOL.DEFAULT_DETAIL)
        check("11. Close up: the volume of the void's cube, read periodically at the grid's own cells; its title",
              ok and x._3d_res["volume"]["shape"] == crop.shape and np.array_equal(cu["values"], crop)
              and np.allclose(v3.lo, wlo) and f"close-up on void {k0}" in view.title.text(),
              f"{x._3d_res['volume']['shape']} {crop.shape}")
        x.whole_box_3d()
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_close is None
                      and (x._3d_res or {}).get("volume") is not None and x._3d_res["volume"]["shape"] == (64,) * 3, 30)
        check("11. Whole box: the whole box's volume again from memory (nothing averaged again)",
              ok and x._3d_res["request"]["progress"] == {}, f"{x._3d_res['request']['progress']}")
        # back to what is shown while another volume is averaged and a third is queued: both moot
        other = {"vkey": ("another window",), "progress": {}, "style": "volume"}
        x._3d_busy, x._3d_inflight, x._3d_pending = True, other, {"vkey": ("a third",), "style": "volume"}
        x._3d_request()
        moot = x._3d_pending is None and other["progress"].get("cancel") is True
        x._3d_busy, x._3d_inflight = False, None
        check("11. a request back to the volume shown drops the one queued and cancels the averaging under way", moot)
        png = TMP / "volume3d.png"
        v3.cam.yaw += 3.0                                  # a new view: its sharp picture not drawn yet
        v3.update()
        qa.processEvents()
        x.save_3d_png(png)
        saved_sharp = v3._sharp is not None and v3._sharp[0] == v3._sharp_key()
        from PySide6.QtGui import QImage
        im = QImage(str(png))
        x._3d_failed("Cancelled: ")                         # a cancelled averaging: no error, the volume stays
        arr = np.frombuffer(im.convertToFormat(QImage.Format_RGB888).constBits(), np.uint8,
                            count=im.height() * im.convertToFormat(QImage.Format_RGB888).bytesPerLine())
        check("11. Save image in 3D writes the volume as on screen, its sharp picture finished first (not the quick "
              "one), not a blank view; a cancelled averaging is no error",
              png.is_file() and im.width() >= v3.width() and v3.volume is not None and saved_sharp
              and arr.std() > 5 and "Could not" not in v3.message)
        # the planes back, then the map: the volume's slices freed
        x.view3d_style.setCurrentIndex(x.view3d_style.findData("planes"))
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_res is not None and x._3d_res.get("volume") is None
                      and any(p is not None for p in x._3d_res["planes"]), 30)
        check("11. the planes again: their controls back, the planes read", ok and v3.style == "planes"
              and x.planes3d_box.isVisibleTo(x) and not x.volume3d_box.isVisibleTo(x))
        qa.processEvents()
        v3.repaint()
        ok_pt = bool(v3._drawn)
        if ok_pt:
            _a, T, rect = v3._drawn[0]
            mid = T.map(rect.center())
            pt = v3.point_at(mid.x(), mid.y())
            ok_pt = pt is not None and T.isAffine() is False
            if ok_pt:
                a_, xyz, value = pt
                ijk = [int(np.floor((xyz[b_] - o.lo[b_]) / (o.length[b_] / o.n))) % o.n for b_ in range(3)]
                ok_pt = abs(value - float(cube[tuple(ijk)])) < 1e-6 and ijk[a_] == x._3d_index[a_]
        check("11. the planes in perspective (each through a homography): a point under the mouse is still the stored "
              "value of that cell", ok_pt)
        x.view3d_style.setCurrentIndex(x.view3d_style.findData("volume"))
        stale_gone = v3.planes is None
        wait_for(qa, lambda: not x._3d_busy and (x._3d_res or {}).get("volume") is not None, 30)
        check("11. back to the volume the planes are let go (never drawn against another window later)", stale_gone)
        details = [x.view3d_detail.itemData(j) for j in range(x.view3d_detail.count())]
        x.view3d_detail.setCurrentIndex(x.view3d_detail.findData(1024))
        ok = wait_for(qa, lambda: not x._3d_busy and (x._3d_res or {}).get("volume") is not None
                      and x._3d_res["request"]["detail"] == 1024, 30)
        real_bpc, real_free = VOL.BYTES_PER_CELL, G.available_gb
        VOL.BYTES_PER_CELL, G.available_gb = 1e4, (lambda: 2.5)    # as if a 64³ cube needed 2.6 GB with 2.5 free
        shown_res, x._3d_res = x._3d_res, None                       # a fresh choice (none shown for it yet)
        try:
            fits = x._3d_detail(o, o.lo, o.lo + o.length)
        finally:
            VOL.BYTES_PER_CELL, G.available_gb = real_bpc, real_free
            x._3d_res = shown_res
        check("11. the detail menu offers 128³ to 1024³; 1024³ of a 64³ grid is its own cells; a cube that would not "
              "fit in the free memory (less 2 GB) falls back to a smaller detail, and says why",
              details == [128, 256, 512, 1024] and ok and x._3d_res["volume"]["shape"] == (64,) * 3
              and "the grid's own cells" in x.view3d_state.text() and fits[0] == 128 and "would need" in fits[1],
              f"{details} {fits} {x.view3d_state.text()}")
        # the shown choice is kept: a recolour never checks the memory again (it would count the shown cube as used)
        x._3d_res["request"].update(want=1024, detail=128, note="1024³ would need 9 GB with 3 GB free: 128³ instead")
        VOL.BYTES_PER_CELL, G.available_gb = 1e4, (lambda: 2.5)
        try:
            kept = x._3d_detail(o, x._3d_res["request"]["lo"], x._3d_res["request"]["hi"])
        finally:
            VOL.BYTES_PER_CELL, G.available_gb = real_bpc, real_free
        x._3d_res["request"].update(want=1024, detail=1024, note="")
        # a DTFE run going: more than 1 GB is asked about first, a 'no' falls back below 1 GB
        asked = []
        x.confirm_heavy = lambda title, text: asked.append(title) or False
        rs.dtfe_processes = lambda max_age=2.0: {"runs": [(4242, "PS-DTFE", 20.0)], "servers": []}
        VOL.BYTES_PER_CELL, G.available_gb = 1e4, (lambda: 5.0)    # 2.6 GB: fits, but not beside the run's 4 GB margin
        shown_res, x._3d_res = x._3d_res, None
        try:
            guarded = x._3d_detail(o, o.lo, o.lo + o.length)
        finally:
            VOL.BYTES_PER_CELL, G.available_gb = real_bpc, real_free
            x._3d_res = shown_res
            rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}
            x.confirm_heavy = None
        check("11. the choice shown is kept (a recolour never re-reads it at another detail); during a DTFE run a "
              "cube over 1 GB is asked about first, a 'no' falling back (said)",
              kept == (128, "1024³ would need 9 GB with 3 GB free: 128³ instead") and asked == ["3D volume"]
              and guarded[0] == 128 and "a DTFE run is going" in guarded[1], f"{kept} {asked} {guarded}")
        x._3d_request()
        wait_for(qa, lambda: not x._3d_busy and (x._3d_res or {}).get("volume") is not None, 30)
        x.view3d_detail.setCurrentIndex(x.view3d_detail.findData(VOL.DEFAULT_DETAIL))
        wait_for(qa, lambda: not x._3d_busy, 30)
        x.view3d_toggle.setChecked(False)
        wait_for(qa, lambda: not x._busy, 30)
        check("11. back to the map: the volume's coloured slices freed", v3.volume is None and not v3._stacks
              and view.canvas.isVisible())
        # the grid written again by a re-run: the volume is read again (the files' mtime and size in its key)
        shown, vkey = x._3d_res, (x._3d_res or {}).get("vkey")      # (last: the void catalogue goes stale too)
        path = o.files["density"].path
        time.sleep(0.01)
        path.write_bytes(path.read_bytes())
        x.view3d_toggle.setChecked(True)
        x._3d_request()
        ok = wait_for(qa, lambda: not x._3d_busy and x._3d_res is not shown and (x._3d_res or {}).get("volume")
                      is not None, 30)
        check("11. a grid written again (a re-run) is read again by the 3D view, as the map does", ok
              and x._3d_res["vkey"] != vkey)
        x._3d_busy, x._3d_inflight = True, {"vkey": ("x",), "progress": {}, "style": "volume", "output": o}
        flag = x._3d_inflight["progress"]
        x.shutdown()
        check("11. closing the window cancels a volume being read (the slice thread comes back at once)",
              flag.get("cancel") is True)
        check("11. the volume's texts carry no '--'", all("--" not in t for t in (
            X3.VIEW3D_TIP, X3.STYLE_TIP, X3.DETAIL_TIP, X3.RANGE_TIP, X3.OPACITY_TIP, x.vol3d_default.toolTip())))
    finally:
        rs.dtfe_processes = real_procs
        x.shutdown()


def fold_map():
    """Explore's 'caustic (fold) cells' map of a periodic PS run is the exact fold
    flag, dtfelib.io.caustic_is_fold(class mask, streams=, hidden=) -- the stored '.caustic' of older grids
    is set only where a folded tetrahedron deposited a sample and misses every fold thinner than the
    sampling -- in the map, the click readout and its tooltip; the caustic-class map stays the stored mask."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (fold map): PySide6 is not importable ({e})")
        return
    import explore as E
    from dtfelib.io import caustic_is_fold
    print("fold map")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "fold data"
    d = make_snapdir(data, 33, 0.0, 3)
    st = np.ones((N, N, N), np.float32)
    st[:, :, 20:22] = 3.0                           # a fold sheet the samples saw (three streams)
    hid = np.zeros((N, N, N), np.float32)
    hid[:, 10, :] = 1.0                             # a fold thinner than the sampling: hidden-streams bit 1 only
    hid[5, :, :] = 2.0                              # bit 2 alone (an unsampled tetrahedron's mass): no fold
    raw = np.zeros((N, N, N), np.float32)
    raw[:, :, 20] = 1.0                             # the per-sample stored flag: half the sheet, none of the thin fold
    ccm = np.full((N, N, N), 1 + 4, np.float32)
    for sfx, arr in (("a_streams", st), ("a_hidden_streams", hid), ("caustic", raw), ("causticClass", ccm)):
        arr.tofile(str(d / f"ps_output.{sfx}"))
    want = caustic_is_fold(ccm, streams=st, hidden=hid)
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(600, 600)
    try:
        x.refresh_outputs()
        wait_for(qa, lambda: not x._busy and x.current is not None and x._plane is not None, 30)
        x.field.setCurrentIndex(x.field.findData("caustic"))
        ok = wait_for(qa, lambda: not x._busy and x._req is not None and x._req["field"] == "caustic", 30)
        ax, k = (x._req["axis"], x._req["index"]) if ok else (2, 0)
        plane = np.asarray(x._plane) if ok else None
        exp = np.take(want, k, axis=ax).astype(np.float64)
        tip = x.field.itemData(x.field.findData("caustic"), Qt.ToolTipRole) or ""
        check("the caustic map of a periodic PS run is the exact fold flag (the thin fold and the whole sheet), not "
              "the stored per-sample grid; its tooltip says so",
              ok and plane is not None and plane.shape == exp.shape and np.array_equal(plane, exp)
              and not np.array_equal(plane, np.take(raw, k, axis=ax)) and "2026-10-13" in tip and "--" not in tip,
              f"{ok} {ax} {k} {None if plane is None else (plane.sum(), exp.sum())}")
        o = x.current
        check("... the click readout too (a thin-fold cell reads 1, a bit-2-only cell 0)",
              o.value("caustic", (3, 10, 40)) == 1.0 and o.value("caustic", (5, 30, 40)) == 0.0
              and o.value("caustic", (7, 30, 21)) == 1.0 and float(raw[7, 30, 21]) == 0.0)
        x.field.setCurrentIndex(x.field.findData("caustic_class"))
        ok = wait_for(qa, lambda: not x._busy and x._req is not None and x._req["field"] == "caustic_class", 30)
        check("... and the caustic-class map stays the stored mask",
              ok and np.array_equal(np.asarray(x._plane), np.take(ccm, x._req["index"], axis=x._req["axis"])))
        # a job running caps the cube cache (1 GB; here 2.5 cubes): the three cubes of the fold map do not
        # fit together, so only their planes are read -- read whole they would evict each other, three
        # 0.5 GB reads per step at 512³. Uncapped (they fit) the cubes are cached: stepping stays fast.
        real_read, real_run = G._read_plane, G.RUN_CUBE_BYTES
        reads = []

        def spy(path, *a):
            reads.append(Path(path).name)
            return real_read(path, *a)
        fold_files = {f"ps_output.{s}" for s in ("a_streams", "a_hidden_streams", "causticClass")}
        G.drop_cubes()
        G._read_plane, G.RUN_CUBE_BYTES = spy, int(2.5 * N ** 3 * 4)
        G.cap_cubes(True)
        try:
            planes = [(k_, o.slice("caustic", 2, k_)) for k_ in (19, 20, 21)]
            cached = {Path(str(key[0])).name for key in G._CUBES}
            G.cap_cubes(False)
            o.slice("caustic", 2, 22)
            cached_free = {Path(str(key[0])).name for key in G._CUBES}
        finally:
            G._read_plane, G.RUN_CUBE_BYTES = real_read, real_run
            G.cap_cubes(False)
        check("a job running (the cube cache capped below the three cubes): the fold map reads only its planes (three "
              "per step), caches no cube, and is the same exact flag; uncapped the three cubes are cached as before",
              sorted(reads) == sorted(list(fold_files) * 3) and not (cached & fold_files)
              and all(np.array_equal(p, np.take(want, k_, axis=2).astype(np.float64)) for k_, p in planes)
              and fold_files <= cached_free, f"{sorted(reads)} {cached} {cached_free}")
        # the exact zoom (load_exact_planes): a new run's grids, one cell thick. Its fold map must not mask both
        # parity bits of the 0/1 '.caustic' grid (that shows no fold cell at all), nor its class map show 0/1
        import types
        zd = TMP / "fold zoom"
        zd.mkdir(parents=True, exist_ok=True)
        zg = (8, 8, 1)
        zc = np.zeros(zg, np.float32)
        zc[:3] = 1.0                                    # the fold cells (the binary's exact flag)
        zcc = np.full(zg, 1 + 4, np.float32)
        zcc[:3] = 1 + 2 + 8
        for sfx, arr in (("a_den", np.ones(zg, np.float32)), ("caustic", zc), ("causticClass", zcc)):
            arr.tofile(str(zd / f"zoom.{sfx}"))

        def zoom_maps():
            stub = types.SimpleNamespace(_zoom_fields=E.load_exact_planes(zd / "zoom", zg),
                                         _zoom={"axis": 2, "output": types.SimpleNamespace(dim=3)})
            return (E.ExploreControls._zoom_plane(stub, "caustic", "norm"),
                    E.ExploreControls._zoom_plane(stub, "caustic_class", "norm"))
        zfold, zcls = zoom_maps()
        (zd / "zoom.causticClass").unlink()
        zfold1, _ = zoom_maps()
        check("the exact zoom's fold map is its '.caustic' and its class map the class mask (both read the 0/1 "
              "grid: no fold cell at all); a zoom without the class grid still shows its folds",
              np.array_equal(zfold, zc[:, :, 0]) and np.array_equal(zcls, zcc[:, :, 0])
              and np.array_equal(zfold1, zc[:, :, 0]), f"{zfold.sum()} {zcls.max()} {zfold1.sum()}")
    finally:
        x.shutdown() if hasattr(x, "shutdown") else None


def write_grids(d: Path, prefix: str, n: int, dim: int = 3, seed: int = 0):
    """A plain output 'd/prefix.*': density (noise around 1), velocity (zero) and stream counts on n^dim cells."""
    shape = (n,) * dim
    (1.0 + 0.1 * np.random.default_rng(seed).normal(size=shape)).astype(np.float32).tofile(str(d / f"{prefix}.a_den"))
    np.zeros(shape + (dim,), np.float32).tofile(str(d / f"{prefix}.a_vel"))
    np.ones(shape, np.float32).tofile(str(d / f"{prefix}.a_streams"))


MF_N, MF_Z, MF_H, MF_BOX = 32, 1.0, 0.7, 32.0
MF_STORED = (("divergence", "velDiv", 1), ("shear", "velShear", 5), ("vorticity", "velVort", 3),
             ("gradient", "velGrad", 9), ("vweb", "velVweb", 1), ("vweb_eigenvalues", "velVwebEig", 3))


def mean_field_data(data: Path) -> Path:
    """A z = 1 snapshot (h = 0.7, a 32 Mpc box) with a PS-DTFE output whose velocity is random (every mode up to the
    grid's Nyquist) and whose stored velocity derivatives are unrelated noise, its run log's lambda_th 0.2, and a
    standard-DTFE 'output' with its own (noise) divergence beside it."""
    import h5py
    d = data / "SYNMF-1-Dark" / "snapdir_050"
    d.mkdir(parents=True, exist_ok=True)
    n = MF_N
    with h5py.File(d / "combined_050.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = MF_BOX * 1000.0, MF_Z, MF_H, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, n ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
    rng = np.random.default_rng(21)
    write_grids(d, "ps_output", n)
    (300.0 * rng.normal(size=(n, n, n, 3))).astype(np.float32).tofile(str(d / "ps_output.a_vel"))
    for _name, sfx, c in MF_STORED:
        rng.normal(size=(n, n, n) + ((c,) if c > 1 else ())).astype(np.float32).tofile(str(d / f"ps_output.a_{sfx}"))
    (d / "ps_output.runlog").write_text("RUNNING: ./PS-DTFE in out --grid 32 --periodic --lambda_th 0.2\n")
    (1.0 + 0.1 * rng.normal(size=(n, n, n))).astype(np.float32).tofile(str(d / "output.a_den"))
    rng.normal(size=(n, n, n, 3)).astype(np.float32).tofile(str(d / "output.a_vel"))
    rng.normal(size=(n, n, n)).astype(np.float32).tofile(str(d / "output.a_velDiv"))
    return d


def mean_field_pure():
    """9. (no Qt) the mean-field planes of grids.OutputSet against dtfelib's FieldSet, the outputs that cannot have
    them, the labels, the cube cache's rules while a job runs."""
    import json
    from dtfelib import FieldSet
    from dtfelib.cli import add_ps_gradient_arg
    import argparse
    print("PS velocity derivatives: the mean field")
    data = TMP / "mf data"
    d = mean_field_data(data)
    outs = {o.prefix: o for _t, o in G.find_outputs(data)}
    ps_o, dt_o = outs["ps_output"], outs["output"]
    fs = FieldSet(d, method="ps")
    # the reference: dtfelib's own mean field (raw u-units, as the binary stores its grids) written as another
    # output's STORED grids, so slice() applies the same components, curl and sqrt(a) to both
    ref = data / "ref"
    ref.mkdir(exist_ok=True)
    write_grids(ref, "ps_ref", MF_N)
    for name, sfx, _c in MF_STORED:
        fs.load(name, scaled=False, mean_field=True).tofile(str(ref / f"ps_ref.a_{sfx}"))
    ref_o = G.OutputSet(ref / "ps_ref", box=ps_o.box, redshift=ps_o.redshift)
    bad = []
    for name, _sfx, _c in MF_STORED:
        for comp in G.components(name, ps_o.files[name].ncomp):
            for axis in range(3):
                got = ps_o.slice(name, axis, 9, comp, "mean-field")
                want = ref_o.slice(name, axis, 9, comp)
                if name == "vweb":
                    off = float(np.mean(got != want))
                    if off > 1e-3:
                        bad.append(f"{name} axis {axis}: {off:.4f} of the labels differ")
                    continue
                err = float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))
                if not err < 1e-5:
                    bad.append(f"{name} [{comp}] axis {axis}: {err:.2e}")
    check("9. a PS output's mean-field planes (every name and component, every axis) are dtfelib's FieldSet.load("
          "name, mean_field=True) cut there, through slice()'s components, curl and sqrt(a) (the run log's lambda_th, "
          "the header's h for the V-web)", not bad, "; ".join(bad[:6]))
    stored = np.fromfile(d / "ps_output.a_velDiv", dtype=np.float32).reshape((MF_N,) * 3)
    sa = np.sqrt(1.0 / (1.0 + MF_Z))
    dt_div = np.fromfile(d / "output.a_velDiv", dtype=np.float32).reshape((MF_N,) * 3)
    check("9. 'stored' (and the default of slice()) is the stored stream average; standard DTFE's stored grid either "
          "way (its families: 'mean' always; PS: 'mean' or 'stream')",
          np.allclose(ps_o.slice("divergence", 2, 9, "value", "stored"), stored[:, :, 9] * sa, rtol=1e-6)
          and np.allclose(ps_o.slice("divergence", 2, 9), stored[:, :, 9] * sa, rtol=1e-6)
          and np.allclose(dt_o.slice("divergence", 2, 9, "value", "mean-field"), dt_div[:, :, 9] * sa, rtol=1e-6)
          and dt_o.gradient_family("divergence", "stored") == "mean" == dt_o.gradient_family("vweb", "mean-field")
          and ps_o.gradient_family("shear", "mean-field") == "mean" and ps_o.gradient_family("shear", "stored") == "stream"
          and ps_o.gradient_family("density", "mean-field") == "" and ps_o.mean_field_reason() == "")
    parser = add_ps_gradient_arg(argparse.ArgumentParser())
    check("9. the launcher's default is the figure scripts' ('--ps-gradient' of dtfelib.cli: the mean field)",
          G.PS_GRADIENT_DEFAULT == parser.parse_args([]).ps_gradient == "mean-field"
          and set(G.PS_GRADIENT_CHOICES) == {"mean-field", "stored"})
    # the outputs that keep the stored grid: 2D, not periodic, a window
    other = data / "other"
    other.mkdir(exist_ok=True)
    for name, side, dim in (("np", {"estimator": "ps", "periodic": False}, 3), ("win", {"estimator": "ps",
                            "window": [0, 10, 0, 10, 0, 10]}, 3), ("flat", {"estimator": "ps", "dim": 2}, 2)):
        write_grids(other, name, 16, dim)
        np.zeros((16,) * dim, np.float32).tofile(str(other / f"{name}.a_velDiv"))
        (other / f"{name}.gui.json").write_text(json.dumps(dict(side, box_mpc=[0.0, 16.0] * dim)))
    kept = {nm: G.OutputSet(other / nm, box=G.custom_box(G.sidecar(other / nm))) for nm in ("np", "win", "flat")}
    reasons = {nm: o.mean_field_reason() for nm, o in kept.items()}
    check("9. a non-periodic run, a window of the grid and a 2D grid keep the stored stream average with 'mean-field', "
          "and say why", reasons["np"] == "not a periodic box" and "window" in reasons["win"] and "2D" in reasons["flat"]
          and all(o.gradient_family("divergence", "mean-field") == "stream" for o in kept.values())
          and all(o.slice("divergence", 2, 3, "value", "mean-field").shape == (16, 16) for o in kept.values()), str(reasons))
    lab = (G.field_label(ps_o, "divergence", "mean"), G.field_label(ps_o, "divergence", "stream"),
           G.field_label(dt_o, "divergence", "mean"), G.field_label(dt_o, "divergence", "mean", qualify=True),
           G.field_label(ps_o, "density", ""))
    check("9. field_label: a PS map says its family, a standard-DTFE map alone keeps its name (qualified beside a PS "
          "one), other fields unchanged; no '--'",
          lab == ("velocity divergence (mean field ∇⟨v⟩)", "velocity divergence (stream average ⟨∇v⟩)",
                  "velocity divergence", "velocity divergence (mean field ∇⟨v⟩)", "density ρ/ρ̄")
          and not any("--" in t for t in lab), str(lab))
    # a face-gradient output (PS-DTFE's default with the exact deposit and the volume weighting, docs/cleanup-plan.md
    # D13): its run writes the stream average's divergence beside the family ('.a_velDivStreams'), whose stored grids
    # are then the mean field itself -- shown as stored with either choice, and labelled the mean field. The same
    # grids without the marker keep both families
    face = face_outputs(d)
    face_o, plain_o = (G.OutputSet(face / p, box=ps_o.box, redshift=ps_o.redshift) for p in ("ps_face", "ps_plain"))
    stored_face = face_o.slice("divergence", 2, 9, "value", "stored")
    check("9. a face-gradient output ('.a_velDivStreams' beside its family): its stored grids are the mean field, "
          "shown as stored with either choice and labelled so; without the marker the same grids keep both families; "
          "dtfelib's FieldSet.face_gradient agrees",
          face_o.face_gradient and not plain_o.face_gradient and not ps_o.face_gradient and not dt_o.face_gradient
          and face_o.gradient_family("shear", "stored") == "mean" == face_o.gradient_family("shear", "mean-field")
          and face_o.gradient_family("density", "mean-field") == ""
          and plain_o.gradient_family("shear", "stored") == "stream"
          and plain_o.gradient_family("shear", "mean-field") == "mean"
          and np.array_equal(face_o.slice("divergence", 2, 9, "value", "mean-field"), stored_face)
          and np.allclose(stored_face, stored[:, :, 9] * sa, rtol=1e-6)
          and not np.allclose(plain_o.slice("divergence", 2, 9, "value", "mean-field"), stored_face)
          and FieldSet(face, method="ps", prefix="ps_face").face_gradient
          and not FieldSet(face, method="ps", prefix="ps_plain").face_gradient,
          f"{face_o.face_gradient} {plain_o.face_gradient} {face_o.gradient_family('shear', 'stored')}")
    # a job running: the cube cache capped below the velocity cube -> slabs from the file, nothing cached whole
    real_run, real_get = G.RUN_CUBE_BYTES, G._SlabFile.__getitem__
    slabs = []

    def spy(self, sl):
        slabs.append(self.path.name)
        return real_get(self, sl)
    try:
        G.drop_cubes()
        free = ps_o.slice("divergence", 1, 7, "value", "mean-field")
        cached_free = {Path(str(k[0])).name for k in G._CUBES}
        G.drop_cubes()
        G._SlabFile.__getitem__, G.RUN_CUBE_BYTES = spy, 300_000      # the velocity grid is 393 kB
        G.cap_cubes(True)
        capped = ps_o.slice("divergence", 1, 7, "value", "mean-field")
        n_first = len(slabs)
        again = ps_o.slice("shear", 1, 7, "1", "mean-field")          # the same plane: no second pass
        cached_capped = {Path(str(k[0])).name for k in G._CUBES}
        check("9. a job running (the cube cache capped below the velocity grid): the plane is computed from slabs read "
              "off the file, the velocity is not cached whole, the numbers are the same; another derivative of the "
              "same plane reads nothing again; uncapped the velocity cube is cached as any grid",
              n_first > 0 and set(slabs) == {"ps_output.a_vel"} and len(slabs) == n_first
              and "ps_output.a_vel" not in cached_capped and np.array_equal(free, capped)
              and "ps_output.a_vel" in cached_free and again.shape == (MF_N, MF_N),
              f"{slabs} {cached_capped} {cached_free}")
    finally:
        G._SlabFile.__getitem__, G.RUN_CUBE_BYTES = real_get, real_run
        G.cap_cubes(False)
    # a rewritten velocity grid (a re-run): the plane cache must not answer for it
    v = np.fromfile(d / "ps_output.a_vel", dtype=np.float32)
    before = ps_o.slice("divergence", 0, 4, "value", "mean-field")
    time.sleep(0.01)
    (2.0 * v).tofile(str(d / "ps_output.a_vel"))
    after = ps_o.slice("divergence", 0, 4, "value", "mean-field")
    v.tofile(str(d / "ps_output.a_vel"))
    check("9. a velocity grid written again: its new mean field, not the cached plane (twice the velocity, twice "
          "the divergence)", np.allclose(after, 2.0 * before, rtol=1e-5, atol=1e-6 * np.abs(before).max()))
    G.drop_cubes()


def face_outputs(d: Path) -> Path:
    """Two copies of part 9's PS-DTFE output beside a copy of its snapshot: 'ps_face' with the face gradient's marker
    ('.a_velDivStreams', here a copy of its divergence), 'ps_plain' without it."""
    face = TMP / "mf face"
    face.mkdir(exist_ok=True)
    shutil.copy(d / "combined_050.hdf5", face / "combined_050.hdf5")
    for p in d.glob("ps_output.a_*"):
        for prefix in ("ps_face", "ps_plain"):
            shutil.copy(p, face / p.name.replace("ps_output", prefix, 1))
    shutil.copy(face / "ps_face.a_velDiv", face / "ps_face.a_velDivStreams")
    return face


def face_gradient_zoom():
    """9. the exact zoom of a phase-space output takes the output's own velocity-gradient route: '--face-gradient' for
    a face-gradient output (its '.a_velDivStreams'), '--per-tet-gradient' for one without it -- the plan only (a
    stand-in for the binary, nothing run)."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (the exact zoom's gradient route): PySide6 is not importable ({e})")
        return
    import explore_zoom as XZ
    QApplication.instance() or QApplication([])
    print("the exact zoom's gradient route")
    face = TMP / "mf face"
    if not (face / "ps_face.a_velDivStreams").is_file():
        face = face_outputs(mean_field_data(TMP / "mf data"))
    box = ((0.0, 0.0, 0.0), (MF_BOX, MF_BOX, MF_BOX))
    outs = {p: G.OutputSet(face / p, box=box, redshift=MF_Z) for p in ("ps_face", "ps_plain")}
    fake = face / "PS-DTFE-standin"
    fake.write_text("")
    real_name, plans = XZ.rs.binary_name, {}
    XZ.rs.binary_name = lambda estimator, precision="single", dim=3: str(fake)     # (an absolute path: REPO_ROOT / it)
    try:
        for p, o in outs.items():
            plans[p] = XZ.exact_zoom_plan(o, 2, 9, [0, 1], (4.0, 12.0), (4.0, 12.0), 16, None, str(face / "cache"), 0)
    finally:
        XZ.rs.binary_name = real_name
        for plan in plans.values():
            if plan.get("root"):
                shutil.rmtree(Path(plan["root"]).parent, ignore_errors=True)
    cmds = {p: plan.get("cmd") or [] for p, plan in plans.items()}
    routes = {p: [a for a in c if a in ("--face-gradient", "--per-tet-gradient")] for p, c in cmds.items()}
    check("9. the exact zoom takes the output's own route: '--face-gradient' for a face-gradient output, "
          "'--per-tet-gradient' for one without the marker (both exact and volume-weighted)",
          routes == {"ps_face": ["--face-gradient"], "ps_plain": ["--per-tet-gradient"]}
          and all("--exact" in c and "--ps-volume-weighted" in c for c in cmds.values()),
          f"{routes} {[plan.get('note') for plan in plans.values()]}")


def mean_field_gui():
    """9. the window: the choice, the map's labels, Compare with, the figure grid, the saved figure, the state."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (PS derivatives in the window): PySide6 is not importable ({e})")
        return
    import types
    import explore as E
    qa = QApplication.instance() or QApplication([])
    data = TMP / "mf data"
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(700, 600)
    texts = []

    def shown():
        ok_ = wait_for(qa, lambda: not x._busy and x._req is not None and x._req["output"] is x.current
                       and x._plane is not None, 30)
        texts.extend([view.title.text(), view.colorbar.label, x.ps_gradient_note.text(), x.zoom_state.text()])
        return ok_

    def pick(combo, value):
        combo.setCurrentIndex(combo.findData(value))
        return shown()
    try:
        x.refresh_outputs()
        shown()
        ps_root = str(data / "SYNMF-1-Dark" / "snapdir_050" / "ps_output")
        dt_root = str(data / "SYNMF-1-Dark" / "snapdir_050" / "output")
        pick(x.prefix_combo, ps_root)
        ok = pick(x.field, "divergence")
        o, req = x.current, x._req
        from dtfelib import FieldSet                    # the figure scripts' mean field, cut at the map's slice
        mf = np.take(FieldSet(Path(ps_root).parent, method="ps").load("divergence", mean_field=True), req["index"],
                     axis=req["axis"])
        title = view.title.text()

        def same(a, b):                                 # equal to float32 rounding (the FFTs sum in other orders)
            return np.shape(a) == np.shape(b) and float(np.abs(np.asarray(a) - b).max()) <= 1e-5 * float(np.abs(b).max())
        check("9. the choice starts at the mean field; a PS divergence map is the mean field, and its title and colour "
              "bar say so", ok and x.ps_gradient.currentData() == "mean-field" and req.get("ps_gradient") == "mean-field"
              and same(x._plane, mf) and "mean field ∇⟨v⟩" in title and "mean field ∇⟨v⟩" in view.colorbar.label
              and not x.ps_gradient_note.text(), f"{title} | {view.colorbar.label}")
        import matplotlib.pyplot as plt
        fig, _dpi = x.map_figure()
        saved = [t.get_text() for ax in fig.axes for t in (ax.title,)] + [ax.get_ylabel() for ax in fig.axes]
        plt.close(fig)
        check("9. ... the saved figure's title and colour bar say it too", any("mean field ∇⟨v⟩" in t for t in saved[:1])
              and any("mean field ∇⟨v⟩" in t for t in saved[1:]), str(saved))
        ok = pick(x.ps_gradient, "stored")
        st = o.slice("divergence", req["axis"], req["index"], "value", "stored")
        check("9. 'stream average': the stored grid, and the title says so", ok and np.allclose(x._plane, st)
              and not np.allclose(st, mf) and "stream average ⟨∇v⟩" in view.title.text(), view.title.text())
        # Compare with: A = PS, B = standard DTFE
        x.compare.setCurrentIndex(x.compare.findData(dt_root))
        ok = shown()
        t_stored = view.title.text()
        b_plane = x._planes[1] if x._planes is not None else None
        x.ps_gradient.setCurrentIndex(x.ps_gradient.findData("mean-field"))
        ok2 = shown()
        t_mean = view.title.text()
        check("9. Compare with a standard-DTFE output: B is its stored grid; the title names A's and B's families when "
              "they differ, the one family when they agree; A follows the choice",
              ok and ok2 and b_plane is not None
              and "(A stream average ⟨∇v⟩, B mean field ∇⟨v⟩)" in t_stored and "A − B: velocity divergence (mean field ∇⟨v⟩)" in t_mean
              and same(x._planes[0], mf), f"{t_stored} | {t_mean}")
        x.compare.setCurrentIndex(0)
        shown()
        # the figure grid: the standard and the phase-space divergence side by side
        x.grid_set_layout(1, 2)
        x.grid_fill("types")
        x.grid_merge.setChecked(True)
        req_g = x.grid_request()
        items = E.grid_items(req_g)
        q_mean = E.grid_differences(items)["quantities"]
        sup_mean = E.grid_suptitle(req_g, items)
        x.ps_gradient.setCurrentIndex(x.ps_gradient.findData("stored"))
        shown()
        req_s = x.grid_request()
        items_s = E.grid_items(req_s)
        q_stored = E.grid_differences(items_s)["quantities"]
        labels_s = sorted(E.panel_label(it, E.grid_differences(items_s)) for it in items_s if it)
        fig, _n = E.grid_figure(req_s, items_s)
        bars = [ax.get_ylabel() for ax in fig.axes[2:]]
        fig_m, _n = E.grid_figure(req_g, items)
        bars_m = [ax.get_ylabel() for ax in fig_m.axes[2:]]
        check("9. the figure grid: with the mean field the standard and PS divergence are one quantity (one colour bar, "
              "the family in the title); stored, two (two bars, each panel named with its family)",
              req_g["ps_gradient"] == "mean-field" and len(q_mean) == 1 and "mean field ∇⟨v⟩" in sup_mean
              and len(bars_m) == 1 and len(q_stored) == 2 and len(bars) == 2
              and any("stream average" in t for t in labels_s) and any("mean field" in t for t in labels_s)
              and any("stream average" in b for b in bars), f"{q_mean} {sup_mean} {bars_m} | {q_stored} {labels_s} {bars}")
        texts.extend([sup_mean] + labels_s + bars + bars_m)
        st_ = x.state()
        x.ps_gradient.setCurrentIndex(x.ps_gradient.findData("mean-field"))
        shown()
        x.restore_state(st_)
        back = x.ps_gradient.currentData()
        x.restore_state({k: v for k, v in st_.items() if k != "ps_gradient"})     # a state saved before the key existed
        check("9. the choice is remembered (state / restore_state); a state saved before it leaves the choice alone",
              st_.get("ps_gradient") == "stored" and back == "stored" and x.ps_gradient.currentData() == "stored")
        x.ps_gradient.setCurrentIndex(x.ps_gradient.findData("mean-field"))
        shown()
        # a zoom is a point evaluation: the stream average, said; the zoom line is text (not the function that
        # writes the window)
        ax_ = x.axis.currentIndex()
        shp = [8, 8, 8]
        shp[ax_] = 1
        g = np.zeros(shp + [3, 3])
        g[..., 0, 0] = 1.0
        f = types.SimpleNamespace(density=np.ones(shp), streams=None, caustic=None, velocity=None, dispersion=None,
                                  sigma=None, velocity_gradient=g, offsets=None)
        zreq = {"output": x.current, "axis": ax_, "index": x.index.value(), "rest": [a for a in range(3) if a != ax_],
                "a0": (2.0, 10.0), "a1": (4.0, 12.0), "res": 8, "depth": 1, "stamp": x._file_stamp(x.current)}
        x._zoomed(zreq, f)
        zl, zt = x.zoom_state.text(), view.title.text()
        check("9. a zoom of the PS divergence: the stream average in the title, the zoom line says why, and its window "
              "is written out (no '<function')", x.is_zoomed() and "stream average ⟨∇v⟩" in zt
              and "a point evaluation has no mean field" in zl and "<function" not in zl and "2.00–10.00" in zl,
              f"{zt} | {zl}")
        texts.extend([zl, zt])
        x.zoom_out()
        shown()
        # an opened non-periodic phase-space output: the stored grid, said under the choice
        ok = x.open_file(str(data / "other" / "np.a_den"))
        ok = ok and wait_for(qa, lambda: not x._busy and x.current is not None and x.current.prefix == "np", 30)
        pick(x.field, "divergence")
        check("9. a non-periodic PS output: its stored stream average, the note under the choice says why",
              ok and "stream average" in view.title.text() and "not a periodic box" in x.ps_gradient_note.text()
              and not x.ps_gradient_note.isHidden(), f"{view.title.text()} | {x.ps_gradient_note.text()}")
        texts.extend([x.ps_gradient.toolTip(), x.ps_gradient.itemText(0), x.ps_gradient.itemText(1)])
        check("9. the choice's texts carry no '--'", not [t for t in texts if "--" in t], str([t for t in texts if "--" in t]))
    finally:
        x.shutdown()


def keep_slice():
    """7. switching the output or the redshift keeps the slice (the axis and the place in
    the box, at the nearest cell of the new grid) and the zoom window; switching the simulation resets both."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (kept slice): PySide6 is not importable ({e})")
        return
    import json
    import types
    import explore as E
    print("the slice kept across outputs and redshifts")
    qa = QApplication.instance() or QApplication([])
    check("7. kept_index: a cell centre gives its own cell back, a cell face the upper cell, clamped to the grid",
          [E.kept_index((i + 0.5) / 64, 64) for i in (0, 10, 63)] == [0, 10, 63] and E.kept_index(7.5 / 32, 64) == 15
          and E.kept_index(10.5 / 64, 32) == 5 and E.kept_index(1.0, 8) == 7 and E.kept_index(-0.2, 8) == 0)
    data, custom, opened = TMP / "keep data", TMP / "keep custom", TMP / "keep opened"
    s50 = make_snapdir(data, 50, 1.0, 11)                 # z = 1: ps_output 64^3 and a standard 'output' 32^3
    write_grids(s50, "output", 32)
    make_snapdir(data, 99, 0.0, 12)                       # z = 0: ps_output 64^3
    sim2 = SIM.replace("-1-", "-2-")
    make_snapdir(data, 99, 0.0, 13, sim=sim2, box=32.0)   # another simulation, a 32 Mpc box

    def run(d_, name, n, dim=3, box=BOX, side=True):
        d_.mkdir(parents=True, exist_ok=True)
        write_grids(d_, name, n, dim)
        if side:
            (d_ / f"{name}.gui.json").write_text(json.dumps({"dim": dim, "box_mpc": [0.0, box] * dim, "redshift": None}))
        return d_ / name
    c3a, c2d, c3b, c3s = run(custom, "c3a", 64), run(custom, "c2d", 64, 2), run(custom, "c3b", 32), run(custom, "c3s", 32, box=32.0)
    o1, o2, o3 = run(opened, "o1", 64), run(opened, "o2", 32), run(opened, "o3", 16, side=False)
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [str(custom)])
    view.resize(700, 600)
    said = []
    x.status.connect(said.append)
    lines = []                                            # every zoom line shown, for the '--' rule

    def shown(root=None):
        ok_ = wait_for(qa, lambda: not x._busy and x.current is not None and x._req is not None
                       and x._req["output"] is x.current and (root is None or str(x.current.root) == str(root)), 30)
        lines.append(x.zoom_state.text())
        return ok_

    def at():
        return x.axis.currentIndex(), x.index.value()

    def pick(combo, data_):
        combo.setCurrentIndex(combo.findData(data_))
        return shown()

    def fields(val, ax=None):
        shp = [8, 8, 8]
        shp[x.axis.currentIndex() if ax is None else ax] = 1      # the slice axis is squeezed out
        return types.SimpleNamespace(density=np.full(shp, float(val)), streams=None, caustic=None, velocity=None,
                                     dispersion=None, sigma=None, velocity_gradient=None, offsets=None)

    def zoom_req(o=None):
        o = o or x.current
        ax, i = x.axis.currentIndex(), x.index.value()
        return {"output": o, "axis": ax, "index": i, "rest": [a for a in range(3) if a != ax], "a0": (20.0, 30.0),
                "a1": (40.0, 50.0), "res": 8, "depth": 1, "stamp": x._file_stamp(o)}

    def memo(o, ax, i, f):                                # a zoom of 'o' made before (the window, slice ax/i)
        req = dict(zoom_req(o), axis=ax, index=i, rest=[a for a in range(3) if a != ax])
        x._memo_put(x._memo_key(o, ax, i, req["a0"], req["a1"], 8, False), req, f)

    try:
        x.refresh_outputs()
        ok = shown()
        check("7. the first open: the latest snapshot of the first simulation, at the middle of the box",
              ok and x.current.sim == SIM and x.current.snap == 99 and x.index.value() == N // 2, f"{at()}")
        x.axis.setCurrentIndex(0)
        x.index.setValue(10)
        shown()
        ok = pick(x.snap_combo, 50)
        check("7. another redshift: the same axis and the same place (x = 10.50 Mpc, the same grid)",
              ok and x.current.snap == 50 and x.current.prefix == "ps_output" and at() == (0, 10)
              and "x = 10.50 Mpc" in x.position.text(), f"{at()} {x.position.text()}")
        p50 = x.current
        ok = pick(x.prefix_combo, str(s50 / "output"))
        check("7. another output on a coarser grid (32³ in the same box): its nearest cell, x = 11.00 Mpc",
              ok and x.current.prefix == "output" and at() == (0, 5) and "x = 11.00 Mpc" in x.position.text(),
              f"{at()} {x.position.text()}")
        d50 = x.current
        ok = pick(x.prefix_combo, str(p50.root))
        check("7. ... and back: the place remembered, not re-read off the coarser grid (cell 10 again, not 11)",
              ok and at() == (0, 10), f"{at()}")
        pick(x.prefix_combo, str(d50.root))
        x.index.setValue(7)                               # its centre, 15 Mpc, is a face between two 64³ cells
        shown()
        ok = pick(x.snap_combo, 99)
        check("7. a place on a face of the finer grid's cells takes the upper cell (another redshift too)",
              ok and x.current.snap == 99 and at() == (0, 15), f"{at()}")
        ok = pick(x.sim_combo, sim2)
        check("7. another simulation: the middle of its box (the axis stays, as before)",
              ok and x.current.sim == sim2 and at() == (0, N // 2) and "x = 16.25 Mpc" in x.position.text(),
              f"{at()} {x.position.text()}")
        ok = pick(x.sim_combo, SIM)
        check("7. ... and back to the first simulation: the middle again, not the place left there",
              ok and x.current.sim == SIM and x.current.snap == 99 and at() == (0, N // 2), f"{at()}")

        # ---- the zoom window: kept across outputs and redshifts where it can be shown, said where it cannot
        f99 = fields(3.0)
        x._zoomed(zoom_req(), f99)                        # a zoom of snapshot 099 lands (and is remembered)
        z99 = x.is_zoomed() and x._zoom_fields is f99
        said.clear()
        ok = pick(x.snap_combo, 50)                       # its 'output' (the prefix chosen last): no zoom of it yet
        st = x.zoom_state.text()
        check("7. zoom window kept, not shown where there is no zoom of it (no memory, no server): the slice map at "
              "the kept place, and the zoom and status lines say which window and why",
              z99 and ok and not x.is_zoomed() and x.current.prefix == "output" and at() == (0, N // 4)
              and x._plane is not None and x._plane.shape == (32, 32) and st.startswith("zoom window y 20.00–30.00, "
              "z 40.00–50.00 Mpc kept") and "query server" in st and st in said, f"{z99} {at()} {st!r} {said}")
        ok = pick(x.snap_combo, 99)
        check("7. ... back on the snapshot zoomed: its zoom of the window, from memory",
              ok and x.is_zoomed() and x._zoom_fields is f99 and x._zoom["a0"] == (20.0, 30.0)
              and x._zoom["a1"] == (40.0, 50.0) and x.zoom_time.text() == "from memory", x.zoom_state.text())
        import figsave as FS                              # a saved zoom's title in the axes' unit
        h99, z = G.hubble(x.current), x._zoom
        zt = x._map_title(lambda a_, v_: FS.position(x.current, a_, v_, "mpc_h", h99))
        r0, r1 = (E.AXES[k] for k in z["rest"])
        want = (f"{r0} {20 * h99:.2f}–{30 * h99:.2f}, {r1} {40 * h99:.2f}–{50 * h99:.2f} Mpc/h — "
                f"{E.AXES[z['axis']]} = {x._slice_position() * h99:.2f} Mpc/h")
        check("8. a zoom's saved title in Mpc/h: its window and the slice both converted (the screen's in Mpc)",
              want in zt and f"{r0} 20.00–30.00, {r1} 40.00–50.00 Mpc — " in x._map_title(), f"{zt!r} {want!r}")
        # the query server running on the other output: the window is evaluated there (the request stood in for)
        sent = []
        spy = sent.append
        x._to_zoom.disconnect(x._server.zoom)
        x._to_zoom.connect(spy)
        x._server_running, x._server_output = True, x._by_root[str(d50.root)]
        try:
            ok = pick(x.snap_combo, 50)
            r = sent[0] if sent else {}
            check("7. ... with the query server on the other output: the same window evaluated there, on its kept "
                  "slice (the cell centre x = 33.00 Mpc of the 32³ grid)",
                  ok and len(sent) == 1 and r.get("output") is x.current and r.get("index") == N // 4 == x.index.value()
                  and r.get("a0") == (20.0, 30.0) and r.get("a1") == (40.0, 50.0) and r.get("res") == 8
                  and abs(float(r["coords"][0][0]) - 33.0) < 1e-9 and x._zoom_pending, f"{sent}")
            if r:
                f50 = fields(5.0)
                x._zoomed(r, f50)                         # its answer
                check("7. ... and shown when it arrives", x.is_zoomed() and x._zoom_fields is f50)
        finally:
            x._server_running, x._server_output = False, None
            x._to_zoom.disconnect(spy)
            x._to_zoom.connect(x._server.zoom)
        # 'exact' ticked: a switch never starts an exact run nor asks about one (stepping through
        # redshifts would otherwise ask, or run PS-DTFE over a whole snapshot, at every step); a drag does
        calls, asked = [], []
        x._exact_zoom = lambda *a: calls.append(a)
        x.confirm_exact = lambda text: (asked.append(text), False)[1]
        x.zoom_exact.setChecked(True)
        try:
            ok = pick(x.prefix_combo, str(p50.root))      # ps_output of 050: no zoom of the window in memory
            st = x.zoom_state.text()
            check("7. ... 'exact' ticked: no exact run started or asked about by the switch, the window kept and the "
                  "zoom line says to drag it again",
                  ok and not calls and not asked and not x.is_zoomed() and at() == (0, N // 2)
                  and st.startswith(E.KEPT_ZOOM_TEXT) and "drag the window again" in st, f"{calls} {st!r}")
        finally:
            del x._exact_zoom
            x.confirm_exact = None
            x.zoom_exact.setChecked(False)
        ok = pick(x.prefix_combo, str(d50.root))
        check("7. ... flipping back to an output zoomed a moment ago: its zoom from memory at once",
              ok and x.is_zoomed() and x._zoom_fields is f50 and x.current.prefix == "output")
        pick(x.prefix_combo, str(p50.root))               # not shown here: said in the zoom line
        x._dragged(2, 2, 10, 10)                          # no server, 'exact' off: no zoom made
        check("7. a drag that makes no zoom (no server, 'exact' off) leaves the kept window and its line",
              bool(x._keep.get("zoom")) and x.zoom_state.text().startswith(E.KEPT_ZOOM_TEXT), x.zoom_state.text())
        x.index.setValue(x.index.value() + 1)             # a slice chosen by hand: the window goes with the old one
        shown()
        gone = x._keep.get("zoom") is None and x.zoom_state.text() == ""
        ok = pick(x.prefix_combo, str(d50.root))
        check("7. a slice chosen by hand ends the kept window (and its line): the next output shows its map",
              gone and ok and not x.is_zoomed() and x.zoom_state.text() == "", x.zoom_state.text())
        x._zoomed(zoom_req(), fields(6.0))
        x.zoom_out()
        ok = pick(x.prefix_combo, str(p50.root))
        check("7. Zoom out ends it too: the whole slice on the next output, nothing said",
              ok and not x.is_zoomed() and x._keep.get("zoom") is None and x.zoom_state.text() == "")
        x._zoomed(zoom_req(), fields(7.0))
        zoomed = x.is_zoomed()
        ok = pick(x.sim_combo, sim2)
        check("7. another simulation with a zoom on screen: the whole box at the middle, the window dropped",
              zoomed and ok and not x.is_zoomed() and x._keep.get("zoom") is None and at() == (0, N // 2)
              and x.zoom_state.text() == "", f"{at()} {x.zoom_state.text()!r}")

        # ---- the custom runs (and opened files): kept in a box of the same size, else the middle
        ok = pick(x.sim_combo, E.GROUP_CUSTOM)            # its last run: c3s (32³ in a 32 Mpc box)
        check("7. the custom snapshots (another entry of the first menu): the middle",
              ok and str(x.current.root) == str(c3s) and at() == (0, 16), f"{x.current.root} {at()}")
        ok = pick(x.snap_combo, str(c3a))
        check("7. a custom run in a box of another size (64 against 32 Mpc): the middle", ok and at() == (0, N // 2))
        x.axis.setCurrentIndex(1)
        x.index.setValue(20)
        shown()
        x._zoomed(zoom_req(), fields(8.0))
        zoomed = x.is_zoomed()
        ok = pick(x.snap_combo, str(c2d))
        st = x.zoom_state.text()
        check("7. a 2D run on the way: its one plane (no axis, no slice), the window not shown and why",
              zoomed and ok and x.current.dim == 2 and not x.axis.isEnabled() and at() == (2, 0)
              and not x.is_zoomed() and st.startswith("zoom window x 20.00–30.00, z 40.00–50.00 Mpc kept")
              and "2D output" in st, f"{at()} {st!r}")
        cb = x._by_root[str(c3b)]
        fb = fields(9.0, ax=1)
        memo(cb, 1, 10, fb)                               # (a zoom of c3b made before)
        ok = pick(x.snap_combo, str(c3b))
        check("7. ... the next 3D run in the same box gets the slice back (axis y, the nearest 32³ cell) and the window",
              ok and at() == (1, 10) and x.is_zoomed() and x._zoom_fields is fb, f"{at()} {x.zoom_state.text()!r}")
        ok = pick(x.snap_combo, str(c3s))
        check("7. ... and a run in a box of another size: the middle, the whole box",
              ok and at() == (1, 16) and not x.is_zoomed() and x._keep.get("zoom") is None)
        ok = x.open_file(str(o1) + ".a_den") and shown(o1)
        check("7. an opened file (another entry of the first menu): the middle", ok and at() == (1, N // 2))
        x.index.setValue(40)
        shown()
        ok = x.open_file(str(o2) + ".a_den") and shown(o2)
        check("7. ... another opened file in the same box: the same place (y = 40.50 Mpc: cell 20 of 32)",
              ok and at() == (1, 20), f"{at()}")
        ok = x.open_file(str(o3) + ".a_den") and shown(o3)
        check("7. ... one whose box is unknown (cells, another grid): the middle", ok and at() == (1, 8), f"{at()}")
        texts = said + lines + [E.KEPT_ZOOM_TEXT]
        check("7. the kept slice's texts carry no '--'", not [t for t in texts if "--" in t], f"{texts}")
    finally:
        x.shutdown()


def volume_rule_pure():
    """11. one rule for the values a map shows (grids.OutputSet.component_values): the 3D volume's cube at one grid
    cell a block is the map's slices themselves, bit for bit -- a vector's norm and components, the vorticity's curl,
    a tensor's trace and one of its components, a scalar; with the velocity factor and without (z = 0). The averaging's
    'done' counted only where asked (volume3d._run's 'count'), one worker count for both pools."""
    import volume3d as VOL
    print("3D volume: the map's values")
    n = 12
    d = TMP / "vol_rule"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(4)
    raw = {}
    for sfx, nc in (("vel", 3), ("velVort", 3), ("velGrad", 9), ("den", 1)):
        raw[sfx] = rng.normal(size=(n, n, n, nc) if nc > 1 else (n, n, n)).astype(np.float32)
        raw[sfx].tofile(str(d / f"r.a_{sfx}"))
    same, curl = [], []
    for z in (0.0, 0.5):
        o = G.OutputSet(d / "r", box=((0.0,) * 3, (float(n),) * 3), redshift=z)
        for name, comps in (("velocity", ["norm", "x", "z"]), ("vorticity", ["norm", "x", "y", "z"]),
                            ("gradient", ["trace (divergence)", "xy", "zz"]), ("density", ["value"])):
            for comp in comps:
                VOL.drop_volumes()
                cube = VOL.average_window(o, name, comp, o.lo, o.lo + o.length, n)["values"]
                ref = np.stack([o.slice(name, 0, i, comp) for i in range(n)]).astype(np.float32)
                same.append(cube.tobytes() == ref.tobytes())
        f = o.velocity_factor("vorticity")
        w = raw["velVort"].astype(np.float64)               # [W_xy, W_xz, W_yz] -> the curl 2 (-W_yz, W_xz, -W_xy)
        curl.append(np.array_equal(o.slice("vorticity", 0, 5, "x"), -2.0 * w[5, ..., 2] * f)
                    and np.array_equal(o.slice("vorticity", 0, 5, "z"), -2.0 * w[5, ..., 0] * f)
                    and (f == 1.0) == (z == 0.0))
    check("11. one rule for the map and the volume (OutputSet.component_values): the volume at one cell a block is the "
          "map's slices bit for bit -- velocity norm and components, the vorticity as the curl, the gradient's trace "
          "and a component, the density -- with the velocity factor (z = 0.5) and without (z = 0)",
          all(same) and len(same) == 22 and all(curl) and not hasattr(VOL, "field_values"), f"{same} {curl}")
    prog = {}
    VOL.drop_volumes()
    avg = VOL.average_window(o, "velocity", "norm", o.lo, o.lo + o.length, 5, prog)
    averaged = dict(prog)
    VOL.prepare(dict(avg, key=None), False, prog)
    check("11. the averaging counts its blocks done ('done' of 'total', volume3d._run with 'count'); the colouring that "
          "follows on the same progress leaves 'done' alone; both pools have WORKERS threads",
          averaged == {"done": 5, "total": 5} and prog.get("done") == 5
          and VOL.pool()._max_workers == VOL.gui_pool()._max_workers == VOL.WORKERS, f"{averaged} {prog}")


def planes3d_alone():
    """10. explore_3d.planes3d without explore imported (grids.smoothed: no import cycle), in a process of its own."""
    print("3D planes without explore")
    d = TMP / "vol_rule"
    code = "\n".join([
        "import os, sys",
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'",
        f"sys.path.insert(0, {str(ROOT / 'python' / 'gui')!r})",
        "import numpy as np",
        "import grids as G",
        "import explore_3d as X3",
        f"o = G.OutputSet({str(d / 'r')!r}, box=((0.0,) * 3, (12.0,) * 3), redshift=0.0)",
        "req = dict(output=o, field='density', component='value', cmap='magma', log=False, clip=1.0, smooth=1.5,",
        "           symmetric=False, ps_gradient='stored', index=[3, 4, 5], shown=[True, True, True],",
        "           lo=o.lo, hi=o.lo + o.length)",
        "res = X3.planes3d(req)",
        "z = res['planes'][2]['values']",
        "ok = np.allclose(z, G.smoothed(o.slice('density', 2, 5, 'value'), 1.5))",
        "print('PLANES', 'explore' in sys.modules, sum(p is not None for p in res['planes']), ok,",
        "      not np.allclose(z, o.slice('density', 2, 5, 'value')))",
    ])
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PLANES")), "")
    check("10. explore_3d.planes3d runs with explore not imported: three planes, smoothed by grids.smoothed (the map's "
          "Gaussian, periodic)", p.returncode == 0 and line == "PLANES False 3 True True",
          f"{line!r} {p.stderr.strip()[-400:]}")


def planes3d_inside():
    """12. the planes with the eye inside the box, on a view of its own: the void cuts on planes that pass the eye,
    and what is picked under the cursor."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (3D planes from inside): PySide6 is not importable ({e})")
        return
    from PySide6.QtGui import QImage
    import explore_3d as X3
    import view3d as V
    print("3D planes from inside the box")
    qa = QApplication.instance() or QApplication([])
    L, m, W, H, nv = 100.0, 64, 640, 480, 10
    rng = np.random.default_rng(5)
    centres, radii = rng.uniform(0.2 * L, 0.8 * L, (nv, 3)), rng.uniform(6.0, 16.0, nv)
    voids = {"centres": centres, "M": V.sphere_matrices(radii), "reach": radii, "rows": np.arange(nv),
             "labels": [str(k) for k in range(nv)], "r_eff": None, "box": L}
    pos = np.array([47.0, 53.0, 50.0])
    t = np.linspace(0.0, 2.0 * np.pi, 36001)[:-1]

    def grown(mask, r):
        o = np.zeros_like(mask)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r + 1:
                    continue
                ys, yd = (slice(dy, None), slice(None, H - dy)) if dy >= 0 else (slice(None, dy), slice(-dy, None))
                xs, xd = (slice(dx, None), slice(None, W - dx)) if dx >= 0 else (slice(None, dx), slice(-dx, None))
                o[yd, xd] |= mask[ys, xs]
        return o

    def pixels():
        v.repaint()
        qa.processEvents()
        img = v.grab().toImage().convertToFormat(QImage.Format_RGB888)
        w, h = img.width(), img.height()
        a = np.frombuffer(img.constBits(), np.uint8, h * img.bytesPerLine()).reshape(h, img.bytesPerLine())
        a = a[:, :3 * w].reshape(h, w, 3).astype(np.int32)
        return a[::w // W, ::w // W]                        # (a device pixel ratio)

    def scene(shown):
        planes = []
        for a in range(3):
            if not shown[a]:
                planes.append(None)
                continue
            vals = 0.25 + 0.1 * rng.random((m, m))
            rgb, _ = G.colorize(np.ascontiguousarray(vals.T[::-1]), "inferno", log=False, symmetric=False,
                                vrange=(0.0, 1.0))
            planes.append({"rgb": np.ascontiguousarray(rgb), "values": vals, "lo2": (0.0, 0.0), "hi2": (L, L),
                           "pos": float(pos[a]), "stride": 1})
        return planes

    def cut_samples(shown):
        """Points on the voids' cuts with the planes shown (n, 3) and the plane of each (n,)."""
        P, A = [], []
        for a in range(3):
            if not shown[a]:
                continue
            r0, r1 = V.rest_axes(a)
            for c, r in zip(centres, radii):
                d = pos[a] - c[a]
                if abs(d) >= r:
                    continue
                rho = np.sqrt(r * r - d * d)
                p = np.zeros((len(t), 3))
                p[:, a], p[:, r0], p[:, r1] = pos[a], c[r0] + rho * np.cos(t), c[r1] + rho * np.sin(t)
                ok = (p[:, r0] >= 0) & (p[:, r0] <= L) & (p[:, r1] >= 0) & (p[:, r1] <= L)
                P.append(p[ok])
                A.append(np.full(int(ok.sum()), a))
        return np.concatenate(P), np.concatenate(A)

    def unhidden(cam, P, A, shown):
        """Not behind another plane (the planes opaque), nor too near one to say."""
        e = cam.eye()
        vis = np.ones(len(P), bool)
        for b in range(3):
            if not shown[b]:
                continue
            de, dp = e[b] - pos[b], P[:, b] - pos[b]
            s = de / (de - dp + 1e-300)
            X = e[None] + s[:, None] * (P - e[None])
            vis &= ~((A != b) & (de * dp < 0) & np.all((X >= 0) & (X <= L), axis=1))
            vis &= ~((A != b) & (np.abs(dp) < 0.6))
        return vis

    v = X3.View3D()
    v.resize(W, H)
    v.show()
    v.style, v.opacity = "planes", 1.0
    v._draw_void = lambda *a, **k: None                     # the cuts only
    cases = []
    for shown in ((False, False, True), (True, True, True)):    # eyes put near a cut of the z plane, looking along it
        for k in range(nv):
            d = pos[2] - centres[k, 2]
            if abs(d) >= radii[k]:
                continue
            rho = float(np.sqrt(radii[k] ** 2 - d * d))
            for phi, dist, pitch in ((0.3, 0.8, 8.0), (2.1, 1.3, 20.0), (4.0, 0.5, -12.0), (5.5, 2.0, 35.0)):
                tgt = np.array([centres[k, 0] + rho * np.cos(phi), centres[k, 1] + rho * np.sin(phi), pos[2]])
                cases.append((shown, (float(np.degrees(phi)) + 77.0, pitch), tgt, dist * rho))
    bad, passing, seen = [], 0, 0
    picks = wrong = nothing_ahead = 0
    try:
        for ci, (shown, turn, tgt, dist) in enumerate(cases):
            v.set_scene(scene(shown), np.zeros(3), np.full(3, L))
            v.set_perspective(True)
            v.cam.yaw, v.cam.pitch = turn
            v.fit()
            v.cam.target, v.cam.scale = tgt.copy(), v.cam.focal / dist
            v.set_voids(voids)
            with_cuts = pixels()
            v.set_voids(None)
            cuts = np.abs(with_cuts - pixels()).sum(axis=2) > 40
            cam = v.cam
            e = cam.eye()
            P, A = cut_samples(shown)
            xy, _depth = cam.project(P)
            front = np.isfinite(xy).all(axis=1)
            on = front & (xy[:, 0] >= 3) & (xy[:, 0] < W - 3) & (xy[:, 1] >= 3) & (xy[:, 1] < H - 3)
            px = np.where(on[:, None], xy, 0.0).astype(int)
            truth = np.zeros((H, W), bool)
            truth[px[on, 1], px[on, 0]] = True
            stray = cuts & ~grown(truth, 3)
            stray[:3] = stray[-3:] = stray[:, :3] = stray[:, -3:] = False
            stray[H - 60:] = False                              # the scale bar's text changes with nothing
            vis = on & unhidden(cam, P, A, shown)
            missed = vis & ~grown(cuts, 2)[px[:, 1], px[:, 0]]
            inside = bool(np.all((e > 0) & (e < L)))
            passing += inside and bool((~front).any())          # a cut that passes the eye's own plane
            seen += int(vis.sum())
            if stray.sum() > 6 or missed.sum() > max(4, 0.004 * vis.sum()):
                bad.append((ci, int(stray.sum()), int(missed.sum()), int(vis.sum())))
            if not all(shown):
                continue
            for x_, y_ in rng.uniform((5.0, 5.0), (W - 5.0, H - 5.0), (60, 2)):     # what is picked under a point
                dvec = cam.unproject(x_, y_) - e
                hits, unsure, behind = [], False, False
                for a in range(3):
                    if abs(dvec[a]) < 1e-12:
                        unsure = True
                        continue
                    s_ = (pos[a] - e[a]) / dvec[a]
                    pt = e + s_ * dvec
                    r0, r1 = V.rest_axes(a)
                    edge = min(pt[r0], L - pt[r0], pt[r1], L - pt[r1])
                    unsure |= abs(edge) < 0.05 or abs(s_) < 1e-9
                    if edge > 0 and s_ > 0:
                        hits.append((s_, a, pt))
                    behind |= edge > 0 and s_ < 0
                if unsure:
                    continue
                picks += 1
                got = v.point_at(float(x_), float(y_))
                nothing_ahead += behind and not hits
                if not hits:
                    wrong += got is not None
                    continue
                _s, a, pt = min(hits, key=lambda h_: h_[0])
                wrong += got is None or got[0] != a or float(np.abs(got[1] - pt).max()) > 0.5 * L / m + 1e-6
    finally:
        v.close()
    check("12. the planes from inside the box: the void cuts on a plane that passes the eye are drawn where they "
          "are (every cut pixel on a projected cut, every unhidden point of a cut drawn; eyes put next to a cut, "
          "one plane and all three)", not bad and len(cases) >= 16 and passing >= 8 and seen > 20000,
          f"bad (case, stray px, missed, of) {bad[:6]}; {len(cases)} cases, {passing} pass the eye, {seen} points")
    check("12. ... the point picked under the cursor is the nearest plane point in FRONT of the eye along that line "
          "of sight; none when the planes lie only behind it", wrong == 0 and picks > 300 and nothing_ahead >= 10,
          f"{wrong} wrong of {picks}; {nothing_ahead} with a plane only behind the eye")


def view3d_remembered():
    """10. the 3D view's choices across a restart (ExploreControls.state / restore_state): set in one window, restored
    into a fresh one; a state saved before them leaves the defaults."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"SKIP (3D view remembered): PySide6 is not importable ({e})")
        return
    import explore as E
    print("3D view remembered")
    QApplication.instance() or QApplication([])
    data = TMP / "data"
    if not (data / SIM).is_dir():
        make_snapdir(data, 50, 0.0, 7)
    made = []

    def fresh():
        view = E.ExploreView()
        x = E.ExploreControls(view, lambda: str(data), lambda: [])
        made.append(x)
        return x, view
    keys = ("view3d_style", "view3d_persp", "view3d_opacity", "view3d_detail", "vol3d_opacity")
    try:
        a, _va = fresh()
        defaults = {k: a.state()[k] for k in keys}
        a.view3d_style.setCurrentIndex(a.view3d_style.findData("planes"))
        a.view3d_persp.setChecked(False)
        a.view3d_opacity.setValue(60)
        a.view3d_detail.setCurrentIndex(a.view3d_detail.findData(512))
        a.vol3d_opacity.setValue(70)
        st = json.loads(json.dumps(a.state()))          # as the launcher keeps it: JSON in its QSettings
        b, vb = fresh()
        b.restore_state(st)
        got = b.state()
        check("10. the 3D view's choices are remembered across a restart: the style, perspective, the planes' opacity, "
              "the volume's detail and its opacity, restored into a fresh window with what follows from them (the "
              "planes' controls shown, the camera, the planes' opacity, the opacity's text); 'Show in 3D' is not",
              all(got[k] == st[k] for k in keys) and st["view3d_style"] == "planes" and st["view3d_persp"] is False
              and st["view3d_opacity"] == 60 and st["view3d_detail"] == 512 and st["vol3d_opacity"] == 70
              and defaults != {k: st[k] for k in keys} and not vb.view3d.cam.perspective and vb.view3d.opacity == 0.6
              and vb.view3d.style == "planes" and b.planes3d_box.isVisibleTo(b) and not b.volume3d_box.isVisibleTo(b)
              and b.vol3d_opacity_text.text() == a.vol3d_opacity_text.text() != "" and not b.view3d_toggle.isChecked()
              and not any("toggle" in k or "range" in k for k in st), f"{st} {got}")
        c, vc = fresh()
        old = {k: v for k, v in st.items() if k not in keys}
        c.restore_state(old)
        c.restore_state({**old, "view3d_style": "cubes", "view3d_detail": 333, "view3d_persp": "yes",
                         "view3d_opacity": None, "vol3d_opacity": "x"})
        check("10. ... a state saved before them (or holding values not offered) leaves today's defaults",
              {k: c.state()[k] for k in keys} == defaults and vc.view3d.cam.perspective
              and c.volume3d_box.isVisibleTo(c) and not c.planes3d_box.isVisibleTo(c), str(c.state()))
    finally:
        for x in made:
            x.shutdown()


def short_tooltips():
    """The Explore and 3D-view tooltips take at most two lines, as the job tabs' do: a long text's first sentence, the
    whole text the control's What's This (app_widgets.tip_auto). Checked over the map, the voids, the 3D view in both
    styles and the Save options dialog; a tooltip naming an output's files (its root + '.*') is a location, not prose."""
    try:
        from PySide6.QtWidgets import QApplication, QComboBox, QWidget
    except ImportError as e:
        print(f"SKIP (short tooltips): PySide6 is not importable ({e})")
        return
    import explore as E
    import explore_3d as X3
    import figsave as FS
    import runspec as rs
    from app_widgets import tip_lines
    print("Explore tooltips")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "data"
    if not (data / SIM).is_dir():
        make_snapdir(data, 50, 0.0, 7)
    E.VOID_CACHE_DIR = TMP / "void cache"
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(900, 760)
    view.show()
    x.show()
    real_procs = rs.dtfe_processes
    rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}
    long_tips, states = {}, []

    def collect(state, roots):
        states.append(state)
        for r in roots:
            for wd in [r] + r.findChildren(QWidget):
                texts = [wd.toolTip()] + ([str(wd.itemData(k, 3) or "") for k in range(wd.count())]
                                          if isinstance(wd, QComboBox) else [])
                for t in texts:
                    if tip_lines(t) > 2 and not (x.current is not None and str(x.current.root) in t):
                        long_tips.setdefault(t[:60], state)
    try:
        x.refresh_outputs()
        wait_for(qa, lambda: not x._busy and x._plane is not None, 30)
        collect("map", (x, view))
        x.voids.setChecked(True)
        wait_for(qa, lambda: x._void_proc is None and bool(x._void_cache), 180)
        collect("voids", (x, view))
        for style in ("volume", "planes"):
            x.view3d_toggle.setChecked(False)
            x.view3d_style.setCurrentIndex(x.view3d_style.findData(style))
            x.view3d_toggle.setChecked(True)
            if wait_for(qa, lambda: (x._3d_res or {}).get(style) is not None and not x._3d_busy, 60):
                collect(f"3D {style}", (x, view))
        x.view3d_toggle.setChecked(False)
        wait_for(qa, lambda: not x._busy, 30)
        dlg = FS.make_dialog(x, x.save_options, x.save_context())
        collect("Save options", (dlg,))
        dlg.deleteLater()
        whole = {x.view3d_toggle: X3.VIEW3D_TIP, x.view3d_style: X3.STYLE_TIP, x.view3d_detail: X3.DETAIL_TIP,
                 x.void3d_shape: X3.SHAPE_TIP, x.voids: E.VOID_TIP, x.ps_gradient: E.PS_GRADIENT_TIP,
                 x.grid_merge: E.GRID_MERGE_TIP, x.zoom_exact: x.zoom_exact.whatsThis()}
        check("every Explore and 3D-view tooltip takes at most two lines (the map, the voids, the 3D view's volume and "
              "planes, the Save options dialog)", not long_tips and len(states) == 5, f"{states} {long_tips}")
        # (the merge tick, greyed out for panels of different boxes, says so first)
        bad = [(w.toolTip(), w.whatsThis()[:60]) for w, t in whole.items()
               if not (w.whatsThis().endswith(t) and w.whatsThis().startswith(w.toolTip().replace("\n", " "))
                       and tip_lines(t) > 2 and tip_lines(w.toolTip()) <= 2)]
        check("... a long text's first sentence, the whole text its What's This (Help > What's This?)", not bad,
              str(bad[:2]))
    finally:
        rs.dtfe_processes = real_procs
        x.shutdown()


def explore_details():
    """The Explore pane's details: one 'Voids' control (its tick the row's label), the stream table's columns (the
    origin takes the room, the density's header whole), the Save buttons in the left pane beside the controls; the
    exact zoom's plan a plain function (the pane's method passes its choices)."""
    try:
        from PySide6.QtWidgets import QApplication, QCheckBox, QFormLayout, QHeaderView, QLabel
    except ImportError as e:
        print(f"SKIP (Explore details): PySide6 is not importable ({e})")
        return
    import explore as E
    import explore_zoom as XZ
    import runspec as rs
    print("Explore details")
    qa = QApplication.instance() or QApplication([])
    data = TMP / "data"
    if not (data / SIM).is_dir():
        make_snapdir(data, 50, 0.0, 7)
    view = E.ExploreView()
    x = E.ExploreControls(view, lambda: str(data), lambda: [])
    view.resize(900, 760)
    view.show()
    x.show()
    real_procs = rs.dtfe_processes
    rs.dtfe_processes = lambda max_age=2.0: {"runs": [], "servers": []}
    try:
        x.refresh_outputs()
        wait_for(qa, lambda: not x._busy and x._plane is not None, 30)
        form = x._out_form_map
        row, role = form.getWidgetPosition(x.voids)
        field = form.itemAt(row, QFormLayout.FieldRole)
        check("one 'Voids' control: its tick is the label of the row of its options (R_eff, sigma, Go to its centre); "
              "no other control or label is called voids",
              role == QFormLayout.LabelRole and field is not None and field.layout() is not None
              and field.layout().indexOf(x.void_reff) >= 0 and field.layout().indexOf(x.void_goto) >= 0
              and [w for w in x.findChildren(QCheckBox) if w.text().lower() == "voids"] == [x.voids]
              and not [w for w in x.findChildren(QLabel) if w.text().lower() == "voids"], f"{row} {role}")
        hdr = view.streams.horizontalHeader()
        fm = hdr.fontMetrics()
        check("the stream table: 'came from q' takes the room, the value column its contents, the 'ρ/ρ̄' header "
              "wider than its text", hdr.sectionResizeMode(4) == QHeaderView.Stretch and not hdr.stretchLastSection()
              and hdr.sectionResizeMode(5) == QHeaderView.ResizeToContents
              and hdr.sectionSize(1) >= fm.horizontalAdvance("ρ/ρ̄") + 2 * fm.averageCharWidth(),
              f"{hdr.sectionSize(1)} {fm.horizontalAdvance('ρ/ρ̄')}")
        check("the Save buttons (still the view's save_btn and save_opts_btn) sit in the left pane's Map group",
              x.isAncestorOf(view.save_btn) and x.isAncestorOf(view.save_opts_btn)
              and not view.isAncestorOf(view.save_btn) and view.save_btn.isVisibleTo(x) and view.save_btn.isEnabled())
        o = x.current
        args = (o, 2, 32, [0, 1], (10.0, 20.0), (30.0, 40.0), 64)
        m = x.exact_zoom_plan(*args)
        f = XZ.exact_zoom_plan(*args, None, x.cache_edit.text(), 0)
        roots = [Path(p["root"]).parent for p in (m, f) if p.get("root")]
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)

        def plain(p):
            return {k: repr(v).replace(str(p["root"]), "ROOT") if p.get("root") else repr(v)
                    for k, v in p.items() if k != "root"}
        check("the exact zoom's plan is a plain function (explore_zoom.exact_zoom_plan); the pane's method gives it the "
              "snapshot chosen, the cache folder and the running server's split: the same plan",
              plain(m) == plain(f) and (m["cmd"] is not None or m["note"]), f"{m['note']!r} {f['note']!r}")
    finally:
        rs.dtfe_processes = real_procs
        x.shutdown()


def main():
    try:
        pure()
        view3d_pure()
        volume_pure()
        volume_rule_pure()
        planes3d_alone()
        perspective_pure()
        mean_field_pure()
        volume_render()
        volume_gl_offscreen()
        volume_gl_parity()
        gui()
        view3d_gui()
        planes3d_inside()
        view3d_volume_gui()
        view3d_remembered()
        short_tooltips()
        explore_details()
        fold_map()
        keep_slice()
        mean_field_gui()
        face_gradient_zoom()
        save_settings()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    if "--gl-child" in sys.argv[1:]:
        try:
            gl_child()
        finally:
            shutil.rmtree(TMP, ignore_errors=True)
    else:
        main()
