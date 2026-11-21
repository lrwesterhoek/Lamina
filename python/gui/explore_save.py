"""The Explore tab's Save image: SaveMixin, the part of ExploreControls that writes the map on screen -- the Save
options (remembered between sessions), the file dialog and the file name it offers (map_stem, grid_stem,
view3d_stem: what the picture shows), the decorated matplotlib figure or the map's own pixels with its contour lines,
arrows and void outlines, the table of the voids outlined, and the map's values as .npy or .csv."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

import contours as CT
import explore as _ex
import figsave as FS
import grids as G

# ---------------------------------------------------------------- the file name a save offers
# One piece per thing that tells a picture from another, joined by '_': simulation, field, redshift, the output's
# prefix, its grid, the plane and where it lies, the smoothing, log or lin, the overlays drawn. No Qt: a pure function
# of the request the picture was drawn from.
GRADIENT_PIECE = {"mean": "mean-field", "stream": "stream-average"}


def _piece(text) -> str:
    """One piece of a file name: letters, digits, '.', '_', '+' and '-' as they are, every run of anything else one
    '-' (none at the ends)."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", str(text)).strip("-.")


def _num(v, decimals: int = 2) -> str:
    """A number in a file name: fixed decimals, never '-0.00'."""
    s = f"{float(v):.{decimals}f}"
    return s.lstrip("-") if float(s) == 0 else s


def output_pieces(o) -> tuple[str, str, str, str]:
    """What names an output in a file name: (the simulation -- a custom output's folder --, the redshift 'z0.00' --
    'snap099' where only the snapshot number is known, '' where neither is --, the prefix, 'grid1024')."""
    when = f"z{_num(o.redshift)}" if o.redshift is not None else f"snap{int(o.snap):03d}" if o.snap is not None else ""
    return _piece(o.sim or o.dir.name), when, _piece(o.prefix), f"grid{int(o.n)}"


def name_piece(req: dict) -> str:
    """The field of a request and its component (not a scalar's 'value'): 'density', 'velocity-x'; a per-particle
    scalar by its dataset ('Potential', grids.scalar_dataset) where the run names one."""
    field = req["field"]
    if field == "scalar" and req.get("output") is not None:
        field = G.scalar_dataset(req["output"]) or field
    parts = [field] + ([] if req.get("component", "value") in ("value", "") else [req["component"]])
    return "-".join(q for q in (_piece(p) for p in parts) if q)


def family_of(req: dict) -> str:
    """Which velocity derivative the map of a request shows ('' another field, 'mean', 'stream'): the output's
    (grids.OutputSet.gradient_family, by the request's 'ps_gradient'); a hi-res slice is a point evaluation, the
    stream average."""
    fam = req["output"].gradient_family(req["field"], req.get("ps_gradient", "stored"))
    return "stream" if fam and req.get("pts") else fam


def zoom_family(req: dict, zoom: dict) -> str:
    """The same for a zoom of that map (ZoomMixin._render_zoom's rule): a point evaluation, so a phase-space output's
    stream average, but the mean field in an exact zoom of an output that stored it; standard DTFE's the mean
    field."""
    o = req["output"]
    if req["field"] not in G.GRADIENT_FIELDS:
        return ""
    ps = G.estimator_label(o) == "PS-DTFE"
    return "stream" if ps and not (zoom.get("exact") and G.zoom_mean_field(o)) else "mean"


def field_piece(req: dict, family: str | None = None) -> str:
    """The field of a request as a file name says it: name_piece, and for a phase-space output's velocity
    derivatives the family shown, 'mean-field' or 'stream-average' ('family', else family_of the request)."""
    fam = family_of(req) if family is None else family
    ps = bool(fam) and G.estimator_label(req["output"]) == "PS-DTFE"
    return name_piece(req) + (f"-{GRADIENT_PIECE[fam]}" if ps else "")


def smooth_piece(req: dict) -> str:
    """'smooth0', 'smooth1.5' (the Gaussian's sigma in cells), 'smooth1.5-3D' when smoothed in 3D (never a hi-res
    slice: its points are smoothed in their plane)."""
    s = float(req.get("smooth", 0.0))
    return f"smooth{s:g}" + ("-3D" if s > 0 and req.get("smooth3d") and not req.get("pts") else "")


def _join(pieces) -> str:
    return "_".join(p for p in pieces if p)


def map_stem(req: dict, geometry=None, zoom: dict | None = None, pts_side: dict | None = None,
             overlays=()) -> str:
    """The file name (without suffix) Save image offers for the map drawn from request 'req', e.g.
    'TNG100-3-Dark_density_z0.00_ps_output1024_grid1024_xy-plane_z55.52Mpc_smooth0_log_contours': the simulation,
    the field (field_piece), the redshift, the output's prefix (and the output a difference map compares with),
    the grid, the plane and where the slice lies (Mpc; the cell without a box; nothing for a 2D output), a hi-res
    slice's points or a zoom's window and points ('geometry' = the map's _geometry(), 'zoom' its zoom), the
    smoothing (not a zoom's: its points are drawn as they are), log or lin (not for a comparison: its mode says),
    and 'overlays', the names of what is drawn over the map."""
    o = req["output"]
    sim, when, prefix, grid = output_pieces(o)
    axes = _ex.AXES
    zoomed = zoom is not None and geometry is not None
    versus, scale = [], ["log" if req.get("log") else "lin"]
    b = req.get("compare")
    if b is not None:                           # what differs of B, after 'minus' or 'log-ratio-to'
        bsim, bwhen, bprefix, bgrid = output_pieces(b)
        versus = ["minus" if req.get("compare_mode") == "diff" else "log-ratio-to"] \
            + [p for p, mine in ((bsim, sim), (bwhen, when)) if p != mine] + [bprefix] \
            + ([bgrid] if bgrid != grid else [])
        scale = []
    place = []
    if req.get("pts") and pts_side is not None:     # a hi-res slice: its own plane and points
        sd = pts_side
        place = [f"{sd['u_axis']}{sd['v_axis']}-plane", f"{sd['axis']}{_num(sd.get('center', 0.0))}Mpc",
                 f"hires{int(sd['nu'])}x{int(sd['nv'])}"]
    elif o.dim == 3:
        axis = int(req["axis"])
        rest = [a for a in range(3) if a != axis]
        pos = o.lo[axis] + (req["index"] + 0.5) * o.length[axis] / o.n
        place = [f"{axes[rest[0]]}{axes[rest[1]]}-plane",
                 f"{axes[axis]}{_num(pos)}Mpc" if o.box is not None else f"{axes[axis]}-cell{int(req['index'])}"]
    if zoomed:
        _axis, _index, rest, lo, hi, n0, _n1 = geometry
        unit = "Mpc" if o.box is not None else "cells"
        place += [f"zoom-{axes[rest[0]]}{_num(lo[0])}to{_num(hi[0])}-{axes[rest[1]]}{_num(lo[1])}to{_num(hi[1])}{unit}",
                  f"{int(n0)}pts" + ("-exact" if zoom.get("exact") else "")]
    field = field_piece(req, zoom_family(req, zoom) if zoomed else None)
    return _join([sim, field, when, prefix, *versus, grid, *place, "" if zoomed else smooth_piece(req), *scale,
                  *[_piece(v) for v in overlays]])


def view3d_stem(req: dict) -> str:
    """The file name (without suffix) offered for the 3D view drawn from its request: simulation, field (the volume
    is of the stored grid: its family, whatever the map's choice), redshift, prefix, grid, '3D-volume' with the
    detail it is drawn at or '3D-planes' with the smoothing, log or lin, 'contours' when the planes' lines or the
    volume's shells are on."""
    sim, when, prefix, grid = output_pieces(req["output"])
    volume = req.get("style") == "volume"
    return _join([sim, field_piece(dict(req, ps_gradient="stored") if volume else req), when, prefix, grid,
                  "3D-volume" if volume else "3D-planes",
                  f"detail{int(req['detail'])}" if volume and req.get("detail") else "",
                  "" if volume else smooth_piece(req), "log" if req.get("log") else "lin",
                  "contours" if req.get("contours") else ""])


def grid_stem(req: dict) -> str:
    """The file name (without suffix) offered for the figure grid of request 'req' (grid_request): what every panel
    shares of simulation, field (and, with a phase-space panel among them, the derivatives' family), redshift, prefix
    and grid (a piece the panels differ in is left out), 'figure-grid-2x3', the plane and where the slice lies (the
    map's place; how far through the box where the request does not say), the smoothing, log or lin where every
    panel agrees, 'contours' when ticked and a panel can have lines."""
    panels = [p for p in req["panels"] if p]
    if not panels:
        return "figure-grid"

    def shared(values) -> str:
        values = list(values)
        return values[0] if all(v == values[0] for v in values) else ""
    named = [output_pieces(p["output"]) for p in panels]
    sim, when, prefix, grid = (shared(n[i] for n in named) for i in range(4))
    asked = [dict(p, ps_gradient=req.get("ps_gradient", "stored")) for p in panels]
    field = shared(name_piece(p) for p in asked)
    fam = shared(family_of(p) for p in asked)
    if field and fam and any(G.estimator_label(p["output"]) == "PS-DTFE" for p in panels):
        field += f"-{GRADIENT_PIECE[fam]}"
    axes, axis = _ex.AXES, int(req["axis"])
    rest = [a for a in range(3) if a != axis]
    place = []
    if all(p["output"].dim == 3 for p in panels):
        at = req.get("position_at")
        boxed = all(p["output"].box is not None for p in panels)
        where = f"{axes[axis]}{round(100 * req['rel'])}pct" if at is None else \
            f"{axes[at[0]]}{_num(at[1])}Mpc" if boxed else f"{axes[axis]}-cell{int(req['index'])}"
        place = [f"{axes[rest[0]]}{axes[rest[1]]}-plane", where]
    lines = req.get("contours") and any(not CT.why_not(p["field"], p.get("cmap", "")) for p in panels)
    return _join([sim, field, when, prefix, grid, f"figure-grid-{int(req['rows'])}x{int(req['cols'])}", *place,
                  smooth_piece(req), shared("log" if p["log"] else "lin" for p in panels),
                  "contours" if lines else ""])


class SaveMixin:
    # ---------------------------------------------------------------- saving: the Save options
    def set_save_options(self, opts: dict, remember: bool = True) -> None:
        """Use these Save options from now on; remembered between sessions (without a typed title) when the launcher
        remembers its settings."""
        self.save_options = FS.options(opts)
        if remember and self._settings is not None:
            self._settings.setValue(self.SAVE_KEY, FS.dumps(self.save_options))
        self._save_tips()

    def _save_tips(self):
        d = FS.describe(self.save_options)
        self.view.save_btn.setToolTip(f"Save the map ({d}; Save options… to change), or its values as .npy or .csv")
        self.grid_save_btn.setToolTip(f"Write the composed figure ({d}; Options… to change)")

    def _map_title(self, units=None) -> str:
        """The single map's own title, as the window shows it (the saved figure's title unless one was typed); with
        'units' ((axis, position) -> (value, unit name)) the same title with its slice position and zoom window in
        that unit, a saved figure's axes'."""
        html = self.view.title.text() if units is None or self._title_make is None else self._title_make(units)
        return html.replace("<b>", "").replace("</b>", "")

    def save_context(self) -> dict:
        """What the Save options dialog needs to know of the picture on screen: the grid or a map, arrows and outlines
        shown, the box's h (Mpc/h offered only when known), a box at all, the figure's own title, the full-resolution
        image size in pixels."""
        ctx = {"grid": bool(self._grid_shown), "arrows": False, "voids": False, "hubble": None, "box": True,
               "auto_title": "", "pixels": None}
        if self._grid_shown:
            req = self.grid_request()
            panels = [p for p in (req or {}).get("panels", []) if p]
            hs = [G.hubble(p["output"]) for p in panels]
            ctx["hubble"] = hs[0] if hs and all(hs) else None
            ctx["box"] = all(p["output"].box is not None for p in panels)
            c = self.view.canvas
            ctx["voids"] = bool(c.circles or c.dashed or c.marks)
            ctx["auto_title"] = self._grid_title
            if req is not None and panels:
                T = max(p["output"].n for p in panels)
                gap = 0 if req["merged"] else 6
                ctx["pixels"] = (req["cols"] * T + gap * (req["cols"] - 1), req["rows"] * T + gap * (req["rows"] - 1))
        elif self._req is not None and self._plane is not None:
            o = self._req["output"]
            ctx["arrows"] = self._arrows is not None and self._arrows.get("vmax", 0) > 0
            ctx["voids"] = self._void_result() is not None and self.view.canvas.circles is not None
            ctx["hubble"] = G.hubble(o)
            ctx["box"] = o.box is not None
            ctx["auto_title"] = self._map_title()
            ctx["pixels"] = (int(self._plane.shape[0]), int(self._plane.shape[1]))    # plane rows = horizontal axis
        return ctx

    def open_save_options(self) -> bool:
        """The Save options dialog: OK keeps the options for every save from now on (remembered between sessions),
        Save… keeps them and saves the map or the figure grid now. False when cancelled."""
        dlg = FS.make_dialog(self, self.save_options, self.save_context(), can_save=self.can_save())
        if dlg.exec() != QDialog.Accepted:
            return False
        self.set_save_options(dlg.options())
        if dlg.save_now:
            self._save_image()
        return True

    def _ask_image_path(self, title: str, stem: str, extra: list | None = None):
        """The file dialog of a save: the image formats (the options' first) and 'extra' filters. Returns (path,
        format or the extra filter chosen) or (None, None); the format is the name's suffix, else the filter's. A
        format chosen here is the options' format from now on."""
        fmt0 = self.save_options["format"]
        images = FS.file_filters()
        order = [t for f, t in images if f == fmt0] + [t for f, t in images if f != fmt0] + list(extra or [])
        f, chosen = QFileDialog.getSaveFileName(self, title, str(Path.home() / f"{stem}.{FS.FORMATS[fmt0][1][0]}"),
                                                ";;".join(order))
        if not f:
            return None, None
        if chosen in (extra or []) or Path(f).suffix.lower() in (".npy", ".csv"):
            return f, chosen
        fmt = FS.format_of(f, next((k for k, t in images if t == chosen), fmt0))
        if fmt != fmt0:
            self.set_save_options(dict(self.save_options, format=fmt))
        return str(FS.with_format(f, fmt)), fmt

    def grid_save_stem(self) -> str:
        """The file name (without suffix) Save figure offers: the figure grid's own (grid_stem of the panels as they
        stand, shown or not); 'figure-grid' with no panel or a name that cannot be built."""
        try:
            req = self.grid_request()
            return grid_stem(req) if req is not None else "figure-grid"
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return "figure-grid"

    def save_stem(self) -> str:
        """The file name (without suffix) Save image offers for the picture on screen: the figure grid's
        (grid_save_stem), the 3D view's (view3d_stem, once the picture of its request is there) or the map's
        (map_stem, with the overlays a saved map will carry: the contour lines it shows, the arrows and void
        outlines the Save options keep); 'explore' with no picture or a name that cannot be built (a save never
        fails for its name)."""
        if self._grid_shown:
            return self.grid_save_stem()
        try:
            if self._3d_shown:
                req = (self._3d_res or {}).get("request")
                shown = req is not None and (req.get("style") != "volume" or self.view.view3d.volume is not None)
                return view3d_stem(req) if shown else "explore-3D"
            if self._req is None or self._plane is None:
                return "explore"
            ctx, opts = self.save_context(), self.save_options
            overlays = [name for name, on in (("contours", bool(getattr(self, "_contours", None))),
                                              ("arrows", ctx["arrows"] and opts["arrows"]),
                                              ("voids", ctx["voids"] and opts["voids"])) if on]
            return map_stem(self._req, self._geometry(), self._zoom if self.is_zoomed() else None, self._pts_side,
                            overlays)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return "explore"

    def _save_image(self):
        if self._grid_shown:
            self._save_grid_dialog()
            return
        if self._3d_shown:                      # the 3D view as on screen: a PNG of its pixels
            f, _chosen = QFileDialog.getSaveFileName(self, "Save the 3D view",
                                                     str(Path.home() / f"{self.save_stem()}.png"),
                                                     "PNG image (*.png)")
            if f:
                try:
                    self.status.emit(f"saved {self.save_3d_png(f if f.lower().endswith('.png') else f + '.png')}")
                except OSError as e:
                    self.status.emit(f"could not save {f}: {e}")
            return
        if self._plane is None:
            return
        void_filter = "CSV table of the voids outlined (*.csv)"
        f, chosen = self._ask_image_path("Save the map", self.save_stem(), ["NumPy array of the values (*.npy)",
                                                                            "CSV table: x, y, value (*.csv)",
                                                                            void_filter])
        if not f:
            return
        if chosen == void_filter:
            try:
                self.status.emit(f"saved {self.save_voids_csv(f)}")
            except OSError as e:
                self.status.emit(f"could not save {f}: {e}")
            return
        if Path(f).suffix.lower() in (".npy", ".csv"):
            try:
                self.status.emit(f"saved {self.save_values(f)}")
            except OSError as e:
                self.status.emit(f"could not save {f}: {e}")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            written = self.save_png(f)
            self.status.emit("saved " + ", ".join(written) + f" ({FS.describe(self.save_options)})"
                             + (f"; {self._save_note}" if self._save_note else ""))
        except FS.TooLarge as e:                # not the screen instead: the options asked for too much
            self.status.emit(f"could not save {f}: {e}")
        except Exception as e:
            if FS.format_of(f) in ("png", "tiff"):
                self.view.canvas.grab().save(f)
                self.status.emit(f"saved {f} as the screen shows it (the figure failed: {e})")
            else:
                self.status.emit(f"could not save {f}: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def save_png(self, f, options: dict | None = None) -> list[str]:
        """The map as a file, as the Save options say ('options', else the ones chosen; until the dialog is used,
        the default figure: a matplotlib PNG at 200 dpi with title, axes, colour bar, arrows and the void
        outlines as on screen), in the format of the file's suffix (.png, .pdf, .svg, .tif). At full resolution,
        and with no margins and 'match the data' (figsave.writes_pixels): the map's own pixels
        (_write_full_map). The contour lines the map shows are in every one of them. With outlines drawn,
        '<name>_voids.csv' beside it: the table of those voids. The paths."""
        opts = FS.options(self.save_options if options is None else options)
        fmt = FS.format_of(f, opts["format"])
        written = [str(f)]
        self._save_note = ""
        if FS.writes_pixels(opts):
            self._write_full_map(f, fmt, opts)
        else:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            with matplotlib.rc_context(FS.rc(opts, fmt)):    # the font size and the text kept as text
                fig, dpi = self.map_figure(opts)
                try:
                    FS.savefig(fig, f, opts, dpi, fmt)     # the pixel guard, the titles' line breaks, savefig
                finally:
                    plt.close(fig)
        if opts["voids"] and self.voids_shown():        # (a name that ends in '_voids' already: not said twice)
            written.append(self.save_voids_csv(Path(f).with_name(Path(f).stem.removesuffix("_voids") + "_voids.csv")))
        return written

    def map_figure(self, options: dict | None = None):
        """The decorated map as a pyplot figure and the dpi to save it at (the options' dpi, or the one matching the
        data): title, axes (frame, ticks, tick labels, axis names in the unit chosen), colour bar and its label, the
        arrows and void outlines as on screen, each as the options say, and the contour lines the map shows; margins
        'none' = the map alone filling the figure. The caller closes it (plt.close) and applies figsave.rc for a
        font scale."""
        opts = FS.options(self.save_options if options is None else options)
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        req = self._req
        o = req["output"]
        data = self._plane.T
        if req["log"]:
            data = np.log10(np.where(data > 0, data, np.nan))
        _, _, rest, glo, ghi, _, _ = self._geometry()    # a hi-res slice's axes are its file's, not the menu's
        h = G.hubble(o) if opts["unit"] == "mpc_h" else None
        um, self._save_note = FS.unit_map(o, rest, opts["unit"], h)
        ext = um.ext([glo[0], ghi[0], glo[1], ghi[1]])
        s = float(opts["font_scale"])
        bare = opts["margins"] == "none"
        if bare:                                # the map alone: its axes are the whole figure
            aspect = (ext[3] - ext[2]) / (ext[1] - ext[0]) if ext[1] > ext[0] and ext[3] > ext[2] else 1.0
            fig = plt.figure(figsize=(7, 7 * aspect))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
        lo, hi = self.view.colorbar.vrange
        kw = {"interpolation": "nearest"} if opts["match_data"] else {}
        if bare:
            kw["aspect"] = "auto"
        im = ax.imshow(data, origin="lower", extent=ext, cmap=req["cmap"], vmin=lo, vmax=hi, **kw)
        if not bare:
            FS.style_axes(ax, opts, (_ex.AXES[rest[0]], _ex.AXES[rest[1]]), um.name)
            if opts["title"]:
                # the slice position (and a zoom's window) in the axes' unit; in Mpc the screen's title
                auto = self._map_title()
                shown = auto if opts["unit"] == "mpc" else \
                    self._map_title(lambda a, v: FS.position(o, a, v, opts["unit"], h))
                t = ax.set_title(FS.title_text(opts, auto, shown), fontsize=8 * s)
                if opts["width"] > 0 or s != 1.0 or opts["match_data"]:
                    # a long title wraps rather than widening a figure of a set width or, in larger fonts (Slide),
                    # running over the colour bar's top tick label; the figure 'match the data'
                    # shrinks for a small map too
                    t.set_wrap(True)
            if opts["colorbar"]:
                if opts["colorbar_label"]:
                    cb = fig.colorbar(im, ax=ax, label=self.view.colorbar.label)
                else:
                    cb = fig.colorbar(im, ax=ax)
                # the contour levels across the bar, as on screen
                G.mark_levels_mpl(cb, getattr(self, "_contours", None), lo, hi, CT.LINE_WIDTH_PT)
        self._map_overlays(ax, opts, um, rest, glo, ghi, ext)
        try:
            FS.fit_width(fig, opts, opts["dpi"])
        except FS.TooLarge:                     # a width far too large: no figure left open
            plt.close(fig)
            raise
        dpi = FS.match_dpi(fig, [(ax, data.shape[1])]) if opts["match_data"] else opts["dpi"]
        return fig, dpi

    def _map_overlays(self, ax, opts: dict, um, rest, glo, ghi, ext) -> None:
        """The contour lines, the arrows and the void outlines over a saved map as on screen, in the coordinates of
        the unit map 'um' (the axes' unit): the lines whenever the map shows them (every preset, the Bare image too),
        the arrows and the outlines when the options keep them."""
        o = self._req["output"]
        lines = getattr(self, "_contours", None)
        if lines:
            n0, n1 = lines["shape"]
            (u0, v0), (u1, v1) = (float(glo[0]), float(glo[1])), (float(ghi[0]), float(ghi[1]))

            def to_xy(ij):                              # a sample (i, j) sits at its cell's or point's centre
                return np.column_stack([um.x(u0 + (ij[:, 0] + 0.5) / n0 * (u1 - u0)),
                                        um.y(v0 + (ij[:, 1] + 0.5) / n1 * (v1 - v0))])
            G.draw_contours_mpl(ax, lines["segments"], lines["colours"], to_xy, lw=CT.LINE_WIDTH_PT)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
        arrows = getattr(self, "_arrows", None)
        if opts["arrows"] and arrows is not None and arrows["vmax"] > 0:
            h = o.length / o.n                              # cell size per axis
            X, Y = np.meshgrid(um.x(o.lo[rest[0]] + arrows["c0"] * h[rest[0]]),
                               um.y(o.lo[rest[1]] + arrows["c1"] * h[rest[1]]), indexing="ij")
            ax.quiver(X, Y, arrows["u"], arrows["v"], color="white", edgecolor="black", linewidth=0.3,
                      angles="xy", scale_units="xy", pivot="middle",
                      scale=arrows["vmax"] / (0.9 * arrows["spacing"] * h[rest[0]] * um.k[0]), width=0.003)
        res = self._void_result()
        if opts["voids"] and res is not None and self.view.canvas.circles is not None:
            axis = self._geometry()[0]
            o = self.current
            half = 0.5 * float(o.length[axis]) / float(o.n)
            # the same voids as on screen (the Sizes range, the shape chosen), in Mpc, then in the axes' unit
            ov = G.void_overlay_mpc(self._void_set(res), axis, self._slice_position(), rest, glo, ghi, half)
            k0, k1 = um.k[0], um.k[1]
            ov["circles"] = [(um.x(u), um.y(v), r * k0, k) for u, v, r, k in ov["circles"]]
            ov["dashed"] = [(um.x(u), um.y(v), r * k0, k) for u, v, r, k in ov["dashed"]]
            ov["ellipses"] = [(um.x(u), um.y(v)) + G.scale_ellipse(rx, ry, a, k0, k1) + (k,)
                              for u, v, rx, ry, a, k in ov["ellipses"]]
            ov["marks"] = [(um.x(u), um.y(v), k, own) for u, v, k, own in ov["marks"]]
            G.draw_void_overlay_mpl(ax, ov)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])

    def _write_full_map(self, f, fmt: str, opts: dict) -> None:
        """The map at its own resolution: the screen's pixels, one per cell, zoom point or hi-res point (the colours
        and range as shown; a hi-res slice over 4096 points a side as shown, block-averaged), the contour lines over
        them when the map shows them, the arrows and outlines when the options keep them."""
        req = self._req
        rgb = None if self._grid_shown else self.view.canvas._buf
        if rgb is None or rgb.shape[:2] != self._plane.T.shape:     # not the map's own pixels: colour them again
            img = np.ascontiguousarray(self._plane.T[::-1])
            rgb, _ = G.colorize(img, req["cmap"], log=req["log"], clip=(req["clip"], 100 - req["clip"]),
                                symmetric=req.get("symmetric", False), vrange=self.view.colorbar.vrange)
        _, _, rest, glo, ghi, _, _ = self._geometry()
        ext = [glo[0], ghi[0], glo[1], ghi[1]]
        arrows = self._arrows is not None and self._arrows["vmax"] > 0 and opts["arrows"]
        voids = opts["voids"] and self._void_result() is not None and self.view.canvas.circles is not None
        lines = bool(getattr(self, "_contours", None))

        def overlay(ax):
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            self._map_overlays(ax, opts, FS.UnitMap(), rest, glo, ghi, ext)
        FS.write_pixels(f, rgb, fmt, overlay if (arrows or voids or lines) else None)

    def save_voids_csv(self, path) -> str:
        """The voids drawn on the map now as a CSV table (tools/void_outlines.write_csv: centre, every R_v the
        catalogue has, the catalogue's R_eff, central delta, resolved and distinct, and the drawn ellipsoid's a, b, c
        and R_ell: _void_csv_columns)."""
        res = self._void_result()
        rows = self.voids_shown()
        if res is None or not rows:
            raise OSError("no void outlines on the map (tick 'voids')")
        cols, note = self._void_csv_columns(res)
        return _ex._void_tool().write_csv(res, path, rows, cols, note)

    def save_values(self, path) -> str:
        """The map's VALUES (not its colours): '.npy' = the 2D array as drawn, rows along the vertical axis from
        the bottom (values[iy, ix]); '.csv' = one row per cell, 'x,y,value' at the cell centres in Mpc (cells
        without a box). The field's own units -- not log10 -- and the difference for a difference map."""
        path = Path(path)
        g = self._geometry()
        if g is None or self._plane is None:
            raise OSError("no map on screen")
        _axis, _index, rest, lo, hi, n0, n1 = g
        values = np.asarray(self._plane, dtype=np.float64).T            # [iy, ix], origin at the bottom
        if path.suffix.lower() == ".npy":
            np.save(path, values)
            return str(path)
        xs = lo[0] + (np.arange(n0) + 0.5) / n0 * (hi[0] - lo[0])
        ys = lo[1] + (np.arange(n1) + 0.5) / n1 * (hi[1] - lo[1])
        X, Y = np.meshgrid(xs, ys)
        unit = "Mpc" if self._req["output"].box is not None else "cells"
        head = f"{_ex.AXES[rest[0]]}_{unit},{_ex.AXES[rest[1]]}_{unit},value"
        np.savetxt(path, np.column_stack([X.ravel(), Y.ravel(), values.ravel()]), delimiter=",", header=head,
                   comments="", fmt="%.8g")
        return str(path)
