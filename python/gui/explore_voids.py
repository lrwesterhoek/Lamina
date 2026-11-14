"""The Explore tab's voids: VoidsMixin, the part of ExploreControls that finds a snapshot's voids (the analysis
pipeline's catalogue, through tools/void_outlines.py in a helper process, one at a time, cached per output and
smoothing), draws them over the map and the figure grid's panels -- as the sections of their ellipsoids at the mean
density (grids.drawn_ellipsoids, the default) or as the circles of their spheres of R_v -- and inspects the void
under the mouse or clicked (its entry, a jump to the slice through its centre). VoidSigmaBox is the smoothing box."""

from __future__ import annotations

import json
import time

import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtWidgets import QDoubleSpinBox, QMessageBox

import grids as G
import runspec as rs
from dtfelib.cli import SMOOTH_MPC_ENV_NAMES
from app_widgets import tip_auto

def _explore():
    """The explore module (its texts, the helper's path): imported when first needed, so this module also loads on its
    own (explore imports it while it is being loaded)."""
    import explore
    return explore


# The cost of a void search (a catalogue not cached yet), measured on TNG300-3-Dark 512^3 (cell 0.591 Mpc): about
# 10 s at the production 10 cells, 23 s at 4.1 cells (7477 minima), 225 s at 1.4 cells (115669 minima: an 18 MB
# result nobody can read on a map). Fitted as (n / 512)^3 (SEARCH_BASE_S + SEARCH_PER_S * sigma_cells^-SEARCH_POWER):
# the minima's count, and so the radii measured, grows as the smoothing shrinks. A cached catalogue takes a second.
SEARCH_BASE_S = 10.0
SEARCH_PER_S = 520.0
SEARCH_POWER = 2.6
SEARCH_ASK_S = 120.0            # a search estimated longer than this asks first
SIGMA_MIN_CELLS = 2.0           # the smallest smoothing offered: two cells of the shown grid
SIGMA_PAUSE_MS = 700            # the arrows of the smoothing box: a search starts this long after the last click
PRODUCTION_CELLS = 10.0         # the analysis scripts' production smoothing (config.PRODUCTION_SMOOTHING), before its
                                # catalogue says the exact value in Mpc


def search_seconds(n: int, sigma_cells: float) -> float:
    """A rough time for a void search on an n^3 grid at a smoothing of sigma_cells cells (the fit above)."""
    s = max(float(sigma_cells), 0.5)
    return (float(n) / 512.0) ** 3 * (SEARCH_BASE_S + SEARCH_PER_S * s ** -SEARCH_POWER)


def duration_text(seconds: float) -> str:
    """'about 20 s', 'about 4 min'."""
    if seconds < 90:
        return f"about {max(5, int(round(seconds / 5.0)) * 5)} s"
    return f"about {int(round(seconds / 60.0))} min"


class VoidSigmaBox(QDoubleSpinBox):
    """The smoothing the voids are found at, in Mpc: 0 is 'production' (the special value); a real value is at least
    two cells of the shown grid (or the production value, on a grid so coarse that it is less). The first step up from
    production lands on the production value itself, then each step is one cell; a step down from the smallest value
    goes back to production. 'stepped' tells the owner the change came from the arrows (it then waits for a pause
    before searching)."""

    def __init__(self):
        super().__init__()
        self.cell = 1.0                 # the shown grid's cell (Mpc)
        self.production = 10.0          # the production smoothing (Mpc)
        self.stepped = False

    def set_grid(self, cell: float, production: float):
        self.cell, self.production = float(cell), float(production)
        self.setSingleStep(self.cell)

    def smallest(self) -> float:
        return min(SIGMA_MIN_CELLS * self.cell, self.production)

    def stepBy(self, steps: int):
        v = self.value()
        for _ in range(abs(int(steps))):
            if steps > 0:
                v = self.production if v <= 0 else v + self.cell
            else:
                v = 0.0 if v - self.cell < self.smallest() - 1e-9 else v - self.cell
        self.stepped = True
        try:
            self.setValue(min(v, self.maximum()))
        finally:
            self.stepped = False


class VoidsMixin:
    # ---------------------------------------------------------------- the voids' choices
    @staticmethod
    def voids_applicable(o) -> str:
        """'' when the void outlines can be drawn over this output, else why not (the checkbox's tip): the
        pipeline's catalogue needs a simulation snapshot's full-box 3D grids with a density."""
        if o is None:
            return "no output shown"
        if not o.sim or o.snap is None:
            return "only for a simulation snapshot's grids (the analysis pipeline finds the voids per snapshot)"
        if o.dim != 3:
            return "only for 3D grids"
        if o.box is None or "density" not in o.files:
            return "needs the density grid and the snapshot's box"
        return ""

    def void_shape(self) -> str:
        """'ellipsoid' (the sections of the drawn ellipsoids: the default) or 'sphere' (R_v's circles)."""
        return self.void3d_shape.currentData() or "ellipsoid"

    def _void_radius_key(self) -> str:
        """The radius the Sizes range, the count and the 3D menu go by: the drawn shape's (R_ell or R_v)."""
        return "_r_ell" if self.void_shape() == "ellipsoid" else "_r_v"

    def _void_cache_key(self, o, sigma: float | None = None):
        """The catalogue's key: the files' root, their grid, the density file's stamp (a rerun rewrites it) and the
        smoothing (the committed value of the box: 0 = production; each sigma is another catalogue)."""
        try:
            stamp = o.files["density"].path.stat().st_mtime_ns
        except (KeyError, OSError):
            stamp = None
        return str(o.root), o.n, stamp, round(self._void_sigma_now if sigma is None else float(sigma), 6)

    def void_rows(self, res: dict, radius: str | None = None) -> np.ndarray:
        """The catalogue rows drawn: the distinct, well-resolved voids whose radius 'radius' (default the drawn
        shape's: R_ell of the ellipsoid, or R_v) lies in the Sizes range (a bound of 0 is none; a radius that is not defined
        passes only when there is no bound)."""
        keep = np.asarray(res["_shown"], dtype=bool).copy()
        r = np.asarray(res[radius or self._void_radius_key()], dtype=np.float64)
        lo, hi = self.void_rmin.value(), self.void_rmax.value()
        with np.errstate(invalid="ignore"):
            if lo > 0:
                keep &= r >= lo
            if hi > 0:
                keep &= r <= hi
        return np.nonzero(keep)[0]

    def _void_range_changed(self, *_):
        """Another Sizes range: the outlines, the 3D view and the figure grid's panels follow."""
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)

    def _void_reff_toggled(self, *_):
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)

    def _void_shape_changed(self, *_):
        """Ellipsoids or spheres: the map, the 3D view, the grid's panels, the other outline's name, the count."""
        self._void_other_text()
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)

    def _void_other_text(self):
        """The dashed outline's tick names its radius: R_eff beside R_v's spheres, R_v beside the ellipsoids."""
        ell = self.void_shape() == "ellipsoid"
        self.void_reff.setText("R_v too (dashed)" if ell else "R_eff too (dashed)")
        tip_auto(self.void_reff, _explore().VOID_RV_TIP if ell else _explore().VOID_REFF_TIP)

    # ---------------------------------------------------------------- the smoothing they are found at
    def _void_sigma_grid(self):
        """The smoothing box for the output shown: its cell (one step) and the production value (Mpc; its catalogue's
        own once known, else the production cells of this grid), the smallest value two cells; a value below that
        (a state saved before) raised to it."""
        o = self.current
        if o is None or o.box is None or not o.n:
            return
        cell = float(o.length[0]) / float(o.n)
        prod = self._void_cache.get(self._void_cache_key(o, 0.0)) or {}
        production = float(prod["sigma_mpc"]) if "sigma_mpc" in prod else PRODUCTION_CELLS * cell
        box = self.void_sigma
        box.set_grid(cell, production)
        v = box.value()
        if 0 < v < box.smallest() - 1e-9:
            box.blockSignals(True)
            box.setValue(box.smallest())
            box.blockSignals(False)
            self._void_sigma_now = box.value()

    def _void_sigma_edited(self, *_):
        """A value in the box: from the arrows, searched after a short pause (stepping on starts nothing yet); typed
        (Return, focus out) or set by the program, at once."""
        if self.void_sigma.stepped:
            self._void_sigma_pause.start()
            return
        self._void_sigma_pause.stop()
        self._void_sigma_commit()

    def _void_sigma_commit(self):
        """The smoothing committed: raised to two cells when below; then its own catalogue (cached by key, so going
        back costs nothing; a search that would take long is asked first, _voids_request: a 'no' puts the box back)."""
        self._void_sigma_pause.stop()
        box = self.void_sigma
        v = box.value()
        if 0 < v < box.smallest() - 1e-9:
            box.blockSignals(True)
            box.setValue(box.smallest())
            box.blockSignals(False)
            v = box.value()
        if abs(v - self._void_sigma_now) < 1e-9:
            return
        self._void_refused = ""
        self._void_sigma_back, self._void_sigma_now = self._void_sigma_now, v
        self._void_sigma_changed()

    def _void_sigma_restored(self):
        """A remembered smoothing set in the box (signals blocked): raised to the smallest value of this grid when below
        it, and committed -- the catalogue's key is the committed value, not the box's."""
        self._void_sigma_grid()
        box = self.void_sigma
        if 0 < box.value() < box.smallest() - 1e-9:
            box.blockSignals(True)
            box.setValue(box.smallest())
            box.blockSignals(False)
        self._void_sigma_now = box.value()

    def _void_refuse(self, o, sigma: float):
        """A long search declined: the box and the committed smoothing back to the value before (production when that
        one was the declined value too), its catalogue asked for, and the row saying so."""
        back = self._void_sigma_back if abs(self._void_sigma_back - sigma) > 1e-9 else 0.0
        self._void_sigma_back = 0.0
        box = self.void_sigma
        box.blockSignals(True)
        box.setValue(back)
        box.blockSignals(False)
        self._void_sigma_now = box.value()
        where = "production" if self._void_sigma_now <= 0 else f"{self._void_sigma_now:.3g} Mpc"
        self._void_refused = f"not started: the voids at σ {sigma:.3g} Mpc would take long (σ is back to {where})"
        self._voids_request(o)

    def _void_search_ok(self, o, sigma: float) -> bool:
        """Before a search at 'sigma' Mpc that is estimated to take longer than SEARCH_ASK_S: ask, in plain words
        (tests set confirm_void: callable(text) -> bool). True at once when it is quicker."""
        cells = sigma / (float(o.length[0]) / float(o.n))
        t = search_seconds(o.n, cells)
        if t <= SEARCH_ASK_S:
            return True
        text = (f"Finding the voids at σ = {sigma:.3g} Mpc ({cells:.1f} cells of this {o.n}³ grid) takes "
                f"{duration_text(t)} the first time, and so small a smoothing finds many thousands of small voids: "
                f"too many to read on a map. Start it anyway?")
        if self.confirm_void is not None:
            return bool(self.confirm_void(text))
        return QMessageBox.question(self, "Find the voids", text) == QMessageBox.Yes

    def _void_sigma_changed(self, *_):
        """Another smoothing: its own catalogue (cached by key, so going back costs nothing)."""
        self._void_picked = None
        self.void_info.setText("")
        self.void_goto.setEnabled(False)
        if self.voids.isChecked() and self.current is not None and not self.voids_applicable(self.current):
            self._voids_request(self.current)
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)

    # ---------------------------------------------------------------- the catalogue
    def _voids_sync(self):
        """The checkbox follows the output: enabled where the outlines apply, else greyed out saying why."""
        why = self.voids_applicable(self.current)
        self._void_refused = ""                     # another output: asked again (_voids_request)
        self.voids.setEnabled(not why)
        tip_auto(self.voids, _explore().VOID_TIP if not why else f"Void outlines: {why}.")
        if not why:
            self._void_sigma_grid()
        if self._void_picked is not None and (why or self._void_picked[0] != self._void_cache_key(self.current)):
            self._void_picked = None                # another output: the inspector's void is not on it
            self.void_info.setText("")
            self.void_goto.setEnabled(False)
        if not why and self.voids.isChecked():
            self._voids_request(self.current)
        self._void_note()

    def _voids_toggled(self, on: bool):
        self._void_refused = ""
        if on:
            for key in [k for k, v in self._void_cache.items() if "error" in v]:
                del self._void_cache[key]           # ticked again: try again (the grid's panels too)
            if self.current is not None and not self.voids_applicable(self.current):
                self._voids_request(self.current)
        else:
            self._void_queue.clear()
            self._void_picked = None
            self.void_info.setText("")
            self.void_goto.setEnabled(False)
        self._voids_redraw()
        if self._grid_shown:
            self.show_grid(True)                    # the panels' outlines follow the tick

    def _voids_request(self, o, queued: bool = False):
        """Find the voids of this output in a helper process (the pipeline's catalogue: cached, else a search),
        unless they are known or already being found. 'queued' (the figure grid's panels): wait for the search
        running instead of stopping it."""
        key = self._void_cache_key(o)
        if key in self._void_cache or (self._void_proc is not None and self._void_key == key):
            return
        if self._void_proc is not None:
            if queued:
                if all(self._void_cache_key(q) != key for q in self._void_queue):
                    self._void_queue.append(o)
                return
            if self._void_out is not None:          # a grid panel's search stopped for the map's: it comes back
                self._void_queue.insert(0, self._void_out)
        if not queued and key[-1] > 0 and not self._void_search_ok(o, key[-1]):
            self._void_refuse(o, key[-1])           # (a grid panel's queue follows the map's smoothing back)
            return
        self._voids_stop()                          # another output's search: this one is wanted now
        n = int(o.n or 0)
        if not self.heavy_ok("Find the voids", f"Finding the voids of {o.title} builds its catalogue when it is "
                             f"not cached yet; that", 80.0 * n ** 3 / 1e9 if n else None):
            self._void_cache[key] = {"error": "not started while a DTFE run is going (tick again to retry)"}
            self.status.emit("void outlines not started: a DTFE run is going")
            self._void_note()
            self._voids_next()
            return
        root = o.dir.parent.parent                  # <data root>/<sim>/snapdir_NNN, or the family folder: both work
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("DTFE_SIM", o.sim)
        env.insert("DTFE_DATA_ROOT", str(root))
        for name in SMOOTH_MPC_ENV_NAMES:           # the smoothing is the --smooth-mpc below, never the shell's
            env.remove(name)
        env.remove("CLICOLOR_FORCE")
        proc.setProcessEnvironment(env)
        proc.finished.connect(lambda code, _status, k=key, pr=proc: self._voids_done(k, pr, code))
        proc.errorOccurred.connect(lambda err, k=key, pr=proc: self._voids_start_failed(k, pr, err))
        self._void_proc, self._void_key, self._void_out = proc, key, o
        method = "ps" if G.estimator_label(o) == "PS-DTFE" else "dtfe"
        args = [str(_explore().VOID_TOOL), "--sim", o.sim, "--snap", str(int(o.snap)), "--estimator", method,
                "--data-root", str(root)]
        if o.prefix != ("ps_output" if method == "ps" else "output"):
            args += ["--prefix", o.prefix]          # the production set: no prefix, so the analysis scripts' cache is shared
        if key[-1] > 0:
            args += ["--smooth-mpc", repr(key[-1])]  # a common physical sigma: its own cache namespace
        cell = float(o.length[0]) / float(n) if n and o.box is not None else 1.0
        cells = key[-1] / cell if key[-1] > 0 else self.void_sigma.production / cell
        self._void_t0, self._void_est = time.time(), search_seconds(n, cells)
        cache = _explore().VOID_CACHE_DIR
        proc.start(rs.PYTHON, args + (["--cache-dir", str(cache)] if cache else []))
        self.status.emit(f"finding the voids of {o.title} ({duration_text(self._void_est)} the first time; a cached "
                         f"catalogue takes a second)")
        self._void_tick.start()
        self._void_note()

    def _voids_done(self, key, proc, code: int):
        if proc is not self._void_proc:             # stopped: a newer search, or the window closing
            proc.deleteLater()
            return
        self._void_proc = self._void_key = self._void_out = None
        self._void_tick.stop()
        out = bytes(proc.readAllStandardOutput()).decode(errors="replace").strip().splitlines()
        err = bytes(proc.readAllStandardError()).decode(errors="replace").strip().splitlines()
        proc.deleteLater()
        try:
            if code != 0 or not out:
                raise ValueError(err[-1] if err else f"the helper exited with {code}")
            res = json.loads(out[-1])
            res["_centres"] = np.asarray(res["centres_mpc"], dtype=np.float64).reshape(-1, 3)
            res["_r_v"] = np.array([np.nan if v is None else v for v in res["r_v_mpc"]], dtype=np.float64)
            res["_shown"] = np.asarray(res["distinct"], dtype=bool)
            res["_r_eff"] = np.array([np.nan if v is None else v for v in res["r_eff_mpc"]], dtype=np.float64)
            n = len(res["_centres"])                # the catalogue's own fitted ellipsoids: semi-axes (NaN where
            res["_semi"] = G._rows_array(res.get("semi_axes_mpc"), n, 3)        # none), directions (rows)
            res["_dirs"] = np.array([np.eye(3) if v is None else v for v in res.get("ellipsoid_axes") or [None] * n],
                                    dtype=np.float64).reshape(n, 3, 3)
            drawn = G.drawn_ellipsoids(res)         # the ellipsoids drawn: at the mean density (the one place)
            res["_r_ell"] = drawn["r"]
            self._void_cache[key] = res
            shown = int(res["_shown"].sum())
            self.status.emit(f"voids: {shown} distinct, well-resolved voids at σ {res['sigma_mpc']:.3g} Mpc")
            if key[-1] == 0:
                self._void_sigma_grid()             # the production value in Mpc, now known exactly
        except (ValueError, KeyError, TypeError) as e:
            self._void_cache[key] = {"error": str(e)}
            self.status.emit(f"no void outlines: {e}")
        self._voids_redraw()
        if self._grid_shown and self.voids.isChecked():
            self.show_grid(True)                    # a panel's catalogue arrived: its outlines
        self._voids_next()

    def _voids_next(self):
        """The next catalogue the figure grid waits for (one helper at a time)."""
        while self._void_proc is None and self._void_queue and self.voids.isChecked() and not self._shut:
            o = self._void_queue.pop(0)
            if self._void_cache_key(o) not in self._void_cache:
                self._voids_request(o)

    def _voids_start_failed(self, key, proc, err):
        """The helper never started (no 'finished' follows): said, and the next tick tries again."""
        if err != QProcess.FailedToStart or proc is not self._void_proc:
            return
        self._void_proc = self._void_key = self._void_out = None
        self._void_tick.stop()
        self._void_cache[key] = {"error": f"could not start {rs.PYTHON}: {proc.errorString()}"}
        proc.deleteLater()
        self.status.emit(f"no void outlines: could not start {rs.PYTHON}")
        self._voids_redraw()
        self._voids_next()

    def stop_void_search(self):
        """Stop a catalogue search and forget the grid's queue: for the window when a heavy job starts (a
        catalogue build costs about 80 B per cell); the next tick or output change asks again. The map's own search
        stopped is said in the line under the row (the tick stays on with nothing drawn)."""
        o = self.current
        mine = self._void_proc is not None and o is not None and self._void_key == self._void_cache_key(o)
        self._void_queue.clear()
        self._voids_stop()
        if mine:
            self._void_refused = "stopped when a job started: tick Voids again to find them"
        self._void_note()

    def _voids_stop(self):
        proc, self._void_proc, self._void_key, self._void_out = self._void_proc, None, None, None
        self._void_tick.stop()
        if proc is not None:
            proc.kill()
            proc.waitForFinished(2000)

    # ---------------------------------------------------------------- the line under the Voids row
    def _void_note_set(self, text: str):
        self.void_note.setText(text)
        self._out_form_map.setRowVisible(self.void_note, bool(text))

    def _void_note(self):
        """The grey line under the Voids row, while the tick is on: a search running (how long it should take, how
        long so far), why there are no voids (an error, a refusal), or which field the voids belong to while the map shows
        another (smoothed in 3D at the catalogue's smoothing, the density); nothing otherwise."""
        o = self.current
        if not self.voids.isChecked() or o is None or self.voids_applicable(o):
            self._void_note_set("")
            return
        if self._void_refused:                      # a long search declined: said until the next one
            self._void_note_set(self._void_refused)
            return
        key = self._void_cache_key(o)
        if self._void_proc is not None and self._void_key == key:
            sigma = "the production smoothing" if key[-1] <= 0 else f"σ {key[-1]:.3g} Mpc"
            so_far = time.time() - (self._void_t0 or time.time())
            est = duration_text(self._void_est or 0.0)
            self._void_note_set(f"finding the voids at {sigma}: {est} the first time, {so_far:.0f} s so far (a cached "
                                f"catalogue takes a second)")
            return
        res = self._void_cache.get(key)
        if res is None:
            self._void_note_set("")
            return
        if "error" in res:
            self._void_note_set(f"no voids: {res['error']}")
            return
        self._void_note_set(self._void_field_text(res))

    def _void_field_text(self, res: dict) -> str:
        """'found in the density smoothed in 3D at 5.91 Mpc (10 cells): tick 3D and set Smoothing to 10 to see that
        field' while the map shows anything else (the steps that apply: show the density, tick 3D, set Smoothing; none
        where the 3D tick is greyed out on the density, its tooltip saying why); '' while it shows that field; 'no
        distinct voids at this smoothing' when there is none (the tick on, nothing drawn: said whatever the map)."""
        if not res["_shown"].any():
            return "no distinct voids at this smoothing"
        o = self.current
        cell = float(o.length[0]) / float(o.n)
        sigma = float(res.get("sigma_mpc") or 0.0)
        cells = sigma / cell
        req = self._req or {}
        density = req.get("field") == "density"
        three_d = self.smooth3d.isChecked() and self.smooth3d.isEnabled()
        near = abs(self.smooth.value() - cells) <= 0.05 + 1e-9
        same = density and three_d and near and not req.get("pts") and req.get("compare") is None \
            and not self.is_zoomed()
        if same:
            return ""
        smooth = f"{cells:.1f}".rstrip("0").rstrip(".")
        text = f"found in the density smoothed in 3D at {sigma:.3g} Mpc ({smooth} cells)"
        if cells > self.smooth.maximum() or (density and not self.smooth3d.isEnabled()):
            return text
        steps = ([] if density else ["show the density"]) + ([] if three_d else ["tick 3D"]) \
            + ([] if near else [f"set Smoothing to {smooth}"])
        if not steps:
            return text
        said = steps[0] if len(steps) == 1 else ", ".join(steps[:-1]) + " and " + steps[-1]
        return f"{text}: {said} to see that field"

    # ---------------------------------------------------------------- drawing
    def _slice_position(self) -> float | None:
        """Where the shown plane lies along its normal (Mpc, or cells without a box)."""
        req = self._req
        if req is None:
            return None
        if req.get("pts") and self._pts_side is not None:
            return float(self._pts_side.get("center", 0.0))
        o, axis = req["output"], req["axis"]
        return float(o.lo[axis] + (req["index"] + 0.5) * o.length[axis] / o.n)

    def _void_result(self):
        """The catalogue of the output on the map (the void_outlines.py result), or None."""
        o = self.current
        if not self.voids.isChecked() or self._grid_shown or o is None or self.voids_applicable(o):
            return None
        res = self._void_cache.get(self._void_cache_key(o))
        if not res or "error" in res or self._req is None or self._req.get("output") is not o:
            return None
        return res

    def _void_set(self, res: dict, rows=None) -> dict:
        """The voids to draw of a catalogue (grids.void_overlay's dict): the rows in the Sizes range (or 'rows'),
        their centres, the shape chosen with its drawn ellipsoids (and how far each reaches along an axis: its marks), R_v,
        the dashed outline's radii when ticked (R_eff beside the spheres, R_v beside the ellipsoids)."""
        rows = self.void_rows(res) if rows is None else np.asarray(rows, dtype=np.int64)
        shape = self.void_shape()
        d = G.drawn_ellipsoids(res)
        M = d["M"][rows]
        other = None
        if self.void_reff.isChecked():
            other = res["_r_v"][rows] if shape == "ellipsoid" else res["_r_eff"][rows]
        return {"rows": rows, "centres": res["_centres"][rows], "box": float(res["box_mpc"]), "shape": shape,
                "r_v": res["_r_v"][rows], "M": M, "reach": np.sqrt(np.einsum("kij,kij->ki", M, M)),   # sqrt(Q_aa)
                "other": other, "cell": float(res.get("cell_mpc") or res["box_mpc"] / res["n_grid"])}

    def _void_overlay(self) -> dict | None:
        """The void outlines over what the map shows now (grids.void_overlay: circles, dashed, ellipses, marks in
        image pixels), or None: off, not applicable, or not found yet."""
        res = self._void_result()
        g = self._geometry()
        if res is None or g is None:
            return None
        axis, _index, rest, lo, hi, n0, n1 = g
        v = self._void_set(res)
        o = self.current
        half = 0.5 * float(o.length[axis]) / float(o.n)
        return G.void_overlay(v, axis, self._slice_position(), rest, lo, hi, n0, n1, half)

    def _void_circles(self, radius: str = "_r_v"):
        """The map's void outlines now (None: off, not applicable, or not found yet): the solid ones of the shape
        chosen (circles of R_v, or the ellipsoids' sections, turned) -- radius '_r_eff' the dashed ones instead."""
        ov = self._void_overlay()
        if ov is None:
            return None
        return ov["dashed"] if radius == "_r_eff" else ov["circles"] + ov["ellipses"]

    def _set_void_canvas(self, ov: dict | None):
        """The map's outlines from a void_overlay dict (None: none)."""
        c = self.view.canvas
        ov = ov or {}
        c.circles = (ov["circles"] + ov["ellipses"]) if ov else None    # the outlines of the shape chosen
        c.dashed = ov.get("dashed") if ov else None
        c.marks = ov.get("marks") or None
        c.mark_clips = None

    def _voids_redraw(self):
        if self._3d_shown:
            self._3d_voids()                        # the shapes of the 3D view (explore_3d)
        self._void_count_note()
        self._void_note()
        c = self.view.canvas
        if c._img is None or self._grid_shown:
            return
        self._set_void_canvas(self._void_overlay())
        c._redraw()

    def _void_count_note(self):
        """Beside Sizes: how many of the output's distinct voids the range keeps (by the drawn shape's radius), how
        many of them have no ellipsoid at the mean density (drawn as their centre alone), and how many minima the
        catalogue holds in all when that is more (the ones outside its distinct sample are not drawn)."""
        o = self.current
        res = None if o is None or not self.voids.isChecked() or self.voids_applicable(o) else \
            self._void_cache.get(self._void_cache_key(o))
        if not res or "error" in res:
            self.void_count.setText("")
            return
        radius = self._void_radius_key()
        rows = self.void_rows(res, radius)
        n, kept = int(np.count_nonzero(res["_shown"])), len(rows)
        text = (f"all {n} voids" if kept == n else f"{kept} of {n} voids") \
            + (" (by R_ell)" if radius == "_r_ell" else "")
        if radius == "_r_ell":
            alone = int(np.count_nonzero(~G.drawn_ellipsoids(res)["ok"][np.nonzero(res["_shown"])[0]]))
            if alone:
                text += f"; {alone} without an ellipsoid (centre only)"
        total = len(res["_shown"])
        if total > n:
            text += f" · {total} minima"
        self.void_count.setText(text)

    def voids_shown(self) -> list[int]:
        """The catalogue rows drawn on the map now (outlines and centre marks), in row order: the CSV's voids."""
        c = self.view.canvas
        rows = {int(o[-1]) for items in (c.circles, c.dashed) for o in items or ()}
        rows |= {int(m[2]) for m in c.marks or ()}
        return sorted(rows)

    def _void_pick(self, r: int, c: int):
        """The catalogue row under image pixel (r, c), or None: the smallest outline holding it (circles, ellipses,
        the dashed ones), else a centre mark next to it."""
        cv = self.view.canvas
        img = cv._img
        reach = 0.012 * max(img.width(), img.height()) if img is not None else 3.0
        k = G.outline_pick(c + 0.5, r + 0.5, cv.circles, (), cv.marks, max(3.0, reach))
        k = k if k is not None else G.void_at(cv.dashed, c + 0.5, r + 0.5)
        # the hit test first: the result's key stats the density file, not wanted on every mouse move
        return k if k is not None and self._void_result() is not None else None

    def _void_goto(self):
        """The slice through the centre of the void last clicked (along the map's own axis)."""
        res, picked, o = self._void_result(), self._void_picked, self.current
        if res is None or picked is None or picked[0] != self._void_cache_key(o) or self.is_zoomed():
            return
        axis = self.axis.currentIndex()
        x = float(res["_centres"][picked[1]][axis])
        i = int(np.floor((x - float(o.lo[axis])) / (float(o.length[axis]) / o.n))) % int(o.n)
        self.index.setValue(i)

    def _panel_voids(self, o) -> dict | None:
        """A grid panel's voids: its output's own catalogue (_void_set: the rows in the Sizes range, the shape chosen)
        when 'voids' is ticked and it applies; a catalogue not found yet is asked for (queued, one at a time; the grid
        is recomposed when it arrives). None otherwise."""
        if not self.voids.isChecked() or self.voids_applicable(o):
            return None
        res = self._void_cache.get(self._void_cache_key(o))
        if res is None:
            self._voids_request(o, queued=True)
            return None
        if "error" in res:
            return None
        return self._void_set(res)

    def _void_csv_columns(self, res: dict) -> tuple[dict, str]:
        """The saved CSV's columns of the drawn ellipsoid (grids.drawn_ellipsoids: a, b, c and R_ell, per catalogue
        row) and the '#' line's words for them."""
        d = G.drawn_ellipsoids(res)
        cols = {f"ell_{s}_mpc": d["semi"][:, j] for j, s in enumerate("abc")}
        cols["r_ell_mpc"] = d["r"]
        return cols, ("ell_a/b/c = the semi-axes of the ellipsoid drawn, where the quadratic fit around the minimum "
                      "returns to the mean density (sqrt(2 |delta| / lambda)), r_ell = (a b c)^(1/3) of it")
