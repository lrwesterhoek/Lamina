"""MainWindow's opt-in Advanced settings (AdvancedMixin): the View menu's switch 'Advanced Settings', off by default
and remembered under one settings key (ADVANCED_SETTING), and while it is on a folded group 'Advanced' on each run
tab (Grids, Custom snapshot, Pipeline) after the tab's own groups. Its rows are the settings the code otherwise
chooses (runspec.ADVANCED_KEYS: controls the two quality modes removed), each on Automatic first with a grey note
of what Automatic gives the job now; a tab whose job has an override says so on a grey line under its 'Runs on' line.

Off is the two-mode launcher: the groups, their rows and the grey lines are built when the switch goes on and
deleted when it goes off, which puts the three tabs' jobs back to Automatic (a queued job keeps what it was queued
with). A job loaded with overrides while it is off turns it on. Every rule -- which rows a job shows, what a row
offers, what Automatic gives, which override is passed -- is runspec's (advanced_rows, advanced_choices,
automatic_text, advanced_refused); this module only shows them. The names app.py defines are read through _app,
imported last (see app_tabs.py)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget

import runspec as rs
from app_tabs import RUN_TABS
from app_widgets import AutoRow, Collapsible, FormNote, _grow_fields, tip

ADVANCED_SETTING = "advanced_settings"     # the launcher's settings key of the switch (json true / false)
ADVANCED_TITLE = "Advanced"
ADVANCED_INTRO = ("For tests and special cases. Automatic lets the code choose, which is right for almost every "
                  "run.")
# each row's tooltip (at most two lines) and What's This, from the texts of the controls before the two modes and the
# programs' own help; no measured number
ADVANCED_TIP = {
    "gpu": ("CPU, GPU, or CPU + GPU (PS-DTFE Exact: a split run's partitions shared between the two).\n"
            "Automatic: the device the Quality and the Machine use pick for this job.",
            "Where the grid deposit (PS-DTFE) or the grid interpolation (standard DTFE) runs. GPU: every partition "
            "on the GPU, in every Machine use. CPU + GPU: a PS-DTFE Exact run split into partitions deposits some of "
            "them on the CPU cores and the others on the GPU at the same time, at the program's own share for this "
            "machine, in every Machine use (Automatic does so under Performance only); a small or unsplit run, or "
            "a Machine use that leaves too few cores beside the GPU, deposits on the GPU alone, and the program's "
            "log says which. The GPU is a choice only when the "
            f"program this job runs is a GPU build ({rs.GPU_BUILD_HINT}); CPU + GPU only for PS-DTFE in single "
            "precision; the 2D programs run on the CPU."),
    "nsub": ("Fast only: the sampled deposit's points per cell axis (nSub³ points per cell).\n"
             "More points: a smoother cell average, more time. Automatic: one. Exact uses no samples.",
             "Automatic is the Fast quality's own count, one point per cell axis: quick, with noisy per-cell values "
             "(smooth the grids before use; void catalogues at the usual smoothing come out nearly the same). Exact "
             "is the reference."),
    "face_gradient": ("How the mean velocity's derivatives (divergence, shear, vorticity) are computed:\n"
                      "exact face integrals, or the stream average of each stream's own gradient.",
                      "The exact face integrals apply with Exact, in 3D and without the halo release; elsewhere the "
                      "job keeps the stream average. Automatic takes the face integrals wherever they apply."),
    "partition": ("The partitions per axis (p³ partitions, p² in 2D), each triangulated on its own.\n"
                  "Automatic: the program's auto-tuner plans them for the memory budget.", ""),
    "max_concurrent": ("How many triangulations are built at once (a cap on the peak memory).\n"
                       "Automatic: the program's auto-tuner plans it for the memory budget.", ""),
    "mem_budget_gb": ("The memory the auto-tuner plans the partitions and concurrency for (DTFE_MEM_BUDGET_GB).\n"
                      "Automatic: what is free when the run starts.",
                      "Set it lower to leave room for other work, higher when you know the memory will be free (it "
                      "may swap)."),
    "parallel_triangulation": ("On: the parallel insert of the particles into the triangulation, at any size.\n"
                               "Automatic: the program's own rule, the parallel insert for large tessellations only.",
                               "Two identical runs with the parallel insert differ at float rounding (the stream "
                               "counts do not). It needs a build with TBB."),
    "ordered_merge": ("How the partitions' grids are added into the result. Automatic: in index order, so two\n"
                      "identical runs give bit-identical grids; Completion order: as each partition finishes.",
                      "Completion order keeps no staged sub-grids; with more than one concurrent triangulation two "
                      "identical runs then differ in the last bits."),
    "precision": ("Which pair of programs runs: single precision (the default), or the double-precision pair\n"
                  "(float64 from the input read onward, about twice the memory). Setup builds the pair.",
                  "The double-precision pair keeps every position, density and field in float64 from the input read "
                  "onward and writes float64 grids."),
}


class AdvancedMixin:
    # ---------------------------------------------------------------- the switch
    def _build_advanced_action(self, menu):
        """View > Advanced Settings: checkable, off by default (self.advanced_on, read in _init_state)."""
        action = QAction("Advanced Settings", self, checkable=True)
        action.setMenuRole(QAction.MenuRole.NoRole)         # (macOS would move a 'Settings' item into the app menu)
        action.setToolTip("An Advanced group on the Grids, Custom snapshot and Pipeline tabs: settings the code "
                          "otherwise chooses")
        action.setChecked(self.advanced_on)
        action.toggled.connect(self._advanced_toggled)
        menu.addAction(action)
        self.advanced_action = action

    def _advanced_toggled(self, on: bool):
        """The user's switch. On: the groups, open. Off: the groups go, and the three tabs' jobs go back to Automatic
        (the status line says how many settings did; nothing when there were none)."""
        if on == self.advanced_on:
            return
        self._read_widgets()            # the rows' values into the jobs first: what goes back to Automatic
        cleared = 0
        if not on:
            for tab in RUN_TABS:
                spec = self._tab_spec(tab)
                cleared += len(rs.job_advanced(spec))
                spec.advanced = {}
        self._set_advanced(on, open_now=True)
        self.refresh()
        self._save()
        if cleared:
            self.status_message(f"Advanced settings off: {cleared} setting{'s' if cleared != 1 else ''} back to "
                                "Automatic")

    def _set_advanced(self, on: bool, open_now: bool = False):
        """The switch's state, the menu item and the groups, without reading, refreshing or saving the jobs (a load
        calls this midway); the settings key is written only by a window that remembers. open_now: the switch has
        just been turned on (by the user or a load), so the groups open; a window that starts with it on shows each
        group as it was left (open the first time)."""
        self.advanced_on = on
        if self.remember:
            self.settings.setValue(ADVANCED_SETTING, "true" if on else "false")
        action = getattr(self, "advanced_action", None)
        if action is not None and action.isChecked() != on:
            action.blockSignals(True)
            action.setChecked(on)
            action.blockSignals(False)
        if on and not self.adv_boxes:
            for tab in RUN_TABS:
                self._build_advanced_tab(tab)
        elif not on:
            for tab in list(self.adv_boxes):
                self._drop_advanced_tab(tab)
        if on and open_now:
            for g in self.adv_boxes.values():
                g.set_open(True)

    def _load_advanced(self):
        """The end of _load_into_widgets: the jobs' overrides into the rows. A job loaded with overrides while the
        switch is off (a remembered job, a run's settings, a job given to the window) turns it on, and the status line
        says so; self._advanced_loaded is then how many overrides that was (0: no switch)."""
        self._advanced_loaded = 0
        if not self.advanced_on:
            n = sum(len(rs.job_advanced(self._tab_spec(tab))) for tab in RUN_TABS)
            if not n:
                return
            self._advanced_loaded = n
            self._set_advanced(True, open_now=True)
            self.status_message(f"Advanced settings on: the loaded job has {rs.advanced_count_text(n)}")
        elif not self.adv_boxes:
            self._set_advanced(True)            # a window that starts with the switch on (remembered)
        for tab in self.adv_boxes:
            self._sync_advanced_tab(tab)

    def _read_advanced(self):
        """The end of _read_widgets: each run tab's rows into its job's overrides (Automatic: no key). Without the
        groups (the switch off) the jobs keep theirs: a load that brings overrides has not built the rows yet."""
        for tab, rows in self.adv_rows.items():
            self._tab_spec(tab).advanced = {k: v for k, row in rows.items() for v in [row.value()] if v is not None}

    # ---------------------------------------------------------------- the groups
    def _tab_page(self, tab: str) -> QWidget:
        """A run tab's form (the widget inside its scroll area)."""
        index = {"grids": _app.GRIDS, "custom": _app.OWN, "pipeline": _app.PIPELINE}[tab]
        return self.tabs.widget(index).widget()

    def _build_advanced_tab(self, tab: str):
        """The tab's Advanced group, after its own groups (before the column's closing stretch): the grey first line,
        then a row per override key of the tab's job class (runspec.advanced_keys), shown where the job takes it
        (_sync_advanced_tab); and the grey 'Advanced: ...' line under the tab's 'Runs on' line. Whether the group is
        open is kept as the other folded groups' is, under its own key (not the '<tab>.advanced' of the launcher
        before the two modes)."""
        spec = self._tab_spec(tab)
        g = Collapsible(ADVANCED_TITLE, f"{tab}.advanced_settings", self._section_settings(),
                        lambda t=tab: self._advanced_summary(t), start_open=True)
        g.form.addRow(FormNote(ADVANCED_INTRO))
        rows = {}
        for key in rs.advanced_keys(type(spec)):
            row = AutoRow(rs.ADVANCED_RANGE.get(key))
            if not row.is_number:
                row.set_choices(rs.advanced_choices(spec, key))
            tip(row, *ADVANCED_TIP[key])
            row.changed.connect(self._changed)
            g.form.addRow(rs.ADVANCED_ROW[key], row)
            tip(g.form.labelForField(row), ADVANCED_TIP[key][0])
            rows[key] = row
        wide = max(row.control.sizeHint().width() for row in rows.values())
        for row in rows.values():
            row.control.setMinimumWidth(wide)       # one column of controls: their grey notes line up
        col = self._tab_page(tab).layout()
        last = col.count() - 1
        col.insertWidget(last if col.itemAt(last).spacerItem() is not None else col.count(), g)
        runs_on = self._shared(tab, "runs_on")
        form = runs_on.parentWidget().layout()
        line = FormNote()
        line.setToolTip("Settings of the Advanced group that differ from Automatic (the group lists them all)")
        form.insertRow(form.getWidgetPosition(runs_on)[0] + 1, "", line)
        form.setRowVisible(line, False)
        _grow_fields(g)
        g.update_summary()
        self.adv_boxes[tab], self.adv_rows[tab], self.adv_lines[tab] = g, rows, line
        self._advanced_focus(g)

    def _drop_advanced_tab(self, tab: str):
        """The switch off: the tab's group and grey line go (the form is the two-mode one again)."""
        g, line = self.adv_boxes.pop(tab), self.adv_lines.pop(tab)
        self.adv_rows.pop(tab, None)
        line.parentWidget().layout().removeRow(line)        # (deletes the line and its empty label)
        self._tab_page(tab).layout().removeWidget(g)
        g.setParent(None)
        g.deleteLater()

    def _advanced_focus(self, g: QWidget):
        """Tab moves from the tab's last field into the group: built last, its fields would otherwise come after the
        whole window's."""
        page, chain, w = g.parentWidget(), [], self.nextInFocusChain()
        while w is not None and w is not self and len(chain) < 20000:
            chain.append(w)
            w = w.nextInFocusChain()
        tabbing = [x for x in chain if x.focusPolicy().value & Qt.FocusPolicy.TabFocus.value]
        before = [x for x in tabbing if page.isAncestorOf(x) and not g.isAncestorOf(x)]
        mine = [x for x in tabbing if g.isAncestorOf(x)]
        if before and mine:
            for a, b in zip([before[-1]] + mine, mine):
                QWidget.setTabOrder(a, b)

    def _advanced_summary(self, tab: str) -> str:
        """The folded group's one line: the job's overrides ('Device: CPU; Partitions: 4'), else 'all Automatic'."""
        return rs.advanced_text(self._tab_spec(tab)) or "all Automatic"

    def _sync_advanced(self):
        """refresh(): every built group follows its job."""
        for tab in self.adv_boxes:
            self._sync_advanced_tab(tab)

    def _sync_advanced_tab(self, tab: str):
        """A tab's rows after a change of its job: shown where the job takes the setting (runspec.advanced_rows) or
        while it holds a value (so it can be set back); each row's choices and value; its grey note -- on Automatic
        what Automatic gives now, else why the value is not passed, if it is not. The samples row is greyed under
        Exact (it uses none) unless it holds a value. Then the grey 'Advanced: ...' line and the summary."""
        spec = self._tab_spec(tab)
        g, rows, line = self.adv_boxes[tab], self.adv_rows[tab], self.adv_lines[tab]
        adv = rs.job_advanced(spec)
        shown = set(rs.advanced_rows(spec)) | set(adv)
        refused = rs.advanced_refused(spec) if adv else {}
        exact = spec.mode().deposit == "exact"
        for key, row in rows.items():
            g.form.setRowVisible(row, key in shown)
            if key not in shown:
                # a hidden row holds no value (the job holds none for it: 'shown' has every key it holds), else the
                # next read (_read_advanced) would put a cleared or replaced job's old setting back into it
                row.set_value(None)
                continue
            if not row.is_number:
                choices = rs.advanced_choices(spec, key)
                if key in adv and not any(v == adv[key] and type(v) is type(adv[key]) for v, _ in choices):
                    choices.append((adv[key], rs.advanced_value_text(key, adv[key])))     # (not on offer here)
                row.set_choices(choices)
            row.set_value(adv.get(key))
            row.set_note(rs.automatic_text(spec, key) if key not in adv
                         else f"Not passed: {refused[key]}" if key in refused else "")
            row.setEnabled(not (key == "nsub" and exact and key not in adv))
        text = rs.advanced_text(spec)
        line.setText(f"Advanced: {text}" if text else "")
        line.parentWidget().layout().setRowVisible(line, bool(text))
        g.update_summary()


import app as _app  # noqa: E402
