"""MainWindow's job tabs (TabsMixin, moved from app.py): the shared data panel, and the Data, Grids, Custom
snapshot, Pipeline and Plots tabs with their option groups (presets, fields, resources, a figure set's
options) and the reactions that stay inside a tab (the Data tab's snapshot table, the deposit choice that
follows the estimator, a new custom snapshot file). The three run tabs (Grids, Custom snapshot, Pipeline) build
the groups they share from one table, SHARED_WIDGETS, which also keeps each tab's attribute names for them. The
names app.py defines are read through _app at call time; that import comes last because app.py also runs as a
script: importing it again under its own name then needs this module's class to exist already."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIntValidator
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QAbstractItemView, QHeaderView,
                               QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget)

import runspec as rs
import presets as P
import grids as G
from app_widgets import (_mono, _note, _checkable, _dir_row, _exclusive, _state_text, Collapsible, GROUP_MARGIN,
                         INDENT, advanced_group, button_row, field_grid, flush, grid_row, group, labelled_row,
                         option_tip, tip, tip_auto)

L = P.LABELS
RUN_TABS = ("grids", "custom", "pipeline")
# the widgets the three run tabs share, built by _estimator_group, _ps_options_group, _fields_group,
# _resources_group and _advanced_group, under each tab's own attribute names (the other mixins, app_specs.py and the
# tests reach them by these names); None: the tab has no such widget
SHARED_WIDGETS = {
    #  role                      Grids            Custom snapshot   Pipeline
    "deposit":                 ("deposit_combo", "o_deposit",      "p_deposit"),
    "grid":                    ("grid_combo",    "o_grid",         "p_grid"),
    "grid_cells":              ("grid_cells",    "o_grid_cells",   "p_grid_cells"),
    "nsub":                    ("nsub_spin",     "o_nsub",         "p_nsub"),
    "gpu":                     ("gpu_check",     "o_gpu",          "p_gpu"),
    "run_mode":                ("run_mode_combo", "o_run_mode",    "p_run_mode"),
    "vertex_mass":             ("vm_check",      "o_vm",           "p_vm"),
    "volume_weighted":         ("vw_check",      "o_vw",           "p_vw"),
    "caustics":                ("ca_check",      "o_ca",           "p_ca"),
    "caustic_cusps":           ("cu_check",      "o_cu",           "p_cu"),
    "linear_deposit":          ("ld_check",      "o_ld",           "p_ld"),
    "face_gradient":           ("fg_check",      "o_fg",           "p_fg"),
    "ps_group":                ("ps_group",      "o_ps_group",     "p_ps_group"),
    "fields":                  ("field_checks",  "o_fields",       "p_fields"),
    "fields_box":              ("fields_box",    "o_fields_box",   "p_fields_box"),
    "partition":               ("part_spin",     "o_part",         "p_part"),
    "concurrent":              ("conc_spin",     "o_conc",         "p_conc"),
    "scratch":                 ("scratch_edit",  "o_scratch",      "p_scratch"),
    "parallel_triangulation":  ("pt_check",      "o_pt",           "p_pt"),
    "ordered_merge":           ("om_check",      "o_om",           "p_om"),
    "start_over":              ("so_check",      None,             "p_so"),
    "precision":               ("prec_combo",    "o_prec",         "p_prec"),
    "prefix":                  ("prefix_edit",   None,             "p_prefix"),
    "budget":                  ("budget_spin",   "o_budget",       "p_budget"),
    "tess":                    ("tess_edit",     "o_tess",         "p_tess"),
    "resources":               ("res_group",     "o_res_group",    "p_res_group"),
    "advanced":                ("adv_group",     "o_adv_group",    "p_adv_group"),
}
# the phase-space options of the 'Phase-space options' group, then the expert ones of the Advanced group (P.label keys).
# The face gradient is in sight, the group's last row (since 2026-11-11, before in the folded Advanced group: a tick set
# by hand, for the mean flow's derivatives and the mass flux, not only through the drivers)
PS_OPTIONS = ("vertex_mass", "volume_weighted", "caustics", "face_gradient")
EXPERT_OPTIONS = ("caustic_cusps", "linear_deposit")


def field_label(name: str, label: str) -> str:
    """A field's name in the run tabs: the Explore tab's (grids.FIELD_LABELS: 'density ρ/ρ̄', 'T-web class'),
    else runspec's."""
    return G.FIELD_LABELS.get(name[:-2] if name.endswith("_a") else name, label)


class TabsMixin:
    # ================================================================ shared data panel
    def _build_data(self) -> QWidget:
        """The data panel every tab shares: the data root (with Setup…), and for the Grids and Plots tabs the
        simulation and its snapshots. No title: the tab bar above it says what it serves."""
        g = QGroupBox()
        f = QFormLayout(g)
        self._data_form = f
        self.root_edit = QLineEdit()
        self.root_edit.editingFinished.connect(self._root_changed)
        root = _dir_row(self.root_edit, self._browse_root)
        self.setup_btn = QPushButton(L["setup"])
        self.setup_btn.setToolTip("Data folder, building the programs, the TNG API key, and the demo")
        self.setup_btn.clicked.connect(lambda: self.run_setup())
        root.addWidget(self.setup_btn)
        f.addRow("Data root", root)
        self.sim_combo = QComboBox()
        self.sim_combo.currentTextChanged.connect(self._sim_changed)
        f.addRow("Simulation", self.sim_combo)
        self.snap_list = QListWidget()
        self.snap_list.setFont(_mono())
        self.snap_list.setMinimumHeight(150)
        self.snap_list.setMaximumHeight(190)
        self.snap_list.itemChanged.connect(self._changed)
        f.addRow("Snapshots", self.snap_list)
        self.all_btn, self.none_btn, self.mark_btn = QPushButton(L["select_all"]), QPushButton(L["select_none"]), \
            QPushButton()
        self.all_btn.clicked.connect(lambda: self._select_snaps("all"))
        self.none_btn.clicked.connect(lambda: self._select_snaps("none"))
        self.mark_btn.clicked.connect(lambda: self._select_snaps("mark"))
        self._snap_buttons = button_row(self.all_btn, self.none_btn, self.mark_btn)
        f.addRow("", self._snap_buttons)
        return g

    # ================================================================ Data tab
    def _build_data_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Simulation")
        f = QFormLayout(g)
        self.d_sim = QComboBox()
        self.d_sim.setEditable(True)
        self.d_sim.setInsertPolicy(QComboBox.NoInsert)
        self.d_sim.activated.connect(self._data_sim_changed)
        self.d_sim.lineEdit().editingFinished.connect(self._data_sim_changed)
        f.addRow("Name", self.d_sim)
        self.d_target = _note()
        f.addRow("", self.d_target)
        col.addWidget(g)

        g = QGroupBox("Snapshots")
        v = QVBoxLayout(g)
        self.d_table = QTableWidget(0, 4)
        self.d_table.setHorizontalHeaderLabels(["Snapshot", "z", "Particles", "Group catalogue"])
        self.d_table.verticalHeader().setVisible(False)
        self.d_table.verticalHeader().setDefaultSectionSize(22)
        self.d_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.d_table.setSelectionMode(QAbstractItemView.NoSelection)
        head = self.d_table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.Stretch)
        for c in (0, 1):                            # snapshot number and z: as wide as they need
            head.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.d_table.setMinimumHeight(230)
        self.d_table.itemChanged.connect(self._changed)
        v.addWidget(self.d_table, 1)        # grows with the window (else ten rows above empty space)
        buttons = []
        for key, mode in (("select_all", "all"), ("select_none", "none"), ("select_not_merged", "missing")):
            b = QPushButton(L[key])
            b.clicked.connect(lambda _=False, m=mode: self._select_data(m))
            buttons.append(b)
        self.d_other = QLineEdit(placeholderText="add snapshots, e.g. 59 72")
        self.d_other.editingFinished.connect(self._add_data_snapshots)
        v.addLayout(button_row(*buttons, field=self.d_other))
        self.d_whole = _note()
        v.addWidget(self.d_whole)
        col.addWidget(g, 1)

        g = QGroupBox("Download from the TNG API")
        v = QVBoxLayout(g)
        self.d_dl_snap = QCheckBox("Particle snapshots")
        self.d_dl_gc = QCheckBox("Group catalogues")
        self.d_dl_tree = QCheckBox("Merger trees (whole simulation)")
        self.d_dl_ics = QCheckBox("Initial conditions (whole simulation)")
        for cb, text in ((self.d_dl_snap, "For the ticked snapshots"),
                        (self.d_dl_gc, "For the ticked snapshots"),
                        (self.d_dl_tree, "One set for the whole simulation"),
                        (self.d_dl_ics, "One set for the whole simulation")):
            cb.setToolTip(text)
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(g)

        g = QGroupBox("Merge into combined files")
        v = QVBoxLayout(g)
        self.d_mg_snap = QCheckBox("Snapshots")
        self.d_mg_gc = QCheckBox("Group catalogues")
        self.d_mg_tree = QCheckBox("Merger trees")
        self.d_mg_ics = QCheckBox("Initial conditions")
        self.d_delete = QCheckBox("Delete the raw chunks after a verified merge")
        for cb, text in ((self.d_mg_snap, "Writes combined_NNN.hdf5 (h-free units)"),
                        (self.d_mg_gc, "Writes combined_fof_subhalo_tab_NNN.hdf5"),
                        (self.d_mg_tree, "Writes combined_tree_extended.hdf5"),
                        (self.d_mg_ics, "Writes combined_ics.hdf5"),
                        (self.d_delete, "The chunks are the only copy of the fields the merge drops "
                                        "(potential, Subfind densities)")):
            cb.setToolTip(text)
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        col.addWidget(g)
        # (no trailing stretch: the snapshot table above takes the room a taller window gives)
        w = QWidget()
        w.setLayout(col)
        return w

    def _data_sim_changed(self, *_):
        if self._loading:
            return
        self.data.sim = self.d_sim.currentText().strip()
        self.data.snapshots = []
        self._fill_data_table()
        self._changed()

    def _fill_data_table(self):
        was = self._loading
        self._loading = True
        sim, root = self.d_sim.currentText().strip(), self.root_edit.text().strip()
        wanted = set(self.data.snapshots)
        self.d_table.setRowCount(0)
        self.d_target.setText("")
        self.d_whole.setText("")
        if sim and rs.root_ok(root):
            sp = rs.sim_dir(sim, root)
            n_part = rs.dm_particles(sim)
            size = (f"  ·  {round(n_part ** (1 / 3))}³ particles, {n_part * rs.MERGED_BYTES_PER_PARTICLE / 1e9:.1f} GB "
                    f"per snapshot") if n_part else ""
            try:
                where = str(sp.relative_to(Path(root)))
            except ValueError:
                where = str(sp)
            self.d_target.setText(where + ("  (new folder)" if not sp.is_dir() else "") + size)
            self.d_target.setToolTip(f"{sp}\n~{n_part * rs.RAW_BYTES_PER_PARTICLE / 1e9:.1f} GB to download, "
                                     f"{n_part * rs.MERGED_BYTES_PER_PARTICLE / 1e9:.1f} GB merged, per snapshot"
                                     if n_part else str(sp))
            for n in sorted(set(rs.data_snapshots(sp)) | wanted):
                st = rs.snapshot_state(sp, n)
                z = rs.SNAPSHOT_LADDER.get(n)
                if z is None and st["merged"]:
                    z = rs._redshift(sp / f"snapdir_{n:03d}" / f"combined_{n:03d}.hdf5")
                row = self.d_table.rowCount()
                self.d_table.insertRow(row)
                first = QTableWidgetItem(f"{n:03d}")
                first.setData(Qt.UserRole, n)
                first.setData(Qt.UserRole + 1, st["merged"])
                first.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                first.setCheckState(Qt.Checked if n in wanted else Qt.Unchecked)
                cells = [first, QTableWidgetItem("?" if z is None else f"{z:.2f}"),
                         QTableWidgetItem(_state_text(st["chunks"], st["merged"])),
                         QTableWidgetItem(_state_text(st["gc_chunks"], st["gc_merged"]))]
                for col, (item, (chunks, merged)) in enumerate(zip(
                        cells, [(0, False), (0, False), (st["chunks"], st["merged"]),
                                (st["gc_chunks"], st["gc_merged"])])):
                    if col >= 2:
                        item.setForeground(_app.MERGED_COLOR if merged else _app.CHUNKS_COLOR if chunks else QColor("gray"))
                    if col:
                        item.setFlags(Qt.ItemIsEnabled)
                    self.d_table.setItem(row, col, item)
            w = rs.sim_state(sp)
            self.d_whole.setText(
                f"Merger trees: {_state_text(w['tree_chunks'], w['tree_merged'])}   ·   Initial conditions: "
                + ("converted" if w["ics_converted"] else "downloaded, not converted" if w["ics_raw"] else "—"))
        self._loading = was

    def _select_data(self, mode: str):
        self._loading = True
        for row in range(self.d_table.rowCount()):
            it = self.d_table.item(row, 0)
            on = mode == "all" or (mode == "missing" and not it.data(Qt.UserRole + 1))
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._loading = False
        self._changed()

    def _add_data_snapshots(self):
        extra = {int(t) for t in self.d_other.text().replace(",", " ").split() if t.isdigit() and int(t) <= 99}
        if not extra:
            return
        self.d_other.clear()
        self._read_widgets()
        self.data.snapshots = sorted(set(self.data.snapshots) | extra)
        self._fill_data_table()
        self._changed()

    # ================================================================ presets
    def _preset_group(self, target: str) -> QGroupBox:
        """'Start from': the built-in and saved presets for the Grids ('grids') or Custom snapshot ('custom') job."""
        g = QGroupBox(L["preset"])
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        combo = QComboBox()
        combo.setToolTip("Sets the grid, fields and options below; everything stays editable")
        save = QPushButton(L["save_preset"])
        delete = QPushButton(L["delete_preset"])
        row.addWidget(combo, 1)
        row.addWidget(save)
        row.addWidget(delete)
        v.addLayout(row)
        about = _note()
        about.hide()
        v.addWidget(about)
        combo.activated.connect(lambda _i: self._apply_preset(target))
        save.clicked.connect(lambda: self._save_preset(target))
        delete.clicked.connect(lambda: self._delete_preset(target))
        setattr(self, f"_preset_{target}", (combo, delete, about))
        self._fill_presets(target)
        return g

    def _fill_presets(self, target: str):
        combo, delete, about = getattr(self, f"_preset_{target}")
        keep = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(L["choose_preset"], "")
        for name in P.preset_names(self.user_presets):
            combo.addItem(name, name)
        combo.setCurrentIndex(max(combo.findData(keep), 0))
        combo.blockSignals(False)
        delete.setEnabled(bool(combo.currentData()) and combo.currentData() not in P.BUILTIN_PRESETS)
        about.setText("")
        about.hide()

    def _apply_preset(self, target: str):
        combo, delete, about = getattr(self, f"_preset_{target}")
        name = combo.currentData()
        delete.setEnabled(bool(name) and name not in P.BUILTIN_PRESETS)
        found = P.preset(name, self.user_presets) if name else None
        if not found:
            about.setText("")
            about.hide()
            return
        text, settings = found
        self._read_widgets()
        spec = self.spec if target == "grids" else self.custom
        planes = rs.planes(self._sim_path()) if (target == "grids" and self._sim_path()) else []
        changed = P.apply_preset(spec, settings, planes)
        if "gpu" not in settings:          # the preset leaves the GPU to the run's size (see recommended_gpu)
            rec = rs.recommended_gpu(spec.deposit, spec.fields, self._tab_particles(target), spec.estimator,
                                     spec.precision, run_mode=spec.run_mode)
            if spec.gpu != rec:
                spec.gpu = rec
                changed.append("gpu")
        if target == "grids" and settings.get("slice_plane") == "largest" and not planes:
            text += "  (no image plane for this simulation yet: make one with the Pipeline tab)"
        about.setText(text)
        about.show()
        self._load_into_widgets()
        self._changed()
        self.status_message(f"preset '{name}': {len(changed)} setting{'s' if len(changed) != 1 else ''} changed")

    def _save_preset(self, target: str):
        name, ok = QInputDialog.getText(self, "Save as preset", "Name for the current settings:")
        name = name.strip()
        if not ok or not name:
            return
        if name in P.BUILTIN_PRESETS:
            QMessageBox.information(self, "Save as preset", f"'{name}' is a built-in preset; choose another name.")
            return
        self._read_widgets()
        self.user_presets[name] = P.capture(self.spec if target == "grids" else self.custom)
        self._save()
        for t in ("grids", "custom"):
            self._fill_presets(t)
        combo = getattr(self, f"_preset_{target}")[0]
        combo.setCurrentIndex(combo.findData(name))
        self.status_message(f"saved preset '{name}'")

    def _delete_preset(self, target: str):
        combo = getattr(self, f"_preset_{target}")[0]
        name = combo.currentData()
        if name and name in self.user_presets:
            del self.user_presets[name]
            self._save()
            for t in ("grids", "custom"):
                self._fill_presets(t)

    # ================================================================ the run tabs' shared groups
    def _shared(self, tab: str, role: str):
        """A run tab's widget of a shared role (SHARED_WIDGETS), by the tab's own attribute name."""
        name = SHARED_WIDGETS[role][RUN_TABS.index(tab)]
        return getattr(self, name) if name else None

    def _set_shared(self, tab: str, role: str, widget):
        name = SHARED_WIDGETS[role][RUN_TABS.index(tab)]
        if name:
            setattr(self, name, widget)

    def _tab_estimator(self, tab: str) -> str:
        """The estimator a run tab's choices apply to now: from its radio, not its spec (an estimator switch changes
        the deposit combo before _read_widgets); the Pipeline runs PS-DTFE."""
        if tab == "grids":
            return "ps" if self.ps_radio.isChecked() else "dtfe"
        if tab == "custom":
            return "ps" if self.o_ps.isChecked() else "dtfe"
        return "ps"

    def _estimator_group(self, tab: str, title: str, *first_rows) -> QGroupBox:
        """The deposit, the grid row, the GPU tick and the machine use of a run tab, after its own first rows (the
        Grids tab's estimator choice)."""
        g = group(title, *first_rows)
        deposit = QComboBox()
        for key in ("sampled", "exact"):
            deposit.addItem(P.label(key), key)
        self._deposit_texts(deposit, True, force=True)
        row, combo, cells, nsub = grid_row(self, self._changed)
        gpu = QCheckBox(P.label("gpu"))
        option_tip(gpu, "gpu")
        gpu.toggled.connect(self._changed)
        gpu.toggled.connect(lambda _c: self._gpu_set_by_hand(gpu))
        # (a 2D custom run has no GPU: _deposit_changed asks for the dimensions)
        deposit.currentIndexChanged.connect(
            lambda _i: self._deposit_changed(deposit, gpu, self._tab_estimator(tab),
                                             self.custom.dim if tab == "custom" else 3))
        # the machine use (--run-mode, the drivers' DTFE_RUN_MODE): the row after the GPU tick, which follows it
        mode = QComboBox()
        mode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)   # (sized like the precision's)
        mode.setMinimumContentsLength(24)
        for key in rs.RUN_MODES:
            mode.addItem(rs.RUN_MODE_TEXT[key][0], key)
            mode.setItemData(mode.count() - 1, rs.RUN_MODE_TEXT[key][1], Qt.ToolTipRole)
        option_tip(mode, "run_mode")
        mode.currentIndexChanged.connect(lambda _i: self._run_mode_changed(tab))
        g.form.addRow("Cell average", deposit)        # both estimators: exact (--exact, the default) or --sampled
        g.form.addRow("Grid", row)
        g.form.addRow("", gpu)
        g.form.addRow(P.label("run_mode"), mode)
        for role, w in (("deposit", deposit), ("grid", combo), ("grid_cells", cells), ("nsub", nsub), ("gpu", gpu),
                        ("run_mode", mode)):
            self._set_shared(tab, role, w)
        return g

    def _ps_options_group(self, tab: str) -> QGroupBox:
        """The phase-space options of a run tab, the face gradient last (its expert ones are in the Advanced group,
        which pairs the face gradient with the linear deposit: built after this group, it finds both boxes)."""
        g = QGroupBox(L["ps_options"])
        v = QVBoxLayout(g)
        for key in PS_OPTIONS:
            cb = QCheckBox(P.label(key))
            option_tip(cb, key)
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
            self._set_shared(tab, key, cb)
        self._set_shared(tab, "ps_group", g)
        return g

    def _advanced_group(self, tab: str, *extra) -> Collapsible:
        """The expert options of a run tab, folded (P.LABELS 'advanced'): the cusps and the linear density, then the
        tab's own 'extra' (key, tick box) pairs. The summary names the ticked ones. Built after _ps_options_group: the
        linear density excludes two of that group's boxes, the volume weighting and the face gradient."""
        boxes = []
        for key in EXPERT_OPTIONS:
            cb = QCheckBox(P.label(key))
            option_tip(cb, key)
            cb.toggled.connect(self._changed)
            self._set_shared(tab, key, cb)
            boxes.append((key, cb))
        _exclusive(self._shared(tab, "linear_deposit"), self._shared(tab, "volume_weighted"))
        # the face gradient (in the phase-space group) needs the volume weighting, which the linear deposit excludes
        _exclusive(self._shared(tab, "face_gradient"), self._shared(tab, "linear_deposit"))
        boxes += list(extra)

        def summary():
            on = [P.ADVANCED_SHORT[k] for k, cb in boxes if cb.isChecked()]
            return ("on: " + ", ".join(on)) if on else L["all_off"]
        g = advanced_group(L["advanced"], f"{tab}.advanced", self._section_settings(), summary,
                           *[("", cb) for _key, cb in boxes])
        self._set_shared(tab, "advanced", g)
        return g

    def _section_settings(self):
        """Where the folded groups keep whether they are open: the launcher's settings, when it remembers."""
        return self.settings if self.remember else None

    def _update_summaries(self):
        for tab in RUN_TABS:
            for role in ("resources", "advanced"):
                g = self._shared(tab, role)
                if g is not None:
                    g.update_summary()

    # ================================================================ Grids tab
    def _build_grids_tab(self) -> QWidget:
        col = QVBoxLayout()
        col.addWidget(self._preset_group("grids"))
        est = QHBoxLayout()
        self.ps_radio = QRadioButton("PS-DTFE (phase space)")
        self.dtfe_radio = QRadioButton("Standard DTFE")
        grp = QButtonGroup(self)
        grp.addButton(self.ps_radio)
        grp.addButton(self.dtfe_radio)
        est.addWidget(self.ps_radio)
        est.addWidget(self.dtfe_radio)
        est.addStretch(1)
        col.addWidget(self._estimator_group("grids", L["estimator_grid"], ("Estimator", est)))
        # after the group (slots run in connection order): the deposit choice follows the estimator, then _changed
        self.ps_radio.toggled.connect(lambda on: self._estimator_switched(self.deposit_combo, on))
        self.ps_radio.toggled.connect(self._changed)

        g, self.field_checks = self._fields_group()
        self.fields_box = g
        self.fields_note = _note()
        g.layout().addWidget(self.fields_note, g.layout().rowCount(), 0, 1, 3)      # below the threshold row
        col.addWidget(g)
        col.addWidget(self._ps_options_group("grids"))

        self.slice_group = QGroupBox(L["slice"])
        f = QFormLayout(self.slice_group)
        self.plane_combo = QComboBox()
        self.plane_combo.currentIndexChanged.connect(self._changed)
        f.addRow("Sample points", self.plane_combo)
        self.vg_check = QCheckBox(P.label("slice_vel_grad"))
        option_tip(self.vg_check, "slice_vel_grad")
        self.vg_check.toggled.connect(self._changed)
        f.addRow("", self.vg_check)
        self.pc_combo = _app._pts_conventions_combo(self._changed)       # PS-DTFE: the plane's conventions
        f.addRow(P.label("pts_conventions"), self.pc_combo)
        col.addWidget(self.slice_group)

        col.addWidget(self._resources_group("grids"))
        col.addWidget(self._advanced_group("grids"))
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _deposit_changed(self, combo: QComboBox, gpu: QCheckBox, estimator: str = "ps", dim: int = 3):
        """Picking the exact deposit sets the GPU box to that deposit's device in the tab's machine use
        (rs.recommended_gpu, the drivers' automatic device): the GPU for both estimators in every machine use
        (PS-DTFE 1156 s against 1133 s on all CPU cores for TNG100-3-Dark at 512³, standard DTFE's exact cell
        average 439 s against 431 s; in performance a partitioned PS-DTFE run also deposits on the free cores); the
        box stays editable. A 2D run has no GPU."""
        if not self._loading:
            tab = self._gpu_tab(gpu)
            mode = self._tab_run_mode(tab)
            if combo.currentData() == "exact":
                want = rs.recommended_gpu("exact", self._tab_spec(tab).fields, None, estimator, dim=dim,
                                          run_mode=mode)
                built = rs.gpu_built(estimator, dim=dim)       # (either pair has the GPU when the single one has)
                changed = built and gpu.isChecked() != want
                if changed:
                    self._gpu_auto_set(gpu, want)
                elif built:
                    self._gpu_by_hand.discard(gpu)      # the box holds the rule's tick now, as if the rule had set it
                if changed or (built and estimator == "ps" and gpu.isChecked() and mode == "performance"):
                    self.status_message(self._exact_device_text(estimator, gpu.isChecked(), mode))
        self._changed()

    @staticmethod
    def _exact_device_text(estimator: str, on: bool, mode: str) -> str:
        """The status line of the exact deposit's device, in the words of the estimator and the machine use."""
        what = "exact deposit" if estimator == "ps" else "exact cell averages"
        if not on:
            return f"{what}: the GPU is unticked ({mode}: the CPU)"
        if estimator == "ps" and mode == "performance":
            return "the exact deposit runs on the GPU and the free cores together (performance)"
        if estimator == "ps":
            return (f"{what}: the GPU is ticked (about as fast as all CPU cores: 1156 s against 1133 s for "
                    "TNG100-3-Dark at 512³)")
        return f"{what}: the GPU is ticked (a tie with all CPU cores: 439 s against 431 s for TNG100-3-Dark at 512³)"

    def _gpu_auto_set(self, gpu: QCheckBox, on: bool):
        """a GPU tick set by a rule, not by hand; the box is the rule's again, so a change of the machine use
        moves it"""
        self._gpu_auto_setting = True
        try:
            gpu.setChecked(on)              # (its toggled signal runs _changed)
        finally:
            self._gpu_auto_setting = False
        self._gpu_by_hand.discard(gpu)

    def _gpu_set_by_hand(self, gpu: QCheckBox):
        """the box changed (by hand, a loaded spec, a preset or a rule). Changed by hand (neither by a rule nor while a
        spec or preset is loaded) it keeps its tick when the machine use changes (_run_mode_changed), this session"""
        if self._gpu_auto_setting or self._loading:
            self._gpu_by_hand.discard(gpu)
        else:
            self._gpu_by_hand.add(gpu)

    def _gpu_tab(self, gpu: QCheckBox) -> str:
        """The run tab a GPU box is on."""
        return next(t for t in RUN_TABS if self._shared(t, "gpu") is gpu)

    def _tab_spec(self, tab: str):
        """The job of a run tab."""
        return {"grids": self.spec, "custom": self.custom, "pipeline": self.pipe}[tab]

    def _tab_run_mode(self, tab: str) -> str:
        """The machine use a run tab's choices apply to now: from its menu (as _tab_estimator, before _read_widgets)."""
        return self._shared(tab, "run_mode").currentData() or "balanced"

    def _tab_particles(self, tab: str) -> int | None:
        """The particles of a run tab's input, where it says: a TNG simulation's DM count (the Pipeline: its largest
        simulation's), the custom snapshot's HDF5 header; None when unknown."""
        if tab == "grids":
            return rs.dm_particles(self.spec.sim)
        if tab == "custom":
            return rs.hdf5_particle_count(Path(self.custom.input_file).expanduser()) if self.custom.input_file else None
        return max((rs.dm_particles(s) or 0 for s in self.pipe.sims), default=0) or None

    def _run_mode_changed(self, tab: str):
        """A new machine use on a run tab: its GPU tick follows it (rs.recommended_gpu, the drivers' automatic device:
        standard DTFE's sampled average on the CPU in performance, the GPU otherwise) unless it was set by hand
        (_gpu_set_by_hand), and the status line says what the machine use does with the run."""
        if self._loading:
            return
        self._read_widgets()
        spec, gpu, est = self._tab_spec(tab), self._shared(tab, "gpu"), self._tab_estimator(tab)
        dim = getattr(spec, "dim", 3)
        changed, built = False, rs.gpu_built(est, spec.precision, dim)
        if gpu not in self._gpu_by_hand and built:
            want = rs.recommended_gpu(spec.deposit, spec.fields, self._tab_particles(tab), est, spec.precision, dim,
                                      run_mode=spec.run_mode)
            changed = gpu.isChecked() != want
            if changed:
                self._gpu_auto_set(gpu, want)       # (its toggled signal runs _changed)
        text = rs.RUN_MODE_TEXT[spec.run_mode][0]
        if built and gpu.isChecked() and est == "ps" and spec.deposit == "exact" and spec.run_mode == "performance":
            text += "; the exact deposit runs on the GPU and the free cores together (performance)"
        elif changed:
            text += "; the GPU is ticked" if gpu.isChecked() else "; the GPU is unticked"
            if not gpu.isChecked() and est == "dtfe" and spec.deposit == "sampled":
                text += (" (standard DTFE's sampled average is faster on the CPU: 315 s against 433 s for "
                         "TNG100-3-Dark at 512³)")
        self.status_message(text)
        self._changed()

    def _estimator_switched(self, combo: QComboBox, ps_now: bool):
        """An estimator switch by the user moves the deposit choice to the new estimator's production setting
        (rs.production_deposit): the exact one for both (PS-DTFE's exact deposit, standard DTFE's exact cell
        average; both drivers' default); keeping 'sampled' would run a standard job on the biased Monte-Carlo
        average unless it was changed by hand. (With two choices, 'a choice at the old estimator's production
        setting follows to the new one's' is the same rule.) The other choice stays selectable afterwards. A choice
        made for an estimator comes back with it (per tab, this session): else a sampled deposit chosen for PS-DTFE
        would be lost on the way PS -> standard -> PS, the switch back giving the exact one."""
        if self._loading:
            return
        new, old = ("ps", "dtfe") if ps_now else ("dtfe", "ps")
        kept = self._deposit_kept.setdefault("grids" if combo is self.deposit_combo else "own", {})
        if combo.currentData():
            kept[old] = combo.currentData()          # the estimator left: its choice, for the way back
        want = kept.get(new) or rs.production_deposit(new)
        gpu = self.gpu_check if combo is self.deposit_combo else self.o_gpu
        if combo.currentData() == want == "exact":
            # the choice stays, the GPU rule is the new estimator's (rs.recommended_gpu: the two may part):
            # applied as a pick of the choice would
            self._deposit_changed(combo, gpu, new, 3 if combo is self.deposit_combo else self.custom.dim)
        elif combo.currentData() != want:
            was = gpu.isChecked()
            combo.setCurrentIndex(max(combo.findData(want), 0))      # its signal: the GPU tick, _changed
            now = gpu.isChecked()
            where = (", on the GPU" if now and not was else ", on the CPU" if was and not now else "")
            words = ({"exact": "the exact deposit", "sampled": "the sampled deposit"} if ps_now else
                     {"exact": "the exact cell average", "sampled": "the Monte-Carlo average"})
            how = (" again, as chosen for it before" if want != rs.production_deposit(new)   # the user's own choice
                   else ", its production setting")
            self.status_message(f"{'PS-DTFE' if ps_now else 'standard DTFE'}: {words[want]}{how}{where}; "
                                f"{words['sampled' if want == 'exact' else 'exact']} stays selectable")

    @staticmethod
    def _deposit_texts(combo: QComboBox, ps: bool, force: bool = False):
        """the deposit choice in the words of the estimator it applies to (the same two settings): each item's
        tooltip the short one, the combo's What's This both whole descriptions"""
        binary = "PS-DTFE" if ps else "DTFE"
        keys = [key if ps else key + "_dtfe" for key in ("sampled", "exact")]
        if not force and all(combo.itemText(i) == P.label(k) for i, k in enumerate(keys)):
            return
        for i, k in enumerate(keys):
            combo.setItemText(i, P.label(k))
            combo.setItemData(i, P.short_tooltip(k, binary), Qt.ToolTipRole)
        combo.setWhatsThis("\n\n".join(f"{P.label(k)}\n{P.tooltip(k, binary)}" for k in keys))

    def _fields_group(self, title: str = L["fields"]) -> tuple[QGroupBox, dict[str, QCheckBox]]:
        g = QGroupBox(title)
        grid, checks = field_grid([(name, field_label(name, label)) for name, label in rs.PS_FIELDS], 3, self._changed)
        g.setLayout(grid)
        row = QHBoxLayout()                 # the T-web / V-web eigenvalue threshold: only with a web class ticked
        lam = QDoubleSpinBox(minimum=0.0, maximum=10.0, decimals=2, singleStep=0.05, value=rs.LAMBDA_TH_DEFAULT)
        tip(lam, "λ_th of the T-web / V-web classes: an eigenvalue above it counts as collapsing.\n"
                 "0.3 is the scripts' default; the binary alone would use 0.",
            "A void has no eigenvalue above λ_th, a wall one, a filament two, a node three.")
        lam.valueChanged.connect(self._changed)
        row.addWidget(QLabel("T-web / V-web threshold λ_th"))
        row.addWidget(lam)
        row.addStretch(1)
        grid.addLayout(row, (len(rs.PS_FIELDS) + 2) // 3, 0, 1, 3)
        g.lambda_spin = lam

        def web_on(_on=False):
            lam.setEnabled(any(checks[w].isChecked() and checks[w].isEnabled() for w in rs.WEB_FIELDS))
        for w in rs.WEB_FIELDS:
            checks[w].toggled.connect(web_on)
        g.sync_lambda = web_on
        web_on()
        return g, checks

    def _resources_group(self, tab: str) -> Collapsible:
        """A run tab's partitioning, memory, precision and output options, folded to a one-line summary
        (_resources_summary). The Custom snapshot tab has no output prefix (its output is named in its Output group)
        and no start-over tick (a Custom run never resumes)."""
        g = Collapsible(L["resources"], f"{tab}.resources", self._section_settings(),
                        lambda: self._resources_summary(tab))
        f = g.form
        part = QSpinBox(minimum=0, maximum=8)
        part.setSpecialValueText("auto")
        part.setSuffix("³ partitions")
        part.valueChanged.connect(self._changed)
        f.addRow("Partition", part)
        conc = QSpinBox(minimum=0, maximum=16)
        conc.setSpecialValueText("auto")
        conc.valueChanged.connect(self._changed)
        f.addRow("Concurrent triangulations", conc)   # --max-concurrent: the triangulations built at once, either binary
        scratch = QLineEdit()
        scratch.setPlaceholderText("none: grids in RAM")
        scratch.editingFinished.connect(self._changed)
        f.addRow("Scratch directory", _dir_row(scratch, lambda: self._browse_scratch(scratch)))
        ptri = QCheckBox(P.label("parallel_triangulation"))
        option_tip(ptri, "parallel_triangulation")
        ptri.toggled.connect(self._changed)
        f.addRow("", ptri)
        # K1: the partitions merged in index order, so two identical runs give bit-identical grids
        ordered = QCheckBox(P.label("ordered_merge"))
        option_tip(ordered, "ordered_merge")
        ordered.toggled.connect(self._changed)
        f.addRow("", ordered)
        widgets = {"partition": part, "concurrent": conc, "scratch": scratch, "parallel_triangulation": ptri,
                   "ordered_merge": ordered}
        if SHARED_WIDGETS["start_over"][RUN_TABS.index(tab)]:
            # a killed partitioned run's checkpoint is resumed by the next run of the same job; this starts it over
            # instead (the only way out when the resume is refused: other settings, an incomplete checkpoint)
            start_over = QCheckBox("Start over: do not resume a killed run's checkpoint")
            tip(start_over, "Ticked, the run starts from the beginning and replaces the checkpoint of a killed run\n"
                            "(DTFE_RESUME=0, both run scripts).",
                "A partitioned run that was killed keeps a checkpoint of its finished partitions, and the next run of "
                "the same job continues from it. Starting over is needed when the settings changed since, as the "
                "program then refuses to resume.")
            start_over.toggled.connect(self._changed)
            f.addRow("", start_over)
            widgets["start_over"] = start_over
        prec = QComboBox()
        prec.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)   # its long double-precision text
        prec.setMinimumContentsLength(24)       # else the tab gets wider than the pane: the '…' buttons get cut off
        for key in rs.PRECISIONS:
            prec.addItem(rs.PRECISION_TEXT[key], key)
        tip(prec, "Which pair of programs runs: single precision (the default), or the double-precision pair\n"
                  "(float64 from the input read onward, about twice the memory). Setup builds the pair.",
            "The double-precision pair keeps every position, density and field in float64 from the input read "
            "onward (TNG stores its coordinates in double) and writes float64 grids.")
        prec.currentIndexChanged.connect(self._changed)
        f.addRow("Precision", prec)
        widgets["precision"] = prec
        if SHARED_WIDGETS["prefix"][RUN_TABS.index(tab)]:
            prefix = QLineEdit()
            prefix.editingFinished.connect(self._changed)
            f.addRow("Output prefix", prefix)
            widgets["prefix"] = prefix
        budget = QDoubleSpinBox(minimum=0.0, maximum=4096.0, decimals=0, singleStep=4.0)
        budget.setSpecialValueText("auto (free memory)")
        tip(budget, "The memory the auto-tuner plans the partitions and concurrency for (DTFE_MEM_BUDGET_GB).\n"
                    "Auto: what is free when the run starts.",
            "Set it lower to leave room for other work, higher when you know the memory will be free (it may "
            "swap).")
        budget.valueChanged.connect(self._changed)
        f.addRow(*labelled_row("Memory budget", budget, " GB"))
        tess = QLineEdit()
        tess.setPlaceholderText("none: triangulate every run")
        tip(tess, "Reuse the Delaunay tessellation of an identical earlier run (the tessellation cache):\n"
                  "a LOCAL folder (iCloud is refused); about 0.15 KB per vertex of disk.",
            "The tessellation is the dominant fixed cost of a rerun with other fields, grids or sample points "
            "(measured 10.9 s -> 4.6 s).")
        tess.editingFinished.connect(self._changed)
        f.addRow("Tessellation cache", _dir_row(tess, lambda: self._browse_dir(tess, "Tessellation cache folder")))
        widgets.update(budget=budget, tess=tess)
        for sb in (part, conc, budget):     # macOS draws a spin box's 'auto' text against its left edge: some room
            sb.setMinimumWidth(sb.sizeHint().width() + 8)
        for role, w in widgets.items():
            self._set_shared(tab, role, w)
        self._set_shared(tab, "resources", g)
        return g

    def _resources_summary(self, tab: str) -> str:
        """The folded Resources group's one line: 'auto plan · 4 triangulations at once · 40 GB budget …'."""
        w = lambda role: self._shared(tab, role)            # noqa: E731
        part, conc = w("partition"), w("concurrent")
        parts = [L["auto_plan"] if not part.value() else f"{part.value()}{part.suffix()}",
                 L["auto_concurrency"] if not conc.value() else
                 f"{conc.value()} triangulation{'s' if conc.value() != 1 else ''} at once"]
        if w("budget").value():
            parts.append(f"{w('budget').value():g} GB budget")
        parts.append("double precision" if w("precision").currentData() == "double" else "single precision")
        if w("scratch").text().strip():
            parts.append("grids on scratch disk")
        if w("prefix") is not None and w("prefix").text().strip():
            parts.append(f"prefix {w('prefix').text().strip()}")
        if w("tess").text().strip():
            parts.append("tessellation cache")
        if w("parallel_triangulation").isChecked():
            parts.append("parallel triangulation")
        if not w("ordered_merge").isChecked():
            parts.append("merge in finishing order")
        if w("start_over") is not None and w("start_over").isChecked():
            parts.append("start over")
        return " · ".join(parts)

    # ================================================================ Custom snapshot tab
    def _build_own_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Snapshot")
        f = QFormLayout(g)
        self.o_file = QLineEdit(placeholderText="choose a snapshot file")
        self.o_file.editingFinished.connect(self._own_file_changed)
        f.addRow("File", _dir_row(self.o_file, self._browse_own_file))
        self.o_type = QComboBox()
        for t, text in rs.INPUT_TYPES.items():
            self.o_type.addItem(text, t)
        self.o_type.currentIndexChanged.connect(self._changed)
        f.addRow("Format", self.o_type)
        self.o_dim = QComboBox()
        for d in rs.DIMS:
            self.o_dim.addItem(rs.DIM_TEXT[d], d)
        tip(self.o_dim, "3D, or 2D for a snapshot in a plane (its coordinates have two columns).\n"
                        "Set from the file when you choose one (HDF5 and text files say which).",
            "The 2D programs DTFE-2d and PS-DTFE-2d run a 2D snapshot, on the CPU, with every field (the 2D T-web "
            "classes: 0 void, 1 filament, 2 node).")
        self.o_dim.currentIndexChanged.connect(self._changed)
        f.addRow("Dimensions", self.o_dim)
        un = QHBoxLayout()
        self.o_unit = QDoubleSpinBox(minimum=1e-6, maximum=1e9, decimals=4, value=1.0)
        self.o_unit.setToolTip("How many of the file's length units make one Mpc: 1 when the coordinates "
                               "are in Mpc, 1000 in kpc (e.g. Gadget's usual kpc/h: 1000, and h then stays "
                               "in the lengths)")
        self.o_unit.valueChanged.connect(self._changed)
        un.addWidget(QLabel("1 Mpc ="))
        un.addWidget(self.o_unit)
        un.addWidget(QLabel("file units"))
        for text, value in (("Mpc", 1.0), ("kpc", 1000.0)):
            b = QToolButton(text=text)
            b.clicked.connect(lambda _=False, v=value: self.o_unit.setValue(v))
            un.addWidget(b)
        un.addStretch(1)
        f.addRow("Length unit", un)
        self.o_periodic = QCheckBox(P.label("periodic"))
        option_tip(self.o_periodic, "periodic", "DTFE")
        self.o_periodic.toggled.connect(self._changed)
        f.addRow("", self.o_periodic)
        self.o_alpha = QDoubleSpinBox(minimum=0.0, maximum=50.0, decimals=1, singleStep=0.5, value=3.0)
        self.o_alpha.setSuffix(" mean particle spacings")
        self.o_alpha.setSpecialValueText("whole convex hull")
        option_tip(self.o_alpha, "alpha_shape")
        self.o_alpha.valueChanged.connect(self._changed)
        f.addRow("Alpha shape", self.o_alpha)
        self.o_snap_form = f
        self.o_box = QLineEdit(placeholderText="from the file header (Gadget BoxSize)")
        self.o_box.setToolTip("Optional: xlo xhi ylo yhi zlo zhi in the file's units (all >= 0)")
        self.o_box.editingFinished.connect(self._changed)
        f.addRow("Box", self.o_box)
        col.addWidget(g)

        g = QGroupBox("Estimator")
        f = QFormLayout(g)
        est = QHBoxLayout()
        self.o_ps = QRadioButton("PS-DTFE (phase space: streams, caustics)")
        self.o_dtfe = QRadioButton("Standard DTFE")
        grp = QButtonGroup(self)
        grp.addButton(self.o_ps)
        grp.addButton(self.o_dtfe)
        self.o_ps.toggled.connect(lambda on: self._estimator_switched(self.o_deposit, on))   # (the Grids tab's rule)
        self.o_ps.toggled.connect(self._changed)
        est.addWidget(self.o_ps)
        est.addWidget(self.o_dtfe)
        est.addStretch(1)
        f.addRow("", est)
        lg = QVBoxLayout()
        self.o_lag_file = QRadioButton("Initial positions stored in the snapshot ('InitialCoordinates')")
        self.o_lag_sep = QRadioButton("Initial positions in a separate snapshot (matched by particle ID):")
        grp2 = QButtonGroup(self)
        grp2.addButton(self.o_lag_file)
        grp2.addButton(self.o_lag_sep)
        self.o_lag_file.toggled.connect(self._changed)
        lg.addWidget(self.o_lag_file)
        lg.addWidget(self.o_lag_sep)
        self.o_lag_path = QLineEdit(placeholderText="the initial-conditions snapshot (Gadget HDF5)")
        self.o_lag_path.editingFinished.connect(self._changed)
        lg.addLayout(_dir_row(self.o_lag_path, lambda: self._browse_file(self.o_lag_path, "Initial conditions")))
        self.o_lag_box = QWidget()
        self.o_lag_box.setLayout(lg)
        f.addRow("Lagrangian", self.o_lag_box)
        self.o_scalar = QLineEdit(placeholderText="none (e.g. Potential)")
        option_tip(self.o_scalar, "scalar")
        self.o_scalar.editingFinished.connect(self._changed)
        f.addRow("Per-particle value", self.o_scalar)
        col.addWidget(g)

        col.addWidget(self._preset_group("custom"))
        col.addWidget(self._estimator_group("custom", L["grid_deposit"]))
        col.addWidget(self._ps_options_group("custom"))
        g = QGroupBox("Points and window")
        f = QFormLayout(g)
        self.o_points = QLineEdit(placeholderText="none: the grid only")
        option_tip(self.o_points, "sample_points")
        self.o_points.editingFinished.connect(self._changed)
        f.addRow("Sample points", _dir_row(self.o_points, lambda: (self._browse_file(self.o_points, "Sample points file"),
                                                                    self._changed())))
        self.o_pvg = QCheckBox(P.label("slice_vel_grad"))
        option_tip(self.o_pvg, "slice_vel_grad")
        self.o_pvg.toggled.connect(self._changed)
        f.addRow("", self.o_pvg)
        self.o_pc = _app._pts_conventions_combo(self._changed)           # PS-DTFE: the points' conventions
        f.addRow(P.label("pts_conventions"), self.o_pc)
        self.o_window = QLineEdit(placeholderText="none: the whole grid")
        option_tip(self.o_window, "window", more="\n\nGive x0 x1 y0 y1 z0 z1 in Mpc (2D: four numbers).")
        self.o_window.editingFinished.connect(self._changed)
        f.addRow("Grid window (Mpc)", self.o_window)     # --grid-window (not the Pipeline's plane window)
        self.o_sdens = QComboBox()
        self.o_sdens.addItem("linear: the DTFE interpolation of the vertex densities (default)", "linear")
        self.o_sdens.addItem("geometric: the tetrahedron's mass over its volume", "geometric")
        option_tip(self.o_sdens, "stream_density")
        self.o_sdens.currentIndexChanged.connect(self._changed)
        f.addRow("Stream density", self.o_sdens)
        hr = QHBoxLayout()
        self.o_halo = QDoubleSpinBox(minimum=0.0, maximum=1e7, decimals=0, singleStep=100.0)
        self.o_halo.setSpecialValueText("off")
        option_tip(self.o_halo, "halo_release")
        self.o_halo.valueChanged.connect(self._changed)
        hr.addWidget(self.o_halo)
        hr.addStretch(1)
        f.addRow("Halo release (ρ/ρ̄)", hr)
        self.o_points_box = g
        col.addWidget(g)
        g, self.o_fields = self._fields_group()
        self.o_fields_box = g
        col.addWidget(g)

        g = QGroupBox("More options")
        f = QFormLayout(g)
        self.o_extra = QLineEdit(placeholderText="options the tab has no control for, as on a command line")
        tip(self.o_extra, "Any of the program's options the tab has no control for, typed as on a command line\n"
                          "(see the program's help): they go after the tab's own.",
            "An option the tab sets already is reported, not passed twice.")
        self.o_extra.editingFinished.connect(self._changed)
        f.addRow("Extra options", self.o_extra)
        col.addWidget(g)

        g = QGroupBox("Output")
        f = QFormLayout(g)
        self.o_outdir = QLineEdit()
        self.o_outdir.editingFinished.connect(self._changed)
        f.addRow("Folder", _dir_row(self.o_outdir, lambda: self._browse_dir(self.o_outdir, "Output folder")))
        self.o_name = QLineEdit(placeholderText="the snapshot's file name")
        self.o_name.editingFinished.connect(self._changed)
        f.addRow("Name", self.o_name)
        col.addWidget(g)
        col.addWidget(self._resources_group("custom"))
        # the sample points' expert outputs (the points' group enables them with the points)
        self.o_pstream = QCheckBox(P.label("per_stream"))
        option_tip(self.o_pstream, "per_stream")
        self.o_pids = QCheckBox(P.label("per_stream_ids"))
        option_tip(self.o_pids, "per_stream_ids")
        for cb in (self.o_pstream, self.o_pids):
            cb.toggled.connect(self._changed)
        col.addWidget(self._advanced_group("custom", ("per_stream", self.o_pstream),
                                           ("per_stream_ids", self.o_pids)))
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _browse_file(self, edit: QLineEdit, title: str):
        start = edit.text() or str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, title, start)
        if f:
            edit.setText(f)
            self._changed()

    def _browse_dir(self, edit: QLineEdit, title: str):
        d = QFileDialog.getExistingDirectory(self, title, edit.text() or str(Path.home()))
        if d:
            edit.setText(d)
            self._changed()

    def _browse_own_file(self):
        self._browse_file(self.o_file, "Your snapshot file")
        self._own_file_changed()

    def _own_file_changed(self):
        """A new file: guess its format from the name and tick the GPU when its particle count says
        the GPU deposit is the faster one (the user can still change both)."""
        f = self.o_file.text().strip()
        if not f or f == self._own_last_file:
            return
        self._own_last_file = f
        suffix = Path(f).suffix.lower()
        guess = 105 if suffix in (".hdf5", ".h5", ".hdf") else 111 if suffix in (".txt", ".dat", ".csv") else None
        if guess is not None:
            self.o_type.setCurrentIndex(max(self.o_type.findData(guess), 0))
        dim = rs.snapshot_dim(Path(f).expanduser(), guess) if guess is not None else None
        if dim is not None:                 # the file says: 2D or 3D
            self.o_dim.setCurrentIndex(max(self.o_dim.findData(dim), 0))
        if guess == 105 and Path(f).expanduser().is_file():
            self._read_widgets()
            c = self.custom
            n = rs.hdf5_particle_count(Path(f).expanduser())
            self.o_gpu.setChecked(rs.recommended_gpu(c.deposit, c.fields, n, c.estimator, c.precision, c.dim,
                                                     run_mode=c.run_mode))
            self._gpu_by_hand.discard(self.o_gpu)          # the file's rule set it, not a hand
        self._changed()

    # ================================================================ Pipeline tab
    def _build_pipeline_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Simulations")
        v = QVBoxLayout(g)
        self.p_sims = QListWidget()
        self.p_sims.setMaximumHeight(100)
        self.p_sims.itemChanged.connect(self._changed)
        self.p_sims.setDragDropMode(QAbstractItemView.InternalMove)     # the run order: drag a simulation up or down
        self.p_sims.setToolTip("Tick the simulations to run. They run from the top down: drag one to change the order")
        self.p_sims.model().rowsMoved.connect(lambda *_: self._changed())
        v.addWidget(self.p_sims)
        v.addWidget(_note("Computes every snapshot whose slice is missing or out of date."))
        col.addWidget(g)

        g = QGroupBox("Image plane")
        g.setToolTip("A full-box slice, evaluated at every pixel (point evaluation)")
        f = QFormLayout(g)
        self.p_nu = QComboBox()
        self.p_nu.setEditable(True)
        self.p_nu.addItems(["2048", "4096", "8192", "16384"])
        self.p_nu.setValidator(QIntValidator(16, 65536, self))
        self.p_nu.currentTextChanged.connect(self._changed)
        f.addRow("Pixels across the box", self.p_nu)
        self.p_adaptive = QCheckBox("Adaptive: the largest size that fits, per snapshot")
        self.p_adaptive.setToolTip("Needs 'Check memory': it asks the binary what each snapshot needs at "
                                   "each size (halving down to 1024²) and plans the pipeline accordingly.")
        self.p_adaptive.toggled.connect(self._changed)
        f.addRow("", self.p_adaptive)
        self.p_axis = QComboBox()
        self.p_axis.addItems(["x", "y", "z"])
        self.p_axis.currentIndexChanged.connect(self._changed)
        f.addRow("Normal axis", self.p_axis)
        self.p_center = QLineEdit(placeholderText="box centre")
        self.p_center.editingFinished.connect(self._changed)
        f.addRow("Position (Mpc)", self.p_center)
        self.p_vg = QCheckBox("Velocity gradient at each point")
        self.p_vg.setToolTip("Also evaluate the velocity gradient at every sample point: the divergence, shear and "
                             "vorticity maps (PTS_VEL_GRAD)")
        self.p_vg.toggled.connect(self._changed)
        f.addRow("", self.p_vg)
        self.p_pc = _app._pts_conventions_combo(self._changed)           # the image plane's conventions
        f.addRow(P.label("pts_conventions"), self.p_pc)
        slab = QHBoxLayout()
        self.p_planes = QSpinBox(minimum=1, maximum=256)
        tip(self.p_planes, "Sampling planes spread across a slab, which the figures' 'slab' projection averages.\n"
                           f"1 = one crisp cross-section; else at least {rs.PLANE_GHOSTING_MIN} for a smooth projection.",
            "The planes option of make_image_plane.py. One plane is a zero-thickness cross-section. A handful of "
            "planes ghosts every inclined structure (one displaced copy per plane).")
        self.p_planes.valueChanged.connect(self._changed)
        self.p_thick = QDoubleSpinBox(minimum=0.01, maximum=1000.0, decimals=2, singleStep=0.5)
        self.p_thick.setSuffix(" Mpc")
        self.p_thick.setToolTip("The slab's depth along the normal axis, over which the planes are spread "
                                "(the thickness option of make_image_plane.py; one plane has no thickness)")
        self.p_thick.valueChanged.connect(self._changed)
        slab.addWidget(self.p_planes)
        slab.addWidget(QLabel("planes across"))
        slab.addWidget(self.p_thick)
        slab.addStretch(1)
        f.addRow("Slab", slab)
        sub = QHBoxLayout()
        self.p_super = QSpinBox(minimum=1, maximum=8)
        tip(self.p_super, "K: evaluate K×K sample points per pixel and average them (costs K² points):\n"
                          "the pixel's mean, as the deposit grid reports it, without aliasing.",
            "The supersample option of make_image_plane.py. The map then estimates the pixel-AREA mean instead of a "
            "point sample: no aliasing where structure is finer than a pixel (halo cores, thin caustics).")
        self.p_super.valueChanged.connect(self._changed)
        self.p_super_note = QLabel()
        sub.addWidget(self.p_super)
        sub.addWidget(self.p_super_note)
        sub.addStretch(1)
        f.addRow("Sub-samples per pixel axis", sub)
        self.p_window = QLineEdit(placeholderText="whole box: u0 u1 v0 v1")
        tip(self.p_window, "Image only this part of the plane: horizontal u0..u1 and vertical v0..v1 in Mpc;\n"
                           "blank = the whole box.",
            "The plane window of make_image_plane.py (PLANE_WINDOW). Pixels stay square, so the vertical pixel count "
            "follows the window's aspect.")
        self.p_window.editingFinished.connect(self._changed)
        f.addRow("Plane window (Mpc)", self.p_window)
        col.addWidget(g)

        # the grid, deposit and GPU in a group of their own (as the Grids tab's 'Estimator and grid'): added to the
        # fields' grid layout they shared a row with its T-web threshold row
        tip_short = "run_ps_pipeline.sh hands these to run_ps_dtfe.sh exactly as a Grids job does."
        tip_more = ("Left alone, the scripts' own defaults are the production settings (GPU, vertex masses, "
                    "volume-weighted velocities).")
        g = self._estimator_group("pipeline", L["grid_deposit"])
        tip(g, tip_short, tip_more)
        col.addWidget(g)

        g, self.p_fields = self._fields_group()
        self.p_fields_box = g
        tip(g, tip_short, tip_more)
        col.addWidget(g)
        col.addWidget(self._ps_options_group("pipeline"))
        col.addWidget(self._resources_group("pipeline"))

        g = QGroupBox("Run")
        v = QVBoxLayout(g)
        self.p_force = QCheckBox("Recompute every snapshot, up to date or not")
        self.p_plan = QCheckBox("Plan only: show what would run")
        self.p_render = QCheckBox("Then render the point-evaluated figures")
        self.p_force.setToolTip("Ignore every freshness check: recompute even snapshots whose grids are newer than "
                                "their inputs")
        self.p_plan.setToolTip("Print what would run for each snapshot, and compute nothing")
        self.p_render.setToolTip("plot_pointeval.py, on this pipeline's own image plane, with the options below")
        for cb in (self.p_force, self.p_plan, self.p_render):
            cb.toggled.connect(self._changed)
            v.addWidget(cb)
        self.p_render_box = QWidget()
        rf = QFormLayout(self.p_render_box)
        rf.setContentsMargins(INDENT, 0, 0, 0)
        self.p_rfields = QLineEdit(placeholderText="all the run wrote: " + ", ".join(rs.POINTEVAL_FIELDS))
        self.p_rfields.setToolTip("A subset of the point-evaluated fields, separated by spaces (the maps option of "
                                  "plot_pointeval.py); blank = every field the run wrote")
        self.p_rfields.editingFinished.connect(self._changed)
        rf.addRow("Fields", self.p_rfields)
        self.p_froot = QLineEdit(placeholderText=f"default: {rs.FIGURES_ROOT}")
        self.p_froot.setToolTip("Where the rendered figures go")
        self.p_froot.editingFinished.connect(self._changed)
        rf.addRow("Figures folder", _dir_row(self.p_froot, lambda: (self._browse_dir(self.p_froot, "Figures folder"),
                                                                    self._changed())))
        self.p_rproject = QComboBox()
        self.p_rproject.addItem("the central plane: a zero-thickness cross-section", "plane")
        self.p_rproject.addItem("slab: the mean over every plane of the slab", "slab")
        self.p_rproject.setToolTip("The projection option of plot_pointeval.py; 'slab' needs several planes across the "
                                   "slab (above)")
        self.p_rproject.currentIndexChanged.connect(self._changed)
        rf.addRow("Projection", self.p_rproject)
        self.p_rsmooth = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=1, singleStep=0.5)
        self.p_rsmooth.valueChanged.connect(self._changed)
        rf.addRow(*labelled_row("Smoothing", self.p_rsmooth, " px",
                                "Gaussian smoothing of every map in output pixels (plot_pointeval.py's smooth); 0 = none.\n"
                                "It also suppresses the smoothed gradient companions below."))
        self.p_rsmoothd = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=1, singleStep=1.0)
        self.p_rsmoothd.setSuffix(" px")
        tip(self.p_rsmoothd, "Each gradient map also gets a Gaussian-smoothed companion figure at this width;\n"
                             "0 = none.",
            "The gradient maps (divergence, shear, vorticity, density gradient) are piecewise constant per tetrahedron "
            "and look faceted. The smooth-derivatives option of plot_pointeval.py.")
        self.p_rsmoothd.valueChanged.connect(self._changed)
        rf.addRow("Gradient maps also smoothed at", self.p_rsmoothd)
        v.addWidget(self.p_render_box)
        col.addWidget(g)
        # the render step's expert options (enabled with the render step)
        self.p_rfixed = QCheckBox("Fixed density range of the grid maps (0.1 .. 1e4)")
        tip(self.p_rfixed, "The fixed-range option of plot_pointeval.py: comparable side by side with the grid maps;\n"
                           "clips ~26% of a z=0 image to the darkest colour")
        self.p_rforce = QCheckBox("Re-render figures newer than their data")
        self.p_rforce.setToolTip("The force option of plot_pointeval.py")
        for cb in (self.p_rfixed, self.p_rforce):
            cb.toggled.connect(self._changed)
        col.addWidget(self._advanced_group("pipeline", ("render_fixed_range", self.p_rfixed),
                                           ("render_force", self.p_rforce)))
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    # ================================================================ Plots tab
    def _build_plots_tab(self) -> QWidget:
        col = QVBoxLayout()
        g = QGroupBox("Figure sets")
        v = QVBoxLayout(g)
        self.fig_list = QListWidget()
        self.fig_list.setMaximumHeight(200)
        for fs in rs.FIGURE_SETS:
            it = QListWidgetItem(fs.title)
            it.setData(Qt.UserRole, fs.key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.fig_list.addItem(it)
        self.fig_list.itemChanged.connect(self._changed)
        v.addWidget(self.fig_list)
        self.fig_stack = QStackedWidget()
        for fs in rs.FIGURE_SETS:
            self.fig_stack.addWidget(self._figure_page(fs))
        self.fig_list.currentRowChanged.connect(self.fig_stack.setCurrentIndex)
        v.addWidget(self.fig_stack)
        col.addWidget(g)

        g = QGroupBox("Grids to plot")
        f = QFormLayout(g)
        self.pl_method = QComboBox()
        for key, label in (("auto", "Auto: PS-DTFE where present, else DTFE"), ("ps", "PS-DTFE"),
                           ("dtfe", "Standard DTFE")):
            self.pl_method.addItem(label, key)
        self.pl_method.currentIndexChanged.connect(self._changed)
        f.addRow("Estimator", self.pl_method)
        self.pl_prefix = QLineEdit(placeholderText="default (ps_output)")
        self.pl_prefix.editingFinished.connect(self._changed)
        f.addRow("Grid prefix", self.pl_prefix)
        self.pl_smooth = QDoubleSpinBox(minimum=0.0, maximum=50.0, decimals=1, singleStep=0.5)
        self.pl_smooth.valueChanged.connect(self._changed)
        f.addRow(*labelled_row("Smoothing", self.pl_smooth, " cells"))
        self.pl_froot = QLineEdit(placeholderText=f"default: {rs.FIGURES_ROOT}")
        self.pl_froot.setToolTip(f"Where the point-evaluated maps go (the other figure sets write under\n{rs.FIGURES_ROOT})")
        self.pl_froot.editingFinished.connect(self._changed)
        f.addRow("Figures folder", _dir_row(self.pl_froot, lambda: (self._browse_dir(self.pl_froot, "Figures folder"),
                                                                     self._changed())))
        col.addWidget(g)
        col.addStretch(1)
        w = QWidget()
        w.setLayout(col)
        return w

    def _figure_page(self, fs: rs.FigureSet) -> QWidget:
        page = QWidget()
        f = QFormLayout(page)
        f.setContentsMargins(GROUP_MARGIN, GROUP_MARGIN, GROUP_MARGIN, GROUP_MARGIN)
        about = QLabel(fs.about)
        about.setWordWrap(True)
        about.setToolTip(f"python/{fs.script}, {_app.PER_TEXT[fs.per]}\n→  {rs.FIGURES_ROOT / fs.output}")
        f.addRow(about)
        rows: dict[str, QHBoxLayout] = {}
        for o in fs.opts:
            if o.kind == "bool":
                w = QCheckBox(o.label)
                w.toggled.connect(self._changed)
            elif o.kind == "int":
                w = QSpinBox(minimum=0, maximum=1_000_000)
                w.valueChanged.connect(self._changed)
            elif o.kind == "float":
                lo, hi, dec = o.limits or (0.0, 1000.0, 2)
                w = QDoubleSpinBox(decimals=dec, singleStep=0.5)     # decimals FIRST: the range rounds to them
                w.setRange(lo, hi)
                w.valueChanged.connect(self._changed)
            elif o.kind == "choice":
                w = QComboBox()
                w.addItems(list(o.choices))
                w.currentIndexChanged.connect(self._changed)
            elif o.kind == "plane":
                w = QComboBox()
                w.currentIndexChanged.connect(self._changed)
            elif o.kind == "multi":
                w = QListWidget()
                for choice in o.choices:
                    it = QListWidgetItem(choice)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked)
                    w.addItem(it)
                w.setMaximumHeight(min(len(o.choices), 8) * 20 + 8)
                w.itemChanged.connect(self._changed)
                _checkable(w)
            else:
                w = QLineEdit(placeholderText=o.help)
                w.editingFinished.connect(self._changed)
            if o.help:
                tip_auto(w, o.help)
            self.opt_widgets[(fs.key, o.key)] = (o, w)
            if not o.row:
                if o.kind == "bool":
                    f.addRow(w)        # the whole row: next to the label column a long text widened the tab
                else:
                    f.addRow(o.label, w)
                continue
            # options sharing a row title sit side by side under it, each with its short label
            if o.row not in rows:
                line = flush(QHBoxLayout())
                rows[o.row] = line
                title = QLabel(o.row)
                f.addRow(title, line)
            line = rows[o.row]
            if o.kind != "bool":
                lab = QLabel(o.label)
                tip_auto(lab, o.help)
                line.addWidget(lab)
            if o.kind == "text":
                w.setPlaceholderText("")
                w.setMaximumWidth(90)
            line.addWidget(w, 1 if o.kind == "text" else 0)
        for line in rows.values():
            line.addStretch(1)
        return page


import app as _app  # noqa: E402
