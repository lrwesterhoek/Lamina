"""The launcher's own small widgets and widget helpers (moved from app.py, which re-exports CheckDelegate and
FigureBrowser): CheckDelegate, which draws the tick boxes of a checkable list or table; FigureBrowser, the
Figures tab's list of the figures a run wrote, with a preview; the helpers the window's forms are built
and read with (_mono, _note, _checkable, _grow_fields, _scroll, _dir_row, _exclusive, _checked, _grid_combo,
_state_text, _int); and the style helpers every tab builds its groups with: the spacing constants, group,
Collapsible / advanced_group, labelled_row, field_grid, grid_row, button_row and tip."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QImageReader, QIntValidator, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QAbstractSpinBox, QSizePolicy, QSpinBox, QSplitter, QStyle,
                               QStyledItemDelegate, QStyleFactory, QStyleOptionButton, QStyleOptionViewItem,
                               QToolButton, QVBoxLayout, QWidget)

import runspec as rs
import presets as P

WIDE_COMBO_PX = 300     # _grow_fields: a combo box wider than this may shrink to 24 characters

# ---- spacing: the few margins and gaps the forms set themselves (everything else is the style's own)
GAP = 2                 # between the small tool buttons of one strip (the log's find strip)
GROUP_MARGIN = 4        # a page inside its group (a figure set's options)
INDENT = 24             # the options under the box that switches them on
SECTION_GAP = 12        # the space above an extra section of a dialog page
TIP_WIDTH = 90          # a tooltip's line length (the binary's help is wrapped to it too)
OPEN_KEY = "open_sections"     # the settings key of the collapsible sections' state: {section: open}


def flush(layout: QLayout) -> QLayout:
    """A layout nested in a row, a pane or a group of its own: no margins of its own."""
    layout.setContentsMargins(0, 0, 0, 0)
    return layout


def _add_rows(form: QFormLayout, rows):
    """Each row: (label, widget or layout) or a widget / layout alone (the whole row)."""
    for row in rows:
        if isinstance(row, tuple):
            form.addRow(*row)
        else:
            form.addRow(row)


def group(title: str, *rows) -> QGroupBox:
    """A titled group with a form; 'rows' as _add_rows takes them. The form is the group's 'form'."""
    g = QGroupBox(title)
    g.form = QFormLayout(g)
    _add_rows(g.form, rows)
    return g


def labelled_row(label: str, widget: QWidget, unit: str | None = None, tip_text: str | None = None) -> tuple:
    """(label, widget) for a form row: the unit as the spin box's suffix, the tooltip set."""
    if unit is not None and isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setSuffix(unit)
    if tip_text:
        tip(widget, tip_text)
    return label, widget


def field_grid(names_labels, columns: int = 3, changed=None) -> tuple[QGridLayout, dict[str, QCheckBox]]:
    """A grid of tick boxes, one per (name, label), filled row by row in equal columns; {name: box}."""
    grid = QGridLayout()
    checks: dict[str, QCheckBox] = {}
    for i, (name, label) in enumerate(names_labels):
        cb = QCheckBox(label)
        if changed is not None:
            cb.toggled.connect(changed)
        checks[name] = cb
        grid.addWidget(cb, i // columns, i % columns)
    for c in range(columns):            # equal columns: the last one must not be starved of width
        grid.setColumnStretch(c, 1)
    return grid, checks


def grid_row(parent, changed, cells: str = "³ cells    sub-samples per axis") -> tuple:
    """The 'Grid' row of a run tab: the grid's cells per axis, then the sub-samples per axis of the sampled
    deposit. Returns (row layout, grid combo, the label between them, the sub-sample spin box)."""
    row = QHBoxLayout()
    combo = _grid_combo(parent)
    combo.currentTextChanged.connect(changed)
    label = QLabel(cells)
    nsub = QSpinBox(minimum=1, maximum=6)
    nsub.setToolTip("nSub³ sample points per cell for the sampled deposit")
    nsub.valueChanged.connect(changed)
    row.addWidget(combo)
    row.addWidget(label)
    row.addWidget(nsub)
    row.addStretch(1)
    return row, combo, label, nsub


def button_row(*buttons, stretch: bool = True, field: QWidget | None = None) -> QHBoxLayout:
    """Buttons side by side (then an optional field taking the rest of the row, or a stretch)."""
    row = QHBoxLayout()
    for b in buttons:
        row.addWidget(b)
    if field is not None:
        row.addWidget(field, 1)
    elif stretch:
        row.addStretch(1)
    return row


def tip(widget: QWidget, short: str, more: str = "") -> QWidget:
    """A tooltip of at most two lines; the long explanation, when there is one, is the widget's What's This text
    (Help > What's This?, then a click on the widget), which repeats the short one first."""
    widget.setToolTip(short)
    if more:
        widget.setWhatsThis(short + "\n\n" + more)
    return widget


def option_tip(widget: QWidget, key: str, binary: str = "PS-DTFE", more: str = "") -> QWidget:
    """A control for a binary option: its tooltip the first sentence of the binary's own description (at most two
    lines), its What's This the whole description (then 'more', the launcher's own words)."""
    widget.setToolTip(P.short_tooltip(key, binary))
    widget.setWhatsThis(P.tooltip(key, binary) + more)
    return widget


def tip_auto(widget: QWidget, text: str) -> QWidget:
    """A tooltip from a text of any length: the text itself when it takes at most two lines, else its first sentence
    (P.shorten), the whole text then the widget's What's This."""
    if tip_lines(text) <= 2:
        widget.setToolTip(text)
    else:
        widget.setToolTip(P.shorten(text, TIP_WIDTH))
        widget.setWhatsThis(text)
    return widget


def tip_lines(text: str) -> int:
    """How many lines a tooltip takes: its own lines, each wrapped at TIP_WIDTH (Qt shows plain text unwrapped,
    so the long ones are written with their own line breaks)."""
    return sum(max(1, len(textwrap.wrap(line, TIP_WIDTH))) for line in text.split("\n")) if text else 0


class ElidedLabel(QLabel):
    """A one-line label that never widens its form: it takes the width it is given and cuts its text there with
    '…' (the whole text is its tooltip). A click on it is its 'clicked' signal."""
    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__()
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setStyleSheet("color: gray")
        self.set_full_text(text)

    def set_full_text(self, text: str):
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.clicked.emit()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(self.width(), 1)))


class Collapsible(QGroupBox):
    """A group that folds to one line: its title, an arrow and a summary of what it holds (the summary function's
    text, kept up to date by update_summary). The body's form is 'form', the body itself 'body': folding hides the
    body, never the fields in it (their own isHidden() stays false). With settings, whether it is open is kept
    under OPEN_KEY ({key: open}) and comes back at the next start; without (a window that does not remember),
    nothing is read or written. Opening or folding it is its 'opened' signal."""
    opened = Signal(bool)            # (not 'toggled': QGroupBox has a signal of that name)

    def __init__(self, title: str, key: str, settings=None, summary=None, start_open: bool = False):
        super().__init__(title)
        self.key, self._settings, self._summary = key, settings, summary
        self._summary_text = ""
        self.arrow = QToolButton()
        self.arrow.setAutoRaise(True)
        self.arrow.setCheckable(True)
        self.arrow.setToolTip("Show or hide these options")
        self.summary = ElidedLabel()
        self.summary.clicked.connect(self.arrow.toggle)          # the summary line opens it too
        head = flush(QHBoxLayout())
        head.addWidget(self.arrow)
        head.addWidget(self.summary, 1)
        self.body = QWidget()
        self.form = flush(QFormLayout(self.body))
        v = QVBoxLayout(self)
        v.addLayout(head)
        v.addWidget(self.body)
        self.arrow.setChecked(self._remembered(start_open))
        self._show(self.arrow.isChecked())
        self.arrow.toggled.connect(self._toggled)

    def _remembered(self, default: bool) -> bool:
        if self._settings is None:
            return default
        try:
            return bool(json.loads(self._settings.value(OPEN_KEY, "") or "{}").get(self.key, default))
        except (ValueError, TypeError, AttributeError):
            return default

    def _show(self, on: bool):
        self.body.setVisible(on)
        self.arrow.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.summary.set_full_text("" if on else self._summary_text)      # (open: the fields say it)

    def _toggled(self, on: bool):
        self._show(on)
        if self._settings is not None:
            try:
                state = json.loads(self._settings.value(OPEN_KEY, "") or "{}")
                state = state if isinstance(state, dict) else {}
            except (ValueError, TypeError):
                state = {}
            state[self.key] = on
            self._settings.setValue(OPEN_KEY, json.dumps(state))
        self.opened.emit(on)

    def is_open(self) -> bool:
        return self.arrow.isChecked()

    def set_open(self, on: bool):
        self.arrow.setChecked(on)

    def update_summary(self):
        if self._summary is not None:
            self._summary_text = self._summary()
            if not self.is_open():
                self.summary.set_full_text(self._summary_text)

    def summary_text(self) -> str:
        """The one-line summary (also while open, when it is not shown)."""
        return self._summary_text


def advanced_group(title: str, key: str, settings=None, summary=None, *rows) -> Collapsible:
    """A folded group of expert options ('rows' as group() takes them); its state remembered (Collapsible)."""
    g = Collapsible(title, key, settings, summary)
    _add_rows(g.form, rows)
    return g


def _mono() -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    f.setPointSize(11)
    return f


def _note(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: gray")
    return lab


class CheckDelegate(QStyledItemDelegate):
    """Draws the tick boxes of a checkable list or table itself. The native macOS style (Qt 6, dark
    appearance) paints the check indicator of the CURRENT row only, so every other row looks as if
    it has none. Fusion's indicator, drawn over the rectangle the view's own style reserves for it
    (so a click there still toggles the item), shows in both appearances."""
    _fusion = None

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        state = index.data(Qt.CheckStateRole)
        if state is None or not (index.flags() & Qt.ItemIsUserCheckable):
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        area = style.subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, opt, widget)
        side = min(14, area.width(), area.height())
        if side <= 0:
            return
        if CheckDelegate._fusion is None:
            CheckDelegate._fusion = QStyleFactory.create("Fusion")
        box = QStyleOptionButton()
        box.rect = QRect(area.x() + (area.width() - side) // 2, area.y() + (area.height() - side) // 2, side, side)
        box.palette = option.palette
        box.state = QStyle.StateFlag.State_On if Qt.CheckState(state) == Qt.Checked else QStyle.StateFlag.State_Off
        if option.state & QStyle.StateFlag.State_Enabled:
            box.state |= QStyle.StateFlag.State_Enabled
        painter.save()
        # clear the whole indicator column -- the item's left edge up to its text -- first: the native
        # indicator, where it is drawn, reaches a little outside the rectangle the style reports
        text = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, widget)
        strip = QRect(option.rect.left(), option.rect.top(), max(text.left() - option.rect.left(), area.width()),
                      option.rect.height())
        painter.fillRect(strip, option.palette.highlight() if option.state & QStyle.StateFlag.State_Selected
                         else option.palette.base())
        CheckDelegate._fusion.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorCheckBox, box, painter, None)
        painter.restore()


def _checkable(view):
    """A list/table whose items carry tick boxes: our own indicator (see CheckDelegate)."""
    view.setItemDelegate(CheckDelegate(view))
    return view


def _grow_fields(root: QWidget):
    """Every form's fields take the width they are given. macOS's default form policy keeps each
    field at its size hint, which squeezed paths and names into a few characters."""
    for f in root.findChildren(QFormLayout):
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    for sb in root.findChildren(QAbstractSpinBox):      # numbers stay compact
        sb.setSizePolicy(QSizePolicy.Policy.Fixed, sb.sizePolicy().verticalPolicy())
    # A combo box is as wide as its longest item, and the forms scroll only vertically: a long item (a deposit's
    # explanation) made the whole tab wider than its pane and cut off the right edge. Such a combo may shrink to
    # 24 characters; it grows back to the width the form gives it, and its list shows the items whole.
    for cb in root.findChildren(QComboBox):
        if cb.sizeAdjustPolicy() == QComboBox.AdjustToContentsOnFirstShow and cb.sizeHint().width() > WIDE_COMBO_PX:
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(24)


def _scroll(inner: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidget(inner)
    s.setWidgetResizable(True)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return s


def _dir_row(edit: QLineEdit, on_browse) -> QHBoxLayout:
    row = QHBoxLayout()
    b = QToolButton(text="…")
    b.clicked.connect(on_browse)
    row.addWidget(edit)
    row.addWidget(b)
    return row


def _exclusive(a: QCheckBox, b: QCheckBox):
    """Two options the binary refuses together: ticking one unticks the other."""
    a.toggled.connect(lambda on: on and b.setChecked(False))
    b.toggled.connect(lambda on: on and a.setChecked(False))


def _checked(lst: QListWidget) -> list:
    return [lst.item(i).data(Qt.UserRole) for i in range(lst.count())
            if lst.item(i).checkState() == Qt.Checked]


def _grid_combo(parent) -> QComboBox:
    c = QComboBox()
    c.setEditable(True)
    c.addItems(["128", "256", "512", "1024"])
    c.setValidator(QIntValidator(8, 4096, parent))
    return c


def _state_text(chunks: int, merged: bool) -> str:
    if merged:
        return "merged" + (f" (+{chunks} chunks)" if chunks else "")
    return f"{chunks} chunks" if chunks else "—"


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return 0


# ==================================================================== figure browser
class FigureBrowser(QWidget):
    """Figures under the figures root (rs.FIGURES_ROOT): the ones the last run wrote, or all of them, with a preview."""
    changed = Signal()          # the selection (what Open / Show in Finder act on): the menu bar follows

    def __init__(self):
        super().__init__()
        QImageReader.setAllocationLimit(4096)       # 8192^2 point-eval PNGs exceed Qt's 256 MB default
        self.roots: list[Path] = [rs.FIGURES_ROOT]
        self.since: float | None = None
        self._current: Path | None = None

        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["From the last run", "All figures"])
        self.mode.currentIndexChanged.connect(self.reload)
        self.filter = QLineEdit(placeholderText="filter: words in the path, e.g. TNG100 snap099 density")
        self.filter.textChanged.connect(self.reload)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.reload)
        top.addWidget(self.mode)
        top.addWidget(self.filter, 1)
        top.addWidget(self.refresh_btn)

        self.list = QListWidget()
        self.list.setTextElideMode(Qt.ElideMiddle)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.currentItemChanged.connect(lambda cur, _prev: self._show(cur))
        self.list.itemDoubleClicked.connect(lambda _it: self.open_current())
        self.preview = QLabel("", alignment=Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.setMinimumSize(200, 200)
        self.path_label = _note()
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        btns = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_current)
        self.reveal_btn = QPushButton("Show in Finder")
        self.reveal_btn.clicked.connect(self.reveal_current)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.reveal_btn)
        btns.addStretch(1)
        right = QWidget()
        rv = flush(QVBoxLayout(right))
        rv.addWidget(self.preview, 1)
        rv.addWidget(self.path_label)
        rv.addLayout(btns)
        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(split, 1)
        self._show(None)

    def show_run(self, since: float, extra_roots=()) -> int:
        """List what a run started at `since` wrote; returns how many figures that is."""
        self.since = since
        self.roots = [rs.FIGURES_ROOT] + [Path(r) for r in extra_roots if Path(r) != rs.FIGURES_ROOT]
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        return self.reload()

    def reload(self, *_) -> int:
        from_run = self.mode.currentIndex() == 0
        if from_run and self.since is None:
            files = []
        else:
            files = rs.figures(self.roots, self.since if from_run else None)
        words = self.filter.text().lower().split()
        if words:
            files = [p for p in files if all(w in str(p).lower() for w in words)]
        keep = str(self._current) if self._current else None
        self.list.blockSignals(True)
        self.list.clear()
        for p in files:
            rel = next((str(p.relative_to(r.resolve())) for r in self.roots
                        if p.is_relative_to(r.resolve())), str(p))
            it = QListWidgetItem(rel)
            it.setData(Qt.UserRole, str(p))
            it.setToolTip(f"{p}\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))}")
            self.list.addItem(it)
        self.list.blockSignals(False)
        row = next((i for i in range(self.list.count()) if self.list.item(i).data(Qt.UserRole) == keep), 0)
        if self.list.count():
            self.list.setCurrentRow(row)
            self._show(self.list.currentItem())
        else:
            self._show(None)
        return len(files)

    def _show(self, item):
        self._current = Path(item.data(Qt.UserRole)) if item is not None else None
        for b in (self.open_btn, self.reveal_btn):
            b.setEnabled(self._current is not None)
        self.changed.emit()
        if self._current is None:
            from_run = self.mode.currentIndex() == 0
            self.preview.setText("No figures from the last run yet." if from_run else "No figures found.")
            self.preview.setPixmap(QPixmap())
            self.path_label.setText("")
            return
        self.path_label.setText(str(self._current))
        self._render_preview()

    def _render_preview(self):
        p = self._current
        if p is None:
            return
        reader = QImageReader(str(p))
        size = reader.size()
        box = self.preview.size()
        if not reader.canRead() or not size.isValid():
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"{p.suffix.upper()[1:]}: no preview (Open shows it)")
            return
        reader.setScaledSize(size.scaled(QSize(max(box.width(), 50), max(box.height(), 50)),
                                         Qt.KeepAspectRatio))
        img = reader.read()
        self.preview.setPixmap(QPixmap.fromImage(img) if not img.isNull() else QPixmap())
        if img.isNull():
            self.preview.setText(f"could not read: {reader.errorString()}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self._render_preview)

    def open_current(self):
        if self._current is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current)))

    def reveal_current(self):
        if self._current is not None:
            if sys.platform == "darwin":
                subprocess.run(["open", "-R", str(self._current)])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current.parent)))
