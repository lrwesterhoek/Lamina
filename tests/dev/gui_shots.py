#!/usr/bin/env python3
"""Screenshots of the launcher, to SEE layout bugs the suites cannot (clipped fields, overlapping rows, a short table).

Renders every tab at each window size, then the Explore tab with "Show in 3D" on (volume, then planes), and prints for
each scrolled tab the width its form needs against the width it is given (a form wider than its viewport is clipped:
the scroll bars are off), folded and with every folded group (More) open. The window never remembers
anything (remember=False: no saved settings are read or written) and the figures go to a temporary folder.

    tests/dev/gui_shots.py <out dir> [--native] [--open] [--fast-dtfe] [--advanced] [--data-root DIR]
                           [--sizes 1500x950,1100x750]

  --native     draw in a real window on the default platform (macOS: the system style, the widths a user sees); the
               default is Qt's offscreen platform, whose generic style sizes some widgets differently
  --open       take the tabs' shots with every folded group open (the widths are measured both ways regardless)
  --fast-dtfe  the run tabs at Fast, the Grids and Custom snapshot tabs on standard DTFE (their PS-only groups hidden)
  --advanced   View > Advanced Settings on: the run tabs' Advanced groups (folded, unless --open), every row on
               Automatic (the window remembers nothing: the switch is not saved)
  --data-root  a simulation root to show (default: a synthetic one made in a temporary folder: one 64^3 PS-DTFE grid
               set, so Explore has a volume to draw)

Needs PySide6 (the launcher's python). The PNGs are named <tab index>_<tab>_<width>.png (the window),
<tab index>_<tab>_<width>_form.png (a tab's whole form, below the fold too; Explore: its left pane) and
explore3d_<style>_<width>.png.
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = "GUISHOTS-1-Dark"
N, BOX = 64, 64.0


def synthetic_root(base: Path) -> Path:
    """base/SIM/snapdir_050: a combined header and an N^3 PS-DTFE grid set ('ps_output': density with six broad dips
    on 2% noise, velocity, streams), as tests/py_gui_explore_test.py makes."""
    import h5py
    import numpy as np
    d = base / SIM / "snapdir_050"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "combined_050.hdf5", "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"], h["Redshift"], h["HubbleParam"], h["HFreeUnits"] = BOX * 1000.0, 0.0, 0.6774, 1
        h["NumPart_ThisFile"], h["MassTable"] = np.array([0, N ** 3, 0, 0, 0, 0]), np.array([0, 1.0, 0, 0, 0, 0])
        h["NumPart_Total"] = np.array([0, N ** 3, 0, 0, 0, 0])
    rho = 1.0 + 0.02 * np.random.default_rng(7).normal(size=(N, N, N))
    ii = np.indices((N, N, N))
    for c, w, depth in (((16, 16, 32), 6.0, 0.9), ((46, 20, 30), 7.0, 0.85), ((20, 48, 34), 5.0, 0.9),
                        ((50, 50, 28), 8.0, 0.8), ((8, 36, 54), 6.0, 0.85), ((36, 6, 10), 5.0, 0.8)):
        r2 = sum((((ii[k] - c[k] + N // 2) % N) - N // 2) ** 2 for k in range(3))
        rho -= depth * np.exp(-0.5 * r2 / w ** 2)
    rho.astype(np.float32).tofile(str(d / "ps_output.a_den"))
    np.zeros((N, N, N, 3), np.float32).tofile(str(d / "ps_output.a_vel"))
    np.ones((N, N, N), np.float32).tofile(str(d / "ps_output.a_streams"))
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--fast-dtfe", action="store_true")
    ap.add_argument("--advanced", action="store_true")
    ap.add_argument("--data-root")
    ap.add_argument("--sizes", default="1500x950,1100x750")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]

    tmp = Path(tempfile.mkdtemp(prefix="gui_shots_"))
    if not a.native:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["DTFE_GUI_NO_NOTIFY"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["DTFE_FIGURES_ROOT"] = str(tmp / "figures")      # BEFORE runspec is imported
    sys.path.insert(0, str(ROOT / "python" / "gui"))
    sys.path.insert(0, str(ROOT / "python"))
    from PySide6.QtWidgets import QApplication, QScrollArea
    import app as A
    import runspec as rs
    from app_widgets import Collapsible

    data = Path(a.data_root) if a.data_root else synthetic_root(tmp / "data")
    qa = QApplication.instance() or QApplication([])
    w = A.MainWindow(rs.RunSpec(data_root=str(data)), remember=False)
    if a.fast_dtfe:
        w.dtfe_radio.setChecked(True)
        w.o_dtfe.setChecked(True)
        for pair in (w.quality_pair, w.o_quality, w.p_quality):
            pair.button("fast").setChecked(True)
    if a.advanced:
        w.advanced_action.setChecked(True)          # as a user switches it on (the groups open) ...
        for g in w.adv_boxes.values():
            g.set_open(False)                       # ... folded like the other groups (--open opens them all)

    def settle(seconds: float = 0.3):
        end = time.time() + seconds
        while time.time() < end:
            qa.processEvents()
            time.sleep(0.01)

    def wait_for(cond, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            qa.processEvents()
            if cond():
                return True
            time.sleep(0.02)
        return False

    def open_groups(groups, on):
        for g in groups:
            g.set_open(on)
        settle(0.2)

    print(f"platform {qa.platformName()}, data root {data}")
    for width, height in sizes:
        w.resize(width, height)
        w.show()
        settle()
        for i in range(w.tabs.count()):
            w.tabs.setCurrentIndex(i)
            settle(0.2)
            s = w.tabs.widget(i)
            groups = s.findChildren(Collapsible)
            if a.open:
                open_groups(groups, True)
            name = w.tabs.tabText(i).replace(" ", "_")
            w.grab().save(str(out / f"{i}_{name}_{width}.png"))
            if isinstance(s, QScrollArea) and s.widget() is not None:
                # ... and the tab's whole form, below the fold too (the window shows only its top; Explore: its left
                # pane down to the Zoom group)
                s.widget().grab().save(str(out / f"{i}_{name}_{width}_form.png"))
            if isinstance(s, QScrollArea) and s.widget() is not None:
                fits = []                   # (needs, given): folded, then with its groups open
                for on in ((False, True) if groups else (False,)):
                    open_groups(groups, on)
                    fits.append((s.widget().minimumSizeHint().width(), s.viewport().width()))
                open_groups(groups, a.open)
                print(f"  {width:>5}  {w.tabs.tabText(i):<16} form needs {fits[0][0]:>4} px, given {fits[0][1]:>4}"
                      + (f"; with its groups open {fits[1][0]:>4} px, given {fits[1][1]:>4}" if groups else "")
                      + ("   CLIPPED" if any(need > have for need, have in fits) else ""))
            open_groups(groups, False)

    # Explore with Show in 3D on: the volume, then the planes (at the last size)
    x = w.explore
    w.tabs.setCurrentIndex(A.TAB_NAMES.index("explore"))
    x.refresh_outputs()
    if not wait_for(lambda: not x._busy and x._plane is not None, 60):
        print("  Explore: no map (no grid set under the data root?): 3D shots skipped")
        return 0
    for style in ("volume", "planes"):
        if x.view3d_toggle.isChecked():
            x.view3d_toggle.setChecked(False)
            settle(0.2)
        x.view3d_style.setCurrentIndex(x.view3d_style.findData(style))
        x.view3d_toggle.setChecked(True)
        ok = wait_for(lambda: (x._3d_res or {}).get(style) is not None and not x._3d_busy, 120)
        settle(0.5)
        w.grab().save(str(out / f"explore3d_{style}_{sizes[-1][0]}.png"))
        print(f"  Explore 3D {style}: {'drawn' if ok else 'NOT READY after 120 s'}")
    x.view3d_toggle.setChecked(False)
    w.close()
    print(f"PNGs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
