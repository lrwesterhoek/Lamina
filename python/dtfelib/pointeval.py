"""Field maps from POINT EVALUATION of the tessellation ('--sample-points'): the field evaluated at each pixel
centre, a zero-width cross-section limited only by the tessellation, not the mass-conserving grid deposit of
plot_PS_DTFE.py. An interpolant, so not for mass-weighted statistics. Shared by plot_PS_DTFE and plot_pointeval."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .io import PointPlane

#: Styling matches the grid slice maps of plot/plot_PS_DTFE.py and plot/plot_DTFE.py (colormaps, colorbar
#: geometry, fonts), so a point-evaluated map and a grid map are interchangeable. Colormaps: density and shear
#: plasma, velDiv RdBu_r, streams inferno, |v|, dispersion and other velocity magnitudes magma.
#: field -> (title, colormap, norm kind, colorbar label)
STYLES = {
    "density":    ("Density",                     "plasma",  "log",    r"Density $\rho/\bar{\rho}$"),
    "streams":    ("Stream Count",                "inferno", "stream", "Number of streams"),
    "speed":      ("Velocity Magnitude",          "magma",   "robust", r"$|v|$ [km/s]"),
    "dispTrace":  ("Velocity Dispersion",         "magma",   "poslog", r"$\sigma_{\rm 3D}$ [km/s]"),
    "dispMag":    ("Dispersion Tensor Magnitude", "magma",   "poslog", r"$|\sigma^2_{ij}|$ [km$^2$/s$^2$]"),
    "velDiv":     ("Velocity Divergence",         "RdBu_r",  "signed", r"$\nabla\!\cdot\!v$ [km/s/Mpc]"),
    "velShear":   ("Velocity Shear",              "plasma",  "poslog", r"$|\Sigma_{ij}|$ [km/s/Mpc]"),
    "velVort":    ("Vorticity",                   "magma",   "poslog", r"$|\omega|$ [km/s/Mpc]"),
    "denGradMag": ("Density Gradient",            "plasma",  "poslog", r"$|\nabla\rho/\bar{\rho}|$ [Mpc$^{-1}$]"),
}

#: projection axis -> the slice-plane name plot_PS_DTFE.py uses for its grid maps
PLANE_NAME = {"x": "yz_plane", "y": "xz_plane", "z": "xy_plane"}

#: Fields whose LOW END IS A TRUE NULL (single-stream, zero dispersion), where a black floor is meaningful. Every
#: other field is positive everywhere, so those get the colormap's black floor trimmed away (CMAP_LIFT) and start
#: at a visible tone.
NULL_AT_BOTTOM = {"streams", "dispTrace", "dispMag"}
CMAP_LIFT = 0.18          # fraction of the dark end to drop, for near-black colormaps only

#: Gradient-derived fields. In DTFE the velocity is piecewise LINEAR over the tessellation, so its gradient is
#: piecewise CONSTANT and shows as flat triangular facets at full resolution. These fields get a SECOND,
#: Gaussian-smoothed rendering ('..._smooth<sigma>.png') alongside the raw one; the other fields are not faceted.
DERIVATIVE_FIELDS = {"velDiv", "velShear", "velVort", "denGradMag"}
DERIVATIVE_SMOOTH = 10.0   # default Gaussian sigma in OUTPUT PIXELS for the smoothed variant

AXIS_UNITS = "Mpc"
FIGSIZE = (8, 7)          # plot_PS_DTFE.plot_density's figure geometry
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
MIN_DPI = 300             # plot_PS_DTFE.DPI; raised per figure to reach native resolution

# The grid maps' fixed density range (plot_PS_DTFE.DENSITY_VMIN/DENSITY_VMAX, shared with plot_DTFE.py), available
# via fixed_range=True but NOT the default: it clips the whole void population against vmin. The default is a
# percentile stretch, which also adapts across redshift.
DENSITY_VMIN, DENSITY_VMAX = 1e-1, 1e4


def starts_near_black(cmap, thresh=0.15):
    """Does this colormap's low end bottom out in black? magma/inferno do; plasma, viridis
    and cividis start at a visible blue/purple and need no trimming."""
    r, g, b, _ = cmap(0.0)
    return max(r, g, b) < thresh


def lift_cmap(cmap, lo):
    """The colormap with its darkest `lo` fraction removed, so nothing renders as black.
    A no-op unless the map actually starts near black -- trimming plasma/viridis would
    move them off the grid maps' look for no benefit."""
    if lo <= 0. or not starts_near_black(cmap):
        return cmap
    return colors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_lift{lo:g}", cmap(np.linspace(lo, 1.0, 256)))


def norm_and_cmap(kind, cmap_name, data, fixed_range=False, lift=0.0):
    """Colour normalization + colormap, following the grid maps' conventions. Under-range and masked values take
    the colormap's lowest colour (cmap.set_under / set_bad), as plot_PS_DTFE does."""
    cmap = lift_cmap(plt.get_cmap(cmap_name), lift).copy()
    cmap.set_bad(cmap(0))
    cmap.set_under(cmap(0))
    if kind == "log":
        if fixed_range:
            lo, hi = DENSITY_VMIN, DENSITY_VMAX
        else:
            lo = max(float(np.percentile(data, 0.1)), 1e-4)
            hi = max(float(np.percentile(data, 99.99)), lo * 10)
        return cmap, colors.LogNorm(vmin=lo, vmax=hi)
    if kind == "stream":
        top = max(float(np.percentile(data, 99.9)), 2.0)
        return cmap, colors.LogNorm(vmin=0.9, vmax=top)
    if kind == "signed":
        # divergence spans ~6 decades: a linear symmetric scale renders the web white, so use the signed symlog
        # convention (dtfelib.figures.norm_signed_log).
        from . import figures as style
        return cmap, style.norm_signed_log(data=data)
    if kind == "poslog":
        # positive quantities spanning many decades: log between robust percentiles of the POSITIVE values; zeros
        # (single-stream cells) fall under vmin and take cmap(0), the background colour.
        pos = data[data > 0]
        if pos.size == 0:
            return cmap, colors.Normalize(vmin=0.0, vmax=1.0)
        hi = float(np.percentile(pos, 99.99))
        lo = float(np.percentile(pos, 1.0))
        if not (hi > 0):
            hi = float(pos.max()) or 1.0
        lo = max(min(lo, hi / 10.0), hi / 1e5)     # cap the span at 5 decades
        return cmap, colors.LogNorm(vmin=lo, vmax=hi)
    vmax = float(np.percentile(data, 99.9)) or 1.0            # "robust" positive linear
    return cmap, colors.Normalize(vmin=0.0, vmax=vmax)


def find_point_plane(fs, project="plane", plane=None):
    """PointPlane for this snapshot's '.pts_*' files, or None when there are none. plane: an explicit sidecar path
    pins the geometry (REQUIRED when several planes share a pixel count); else it is auto-discovered by geometry.
    The redshift comes from the FieldSet, NOT the sidecar (it records whichever snapshot defined the geometry)."""
    prefix = fs.snapdir / fs.prefix.rstrip(".")
    if not Path(f"{prefix}.pts_den").is_file():
        return None
    if plane is not None:
        return PointPlane(prefix, plane, redshift=fs.meta.redshift, project=project)
    sidecars = sorted(fs.snapdir.parent.glob("pointeval_plane_*.json")) \
        + sorted(fs.snapdir.parent.glob("hires_plane_*.json"))   # legacy name, still on disk
    for side in sidecars:
        try:
            return PointPlane(prefix, side, redshift=fs.meta.redshift, project=project)
        except ValueError:
            continue        # a different plane geometry (other NU/AXIS run): try the next
    print(f"  point eval: {prefix}.pts_den exists but no matching *_plane_*.json in "
          f"{fs.snapdir.parent} -- skipping the point-evaluated maps")
    return None


def render_fields(fs, out_dir, fields=None, label=None, force=True, quiet=False,
                  fixed_range=False, project="plane", smooth=0.0, plane=None,
                  smooth_derivatives=DERIVATIVE_SMOOTH):
    """Render one figure per available point-evaluated field into out_dir, styled like the grid slice maps. fields
    restricts the names; force=False skips a PNG newer than its '.pts_*' source; fixed_range: the grid maps' density
    range; project 'plane' or 'slab'; smooth: Gaussian sigma in OUTPUT PIXELS. Returns (written_paths, n_up_to_date)."""
    pp = find_point_plane(fs, project=project, plane=plane)
    if pp is None:
        return [], -1
    label = label or ("PS-DTFE" if fs.method == "ps" else "DTFE")
    out_dir = Path(out_dir)
    side, z = pp.side, fs.meta.redshift
    if not quiet:
        print(f"  point eval: {pp!r}")

    plane_name = PLANE_NAME.get(str(side.get("axis", "z")).lower(), "xy_plane")
    out_dir = out_dir / plane_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from scipy.ndimage import gaussian_filter

    def _render(name, data, save_path, sigma):
        title, cmap_name, kind, cb_label = STYLES[name]
        # a diverging map is symmetric about white and has no black floor to trim
        lift = 0.0 if (name in NULL_AT_BOTTOM or kind == "signed") else CMAP_LIFT
        cmap, norm = norm_and_cmap(kind, cmap_name, data, fixed_range=fixed_range, lift=lift)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.imshow(data, origin="lower", extent=pp.extent, cmap=cmap, norm=norm,
                       interpolation="nearest")
        full = f"{label} {title}"
        if z is not None:
            full += f" (z={z:.2f}" + (f", smoothed {sigma:g}px)" if sigma > 0 else ")")
        elif sigma > 0:
            full += f" (smoothed {sigma:g}px)"
        ax.set_title(full, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(f"{side['u_axis'].upper()} [{AXIS_UNITS}]", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f"{side['v_axis'].upper()} [{AXIS_UNITS}]", fontsize=LABEL_FONTSIZE)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cb_label, fontsize=LABEL_FONTSIZE)
        ax.set_aspect("equal")
        fig.tight_layout()
        # native resolution: dpi so the DATA AREA gets >= one output pixel per sample pixel
        # (the axes is only part of the 8-inch figure, hence measuring it after layout)
        ax_w_in = ax.get_position().width * fig.get_size_inches()[0]
        dpi = max(MIN_DPI, int(np.ceil(pp.nu / max(ax_w_in, 1e-6))))
        from .figures import save_figure        # here: figures imports config, which 'import dtfelib' must not
        save_figure(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    written, up_to_date = [], 0
    names = [n for n in STYLES if n in pp.available()]
    if fields:
        names = [n for n in names if n in set(fields)]
    for name in names:
        src = Path(f"{pp.prefix}{PointPlane._PTS_FIELDS.get(name, ('.pts_den',))[0]}")
        base = f"{fs.method}_{name}_pointeval_{plane_name}_z{z:.2f}"
        # a non-default rendering gets its own name: under the plane render's, --project
        # slab / --smooth-px / --fixed-range would overwrite it -- or be skipped as 'up to date' against it
        base += ("_slab" if project == "slab" else "") + (f"_smooth{smooth:g}" if smooth > 0 else "") \
            + ("_fixed" if fixed_range else "")
        # the raw (primary) output, plus -- for the faceted gradient fields -- a smoothed
        # companion. A user-supplied global --smooth-px applies to the primary and suppresses
        # the auto companion (they asked for one specific smoothing).
        variants = [(out_dir / f"{base}.png", smooth)]
        if name in DERIVATIVE_FIELDS and smooth_derivatives > 0 and smooth == 0:
            variants.append((out_dir / f"{base}_smooth{smooth_derivatives:g}.png",
                             smooth_derivatives))
        todo = []
        for path, sigma in variants:
            fresh = (not force and path.is_file() and src.is_file()
                     and path.stat().st_mtime > src.stat().st_mtime)
            if fresh:
                up_to_date += 1
                if not quiet:
                    print(f"    skip {path.name} (up to date)")
            else:
                todo.append((path, sigma))
        if not todo:
            continue
        data = pp.field(name)                       # read + derive once, reuse for both
        for path, sigma in todo:
            d = gaussian_filter(data, sigma=sigma, mode="wrap") if sigma > 0 else data
            _render(name, d, path, sigma)
            written.append(path)
            if not quiet:
                print(f"    wrote {path}  ({pp.nu} x {pp.nv} samples)")
    if not names:
        return [], -1                       # plane exists but holds no renderable field
    return written, up_to_date
