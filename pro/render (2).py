"""pro.render — matplotlib renderers for maps, Hovmollers and composites."""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from . import config

def _chi_cmap():
    cdict = {
        "red":   [(0.0, 0.08, 0.08), (0.35, 0.40, 0.40),
                  (0.50, 0.97, 0.97), (0.65, 0.92, 0.92), (1.0, 0.55, 0.55)],
        "green": [(0.0, 0.38, 0.38), (0.35, 0.72, 0.72),
                  (0.50, 0.97, 0.97), (0.65, 0.78, 0.78), (1.0, 0.30, 0.30)],
        "blue":  [(0.0, 0.45, 0.45), (0.35, 0.78, 0.78),
                  (0.50, 0.97, 0.97), (0.65, 0.52, 0.52), (1.0, 0.10, 0.10)],
    }
    return LinearSegmentedColormap("chi_cmap", cdict, N=512)

def _chi_cmap_inv():
    """Inverse of the diverging map — positive shades cool/blue, negative warm.
    Used for MSE where the user wants the (usually negative) anomaly to be the
    warm/brown side, matching the inverted colour convention."""
    return _chi_cmap().reversed()

def _pos_cmap():
    """White/pale -> teal -> deep green, for strictly non-negative fields.
    Used e.g. IVT (atmospheric-river moisture) where green reads as "moist".
    """
    cdict = {
        "red":   [(0.0, 0.97, 0.97), (0.40, 0.72, 0.72), (0.70, 0.30, 0.30),
                  (1.0, 0.02, 0.02)],
        "green": [(0.0, 0.97, 0.97), (0.40, 0.90, 0.90), (0.70, 0.68, 0.68),
                  (1.0, 0.40, 0.40)],
        "blue":  [(0.0, 0.97, 0.97), (0.40, 0.82, 0.82), (0.70, 0.45, 0.45),
                  (1.0, 0.20, 0.20)],
    }
    return LinearSegmentedColormap("pos_cmap", cdict, N=512)

def _source_cmap():
    """Diverging map for the Rossby-wave source: green = χ′<0 (upper divergence,
    enhanced convection = wave source), brown = χ′>0 (convergence / suppressed)."""
    cdict = {
        "red":   [(0.0, 0.03, 0.03), (0.50, 0.97, 0.97), (1.0, 0.60, 0.60)],
        "green": [(0.0, 0.45, 0.45), (0.50, 0.97, 0.97), (1.0, 0.33, 0.33)],
        "blue":  [(0.0, 0.22, 0.22), (0.50, 0.97, 0.97), (1.0, 0.12, 0.12)],
    }
    return LinearSegmentedColormap("source_cmap", cdict, N=512)

def _xlabel(v):
    if v in (0, 360):
        return "0°"
    if v == 180:
        return "180°"
    if v <= 180:
        return f"{v}°E"
    return f"{360 - v}°W"

def _ylabel(v):
    return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

def _domain_xticks(lon_min, lon_max):
    """Ticks for a possibly-symmetric (wrappable) longitude range."""
    if lon_max < lon_min:
        lon_max += 360
    step = 30
    ticks = []
    v = int(np.floor(lon_min / step)) * step
    while v <= lon_max:
        t = v % 360
        if lon_max - lon_min >= 359:
            if t not in (0, 360):
                ticks.append(t)
        else:
            ticks.append(t)
        v += step
    if lon_max - lon_min >= 359:
        ticks = list(range(0, 360, step))
    return ticks

def _domain_yticks(lat_min, lat_max):
    step = 20
    s = int(np.floor(lat_min / step)) * step
    ticks = []
    while s <= lat_max:
        ticks.append(s)
        s += step
    return ticks

def _lat_runs(mask, lat):
    """Contiguous lat rows where mask is True -> list of (lo, hi)."""
    runs, i, n = [], 0, len(lat)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((lat[i], lat[j - 1]))
            i = j
        else:
            i += 1
    return runs

def _psi_centers(psi, lat, lon, levels):
    """Detect enclosed circulation centres from CLOSED contour loops.

    Uses the actual contour polylines of the streamfunction field (at the same
    levels the map draws, ``levels``) and keeps only loops that form closed,
    rounded paths not touching the map edge. Open ridges/troughs are ignored,
    so an H/L marker is placed only where a cell is genuinely enclosed by a
    fully closed contour. Longitudes are wrapped so a loop spanning the
    dateline stays whole.

    Returns [(lon, lat, 'H'), (lon, lat, 'L'), ...], one marker per distinct
    cell (deduped at 5 grid cells).
    """
    from skimage import measure
    p = np.nan_to_num(psi, nan=0.0)
    nlon = p.shape[1]
    # wrap longitude so a loop crossing 0/360 stays one closed path
    pw = np.concatenate([p[:, -1:], p, p[:, 0:1]], axis=1)
    markers = []
    seen = []  # accepted cell centres, for de-duplication
    dl = abs(float(np.mean(np.diff(lon))))
    da = abs(float(np.mean(np.diff(lat))))

    for lev in levels:
        sign = 1 if lev > 0 else -1
        if abs(lev) < 1e-6:
            continue
        try:
            segs = measure.find_contours(pw, lev)
        except Exception:  # noqa: BLE001
            continue
        for seg in segs:
            # seg is (N,2) as (row, col) in the wrapped array
            rows, cols = seg[:, 0], seg[:, 1]
            # is it a fully rounded / closed loop? first point ~ last point
            if rows[0] == rows[-1] and cols[0] == cols[-1]:
                pass
            elif abs(rows[0] - rows[-1]) <= 1 and abs(cols[0] - cols[-1]) <= 1:
                pass
            else:
                continue  # open path that runs off the domain
            # does the loop touch the real map edge (top/bottom/padded sides)?
            if rows.min() <= 0 or rows.max() >= pw.shape[0] - 1:
                continue
            if cols.min() <= 0 or cols.max() >= pw.shape[1] - 1:
                continue
            # unwrap column indices back to real integer longitudes & rows
            gr = rows.astype(int)
            gc = ((cols.astype(int) - 1) % nlon)
            # loop centre = centroid of the polyline (row, real-lon)
            row_c = float(np.mean(gr))
            col_c = float(np.median(gc))          # median avoids wrap issues
            lon_center = float(lon[int(round(col_c)) % nlon])
            lat_center = float(lat[min(max(int(round(row_c)), 0), len(lat) - 1)])
            # de-duplicate & skip if it overlaps an accepted cell (5 grid cells)
            if any(abs(lon_center - kx) < 5 * dl and abs(lat_center - ky) < 5 * da
                   for kx, ky, _ in seen):
                continue
            seen.append((lon_center, lat_center, "H" if sign > 0 else "L"))
            markers.append((lon_center, lat_center, "H" if sign > 0 else "L"))
    return markers

def _draw_axis(ax, lon_min, lon_max, lat_min, lat_max):
    """Shared map-frame cosmetics (gridlines, ticks, spines) for the global frame."""
    xticks = _domain_xticks(lon_min, lon_max)
    yticks = _domain_yticks(lat_min, lat_max)
    for x in xticks:
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in yticks:
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.75, zorder=0, alpha=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels([_xlabel(x) for x in xticks], fontsize=9.5,
                       color="#333322", fontfamily="DejaVu Sans")
    ax.set_yticks(yticks)
    ax.set_yticklabels([_ylabel(y) for y in yticks], fontsize=9.5,
                       color="#333322", fontfamily="DejaVu Sans")
    ax.tick_params(axis="both", length=3.5, color="#888878", width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988")
        spine.set_linewidth(0.8)

def _draw_coasts(ax, coast_segs):
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

def render(lat, lon, data, pkg, coast_segs, dates, out_buf=None,
           title=None, cbar_label=None):
    fplot = data["main"]
    vlim, cint = pkg["vlim"], pkg["cint"]
    LON2D, LAT2D = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    # reserve a clean title band above the axes so the title never overlaps the map
    ax = fig.add_axes([0.045, 0.145, 0.910, 0.750])
    ax.set_facecolor("#f4f0e8")
    lon_min, lon_max, lat_min, lat_max = 0.0, 360.0, -80.0, 80.0
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # lon ticks can wrap (e.g. Atlantic 300E..60E) -> normalise
    xticks = _domain_xticks(lon_min, lon_max)
    yticks = _domain_yticks(lat_min, lat_max)

    # filled shading. A magnitude field (one_sided) is shaded 0→vlim with a
    # non-negative (green) colormap; a signed anomaly uses the symmetric
    # diverging map (or its inverse when invert_cbar is set, e.g. MSE).
    invert = pkg.get("invert_cbar", False)
    if pkg.get("one_sided"):
        levels_fill = np.linspace(0.0, vlim, 20)
        cf = ax.contourf(LON2D, LAT2D, fplot, levels=levels_fill,
                         cmap=_pos_cmap(), extend="max", zorder=1, alpha=0.88)
    else:
        n_fill = 25 if vlim >= 100 else 20
        levels_fill = np.linspace(-vlim, vlim, n_fill)
        cmap = _chi_cmap_inv() if invert else _chi_cmap()
        cf = ax.contourf(LON2D, LAT2D, fplot, levels=levels_fill,
                         cmap=cmap, extend="both", zorder=1, alpha=0.88)

    # thin contour lines (solid positive, dashed negative); swapped when inverted
    pos_col = "#1b4f6b" if invert else "#5c3d11"
    neg_col = "#5c3d11" if invert else "#1b4f6b"
    line_lev = np.arange(0 if pkg.get("one_sided") else -vlim, vlim + 0.01, cint)
    line_lev = line_lev[line_lev != 0]
    ax.contour(LON2D, LAT2D, fplot, levels=line_lev[line_lev > 0],
               colors=pos_col, linewidths=0.55, alpha=0.55, zorder=2)
    ax.contour(LON2D, LAT2D, fplot, levels=line_lev[line_lev < 0],
               colors=neg_col, linewidths=0.55, linestyles="--",
               alpha=0.55, zorder=2)

    # vector overlay — either wind (u,v) or a generic flux (vec_u, vec_v, e.g. WAF)
    vec = data.get("vec_u"), data.get("vec_v")
    if pkg.get("show_wind") or vec[0] is not None:
        if vec[0] is not None:
            U0, V0 = vec
            ref_mag = pkg.get("vec_ref", 5.0)
            ref_unit = pkg.get("vec_unit", "5 m/s")
            vscale = pkg.get("wind_scale", 50.0)
        else:
            U0, V0 = data["u"], data["v"]
            ref_mag = 5.0
            ref_unit = "5 m/s"
            vscale = pkg["wind_scale"]
        # flux overlays (e.g. WAF): sparser grid + minimum-magnitude filter so
        # the field is readable; wind overlays keep a denser, un-thresholded grid.
        is_flux = vec[0] is not None
        step = pkg.get("vec_step", 3) if is_flux else 3
        vmin = pkg.get("vec_min", 0.0) if is_flux else 0.0
        qs = slice(None, None, step)
        Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
        Uq, Vq = U0[qs, qs], V0[qs, qs]
        mag = np.sqrt(Uq**2 + Vq**2)
        mask = (~np.isnan(mag)) & (np.abs(Yq) <= lat_max) & (mag >= vmin)
        ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask], color="#111111",
                  scale=vscale, scale_units="inches", width=0.0018,
                  headwidth=4.5, headlength=5.5, headaxislength=4.8,
                  minshaft=1.2, pivot="middle", zorder=6, alpha=0.92)
        # reference arrow (domain-aware placement)
        rx = lon_min + 0.4 * (lon_max - lon_min)
        ry = lat_min + 0.06 * (lat_max - lat_min)
        ax.quiver(rx, ry, ref_mag, 0, color="#111111",
                  scale=vscale, scale_units="inches", width=0.0018,
                  headwidth=4.5, headlength=5.5, headaxislength=4.8,
                  pivot="tail", zorder=9)
        ax.text(rx, ry - 0.08 * (lat_max - lat_min), ref_unit, fontsize=8,
                color="#111111", ha="center", zorder=9)

    # coastlines
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

    # grid lines (use the domain-aware tick positions)
    for x in xticks:
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in yticks:
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.75, zorder=0, alpha=0.8)

    # axes
    ax.set_xticks(xticks)
    ax.set_xticklabels([_xlabel(x) for x in xticks], fontsize=9.5,
                       color="#333322", fontfamily="DejaVu Sans")
    ax.set_yticks(yticks)
    ax.set_yticklabels([_ylabel(y) for y in yticks], fontsize=9.5,
                       color="#333322", fontfamily="DejaVu Sans")
    ax.tick_params(axis="both", length=3.5, color="#888878", width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988")
        spine.set_linewidth(0.8)

    # colorbar
    cax = fig.add_axes([0.12, 0.057, 0.760, 0.028])
    lo = 0.0 if pkg.get("one_sided") else -vlim
    ticks = np.array([round(v, 8) for v in np.arange(lo, vlim + 0.001, cint)])
    if pkg.get("one_sided"):
        ticks = ticks[ticks > 0.0]                 # no negative labels
    ticks = ticks[~np.isclose(ticks, 0.0, atol=cint * 0.01)]   # kill fp noise at 0
    ticks = np.append(0.0, ticks)  # keep an exact 0 label
    ticks = np.unique(ticks)
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.tick_params(labelsize=8.5, colors="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8.5,
                            color="#222211")
    cbar.outline.set_edgecolor("#999988")
    cbar.outline.set_linewidth(0.7)
    cb_lbl = cbar_label if cbar_label is not None else pkg["cb_label"]
    cax.text(0.5, -1.55, cb_lbl, transform=cax.transAxes, ha="center",
             va="top", fontsize=12, color="#222211", fontstyle="italic")

    # title & branding  (single line, in its own band — never over the map)
    if title is None:
        ttext = (f"{pkg['title']}  ·  {dates[0]:%-d %b} – {dates[-1]:%-d %b %Y}"
                 f"  ({len(dates)}-day mean)")
    else:
        ttext = title
    fig.text(0.50, 0.965, ttext, ha="center", va="top", fontsize=16,
             fontweight="bold", color="#111100", fontfamily="DejaVu Sans")
    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=11,
            va="bottom", ha="right", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#ccccbb", alpha=0.92, lw=0.9), zorder=10)
    ax.text(0.005, 0.016, "NCEP/NCAR Reanalysis  ·  PSL/NOAA",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            color="#666655", zorder=10)

    if out_buf is None:
        out_buf = io.BytesIO()
    plt.savefig(out_buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    out_buf.seek(0)
    return out_buf

def render_hov(day_dates, lon, matrix, pkg, out_buf=None, title=None,
               cbar_label=None, lat_lab=None):
    """Render a Hovmöller: longitude (x) vs date (y) as a filled contour.
    The time axis runs top (oldest) -> bottom (latest date)."""
    vlim, cint = pkg["vlim"], pkg["cint"]
    ntime, nlon = matrix.shape
    # date axis as fractional day for even spacing
    t0 = day_dates[0]
    days = np.array([(d - t0).days for d in day_dates], dtype=np.float64)
    LON2D, DAY2D = np.meshgrid(lon, days)

    # taller figure so the date axis has more vertical room
    fig = plt.figure(figsize=(12, 8.6), facecolor="white")
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.77])
    ax.set_facecolor("#f4f0e8")
    # latest (newest) date at the bottom
    ax.set_ylim(days.max(), days.min())

    invert = pkg.get("invert_cbar", False)
    n_fill = 25 if vlim >= 100 else 20
    levels = np.linspace(-vlim, vlim, n_fill)
    cmap = _chi_cmap_inv() if invert else _chi_cmap()
    cf = ax.contourf(LON2D, DAY2D, np.nan_to_num(matrix, nan=0.0),
                     levels=levels, cmap=cmap, extend="both", zorder=1, alpha=0.9)
    line_lev = np.arange(-vlim, vlim + 0.01, cint)
    line_lev = line_lev[line_lev != 0]
    ax.contour(LON2D, DAY2D, np.nan_to_num(matrix, nan=0.0),
               levels=line_lev[line_lev > 0], colors="#5c3d11",
               linewidths=0.55, alpha=0.55, zorder=2)
    ax.contour(LON2D, DAY2D, np.nan_to_num(matrix, nan=0.0),
               levels=line_lev[line_lev < 0], colors="#1b4f6b",
               linewidths=0.55, linestyles="--", alpha=0.55, zorder=2)

    # longitude ticks
    xticks = _domain_xticks(lon.min(), lon.max())
    ax.set_xticks(xticks)
    ax.set_xticklabels([_xlabel(x) for x in xticks], fontsize=9.5,
                       color="#333322")
    # date ticks (about 6 evenly spaced)
    nd = min(6, ntime)
    idxs = np.linspace(0, ntime - 1, nd).round().astype(int)
    ax.set_yticks([days[i] for i in idxs])
    ax.set_yticklabels([day_dates[i].strftime("%d %b") for i in idxs],
                       fontsize=9.5, color="#333322")

    # zero-dateline vertical reference + equator label
    ax.axvline(0, color="#666655", lw=0.7, alpha=0.6)
    ax.grid(True, ls=":", color="#b0a898", lw=0.35, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988")
        spine.set_linewidth(0.8)

    # colorbar
    cax = fig.add_axes([0.12, 0.055, 0.760, 0.028])
    ticks = np.array([round(v, 8) for v in np.arange(-vlim, vlim + 0.001, cint)])
    ticks = ticks[~np.isclose(ticks, 0.0, atol=cint * 0.01)]
    ticks = np.append(0.0, ticks)
    ticks = np.unique(ticks)
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.tick_params(labelsize=8.5, color="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8.5,
                            color="#222211")
    cbar.outline.set_edgecolor("#999988"); cbar.outline.set_linewidth(0.7)
    cb_lbl = cbar_label if cbar_label is not None else pkg["cb_label"]
    cax.text(0.5, -1.55, cb_lbl, transform=cax.transAxes, ha="center",
             va="top", fontsize=12, color="#222211", fontstyle="italic")

    if title is None:
        ttext = (f"{pkg['title']}  ·  Longitude–Time (Hovmöller)  ·  "
                 f"{day_dates[0]:%-d %b} – {day_dates[-1]:%-d %b %Y}")
    else:
        ttext = title
    ax.set_title(ttext, fontsize=16, fontweight="bold", color="#111100", pad=14)
    if lat_lab:
        ax.text(0.006, 0.035, lat_lab, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=10.5, color="#555544",
                fontweight="semibold", zorder=9)
    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=11,
            va="bottom", ha="right", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#ccccbb", alpha=0.92, lw=0.9), zorder=10)

    if out_buf is None:
        out_buf = io.BytesIO()
    plt.savefig(out_buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    out_buf.seek(0)
    return out_buf

def render_rossby(lat, lon, data, pkg, coast_segs, dates, out_buf=None,
                  title=None, cbar_label=None):
    """Rossby-wave composite: χ′ shading (green source / brown suppressed),
    ψ′ contours (firebrick ridge / royal-blue dashed trough), Takaya–Nakamura
    wave-activity-flux arrows (black), and the Ks≥K waveguide band (indigo)."""
    chi = data["main"]
    psi = data["psi"]
    ks = data["ks"]
    wu, wv = data["vec_u"], data["vec_v"]
    vlim, cint = pkg["vlim"], pkg["cint"]
    ks_thr = pkg.get("ks_threshold", 5.0)
    LON2D, LAT2D = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(13, 7.4), facecolor="white")
    ax = fig.add_axes([0.045, 0.145, 0.910, 0.750])
    ax.set_facecolor("#f4f0e8")
    lon_min, lon_max, lat_min, lat_max = 0.0, 360.0, -80.0, 80.0
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # (0) Ks waveguide band — drawn beneath the shading, across all longitudes
    kmask = np.isfinite(ks) & (ks >= ks_thr)
    for lo, hi in _lat_runs(kmask, lat):
        ax.fill_between([lon_min, lon_max], lo, hi, color="#3b0f98",
                        alpha=0.20, lw=0, zorder=0.3)
        ax.plot([lon_min, lon_max], [lo, lo], color="#3b0f98", lw=1.0,
                alpha=0.55, zorder=0.3)
        ax.plot([lon_min, lon_max], [hi, hi], color="#3b0f98", lw=1.0,
                alpha=0.55, zorder=0.3)

    # (1) χ′ source shading (green = negative/source, brown = positive/suppressed)
    levels_fill = np.linspace(-vlim, vlim, 20)
    cf = ax.contourf(LON2D, LAT2D, chi, levels=levels_fill,
                     cmap=_source_cmap(), extend="both", zorder=1, alpha=0.9)

    # (2) ψ′ wave-train contours. Consistent convention in BOTH hemispheres:
    #     firebrick solid = ridge / anticyclonic (H), blue dashed = trough /
    #     cyclonic (L).
    pstep = pkg["psi_interval"]
    pvlim = pkg["psi_vlim"]
    plev = np.arange(-pvlim, pvlim + 0.01, pstep)
    plev = plev[plev != 0]
    ax.contour(LON2D, LAT2D, psi, levels=plev[plev > 0], colors="#c0392b",
               linewidths=1.0, alpha=0.95, zorder=3)
    ax.contour(LON2D, LAT2D, psi, levels=plev[plev < 0], colors="#1e2f9c",
               linewidths=1.0, linestyles="--", alpha=0.95, zorder=3)

    # (2b) H / L centre labels at fully enclosed wave-train circulation cells
    # (only cells wrapped by a closed, rounded contour loop are marked).
    # Matches the contour convention: H = ridge (firebrick), L = trough
    # (royal-blue). In the SOUTHERN hemisphere the H and L letters are swapped
    # (H <-> L) — the marker colour still follows the contour type.
    for cx, cy, lab in _psi_centers(psi, lat, lon, plev):
        col = "#c0392b" if lab == "H" else "#1e2f9c"
        disp = lab
        if cy < 0.0:                            # Southern hemisphere: swap H <-> L
            disp = "L" if lab == "H" else "H"
        ax.text(cx, cy, disp, color=col, fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.06", fc="white",
                          ec="none", alpha=0.85, lw=0))

    # (3) Takaya–Nakamura wave-activity flux (black arrows)
    step = pkg.get("vec_step", 5)
    vmin = pkg.get("vec_min", 15.0)
    vscale = pkg.get("wind_scale", 400.0)
    qs = slice(None, None, step)
    Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
    Uq, Vq = wu[qs, qs], wv[qs, qs]
    mag = np.sqrt(Uq ** 2 + Vq ** 2)
    mask = (~np.isnan(mag)) & (np.abs(Yq) <= lat_max) & (mag >= vmin)
    ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask], color="#111111",
              scale=vscale, scale_units="inches", width=0.0020,
              headwidth=4.5, headlength=5.5, headaxislength=4.8,
              minshaft=1.2, pivot="middle", zorder=6, alpha=0.95)

    _draw_coasts(ax, coast_segs)
    _draw_axis(ax, lon_min, lon_max, lat_min, lat_max)

    # colorbar (source shading)
    cax = fig.add_axes([0.12, 0.057, 0.760, 0.028])
    ticks = np.array([round(v, 8) for v in np.arange(-vlim, vlim + 0.001, cint)])
    ticks = ticks[~np.isclose(ticks, 0.0, atol=cint * 0.01)]
    ticks = np.append(0.0, ticks)
    ticks = np.unique(ticks)
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.tick_params(labelsize=8.5, color="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8.5,
                            color="#222211")
    cbar.outline.set_edgecolor("#999988"); cbar.outline.set_linewidth(0.7)
    cb_lbl = cbar_label if cbar_label is not None else pkg["cb_label"]
    cax.text(0.5, -1.55, cb_lbl, transform=cax.transAxes, ha="center",
             va="top", fontsize=12, color="#222211", fontstyle="italic")

    if title is None:
        ttext = (f"{pkg['title']}  ·  {dates[0]:%-d %b} – {dates[-1]:%-d %b %Y}"
                 f"  ({len(dates)}-day mean)")
    else:
        ttext = title
    fig.text(0.50, 0.965, ttext, ha="center", va="top", fontsize=16,
             fontweight="bold", color="#111100", fontfamily="DejaVu Sans")
    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=11,
            va="bottom", ha="right", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#ccccbb", alpha=0.92, lw=0.9), zorder=10)
    if out_buf is None:
        out_buf = io.BytesIO()
    plt.savefig(out_buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    out_buf.seek(0)
    return out_buf
