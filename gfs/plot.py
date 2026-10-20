"""
plot.py  —  GFS Map Drawing
============================
Changes:
  • U/V wind: strict zero-centred symmetric colorbar (Blue=−, Red=+)
  • VP / VP-anomaly: proper rendering with ×10⁶ scaling label
  • Stream Function: contour + streamlines overlay
  • SF + PWAT overlay: PWAT shaded + SF streamlines + SF contours
  • Anomaly maps: diverging colourmap, zero-centred, "ANOMALY" badge
  • Precip title updated: shows N-day SUM when avg_days>0
"""

import os, io, zipfile, datetime
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from shp_reader import load_country_shapes as _shp_countries, load_coastlines as _shp_coasts
import requests

from config import (
    SHP_DIR, SHP_PATH, SHP_URL, COAST_PATH, COAST_URL,
    SMOOTH_SIGMA, PLOT_CONFIGS,
)

# ── Runtime region bounds ────────────────────────────────────────────
LON_MIN = 65.0
LON_MAX = 100.0
LAT_MIN = 5.0
LAT_MAX = 40.0

def set_region(lon_min, lon_max, lat_min, lat_max):
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = lon_min, lon_max, lat_min, lat_max


# ════════════════════════════════════════════════════════════════════
#  Precipitation colormap
# ════════════════════════════════════════════════════════════════════
precip_colors = [
    "#ffffff", "#e0ffff", "#00c8ff", "#0000ff", "#00ff00",
    "#009900", "#ffff00", "#ff9900", "#ff0000", "#cc00cc", "#ffffff"
]
PROFESSIONAL_PRECIP_CMAP = LinearSegmentedColormap.from_list(
    "precip_pro", precip_colors, N=256)


# ════════════════════════════════════════════════════════════════════
#  Shapefile helpers
# ════════════════════════════════════════════════════════════════════

def ensure_shapefile(shp_path, url, label):
    if os.path.exists(shp_path): return
    os.makedirs(SHP_DIR, exist_ok=True)
    print(f"  Downloading {label} ...")
    r = requests.get(url, timeout=90); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if os.path.splitext(name)[1] in (".shp",".shx",".dbf",".prj",".cpg"):
            with open(os.path.join(SHP_DIR, os.path.basename(name)), "wb") as f:
                f.write(z.read(name))


def load_country_shapes(shp_path):
    return _shp_countries(shp_path)

def load_coastlines(path):
    return _shp_coasts(path)


def plot_shape(ax, seg, color, lw, zorder):
    lons = np.where(seg[:,0] < 0, seg[:,0]+360, seg[:,0])
    lats = seg[:,1]
    brks = np.where(np.abs(np.diff(lons)) > 100)[0] + 1
    for pl, pt in zip(np.split(lons, brks), np.split(lats, brks)):
        if len(pl) > 1: ax.plot(pl, pt, color=color, lw=lw, zorder=zorder)


def load_shapes():
    ensure_shapefile(COAST_PATH, COAST_URL,  "NE-50m coastline")
    ensure_shapefile(SHP_PATH,   SHP_URL,    "NE-50m countries")
    coast_segs   = load_coastlines(COAST_PATH)
    country_segs = load_country_shapes(SHP_PATH)
    return country_segs, coast_segs


# ════════════════════════════════════════════════════════════════════
#  Smooth
# ════════════════════════════════════════════════════════════════════

def smooth(arr):
    from scipy.interpolate  import RegularGridInterpolator
    from scipy.ndimage      import distance_transform_edt
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    nm = np.isnan(arr); af = arr.copy()
    if nm.any():
        _, idx = distance_transform_edt(nm, return_indices=True)
        af = af[tuple(idx)]
    af = np.where(np.isnan(af), 0.0, af)
    nl, no = af.shape; factor = 10
    try:
        itp = RegularGridInterpolator(
            (np.arange(nl), np.arange(no)), af,
            method="cubic", bounds_error=False, fill_value=None)
    except:
        itp = RegularGridInterpolator(
            (np.arange(nl), np.arange(no)), af,
            method="linear", bounds_error=False, fill_value=None)
    lh  = np.linspace(0, nl-1, nl*factor)
    loh = np.linspace(0, no-1, no*factor)
    LH, LOH = np.meshgrid(loh, lh)
    out = itp((LOH, LH))
    out = np.where(np.isnan(out), 0.0, out)
    return gaussian_filter(out, max(SMOOTH_SIGMA*factor, 1.0))


# ════════════════════════════════════════════════════════════════════
#  Colour-level helpers
# ════════════════════════════════════════════════════════════════════

def _symmetric_levels(data, n=60):
    """Strictly zero-centred levels — ensures 0 sits exactly at colormap midpoint."""
    absmax = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))), 1e-6)
    return np.linspace(-absmax, absmax, n)


def _is_anomaly_var(variable):
    return variable.endswith("_anomaly")


def _is_symmetric_var(variable):
    return variable in {"u", "v", "vp", "vp_anomaly",
                        "temp_anomaly", "mslp_anomaly", "rh_anomaly",
                        "streamfunc", "streamfunc_anomaly",
                        "sf_pwat_anomaly", "wind_anomaly"}


# ════════════════════════════════════════════════════════════════════
#  Main draw function
# ════════════════════════════════════════════════════════════════════

def draw_map(run_dt, step, country_segs, coast_segs,
             lat, lon, data, extra_data=None,
             variable="wind", level=850,
             avg_days=0, region_name="South Asia",
             out_file=None):

    cfg = PLOT_CONFIGS.get(variable, PLOT_CONFIGS.get(variable.replace("_anomaly",""), PLOT_CONFIGS["wind"]))

    # ── Original stats ──
    orig_min = float(np.nanmin(data))
    orig_max = float(np.nanmax(data))

    # ── Smooth ──
    if SMOOTH_SIGMA > 0:
        sdata = smooth(data)
        s_min, s_max = np.nanmin(sdata), np.nanmax(sdata)
        if s_max > s_min:
            sdata = (sdata - s_min) / (s_max - s_min) * (orig_max - orig_min) + orig_min
        lons_s = np.linspace(lon[0], lon[-1], sdata.shape[1])
        lats_s = np.linspace(lat[0], lat[-1], sdata.shape[0])
        LON_G, LAT_G = np.meshgrid(lons_s, lats_s)
    else:
        LON_G, LAT_G = np.meshgrid(lon, lat)
        sdata = data

    # ── Colour levels & colormap ──
    is_precip    = (variable == "precip")
    is_sf_pwat   = variable in ("sf_pwat", "sf_pwat_anomaly")
    is_symmetric = _is_symmetric_var(variable)
    is_anom      = _is_anomaly_var(variable)

    if is_precip:
        cmap_use  = PROFESSIONAL_PRECIP_CMAP
        max_val   = max(orig_max, 1.0)
        clevs     = np.linspace(0, max_val, 150)
        extend_opt = "neither"
        norm_use  = None

    elif is_sf_pwat:
        # PWAT shading
        cmap_use  = cfg["cmap"]
        clevs     = cfg["clevs"](sdata)
        extend_opt = "both"
        norm_use  = None

    elif is_symmetric or is_anom:
        # Zero-centred diverging colourmap
        cmap_use  = cfg["cmap"]
        clevs     = _symmetric_levels(sdata, 60)
        extend_opt = "both"
        norm_use  = TwoSlopeNorm(vcenter=0.0,
                                  vmin=clevs[0], vmax=clevs[-1])
    else:
        cmap_use  = cfg["cmap"]
        clevs     = cfg["clevs"](sdata)
        extend_opt = "both"
        norm_use  = None

    # ── Dynamic figure size ──
    lon_span = abs(LON_MAX - LON_MIN)
    lat_span = abs(LAT_MAX - LAT_MIN)
    mean_lat = np.radians((LAT_MIN + LAT_MAX) / 2.0)
    aspect_ratio = (lon_span * np.cos(mean_lat)) / lat_span if lat_span > 0 else 1.0

    base_height = 9.0
    calc_width  = base_height * aspect_ratio
    fig_width   = max(8.0, min(16.0, calc_width))
    fig_height  = fig_width / aspect_ratio
    fig_height  = max(6.0, min(12.0, fig_height))

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    ax  = fig.add_axes([0.06, 0.06, 0.84, 0.86])
    ax.set_facecolor("#f8f8f8")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(1.0 / np.cos(mean_lat), adjustable='box')

    # ── Filled contour (main shading) ──
    cf_kwargs = dict(levels=clevs, cmap=cmap_use,
                     extend=extend_opt, zorder=1, alpha=cfg["alpha"])
    if norm_use is not None:
        cf_kwargs["norm"] = norm_use
    cf = ax.contourf(LON_G, LAT_G, sdata, **cf_kwargs)

    # ── Optional line contours ──
    if cfg.get("contour", False) and not is_sf_pwat:
        if is_symmetric or is_anom:
            step_c = max(abs(clevs[-1]) / 8, 0.5)
            c_lev  = np.arange(-abs(clevs[-1]), abs(clevs[-1]) + step_c, step_c)
            c_lev  = c_lev[c_lev != 0]
        elif variable == "mslp":
            step_c = 4.0
            c_lev  = np.arange(np.floor(orig_min), np.ceil(orig_max) + step_c, step_c)
        else:
            step_c = 2.0
            c_lev  = np.arange(np.floor(orig_min), np.ceil(orig_max) + step_c, step_c)
        cs = ax.contour(LON_G, LAT_G, sdata, levels=c_lev,
                        colors="#333333", linewidths=0.5, zorder=3)
        ax.clabel(cs, fmt="%.0f", fontsize=7, inline=True)

    # ── Stream Function overlay on SF+PWAT (normal or anomaly) ──
    if is_sf_pwat and extra_data is not None and len(extra_data) == 3:
        u_sf, v_sf, psi = extra_data
        # SF contour lines over PWAT shading
        psi_s = smooth(psi)
        p_abs  = max(abs(float(np.nanmin(psi_s))), abs(float(np.nanmax(psi_s))), 1e-6)
        sf_levs = np.linspace(-p_abs, p_abs, 20)
        lons_p = np.linspace(lon[0], lon[-1], psi_s.shape[1])
        lats_p = np.linspace(lat[0], lat[-1], psi_s.shape[0])
        LON_P, LAT_P = np.meshgrid(lons_p, lats_p)
        cs_sf = ax.contour(LON_P, LAT_P, psi_s, levels=sf_levs,
                           colors=["#000080" if l < 0 else "#8b0000" for l in sf_levs],
                           linewidths=0.8, zorder=3, alpha=0.85)
        ax.clabel(cs_sf, fmt="%.1f", fontsize=6.5, inline=True)
        # Wind arrows (quiver) from U,V
        _draw_arrows(ax, lat, lon, u_sf, v_sf, zorder=4)

    # ── SF / SF-anomaly: contours + wind arrows ──
    elif variable in ("streamfunc", "streamfunc_anomaly") and extra_data is not None and len(extra_data) == 2:
        u2d, v2d = extra_data
        # SF contour lines
        sf_s = smooth(data)
        sf_abs = max(abs(float(np.nanmin(sf_s))), abs(float(np.nanmax(sf_s))), 1e-6)
        sf_levs = np.linspace(-sf_abs, sf_abs, 16)
        LON_SF, LAT_SF = np.meshgrid(
            np.linspace(lon[0], lon[-1], sf_s.shape[1]),
            np.linspace(lat[0], lat[-1], sf_s.shape[0]))
        cs_sf = ax.contour(LON_SF, LAT_SF, sf_s, levels=sf_levs,
                           colors=["#000080" if l < 0 else "#8b0000" for l in sf_levs],
                           linewidths=0.7, zorder=3, alpha=0.9)
        ax.clabel(cs_sf, fmt="%.1f", fontsize=6.5, inline=True)
        # Wind arrows (quiver) — actual U,V wind
        _draw_arrows(ax, lat, lon, u2d, v2d, zorder=4)

    # ── Regular wind streamlines (wind speed map only) ──
    elif cfg.get("streamlines", False) and extra_data is not None and len(extra_data) == 2:
        u2d, v2d = extra_data
        _draw_streamlines(ax, lat, lon, u2d, v2d, zorder=4)

    # ── Colorbar ──
    cbar_ax = fig.add_axes([0.91, 0.08, 0.02, 0.82])

    if is_precip:
        cbar_ticks = np.linspace(0, max_val, 10)
    elif is_symmetric or is_anom:
        absmax = max(abs(clevs[0]), abs(clevs[-1]))
        cbar_ticks = np.linspace(-absmax, absmax, 9)
    else:
        cbar_ticks = np.linspace(np.nanmin(clevs), np.nanmax(clevs), 9)

    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="vertical", ticks=cbar_ticks)
    cbar.set_label(f"{cfg['label'](level)} ({cfg['unit']})",
                   fontsize=10.5, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # ── Add zero line marker to symmetric colorbars ──
    if is_symmetric or is_anom:
        cbar.ax.axhline(0, color="black", linewidth=1.2, zorder=10)

    # ── Borders ──
    for seg in country_segs["others"]:
        plot_shape(ax, seg, "#444444", 0.60, 5)
    for seg in country_segs["India"]:
        plot_shape(ax, seg, "#1b4d1b", 1.50, 6)
    for seg in country_segs["Bangladesh"]:
        plot_shape(ax, seg, "#8b0000", 1.80, 7)
    for seg in coast_segs:
        plot_shape(ax, seg, "#111111", 0.85, 8)

    # ── Grid ──
    lon_tick_step = max(2, int(lon_span/7/2)*2) if lon_span<=40 else max(5,int(lon_span/8/5)*5)
    lat_tick_step = max(2, int(lat_span/7/2)*2) if lat_span<=40 else max(5,int(lat_span/8/5)*5)
    ax.grid(True, color="#333333", linestyle="--", linewidth=0.4, alpha=0.4, zorder=3)
    lon_ticks = range(int(np.ceil(LON_MIN)), int(np.floor(LON_MAX))+1, max(1,lon_tick_step))
    lat_ticks = range(int(np.ceil(LAT_MIN)), int(np.floor(LAT_MAX))+1, max(1,lat_tick_step))
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{x}°E" if x>=0 else f"{-x}°W" for x in lon_ticks], fontsize=8)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{y}°N" if y>=0 else f"{-y}°S" for y in lat_ticks], fontsize=8)

    # ── Value stats box ──
    stat_txt = f"Max: {orig_max:.2f} | Min: {orig_min:.2f}"
    ax.text(0.98, 0.975, stat_txt, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#111111",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#a0a0a0", alpha=0.9, lw=0.8), zorder=12)

    # ── Country labels ──
    if LAT_MIN < 30 and LON_MIN < 90 and LON_MAX > 75:
        for txt, x, y, fs, col in [("INDIA", 79.0, 22.5, 11, "#1a5c1a"),
                                     ("BGD",  90.3, 23.7,  7, "#8b0000")]:
            if LON_MIN <= x <= LON_MAX and LAT_MIN <= y <= LAT_MAX:
                ax.text(x, y, txt, fontsize=fs, fontweight="bold", color=col,
                        ha="center", va="center", zorder=9, alpha=0.85,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

    # ── Badges ──
    badge_x = 0.015
    if avg_days > 0:
        if variable == "precip":
            badge_label = f"{avg_days}-DAY TOTAL"
            badge_fc    = "#fff0d0"
            badge_ec    = "#cc6600"
            badge_col   = "#7a3300"
        else:
            badge_label = f"{avg_days}-DAY AVG"
            badge_fc    = "#ddeeff"
            badge_ec    = "#3366cc"
            badge_col   = "#003399"
        ax.text(badge_x, 0.975, badge_label,
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color=badge_col, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc=badge_fc,
                          ec=badge_ec, alpha=0.95, lw=1.0), zorder=12)

    if is_anom:
        ax.text(0.015, 0.935, "ANOMALY",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                color="#660000", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffe0e0",
                          ec="#cc0000", alpha=0.95, lw=1.0), zorder=12)

    # ── Title ──
    avg_label = avg_days if avg_days > 0 else 0
    title_var = cfg["title_var"](level, avg_label if avg_label > 0 else None)

    if avg_days > 0:
        total_hours = avg_days * 24
        valid_start = run_dt
        valid_end   = run_dt + datetime.timedelta(hours=total_hours)
        valid_str   = (f"Valid: {valid_start.strftime('%d %b')} – "
                       f"{valid_end.strftime('%d %b %Y %H:%M UTC')}  (0h – {total_hours}h)")
    else:
        valid_dt  = run_dt + datetime.timedelta(hours=step)
        valid_str = f"Valid: {valid_dt.strftime('%d %b %Y %H:%M UTC')}  (+{step}h)"

    ax.set_title(
        f"NCEP GFS  |  {title_var}  |  Region: {region_name}\n"
        f"Run: {run_dt.strftime('%d %b %Y %H:%M UTC')}   {valid_str}",
        fontsize=11, fontweight="bold", color="#111100", pad=8)

    # ── Source & watermark ──
    ax.text(0.015, 0.020, "NCEP GFS",
            transform=ax.transAxes, fontsize=8, color="#222222",
            alpha=0.65, ha="left", va="bottom", zorder=10)
    ax.text(0.985, 0.015, "@XPWEATHER",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom", color="#222211", zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#ccccbb", alpha=0.9, lw=0.8))

    # ── Save ──
    avg_tag  = f"avg{avg_days}d_" if avg_days > 0 else ""
    anom_tag = "anom_" if is_anom else ""
    if out_file is None:
        out_file = f"gfs_{variable}_{level}mb_{anom_tag}{avg_tag}f{step:03d}.png"
    plt.savefig(out_file, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Map saved → {out_file}")
    return out_file


# ════════════════════════════════════════════════════════════════════
#  Wind vector helpers
# ════════════════════════════════════════════════════════════════════

def _draw_streamlines(ax, lat, lon, u2d, v2d, zorder=4):
    """Streamplot — used only for plain wind-speed maps."""
    lat_w, lon_w = lat.copy(), lon.copy()
    if lat_w[0] > lat_w[-1]:
        lat_w = lat_w[::-1]; u2d = u2d[::-1,:]; v2d = v2d[::-1,:]
    try:
        ax.streamplot(lon_w, lat_w, u2d, v2d,
                      color="#111111", density=1.4,
                      linewidth=0.75, arrowsize=0.85, zorder=zorder)
    except Exception as e:
        print(f"  Streamlines failed: {e}")


def _draw_arrows(ax, lat, lon, u2d, v2d, zorder=4, skip=None):
    """
    Wind arrows (quiver) — used for Stream Function maps.
    Adaptively subsamples so arrow density stays readable regardless of domain size.
    skip=None → auto computed from grid size.
    """
    lat_w, lon_w = lat.copy(), lon.copy()
    if lat_w[0] > lat_w[-1]:
        lat_w = lat_w[::-1]; u2d = u2d[::-1, :]; v2d = v2d[::-1, :]

    nj, ni = u2d.shape
    if skip is None:
        # Target ~20 arrows in each direction
        skip = max(1, min(nj, ni) // 20)

    u_s = u2d[::skip, ::skip]
    v_s = v2d[::skip, ::skip]
    lo_s = lon_w[::skip]
    la_s = lat_w[::skip]
    LON_Q, LAT_Q = np.meshgrid(lo_s, la_s)

    # Normalise to unit length so all arrows same size (direction only)
    spd = np.sqrt(u_s**2 + v_s**2)
    spd = np.where(spd < 1e-6, 1.0, spd)
    un = u_s / spd
    vn = v_s / spd

    try:
        ax.quiver(LON_Q, LAT_Q, un, vn,
                  scale=28, scale_units="width",
                  width=0.0025, headwidth=4.5, headlength=5,
                  color="#111111", alpha=0.82, zorder=zorder)
    except Exception as e:
        print(f"  Arrows (quiver) failed: {e}")
