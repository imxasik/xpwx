"""custom/rel_aam.py — Relative AAM: latitude–time section + global anomaly.

Drop-in addon (no core file is touched).  Sidebar group: **Hovmöller**,
product id ``rel_aam``.

The chart follows the classic GSDM/AAM layout, two panels:

  1. **Latitude–time section** of the vertically and zonally integrated
     relative angular momentum,

         A(phi, t) = (a^3/g) ∬ u cos^2(phi) dp dlambda   [kg m^2 s^-1 per degree]

     * shading  — the anomaly against the 1991–2020 daily climatology at the
                  same latitudes, in 10^24 kg m^2 s^-1 per 2.5-degree latitude
                  box (the native grid spacing, as on the classic GSDM chart).
                  The palette is the classic AAM one: blue/purple/green below
                  −0.5, white inside ±0.5, yellow/orange/red above +0.5.
     * contours — the total (absolute) field, solid above zero and dashed
                  below, at a round contour interval printed in the caption.
  2. **Global AAM anomaly** — the section integrated over the whole globe, in
     Hadley units (10^25 kg m^2 s^-1); it is the numerator of the standardised
     anomaly M = Mr' / sigma_M used by the GWO phase diagram.

The integration, the 2/2 latitude/longitude strides and the NOAA baseline
sigmas come from ``custom/gwo.py``, so this product and the GWO Phase product
can never disagree.  The latitude-resolved daily climatology is derived once
from the 1991–2020 LTM file and cached in ``ltm_rel_aam_cache.npy`` in the
project root — rebuilt only when the LTM file or the grid changes, never on a
plain restart (``custom/_warm_cache.py`` builds it at deploy time, so no request
ever pays for it).  The committed ``ltm_aam_cache.npy`` of the GWO product is
never rewritten.

Data: NOAA PSL NCEP/NCAR daily ``uwnd`` plus ``uwnd.day.ltm.1991-2020.nc`` — the
same files the rest of the app downloads and caches.
"""
import io
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

from pro import config
from pro.data import open_remote, release_memory

# ---------------------------------------------------------------------------
# shared AAM machinery — one implementation for both AAM products
# ---------------------------------------------------------------------------
_ROOT = config.BASE_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:                                        # the sibling addon, imported normally
    from custom import gwo as _gwo
except Exception:                           # pragma: no cover - odd sys.path
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "xp_rel_aam_gwo", os.path.join(_ROOT, "custom", "gwo.py"))
    _gwo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gwo)

_LOCK = threading.Lock()          # pyplot state is global; serialise draws

# ---------------------------------------------------------------------------
# settings (the product dict may override window / smoothing)
# ---------------------------------------------------------------------------
WINDOW_DAYS = 120                 # days ending on the app's selected end date
SMOOTH_DAYS = 5                   # running mean used for the drawn fields
CLIM_SMOOTH = 31                  # cyclic smoothing of the daily climatology

HU = 1.0e25                       # Hadley unit, 10^25 kg m^2 s^-1
DEG = np.pi / 180.0               # per-radian -> per-degree latitude

# Quantities per latitude grid *box* rather than per degree.
#
# The classic GSDM/AAM chart shades the AAM anomaly per latitude box of the
# reanalysis grid, while its total-field contours are per degree.  The two are
# related by the grid spacing: shading = (per-degree anomaly) x SHADE_BAND.
# Without this factor the per-degree anomaly of the real atmosphere (~0.2-1.0
# in 10^24 kg m^2 s^-1/deg, i.e. 3-10 m/s zonal-mean wind anomalies) sits
# almost entirely inside the white band of the scale and the section looks
# empty.  The contour field is deliberately left per degree and unchanged.
SHADE_BAND = 2.5                  # native NCEP/NCAR latitude spacing, degrees

# shading scale of panel 1 (10^24 kg m^2 s^-1 per 2.5-degree latitude box)
FIELD_UNIT = 1.0e24
# shading levels: white inside ±0.5 and half-unit steps beyond — the classic AAM
# scale whose ticks are ±0.5, ±1.0, ±1.5, ±2.0
FIELD_LEVELS = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5])
# possible contour intervals (x FIELD_UNIT) of the total field.  The list spans
# two decades so that both a quiet chart and a strong one get a readable number
# of lines: the largest round interval that still gives three or more is used and
# printed in the caption.
CI_CHOICES = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

# AAM cache of the latitude-resolved daily climatology.
#
# It lives in the project root, *not* in the git-ignored ``data_cache/``: Render
# (and any container host) throws ``data_cache/`` away on every deploy, restart
# and free-tier spin-down, so a copy kept there means the 366 x 17-level build
# runs again inside the next request — which is what used to push a 512 MB free
# instance into an OOM kill.  ``gwo.LTM_CACHE_FILE`` is the same pattern for the
# GWO product: build it once (deploy-time via ``custom/_warm_cache.py``, or by
# the first request on a long-lived host), then reuse it forever.
LTM_CACHE = os.path.join(config.BASE_DIR, "ltm_rel_aam_cache.npy")

# bump when the stored climatology must be rebuilt (grid, strides, scale)
LTM_CACHE_VERSION = 4

# palette — the classic AAM colouring of the reference chart: blue / purple /
# green for negative anomalies, white / yellow / orange / red for positive
C_FACE = "#ffffff"
C_LINE = "#8d8878"
C_TEXT = "#111110"
C_GRID = "#d9d5c9"
C_SERIES = "#1d1d18"
# one colour per bounded interval of FIELD_LEVELS, in the reference's order:
# blue / purple / green / light green — white — pale yellow / yellow / orange / red
SHADE_COLOURS = ["#4a58c8",   # -2.5 … -2.0
                 "#8e7ad0",   # -2.0 … -1.5
                 "#4fa04a",   # -1.5 … -1.0
                 "#aadf95",   # -1.0 … -0.5
                 "#ffffff",   # -0.5 …  0.5   (neutral band, left unfilled)
                 "#fdf0b0",   #  0.5 …  1.0
                 "#ffd63f",   #  1.0 …  1.5
                 "#f7952c",   #  1.5 …  2.0
                 "#e4592a"]   #  2.0 …  2.5
SHADE_UNDER = "#2b3d97"       # beyond -2.5
SHADE_OVER = "#b7331c"        # beyond  2.5

# ===========================================================================
# 1.  the vertically + zonally integrated field, per radian of latitude
# ===========================================================================
def lat_field(u_all, gc):
    """(T, nlat) vertically and zonally integrated AAM per radian of latitude.

    Same quadrature as ``gwo.compute_aam_batch`` (which is the latitude
    integral of this field), so the two can never drift apart.
    """
    u_s = u_all[:, gc["lev_order"], :, :]
    i_p = -_gwo._trapz_axis1(u_s, gc["lev_pa"])                      # (T, lat, lon)

    i_p_sorted = i_p[:, :, gc["lon_order"]]
    i_p_ext = np.concatenate([i_p_sorted, i_p_sorted[:, :, :1]], axis=2)
    i_lon = _gwo._trapz_axis1(i_p_ext.transpose(0, 2, 1), gc["lon_ext"])  # (T, lat)

    return gc["scale"] * i_lon[:, gc["lat_order"]] * gc["cos2_lat"]


def integrate_field(latf, gc):
    """(T,) global relative AAM = latitude integral of :func:`lat_field`."""
    return _gwo._trapz_axis1(latf, gc["lat_sorted"])


# ===========================================================================
# 2.  latitude-resolved LTM climatology (1991–2020), cached on disk
# ===========================================================================
def _fetch_ltm_chunk(args):
    """Worker: integrate one LTM time chunk into the latitude field."""
    (url, t0, t1, lev_idx, gc, attrs, chunk_id, total,
     retries, retry_wait) = args
    for attempt in range(1, retries + 1):
        try:
            ds = open_remote(url)
            raw = ds["uwnd"].array[t0:t1, :, ::_gwo.LAT_STRIDE, ::_gwo.LON_STRIDE]
            chunk = _gwo._unpack(raw, attrs)
            if chunk.ndim == 3:
                chunk = chunk[np.newaxis]
            field = lat_field(chunk[:, lev_idx, :, :], gc)
            print(f"  [rel_aam] LTM chunk {chunk_id}/{total}: DOY {t0 + 1}-{t1} "
                  f"({len(field)} days)", flush=True)
            return t0, field
        except Exception as exc:                        # noqa: BLE001
            if attempt < retries:
                print(f"  [rel_aam] LTM chunk {chunk_id} attempt {attempt} failed: "
                      f"{exc}; retrying in {retry_wait}s …", flush=True)
                time.sleep(retry_wait)
            else:
                raise RuntimeError(
                    f"LTM chunk [{t0}:{t1}] failed after {retries} attempts: {exc}")


def compute_ltm_climatology(live_lev, cache_file=None):
    """(clim_field, lat_axis, lev_common) of the 1991–2020 daily LTM.

    ``clim_field`` is (n_doy, nlat) in kg m^2 s^-1 per radian of latitude, rows in
    ascending latitude, so it can be subtracted from :func:`lat_field` row by row.
    """
    cache_file = LTM_CACHE if cache_file is None else cache_file
    live_lev = np.asarray(live_lev, dtype=float)
    # Open the LTM file first (a no-op when it is already downloaded): its size
    # goes into the cache key, so a genuinely different LTM file invalidates the
    # cached climatology by itself.  The source URL is part of the key too, so a
    # mirror can never be mistaken for NOAA.
    ds = open_remote(_gwo.LTM_URL)
    # The key deliberately does *not* contain the file's mtime.  NOAA re-serves
    # the same immutable 1991-2020 file with a fresh mtime every time a container
    # is rebuilt, so an mtime-based key would discard the cached climatology on
    # every deploy — exactly the rebuild we are avoiding.  Size + grid + levels +
    # source URL is the identity that matters: a different period, grid or server
    # still invalidates the cache, a re-download of the same file does not.
    try:
        size_ltm = int(os.stat(ds.path).st_size)
    except OSError:                                             # pragma: no cover
        size_ltm = 0
    key = (LTM_CACHE_VERSION, _gwo.LAT_STRIDE, _gwo.LON_STRIDE, size_ltm,
           tuple(np.sort(np.round(live_lev).astype(int))), config.PSL)
    if os.path.exists(cache_file):
        try:
            cached = np.load(cache_file, allow_pickle=True).item()
        except Exception:                                       # noqa: BLE001
            cached = None
        if (cached is not None and cached.get("key") == key
                and np.asarray(cached.get("clim_field")).ndim == 2):
            print(f"[rel_aam] LTM climatology from cache: {cache_file}", flush=True)
            return cached["clim_field"], cached["lat_axis"], cached["lev_common"]

    print("[rel_aam] computing the latitude-resolved LTM climatology …", flush=True)
    attrs = ds["uwnd"].attributes
    lev_ltm = _gwo._to_numpy(ds["level"])
    lat = _gwo._to_numpy(ds["lat"])[::_gwo.LAT_STRIDE]
    lon = _gwo._to_numpy(ds["lon"])[::_gwo.LON_STRIDE]

    common = np.intersect1d(np.round(lev_ltm).astype(int),
                            np.round(live_lev).astype(int))
    if len(common) == 0:
        raise RuntimeError("No common pressure levels between the LTM and the live data.")
    lev_idx = np.array([int(np.where(np.round(lev_ltm).astype(int) == p)[0][0])
                        for p in common])
    lev_common = lev_ltm[lev_idx]

    time_var = ds.get("time", None)
    ntime = len(_gwo._to_numpy(time_var)) if time_var is not None else 365
    gc = _gwo.build_grid_constants(lev_common, lat, lon)
    print(f"[rel_aam] LTM: {ntime} days × {len(lev_common)} levels × "
          f"{len(lat)} lat × {len(lon)} lon", flush=True)

    edges = np.linspace(0, ntime, _gwo.LTM_CHUNKS + 1, dtype=int)
    tasks = [(_gwo.LTM_URL, int(edges[k]), int(edges[k + 1]), lev_idx, gc, attrs,
              k + 1, _gwo.LTM_CHUNKS, _gwo.LTM_RETRIES, _gwo.LTM_RETRY_WAIT)
             for k in range(_gwo.LTM_CHUNKS)]

    results = {}
    with ThreadPoolExecutor(max_workers=_gwo.LTM_MAX_WORKERS) as pool:
        futures = [pool.submit(_fetch_ltm_chunk, t) for t in tasks]
        for fut in as_completed(futures):
            t0_idx, field = fut.result()
            results[t0_idx] = field

    # the chunk arrays are dead now; hand their pages back before building the
    # (tiny) climatology so a 512 MB instance never sees the peak twice
    release_memory()

    clim_field = np.concatenate([results[k] for k in sorted(results)], axis=0)
    lat_axis = np.rad2deg(gc["lat_sorted"])            # degrees, ascending
    total = _gwo._trapz_axis1(clim_field, gc["lat_sorted"])
    print(f"[rel_aam] LTM Mr range {total.min():.4e} – {total.max():.4e} kg m² s⁻¹ "
          f"(mean {total.mean():.4e})", flush=True)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, {"clim_field": clim_field, "lat_axis": lat_axis,
                         "lev_common": lev_common, "key": key})
    print(f"[rel_aam] LTM climatology cached → {cache_file}", flush=True)
    release_memory()
    return clim_field, lat_axis, lev_common


# ===========================================================================
# 3.  small helpers
# ===========================================================================
def smooth_time(mat, n):
    """Running mean along axis 0 (time).

    Each row is convolved with a boxcar and divided by the actual weights, so a
    constant field comes back unchanged and the edges are the mean of the days
    they really have.  Everything is linear, so smoothing the field and then
    integrating gives exactly the same global series as integrating first.
    """
    mat = np.asarray(mat, dtype=float)
    n = max(1, int(n))
    if n <= 1 or mat.shape[0] < 3:
        return mat
    ker = np.ones(n) / n
    den = np.convolve(np.ones(mat.shape[0]), ker, mode="same")
    out = np.empty_like(mat)
    for k in range(mat.shape[1]):
        out[:, k] = np.convolve(mat[:, k], ker, mode="same") / den
    return out


def cyclic_running_mean(x, n):
    """Running mean that wraps around the year (for the daily climatology)."""
    x = np.asarray(x, dtype=float)
    n = max(1, int(n))
    if n == 1 or len(x) < 3:
        return x
    pad = n // 2
    ext = np.concatenate([x[-pad:], x, x[:pad]])
    return np.convolve(ext, np.ones(n) / n, mode="same")[pad:pad + len(x)]


def align_lat(target_deg, field, source_deg):
    """Interpolate a (n, nlat_src) field onto ``target_deg`` if the axes differ."""
    target_deg = np.asarray(target_deg, dtype=float)
    source_deg = np.asarray(source_deg, dtype=float)
    if (source_deg.shape == target_deg.shape
            and np.allclose(source_deg, target_deg)):
        return field
    return np.stack([np.interp(target_deg, source_deg, row) for row in field], axis=0)


def nice_ci(peak):
    """A round contour interval (x FIELD_UNIT) giving a readable number of lines.

    The largest round interval that still yields three or more contour lines is
    used, so a quiet chart is not buried under lines and a busy one is not empty.
    """
    for ci in CI_CHOICES:
        if peak / ci >= 3.0:
            return ci
    for ci in reversed(CI_CHOICES):
        if peak / ci >= 1.0:
            return ci
    return CI_CHOICES[-1]


# ===========================================================================
# 4.  compute
# ===========================================================================
def _compute_rel_aam(pkg, dates):
    """Fetch the window, integrate the field, subtract the LTM climatology."""
    window = int(pkg.get("window", WINDOW_DAYS))
    smooth = int(pkg.get("smooth_days", SMOOTH_DAYS))
    end_date = dates[-1] if dates else None

    times, lev, lat, lon, uwnd = _gwo.fetch_recent_days(window, end_date=end_date)
    clim_field, clim_lat, lev_common = compute_ltm_climatology(lev)

    gc = _gwo.build_grid_constants(lev, lat, lon)
    lat_axis = np.rad2deg(gc["lat_sorted"])                 # degrees, ascending
    field = lat_field(uwnd, gc)                             # (T, nlat) per radian

    # daily climatology sampled on the day of the year, smoothed cyclically
    clim_smooth = np.stack([cyclic_running_mean(clim_field[:, k], CLIM_SMOOTH)
                            for k in range(clim_field.shape[1])], axis=1)
    clim_smooth = align_lat(lat_axis, clim_smooth, clim_lat)
    doys = np.array([t.timetuple().tm_yday for t in times])
    idx = np.clip(doys - 1, 0, clim_smooth.shape[0] - 1)
    clim = clim_smooth[idx]                                 # (T, nlat)

    field_s = smooth_time(field, smooth)
    anom = field_s - clim
    mr = integrate_field(field_s, gc)
    mr_clim = clim.sum(axis=1) * 0.0                        # placeholder, replaced below
    mr_clim = integrate_field(clim, gc)
    anom_tot = mr - mr_clim
    M = anom_tot / _gwo.AAM_STD
    dMdt = np.gradient(mr, 86400.0) / _gwo.TEND_STD

    data = {
        "main": np.array([[mr[-1] / HU]], dtype=float),
        "times": times, "lat_axis": lat_axis, "lon": np.asarray(lon, float),
        "lev": np.asarray(lev, float),
        "lev_common": np.asarray(lev_common, float),
        "field": field_s, "clim": clim, "anom": anom,
        "mr": mr, "mr_clim": mr_clim, "anom_tot": anom_tot,
        "M": M, "dMdt": dMdt, "smooth_days": smooth,
    }
    return lat, lon, data


# ===========================================================================
# 5.  render
# ===========================================================================
def _aam_cmap():
    """One colour per shading interval, in the same order as the levels."""
    n = len(FIELD_LEVELS) - 1
    assert n == len(SHADE_COLOURS), (n, len(SHADE_COLOURS))
    cmap = LinearSegmentedColormap.from_list("xp_aam", SHADE_COLOURS, N=n)
    cmap.set_under(SHADE_UNDER)
    cmap.set_over(SHADE_OVER)
    return cmap


def _frame(ax, label_left=True, label_bottom=True):
    """Thin frame with inward ticks on all four sides (classic chart style)."""
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_edgecolor("#111110")
        side.set_linewidth(0.9)
    ax.tick_params(direction="in", top=True, right=True, length=3.8, width=0.8,
                   labelsize=8.5, colors="#111110", pad=3)
    ax.tick_params(labelleft=label_left, labelbottom=label_bottom)


def _render_rel_aam(lat, lon, data, pkg, coast_segs, dates, out_buf=None, **_kw):
    times = data["times"]
    lat_axis = np.asarray(data["lat_axis"], float)
    field = np.asarray(data["field"], float) * DEG          # per degree latitude
    anom = np.asarray(data["anom"], float) * DEG * SHADE_BAND   # per 2.5-deg box
    anom_tot = np.asarray(data["anom_tot"], float)
    smooth = int(data.get("smooth_days", SMOOTH_DAYS))

    t = mdates.date2num(times)
    t_lo, t_hi = t[0] - 1.0, t[-1] + 1.0
    peak_ci = float(np.nanmax(np.abs(field)) / FIELD_UNIT)
    ci = nice_ci(peak_ci)

    tt2d, la2d = np.meshgrid(t, lat_axis, indexing="ij")

    # square figure; panel proportions follow the reference chart
    fig = plt.figure(figsize=(6.8, 6.8), facecolor="white")
    ax_left, ax_w = 0.080, 0.828
    ax_right = ax_left + ax_w

    # ---- caption, four centred lines as on the reference ------------------
    fig.text(0.5, 0.982, "VERTICAL AND ZONAL INTEGRAL OF", ha="center", va="top",
             fontsize=13.5, fontweight="bold", color="#111110")
    fig.text(0.5, 0.949, "RELATIVE ATMOSPHERIC ANGULAR MOMENTUM", ha="center",
             va="top", fontsize=13.5, fontweight="bold", color="#111110")
    fig.text(0.5, 0.913,
             f"FROM NCEP/NCAR REANALYSIS U WIND, {smooth}-DAY RUNNING MEAN",
             ha="center", va="top", fontsize=9, color="#22221c")
    fig.text(0.5, 0.888,
             f"TOTAL FIELD CONTOURS (CI {ci:g} × 10²⁴), "
             f"ANOMALIES SHADED (SCALED BY 10²⁴)",
             ha="center", va="top", fontsize=9, color="#22221c")

    # ---- panel 1 : latitude–time section ---------------------------------
    ax1 = fig.add_axes([ax_left, 0.469, ax_w, 0.397])
    ax1.set_facecolor("white")
    cf = ax1.contourf(tt2d, la2d, np.nan_to_num(anom / FIELD_UNIT, nan=0.0),
                      levels=FIELD_LEVELS, cmap=_aam_cmap(), extend="both", zorder=2)
    levels = np.arange(ci, peak_ci + ci * 0.5, ci)
    if levels.size:
        ax1.contour(tt2d, la2d, field / FIELD_UNIT, levels=levels, colors="#111110",
                    linewidths=0.75, linestyles="solid", zorder=4)
        ax1.contour(tt2d, la2d, field / FIELD_UNIT, levels=-levels[::-1],
                    colors="#111110", linewidths=0.75, linestyles="dashed", zorder=4)
    ax1.axhline(0.0, color="#44443c", lw=0.8, zorder=5)          # equator

    ax1.set_xlim(t_lo, t_hi)
    ax1.set_ylim(-90, 90)
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax1.set_yticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])
    _frame(ax1, label_bottom=False)

    # vertical colorbar on the right, no label, as on the reference
    cax = fig.add_axes([0.918, 0.519, 0.022, 0.287])
    ticks = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
    cbar = fig.colorbar(cf, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels([f"{v:.1f}" for v in ticks])
    cbar.ax.tick_params(direction="in", length=2.6, width=0.7, labelsize=8.5,
                        colors="#111110", pad=3)
    cbar.outline.set_edgecolor("#111110")
    cbar.outline.set_linewidth(0.9)

    # ---- panel 2 : global anomaly ----------------------------------------
    fig.text(0.5, 0.428, "GLOBAL RELATIVE AAM ANOMALY   (SCALED BY 10²⁵ kg m² s⁻¹)",
             ha="center", va="bottom", fontsize=9.5, color="#22221c")
    ax2 = fig.add_axes([ax_left, 0.150, ax_w, 0.253])
    ax2.set_facecolor("white")
    y = anom_tot / HU
    lim = max(1.0, float(np.ceil(np.nanmax(np.abs(y)))))
    ax2.set_ylim(-lim, lim)
    ax2.yaxis.set_major_locator(MultipleLocator(1.0))
    ax2.axhline(0.0, color="#111110", lw=0.8, zorder=3)
    ax2.plot(t, y, color="#111110", lw=1.2, zorder=5)

    ax2.set_xlim(t_lo, t_hi)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.canvas.draw()                                   # ensure tick labels exist
    for lbl in ax2.get_xticklabels():
        lbl.set_text(lbl.get_text().upper())
    _frame(ax2)
    ax2.text(0.5, -0.175, f"{times[-1]:%Y}", transform=ax2.transAxes, ha="center",
             va="top", fontsize=9.5, color="#111110")

    # ---- footer, as on the reference --------------------------------------
    fig.text(ax_left, 0.062, "BASE PERIOD   1991–2020", ha="left", va="bottom",
             fontsize=10, color="#111110")
    fig.text(ax_right, 0.062, f"LAST DAY USED   {times[-1]:%d/%m/%Y}", ha="right",
             va="bottom", fontsize=10, color="#111110")
    fig.text(0.5, 0.012, "@XPWEATHER", ha="center", va="bottom", fontsize=8,
             color="#8f8d82")

    if out_buf is None:
        out_buf = io.BytesIO()
    plt.savefig(out_buf, format="png", dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    out_buf.seek(0)
    return out_buf


def _render_locked(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return _render_rel_aam(lat, lon, data, pkg, coast_segs, dates, **_kw)


# ===========================================================================
# 6.  registry
# ===========================================================================
PRODUCTS = {
    "rel_aam": {
        "id": "rel_aam",
        "tag": "Hovmöller",
        "title": "Relative Angular Momentum — latitude–time and global anomaly",
        "name": "Relative AAM",
        "desc": "Latitude–time section of the vertically and zonally integrated "
                "relative AAM anomaly (shaded, 1000–10 hPa) with the total field "
                "as contours, plus the global AAM anomaly — last 120 days.",
        "kind": "rel_aam",
        "level": None,
        "window": WINDOW_DAYS,
        "smooth_days": SMOOTH_DAYS,
        "cb_label": "Relative AAM anomaly  (10²⁴ kg m² s⁻¹ per degree)",
    },
}

KINDS = {
    "rel_aam": {
        "compute": _compute_rel_aam,
        "render": _render_locked,
        "tag": "Hovmöller",
        "title": "Relative Angular Momentum — latitude–time and global anomaly",
    },
}


if __name__ == "__main__":                       # local smoke test
    from pro import engine
    buf, meta = engine.generate(product_id="rel_aam", mode="auto",
                                n_days=config.DEFAULT_N_DAYS)
    out = os.path.join(_ROOT, "rel_aam.png")
    with open(out, "wb") as fh:
        fh.write(buf.getvalue())
    print("wrote", out, meta)
