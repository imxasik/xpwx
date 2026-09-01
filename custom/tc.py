"""custom/tc.py — TC Related diagnostic products.

Products (each available per-region via the sidebar domain selector):
  Chi200 Anomaly        — 200 hPa Velocity Potential Anomaly
  VWS Anomaly           — 850–200 hPa Vertical Wind Shear Anomaly
  VWS Total             — 850–200 hPa Total Vertical Wind Shear
  Wind Shear (Diff)     — 850–200 hPa Directional Shear (shear.py)
  Instability Anomaly   — 850–200 hPa Atmospheric Instability Anomaly
  Mid-level Wave Trend  — 700–500 hPa CCKW/MJO Wave Trend

Regions: Indian Ocean · Western Pacific · Central Pacific · Eastern Pacific · Atlantic · Global
"""

import io
import re
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from pydap.client import open_url
from pro import config   # BASE_DIR, SHP_PATH, PSL, coastline cache

PSL = config.PSL

# ── Region bounds ────────────────────────────────────────────────────────────
_REGION_BOUNDS = {
    "global": (0,   360, -75,  75),
    "io":     (40,  120, -30,  30),
    "wp":     (100, 180, -30,  30),
    "cp":     (160, 210, -30,  30),
    "ep":     (210, 280, -30,  30),
    "atl":    (290, 360, -30,  30),
}

_REGION_DISPLAY = {
    "global": "Global",
    "io":     "Indian Ocean",
    "wp":     "W. Pacific",
    "cp":     "C. Pacific",
    "ep":     "E. Pacific",
    "atl":    "Atlantic",
}

# ── Chi200 REGIONS (same as original chic.py) ────────────────────────────────
_CHI_REGION_BOUNDS = {
    "io":     {"lon": (30,  130), "lat": (-30, 30), "step": 3, "lon_step": 15, "lat_step": 10},
    "global": {"lon": (0,   360), "lat": (-90, 90), "step": 6, "lon_step": 60, "lat_step": 20},
    "atl":    {"lon": (280, 360), "lat": (0,   60), "step": 3, "lon_step": 20, "lat_step": 10},
    "pac":    {"lon": (100, 290), "lat": (-30, 30), "step": 3, "lon_step": 20, "lat_step": 10},
    "wp":     {"lon": (100, 180), "lat": (-30, 30), "step": 3, "lon_step": 20, "lat_step": 10},
}


# ===========================================================================
# Shared low-level helpers (all stateless, take explicit region/dates args)
# ===========================================================================

def _get_bounds(region):
    return _REGION_BOUNDS.get(region.lower(), _REGION_BOUNDS["global"])


def _open_ds(varname, year):
    url = f"{PSL}/{varname}.{year}.nc"
    return open_url(url)


def _open_ltm(varname):
    url = f"{PSL}/{varname}.day.ltm.1991-2020.nc"
    return open_url(url)


def _latlon(ds):
    return np.array(ds["lat"][:]), np.array(ds["lon"][:])


def _level_idx(ds, hPa):
    lev = np.array(ds["level"][:])
    return int(np.argmin(np.abs(lev - hPa)))


def _time_idx(ds, target):
    raw   = np.array(ds["time"][:])
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    scale = 1.0 / 24.0 if "hours" in units else 1.0
    m     = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
    epoch = (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
             if m else datetime.date(1800, 1, 1))
    for i, t in enumerate(raw):
        d = epoch + datetime.timedelta(days=float(t) * scale)
        if d.year == target.year and d.month == target.month and d.day == target.day:
            return i
    raise ValueError(f"Date {target} not found in dataset")


def _spatial_indices(lat, lon, lon_min, lon_max, lat_min, lat_max):
    lat_idx = np.where((lat >= lat_min - 2.0) & (lat <= lat_max + 2.0))[0]
    lon_idx = np.where((lon >= lon_min - 2.0) & (lon <= lon_max + 2.0))[0]
    if len(lat_idx) == 0:
        lat_idx = np.arange(len(lat))
    if len(lon_idx) == 0:
        lon_idx = np.arange(len(lon))
    return lat_idx, lon_idx


def _read_slice_lev(ds, varname, t, lv, lat_idx, lon_idx):
    lat_s = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    lon_s = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    raw   = np.array(ds[varname][t, lv, lat_s, lon_s].data).squeeze().astype(np.float64)
    attr  = ds[varname].attributes
    sf    = float(attr.get("scale_factor",  1.0))
    ao    = float(attr.get("add_offset",    0.0))
    mv    = float(attr.get("missing_value", 32767.0))
    data  = raw * sf + ao
    data[np.abs(raw - mv) < 0.5] = np.nan
    return data


def _read_slice_sfc(ds, varname, t, lat_idx, lon_idx):
    lat_s = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    lon_s = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    raw   = np.array(ds[varname][t, lat_s, lon_s].data).squeeze().astype(np.float64)
    attr  = ds[varname].attributes
    sf    = float(attr.get("scale_factor",  1.0))
    ao    = float(attr.get("add_offset",    0.0))
    mv    = float(attr.get("missing_value", 32767.0))
    data  = raw * sf + ao
    data[np.abs(raw - mv) < 0.5] = np.nan
    return data


# ── fetch helpers (take dates, region bounds explicitly) ─────────────────────

def _fetch_uv_obs(hPa, dates, lon_min, lon_max, lat_min, lat_max):
    by_year = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)
    lat = lon = lat_idx = lon_idx = None
    u_slices, v_slices = [], []
    for year, ydates in sorted(by_year.items()):
        ds_u = _open_ds("uwnd", year)
        ds_v = _open_ds("vwnd", year)
        if lat is None:
            full_lat, full_lon = _latlon(ds_u)
            lat_idx, lon_idx = _spatial_indices(full_lat, full_lon,
                                                lon_min, lon_max, lat_min, lat_max)
            lat = full_lat[lat_idx]
            lon = full_lon[lon_idx]
        lv_u = _level_idx(ds_u, hPa)
        lv_v = _level_idx(ds_v, hPa)
        for d in ydates:
            ti_u = _time_idx(ds_u, d)
            ti_v = _time_idx(ds_v, d)
            u_slices.append(_read_slice_lev(ds_u, "uwnd", ti_u, lv_u, lat_idx, lon_idx))
            v_slices.append(_read_slice_lev(ds_v, "vwnd", ti_v, lv_v, lat_idx, lon_idx))
    return lat, lon, np.nanmean(u_slices, axis=0), np.nanmean(v_slices, axis=0)


def _fetch_uv_ltm(hPa, dates, lon_min, lon_max, lat_min, lat_max):
    ds_u = _open_ltm("uwnd")
    ds_v = _open_ltm("vwnd")
    full_lat, full_lon = _latlon(ds_u)
    lat_idx, lon_idx = _spatial_indices(full_lat, full_lon,
                                        lon_min, lon_max, lat_min, lat_max)
    lat = full_lat[lat_idx]
    lon = full_lon[lon_idx]
    lv_u = _level_idx(ds_u, hPa)
    lv_v = _level_idx(ds_v, hPa)
    n_u  = len(np.array(ds_u["time"][:]))
    n_v  = len(np.array(ds_v["time"][:]))
    u_slices, v_slices = [], []
    for d in dates:
        doy  = d.timetuple().tm_yday
        ti_u = min(doy - 1, n_u - 1)
        ti_v = min(doy - 1, n_v - 1)
        u_slices.append(_read_slice_lev(ds_u, "uwnd", ti_u, lv_u, lat_idx, lon_idx))
        v_slices.append(_read_slice_lev(ds_v, "vwnd", ti_v, lv_v, lat_idx, lon_idx))
    return lat, lon, np.nanmean(u_slices, axis=0), np.nanmean(v_slices, axis=0)


def _fetch_air_obs(hPa, dates, lon_min, lon_max, lat_min, lat_max):
    by_year = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)
    lat = lon = lat_idx = lon_idx = None
    slices = []
    for year, ydates in sorted(by_year.items()):
        ds = _open_ds("air", year)
        if lat is None:
            full_lat, full_lon = _latlon(ds)
            lat_idx, lon_idx = _spatial_indices(full_lat, full_lon,
                                                lon_min, lon_max, lat_min, lat_max)
            lat = full_lat[lat_idx]
            lon = full_lon[lon_idx]
        lv = _level_idx(ds, hPa)
        for d in ydates:
            ti = _time_idx(ds, d)
            slices.append(_read_slice_lev(ds, "air", ti, lv, lat_idx, lon_idx))
    return lat, lon, np.nanmean(slices, axis=0)


def _fetch_air_ltm(hPa, dates, lon_min, lon_max, lat_min, lat_max):
    ds = _open_ltm("air")
    full_lat, full_lon = _latlon(ds)
    lat_idx, lon_idx = _spatial_indices(full_lat, full_lon,
                                        lon_min, lon_max, lat_min, lat_max)
    lat = full_lat[lat_idx]
    lon = full_lon[lon_idx]
    lv  = _level_idx(ds, hPa)
    n   = len(np.array(ds["time"][:]))
    slices = []
    for d in dates:
        doy = d.timetuple().tm_yday
        slices.append(_read_slice_lev(ds, "air", min(doy - 1, n - 1), lv, lat_idx, lon_idx))
    return lat, lon, np.nanmean(slices, axis=0)


# ===========================================================================
# Shared map drawing helpers
# ===========================================================================

def _fig_layout(is_global):
    if is_global:
        return (
            plt.figure(figsize=(12, 7), facecolor="white"),
            [0.045, 0.145, 0.910, 0.785],
            [0.12, 0.057, 0.760, 0.028],
            -1.55, 0.985, False,
        )
    else:
        return (
            plt.figure(figsize=(10, 9), facecolor="white"),
            [0.10, 0.16, 0.82, 0.73],
            [0.15, 0.08, 0.70, 0.025],
            -1.5, 0.920, True,
        )


def _draw_coastlines(ax, coast_segs):
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)


def _draw_grid_ticks(ax, lon_min, lon_max, lat_min, lat_max):
    lon_interval = 10 if (lon_max - lon_min) <= 90 else 30
    lat_interval = 10 if (lat_max - lat_min) <= 90 else 20
    for x in range(int(lon_min), int(lon_max) + 1, lon_interval):
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in range(int(lat_min), int(lat_max) + 1, lat_interval):
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.75, zorder=0, alpha=0.8)

    def xlab(v):
        v = v % 360
        if v == 0:   return "0°"
        if v == 180: return "180°"
        if v < 180:  return f"{v}°E"
        return f"{360-v}°W"
    def ylab(v):
        return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

    xticks = list(range(int(lon_min), int(lon_max) + 1, lon_interval))
    yticks = list(range(int(lat_min), int(lat_max) + 1, lat_interval))
    ax.set_xticks(xticks)
    ax.set_xticklabels([xlab(x) for x in xticks], fontsize=9.5, color="#333322")
    ax.set_yticks(yticks)
    ax.set_yticklabels([ylab(y) for y in yticks], fontsize=9.5, color="#333322")
    ax.tick_params(axis="both", length=3.5, color="#888878", width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988")
        spine.set_linewidth(0.8)


def _watermark(ax):
    ax.text(0.985, 0.016, "@XPWEATHER",
            transform=ax.transAxes, fontsize=11, va="bottom", ha="right",
            color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccccbb", alpha=0.92, lw=0.9),
            zorder=10)
    ax.text(0.005, 0.016, "NCEP/NCAR Reanalysis  ·  PSL/NOAA",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            color="#666655", zorder=10)


# ===========================================================================
# ── 1. CHI200 Anomaly ────────────────────────────────────────────────────────
# ===========================================================================

def _divergence(u, v, lat, lon):
    R      = 6.371e6
    lat_r  = np.deg2rad(lat)
    lon_r  = np.deg2rad(lon)
    coslat = np.cos(lat_r)
    dudx   = np.gradient(u, lon_r, axis=1) / (R * coslat[:, None])
    vcoslat = v * coslat[:, None]
    dvdy   = np.gradient(vcoslat, lat_r, axis=0) / (R * coslat[:, None])
    return dudx + dvdy


def _poisson_fft(rhs, lat, lon):
    R       = 6.371e6
    lat_r   = np.deg2rad(lat)
    lon_r   = np.deg2rad(lon)
    dy      = R * np.abs(np.mean(np.diff(lat_r)))
    coslat  = np.cos(lat_r)
    dx_mean = R * np.mean(np.diff(lon_r)) * np.mean(np.abs(coslat))
    nlat, nlon = rhs.shape
    rhs_clean  = np.nan_to_num(rhs, nan=0.0)
    taper = np.ones(nlat)
    for i, la in enumerate(lat):
        if abs(la) > 75.0:
            taper[i] = np.cos(np.deg2rad((abs(la) - 75.0) * 90.0 / 15.0)) ** 2
    rhs_clean *= taper[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(nlon, d=dx_mean)
    ky = 2.0 * np.pi * np.fft.fftfreq(nlat, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    F = np.fft.fft2(rhs_clean)
    F /= -K2
    F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F))


def _chi_to_wind(chi, lat, lon):
    R      = 6.371e6
    lat_r  = np.deg2rad(lat)
    lon_r  = np.deg2rad(lon)
    coslat = np.cos(lat_r)
    u = np.gradient(chi, lon_r, axis=1) / (R * coslat[:, None])
    v = np.gradient(chi, lat_r, axis=0) / R
    return u, v


def _fetch_uv_obs_global(dates):
    """Fetch full-globe u/v at 200 hPa (for chi200)."""
    by_year = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)
    lat = lon = None
    u_slices, v_slices = [], []
    for year, ydates in sorted(by_year.items()):
        ds_u = _open_ds("uwnd", year)
        ds_v = _open_ds("vwnd", year)
        if lat is None:
            lat, lon = _latlon(ds_u)
        lv_u = _level_idx(ds_u, 200)
        lv_v = _level_idx(ds_v, 200)
        for d in ydates:
            ti_u = _time_idx(ds_u, d)
            ti_v = _time_idx(ds_v, d)
            u_slices.append(np.array(ds_u["uwnd"][ti_u, lv_u, :, :].data).squeeze().astype(np.float64))
            v_slices.append(np.array(ds_v["vwnd"][ti_v, lv_v, :, :].data).squeeze().astype(np.float64))
    # apply scale/offset
    def _scale(ds, var, arr):
        attr = ds[var].attributes
        sf = float(attr.get("scale_factor", 1.0))
        ao = float(attr.get("add_offset",   0.0))
        mv = float(attr.get("missing_value", 32767.0))
        out = arr * sf + ao
        out[np.abs(arr - mv) < 0.5] = np.nan
        return out
    # re-open to get attrs (already cached by pydap)
    ds_u0 = _open_ds("uwnd", list(by_year.keys())[0])
    ds_v0 = _open_ds("vwnd", list(by_year.keys())[0])
    u_mean = np.nanmean([_scale(ds_u0, "uwnd", s) for s in u_slices], axis=0)
    v_mean = np.nanmean([_scale(ds_v0, "vwnd", s) for s in v_slices], axis=0)
    return lat, lon, u_mean, v_mean


def _fetch_uv_ltm_global(dates):
    ds_u = _open_ltm("uwnd")
    ds_v = _open_ltm("vwnd")
    lat, lon = _latlon(ds_u)
    lv_u = _level_idx(ds_u, 200)
    lv_v = _level_idx(ds_v, 200)
    n_u  = len(np.array(ds_u["time"][:]))
    n_v  = len(np.array(ds_v["time"][:]))
    u_slices, v_slices = [], []
    for d in dates:
        doy  = d.timetuple().tm_yday
        ti_u = min(doy - 1, n_u - 1)
        ti_v = min(doy - 1, n_v - 1)
        raw_u = np.array(ds_u["uwnd"][ti_u, lv_u, :, :].data).squeeze().astype(np.float64)
        raw_v = np.array(ds_v["vwnd"][ti_v, lv_v, :, :].data).squeeze().astype(np.float64)
        def _sc(ds, var, raw):
            attr = ds[var].attributes
            sf = float(attr.get("scale_factor", 1.0))
            ao = float(attr.get("add_offset",   0.0))
            mv = float(attr.get("missing_value", 32767.0))
            out = raw * sf + ao
            out[np.abs(raw - mv) < 0.5] = np.nan
            return out
        u_slices.append(_sc(ds_u, "uwnd", raw_u))
        v_slices.append(_sc(ds_v, "vwnd", raw_v))
    return lat, lon, np.nanmean(u_slices, axis=0), np.nanmean(v_slices, axis=0)


def _compute_chi(pkg, dates):
    region = pkg.get("region", "io")
    reg    = _CHI_REGION_BOUNDS.get(region, _CHI_REGION_BOUNDS["io"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_obs = pool.submit(_fetch_uv_obs_global, dates)
        fut_ltm = pool.submit(_fetch_uv_ltm_global, dates)
        lat, lon, u_obs, v_obs = fut_obs.result()
        _,   _,   u_ltm, v_ltm = fut_ltm.result()

    u_anom = gaussian_filter(u_obs - u_ltm, sigma=1.5)
    v_anom = gaussian_filter(v_obs - v_ltm, sigma=1.5)
    div    = _divergence(u_anom, v_anom, lat, lon)
    chi    = gaussian_filter(_poisson_fft(div, lat, lon), sigma=2.0)
    u_div, v_div = _chi_to_wind(chi, lat, lon)

    return lat, lon, {"main": chi, "u": u_div, "v": v_div,
                      "_reg": reg, "_region": region}


def _render_chi(lat, lon, data, pkg, coast_segs, dates):
    chi    = data["main"]
    u_div  = data["u"]
    v_div  = data["v"]
    reg    = data["_reg"]
    region = data["_region"]

    lon_min, lon_max = reg["lon"]
    lat_min, lat_max = reg["lat"]
    LON2D, LAT2D = np.meshgrid(lon, lat)
    chi_plot = chi * 1e-6

    date_start = dates[0]
    date_end   = dates[-1]

    fig = plt.figure(figsize=(11, 7), facecolor="white")
    ax  = fig.add_axes([0.08, 0.1, 0.8, 0.81])
    ax.set_facecolor("#eef3f9")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    vlim   = 10
    levels = np.linspace(-vlim, vlim, 97)
    cf = ax.contourf(LON2D, LAT2D, chi_plot, levels=levels,
                     cmap="RdBu_r", extend="both", zorder=1)

    pos_lev = np.arange(5, vlim + 1, 5)
    ax.contour(LON2D, LAT2D, chi_plot, levels=pos_lev,
               colors="black", linewidths=0.9, zorder=2)
    ax.contour(LON2D, LAT2D, chi_plot, levels=-pos_lev[::-1],
               colors="black", linewidths=0.9, linestyles="dashed", zorder=2)

    step = reg["step"]
    qs   = slice(None, None, step)
    Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
    Uq, Vq = u_div[qs, qs], v_div[qs, qs]
    mag  = np.sqrt(Uq**2 + Vq**2)
    mask = (mag > 1e-10) & ~np.isnan(mag)
    ax.quiver(Xq[mask], Yq[mask],
              Uq[mask] / mag[mask], Vq[mask] / mag[mask],
              color="black", scale=4, scale_units="inches",
              width=0.0035, headwidth=6, headlength=7,
              headaxislength=5, minshaft=1.5, pivot="middle", zorder=6)

    _draw_coastlines(ax, coast_segs)

    xticks = list(range(lon_min, lon_max + 1, reg["lon_step"]))
    yticks = list(range(lat_min, lat_max + 1, reg["lat_step"]))
    for x in xticks:
        ax.axvline(x, color="#999999", lw=0.4, ls="--", zorder=0)
    for y in yticks:
        ax.axhline(y, color="#999999", lw=0.4, ls="--", zorder=0)
    ax.axhline(0, color="#444444", lw=0.9, zorder=0)

    def xlab(v):
        if v == 0 or v == 360: return "0°"
        return f"{v}°E" if v <= 180 else f"{360-v}°W"
    def ylab(v):
        return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

    ax.set_xticks(xticks)
    ax.set_xticklabels([xlab(x) for x in xticks], fontsize=11, color="#222222")
    ax.set_yticks(yticks)
    ax.set_yticklabels([ylab(y) for y in yticks], fontsize=11, color="#222222")
    ax.tick_params(axis="both", length=4, color="#888888", width=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.7)

    cax  = fig.add_axes([0.89, 0.1, 0.02, 0.81])
    cbar = plt.colorbar(cf, cax=cax, ticks=np.arange(-10, 11, 5))
    cbar.set_label("Chi200 Anomaly [M²S×10⁶]",
                   fontsize=12, color="#111111", labelpad=5)
    cbar.ax.tick_params(labelsize=12, colors="#222222", length=3)
    cbar.outline.set_edgecolor("#aaaaaa")
    cbar.outline.set_linewidth(0.7)

    rdisp = _REGION_DISPLAY.get(region, region.upper())
    ax.set_title(
        f"200 hPa Velocity Potential Anomaly ({rdisp})  ·  "
        f"{date_start:%d %b} – {date_end:%d %b %Y}  ({len(dates)}-day mean)\n"
        "NCEP/NCAR Reanalysis  ·  Shading: Velocity Potential  |  Arrows: Wind Direction",
        fontsize=13, fontweight="bold", color="#111111", pad=10, loc="center",
    )
    ax.text(0.830, 0.03, "@XPWEATHER",
            transform=ax.transAxes, fontsize=12, va="bottom", color="#111111",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbbbbb", alpha=1.0, lw=1.0),
            zorder=8)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# ===========================================================================
# ── 2. VWS Anomaly ──────────────────────────────────────────────────────────
# ===========================================================================

def _compute_vwsa(pkg, dates):
    region = pkg.get("region", "io")
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)

    with ThreadPoolExecutor(max_workers=4) as pool:
        f1 = pool.submit(_fetch_uv_obs, 850, dates, lon_min, lon_max, lat_min, lat_max)
        f2 = pool.submit(_fetch_uv_ltm, 850, dates, lon_min, lon_max, lat_min, lat_max)
        f3 = pool.submit(_fetch_uv_obs, 200, dates, lon_min, lon_max, lat_min, lat_max)
        f4 = pool.submit(_fetch_uv_ltm, 200, dates, lon_min, lon_max, lat_min, lat_max)
        lat, lon, u_obs_850, v_obs_850 = f1.result()
        _,   _,   u_ltm_850, v_ltm_850 = f2.result()
        _,   _,   u_obs_200, v_obs_200 = f3.result()
        _,   _,   u_ltm_200, v_ltm_200 = f4.result()

    u_a850 = gaussian_filter(u_obs_850 - u_ltm_850, sigma=1.5)
    v_a850 = gaussian_filter(v_obs_850 - v_ltm_850, sigma=1.5)
    u_a200 = gaussian_filter(u_obs_200 - u_ltm_200, sigma=1.5)
    v_a200 = gaussian_filter(v_obs_200 - v_ltm_200, sigma=1.5)
    u_shear = u_a200 - u_a850
    v_shear = v_a200 - v_a850
    shear_speed = gaussian_filter(np.sqrt(u_shear**2 + v_shear**2), sigma=2.0)

    return lat, lon, {"main": shear_speed, "u": u_shear, "v": v_shear,
                      "_region": region}


# ===========================================================================
# ── 3. VWS Total ─────────────────────────────────────────────────────────────
# ===========================================================================

def _compute_vws(pkg, dates):
    region = pkg.get("region", "io")
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)

    with ThreadPoolExecutor(max_workers=4) as pool:
        f1 = pool.submit(_fetch_uv_obs, 850, dates, lon_min, lon_max, lat_min, lat_max)
        f2 = pool.submit(_fetch_uv_ltm, 850, dates, lon_min, lon_max, lat_min, lat_max)
        f3 = pool.submit(_fetch_uv_obs, 200, dates, lon_min, lon_max, lat_min, lat_max)
        f4 = pool.submit(_fetch_uv_ltm, 200, dates, lon_min, lon_max, lat_min, lat_max)
        lat, lon, u_obs_850, v_obs_850 = f1.result()
        _,   _,   u_ltm_850, _         = f2.result()
        _,   _,   u_obs_200, v_obs_200 = f3.result()
        _,   _,   u_ltm_200, _         = f4.result()

    u_shear = gaussian_filter(u_obs_200 - u_obs_850, sigma=1.5)
    v_shear = gaussian_filter(v_obs_200 - v_obs_850, sigma=1.5)
    shear_speed = gaussian_filter(np.sqrt(u_shear**2 + v_shear**2), sigma=2.0)

    return lat, lon, {"main": shear_speed, "u": u_shear, "v": v_shear,
                      "_region": region, "_total": True}


# ===========================================================================
# ── 4. Shear (shear.py — directional, for reference) ────────────────────────
# ===========================================================================

def _compute_shear(pkg, dates):
    # identical to _compute_vwsa logic; kept as separate kind for labelling
    return _compute_vwsa(pkg, dates)


# ── Shared render for all three shear products ───────────────────────────────

def _render_shear(lat, lon, data, pkg, coast_segs, dates):
    shear_speed = data["main"]
    u_shear     = data["u"]
    v_shear     = data["v"]
    region      = data.get("_region", "io")
    is_total    = data.get("_total", False)

    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)
    is_global = region.lower() == "global"
    fig, map_rect, cax_rect, cax_text_y, title_y, use_aspect = _fig_layout(is_global)

    ax = fig.add_axes(map_rect)
    ax.set_facecolor("#f4f0e8")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    if use_aspect:
        ax.set_aspect("equal", adjustable="box")

    LON2D, LAT2D = np.meshgrid(lon, lat)

    _cdict = {
        "red":   [(0.0, 0.90, 0.90), (0.35, 0.70, 0.70),
                  (0.50, 0.95, 0.95), (0.65, 0.85, 0.85), (1.0, 0.50, 0.50)],
        "green": [(0.0, 0.90, 0.90), (0.35, 0.85, 0.85),
                  (0.50, 0.95, 0.95), (0.65, 0.50, 0.50), (1.0, 0.10, 0.10)],
        "blue":  [(0.0, 0.95, 0.95), (0.35, 0.95, 0.95),
                  (0.50, 0.70, 0.70), (0.65, 0.20, 0.20), (1.0, 0.10, 0.10)],
    }
    shear_cmap = LinearSegmentedColormap("shear_cmap", _cdict, N=512)
    vlim = 30.0
    cf = ax.contourf(LON2D, LAT2D, shear_speed,
                     levels=np.linspace(0, vlim, 20),
                     cmap=shear_cmap, extend="max", zorder=1, alpha=0.88)
    ax.contour(LON2D, LAT2D, shear_speed,
               levels=np.arange(5, 31, 5),
               colors="#333333", linewidths=0.55, alpha=0.55, zorder=2)

    v_scale = 100.0 if is_global else 45.0
    step = 3 if is_global else 2
    qs   = slice(None, None, step)
    Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
    Uq, Vq = u_shear[qs, qs], v_shear[qs, qs]
    mag  = np.sqrt(Uq**2 + Vq**2)
    mask = ~np.isnan(mag) & (Yq >= lat_min) & (Yq <= lat_max) & (Xq >= lon_min) & (Xq <= lon_max)
    w = 0.0018 if is_global else 0.0022
    ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask],
              color="#111111", scale=v_scale, scale_units="inches",
              width=w, headwidth=4.5, headlength=5.5,
              headaxislength=4.8, minshaft=1.2, pivot="middle", zorder=6, alpha=0.92)

    ref_x = lon_max - (lon_max - lon_min) * 0.15
    ref_y = lat_min + (lat_max - lat_min) * 0.08
    ax.quiver(ref_x, ref_y, 10.0, 0, color="#111111",
              scale=v_scale, scale_units="inches", width=w,
              headwidth=4.5, headlength=5.5, headaxislength=4.8, pivot="tail", zorder=9)
    ax.text(ref_x, ref_y - (lat_max - lat_min) * 0.05, "10 m/s",
            fontsize=8, color="#111111", ha="center", zorder=9)

    _draw_coastlines(ax, coast_segs)
    _draw_grid_ticks(ax, lon_min, lon_max, lat_min, lat_max)

    cax  = fig.add_axes(cax_rect)
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=np.arange(0, 31, 5))
    cbar.ax.tick_params(labelsize=8.5, colors="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 31, 5)], fontsize=8.5, color="#222211")
    cbar.outline.set_edgecolor("#999988")
    cbar.outline.set_linewidth(0.7)

    cb_label = ("850–200 hPa Total Vertical Wind Shear  (m s⁻¹)" if is_total
                else "850–200 hPa Vertical Wind Shear Anomaly Speed  (m s⁻¹)")
    cax.text(0.5, cax_text_y, cb_label,
             transform=cax.transAxes, ha="center", va="top",
             fontsize=11 if not is_global else 12, color="#222211", fontstyle="italic")

    rdisp = _REGION_DISPLAY.get(region, region.upper())
    title_str = ("850–200 hPa Total Vertical Wind Shear" if is_total
                 else "850–200 hPa Vertical Wind Shear Anomaly")
    fig.text(0.50, title_y,
             f"{title_str} ({rdisp})  ·  {dates[0]:%-d %b}–{dates[-1]:%-d %b %Y}",
             ha="center", va="top",
             fontsize=15 if not is_global else 16,
             fontweight="bold", color="#111100")

    _watermark(ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ===========================================================================
# ── 5. Instability (lift.py) ─────────────────────────────────────────────────
# ===========================================================================

def _compute_lift(pkg, dates):
    region = pkg.get("region", "io")
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)

    with ThreadPoolExecutor(max_workers=4) as pool:
        f1 = pool.submit(_fetch_air_obs, 850, dates, lon_min, lon_max, lat_min, lat_max)
        f2 = pool.submit(_fetch_air_ltm, 850, dates, lon_min, lon_max, lat_min, lat_max)
        f3 = pool.submit(_fetch_air_obs, 200, dates, lon_min, lon_max, lat_min, lat_max)
        f4 = pool.submit(_fetch_air_ltm, 200, dates, lon_min, lon_max, lat_min, lat_max)
        lat, lon, air_obs_850 = f1.result()
        _,   _,   air_ltm_850 = f2.result()
        _,   _,   air_obs_200 = f3.result()
        _,   _,   air_ltm_200 = f4.result()

    inst_obs   = air_obs_850 - air_obs_200
    inst_ltm   = air_ltm_850 - air_ltm_200
    anom_raw   = inst_obs - inst_ltm
    anom       = gaussian_filter(anom_raw, sigma=2.0)

    return lat, lon, {"main": anom, "_region": region}


def _render_lift(lat, lon, data, pkg, coast_segs, dates):
    anom   = data["main"]
    region = data.get("_region", "io")
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)
    is_global = region.lower() == "global"
    fig, map_rect, cax_rect, cax_text_y, title_y, use_aspect = _fig_layout(is_global)

    ax = fig.add_axes(map_rect)
    ax.set_facecolor("#f4f0e8")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    if use_aspect:
        ax.set_aspect("equal", adjustable="box")

    LON2D, LAT2D = np.meshgrid(lon, lat)
    vlim = max(2.0, np.nanpercentile(np.abs(anom), 95))
    vlim = np.ceil(vlim / 1) * 1

    _cdict = {
        "red":   [(0.0, 0.09, 0.09), (0.35, 0.30, 0.30),
                  (0.50, 0.97, 0.97), (0.65, 0.98, 0.98), (1.0, 0.72, 0.72)],
        "green": [(0.0, 0.25, 0.25), (0.35, 0.60, 0.60),
                  (0.50, 0.97, 0.97), (0.65, 0.68, 0.68), (1.0, 0.18, 0.18)],
        "blue":  [(0.0, 0.68, 0.68), (0.35, 0.85, 0.85),
                  (0.50, 0.97, 0.97), (0.65, 0.25, 0.25), (1.0, 0.10, 0.10)],
    }
    trend_cmap = LinearSegmentedColormap("trend_cmap", _cdict, N=512)
    n_half = 12
    levels_fill = np.concatenate([
        np.linspace(-vlim, 0, n_half + 1)[:-1],
        np.linspace(0, vlim, n_half + 1),
    ])
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    cf = ax.contourf(LON2D, LAT2D, anom, levels=levels_fill,
                     cmap=trend_cmap, norm=norm, extend="both", zorder=1, alpha=0.88)
    line_levs = np.arange(-vlim, vlim + 0.5, 1)
    line_levs = line_levs[line_levs != 0]
    ax.contour(LON2D, LAT2D, anom, levels=line_levs,
               colors="#444433", linewidths=0.50, alpha=0.50, zorder=2)

    _draw_coastlines(ax, coast_segs)
    _draw_grid_ticks(ax, lon_min, lon_max, lat_min, lat_max)

    cax  = fig.add_axes(cax_rect)
    cb_ticks = np.arange(-vlim, vlim + 0.5, 1)
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=cb_ticks)
    cbar.ax.tick_params(labelsize=8.0, colors="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:+.1f}" for v in cb_ticks], fontsize=8.0, color="#222211")
    cbar.outline.set_edgecolor("#999988")
    cbar.outline.set_linewidth(0.7)
    cax.text(0.02, 0.5, "◄ Unfavourable", transform=cax.transAxes,
             ha="left", va="center", fontsize=8.5, color="white", fontweight="bold")
    cax.text(0.98, 0.5, "Favourable ►", transform=cax.transAxes,
             ha="right", va="center", fontsize=8.5, color="white", fontweight="bold")
    cax.text(0.5, cax_text_y, "Atmospheric Instability Anomaly (°C)",
             transform=cax.transAxes, ha="center", va="top",
             fontsize=11 if not is_global else 12, color="#222211", fontstyle="italic")

    fav_p   = mpatches.Patch(facecolor="#b84010", edgecolor="#888", linewidth=0.6, label="Favourable")
    unfav_p = mpatches.Patch(facecolor="#1a5c9e", edgecolor="#888", linewidth=0.6, label="Unfavorable")
    leg = ax.legend(handles=[fav_p, unfav_p], loc="upper left", fontsize=12,
                    framealpha=0.92, edgecolor="#ccccbb", fancybox=True,
                    handlelength=1.4, handleheight=1.1,
                    borderpad=0.7, labelspacing=0.6, title_fontsize=8.0)
    leg.get_frame().set_linewidth(0.8)
    leg.set_zorder(10)

    rdisp = _REGION_DISPLAY.get(region, region.upper())
    fig.text(0.50, title_y,
             f"Atmospheric Instability Anomaly ({rdisp})  ·  {dates[0]:%-d %b}–{dates[-1]:%-d %b %Y}",
             ha="center", va="top",
             fontsize=15 if not is_global else 16,
             fontweight="bold", color="#111100")

    _watermark(ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ===========================================================================
# ── 6. Mid-level Wave Trend (mid.py) ─────────────────────────────────────────
# ===========================================================================

def _compute_mid(pkg, dates):
    region = pkg.get("region", "io")
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)

    with ThreadPoolExecutor(max_workers=4) as pool:
        f1 = pool.submit(_fetch_uv_obs, 700, dates, lon_min, lon_max, lat_min, lat_max)
        f2 = pool.submit(_fetch_uv_ltm, 700, dates, lon_min, lon_max, lat_min, lat_max)
        f3 = pool.submit(_fetch_uv_obs, 500, dates, lon_min, lon_max, lat_min, lat_max)
        f4 = pool.submit(_fetch_uv_ltm, 500, dates, lon_min, lon_max, lat_min, lat_max)
        lat, lon, u_obs_700, v_obs_700 = f1.result()
        _,   _,   u_ltm_700, v_ltm_700 = f2.result()
        _,   _,   u_obs_500, v_obs_500 = f3.result()
        _,   _,   u_ltm_500, v_ltm_500 = f4.result()

    u_obs_mean = (u_obs_700 + u_obs_500) / 2.0
    v_obs_mean = (v_obs_700 + v_obs_500) / 2.0
    u_ltm_mean = (u_ltm_700 + u_ltm_500) / 2.0

    u_trend = gaussian_filter(u_obs_mean - u_ltm_mean, sigma=2.0)
    u_mean  = gaussian_filter(u_obs_mean, sigma=1.5)
    v_mean  = gaussian_filter(v_obs_mean, sigma=1.5)

    return lat, lon, {"main": u_trend, "u": u_mean, "v": v_mean,
                      "_region": region}


def _render_mid(lat, lon, data, pkg, coast_segs, dates):
    u_trend = data["main"]
    u_mean  = data["u"]
    v_mean  = data["v"]
    region  = data.get("_region", "io")
    lon_min, lon_max, lat_min, lat_max = _get_bounds(region)
    is_global = region.lower() == "global"
    fig, map_rect, cax_rect, cax_text_y, title_y, use_aspect = _fig_layout(is_global)

    ax = fig.add_axes(map_rect)
    ax.set_facecolor("#f4f0e8")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    if use_aspect:
        ax.set_aspect("equal", adjustable="box")

    LON2D, LAT2D = np.meshgrid(lon, lat)
    vlim = max(2.0, np.nanpercentile(np.abs(u_trend), 95))
    vlim = np.ceil(vlim / 1) * 1

    _cdict = {
        "red":   [(0.0, 0.09, 0.09), (0.35, 0.30, 0.30),
                  (0.50, 0.97, 0.97), (0.65, 0.98, 0.98), (1.0, 0.72, 0.72)],
        "green": [(0.0, 0.25, 0.25), (0.35, 0.60, 0.60),
                  (0.50, 0.97, 0.97), (0.65, 0.68, 0.68), (1.0, 0.18, 0.18)],
        "blue":  [(0.0, 0.68, 0.68), (0.35, 0.85, 0.85),
                  (0.50, 0.97, 0.97), (0.65, 0.25, 0.25), (1.0, 0.10, 0.10)],
    }
    trend_cmap = LinearSegmentedColormap("trend_cmap2", _cdict, N=512)
    n_half = 12
    levels_fill = np.concatenate([
        np.linspace(-vlim, 0, n_half + 1)[:-1],
        np.linspace(0, vlim, n_half + 1),
    ])
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    cf = ax.contourf(LON2D, LAT2D, u_trend, levels=levels_fill,
                     cmap=trend_cmap, norm=norm, extend="both", zorder=1, alpha=0.88)
    line_levs = np.arange(-vlim, vlim + 0.5, 1)
    line_levs = line_levs[line_levs != 0]
    ax.contour(LON2D, LAT2D, u_trend, levels=line_levs,
               colors="#444433", linewidths=0.50, alpha=0.50, zorder=2)

    v_scale = 100.0 if is_global else 45.0
    step = 3 if is_global else 2
    qs   = slice(None, None, step)
    Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
    Uq, Vq = u_mean[qs, qs], v_mean[qs, qs]
    mag  = np.sqrt(Uq**2 + Vq**2)
    w    = 0.0018 if is_global else 0.0022
    mask = ~np.isnan(mag) & (Yq >= lat_min) & (Yq <= lat_max) & (Xq >= lon_min) & (Xq <= lon_max)
    ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask],
              color="#111111", scale=v_scale, scale_units="inches",
              width=w, headwidth=4.5, headlength=5.5,
              headaxislength=4.8, minshaft=1.2, pivot="middle", zorder=6, alpha=0.92)

    _draw_coastlines(ax, coast_segs)
    _draw_grid_ticks(ax, lon_min, lon_max, lat_min, lat_max)

    cax  = fig.add_axes(cax_rect)
    cb_ticks = np.arange(-vlim, vlim + 0.5, 1)
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=cb_ticks)
    cbar.ax.tick_params(labelsize=8.0, colors="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:+.1f}" for v in cb_ticks], fontsize=8.0, color="#222211")
    cbar.outline.set_edgecolor("#999988")
    cbar.outline.set_linewidth(0.7)
    cax.text(0.02, 0.5, "◄ Unfavourable (Stronger Easterlies)",
             transform=cax.transAxes, ha="left", va="center",
             fontsize=7.5, color="white", fontweight="bold")
    cax.text(0.98, 0.5, "Favourable (Westerly Anomaly) ►",
             transform=cax.transAxes, ha="right", va="center",
             fontsize=7.5, color="white", fontweight="bold")
    cax.text(0.5, cax_text_y, "700–500 hPa U-Wind Trend Anomaly (m s⁻¹)",
             transform=cax.transAxes, ha="center", va="top",
             fontsize=11 if not is_global else 12, color="#222211", fontstyle="italic")

    fav_p   = mpatches.Patch(facecolor="#b84010", edgecolor="#888", linewidth=0.6, label="Favourable")
    unfav_p = mpatches.Patch(facecolor="#1a5c9e", edgecolor="#888", linewidth=0.6, label="Unfavorable")
    leg = ax.legend(handles=[fav_p, unfav_p], loc="upper left", fontsize=12,
                    framealpha=0.92, edgecolor="#ccccbb", fancybox=True,
                    handlelength=1.4, handleheight=1.1,
                    borderpad=0.7, labelspacing=0.6, title_fontsize=8.0)
    leg.get_frame().set_linewidth(0.8)
    leg.set_zorder(10)

    rdisp = _REGION_DISPLAY.get(region, region.upper())
    fig.text(0.50, title_y,
             f"CCKW/MJO Wave Trend ({rdisp})  ·  {dates[0]:%-d %b}–{dates[-1]:%-d %b %Y}",
             ha="center", va="top",
             fontsize=15 if not is_global else 16,
             fontweight="bold", color="#111100")

    _watermark(ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ===========================================================================
# Product registry
# ===========================================================================

_TC_REGIONS = [
    ("io",  "Indian Ocean"),
    ("wp",  "W. Pacific"),
    ("cp",  "C. Pacific"),
    ("ep",  "E. Pacific"),
    ("atl", "Atlantic"),
    ("global", "Global"),
]

PRODUCTS = {}

for _rid, _rname in _TC_REGIONS:
    PRODUCTS[f"chi200_{_rid}"] = {
        "id": f"chi200_{_rid}", "tag": "TC",
        "title": f"Chi 200 • Wind · {_rname}",
        "name":  f"Chi 200 • Wind · {_rname}",
        "desc":  f"200 hPa velocity-potential anomaly ({_rname}).",
        "kind": "tc_chi", "level": 200, "region": _rid,
    }
    PRODUCTS[f"vwsa_{_rid}"] = {
        "id": f"vwsa_{_rid}", "tag": "TC",
        "title": f"VWS Anomaly · {_rname}",
        "name":  f"VWS Anom · {_rname}",
        "desc":  f"850–200 hPa vertical wind shear anomaly ({_rname}).",
        "kind": "tc_vwsa", "level": None, "region": _rid,
    }
    PRODUCTS[f"vws_{_rid}"] = {
        "id": f"vws_{_rid}", "tag": "TC",
        "title": f"VWS Total · {_rname}",
        "name":  f"VWS Total · {_rname}",
        "desc":  f"850–200 hPa total vertical wind shear ({_rname}).",
        "kind": "tc_vws", "level": None, "region": _rid,
    }
    PRODUCTS[f"shear_{_rid}"] = {
        "id": f"shear_{_rid}", "tag": "TC",
        "title": f"Wind Shear · {_rname}",
        "name":  f"Shear · {_rname}",
        "desc":  f"850–200 hPa directional wind shear anomaly ({_rname}).",
        "kind": "tc_shear", "level": None, "region": _rid,
    }
    PRODUCTS[f"lift_{_rid}"] = {
        "id": f"lift_{_rid}", "tag": "TC",
        "title": f"Instability Anomaly · {_rname}",
        "name":  f"Instability · {_rname}",
        "desc":  f"850–200 hPa atmospheric instability anomaly ({_rname}).",
        "kind": "tc_lift", "level": None, "region": _rid,
    }
    PRODUCTS[f"mid_{_rid}"] = {
        "id": f"mid_{_rid}", "tag": "TC",
        "title": f"Wave Trend · {_rname}",
        "name":  f"Wave Trend · {_rname}",
        "desc":  f"700–500 hPa CCKW/MJO wave trend ({_rname}).",
        "kind": "tc_mid", "level": None, "region": _rid,
    }


# ── Kind registration ─────────────────────────────────────────────────────────
KINDS = {
    "tc_chi":   {"compute": _compute_chi,   "render": _render_chi,   "tag": "TC"},
    "tc_vwsa":  {"compute": _compute_vwsa,  "render": _render_shear, "tag": "TC"},
    "tc_vws":   {"compute": _compute_vws,   "render": _render_shear, "tag": "TC"},
    "tc_shear": {"compute": _compute_shear, "render": _render_shear, "tag": "TC"},
    "tc_lift":  {"compute": _compute_lift,  "render": _render_lift,  "tag": "TC"},
    "tc_mid":   {"compute": _compute_mid,   "render": _render_mid,   "tag": "TC"},
}
