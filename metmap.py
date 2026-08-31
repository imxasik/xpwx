"""
metmap.py — Core map engine for the 200-hPa Velocity Potential & Wind Anomaly map.

Wraps the original standalone script into a reusable module so a web app
(Flask) can request a map for any date/mode and get PNG bytes back.

Data source: NCEP/NCAR Reanalysis via PSL / NOAA (THREDDS / OPeNDAP).
"""

import os
import io
import re
import zipfile
import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import shapefile
import requests
from pydap.client import open_url

# ================================================================
# Configuration
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_DIR = os.path.join(BASE_DIR, "map")
SHP_PATH = os.path.join(SHP_DIR, "ne_110m_coastline.shp")
SHP_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip"
PSL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep"
DEFAULT_N_DAYS = 5
DEFAULT_HPA = 200.0

# ================================================================
# Coastline handling
# ================================================================
def ensure_coastline():
    if os.path.exists(SHP_PATH):
        return
    os.makedirs(SHP_DIR, exist_ok=True)
    r = requests.get(SHP_URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if os.path.splitext(name)[1] in (".shp", ".shx", ".dbf", ".prj"):
            with open(os.path.join(SHP_DIR, os.path.basename(name)), "wb") as f:
                f.write(z.read(name))


def load_coastlines():
    sf = shapefile.Reader(SHP_PATH)
    segs = []
    for shape in sf.shapes():
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(shape.parts)):
            segs.append(np.array(pts[parts[i]:parts[i + 1]]))
    return segs


# ================================================================
# OPeNDAP helpers
# ================================================================
def _open(varname, year):
    url = f"{PSL}/{varname}.{year}.nc"
    return open_url(url)


def _open_ltm(varname):
    url = f"{PSL}/{varname}.day.ltm.1991-2020.nc"
    return open_url(url)


def _latlon(ds):
    return np.array(ds["lat"][:]), np.array(ds["lon"][:])


def _level_idx(ds, hPa=DEFAULT_HPA):
    lev = np.array(ds["level"][:])
    return int(np.argmin(np.abs(lev - hPa)))


def _epoch(ds):
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
    return (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if m else datetime.date(1800, 1, 1))


def _time_idx(ds, target):
    raw = np.array(ds["time"][:])
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    scale = 1.0 / 24.0 if "hours" in units else 1.0
    epoch = _epoch(ds)
    for i, t in enumerate(raw):
        d = epoch + datetime.timedelta(days=float(t) * scale)
        if d.year == target.year and d.month == target.month and d.day == target.day:
            return i
    raise ValueError(f"Date {target} not found in dataset")


def _read_slice(ds, varname, t, lv):
    raw = np.array(ds[varname][t, lv, :, :].data).squeeze().astype(np.float64)
    attr = ds[varname].attributes
    sf = float(attr.get("scale_factor", 1.0))
    ao = float(attr.get("add_offset", 0.0))
    mv = float(attr.get("missing_value", 32767.0))
    fill_mask = np.abs(raw - mv) < 0.5
    data = raw * sf + ao
    data[fill_mask] = np.nan
    return data


# ================================================================
# Data fetching
# ================================================================
def fetch_both_uv_obs(dates):
    by_year = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)
    lat = lon = None
    u_slices, v_slices = [], []
    for year, ydates in sorted(by_year.items()):
        ds_u = _open("uwnd", year)
        ds_v = _open("vwnd", year)
        if lat is None:
            lat, lon = _latlon(ds_u)
        lv_u = _level_idx(ds_u)
        lv_v = _level_idx(ds_v)
        for d in ydates:
            ti_u = _time_idx(ds_u, d)
            ti_v = _time_idx(ds_v, d)
            u_slices.append(_read_slice(ds_u, "uwnd", ti_u, lv_u))
            v_slices.append(_read_slice(ds_v, "vwnd", ti_v, lv_v))
    return lat, lon, np.nanmean(u_slices, axis=0), np.nanmean(v_slices, axis=0)


def fetch_both_uv_ltm(dates):
    ds_u = _open_ltm("uwnd")
    ds_v = _open_ltm("vwnd")
    lat, lon = _latlon(ds_u)
    lv_u = _level_idx(ds_u)
    lv_v = _level_idx(ds_v)
    n_u = len(np.array(ds_u["time"][:]))
    n_v = len(np.array(ds_v["time"][:]))
    u_slices, v_slices = [], []
    for d in dates:
        doy = d.timetuple().tm_yday
        ti_u = min(doy - 1, n_u - 1)
        ti_v = min(doy - 1, n_v - 1)
        u_slices.append(_read_slice(ds_u, "uwnd", ti_u, lv_u))
        v_slices.append(_read_slice(ds_v, "vwnd", ti_v, lv_v))
    return lat, lon, np.nanmean(u_slices, axis=0), np.nanmean(v_slices, axis=0)


# ================================================================
# Physics: divergence & Poisson solver
# ================================================================
def divergence(u, v, lat, lon):
    R = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)
    dudx = np.gradient(u, lon_r, axis=1) / (R * coslat[:, None])
    vcoslat = v * coslat[:, None]
    dvdy = np.gradient(vcoslat, lat_r, axis=0) / (R * coslat[:, None])
    return dudx + dvdy


def poisson_fft(rhs, lat, lon):
    R = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dy = R * np.abs(np.mean(np.diff(lat_r)))
    coslat = np.cos(lat_r)
    dx_mean = R * np.mean(np.diff(lon_r)) * np.mean(np.abs(coslat))
    nlat, nlon = rhs.shape
    rhs_clean = np.nan_to_num(rhs, nan=0.0)
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


# ================================================================
# Map rendering
# ================================================================
def draw_map(lat, lon, chi_anom, u_anom, v_anom, coast_segs, dates, out_buf=None):
    LON2D, LAT2D = np.meshgrid(lon, lat)
    chi_plot = chi_anom * 1e-6

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    MAP_L, MAP_B, MAP_W, MAP_H = 0.045, 0.145, 0.910, 0.785
    ax = fig.add_axes([MAP_L, MAP_B, MAP_W, MAP_H])
    ax.set_facecolor("#f4f0e8")

    lon_min, lon_max = lon.min(), lon.max()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(-75, 75)

    # 1. Filled shading
    vlim = 10.0
    levels_fill = np.linspace(-vlim, vlim, 20)
    _cdict = {
        "red":   [(0.0, 0.08, 0.08), (0.35, 0.40, 0.40),
                  (0.50, 0.97, 0.97), (0.65, 0.92, 0.92), (1.0, 0.55, 0.55)],
        "green": [(0.0, 0.38, 0.38), (0.35, 0.72, 0.72),
                  (0.50, 0.97, 0.97), (0.65, 0.78, 0.78), (1.0, 0.30, 0.30)],
        "blue":  [(0.0, 0.45, 0.45), (0.35, 0.78, 0.78),
                  (0.50, 0.97, 0.97), (0.65, 0.52, 0.52), (1.0, 0.10, 0.10)],
    }
    chi_cmap = LinearSegmentedColormap("chi_cmap", _cdict, N=512)
    cf = ax.contourf(LON2D, LAT2D, chi_plot, levels=levels_fill,
                     cmap=chi_cmap, extend="both", zorder=1, alpha=0.88)

    # 2. Thin contour lines
    chi_line_lev = np.arange(-10, 10.1, 2.5)
    chi_line_lev = chi_line_lev[chi_line_lev != 0]
    ax.contour(LON2D, LAT2D, chi_plot, levels=chi_line_lev[chi_line_lev > 0],
               colors="#5c3d11", linewidths=0.55, alpha=0.55, zorder=2)
    ax.contour(LON2D, LAT2D, chi_plot, levels=chi_line_lev[chi_line_lev < 0],
               colors="#1b4f6b", linewidths=0.55, linestyles="--",
               alpha=0.55, zorder=2)

    # 3. Wind anomaly vectors
    step = 3
    qs = slice(None, None, step)
    Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
    Uq, Vq = u_anom[qs, qs], v_anom[qs, qs]
    mag = np.sqrt(Uq**2 + Vq**2)
    mask = ~np.isnan(mag) & (np.abs(Yq) <= 70)
    ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask],
              color="#111111", scale=50.0, scale_units="inches",
              width=0.0018, headwidth=4.5, headlength=5.5,
              headaxislength=4.8, minshaft=1.2, pivot="middle",
              zorder=6, alpha=0.92)
    ax.quiver(lon_max - 28, -68, 5.0, 0, color="#111111",
              scale=50.0, scale_units="inches", width=0.0018, headwidth=4.5,
              headlength=5.5, headaxislength=4.8, pivot="tail", zorder=9)
    ax.text(lon_max - 28, -72, "5 m/s", fontsize=8, color="#111111",
            ha="center", zorder=9)

    # 4. Coastlines
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

    # 5. Grid lines
    for x in range(int(lon_min), int(lon_max) + 1, 30):
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in range(-60, 61, 20):
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.75, zorder=0, alpha=0.8)

    def xlab(v):
        if v in (0, 360): return "0°"
        if v == 180: return "180°"
        if v <= 180: return f"{v}°E"
        return f"{360 - v}°W"

    def ylab(v):
        return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

    xticks = list(range(0, 360, 30))
    yticks = list(range(-80, 81, 20))
    ax.set_xticks(xticks)
    ax.set_xticklabels([xlab(x) for x in xticks], fontsize=9.5,
                       color="#333322", fontfamily="DejaVu Sans")
    ax.set_yticks(yticks)
    ax.set_yticklabels([ylab(y) for y in yticks], fontsize=9.5,
                       color="#333322", fontfamily="DejaVu Sans")
    ax.tick_params(axis="both", length=3.5, color="#888878", width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988")
        spine.set_linewidth(0.8)

    # 6. Colorbar
    cax = fig.add_axes([0.12, 0.057, 0.760, 0.028])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal",
                        ticks=np.arange(-10, 11, 2.5))
    cbar.ax.tick_params(labelsize=8.5, colors="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:.1f}" for v in np.arange(-10, 11, 2.5)],
                            fontsize=8.5, color="#222211")
    cbar.outline.set_edgecolor("#999988")
    cbar.outline.set_linewidth(0.7)
    cax.text(0.5, -1.55, r"Velocity-Potential Anomaly  (1e6 m²s)",
             transform=cax.transAxes, ha="center", va="top",
             fontsize=12, color="#222211", fontstyle="italic")

    # 7. Titles & labels
    dstart, dend = dates[0], dates[-1]
    fig.text(0.50, 0.985,
             f"Velocity Potential Anomaly & Wind Anomaly  ·  "
             f"{dstart:%-d %b}–{dend:%-d %b %Y}\n",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color="#111100", fontfamily="DejaVu Sans")

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


# ================================================================
# Top-level API used by the web app
# ================================================================
def _resolve_dates(mode, manual_date, n_days):
    n_days = max(1, int(n_days))
    if mode == "manual" and manual_date:
        date_end = datetime.date.fromisoformat(manual_date)
    else:
        # auto: end at most recent available date
        current_year = datetime.date.today().year
        try:
            ds_temp = open_url(f"{PSL}/uwnd.{current_year}.nc")
        except Exception:
            current_year -= 1
            ds_temp = open_url(f"{PSL}/uwnd.{current_year}.nc")
        raw_times = np.array(ds_temp["time"][:])
        units = ds_temp["time"].attributes.get("units", "hours since 1800-01-01")
        scale = 1.0 / 24.0 if "hours" in units else 1.0
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
        epoch = (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                 if m else datetime.date(1800, 1, 1))
        date_end = epoch + datetime.timedelta(days=float(raw_times[-1]) * scale)

    date_start = date_end - datetime.timedelta(days=n_days - 1)
    dates = [date_start + datetime.timedelta(days=i) for i in range(n_days)]
    return dates


def generate_map(mode="auto", manual_date=None, n_days=DEFAULT_N_DAYS,
                 log=None):
    """Return (png_bytes, meta_dict) for the requested map."""

    def say(msg):
        if log is not None:
            log.append(msg)
        return msg

    dates = _resolve_dates(mode, manual_date, n_days)
    say(f"[0/4] Dates: {dates[0]} → {dates[-1]}  ({len(dates)}-day mean)")

    say("[0/4] Loading coastline …")
    ensure_coastline()
    coast_segs = load_coastlines()
    say(f"  {len(coast_segs)} coastline segments loaded.")

    say("[1-2/4] Fetching observations & LTM (parallel) …")
    obs = fetch_both_uv_obs(dates)
    ltm = fetch_both_uv_ltm(dates)
    lat, lon, u_obs, v_obs = obs
    _, _, u_ltm, v_ltm = ltm

    say("[3/4] Computing χ200 anomaly & wind anomaly …")
    u_anom = gaussian_filter(u_obs - u_ltm, sigma=1.5)
    v_anom = gaussian_filter(v_obs - v_ltm, sigma=1.5)
    div = divergence(u_anom, v_anom, lat, lon)
    chi = gaussian_filter(poisson_fft(div, lat, lon), sigma=2.0)
    say(f"  chi range: {np.nanmin(chi) * 1e-6:.2f} … "
        f"{np.nanmax(chi) * 1e-6:.2f} ×10⁶ m²/s")

    say("[4/4] Drawing map …")
    buf = draw_map(lat, lon, chi, u_anom, v_anom, coast_segs, dates)

    meta = {
        "date_start": dates[0].isoformat(),
        "date_end": dates[-1].isoformat(),
        "n_days": len(dates),
    }
    return buf, meta
