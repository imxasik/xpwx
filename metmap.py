"""
metmap.py — Fully data-driven map engine for NCEP/NCAR Reanalysis.

Design goals
------------
1. ADD A MAP WITHOUT EDITING CODE:
   Every product is a small config dict in PRODUCTS. One generic compute +
   one generic renderer handle all of them. To add a map, append one dict
   and restart — the sidebar, /products endpoint and /generate all pick it up.

2. ACCURATE:
   - Velocity potential chi inverts the divergence of the wind anomaly.
   - Streamfunction psi inverts the relative vorticity of the wind anomaly
     (zeta = (1/(R cosphi))[ dv/dlon - d(u cosphi)/dphi ], solved for del^2 psi
     with a spherical FFT Poisson solver, band-limited planetary waves).
   - Anomalies are always field minus the 1991-2020 daily climatology.
   - Zonal-wind and temperature anomalies are straightforward obs - clim.

3. SUPER FAST:
   - OPeNDAP datasets cached (opened once) -> _DS_CACHE
   - Raw obs/climatology field means cached per (var, level, dates) -> _FIELD_CACHE
   - Rendered PNGs cached server-side (in app.py) -> near-instant repeats
"""

import os
import io
import re
import zipfile
import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor

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
DEFAULT_PRODUCT = "vtp200"

_DS_CACHE = {}       # url -> pydap dataset
_FIELD_CACHE = {}    # (var, level, dates, kind) -> mean field
_LATLON_CACHE = {}   # var -> (lat, lon)
_COAST = None        # list of coastline segments


# ================================================================
# Coastline
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
    global _COAST
    if _COAST is None:
        ensure_coastline()
        sf = shapefile.Reader(SHP_PATH)
        segs = []
        for shape in sf.shapes():
            pts = shape.points
            parts = list(shape.parts) + [len(pts)]
            for i in range(len(shape.parts)):
                segs.append(np.array(pts[parts[i]:parts[i + 1]]))
        _COAST = segs
    return _COAST


# ================================================================
# OPeNDAP helpers (cached)
# ================================================================
def _open(varname, year=None):
    if year is None:
        url = f"{PSL}/{varname}.day.ltm.1991-2020.nc"
    else:
        url = f"{PSL}/{varname}.{year}.nc"
    if url not in _DS_CACHE:
        _DS_CACHE[url] = open_url(url)
    return _DS_CACHE[url]


def _latlon(var):
    if var not in _LATLON_CACHE:
        ds = _open(var, year=2024)
        _LATLON_CACHE[var] = (np.array(ds["lat"][:]), np.array(ds["lon"][:]))
    return _LATLON_CACHE[var]


def _level_idx(ds, hPa):
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
# Cached field mean  (obs or climatology) for (var, level, dates)
# ================================================================
def _mean_field(var, level, dates, kind):
    key = (var, level, tuple(d.isoformat() for d in dates), kind)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    if kind == "obs":
        by_year = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        slices = []
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            lv = _level_idx(ds, level) if level is not None else 0
            for d in ydates:
                ti = _time_idx(ds, d)
                slices.append(_read_slice(ds, var, ti, lv))
    else:
        ds = _open(var)
        lv = _level_idx(ds, level) if level is not None else 0
        n = len(np.array(ds["time"][:]))
        slices = []
        for d in dates:
            doy = d.timetuple().tm_yday
            ti = min(doy - 1, n - 1)
            slices.append(_read_slice(ds, var, ti, lv))

    mean = np.nanmean(slices, axis=0)
    _FIELD_CACHE[key] = mean
    return mean


# ================================================================
# Physics
# ================================================================
R_EARTH = 6.371e6


def divergence(u, v, lat, lon):
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)
    dudx = np.gradient(u, lon_r, axis=1) / (R_EARTH * coslat[:, None])
    vcoslat = v * coslat[:, None]
    dvdy = np.gradient(vcoslat, lat_r, axis=0) / (R_EARTH * coslat[:, None])
    return dudx + dvdy


def vorticity(u, v, lat, lon):
    """Relative vorticity (vertical component) from u,v on a lon-lat grid."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)[:, None]
    dudphi = np.gradient(u * coslat, lat_r, axis=0)
    dvdlon = np.gradient(v, lon_r, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(np.abs(coslat) > 1e-4,
                        (1.0 / (R_EARTH * coslat)) * (dvdlon - dudphi), 0.0)
    # fix the (ill-defined) pole rows with the adjacent row
    zeta[0] = zeta[1]
    zeta[-1] = zeta[-2]
    return zeta


def poisson_fft(rhs, lat, lon):
    """Solve del^2(psi) = rhs on the sphere via FFT (band-limited)."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dy = R_EARTH * np.abs(np.mean(np.diff(lat_r)))
    dx_mean = R_EARTH * np.mean(np.diff(lon_r)) * np.mean(np.abs(np.cos(lat_r)))
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
# Product registry
# ================================================================
# kind:
#   "vtp"  -> chi (velocity potential) anomaly + wind anomaly overlay
#   "psi"  -> psi (streamfunction) anomaly + wind anomaly overlay
#   "anom" -> single-field anomaly (pkg["variable"])
# config:
#   variables, variable (for anom), level, plot_scale, vlim, cint, cb_label
#   show_wind, wind_scale
#
# To ADD a map: copy a block, change the values, restart. Done.
PRODUCTS = {
    # ---- velocity potential + wind (upper-level convergence/divergence) ----
    "vtp200": {"id": "vtp200", "title": "Velocity Potential & Wind Anomaly — 200 hPa",
               "name": "χ200 · Wind", "tag": "Upper",
               "desc": "200-hPa velocity-potential (divergence) and wind anomalies.",
               "kind": "vtp", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 10.0, "cint": 2.5,
               "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},
    "vtp500": {"id": "vtp500", "title": "Velocity Potential & Wind Anomaly — 500 hPa",
               "name": "χ500 · Wind", "tag": "Mid",
               "desc": "500-hPa velocity-potential and wind anomalies.",
               "kind": "vtp", "level": 500, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 10.0, "cint": 2.5,
               "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},
    "vtp850": {"id": "vtp850", "title": "Velocity Potential & Wind Anomaly — 850 hPa",
               "name": "χ850 · Wind", "tag": "Low",
               "desc": "850-hPa velocity-potential and wind anomalies.",
               "kind": "vtp", "level": 850, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 10.0, "cint": 2.5,
               "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},

    # ---- streamfunction + Rossby wave train ----
    "psi200": {"id": "psi200", "title": "Streamfunction Anomaly — 200 hPa",
               "name": "ψ200", "tag": "Upper",
               "desc": "200-hPa streamfunction anomaly (rotational circulation centers).",
               "kind": "psi", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": False, "plot_scale": 1e-6,
               "vlim": 12.0, "cint": 3.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rwt200": {"id": "rwt200", "title": "Rossby Wave Train Circulation — 200 hPa",
               "name": "Wave Train ψ200", "tag": "Upper",
               "desc": "200-hPa streamfunction anomaly + wind: Rossby wave train "
                       "of alternating cyclonic/anticyclonic cells.",
               "kind": "psi", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 12.0, "cint": 3.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},

    # ---- geopotential height anomaly ----
    "hgt200": {"id": "hgt200", "title": "Geopotential Height Anomaly — 200 hPa",
               "name": "H200", "tag": "Upper",
               "desc": "200-hPa geopotential height anomaly (upper ridges & troughs).",
               "kind": "anom", "variable": "hgt", "level": 200,
               "show_wind": False, "plot_scale": 1.0,
               "vlim": 150.0, "cint": 30.0,
               "cb_label": "Geopotential Height Anomaly  (gpm)"},
    "hgt500": {"id": "hgt500", "title": "Geopotential Height Anomaly — 500 hPa",
               "name": "H500", "tag": "Mid",
               "desc": "500-hPa geopotential height anomaly (mid-tropospheric ridges & troughs).",
               "kind": "anom", "variable": "hgt", "level": 500,
               "show_wind": False, "plot_scale": 1.0,
               "vlim": 150.0, "cint": 30.0,
               "cb_label": "Geopotential Height Anomaly  (gpm)"},
    "hgt850": {"id": "hgt850", "title": "Geopotential Height Anomaly — 850 hPa",
               "name": "H850", "tag": "Low",
               "desc": "850-hPa geopotential height anomaly (low-level ridges & troughs).",
               "kind": "anom", "variable": "hgt", "level": 850,
               "show_wind": False, "plot_scale": 1.0,
               "vlim": 150.0, "cint": 30.0,
               "cb_label": "Geopotential Height Anomaly  (gpm)"},

    # ---- zonal wind anomaly ----
    "u200": {"id": "u200", "title": "Zonal Wind Anomaly — 200 hPa",
             "name": "U200", "tag": "Upper",
             "desc": "200-hPa zonal (east-west) wind anomaly.",
             "kind": "anom", "variable": "uwnd", "level": 200,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 15.0, "cint": 3.0,
             "cb_label": "Zonal Wind Anomaly  (m/s)"},
    "u850": {"id": "u850", "title": "Zonal Wind Anomaly — 850 hPa",
             "name": "U850", "tag": "Low",
             "desc": "850-hPa zonal (east-west) wind anomaly.",
             "kind": "anom", "variable": "uwnd", "level": 850,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 10.0, "cint": 2.0,
             "cb_label": "Zonal Wind Anomaly  (m/s)"},

    # ---- temperature anomaly ----
    "t200": {"id": "t200", "title": "Temperature Anomaly — 200 hPa",
             "name": "T200", "tag": "Upper",
             "desc": "200-hPa temperature anomaly.",
             "kind": "anom", "variable": "air", "level": 200,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 6.0, "cint": 1.5,
             "cb_label": "Temperature Anomaly  (K)"},
    "t850": {"id": "t850", "title": "Temperature Anomaly — 850 hPa",
             "name": "T850", "tag": "Low",
             "desc": "850-hPa temperature anomaly.",
             "kind": "anom", "variable": "air", "level": 850,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 8.0, "cint": 2.0,
             "cb_label": "Temperature Anomaly  (K)"},
}


# ================================================================
# Compute  -> returns (lat, lon, data)
#   data = {"main": 2D field (already scaled for display),
#           "u","v": optional wind-anomaly vectors (m/s)}
# ================================================================
def compute(pkg, dates):
    kind = pkg["kind"]

    if kind in ("vtp", "psi"):
        lat, lon = _latlon("uwnd")
        u_obs = _mean_field("uwnd", pkg["level"], dates, "obs")
        u_clim = _mean_field("uwnd", pkg["level"], dates, "clim")
        v_obs = _mean_field("vwnd", pkg["level"], dates, "obs")
        v_clim = _mean_field("vwnd", pkg["level"], dates, "clim")
        u_anom = gaussian_filter(u_obs - u_clim, sigma=1.5)
        v_anom = gaussian_filter(v_obs - v_clim, sigma=1.5)
        if kind == "vtp":
            div = divergence(u_anom, v_anom, lat, lon)
            main = gaussian_filter(poisson_fft(div, lat, lon), sigma=2.0) * pkg["plot_scale"]
        else:
            zeta = vorticity(u_anom, v_anom, lat, lon)
            main = gaussian_filter(poisson_fft(zeta, lat, lon), sigma=2.0) * pkg["plot_scale"]
        return lat, lon, {"main": main, "u": u_anom, "v": v_anom}

    else:  # "anom" — single-variable anomaly
        var = pkg["variable"]
        lat, lon = _latlon(var)
        obs = _mean_field(var, pkg["level"], dates, "obs")
        clim = _mean_field(var, pkg["level"], dates, "clim")
        anom = gaussian_filter(obs - clim, sigma=1.5) * pkg["plot_scale"]
        return lat, lon, {"main": anom}


# ================================================================
# Generic renderer
# ================================================================
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


def render(lat, lon, data, pkg, coast_segs, dates, out_buf=None,
           title=None, cbar_label=None):
    fplot = data["main"]
    vlim, cint = pkg["vlim"], pkg["cint"]
    LON2D, LAT2D = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    # reserve a clean title band above the axes so the title never overlaps the map
    ax = fig.add_axes([0.045, 0.145, 0.910, 0.750])
    ax.set_facecolor("#f4f0e8")
    lon_min, lon_max = lon.min(), lon.max()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(-75, 75)

    # filled shading
    n_fill = 25 if vlim >= 100 else 20
    levels_fill = np.linspace(-vlim, vlim, n_fill)
    cf = ax.contourf(LON2D, LAT2D, fplot, levels=levels_fill,
                     cmap=_chi_cmap(), extend="both", zorder=1, alpha=0.88)

    # thin contour lines (solid positive, dashed negative)
    line_lev = np.arange(-vlim, vlim + 0.01, cint)
    line_lev = line_lev[line_lev != 0]
    ax.contour(LON2D, LAT2D, fplot, levels=line_lev[line_lev > 0],
               colors="#5c3d11", linewidths=0.55, alpha=0.55, zorder=2)
    ax.contour(LON2D, LAT2D, fplot, levels=line_lev[line_lev < 0],
               colors="#1b4f6b", linewidths=0.55, linestyles="--",
               alpha=0.55, zorder=2)

    # wind overlay
    if pkg.get("show_wind") and "u" in data:
        step = 3
        qs = slice(None, None, step)
        Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
        Uq, Vq = data["u"][qs, qs], data["v"][qs, qs]
        mag = np.sqrt(Uq**2 + Vq**2)
        mask = ~np.isnan(mag) & (np.abs(Yq) <= 70)
        ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask], color="#111111",
                  scale=pkg["wind_scale"], scale_units="inches", width=0.0018,
                  headwidth=4.5, headlength=5.5, headaxislength=4.8,
                  minshaft=1.2, pivot="middle", zorder=6, alpha=0.92)
        ax.quiver(lon_max - 28, -68, 5.0, 0, color="#111111",
                  scale=pkg["wind_scale"], scale_units="inches", width=0.0018,
                  headwidth=4.5, headlength=5.5, headaxislength=4.8,
                  pivot="tail", zorder=9)
        ax.text(lon_max - 28, -72, "5 m/s", fontsize=8, color="#111111",
                ha="center", zorder=9)

    # coastlines
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

    # grid lines
    for x in range(int(lon_min), int(lon_max) + 1, 30):
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in range(-60, 61, 20):
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.75, zorder=0, alpha=0.8)

    # axes
    xticks = list(range(0, 360, 30))
    yticks = list(range(-80, 81, 20))
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
    ticks = list(np.arange(-vlim, vlim + 0.001, cint))
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


# ================================================================
# Date resolution
# ================================================================
def _resolve_dates(mode, manual_date, n_days):
    n_days = max(1, int(n_days))
    if mode == "manual" and manual_date:
        date_end = datetime.date.fromisoformat(manual_date)
    else:
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
    return [date_start + datetime.timedelta(days=i) for i in range(n_days)]


# ================================================================
# Public API
# ================================================================
def list_products():
    return [{"id": p["id"], "title": p["title"], "name": p["name"],
             "desc": p["desc"], "level": p["level"], "tag": p["tag"]}
            for p in PRODUCTS.values()]


def generate(product_id=DEFAULT_PRODUCT, mode="auto", manual_date=None,
             n_days=DEFAULT_N_DAYS, log=None):
    pkg = PRODUCTS.get(product_id, PRODUCTS[DEFAULT_PRODUCT])
    say = (lambda m: log.append(m)) if log is not None else (lambda m: None)

    dates = _resolve_dates(mode, manual_date, n_days)
    say(f"[0/4] {pkg['title']} | {dates[0]} → {dates[-1]} ({len(dates)}-day mean)")
    say("[0/4] Loading coastline …")
    coast_segs = load_coastlines()
    say(f"  {len(coast_segs)} segments loaded.")
    say("[1-2/4] Fetching obs & climatology (cached) …")
    lat, lon, data = compute(pkg, dates)
    say(f"  grid {lat.size}×{lon.size} @ {pkg['level']} hPa")
    say("[3/4] Rendering …")
    buf = render(lat, lon, data, pkg, coast_segs, dates)
    meta = {"product": pkg["id"], "title": pkg["title"],
            "date_start": dates[0].isoformat(), "date_end": dates[-1].isoformat(),
            "n_days": len(dates), "level": pkg["level"]}
    return buf, meta


def generate_diff(product_id=DEFAULT_PRODUCT, date1=None, n_days1=DEFAULT_N_DAYS,
                  date2=None, n_days2=DEFAULT_N_DAYS, log=None):
    """Return one map of (Range A − Range B) for the given product."""
    pkg = PRODUCTS.get(product_id, PRODUCTS[DEFAULT_PRODUCT])
    say = (lambda m: log.append(m)) if log is not None else (lambda m: None)

    dates_a = _resolve_dates("manual", date1, n_days1)
    dates_b = _resolve_dates("manual", date2, n_days2)
    say(f"[diff] {pkg['title']}: A={dates_a[0]}→{dates_a[-1]}  "
        f"B={dates_b[0]}→{dates_b[-1]}")

    say("[0] Loading coastline …")
    coast_segs = load_coastlines()

    say("[1] Computing Range A …")
    lat, lon, data_a = compute(pkg, dates_a)
    say("[2] Computing Range B …")
    _, _, data_b = compute(pkg, dates_b)

    say("[3] Difference B − A …")
    data = {"main": data_b["main"] - data_a["main"]}
    if "u" in data_a and "u" in data_b:
        data["u"] = data_b["u"] - data_a["u"]
        data["v"] = data_b["v"] - data_a["v"]

    f_ab = lambda d: f"{d[0]:%-d %b}–{d[-1]:%-d %b %Y}"
    title = (f"{pkg['title']}  ·  B−A   "
             f"({f_ab(dates_b)}) − ({f_ab(dates_a)})")
    buf = render(lat, lon, data, pkg, coast_segs, dates_a,
                 title=title, cbar_label=pkg["cb_label"] + "  (B − A)")

    meta = {"product": pkg["id"], "title": pkg["title"],
            "date_start": dates_a[0].isoformat(), "date_end": dates_a[-1].isoformat(),
            "date_b_start": dates_b[0].isoformat(), "date_b_end": dates_b[-1].isoformat(),
            "n_days": len(dates_a), "level": pkg["level"], "diff": True}
    return buf, meta
