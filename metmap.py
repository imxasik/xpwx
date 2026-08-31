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
# file-prefix key -> actual pydap variable name (pydap splits on '.')
_PVAR = {"uwnd.sfc": "uwnd", "vwnd.sfc": "vwnd",
         "air.sfc": "air", "rhum.sfc": "rhum"}


def _pvar(key):
    return _PVAR.get(key, key)


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
    """Read one time slice. If the dataset has a 'level' dim use it, else surface."""
    var = _pvar(varname)
    if "level" in ds:
        raw = np.array(ds[var][t, lv, :, :].data).squeeze().astype(np.float64)
    else:
        raw = np.array(ds[var][t, :, :].data).squeeze().astype(np.float64)
    attr = ds[var].attributes
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


def _mean_multi(var, levels, dates, kind):
    """Period-mean field over several pressure levels -> (nlevels, nlat, nlon)."""
    key = (var, "multi", tuple(levels), tuple(d.isoformat() for d in dates), kind)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    if kind == "obs":
        by_year = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        stack = []
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            for lv in range(len(levels)):
                sl = []
                for d in ydates:
                    ti = _time_idx(ds, d)
                    sl.append(_read_slice(ds, var, ti, lv))
                stack.append(np.nanmean(sl, axis=0))
    else:
        ds = _open(var)
        n = len(np.array(ds["time"][:]))
        stack = []
        for lv in range(len(levels)):
            sl = []
            for d in dates:
                doy = d.timetuple().tm_yday
                ti = min(doy - 1, n - 1)
                sl.append(_read_slice(ds, var, ti, lv))
            stack.append(np.nanmean(sl, axis=0))

    arr = np.stack(stack, axis=0)          # (nlevels, nlat, nlon)
    _FIELD_CACHE[key] = arr
    return arr


# ================================================================
# Physics
# ================================================================
R_EARTH = 6.371e6
# pressure levels available in the NCEP/NCAR daily multi-level files
AAM_LEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50]


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
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rwt200": {"id": "rwt200", "title": "Rossby Wave Train Circulation — 200 hPa",
               "name": "Wave Train ψ200", "tag": "Upper",
               "desc": "200-hPa streamfunction anomaly + wind: Rossby wave train "
                       "of alternating cyclonic/anticyclonic cells.",
               "kind": "psi", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
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

    # ---- angular momentum budget ----
    "frict": {"id": "frict", "title": "Frictional (Surface Stress) Torque Anomaly",
              "name": "Friction τx", "tag": "Surface",
              "desc": "Surface zonal wind-stress anomaly (the frictional-torque "
                      "driver), from 10-m winds via bulk drag.",
              "kind": "ft", "level": None, "variables": [],
              "show_wind": False, "plot_scale": 100.0,
              "vlim": 36.0, "cint": 6.0,
              "cb_label": "Surface Zonal Stress Anomaly  (×10⁻² N/m²)"},

    # ---- meridional wind anomaly (V) ----
    "v200": {"id": "v200", "title": "Meridional Wind Anomaly — 200 hPa",
             "name": "V200", "tag": "Upper",
             "desc": "200-hPa meridional (south-north) wind anomaly.",
             "kind": "anom", "variable": "vwnd", "level": 200,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 25.0, "cint": 5.0,
             "cb_label": "Meridional Wind Anomaly  (m/s)"},
    "v850": {"id": "v850", "title": "Meridional Wind Anomaly — 850 hPa",
             "name": "V850", "tag": "Low",
             "desc": "850-hPa meridional (south-north) wind anomaly.",
             "kind": "anom", "variable": "vwnd", "level": 850,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 15.0, "cint": 3.0,
             "cb_label": "Meridional Wind Anomaly  (m/s)"},

    # ---- relative humidity anomaly ----
    "rh850": {"id": "rh850", "title": "Relative Humidity Anomaly — 850 hPa",
              "name": "RH850", "tag": "Low",
              "desc": "850-hPa relative humidity anomaly.",
              "kind": "anom", "variable": "rhum", "level": 850,
              "show_wind": False, "plot_scale": 1.0,
              "vlim": 40.0, "cint": 8.0,
              "cb_label": "Relative Humidity Anomaly  (%)"},
    "rh700": {"id": "rh700", "title": "Relative Humidity Anomaly — 700 hPa",
              "name": "RH700", "tag": "Mid",
              "desc": "700-hPa relative humidity anomaly.",
              "kind": "anom", "variable": "rhum", "level": 700,
              "show_wind": False, "plot_scale": 1.0,
              "vlim": 40.0, "cint": 8.0,
              "cb_label": "Relative Humidity Anomaly  (%)"},
    "rh500": {"id": "rh500", "title": "Relative Humidity Anomaly — 500 hPa",
              "name": "RH500", "tag": "Mid",
              "desc": "500-hPa relative humidity anomaly.",
              "kind": "anom", "variable": "rhum", "level": 500,
              "show_wind": False, "plot_scale": 1.0,
              "vlim": 40.0, "cint": 8.0,
              "cb_label": "Relative Humidity Anomaly  (%)"},

    # ---- sea-level pressure anomaly ----
    "slp": {"id": "slp", "title": "Sea-Level Pressure Anomaly",
            "name": "SLP", "tag": "Surface",
            "desc": "MSLP anomaly — the classic surface pressure chart.",
            "kind": "anom", "variable": "slp", "level": None,
            "show_wind": False, "plot_scale": 1.0,
            "vlim": 35.0, "cint": 7.0,
            "cb_label": "Sea-Level Pressure Anomaly  (hPa)"},

    # ---- streamfunction + wave train at 500 & 850 ----
    "psi500": {"id": "psi500", "title": "Streamfunction Anomaly — 500 hPa",
               "name": "ψ500", "tag": "Mid",
               "desc": "500-hPa streamfunction anomaly (mid-tropospheric circulation centers).",
               "kind": "psi", "level": 500, "variables": ["uwnd", "vwnd"],
               "show_wind": False, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "psi850": {"id": "psi850", "title": "Streamfunction Anomaly — 850 hPa",
               "name": "ψ850", "tag": "Low",
               "desc": "850-hPa streamfunction anomaly (low-level circulation centers).",
               "kind": "psi", "level": 850, "variables": ["uwnd", "vwnd"],
               "show_wind": False, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rwt500": {"id": "rwt500", "title": "Rossby Wave Train Circulation — 500 hPa",
               "name": "Wave Train ψ500", "tag": "Mid",
               "desc": "500-hPa streamfunction anomaly + wind: mid-level Rossby wave train.",
               "kind": "psi", "level": 500, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},

    # ---- advanced diagnostics ----
    "waf200": {"id": "waf200",
               "title": "Wave Flux — 200 hPa",
               "name": "Wave Flux 200", "tag": "Advanced",
               "desc": "Takaya–Nakamura wave-activity flux vectors over 200-hPa "
                       "streamfunction anomaly (Rossby wave propagation source/sink).",
               "kind": "waf", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 400.0, "plot_scale": 1e-6,
               "vec_scale": 1e-4, "vec_ref": 50.0, "vec_unit": "5×10⁵ m²/s²",
               "vec_step": 5, "vec_min": 15.0,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "qgpv200": {"id": "qgpv200",
                "title": "QG Potential Vorticity Anomaly — 200 hPa",
                "name": "QG PV 200", "tag": "Advanced",
                "desc": "Quasi-geostrophic potential-vorticity anomaly at 200 hPa "
                        "(jet & wave-breaking diagnostics).",
                "kind": "qgpv", "level": 200, "variables": ["uwnd", "vwnd", "air"],
                "show_wind": False, "plot_scale": 1e6,
                "vlim": 320.0, "cint": 40.0,
                "cb_label": "QG PV Anomaly  (×10⁻⁶ s⁻¹)"},
    "eddy_vt": {"id": "eddy_vt",
                "title": "Eddy Meridional Flux v′T′ — 200 hPa",
                "name": "Eddy v′T′ 200", "tag": "Advanced",
                "desc": "Transient-eddy meridional heat flux v′T′ (deviation from "
                        "zonal mean of the anomaly) at 200 hPa.",
                "kind": "eddy", "level": 200, "variables": ["uwnd", "vwnd", "air"],
                "flux": "vt", "show_wind": False, "plot_scale": 1e-2,
                "vlim": 1.5, "cint": 0.25,
                "cb_label": "Eddy v′T′ Anomaly  (×10⁻² m s⁻¹ K)"},
    "eddy_uv": {"id": "eddy_uv",
                "title": "Eddy Momentum Flux u′v′ — 200 hPa",
                "name": "Eddy u′v′ 200", "tag": "Advanced",
                "desc": "Transient-eddy meridional momentum flux u′v′ at 200 hPa.",
                "kind": "eddy", "level": 200, "variables": ["uwnd", "vwnd", "air"],
                "flux": "uv", "show_wind": False, "plot_scale": 1e-2,
                "vlim": 4.0, "cint": 0.5,
                "cb_label": "Eddy u′v′  (×10⁻² m²/s²)"},
    "eady": {"id": "eady",
             "title": "Eady Baroclinic Growth Rate",
             "name": "Eady σ 850–500", "tag": "Advanced",
             "desc": "Eady baroclinic growth rate (850–500 hPa shear × static "
                     "stability); the 'storm-fuelling' instability index.",
             "kind": "eady", "level": 700, "variables": ["uwnd", "air"],
             "p_low": 850, "p_high": 500, "show_wind": False,
             "plot_scale": 1.0, "vlim": 1.2, "cint": 0.3,
             "cb_label": "Eady Growth Rate  (1/day)"},
}


# ================================================================
# Compute  -> returns (lat, lon, data)
#   data = {"main": 2D field (already scaled for display),
#           "u","v": optional wind-anomaly vectors (m/s)}
# ================================================================
def streamfunction_from_uv(u_anom, v_anom, lat, lon):
    """Streamfunction psi (m^2/s) from relative-vorticity inversion."""
    zeta = vorticity(u_anom, v_anom, lat, lon)
    return gaussian_filter(poisson_fft(zeta, lat, lon), sigma=2.0)


def _lerp_levels(levels, target):
    """Return (index, above_idx, below_idx) for a central-difference window."""
    arr = np.asarray(levels, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - target)))
    lo = max(0, idx - 1)
    hi = min(len(arr) - 1, idx + 1)
    if lo == hi:
        lo = max(0, idx - 1)
        hi = min(len(arr) - 1, idx + 1)
        if lo == hi:
            lo = idx
            hi = idx
    return idx, hi, lo


DEG_PER_S = 7.292e-5
KAPPA = 0.2854
P0 = 100000.0


def potential_temp(T, press_hpa):
    """theta = T (1000/p)^kappa."""
    return T * (1000.0 / press_hpa) ** KAPPA


def takaya_nakamura_flux(psi_anom, u_bar, v_bar, lat, lon, p_pa=20000.0, a=R_EARTH):
    """Horizontal Takaya-Nakamura (2001) wave-activity flux at pressure p_pa.

    W = (p cosφ / (2 |U| a²)) · ( U·A + V·B ,  U·B + V·C )

    with (λ,φ in radians, unitless derivatives — geometry lives in the prefactor):
        A = (∂ψ'/∂λ)² − ψ' ∂²ψ'/∂λ²
        B = (∂ψ'/∂λ)(∂ψ'/∂φ) − ψ' ∂²ψ'/∂λ∂φ
        C = (∂ψ'/∂φ)² − ψ' ∂²ψ'/∂φ²

    Basic-state wind is the slowly-varying background (zonal mean of the total
    wind). Result is in m²/s², direction = local group-velocity propagation.
    """
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)[:, None]
    ubar_z = zonal_mean(u_bar)
    ub = np.broadcast_to(ubar_z[:, None], psi_anom.shape)
    vb = zonal_mean(v_bar)[:, None]
    U = np.sqrt(ub**2 + vb**2) + 1e-8

    # NOTE: psi_anom is kept NaN-free. Placing NaN inside the field first makes
    # np.gradient blow up at the edges; the polar rows are handled by the output
    # mask below (cosφ→0 makes the inversion unreliable there anyway).

    dpsi_dlam = np.gradient(psi_anom, lam, axis=1)
    d2psi_dlam2 = np.gradient(dpsi_dlam, lam, axis=1)
    dpsi_dphi = np.gradient(psi_anom, phi, axis=0)
    d2psi_dphidlam = np.gradient(dpsi_dlam, phi, axis=0)
    d2psi_dphi2 = np.gradient(dpsi_dphi, phi, axis=0)

    A = dpsi_dlam**2 - psi_anom * d2psi_dlam2
    B = dpsi_dlam * dpsi_dphi - psi_anom * d2psi_dphidlam
    C = dpsi_dphi**2 - psi_anom * d2psi_dphi2

    pref = (p_pa * cosphi) / (2.0 * U * a**2)
    Wx = pref * (ub * A + vb * B)
    Wy = pref * (ub * B + vb * C)

    # Mask only the topmost rows where the streamfunction inversion is
    # genuinely degenerate as cosφ→0; keep the field out to ±80°.
    bad = np.abs(lat)[:, None] > 80.0
    Wx = np.where(bad, np.nan, Wx)
    Wy = np.where(bad, np.nan, Wy)
    return Wx, Wy


def zonal_mean(f):
    return np.nanmean(f, axis=1)


def _anom(var, level, dates, lat, lon):
    obs = _mean_field(var, level, dates, "obs")
    clim = _mean_field(var, level, dates, "clim")
    return gaussian_filter(obs - clim, sigma=1.5)


def _psi_level(level, dates):
    lat, lon = _latlon("uwnd")
    u = _mean_field("uwnd", level, dates, "obs")
    uc = _mean_field("uwnd", level, dates, "clim")
    v = _mean_field("vwnd", level, dates, "obs")
    vc = _mean_field("vwnd", level, dates, "clim")
    u_anom = gaussian_filter(u - uc, sigma=1.5)
    v_anom = gaussian_filter(v - vc, sigma=1.5)
    return streamfunction_from_uv(u_anom, v_anom, lat, lon)


def _mean_air(level, dates):
    obs = _mean_field("air", level, dates, "obs")
    clim = _mean_field("air", level, dates, "clim")
    return gaussian_filter(obs - clim, sigma=1.5)


def _temp_k(level, dates):
    """Mean absolute temperature (K) at a level for static-stability/theta maps."""
    return _mean_field("air", level, dates, "obs")


def _static_stability(level, dates):
    """s(p) = -alpha * dln(theta)/dp, evaluated with a 3-level centred difference
    using the absolute temperature profile."""
    idx, hi, lo = _lerp_levels(AAM_LEVELS, level)
    p_c, p_hi, p_lo = AAM_LEVELS[idx]*100.0, AAM_LEVELS[hi]*100.0, AAM_LEVELS[lo]*100.0
    Tc = _temp_k(AAM_LEVELS[idx], dates)
    Thi = _temp_k(AAM_LEVELS[hi], dates)
    Tlo = _temp_k(AAM_LEVELS[lo], dates)
    th_c = potential_temp(Tc, AAM_LEVELS[idx])
    th_hi = potential_temp(Thi, AAM_LEVELS[hi])
    th_lo = potential_temp(Tlo, AAM_LEVELS[lo])
    dlnth = (np.log(th_hi) - np.log(th_lo)) / (p_hi - p_lo)
    p_mid = (p_hi + p_lo) * 0.5
    T_mid = (Thi + Tlo) * 0.5
    alpha = 287.05 * T_mid / p_mid
    return -alpha * dlnth + 1e-9


def _laplacian(psi, lat, lon):
    """Spherical Laplacian ∇²ψ = (1/R²)[ ∂²ψ/∂φ² − tanφ ∂ψ/∂φ + (1/cos²φ) ∂²ψ/∂λ² ]."""
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    coslat = np.cos(phi)[:, None]
    tanlat = np.tan(phi)[:, None]
    d_phi = np.gradient(psi, phi, axis=0)
    d2phi = np.gradient(d_phi, phi, axis=0)
    d2lam = np.gradient(np.gradient(psi, lam, axis=1), lam, axis=1)
    return (d2phi - tanlat * d_phi + d2lam / coslat**2) / (R_EARTH**2)


def eady_growth(u_low, u_up, T_low, T_up, p_low, p_high, lat, a=R_EARTH):
    """Eady baroclinic growth rate sigma = 0.31 f |du/dz| / N  (1/day)."""
    phi = np.deg2rad(lat)
    f = 2 * DEG_PER_S * np.sin(phi)[:, None]
    g = 9.80665
    # vertical wind shear du/dp
    du_dp = (u_up - u_low) / (p_high - p_low)      # p_high > p_low (pressure coords)
    # rho ~ p/(R T) hydrostatic; du/dz = -rho*g*du/dp
    R = 287.05
    p_mean = (p_low + p_high) * 0.5
    T_mean = (T_low + T_up) * 0.5
    rho = p_mean / (R * T_mean)
    du_dz = -rho * g * du_dp
    # N^2 from theta: N2 = (g/theta) dtheta/dz = -g*rho*(g/theta) dtheta/dp
    th_low = potential_temp(T_low, p_low)
    th_up = potential_temp(T_up, p_high)
    dth_dp = (th_up - th_low) / (p_high - p_low)
    N2 = -g * g * rho * dth_dp / (th_low + th_up) * 2.0
    N2 = np.maximum(N2, 1e-8)
    sigma = 0.31 * np.abs(f) * np.abs(du_dz) / np.sqrt(N2)
    return sigma   # 1/s


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

    elif kind == "ft":
        # Frictional torque driver: surface zonal wind stress anomaly.
        # tau_x = rho * Cd * |V10| * u10  (bulk drag law), observed minus climatology.
        lat, lon = _latlon("uwnd.sfc")
        u_obs = _mean_field("uwnd.sfc", None, dates, "obs")
        u_clim = _mean_field("uwnd.sfc", None, dates, "clim")
        v_obs = _mean_field("vwnd.sfc", None, dates, "obs")
        v_clim = _mean_field("vwnd.sfc", None, dates, "clim")

        rho, cd = 1.225, 1.4e-3
        def stress(u, v):
            spd = np.sqrt(u * u + v * v)
            return rho * cd * spd * u
        tau_obs = stress(u_obs, v_obs)
        tau_clim = stress(u_clim, v_clim)
        anom = gaussian_filter(tau_obs - tau_clim, sigma=1.5) * pkg["plot_scale"]
        return lat, lon, {"main": anom}

    elif kind == "waf":
        # Takaya–Nakamura wave-activity flux at pkg["level"].
        lat, lon = _latlon("uwnd")
        u_obs = _mean_field("uwnd", pkg["level"], dates, "obs")
        u_clim = _mean_field("uwnd", pkg["level"], dates, "clim")
        v_obs = _mean_field("vwnd", pkg["level"], dates, "obs")
        v_clim = _mean_field("vwnd", pkg["level"], dates, "clim")
        u_anom = gaussian_filter(u_obs - u_clim, sigma=1.5)
        v_anom = gaussian_filter(v_obs - v_clim, sigma=1.5)
        psi = streamfunction_from_uv(u_anom, v_anom, lat, lon)

        # basic-state wind from the climatology (zonal-mean U, full V)
        u_basic = u_clim
        v_basic = v_clim
        waf_u, waf_v = takaya_nakamura_flux(psi, u_basic, v_basic, lat, lon,
                                            p_pa=pkg["level"] * 100.0)

        main = psi * pkg["plot_scale"]
        vec_sc = pkg["vec_scale"]
        # NaN mask must match the flux mask ({>80°} treated as NaN) so no
        # garbage arrows or shading are drawn right at the poles.
        return lat, lon, {"main": np.where(np.abs(lat)[:, None] <= 80.0,
                                           main, np.nan),
                          "vec_u": waf_u * vec_sc, "vec_v": waf_v * vec_sc}

    elif kind == "qgpv":
        # Quasi-geostrophic potential-vorticity anomaly:
        #   q' = lap_h(psi') + f^2 * d/dp[ (1/s) dpsi'/dp ]
        # with s = -alpha * dln(theta)/dp  (static stability, from the mean temperature).
        idx, hi, lo = _lerp_levels(AAM_LEVELS, pkg["level"])
        p_c = AAM_LEVELS[idx] * 100.0
        p_hi = AAM_LEVELS[hi] * 100.0
        p_lo = AAM_LEVELS[lo] * 100.0
        lat, lon = _latlon("uwnd")
        f = 2 * DEG_PER_S * np.sin(np.deg2rad(lat))[:, None]
        psi_c = _psi_level(pkg["level"], dates)
        psi_hi = _psi_level(AAM_LEVELS[hi], dates)
        psi_lo = _psi_level(AAM_LEVELS[lo], dates)
        lap_psi = _laplacian(psi_c, lat, lon)
        # vertical static stability at the level (absolute temperature)
        s_c = _static_stability(AAM_LEVELS[idx], dates)
        s_hi = _static_stability(AAM_LEVELS[hi], dates)
        s_lo = _static_stability(AAM_LEVELS[lo], dates)
        # (1/s)*dpsi/dp at the two sub-intervals (clip s to a realistic floor)
        s_hi_c = np.maximum(s_hi, 1e-6)
        s_lo_c = np.maximum(s_lo, 1e-6)
        g_up = (psi_hi - psi_c) / (p_hi - p_c) / s_hi_c
        g_dn = (psi_c - psi_lo) / (p_c - p_lo) / s_lo_c
        dg_dp = 2.0 * (g_up - g_dn) / (p_hi - p_lo)
        q = lap_psi + f**2 * dg_dp
        # The spherical Laplacian degenerates as cosφ→0; the pole rows are not
        # a real signal. Mask them so they render blank rather than saturating.
        q = np.where(np.abs(lat)[:, None] > 78.0, np.nan, q)
        return lat, lon, {"main": q * pkg["plot_scale"]}

    elif kind == "eddy":
        # Transient-eddy meridional fluxes v'T' and u'v' (deviation from zonal mean
        # of the anomaly fields) at pkg["level"].
        lat, lon = _latlon("uwnd")
        u_a = _anom("uwnd", pkg["level"], dates, lat, lon)
        v_a = _anom("vwnd", pkg["level"], dates, lat, lon)
        T_a = _anom("air", pkg["level"], dates, lat, lon)
        # deviation from zonal mean
        v_e = v_a - zonal_mean(v_a)[:, None]
        T_e = T_a - zonal_mean(T_a)[:, None]
        u_e = u_a - zonal_mean(u_a)[:, None]
        vt = v_e * T_e
        uv = u_e * v_e
        main = vt if pkg.get("flux") == "vt" else uv
        return lat, lon, {"main": main * pkg["plot_scale"],
                          "u": u_e, "v": v_e}

    elif kind == "eady":
        # Eady baroclinic growth rate (lower/mid troposphere) between two levels.
        # Uses the ABSOLUTE (observed) wind shear and temperature, so N^2 and the
        # growth rate are physically realistic (1/day).
        lat, lon = _latlon("uwnd")
        p_lo = pkg["p_low"]; p_hi = pkg["p_high"]
        u_low = _mean_field("uwnd", p_lo, dates, "obs")
        u_up = _mean_field("uwnd", p_hi, dates, "obs")
        T_low = _temp_k(p_lo, dates)
        T_up = _temp_k(p_hi, dates)
        sigma = eady_growth(u_low, u_up, T_low, T_up, p_lo*100.0, p_hi*100.0, lat)
        main = sigma * 86400.0 * pkg["plot_scale"]   # 1/day
        return lat, lon, {"main": main}

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


# ================================================================
# Map domains  (name -> lon_min, lon_max, lat_min, lat_max in 0-360)
# ================================================================
DOMAINS = {
    "global":      {"name": "Global",          "box": (0, 360, -80, 80)},
    "indian_ocean": {"name": "Indian Ocean",   "box": (40, 120, -50, 30)},
    "pacific":     {"name": "Pacific Ocean",   "box": (110, 260, -60, 55)},
    "atlantic":    {"name": "Atlantic Ocean",  "box": (300, 60, -60, 60)},
    "south_asia":  {"name": "South Asia",      "box": (60, 100, 5, 40)},
    "se_asia":     {"name": "Southeast Asia",  "box": (90, 140, -10, 30)},
    "east_asia":   {"name": "East Asia",       "box": (100, 150, 20, 55)},
    "europe":      {"name": "Europe",          "box": (340, 50, 30, 70)},
    "asia":        {"name": "Asia",            "box": (40, 160, -10, 70)},
}


def _domain_box(name):
    d = DOMAINS.get(name, DOMAINS["global"])
    return d["box"]


def render(lat, lon, data, pkg, coast_segs, dates, out_buf=None,
           title=None, cbar_label=None, domain="global"):
    fplot = data["main"]
    vlim, cint = pkg["vlim"], pkg["cint"]
    LON2D, LAT2D = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    # reserve a clean title band above the axes so the title never overlaps the map
    ax = fig.add_axes([0.045, 0.145, 0.910, 0.750])
    ax.set_facecolor("#f4f0e8")
    lon_min, lon_max, lat_min, lat_max = _domain_box(domain)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # lon ticks can wrap (e.g. Atlantic 300E..60E) -> normalise
    xticks = _domain_xticks(lon_min, lon_max)
    yticks = _domain_yticks(lat_min, lat_max)

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
    ticks = np.array([round(v, 8) for v in
                      np.arange(-vlim, vlim + 0.001, cint)])
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
             n_days=DEFAULT_N_DAYS, log=None, domain="global"):
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
    buf = render(lat, lon, data, pkg, coast_segs, dates, domain=domain)
    meta = {"product": pkg["id"], "title": pkg["title"], "domain": domain,
            "date_start": dates[0].isoformat(), "date_end": dates[-1].isoformat(),
            "n_days": len(dates), "level": pkg["level"]}
    return buf, meta


def generate_diff(product_id=DEFAULT_PRODUCT, date1=None, n_days1=DEFAULT_N_DAYS,
                  date2=None, n_days2=DEFAULT_N_DAYS, inverse=False, log=None,
                  domain="global"):
    """Return one map of (Range A − Range B), or (B − A) if inverse=True."""
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

    say("[3] Difference …")
    sign = -1.0 if inverse else 1.0     # A−B default; B−A when inverse
    data = {"main": sign * (data_a["main"] - data_b["main"])}
    if "u" in data_a and "u" in data_b:
        data["u"] = sign * (data_a["u"] - data_b["u"])
        data["v"] = sign * (data_a["v"] - data_b["v"])

    tag = "B − A" if inverse else "A − B"     # only shown on the colorbar, not the title
    # concise title: just the product (no operation tag, no long date string)
    title = pkg["title"]
    buf = render(lat, lon, data, pkg, coast_segs, dates_a,
                 title=title, cbar_label=pkg["cb_label"] + f"  ({tag})",
                 domain=domain)

    meta = {"product": pkg["id"], "title": pkg["title"],
            "date_start": dates_a[0].isoformat(), "date_end": dates_a[-1].isoformat(),
            "date_b_start": dates_b[0].isoformat(), "date_b_end": dates_b[-1].isoformat(),
            "n_days": len(dates_a), "level": pkg["level"], "diff": True, "inverse": inverse}
    return buf, meta
