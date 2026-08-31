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

    # ---- angular momentum budget: frictional torque (mountain torque needs
    #      orography, which this dataset does not carry — see README) ----
    "frict": {"id": "frict", "title": "Frictional Torque — Zonal (τx)",
              "name": "Friction τx", "tag": "Torque",
              "desc": "Surface zonal wind-stress anomaly (the zonal frictional-"
                      "torque driver) with the full surface stress vector, from "
                      "10-m winds via the bulk drag law.",
              "kind": "ft", "level": None, "variables": [],
              "comp": "x", "show_wind": True, "wind_scale": 55.0,
              "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0,
              "plot_scale": 100.0,
              "vlim": 30.0, "cint": 6.0,
              "cb_label": "Surface Zonal Stress Anomaly  (×10⁻² N/m²)"},
    "frict_y": {"id": "frict_y", "title": "Frictional Torque — Meridional (τy)",
                "name": "Friction τy", "tag": "Torque",
                "desc": "Surface meridional wind-stress anomaly (the meridional "
                        "frictional-torque driver) with the full stress vector.",
                "kind": "ft", "level": None, "variables": [],
                "comp": "y", "show_wind": True, "wind_scale": 55.0,
                "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0,
                "plot_scale": 100.0,
                "vlim": 18.0, "cint": 3.0,
                "cb_label": "Surface Meridional Stress Anomaly  (×10⁻² N/m²)"},
    "sstress": {"id": "sstress", "title": "Surface Wind Stress Magnitude (|τ|)",
                "name": "Stress |τ|", "tag": "Torque",
                "desc": "Magnitude of the surface wind-stress anomaly with the "
                        "stress vector — the full frictional-forcing field.",
                "kind": "ft", "level": None, "variables": [],
                "comp": "mag", "show_wind": True, "wind_scale": 55.0,
                "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0,
                "plot_scale": 100.0, "one_sided": True,
                "vlim": 30.0, "cint": 6.0,
                "cb_label": "Surface Stress Magnitude Anomaly  (×10⁻² N/m²)"},

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
    "srfp": {"id": "srfp", "title": "Surface Pressure Anomaly (ps)",
             "name": "ps", "tag": "Surface",
             "desc": "Daily surface-pressure anomaly — the real terrain-influenced "
                     "surface pressure field (not reduced to sea level).",
             "kind": "anom", "variable": "srfp", "level": None,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 25.0, "cint": 5.0,
             "cb_label": "Surface Pressure Anomaly  (hPa)"},

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

    # ---- integrated water vapour transport ----
    "ivt": {"id": "ivt", "title": "Integrated Water Vapour Transport",
            "name": "IVT", "tag": "Moisture",
            "desc": "Column-integrated water-vapour transport |∫q·V dp| — the "
                    "atmospheric-river 'moisture highway' map.",
            "kind": "ivt", "level": None, "variables": ["uwnd", "vwnd", "air", "rhum"],
            "show_wind": True, "wind_scale": 1400.0, "plot_scale": 1.0,
            "vec_scale": 1.0, "vec_ref": 400.0, "vec_unit": "400 kg m⁻¹ s⁻¹",
            "vec_step": 5, "vec_min": 80.0,
            "one_sided": True,
            "vlim": 400.0, "cint": 50.0,
            "cb_label": "Integrated Water Vapour Transport  (kg m⁻¹ s⁻¹)"},

    # ---- QG omega forcing ----
    "qgforcing500": {"id": "qgforcing500", "title": "QG Omega Forcing — 500 hPa",
                     "name": "QG ω-forcing 500", "tag": "Dynamics",
                     "desc": "Quasi-geostrophic omega forcing −2∇·Q (Hoskins "
                             "Q-vector): red = forced ascent, blue = descent.",
                     "kind": "qgforcing", "level": 500,
                     "variables": ["uwnd", "vwnd", "air", "hgt"],
                     "min_lat": 12.0, "show_wind": False,
                     "plot_scale": 1e12,
                     "vlim": 4.0, "cint": 1.0,
                     "cb_label": "QG Omega Forcing  (×10⁻¹² K m⁻² s⁻¹)"},

    # ---- moist static energy anomaly ----
    "mse850": {"id": "mse850", "title": "Moist Static Energy Anomaly — 850 hPa",
               "name": "MSE 850", "tag": "Thermo",
               "desc": "Moist Static Energy (Cp·T + Lv·q + g·z) anomaly at 850 hPa "
                       "— boundary-layer convective/energetics field.",
               "kind": "mse", "level": 850, "variables": ["air", "rhum", "hgt"],
               "show_wind": False, "plot_scale": 1e-3, "invert_cbar": True,
               "vlim": 12.0, "cint": 3.0,
               "cb_label": "MSE Anomaly  (×10³ J/kg)"},
    "mse500": {"id": "mse500", "title": "Moist Static Energy Anomaly — 500 hPa",
               "name": "MSE 500", "tag": "Thermo",
               "desc": "Moist Static Energy (Cp·T + Lv·q + g·z) anomaly at 500 hPa.",
               "kind": "mse", "level": 500, "variables": ["air", "rhum", "hgt"],
               "show_wind": False, "plot_scale": 1e-3, "invert_cbar": True,
               "vlim": 9.0, "cint": 2.0,
               "cb_label": "MSE Anomaly  (×10³ J/kg)"},

    # ---- temperature advection ----
    "tadv850": {"id": "tadv850", "title": "Temperature Advection — 850 hPa",
                "name": "T-adv 850", "tag": "Dynamics",
                "desc": "−V·∇T at 850 hPa (warm advection red, cold advection blue) "
                        "in K/day — the classic frontal/isentropic forcing map.",
                "kind": "tadv", "level": 850, "variables": ["uwnd", "vwnd", "air"],
                "show_wind": False, "plot_scale": 86400.0,
                "vlim": 8.0, "cint": 2.0,
                "cb_label": "Temperature Advection  (K/day)"},

    # ---- geostrophic / ageostrophic wind ----
    "geowind300": {"id": "geowind300", "title": "Geostrophic Wind — 300 hPa",
                   "name": "Geo-wind 300", "tag": "Flow",
                   "desc": "Geostrophic wind speed from the height field with the "
                           "geostrophic vector (equator masked, f→0).",
                   "kind": "geowind", "level": 300, "variables": ["hgt"],
                   "min_lat": 12.0, "show_wind": True, "wind_scale": 45.0,
                   "vec_ref": 20.0, "vec_unit": "20 m/s", "plot_scale": 1.0,
                   "vec_step": 5, "vec_min": 12.0, "one_sided": True,
                   "vlim": 90.0, "cint": 15.0,
                   "cb_label": "Geostrophic Wind Speed  (m/s)"},
    "ageowind300": {"id": "ageowind300", "title": "Ageostrophic Wind — 300 hPa",
                    "name": "Ageo-wind 300", "tag": "Flow",
                    "desc": "Ageostrophic wind (V − Vg) magnitude & vector at 300 hPa "
                            "— the divergent/accelerating part of the flow.",
                    "kind": "ageowind", "level": 300, "variables": ["uwnd", "vwnd", "hgt"],
                    "min_lat": 12.0, "show_wind": True, "wind_scale": 20.0,
                    "vec_ref": 5.0, "vec_unit": "5 m/s", "plot_scale": 1.0,
                    "vec_step": 5, "vec_min": 2.5, "one_sided": True,
                    "vlim": 20.0, "cint": 4.0,
                    "cb_label": "Ageostrophic Wind Speed  (m/s)"},

    # ---- Hovmöller diagrams (daily, latitude-band averaged, longitude–time) ----
    "hov_u850": {"id": "hov_u850", "title": "Zonal Wind 850 hPa",
                 "name": "Hovmöller U850", "tag": "Hovmöller",
                 "desc": "Longitude–time Hovmöller of the 850-hPa zonal-wind "
                         "anomaly averaged 5°S–5°N (equatorial waves / MJO).",
                 "kind": "hov", "variable": "uwnd", "level": 850,
                 "lat_band": (-5, 5), "window": 120,
                 "plot_scale": 1.0,
                 "vlim": 6.0, "cint": 1.0,
                 "cb_label": "Zonal Wind Anomaly 850 hPa  (m/s)"},
    "hov_chi200": {"id": "hov_chi200", "title": "Velocity Potential 200 hPa",
                   "name": "Hovmöller χ200", "tag": "Hovmöller",
                   "desc": "Longitude–time Hovmöller of the 200-hPa velocity-"
                           "potential anomaly averaged 15°S–15°N (convection "
                           "propagation / MJO).",
                   "kind": "hov", "variable": "chi", "level": 200,
                   "lat_band": (-15, 15), "window": 120,
                   "plot_scale": 1e-6,
                   "vlim": 5.0, "cint": 1.0,
                   "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},
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


# ================================================================
# Extra physics for the expanded catalogue
# ================================================================
CP = 1004.0        # J/(kg K) dry air
LV = 2.5e6         # J/kg latent heat of vapourisation
GRAV = 9.80665
RD = 287.05        # J/(kg K) gas constant dry air


def _sat_vp(T_k):
    """Saturation vapour pressure (Pa), Bolton (1980)."""
    Tc = T_k - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def _spec_hum(T_k, rh, p_pa):
    """Specific humidity q (kg/kg) from temperature, RH% and pressure (Pa)."""
    e = _sat_vp(T_k) * (np.clip(rh, 0.0, 100.0) / 100.0)
    e = np.minimum(e, 0.95 * p_pa)          # guard vs. supersaturation at low p
    return 0.622 * e / (p_pa - 0.378 * e)


def _q_level(level, dates, kind):
    """Specific humidity field (kg/kg) at a level, obs or climatology."""
    T = _mean_field("air", level, dates, kind)
    rh = _mean_field("rhum", level, dates, kind)
    return _spec_hum(T, rh, level * 100.0)


def _geopot(level, dates, kind="obs"):
    """Geopotential Phi = g·z (m²/s²) from the hgt (gpm) field."""
    return 9.80665 * _mean_field("hgt", level, dates, kind)


def _grad_x(a, lat, lon):
    """d/dx (eastward) of a 2-D field on the sphere."""
    lon_r = np.deg2rad(lon)
    coslat = np.cos(np.deg2rad(lat))[:, None]
    return np.gradient(a, lon_r, axis=1) / (R_EARTH * coslat)


def _grad_y(a, lat, lon):
    """d/dy (northward) of a 2-D field on the sphere."""
    lat_r = np.deg2rad(lat)
    return np.gradient(a, lat_r, axis=0) / R_EARTH


def _geo_wind(level, dates, lat, lon):
    """Geostrophic wind (m/s) from the geopotential field. f→0 near the equator
    is guarded (returns 0) so gradients stay finite; tropics are masked at render."""
    phi = _geopot(level, dates)
    f = 2 * DEG_PER_S * np.sin(np.deg2rad(lat))[:, None]
    dphidx = _grad_x(phi, lat, lon)
    dphidy = _grad_y(phi, lat, lon)
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(np.abs(f) > 1e-6, -dphidy / f, 0.0)
        vg = np.where(np.abs(f) > 1e-6, dphidx / f, 0.0)
    return ug, vg


def _qvector_forcing(level, dates, lat, lon):
    """QG omega forcing = −2∇·Q (Hoskins Q-vector form). Positive ⇒ ascent.
    Geostrophic wind from height field, temperature from air; Q-vector is built
    from the geostrophic deformation of the temperature gradient."""
    ug, vg = _geo_wind(level, dates, lat, lon)
    T = _mean_field("air", level, dates, "obs")
    dTdx = _grad_x(T, lat, lon)
    dTdy = _grad_y(T, lat, lon)
    dUgdx = _grad_x(ug, lat, lon); dVgdx = _grad_x(vg, lat, lon)
    dUgdy = _grad_y(ug, lat, lon); dVgdy = _grad_y(vg, lat, lon)
    sigma = _static_stability(level, dates)
    coef = RD / (sigma * level * 100.0) * 1.0
    Q1 = -coef * (dUgdx * dTdx + dVgdx * dTdy)
    Q2 = -coef * (dUgdy * dTdx + dVgdy * dTdy)
    divQ = _grad_x(Q1, lat, lon) + _grad_y(Q2, lat, lon)
    return -2.0 * divQ


def _temp_advection(level, dates, lat, lon):
    """−V·∇T (K/s), positive = warm advection, from absolute obs fields."""
    u = _mean_field("uwnd", level, dates, "obs")
    v = _mean_field("vwnd", level, dates, "obs")
    T = _mean_field("air", level, dates, "obs")
    dTdx = _grad_x(T, lat, lon)
    dTdy = _grad_y(T, lat, lon)
    return -(u * dTdx + v * dTdy)


# ================================================================
# Hovmöller helpers  (daily data, latitude-band averaged, time×longitude)
# ================================================================
def _daily_stack(var, level, dates, kind):
    """Daily per-day field (ntime, nlat, nlon) for obs or climatology."""
    key = (var, level, tuple(d.isoformat() for d in dates), "daily", kind)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]
    if kind == "obs":
        by_year = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        rows = []
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            lv = _level_idx(ds, level) if level is not None else 0
            for d in ydates:
                ti = _time_idx(ds, d)
                rows.append(_read_slice(ds, var, ti, lv))
    else:
        ds = _open(var)
        lv = _level_idx(ds, level) if level is not None else 0
        n = len(np.array(ds["time"][:]))
        rows = []
        for d in dates:
            ti = min(d.timetuple().tm_yday - 1, n - 1)
            rows.append(_read_slice(ds, var, ti, lv))
    arr = np.stack(rows, axis=0)
    _FIELD_CACHE[key] = arr
    return arr


def _band_axis(lat, lat_min, lat_max):
    """Indices of the latitude rows inside [lat_min, lat_max]."""
    return np.where((lat >= lat_min) & (lat <= lat_max))[0]


def _band_label(band):
    """'5°S–5°N' style label for a (lat_min, lat_max) band."""
    lo, hi = band
    f = lambda x: f"{abs(x):g}°{'S' if x < 0 else 'N'}"
    return f"averaged {f(lo)}–{f(hi)}"


def compute_hov(pkg, dates):
    """Approach: build a daily time×longitude Hovmöller for pkg['variable'] at
    pkg['level'], averaged over the latitude band pkg['lat_band'], over the last
    pkg['window'] days ending at the latest requested date. Returns
    (day_dates, lon, matrix) where matrix is (ntime, nlon) already scaled."""
    var = pkg["variable"]
    level = pkg["level"]
    lat_min, lat_max = pkg["lat_band"]
    window = int(pkg.get("window", 120))
    end = dates[-1]
    day_dates = [end - datetime.timedelta(days=i) for i in range(window)][::-1]

    lat, lon = _latlon("uwnd")          # shared 73×144 grid
    band = _band_axis(lat, lat_min, lat_max)

    if var == "chi":
        # velocity potential: per-day u,v anomaly -> divergence -> Poisson
        u = _daily_stack("uwnd", level, day_dates, "obs")
        uc = _daily_stack("uwnd", level, day_dates, "clim")
        v = _daily_stack("vwnd", level, day_dates, "obs")
        vc = _daily_stack("vwnd", level, day_dates, "clim")
        u_a = gaussian_filter(u - uc, sigma=1.2)
        v_a = gaussian_filter(v - vc, sigma=1.2)
        rows = []
        for i in range(day_dates.__len__()):
            div = divergence(u_a[i], v_a[i], lat, lon)
            chi = gaussian_filter(poisson_fft(div, lat, lon), sigma=2.0)
            rows.append(chi[band].mean(axis=0))
        matrix = np.stack(rows, axis=0) * pkg["plot_scale"]
    else:
        obs = _daily_stack(var, level, day_dates, "obs")
        clim = _daily_stack(var, level, day_dates, "clim")
        anom = gaussian_filter(obs - clim, sigma=1.2)
        matrix = anom[:, band, :].mean(axis=1) * pkg["plot_scale"]
        scale = 1.0
    # Remove the daily zonal mean so the eastward-propagating wave structure is
    # visible (the band-mean otherwise keeps a large global-mean baseline, e.g.
    # the planetary-scale chi component that would saturate the colour scale).
    if pkg.get("zonal_anom", True):
        matrix = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    return day_dates, lon, matrix


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
        # Frictional torque driver: surface wind stress anomaly via the bulk drag
        # law  tau = rho * Cd * |V10| * V10  (N/m²). All three flavours share the
        # computation; pkg["comp"] selects which scalar to show, and the full
        # (tau_x, tau_y) vector is always returned for the arrow overlay.
        lat, lon = _latlon("uwnd.sfc")
        u_obs = _mean_field("uwnd.sfc", None, dates, "obs")
        u_clim = _mean_field("uwnd.sfc", None, dates, "clim")
        v_obs = _mean_field("vwnd.sfc", None, dates, "obs")
        v_clim = _mean_field("vwnd.sfc", None, dates, "clim")

        rho, cd = 1.225, 1.4e-3
        def stress(u, v):
            spd = np.sqrt(u * u + v * v)
            tau_u = rho * cd * spd * u
            tau_v = rho * cd * spd * v
            return tau_u, tau_v
        tx_o, ty_o = stress(u_obs, v_obs)
        tx_c, ty_c = stress(u_clim, v_clim)
        tx = gaussian_filter(tx_o - tx_c, sigma=1.5)
        ty = gaussian_filter(ty_o - ty_c, sigma=1.5)

        comp = pkg.get("comp", "x")
        if comp == "y":
            main = ty
        elif comp == "mag":
            main = np.sqrt(tx**2 + ty**2)
        else:
            main = tx
        vs = pkg.get("vec_scale", 1.0)
        return lat, lon, {"main": main * pkg["plot_scale"],
                          "vec_u": tx * vs, "vec_v": ty * vs}

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

    elif kind == "ivt":
        # Integrated Water Vapour Transport (kg m⁻¹ s⁻¹): column integral of q·V.
        # Q = (1/g) ∫ q·(u,v) dp; magnitude is the standard atmospheric-river metric.
        # All 4 fields are pulled as multi-level stacks (one set of reads each),
        # then combined as arrays, so the whole computation is a handful of
        # dataset requests rather than ~50.
        # rhum is archived on only 6 levels (1000–300 hPa), so integrate the
        # moisture up to 300 hPa (the overwhelming fraction of PW sits below).
        levels = [1000, 850, 700, 500, 400, 300]
        lat, lon = _latlon("uwnd")
        u = _mean_multi("uwnd", levels, dates, "obs")
        v = _mean_multi("vwnd", levels, dates, "obs")
        T = _mean_multi("air", levels, dates, "obs")
        RH = _mean_multi("rhum", levels, dates, "obs")
        p_pa = np.array(levels, dtype=np.float64)[:, None, None] * 100.0
        q = _spec_hum(T, RH, p_pa)                       # (nlev, nlat, nlon)
        pt = p_pa[:, 0, 0]
        Qx = np.trapezoid(q * u, x=pt[::-1], axis=0) / GRAV
        Qy = np.trapezoid(q * v, x=pt[::-1], axis=0) / GRAV
        main = np.sqrt(Qx**2 + Qy**2) * pkg["plot_scale"]
        vs = pkg.get("vec_scale", 1.0)
        return lat, lon, {"main": main, "vec_u": Qx * vs, "vec_v": Qy * vs}

    elif kind == "qgforcing":
        # QG omega forcing = −2∇·Q at pkg["level"] (positive ⇒ ascent). Geostrophic
        # winds break down near the equator, so the deep tropics are masked.
        level = pkg["level"]
        lat, lon = _latlon("uwnd")
        forcing = _qvector_forcing(level, dates, lat, lon)
        min_lat = pkg.get("min_lat", 12.0)
        forcing = np.where(np.abs(lat)[:, None] < min_lat, np.nan, forcing)
        forcing = np.where(np.abs(lat)[:, None] > 80.0, np.nan, forcing)
        return lat, lon, {"main": forcing * pkg["plot_scale"]}

    elif kind == "mse":
        # Moist Static Energy anomaly at pkg["level"]:  MSE = Cp·T + Lv·q + gz.
        level = pkg["level"]
        lat, lon = _latlon("air")
        T_o = _mean_field("air", level, dates, "obs")
        T_c = _mean_field("air", level, dates, "clim")
        q_o = _q_level(level, dates, "obs")
        q_c = _q_level(level, dates, "clim")
        phi_o = _geopot(level, dates, "obs")
        phi_c = _geopot(level, dates, "clim")
        mse_o = CP * T_o + LV * q_o + phi_o
        mse_c = CP * T_c + LV * q_c + phi_c
        return lat, lon, {"main": (mse_o - mse_c) * pkg["plot_scale"]}

    elif kind == "tadv":
        # Temperature advection −V·∇T at pkg["level"], in K/s (×24 h scale to K/day).
        level = pkg["level"]
        lat, lon = _latlon("uwnd")
        ta = _temp_advection(level, dates, lat, lon)
        # the 1/cosφ term in dT/dx amplifies to noise in the high latitudes;
        # the meaningful frontal/advective signal sits in the mid-latitudes.
        ta = np.where(np.abs(lat)[:, None] > 68.0, np.nan, ta)
        return lat, lon, {"main": ta * pkg["plot_scale"]}

    elif kind in ("geowind", "ageowind"):
        # Geostrophic Vg from the height field, or ageostrophic V − Vg. Magnitude
        # is shaded; the vector is overlaid. Tropics masked (f→0, geostrophy fails).
        level = pkg["level"]
        lat, lon = _latlon("uwnd")
        ug, vg = _geo_wind(level, dates, lat, lon)
        if kind == "geowind":
            U0, V0 = ug, vg
        else:
            u = _mean_field("uwnd", level, dates, "obs")
            v = _mean_field("vwnd", level, dates, "obs")
            U0, V0 = u - ug, v - vg
        main = np.sqrt(U0**2 + V0**2)
        min_lat = pkg.get("min_lat", 12.0)
        # mask the deep tropics (f→0, geostrophy fails) and the polar rows where
        # the 1/cosφ derivative explodes — in BOTH the field and the vectors.
        bad = (np.abs(lat)[:, None] < min_lat) | (np.abs(lat)[:, None] > 78.0)
        main = np.where(bad, np.nan, main)
        return lat, lon, {"main": main * pkg["plot_scale"],
                          "vec_u": np.where(bad, np.nan, U0),
                          "vec_v": np.where(bad, np.nan, V0)}

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
# sidebar groups, in display order (any tag not listed goes last)
GROUP_ORDER = ["Upper", "Mid", "Low", "Dynamics", "Thermo",
               "Moisture", "Torque", "Flow", "Advanced", "Surface",
               "Hovmöller"]


def list_products():
    return [{"id": p["id"], "title": p["title"], "name": p["name"],
             "desc": p["desc"], "level": p["level"], "tag": p["tag"],
             "kind": p["kind"]}
            for p in PRODUCTS.values()]


def group_products():
    """Products grouped by tag, in GROUP_ORDER."""
    by_tag = {}
    for p in PRODUCTS.values():
        by_tag.setdefault(p["tag"], []).append(p)
    order = [t for t in GROUP_ORDER if t in by_tag] + \
            [t for t in by_tag if t not in GROUP_ORDER]
    out = []
    for tag in order:
        out.append({"tag": tag, "product_list": [
            {"id": p["id"], "title": p["title"], "name": p["name"],
             "desc": p["desc"], "level": p["level"], "tag": p["tag"],
             "kind": p["kind"]} for p in by_tag[tag]]})
    return out


def generate(product_id=DEFAULT_PRODUCT, mode="auto", manual_date=None,
             n_days=DEFAULT_N_DAYS, log=None):
    pkg = PRODUCTS.get(product_id, PRODUCTS[DEFAULT_PRODUCT])
    say = (lambda m: log.append(m)) if log is not None else (lambda m: None)

    dates = _resolve_dates(mode, manual_date, n_days)

    if pkg["kind"] == "hov":
        say(f"[0/3] {pkg['title']} | {pkg.get('window', 120)}-day Hovmöller")
        say("[1/3] Fetching daily obs & climatology (cached) …")
        day_dates, lon, matrix = compute_hov(pkg, dates)
        say(f"  band {pkg['lat_band']} · {matrix.shape[0]} days × {matrix.shape[1]} lon")
        say("[2/3] Rendering Hovmöller …")
        lat_lab = _band_label(pkg["lat_band"])
        buf = render_hov(day_dates, lon, matrix, pkg, lat_lab=lat_lab)
        meta = {"product": pkg["id"], "title": pkg["title"], "hov": True,
                "date_start": day_dates[0].isoformat(),
                "date_end": day_dates[-1].isoformat(),
                "n_days": len(day_dates), "level": pkg["level"]}
        return buf, meta

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
                  date2=None, n_days2=DEFAULT_N_DAYS, inverse=False, log=None):
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
                 title=title, cbar_label=pkg["cb_label"] + f"  ({tag})")

    meta = {"product": pkg["id"], "title": pkg["title"],
            "date_start": dates_a[0].isoformat(), "date_end": dates_a[-1].isoformat(),
            "date_b_start": dates_b[0].isoformat(), "date_b_end": dates_b[-1].isoformat(),
            "n_days": len(dates_a), "level": pkg["level"], "diff": True, "inverse": inverse}
    return buf, meta
