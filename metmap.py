"""
metmap.py — Fully data-driven map engine for NCEP/NCAR Reanalysis.
"""

import os, io, re, zipfile, datetime, warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shapefile, requests
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

_DS_CACHE, _FIELD_CACHE, _LATLON_CACHE = {}, {}, {}
_COAST = None

# ================================================================
# Coastline
# ================================================================
def ensure_coastline():
    if os.path.exists(SHP_PATH): return
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
            pts, parts = shape.points, list(shape.parts) + [len(shape.points)]
            segs.extend([np.array(pts[parts[i]:parts[i + 1]]) for i in range(len(shape.parts))])
        _COAST = segs
    return _COAST

# ================================================================
# OPeNDAP helpers (cached)
# ================================================================
_PVAR = {"uwnd.sfc": "uwnd", "vwnd.sfc": "vwnd", "air.sfc": "air", "rhum.sfc": "rhum"}

def _pvar(key): return _PVAR.get(key, key)

def _open(varname, year=None):
    url = f"{PSL}/{varname}.{f'{year}.nc' if year else 'day.ltm.1991-2020.nc'}"
    if url not in _DS_CACHE: _DS_CACHE[url] = open_url(url)
    return _DS_CACHE[url]

def _latlon(var):
    if var not in _LATLON_CACHE:
        ds = _open(var, year=2024)
        _LATLON_CACHE[var] = (np.array(ds["lat"][:]), np.array(ds["lon"][:]))
    return _LATLON_CACHE[var]

def _level_idx(ds, hPa):
    return int(np.argmin(np.abs(np.array(ds["level"][:]) - hPa)))

def _epoch(ds):
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
    return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else datetime.date(1800, 1, 1)

def _time_idx(ds, target):
    raw = np.array(ds["time"][:])
    scale = 1.0 / 24.0 if "hours" in ds["time"].attributes.get("units", "") else 1.0
    epoch = _epoch(ds)
    for i, t in enumerate(raw):
        d = epoch + datetime.timedelta(days=float(t) * scale)
        if (d.year, d.month, d.day) == (target.year, target.month, target.day): return i
    raise ValueError(f"Date {target} not found in dataset")

def _read_slice(ds, varname, t, lv):
    var = _pvar(varname)
    raw = np.array((ds[var][t, lv, :, :] if "level" in ds else ds[var][t, :, :]).data).squeeze().astype(np.float64)
    attr = ds[var].attributes
    sf, ao, mv = float(attr.get("scale_factor", 1.0)), float(attr.get("add_offset", 0.0)), float(attr.get("missing_value", 32767.0))
    fill_mask = np.abs(raw - mv) < 0.5
    data = raw * sf + ao
    data[fill_mask] = np.nan
    return data

# ================================================================
# Cached field mean (obs or climatology)
# ================================================================
def _mean_field(var, level, dates, kind):
    key = (var, level, tuple(d.isoformat() for d in dates), kind)
    if key in _FIELD_CACHE: return _FIELD_CACHE[key]
    slices = []
    if kind == "obs":
        by_year = {}
        for d in dates: by_year.setdefault(d.year, []).append(d)
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            lv = _level_idx(ds, level) if level is not None else 0
            slices.extend([_read_slice(ds, var, _time_idx(ds, d), lv) for d in ydates])
    else:
        ds = _open(var)
        lv = _level_idx(ds, level) if level is not None else 0
        n = len(np.array(ds["time"][:]))
        slices = [_read_slice(ds, var, min(d.timetuple().tm_yday - 1, n - 1), lv) for d in dates]

    mean = np.nanmean(slices, axis=0)
    _FIELD_CACHE[key] = mean
    return mean

def _mean_multi(var, levels, dates, kind):
    key = (var, "multi", tuple(levels), tuple(d.isoformat() for d in dates), kind)
    if key in _FIELD_CACHE: return _FIELD_CACHE[key]
    stack = []
    if kind == "obs":
        by_year = {}
        for d in dates: by_year.setdefault(d.year, []).append(d)
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            for lv in range(len(levels)):
                stack.append(np.nanmean([_read_slice(ds, var, _time_idx(ds, d), lv) for d in ydates], axis=0))
    else:
        ds = _open(var)
        n = len(np.array(ds["time"][:]))
        for lv in range(len(levels)):
            stack.append(np.nanmean([_read_slice(ds, var, min(d.timetuple().tm_yday - 1, n - 1), lv) for d in dates], axis=0))

    arr = np.stack(stack, axis=0)
    _FIELD_CACHE[key] = arr
    return arr

# ================================================================
# Physics
# ================================================================
R_EARTH = 6.371e6
AAM_LEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50]

def divergence(u, v, lat, lon):
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    coslat = np.cos(lat_r)
    dudx = np.gradient(u, lon_r, axis=1) / (R_EARTH * coslat[:, None])
    dvdy = np.gradient(v * coslat[:, None], lat_r, axis=0) / (R_EARTH * coslat[:, None])
    return dudx + dvdy

def vorticity(u, v, lat, lon):
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    coslat = np.cos(lat_r)[:, None]
    dudphi = np.gradient(u * coslat, lat_r, axis=0)
    dvdlon = np.gradient(v, lon_r, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(np.abs(coslat) > 1e-4, (1.0 / (R_EARTH * coslat)) * (dvdlon - dudphi), 0.0)
    zeta[0], zeta[-1] = zeta[1], zeta[-2]
    return zeta

def poisson_fft(rhs, lat, lon):
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    dy = R_EARTH * np.abs(np.mean(np.diff(lat_r)))
    dx_mean = R_EARTH * np.mean(np.diff(lon_r)) * np.mean(np.abs(np.cos(lat_r)))
    nlat, nlon = rhs.shape
    rhs_clean = np.nan_to_num(rhs, nan=0.0)
    taper = np.array([np.cos(np.deg2rad((abs(la) - 75.0) * 90.0 / 15.0)) ** 2 if abs(la) > 75.0 else 1.0 for la in lat])
    rhs_clean *= taper[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(nlon, d=dx_mean)
    ky = 2.0 * np.pi * np.fft.fftfreq(nlat, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    F = np.fft.fft2(rhs_clean) / -K2
    F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F))

# ================================================================
# Product registry
# ================================================================
PRODUCTS = {
    # ---- velocity potential + wind ----
    **{f"vtp{l}": {"id": f"vtp{l}", "title": f"Velocity Potential & Wind Anomaly — {l} hPa",
                   "name": f"χ{l} · Wind", "tag": tag, "desc": f"{l}-hPa velocity-potential and wind anomalies.",
                   "kind": "vtp", "level": l, "variables": ["uwnd", "vwnd"], "show_wind": True,
                   "wind_scale": 50.0, "plot_scale": 1e-6, "vlim": 10.0, "cint": 2.5,
                   "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"}
       for l, tag in [(200, "Upper"), (500, "Mid"), (850, "Low")]},

    # ---- streamfunction + Rossby wave train ----
    "psi200": {"id": "psi200", "title": "Streamfunction Anomaly — 200 hPa", "name": "ψ200", "tag": "Upper",
               "desc": "200-hPa streamfunction anomaly.", "kind": "psi", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": False, "plot_scale": 1e-6, "vlim": 40.0, "cint": 8.0, "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rwt200": {"id": "rwt200", "title": "Rossby Wave Train Circulation — 200 hPa", "name": "Wave Train ψ200", "tag": "Upper",
               "desc": "200-hPa streamfunction anomaly + wind.", "kind": "psi", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6, "vlim": 40.0, "cint": 8.0, "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},

    # ---- geopotential height anomaly ----
    **{f"hgt{l}": {"id": f"hgt{l}", "title": f"Geopotential Height Anomaly — {l} hPa", "name": f"H{l}", "tag": tag,
                   "desc": f"{l}-hPa geopotential height anomaly.", "kind": "anom", "variable": "hgt", "level": l,
                   "show_wind": False, "plot_scale": 1.0, "vlim": 150.0, "cint": 30.0, "cb_label": "Geopotential Height Anomaly  (gpm)"}
       for l, tag in [(200, "Upper"), (500, "Mid"), (850, "Low")]},

    # ---- zonal wind anomaly ----
    "u200": {"id": "u200", "title": "Zonal Wind Anomaly — 200 hPa", "name": "U200", "tag": "Upper", "desc": "200-hPa zonal wind anomaly.", "kind": "anom", "variable": "uwnd", "level": 200, "show_wind": False, "plot_scale": 1.0, "vlim": 15.0, "cint": 3.0, "cb_label": "Zonal Wind Anomaly  (m/s)"},
    "u850": {"id": "u850", "title": "Zonal Wind Anomaly — 850 hPa", "name": "U850", "tag": "Low", "desc": "850-hPa zonal wind anomaly.", "kind": "anom", "variable": "uwnd", "level": 850, "show_wind": False, "plot_scale": 1.0, "vlim": 10.0, "cint": 2.0, "cb_label": "Zonal Wind Anomaly  (m/s)"},

    # ---- temperature anomaly ----
    "t200": {"id": "t200", "title": "Temperature Anomaly — 200 hPa", "name": "T200", "tag": "Upper", "desc": "200-hPa temperature anomaly.", "kind": "anom", "variable": "air", "level": 200, "show_wind": False, "plot_scale": 1.0, "vlim": 6.0, "cint": 1.5, "cb_label": "Temperature Anomaly  (K)"},
    "t850": {"id": "t850", "title": "Temperature Anomaly — 850 hPa", "name": "T850", "tag": "Low", "desc": "850-hPa temperature anomaly.", "kind": "anom", "variable": "air", "level": 850, "show_wind": False, "plot_scale": 1.0, "vlim": 8.0, "cint": 2.0, "cb_label": "Temperature Anomaly  (K)"},

    # ---- angular momentum budget ----
    "frict": {"id": "frict", "title": "Frictional Torque — Zonal (τx)", "name": "Friction τx", "tag": "Torque", "desc": "Surface zonal wind-stress anomaly.", "kind": "ft", "level": None, "variables": [], "comp": "x", "show_wind": True, "wind_scale": 55.0, "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0, "plot_scale": 100.0, "vlim": 30.0, "cint": 6.0, "cb_label": "Surface Zonal Stress Anomaly  (×10⁻² N/m²)"},
    "frict_y": {"id": "frict_y", "title": "Frictional Torque — Meridional (τy)", "name": "Friction τy", "tag": "Torque", "desc": "Surface meridional wind-stress anomaly.", "kind": "ft", "level": None, "variables": [], "comp": "y", "show_wind": True, "wind_scale": 55.0, "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0, "plot_scale": 100.0, "vlim": 18.0, "cint": 3.0, "cb_label": "Surface Meridional Stress Anomaly  (×10⁻² N/m²)"},
    "sstress": {"id": "sstress", "title": "Surface Wind Stress Magnitude (|τ|)", "name": "Stress |τ|", "tag": "Torque", "desc": "Magnitude of surface wind-stress anomaly.", "kind": "ft", "level": None, "variables": [], "comp": "mag", "show_wind": True, "wind_scale": 55.0, "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0, "plot_scale": 100.0, "one_sided": True, "vlim": 30.0, "cint": 6.0, "cb_label": "Surface Stress Magnitude Anomaly  (×10⁻² N/m²)"},

    # ---- meridional wind anomaly ----
    "v200": {"id": "v200", "title": "Meridional Wind Anomaly — 200 hPa", "name": "V200", "tag": "Upper", "desc": "200-hPa meridional wind anomaly.", "kind": "anom", "variable": "vwnd", "level": 200, "show_wind": False, "plot_scale": 1.0, "vlim": 25.0, "cint": 5.0, "cb_label": "Meridional Wind Anomaly  (m/s)"},
    "v850": {"id": "v850", "title": "Meridional Wind Anomaly — 850 hPa", "name": "V850", "tag": "Low", "desc": "850-hPa meridional wind anomaly.", "kind": "anom", "variable": "vwnd", "level": 850, "show_wind": False, "plot_scale": 1.0, "vlim": 15.0, "cint": 3.0, "cb_label": "Meridional Wind Anomaly  (m/s)"},

    # ---- relative humidity anomaly ----
    **{f"rh{l}": {"id": f"rh{l}", "title": f"Relative Humidity Anomaly — {l} hPa", "name": f"RH{l}", "tag": tag, "desc": f"{l}-hPa relative humidity anomaly.", "kind": "anom", "variable": "rhum", "level": l, "show_wind": False, "plot_scale": 1.0, "vlim": 40.0, "cint": 8.0, "cb_label": "Relative Humidity Anomaly  (%)"}
       for l, tag in [(850, "Low"), (700, "Mid"), (500, "Mid")]},

    # ---- pressure anomalies ----
    "slp": {"id": "slp", "title": "Sea-Level Pressure Anomaly", "name": "SLP", "tag": "Surface", "desc": "MSLP anomaly.", "kind": "anom", "variable": "slp", "level": None, "show_wind": False, "plot_scale": 1.0, "vlim": 35.0, "cint": 7.0, "cb_label": "Sea-Level Pressure Anomaly  (hPa)"},
    "srfp": {"id": "srfp", "title": "Surface Pressure Anomaly (ps)", "name": "ps", "tag": "Surface", "desc": "Daily surface-pressure anomaly.", "kind": "anom", "variable": "srfp", "level": None, "show_wind": False, "plot_scale": 1.0, "vlim": 25.0, "cint": 5.0, "cb_label": "Surface Pressure Anomaly  (hPa)"},

    # ---- streamfunction at 500 & 850 ----
    "psi500": {"id": "psi500", "title": "Streamfunction Anomaly — 500 hPa", "name": "ψ500", "tag": "Mid", "desc": "500-hPa streamfunction anomaly.", "kind": "psi", "level": 500, "variables": ["uwnd", "vwnd"], "show_wind": False, "plot_scale": 1e-6, "vlim": 40.0, "cint": 8.0, "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "psi850": {"id": "psi850", "title": "Streamfunction Anomaly — 850 hPa", "name": "ψ850", "tag": "Low", "desc": "850-hPa streamfunction anomaly.", "kind": "psi", "level": 850, "variables": ["uwnd", "vwnd"], "show_wind": False, "plot_scale": 1e-6, "vlim": 40.0, "cint": 8.0, "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rwt500": {"id": "rwt500", "title": "Rossby Wave Train Circulation — 500 hPa", "name": "Wave Train ψ500", "tag": "Mid", "desc": "500-hPa streamfunction anomaly + wind.", "kind": "psi", "level": 500, "variables": ["uwnd", "vwnd"], "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6, "vlim": 40.0, "cint": 8.0, "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},

    # ---- advanced diagnostics ----
    "waf200": {"id": "waf200", "title": "Wave Flux — 200 hPa", "name": "Wave Flux 200", "tag": "Advanced", "desc": "Takaya–Nakamura wave-activity flux vectors.", "kind": "waf", "level": 200, "variables": ["uwnd", "vwnd"], "show_wind": True, "wind_scale": 400.0, "plot_scale": 1e-6, "vec_scale": 1e-4, "vec_ref": 50.0, "vec_unit": "5×10⁵ m²/s²", "vec_step": 5, "vec_min": 15.0, "vlim": 40.0, "cint": 8.0, "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "qgpv200": {"id": "qgpv200", "title": "QG Potential Vorticity Anomaly — 200 hPa", "name": "QG PV 200", "tag": "Advanced", "desc": "Quasi-geostrophic PV anomaly.", "kind": "qgpv", "level": 200, "variables": ["uwnd", "vwnd", "air"], "show_wind": False, "plot_scale": 1e6, "vlim": 320.0, "cint": 40.0, "cb_label": "QG PV Anomaly  (×10⁻⁶ s⁻¹)"},
    "eddy_vt": {"id": "eddy_vt", "title": "Eddy Meridional Flux v′T′ — 200 hPa", "name": "Eddy v′T′ 200", "tag": "Advanced", "desc": "Transient-eddy heat flux v′T′.", "kind": "eddy", "level": 200, "variables": ["uwnd", "vwnd", "air"], "flux": "vt", "show_wind": False, "plot_scale": 1e-2, "vlim": 1.5, "cint": 0.25, "cb_label": "Eddy v′T′ Anomaly  (×10⁻² m s⁻¹ K)"},
    "eddy_uv": {"id": "eddy_uv", "title": "Eddy Momentum Flux u′v′ — 200 hPa", "name": "Eddy u′v′ 200", "tag": "Advanced", "desc": "Transient-eddy momentum flux u′v′.", "kind": "eddy", "level": 200, "variables": ["uwnd", "vwnd", "air"], "flux": "uv", "show_wind": False, "plot_scale": 1e-2, "vlim": 4.0, "cint": 0.5, "cb_label": "Eddy u′v′  (×10⁻² m²/s²)"},
    "eady": {"id": "eady", "title": "Eady Baroclinic Growth Rate", "name": "Eady σ 850–500", "tag": "Advanced", "desc": "Eady growth rate.", "kind": "eady", "level": 700, "variables": ["uwnd", "air"], "p_low": 850, "p_high": 500, "show_wind": False, "plot_scale": 1.0, "vlim": 1.2, "cint": 0.3, "cb_label": "Eady Growth Rate  (1/day)"},

    # ---- moisture & dynamics ----
    "ivt": {"id": "ivt", "title": "Integrated Water Vapour Transport", "name": "IVT", "tag": "Moisture", "desc": "Column-integrated water-vapour transport.", "kind": "ivt", "level": None, "variables": ["uwnd", "vwnd", "air", "rhum"], "show_wind": True, "wind_scale": 1400.0, "plot_scale": 1.0, "vec_scale": 1.0, "vec_ref": 400.0, "vec_unit": "400 kg m⁻¹ s⁻¹", "vec_step": 5, "vec_min": 80.0, "one_sided": True, "vlim": 400.0, "cint": 50.0, "cb_label": "Integrated Water Vapour Transport  (kg m⁻¹ s⁻¹)"},
    "qgforcing500": {"id": "qgforcing500", "title": "QG Omega Forcing — 500 hPa", "name": "QG ω-forcing 500", "tag": "Dynamics", "desc": "QG omega forcing −2∇·Q.", "kind": "qgforcing", "level": 500, "variables": ["uwnd", "vwnd", "air", "hgt"], "min_lat": 12.0, "show_wind": False, "plot_scale": 1e12, "vlim": 4.0, "cint": 1.0, "cb_label": "QG Omega Forcing  (×10⁻¹² K m⁻² s⁻¹)"},

    # ---- moist static energy ----
    "mse850": {"id": "mse850", "title": "Moist Static Energy Anomaly — 850 hPa", "name": "MSE 850", "tag": "Thermo", "desc": "MSE anomaly at 850 hPa.", "kind": "mse", "level": 850, "variables": ["air", "rhum", "hgt"], "show_wind": False, "plot_scale": 1e-3, "invert_cbar": True, "vlim": 12.0, "cint": 3.0, "cb_label": "MSE Anomaly  (×10³ J/kg)"},
    "mse500": {"id": "mse500", "title": "Moist Static Energy Anomaly — 500 hPa", "name": "MSE 500", "tag": "Thermo", "desc": "MSE anomaly at 500 hPa.", "kind": "mse", "level": 500, "variables": ["air", "rhum", "hgt"], "show_wind": False, "plot_scale": 1e-3, "invert_cbar": True, "vlim": 9.0, "cint": 2.0, "cb_label": "MSE Anomaly  (×10³ J/kg)"},

    # ---- temperature advection & geostrophic wind ----
    "tadv850": {"id": "tadv850", "title": "Temperature Advection — 850 hPa", "name": "T-adv 850", "tag": "Dynamics", "desc": "−V·∇T at 850 hPa.", "kind": "tadv", "level": 850, "variables": ["uwnd", "vwnd", "air"], "show_wind": False, "plot_scale": 86400.0, "vlim": 8.0, "cint": 2.0, "cb_label": "Temperature Advection  (K/day)"},
    "geowind300": {"id": "geowind300", "title": "Geostrophic Wind — 300 hPa", "name": "Geo-wind 300", "tag": "Flow", "desc": "Geostrophic wind speed.", "kind": "geowind", "level": 300, "variables": ["hgt"], "min_lat": 12.0, "show_wind": True, "wind_scale": 45.0, "vec_ref": 20.0, "vec_unit": "20 m/s", "plot_scale": 1.0, "vec_step": 5, "vec_min": 12.0, "one_sided": True, "vlim": 90.0, "cint": 15.0, "cb_label": "Geostrophic Wind Speed  (m/s)"},
    "ageowind300": {"id": "ageowind300", "title": "Ageostrophic Wind — 300 hPa", "name": "Ageo-wind 300", "tag": "Flow", "desc": "Ageostrophic wind magnitude.", "kind": "ageowind", "level": 300, "variables": ["uwnd", "vwnd", "hgt"], "min_lat": 12.0, "show_wind": True, "wind_scale": 20.0, "vec_ref": 5.0, "vec_unit": "5 m/s", "plot_scale": 1.0, "vec_step": 5, "vec_min": 2.5, "one_sided": True, "vlim": 20.0, "cint": 4.0, "cb_label": "Ageostrophic Wind Speed  (m/s)"},
}

# ================================================================
# Compute helpers
# ================================================================
def streamfunction_from_uv(u_anom, v_anom, lat, lon):
    return gaussian_filter(poisson_fft(vorticity(u_anom, v_anom, lat, lon), lat, lon), sigma=2.0)

def _lerp_levels(levels, target):
    arr = np.asarray(levels, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - target)))
    lo, hi = max(0, idx - 1), min(len(arr) - 1, idx + 1)
    return idx, hi, lo

DEG_PER_S, KAPPA, P0, CP, LV, GRAV, RD = 7.292e-5, 0.2854, 100000.0, 1004.0, 2.5e6, 9.80665, 287.05

def potential_temp(T, press_hpa): return T * (1000.0 / press_hpa) ** KAPPA

def takaya_nakamura_flux(psi_anom, u_bar, v_bar, lat, lon, p_pa=20000.0, a=R_EARTH):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    cosphi = np.cos(phi)[:, None]
    ub = np.broadcast_to(zonal_mean(u_bar)[:, None], psi_anom.shape)
    vb = zonal_mean(v_bar)[:, None]
    U = np.sqrt(ub**2 + vb**2) + 1e-8

    dpsi_dlam = np.gradient(psi_anom, lam, axis=1)
    d2psi_dlam2 = np.gradient(dpsi_dlam, lam, axis=1)
    dpsi_dphi = np.gradient(psi_anom, phi, axis=0)
    d2psi_dphidlam = np.gradient(dpsi_dlam, phi, axis=0)
    d2psi_dphi2 = np.gradient(dpsi_dphi, phi, axis=0)

    A = dpsi_dlam**2 - psi_anom * d2psi_dlam2
    B = dpsi_dlam * dpsi_dphi - psi_anom * d2psi_dphidlam
    C = dpsi_dphi**2 - psi_anom * d2psi_dphi2

    pref = (p_pa * cosphi) / (2.0 * U * a**2)
    Wx, Wy = pref * (ub * A + vb * B), pref * (ub * B + vb * C)
    bad = np.abs(lat)[:, None] > 80.0
    return np.where(bad, np.nan, Wx), np.where(bad, np.nan, Wy)

def zonal_mean(f): return np.nanmean(f, axis=1)

def _anom(var, level, dates, lat, lon):
    return gaussian_filter(_mean_field(var, level, dates, "obs") - _mean_field(var, level, dates, "clim"), sigma=1.5)

def _psi_level(level, dates):
    lat, lon = _latlon("uwnd")
    u_anom = gaussian_filter(_mean_field("uwnd", level, dates, "obs") - _mean_field("uwnd", level, dates, "clim"), sigma=1.5)
    v_anom = gaussian_filter(_mean_field("vwnd", level, dates, "obs") - _mean_field("vwnd", level, dates, "clim"), sigma=1.5)
    return streamfunction_from_uv(u_anom, v_anom, lat, lon)

def _temp_k(level, dates): return _mean_field("air", level, dates, "obs")

def _static_stability(level, dates):
    idx, hi, lo = _lerp_levels(AAM_LEVELS, level)
    p_c, p_hi, p_lo = [AAM_LEVELS[i]*100.0 for i in (idx, hi, lo)]
    Tc, Thi, Tlo = [_temp_k(AAM_LEVELS[i], dates) for i in (idx, hi, lo)]
    th_hi, th_lo = potential_temp(Thi, AAM_LEVELS[hi]), potential_temp(Tlo, AAM_LEVELS[lo])
    dlnth = (np.log(th_hi) - np.log(th_lo)) / (p_hi - p_lo)
    return -(287.05 * ((Thi + Tlo) * 0.5) / ((p_hi + p_lo) * 0.5)) * dlnth + 1e-9

def _laplacian(psi, lat, lon):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    coslat, tanlat = np.cos(phi)[:, None], np.tan(phi)[:, None]
    d_phi = np.gradient(psi, phi, axis=0)
    return (np.gradient(d_phi, phi, axis=0) - tanlat * d_phi + np.gradient(np.gradient(psi, lam, axis=1), lam, axis=1) / coslat**2) / (R_EARTH**2)

def eady_growth(u_low, u_up, T_low, T_up, p_low, p_high, lat):
    f = 2 * DEG_PER_S * np.sin(np.deg2rad(lat))[:, None]
    rho = ((p_low + p_high) * 0.5) / (287.05 * ((T_low + T_up) * 0.5))
    du_dz = -rho * GRAV * ((u_up - u_low) / (p_high - p_low))
    th_low, th_up = potential_temp(T_low, p_low), potential_temp(T_up, p_high)
    N2 = np.maximum(-GRAV * GRAV * rho * ((th_up - th_low) / (p_high - p_low)) / (th_low + th_up) * 2.0, 1e-8)
    return 0.31 * np.abs(f) * np.abs(du_dz) / np.sqrt(N2)

def _sat_vp(T_k):
    Tc = T_k - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))

def _spec_hum(T_k, rh, p_pa):
    e = np.minimum(_sat_vp(T_k) * (np.clip(rh, 0.0, 100.0) / 100.0), 0.95 * p_pa)
    return 0.622 * e / (p_pa - 0.378 * e)

def _q_level(level, dates, kind):
    return _spec_hum(_mean_field("air", level, dates, kind), _mean_field("rhum", level, dates, kind), level * 100.0)

def _geopot(level, dates, kind="obs"):
    return 9.80665 * _mean_field("hgt", level, dates, kind)

def _grad_x(a, lat, lon): return np.gradient(a, np.deg2rad(lon), axis=1) / (R_EARTH * np.cos(np.deg2rad(lat))[:, None])
def _grad_y(a, lat, lon): return np.gradient(a, np.deg2rad(lat), axis=0) / R_EARTH

def _geo_wind(level, dates, lat, lon):
    phi = _geopot(level, dates)
    f = 2 * DEG_PER_S * np.sin(np.deg2rad(lat))[:, None]
    dphidx, dphidy = _grad_x(phi, lat, lon), _grad_y(phi, lat, lon)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(f) > 1e-6, -dphidy / f, 0.0), np.where(np.abs(f) > 1e-6, dphidx / f, 0.0)

def _qvector_forcing(level, dates, lat, lon):
    ug, vg = _geo_wind(level, dates, lat, lon)
    T = _mean_field("air", level, dates, "obs")
    dTdx, dTdy = _grad_x(T, lat, lon), _grad_y(T, lat, lon)
    dUgdx, dVgdx, dUgdy, dVgdy = _grad_x(ug, lat, lon), _grad_x(vg, lat, lon), _grad_y(ug, lat, lon), _grad_y(vg, lat, lon)
    coef = RD / (_static_stability(level, dates) * level * 100.0)
    Q1, Q2 = -coef * (dUgdx * dTdx + dVgdx * dTdy), -coef * (dUgdy * dTdx + dVgdy * dTdy)
    return -2.0 * (_grad_x(Q1, lat, lon) + _grad_y(Q2, lat, lon))

def _temp_advection(level, dates, lat, lon):
    T = _mean_field("air", level, dates, "obs")
    return -(_mean_field("uwnd", level, dates, "obs") * _grad_x(T, lat, lon) + _mean_field("vwnd", level, dates, "obs") * _grad_y(T, lat, lon))

def compute(pkg, dates):
    kind = pkg["kind"]

    if kind in ("vtp", "psi"):
        lat, lon = _latlon("uwnd")
        u_anom = gaussian_filter(_mean_field("uwnd", pkg["level"], dates, "obs") - _mean_field("uwnd", pkg["level"], dates, "clim"), sigma=1.5)
        v_anom = gaussian_filter(_mean_field("vwnd", pkg["level"], dates, "obs") - _mean_field("vwnd", pkg["level"], dates, "clim"), sigma=1.5)
        field = divergence(u_anom, v_anom, lat, lon) if kind == "vtp" else vorticity(u_anom, v_anom, lat, lon)
        main = gaussian_filter(poisson_fft(field, lat, lon), sigma=2.0) * pkg["plot_scale"]
        return lat, lon, {"main": main, "u": u_anom, "v": v_anom}

    elif kind == "ft":
        lat, lon = _latlon("uwnd.sfc")
        u_o, u_c = _mean_field("uwnd.sfc", None, dates, "obs"), _mean_field("uwnd.sfc", None, dates, "clim")
        v_o, v_c = _mean_field("vwnd.sfc", None, dates, "obs"), _mean_field("vwnd.sfc", None, dates, "clim")
        rho, cd = 1.225, 1.4e-3
        stress = lambda u, v: (rho * cd * np.sqrt(u**2 + v**2) * u, rho * cd * np.sqrt(u**2 + v**2) * v)
        tx_o, ty_o = stress(u_o, v_o)
        tx_c, ty_c = stress(u_c, v_c)
        tx, ty = gaussian_filter(tx_o - tx_c, sigma=1.5), gaussian_filter(ty_o - ty_c, sigma=1.5)
        comp = pkg.get("comp", "x")
        main = ty if comp == "y" else (np.sqrt(tx**2 + ty**2) if comp == "mag" else tx)
        vs = pkg.get("vec_scale", 1.0)
        return lat, lon, {"main": main * pkg["plot_scale"], "vec_u": tx * vs, "vec_v": ty * vs}

    elif kind == "waf":
        lat, lon = _latlon("uwnd")
        u_o, u_c = _mean_field("uwnd", pkg["level"], dates, "obs"), _mean_field("uwnd", pkg["level"], dates, "clim")
        v_o, v_c = _mean_field("vwnd", pkg["level"], dates, "obs"), _mean_field("vwnd", pkg["level"], dates, "clim")
        psi = streamfunction_from_uv(gaussian_filter(u_o - u_c, sigma=1.5), gaussian_filter(v_o - v_c, sigma=1.5), lat, lon)
        waf_u, waf_v = takaya_nakamura_flux(psi, u_c, v_c, lat, lon, p_pa=pkg["level"] * 100.0)
        main = np.where(np.abs(lat)[:, None] <= 80.0, psi * pkg["plot_scale"], np.nan)
        return lat, lon, {"main": main, "vec_u": waf_u * pkg["vec_scale"], "vec_v": waf_v * pkg["vec_scale"]}

    elif kind == "qgpv":
        idx, hi, lo = _lerp_levels(AAM_LEVELS, pkg["level"])
        p_c, p_hi, p_lo = [AAM_LEVELS[i] * 100.0 for i in (idx, hi, lo)]
        lat, lon = _latlon("uwnd")
        f = 2 * DEG_PER_S * np.sin(np.deg2rad(lat))[:, None]
        psi_c, psi_hi, psi_lo = [_psi_level(AAM_LEVELS[i], dates) for i in (idx, hi, lo)]
        s_hi, s_lo = [np.maximum(_static_stability(AAM_LEVELS[i], dates), 1e-6) for i in (hi, lo)]
        g_up, g_dn = (psi_hi - psi_c) / (p_hi - p_c) / s_hi, (psi_c - psi_lo) / (p_c - p_lo) / s_lo
        q = np.where(np.abs(lat)[:, None] > 78.0, np.nan, _laplacian(psi_c, lat, lon) + f**2 * (2.0 * (g_up - g_dn) / (p_hi - p_lo)))
        return lat, lon, {"main": q * pkg["plot_scale"]}

    elif kind == "eddy":
        lat, lon = _latlon("uwnd")
        u_a, v_a, T_a = [_anom(v, pkg["level"], dates, lat, lon) for v in ("uwnd", "vwnd", "air")]
        u_e, v_e, T_e = u_a - zonal_mean(u_a)[:, None], v_a - zonal_mean(v_a)[:, None], T_a - zonal_mean(T_a)[:, None]
        main = v_e * T_e if pkg.get("flux") == "vt" else u_e * v_e
        return lat, lon, {"main": main * pkg["plot_scale"], "u": u_e, "v": v_e}

    elif kind == "eady":
        lat, lon = _latlon("uwnd")
        p_lo, p_hi = pkg["p_low"], pkg["p_high"]
        sigma = eady_growth(_mean_field("uwnd", p_lo, dates, "obs"), _mean_field("uwnd", p_hi, dates, "obs"),
                            _temp_k(p_lo, dates), _temp_k(p_hi, dates), p_lo*100.0, p_hi*100.0, lat)
        return lat, lon, {"main": sigma * 86400.0 * pkg["plot_scale"]}

    elif kind == "ivt":
        levels = [1000, 850, 700, 500, 400, 300]
        lat, lon = _latlon("uwnd")
        u, v = _mean_multi("uwnd", levels, dates, "obs"), _mean_multi("vwnd", levels, dates, "obs")
        q = _spec_hum(_mean_multi("air", levels, dates, "obs"), _mean_multi("rhum", levels, dates, "obs"), np.array(levels, dtype=np.float64)[:, None, None] * 100.0)
        pt = np.array(levels, dtype=np.float64) * 100.0
        Qx, Qy = np.trapezoid(q * u, x=pt[::-1], axis=0) / GRAV, np.trapezoid(q * v, x=pt[::-1], axis=0) / GRAV
        vs = pkg.get("vec_scale", 1.0)
        return lat, lon, {"main": np.sqrt(Qx**2 + Qy**2) * pkg["plot_scale"], "vec_u": Qx * vs, "vec_v": Qy * vs}

    elif kind == "qgforcing":
        lat, lon = _latlon("uwnd")
        forcing = _qvector_forcing(pkg["level"], dates, lat, lon)
        bad = (np.abs(lat)[:, None] < pkg.get("min_lat", 12.0)) | (np.abs(lat)[:, None] > 80.0)
        return lat, lon, {"main": np.where(bad, np.nan, forcing) * pkg["plot_scale"]}

    elif kind == "mse":
        lat, lon = _latlon("air")
        l = pkg["level"]
        mse = lambda k: CP * _mean_field("air", l, dates, k) + LV * _q_level(l, dates, k) + _geopot(l, dates, k)
        return lat, lon, {"main": (mse("obs") - mse("clim")) * pkg["plot_scale"]}

    elif kind == "tadv":
        lat, lon = _latlon("uwnd")
        ta = np.where(np.abs(lat)[:, None] > 68.0, np.nan, _temp_advection(pkg["level"], dates, lat, lon))
        return lat, lon, {"main": ta * pkg["plot_scale"]}

    elif kind in ("geowind", "ageowind"):
        lat, lon = _latlon("uwnd")
        l = pkg["level"]
        ug, vg = _geo_wind(l, dates, lat, lon)
        if kind == "geowind": U0, V0 = ug, vg
        else: U0, V0 = _mean_field("uwnd", l, dates, "obs") - ug, _mean_field("vwnd", l, dates, "obs") - vg
        bad = (np.abs(lat)[:, None] < pkg.get("min_lat", 12.0)) | (np.abs(lat)[:, None] > 78.0)
        return lat, lon, {"main": np.where(bad, np.nan, np.sqrt(U0**2 + V0**2)) * pkg["plot_scale"],
                          "vec_u": np.where(bad, np.nan, U0), "vec_v": np.where(bad, np.nan, V0)}

    else:
        var = pkg["variable"]
        lat, lon = _latlon(var)
        return lat, lon, {"main": _anom(var, pkg["level"], dates, lat, lon) * pkg["plot_scale"]}

# ================================================================
# Generic renderer
# ================================================================
def _chi_cmap():
    cdict = {
        "red":   [(0.0, 0.08, 0.08), (0.35, 0.40, 0.40), (0.50, 0.97, 0.97), (0.65, 0.92, 0.92), (1.0, 0.55, 0.55)],
        "green": [(0.0, 0.38, 0.38), (0.35, 0.72, 0.72), (0.50, 0.97, 0.97), (0.65, 0.78, 0.78), (1.0, 0.30, 0.30)],
        "blue":  [(0.0, 0.45, 0.45), (0.35, 0.78, 0.78), (0.50, 0.97, 0.97), (0.65, 0.52, 0.52), (1.0, 0.10, 0.10)],
    }
    return LinearSegmentedColormap("chi_cmap", cdict, N=512)

def _chi_cmap_inv(): return _chi_cmap().reversed()

def _pos_cmap():
    cdict = {
        "red":   [(0.0, 0.97, 0.97), (0.40, 0.72, 0.72), (0.70, 0.30, 0.30), (1.0, 0.02, 0.02)],
        "green": [(0.0, 0.97, 0.97), (0.40, 0.90, 0.90), (0.70, 0.68, 0.68), (1.0, 0.40, 0.40)],
        "blue":  [(0.0, 0.97, 0.97), (0.40, 0.82, 0.82), (0.70, 0.45, 0.45), (1.0, 0.20, 0.20)],
    }
    return LinearSegmentedColormap("pos_cmap", cdict, N=512)

def _xlabel(v): return "0°" if v in (0, 360) else ("180°" if v == 180 else f"{v}°E" if v <= 180 else f"{360 - v}°W")
def _ylabel(v): return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

def render(lat, lon, data, pkg, coast_segs, dates, out_buf=None, title=None, cbar_label=None):
    fplot, vlim, cint = data["main"], pkg["vlim"], pkg["cint"]
    LON2D, LAT2D = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    lon_min, lon_max, lat_min, lat_max = 0, 360, -80, 80

    aspect = max(0.20, min(3.5, 360.0 / 160.0))
    avail_w, avail_h = 0.92 * 12.0, 0.80 * 7.0
    w_in, h_in = (aspect * avail_h, avail_h) if aspect * avail_h <= avail_w else (avail_w, avail_w / aspect)
    ax = fig.add_axes([0.04 + (11.04 - w_in) / 24.0, 0.10 + (5.6 - h_in) / 14.0, w_in / 12.0, h_in / 7.0])
    ax.set_facecolor("#f4f0e8")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    xticks, yticks = list(range(0, 360, 30)), list(range(-80, 81, 20))

    invert = pkg.get("invert_cbar", False)
    if pkg.get("one_sided"):
        cf = ax.contourf(LON2D, LAT2D, fplot, levels=np.linspace(0.0, vlim, 20), cmap=_pos_cmap(), extend="max", zorder=1, alpha=0.88)
    else:
        cf = ax.contourf(LON2D, LAT2D, fplot, levels=np.linspace(-vlim, vlim, 25 if vlim >= 100 else 20),
                         cmap=_chi_cmap_inv() if invert else _chi_cmap(), extend="both", zorder=1, alpha=0.88)

    pos_col, neg_col = ("#1b4f6b", "#5c3d11") if invert else ("#5c3d11", "#1b4f6b")
    line_lev = np.arange(0 if pkg.get("one_sided") else -vlim, vlim + 0.01, cint)
    line_lev = line_lev[line_lev != 0]
    ax.contour(LON2D, LAT2D, fplot, levels=line_lev[line_lev > 0], colors=pos_col, linewidths=0.55, alpha=0.55, zorder=2)
    ax.contour(LON2D, LAT2D, fplot, levels=line_lev[line_lev < 0], colors=neg_col, linewidths=0.55, linestyles="--", alpha=0.55, zorder=2)

    vec = data.get("vec_u"), data.get("vec_v")
    if pkg.get("show_wind") or vec[0] is not None:
        U0, V0 = vec if vec[0] is not None else (data["u"], data["v"])
        ref_mag, ref_unit, vscale = (pkg.get("vec_ref", 5.0), pkg.get("vec_unit", "5 m/s"), pkg.get("wind_scale", 50.0)) if vec[0] is not None else (5.0, "5 m/s", pkg["wind_scale"])
        step, vmin = (pkg.get("vec_step", 3), pkg.get("vec_min", 0.0)) if vec[0] is not None else (3, 0.0)
        qs = slice(None, None, step)
        Xq, Yq, Uq, Vq = LON2D[qs, qs], LAT2D[qs, qs], U0[qs, qs], V0[qs, qs]
        mag = np.sqrt(Uq**2 + Vq**2)
        mask = (~np.isnan(mag)) & (np.abs(Yq) <= lat_max) & (mag >= vmin)
        ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask], color="#111111", scale=vscale, scale_units="inches", width=0.0018, headwidth=4.5, headlength=5.5, headaxislength=4.8, minshaft=1.2, pivot="middle", zorder=6, alpha=0.92)

        rx, ry = lon_min + 144.0, lat_min + 9.6
        ax.quiver(rx, ry, ref_mag, 0, color="#111111", scale=vscale, scale_units="inches", width=0.0018, headwidth=4.5, headlength=5.5, headaxislength=4.8, pivot="tail", zorder=9)
        ax.text(rx, ry - 12.8, ref_unit, fontsize=8, color="#111111", ha="center", zorder=9)

    for seg in coast_segs:
        lons, lats = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0]), seg[:, 1]
        for part in np.split(np.column_stack([lons, lats]), np.where(np.abs(np.diff(lons)) > 180)[0] + 1):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

    for x in xticks: ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in yticks: ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.75, zorder=0, alpha=0.8)

    ax.set_xticks(xticks)
    ax.set_xticklabels([_xlabel(x) for x in xticks], fontsize=9.5, color="#333322", fontfamily="DejaVu Sans")
    ax.set_yticks(yticks)
    ax.set_yticklabels([_ylabel(y) for y in yticks], fontsize=9.5, color="#333322", fontfamily="DejaVu Sans")
    ax.tick_params(axis="both", length=3.5, color="#888878", width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988")
        spine.set_linewidth(0.8)

    cbar_w = min(0.760, (w_in / 12.0))
    cax = fig.add_axes([0.04 + (w_in / 12.0) * 0.5 - cbar_w * 0.5, 0.057, cbar_w, 0.028])
    lo = 0.0 if pkg.get("one_sided") else -vlim
    ticks = np.array([round(v, 8) for v in np.arange(lo, vlim + 0.001, cint)])
    if pkg.get("one_sided"): ticks = ticks[ticks > 0.0]
    ticks = np.unique(np.append(0.0, ticks[~np.isclose(ticks, 0.0, atol=cint * 0.01)]))
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.tick_params(labelsize=8.5, colors="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8.5, color="#222211")
    cbar.outline.set_edgecolor("#999988")
    cbar.outline.set_linewidth(0.7)
    cax.text(0.5, -1.55, cbar_label or pkg["cb_label"], transform=cax.transAxes, ha="center", va="top", fontsize=12, color="#222211", fontstyle="italic")

    ttext = title or f"{pkg['title']}  ·  {dates[0]:%-d %b} – {dates[-1]:%-d %b %Y}  ({len(dates)}-day mean)"
    fig.text(0.50, 0.965, ttext, ha="center", va="top", fontsize=16, fontweight="bold", color="#111100", fontfamily="DejaVu Sans")
    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=11, va="bottom", ha="right", color="#222211", fontweight="semibold", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccccbb", alpha=0.92, lw=0.9), zorder=10)
    ax.text(0.005, 0.016, "NCEP/NCAR Reanalysis  ·  PSL/NOAA", transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color="#666655", zorder=10)

    if out_buf is None: out_buf = io.BytesIO()
    plt.savefig(out_buf, format="png", dpi=220, bbox_inches="tight", facecolor="white", edgecolor="none")
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
            ds_temp = open_url(f"{PSL}/uwnd.{current_year - 1}.nc")
        units = ds_temp["time"].attributes.get("units", "hours since 1800-01-01")
        scale = 1.0 / 24.0 if "hours" in units else 1.0
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
        epoch = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else datetime.date(1800, 1, 1)
        date_end = epoch + datetime.timedelta(days=float(np.array(ds_temp["time"][:])[-1]) * scale)

    return [date_end - datetime.timedelta(days=n_days - 1) + datetime.timedelta(days=i) for i in range(n_days)]

# ================================================================
# Public API
# ================================================================
def list_products():
    return [{"id": p["id"], "title": p["title"], "name": p["name"], "desc": p["desc"], "level": p["level"], "tag": p["tag"]} for p in PRODUCTS.values()]

def generate(product_id=DEFAULT_PRODUCT, mode="auto", manual_date=None, n_days=DEFAULT_N_DAYS, log=None):
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
    meta = {"product": pkg["id"], "title": pkg["title"], "date_start": dates[0].isoformat(),
            "date_end": dates[-1].isoformat(), "n_days": len(dates), "level": pkg["level"]}
    return buf, meta

def generate_diff(product_id=DEFAULT_PRODUCT, date1=None, n_days1=DEFAULT_N_DAYS, date2=None, n_days2=DEFAULT_N_DAYS, inverse=False, log=None):
    pkg = PRODUCTS.get(product_id, PRODUCTS[DEFAULT_PRODUCT])
    say = (lambda m: log.append(m)) if log is not None else (lambda m: None)

    dates_a, dates_b = _resolve_dates("manual", date1, n_days1), _resolve_dates("manual", date2, n_days2)
    say(f"[diff] {pkg['title']}: A={dates_a[0]}→{dates_a[-1]}  B={dates_b[0]}→{dates_b[-1]}")

    say("[0] Loading coastline …")
    coast_segs = load_coastlines()

    say("[1] Computing Range A …")
    lat, lon, data_a = compute(pkg, dates_a)
    say("[2] Computing Range B …")
    _, _, data_b = compute(pkg, dates_b)

    say("[3] Difference …")
    sign = -1.0 if inverse else 1.0
    data = {"main": sign * (data_a["main"] - data_b["main"])}
    if "u" in data_a and "u" in data_b:
        data["u"], data["v"] = sign * (data_a["u"] - data_b["u"]), sign * (data_a["v"] - data_b["v"])

    tag = "B − A" if inverse else "A − B"
    buf = render(lat, lon, data, pkg, coast_segs, dates_a, title=pkg["title"], cbar_label=pkg["cb_label"] + f"  ({tag})")

    meta = {"product": pkg["id"], "title": pkg["title"], "date_start": dates_a[0].isoformat(), "date_end": dates_a[-1].isoformat(),
            "date_b_start": dates_b[0].isoformat(), "date_b_end": dates_b[-1].isoformat(), "n_days": len(dates_a), "level": pkg["level"], "diff": True, "inverse": inverse}
    return buf, meta
