"""
climo.py  —  PSL NCEP LTM Climatology Engine
=============================================
Source: https://psl.noaa.gov/thredds/catalog/Datasets/ncep/catalog.html

PSL 1991-2020 Daily LTM files used:
  air.sfc.day.ltm.1991-2020.nc   → 2m Temperature         (K   → °C)
  slp.day.ltm.1991-2020.nc       → Sea Level Pressure      (Pa  → hPa)
  rhum.sfc.day.ltm.1991-2020.nc  → Surface Rel. Humidity   (%)
  rhum.day.ltm.1991-2020.nc      → Isobaric Rel. Humidity  (%, multi-level)
  pr_wtr.day.ltm.1991-2020.nc    → Precipitable Water      (kg/m²)
  uwnd.day.ltm.1991-2020.nc      → U-Wind isobaric         (m/s, multi-level)
  vwnd.day.ltm.1991-2020.nc      → V-Wind isobaric         (m/s, multi-level)  ← NEW
  trpp.day.ltm.1991-2020.nc      → Tropopause Pressure     (mb  = hPa, no conv)
  trpt.day.ltm.1991-2020.nc      → Tropopause Temperature  (K   → °C)
  srfp.day.ltm.1991-2020.nc      → Surface Pressure        (mb  = hPa, no conv)
  srfpt.day.ltm.1994.2020.nc     → Surface Potential Temp  (K   → °C)
  hgt.day.ltm.1991-2020.nc       → Geopotential Height     (m)

Speed design:
  • PSL THREDDS NCSS → only bbox + today's DOY slice (~100-500 KB, not full file)
  • In-memory cache per (variable, doy, bbox) → zero cost on repeat calls
  • Parallel fetch with threading for combined fetches (e.g. wind speed)
  • Bilinear regrid LTM → GFS grid via scipy

VP & SF Anomaly:
  • Fetch both uwnd + vwnd LTM from PSL
  • Compute LTM VP via divergence Poisson solver
  • Compute LTM SF via vorticity Poisson solver
  • Anomaly = GFS_VP/SF − LTM_VP/SF
"""

import io, datetime, warnings, threading, concurrent.futures
import numpy as np
import requests
from scipy.io.netcdf import netcdf_file
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")

# ── PSL THREDDS NCSS base ─────────────────────────────────────────────
PSL_NCSS = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep"

# ════════════════════════════════════════════════════════════════════
#  LTM file registry
#  key → (filename, ncss_varname, raw_unit, output_unit, note)
# ════════════════════════════════════════════════════════════════════
LTM_REGISTRY = {
    # Surface / single-level
    "temp"  : ("air.sfc.day.ltm.1991-2020.nc",  "air",    "K",   "degC",  None),
    "mslp"  : ("slp.day.ltm.1991-2020.nc",       "slp",    "Pa",  "hPa",   None),
    "rh_sfc": ("rhum.sfc.day.ltm.1991-2020.nc",  "rhum",   "%",   "%",     None),
    "pwat"  : ("pr_wtr.day.ltm.1991-2020.nc",    "pr_wtr", "kg/m2","kg/m2",None),
    "trpp"  : ("trpp.day.ltm.1991-2020.nc",       "trpp",   "mb",  "hPa",   None),  # unit=mb=hPa, no conversion
    "trpt"  : ("trpt.day.ltm.1991-2020.nc",       "trpt",   "K",   "degC",  None),
    "srfp"  : ("srfp.day.ltm.1991-2020.nc",       "srfp",   "mb",  "hPa",   None),  # varname=srfp, unit=mb=hPa
    "srfpt" : ("srfpt.day.ltm.1994.2020.nc",      "srfpt",  "K",   "degC",  None),  # varname=srfpt
    # Pressure-level (need level_hpa)
    "rh_pr" : ("rhum.day.ltm.1991-2020.nc",       "rhum",   "%",   "%",     "isobaric"),
    "uwnd"  : ("uwnd.day.ltm.1991-2020.nc",        "uwnd",   "m/s", "m/s",   "isobaric"),
    "vwnd"  : ("vwnd.day.ltm.1991-2020.nc",        "vwnd",   "m/s", "m/s",   "isobaric"),  # ← NEW
    "hgt"   : ("hgt.day.ltm.1991-2020.nc",         "hgt",    "m",   "m",     "isobaric"),
}

# Map GFS variable keys → LTM registry key(s)
GFS_TO_LTM = {
    "temp"      : "temp",
    "mslp"      : "mslp",
    "rh"        : None,          # resolved dynamically (sfc or isobaric)
    "pwat"      : "pwat",
    "u"         : "uwnd",
    "v"         : "vwnd",        # ← NOW uses real vwnd LTM
    "wind"      : "wind_derived", # wind speed LTM from sqrt(uwnd²+vwnd²)
    "vp"        : "vp_derived",  # ← computed from uwnd+vwnd LTM
    "streamfunc": "sf_derived",  # ← computed from uwnd+vwnd LTM
    "sf_pwat"   : "pwat",        # ← PWAT anomaly shading (SF contours unchanged)
    "cape"      : None,          # no PSL LTM
    "vvel"      : None,          # no PSL LTM
    "trpp"      : "trpp",
    "trpt"      : "trpt",
    "srfp"      : "srfp",
    "srfpt"     : "srfpt",
}

# ── In-memory cache ───────────────────────────────────────────────────
_CACHE: dict = {}
_LOCK = threading.Lock()


# ════════════════════════════════════════════════════════════════════
#  NCSS fetch — spatial + temporal subset only
# ════════════════════════════════════════════════════════════════════

def _today_ltm_timestr() -> str:
    """
    Today's month/day mapped to the LTM file's reference year.
    PSL NCEP LTM files store their time axis as year 0001
    (days since 1800-01-01, actual range 0001-01-01 to 0001-12-31).
    NCSS time_start/time_end must use '0001-MM-DD' format.
    """
    t = datetime.date.today()
    return f"0001-{t.month:02d}-{t.day:02d}T00:00:00Z"


def _ncss_fetch(ltm_key: str,
                lat_min, lat_max, lon_min, lon_max,
                level_hpa: int = None) -> tuple:
    """
    Fetch ONE LTM field from PSL NCSS as a spatial bbox + single DOY slice.
    Returns (lat_1d, lon_1d, data_2d) in OUTPUT units.
    Result is ~100-500 KB — fast.
    """
    fname, varname, raw_unit, out_unit, ltype = LTM_REGISTRY[ltm_key]
    time_str = _today_ltm_timestr()

    params = {
        "var"        : varname,
        "north"      : lat_max,
        "south"      : lat_min,
        "west"       : lon_min,
        "east"       : lon_max,
        "time_start" : time_str,
        "time_end"   : time_str,
        "accept"     : "netCDF",
        "horizStride": 1,
    }
    if ltype == "isobaric" and level_hpa:
        params["vertCoord"] = str(int(level_hpa) * 100)  # Pa

    url = f"{PSL_NCSS}/{fname}"
    label = f"{fname.split('.')[0]}({level_hpa or 'sfc'})"
    print(f"  [LTM] {label}  {datetime.date.today().strftime('%b %d')} ...", end=" ", flush=True)

    try:
        r = requests.get(url, params=params, timeout=25)
    except Exception as e:
        raise RuntimeError(f"Network error: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")

    print(f"{len(r.content)/1024:.0f} KB  {r.elapsed.total_seconds():.2f}s")

    lat, lon, data = _parse_nc(r.content, varname, lat_min, lat_max, lon_min, lon_max)

    # Unit conversion
    if   raw_unit == "K"  and out_unit == "degC": data = data - 273.15
    elif raw_unit == "Pa" and out_unit == "hPa":  data = data / 100.0
    # mb == hPa numerically — no conversion needed

    return lat, lon, data


# ════════════════════════════════════════════════════════════════════
#  NetCDF3 parser (scipy — no heavy dependency)
# ════════════════════════════════════════════════════════════════════

def _parse_nc(raw: bytes, varname: str,
              lat_min, lat_max, lon_min, lon_max) -> tuple:
    buf = io.BytesIO(raw)
    ds  = netcdf_file(buf, 'r', mmap=False)
    keys = list(ds.variables.keys())

    def _arr(v):
        a = np.array(v.data, dtype=np.float64)
        for attr in ('_FillValue', 'missing_value', 'fill_value'):
            fv = getattr(v, attr, None)
            if fv is not None:
                try: a[np.isclose(a, float(fv), rtol=0, atol=1.0)] = np.nan
                except: pass
        sc  = float(getattr(v, 'scale_factor', 1.0) or 1.0)
        off = float(getattr(v, 'add_offset',   0.0) or 0.0)
        return a * sc + off

    lat_k = next((k for k in keys if k.lower() in ("lat","latitude")), None)
    lon_k = next((k for k in keys if k.lower() in ("lon","longitude")), None)
    if not lat_k or not lon_k:
        ds.close(); raise RuntimeError(f"lat/lon not found. keys={keys}")

    lat = _arr(ds.variables[lat_k])
    lon = _arr(ds.variables[lon_k])

    # Match variable name loosely
    var_k = next(
        (k for k in keys
         if k.lower() == varname.lower()
         or k.lower().startswith(varname.lower()[:5])),
        None)
    if not var_k:
        ds.close(); raise RuntimeError(f"Var '{varname}' not found. keys={keys}")

    data = _arr(ds.variables[var_k])
    ds.close()

    # Squeeze time / level dims → 2D
    while data.ndim > 2:
        data = data[0]

    # Ensure lat ascending
    if lat[0] > lat[-1]:
        lat  = lat[::-1]
        data = data[::-1, :]

    # Crop to bbox
    li  = np.where((lat >= lat_min - 0.5) & (lat <= lat_max + 0.5))[0]
    loi = np.where((lon >= lon_min - 0.5) & (lon <= lon_max + 0.5))[0]
    if len(li) == 0 or len(loi) == 0:
        raise RuntimeError(f"Empty crop. lat={lat[[0,-1]]} lon={lon[[0,-1]]}")

    return lat[li], lon[loi], data[np.ix_(li, loi)]


# ════════════════════════════════════════════════════════════════════
#  Cache wrapper
# ════════════════════════════════════════════════════════════════════

def _cached_fetch(ltm_key, lat_min, lat_max, lon_min, lon_max,
                  level_hpa=None):
    today = datetime.date.today()
    ck = (ltm_key, today.timetuple().tm_yday,
          round(lon_min,1), round(lon_max,1),
          round(lat_min,1), round(lat_max,1),
          level_hpa)
    with _LOCK:
        if ck in _CACHE:
            print(f"  [LTM] cache hit: {ltm_key}({level_hpa or 'sfc'})")
            return _CACHE[ck]

    result = _ncss_fetch(ltm_key, lat_min, lat_max, lon_min, lon_max, level_hpa)
    with _LOCK:
        _CACHE[ck] = result
    return result


# ════════════════════════════════════════════════════════════════════
#  LTM VP and SF computation from U+V wind climatology
# ════════════════════════════════════════════════════════════════════

def _compute_ltm_velocity_potential(lat, lon, u_ltm, v_ltm):
    """
    Compute LTM Velocity Potential χ from LTM U,V via divergence Poisson solver.
    Returns χ × 10⁶ m²/s  (same scaling as GFS VP output in fetch.py).
    """
    dlat_deg = abs(lat[1] - lat[0]) if len(lat) > 1 else 2.5
    dlon_deg = abs(lon[1] - lon[0]) if len(lon) > 1 else 2.5
    R = 6.371e6
    lat_rad = np.radians(lat)
    dlat_m  = np.radians(dlat_deg) * R

    # Divergence: ∂u/∂x + ∂v/∂y
    div = np.zeros_like(u_ltm)
    for j in range(u_ltm.shape[0]):
        cos_lat = np.cos(lat_rad[j])
        if abs(cos_lat) < 1e-6: continue
        dx = np.radians(dlon_deg) * R * cos_lat
        dv_dy = np.gradient(v_ltm[:, :], dlat_m, axis=0)
        du_dx = np.gradient(u_ltm[j, :], dx)
        div[j, :] = dv_dy[j, :] + du_dx

    # Poisson: ∇²χ = D  (FFT spectral solver — fast)
    try:
        from scipy.fft import dstn, idstn
        F = dstn(div / (dlat_m**2), type=1, norm="ortho")
        nj2, ni2 = div.shape
        ii = np.arange(1, ni2+1, dtype=np.float64)
        jj = np.arange(1, nj2+1, dtype=np.float64)
        LAM = (-4*(np.sin(np.pi*jj/(2*(nj2+1)))**2))[:,None] + (-4*(np.sin(np.pi*ii/(2*(ni2+1)))**2))[None,:]
        LAM[LAM==0] = -1e-10
        chi = idstn(F / LAM, type=1, norm="ortho")
    except Exception:
        chi = np.zeros_like(div)
        for _ in range(200):
            chi_new = np.zeros_like(chi)
            chi_new[1:-1, 1:-1] = 0.25*(chi[2:,1:-1]+chi[:-2,1:-1]+chi[1:-1,2:]+chi[1:-1,:-2]-div[1:-1,1:-1]*(dlat_m**2))
            chi_new[0,:]=chi_new[1,:]; chi_new[-1,:]=chi_new[-2,:]
            chi_new[:,0]=chi_new[:,-1]=0; chi=chi_new
    chi -= np.nanmean(chi)
    return chi * 1e-6   # → ×10⁶ m²/s


def _compute_ltm_stream_function(lat, lon, u_ltm, v_ltm):
    """
    Compute LTM Stream Function ψ from LTM U,V via vorticity Poisson solver.
    Returns ψ × 10⁶ m²/s  (same scaling as GFS SF output in fetch.py).
    """
    dlat_deg = abs(lat[1] - lat[0]) if len(lat) > 1 else 2.5
    dlon_deg = abs(lon[1] - lon[0]) if len(lon) > 1 else 2.5
    R = 6.371e6
    lat_rad = np.radians(lat)
    dlat_m  = np.radians(dlat_deg) * R

    # Vorticity: ∂v/∂x − ∂u/∂y
    vort = np.zeros_like(u_ltm)
    for j in range(u_ltm.shape[0]):
        cos_lat = np.cos(lat_rad[j])
        if abs(cos_lat) < 1e-6: continue
        dx = np.radians(dlon_deg) * R * cos_lat
        dv_dx = np.gradient(v_ltm[j, :], dx)
        du_dy = np.gradient(u_ltm[:, :], dlat_m, axis=0)
        vort[j, :] = dv_dx - du_dy[j, :]

    # Poisson: ∇²ψ = ζ  (FFT spectral solver — fast)
    try:
        from scipy.fft import dstn, idstn
        F = dstn(vort / (dlat_m**2), type=1, norm="ortho")
        nj2, ni2 = vort.shape
        ii = np.arange(1, ni2+1, dtype=np.float64)
        jj = np.arange(1, nj2+1, dtype=np.float64)
        LAM = (-4*(np.sin(np.pi*jj/(2*(nj2+1)))**2))[:,None] + (-4*(np.sin(np.pi*ii/(2*(ni2+1)))**2))[None,:]
        LAM[LAM==0] = -1e-10
        psi = idstn(F / LAM, type=1, norm="ortho")
    except Exception:
        psi = np.zeros_like(vort)
        for _ in range(200):
            psi_new = np.zeros_like(psi)
            psi_new[1:-1,1:-1] = 0.25*(psi[2:,1:-1]+psi[:-2,1:-1]+psi[1:-1,2:]+psi[1:-1,:-2]-vort[1:-1,1:-1]*(dlat_m**2))
            psi_new[0,:]=psi_new[1,:]; psi_new[-1,:]=psi_new[-2,:]
            psi_new[:,0]=psi_new[:,1]; psi_new[:,-1]=psi_new[:,-2]; psi=psi_new
    psi -= np.nanmean(psi)
    return psi * 1e-6   # → ×10⁶ m²/s


def _fetch_ltm_uv(lat_min, lat_max, lon_min, lon_max, level_hpa):
    """
    Parallel fetch of LTM U and V wind from PSL.
    Returns (lat, lon, u_ltm, v_ltm) — all on PSL grid.
    """
    print(f"  [LTM] Fetching uwnd + vwnd LTM at {level_hpa} mb (parallel) ...")

    results = {}
    errors  = {}

    def _fetch_u():
        try:
            results["u"] = _cached_fetch("uwnd", lat_min, lat_max, lon_min, lon_max, level_hpa)
        except Exception as e:
            errors["u"] = e

    def _fetch_v():
        try:
            results["v"] = _cached_fetch("vwnd", lat_min, lat_max, lon_min, lon_max, level_hpa)
        except Exception as e:
            errors["v"] = e

    t_u = threading.Thread(target=_fetch_u)
    t_v = threading.Thread(target=_fetch_v)
    t_u.start(); t_v.start()
    t_u.join(timeout=60); t_v.join(timeout=60)

    if "u" in errors: raise RuntimeError(f"LTM uwnd fetch failed: {errors['u']}")
    if "v" in errors: raise RuntimeError(f"LTM vwnd fetch failed: {errors['v']}")

    lat_u, lon_u, u_ltm = results["u"]
    lat_v, lon_v, v_ltm = results["v"]

    # If grids differ slightly, interpolate V onto U grid
    if not (np.allclose(lat_u, lat_v) and np.allclose(lon_u, lon_v)):
        print("  [LTM] Regriding vwnd → uwnd grid ...")
        itp = RegularGridInterpolator(
            (lat_v, lon_v), v_ltm,
            method="linear", bounds_error=False, fill_value=np.nanmean(v_ltm))
        LON_G, LAT_G = np.meshgrid(lon_u, lat_u)
        v_ltm = itp(np.stack([LAT_G.ravel(), LON_G.ravel()], axis=1)).reshape(LAT_G.shape)

    return lat_u, lon_u, u_ltm, v_ltm


def _get_ltm_wind_speed(lat_min, lat_max, lon_min, lon_max, level_hpa):
    """
    Get LTM Wind Speed (m/s) = sqrt(uwnd² + vwnd²) from PSL LTM.
    Uses cache key 'wind_derived'.
    """
    today = datetime.date.today()
    ck = ("wind_derived", today.timetuple().tm_yday,
          round(lon_min,1), round(lon_max,1),
          round(lat_min,1), round(lat_max,1),
          level_hpa)
    with _LOCK:
        if ck in _CACHE:
            print(f"  [LTM] cache hit: wind_derived({level_hpa})")
            return _CACHE[ck]

    lat, lon, u_ltm, v_ltm = _fetch_ltm_uv(lat_min, lat_max, lon_min, lon_max, level_hpa)
    print(f"  [LTM] Computing LTM Wind Speed from sqrt(U²+V²) ...")
    spd_ltm = np.sqrt(u_ltm**2 + v_ltm**2)
    result = (lat, lon, spd_ltm)
    with _LOCK:
        _CACHE[ck] = result
    return result


def _get_ltm_vp(lat_min, lat_max, lon_min, lon_max, level_hpa):
    """
    Get LTM Velocity Potential (×10⁶ m²/s) from PSL uwnd+vwnd LTM.
    Uses cache key 'vp_derived'.
    """
    today = datetime.date.today()
    ck = ("vp_derived", today.timetuple().tm_yday,
          round(lon_min,1), round(lon_max,1),
          round(lat_min,1), round(lat_max,1),
          level_hpa)
    with _LOCK:
        if ck in _CACHE:
            print(f"  [LTM] cache hit: vp_derived({level_hpa})")
            return _CACHE[ck]

    lat, lon, u_ltm, v_ltm = _fetch_ltm_uv(lat_min, lat_max, lon_min, lon_max, level_hpa)
    print(f"  [LTM] Computing LTM VP from U+V wind ...")
    vp_ltm = _compute_ltm_velocity_potential(lat, lon, u_ltm, v_ltm)
    result = (lat, lon, vp_ltm)
    with _LOCK:
        _CACHE[ck] = result
    return result


def _get_ltm_sf(lat_min, lat_max, lon_min, lon_max, level_hpa):
    """
    Get LTM Stream Function (×10⁶ m²/s) from PSL uwnd+vwnd LTM.
    Uses cache key 'sf_derived'.
    """
    today = datetime.date.today()
    ck = ("sf_derived", today.timetuple().tm_yday,
          round(lon_min,1), round(lon_max,1),
          round(lat_min,1), round(lat_max,1),
          level_hpa)
    with _LOCK:
        if ck in _CACHE:
            print(f"  [LTM] cache hit: sf_derived({level_hpa})")
            return _CACHE[ck]

    lat, lon, u_ltm, v_ltm = _fetch_ltm_uv(lat_min, lat_max, lon_min, lon_max, level_hpa)
    print(f"  [LTM] Computing LTM SF from U+V wind ...")
    sf_ltm = _compute_ltm_stream_function(lat, lon, u_ltm, v_ltm)
    result = (lat, lon, sf_ltm)
    with _LOCK:
        _CACHE[ck] = result
    return result


# ════════════════════════════════════════════════════════════════════
#  Regrid LTM → GFS grid
# ════════════════════════════════════════════════════════════════════

def _regrid(lat_c, lon_c, clim, gfs_lat, gfs_lon, fill=None):
    """Bilinear interpolation: LTM grid → GFS grid."""
    # Normalise lon ranges
    if gfs_lon.min() >= 0 and lon_c.min() < 0:
        lon_c = np.where(lon_c < 0, lon_c + 360, lon_c)
    elif gfs_lon.min() < 0 and lon_c.min() >= 0:
        lon_c = np.where(lon_c > 180, lon_c - 360, lon_c)

    fill_val = np.nanmean(clim) if fill is None else fill
    itp = RegularGridInterpolator(
        (lat_c, lon_c), clim,
        method="linear", bounds_error=False, fill_value=fill_val)

    LON_G, LAT_G = np.meshgrid(gfs_lon, gfs_lat)
    pts = np.stack([LAT_G.ravel(), LON_G.ravel()], axis=1)
    return itp(pts).reshape(LAT_G.shape)


# ════════════════════════════════════════════════════════════════════
#  Public API — compute_anomaly_from_ltm
# ════════════════════════════════════════════════════════════════════

def compute_anomaly_from_ltm(gfs_data, gfs_lat, gfs_lon,
                              variable_key,
                              lat_min, lat_max, lon_min, lon_max,
                              level_hpa=None):
    """
    GFS forecast − PSL 1991-2020 daily LTM climatology (today's DOY).
    Handles all GFS variable keys. Falls back gracefully if LTM unavailable.

    VP & SF: fetches real LTM from PSL uwnd+vwnd, computes derived climatology.
    """
    ltm_key = _resolve_ltm_key(variable_key, level_hpa)

    # ── Wind Speed: sqrt(U²+V²) from real PSL uwnd + vwnd LTM ────────
    if ltm_key == "wind_derived":
        try:
            lat_c, lon_c, spd_ltm = _get_ltm_wind_speed(
                lat_min, lat_max, lon_min, lon_max, level_hpa)
            clim_r  = _regrid(lat_c, lon_c, spd_ltm, gfs_lat, gfs_lon)
            anomaly = gfs_data - clim_r
            print(f"  [LTM] Wind anomaly range: {np.nanmin(anomaly):.3f} … {np.nanmax(anomaly):.3f}"
                  f"  (LTM wind mean: {np.nanmean(clim_r):.3f} m/s)")
            return anomaly
        except Exception as e:
            print(f"  [LTM] WARNING: Wind LTM fetch failed ({e})")
            print(f"  [LTM] Fallback: domain-mean reference for wind")
            return gfs_data - np.nanmean(gfs_data)

    # ── VP: derived from real PSL uwnd + vwnd LTM ─────────────────────
    if ltm_key == "vp_derived":
        try:
            lat_c, lon_c, vp_ltm = _get_ltm_vp(
                lat_min, lat_max, lon_min, lon_max, level_hpa)
            clim_r  = _regrid(lat_c, lon_c, vp_ltm, gfs_lat, gfs_lon)
            anomaly = gfs_data - clim_r
            print(f"  [LTM] VP anomaly range: {np.nanmin(anomaly):.3f} … {np.nanmax(anomaly):.3f}"
                  f"  (LTM VP mean: {np.nanmean(clim_r):.3f})")
            return anomaly
        except Exception as e:
            print(f"  [LTM] WARNING: VP LTM fetch failed ({e})")
            print(f"  [LTM] Fallback: domain-mean reference for VP")
            return gfs_data - np.nanmean(gfs_data)

    # ── SF: derived from real PSL uwnd + vwnd LTM ─────────────────────
    if ltm_key == "sf_derived":
        try:
            lat_c, lon_c, sf_ltm = _get_ltm_sf(
                lat_min, lat_max, lon_min, lon_max, level_hpa)
            clim_r  = _regrid(lat_c, lon_c, sf_ltm, gfs_lat, gfs_lon)
            anomaly = gfs_data - clim_r
            print(f"  [LTM] SF anomaly range: {np.nanmin(anomaly):.3f} … {np.nanmax(anomaly):.3f}"
                  f"  (LTM SF mean: {np.nanmean(clim_r):.3f})")
            return anomaly
        except Exception as e:
            print(f"  [LTM] WARNING: SF LTM fetch failed ({e})")
            print(f"  [LTM] Fallback: domain-mean reference for SF")
            return gfs_data - np.nanmean(gfs_data)

    # ── No external LTM (CAPE, VVEL, etc.) ────────────────────────────
    if ltm_key is None:
        ref = np.nanmean(gfs_data)
        print(f"  [LTM] No external LTM for '{variable_key}' — "
              f"using domain mean {ref:.3f} as reference")
        return gfs_data - ref

    # ── Standard LTM (temp, mslp, rh, pwat, u, v) ────────────────────
    try:
        lat_c, lon_c, clim = _cached_fetch(
            ltm_key, lat_min, lat_max, lon_min, lon_max, level_hpa)
        clim_r = _regrid(lat_c, lon_c, clim, gfs_lat, gfs_lon)
        anomaly = gfs_data - clim_r
        print(f"  [LTM] Anomaly range: {np.nanmin(anomaly):.3f} … {np.nanmax(anomaly):.3f}"
              f"  (ref mean: {np.nanmean(clim_r):.3f})")
        return anomaly

    except Exception as e:
        print(f"  [LTM] WARNING: PSL fetch failed ({e})")
        print(f"  [LTM] Fallback: domain-mean reference")
        return gfs_data - np.nanmean(gfs_data)


def _resolve_ltm_key(variable_key, level_hpa):
    """Return the LTM registry key for a GFS variable key, or None."""
    base = variable_key.replace("_anomaly", "")

    if base == "rh":
        # Use surface RH LTM if no pressure level, else isobaric
        return "rh_sfc" if (not level_hpa or level_hpa == 0) else "rh_pr"

    return GFS_TO_LTM.get(base, None)


# ════════════════════════════════════════════════════════════════════
#  Parallel prefetch  — call this ONCE at startup when anomaly is requested
# ════════════════════════════════════════════════════════════════════

def prefetch_ltm(variable_key, lat_min, lat_max, lon_min, lon_max,
                 level_hpa=None):
    """
    Trigger LTM fetch in background thread(s) concurrently with GFS fetch.
    For VP/SF: fires both uwnd and vwnd downloads in parallel.
    """
    base = variable_key.replace("_anomaly", "")
    ltm_key = _resolve_ltm_key(variable_key, level_hpa)

    # Wind speed / VP / SF: prefetch both U and V LTM winds in parallel
    if ltm_key in ("wind_derived", "vp_derived", "sf_derived"):
        def _bg():
            try:
                _fetch_ltm_uv(lat_min, lat_max, lon_min, lon_max, level_hpa)
                # Pre-compute the derived field so it's cached
                if ltm_key == "wind_derived":
                    _get_ltm_wind_speed(lat_min, lat_max, lon_min, lon_max, level_hpa)
                elif ltm_key == "vp_derived":
                    _get_ltm_vp(lat_min, lat_max, lon_min, lon_max, level_hpa)
                else:
                    _get_ltm_sf(lat_min, lat_max, lon_min, lon_max, level_hpa)
            except Exception as e:
                print(f"  [LTM prefetch wind/VP/SF] failed: {e}")
        t = threading.Thread(target=_bg, daemon=True)
        t.start()
        return t

    if ltm_key is None:
        return None  # nothing to prefetch

    def _bg():
        try:
            _cached_fetch(ltm_key, lat_min, lat_max, lon_min, lon_max, level_hpa)
        except Exception as e:
            print(f"  [LTM prefetch] failed: {e}")

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return t


def wait_prefetch(thread):
    """Wait for prefetch thread to finish (call just before anomaly compute)."""
    if thread is not None:
        thread.join(timeout=60)


# ════════════════════════════════════════════════════════════════════
#  Utility
# ════════════════════════════════════════════════════════════════════

def clear_cache():
    with _LOCK:
        _CACHE.clear()
    print("  [LTM] Cache cleared.")


def cache_status():
    with _LOCK:
        keys = list(_CACHE.keys())
    print(f"  [LTM] Cache: {len(keys)} entries")
    for k in keys:
        print(f"    {k}")
