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

# =====================================================================
#  GLOBAL CACHES
# =====================================================================
_DS_CACHE = {}         # url -> pydap Dataset
_FIELD_CACHE = {}      # (var, level, tuple(dates)) -> 2D numpy array
_COASTLINE_SEGS = None # loaded once on first render

# Target grid specs for NCEP/NCAR Reanalysis 1 (2.5° × 2.5°)
LATS = np.arange(90.0, -92.5, -2.5)   # 73 values
LONS = np.arange(0.0, 360.0, 2.5)     # 144 values

EARTH_R = 6.371e6  # meters

# =====================================================================
#  DOMAINS  (lat_min, lat_max, lon_min, lon_max)  in [0, 360) lon
# =====================================================================
DOMAINS = {
    "global":      (-90.0,  90.0,   0.0, 360.0),
    "tropics":     (-30.0,  30.0,   0.0, 360.0),
    "nh":          (  0.0,  90.0,   0.0, 360.0),
    "sh":          (-90.0,   0.0,   0.0, 360.0),
    "south_asia":  ( -10.0,  40.0,  40.0, 110.0),
    "io":          ( -40.0,  30.0,  30.0, 120.0),
    "pac":         ( -30.0,  30.0, 100.0, 290.0),
    "atl":         ( -30.0,  60.0, 260.0, 360.0),
}

DOMAIN_LABELS = {
    "global": "Global",
    "tropics": "Tropics (30°S–30°N)",
    "nh": "Northern Hemisphere",
    "sh": "Southern Hemisphere",
    "south_asia": "South Asia (10°S–40°N, 40°E–110°E)",
    "io": "Indian Ocean (40°S–30°N, 30°E–120°E)",
    "pac": "Pacific (30°S–30°N, 100°E–70°W)",
    "atl": "Atlantic (30°S–60°N, 100°W–0°)",
}

# Standard custom colormaps
def _make_cmap(colors, name="custom"):
    return LinearSegmentedColormap.from_list(name, colors, N=256)

CMAPS = {
    "chi": _make_cmap([
        "#40004b", "#762a83", "#9970ab", "#c2a5cf", "#e7d4e8",
        "#ffffff",
        "#d9f0d3", "#a6dba0", "#5aae61", "#1b7837", "#00441b"
    ], "chi"),
    "psi": _make_cmap([
        "#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
        "#ffffff",
        "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"
    ], "psi"),
    "u": _make_cmap([
        "#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
        "#ffffff",
        "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"
    ], "u"),
    "temp": _make_cmap([
        "#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
        "#ffffff",
        "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"
    ], "temp"),
}

# =====================================================================
#  PRODUCT DEFINITIONS (DATA-DRIVEN ENGINE)
#  To add a map, just append a dict here!
# =====================================================================
PRODUCTS = {
    "chi200": {
        "id": "chi200",
        "title": "200-hPa Velocity Potential & Divergent Wind Anomaly",
        "subtitle": "VP (10⁶ m²/s, contours/shading) & Wind Vector Anomaly (m/s)",
        "type": "chi",
        "level": 200,
        "cint": 1.5,
        "cmin": -15.0,
        "cmax": 15.0,
        "cmap": CMAPS["chi"],
        "unit": "10⁶ m²/s",
        "scale": 1e-6,
        "vector_scale": 45,
        "smooth": 1.2,
        "clim_years": (1991, 2020),
    },
    "psi200": {
        "id": "psi200",
        "title": "200-hPa Streamfunction & Rotational Wind Anomaly",
        "subtitle": "Streamfunction (10⁶ m²/s) & Rotational Wind Anomaly (m/s)",
        "type": "psi",
        "level": 200,
        "cint": 3.0,
        "cmin": -30.0,
        "cmax": 30.0,
        "cmap": CMAPS["psi"],
        "unit": "10⁶ m²/s",
        "scale": 1e-6,
        "vector_scale": 60,
        "smooth": 1.0,
        "clim_years": (1991, 2020),
    },
    "u200": {
        "id": "u200",
        "title": "200-hPa Zonal Wind Anomaly",
        "subtitle": "U-Wind Anomaly (m/s, shading/contours) & Total Vector Anomaly",
        "type": "zonal_wind",
        "level": 200,
        "cint": 4.0,
        "cmin": -32.0,
        "cmax": 32.0,
        "cmap": CMAPS["u"],
        "unit": "m/s",
        "scale": 1.0,
        "vector_scale": 50,
        "smooth": 0.8,
        "clim_years": (1991, 2020),
    },
    "u850": {
        "id": "u850",
        "title": "850-hPa Zonal Wind Anomaly",
        "subtitle": "U-Wind Anomaly (m/s, shading/contours) & Total Vector Anomaly",
        "type": "zonal_wind",
        "level": 850,
        "cint": 2.0,
        "cmin": -16.0,
        "cmax": 16.0,
        "cmap": CMAPS["u"],
        "unit": "m/s",
        "scale": 1.0,
        "vector_scale": 35,
        "smooth": 0.8,
        "clim_years": (1991, 2020),
    },
    "temp850": {
        "id": "temp850",
        "title": "850-hPa Temperature Anomaly",
        "subtitle": "Temperature Anomaly (°C, shading) & Total Wind Vector Anomaly",
        "type": "temp",
        "level": 850,
        "cint": 1.0,
        "cmin": -8.0,
        "cmax": 8.0,
        "cmap": CMAPS["temp"],
        "unit": "°C",
        "scale": 1.0,
        "vector_scale": 35,
        "smooth": 0.8,
        "clim_years": (1991, 2020),
    },
    "mse850": {
        "id": "mse850",
        "title": "850-hPa Equivalent Potential Temperature (Theta-e) Anomaly",
        "subtitle": "Moist Static Energy proxy — Theta-e Anomaly (K, shading) & Wind Vector Anomaly",
        "type": "mse",
        "level": 850,
        "cint": 1.5,
        "cmin": -12.0,
        "cmax": 12.0,
        "cmap": CMAPS["temp"],
        "unit": "K",
        "scale": 1.0,
        "vector_scale": 35,
        "smooth": 0.8,
        "clim_years": (1991, 2020),
    },
}

# =====================================================================
#  OPeNDAP / DATA FETCHING
# =====================================================================
def get_dataset(url):
    """Open OPeNDAP dataset with caching."""
    if url not in _DS_CACHE:
        _DS_CACHE[url] = open_url(url)
    return _DS_CACHE[url]

def _opendap_url(var_name, year):
    """NCEP/NCAR Reanalysis 1 daily OPeNDAP URL."""
    base = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyseries"
    if var_name in ["uwnd", "vwnd"]:
        return f"{base}/pressure/{var_name}.{year}.nc"
    elif var_name in ["air", "rhum"]:
        return f"{base}/pressure/{var_name}.{year}.nc"
    else:
        raise ValueError(f"Unknown variable: {var_name}")

def _clim_url(var_name):
    """NCEP 1991-2020 daily climatology URL."""
    return f"https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/pressure/{var_name}.day.1991-2020.ltm.nc"

def fetch_day_data(var_name, level, date_obj):
    """Fetch 2D lat-lon field for a single date (obs or clim)."""
    yr = date_obj.year
    url = _opendap_url(var_name, yr)
    ds = get_dataset(url)

    # find level index
    levels = np.array(ds["level"][:])
    lev_idx = int(np.argmin(np.abs(levels - level)))

    # calculate day of year index (0-based)
    t0 = datetime.date(yr, 1, 1)
    day_idx = (date_obj - t0).days

    var = ds[var_name]
    # shape: (time, level, lat, lon)
    data = var[day_idx, lev_idx, :, :]
    arr = np.squeeze(np.array(data))
    return arr

def fetch_clim_day(var_name, level, date_obj):
    """Fetch 1991-2020 daily climatology for given day-of-year."""
    url = _clim_url(var_name)
    ds = get_dataset(url)

    levels = np.array(ds["level"][:])
    lev_idx = int(np.argmin(np.abs(levels - level)))

    # day of year (0..365)
    doy = date_obj.timetuple().tm_yday - 1
    # clamp for leap year 366 -> 365
    if doy > 365:
        doy = 365

    var = ds[var_name]
    data = var[doy, lev_idx, :, :]
    return np.squeeze(np.array(data))

def fetch_field_mean(var_name, level, dates):
    """Average a field over a set of dates (obs or clim), cached."""
    key = (var_name, level, tuple(dates))
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    def _get_one(d):
        return fetch_day_data(var_name, level, d)

    with ThreadPoolExecutor(max_workers=min(len(dates), 8)) as ex:
        arrs = list(ex.map(_get_one, dates))

    mean_arr = np.mean(arrs, axis=0)
    _FIELD_CACHE[key] = mean_arr
    return mean_arr

def fetch_clim_mean(var_name, level, dates):
    """Average 1991-2020 climatology over a set of dates, cached."""
    key = (f"CLIM_{var_name}", level, tuple(dates))
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    def _get_one(d):
        return fetch_clim_day(var_name, level, d)

    with ThreadPoolExecutor(max_workers=min(len(dates), 8)) as ex:
        arrs = list(ex.map(_get_one, dates))

    mean_arr = np.mean(arrs, axis=0)
    _FIELD_CACHE[key] = mean_arr
    return mean_arr

# =====================================================================
#  SPHERICAL HARMONICS & POISSON SOLVERS (VELOCITY POTENTIAL / STREAMFUNCTION)
# =====================================================================
def compute_divergence(u, v, lats, lons):
    """
    Compute divergence on regular lat-lon grid:
    div = (1 / R cos phi) [ d u / d lon + d(v cos phi) / d phi ]
    """
    phi = np.deg2rad(lats)
    cos_phi = np.cos(phi)[:, None]
    cos_phi = np.where(np.abs(cos_phi) < 1e-4, 1e-4, cos_phi)

    dlon = np.deg2rad(2.5)
    dphi = np.deg2rad(-2.5)  # lats decrease 90 -> -90

    # du/dlon (axis=1)
    du_dlon = np.gradient(u, dlon, axis=1)

    # d(v cos phi) / dphi (axis=0)
    v_cos = v * cos_phi
    dvcos_dphi = np.gradient(v_cos, dphi, axis=0)

    div = (1.0 / (EARTH_R * cos_phi)) * (du_dlon + dvcos_dphi)
    return div

def compute_vorticity(u, v, lats, lons):
    """
    Compute relative vorticity on regular lat-lon grid:
    zeta = (1 / R cos phi) [ d v / d lon - d(u cos phi) / d phi ]
    """
    phi = np.deg2rad(lats)
    cos_phi = np.cos(phi)[:, None]
    cos_phi = np.where(np.abs(cos_phi) < 1e-4, 1e-4, cos_phi)

    dlon = np.deg2rad(2.5)
    dphi = np.deg2rad(-2.5)

    dv_dlon = np.gradient(v, dlon, axis=1)
    u_cos = u * cos_phi
    ducos_dphi = np.gradient(u_cos, dphi, axis=0)

    zeta = (1.0 / (EARTH_R * cos_phi)) * (dv_dlon - ducos_dphi)
    return zeta

def solve_poisson_fft(rhs, lats, lons):
    """
    Global Poisson solver del^2 psi = rhs on regular grid using 2D FFT.
    Returns psi (with mean zero).
    """
    nlat, nlon = rhs.shape

    # 2D FFT
    F = np.fft.fft2(rhs)

    # Wave numbers
    kx = np.fft.fftfreq(nlon, d=np.deg2rad(2.5))
    ky = np.fft.fftfreq(nlat, d=np.deg2rad(2.5))
    KX, KY = np.meshgrid(kx, ky)

    # Laplacian operator in spectral space: -(kx^2 + ky^2) / R^2
    denom = - (KX**2 + KY**2) / (EARTH_R**2)
    denom[0, 0] = 1.0  # avoid div by zero for mean mode

    F_sol = F / denom
    F_sol[0, 0] = 0.0  # mean = 0

    sol = np.real(np.fft.ifft2(F_sol))
    return sol

def invert_divergence_to_chi(u_anom, v_anom, lats, lons):
    """
    Compute Velocity Potential (chi) and divergent wind (u_div, v_div).
    del^2 chi = div  =>  u_div = d chi / d x, v_div = d chi / d y
    """
    div = compute_divergence(u_anom, v_anom, lats, lons)
    chi = solve_poisson_fft(div, lats, lons)

    # Divergent wind
    phi = np.deg2rad(lats)
    cos_phi = np.cos(phi)[:, None]
    cos_phi = np.where(np.abs(cos_phi) < 1e-4, 1e-4, cos_phi)

    dlon = np.deg2rad(2.5)
    dphi = np.deg2rad(-2.5)

    u_div = (1.0 / (EARTH_R * cos_phi)) * np.gradient(chi, dlon, axis=1)
    v_div = (1.0 / EARTH_R) * np.gradient(chi, dphi, axis=0)

    return chi, u_div, v_div

def invert_vorticity_to_psi(u_anom, v_anom, lats, lons):
    """
    Compute Streamfunction (psi) and rotational wind (u_rot, v_rot).
    del^2 psi = zeta => u_rot = -d psi / d y, v_rot = d psi / d x
    """
    zeta = compute_vorticity(u_anom, v_anom, lats, lons)
    psi = solve_poisson_fft(zeta, lats, lons)

    phi = np.deg2rad(lats)
    cos_phi = np.cos(phi)[:, None]
    cos_phi = np.where(np.abs(cos_phi) < 1e-4, 1e-4, cos_phi)

    dlon = np.deg2rad(2.5)
    dphi = np.deg2rad(-2.5)

    u_rot = - (1.0 / EARTH_R) * np.gradient(psi, dphi, axis=0)
    v_rot = (1.0 / (EARTH_R * cos_phi)) * np.gradient(psi, dlon, axis=1)

    return psi, u_rot, v_rot

def compute_theta_e(temp_k, rhum_pct, level_hpa):
    """
    Compute Equivalent Potential Temperature (Theta-e) in Kelvin.
    Bolton (1980) empirical formulation:
    e = RH/100 * 6.112 * exp(17.67*(T-273.15)/(T-29.65))
    w = 0.622 * e / (p - e)
    T_LCL = 56 + 1 / (1/(T - 55) - ln(RH/100)/2840)
    theta_e = T * (1000/p)^(0.2854 * (1 - 0.28*w)) * exp((3.376/T_LCL - 0.00254) * w * (1 + 0.81*w))
    """
    T = temp_k
    p = float(level_hpa)
    rh = np.clip(rhum_pct, 1.0, 100.0)

    tc = T - 273.15
    # vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * tc) / (tc + 243.5))
    e = (rh / 100.0) * es
    # mixing ratio (kg/kg)
    w = 0.622 * e / (p - e)
    w = np.maximum(w, 1e-6)

    # LCL temperature
    t_lcl = 56.0 + 1.0 / (1.0 / (T - 55.0) - np.log(rh / 100.0) / 2840.0)

    # Theta-e
    theta_e = T * ((1000.0 / p) ** (0.2854 * (1.0 - 0.28 * w))) * \
              np.exp(((3376.0 / t_lcl) - 2.54) * w * (1.0 + 0.81 * w))

    return theta_e

# =====================================================================
#  COASTLINES (Natural Earth shapefile)
# =====================================================================
def load_coastlines():
    """Load Natural Earth 1:110m coastlines once into cache."""
    global _COASTLINE_SEGS
    if _COASTLINE_SEGS is not None:
        return _COASTLINE_SEGS

    shp_path = os.path.join(os.path.dirname(__file__), "data", "ne_110m_coastline.shp")
    segs = []

    if os.path.exists(shp_path):
        sf = shapefile.Reader(shp_path)
        for shape in sf.shapes():
            pts = np.array(shape.points)
            segs.append(pts)
    else:
        # Fallback: simple world coast outline using basic coords
        print("[WARN] Coastline shapefile not found! Coastlines disabled.")

    _COASTLINE_SEGS = segs
    return segs

# =====================================================================
#  COMPUTE ENGINE
# =====================================================================
def compute(pkg, dates):
    """Generic calculation for any product package."""
    ptype = pkg["type"]
    level = pkg["level"]

    # 1. Fetch U & V wind anomalies
    u_obs = fetch_field_mean("uwnd", level, dates)
    u_clim = fetch_clim_mean("uwnd", level, dates)
    u_anom = u_obs - u_clim

    v_obs = fetch_field_mean("vwnd", level, dates)
    v_clim = fetch_clim_mean("vwnd", level, dates)
    v_anom = v_obs - v_clim

    data = {}

    if ptype == "chi":
        chi, u_div, v_div = invert_divergence_to_chi(u_anom, v_anom, LATS, LONS)
        data["main"] = chi
        data["u"] = u_div
        data["v"] = v_div

    elif ptype == "psi":
        psi, u_rot, v_rot = invert_vorticity_to_psi(u_anom, v_anom, LATS, LONS)
        data["main"] = psi
        data["u"] = u_rot
        data["v"] = v_rot

    elif ptype == "zonal_wind":
        data["main"] = u_anom
        data["u"] = u_anom
        data["v"] = v_anom

    elif ptype == "temp":
        t_obs = fetch_field_mean("air", level, dates)
        t_clim = fetch_clim_mean("air", level, dates)
        data["main"] = t_obs - t_clim
        data["u"] = u_anom
        data["v"] = v_anom

    elif ptype == "mse":
        t_obs = fetch_field_mean("air", level, dates)
        rh_obs = fetch_field_mean("rhum", level, dates)
        te_obs = compute_theta_e(t_obs, rh_obs, level)

        t_clim = fetch_clim_mean("air", level, dates)
        rh_clim = fetch_clim_mean("rhum", level, dates)
        te_clim = compute_theta_e(t_clim, rh_clim, level)

        data["main"] = te_obs - te_clim
        data["u"] = u_anom
        data["v"] = v_anom

    else:
        raise ValueError(f"Unknown product type: {ptype}")

    return LATS, LONS, data

# =====================================================================
#  RENDERER ENGINE
# =====================================================================
def render(lats, lons, data, pkg, coast_segs, domain_key="global", title_override=None):
    """
    Render product to PNG bytes using pure Matplotlib (publication quality).
    """
    dom = DOMAINS.get(domain_key, DOMAINS["global"])
    lat_min, lat_max, lon_min, lon_max = dom

    # Dynamic Figure Size calculation based on aspect ratio
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    aspect = lon_span / lat_span

    fig_w = 12.0
    fig_h = max(4.0, min(8.0, fig_w / aspect + 1.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.86, bottom=0.12)

    # Smooth main field
    main_field = data["main"] * pkg["scale"]
    if pkg.get("smooth", 0) > 0:
        main_field = gaussian_filter(main_field, sigma=pkg["smooth"])

    # Meshgrid
    LON, LAT = np.meshgrid(lons, lats)

    # Contour levels
    cmin, cmax, cint = pkg["cmin"], pkg["cmax"], pkg["cint"]
    levels = np.arange(cmin, cmax + cint, cint)

    # Shading (Filled Contours)
    cf = ax.contourf(LON, LAT, main_field, levels=levels, cmap=pkg["cmap"], extend="both")

    # Black Contours
    cs = ax.contour(LON, LAT, main_field, levels=levels, colors="black", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")

    # Vectors (Quiver)
    if "u" in data and "v" in data:
        u_vec = data["u"]
        v_vec = data["v"]

        # Subsample vectors for clean display
        skip = 2 if domain_key in ["south_asia", "io", "pac", "atl"] else 3
        sl_y = slice(None, None, skip)
        sl_x = slice(None, None, skip)

        q = ax.quiver(
            LON[sl_y, sl_x], LAT[sl_y, sl_x],
            u_vec[sl_y, sl_x], v_vec[sl_y, sl_x],
            color="black", scale=pkg["vector_scale"] * 10,
            width=0.002, headwidth=3, headlength=4, alpha=0.85
        )
        # Quiver key
        key_val = 10.0 if "200" in pkg["id"] else 5.0
        ax.quiverkey(q, 0.88, 1.03, key_val, f"{key_val:.0f} m/s", labelpos="E", coordinates="axes", fontproperties={"size": 8})

    # Coastlines
    for seg in coast_segs:
        # handle longitude wrap 0..360
        pts = seg.copy()
        # draw standard (-180..180 converted to 0..360)
        pts_360 = pts.copy()
        pts_360[:, 0] = np.where(pts_360[:, 0] < 0, pts_360[:, 0] + 360, pts_360[:, 0])
        ax.plot(pts_360[:, 0], pts_360[:, 1], color="#222222", linewidth=0.8, alpha=0.85)

    # Set Domain Bounds
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")

    # Gridlines & Ticks
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")

    # Format Lat/Lon Labels
    xtick_vals = np.linspace(lon_min, lon_max, 7)
    ytick_vals = np.linspace(lat_min, lat_max, 5)

    def _fmt_lon(x):
        x = x % 360
        if x == 0 or x == 360: return "0°"
        if x < 180: return f"{x:.0f}°E"
        if x == 180: return "180°"
        return f"{360-x:.0f}°W"

    def _fmt_lat(y):
        if y == 0: return "EQ"
        if y > 0: return f"{y:.0f}°N"
        return f"{-y:.0f}°S"

    ax.set_xticks(xtick_vals)
    ax.set_xticklabels([_fmt_lon(x) for x in xtick_vals], fontsize=8)
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels([_fmt_lat(y) for y in ytick_vals], fontsize=8)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08, shrink=0.7, aspect=30)
    cbar.set_label(f"Anomaly ({pkg['unit']})", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    # Titles (Dynamic y-positioning relative to dynamic dynamic height)
    title_text = title_override if title_override else pkg["title"]
    fig.suptitle(title_text, fontsize=12, fontweight="bold", y=0.98)
    ax.set_title(pkg["subtitle"], fontsize=8, color="#444444", pad=8)

    # Save to PNG buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# =====================================================================
#  DATE RESOLVER HELPERS
# =====================================================================
def _resolve_dates(mode, date_str, n_days):
    """Helper to generate list of datetime.date objects."""
    if mode == "auto" or not date_str:
        end = datetime.date.today() - datetime.timedelta(days=2)
    else:
        end = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

    n = max(1, int(n_days))
    dates = [end - datetime.timedelta(days=i) for i in reversed(range(n))]
    return dates

# =====================================================================
#  HIGH-LEVEL PUBLIC PIPELINE API
# =====================================================================
def run_pipeline(product_id, domain="global", mode="auto", date_str=None, n_days=5, progress_cb=None):
    """
    Run full single-map pipeline: fetch -> compute -> render.
    Returns PNG bytes.
    """
    say = progress_cb if progress_cb is not None else (lambda m: None)

    if product_id not in PRODUCTS:
        raise ValueError(f"Product '{product_id}' not found.")

    pkg = PRODUCTS[product_id]
    dates = _resolve_dates(mode, date_str, n_days)

    say(f"[1/4] Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    say("[2/4] Loading coastline shapefile …")
    coast_segs = load_coastlines()

    say(f"[3/4] Computing {pkg['title']} …")
    lat, lon, data = compute(pkg, dates)

    say("[4/4] Rendering map graphics …")
    buf = render(lat, lon, data, pkg, coast_segs, domain_key=domain)
    say("Done!")

    return buf

def run_diff_pipeline(product_id, domain="global",
                      date1=None, n_days1=5,
                      date2=None, n_days2=5,
                      inverse=False, progress_cb=None):
    """
    Run difference pipeline: Compute (Range A) - (Range B) [or B - A if inverse].
    Returns PNG bytes.
    """
    say = progress_cb if progress_cb is not None else (lambda m: None)

    if product_id not in PRODUCTS:
        raise ValueError(f"Product '{product_id}' not found.")

    pkg = PRODUCTS[product_id]

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
    buf = render(lat, lon, data, pkg, coast_segs, domain_key=domain, title_override=title)
    say("Done!")

    return buf
