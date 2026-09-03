"""
config.py  —  GFS Project Configuration
========================================
সব ধ্রুবক (constants), রিজিয়ন বাউন্ডারি, ভ্যারিয়েবল মেটাডেটা এখানে।
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Gaussian smooth ──────────────────────────────────────────────────
SMOOTH_SIGMA = 1.0

# ── Shapefile paths / URLs ───────────────────────────────────────────
SHP_DIR    = "map"
SHP_PATH   = os.path.join(SHP_DIR, "ne_50m_admin_0_countries.shp")
SHP_URL    = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
COAST_PATH = os.path.join(SHP_DIR, "ne_50m_coastline.shp")
COAST_URL  = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"

# ── NOMADS endpoints ─────────────────────────────────────────────────
NOMADS_FILTER_BASE    = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
NOMADS_FILTER_BASE_3H = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
OPENDAP_BASE          = "https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr"
THREDDS_SERVERS = [
    "https://thredds.ucar.edu/thredds/ncss/grib/NCEP/GFS/Global_0p5deg/Best",
    "https://thredds.aos.wisc.edu/thredds/ncss/grid/grib/NCEP/GFS/Global_0p5deg/Best",
]

# ════════════════════════════════════════════════════════════════════
#  REGIONS  —  (lon_min, lon_max, lat_min, lat_max)
# ════════════════════════════════════════════════════════════════════
REGIONS = {
    # ── South Asia ──────────────────────────────────────────────────
    "1":  {
        "name"   : "Bangladesh",
        "bounds" : (85.0, 95.0, 20.0, 28.0),
    },
    "2":  {
        "name"   : "South Asia",
        "bounds" : (65.0, 100.0, 5.0, 40.0),
    },
    "3":  {
        "name"   : "Bay of Bengal",
        "bounds" : (78.0, 100.0, 5.0, 25.0),
    },
    "4":  {
        "name"   : "Arabian Sea",
        "bounds" : (50.0, 78.0, 5.0, 30.0),
    },

    # ── Ocean Basins ─────────────────────────────────────────────────
    "5":  {
        "name"   : "Indian Ocean (Full)",
        "bounds" : (20.0, 120.0, -60.0, 30.0),
    },
    "6":  {
        "name"   : "Indian Ocean (North)",
        "bounds" : (40.0, 110.0, -10.0, 30.0),
    },
    "7":  {
        "name"   : "Pacific (West)",
        "bounds" : (100.0, 180.0, -20.0, 50.0),
    },
    "8":  {
        "name"   : "Pacific (East)",
        "bounds" : (180.0, 280.0, -20.0, 60.0),
    },
    "9":  {
        "name"   : "Pacific (Full)",
        "bounds" : (100.0, 280.0, -30.0, 60.0),
    },

    # ── Global / Continental ─────────────────────────────────────────
    "10": {
        "name"   : "Global",
        "bounds" : (0.0, 360.0, -90.0, 90.0),
    },
    "11": {
        "name"   : "Asia",
        "bounds" : (25.0, 145.0, 0.0, 55.0),
    },
    "12": {
        "name"   : "Southeast Asia",
        "bounds" : (90.0, 145.0, -10.0, 30.0),
    },
    "13": {
        "name"   : "Middle East",
        "bounds" : (25.0, 70.0, 10.0, 45.0),
    },
    "14": {
        "name"   : "Africa",
        "bounds" : (-20.0, 55.0, -40.0, 40.0),
    },
    "15": {
        "name"   : "Europe",
        "bounds" : (-25.0, 50.0, 30.0, 75.0),
    },
    "16": {
        "name"   : "North America",
        "bounds" : (-170.0, -50.0, 10.0, 75.0),
    },
    "17": {
        "name"   : "Australia",
        "bounds" : (110.0, 180.0, -50.0, 0.0),
    },
}

# ════════════════════════════════════════════════════════════════════
#  VARIABLES
# ════════════════════════════════════════════════════════════════════
VARIABLES = {
    "1" : {"key": "wind",       "name": "Wind Speed + Streamlines (U, V)"},
    "2" : {"key": "temp",       "name": "2m Temperature (°C)"},
    "3" : {"key": "mslp",       "name": "Mean Sea Level Pressure (hPa)"},
    "4" : {"key": "rh",         "name": "Relative Humidity (%)"},
    "5" : {"key": "precip",     "name": "Accumulated Precipitation (mm)"},
    "6" : {"key": "cape",       "name": "CAPE — (J/kg)"},
    "7" : {"key": "vvel",       "name": "500 mb Vertical Velocity (Pa/s)"},
    "8" : {"key": "pwat",       "name": "Precipitable Water (kg/m²)"},
    "9" : {"key": "u",          "name": "U-Wind (m/s)"},
    "10": {"key": "v",          "name": "V-Wind (m/s)"},
    "11": {"key": "vp",         "name": "Velocity Potential (derived, m²/s)"},
    "12": {"key": "streamfunc", "name": "Stream Function (ψ, derived, m²/s)"},
    "13": {"key": "sf_pwat",    "name": "Stream Function + Precipitable Water Overlay"},
    "14": {"key": "trueconverge", "name": "True Converge"},
}

# ── ভ্যারিয়েবলগুলো যেগুলো pressure level দরকার ────────────────────
LEVEL_REQUIRED_VARS = {"wind", "rh", "vvel", "u", "v", "vp", "streamfunc", "sf_pwat", "trueconverge"}

# ════════════════════════════════════════════════════════════════════
#  PRESSURE LEVELS
# ════════════════════════════════════════════════════════════════════
PRESSURE_LEVELS = {
    "1" : 1000,
    "2" : 925,
    "3" : 850,
    "4" : 700,
    "5" : 500,
    "6" : 400,
    "7" : 300,
    "8" : 200,
    "9" : 100,
    "10": 50,
}

# ════════════════════════════════════════════════════════════════════
#  AVERAGE / FORECAST OPTIONS
# ════════════════════════════════════════════════════════════════════
AVERAGE_OPTIONS = {
    "1": {"label": "Single snapshot  (no average — just current step)",
          "days" : 0},
    "2": {"label": "1-day total      (0h – 24h accumulation)",
          "days" : 1},
    "3": {"label": "3-day total      (0h – 72h accumulation)",
          "days" : 3},
    "4": {"label": "5-day total      (0h – 120h accumulation)",
          "days" : 5},
    "5": {"label": "7-day total      (0h – 168h accumulation)",
          "days" : 7},
}

# Default single forecast step (used when average_days == 0)
DEFAULT_STEP = 0

# Step interval for multi-day fetching (hours)
AVG_STEP_INTERVAL = 24

# ════════════════════════════════════════════════════════════════════
#  ANOMALY VARIABLES — এগুলোতে Anomaly অপশন আছে
# ════════════════════════════════════════════════════════════════════
ANOMALY_SUPPORTED_VARS = {"temp", "mslp", "rh", "vp", "streamfunc", "sf_pwat", "wind"}

# ════════════════════════════════════════════════════════════════════
#  GFS CLIMATOLOGICAL REFERENCE (1991-2020 approximate)
#  Used for anomaly computation when no external climatology file found
#  Key: variable_key → approximate global mean (used as fallback only)
# ════════════════════════════════════════════════════════════════════
CLIM_REF = {
    "temp" : 15.0,   # °C  (rough global 2m mean)
    "mslp" : 1013.25, # hPa
    "rh"   : 70.0,   # %
    "vp"   : 0.0,    # m²/s (zero mean expected)
}

# ════════════════════════════════════════════════════════════════════
#  SMOOTH GRADIENT COLORMAP FOR PRECIPITATION
# ════════════════════════════════════════════════════════════════════
def create_precip_cmap(transparent=True):
    start_white = (1.0, 1.0, 1.0, 0.0) if transparent else (1.0, 1.0, 1.0, 1.0)
    colors = [
        start_white,
        (0.85, 0.98, 1.0, 1.0),
        (0.40, 0.88, 1.0, 1.0),
        (0.20, 0.85, 0.4, 1.0),
        (0.95, 0.90, 0.1, 1.0),
        (1.00, 0.35, 0.0, 1.0),
        (0.85, 0.10, 0.5, 1.0),
        (0.50, 0.00, 0.7, 1.0),
        (1.00, 1.00, 1.0, 1.0),
    ]
    return mcolors.LinearSegmentedColormap.from_list("smooth_precip_cmap", colors, N=256)

PRECIP_CMAP = create_precip_cmap(transparent=True)

def get_dynamic_precip_levels(d):
    max_val = float(np.nanmax(d)) if np.any(~np.isnan(d)) else 1.0
    if max_val <= 0.1:
        max_limit = 1.0
    else:
        max_limit = float(np.ceil(max_val))
    return np.linspace(0, max_limit, 100)

# ════════════════════════════════════════════════════════════════════
#  PLOT CONFIGS
# ════════════════════════════════════════════════════════════════════
PLOT_CONFIGS = {
    "wind": {
        "cmap"       : "turbo",
        "unit"       : "m/s",
        "label"      : lambda lv: f"{lv} mb Wind Speed",
        "title_var"  : lambda lv, avg: f"{lv} mb Wind Speed (Shaded) & Streamlines" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(0, max(np.ceil(float(np.nanmax(d))/5)*5, 15), 80),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": True,
    },
    # ── U-Wind: Blue=negative, Red=positive, strict zero-centred ──────
    "u": {
        "cmap"       : "RdBu_r",
        "unit"       : "m/s",
        "label"      : lambda lv: f"{lv} mb U-Wind",
        "title_var"  : lambda lv, avg: f"{lv} mb U-component of Wind (m/s)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": False,
    },
    # ── V-Wind: Blue=negative, Red=positive, strict zero-centred ──────
    "v": {
        "cmap"       : "RdBu_r",
        "unit"       : "m/s",
        "label"      : lambda lv: f"{lv} mb V-Wind",
        "title_var"  : lambda lv, avg: f"{lv} mb V-component of Wind (m/s)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": False,
    },
    "vp": {
        "cmap"       : "seismic",
        "unit"       : "×10⁶ m²/s",
        "label"      : lambda lv: f"{lv} mb Velocity Potential",
        "title_var"  : lambda lv, avg: f"{lv} mb Velocity Potential (×10⁶ m²/s)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : True,
        "streamlines": False,
    },
    "vp_anomaly": {
        "cmap"       : "seismic",
        "unit"       : "×10⁶ m²/s",
        "label"      : lambda lv: f"{lv} mb VP Anomaly",
        "title_var"  : lambda lv, avg: f"{lv} mb Velocity Potential Anomaly (×10⁶ m²/s)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : True,
        "streamlines": False,
    },
    "temp": {
        "cmap"       : "RdBu_r",
        "unit"       : "°C",
        "label"      : lambda lv: "2 m Temperature",
        "title_var"  : lambda lv, avg: "2 m Temperature (°C)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(np.nanmin(d)-1, np.nanmax(d)+1, 60),
        "alpha"      : 0.90,
        "contour"    : True,
        "streamlines": False,
    },
    "temp_anomaly": {
        "cmap"       : "RdBu_r",
        "unit"       : "°C",
        "label"      : lambda lv: "2 m Temperature Anomaly",
        "title_var"  : lambda lv, avg: "2 m Temperature Anomaly (°C)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.90,
        "contour"    : True,
        "streamlines": False,
    },
    "mslp": {
        "cmap"       : "RdYlBu_r",
        "unit"       : "hPa",
        "label"      : lambda lv: "MSLP",
        "title_var"  : lambda lv, avg: "Mean Sea Level Pressure (hPa)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(np.nanmin(d)-0.5, np.nanmax(d)+0.5, 60),
        "alpha"      : 0.85,
        "contour"    : True,
        "streamlines": False,
    },
    "mslp_anomaly": {
        "cmap"       : "RdBu_r",
        "unit"       : "hPa",
        "label"      : lambda lv: "MSLP Anomaly",
        "title_var"  : lambda lv, avg: "MSLP Anomaly (hPa)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.85,
        "contour"    : True,
        "streamlines": False,
    },
    "rh": {
        "cmap"       : "BrBG",
        "unit"       : "%",
        "label"      : lambda lv: f"{lv} mb Relative Humidity",
        "title_var"  : lambda lv, avg: f"{lv} mb Relative Humidity (%)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(0, 100, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": False,
    },
    "rh_anomaly": {
        "cmap"       : "BrBG",
        "unit"       : "%",
        "label"      : lambda lv: f"{lv} mb RH Anomaly",
        "title_var"  : lambda lv, avg: f"{lv} mb Relative Humidity Anomaly (%)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": False,
    },
    "precip": {
        "cmap"       : PRECIP_CMAP,
        "unit"       : "mm",
        "label"      : lambda lv: "Total Precipitation",
        "title_var"  : lambda lv, avg: (
            f"Total Accumulated Precipitation — {avg}-day Sum (mm)" if avg
            else "Accumulated Precipitation (mm)"
        ),
        "clevs"      : get_dynamic_precip_levels,
        "alpha"      : 0.90,
        "contour"    : False,
        "streamlines": False,
    },
    "cape": {
        "cmap"       : "hot_r",
        "unit"       : "J/kg",
        "label"      : lambda lv: "CAPE",
        "title_var"  : lambda lv, avg: "Surface CAPE (J/kg)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(0, max(float(np.nanmax(d)), 100), 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": False,
    },
    "vvel": {
        "cmap"       : "bwr",
        "unit"       : "Pa/s",
        "label"      : lambda lv: "500 mb Vertical Velocity",
        "title_var"  : lambda lv, avg: "500 mb Vertical Velocity (Pa/s)  [− = upward]" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(-abs(np.nanmax(np.abs(d)))-0.01,
                                              abs(np.nanmax(np.abs(d)))+0.01, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": False,
    },
    "pwat": {
        "cmap"       : "YlGnBu",
        "unit"       : "kg/m²",
        "label"      : lambda lv: "Precipitable Water",
        "title_var"  : lambda lv, avg: "Precipitable Water (kg/m²)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(np.nanmin(d), np.nanmax(d)+0.1, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": False,
    },
    "streamfunc": {
        "cmap"       : "RdYlBu_r",
        "unit"       : "×10⁶ m²/s",
        "label"      : lambda lv: f"{lv} mb Stream Function",
        "title_var"  : lambda lv, avg: f"{lv} mb Stream Function ψ (×10⁶ m²/s)" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : True,
        "streamlines": True,
    },
    # ── True Convergence Wind: convergent flow + speed change diagnostic ──
    "trueconverge": {
        "cmap"       : "RdBu_r",
        "unit"       : "×10⁻⁵ s⁻¹",
        "label"      : lambda lv: f"{lv} mb True Convergence Wind",
        "title_var"  : lambda lv, avg: f"{lv} mb True Convergence Wind",
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": True,
    },
    "sf_pwat": {
        "cmap"       : "YlGnBu",
        "unit"       : "kg/m²",
        "label"      : lambda lv: "Precipitable Water",
        "title_var"  : lambda lv, avg: f"{lv} mb Stream Function + Precipitable Water Overlay" + (f"  [{avg}-day avg]" if avg else ""),
        "clevs"      : lambda d: np.linspace(np.nanmin(d), np.nanmax(d)+0.1, 60),
        "alpha"      : 0.75,
        "contour"    : False,
        "streamlines": True,
    },
    # ── SF + PWAT Anomaly: PWAT anomaly shading + SF contours + wind arrows ──
    "sf_pwat_anomaly": {
        "cmap"       : "BrBG",
        "unit"       : "kg/m²",
        "label"      : lambda lv: "PWAT Anomaly",
        "title_var"  : lambda lv, avg: (
            f"{lv} mb Stream Function + PWAT Anomaly  "
            f"[GFS − PSL 1991-2020 LTM]" + (f"  [{avg}-day avg]" if avg else "")
        ),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.75,
        "contour"    : False,
        "streamlines": True,   # triggers arrow overlay via is_sf_pwat branch
    },
    # ── Wind Speed Anomaly: derived from PSL uwnd+vwnd LTM ────────────
    "wind_anomaly": {
        "cmap"       : "RdBu_r",
        "unit"       : "m/s",
        "label"      : lambda lv: f"{lv} mb Wind Speed Anomaly",
        "title_var"  : lambda lv, avg: (
            f"{lv} mb Wind Speed Anomaly (m/s)  "
            f"[GFS − PSL 1991-2020 LTM]" + (f"  [{avg}-day avg]" if avg else "")
        ),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : False,
        "streamlines": True,   # show actual GFS wind streamlines as overlay
    },
    # ── VP Anomaly: derived from PSL uwnd+vwnd LTM ────────────────────
    "vp_anomaly": {
        "cmap"       : "seismic",
        "unit"       : "×10⁶ m²/s",
        "label"      : lambda lv: f"{lv} mb VP Anomaly",
        "title_var"  : lambda lv, avg: (
            f"{lv} mb Velocity Potential Anomaly (×10⁶ m²/s)  "
            f"[GFS − PSL 1991-2020 LTM]" + (f"  [{avg}-day avg]" if avg else "")
        ),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : True,
        "streamlines": False,
    },
    # ── SF Anomaly: derived from PSL uwnd+vwnd LTM ────────────────────
    "streamfunc_anomaly": {
        "cmap"       : "RdYlBu_r",
        "unit"       : "×10⁶ m²/s",
        "label"      : lambda lv: f"{lv} mb Stream Function Anomaly",
        "title_var"  : lambda lv, avg: (
            f"{lv} mb Stream Function Anomaly ψ (×10⁶ m²/s)  "
            f"[GFS − PSL 1991-2020 LTM]" + (f"  [{avg}-day avg]" if avg else "")
        ),
        "clevs"      : lambda d: _symmetric_levels(d, 60),
        "alpha"      : 0.88,
        "contour"    : True,
        "streamlines": True,   # U,V streamlines from extra_data
    },
}


def _symmetric_levels(d, n=60):
    """Zero-centred symmetric colour levels — Blue=negative, Red=positive."""
    absmax = max(abs(float(np.nanmin(d))), abs(float(np.nanmax(d))), 1e-6)
    return np.linspace(-absmax, absmax, n)
