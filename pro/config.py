"""pro.config — module-level constants and process-wide caches."""
import os

# project root = the parent of this package (so the coastline stays in ./map)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SHP_DIR = os.path.join(BASE_DIR, "map")

SHP_PATH = os.path.join(SHP_DIR, "ne_110m_coastline.shp")

SHP_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip"

# NOAA PSL NCEP/NCAR Reanalysis files, served as plain netCDF downloads
# (https://downloads.psl.noaa.gov/Datasets/ncep/).  No trailing slash: the data
# layer appends "/<file>.nc" itself.  Override with the PSL_BASE_URL env var
# (e.g. to point at a local mirror).
PSL = os.environ.get(
    "PSL_BASE_URL", "https://downloads.psl.noaa.gov/Datasets/ncep"
).rstrip("/")

# NOAA OISST v2 high-resolution daily SST (used by custom/sst.py).
OISST = os.environ.get(
    "OISST_BASE_URL", "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
).rstrip("/")

# Downloaded netCDF files are kept here (git-ignored).  Files for the current
# year are still being appended by NOAA, so they are re-downloaded once older
# than CACHE_TTL_HOURS; every other file is kept as-is.
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
CACHE_TTL_HOURS = 12

DEFAULT_N_DAYS = 3

DEFAULT_PRODUCT = "vtp200"

_DS_CACHE = {}       # url -> (local mtime, NCFile)

_FIELD_CACHE = {}    # (var, level, dates, kind) -> mean field

_LATLON_CACHE = {}   # var -> (lat, lon)

_COAST = None        # list of coastline segments

R_EARTH = 6.371e6

AAM_LEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50]

DEG_PER_S = 7.292e-5

KAPPA = 0.2854

P0 = 100000.0

CP = 1004.0        # J/(kg K) dry air

LV = 2.5e6         # J/kg latent heat of vapourisation

GRAV = 9.80665

RD = 287.05        # J/(kg K) gas constant dry air
