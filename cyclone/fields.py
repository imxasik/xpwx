"""Read one day of the fields used by the cyclone diagnostics.

Files come from the project's PSL/NCEP downloads (``pro.config.PSL``), read with
``pro.data.open_remote``. Packed values are unpacked here with the file's
scale_factor, add_offset and missing_value.

Returned fields: u, v (m/s), T (K), z (gpm), w = omega (Pa/s); lev (hPa, ascending),
lat (ascending), lon (0-360).

NOAA PSL does not publish a daily omega file for the NCEP/NCAR reanalysis (there is
no omega.YEAR.nc on the server). w is therefore diagnosed from the winds by mass
continuity (core.omega_from_divergence), with omega = 0 at the top level.
"""
import threading

import numpy as np

from pro import config
from pro import data as pdata
from cyclone import core

VARS = {"uwnd": "u", "vwnd": "v", "air": "T", "hgt": "z"}

OUT_TOP_HPA = 100.0     # highest level kept for the figures

_LOCK = threading.Lock()
_DAY_CACHE = {}


def _unpack(ds, name, t):
    """(nlev, nlat, nlon) float array of variable ``name`` at time index ``t``."""
    attr = ds[name].attributes
    sf = float(attr.get("scale_factor", 1.0))
    ao = float(attr.get("add_offset", 0.0))
    mv = attr.get("missing_value", None)
    raw = np.asarray(ds[name][t, :, :, :].data, dtype=np.float64)
    f = raw * sf + ao
    if mv is not None:
        f[raw == float(mv)] = np.nan
    return f


def read_day(date):
    """Fields for one day (cached per data root and date)."""
    key = (config.PSL, date.isoformat())
    with _LOCK:
        if key in _DAY_CACHE:
            return _DAY_CACHE[key]
        out = {"date": date}
        lev = lat = lon = None
        for name, short in VARS.items():
            ds = pdata.open_remote(f"{config.PSL}/{name}.{date.year}.nc")
            t = pdata._time_idx(ds, date)
            if lev is None:
                lev = np.asarray(ds["level"].data, dtype=float)
                lat = np.asarray(ds["lat"].data, dtype=float)
                lon = np.asarray(ds["lon"].data, dtype=float)
            out[short] = _unpack(ds, name, t)
        # ascending pressure and latitude
        ol = np.argsort(lev)
        oa = np.argsort(lat)
        for k in "uvTz":
            out[k] = out[k][ol][:, oa, :]
        out["lev"] = lev[ol]
        out["lat"] = lat[oa]
        out["lon"] = lon
        # omega is integrated from the top of the file (10 hPa); the outputs are then
        # kept from 1000 to OUT_TOP_HPA only (the figures end at 100 hPa).
        out["w"] = core.omega_from_divergence(out["u"], out["v"], out["lev"],
                                              out["lat"], out["lon"])
        keep = out["lev"] >= OUT_TOP_HPA - 1e-6
        for k in ("u", "v", "T", "z", "w"):
            out[k] = out[k][keep]
        out["lev"] = out["lev"][keep]
        _DAY_CACHE[key] = out
        return out
