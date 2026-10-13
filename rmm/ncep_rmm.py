"""rmm/ncep_rmm.py — MJO (RMM) index from NCEP/NCAR Reanalysis daily fields.

Method: Wheeler & Hendon (2004), with OLR replaced by the 200-hPa velocity
potential χ200, which is computed from the NCEP u200/v200 fields (rmm/velpot.py).

Steps (the method is WH04; the base period 1991–2020 is the project's choice;
every fitted quantity — seasonal cycle, EOFs, scales — is computed from the data):
1. Daily u850, u200 and χ200, averaged over 15°S–15°N (2.5° grid).
   χ200 is inverted from the full global 200-hPa wind (flux form, no smoothing).
2. Seasonal cycle: a 3-harmonic fit at each grid point over the base period.
3. Trailing 120-day mean removed at each grid point.
4. Each variable is divided by its domain-averaged standard deviation.
5. Combined EOFs (top 2) of the three variables, fitted on the base period.
   The EOF signs are fixed so that eastward propagation appears as
   counter-clockwise rotation in the RMM1–RMM2 plane.
6. Daily fields are projected onto the EOFs and divided by the PC standard
   deviation, giving RMM1 and RMM2.

Usage
-----
    python -m rmm.ncep_rmm build     # one-time: fit the EOFs (reads uwnd and vwnd)
"""
import os
import sys
import datetime

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pro import config          # noqa: E402
from pro import data as pdata   # noqa: E402
from rmm.velpot import VelocityPotential  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EOF_FILE = os.path.join(HERE, "rmm_eofs_ncep.npz")

REF_YEARS = (1991, 2020)   # base period (seasonal cycle, scales, EOFs)
LAT_LIMIT = 15.0           # 15°S–15°N
LEVEL_U_LOW = 850          # hPa
LEVEL_UPPER = 200          # hPa (u200, v200 → χ200)
VARIABLES = ("u850", "u200", "chi200")
LAG = 120                  # trailing-mean length (days)
N_HARM = 3                 # harmonics of the seasonal cycle
EPOCH = datetime.date(1991, 1, 1)
OMEGA = 2.0 * np.pi / 365.25


# ---------------------------------------------------------------------------
# Reading NCEP daily fields
# ---------------------------------------------------------------------------

_VP_CACHE = {}


def _velpot_for(lat, lon):
    key = (lat.tobytes(), lon.tobytes())
    if key not in _VP_CACHE:
        _VP_CACHE[key] = VelocityPotential(lat, lon)
    return _VP_CACHE[key]


def _read_level(ds, varname, hPa):
    """Full (T, lat, lon) field of one pressure level, with NaN for missing values."""
    lev = np.asarray(ds["level"].data, dtype=np.float64)
    lv = int(np.argmin(np.abs(lev - hPa)))
    attr = ds[varname].attributes
    sf = float(attr.get("scale_factor", 1.0))
    ao = float(attr.get("add_offset", 0.0))
    mv = attr.get("missing_value", None)
    raw = np.asarray(ds[varname][:, lv, :, :].data, dtype=np.float64)
    f = raw * sf + ao
    if mv is not None:
        f[raw == float(mv)] = np.nan
    return f


def _read_year(year):
    """One year of u850 (band), u200 (band) and χ200 (band), plus dates and grid.

    χ200 needs the whole globe, so u200/v200 are read in full and only the
    15°S–15°N band of the result is kept.
    """
    ds_u = pdata.open_remote(f"{config.PSL}/uwnd.{year}.nc")
    ds_v = pdata.open_remote(f"{config.PSL}/vwnd.{year}.nc")
    lat = np.asarray(ds_u["lat"].data, dtype=np.float64)
    lon = np.asarray(ds_u["lon"].data, dtype=np.float64)
    band = np.where(np.abs(lat) <= LAT_LIMIT)[0]
    b0, b1 = int(band[0]), int(band[-1]) + 1

    raw_t = np.asarray(ds_u["time"].data, dtype=np.float64)
    units = str(ds_u["time"].attributes.get("units", "hours since 1800-01-01"))
    scale = 1.0 / 24.0 if "hours" in units else 1.0
    epoch = pdata._epoch(ds_u)
    dates = [epoch + datetime.timedelta(days=float(t) * scale) for t in raw_t]

    u850 = _read_level(ds_u, "uwnd", LEVEL_U_LOW)[:, b0:b1, :]
    u200_full = _read_level(ds_u, "uwnd", LEVEL_UPPER)
    v200_full = _read_level(ds_v, "vwnd", LEVEL_UPPER)

    vp = _velpot_for(lat, lon)
    chi_full = vp.chi(np.nan_to_num(u200_full, nan=0.0), np.nan_to_num(v200_full, nan=0.0))
    return dates, lat[b0:b1], lon, u850, u200_full[:, b0:b1, :], chi_full[:, b0:b1, :]


def _read_years(years):
    dates, u8, u2, cx = [], [], [], []
    lat = lon = None
    for y in years:
        d, la, lo, a, b, c = _read_year(y)
        dates += d
        u8.append(a)
        u2.append(b)
        cx.append(c)
        lat, lon = la, lo
    return dates, lat, lon, [np.concatenate(u8), np.concatenate(u2), np.concatenate(cx)]


# ---------------------------------------------------------------------------
# Pure numerical core (no I/O) — shared by the fit and the projection
# ---------------------------------------------------------------------------

def _flatten(a):
    return np.nan_to_num(a.reshape(a.shape[0], -1), nan=0.0)


def _design(dates):
    """Harmonic design matrix on a fixed time axis (days since EPOCH)."""
    t = np.array([(d - EPOCH).days for d in dates], dtype=np.float64)
    cols = [np.ones_like(t)]
    for k in range(1, N_HARM + 1):
        cols += [np.cos(k * OMEGA * t), np.sin(k * OMEGA * t)]
    return np.stack(cols, axis=1)


def _remove_trailing_mean(A):
    """Subtract the mean of the previous LAG days. First LAG rows are NaN."""
    T = A.shape[0]
    cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])
    out = np.full_like(A, np.nan)
    idx = np.arange(LAG, T)
    out[idx] = A[idx] - (cs[idx] - cs[idx - LAG]) / LAG
    return out


def fit_eofs(X_list, dates, ref_mask):
    """Fit the index on the reference period.

    X_list : list of raw daily fields (T, P), one per variable.
    Returns a dict that holds everything needed to project new days.
    """
    T = X_list[0].shape[0]
    design = _design(dates)
    coefs, scales, A_list = [], [], []
    for X in X_list:
        coef, *_ = np.linalg.lstsq(design[ref_mask], X[ref_mask], rcond=None)
        A = _remove_trailing_mean(X - design @ coef)
        coefs.append(coef)
        A_list.append(A)

    valid = ref_mask & (np.arange(T) >= LAG)
    for A in A_list:
        scales.append(np.sqrt(np.mean(A[valid] ** 2)))
    Z = np.hstack([A[valid] / s for A, s in zip(A_list, scales)])

    C = Z.T @ Z / Z.shape[0]
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1][:2]
    eofs = vecs[:, order].T
    pc = Z @ eofs.T

    # EOF signs are arbitrary, and flipping EOF2 reverses the phase rotation.
    # Use the convention that eastward propagation is counter-clockwise in the
    # RMM1–RMM2 plane (as in the official index and the diagram's phase
    # numbering). The signed area swept by the trajectory gives the rotation.
    rot = np.sum(pc[:-1, 0] * pc[1:, 1] - pc[:-1, 1] * pc[1:, 0])
    if rot < 0:
        eofs[1] *= -1
        pc[:, 1] *= -1
    pc_std = pc.std(axis=0)
    return {
        "eofs": eofs,
        "pc_std": pc_std,
        "scales": np.array(scales),
        "coefs": np.stack(coefs),              # (nvar, 1+2*N_HARM, P)
        "explained": vals[order] / vals.sum(),
    }


def project(X_list, dates, pack):
    """RMM1/RMM2 for each day that has a full LAG history. Returns (idx, rmm1, rmm2)."""
    design = _design(dates)
    Z_list = []
    for X, coef, s in zip(X_list, pack["coefs"], pack["scales"]):
        A = _remove_trailing_mean(X - design @ coef)
        Z_list.append(A / s)
    Z = np.hstack(Z_list)
    idx = np.arange(Z.shape[0])[LAG:]
    pc = Z[idx] @ pack["eofs"].T / pack["pc_std"]
    return idx, pc[:, 0], pc[:, 1]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_eofs(out_file=EOF_FILE, ref_years=REF_YEARS):
    """One-time fit on the reference period; writes a small .npz file."""
    years = range(ref_years[0], ref_years[1] + 1)
    dates, lat, lon, X_list = _read_years(years)
    ref_mask = np.array([ref_years[0] <= d.year <= ref_years[1] for d in dates])
    pack = fit_eofs([_flatten(X) for X in X_list], dates, ref_mask)
    np.savez(out_file, lat=lat, lon=lon, ref_years=np.array(ref_years),
             variables=np.array(VARIABLES), **pack)
    return out_file


def _load_pack(eof_file):
    if not os.path.exists(eof_file):
        raise FileNotFoundError(
            f"{eof_file} not found. Build it once with:  python -m rmm.ncep_rmm build"
        )
    z = np.load(eof_file)
    return {k: z[k] for k in z.files}


def _latest_year():
    y = datetime.date.today().year
    try:
        pdata.open_remote(f"{config.PSL}/uwnd.{y}.nc")
        return y
    except Exception:
        return y - 1


def recent_rmm(n_days=40, eof_file=EOF_FILE):
    """RMM1/RMM2 for the last n_days available in the NCEP record.

    Returns (dates as 'YYYY-MM-DD' strings, rmm1 array, rmm2 array).
    """
    pack = _load_pack(eof_file)
    end_year = _latest_year()
    # The previous year is needed for the 120-day lag; fall back if it is missing.
    try:
        dates, lat, lon, X_list = _read_years([end_year - 1, end_year])
    except Exception:
        dates, lat, lon, X_list = _read_years([end_year])
    if not np.allclose(lat, pack["lat"]) or not np.allclose(lon, pack["lon"]):
        raise ValueError("NCEP grid differs from the grid used to build the EOFs")

    idx, r1, r2 = project([_flatten(X) for X in X_list], dates, pack)
    if len(idx) < n_days:
        raise ValueError(f"only {len(idx)} days with a full {LAG}-day history available")
    sl = slice(len(idx) - n_days, None)
    days = [dates[i].isoformat() for i in idx[sl]]
    return days, r1[sl], r2[sl]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        path = build_eofs()
        print(f"EOFs written to {path}")
    else:
        print("usage: python -m rmm.ncep_rmm build")
