"""rmm/ncep_rmm.py — wind-based MJO (RMM) index from NCEP/NCAR Reanalysis.

This follows the Wheeler & Hendon (2004) RMM procedure with one change forced
by the data: the official NOAA index is a combined EOF of OLR, u850 and u200.
The NCEP/NCAR daily files used by this project do not include OLR, so this
index uses u850 and u200 only.  Its RMM1/RMM2 values are therefore an
approximation and are NOT identical to the official NOAA index.

Steps
-----
1. Daily u850 and u200, averaged over 15°S–15°N (all longitudes, 2.5° grid).
2. Remove the seasonal cycle: a 3-harmonic fit per grid point (reference period).
3. Remove the trailing 120-day mean (low-frequency / interannual part).
4. Normalise each variable by its domain-averaged standard deviation.
5. Combined EOFs (top 2) of the two variables, fitted on the reference period.
6. Project daily data on EOF1 / EOF2 and divide by the PC standard deviation,
   giving RMM1 and RMM2.

Usage
-----
    python -m rmm.ncep_rmm build     # one-time: fit EOFs on 1991–2020 (downloads uwnd files)
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

HERE = os.path.dirname(os.path.abspath(__file__))
EOF_FILE = os.path.join(HERE, "rmm_eofs_ncep.npz")

REF_YEARS = (1991, 2020)   # reference period for the seasonal cycle and EOFs
LAT_LIMIT = 15.0           # 15°S–15°N
LEVELS = (850, 200)        # hPa
LAG = 120                  # trailing-mean length (days), Wheeler & Hendon (2004)
N_HARM = 3                 # harmonics of the seasonal cycle
EPOCH = datetime.date(1991, 1, 1)
OMEGA = 2.0 * np.pi / 365.25


# ---------------------------------------------------------------------------
# Reading NCEP daily u850 / u200
# ---------------------------------------------------------------------------

def _read_uwnd(year):
    """One year of u850 and u200 in the 15°S–15°N band.

    Returns (dates, lat_band, lon, u850, u200); u arrays are (T, nlat, nlon).
    """
    ds = pdata.open_remote(f"{config.PSL}/uwnd.{year}.nc")
    lat = np.asarray(ds["lat"].data, dtype=np.float64)
    lon = np.asarray(ds["lon"].data, dtype=np.float64)
    band = np.where(np.abs(lat) <= LAT_LIMIT)[0]
    b0, b1 = int(band[0]), int(band[-1]) + 1

    raw_t = np.asarray(ds["time"].data, dtype=np.float64)
    units = str(ds["time"].attributes.get("units", "hours since 1800-01-01"))
    scale = 1.0 / 24.0 if "hours" in units else 1.0
    epoch = pdata._epoch(ds)
    dates = [epoch + datetime.timedelta(days=float(t) * scale) for t in raw_t]

    attr = ds["uwnd"].attributes
    sf = float(attr.get("scale_factor", 1.0))
    ao = float(attr.get("add_offset", 0.0))
    mv = attr.get("missing_value", None)
    levs = np.asarray(ds["level"].data, dtype=np.float64)

    fields = []
    for hPa in LEVELS:
        lv = int(np.argmin(np.abs(levs - hPa)))
        raw = np.asarray(ds["uwnd"][:, lv, b0:b1, :].data, dtype=np.float64)
        f = raw * sf + ao
        if mv is not None:
            f[raw == float(mv)] = np.nan
        fields.append(f)
    return dates, lat[b0:b1], lon, fields[0], fields[1]


def _read_years(years):
    dates, u8, u2 = [], [], []
    lat = lon = None
    for y in years:
        d, la, lo, a, b = _read_uwnd(y)
        dates += d
        u8.append(a)
        u2.append(b)
        lat, lon = la, lo
    return dates, lat, lon, np.concatenate(u8), np.concatenate(u2)


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

    X_list : [u850 (T,P), u200 (T,P)] raw daily fields, flattened.
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
    # RMM1-RMM2 plane (as in the official index and the phase numbering of the
    # diagram). The signed area swept by the trajectory gives the sense of rotation.
    rot = np.sum(pc[:-1, 0] * pc[1:, 1] - pc[:-1, 1] * pc[1:, 0])
    if rot < 0:
        eofs[1] *= -1
        pc[:, 1] *= -1
    pc_std = pc.std(axis=0)
    return {
        "eofs": eofs,
        "pc_std": pc_std,
        "scales": np.array(scales),
        "coef0": coefs[0],
        "coef1": coefs[1],
        "explained": vals[order] / vals.sum(),
    }


def project(X_list, dates, pack):
    """RMM1/RMM2 for each day that has a full LAG history. Returns (idx, rmm1, rmm2)."""
    design = _design(dates)
    Z_list = []
    for X, coef, s in zip(X_list, [pack["coef0"], pack["coef1"]], pack["scales"]):
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
    dates, lat, lon, u8, u2 = _read_years(years)
    ref_mask = np.array([ref_years[0] <= d.year <= ref_years[1] for d in dates])
    pack = fit_eofs([_flatten(u8), _flatten(u2)], dates, ref_mask)
    np.savez(out_file, lat=lat, lon=lon, ref_years=np.array(ref_years), **pack)
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
        dates, lat, lon, u8, u2 = _read_years([end_year - 1, end_year])
    except Exception:
        dates, lat, lon, u8, u2 = _read_years([end_year])
    if not np.allclose(lat, pack["lat"]) or not np.allclose(lon, pack["lon"]):
        raise ValueError("NCEP grid differs from the grid used to build the EOFs")

    idx, r1, r2 = project([_flatten(u8), _flatten(u2)], dates, pack)
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
