"""custom/time_height.py — time–height (pressure) section at a point, from PSL data.

Drop-in addon: no core file is changed. Data come from the project's NOAA PSL
NCEP/NCAR files (``pro.config.PSL``), read with ``pro.data.open_remote``, the same
way as the other products.

Layout (time on x, pressure on y, 1000 hPa at the bottom, 100 hPa at the top):
  * shading  : relative humidity (%)            from rhum
  * barbs    : wind (kt)                        from uwnd, vwnd
  * contours : stability  -dθe/dp  (K/mb)      from air, rhum (θe by Bolton 1980)

Window: the selected number of days, ending on the app's selected end date, so
n_days = 1 gives 24 h and n_days = 3 gives 72 h. PSL NCEP/NCAR files hold one
record per day, so the time axis is daily: one column per day, centred on the
day, with a 24-h cell per record. The point is the PSL grid point nearest the
given lat/lon.

Sidebar group: Hovmöller
"""
import io
import sys
import threading
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

from pro import config
from pro import data as pdata

_ROOT = config.BASE_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LOCK = threading.Lock()     # matplotlib's pyplot state is global
_PROFILE_CACHE = {}          # (PSL root, date, lat, lon) -> profiles at one grid point

VARS = {"uwnd": "u", "vwnd": "v", "air": "T", "rhum": "rh"}
P_TICKS = [1000, 850, 700, 500, 300, 200, 100]
RH_LEVELS = np.arange(0, 101, 5)
STAB_LEVELS = [0.025, 0.05, 0.1, 0.2]
BARB_LEVELS = [1000, 850, 700, 500, 300, 200, 100]
MS_TO_KT = 1.9438444924
OUT_TOP_HPA = 100.0


def _unpack(ds, name, t):
    """(nlev, nlat, nlon) float array of ``name`` at record ``t``, unpacked and NaN-masked."""
    attr = ds[name].attributes
    sf = float(attr.get("scale_factor", 1.0))
    ao = float(attr.get("add_offset", 0.0))
    mv = attr.get("missing_value", None)
    raw = np.asarray(ds[name][t, :, :, :].data, dtype=np.float64)
    f = raw * sf + ao
    if mv is not None:
        f[raw == float(mv)] = np.nan
    return f


def _profile(date, lat0, lon0):
    """Each variable at the grid point nearest (lat0, lon0) on ``date``.

    Every variable is read with its own time, level and lat/lon coordinates, so the
    files are never mixed by position. Returns {"u": (lev, values, (lat, lon)), ...},
    or raises ValueError with the reason if the date is not in the record.
    """
    key = (config.PSL, date.isoformat(), round(lat0, 4), round(lon0, 4))
    with _LOCK:
        if key in _PROFILE_CACHE:
            out = _PROFILE_CACHE[key]
            if isinstance(out, str):
                raise ValueError(out)
            return out
        try:
            out = {}
            for name, short in VARS.items():
                ds = pdata.open_remote(f"{config.PSL}/{name}.{date.year}.nc")
                t = pdata._time_idx(ds, date)
                lev = np.asarray(ds["level"].data, dtype=float)
                lat = np.asarray(ds["lat"].data, dtype=float)
                lon = np.asarray(ds["lon"].data, dtype=float)
                i = int(np.argmin(np.abs(lat - lat0)))
                j = int(np.argmin(np.abs((lon - lon0 + 180.0) % 360.0 - 180.0)))
                raw = _unpack(ds, name, t)
                out[short] = (lev, raw[:, i, j], (float(lat[i]), float(lon[j])))
        except ValueError as e:
            _PROFILE_CACHE[key] = str(e)
            raise
        except Exception as e:                       # pragma: no cover - reported to the user
            msg = f"{name}.{date.year}: {type(e).__name__}: {e}"
            _PROFILE_CACHE[key] = msg
            raise ValueError(msg) from e
        _PROFILE_CACHE[key] = out
        return out


def _at_levels(lev, values, target):
    """Values of one variable at the target pressures (matched by value, not by position)."""
    idx = []
    for p in target:
        k = int(np.argmin(np.abs(lev - p)))
        if abs(lev[k] - p) > 0.5:
            raise ValueError(f"level {p:g} hPa is not in the file")
        idx.append(k)
    return values[idx]


def _theta_e(T, rh, p):
    """Equivalent potential temperature (K) on a column. T in K, rh in %, p in hPa (Bolton 1980)."""
    Tc = T - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    e = np.clip(rh, 0.1, 100.0) / 100.0 * es
    e = np.minimum(e, 0.5 * p)
    r = 0.622 * e / (p - e)                                   # kg/kg
    td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112)) + 273.15
    tl = 56.0 + 1.0 / (1.0 / (td - 56.0) + np.log(T / td) / 800.0)
    return T * (1000.0 / p) ** (0.2854 * (1.0 - 0.28 * r)) * \
        np.exp((3376.0 / tl - 2.54) * r * (1.0 + 0.81 * r))


def _stability(theta_e, p):
    """-dθe/dp along the level axis (axis 0), K per hPa (= K/mb)."""
    return -np.gradient(theta_e, p, axis=0)


def _compute(pkg, dates):
    lat0 = float(pkg["lat"])
    lon0 = float(pkg["lon"]) % 360.0
    days = list(dates)                       # the selected window, n_days long

    rows = {"rh": [], "u": [], "v": [], "stab": []}
    valid, reasons = [], []
    lev = None
    point = None
    for d in days:
        try:
            prof = _profile(d, lat0, lon0)
            if lev is None:
                lev_u = prof["u"][0]
                lev = np.sort(lev_u[lev_u >= OUT_TOP_HPA - 1e-6])          # ascending, 100 … 1000
                point = prof["u"][2]
            u = _at_levels(prof["u"][0], prof["u"][1], lev)
            v = _at_levels(prof["v"][0], prof["v"][1], lev)
            T = _at_levels(prof["T"][0], prof["T"][1], lev)
            rh = _at_levels(prof["rh"][0], prof["rh"][1], lev)
        except ValueError as e:
            valid.append(False)
            reasons.append(f"{d.isoformat()}: {e}")
            for k in rows:
                rows[k].append(None)
            continue
        te = _theta_e(T, rh, lev)
        rows["rh"].append(rh)
        rows["u"].append(u)
        rows["v"].append(v)
        rows["stab"].append(_stability(te, lev))
        valid.append(True)

    if lev is None:
        raise ValueError("no PSL record for the selected days (" + "; ".join(reasons[:2]) + ")")
    nl = lev.size
    nan = np.full(nl, np.nan)
    stack = {k: np.stack([r if r is not None else nan for r in v]) for k, v in rows.items()}
    data = {
        "main": np.zeros((1, 1)),
        "dates": days, "valid": np.array(valid),
        "lev": lev, "rh": stack["rh"], "u": stack["u"], "v": stack["v"], "stab": stack["stab"],
        "point": point, "req": (lat0, lon0), "reasons": reasons,
    }
    return np.zeros(1), np.zeros(1), data


def _edges_log(v):
    """Cell edges for pressure centres v (ascending), in log space."""
    lv = np.log(v)
    mid = 0.5 * (lv[:-1] + lv[1:])
    first = lv[0] - (mid[0] - lv[0])
    last = lv[-1] + (lv[-1] - mid[-1])
    return np.exp(np.concatenate([[first], mid, [last]]))


def _plot(data, pkg):
    days = data["dates"]
    nt = len(days)
    lev = data["lev"]                                  # ascending, 100 … 1000 hPa
    ok = data["valid"]
    x0 = datetime.datetime.combine(days[0], datetime.time(0))
    x_edges = np.arange(0, 24.0 * (nt + 1), 24.0)      # one 24-h cell per record
    x_mid = x_edges[:-1] + 12.0                        # barbs and contours at day centres

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    rh = np.where(ok[:, None], data["rh"], np.nan)
    cf = ax.pcolormesh(x_edges, _edges_log(lev), rh.T, cmap=_rh_cmap(), vmin=0, vmax=100,
                       shading="flat")
    cb = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.035, aspect=40,
                      ticks=list(range(0, 101, 10)))
    cb.set_label("Relative humidity (%)", fontsize=10)

    stab = np.where(ok[:, None], data["stab"], np.nan)
    if nt >= 2 and np.isfinite(stab).sum() > 4:
        cs = ax.contour(x_mid, lev, stab.T, levels=STAB_LEVELS, colors="#1aa7b8", linewidths=1.2)
        ax.clabel(cs, fmt="%g", fontsize=8, inline=True, inline_spacing=6)

    # wind barbs in knots, at each day centre and on the chosen levels
    bl = [p for p in BARB_LEVELS if lev.min() - 1e-6 <= p <= lev.max() + 1e-6]
    li = [int(np.argmin(np.abs(lev - p))) for p in bl]
    u_kt = np.where(ok[:, None], data["u"], np.nan)[:, li] * MS_TO_KT
    v_kt = np.where(ok[:, None], data["v"], np.nan)[:, li] * MS_TO_KT
    X, Y = np.meshgrid(x_mid, lev[li], indexing="ij")
    ax.barbs(X, Y, u_kt, v_kt, length=6.2, linewidth=0.7, color="#111111", pivot="tip",
             sizes=dict(spacing=0.15, height=0.45))

    ax.set_yscale("log")
    ax.set_ylim(float(lev.max()), float(lev.min()))
    ax.set_yticks([p for p in P_TICKS if lev.min() <= p <= lev.max()])
    ax.set_yticklabels([str(p) for p in P_TICKS if lev.min() <= p <= lev.max()])
    ax.minorticks_off()
    ax.set_ylabel("Pressure (hPa)", fontsize=10)

    ax.set_xlim(0, 24.0 * nt)
    ax.set_xticks(x_edges)
    ax.set_xticklabels([(x0 + datetime.timedelta(hours=float(h))).strftime("%d %b\n%H") + "Z"
                        for h in x_edges], fontsize=8)
    ax.set_xlabel("Date (UTC) · one record per day, 24 h apart", fontsize=10)
    ax.grid(True, axis="x", color="#dddddd", lw=0.5)

    plat, plon = data["point"]
    rlat, rlon = data["req"]
    ax.set_title(f"Time–Height Section · PSL NCEP/NCAR daily fields\n"
                 f"Point {rlat:.2f}°N {rlon:.2f}°E (PSL grid {plat:g}°N {plon:g}°E) · "
                 f"{nt} day{'s' if nt != 1 else ''} = {24 * nt} h",
                 loc="left", fontsize=11, fontweight="bold", pad=10)
    fig.text(0.01, 0.01, "Shading: RH (%) · barbs: wind (kt) · contours: −dθe/dp (K/mb)",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _rh_cmap():
    cols = ["#7a4a12", "#b07d3a", "#dcc38f", "#f2ead6", "#c9e8c4", "#7fc27a", "#2f8f3f", "#1a5f2a"]
    return mcolors.LinearSegmentedColormap.from_list("rh_tw", cols, N=20)


def _render(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return _plot(data, pkg)


PRODUCTS = {
    "time_height_psl": {
        "id": "time_height_psl",
        "title": "Time–Height Section — RH, wind and stability (PSL)",
        "name": "Time–Height",
        "tag": "Hovmöller",
        "desc": "Pressure–time section at a lat/lon: relative humidity, wind barbs and "
                "−dθe/dp from NOAA PSL NCEP/NCAR daily fields, for the selected days "
                "(n days = n × 24 h).",
        "kind": "time_height",
        "level": None,
        "needs_latlon": True,
        "uses_days": True,          # the day count box sets the window length
        "cb_label": "Relative humidity (%)",
    },
}

KINDS = {
    "time_height": {
        "compute": _compute,
        "render": _render,
        "tag": "Hovmöller",
        "title": "Time–Height Section — RH, wind and stability (PSL)",
    },
}
