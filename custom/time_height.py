"""custom/time_height.py — time–height (pressure) section, box-averaged, from PSL daily fields.

Drop-in addon: no core file is changed. Data come from the project's NOAA PSL
NCEP/NCAR files (``pro.config.PSL``), read with ``pro.data.open_remote`` like the
other products.

Box: the average over the grid points within ±5° latitude and ±10° longitude of the
given centre (lat, lon). The inset map at the top right shows that box.

Layout (time on x, pressure on y, 1000 hPa at the bottom, 100 hPa at the top):
  * shading  : box-mean wind speed (kt)          from uwnd, vwnd
  * barbs    : box-mean wind (kt), every 3 h     from uwnd, vwnd

Window: the selected number of days, ending on the app's selected end date, so
n_days = 1 gives 24 h and n_days = 3 gives 72 h. PSL NCEP/NCAR files hold one record
per day: the record is taken as the day's value (centred on 12 UTC), and shading and
barbs are interpolated linearly in time between those records.

Sidebar group: Hovmöller
"""
import io
import sys
import threading
import datetime
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

from pro import config
from pro import data as pdata

_ROOT = config.BASE_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:                                   # country borders for the inset (same as the cyclone maps)
    from cyclone import borders as _borders
except Exception:                      # pragma: no cover - the inset still works without them
    _borders = None

_LOCK = threading.Lock()     # matplotlib's pyplot state is global
_PROFILE_CACHE = {}          # (PSL root, date, lat, lon) -> box-mean profiles

VARS = {"uwnd": "u", "vwnd": "v"}
BOX_LAT = 5.0                # half-width of the averaging box, degrees latitude
BOX_LON = 10.0               # half-width of the averaging box, degrees longitude
INSET_LAT = 12.0             # half-size of the inset map, degrees
INSET_LON = 20.0
P_TICKS = [1000, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]     # Y-axis labels
# extra levels for shading and barbs only (not labelled). PSL daily files do not hold
# these, so they are interpolated in ln(p) between the file's own levels.
EXTRA_LEVELS_HPA = [950, 900, 800, 750, 650, 550, 450, 350, 175, 125]
SPEED_LEVELS = np.arange(0.0, 75.0, 5.0)   # wind-speed shading bins, kt
BARB_EVERY_H = 3.0           # barb spacing in time, hours
BARB_LENGTH = 6.5            # barb shaft length, points
MS_TO_KT = 1.9438444924
OUT_TOP_HPA = 100.0
COAST_COLOUR = "#2c2c2c"


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


def _box_mean(raw, lat, lon, lat0, lon0):
    """Mean over the grid points inside the box about (lat0, lon0). Returns (nlev,), npts."""
    ii = np.where(np.abs(lat - lat0) <= BOX_LAT + 1e-6)[0]
    jj = np.where(np.abs(((lon - lon0 + 180.0) % 360.0) - 180.0) <= BOX_LON + 1e-6)[0]
    if ii.size == 0 or jj.size == 0:
        raise ValueError("the averaging box lies outside the PSL grid")
    sub = raw[:, ii][:, :, jj]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(sub.reshape(sub.shape[0], -1), axis=1)
    return mean, int(ii.size * jj.size)


def _profile(date, lat0, lon0):
    """Box-mean profile of each variable on ``date``.

    Every variable is read with its own time, level and lat/lon coordinates, so the
    files are never mixed by position. Returns {"u": (lev, values), ..., "npts": n},
    or raises ValueError with the reason if the date is not in the record.
    """
    key = (config.PSL, date.isoformat(), round(lat0, 4), round(lon0, 4))
    with _LOCK:
        if key in _PROFILE_CACHE:
            out = _PROFILE_CACHE[key]
            if isinstance(out, str):
                raise ValueError(out)
            return out
        name = "?"
        try:
            out = {}
            npts = None
            for name, short in VARS.items():
                ds = pdata.open_remote(f"{config.PSL}/{name}.{date.year}.nc")
                t = pdata._time_idx(ds, date)
                lev = np.asarray(ds["level"].data, dtype=float)
                lat = np.asarray(ds["lat"].data, dtype=float)
                lon = np.asarray(ds["lon"].data, dtype=float)
                mean, n = _box_mean(_unpack(ds, name, t), lat, lon, lat0, lon0)
                out[short] = (lev, mean)
                npts = n if npts is None else npts
            out["npts"] = npts
        except ValueError as e:
            _PROFILE_CACHE[key] = str(e)
            raise
        except Exception as e:                       # reported to the user with the file name
            msg = f"{name}.{date.year}: {type(e).__name__}: {e}"
            _PROFILE_CACHE[key] = msg
            raise ValueError(msg) from e
        _PROFILE_CACHE[key] = out
        return out


def _at_levels(lev, values, target):
    """One variable at the target pressures.

    Each file may carry its own set of pressure levels (PSL's rhum has fewer levels than
    uwnd). The values are interpolated linearly in ln(p) to the target levels; a target
    outside the file's range stays NaN. Exact matches return the file value.
    """
    lev = np.asarray(lev, dtype=float)
    values = np.asarray(values, dtype=float)
    o = np.argsort(lev)
    return np.interp(np.log(np.asarray(target, dtype=float)), np.log(lev[o]), values[o],
                     left=np.nan, right=np.nan)


def _latest_with_values(end, lat0, lon0, max_back=60):
    """Most recent date on or before ``end`` whose PSL fields have values in the box."""
    for k in range(max_back):
        d = end - datetime.timedelta(days=k)
        try:
            prof = _profile(d, lat0, lon0)
        except ValueError:
            continue
        if all(np.isfinite(prof[n][1]).any() for n in ("u", "v")):
            return d
    return None


def _levels_for_file(lev_file):
    """Pressure levels of the section: the file's levels from 100 hPa down, plus the extra
    levels that lie inside that range. Ascending."""
    base = np.asarray(lev_file, dtype=float)
    base = base[base >= OUT_TOP_HPA - 1e-6]
    extra = [p for p in EXTRA_LEVELS_HPA if base.min() - 1e-6 <= p <= base.max() + 1e-6]
    return np.unique(np.concatenate([base, extra]))


def _fill_missing(lev, vals):
    """Fill levels with no value (NaN) inside the file's range by linear interpolation in
    ln(p) between the nearest levels that have values. Levels outside that range stay NaN."""
    lev = np.asarray(lev, dtype=float)
    vals = np.asarray(vals, dtype=float)
    o = np.argsort(lev)
    lev, vals = lev[o], vals[o]
    ok = np.isfinite(vals)
    if not ok.any() or ok.all():
        return vals
    out = vals.copy()
    inside = (~ok) & (lev > lev[ok].min()) & (lev < lev[ok].max())
    out[inside] = np.interp(np.log(lev[inside]), np.log(lev[ok]), vals[ok])
    return out


def _compute(pkg, dates):
    lat0 = float(pkg["lat"])
    lon0 = float(pkg["lon"]) % 360.0
    days = list(dates)                       # the selected window, n_days long

    rows = {"u": [], "v": []}
    valid, reasons = [], []
    lev = None
    npts = None
    for d in days:
        try:
            prof = _profile(d, lat0, lon0)
            if lev is None:
                lev = _levels_for_file(prof["u"][0])
                npts = prof["npts"]
            u = _at_levels(prof["u"][0], _fill_missing(prof["u"][0], prof["u"][1]), lev)
            v = _at_levels(prof["v"][0], _fill_missing(prof["v"][0], prof["v"][1]), lev)
            if not (np.isfinite(u).any() and np.isfinite(v).any()):
                raise ValueError("PSL file has no values for this date yet")
        except ValueError as e:
            valid.append(False)
            reasons.append(f"{d.isoformat()}: {e}")
            for k in rows:
                rows[k].append(None)
            continue
        rows["u"].append(u)
        rows["v"].append(v)
        valid.append(True)

    if lev is None or not any(valid):
        latest = _latest_with_values(days[-1], lat0, lon0)
        hint = (f" The latest date with PSL values is {latest:%d %b %Y}; "
                f"choose it as the end date (Manual Date).") if latest else ""
        raise ValueError("no PSL values for the selected days." + hint +
                         " (" + "; ".join(reasons[:2]) + ")")
    if not valid[-1]:
        latest = _latest_with_values(days[-1], lat0, lon0)
        hint = (f" The latest date with PSL values is {latest:%d %b %Y}; "
                f"choose it as the end date (Manual Date).") if latest else ""
        raise ValueError("the end date has no PSL values yet." + hint)

    nl = lev.size
    nan = np.full(nl, np.nan)
    stack = {k: np.stack([r if r is not None else nan for r in v]) for k, v in rows.items()}
    data = {
        "main": np.zeros((1, 1)),
        "dates": days, "valid": np.array(valid),
        "lev": lev, "u": stack["u"], "v": stack["v"],
        "centre": (lat0, lon0), "npts": npts, "reasons": reasons,
    }
    return np.zeros(1), np.zeros(1), data


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _wind_cmap():
    """Sequential palette for wind speed: pale blue (calm) → green → yellow → red (strong)."""
    from matplotlib.colors import LinearSegmentedColormap
    colours = ["#f2f8ff", "#c6e2ff", "#7fb8e6", "#3f8fd2", "#2a9d8f", "#8ac926",
               "#e9c46a", "#f4a261", "#e76f51", "#c1121f", "#6a040f"]
    cmap = LinearSegmentedColormap.from_list("wind_kt", colours, N=len(SPEED_LEVELS) - 1)
    cmap.set_over(colours[-1])
    return cmap


def _series_interp(x, y, tg):
    """Linear interpolation of one series to tg. Gaps of more than one day stay NaN."""
    ok = np.isfinite(y)
    xv, yv = x[ok], y[ok]
    if xv.size == 0:
        return np.full(tg.shape, np.nan)
    out = np.interp(tg, xv, yv)               # flat beyond the first and last record
    for i in range(xv.size - 1):
        if xv[i + 1] - xv[i] > 24.0 + 1e-6:
            out[(tg > xv[i]) & (tg < xv[i + 1])] = np.nan
    return out


def _interp_matrix(x, mat, tg):
    """Interpolate a (nt, nl) matrix column by column to the times tg. Returns (ntg, nl)."""
    out = np.empty((tg.size, mat.shape[1]))
    for k in range(mat.shape[1]):
        out[:, k] = _series_interp(x, mat[:, k], tg)
    return out


def _draw_coast(ax, segs, lon_lo, lon_hi, lat_lo, lat_hi, lw=0.6):
    for seg in segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            if part[:, 0].max() < lon_lo or part[:, 0].min() > lon_hi:
                continue
            if part[:, 1].max() < lat_lo or part[:, 1].min() > lat_hi:
                continue
            ax.plot(part[:, 0], part[:, 1], color=COAST_COLOUR, lw=lw, zorder=5)


def _inset(ax_parent, lat0, lon0, coast_segs):
    """Small map at the top right: the averaging box and the centre, as on the reference chart."""
    ax = ax_parent.inset_axes([0.80, 0.77, 0.18, 0.194])
    lo, hi = lon0 - INSET_LON, lon0 + INSET_LON
    la, lb = lat0 - INSET_LAT, lat0 + INSET_LAT
    ax.set_facecolor("#e9f1f8")
    if _borders is not None:
        try:
            _borders.draw(ax, lo, hi, la, lb)
        except Exception:                 # pragma: no cover - borders are optional
            pass
    _draw_coast(ax, coast_segs, lo, hi, la, lb)
    ax.add_patch(plt.Rectangle((lon0 - BOX_LON, lat0 - BOX_LAT), 2 * BOX_LON, 2 * BOX_LAT,
                               fill=False, ec="#c0392b", lw=1.6, zorder=8))
    ax.plot(lon0, lat0, marker="o", ms=4, color="#c0392b", zorder=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(la, lb)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_edgecolor("#444444")
    return ax


def _plot(data, pkg, coast_segs):
    days = data["dates"]
    nt = len(days)
    lev = data["lev"]                                   # ascending, 100 … 1000 hPa
    lat0, lon0 = data["centre"]
    x_rec = 24.0 * np.arange(nt) + 12.0                  # daily records, centred on 12 UTC
    tg = np.linspace(0.0, 24.0 * nt, 24 * nt * 2 + 1)    # 30-minute grid
    yl = np.log(lev)                                    # pressure axis in ln p

    uu = _interp_matrix(x_rec, data["u"], tg) * MS_TO_KT   # box-mean wind, kt
    vv = _interp_matrix(x_rec, data["v"], tg) * MS_TO_KT
    spd = np.hypot(uu, vv)

    fig = plt.figure(figsize=(14, 8.6), facecolor="white")
    ax = fig.add_axes([0.075, 0.13, 0.80, 0.76])
    cf = ax.contourf(tg, yl, spd.T, levels=SPEED_LEVELS, cmap=_wind_cmap(), extend="max")
    cax = fig.add_axes([0.895, 0.13, 0.012, 0.76])
    cb = fig.colorbar(cf, cax=cax, ticks=list(range(0, int(SPEED_LEVELS[-1]) + 1, 10)))
    cb.set_label("Wind speed (kt)", fontsize=10)

    # wind barbs in knots: every 3 h (interpolated), on every level listed above
    tb = np.arange(0.0, 24.0 * nt + 1e-6, BARB_EVERY_H)
    u_b = _interp_matrix(x_rec, data["u"], tb) * MS_TO_KT
    v_b = _interp_matrix(x_rec, data["v"], tb) * MS_TO_KT
    X, Y = np.meshgrid(tb, yl, indexing="ij")
    ax.barbs(X, Y, u_b, v_b, length=BARB_LENGTH, linewidth=0.9, color="#111111", pivot="middle",
             sizes=dict(emptybarb=0.12, spacing=0.14, height=0.45), zorder=6)

    ax.set_ylim(np.log(lev.max()), np.log(lev.min()))
    ticks = [p for p in P_TICKS if lev.min() <= p <= lev.max()]
    ax.set_yticks(np.log(ticks))
    ax.set_yticklabels([str(p) for p in ticks])
    ax.minorticks_off()
    ax.set_ylabel("Pressure (hPa)", fontsize=10)

    ax.set_xlim(0.0, 24.0 * nt)
    x0 = datetime.datetime.combine(days[0], datetime.time(0))
    xt = np.arange(0.0, 24.0 * nt + 1e-6, 12.0)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(t % 24):02d}" for t in xt], fontsize=8)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for d in range(nt):
        when = x0 + datetime.timedelta(hours=24 * d)
        ax.text(24.0 * d + 12.0, -0.085, when.strftime("%a %d %b"), transform=trans,
                ha="center", va="top", fontsize=9, color="#222222")
        if d > 0:
            ax.axvline(24.0 * d, color="#666666", lw=0.8, ls=":", zorder=4)
    ax.grid(False)

    _inset(ax, lat0, lon0, coast_segs)

    n = data["npts"]
    ax.set_title(f"Time–Height Section · box average about {lat0:.2f}°N {lon0:.2f}°E\n"
                 f"Box ±{BOX_LAT:g}° lat × ±{BOX_LON:g}° lon ({n} grid points) · "
                 f"{nt} day{'s' if nt != 1 else ''} = {24 * nt} h",
                 loc="left", fontsize=11, fontweight="bold", pad=12)
    fig.text(0.975, 0.945, f"Date Range: {days[0]:%d %b %Y} – {days[-1]:%d %b %Y}",
             ha="right", va="bottom", fontsize=10.5, fontweight="bold", color="#222222")
    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=11, va="bottom",
            ha="right", color="#222211", fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccccbb", alpha=0.92, lw=0.9))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return _plot(data, pkg, coast_segs)


PRODUCTS = {
    "time_height_psl": {
        "id": "time_height_psl",
        "title": "Time–Height Section — wind speed and barbs (PSL)",
        "name": "Time–Height",
        "tag": "Hovmöller",
        "desc": "Pressure–time section for a box (±5° lat, ±10° lon) about a lat/lon, from "
                "NOAA PSL NCEP/NCAR daily wind fields: wind-speed shading and barbs for the "
                "selected days (n days = n × 24 h).",
        "kind": "time_height",
        "level": None,
        "needs_latlon": True,
        "uses_days": True,          # the day count box sets the window length
        "cb_label": "Wind speed (kt)",
    },
}

KINDS = {
    "time_height": {
        "compute": _compute,
        "render": _render,
        "tag": "Hovmöller",
        "title": "Time–Height Section — wind speed and barbs (PSL)",
    },
}
