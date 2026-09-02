"""custom/gfs.py — NCEP GFS forecast products.

Added as a drop-in addon: no core engine changes required.

Sidebar group: GFS FC
Time frames: +06h ... +120h (6-hourly, five days)
Products:
  Temperature (2 m)
  Wind at 1000/925/850/700/500/200 hPa
  Sea Level Pressure
  U-Wind / V-Wind at 850 hPa only

The fetch/render path is self-contained so the original GFS forecast project
can be integrated into XPWX's custom/ folder without adding the other GFS
project files to XPWX.
"""

import datetime
import io
import struct
import threading
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# ── GFS source / domain ─────────────────────────────────────────────────────
NOMADS_FILTER_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
NOMADS_FILTER_BASE_3H = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# Default map domain: Bangladesh (same bounds as the supplied GFS project config).
LON_MIN, LON_MAX = 85.0, 95.0
LAT_MIN, LAT_MAX = 20.0, 28.0

FORECAST_HOURS = tuple(range(6, 121, 6))
WIND_LEVELS = (1000, 925, 850, 700, 500, 200)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "XPWeather-GFS/1.0"})
_CACHE = {}
_CACHE_LOCK = threading.Lock()

# ── GRIB2 helpers ───────────────────────────────────────────────────────────
def _u32(b, o): return struct.unpack_from(">I", b, o)[0]
def _u16(b, o): return struct.unpack_from(">H", b, o)[0]
def _u8(b, o): return b[o]
def _i32(b, o):
    v = _u32(b, o)
    return v - (1 << 32) if v & 0x80000000 else v

def _scaled(b, o):
    raw = _u32(b, o)
    sign = -1 if (raw >> 31) else 1
    return sign * (raw & 0x7FFFFFFF)

GFS_PARAM = {
    (0, 0, 0): "TMP",
    (0, 1, 0): "SPFH",
    (0, 1, 1): "RH",
    (0, 2, 2): "UGRD",
    (0, 2, 3): "VGRD",
    (0, 2, 8): "VVEL",
    (0, 3, 1): "PRMSL",
}


def _parse_grib2(data):
    messages = []
    i = 0
    n = len(data)
    while i < n - 16:
        idx = data.find(b"GRIB", i)
        if idx < 0:
            break
        i = idx
        if len(data) < i + 16:
            break
        discipline = _u8(data, i + 6)
        edition = _u8(data, i + 7)
        if edition != 2:
            i += 4
            continue
        try:
            msg_len = struct.unpack_from(">Q", data, i + 8)[0]
        except Exception:
            i += 4
            continue
        if msg_len < 16 or i + msg_len > n:
            i += 4
            continue
        msg = data[i:i + msg_len]
        i += msg_len
        try:
            item = _decode_message(msg, discipline)
            if item:
                messages.append(item)
        except Exception:
            pass
    return messages


def _decode_message(msg, discipline):
    pos = 16
    sec3 = sec4 = sec5 = sec7 = None
    while pos < len(msg) - 4:
        if pos + 5 > len(msg):
            break
        sec_len = _u32(msg, pos)
        if sec_len < 5 or pos + sec_len > len(msg):
            break
        sec_num = _u8(msg, pos + 4)
        if sec_num == 3: sec3 = msg[pos:pos + sec_len]
        elif sec_num == 4: sec4 = msg[pos:pos + sec_len]
        elif sec_num == 5: sec5 = msg[pos:pos + sec_len]
        elif sec_num == 7: sec7 = msg[pos:pos + sec_len]
        elif sec_num == 8: break
        pos += sec_len
    if not all((sec3, sec4, sec5, sec7)):
        return None

    if _u16(sec3, 12) != 0:  # only regular lat/lon grid
        return None
    ni = _u32(sec3, 30)
    nj = _u32(sec3, 34)
    lat1 = _i32(sec3, 46) * 1e-6
    lon1 = _i32(sec3, 50) * 1e-6
    lat2 = _i32(sec3, 55) * 1e-6
    lon2 = _i32(sec3, 59) * 1e-6
    lats = np.linspace(lat1, lat2, nj)
    lons = np.linspace(lon1, lon2, ni)

    pdt = _u16(sec4, 7)
    if pdt not in (0, 1, 2, 8, 11):
        return None
    cat = _u8(sec4, 9)
    param = _u8(sec4, 10)
    ltype = _u8(sec4, 22)
    level_val = (_u32(sec4, 23) / 100.0) if ltype == 100 else _u32(sec4, 23)

    param_name = GFS_PARAM.get((discipline, cat, param),
                               f"d{discipline}c{cat}p{param}")
    level_type = {1: "surface", 2: "mean_sea_level",
                  100: "isobaric", 103: "above_ground_m",
                  200: "entire_atmos"}.get(ltype, f"lt{ltype}")

    ndata = _u32(sec5, 5)
    drt = _u16(sec5, 9)
    if drt != 0:
        return None
    ref = struct.unpack_from(">f", sec5, 11)[0]
    escale = _scaled(sec5, 15)
    dscale = _scaled(sec5, 19)
    nbits = _u8(sec5, 23)
    if nbits <= 0 or nbits > 32:
        return None

    raw = sec7[5:]
    total_bits = ndata * nbits
    needed = (total_bits + 7) // 8
    if len(raw) < needed:
        return None
    packed = np.frombuffer(raw[:needed], dtype=np.uint8)
    bits = np.unpackbits(packed)[:total_bits]
    bits2d = bits.reshape(ndata, nbits)
    powers = (1 << np.arange(nbits - 1, -1, -1, dtype=np.int64))
    x = (bits2d.astype(np.int64) * powers).sum(axis=1)
    values = (float(ref) + x.astype(np.float64) * (2.0 ** float(escale))) / (10.0 ** float(dscale))
    grid = values.reshape(nj, ni)

    if lat1 > lat2:
        grid = grid[::-1, :]
        lats = lats[::-1]

    return {"param": param_name, "level_type": level_type,
            "level": level_val, "lat": lats, "lon": lons,
            "values": grid}


def _find_msg(messages, param, level_type=None, level=None):
    for m in messages:
        if m["param"] != param:
            continue
        if level_type and m["level_type"] != level_type:
            continue
        if level is not None and abs(m["level"] - level) > 5:
            continue
        return m
    return None


def _crop(lat, lon, arr):
    li = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    loi = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    if len(li) == 0 or len(loi) == 0:
        raise RuntimeError("GFS grid does not overlap configured domain")
    return lat[li], lon[loi], arr[np.ix_(li, loi)]


def _grid(messages, param, level_type=None, level=None):
    m = _find_msg(messages, param, level_type, level)
    if m is None:
        raise RuntimeError(f"GFS field '{param}' at {level or level_type} not found")
    return _crop(m["lat"], m["lon"], m["values"])


# ── GFS run / download ──────────────────────────────────────────────────────
def latest_gfs_run_dt():
    """Most recent GFS cycle expected to be safely available on NOMADS."""
    now = datetime.datetime.utcnow()
    candidates = []
    for delta_days in range(2):
        day = now - datetime.timedelta(days=delta_days)
        for hour in (18, 12, 6, 0):
            dt = day.replace(hour=hour, minute=0, second=0, microsecond=0)
            if dt <= now - datetime.timedelta(hours=3):
                candidates.append(dt)
    if not candidates:
        raise RuntimeError("Unable to determine a recent GFS run")
    return candidates[0]


def _gfs_url(run_dt, step, var_flags, level_flags, base):
    date_str = run_dt.strftime("%Y%m%d")
    hr_str = run_dt.strftime("%H")
    fname = f"gfs.t{hr_str}z.pgrb2.0p25.f{step:03d}"
    params = {
        "file": fname,
        "leftlon": LON_MIN, "rightlon": LON_MAX,
        "toplat": LAT_MAX, "bottomlat": LAT_MIN,
        "dir": f"/gfs.{date_str}/{hr_str}/atmos",
    }
    params.update(var_flags)
    params.update(level_flags)
    return base + "?" + "&".join(f"{k}={v}" for k, v in params.items())


def _download(run_dt, step, var_flags, level_flags):
    # Try the requested cycle, then the previous cycle at step+6 as in the
    # supplied GFS project's robust fetch strategy.
    attempts = ((run_dt, step), (run_dt - datetime.timedelta(hours=6), step + 6))
    for base in (NOMADS_FILTER_BASE, NOMADS_FILTER_BASE_3H):
        for try_run, try_step in attempts:
            url = _gfs_url(try_run, try_step, var_flags, level_flags, base)
            try:
                r = _SESSION.get(url, timeout=90)
                if r.status_code != 200:
                    continue
                ct = r.headers.get("Content-Type", "")
                if "html" in ct.lower():
                    continue
                if len(r.content) > 500 and r.content[:4] == b"GRIB":
                    return r.content, try_run, try_step
            except Exception:
                continue
    raise RuntimeError("GFS download failed on both NOMADS endpoints")


def _fetch(step, product, level=None):
    run_dt = latest_gfs_run_dt()
    key = (run_dt, step, product, level)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    if product == "temp":
        raw, used_run, used_step = _download(
            run_dt, step, {"var_TMP": "on"}, {"lev_2_m_above_ground": "on"})
        messages = _parse_grib2(raw)
        lat, lon, data = _grid(messages, "TMP", "above_ground_m", 2)
        data = data - 273.15
    elif product == "mslp":
        raw, used_run, used_step = _download(
            run_dt, step, {"var_PRMSL": "on"}, {"lev_mean_sea_level": "on"})
        messages = _parse_grib2(raw)
        lat, lon, data = _grid(messages, "PRMSL", "mean_sea_level")
        data = data / 100.0
    elif product in ("wind", "u", "v"):
        raw, used_run, used_step = _download(
            run_dt, step,
            {"var_UGRD": "on", "var_VGRD": "on"},
            {f"lev_{level}_mb": "on"})
        messages = _parse_grib2(raw)
        lat, lon, u = _grid(messages, "UGRD", "isobaric", level)
        _, _, v = _grid(messages, "VGRD", "isobaric", level)
        if product == "wind":
            data = np.sqrt(u ** 2 + v ** 2)
            extra = {"u": u, "v": v}
        elif product == "u":
            data = u
            extra = None
        else:
            data = v
            extra = None
    else:
        raise ValueError(f"Unknown GFS product: {product}")

    result = (lat, lon, np.asarray(data, dtype=float), extra if product == "wind" else None,
              used_run, used_step)
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result


# ── Rendering ───────────────────────────────────────────────────────────────
def _cmap_temp():
    return plt.get_cmap("turbo")


def _cmap_signed():
    return plt.get_cmap("RdBu_r")


def _cmap_pressure():
    return plt.get_cmap("viridis")


def _draw_map(lat, lon, field, pkg, coast_segs, title, cb_label, levels, cmap,
              extra=None):
    LON2D, LAT2D = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(11, 8), facecolor="white", dpi=190)
    ax = fig.add_axes([0.07, 0.11, 0.86, 0.80])
    ax.set_facecolor("#f4f0e8")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    # Smooth only for display; preserve the original min/max range.
    sf = gaussian_filter(np.nan_to_num(field, nan=np.nanmean(field)), sigma=1.0)
    cf = ax.contourf(LON2D, LAT2D, sf, levels=levels, cmap=cmap,
                     extend="both", zorder=1, alpha=0.90)
    ax.contour(LON2D, LAT2D, sf, levels=levels[::max(1, len(levels)//12)],
               colors="#333333", linewidths=0.45, alpha=0.45, zorder=2)

    if extra and "u" in extra:
        u, v = extra["u"], extra["v"]
        step = 6
        q = slice(None, None, step)
        ax.quiver(LON2D[q, q], LAT2D[q, q], u[q, q], v[q, q],
                  color="#111111", scale=650, width=0.0017,
                  headwidth=4.2, headlength=5.2, headaxislength=4.7,
                  pivot="middle", zorder=6, alpha=0.88)
        rx = LON_MIN + 0.18 * (LON_MAX - LON_MIN)
        ry = LAT_MIN + 0.07 * (LAT_MAX - LAT_MIN)
        ax.quiver(rx, ry, 10, 0, color="#111111", scale=650,
                  scale_units="inches", width=0.0017, zorder=9)
        ax.text(rx + 1.2, ry + 0.7, "10 m/s", fontsize=8.5, color="#111111", zorder=9)

    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            if len(part) > 1:
                ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.75, zorder=7)

    for x in range(70, 101, 10):
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.65)
    for y in range(10, 41, 10):
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.65)

    ax.set_xticks(range(70, 101, 10))
    ax.set_yticks(range(10, 41, 10))
    ax.set_xticklabels([f"{x}°E" for x in range(70, 101, 10)], fontsize=9)
    ax.set_yticklabels([f"{y}°N" for y in range(10, 41, 10)], fontsize=9)
    ax.tick_params(length=3, color="#888878", width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988")
        spine.set_linewidth(0.8)

    cax = fig.add_axes([0.15, 0.045, 0.70, 0.026])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8, colors="#222211", length=3)
    cbar.outline.set_edgecolor("#999988")
    cbar.outline.set_linewidth(0.7)
    cax.text(0.5, -1.65, cb_label, transform=cax.transAxes,
             ha="center", va="top", fontsize=11, color="#222211", fontstyle="italic")

    fig.text(0.50, 0.965, title, ha="center", va="top", fontsize=15,
             fontweight="bold", color="#111100")
    ax.text(0.985, 0.015, "@XPWEATHER", transform=ax.transAxes,
            fontsize=10.5, va="bottom", ha="right", color="#222211",
            fontweight="semibold", bbox=dict(boxstyle="round,pad=0.30",
            fc="white", ec="#ccccbb", alpha=0.90, lw=0.8), zorder=10)
    ax.text(0.005, 0.015, "NCEP GFS · NOMADS/NOAA", transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left", color="#666655", zorder=10)

    out = io.BytesIO()
    plt.savefig(out, format="png", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    out.seek(0)
    return out


def _render_gfs(lat, lon, data, pkg, coast_segs, dates, **_kw):
    hour = int(pkg["forecast_hour"])
    run_dt = data.get("_run_dt")
    used_step = data.get("_used_step", hour)
    field = data["main"]
    extra = {k: data[k] for k in ("u", "v") if k in data}

    if pkg["gfs_product"] == "temp":
        vmin, vmax = np.nanpercentile(field, [2, 98])
        vmin = min(vmin, -5); vmax = max(vmax, 35)
        levels = np.linspace(vmin, vmax, 31)
        cmap = _cmap_temp()
        cb = "2 m Temperature (°C)"
    elif pkg["gfs_product"] == "mslp":
        vmin, vmax = np.nanpercentile(field, [2, 98])
        levels = np.arange(np.floor(vmin / 2) * 2, np.ceil(vmax / 2) * 2 + 0.1, 2)
        cmap = _cmap_pressure()
        cb = "Sea Level Pressure (hPa)"
    elif pkg["gfs_product"] == "wind":
        vmax = max(20.0, float(np.nanpercentile(field, 98)))
        levels = np.linspace(0, vmax, 25)
        cmap = plt.get_cmap("YlGnBu")
        cb = f"Wind Speed ({pkg['level']} hPa) (m/s)"
    else:
        vmax = max(10.0, float(np.nanpercentile(np.abs(field), 98)))
        levels = np.linspace(-vmax, vmax, 25)
        cmap = _cmap_signed()
        comp = "U-Wind" if pkg["gfs_product"] == "u" else "V-Wind"
        cb = f"{comp} ({pkg['level']} hPa) (m/s)"

    run_label = run_dt.strftime("%Y-%m-%d %HZ") if run_dt else "latest GFS run"
    valid_dt = run_dt + datetime.timedelta(hours=used_step) if run_dt else None
    valid_label = valid_dt.strftime("%Y-%m-%d %HZ") if valid_dt else f"+{hour:02d}h"
    title = f"{pkg['base_label']} · +{hour:02d}h  |  Valid {valid_label}  |  Run {run_label}"
    return _draw_map(lat, lon, field, pkg, coast_segs, title, cb, levels, cmap,
                     extra=extra or None)


# ── Custom-kind compute ─────────────────────────────────────────────────────
def _compute_gfs(pkg, dates):
    hour = int(pkg["forecast_hour"])
    lat, lon, field, extra, run_dt, used_step = _fetch(hour, pkg["gfs_product"], pkg.get("level"))
    data = {"main": field, "_run_dt": run_dt, "_used_step": used_step}
    if extra:
        data.update(extra)
    return lat, lon, data


# ── Product registry ────────────────────────────────────────────────────────
PRODUCTS = {}


def _add(pid, base_label, gfs_product, level=None):
    for hour in FORECAST_HOURS:
        full_id = f"{pid}_{hour}"
        PRODUCTS[full_id] = {
            "id": full_id,
            "title": f"{base_label} · +{hour:02d}h",
            "name": base_label,
            "tag": "GFS FC",
            "desc": f"GFS forecast · +{hour:02d}h",
            "kind": "gfs_fc",
            "level": level,
            "forecast_hour": hour,
            "gfs_product": gfs_product,
            "base_label": base_label,
        }


_add("gfs_temp", "Temperature", "temp")
for _lev in WIND_LEVELS:
    _add(f"gfs_wind_{_lev}", f"Wind {_lev}", "wind", _lev)
_add("gfs_mslp", "Sea Level Pressure", "mslp")
for _lev in (850,):
    _add(f"gfs_u_{_lev}", f"U-Wind {_lev}", "u", _lev)
    _add(f"gfs_v_{_lev}", f"V-Wind {_lev}", "v", _lev)


KINDS = {
    "gfs_fc": {
        "compute": _compute_gfs,
        "render": _render_gfs,
        "tag": "GFS FC",
        "title": "NCEP GFS Forecast",
    }
}
