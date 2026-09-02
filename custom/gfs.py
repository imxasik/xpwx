"""custom/gfs.py — GFS Forecast Maps (NCEP NOMADS real-time)

Product families (rendered by JS time-frame slider in sidebar):
  Temperature 2m       — 2 m air temperature
  Wind 1000/925/850/   — Wind speed + streamlines at pressure levels
       700/500/200 mb
  Sea Level Pressure   — MSLP contour shading
  U-Wind (isobaric)    — Zonal component at selected levels
  V-Wind (isobaric)    — Meridional component at selected levels

Time frames: +06, +12, +18, +24, +30, +36, +42, +48, +54, +60, +66,
             +72, +84, +96, +108, +120 h  (5-day range, 6-h steps)

All fetch calls go to NOMADS GRIB2 filter endpoint (0.25° 1hr / 3hr).
A pure-Python GRIB2 parser is used — no eccodes / cfgrib dependency.

Product IDs follow the pattern:
    gfs_<family>_<step>h
  e.g.  gfs_temp_06h, gfs_wind850_36h, gfs_mslp_120h

tag: "GFS FC" (appears as the group header in the sidebar)
"""

from __future__ import annotations

import io
import os
import struct
import datetime
import time
import warnings
import threading
import functools
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import requests
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from pro import config  # BASE_DIR, SHP_PATH, coastline utilities

# ── NOMADS endpoints ────────────────────────────────────────────────────────
_NOMADS_1H = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
_NOMADS_3H = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# ── module-level caches ─────────────────────────────────────────────────────
_GRIB_CACHE: dict = {}   # (run_str, step, var_flags, level_flags) -> raw bytes
_FIELD_CACHE: dict = {}  # product_id -> (png_bytes, fetched_at)
_CACHE_TTL = 3600        # 1 hour — GFS 0z/6z/12z/18z; cache avoids duplicate fetches

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "xpwx-gfs/2.0 (+https://xpweather.com)"})

# ── Time-frame steps (hours) for the 5-day slider ──────────────────────────
GFS_STEPS = list(range(6, 121, 6))

# ── Isobaric wind levels ─────────────────────────────────────────────────────
WIND_LEVELS = [1000, 925, 850, 700, 500, 200]

# ── GRIB2 parameter table (discipline-category-number → short name) ─────────
_PARAM = {
    (0, 0, 0): "TMP", (0, 1, 1): "RH",   (0, 1, 3): "PWAT",
    (0, 1, 8): "APCP",(0, 2, 2): "UGRD", (0, 2, 3): "VGRD",
    (0, 3, 1): "PRMSL",(0, 3, 5): "HGT",
}
_LEVEL_TYPE = {1: "surface", 2: "msl", 100: "isobaric", 103: "height_agl"}


# ═══════════════════════════════════════════════════════════════════════════
#  GRIB2 helpers (pure-Python, no eccodes)
# ═══════════════════════════════════════════════════════════════════════════

def _u32(b, o): return struct.unpack_from(">I", b, o)[0]
def _u16(b, o): return struct.unpack_from(">H", b, o)[0]
def _u8(b, o):  return b[o]
def _i32(b, o):
    v = _u32(b, o)
    return v - (1 << 32) if v & 0x80000000 else v


def _decode_grib2(msg: bytes, discipline: int):
    """Decode a single GRIB2 message. Returns dict or None."""
    pos = 16
    # Section 1 (Identification)
    sec1_len = _u32(msg, pos)
    year = _u16(msg, pos + 12); mon = _u8(msg, pos + 14); day = _u8(msg, pos + 15)
    hour = _u8(msg, pos + 16)
    pos += sec1_len
    # Skip sections 2 & 3 if present
    while pos < len(msg) - 5:
        slen = _u32(msg, pos); snum = _u8(msg, pos + 4)
        if snum == 4: break
        pos += slen
    # Section 4 (Product Definition)
    if pos + 34 > len(msg): return None
    sec4_len = _u32(msg, pos)
    pdt = _u16(msg, pos + 7)
    cat  = _u8(msg, pos + 9)
    num  = _u8(msg, pos + 10)
    level_type = _u8(msg, pos + 22)
    level_val  = _u32(msg, pos + 23)
    param = _PARAM.get((discipline, cat, num))
    lvl_str = _LEVEL_TYPE.get(level_type, f"t{level_type}")
    pos += sec4_len
    # Skip section 5 header
    sec5_len = _u32(msg, pos)
    tmpl = _u16(msg, pos + 9)
    ref_f = struct.unpack_from(">f", msg, pos + 11)[0]
    e_bin = struct.unpack_from(">h", msg, pos + 15)[0]
    d_dec = struct.unpack_from(">h", msg, pos + 17)[0]
    nbits = _u8(msg, pos + 19)
    n_pts = _u32(msg, pos + 5)
    pos += sec5_len
    # Section 6 (Bitmap)
    sec6_len = _u32(msg, pos); bitmap_flag = _u8(msg, pos + 5)
    bitmap = None
    if bitmap_flag == 0:
        bm_bytes = msg[pos + 6: pos + sec6_len]
        bitmap = np.unpackbits(np.frombuffer(bm_bytes, dtype=np.uint8))
    pos += sec6_len
    # Section 7 (Data)
    sec7_len = _u32(msg, pos)
    raw_data = msg[pos + 5: pos + sec7_len]
    pos += sec7_len
    if nbits == 0 or len(raw_data) == 0: return None
    # Unpack bit-packed values
    bits = np.unpackbits(np.frombuffer(raw_data, dtype=np.uint8))
    # GRIB2 section 7 is byte-padded. Decode exactly n_pts values;
    # do not treat the padding bits as additional grid points.
    needed_bits = int(n_pts) * int(nbits)
    if needed_bits <= 0 or needed_bits > len(bits):
        return None
    packed = bits[:needed_bits].reshape(int(n_pts), nbits)
    vals = packed.dot(1 << np.arange(nbits - 1, -1, -1, dtype=np.int64)).astype(np.float64)
    R = ref_f * (2 ** e_bin) / (10 ** d_dec)
    scale = (2 ** e_bin) / (10 ** d_dec)
    vals = R + scale * vals
    if bitmap is not None:
        full = np.full(int(bitmap[:n_pts + 8].sum() + len(vals)), np.nan)
        # simpler: just use vals directly (bitmap rarely needed for GFS)
        pass
    return {"param": param, "level_type": lvl_str, "level": float(level_val),
            "values_1d": vals, "ref_time": datetime.datetime(year, mon, day, hour)}


def parse_grib2(data: bytes):
    """Parse all GRIB2 messages from raw bytes. Returns list of dicts."""
    messages = []
    i = 0; n = len(data)
    while i < n - 16:
        idx = data.find(b'GRIB', i)
        if idx == -1: break
        i = idx
        if len(data) < i + 16: break
        disc = _u8(data, i + 6); ed = _u8(data, i + 7)
        if ed != 2: i += 4; continue
        try:
            msg_len = struct.unpack_from(">Q", data, i + 8)[0]
        except Exception: i += 4; continue
        if msg_len < 16 or i + msg_len > n: i += 4; continue
        msg = data[i: i + msg_len]; i += msg_len
        try:
            r = _decode_grib2(msg, disc)
            if r: messages.append(r)
        except Exception:
            pass
    return messages


# ═══════════════════════════════════════════════════════════════════════════
#  GFS run helpers
# ═══════════════════════════════════════════════════════════════════════════

def _latest_run() -> datetime.datetime:
    now = datetime.datetime.utcnow()
    for d in range(2):
        day = now - datetime.timedelta(days=d)
        for h in (18, 12, 6, 0):
            c = day.replace(hour=h, minute=0, second=0, microsecond=0)
            if c <= now - datetime.timedelta(hours=3):
                return c
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _run_id(run: datetime.datetime) -> tuple:
    return run.strftime("%Y%m%d"), f"{run.hour:02d}"


# ═══════════════════════════════════════════════════════════════════════════
#  NOMADS fetch
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_nomads(step: int, var_flags: dict, level_flags: dict,
                  run: datetime.datetime | None = None) -> bytes | None:
    """Fetch GRIB2 bytes from NOMADS for given step + variable/level flags."""
    if run is None:
        run = _latest_run()
    date_str, hr_str = _run_id(run)

    for base in [_NOMADS_1H, _NOMADS_3H]:
        params = {
            "file": f"gfs.t{hr_str}z.pgrb2.0p25.f{step:03d}",
            "dir":  f"/gfs.{date_str}/{hr_str}/atmos",
            **var_flags,
            **level_flags,
        }
        try:
            r = _SESSION.get(base, params=params, timeout=90)
            if r.status_code == 200 and len(r.content) > 500 and r.content[:4] == b'GRIB':
                return r.content
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Variable-specific fetchers  →  (lat, lon, *fields)
# ═══════════════════════════════════════════════════════════════════════════

def _std_grid(ny=721, nx=1440):
    """Standard GFS 0.25° global grid."""
    lat = np.linspace(90, -90, ny)
    lon = np.linspace(0, 359.75, nx)
    return lat, lon


def _reshape(vals_1d, ny=721, nx=1440):
    if len(vals_1d) == ny * nx:
        return vals_1d.reshape(ny, nx)
    # fallback for smaller arrays
    n = int(np.round(np.sqrt(len(vals_1d) * 2)))
    ny2 = n // 2; nx2 = n
    if ny2 * nx2 == len(vals_1d):
        return vals_1d.reshape(ny2, nx2)
    return vals_1d.reshape(-1, nx)


def _find(msgs, param, level_type=None, level=None):
    for m in msgs:
        if m["param"] != param: continue
        if level_type and m["level_type"] != level_type: continue
        if level is not None and abs(m["level"] - level) > 5: continue
        return m
    return None


def _crop(lat, lon, arr, lat_min, lat_max, lon_min, lon_max):
    li  = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    loi = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if len(li) == 0 or len(loi) == 0:
        return lat, lon, arr
    return lat[li], lon[loi], arr[np.ix_(li, loi)]


def fetch_temp2m(step: int, region: tuple) -> tuple:
    """2 m temperature in °C."""
    raw = _fetch_nomads(step,
                        {"var_TMP": "on"},
                        {"lev_2_m_above_ground": "on"})
    if raw is None:
        raise RuntimeError("NOMADS fetch failed for TMP 2m")
    msgs = parse_grib2(raw)
    m = _find(msgs, "TMP", level_type="height_agl", level=2)
    if m is None:
        # try any TMP message
        m = next((x for x in msgs if x["param"] == "TMP"), None)
    if m is None:
        raise RuntimeError("TMP 2m not found in GRIB2")
    lat, lon = _std_grid()
    arr = _reshape(m["values_1d"]) - 273.15   # K → °C
    lat_min, lat_max, lon_min, lon_max = region
    lat, lon, arr = _crop(lat, lon, arr, lat_min, lat_max, lon_min, lon_max)
    return lat, lon, arr


def fetch_wind_level(step: int, level_hpa: int, region: tuple) -> tuple:
    """Wind speed + U + V at an isobaric level."""
    lev_key = f"lev_{level_hpa}_mb"
    raw = _fetch_nomads(step,
                        {"var_UGRD": "on", "var_VGRD": "on"},
                        {lev_key: "on"})
    if raw is None:
        raise RuntimeError(f"NOMADS fetch failed for wind {level_hpa} mb")
    msgs = parse_grib2(raw)
    mu = _find(msgs, "UGRD", level_type="isobaric", level=level_hpa)
    mv = _find(msgs, "VGRD", level_type="isobaric", level=level_hpa)
    if mu is None or mv is None:
        mu = next((x for x in msgs if x["param"] == "UGRD"), None)
        mv = next((x for x in msgs if x["param"] == "VGRD"), None)
    if mu is None or mv is None:
        raise RuntimeError(f"UGRD/VGRD not found for {level_hpa} mb")
    lat, lon = _std_grid()
    u = _reshape(mu["values_1d"])
    v = _reshape(mv["values_1d"])
    spd = np.sqrt(u**2 + v**2)
    lat_min, lat_max, lon_min, lon_max = region
    lat, lon, spd = _crop(lat, lon, spd, lat_min, lat_max, lon_min, lon_max)
    _, _, u   = _crop(*_std_grid(), u, lat_min, lat_max, lon_min, lon_max)
    _, _, v   = _crop(*_std_grid(), v, lat_min, lat_max, lon_min, lon_max)
    return lat, lon, spd, u, v


def fetch_mslp(step: int, region: tuple) -> tuple:
    """Mean Sea Level Pressure in hPa."""
    raw = _fetch_nomads(step,
                        {"var_PRMSL": "on"},
                        {"lev_mean_sea_level": "on"})
    if raw is None:
        raise RuntimeError("NOMADS fetch failed for MSLP")
    msgs = parse_grib2(raw)
    m = _find(msgs, "PRMSL")
    if m is None:
        m = next((x for x in msgs if x["param"] == "PRMSL"), None)
    if m is None:
        raise RuntimeError("PRMSL not found in GRIB2")
    lat, lon = _std_grid()
    arr = _reshape(m["values_1d"]) / 100.0   # Pa → hPa
    lat_min, lat_max, lon_min, lon_max = region
    lat, lon, arr = _crop(lat, lon, arr, lat_min, lat_max, lon_min, lon_max)
    return lat, lon, arr


def fetch_uwind(step: int, level_hpa: int, region: tuple) -> tuple:
    """U-component (zonal wind) at isobaric level, m/s."""
    lev_key = f"lev_{level_hpa}_mb"
    raw = _fetch_nomads(step, {"var_UGRD": "on"}, {lev_key: "on"})
    if raw is None:
        raise RuntimeError(f"NOMADS fetch failed for UGRD {level_hpa} mb")
    msgs = parse_grib2(raw)
    m = _find(msgs, "UGRD", level_type="isobaric", level=level_hpa)
    if m is None:
        m = next((x for x in msgs if x["param"] == "UGRD"), None)
    if m is None:
        raise RuntimeError(f"UGRD not found for {level_hpa} mb")
    lat, lon = _std_grid()
    arr = _reshape(m["values_1d"])
    lat_min, lat_max, lon_min, lon_max = region
    lat, lon, arr = _crop(lat, lon, arr, lat_min, lat_max, lon_min, lon_max)
    return lat, lon, arr


def fetch_vwind(step: int, level_hpa: int, region: tuple) -> tuple:
    """V-component (meridional wind) at isobaric level, m/s."""
    lev_key = f"lev_{level_hpa}_mb"
    raw = _fetch_nomads(step, {"var_VGRD": "on"}, {lev_key: "on"})
    if raw is None:
        raise RuntimeError(f"NOMADS fetch failed for VGRD {level_hpa} mb")
    msgs = parse_grib2(raw)
    m = _find(msgs, "VGRD", level_type="isobaric", level=level_hpa)
    if m is None:
        m = next((x for x in msgs if x["param"] == "VGRD"), None)
    if m is None:
        raise RuntimeError(f"VGRD not found for {level_hpa} mb")
    lat, lon = _std_grid()
    arr = _reshape(m["values_1d"])
    lat_min, lat_max, lon_min, lon_max = region
    lat, lon, arr = _crop(lat, lon, arr, lat_min, lat_max, lon_min, lon_max)
    return lat, lon, arr


# ═══════════════════════════════════════════════════════════════════════════
#  Colormaps
# ═══════════════════════════════════════════════════════════════════════════

def _temp_cmap():
    return plt.get_cmap("RdBu_r")

def _wind_cmap():
    colors = ["#f7fbff","#c6dbef","#9ecae1","#6baed6","#3182bd",
              "#08519c","#00441b","#41ab5d","#addd8e","#f7fcb9",
              "#fec44f","#fe9929","#ec7014","#cc4c02","#8c2d04"]
    return LinearSegmentedColormap.from_list("wind_spd", colors, N=256)

def _mslp_cmap():
    return plt.get_cmap("RdYlBu_r")

def _uv_cmap():
    return plt.get_cmap("RdBu_r")


# ═══════════════════════════════════════════════════════════════════════════
#  Render helpers
# ═══════════════════════════════════════════════════════════════════════════

def _xlabel(v):
    v = float(v)
    if v in (0.0, 360.0): return "0°"
    if v == 180.0: return "180°"
    return f"{v:.0f}°E" if v < 180 else f"{360 - v:.0f}°W"

def _ylabel(v):
    v = float(v)
    return "EQ" if v == 0 else f"{abs(v):.0f}°{'N' if v > 0 else 'S'}"

def _xticks(lon_min, lon_max, step=10):
    ticks = []
    v = int(np.floor(lon_min / step)) * step
    while v <= lon_max + step:
        if lon_min <= v <= lon_max:
            ticks.append(v % 360)
        v += step
    return sorted(set(ticks))

def _yticks(lat_min, lat_max, step=5):
    ticks = []
    v = int(np.floor(lat_min / step)) * step
    while v <= lat_max:
        if lat_min <= v <= lat_max:
            ticks.append(v)
        v += step
    return ticks


# ═══════════════════════════════════════════════════════════════════════════
#  Master render function
# ═══════════════════════════════════════════════════════════════════════════

def _render_map(lat, lon, field, title, cb_label, cmap, clevs,
                run_dt, step, coast_segs,
                u=None, v=None, show_wind=False,
                smooth_sigma=0.8, contour=False):
    """Draw a GFS forecast map and return PNG BytesIO."""
    field = gaussian_filter(field.astype(float), sigma=smooth_sigma)
    if u is not None:
        u = gaussian_filter(u.astype(float), sigma=smooth_sigma)
        v = gaussian_filter(v.astype(float), sigma=smooth_sigma)

    LON2D, LAT2D = np.meshgrid(lon, lat)
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    valid_dt = run_dt + datetime.timedelta(hours=step)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    ax = fig.add_axes([0.045, 0.145, 0.910, 0.750])
    ax.set_facecolor("#f4f0e8")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # ── shading ────────────────────────────────────────────────────────────
    cf = ax.contourf(LON2D, LAT2D, field, levels=clevs,
                     cmap=cmap, extend="both", zorder=1, alpha=0.88)

    # ── optional contour lines ──────────────────────────────────────────────
    if contour:
        n_c = min(12, len(clevs) // 2)
        cl = ax.contour(LON2D, LAT2D, field,
                        levels=clevs[::max(1, len(clevs) // n_c)],
                        colors="#111111", linewidths=0.4, alpha=0.45, zorder=2)
        ax.clabel(cl, fmt="%g", fontsize=7, inline=True, inline_spacing=3)

    # ── wind quivers (for wind-speed maps) ─────────────────────────────────
    if show_wind and u is not None and v is not None:
        step_q = max(1, int(round(len(lat) / 20)))
        qs = slice(None, None, step_q)
        Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
        Uq, Vq = u[qs, qs], v[qs, qs]
        spd_q = np.sqrt(Uq**2 + Vq**2)
        mask = ~np.isnan(spd_q) & (spd_q > 1.0)
        ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask],
                  color="#111111", scale=600, scale_units="inches",
                  width=0.0016, headwidth=4, headlength=5,
                  headaxislength=4, minshaft=1.2, pivot="middle",
                  zorder=6, alpha=0.85)

    # ── coastlines ─────────────────────────────────────────────────────────
    for seg in coast_segs:
        lons_s = np.where(seg[:, 0] < 0, seg[:, 0] + 360, seg[:, 0])
        lats_s = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons_s)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons_s, lats_s]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

    # ── grid lines ─────────────────────────────────────────────────────────
    lon_step = 10 if (lon_max - lon_min) < 60 else 20
    lat_step = 5  if (lat_max - lat_min) < 40 else 10
    for x in _xticks(lon_min, lon_max, lon_step):
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in _yticks(lat_min, lat_max, lat_step):
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.65, zorder=0, alpha=0.75)

    ax.set_xticks(_xticks(lon_min, lon_max, lon_step))
    ax.set_xticklabels([_xlabel(x) for x in _xticks(lon_min, lon_max, lon_step)],
                       fontsize=9, color="#333322")
    ax.set_yticks(_yticks(lat_min, lat_max, lat_step))
    ax.set_yticklabels([_ylabel(y) for y in _yticks(lat_min, lat_max, lat_step)],
                       fontsize=9, color="#333322")
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988"); spine.set_linewidth(0.8)

    # ── colorbar ───────────────────────────────────────────────────────────
    cax = fig.add_axes([0.12, 0.057, 0.760, 0.026])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8, colors="#222211", length=3, width=0.7)
    cbar.outline.set_edgecolor("#999988"); cbar.outline.set_linewidth(0.7)
    cax.text(0.5, -1.6, cb_label, transform=cax.transAxes, ha="center",
             va="top", fontsize=11.5, color="#222211", fontstyle="italic")

    # ── title ──────────────────────────────────────────────────────────────
    run_s  = run_dt.strftime("%Y-%m-%d %HZ")
    valid_s = valid_dt.strftime("%Y-%m-%d %HZ")
    ttext = f"{title}  ·  Run {run_s}  →  Valid {valid_s}  (+{step}h)"
    fig.text(0.50, 0.965, ttext, ha="center", va="top", fontsize=13,
             fontweight="bold", color="#111100", fontfamily="DejaVu Sans")

    # ── branding ───────────────────────────────────────────────────────────
    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#ccccbb", alpha=0.92, lw=0.9), zorder=10)
    ax.text(0.005, 0.016, "NCEP GFS  ·  NOMADS/NCEP",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            color="#666655", zorder=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════
#  Custom compute + render (Tier-2 kind)
# ═══════════════════════════════════════════════════════════════════════════

def _gfs_compute(pkg: dict, dates) -> tuple:
    """Tier-2 compute: fetch from NOMADS and return (lat, lon, data_dict)."""
    step      = pkg["gfs_step"]
    family    = pkg["gfs_family"]
    level_hpa = pkg.get("gfs_level", 0)
    region    = pkg.get("gfs_region", (0.0, 360.0, -80.0, 80.0))

    run_dt = _latest_run()

    if family == "temp":
        lat, lon, arr = fetch_temp2m(step, region)
        return lat, lon, {"main": arr, "run_dt": run_dt, "step": step}

    elif family.startswith("wind"):
        lat, lon, spd, u, v = fetch_wind_level(step, level_hpa, region)
        return lat, lon, {"main": spd, "u": u, "v": v, "run_dt": run_dt, "step": step}

    elif family == "mslp":
        lat, lon, arr = fetch_mslp(step, region)
        return lat, lon, {"main": arr, "run_dt": run_dt, "step": step}

    elif family == "uwnd":
        lat, lon, arr = fetch_uwind(step, level_hpa, region)
        return lat, lon, {"main": arr, "run_dt": run_dt, "step": step}

    elif family == "vwnd":
        lat, lon, arr = fetch_vwind(step, level_hpa, region)
        return lat, lon, {"main": arr, "run_dt": run_dt, "step": step}

    else:
        raise ValueError(f"Unknown GFS family: {family!r}")


def _gfs_render(lat, lon, data: dict, pkg: dict, coast_segs, dates) -> io.BytesIO:
    """Tier-2 render: draw map from data dict returned by _gfs_compute."""
    family    = pkg["gfs_family"]
    level_hpa = pkg.get("gfs_level", 0)
    step      = pkg["gfs_step"]
    run_dt    = data.get("run_dt", _latest_run())
    field     = data["main"]

    if family == "temp":
        clevs    = np.linspace(max(float(np.nanmin(field)), -30),
                               min(float(np.nanmax(field)), 50), 50)
        cb_label = "2 m Temperature  (°C)"
        title    = "GFS · 2 m Temperature"
        cmap     = _temp_cmap()
        contour  = True
        u, v, show_wind = None, None, False

    elif family.startswith("wind"):
        vmax     = max(float(np.nanmax(field)), 5.0)
        clevs    = np.linspace(0, vmax, 40)
        cb_label = f"{level_hpa} mb Wind Speed  (m/s)"
        title    = f"GFS · {level_hpa} mb Wind Speed"
        cmap     = _wind_cmap()
        contour  = False
        u        = data.get("u")
        v        = data.get("v")
        show_wind = True

    elif family == "mslp":
        vmin     = float(np.nanmin(field)) - 0.5
        vmax     = float(np.nanmax(field)) + 0.5
        clevs    = np.linspace(vmin, vmax, 50)
        cb_label = "Mean Sea Level Pressure  (hPa)"
        title    = "GFS · Sea Level Pressure"
        cmap     = _mslp_cmap()
        contour  = True
        u, v, show_wind = None, None, False

    elif family == "uwnd":
        lim      = max(abs(float(np.nanmin(field))), abs(float(np.nanmax(field))), 5.0)
        clevs    = np.linspace(-lim, lim, 50)
        cb_label = f"{level_hpa} mb U-Wind  (m/s)"
        title    = f"GFS · {level_hpa} mb U-Component (Zonal Wind)"
        cmap     = _uv_cmap()
        contour  = False
        u, v, show_wind = None, None, False

    elif family == "vwnd":
        lim      = max(abs(float(np.nanmin(field))), abs(float(np.nanmax(field))), 5.0)
        clevs    = np.linspace(-lim, lim, 50)
        cb_label = f"{level_hpa} mb V-Wind  (m/s)"
        title    = f"GFS · {level_hpa} mb V-Component (Meridional Wind)"
        cmap     = _uv_cmap()
        contour  = False
        u, v, show_wind = None, None, False

    else:
        raise ValueError(f"Unknown GFS family: {family!r}")

    return _render_map(
        lat, lon, field, title, cb_label, cmap, clevs,
        run_dt, step, coast_segs,
        u=u, v=v, show_wind=show_wind,
        contour=contour,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Product registry builder
# ═══════════════════════════════════════════════════════════════════════════

def _pid(family: str, step: int) -> str:
    return f"gfs_{family}_{step:02d}h"


def _make_product(pid: str, family: str, step: int,
                  name: str, desc: str,
                  level_hpa: int = 0) -> dict:
    lv_str = f" {level_hpa} mb" if level_hpa else ""
    return {
        "id":         pid,
        "title":      f"GFS · {name}  (+{step}h)",
        "name":       name,
        "tag":        "GFS FC",
        "desc":       desc,
        "kind":       "gfs_fc",
        # ── Tier-2 required keys ────────────────────
        "level":      level_hpa,
        "variables":  [],
        "show_wind":  False,
        "vlim":       1.0,
        "cint":       0.5,
        "cb_label":   name,
        "plot_scale": 1.0,
        # ── GFS-specific ────────────────────────────
        "gfs_family": family,
        "gfs_step":   step,
        "gfs_level":  level_hpa,
        "gfs_region": (0.0, 360.0, -80.0, 80.0),  # default global; overridden per-request
    }


# Build the full product table
PRODUCTS: dict = {}

for _step in GFS_STEPS:
    # Temperature 2m
    _pid_t = _pid("temp", _step)
    PRODUCTS[_pid_t] = _make_product(
        _pid_t, "temp", _step,
        "Temperature 2m", "GFS 2 m air temperature forecast.")

    # Wind at each isobaric level
    for _lv in WIND_LEVELS:
        _fam = f"wind{_lv}"
        _pid_w = _pid(_fam, _step)
        PRODUCTS[_pid_w] = _make_product(
            _pid_w, _fam, _step,
            f"Wind {_lv} mb", f"GFS {_lv} mb wind speed + vectors.",
            level_hpa=_lv)

    # Sea Level Pressure
    _pid_m = _pid("mslp", _step)
    PRODUCTS[_pid_m] = _make_product(
        _pid_m, "mslp", _step,
        "Sea Level Pressure", "GFS mean sea level pressure.")

    # U-Wind (850 mb default shown; all levels available via JS)
    for _lv in WIND_LEVELS:
        _pid_u = _pid(f"uwnd{_lv}", _step)
        PRODUCTS[_pid_u] = _make_product(
            _pid_u, "uwnd", _step,
            f"U-Wind {_lv} mb", f"GFS {_lv} mb zonal (U) wind component.",
            level_hpa=_lv)

    # V-Wind
    for _lv in WIND_LEVELS:
        _pid_v = _pid(f"vwnd{_lv}", _step)
        PRODUCTS[_pid_v] = _make_product(
            _pid_v, "vwnd", _step,
            f"V-Wind {_lv} mb", f"GFS {_lv} mb meridional (V) wind component.",
            level_hpa=_lv)


# ═══════════════════════════════════════════════════════════════════════════
#  Custom kind registration
# ═══════════════════════════════════════════════════════════════════════════

KINDS = {
    "gfs_fc": {
        "compute": _gfs_compute,
        "render":  _gfs_render,
        "tag":     "GFS FC",
        "title":   "GFS Forecast",
    }
}
