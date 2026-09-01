"""custom/sst.py — Sea Surface Temperature (OISST) products.

Product families (each available per-region via the sidebar region selector):
  SST Mean       — absolute SST field
  SST Anomaly    — SST anomaly (obs – 1991-2020 climatology)
  SST Boxes      — SST anomaly with all climate-index boxes drawn

Regions: Global · Pacific · Indian Ocean · Atlantic

All parameters live entirely in the product config dicts below.
"""

import io
import re
import time
import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pydap.client import open_url

from pro import config   # BASE_DIR, SHP_PATH, SHP_URL, coastline cache

# ── OISST & NCEP OPeNDAP roots ──────────────────────────────────────────────
OISST = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"
PSL   = config.PSL

# ── module-level caches ──────────────────────────────────────────────────────
_OISST_CACHE  = {}   # url  -> pydap dataset
_SFIELD_CACHE = {}   # key  -> cached field

# ── climate-index box definitions ────────────────────────────────────────────
_INDEX_BOXES = {
    "nino4":    {"name": "Niño 4",      "lon": (160, 210),  "lat": (-5,  5)},
    "nino3.4":  {"name": "Niño 3.4",    "lon": (190, 240),  "lat": (-5,  5)},
    "nino3":    {"name": "Niño 3",      "lon": (210, 270),  "lat": (-5,  5)},
    "nino1+2":  {"name": "Niño 1+2",    "lon": (270, 280),  "lat": (-10, 0)},
    "iod_west": {"name": "WTIO (IOD)",  "lon": (50,   70),  "lat": (-10, 10)},
    "iod_east": {"name": "SETIO (IOD)", "lon": (90,  110),  "lat": (-10, 0)},
}

# ── region bounds: (lon_min, lon_max, lat_min, lat_max) ─────────────────────
_REGION_BOUNDS = {
    "global": (0.0,   360.0, -90.0,  90.0),
    "io":     (30.0,  120.0, -40.0,  30.0),
    "atl":    (280.0,  20.0, -20.0,  60.0),   # wraps
    "pac":    (100.0, 290.0, -40.0,  40.0),
}

_REGION_DISPLAY = {
    "global": "Global",
    "io":     "Indian Ocean",
    "atl":    "Atlantic Ocean",
    "pac":    "Pacific Ocean",
}

# ── which index boxes to show per region ────────────────────────────────────
_REGION_BOXES = {
    "global": ["nino4", "nino3.4", "nino3", "nino1+2", "iod_west", "iod_east"],
    "pac":    ["nino4", "nino3.4", "nino3", "nino1+2"],
    "io":     ["iod_west", "iod_east"],
    "atl":    [],
}


# ===========================================================================
# Helpers
# ===========================================================================

def _with_retry(func, *args, max_retries=6, base_delay=6, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if attempt == max_retries or ("429" not in msg and "Too Many" not in msg):
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))


def _open_oisst(url):
    if url not in _OISST_CACHE:
        _OISST_CACHE[url] = _with_retry(open_url, url)
    return _OISST_CACHE[url]


def _latlon_oisst(ds):
    return (_with_retry(lambda: np.array(ds["lat"][:])),
            _with_retry(lambda: np.array(ds["lon"][:])))


def _time_idx_oisst(ds, target):
    raw   = _with_retry(lambda: np.array(ds["time"][:]))
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    scale = 1.0 / 24.0 if "hours" in units else 1.0
    m     = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
    epoch = (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
             if m else datetime.date(1800, 1, 1))
    dates_ds = []
    for t in raw:
        try:
            val = float(t)
            dates_ds.append(epoch + datetime.timedelta(days=val * scale)
                            if not (np.isnan(val) or abs(val) > 1e7) else None)
        except Exception:
            dates_ds.append(None)
    for i, d in enumerate(dates_ds):
        if d and (d.year, d.month, d.day) == (target.year, target.month, target.day):
            return i
    cands = [i for i, d in enumerate(dates_ds) if d and d <= target]
    if cands:
        return cands[-1]
    raise ValueError(f"Date {target} not found in OISST dataset")


def _get_lat_lon_indices(lat_all, lon_all, region):
    bounds = _REGION_BOUNDS.get(region.lower(), _REGION_BOUNDS["global"])
    lon_min, lon_max, lat_min, lat_max = bounds
    lat_idx = np.where((lat_all >= lat_min) & (lat_all <= lat_max))[0]
    if region.lower() == "atl":
        idx1 = np.where(lon_all >= 280.0)[0]
        idx2 = np.where(lon_all <=  20.0)[0]
        lon_idx = (idx1, idx2)
    else:
        lon_idx = np.where((lon_all >= lon_min) & (lon_all <= lon_max))[0]
    return lat_idx, lon_idx


def _read_regional_sst(ds, t, lat_idx, lon_idx, stride=1):
    lat_sl = slice(lat_idx[0], lat_idx[-1] + 1, stride)
    if isinstance(lon_idx, tuple):
        idx1, idx2 = lon_idx
        r1 = np.array(ds["sst"][t, lat_sl, slice(idx1[0], idx1[-1]+1, stride)].data).squeeze()
        r2 = np.array(ds["sst"][t, lat_sl, slice(idx2[0], idx2[-1]+1, stride)].data).squeeze()
        raw = np.hstack([r1, r2])
    else:
        raw = np.array(ds["sst"][t, lat_sl, slice(lon_idx[0], lon_idx[-1]+1, stride)].data).squeeze()

    raw  = raw.astype(np.float64)
    attr = ds["sst"].attributes
    sf   = float(attr.get("scale_factor", 1.0))
    ao   = float(attr.get("add_offset",   0.0))
    data = raw * sf + ao
    mv   = attr.get("missing_value", attr.get("_FillValue", None))
    if mv is not None:
        try:
            mv = float(mv)
            data[np.abs(raw - mv) < max(abs(mv) * 1e-4, 0.5)] = np.nan
        except Exception:
            pass
    data[np.abs(data) > 500] = np.nan
    return data


def _sst_obs_mean(dates, region, stride=1):
    by_year = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)
    lat_full = lon_full = lat_idx = lon_idx = None
    slices = []
    for year, ydates in sorted(by_year.items()):
        ds = _open_oisst(f"{OISST}/sst.day.mean.{year}.nc")
        if lat_full is None:
            lat_full, lon_full = _latlon_oisst(ds)
            lat_idx, lon_idx   = _get_lat_lon_indices(lat_full, lon_full, region)
        for d in ydates:
            ti = _time_idx_oisst(ds, d)
            slices.append(_read_regional_sst(ds, ti, lat_idx, lon_idx, stride))

    lat_ret = lat_full[lat_idx][::stride]
    if isinstance(lon_idx, tuple):
        lon_ret = np.concatenate([lon_full[lon_idx[0]], lon_full[lon_idx[1]] + 360.0])[::stride]
    else:
        lon_ret = lon_full[lon_idx][::stride]
    return lat_ret, lon_ret, np.nanmean(slices, axis=0)


def _sst_clim_mean(dates, region, stride=1):
    key = ("sst_clim", region, tuple(d.isoformat() for d in dates), stride)
    if key in _SFIELD_CACHE:
        return _SFIELD_CACHE[key]
    ds = _open_oisst(f"{OISST}/sst.day.mean.ltm.1991-2020.nc")
    lat_full, lon_full = _latlon_oisst(ds)
    lat_idx, lon_idx   = _get_lat_lon_indices(lat_full, lon_full, region)
    n = len(_with_retry(lambda: np.array(ds["time"][:])))
    slices = [_read_regional_sst(ds, min(d.timetuple().tm_yday - 1, n - 1), lat_idx, lon_idx, stride)
              for d in dates]
    result = np.nanmean(slices, axis=0)
    _SFIELD_CACHE[key] = result
    return result


# ===========================================================================
# Compute
# ===========================================================================

def _compute_sst(pkg, dates):
    region   = pkg.get("region",   "global")
    sst_mode = pkg.get("sst_mode", "anomaly")   # "anomaly" | "mean"
    stride   = pkg.get("stride",   1)

    lat, lon, obs = _sst_obs_mean(dates, region, stride)

    if sst_mode == "anomaly":
        clim  = _sst_clim_mean(dates, region, stride)
        sst_f = gaussian_filter(obs - clim, sigma=1.5)
    else:
        sst_f = obs

    return lat, lon, {"main": sst_f}


# ===========================================================================
# Render
# ===========================================================================

def _render_sst(lat, lon, data, pkg, coast_segs, dates, out_buf=None, **_kw):
    region     = pkg.get("region",     "global")
    sst_mode   = pkg.get("sst_mode",   "anomaly")
    show_boxes = pkg.get("show_boxes", False)
    show_box   = pkg.get("show_box",   False)
    box_reg    = pkg.get("box_region", "nino3.4")

    sst_field = data["main"]
    LON2D, LAT2D = np.meshgrid(lon, lat)

    bounds = _REGION_BOUNDS.get(region.lower(), _REGION_BOUNDS["global"])
    lon_min, lon_max, lat_min, lat_max = bounds
    dlon = 100.0 if region.lower() == "atl" else lon_max - lon_min
    dlat = lat_max - lat_min
    aspect = dlon / dlat
    base_w = 12.0
    fig_h  = np.clip(base_w / aspect, 5.0, 9.0)
    fig_w  = fig_h * aspect if fig_h in (5.0, 9.0) else base_w

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=200)
    ax  = fig.add_axes([0.07, 0.08, 0.82, 0.82])
    ax.set_facecolor("#e8f1f5")
    ax.set_aspect("equal", adjustable="box")

    if region.lower() == "atl":
        ax.set_xlim(280, 380)
    else:
        ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    if sst_mode == "anomaly":
        vlim        = pkg.get("vlim", 5.0)
        cbar_ticks  = np.arange(-vlim, vlim + 0.01, pkg.get("cint", 1.0))
        cbar_label  = pkg.get("cb_label", "SST Anomaly [°C]")
        shade_title = "SST Anomaly"
        levels      = np.linspace(-vlim, vlim, 101)
        cmap        = "RdBu_r"
    else:
        vmin, vmax  = 14.0, 34.0
        vlim        = vmax
        cbar_ticks  = np.arange(vmin, vmax + 0.01, 2.0)
        cbar_label  = pkg.get("cb_label", "SST [°C]")
        shade_title = "SST"
        levels      = np.linspace(vmin, vmax, 41)
        cmap        = "turbo"

    cf = ax.contourf(LON2D, LAT2D, sst_field, levels=levels,
                     cmap=cmap, extend="both", zorder=1)

    # ── boxes ────────────────────────────────────────────────────────────────
    drawn_boxes = []
    if show_boxes:
        drawn_boxes = _REGION_BOXES.get(region.lower(), [])
    elif show_box and box_reg.lower() in _INDEX_BOXES:
        drawn_boxes = [box_reg.lower()]

    region_val_tag = None
    for bkey in drawn_boxes:
        b = _INDEX_BOXES[bkey]
        x0, x1 = b["lon"]
        y0, y1 = b["lat"]
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        fill=False, edgecolor="#000000",
                                        linewidth=2.0, zorder=8))
        ax.text(x0 + (x1 - x0) / 2.0, y1 + 1.2, b["name"],
                color="#000000", fontsize=9, fontweight="bold",
                ha="center", va="bottom", zorder=9)
        mask = ((LON2D >= x0) & (LON2D <= x1) &
                (LAT2D >= y0) & (LAT2D <= y1))
        val  = np.nanmean(sst_field[mask])
        region_val_tag = (region_val_tag or "") + f"{b['name']}: {val:+.2f}°C  "

    if not region_val_tag:
        rd = _REGION_DISPLAY.get(region.lower(), region.upper())
        region_val_tag = f"{rd}: {np.nanmean(sst_field):.2f}°C"

    ax.text(0.985, 0.975, region_val_tag.strip(), transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#111111",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#a0a0a0", alpha=0.90, lw=0.8), zorder=10)

    # ── coastlines ──────────────────────────────────────────────────────────
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, seg[:, 1]]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#222222", lw=0.85, zorder=7)

    # ── grid ─────────────────────────────────────────────────────────────────
    dx = 20 if region.lower() != "global" else 60
    dy = 10 if region.lower() != "global" else 20
    for x in range(0, 381, dx):
        ax.axvline(x, color="#a6b1c2", lw=0.45, ls="--", zorder=0)
    for y in range(-80, 81, dy):
        ax.axhline(y, color="#a6b1c2", lw=0.45, ls="--", zorder=0)
    ax.axhline(0, color="#2c3e50", lw=0.95, zorder=0)

    # ── ticks ────────────────────────────────────────────────────────────────
    xticks = (list(range(280, 381, dx)) if region.lower() == "atl"
              else list(range(int(lon_min), int(lon_max) + 1, dx)))
    yticks = list(range(int(lat_min), int(lat_max) + 1, dy))

    def _xlab(v):
        return ("0°" if v in (0, 360) else "180°" if v == 180
                else f"{v}°E" if v < 180 else f"{360 - v if v <= 360 else 720 - v}°W")
    def _ylab(v):
        return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

    ax.set_xticks(xticks)
    ax.set_xticklabels([_xlab(x) for x in xticks],
                       fontsize=11, color="#111111", fontweight="medium")
    ax.set_yticks(yticks)
    ax.set_yticklabels([_ylab(y) for y in yticks],
                       fontsize=11, color="#111111", fontweight="medium")

    # ── colourbar ────────────────────────────────────────────────────────────
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.018, pos.height])
    cbar = plt.colorbar(cf, cax=cax, ticks=cbar_ticks)
    cbar.set_label(cbar_label, fontsize=12, fontweight="bold",
                   color="#111111", labelpad=6)

    # ── title & branding ──────────────────────────────────────────────────────
    rd2 = _REGION_DISPLAY.get(region.lower(), region.upper())
    date_str = (f"{dates[0]:%-d %b} – {dates[-1]:%-d %b %Y}"
                if len(dates) > 1 else f"{dates[0]:%-d %b %Y}")
    ax.set_title(
        f"{shade_title}  [{rd2}]\n"
        f"{date_str}  ({len(dates)}-day mean)",
        fontsize=14, fontweight="bold", color="#111111", pad=12, loc="center")

    ax.text(0.015, 0.020, "OISST & NCEP/NCAR", transform=ax.transAxes,
            fontsize=8, color="#222222", alpha=0.55, ha="left", va="bottom", zorder=10)
    ax.text(0.985, 0.020, "@XPWEATHER", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="right", va="bottom", color="#111111",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a0a0a0",
                      alpha=0.88, lw=0.7), zorder=10)

    if out_buf is None:
        out_buf = io.BytesIO()
    plt.savefig(out_buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    out_buf.seek(0)
    return out_buf


# ===========================================================================
# Product registry
# ===========================================================================

def _sst_product(pid, name, desc, region, sst_mode,
                 show_box=False, show_boxes=False, box_region="nino3.4",
                 vlim=5.0, cint=1.0):
    """Helper to build a product dict without repetition."""
    if sst_mode == "mean":
        cb = "SST [°C]"
        vlim_v = 34.0
        cint_v = 2.0
    else:
        cb = "SST Anomaly [°C]"
        vlim_v = vlim
        cint_v = cint
    return {
        "id": pid, "title": name, "name": name, "tag": "SST",
        "desc": desc, "kind": "sst", "level": None,
        "region": region, "sst_mode": sst_mode,
        "show_box": show_box, "show_boxes": show_boxes, "box_region": box_region,
        "vlim": vlim_v, "cint": cint_v, "cb_label": cb,
    }


_REGIONS = [
    ("global", "Global"),
    ("pac",    "Pacific"),
    ("io",     "Indian Ocean"),
    ("atl",    "Atlantic"),
]

PRODUCTS = {}

for _rid, _rname in _REGIONS:
    _box = "nino3.4" if _rid in ("global","pac") else ("iod_west" if _rid=="io" else "nino3.4")
    _sb  = _rid in ("global","pac","io")    # show single box for these regions

    # 1. SST Mean
    PRODUCTS[f"sst_mean_{_rid}"] = _sst_product(
        f"sst_mean_{_rid}", f"SST Mean · {_rname}",
        f"{_rname} absolute SST (OISST).",
        _rid, "mean")

    # 2. SST Anomaly
    PRODUCTS[f"sst_anom_{_rid}"] = _sst_product(
        f"sst_anom_{_rid}", f"SST Anomaly · {_rname}",
        f"{_rname} SST anomaly (OISST – 1991-2020 climatology).",
        _rid, "anomaly")

    # 3. SST Boxes  (SST anomaly, all index boxes)
    PRODUCTS[f"sst_boxes_{_rid}"] = _sst_product(
        f"sst_boxes_{_rid}", f"SST Boxes · {_rname}",
        f"{_rname} SST anomaly with all climate-index region boxes.",
        _rid, "anomaly", show_boxes=True)


# ── Custom kind registration ──────────────────────────────────────────────────
KINDS = {
    "sst": {
        "compute": _compute_sst,
        "render":  _render_sst,
        "tag":     "SST",
        "title":   "Sea Surface Temperature",
    },
}
