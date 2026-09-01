"""custom/gwo.py — Global Wind Oscillation (GWO) Phase diagnostic.

Uses the supplied GWO/AAM calculation and renders the standard (dM/dt, M)
GWO phase-space diagram. It is a drop-in custom addon: no core files need
to be changed. The product is registered under the existing Torque group.
"""

import io
from pro import config

# Keep the supplied GWO implementation, but make its cache path project-local
# so it works correctly when deployed with the XP Weather app.
import datetime as dt
import time
import hashlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydap.client import open_url
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------
NUM_DAYS    = 60   # most-recent days to plot
RM_DAYS     = 3    # running-mean window (days); set to 1 to disable smoothing
LAT_STRIDE  = 2
LON_STRIDE  = 2
OUTPUT_FILE = "gwo.png"

BASE_URL = f"{config.PSL}/uwnd.{{year}}.nc"
LTM_URL  = f"{config.PSL}/uwnd.day.ltm.1991-2020.nc"

# Disk cache for LTM climatology (recomputed once per LTM dataset version)
LTM_CACHE_FILE = os.path.join(config.BASE_DIR, "ltm_aam_cache.npy")

LTM_CHUNKS      = 10
LTM_RETRIES     = 3
LTM_RETRY_WAIT  = 5
LTM_MAX_WORKERS = 4   # parallel chunk fetches

# NOAA 1968-1996 baseline sigma values (Weickmann & Berry 2009)
AAM_STD  = 1.16e25
TEND_STD = 1.5e19

EARTH_RADIUS = 6.371e6
GRAVITY      = 9.80665


# ----------------------------------------------------------------------
# 1. MATH  — vectorised batch trapz
# ----------------------------------------------------------------------
def _trapz_batch(y, x):
    """
    Trapezoidal integration along axis 0 for a batch array y [..., N, ...].
    y shape: (N, ...), x shape: (N,).
    Returns shape: (...) — same as y with axis-0 reduced.
    """
    dx  = np.diff(x)                                    # (N-1,)
    avg = (y[1:] + y[:-1]) / 2.0                       # (N-1, ...)
    # broadcast dx over remaining dims
    return np.tensordot(dx, avg, axes=([0], [0]))       # scalar or (...)


def _trapz_axis1(y, x):
    """Integrate along axis 1:  y shape (T, N, ...) → (T, ...)."""
    dx  = np.diff(x)                                    # (N-1,)
    avg = (y[:, 1:] + y[:, :-1]) / 2.0                 # (T, N-1, ...)
    return np.tensordot(avg, dx, axes=([1], [0]))       # (T, ...)


# ----------------------------------------------------------------------
# 2. PRE-COMPUTE GRID CONSTANTS  (call once, reuse everywhere)
# ----------------------------------------------------------------------
def build_grid_constants(lev_hpa, lat_deg, lon_deg):
    """Return a dict of sorted indices / precomputed factors."""
    # pressure: sort surface→top
    lev_order  = np.argsort(lev_hpa)[::-1]
    lev_pa     = lev_hpa[lev_order] * 100.0            # Pa, descending

    # longitude: sort 0→2π, close the circle
    lon_rad    = np.deg2rad(lon_deg) % (2 * np.pi)
    lon_order  = np.argsort(lon_rad)
    lon_sorted = lon_rad[lon_order]
    lon_ext    = np.concatenate([lon_sorted, [lon_sorted[0] + 2 * np.pi]])

    # latitude: sort south→north
    lat_rad    = np.deg2rad(lat_deg)
    lat_order  = np.argsort(lat_rad)
    lat_sorted = lat_rad[lat_order]
    cos2_lat   = np.cos(lat_sorted) ** 2               # (nlat,)

    scale = (EARTH_RADIUS ** 3 / GRAVITY)

    return dict(
        lev_order  = lev_order,
        lev_pa     = lev_pa,
        lon_order  = lon_order,
        lon_ext    = lon_ext,
        lat_order  = lat_order,
        lat_sorted = lat_sorted,
        cos2_lat   = cos2_lat,
        scale      = scale,
    )


# ----------------------------------------------------------------------
# 3. BATCH AAM  (all time steps at once)
# ----------------------------------------------------------------------
def compute_aam_batch(u_all, gc):
    """
    u_all : (T, nlev, nlat, nlon)
    gc    : grid constants dict from build_grid_constants()
    returns: (T,) array of relative AAM values
    """
    T = u_all.shape[0]

    # --- pressure integral (axis 1 = level) ---
    u_s = u_all[:, gc['lev_order'], :, :]              # sort levels
    # _trapz_axis1: integrate axis-1 (levels) → (T, nlat, nlon)
    I_p = -_trapz_axis1(u_s, gc['lev_pa'])

    # --- longitude integral (axis 2 = lon) ---
    I_p_sorted  = I_p[:, :, gc['lon_order']]           # (T, nlat, nlon_s)
    I_p_ext     = np.concatenate(
        [I_p_sorted, I_p_sorted[:, :, :1]], axis=2
    )                                                   # (T, nlat, nlon+1)
    # integrate over lon → (T, nlat)
    I_p_lon = _trapz_axis1(
        I_p_ext.transpose(0, 2, 1),                    # (T, nlon+1, nlat)
        gc['lon_ext']
    )                                                   # (T, nlat)

    # --- latitude integral ---
    integrand = I_p_lon[:, gc['lat_order']] * gc['cos2_lat']  # (T, nlat)
    aam = _trapz_axis1(integrand, gc['lat_sorted'])            # (T,)

    return gc['scale'] * aam


# ----------------------------------------------------------------------
# 4. SAFE ARRAY EXTRACTION
# ----------------------------------------------------------------------
def _to_numpy(obj):
    if hasattr(obj, 'data'):
        val = obj.data
        if hasattr(val, 'data'):
            val = val.data
        return np.array(val, dtype=np.float64)
    return np.array(obj[:], dtype=np.float64)


def _unpack(raw_obj, attrs):
    data    = _to_numpy(raw_obj)
    missing = attrs.get("missing_value", attrs.get("_FillValue", None))
    if missing is not None:
        mv = float(missing)
        data[np.abs(data - mv) < np.abs(mv) * 1e-4] = np.nan
    scale  = attrs.get("scale_factor", None)
    offset = attrs.get("add_offset",   None)
    if scale is not None or offset is not None:
        data = data * float(scale  if scale  is not None else 1.0) \
                    + float(offset if offset is not None else 0.0)
    return data


# ----------------------------------------------------------------------
# 5. LTM — parallel fetch + disk cache
# ----------------------------------------------------------------------
def _fetch_ltm_chunk(args):
    """Worker: fetch one time chunk from OPeNDAP and return AAM values."""
    url, t0, t1, ltm_lev_idx, gc, attrs, chunk_id, total = args
    for attempt in range(1, LTM_RETRIES + 1):
        try:
            ds_w     = open_url(url)
            uwnd_var = ds_w["uwnd"]
            raw      = uwnd_var.array[t0:t1, :, ::LAT_STRIDE, ::LON_STRIDE]
            chunk    = _unpack(raw, attrs)
            if chunk.ndim == 3:
                chunk = chunk[np.newaxis]
            chunk = chunk[:, ltm_lev_idx, :, :]     # (days, nlev, nlat, nlon)
            aam_chunk = compute_aam_batch(chunk, gc)
            print(f"  Chunk {chunk_id}/{total}: DOY {t0+1}-{t1} done "
                  f"({len(aam_chunk)} days)", flush=True)
            return t0, aam_chunk
        except Exception as e:
            if attempt < LTM_RETRIES:
                print(f"\n  Chunk {chunk_id} attempt {attempt} failed: {e}. "
                      f"Retrying in {LTM_RETRY_WAIT}s ...")
                time.sleep(LTM_RETRY_WAIT)
            else:
                raise RuntimeError(
                    f"LTM chunk [{t0}:{t1}] failed after {LTM_RETRIES} attempts: {e}"
                )


def compute_ltm_aam_clim(live_lev):
    # --- Try disk cache first ---
    if os.path.exists(LTM_CACHE_FILE):
        print(f"Loading LTM climatology from cache: {LTM_CACHE_FILE}")
        cached = np.load(LTM_CACHE_FILE, allow_pickle=True).item()
        # Validate that strides/levels match
        cache_key = (LAT_STRIDE, LON_STRIDE,
                     tuple(np.sort(live_lev).astype(int)))
        if cached.get('key') == cache_key:
            print(f"  Cache hit — skipping all LTM downloads.\n")
            return cached['aam_clim'], cached['lev_common']
        else:
            print("  Cache key mismatch (settings changed) — recomputing.")

    print("Opening LTM OPeNDAP dataset ...")
    ds_ltm   = open_url(LTM_URL)
    uwnd_var = ds_ltm["uwnd"]
    attrs    = uwnd_var.attributes

    lev_ltm  = _to_numpy(ds_ltm["level"])
    lat_full = _to_numpy(ds_ltm["lat"])
    lon_full = _to_numpy(ds_ltm["lon"])

    common = np.intersect1d(np.round(lev_ltm).astype(int),
                            np.round(live_lev).astype(int))
    if len(common) == 0:
        raise RuntimeError("No common pressure levels between LTM and live data.")
    ltm_lev_idx = np.array([
        np.where(np.round(lev_ltm).astype(int) == p)[0][0] for p in common
    ])
    lev_common = lev_ltm[ltm_lev_idx]

    lat = lat_full[::LAT_STRIDE]
    lon = lon_full[::LON_STRIDE]

    time_var = ds_ltm.get("time", None)
    ntime    = len(_to_numpy(time_var)) if time_var is not None else 365

    print(f"  Common levels: {np.sort(lev_common)[::-1].astype(int).tolist()}")
    print(f"  LTM grid: {ntime} days x {len(lev_common)} lev "
          f"x {len(lat)} lat x {len(lon)} lon")
    print(f"  Fetching in {LTM_CHUNKS} chunks with "
          f"{LTM_MAX_WORKERS} parallel workers ...")

    gc    = build_grid_constants(lev_common, lat, lon)
    edges = np.linspace(0, ntime, LTM_CHUNKS + 1, dtype=int)

    tasks = [
        (LTM_URL, int(edges[k]), int(edges[k+1]),
         ltm_lev_idx, gc, attrs, k+1, LTM_CHUNKS)
        for k in range(LTM_CHUNKS)
    ]

    # Parallel fetch
    results = {}
    with ThreadPoolExecutor(max_workers=LTM_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_ltm_chunk, t): t for t in tasks}
        for fut in as_completed(futures):
            t0_idx, aam_chunk = fut.result()
            results[t0_idx] = aam_chunk

    # Reassemble in correct DOY order
    aam_clim = np.concatenate([results[k] for k in sorted(results)])
    print(f"\n  LTM AAM range: {aam_clim.min():.4e} – {aam_clim.max():.4e}")
    print(f"  LTM AAM annual mean: {aam_clim.mean():.4e}\n")

    # Save to cache
    cache_key = (LAT_STRIDE, LON_STRIDE, tuple(np.sort(live_lev).astype(int)))
    np.save(LTM_CACHE_FILE, {'aam_clim': aam_clim,
                              'lev_common': lev_common,
                              'key': cache_key})
    print(f"  LTM cache saved → {LTM_CACHE_FILE}\n")
    return aam_clim, lev_common


# ----------------------------------------------------------------------
# 6. LIVE DATA
# ----------------------------------------------------------------------
def decode_days_since_1800(days):
    return dt.datetime(1800, 1, 1) + dt.timedelta(days=float(days))


def fetch_year(year, n_recent=None):
    url = BASE_URL.format(year=year)
    print(f"Connecting to {url} ...")
    ds = open_url(url)

    lev      = _to_numpy(ds["level"])
    lat      = _to_numpy(ds["lat"])[::LAT_STRIDE]
    lon      = _to_numpy(ds["lon"])[::LON_STRIDE]
    time_raw = _to_numpy(ds["time"])
    ntime    = len(time_raw)

    t0 = max(0, ntime - n_recent) if n_recent is not None else 0
    print(f"  Year {year}: {ntime} records; fetching indices {t0}..{ntime-1}")

    uwnd_var = ds["uwnd"]
    raw      = uwnd_var.array[t0:ntime, :, ::LAT_STRIDE, ::LON_STRIDE]
    data     = _unpack(raw, uwnd_var.attributes)

    finite = data[np.isfinite(data)]
    if finite.size:
        print(f"  uwnd range: {finite.min():.1f} to {finite.max():.1f} m/s")

    times = [decode_days_since_1800(d) for d in time_raw[t0:]]
    return times, lev, lat, lon, data


def fetch_recent_days(num_days, end_date=None):
    """Fetch up to num_days daily records ending on end_date."""
    if end_date is None:
        end_date = dt.datetime.now(dt.timezone.utc).date()
    elif isinstance(end_date, dt.datetime):
        end_date = end_date.date()

    requested_days = int(num_days)
    remaining = requested_days
    years = [end_date.year, end_date.year - 1]
    chunks = []
    lev = lat = lon = None

    for year in years:
        url = BASE_URL.format(year=year)
        print(f"Connecting to {url} ...")
        ds = open_url(url)
        lev_y = _to_numpy(ds["level"])
        lat_y = _to_numpy(ds["lat"])[::LAT_STRIDE]
        lon_y = _to_numpy(ds["lon"])[::LON_STRIDE]
        time_raw = _to_numpy(ds["time"])
        units = ds["time"].attributes.get("units", "days since 1800-01-01")
        scale = 1.0 / 24.0 if "hours" in units else 1.0
        import re
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
        epoch = (dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                 if m else dt.datetime(1800, 1, 1))
        times_y = [epoch + dt.timedelta(days=float(v) * scale) for v in time_raw]
        wanted = [i for i, t in enumerate(times_y) if t.date() <= end_date]
        if not wanted:
            continue
        # Keep only the tail needed from this year.
        idx = wanted[-remaining:]
        uwnd_var = ds["uwnd"]
        raw = uwnd_var.array[idx[0]:idx[-1] + 1, :, ::LAT_STRIDE, ::LON_STRIDE]
        data = _unpack(raw, uwnd_var.attributes)
        all_times = times_y[idx[0]:idx[-1] + 1]
        # The yearly dataset is daily/continuous; mask anything outside the
        # selected requested dates when an index slice spans missing records.
        if len(all_times) != len(idx):
            keep = {t for t in (times_y[i] for i in idx)}
            mask = np.array([t in keep for t in all_times])
            data = data[mask]
            all_times = [t for t, k in zip(all_times, mask) if k]
        chunks.insert(0, (all_times, lev_y, lat_y, lon_y, data))
        remaining -= len(all_times)
        if remaining <= 0:
            break

    if not chunks:
        raise RuntimeError(f"No GWO wind data available through {end_date}.")

    times = []
    data_parts = []
    for t, _, _, _, d in chunks:
        times.extend(t)
        data_parts.append(d)
    data = np.concatenate(data_parts, axis=0)
    lev, lat, lon = chunks[-1][1], chunks[-1][2], chunks[-1][3]

    # Ensure chronological order and exactly the requested window.
    order = np.argsort(np.array([t.timestamp() for t in times]))
    times = [times[i] for i in order][-requested_days:]
    data = data[order][-requested_days:]
    return times, lev, lat, lon, data




def _compute_gwo(pkg, dates):
    """Compute the GWO series; rendering is handled by _render_gwo."""
    # GWO is conventionally displayed over the latest 60 days.
    end_date = dates[-1] if dates else None
    times, lev, lat, lon, uwnd = fetch_recent_days(NUM_DAYS, end_date=end_date)

    aam_clim, _lev_common = compute_ltm_aam_clim(live_lev=lev)
    gc = build_grid_constants(lev, lat, lon)
    aam_raw = compute_aam_batch(uwnd, gc)

    doys = np.array([t.timetuple().tm_yday for t in times])
    clim_idx = np.clip(doys - 1, 0, len(aam_clim) - 1)
    aam_anom = aam_raw - aam_clim[clim_idx]
    M = aam_anom / AAM_STD
    dMdt = np.gradient(aam_raw, 86400.0) / TEND_STD

    return np.array([0.0]), np.array([0.0]), {
        "main": np.array([[M[-1]]], dtype=float),
        "M": M, "dMdt": dMdt, "times": times,
    }


def running_mean(x, n=5):
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same") / np.convolve(
        np.ones(len(x)), kernel, mode="same")


def _render_gwo(lat, lon, data, pkg, coast_segs, dates, **_kw):
    times = data["times"]
    M = data["M"]
    dMdt = data["dMdt"]
    M_sm = running_mean(M, RM_DAYS)
    dM_sm = running_mean(dMdt, RM_DAYS)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#f0f2f5")
    ax.set_facecolor("#f7f9fc")
    LIM = 4
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")

    points = np.array([dM_sm, M_sm]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(M_sm) - 1)
    lc = LineCollection(segments, cmap="cool", norm=norm, linewidth=1.5, alpha=0.9)
    lc.set_array(np.arange(len(M_sm)))
    ax.add_collection(lc)

    ax.scatter(dM_sm, M_sm, c=np.arange(len(M_sm)), cmap="cool", s=40,
               edgecolor="k", linewidth=0.5, zorder=5)
    ax.scatter(dM_sm[0], M_sm[0], c="limegreen", s=100, marker="*", edgecolor="k", zorder=10)
    ax.scatter(dM_sm[-1], M_sm[-1], c="crimson", s=100, marker="X", edgecolor="k", zorder=10)

    for i in range(0, len(times), 5):
        ax.annotate(times[i].strftime("%-d %b"), (dM_sm[i], M_sm[i]),
                    fontsize=8, ha="center", va="bottom", color="black",
                    xytext=(0, 8), textcoords="offset points", zorder=6)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.plot([-4, 4], [4, -4], "k--", linewidth=0.8, alpha=0.5)
    ax.plot([-4, 4], [-4, 4], "k--", linewidth=0.8, alpha=0.5)
    ax.add_artist(plt.Circle((0, 0), 1, color="black", fill=False, linestyle="-", linewidth=1.0, alpha=0.7))

    phases = [(22.5, "5"), (67.5, "6"), (112.5, "7"), (157.5, "8"),
              (202.5, "1"), (247.5, "2"), (292.5, "3"), (337.5, "4")]
    for ang_deg, label in phases:
        ang = np.deg2rad(ang_deg)
        x, y = LIM * 0.85 * np.cos(ang), LIM * 0.85 * np.sin(ang)
        ax.text(x, y, label, fontsize=12, ha="center", va="center", color="darkblue", weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"))

    ax.text(0, -3.8, "LOW AAM", fontsize=9, ha="center", color="teal", fontstyle="italic")
    ax.text(3.8, 0, "HIGH TENDENCY", fontsize=9, rotation=90, ha="center", color="teal", fontstyle="italic", va="center")
    ax.text(0, 3.8, "HIGH AAM", fontsize=9, ha="center", color="teal", fontstyle="italic")
    ax.text(-3.8, 0, "LOW TENDENCY", fontsize=9, rotation=90, ha="center", color="teal", fontstyle="italic", va="center")

    cc = ax.text(0.99, 0.01, "© XP WEATHER", fontsize=10, ha="right", va="bottom", color="white", transform=ax.transAxes)
    cc.set_bbox(dict(facecolor="black", alpha=0.3, edgecolor="none"))

    ax.set_xlabel("dM/dt (standardised tendency)", fontsize=12, color="navy")
    ax.set_ylabel("M (standardised AAM anomaly)", fontsize=12, color="navy")
    ax.set_title(f"(dM/dt, M) GWO PHASE: {times[0].strftime('%d %B %Y')} TO {times[-1].strftime('%d %B %Y')}",
                 fontsize=14, weight="bold", color="navy", pad=12,
                 bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.5", alpha=0.9))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


PRODUCTS = {
    "gwo_phase": {
        "id": "gwo_phase",
        "tag": "Torque",
        "title": "GWO Phase",
        "name": "GWO Phase",
        "desc": "Global Wind Oscillation phase-space diagram using standardised AAM anomaly and tendency.",
        "kind": "gwo_phase",
        "level": None,
        "cb_label": "GWO Phase",
    },
}

KINDS = {
    "gwo_phase": {
        "compute": _compute_gwo,
        "render": _render_gwo,
        "tag": "Torque",
        "title": "GWO Phase",
    },
}
