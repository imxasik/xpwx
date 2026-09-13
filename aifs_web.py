"""
aifs_web.py — Web adapter for the ECMWF AIFS Forecast anomaly products.

Products:
  - Velocity Potential (vp)        — 200 hPa velocity potential + divergent wind anomaly
  - Geopotential Height (z)        — 500/200/etc hPa geopotential height anomaly
  - Zonal Wind (u)                 — 200/850/etc hPa U-wind anomaly
  - Meridional Wind (v)            — 200/850/etc hPa V-wind anomaly
  - Vertical Wind Shear (vws)      — 850–200 hPa shear magnitude anomaly + arrows

Generation is done in a background thread so the HTTP request returns
immediately (job-ID), and the client polls /aifs/status/<job_id>.
This prevents Gunicorn / Render worker timeouts on slow ECMWF+PSL fetches.
"""
from __future__ import annotations

import io
import os
import sys
import uuid
import time
import datetime
import threading
import tempfile
from collections import OrderedDict

# ── put aifs/ on sys.path so aifs_core imports work without touching those files ──
AIFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aifs")
if AIFS_DIR not in sys.path:
    sys.path.insert(0, AIFS_DIR)

import aifs_core as C

# ── Map folder (coastline shapefile) ──────────────────────────────────────────
_MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map")

# ── Cache directory for AIFS GRIB byte-range files + LTM climatology .npy ─────
# On Fly.io a persistent volume is mounted at /data (see fly.toml [[mounts]]).
# Fall back to a local dir when running in dev (no volume present).
_FLY_DATA = "/data"
_AIFS_CACHE_DIR = (
    os.path.join(_FLY_DATA, "aifs_cache")
    if os.path.isdir(_FLY_DATA)
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "aifs_cache")
)
_CACHE_KEEP_DAYS = 3

# ── Supported levels ──────────────────────────────────────────────────────────
VALID_LEVELS  = [850, 700, 500, 200]
DEFAULT_LEVEL = 200

# ── Product definitions ───────────────────────────────────────────────────────
PRODUCTS = {
    "vp":  {
        "name": "Velocity Potential",
        "desc": "200 hPa velocity-potential anomaly & divergent wind arrows.",
        "levels_supported": [200],
        "default_level": 200,
    },
    "z":   {
        "name": "Geopotential Height",
        "desc": "Geopotential height anomaly vs NCEP 1991–2020 climatology.",
        "levels_supported": VALID_LEVELS,
        "default_level": 500,
    },
    "u":   {
        "name": "Zonal Wind",
        "desc": "Zonal (east–west) wind anomaly vs NCEP 1991–2020 climatology.",
        "levels_supported": VALID_LEVELS,
        "default_level": DEFAULT_LEVEL,
    },
    "v":   {
        "name": "Meridional Wind",
        "desc": "Meridional (north–south) wind anomaly vs NCEP 1991–2020 climatology.",
        "levels_supported": VALID_LEVELS,
        "default_level": DEFAULT_LEVEL,
    },
    "vws": {
        "name": "Vertical Wind Shear",
        "desc": "850–200 hPa vertical wind shear magnitude anomaly with vector arrows.",
        "levels_supported": [],
        "default_level": 0,
    },
}

# ── Default AIFS run parameters ────────────────────────────────────────────────
_N_DAYS      = 3
_LEAD_HOURS  = 216
_BASE_HOURS  = (0, 6, 12, 18)
_WORK_STRIDE = 2
_SMOOTH_DEG  = 3.75
_SHEAR_TOP   = 200
_SHEAR_BOT   = 850

# ── PNG result cache (keyed by render params) ──────────────────────────────────
_RESULT_LOCK = threading.Lock()
_RESULT_CACHE: OrderedDict = OrderedDict()   # key → {png, meta, ts}
_RESULT_MAX   = 20

# ── Background job registry ────────────────────────────────────────────────────
# job_id → {status: "pending"|"done"|"error", png, meta, error, ts}
_JOB_LOCK = threading.Lock()
_JOBS: dict = {}
_JOB_TTL  = 1800          # seconds to keep completed jobs
_JOB_MAX  = 40            # prune old jobs when over this count


def metadata():
    """Return UI metadata consumed by the Jinja template / JS."""
    return {
        "products": [
            {
                "id":              pid,
                "name":            pdef["name"],
                "desc":            pdef["desc"],
                "levels_supported": pdef["levels_supported"],
                "default_level":   pdef["default_level"],
            }
            for pid, pdef in PRODUCTS.items()
        ],
        "levels":          VALID_LEVELS,
        "default_level":   DEFAULT_LEVEL,
        "default_product": "vp",
    }


# ── Result cache helpers ───────────────────────────────────────────────────────
def _rcache_get(key):
    with _RESULT_LOCK:
        v = _RESULT_CACHE.get(key)
        if v:
            _RESULT_CACHE.move_to_end(key)
        return v


def _rcache_put(key, png, meta):
    with _RESULT_LOCK:
        _RESULT_CACHE[key] = {"png": png, "meta": meta, "ts": time.time()}
        _RESULT_CACHE.move_to_end(key)
        while len(_RESULT_CACHE) > _RESULT_MAX:
            _RESULT_CACHE.popitem(last=False)


# ── Job helpers ────────────────────────────────────────────────────────────────
def _job_create() -> str:
    job_id = str(uuid.uuid4())
    with _JOB_LOCK:
        _JOBS[job_id] = {"status": "pending", "png": None,
                         "meta": None, "error": None,
                         "ts": time.time()}
        # prune stale jobs
        if len(_JOBS) > _JOB_MAX:
            cutoff = time.time() - _JOB_TTL
            stale = [jid for jid, j in _JOBS.items()
                     if j["ts"] < cutoff and j["status"] != "pending"]
            for jid in stale:
                del _JOBS[jid]
    return job_id


def job_get(job_id: str):
    with _JOB_LOCK:
        return dict(_JOBS.get(job_id, {}))


def _job_done(job_id, png, meta):
    with _JOB_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(status="done", png=png, meta=meta, ts=time.time())


def _job_error(job_id, msg):
    with _JOB_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(status="error", error=msg, ts=time.time())


# ── Public API ─────────────────────────────────────────────────────────────────
def enqueue(product_id: str, level: int,
            n_days: int | None = None,
            lead_hours: int | None = None) -> tuple[str, dict | None]:
    """
    Start background rendering.  Returns (job_id, cached_result_or_None).
    If the result is already cached, job_id is a synthetic "hit" string and
    the second element holds {png, meta} so the caller can serve immediately.
    """
    if product_id not in PRODUCTS:
        raise ValueError(f"Unknown AIFS product '{product_id}'")

    pdef = PRODUCTS[product_id]
    if product_id == "vp":
        level = 200
    elif product_id == "vws":
        level = 0
    else:
        if level not in VALID_LEVELS:
            level = pdef["default_level"]

    n_days_eff     = int(n_days)     if n_days     is not None else _N_DAYS
    lead_hours_eff = int(lead_hours) if lead_hours is not None else _LEAD_HOURS
    n_days_eff     = max(1, min(30, n_days_eff))
    lead_hours_eff = max(6, min(360, lead_hours_eff))

    cache_key = (product_id, level, n_days_eff, lead_hours_eff)
    cached = _rcache_get(cache_key)
    if cached:
        return "cached", cached          # instant hit — caller serves PNG directly

    job_id = _job_create()
    t = threading.Thread(
        target=_worker,
        args=(job_id, product_id, level, n_days_eff, lead_hours_eff, cache_key),
        daemon=True,
    )
    t.start()
    return job_id, None


def _worker(job_id, product_id, level, n_days, lead_hours, cache_key):
    try:
        t0  = time.time()
        png, meta = _render(product_id, level, n_days, lead_hours)
        meta["seconds"] = round(time.time() - t0, 1)
        meta["cache"]   = False
        _rcache_put(cache_key, png, meta)
        _job_done(job_id, png, meta)
    except Exception as exc:
        import traceback
        _job_error(job_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _render(product_id: str, level: int,
            n_days: int = _N_DAYS,
            lead_hours: int = _LEAD_HOURS) -> tuple[bytes, dict]:
    """Blocking render — runs inside a daemon thread."""
    os.makedirs(_AIFS_CACHE_DIR, exist_ok=True)
    C.prune_cache(_AIFS_CACHE_DIR, _CACHE_KEEP_DAYS)

    steps = C.steps_from_config(n_days, lead_hours)
    coast = C.load_coastlines(C.ensure_coastline(_MAP_DIR))
    base  = C.find_latest_run(steps, _BASE_HOURS)
    valid = [base + datetime.timedelta(hours=s) for s in steps]
    period  = C.period_text(valid, steps)
    run_txt = f"Run: {base:%HZ} • {base:%-d %b %Y}"

    fd, out_path = tempfile.mkstemp(prefix="xpwx_aifs_", suffix=".png")
    os.close(fd)
    try:
        if   product_id == "vp":
            _render_vp(base, steps, valid, coast, period, run_txt, out_path)
        elif product_id == "z":
            _render_z(base, steps, valid, coast, period, run_txt, out_path, level)
        elif product_id == "u":
            _render_u(base, steps, valid, coast, period, run_txt, out_path, level)
        elif product_id == "v":
            _render_v(base, steps, valid, coast, period, run_txt, out_path, level)
        elif product_id == "vws":
            _render_vws(base, steps, valid, coast, period, run_txt, out_path)
        with open(out_path, "rb") as f:
            png = f.read()
    finally:
        try: os.remove(out_path)
        except OSError: pass

    return png, {"run": run_txt, "period": period}


# ── Per-product render helpers ─────────────────────────────────────────────────

def _divergence(u, v, lat, lon):
    import numpy as np
    R = 6.371e6
    lat_r  = np.deg2rad(lat)
    lon_r  = np.deg2rad(lon)
    coslat = np.cos(lat_r)
    dlon   = lon_r[1] - lon_r[0]
    dudx   = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dlon * R * coslat[:, None])
    dvdy   = np.gradient(v * coslat[:, None], lat_r, axis=0) / (R * coslat[:, None])
    return dudx + dvdy


def _poisson_fft(rhs, lat, lon):
    import numpy as np
    R      = 6.371e6
    lat_r  = np.deg2rad(lat)
    lon_r  = np.deg2rad(lon)
    dy     = R * abs(lat_r[1] - lat_r[0])
    coslat = np.cos(lat_r)
    dx_mean = R * (lon_r[1] - lon_r[0]) * np.mean(np.abs(coslat))
    nlat, nlon = rhs.shape
    rhs_clean  = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
    taper = np.ones(nlat)
    for i, la in enumerate(lat):
        if abs(la) > 75.0:
            taper[i] = np.cos(np.deg2rad((abs(la) - 75.0) * 90.0 / 15.0)) ** 2
    taper[np.abs(lat) > 88.0] = 0.0
    rhs_clean *= taper[:, None]
    kx = 2.0 * np.pi * np.fft.fftfreq(nlon, d=dx_mean)
    ky = 2.0 * np.pi * np.fft.fftfreq(nlat, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    F = np.fft.fft2(rhs_clean) / (-K2)
    F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F))


def _render_vp(base, steps, valid, coast, period, run_txt, out_path):
    import numpy as np
    level = 200
    lat, lon, fc = C.fetch_aifs_fields(base, steps, [("u", level), ("v", level)], _AIFS_CACHE_DIR)
    latc, lonc, cl = C.fetch_ltm_fields(
        [v.date() for v in valid], [("uwnd", level), ("vwnd", level)], _AIFS_CACHE_DIR)
    u_fc = fc[("u", level)];  v_fc = fc[("v", level)]
    u_cl = C.interp_to_grid(cl[("uwnd", level)], latc, lonc, lat, lon)
    v_cl = C.interp_to_grid(cl[("vwnd", level)], latc, lonc, lat, lon)
    s = _WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat[sl], lon[sl]
        u_fc, v_fc = u_fc[sl, sl], v_fc[sl, sl]
        u_cl, v_cl = u_cl[sl, sl], v_cl[sl, sl]
    ddeg = abs(lon[1] - lon[0])
    u_anom = C.smooth2d(u_fc - u_cl, 3.75 / ddeg)
    v_anom = C.smooth2d(v_fc - v_cl, 3.75 / ddeg)
    div    = _divergence(u_anom, v_anom, lat, lon)
    chi    = C.smooth2d(_poisson_fft(div, lat, lon), 5.00 / ddeg)
    C.draw_anomaly(lat, lon, chi * 1e-6, coast,
                   meta=dict(title="Velocity Potential Anomaly & Wind Anomaly",
                             period=period, run_txt=run_txt),
                   cb_label="Velocity-Potential Anomaly  (1e6 m²s)",
                   vlim=10.0, arrows=(u_anom, v_anom), arrow_ref=5, out_file=out_path)


def _render_z(base, steps, valid, coast, period, run_txt, out_path, level):
    import numpy as np
    lat, lon, fc = C.fetch_aifs_fields(base, steps, [("z", level)], _AIFS_CACHE_DIR)
    latc, lonc, cl = C.fetch_ltm_fields(
        [v.date() for v in valid], [("hgt", level)], _AIFS_CACHE_DIR)
    z_fc = fc[("z", level)] / C.G_STD
    z_cl = C.interp_to_grid(cl[("hgt", level)], latc, lonc, lat, lon)
    s = _WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat[sl], lon[sl]
        z_fc, z_cl = z_fc[sl, sl], z_cl[sl, sl]
    anom = C.smooth2d(z_fc - z_cl, _SMOOTH_DEG / abs(lon[1] - lon[0]))
    C.draw_anomaly(lat, lon, anom, coast,
                   meta=dict(title=f"{level} hPa Geopotential Height Anomaly",
                             period=period, run_txt=run_txt),
                   cb_label=f"Z{level} Anomaly (m)",
                   vlim=C.nice_vlim(anom), out_file=out_path)


def _render_u(base, steps, valid, coast, period, run_txt, out_path, level):
    lat, lon, fc = C.fetch_aifs_fields(base, steps, [("u", level)], _AIFS_CACHE_DIR)
    latc, lonc, cl = C.fetch_ltm_fields(
        [v.date() for v in valid], [("uwnd", level)], _AIFS_CACHE_DIR)
    u_fc = fc[("u", level)]
    u_cl = C.interp_to_grid(cl[("uwnd", level)], latc, lonc, lat, lon)
    s = _WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat[sl], lon[sl]
        u_fc, u_cl = u_fc[sl, sl], u_cl[sl, sl]
    anom = C.smooth2d(u_fc - u_cl, _SMOOTH_DEG / abs(lon[1] - lon[0]))
    C.draw_anomaly(lat, lon, anom, coast,
                   meta=dict(title=f"{level} hPa Zonal Wind (U) Anomaly",
                             period=period, run_txt=run_txt),
                   cb_label=f"U{level} Anomaly (m/s)",
                   vlim=C.nice_vlim(anom), out_file=out_path)


def _render_v(base, steps, valid, coast, period, run_txt, out_path, level):
    lat, lon, fc = C.fetch_aifs_fields(base, steps, [("v", level)], _AIFS_CACHE_DIR)
    latc, lonc, cl = C.fetch_ltm_fields(
        [v.date() for v in valid], [("vwnd", level)], _AIFS_CACHE_DIR)
    v_fc = fc[("v", level)]
    v_cl = C.interp_to_grid(cl[("vwnd", level)], latc, lonc, lat, lon)
    s = _WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat[sl], lon[sl]
        v_fc, v_cl = v_fc[sl, sl], v_cl[sl, sl]
    anom = C.smooth2d(v_fc - v_cl, _SMOOTH_DEG / abs(lon[1] - lon[0]))
    C.draw_anomaly(lat, lon, anom, coast,
                   meta=dict(title=f"{level} hPa Meridional Wind (V) Anomaly",
                             period=period, run_txt=run_txt),
                   cb_label=f"V{level} Anomaly (m/s)",
                   vlim=C.nice_vlim(anom), out_file=out_path)


def _render_vws(base, steps, valid, coast, period, run_txt, out_path):
    import numpy as np
    top, bot = _SHEAR_TOP, _SHEAR_BOT
    wants_aifs = [("u", top), ("v", top), ("u", bot), ("v", bot)]
    wants_ltm  = [("uwnd", top), ("vwnd", top), ("uwnd", bot), ("vwnd", bot)]
    lat, lon, fc  = C.fetch_aifs_fields(base, steps, wants_aifs, _AIFS_CACHE_DIR)
    latc, lonc, cl = C.fetch_ltm_fields(
        [v.date() for v in valid], wants_ltm, _AIFS_CACHE_DIR)
    shearf = np.hypot(fc[("u", top)] - fc[("u", bot)],
                      fc[("v", top)] - fc[("v", bot)])
    shearc = np.hypot(cl[("uwnd", top)] - cl[("uwnd", bot)],
                      cl[("vwnd", top)] - cl[("vwnd", bot)])
    shearc = C.interp_to_grid(shearc, latc, lonc, lat, lon)
    vec_u  = fc[("u", top)] - fc[("u", bot)]
    vec_v  = fc[("v", top)] - fc[("v", bot)]
    s = _WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat[sl], lon[sl]
        shearf, shearc = shearf[sl, sl], shearc[sl, sl]
        vec_u, vec_v   = vec_u[sl, sl], vec_v[sl, sl]
    ddeg   = abs(lon[1] - lon[0])
    anom   = C.smooth2d(shearf - shearc, _SMOOTH_DEG / ddeg)
    arrows = (C.smooth2d(vec_u, _SMOOTH_DEG / ddeg),
              C.smooth2d(vec_v, _SMOOTH_DEG / ddeg))
    C.draw_anomaly(lat, lon, anom, coast,
                   meta=dict(title=f"Vertical Wind Shear ({bot}–{top} hPa) Anomaly",
                             period=period, run_txt=run_txt),
                   cb_label=f"Shear {bot}–{top} hPa Anomaly (m/s)",
                   vlim=C.nice_vlim(anom), arrows=arrows, arrow_ref=10, out_file=out_path)
