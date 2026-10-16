"""
ecmwf_web.py — Web adapter for the ECMWF Forecast Map Maker.

The ECMWF engine (user's standalone script, adapted only to return PNG
bytes instead of writing files) lives in ./ecmwf/engine.py and is kept
as-is design-wise. This module only exposes that engine to the Flask
application, following the same adapter pattern as gfs_web.py, with a
thread lock because the engine keeps HTTP/cache state process-wide.

Date behaviour (as requested):
  * mode="auto"   -> the latest available ECMWF IFS run is used.
  * mode="manual" -> the picked run date + run hour (00/06/12/18Z).
The number input that the reanalysis panel uses for "days" is used
here purely as the *forecast hour* (single snapshot, no averaging).
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import threading
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_HERE, "ecmwf", "engine.py")

_spec = importlib.util.spec_from_file_location("ecmwf_engine", _ENGINE_PATH)
eengine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eengine)

_LOCK = threading.RLock()
_CACHE = OrderedDict()
_CACHE_MAX = 12

# Latest-run resolution hits data.ecmwf.int; cache it briefly so paging
# through maps/hours does not re-probe the server on every request.
_RUN_TTL = 20 * 60  # seconds
_latest_run = {"run": None, "at": 0.0}

DEFAULT_MAP = "combo"
DEFAULT_REGION = "bangladesh"
DEFAULT_STEP = 24
RUN_HOURS = (0, 6, 12, 18)
MAX_STEP = 240


def _cache_get(key):
    with _LOCK:
        val = _CACHE.get(key)
        if val is not None:
            _CACHE.move_to_end(key)
        return val


def _cache_put(key, data):
    with _LOCK:
        _CACHE[key] = data
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def metadata():
    return {
        "maps": [
            {
                "id": k,
                "name": eengine.MAP_TITLES[k][0],
                "unit": eengine.MAP_TITLES[k][1],
                "desc": eengine.MAP_DESCRIPTIONS[k],
            }
            for k in eengine.MAP_ORDER
        ],
        "regions": [
            {"id": k, "name": eengine.REGION_LABELS[k]}
            for k in eengine.REGION_PRESETS
        ],
        "run_hours": list(RUN_HOURS),
        "default_map": DEFAULT_MAP,
        "default_region": DEFAULT_REGION,
        "default_step": DEFAULT_STEP,
        "step_interval": eengine.STEP_H,
        "max_step": MAX_STEP,
    }


def resolve_run(mode, manual_date=None, run_hour=0):
    """Return (YYYYMMDD, HH) for auto (latest available) or manual mode."""
    if mode == "manual":
        if not manual_date:
            raise ValueError("manual mode requires a run date")
        try:
            day = datetime.date.fromisoformat(str(manual_date))
        except ValueError:
            raise ValueError("invalid run date, use YYYY-MM-DD")
        run_hour = int(run_hour or 0)
        if run_hour not in RUN_HOURS:
            raise ValueError("run hour must be one of 00, 06, 12, 18")
        return day.strftime("%Y%m%d"), f"{run_hour:02d}"

    now = time.time()
    if _latest_run["run"] is not None and now - _latest_run["at"] < _RUN_TTL:
        return _latest_run["run"]
    run = eengine.find_latest_run()
    _latest_run.update(run=run, at=now)
    return run


def generate(map_key, mode="auto", manual_date=None, run_hour=0,
             forecast_hour=DEFAULT_STEP, region_id=DEFAULT_REGION):
    """Return PNG bytes + useful metadata for one ECMWF forecast map."""
    map_key = str(map_key or DEFAULT_MAP)
    if map_key not in eengine.MAP_TITLES:
        raise ValueError(f"unknown ECMWF map '{map_key}'")
    region_id = str(region_id or DEFAULT_REGION)
    if region_id not in eengine.REGION_PRESETS:
        raise ValueError(f"unknown ECMWF region '{region_id}'")

    try:
        step = int(forecast_hour)
    except (TypeError, ValueError):
        raise ValueError("forecast hour must be a number")
    if step < 0 or step > MAX_STEP:
        raise ValueError(f"forecast hour must be between 0 and {MAX_STEP}")
    if step % eengine.STEP_H != 0:
        raise ValueError(
            f"forecast hour must be a multiple of {eengine.STEP_H} "
            f"(ECMWF open-data step interval)")
    if map_key == "tp24" and step < 24:
        raise ValueError(
            "24-hour Precipitation (tp24) needs a forecast hour of at least 24.")

    mode = "manual" if str(mode).lower() == "manual" else "auto"
    run = resolve_run(mode, manual_date, run_hour)

    key = (run, map_key, step, region_id)
    cached = _cache_get(key)
    if cached is not None:
        png, meta = cached
        return png, {**meta, "cache": True}

    with _LOCK:
        # Re-check after waiting for another request in this process.
        cached = _cache_get(key)
        if cached is not None:
            png, meta = cached
            return png, {**meta, "cache": True}

        t0 = time.time()
        png, meta = eengine.render_map(map_key, run, step, region_id)
        meta = {**meta, "seconds": round(time.time() - t0, 1), "cache": False}
        _cache_put(key, (png, meta))
        return png, meta
