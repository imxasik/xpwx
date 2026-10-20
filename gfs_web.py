"""
gfs_web.py — Web adapter for the existing GFS Forecast Map Generator.

The original GFS engine in ./gfs is kept intact. This module only exposes
that engine to the existing Flask application, with a small thread lock
because the original fetch/plot modules keep region/step in process state.
"""
from __future__ import annotations

import io
import os
import sys
import time
import tempfile
import threading
from collections import OrderedDict

# The original GFS files use top-level imports (config, fetch, plot, climo).
# Put their directory first on sys.path without rewriting those modules.
GFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfs")
if GFS_DIR not in sys.path:
    sys.path.insert(0, GFS_DIR)

import config as gconfig
import fetch as gfetch
import plot as gplot
import climo as gclimo

_LOCK = threading.RLock()
_CACHE = OrderedDict()
_CACHE_MAX = 12

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
        "variables": [
            {
                "id": k,
                "key": v["key"],
                "name": v["name"],
                "level_required": v["key"] in gconfig.LEVEL_REQUIRED_VARS,
                "anomaly_supported": v["key"] in gconfig.ANOMALY_SUPPORTED_VARS,
            }
            for k, v in gconfig.VARIABLES.items()
        ],
        "levels": [{"id": k, "mb": v} for k, v in gconfig.PRESSURE_LEVELS.items()],
        "periods": [
            {"id": k, "days": v["days"], "label": v["label"]}
            for k, v in gconfig.AVERAGE_OPTIONS.items()
        ],
        "regions": [
            {"id": k, "name": v["name"], "bounds": list(v["bounds"])}
            for k, v in gconfig.REGIONS.items()
        ],
        "default_step": gconfig.DEFAULT_STEP,
        "default_region": "1",
        "default_variable": "1",
        "default_level": "3",
        "default_period": "1",
    }

def generate(variable_key, level, avg_days, step, compute_anom, region_id):
    """Return PNG bytes + useful metadata for one GFS map."""
    if variable_key not in {v["key"] for v in gconfig.VARIABLES.values()}:
        raise ValueError(f"unknown GFS variable '{variable_key}'")

    level = int(level or 0)
    avg_days = int(avg_days or 0)
    step = int(step or gconfig.DEFAULT_STEP)
    region_id = str(region_id or "2")
    if region_id not in gconfig.REGIONS:
        raise ValueError(f"unknown GFS region '{region_id}'")
    if avg_days not in {0, 1, 3, 5, 7}:
        raise ValueError("GFS period must be 0, 1, 3, 5, or 7 days")
    if step < 0 or step > 384:
        raise ValueError("forecast step must be between 0 and 384 hours")

    if variable_key not in gconfig.LEVEL_REQUIRED_VARS:
        level = 0
    if variable_key not in gconfig.ANOMALY_SUPPORTED_VARS:
        compute_anom = False

    region = gconfig.REGIONS[region_id]
    lon_min, lon_max, lat_min, lat_max = region["bounds"]
    plot_variable = f"{variable_key}_anomaly" if compute_anom else variable_key

    # Include the current GFS cycle in the cache key so old images don't persist forever.
    run_dt = gfetch.latest_gfs_run_dt()
    key = (
        run_dt.strftime("%Y%m%d%H"), variable_key, level, avg_days, step,
        bool(compute_anom), region_id
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached, {
            "run": run_dt.strftime("%Y-%m-%d %HZ"),
            "region": region["name"],
            "cache": True,
        }

    with _LOCK:
        # Re-check after waiting for another request in this process.
        cached = _cache_get(key)
        if cached is not None:
            return cached, {
                "run": run_dt.strftime("%Y-%m-%d %HZ"),
                "region": region["name"],
                "cache": True,
            }

        gfetch.set_region(lon_min, lon_max, lat_min, lat_max)
        gfetch.set_step(step)
        gplot.set_region(lon_min, lon_max, lat_min, lat_max)

        t0 = time.time()
        ltm_thread = None
        if compute_anom:
            ltm_thread = gclimo.prefetch_ltm(
                variable_key, lat_min, lat_max, lon_min, lon_max,
                level_hpa=level if level else None)

        country_segs, coast_segs = gplot.load_shapes()

        lat, lon, data, extra_data = gfetch.get_data(
            variable_key, level, avg_days, compute_anom=compute_anom)

        if compute_anom:
            gclimo.wait_prefetch(ltm_thread)

        actual_step = avg_days * 24 if avg_days > 0 else step

        fd, out_path = tempfile.mkstemp(prefix="xpwx_gfs_", suffix=".png")
        os.close(fd)
        try:
            gplot.draw_map(
                run_dt, actual_step,
                country_segs, coast_segs,
                lat, lon, data,
                extra_data=extra_data,
                variable=plot_variable,
                level=level,
                avg_days=avg_days,
                region_name=region["name"],
                out_file=out_path,
            )
            with open(out_path, "rb") as f:
                png = f.read()
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

        _cache_put(key, png)
        return png, {
            "run": run_dt.strftime("%Y-%m-%d %HZ"),
            "region": region["name"],
            "seconds": round(time.time() - t0, 1),
            "cache": False,
        }
