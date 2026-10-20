"""custom/kelvin_phase.py — ECMWF IFS Kelvin VP200 phase-space forecast.

Drop-in addon: the sidebar group is **Hovmöller**, and no reanalysis/NCEP
data are used.  The computation lives in the ``kelvin`` package:

    kelvin.fetch      run selection + polite d200 range requests (reuses
                      ecmwf/engine.py's HTTP client and GRIB2 decoder)
    kelvin.physics    spherical Poisson inversion, 15S-15N VP200,
                      eastward 2.5-20 day Kelvin filter, EOF projection
    kelvin.plot       the user's reference phase-space chart
    kelvin.eofs       the two digitized EOF Fourier vectors

Controls
--------
* **Auto Date** — newest complete ECMWF IFS 00Z/12Z run (360 h horizon).
* **Manual Date** — newest complete 00Z/12Z run on or before the picked
  date.  ECMWF Open Data keeps only the latest ~12 runs (2-3 days), so a
  much older date raises a clear message instead of silently showing a
  different run.
* The day-count box is hidden for this product: the trajectory length is
  the IFS forecast horizon, not an averaging window.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np

from pro import config

_ROOT = config.BASE_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kelvin import pipeline, plot  # noqa: E402


def _as_date(value):
    if value is None or isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"invalid forecast run date '{value}' (use YYYY-MM-DD)")


def _compute(pkg, dates):
    mode = str(pkg.get("request_mode", "auto")).lower()
    if mode != "manual":
        mode = "auto"
    date = _as_date(pkg.get("request_date")) if mode == "manual" else None
    forecast = pipeline.build_forecast(mode=mode, date=date)
    data = dict(forecast)
    data["main"] = np.zeros((1, 1), dtype=float)
    return np.zeros(1), np.zeros(1), data


def _render(lat, lon, data, pkg, coast_segs, dates, **_kw):
    buf, _n = plot.render_png_locked(data)
    return buf


PRODUCTS = {
    "kelvin_phase": {
        "id": "kelvin_phase",
        "title": "ECMWF IFS Kelvin VP200 Phase-Space Forecast",
        "name": "Kelvin VP200",
        "tag": "Hovmöller",
        "desc": "15S-15N Kelvin-filtered 200-hPa velocity-potential EOF phase "
                "diagram from the latest complete ECMWF IFS 00Z/12Z run "
                "(15-day daily trajectory).",
        "kind": "kelvin_phase",
        "level": None,
        "forecast": True,
        "uses_days": False,
        "needs_coast": False,
        "csv": "/kelvin/csv",
        "cb_label": "Kelvin VP200 Phase",
    },
}

KINDS = {
    "kelvin_phase": {
        "compute": _compute,
        "render": _render,
        "tag": "Hovmöller",
        "title": "ECMWF IFS Kelvin VP200 Phase-Space Forecast",
    },
}
