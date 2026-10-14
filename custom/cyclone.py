"""custom/cyclone.py — tropical-cyclone diagnostic figures for a user-given location.

Drop-in addon. Diagnostics are in cyclone/ (core.py, fields.py, analysis.py) and
the two figures in cyclone/plot.py.

Two products (sidebar group: Cyclone):
  Dynamic Fig        200-hPa divergence · deep-layer shear · vortex tilt · Hadley φ–p
  ThermoDynamic Fig  tangential wind & θ′ · U with overturning · hodograph · steering map

Each product needs a latitude and longitude from the user (``needs_latlon``). The
date is the end date of the date selector (the last date of the window), and the
tracking uses D-1 and D+1.
"""
import sys
import threading

import numpy as np

from pro import config

_ROOT = config.BASE_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cyclone import analysis, plot  # noqa: E402

_LOCK = threading.Lock()    # matplotlib's pyplot state is global
_RESULTS = {}               # (lat, lon, date) -> analysis result


def _result(lat, lon, date):
    key = (round(float(lat), 4), round(float(lon), 4), date.isoformat())
    with _LOCK:
        if key not in _RESULTS:
            _RESULTS[key] = analysis.analyze(lat, lon, date)
        return _RESULTS[key]


def _compute(pkg, dates):
    if "lat" not in pkg or "lon" not in pkg:
        raise ValueError("latitude and longitude are required for this product")
    res = _result(pkg["lat"], pkg["lon"], dates[-1])
    m = res["map"]
    return m["lat"], m["lon"], {"main": np.zeros((1, 1)), "result": res}


def _render1(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return plot.figure1(data["result"], coast_segs)


def _render2(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return plot.figure2(data["result"], coast_segs)


_COMMON = {
    "level": None,
    "tag": "Cyclone",
    "needs_latlon": True,
    "cb_label": "Cyclone diagnostics",
}

PRODUCTS = {
    "cyc_dynamic": dict(
        _COMMON,
        id="cyc_dynamic",
        name="Dynamic Fig",
        title="Dynamic Fig · tropical-cyclone structure at a chosen location",
        desc="200-hPa divergence, max-layer shear (200–850 hPa search), vortex tilt and Hadley-type "
             "φ–p section around a lat/lon you enter.",
        kind="cyc_fig1",
    ),
    "cyc_thermo": dict(
        _COMMON,
        id="cyc_thermo",
        name="ThermoDynamic Fig",
        title="ThermoDynamic Fig · tropical-cyclone thermal structure at a chosen location",
        desc="Tangential wind and θ′, U with overturning, hodograph and steering "
             "map around a lat/lon you enter.",
        kind="cyc_fig2",
    ),
}

KINDS = {
    "cyc_fig1": {"compute": _compute, "render": _render1, "tag": "Cyclone",
                 "title": "Tropical-cyclone structure, Dynamic Fig"},
    "cyc_fig2": {"compute": _compute, "render": _render2, "tag": "Cyclone",
                 "title": "Tropical-cyclone structure, ThermoDynamic Fig"},
}
