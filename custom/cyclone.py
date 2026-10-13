"""custom/cyclone.py — tropical-cyclone diagnostic figures, one set per storm.

Drop-in addon. Diagnostics are in cyclone/ (core.py, fields.py, analysis.py) and
the two figures in cyclone/plot.py.

Products are per storm x figure (storms listed in cyclone/storms.py):
  Figure 1  200-hPa divergence · deep-layer shear · vortex tilt · Hadley φ–p
  Figure 2  tangential wind & θ′ · U with overturning · hodograph · steering map

Storm products use the storm's own date (D), not the date window. The centre is
found in the data, and the tracking uses D-1 and D+1.

Sidebar group: Cyclone
"""
import sys
import threading

import numpy as np

from pro import config

_ROOT = config.BASE_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cyclone import analysis, plot  # noqa: E402
from cyclone.storms import STORMS  # noqa: E402

_LOCK = threading.Lock()    # matplotlib's pyplot state is global
_RESULTS = {}               # storm key -> analysis result (computed once)


def _result(key):
    with _LOCK:
        if key not in _RESULTS:
            _RESULTS[key] = analysis.analyze(key)
        return _RESULTS[key]


def _compute(pkg, dates):
    res = _result(pkg["storm"])
    m = res["map"]
    return m["lat"], m["lon"], {"main": np.zeros((1, 1)), "result": res}


def _render1(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return plot.figure1(data["result"], coast_segs)


def _render2(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return plot.figure2(data["result"], coast_segs)


PRODUCTS = {}
for _key, _spec in STORMS.items():
    _name = _spec["name"].title()
    for _fig, _kind, _label in (
        ("fig1", "cyc_fig1", "Structure · vortex tilt, shear, divergence, Hadley section"),
        ("fig2", "cyc_fig2", "Structure · tangential wind & θ′, U with overturning, hodograph, steering"),
    ):
        _pid = f"cyc_{_key}_{_fig}"
        PRODUCTS[_pid] = {
            "id": _pid,
            "title": f"{_name} ({_spec['date']}) · {_label}",
            "name": f"{_name} · {'Fig 1' if _fig == 'fig1' else 'Fig 2'}",
            "tag": "Cyclone",
            "desc": f"{_spec['kind']} {_name}, {_spec['date']}: {_label}.",
            "kind": _kind,
            "level": None,
            "storm": _key,
            "cb_label": "Cyclone diagnostics",
        }

KINDS = {
    "cyc_fig1": {"compute": _compute, "render": _render1, "tag": "Cyclone",
                 "title": "Tropical-cyclone structure, figure 1"},
    "cyc_fig2": {"compute": _compute, "render": _render2, "tag": "Cyclone",
                 "title": "Tropical-cyclone structure, figure 2"},
}
