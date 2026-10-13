"""custom/mjo_rmm8.py — MJO phase diagram (RMM1 / RMM2) in the sidebar.

Drop-in addon: no core file is changed. The drawing is rmm/plot_mjo.py (the
original design). The data are the JMA RMM series in rmm/rmm8.py.

The window is the 40 days ending on the app's selected end date, the same
date the other products use. In auto mode that is the NCEP record end.

Sidebar group: Hovmöller
"""
import os
import sys
import threading

import numpy as np

from pro import config

_ROOT = config.BASE_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rmm import plot_mjo, rmm8  # noqa: E402

_LOCK = threading.Lock()   # matplotlib's pyplot state is global
N_DAYS = 40


def _compute(pkg, dates):
    end = dates[-1]
    days, r1, r2 = rmm8.window(end, N_DAYS)
    data = {"main": np.zeros((1, 1)), "dates": days, "rmm1": r1, "rmm2": r2}
    return np.zeros(1), np.zeros(1), data


def _render(lat, lon, data, pkg, coast_segs, dates, **_kw):
    with _LOCK:
        return plot_mjo.render_png(data["dates"], data["rmm1"], data["rmm2"])


PRODUCTS = {
    "mjo_rmm8": {
        "id": "mjo_rmm8",
        "title": "MJO Phase Diagram — RMM (JMA)",
        "name": "MJO Phase",
        "tag": "Hovmöller",
        "desc": "RMM1/RMM2 MJO phase diagram (chi200 + u850 + u200, JMA; "
                "base period 1991–2020), last 40 days.",
        "kind": "mjo_rmm8",
        "level": None,
        "cb_label": "MJO Phase",
    },
}

KINDS = {
    "mjo_rmm8": {
        "compute": _compute,
        "render": _render,
        "tag": "Hovmöller",
        "title": "MJO Phase Diagram — RMM (JMA)",
    },
}
