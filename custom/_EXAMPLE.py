"""REFERENCE TEMPLATE — this file is NOT loaded (starts with '_').

To use it: copy this file into custom/ with a name that does NOT begin with
'_' (e.g. custom/my_products.py), edit, redeploy. The engine auto-discovers it.

Below are BOTH ways to add a product. Uncomment what you need.
"""

# ============================================================================
# TIER 1 — reuse a built-in kind. No code needed, just a config block.
# The simplest possible new map: an anomaly of a single variable.
# ============================================================================
PRODUCTS = {
    "myu500": {
        "id": "myu500",
        "title": "My Zonal Wind Anomaly — 500 hPa",
        "name": "My U500",
        "tag": "Custom",                     # sidebar group (made-up groups work)
        "desc": "500-hPa zonal-wind anomaly (obs minus 1991-2020 climatology).",
        "kind": "anom",                      # single-variable anomaly
        "variable": "uwnd", "level": 500,
        "show_wind": False, "plot_scale": 1.0,
        "vlim": 10.0, "cint": 2.0,
        "cb_label": "Zonal Wind Anomaly  (m/s)",
    },
}

# ============================================================================
# TIER 2 — a brand-new diagnostic (a new "kind"). Uncomment and edit to use.
#   compute(pkg, dates) -> (lat, lon, data)   data must have data["main"]
#   render(...) is optional; omit it to use the default map renderer.
# ============================================================================
# import numpy as np
# def _my_field(pkg, dates):
#     """Example: absolute zonal wind (NOT an anomaly) at pkg['level']."""
#     from pro import data                 # helpers live here
#     lat, lon = data._latlon("uwnd")
#     u = data._mean_field("uwnd", pkg["level"], dates, "obs")
#     return lat, lon, {"main": np.asarray(u, dtype=float) * pkg["plot_scale"]}
#
# KINDS = {
#     "mykind": {
#         "compute": _my_field,            # required
#         # "render": my_render_function,  # optional — default renderer if omitted
#         "tag": "Custom",
#         "title": "My Diagnostic — 500 hPa",
#     },
# }
```
