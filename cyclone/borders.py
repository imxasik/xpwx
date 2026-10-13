"""cyclone/borders.py — country-border lines for the map panels.

Source: map/ne_50m_admin_0_borders.shp, built by cyclone/build_borders.py from the
Natural Earth 50 m countries (public domain). Read with pyshp. Longitudes are
returned on 0–360, and each line is split where it crosses the dateline.
"""
import os
import threading

import numpy as np
import shapefile

from pro import config

PATH = os.path.join(config.SHP_DIR, "ne_50m_admin_0_borders.shp")

_LOCK = threading.Lock()
_CACHE = None


def segments():
    """List of (n, 2) arrays [lon (0-360), lat] for every border polyline."""
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            out = []
            for shp in shapefile.Reader(PATH).shapes():
                pts = np.asarray(shp.points, dtype=float)
                parts = list(shp.parts) + [len(pts)]
                for i in range(len(shp.parts)):
                    p = pts[parts[i]:parts[i + 1]]
                    if p.shape[0] < 2:
                        continue
                    lon = np.where(p[:, 0] < 0, p[:, 0] + 360.0, p[:, 0])
                    seg = np.column_stack([lon, p[:, 1]])
                    breaks = np.where(np.abs(np.diff(seg[:, 0])) > 180)[0] + 1
                    out.extend(np.split(seg, breaks))
            _CACHE = out
        return _CACHE


def draw(ax, lon_lo, lon_hi, lat_lo, lat_hi):
    """Draw the borders that fall inside the given window on ax."""
    for part in segments():
        if part[:, 0].max() < lon_lo or part[:, 0].min() > lon_hi:
            continue
        if part[:, 1].max() < lat_lo or part[:, 1].min() > lat_hi:
            continue
        ax.plot(part[:, 0], part[:, 1], color="#3a3a3a", lw=0.8, zorder=7,
                solid_capstyle="round")
