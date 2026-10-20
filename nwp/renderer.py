from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import shapefile

WEST, EAST, SOUTH, NORTH = 70.0, 107.0, -5.0, 30.0
GEODATA = Path(__file__).parent / "static" / "geodata"

# Original, restrained grey-to-blue ramp inspired by operational forecast
# graphics. It carries no WeatherBell logo, watermark, or proprietary artwork.
WIND_LEVELS = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80], dtype=float)
WIND_COLORS = [
    "#fbfbf8", "#eeeeeb", "#d8d9d8", "#c0c2c3", "#a6a9aa",
    "#8f9699", "#79b7ec", "#5aa5e6", "#378fe0", "#237bd7",
    "#1767c5", "#1554ad", "#19458f", "#343d82", "#593875",
    "#793667", "#993851", "#b94840", "#d45a3b", "#ed743d",
]
WIND_CMAP = ListedColormap(WIND_COLORS, name="baywind-grey-blue")
WIND_NORM = BoundaryNorm(WIND_LEVELS, WIND_CMAP.N, extend="max")
UNIT_INFO = {
    "knots": (1.943844, "knots", "kn"),
    "m/s": (1.0, "m/s", "m/s"),
    "km/h": (3.6, "km/h", "km/h"),
}


def _init_time(init: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(init.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _add_shapefile_lines(ax: Any, filename: str, color: str, linewidth: float, zorder: int) -> None:
    path = GEODATA / filename
    reader = shapefile.Reader(str(path))
    segments = []
    for shape_record in reader.iterShapes():
        bounds = shape_record.bbox
        if not bounds or bounds[2] < WEST or bounds[0] > EAST or bounds[3] < SOUTH or bounds[1] > NORTH:
            continue
        points = np.asarray(shape_record.points, dtype=np.float32)
        parts = list(shape_record.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            line = points[start:end]
            if len(line) >= 2:
                segments.append(line)
    if segments:
        ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth, zorder=zorder, clip_on=True))


def render_map(
    fields: dict[str, Any],
    model: dict[str, Any],
    step: int,
    units: str = "knots",
    show_speed: bool = True,
    show_streamlines: bool = True,
) -> bytes:
    factor, unit_label, short_unit = UNIT_INFO[units]
    lons = np.asarray(fields["lons"], dtype=np.float64)
    lats = np.asarray(fields["lats"], dtype=np.float64)
    u_ms = np.asarray(fields["u"], dtype=np.float32)
    v_ms = np.asarray(fields["v"], dtype=np.float32)
    speed = np.hypot(u_ms, v_ms) * factor

    init = _init_time(fields.get("init", ""))
    valid = init + timedelta(hours=int(step))
    init_label = init.strftime("%d %b %Y %H%M UTC")
    valid_label = valid.strftime("%H:%M %d %b %Y UTC")

    fig = plt.figure(figsize=(10.25, 8.15), dpi=100, facecolor="white")
    ax = fig.add_axes([0.025, 0.115, 0.95, 0.82])
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ax.set_facecolor("#fbfbf8")
    ax.set_xticks(np.arange(70, 111, 5))
    ax.set_yticks(np.arange(-5, 31, 5))
    ax.grid(True, color="#6f777c", alpha=0.40, linewidth=0.42, linestyle="-", zorder=2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#292d2f")
        spine.set_linewidth(0.55)

    if show_speed:
        ax.pcolormesh(
            lons,
            lats,
            speed,
            cmap=WIND_CMAP,
            norm=WIND_NORM,
            shading="auto",
            alpha=0.96,
            zorder=1,
            rasterized=True,
        )

    # Packaged Natural Earth 1:50m coastlines and country boundaries keep map
    # rendering fast, offline, and independent of any map-tile API.
    _add_shapefile_lines(ax, "ne_50m_admin_0_boundary_lines_land.shp", "#25292b", 0.48, 5)
    _add_shapefile_lines(ax, "ne_50m_coastline.shp", "#171a1b", 0.62, 6)

    if show_streamlines:
        # Components remain in m/s; selecting display units changes only the
        # scalar shading and legend, not the vector-field integration.
        uu = np.ma.masked_invalid(u_ms)
        vv = np.ma.masked_invalid(v_ms)
        ax.streamplot(
            lons,
            lats,
            uu,
            vv,
            density=1.12,
            color="#4a5054",
            linewidth=0.48,
            arrowsize=0.62,
            minlength=0.08,
            maxlength=4.0,
            zorder=4,
        )

    # Longitude labels along the top and latitude labels at the right edge.
    for lon in range(75, 106, 5):
        ax.text(lon, NORTH - 0.12, f"{lon}°E", ha="center", va="top", fontsize=7.4, color="#202426", zorder=8)
    for lat in range(5, 30, 5):
        ax.text(EAST - 0.12, lat, f"{lat}°N", ha="right", va="center", fontsize=7.4, color="#202426", zorder=8)

    fig.text(
        0.035,
        0.973,
        f"{model['name']} {model['resolution']}  |  Init {init_label}  |  10 m Wind Speed ({unit_label}) + Streamlines",
        ha="left",
        va="top",
        fontsize=9.7,
        fontweight="medium",
        color="#141719",
    )
    fig.text(
        0.965,
        0.973,
        f"Hour: {int(step):03d}  •  Valid: {valid_label}",
        ha="right",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#141719",
    )
    fig.add_artist(plt.Line2D([0.025, 0.975], [0.952, 0.952], transform=fig.transFigure, color="#6d7172", linewidth=0.45))

    if show_speed:
        cax = fig.add_axes([0.085, 0.064, 0.70, 0.017])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=WIND_NORM, cmap=WIND_CMAP),
            cax=cax,
            orientation="horizontal",
            ticks=np.arange(0, 81, 5),
            extend="max",
            spacing="proportional",
        )
        colorbar.ax.tick_params(labelsize=7, length=2, pad=1.5, colors="#24282a")
        colorbar.outline.set_linewidth(0.45)
        for label in colorbar.ax.get_xticklabels():
            label.set_fontfamily("DejaVu Sans")
        maximum = float(np.nanmax(speed)) if np.isfinite(speed).any() else 0.0
        fig.text(0.955, 0.067, f"Max: {maximum:.1f} {short_unit}", ha="right", va="center", fontsize=8.5, fontweight="bold", color="#141719")
        fig.text(0.085, 0.039, f"10 m wind speed · {unit_label}", ha="left", va="center", fontsize=7.4, color="#313638")
    else:
        fig.text(0.08, 0.065, "10 m wind streamlines", ha="left", va="center", fontsize=8.2, color="#313638")

    attribution = model["source_label"]
    if attribution == "ECMWF Open Data":
        attribution = "ECMWF Open Data · CC BY 4.0 · data.ecmwf.int"
    fig.text(0.965, 0.022, f"Source: {attribution}  |  Map outlines: Natural Earth  |  Baywind", ha="right", va="center", fontsize=6.4, color="#4b5052")

    output = BytesIO()
    fig.savefig(output, format="png", dpi=100, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return output.getvalue()
