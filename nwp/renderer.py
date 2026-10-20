from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import shapefile

WEST, EAST, SOUTH, NORTH = 70.0, 107.0, -5.0, 30.0
GEODATA = Path(__file__).parent / "static" / "geodata"

# A smooth grey/blue/purple/red wind ramp with reference-style tick levels
# through 135 knots. Coarse forecast vectors are interpolated for display only.
WIND_LEVELS_KN = np.array([
    4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90,
    94, 98, 102, 110, 120, 135,
], dtype=float)
WIND_TICKS_KN = WIND_LEVELS_KN
WIND_COLOR_STOPS = [
    "#fbfbf8", "#f0f0ed", "#d9dad9", "#c1c3c4", "#a7aaac", "#8e9598",
    "#78b9ed", "#55a5e7", "#388fdf", "#237bd6", "#1769c8", "#1558b5",
    "#174a99", "#283f88", "#403b7d", "#5b3976", "#73366e", "#88345f",
    "#9c354f", "#ad3947", "#bc4140", "#cb493b", "#d65337", "#df6035",
    "#e76c35", "#ed7937", "#e98234", "#dd7830", "#cf682d", "#c05a2b",
    "#b14c2a", "#a3402a", "#93362a", "#832d29", "#702629", "#60232a",
    "#521e27",
]
WIND_CMAP = LinearSegmentedColormap.from_list(
    "baywind-operational", WIND_COLOR_STOPS, N=256
)
KNOT_FACTOR = 1.943844
UNIT_INFO = {
    "knots": (KNOT_FACTOR, "knots", "kn"),
    "m/s": (1.0, "m/s", "m/s"),
    "km/h": (3.6, "km/h", "km/h"),
}

# Read the Natural Earth outlines once per worker rather than reopening and
# rebuilding their geometry for each new forecast hour.
_SHAPE_CACHE: dict[str, list[np.ndarray]] = {}


def _init_time(init: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(init.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _outline_segments(filename: str) -> list[np.ndarray]:
    if filename in _SHAPE_CACHE:
        return _SHAPE_CACHE[filename]
    path = GEODATA / filename
    reader = shapefile.Reader(str(path))
    segments: list[np.ndarray] = []
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
    _SHAPE_CACHE[filename] = segments
    return segments


def _add_shapefile_lines(ax: Any, filename: str, color: str, linewidth: float, zorder: int) -> None:
    segments = _outline_segments(filename)
    if segments:
        ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth, zorder=zorder, clip_on=True))


def _smooth_regular_vectors(
    lons: np.ndarray,
    lats: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    target_step: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly densify coarse regular grids for clean shading and flow lines."""
    dx = float(np.nanmedian(np.diff(lons))) if len(lons) > 1 else target_step
    dy = float(np.nanmedian(np.diff(lats))) if len(lats) > 1 else target_step
    if max(abs(dx), abs(dy)) <= target_step * 1.15:
        return lons, lats, u, v

    nx = max(len(lons), int(round((lons[-1] - lons[0]) / target_step)) + 1)
    ny = max(len(lats), int(round((lats[-1] - lats[0]) / target_step)) + 1)
    fine_lons = np.linspace(lons[0], lons[-1], nx, dtype=np.float64)
    fine_lats = np.linspace(lats[0], lats[-1], ny, dtype=np.float64)

    def interp(values: np.ndarray) -> np.ndarray:
        across = np.vstack([np.interp(fine_lons, lons, row) for row in values]).astype(np.float32)
        return np.vstack([np.interp(fine_lats, lats, across[:, col]) for col in range(nx)]).T.astype(np.float32)

    return fine_lons, fine_lats, interp(u), interp(v)


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
    # HGEFS/GEFS mean grids are coarse. Bilinear vector interpolation to a
    # finer plotting mesh removes the visible square cells without inventing
    # extra forecast detail beyond a smooth interpolation of official values.
    plot_lons, plot_lats, plot_u, plot_v = _smooth_regular_vectors(lons, lats, u_ms, v_ms)
    speed = np.hypot(plot_u, plot_v) * factor

    # Continuous ramp, with reference-style tick marks over the full range.
    unit_scale = factor / KNOT_FACTOR
    wind_levels = WIND_LEVELS_KN * unit_scale
    wind_ticks = WIND_TICKS_KN * unit_scale
    wind_norm = Normalize(vmin=wind_levels[0], vmax=wind_levels[-1], clip=False)

    init = _init_time(fields.get("init", ""))
    valid = init + timedelta(hours=int(step))
    init_label = init.strftime("%d %b %Y %H%M UTC")
    valid_label = valid.strftime("%H:%M %d %b %Y UTC")

    # A near-square plotting field prevents the Bay of Bengal from appearing
    # horizontally compressed; the chart itself remains a full-width image.
    fig = plt.figure(figsize=(10.25, 9.75), dpi=100, facecolor="white")
    ax = fig.add_axes([0.035, 0.072, 0.93, 0.875])
    ax.set_facecolor("#fbfbf8")
    ax.set_xticks(np.arange(70, 107, 5))
    ax.set_yticks(np.arange(-5, 31, 5))
    # set_xticks can expand Matplotlib's view limits (previously to 110°E),
    # which left a blank strip east of the data. Reapply the exact data domain.
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ax.grid(True, color="#6f777c", alpha=0.40, linewidth=0.42, linestyle="-", zorder=2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#292d2f")
        spine.set_linewidth(0.55)

    if show_speed:
        ax.pcolormesh(
            plot_lons,
            plot_lats,
            speed,
            cmap=WIND_CMAP,
            norm=wind_norm,
            shading="auto",
            alpha=1.0,
            zorder=1,
            rasterized=True,
        )

    # Bundled Natural Earth boundaries keep plotting offline and lightweight.
    _add_shapefile_lines(ax, "ne_50m_admin_0_boundary_lines_land.shp", "#25292b", 0.48, 5)
    _add_shapefile_lines(ax, "ne_50m_coastline.shp", "#171a1b", 0.62, 6)

    if show_streamlines:
        # Plot streamline vectors on a ~0.2° mesh for 0.1° data. Shading stays
        # full resolution, but streamline integration has far fewer grid cells.
        lon_step = float(np.nanmedian(np.diff(plot_lons))) if len(plot_lons) > 1 else 0.25
        stride = max(1, int(round(0.2 / lon_step))) if lon_step < 0.18 else 1
        uu = np.ma.masked_invalid(plot_u[::stride, ::stride])
        vv = np.ma.masked_invalid(plot_v[::stride, ::stride])
        ax.streamplot(
            plot_lons[::stride],
            plot_lats[::stride],
            uu,
            vv,
            density=0.95,
            color="#4a5054",
            linewidth=0.48,
            arrowsize=0.62,
            minlength=0.08,
            maxlength=4.0,
            zorder=4,
        )

    # Longitude labels along the top and latitude labels at the right edge.
    for lon in range(75, 107, 5):
        ax.text(lon, NORTH - 0.12, f"{lon}°E", ha="center", va="top", fontsize=7.4, color="#202426", zorder=8)
    for lat in range(5, 30, 5):
        ax.text(EAST - 0.12, lat, f"{lat}°N", ha="right", va="center", fontsize=7.4, color="#202426", zorder=8)

    # User-requested XP Weather credit, overlaid neatly inside the map.
    ax.text(
        EAST - 0.45,
        SOUTH + 0.52,
        "@XP WEATHER",
        transform=ax.transData,
        ha="right",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color="white",
        zorder=12,
        bbox={"boxstyle": "round,pad=0.34", "facecolor": "#14243a", "edgecolor": "#ffffff", "linewidth": 0.6, "alpha": 0.88},
    )

    resolution = model["resolution"]
    if "ensemble mean" in resolution.lower():
        resolution = "Hybrid ensemble"
    header_left = f"{model['name']} · {resolution} · Init {init_label}"
    fig.text(
        0.035,
        0.979,
        header_left,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="medium",
        color="#141719",
    )
    fig.text(
        0.965,
        0.979,
        f"F{int(step):03d}  ·  Valid: {valid_label}",
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#141719",
    )
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.961, 0.961], transform=fig.transFigure, color="#6d7172", linewidth=0.45))

    if show_speed:
        cax = fig.add_axes([0.085, 0.047, 0.70, 0.018])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=wind_norm, cmap=WIND_CMAP),
            cax=cax,
            orientation="horizontal",
            boundaries=wind_levels,
            ticks=wind_ticks,
            extend="max",
            spacing="uniform",
        )
        colorbar.ax.tick_params(labelsize=6.5, length=1.8, pad=1.0, colors="#24282a")
        colorbar.outline.set_linewidth(0.45)
        if units == "m/s":
            labels = [f"{value:.0f}" if value.is_integer() else f"{value:.1f}" for value in WIND_TICKS_KN * 0.514444]
        elif units == "km/h":
            labels = [f"{value:.0f}" for value in WIND_TICKS_KN * 1.852]
        else:
            labels = [str(int(value)) for value in WIND_TICKS_KN]
        colorbar.ax.set_xticklabels(labels)
        maximum = float(np.nanmax(speed)) if np.isfinite(speed).any() else 0.0
        ax.text(
            WEST + 0.55,
            SOUTH + 0.42,
            f"Max: {maximum:.1f} {short_unit}",
            transform=ax.transData,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#1b2733",
            zorder=12,
            bbox={"boxstyle": "round,pad=0.34", "facecolor": "#ffffff", "edgecolor": "#45515c", "linewidth": 0.55, "alpha": 0.88},
        )
        attribution = "ECMWF Open Data · CC BY 4.0" if model.get("source_label") == "ECMWF Open Data" else model.get("source_label", "Official forecast data")
        fig.text(0.085, 0.018, f"10 m wind speed · {unit_label} · Streamlines  |  Source: {attribution}", ha="left", va="center", fontsize=6.3, color="#313638")
    else:
        attribution = "ECMWF Open Data · CC BY 4.0" if model.get("source_label") == "ECMWF Open Data" else model.get("source_label", "Official forecast data")
        fig.text(0.085, 0.018, f"10 m wind streamlines  |  Source: {attribution}", ha="left", va="center", fontsize=6.3, color="#313638")

    output = BytesIO()
    fig.savefig(output, format="png", dpi=100, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return output.getvalue()
