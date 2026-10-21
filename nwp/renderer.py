from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import numpy as np
import shapefile

WEST, EAST, SOUTH, NORTH = 70.0, 110.0, 0.0, 30.0
GEODATA = Path(__file__).parent / "static" / "geodata"

# A smooth grey/blue/purple/red wind ramp with reference-style tick levels
# through 135 knots. Coarse forecast vectors are interpolated for display only.
WIND_LEVELS_KN = np.array([
    4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90,
    94, 98, 102, 110, 120, 135,
], dtype=float)
WIND_TICKS_KN = WIND_LEVELS_KN
# Approximate sampled colors from the user-provided reference legend: grey,
# blues/cyan, green/yellow, orange/red, and the warm high-end tones.
WIND_COLORS = [
    "#e5e5e5", "#d7d7d7", "#bbbbbb", "#959595", "#1365d4", "#2781f0",
    "#4fa6f5", "#96d3fd", "#b4f1fb", "#54f055", "#78f771", "#b0fda4",
    "#c4febb", "#eefcb3", "#f8c953", "#f9a815", "#ff8d09", "#fe4e05",
    "#fa3003", "#bb0104", "#a40307", "#643c31", "#794f46", "#8a635a",
    "#debcb4", "#f1ddd5", "#fcf0e6", "#fee6e5", "#fdc7c5", "#e58382",
    "#e06464", "#d55051", "#c73b3a", "#a3201f",
]
# One swatch for each legend interval, plus a final over-range swatch.
WIND_CMAP = ListedColormap(WIND_COLORS + [WIND_COLORS[-1]], name="baywind-reference", N=len(WIND_LEVELS_KN))
WIND_CMAP.set_under("#ffffff")
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
    dpi: int = 150,
    field_id: str = "wind",
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

    # Reference-matched wind bands and full 4–135 range. Spatial values are
    # interpolated below; the legend's discrete intervals preserve its color key.
    unit_scale = factor / KNOT_FACTOR
    wind_levels = WIND_LEVELS_KN * unit_scale
    wind_ticks = WIND_TICKS_KN * unit_scale
    wind_norm = BoundaryNorm(wind_levels, WIND_CMAP.N, extend="max")

    init = _init_time(fields.get("init", ""))
    valid = init + timedelta(hours=int(step))
    init_label = init.strftime("%Hz %d %b %Y")
    valid_label = valid.strftime("%Hz %d %b %Y")

    # A near-square plotting field prevents the Bay of Bengal from appearing
    # horizontally compressed; the chart itself remains a full-width image.
    fig = plt.figure(figsize=(10, 8), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.035, 0.072, 0.93, 0.875])
    ax.set_facecolor("#fbfbf8")
    ax.set_xticks(np.arange(70, 111, 5))
    ax.set_yticks(np.arange(0, 31, 5))
    # set_xticks can expand Matplotlib's view limits; reapply the exact domain.
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
    for lon in range(75, 110, 5):
        ax.text(lon, NORTH - 0.12, f"{lon}°E", ha="center", va="top", fontsize=7.4, color="#202426", zorder=8)
    ax.text(EAST - 0.12, NORTH - 0.12, f"{int(EAST)}°E", ha="right", va="top", fontsize=7.4, color="#202426", zorder=8)
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

    display_unit = {"knots": "Knots", "m/s": "m/s", "km/h": "km/h"}[units]
    level_label = "200 hPa" if field_id == "wind_200" else "10 m"
    if show_speed and show_streamlines:
        layer_label = f"{level_label} Wind Speed ({display_unit}) & Streamlines"
    elif show_speed:
        layer_label = f"{level_label} Wind Speed ({display_unit})"
    else:
        layer_label = f"{level_label} Streamlines"
    left_parts = [
        (model["name"], True),
        ("•", True),
        (model["resolution"], True),
        ("•", True),
        ("Init:", True),
        (init_label, False),
        ("•", True),
        (layer_label, False),
    ]
    right_parts = [
        ("Hour:", True),
        (str(int(step)), False),
        ("•", True),
        ("Valid:", True),
        (valid_label, False),
    ]

    def measure_parts(parts: list[tuple[str, bool]], font_size: float) -> tuple[list[float], float]:
        widths = []
        for text, bold in parts:
            prop = FontProperties(family="DejaVu Sans", size=font_size, weight="bold" if bold else "normal")
            width = TextPath((0, 0), text, prop=prop).get_extents().width / (72 * fig.get_figwidth())
            widths.append(width)
        total = sum(widths) + 0.008 * (len(widths) - 1)
        return widths, total

    right_size = 8.5
    right_widths, right_width = measure_parts(right_parts, right_size)
    max_left_width = 0.965 - 0.035 - right_width - 0.018
    header_size = 8.5
    widths, total_width = measure_parts(left_parts, header_size)
    while total_width > max_left_width and header_size > 7.0:
        header_size -= 0.25
        widths, total_width = measure_parts(left_parts, header_size)

    cursor = 0.035
    for (text, bold), width in zip(left_parts, widths):
        fig.text(cursor, 0.979, text, ha="left", va="top", fontsize=header_size,
                 fontweight="bold" if bold else "normal", color="#141719")
        cursor += width + 0.008
    cursor = 0.965 - right_width
    for (text, bold), width in zip(right_parts, right_widths):
        fig.text(cursor, 0.979, text, ha="left", va="top", fontsize=right_size,
                 fontweight="bold" if bold else "normal", color="#141719")
        cursor += width + 0.008
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.961, 0.961], transform=fig.transFigure, color="#6d7172", linewidth=0.45))

    if show_speed:
        cax = fig.add_axes([0.15, 0.038, 0.70, 0.018])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=wind_norm, cmap=WIND_CMAP),
            cax=cax,
            orientation="horizontal",
            boundaries=wind_levels,
            ticks=wind_ticks,
            extend="max",
            spacing="uniform",
        )
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True,
                                labelsize=6.5, length=1.8, pad=1.0, colors="#24282a")
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
        fig.text(0.085, 0.018, f"{level_label} wind speed · {unit_label} · Streamlines  |  Source: {attribution}", ha="left", va="center", fontsize=6.3, color="#313638")
    else:
        attribution = "ECMWF Open Data · CC BY 4.0" if model.get("source_label") == "ECMWF Open Data" else model.get("source_label", "Official forecast data")
        fig.text(0.085, 0.018, f"{level_label} wind streamlines  |  Source: {attribution}", ha="left", va="center", fontsize=6.3, color="#313638")

    output = BytesIO()
    fig.savefig(output, format="png", dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return output.getvalue()


SCALAR_PALETTE = {
    "temperature": ("°C", np.arange(-10, 42, 2), "RdYlBu_r"),
    "mslp": ("hPa", np.arange(960, 1042, 2), "RdYlBu_r"),
    "precipitation": ("mm", np.array([0.0, 0.1, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]),
                      ListedColormap(["#ffffff", "#dff3ff", "#9edcff", "#42b6e8", "#238b45", "#51ad32", "#a8cf32", "#f4e33b", "#f6a433", "#e85d3f", "#c52640", "#7c1e70", "#4a104f"])),
}


def _smooth_scalar_grid(lons: np.ndarray, lats: np.ndarray, values: np.ndarray, target_step: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = float(np.nanmedian(np.diff(lons))) if len(lons) > 1 else target_step
    dy = float(np.nanmedian(np.diff(lats))) if len(lats) > 1 else target_step
    if max(abs(dx), abs(dy)) <= target_step * 1.15:
        return lons, lats, values
    nx = max(len(lons), int(round((lons[-1] - lons[0]) / target_step)) + 1)
    ny = max(len(lats), int(round((lats[-1] - lats[0]) / target_step)) + 1)
    fine_lons = np.linspace(lons[0], lons[-1], nx, dtype=np.float64)
    fine_lats = np.linspace(lats[0], lats[-1], ny, dtype=np.float64)
    valid = np.isfinite(values)
    safe = np.where(valid, values, np.nan)
    # Interpolate rows first and then columns, preserving missing cells.
    across = np.vstack([np.interp(fine_lons, lons, row) for row in safe]).astype(np.float32)
    fine = np.vstack([np.interp(fine_lats, lats, across[:, col]) for col in range(nx)]).T.astype(np.float32)
    # Restore missing areas by interpolating a validity mask with the same grid.
    mask_across = np.vstack([np.interp(fine_lons, lons, row.astype(float)) for row in valid]).astype(np.float32)
    mask_fine = np.vstack([np.interp(fine_lats, lats, mask_across[:, col]) for col in range(nx)]).T
    fine[mask_fine < 0.5] = np.nan
    return fine_lons, fine_lats, fine


def render_scalar_map(fields: dict[str, Any], model: dict[str, Any], step: int, field_id: str, units: str = "hPa", dpi: int = 150) -> bytes:
    """Weather-chart-style scalar map; same domain, dimensions, header, and frame as the wind chart."""
    lons = np.asarray(fields["lons"], dtype=np.float64)
    lats = np.asarray(fields["lats"], dtype=np.float64)
    values = np.asarray(fields["values"], dtype=np.float32)
    if values.shape != (len(lats), len(lons)):
        raise ValueError("Scalar field array does not match its latitude/longitude grid.")
    plot_lons, plot_lats, plot_values = _smooth_scalar_grid(lons, lats, values)
    init = _init_time(fields.get("init", ""))
    valid = init + timedelta(hours=int(step))
    init_label = init.strftime("%Hz %d %b %Y")
    valid_label = valid.strftime("%Hz %d %b %Y")

    if field_id == "temperature":
        if units == "°F":
            display = plot_values * 9.0 / 5.0 + 32.0
            scale = np.arange(-10, 111, 4)
            cmap = "RdYlBu_r"
            colorbar_ticks = np.arange(-10, 111, 20)
            unit_label = "°F"
        else:
            display = plot_values
            scale = np.arange(-10, 42, 2)
            cmap = "RdYlBu_r"
            colorbar_ticks = np.arange(-10, 41, 10)
            unit_label = "°C"
        layer_label = "2 m Temperature"
        contour_stride = 10 if units == "°F" else 5
        major_lines = np.arange(-10, 112 if units == "°F" else 42, contour_stride)
        cmap_obj = cmap
        norm = BoundaryNorm(scale, plt.get_cmap(cmap).N, extend="both")
        extend = "both"
    elif field_id == "mslp":
        display = plot_values
        scale = np.arange(960, 1042, 2)
        cmap_obj = "RdYlBu_r"
        norm = BoundaryNorm(scale, plt.get_cmap(cmap_obj).N, extend="both")
        colorbar_ticks = np.arange(960, 1041, 10)
        major_lines = np.arange(960, 1041, 4)
        layer_label = "Mean Sea-Level Pressure"
        unit_label = "hPa"
        extend = "both"
    elif field_id in {"precipitation", "precipitation_6h"}:
        display = plot_values
        levels = SCALAR_PALETTE["precipitation"][1]
        cmap_obj = SCALAR_PALETTE["precipitation"][2]
        scale = levels
        norm = BoundaryNorm(scale, cmap_obj.N, extend="max")
        colorbar_ticks = [0.1, 1, 5, 10, 20, 50, 100, 200]
        major_lines = np.array([])
        layer_label = "Accumulated Precipitation" if field_id == "precipitation" else "6 h Precipitation"
        unit_label = "mm"
        extend = "max"
    elif field_id == "gust":
        factor, _, short_unit = UNIT_INFO[units]
        display = plot_values * factor
        scale = WIND_LEVELS_KN * (factor / KNOT_FACTOR)
        cmap_obj = WIND_CMAP
        norm = BoundaryNorm(scale, WIND_CMAP.N, extend="max")
        colorbar_ticks = np.array([5, 10, 20, 30, 40, 50, 70, 90, 120]) * (factor / KNOT_FACTOR)
        major_lines = np.array([])
        layer_label = f"10 m Wind Gust ({units})"
        unit_label = units
        extend = "max"
    elif field_id == "shear_200_850":
        factor, _, short_unit = UNIT_INFO[units]
        display = plot_values * factor
        scale = WIND_LEVELS_KN * (factor / KNOT_FACTOR)
        cmap_obj = WIND_CMAP
        norm = BoundaryNorm(scale, WIND_CMAP.N, extend="max")
        colorbar_ticks = WIND_TICKS_KN * (factor / KNOT_FACTOR)
        major_lines = np.array([])
        layer_label = f"200–850 hPa Wind Shear ({units})"
        unit_label = units
        extend = "max"
    else:
        raise ValueError(f"Unsupported scalar map parameter: {field_id}")

    fig = plt.figure(figsize=(10, 8), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.035, 0.072, 0.93, 0.875])
    ax.set_facecolor("#fbfbf8")
    ax.set_xticks(np.arange(70, 111, 5))
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ax.grid(True, color="#6f777c", alpha=0.40, linewidth=0.42, linestyle="-", zorder=2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#292d2f")
        spine.set_linewidth(0.55)

    mesh = ax.pcolormesh(plot_lons, plot_lats, np.ma.masked_invalid(display), cmap=cmap_obj, norm=norm, shading="auto", zorder=1, rasterized=True)
    if major_lines.size:
        in_range = major_lines[(major_lines >= np.nanmin(display)) & (major_lines <= np.nanmax(display))]
        if in_range.size:
            contours = ax.contour(plot_lons, plot_lats, display, levels=in_range, colors="#1c2b38", linewidths=0.38, alpha=0.55, zorder=3)
            if field_id == "mslp" or field_id == "temperature":
                ax.clabel(contours, contours.levels[::2] if len(contours.levels) > 1 else contours.levels,
                          inline=True, fontsize=6.3, fmt="%.0f", colors="#273441")

    _add_shapefile_lines(ax, "ne_50m_admin_0_boundary_lines_land.shp", "#25292b", 0.48, 5)
    _add_shapefile_lines(ax, "ne_50m_coastline.shp", "#171a1b", 0.62, 6)
    for lon in range(75, 110, 5):
        ax.text(lon, NORTH - 0.12, f"{lon}°E", ha="center", va="top", fontsize=7.4, color="#202426", zorder=8)
    ax.text(EAST - 0.12, NORTH - 0.12, f"{int(EAST)}°E", ha="right", va="top", fontsize=7.4, color="#202426", zorder=8)
    for lat in range(5, 30, 5):
        ax.text(EAST - 0.12, lat, f"{lat}°N", ha="right", va="center", fontsize=7.4, color="#202426", zorder=8)

    ax.text(EAST - 0.45, SOUTH + 0.52, "@XP WEATHER", transform=ax.transData, ha="right", va="bottom",
            fontsize=8.3, fontweight="bold", color="white", zorder=12,
            bbox={"boxstyle": "round,pad=0.34", "facecolor": "#14243a", "edgecolor": "#ffffff", "linewidth": 0.6, "alpha": 0.88})

    if field_id == "mslp":
        minimum = float(np.nanmin(display))
        maximum = float(np.nanmax(display))
        min_rounded = int(np.floor(minimum + 0.5))
        max_rounded = int(np.floor(maximum + 0.5))
        extrema = f"Min: {min_rounded} hPa | Max: {max_rounded} hPa"
        min_row, min_col = np.unravel_index(np.nanargmin(display), display.shape)
        min_lon = float(plot_lons[min_col])
        min_lat = float(plot_lats[min_row])
        ax.text(
            min_lon, min_lat, str(min_rounded),
            ha="center", va="center", fontsize=8.2, fontweight="bold",
            color="#162638", zorder=13,
            path_effects=[path_effects.withStroke(linewidth=1.25, foreground="white")],
        )
    elif field_id == "temperature":
        shown_values = display[np.isfinite(display)]
        extrema = f"Min/Max: {np.nanmin(shown_values):.1f}/{np.nanmax(shown_values):.1f} {unit_label}"
    else:
        extrema = f"Max: {np.nanmax(display):.1f} {unit_label}"
    ax.text(WEST + 0.55, SOUTH + 0.42, extrema, transform=ax.transData, ha="left", va="bottom",
            fontsize=8.1, fontweight="bold", color="#1b2733", zorder=12,
            bbox={"boxstyle": "round,pad=0.34", "facecolor": "#ffffff", "edgecolor": "#45515c", "linewidth": 0.55, "alpha": 0.90})

    header_parts = [
        (model["name"], True), ("•", True), (model["resolution"], True), ("•", True),
        ("Init:", True), (init_label, False), ("•", True), (layer_label, False),
    ]
    right_parts = [("Hour:", True), (str(int(step)), False), ("•", True), ("Valid:", True), (valid_label, False)]
    def measure_parts(parts: list[tuple[str, bool]], font_size: float) -> tuple[list[float], float]:
        widths = []
        for text, bold in parts:
            prop = FontProperties(family="DejaVu Sans", size=font_size, weight="bold" if bold else "normal")
            widths.append(TextPath((0, 0), text, prop=prop).get_extents().width / (72 * fig.get_figwidth()))
        return widths, sum(widths) + 0.008 * (len(widths) - 1)
    right_widths, right_width = measure_parts(right_parts, 8.5)
    max_left_width = 0.965 - 0.035 - right_width - 0.018
    header_size = 8.5
    widths, total_width = measure_parts(header_parts, header_size)
    while total_width > max_left_width and header_size > 7.0:
        header_size -= 0.25
        widths, total_width = measure_parts(header_parts, header_size)
    cursor = 0.035
    for (text, bold), width in zip(header_parts, widths):
        fig.text(cursor, 0.979, text, ha="left", va="top", fontsize=header_size,
                 fontweight="bold" if bold else "normal", color="#141719")
        cursor += width + 0.008
    cursor = 0.965 - right_width
    for (text, bold), width in zip(right_parts, right_widths):
        fig.text(cursor, 0.979, text, ha="left", va="top", fontsize=8.5,
                 fontweight="bold" if bold else "normal", color="#141719")
        cursor += width + 0.008
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.961, 0.961], transform=fig.transFigure, color="#6d7172", linewidth=0.45))

    cax = fig.add_axes([0.15, 0.038, 0.70, 0.018])
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=colorbar_ticks,
                            extend=extend, spacing="uniform")
    colorbar.ax.xaxis.set_ticks_position("bottom")
    colorbar.ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True,
                            labelsize=6.5, length=1.8, pad=1.0, colors="#24282a")
    colorbar.outline.set_linewidth(0.45)
    if field_id in {"gust", "shear_200_850"}:
        ticks_kn = [5, 10, 20, 30, 40, 50, 70, 90, 120] if field_id == "gust" else WIND_TICKS_KN
        if units == "m/s":
            tick_labels = [f"{x * 0.514444:.0f}" for x in ticks_kn]
        elif units == "km/h":
            tick_labels = [f"{x * 1.852:.0f}" for x in ticks_kn]
        else:
            tick_labels = [str(int(x)) for x in ticks_kn]
        colorbar.ax.set_xticklabels(tick_labels)
    elif field_id in {"precipitation", "precipitation_6h"}:
        colorbar.ax.set_xticklabels(["0.1", "1", "5", "10", "20", "50", "100", "200"])
    attribution = "ECMWF Open Data · CC BY 4.0" if model.get("source_label") == "ECMWF Open Data" else model.get("source_label", "Official forecast data")
    interval_text = fields.get("interval", "")
    footer_label = layer_label + (f" · {interval_text}" if interval_text else "")
   # fig.text(0.085, 0.018, f"{footer_label} · {unit_label}  |  Source: {attribution}", ha="left", va="center", fontsize=6.3, color="#313638")
    output = BytesIO()
    fig.savefig(output, format="png", dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return output.getvalue()
