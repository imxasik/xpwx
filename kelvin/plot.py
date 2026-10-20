"""kelvin.plot — the reference Kelvin VP200 phase-space chart.

The layout, colours, labels, marker rings, day-of-month annotations and the
@XPWEATHER watermark are the user's original design.  The only adaptation is
that the figure is returned as PNG bytes instead of being written to disk.
"""
from __future__ import annotations

import datetime as dt
import io
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import patheffects                  # noqa: E402
import numpy as np                                  # noqa: E402

from . import physics                               # noqa: E402

DPI = 300
_LOCK = threading.Lock()      # matplotlib's pyplot state is process-global


def csv_text(forecast):
    """The trajectory table: one row per forecast lead time."""
    x = np.asarray(forecast["x"], dtype=float)
    y = np.asarray(forecast["y"], dtype=float)
    phase_deg = np.asarray(forecast["phase_deg"], dtype=float)
    amp = np.asarray(forecast["amplitude"], dtype=float)
    hours = np.asarray(forecast["steps"], dtype=float)
    run = forecast["run"]
    lines = ["run_date,run_hour,lead_hours,lead_days,phase_deg,"
             "relative_amplitude,pc1,pc2"]
    for h, ph, a, xx, yy in zip(hours, phase_deg, amp, x, y):
        lines.append(f"{run[0]},{run[1]},{int(h)},{h / 24:.3f},{ph:.3f},"
                     f"{a:.6f},{xx:.6f},{yy:.6f}")
    return "\n".join(lines) + "\n"


def _write_csv(path, forecast):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(csv_text(forecast))


def render_png(forecast, out_csv=None):
    """Render one forecast dict to PNG bytes.  Returns ``(BytesIO, n_points)``."""
    x = np.asarray(forecast["x"], dtype=float)
    y = np.asarray(forecast["y"], dtype=float)
    phase_deg = np.asarray(forecast["phase_deg"], dtype=float)
    amp = np.asarray(forecast["amplitude"], dtype=float)
    hours = np.asarray(forecast["steps"], dtype=float)
    rdt = forecast["run_datetime"]

    valid = np.isfinite(x) & np.isfinite(y)
    x, y, phase_deg, amp = x[valid], y[valid], phase_deg[valid], amp[valid]
    hours = hours[valid]
    if x.size < 2:
        raise ValueError("the forecast produced fewer than two valid phase points")

    fig = plt.figure(figsize=(6.2, 6.3), dpi=DPI, facecolor="white")
    try:
        ax = fig.add_axes([0.118, 0.110, 0.785, 0.775])
        ax.set_facecolor("white")
        lim = 4.0

        # Reference-style crosshairs and corner-to-corner phase-sector guides.
        guide = dict(color="black", lw=0.8, ls=(0, (7, 5)), zorder=1)
        ax.plot([-lim, lim], [0, 0], **guide)
        ax.plot([0, 0], [-lim, lim], **guide)
        ax.plot([-lim, lim], [-lim, lim], **guide)
        ax.plot([-lim, lim], [lim, -lim], **guide)

        # White central zone; its border is redrawn above the trajectory.
        ax.add_patch(plt.Circle((0, 0), 1.0, facecolor="white",
                                edgecolor="none", zorder=2))

        # PCHIP keeps the daily path smooth without inventing overshoot.
        curve = np.column_stack([physics.pchip_curve(x),
                                 physics.pchip_curve(y)])
        ax.plot(curve[:, 0], curve[:, 1], color="#0067A8", linewidth=2.5,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)

        daily_idx = np.flatnonzero((hours % 24) == 0)
        ax.scatter(x[daily_idx], y[daily_idx], s=72, color="#101010",
                   edgecolors="#D32F2F", linewidths=0.65, zorder=6)

        ax.add_patch(plt.Circle((0, 0), 1.0, facecolor="none",
                                edgecolor="black", linewidth=0.9, zorder=7))

        ax.scatter([x[0]], [y[0]], s=88, color="#101010",
                   edgecolors="green", linewidths=1.6, zorder=8)
        ax.scatter([x[-1]], [y[-1]], s=88, color="#101010",
                   edgecolors="#B0008E", linewidths=1.6, zorder=8)

        ax.text(0.75, 0.94, "ECMWF IFS", transform=ax.transAxes,
                color="#0067A8", fontsize=9.2, weight="bold",
                ha="left", va="center")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(-4, 4.1, 2))
        ax.set_yticks(np.arange(-4, 4.1, 2))
        ax.set_xticks(np.arange(-4, 4.01, 0.5), minor=True)
        ax.set_yticks(np.arange(-4, 4.01, 0.5), minor=True)
        ax.tick_params(axis="both", which="major", direction="out",
                       length=7, width=1.0, labelsize=11, colors="black",
                       top=True, right=True, labeltop=False, labelright=False)
        ax.tick_params(axis="both", which="minor", direction="out",
                       length=3.5, width=0.7, colors="black",
                       top=True, right=True)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.35)

        # Day-of-month printed on the daily marker itself.
        for j in daily_idx:
            valid_time = rdt + dt.timedelta(hours=int(hours[j]))
            txt = ax.annotate(valid_time.strftime("%d"), (x[j], y[j]),
                              xytext=(0, 0), textcoords="offset points",
                              ha="center", va="center", fontsize=7.0,
                              color="white", weight="bold", zorder=10)
            txt.set_path_effects([patheffects.withStroke(
                linewidth=0.20, foreground="#202020", alpha=0.9)])

        wm = ax.text(0.985, 0.014, "@XPWEATHER", transform=ax.transAxes,
                     ha="right", va="bottom", fontsize=8.0, weight="bold",
                     color="#263744", alpha=0.68, zorder=20)
        wm.set_path_effects([patheffects.withStroke(
            linewidth=2.0, foreground="white", alpha=0.9)])

        # Exact phase/geographic labels used in the reference.
        ax.set_xlabel("Phase 2 (EastPac/Indian)   Phase 3",
                      fontsize=13, color="black", labelpad=3)
        ax.set_ylabel("Phase 1 (EastPac/Africa)   Phase 8",
                      fontsize=13, color="black", labelpad=4)
        ax.yaxis.set_label_coords(-0.085, 0.5)
        fig.text(0.50, 0.914, "Phase 7 (Africa/WestPac)   Phase 6",
                 fontsize=13, color="black", ha="center", va="center")
        fig.text(0.94, 0.54, "Phase 5 (SAmerica/Maritime)   Phase 4",
                 fontsize=13, color="black", ha="center", va="center",
                 rotation=270)
        fig.text(0.50, 0.993,
                 f"{rdt:%m/%d/%Y/%HZ}: 15S-15N - Kelvin filtered 200VP",
                 fontsize=14, color="black", ha="center", va="top")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    finally:
        plt.close(fig)
    buf.seek(0)

    if out_csv:
        _write_csv(out_csv, dict(forecast, x=x, y=y, phase_deg=phase_deg,
                                 amplitude=amp, steps=hours))
    return buf, len(hours)


def render_png_locked(forecast, out_csv=None):
    with _LOCK:
        return render_png(forecast, out_csv=out_csv)
