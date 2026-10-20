"""rmm/plot_mjo.py — MJO phase diagram (RMM1 / RMM2).

The diagram design is the original script's, unchanged. The data are the JMA
RMM series (chi200 + u850 + u200, base period 1991–2020) read by rmm/rmm8.py.

    python -m rmm.plot_mjo              # latest 40 days -> rmm/output/mjo_diagram.png
    python -m rmm.plot_mjo 2026-03-17   # 40 days ending on a given date
"""
import datetime
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from rmm import rmm8

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
OUT_FILE = os.path.join(OUT_DIR, "mjo_diagram.png")
N_DAYS = 40


def draw(dates, rmm1, rmm2):
    """Draw the diagram for the given days. Returns the figure (not saved)."""
    dates = np.array(dates)
    rmm1 = np.array(rmm1)
    rmm2 = np.array(rmm2)

    # Limit to last 40 days
    if len(dates) > N_DAYS:
        dates = dates[-N_DAYS:]
        rmm1 = rmm1[-N_DAYS:]
        rmm2 = rmm2[-N_DAYS:]

    # Define start and end points
    start_date, end_date = dates[0], dates[-1]

    # Create figure with a subtle gradient background
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#f0f2f5')
    ax.set_facecolor('#f7f9fc')

    # Gradient trajectory line
    points = np.array([rmm1, rmm2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(rmm1)-1)
    lc = LineCollection(segments, cmap='cool', norm=norm, linewidth=1.0, alpha=0.8)
    lc.set_array(np.arange(len(rmm1)))
    ax.add_collection(lc)

    # Scatter points with edge
    plt.scatter(rmm1, rmm2, c=np.arange(len(rmm1)), cmap='cool', s=40, edgecolor='k', linewidth=0.5, zorder=5)

    # Mark start and end points
    plt.scatter(rmm1[0], rmm2[0], c='limegreen', s=100, marker='*', edgecolor='k', label=f'Start: {start_date}', zorder=10)
    plt.scatter(rmm1[-1], rmm2[-1], c='crimson', s=100, marker='X', edgecolor='k', label=f'End: {end_date}', zorder=10)

    # Grid lines and unit circle
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=1.0)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=1.0)
    for i in [-3, -2, -1, 1, 2, 3]:
        ax.axhline(i, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.axvline(i, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='-', linewidth=1.0, alpha=0.7)
    ax.add_artist(circle)

    # Phase boundaries
    ax.plot([-4, 4], [4, -4], 'k--', linewidth=0.8, alpha=0.5)
    ax.plot([-4, 4], [-4, 4], 'k--', linewidth=0.8, alpha=0.5)

    # Label MJO phases with shadow effect
    for x, y, label in [(-3.5, -1.5, '1'), (-1.5, -3.5, '2'), (1.5, -3.5, '3'),
                        (3.5, -1.5, '4'), (3.5, 1.5, '5'), (1.5, 3.5, '6'),
                        (-1.5, 3.5, '7'), (-3.5, 1.5, '8')]:
        ax.text(x, y, label, fontsize=12, ha='center', color='darkblue', weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add region labels with elegant styling
    ax.text(0, -3.8, 'INDIAN OCEAN', fontsize=10, ha='center', color='teal', fontstyle='italic', va='center')
    ax.text(3.8, 0, 'MARITIME CONTINENT', fontsize=10, rotation=90, ha='center', color='teal', fontstyle='italic', va='center')
    ax.text(0, 3.8, 'WESTERN PACIFIC', fontsize=10, ha='center', color='teal', fontstyle='italic', va='center')
    ax.text(-3.8, 0, 'WEST HEMP & AFRICA', fontsize=10, rotation=90, ha='center', color='teal', fontstyle='italic', va='center')

    # Set limits and labels with professional font
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('RMM1', fontsize=12, color='navy')
    ax.set_ylabel('RMM2', fontsize=12, color='navy')

    # Title with a box
    title = ax.set_title(f'[RMM1, 2] MJO PHASE FROM {start_date} TO {end_date}', fontsize=14, weight='bold', color='navy', pad=12,
                         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.9))

    cc = ax.text(0.99, 0.01, "© XP WEATHER", fontsize=10, ha="right", va="bottom", color='white', transform=ax.transAxes)
    cc.set_bbox(dict(facecolor='black', alpha=0.3, edgecolor='none'))

    return fig


def render_png(dates, rmm1, rmm2):
    """PNG bytes, saved with the same settings as the script output."""
    fig = draw(dates, rmm1, rmm2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    end = datetime.date.fromisoformat(argv[0]) if argv else None
    dates, rmm1, rmm2 = rmm8.window(end, N_DAYS)
    fig = draw(dates, rmm1, rmm2)

    # Save the plot
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Diagram saved to {OUT_FILE}")
    plt.close()


if __name__ == "__main__":
    main()
