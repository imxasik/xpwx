"""cyclone/plot.py — the two tropical-cyclone figures.

Vertical axis: log-pressure, 1000 hPa at the bottom to 100 hPa at the top (the
figures stop at 100 hPa; see fields.OUT_TOP_HPA). Each level is placed at
z = ln(1000 / p), so the spacing is even in log-pressure.

Dynamic Fig (2 x 2):
  A  200-hPa divergence · outflow streamlines · heights
  B  850–200 hPa shear magnitude and vectors; max-shear point
  C  vortex tilt: 850 hPa centre, 700/500/300/200 hPa centres, 850-hPa vorticity,
     850-hPa wind-speed contours, and the 850→200 hPa shear arrow with up/downshear
  D  Hadley-type φ–p section through the centre (ω shaded; arrows v and w)
ThermoDynamic Fig (2 x 2):
  A  tangential wind (contours) and θ′ (shaded), N–S section; arrows: in-plane v, w
  B  zonal wind U (shaded) with the secondary circulation (arrows v, w)
  C  hodograph (0–500 km mean), coloured by pressure, with the storm-motion vector
  D  steering map: 850–200 hPa mean wind and the storm track; motion and steering
     vectors, with upstream and downstream sides marked

Section arrows: the horizontal and vertical components are scaled separately to
the largest value on each panel, so vertical motion is visible. The scaling is
stated on each panel. Arrow directions are drawn in the plot's own coordinates.
Colour limits are taken from the data in each panel.
"""
import io

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.patheffects as pe
import matplotlib.ticker
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from cyclone import borders, core

KT = core.KT
P_TICKS = (1000, 850, 700, 500, 300, 200, 100)
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]
LEVEL_COLOURS = {850: "#111111", 700: "#2b6cb0", 500: "#2f855a", 300: "#b7791f", 200: "#c53030"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _symlim(arr, floor=1e-12):
    return max(float(np.nanmax(np.abs(arr))), floor)


def _filled(ax, x, y, z, cmap, vmin, vmax):
    """Smooth filled contours (no cell blocks on the 2.5-deg input)."""
    levels = np.linspace(vmin, vmax, 25)
    return ax.contourf(x, y, z, levels=levels, cmap=cmap, extend="both")


def _cbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.locator = matplotlib.ticker.MaxNLocator(7)
    cb.update_ticks()
    return cb


def _pz(p):
    """Log-pressure height z = ln(1000 / p), 1000 hPa at z = 0."""
    return np.log(1000.0 / np.asarray(p, dtype=float))


def _sort_by_pz(lev, *fields):
    """z for the levels, sorted ascending, and the fields reordered to match (level = axis 0)."""
    z = _pz(lev)
    o = np.argsort(z)
    return z[o], [np.asarray(f)[o] for f in fields]


def _outlines(ax, coast_segs, lon_lo, lon_hi, lat_lo, lat_hi):
    """Coastlines and country borders, both clipped to the map window."""
    borders.draw(ax, lon_lo, lon_hi, lat_lo, lat_hi)
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            if part[:, 0].max() < lon_lo or part[:, 0].min() > lon_hi:
                continue
            if part[:, 1].max() < lat_lo or part[:, 1].min() > lat_hi:
                continue
            ax.plot(part[:, 0], part[:, 1], color=borders.LINE_COLOUR, lw=borders.LINE_WIDTH, zorder=6)


def _map_axes(ax, lat, lon, title):
    ax.set_xlim(lon[0], lon[-1])
    ax.set_ylim(lat[0], lat[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.tick_params(labelsize=8)


def _mark_center(ax, lat_c, lon_c, label=None):
    ax.plot(lon_c, lat_c, marker="+", ms=12, mew=2, color="black", zorder=8)
    if label:
        ax.text(lon_c + 0.2, lat_c + 0.2, label, fontsize=8, zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def _box(ax, text, loc="upper left"):
    ha, x = ("left", 0.02) if "left" in loc else ("right", 0.98)
    yy, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, yy, text, transform=ax.transAxes, fontsize=8, va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.92), zorder=12)


def _pressure_axis(ax, lev):
    lev = np.asarray(lev, dtype=float)
    ax.set_ylim(0.0, float(_pz(lev.min())))           # bottom 1000 hPa, top = top level
    ticks = [t for t in P_TICKS if lev.min() <= t <= lev.max()]
    ax.set_yticks([float(_pz(t)) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel("Pressure (hPa)", fontsize=8)
    ax.tick_params(labelsize=8)


def _section_axes(ax, y_km, title):
    ax.set_xlim(y_km[0], y_km[-1])
    ax.set_xlabel("North–south distance from centre (km)", fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.axvline(0, color="#888877", lw=0.6, ls=":")
    ax.tick_params(labelsize=8)


def _section_arrows(ax, x, z, u, w, stride=None):
    """In-plane flow arrows on a section: u along x, w upward along z.

    Each component is scaled so that the largest value fills about 6 % of the
    panel span. Arrows use angles='xy', so each one is drawn between its two
    plotted endpoints and follows the flow as plotted. Upward w points up.
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    u = np.asarray(u, float)
    w = np.asarray(w, float)
    xs = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    zs = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    kx = 0.06 * xs / _symlim(u)
    kz = 0.06 * zs / _symlim(w)
    sx = stride[0] if stride else max(1, x.size // 16)
    sz = stride[1] if stride else max(1, z.size // 12)
    X, Z = np.meshgrid(x[::sx], z[::sz])
    ax.quiver(X, Z, (u[::sz, ::sx] * kx), (w[::sz, ::sx] * kz),
              angles="xy", scale_units="xy", scale=1.0,
              width=0.0032, headwidth=4.0, headlength=4.4, headaxislength=4.0,
              color="#111111", zorder=7)


def _arrow(ax, x0, y0, x1, y1, color, lw=2.2, dashed=False, zorder=10,
           head_length=0.9, head_width=0.42):
    style = f"-|>,head_length={head_length},head_width={head_width}"
    p = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                        lw=lw, color=color, zorder=zorder, path_effects=HALO,
                        linestyle="--" if dashed else "-")
    ax.add_patch(p)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _watermark(fig, src):
    fig.text(0.99, 0.005, "@XPWEATHER", ha="right", va="bottom", fontsize=9, color="#222211")
    fig.text(0.01, 0.005, src, ha="left", va="bottom", fontsize=7, color="#666655")


SRC = "NCEP/NCAR Reanalysis · PSL/NOAA · diagnostics: cyclone/"


def _centre_note(res):
    """Title suffix for the centre; flags a search-box edge maximum."""
    c = res["center"]
    txt = f"centre {c['lat']:.2f}°N {c['lon']:.2f}°E"
    if c.get("at_edge"):
        txt += "  (centre on the search-box edge: check the input lat/lon)"
    return txt


def _bearing_vec(bearing_deg):
    """Unit (east, north) vector for a bearing (deg, toward)."""
    b = np.deg2rad(bearing_deg)
    return np.sin(b), np.cos(b)


# ---------------------------------------------------------------------------
# Dynamic Fig
# ---------------------------------------------------------------------------

def _panel_A(ax, fig, res, coast_segs):
    m = res["map"]
    lat, lon = m["lat"], m["lon"]
    lat_c, lon_c = res["center"]["lat"], res["center"]["lon"]
    dv = m["div200"]
    pm = _filled(ax, lon, lat, dv, "RdBu_r", -_symlim(dv), _symlim(dv))
    _cbar(fig, pm, ax, "200-hPa divergence (10⁻⁵ s⁻¹)")
    cs = ax.contour(lon, lat, m["z200"] / 10.0, colors="#333333", linewidths=0.6, levels=12)
    ax.clabel(cs, fmt="%d", fontsize=6)
    ax.streamplot(lon, lat, m["u200"], m["v200"], color="#1f1f1f", density=1.1, linewidth=0.6, arrowsize=0.7)
    _mark_center(ax, lat_c, lon_c)
    _outlines(ax, coast_segs, lon[0], lon[-1], lat[0], lat[-1])
    _map_axes(ax, lat, lon, "A  200-hPa divergence, outflow streamlines and heights (dam)")


def _panel_B(ax, fig, res, coast_segs):
    m = res["map"]
    lat, lon = m["lat"], m["lon"]
    lat_c, lon_c = res["center"]["lat"], res["center"]["lon"]
    sb = res["shear_box"]                                # box mean about the input point
    sm = m["shear_mag"]
    pm = _filled(ax, lon, lat, sm, "YlOrRd", 0.0, float(np.nanmax(sm)))
    cb = _cbar(fig, pm, ax, "max-layer shear magnitude, 200–850 hPa (kt)")
    step = max(1, lat.size // 14)
    ax.quiver(lon[::step], lat[::step], m["shear_u"][::step, ::step], m["shear_v"][::step, ::step],
              color="#1f1f1f", scale=12.0 * max(_symlim(m["shear_u"]), _symlim(m["shear_v"])), width=0.003)
    h = sb["box_deg"]
    ax.add_patch(Rectangle((sb["lon_lo"], sb["lat_lo"]), 2 * h, 2 * h, fill=False,
                           ec="#1f1f1f", lw=1.5, ls="--", zorder=8))
    # direction of the box-mean shear vector, drawn from the centre (points downshear)
    mag = max(float(np.hypot(sb["u_ms"], sb["v_ms"])), 1e-9)
    ux, vy = sb["u_ms"] / mag, sb["v_ms"] / mag
    toward = (np.degrees(np.arctan2(sb["u_ms"], sb["v_ms"])) + 360.0) % 360.0
    L = 0.8 * h
    _arrow(ax, lon_c, lat_c, lon_c + ux * L, lat_c + vy * L, "#111111", lw=2.4, zorder=11)
    _mark_center(ax, lat_c, lon_c)
    _box(ax, f"box-mean max shear {sb['top_hPa']:.0f}–{sb['bottom_hPa']:.0f} hPa: {sb['max_kt']:.1f} kt\n"
             f"box {sb['lat_lo']:g}–{sb['lat_hi']:g}°N, {sb['lon_lo']:g}–{sb['lon_hi']:g}°E (±{h:g}°)\n"
             f"arrow: shear toward {toward:.0f}°")
    _outlines(ax, coast_segs, lon[0], lon[-1], lat[0], lat[-1])
    _map_axes(ax, lat, lon, f"B  Max-layer shear, 200–850 hPa (box mean ±{h:g}°)")
    return cb


def _panel_C_tilt(ax, fig, res, coast_segs):
    """Vortex tilt: 850 -> 700 -> 500 -> 300 -> 200 hPa centres joined as one track."""
    t = res["tilt_map"]
    tl, to = t["lat"], t["lon"]
    lat_c, lon_c = res["center"]["lat"], res["center"]["lon"]
    z = t["zeta850"]
    # sequential colours so the whole window is filled (no pale empty band)
    pm = _filled(ax, to, tl, z, "YlOrRd", float(np.nanmin(z)), float(np.nanmax(z)))
    _cbar(fig, pm, ax, "850-hPa vorticity (10⁻⁵ s⁻¹)")

    spd = np.hypot(t["u850"], t["v850"]) / KT
    cw = ax.contour(to, tl, spd, colors="#4d4d4d", linewidths=0.6,
                    levels=matplotlib.ticker.MaxNLocator(6).tick_values(0, float(np.nanmax(spd))))
    ax.clabel(cw, fmt="%d kt", fontsize=6, inline=True)

    # one track through the levels, in order 850 -> 700 -> 500 -> 300 -> 200 hPa
    tilt = res["tilt"]
    ax.plot([lv["lon"] for lv in tilt], [lv["lat"] for lv in tilt],
            color="#2b2b2b", lw=1.5, zorder=7)
    for lv in tilt:
        p = int(lv["hPa"])
        ax.plot(lv["lon"], lv["lat"], marker="o", ms=6.5, mfc=LEVEL_COLOURS.get(p, "white"),
                mec="white", mew=0.9, zorder=8)

    # labels stacked in a column to the west, each joined to its marker by a thin leader
    col_x = lon_c - 2.7
    rows = {200: lat_c + 1.6, 300: lat_c + 0.8, 500: lat_c + 0.0, 700: lat_c - 0.8}
    for lv in tilt[1:]:
        p = int(lv["hPa"])
        yy = rows.get(p, lat_c)
        ax.plot([lv["lon"], col_x + 0.05], [lv["lat"], yy], color="#555555", lw=0.6, ls=":", zorder=7)
        ax.text(col_x, yy, f"{p} hPa", fontsize=8, fontweight="bold", ha="left", va="center",
                color=LEVEL_COLOURS.get(p, "black"), zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LEVEL_COLOURS.get(p, "#999"), lw=0.6))
    ax.plot(lon_c, lat_c, marker="o", ms=8.5, mfc=LEVEL_COLOURS[850], mec="white", mew=1.2, zorder=10)
    ax.text(lon_c + 0.9, lat_c - 0.9, "850 hPa centre", fontsize=8, fontweight="bold",
            color=LEVEL_COLOURS[850], ha="center", va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LEVEL_COLOURS[850], lw=0.6))

    # 850 -> 200 hPa shear arrow from the centre, with the downshear / upshear sides
    sh = res["shear_box"]                                 # max-layer shear, box mean about the centre
    mag = max(float(np.hypot(sh["u_ms"], sh["v_ms"])), 1e-9)
    ux, vy = sh["u_ms"] / mag, sh["v_ms"] / mag
    half = 0.5 * (to[-1] - to[0])
    Lg = 0.32 * half
    tip = (lon_c + ux * Lg, lat_c + vy * Lg)
    tail = (lon_c - ux * 0.55 * Lg, lat_c - vy * 0.55 * Lg)
    _arrow(ax, lon_c, lat_c, tip[0], tip[1], "#111111", lw=2.4, zorder=11)
    _box(ax, f"max-layer shear {sh['top_hPa']:.0f}–{sh['bottom_hPa']:.0f} hPa: {sh['max_kt']:.1f} kt\n"
             f"(box mean, ±{sh['box_deg']:g}° about the centre)",
         loc="upper left")
    _outlines(ax, coast_segs, to[0], to[-1], tl[0], tl[-1])
    _map_axes(ax, tl, to, "C  Vortex tilt: 850-hPa centre, 700 · 500 · 300 · 200 hPa centres")
    ax.set_xlim(to[0], to[-1])
    ax.set_ylim(tl[0], tl[-1])
    # fill the whole panel box: the window is a fixed 2.5-deg grid, so do not lock the aspect
    ax.set_aspect("auto")


def _panel_D_hadley(ax, fig, res):
    h = res["hadley"]
    hl = h["lat"]
    lev = h["lev"]
    z, (om, vs, wv) = _sort_by_pz(lev, h["omega"] * 100.0, h["v"], h["w"])
    vlim = _symlim(om)
    pm = _filled(ax, hl, z, om, "RdBu_r", -vlim, vlim)
    _cbar(fig, pm, ax, "ω (10⁻² Pa s⁻¹)  — blue = ascent")
    lat_c = res["center"]["lat"]
    ax.axvline(lat_c, color="#888877", lw=0.7, ls=":")
    _pressure_axis(ax, lev)
    ax.set_xlim(hl[0], hl[-1])
    ax.set_xlabel("Latitude (°N)", fontsize=8)
    _section_arrows(ax, hl, z, vs, wv)
    lon_c = res["center"]["lon"]
    ax.set_title(f"D  Hadley-type φ–p section at {lon_c:.1f}°E, ±500 km  "
                 f"(arrows: v, w; up = ascent)", fontsize=9, loc="left")


def _shrink_panel(ax, cb, s):
    """Scale a map and its colourbar about their common centre by factor s (after layout)."""
    fig = ax.figure
    fig.canvas.draw()                                    # settle the equal-aspect boxes
    p, c = ax.get_position(), cb.ax.get_position()
    cx, cy = p.x0 + 0.5 * p.width, p.y0 + 0.5 * p.height
    ax.set_position([cx - 0.5 * s * p.width, cy - 0.5 * s * p.height, s * p.width, s * p.height])
    cb.ax.set_position([c.x0, cy - 0.5 * s * p.height, c.width, s * p.height])


def _size_like(ax, ref, fig):
    """Give ax the width and height of ref, keeping its own row and aligning the left edges.

    Equal-aspect axes get their data window widened or narrowed so the box fills the
    target exactly (no distortion); auto-aspect axes simply take the box.
    """
    fig.canvas.draw()
    rr = ref.get_position()                              # ref after its own aspect is applied
    own = ax.get_position(original=True)                 # ax's grid cell (its row)
    W, H = fig.get_size_inches()
    if ax.get_aspect() != "auto":
        ratio = (rr.width * W) / (rr.height * H)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx, cy, sx, sy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0
        if sx / sy < ratio:
            sx = sy * ratio
        else:
            sy = sx / ratio
        ax.set_xlim(cx - 0.5 * sx, cx + 0.5 * sx)
        ax.set_ylim(cy - 0.5 * sy, cy + 0.5 * sy)
    ax.set_position([rr.x0, own.y0, rr.width, rr.height])


def figure1(res, coast_segs=()):
    lat_c, lon_c = res["center"]["lat"], res["center"]["lon"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), facecolor="white")
    fig.suptitle(f"Dynamic Fig — tropical cyclone diagnostics · {res['date']}  ·  "
                 f"{_centre_note(res)}", fontsize=13, y=0.995)
    _panel_A(axs[0, 0], fig, res, coast_segs)
    cbB = _panel_B(axs[0, 1], fig, res, coast_segs)
    _panel_C_tilt(axs[1, 0], fig, res, coast_segs)
    _panel_D_hadley(axs[1, 1], fig, res)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _shrink_panel(axs[0, 1], cbB, 1.0)                   # B slightly smaller
    _size_like(axs[1, 0], axs[0, 0], fig)                # tilt map: same size as A (filled, aspect free)
    _watermark(fig, SRC)
    return _png(fig)


# ---------------------------------------------------------------------------
# ThermoDynamic Fig
# ---------------------------------------------------------------------------

def _tw_panel_A(ax, fig, res):
    tw = res["tangential"]
    vt, yk = tw["vt"], tw["y_km"]
    lev = res["lev"]
    th = res["theta"]["th"]
    ov = res["overturn"]
    z, (th_s, vt_s, v_s, w_s) = _sort_by_pz(lev, th, vt, ov["v"], ov["w"])
    tlim = _symlim(th_s)
    pm = _filled(ax, yk, z, th_s, "RdBu_r", -tlim, tlim)
    _cbar(fig, pm, ax, "θ′ (K)")
    cs = ax.contour(yk, z, vt_s, colors="#111111", linewidths=0.8, levels=np.arange(-40, 41, 5.0))
    ax.clabel(cs, fmt="%d", fontsize=7)
    _pressure_axis(ax, lev)
    _section_axes(ax, yk, "A  Tangential wind (contours, m s⁻¹) and θ′ (shaded), N–S section")
    _section_arrows(ax, yk, z, v_s, w_s)


def _tw_panel_B(ax, fig, res):
    ov = res["overturn"]
    lev = res["lev"]
    z, (uu, v_s, w_s) = _sort_by_pz(lev, ov["u"], ov["v"], ov["w"])
    ulim = _symlim(uu)
    pm = _filled(ax, ov["y_km"], z, uu, "RdBu_r", -ulim, ulim)
    _cbar(fig, pm, ax, "Zonal wind U (m s⁻¹)")
    _pressure_axis(ax, lev)
    _section_axes(ax, ov["y_km"], "B  U (shaded) with secondary circulation (arrows: v, w)")
    _section_arrows(ax, ov["y_km"], z, v_s, w_s)


def _tw_panel_C_hodo(ax, fig, res):
    """Hodograph: zoomed to the data, small markers, labels only at 850, 500 and 200 hPa."""
    hd = res["hodo"]
    hu, hv, hl = hd["u"], hd["v"], hd["lev"]
    z = _pz(hl)
    zmax = float(_pz(min(hl)))
    norm = mcolors.Normalize(vmin=0.0, vmax=zmax)
    cmap = plt.get_cmap("viridis_r")
    order = np.argsort(z)
    mv_dir, mv_kt = res["motion_dir"], res["motion_kt"]
    ms = mv_kt * KT
    mx, my = _bearing_vec(mv_dir)

    # zoom to the data (origin and the motion arrow included)
    xs_ = np.concatenate([hu, [0.0, ms * mx]])
    ys_ = np.concatenate([hv, [0.0, ms * my]])
    x0, x1 = float(xs_.min()), float(xs_.max())
    y0, y1 = float(ys_.min()), float(ys_.max())
    span = max(x1 - x0, y1 - y0, 2.0)
    half = 0.5 * span * 1.2
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    for r in np.arange(5.0, 1.6 * half, 5.0):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=(0, (2, 3)), lw=0.6, ec="#b8b8b8", zorder=1))
    ax.axhline(0, color="#8a8a8a", lw=0.7, zorder=1)
    ax.axvline(0, color="#8a8a8a", lw=0.7, zorder=1)
    ax.plot(hu[order], hv[order], color="#7a7a7a", lw=1.0, zorder=2)
    ax.scatter(hu, hv, c=z, cmap=cmap, norm=norm, s=20, edgecolor="black", lw=0.35, zorder=4)
    off = 0.03 * 2 * half
    for p in (850, 500, 200):
        if min(hl) <= p <= max(hl):
            k = int(np.argmin(np.abs(hl - p)))
            ax.text(hu[k] + off, hv[k] + off, f"{p}", fontsize=8, fontweight="bold", zorder=6)

    _arrow(ax, 0, 0, ms * mx, ms * my, "#c53030", lw=1.6, zorder=6, head_length=0.6, head_width=0.22)

    cb = _cbar(fig, mcm.ScalarMappable(norm=norm, cmap=cmap), ax, "Pressure (hPa)")
    tk = [float(_pz(p)) for p in P_TICKS if min(hl) <= p <= max(hl)]
    cb.locator = matplotlib.ticker.FixedLocator(tk)
    cb.formatter = matplotlib.ticker.FuncFormatter(lambda zz, pos: f"{1000.0 / np.exp(zz):.0f}")
    cb.update_ticks()
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xlabel("u (m s⁻¹)", fontsize=8)
    ax.set_ylabel("v (m s⁻¹)", fontsize=8)
    ax.set_title("C  Hodograph: 0–500 km mean wind, 1000 → 100 hPa", fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(color="#ececec", lw=0.5, zorder=0)


def _tw_panel_D_steer(ax, fig, res, coast_segs):
    """Steering map: storm track, motion and steering arrows, short plain legend."""
    sm = res["steer_map"]
    sl, so = sm["lat"], sm["lon"]
    lat_c, lon_c = res["center"]["lat"], res["center"]["lon"]
    spd = np.hypot(sm["u"], sm["v"]) / KT
    pm = _filled(ax, so, sl, spd, "YlGnBu", 0.0, float(np.nanmax(spd)))
    _cbar(fig, pm, ax, "850–200 hPa mean wind speed (kt)")
    ax.streamplot(so, sl, sm["u"], sm["v"], color="#1f1f1f", density=1.2, linewidth=0.6, arrowsize=0.7)

    tr = sm["track"]
    tla = [p[0] for p in tr]
    tlo = [p[1] for p in tr]
    ax.plot(tlo, tla, color="#c53030", lw=1.6, zorder=6, ls=(0, (1, 1.2)))
    ax.plot(tlo[0], tla[0], marker="o", ms=5, mfc="white", mec="#c53030", mew=1.4, zorder=7)
    ax.plot(tlo[-1], tla[-1], marker="o", ms=5, mfc="#c53030", mec="#c53030", zorder=7)
    ax.text(tlo[0] + 0.4, tla[0] - 0.3, "D−1", fontsize=8, color="#c53030", zorder=8, path_effects=HALO)
    ax.text(tlo[-1] + 0.4, tla[-1] + 0.2, "D+1", fontsize=8, color="#c53030", zorder=8, path_effects=HALO)

    half = 0.5 * (so[-1] - so[0])
    Lg = 0.36 * half
    mv_dir, mv_kt = res["motion_dir"], res["motion_kt"]
    st = res["steer_at_center"]
    st_dir, st_kt = st["bearing_deg"], st["speed_kt"]
    kmax = max(mv_kt, st_kt, 1e-6)
    mx, my = _bearing_vec(mv_dir)
    Lm = Lg * mv_kt / kmax
    _arrow(ax, lon_c, lat_c, lon_c + mx * Lm, lat_c + my * Lm, "#c53030", lw=2.4, zorder=9)
    sx, sy = _bearing_vec(st_dir)
    Ls = Lg * st_kt / kmax
    _arrow(ax, lon_c, lat_c, lon_c + sx * Ls, lat_c + sy * Ls, "#1f4e8c", lw=1.8, dashed=True, zorder=9)

    _mark_center(ax, lat_c, lon_c)
    _box(ax, f"Red arrow: storm motion {mv_dir:03.0f}°, {mv_kt:.1f} kt\n"
             f"Blue dashed arrow: steering flow {st_dir:03.0f}°, {st_kt:.1f} kt\n"
             f"Dotted line: storm track, D−1 to D+1",
         loc="upper left")
    _outlines(ax, coast_segs, so[0], so[-1], sl[0], sl[-1])
    _map_axes(ax, sl, so, "D  Steering map: 850–200 hPa mean wind, storm track and motion")


def figure2(res, coast_segs=()):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), facecolor="white")
    fig.suptitle(f"ThermoDynamic Fig — tropical cyclone structure · {res['date']}  ·  "
                 f"{_centre_note(res)}", fontsize=13, y=0.995)
    _tw_panel_A(axs[0, 0], fig, res)
    _tw_panel_B(axs[0, 1], fig, res)
    _tw_panel_C_hodo(axs[1, 0], fig, res)
    _tw_panel_D_steer(axs[1, 1], fig, res, coast_segs)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    axs[1, 1].set_aspect("auto")                         # steering window is square; fill B's shape
    _size_like(axs[1, 0], axs[0, 0], fig)                # hodograph: same size as A
    _size_like(axs[1, 1], axs[0, 1], fig)                # steering map: same size as B
    _watermark(fig, SRC)
    return _png(fig)
