"""cyclone/plot.py — the two tropical-cyclone figures.

Figure 1 (2 x 2):  A  200-hPa divergence, outflow streamlines, heights
                   B  deep-layer (850-200 hPa) shear magnitude and vectors; max-shear point
                   C  vortex tilt: centres at 850/700/500/300/200 hPa, 850-hPa vorticity,
                      850-hPa wind-speed contours, and the max-shear vector arrow
                   D  Hadley-type phi-p section through the centre
Figure 2 (2 x 2):  A  tangential wind (contours) and theta' (shaded), N-S section
                   B  zonal wind U (shaded) with secondary overturning (v, w)
                   C  hodograph (0-500 km mean), coloured by level, motion arrow
                   D  steering map: 850-200 hPa mean wind streamlines and the track

Colour limits are taken from the data in each panel. Vertical arrows are scaled
for display, and the factor is printed on the panel.
"""
import io

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from cyclone import core

KT = core.KT


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _symlim(arr, floor=1e-12):
    return max(float(np.nanmax(np.abs(arr))), floor)


def _filled(ax, x, y, z, cmap, vmin, vmax):
    """Smooth filled contours (no cell blocks on the 2.5-deg input)."""
    levels = np.linspace(vmin, vmax, 25)
    return ax.contourf(x, y, z, levels=levels, cmap=cmap, extend="both")


def _quiver_scale(u, v):
    """quiver scale so the longest arrow is about 1/12 of the axis width (data-derived)."""
    return 12.0 * max(_symlim(u), _symlim(v))


def _cbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.locator = matplotlib.ticker.MaxNLocator(7)
    cb.update_ticks()
    return cb


def _coast(ax, coast_segs, lon_lo, lon_hi, lat_lo, lat_hi):
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            if part[:, 0].max() < lon_lo or part[:, 0].min() > lon_hi:
                continue
            if part[:, 1].max() < lat_lo or part[:, 1].min() > lat_hi:
                continue
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.7, zorder=6)


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


def _box(ax, x, y, text, loc="upper"):
    yy, va = (0.97, "top") if loc == "upper" else (0.03, "bottom")
    ax.text(0.02, yy, text, transform=ax.transAxes, fontsize=8, va=va,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.9))


def _pressure_axis(ax, lev):
    ax.set_yscale("log")
    ax.set_ylim(max(lev), min(lev))
    ticks = [t for t in (1000, 850, 700, 500, 300, 200, 100, 50, 10) if min(lev) <= t <= max(lev)]
    ax.set_yticks(ticks)
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


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def figure1(res, coast_segs=()):
    m = res["map"]
    lat, lon = m["lat"], m["lon"]
    lat_c, lon_c = res["center"]["lat"], res["center"]["lon"]
    sh = res["shear"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), facecolor="white")
    fig.suptitle(f"Tropical cyclone diagnostics — {res['storm']} · {res['date']}  ·  "
                 f"centre {lat_c:.2f}°N {lon_c:.2f}°E", fontsize=13, y=0.995)
    LO, HI, LAL, LAH = lon[0], lon[-1], lat[0], lat[-1]

    # A: 200-hPa divergence, streamlines, heights
    ax = axs[0, 0]
    dv = m["div200"]
    pm = _filled(ax, lon, lat, dv, "RdBu_r", -_symlim(dv), _symlim(dv))
    _cbar(fig, pm, ax, "200-hPa divergence (10⁻⁵ s⁻¹)")
    cs = ax.contour(lon, lat, m["z200"] / 10.0, colors="#333333", linewidths=0.6, levels=12)
    ax.clabel(cs, fmt="%d", fontsize=6)
    ax.streamplot(lon, lat, m["u200"], m["v200"], color="#1f1f1f", density=1.1, linewidth=0.6, arrowsize=0.7)
    _mark_center(ax, lat_c, lon_c)
    _coast(ax, coast_segs, LO, HI, LAL, LAH)
    _map_axes(ax, lat, lon, "A  200-hPa divergence, outflow streamlines and heights (dam)")

    # B: deep-layer shear magnitude, vectors, max-shear point
    ax = axs[0, 1]
    sm = m["shear_mag"]
    pm = _filled(ax, lon, lat, sm, "YlOrRd", 0.0, float(np.nanmax(sm)))
    _cbar(fig, pm, ax, "850–200 hPa shear magnitude (kt)")
    step = max(1, lat.size // 14)
    ax.quiver(lon[::step], lat[::step], m["shear_u"][::step, ::step], m["shear_v"][::step, ::step],
              color="#1f1f1f", scale=_quiver_scale(m["shear_u"], m["shear_v"]), width=0.003)
    ax.plot(sh["max_lon"], sh["max_lat"], marker="s", ms=8, mfc="none", mec="#1f1f1f", mew=1.5, zorder=8)
    _mark_center(ax, lat_c, lon_c)
    _box(ax, None, None,
         f"max shear {sh['max_kt']:.1f} kt at r = {sh['max_r_km']:.0f} km (square)\n"
         f"shear vector from {sh['dir_from_deg']:.0f}°")
    _coast(ax, coast_segs, LO, HI, LAL, LAH)
    _map_axes(ax, lat, lon, "B  Deep-layer shear 850–200 hPa (pointwise, 0–500 km)")

    # C: vortex tilt, 850-hPa vorticity, max-shear vector arrow
    ax = axs[1, 0]
    t = res["tilt_map"]
    tl, to = t["lat"], t["lon"]
    z = t["zeta850"]
    pm = _filled(ax, to, tl, z, "RdBu_r", -_symlim(z), _symlim(z))
    _cbar(fig, pm, ax, "850-hPa vorticity (10⁻⁵ s⁻¹)")
    spd = np.hypot(t["u850"], t["v850"]) / KT
    cw = ax.contour(to, tl, spd, colors="#5a5a5a", linewidths=0.6, levels=8)
    ax.clabel(cw, fmt="%d", fontsize=6)
    colors = {850: "#111111", 700: "#2b6cb0", 500: "#2f855a", 300: "#b7791f", 200: "#c53030"}
    for lv in res["tilt"]:
        p = int(lv["hPa"])
        ax.plot(lv["lon"], lv["lat"], marker="o", ms=6, mec="black", mfc=colors.get(p, "white"), zorder=8)
        ax.text(lv["lon"] + 0.15, lv["lat"] + 0.15, f"{p}", fontsize=8, zorder=9,
                color=colors.get(p, "black"), fontweight="bold")
    # max-shear vector drawn from the 850-hPa centre; length scaled to a quarter of the window
    half = 0.5 * (to[-1] - to[0])
    mag = max(np.hypot(sh["u_ms"], sh["v_ms"]), 1e-12)
    sc = 0.25 * half / mag
    ax.annotate("", xy=(lon_c + sh["u_ms"] * sc, lat_c + sh["v_ms"] * sc), xytext=(lon_c, lat_c),
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.2), zorder=10)
    ax.plot(sh["max_lon"], sh["max_lat"], marker="s", ms=6, mfc="none", mec="#111111", mew=1.2, zorder=9)
    _box(ax, None, None,
         f"arrow = max shear {sh['max_kt']:.1f} kt (r = {sh['max_r_km']:.0f} km)\n"
         f"arrow scale ×{sc * 1e3:.2f} km per m/s")
    _coast(ax, coast_segs, to[0], to[-1], tl[0], tl[-1])
    _map_axes(ax, tl, to, "C  Vortex tilt: 850 (centre) · 700 · 500 · 300 · 200 hPa centres")
    ax.set_xlim(to[0], to[-1])
    ax.set_ylim(tl[0], tl[-1])

    # D: Hadley-type phi-p section through the centre
    ax = axs[1, 1]
    h = res["hadley"]
    hl, lev = h["lat"], h["lev"]
    om = h["omega"] * 100.0                      # 10^-2 Pa s-1
    vlim = _symlim(om)
    pm = _filled(ax, hl, lev, om, "RdBu_r", -vlim, vlim)
    _cbar(fig, pm, ax, "ω (10⁻² Pa s⁻¹)  — blue = ascent")
    vs = h["v"]
    wsc = _symlim(vs) / max(_symlim(h["w"]), 1e-12) * 0.5
    ax.quiver(hl, lev, vs, -h["w"] * wsc, color="#1f1f1f",
              scale=_quiver_scale(vs, h["w"] * wsc), width=0.003)
    ax.axvline(lat_c, color="#888877", lw=0.7, ls=":")
    _pressure_axis(ax, lev)
    ax.set_xlim(hl[0], hl[-1])
    ax.set_xlabel("Latitude (°N)", fontsize=8)
    ax.set_title(f"D  Hadley-type φ–p section at {lon_c:.1f}°E, ±500 km  "
                 f"(arrows: v, w×{wsc:.0f} display scale)", fontsize=10, loc="left")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _watermark(fig, SRC)
    return _png(fig)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def figure2(res, coast_segs=()):
    lat_c, lon_c = res["center"]["lat"], res["center"]["lon"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), facecolor="white")
    fig.suptitle(f"Tropical cyclone structure — {res['storm']} · {res['date']}  ·  "
                 f"centre {lat_c:.2f}°N {lon_c:.2f}°E", fontsize=13, y=0.995)
    lev = res["lev"]

    # A: tangential wind (contours) and theta' (shaded)
    ax = axs[0, 0]
    tw = res["tangential"]
    vt, yk = tw["vt"], tw["y_km"]
    th = res["theta"]["th"]
    tlim = _symlim(th)
    pm = _filled(ax, yk, lev, th, "RdBu_r", -tlim, tlim)
    _cbar(fig, pm, ax, "θ′ (K)")
    cs = ax.contour(yk, lev, vt, colors="#111111", linewidths=0.8, levels=np.arange(-40, 41, 5.0))
    ax.clabel(cs, fmt="%d", fontsize=7)
    _pressure_axis(ax, lev)
    _section_axes(ax, yk, "A  Tangential wind (contours, m s⁻¹) and θ′ (shaded), N–S section")
    ax.set_ylim(max(lev), min(lev))

    # B: zonal wind with secondary overturning
    ax = axs[0, 1]
    ov = res["overturn"]
    uu = ov["u"]
    ulim = _symlim(uu)
    pm = _filled(ax, ov["y_km"], lev, uu, "RdBu_r", -ulim, ulim)
    _cbar(fig, pm, ax, "Zonal wind U (m s⁻¹)")
    v_s, w_s = ov["v"], ov["w"]
    wsc = _symlim(v_s) / max(_symlim(w_s), 1e-12) * 0.5
    YY, PP = np.meshgrid(ov["y_km"], lev)
    ax.quiver(YY[::2, ::2], PP[::2, ::2], v_s[::2, ::2], -w_s[::2, ::2] * wsc, color="#111111",
              scale=_quiver_scale(v_s[::2, ::2], w_s[::2, ::2] * wsc), width=0.003)
    _pressure_axis(ax, lev)
    _section_axes(ax, ov["y_km"], f"B  U with secondary overturning (v, w×{wsc:.0f} display scale)")
    ax.set_ylim(max(lev), min(lev))

    # C: hodograph
    ax = axs[1, 0]
    hd = res["hodo"]
    hu, hv, hl = hd["u"], hd["v"], hd["lev"]
    order = np.argsort(-hl)
    cmap = plt.get_cmap("viridis")
    ax.plot(hu[order], hv[order], color="#888888", lw=1.0, zorder=2)
    for i, k in enumerate(order):
        ax.scatter(hu[k], hv[k], color=cmap(i / max(order.size - 1, 1)), s=40, zorder=3,
                   edgecolor="black", lw=0.4)
        ax.text(hu[k] + 0.4, hv[k] + 0.4, f"{int(hl[k])}", fontsize=7, zorder=4)
    lim = max(_symlim(hu), _symlim(hv)) * 1.15
    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    mv_dir, mv_kt = res["motion_dir"], res["motion_kt"]
    ms = mv_kt * KT
    ax.annotate("", xy=(ms * np.sin(np.deg2rad(mv_dir)), ms * np.cos(np.deg2rad(mv_dir))), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#c53030", lw=2.0), zorder=5)
    ax.text(0.02, 0.97, f"storm motion {mv_dir:03.0f}° at {mv_kt:.1f} kt (red arrow)",
            transform=ax.transAxes, fontsize=8, va="top", color="#c53030",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("u (m s⁻¹)", fontsize=8)
    ax.set_ylabel("v (m s⁻¹)", fontsize=8)
    ax.set_title("C  Hodograph: 0–500 km mean wind by level (1000 → 150 hPa)", fontsize=10, loc="left")
    ax.tick_params(labelsize=8)

    # D: steering map with streamlines and the track
    ax = axs[1, 1]
    sm = res["steer_map"]
    sl, so = sm["lat"], sm["lon"]
    spd = np.hypot(sm["u"], sm["v"]) / KT
    pm = _filled(ax, so, sl, spd, "YlGnBu", 0.0, float(np.nanmax(spd)))
    _cbar(fig, pm, ax, "850–200 hPa mean wind speed (kt)")
    ax.streamplot(so, sl, sm["u"], sm["v"], color="#1f1f1f", density=1.2, linewidth=0.6, arrowsize=0.7)
    tr = sm["track"]
    tla = [p[0] for p in tr]
    tlo = [p[1] for p in tr]
    ax.plot(tlo, tla, color="#c53030", lw=2.0, zorder=6, marker="o", ms=5)
    ax.text(tlo[0] + 0.15, tla[0] + 0.15, "D−1", fontsize=8, color="#c53030", zorder=7)
    ax.text(tlo[-1] + 0.15, tla[-1] + 0.15, "D+1", fontsize=8, color="#c53030", zorder=7)
    st = res["steer_at_center"]
    _mark_center(ax, lat_c, lon_c)
    _box(ax, None, None,
         f"steering (850–200 hPa mean, {st['r1_km']:.0f}–{st['r2_km']:.0f} km annulus) "
         f"{st['bearing_deg']:03.0f}° at {st['speed_kt']:.1f} kt\n"
         f"storm motion {mv_dir:03.0f}° at {mv_kt:.1f} kt (red track D−1 → D+1)")
    _coast(ax, coast_segs, so[0], so[-1], sl[0], sl[-1])
    _map_axes(ax, sl, so, "D  Steering map: 850–200 hPa mean wind and storm track")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _watermark(fig, SRC)
    return _png(fig)
