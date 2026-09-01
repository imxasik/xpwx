"""custom/chi200.py — 200-hPa velocity-potential anomaly + wind.

This is a drop-in addon: it does not modify any file under ``pro/``.
The diagnostic follows the supplied chi.py calculation:
  1. 200-hPa u/v observations minus 1991-2020 climatology
  2. Gaussian smoothing of wind anomalies
  3. spherical horizontal divergence
  4. FFT Poisson inversion to obtain velocity potential chi
  5. chi-gradient wind arrows

Sidebar group: Upper
Product name: Chi 200 • Wind
"""

import io
import datetime
import re

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pydap.client import open_url

from pro import config

PSL = config.PSL


def _open(varname, year):
    return open_url(f"{PSL}/{varname}.{year}.nc")


def _open_ltm(varname):
    return open_url(f"{PSL}/{varname}.day.ltm.1991-2020.nc")


def _latlon(ds):
    return np.array(ds["lat"][:]), np.array(ds["lon"][:])


def _level_idx(ds, hPa=200):
    lev = np.array(ds["level"][:])
    return int(np.argmin(np.abs(lev - hPa)))


def _time_idx(ds, target):
    raw = np.array(ds["time"][:])
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    scale = 1.0 / 24.0 if "hours" in units else 1.0
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
    epoch = (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
             if m else datetime.date(1800, 1, 1))
    for i, t in enumerate(raw):
        d = epoch + datetime.timedelta(days=float(t) * scale)
        if (d.year, d.month, d.day) == (target.year, target.month, target.day):
            return i
    raise ValueError(f"Date {target} not found")


def _read_slice(ds, varname, t, lv):
    raw = np.array(ds[varname][t, lv, :, :].data).squeeze().astype(np.float64)
    attr = ds[varname].attributes
    sf = float(attr.get("scale_factor", 1.0))
    ao = float(attr.get("add_offset", 0.0))
    mv = float(attr.get("missing_value", 32767.0))
    fill_mask = np.abs(raw - mv) < 0.5
    data = raw * sf + ao
    data[fill_mask] = np.nan
    return data


def _fetch_uv_obs(dates):
    by_year = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)

    lat = lon = None
    u_slices, v_slices = [], []

    for year, ydates in sorted(by_year.items()):
        ds_u = _open("uwnd", year)
        ds_v = _open("vwnd", year)
        if lat is None:
            lat, lon = _latlon(ds_u)

        lv_u = _level_idx(ds_u, 200)
        lv_v = _level_idx(ds_v, 200)
        for d in ydates:
            ti_u = _time_idx(ds_u, d)
            ti_v = _time_idx(ds_v, d)
            u_slices.append(_read_slice(ds_u, "uwnd", ti_u, lv_u))
            v_slices.append(_read_slice(ds_v, "vwnd", ti_v, lv_v))

    return lat, lon, np.nanmean(u_slices, axis=0), np.nanmean(v_slices, axis=0)


def _fetch_uv_ltm(dates):
    ds_u = _open_ltm("uwnd")
    ds_v = _open_ltm("vwnd")
    lat, lon = _latlon(ds_u)
    lv_u = _level_idx(ds_u, 200)
    lv_v = _level_idx(ds_v, 200)
    n_u = len(np.array(ds_u["time"][:]))
    n_v = len(np.array(ds_v["time"][:]))

    u_slices, v_slices = [], []
    for d in dates:
        doy = d.timetuple().tm_yday
        ti_u = min(doy - 1, n_u - 1)
        ti_v = min(doy - 1, n_v - 1)
        u_slices.append(_read_slice(ds_u, "uwnd", ti_u, lv_u))
        v_slices.append(_read_slice(ds_v, "vwnd", ti_v, lv_v))

    return lat, lon, np.nanmean(u_slices, axis=0), np.nanmean(v_slices, axis=0)


def _divergence(u, v, lat, lon):
    R = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)

    dudx = np.gradient(u, lon_r, axis=1) / (R * coslat[:, None])
    vcoslat = v * coslat[:, None]
    dvdy = np.gradient(vcoslat, lat_r, axis=0) / (R * coslat[:, None])
    return dudx + dvdy


def _poisson_fft(rhs, lat, lon):
    R = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dy = R * np.abs(np.mean(np.diff(lat_r)))
    coslat = np.cos(lat_r)
    dx_mean = R * np.mean(np.diff(lon_r)) * np.mean(np.abs(coslat))

    nlat, nlon = rhs.shape
    rhs_clean = np.nan_to_num(rhs, nan=0.0)

    taper = np.ones(nlat)
    for i, la in enumerate(lat):
        if abs(la) > 75.0:
            taper[i] = np.cos(np.deg2rad((abs(la) - 75.0) * 90.0 / 15.0)) ** 2
    rhs_clean *= taper[:, None]

    kx = 2.0 * np.pi * np.fft.fftfreq(nlon, d=dx_mean)
    ky = 2.0 * np.pi * np.fft.fftfreq(nlat, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0

    F = np.fft.fft2(rhs_clean)
    F /= -K2
    F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F))


def _chi_to_wind(chi, lat, lon):
    R = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)

    u = np.gradient(chi, lon_r, axis=1) / (R * coslat[:, None])
    v = np.gradient(chi, lat_r, axis=0) / R
    return u, v


def _compute_chi200(pkg, dates):
    lat, lon, u_obs, v_obs = _fetch_uv_obs(dates)
    _, _, u_ltm, v_ltm = _fetch_uv_ltm(dates)

    u_anom = gaussian_filter(u_obs - u_ltm, sigma=1.5)
    v_anom = gaussian_filter(v_obs - v_ltm, sigma=1.5)

    div = _divergence(u_anom, v_anom, lat, lon)
    chi = gaussian_filter(_poisson_fft(div, lat, lon), sigma=2.0)
    u_div, v_div = _chi_to_wind(chi, lat, lon)

    return lat, lon, {
        "main": chi * pkg.get("plot_scale", 1e-6),
        "u": u_div,
        "v": v_div,
    }


def _render_chi200(lat, lon, data, pkg, coast_segs, dates, out_buf=None, **_kw):
    """Render the Chi 200 • Wind map in the style of the supplied chi.py."""
    LON2D, LAT2D = np.meshgrid(lon, lat)
    chi_plot = np.asarray(data["main"], dtype=float)

    fig = plt.figure(figsize=(13, 8), facecolor="white")
    ax = fig.add_axes([0.055, 0.07, 0.855, 0.88])
    ax.set_facecolor("#eef3f9")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(-90, 90)

    vlim = pkg.get("vlim", 10.0)
    levels = np.linspace(-vlim, vlim, 97)
    cf = ax.contourf(LON2D, LAT2D, chi_plot, levels=levels,
                     cmap="RdBu_r", extend="both", zorder=1)

    pos_lev = np.arange(pkg.get("cint", 2.5), vlim + 0.001, pkg.get("cint", 2.5))
    ax.contour(LON2D, LAT2D, chi_plot, levels=pos_lev,
               colors="black", linewidths=0.8, zorder=2)
    ax.contour(LON2D, LAT2D, chi_plot, levels=-pos_lev[::-1],
               colors="black", linewidths=0.8, linestyles="dashed", zorder=2)

    u_div = np.asarray(data["u"], dtype=float)
    v_div = np.asarray(data["v"], dtype=float)
    step = int(pkg.get("vec_step", 6))
    qs = slice(None, None, step)
    Xq = LON2D[qs, qs]
    Yq = LAT2D[qs, qs]
    Uq = u_div[qs, qs]
    Vq = v_div[qs, qs]
    mag = np.sqrt(Uq**2 + Vq**2)
    mask = (mag > 1e-10) & np.isfinite(mag)

    ax.quiver(
        Xq[mask], Yq[mask],
        Uq[mask] / mag[mask], Vq[mask] / mag[mask],
        color="black", scale=5, scale_units="inches",
        width=0.0028, headwidth=7, headlength=8, headaxislength=6,
        minshaft=1.5, pivot="middle", zorder=6,
    )

    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#1a1a1a", lw=0.9, zorder=7)

    for x in range(0, 360, 60):
        ax.axvline(x, color="#999999", lw=0.4, ls="--", zorder=0)
    for y in range(-80, 81, 20):
        ax.axhline(y, color="#999999", lw=0.4, ls="--", zorder=0)
    ax.axhline(0, color="#444444", lw=0.9, zorder=0)

    def xlab(v):
        if v in (0, 360):
            return "0°"
        if v == 180:
            return "180°"
        return f"{v}°E" if v < 180 else f"{360 - v}°W"

    def ylab(v):
        return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

    xticks = list(range(0, 360, 60))
    yticks = list(range(-80, 81, 20))
    ax.set_xticks(xticks)
    ax.set_xticklabels([xlab(x) for x in xticks], fontsize=12, color="#222222")
    ax.set_yticks(yticks)
    ax.set_yticklabels([ylab(y) for y in yticks], fontsize=12, color="#222222")
    ax.tick_params(axis="both", length=4, color="#888888", width=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.7)

    cax = fig.add_axes([0.915, 0.11, 0.016, 0.80])
    cbar = plt.colorbar(cf, cax=cax, ticks=np.arange(-vlim, vlim + 0.001, pkg.get("cint", 2.5)))
    cbar.set_label(pkg.get("cb_label", "Velocity-Potential Anomaly (1e6 m²s)"),
                   fontsize=14, color="#111111", labelpad=0)
    cbar.ax.tick_params(labelsize=13, colors="#222222", length=3)
    cbar.outline.set_edgecolor("#aaaaaa")
    cbar.outline.set_linewidth(0.7)

    date_str = (f"{dates[0]:%d %b} – {dates[-1]:%d %b %Y}"
                if len(dates) > 1 else f"{dates[0]:%d %b %Y}")
    ax.set_title(
        f"Chi 200 • Wind  ·  {date_str}  ({len(dates)}-day mean)\n"
        "NCEP/NCAR Reanalysis  ·  Shading: Velocity Potential  |   Arrows: Wind Direction",
        fontsize=15, fontweight="bold", color="#111111", pad=6, loc="center",
    )

    ax.text(0.840, 0.015, "@XPWEATHER", transform=ax.transAxes,
            fontsize=16, va="bottom", color="#111111",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbbbbb",
                      alpha=1.0, lw=1.0), zorder=8)

    if out_buf is None:
        out_buf = io.BytesIO()
    plt.savefig(out_buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out_buf.seek(0)
    return out_buf


PRODUCTS = {
    "chi200_wind": {
        "id": "chi200_wind",
        "title": "Chi 200 • Wind",
        "name": "Chi 200 • Wind",
        "tag": "Upper",
        "desc": "200-hPa velocity-potential anomaly with divergent-wind direction arrows.",
        "kind": "chi200",
        "level": 200,
        "variables": ["uwnd", "vwnd"],
        "show_wind": True,
        "plot_scale": 1e-6,
        "vlim": 10.0,
        "cint": 2.5,
        "vec_step": 6,
        "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)",
    },
}


KINDS = {
    "chi200": {
        "compute": _compute_chi200,
        "render": _render_chi200,
        "tag": "Upper",
        "title": "Chi 200 • Wind",
    },
}
