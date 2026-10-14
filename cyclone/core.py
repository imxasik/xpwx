"""cyclone/core.py — pure numerical diagnostics for tropical cyclones (no I/O).

Grid convention: fields are (nlat, nlon) or (nlev, nlat, nlon), lat and lon
ascending, pressure levels in hPa (ascending after sorting). Winds in m/s.

Only physical and unit constants are declared here. Every analysis value comes
from the data.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline

# physical and unit constants (not analysis parameters)
G = 9.80665          # m s-2
R_D = 287.04         # J kg-1 K-1, gas constant of dry air
CP = 1004.0          # J kg-1 K-1
KAPPA = R_D / CP
EARTH_R = 6.371e6    # m
KT = 0.5144444       # m s-1 per knot
DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# spherical operators
# ---------------------------------------------------------------------------

def _safe_cos(lat):
    """cos(lat), with the polar rows (cos < 0.1) set to NaN.

    The 1/cos(lat) factor of the spherical operators diverges at the pole.
    """
    c = np.cos(np.deg2rad(lat))
    return np.where(c < 0.1, np.nan, c)


def vorticity(u, v, lat, lon):
    """Relative vorticity (s-1) of a (nlat, nlon) or (nlev, nlat, nlon) wind."""
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cos = _safe_cos(lat)
    cb = cos[None, :, None] if u.ndim == 3 else cos[:, None]
    dv_dlam = np.gradient(v, lam, axis=-1)
    du_cos_dphi = np.gradient(u * cb, phi, axis=-2)
    return (dv_dlam - du_cos_dphi) / (EARTH_R * cb)


def divergence(u, v, lat, lon):
    """Horizontal divergence (s-1) of a (nlat, nlon) or (nlev, nlat, nlon) wind."""
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cos = _safe_cos(lat)
    cb = cos[None, :, None] if u.ndim == 3 else cos[:, None]
    du_dlam = np.gradient(u, lam, axis=-1)
    dvc_dphi = np.gradient(v * cb, phi, axis=-2)
    return (du_dlam + dvc_dphi) / (EARTH_R * cb)


def omega_from_divergence(u3d, v3d, lev, lat, lon):
    """Vertical pressure velocity ω (Pa s-1) from mass continuity.

    Continuity in pressure coordinates gives ∂ω/∂p = -∇·V. With ω = 0 at the
    top level (the first level of ``lev``, ascending hPa), the trapezoid rule
    integrates downward level by level. The surface value is not constrained,
    so the result is accurate away from the lowest level.

    Returns (nlev, nlat, nlon). Levels with no divergence data give NaN.
    """
    lev = np.asarray(lev, dtype=float)
    div = divergence(u3d, v3d, lat, lon)                 # s-1
    dp = np.diff(lev * 100.0)                            # Pa, positive downward
    layer = -0.5 * (div[1:] + div[:-1]) * dp[:, None, None]
    omega = np.zeros(div.shape, dtype=float)
    omega[1:] = np.cumsum(layer, axis=0)
    return omega


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def make_sampler(field2d, lat, lon):
    """Cubic-spline sampler for a (nlat, nlon) field. Longitudes wrap at 360.

    Bilinear interpolation is not used: its maximum always lies on a grid corner,
    so it cannot place a vorticity or wind peak between grid points. The spline is
    built over the contiguous block of rows where the field is finite (the polar
    rows are NaN after vorticity/divergence). Points outside that block return NaN.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    good = np.where(np.all(np.isfinite(field2d), axis=1))[0]
    if good.size < 4:
        raise ValueError("too few finite rows to build a sampler")
    r0, r1 = good[0], good[-1] + 1
    la_c = lat[r0:r1]
    fld = np.asarray(field2d[r0:r1], dtype=float)
    fld = np.where(np.isfinite(fld), fld, 0.0)
    lonp = np.concatenate([lon, [lon[0] + 360.0]])
    fp = np.concatenate([fld, fld[:, :1]], axis=1)
    spl = RectBivariateSpline(la_c, lonp, fp, kx=3, ky=3, s=0)

    def f(la, lo):
        la = np.asarray(la, dtype=float)
        lo = np.mod(np.asarray(lo, dtype=float) - lon[0], 360.0) + lon[0]
        out = spl.ev(la.ravel(), lo.ravel()).reshape(la.shape)
        inside = (la >= la_c[0]) & (la <= la_c[-1])
        return np.where(inside, out, np.nan)

    return f


def km_to_latlon(lat0, lon0, x_km, y_km):
    """Offsets (east x, north y, km) from (lat0, lon0) to (lat, lon)."""
    lat = lat0 + np.asarray(y_km, dtype=float) * 1000.0 / EARTH_R * DEG
    lon = lon0 + np.asarray(x_km, dtype=float) * 1000.0 / (EARTH_R * np.cos(np.deg2rad(lat0))) * DEG
    return lat, lon


def latlon_to_km(lat0, lon0, lat, lon):
    """(x east, y north) in km from (lat0, lon0) to (lat, lon)."""
    x = (np.asarray(lon, dtype=float) - lon0) / DEG * EARTH_R * np.cos(np.deg2rad(lat0)) / 1000.0
    y = (np.asarray(lat, dtype=float) - lat0) / DEG * EARTH_R / 1000.0
    return x, y


def polar_points(lat0, lon0, rmax_km, dr_km=25.0, dth_deg=10.0):
    """Polar sample grid around a centre, shape (nr, nth).

    Returns R (km), TH (deg, clockwise from north), x, y (km), lat, lon.
    """
    r = np.arange(0.0, rmax_km + dr_km / 2, dr_km)
    th = np.arange(0.0, 360.0, dth_deg)
    R, TH = np.meshgrid(r, th, indexing="ij")
    x = R * np.sin(np.deg2rad(TH))
    y = R * np.cos(np.deg2rad(TH))
    lat, lon = km_to_latlon(lat0, lon0, x, y)
    return R, TH, x, y, lat, lon


def _annulus_weights(R, r1, r2):
    """Area weights (proportional to r) inside r1 <= R <= r2, zero outside."""
    m = (R >= r1 - 1e-9) & (R <= r2 + 1e-9)
    return np.where(m, R, 0.0)


def annulus_mean(values, R, r1=0.0, r2=500.0):
    """Area-weighted mean over r1..r2 of a polar-grid array (nr, nth)."""
    w = _annulus_weights(R, r1, r2) * np.isfinite(values)
    v = np.where(np.isfinite(values), values, 0.0)
    return float(np.sum(v * w) / np.sum(w))


def sample_levels(field3d, lat, lon, pla, plo):
    """Sample every level of a (nlev, nlat, nlon) field at points (pla, plo)."""
    out = np.empty((field3d.shape[0],) + np.shape(pla))
    for k in range(field3d.shape[0]):
        out[k] = make_sampler(field3d[k], lat, lon)(pla, plo)
    return out


def level_index(lev, hPa):
    """Index of pressure level hPa (must be in the file)."""
    lev = np.asarray(lev, dtype=float)
    k = int(np.argmin(np.abs(lev - hPa)))
    if abs(lev[k] - hPa) > 1e-6:
        raise ValueError(f"level {hPa} hPa not in the file ({lev.tolist()})")
    return k


# ---------------------------------------------------------------------------
# centre, tilt
# ---------------------------------------------------------------------------

def find_center(zeta2d, lat, lon, lat0, lon0, box_deg=2.5, step_deg=0.25):
    """Cyclonic vorticity maximum within box_deg of (lat0, lon0).

    Searched on a step_deg grid of the cubic-spline sampler. Returns
    (lat_c, lon_c, at_edge); at_edge flags a maximum on the search-box boundary.
    """
    la = np.arange(lat0 - box_deg, lat0 + box_deg + step_deg / 2, step_deg)
    lo = np.arange(lon0 - box_deg, lon0 + box_deg + step_deg / 2, step_deg)
    la = la[(la >= lat.min()) & (la <= lat.max())]
    lo = lo[(lo >= lon.min()) & (lo <= lon.max())]
    if la.size == 0 or lo.size == 0:
        raise ValueError("search box is outside the grid")
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    vals = make_sampler(zeta2d, lat, lon)(LA, LO)
    i, j = np.unravel_index(np.nanargmax(vals), vals.shape)
    at_edge = i in (0, vals.shape[0] - 1) or j in (0, vals.shape[1] - 1)
    return float(LA[i, j]), float(LO[i, j]), bool(at_edge)


def tilt(zeta3d, lev, lat, lon, lat_c, lon_c, levels_hPa, box_deg=3.0):
    """Cyclonic-vorticity centre at each listed level, searched near (lat_c, lon_c)."""
    out = []
    for p in levels_hPa:
        k = level_index(lev, p)
        la, lo, e = find_center(zeta3d[k], lat, lon, lat_c, lon_c, box_deg=box_deg)
        out.append({"hPa": float(p), "lat": la, "lon": lo, "at_edge": e})
    return out


# ---------------------------------------------------------------------------
# shear
# ---------------------------------------------------------------------------

def max_layer_shear_field(u3d, v3d, lev, top_hPa=200.0, bottom_hPa=850.0):
    """Pointwise max-layer shear on the grid.

    For every grid point, every pair of levels between top_hPa and bottom_hPa is tried;
    the layer shear is V_top - V_bottom. The pair with the largest magnitude is kept.
    Returns (du, dv) in m/s, each (nlat, nlon).
    """
    lev = np.asarray(lev, dtype=float)
    sel = np.where((lev >= top_hPa - 1e-6) & (lev <= bottom_hPa + 1e-6))[0]
    best_m2 = bu = bv = None
    for i in sel:                                    # i: bottom (higher pressure)
        for j in sel:                                # j: top (lower pressure)
            if lev[j] >= lev[i]:
                continue
            du = u3d[j] - u3d[i]
            dv = v3d[j] - v3d[i]
            m2 = du * du + dv * dv
            if best_m2 is None:
                best_m2, bu, bv = m2, du, dv
            else:
                better = m2 > best_m2                # NaN compares False: keeps the earlier pair
                best_m2 = np.where(better, m2, best_m2)
                bu = np.where(better, du, bu)
                bv = np.where(better, dv, bv)
    return bu, bv


def box_mean_shear(u3d, v3d, lev, lat, lon, lat_c, lon_c, half_deg=2.5,
                   top_hPa=200.0, bottom_hPa=850.0, step_deg=0.25):
    """Maximum layer shear from box-mean winds in a square box centred on (lat_c, lon_c).

    The box is exactly +/- half_deg about the centre. The winds at every level between
    top_hPa and bottom_hPa are averaged over points spaced step_deg inside that box (not
    snapped to the grid). For every pair of levels inside the window the layer shear is
    mean(V_top) - mean(V_bottom). The layer with the largest magnitude is the max shear.
    """
    lev = np.asarray(lev, dtype=float)
    n = int(round(2.0 * half_deg / step_deg)) + 1
    off = np.linspace(-half_deg, half_deg, n)
    LA, LO = np.meshgrid(lat_c + off, lon_c + off, indexing="ij")
    sel = np.where((lev >= top_hPa - 1e-6) & (lev <= bottom_hPa + 1e-6))[0]
    ub, vb = {}, {}
    for k in sel:
        ub[k] = float(np.nanmean(make_sampler(u3d[k], lat, lon)(LA, LO)))
        vb[k] = float(np.nanmean(make_sampler(v3d[k], lat, lon)(LA, LO)))
    best = None
    for i in sel:                                   # i: bottom (higher pressure)
        for j in sel:                               # j: top (lower pressure)
            if lev[j] >= lev[i]:
                continue
            du, dv = ub[j] - ub[i], vb[j] - vb[i]
            mag = float(np.hypot(du, dv))
            if best is None or mag > best["mag"]:
                best = {"mag": mag, "u_ms": du, "v_ms": dv,
                        "top_hPa": float(lev[j]), "bottom_hPa": float(lev[i])}
    return {
        "u_ms": best["u_ms"], "v_ms": best["v_ms"],
        "max_kt": best["mag"] / KT,
        "top_hPa": best["top_hPa"], "bottom_hPa": best["bottom_hPa"],
        "dir_from_deg": float((np.degrees(np.arctan2(-best["u_ms"], -best["v_ms"])) + 360.0) % 360.0),
        "box_deg": float(half_deg),
        "lat_lo": float(lat_c - half_deg), "lat_hi": float(lat_c + half_deg),
        "lon_lo": float(lon_c - half_deg), "lon_hi": float(lon_c + half_deg),
    }


# ---------------------------------------------------------------------------
# radial profiles
# ---------------------------------------------------------------------------

def ring_mean_profile(field3d, lat, lon, lat_c, lon_c, r1=0.0, r2=500.0):
    """Area-weighted annulus mean of each level of a (nlev, nlat, nlon) field. Returns (nlev,)."""
    R, TH, x, y, pla, plo = polar_points(lat_c, lon_c, r2)
    vals = sample_levels(field3d, lat, lon, pla, plo)            # (nlev, nr, nth)
    w = _annulus_weights(R, r1, r2)[None] * np.isfinite(vals)
    v = np.where(np.isfinite(vals), vals, 0.0)
    return np.sum(v * w, axis=(1, 2)) / np.sum(w, axis=(1, 2))


def hodograph(u3d, v3d, lev, lat, lon, lat_c, lon_c, rmax_km=500.0):
    """Vortex-averaged (0..rmax) u and v at every level, in m/s."""
    return {"lev": np.asarray(lev, float),
            "u": ring_mean_profile(u3d, lat, lon, lat_c, lon_c, 0.0, rmax_km),
            "v": ring_mean_profile(v3d, lat, lon, lat_c, lon_c, 0.0, rmax_km)}


# ---------------------------------------------------------------------------
# motion and steering
# ---------------------------------------------------------------------------

def storm_motion(c_before, c_after, days=2.0):
    """Storm motion between two centres (lat, lon): speed (kt) and bearing (deg, toward)."""
    dx, dy = latlon_to_km(c_before[0], c_before[1], c_after[0], c_after[1])
    dx, dy = float(dx), float(dy)
    speed_ms = np.hypot(dx, dy) * 1000.0 / (days * 86400.0)
    bearing = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    return {"speed_kt": float(speed_ms / KT), "bearing_deg": float(bearing),
            "dx_km": dx, "dy_km": dy}


def layer_mean(field3d, lev, lo_hPa=850.0, hi_hPa=200.0):
    """Pressure-weighted (trapezoid) mean of (nlev, nlat, nlon) over [hi, lo] hPa."""
    lev = np.asarray(lev, float)
    sel = (lev >= hi_hPa - 1e-6) & (lev <= lo_hPa + 1e-6)
    p = lev[sel]
    f = field3d[sel]
    if p.size < 2:
        raise ValueError("layer needs at least two levels")
    w = np.abs(np.gradient(p))
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.tensordot(w, f, axes=(0, 0)) / w.sum()


# ---------------------------------------------------------------------------
# sections through the centre
# ---------------------------------------------------------------------------

def ns_section(field3d, lat, lon, lat_c, lon_c, half_km=600.0, dy_km=25.0):
    """Values along the north-south line through the centre. Returns (nlev, ny), y_km."""
    y = np.arange(-half_km, half_km + dy_km / 2, dy_km)
    x = np.zeros_like(y)
    pla, plo = km_to_latlon(lat_c, lon_c, x, y)
    out = np.empty((field3d.shape[0], y.size))
    for k in range(field3d.shape[0]):
        out[k] = make_sampler(field3d[k], lat, lon)(pla, plo)
    return out, y


def tangential_wind(u3d, v3d, lat, lon, lat_c, lon_c, half_km=600.0, dy_km=25.0):
    """Tangential wind (m/s, cyclonic positive) on the N-S section, and y (km).

    On the line x = 0, V_t = (v x - u y) / r reduces to -u sign(y).
    """
    u_s, y = ns_section(u3d, lat, lon, lat_c, lon_c, half_km, dy_km)
    vt = -u_s * np.sign(y)[None, :]
    return vt, y


def theta_anomaly(T3d, lev, lat, lon, lat_c, lon_c, half_km=600.0, dy_km=25.0,
                  env_r1=500.0, env_r2=800.0):
    """Potential-temperature anomaly theta' (K) on the N-S section.

    theta = T (1000/p)^kappa; theta' = theta minus the mean on the annulus
    env_r1..env_r2 at each level.
    """
    lev = np.asarray(lev, float)
    th3d = T3d * (1000.0 / lev)[:, None, None] ** KAPPA
    sec, y = ns_section(th3d, lat, lon, lat_c, lon_c, half_km, dy_km)
    env = ring_mean_profile(th3d, lat, lon, lat_c, lon_c, env_r1, env_r2)
    return sec - env[:, None], y


def overturning(v3d, om3d, T3d, lev, lat, lon, lat_c, lon_c, half_km=600.0, dy_km=25.0):
    """Secondary circulation on the N-S section.

    Returns (v, w, y_km): v in m/s and w in m/s, with w = -omega R_d T / (p g).
    """
    lev = np.asarray(lev, float)
    v_s, y = ns_section(v3d, lat, lon, lat_c, lon_c, half_km, dy_km)
    om_s, _ = ns_section(om3d, lat, lon, lat_c, lon_c, half_km, dy_km)
    T_s, _ = ns_section(T3d, lat, lon, lat_c, lon_c, half_km, dy_km)
    p = (lev * 100.0)[:, None]
    w_s = -om_s * R_D * T_s / (p * G)
    return v_s, w_s, y


def meridional_band(field3d, lat, lon, lon_c, rmax_km=500.0):
    """Mean over the longitudes within rmax_km of lon_c at each latitude. Returns (nlev, nlat)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dlon = (lon - lon_c + 180.0) % 360.0 - 180.0
    x = dlon[None, :] / DEG * EARTH_R * np.cos(np.deg2rad(lat))[:, None] / 1000.0
    m = np.abs(x) <= rmax_km + 1e-9                               # (nlat, nlon)
    f = field3d
    w = m[None].astype(float) * np.isfinite(f)
    v = np.where(np.isfinite(f), f, 0.0)
    return np.sum(v * w, axis=2) / np.sum(w, axis=2)
