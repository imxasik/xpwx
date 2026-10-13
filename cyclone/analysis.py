"""cyclone/analysis.py — runs every diagnostic for one location and date.

The user gives a nominal centre (lat, lon) and the date D. Centre: the 850-hPa
cyclonic vorticity maximum within CENTER_BOX_DEG of the nominal centre on D. The
centres on D-1 and D+1 are searched within TRACK_BOX_DEG of the D centre. Motion
is the displacement between them over two days.

Shear arrow: the 850->200 hPa shear vector at the point of largest magnitude
inside SHEAR_RMAX_KM of the centre (core.deep_shear).

Search boxes, radii and the annulus widths are analysis parameters declared
below. Physical constants live in core.py.
"""
import datetime

import numpy as np

from cyclone import core
from cyclone import fields as F

CENTER_BOX_DEG = 2.5     # search box for the 850-hPa centre on the storm date
TRACK_BOX_DEG = 5.0      # search box for the centres on D-1 and D+1
TILT_BOX_DEG = 3.0       # search box for the centre at each upper level
TILT_LEVELS = [700.0, 500.0, 300.0, 200.0]     # with 850 (the centre) they form the tilt
SHEAR_RMAX_KM = 500.0    # radius inside which the max shear is searched
STEER_R1_KM = 500.0      # steering = layer-mean wind averaged over this annulus
STEER_R2_KM = 1000.0     # (keeps the vortex core out)
HADLEY_BAND_KM = 500.0   # longitudes within this distance of the centre form the φ-p section
MAP_HALF_DEG = 15.0      # half width of the regional maps
TILT_MAP_HALF_DEG = 4.0  # half width of the tilt map


def _window(lat, lon, la_rng, lo_rng):
    ia = np.where((lat >= la_rng[0]) & (lat <= la_rng[1]))[0]
    io = np.where((lon >= lo_rng[0]) & (lon <= lo_rng[1]))[0]
    return ia, io


def analyze(lat0, lon0, D):
    """Diagnostics for the centre nearest (lat0, lon0) on date D (datetime.date)."""
    lat0, lon0 = float(lat0), float(lon0)
    dates = {"prev": D - datetime.timedelta(days=1), "date": D,
             "next": D + datetime.timedelta(days=1)}
    day = {k: F.read_day(d) for k, d in dates.items()}
    g = day["date"]
    lev, lat, lon = g["lev"], g["lat"], g["lon"]
    k850 = core.level_index(lev, 850.0)
    k200 = core.level_index(lev, 200.0)

    # --- storm centres -----------------------------------------------------
    zeta850 = {k: core.vorticity(d["u"][k850], d["v"][k850], lat, lon) for k, d in day.items()}
    la_c, lo_c, edge = core.find_center(zeta850["date"], lat, lon,
                                        lat0, lon0,
                                        box_deg=CENTER_BOX_DEG)
    centers = {"date": (la_c, lo_c)}
    track_edge = {}
    for k in ("prev", "next"):
        a, b, e = core.find_center(zeta850[k], lat, lon, la_c, lo_c, box_deg=TRACK_BOX_DEG)
        centers[k] = (a, b)
        track_edge[k] = e
    motion = core.storm_motion(centers["prev"], centers["next"], days=2.0)
    motion["track_at_edge"] = track_edge

    # --- tilt: 850-hPa centre plus the upper-level centres ------------------
    zeta3d = core.vorticity(g["u"], g["v"], lat, lon)
    tilt = [{"hPa": 850.0, "lat": la_c, "lon": lo_c, "at_edge": edge}]
    tilt += core.tilt(zeta3d, lev, lat, lon, la_c, lo_c, TILT_LEVELS, box_deg=TILT_BOX_DEG)

    # --- deep-layer shear: max vector inside SHEAR_RMAX_KM -------------------
    shear = core.deep_shear(g["u"], g["v"], lev, lat, lon, la_c, lo_c, rmax_km=SHEAR_RMAX_KM)

    # --- sections through the centre --------------------------------------
    vt, y_tan = core.tangential_wind(g["u"], g["v"], lat, lon, la_c, lo_c)
    th_p, y_th = core.theta_anomaly(g["T"], lev, lat, lon, la_c, lo_c)
    v_s, w_s, y_ov = core.overturning(g["v"], g["w"], g["T"], lev, lat, lon, la_c, lo_c)
    u_s, _ = core.ns_section(g["u"], lat, lon, la_c, lo_c)

    # --- hodograph and steering --------------------------------------------
    hodo = core.hodograph(g["u"], g["v"], lev, lat, lon, la_c, lo_c, rmax_km=500.0)
    u_layer = core.layer_mean(g["u"], lev, 850.0, 200.0)
    v_layer = core.layer_mean(g["v"], lev, 850.0, 200.0)
    Rs, _th, _x, _y, pla, plo = core.polar_points(la_c, lo_c, STEER_R2_KM)
    su = core.annulus_mean(core.make_sampler(u_layer, lat, lon)(pla, plo), Rs, STEER_R1_KM, STEER_R2_KM)
    sv = core.annulus_mean(core.make_sampler(v_layer, lat, lon)(pla, plo), Rs, STEER_R1_KM, STEER_R2_KM)
    steer_at_center = {
        "u_ms": su, "v_ms": sv, "speed_kt": float(np.hypot(su, sv) / core.KT),
        "bearing_deg": float((np.degrees(np.arctan2(su, sv)) + 360.0) % 360.0),
        "r1_km": STEER_R1_KM, "r2_km": STEER_R2_KM,
    }

    res = {
        "date": D.isoformat(), "nominal_center": (lat0, lon0),
        "center": {"lat": la_c, "lon": lo_c, "at_edge": edge},
        "centers": centers, "motion": motion,
        "motion_kt": motion["speed_kt"], "motion_dir": motion["bearing_deg"],
        "tilt": tilt, "shear": shear,
        "tangential": {"vt": vt, "y_km": y_tan},
        "theta": {"th": th_p, "y_km": y_th},
        "overturn": {"v": v_s, "w": w_s, "y_km": y_ov, "u": u_s},
        "hodo": hodo, "lev": lev, "lat": lat, "lon": lon,
        "steer": {"u": u_layer, "v": v_layer},
        "steer_at_center": steer_at_center,
    }

    # --- regional map: 200-hPa divergence, heights, winds, shear -----------
    ia, io = _window(lat, lon, (la_c - MAP_HALF_DEG, la_c + MAP_HALF_DEG),
                     (lo_c - MAP_HALF_DEG, lo_c + MAP_HALF_DEG))
    u200, v200 = g["u"][k200], g["v"][k200]
    div200 = core.divergence(u200, v200, lat, lon)
    du_map = u200 - g["u"][k850]
    dv_map = v200 - g["v"][k850]
    res["map"] = {
        "lat": lat[ia], "lon": lon[io],
        "div200": div200[np.ix_(ia, io)] * 1e5,                # 1e-5 s-1
        "z200": g["z"][k200][np.ix_(ia, io)],
        "u200": u200[np.ix_(ia, io)], "v200": v200[np.ix_(ia, io)],
        "shear_mag": np.hypot(du_map, dv_map)[np.ix_(ia, io)] / core.KT,
        "shear_u": du_map[np.ix_(ia, io)], "shear_v": dv_map[np.ix_(ia, io)],
    }

    # --- tilt map: 850-hPa vorticity and wind ------------------------------
    ib, jo = _window(lat, lon, (la_c - TILT_MAP_HALF_DEG, la_c + TILT_MAP_HALF_DEG),
                     (lo_c - TILT_MAP_HALF_DEG, lo_c + TILT_MAP_HALF_DEG))
    res["tilt_map"] = {
        "lat": lat[ib], "lon": lon[jo],
        "zeta850": zeta850["date"][np.ix_(ib, jo)] * 1e5,
        "u850": g["u"][k850][np.ix_(ib, jo)], "v850": g["v"][k850][np.ix_(ib, jo)],
    }

    # --- steering map and track --------------------------------------------
    ic, jc = _window(lat, lon, (la_c - MAP_HALF_DEG, la_c + MAP_HALF_DEG),
                     (lo_c - MAP_HALF_DEG, lo_c + MAP_HALF_DEG))
    res["steer_map"] = {
        "lat": lat[ic], "lon": lon[jc],
        "u": u_layer[np.ix_(ic, jc)], "v": v_layer[np.ix_(ic, jc)],
        "track": [centers["prev"], centers["date"], centers["next"]],
    }

    # --- Hadley-type phi-p section (longitudes within HADLEY_BAND_KM) -------
    im = np.where((lat >= la_c - MAP_HALF_DEG) & (lat <= la_c + MAP_HALF_DEG))[0]
    vb = core.meridional_band(g["v"], lat, lon, lo_c, HADLEY_BAND_KM)[:, im]
    ob = core.meridional_band(g["w"], lat, lon, lo_c, HADLEY_BAND_KM)[:, im]
    Tb = core.meridional_band(g["T"], lat, lon, lo_c, HADLEY_BAND_KM)[:, im]
    p_pa = (lev * 100.0)[:, None]
    res["hadley"] = {"lat": lat[im], "lev": lev, "omega": ob, "v": vb,
                     "w": -ob * core.R_D * Tb / (p_pa * core.G)}
    return res
