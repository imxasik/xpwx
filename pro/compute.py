"""pro.compute — per-product computation (obs minus climatology anomalies)."""
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter
from . import config, data, physics

AAM_LEVELS = config.AAM_LEVELS
DEG_PER_S = config.DEG_PER_S
RD = config.RD
GRAV = config.GRAV
CP = config.CP
LV = config.LV

# bring data & physics function names into this module's scope
from . import data as _data, physics as _physics  # noqa: E402
globals().update({k: v for k, v in vars(_data).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_physics).items() if not k.startswith("__")})

def streamfunction_from_uv(u_anom, v_anom, lat, lon):
    """Streamfunction psi (m^2/s) from relative-vorticity inversion."""
    zeta = vorticity(u_anom, v_anom, lat, lon)
    return gaussian_filter(poisson_fft(zeta, lat, lon), sigma=2.0)

def _anom(var, level, dates, lat, lon):
    obs = _mean_field(var, level, dates, "obs")
    clim = _mean_field(var, level, dates, "clim")
    return gaussian_filter(obs - clim, sigma=1.5)

def _psi_level(level, dates):
    lat, lon = _latlon("uwnd")
    u = _mean_field("uwnd", level, dates, "obs")
    uc = _mean_field("uwnd", level, dates, "clim")
    v = _mean_field("vwnd", level, dates, "obs")
    vc = _mean_field("vwnd", level, dates, "clim")
    u_anom = gaussian_filter(u - uc, sigma=1.5)
    v_anom = gaussian_filter(v - vc, sigma=1.5)
    return streamfunction_from_uv(u_anom, v_anom, lat, lon)

def _mean_air(level, dates):
    obs = _mean_field("air", level, dates, "obs")
    clim = _mean_field("air", level, dates, "clim")
    return gaussian_filter(obs - clim, sigma=1.5)

def _temp_k(level, dates):
    """Mean absolute temperature (K) at a level for static-stability/theta maps."""
    return _mean_field("air", level, dates, "obs")

def _static_stability(level, dates):
    """s(p) = -alpha * dln(theta)/dp, evaluated with a 3-level centred difference
    using the absolute temperature profile."""
    idx, hi, lo = _lerp_levels(AAM_LEVELS, level)
    p_c, p_hi, p_lo = AAM_LEVELS[idx]*100.0, AAM_LEVELS[hi]*100.0, AAM_LEVELS[lo]*100.0
    Tc = _temp_k(AAM_LEVELS[idx], dates)
    Thi = _temp_k(AAM_LEVELS[hi], dates)
    Tlo = _temp_k(AAM_LEVELS[lo], dates)
    th_c = potential_temp(Tc, AAM_LEVELS[idx])
    th_hi = potential_temp(Thi, AAM_LEVELS[hi])
    th_lo = potential_temp(Tlo, AAM_LEVELS[lo])
    dlnth = (np.log(th_hi) - np.log(th_lo)) / (p_hi - p_lo)
    p_mid = (p_hi + p_lo) * 0.5
    T_mid = (Thi + Tlo) * 0.5
    alpha = 287.05 * T_mid / p_mid
    return -alpha * dlnth + 1e-9

def _q_level(level, dates, kind):
    """Specific humidity field (kg/kg) at a level, obs or climatology."""
    T = _mean_field("air", level, dates, kind)
    rh = _mean_field("rhum", level, dates, kind)
    return _spec_hum(T, rh, level * 100.0)

def _geopot(level, dates, kind="obs"):
    """Geopotential Phi = g·z (m²/s²) from the hgt (gpm) field."""
    return 9.80665 * _mean_field("hgt", level, dates, kind)

def _geo_wind(level, dates, lat, lon):
    """Geostrophic wind (m/s) from the geopotential field. f→0 near the equator
    is guarded (returns 0) so gradients stay finite; tropics are masked at render."""
    phi = _geopot(level, dates)
    f = 2 * DEG_PER_S * np.sin(np.deg2rad(lat))[:, None]
    dphidx = _grad_x(phi, lat, lon)
    dphidy = _grad_y(phi, lat, lon)
    with np.errstate(divide="ignore", invalid="ignore"):
        ug = np.where(np.abs(f) > 1e-6, -dphidy / f, 0.0)
        vg = np.where(np.abs(f) > 1e-6, dphidx / f, 0.0)
    return ug, vg

def _qvector_forcing(level, dates, lat, lon):
    """QG omega forcing = −2∇·Q (Hoskins Q-vector form). Positive ⇒ ascent.
    Geostrophic wind from height field, temperature from air; Q-vector is built
    from the geostrophic deformation of the temperature gradient."""
    ug, vg = _geo_wind(level, dates, lat, lon)
    T = _mean_field("air", level, dates, "obs")
    dTdx = _grad_x(T, lat, lon)
    dTdy = _grad_y(T, lat, lon)
    dUgdx = _grad_x(ug, lat, lon); dVgdx = _grad_x(vg, lat, lon)
    dUgdy = _grad_y(ug, lat, lon); dVgdy = _grad_y(vg, lat, lon)
    sigma = _static_stability(level, dates)
    coef = RD / (sigma * level * 100.0) * 1.0
    Q1 = -coef * (dUgdx * dTdx + dVgdx * dTdy)
    Q2 = -coef * (dUgdy * dTdx + dVgdy * dTdy)
    divQ = _grad_x(Q1, lat, lon) + _grad_y(Q2, lat, lon)
    return -2.0 * divQ

def _temp_advection(level, dates, lat, lon):
    """−V·∇T (K/s), positive = warm advection, from absolute obs fields."""
    u = _mean_field("uwnd", level, dates, "obs")
    v = _mean_field("vwnd", level, dates, "obs")
    T = _mean_field("air", level, dates, "obs")
    dTdx = _grad_x(T, lat, lon)
    dTdy = _grad_y(T, lat, lon)
    return -(u * dTdx + v * dTdy)

def _band_axis(lat, lat_min, lat_max):
    """Indices of the latitude rows inside [lat_min, lat_max]."""
    return np.where((lat >= lat_min) & (lat <= lat_max))[0]

def _band_label(band):
    """'5°S–5°N' style label for a (lat_min, lat_max) band."""
    lo, hi = band
    f = lambda x: f"{abs(x):g}°{'S' if x < 0 else 'N'}"
    return f"averaged {f(lo)}–{f(hi)}"

def compute_hov(pkg, dates):
    """Approach: build a daily time×longitude Hovmöller for pkg['variable'] at
    pkg['level'], averaged over the latitude band pkg['lat_band'], over the last
    pkg['window'] days ending at the latest requested date. Returns
    (day_dates, lon, matrix) where matrix is (ntime, nlon) already scaled."""
    var = pkg["variable"]
    level = pkg["level"]
    lat_min, lat_max = pkg["lat_band"]
    window = int(pkg.get("window", 120))
    end = dates[-1]
    day_dates = [end - datetime.timedelta(days=i) for i in range(window)][::-1]

    lat, lon = _latlon("uwnd")          # shared 73×144 grid
    band = _band_axis(lat, lat_min, lat_max)

    if var == "chi":
        # velocity potential: per-day u,v anomaly -> divergence -> Poisson
        u = _daily_stack("uwnd", level, day_dates, "obs")
        uc = _daily_stack("uwnd", level, day_dates, "clim")
        v = _daily_stack("vwnd", level, day_dates, "obs")
        vc = _daily_stack("vwnd", level, day_dates, "clim")
        u_a = gaussian_filter(u - uc, sigma=1.2)
        v_a = gaussian_filter(v - vc, sigma=1.2)
        rows = []
        for i in range(day_dates.__len__()):
            div = divergence(u_a[i], v_a[i], lat, lon)
            chi = gaussian_filter(poisson_fft(div, lat, lon), sigma=2.0)
            rows.append(chi[band].mean(axis=0))
        matrix = np.stack(rows, axis=0) * pkg["plot_scale"]
    else:
        obs = _daily_stack(var, level, day_dates, "obs")
        clim = _daily_stack(var, level, day_dates, "clim")
        anom = gaussian_filter(obs - clim, sigma=1.2)
        matrix = anom[:, band, :].mean(axis=1) * pkg["plot_scale"]
        scale = 1.0
    # Remove the daily zonal mean so the eastward-propagating wave structure is
    # visible (the band-mean otherwise keeps a large global-mean baseline, e.g.
    # the planetary-scale chi component that would saturate the colour scale).
    if pkg.get("zonal_anom", True):
        matrix = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    return day_dates, lon, matrix

def compute(pkg, dates):
    kind = pkg["kind"]

    # custom addon kind? dispatch to its own compute before the built-ins.
    custom_fn = _custom_compute(kind)
    if custom_fn is not None:
        return custom_fn(pkg, dates)

    if kind in ("vtp", "psi"):
        lat, lon = _latlon("uwnd")
        u_obs = _mean_field("uwnd", pkg["level"], dates, "obs")
        u_clim = _mean_field("uwnd", pkg["level"], dates, "clim")
        v_obs = _mean_field("vwnd", pkg["level"], dates, "obs")
        v_clim = _mean_field("vwnd", pkg["level"], dates, "clim")
        u_anom = gaussian_filter(u_obs - u_clim, sigma=1.5)
        v_anom = gaussian_filter(v_obs - v_clim, sigma=1.5)
        if kind == "vtp":
            div = divergence(u_anom, v_anom, lat, lon)
            main = gaussian_filter(poisson_fft(div, lat, lon), sigma=2.0) * pkg["plot_scale"]
        else:
            zeta = vorticity(u_anom, v_anom, lat, lon)
            main = gaussian_filter(poisson_fft(zeta, lat, lon), sigma=2.0) * pkg["plot_scale"]
        return lat, lon, {"main": main, "u": u_anom, "v": v_anom}

    elif kind == "ft":
        # Frictional torque driver: surface wind stress anomaly via the bulk drag
        # law  tau = rho * Cd * |V10| * V10  (N/m²). All three flavours share the
        # computation; pkg["comp"] selects which scalar to show, and the full
        # (tau_x, tau_y) vector is always returned for the arrow overlay.
        lat, lon = _latlon("uwnd.sfc")
        u_obs = _mean_field("uwnd.sfc", None, dates, "obs")
        u_clim = _mean_field("uwnd.sfc", None, dates, "clim")
        v_obs = _mean_field("vwnd.sfc", None, dates, "obs")
        v_clim = _mean_field("vwnd.sfc", None, dates, "clim")

        rho, cd = 1.225, 1.4e-3
        def stress(u, v):
            spd = np.sqrt(u * u + v * v)
            tau_u = rho * cd * spd * u
            tau_v = rho * cd * spd * v
            return tau_u, tau_v
        tx_o, ty_o = stress(u_obs, v_obs)
        tx_c, ty_c = stress(u_clim, v_clim)
        tx = gaussian_filter(tx_o - tx_c, sigma=1.5)
        ty = gaussian_filter(ty_o - ty_c, sigma=1.5)

        comp = pkg.get("comp", "x")
        if comp == "y":
            main = ty
        elif comp == "mag":
            main = np.sqrt(tx**2 + ty**2)
        else:
            main = tx
        vs = pkg.get("vec_scale", 1.0)
        return lat, lon, {"main": main * pkg["plot_scale"],
                          "vec_u": tx * vs, "vec_v": ty * vs}

    elif kind == "waf":
        # Takaya–Nakamura wave-activity flux at pkg["level"].
        lat, lon = _latlon("uwnd")
        u_obs = _mean_field("uwnd", pkg["level"], dates, "obs")
        u_clim = _mean_field("uwnd", pkg["level"], dates, "clim")
        v_obs = _mean_field("vwnd", pkg["level"], dates, "obs")
        v_clim = _mean_field("vwnd", pkg["level"], dates, "clim")
        u_anom = gaussian_filter(u_obs - u_clim, sigma=1.5)
        v_anom = gaussian_filter(v_obs - v_clim, sigma=1.5)
        psi = streamfunction_from_uv(u_anom, v_anom, lat, lon)

        # basic-state wind from the climatology (zonal-mean U, full V)
        u_basic = u_clim
        v_basic = v_clim
        waf_u, waf_v = takaya_nakamura_flux(psi, u_basic, v_basic, lat, lon,
                                            p_pa=pkg["level"] * 100.0)

        main = psi * pkg["plot_scale"]
        vec_sc = pkg["vec_scale"]
        # NaN mask must match the flux mask ({>80°} treated as NaN) so no
        # garbage arrows or shading are drawn right at the poles.
        return lat, lon, {"main": np.where(np.abs(lat)[:, None] <= 80.0,
                                           main, np.nan),
                          "vec_u": waf_u * vec_sc, "vec_v": waf_v * vec_sc}

    elif kind == "qgpv":
        # Quasi-geostrophic potential-vorticity anomaly:
        #   q' = lap_h(psi') + f^2 * d/dp[ (1/s) dpsi'/dp ]
        # with s = -alpha * dln(theta)/dp  (static stability, from the mean temperature).
        idx, hi, lo = _lerp_levels(AAM_LEVELS, pkg["level"])
        p_c = AAM_LEVELS[idx] * 100.0
        p_hi = AAM_LEVELS[hi] * 100.0
        p_lo = AAM_LEVELS[lo] * 100.0
        lat, lon = _latlon("uwnd")
        f = 2 * DEG_PER_S * np.sin(np.deg2rad(lat))[:, None]
        psi_c = _psi_level(pkg["level"], dates)
        psi_hi = _psi_level(AAM_LEVELS[hi], dates)
        psi_lo = _psi_level(AAM_LEVELS[lo], dates)
        lap_psi = _laplacian(psi_c, lat, lon)
        # vertical static stability at the level (absolute temperature)
        s_c = _static_stability(AAM_LEVELS[idx], dates)
        s_hi = _static_stability(AAM_LEVELS[hi], dates)
        s_lo = _static_stability(AAM_LEVELS[lo], dates)
        # (1/s)*dpsi/dp at the two sub-intervals (clip s to a realistic floor)
        s_hi_c = np.maximum(s_hi, 1e-6)
        s_lo_c = np.maximum(s_lo, 1e-6)
        g_up = (psi_hi - psi_c) / (p_hi - p_c) / s_hi_c
        g_dn = (psi_c - psi_lo) / (p_c - p_lo) / s_lo_c
        dg_dp = 2.0 * (g_up - g_dn) / (p_hi - p_lo)
        q = lap_psi + f**2 * dg_dp
        # The spherical Laplacian degenerates as cosφ→0; the pole rows are not
        # a real signal. Mask them so they render blank rather than saturating.
        q = np.where(np.abs(lat)[:, None] > 78.0, np.nan, q)
        return lat, lon, {"main": q * pkg["plot_scale"]}

    elif kind == "eddy":
        # Transient-eddy meridional fluxes v'T' and u'v' (deviation from zonal mean
        # of the anomaly fields) at pkg["level"].
        lat, lon = _latlon("uwnd")
        u_a = _anom("uwnd", pkg["level"], dates, lat, lon)
        v_a = _anom("vwnd", pkg["level"], dates, lat, lon)
        T_a = _anom("air", pkg["level"], dates, lat, lon)
        # deviation from zonal mean
        v_e = v_a - zonal_mean(v_a)[:, None]
        T_e = T_a - zonal_mean(T_a)[:, None]
        u_e = u_a - zonal_mean(u_a)[:, None]
        vt = v_e * T_e
        uv = u_e * v_e
        main = vt if pkg.get("flux") == "vt" else uv
        return lat, lon, {"main": main * pkg["plot_scale"],
                          "u": u_e, "v": v_e}

    elif kind == "eady":
        # Eady baroclinic growth rate (lower/mid troposphere) between two levels.
        # Uses the ABSOLUTE (observed) wind shear and temperature, so N^2 and the
        # growth rate are physically realistic (1/day).
        lat, lon = _latlon("uwnd")
        p_lo = pkg["p_low"]; p_hi = pkg["p_high"]
        u_low = _mean_field("uwnd", p_lo, dates, "obs")
        u_up = _mean_field("uwnd", p_hi, dates, "obs")
        T_low = _temp_k(p_lo, dates)
        T_up = _temp_k(p_hi, dates)
        sigma = eady_growth(u_low, u_up, T_low, T_up, p_lo*100.0, p_hi*100.0, lat)
        main = sigma * 86400.0 * pkg["plot_scale"]   # 1/day
        return lat, lon, {"main": main}

    elif kind == "ivt":
        # Integrated Water Vapour Transport (kg m⁻¹ s⁻¹): column integral of q·V.
        # Q = (1/g) ∫ q·(u,v) dp; magnitude is the standard atmospheric-river metric.
        # All 4 fields are pulled as multi-level stacks (one set of reads each),
        # then combined as arrays, so the whole computation is a handful of
        # dataset requests rather than ~50.
        # rhum is archived on only 6 levels (1000–300 hPa), so integrate the
        # moisture up to 300 hPa (the overwhelming fraction of PW sits below).
        levels = [1000, 850, 700, 500, 400, 300]
        lat, lon = _latlon("uwnd")
        u = _mean_multi("uwnd", levels, dates, "obs")
        v = _mean_multi("vwnd", levels, dates, "obs")
        T = _mean_multi("air", levels, dates, "obs")
        RH = _mean_multi("rhum", levels, dates, "obs")
        p_pa = np.array(levels, dtype=np.float64)[:, None, None] * 100.0
        q = _spec_hum(T, RH, p_pa)                       # (nlev, nlat, nlon)
        pt = p_pa[:, 0, 0]
        Qx = np.trapezoid(q * u, x=pt[::-1], axis=0) / GRAV
        Qy = np.trapezoid(q * v, x=pt[::-1], axis=0) / GRAV
        main = np.sqrt(Qx**2 + Qy**2) * pkg["plot_scale"]
        vs = pkg.get("vec_scale", 1.0)
        return lat, lon, {"main": main, "vec_u": Qx * vs, "vec_v": Qy * vs}

    elif kind == "qgforcing":
        # QG omega forcing = −2∇·Q at pkg["level"] (positive ⇒ ascent). Geostrophic
        # winds break down near the equator, so the deep tropics are masked.
        level = pkg["level"]
        lat, lon = _latlon("uwnd")
        forcing = _qvector_forcing(level, dates, lat, lon)
        min_lat = pkg.get("min_lat", 12.0)
        forcing = np.where(np.abs(lat)[:, None] < min_lat, np.nan, forcing)
        forcing = np.where(np.abs(lat)[:, None] > 80.0, np.nan, forcing)
        return lat, lon, {"main": forcing * pkg["plot_scale"]}

    elif kind == "mse":
        # Moist Static Energy anomaly at pkg["level"]:  MSE = Cp·T + Lv·q + gz.
        level = pkg["level"]
        lat, lon = _latlon("air")
        T_o = _mean_field("air", level, dates, "obs")
        T_c = _mean_field("air", level, dates, "clim")
        q_o = _q_level(level, dates, "obs")
        q_c = _q_level(level, dates, "clim")
        phi_o = _geopot(level, dates, "obs")
        phi_c = _geopot(level, dates, "clim")
        mse_o = CP * T_o + LV * q_o + phi_o
        mse_c = CP * T_c + LV * q_c + phi_c
        return lat, lon, {"main": (mse_o - mse_c) * pkg["plot_scale"]}

    elif kind == "tadv":
        # Temperature advection −V·∇T at pkg["level"], in K/s (×24 h scale to K/day).
        level = pkg["level"]
        lat, lon = _latlon("uwnd")
        ta = _temp_advection(level, dates, lat, lon)
        # the 1/cosφ term in dT/dx amplifies to noise in the high latitudes;
        # the meaningful frontal/advective signal sits in the mid-latitudes.
        ta = np.where(np.abs(lat)[:, None] > 68.0, np.nan, ta)
        return lat, lon, {"main": ta * pkg["plot_scale"]}

    elif kind in ("geowind", "ageowind"):
        # Geostrophic Vg from the height field, or ageostrophic V − Vg. Magnitude
        # is shaded; the vector is overlaid. Tropics masked (f→0, geostrophy fails).
        level = pkg["level"]
        lat, lon = _latlon("uwnd")
        ug, vg = _geo_wind(level, dates, lat, lon)
        if kind == "geowind":
            U0, V0 = ug, vg
        else:
            u = _mean_field("uwnd", level, dates, "obs")
            v = _mean_field("vwnd", level, dates, "obs")
            U0, V0 = u - ug, v - vg
        main = np.sqrt(U0**2 + V0**2)
        min_lat = pkg.get("min_lat", 12.0)
        # mask the deep tropics (f→0, geostrophy fails) and the polar rows where
        # the 1/cosφ derivative explodes — in BOTH the field and the vectors.
        bad = (np.abs(lat)[:, None] < min_lat) | (np.abs(lat)[:, None] > 78.0)
        main = np.where(bad, np.nan, main)
        return lat, lon, {"main": main * pkg["plot_scale"],
                          "vec_u": np.where(bad, np.nan, U0),
                          "vec_v": np.where(bad, np.nan, V0)}

    elif kind == "rossby":
        # Rossby-wave composite at pkg["level"]: χ′ source shading + ψ′ wave-train
        # contours + Takaya–Nakamura wave-activity flux + Ks stationary wavenumber.
        level = pkg["level"]
        lat, lon = _latlon("uwnd")
        u_obs = _mean_field("uwnd", level, dates, "obs")
        u_clim = _mean_field("uwnd", level, dates, "clim")
        v_obs = _mean_field("vwnd", level, dates, "obs")
        v_clim = _mean_field("vwnd", level, dates, "clim")
        u_anom = gaussian_filter(u_obs - u_clim, sigma=1.5)
        v_anom = gaussian_filter(v_obs - v_clim, sigma=1.5)

        # χ′ velocity-potential source (upper divergence = convection source)
        div = divergence(u_anom, v_anom, lat, lon)
        chi = gaussian_filter(poisson_fft(div, lat, lon), sigma=2.0) * pkg["plot_scale"]
        chi = np.where(np.abs(lat)[:, None] <= 80.0, chi, np.nan)

        # ψ′ streamfunction wave train (m²/s, kept unscaled for the contours)
        psi = streamfunction_from_uv(u_anom, v_anom, lat, lon)
        psi = np.where(np.abs(lat)[:, None] <= 80.0, psi, np.nan)

        # Takaya–Nakamura wave-activity flux (m²/s², scaled for the arrows)
        waf_u, waf_v = takaya_nakamura_flux(psi, u_clim, v_clim, lat, lon,
                                            p_pa=level * 100.0)
        vec_sc = pkg["vec_scale"]

        # Ks stationary wavenumber (dimensionless; weak-wind rows masked)
        ks = stationary_wavenumber(u_obs, lat, ubar_min=pkg.get("ubar_min", 4.0))

        return lat, lon, {"main": chi, "psi": psi,
                          "vec_u": waf_u * vec_sc, "vec_v": waf_v * vec_sc,
                          "ks": ks}

    else:  # "anom" — single-variable anomaly
        var = pkg["variable"]
        lat, lon = _latlon(var)
        obs = _mean_field(var, pkg["level"], dates, "obs")
        clim = _mean_field(var, pkg["level"], dates, "clim")
        anom = gaussian_filter(obs - clim, sigma=1.5) * pkg["plot_scale"]
        return lat, lon, {"main": anom}


# --- custom addon kinds -----------------------------------------------------
# If ``kind`` was defined in a ./custom/*.py file (see pro/addons.py), dispatch
# to its own compute function. The addon's compute(pkg, dates) returns
# (lat, lon, data); it may import any encoder helpers from pro.data/pro.physics.
from . import addons  # noqa: E402

_addon_kinds = addons.custom_kinds()


def _custom_compute(kind):
    spec = _addon_kinds.get(kind)
    if spec and callable(spec.get("compute")):
        return spec["compute"]
    return None


def _custom_render(kind):
    spec = _addon_kinds.get(kind)
    if spec and callable(spec.get("render")):
        return spec["render"]
    return None


def _custom_uses_builtin_render(kind):
    """Custom kinds with no render of their own fall back to the generic map
    renderer, so a config-only addon can also be a custom kind."""
    spec = _addon_kinds.get(kind)
    return bool(spec is not None and not callable(spec.get("render")))
