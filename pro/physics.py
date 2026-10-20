"""pro.physics — pure spherical/physical operators (no data access)."""
import numpy as np
from scipy.ndimage import gaussian_filter
from . import config
R_EARTH = config.R_EARTH
DEG_PER_S = config.DEG_PER_S
KAPPA = config.KAPPA

def divergence(u, v, lat, lon):
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)
    dudx = np.gradient(u, lon_r, axis=1) / (R_EARTH * coslat[:, None])
    vcoslat = v * coslat[:, None]
    dvdy = np.gradient(vcoslat, lat_r, axis=0) / (R_EARTH * coslat[:, None])
    return dudx + dvdy

def vorticity(u, v, lat, lon):
    """Relative vorticity (vertical component) from u,v on a lon-lat grid."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)[:, None]
    dudphi = np.gradient(u * coslat, lat_r, axis=0)
    dvdlon = np.gradient(v, lon_r, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(np.abs(coslat) > 1e-4,
                        (1.0 / (R_EARTH * coslat)) * (dvdlon - dudphi), 0.0)
    # fix the (ill-defined) pole rows with the adjacent row
    zeta[0] = zeta[1]
    zeta[-1] = zeta[-2]
    return zeta

def poisson_fft(rhs, lat, lon):
    """Solve del^2(psi) = rhs on the sphere via FFT (band-limited)."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dy = R_EARTH * np.abs(np.mean(np.diff(lat_r)))
    dx_mean = R_EARTH * np.mean(np.diff(lon_r)) * np.mean(np.abs(np.cos(lat_r)))
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

def potential_temp(T, press_hpa):
    """theta = T (1000/p)^kappa."""
    return T * (1000.0 / press_hpa) ** KAPPA

def takaya_nakamura_flux(psi_anom, u_bar, v_bar, lat, lon, p_pa=20000.0, a=R_EARTH):
    """Horizontal Takaya-Nakamura (2001) wave-activity flux at pressure p_pa.

    W = (p cosφ / (2 |U| a²)) · ( U·A + V·B ,  U·B + V·C )

    with (λ,φ in radians, unitless derivatives — geometry lives in the prefactor):
        A = (∂ψ'/∂λ)² − ψ' ∂²ψ'/∂λ²
        B = (∂ψ'/∂λ)(∂ψ'/∂φ) − ψ' ∂²ψ'/∂λ∂φ
        C = (∂ψ'/∂φ)² − ψ' ∂²ψ'/∂φ²

    Basic-state wind is the slowly-varying background (zonal mean of the total
    wind). Result is in m²/s², direction = local group-velocity propagation.
    """
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)[:, None]
    ubar_z = zonal_mean(u_bar)
    ub = np.broadcast_to(ubar_z[:, None], psi_anom.shape)
    vb = zonal_mean(v_bar)[:, None]
    U = np.sqrt(ub**2 + vb**2) + 1e-8

    # NOTE: psi_anom is kept NaN-free. Placing NaN inside the field first makes
    # np.gradient blow up at the edges; the polar rows are handled by the output
    # mask below (cosφ→0 makes the inversion unreliable there anyway).

    dpsi_dlam = np.gradient(psi_anom, lam, axis=1)
    d2psi_dlam2 = np.gradient(dpsi_dlam, lam, axis=1)
    dpsi_dphi = np.gradient(psi_anom, phi, axis=0)
    d2psi_dphidlam = np.gradient(dpsi_dlam, phi, axis=0)
    d2psi_dphi2 = np.gradient(dpsi_dphi, phi, axis=0)

    A = dpsi_dlam**2 - psi_anom * d2psi_dlam2
    B = dpsi_dlam * dpsi_dphi - psi_anom * d2psi_dphidlam
    C = dpsi_dphi**2 - psi_anom * d2psi_dphi2

    pref = (p_pa * cosphi) / (2.0 * U * a**2)
    Wx = pref * (ub * A + vb * B)
    Wy = pref * (ub * B + vb * C)

    # Mask only the topmost rows where the streamfunction inversion is
    # genuinely degenerate as cosφ→0; keep the field out to ±80°.
    bad = np.abs(lat)[:, None] > 80.0
    Wx = np.where(bad, np.nan, Wx)
    Wy = np.where(bad, np.nan, Wy)
    return Wx, Wy

def zonal_mean(f):
    return np.nanmean(f, axis=1)

def stationary_wavenumber(U, lat, ubar_min=4.0):
    """Stationary Rossby wavenumber Ks = a cosφ sqrt(βM / ū)  (dimensionless).

    ū is the zonal-mean zonal wind, βM = β − d²ū/dy² the meridional-gradient
    (beta) term including the curvature of the jet. Ks is only defined where
    the background zonal flow is westerly and βM > 0; where the jet is strong
    Ks is large and planetary waves are trapped/ducted (a 'waveguide'). The
    weak-wind rows (ū → 0, where Ks blows up unphysically) are masked, so the
    band reflects the genuine jet corridors.
    """
    ubar = zonal_mean(U)
    phi = np.deg2rad(lat)
    a, Om = R_EARTH, DEG_PER_S
    beta = 2.0 * Om * np.cos(phi) / a          # 1/(m·s)
    us = gaussian_filter(ubar, 1.2)            # smooth before the 2nd derivative
    uyy = np.gradient(np.gradient(us, phi), phi) / (a * a)
    betaM = beta - uyy                          # 1/(m·s)
    with np.errstate(divide="ignore", invalid="ignore"):
        ks = a * np.cos(phi) * np.sqrt(betaM / ubar)
    ks = np.where((ubar > ubar_min) & (betaM > 0.0), ks, np.nan)
    return ks

def _laplacian(psi, lat, lon):
    """Spherical Laplacian ∇²ψ = (1/R²)[ ∂²ψ/∂φ² − tanφ ∂ψ/∂φ + (1/cos²φ) ∂²ψ/∂λ² ]."""
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    coslat = np.cos(phi)[:, None]
    tanlat = np.tan(phi)[:, None]
    d_phi = np.gradient(psi, phi, axis=0)
    d2phi = np.gradient(d_phi, phi, axis=0)
    d2lam = np.gradient(np.gradient(psi, lam, axis=1), lam, axis=1)
    return (d2phi - tanlat * d_phi + d2lam / coslat**2) / (R_EARTH**2)

def eady_growth(u_low, u_up, T_low, T_up, p_low, p_high, lat, a=R_EARTH):
    """Eady baroclinic growth rate sigma = 0.31 f |du/dz| / N  (1/day)."""
    phi = np.deg2rad(lat)
    f = 2 * DEG_PER_S * np.sin(phi)[:, None]
    g = 9.80665
    # vertical wind shear du/dp
    du_dp = (u_up - u_low) / (p_high - p_low)      # p_high > p_low (pressure coords)
    # rho ~ p/(R T) hydrostatic; du/dz = -rho*g*du/dp
    R = 287.05
    p_mean = (p_low + p_high) * 0.5
    T_mean = (T_low + T_up) * 0.5
    rho = p_mean / (R * T_mean)
    du_dz = -rho * g * du_dp
    # N^2 from theta: N2 = (g/theta) dtheta/dz = -g*rho*(g/theta) dtheta/dp
    th_low = potential_temp(T_low, p_low)
    th_up = potential_temp(T_up, p_high)
    dth_dp = (th_up - th_low) / (p_high - p_low)
    N2 = -g * g * rho * dth_dp / (th_low + th_up) * 2.0
    N2 = np.maximum(N2, 1e-8)
    sigma = 0.31 * np.abs(f) * np.abs(du_dz) / np.sqrt(N2)
    return sigma   # 1/s

def _sat_vp(T_k):
    """Saturation vapour pressure (Pa), Bolton (1980)."""
    Tc = T_k - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))

def _spec_hum(T_k, rh, p_pa):
    """Specific humidity q (kg/kg) from temperature, RH% and pressure (Pa)."""
    e = _sat_vp(T_k) * (np.clip(rh, 0.0, 100.0) / 100.0)
    e = np.minimum(e, 0.95 * p_pa)          # guard vs. supersaturation at low p
    return 0.622 * e / (p_pa - 0.378 * e)

def _grad_x(a, lat, lon):
    """d/dx (eastward) of a 2-D field on the sphere."""
    lon_r = np.deg2rad(lon)
    coslat = np.cos(np.deg2rad(lat))[:, None]
    return np.gradient(a, lon_r, axis=1) / (R_EARTH * coslat)

def _grad_y(a, lat, lon):
    """d/dy (northward) of a 2-D field on the sphere."""
    lat_r = np.deg2rad(lat)
    return np.gradient(a, lat_r, axis=0) / R_EARTH

def _lerp_levels(levels, target):
    """Return (index, above_idx, below_idx) for a central-difference window."""
    arr = np.asarray(levels, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - target)))
    lo = max(0, idx - 1)
    hi = min(len(arr) - 1, idx + 1)
    if lo == hi:
        lo = max(0, idx - 1)
        hi = min(len(arr) - 1, idx + 1)
        if lo == hi:
            lo = idx
            hi = idx
    return idx, hi, lo
