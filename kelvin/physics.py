"""kelvin.physics — spherical VP inversion and the Kelvin phase-space transform.

The operators are the ones from the user's standalone script, kept in one
place so the ECMWF fetcher and the plot never disagree:

* ``d = Laplacian(chi)`` is inverted for every retained zonal wavenumber on
  the full latitude circle (Dirichlet zero at the poles), then the
  cosine-weighted 15S-15N average is formed.
* the resulting complex coefficients are band-passed in time to keep only
  eastward frequencies with periods between 2.5 and 20 days (k = 1..14);
* the filtered equatorial VP200 longitude field is projected onto the two
  digitized reference EOFs and standardised per forecast run.
"""
from __future__ import annotations

import numpy as np

from . import eofs

#: latitude band of the reference product
LAT_BAND = (-15.0, 15.0)
#: highest zonal wavenumber retained by the Kelvin filter
K_WAVENUMBER_MAX = 14
#: eastward Kelvin band (days); the low-frequency edge is limited by the
#: 15-day forecast window and by the mirror padding used below
PERIOD_DAYS = (2.5, 20.0)
#: mean Earth radius used by the IFS grid
EARTH_RADIUS_M = 6_371_000.0


def _tridiagonal_solve(lower, diag, upper, rhs):
    """Thomas algorithm for a complex tridiagonal system.

    ``lower`` and ``upper`` have ``len(diag) - 1`` elements; ``lower[j]``
    multiplies ``x[j]`` in row ``j+1`` and ``upper[j]`` multiplies ``x[j+1]``
    in row ``j``.
    """
    n = len(diag)
    if n == 0:
        return np.empty(0, dtype=np.complex128)
    if len(lower) != n - 1 or len(upper) != n - 1 or len(rhs) != n:
        raise ValueError("inconsistent tridiagonal system sizes")
    c = np.empty(n - 1, dtype=np.complex128)
    d = np.empty(n, dtype=np.complex128)
    den = diag[0]
    if abs(den) < 1e-30:
        raise ArithmeticError("singular tridiagonal system")
    c[0] = upper[0] / den if n > 1 else 0.0
    d[0] = rhs[0] / den
    for i in range(1, n):
        den = diag[i] - lower[i - 1] * c[i - 1]
        if abs(den) < 1e-30:
            raise ArithmeticError("singular tridiagonal system")
        if i < n - 1:
            c[i] = upper[i] / den
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / den
    x = np.empty(n, dtype=np.complex128)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def vp_equatorial_coefficients(values, lats, kmax=K_WAVENUMBER_MAX):
    """Zonal Fourier coefficients of equatorial VP200 from a global d200 field.

    Parameters
    ----------
    values : (nlat, nlon) array
        Global 200-hPa divergence (s^-1) on a regular latitude/longitude grid.
    lats : (nlat,) array
        Latitudes in degrees, either ascending or descending.
    kmax : int
        Highest zonal wavenumber returned.

    Returns
    -------
    (kmax + 1,) complex ndarray
        Coefficient for wavenumbers 0..kmax.  The constant (k=0) mode is
        returned for completeness but is discarded by the Kelvin filter.
    """
    values = np.asarray(values, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"d200 field must be 2-D, got {values.shape}")
    nlat, nlon = values.shape
    if lats.size != nlat:
        raise ValueError("latitude vector does not match the field")
    if nlat < 100 or nlon < 360:
        raise ValueError(f"expected a global regular lat/lon grid, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("d200 field contains non-finite values")

    # Work south -> north so that the latitude index increases.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]

    phi = np.deg2rad(lats)
    dphi = float(np.median(np.diff(phi)))
    if not np.isfinite(dphi) or dphi <= 0:
        raise ValueError("invalid latitude spacing")

    # numpy's positive-zonal-wavenumber convention: arr = sum d_hat[k] e^{ik lon}
    d_hat = np.fft.rfft(values, axis=1) / float(nlon)

    band = (lats >= LAT_BAND[0]) & (lats <= LAT_BAND[1])
    if not band.any():
        raise ValueError("no grid rows inside the 15S-15N band")
    weights = np.cos(phi[band])
    weights = weights / weights.sum()

    out = np.zeros(kmax + 1, dtype=np.complex128)
    cfaces = np.cos((phi[:-1] + phi[1:]) * 0.5)
    for k in range(1, min(kmax, d_hat.shape[1] - 1) + 1):
        c = np.cos(phi[1:-1])
        # Coefficients of chi[p-1] and chi[p+1] in the latitude row p.
        lo = cfaces[:-1] / (c * dphi * dphi)
        up = cfaces[1:] / (c * dphi * dphi)
        diag = -(lo + up) - (k * k) / (c * c)
        # Interior rows exclude the two nearest-pole points (chi = 0 there).
        rhs = (EARTH_RADIUS_M ** 2) * d_hat[1:-1, k]
        chi_in = _tridiagonal_solve(lo[1:], diag.astype(np.complex128),
                                    up[:-1], rhs)
        chi = np.zeros(nlat, dtype=np.complex128)
        chi[1:-1] = chi_in
        out[k] = np.sum(chi[band] * weights)
    return out


def _mirror_extend(z):
    """Three-copy reflection at both ends, conjugated in the mirrored halves.

    A plain (real) reflection reverses the direction of propagation: a
    time-reversed eastward wave looks westward, so mirroring the complex
    coefficients without conjugation injects a spurious westward copy
    (measured at ~24% of the eastward amplitude for a 10-day wave) and damps
    the first/last forecast day to 60% of its amplitude.  Conjugating the
    mirrored halves keeps the sense of propagation, which removes both
    artefacts; the taper below then only has to soften the very ends.
    """
    z = np.asarray(z)
    left = np.conj(z[1:])[::-1]          # reflection about the first sample
    right = np.conj(z[-2:0:-1])          # reflection about the last sample
    return np.concatenate((left, z, right))


def _tukey(n, alpha=0.12):
    """Tukey window, used to soften the ends of the reflected series."""
    if n <= 1:
        return np.ones(n)
    w = np.ones(n)
    edge = int(alpha * (n - 1) / 2)
    if edge < 1:
        return w
    i = np.arange(edge + 1)
    ramp = 0.5 * (1 - np.cos(np.pi * i / edge))
    w[:edge + 1] = ramp
    w[-edge - 1:] = ramp[::-1]
    return w


def filter_eastward_kelvin(coeff, step_hours):
    """Keep eastward Kelvin-band power in the complex zonal coefficients.

    ``coeff`` has shape ``(nt, kmax+1)`` (time x wavenumber) and is sampled
    every ``step_hours`` hours.  Frequencies are in cycles/day; with the
    ``e^{i(k lon - omega t)}`` convention an eastward wave appears at a
    negative FFT frequency, which is the sign selected here.
    """
    coeff = np.asarray(coeff, dtype=np.complex128)
    if coeff.ndim != 2:
        raise ValueError(f"coefficient array must be 2-D, got {coeff.shape}")
    nt, nk = coeff.shape
    if nt < 3:
        raise ValueError("at least three forecast times are needed for the filter")
    dt_days = float(step_hours) / 24.0
    if not np.isfinite(dt_days) or dt_days <= 0:
        raise ValueError("step_hours must be positive")

    ext = np.stack([_mirror_extend(coeff[:, k]) for k in range(nk)], axis=1)
    n_ext = ext.shape[0]
    taper = _tukey(n_ext)[:, None]
    spec = np.fft.fft(ext * taper, axis=0)
    freq = np.fft.fftfreq(n_ext, d=dt_days)   # cycles/day
    period_ok = ((np.abs(freq) >= 1.0 / PERIOD_DAYS[1]) &
                 (np.abs(freq) <= 1.0 / PERIOD_DAYS[0]))
    mask = np.repeat((period_ok & (freq < 0))[:, None], nk, axis=1)
    mask[:, 0] = False                        # discard the zonal mean
    filtered = np.fft.ifft(spec * mask, axis=0)
    # The original sequence is the centre of the reflected extension.
    start = nt - 1
    return filtered[start:start + nt]


def phase_coordinates(coeff_filtered, nlon, eof1=None, eof2=None):
    """Project the filtered equatorial VP200 field onto the two EOFs.

    Returns ``(x, y, phase_deg, amplitude, (scale1, scale2))``.  ``x`` and
    ``y`` are the per-run standardised PC1/PC2, ``phase_deg`` runs 0..360
    anticlockwise from +x, and ``amplitude`` is the distance from the origin
    in standard-deviation units.
    """
    coeff_filtered = np.asarray(coeff_filtered, dtype=np.complex128)
    if coeff_filtered.ndim != 2:
        raise ValueError("filtered coefficients must be 2-D")
    nt, nk = coeff_filtered.shape
    if nk < 3:
        raise ValueError("need at least wavenumbers 1 and 2")
    nlon = int(nlon)
    if nlon < 360:
        raise ValueError("reconstruction grid is too coarse")

    grid_index = np.arange(nlon)
    basis = np.exp(2j * np.pi * np.arange(1, nk)[:, None]
                   * grid_index[None, :] / nlon)
    # Real VP200 field: the negative-zonal modes are the conjugate of these.
    vp = 2.0 * np.real(coeff_filtered[:, 1:] @ basis)

    # The sorted ECMWF longitude grid starts at 180W, i.e. 180E in [0, 360).
    lon_deg = (np.arange(nlon) * 360.0 / nlon + 180.0) % 360.0
    e1 = eofs.eval_eof(lon_deg, eofs.EOF1_FOURIER if eof1 is None else eof1)
    e2 = eofs.eval_eof(lon_deg, eofs.EOF2_FOURIER if eof2 is None else eof2)
    pc1 = vp @ e1 / np.sum(e1 * e1)
    pc2 = vp @ e2 / np.sum(e2 * e2)

    s1 = float(np.std(pc1, ddof=1)) if nt > 1 else 0.0
    s2 = float(np.std(pc2, ddof=1)) if nt > 1 else 0.0
    if not np.isfinite(s1 + s2) or min(s1, s2) <= 0.0:
        raise ValueError("EOF-projected forecast PCs are zero or invalid")
    x = (pc1 - np.mean(pc1)) / s1
    y = (pc2 - np.mean(pc2)) / s2
    phase_deg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    amp = np.hypot(x, y)
    return x, y, phase_deg, amp, (s1, s2)


def pchip_curve(values, samples=16):
    """Shape-preserving PCHIP interpolation of one 1-D series.

    ``samples`` points are produced per interval; the final sample is
    appended, so the curve passes exactly through every input point.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return np.empty(0)
    if n < 3:
        return np.repeat(v, samples)
    delta = np.diff(v)
    slope = np.zeros(n, dtype=np.float64)
    for q in range(1, n - 1):
        if delta[q - 1] * delta[q] > 0:
            slope[q] = (2.0 * delta[q - 1] * delta[q]
                        / (delta[q - 1] + delta[q]))
    slope[0] = 0.5 * (3 * delta[0] - delta[1])
    if slope[0] * delta[0] <= 0:
        slope[0] = 0.0
    elif abs(slope[0]) > 3 * abs(delta[0]):
        slope[0] = 3 * delta[0]
    slope[-1] = 0.5 * (3 * delta[-1] - delta[-2])
    if slope[-1] * delta[-1] <= 0:
        slope[-1] = 0.0
    elif abs(slope[-1]) > 3 * abs(delta[-1]):
        slope[-1] = 3 * delta[-1]

    u = np.linspace(0.0, 1.0, samples, endpoint=False)
    u2, u3 = u * u, u * u * u
    h00, h10 = 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u
    h01, h11 = -2 * u3 + 3 * u2, u3 - u2
    chunks = [
        h00 * v[q] + h10 * slope[q] + h01 * v[q + 1] + h11 * slope[q + 1]
        for q in range(n - 1)
    ]
    return np.concatenate(chunks + [v[-1:]])
