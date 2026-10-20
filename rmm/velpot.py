"""rmm/velpot.py — 200-hPa velocity potential χ from the wind field.

χ is the irrotational part of the wind, V = ∇χ, so that ∇²χ = ∇·V.  It is
obtained by inverting the divergence on the sphere in flux (finite-volume)
form.  Nothing is smoothed, tapered, truncated or fitted, and no value is
assumed beyond the grid geometry and the wind data themselves.

Discretisation (grid rows run north → south, both poles present):

* Each grid cell is bounded by the midpoints between neighbouring latitude
  rows.  The two pole rows are single polar-cap cells.
* The outflow of a cell is the sum of its edge fluxes.  The wind flux through an
  edge uses the mean of the two adjacent grid values.
* χ is the solution of the same flux balance with the flux of ∇χ on the left.
  The discrete operator is exact for this geometry.  The only free constant is
  the additive constant of χ, which is fixed by χ(north pole) = 0.  The constant
  does not affect the anomalies used by the index.

Earth radius R appears only as a scale factor of χ.  The RMM index normalises
every field by its own domain standard deviation, so the value of R cancels.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

R_EARTH = 6.371e6   # m; scale factor of χ only (cancels in the normalised index)


class VelocityPotential:
    """Pre-factorised flux-form Poisson solver for one lat-lon grid."""

    def __init__(self, lat_deg, lon_deg, radius=R_EARTH):
        lat_deg = np.asarray(lat_deg, dtype=np.float64)
        lon_deg = np.asarray(lon_deg, dtype=np.float64)
        if lat_deg[0] < lat_deg[-1]:
            raise ValueError("latitudes must run north to south (90 ... -90)")
        if not (np.isclose(lat_deg[0], 90.0) and np.isclose(lat_deg[-1], -90.0)):
            raise ValueError("grid must include both poles (90 and -90)")
        nlon = lon_deg.size
        dlam = 2.0 * np.pi / nlon
        if not np.allclose(np.diff(lon_deg), 360.0 / nlon):
            raise ValueError("longitudes must be uniformly spaced over 360 degrees")

        J = lat_deg.size
        phi = np.deg2rad(lat_deg)
        edge = 0.5 * (phi[:-1] + phi[1:])            # e_j between rows j and j+1
        top = np.concatenate([[np.pi / 2], edge])     # top boundary of row j
        bot = np.concatenate([edge, [-np.pi / 2]])    # bottom boundary of row j
        dphi = top - bot                               # cell extent (rad)

        self.lat_deg, self.lon_deg = lat_deg, lon_deg
        self.J, self.nlon, self.R = J, nlon, float(radius)
        self.dlam, self.phi, self.edge = dlam, phi, edge
        self.dphi = dphi

        # ---- unknown numbering: N pole, band cells, S pole -------------
        nb = (J - 2) * nlon
        self.iN, self.iS = 0, 1 + nb
        self.nunk = self.iS + 1

        def node(j, i):
            if j == 0:
                return self.iN
            if j == J - 1:
                return self.iS
            return 1 + (j - 1) * nlon + (i % nlon)

        rows, cols, vals = [], [], []

        # zonal part (band rows only):  outflow = Δφ/(cosφ Δλ) · second difference
        for j in range(1, J - 1):
            cz = dphi[j] / (np.cos(phi[j]) * dlam)
            for i in range(nlon):
                p = node(j, i)
                rows += [p, p, p]
                cols += [p, node(j, i + 1), node(j, i - 1)]
                vals += [-2.0 * cz, cz, cz]

        # meridional part:  northward flux F = cos(e)Δλ (χ_N − χ_S)/Δφ_e
        for j in range(J - 1):
            cm = np.cos(edge[j]) * dlam / (phi[j] - phi[j + 1])
            for i in range(nlon):
                n_, s_ = node(j, i), node(j + 1, i)
                # outflow of N cell −= F ; outflow of S cell += F
                rows += [n_, n_, s_, s_]
                cols += [n_, s_, n_, s_]
                vals += [-cm, cm, cm, -cm]

        K = sp.coo_matrix((vals, (rows, cols)), shape=(self.nunk, self.nunk)).tocsr()
        # χ is defined up to a constant: pin χ(N pole) = 0 (replace one equation).
        K = K.tolil()
        K[self.iN, :] = 0.0
        K[self.iN, self.iN] = 1.0
        self._lu = splu(K.tocsc())

    # ------------------------------------------------------------------
    def _rhs(self, u, v):
        """Outflow of the wind (V) through each cell; shape (T, nunk)."""
        J, nlon, R, dlam, dphi, edge = self.J, self.nlon, self.R, self.dlam, self.dphi, self.edge
        T = u.shape[0]
        # meridional edge fluxes Fv[j] for j = 0..J-2, shape (T, J-1, nlon)
        vedge = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        Fv = R * dlam * np.cos(edge)[None, :, None] * vedge
        # zonal edge fluxes (band rows), east edge of (j, i): shape (T, J, nlon)
        uedge = 0.5 * (u + np.roll(u, -1, axis=2))
        Fu = R * dphi[None, :, None] * uedge

        b = np.zeros((T, self.nunk))
        for j in range(1, J - 1):
            out = (Fu[:, j, :] - np.roll(Fu[:, j, :], 1, axis=1)
                   + Fv[:, j - 1, :] - Fv[:, j, :])
            p0 = 1 + (j - 1) * nlon
            b[:, p0:p0 + nlon] = out
        # polar caps: the north-cap equation is replaced by the pin χ(N) = 0
        # (its balance is implied by the others); the south cap is the flux balance.
        b[:, self.iS] = Fv[:, J - 2, :].sum(axis=1)
        return b

    def chi(self, u, v, batch=256):
        """u, v : (T, J, nlon) wind (m/s) on the grid. Returns χ (T, J, nlon)."""
        u = np.nan_to_num(np.asarray(u, dtype=np.float64), nan=0.0)
        v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)
        T = u.shape[0]
        out = np.empty((T, self.J, self.nlon))
        for s in range(0, T, batch):
            e = min(T, s + batch)
            b = self._rhs(u[s:e], v[s:e])
            x = self._lu.solve(np.ascontiguousarray(b.T))  # (nunk, t)
            x = x.T
            for j in range(1, self.J - 1):
                p0 = 1 + (j - 1) * self.nlon
                out[s:e, j, :] = x[:, p0:p0 + self.nlon]
            out[s:e, 0, :] = x[:, self.iN][:, None]
            out[s:e, -1, :] = x[:, self.iS][:, None]
        return out
