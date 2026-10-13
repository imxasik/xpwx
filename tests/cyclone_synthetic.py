"""Synthetic NCEP-format daily fields with a known tropical cyclone (TEST ONLY).

A cyclonic Gaussian vortex (streamfunction), a warm core, a low-level height minimum,
a known tilt with height, a known motion and a known environmental zonal wind.

Vertical motion: NOAA PSL publishes no daily omega file, so the vertical velocity
is diagnosed from the winds (core.omega_from_divergence). To test that step, the
truth omega  w(r, p) = W0 * exp(-r^2 / 2 SIGMA^2) * f(p)  is built in, with f(10 hPa) = 0
(omega = 0 at the top). Its divergence -dw/dp is then supplied by an axisymmetric
radial wind (the solution of (1/r) d(r u_r)/dr = D). The truth omega is written to JSON.

Only uwnd, vwnd, air and hgt are written, the same files NOAA publishes.

Environment variables:
  OUT          output directory (default /tmp/psl_serve/Datasets/ncep)
  TRUTH        truth JSON path (default /tmp/cyclone_truth.json)
  TILT_SCALE   multiplies the tilt (0 = no tilt)            default 1
  ENV_SHEAR    multiplies the environmental shear (8 m/s)   default 1
"""
import datetime as dt
import json
import os

import numpy as np
import netCDF4

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cyclone import core  # noqa: E402

TILT_SCALE = float(os.environ.get("TILT_SCALE", "1"))
ENV_SHEAR = float(os.environ.get("ENV_SHEAR", "1"))
OUT = os.environ.get("OUT", "/tmp/psl_serve/Datasets/ncep")
TRUTH = os.environ.get("TRUTH", "/tmp/cyclone_truth.json")

LEVELS = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10], np.float32)
LAT = np.linspace(90, -90, 73, dtype=np.float32)
LON = np.arange(0, 360, 2.5, dtype=np.float32)
EPOCH = dt.date(1800, 1, 1)
SIGMA = 400.0            # km, vortex radius scale (synthetic)
A0 = 1.65e7              # m2 s-1 streamfunction amplitude; max wind ~25 m/s at r = SIGMA
W0 = -0.08               # Pa s-1, truth omega amplitude (ascent)
P_TOP = 10.0             # hPa, top level where omega = 0

STORMS = {
    2019: dict(name="synthetic", date=dt.date(2019, 5, 2), c0=(16.5, 84.5), bearing=350.0, speed_kt=8.0),
}


def profile_A(p):
    return A0 * np.clip(1.0 - 0.5 * (1000.0 - p) / 800.0, 0.4, 1.0)


def env_u(p):
    """Environmental zonal wind (m/s): -3 at 850 hPa, +5 at 200 hPa for ENV_SHEAR = 1 (mean 1.0 m/s)."""
    return -3.0 + ENV_SHEAR * 8.0 * (850.0 - p) / 650.0


def tilt_km(p):
    f = np.clip((1000.0 - p) / 800.0, 0.0, 1.0)
    return TILT_SCALE * -200.0 * f, TILT_SCALE * 120.0 * f     # (east, north) offset in km


def omega_f(p):
    """Vertical profile f(p) of the truth omega; f(P_TOP) = 0."""
    g = lambda q: np.exp(-((q - 450.0) / 250.0) ** 2)
    return g(p) - g(P_TOP)


def omega_f_prime(p):
    """df/dp with p in hPa."""
    return np.exp(-((p - 450.0) / 250.0) ** 2) * (-2.0 * (p - 450.0) / 250.0 ** 2)


def make_year(year, spec):
    days = (dt.date(year, 6, 1) - dt.date(year, 1, 1)).days
    d0 = (dt.date(year, 1, 1) - EPOCH).days
    times = np.array([(d0 + i) * 24.0 for i in range(days)])
    D = spec["date"]
    dist_km_per_day = spec["speed_kt"] * core.KT * 86400.0 / 1000.0
    b = np.deg2rad(spec["bearing"])
    vx, vy = dist_km_per_day * np.sin(b), dist_km_per_day * np.cos(b)

    p = LEVELS[:, None, None].astype(float)
    shape = (days, LEVELS.size, LAT.size, LON.size)
    U = np.zeros(shape, np.float32); V = np.zeros(shape, np.float32)
    T = np.zeros(shape, np.float32); Z = np.zeros(shape, np.float32)
    truth = {"centers": {}, "levels": {}, "omega_centre_Pa_s": {}}
    lat0, lon0 = spec["c0"]
    rng = np.random.default_rng(year)
    LA = np.deg2rad(LAT)[:, None]
    LO = np.deg2rad(LON)[None, :]
    sig_m = SIGMA * 1000.0
    for k in range(days):
        date = dt.date(year, 1, 1) + dt.timedelta(days=k)
        dd = (date - D).days
        clat0, clon0 = core.km_to_latlon(lat0, lon0, dd * vx, dd * vy)
        lat_l = np.zeros(LEVELS.size); lon_l = np.zeros(LEVELS.size)
        for i, pl in enumerate(LEVELS):
            tx, ty = tilt_km(float(pl))
            la, lo = core.km_to_latlon(float(clat0), float(clon0), tx, ty)
            lat_l[i], lon_l[i] = la, lo
        if dd == 0:
            truth["centers"][date.isoformat()] = [float(clat0), float(clon0)]
            truth["levels"] = {int(pl): [float(lat_l[i]), float(lon_l[i])] for i, pl in enumerate(LEVELS)}
            truth["omega_centre_Pa_s"] = {int(pl): float(W0 * omega_f(float(pl))) for pl in LEVELS}
        Xk = np.zeros((LEVELS.size, LAT.size, LON.size)); Yk = np.zeros_like(Xk)
        for i in range(LEVELS.size):
            la0 = np.deg2rad(lat_l[i]); lo0 = np.deg2rad(lon_l[i])
            Xk[i] = core.EARTH_R * np.cos(la0) * (LO - lo0) / 1000.0
            Yk[i] = core.EARTH_R * (LA - la0) / 1000.0
        g = np.exp(-(Xk ** 2 + Yk ** 2) / (2 * SIGMA ** 2))
        A = profile_A(p) * np.ones((1, LAT.size, LON.size))
        dgdx = -Xk / SIGMA ** 2 * g * 1e-3          # per metre (km -> m)
        dgdy = -Yk / SIGMA ** 2 * g * 1e-3

        # axisymmetric radial wind whose divergence is D = K(p) g(r), K = -W0 f'(p) / 100 (s-1)
        K = (-W0 * omega_f_prime(p) / 100.0) * np.ones((1, LAT.size, LON.size))
        xm, ym = Xk * 1000.0, Yk * 1000.0
        r2 = np.maximum(xm ** 2 + ym ** 2, 1.0)
        radial = K * sig_m ** 2 * (1.0 - g) / r2     # multiplies (x, y) to give (u, v)
        u_rad, v_rad = radial * xm, radial * ym

        uu = A * dgdy + env_u(p) + u_rad + rng.normal(0, 0.3, size=Xk.shape)   # u = -dpsi/dy, psi = -A g
        vv = -A * dgdx + v_rad + rng.normal(0, 0.3, size=Xk.shape)              # v =  dpsi/dx
        U[k] = uu; V[k] = vv
        warm = 6.0 * g * np.exp(-((p - 350.0) / 150.0) ** 2)
        T[k] = 288.0 * (p / 1000.0) ** 0.19 + warm
        Z[k] = 8000.0 * np.log(1000.0 / p) - 120.0 * g * np.clip(1 - (p - 200.0) / 800.0, 0, 1)
    return times, U, V, T, Z, truth, days


def write(year, spec):
    times, U, V, T, Z, truth, days = make_year(year, spec)
    specs = {"uwnd": (0.01, U), "vwnd": (0.01, V), "air": (0.01, T), "hgt": (1.0, Z)}
    os.makedirs(OUT, exist_ok=True)
    for fname, (scale, arr) in specs.items():
        ds = netCDF4.Dataset(f"{OUT}/{fname}.{year}.nc", "w", format="NETCDF4_CLASSIC")
        ds.createDimension("time", None); ds.createDimension("level", LEVELS.size)
        ds.createDimension("lat", LAT.size); ds.createDimension("lon", LON.size)
        tv = ds.createVariable("time", "f8", ("time",)); tv.units = "hours since 1800-01-01"; tv.calendar = "standard"; tv[:] = times
        lv = ds.createVariable("level", "f4", ("level",)); lv.units = "millibar"; lv[:] = LEVELS
        la = ds.createVariable("lat", "f4", ("lat",)); la.units = "degrees_north"; la[:] = LAT
        lo = ds.createVariable("lon", "f4", ("lon",)); lo.units = "degrees_east"; lo[:] = LON
        v_ = ds.createVariable(fname, "i2", ("time", "level", "lat", "lon"), zlib=True, fill_value=np.int16(32767))
        v_.scale_factor = np.float32(scale); v_.add_offset = np.float32(0.0)
        v_.missing_value = np.int16(32767)
        for i in range(days):
            v_[i] = arr[i].astype(np.float32)
        ds.close()
    return truth


if __name__ == "__main__":
    all_truth = {}
    for year, spec in STORMS.items():
        all_truth[spec["name"]] = dict(spec, date=spec["date"].isoformat(), truth=write(year, spec))
        print("wrote", year, spec["name"], flush=True)
    with open(TRUTH, "w") as f:
        json.dump(all_truth, f, indent=1, default=float)
    print("truth saved", TRUTH)
