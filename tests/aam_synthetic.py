"""Synthetic NCEP-format daily zonal wind for the AAM products (TEST ONLY).

NOAA PSL is not reachable from an offline sandbox, so this writes the same files
the app downloads — daily ``uwnd.<year>.nc`` and the 1991–2020 daily climatology
``uwnd.day.ltm.1991-2020.nc`` — with a *physically shaped* zonal wind:

  * a zonal-mean climatology: subtropical + polar-night jets with the observed
    annual cycle (strong/ equatorward in winter), tropical trade easterlies
    below ~200 hPa and tropical upper westerlies above,
  * baroclinic transient eddies (zonal wavenumbers 4–8, mid-latitude maximum),
  * an intraseasonal MJO-like wavenumber-1/2 oscillation with a 45-day period
    and opposite signs at 200 and 850 hPa — the driver of the AAM swings,
  * stationary waves (wavenumbers 2–3) in the climatology file.

The absolute values are synthetic; the *structure* is what the AAM integration
needs.  Serve it and point the app at it:

    python tests/aam_synthetic.py
    (cd /tmp/psl_serve && python3 -m http.server 8765)
    PSL_BASE_URL=http://127.0.0.1:8765/Datasets/ncep python tests/check_rel_aam.py

Environment variables:
  OUT   output directory (default /tmp/psl_serve/Datasets/ncep)
  END   last day of the current-year file (default: today − 2 days)
"""
import datetime as dt
import json
import os

import numpy as np
import netCDF4

LEVELS = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200,
                   150, 100, 70, 50, 30, 20, 10], np.float32)
LAT = np.linspace(90, -90, 73, dtype=np.float32)          # NCEP T62: 90 -> -90
LON = np.arange(0, 360, 2.5, dtype=np.float32)
EPOCH = dt.date(1800, 1, 1)
OUT = os.environ.get("OUT", "/tmp/psl_serve/Datasets/ncep")
END = dt.date.fromisoformat(os.environ["END"]) if os.environ.get("END") \
    else dt.date.today() - dt.timedelta(days=2)
N_LTM = 366

LA = LAT.astype(float)[:, None]                 # (nlat, 1)  90 -> -90
LO = LON.astype(float)[None, :]                 # (1, nlon)
P = LEVELS.astype(float)[:, None, None]         # (nlev, 1, 1)
ABS_LAT = np.abs(LA)


# ---------------------------------------------------------------------------
# zonal-mean climatology
# ---------------------------------------------------------------------------
def _jet_p(p, p0=250.0, width=170.0):
    return np.exp(-((p - p0) / width) ** 2)


def _low_p(p):
    """1 at the surface, 0 above ~250 hPa (trade-wind layer)."""
    return np.clip((p - 200.0) / 500.0, 0.0, 1.0)


def _up_p(p):
    return np.clip((300.0 - p) / 200.0, 0.0, 1.0)


def uclim(lat_abs, p, seasonal):
    """Zonal-mean u (m/s); ``seasonal`` = +1 in that hemisphere's winter."""
    u = np.zeros(np.broadcast(lat_abs, p).shape, dtype=float)
    # subtropical jet: stronger and equatorward in winter
    u += (34.0 + 22.0 * seasonal) * np.exp(-((lat_abs - 33.0 - 6.0 * seasonal) / 13.0) ** 2) \
        * _jet_p(p, 240.0, 190.0)
    # polar-front jet / broad extratropical westerlies
    u += (13.0 + 9.0 * seasonal) * np.exp(-((lat_abs - 53.0 - 4.0 * seasonal) / 17.0) ** 2) \
        * _jet_p(p, 420.0, 320.0)
    # weak westerly surface layer in the extratropics
    u += (5.0 + 3.0 * seasonal) * np.exp(-((lat_abs - 45.0) / 22.0) ** 2) * _low_p(p)
    # tropical trade easterlies below ~200 hPa
    u -= 12.0 * np.exp(-(lat_abs / 13.0) ** 2) * _low_p(p)
    # tropical upper-tropospheric westerlies
    u += 9.0 * np.exp(-(lat_abs / 22.0) ** 2) * _up_p(p)
    # stratospheric westerlies
    u += 18.0 * np.exp(-((lat_abs - 55.0) / 18.0) ** 2) * _jet_p(p, 30.0, 60.0)
    return u


def seasonal_of(doy, north=True):
    s = np.cos(2.0 * np.pi * (doy - 15.0) / 365.0)      # +1 around 15 January
    return s if north else -s


def zonal_mean_day(doy):
    """(nlev, nlat) zonal-mean u for one day of year."""
    north = (LA >= 0).astype(float)
    seas = seasonal_of(doy, True) * north + seasonal_of(doy, False) * (1.0 - north)
    # NH amplitude 1.0, SH 0.75 (weaker annual cycle / weaker jets)
    amp = north + 0.75 * (1.0 - north)
    return uclim(ABS_LAT, P, seas) * amp


# ---------------------------------------------------------------------------
# eddies
# ---------------------------------------------------------------------------
def stationary_waves(doy, rng):
    """Wavenumber 2–3 stationary waves (kept in the LTM file)."""
    out = np.zeros((LEVELS.size, LAT.size, LON.size))
    for k, amp in ((2, 6.0), (3, 3.5)):
        prof = _jet_p(P, 300.0, 260.0) * np.exp(-(ABS_LAT / 30.0) ** 2)
        ph = rng.uniform(0, 2 * np.pi)
        out += amp * prof * np.cos(k * np.deg2rad(LO) + ph) \
            * np.sin(np.deg2rad(ABS_LAT) * 2.0)
    return out


def transient_eddies(day_index, doy, rng):
    """Baroclinic eddies: wavenumbers 3–8, mid-latitude maximum, drifting phase.

    The low wavenumbers carry most of the amplitude: only them survive the zonal
    integral, which is what makes the AAM of a latitude band swing.
    """
    out = np.zeros((LEVELS.size, LAT.size, LON.size))
    prof = _jet_p(P, 350.0, 300.0) * np.exp(-((ABS_LAT - 45.0) / 22.0) ** 2)
    for k in (3, 4, 5, 6, 7, 8):
        amp = 26.0 / (k - 1.0) ** 1.15
        ph = 2.0 * np.pi * ((day_index * (0.9 + 0.35 * (k - 3))) % 9.0) / 9.0 \
            + rng.normal(0, 0.25)
        out += amp * prof * np.cos(k * np.deg2rad(LO) - ph)
    return out + rng.normal(0.0, 1.2, size=out.shape)


def intraseasonal(day_index, doy):
    """MJO-like wavenumber 1/2 oscillation, 45-day period, opposite in the
    upper and lower tropical troposphere — the main sub-seasonal AAM driver."""
    period = 45.0
    ph = 2.0 * np.pi * day_index / period
    prof_up = _up_p(P) * np.exp(-(ABS_LAT / 16.0) ** 2)
    prof_lo = _low_p(P) * np.exp(-(ABS_LAT / 16.0) ** 2)
    wave = (np.cos(np.deg2rad(LO) - ph) + 0.6 * np.cos(2 * np.deg2rad(LO) - 1.6 * ph))
    return 7.5 * wave * prof_up - 6.0 * wave * prof_lo


def regional_blocking(day_index, idx_nh, idx_sh):
    """Slow, zonally asymmetric, hemispherically opposite mid-latitude anomaly.

    Gives the large red/blue patches of the latitude–time section: it is strong
    in one hemisphere's storm track and of the opposite sign in the other.
    """
    ph = 2.0 * np.pi * (day_index / 26.0) % (2.0 * np.pi)
    prof = _jet_p(P, 320.0, 280.0) * np.exp(-((ABS_LAT - 47.0) / 12.0) ** 2)
    nh = np.where(LA >= 0, 1.0, 0.0)
    wave_nh = np.cos(np.deg2rad(LO) - ph) + 0.5 * np.cos(2.0 * np.deg2rad(LO) - ph)
    wave_sh = np.cos(np.deg2rad(LO) - ph + 2.1) + 0.5 * np.cos(
        2.0 * np.deg2rad(LO) - ph + 2.1)
    shape = wave_nh * nh + wave_sh * (1.0 - nh)
    return prof * shape * (16.0 * idx_nh - 14.0 * idx_sh)


AR_ZONAL_NH = 0.94        # persistence of the NH zonal-mean index  (≈17 days)
AR_ZONAL_SH = 0.90        # persistence of the SH zonal-mean index  (≈10 days)
AR_BLOCK = 0.93           # persistence of the regional blocking indices
AR_SYNOPTIC = 0.90        # persistence of the transient index


def unit_ar1(prev, ar, rng):
    """One AR(1) step, normalised to unit variance: returns (raw, index)."""
    raw = ar * prev + rng.normal(0.0, 1.0)
    return raw, raw * np.sqrt(1.0 - ar * ar)


def zonal_mean_anomaly(day_index, nh, sh):
    """Zonal-mean wind anomaly (m/s) — the structure the AAM section shows.

    Only the zonal-mean wind survives the longitude integral, so this is what
    the latitude–time section draws.  Three bands, each with its own time scale
    and a slow meridional migration, the hemispheres partly out of phase: it
    gives the drifting red/blue stripes of the real chart, while the global
    AAM anomaly stays a small residual (~±2 Hadley units).
    """
    t = day_index
    mig = np.sin(2.0 * np.pi * t / 61.0)                     # slow migration
    mask_nh = (LA >= 0.0).astype(float)
    mask_sh = 1.0 - mask_nh
    cn, cs = 34.0 + 9.0 * mig, 30.0 - 9.0 * mig
    # jet-shift dipole: the anomaly is positive where the jet is displaced to and
    # negative on the other side — bold shading per latitude, small global residual
    def dipole(centre):
        return (np.exp(-((ABS_LAT - centre) / 8.0) ** 2)
                - 0.70 * np.exp(-((ABS_LAT - (centre + 12.0)) / 8.0) ** 2))
    jet = (10.0 * nh * dipole(cn) * mask_nh
           + 10.0 * sh * dipole(cs) * mask_sh) * _jet_p(P, 320.0, 300.0)
    storm = (6.0 * np.sin(2.0 * np.pi * t / 28.0 + 0.6) * mask_nh
             + 6.0 * np.sin(2.0 * np.pi * t / 28.0 + 2.9) * mask_sh) \
        * np.exp(-((ABS_LAT - 56.0) / 9.0) ** 2) * _jet_p(P, 450.0, 300.0)
    trop = (5.5 * np.sin(2.0 * np.pi * t / 45.0) * np.exp(-(ABS_LAT / 17.0) ** 2)
            * _up_p(P)
            - 4.0 * np.sin(2.0 * np.pi * t / 45.0 - 1.1)
            * np.exp(-(ABS_LAT / 13.0) ** 2) * _low_p(P))
    return jet + storm + trop



# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def _new_file(path, var="uwnd"):
    ds = netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC")
    ds.createDimension("time", None)
    ds.createDimension("level", LEVELS.size)
    ds.createDimension("lat", LAT.size)
    ds.createDimension("lon", LON.size)
    tv = ds.createVariable("time", "f8", ("time",))
    tv.units = "hours since 1800-01-01"
    tv.calendar = "standard"
    lv = ds.createVariable("level", "f4", ("level",))
    lv.units = "millibar"
    lv[:] = LEVELS
    la = ds.createVariable("lat", "f4", ("lat",))
    la.units = "degrees_north"
    la[:] = LAT
    lo = ds.createVariable("lon", "f4", ("lon",))
    lo.units = "degrees_east"
    lo[:] = LON
    v = ds.createVariable(var, "i2", ("time", "level", "lat", "lon"),
                          zlib=True, fill_value=np.int16(32767))
    v.scale_factor = np.float32(0.01)
    v.add_offset = np.float32(0.0)
    v.missing_value = np.int16(32767)
    v.units = "m/s"
    return ds, tv, v


def write_daily(path, days):
    """One file with one record per date in ``days`` (chronological)."""
    ds, tv, v = _new_file(path)
    rng = np.random.default_rng(20260101)
    raw = {"nh": 0.0, "sh": 0.0, "bnh": 0.0, "bsh": 0.0}     # AR(1) states
    for i, d in enumerate(days):
        doy = d.timetuple().tm_yday
        raw["nh"], w = unit_ar1(raw["nh"], AR_ZONAL_NH, rng)
        raw["sh"], w2 = unit_ar1(raw["sh"], AR_ZONAL_SH, rng)
        raw["bnh"], bnh = unit_ar1(raw["bnh"], AR_BLOCK, rng)
        raw["bsh"], bsh = unit_ar1(raw["bsh"], AR_BLOCK, rng)
        u = (zonal_mean_day(doy)
             + stationary_waves(doy, np.random.default_rng(7 + doy))
             + transient_eddies(i, doy, rng)
             + intraseasonal(i, doy)
             + regional_blocking(i, bnh, bsh)
             + zonal_mean_anomaly(i, w, w2))
        tv[i] = (d - EPOCH).days * 24.0
        v[i] = np.clip(u, -300.0, 300.0).astype(np.float32)   # netCDF4 packs it
    ds.close()
    return len(days)


def write_ltm(path):
    """The 1991–2020 daily climatology: zonal mean + stationary waves."""
    ds, tv, v = _new_file(path)
    base = dt.date(2001, 1, 1)                       # any year with 366 slots
    for k in range(N_LTM):
        d = base + dt.timedelta(days=k) if k < 365 else base + dt.timedelta(days=364)
        doy = min(k + 1, 366)
        u = (zonal_mean_day(doy)
             + stationary_waves(doy, np.random.default_rng(7 + (doy % 365))))
        tv[k] = k * 24.0
        v[k] = np.clip(u, -300.0, 300.0).astype(np.float32)   # netCDF4 packs it
    ds.close()
    return N_LTM


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    year = END.year
    start = END - dt.timedelta(days=119)              # exactly a 120-day window
    if start.year != year:                            # keep it inside one file
        start = dt.date(year, 1, 1)
    days = [start + dt.timedelta(days=i) for i in range((END - start).days + 1)]
    n = write_daily(f"{OUT}/uwnd.{year}.nc", days)
    print(f"wrote uwnd.{year}.nc  {days[0]} → {days[-1]}  ({n} days)", flush=True)

    # full previous years, so a 120-day window ending in December really has to
    # be stitched together from two files (as it is with the real PSL files)
    for y in (year - 1, year - 2):
        prev = [dt.date(y, 1, 1) + dt.timedelta(days=i)
                for i in range((dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days)]
        n = write_daily(f"{OUT}/uwnd.{y}.nc", prev)
        print(f"wrote uwnd.{y}.nc  {prev[0]} → {prev[-1]}  ({n} days)", flush=True)

    n = write_ltm(f"{OUT}/uwnd.day.ltm.1991-2020.nc")
    print(f"wrote uwnd.day.ltm.1991-2020.nc  ({n} days)", flush=True)

    with open(os.environ.get("TRUTH", "/tmp/aam_truth.json"), "w") as fh:
        json.dump({"year": year, "end": END.isoformat(),
                   "start": start.isoformat(), "n_days": len(days),
                   "levels": LEVELS.tolist(), "lat": LAT.tolist(),
                   "lon": LON.tolist()}, fh, indent=1)
    print("truth saved", os.environ.get("TRUTH", "/tmp/aam_truth.json"))
