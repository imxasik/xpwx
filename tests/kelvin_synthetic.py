"""Build a tiny offline ECMWF open-data mirror for the Kelvin product (TEST ONLY).

Creates GRIB2 messages (regular lat/lon, template 3.0, simple packing 5.0)
holding an analytic eastward Kelvin wave in 200-hPa divergence, plus the
JSON index files that ``ecmwf/engine.py`` expects.  Serve the resulting
directory:

    python tests/kelvin_synthetic.py
    # -> /tmp/kelvin_mirror/forecasts/<date>/<hh>z/ifs/0p25/oper/...

    (cd /tmp/kelvin_mirror && python -m http.server 8766)

The test ``tests/check_kelvin.py`` does all of that automatically.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Analytic wave parameters.
AMPLITUDE = 1.0e5          # m^2/s, velocity-potential amplitude
PHI0_DEG = 10.0            # equatorial trapping scale
WAVENUMBER = 2
PERIOD_DAYS = 10.0
STEP_HOURS = 24
MAX_LEAD_HOURS = 360

NI, NJ = 360, 181          # 1-degree grid (enough for the physics guards)
LON0 = 0.0


def _signed16(value):
    value = int(value)
    return (abs(value) | 0x8000) if value < 0 else value


def _pack_bits(values, nbits):
    values = np.asarray(values, dtype=np.uint32)
    raw = values.astype(">u4").view(np.uint8).reshape(-1, 4)
    bits = np.unpackbits(raw, axis=1)[:, -int(nbits):]
    return np.packbits(bits.ravel()).tobytes()


def encode_grib(values, lats, lons, nbits=20):
    """One GRIB2 message, grid template 3.0, product 4.0, packing 5.0."""
    values = np.asarray(values, dtype=np.float64)
    nj, ni = values.shape
    npts = ni * nj
    if int(round(abs(lats[0] - lats[-1]) / abs(lats[1] - lats[0]))) + 1 != nj:
        raise ValueError("latitude vector does not match the grid")
    if lons.size != ni:
        raise ValueError("longitude vector does not match the grid")
    lat1, lat2 = float(lats[0]), float(lats[-1])
    lon1 = float(lons[0])
    di = float(abs(lons[1] - lons[0]))
    dj = float(abs(lats[1] - lats[0]))

    rmin = float(np.nanmin(values))
    R = rmin - 1e-9 if rmin < 0 else 0.0
    E = -34
    scale = 2.0 ** (-E)
    ints = np.rint((values - R) * scale).astype(np.int64)
    if ints.min() < 0 or ints.max() >= (1 << nbits):
        raise OverflowError("field does not fit the chosen packing scale")
    packed = _pack_bits(ints, nbits)

    def i32(v, signed=False):
        return int(v).to_bytes(4, "big", signed=signed)

    sec1 = (i32(21) + bytes([1]) + i32(34)[2:] + i32(0)[2:]
            + bytes([2, 1, 1]) + (2026).to_bytes(2, "big")
            + bytes([10, 17, 0, 0, 0, 0, 1]))
    s3 = bytearray()
    s3 += i32(72) + bytes([3, 0]) + i32(npts) + bytes([0, 0]) + i32(0)[2:]
    s3 += bytes([6, 0]) + i32(0) + bytes([0]) + i32(0) + bytes([0]) + i32(0)
    s3 += i32(ni) + i32(nj) + i32(0) + i32(0)
    s3 += i32(round(lat1 * 1e6), True) + i32(round(lon1 * 1e6))
    s3 += bytes([0]) + i32(round(lat2 * 1e6), True) + i32(round((lon1 + (ni - 1) * di) * 1e6))
    s3 += i32(round(di * 1e6)) + i32(round(dj * 1e6)) + bytes([0])
    if len(s3) != 72:
        raise AssertionError(f"section 3 length {len(s3)}")

    s4 = (i32(34) + bytes([4]) + i32(0)[2:] + i32(0)[2:] + bytes(25))
    s5 = (i32(20) + bytes([5]) + i32(npts) + i32(0)[2:]
          + struct.pack(">f", R) + _signed16(E).to_bytes(2, "big")
          + _signed16(0).to_bytes(2, "big") + bytes([nbits]))
    s6 = i32(6) + bytes([6, 255])
    s7 = i32(5 + len(packed)) + bytes([7]) + packed
    body = sec1 + bytes(s3) + s4 + s5 + s6 + s7
    total = 16 + len(body) + 4
    return b"GRIB" + bytes([0, 0, 0, 2]) + total.to_bytes(8, "big") + body + b"7777"


def analytic_divergence(step_hours):
    """d = Laplacian(chi) for chi = A exp(-(phi/phi0)^2) cos(k lon - omega t)."""
    lat = np.linspace(90.0, -90.0, NJ)
    lon = np.linspace(LON0, LON0 + (NI - 1) * (360.0 / NI), NI)
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]
    phi0 = np.deg2rad(PHI0_DEG)
    F = np.exp(-(phi / phi0) ** 2)
    F1 = -2.0 * phi / phi0 ** 2 * F
    F2 = (4.0 * phi ** 2 / phi0 ** 4 - 2.0 / phi0 ** 2) * F
    a = 6_371_000.0
    omega = 2.0 * np.pi / (PERIOD_DAYS * 24.0)     # radians per hour
    phase = WAVENUMBER * lam - omega * step_hours
    with np.errstate(divide="ignore", invalid="ignore"):
        lat_term = F2 - np.tan(phi) * F1
        d = (AMPLITUDE / (a * a)) * (
            -(WAVENUMBER ** 2) * F / np.cos(phi) ** 2 + lat_term
        ) * np.cos(phase)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def write_mirror(root, run_dt, steps=None):
    """Write one run under ``root/forecasts/...`` and return the run tuple."""
    steps = list(range(0, MAX_LEAD_HOURS + 1, STEP_HOURS)) if steps is None else list(steps)
    lat = np.linspace(90.0, -90.0, NJ)
    lon = np.linspace(LON0, LON0 + (NI - 1) * (360.0 / NI), NI)
    run = (run_dt.strftime("%Y%m%d"), f"{run_dt.hour:02d}")
    d = os.path.join(root, "forecasts", run[0], f"{run[1]}z", "ifs", "0p25", "oper")
    os.makedirs(d, exist_ok=True)
    for step in steps:
        grib = encode_grib(analytic_divergence(step), lat, lon)
        base = f"{run[0]}{run[1]}0000-{step}h-oper-fc"
        with open(os.path.join(d, base + ".grib2"), "wb") as fh:
            fh.write(grib)
        entry = {"param": "d", "levtype": "pl", "levelist": "200",
                 "_offset": 0, "_length": len(grib)}
        with open(os.path.join(d, base + ".index"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    return run


def main():
    now = dt.datetime.now(dt.timezone.utc)
    run_dt = (now - dt.timedelta(days=1)).replace(hour=12, minute=0, second=0,
                                                  microsecond=0)
    root = os.environ.get("KELVIN_MIRROR", "/tmp/kelvin_mirror")
    run = write_mirror(root, run_dt)
    print(f"mirror: {root}")
    print(f"run:    {run[0]} {run[1]}Z")
    print("serve:  (cd %s && python -m http.server 8766)" % root)


if __name__ == "__main__":
    main()
