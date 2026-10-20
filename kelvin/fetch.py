"""kelvin.fetch — pick an IFS run, fetch d200 messages, cache VP coefficients.

The acquisition layer deliberately reuses the project's ECMWF engine
(``ecmwf/engine.py``): the same polite HTTP client, the same JSON index
parser and the same pure-Python GRIB2/CCSDS decoder that already serve the
ECMWF Forecast maps.  No second copy of that machinery lives here.

Run policy
----------
Only the 00Z and 12Z IFS cycles are used: they carry the full 15-day
(360 h) open-data horizon.  The 06Z/18Z cycles stop at 144 h, which is too
short for a phase-space trajectory.  ``Auto`` takes the newest *complete*
00/12Z run; a manual date takes the newest complete 00/12Z run on or before
that date.  ECMWF Open Data keeps only the latest ~12 runs (~2-3 days), so
older manual dates raise a clear error instead of silently substituting a
different run.

The divergence fields are daily (24 h) samples by default.  This keeps the
first cold request inside the web host's timeout while still resolving the
2.5-20 day Kelvin band (Nyquist = 2 days).  Set ``KELVIN_STEP_HOURS=12`` on a
fast host to sample like the reference script; everything downstream adapts
automatically.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys
import threading
import time

import numpy as np

from . import physics

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _step_hours():
    """Sampling interval, in hours.

    The default is 24 h: 16 daily fields keep the first cold render inside the
    web host's timeout and still resolve the 2.5-20 day Kelvin band (Nyquist =
    2 days).  Set ``KELVIN_STEP_HOURS=12`` on a fast host to sample like the
    reference script (31 fields).
    """
    try:
        value = int(os.environ.get("KELVIN_STEP_HOURS", "24"))
    except ValueError:
        value = 24
    return value if value in (12, 24) else 24


STEP_HOURS = _step_hours()
MAX_LEAD_HOURS = 360
MIN_STEPS = 10                    # samples, for a meaningful FFT
MIN_SPAN_HOURS = 240             # 10 days of contiguous coverage
LONG_RUN_HOURS = (12, 0)          # cycles with the 360 h horizon
RUN_LOOKBACK_DAYS = 3             # the open-data rolling archive is ~12 runs
COEFF_VERSION = 1
RUN_CACHE_TTL = 10 * 60           # seconds

class KelvinDataError(RuntimeError):
    """User-facing data/acquisition failure."""


_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_COEFF_LOCK = threading.Lock()
_RUN_CACHE = {}


def engine():
    """Return the shared ``ecmwf/engine.py`` module, loading it once."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        mod = sys.modules.get("ecmwf_engine")
        if mod is None:
            path = os.path.join(_ROOT, "ecmwf", "engine.py")
            spec = importlib.util.spec_from_file_location("ecmwf_engine", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["ecmwf_engine"] = mod
            spec.loader.exec_module(mod)
        _ENGINE = mod
        return mod


def run_datetime(run):
    return dt.datetime.strptime(f"{run[0]}{int(run[1]):02d}",
                                "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)


def run_label(run):
    return run_datetime(run).strftime("%Y-%m-%d %HZ")


def _log(message):
    print(message, flush=True)


def _find_d200(index, step=None):
    """First 200-hPa divergence entry in one index file.

    ``step`` filters entries that carry a ``step`` key (the real ECMWF index
    does); the synthetic test indexes omit it.
    """
    for entry in index or []:
        if step is not None and entry.get("step") not in (None, str(step)):
            continue
        if (entry.get("param") == "d"
                and str(entry.get("levelist")) == "200"
                and entry.get("levtype") in (None, "pl")):
            return entry
    return None


def _step_available(run, step):
    idx = engine().load_index(run, step)
    return idx is not None and _find_d200(idx, step) is not None


def _candidate_runs(now, date=None):
    """Yield ``(YYYYMMDD, HH)`` runs newest first, long cycles only."""
    if date is not None:
        for back in range(RUN_LOOKBACK_DAYS + 1):
            day = date - dt.timedelta(days=back)
            for hh in LONG_RUN_HOURS:
                rdt = dt.datetime(day.year, day.month, day.day, hh,
                                  tzinfo=dt.timezone.utc)
                if rdt <= now:
                    yield rdt.strftime("%Y%m%d"), f"{hh:02d}"
    else:
        for back in range(RUN_LOOKBACK_DAYS + 1):
            day = now.date() - dt.timedelta(days=back)
            for hh in LONG_RUN_HOURS:
                rdt = dt.datetime(day.year, day.month, day.day, hh,
                                  tzinfo=dt.timezone.utc)
                if rdt <= now:
                    yield rdt.strftime("%Y%m%d"), f"{hh:02d}"


def _full_steps():
    return list(range(0, MAX_LEAD_HOURS + 1, STEP_HOURS))


def _discover_steps(run):
    """Contiguous available steps from +0 h until the first missing one."""
    steps = []
    for step in _full_steps():
        if not _step_available(run, step):
            break
        steps.append(step)
    return steps


def select_run(mode="auto", date=None, now=None):
    """Return ``(run, steps, step_hours)``; raise ``KelvinDataError`` if none."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if date is not None and not isinstance(date, dt.date):
        raise TypeError("date must be a datetime.date")
    key = (str(mode), date.isoformat() if date else None)
    hit = _RUN_CACHE.get(key)
    if hit and (time.time() - hit[0]) < RUN_CACHE_TTL:
        return hit[1]

    candidates = list(_candidate_runs(now, date if mode == "manual" else None))
    if not candidates:
        raise KelvinDataError(
            "No 00Z/12Z IFS run is old enough to be published yet. "
            "Try again in a few hours, or use a manual date."
        )

    steps = _full_steps()
    last = steps[-1]
    # 1) A complete 15-day run is what the product wants.  Every step's index
    #    is probed here: a single missing intermediate step would otherwise
    #    abort the FFT later, after most of the downloads were already paid
    #    for.  These index files are disk-cached, so fetch_coefficients
    #    reuses them instead of issuing a second round of requests.
    for run in candidates:
        try:
            if not _step_available(run, 0):
                continue
            if not _step_available(run, last):
                continue
            if not all(_step_available(run, s) for s in steps[1:-1]):
                continue
        except Exception:                                  # noqa: BLE001
            continue
        result = (run, steps, STEP_HOURS)
        _RUN_CACHE[key] = (time.time(), result)
        _log(f"  IFS run selected: {run_label(run)} (+0..{last} h, "
             f"{len(steps)} daily fields)")
        return result

    # 2) A run that is still being published (or a manual date whose only
    #    available cycle is incomplete): use the contiguous part, if long
    #    enough for a meaningful FFT.
    for run in candidates:
        try:
            if not _step_available(run, 0):
                continue
            avail = _discover_steps(run)
        except Exception:                                  # noqa: BLE001
            continue
        if len(avail) >= MIN_STEPS and (avail[-1] - avail[0]) >= MIN_SPAN_HOURS:
            result = (run, avail, STEP_HOURS)
            _RUN_CACHE[key] = (time.time(), result)
            _log(f"  IFS run selected: {run_label(run)} "
                 f"(+0..{avail[-1]} h, {len(avail)} daily fields; run still publishing)")
            return result

    where = f" on or before {date.isoformat()}" if date else ""
    raise KelvinDataError(
        f"No usable ECMWF IFS 00Z/12Z run with 200-hPa divergence was found"
        f"{where}. ECMWF Open Data keeps only the latest ~12 runs (2-3 days), "
        "so choose Auto or a more recent date. If the date is recent, the "
        "ECMWF server may be unreachable from this host."
    )


def fetch_field(run, step):
    """Download/read and decode one d200 message."""
    eng = engine()
    idx = eng.load_index(run, step)
    if idx is None:
        raise KelvinDataError(f"index for +{step} h is not published")
    entry = _find_d200(idx, step)
    if entry is None:
        raise KelvinDataError(f"no 200-hPa divergence field at +{step} h")

    path = eng.msg_cache_path(run, step, "d", 200)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                buf = fh.read()
            return eng.decode_grib2_message(buf)
        except Exception:                                  # noqa: BLE001
            try:
                os.remove(path)
            except OSError:
                pass

    start = int(entry["_offset"])
    length = int(entry["_length"])
    try:
        buf = eng.HTTP.get(eng.grib_url(run, step),
                           byte_range=(start, start + length - 1))
    except FileNotFoundError as exc:
        raise KelvinDataError(
            f"the +{step} h GRIB file disappeared from the ECMWF server"
        ) from exc
    except Exception as exc:                               # noqa: BLE001
        raise KelvinDataError(
            f"ECMWF download failed at +{step} h: {exc}"
        ) from exc
    if len(buf) != length or buf[:4] != b"GRIB":
        raise KelvinDataError(
            f"bad range response at +{step} h "
            f"({len(buf)} bytes, expected {length})"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf)
    os.replace(tmp, path)
    return eng.decode_grib2_message(buf)


def _coeff_dir():
    return os.path.join(engine().CACHE_DIR, "kelvin")


def _coeff_path(run, step):
    # The run is part of the name: forecasts are re-issued twice a day, and a
    # "+120 h" coefficient from yesterday's run must never satisfy today's.
    return os.path.join(_coeff_dir(),
                        f"vp_v{COEFF_VERSION}_d200_"
                        f"{run[0]}{str(run[1]).zfill(2)}_{int(step):03d}.npz")


def _load_coeff(run, step):
    path = _coeff_path(run, step)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            coeff = np.asarray(data["coeff"], dtype=np.complex128)
            nlon = int(data["nlon"])
        if coeff.shape != (physics.K_WAVENUMBER_MAX + 1,):
            return None
        if nlon < 360:
            return None
        return coeff, nlon
    except Exception:                                      # noqa: BLE001
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _save_coeff(run, step, coeff, nlon):
    os.makedirs(_coeff_dir(), exist_ok=True)
    path = _coeff_path(run, step)
    tmp = f"{path}.tmp.npz"
    np.savez(tmp, coeff=np.asarray(coeff, dtype=np.complex128), nlon=int(nlon))
    os.replace(tmp, path)


def fetch_coefficients(run, steps, step_hours=STEP_HOURS):
    """Return ``(coeff[nt, kmax+1], nlon)`` for one forecast run.

    Per-step coefficients are cached on disk, so a second render of the same
    run (or a restart of the Flask process) does not download or decode the
    GRIB messages again.
    """
    if not steps:
        raise KelvinDataError("the selected IFS run has no forecast steps")
    coeffs = []
    nlon = None
    with _COEFF_LOCK:
        for i, step in enumerate(steps, 1):
            cached = _load_coeff(run, step)
            if cached is not None:
                coeff, this_nlon = cached
                _log(f"  VP coefficients {i}/{len(steps)} (+{step:03d} h, cache)")
            else:
                try:
                    field = fetch_field(run, step)
                    coeff = physics.vp_equatorial_coefficients(
                        field.values, field.lats)
                    this_nlon = int(field.values.shape[1])
                    _save_coeff(run, step, coeff, this_nlon)
                except KelvinDataError:
                    raise
                except Exception as exc:                   # noqa: BLE001
                    raise KelvinDataError(
                        f"could not build VP coefficients at +{step} h: {exc}"
                    ) from exc
                _log(f"  VP coefficients {i}/{len(steps)} (+{step:03d} h, fetched)")
            if nlon is None:
                nlon = this_nlon
            elif this_nlon != nlon:
                raise KelvinDataError(
                    "the IFS grid changed inside the forecast sequence "
                    f"({nlon} vs {this_nlon} longitudes)"
                )
            coeffs.append(coeff)
    if nlon is None:
        raise KelvinDataError("no IFS fields were decoded")
    return np.asarray(coeffs, dtype=np.complex128), nlon
