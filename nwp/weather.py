from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from eccodes import (
    codes_get,
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
)

# A compact set of global deterministic and published ensemble-mean products.
# All weather fields come from official ECMWF, NOAA/NCEP, ECCC, or DWD
# distributions; ensemble means are never approximated with a single member.
_IFS_STEPS = list(range(0, 145, 3)) + list(range(150, 361, 6))
_AIFS_STEPS = list(range(0, 361, 6))
_GFS_STEPS = list(range(0, 121, 3)) + list(range(126, 385, 6))
_AIGFS_STEPS = list(range(0, 385, 6))
_GEFS_STEPS = list(range(0, 385, 6))
_HGEFS_STEPS = list(range(0, 241, 6))
_GDPS_STEPS = list(range(0, 85)) + list(range(87, 241, 3))
_GEML_STEPS = list(range(0, 241, 6))
_GEPS_STEPS = list(range(0, 193, 3)) + list(range(198, 385, 6))
_ICON_STEPS = list(range(0, 79)) + list(range(81, 181, 3))
_AICON_STEPS = list(range(3, 181, 3))

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "ifs_01": {
        "name": "ECMWF IFS",
        "resolution": "0.1°",
        "source": "ecmwf",
        "source_label": "ECMWF Open Data",
        "steps": _IFS_STEPS,
        "note": "High-resolution deterministic forecast. 0.1° open-data surface fields can be published later than the 0.25° tier.",
        "ecmwf_model": "ifs",
        "resol": "0p1",
        "stream": "oper",
        "type": "fc",
    },
    "ifs_025": {
        "name": "ECMWF IFS",
        "resolution": "0.25°",
        "source": "ecmwf",
        "source_label": "ECMWF Open Data",
        "steps": _IFS_STEPS,
        "note": "Standard open-data grid; offered as a lighter alternative to 0.1°.",
        "ecmwf_model": "ifs",
        "resol": "0p25",
        "stream": "oper",
        "type": "fc",
    },
    "aifs": {
        "name": "ECMWF AIFS",
        "resolution": "0.25°",
        "source": "ecmwf",
        "source_label": "ECMWF Open Data",
        "steps": _AIFS_STEPS,
        "note": "Operational deterministic AI forecast.",
        "ecmwf_model": "aifs-single",
        "resol": "0p25",
        "stream": "oper",
        "type": "fc",
    },
    "aifs_ens_control": {
        "name": "AIFS ENS · Control",
        "resolution": "0.25°",
        "source": "ecmwf",
        "source_label": "ECMWF Open Data",
        "steps": _AIFS_STEPS,
        "note": "Control member only—not the ensemble mean. Kept separate so the map does not misrepresent a single member as a consensus.",
        "ecmwf_model": "aifs-ens",
        "resol": "0p25",
        "stream": "enfo",
        "type": "cf",
    },
    "gfs": {
        "name": "GFS",
        "resolution": "0.25°",
        "source": "noaa_gfs",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _GFS_STEPS,
        "note": "Regional GRIB filter requests only 10 m U/V wind over the Bay of Bengal box.",
    },
    "aigfs": {
        "name": "AIGFS",
        "resolution": "AI",
        "source": "noaa_aigfs",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _AIGFS_STEPS,
        "note": "NCEP AI global forecast; only the two 10 m wind fields are transferred, then cropped to the selected domain.",
    },
    "gefs_mean": {
        "name": "GEFS",
        "resolution": "0.5° ENS",
        "source": "noaa_gefs",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _GEFS_STEPS,
        "note": "Official precomputed ensemble-mean U/V winds; not a single member and no local ensemble averaging is needed.",
    },
    "aigefs_mean": {
        "name": "AIGEFS",
        "resolution": "AI ENS",
        "source": "noaa_aigefs_mean",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _AIGFS_STEPS,
        "note": "Official AI ensemble mean; only indexed 10 m U/V wind messages are transferred.",
    },
    "hgefs_mean": {
        "name": "HGEFS",
        "resolution": "HYBRID ENS",
        "source": "noaa_hgefs_mean",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _HGEFS_STEPS,
        "note": "Official hybrid ensemble mean. Product currently provides the listed shorter lead-hour range.",
    },
    "eccc_gdps": {
        "name": "GDPS",
        "resolution": "15 km",
        "source": "eccc_gdps",
        "source_label": "ECCC MSC Datamart (HPFX)",
        "steps": _GDPS_STEPS,
        "note": "Official global deterministic forecast; downloads only the 10 m U/V GRIB fields, then crops the map to the Bay domain.",
    },
    "eccc_geml": {
        "name": "GEML",
        "resolution": "25 km",
        "source": "eccc_geml",
        "source_label": "ECCC MSC Datamart (HPFX)",
        "steps": _GEML_STEPS,
        "note": "Official global AI forecast; downloads only the 10 m U/V GRIB fields, then crops the map to the Bay domain.",
    },
    "eccc_geps_mean": {
        "name": "GEPS",
        "resolution": "0.5° ENS",
        "source": "eccc_geps_mean",
        "source_label": "ECCC MSC Datamart (HPFX)",
        "steps": _GEPS_STEPS,
        "note": "Official 21-member ensemble mean, computed from all 10 m U/V members; only the two required all-member files are fetched.",
    },
    "dwd_icon": {
        "name": "DWD ICON",
        "resolution": "13 km",
        "source": "dwd_icon",
        "source_label": "DWD Open Data",
        "steps": _ICON_STEPS,
        "note": "Official global ICON 10 m winds; its native unstructured grid is remapped to the Bay domain at 0.1°.",
    },
    "dwd_aicon": {
        "name": "DWD AICON",
        "resolution": "13 km",
        "source": "dwd_aicon",
        "source_label": "DWD Open Data",
        "steps": _AICON_STEPS,
        "note": "Official global AI forecast. DWD ICON grid coordinates are used as specified by DWD; winds are remapped to the Bay domain at 0.1°.",
    },
}

DOMAIN = {"west": 70.0, "east": 110.0, "south": 0.0, "north": 30.0}
FIELD_CACHE_VERSION = 5  # new ECCC/DWD model decoders and regional remapping
CACHE_ROOT = Path(os.environ.get("BAYWIND_CACHE_DIR", "/tmp/baywind-cache"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_TTL = int(os.environ.get("BAYWIND_CACHE_TTL", "5400"))
HTTP_TIMEOUT = (8, int(os.environ.get("BAYWIND_HTTP_TIMEOUT", "45")))
USER_AGENT = "BaywindModelViewer/1.0 (official open model-data visualization)"
_CACHE_LOCK = threading.RLock()
# A shared two-slot pool bounds concurrent downloads across all paired
# non-ECMWF fields in this worker. ECMWF remains one Client.retrieve request.
MAX_PARALLEL_FIELD_DOWNLOADS = 2
_PAIR_DOWNLOAD_POOL = ThreadPoolExecutor(
    max_workers=MAX_PARALLEL_FIELD_DOWNLOADS,
    thread_name_prefix="baywind-field-download",
)
_DOWNLOAD_LOCAL = threading.local()
_PERF_LOG = logging.getLogger("baywind.performance")


def _valid_cache(path: Path) -> bool:
    return path.exists() and (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime < CACHE_TTL)


def _candidate_cycles() -> list[datetime]:
    now = datetime.now(timezone.utc)
    hour = (now.hour // 6) * 6
    latest = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return [latest - timedelta(hours=6 * i) for i in range(8)]


def _cycle_parts(cycle: datetime) -> tuple[str, str]:
    return cycle.strftime("%Y%m%d"), cycle.strftime("%H")


def _eccc_cycle_root(cycle: datetime) -> str:
    """Use the Datamart's current-day alias or its dated retention tree."""
    if cycle.date() == datetime.now(timezone.utc).date():
        return "today"
    return f"{cycle:%Y%m%d}/WXO-DD"


def _request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return session


def _download_session() -> requests.Session:
    """One reusable Requests session per bounded downloader thread."""
    session = getattr(_DOWNLOAD_LOCAL, "session", None)
    if session is None:
        session = _request_session()
        _DOWNLOAD_LOCAL.session = session
    return session


def _download_one_grib(
    url: str,
    headers: dict[str, str] | None = None,
    expected_status: int = 200,
) -> bytes:
    response = _download_session().get(url, headers=headers, timeout=HTTP_TIMEOUT)
    if response.status_code != expected_status or not response.content.startswith(b"GRIB"):
        raise RuntimeError(f"HTTP {response.status_code}; response was not the expected GRIB field")
    return response.content


def _download_pair(
    urls: dict[str, str],
    *,
    headers: dict[str, dict[str, str]] | None = None,
    expected_status: dict[str, int] | None = None,
) -> dict[str, bytes]:
    """Download only a small pair of required fields with a global cap of two."""
    if not urls or len(urls) > MAX_PARALLEL_FIELD_DOWNLOADS:
        raise ValueError("Paired field download must contain one or two required files.")
    headers = headers or {}
    expected_status = expected_status or {}
    futures = {
        key: _PAIR_DOWNLOAD_POOL.submit(
            _download_one_grib,
            url,
            headers.get(key),
            expected_status.get(key, 200),
        )
        for key, url in urls.items()
    }
    payloads: dict[str, bytes] = {}
    errors: list[str] = []
    for key, future in futures.items():
        try:
            payloads[key] = future.result()
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return payloads


def get_forecast(model_id: str, step: int) -> dict[str, Any]:
    if model_id not in MODEL_CATALOG:
        raise ValueError("Unknown model.")
    total_started = time.perf_counter()
    field_cache = CACHE_ROOT / f"fields-v{FIELD_CACHE_VERSION}-{model_id}-f{step:03d}.npz"

    def load_cached() -> dict[str, Any]:
        cache_started = time.perf_counter()
        fields = _load_fields(field_cache)
        fields["_cache_status"] = "hit"
        fields["_timings"] = {
            "field_cache_read_ms": (time.perf_counter() - cache_started) * 1000.0,
            "fetch_ms": 0.0,
            "decode_ms": 0.0,
            "save_ms": 0.0,
        }
        _PERF_LOG.info(
            "fields model=%s step=%03d cache=hit total_ms=%.1f",
            model_id,
            step,
            (time.perf_counter() - total_started) * 1000.0,
        )
        return fields

    if _valid_cache(field_cache):
        return load_cached()

    # Coalesce simultaneous misses and avoid duplicate downloads/decodes.
    with _CACHE_LOCK:
        if _valid_cache(field_cache):
            return load_cached()

        descriptor = MODEL_CATALOG[model_id]
        with tempfile.NamedTemporaryFile(prefix="baywind-", suffix=".grib2", dir=CACHE_ROOT, delete=False) as handle:
            grib_path = Path(handle.name)
        try:
            fetch_started = time.perf_counter()
            if descriptor["source"] == "ecmwf":
                # Deliberately one ECMWF Open Data client request for both
                # required parameters; do not fan out or probe fallback cycles.
                init = _fetch_ecmwf(descriptor, step, grib_path)
            elif descriptor["source"] == "noaa_gfs":
                init = _fetch_gfs(step, grib_path)
            elif descriptor["source"] == "noaa_gefs":
                init = _fetch_gefs_mean(step, grib_path)
            elif descriptor["source"] == "noaa_aigefs_mean":
                init = _fetch_noaa_indexed_mean("aigefs", step, grib_path)
            elif descriptor["source"] == "noaa_hgefs_mean":
                init = _fetch_noaa_indexed_mean("hgefs", step, grib_path)
            elif descriptor["source"] == "noaa_aigfs":
                init = _fetch_aigfs(step, grib_path)
            elif descriptor["source"] in {"eccc_gdps", "eccc_geml"}:
                init = _fetch_eccc_deterministic(descriptor["source"], step, grib_path)
            elif descriptor["source"] == "eccc_geps_mean":
                init = _fetch_eccc_geps_mean(step, grib_path)
            elif descriptor["source"] in {"dwd_icon", "dwd_aicon"}:
                init = _fetch_dwd_global(descriptor["source"], step, grib_path)
            else:
                raise ValueError("This model source is not configured.")
            fetch_ms = (time.perf_counter() - fetch_started) * 1000.0

            decode_started = time.perf_counter()
            if descriptor["source"] in {"dwd_icon", "dwd_aicon"}:
                fields = _decode_dwd_unstructured(grib_path, init)
            elif descriptor["source"] == "eccc_geps_mean":
                fields = _decode_geps_mean(grib_path)
            else:
                fields = _decode_wind(grib_path)
            fields["init"] = init or fields.get("init") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            fields["model_id"] = model_id
            fields["source"] = descriptor["source_label"]
            decode_ms = (time.perf_counter() - decode_started) * 1000.0

            save_started = time.perf_counter()
            _save_fields(field_cache, fields)
            save_ms = (time.perf_counter() - save_started) * 1000.0
            fields["_cache_status"] = "miss"
            fields["_timings"] = {
                "field_cache_read_ms": 0.0,
                "fetch_ms": fetch_ms,
                "decode_ms": decode_ms,
                "save_ms": save_ms,
            }
            _PERF_LOG.info(
                "fields model=%s step=%03d cache=miss fetch_ms=%.1f decode_ms=%.1f save_ms=%.1f total_ms=%.1f",
                model_id,
                step,
                fetch_ms,
                decode_ms,
                save_ms,
                (time.perf_counter() - total_started) * 1000.0,
            )
            return fields
        finally:
            try:
                grib_path.unlink(missing_ok=True)
            except OSError:
                pass


def _fetch_eccc_deterministic(model_source: str, step: int, target: Path) -> str:
    """Fetch exactly the 10 m U and V files from the official ECCC HPFX mirror."""
    if model_source == "eccc_gdps":
        folder, resolution, model_name = "model_gdps", "15km", "GDPS"
    else:
        folder, resolution, model_name = "model_gdps-geml", "25km", "GDPS-GEML"
    grid_tag = "0.15" if model_source == "eccc_gdps" else "0.25"
    failures: list[str] = []
    for cycle in _candidate_cycles():
        if cycle.hour not in (0, 12):
            continue
        date_text, hour = _cycle_parts(cycle)
        lead = f"{step:03d}"
        root = _eccc_cycle_root(cycle)
        urls = {}
        for component, variable in (("u", "WindU"), ("v", "WindV")):
            name = f"{date_text}T{hour}Z_MSC_{model_name}_{variable}_AGL-10m_LatLon{grid_tag}_PT{lead}H.grib2"
            urls[component] = f"https://hpfx.collab.science.gc.ca/{root}/{folder}/{resolution}/{hour}/{lead}/{name}"
        try:
            payloads = _download_pair(urls)
            target.write_bytes(payloads["u"] + payloads["v"])
            return cycle.strftime("%Y-%m-%dT%H:%MZ")
        except Exception as exc:
            failures.append(f"{cycle:%Y-%m-%d %H} U/V pair: {exc}")
    detail = failures[-1] if failures else "no recent 00/12 UTC run was found"
    raise RuntimeError(f"ECCC {model_name} 10 m wind files are not currently available for lead {step} h. Try a different forecast hour. Latest check: {detail}")


def _fetch_eccc_geps_mean(step: int, target: Path) -> str:
    """Fetch two official all-member files; the decoder averages their 21 members."""
    failures: list[str] = []
    for cycle in _candidate_cycles():
        if cycle.hour not in (0, 12):
            continue
        date_text, hour = _cycle_parts(cycle)
        lead = f"{step:03d}"
        root = _eccc_cycle_root(cycle)
        urls = {}
        for key, component in (("u", "UGRD"), ("v", "VGRD")):
            name = f"CMC_geps-raw_{component}_TGL_10m_latlon0p5x0p5_{date_text}{hour}_P{lead}_allmbrs.grib2"
            urls[key] = f"https://hpfx.collab.science.gc.ca/{root}/ensemble/geps/grib2/raw/{hour}/{lead}/{name}"
        try:
            payloads = _download_pair(urls)
            target.write_bytes(payloads["u"] + payloads["v"])
            return cycle.strftime("%Y-%m-%dT%H:%MZ")
        except Exception as exc:
            failures.append(f"{cycle:%Y-%m-%d %H} U/V pair: {exc}")
    detail = failures[-1] if failures else "no recent 00/12 UTC ensemble run was found"
    raise RuntimeError(f"ECCC GEPS all-member 10 m winds are not currently available for lead {step} h. Try a different forecast hour. Latest check: {detail}")


def _fetch_dwd_global(model_source: str, step: int, target: Path) -> str:
    """Fetch only the 10 m U/V GRIB files from DWD's current global NWP tree."""
    model = "icon" if model_source == "dwd_icon" else "aicon"
    valid_hours = (0, 12) if step > 120 else (0, 6, 12, 18)
    failures: list[str] = []
    for cycle in _candidate_cycles():
        if cycle.hour not in valid_hours:
            continue
        run = cycle.strftime("%Y-%m-%dT%H:00")
        lead = f"PT{step:03d}H00M"
        urls = {
            component: f"https://opendata.dwd.de/weather/nwp/v1/m/{model}/p/{component}/r/{run}/s/{lead}.grib2"
            for component in ("U_10M", "V_10M")
        }
        try:
            payloads = _download_pair(urls)
            target.write_bytes(payloads["U_10M"] + payloads["V_10M"])
            return cycle.strftime("%Y-%m-%dT%H:%MZ")
        except Exception as exc:
            failures.append(f"{run} U/V pair: {exc}")
    detail = failures[-1] if failures else "no recent compatible cycle was found"
    raise RuntimeError(f"DWD {model.upper()} 10 m U/V fields are not available for lead {step} h. Try another lead hour. Latest check: {detail}")


def _fetch_ecmwf(descriptor: dict[str, Any], step: int, target: Path) -> str | None:
    try:
        from ecmwf.opendata import Client
    except ImportError as exc:
        raise RuntimeError("ECMWF downloader is not installed. Rebuild the Render service from the supplied Dockerfile.") from exc

    client = Client(
        source="ecmwf",
        model=descriptor["ecmwf_model"],
        resol=descriptor["resol"],
        infer_stream_keyword=False,
    )
    try:
        result = client.retrieve(
            stream=descriptor["stream"],
            type=descriptor["type"],
            step=step,
            param=["10u", "10v"],
            target=str(target),
        )
    except Exception as exc:
        label = descriptor["name"]
        raise RuntimeError(f"{label} did not return this step from ECMWF Open Data. Try another lead hour or the 0.25° IFS option. Provider said: {exc}") from exc
    if not target.exists() or target.stat().st_size < 16:
        raise RuntimeError("ECMWF returned an empty GRIB file for this run/step.")
    result_date = getattr(result, "datetime", None)
    if result_date is not None:
        try:
            return result_date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        except Exception:
            pass
    return None


def _fetch_gfs(step: int, target: Path) -> str:
    session = _request_session()
    failures: list[str] = []
    for cycle in _candidate_cycles():
        date, hh = _cycle_parts(cycle)
        filename = f"gfs.t{hh}z.pgrb2.0p25.f{step:03d}"
        params = {
            "file": filename,
            "var_UGRD": "on",
            "var_VGRD": "on",
            "lev_10_m_above_ground": "on",
            "subregion": "",
            "leftlon": str(DOMAIN["west"]),
            "rightlon": str(DOMAIN["east"]),
            "toplat": str(DOMAIN["north"]),
            "bottomlat": str(DOMAIN["south"]),
            "dir": f"/gfs.{date}/{hh}/atmos",
        }
        try:
            response = session.get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl", params=params, timeout=HTTP_TIMEOUT)
            if response.ok and response.content[:4] == b"GRIB":
                target.write_bytes(response.content)
                return cycle.strftime("%Y-%m-%dT%H:%MZ")
            failures.append(f"{hh}z: HTTP {response.status_code}")
        except requests.RequestException as exc:
            failures.append(f"{hh}z: {exc.__class__.__name__}")
    raise RuntimeError("NOAA NOMADS did not return the requested GFS hour. Latest-cycle attempts: " + ", ".join(failures[-4:]))


def _fetch_gefs_mean(step: int, target: Path) -> str:
    session = _request_session()
    failures: list[str] = []
    for cycle in _candidate_cycles():
        date, hh = _cycle_parts(cycle)
        filename = f"geavg.t{hh}z.pgrb2a.0p50.f{step:03d}"
        params = {
            "file": filename,
            "var_UGRD": "on",
            "var_VGRD": "on",
            "lev_10_m_above_ground": "on",
            "subregion": "",
            "leftlon": str(DOMAIN["west"]),
            "rightlon": str(DOMAIN["east"]),
            "toplat": str(DOMAIN["north"]),
            "bottomlat": str(DOMAIN["south"]),
            "dir": f"/gefs.{date}/{hh}/atmos/pgrb2ap5",
        }
        try:
            response = session.get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl", params=params, timeout=HTTP_TIMEOUT)
            if response.ok and response.content[:4] == b"GRIB":
                target.write_bytes(response.content)
                return cycle.strftime("%Y-%m-%dT%H:%MZ")
            failures.append(f"{hh}z: HTTP {response.status_code}")
        except requests.RequestException as exc:
            failures.append(f"{hh}z: {exc.__class__.__name__}")
    raise RuntimeError("NOAA NOMADS did not return the requested GEFS ensemble-mean wind. Latest-cycle attempts: " + ", ".join(failures[-4:]))


def _index_field_bounds(index_text: str) -> dict[str, tuple[int, int]]:
    records: list[tuple[int, str]] = []
    for line in index_text.splitlines():
        pieces = line.strip().split(":")
        if len(pieces) < 5:
            continue
        try:
            offset = int(pieces[1])
        except ValueError:
            continue
        records.append((offset, line))

    found: dict[str, tuple[int, int]] = {}
    for index, (offset, line) in enumerate(records):
        upper = line.upper()
        is_10m = "10 M ABOVE GROUND" in upper or "10M ABOVE GROUND" in upper
        if not is_10m:
            continue
        key = None
        if ":UGRD:" in upper or ":10U:" in upper or ":U10:" in upper:
            key = "u"
        elif ":VGRD:" in upper or ":10V:" in upper or ":V10:" in upper:
            key = "v"
        if key is None or index + 1 >= len(records):
            continue
        end = records[index + 1][0] - 1
        if end >= offset:
            found[key] = (offset, end)
    return found


def _fetch_noaa_indexed_mean(model_key: str, step: int, target: Path) -> str:
    session = _request_session()
    failures: list[str] = []
    for cycle in _candidate_cycles():
        date, hh = _cycle_parts(cycle)
        base = (
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/{model_key}/prod/"
            f"{model_key}.{date}/{hh}/ensstat/products/atmos/grib2/"
            f"{model_key}.t{hh}z.sfc.avg.f{step:03d}.grib2"
        )
        try:
            idx_response = session.get(base + ".idx", timeout=HTTP_TIMEOUT)
            if not idx_response.ok:
                failures.append(f"{hh}z index: HTTP {idx_response.status_code}")
                continue
            bounds = _index_field_bounds(idx_response.text)
            if set(bounds) != {"u", "v"}:
                failures.append(f"{hh}z index: 10 m U/V mean fields not listed")
                continue
            ordered_bounds = sorted(bounds.items(), key=lambda item: item[1][0])
            range_headers = {key: {"Range": f"bytes={start}-{end}"} for key, (start, end) in ordered_bounds}
            range_payloads = _download_pair(
                {key: base for key, _ in ordered_bounds},
                headers=range_headers,
                expected_status={key: 206 for key, _ in ordered_bounds},
            )
            with target.open("wb") as handle:
                for key, _ in ordered_bounds:
                    handle.write(range_payloads[key])
            return cycle.strftime("%Y-%m-%dT%H:%MZ")
        except requests.RequestException as exc:
            failures.append(f"{hh}z: {exc.__class__.__name__}")
        except RuntimeError as exc:
            failures.append(f"{hh}z: {exc}")
    raise RuntimeError(f"NOAA NOMADS did not return the requested {model_key.upper()} mean wind hour. Latest-cycle attempts: " + ", ".join(failures[-4:]))


def _fetch_aigfs(step: int, target: Path) -> str:
    session = _request_session()
    failures: list[str] = []
    for cycle in _candidate_cycles():
        date, hh = _cycle_parts(cycle)
        base = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{date}/{hh}/model/atmos/grib2/aigfs.t{hh}z.sfc.f{step:03d}.grib2"
        try:
            idx_response = session.get(base + ".idx", timeout=HTTP_TIMEOUT)
            if not idx_response.ok:
                failures.append(f"{hh}z index: HTTP {idx_response.status_code}")
                continue
            bounds = _index_field_bounds(idx_response.text)
            if set(bounds) != {"u", "v"}:
                failures.append(f"{hh}z index: 10 m U/V fields not listed")
                continue
            ordered_bounds = sorted(bounds.items(), key=lambda item: item[1][0])
            range_headers = {key: {"Range": f"bytes={start}-{end}"} for key, (start, end) in ordered_bounds}
            range_payloads = _download_pair(
                {key: base for key, _ in ordered_bounds},
                headers=range_headers,
                expected_status={key: 206 for key, _ in ordered_bounds},
            )
            with target.open("wb") as handle:
                for key, _ in ordered_bounds:
                    handle.write(range_payloads[key])
            return cycle.strftime("%Y-%m-%dT%H:%MZ")
        except requests.RequestException as exc:
            failures.append(f"{hh}z: {exc.__class__.__name__}")
        except RuntimeError as exc:
            failures.append(f"{hh}z: {exc}")
    raise RuntimeError("NOAA NOMADS did not return the requested AIGFS hour. Latest-cycle attempts: " + ", ".join(failures[-4:]))


def _safe_get(gid: Any, key: str, default: Any = None) -> Any:
    try:
        return codes_get(gid, key)
    except Exception:
        return default


def _field_component(gid: Any) -> str | None:
    short = str(_safe_get(gid, "shortName", "")).lower()
    if short in {"10u", "u10", "ugrd", "u"}:
        if short in {"ugrd", "u"} and _safe_get(gid, "level") not in (10, 10.0):
            return None
        return "u"
    if short in {"10v", "v10", "vgrd", "v"}:
        if short in {"vgrd", "v"} and _safe_get(gid, "level") not in (10, 10.0):
            return None
        return "v"
    name = str(_safe_get(gid, "name", "")).lower()
    level = _safe_get(gid, "level")
    if level in (10, 10.0):
        if "u-component" in name or "u wind component" in name:
            return "u"
        if "v-component" in name or "v wind component" in name:
            return "v"
    return None


def _crop_regular_grid(gid: Any, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ni = int(_safe_get(gid, "Ni", 0) or 0)
    nj = int(_safe_get(gid, "Nj", 0) or 0)
    j_consecutive = int(_safe_get(gid, "jPointsAreConsecutive", 0) or 0)
    alternating = int(_safe_get(gid, "alternativeRowScanning", 0) or 0)
    grid_type = str(_safe_get(gid, "gridType", ""))

    if ni > 1 and nj > 1 and values.size == ni * nj and grid_type == "regular_ll" and not j_consecutive and not alternating:
        lat0 = float(_safe_get(gid, "latitudeOfFirstGridPointInDegrees"))
        lon0 = float(_safe_get(gid, "longitudeOfFirstGridPointInDegrees"))
        di = float(_safe_get(gid, "iDirectionIncrementInDegrees"))
        dj = float(_safe_get(gid, "jDirectionIncrementInDegrees"))
        i_negative = int(_safe_get(gid, "iScansNegatively", 0) or 0)
        j_positive = int(_safe_get(gid, "jScansPositively", 0) or 0)
        lon_axis = lon0 + (-1.0 if i_negative else 1.0) * di * np.arange(ni, dtype=np.float64)
        lat_axis = lat0 + (1.0 if j_positive else -1.0) * dj * np.arange(nj, dtype=np.float64)
        lon_axis = np.round(((lon_axis + 180.0) % 360.0) - 180.0, 8)
        lat_axis = np.round(lat_axis, 8)
        rows = np.flatnonzero((lat_axis >= DOMAIN["south"] - 0.5) & (lat_axis <= DOMAIN["north"] + 0.5))
        cols = np.flatnonzero((lon_axis >= DOMAIN["west"] - 0.5) & (lon_axis <= DOMAIN["east"] + 0.5))
        if not len(rows) or not len(cols):
            raise RuntimeError("The downloaded GRIB grid does not intersect the Bay of Bengal domain.")
        data = values.reshape((nj, ni))
        rows = rows[np.argsort(lat_axis[rows])]
        cols = cols[np.argsort(lon_axis[cols])]
        return lon_axis[cols], lat_axis[rows], data[np.ix_(rows, cols)]

    # Defensive fallback for a non-regular GRIB grid. This is uncommon for the
    # selected operational products, but handles rotated/metadata-rich grids.
    latitudes = np.asarray(codes_get_array(gid, "latitudes"), dtype=np.float32).reshape((nj, ni))
    longitudes = np.asarray(codes_get_array(gid, "longitudes"), dtype=np.float32).reshape((nj, ni))
    longitudes = ((longitudes + 180.0) % 360.0) - 180.0
    data = values.reshape((nj, ni))
    row_mask = np.any((latitudes >= DOMAIN["south"] - 0.5) & (latitudes <= DOMAIN["north"] + 0.5), axis=1)
    col_mask = np.any((longitudes >= DOMAIN["west"] - 0.5) & (longitudes <= DOMAIN["east"] + 0.5), axis=0)
    rows = np.flatnonzero(row_mask)
    cols = np.flatnonzero(col_mask)
    if not len(rows) or not len(cols):
        raise RuntimeError("The downloaded GRIB grid does not intersect the Bay of Bengal domain.")
    sub_lon = longitudes[np.ix_(rows, cols)]
    sub_lat = latitudes[np.ix_(rows, cols)]
    sub_values = data[np.ix_(rows, cols)]
    # These selected sources are regular lat/lon. Sorting the first row/column
    # normalizes scans from either direction for the renderer.
    col_order = np.argsort(np.nanmean(sub_lon, axis=0))
    row_order = np.argsort(np.nanmean(sub_lat, axis=1))
    return (
        np.nanmean(sub_lon, axis=0)[col_order],
        np.nanmean(sub_lat, axis=1)[row_order],
        sub_values[np.ix_(row_order, col_order)],
    )


def _decode_wind(path: Path) -> dict[str, Any]:
    components: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    init = None
    with path.open("rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                component = _field_component(gid)
                if component and component not in components:
                    data_date = int(_safe_get(gid, "dataDate", 0) or 0)
                    data_time = int(_safe_get(gid, "dataTime", 0) or 0)
                    if data_date:
                        date_text = datetime.strptime(str(data_date), "%Y%m%d").strftime("%Y-%m-%d")
                        init = f"{date_text}T{data_time // 100:02d}:{data_time % 100:02d}Z"
                    values = np.asarray(codes_get_array(gid, "values"), dtype=np.float32)
                    components[component] = _crop_regular_grid(gid, values)
            finally:
                codes_release(gid)

    if "u" not in components or "v" not in components:
        raise RuntimeError("The source GRIB did not contain both 10 m U- and V-wind fields. Check current model availability and try a different forecast hour.")
    lon_u, lat_u, u = components["u"]
    lon_v, lat_v, v = components["v"]
    if u.shape != v.shape or len(lon_u) != len(lon_v) or len(lat_u) != len(lat_v):
        raise RuntimeError("The source wind components use different grids; this model run cannot be plotted as a wind vector field.")
    return {"lons": lon_u.astype(np.float64), "lats": lat_u.astype(np.float64), "u": u.astype(np.float32), "v": v.astype(np.float32), "init": init}


def _decode_geps_mean(path: Path) -> dict[str, Any]:
    """Average every official GEPS member for each wind component before cropping."""
    sums: dict[str, np.ndarray] = {}
    counts = {"u": 0, "v": 0}
    grids: dict[str, Any] = {}
    members: dict[str, set[int]] = {"u": set(), "v": set()}
    init = None
    try:
        with path.open("rb") as handle:
            while True:
                gid = codes_grib_new_from_file(handle)
                if gid is None:
                    break
                keep_gid = False
                try:
                    component = _field_component(gid)
                    if component not in {"u", "v"}:
                        continue
                    if init is None:
                        data_date = int(_safe_get(gid, "dataDate", 0) or 0)
                        data_time = int(_safe_get(gid, "dataTime", 0) or 0)
                        if data_date:
                            date_text = datetime.strptime(str(data_date), "%Y%m%d").strftime("%Y-%m-%d")
                            init = f"{date_text}T{data_time // 100:02d}:{data_time % 100:02d}Z"
                    member = _safe_get(gid, "perturbationNumber")
                    if member is not None:
                        members[component].add(int(member))
                    values = np.asarray(codes_get_array(gid, "values"), dtype=np.float64)
                    if component not in sums:
                        sums[component] = values.copy()
                        grids[component] = gid
                        keep_gid = True
                    else:
                        if values.shape != sums[component].shape:
                            raise RuntimeError("ECCC GEPS ensemble members use inconsistent grids.")
                        sums[component] += values
                    counts[component] += 1
                finally:
                    if not keep_gid:
                        codes_release(gid)

        expected_members = set(range(21))  # control (0) plus 20 perturbed members
        if counts["u"] != 21 or counts["v"] != 21:
            raise RuntimeError("The official ECCC GEPS files did not contain the expected 21 10 m U/V members.")
        if members["u"] != expected_members or members["v"] != expected_members:
            raise RuntimeError("The ECCC GEPS U/V files contain duplicate or missing ensemble-member identifiers.")

        lon_u, lat_u, u = _crop_regular_grid(grids["u"], (sums["u"] / counts["u"]).astype(np.float32))
        lon_v, lat_v, v = _crop_regular_grid(grids["v"], (sums["v"] / counts["v"]).astype(np.float32))
        if u.shape != v.shape or not np.array_equal(lon_u, lon_v) or not np.array_equal(lat_u, lat_v):
            raise RuntimeError("ECCC GEPS ensemble components use different grids.")
        return {"lons": lon_u, "lats": lat_u, "u": u, "v": v, "init": init, "ensemble_members": counts["u"]}
    finally:
        for gid in grids.values():
            codes_release(gid)


_ICON_REGRID_PATH = CACHE_ROOT / "dwd-icon-bay-regrid-v2.npz"
_ICON_REGRID_BUNDLED_PATH = Path(__file__).parent / "static" / "geodata" / "dwd-icon-bay-regrid-v2.npz"
_ICON_REGRID_LOCK = threading.RLock()


def _unit_sphere_xyz(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(longitudes, dtype=np.float64))
    cos_lat = np.cos(lat_rad)
    return np.column_stack((cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)))


def _decode_dwd_coordinate_payload(payload: bytes) -> tuple[np.ndarray, str]:
    with tempfile.NamedTemporaryFile(prefix="baywind-coordinate-", suffix=".grib2", dir=CACHE_ROOT) as temp:
        temp.write(payload)
        temp.flush()
        with open(temp.name, "rb") as handle:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                raise RuntimeError("DWD returned an unreadable ICON grid-coordinate GRIB.")
            try:
                grid_uuid = str(_safe_get(gid, "uuidOfHGrid", ""))
                if not grid_uuid:
                    raise RuntimeError("DWD ICON coordinate GRIB did not include its grid UUID.")
                values = np.asarray(codes_get_array(gid, "values"), dtype=np.float32)
                return values, grid_uuid
            finally:
                codes_release(gid)


def _load_dwd_regrid_map(init: str | None) -> dict[str, np.ndarray]:
    """Build and cache a 4-neighbour spherical map from the ICON native grid."""
    with _ICON_REGRID_LOCK:
        if _ICON_REGRID_PATH.exists():
            with np.load(_ICON_REGRID_PATH, allow_pickle=False) as archive:
                return {key: archive[key] for key in ("lons", "lats", "indices", "weights", "grid_uuid")}

        # The grid mapping is static for this DWD global mesh/domain. Bundle the
        # small derived lookup so Render cold starts do not re-download full
        # CLAT/CLON fields and rebuild the tree for every fresh instance.
        if _ICON_REGRID_BUNDLED_PATH.is_file():
            with np.load(_ICON_REGRID_BUNDLED_PATH, allow_pickle=False) as archive:
                result = {key: archive[key] for key in ("lons", "lats", "indices", "weights", "grid_uuid")}
            try:
                shutil.copyfile(_ICON_REGRID_BUNDLED_PATH, _ICON_REGRID_PATH)
            except OSError:
                pass
            return result

        if not init:
            raise RuntimeError("The DWD run time is missing; cannot request its ICON grid coordinates.")
        cycle = datetime.strptime(init, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
        run = cycle.strftime("%Y-%m-%dT%H:00")
        root = "https://opendata.dwd.de/weather/nwp/v1/m/icon/p"
        coordinate_payloads = _download_pair({
            "clat": f"{root}/CLAT/r/{run}/s/PT000H00M.grib2",
            "clon": f"{root}/CLON/r/{run}/s/PT000H00M.grib2",
        })
        clat, clat_uuid = _decode_dwd_coordinate_payload(coordinate_payloads["clat"])
        clon, clon_uuid = _decode_dwd_coordinate_payload(coordinate_payloads["clon"])
        if clat_uuid != clon_uuid:
            raise RuntimeError("DWD ICON latitude/longitude fields use different native-grid UUIDs.")
        if clat.shape != clon.shape:
            raise RuntimeError("DWD ICON latitude/longitude fields have different native-grid sizes.")
        clon = ((clon + 180.0) % 360.0) - 180.0
        valid = np.isfinite(clat) & np.isfinite(clon)
        valid &= (clat >= DOMAIN["south"] - 0.75) & (clat <= DOMAIN["north"] + 0.75)
        valid &= (clon >= DOMAIN["west"] - 0.75) & (clon <= DOMAIN["east"] + 0.75)
        source_indices = np.flatnonzero(valid)
        if source_indices.size < 16:
            raise RuntimeError("DWD ICON grid coordinates do not cover the Bay of Bengal map domain.")

        lons = np.round(np.arange(DOMAIN["west"], DOMAIN["east"] + 0.0001, 0.1), 8)
        lats = np.round(np.arange(DOMAIN["south"], DOMAIN["north"] + 0.0001, 0.1), 8)
        target_lat, target_lon = np.meshgrid(lats, lons, indexing="ij")
        target_xyz = _unit_sphere_xyz(target_lat.ravel(), target_lon.ravel())
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise RuntimeError("SciPy is required to remap the official DWD unstructured grid.") from exc
        tree = cKDTree(_unit_sphere_xyz(clat[source_indices], clon[source_indices]))
        distances, local_indices = tree.query(target_xyz, k=4, workers=1)
        if not np.all(np.isfinite(distances)):
            raise RuntimeError("DWD ICON regridding could not find neighbours for the selected domain.")
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / np.square(distances)
        weights /= weights.sum(axis=1, keepdims=True)
        indices = source_indices[local_indices].astype(np.int32).reshape((*target_lat.shape, 4))
        weights = weights.astype(np.float32).reshape((*target_lat.shape, 4))
        result = {
            "lons": lons,
            "lats": lats,
            "indices": indices,
            "weights": weights,
            "grid_uuid": np.array(clat_uuid),
        }
        temp = _ICON_REGRID_PATH.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **result)
        temp.replace(_ICON_REGRID_PATH)
        return result


def _decode_dwd_unstructured(path: Path, init: str | None) -> dict[str, Any]:
    remap = _load_dwd_regrid_map(init)
    components: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                component = _field_component(gid)
                if component and component not in components:
                    if str(_safe_get(gid, "gridType", "")) != "unstructured_grid":
                        raise RuntimeError("DWD global field grid layout changed; refusing to apply the ICON coordinate map.")
                    actual_uuid = str(_safe_get(gid, "uuidOfHGrid", ""))
                    expected_uuid = str(np.asarray(remap["grid_uuid"]).item())
                    if not actual_uuid or actual_uuid != expected_uuid:
                        raise RuntimeError("DWD ICON grid UUID changed; refusing to use a stale remapping table.")
                    values = np.asarray(codes_get_array(gid, "values"), dtype=np.float32)
                    if values.size <= int(remap["indices"].max()):
                        raise RuntimeError("DWD wind field and ICON grid-coordinate sizes do not match.")
                    components[component] = np.sum(values[remap["indices"]] * remap["weights"], axis=-1, dtype=np.float32)
            finally:
                codes_release(gid)
    if "u" not in components or "v" not in components:
        raise RuntimeError("DWD GRIB did not contain both 10 m U/V fields.")
    if components["u"].shape != components["v"].shape:
        raise RuntimeError("DWD ICON wind components use different native-grid shapes.")
    return {"lons": remap["lons"], "lats": remap["lats"], "u": components["u"], "v": components["v"], "init": init}


def _save_fields(path: Path, fields: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            lons=fields["lons"],
            lats=fields["lats"],
            u=fields["u"],
            v=fields["v"],
            init=np.array(fields.get("init") or ""),
            source=np.array(fields.get("source") or ""),
        )
    temp.replace(path)


def _load_fields(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "lons": archive["lons"],
            "lats": archive["lats"],
            "u": archive["u"],
            "v": archive["v"],
            "init": str(archive["init"].item()),
            "source": str(archive["source"].item()),
        }
