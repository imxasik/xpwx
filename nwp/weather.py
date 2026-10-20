from __future__ import annotations

import json
import os
import re
import tempfile
import threading
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
# All weather fields come from official ECMWF or NCEP/NOMADS distributions;
# ensemble means are never approximated with a single member.
_IFS_STEPS = list(range(0, 145, 3)) + list(range(150, 361, 6))
_AIFS_STEPS = list(range(0, 361, 6))
_GFS_STEPS = list(range(0, 121, 3)) + list(range(126, 385, 6))
_AIGFS_STEPS = list(range(0, 385, 6))
_GEFS_STEPS = list(range(0, 385, 6))
_HGEFS_STEPS = list(range(0, 241, 6))

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
        "name": "ECMWF AIFS Single",
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
        "name": "ECMWF AIFS ENS · Control",
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
        "name": "NOAA GFS",
        "resolution": "0.25°",
        "source": "noaa_gfs",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _GFS_STEPS,
        "note": "Regional GRIB filter requests only 10 m U/V wind over the Bay of Bengal box.",
    },
    "aigfs": {
        "name": "NOAA AIGFS",
        "resolution": "Global AI model",
        "source": "noaa_aigfs",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _AIGFS_STEPS,
        "note": "NCEP AI global forecast; only the two 10 m wind fields are transferred, then cropped to the selected domain.",
    },
    "gefs_mean": {
        "name": "NOAA GEFS Mean",
        "resolution": "0.5° ensemble mean",
        "source": "noaa_gefs",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _GEFS_STEPS,
        "note": "Official precomputed ensemble-mean U/V winds; not a single member and no local ensemble averaging is needed.",
    },
    "aigefs_mean": {
        "name": "NOAA AIGEFS Mean",
        "resolution": "AI ensemble mean",
        "source": "noaa_aigefs_mean",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _AIGFS_STEPS,
        "note": "Official AI ensemble mean; only indexed 10 m U/V wind messages are transferred.",
    },
    "hgefs_mean": {
        "name": "NOAA HGEFS Mean",
        "resolution": "Hybrid ensemble mean",
        "source": "noaa_hgefs_mean",
        "source_label": "NOAA NCEP NOMADS",
        "steps": _HGEFS_STEPS,
        "note": "Official hybrid ensemble mean. Product currently provides the listed shorter lead-hour range.",
    },
}

DOMAIN = {"west": 70.0, "east": 107.0, "south": 0.0, "north": 30.0}
FIELD_CACHE_VERSION = 3  # refresh cache for the 0°S-to-30°N crop
CACHE_ROOT = Path(os.environ.get("BAYWIND_CACHE_DIR", "/tmp/baywind-cache"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_TTL = int(os.environ.get("BAYWIND_CACHE_TTL", "5400"))
HTTP_TIMEOUT = (8, int(os.environ.get("BAYWIND_HTTP_TIMEOUT", "45")))
USER_AGENT = "BaywindModelViewer/1.0 (official open model-data visualization)"
_CACHE_LOCK = threading.RLock()


def _valid_cache(path: Path) -> bool:
    return path.exists() and (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime < CACHE_TTL)


def _candidate_cycles() -> list[datetime]:
    now = datetime.now(timezone.utc)
    hour = (now.hour // 6) * 6
    latest = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return [latest - timedelta(hours=6 * i) for i in range(8)]


def _cycle_parts(cycle: datetime) -> tuple[str, str]:
    return cycle.strftime("%Y%m%d"), cycle.strftime("%H")


def _request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return session


def get_forecast(model_id: str, step: int) -> dict[str, Any]:
    if model_id not in MODEL_CATALOG:
        raise ValueError("Unknown model.")
    field_cache = CACHE_ROOT / f"fields-v{FIELD_CACHE_VERSION}-{model_id}-f{step:03d}.npz"
    with _CACHE_LOCK:
        if _valid_cache(field_cache):
            return _load_fields(field_cache)

        descriptor = MODEL_CATALOG[model_id]
        with tempfile.NamedTemporaryFile(prefix="baywind-", suffix=".grib2", dir=CACHE_ROOT, delete=False) as handle:
            grib_path = Path(handle.name)
        try:
            if descriptor["source"] == "ecmwf":
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
            else:
                raise ValueError("This model source is not configured.")

            fields = _decode_wind(grib_path)
            fields["init"] = init or fields.get("init") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            fields["model_id"] = model_id
            fields["source"] = descriptor["source_label"]
            _save_fields(field_cache, fields)
            return fields
        finally:
            try:
                grib_path.unlink(missing_ok=True)
            except OSError:
                pass


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
            pieces: list[tuple[int, bytes]] = []
            for key, (start, end) in bounds.items():
                response = session.get(base, headers={"Range": f"bytes={start}-{end}"}, timeout=HTTP_TIMEOUT)
                if response.status_code != 206 or response.content[:4] != b"GRIB":
                    raise RuntimeError(f"Could not range-download the {model_key.upper()} {key.upper()} mean wind field (HTTP {response.status_code}).")
                pieces.append((start, response.content))
            with target.open("wb") as handle:
                for _, content in sorted(pieces):
                    handle.write(content)
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
            pieces: list[tuple[int, bytes]] = []
            for key, (start, end) in bounds.items():
                response = session.get(base, headers={"Range": f"bytes={start}-{end}"}, timeout=HTTP_TIMEOUT)
                if response.status_code != 206 or response.content[:4] != b"GRIB":
                    raise RuntimeError(f"Could not range-download the AIGFS {key.upper()} wind field (HTTP {response.status_code}); server must honor byte ranges to avoid a full-file transfer.")
                pieces.append((start, response.content))
            with target.open("wb") as handle:
                for _, content in sorted(pieces):
                    handle.write(content)
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
