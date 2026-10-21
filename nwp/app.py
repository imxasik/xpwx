from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from io import BytesIO
import os
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template, request, send_file

from renderer import render_map, render_scalar_map, render_track_map
from weather import FIELD_DEFINITIONS, MODEL_CATALOG, get_forecast, get_scalar_forecast, get_track_series, get_upperair_forecast, model_fields

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
PNG_CACHE_DIR = Path(os.environ.get("BAYWIND_CACHE_DIR", "/tmp/baywind-cache"))
PNG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PNG_CACHE_TTL = int(os.environ.get("BAYWIND_CACHE_TTL", "5400"))
RENDER_VERSION = 13  # dedicated track maps and upper-air fields
_RENDER_LOCK = threading.RLock()


def _safe_header(value: object) -> str:
    text = str(value).replace("–", "-").replace("—", "-")
    return text.encode("latin-1", errors="replace").decode("latin-1")


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/healthz")
def health():
    return {"status": "ok", "service": "Baywind"}


@app.get("/api/models")
def models():
    return jsonify({
        "domain": {"name": "Bay of Bengal", "west": 70, "east": 110, "south": 0, "north": 30},
        "models": [
            {
                "id": model_id,
                "name": item["name"],
                "resolution": item["resolution"],
                "source": item["source_label"],
                "steps": item["steps"],
                "note": item.get("note", ""),
                "enabled": item.get("enabled", True),
                "fields": model_fields(model_id),
            }
            for model_id, item in MODEL_CATALOG.items()
        ],
        "layers": ["10 m wind speed", "Streamlines", "200 hPa wind", "200–850 hPa wind shear", "Cyclone Track"],
        "units": ["knots", "m/s", "km/h"],
    })


TRACK_MODEL_IDS = ("ifs_025", "aifs", "gfs", "aigfs", "dwd_icon", "dwd_aicon")


@app.get("/api/track.png")
def track_png():
    """Dedicated cyclone track parameter image. Track lines never overlay field maps."""
    raw_models = request.args.get("models", request.args.get("model", "ifs_025"))
    model_ids = list(dict.fromkeys(part.strip() for part in raw_models.split(",") if part.strip()))
    if not model_ids or any(model_id not in TRACK_MODEL_IDS for model_id in model_ids):
        return jsonify({"error": "Select one or more supported cyclone-track models."}), 400
    try:
        dpi = int(request.args.get("dpi", "150"))
    except (TypeError, ValueError):
        return jsonify({"error": "DPI must be 120, 150, or 300."}), 400
    if dpi not in {120, 150, 300}:
        return jsonify({"error": "DPI must be 120, 150, or 300."}), 400

    cache_key = "-".join(model_ids)
    image_name = f"baywind-cyclone-track-{cache_key}-{dpi}dpi.png"
    image_cache = PNG_CACHE_DIR / f"track-v3-{cache_key}-d{dpi}.png"
    metadata_cache = image_cache.with_suffix(".json")
    def cached_track_map():
        try:
            fresh = image_cache.exists() and metadata_cache.exists() and (
                datetime.now(timezone.utc).timestamp() - image_cache.stat().st_mtime < PNG_CACHE_TTL
            )
            if not fresh:
                return None
            cached_png = image_cache.read_bytes()
            cached_metadata = json.loads(metadata_cache.read_text(encoding="utf-8"))
            if (not isinstance(cached_metadata, dict) or cached_metadata.get("requested") != model_ids
                    or not isinstance(cached_metadata.get("tracks"), list)):
                return None
            return cached_png, cached_metadata
        except (OSError, ValueError, TypeError):
            return None

    try:
        cached = cached_track_map()
        if cached is not None:
            png, metadata = cached
        else:
            with _RENDER_LOCK:
                cached = cached_track_map()
                if cached is not None:
                    png, metadata = cached
                else:
                    track_sets, model_specs, unavailable = [], [], []
                    for model_id in model_ids:
                        try:
                            track_sets.append(get_track_series(model_id))
                            model_specs.append({"id": model_id, "name": MODEL_CATALOG[model_id]["name"],
                                                "resolution": MODEL_CATALOG[model_id]["resolution"]})
                        except Exception as model_exc:
                            app.logger.warning("Track unavailable for %s: %s", model_id, model_exc)
                            unavailable.append(MODEL_CATALOG[model_id]["name"])
                    if not track_sets:
                        detail = ", ".join(unavailable) or "selected models"
                        raise RuntimeError(f"No selected cyclone tracks are available right now ({detail}). Try again when a new official model cycle is published.")
                    png = render_track_map(track_sets, model_specs, dpi=dpi)
                    metadata = {
                        "requested": model_ids,
                        "tracks": [{"model": item["model"], "init": item["init"], "horizon": item["horizon"]} for item in track_sets],
                        "unavailable": unavailable,
                    }
                    temp_png = image_cache.with_name(image_cache.stem + f"-img-{threading.get_ident()}.tmp")
                    temp_meta = metadata_cache.with_name(metadata_cache.stem + f"-meta-{threading.get_ident()}.tmp")
                    temp_png.write_bytes(png)
                    temp_meta.write_text(json.dumps(metadata), encoding="utf-8")
                    temp_meta.replace(metadata_cache)
                    temp_png.replace(image_cache)
    except Exception as exc:
        app.logger.exception("Cyclone track map generation failed for %s", cache_key)
        return jsonify({"error": str(exc) or "Could not generate the selected cyclone track map."}), 502

    response = make_response(send_file(BytesIO(png), mimetype="image/png", download_name=image_name))
    response.headers["Cache-Control"] = "public, max-age=900"
    response.headers["X-Model-Name"] = "Cyclone Track"
    tracks_meta = metadata["tracks"]
    response.headers["X-Model-Run"] = tracks_meta[0]["init"] if len(tracks_meta) == 1 else "Multiple model cycles"
    response.headers["X-Valid-Time"] = "Full selected forecast runs"
    response.headers["X-Track-Horizons"] = _safe_header("; ".join(
        f"{MODEL_CATALOG[item['model']]['name']} +{item['horizon']} h" for item in tracks_meta
    ))
    response.headers["X-Track-Count"] = str(len(tracks_meta))
    if metadata.get("unavailable"):
        response.headers["X-Track-Unavailable"] = _safe_header(", ".join(metadata["unavailable"]))
    response.headers["X-Render-DPI"] = str(dpi)
    return response


@app.get("/api/track.json")
def track_json():
    model_id = request.args.get("model", "")
    try:
        result = get_track_series(model_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Cyclone track generation failed for %s", model_id)
        return jsonify({"error": str(exc) or "Could not retrieve this cyclone track."}), 502
    response = jsonify(result)
    response.headers["Cache-Control"] = "public, max-age=900"
    return response


@app.get("/api/map.png")
def map_png():
    request_started = time.perf_counter()
    model_id = request.args.get("model", "ifs_01")
    model = MODEL_CATALOG.get(model_id)
    if model is None:
        return jsonify({"error": "Unknown model selection."}), 400
    field_id = request.args.get("field", "wind")
    model_field_options = {item["id"]: item for item in model_fields(model_id)}
    if field_id not in model_field_options:
        return jsonify({"error": "That parameter is not available from the selected model."}), 400

    try:
        step = int(request.args.get("step", "24"))
    except (TypeError, ValueError):
        return jsonify({"error": "Forecast hour must be a number."}), 400
    if step not in model["steps"] or step not in model_field_options[field_id].get("steps", model["steps"]):
        return jsonify({"error": "That forecast hour is not available for this parameter on this model."}), 400

    default_units = "knots" if field_id in {"wind", "wind_200", "shear_200_850", "gust"} else ("°C" if field_id == "temperature" else ("hPa" if field_id == "mslp" else "mm"))
    units = request.args.get("units", default_units)
    allowed_units = {item["id"] for item in model_field_options[field_id]["units"]}
    if units not in allowed_units:
        return jsonify({"error": "Choose a display unit supported by the selected parameter."}), 400

    try:
        dpi = int(request.args.get("dpi", "150"))
    except (TypeError, ValueError):
        return jsonify({"error": "DPI must be 120, 150, or 300."}), 400
    if dpi not in {120, 150, 300}:
        return jsonify({"error": "DPI must be 120, 150, or 300."}), 400

    vector_field = field_id in {"wind", "wind_200"}
    speed_layer = request.args.get("speed", "1") == "1" if vector_field else False
    stream_layer = request.args.get("streamlines", "1") == "1" if vector_field else False
    if vector_field and not speed_layer and not stream_layer:
        return jsonify({"error": "Turn on at least one wind layer."}), 400

    image_name = f"baywind-{model_id}-{field_id}-f{step:03d}-{dpi}dpi.png"
    cache_unit = units.replace("/", "_").replace("°", "deg")
    image_cache = PNG_CACHE_DIR / f"map-v{RENDER_VERSION}-{model_id}-{field_id}-f{step:03d}-{cache_unit}-d{dpi}-s{int(speed_layer)}-l{int(stream_layer)}.png"
    metadata_cache = image_cache.with_suffix(".meta")

    def cached_png() -> tuple[bytes, str] | None:
        try:
            fresh = image_cache.exists() and (datetime.now(timezone.utc).timestamp() - image_cache.stat().st_mtime < PNG_CACHE_TTL)
            if not fresh or not metadata_cache.exists():
                return None
            init_text = metadata_cache.read_text(encoding="utf-8").strip()
            datetime.fromisoformat(init_text.replace("Z", "+00:00"))
            return image_cache.read_bytes(), init_text
        except (OSError, ValueError):
            return None

    cache_status = "miss"
    field_cache_status = "not-read"
    field_timings: dict[str, float] = {}
    render_ms = 0.0
    png_cache_read_ms = 0.0
    try:
        # A cached PNG includes its run-time sidecar, so repeat views skip both
        # GRIB loading and rendering without losing the response time headers.
        png_cache_started = time.perf_counter()
        cached = cached_png()
        png_cache_read_ms = (time.perf_counter() - png_cache_started) * 1000.0
        if cached is not None:
            cache_status = "png-hit"
            png, init_time = cached
        else:
            with _RENDER_LOCK:
                png_cache_started = time.perf_counter()
                cached = cached_png()
                png_cache_read_ms += (time.perf_counter() - png_cache_started) * 1000.0
                if cached is not None:
                    cache_status = "png-hit-after-wait"
                    png, init_time = cached
                else:
                    if field_id == "wind":
                        fields = get_forecast(model_id, step)
                        render_function = lambda: render_map(fields, model, step, units, speed_layer, stream_layer, dpi=dpi)
                    elif field_id == "wind_200":
                        fields = get_upperair_forecast(model_id, step, field_id)
                        render_function = lambda: render_map(fields, model, step, units, speed_layer, stream_layer, dpi=dpi, field_id=field_id)
                    elif field_id == "shear_200_850":
                        fields = get_upperair_forecast(model_id, step, field_id)
                        render_function = lambda: render_scalar_map(fields, model, step, field_id, units, dpi=dpi)
                    else:
                        fields = get_scalar_forecast(model_id, step, field_id)
                        render_function = lambda: render_scalar_map(fields, model, step, field_id, units, dpi=dpi)
                    field_cache_status = str(fields.get("_cache_status", "unknown"))
                    field_timings = dict(fields.get("_timings", {}))
                    init_time = str(fields.get("init") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
                    datetime.fromisoformat(init_time.replace("Z", "+00:00"))
                    render_started = time.perf_counter()
                    png = render_function()
                    render_ms = (time.perf_counter() - render_started) * 1000.0
                    thread_tag = threading.get_ident()
                    image_tmp = image_cache.with_name(image_cache.stem + f"-img-{thread_tag}.tmp")
                    meta_tmp = metadata_cache.with_name(metadata_cache.stem + f"-meta-{thread_tag}.tmp")
                    image_tmp.write_bytes(png)
                    meta_tmp.write_text(init_time, encoding="utf-8")
                    meta_tmp.replace(metadata_cache)
                    image_tmp.replace(image_cache)
    except Exception as exc:  # UI gets a concise, actionable provider error.
        app.logger.exception("Map generation failed for %s step %s", model_id, step)
        return jsonify({"error": str(exc) or "Could not retrieve this forecast."}), 502

    response = make_response(send_file(BytesIO(png), mimetype="image/png", download_name=image_name))
    response.headers["Cache-Control"] = "public, max-age=900"
    response.headers["X-Model-Name"] = _safe_header(model["name"])
    response.headers["X-Model-Run"] = init_time
    init_dt = datetime.fromisoformat(init_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    response.headers["X-Valid-Time"] = (init_dt + timedelta(hours=step)).strftime("%Y-%m-%d %H:%M UTC")
    response.headers["X-Forecast-Hour"] = str(step)
    response.headers["X-Data-Source"] = _safe_header(model["source_label"])
    response.headers["X-Parameter"] = _safe_header(FIELD_DEFINITIONS[field_id]["label"])
    response.headers["X-Display-Units"] = _safe_header(units)
    response.headers["X-Render-DPI"] = str(dpi)
    total_ms = (time.perf_counter() - request_started) * 1000.0
    field_cache_read_ms = float(field_timings.get("field_cache_read_ms", 0.0))
    fetch_ms = float(field_timings.get("fetch_ms", 0.0))
    decode_ms = float(field_timings.get("decode_ms", 0.0))
    save_ms = float(field_timings.get("save_ms", 0.0))
    response.headers["X-Baywind-PNG-Cache"] = cache_status
    response.headers["X-Baywind-Field-Cache"] = field_cache_status
    response.headers["X-Baywind-Timing-ms"] = (
        f"png-cache={png_cache_read_ms:.1f};field-cache-read={field_cache_read_ms:.1f};"
        f"fetch={fetch_ms:.1f};decode={decode_ms:.1f};field-save={save_ms:.1f};"
        f"render={render_ms:.1f};total={total_ms:.1f}"
    )
    response.headers["Server-Timing"] = (
        f"total;dur={total_ms:.1f}, png-cache;dur={png_cache_read_ms:.1f}, "
        f"field-cache-read;dur={field_cache_read_ms:.1f}, fetch;dur={fetch_ms:.1f}, "
        f"decode;dur={decode_ms:.1f}, render;dur={render_ms:.1f}"
    )
    app.logger.info(
        "map_timing model=%s step=%03d dpi=%d png_cache=%s field_cache=%s "
        "field_cache_read_ms=%.1f fetch_ms=%.1f decode_ms=%.1f field_save_ms=%.1f render_ms=%.1f total_ms=%.1f",
        model_id,
        step,
        dpi,
        cache_status,
        field_cache_status,
        field_cache_read_ms,
        fetch_ms,
        decode_ms,
        save_ms,
        render_ms,
        total_ms,
    )
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
