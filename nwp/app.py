from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template, request, send_file

from renderer import render_map
from weather import MODEL_CATALOG, get_forecast

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
PNG_CACHE_DIR = Path(os.environ.get("BAYWIND_CACHE_DIR", "/tmp/baywind-cache"))
PNG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PNG_CACHE_TTL = int(os.environ.get("BAYWIND_CACHE_TTL", "5400"))
RENDER_VERSION = 5  # invalidate charts after 110E extent and 300-DPI title/legend updates
_RENDER_LOCK = threading.RLock()


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
            }
            for model_id, item in MODEL_CATALOG.items()
        ],
        "layers": ["10 m wind speed", "Streamlines"],
        "units": ["knots", "m/s", "km/h"],
    })


@app.get("/api/map.png")
def map_png():
    model_id = request.args.get("model", "ifs_01")
    model = MODEL_CATALOG.get(model_id)
    if model is None:
        return jsonify({"error": "Unknown model selection."}), 400

    try:
        step = int(request.args.get("step", "24"))
    except (TypeError, ValueError):
        return jsonify({"error": "Forecast hour must be a number."}), 400
    if step not in model["steps"]:
        return jsonify({"error": "That forecast hour is not available for this model."}), 400

    units = request.args.get("units", "knots").lower()
    if units not in {"knots", "m/s", "km/h"}:
        return jsonify({"error": "Choose knots, m/s, or km/h."}), 400

    speed_layer = request.args.get("speed", "1") == "1"
    stream_layer = request.args.get("streamlines", "1") == "1"
    if not speed_layer and not stream_layer:
        return jsonify({"error": "Turn on at least one map layer."}), 400

    image_name = f"baywind-{model_id}-f{step:03d}.png"
    cache_unit = units.replace("/", "_")
    image_cache = PNG_CACHE_DIR / f"map-v{RENDER_VERSION}-{model_id}-f{step:03d}-{cache_unit}-s{int(speed_layer)}-l{int(stream_layer)}.png"
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

    try:
        # A cached PNG includes its run-time sidecar, so repeat views skip both
        # GRIB loading and rendering without losing the response time headers.
        cached = cached_png()
        if cached is not None:
            png, init_time = cached
        else:
            with _RENDER_LOCK:
                cached = cached_png()
                if cached is not None:
                    png, init_time = cached
                else:
                    fields = get_forecast(model_id, step)
                    init_time = str(fields.get("init") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
                    datetime.fromisoformat(init_time.replace("Z", "+00:00"))
                    png = render_map(fields, model, step, units, speed_layer, stream_layer)
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
    response.headers["X-Model-Name"] = model["name"]
    response.headers["X-Model-Run"] = init_time
    init_dt = datetime.fromisoformat(init_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    response.headers["X-Valid-Time"] = (init_dt + timedelta(hours=step)).strftime("%Y-%m-%d %H:%M UTC")
    response.headers["X-Forecast-Hour"] = str(step)
    response.headers["X-Data-Source"] = model["source_label"]
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
