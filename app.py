"""
app.py — Flask app serving the NCEP/NCAR Reanalysis map products.

Endpoints
---------
GET  /            -> web page (product sidebar + controls + viewer)
GET  /products    -> JSON list of available products (auto-built from metmap)
GET  /generate    -> POST JSON {product, mode, date, n_days} -> PNG
GET  /health      -> health check for uptime monitors / Render
"""

import io
import os
import json
import datetime

from flask import Flask, render_template, request, send_file, jsonify

import pro as metmap
import gfs_web

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# Server-side render cache: identical key -> raw PNG bytes (near-instant).
# The engine also caches raw dataset/field fetches, so even a "cold" render of
# a period that was fetched before is cheap.
_cache = {}


def _serve_png(data):
    resp = send_file(io.BytesIO(data), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return render_template("index.html",
                           today=datetime.date.today().isoformat(),
                           default_n_days=metmap.DEFAULT_N_DAYS,
                           default_product=metmap.DEFAULT_PRODUCT,
                           products=metmap.list_products(),
                           groups=metmap.group_products(),
                           gfs_config=gfs_web.metadata())


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.datetime.utcnow().isoformat()})


@app.route("/products")
def products():
    return jsonify({"products": metmap.list_products(),
                    "groups": metmap.group_products(),
                    "default": metmap.DEFAULT_PRODUCT})


@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(silent=True) or {}
    product_id = body.get("product", metmap.DEFAULT_PRODUCT)
    mode = body.get("mode", "auto")
    manual_date = body.get("date")
    n_days = int(body.get("n_days", metmap.DEFAULT_N_DAYS))
    n_days = max(1, min(30, n_days))
    if product_id not in metmap.PRODUCTS:
        return jsonify({"error": f"unknown product '{product_id}'",
                        "code": "unknown_product"}), 400

    if mode == "manual":
        if not manual_date:
            return jsonify({"error": "manual mode requires a date"}), 400
        try:
            datetime.date.fromisoformat(manual_date)
        except ValueError:
            return jsonify({"error": "invalid date format, use YYYY-MM-DD"}), 400

    key = json.dumps([product_id, mode, manual_date, n_days])
    if key in _cache:
        return _serve_png(_cache[key])

    log = []
    try:
        buf, meta = metmap.generate(product_id=product_id, mode=mode,
                                    manual_date=manual_date, n_days=n_days,
                                    log=log)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("map generation failed")
        return jsonify({"error": str(exc), "code": "generation_failed",
                        "log": log[-40:]}), 500

    data = buf.getvalue()
    _cache[key] = data
    return _serve_png(data)


@app.route("/gfs/config")
def gfs_config():
    return jsonify(gfs_web.metadata())


@app.route("/gfs/generate", methods=["POST"])
def gfs_generate():
    body = request.get_json(silent=True) or {}
    variable_key = body.get("variable", "wind")
    level = int(body.get("level", 850) or 0)
    avg_days = int(body.get("avg_days", 0) or 0)
    step = int(body.get("step", gfs_web.gconfig.DEFAULT_STEP) or 0)
    compute_anom = bool(body.get("anomaly", False))
    region_id = str(body.get("region", "2"))

    try:
        data, meta = gfs_web.generate(
            variable_key=variable_key,
            level=level,
            avg_days=avg_days,
            step=step,
            compute_anom=compute_anom,
            region_id=region_id,
        )
        resp = _serve_png(data)
        resp.headers["X-GFS-Run"] = meta.get("run", "")
        resp.headers["X-GFS-Region"] = meta.get("region", "")
        resp.headers["X-GFS-Cache"] = str(meta.get("cache", False)).lower()
        if "seconds" in meta:
            resp.headers["X-GFS-Seconds"] = str(meta["seconds"])
        return resp
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("GFS map generation failed")
        return jsonify({
            "error": str(exc),
            "code": "gfs_generation_failed",
        }), 500


@app.route("/diff", methods=["POST"])
def diff():
    """Return one map of (Range A − Range B)."""
    body = request.get_json(silent=True) or {}
    product_id = body.get("product", metmap.DEFAULT_PRODUCT)
    date1, date2 = body.get("date1"), body.get("date2")
    n_days1 = int(body.get("n_days1", metmap.DEFAULT_N_DAYS))
    n_days2 = int(body.get("n_days2", metmap.DEFAULT_N_DAYS))
    n_days1 = max(1, min(30, n_days1))
    n_days2 = max(1, min(30, n_days2))
    inverse = bool(body.get("inverse", False))

    if product_id not in metmap.PRODUCTS:
        return jsonify({"error": f"unknown product '{product_id}'"}), 400
    if not date1 or not date2:
        return jsonify({"error": "both dates required for comparison"}), 400
    try:
        datetime.date.fromisoformat(date1)
        datetime.date.fromisoformat(date2)
    except ValueError:
        return jsonify({"error": "invalid date format, use YYYY-MM-DD"}), 400

    key = json.dumps(["diff", product_id, date1, n_days1, date2, n_days2, inverse])
    if key in _cache:
        return _serve_png(_cache[key])

    log = []
    try:
        buf, meta = metmap.generate_diff(product_id=product_id,
                                         date1=date1, n_days1=n_days1,
                                         date2=date2, n_days2=n_days2,
                                         inverse=inverse, log=log)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("diff generation failed")
        return jsonify({"error": str(exc), "code": "diff_failed",
                        "log": log[-40:]}), 500

    data = buf.getvalue()
    _cache[key] = data
    return _serve_png(data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
