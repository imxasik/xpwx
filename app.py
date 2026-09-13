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
import aifs_web

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
                           gfs_config=gfs_web.metadata(),
                           aifs_config=aifs_web.metadata())


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


@app.route("/aifs/config")
def aifs_config():
    return jsonify(aifs_web.metadata())


@app.route("/aifs/generate", methods=["POST"])
def aifs_generate():
    """
    Enqueue an AIFS render job and return immediately.
    - Cache hit  → serves PNG directly (200 image/png).
    - Cache miss → starts background thread, returns JSON {job_id, status:"pending"} (202).
    Client polls /aifs/status/<job_id> until done.
    """
    body       = request.get_json(silent=True) or {}
    product_id = str(body.get("product", "vp"))
    level      = int(body.get("level", aifs_web.DEFAULT_LEVEL) or aifs_web.DEFAULT_LEVEL)
    avg_days   = body.get("avg_days")
    lead_hour  = body.get("lead_hour")
    avg_days   = int(avg_days)  if avg_days  not in (None, "", 0) else None
    lead_hour  = int(lead_hour) if lead_hour not in (None, "", 0) else None

    try:
        job_id, cached = aifs_web.enqueue(product_id=product_id, level=level,
                                          n_days=avg_days, lead_hours=lead_hour)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "bad_params"}), 400
    except Exception as exc:
        app.logger.exception("AIFS enqueue failed")
        return jsonify({"error": str(exc), "code": "enqueue_failed"}), 500

    if cached:
        meta = cached["meta"]
        resp = _serve_png(cached["png"])
        resp.headers["X-AIFS-Run"]    = meta.get("run", "")
        resp.headers["X-AIFS-Period"] = meta.get("period", "")
        resp.headers["X-AIFS-Cache"]  = "true"
        return resp

    return jsonify({"job_id": job_id, "status": "pending"}), 202


@app.route("/aifs/status/<job_id>")
def aifs_status(job_id):
    """
    Poll for a background AIFS render.
    Returns:
      - {status:"pending"} (202)        — still running
      - PNG bytes (200 image/png)       — done
      - {status:"error", error:"..."} (500) — failed
    """
    job = aifs_web.job_get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "unknown job_id"}), 404

    if job["status"] == "pending":
        return jsonify({"status": "pending"}), 202

    if job["status"] == "error":
        return jsonify({"status": "error", "error": job.get("error", "unknown")}), 500

    meta = job.get("meta") or {}
    resp = _serve_png(job["png"])
    resp.headers["X-AIFS-Run"]     = meta.get("run", "")
    resp.headers["X-AIFS-Period"]  = meta.get("period", "")
    resp.headers["X-AIFS-Cache"]   = "false"
    if "seconds" in meta:
        resp.headers["X-AIFS-Seconds"] = str(meta["seconds"])
    return resp


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
