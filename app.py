"""
app.py — Flask web app that runs the Velocity-Potential & Wind Anomaly map.

Endpoints
---------
GET  /            -> web page with controls (date picker, N-days, mode)
POST /generate    -> JSON {mode, date, n_days} -> PNG image of the map
GET  /health       -> simple health check for uptime monitors / Render
"""

import io
import os
import json
import datetime

from flask import Flask, render_template, request, send_file, jsonify

import metmap

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# A lightweight in-memory cache so repeated requests for the same setup are fast.
_cache = {}


@app.route("/")
def index():
    return render_template("index.html",
                           today=datetime.date.today().isoformat(),
                           default_n_days=metmap.DEFAULT_N_DAYS)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.datetime.utcnow().isoformat()})


@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "auto")
    manual_date = body.get("date")
    n_days = int(body.get("n_days", metmap.DEFAULT_N_DAYS))
    n_days = max(1, min(30, n_days))          # keep it reasonable
    hpa = float(body.get("hpa", metmap.DEFAULT_HPA))

    if mode == "manual":
        if not manual_date:
            return jsonify({"error": "manual mode requires a date"}), 400
        try:
            datetime.date.fromisoformat(manual_date)
        except ValueError:
            return jsonify({"error": "invalid date format, use YYYY-MM-DD"}), 400

    key = json.dumps([mode, manual_date, n_days, hpa])
    if key in _cache:
        buf = _cache[key]
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    log = []
    try:
        buf, meta = metmap.generate_map(mode=mode, manual_date=manual_date,
                                        n_days=n_days, log=log)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("map generation failed")
        return jsonify({
            "error": str(exc),
            "code": "generation_failed",
            "log": log[-40:],
        }), 500

    _cache[key] = buf
    buf.seek(0)
    resp = send_file(buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
