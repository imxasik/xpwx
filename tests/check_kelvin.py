"""Offline end-to-end check of the Kelvin VP200 product.

No internet access is needed: ``tests/kelvin_synthetic.py`` writes a tiny
ECMWF-open-data mirror holding an analytic eastward Kelvin wave, this script
serves it on 127.0.0.1, points the fetcher at it, and then exercises the real
public paths:

    fetch -> VP inversion -> Kelvin filter -> EOF phase -> PNG + CSV
    Flask /products, POST /generate, GET /kelvin/csv

Run from the project root (needs the app's dependencies installed):

    python tests/check_kelvin.py

Expected: a nearly circular trajectory advancing ~36 deg/day (10-day wave),
a valid PNG > 20 kB, and a CSV with one row per lead time.
"""
from __future__ import annotations

import datetime as dt
import http.server
import os
import shutil
import socketserver
import sys
import tempfile
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests import kelvin_synthetic as synth  # noqa: E402


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: D401 - silence the access log
        pass


def _serve(directory):
    handler = lambda *a, **k: _Quiet(*a, directory=directory, **k)  # noqa: E731
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    work = tempfile.mkdtemp(prefix="kelvin_check_")
    mirror = os.path.join(work, "mirror")
    cache = os.path.join(work, "cache")
    now = dt.datetime.now(dt.timezone.utc)
    # A complete run whose +360 h step exists and is not in the future.
    run_dt = (now - dt.timedelta(days=1)).replace(hour=12, minute=0, second=0,
                                                  microsecond=0)
    run = synth.write_mirror(mirror, run_dt)
    srv = _serve(mirror)

    try:
        from kelvin import fetch, pipeline, plot
        eng = fetch.engine()
        eng.BASE_URL = f"http://127.0.0.1:{srv.server_address[1]}/forecasts"
        eng.CACHE_DIR = cache

        fc = pipeline.build_forecast("auto")
        assert tuple(fc["run"]) == run, (fc["run"], run)
        # A manual date must resolve to the newest complete run on/before it.
        manual = pipeline.build_forecast("manual", date=run_dt.date())
        assert tuple(manual["run"]) == run, (manual["run"], run)
        assert fc["n_steps"] == 16, fc["n_steps"]
        assert np.all(np.isfinite(fc["x"])) and np.all(np.isfinite(fc["y"]))
        assert np.all(fc["amplitude"] > 0.2), fc["amplitude"]

        # Eastward propagation: unwrapped phase must advance, and 15 days of a
        # 10-day wave is ~540 degrees (allow for the EOF-projection ellipse).
        advance = np.degrees(np.unwrap(np.radians(fc["phase_deg"])))
        total = advance[-1] - advance[0]
        assert np.all(np.diff(advance) > 0), advance
        assert 400.0 < total < 700.0, total

        buf, n = plot.render_png_locked(fc, out_csv=os.path.join(work, "t.csv"))
        png = buf.getvalue()
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 20_000, len(png)
        assert n == 16
        with open(os.path.join(work, "t.csv"), encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        assert len(rows) == 17, len(rows)
        assert rows[0].startswith("run_date,run_hour,lead_hours")

        # The Flask paths the sidebar actually calls.
        import app as appmod
        client = appmod.app.test_client()
        listing = client.get("/products").get_json()
        product = [p for p in listing["products"] if p["id"] == "kelvin_phase"]
        assert product and product[0]["forecast"] and product[0]["csv"]
        group = [g for g in listing["groups"] if g["tag"] == "Hovmöller"][0]
        assert "kelvin_phase" in [p["id"] for p in group["product_list"]]

        resp = client.post("/generate", json={"product": "kelvin_phase",
                                              "mode": "auto"})
        assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
        assert resp.mimetype == "image/png" and len(resp.data) > 20_000

        resp = client.get("/kelvin/csv?mode=auto")
        assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
        assert resp.mimetype == "text/csv"
        assert resp.data.count(b"\n") == 17, resp.data[:200]

        resp = client.get("/kelvin/csv?mode=manual")
        assert resp.status_code == 400, resp.status_code
    finally:
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(work, ignore_errors=True)

    print("OK — Kelvin VP200 product: run %s %sZ, 16 daily fields, "
          "%.0f deg phase advance, PNG + CSV + Flask routes."
          % (run[0], run[1], total))


if __name__ == "__main__":
    main()
