"""Checks for the cyclone diagnostics on the synthetic case (TEST ONLY).

Needs the synthetic files (python tests/cyclone_synthetic.py) served as a PSL mirror:
  PSL_BASE_URL=http://127.0.0.1:8765/Datasets/ncep python tests/check_cyclone.py

Checks
  1. omega is zero at the top level (boundary condition of the continuity integral).
  2. omega from divergence matches the analytic truth at the centre of each level
     that the 2.5-degree grid resolves (200, 300, 500 hPa), within 15 %.
  3. The analysis recovers the known centre (within 0.5 deg of the truth at D) and the
     known motion (speed within 2 kt, bearing within 10 degrees).
"""
import datetime as dt
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from cyclone import analysis, core, fields as F  # noqa: E402

TRUTH = os.environ.get("TRUTH", "/tmp/cyclone_truth.json")


def main():
    spec = json.load(open(TRUTH))["synthetic"]
    truth = spec["truth"]
    D = dt.date.fromisoformat(spec["date"])
    ok = True

    g = F.read_day(D)
    lev, lat, lon = g["lev"], g["lat"], g["lon"]
    w = g["w"]
    # 1. boundary condition: omega = 0 at the first level of the integration (top of column)
    rng = np.random.default_rng(0)
    uu = rng.normal(size=(4, 9, 9)); vv = rng.normal(size=(4, 9, 9))
    la = np.linspace(-10, 10, 9); lo = np.linspace(80, 90, 9)
    om = core.omega_from_divergence(uu, vv, [10.0, 100.0, 300.0, 500.0], la, lo)
    print("1. omega at the top of the integration:", float(np.nanmax(np.abs(om[0]))), "Pa/s  (expect 0)")
    ok &= bool(np.nanmax(np.abs(om[0])) == 0.0)
    print("   output levels:", lev.tolist(), "(expect 1000 to 100 hPa)")
    ok &= bool(lev.min() >= 100.0 and lev.max() == 1000.0)

    print("2. omega vs analytic truth at the level centres (Pa/s)")
    for p in (200, 300, 500, 700, 850):
        k = int(np.argmin(np.abs(lev - p)))
        la, lo = truth["levels"][str(p)]
        i = int(np.argmin(np.abs(lat - la)))
        j = int(np.argmin(np.abs(lon - (lo % 360.0))))
        wd = float(w[k, i, j])
        wt = float(truth["omega_centre_Pa_s"][str(p)])
        rel = abs(wd - wt) / abs(wt)
        print(f"   {p:4d} hPa  derived {wd:+.4f}  truth {wt:+.4f}  rel.err {rel:.1%}")
        if p in (200, 300, 500):
            ok &= rel < 0.15

    print("3. analysis at the nominal centre")
    res = analysis.analyze(16.5, 84.5, D)
    c = res["center"]
    tc = truth["centers"][D.isoformat()]
    dlat, dlon = c["lat"] - tc[0], c["lon"] - tc[1]
    print(f"   centre {c['lat']:.2f}N {c['lon']:.2f}E  truth {tc[0]:.2f}N {tc[1]:.2f}E")
    ok &= abs(dlat) < 0.5 and abs(dlon) < 0.5
    m = res["motion"]
    print(f"   motion {m['speed_kt']:.2f} kt toward {m['bearing_deg']:.1f}  "
          f"(truth {spec['speed_kt']} kt toward {spec['bearing']})")
    ok &= abs(m["speed_kt"] - spec["speed_kt"]) < 2.0
    db = (m["bearing_deg"] - spec["bearing"] + 180.0) % 360.0 - 180.0
    ok &= abs(db) < 10.0

    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
