# -*- coding: utf-8 -*-
"""
 Vertical Wind Shear (850–200 hPa) magnitude ANOMALY
   shear = |V200 − V850| = sqrt((u200−u850)² + (v200−v850)²)
   anomaly = AIFS forecast shear − NCEP 1991-2020 climo shear
 তীর চিহ্ন = forecast shear vector (V200 − V850)
 engine: aifs_core.py (একই ফোল্ডারে রাখতে হবে)
"""
import datetime
import numpy as np
import aifs_core as C

# ================= কনফিগারেশন =================
AUTO        = True
MANUAL_BASE = datetime.datetime(2026, 9, 12, 0)
N_DAYS      = 1            # কত দিনের গড়; LEAD_HOURS=L দিলে window টা L-এ শেষ হয়
                           #   (N=3, L=240 → +192/+216/+240 h → title '20–22 Sep')
                           #   N_DAYS=1 → single sample;  N_DAYS=0/None → L-এর snapshot
LEAD_HOURS  = 240         # None = +24…+24*N_DAYS;  অথবা 240 / [240] / [96,102,108,114] …
                           #   (৬ এর গুণিতক, 0-360)
BASE_HOURS  = (0, 6, 12, 18)   # কোন base run গুলো খোঁজা হবে
SHEAR_TOP   = 200          # upper level
SHEAR_BOT   = 850          # lower level
WORK_STRIDE = 2
SMOOTH_DEG  = 3.75
VLIM        = None         # None = auto, অথবা সংখ্যা (যেমন 10)
ARROWS      = True         # forecast shear vector তীর দেখাবে
OUT_FILE    = "VWS.png"
CACHE_DIR   = "aifs_cache"
CACHE_KEEP_DAYS = 3        # এর চেয়ে পুরনো ক্যাশ GRIB মুছে যাবে (দিন)
SHP_DIR     = "map"
# ==============================================

WANTS_AIFS = [("u", SHEAR_TOP), ("v", SHEAR_TOP), ("u", SHEAR_BOT), ("v", SHEAR_BOT)]
WANTS_LTM  = [("uwnd", SHEAR_TOP), ("vwnd", SHEAR_TOP),
              ("uwnd", SHEAR_BOT), ("vwnd", SHEAR_BOT)]


def main():
    C.prune_cache(CACHE_DIR, CACHE_KEEP_DAYS)
    steps = C.steps_from_config(N_DAYS, LEAD_HOURS)
    print("=" * 64)
    print(f"  AIFS  Vertical Wind Shear ({SHEAR_BOT}–{SHEAR_TOP} hPa) Anomaly")
    print("=" * 64)

    coast = C.load_coastlines(C.ensure_coastline(SHP_DIR))
    print(f"  {len(coast)} coastline segments.")

    print("\n[1/4] AIFS run খোঁজা …")
    base = C.find_latest_run(steps, BASE_HOURS) if AUTO else MANUAL_BASE
    valid = [base + datetime.timedelta(hours=s) for s in steps]
    period = C.period_text(valid, steps)
    print(f"    valid {period}   steps {steps}")

    print("\n[2/4] AIFS forecast u,v @ two levels …")
    lat, lon, fc = C.fetch_aifs_fields(base, steps, WANTS_AIFS, CACHE_DIR)

    print("\n[3/4] NCEP climatology uwnd,vwnd @ two levels …")
    latc, lonc, cl = C.fetch_ltm_fields([v.date() for v in valid], WANTS_LTM, CACHE_DIR)

    shearf = np.hypot(fc[("u", SHEAR_TOP)] - fc[("u", SHEAR_BOT)],
                      fc[("v", SHEAR_TOP)] - fc[("v", SHEAR_BOT)])
    shearc = np.hypot(cl[("uwnd", SHEAR_TOP)] - cl[("uwnd", SHEAR_BOT)],
                      cl[("vwnd", SHEAR_TOP)] - cl[("vwnd", SHEAR_BOT)])
    shearc = C.interp_to_grid(shearc, latc, lonc, lat, lon)
    vec_u = fc[("u", SHEAR_TOP)] - fc[("u", SHEAR_BOT)]
    vec_v = fc[("v", SHEAR_TOP)] - fc[("v", SHEAR_BOT)]

    s = WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat[sl], lon[sl]
        shearf, shearc = shearf[sl, sl], shearc[sl, sl]
        vec_u, vec_v = vec_u[sl, sl], vec_v[sl, sl]
    print(f"    grid {lat.size}×{lon.size} ({abs(lon[1]-lon[0]):.2f}°)")

    ddeg = abs(lon[1] - lon[0])
    anom = C.smooth2d(shearf - shearc, SMOOTH_DEG / ddeg)
    vlim = VLIM if VLIM else C.nice_vlim(anom)
    print(f"    shear fc {np.nanmin(shearf):6.1f} … {np.nanmax(shearf):6.1f} m/s")
    print(f"    anomaly  {np.nanmin(anom):7.1f} … {np.nanmax(anom):7.1f} m/s   →  vlim ±{vlim:g}")

    arrows = (C.smooth2d(vec_u, SMOOTH_DEG / ddeg),
              C.smooth2d(vec_v, SMOOTH_DEG / ddeg)) if ARROWS else None

    print("\n[4/4] Drawing …")
    C.draw_anomaly(lat, lon, anom, coast,
                   meta=dict(title=f"Vertical Wind Shear ({SHEAR_BOT}–{SHEAR_TOP} hPa) Anomaly",
                             period=period,
                             run_txt=f"Run: {base:%HZ} • {base:%-d %b %Y}"),
                   cb_label=f"Shear {SHEAR_BOT}–{SHEAR_TOP} hPa Anomaly (m/s)",
                   vlim=vlim, arrows=arrows, arrow_ref=10, out_file=OUT_FILE)


if __name__ == "__main__":
    main()
