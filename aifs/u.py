# -*- coding: utf-8 -*-
"""
 200 hPa Zonal Wind (U) ANOMALY  (ECMWF AIFS forecast − NCEP 1991-2020 climo)
 engine: aifs_core.py (একই ফোল্ডারে রাখতে হবে)
"""
import datetime
import numpy as np
import aifs_core as C

# ================= কনফিগারেশন =================
AUTO        = True
MANUAL_BASE = datetime.datetime(2026, 9, 12, 0)
N_DAYS      = 3            # কত দিনের গড়; LEAD_HOURS=L দিলে window টা L-এ শেষ হয়
                           #   (N=3, L=240 → +192/+216/+240 h → title '20–22 Sep')
                           #   N_DAYS=1 → single sample;  N_DAYS=0/None → L-এর snapshot
LEAD_HOURS  = 216         # None = +24…+24*N_DAYS;  অথবা 240 / [240] / [96,102,108,114] …
                           #   (৬ এর গুণিতক, 0-360)
BASE_HOURS  = (0, 6, 12, 18)   # কোন base run গুলো খোঁজা হবে
LEVEL_HPA   = 200
WORK_STRIDE = 2
SMOOTH_DEG  = 3.75
VLIM        = None         # None = auto, অথবা সংখ্যা (যেমন 20)
OUT_FILE    = "U.png"
CACHE_DIR   = "aifs_cache"
CACHE_KEEP_DAYS = 3        # এর চেয়ে পুরনো ক্যাশ GRIB মুছে যাবে (দিন)
SHP_DIR     = "map"
# ==============================================


def main():
    C.prune_cache(CACHE_DIR, CACHE_KEEP_DAYS)
    steps = C.steps_from_config(N_DAYS, LEAD_HOURS)
    print("=" * 64)
    print(f"  AIFS  {LEVEL_HPA} hPa Zonal Wind (U) Anomaly")
    print("=" * 64)

    coast = C.load_coastlines(C.ensure_coastline(SHP_DIR))
    print(f"  {len(coast)} coastline segments.")

    print("\n[1/4] AIFS run খোঁজা …")
    base = C.find_latest_run(steps, BASE_HOURS) if AUTO else MANUAL_BASE
    valid = [base + datetime.timedelta(hours=s) for s in steps]
    period = C.period_text(valid, steps)
    print(f"    valid {period}   steps {steps}")

    print("\n[2/4] AIFS forecast U …")
    lat, lon, fc = C.fetch_aifs_fields(base, steps, [("u", LEVEL_HPA)], CACHE_DIR)

    print("\n[3/4] NCEP climatology uwnd …")
    latc, lonc, cl = C.fetch_ltm_fields([v.date() for v in valid],
                                        [("uwnd", LEVEL_HPA)], CACHE_DIR)

    u_fc = fc[("u", LEVEL_HPA)]
    u_cl = C.interp_to_grid(cl[("uwnd", LEVEL_HPA)], latc, lonc, lat, lon)

    s = WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat[sl], lon[sl]
        u_fc, u_cl = u_fc[sl, sl], u_cl[sl, sl]
    print(f"    grid {lat.size}×{lon.size} ({abs(lon[1]-lon[0]):.2f}°)")

    anom = C.smooth2d(u_fc - u_cl, SMOOTH_DEG / abs(lon[1] - lon[0]))
    vlim = VLIM if VLIM else C.nice_vlim(anom)
    print(f"    anomaly {np.nanmin(anom):7.1f} … {np.nanmax(anom):7.1f} m/s   →  vlim ±{vlim:g}")

    print("\n[4/4] Drawing …")
    C.draw_anomaly(lat, lon, anom, coast,
                   meta=dict(title=f"{LEVEL_HPA} hPa Zonal Wind (U) Anomaly",
                             period=period,
                             run_txt=f"Run: {base:%HZ} • {base:%-d %b %Y}"),
                   cb_label=f"U{LEVEL_HPA} Anomaly (m/s)",
                   vlim=vlim, out_file=OUT_FILE)


if __name__ == "__main__":
    main()
