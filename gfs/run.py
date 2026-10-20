"""
run.py  —  GFS Forecast Map Generator  (Interactive CLI)
==========================================================
চালাও:  python run.py

ধাপগুলো:
  1. ভ্যারিয়েবল সিলেক্ট
  2. Pressure level সিলেক্ট  (level-required vars এর জন্য)
  3. Average period / Precip total সিলেক্ট
  4. Anomaly (yes/no) — temp, mslp, rh, vp-র জন্য
  5. Region সিলেক্ট
  → ডেটা ফেচ → ম্যাপ আঁকো → PNG সেভ
"""

import sys, time, datetime

from climo import prefetch_ltm, wait_prefetch
from config import (
    VARIABLES, PRESSURE_LEVELS, AVERAGE_OPTIONS, REGIONS,
    LEVEL_REQUIRED_VARS, DEFAULT_STEP,
    ANOMALY_SUPPORTED_VARS,
)
import fetch as _fetch
import plot  as _plot


# ════════════════════════════════════════════════════════════════════
#  Terminal helpers
# ════════════════════════════════════════════════════════════════════

def _hr(char="═", n=62):
    print(char * n)

def _banner():
    _hr()
    print("  🌏  NCEP GFS Forecast Map Generator")
    print("  Data source : NOMADS  (nomads.ncep.noaa.gov)")
    print(f"  Current UTC : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    _hr()
    print()

def _menu(title: str, options: dict, key_fn=None) -> str:
    print(f"\n  ┌─ {title} {'─'*(54-len(title))}")
    for k, v in options.items():
        label = v["name"] if isinstance(v, dict) else v
        print(f"  │  [{k:>2}]  {label}")
    print("  └" + "─"*56)

    while True:
        raw = input("  ➤  আপনার পছন্দ নম্বর লিখুন: ").strip()
        if raw in options:
            return raw
        matches = [k for k, v in options.items()
                   if raw.lower() in (v["name"] if isinstance(v,dict) else v).lower()]
        if len(matches) == 1:
            return matches[0]
        print(f"  ✗  '{raw}' বৈধ নয়। আবার চেষ্টা করুন।")


def _confirm(summary: dict) -> bool:
    _hr("─")
    print("\n  ✅  নিশ্চিত করুন:\n")
    for k, v in summary.items():
        print(f"     {k:<22}: {v}")
    print()
    ans = input("  ➤  এগিয়ে যাব? [Y/n] : ").strip().lower()
    return ans in ("", "y", "yes", "হ্যাঁ", "ha")


# ════════════════════════════════════════════════════════════════════
#  Step-1 : Variable
# ════════════════════════════════════════════════════════════════════

def ask_variable() -> tuple:
    k = _menu("ভ্যারিয়েবল সিলেক্ট করুন", VARIABLES)
    return k, VARIABLES[k]["key"]


# ════════════════════════════════════════════════════════════════════
#  Step-2 : Pressure Level
# ════════════════════════════════════════════════════════════════════

def ask_level() -> int:
    opts = {k: {"name": f"{v} mb"} for k, v in PRESSURE_LEVELS.items()}
    k = _menu("Pressure Level সিলেক্ট করুন", opts)
    return PRESSURE_LEVELS[k]


# ════════════════════════════════════════════════════════════════════
#  Step-3 : Average / Total period
# ════════════════════════════════════════════════════════════════════

def ask_average(variable_key: str) -> tuple:
    """Returns (avg_days, forecast_step)."""
    opts = {k: {"name": v["label"]} for k, v in AVERAGE_OPTIONS.items()}
    k = _menu("Forecast Period সিলেক্ট করুন", opts)
    days = AVERAGE_OPTIONS[k]["days"]
    if days == 0:
        print()
        try:
            step = int(input(f"  ➤  Forecast step (ঘণ্টা, যেমন 18, 24, 48) [{DEFAULT_STEP}]: ").strip() or DEFAULT_STEP)
        except ValueError:
            step = DEFAULT_STEP
        return 0, step
    return days, DEFAULT_STEP


# ════════════════════════════════════════════════════════════════════
#  Step-4 : Anomaly (yes/no)
# ════════════════════════════════════════════════════════════════════

def ask_anomaly(variable_key: str) -> bool:
    """Only shown for variables that support anomaly computation."""
    if variable_key not in ANOMALY_SUPPORTED_VARS:
        return False
    print(f"\n  ┌─ Anomaly Option {'─'*39}")
    print(f"  │  [ 1]  No  — Normal map (actual values)")
    print(f"  │  [ 2]  Yes — Anomaly map (deviation from climatology)")
    print("  └" + "─"*56)
    while True:
        raw = input("  ➤  Anomaly চান? [1/2]: ").strip()
        if raw in ("1", "no",  "n", "না"):  return False
        if raw in ("2", "yes", "y", "হ্যাঁ"): return True
        print("  ✗  1 বা 2 লিখুন।")


# ════════════════════════════════════════════════════════════════════
#  Step-5 : Region
# ════════════════════════════════════════════════════════════════════

def ask_region() -> tuple:
    opts = {k: {"name": v["name"]} for k, v in REGIONS.items()}
    k = _menu("Region সিলেক্ট করুন", opts)
    r = REGIONS[k]
    return r["name"], r["bounds"]


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    _banner()

    # ── Step 1: Variable ────────────────────────────────────────────
    var_choice_k, variable_key = ask_variable()
    var_name = VARIABLES[var_choice_k]["name"]

    # ── Step 2: Pressure Level ──────────────────────────────────────
    if variable_key in LEVEL_REQUIRED_VARS:
        level = ask_level()
    else:
        level = 0

    # ── Step 3: Period ──────────────────────────────────────────────
    avg_days, step = ask_average(variable_key)

    # ── Step 4: Anomaly ─────────────────────────────────────────────
    compute_anom = ask_anomaly(variable_key)

    # Build the effective variable key for plot config lookup
    plot_variable = f"{variable_key}_anomaly" if compute_anom else variable_key

    # ── Step 5: Region ──────────────────────────────────────────────
    region_name, bounds = ask_region()
    lon_min, lon_max, lat_min, lat_max = bounds

    # ── Summary & Confirm ───────────────────────────────────────────
    period_label = (
        f"{avg_days}-day {'TOTAL' if variable_key=='precip' else 'average'}"
        if avg_days else f"Single snapshot (+{step}h)"
    )
    summary = {
        "ভ্যারিয়েবল"     : var_name,
        "Pressure Level"  : f"{level} mb" if level else "N/A (surface)",
        "Period"          : period_label,
        "Anomaly"         : "হ্যাঁ (Yes)" if compute_anom else "না (No)",
        "Region"          : region_name,
        "Domain"          : f"Lon {lon_min}–{lon_max}°  Lat {lat_min}–{lat_max}°",
    }
    if not _confirm(summary):
        print("\n  ✗  বাতিল করা হয়েছে।"); sys.exit(0)

    # ── Apply region ─────────────────────────────────────────────────
    _fetch.set_region(lon_min, lon_max, lat_min, lat_max)
    _fetch.set_step(step)
    _plot.set_region(lon_min, lon_max, lat_min, lat_max)

    # ── Run ──────────────────────────────────────────────────────────
    t0 = time.time()
    run_dt = _fetch.latest_gfs_run_dt()

    # ── [0] Prefetch LTM in background (parallel with shapefile load) ──
    ltm_thread = None
    if compute_anom:
        print("\n[0/3] Prefetching LTM climatology (background) ...")
        ltm_thread = prefetch_ltm(
            variable_key, lat_min, lat_max, lon_min, lon_max,
            level_hpa=level if level else None)

    _hr()
    print(f"\n  GFS Run  : {run_dt.strftime('%Y-%m-%d %HZ')}")
    print(f"  Variable : {var_name}" + (" [ANOMALY]" if compute_anom else ""))
    print(f"  Level    : {level} mb" if level else "  Level    : surface")
    print(f"  Period   : {period_label}")
    print(f"  Region   : {region_name}")
    _hr()

    # ── [1/3] Shapefiles ─────────────────────────────────────────────
    print("\n[1/3] Loading Shapefiles ...")
    country_segs, coast_segs = _plot.load_shapes()
    print(f"      OK  ({time.time()-t0:.1f}s)")

    # ── [2/3] Fetch ──────────────────────────────────────────────────
    print(f"\n[2/3] Fetching {var_name} ...")
    import numpy as np
    lat, lon, data, extra_data = _fetch.get_data(
        variable_key, level, avg_days, compute_anom=compute_anom)
    print(f"      Data range: {np.nanmin(data):.4f} – {np.nanmax(data):.4f}  "
          f"shape={data.shape}  ({time.time()-t0:.1f}s)")

    # ── Wait for LTM prefetch to finish ─────────────────────────────
    if compute_anom and ltm_thread is not None:
        wait_prefetch(ltm_thread)

    # ── [3/3] Draw ───────────────────────────────────────────────────
    print(f"\n[3/3] Drawing Map ...")
    actual_step = (avg_days * 24) if avg_days > 0 else step

    out_file = _plot.draw_map(
        run_dt,
        actual_step,
        country_segs, coast_segs,
        lat, lon, data,
        extra_data  = extra_data,
        variable    = plot_variable,
        level       = level,
        avg_days    = avg_days,
        region_name = region_name,
    )

    _hr()
    print(f"\n  ✓  সম্পন্ন!  ({time.time()-t0:.1f}s)  →  {out_file}\n")
    _hr()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ✗  ব্যবহারকারী বন্ধ করেছেন.")
        sys.exit(0)
