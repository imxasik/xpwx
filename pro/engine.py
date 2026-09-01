"""pro.engine — top-level public API: generate / generate_diff."""
import datetime
from . import config, data, physics, products, compute as _compute, render as _render

DEFAULT_N_DAYS = config.DEFAULT_N_DAYS
DEFAULT_PRODUCT = config.DEFAULT_PRODUCT
PRODUCTS = products.PRODUCTS
_resolve_dates = data._resolve_dates
load_coastlines = data.load_coastlines
compute = _compute.compute
compute_hov = _compute.compute_hov
_band_label = _compute._band_label
render = _render.render
render_hov = _render.render_hov
render_rossby = _render.render_rossby

def generate(product_id=DEFAULT_PRODUCT, mode="auto", manual_date=None,
             n_days=DEFAULT_N_DAYS, log=None):
    pkg = PRODUCTS.get(product_id, PRODUCTS[DEFAULT_PRODUCT])
    say = (lambda m: log.append(m)) if log is not None else (lambda m: None)

    dates = _resolve_dates(mode, manual_date, n_days)

    if pkg["kind"] == "hov":
        say(f"[0/3] {pkg['title']} | {pkg.get('window', 120)}-day Hovmöller")
        say("[1/3] Fetching daily obs & climatology (cached) …")
        day_dates, lon, matrix = compute_hov(pkg, dates)
        say(f"  band {pkg['lat_band']} · {matrix.shape[0]} days × {matrix.shape[1]} lon")
        say("[2/3] Rendering Hovmöller …")
        lat_lab = _band_label(pkg["lat_band"])
        buf = render_hov(day_dates, lon, matrix, pkg, lat_lab=lat_lab)
        meta = {"product": pkg["id"], "title": pkg["title"], "hov": True,
                "date_start": day_dates[0].isoformat(),
                "date_end": day_dates[-1].isoformat(),
                "n_days": len(day_dates), "level": pkg["level"]}
        return buf, meta

    say(f"[0/4] {pkg['title']} | {dates[0]} → {dates[-1]} ({len(dates)}-day mean)")
    say("[0/4] Loading coastline …")
    coast_segs = load_coastlines()
    say(f"  {len(coast_segs)} segments loaded.")
    say("[1-2/4] Fetching obs & climatology (cached) …")
    lat, lon, data = compute(pkg, dates)
    say(f"  grid {lat.size}×{lon.size} @ {pkg['level']} hPa")
    say("[3/4] Rendering …")
    rfn = _compute._custom_render(pkg["kind"])
    if rfn is not None:
        buf = rfn(lat, lon, data, pkg, coast_segs, dates)
    elif pkg["kind"] == "rossby":
        buf = render_rossby(lat, lon, data, pkg, coast_segs, dates)
    elif _compute._custom_uses_builtin_render(pkg["kind"]):
        buf = render(lat, lon, data, pkg, coast_segs, dates)
    else:
        buf = render(lat, lon, data, pkg, coast_segs, dates)
    meta = {"product": pkg["id"], "title": pkg["title"],
            "date_start": dates[0].isoformat(), "date_end": dates[-1].isoformat(),
            "n_days": len(dates), "level": pkg["level"]}
    return buf, meta

def generate_diff(product_id=DEFAULT_PRODUCT, date1=None, n_days1=DEFAULT_N_DAYS,
                  date2=None, n_days2=DEFAULT_N_DAYS, inverse=False, log=None):
    """Return one map of (Range A − Range B), or (B − A) if inverse=True."""
    pkg = PRODUCTS.get(product_id, PRODUCTS[DEFAULT_PRODUCT])
    say = (lambda m: log.append(m)) if log is not None else (lambda m: None)

    dates_a = _resolve_dates("manual", date1, n_days1)
    dates_b = _resolve_dates("manual", date2, n_days2)
    say(f"[diff] {pkg['title']}: A={dates_a[0]}→{dates_a[-1]}  "
        f"B={dates_b[0]}→{dates_b[-1]}")

    say("[0] Loading coastline …")
    coast_segs = load_coastlines()

    say("[1] Computing Range A …")
    lat, lon, data_a = compute(pkg, dates_a)
    say("[2] Computing Range B …")
    _, _, data_b = compute(pkg, dates_b)

    say("[3] Difference …")
    sign = -1.0 if inverse else 1.0     # A−B default; B−A when inverse
    data = {"main": sign * (data_a["main"] - data_b["main"])}
    if "u" in data_a and "u" in data_b:
        data["u"] = sign * (data_a["u"] - data_b["u"])
        data["v"] = sign * (data_a["v"] - data_b["v"])

    tag = "B − A" if inverse else "A − B"     # only shown on the colorbar, not the title
    # concise title: just the product (no operation tag, no long date string)
    title = pkg["title"]
    buf = render(lat, lon, data, pkg, coast_segs, dates_a,
                 title=title, cbar_label=pkg["cb_label"] + f"  ({tag})")

    meta = {"product": pkg["id"], "title": pkg["title"],
            "date_start": dates_a[0].isoformat(), "date_end": dates_a[-1].isoformat(),
            "date_b_start": dates_b[0].isoformat(), "date_b_end": dates_b[-1].isoformat(),
            "n_days": len(dates_a), "level": pkg["level"], "diff": True, "inverse": inverse}
    return buf, meta
