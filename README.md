# XPWEATHER Reanalysis Maps (NCEP/NCAR)

Interactively render NCEP/NCAR Reanalysis **anomaly maps** (obs minus the
1991–2020 daily climatology) from NOAA PSL, served through a Flask + React-style
web UI. Every product is data-driven — the engine downloads the raw NCEP/NCAR
netCDF fields from NOAA PSL once (cached locally), computes the diagnostics on
the sphere, and renders a full-width map. There is no hard-coded imagery.

## Project structure
The engine lives package-by-package under `pro/` so no single module balloons:

| Path | Purpose |
|------|---------|
| `pro/config.py` | Module constants (PSL base URL, cache dir) + process-wide caches (datasets, fields, lat/lon, coast). |
| `pro/data.py` | Downloads PSL netCDF files to `data_cache/`, reads them locally (`open_remote`), cached field means, coastline, date resolution. |
| `pro/physics.py` | Pure spherical/physical operators (divergence, vorticity, Poisson, Eady, Takaya–Nakamura, stationary wavenumber). |
| `pro/products.py` | `PRODUCTS` registry + sidebar grouping. |
| `pro/compute.py` | Per-product computation (obs − climatology anomalies). |
| `pro/render.py` | Matplotlib renderers (maps, Hovmöller, Rossby-wave composite). |
| `pro/engine.py` | Public API: `generate()` / `generate_diff()`. |
| `pro/addons.py` | Drop-in product loader (scans `custom/*.py`). |
| `pro/__init__.py` | Re-exports the public names. |
| `custom/` | **Your own maps.** Drop a `.py` file here (see `custom/README.md`). Files starting with `_` are skipped. |
| `metmap.py` | Thin backward-compatible shim → forwards to `pro`. |
| `app.py` | Flask web app (`GET /`, `/health`, `/products`; `POST /generate`). |
| `templates/index.html` | Web page (grouped product sidebar, date picker, viewer). |
| `requirements.txt` | Python dependencies (incl. gunicorn). |
| `render.yaml` | Render deployment config (no Procfile). |
| `.gitignore` | Keeps `map/`, PNGs, caches out of git. |

## Products (40, grouped in the sidebar)
Grouped by category: **Upper / Mid / Low / Dynamics / Thermo / Moisture /
Torque / Flow / Advanced / Surface / Hovmöller**. Examples:
- Velocity potential + wind (χ200/500/850) and streamfunction wave trains (ψ).
- Geopotential-height, zonal/meridional wind, temperature, RH, SLP anomalies.
- **Advanced:** Wave Flux (WAF), QG PV, eddy v′T′/u′v′, Eady growth rate.
- **Rossby-Wave Source, Train & Waveguide (200 hPa)** — new composite:
  χ200′ source shading (green = upper divergence / enhanced convection,
  brown = suppressed), ψ200′ wave-train contours (firebrick ridge / blue dashed
  trough), Takaya–Nakamura wave-activity-flux arrows, and the Ks≥5 stationary
  wavenumber waveguide band (indigo).
- **Hovmöller:** U850 (5°S–5°N) and χ200 (15°S–15°N), 120-day longitude–time;
  MJO phase diagram (RMM, JMA); Time–Height section; **Relative AAM** —
  latitude–time section of the vertically and zonally integrated AAM anomaly
  (1000–10 hPa) with the total field as contours, plus the global AAM anomaly
  (see below).
- Frictional torque (τx/τy/|τ|), IVT, QG omega forcing, MSE, temperature
  advection, geostrophic/ageostrophic wind.

## Run locally
```bash
pip install -r requirements.txt   # netCDF4 replaces the former pydap dependency
python app.py            # Flask dev server -> http://localhost:8000
# or the production way:
PORT=8000 gunicorn -b 0.0.0.0:$PORT app:app
```

## API
- `GET  /`            → web page
- `GET  /health`      → `{"status":"ok"}`
- `GET  /products`    → JSON product list + grouped sidebar data
- `POST /generate`    → returns a PNG
  ```json
  {"product": "rossby200", "mode": "auto" | "manual", "date": "YYYY-MM-DD", "n_days": 5}
  ```
  - `auto`: end date = latest available in the current year's dataset
  - `manual`: end date = the date you pick
  - `n_days`: how many days to average (1–30), default 5

## Deploy on Render (free, from GitHub)
1. Push to a GitHub repo.
2. Render → **New → Web Service** → connect GitHub → select the repo.
3. Fill the form: **Free** instance;
   Build `pip install -r requirements.txt && (python custom/_warm_cache.py || true)`;
   Start `gunicorn -b 0.0.0.0:$PORT --timeout 300 app:app`
   (Render also auto-reads `render.yaml` if you choose **Blueprint**).
4. **Create Web Service** → open the public URL. ✅

**Why the build step warms a cache.** A free instance has 0.1 CPU and **512 MB**
of RAM, and its disk is *ephemeral*: `data_cache/` is thrown away on every
deploy, restart and spin-down. The Relative AAM product needs the 1991–2020
daily climatology (366 days x 17 levels), and building that inside a request
peaked at ~470 MB — an OOM kill, which is what a 502/503 from `xpwx.onrender.com`
was. `custom/_warm_cache.py` builds it **once, during the build**, into
`ltm_rel_aam_cache.npy` in the project root, which the deployed image then keeps
(so no request ever pays for it: 494 MB -> 327 MB peak). `|| true` means a NOAA
hiccup cannot fail the deploy — the app then simply builds the cache on its
first request, which is serial and fits the box (measured 375 MB peak).

### Notes
- `$PORT` is provided by Render (set it manually when testing locally).
- First (cold) map takes ~5–95 s depending on product (Hovmöller fetches 120
  daily fields); identical repeat requests hit an in-memory cache and are instant.
- Mountain Torque is intentionally **not** offered — the dataset carries no
  orography, so only frictional torque is physically derivable. No fabricated maps.
- Data source: `https://downloads.psl.noaa.gov/Datasets/ncep/` (NCEP/NCAR Reanalysis
  netCDF files, e.g. `uwnd.2026.nc`, `uwnd.day.ltm.1991-2020.nc`). OISST SST comes from
  `https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/`.
- Files are downloaded on first use into `data_cache/` (git-ignored). Current-year
  files are re-downloaded after 12 h because NOAA appends to them; if a refresh
  fails, the cached copy is used. Override the sources with the `PSL_BASE_URL` and
  `OISST_BASE_URL` environment variables (e.g. for a local mirror).
- Storage: each full yearly file is roughly 50 MB; the OISST yearly file (used by
  the SST products in `custom/sst.py`) is roughly 740 MB. Make sure the host has
  enough disk space.
- Memory: the engine hands freed temporaries back to the OS after every request
  (`pro.data.release_memory`), keeps the PNG response cache bounded and builds the
  AAM climatology serially — together these keep a long session of map requests
  inside a 512 MB instance instead of slowly climbing into an OOM kill.
- **Data end date:** NOAA states NCEP/NCAR Reanalysis 1 production stopped with a
  last date of 17 March 2026, so "auto" mode ends at that date.

## Add your own maps (no core edits, no full re-upload)
The `custom/` folder is a plugin slot. Drop a `.py` file (or a few) there and the
engine auto-discovers it — no `pro/` edit, and you only push that one file.
- **Tier 1 — config only.** Add a product that reuses a built-in `kind` (anomaly,
  χ/vtp, ψ/psi, IVT, Eady, Hovmöller, …) with just a config dict.
- **Tier 2 — new diagnostic.** Define a new `kind` with a `compute` function
  (and optional `render`).

See `custom/README.md` and `custom/_EXAMPLE.py`. Files starting with `_` are
skipped (safe reference templates).

## MJO phase diagram (`rmm/`, sidebar product `mjo_rmm8`)

The RMM1/RMM2 phase diagram (last 40 days) uses the JMA MJO index (chi200 + u850 + u200):

    https://ds.data.jma.go.jp/tcc/tcc/products/clisys/mjo/figs/chi200-sst1_1991-2021/rmm8.csv

- `rmm/rmm8.py` downloads the CSV (cached, refreshed every `CACHE_TTL_HOURS`) and
  reads RMM1/RMM2 by header name. Set `RMM8_URL` to point to another copy.
- `rmm/plot_mjo.py` draws the diagram. The design code is the original script's, unchanged.
  `python -m rmm.plot_mjo [YYYY-MM-DD]` → `rmm/output/mjo_diagram.png`.
- `custom/mjo_rmm8.py` adds the sidebar product (Hovmöller group). Its window ends on the
  app's selected date, the same as the other products.

`rmm/ncep_rmm.py` and `rmm/velpot.py` (an NCEP-based index using a χ200 inversion) are kept
but are not used by the diagram.

## Relative AAM diagram (sidebar product `rel_aam`)

`custom/rel_aam.py` draws the **relative** angular momentum of the atmosphere —
the wind term of the global angular-momentum budget — vertically and zonally
integrated over 1000–10 hPa:

    A(phi, t) = (a³/g) ∬ u cos²φ dp dλ     [kg m² s⁻¹ per radian of latitude]
    Mr(t)     = ∫ A dφ                     [kg m² s⁻¹]

Two panels, both for the 120 days ending on the app's selected date:

1. **Latitude–time section.** Shading is the anomaly of `A` against the daily
   1991–2020 climatology, in the classic AAM palette — white inside ±0.5, then
   yellow/orange/red above and light-green/green/purple/blue below, with ticks
   at ±0.5 … ±2. As on the reference GSDM chart, the shaded anomaly is the
   value per latitude **grid box** (the native 2.5° spacing, ×2.5 relative to
   the per-degree contours), which is what gives the chart its large, readable
   patches; the numbers are therefore 10²⁴ kg m² s⁻¹ per 2.5° box. The black
   contours are the **total** field per degree, solid above zero and dashed
   below. The contour interval is chosen automatically from a list of round
   values so that every chart — quiet or strong — gets three to nine lines, and
   it is printed in the caption.
2. **Global AAM anomaly**, in Hadley units (10²⁵ kg m² s⁻¹), i.e. the numerator
   of the standardised anomaly M = Mr′ / σ_M used by the GWO phase diagram.

- The integration, the 2/2 latitude/longitude strides and the NOAA baseline
  sigmas (σ_M = 1.16×10²⁵ kg m² s⁻¹, σ_dM/dt = 1.5×10¹⁹ kg m² s⁻²) come from
  `custom/gwo.py`, and the section is the integrand of the GWO's Mr — so the two
  products can never disagree.
- The 1991–2020 latitude-resolved climatology is cached in
  `ltm_rel_aam_cache.npy` in the project root (same pattern as the GWO product's
  committed `ltm_aam_cache.npy`). It is built by `custom/_warm_cache.py` at
  deploy time, or by the first request on a plain host; the cache key ignores the
  file's mtime, so a re-downloaded LTM file never invalidates it.
- A 5-day running mean is applied to the section and the global curve (the plain
  running mean is linear, so the section integrates exactly to the curve).
- The latitude-resolved daily climatology is derived once from
  `uwnd.day.ltm.1991-2020.nc` and cached in `data_cache/ltm_rel_aam_cache.npy`
  (git-ignored); the committed `ltm_aam_cache.npy` used by the GWO product is
  never rewritten.
- The frame is square with inward ticks on all four sides, and the footer carries
  only BASE PERIOD (1991–2020) and LAST DAY USED, as on the reference chart.
- Checks (offline, on synthetic NCEP-format files): `tests/check_rel_aam.py` +
  `tests/aam_synthetic.py` — closed-form integration per latitude and globally,
  exact field ↔ Mr bookkeeping, an independent full-resolution quadrature, the
  contour-interval rule, smoothing properties, and an end-to-end `/generate`.
