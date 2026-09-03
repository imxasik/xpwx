# XPWEATHER Reanalysis Maps (NCEP/NCAR)

Interactively render NCEP/NCAR Reanalysis **anomaly maps** (obs minus the
1991–2020 daily climatology) from NOAA PSL, served through a Flask + React-style
web UI. Every product is data-driven — the engine fetches the raw
OPeNDAP fields, computes the diagnostics on the sphere, and renders a
full-width map. There is no hard-coded imagery.

## Project structure
The engine lives package-by-package under `pro/` so no single module balloons:

| Path | Purpose |
|------|---------|
| `pro/config.py` | Module constants + process-wide caches (datasets, fields, lat/lon, coast). |
| `pro/data.py` | OPeNDAP access, cached field means, coastline, date resolution. |
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
- **Hovmöller:** U850 (5°S–5°N) and χ200 (15°S–15°N), 120-day longitude–time.
- Frictional torque (τx/τy/|τ|), IVT, QG omega forcing, MSE, temperature
  advection, geostrophic/ageostrophic wind.

## Run locally
```bash
pip install -r requirements.txt
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
3. Fill the form: **Free** instance; Build `pip install -r requirements.txt`;
   Start `gunicorn -b 0.0.0.0:$PORT --timeout 120 app:app`
   (Render also auto-reads `render.yaml` if you choose **Blueprint**).
4. **Create Web Service** → open the public URL. ✅

### Notes
- `$PORT` is provided by Render (set it manually when testing locally).
- First (cold) map takes ~5–95 s depending on product (Hovmöller fetches 120
  daily fields); identical repeat requests hit an in-memory cache and are instant.
- Mountain Torque is intentionally **not** offered — the dataset carries no
  orography, so only frictional torque is physically derivable. No fabricated maps.
- Data source: `https://psl.noaa.gov/thredds/dodsC/Datasets/ncep/...`

## Add your own maps (no core edits, no full re-upload)
The `custom/` folder is a plugin slot. Drop a `.py` file (or a few) there and the
engine auto-discovers it — no `pro/` edit, and you only push that one file.
- **Tier 1 — config only.** Add a product that reuses a built-in `kind` (anomaly,
  χ/vtp, ψ/psi, IVT, Eady, Hovmöller, …) with just a config dict.
- **Tier 2 — new diagnostic.** Define a new `kind` with a `compute` function
  (and optional `render`).

See `custom/README.md` and `custom/_EXAMPLE.py`. Files starting with `_` are
skipped (safe reference templates).
