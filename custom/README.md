# Add your own maps here — no core edits, no re-uploading

This folder is a **plugin slot**. Any `.py` file you drop in here is loaded
automatically when the engine starts, and its products appear in the sidebar
(`/products`) and work with `/generate` — without touching `pro/products.py`,
`pro/compute.py`, or `pro/render.py`.

- Files **starting with `_` are skipped** (so `_EXAMPLE.py` is just a reference
  template — copy it, rename without the underscore, edit, deploy).
- Edit a file → **redeploy / restart the server** → the product is live.
- The whole slot is one small file, so you can push a single file to GitHub
  rather than re-uploading the project.

## Two ways to add a product

### Tier 1 — reuse a built-in kind (no code, just a config block)
Copy a product block from `pro/products.py`, change the values, done. The
common `kind`s and their required keys:

| kind | what it shows | key extras you set |
|------|---------------|--------------------|
| `anom` | anomaly of one variable | `variable`, `level`, `plot_scale`, `vlim`, `cint` |
| `vtp`  | χ velocity-potential + wind | `level`, `plot_scale`, `vlim`, `cint`, `show_wind` |
| `psi`  | ψ streamfunction + wind | `level`, `plot_scale`, `vlim`, `cint`, `show_wind` |
| `waf`  | Takaya–Nakamura flux | `level`, `vec_scale`, `wind_scale`, `vlim`, `cint` |
| `qgpv` | QG PV anomaly | `level`, `plot_scale`, `vlim`, `cint` |
| `eddy` | eddy v′T′ / u′v′ | `level`, `flux` (`"vt"`/`"uv"`), `vlim`, `cint` |
| `eady` | Eady growth rate | `p_low`, `p_high`, `vlim`, `cint` |
| `ivt`  | integrated water vapour transport | `vlim`, `cint` |
| `qgforcing` | QG ω forcing | `level`, `vlim`, `cint` |
| `mse` | moist static energy | `level`, `vlim`, `cint` |
| `tadv`| temperature advection | `level`, `vlim`, `cint` |
| `geowind` / `ageowind` | geostrophic / ageostrophic wind | `level`, `vlim`, `cint` |
| `ft`   | frictional stress (τx/τy/|τ|) | `comp` (`"x"`/`"y"`/`"mag"`), `vlim` |
| `hov`  | longitude–time Hovmöller | `variable`, `level`, `lat_band`, `window`, `vlim` |

Every product dict also needs: `id`, `title`, `name`, `tag` (sidebar group),
`desc`, `kind`, and `cb_label`. Optional keys: `one_sided`, `invert_cbar`,
`show_wind`, `wind_scale`, `vec_scale`, `vec_ref`, `vec_unit`, `vec_step`,
`vec_min`.

### Tier 2 — a brand-new diagnostic (new `kind`)
Define a `KINDS` dict with a `compute` function (must return
`(lat, lon, data)` where `data["main"]` is the scaled 2-D field), plus an
optional `render` function. Copy `_EXAMPLE.py` for the full template.

## Quick reference (Tier 1)
```python
PRODUCTS = {
  "mywu500": {
     "id": "mywu500", "title": "My Zonal Wind Anomaly — 500 hPa",
     "name": "My U500", "tag": "Custom",
     "desc": "500-hPa zonal-wind anomaly.",
     "kind": "anom", "variable": "uwnd", "level": 500,
     "show_wind": False, "plot_scale": 1.0,
     "vlim": 10.0, "cint": 2.0,
     "cb_label": "Zonal Wind Anomaly  (m/s)",
  },
}
```
Save it as `custom/mywu500.py`, redeploy, and it appears under the **Custom**
group. That's it.

## What is already in here

| file | what it adds | sidebar group |
|------|--------------|---------------|
| `gwo.py` | GWO phase-space diagram (M, dM/dt) | Torque |
| `rel_aam.py` | Relative AAM: latitude–time section (anomaly shaded, total field contoured) + the global AAM anomaly, in the classic GSDM/AAM layout. Reuses `gwo.py`'s integration | Hovmöller |
| `_warm_cache.py` | Not a product (the leading `_` keeps it out of the sidebar): the deploy-time warm-up that builds `rel_aam`'s 1991–2020 climatology cache into `ltm_rel_aam_cache.npy` in the project root, so no request has to build it (512 MB hosts → see the main README's Render notes) | — |
| `mjo_rmm8.py` | MJO phase diagram (RMM, JMA) | Hovmöller |
| `kelvin_phase.py` | ECMWF IFS Kelvin-filtered VP200 phase-space forecast (15-day trajectory + CSV). Uses the `kelvin/` package and the shared `ecmwf/engine.py` client; forecast-only, so the DAYS box is hidden | Hovmöller |
| `time_height.py` | Time–height section (wind speed + barbs) | Hovmöller |
| `chi200.py`, `sst.py`, `tc.py`, `cyclone.py` | χ200, OISST/CoralTemp SST, TC diagnostics | Upper / SST / TC / Cyclone |

Addons may import each other (a plain `from custom import gwo as _gwo` works,
because the project root is on `sys.path`); keep shared physics in the file that
owns it so there is only ever one implementation to fix.
