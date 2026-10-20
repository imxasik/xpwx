# Baywind — Bay of Bengal model wind viewer

A Render-ready Flask application that downloads **only 10 m U/V wind** from official model-data providers, crops it to a Bay of Bengal domain, and renders a compact forecast graphic with shaded wind speed and streamlines. It does not fetch or reuse WeatherBell images, scrape its account area, or call third-party weather APIs.

The map presentation follows the supplied reference's clean forecast-chart layout (header, domain graticule, coastlines, streamlines, and an extended 135-knot speed legend), with original styling and an `@XP WEATHER` watermark. It does not reuse WeatherBell artwork or branding.

## Included model choices

| Selector | Official source | Notes |
|---|---|---|
| ECMWF IFS 0.1° | ECMWF Open Data | Highest-resolution IFS surface winds; the 0.1° tier may be published later than 0.25°. |
| ECMWF IFS 0.25° | ECMWF Open Data | Lighter, standard open-data grid; selected by default for quicker uncached loads. |
| ECMWF AIFS Single 0.25° | ECMWF Open Data | Operational deterministic AI forecast. |
| ECMWF AIFS ENS Control 0.25° | ECMWF Open Data | Control member only, **not** an ensemble mean. Clearly labelled to avoid presenting one member as consensus. |
| NOAA GFS 0.25° | NCEP NOMADS | Requests only 10 m UGRD/VGRD and asks NOMADS' GRIB filter for the regional subset. |
| NOAA AIGFS | NCEP NOMADS | Downloads only the 10 m U/V GRIB messages using the official index and HTTP byte ranges, then crops locally. |
| NOAA GEFS Mean 0.5° | NCEP NOMADS | Uses the official precomputed ensemble-mean 10 m U/V fields; no local member averaging. |
| NOAA AIGEFS Mean | NCEP NOMADS | AI ensemble-mean 10 m wind messages fetched through official GRIB indexes and byte ranges. |
| NOAA HGEFS Mean | NCEP NOMADS | Hybrid ensemble-mean 10 m wind messages; shorter published forecast-hour range. |

These models cover the Bay of Bengal because they provide global fields. The implementation favors deterministic products and official precomputed ensemble means, while avoiding downloads of dozens of individual members or domain-limited regional products. Forecast lead-hour choices are configured per model family and validated on the server.

## Run locally

Requires Python 3.12. The `eccodes` wheel in `requirements.txt` bundles ecCodes for GRIB decoding; no separate GIS stack or key is needed.

```bash
docker build -t baywind .
docker run --rm -p 10000:10000 -e PORT=10000 baywind
```

Open `http://localhost:10000`.

For a non-Docker local install:

```bash
pip install -r requirements.txt
python app.py
```

## Deploy on Render

For a repository where this project is stored in `baywind/`, create **New → Web Service**, select the repository and branch, set **Root Directory** to `baywind`, choose **Docker**, and confirm **Free** before creating the service. The included `render.yaml` also specifies `plan: free` if you choose to use a Blueprint.

The first uncached map still has to download the chosen official model fields; ECMWF 0.1° is heavier, so IFS 0.25° is selected by default for quicker loads. Rendered PNGs are checked first and returned without reopening GRIB-derived arrays; both map and cropped-field caches last 90 minutes. 0.1° streamlines are integrated on a coarser mesh while wind-speed shading stays full resolution. Render's ephemeral filesystem clears the cache after a restart, and forecast animation loads frames lazily rather than prefetching a large sequence.

## Data and attribution

- **ECMWF:** `https://data.ecmwf.int/forecasts/` (ECMWF Open Data). ECMWF open data is distributed under **CC BY 4.0**; attribution is printed on the graphic and shown in the UI.
- **NOAA/NCEP:** `https://nomads.ncep.noaa.gov/` (including the NOMADS GFS regional GRIB filter and official AIGFS GRIB files).
- Geographic outlines are Natural Earth 1:50m shapefiles bundled with the project; the weather fields themselves come only from the two official providers above.

ECMWF 0.1° and specific recent forecast steps can arrive later than the 0.25° products. If a run/step is not yet available, the viewer shows a provider error rather than silently substituting a different model or lead hour.

## Project structure

```text
app.py                  Flask routes and forecast image API
weather.py              official-source retrieval, GRIB decoding, regional crop, cache
renderer.py             forecast graphic renderer
templates/index.html    viewer page
static/style.css        responsive UI
static/app.js           controls, loading states, animation, download
Dockerfile              Render-compatible image
render.yaml             Render Blueprint service
```

## Important notes

- Image is newly rendered from GRIB vectors; this is not a pixel-for-pixel copy of WeatherBell's proprietary map or branding.
- Forecast wind speed is computed from the selected model's 10 m U/V components. Streamline direction is integrated from those same components.
- The longitude/latitude crop is 70°E–107°E, 5°S–30°N. Adjust `DOMAIN` in `weather.py` and `WEST/EAST/SOUTH/NORTH` in `renderer.py` together if you want a different extent.
- NOAA GFS/GEFS products are subset at NOMADS. ECMWF open-data files are global grids, so the client selects only the two 10 m wind messages per lead and the app crops them immediately after decode; it does not download the full multi-parameter forecast bundle.
- No secrets, API keys, or WeatherBell credentials are required.
