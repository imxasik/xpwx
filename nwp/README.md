# Baywind — Bay of Bengal model wind viewer

A Render-ready Flask application that downloads **only 10 m U/V wind** from official model-data providers, crops it to a Bay of Bengal domain, and renders a compact forecast graphic with shaded wind speed and streamlines. It does not fetch or reuse WeatherBell images, scrape its account area, or call third-party weather APIs.

The map presentation follows the supplied reference's clean forecast-chart layout (taller regional field, domain graticule, coastlines, streamlines, a smooth extended 135-knot legend, and in-map Max/@XP WEATHER labels), with original styling. Coarse official ensemble grids are bilinearly interpolated for display; forecast information itself remains sourced from the original model grid. It does not reuse WeatherBell artwork or branding.

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
| ECCC GDPS 15 km | ECCC MSC Datamart (HPFX) | Official global deterministic 10 m U/V, cropped to the Bay after decode. |
| ECCC GDPS-GEML 25 km | ECCC MSC Datamart (HPFX) | Official global AI-model 10 m U/V. |
| ECCC GEPS Mean 0.5° | ECCC MSC Datamart (HPFX) | Downloads the two all-member 10 m U/V files; computes the mean of control plus 20 perturbed members locally. |
| DWD ICON Global | DWD Open Data | Global triangular-grid 10 m winds; remapped to the map grid using DWD CLAT/CLON coordinates. |
| DWD AICON Global | DWD Open Data | Official global AI forecast; uses the ICON CLAT/CLON coordinates as specified by DWD, then regrids for display. |

All added products are official global fields that cover the Bay of Bengal. Fetching is limited to 10 m U/V wind products (and the DWD ICON coordinate fields needed for native-grid remapping). Regional cropping follows decode; ECCC GEPS is the exception to single-message products because the official all-member GRIB files contain all 21 ensemble members, which are averaged locally to avoid mislabeling a single member. Forecast lead-hour choices are configured per model family and validated on the server.

## Run locally

Requires Python 3.12. The `eccodes` wheel in `requirements.txt` bundles ecCodes for GRIB decoding; SciPy is used only when remapping DWD's native unstructured grids. No API keys or third-party weather services are needed for the integrated providers.

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

The existing map design is retained with a 10 × 8 inch figure (`figsize=(10, 8)`). The on-screen preview renders at 150 DPI for quicker display; the **HD PNG** button creates a separate 300-DPI export only when requested. The first uncached map still has to download the selected official model fields; ECMWF 0.1° is heavier, so IFS 0.25° is selected by default for quicker loads. Rendered PNGs are checked first and returned without reopening GRIB-derived arrays; map and cropped-field caches last 90 minutes. Coarse forecast fields are interpolated only for display, and DWD's triangular grid is remapped to a regional 0.1° grid from the official ICON CLAT/CLON coordinate fields. The cached remap table is reused. Streamlines are integrated on a coarser mesh. Render's ephemeral filesystem clears the cache after a restart, and forecast animation loads frames lazily rather than prefetching a large sequence.

## Data and attribution

- **ECMWF:** `https://data.ecmwf.int/forecasts/` (ECMWF Open Data; CC BY 4.0).
- **NOAA/NCEP:** `https://nomads.ncep.noaa.gov/` (GFS and ensemble products).
- **Environment and Climate Change Canada:** `https://hpfx.collab.science.gc.ca/` (MSC Datamart/HPFX; GDPS, GDPS-GEML and GEPS). Follow the MSC data licence and service-usage policy.
- **Deutscher Wetterdienst:** `https://opendata.dwd.de/weather/nwp/` (ICON and AICON). Follow DWD's Open Data terms and attribution requirements.
- Geographic outlines are Natural Earth 1:50m shapefiles bundled with the project; weather fields come only from the official providers listed above.

Google WeatherNext 3 is not connected: direct forecast access requires [Google allowlisting](https://developers.google.com/weathernext/guides/access-forecast) and a requester-pays [Cloud Storage project](https://developers.google.com/weathernext/guides/gcs) with billing, which is not configured here. NOAA's [GraphCastGFS registry](https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/) does not provide a verified current operational feed for this app. NOAA AIGFS remains correctly labeled as NOAA AIGFS and is not presented as Google GraphCast. If a run/step is not available, the viewer shows a provider error instead of silently substituting a model or forecast hour.

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
- The longitude/latitude crop is 70°E–110°E, 0°–30°N. Adjust `DOMAIN` in `weather.py` and `WEST/EAST/SOUTH/NORTH` in `renderer.py` together if you want a different extent.
- NOAA GFS/GEFS products are subset at NOMADS. ECMWF, ECCC GDPS/GEML, and DWD ICON/AICON provide global files; the app requests only 10 m U/V products, then crops or remaps to the selected regional grid. It does not download unrelated forecast variables.
- DWD AICON starts at forecast hour +3 (there is no PT0 forecast file); DWD ICON provides hourly steps through +78 h and then 3-hourly steps. The catalog exposes the corresponding model-specific lead hours.
- Google WeatherNext access and a live GraphCastGFS source are not configured; no NOAA product is mislabeled as a Google model.
- No secrets, API keys, or WeatherBell credentials are required for the integrated official data sources.
