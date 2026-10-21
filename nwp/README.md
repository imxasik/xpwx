# Baywind — Bay of Bengal model wind viewer

A Render-ready Flask application that retrieves one selected forecast parameter from official model-data providers, crops it to the Bay of Bengal domain, and renders a weather-chart-style PNG. Supported charts include 10 m wind, mean sea-level pressure, 2 m temperature, accumulated precipitation, and 10 m gust where the official source publishes them. It does not fetch or reuse WeatherBell images, scrape its account area, or call third-party weather APIs.

The map presentation follows the supplied reference's clean forecast-chart layout (taller regional field, domain graticule, coastlines, streamlines, a smooth extended 135-knot legend, and in-map Max/@XP WEATHER labels), with original styling. Coarse official ensemble grids are bilinearly interpolated for display; forecast information itself remains sourced from the original model grid. Scalar charts retain the same 10 × 8 inch figure, domain, typography, map frame, source attribution, and bottom-centered legend; they add parameter-appropriate contours and units without changing the existing wind chart. It does not reuse WeatherBell artwork or branding.

## Quick parameter charts

The Map Layers selector exposes only parameters published by the selected model's configured official endpoint:

| Chart | Display | Currently enabled model families |
|---|---|---|
| 10 m Wind | Wind-speed shading and streamlines | All existing wind integrations |
| Mean Sea-Level Pressure | Shaded hPa field with labelled isobars; rounded lower-left Min/Max and a bold, centered minimum value with no callout box | ECMWF IFS/ENS, NOAA GFS/GEFS/NAEFS |
| 2 m Temperature | Fixed °C/°F colour scale with temperature contours | ECMWF IFS/ENS/AIFS, NOAA GFS/GEFS/NAEFS |
| Accumulated Precipitation | Fixed mm palette; GRIB accumulation interval is shown on the chart | ECMWF IFS/ENS/AIFS, NOAA GFS/GEFS |
| 6 h Precipitation | Exact six-hour amount in mm; cumulative ECMWF fields are differenced, adjacent NOAA 3 h intervals are summed only after their metadata is checked | ECMWF IFS/ENS/AIFS, NOAA GFS/GEFS |
| 10 m Wind Gust | Gust shading using the existing wind-speed units and colour language; ECMWF IFS uses `10fg3` while the indexed ENS member uses `10fg` | ECMWF IFS/ENS and NOAA GFS |

Availability is field- and lead-specific: ECMWF gust is limited to published short-interval leads; accumulations are not offered at forecast hour zero. ARPEGE stays disabled without the Météo-France credential and live coverage validation; MOGREPS-G's configured public NetCDF feed currently exposes the two 10 m wind variables only. Other models remain wind-only until a suitable official field, grid, lead range, and regional retrieval are verified. The selector never implies that an unverified parameter exists.

For speed, the app fetches only the selected model, forecast hour, and parameter. ECMWF uses a single-parameter Open Data request; NOAA scalar requests use NOMADS' regional filter. Scalar field and rendered-image caches include model, lead, and parameter. No model-wide prefetch, all-member calculation, or parameter batch is performed. Provider delays are outside the app's control, so one-minute rendering is an engineering target rather than a provider-time guarantee.

## Included model choices

| Selector | Official source | Notes |
|---|---|---|
| ECMWF IFS 0.1° | ECMWF Open Data | Highest-resolution IFS surface winds; the 0.1° tier may be published later than 0.25°. |
| ECMWF IFS 0.25° | ECMWF Open Data | Lighter, standard open-data grid; selected by default for quicker uncached loads. |
| ECMWF ENS · Member 1 0.25° | ECMWF Open Data | One official IFS ENS perturbed member (`stream=enfo`, `type=pf`, `number=1`), not an ensemble mean. The open ensemble-mean product does not provide 10 m U/V vectors. |
| ARPEGE 0.25° | Météo-France official WCS API | Regional WCS requests fetch only 10 m U/V over the Bay box. Requires a Météo-France API credential; shown disabled until configured. |
| ECMWF AIFS 0.25° | ECMWF Open Data | Operational deterministic AI forecast. |
| AIFS ENS · Control 0.25° | ECMWF Open Data | Control member only, **not** an ensemble mean. Clearly labelled to avoid presenting one member as consensus. |
| GFS 0.25° | NCEP NOMADS | 10 m wind plus supported scalar fields; each NOMADS filter request is regional and single-parameter. |
| AIGFS | NCEP NOMADS | Downloads only the 10 m U/V GRIB messages using the official index and HTTP byte ranges, then crops locally. |
| GEFS 0.5° ENS | NCEP NOMADS | Official precomputed ensemble-mean wind; MSLP, 2 m temperature, and precipitation are separately regional-filtered scalar fields. |
| NAEFS 0.5° ENS Mean | NCEP NOMADS | Official bias-corrected NAEFS mean wind, MSLP, and 2 m temperature, fetched through regional filters. |
| AIGEFS AI ENS | NCEP NOMADS | AI ensemble-mean 10 m wind messages fetched through official GRIB indexes and byte ranges. |
| HGEFS HYBRID ENS | NCEP NOMADS | Hybrid ensemble-mean 10 m wind messages; shorter published forecast-hour range. |
| GDPS 15 km | ECCC MSC Datamart (HPFX) | Official global deterministic 10 m U/V, cropped to the Bay after decode. |
| GEML 25 km | ECCC MSC Datamart (HPFX) | Official global AI-model 10 m U/V. |
| GEPS 0.5° ENS | ECCC MSC Datamart (HPFX) | Downloads the two all-member 10 m U/V files; computes the mean of control plus 20 perturbed members locally. |
| DWD ICON 13 km | DWD Open Data | Global triangular-grid 10 m winds; remapped to the map grid using DWD CLAT/CLON coordinates. |
| DWD AICON 13 km | DWD Open Data | Official global AI forecast; uses the ICON CLAT/CLON coordinates as specified by DWD, then regrids for display. |
| MOGREPS-G 20 km ENS Mean | Met Office public AWS/ASDI | Reads the two 10 m NetCDF variables, converts each member's wind speed/direction to U/V, and averages all 18 vectors. The public AWS copy has about 24-hour latency. |

All charts cover the Bay of Bengal. Wind charts use only the selected model's 10 m U/V; scalar charts use only their selected scalar field. NOAA GFS/GEFS/NAEFS use regional NOMADS filters. MOGREPS-G publishes 10 m speed and direction rather than vector components in the public NetCDF files; Baywind converts each of the 18 members to U/V and takes the vector mean, reading only the Bay crop after downloading the two required NetCDF files. The ECMWF ENS wind selector fetches one official perturbed member (`enfo`/`pf`, member 1), explicitly not an ensemble mean; it does not download all 50 members. Forecast lead-hour choices are configured per model family and validated on the server.

Performance safeguards: repeated PNG and decoded-field requests use the existing 90-minute cache; model/parameter selection remains lazy, and only the selected model, lead, and field are fetched. The service records fetch/decode/save/render timings in response headers and logs. Independent paired downloads stay under the shared two-request cap; NOAA scalar filter requests ask only for one parameter and the Bay domain. ECMWF wind remains one combined `ecmwf-opendata` retrieval for 10u+10v, while a scalar chart makes one separate request for its chosen parameter. ARPEGE wind requests use Bay-bounded WCS coverages; its small capabilities response is cached for five minutes. MOGREPS-G downloads just the two required wind NetCDF files concurrently, then caches the cropped vector mean; its public AWS copy is delayed by about 24 hours, so it is not presented as a current-cycle feed.

## Run locally

Requires Python 3.12. The `eccodes` wheel in `requirements.txt` bundles ecCodes for GRIB decoding; `h5py` reads the official MOGREPS-G NetCDF files, and SciPy is used only when remapping DWD's native unstructured grids. All public feeds work without a key except ARPEGE, which requires an API subscription/credential from Météo-France. Set `METEOFRANCE_API_KEY` (or `METEOFRANCE_BEARER_TOKEN`) in the server environment; do not commit credentials to the repository. Without it, ARPEGE remains visible but disabled in the model selector.

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

The existing map design is retained with a 10 × 8 inch figure (`figsize=(10, 8)`). The on-screen preview renders at 150 DPI for quicker display; the **HD PNG** button creates a separate 300-DPI export only when requested. The first uncached map still has to download the selected official model fields; ECMWF 0.1° is heavier, so IFS 0.25° is selected by default for quicker loads. Rendered PNGs are checked first and returned without reopening GRIB-derived arrays; map and cropped-field caches last 90 minutes. Coarse forecast fields are interpolated only for display, and DWD's triangular grid is remapped to a regional 0.1° grid from the official ICON CLAT/CLON coordinate fields. The remap table is bundled from the official DWD ICON coordinate fields and reused, avoiding large coordinate downloads/tree construction on a Render cold start. Streamlines are integrated on a coarser mesh. Render's ephemeral filesystem clears the cache after a restart, and forecast animation loads frames lazily rather than prefetching a large sequence.

## Data and attribution

- **ECMWF:** `https://data.ecmwf.int/forecasts/` (ECMWF Open Data; CC BY 4.0).
- **NOAA/NCEP:** `https://nomads.ncep.noaa.gov/` (GFS, GEFS, NAEFS, and other ensemble products).
- **Météo-France:** `https://portail-api.meteofrance.fr/` (ARPEGE official open API; requires a subscribed API credential; follow its Open Data licence and attribution terms).
- **Met Office:** `https://metoffice-global-ensemble-model-data.s3.eu-west-2.amazonaws.com/` (MOGREPS-G public AWS/ASDI data; follow the Met Office licence/attribution requirements and note the public copy's delay).
- **Environment and Climate Change Canada:** `https://hpfx.collab.science.gc.ca/` (MSC Datamart/HPFX; GDPS, GDPS-GEML and GEPS). Follow the MSC data licence and service-usage policy.
- **Deutscher Wetterdienst:** `https://opendata.dwd.de/weather/nwp/` (ICON and AICON). Follow DWD's Open Data terms and attribution requirements.
- Geographic outlines are Natural Earth 1:50m shapefiles bundled with the project; weather fields come only from the official providers listed above.

Google WeatherNext 3 is not connected: live data access requires [Google allowlisting](https://developers.google.com/weathernext/guides/access-forecast). A paid Cloud contract is not required for allowlisting; downloading the full raw ensemble from [Cloud Storage](https://developers.google.com/weathernext/guides/gcs) is requester-pays, so it is intentionally not fetched by this app. NOAA's [GraphCastGFS registry](https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/) does not provide a verified current operational feed for this app. The selector label AIGFS retains NOAA NCEP attribution and is not presented as a Google product. If a run/step is not available, the viewer shows a provider error instead of silently substituting a model or forecast hour.

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
- NOAA GFS/GEFS/NAEFS supported scalar products are subset at NOMADS. ECMWF scalar charts request one official parameter at a time. ARPEGE uses regional WCS subsetting for its currently configured wind layer. ECCC/DWD integrations remain 10 m U/V-only. MOGREPS-G's public NetCDF distribution is global, so the app downloads only the two 10 m wind variables and crops them to the Bay before calculation.
- DWD AICON starts at forecast hour +3 (there is no PT0 forecast file); DWD ICON provides hourly steps through +78 h and then 3-hourly steps. The catalog exposes the corresponding model-specific lead hours.
- Google WeatherNext access and a live GraphCastGFS source are not configured; no NOAA product is mislabeled as a Google model.
- Do not commit credentials. ARPEGE is the only newly added selector item that requires a secret (`METEOFRANCE_API_KEY` or `METEOFRANCE_BEARER_TOKEN`); the other new sources use public open-data endpoints.
