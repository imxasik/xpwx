# Velocity-Potential & Wind Anomaly Map (200 hPa)

Generate the NCEP/NCAR Reanalysis velocity-potential & wind-anomaly map at 200 hPa,
with an interactive web interface. Built from a standalone script, now wrapped in a
reusable engine + a Flask web app so it runs online / on a server.

## Files
| File | Purpose |
|------|---------|
| `metmap.py` | Core map engine (data fetch, χ200 solve, rendering). Importable. |
| `app.py`    | Flask web app (UI + `POST /generate` + `/health`). |
| `templates/index.html` | Web page with date picker, N-days, Auto/Manual mode. |
| `requirements.txt` | Python dependencies (incl. gunicorn). |
| `Procfile` / `render.yaml` | Deployment configs. |
| `.gitignore` | Keeps `map/`, PNGs, caches out of git. |

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
- `POST /generate`    → returns a PNG
  ```json
  {"mode": "auto" | "manual", "date": "YYYY-MM-DD", "n_days": 5}
  ```
  - `auto`: end date = latest available in the current year's dataset
  - `manual`: end date = the date you pick
  - `n_days`: how many days to average (1–30), default 5

## Deploy on Render (free, from GitHub) — click-by-click

1. **Push this folder to a GitHub repo.**
   ```bash
   git init
   git add -A
   git commit -m "Velocity potential & wind anomaly map app"
   # create an empty repo on github.com, then:
   git remote add origin https://github.com/YOUR_USERNAME/vp-map.git
   git push -u origin main
   ```
   (Simpler: on github.com click **New repository**, name it, then upload these
   files via the web "uploading files" button.)

2. **Create the service on Render.** Go to https://render.com → sign up / log in →
   **New → Web Service**.

3. **Connect GitHub** → authorize the Render GitHub App → **select your repo**.

4. Fill the form:
   - **Name**: `vp-map` (becomes `vp-map.onrender.com`)
   - **Region**: nearest to you (e.g. Frankfurt / Oregon)
   - **Instance type**: **Free**
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn -b 0.0.0.0:$PORT app:app`
   - (Render auto-detects those from `Procfile`/`render.yaml` too.)

5. Click **Create Web Service**. Render builds (~1–2 min) and gives you a public
   URL like `https://vp-map.onrender.com`. Open it — done. ✅

### Notes
- **`$PORT` is provided by Render** (your local shell may not have it set — set it
  manually when testing locally).
- **Outbound internet is allowed on Render's free tier**, so the app reaches the
  NOAA PSL OPeNDAP feed. First map takes ~5–10 s (fetch + compute); identical
  repeat requests hit an in-memory cache and are instant.
- **Free tier spins down after inactivity**, so the first visitor waits ~30–60 s
  while the service boots. That's normal on the free plan.
- Data source: `https://psl.noaa.gov/thredds/dodsC/Datasets/ncep/...` — the server
  needs outbound access (any standard host has this).

## (Optional) Blueprint alternative
You can skip step 4 by choosing **New → Blueprint** and selecting the repo — the
included `render.yaml` configures the service automatically.
