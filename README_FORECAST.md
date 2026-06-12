# Live NBM WBGT Forecast — work environment

This is the **work copy** of `heat-stress-flask`, converted from the typed-input
tool into a **live forecast** tool. It pulls the National Blend of Models (NBM)
forecast for the **2–3 PM** window on the next few days, runs the *unchanged*
science pipeline (min/max models → z-score raster scaling → Liljegren WBGT), and
renders every map with a **black roads overlay** (`data/roads/roads.shp`).

Test here first. Deploy only after you're happy (see "Deploying" below).

## What changed vs. the original `app.py`

- **Input:** live NBM forecast instead of the typed form. The four model inputs
  (air temp, dew point, wind, cloud) come from the bbox-mean of the NBM 2–3 PM
  hours. NBM core has no surface-pressure element, so reference pressure falls
  back to 1008 mb (it barely affects WBGT).
- **Multi-day:** the next `WBGT_FCST_DAYS` days (default 3).
- **Background + cache:** the heavy WBGT solve runs in a background thread; each
  finished day is cached to `cache/` + `outputs/` and the page auto-updates.
- **Roads overlay** on all maps.
- **Units:** maps render in **both °F and °C**; the page is **°F-native** with a
  one-click toggle to °C (persists via localStorage). Summary values convert
  client-side.
- **Two tabs:** a **Forecast** tab and an **About & Methods** tab that describes
  the SERCC citizen-science mobile-transect campaign and the methods (sourced
  from the dissertation Ch.1–2 and sercc.com/wbgt-heat-mapping).
- **Same science:** `tg_iter` / `tnw_iter` / `wbgt_from_fields` are byte-for-byte
  the originals. `WBGT_STRIDE=1` (default) is the exact original computation.

## Run it locally (conda `work` env has the full stack)

```powershell
$py = "C:\Users\18286\miniconda3\envs\work\python.exe"
cd "C:\Users\18286\Desktop\heat-stress-flask-work"

# Fastest sanity check — one day, no NBM needed, fixed inputs (~25 s @ stride 8):
& $py app.py --selftest --stride 8

# Quick LIVE check — real NBM, 1 day, coarse/fast (~30–60 s):
& $py app.py --build --days 1 --stride 6

# Run the web app (production settings: full resolution, 3 days):
& $py app.py            # then open http://127.0.0.1:5000
```

When the server starts it kicks off the build in the background. The page is
live immediately and each day appears as it finishes.

### Speed knob: `WBGT_STRIDE`
The Liljegren solver runs **per pixel in pure Python**: the full 1385×1466 grid
is ~2.03 M pixels ≈ **7 minutes per day** at full resolution (`stride=1`).
`--stride N` solves every Nth pixel and nearest-fills (coarse but fast) for
testing only. **Production must use stride 1.** First full build of 3 days ≈ 20
min; after that everything is cached and instant until the next NBM run.

### Useful env vars
| var | default | meaning |
|-----|---------|---------|
| `WBGT_FCST_DAYS` | 3 | number of forecast days |
| `WBGT_STRIDE` | 1 | solve stride (1 = full res) |
| `WBGT_UNITS` | F | default display unit for the toggle (both °F and °C are always rendered) |
| `WBGT_LOCAL_TZ` | America/New_York | local time zone for the 2–3 PM window |
| `WBGT_ROADS_SHP` | data/roads/roads.shp | roads overlay source |
| `PORT` / `HOST` | 5000 / 0.0.0.0 | server bind |

### Endpoints
- `/` — forecast page (tabs per day, auto-polling)
- `/api/status` — JSON build state (what the page polls)
- `/api/refresh` — force re-detect the latest NBM run and rebuild
- `/healthz` — readiness + config
- `/outputs/<png>` — rendered maps

## Deploying to Render (read before you push)

The original deploy was `gunicorn app:app` on Render from GitHub. Three things
about this version need attention:

1. **GRIB decoding (ecCodes).** `requirements.txt` lists `eccodes`/`cfgrib`, but
   ecCodes needs a C library that Render's pip-only buildpack does not provide.
   **Recommended:** deploy with a Docker image (micromamba/conda) that installs
   `eccodes geopandas rasterio cfgrib`. Ask and I'll add a `Dockerfile`.
2. **Compute time + free tier.** ~7 min/day at full res on a full CPU; Render's
   free tier (0.1 CPU) makes that far longer, and it spins down after 15 min
   idle. Use at least a paid instance for real use.
3. **Cache persistence.** `cache/` + `outputs/` are the disk cache. Render's
   filesystem is **ephemeral** — add a **persistent disk** mounted at the repo
   dir (or change `CACHE_DIR`/`OUTPUT_DIR` to a mounted path) so a restart does
   not recompute everything.
   The `Procfile` is set to `--workers 1 --threads 8 --timeout 180` so the single
   shared background worker/state lives in one process.

If you'd rather keep the free tier and instant page loads, the clean fix is to
**vectorize the WBGT solver** (NumPy over the whole grid, ~1–3 s/day, output
verified to match the per-pixel solver). Say the word and I'll do that pass.

## Deploy = copy these back to the GitHub repo, then push
`app.py`, `Procfile`, `requirements.txt`, and (if not already committed)
`data/roads/`. The `data/models`, `data/rasters` are unchanged.
