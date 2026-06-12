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
- **Vectorized solver:** the Liljegren WBGT is solved over the whole grid with
  NumPy — verified **bit-identical** to the original per-pixel `tg_iter`/`tnw_iter`
  (`python app.py --verify`, max|diff| = 0.00 °C). Full-res is now **~30–50 s/day**
  (was ~7 min); a 3-day build is ~2–3 min, then cached.
- **Prebaked roads:** the overlay is precomputed to `data/roads_segments.json`
  (`python app.py --bake-roads`), so the running server needs **no geopandas/shapefile**.

## Run it locally (conda `work` env has the full stack)

```powershell
$py = "C:\Users\18286\miniconda3\envs\work\python.exe"
cd "C:\Users\18286\Desktop\heat-stress-flask-work"

& $py app.py --verify              # confirm vectorized == per-pixel solver
& $py app.py --bake-roads          # regenerate data/roads_segments.json (only if rasters change)
& $py app.py --selftest            # render one day from fixed inputs, no NBM (~30 s, full res)
& $py app.py --build --days 1      # real NBM, one day, full res (~45 s incl. download)
& $py app.py                       # the web app -> http://127.0.0.1:5000  (3 days)
```

When the server starts it kicks off the build in the background. The page is
live immediately and each day appears as it finishes (a few seconds each).

### Speed knob: `WBGT_STRIDE`
Now that the solver is vectorized, full resolution (`WBGT_STRIDE=1`, the default)
is fast — you should not need to change this. `--stride N>1` still exists as a
coarse/fast escape hatch but is rarely useful.

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

## Deploying to Render (Docker)

Deploy is via the included **`Dockerfile`** + **`render.yaml`** (Blueprint), not
the pip buildpack — the one hard dependency is **ecCodes** (for NBM GRIB2
decoding), a C library pip can't reliably install but conda-forge can.

1. **GRIB decoding (ecCodes).** Handled by the Dockerfile (micromamba/conda-forge).
   Roads are prebaked and geopandas was dropped, so the image is the science
   stack + ecCodes only.
2. **Compute.** The vectorized solver makes a 3-day build ~2–3 min, so even a
   small instance is fine. (`render.yaml` requests **Starter**; **Free** also
   works now — it just sleeps after 15 min and rebuilds on wake.)
3. **Cache.** `render.yaml` mounts a 1 GB disk at `/var/data` and points
   `WBGT_CACHE_DIR`/`WBGT_OUTPUT_DIR` there so maps survive restarts. Optional now
   that rebuilds are fast — drop the disk to run on Free.

Push the repo, then on Render: **New → Blueprint** (reads `render.yaml`) or a
Docker **Web Service**. See the deploy walkthrough for exact steps.

## Files that make up the deploy
`app.py`, `Dockerfile`, `env.yml`, `render.yaml`, `.dockerignore`,
`data/roads_segments.json` (prebaked), and the unchanged `data/models` +
`data/rasters`. The `data/roads/` shapefile is kept in git for re-baking but is
excluded from the image.
