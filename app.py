import os
import io
import json
import math
import pickle
import uuid
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, send_file, url_for
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except Exception:
    rasterio = None


app = Flask(__name__)

# ================================
# Constants
# ================================
STEFAN_BOLTZMANN = 5.6696e-8
Cp = 1003.5
M_AIR = 28.97
M_H2O = 18.015
R_GAS = 8314.34
R_AIR = R_GAS / M_AIR
Pr = Cp / (Cp + 1.25 * R_AIR)
EMIS_WICK = 0.95
ALB_WICK = 0.4
D_WICK = 0.007
L_WICK = 0.0254
EMIS_GLOBE = 0.95
ALB_GLOBE = 0.05
D_GLOBE = 0.0508
EMIS_SFC = 0.999
ALB_SFC = 0.45
MIN_SPEED = 0.13
CONVERGENCE = 0.02
MAX_ITER = 50
SITE_LAT = 35.934
SITE_LON = -79.064
SOLAR_CONSTANT = 1367.0
R = 287.05
g = 9.80665

# ================================
# Paths
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RASTER_DIR = os.path.join(DATA_DIR, "rasters")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMP_MAX_PKL = os.path.join(MODEL_DIR, "atobs_orange_temp_max.pkl")
TEMP_MIN_PKL = os.path.join(MODEL_DIR, "atobs_orange_temp_min.pkl")
DEW_MAX_PKL = os.path.join(MODEL_DIR, "atobs_orange_dew_max.pkl")
DEW_MIN_PKL = os.path.join(MODEL_DIR, "atobs_orange_dew_min.pkl")
SOL_MAX_PKL = os.path.join(MODEL_DIR, "atobs_only_sol_max.pkl")

TA_Z_TIF = os.path.join(RASTER_DIR, "ta_z_orange.tif")
TD_Z_TIF = os.path.join(RASTER_DIR, "td_z_orange.tif")
SOL_Z_TIF = os.path.join(RASTER_DIR, "sr_z_orange.tif")
ROUGHNESS_TIF = os.path.join(RASTER_DIR, "surface_roughness.tif")
ELEV_TIF = os.path.join(RASTER_DIR, "terrain.tif")

ATOBS_FEATURES = [
    "era_tair_C_at_obs",
    "era_dpt_C_at_obs",
    "era_wspd_ms_at_obs",
    "era_cloud_pct_at_obs",
]

# ================================
# Cached global objects
# ================================
MODELS = None
RASTERS = None


# ================================
# Helpers
# ================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models():
    return {
        "temp_max": load_pickle(TEMP_MAX_PKL),
        "temp_min": load_pickle(TEMP_MIN_PKL),
        "dew_max": load_pickle(DEW_MAX_PKL) if os.path.exists(DEW_MAX_PKL) else None,
        "dew_min": load_pickle(DEW_MIN_PKL) if os.path.exists(DEW_MIN_PKL) else None,
        "sol_max": load_pickle(SOL_MAX_PKL) if os.path.exists(SOL_MAX_PKL) else None,
    }


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        meta = src.meta.copy()
    return arr, meta


def load_rasters():
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")

    rasters = {}
    rasters["ta_z"], rasters["meta"] = read_raster(TA_Z_TIF)
    rasters["td_z"], _ = read_raster(TD_Z_TIF) if os.path.exists(TD_Z_TIF) else (None, None)
    rasters["sol_z"], _ = read_raster(SOL_Z_TIF) if os.path.exists(SOL_Z_TIF) else (None, None)
    rasters["z0"], rasters["z0_meta"] = read_raster(ROUGHNESS_TIF) if os.path.exists(ROUGHNESS_TIF) else (None, None)
    rasters["elev"], rasters["elev_meta"] = read_raster(ELEV_TIF) if os.path.exists(ELEV_TIF) else (None, None)
    return rasters


def ensure_loaded():
    global MODELS, RASTERS
    if MODELS is None:
        MODELS = load_models()
    if RASTERS is None:
        RASTERS = load_rasters()


def align_to_template(src_arr, src_meta, tmpl_meta, resampling=Resampling.bilinear):
    dst_arr = np.empty((tmpl_meta["height"], tmpl_meta["width"]), dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=tmpl_meta["transform"],
        dst_crs=tmpl_meta["crs"],
        resampling=resampling,
        dst_nodata=np.nan,
    )
    return dst_arr


def z_to_01(z):
    z_clipped = np.clip(np.where(np.isfinite(z), z, 0), -20, 20)
    out = 1.0 / (1.0 + np.exp(-z_clipped))
    out[~np.isfinite(z)] = np.nan
    return out


def scale_map(mod01, vmin, vmax):
    return (mod01 * (vmax - vmin)) + vmin


def downscale_wind_factor_z0(z0, z=2.0, zref=10.0):
    z0_eff = np.clip(z0.astype(np.float32), 1e-4, 1.5)
    numer = np.log(z / z0_eff)
    denom = np.log(zref / z0_eff)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = numer / denom
    f = np.where(np.isfinite(f) & (f > 0), f, np.nan)
    return f


def compute_u2_from_u10_and_z0(u10_ms, z0_aligned):
    return u10_ms * downscale_wind_factor_z0(z0_aligned, z=2.0, zref=10.0)


def calculate_pressure(elevation_m, temperature_c, ref_pressure_mb, ref_elevation_m):
    temperature_k = temperature_c + 273.15
    return ref_pressure_mb * np.exp(-g * (elevation_m - ref_elevation_m) / (R * temperature_k))


def esat(tk, phase=0):
    if phase == 0:
        y = (tk - 273.15) / (tk - 32.18)
        return 6.1121 * np.exp(17.502 * y)
    y = (tk - 273.15) / (tk - 0.6)
    return 6.1115 * np.exp(22.452 * y)


def viscosity(tk):
    return 1.458e-6 * tk**1.5 / (tk + 110.4)


def thermal_cond(tk):
    return (Cp + 1.25 * R_AIR) * viscosity(tk)


def h_cylinder(diameter, tk, pair_mb, speed):
    density = pair_mb * 100.0 / (R_AIR * tk)
    re = max(speed, MIN_SPEED) * density * diameter / viscosity(tk)
    nu = 0.281 * re**(1.0 - 0.4) * Pr**(1.0 - 0.56)
    return nu * thermal_cond(tk) / diameter


def emis_atm(tair_k, rh_frac):
    e = rh_frac * esat(tair_k)
    return 0.575 * e**0.143


def evap(tk):
    return (313.15 - tk) / 30.0 * (-71100.0) + 2.4073e6


def solar_declination(day_of_year):
    return math.radians(23.44) * math.sin(math.radians(360 / 365 * (day_of_year - 81)))


def equation_of_time(day_of_year):
    B = math.radians(360 / 365 * (day_of_year - 81))
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


def solar_zenith_cos(lat_deg, lon_deg, dt_local):
    day_of_year = dt_local.timetuple().tm_yday
    decl = solar_declination(day_of_year)
    standard_meridian = round(lon_deg / 15) * 15
    time_offset = equation_of_time(day_of_year) + 4 * (lon_deg - standard_meridian)
    solar_time = dt_local.hour + dt_local.minute / 60 + time_offset / 60
    hour_angle = math.radians(15 * (solar_time - 12))
    lat_rad = math.radians(lat_deg)
    cza = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    return max(cza, 0.00873)


def calculate_smax(solar_constant, cza, distance_au=1.0):
    return solar_constant * cza / (distance_au ** 2) if cza > 0 else 0.0


def calculate_fdir(solar, smax):
    solar = np.asarray(solar, dtype=float)
    if smax <= 0:
        return np.zeros_like(solar, dtype=float)
    s_star = solar / smax
    out = np.zeros_like(solar, dtype=float)
    mask = np.isfinite(s_star) & (s_star > 0)
    out[mask] = np.exp(3.0 - 1.34 * s_star[mask] - 1.65 / s_star[mask])
    np.clip(out, 0.0, 1.0, out=out)
    return out


def tg_iter(tair_k, rh_frac, pair_mb, speed_ms, solar_wm2, fdir, cza):
    tsfc = tair_k
    tglobe = tair_k
    for _ in range(MAX_ITER):
        tref = 0.5 * (tglobe + tair_k)
        h = h_cylinder(D_GLOBE, tref, pair_mb, speed_ms)
        new = (
            0.5 * (emis_atm(tair_k, rh_frac) * tair_k**4 + EMIS_SFC * tsfc**4)
            - h / (STEFAN_BOLTZMANN * EMIS_GLOBE) * (tglobe - tair_k)
            + solar_wm2 / (2.0 * STEFAN_BOLTZMANN * EMIS_GLOBE)
              * (1.0 - ALB_GLOBE)
              * (float(fdir) * (1.0 / (2.0 * cza) - 1.0) + 1.0 + ALB_SFC)
        ) ** 0.25
        if abs(new - tglobe) < CONVERGENCE:
            return new
        tglobe = 0.9 * tglobe + 0.1 * new
    return np.nan


def tnw_iter(tair_k, rh_frac, pair_mb, speed_ms, solar_wm2, fdir, cza):
    eair = rh_frac * esat(tair_k)
    z = np.log(max(eair, 1e-6) / (6.1121 * 1.004))
    tdew_k = 273.15 + 240.97 * z / (17.502 - z)
    twb = tdew_k
    for _ in range(MAX_ITER):
        tref = 0.5 * (twb + tair_k)
        h = h_cylinder(D_WICK, tref, pair_mb, speed_ms)
        ewick = esat(twb)
        den = max(pair_mb - ewick, 1e-3)
        new = tair_k - evap(tref) / (Cp * M_AIR / M_H2O) * (ewick - eair) / den * Pr**0.56
        if abs(new - twb) < CONVERGENCE:
            return new
        twb = 0.9 * twb + 0.1 * new
    return np.nan


def wbgt_from_fields(tair_c, td_c, solar_wm2, pair_mb, speed_ms, dt_local, lat=SITE_LAT, lon=SITE_LON):
    rh_pct = 100.0 * (
        np.exp((17.625 * td_c) / (243.04 + td_c)) /
        np.exp((17.625 * tair_c) / (243.04 + tair_c))
    )
    rh_pct = np.clip(rh_pct, 1.0, 100.0)
    rh = rh_pct / 100.0

    cza = solar_zenith_cos(lat, lon, dt_local)
    smax = calculate_smax(SOLAR_CONSTANT, cza, 1.0)
    fdir_arr = calculate_fdir(solar_wm2, smax)

    tair_k = tair_c + 273.15
    wbgt_k = np.full_like(tair_k, np.nan, dtype=np.float32)
    tnw_k = np.full_like(tair_k, np.nan, dtype=np.float32)
    tg_k = np.full_like(tair_k, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(tair_k) & np.isfinite(td_c) & np.isfinite(solar_wm2) &
        np.isfinite(pair_mb) & np.isfinite(speed_ms)
    )
    idxs = np.argwhere(valid)

    for r, c in idxs:
        T = float(tair_k[r, c])
        RH = float(rh[r, c])
        WS = max(float(speed_ms[r, c]), MIN_SPEED)
        P = float(pair_mb[r, c])
        S = float(solar_wm2[r, c])
        F = float(fdir_arr[r, c])

        tg = tg_iter(T, RH, P, WS, S, F, cza)
        tnw = tnw_iter(T, RH, P, WS, S, F, cza)
        if np.isfinite(tg) and np.isfinite(tnw):
            wbgt_k[r, c] = 0.1 * T + 0.2 * tg + 0.7 * tnw
            tg_k[r, c] = tg
            tnw_k[r, c] = tnw

    return wbgt_k, tg_k, tnw_k


def save_map(arr, title, out_path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_flag_map(wbgt_array, out_path, unit="C"):
    if unit.upper() == "F":
        bounds = [0, 80, 85, 88, 90, 120]
        title = "WBGT Flag Levels (°F)"
    else:
        bounds = [0.0, 26.7, 29.4, 31.1, 32.2, 50.0]
        title = "WBGT Flag Levels (°C)"

    colors = ["white", "green", "yellow", "red", "black"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(wbgt_array, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, ticks=bounds)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_output_name(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}.png"


def cleanup_old_outputs(max_files=100):
    files = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(".png")
    ]
    if len(files) <= max_files:
        return
    files.sort(key=os.path.getmtime)
    for f in files[:-max_files]:
        try:
            os.remove(f)
        except Exception:
            pass


# ================================
# Routes
# ================================
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None

    defaults = {
        "era_tair": 32.0,
        "era_dpt": 22.0,
        "era_wspd": 2.5,
        "era_cloud": 35.0,
        "ref_pressure_mb": 1008.0,
        "date_val": datetime.now().strftime("%Y-%m-%d"),
        "time_val": datetime.now().strftime("%H:%M"),
    }

    form_data = defaults.copy()

    if request.method == "POST":
        form_data["era_tair"] = request.form.get("era_tair", "32.0")
        form_data["era_dpt"] = request.form.get("era_dpt", "22.0")
        form_data["era_wspd"] = request.form.get("era_wspd", "2.5")
        form_data["era_cloud"] = request.form.get("era_cloud", "35.0")
        form_data["ref_pressure_mb"] = request.form.get("ref_pressure_mb", "1008.0")
        form_data["date_val"] = request.form.get("date_val", defaults["date_val"])
        form_data["time_val"] = request.form.get("time_val", defaults["time_val"])

        try:
            if rasterio is None:
                raise RuntimeError("rasterio is required for this app and is not installed.")

            ensure_loaded()

            era_tair = float(form_data["era_tair"])
            era_dpt = float(form_data["era_dpt"])
            era_wspd = float(form_data["era_wspd"])
            era_cloud = float(form_data["era_cloud"])
            ref_pressure_mb = float(form_data["ref_pressure_mb"])
            dt_local = datetime.strptime(
                f'{form_data["date_val"]} {form_data["time_val"]}',
                "%Y-%m-%d %H:%M"
            )

            X = np.array([[era_tair, era_dpt, era_wspd, era_cloud]], dtype=float)

            t_max = float(MODELS["temp_max"].predict(X)[0])
            t_min = float(MODELS["temp_min"].predict(X)[0])

            d_max = float(MODELS["dew_max"].predict(X)[0]) if MODELS["dew_max"] is not None else None
            d_min = float(MODELS["dew_min"].predict(X)[0]) if MODELS["dew_min"] is not None else None
            s_max = float(MODELS["sol_max"].predict(X)[0]) if MODELS["sol_max"] is not None else None

            ta_w = z_to_01(RASTERS["ta_z"])
            temp_map = scale_map(ta_w, t_min, t_max)

            dew_map = None
            if RASTERS["td_z"] is not None and d_min is not None and d_max is not None:
                td_w = z_to_01(RASTERS["td_z"])
                dew_map = scale_map(td_w, d_min, d_max)

            sol_map = None
            if RASTERS["sol_z"] is not None and s_max is not None:
                sol_w = z_to_01(RASTERS["sol_z"])
                sol_map = scale_map(sol_w, 0.0, s_max)

            u2_map = None
            if RASTERS["z0"] is not None:
                z0_arr = RASTERS["z0"]
                z0_meta = RASTERS["z0_meta"]
                tmpl_meta = RASTERS["meta"]
                needs_align = (
                    (z0_meta["crs"] != tmpl_meta["crs"]) or
                    (z0_meta["transform"] != tmpl_meta["transform"]) or
                    (z0_meta["width"] != tmpl_meta["width"]) or
                    (z0_meta["height"] != tmpl_meta["height"])
                )
                z0_aligned = align_to_template(z0_arr, z0_meta, tmpl_meta) if needs_align else z0_arr
                u2_map = compute_u2_from_u10_and_z0(era_wspd, z0_aligned)

            pressure_map = None
            if RASTERS["elev"] is not None:
                elev_arr = RASTERS["elev"]
                elev_meta = RASTERS["elev_meta"]
                tmpl_meta = RASTERS["meta"]
                needs_align = (
                    (elev_meta["crs"] != tmpl_meta["crs"]) or
                    (elev_meta["transform"] != tmpl_meta["transform"]) or
                    (elev_meta["width"] != tmpl_meta["width"]) or
                    (elev_meta["height"] != tmpl_meta["height"])
                )
                elev_aligned = align_to_template(elev_arr, elev_meta, tmpl_meta) if needs_align else elev_arr
                ref_elevation_m = float(np.nanmean(elev_aligned))
                pressure_map = calculate_pressure(elev_aligned, temp_map, ref_pressure_mb, ref_elevation_m)

            image_urls = {}

            temp_name = make_output_name("temp")
            save_map(temp_map, "Temperature (°C)", os.path.join(OUTPUT_DIR, temp_name), cmap="coolwarm")
            image_urls["temp_map"] = url_for("output_file", filename=temp_name)

            if dew_map is not None:
                dew_name = make_output_name("dew")
                save_map(dew_map, "Dew Point (°C)", os.path.join(OUTPUT_DIR, dew_name), cmap="BrBG")
                image_urls["dew_map"] = url_for("output_file", filename=dew_name)

            if sol_map is not None:
                sol_name = make_output_name("solar")
                save_map(sol_map, "Solar Radiation (W/m²)", os.path.join(OUTPUT_DIR, sol_name), cmap="inferno")
                image_urls["sol_map"] = url_for("output_file", filename=sol_name)

            if u2_map is not None:
                wind_name = make_output_name("wind")
                save_map(u2_map, "Wind Speed at 2 m (m/s)", os.path.join(OUTPUT_DIR, wind_name), cmap="viridis")
                image_urls["u2_map"] = url_for("output_file", filename=wind_name)

            if pressure_map is not None:
                pressure_name = make_output_name("pressure")
                save_map(pressure_map, "Surface Pressure (mb)", os.path.join(OUTPUT_DIR, pressure_name), cmap="viridis")
                image_urls["pressure_map"] = url_for("output_file", filename=pressure_name)

            wbgt_available = (
                dew_map is not None and
                sol_map is not None and
                u2_map is not None and
                pressure_map is not None
            )

            wbgt_warning = None
            summary = {
                "run_at": datetime.utcnow().isoformat() + "Z",
                "inputs": {
                    "era_tair_C_at_obs": era_tair,
                    "era_dpt_C_at_obs": era_dpt,
                    "era_wspd_ms_at_obs": era_wspd,
                    "era_cloud_pct_at_obs": era_cloud,
                    "ref_pressure_mb": ref_pressure_mb,
                    "wbgt_time_local": dt_local.strftime("%Y-%m-%d %H:%M"),
                },
                "synoptic_predictions": {
                    "temp_max_C": t_max,
                    "temp_min_C": t_min,
                    "dew_max_C": d_max,
                    "dew_min_C": d_min,
                    "sol_max_Wm2": s_max,
                },
            }

            if wbgt_available:
                wbgt_k, tg_k, tnw_k = wbgt_from_fields(
                    tair_c=temp_map,
                    td_c=dew_map,
                    solar_wm2=sol_map,
                    pair_mb=pressure_map,
                    speed_ms=u2_map,
                    dt_local=dt_local,
                    lat=SITE_LAT,
                    lon=SITE_LON,
                )

                wbgt_c = wbgt_k - 273.15
                tg_c = tg_k - 273.15
                tnw_c = tnw_k - 273.15
                wbgt_f = wbgt_c * 9 / 5 + 32

                wbgt_name = make_output_name("wbgt")
                save_map(wbgt_c, "WBGT (°C)", os.path.join(OUTPUT_DIR, wbgt_name), cmap="bwr")
                image_urls["wbgt_map"] = url_for("output_file", filename=wbgt_name)

                tg_name = make_output_name("tg")
                save_map(tg_c, "Black Globe Temperature (°C)", os.path.join(OUTPUT_DIR, tg_name), cmap="Reds")
                image_urls["tg_map"] = url_for("output_file", filename=tg_name)

                tnw_name = make_output_name("tnw")
                save_map(tnw_c, "Natural Wet Bulb Temperature (°C)", os.path.join(OUTPUT_DIR, tnw_name), cmap="BrBG")
                image_urls["tnw_map"] = url_for("output_file", filename=tnw_name)

                flag_c_name = make_output_name("flag_c")
                save_flag_map(wbgt_c, os.path.join(OUTPUT_DIR, flag_c_name), unit="C")
                image_urls["flag_c_map"] = url_for("output_file", filename=flag_c_name)

                flag_f_name = make_output_name("flag_f")
                save_flag_map(wbgt_f, os.path.join(OUTPUT_DIR, flag_f_name), unit="F")
                image_urls["flag_f_map"] = url_for("output_file", filename=flag_f_name)
            else:
                wbgt_warning = "WBGT step skipped because one or more required rasters or models are missing."

            cleanup_old_outputs()

            result = {
                "summary": summary,
                "synoptic_predictions": summary["synoptic_predictions"],
                "image_urls": image_urls,
                "wbgt_warning": wbgt_warning,
            }

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        form_data=form_data,
        result=result,
        error=error
    )


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)