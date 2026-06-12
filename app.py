#!/usr/bin/env python3
r"""
Carrboro / Chapel Hill afternoon WBGT forecast web app (NBM-driven).

This is the live-forecast successor to the original typed-input tool. Instead of
the user entering ambient conditions, the app pulls the National Blend of Models
(NBM) forecast over the study-area bounding box for the 2-3 PM local window on
each of the next few days, reduces it to the four "at-observation" ambient inputs
the trained models expect (air temp, dew point, wind speed, cloud cover) plus a
reference surface pressure, and then runs the EXACT same science pipeline as
before:

    ambient -> min/max models -> z-score raster scaling -> Liljegren WBGT.

What changed vs. the original app:
  * Input source: live NBM forecast (no typed form).
  * Multi-day: the next N days at the 2-3 PM peak-heat window.
  * Heavy WBGT solve runs in a background thread and is cached to disk, so page
    loads are fast and partial progress survives restarts.
  * Every output map gets a black roads overlay (data/roads/roads.shp).

What did NOT change: every meteorological / WBGT function below is identical to
the original tool, so the numbers are the same science. The per-pixel Liljegren
solver (tg_iter / tnw_iter / wbgt_from_fields) is untouched.
"""

import argparse
import json
import math
import os
import pickle
import sys
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from flask import Flask, abort, jsonify, render_template_string, request, send_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import array_bounds
except Exception:
    rasterio = None

import requests


# ================================
# Runtime configuration for PROJ / ecCodes (helps on bundled Windows envs)
# ================================
def _configure_proj_runtime():
    try:
        import pyproj
    except Exception:
        return
    import pathlib
    candidates = []
    try:
        candidates.append(pathlib.Path(pyproj.datadir.get_data_dir()))
    except Exception:
        pass
    p = pathlib.Path(sys.prefix)
    candidates += [p / "Library" / "share" / "proj", p / "share" / "proj"]
    for c in candidates:
        if c and (c / "proj.db").exists():
            os.environ.setdefault("PROJ_DATA", str(c))
            os.environ.setdefault("PROJ_LIB", str(c))
            try:
                pyproj.datadir.set_data_dir(str(c))
            except Exception:
                pass
            return


def _configure_eccodes_runtime():
    try:
        import eccodes
    except Exception:
        return
    try:
        if not os.environ.get("ECCODES_DEFINITION_PATH"):
            os.environ["ECCODES_DEFINITION_PATH"] = eccodes.codes_definition_path()
        if not os.environ.get("ECCODES_SAMPLES_PATH"):
            os.environ["ECCODES_SAMPLES_PATH"] = eccodes.codes_samples_path()
    except Exception:
        pass


_configure_proj_runtime()
_configure_eccodes_runtime()


app = Flask(__name__)

# ================================
# Physical constants (UNCHANGED from original tool)
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

# WBGT solar geometry is standardized to the 2:15 PM mid-window time, exactly as
# the original tool. Ambient inputs now come from the NBM 2-3 PM window mean.
LOCKED_TIME_STR = "14:15"
LOCKED_TIME_LABEL = "2:15 PM Local Time"

# ================================
# Forecast configuration (new)
# ================================
N_FORECAST_DAYS = int(os.environ.get("WBGT_FCST_DAYS", "3"))
# Local clock hours that bracket the "2-3 PM" peak-heat window.
WINDOW_HOURS_LOCAL = [14, 15]
LOCAL_TZ_NAME = os.environ.get("WBGT_LOCAL_TZ", "America/New_York")
try:
    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME) if ZoneInfo else timezone(timedelta(hours=-5))
except Exception:
    LOCAL_TZ = timezone(timedelta(hours=-5))
    LOCAL_TZ_NAME = "US/Eastern(fixed)"

# Default display unit for the in-page toggle. Every map is rendered in BOTH
# units; this only sets which one shows first. Native = Fahrenheit.
DEFAULT_UNIT = os.environ.get("WBGT_UNITS", "F").upper()
if DEFAULT_UNIT not in ("C", "F"):
    DEFAULT_UNIT = "F"

# Resolution knob for the heavy WBGT solve. 1 = full resolution (production).
# Larger = coarse/fast smoke test (e.g. WBGT_STRIDE=8 for a quick local check).
WBGT_STRIDE = max(1, int(os.environ.get("WBGT_STRIDE", "1")))

RENDER_SCHEMA = "wbgt-nbm-forecast-v2"
ROAD_OVERLAY_VERSION = "roads-black-shp-v1"
RUN_DETECT_TTL_SEC = int(os.environ.get("WBGT_RUN_TTL", "1800"))

# ================================
# Paths
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RASTER_DIR = os.path.join(DATA_DIR, "rasters")
ROADS_DIR = os.path.join(DATA_DIR, "roads")
# Overridable so a Render persistent disk (e.g. /var/data) can hold the cache,
# which otherwise lives on Render's ephemeral filesystem and is lost on restart.
OUTPUT_DIR = os.environ.get("WBGT_OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
CACHE_DIR = os.environ.get("WBGT_CACHE_DIR", os.path.join(BASE_DIR, "cache"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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

ROADS_SHP = os.environ.get("WBGT_ROADS_SHP", os.path.join(ROADS_DIR, "roads.shp"))
# Precomputed road overlay (built offline by `--bake-roads`) so the running
# server needs neither geopandas nor the shapefile at runtime.
ROADS_SEGMENTS_JSON = os.environ.get("WBGT_ROADS_JSON", os.path.join(DATA_DIR, "roads_segments.json"))

ATOBS_FEATURES = [
    "era_tair_C_at_obs",
    "era_dpt_C_at_obs",
    "era_wspd_ms_at_obs",
    "era_cloud_pct_at_obs",
]

# ================================
# NBM (National Blend of Models) source configuration
# ================================
NBM_S3 = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
NBM_PATH_VARIANTS = [
    "blend.{ymd}/{hh:02d}/core/blend.t{hh:02d}z.core.f{fff:03d}.co.grib2",
    "blend.{ymd}/{hh:02d}/blend.t{hh:02d}z.core.f{fff:03d}.co.grib2",
]
NBM_TCDC_LEVELS = [
    "surface",                                          # NBM total sky cover
    "entire atmosphere (considered as a single layer)",
    "atmosphere single layer",
    "entire atmosphere",
]
NBM_WANTED = [
    ("TMP", ["2 m above ground"], "Ta_K"),
    ("DPT", ["2 m above ground"], "Td_K"),
    ("UGRD", ["10 m above ground"], "_U10"),
    ("VGRD", ["10 m above ground"], "_V10"),
    ("WIND", ["10 m above ground"], "WS_direct"),
    ("TCDC", NBM_TCDC_LEVELS, "Cloud_pct"),
    ("PRES", ["surface"], "Pres_Pa"),
    ("PRMSL", ["mean sea level"], "PRMSL_Pa"),
]
NBM_LOOSE_LEVEL_VARS = {"TCDC"}
NBM_MAX_FHR = 264
NBM_DEBUG = bool(int(os.environ.get("WBGT_NBM_DEBUG", "1")))
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 1.0
HEADERS = {"User-Agent": "carrboro-chapel-hill-wbgt-forecast/1.0"}
GRIB_READ_LOCK = threading.Lock()
_PLOT_LOCK = threading.Lock()

# ================================
# Cached global objects
# ================================
MODELS = None
RASTERS = None
_ROAD_SEG_CACHE = {}
_NBM_BBOX = None

# Forecast state shared between the background worker and the web routes.
_STATE_LOCK = threading.Lock()
FORECAST_STATE = {
    "schema": RENDER_SCHEMA,
    "status": "idle",        # idle | detecting | building | ready | error
    "message": "",
    "run_id": None,
    "run_label": None,
    "window_label": "2-3 PM local (mean of the 2 PM and 3 PM NBM hours)",
    "default_unit": DEFAULT_UNIT,
    "generated_at": None,
    "building": False,
    "days": [],
}
_BUILD_THREAD = None
_RUN_DETECT = {"ts": 0.0, "run_dt": None}


# ================================
# Unit helpers (UNCHANGED)
# ================================
def c_to_f(temp_c):
    return temp_c * 9.0 / 5.0 + 32.0


def f_to_c(temp_f):
    return (temp_f - 32.0) * 5.0 / 9.0


def ms_to_mph(speed_ms):
    return speed_ms * 2.2369362920544


def mph_to_ms(speed_mph):
    return speed_mph * 0.44704


# ================================
# Model / raster loading (UNCHANGED)
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


# ================================
# Liljegren WBGT energy balance (UNCHANGED per-pixel solver)
# ================================
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


def _wbgt_from_fields_scalar(tair_c, td_c, solar_wm2, pair_mb, speed_ms, dt_local, lat=SITE_LAT, lon=SITE_LON):
    """Original per-pixel reference solver. Kept only for the equivalence check
    (`python app.py --verify`). The vectorized wbgt_from_fields below is the
    production path and is verified equal to this to < 0.01 degC."""
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


# ---- Vectorized Liljegren solver (identical math, whole grid at once) ----
# Each function mirrors the scalar helper above but operates on numpy arrays.
def _esat_v(tk):
    y = (tk - 273.15) / (tk - 32.18)
    return 6.1121 * np.exp(17.502 * y)


def _viscosity_v(tk):
    return 1.458e-6 * tk ** 1.5 / (tk + 110.4)


def _thermal_cond_v(tk):
    return (Cp + 1.25 * R_AIR) * _viscosity_v(tk)


def _h_cylinder_v(diameter, tk, pair_mb, speed):
    density = pair_mb * 100.0 / (R_AIR * tk)
    re = np.maximum(speed, MIN_SPEED) * density * diameter / _viscosity_v(tk)
    nu = 0.281 * re ** (1.0 - 0.4) * Pr ** (1.0 - 0.56)
    return nu * _thermal_cond_v(tk) / diameter


def _emis_atm_v(tair_k, rh_frac):
    e = rh_frac * _esat_v(tair_k)
    return 0.575 * e ** 0.143


def _evap_v(tk):
    return (313.15 - tk) / 30.0 * (-71100.0) + 2.4073e6


def _tg_solve_v(T, RH, P, WS, S, F, cza):
    # tsfc == tair_k and the radiative + solar terms are constant across the
    # iteration, so compute them once (matches the scalar solver's values).
    base_const = (0.5 * (_emis_atm_v(T, RH) * T ** 4 + EMIS_SFC * T ** 4)
                  + S / (2.0 * STEFAN_BOLTZMANN * EMIS_GLOBE) * (1.0 - ALB_GLOBE)
                  * (F * (1.0 / (2.0 * cza) - 1.0) + 1.0 + ALB_SFC))
    tglobe = T.copy()
    result = np.full(T.shape, np.nan)
    done = np.zeros(T.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for _ in range(MAX_ITER):
            tref = 0.5 * (tglobe + T)
            h = _h_cylinder_v(D_GLOBE, tref, P, WS)
            new = (base_const - h / (STEFAN_BOLTZMANN * EMIS_GLOBE) * (tglobe - T)) ** 0.25
            conv = (~done) & (np.abs(new - tglobe) < CONVERGENCE)
            result[conv] = new[conv]
            done |= conv
            if done.all():
                break
            upd = ~done
            tglobe[upd] = 0.9 * tglobe[upd] + 0.1 * new[upd]
    return result


def _tnw_solve_v(T, RH, P, WS, S, F, cza):
    eair = RH * _esat_v(T)
    z = np.log(np.maximum(eair, 1e-6) / (6.1121 * 1.004))
    twb = (273.15 + 240.97 * z / (17.502 - z)).copy()
    result = np.full(T.shape, np.nan)
    done = np.zeros(T.shape, dtype=bool)
    coef = Cp * M_AIR / M_H2O
    with np.errstate(invalid="ignore"):
        for _ in range(MAX_ITER):
            tref = 0.5 * (twb + T)
            ewick = _esat_v(twb)
            den = np.maximum(P - ewick, 1e-3)
            new = T - _evap_v(tref) / coef * (ewick - eair) / den * Pr ** 0.56
            conv = (~done) & (np.abs(new - twb) < CONVERGENCE)
            result[conv] = new[conv]
            done |= conv
            if done.all():
                break
            upd = ~done
            twb[upd] = 0.9 * twb[upd] + 0.1 * new[upd]
    return result


def wbgt_from_fields(tair_c, td_c, solar_wm2, pair_mb, speed_ms, dt_local, lat=SITE_LAT, lon=SITE_LON):
    """Vectorized Liljegren WBGT over the whole grid — same math as the original
    per-pixel solver (_wbgt_from_fields_scalar), ~100-1000x faster. The per-pixel
    arrays are promoted to float64 so the result matches the scalar solver
    exactly (verified to < 0.01 degC via `python app.py --verify`)."""
    rh_pct = 100.0 * (
        np.exp((17.625 * td_c) / (243.04 + td_c)) /
        np.exp((17.625 * tair_c) / (243.04 + tair_c))
    )
    rh_pct = np.clip(rh_pct, 1.0, 100.0)
    rh = rh_pct / 100.0

    cza = float(solar_zenith_cos(lat, lon, dt_local))
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
    if not valid.any():
        return wbgt_k, tg_k, tnw_k

    T = np.asarray(tair_k[valid], dtype=np.float64)
    RH = np.asarray(rh[valid], dtype=np.float64)
    WS = np.maximum(np.asarray(speed_ms[valid], dtype=np.float64), MIN_SPEED)
    P = np.asarray(pair_mb[valid], dtype=np.float64)
    S = np.asarray(solar_wm2[valid], dtype=np.float64)
    F = np.asarray(fdir_arr[valid], dtype=np.float64)

    tg = _tg_solve_v(T, RH, P, WS, S, F, cza)
    tnw = _tnw_solve_v(T, RH, P, WS, S, F, cza)
    good = np.isfinite(tg) & np.isfinite(tnw)

    flat_idx = np.flatnonzero(valid)[good]
    wbgt_k.flat[flat_idx] = (0.1 * T[good] + 0.2 * tg[good] + 0.7 * tnw[good]).astype(np.float32)
    tg_k.flat[flat_idx] = tg[good].astype(np.float32)
    tnw_k.flat[flat_idx] = tnw[good].astype(np.float32)
    return wbgt_k, tg_k, tnw_k


def wbgt_maps_strided(tair_c, td_c, solar_wm2, pair_mb, speed_ms, dt_local,
                      lat=SITE_LAT, lon=SITE_LON, stride=1):
    """Speed knob ONLY. stride=1 is byte-identical to wbgt_from_fields. stride>1
    solves on a subsampled grid and nearest-fills back to full shape (coarse,
    for fast local smoke tests)."""
    if stride <= 1:
        return wbgt_from_fields(tair_c, td_c, solar_wm2, pair_mb, speed_ms, dt_local, lat, lon)
    H, W = tair_c.shape
    sl = (slice(None, None, stride), slice(None, None, stride))
    wk, gk, nk = wbgt_from_fields(tair_c[sl], td_c[sl], solar_wm2[sl], pair_mb[sl],
                                  speed_ms[sl], dt_local, lat, lon)

    def _up(a):
        up = np.repeat(np.repeat(a, stride, axis=0), stride, axis=1)
        return up[:H, :W]

    return _up(wk), _up(gk), _up(nk)


# ================================
# Roads overlay
# ================================
def _iter_line_coords(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield list(geom.coords)
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            yield list(part.coords)
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_line_coords(part)


def _build_road_segments(transform, h, w):
    """Build road polylines in array-index (col,row) space from the shapefile.
    Needs geopandas/shapely — used offline by --bake-roads and as a runtime
    fallback if the baked JSON is absent."""
    if not os.path.exists(ROADS_SHP):
        return []
    import geopandas as gpd
    from shapely.geometry import box
    left, bottom, right, top = array_bounds(h, w, transform)
    # Raster CRS is a Pseudo-Mercator LOCAL_CS; reproject roads to EPSG:3857
    # (same meters) so world coords line up with the raster transform.
    roads = gpd.read_file(ROADS_SHP)
    roads = roads[roads.geometry.notna()].to_crs("EPSG:3857")
    clip = box(left, bottom, right, top)
    roads = roads[roads.intersects(clip)]
    if roads.empty:
        return []
    clipped = roads.geometry.intersection(clip)
    inv = ~transform
    segments = []
    for geom in clipped:
        for coords in _iter_line_coords(geom):
            pts = []
            for xy in coords:
                col, row = inv * (xy[0], xy[1])
                pts.append((col, row))
            if len(pts) >= 2:
                segments.append(pts)
    return segments


def bake_roads():
    """Precompute the road overlay for the current raster grid and write it to
    ROADS_SEGMENTS_JSON so the running server needs no geopandas/shapefile."""
    ensure_loaded()
    meta = RASTERS["meta"]
    transform = meta["transform"]
    h, w = int(meta["height"]), int(meta["width"])
    segments = _build_road_segments(transform, h, w)
    payload = {
        "schema": "roads-seg-v1",
        "transform": [float(v) for v in tuple(transform)[:6]],
        "width": w, "height": h,
        "segments": [[[round(float(c), 1), round(float(r), 1)] for (c, r) in seg]
                     for seg in segments],
    }
    with open(ROADS_SEGMENTS_JSON, "w") as f:
        json.dump(payload, f)
    print("Baked {} road segments -> {}".format(len(segments), ROADS_SEGMENTS_JSON))
    return len(segments)


def _load_baked_roads(transform, h, w):
    """Load precomputed road segments if present and matching this grid."""
    if not os.path.exists(ROADS_SEGMENTS_JSON):
        return None
    try:
        with open(ROADS_SEGMENTS_JSON) as f:
            data = json.load(f)
    except Exception:
        return None
    if int(data.get("width", -1)) != w or int(data.get("height", -1)) != h:
        return None
    bt = data.get("transform") or []
    cur = [float(v) for v in tuple(transform)[:6]]
    if len(bt) != 6 or any(abs(float(a) - b) > 1e-6 for a, b in zip(bt, cur)):
        return None
    return [[(p[0], p[1]) for p in seg] for seg in data.get("segments", [])]


def road_segments_for_template(tmpl_meta):
    """Roads as [(col,row), ...] polylines in the raster's array-index space for
    a matplotlib LineCollection. Prefers the precomputed ROADS_SEGMENTS_JSON (no
    geopandas at runtime); falls back to reading the shapefile if it is absent."""
    transform = tmpl_meta["transform"]
    h, w = int(tmpl_meta["height"]), int(tmpl_meta["width"])
    key = (tuple(transform)[:6], w, h)
    if key in _ROAD_SEG_CACHE:
        return _ROAD_SEG_CACHE[key]
    segments = _load_baked_roads(transform, h, w)
    source = "baked"
    if segments is None:
        try:
            segments = _build_road_segments(transform, h, w)
            source = "shapefile"
        except Exception as exc:
            print("WARN: road overlay skipped: {}".format(exc))
            segments, source = [], "none"
    _ROAD_SEG_CACHE[key] = segments
    if NBM_DEBUG:
        print("Roads overlay: {} segments ({})".format(len(segments), source))
    return segments


def _add_roads(ax, segments):
    if segments:
        ax.add_collection(LineCollection(segments, colors="black",
                                         linewidths=0.5, alpha=0.85, zorder=5))


# ================================
# Map rendering (roads overlay added)
# ================================
def save_map(arr, title, out_path, cmap="viridis", roads=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap=cmap)
    _add_roads(ax, roads)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_flag_map(wbgt_array, out_path, unit="C", roads=None):
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
    _add_roads(ax, roads)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, ticks=bounds)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cleanup_old_outputs(max_files=240):
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
             if f.lower().endswith(".png")]
    if len(files) <= max_files:
        return
    files.sort(key=os.path.getmtime)
    for f in files[:-max_files]:
        try:
            os.remove(f)
        except Exception:
            pass


# ================================
# NBM download + decode
# ================================
def _http_get(session, url, headers=None, timeout=REQUEST_TIMEOUT):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return session.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last = exc
            time.sleep(SLEEP_BETWEEN_RETRIES * (attempt + 1))
    raise last


def _parse_idx(text):
    out = []
    for line in text.strip().splitlines():
        parts = line.strip().split(":")
        if len(parts) < 5:
            continue
        try:
            out.append({"idx": int(parts[0]), "offset": int(parts[1]),
                        "var": parts[3].strip(), "level": parts[4].strip()})
        except (ValueError, IndexError):
            continue
    return out


def _find_byte_ranges_for_nbm(entries):
    """Pick one message per wanted variable, honoring accept_levels as a
    PREFERENCE order (so e.g. TCDC 'surface' total sky cover wins over a
    'high cloud layer' message that happens to appear first)."""
    result = []
    for var, accept_levels, alias in NBM_WANTED:
        cand = [(i, e) for i, e in enumerate(entries) if e["var"] == var]
        if not cand:
            continue
        chosen = None
        for lvl in accept_levels:
            for i, e in cand:
                if e["level"] == lvl:
                    chosen = (i, e)
                    break
            if chosen:
                break
        if chosen is None and var in NBM_LOOSE_LEVEL_VARS:
            chosen = cand[0]
        if chosen is None:
            continue
        i, e = chosen
        start = e["offset"]
        end = entries[i + 1]["offset"] if i + 1 < len(entries) else None
        result.append((var, e["level"], start, end, alias))
    return result


def _nbm_path(run_dt, fff, vi=0):
    return NBM_PATH_VARIANTS[vi].format(ymd=run_dt.strftime("%Y%m%d"), hh=run_dt.hour, fff=fff)


def _idx_looks_real(text):
    if not text or "<Error>" in text[:200]:
        return False
    lines = text.strip().splitlines()
    return len(lines) >= 20 and any(":TMP:" in ln for ln in lines)


def fetch_idx(session, run_dt, fhr):
    for vi in range(len(NBM_PATH_VARIANTS)):
        url = "{}/{}.idx".format(NBM_S3, _nbm_path(run_dt, fhr, vi))
        try:
            r = _http_get(session, url, headers=HEADERS, timeout=10)
            text = r.text if r.status_code == 200 else ""
        except requests.RequestException:
            continue
        if r.status_code == 200 and _idx_looks_real(text):
            return text, vi
    return None, None


def find_latest_nbm_run(session, look_back_hours=14):
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if NBM_DEBUG:
        print("Looking for latest NBM core run back {}h from {}z".format(
            look_back_hours, now.strftime("%Y-%m-%d %H")))
    for hours_ago in range(look_back_hours):
        run = now - timedelta(hours=hours_ago)
        idx_text, _vi = fetch_idx(session, run, 1)
        if idx_text is not None:
            if NBM_DEBUG:
                print("  -> using NBM run {}z".format(run.strftime("%Y-%m-%d %H")))
            return run
    return None


def _read_grib_message_eccodes(data_bytes, tmp_dir):
    _configure_eccodes_runtime()
    try:
        import eccodes
    except Exception:
        return None, None, None
    fd, tmp_path = tempfile.mkstemp(suffix=".grib2", dir=tmp_dir)
    os.close(fd)
    gid = None
    try:
        with open(tmp_path, "wb") as f:
            f.write(data_bytes)
        with GRIB_READ_LOCK:
            with open(tmp_path, "rb") as f:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    return None, None, None
                vals = np.asarray(eccodes.codes_get_values(gid), dtype=np.float64)
                lats = np.asarray(eccodes.codes_get_array(gid, "latitudes"), dtype=np.float64)
                lons = np.asarray(eccodes.codes_get_array(gid, "longitudes"), dtype=np.float64)
        lons = np.where(lons > 180.0, lons - 360.0, lons)
        return lats.flatten(), lons.flatten(), vals.flatten()
    except Exception as exc:
        print("WARN: ecCodes GRIB decode failed: {}".format(exc))
        return None, None, None
    finally:
        if gid is not None:
            try:
                eccodes.codes_release(gid)
            except Exception:
                pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _read_grib_message_cfgrib(data_bytes, tmp_dir):
    try:
        import xarray as xr
    except Exception:
        return None, None, None
    fd, tmp_path = tempfile.mkstemp(suffix=".grib2", dir=tmp_dir)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(data_bytes)
        with GRIB_READ_LOCK:
            try:
                ds = xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
            except Exception:
                return None, None, None
            try:
                if not list(ds.data_vars) or "latitude" not in ds.coords or "longitude" not in ds.coords:
                    return None, None, None
                v = list(ds.data_vars)[0]
                vals = np.asarray(ds[v].values).flatten()
                lats = np.asarray(ds["latitude"].values).flatten()
                lons = np.asarray(ds["longitude"].values).flatten()
            finally:
                ds.close()
        lons = np.where(lons > 180.0, lons - 360.0, lons)
        return lats, lons, vals
    finally:
        for fn in os.listdir(tmp_dir):
            fp = os.path.join(tmp_dir, fn)
            if fp == tmp_path or fp.startswith(tmp_path + "."):
                try:
                    os.remove(fp)
                except OSError:
                    pass


def _read_grib_message(data_bytes, tmp_dir):
    lats, lons, vals = _read_grib_message_eccodes(data_bytes, tmp_dir)
    if lats is not None:
        return lats, lons, vals
    return _read_grib_message_cfgrib(data_bytes, tmp_dir)


def fetch_nbm_hour_points(session, run_dt, fhr, bbox, tmp_dir):
    """Return {alias: 1-D array of bbox-subset values} for one NBM forecast hour,
    or None if not available."""
    idx_text, vi = fetch_idx(session, run_dt, fhr)
    if idx_text is None:
        return None
    grb_url = "{}/{}".format(NBM_S3, _nbm_path(run_dt, fhr, vi))
    ranges = _find_byte_ranges_for_nbm(_parse_idx(idx_text))
    if not ranges:
        return None

    lat_arr = mask = None
    point_data = {}
    for var, level, start, end, alias in ranges:
        headers = dict(HEADERS)
        headers["Range"] = ("bytes={}-{}".format(start, end - 1)
                            if end is not None else "bytes={}-".format(start))
        try:
            rr = _http_get(session, grb_url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            continue
        if rr.status_code not in (200, 206):
            continue
        lats, lons, vals = _read_grib_message(rr.content, tmp_dir)
        if lats is None:
            continue
        if mask is None:
            lat_arr = lats
            mask = ((lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"]) &
                    (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"]))
            if not mask.any():
                return None
        if vals.shape != lat_arr.shape:
            continue
        point_data[alias] = vals[mask]

    if mask is None or "Ta_K" not in point_data:
        return None
    point_data["_n"] = int(mask.sum())
    return point_data


def _finite_mean(a):
    if a is None or len(a) == 0:
        return float("nan")
    b = np.asarray(a, dtype=float)
    b = b[np.isfinite(b)]
    return float(np.mean(b)) if b.size else float("nan")


def ambient_from_points(merged):
    """Reduce concatenated NBM bbox points to the 4 model inputs + ref pressure."""
    ta = _finite_mean(merged.get("Ta_K"))
    td = _finite_mean(merged.get("Td_K"))
    u = merged.get("_U10")
    v = merged.get("_V10")
    if u is not None and v is not None:
        ws = _finite_mean(np.sqrt(np.asarray(u, float) ** 2 + np.asarray(v, float) ** 2))
    else:
        ws = _finite_mean(merged.get("WS_direct"))
    cloud = _finite_mean(merged.get("Cloud_pct"))
    pres_pa = merged.get("Pres_Pa")
    if pres_pa is None or not np.isfinite(_finite_mean(pres_pa)):
        pres_pa = merged.get("PRMSL_Pa")
    pres_mb = _finite_mean(pres_pa) / 100.0 if pres_pa is not None else float("nan")

    return {
        "era_tair_C": (ta - 273.15) if np.isfinite(ta) else float("nan"),
        "era_dpt_C": (td - 273.15) if np.isfinite(td) else float("nan"),
        "era_wspd_ms": ws if np.isfinite(ws) else 2.0,
        "era_cloud_pct": cloud if np.isfinite(cloud) else 30.0,
        # NBM core carries no surface-pressure element; fall back to the
        # original tool's reference default for the study area (~150 m elev).
        "ref_pressure_mb": pres_mb if np.isfinite(pres_mb) else 1008.0,
        "n_points": int(merged.get("_n_total", 0)),
    }


# ================================
# NBM bounding box derived from the raster footprint
# ================================
def get_nbm_bbox():
    global _NBM_BBOX
    if _NBM_BBOX is not None:
        return _NBM_BBOX
    ensure_loaded()
    meta = RASTERS["meta"]
    h, w = int(meta["height"]), int(meta["width"])
    left, bottom, right, top = array_bounds(h, w, meta["transform"])
    # Raster is Pseudo-Mercator (EPSG:3857) meters; convert corners to lon/lat.
    try:
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon0, lat0 = tf.transform(left, bottom)
        lon1, lat1 = tf.transform(right, top)
    except Exception:
        # Manual web-mercator inverse fallback.
        def _ll(x, y):
            lon = x / 20037508.342789244 * 180.0
            lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
            return lon, lat
        lon0, lat0 = _ll(left, bottom)
        lon1, lat1 = _ll(right, top)
    pad = 0.06
    _NBM_BBOX = {
        "lat_min": min(lat0, lat1) - pad,
        "lat_max": max(lat0, lat1) + pad,
        "lon_min": min(lon0, lon1) - pad,
        "lon_max": max(lon0, lon1) + pad,
    }
    if NBM_DEBUG:
        print("NBM bbox: {}".format(_NBM_BBOX))
    return _NBM_BBOX


# ================================
# Forecast planning + per-day compute
# ================================
def _local_label(dt_utc):
    local = dt_utc.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
    hour = local.strftime("%I").lstrip("0") or "12"
    return {
        "date": local.strftime("%a %b %d"),
        "time": "{}:{} {}".format(hour, local.strftime("%M"), local.strftime("%p")),
        "tz": local.tzname() or LOCAL_TZ_NAME,
        "iso": local.strftime("%Y-%m-%d %H:%M"),
        "ymd": local.strftime("%Y%m%d"),
    }


def plan_forecast_days(run_dt, n_days):
    """Pick the next n_days local dates and, for each, the target NBM forecast
    hours covering the 2-3 PM window (only hours that are in the future vs. the
    run and within the NBM horizon)."""
    now_local = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
    plan = []
    d = 0
    while len(plan) < n_days and d < n_days + 4:
        day = (now_local + timedelta(days=d)).date()
        targets = []
        for hh in WINDOW_HOURS_LOCAL:
            local_dt = datetime(day.year, day.month, day.day, hh, 0, tzinfo=LOCAL_TZ)
            valid_utc = local_dt.astimezone(timezone.utc).replace(tzinfo=None)
            fhr = round((valid_utc - run_dt).total_seconds() / 3600.0)
            if 1 <= fhr <= NBM_MAX_FHR:
                targets.append({"hh": hh, "valid_utc": valid_utc, "fhr": int(fhr)})
        if targets:
            mid_local = datetime(day.year, day.month, day.day, 14, 15, tzinfo=LOCAL_TZ)
            plan.append({
                "date": day,
                "date_key": day.strftime("%Y%m%d"),
                "targets": targets,
                "dt_local_solar": mid_local.replace(tzinfo=None),
            })
        d += 1
    return plan


def fetch_window_ambient(session, run_dt, targets, bbox):
    """Fetch every available NBM hour in a day's window (snapping the forecast
    hour to availability), concatenate the bbox points, and reduce to ambient
    inputs. Returns (ambient_dict, used_valid_dts) or (None, [])."""
    acc = defaultdict(list)
    used = []
    n_total = 0
    tmp_dir = tempfile.mkdtemp(prefix="nbm_wbgt_")
    try:
        for t in targets:
            pts = None
            fhr_used = None
            for fhr in (t["fhr"], t["fhr"] + 1, t["fhr"] - 1, t["fhr"] + 2, t["fhr"] - 2):
                if fhr < 1 or fhr > NBM_MAX_FHR:
                    continue
                pts = fetch_nbm_hour_points(session, run_dt, fhr, bbox, tmp_dir)
                if pts:
                    fhr_used = fhr
                    break
            if pts:
                n_total += pts.pop("_n", 0)
                for k, v in pts.items():
                    acc[k].append(v)
                used.append(run_dt + timedelta(hours=fhr_used))
    finally:
        try:
            for fn in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, fn))
                except OSError:
                    pass
            os.rmdir(tmp_dir)
        except OSError:
            pass
    if not acc:
        return None, []
    merged = {k: np.concatenate(v) for k, v in acc.items()}
    merged["_n_total"] = n_total
    return ambient_from_points(merged), used


def compute_fields(ambient, dt_local_solar):
    """Run the unchanged science pipeline for one day's ambient inputs.
    Returns a dict of named arrays plus the scalar model predictions."""
    ensure_loaded()
    era_tair = ambient["era_tair_C"]
    era_dpt = ambient["era_dpt_C"]
    era_wspd = ambient["era_wspd_ms"]
    era_cloud = ambient["era_cloud_pct"]
    ref_pressure_mb = ambient["ref_pressure_mb"]

    X = np.array([[era_tair, era_dpt, era_wspd, era_cloud]], dtype=float)
    t_max = float(MODELS["temp_max"].predict(X)[0])
    t_min = float(MODELS["temp_min"].predict(X)[0])
    d_max = float(MODELS["dew_max"].predict(X)[0]) if MODELS["dew_max"] is not None else None
    d_min = float(MODELS["dew_min"].predict(X)[0]) if MODELS["dew_min"] is not None else None
    s_max = float(MODELS["sol_max"].predict(X)[0]) if MODELS["sol_max"] is not None else None

    ta_map = scale_map(z_to_01(RASTERS["ta_z"]), t_min, t_max)

    dew_map = None
    if RASTERS["td_z"] is not None and d_min is not None and d_max is not None:
        dew_map = scale_map(z_to_01(RASTERS["td_z"]), d_min, d_max)

    sol_map = None
    if RASTERS["sol_z"] is not None and s_max is not None:
        sol_map = scale_map(z_to_01(RASTERS["sol_z"]), 0.0, s_max)

    u2_map = None
    if RASTERS["z0"] is not None:
        z0_meta = RASTERS["z0_meta"]
        tmpl = RASTERS["meta"]
        needs = (z0_meta["crs"] != tmpl["crs"] or z0_meta["transform"] != tmpl["transform"]
                 or z0_meta["width"] != tmpl["width"] or z0_meta["height"] != tmpl["height"])
        z0 = align_to_template(RASTERS["z0"], z0_meta, tmpl) if needs else RASTERS["z0"]
        u2_map = compute_u2_from_u10_and_z0(era_wspd, z0)

    pressure_map = None
    if RASTERS["elev"] is not None:
        elev_meta = RASTERS["elev_meta"]
        tmpl = RASTERS["meta"]
        needs = (elev_meta["crs"] != tmpl["crs"] or elev_meta["transform"] != tmpl["transform"]
                 or elev_meta["width"] != tmpl["width"] or elev_meta["height"] != tmpl["height"])
        elev = align_to_template(RASTERS["elev"], elev_meta, tmpl) if needs else RASTERS["elev"]
        ref_elev = float(np.nanmean(elev))
        pressure_map = calculate_pressure(elev, ta_map, ref_pressure_mb, ref_elev)

    wbgt_c = tg_c = tnw_c = None
    if dew_map is not None and sol_map is not None and u2_map is not None and pressure_map is not None:
        wbgt_k, tg_k, tnw_k = wbgt_maps_strided(
            ta_map, dew_map, sol_map, pressure_map, u2_map, dt_local_solar,
            lat=SITE_LAT, lon=SITE_LON, stride=WBGT_STRIDE)
        wbgt_c = wbgt_k - 273.15
        tg_c = tg_k - 273.15
        tnw_c = tnw_k - 273.15

    return {
        "ta": ta_map, "dew": dew_map, "sol": sol_map, "u2": u2_map, "pressure": pressure_map,
        "wbgt": wbgt_c, "tg": tg_c, "tnw": tnw_c,
        "pred": {"t_min": t_min, "t_max": t_max, "d_min": d_min, "d_max": d_max, "s_max": s_max},
    }


# ================================
# Disk cache for a built day
# ================================
def _raster_stamp():
    try:
        return str(int(os.path.getmtime(TA_Z_TIF)))
    except OSError:
        return "0"


def _signature():
    return {
        "schema": RENDER_SCHEMA, "roads": ROAD_OVERLAY_VERSION,
        "stride": WBGT_STRIDE, "raster_stamp": _raster_stamp(),
    }


def _manifest_path(day_key):
    return os.path.join(CACHE_DIR, "{}.json".format(day_key))


def _load_cached_day(day_key):
    path = _manifest_path(day_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            man = json.load(f)
    except Exception:
        return None
    if man.get("signature") != _signature():
        return None
    for img in man.get("images", []):
        for fld in ("file", "file_f"):
            fn = img.get(fld)
            if fn and not os.path.exists(os.path.join(OUTPUT_DIR, fn)):
                return None
    return man


def render_day(day, ambient, fields, used_dts, run_id):
    """Save every map in BOTH °C and °F (with roads overlay) and return a
    manifest. The browser unit toggle picks which to show; summary values are
    stored canonically (°C / m/s) and converted client-side."""
    roads = road_segments_for_template(RASTERS["meta"])
    day_key = "{}_{}".format(run_id, day["date_key"])
    images = []

    def _emit_dual(key, base_title, arr_c, cmap):
        """Temperature-like field: render a °C and a °F variant."""
        fc = "{}_c_{}.png".format(key, day_key)
        ff = "{}_f_{}.png".format(key, day_key)
        with _PLOT_LOCK:
            save_map(arr_c, "{} (°C)".format(base_title),
                     os.path.join(OUTPUT_DIR, fc), cmap=cmap, roads=roads)
            save_map(c_to_f(arr_c), "{} (°F)".format(base_title),
                     os.path.join(OUTPUT_DIR, ff), cmap=cmap, roads=roads)
        images.append({"key": key, "title": base_title, "file": fc, "file_f": ff,
                       "url_c": "/outputs/{}".format(fc), "url_f": "/outputs/{}".format(ff)})

    def _emit_single(key, title, arr, cmap):
        """Unit-neutral field (solar W/m², pressure mb): one render, both modes."""
        fn = "{}_{}.png".format(key, day_key)
        with _PLOT_LOCK:
            save_map(arr, title, os.path.join(OUTPUT_DIR, fn), cmap=cmap, roads=roads)
        url = "/outputs/{}".format(fn)
        images.append({"key": key, "title": title, "file": fn, "url_c": url, "url_f": url})

    _emit_dual("temp", "Air Temperature", fields["ta"], "coolwarm")
    if fields["dew"] is not None:
        _emit_dual("dew", "Dew Point", fields["dew"], "BrBG")
    if fields["sol"] is not None:
        _emit_single("sol", "Solar Radiation (W/m²)", fields["sol"], "inferno")
    if fields["u2"] is not None:
        fc = "wind_c_{}.png".format(day_key)
        ff = "wind_f_{}.png".format(day_key)
        with _PLOT_LOCK:
            save_map(fields["u2"], "Wind Speed at 2 m (m/s)",
                     os.path.join(OUTPUT_DIR, fc), cmap="viridis", roads=roads)
            save_map(ms_to_mph(fields["u2"]), "Wind Speed at 2 m (mph)",
                     os.path.join(OUTPUT_DIR, ff), cmap="viridis", roads=roads)
        images.append({"key": "wind", "title": "Wind Speed at 2 m", "file": fc, "file_f": ff,
                       "url_c": "/outputs/{}".format(fc), "url_f": "/outputs/{}".format(ff)})
    if fields["pressure"] is not None:
        _emit_single("pressure", "Surface Pressure (mb)", fields["pressure"], "viridis")

    flag_summary = None
    if fields["wbgt"] is not None:
        _emit_dual("wbgt", "WBGT", fields["wbgt"], "bwr")
        _emit_dual("tg", "Black Globe Temperature", fields["tg"], "Reds")
        _emit_dual("tnw", "Natural Wet Bulb Temperature", fields["tnw"], "BrBG")
        fc = "flag_c_{}.png".format(day_key)
        ff = "flag_f_{}.png".format(day_key)
        with _PLOT_LOCK:
            save_flag_map(fields["wbgt"], os.path.join(OUTPUT_DIR, fc), unit="C", roads=roads)
            save_flag_map(c_to_f(fields["wbgt"]), os.path.join(OUTPUT_DIR, ff), unit="F", roads=roads)
        images.append({"key": "flag", "title": "WBGT Flag Levels", "file": fc, "file_f": ff,
                       "url_c": "/outputs/{}".format(fc), "url_f": "/outputs/{}".format(ff)})
        flag_summary = _flag_distribution(fields["wbgt"])

    pred = fields["pred"]

    def _rc(v):
        return round(float(v), 1) if (v is not None and np.isfinite(v)) else None

    run_dt = _state_run_dt()
    valid_tz = _local_label(used_dts[0])["tz"] if used_dts else LOCAL_TZ_NAME
    lead_hr = None
    if used_dts and run_dt is not None:
        lead_hr = int(round((min(used_dts) - run_dt).total_seconds() / 3600.0))

    summary = {
        "date_label": _local_label(used_dts[0])["date"] if used_dts
        else day["date"].strftime("%a %b %d"),
        "valid_label": "2-3 PM " + valid_tz,
        "lead_label": ("f{:03d}".format(lead_hr) if lead_hr is not None else "—"),
        # canonical units: temperatures °C, wind m/s, pressure mb, solar W/m²
        "ambient": {
            "tair_c": _rc(ambient["era_tair_C"]),
            "dpt_c": _rc(ambient["era_dpt_C"]),
            "wspd_ms": _rc(ambient["era_wspd_ms"]),
            "cloud": round(ambient["era_cloud_pct"], 0),
            "pres_mb": _rc(ambient["ref_pressure_mb"]),
            "n_points": ambient.get("n_points", 0),
        },
        "pred": {
            "temp_min_c": _rc(pred["t_min"]), "temp_max_c": _rc(pred["t_max"]),
            "dew_min_c": _rc(pred["d_min"]), "dew_max_c": _rc(pred["d_max"]),
            "sol_max": (round(pred["s_max"], 0) if pred["s_max"] is not None else None),
        },
        "flag": flag_summary,
    }

    manifest = {
        "schema": RENDER_SCHEMA, "signature": _signature(), "run_id": run_id,
        "date_key": day["date_key"], "images": images, "summary": summary,
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(_manifest_path(day_key), "w") as f:
        json.dump(manifest, f)
    return manifest


def _flag_distribution(wbgt_c):
    """Percent of valid pixels in each NCHSAA flag category (always computed in
    °C thresholds; the categories are unit-independent)."""
    a = np.asarray(wbgt_c, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    bounds = [-1e9, 26.7, 29.4, 31.1, 32.2, 1e9]
    names = ["white", "green", "yellow", "red", "black"]
    out = {}
    for i, nm in enumerate(names):
        out[nm] = round(100.0 * float(np.mean((a >= bounds[i]) & (a < bounds[i + 1]))), 1)
    return out


# ================================
# Background build worker + shared state
# ================================
def _state_run_dt():
    with _STATE_LOCK:
        return FORECAST_STATE.get("_run_dt")


def _state_set(**kw):
    with _STATE_LOCK:
        FORECAST_STATE.update(kw)


def _state_update_day(date_key, **kw):
    with _STATE_LOCK:
        for day in FORECAST_STATE["days"]:
            if day["date_key"] == date_key:
                day.update(kw)
                return


def _public_state():
    """Snapshot of state safe to serialize to the browser."""
    with _STATE_LOCK:
        days = []
        for d in FORECAST_STATE["days"]:
            days.append({k: v for k, v in d.items() if not k.startswith("_")})
        return {
            "schema": FORECAST_STATE["schema"],
            "status": FORECAST_STATE["status"],
            "message": FORECAST_STATE["message"],
            "run_label": FORECAST_STATE["run_label"],
            "window_label": FORECAST_STATE["window_label"],
            "default_unit": FORECAST_STATE["default_unit"],
            "generated_at": FORECAST_STATE["generated_at"],
            "building": FORECAST_STATE["building"],
            "days": days,
        }


def _build_worker(run_dt, run_id, plan, bbox):
    session = requests.Session()
    session.headers.update(HEADERS)
    _state_set(status="building", building=True, message="Fetching NBM and computing WBGT...")
    for day in plan:
        date_key = day["date_key"]
        cached = _load_cached_day("{}_{}".format(run_id, date_key))
        if cached:
            _state_update_day(date_key, status="done",
                              images=cached["images"], summary=cached["summary"])
            continue
        _state_update_day(date_key, status="computing")
        try:
            ambient, used = fetch_window_ambient(session, run_dt, day["targets"], bbox)
            if ambient is None or not np.isfinite(ambient["era_tair_C"]) \
                    or not np.isfinite(ambient["era_dpt_C"]):
                _state_update_day(date_key, status="error",
                                  error="NBM data not available for this day's 2-3 PM window.")
                continue
            fields = compute_fields(ambient, day["dt_local_solar"])
            manifest = render_day(day, ambient, fields, used, run_id)
            cleanup_old_outputs()
            _state_update_day(date_key, status="done",
                              images=manifest["images"], summary=manifest["summary"])
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _state_update_day(date_key, status="error", error=str(exc))
    with _STATE_LOCK:
        any_done = any(d["status"] == "done" for d in FORECAST_STATE["days"])
        FORECAST_STATE["building"] = False
        FORECAST_STATE["status"] = "ready" if any_done else "error"
        FORECAST_STATE["generated_at"] = datetime.utcnow().isoformat() + "Z"
        if not any_done and not FORECAST_STATE["message"]:
            FORECAST_STATE["message"] = "No forecast days could be built."
    if NBM_DEBUG:
        print("Forecast build finished for run {}".format(run_id))


def detect_run(session):
    """Latest NBM run, cached for RUN_DETECT_TTL_SEC to avoid hammering S3."""
    now = time.time()
    if _RUN_DETECT["run_dt"] is not None and (now - _RUN_DETECT["ts"]) < RUN_DETECT_TTL_SEC:
        return _RUN_DETECT["run_dt"]
    run_dt = find_latest_nbm_run(session)
    _RUN_DETECT["run_dt"] = run_dt
    _RUN_DETECT["ts"] = now
    return run_dt


def ensure_forecast(force=False):
    """Make sure a build exists/started for the latest NBM run. Non-blocking."""
    global _BUILD_THREAD
    ensure_loaded()
    bbox = get_nbm_bbox()
    session = requests.Session()
    session.headers.update(HEADERS)

    _state_set(status="detecting")
    run_dt = detect_run(session)
    if run_dt is None:
        _state_set(status="error", message="Could not locate a recent NBM run. "
                   "Check internet access to the NOAA NBM S3 bucket.", days=[])
        return
    run_id = "{}_{}".format(run_dt.strftime("%Y%m%d%H"), _raster_stamp())

    with _STATE_LOCK:
        same_run = FORECAST_STATE.get("run_id") == run_id
        thread_alive = _BUILD_THREAD is not None and _BUILD_THREAD.is_alive()
        cur_status = FORECAST_STATE["status"]
    if same_run and not force and (thread_alive or cur_status in ("ready", "building")):
        return

    plan = plan_forecast_days(run_dt, N_FORECAST_DAYS)
    days_state = []
    for day in plan:
        days_state.append({
            "date_key": day["date_key"],
            "date_label": day["date"].strftime("%a %b %d"),
            "status": "pending", "error": None, "images": [], "summary": None,
        })
    rl = _local_label(run_dt)
    with _STATE_LOCK:
        FORECAST_STATE["run_id"] = run_id
        FORECAST_STATE["_run_dt"] = run_dt
        FORECAST_STATE["run_label"] = "{} {} {}  ({}Z)".format(
            rl["date"], rl["time"], rl["tz"], run_dt.strftime("%Y-%m-%d %H"))
        FORECAST_STATE["days"] = days_state
        FORECAST_STATE["status"] = "building"
        FORECAST_STATE["message"] = ""
        FORECAST_STATE["default_unit"] = DEFAULT_UNIT

    if not plan:
        _state_set(status="error", message="No forecast days fall within the NBM horizon.")
        return

    _BUILD_THREAD = threading.Thread(target=_build_worker,
                                     args=(run_dt, run_id, plan, bbox), daemon=True)
    _BUILD_THREAD.start()


# ================================
# HTML template
# ================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neighborhood WBGT Forecast · Carrboro &amp; Chapel Hill</title>
<style>
  :root{--bg:#f4f6f8;--card:#fff;--ink:#1d3557;--ink2:#16324f;--muted:#5a6573;--accent:#457b9d;
        --line:#dce5ec;--soft:#eef6ff;--softline:#cfe0ff;}
  *{box-sizing:border-box;}
  body{font-family:-apple-system,Segoe UI,Arial,Helvetica,sans-serif;margin:0;background:var(--bg);color:#222;}
  .wrap{max-width:1180px;margin:0 auto;padding:0 22px;}
  a{color:var(--accent);}
  /* top bar */
  .topbar{position:sticky;top:0;z-index:50;background:linear-gradient(180deg,#fff,#fbfdff);
          border-bottom:1px solid var(--line);box-shadow:0 1px 6px rgba(15,23,42,.05);}
  .topbar-in{display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:12px 0;}
  .brand{font-weight:800;color:var(--ink);font-size:1.05rem;line-height:1.1;}
  .brand small{display:block;font-weight:600;color:var(--muted);font-size:.74rem;margin-top:2px;}
  .nav-tabs{display:flex;gap:6px;margin-left:6px;}
  .nav-tab{border:1px solid var(--line);background:#fff;color:var(--muted);font-weight:700;
           padding:8px 14px;border-radius:999px;cursor:pointer;font-size:.9rem;}
  .nav-tab.active{background:var(--ink);color:#fff;border-color:var(--ink);}
  .spacer{flex:1;}
  .unit-toggle{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;}
  .unit-toggle button{border:0;background:#fff;color:var(--muted);font-weight:800;padding:8px 14px;cursor:pointer;font-size:.9rem;}
  .unit-toggle button.active{background:var(--accent);color:#fff;}
  /* generic */
  .card{background:var(--card);border-radius:12px;padding:22px;margin:22px 0;box-shadow:0 2px 10px rgba(15,23,42,.07);border:1px solid var(--line);}
  h1{margin:0 0 6px;color:var(--ink);font-size:1.5rem;}
  h2{margin:0 0 12px;color:var(--ink);font-size:1.25rem;}
  h3{margin:0 0 10px;color:var(--ink);font-size:1.02rem;}
  h4{margin:18px 0 6px;color:var(--ink2);font-size:1rem;}
  p{line-height:1.65;}
  .sub{color:var(--muted);max-width:80ch;}
  .pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
  .pill{background:var(--soft);border:1px solid var(--softline);color:#1e3a8a;border-radius:999px;padding:5px 12px;font-size:.84rem;}
  .tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;}
  .tab{padding:9px 15px;border:1px solid var(--line);background:#fff;border-radius:8px;cursor:pointer;font-weight:700;color:var(--muted);}
  .tab.active{background:var(--ink);color:#fff;border-color:var(--ink);}
  .tab .st{display:block;font-size:.7rem;font-weight:600;margin-top:2px;opacity:.9;}
  .summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:6px 0 18px;}
  .si{background:#f7f9fb;border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:.92rem;}
  .si b{color:var(--ink);font-weight:700;}
  .images-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;}
  .image-card{background:#fafbfc;border:1px solid var(--line);border-radius:8px;padding:10px;}
  .image-card img{width:100%;height:auto;border-radius:6px;border:1px solid #d0d7de;background:#fff;}
  .banner{padding:12px 16px;border-radius:8px;margin:0 0 16px;}
  .banner.info{background:var(--soft);border-left:5px solid var(--accent);color:#1e3a8a;}
  .banner.warn{background:#fff8e6;border-left:5px solid #d4a017;color:#7a5a00;}
  .banner.err{background:#ffe6e6;border-left:5px solid #c00;color:#8b0000;}
  .spinner{display:inline-block;width:14px;height:14px;border:3px solid var(--softline);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:8px;}
  @keyframes spin{to{transform:rotate(360deg);}}
  .flagbar{display:flex;height:18px;border-radius:4px;overflow:hidden;border:1px solid var(--line);margin-top:6px;}
  .muted{color:var(--muted);font-size:.85rem;}
  /* about */
  .hero-about h1{font-size:1.7rem;}
  .chips{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 4px;}
  .chip{background:var(--soft);border:1px solid var(--softline);border-radius:10px;padding:10px 14px;min-width:150px;}
  .chip b{display:block;color:var(--ink);font-size:1.15rem;}
  .chip span{color:var(--muted);font-size:.82rem;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px;}
  .formula{background:#0d1f33;color:#eaf2fb;border-radius:10px;padding:14px 16px;font-size:1.02rem;text-align:center;letter-spacing:.3px;}
  table.flag{width:100%;border-collapse:collapse;margin-top:10px;font-size:.92rem;}
  table.flag th,table.flag td{border:1px solid var(--line);padding:8px 10px;text-align:left;}
  table.flag th{background:#f3f6fa;color:var(--ink);}
  .sw{display:inline-block;width:13px;height:13px;border-radius:3px;border:1px solid #99a;vertical-align:middle;margin-right:7px;}
  ul.tight{margin:8px 0;padding-left:20px;} ul.tight li{margin:5px 0;line-height:1.55;}
  .steps{counter-reset:s;margin:6px 0;padding:0;list-style:none;}
  .steps li{position:relative;padding:6px 0 12px 40px;line-height:1.55;}
  .steps li:before{counter-increment:s;content:counter(s);position:absolute;left:0;top:4px;width:26px;height:26px;
                   background:var(--accent);color:#fff;border-radius:50%;text-align:center;line-height:26px;font-weight:700;font-size:.85rem;}
  .note{background:var(--soft);border-left:5px solid var(--accent);border-radius:8px;padding:12px 16px;color:#1e3a8a;margin-top:14px;}
  footer{color:var(--muted);text-align:center;padding:18px 0 30px;font-size:.85rem;}
  @media (max-width:820px){ .grid2{grid-template-columns:1fr;} }
</style>
</head>
<body>
<div class="topbar"><div class="wrap topbar-in">
  <div class="brand">Neighborhood WBGT Forecast<small>Carrboro &amp; Chapel Hill, NC · SERCC heat-mapping</small></div>
  <div class="nav-tabs">
    <button class="nav-tab active" data-view="forecast">Forecast</button>
    <button class="nav-tab" data-view="about">About &amp; Methods</button>
  </div>
  <div class="spacer"></div>
  <div class="unit-toggle" id="unit-toggle">
    <button data-u="F">&deg;F</button><button data-u="C">&deg;C</button>
  </div>
</div></div>

<div class="wrap">

<!-- ===================== FORECAST VIEW ===================== -->
<div id="view-forecast">
  <div class="card">
    <h1>Live afternoon heat-stress forecast</h1>
    <p class="sub">Real <b>2&ndash;3&nbsp;PM</b> Wet Bulb Globe Temperature (WBGT) for the next few days at <b>10&nbsp;m</b> resolution, driven automatically by the National&nbsp;Blend&nbsp;of&nbsp;Models (NBM) forecast &mdash; no manual inputs. Heat stress is solved with the Liljegren&nbsp;et&nbsp;al. (2008) energy balance and classified by NCHSAA flag levels. Black lines are roads. See <a href="#" data-goto="about">About &amp; Methods</a> for how the underlying maps were built.</p>
    <div class="pills">
      <span class="pill" id="pill-run">NBM run: &mdash;</span>
      <span class="pill">Window: 2&ndash;3&nbsp;PM (climatological WBGT peak)</span>
      <span class="pill" id="pill-status">Status: &mdash;</span>
    </div>
  </div>

  <div id="banner-wrap"></div>

  <div class="card">
    <div class="tabs" id="tabs"></div>
    <div id="panel"></div>
  </div>
</div>

<!-- ===================== ABOUT VIEW ===================== -->
<div id="view-about" style="display:none">
  <div class="card hero-about">
    <h1>About this tool &amp; how the maps are made</h1>
    <p class="sub">This application turns a broad weather forecast into a block-by-block picture of afternoon heat stress across Carrboro and Chapel Hill. It is built on the Southeast Regional Climate Center&rsquo;s (SERCC) citizen-science heat-mapping campaign and the doctoral research of Andrew Robinson (UNC Chapel Hill, advisor Dr.&nbsp;Charles Konrad).</p>
    <div class="chips">
      <div class="chip"><b>10 m</b><span>map resolution</span></div>
      <div class="chip"><b>2:15 PM</b><span>climatological WBGT peak</span></div>
      <div class="chip"><b>2024&ndash;2025</b><span>heat-season campaign</span></div>
      <div class="chip"><b>Walk · Bike · Drive</b><span>mobile transects</span></div>
      <div class="chip"><b>NBM</b><span>live forecast input</span></div>
    </div>
  </div>

  <div class="card">
    <div class="grid2">
      <div>
        <h2>What is WBGT, and why use it?</h2>
        <p>Extreme heat is the leading cause of weather-related death in the United States. How dangerous heat feels is governed not by air temperature alone but by the combined effect of <b>temperature, humidity, wind, and solar radiation</b> on the body during activity. Wet Bulb Globe Temperature (WBGT) integrates all four into a single index:</p>
        <div class="formula">WBGT = 0.7&nbsp;&times;&nbsp;NWB + 0.2&nbsp;&times;&nbsp;T<sub>g</sub> + 0.1&nbsp;&times;&nbsp;T<sub>a</sub></div>
        <p class="muted" style="margin-top:8px">NWB = natural wet-bulb temperature, T<sub>g</sub> = black-globe temperature, T<sub>a</sub> = air temperature.</p>
        <p>Because it captures shade, sun, and ventilation &mdash; not just heat and humidity &mdash; WBGT outperforms the heat index as a predictor of heat illness and is the standard used by the U.S. military, OSHA, and state high-school athletic associations. Operational WBGT forecasts, however, are produced at a coarse <b>2.5&nbsp;km</b> grid, while real human exposure varies over <b>tens of meters</b> &mdash; the scale of a sidewalk, a school field, or a shaded greenway. This tool resolves that fine-scale variation.</p>
      </div>
      <div>
        <h2>WBGT flag levels (NCHSAA)</h2>
        <p>Each pixel is classified into the North Carolina High School Athletic Association heat-risk categories used to guide hydration, work&ndash;rest cycles, and activity limits.</p>
        <table class="flag">
          <tr><th>Flag</th><th>&deg;F</th><th>&deg;C</th><th>Typical guidance</th></tr>
          <tr><td><span class="sw" style="background:#fff"></span>White</td><td>&lt; 80</td><td>&lt; 26.7</td><td>Normal activity</td></tr>
          <tr><td><span class="sw" style="background:#2e7d32"></span>Green</td><td>80&ndash;85</td><td>26.7&ndash;29.4</td><td>Use discretion</td></tr>
          <tr><td><span class="sw" style="background:#f9d100"></span>Yellow</td><td>85&ndash;88</td><td>29.4&ndash;31.1</td><td>Added rest, monitor athletes</td></tr>
          <tr><td><span class="sw" style="background:#d32f2f"></span>Red</td><td>88&ndash;90</td><td>31.1&ndash;32.2</td><td>Heightened risk; limit activity</td></tr>
          <tr><td><span class="sw" style="background:#000"></span>Black</td><td>&gt; 90</td><td>&gt; 32.2</td><td>Extreme; cancel or postpone</td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>The citizen-science heat-mapping campaign</h2>
    <p>The maps behind this tool are grounded in one of the most extensive mobile-transect urban-heat campaigns ever conducted. Across the <b>2024 and 2025 heat seasons (May&ndash;September)</b>, citizen-science volunteers and SERCC staff measured heat-stress conditions on <b>44 meteorologically diverse days</b> across two North Carolina Piedmont study areas &mdash; Carrboro/Chapel Hill (the domain of this tool) and South Durham. Unlike earlier campaigns that sampled air temperature on one or two unusually hot days, this effort captured the full suite of WBGT variables across a whole range of summer weather, from calm and sunny to breezy and cloudy.</p>
    <div class="grid2">
      <div>
        <h4>Multi-modal mobile transects</h4>
        <p>Observers traversed the rural-to-urban gradient three ways &mdash; <b>driving, cycling, and walking</b> &mdash; each timed to begin at <b>2:15&nbsp;PM</b>, the climatological peak of WBGT in the Piedmont. Vehicles covered major and minor road networks; bicycles and walkers reached greenways, forested trails, parks, and pedestrian corridors that vehicles cannot. Routes were designed to cross as many land-surface types as possible and to prioritize places where people are actually outdoors. (The study protocol was reviewed and deemed exempt by the UNC Institutional Review Board.)</p>
      </div>
      <div>
        <h4>Instrumentation</h4>
        <ul class="tight">
          <li><b>Air temperature &amp; humidity:</b> factory-calibrated HOBO U23 Pro&nbsp;v2 loggers in ventilated radiation shields (pre-ventilated 5&nbsp;min; agreement &lt;0.1&nbsp;&deg;C).</li>
          <li><b>Solar radiation:</b> Apogee silicon-cell pyranometer mounted on the vehicle &mdash; one of the first campaigns to map solar load for heat stress across a whole season.</li>
          <li><b>Position &amp; speed:</b> Wahoo Element GPS; stationary points were dropped to avoid heat buildup in the shield.</li>
          <li><b>Wind &amp; pressure:</b> the NCEconet research station at the former Horace Williams Airport (with a black-globe thermometer) anchors wind and pressure for the WBGT calculation.</li>
        </ul>
      </div>
    </div>
    <p class="muted" style="margin-top:12px">Campaign details and weekly field reports: <a href="https://sercc.com/wbgt-heat-mapping/" target="_blank" rel="noopener">SERCC WBGT Heat Mapping Project</a>.</p>
  </div>

  <div class="card">
    <h2>How the observations become a map</h2>
    <p>The thousands of geolocated transect points are translated into a continuous 10&nbsp;m surface using land-surface predictors and machine learning &mdash; the methodology of Robinson&rsquo;s dissertation, <i>Quantifying the Microscale Details of Wet Bulb Globe Temperature Across Urban Heat Islands</i>.</p>
    <ol class="steps">
      <li><b>Land-surface predictors.</b> Physically meaningful rasters &mdash; tree canopy, impervious surface, building volume, albedo, sky-view factor, elevation, slope, aspect, and distance to road (from NLCD, USGS, and Google Earth Engine) &mdash; plus <b>focal-distance</b> versions averaged over 30&ndash;990&nbsp;m to capture each location&rsquo;s neighborhood context.</li>
      <li><b>Machine learning.</b> Random-forest regressions (scikit-learn, 100 trees) learn how those surfaces shape air temperature, dew point, and solar radiation, trained on 90% of transect points and validated on the rest (R&sup2; &asymp; 0.85&ndash;0.99 for temperature and dew point).</li>
      <li><b>Wind &amp; pressure.</b> 10&nbsp;m wind from the NCEconet station is downscaled to 2&nbsp;m with a surface-roughness logarithmic law (forests slow the wind, open ground does not); pressure is estimated from elevation via the hypsometric equation.</li>
      <li><b>WBGT energy balance.</b> Air temperature, humidity, solar, wind, and pressure feed the Liljegren&nbsp;et&nbsp;al. (2008) outdoor WBGT model, solved iteratively at every pixel for natural wet-bulb and black-globe temperature, then combined into WBGT and NCHSAA flag levels.</li>
    </ol>
  </div>

  <div class="card">
    <h2>From campaign to live forecast</h2>
    <p>The campaign established the <b>persistent spatial fingerprint</b> of heat stress &mdash; which blocks run hot and which stay cool, and how that pattern shifts with the weather. This tool puts that fingerprint to work on tomorrow&rsquo;s weather:</p>
    <ul class="tight">
      <li>For each of the next few days, the app pulls the <b>NBM forecast</b> over the study area for the 2&ndash;3&nbsp;PM window and reduces it to the broad ambient conditions (air temperature, dew point, wind, cloud).</li>
      <li>Models translate those broad conditions into the day&rsquo;s expected neighborhood range, and the observed spatial structure is scaled between them to produce 10&nbsp;m air-temperature, dew-point, and solar fields.</li>
      <li>The Liljegren energy balance then yields the WBGT, black-globe, and natural-wet-bulb maps, classified into flag levels &mdash; the same science the field campaign was built to support, now run forward on a live forecast.</li>
    </ul>
    <div class="note"><b>Best use &amp; limits.</b> The tool is most reliable when forecast conditions resemble the heat-season days it was trained on, and it is designed for the early-afternoon (around 2:15&nbsp;PM) heat peak. It is a research and situational-awareness tool &mdash; not a substitute for official National Weather Service heat products or <a href="https://www.heat.gov/" target="_blank" rel="noopener">heat.gov</a> guidance.</div>
  </div>

  <div class="card">
    <h2>Sources &amp; credits</h2>
    <ul class="tight">
      <li>SERCC WBGT Heat Mapping Project &mdash; <a href="https://sercc.com/wbgt-heat-mapping/" target="_blank" rel="noopener">sercc.com/wbgt-heat-mapping</a></li>
      <li>A. Robinson (2026), <i>Quantifying the Microscale Details of Wet Bulb Globe Temperature Across Urban Heat Islands</i>, doctoral dissertation, UNC Chapel Hill.</li>
      <li>Liljegren et&nbsp;al. (2008), <i>Modeling the Wet Bulb Globe Temperature Using Standard Meteorological Measurements</i>.</li>
      <li>Forecast input: NOAA National Blend of Models (NBM). Land cover: NLCD / USGS / Google Earth Engine.</li>
      <li>Developed by Andrew Robinson, PhD, with NOAA&rsquo;s Southeast Regional Climate Center and the UNC Department of Geography &amp; Environment.</li>
    </ul>
  </div>
</div>

<footer>Liljegren WBGT &middot; NBM live forecast &middot; SERCC mobile-transect campaign &middot; 10&nbsp;m grid with roads overlay</footer>
</div>

<script>
let STATE = {{ initial_state|tojson }};
let active = null;
let U = 'F';

const el=(tag,attrs,html)=>{ const e=document.createElement(tag); if(attrs) for(const k in attrs) e.setAttribute(k,attrs[k]); if(html!=null) e.innerHTML=html; return e; };
const toF=c=>c*9/5+32;
const shownT=c=>U==='F'?toF(c):c;
function tcell(c){ return c==null?'&mdash;':shownT(c).toFixed(1)+' &deg;'+U; }
function trange(a,b){ if(a==null&&b==null)return '&mdash;'; const lo=a==null?'&mdash;':shownT(a).toFixed(1); const hi=b==null?'&mdash;':shownT(b).toFixed(1); return lo+' to '+hi+' &deg;'+U; }
function wcell(ms){ if(ms==null)return '&mdash;'; return U==='F'?(ms*2.2369363).toFixed(1)+' mph':ms.toFixed(1)+' m/s'; }
const imgUrl=img=>(U==='F'?(img.url_f||img.url_c):(img.url_c||img.url_f));

function flagBar(flag){
  if(!flag) return '';
  const cols={white:'#ffffff',green:'#2e7d32',yellow:'#f9d100',red:'#d32f2f',black:'#000000'};
  let segs='';
  for(const k of ['white','green','yellow','red','black']){
    const pct=flag[k]||0; if(pct>0) segs+=`<div title="${k}: ${pct}%" style="width:${pct}%;background:${cols[k]}"></div>`;
  }
  const lab = U==='F' ? 'white &lt;80, green 80&ndash;85, yellow 85&ndash;88, red 88&ndash;90, black &gt;90 &deg;F'
                      : 'white &lt;26.7, green 26.7&ndash;29.4, yellow 29.4&ndash;31.1, red 31.1&ndash;32.2, black &gt;32.2 &deg;C';
  return `<div class="flagbar">${segs}</div><div class="muted">NCHSAA flag coverage &mdash; ${lab}</div>`;
}

function renderPanel(){
  const panel=document.getElementById('panel'); if(!panel) return; panel.innerHTML='';
  if(!STATE.days||!STATE.days.length){ panel.appendChild(el('p',{class:'sub'},'No forecast days available yet.')); return; }
  const day=STATE.days.find(d=>d.date_key===active) || STATE.days[0];
  if(day.status!=='done'){
    let msg = day.status==='error' ? ('<b>Could not build this day.</b> '+(day.error||''))
            : (day.status==='computing' ? '<span class="spinner"></span>Computing this day&rsquo;s WBGT maps &mdash; they appear here automatically (a few minutes on a small free instance; faster on a paid one).'
            : 'Queued &mdash; waiting for an earlier day to finish.');
    panel.appendChild(el('div',{class:'banner '+(day.status==='error'?'err':'info')}, msg));
    return;
  }
  const s=day.summary||{}, a=s.ambient||{}, p=s.pred||{};
  const grid=el('div',{class:'summary-grid'});
  const cells=[
    ['Valid', (s.date_label||'')+' &middot; '+(s.valid_label||'')+'  (lead '+(s.lead_label||'&mdash;')+')'],
    ['NBM ambient air temp', tcell(a.tair_c)],
    ['NBM ambient dew point', tcell(a.dpt_c)],
    ['NBM wind &middot; cloud', wcell(a.wspd_ms)+' &middot; '+(a.cloud!=null?a.cloud+'%':'&mdash;')],
    ['Reference pressure', a.pres_mb!=null?(a.pres_mb+' mb'):'&mdash;'],
    ['Predicted air-temp range', trange(p.temp_min_c,p.temp_max_c)],
    ['Predicted dew-point range', trange(p.dew_min_c,p.dew_max_c)],
    ['Predicted peak solar', p.sol_max!=null?(p.sol_max+' W/m&sup2;'):'&mdash;'],
  ];
  for(const [k,v] of cells){ grid.appendChild(el('div',{class:'si'}, '<b>'+k+'</b><br>'+v)); }
  panel.appendChild(grid);
  if(s.flag){ const fb=el('div'); fb.innerHTML=flagBar(s.flag); panel.appendChild(fb); }

  const ig=el('div',{class:'images-grid'});
  for(const img of (day.images||[])){
    const c=el('div',{class:'image-card'});
    c.appendChild(el('h3',null,img.title));
    c.appendChild(el('img',{src:imgUrl(img),alt:img.title,loading:'lazy'}));
    ig.appendChild(c);
  }
  panel.appendChild(ig);
}

function renderTabs(){
  const tabs=document.getElementById('tabs'); if(!tabs) return; tabs.innerHTML='';
  (STATE.days||[]).forEach(d=>{
    const stTxt = d.status==='done'?'ready':(d.status==='computing'?'computing…':(d.status==='error'?'error':'queued'));
    const t=el('div',{class:'tab'+(d.date_key===active?' active':'')}, d.date_label+'<span class="st">'+stTxt+'</span>');
    t.onclick=()=>{active=d.date_key; renderTabs(); renderPanel();};
    tabs.appendChild(t);
  });
}

function renderBanners(){
  const w=document.getElementById('banner-wrap'); if(!w) return; w.innerHTML='';
  if(STATE.status==='error' && (!STATE.days||STATE.days.every(d=>d.status!=='done'))){
    w.appendChild(el('div',{class:'banner err'}, '<b>Forecast unavailable.</b> '+(STATE.message||'')));
  } else if(STATE.building){
    w.appendChild(el('div',{class:'banner warn'}, '<span class="spinner"></span>Building the forecast. Days appear as they finish &mdash; this page updates itself.'));
  }
}

function renderHeader(){
  const r=document.getElementById('pill-run'); if(r) r.innerHTML='NBM run: '+(STATE.run_label||'&mdash;');
  const st=document.getElementById('pill-status'); if(st) st.textContent='Status: '+(STATE.status||'—');
}

function renderAll(){
  if(active===null && STATE.days && STATE.days.length){
    const done=STATE.days.find(d=>d.status==='done'); active=(done||STATE.days[0]).date_key;
  }
  renderHeader(); renderBanners(); renderTabs(); renderPanel();
}

function setUnit(u){
  U=u; try{localStorage.setItem('wbgt_unit',u);}catch(e){}
  document.querySelectorAll('#unit-toggle button').forEach(b=>b.classList.toggle('active',b.dataset.u===u));
  renderPanel();
}
function setView(v){
  document.querySelectorAll('.nav-tab').forEach(b=>b.classList.toggle('active',b.dataset.view===v));
  document.getElementById('view-forecast').style.display=(v==='forecast')?'':'none';
  document.getElementById('view-about').style.display=(v==='about')?'':'none';
  window.scrollTo(0,0);
}

async function poll(){
  try{
    const r=await fetch('/api/status',{cache:'no-store'}); if(r.ok){ STATE=await r.json(); renderAll(); }
  }catch(e){}
  const busy = STATE.building || (STATE.days||[]).some(d=>d.status==='computing'||d.status==='pending');
  setTimeout(poll, busy?4000:30000);
}

try{ U = localStorage.getItem('wbgt_unit') || STATE.default_unit || 'F'; }catch(e){ U = STATE.default_unit || 'F'; }
document.querySelectorAll('#unit-toggle button').forEach(b=>{ b.classList.toggle('active',b.dataset.u===U); b.onclick=()=>setUnit(b.dataset.u); });
document.querySelectorAll('.nav-tab').forEach(b=> b.onclick=()=>setView(b.dataset.view));
document.querySelectorAll('[data-goto]').forEach(a=> a.onclick=(e)=>{e.preventDefault(); setView(a.dataset.goto);});
renderAll();
poll();
</script>
</body>
</html>
"""


# ================================
# Routes
# ================================
_KICK_LOCK = threading.Lock()
_KICK_THREAD = None


def _kick_forecast():
    """Start the detect+build orchestration in a background thread so the web
    request never blocks on it (detection does network I/O; the build is heavy).
    No-op if an orchestration is already in flight."""
    global _KICK_THREAD
    with _KICK_LOCK:
        if _KICK_THREAD is not None and _KICK_THREAD.is_alive():
            return
        with _STATE_LOCK:
            if FORECAST_STATE["status"] in ("idle", "error"):
                FORECAST_STATE["status"] = "detecting"
                FORECAST_STATE["building"] = True
                FORECAST_STATE["message"] = ""

        def _run():
            try:
                ensure_forecast()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                _state_set(status="error", building=False, message=str(exc))

        _KICK_THREAD = threading.Thread(target=_run, daemon=True)
        _KICK_THREAD.start()


@app.route("/")
def index():
    _kick_forecast()
    return render_template_string(HTML_TEMPLATE, initial_state=_public_state())


@app.route("/api/status")
def api_status():
    return jsonify(_public_state())


@app.route("/api/refresh", methods=["POST", "GET"])
def api_refresh():
    _RUN_DETECT["ts"] = 0.0
    threading.Thread(target=lambda: ensure_forecast(force=True), daemon=True).start()
    return jsonify(ok=True)


@app.route("/outputs/<path:filename>")
def output_file(filename):
    p = os.path.realpath(os.path.join(OUTPUT_DIR, filename))
    if not p.startswith(os.path.realpath(OUTPUT_DIR)):
        abort(403)
    if not os.path.exists(p):
        abort(404)
    return send_file(p)


@app.route("/healthz")
def healthz():
    try:
        ensure_loaded()
        bbox = get_nbm_bbox()
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 503
    # The overlay renders from the prebaked JSON (the shapefile is excluded from
    # the image), so report that, not just the shapefile.
    roads_ok = os.path.exists(ROADS_SEGMENTS_JSON) or os.path.exists(ROADS_SHP)
    return jsonify(ok=True, schema=RENDER_SCHEMA, default_unit=DEFAULT_UNIT, stride=WBGT_STRIDE,
                   forecast_days=N_FORECAST_DAYS, nbm_bbox=bbox,
                   roads=roads_ok, roads_baked=os.path.exists(ROADS_SEGMENTS_JSON),
                   state=_public_state())


# ================================
# CLI entry points
# ================================
def _cli_verify():
    """Confirm the vectorized solver matches the original per-pixel one."""
    ensure_loaded()
    era_tair, era_dpt, era_wspd, era_cloud, ref_p = 32.0, 22.0, 2.5, 35.0, 1008.0
    X = np.array([[era_tair, era_dpt, era_wspd, era_cloud]], dtype=float)
    t_min = float(MODELS["temp_min"].predict(X)[0]); t_max = float(MODELS["temp_max"].predict(X)[0])
    d_min = float(MODELS["dew_min"].predict(X)[0]); d_max = float(MODELS["dew_max"].predict(X)[0])
    s_max = float(MODELS["sol_max"].predict(X)[0])
    ta = scale_map(z_to_01(RASTERS["ta_z"]), t_min, t_max)
    td = scale_map(z_to_01(RASTERS["td_z"]), d_min, d_max)
    sol = scale_map(z_to_01(RASTERS["sol_z"]), 0.0, s_max)
    u2 = compute_u2_from_u10_and_z0(era_wspd, RASTERS["z0"])
    elev = RASTERS["elev"]; pmap = calculate_pressure(elev, ta, ref_p, float(np.nanmean(elev)))
    td = np.minimum(td, ta)
    dt_local = datetime(2026, 7, 15, 14, 15)

    # Compare on a crop (the scalar solver is slow) of meaningful size.
    sl = (slice(300, 480), slice(300, 480))
    a = [x[sl] for x in (ta, td, sol, pmap, u2)]
    t0 = time.time()
    ws_s, tg_s, tn_s = _wbgt_from_fields_scalar(*a, dt_local)
    t_scalar = time.time() - t0
    t0 = time.time()
    ws_v, tg_v, tn_v = wbgt_from_fields(*a, dt_local)
    t_vec = time.time() - t0
    n = int(np.isfinite(ws_s).sum())
    print("Equivalence check on a {}x{} crop ({} valid px):".format(
        a[0].shape[0], a[0].shape[1], n))
    for name, sc, ve in (("WBGT", ws_s, ws_v), ("Tg", tg_s, tg_v), ("Tnw", tn_s, tn_v)):
        d = np.abs(np.asarray(sc, float) - np.asarray(ve, float))
        d = d[np.isfinite(d)]
        print("  {:5s} max|diff| = {:.2e} degC   mean|diff| = {:.2e}".format(
            name, float(d.max()) if d.size else 0.0, float(d.mean()) if d.size else 0.0))
    print("  scalar {:.2f}s vs vectorized {:.3f}s  ->  {:.0f}x faster".format(
        t_scalar, t_vec, (t_scalar / t_vec) if t_vec else float("inf")))
    gh, gw = RASTERS["ta_z"].shape
    print("  (full {}x{} grid is ~{:.0f}x this crop)".format(
        gh, gw, (gh * gw) / (a[0].shape[0] * a[0].shape[1])))


def _cli_build(stride=None, days=None, selftest=False):
    """Run a forecast build synchronously from the terminal (good for testing)."""
    global WBGT_STRIDE, N_FORECAST_DAYS
    if stride:
        WBGT_STRIDE = max(1, int(stride))
    if days:
        N_FORECAST_DAYS = int(days)
    ensure_loaded()
    print("Roads overlay source:", ROADS_SHP, "(exists)" if os.path.exists(ROADS_SHP) else "(MISSING)")
    print("WBGT stride:", WBGT_STRIDE, "| forecast days:", N_FORECAST_DAYS, "| default unit:", DEFAULT_UNIT)

    if selftest:
        # Pipeline check that does NOT need NBM: fixed ambient, today's date.
        print("\n[selftest] Rendering one day from fixed ambient inputs (no NBM)...")
        ambient = {"era_tair_C": 32.0, "era_dpt_C": 22.0, "era_wspd_ms": 2.5,
                   "era_cloud_pct": 35.0, "ref_pressure_mb": 1008.0, "n_points": 0}
        day = {"date": datetime.now().date(), "date_key": datetime.now().strftime("%Y%m%d"),
               "targets": [], "dt_local_solar": datetime.now().replace(hour=14, minute=15,
                                                                        second=0, microsecond=0)}
        t0 = time.time()
        fields = compute_fields(ambient, day["dt_local_solar"])
        man = render_day(day, ambient, fields, [], "selftest")
        print("[selftest] done in {:.1f}s. Maps written to {}:".format(time.time() - t0, OUTPUT_DIR))
        for img in man["images"]:
            print("   ", img["file"], "-", img["title"])
        if man["summary"].get("flag"):
            print("[selftest] flag coverage:", man["summary"]["flag"])
        return

    session = requests.Session()
    session.headers.update(HEADERS)
    run_dt = find_latest_nbm_run(session)
    if run_dt is None:
        print("ERROR: no NBM run found (check internet / NOAA bucket).")
        return
    run_id = "{}_{}".format(run_dt.strftime("%Y%m%d%H"), _raster_stamp())
    bbox = get_nbm_bbox()
    plan = plan_forecast_days(run_dt, N_FORECAST_DAYS)
    print("Run {}  ->  {} forecast day(s)".format(run_id, len(plan)))
    with _STATE_LOCK:
        FORECAST_STATE["run_id"] = run_id
        FORECAST_STATE["_run_dt"] = run_dt
        FORECAST_STATE["run_label"] = run_dt.strftime("%Y-%m-%d %HZ")
        FORECAST_STATE["days"] = [{"date_key": d["date_key"],
                                   "date_label": d["date"].strftime("%a %b %d"),
                                   "status": "pending", "error": None,
                                   "images": [], "summary": None} for d in plan]
    _build_worker(run_dt, run_id, plan, bbox)
    print("\nBuild complete. Start the server (python app.py) to view the forecast page.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carrboro/Chapel Hill NBM WBGT forecast app")
    parser.add_argument("--build", action="store_true", help="Build the forecast from NBM and exit")
    parser.add_argument("--selftest", action="store_true", help="Render one day from fixed ambient inputs (no NBM)")
    parser.add_argument("--bake-roads", action="store_true", help="Precompute the roads overlay JSON and exit")
    parser.add_argument("--verify", action="store_true", help="Check the vectorized WBGT solver matches the per-pixel one and exit")
    parser.add_argument("--stride", type=int, default=None, help="WBGT solve stride (1=full res)")
    parser.add_argument("--days", type=int, default=None, help="Number of forecast days")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    args = parser.parse_args()

    if getattr(args, "bake_roads", False):
        bake_roads()
        sys.exit(0)

    if args.verify:
        _cli_verify()
        sys.exit(0)

    if args.build or args.selftest:
        _cli_build(stride=args.stride, days=args.days, selftest=args.selftest)
        sys.exit(0)

    if args.stride:
        WBGT_STRIDE = max(1, int(args.stride))
    if args.days:
        N_FORECAST_DAYS = int(args.days)

    print("WBGT NBM forecast app  |  default_unit={}  stride={}  days={}".format(
        DEFAULT_UNIT, WBGT_STRIDE, N_FORECAST_DAYS))
    # Kick off the first build in the background so the page is live immediately.
    threading.Thread(target=lambda: ensure_forecast(force=False), daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
