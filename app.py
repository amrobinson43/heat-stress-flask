import os
import json
import math
import pickle
from datetime import datetime

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except Exception:
    rasterio = None

# ================================
# Page config
# ================================
st.set_page_config(page_title="Afternoon Heat Stress Prediction Tool", layout="wide")

st.title("Afternoon Heat Stress Prediction Tool")
st.write(
    "Predict afternoon temperature, dew point, solar radiation, 2 m wind speed, pressure, "
    "WBGT, black globe temperature, and natural wet bulb temperature from at-observation inputs."
)

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
# Relative paths for deployment
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RASTER_DIR = os.path.join(DATA_DIR, "rasters")

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
# Helpers
# ================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
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

@st.cache_data
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

def show_map(arr, title, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)

def show_flag_map(wbgt_array, unit="C"):
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
    st.pyplot(fig)

# ================================
# Sidebar inputs
# ================================
st.sidebar.header("Inputs")

era_tair = st.sidebar.number_input("Air temperature at observation (°C)", value=32.0)
era_dpt = st.sidebar.number_input("Dew point at observation (°C)", value=22.0)
era_wspd = st.sidebar.number_input("Wind speed at 10 m (m/s)", value=2.5)
era_cloud = st.sidebar.number_input("Cloud cover (%)", value=35.0, min_value=0.0, max_value=100.0)
ref_pressure_mb = st.sidebar.number_input("Reference mean surface pressure (mb)", value=1008.0)

date_val = st.sidebar.date_input("Local date", value=datetime.now().date())
time_val = st.sidebar.time_input("Local time", value=datetime.now().time())

run_button = st.sidebar.button("Run prediction")

# ================================
# Main run
# ================================
if run_button:
    if rasterio is None:
        st.error("rasterio is required for this app and is not installed.")
        st.stop()

    try:
        models = load_models()
        rasters = load_rasters()
    except Exception as e:
        st.error(f"Failed to load models or rasters: {e}")
        st.stop()

    dt_local = datetime.combine(date_val, time_val)
    X = np.array([[era_tair, era_dpt, era_wspd, era_cloud]], dtype=float)

    try:
        t_max = float(models["temp_max"].predict(X)[0])
        t_min = float(models["temp_min"].predict(X)[0])

        d_max = float(models["dew_max"].predict(X)[0]) if models["dew_max"] is not None else None
        d_min = float(models["dew_min"].predict(X)[0]) if models["dew_min"] is not None else None
        s_max = float(models["sol_max"].predict(X)[0]) if models["sol_max"] is not None else None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    ta_w = z_to_01(rasters["ta_z"])
    temp_map = scale_map(ta_w, t_min, t_max)

    dew_map = None
    if rasters["td_z"] is not None and d_min is not None and d_max is not None:
        td_w = z_to_01(rasters["td_z"])
        dew_map = scale_map(td_w, d_min, d_max)

    sol_map = None
    if rasters["sol_z"] is not None and s_max is not None:
        sol_w = z_to_01(rasters["sol_z"])
        sol_map = scale_map(sol_w, 0.0, s_max)

    u2_map = None
    if rasters["z0"] is not None:
        z0_arr = rasters["z0"]
        z0_meta = rasters["z0_meta"]
        tmpl_meta = rasters["meta"]
        needs_align = (
            (z0_meta["crs"] != tmpl_meta["crs"]) or
            (z0_meta["transform"] != tmpl_meta["transform"]) or
            (z0_meta["width"] != tmpl_meta["width"]) or
            (z0_meta["height"] != tmpl_meta["height"])
        )
        z0_aligned = align_to_template(z0_arr, z0_meta, tmpl_meta) if needs_align else z0_arr
        u2_map = compute_u2_from_u10_and_z0(era_wspd, z0_aligned)

    pressure_map = None
    if rasters["elev"] is not None:
        elev_arr = rasters["elev"]
        elev_meta = rasters["elev_meta"]
        tmpl_meta = rasters["meta"]
        needs_align = (
            (elev_meta["crs"] != tmpl_meta["crs"]) or
            (elev_meta["transform"] != tmpl_meta["transform"]) or
            (elev_meta["width"] != tmpl_meta["width"]) or
            (elev_meta["height"] != tmpl_meta["height"])
        )
        elev_aligned = align_to_template(elev_arr, elev_meta, tmpl_meta) if needs_align else elev_arr
        ref_elevation_m = float(np.nanmean(elev_aligned))
        pressure_map = calculate_pressure(elev_aligned, temp_map, ref_pressure_mb, ref_elevation_m)

    st.subheader("Synoptic predictions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature min (°C)", f"{t_min:.2f}")
    c1.metric("Temperature max (°C)", f"{t_max:.2f}")
    c2.metric("Dew point min (°C)", "NA" if d_min is None else f"{d_min:.2f}")
    c2.metric("Dew point max (°C)", "NA" if d_max is None else f"{d_max:.2f}")
    c3.metric("Solar max (W/m²)", "NA" if s_max is None else f"{s_max:.1f}")

    st.subheader("Continuous fields")
    col1, col2 = st.columns(2)
    with col1:
        show_map(temp_map, "Temperature (°C)", cmap="coolwarm")
        if dew_map is not None:
            show_map(dew_map, "Dew Point (°C)", cmap="BrBG")
        if u2_map is not None:
            show_map(u2_map, "Wind Speed at 2 m (m/s)", cmap="viridis")
    with col2:
        if sol_map is not None:
            show_map(sol_map, "Solar Radiation (W/m²)", cmap="inferno")
        if pressure_map is not None:
            show_map(pressure_map, "Surface Pressure (mb)", cmap="viridis")

    if dew_map is not None and sol_map is not None and u2_map is not None and pressure_map is not None:
        st.subheader("WBGT fields")

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

        col3, col4 = st.columns(2)
        with col3:
            show_map(wbgt_c, "WBGT (°C)", cmap="bwr")
            show_map(tg_c, "Black Globe Temperature (°C)", cmap="Reds")
        with col4:
            show_map(tnw_c, "Natural Wet Bulb Temperature (°C)", cmap="BrBG")
            show_flag_map(wbgt_c, unit="C")

        st.subheader("WBGT flag map (°F)")
        show_flag_map(wbgt_f, unit="F")

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

        st.subheader("Summary JSON")
        st.json(summary)
        st.download_button(
            "Download summary.json",
            data=json.dumps(summary, indent=2),
            file_name="prediction_summary.json",
            mime="application/json",
        )
    else:

        st.warning("WBGT step skipped because one or more required rasters or models are missing.")

