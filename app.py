import os
import math
import pickle
import uuid
from datetime import datetime

import numpy as np
from flask import Flask, request, send_file, url_for, render_template_string
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

LOCKED_TIME_STR = "14:15"
LOCKED_TIME_LABEL = "2:15 PM Local Time"

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
# Integrated HTML template
# ================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Afternoon Heat Stress Prediction Tool</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            margin: 0;
            padding: 0;
            background: #f4f6f8;
            color: #222;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }

        .card {
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }

        h1, h2, h3 {
            margin-top: 0;
            color: #1d3557;
        }

        p {
            line-height: 1.65;
            margin-bottom: 14px;
        }

        .subtle {
            color: #556;
        }

        .note {
            background: #eef6ff;
            border-left: 5px solid #457b9d;
            padding: 14px 16px;
            border-radius: 6px;
            margin-top: 16px;
        }

        .error {
            background: #ffe6e6;
            color: #8b0000;
            border-left: 5px solid #cc0000;
            padding: 14px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        .warning {
            background: #fff8e6;
            color: #7a5a00;
            border-left: 5px solid #d4a017;
            padding: 14px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
            align-items: end;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: bold;
            margin-bottom: 6px;
        }

        input[type="text"],
        input[type="number"],
        input[type="date"],
        select {
            padding: 10px;
            border: 1px solid #c9d2da;
            border-radius: 6px;
            font-size: 14px;
        }

        button {
            padding: 12px 18px;
            background: #1d3557;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
            font-weight: bold;
        }

        button:hover {
            background: #16324f;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }

        .summary-item {
            background: #f7f9fb;
            border: 1px solid #dce5ec;
            border-radius: 8px;
            padding: 12px;
        }

        .images-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }

        .image-card {
            background: #fafafa;
            border: 1px solid #dce5ec;
            border-radius: 8px;
            padding: 12px;
        }

        .image-card h3 {
            font-size: 18px;
            margin-bottom: 10px;
        }

        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #d0d7de;
        }

        .top-text p {
            margin-bottom: 12px;
        }

        .locked-time {
            font-weight: bold;
            color: #1d3557;
        }

        .small {
            font-size: 0.95rem;
        }

        .progress-overlay {
            position: fixed;
            inset: 0;
            background: rgba(244, 246, 248, 0.92);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }

        .progress-overlay.active {
            display: flex;
        }

        .progress-box {
            width: min(520px, calc(100vw - 40px));
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            border: 1px solid #dce5ec;
        }

        .progress-box h3 {
            margin-bottom: 10px;
        }

        .progress-box p {
            margin-bottom: 14px;
            color: #556;
        }

        .progress-track {
            width: 100%;
            height: 16px;
            background: #e7edf3;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid #d3dce5;
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #1d3557, #457b9d);
            border-radius: 999px;
            transition: width 0.35s ease;
        }

        .progress-percent {
            margin-top: 10px;
            font-weight: bold;
            color: #1d3557;
            text-align: right;
        }
    </style>
</head>
<body>
    <div id="progressOverlay" class="progress-overlay" aria-hidden="true">
        <div class="progress-box">
            <h3>Generating maps...</h3>
            <p>Please wait while the tool computes the meteorological fields and WBGT outputs.</p>
            <div class="progress-track">
                <div id="progressFill" class="progress-fill"></div>
            </div>
            <div id="progressPercent" class="progress-percent">0%</div>
        </div>
    </div>

    <div class="container">

        <div class="card">
            <h1>Afternoon Wet Bulb Globe Temperature Prediction Tool</h1>

            <div class="top-text">
                <p>
                    This tool predicts the spatial distribution of afternoon Wet Bulb Globe Temperature (WBGT) across Carrboro and Chapel Hill at 2:15PM, the 
                    climatological peak of heat stress in the area. <span class="locked-time">{{ locked_time_label }}</span>. It uses broad-scale weather conditions 
                    (i.e., what a weather app says) to generate spatially continuous maps of 2 m air temperature, dew point temperature, solar radiation, wind speed, 
                    surface pressure, WBGT, black globe temperature, and natural wet-bulb temperature.
                </p>

                <p>
                    The results generated using machine-learning models trained on observations collected during the summer 2025 Southeast Regional Climate Center citizen-science campaign
                    This campaign used walking, cycling, and driving mobile transects to capture fine-scale spatial variability across the urban landscape. For each meteorological
                    variable, the machine-learning models predict the day-specific minimum and maximum conditions expected across the study area from
                    the at-observation ambient inputs entered below. Precomputed raster surfaces derived from the transect data preserve the observed
                    spatial patterning, and those surfaces are then scaled using the model-predicted ranges to create spatially continuous gridded fields.
                    In other words, the models determine the magnitude of the day’s meteorological range, while the raster layers preserve the fine-scale
                    spatial structure identified from the mobile observations.
                </p>

                <p>
                    WBGT is then calculated using the outdoor formulation from Liljegren et al. (2008). The tool computes WBGT as
                    <strong>0.7 × natural wet-bulb temperature + 0.2 × black globe temperature + 0.1 × air temperature</strong>. Rather than using a
                    simplified empirical estimate, the app solves the underlying energy-balance equations iteratively at each grid cell using air
                    temperature, dew point temperature, solar radiation, wind speed, and pressure. This allows the resulting WBGT map to reflect the
                    combined effects of humidity, radiative loading, ventilation, and ambient thermal conditions.
                </p>

                <p>
                    The WBGT flag maps classify each pixel into standard heat-risk categories as defined by the North Carolina High School Athletic Association. In degrees Fahrenheit, the thresholds are
                    <strong>white: below 80°F</strong>,
                    <strong>green: 80–85°F</strong>,
                    <strong>yellow: 85–88°F</strong>,
                    <strong>red: 88–90°F</strong>, and
                    <strong>black: above 90°F</strong>.
                    In degrees Celsius, these correspond to
                    <strong>below 26.7°C</strong>,
                    <strong>26.7–29.4°C</strong>,
                    <strong>29.4–31.1°C</strong>,
                    <strong>31.1–32.2°C</strong>, and
                    <strong>above 32.2°C</strong>, respectively. These categories are intended to help interpret heat stress and provide guidance to prevent heat related illness.
                </p>

                <div class="note small">
                    This tool is locked to <strong>{{ locked_time_label }}</strong> so that all runs represent the same standardized point in the daytime heat-stress cycle.
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Input Conditions</h2>

            {% if error %}
                <div class="error">
                    <strong>Error:</strong> {{ error }}
                </div>
            {% endif %}

            <form method="POST" id="predictionForm">
                <div class="form-group">
                    <label for="unit_system">Input / Output Units</label>
                    <select name="unit_system" id="unit_system" required>
                        <option value="C" {% if form_data['unit_system'] == 'C' %}selected{% endif %}>Celsius / m/s</option>
                        <option value="F" {% if form_data['unit_system'] == 'F' %}selected{% endif %}>Fahrenheit / mph</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="era_tair">
                        Air Temperature at Observation
                        {% if show_input_units %}
                            {% if form_data['unit_system'] == 'F' %}(°F){% else %}(°C){% endif %}
                        {% endif %}
                    </label>
                    <input type="number" step="any" name="era_tair" id="era_tair" value="{{ form_data['era_tair'] }}" required>
                </div>

                <div class="form-group">
                    <label for="era_dpt">
                        Dew Point at Observation
                        {% if show_input_units %}
                            {% if form_data['unit_system'] == 'F' %}(°F){% else %}(°C){% endif %}
                        {% endif %}
                    </label>
                    <input type="number" step="any" name="era_dpt" id="era_dpt" value="{{ form_data['era_dpt'] }}" required>
                </div>

                <div class="form-group">
                    <label for="era_wspd">
                        Wind Speed at Observation
                        {% if show_input_units %}
                            {% if form_data['unit_system'] == 'F' %}(mph){% else %}(m/s){% endif %}
                        {% endif %}
                    </label>
                    <input type="number" step="any" name="era_wspd" id="era_wspd" value="{{ form_data['era_wspd'] }}" required>
                </div>

                <div class="form-group">
                    <label for="era_cloud">Cloud Cover at Observation (%)</label>
                    <input type="number" step="any" name="era_cloud" id="era_cloud" value="{{ form_data['era_cloud'] }}" required>
                </div>

                <div class="form-group">
                    <label for="ref_pressure_mb">Reference Pressure (mb)</label>
                    <input type="number" step="any" name="ref_pressure_mb" id="ref_pressure_mb" value="{{ form_data['ref_pressure_mb'] }}" required>
                </div>

                <div class="form-group">
                    <label for="date_val">Date</label>
                    <input type="date" name="date_val" id="date_val" value="{{ form_data['date_val'] }}" required>
                </div>

                <div class="form-group">
                    <label for="time_display">Analysis Time</label>
                    <input type="text" id="time_display" value="{{ locked_time_label }}" readonly>
                    <input type="hidden" name="time_val" value="{{ locked_time_str }}">
                </div>

                <div class="form-group">
                    <button type="submit" id="runButton">Run Prediction</button>
                </div>
            </form>
        </div>

        {% if result %}
            <div class="card">
                <h2>Prediction Summary</h2>

                {% if result.wbgt_warning %}
                    <div class="warning">
                        <strong>Warning:</strong> {{ result.wbgt_warning }}
                    </div>
                {% endif %}

                <div class="summary-grid">
                    <div class="summary-item"><strong>Run Time (UTC):</strong><br>{{ result.summary.run_at }}</div>
                    <div class="summary-item"><strong>Locked Analysis Time:</strong><br>{{ result.summary.inputs.locked_time_note }}</div>
                    <div class="summary-item"><strong>WBGT Time (Local):</strong><br>{{ result.summary.inputs.wbgt_time_local }}</div>
                    <div class="summary-item"><strong>Unit System:</strong><br>{{ result.summary.inputs.unit_system_label }}</div>
                    <div class="summary-item"><strong>Predicted Temperature Min:</strong><br>{{ "%.2f"|format(result.synoptic_predictions.temp_min_display) }} {{ result.summary.inputs.temp_unit }}</div>
                    <div class="summary-item"><strong>Predicted Temperature Max:</strong><br>{{ "%.2f"|format(result.synoptic_predictions.temp_max_display) }} {{ result.summary.inputs.temp_unit }}</div>

                    {% if result.synoptic_predictions.dew_min_display is not none %}
                    <div class="summary-item"><strong>Predicted Dew Point Min:</strong><br>{{ "%.2f"|format(result.synoptic_predictions.dew_min_display) }} {{ result.summary.inputs.temp_unit }}</div>
                    {% endif %}

                    {% if result.synoptic_predictions.dew_max_display is not none %}
                    <div class="summary-item"><strong>Predicted Dew Point Max:</strong><br>{{ "%.2f"|format(result.synoptic_predictions.dew_max_display) }} {{ result.summary.inputs.temp_unit }}</div>
                    {% endif %}

                    {% if result.synoptic_predictions.sol_max_Wm2 is not none %}
                    <div class="summary-item"><strong>Predicted Solar Radiation Max:</strong><br>{{ "%.2f"|format(result.synoptic_predictions.sol_max_Wm2) }} W/m²</div>
                    {% endif %}
                </div>
            </div>

            <div class="card">
                <h2>Generated Maps</h2>
                <div class="images-grid">
                    {% for key, item in result.image_urls.items() %}
                        <div class="image-card">
                            <h3>{{ item.title }}</h3>
                            <img src="{{ item.url }}" alt="{{ key }}">
                        </div>
                    {% endfor %}
                </div>
            </div>
        {% endif %}

    </div>

    <script>
        (function () {
            const form = document.getElementById("predictionForm");
            const overlay = document.getElementById("progressOverlay");
            const fill = document.getElementById("progressFill");
            const percent = document.getElementById("progressPercent");
            const button = document.getElementById("runButton");

            let timer = null;
            let progress = 0;

            function setProgress(value) {
                progress = Math.max(0, Math.min(95, value));
                fill.style.width = progress + "%";
                percent.textContent = Math.round(progress) + "%";
            }

            function startProgress() {
                overlay.classList.add("active");
                overlay.setAttribute("aria-hidden", "false");
                button.disabled = true;
                button.textContent = "Running...";

                setProgress(6);

                timer = setInterval(function () {
                    if (progress < 35) {
                        setProgress(progress + 6);
                    } else if (progress < 60) {
                        setProgress(progress + 3);
                    } else if (progress < 78) {
                        setProgress(progress + 2);
                    } else if (progress < 90) {
                        setProgress(progress + 1);
                    }
                }, 350);
            }

            if (form) {
                form.addEventListener("submit", function () {
                    startProgress();
                });
            }

            window.addEventListener("pageshow", function () {
                if (timer) {
                    clearInterval(timer);
                }
                setProgress(100);
                setTimeout(function () {
                    overlay.classList.remove("active");
                    overlay.setAttribute("aria-hidden", "true");
                    button.disabled = false;
                    button.textContent = "Run Prediction";
                    setProgress(0);
                }, 150);
            });
        })();
    </script>
</body>
</html>
"""

# ================================
# Unit helpers
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
    show_input_units = request.method == "POST"

    defaults = {
        "unit_system": "C",
        "era_tair": 32.0,
        "era_dpt": 22.0,
        "era_wspd": 2.5,
        "era_cloud": 35.0,
        "ref_pressure_mb": 1008.0,
        "date_val": datetime.now().strftime("%Y-%m-%d"),
        "time_val": LOCKED_TIME_STR,
    }

    form_data = defaults.copy()

    if request.method == "POST":
        form_data["unit_system"] = request.form.get("unit_system", "C").upper()
        if form_data["unit_system"] not in ("C", "F"):
            form_data["unit_system"] = "C"

        form_data["era_tair"] = request.form.get("era_tair", "32.0")
        form_data["era_dpt"] = request.form.get("era_dpt", "22.0")
        form_data["era_wspd"] = request.form.get("era_wspd", "2.5")
        form_data["era_cloud"] = request.form.get("era_cloud", "35.0")
        form_data["ref_pressure_mb"] = request.form.get("ref_pressure_mb", "1008.0")
        form_data["date_val"] = request.form.get("date_val", defaults["date_val"])
        form_data["time_val"] = LOCKED_TIME_STR

        try:
            if rasterio is None:
                raise RuntimeError("rasterio is required for this app and is not installed.")

            ensure_loaded()

            unit_system = form_data["unit_system"]

            era_tair_in = float(form_data["era_tair"])
            era_dpt_in = float(form_data["era_dpt"])
            era_wspd_in = float(form_data["era_wspd"])
            era_cloud = float(form_data["era_cloud"])
            ref_pressure_mb = float(form_data["ref_pressure_mb"])

            # Convert user inputs to internal SI/Celsius units
            if unit_system == "F":
                era_tair = f_to_c(era_tair_in)
                era_dpt = f_to_c(era_dpt_in)
                era_wspd = mph_to_ms(era_wspd_in)
                temp_unit = "°F"
                wind_unit = "mph"
                unit_system_label = "Fahrenheit / mph"
            else:
                era_tair = era_tair_in
                era_dpt = era_dpt_in
                era_wspd = era_wspd_in
                temp_unit = "°C"
                wind_unit = "m/s"
                unit_system_label = "Celsius / m/s"

            dt_local = datetime.strptime(
                f'{form_data["date_val"]} {LOCKED_TIME_STR}',
                "%Y-%m-%d %H:%M"
            )

            X = np.array([[era_tair, era_dpt, era_wspd, era_cloud]], dtype=float)

            t_max_c = float(MODELS["temp_max"].predict(X)[0])
            t_min_c = float(MODELS["temp_min"].predict(X)[0])

            d_max_c = float(MODELS["dew_max"].predict(X)[0]) if MODELS["dew_max"] is not None else None
            d_min_c = float(MODELS["dew_min"].predict(X)[0]) if MODELS["dew_min"] is not None else None
            s_max = float(MODELS["sol_max"].predict(X)[0]) if MODELS["sol_max"] is not None else None

            ta_w = z_to_01(RASTERS["ta_z"])
            temp_map_c = scale_map(ta_w, t_min_c, t_max_c)

            dew_map_c = None
            if RASTERS["td_z"] is not None and d_min_c is not None and d_max_c is not None:
                td_w = z_to_01(RASTERS["td_z"])
                dew_map_c = scale_map(td_w, d_min_c, d_max_c)

            sol_map = None
            if RASTERS["sol_z"] is not None and s_max is not None:
                sol_w = z_to_01(RASTERS["sol_z"])
                sol_map = scale_map(sol_w, 0.0, s_max)

            u2_map_ms = None
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
                u2_map_ms = compute_u2_from_u10_and_z0(era_wspd, z0_aligned)

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
                pressure_map = calculate_pressure(elev_aligned, temp_map_c, ref_pressure_mb, ref_elevation_m)

            image_urls = {}

            # Convert export maps to selected unit system
            if unit_system == "F":
                temp_map_display = c_to_f(temp_map_c)
                dew_map_display = c_to_f(dew_map_c) if dew_map_c is not None else None
                u2_map_display = ms_to_mph(u2_map_ms) if u2_map_ms is not None else None
            else:
                temp_map_display = temp_map_c
                dew_map_display = dew_map_c
                u2_map_display = u2_map_ms

            temp_name = make_output_name("temp")
            save_map(
                temp_map_display,
                f"Temperature ({temp_unit})",
                os.path.join(OUTPUT_DIR, temp_name),
                cmap="coolwarm"
            )
            image_urls["temp_map"] = {
                "url": url_for("output_file", filename=temp_name),
                "title": f"Temperature ({temp_unit})"
            }

            if dew_map_display is not None:
                dew_name = make_output_name("dew")
                save_map(
                    dew_map_display,
                    f"Dew Point ({temp_unit})",
                    os.path.join(OUTPUT_DIR, dew_name),
                    cmap="BrBG"
                )
                image_urls["dew_map"] = {
                    "url": url_for("output_file", filename=dew_name),
                    "title": f"Dew Point ({temp_unit})"
                }

            if sol_map is not None:
                sol_name = make_output_name("solar")
                save_map(
                    sol_map,
                    "Solar Radiation (W/m²)",
                    os.path.join(OUTPUT_DIR, sol_name),
                    cmap="inferno"
                )
                image_urls["sol_map"] = {
                    "url": url_for("output_file", filename=sol_name),
                    "title": "Solar Radiation (W/m²)"
                }

            if u2_map_display is not None:
                wind_name = make_output_name("wind")
                save_map(
                    u2_map_display,
                    f"Wind Speed at 2 m ({wind_unit})",
                    os.path.join(OUTPUT_DIR, wind_name),
                    cmap="viridis"
                )
                image_urls["u2_map"] = {
                    "url": url_for("output_file", filename=wind_name),
                    "title": f"Wind Speed at 2 m ({wind_unit})"
                }

            if pressure_map is not None:
                pressure_name = make_output_name("pressure")
                save_map(
                    pressure_map,
                    "Surface Pressure (mb)",
                    os.path.join(OUTPUT_DIR, pressure_name),
                    cmap="viridis"
                )
                image_urls["pressure_map"] = {
                    "url": url_for("output_file", filename=pressure_name),
                    "title": "Surface Pressure (mb)"
                }

            wbgt_available = (
                dew_map_c is not None and
                sol_map is not None and
                u2_map_ms is not None and
                pressure_map is not None
            )

            wbgt_warning = None

            # Summary display values
            if unit_system == "F":
                temp_min_display = c_to_f(t_min_c)
                temp_max_display = c_to_f(t_max_c)
                dew_min_display = c_to_f(d_min_c) if d_min_c is not None else None
                dew_max_display = c_to_f(d_max_c) if d_max_c is not None else None
            else:
                temp_min_display = t_min_c
                temp_max_display = t_max_c
                dew_min_display = d_min_c
                dew_max_display = d_max_c

            summary = {
                "run_at": datetime.utcnow().isoformat() + "Z",
                "inputs": {
                    "era_tair_input": era_tair_in,
                    "era_dpt_input": era_dpt_in,
                    "era_wspd_input": era_wspd_in,
                    "era_cloud_pct_at_obs": era_cloud,
                    "ref_pressure_mb": ref_pressure_mb,
                    "wbgt_time_local": dt_local.strftime("%Y-%m-%d %H:%M"),
                    "locked_time_note": LOCKED_TIME_LABEL,
                    "unit_system_label": unit_system_label,
                    "temp_unit": temp_unit,
                    "wind_unit": wind_unit,
                },
                "synoptic_predictions": {
                    "temp_max_C": t_max_c,
                    "temp_min_C": t_min_c,
                    "dew_max_C": d_max_c,
                    "dew_min_C": d_min_c,
                    "sol_max_Wm2": s_max,
                    "temp_max_display": temp_max_display,
                    "temp_min_display": temp_min_display,
                    "dew_max_display": dew_max_display,
                    "dew_min_display": dew_min_display,
                },
            }

            if wbgt_available:
                wbgt_k, tg_k, tnw_k = wbgt_from_fields(
                    tair_c=temp_map_c,
                    td_c=dew_map_c,
                    solar_wm2=sol_map,
                    pair_mb=pressure_map,
                    speed_ms=u2_map_ms,
                    dt_local=dt_local,
                    lat=SITE_LAT,
                    lon=SITE_LON,
                )

                wbgt_c = wbgt_k - 273.15
                tg_c = tg_k - 273.15
                tnw_c = tnw_k - 273.15

                if unit_system == "F":
                    wbgt_display = c_to_f(wbgt_c)
                    tg_display = c_to_f(tg_c)
                    tnw_display = c_to_f(tnw_c)
                    flag_unit = "F"
                else:
                    wbgt_display = wbgt_c
                    tg_display = tg_c
                    tnw_display = tnw_c
                    flag_unit = "C"

                wbgt_name = make_output_name("wbgt")
                save_map(
                    wbgt_display,
                    f"WBGT ({temp_unit})",
                    os.path.join(OUTPUT_DIR, wbgt_name),
                    cmap="bwr"
                )
                image_urls["wbgt_map"] = {
                    "url": url_for("output_file", filename=wbgt_name),
                    "title": f"WBGT ({temp_unit})"
                }

                tg_name = make_output_name("tg")
                save_map(
                    tg_display,
                    f"Black Globe Temperature ({temp_unit})",
                    os.path.join(OUTPUT_DIR, tg_name),
                    cmap="Reds"
                )
                image_urls["tg_map"] = {
                    "url": url_for("output_file", filename=tg_name),
                    "title": f"Black Globe Temperature ({temp_unit})"
                }

                tnw_name = make_output_name("tnw")
                save_map(
                    tnw_display,
                    f"Natural Wet Bulb Temperature ({temp_unit})",
                    os.path.join(OUTPUT_DIR, tnw_name),
                    cmap="BrBG"
                )
                image_urls["tnw_map"] = {
                    "url": url_for("output_file", filename=tnw_name),
                    "title": f"Natural Wet Bulb Temperature ({temp_unit})"
                }

                flag_name = make_output_name("flag")
                save_flag_map(
                    wbgt_display,
                    os.path.join(OUTPUT_DIR, flag_name),
                    unit=flag_unit
                )
                image_urls["flag_map"] = {
                    "url": url_for("output_file", filename=flag_name),
                    "title": f"WBGT Flag Levels ({temp_unit})"
                }
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

    return render_template_string(
        HTML_TEMPLATE,
        form_data=form_data,
        result=result,
        error=error,
        locked_time_label=LOCKED_TIME_LABEL,
        locked_time_str=LOCKED_TIME_STR,
        show_input_units=show_input_units,
    )


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)