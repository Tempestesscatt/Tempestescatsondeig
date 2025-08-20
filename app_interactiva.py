# -*- coding: utf-8 -*-
import streamlit as st
import openmeteo_requests
import requests_cache
from retry_requests import retry
import requests
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap, BoundaryNorm
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai
import io
from PIL import Image
import json
import hashlib
import os

# --- 0. CONFIGURACIÓ I CONSTANTS ---
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
FORECAST_DAYS = 4
API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = pytz.timezone('Europe/Madrid')
CIUTATS_CATALUNYA = {
    'Amposta': {'lat': 40.7093, 'lon': 0.5810}, 'Balaguer': {'lat': 41.7904, 'lon': 0.8066},
    'Banyoles': {'lat': 42.1197, 'lon': 2.7667}, 'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Berga': {'lat': 42.1051, 'lon': 1.8458}, 'Cervera': {'lat': 41.6669, 'lon': 1.2721},
    'El Pont de Suert': {'lat': 42.4101, 'lon': 0.7423}, 'El Vendrell': {'lat': 41.2201, 'lon': 1.5348},
    'Falset': {'lat': 41.1449, 'lon': 0.8197}, 'Figueres': {'lat': 42.2662, 'lon': 2.9622},
    'Gandesa': {'lat': 41.0526, 'lon': 0.4357}, 'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Granollers': {'lat': 41.6083, 'lon': 2.2886}, 'Igualada': {'lat': 41.5791, 'lon': 1.6174},
    'La Bisbal d\'Empordà': {'lat': 41.9602, 'lon': 3.0378}, 'La Seu d\'Urgell': {'lat': 42.3582, 'lon': 1.4593},
    'Les Borges Blanques': {'lat': 41.5226, 'lon': 0.8698}, 'Lleida': {'lat': 41.6177, 'lon': 0.6200},
    'Manresa': {'lat': 41.7230, 'lon': 1.8268}, 'Mataró': {'lat': 41.5388, 'lon': 2.4449},
    'Moià': {'lat': 41.8106, 'lon': 2.0975}, 'Mollerussa': {'lat': 41.6301, 'lon': 0.8958},
    'Montblanc': {'lat': 41.3761, 'lon': 1.1610}, 'Móra d\'Ebre': {'lat': 41.0945, 'lon': 0.6433},
    'Olot': {'lat': 42.1818, 'lon': 2.4900}, 'Prats de Lluçanès': {'lat': 42.0135, 'lon': 2.0305},
    'Puigcerdà': {'lat': 42.4331, 'lon': 1.9287}, 'Reus': {'lat': 41.1550, 'lon': 1.1075},
    'Ripoll': {'lat': 42.2013, 'lon': 2.1903}, 'Sant Feliu de Llobregat': {'lat': 41.3833, 'lon': 2.0500},
    'Santa Coloma de Farners': {'lat': 41.8596, 'lon': 2.6703}, 'Solsona': {'lat': 41.9942, 'lon': 1.5161},
    'Sort': {'lat': 42.4131, 'lon': 1.1278}, 'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
    'Tàrrega': {'lat': 41.6468, 'lon': 1.1416}, 'Terrassa': {'lat': 41.5615, 'lon': 2.0084},
    'Tortosa': {'lat': 40.8126, 'lon': 0.5211}, 'Tremp': {'lat': 42.1664, 'lon': 0.8953},
    'Valls': {'lat': 41.2872, 'lon': 1.2505}, 'Vic': {'lat': 41.9301, 'lon': 2.2545},
    'Vielha': {'lat': 42.7027, 'lon': 0.7966}, 'Vilafranca del Penedès': {'lat': 41.3453, 'lon': 1.6995},
    'Vilanova i la Geltrú': {'lat': 41.2241, 'lon': 1.7252},
}
# --- NOU: Llista de ciutats per al mode convidat ---
CIUTATS_CONVIDAT = {
    'Barcelona': CIUTATS_CATALUNYA['Barcelona'], 'Girona': CIUTATS_CATALUNYA['Girona'],
    'Lleida': CIUTATS_CATALUNYA['Lleida'], 'Tarragona': CIUTATS_CATALUNYA['Tarragona']
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
USERS_FILE = 'users.json'
RATE_LIMIT_FILE = 'rate_limits.json'

# --- 0.1 FUNCIONS D'AUTENTICACIÓ I LÍMITS ---

def get_hashed_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE): return {}
    try:
        with open(USERS_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_users(users_data):
    with open(USERS_FILE, 'w') as f: json.dump(users_data, f, indent=4)

def load_rate_limits():
    if not os.path.exists(RATE_LIMIT_FILE): return {}
    try:
        with open(RATE_LIMIT_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_rate_limits(rate_limits_data):
    with open(RATE_LIMIT_FILE, 'w') as f: json.dump(rate_limits_data, f, indent=4)

def format_time_left(time_delta):
    total_seconds = int(time_delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0: return f"{hours}h {minutes}min"
    else: return f"{minutes} min"

def show_login_page():
    st.markdown("<h1 style='text-align: center;'>Benvingut/da al Terminal de Temps Sever</h1>", unsafe_allow_html=True)

    selected = st.sidebar.radio("Menú", ["Inicia Sessió", "Registra't"])

    if selected == "Inicia Sessió":
        st.subheader("Inicia Sessió")
        with st.form("login_form"):
            username = st.text_input("Nom d'usuari", key="login_user")
            password = st.text_input("Contrasenya", type="password", key="login_pass")
            submitted = st.form_submit_button("Entra")
            if submitted:
                users = load_users()
                if username in users and users[username] == get_hashed_password(password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['guest_mode'] = False
                    st.rerun()
                else:
                    st.error("Nom d'usuari o contrasenya incorrectes.")

    elif selected == "Registra't":
        st.subheader("Crea un nou compte")
        with st.form("register_form"):
            new_username = st.text_input("Tria un nom d'usuari", key="reg_user")
            new_password = st.text_input("Tria una contrasenya", type="password", key="reg_pass")
            submitted = st.form_submit_button("Registra'm")
            if submitted:
                users = load_users()
                if new_username in users: st.error("Aquest nom d'usuari ja existeix.")
                elif len(new_password) < 6: st.error("La contrasenya ha de tenir com a mínim 6 caràcters.")
                else:
                    users[new_username] = get_hashed_password(new_password)
                    save_users(users)
                    st.success("Compte creat amb èxit! Ara pots iniciar sessió.")
                    st.balloons()
    
    # --- NOU: Botó per entrar com a convidat ---
    st.divider()
    if st.button("Entrar com a Convidat", use_container_width=True, type="secondary"):
        st.session_state['guest_mode'] = True
        st.session_state['logged_in'] = False
        st.rerun()

# --- 1. FUNCIONS D'OBTENCIÓ DE DADES (Sense Canvis) ---
@st.cache_data(ttl=3600)
def carregar_dades_sondeig(lat, lon, hourly_index):
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
        response = openmeteo.weather_api(API_URL, params=params)[0]
        hourly = response.Hourly()
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
        if any(np.isnan(val) for val in sfc_data.values()): return None, "Dades de superfície invàlides."
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS) + j).ValuesAsNumpy()[hourly_index] for j in range(len(PRESS_LEVELS))]
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]]
        u_profile, v_profile, h_profile = [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]
        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val); T_profile.append(p_data["T"][i]); Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
        p, T, Td = np.array(p_profile) * units.hPa, np.array(T_profile) * units.degC, np.array(Td_profile) * units.degC
        u, v, heights = np.array(u_profile) * units('m/s'), np.array(v_profile) * units('m/s'), np.array(h_profile) * units.meter
        prof = mpcalc.parcel_profile(p, T[0], Td[0])
        params_calc = {}; cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'], params_calc['CIN'] = (cape.to('J/kg').m if cape.magnitude > 0 else 0), cin.to('J/kg').m
        try: p_lcl, t_lcl = mpcalc.lcl(p[0], T[0], Td[0]); params_calc['LCL_hPa'] = p_lcl.m
        except Exception: params_calc['LCL_hPa'] = np.nan
        try: p_lfc, _ = mpcalc.lfc(p, T, Td); params_calc['LFC_hPa'] = p_lfc.m if not np.isnan(p_lfc.m) else np.nan
        except Exception: params_calc['LFC_hPa'] = np.nan
        try: p_el, _ = mpcalc.el(p, T, Td, prof); params_calc['EL_hPa'] = p_el.m if not np.isnan(p_el.m) else np.nan
        except Exception: params_calc['EL_hPa'] = np.nan
        params_calc['Shear 0-1km'], params_calc['Shear 0-6km'] = np.nan, np.nan
        try:
            shear_0_1km_u, shear_0_1km_v = mpcalc.bulk_shear(p, u, v, height=heights, depth=1000 * units.m)
            params_calc['Shear 0-1km'] = mpcalc.wind_speed(shear_0_1km_u, shear_0_1km_v).to('knots').m
        except (ValueError, IndexError): pass
        try:
            shear_0_6km_u, shear_0_6km_v = mpcalc.bulk_shear(p, u, v, height=heights, depth=6000 * units.m)
            params_calc['Shear 0-6km'] = mpcalc.wind_speed(shear_0_6km_u, shear_0_6km_v).to('knots').m
        except (ValueError, IndexError): pass
        try:
            srh_3km = mpcalc.storm_relative_helicity(heights, u, v, depth=3000 * units.meter)
            params_calc['SRH 0-3km'] = srh_3km[0].to('meter**2 / second**2').m
        except:
            params_calc['SRH 0-3km'] = np.nan
        return ((p, T, Td, u, v), params_calc), None
    except Exception as e: return None, f"Error en processar dades del sondeig: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_base(variables, hourly_index):
    try:
        lats, lons = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 12), np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
        responses = openmeteo.weather_api(API_URL, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
            if not any(np.isnan(v) for v in vals):
                output["lats"].append(r.Latitude())
                output["lons"].append(r.Longitude())
                for i, var in enumerate(variables):
                    output[var].append(vals[i])
        if not output["lats"]:
            return None, "No s'han rebut dades vàlides."
        return output, None
    except Exception as e:
        return None, f"Error en carregar dades del mapa: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa(nivell, hourly_index):
    try:
        if nivell >= 950:
            variables = ["dew_point_2m", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base(variables, hourly_index)
            if error: return None, error
            map_data_raw['dewpoint_data'] = map_data_raw.pop('dew_point_2m')
        else:
            variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base(variables, hourly_index)
            if error: return None, error
            temp_data = np.array(map_data_raw.pop(f'temperature_{nivell}hPa')) * units.degC
            rh_data = np.array(map_data_raw.pop(f'relative_humidity_{nivell}hPa')) * units.percent
            map_data_raw['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m

        map_data_raw['speed_data'] = map_data_raw.pop(f'wind_speed_{nivell}hPa')
        map_data_raw['dir_data'] = map_data_raw.pop(f'wind_direction_{nivell}hPa')

        return map_data_raw, None
    except Exception as e:
        return None, f"Error en processar dades del mapa: {e}"

# --- 2. FUNCIONS DE VISUALITZACIÓ (Sense Canvis) ---
def crear_mapa_base():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=90, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    return fig, ax
def crear_mapa_forecast_combinat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 400), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 400))
    grid_speed, grid_dewpoint = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    colors_wind_new = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db', '#87ceeb', '#48d1cc', '#b0c4de', '#da70d6', '#ffdead', '#ffd700', '#9acd32', '#a9a9a9']
    speed_levels_new = [0, 4, 11, 18, 25, 32, 40, 47, 54, 61, 68, 76, 86, 97, 104, 130, 166, 184, 277, 374, 400]
    cbar_ticks = [0, 18, 40, 61, 86, 130, 184, 374]
    custom_cmap = ListedColormap(colors_wind_new)
    norm_speed = BoundaryNorm(speed_levels_new, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=cbar_ticks)
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density= 4, arrowsize=0.4, zorder=4, transform=ccrs.PlateCarree())
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    dudx = mpcalc.first_derivative(grid_u * units('m/s'), delta=dx, axis=1)
    dvdy = mpcalc.first_derivative(grid_v * units('m/s'), delta=dy, axis=0)
    divergence = (dudx + dvdy).to('1/s')
    convergence_scaled = -divergence.magnitude * 1e5
    CONVERGENCE_THRESHOLD = 20
    if nivell >= 950: DEWPOINT_THRESHOLD = 14
    elif nivell >= 925: DEWPOINT_THRESHOLD = 12
    else: DEWPOINT_THRESHOLD = 7
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD, convergence_scaled, 0)
    max_convergence = np.nanmax(convergence_in_humid_areas)
    if max_convergence >= CONVERGENCE_THRESHOLD:
        single_level = max_convergence * 0.80
        if single_level >= CONVERGENCE_THRESHOLD:
            fill_levels = [single_level, max_convergence]
            line_levels = [single_level]
            ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=['#FF0000'], alpha=0.3, zorder=5, transform=ccrs.PlateCarree())
            contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors='black', linestyles='-', linewidths=1.5, zorder=6, transform=ccrs.PlateCarree())
            labels = ax.clabel(contours, inline=True, fontsize=5, fmt='%1.0f')
            for label in labels: label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.4))
    ax.set_title(f"Anàlisi de Vent i Nuclis de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig
def crear_mapa_vents(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    colors_wind_new = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db', '#87ceeb', '#48d1cc', '#b0c4de', '#da70d6', '#ffdead', '#ffd700', '#9acd32', '#a9a9a9']
    speed_levels_new = [0, 4, 11, 18, 25, 32, 40, 47, 54, 61, 68, 76, 86, 97, 104, 130, 166, 184, 277, 374, 400]
    cbar_ticks = [0, 18, 40, 61, 86, 130, 184, 374]
    custom_cmap = ListedColormap(colors_wind_new)
    norm_speed = BoundaryNorm(speed_levels_new, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, arrowsize=0.6, zorder=3, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, ticks=cbar_ticks)
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig
def crear_skewt(p, T, Td, u, v, titol):
    fig = plt.figure(figsize=(9, 9), dpi=150); skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.8, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5); skew.plot(p, T, 'r', lw=2, label='Temperatura'); skew.plot(p, Td, 'g', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03); skew.plot_dry_adiabats(color='brown', linestyle='--', alpha=0.6)
    skew.plot_moist_adiabats(color='blue', linestyle='--', alpha=0.6); skew.plot_mixing_lines(color='green', linestyle='--', alpha=0.6)
    prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', linewidth=2, label='Trajectòria Parcel·la')
    skew.shade_cape(p, T, prof, color='red', alpha=0.3); skew.shade_cin(p, T, prof, color='blue', alpha=0.3)
    skew.ax.set_ylim(1000, 100); skew.ax.set_xlim(-40, 40); skew.ax.set_title(titol, weight='bold', fontsize=14); skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")
    skew.ax.legend(); return fig
def crear_hodograf(u, v):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150); h = Hodograph(ax, component_range=60.)
    h.add_grid(increment=20, color='gray'); h.plot(u.to('kt'), v.to('kt'), color='red', linewidth=2)
    ax.set_title("Hodògraf", weight='bold'); return fig
def mostrar_imatge_temps_real(tipus):
    if tipus == "Satèl·lit (Europa)": url = "https://modeles20.meteociel.fr/satellite/animsatsandvisirmtgeu.gif"; caption = "Satèl·lit Sandvitx (Visible + Infraroig). Font: Meteociel"
    elif tipus == "Satèl·lit (NE Península)":
        now_local = datetime.now(TIMEZONE)
        if 7 <= now_local.hour < 21: url = "https://modeles20.meteociel.fr/satellite/animsatviscolmtgsp.gif"; caption = "Satèl·lit Visible (Nord-est). Font: Meteociel"
        else: url = "https://modeles20.meteociel.fr/satellite/animsatirmtgsp.gif"; caption = "Satèl·lit Infraroig (Nord-est). Font: Meteociel"
    else: st.error("Tipus d'imatge no reconegut."); return
    try:
        response = requests.get(f"{url}?ver={int(time.time())}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code == 200: st.image(response.content, caption=caption, use_container_width=True)
        else: st.warning(f"No s'ha pogut carregar la imatge. (Codi: {response.status_code})")
    except Exception as e: st.error(f"Error de xarxa en carregar la imatge.")

# --- 3. FUNCIONS PER A L'ASSISTENT D'IA (Sense Canvis) ---
def get_color_for_param(param_name, value):
    if value is None or np.isnan(value): return "#808080"
    if param_name == 'CAPE':
        if value < 100: return "#808080";
        if value < 1000: return "#39FF14"
        if value < 2500: return "#FF3131"
        return "#BC13FE"
    elif param_name == 'CIN':
        if value > -25: return "#39FF14"
        if value > -75: return "#FF3131"
        return "#BC13FE"
    elif param_name == 'LFC_hPa':
        if value > 900: return "#39FF14"
        if value > 800: return "#FF3131"
        return "#BC13FE"
    elif param_name == 'Shear 0-1km':
        if value < 5: return "#808080"
        if value < 15: return "#39FF14"
        if value < 25: return "#FF3131"
        return "#BC13FE"
    elif param_name == 'Shear 0-6km':
        if value < 20: return "#808080"
        if value < 35: return "#39FF14"
        if value < 50: return "#FF3131"
        return "#BC13FE"
    return "#FFFFFF"
def ui_pestanya_ia(data_tuple, hourly_index_sel, poble_sel, timestamp_str):
    st.subheader("Assistent MeteoIA (amb Google Gemini)")
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        st.error("No s'ha pogut configurar l'API de Gemini. Assegura't que la GEMINI_API_KEY està als secrets.")
        return
    username = st.session_state.get('username')
    if not username:
        st.error("Error d'autenticació. Si us plau, torna a iniciar sessió.")
        return
    LIMIT_PER_WINDOW = 10
    WINDOW_HOURS = 3
    rate_limits = load_rate_limits()
    user_limit_data = rate_limits.get(username, {"count": 0, "window_start_time": None})
    limit_reached = False
    if user_limit_data["window_start_time"]:
        start_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc)
        elapsed_time = datetime.now(pytz.utc) - start_time
        if elapsed_time > timedelta(hours=WINDOW_HOURS):
            user_limit_data["count"] = 0
            user_limit_data["window_start_time"] = None
    if user_limit_data["count"] >= LIMIT_PER_WINDOW:
        limit_reached = True
        start_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc)
        time_to_reset = (start_time + timedelta(hours=WINDOW_HOURS))
        time_left = time_to_reset - datetime.now(pytz.utc)
        if time_left.total_seconds() > 0:
            st.warning(f"""**Has arribat al límit de {LIMIT_PER_WINDOW} preguntes.** 
                        El teu accés es renovarà en aproximadament **{format_time_left(time_left)}**.""")
        else:
            user_limit_data["count"] = 0; user_limit_data["window_start_time"] = None
            rate_limits[username] = user_limit_data; save_rate_limits(rate_limits)
            limit_reached = False
    if "chat" not in st.session_state:
        model = genai.GenerativeModel('gemini-1.5-flash')
        system_prompt = "..." # (El prompt llarg de l'IA es manté igual)
        st.session_state.chat = model.start_chat(history=[{'role': 'user', 'parts': [system_prompt]},{'role': 'model', 'parts': ["Hola! Sóc Tempestes.CAT-IA..."]}])
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola! Sóc Tempestes.CAT-IA..."}]
    st.markdown(f"**Anàlisi per:** `{poble_sel.upper()}` | **Dia:** `{timestamp_str}`")
    nivell_mapa_ia = st.selectbox("Canvia el nivell d'anàlisi del mapa (només per a l'IA):",options=[1000, 950, 925, 850, 800, 700],format_func=lambda x: f"{x} hPa",key="ia_level_selector_chat",disabled=limit_reached)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt_usuari := st.chat_input("Quina és la teva pregunta sobre el temps?", disabled=limit_reached):
        st.session_state.messages.append({"role": "user", "content": prompt_usuari})
        with st.chat_message("user"): st.markdown(prompt_usuari)
        with st.chat_message("assistant"):
            with st.spinner("Generant mapa i consultant l'IA..."):
                if user_limit_data["window_start_time"] is None: user_limit_data["window_start_time"] = datetime.now(pytz.utc).timestamp()
                user_limit_data["count"] += 1; rate_limits[username] = user_limit_data; save_rate_limits(rate_limits)
                map_data_ia, error_map_ia = carregar_dades_mapa(nivell_mapa_ia, hourly_index_sel)
                if error_map_ia: st.error(f"No s'han pogut carregar les dades: {error_map_ia}"); return
                fig_mapa = crear_mapa_forecast_combinat(map_data_ia['lons'], map_data_ia['lats'], map_data_ia['speed_data'], map_data_ia['dir_data'], map_data_ia['dewpoint_data'], nivell_mapa_ia, timestamp_str)
                buf = io.BytesIO(); fig_mapa.savefig(buf, format='png', dpi=150, bbox_inches='tight'); buf.seek(0); img_mapa = Image.open(buf); plt.close(fig_mapa)
                resum_sondeig = "No hi ha dades de sondeig."
                if data_tuple: _, params_calculats = data_tuple; resum_sondeig = f"- CAPE: {params_calculats.get('CAPE', 0):.0f} J/kg. ..."
                prompt_context_torn_actual = f"DADES:\n- Localització: {poble_sel}\n- Sondeig: {resum_sondeig}\nTASCA: Analitza la imatge i les dades per respondre: '{prompt_usuari}'"
                try:
                    resposta_completa = st.session_state.chat.send_message([prompt_context_torn_actual, img_mapa])
                    full_response = resposta_completa.text
                    st.markdown(full_response)
                except Exception as e:
                    error_text = str(e)
                    if "429" in error_text: full_response = "**Límit de consultes a l'API de Google superat.**"
                    else: full_response = f"Error inesperat contactant amb l'IA."
                    st.error(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- 4. LÒGICA DE LA INTERFÍCIE D'USUARI (MODIFICADA) ---
def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    
    # --- NOU: Lògica per adaptar la capçalera al mode (convidat o usuari) ---
    col_text, col_button = st.columns([0.85, 0.15])
    
    with col_text:
        if st.session_state.get('guest_mode'):
            st.info("ℹ️ Estàs en **Mode Convidat** amb funcionalitats limitades. Per accedir a l'assistent IA i a totes les localitats, registra't i inicia sessió.")
        else:
            st.markdown(f"Benvingut/da, **{st.session_state.get('username')}**!")

    with col_button:
        button_text = "Sortir" if st.session_state.get('guest_mode') else "Tanca Sessió"
        if st.button(button_text):
            # Esborra totes les claus de sessió rellevants per reiniciar
            for key in ['logged_in', 'username', 'guest_mode', 'chat', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            # --- NOU: Tria la llista de ciutats segons el mode ---
            if st.session_state.get('guest_mode'):
                ciutats_a_mostrar = CIUTATS_CONVIDAT
            else:
                ciutats_a_mostrar = CIUTATS_CATALUNYA
            st.selectbox("Capital de referència:", sorted(ciutats_a_mostrar.keys()), key="poble_selector")
        
        with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3: st.selectbox("Hora del pronòstic (Hora Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

def ui_explicacio_alertes():
    with st.expander("Què signifiquen les isòlines de convergència?"):
        st.markdown("""
        Les línies vermelles discontínues (`---`) marquen zones de **convergència d'humitat**. Són els **disparadors** potencials de tempestes.
        - **Què són?** Àrees on el vent força l'aire humit a ajuntar-se i ascendir.
        - **Com interpretar-les?** El número sobre la línia indica la seva intensitat (més alt = més fort). Valors > 20 són significatius.
        """)
def ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple):
    col_map_1, col_map_2 = st.columns([0.7, 0.3], gap="large")
    with col_map_1:
        map_options = {"Anàlisi de Vent i Convergència": "forecast_estatic", "Vent a 700hPa": "vent_700", "Vent a 300hPa": "vent_300"}
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
        map_key = map_options[mapa_sel]
        if map_key == "forecast_estatic":
            if data_tuple and data_tuple[1]:
                cin_value, lfc_hpa = data_tuple[1].get('CIN', 0), data_tuple[1].get('LFC_hPa', np.nan)
                if cin_value < -25: st.warning(f"**AVÍS DE 'TAPA' (CIN = {cin_value:.0f} J/kg):** El sondeig mostra una forta inversió.")
                if np.isnan(lfc_hpa): st.error("**DIAGNÒSTIC LFC:** No s'ha trobat LFC. Atmosfera estable.")
                elif lfc_hpa >= 900: st.success(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció superficial. Recomanació: 1000-925 hPa.")
                elif lfc_hpa >= 750: st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció baixa. Recomanació: 850-800 hPa.")
                else: st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció elevada. Recomanació: 700 hPa.")
            nivell_sel = st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa")
            with progress_placeholder.container():
                progress_bar = st.progress(0, text="Carregant dades del model...")
                map_data, error_map = carregar_dades_mapa(nivell_sel, hourly_index_sel)
                if not error_map: progress_bar.progress(50, text="Generant visualització...")
            if error_map: st.error(f"Error en carregar el mapa: {error_map}"); progress_placeholder.empty()
            elif map_data:
                fig = crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str)
                st.pyplot(fig); plt.close(fig)
                with progress_placeholder.container(): progress_bar.progress(100, text="Completat!"); time.sleep(1); progress_bar.empty()
                ui_explicacio_alertes()
        elif map_key in ["vent_700", "vent_300"]:
            nivell = 700 if map_key == "vent_700" else 300
            variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data, error_map = carregar_dades_mapa_base(variables, hourly_index_sel)
            if error_map: st.error(f"Error en carregar el mapa: {error_map}")
            elif map_data: 
                fig = crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str)
                st.pyplot(fig); plt.close(fig)
    with col_map_2:
        st.subheader("Imatges en Temps Real")
        tab_europa, tab_ne = st.tabs(["Europa", "NE Peninsula"])
        with tab_europa: mostrar_imatge_temps_real("Satèl·lit (Europa)")
        with tab_ne: mostrar_imatge_temps_real("Satèl·lit (NE Península)")
def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        cols = st.columns(5)
        metric_params = {'CAPE': 'J/kg', 'CIN': 'J/kg', 'LFC_hPa': 'hPa', 'Shear 0-1km': 'nusos', 'Shear 0-6km': 'nusos'}
        for i, (param, unit) in enumerate(metric_params.items()):
            with cols[i]:
                val = params_calculats.get(param)
                color = get_color_for_param(param, val)
                value_str = f"{val:.0f}" if val is not None and not np.isnan(val) else "---"
                st.markdown(f"""<div style="text-align: left;"><span style="font-size: 0.8em; color: #A0A0A0;">{param}</span><br><strong style="font-size: 1.8em; color: {color};">{value_str}</strong> <span style="font-size: 1.1em; color: #A0A0A0;">{unit}</span></div>""", unsafe_allow_html=True)
        with st.expander("Què signifiquen aquests paràmetres?"):
            st.markdown("- **CAPE:** Energia per a tempestes. >1000 J/kg és significatiu.\n- **CIN:** \"Tapa\" que impedeix la convecció. > -50 és forta.\n- **LFC:** Nivell on comença la convecció. Com més baix, millor.\n- **Shear 0-1km:** Cisallament baix. >15-20 nusos afavoreix rotació i **tornados**.\n- **Shear 0-6km:** Cisallament profund. >35-40 nusos és clau per a **supercèl·lules**.")
        st.divider()
        col1, col2 = st.columns(2)
        with col1: fig = crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], f"Sondeig Vertical - {poble_sel}"); st.pyplot(fig); plt.close(fig)
        with col2: fig = crear_hodograf(sounding_data[3], sounding_data[4]); st.pyplot(fig); plt.close(fig)
    else: st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")
def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- 5. APLICACIÓ PRINCIPAL (MODIFICADA) ---
def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'guest_mode' not in st.session_state: st.session_state['guest_mode'] = False
    
    # Si no està ni loguejat ni en mode convidat, mostra la pàgina de login
    if not st.session_state['logged_in'] and not st.session_state['guest_mode']:
        show_login_page()
    else:
        # Lògica de l'aplicació principal (visible per a usuaris i convidats)
        if 'poble_selector' not in st.session_state:
            st.session_state.poble_selector = 'Barcelona'
            st.session_state.dia_selector = 'Avui'
            st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"
        ui_capcalera_selectors()
        current_selection = f"{st.session_state.poble_selector}-{st.session_state.dia_selector}-{st.session_state.hora_selector}"
        if current_selection != st.session_state.get('last_selection'):
            if 'messages' in st.session_state: del st.session_state.messages
            if 'chat' in st.session_state: del st.session_state.chat
            st.session_state.last_selection = current_selection

        poble_sel, dia_sel, hora_sel = st.session_state.poble_selector, st.session_state.dia_selector, st.session_state.hora_selector
        hora_int = int(hora_sel.split(':')[0]); now_local = datetime.now(TIMEZONE); target_date = now_local.date()
        if dia_sel == "Demà": target_date += timedelta(days=1)
        local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int)); utc_dt = local_dt.astimezone(pytz.utc)
        start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600); hourly_index_sel = max(0, time_diff_hours)
        timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
        
        # --- NOU: Tria el diccionari de ciutats correcte per obtenir lat/lon ---
        if st.session_state.get('guest_mode'):
            lat_sel, lon_sel = CIUTATS_CONVIDAT[poble_sel]['lat'], CIUTATS_CONVIDAT[poble_sel]['lon']
        else:
            lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']

        data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
        if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
        
        st.markdown("---")
        global progress_placeholder; progress_placeholder = st.empty()

        # --- NOU: Lògica per mostrar pestanyes segons el mode ---
        if st.session_state.get('guest_mode'):
            # Mode Convidat: només mapes i anàlisi vertical
            tab_mapes, tab_vertical = st.tabs(["Anàlisi de Mapes", "Anàlisi Vertical"])
            with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
            with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
        else:
            # Mode Usuari: totes les pestanyes
            tab_ia, tab_mapes, tab_vertical = st.tabs(["Assistent MeteoIA", "Anàlisi de Mapes", "Anàlisi Vertical"])
            with tab_ia: ui_pestanya_ia(data_tuple, hourly_index_sel, poble_sel, timestamp_str)
            with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
            with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)

        ui_peu_de_pagina()

if __name__ == "__main__":
    main()
