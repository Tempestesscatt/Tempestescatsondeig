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
from scipy.interpolate import griddata, interp1d
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai
import io
from PIL import Image
import json
import hashlib
import os
import base64
import threading

# --- 0. CONFIGURACIÓ I CONSTANTS ---
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")

parcel_lock = threading.Lock()
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
CIUTATS_CONVIDAT = {
    'Barcelona': CIUTATS_CATALUNYA['Barcelona'], 'Girona': CIUTATS_CATALUNYA['Girona'],
    'Lleida': CIUTATS_CATALUNYA['Lleida'], 'Tarragona': CIUTATS_CATALUNYA['Tarragona']
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
USERS_FILE = 'users.json'
RATE_LIMIT_FILE = 'rate_limits.json'
CHAT_FILE = 'chat_history.json'
MAP_ZOOM_LEVELS = {'Catalunya (Complet)': MAP_EXTENT, 'Nord-est (Girona)': [1.8, 3.4, 41.7, 42.6], 'Sud (Tarragona i Ebre)': [0.2, 1.8, 40.5, 41.4], 'Ponent i Pirineu (Lleida)': [0.4, 1.9, 41.4, 42.6], 'Àrea Metropolitana (BCN)': [1.7, 2.7, 41.2, 41.8]}

def get_hashed_password(password): return hashlib.sha256(password.encode()).hexdigest()
def load_json_file(filename):
    if not os.path.exists(filename): return {} if 'users' in filename or 'rate' in filename else []
    try:
        with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {} if 'users' in filename or 'rate' in filename else []
def save_json_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
def load_and_clean_chat_history():
    if not os.path.exists(CHAT_FILE): return []
    try:
        with open(CHAT_FILE, 'r', encoding='utf-8') as f: history = json.load(f)
        one_hour_ago_ts = datetime.now(pytz.utc).timestamp() - 3600
        cleaned_history = [msg for msg in history if msg.get('timestamp', 0) > one_hour_ago_ts]
        if len(cleaned_history) < len(history): save_json_file(cleaned_history, CHAT_FILE)
        return cleaned_history
    except (json.JSONDecodeError, FileNotFoundError): return []
def count_unread_messages(history):
    last_seen = st.session_state.get('last_seen_timestamp', 0); current_user = st.session_state.get('username')
    return sum(1 for msg in history if msg['timestamp'] > last_seen and msg['username'] != current_user)
def format_time_left(time_delta):
    total_seconds = int(time_delta.total_seconds()); hours, remainder = divmod(total_seconds, 3600); minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}min" if hours > 0 else f"{minutes} min"
def show_login_page():
    st.markdown("<h1 style='text-align: center;'>Benvingut/da al Terminal de Temps Sever</h1>", unsafe_allow_html=True)
    selected = st.sidebar.radio("Menú", ["Inicia Sessió", "Registra't"])
    if selected == "Inicia Sessió":
        st.subheader("Inicia Sessió")
        with st.form("login_form"):
            username = st.text_input("Nom d'usuari"); password = st.text_input("Contrasenya", type="password")
            if st.form_submit_button("Entra"):
                users = load_json_file(USERS_FILE)
                if username in users and users[username] == get_hashed_password(password):
                    st.session_state.update({'logged_in': True, 'username': username, 'guest_mode': False}); st.rerun()
                else: st.error("Nom d'usuari o contrasenya incorrectes.")
    elif selected == "Registra't":
        st.subheader("Crea un nou compte")
        with st.form("register_form"):
            new_username = st.text_input("Tria un nom d'usuari"); new_password = st.text_input("Tria una contrasenya", type="password")
            if st.form_submit_button("Registra'm"):
                users = load_json_file(USERS_FILE)
                if not new_username or not new_password: st.error("El nom d'usuari i la contrasenya no poden estar buits.")
                elif new_username in users: st.error("Aquest nom d'usuari ja existeix.")
                elif len(new_password) < 6: st.error("La contrasenya ha de tenir com a mínim 6 caràcters.")
                else:
                    users[new_username] = get_hashed_password(new_password); save_json_file(users, USERS_FILE)
                    st.success("Compte creat amb èxit! Ara pots iniciar sessió.")
    st.divider()
    if st.button("Entrar com a Convidat", use_container_width=True, type="secondary"):
        st.session_state.update({'guest_mode': True, 'logged_in': False}); st.rerun()

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
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]
        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val); T_profile.append(p_data["T"][i]); Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
        valid_indices = ~np.isnan(p_profile) & ~np.isnan(T_profile) & ~np.isnan(Td_profile) & ~np.isnan(u_profile) & ~np.isnan(v_profile) & ~np.isnan(h_profile)
        p, T, Td = np.array(p_profile)[valid_indices] * units.hPa, np.array(T_profile)[valid_indices] * units.degC, np.array(Td_profile)[valid_indices] * units.degC
        u, v, heights = np.array(u_profile)[valid_indices] * units('m/s'), np.array(v_profile)[valid_indices] * units('m/s'), np.array(h_profile)[valid_indices] * units.meter
        
        params_calc = {}
        with parcel_lock:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE_total'] = cape.to('J/kg').m; params_calc['CIN_total'] = cin.to('J/kg').m
        
        try: params_calc['MU_CAPE'] = mpcalc.most_unstable_cape_cin(p, T, Td)[0].to('J/kg').m
        except: params_calc['MU_CAPE'] = np.nan
        try: params_calc['ML_CAPE'] = mpcalc.mixed_layer_cape_cin(p, T, Td)[0].to('J/kg').m
        except: params_calc['ML_CAPE'] = np.nan
        try: params_calc['PWAT'] = mpcalc.precipitable_water(p, Td).to('mm').m
        except: params_calc['PWAT'] = np.nan
        try: params_calc['DCAPE'] = mpcalc.dcape(p, T, Td)[0].to('J/kg').m
        except: params_calc['DCAPE'] = np.nan

        heights_agl = heights - heights[0]
        try:
            lcl_p, lcl_h = mpcalc.lcl(p[0], T[0], Td[0], max_iters=50)
            params_calc['LCL_p'] = lcl_p; params_calc['LCL_Hgt'] = np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1])
        except: params_calc['LCL_p'], params_calc['LCL_Hgt'] = np.nan, np.nan
        try:
            lfc_p, _ = mpcalc.lfc(p, T, Td, prof)
            params_calc['LFC_p'] = lfc_p; params_calc['LFC_Hgt'] = mpcalc.pressure_to_height_std(lfc_p).to('m').m if lfc_p else np.nan
        except: params_calc['LFC_p'], params_calc['LFC_Hgt'] = np.nan, np.nan
        try:
            el_p, _ = mpcalc.el(p, T, Td, prof)
            params_calc['EL_p'] = el_p; params_calc['EL_Hgt'] = mpcalc.pressure_to_height_std(el_p).to('m').m if el_p else np.nan
        except: params_calc['EL_p'], params_calc['EL_Hgt'] = np.nan, np.nan
            
        try:
            p_frz = np.interp(0, T.to('degC').m[::-1], p.m[::-1]) * units.hPa
            h_frz = mpcalc.pressure_to_height_std(p_frz).to('m').m
            params_calc['FRZG_Lvl_p'] = p_frz; params_calc['FRZG_Lvl_Hgt'] = h_frz
        except: params_calc['FRZG_Lvl_p'], params_calc['FRZG_Lvl_Hgt'] = np.nan, np.nan
        
        try: params_calc['KI'] = mpcalc.k_index(p, T, Td).m
        except: params_calc['KI'] = np.nan
        try: params_calc['TT'] = mpcalc.total_totals_index(p, T, Td).m
        except: params_calc['TT'] = np.nan
        try: params_calc['BWD_0_6km'] = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u, v, height=heights, depth=6000*units.m)).to('kt').m
        except: params_calc['BWD_0_6km'] = np.nan
        
        return ((p, T, Td, u, v, heights), params_calc), None
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
                output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                for i, var in enumerate(variables): output[var].append(vals[i])
        if not output["lats"]: return None, "No s'han rebut dades vàlides."
        return output, None
    except Exception as e: return None, f"Error en carregar dades del mapa: {e}"

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
    except Exception as e: return None, f"Error en processar dades del mapa: {e}"

@st.cache_data(ttl=3600)
def obtenir_ciutats_actives(hourly_index):
    nivell = 925
    map_data, error_map = carregar_dades_mapa(nivell, hourly_index)
    if error_map or not map_data: return CIUTATS_CONVIDAT, "No s'ha pogut determinar les zones de convergència."
    try:
        lons, lats, speed_data, dir_data, dewpoint_data = map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data']
        grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100))
        grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        dudx = mpcalc.first_derivative(grid_u * units('m/s'), delta=dx, axis=1); dvdy = mpcalc.first_derivative(grid_v * units('m/s'), delta=dy, axis=0)
        convergence_scaled = - (dudx + dvdy).to('1/s').magnitude * 1e5
        convergence_in_humid_areas = np.where(grid_dewpoint >= 12, convergence_scaled, 0)
        points_with_conv = np.argwhere(convergence_in_humid_areas >= 15)
        if points_with_conv.size == 0: return CIUTATS_CONVIDAT, "No s'han detectat zones de convergència significatives."
        conv_coords = [(grid_lat[y, x], grid_lon[y, x]) for y, x in points_with_conv]
        zone_centers = []; ZONE_RADIUS = 0.75
        for lat, lon in sorted(conv_coords, key=lambda c: convergence_in_humid_areas[np.argmin(np.abs(grid_lat[:,0]-c[0])), np.argmin(np.abs(grid_lon[0,:]-c[1]))], reverse=True):
            if all(np.sqrt((lat - clat)**2 + (lon - clon)**2) >= ZONE_RADIUS for clat, clon in zone_centers): zone_centers.append((lat, lon))
        if not zone_centers: return CIUTATS_CONVIDAT, "No s'han pogut identificar nuclis de convergència."
        ciutat_noms = list(CIUTATS_CATALUNYA.keys()); ciutat_coords = np.array([[v['lat'], v['lon']] for v in CIUTATS_CATALUNYA.values()])
        closest_cities_names = {ciutat_noms[np.argmin(cdist(np.array([[zlat, zlon]]), ciutat_coords))] for zlat, zlon in zone_centers}
        if not closest_cities_names: return CIUTATS_CONVIDAT, "No s'ha trobat cap ciutat propera als nuclis."
        return {name: CIUTATS_CATALUNYA[name] for name in closest_cities_names}, f"Selecció de {len(closest_cities_names)} poblacions properes a nuclis d'activitat."
    except Exception as e: return CIUTATS_CONVIDAT, f"Error calculant zones actives: {e}."

def crear_mapa_base(map_extent):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=90, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(map_extent, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0); ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5); return fig, ax
def crear_mapa_forecast_combinat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent)
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 400), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 400))
    grid_speed, grid_dewpoint = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    colors_wind_new = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db', '#87ceeb', '#48d1cc', '#b0c4de', '#da70d6', '#ffdead', '#ffd700', '#9acd32', '#a9a9a9']
    speed_levels_new = [0, 4, 11, 18, 25, 32, 40, 47, 54, 61, 68, 76, 86, 97, 104, 130, 166, 184, 277, 374, 400]; cbar_ticks = [0, 18, 40, 61, 86, 130, 184, 374]
    custom_cmap = ListedColormap(colors_wind_new); norm_speed = BoundaryNorm(speed_levels_new, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=cbar_ticks)
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density= 4, arrowsize=0.4, zorder=4, transform=ccrs.PlateCarree())
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    dudx = mpcalc.first_derivative(grid_u * units('m/s'), delta=dx, axis=1); dvdy = mpcalc.first_derivative(grid_v * units('m/s'), delta=dy, axis=0)
    convergence_scaled = -(dudx + dvdy).to('1/s').magnitude * 1e5
    DEWPOINT_THRESHOLD = 14 if nivell >= 950 else 12 if nivell >= 925 else 7
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD, convergence_scaled, 0)
    fill_levels = [15, 25, 40, 150]; fill_colors = ['#ffc107', '#ff9800', '#f44336']; line_levels = [15, 25, 40]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    line_styles = ['--', '--', '-']; line_widths = [1, 1.2, 1.5]
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.4, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles=line_styles, linewidths=line_widths, zorder=6, transform=ccrs.PlateCarree())
    labels = ax.clabel(contours, inline=True, fontsize=6, fmt='%1.0f')
    for label in labels: label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.6))
    ax.set_title(f"Vent i Nuclis de convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16); return fig
def crear_mapa_vents(lons, lats, speed_data, dir_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent)
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    colors_wind_new = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db', '#87ceeb', '#48d1cc', '#b0c4de', '#da70d6', '#ffdead', '#ffd700', '#9acd32', '#a9a9a9']
    speed_levels_new = [0, 4, 11, 18, 25, 32, 40, 47, 54, 61, 68, 76, 86, 97, 104, 130, 166, 184, 277, 374, 400]; cbar_ticks = [0, 18, 40, 61, 86, 130, 184, 374]
    custom_cmap = ListedColormap(colors_wind_new); norm_speed = BoundaryNorm(speed_levels_new, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, arrowsize=0.6, zorder=3, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, ticks=cbar_ticks)
    cbar.set_label("Velocitat del Vent (km/h)"); ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16); return fig
def crear_skewt(p, T, Td, u, v, params_calc, titol):
    fig = plt.figure(dpi=150); fig.set_figheight(fig.get_figwidth() * 1.1)
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.05, 0.8, 0.9))
    skew.ax.grid(True, linestyle='-', alpha=0.5); skew.plot(p, T, 'r', lw=2.5, label='Temperatura'); skew.plot(p, Td, 'g', lw=2.5, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03); skew.plot_dry_adiabats(color='brown', linestyle='--', alpha=0.6)
    skew.plot_moist_adiabats(color='blue', linestyle='--', alpha=0.6); skew.plot_mixing_lines(color='green', linestyle='--', alpha=0.6)
    prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', linewidth=3, label='Trajectòria Parcel·la', path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])
    skew.shade_cape(p, T, prof, color='red', alpha=0.3); skew.shade_cin(p, T, prof, color='blue', alpha=0.3)
    skew.ax.set_ylim(1000, 100); skew.ax.set_xlim(-40, 40); skew.ax.set_title(titol, weight='bold', fontsize=14); skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")
    
    levels_to_plot = {'LCL_p': 'LCL', 'FRZG_Lvl_p': '0°C', 'LFC_p': None, 'EL_p': None}
    for key, name in levels_to_plot.items():
        p_lvl = params_calc.get(key)
        if p_lvl is not None and hasattr(p_lvl, 'm') and not np.isnan(p_lvl.m):
            skew.ax.axhline(p_lvl.m, color='blue', linestyle='--', linewidth=1.5)
            if name: skew.ax.text(skew.ax.get_xlim()[1], p_lvl.m, f' {name}', color='blue', ha='left', va='center', fontsize=10, weight='bold')
    try:
        for bottom_p, top_p in mpcalc.inversions(p, T, Td):
            skew.ax.axhspan(bottom_p.m, top_p.m, color='blue', alpha=0.15, linewidth=0)
            skew.ax.text(skew.ax.get_xlim()[0] + 2, (bottom_p.m + top_p.m) / 2, 'Subsidència', color='blue', ha='left', va='center', fontsize=9)
    except: pass
    skew.ax.legend(); return fig
def crear_hodograf_avancat(p, u, v, heights, titol):
    fig = plt.figure(dpi=150); fig.set_figheight(fig.get_figwidth() * 1.1)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.2, 1], top=0.92, bottom=0.05, left=0.05, right=0.95)
    ax_hodo = fig.add_subplot(gs[0, 0]); ax_info = fig.add_subplot(gs[0, 1]); ax_info.axis('off')
    fig.suptitle(titol, weight='bold', fontsize=16); h = Hodograph(ax_hodo, component_range=80.)
    h.add_grid(increment=20, color='gray', linestyle='--'); h.plot_colormapped(u.to('kt'), v.to('kt'), heights.to('km'), intervals=np.array([0, 3, 6, 12]) * units.km, colors=['red', 'green', 'blue'], linewidth=4)
    params = {'BWD (nusos)': {}}; motion = {}; right_mover, critical_angle, sr_wind_speed = None, np.nan, None
    try:
        right_mover, left_mover, mean_wind_vec = mpcalc.bunkers_storm_motion(p, u, v, heights)
        motion['RM'] = right_mover; motion['LM'] = left_mover; motion['Vent Mitjà'] = mean_wind_vec
        for name, vec in motion.items():
            u_comp, v_comp = vec[0].to('kt').m, vec[1].to('kt').m; marker = 's' if 'Mitjà' in name else 'o'
            ax_hodo.plot(u_comp, v_comp, marker=marker, color='black', markersize=8, fillstyle='none', mew=1.5)
    except (ValueError, IndexError): right_mover = None
    depths = {'0-1 km': 1000 * units.m, '0-3 km': 3000 * units.m, '0-6 km': 6000 * units.m}
    for name, depth in depths.items():
        try: params['BWD (nusos)'][name] = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u, v, height=heights, depth=depth)).to('kt').m
        except (ValueError, IndexError): params['BWD (nusos)'][name] = np.nan
    if right_mover is not None:
        try:
            u_storm, v_storm = right_mover; critical_angle = mpcalc.critical_angle(p, u, v, heights, u_storm=u_storm, v_storm=v_storm).to('deg').m
            sr_wind_speed = mpcalc.wind_speed(u - u_storm, v - v_storm).to('kt')
        except (ValueError, IndexError, TypeError): pass

    y_pos = 0.95; x_label = 0.1; x_value = 0.6; y_step = 0.05
    ax_info.text(0.5, y_pos, "Paràmetres", ha='center', weight='bold', fontsize=12); y_pos -= y_step*1.5
    ax_info.text(x_value, y_pos, "BWD\n(nusos)", ha='center'); y_pos -= y_step*1.5
    for key, val in params['BWD (nusos)'].items():
        ax_info.text(x_label, y_pos, key); ax_info.text(x_value, y_pos, f"{val:.0f}" if not np.isnan(val) else "---", ha='center'); y_pos -= y_step
    
    y_pos -= y_step; ax_info.text(0.5, y_pos, "Moviment Tempesta (dir/kts)", ha='center', weight='bold', fontsize=12); y_pos -= y_step*1.5
    if motion:
        for name, vec in motion.items():
            speed = mpcalc.wind_speed(*vec).to('kt').m; direction = mpcalc.wind_direction(*vec).to('deg').m
            ax_info.text(x_label, y_pos, f"{name}:"); ax_info.text(x_value, y_pos, f"{direction:.0f}°/{speed:.0f} kts", ha='left'); y_pos -= y_step
    else: ax_info.text(0.5, y_pos, "Càlcul no disponible", ha='center', fontsize=9, color='gray'); y_pos -= y_step
    ax_info.text(x_label, y_pos, "Angle Crític:"); ax_info.text(x_value, y_pos, f"{critical_angle:.0f}°" if not np.isnan(critical_angle) else '---', ha='left'); y_pos -= y_step

    ax_sr_wind = fig.add_axes([0.62, 0.08, 0.3, 0.25])
    ax_sr_wind.set_title("Vent Relatiu vs. Altura (RM)", fontsize=10);
    if sr_wind_speed is not None:
        ax_sr_wind.plot(sr_wind_speed, heights.to('km').m); ax_sr_wind.set_xlim(0, max(60, sr_wind_speed[~np.isnan(sr_wind_speed)].max().m + 5 if np.any(~np.isnan(sr_wind_speed)) else 60))
        ax_sr_wind.fill_betweenx([0, 2], 40, 60, color='gray', alpha=0.2); ax_sr_wind.fill_betweenx([7, 11], 40, 60, color='gray', alpha=0.2)
    else: ax_sr_wind.text(0.5, 0.5, "No disponible", ha='center', va='center', transform=ax_sr_wind.transAxes, fontsize=9, color='gray')
    ax_sr_wind.set_xlabel("Vent Relatiu (nusos)", fontsize=8); ax_sr_wind.set_ylabel("Altura (km)", fontsize=8); ax_sr_wind.set_ylim(0, 12); ax_sr_wind.grid(True, linestyle='--')
    ax_sr_wind.tick_params(axis='both', which='major', labelsize=8)
    return fig
@st.cache_data(ttl=600)
def carregar_imatge_satelit(url):
    try:
        response = requests.get(f"{url}?ver={int(time.time() // 600)}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        return (response.content, None) if response.status_code == 200 else (None, f"No s'ha pogut carregar la imatge. (Codi: {response.status_code})")
    except Exception as e: return None, "Error de xarxa en carregar la imatge."
def mostrar_imatge_temps_real(tipus):
    if tipus == "Satèl·lit (Europa)": url, caption = "https://modeles20.meteociel.fr/satellite/animsatsandvisirmtgeu.gif", "Satèl·lit Sandvitx (Visible + Infraroig). Font: Meteociel"
    elif tipus == "Satèl·lit (NE Península)":
        now_local = datetime.now(TIMEZONE)
        if 7 <= now_local.hour < 21: url, caption = "https://modeles20.meteociel.fr/satellite/animsatviscolmtgsp.gif", "Satèl·lit Visible (Nord-est). Font: Meteociel"
        else: url, caption = "https://modeles20.meteociel.fr/satellite/animsatirmtgsp.gif", "Satèl·lit Infraroig (Nord-est). Font: Meteociel"
    else: st.error("Tipus d'imatge no reconegut."); return
    image_content, error_msg = carregar_imatge_satelit(url)
    if image_content: st.image(image_content, caption=caption, use_container_width=True)
    else: st.warning(error_msg)

def analitzar_tipus_sondeig(params):
    cape = params.get('CAPE_total', 0); cin = params.get('CIN_total', 0)
    pwat = params.get('PWAT', 0); dcape = params.get('DCAPE', 0)
    if cape > 2000 and cin < -75: return ("SONDEIG CARREGAT (LOADED GUN)", "Potencial de tempestes explosives si es trenca la 'tapa' (CIN). Molt CAPE contingut per una forta inversió. Risc de temps sever si s'allibera la inestabilitat.")
    elif dcape > 1000: return ("SONDEIG EN V INVERTIDA", "Perfil característic d'un alt risc de **rebentades seques** (*dry downbursts*). La presència d'aire sec a nivells mitjans afavoreix forts corrents descendents per evaporació.")
    elif pwat > 40: return ("SONDEIG TROPICAL / SATURAT", "Perfil molt humit en tota la columna. Favorable per a pluges molt eficients i abundants, amb risc de **precipitacions intenses o estacionàries**.")
    elif cape < 100 and cin > -25: return ("SONDEIG ESTABLE", "Atmosfera generalment estable. La convecció és molt improbable. No s'esperen fenòmens de temps sever.")
    else: return ("SONDEIG DE CONVECCIÓ ESTÀNDARD", "Condicions típiques per a la formació de tempestes de tarda. La severitat dependrà d'altres factors com el cisallament del vent (veure hodògraf).")
def analitzar_tipus_hodograf(params):
    bwd = params.get('BWD_0_6km', 0)
    if bwd > 50: return("POTENCIAL DE SUPERCÈL·LULES", "El cisallament del vent és molt fort. Aquest entorn és extremadament favorable per a l'organització de tempestes i la formació de supercèl·lules amb rotació (mesociclons).")
    elif bwd > 35: return("POTENCIAL DE TEMPESTES ORGANITZADES", "El cisallament del vent és significatiu. Les tempestes poden organitzar-se en sistemes multicel·lulars (línies de torbonada, MCS) o fins i tot supercèl·lules aïllades.")
    else: return("POTENCIAL DE TEMPESTES DESORGANITZADES", "El cisallament del vent és feble. Si es formen tempestes, probablement seran de cicle de vida únic, de curta durada i amb menys potencial de severitat.")
def ui_pestanya_ia_final(data_tuple, hourly_index_sel, poble_sel, timestamp_str):
    st.subheader("Assistent Meteo-Col·lega (amb Google Gemini)")
    try: genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except (KeyError, AttributeError): st.error("Falta la GEMINI_API_KEY als secrets de Streamlit."); return
    username = st.session_state.get('username');
    if not username: st.error("Error d'autenticació."); return
    LIMIT_PER_WINDOW = 10; WINDOW_HOURS = 3; rate_limits = load_json_file(RATE_LIMIT_FILE)
    user_limit_data = rate_limits.get(username, {"count": 0, "window_start_time": None}); limit_reached = False
    if user_limit_data.get("window_start_time"):
        start_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc)
        if (datetime.now(pytz.utc) - start_time) > timedelta(hours=WINDOW_HOURS): user_limit_data.update({"count": 0, "window_start_time": None})
    if user_limit_data.get("count", 0) >= LIMIT_PER_WINDOW:
        limit_reached = True; time_left = (datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc) + timedelta(hours=WINDOW_HOURS)) - datetime.now(pytz.utc)
        if time_left.total_seconds() > 0: st.warning(f"**Límit de {LIMIT_PER_WINDOW} preguntes assolit.** Accés renovat en **{format_time_left(time_left)}**.")
        else: user_limit_data.update({"count": 0, "window_start_time": None}); rate_limits[username] = user_limit_data; save_json_file(rate_limits, RATE_LIMIT_FILE); limit_reached = False
    if not limit_reached:
        preguntes_restants = LIMIT_PER_WINDOW - user_limit_data.get("count", 0)
        color = "green" if preguntes_restants > 3 else "orange" if 1 <= preguntes_restants <= 3 else "red"
        st.markdown(f"""<div style="text-align: right; margin-top: -30px; margin-bottom: 10px;"><span style="font-size: 0.9em;">Preguntes restants: <strong style="color: {color}; font-size: 1.1em;">{preguntes_restants}/{LIMIT_PER_WINDOW}</strong></span></div>""", unsafe_allow_html=True)
    if "chat" not in st.session_state:
        model = genai.GenerativeModel('gemini-1.5-flash')
        system_prompt = """Ets 'Meteo-Col·lega', un expert en meteorologia de Catalunya. Ets directe, proper i parles de 'tu'. La teva única missió és donar LA CONCLUSIÓ FINAL.
# REGLA D'OR: NO descriguis les dades. No diguis "el CAPE és X" o "l'hodògraf mostra Y". Això ja ho veu l'usuari. Tu has d'ajuntar totes les peces (mapa, sondeig, hodògraf) i donar el diagnòstic final: què passarà i on. Sigues breu i ves al gra.
# EL TEU PROCÉS MENTAL:
1. **On som?** L'usuari et preguntarà per un poble concret. Centra la teva resposta en aquella zona.
2. **Hi ha disparador a prop?** Mira el mapa. Si hi ha una zona de convergència (línies de colors) a prop del poble, és un SÍ. Si no n'hi ha, és un NO.
3. **Si es dispara, què passarà?** Mira el sondeig i l'hodògraf per saber el potencial.
4. **Dóna la conclusió final:** Ajunta-ho tot en una resposta clara.
# EXEMPLES DE RESPOSTES PERFECTES:
- **(Pregunta per Lleida, amb convergència a prop i bon sondeig):** "Bona tarda! Avui a la teva zona de Ponent ho teniu tot de cara. Hi ha una bona línia de convergència a prop que actuarà de disparador, i el sondeig mostra prou 'benzina' i organització per a tempestes fortes. Compte a la tarda, que es pot posar interessant."
- **(Pregunta per Mataró, sense convergència a prop):** "Què tal! Avui pel Maresme la cosa sembla tranquil·la. El problema és que no teniu cap disparador a prop; les línies de convergència queden molt a l'interior. Encara que el sondeig té potencial, si no hi ha qui encengui la metxa, no passarà gran cosa."
- **(Pregunta per Berga, amb convergència llunyana):** "Ei! Per la teva zona del Berguedà avui sembla que calma. Ara bé, compte a les comarques de Girona! Allà sí que s'està formant una bona línia de convergència. Si vols veure el potencial real d'aquella zona, et recomano que canviïs al sondeig de **Girona** o **Figueres**."
- **(Pregunta per Reus, amb convergència a prop però sondeig molt estable):** "Avui per la teva zona teniu un bon disparador amb aquesta convergència, però el sondeig està molt estable, gairebé no hi ha 'benzina' (CAPE). Així que, tot i la convergència, el més probable és que només es formin alguns núvols sense més conseqüències. Un dia tranquil."
Recorda, l'usuari té accés a aquests pobles: """ + ', '.join(CIUTATS_CATALUNYA.keys())
        missatge_inicial_model = "Ei! Sóc el teu Meteo-Col·lega. Tria un poble, fes-me una pregunta i et dono la conclusió del que pot passar avui."
        st.session_state.chat = model.start_chat(history=[{'role': 'user', 'parts': [system_prompt]}, {'role': 'model', 'parts': [missatge_inicial_model]}]); st.session_state.messages = [{"role": "assistant", "content": missatge_inicial_model}]
    
    st.markdown(f"**Anàlisi per:** `{poble_sel.upper()}` | **Dia:** `{timestamp_str}`")
    nivell_mapa_ia = st.selectbox("Nivell d'anàlisi del mapa:", [1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa", key="ia_level_selector_chat_final", disabled=limit_reached)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
        
    if prompt_usuari := st.chat_input("Què vols saber sobre el temps avui?", disabled=limit_reached):
        st.session_state.messages.append({"role": "user", "content": prompt_usuari})
        with st.chat_message("user"): st.markdown(prompt_usuari)
        with st.chat_message("assistant"):
            with st.spinner("Connectant totes les peces..."):
                if user_limit_data.get("window_start_time") is None: user_limit_data["window_start_time"] = datetime.now(pytz.utc).timestamp()
                user_limit_data["count"] += 1; rate_limits[username] = user_limit_data; save_json_file(rate_limits, RATE_LIMIT_FILE)
                
                map_data_ia, error_map_ia = carregar_dades_mapa(nivell_mapa_ia, hourly_index_sel)
                if error_map_ia: st.error(f"Error en carregar dades del mapa: {error_map_ia}"); return
                fig_mapa = crear_mapa_forecast_combinat(map_data_ia['lons'], map_data_ia['lats'], map_data_ia['speed_data'], map_data_ia['dir_data'], map_data_ia['dewpoint_data'], nivell_mapa_ia, timestamp_str, MAP_EXTENT)
                buf_mapa = io.BytesIO(); fig_mapa.savefig(buf_mapa, format='png', dpi=150, bbox_inches='tight'); buf_mapa.seek(0); img_mapa = Image.open(buf_mapa); plt.close(fig_mapa)
                
                contingut_per_ia = [img_mapa]; resum_sondeig_text = "No s'han pogut carregar les dades del sondeig."
                
                if data_tuple: 
                    sounding_data, params_calculats = data_tuple; p, T, Td, u, v, heights = sounding_data
                    fig_skewt = crear_skewt(p, T, Td, u, v, params_calculats, f"Sondeig per a {poble_sel}")
                    buf_skewt = io.BytesIO(); fig_skewt.savefig(buf_skewt, format='png', dpi=150); buf_skewt.seek(0); img_skewt = Image.open(buf_skewt); plt.close(fig_skewt); contingut_per_ia.append(img_skewt)
                    
                    fig_hodo = crear_hodograf_avancat(p, u, v, heights, f"Hodògraf Avançat - {poble_sel}")
                    buf_hodo = io.BytesIO(); fig_hodo.savefig(buf_hodo, format='png', dpi=150); buf_hodo.seek(0); img_hodo = Image.open(buf_hodo); plt.close(fig_hodo); contingut_per_ia.append(img_hodo)

                    titol_s, _ = analitzar_tipus_sondeig(params_calculats)
                    titol_h, _ = analitzar_tipus_hodograf(params_calculats)
                    resum_sondeig_text = (f"Poble seleccionat per l'usuari: {poble_sel}. "
                                          f"Anàlisi automàtica del sondeig: {titol_s}. "
                                          f"Anàlisi automàtica de l'hodògraf: {titol_h}.")
                
                prompt_context = f"INFO ADDICIONAL PER A TU:\n- {resum_sondeig_text}\n\nPREGUNTA DE L'USUARI: '{prompt_usuari}'"
                contingut_per_ia.insert(0, prompt_context)
                
                try:
                    resposta = st.session_state.chat.send_message(contingut_per_ia)
                    full_response = resposta.text
                except Exception as e:
                    full_response = f"Vaja, hi ha hagut un error contactant la IA: {e}"
                    if "429" in str(e): full_response = "**Ep, hem superat el límit de consultes a l'API de Google per avui.**"
                st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
def ui_pestanya_xat(chat_history):
    st.subheader("Xat en Línia per a Usuaris"); col1, col2 = st.columns([0.7, 0.3]);
    with col1: st.caption("Els missatges s'esborren automàticament després d'una hora.")
    with col2:
        if st.button("🔄 Refrescar", use_container_width=True): st.rerun()
    if chat_history: st.session_state.last_seen_timestamp = chat_history[-1]['timestamp']
    with st.container(height=400):
        for msg in chat_history:
            with st.chat_message(name=msg['username']):
                if msg['type'] == 'text': st.markdown(msg['content'])
                elif msg['type'] == 'image':
                    try: st.image(base64.b64decode(msg['content']))
                    except Exception: st.error("No s'ha pogut carregar la imatge.")
    prompt = st.chat_input("Escriu el teu missatge..."); pujada_img = st.file_uploader("O arrossega una imatge", type=['png', 'jpg', 'jpeg'], key="chat_uploader")
    if prompt or pujada_img:
        with st.spinner("Enviant..."):
            username = st.session_state.get("username", "Anònim"); current_history = load_and_clean_chat_history()
            if pujada_img and pujada_img.file_id != st.session_state.get('last_uploaded_id'):
                b64_string = base64.b64encode(pujada_img.getvalue()).decode('utf-8')
                current_history.append({"username": username, "timestamp": datetime.now(pytz.utc).timestamp(), "type": "image", "content": b64_string})
                st.session_state['last_uploaded_id'] = pujada_img.file_id
            if prompt: current_history.append({"username": username, "timestamp": datetime.now(pytz.utc).timestamp(), "type": "text", "content": prompt})
            save_json_file(current_history, CHAT_FILE); st.rerun()

def ui_capcalera_selectors(ciutats_a_mostrar, info_msg=None):
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False); col_text, col_button = st.columns([0.85, 0.15])
    with col_text:
        if not is_guest: st.markdown(f"Benvingut/da, **{st.session_state.get('username')}**!")
    with col_button:
        if st.button("Sortir" if is_guest else "Tanca Sessió"):
            for key in ['logged_in', 'username', 'guest_mode', 'chat', 'messages', 'last_uploaded_id', 'last_seen_timestamp']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            if is_guest: st.info(f"ℹ️ **Mode Convidat:** {info_msg}")
            poble_actual = st.session_state.get('poble_selector'); sorted_ciutats = sorted(ciutats_a_mostrar.keys())
            index_poble = sorted_ciutats.index(poble_actual) if poble_actual in sorted_ciutats else 0
            st.selectbox("Població de referència:", sorted_ciutats, key="poble_selector", index=index_poble)
        now_local = datetime.now(TIMEZONE)
        with col2: st.selectbox("Dia del pronòstic:", ("Avui",) if is_guest else ("Avui", "Demà"), key="dia_selector", disabled=is_guest, index=0)
        with col3:
            st.selectbox("Hora del pronòstic (Hora Local):", (f"{now_local.hour:02d}:00h",) if is_guest else [f"{h:02d}:00h" for h in range(24)], key="hora_selector", disabled=is_guest, index=0 if is_guest else now_local.hour)
def ui_explicacio_alertes():
    with st.expander("Com interpretar el mapa de convergència?"):
        st.markdown("""Les zones acolorides marquen àrees de **convergència d'humitat**, que actuen com a **disparadors** potencials de tempestes.
- **Què són?** Àrees on el vent en nivells baixos força l'aire humit a ajuntar-se i ascendir.
- **Colors:** <span style="color:#ffc107; font-weight:bold;">GROC (>15):</span> Moderada; <span style="color:#ff9800; font-weight:bold;">TARONJA (>25):</span> Forta; <span style="color:#f44336; font-weight:bold;">VERMELL (>40):</span> Molt forta.""", unsafe_allow_html=True)
def ui_info_desenvolupament_tempesta():
    with st.expander("⏳ De la Convergència a la Tempesta: Quant triga?", expanded=True):
        st.markdown("""Un cop s'activa un nucli de convergència, el temps estimat per al desenvolupament d'un Cumulonimbus sol ser d'entre **20 a 60 minuts**.
- **Més ràpid (< 30 min):** Convergència intensa, **CAPE alt** i **CIN baix**.
- **Més lent (> 45 min):** Convergència feble, **CIN alt** o aire sec a nivells mitjans.""")
def ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple):
    is_guest = st.session_state.get('guest_mode', False)
    _, contingut_principal, _ = st.columns([0.05, 0.9, 0.05])
    with contingut_principal:
        col_map_1, col_map_2 = st.columns([0.7, 0.3], gap="large")
        with col_map_1:
            col_capa, col_zoom = st.columns(2)
            with col_capa:
                map_options = {"Anàlisi de Vent i Convergència": "forecast_estatic", "Vent a 700hPa": "vent_700", "Vent a 300hPa": "vent_300"}
                mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
            with col_zoom: zoom_sel = st.selectbox("Nivell de Zoom:", options=list(MAP_ZOOM_LEVELS.keys()))
            selected_extent = MAP_ZOOM_LEVELS[zoom_sel]; map_key = map_options[mapa_sel]
            if map_key == "forecast_estatic":
                nivell_sel = 925 if is_guest else st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa")
                if is_guest: st.info("ℹ️ L'anàlisi de vent i convergència està fixada a **925 hPa**.")
                with progress_placeholder.container():
                    progress_bar = st.progress(0, text="Carregant dades del model...")
                    map_data, error_map = carregar_dades_mapa(nivell_sel, hourly_index_sel)
                    if not error_map: progress_bar.progress(50, text="Generant visualització...")
                if error_map: st.error(f"Error en carregar el mapa: {error_map}"); progress_placeholder.empty()
                elif map_data:
                    fig = crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str, selected_extent)
                    st.pyplot(fig); plt.close(fig); ui_explicacio_alertes()
                    with progress_placeholder.container(): progress_bar.progress(100, text="Completat!"); time.sleep(1); progress_placeholder.empty()
            elif map_key in ["vent_700", "vent_300"]:
                nivell = 700 if map_key == "vent_700" else 300
                variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
                map_data, error_map = carregar_dades_mapa_base(variables, hourly_index_sel)
                if error_map: st.error(f"Error en carregar el mapa: {error_map}")
                elif map_data: 
                    fig = crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str, selected_extent)
                    st.pyplot(fig); plt.close(fig)
        with col_map_2:
            st.subheader("Imatges en Temps Real"); tab_europa, tab_ne = st.tabs(["Europa", "NE Peninsula"])
            with tab_europa: mostrar_imatge_temps_real("Satèl·lit (Europa)")
            with tab_ne: mostrar_imatge_temps_real("Satèl·lit (NE Península)")
            st.markdown("---"); ui_info_desenvolupament_tempesta()
def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        _, contingut_principal, _ = st.columns([0.05, 0.9, 0.05])
        with contingut_principal:
            st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
            p, T, Td, u, v, heights = sounding_data
            
            titol_s, _ = analitzar_tipus_sondeig(params_calculats)
            titol_h, _ = analitzar_tipus_hodograf(params_calculats)

            col1, col2 = st.columns(2)
            with col1:
                fig_skewt = crear_skewt(p, T, Td, u, v, params_calculats, f"Sondeig Vertical - {poble_sel}\n{titol_s}")
                st.pyplot(fig_skewt, use_container_width=True); plt.close(fig_skewt)
            with col2:
                fig_hodo = crear_hodograf_avancat(p, u, v, heights, f"Hodògraf Avançat - {poble_sel}\n{titol_h}")
                st.pyplot(fig_hodo, use_container_width=True); plt.close(fig_hodo)

            with st.expander("❔ Com interpretar els paràmetres i gràfics"):
                st.markdown("""
                **Paràmetres del Sondeig:**
                - **CAPE (SB/MU/ML):** Energia disponible per a una tempesta. >1500 J/kg es considera alt.
                - **CIN:** "Tapa" que impedeix la convecció. Valors més negatius que -50 J/kg indiquen una tapa forta.
                - **LCL/LFC/EL:** Altures (en metres sobre el terra) de la base del núvol, inici de l'ascens lliure i sostre teòric de la tempesta.
                - **KI / TT:** Índexs clàssics d'inestabilitat. Com més alts, més potencial de tempesta.
                - **PWAT (Aigua Precipitable):** Quantitat total de vapor d'aigua a la columna. Valors > 40 mm indiquen un alt contingut d'humitat, favorable per a pluges intenses.
                - **DCAPE:** Energia potencial per a corrents descendents forts (rebentades). Valors > 1000 J/kg són un avís de risc.

                **Hodògraf:**
                - **Forma:** Una corba pronunciada indica cisallament direccional, favorable per a tempestes organitzades.
                - **BWD (Cisallament):** Valors > 40 nusos (0-6 km) afavoreixen l'organització de les tempestes.
                - **Vent Relatiu vs. Altura:** Mostra com de fort és el vent relatiu a la tempesta a diferents altures. Valors alts a nivells baixos afavoreixen la formació de tornados.
                """)
    else: st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")
def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades AROME via Open-Meteo | Imatges via Meteociel | IA per Google Gemini.</p>", unsafe_allow_html=True)

def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'guest_mode' not in st.session_state: st.session_state['guest_mode'] = False
    if not st.session_state['logged_in'] and not st.session_state['guest_mode']: show_login_page()
    else:
        is_guest = st.session_state.get('guest_mode', False); now_local = datetime.now(TIMEZONE)
        hora_sel_str = f"{now_local.hour:02d}:00h" if is_guest else st.session_state.get('hora_selector', f"{now_local.hour:02d}:00h")
        dia_sel_str = "Avui" if is_guest else st.session_state.get('dia_selector', "Avui")
        target_date = now_local.date() + timedelta(days=1) if dia_sel_str == "Demà" else now_local.date()
        local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_str.split(':')[0])))
        start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        hourly_index_sel = max(0, int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600))
        ciutats_per_selector, info_msg = obtenir_ciutats_actives(hourly_index_sel) if is_guest else (CIUTATS_CATALUNYA, None)
        ui_capcalera_selectors(ciutats_per_selector, info_msg)
        poble_sel = st.session_state.poble_selector
        if poble_sel not in ciutats_per_selector: st.session_state.poble_selector = sorted(ciutats_per_selector.keys())[0]; st.rerun()
        dia_sel, hora_sel = st.session_state.dia_selector, st.session_state.hora_selector
        timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
        lat_sel, lon_sel = ciutats_per_selector[poble_sel]['lat'], ciutats_per_selector[poble_sel]['lon']
        data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
        if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
        st.markdown("---"); global progress_placeholder; progress_placeholder = st.empty()
        if is_guest:
            tab_mapes, tab_vertical = st.tabs(["Anàlisi de Mapes", "Anàlisi Vertical"])
            with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
            with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
        else:
            current_selection = f"{poble_sel}-{dia_sel}-{hora_sel}"
            if current_selection != st.session_state.get('last_selection'):
                if 'messages' in st.session_state: del st.session_state.messages
                if 'chat' in st.session_state: del st.session_state.chat
                st.session_state.last_selection = current_selection
            chat_history = load_and_clean_chat_history()
            if 'last_seen_timestamp' not in st.session_state: st.session_state.last_seen_timestamp = chat_history[-1]['timestamp'] if chat_history else 0
            unread_count = count_unread_messages(chat_history)
            chat_tab_label = f"💬 Xat ({unread_count})" if unread_count > 0 else "💬 Xat"
            tab_ia, tab_xat, tab_mapes, tab_vertical = st.tabs(["Assistent MeteoIA", chat_tab_label, "Anàlisi de Mapes", "Anàlisi Vertical"])
            with tab_ia: ui_pestanya_ia_final(data_tuple, hourly_index_sel, poble_sel, timestamp_str)
            with tab_xat: ui_pestanya_xat(chat_history)
            with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
            with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
        ui_peu_de_pagina()
if __name__ == "__main__": main()
