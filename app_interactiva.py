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
CIUTATS_CONVIDAT = {
    'Barcelona': CIUTATS_CATALUNYA['Barcelona'], 'Girona': CIUTATS_CATALUNYA['Girona'],
    'Lleida': CIUTATS_CATALUNYA['Lleida'], 'Tarragona': CIUTATS_CATALUNYA['Tarragona']
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
USERS_FILE = 'users.json'
RATE_LIMIT_FILE = 'rate_limits.json'
CHAT_FILE = 'chat_history.json'

MAP_ZOOM_LEVELS = {
    'Catalunya (Complet)': MAP_EXTENT,
    'Nord-est (Girona)': [1.8, 3.4, 41.7, 42.6],
    'Sud (Tarragona i Ebre)': [0.2, 1.8, 40.5, 41.4],
    'Ponent i Pirineu (Lleida)': [0.4, 1.9, 41.4, 42.6],
    'Àrea Metropolitana (BCN)': [1.7, 2.7, 41.2, 41.8]
}

# --- 0.1 FUNCIONS D'AUTENTICACIÓ, LÍMITS I XAT ---
def get_hashed_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
        if len(cleaned_history) < len(history):
            save_json_file(cleaned_history, CHAT_FILE)
        return cleaned_history
    except (json.JSONDecodeError, FileNotFoundError): return []

def count_unread_messages(history):
    last_seen = st.session_state.get('last_seen_timestamp', 0)
    current_user = st.session_state.get('username')
    new_messages_count = sum(1 for msg in history if msg['timestamp'] > last_seen and msg['username'] != current_user)
    return new_messages_count

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
            username = st.text_input("Nom d'usuari")
            password = st.text_input("Contrasenya", type="password")
            if st.form_submit_button("Entra"):
                users = load_json_file(USERS_FILE)
                if username in users and users[username] == get_hashed_password(password):
                    st.session_state.update({'logged_in': True, 'username': username, 'guest_mode': False})
                    st.rerun()
                else: st.error("Nom d'usuari o contrasenya incorrectes.")
    elif selected == "Registra't":
        st.subheader("Crea un nou compte")
        with st.form("register_form"):
            new_username = st.text_input("Tria un nom d'usuari")
            new_password = st.text_input("Tria una contrasenya", type="password")
            if st.form_submit_button("Registra'm"):
                users = load_json_file(USERS_FILE)
                if not new_username or not new_password: st.error("El nom d'usuari i la contrasenya no poden estar buits.")
                elif new_username in users: st.error("Aquest nom d'usuari ja existeix.")
                elif len(new_password) < 6: st.error("La contrasenya ha de tenir com a mínim 6 caràcters.")
                else:
                    users[new_username] = get_hashed_password(new_password)
                    save_json_file(users, USERS_FILE)
                    st.success("Compte creat amb èxit! Ara pots iniciar sessió.")
    st.divider()
    if st.button("Entrar com a Convidat", use_container_width=True, type="secondary"):
        st.session_state.update({'guest_mode': True, 'logged_in': False})
        st.rerun()

# --- 1. FUNCIONS D'OBTENCIÓ DE DADES ---
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

@st.cache_data(ttl=3600)
def obtenir_ciutats_actives(hourly_index):
    """
    Identifica nuclis de convergència discrets a 925hPa i retorna la ciutat més propera a cada nucli.
    """
    nivell = 925
    map_data, error_map = carregar_dades_mapa(nivell, hourly_index)
    if error_map or not map_data:
        return CIUTATS_CONVIDAT, "No s'ha pogut determinar les zones de convergència. Mostrant capitals per defecte."

    try:
        lons, lats = map_data['lons'], map_data['lats']
        speed_data, dir_data, dewpoint_data = map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data']

        grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100))
        grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')

        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        dudx = mpcalc.first_derivative(grid_u * units('m/s'), delta=dx, axis=1)
        dvdy = mpcalc.first_derivative(grid_v * units('m/s'), delta=dy, axis=0)
        convergence_scaled = - (dudx + dvdy).to('1/s').magnitude * 1e5

        CONVERGENCE_THRESHOLD = 15
        DEWPOINT_THRESHOLD = 12
        convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD, convergence_scaled, 0)
        
        points_with_conv = np.argwhere(convergence_in_humid_areas >= CONVERGENCE_THRESHOLD)
        if points_with_conv.size == 0:
            return CIUTATS_CONVIDAT, "No s'han detectat zones de convergència significatives. Mostrant capitals."

        conv_values = [convergence_in_humid_areas[y, x] for y, x in points_with_conv]
        conv_coords = [(grid_lat[y, x], grid_lon[y, x]) for y, x in points_with_conv]

        sorted_points = sorted(zip(conv_values, conv_coords), key=lambda item: item[0], reverse=True)
        
        zone_centers = []
        ZONE_RADIUS = 0.75
        
        for value, (lat, lon) in sorted_points:
            is_new_zone = True
            for center_lat, center_lon in zone_centers:
                dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
                if dist < ZONE_RADIUS:
                    is_new_zone = False
                    break
            if is_new_zone:
                zone_centers.append((lat, lon))

        if not zone_centers:
             return CIUTATS_CONVIDAT, "No s'han pogut identificar nuclis de convergència. Mostrant capitals."

        ciutat_noms = list(CIUTATS_CATALUNYA.keys())
        ciutat_coords = np.array([[v['lat'], v['lon']] for v in CIUTATS_CATALUNYA.values()])
        
        closest_cities_names = set()
        for zone_lat, zone_lon in zone_centers:
            zone_coord = np.array([[zone_lat, zone_lon]])
            dist_matrix = cdist(zone_coord, ciutat_coords)
            closest_city_index = np.argmin(dist_matrix)
            closest_cities_names.add(ciutat_noms[closest_city_index])
            
        if not closest_cities_names:
            return CIUTATS_CONVIDAT, "No s'ha trobat cap ciutat propera als nuclis de convergència."

        ciutats_actives = {name: CIUTATS_CATALUNYA[name] for name in closest_cities_names}
        
        return ciutats_actives, f"Selecció de {len(ciutats_actives)} poblacions properes a nuclis d'activitat."

    except Exception as e:
        return CIUTATS_CONVIDAT, f"Error calculant zones actives: {e}. Mostrant capitals."

# --- 2. FUNCIONS DE VISUALITZACIÓ ---
# Substitueix la funció original
def crear_mapa_base(map_extent):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=90, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(map_extent, crs=ccrs.PlateCarree()) # <--- Canvi clau aquí
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    return fig, ax

def crear_mapa_forecast_combinat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent) # <--- Passa l'extent a la funció base
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 400), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 400))
    grid_speed, grid_dewpoint = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    
    # ... (la resta de la funció és idèntica, no cal copiar-la de nou, només assegura't que la definició inclou 'map_extent')
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
    
    if nivell >= 950: DEWPOINT_THRESHOLD = 14
    elif nivell >= 925: DEWPOINT_THRESHOLD = 12
    else: DEWPOINT_THRESHOLD = 7
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD, convergence_scaled, 0)

    fill_levels = [15, 25, 40, 150]
    fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [15, 25, 40]
    line_colors = ['#e65100', '#bf360c', '#b71c1c']
    line_styles = ['--', '--', '-']
    line_widths = [1, 1.2, 1.5]

    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, 
                levels=fill_levels, colors=fill_colors, alpha=0.4, 
                zorder=5, transform=ccrs.PlateCarree())

    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas,
                          levels=line_levels, colors=line_colors,
                          linestyles=line_styles, linewidths=line_widths,
                          zorder=6, transform=ccrs.PlateCarree())

    labels = ax.clabel(contours, inline=True, fontsize=6, fmt='%1.0f')
    for label in labels:
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.6))
        
    ax.set_title(f"Anàlisi de Vent i Nuclis de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig
    
def crear_mapa_vents(lons, lats, speed_data, dir_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent) # <--- Passa l'extent a la funció base
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    
    # ... (la resta de la funció és idèntica)
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

@st.cache_data(ttl=600)
def carregar_imatge_satelit(url):
    try:
        url_amb_timestamp = f"{url}?ver={int(time.time() // 600)}"
        response = requests.get(url_amb_timestamp, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code == 200:
            return response.content, None
        else:
            return None, f"No s'ha pogut carregar la imatge. (Codi: {response.status_code})"
    except Exception as e:
        return None, f"Error de xarxa en carregar la imatge."

def mostrar_imatge_temps_real(tipus):
    url, caption = None, ""
    if tipus == "Satèl·lit (Europa)":
        url = "https://modeles20.meteociel.fr/satellite/animsatsandvisirmtgeu.gif"
        caption = "Satèl·lit Sandvitx (Visible + Infraroig). Font: Meteociel"
    elif tipus == "Satèl·lit (NE Península)":
        now_local = datetime.now(TIMEZONE)
        if 7 <= now_local.hour < 21:
            url = "https://modeles20.meteociel.fr/satellite/animsatviscolmtgsp.gif"
            caption = "Satèl·lit Visible (Nord-est). Font: Meteociel"
        else:
            url = "https://modeles20.meteociel.fr/satellite/animsatirmtgsp.gif"
            caption = "Satèl·lit Infraroig (Nord-est). Font: Meteociel"
    if url:
        image_content, error_msg = carregar_imatge_satelit(url)
        if image_content:
            st.image(image_content, caption=caption, use_container_width=True)
        else:
            st.warning(error_msg)
    else:
        st.error("Tipus d'imatge no reconegut.")

# --- 3. FUNCIONS PER A PESTANYES ---
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





def ui_pestanya_ia_final(data_tuple, hourly_index_sel, poble_sel, timestamp_str):
    """
    Versió final de l'assistent d'IA ("Meteo-Col·lega" Guia).
    - Personalitat: Col·lega directe i amigable.
    - Enfocament: Detecta fenòmens al mapa i GUIA l'usuari cap al sondeig de la població més rellevant.
    - Comunicació: Dóna un diagnòstic inicial breu i una recomanació clara.
    """
    st.subheader("Assistent Meteo-Col·lega (amb Google Gemini)")

    # --- Configuració de l'API de Gemini ---
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=GEMINI_API_KEY)
    except (KeyError, AttributeError):
        st.error("No s'ha pogut configurar l'API de Gemini. Assegura't que la GEMINI_API_KEY està als secrets de Streamlit.")
        return

    # --- Gestió d'autenticació i límits d'ús (el teu codi original) ---
    username = st.session_state.get('username')
    if not username:
        st.error("Error d'autenticació. Si us plau, torna a iniciar sessió.")
        return
        
    LIMIT_PER_WINDOW = 10; WINDOW_HOURS = 3
    rate_limits = load_json_file(RATE_LIMIT_FILE)
    user_limit_data = rate_limits.get(username, {"count": 0, "window_start_time": None})
    
    limit_reached = False
    if user_limit_data.get("window_start_time"):
        start_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc)
        if (datetime.now(pytz.utc) - start_time) > timedelta(hours=WINDOW_HOURS):
            user_limit_data.update({"count": 0, "window_start_time": None})
    if user_limit_data.get("count", 0) >= LIMIT_PER_WINDOW:
        limit_reached = True
        start_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc)
        time_left = (start_time + timedelta(hours=WINDOW_HOURS)) - datetime.now(pytz.utc)
        if time_left.total_seconds() > 0:
            st.warning(f"**Has arribat al límit de {LIMIT_PER_WINDOW} preguntes.** El teu accés es renovarà en **{format_time_left(time_left)}**.")
        else: 
            user_limit_data.update({"count": 0, "window_start_time": None})
            rate_limits[username] = user_limit_data; save_json_file(rate_limits, RATE_LIMIT_FILE)
            limit_reached = False
            
    if not limit_reached:
        preguntes_restants = LIMIT_PER_WINDOW - user_limit_data.get("count", 0)
        color = "green" if preguntes_restants > 3 else "orange" if 1 <= preguntes_restants <= 3 else "red"
        st.markdown(f"""<div style="text-align: right; margin-top: -30px; margin-bottom: 10px;">
            <span style="font-size: 0.9em;">Preguntes restants: <strong style="color: {color}; font-size: 1.1em;">{preguntes_restants}/{LIMIT_PER_WINDOW}</strong></span>
        </div>""", unsafe_allow_html=True)

    # --- Definició del System Prompt (Cervell de la IA) ---
    if "chat" not in st.session_state:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # --- NOU SYSTEM PROMPT MILLORAT ---
        system_prompt = """
Ets Meteo-Col·lega, un expert en meteorologia a Catalunya. Ets directe, planera i parles com si estiguéssim analitzant mapes junts. Respostes curtes i al gra.

# LA TEVA MISSIÓ: SER UN GUIA
La teva feina és fer dues coses en ordre:
1.  **Fer un anàlisi visual RÀPID del mapa** que et donen, buscant convergències de vent (on els numeros i lineas  intercalsdes ).
2.  **DIRIGIR l'usuari cap al SONDEIG correcte** perquè pugui analitzar la situació a fons.

# GUIA DE SONDEJOS (MOLT IMPORTANT)
Només hi ha sondejos disponibles per a 
Amposta
Balaguer
Banyoles
Barcelona
Berga
Cervera
El Pont de Suert
El Vendrell
Falset
Figueres
Gandesa
Girona
Granollers
Granollers
Igualada
La Bisbal d'Empordà
La Seu d'Urgell
Les Borges Blanques
Lleida
Manresa
Mataró
Moià
Mollerussa
Montblanc
Móra d'Ebre
Olot
Prats de Lluçanès
Puigcerdà
Reus
Ripoll
Sant Feliu de Llobregat
Santa Coloma de Farners
Solsona
Sort
Tarragona
Terrassa
Tortosa
Tremp
Tàrrega
Valls
Vic
Vielha
Vilafranca del Penedès

. La teva feina és relacionar el que veus al mapa amb la capital més propera o representativa. Fes servir aquesta guia:
- Si veus moviment a **Ponent, Pirineu/Prepirineu de Lleida**: Recomana el sondeig de **Lleida**.
- Si veus moviment a la **Costa Central, Prelitoral Central, Catalunya Central, Pirineu Oriental**: Recomana el sondeig de **Barcelona**.
- Si veus moviment a la **Costa Brava, Interior/Nord-est de Girona**: Recomana el sondeig de **Girona**.
- Si veus moviment a la **Costa Daurada, Terres de l'Ebre, sud de Ponent**: Recomana el sondeig de **Tarragona**.

**REGLA D'OR:** El teu objectiu final NO és donar el pronòstic complet, sinó dir-li a l'usuari "Ei, mira què passa aquí. Per saber-ne més, ves a mirar el sondeig de [Capital]".

# EL TEU PROCÉS DE RESPOSTA

**PAS 1: Mira el mapa. Hi ha alguna cosa que cridi l'atenció?**

**PAS 2: SI NO VEUS RES:** Sigues sincer i breu.
- **Exemple:** "Doncs ara mateix, el mapa està força tranquil. No veig cap focus clar que mereixi una anàlisi més profunda."

**PAS 3: SI VEUS UNA CONVERGÈNCIA O PATRÓ INTERESSANT:**
    a. **Localitza-la** de manera general (Ponent, Costa Central, etc.).
    b. **Fes un comentari inicial breu** sobre el que veus.
    c. **Recomana el sondeig** de la capital corresponent de manera clara.

    - **Exemple 1 (Convergència a Ponent):** "Uep! S'està formant una línia de convergència molt maca a Ponent. Té bona pinta. Per veure si això té 'benzina' per disparar tempestes, et recomano que canviïs al sondeig de **Lleida** i m'ho tornis a preguntar."
    - **Exemple 2 (Marinada a la costa):** "S'aprecia com la marinada està entrant amb ganes per la costa de Barcelona. A vegades, això sol ja porta ruixats. El millor és que facis una ullada al sondeig de **Barcelona** per analitzar bé la capa baixa."
    - **Exemple 3 (Vent del nord a Girona):** "Compte al nord-est. Es veu una entrada de vent de nord (tramuntana) que podria deixar l'ambient mogut. Per veure l'estabilitat real, el sondeig de **Girona** serà el més útil."
"""
        missatge_inicial_model = "Ei! Sóc el teu Meteo-Col·lega. Fes-me una pregunta sobre el mapa i et diré quina zona cal vigilar i quin sondeig analitzar a fons."
        st.session_state.chat = model.start_chat(history=[
            {'role': 'user', 'parts': [system_prompt]},
            {'role': 'model', 'parts': [missatge_inicial_model]}
        ])
        st.session_state.messages = [{"role": "assistant", "content": missatge_inicial_model}]

    # --- Interfície del Xat ---
    st.markdown(f"**Anàlisi per:** `{poble_sel.upper()}` | **Dia:** `{timestamp_str}`")
    nivell_mapa_ia = st.selectbox("Nivell d'anàlisi del mapa:", [1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa", key="ia_level_selector_chat_final", disabled=limit_reached)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_usuari := st.chat_input("Pregunta'm sobre el mapa!", disabled=limit_reached):
        st.session_state.messages.append({"role": "user", "content": prompt_usuari})
        with st.chat_message("user"):
            st.markdown(prompt_usuari)

        with st.chat_message("assistant"):
            with st.spinner("Interpretant el mapa..."):
                # Actualització del comptador de límits
                if user_limit_data.get("window_start_time") is None:
                    user_limit_data["window_start_time"] = datetime.now(pytz.utc).timestamp()
                user_limit_data["count"] += 1
                rate_limits[username] = user_limit_data; save_json_file(rate_limits, RATE_LIMIT_FILE)

                # Generació de la imatge i dades per a la IA
                map_data_ia, error_map_ia = carregar_dades_mapa(nivell_mapa_ia, hourly_index_sel)
                if error_map_ia:
                    st.error(f"Error en carregar dades del mapa: {error_map_ia}"); return
                
                fig_mapa = crear_mapa_forecast_combinat(map_data_ia['lons'], map_data_ia['lats'], map_data_ia['speed_data'], map_data_ia['dir_data'], map_data_ia['dewpoint_data'], nivell_mapa_ia, timestamp_str)
                
                buf = io.BytesIO(); fig_mapa.savefig(buf, format='png', dpi=150, bbox_inches='tight'); buf.seek(0); img_mapa = Image.open(buf); plt.close(fig_mapa)

                resum_sondeig = "No hi ha dades de sondeig."
                if data_tuple: 
                    _, params_calculats = data_tuple
                    resum_sondeig = (f"- CAPE: {params_calculats.get('CAPE', 0):.0f} J/kg, "
                                     f"CIN: {params_calculats.get('CIN', 0):.0f} J/kg, "
                                     f"Shear 0-6km: {params_calculats.get('Shear 0-6km', 'N/A')} nusos.")

                prompt_context = f"DADES ADDICIONALS:\n- Localització de referència de l'usuari: {poble_sel}\n- Sondeig (per al teu context intern): {resum_sondeig}\n\nPREGUNTA DE L'USUARI: '{prompt_usuari}'"
                
                try:
                    resposta = st.session_state.chat.send_message([prompt_context, img_mapa])
                    full_response = resposta.text
                except Exception as e:
                    full_response = f"Vaja, hi ha hagut un error contactant la IA: {e}"
                    if "429" in str(e):
                        full_response = "**Ep, hem superat el límit de consultes a l'API de Google per avui.**"
                
                st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()


def ui_pestanya_xat(chat_history):
    st.subheader("Xat en Línia per a Usuaris")
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.caption("Els missatges s'esborren automàticament després d'una hora.")
    with col2:
        if st.button("🔄 Refrescar per veure nous missatges", use_container_width=True):
            st.rerun()
    if chat_history:
        st.session_state.last_seen_timestamp = chat_history[-1]['timestamp']
    with st.container(height=400):
        for msg in chat_history:
            with st.chat_message(name=msg['username']):
                if msg['type'] == 'text': st.markdown(msg['content'])
                elif msg['type'] == 'image':
                    try:
                        img_bytes = base64.b64decode(msg['content'])
                        st.image(img_bytes)
                    except Exception: st.error("No s'ha pogut carregar una imatge.")
    prompt = st.chat_input("Escriu el teu missatge...")
    pujada_img = st.file_uploader("O arrossega una imatge aquí", type=['png', 'jpg', 'jpeg'], key="chat_uploader")
    if prompt or pujada_img:
        with st.spinner("Enviant..."):
            username = st.session_state.get("username", "Anònim")
            current_history = load_and_clean_chat_history()
            if pujada_img and pujada_img.file_id != st.session_state.get('last_uploaded_id'):
                img_bytes = pujada_img.getvalue()
                b64_string = base64.b64encode(img_bytes).decode('utf-8')
                current_history.append({"username": username, "timestamp": datetime.now(pytz.utc).timestamp(), "type": "image", "content": b64_string})
                st.session_state['last_uploaded_id'] = pujada_img.file_id
            if prompt:
                current_history.append({"username": username, "timestamp": datetime.now(pytz.utc).timestamp(), "type": "text", "content": prompt})
            save_json_file(current_history, CHAT_FILE)
        st.rerun()

# --- 4. LÒGICA DE LA INTERFÍCIE D'USUARI ---
def ui_capcalera_selectors(ciutats_a_mostrar, info_msg=None):
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)
    col_text, col_button = st.columns([0.85, 0.15])
    
    with col_text:
        if not is_guest:
            st.markdown(f"Benvingut/da, **{st.session_state.get('username')}**!")

    with col_button:
        button_text = "Sortir" if is_guest else "Tanca Sessió"
        if st.button(button_text):
            for key in ['logged_in', 'username', 'guest_mode', 'chat', 'messages', 'last_uploaded_id', 'last_seen_timestamp']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
            
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            if is_guest:
                st.info(f"ℹ️ **Mode Convidat:** {info_msg}")

            poble_actual = st.session_state.get('poble_selector')
            sorted_ciutats = sorted(ciutats_a_mostrar.keys())
            index_poble = 0
            if poble_actual in sorted_ciutats:
                index_poble = sorted_ciutats.index(poble_actual)
            
            st.selectbox("Població de referència:", sorted_ciutats, key="poble_selector", index=index_poble)

        now_local = datetime.now(TIMEZONE)
        with col2:
            st.selectbox("Dia del pronòstic:", ("Avui",) if is_guest else ("Avui", "Demà"), key="dia_selector", disabled=is_guest, index=0)
        with col3:
            default_hour_index = now_local.hour
            st.selectbox("Hora del pronòstic (Hora Local):", (f"{now_local.hour:02d}:00h",) if is_guest else [f"{h:02d}:00h" for h in range(24)], key="hora_selector", disabled=is_guest, index=0 if is_guest else default_hour_index)


def ui_explicacio_alertes():
    with st.expander("Com interpretar el mapa de convergència?"):
        st.markdown("""
        Les zones acolorides marquen àrees de **convergència d'humitat**, que actuen com a **disparadors** potencials de tempestes.
        
        - **Què són?** Àrees on el vent en nivells baixos força l'aire humit a ajuntar-se i ascendir, iniciant el procés de formació de núvols.
        - **Com interpretar els colors i les línies?**
            - <span style="color:#ffc107; font-weight:bold;">GROC (>15):</span> Convergència **moderada**. Indica zones on es poden començar a formar xàfecs o tempestes disperses.
            - <span style="color:#ff9800; font-weight:bold;">TARONJA (>25):</span> Convergència **forta**. Entorn més favorable per a la formació de tempestes més organitzades.
            - <span style="color:#f44336; font-weight:bold;">VERMELL (>40):</span> Convergència **molt forta**. Són els focus principals a vigilar per al desenvolupament de temps potencialment sever.
        
        El número sobre la línia indica la intensitat de la convergència en aquella zona.
        """, unsafe_allow_html=True)

def ui_info_desenvolupament_tempesta():
    """Mostra un quadre informatiu sobre el temps de formació de tempestes."""
    with st.expander("⏳ De la Convergència a la Tempesta: Quant triga?", expanded=True):
        st.markdown("""
        Un cop s'activa un nucli de convergència (zones acolorides al mapa), el temps estimat per al desenvolupament d'un núvol de tempesta complet (Cumulonimbus) sol ser d'entre **20 a 60 minuts**.

        La velocitat depèn de:
        -   **Més ràpid (< 30 min):** Convergència intensa, molta energia (**CAPE alt**) i poca "tapa" atmosfèrica (**CIN baix**).
        -   **Més lent (> 45 min):** Convergència feble, una "tapa" forta (**CIN alt**) o aire sec a nivells mitjans.
        """)

# Substitueix tota la funció
def ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple):
    is_guest = st.session_state.get('guest_mode', False)
    
    marge_esq, contingut_principal, marge_dret = st.columns([0.05, 0.9, 0.05])
    with contingut_principal:
        col_map_1, col_map_2 = st.columns([0.7, 0.3], gap="large")
        with col_map_1:
            # --- NOUS SELECTORS PER CAPA I ZOOM ---
            col_capa, col_zoom = st.columns(2)
            with col_capa:
                map_options = {"Anàlisi de Vent i Convergència": "forecast_estatic", "Vent a 700hPa": "vent_700", "Vent a 300hPa": "vent_300"}
                mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
            with col_zoom:
                zoom_sel = st.selectbox("Nivell de Zoom:", options=list(MAP_ZOOM_LEVELS.keys()))
            
            selected_extent = MAP_ZOOM_LEVELS[zoom_sel]
            map_key = map_options[mapa_sel]
            
            if map_key == "forecast_estatic":
                if is_guest:
                    st.info("ℹ️ L'anàlisi de vent i convergència està fixada a **925 hPa**.")
                    nivell_sel = 925
                else:
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

                if error_map: 
                    st.error(f"Error en carregar el mapa: {error_map}")
                    progress_placeholder.empty()
                elif map_data:
                    # Passa el 'selected_extent' a la funció de creació del mapa
                    fig = crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str, selected_extent)
                    st.pyplot(fig)
                    plt.close(fig)
                    ui_explicacio_alertes()
                    
                    with progress_placeholder.container(): 
                        progress_bar.progress(100, text="Completat!")
                        time.sleep(1)
                        progress_placeholder.empty()

            elif map_key in ["vent_700", "vent_300"]:
                nivell = 700 if map_key == "vent_700" else 300
                variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
                map_data, error_map = carregar_dades_mapa_base(variables, hourly_index_sel)
                if error_map: st.error(f"Error en carregar el mapa: {error_map}")
                elif map_data: 
                    # Passa el 'selected_extent' a la funció de creació del mapa
                    fig = crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str, selected_extent)
                    st.pyplot(fig)
                    plt.close(fig)

        with col_map_2:
            st.subheader("Imatges en Temps Real")
            tab_europa, tab_ne = st.tabs(["Europa", "NE Peninsula"])
            with tab_europa: 
                mostrar_imatge_temps_real("Satèl·lit (Europa)")
            with tab_ne: 
                mostrar_imatge_temps_real("Satèl·lit (NE Península)")
            
            st.markdown("---")
            ui_info_desenvolupament_tempesta()
            
def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        
        marge_esq, contingut_principal, marge_dret = st.columns([0.05, 0.9, 0.05])
        with contingut_principal:
            st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")

            col1, col2 = st.columns(2)
            with col1:
                fig_skewt = crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], f"Sondeig Vertical - {poble_sel}")
                st.pyplot(fig_skewt)
                plt.close(fig_skewt)
            with col2:
                fig_hodo = crear_hodograf(sounding_data[3], sounding_data[4])
                st.pyplot(fig_hodo)
                plt.close(fig_hodo)

            st.divider()

            cols = st.columns(5)
            metric_params = {'CAPE': 'J/kg', 'CIN': 'J/kg', 'LFC_hPa': 'hPa', 'Shear 0-1km': 'nusos', 'Shear 0-6km': 'nusos'}
            for i, (param, unit) in enumerate(metric_params.items()):
                with cols[i]:
                    val = params_calculats.get(param)
                    color = get_color_for_param(param, val)
                    value_str = f"{val:.0f}" if val is not None and not np.isnan(val) else "---"
                    st.markdown(f"""<div style="text-align: left;"><span style="font-size: 0.8em; color: #A0A0A0;">{param}</span><br><strong style="font-size: 1.8em; color: {color};">{value_str}</strong> <span style="font-size: 1.1em; color: #A0A0A0;">{unit}</span></div>""", unsafe_allow_html=True)

            with st.expander("Què signifiquen aquests paràmetres?"):
                st.markdown("""
                - **CAPE:** Energia disponible per a les tempestes. Valors més alts indiquen un potencial per a corrents ascendents més forts.
                - **CIN:** Actua com una "tapa" que impedeix que l'aire pugi. Si és molt negatiu, es necessita molta força per iniciar la convecció.
                - **LFC:** És l'altitud a la qual una bombolla d'aire comença a pujar lliurement per si sola. Com més baix, més fàcil és iniciar tempestes.
                - **Shear 0-1km:** Diferència de vent en el primer quilòmetre. Valors alts afavoreixen la rotació en tempestes (mesociclons).
                - **Shear 0-6km:** Diferència de vent en capes profundes. Essencial per a l'organització de les tempestes (multicèl·lules, supercèl·lules).
                """)
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")
        
def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades AROME via Open-Meteo | Imatges via Meteociel | IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- 5. APLICACIÓ PRINCIPAL ---
def main():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'guest_mode' not in st.session_state: st.session_state['guest_mode'] = False
    
    if not st.session_state['logged_in'] and not st.session_state['guest_mode']:
        show_login_page()
    else:
        is_guest = st.session_state.get('guest_mode', False)

        now_local = datetime.now(TIMEZONE)
        hora_sel_str = f"{now_local.hour:02d}:00h" if is_guest else st.session_state.get('hora_selector', f"{now_local.hour:02d}:00h")
        dia_sel_str = "Avui" if is_guest else st.session_state.get('dia_selector', "Avui")
        hora_int = int(hora_sel_str.split(':')[0])
        target_date = now_local.date()
        if dia_sel_str == "Demà": target_date += timedelta(days=1)
        local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int))
        utc_dt = local_dt.astimezone(pytz.utc)
        start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600)
        hourly_index_sel = max(0, time_diff_hours)
        
        if is_guest:
            ciutats_per_selector, info_msg = obtenir_ciutats_actives(hourly_index_sel)
        else:
            ciutats_per_selector = CIUTATS_CATALUNYA
            info_msg = None

        ui_capcalera_selectors(ciutats_per_selector, info_msg)
        
        poble_sel = st.session_state.poble_selector
        
        if poble_sel not in ciutats_per_selector:
            st.session_state.poble_selector = sorted(ciutats_per_selector.keys())[0]
            st.rerun()

        dia_sel = st.session_state.dia_selector
        hora_sel = st.session_state.hora_selector
        timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
        
        lat_sel, lon_sel = ciutats_per_selector[poble_sel]['lat'], ciutats_per_selector[poble_sel]['lon']
        data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
        if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
        
        st.markdown("---")
        global progress_placeholder; progress_placeholder = st.empty()
        
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
            if 'last_seen_timestamp' not in st.session_state:
                st.session_state.last_seen_timestamp = chat_history[-1]['timestamp'] if chat_history else 0
            
            unread_count = count_unread_messages(chat_history)
            chat_tab_label = f"💬 Xat en Línia ({unread_count})" if unread_count > 0 else "💬 Xat en Línia"

            tab_ia, tab_xat, tab_mapes, tab_vertical = st.tabs(["Assistent MeteoIA", chat_tab_label, "Anàlisi de Mapes", "Anàlisi Vertical"])
            
            # ===> LÍNIA CORREGIDA <===
            with tab_ia: ui_pestanya_ia_final(data_tuple, hourly_index_sel, poble_sel, timestamp_str)
            
            with tab_xat: ui_pestanya_xat(chat_history)
            with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
            with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
            
        ui_peu_de_pagina()

if __name__ == "__main__":
    main()
