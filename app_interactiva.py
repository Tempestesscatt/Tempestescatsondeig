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
import threading
import pandas as pd
import xml.etree.ElementTree as ET

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
    'Falset': {'lat': 41.1499, 'lon': 0.8197}, 'Figueres': {'lat': 42.2662, 'lon': 2.9622},
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

# --- Funcions auxiliars ---
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
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m)
                v_profile.append(v.to('m/s').m)
                h_profile.append(p_data["H"][i])
        
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
        
        valid_indices = ~np.isnan(p_profile) & ~np.isnan(T_profile) & ~np.isnan(Td_profile) & ~np.isnan(u_profile) & ~np.isnan(v_profile)
        p, T, Td = np.array(p_profile)[valid_indices] * units.hPa, np.array(T_profile)[valid_indices] * units.degC, np.array(Td_profile)[valid_indices] * units.degC
        u, v, heights = np.array(u_profile)[valid_indices] * units('m/s'), np.array(v_profile)[valid_indices] * units('m/s'), np.array(h_profile)[valid_indices] * units.meter
        
        params_calc = {}
        prof = None
        heights_agl = heights - heights[0]

        with parcel_lock:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')

            # --- CÀLCULS D'ENERGIA ---
            try:
                sbcape, sbcin = mpcalc.cape_cin(p, T, Td, prof)
                params_calc['SBCAPE'] = sbcape.m
                params_calc['SBCIN'] = sbcin.m
                params_calc['MAX_UPDRAFT'] = np.sqrt(2 * sbcape.m) if sbcape.m > 0 else 0.0
            except Exception:
                params_calc.update({'SBCAPE': np.nan, 'SBCIN': np.nan, 'MAX_UPDRAFT': np.nan})

            try:
                mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, as_pentads=False)
                params_calc['MUCAPE'] = mucape.m
                params_calc['MUCIN'] = mucin.m
            except Exception:
                params_calc.update({'MUCAPE': np.nan, 'MUCIN': np.nan})

            try:
                mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=100 * units.hPa)
                params_calc['MLCAPE'] = mlcape.m
                params_calc['MLCIN'] = mlcin.m
            except Exception:
                params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
            
            # --- CÀLCULS DE NIVELLS I ÍNDEXS ---
            try:
                # CORREGIT: Es passa 'prof' a la funció
                li, _ = mpcalc.lifted_index(p, T, prof)
                params_calc['LI'] = li.m
            except Exception:
                params_calc['LI'] = np.nan
            
            try:
                # CORREGIT: Es passa 'prof' a la funció
                dcape, _ = mpcalc.dcape(p, T, Td, prof)
                params_calc['DCAPE'] = dcape.m
            except Exception:
                params_calc['DCAPE'] = np.nan

            try:
                # CORREGIT: Es passa 'prof' a la funció
                lfc_p, _ = mpcalc.lfc(p, T, Td, prof)
                params_calc['LFC_p'] = lfc_p.m
                params_calc['LFC_Hgt'] = np.interp(lfc_p.m, p.m[::-1], heights_agl.m[::-1])
            except Exception:
                params_calc.update({'LFC_p': np.nan, 'LFC_Hgt': np.nan})
                
            try:
                # CORREGIT: Es passa 'prof' a la funció
                el_p, _ = mpcalc.el(p, T, Td, prof)
                params_calc['EL_p'] = el_p.m
                params_calc['EL_Hgt'] = np.interp(el_p.m, p.m[::-1], heights_agl.m[::-1])
            except Exception:
                params_calc.update({'EL_p': np.nan, 'EL_Hgt': np.nan})
            
            try:
                lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0])
                params_calc['LCL_p'] = lcl_p.m
                params_calc['LCL_Hgt'] = np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1])
            except Exception:
                params_calc.update({'LCL_p': np.nan, 'LCL_Hgt': np.nan})

            try:
                pwat = mpcalc.precipitable_water(p, Td)
                params_calc['PWAT'] = pwat.to('mm').m
            except Exception:
                params_calc['PWAT'] = np.nan
                
            try:
                frz_lvl, _ = mpcalc.freezing_level(p, T)
                params_calc['FRZG_Lvl_p'] = frz_lvl.m
            except Exception:
                params_calc['FRZG_Lvl_p'] = np.nan

        # --- CÀLCULS DE CINEMÀTICA (VENT) ---
        try:
            rm, lm, mean_wind = mpcalc.storm_motion(p, u, v, heights)
            params_calc['RM'] = (rm[0].m, rm[1].m)
            params_calc['LM'] = (lm[0].m, lm[1].m)
            params_calc['Mean_Wind'] = (mean_wind[0].m, mean_wind[1].m)
        except Exception:
            params_calc.update({'RM': (np.nan, np.nan), 'LM': (np.nan, np.nan), 'Mean_Wind': (np.nan, np.nan)})
        
        for name, depth_m in [('0-1km', 1000), ('0-6km', 6000)]:
            try:
                bwd_u, bwd_v = mpcalc.bulk_shear(p, u, v, height=heights_agl, depth=depth_m * units.m)
                shear_speed = mpcalc.wind_speed(bwd_u, bwd_v)
                params_calc[f'BWD_{name}'] = shear_speed.to('kt').m
            except Exception:
                params_calc[f'BWD_{name}'] = np.nan
        
        if not np.isnan(params_calc.get('RM', [(np.nan, np.nan)])[0]):
            try:
                u_storm, v_storm = params_calc['RM'] * units('m/s')
                for name, depth_m in [('0-1km', 1000), ('0-3km', 3000)]:
                    srh, _, _ = mpcalc.storm_relative_helicity(heights, u, v, depth=depth_m * units.m, storm_u=u_storm, storm_v=v_storm)
                    params_calc[f'SRH_{name}'] = srh.m
            except Exception:
                params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})
        else:
            params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})

        try:
            # CAPE a la capa 0-3km
            idx_3km = np.argmin(np.abs(heights_agl.m - 3000))
            cape_0_3, _ = mpcalc.cape_cin(p[:idx_3km+1], T[:idx_3km+1], Td[:idx_3km+1], prof[:idx_3km+1])
            params_calc['CAPE_0-3km'] = cape_0_3.m
        except Exception:
            params_calc['CAPE_0-3km'] = np.nan

        return ((p, T, Td, u, v, heights, prof), params_calc), None
    except Exception as e:
        st.error(f"Error crític en carregar les dades del sondeig: {e}")
        return None, f"Error en processar dades del sondeig: {e}"
        
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

def crear_skewt(p, T, Td, u, v, prof, params_calc, titol):
    fig = plt.figure(dpi=150, figsize=(7, 8))
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.05, 0.85, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)
    
    skew.plot_dry_adiabats(color='coral', linestyle='--', alpha=0.5)
    skew.plot_moist_adiabats(color='cornflowerblue', linestyle='--', alpha=0.5)
    skew.plot_mixing_lines(color='limegreen', linestyle='--', alpha=0.5)

    if prof is not None:
        skew.shade_cape(p, T, prof, color='red', alpha=0.2)
        skew.shade_cin(p, T, prof, color='blue', alpha=0.2)
    
    skew.plot(p, T, 'red', lw=2.5, label='Temperatura')
    skew.plot(p, Td, 'green', lw=2.5, label='Punt de Rosada')
    if prof is not None:
        skew.plot(p, prof, 'k', linewidth=3, label='Trajectòria Parcel·la', path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])
    
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03)
    
    skew.ax.set_ylim(1000, 100); skew.ax.set_xlim(-40, 40)
    skew.ax.set_title(titol, weight='bold', fontsize=14, pad=15)
    skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")
    
    levels_to_plot = {'LCL_p': 'LCL', 'FRZG_Lvl_p': '0°C', 'LFC_p': 'LFC'}
    for key, name in levels_to_plot.items():
        p_lvl = params_calc.get(key)
        if p_lvl is not None and hasattr(p_lvl, 'm') and not np.isnan(p_lvl.m):
            skew.ax.axhline(p_lvl.m, color='blue', linestyle='--', linewidth=1.5)
            skew.ax.text(skew.ax.get_xlim()[1] - 2, p_lvl.m, f' {name}', color='blue', ha='right', va='center', fontsize=10, weight='bold')

    skew.ax.legend()
    return fig

def crear_hodograf_avancat(p, u, v, heights, params_calc, titol):
    fig = plt.figure(dpi=150, figsize=(8, 8))
    
    gs = fig.add_gridspec(nrows=2, ncols=2,
                          height_ratios=[1.5, 6],
                          width_ratios=[1.5, 1],
                          hspace=0.4, wspace=0.3)

    ax_barbs = fig.add_subplot(gs[0, :])
    ax_hodo = fig.add_subplot(gs[1, 0])
    ax_params = fig.add_subplot(gs[1, 1])

    fig.suptitle(titol, weight='bold', fontsize=16)

    # --- Dibuixem les barbes de vent ---
    ax_barbs.set_title("Vent a Nivells Clau", fontsize=11, pad=15)
    heights_agl = heights - heights[0]
    barb_altitudes_km = [1, 3, 6, 9]
    barb_altitudes_m = [h * 1000 for h in barb_altitudes_km] * units.m
    u_barbs_list, v_barbs_list = [], []

    for h_m in barb_altitudes_m:
        if h_m <= heights_agl.max():
            u_interp_val = np.interp(h_m.m, heights_agl.m, u.m)
            v_interp_val = np.interp(h_m.m, heights_agl.m, v.m)
            u_barbs_list.append(u_interp_val)
            v_barbs_list.append(v_interp_val)
        else:
            u_barbs_list.append(np.nan)
            v_barbs_list.append(np.nan)

    u_barbs = units.Quantity(u_barbs_list, u.units)
    v_barbs = units.Quantity(v_barbs_list, v.units)
    
    speed_kmh_barbs = np.sqrt(u_barbs**2 + v_barbs**2).to('km/h').m
    thresholds_barbs = [10, 40, 70, 100, 130]
    colors_barbs = ['dimgrey', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    x_pos = np.arange(len(barb_altitudes_km))
    u_barbs_kt = u_barbs.to('kt')
    v_barbs_kt = v_barbs.to('kt')

    for i, spd_kmh in enumerate(speed_kmh_barbs):
        if not np.isnan(spd_kmh):
            color_index = np.searchsorted(thresholds_barbs, spd_kmh)
            color = colors_barbs[color_index]
            ax_barbs.barbs(x_pos[i], 0, u_barbs_kt[i], v_barbs_kt[i], length=8, pivot='middle', color=color)
            ax_barbs.text(x_pos[i], -0.8, f"{spd_kmh:.0f} km/h", ha='center', va='top', fontsize=9, color=color, weight='bold')
        else:
            ax_barbs.text(x_pos[i], 0, "N/A", ha='center', va='center', fontsize=9, color='grey')

    ax_barbs.set_xticks(x_pos); ax_barbs.set_xticklabels([f"{h} km" for h in barb_altitudes_km])
    ax_barbs.set_yticks([]); ax_barbs.spines[:].set_visible(False)
    ax_barbs.tick_params(axis='x', length=0, pad=5); ax_barbs.set_xlim(-0.5, len(barb_altitudes_km) - 0.5)
    ax_barbs.set_ylim(-1.5, 1.5)

    # --- Dibuix de l'hodògraf ---
    h = Hodograph(ax_hodo, component_range=80.)
    h.add_grid(increment=20, color='gray', linestyle='--')
    
    intervals = np.array([0, 1, 3, 6, 9, 12]) * units.km
    colors_hodo = ['red', 'blue', 'green', 'purple', 'gold']
    h.plot_colormapped(u.to('kt'), v.to('kt'), heights, intervals=intervals, colors=colors_hodo, linewidth=2)
    
    try:
        rm_vec = params_calc.get('RM')
        if rm_vec and not np.isnan(rm_vec[0]):
            u_rm_kt = (rm_vec[0] * units('m/s')).to('kt').m
            v_rm_kt = (rm_vec[1] * units('m/s')).to('kt').m
            ax_hodo.plot(u_rm_kt, v_rm_kt, 'o', color='blue', markersize=8, label='Mov. Dret')
    except Exception: pass
    
    ax_hodo.set_xlabel('U-Component (nusos)')
    ax_hodo.set_ylabel('V-Component (nusos)')

    # --- Dibuix del panell de text ---
    ax_params.axis('off')
    
    def degrees_to_cardinal_ca(d):
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        ix = int(round(((d % 360) / 45)))
        return dirs[ix % 8]

    def get_color(value, thresholds):
        if pd.isna(value): return "grey"
        colors = ["grey", "green", "#E69F00", "orange", "red", "#C71585"]
        thresholds = sorted(thresholds)
        for i, threshold in enumerate(thresholds):
            if value < threshold: return colors[i]
        return colors[-1]

    THRESHOLDS = {
        'BWD': (10, 20, 30, 40), 
        'SRH': (50, 150, 250, 400),
        'UPDRAFT': (15, 25, 40, 50) # Llindars per W_max en m/s
    }
    
    y = 0.98
    
    motion_data = {
        'MD': params_calc.get('RM'),
        'ML': params_calc.get('LM'),
        'VM (0-6 km)': params_calc.get('Mean_Wind')
    }

    ax_params.text(0, y, "Moviment (dir/km/h)", ha='left', weight='bold', fontsize=11); y-=0.1

    for display_name, vec in motion_data.items():
        if vec and not any(pd.isna(v) for v in vec):
            u_motion_ms, v_motion_ms = vec[0] * units('m/s'), vec[1] * units('m/s')
            speed_kmh = mpcalc.wind_speed(u_motion_ms, v_motion_ms).to('km/h').m
            direction_from_deg = mpcalc.wind_direction(u_motion_ms, v_motion_ms, convention='from').to('deg').m
            cardinal_dir_ca = degrees_to_cardinal_ca(direction_from_deg)
            ax_params.text(0, y, f"{display_name}:", ha='left', va='center')
            ax_params.text(1, y, f"{direction_from_deg:.0f}° ({cardinal_dir_ca}) / {speed_kmh:.0f}", ha='right', va='center')
        else:
            ax_params.text(0, y, f"{display_name}:", ha='left', va='center')
            ax_params.text(1, y, "---", ha='right', va='center')
        y-=0.08

    y-=0.05
    ax_params.text(0, y, "Cisallament (nusos)", ha='left', weight='bold', fontsize=11); y-=0.1
    # --- BLOC MODIFICAT: S'ha eliminat la línia 'Efectiu' ---
    for key, label in [('0-1km', '0-1 km'), ('0-6km', '0-6 km')]:
        val = params_calc.get(f'BWD_{key}', np.nan)
        color = get_color(val, THRESHOLDS['BWD'])
        ax_params.text(0, y, f"{label}:", ha='left', va='center')
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color)
        y-=0.08

    y-=0.05
    ax_params.text(0, y, "Helicitat (m²/s²)", ha='left', weight='bold', fontsize=11); y-=0.1
    # --- BLOC MODIFICAT: S'ha eliminat la línia 'Efectiva' ---
    for key, label in [('0-1km', '0-1 km'), ('0-3km', '0-3 km')]:
        val = params_calc.get(f'SRH_{key}', np.nan)
        color = get_color(val, THRESHOLDS['SRH'])
        ax_params.text(0, y, f"{label}:", ha='left', va='center')
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color)
        y-=0.08
    
    # --- NOU BLOC COMPLET: S'afegeix la secció 'Corrent Ascendent' ---
    y-=0.05
    ax_params.text(0, y, "Corrent Ascendent", ha='left', weight='bold', fontsize=11); y-=0.1
    val_updraft = params_calc.get('MAX_UPDRAFT', np.nan)
    color_updraft = get_color(val_updraft, THRESHOLDS['UPDRAFT'])
    ax_params.text(0, y, f"Vel. Max (0-6km):", ha='left', va='center')
    ax_params.text(1, y, f"{val_updraft:.1f} m/s" if not pd.isna(val_updraft) else "---", ha='right', va='center', weight='bold', color=color_updraft)
    y-=0.08
        
    return fig
    
def ui_caixa_parametres_sondeig(params):
    def get_color(value, thresholds, reverse_colors=False):
        if pd.isna(value): return "#808080"
        colors = ["#808080", "#28a745", "#ffc107", "#fd7e14", "#dc3545"]
        if reverse_colors:
            thresholds = sorted(thresholds, reverse=True); colors = list(reversed(colors))
        else: thresholds = sorted(thresholds)
        for i, threshold in enumerate(thresholds):
            if value < threshold: return colors[i]
        return colors[-1]

    THRESHOLDS = {
        'SBCAPE': (100, 500, 1500, 2500), 'MUCAPE': (100, 500, 1500, 2500),
        'MLCAPE': (50, 250, 1000, 2000), 'CAPE_0-3km': (25, 75, 150, 250),
        'DCAPE': (200, 500, 800, 1200), 'SBCIN': (0, -25, -75, -150), 
        'LI': (0, -2, -5, -8), 'PWAT': (20, 30, 40, 50),
        'BWD_0-6km': (10, 20, 30, 40), 'SRH_0-1km': (50, 100, 150, 250)
    }

    def styled_metric(label, value, unit, param_key, precision=0, reverse_colors=False):
        color = get_color(value, THRESHOLDS.get(param_key, []), reverse_colors)
        val_str = f"{value:.{precision}f}" if not pd.isna(value) else "---"
        st.markdown(f"""
            <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;">
                <span style="font-size: 0.8em; color: #FAFAFA;">{label} ({unit})</span><br>
                <strong style="font-size: 1.6em; color: {color};">{val_str}</strong>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("##### Paràmetres del Sondeig")
    
    # Fila 1: Energia Principal
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCAPE", params.get('SBCAPE', np.nan), "J/kg", 'SBCAPE')
    with cols[1]: styled_metric("MUCAPE", params.get('MUCAPE', np.nan), "J/kg", 'MUCAPE')
    with cols[2]: styled_metric("MLCAPE", params.get('MLCAPE', np.nan), "J/kg", 'MLCAPE')
    
    # Fila 2: Estabilitat i Humitat
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCIN", params.get('SBCIN', np.nan), "J/kg", 'SBCIN', reverse_colors=True)
    with cols[1]: styled_metric("LI", params.get('LI', np.nan), "°C", 'LI', precision=1, reverse_colors=True)
    with cols[2]: styled_metric("PWAT", params.get('PWAT', np.nan), "mm", 'PWAT', precision=1)
    
    # Fila 3: Nivells i Potencial Descendent
    cols = st.columns(3)
    with cols[0]: styled_metric("LCL", params.get('LCL_Hgt', np.nan), "m", '', precision=0)
    with cols[1]: styled_metric("LFC", params.get('LFC_Hgt', np.nan), "m", '', precision=0)
    with cols[2]: styled_metric("DCAPE", params.get('DCAPE', np.nan), "J/kg", 'DCAPE')
    
    # Fila 4: Paràmetres de Cisallament / Rotació
    cols = st.columns(3)
    with cols[0]: styled_metric("BWD 0-6km", params.get('BWD_0-6km', np.nan), "nusos", 'BWD_0-6km')
    with cols[1]: styled_metric("SRH 0-1km", params.get('SRH_0-1km', np.nan), "m²/s²", 'SRH_0-1km')
    with cols[2]: styled_metric("CAPE 0-3km", params.get('CAPE_0-3km', np.nan), "J/kg", 'CAPE_0-3km')
        
        

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        lat_sel = CIUTATS_CATALUNYA[poble_sel]['lat']
        lon_sel = CIUTATS_CATALUNYA[poble_sel]['lon']
        
        sounding_data, params_calculats = data_tuple
        p, T, Td, u, v, heights, prof = sounding_data
        
        col1, col2 = st.columns(2, gap="large")

        with col1:
            fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Sondeig Vertical\n{poble_sel}")
            st.pyplot(fig_skewt, use_container_width=True)
            plt.close(fig_skewt)
            
            # PRIMERO los parámetros del sondeig
            with st.container(border=True):
                ui_caixa_parametres_sondeig(params_calculats)
            
            # LUEGO la caja desplegable explicativa DEBAJO de los parámetros
            with st.expander("📚 Guia Completa d'Interpretació dels Paràmetres", expanded=False):
                st.markdown("""
                ## 🎯 **Com Interpretar els Paràmetres de Sondeig**
                
                ### ⚡ **ENERGIA DISPONIBLE (CAPE)**
                **Què és**: Energia que té una parcel·la d'aire per pujar i formar núvols de desenvolupament vertical.
                
                **Escala i Exemples**:
                - **0-500 J/kg**: 🟢 Energia baixa - Núvols poc desenvolupats
                - **500-1500 J/kg**: 🟡 Energia moderada - Possibles xàfecs
                - **1500-2500 J/kg**: 🟠 Energia alta - Tempestes fortes
                - **>2500 J/kg**: 🔴 Energia molt alta - Tempestes violentes
                *Exemple: 1800 J/kg = Bon potencial per a calamarsa*
                
                ### 🚫 **ENERGIA D'INHIBICIÓ (CIN)**
                **Què és**: Energia que impedeix que comenci la convecció (com una "tapa").
                
                **Escala i Exemples**:
                - **0 a -25 J/kg**: 🟢 Poca inhibició - Fàcil inici
                - **-25 a -75 J/kg**: 🟡 Inhibició moderada - Cal forçament
                - **-75 a -150 J/kg**: 🟠 Forta inhibició - Difícil inici
                - **< -150 J/kg**: 🔴 Inhibició molt forta - Gairebé impossible
                *Exemple: -50 J/kg = Cal un front o orografia per iniciar*
                
                ### 🌬️ **CISALLAMENT DEL VENT**
                **Què és**: Canvi de vent amb l'altura. Essencial per organitzar les tempestes.
                
                **0-6 km Cisallament**:
                - **<20 nusos**: 🟢 Cisallament feble - Tempestes poc organitzades
                - **20-30 nusos**: 🟡 Cisallament moderat - Cèl·lules multicel·lulars
                - **30-40 nusos**: 🟠 Cisallament fort - Possibles supercèl·lules
                - **>40 nusos**: 🔴 Cisallament molt fort - Supercèl·lules probables
                *Exemple: 35 nusos = Condicions favorables per a supercèl·lules*
                
                ### 🌀 **HELICITAT (SRH)**
                **Què és**: Potencial de rotació en les tempestes.
                
                **SRH 0-3 km**:
                - **<150 m²/s²**: 🟢 Poca rotació - Tempestes sense rotació
                - **150-300 m²/s²**: 🟡 Rotació moderada - Possibles mesociclons
                - **300-450 m²/s²**: 🟠 Rotació forta - Alt potencial de rotació
                - **>450 m²/s²**: 🔴 Rotació molt forta - Alt risc de tornados
                *Exemple: 280 m²/s² = Possible mesociclon i calamarsa grossa*
                
                ### 🌡️ **NIVELLS CLAU**
                **LCL (Nivell de Condensació)**:
                - **<800m**: 🟢 Molt favorable - Base de núvols baixa
                - **800-1200m**: 🟡 Favorable - Bones condicions
                - **1200-1500m**: 🟠 Regular - Condicions mitjanes
                - **>1500m**: 🔴 Desfavorable - Base massa alta
                
                **LI (Índex d'Elevació)**:
                - **>0**: 🔴 Estable - Cap tempesta
                - **0 a -2**: 🟡 lleugerament inestable - Xàfecs febles
                - **-2 a -5**: 🟠 Moderadament inestable - Tempestes moderades
                - **< -5**: 🟢 Molt inestable - Tempestes fortes
                *Exemple: LI = -4 i LCL = 900m = Bones condicions*
                
                ### 💧 **HUMITAT (PWAT)**
                **Què és**: Aigua precipitable total en la columna d'aire.
                
                **Valors**:
                - **<20mm**: 🔴 Sec - Poca humitat
                - **20-30mm**: 🟡 Normal - Humitat moderada
                - **30-40mm**: 🟠 Humit - Molta humitat
                - **>40mm**: 🟢 Molt humit - Alt potencial pluja
                *Exemple: PWAT = 35mm + Alt CAPE = Risc d'inundacions sobtades*
                
                ## 🎪 **ESCENARIS TÍPICS A CATALUNYA**
                
                **Escenari 1 - Tempestes d'estiu típiques**:
                - CAPE: 800-1500 J/kg | CIN: -30 J/kg | Cisallament: 15-20 nusos
                - *Resultat: Xàfecs aïllats, calamarsa petita*
                
                **Escenari 2 - Tempestes organitzades**:
                - CAPE: 1500-2500 J/kg | CIN: -20 J/kg | Cisallament: 25-35 nusos
                - *Resultat: Linies de tempesta, calamarsa grossa*
                
                **Escenari 3 - Supercèl·lules**:
                - CAPE: 2000-3000 J/kg | CIN: -10 J/kg | Cisallament: 35-45 nusos | SRH: 250-400 m²/s²
                - *Resultat: Tempestes rotatives, risc de tornado*
                
                **Escenari 4 - Inestabilitat seca**:
                - CAPE: 1000-1800 J/kg | PWAT: <25mm | CIN: -40 J/kg
                - *Resultat: Tempestes amb poca pluja però molta calamarsa*
                """)
        
        with col2:
            fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat\n{poble_sel}")
            st.pyplot(fig_hodo, use_container_width=True)
            plt.close(fig_hodo)
            
            # CAJA DESPLEGABLE BAJO EL HODÓGRAFO - NUEVA SECCIÓN
            with st.expander("🌀 Com Interpretar l'Hodògraf", expanded=False):
                st.markdown("""
                ## 🌀 **Guia d'Interpretació de l'Hodògraf**
                
                ### 🎨 **Colors i Capes d'Alçada**
                - **Vermell**: 0-1 km (Capa límit)
                - **Blau**: 1-3 km (Baixa mitjana troposfera)  
                - **Verd**: 3-6 km (Mitjana troposfera)
                - **Lila**: 6-9 km (Alta troposfera)
                - **Groc**: 9-12 km (Alta troposfera)
                
                ### 📍 **Marcadors de Moviment**
                - **🔷 MD (Moviment Dret)**: Trajectòria de la part dreta de la tempesta
                - **🔶 ML (Moviment Esquerre)**: Trajectòria de la part esquerra
                - **⚫ VM (Vent Mitjà)**: Moviment mitjà de la tempesta
                
                ### 🎯 **Interpretació dels Moviments**
                
                **MD vs ML a Catalunya**:
                - **MD (Dret)**: Almost sempre la cèl·lula dominant i més perillosa
                - **ML (Esquerre)**: Tendència a dissipar-se més ràpidament
                
                **Exemples de Configuracions**:
                
                **Hodògraf Corbat (Favorable)**:
                - Forma de "C" o "S" marcada
                - Vent canviant amb l'altura
                - **Resultat**: Tempestes organitzades amb rotació
                
                **Hodògraf Recte (Poc organitzat)**:
                - Línia quasi recta
                - Poc canvi de direcció
                - **Resultat**: Tempestes aïllades poc organitzades
                
                **Hodògraf Gran (Supercèl·lules)**:
                - Gran extensió en totes direccions
                - Fort cisallament
                - **Resultat**: Alt potencial per supercèl·lules
                
                ### 📏 **Mides i Escales**
                - **Eixos**: Mesuren components del vent (nusos)
                - **Distància al centre**: Indica velocitat del vent
                - **Grandària general**: Indica cisallament total
                
                ### 🎪 **Escenaris Pratcis a Catalunya**
                
                **Configuració de Ponent**:
                - Vent de ponent a baixos nivells
                - Vent del sud a mitjans nivells
                - **Resultat**: Tempestes que es mouen cap a la costa
                
                **Configuració de Llevant**:
                - Vent de llevant a baixos nivells  
                - Vent de ponent a alts nivells
                - **Resultat**: Tempestes estacionàries o retrògrades
                
                **Configuració de Sud**:
                - Vent del sud a tots els nivells
                - **Resultat**: Tempestes que pugen cap al Pirineu
                
                ### ⚠️ **Atenció a Aquests Patterns**
                
                **Signes de Rotació**:
                - Hodògraf corbat i ample
                - Gran SRH (Helicitat)
                - Moviment MD ben definit
                
                **Signes de Cisallament**:
                - Hodògraf gran i estès
                - Diferència clara entre capes
                - BWD (Bulk Wind Difference) alt
                
                **Signes de Inestabilitat**:
                - Vent feble a baixos nivells
                - Fort vent a mitjans nivells
                - Canvis sobtats de direcció
                """)

            st.markdown("##### Radar de Precipitació en Temps Real")
            radar_url = f"https://www.rainviewer.com/map.html?loc={lat_sel},{lon_sel},10&oCS=1&c=3&o=83&lm=0&layer=radar&sm=1&sn=1&ts=2&play=1"
            
            html_code = f"""
            <div style="position: relative; width: 100%; height: 410px; border-radius: 10px; overflow: hidden;">
                <iframe
                    src="{radar_url}"
                    width="100%"
                    height="410"
                    frameborder="0"
                    style="border:0;"
                ></iframe>
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; cursor: default;"></div>
            </div>
            """
            st.components.v1.html(html_code, height=410)

    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")
        
        
def ui_pestanya_ia_final(data_tuple, hourly_index_sel, poble_sel, timestamp_str):
    # Configuración inicial
    st.subheader("🌤️ Assistente Meteo-Col·lega Pro (con Google Gemini)")
    
    # Verificar configuración de API
    try: 
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except (KeyError, AttributeError): 
        st.error("❌ Falta la GEMINI_API_KEY en los secrets de Streamlit.")
        st.info("Por favor, configura tu API key en la sección de secrets de Streamlit.")
        return
    
    # Verificar autenticación
    username = st.session_state.get('username')
    if not username: 
        st.error("🔒 Error de autenticación. Por favor, inicia sesión.")
        return
    
    # Configuración de límites de uso mejorada
    LIMIT_PER_WINDOW = 15  # Aumentado ligeramente
    WINDOW_HOURS = 3
    PREMIUM_LIMIT = 30  # Límite para usuarios premium
    
    # Cargar límites de uso
    rate_limits = load_json_file(RATE_LIMIT_FILE)
    user_limit_data = rate_limits.get(username, {"count": 0, "window_start_time": None, "is_premium": False})
    
    # Verificar si el usuario es premium (lógica de ejemplo)
    if user_limit_data.get("is_premium", False):
        user_limit = PREMIUM_LIMIT
        user_type = "Premium"
    else:
        user_limit = LIMIT_PER_WINDOW
        user_type = "Estándar"
    
    # Gestión de ventana de tiempo
    current_time = datetime.now(pytz.utc)
    if user_limit_data.get("window_start_time"):
        start_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc)
        time_since_start = current_time - start_time
        
        if time_since_start > timedelta(hours=WINDOW_HOURS):
            # Reiniciar contador si ha pasado la ventana de tiempo
            user_limit_data.update({"count": 0, "window_start_time": current_time.timestamp()})
            rate_limits[username] = user_limit_data
            save_json_file(rate_limits, RATE_LIMIT_FILE)
    
    # Verificar si se ha alcanzado el límite
    current_count = user_limit_data.get("count", 0)
    limit_reached = current_count >= user_limit
    
    # Mostrar información de límites de uso
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Usuario:** {username} ({user_type})")
    with col2:
        preguntes_restants = max(0, user_limit - current_count)
        color = "green" if preguntes_restants > 5 else "orange" if preguntes_restants > 0 else "red"
        emoji = "✅" if preguntes_restants > 5 else "⚠️" if preguntes_restants > 0 else "❌"
        st.markdown(f"{emoji} **Preguntas restantes:** <span style='color: {color}; font-weight: bold;'>{preguntes_restants}/{user_limit}</span>", 
                   unsafe_allow_html=True)
    with col3:
        if user_limit_data.get("window_start_time"):
            reset_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc) + timedelta(hours=WINDOW_HOURS)
            time_left = reset_time - current_time
            if time_left.total_seconds() > 0:
                st.markdown(f"🕒 **Renovación:** {format_time_left(time_left)}")
    
    # Mensaje de límite alcanzado
    if limit_reached:
        reset_time = datetime.fromtimestamp(user_limit_data["window_start_time"], tz=pytz.utc) + timedelta(hours=WINDOW_HOURS)
        time_left = reset_time - current_time
        
        if time_left.total_seconds() > 0:
            st.warning(f"""
            **Límite de consultas alcanzado.**
            
            Has utilizado {user_limit} preguntas en las últimas {WINDOW_HOURS} horas. 
            Podrás realizar nuevas consultas en **{format_time_left(time_left)}**.
            
            *¿Eres un usuario frecuente? Considera actualizar a premium para aumentar tu límite.*
            """)
        else:
            # Reiniciar si el tiempo ha expirado
            user_limit_data.update({"count": 0, "window_start_time": None})
            rate_limits[username] = user_limit_data
            save_json_file(rate_limits, RATE_LIMIT_FILE)
            limit_reached = False
            st.rerun()
    
    # Inicializar chat si no existe
    if "chat" not in st.session_state:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt del sistema mejorado
        system_prompt = """Eres 'Meteo-Col·lega Pro', un experto en meteorología de Catalunya con más de 20 años de experiencia. 
        Eres directo, cercano y hablas de 'tú'. Tu misión es proporcionar un análisis completo pero conciso.

        # REGLAS PRINCIPALES:
        1. **NO describas los datos crudos** - el usuario ya puede ver los valores específicos.
        2. **Proporciona contexto regional** - explica cómo afecta la situación a la zona específica.
        3. **Identifica patrones clave** - destaca los factores más relevantes para el pronóstico.
        4. **Sé práctico** - ofrece insights útiles para la toma de decisiones.
        5. **Menciona limitaciones** - si los datos tienen limitaciones, menciónalo brevemente.

        # ESTRUCTURA DE RESPUESTA RECOMENDADA:
        - **Situación actual:** Breve contexto de lo que está ocurriendo.
        - **Factores clave:** 2-3 elementos más importantes a considerar.
        - **Pronóstico conciso:** Qué esperar en las próximas horas.
        - **Recomendación práctica:** Consejo específico para el usuario.

        # EJEMPLOS DE RESPUESTAS:
        - Para riesgo de tormentas: "La combinación de alta inestabilidad (CAPE > 2000 J/kg) y cizalladura moderada sugiere posible desarrollo de tormentas organizadas. Espera actividad entre las 15-18h, con riesgo de granizo pequeño. Recomiendo monitorizar hacia el noroeste después de las 14h."
        - Para condiciones estables: "La atmósfera muestra notable estabilidad con inversión térmica en capas bajas. No se espera desarrollo convectivo significativo. Ideal para actividades al aire libre sin preocupaciones por lluvia."
        - Para situaciones complejas: "Hay señales contradictorias: buena humedad superficial pero capa seca en niveles medios. Si se desarrollan tormentas, serían aisladas pero potencialmente intensas. Vigila especialmente entre las 17-19h."

        Zonas disponibles: """ + ', '.join(CIUTATS_CATALUNYA.keys())
        
        missatge_inicial_model = f"""
        ¡Hola! Soy tu Meteo-Col·lega Pro. 👋 
        
        He analizado los datos para **{poble_sel.upper()}** en la fecha **{timestamp_str}**.
        
        ¿En qué puedo ayudarte hoy? Puedes preguntarme sobre:
        - Riesgo de tormentas o precipitación
        - Condiciones para actividades específicas
        - Explicación de patrones meteorológicos
        - Comparación con otros días
        - Cualquier otra duda meteorológica
        """
        
        st.session_state.chat = model.start_chat(history=[
            {'role': 'user', 'parts': [system_prompt]}, 
            {'role': 'model', 'parts': [missatge_inicial_model]}
        ])
        st.session_state.messages = [{"role": "assistant", "content": missatge_inicial_model}]
    
    # Encabezado de análisis
    st.markdown(f"### 📍 Análisis para: `{poble_sel.upper()}` | 📅 Fecha: `{timestamp_str}`")
    
    # Selector de nivel con descripción mejorada
    nivell_options = {
        1000: "Superficie (1000 hPa) - Condiciones en superficie",
        950: "Nivel bajo (950 hPa) - ~500m altitud",
        925: "Baja atmósfera (925 hPa) ~750m",
        850: "Nivel medio (850 hPa) ~1500m - Importante para precipitación",
        800: "Nivel medio-alto (800 hPa) ~2000m",
        700: "Nivel alto (700 hPa) ~3000m - Dirección de sistemas"
    }
    
    nivell_mapa_ia = st.selectbox(
        "**Selecciona el nivel de análisis:**",
        options=list(nivell_options.keys()),
        format_func=lambda x: f"{x} hPa - {nivell_options[x]}",
        key="ia_level_selector_chat_final",
        disabled=limit_reached,
        help="Selecciona el nivel atmosférico para el análisis. Diferentes niveles muestran patrones meteorológicos distintos."
    )
    
    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input de chat con sugerencias
    if not limit_reached:
        sugerencias = [
            "¿Qué riesgo de tormentas hay hoy?",
            "¿Es un buen día para actividades al aire libre?",
            "Explica los patrones principales que ves",
            "¿Cómo comparas hoy con días anteriores?",
            "¿Hay algún factor inusual que deba conocer?"
        ]
        
        st.markdown("**💡 Sugerencias de preguntas:**")
        cols = st.columns(3)
        for i, sug in enumerate(sugerencias):
            with cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", help="Haz clic para usar esta pregunta"):
                    st.session_state.pregunta_predefinida = sug
                    st.rerun()
    
    # Manejar preguntas predefinidas
    if "pregunta_predefinida" in st.session_state:
        prompt_usuari = st.session_state.pregunta_predefinida
        del st.session_state.pregunta_predefinida
    else:
        prompt_usuari = st.chat_input("¿Qué te gustaría saber sobre el tiempo hoy?", disabled=limit_reached)
    
    # Procesar pregunta del usuario
    if prompt_usuari and not limit_reached:
        # Actualizar contador de uso
        if user_limit_data.get("window_start_time") is None:
            user_limit_data["window_start_time"] = current_time.timestamp()
        
        user_limit_data["count"] += 1
        rate_limits[username] = user_limit_data
        save_json_file(rate_limits, RATE_LIMIT_FILE)
        
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt_usuari})
        with st.chat_message("user"):
            st.markdown(prompt_usuari)
        
        # Procesar con asistente
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analizando datos y generando insights..."):
                try:
                    # Cargar y procesar datos del mapa
                    map_data_ia, error_map_ia = carregar_dades_mapa(nivell_mapa_ia, hourly_index_sel)
                    if error_map_ia:
                        st.error(f"Error cargando datos del mapa: {error_map_ia}")
                        # Intentar con nivel alternativo si falla
                        nivell_alternativo = 850 if nivell_mapa_ia != 850 else 700
                        st.info(f"Intentando con nivel alternativo: {nivell_alternativo} hPa")
                        map_data_ia, error_map_ia = carregar_dades_mapa(nivell_alternativo, hourly_index_sel)
                        
                        if error_map_ia:
                            st.error(f"También falló el nivel alternativo: {error_map_ia}")
                            return
                    
                    # Generar visualizaciones
                    fig_mapa = crear_mapa_forecast_combinat(
                        map_data_ia['lons'], map_data_ia['lats'], 
                        map_data_ia['speed_data'], map_data_ia['dir_data'], 
                        map_data_ia['dewpoint_data'], nivell_mapa_ia, 
                        timestamp_str, MAP_EXTENT
                    )
                    buf_mapa = io.BytesIO()
                    fig_mapa.savefig(buf_mapa, format='png', dpi=150, bbox_inches='tight')
                    buf_mapa.seek(0)
                    img_mapa = Image.open(buf_mapa)
                    plt.close(fig_mapa)
                    
                    # Preparar contenido para IA
                    contingut_per_ia = [img_mapa]
                    
                    # Añadir sondeo y hodógrafo si están disponibles
                    if data_tuple: 
                        sounding_data, params_calculats = data_tuple
                        p, T, Td, u, v, heights, prof = sounding_data
                        
                        # Crear gráfico skew-T
                        fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Perfil Vertical - {poble_sel}")
                        buf_skewt = io.BytesIO()
                        fig_skewt.savefig(buf_skewt, format='png', dpi=150, bbox_inches='tight')
                        buf_skewt.seek(0)
                        img_skewt = Image.open(buf_skewt)
                        plt.close(fig_skewt)
                        contingut_per_ia.append(img_skewt)
                        
                        # Crear hodógrafo
                        fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodógrafo - {poble_sel}")
                        buf_hodo = io.BytesIO()
                        fig_hodo.savefig(buf_hodo, format='png', dpi=150, bbox_inches='tight')
                        buf_hodo.seek(0)
                        img_hodo = Image.open(buf_hodo)
                        plt.close(fig_hodo)
                        contingut_per_ia.append(img_hodo)
                    
                    # Añadir contexto de la pregunta
                    prompt_context = f"""
                    PREGUNTA DEL USUARIO: '{prompt_usuari}'
                    UBICACIÓN: {poble_sel}
                    FECHA: {timestamp_str}
                    NIVEL DE ANÁLISIS: {nivell_mapa_ia} hPa ({nivell_options.get(nivell_mapa_ia, '')})
                    """
                    contingut_per_ia.insert(0, prompt_context)
                    
                    # Obtener respuesta del modelo
                    resposta = st.session_state.chat.send_message(contingut_per_ia)
                    full_response = resposta.text
                    
                except Exception as e:
                    if "429" in str(e):
                        full_response = """
                        **⚠️ Límite de frecuencia alcanzado en la API.**
                        
                        Hemos excedido el límite de consultas a la API de Google Gemini. 
                        Por favor, espera unos minutos antes de realizar otra consulta.
                        
                        Mientras tanto, puedes:
                        - Revisar los datos visuales disponibles
                        - Consultar predicciones de otras fuentes
                        - Intentar de nuevo en 5-10 minutos
                        """
                        # Revertir contador por error de API
                        user_limit_data["count"] = max(0, user_limit_data.get("count", 1) - 1)
                        rate_limits[username] = user_limit_data
                        save_json_file(rate_limits, RATE_LIMIT_FILE)
                    else:
                        full_response = f"""
                        **❌ Error técnico inesperado.**
                        
                        Hemos encontrado un problema al procesar tu consulta: 
                        `{str(e)}`
                        
                        Por favor, intenta de nuevo o contacta con soporte si el problema persiste.
                        """
            
            # Mostrar respuesta
            st.markdown(full_response)
            
            # Añadir botones de feedback
            col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 3])
            with col_fb1:
                if st.button("👍 Útil", key="feedback_positive"):
                    st.success("¡Gracias por tu feedback!")
            with col_fb2:
                if st.button("👎 Poco útil", key="feedback_negative"):
                    st.info("Lamentamos que no fuera útil. ¿Podrías especificar qué mejorar?")
            with col_fb3:
                if st.button("🔄 Reformular", key="reformular"):
                    st.info("Reformulando la respuesta...")
                    # Lógica para reformular (simplificada)
                    try:
                        reformulada = st.session_state.chat.send_message(
                            f"Reformula esta respuesta de manera más clara o concisa: {full_response}"
                        )
                        st.markdown("**Respuesta reformulada:**")
                        st.markdown(reformulada.text)
                        full_response = reformulada.text
                    except:
                        st.warning("No se pudo reformular en este momento.")
        
        # Añadir respuesta al historial y rerun
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

    # Pie de página informativo
    if not limit_reached:
        st.markdown("---")
        st.markdown("""
        <div style='font-size: 0.8em; color: #666;'>
        <b>Nota:</b> Este asistente utiliza modelos de IA para interpretar datos meteorológicos. 
        Las predicciones pueden tener incertidumbre inherente y deben considerarse como guías, 
        no como pronósticos definitivos. Para alertas oficiales, consulta siempre fuentes oficiales.
        </div>
        """, unsafe_allow_html=True)

# Función auxiliar para formatear tiempo restante
def format_time_left(time_left):
    hours, remainder = divmod(time_left.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"
        
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
    
    st.markdown("---")
    
    # --- LÍNIA MODIFICADA: Canviem la proporció de les columnes ---
    col1, col2 = st.columns([0.7, 0.3], gap="large") # El mapa ocuparà el 70% i el satèl·lit el 30%

    with col1:
        st.markdown("#### Mapes de Pronòstic (Model AROME)")
        col_capa, col_zoom = st.columns(2)
        with col_capa:
            map_options = {"Anàlisi de Vent i Convergència": "forecast_estatic", "Vent a 700hPa": "vent_700", "Vent a 300hPa": "vent_300"}
            mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
        with col_zoom: 
            zoom_sel = st.selectbox("Nivell de Zoom:", options=list(MAP_ZOOM_LEVELS.keys()))
        
        selected_extent = MAP_ZOOM_LEVELS[zoom_sel]
        map_key = map_options[mapa_sel]
        
        if map_key == "forecast_estatic":
            nivell_sel = 925 if is_guest else st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa")
            if is_guest: st.info("ℹ️ L'anàlisi de vent i convergència està fixada a **925 hPa**.")
            
            with st.spinner("Carregant dades del mapa..."):
                map_data, error_map = carregar_dades_mapa(nivell_sel, hourly_index_sel)
            
            if error_map: 
                st.error(f"Error en carregar el mapa: {error_map}")
            elif map_data:
                fig = crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str, selected_extent)
                st.pyplot(fig); plt.close(fig)
                with st.container(border=True):
                    ui_explicacio_alertes()

        elif map_key in ["vent_700", "vent_300"]:
            nivell = 700 if map_key == "vent_700" else 300
            variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            with st.spinner("Carregant dades del mapa de vent..."):
                map_data, error_map = carregar_dades_mapa_base(variables, hourly_index_sel)

            if error_map: 
                st.error(f"Error en carregar el mapa: {error_map}")
            elif map_data: 
                fig = crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str, selected_extent)
                st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown("#### Visualització en Temps Real (Meteociel)")
        
        now_local = datetime.now(TIMEZONE)
        if 7 <= now_local.hour < 21:
            sat_url = f"https://modeles20.meteociel.fr/satellite/animsatviscolmtgsp.gif?ver={int(time.time())}"
            caption = "Satèl·lit Visible (NE Peninsular)"
        else:
            sat_url = f"https://modeles20.meteociel.fr/satellite/animsatirmtgsp.gif?ver={int(time.time())}"
            caption = "Satèl·lit Infraroig (NE Peninsular)"
        
        st.image(sat_url, caption=caption, use_container_width=True)
        
        with st.container(border=True):
            ui_info_desenvolupament_tempesta()
            
# ================= INICI DEL NOU BLOC DE CODI =================

# Diccionari complet amb els codis d'estació de Meteoclimatic per a cada capital de comarca
SMC_STATION_CODES = {
    'Amposta': 'D5', 'Balaguer': 'C9', 'Banyoles': 'UB', 'Barcelona': 'X4',
    'Berga': 'C8', 'Cervera': 'CE', 'El Pont de Suert': 'C7', 'El Vendrell': 'TT',
    'Falset': 'T5', 'Figueres': 'UF', 'Gandesa': 'T9', 'Girona': 'UG',
    'Granollers': 'XN', 'Igualada': 'C6', 'La Bisbal d\'Empordà': 'UH',
    'La Seu d\'Urgell': 'U7', 'Les Borges Blanques': 'C5', 'Lleida': 'UL',
    'Manresa': 'C4', 'Mataró': 'XL', 'Moià': 'WM', 'Mollerussa': 'U4',
    'Montblanc': 'T2', 'Móra d\'Ebre': 'T8', 'Olot': 'U6', 'Prats de Lluçanès': 'WP',
    'Puigcerdà': 'U8', 'Reus': 'T4', 'Ripoll': 'U5', 'Sant Feliu de Llobregat': 'WZ',
    'Santa Coloma de Farners': 'U1', 'Solsona': 'C3', 'Sort': 'U2',
    'Tarragona': 'UT', 'Tàrrega': 'U3', 'Terrassa': 'X2', 'Tortosa': 'D4',
    'Tremp': 'C2', 'Valls': 'T3', 'Vic': 'W2', 'Vielha': 'VA',
    'Vilafranca del Penedès': 'X8', 'Vilanova i la Geltrú': 'XD'
}

@st.cache_data(ttl=600)
def obtenir_dades_estacio_smc():
    """
    Obté les últimes dades de totes les estacions de la XEMA (SMC).
    """
    try:
        api_key = st.secrets["SMC_API_KEY"]
    except KeyError:
        # Aquest missatge només apareixerà si no has configurat la clau API
        st.error("Error de configuració: Falta la clau 'SMC_API_KEY' als secrets de Streamlit.")
        st.info("Per solucionar-ho, sol·licita una clau a https://apidocs.meteo.cat/ i afegeix-la als secrets del teu projecte.")
        return None

    url = "https://api.meteo.cat/xema/v1/observacions/mesurades/ultimes"
    headers = {"X-Api-Key": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error de xarxa en contactar amb l'API de l'SMC. Detalls: {e}")
        return None

# ================= SUBSTITUEIX LA TEVA FUNCIÓ PER AQUESTA =================

def ui_pestanya_estacions_meteorologiques():
    st.markdown("#### Dades en Temps Real (Xarxa d'Estacions de l'SMC)")

    # ---> INICI DE LA MILLORA: Comprovem si la clau API existeix <---
    if "SMC_API_KEY" in st.secrets and st.secrets["SMC_API_KEY"]:
        # Si la clau existeix, executem la lògica normal per mostrar les dades
        st.caption("Dades oficials de la Xarxa d'Estacions Meteorològiques Automàtiques (XEMA) del Servei Meteorològic de Catalunya.")

        with st.spinner("Carregant dades de la XEMA..."):
            dades_xema = obtenir_dades_estacio_smc()

        if not dades_xema:
            st.warning("No s'han pogut carregar les dades de les estacions de l'SMC en aquests moments.")
            return

        col1, col2 = st.columns([0.6, 0.4], gap="large")

        with col1:
            st.markdown("##### Mapa d'Ubicacions")
            fig, ax = crear_mapa_base(MAP_EXTENT)
            for ciutat, coords in CIUTATS_CATALUNYA.items():
                if ciutat in SMC_STATION_CODES:
                    lon, lat = coords['lon'], coords['lat']
                    ax.plot(lon, lat, 'o', color='darkblue', markersize=8, markeredgecolor='white', transform=ccrs.PlateCarree(), zorder=10)
                    ax.text(lon + 0.03, lat, ciutat, fontsize=7, transform=ccrs.PlateCarree(), zorder=11, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.markdown("##### Dades de l'Estació")
            ciutat_seleccionada = st.selectbox("Selecciona una capital de comarca:", options=sorted(SMC_STATION_CODES.keys()))

            if ciutat_seleccionada:
                station_code = SMC_STATION_CODES.get(ciutat_seleccionada)
                dades_estacio = next((item for item in dades_xema if item.get("codi") == station_code), None)

                if dades_estacio:
                    nom_estacio = dades_estacio.get("nom", "N/A")
                    data_lectura = dades_estacio.get("data", "N/A").replace("T", " ").replace("Z", "")
                    variables = {var['codi']: var['valor'] for var in dades_estacio.get('variables', [])}
                    temp = variables.get(32, "--")
                    humitat = variables.get(33, "--")
                    pressio = variables.get(35, "--")
                    vel_vent = variables.get(30, "--")
                    dir_vent = variables.get(31, "--")
                    precip = variables.get(34, "--")
                    rafaga = variables.get(2004, "--")

                    st.info(f"**Estació:** {nom_estacio} | **Lectura:** {data_lectura} UTC")
                    c1, c2 = st.columns(2)
                    c1.metric("Temperatura", f"{temp} °C")
                    c2.metric("Humitat", f"{humitat} %")
                    st.metric("Pressió atmosfàrica", f"{pressio} hPa")
                    st.metric("Vent", f"{dir_vent}° a {vel_vent} km/h (Ràfega: {rafaga} km/h)")
                    st.metric("Precipitació (30 min)", f"{precip} mm")
                    st.markdown(f"🔗 [Veure a la web de l'SMC](https://www.meteo.cat/observacions/xema/dades?codi={station_code})", unsafe_allow_html=True)
                else:
                    st.error("No s'han trobat dades recents per a aquesta estació a la resposta de l'SMC.")

    else:
        # Si la clau NO existeix, mostrem el cartell informatiu
        st.info(
            "🚧 **Pestanya en Desenvolupament**\n\n"
            "Aquesta secció està esperant la validació de la clau d'accés a les dades oficials del Servei Meteorològic de Catalunya (SMC).\n\n"
            "Tornarà a estar operativa pròximament. Gràcies per la paciència!",
            icon="🚧"
        )
    # ---> FI DE LA MILLORA <---

# =======================================================================

def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades AROME via Open-Meteo | Imatges via Meteologix & Rainviewer | IA per Google Gemini.</p>", unsafe_allow_html=True)

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
        
        global progress_placeholder; progress_placeholder = st.empty()
        if is_guest:
            tab_mapes, tab_estacions, tab_vertical = st.tabs(["Anàlisi de Mapes", "Estacions Meteorològiques", "Anàlisi Vertical"])
            with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
            with tab_estacions: ui_pestanya_estacions_meteorologiques()
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
            tab_ia, tab_xat, tab_mapes, tab_estacions, tab_vertical = st.tabs(["Assistent MeteoIA", chat_tab_label, "Anàlisi de Mapes", "Estacions Meteorològiques", "Anàlisi Vertical"])
            with tab_ia: ui_pestanya_ia_final(data_tuple, hourly_index_sel, poble_sel, timestamp_str)
            with tab_xat: ui_pestanya_xat(chat_history)
            with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
            with tab_estacions: ui_pestanya_estacions_meteorologiques()
            with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
        ui_peu_de_pagina()

if __name__ == "__main__":
    main()
