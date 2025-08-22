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

# --- Funcions de càrrega de dades ---
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
        prof = None 
        heights_agl = heights - heights[0]
        
        with parcel_lock:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
            sbcape, sbcin = mpcalc.cape_cin(p, T, Td, prof)
            params_calc['SBCAPE'] = sbcape.to('J/kg').m
            params_calc['SBCIN'] = sbcin.to('J/kg').m

            mucape_cin_result = mpcalc.most_unstable_cape_cin(p, T, Td)
            params_calc['MUCAPE'] = mucape_cin_result[0].to('J/kg').m if isinstance(mucape_cin_result, tuple) else np.nan
            
            mlcape_cin_result = mpcalc.mixed_layer_cape_cin(p, T, Td)
            params_calc['MLCAPE'] = mlcape_cin_result[0].to('J/kg').m if isinstance(mlcape_cin_result, tuple) else np.nan
            
            li_result = mpcalc.lifted_index(p, T.to('degC'), Td.to('degC'))
            params_calc['LI'] = li_result[0].to('delta_degC').m if isinstance(li_result, tuple) else np.nan

        try:
            h_agl_m = heights_agl.m
            p_m = p.m
            if 3000 <= h_agl_m[-1]:
                p_3km_agl = np.interp(3000, h_agl_m, p_m) * units.hPa
                cape_0_3, _ = mpcalc.cape_cin(p, T, Td, prof, top=p_3km_agl)
                params_calc['CAPE_0-3km'] = cape_0_3.to('J/kg').m
            else:
                params_calc['CAPE_0-3km'] = np.nan
        except: 
            params_calc['CAPE_0-3km'] = np.nan
            
        try:
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0], max_iters=50)
            params_calc['LCL_p'] = lcl_p; params_calc['LCL_Hgt'] = np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1])
        except: 
            params_calc['LCL_p'], params_calc['LCL_Hgt'] = np.nan * units.hPa, np.nan
        
        try:
            lfc_result = mpcalc.lfc(p, T, Td, which='surface', parcel_profile=prof)
            if isinstance(lfc_result, tuple) and len(lfc_result) == 2:
                lfc_p = lfc_result[0]
                params_calc['LFC_p'] = lfc_p
                params_calc['LFC_Hgt'] = mpcalc.pressure_to_height_std(lfc_p).to('m').m if hasattr(lfc_p, 'm') else np.nan
            else: 
                params_calc['LFC_p'], params_calc['LFC_Hgt'] = np.nan * units.hPa, np.nan
        except: 
            params_calc['LFC_p'], params_calc['LFC_Hgt'] = np.nan * units.hPa, np.nan
        
        try:
            el_result = mpcalc.el(p, T, Td, which='surface', parcel_profile=prof)
            if isinstance(el_result, tuple) and len(el_result) == 2:
                el_p = el_result[0]
                params_calc['EL_p'] = el_p
                params_calc['EL_Hgt'] = mpcalc.pressure_to_height_std(el_p).to('m').m if hasattr(el_p, 'm') else np.nan
            else: 
                params_calc['EL_p'], params_calc['EL_Hgt'] = np.nan * units.hPa, np.nan
        except: 
            params_calc['EL_p'], params_calc['EL_Hgt'] = np.nan * units.hPa, np.nan
        
        try:
            p_frz = np.interp(0, T.to('degC').m[::-1], p.m[::-1]) * units.hPa
            params_calc['FRZG_Lvl_p'] = p_frz
        except: 
            params_calc['FRZG_Lvl_p'] = np.nan * units.hPa

        try:
            right_mover, left_mover, mean_wind = mpcalc.bunkers_storm_motion(p, u, v, heights)
            params_calc['RM'] = right_mover; params_calc['LM'] = left_mover; params_calc['Mean_Wind'] = mean_wind
        except: 
            params_calc.update({'RM': None, 'LM': None, 'Mean_Wind': None})
        
        depths = {'0-1km': 1000 * units.m, '0-6km': 6000 * units.m}
        for name, depth in depths.items():
            try: 
                params_calc[f'BWD_{name}'] = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u, v, height=heights_agl, depth=depth)).to('kt').m
            except: 
                params_calc[f'BWD_{name}'] = np.nan
        
        if params_calc.get('RM') is not None:
            try:
                u_storm, v_storm = params_calc['RM']
                params_calc['SR_Wind'] = mpcalc.wind_speed(u - u_storm, v - v_storm).to('kt')
                srh_0_1, _, _ = mpcalc.storm_relative_helicity(heights_agl, u, v, depth=1000 * units.m, storm_u=u_storm, storm_v=v_storm)
                srh_0_3, _, _ = mpcalc.storm_relative_helicity(heights_agl, u, v, depth=3000 * units.m, storm_u=u_storm, storm_v=v_storm)
                params_calc['SRH_0-1km'] = srh_0_1.to('m**2/s**2').m
                params_calc['SRH_0-3km'] = srh_0_3.to('m**2/s**2').m
                
                eff_p_bottom, eff_p_top = mpcalc.effective_inflow_layer(p, T, Td, prof)
                if hasattr(eff_p_bottom, 'm'):
                    eff_h_bottom = np.interp(eff_p_bottom.m, p.m[::-1], heights_agl.m[::-1]) * units.m
                    eff_h_top = np.interp(eff_p_top.m, p.m[::-1], heights_agl.m[::-1]) * units.m
                    eff_depth = eff_h_top - eff_h_bottom
                    params_calc['EBWD'] = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u, v, height=heights_agl, bottom=eff_h_bottom, depth=eff_depth)).to('kt').m
                    esrh, _, _ = mpcalc.storm_relative_helicity(heights_agl, u, v, bottom=eff_h_bottom, depth=eff_depth, storm_u=u_storm, storm_v=v_storm)
                    params_calc['ESRH'] = esrh.to('m**2/s**2').m
                else: 
                    params_calc.update({'EBWD': np.nan, 'ESRH': np.nan})
            except: 
                params_calc.update({'SR_Wind':None, 'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan, 'EBWD': np.nan, 'ESRH': np.nan})
        else: 
            params_calc.update({'SR_Wind': None, 'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan, 'EBWD': np.nan, 'ESRH': np.nan})
            
        return ((p, T, Td, u, v, heights, prof), params_calc), None
    except Exception as e: 
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
    fig = plt.figure(dpi=150, figsize=(10, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.5, 1], wspace=0.3)
    ax_hodo = fig.add_subplot(gs[0])
    ax_params = fig.add_subplot(gs[1])
    fig.suptitle(titol, weight='bold', fontsize=16)

    # --- Configuració del gràfic principal de l'Hodògraf ---
    h = Hodograph(ax_hodo, component_range=80.)
    h.add_grid(increment=20, color='gray', linestyle='--')
    
    # Dibuixar la línia de l'hodògraf amb colors per altitud
    intervals = np.array([0, 1, 3, 6, 9, 12]) * units.km
    colors = ['red', 'blue', 'green', 'purple', 'gold']
    h.plot_colormapped(u.to('kt'), v.to('kt'), heights, intervals=intervals, colors=colors, linewidth=2)

    # Ressaltar la capa efectiva (si existeix)
    if not np.isnan(params_calc.get('ESRH', np.nan)):
        try:
            heights_agl = heights - heights[0]
            eff_p_bottom, eff_p_top = mpcalc.effective_inflow_layer(p, T, Td, prof)
            mask = (p >= eff_p_top) & (p <= eff_p_bottom)
            h.plot(u[mask].to('kt'), v[mask].to('kt'), color='cyan', linewidth=6, alpha=0.6, label='Capa Efectiva')
        except: pass # Si falla, simplement no es dibuixa

    # Marcadors d'altitud
    for alt_km in [1, 3, 6, 9]:
        try:
            idx = np.argmin(np.abs(heights - alt_km * 1000 * units.m))
            ax_hodo.text(u[idx].to('kt').m + 1, v[idx].to('kt').m + 1, f'{alt_km}km', fontsize=9, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
        except: continue
    
    # Vectors de moviment de tempesta i vent mitjà
    motion = {'RM': params_calc.get('RM'), 'LM': params_calc.get('LM'), 'Vent Mitjà': params_calc.get('Mean_Wind')}
    if all(v is not None for v in motion.values()):
        for name, vec in motion.items():
            u_comp, v_comp = vec[0].to('kt').m, vec[1].to('kt').m
            marker = 's' if 'Mitjà' in name else 'o'
            ax_hodo.plot(u_comp, v_comp, marker=marker, color='black', markersize=8, fillstyle='none', mew=1.5)
            ax_hodo.text(u_comp + 2, v_comp + 2, name, ha='left', va='bottom', weight='bold', fontsize=10)

    # Vector de cisallament 0-6km
    try:
        shear_vec = mpcalc.bulk_shear(p, u, v, height=heights - heights[0], depth=6000 * units.m)
        ax_hodo.arrow(0, 0, shear_vec[0].to('kt').m, shear_vec[1].to('kt').m,
                      color='black', linestyle='--', alpha=0.7, head_width=2, length_includes_head=True)
    except: pass
    ax_hodo.set_xlabel('U-Component (nusos)')
    ax_hodo.set_ylabel('V-Component (nusos)')

    # --- Taula de Paràmetres a la dreta ---
    ax_params.axis('off')
    def get_color(value, thresholds):
        if pd.isna(value): return "#808080"
        colors = ["#808080", "#28a745", "#ffc107", "#fd7e14", "#dc3545"]
        thresholds = sorted(thresholds)
        for i, threshold in enumerate(thresholds):
            if value < threshold: return colors[i]
        return colors[-1]

    THRESHOLDS = {'BWD': (10, 20, 30, 40), 'SRH': (100, 150, 250, 400)}
    
    y = 0.95
    ax_params.text(0, y, "Moviment (dir/kts)", ha='left', weight='bold', fontsize=11); y-=0.07
    for name, vec in motion.items():
        if vec is not None:
            speed = mpcalc.wind_speed(*vec).to('kt').m; direction = mpcalc.wind_direction(*vec).to('deg').m
            ax_params.text(0.05, y, f"{name}:"); ax_params.text(0.95, y, f"{direction:.0f}°/{speed:.0f} kts", ha='right')
        else:
            ax_params.text(0.05, y, f"{name}:"); ax_params.text(0.95, y, "---", ha='right')
        y-=0.06

    y-=0.03
    ax_params.text(0, y, "Cisallament (nusos)", ha='left', weight='bold', fontsize=11); y-=0.07
    for key, label in [('0-1km', '0-1 km'), ('0-6km', '0-6 km'), ('EBWD', 'Efectiu')]:
        val = params_calc.get(key if key == 'EBWD' else f'BWD_{key}', np.nan)
        color = get_color(val, THRESHOLDS['BWD'])
        ax_params.text(0.05, y, f"{label}:"); ax_params.text(0.95, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', weight='bold', color=color)
        y-=0.06

    y-=0.03
    ax_params.text(0, y, "Helicitat (m²/s²)", ha='left', weight='bold', fontsize=11); y-=0.07
    for key, label in [('0-1km', '0-1 km'), ('0-3km', '0-3 km'), ('ESRH', 'Efectiva')]:
        val = params_calc.get(key if key == 'ESRH' else f'SRH_{key}', np.nan)
        color = get_color(val, THRESHOLDS['SRH'])
        ax_params.text(0.05, y, f"{label}:"); ax_params.text(0.95, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', weight='bold', color=color)
        y-=0.06

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

    THRESHOLDS = {'SBCAPE': (100, 500, 1500, 2500), 'MUCAPE': (100, 500, 1500, 2500),
                  'MLCAPE': (50, 250, 1000, 2000), 'CAPE_0-3km': (25, 75, 150, 250),
                  'SBCIN': (0, -25, -75, -150), 'LI': (0, -2, -5, -8),
                  'BWD_0-6km': (10, 20, 30, 40), 'SRH_0-3km': (100, 150, 250, 400),
                  'ESRH': (100, 150, 250, 400)}

    def styled_metric(label, value, unit, param_key, precision=0, reverse_colors=False):
        color = get_color(value, THRESHOLDS.get(param_key, []), reverse_colors)
        val_str = f"{value:.{precision}f}" if not pd.isna(value) else "---"
        st.markdown(f"""
            <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34;">
                <span style="font-size: 0.8em; color: #FAFAFA;">{label} ({unit})</span><br>
                <strong style="font-size: 1.6em; color: {color};">{val_str}</strong>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("##### Paràmetres Termodinàmics")
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCAPE", params.get('SBCAPE', np.nan), "J/kg", 'SBCAPE')
    with cols[1]: styled_metric("MUCAPE", params.get('MUCAPE', np.nan), "J/kg", 'MUCAPE')
    with cols[2]: styled_metric("MLCAPE", params.get('MLCAPE', np.nan), "J/kg", 'MLCAPE')
    
    st.markdown("")
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCIN", params.get('SBCIN', np.nan), "J/kg", 'SBCIN', reverse_colors=True)
    with cols[1]: styled_metric("LI", params.get('LI', np.nan), "°C", 'LI', precision=1, reverse_colors=True)
    with cols[2]: styled_metric("CAPE 0-3km", params.get('CAPE_0-3km', np.nan), "J/kg", 'CAPE_0-3km')

    st.divider()
    st.markdown("##### Paràmetres de Cisallament i Helicitat")
    cols = st.columns(3)
    with cols[0]: styled_metric("BWD 0-6km", params.get('BWD_0-6km', np.nan), "nusos", 'BWD_0-6km')
    with cols[1]: styled_metric("SRH 0-3km", params.get('SRH_0-3km', np.nan), "m²/s²", 'SRH_0-3km')
    with cols[2]: styled_metric("ESRH", params.get('ESRH', np.nan), "m²/s²", 'ESRH')

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        p, T, Td, u, v, heights, prof = sounding_data
        
        col1, col2 = st.columns(2, gap="large")

        with col1:
            fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Sondeig Vertical\n{poble_sel}")
            st.pyplot(fig_skewt, use_container_width=True)
            plt.close(fig_skewt)
            with st.container(border=True):
                ui_caixa_parametres_sondeig(params_calculats)
        
        with col2:
            fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat\n{poble_sel}")
            st.pyplot(fig_hodo, use_container_width=True)
            plt.close(fig_hodo)
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

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
- **(Pregunta per Reus, amb convergència a prop però sondeig molt estable):** "Avui per la teva zona teniu un bon disparador amb aquesta convergència, però el sondeig està molt estable, gairebé no hi ha 'benzina' (CAPE). Així que, tot i la convergència, el més probable és que només es formin alguns núvols sin més conseqüències. Un dia tranquil."
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
                
                contingut_per_ia = [img_mapa]
                
                if data_tuple: 
                    sounding_data, params_calculats = data_tuple
                    p, T, Td, u, v, heights, prof = sounding_data
                    
                    fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Sondeig Vertical\n{poble_sel}")
                    buf_skewt = io.BytesIO(); fig_skewt.savefig(buf_skewt, format='png', dpi=150, bbox_inches='tight'); buf_skewt.seek(0); img_skewt = Image.open(buf_skewt); plt.close(fig_skewt); contingut_per_ia.append(img_skewt)
                    
                    fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat\n{poble_sel}")
                    buf_hodo = io.BytesIO(); fig_hodo.savefig(buf_hodo, format='png', dpi=150, bbox_inches='tight'); buf_hodo.seek(0); img_hodo = Image.open(buf_hodo); plt.close(fig_hodo); contingut_per_ia.append(img_hodo)

                prompt_context = f"PREGUNTA DE L'USUARI: '{prompt_usuari}'"
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

if __name__ == "__main__":
    main()
