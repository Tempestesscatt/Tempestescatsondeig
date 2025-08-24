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
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever")

# --- Clients API ---
parcel_lock = threading.Lock()
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- Constants per Catalunya ---
API_URL_CAT = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_CAT = pytz.timezone('Europe/Madrid')
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
MAP_EXTENT_CAT = [0, 3.5, 40.4, 43]
PRESS_LEVELS_AROME = sorted([1000, 950, 925, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
MAP_ZOOM_LEVELS_CAT = {'Catalunya (Complet)': MAP_EXTENT_CAT, 'Nord-est (Girona)': [1.8, 3.4, 41.7, 42.6], 'Sud (Tarragona i Ebre)': [0.2, 1.8, 40.5, 41.4], 'Ponent i Pirineu (Lleida)': [0.4, 1.9, 41.4, 42.6], 'Àrea Metropolitana (BCN)': [1.7, 2.7, 41.2, 41.8]}

# --- Constants per Tornado Alley ---
API_URL_USA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_USA = pytz.timezone('America/Chicago')
USA_CITIES = {
    'Dallas, TX': {'lat': 32.7767, 'lon': -96.7970}, 'Oklahoma City, OK': {'lat': 35.4676, 'lon': -97.5164},
    'Kansas City, MO': {'lat': 39.0997, 'lon': -94.5786}, 'Omaha, NE': {'lat': 41.2565, 'lon': -95.9345},
    'Wichita, KS': {'lat': 37.6872, 'lon': -97.3301}, 'Tulsa, OK': {'lat': 36.1540, 'lon': -95.9928},
}
MAP_EXTENT_USA = [-105, -85, 30, 48]
PRESS_LEVELS_GFS = sorted([1000, 975, 950, 925, 900, 850, 800, 750, 700, 600, 500, 400, 300, 250, 200, 100], reverse=True)

# --- Constants Generals ---
USERS_FILE = 'users.json'
RATE_LIMIT_FILE = 'rate_limits.json'
CHAT_FILE = 'chat_history.json'


# --- Funcions auxiliars (Compartides) ---
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
    st.markdown("<h1 style='text-align: center;'>Tempestes.cat</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Controlem quina vista es mostra (login o registre) amb st.session_state
    if 'view' not in st.session_state:
        st.session_state.view = 'login'

    # --- VISTA D'INICI DE SESSIÓ (PER DEFECTE) ---
    if st.session_state.view == 'login':
        st.subheader("Inicia Sessió")
        with st.form("login_form"):
            username = st.text_input("Nom d'usuari", key="login_user")
            password = st.text_input("Contrasenya", type="password", key="login_pass")
            
            if st.form_submit_button("Entra", use_container_width=True, type="primary"):
                users = load_json_file(USERS_FILE)
                if username in users and users[username] == get_hashed_password(password):
                    st.session_state.update({'logged_in': True, 'username': username, 'guest_mode': False})
                    st.rerun()
                else:
                    st.error("Nom d'usuari o contrasenya incorrectes.")
        
        # Botó per canviar a la vista de registre
        if st.button("No tens un compte? Registra't aquí"):
            st.session_state.view = 'register'
            st.rerun()

    # --- VISTA DE REGISTRE ---
    elif st.session_state.view == 'register':
        st.subheader("Crea un nou compte")
        with st.form("register_form"):
            new_username = st.text_input("Tria un nom d'usuari", key="reg_user")
            new_password = st.text_input("Tria una contrasenya", type="password", key="reg_pass")
            
            if st.form_submit_button("Registra'm", use_container_width=True):
                users = load_json_file(USERS_FILE)
                if not new_username or not new_password:
                    st.error("El nom d'usuari i la contrasenya no poden estar buits.")
                elif new_username in users:
                    st.error("Aquest nom d'usuari ja existeix.")
                elif len(new_password) < 6:
                    st.error("La contrasenya ha de tenir com a mínim 6 caràcters.")
                else:
                    users[new_username] = get_hashed_password(new_password)
                    save_json_file(users, USERS_FILE)
                    st.success("Compte creat amb èxit! Ara pots iniciar sessió.")
        
        # Botó per tornar a la vista d'inici de sessió
        if st.button("Ja tens un compte? Inicia sessió"):
            st.session_state.view = 'login'
            st.rerun()
    
    st.divider()
    st.markdown("<p style='text-align: center;'>O si ho prefereixes...</p>", unsafe_allow_html=True)

    # Botó per entrar com a convidat
    if st.button("Entrar com a Convidat (simple i ràpid)", use_container_width=True, type="secondary"):
        st.session_state.update({'guest_mode': True, 'logged_in': True})
        st.rerun()

# --- Funcions Base de Càlcul i Gràfics (Compartides) ---

def calcular_convergencia_puntual(map_data, lat_sel, lon_sel):
    """
    Calcula la convergència en un punt específic (lat_sel, lon_sel)
    a partir de les dades d'una graella de model (map_data).

    Retorna:
        El valor de la convergència (escalat per 1e5) o np.nan si no es pot calcular.
    """
    if not map_data or 'lons' not in map_data or len(map_data['lons']) < 4:
        return np.nan

    try:
        lons, lats = map_data['lons'], map_data['lats']
        speed_data, dir_data = map_data['speed_data'], map_data['dir_data']

        # Crear una graella fina per a la interpolació
        grid_lon, grid_lat = np.meshgrid(
            np.linspace(min(lons), max(lons), 100),
            np.linspace(min(lats), max(lats), 100)
        )

        # Calcular components del vent i interpolar-los
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')

        # Calcular la divergència a tota la graella
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)
        
        # La convergència és la divergència negativa, escalada per a una millor visualització.
        convergence_scaled = -divergence.to('1/s').magnitude * 1e5

        # Trobar l'índex de la graella més proper al punt seleccionat
        dist_sq = (grid_lat - lat_sel)**2 + (grid_lon - lon_sel)**2
        min_dist_idx = np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)

        # Retornar el valor de convergència en aquest punt
        return convergence_scaled[min_dist_idx]
    except Exception as e:
        print(f"Error calculant la convergència puntual: {e}")
        return np.nan
        

def calcular_li_manual(p, T, prof):
    """Cálculo manual del Lifted Index"""
    try:
        # Encontrar la presión de 500 hPa
        idx_500 = np.argmin(np.abs(p.m - 500))
        if idx_500 < len(T) and idx_500 < len(prof):
            T_500_ambient = T[idx_500].m
            T_500_parcel = prof[idx_500]
            li = T_500_ambient - T_500_parcel
            return li
        return np.nan
    except:
        return np.nan


def processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile):
    """
    Processa les dades de sondeig brutes per calcular un conjunt complet de paràmetres
    termodinàmics i cinemàtics, incloent paràmetres de capa efectiva.
    """
    # --- 1. PREPARACIÓ I NETEJA DE DADES ---
    if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
    p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC
    Td = np.array(Td_profile) * units.degC; u = np.array(u_profile) * units('m/s')
    v = np.array(v_profile) * units('m/s'); heights = np.array(h_profile) * units.meter
    valid_indices = ~np.isnan(p.m) & ~np.isnan(T.m) & ~np.isnan(Td.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
    p, T, Td, u, v, heights = p[valid_indices], T[valid_indices], Td[valid_indices], u[valid_indices], v[valid_indices], heights[valid_indices]
    if len(p) < 3: return None, "No hi ha prou dades vàlides."
    sort_idx = np.argsort(p.m)[::-1]
    p, T, Td, u, v, heights = p[sort_idx], T[sort_idx], Td[sort_idx], u[sort_idx], v[sort_idx], heights[sort_idx]
    params_calc = {}; prof = None; heights_agl = heights - heights[0]

    # --- 2. CÀLCULS TERMODINÀMICS ---
    with parcel_lock:
        try: prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        except Exception as e: return None, f"Error crític en el perfil de la parcel·la: {e}"
        try:
            sbcape, sbcin = mpcalc.cape_cin(p, T, Td, prof)
            params_calc['SBCAPE'] = float(sbcape.m); params_calc['SBCIN'] = float(sbcin.m)
            params_calc['MAX_UPDRAFT'] = np.sqrt(2 * float(sbcape.m)) if sbcape.m > 0 else 0.0
        except: params_calc.update({'SBCAPE': np.nan, 'SBCIN': np.nan, 'MAX_UPDRAFT': np.nan})
        try:
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=300 * units.hPa)
            params_calc['MUCAPE'] = float(mucape.m); params_calc['MUCIN'] = float(mucin.m)
        except: params_calc.update({'MUCAPE': np.nan, 'MUCIN': np.nan})
        try:
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=100 * units.hPa)
            params_calc['MLCAPE'] = float(mlcape.m); params_calc['MLCIN'] = float(mlcin.m)
        except: params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
        try:
            li = mpcalc.lifted_index(p, T, prof)
            params_calc['LI'] = float(li.m)
        except: params_calc['LI'] = np.nan
        try:
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0])
            params_calc['LCL_p'] = float(lcl_p.m); params_calc['LCL_Hgt'] = float(np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1]))
        except: params_calc.update({'LCL_p': np.nan, 'LCL_Hgt': np.nan})
        try:
            lfc_p, _ = mpcalc.lfc(p, T, Td, prof)
            params_calc['LFC_p'] = float(lfc_p.m); params_calc['LFC_Hgt'] = float(np.interp(lfc_p.m, p.m[::-1], heights_agl.m[::-1]))
        except: params_calc.update({'LFC_p': np.nan, 'LFC_Hgt': np.nan})
        try:
            pwat = mpcalc.precipitable_water(p, Td)
            params_calc['PWAT'] = float(pwat.to('mm').m)
        except: params_calc['PWAT'] = np.nan
        
    # --- 3. CÀLCULS CINEMÀTICS ---
    try:
        rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p, u, v, heights)
        params_calc['RM'] = (float(rm[0].m), float(rm[1].m))
        params_calc['LM'] = (float(lm[0].m), float(lm[1].m))
        params_calc['Mean_Wind'] = (float(mean_wind[0].m), float(mean_wind[1].m))
    except Exception:
        params_calc.update({'RM': (np.nan, np.nan), 'LM': (np.nan, np.nan), 'Mean_Wind': (np.nan, np.nan)})

    for name, depth_m in [('0-1km', 1000), ('0-6km', 6000)]:
        try:
            bwd_u, bwd_v = mpcalc.bulk_shear(p, u, v, height=heights, depth=depth_m * units.meter)
            params_calc[f'BWD_{name}'] = float(mpcalc.wind_speed(bwd_u, bwd_v).to('kt').m)
        except: params_calc[f'BWD_{name}'] = np.nan
    
    if not np.isnan(params_calc.get('RM', (np.nan,))[0]):
        u_storm, v_storm = params_calc['RM'][0] * units('m/s'), params_calc['RM'][1] * units('m/s')
        for name, depth_m in [('0-1km', 1000), ('0-3km', 3000)]:
            try:
                srh = mpcalc.storm_relative_helicity(heights, u, v, depth=depth_m * units.meter, storm_u=u_storm, storm_v=v_storm)[0]
                params_calc[f'SRH_{name}'] = float(srh.m)
            except: params_calc[f'SRH_{name}'] = np.nan
    else:
        params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})

    try:
        eff_bottom, eff_top = mpcalc.effective_inflow_layer(p, T, Td, heights=heights_agl)
        ebwd_u, ebwd_v = mpcalc.bulk_shear(p, u, v, height=heights_agl, bottom=eff_bottom, top=eff_top)
        params_calc['EBWD'] = float(mpcalc.wind_speed(ebwd_u, ebwd_v).to('kt').m)
        if not np.isnan(params_calc.get('RM', (np.nan,))[0]):
            u_storm_eff, v_storm_eff = params_calc['RM'][0] * units('m/s'), params_calc['RM'][1] * units('m/s')
            esrh, _, _ = mpcalc.storm_relative_helicity(heights, u, v, bottom=eff_bottom, top=eff_top, storm_u=u_storm_eff, storm_v=v_storm_eff)
            params_calc['ESRH'] = float(esrh.m)
        else:
            params_calc['ESRH'] = np.nan
    except Exception:
        params_calc.update({'EBWD': np.nan, 'ESRH': np.nan})

    try:
        idx_3km = np.argmin(np.abs(heights_agl.m - 3000))
        cape_0_3, _ = mpcalc.cape_cin(p[:idx_3km+1], T[:idx_3km+1], Td[:idx_3km+1], prof[:idx_3km+1])
        params_calc['CAPE_0-3km'] = float(cape_0_3.m)
    except: params_calc['CAPE_0-3km'] = np.nan

    return ((p, T, Td, u, v, heights, prof), params_calc), None

    

def debug_calculos(p, T, Td, u, v, heights, prof):
    """Función para depurar los cálculos problemáticos"""
    print("=== DEBUG: Cálculos problemáticos ===")
    
    # Debug LI
    try:
        li = mpcalc.lifted_index(p, T, prof)
        print(f"LI raw: {li}")
        print(f"LI type: {type(li)}")
        if hasattr(li, 'm'): print(f"LI.m: {li.m}")
        if hasattr(li, 'magnitude'): print(f"LI.magnitude: {li.magnitude}")
    except Exception as e:
        print(f"LI error: {e}")
    
    # Debug DCAPE
    try:
        dcape = mpcalc.dcape(p, T, Td)
        print(f"DCAPE raw: {dcape}")
        print(f"DCAPE type: {type(dcape)}")
        if hasattr(dcape, 'm'): print(f"DCAPE.m: {dcape.m}")
        if hasattr(dcape, 'magnitude'): print(f"DCAPE.magnitude: {dcape.magnitude}")
    except Exception as e:
        print(f"DCAPE error: {e}")
    
    # Debug SRH (necesita movimiento de tormenta)
    try:
        rm, _, _ = mpcalc.bunkers_storm_motion(p, u, v, heights)
        u_storm, v_storm = rm[0] * units('m/s'), rm[1] * units('m/s')
        srh = mpcalc.storm_relative_helicity(heights, u, v, depth=1000 * units.meter, 
                                           storm_u=u_storm, storm_v=v_storm)
        print(f"SRH raw: {srh}")
        print(f"SRH type: {type(srh)}")
        if hasattr(srh, 'm'): print(f"SRH.m: {srh.m}")
        if hasattr(srh, 'magnitude'): print(f"SRH.magnitude: {srh.magnitude}")
    except Exception as e:
        print(f"SRH error: {e}")
    
    print("=====================================")




    
def crear_mapa_base(map_extent, projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, subplot_kw={'projection': projection})
    ax.set_extent(map_extent, crs=ccrs.PlateCarree()) 
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    if projection != ccrs.PlateCarree():
        ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray', zorder=5)
    return fig, ax


def verificar_datos_entrada(p, T, Td, u, v, heights):
    """Verificar que los datos de entrada son válidos"""
    print("=== VERIFICACIÓN DE DATOS ===")
    print(f"Presión: {p.m[:5]}... (len: {len(p)})")
    print(f"Temperatura: {T.m[:5]}... (len: {len(T)})")
    print(f"Punto rocío: {Td.m[:5]}... (len: {len(Td)})")
    print(f"Alturas: {heights.m[:5]}... (len: {len(heights)})")
    
    # Verificar que tenemos datos suficientes para cálculos
    if len(p) < 10:
        print("ADVERTENCIA: Muy pocos niveles para cálculos precisos")
    
    # Verificar rango de temperaturas
    if np.max(T.m) < -20 or np.min(T.m) > 50:
        print("ADVERTENCIA: Temperaturas fuera de rango normal")
    
    print("=============================")

def crear_skewt(p, T, Td, u, v, prof, params_calc, titol):
    fig = plt.figure(dpi=150, figsize=(7, 8))
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.05, 0.85, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)
    
    # Podem mantenir la línia de referència de 0°C, ja que és útil
    skew.ax.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)

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

    # Dibuixar LCL i LFC
    levels_to_plot = {'LCL_p': 'LCL', 'LFC_p': 'LFC'}
    for key, name in levels_to_plot.items():
        p_lvl = params_calc.get(key)
        if p_lvl is not None and not np.isnan(p_lvl):
            p_val = p_lvl.m if hasattr(p_lvl, 'm') else p_lvl
            skew.ax.axhline(p_val, color='blue', linestyle='--', linewidth=1.5)
            skew.ax.text(skew.ax.get_xlim()[1] - 2, p_val, f' {name}', color='blue', ha='right', va='center', fontsize=10, weight='bold')

    # ***** LÍNIES DE CODI PER DIBUIXAR LA LÍNIA HORITZONTAL ELIMINADES D'AQUÍ *****

    skew.ax.legend()
    return fig


def crear_hodograf_avancat(p, u, v, heights, params_calc, titol):
    fig = plt.figure(dpi=150, figsize=(8, 8))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.5, 6], width_ratios=[1.5, 1], hspace=0.4, wspace=0.3)
    ax_barbs = fig.add_subplot(gs[0, :]); ax_hodo = fig.add_subplot(gs[1, 0]); ax_params = fig.add_subplot(gs[1, 1])
    fig.suptitle(titol, weight='bold', fontsize=16)
    
    # --- GRÀFIC DE BARBES DE VENT (Sense canvis) ---
    ax_barbs.set_title("Vent a Nivells Clau", fontsize=11, pad=15)
    heights_agl = heights - heights[0]
    barb_altitudes_km = [1, 3, 6, 9]; barb_altitudes_m = [h * 1000 for h in barb_altitudes_km] * units.m
    u_barbs_list, v_barbs_list = [], []
    for h_m in barb_altitudes_m:
        if h_m <= heights_agl.max():
            u_interp_val = np.interp(h_m.m, heights_agl.m, u.m); v_interp_val = np.interp(h_m.m, heights_agl.m, v.m)
            u_barbs_list.append(u_interp_val); v_barbs_list.append(v_interp_val)
        else:
            u_barbs_list.append(np.nan); v_barbs_list.append(np.nan)
    u_barbs = units.Quantity(u_barbs_list, u.units); v_barbs = units.Quantity(v_barbs_list, v.units)
    speed_kmh_barbs = np.sqrt(u_barbs**2 + v_barbs**2).to('km/h').m
    thresholds_barbs = [10, 40, 70, 100, 130]; colors_barbs = ['dimgrey', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    x_pos = np.arange(len(barb_altitudes_km)); u_barbs_kt = u_barbs.to('kt'); v_barbs_kt = v_barbs.to('kt')
    for i, spd_kmh in enumerate(speed_kmh_barbs):
        if not np.isnan(spd_kmh):
            color_index = np.searchsorted(thresholds_barbs, spd_kmh); color = colors_barbs[color_index]
            ax_barbs.barbs(x_pos[i], 0, u_barbs_kt[i], v_barbs_kt[i], length=8, pivot='middle', color=color)
            ax_barbs.text(x_pos[i], -0.8, f"{spd_kmh:.0f} km/h", ha='center', va='top', fontsize=9, color=color, weight='bold')
        else:
            ax_barbs.text(x_pos[i], 0, "N/A", ha='center', va='center', fontsize=9, color='grey')
    ax_barbs.set_xticks(x_pos); ax_barbs.set_xticklabels([f"{h} km" for h in barb_altitudes_km]); ax_barbs.set_yticks([]); ax_barbs.spines[:].set_visible(False); ax_barbs.tick_params(axis='x', length=0, pad=5); ax_barbs.set_xlim(-0.5, len(barb_altitudes_km) - 0.5); ax_barbs.set_ylim(-1.5, 1.5)
    
    # --- HODÒGRAF (Sense canvis) ---
    h = Hodograph(ax_hodo, component_range=80.); h.add_grid(increment=20, color='gray', linestyle='--')
    intervals = np.array([0, 1, 3, 6, 9, 12]) * units.km; colors_hodo = ['red', 'blue', 'green', 'purple', 'gold']
    h.plot_colormapped(u.to('kt'), v.to('kt'), heights, intervals=intervals, colors=colors_hodo, linewidth=2)
    ax_hodo.set_xlabel('U-Component (nusos)'); ax_hodo.set_ylabel('V-Component (nusos)')
    
    # --- PANELL DE PARÀMETRES (VERSIÓ AMB LÒGICA DE COLORS INTEL·LIGENT) ---
    ax_params.axis('off')
    def degrees_to_cardinal_ca(d):
        dirs = ["Nord", "N-NE", "Nord-est", "E-NE", "Est", "E-SE", "Sud-est", "S-SE", "Sud", "S-SO", "Sud-oest", "O-SO", "Oest", "O-NO", "Nord-oest", "N-NO"]
        return dirs[int(round(d / 22.5)) % 16]
    
    # NOU: Funció específica per al color del "split" de la tempesta
    def get_split_color(angle_diff):
        if pd.isna(angle_diff) or angle_diff < 30: return 'white' # Poc significatiu
        if angle_diff < 60: return '#ffc107' # Groc: Split moderat
        if angle_diff < 90: return '#fd7e14' # Taronja: Split fort (alerta)
        return '#dc3545' # Vermell: Split extrem (perill)

    THRESHOLDS = {'BWD': (10, 20, 30, 40), 'SRH': (100, 150, 250, 400)}
    y = 0.95
    
    motion_data = {
        'M. Dret': params_calc.get('RM'), 
        'M. Esquerre': params_calc.get('LM'), 
        'Es mourà cap a': params_calc.get('Mean_Wind')
    }
    
    ax_params.text(0, y, "Moviment (cap a dir/km/h)", ha='left', weight='bold', fontsize=11); y-=0.1

    # NOU: Lògica per calcular la diferència d'angle
    dir_rm, dir_lm = np.nan, np.nan
    rm_vec = motion_data['M. Dret']; lm_vec = motion_data['M. Esquerre']
    if rm_vec and not pd.isna(rm_vec[0]):
        dir_rm = mpcalc.wind_direction(rm_vec[0] * units('m/s'), rm_vec[1] * units('m/s'), convention='to').m
    if lm_vec and not pd.isna(lm_vec[0]):
        dir_lm = mpcalc.wind_direction(lm_vec[0] * units('m/s'), lm_vec[1] * units('m/s'), convention='to').m

    angle_difference = np.nan
    if not np.isnan(dir_rm) and not np.isnan(dir_lm):
        # Fórmula per a la diferència angular més curta en un cercle de 360°
        angle_difference = 180 - abs(abs(dir_rm - dir_lm) - 180)
    
    split_color = get_split_color(angle_difference)

    for display_name, vec in motion_data.items():
        if vec and not pd.isna(vec[0]):
            u_motion = vec[0] * units('m/s'); v_motion = vec[1] * units('m/s')
            speed = mpcalc.wind_speed(u_motion, v_motion).to('km/h').m
            direction = mpcalc.wind_direction(u_motion, v_motion, convention='to').to('deg').m
            cardinal = degrees_to_cardinal_ca(direction)
            
            # Determinem el color del text
            text_color = 'white'
            if display_name in ['M. Dret', 'M. Esquerre']:
                text_color = split_color

            ax_params.text(0, y, f"{display_name}:", ha='left', va='center', color=text_color)
            ax_params.text(1, y, f"{cardinal} / {speed:.0f} km/h", ha='right', va='center', color=text_color)
        else:
            ax_params.text(0, y, f"{display_name}:", ha='left', va='center')
            ax_params.text(1, y, "---", ha='right', va='center')
        y-=0.1

    # La resta de la funció es queda igual
    y-=0.05
    ax_params.text(0, y, "Cisallament (nusos)", ha='left', weight='bold', fontsize=11); y-=0.1
    # ... (codi de cisallament i helicitat sense canvis)
    def get_color(value, thresholds): # Funció de color genèrica per a la resta
        if pd.isna(value): return "grey"
        colors = ["grey", "#2ca02c", "#ffc107", "#fd7e14", "#dc3545"]
        for i, threshold in enumerate(thresholds):
            if value < threshold: return colors[i]
        return colors[-1]

    for key, label in [('BWD_0-1km', '0-1 km'), ('BWD_0-6km', '0-6 km'), ('EBWD', 'Efectiu')]:
        val = params_calc.get(key, np.nan)
        color = get_color(val, THRESHOLDS['BWD'])
        ax_params.text(0, y, f"{label}:", ha='left', va='center', weight='bold', color=color)
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color)
        y-=0.07

    y-=0.05
    ax_params.text(0, y, "Helicitat (m²/s²)", ha='left', weight='bold', fontsize=11); y-=0.1
    for key, label in [('SRH_0-1km', '0-1 km'), ('SRH_0-3km', '0-3 km'), ('ESRH', 'Efectiva')]:
        val = params_calc.get(key, np.nan)
        color = get_color(val, THRESHOLDS['SRH'])
        ax_params.text(0, y, f"{label}:", ha='left', va='center', weight='bold', color=color)
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color)
        y-=0.07
        
    return fig
    

        

def ui_caixa_parametres_sondeig(params, nivell_conv):
    def get_color(value, thresholds, param_key, reverse_colors=False):
        if pd.isna(value): return "#808080"
        if 'CONV' in param_key:
            colors = ["#28a745", "#808080", "#ffc107", "#fd7e14", "#dc3545"]
            idx = np.searchsorted(thresholds, value)
            return colors[idx]
        colors = ["#808080", "#28a745", "#ffc107", "#fd7e14", "#dc3545"]
        if reverse_colors:
            thresholds = sorted(thresholds, reverse=True)
            colors = list(reversed(colors))
        else:
            thresholds = sorted(thresholds)
        for i, threshold in enumerate(thresholds):
            if value < threshold: return colors[i]
        return colors[-1]

    THRESHOLDS = {
        'SBCAPE': (100, 500, 1500, 2500), 'MUCAPE': (100, 500, 1500, 2500), 
        'MLCAPE': (50, 250, 1000, 2000), 'CAPE_0-3km': (25, 75, 150, 250), 
        'SBCIN': (0, -25, -75, -150), 'MUCIN': (0, -25, -75, -150),
        'MLCIN': (0, -25, -75, -150), 'LI': (0, -2, -5, -8), 
        'PWAT': (20, 30, 40, 50), 'BWD_0-6km': (10, 20, 30, 40), 
        'BWD_0-1km': (5, 10, 15, 20), 'SRH_0-1km': (50, 100, 150, 250),
        'SRH_0-3km': (100, 200, 300, 400),
        f'CONV_{nivell_conv}hPa': [-2, 2, 5, 10]
    }
    
    def styled_metric(label, value, unit, param_key, precision=0, reverse_colors=False):
        if hasattr(value, '__len__') and not isinstance(value, str):
            value = value[0] if len(value) > 0 else np.nan
        color = get_color(value, THRESHOLDS.get(param_key, []), param_key, reverse_colors)
        val_str = f"{value:.{precision}f}" if not pd.isna(value) else "---"
        st.markdown(f"""<div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;"><span style="font-size: 0.8em; color: #FAFAFA;">{label} ({unit})</span><br><strong style="font-size: 1.6em; color: {color};">{val_str}</strong></div>""", unsafe_allow_html=True)

    st.markdown("##### Paràmetres del Sondeig")
    
    # MODIFICAT: Passem 'nivell_conv' a la funció de diagnòstic.
    emoji, descripcio = determinar_emoji_temps(params, nivell_conv)

    # El reste de la funció es queda exactament igual...
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCAPE", params.get('SBCAPE', np.nan), "J/kg", 'SBCAPE')
    with cols[1]: styled_metric("MUCAPE", params.get('MUCAPE', np.nan), "J/kg", 'MUCAPE')
    with cols[2]: styled_metric("MLCAPE", params.get('MLCAPE', np.nan), "J/kg", 'MLCAPE')
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCIN", params.get('SBCIN', np.nan), "J/kg", 'SBCIN', reverse_colors=True)
    with cols[1]: styled_metric("MUCIN", params.get('MUCIN', np.nan), "J/kg", 'MUCIN', reverse_colors=True)
    with cols[2]: styled_metric("MLCIN", params.get('MLCIN', np.nan), "J/kg", 'MLCIN', reverse_colors=True)
    cols = st.columns(3)
    with cols[0]: 
        li_value = params.get('LI', np.nan)
        if hasattr(li_value, '__len__') and not isinstance(li_value, str) and len(li_value) > 0: li_value = li_value[0]
        styled_metric("LI", li_value, "°C", 'LI', precision=1, reverse_colors=True)
    with cols[1]: 
        styled_metric("PWAT", params.get('PWAT', np.nan), "mm", 'PWAT', precision=1)
    with cols[2]:
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 0.8em; color: #FAFAFA;">Tipus de Cel Previst</span>
            <strong style="font-size: 1.8em; line-height: 1;">{emoji}</strong>
            <span style="font-size: 0.8em; color: #E0E0E0;">{descripcio}</span>
        </div>
        """, unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]: styled_metric("LCL", params.get('LCL_Hgt', np.nan), "m", '', precision=0)
    with cols[1]: styled_metric("LFC", params.get('LFC_Hgt', np.nan), "m", '', precision=0)
    with cols[2]: 
        param_key_conv = f'CONV_{nivell_conv}hPa'
        conv_value = params.get(param_key_conv, np.nan)
        styled_metric(f"Conv. {nivell_conv}hPa", conv_value, "10⁻⁵/s", param_key_conv, precision=1)
    cols = st.columns(3)
    with cols[0]: styled_metric("BWD 0-6km", params.get('BWD_0-6km', np.nan), "nusos", 'BWD_0-6km')
    with cols[1]: styled_metric("BWD 0-1km", params.get('BWD_0-1km', np.nan), "nusos", 'BWD_0-1km')
    with cols[2]: styled_metric("CAPE 0-3km", params.get('CAPE_0-3km', np.nan), "J/kg", 'CAPE_0-3km')
    cols = st.columns(3)
    with cols[0]: 
        srh1_value = params.get('SRH_0-1km', np.nan)
        if hasattr(srh1_value, '__len__') and not isinstance(srh1_value, str) and len(srh1_value) > 0: srh1_value = srh1_value[0]
        styled_metric("SRH 0-1km", srh1_value, "m²/s²", 'SRH_0-1km')
    with cols[1]: 
        srh3_value = params.get('SRH_0-3km', np.nan)
        if hasattr(srh3_value, '__len__') and not isinstance(srh3_value, str) and len(srh3_value) > 0: srh3_value = srh3_value[0]
        styled_metric("SRH 0-3km", srh3_value, "m²/s²", 'SRH_0-3km')
    with cols[2]: 
        styled_metric("UPDRAFT", params.get('MAX_UPDRAFT', np.nan), "m/s", 'UPDRAFT', precision=1)
# --- Funcions Específiques per a Catalunya ---

@st.cache_data(ttl=3600)
def carregar_dades_sondeig_cat(lat, lon, hourly_index):
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS_AROME]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": 4}
        response = openmeteo.weather_api(API_URL_CAT, params=params)[0]
        hourly = response.Hourly()
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
        if any(np.isnan(val) for val in sfc_data.values()): return None, "Dades de superfície invàlides."
        
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_AROME) + j).ValuesAsNumpy()[hourly_index] for j in range(len(PRESS_LEVELS_AROME))]
        
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]
        
        for i, p_val in enumerate(PRESS_LEVELS_AROME):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])
        
        return processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
    except Exception as e: return None, f"Error en carregar dades del sondeig AROME: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_base_cat(variables, hourly_index):
    try:
        lats, lons = np.linspace(MAP_EXTENT_CAT[2], MAP_EXTENT_CAT[3], 12), np.linspace(MAP_EXTENT_CAT[0], MAP_EXTENT_CAT[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": 4}
        responses = openmeteo.weather_api(API_URL_CAT, params=params)
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
def carregar_dades_mapa_cat(nivell, hourly_index):
    try:
        if nivell >= 950:
            variables = ["dew_point_2m", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base_cat(variables, hourly_index)
            if error: return None, error
            map_data_raw['dewpoint_data'] = map_data_raw.pop('dew_point_2m')
        else:
            variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base_cat(variables, hourly_index)
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
    map_data, error_map = carregar_dades_mapa_cat(nivell, hourly_index)
    if error_map or not map_data: return CIUTATS_CONVIDAT, "No s'ha pogut determinar les zones de convergència."
    try:
        lons, lats, speed_data, dir_data, dewpoint_data = map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data']
        grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_CAT[0], MAP_EXTENT_CAT[1], 100), np.linspace(MAP_EXTENT_CAT[2], MAP_EXTENT_CAT[3], 100))
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

def crear_mapa_forecast_combinat_cat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent)
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 400), np.linspace(map_extent[2], map_extent[3], 400))
    grid_speed, grid_dewpoint = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]; 
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6,arrowsize=0.5, density= 4.5, zorder=4, transform=ccrs.PlateCarree())
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

def crear_mapa_vents_cat(lons, lats, speed_data, dir_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent)
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 200), np.linspace(map_extent[2], map_extent[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, zorder=3, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label("Velocitat del Vent (km/h)"); ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16); return fig

# --- Funcions Específiques per a Tornado Alley ---

@st.cache_data(ttl=3600)
def carregar_dades_sondeig_usa(lat, lon, hourly_index):
    try:
        h_base = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS_GFS]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "gfs_seamless", "forecast_days": 3}
        response = openmeteo.weather_api(API_URL_USA, params=params)[0]
        hourly = response.Hourly()
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
        if any(np.isnan(val) for val in sfc_data.values()): return None, "Dades de superfície invàlides."

        # Calculem el punt de rosada a partir de la temperatura i la humitat relativa
        sfc_temp_C = sfc_data["temperature_2m"] * units.degC
        sfc_rh_percent = sfc_data["relative_humidity_2m"] * units.percent
        sfc_dew_point = mpcalc.dewpoint_from_relative_humidity(sfc_temp_C, sfc_rh_percent).m
        
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_GFS) + j).ValuesAsNumpy()[hourly_index] for j in range(len(PRESS_LEVELS_GFS))]
        
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_dew_point], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for i, p_val in enumerate(PRESS_LEVELS_GFS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])

        return processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
    except Exception as e:
        return None, f"Error en carregar dades del sondeig GFS: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_base_usa(variables, hourly_index):
    try:
        lats, lons = np.linspace(MAP_EXTENT_USA[2], MAP_EXTENT_USA[3], 10), np.linspace(MAP_EXTENT_USA[0], MAP_EXTENT_USA[1], 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "gfs_seamless", "forecast_days": 3}
        responses = openmeteo.weather_api(API_URL_USA, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            if not r.Hourly(): continue
            hourly_vars = r.Hourly()
            vals = [hourly_vars.Variables(i).ValuesAsNumpy() for i in range(len(variables))]
            if any(hourly_index >= len(v) for v in vals): continue
            current_vals = [v[hourly_index] for v in vals]
            if not any(np.isnan(v) for v in current_vals):
                output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                for i, var in enumerate(variables): output[var].append(current_vals[i])
        if not output["lats"]: return None, "No s'han rebut dades vàlides del GFS."
        return output, None
    except Exception as e: return None, f"Error en carregar dades del mapa GFS: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_usa(nivell, hourly_index):
    try:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        map_data_raw, error = carregar_dades_mapa_base_usa(variables, hourly_index)
        if error: return None, error
        temp_data = np.array(map_data_raw.pop(f'temperature_{nivell}hPa')) * units.degC
        rh_data = np.array(map_data_raw.pop(f'relative_humidity_{nivell}hPa')) * units.percent
        map_data_raw['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
        map_data_raw['speed_data'] = map_data_raw.pop(f'wind_speed_{nivell}hPa')
        map_data_raw['dir_data'] = map_data_raw.pop(f'wind_direction_{nivell}hPa')
        return map_data_raw, None
    except Exception as e: return None, f"Error en processar dades del mapa GFS: {e}"
        
def crear_mapa_forecast_combinat_usa(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    # 1. Crear el mapa base amb la projecció correcta per als EUA
    fig, ax = crear_mapa_base(MAP_EXTENT_USA, projection=ccrs.LambertConformal(central_longitude=-95, central_latitude=35))
    
    # Assegurem que tenim prous dades per a la interpolació
    if len(lons) < 4:
        st.warning("No hi ha prou dades per generar un mapa interpolat.")
        ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
        return fig

    # 2. Crear una graella fina i interpolar les dades del model
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_USA[0], MAP_EXTENT_USA[1], 200), np.linspace(MAP_EXTENT_USA[2], MAP_EXTENT_USA[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')

    # 3. Dibuixar la velocitat del vent amb pcolormesh
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind)
    norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    mesh = ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")

    # 4. Dibuixar les línies de corrent del vent (streamplot)
    # --- LÍNIA MODIFICADA ---
    # S'ha afegit el paràmetre 'arrowsize' per controlar la mida de les fletxes.
    # Pots canviar el valor (ex: 0.8) per fer-les més petites o més grans.
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    # 5. Calcular i dibuixar la convergència
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    dudx = mpcalc.first_derivative(grid_u * units('m/s'), delta=dx, axis=1)
    dvdy = mpcalc.first_derivative(grid_v * units('m/s'), delta=dy, axis=0)
    convergence_scaled = -(dudx + dvdy).to('1/s').magnitude * 1e5
    
    DEWPOINT_THRESHOLD_USA = 16 
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD_USA, convergence_scaled, 0)
    
    fill_levels = [5, 10, 15, 25]
    fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]
    line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.5, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    labels = ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')
    for label in labels:
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
    
    # Afegir ciutats per a referència
    for city, coords in USA_CITIES.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=1, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.2, coords['lat'] + 0.2, city, fontsize=7, transform=ccrs.PlateCarree(), zorder=10,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax.set_title(f"Vent i Nuclis de convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig
# --- Seccions UI i Lògica Principal ---

def hide_streamlit_style():
    """Injecta CSS per amagar el peu de pàgina i el menú de Streamlit."""
    hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
    st.markdown(hide_style, unsafe_allow_html=True)

def generar_prompt_per_ia(params, pregunta_usuari, poble):
    """
    Crea un prompt detallat i estructurat per a l'assistent d'IA,
    combinant el rol, les dades del sondeig i la pregunta de l'usuari.
    """
    # Iniciem la construcció del prompt amb el rol i les instruccions
    prompt_parts = [
        "### ROL I INSTRUCCIONS",
        "Ets un meteoròleg expert en temps sever, especialitzat en la interpretació de sondejos atmosfèrics.",
        "Analitza les següents dades de sondeig per a la localitat indicada i respon la pregunta de l'usuari.",
        "La teva resposta ha de ser tècnica però clara, concisa i basada ÚNICAMENT en les dades proporcionades.",
        "No inventis informació. Si una dada no hi és, indica que no està disponible.",
        "Respon sempre en català.",
        "\n### DADES DEL SONDEIG",
        f"**Localitat:** {poble}",
    ]

    # Diccionari per a noms més clars i unitats
    noms_parametres = {
        'SBCAPE': ('SBCAPE', 'J/kg'), 'MUCAPE': ('MUCAPE', 'J/kg'), 'MLCAPE': ('MLCAPE', 'J/kg'),
        'SBCIN': ('SBCIN', 'J/kg'), 'MUCIN': ('MUCIN', 'J/kg'), 'MLCIN': ('MLCIN', 'J/kg'),
        'LI': ('Lifted Index', '°C'), 'PWAT': ('Aigua Precipitable', 'mm'),
        'LCL_Hgt': ('Base del Núvol (LCL)', 'm'), 'LFC_Hgt': ('Nivell de Conv. Lliure (LFC)', 'm'),
        'BWD_0-6km': ('Cisallament 0-6km', 'nusos'), 'BWD_0-1km': ('Cisallament 0-1km', 'nusos'),
        'SRH_0-1km': ('Helicitat 0-1km', 'm²/s²'), 'SRH_0-3km': ('Helicitat 0-3km', 'm²/s²'),
        'MAX_UPDRAFT': ('Corrent Ascendent Màx.', 'm/s')
    }

    # Afegim cada paràmetre al prompt de forma estructurada
    for key, (nom, unitat) in noms_parametres.items():
        valor = params.get(key)
        if valor is not None and not np.isnan(valor):
            prompt_parts.append(f"- **{nom}:** {valor:.1f} {unitat}")
        else:
            prompt_parts.append(f"- **{nom}:** No disponible")

    # Afegim la pregunta de l'usuari al final
    prompt_parts.append("\n### PREGUNTA DE L'USUARI")
    prompt_parts.append(pregunta_usuari)

    return "\n".join(prompt_parts)

def ui_pestanya_assistent_ia(params_calc, poble_sel):
    """
    Crea la interfície d'usuari per a la pestanya de l'assistent d'IA.
    """
    st.markdown("#### Assistent d'Anàlisi (IA Gemini)")
    st.info("Fes una pregunta en llenguatge natural sobre les dades del sondeig. Per exemple: *'Quin és el potencial de calamarsa?'* o *'Hi ha risc de tornados segons aquestes dades?'*")

    # Inicialització de l'historial del xat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar l'historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura de la pregunta de l'usuari
    if prompt := st.chat_input("Fes una pregunta sobre el sondeig..."):
        # Afegir i mostrar el missatge de l'usuari
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar i mostrar la resposta de l'IA
        with st.chat_message("assistant"):
            try:
                # Configurem el model de Gemini
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Creem el prompt complet i el mostrem en un expander (per depuració)
                prompt_complet = generar_prompt_per_ia(params_calc, prompt, poble_sel)
                with st.expander("Veure el prompt enviat a la IA"):
                    st.text(prompt_complet)
                
                # Cridem a la IA i mostrem la resposta amb efecte "stream"
                response = model.generate_content(prompt_complet, stream=True)
                resposta_completa = st.write_stream(response)
                
                # Guardem la resposta completa a l'historial
                st.session_state.messages.append({"role": "assistant", "content": resposta_completa})
            
            except Exception as e:
                st.error(f"Hi ha hagut un error en contactar amb l'assistent d'IA: {e}")

def ui_capcalera_selectors(ciutats_a_mostrar, info_msg=None, zona_activa="catalunya"):
    st.markdown(f'<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | {zona_activa.replace("_", " ").title()}</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)
    
    col_text, col_change, col_logout = st.columns([0.7, 0.15, 0.15])
    with col_text:
        if not is_guest: st.markdown(f"Benvingut/da, **{st.session_state.get('username')}**!")
    with col_change:
        if st.button("Canviar Anàlisi", use_container_width=True):
            del st.session_state['zone_selected']
            st.rerun()
    with col_logout:
        if st.button("Sortir" if is_guest else "Tanca Sessió", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        if zona_activa == 'catalunya':
            with col1:
                if is_guest: st.info(f"ℹ️ **Mode Convidat:** {info_msg}")
                poble_actual = st.session_state.get('poble_selector')
                sorted_ciutats = sorted(ciutats_a_mostrar.keys())
                index_poble = sorted_ciutats.index(poble_actual) if poble_actual in sorted_ciutats else 0
                st.selectbox("Població de referència:", sorted_ciutats, key="poble_selector", index=index_poble)
            now_local = datetime.now(TIMEZONE_CAT)
            with col2: st.selectbox("Dia del pronòstic:", ("Avui",) if is_guest else ("Avui", "Demà"), key="dia_selector", disabled=is_guest, index=0)
            with col3: st.selectbox("Hora del pronòstic (Local):", (f"{now_local.hour:02d}:00h",) if is_guest else [f"{h:02d}:00h" for h in range(24)], key="hora_selector", disabled=is_guest, index=0 if is_guest else now_local.hour)
        else: # Zona USA
             with col1:
                st.selectbox("Ciutat de referència:", sorted(USA_CITIES.keys()), key="poble_selector_usa")
             now_local = datetime.now(TIMEZONE_USA)
             with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà", "Demà passat"), key="dia_selector_usa", index=0)
             with col3: st.selectbox("Hora del pronòstic (Local - CST):", [f"{h:02d}:00" for h in range(24)], key="hora_selector_usa", index=now_local.hour)

def ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel):
    st.markdown("#### Mapes de Pronòstic (Model AROME)")
    col_capa, col_zoom = st.columns(2)
    with col_capa:
        # ELIMINAT: El selector de nivell ja no és aquí.
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", ["Anàlisi de Vent i Convergència", "Vent a 700hPa", "Vent a 300hPa"], key="map_cat")
    with col_zoom: zoom_sel = st.selectbox("Nivell de Zoom:", options=list(MAP_ZOOM_LEVELS_CAT.keys()), key="zoom_cat")
    selected_extent = MAP_ZOOM_LEVELS_CAT[zoom_sel]
    
    if "Convergència" in mapa_sel:
        # MODIFICAT: Utilitzem el 'nivell_sel' que rebem com a argument.
        with st.spinner(f"Carregant dades del mapa AROME a {nivell_sel}hPa..."): 
            map_data, error_map = carregar_dades_mapa_cat(nivell_sel, hourly_index_sel)
        if error_map: st.error(f"Error en carregar el mapa: {error_map}")
        elif map_data:
            fig = crear_mapa_forecast_combinat_cat(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str, selected_extent)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        nivell = 700 if "700" in mapa_sel else 300; variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        with st.spinner(f"Carregant vent a {nivell}hPa..."): map_data, error_map = carregar_dades_mapa_base_cat(variables, hourly_index_sel)
        if error_map: st.error(f"Error: {error_map}")
        elif map_data: 
            fig = crear_mapa_vents_cat(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str, selected_extent)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

def ui_pestanya_vertical(data_tuple, poble_sel, lat, lon, nivell_conv):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        p, T, Td, u, v, heights, prof = sounding_data
        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Sondeig Vertical\n{poble_sel}")
            st.pyplot(fig_skewt, use_container_width=True); plt.close(fig_skewt)
            with st.container(border=True): 
                # MODIFICAT: Passem el 'nivell_conv' a la caixa de paràmetres.
                ui_caixa_parametres_sondeig(params_calculats, nivell_conv)
        with col2:
            fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat\n{poble_sel}")
            st.pyplot(fig_hodo, use_container_width=True); plt.close(fig_hodo)
            st.markdown("##### Radar de Precipitació en Temps Real")
            radar_url = f"https://www.rainviewer.com/map.html?loc={lat},{lon},8&oCS=1&c=3&o=83&lm=0&layer=radar&sm=1&sn=1&ts=2&play=1"
            html_code = f"""<div style="position: relative; width: 100%; height: 410px; border-radius: 10px; overflow: hidden;"><iframe src="{radar_url}" width="100%" height="410" frameborder="0" style="border:0;"></iframe><div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; cursor: default;"></div></div>"""
            st.components.v1.html(html_code, height=410)
    else: st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def ui_pestanya_mapes_usa(hourly_index_sel, timestamp_str, nivell_sel):
    st.markdown("#### Mapes de Pronòstic (Model GFS)")
    
    # ELIMINAT: El selector de nivell ja no és aquí, el rebem com a 'nivell_sel'.
    # nivell_sel = st.selectbox(...)

    with st.spinner(f"Carregant dades del mapa GFS a {nivell_sel}hPa..."):
        map_data, error_map = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
    
    if error_map:
        st.error(f"Error en carregar el mapa: {error_map}")
    elif map_data:
        fig = crear_mapa_forecast_combinat_usa(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

def ui_pestanya_satelit_usa():
    st.markdown("#### Imatge de Satèl·lit GOES-East (Temps Real)")
    
    # --- LÍNIA CORREGIDA ---
    # S'ha canviat l'URL del satèl·lit de MESO (mòbil) a CONUS (fixa).
    sat_url = f"https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/GEOCOLOR/latest.jpg?{int(time.time())}"
    
    st.image(sat_url, caption="Imatge del satèl·lit GOES-East - Vista CONUS (NOAA STAR)", use_container_width=True)
    
    st.info(
        """
        Aquesta imatge mostra la vista **CONUS (Contiguous United States)**, que cobreix tots els Estats Units continentals. 
        S'actualitza cada 5-10 minuts i garanteix que sempre puguem veure la "Tornado Alley", a diferència dels sectors de mesoescala mòbils.
        """
    )
    st.markdown("<p style='text-align: center;'>[Font: NOAA STAR](https://www.star.nesdis.noaa.gov/GOES/index.php)</p>", unsafe_allow_html=True)




@st.cache_data(ttl=600)
def obtenir_dades_estacio_smc():
    try: api_key = st.secrets["SMC_API_KEY"]
    except KeyError: return None, "Falta la clau 'SMC_API_KEY' als secrets."
    url = "https://api.meteo.cat/xema/v1/observacions/mesurades/ultimes"; headers = {"X-Api-Key": api_key}
    try:
        response = requests.get(url, headers=headers, timeout=15); response.raise_for_status(); return response.json(), None
    except requests.exceptions.RequestException as e: return None, f"Error de xarxa en contactar amb l'API de l'SMC: {e}"

def ui_pestanya_estacions_meteorologiques():
    st.markdown("#### Dades en Temps Real (Xarxa d'Estacions de l'SMC)")
    if "SMC_API_KEY" not in st.secrets or not st.secrets["SMC_API_KEY"]:
        st.info("🚧 **Pestanya en Desenvolupament**\n\nAquesta secció està pendent de la validació de la clau d'accés a les dades oficials del Servei Meteorològic de Catalunya (SMC).", icon="🚧")
        return

    st.caption("Dades oficials de la Xarxa d'Estacions Meteorològiques Automàtiques (XEMA) del Servei Meteorològic de Catalunya.")
    with st.spinner("Carregant dades de la XEMA..."): dades_xema, error = obtenir_dades_estacio_smc()
    if error: st.error(error); return
    if not dades_xema: st.warning("No s'han pogut carregar les dades de les estacions de l'SMC."); return
    
    col1, col2 = st.columns([0.6, 0.4], gap="large")
    with col1:
        st.markdown("##### Mapa d'Ubicacions")
        fig, ax = crear_mapa_base(MAP_EXTENT_CAT)
        for ciutat, coords in CIUTATS_CATALUNYA.items():
            if ciutat in SMC_STATION_CODES:
                lon, lat = coords['lon'], coords['lat']
                ax.plot(lon, lat, 'o', color='darkblue', markersize=8, markeredgecolor='white', transform=ccrs.PlateCarree(), zorder=10)
                ax.text(lon + 0.03, lat, ciutat, fontsize=7, transform=ccrs.PlateCarree(), zorder=11, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col2:
        st.markdown("##### Dades de l'Estació")
        ciutat_seleccionada = st.selectbox("Selecciona una capital de comarca:", options=sorted(SMC_STATION_CODES.keys()))
        if ciutat_seleccionada:
            station_code = SMC_STATION_CODES.get(ciutat_seleccionada)
            dades_estacio = next((item for item in dades_xema if item.get("codi") == station_code), None)
            if dades_estacio:
                nom = dades_estacio.get("nom", "N/A"); data = dades_estacio.get("data", "N/A").replace("T", " ").replace("Z", "")
                variables = {var['codi']: var['valor'] for var in dades_estacio.get('variables', [])}
                st.info(f"**Estació:** {nom} | **Lectura:** {data} UTC")
                c1, c2 = st.columns(2)
                c1.metric("Temperatura", f"{variables.get(32, '--')} °C"); c2.metric("Humitat", f"{variables.get(33, '--')} %")
                st.metric("Pressió atmosfàrica", f"{variables.get(35, '--')} hPa")
                st.metric("Vent", f"{variables.get(31, '--')}° a {variables.get(30, '--')} km/h (Ràfega: {variables.get(2004, '--')} km/h)")
                st.metric("Precipitació (30 min)", f"{variables.get(34, '--')} mm")
                st.markdown(f"🔗 [Veure a la web de l'SMC](https://www.meteo.cat/observacions/xema/dades?codi={station_code})", unsafe_allow_html=True)
            else: st.error("No s'han trobat dades recents per a aquesta estació.")

def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades AROME/GFS via Open-Meteo | Imatges via Meteociel & NOAA | IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- Lògica Principal de l'Aplicació ---

def run_catalunya_app():
    is_guest = st.session_state.get('guest_mode', False)
    now_local = datetime.now(TIMEZONE_CAT)
    hora_sel_str = f"{now_local.hour:02d}:00h" if is_guest else st.session_state.get('hora_selector', f"{now_local.hour:02d}:00h")
    dia_sel_str = "Avui" if is_guest else st.session_state.get('dia_selector', "Avui")
    
    target_date = now_local.date() + timedelta(days=1) if dia_sel_str == "Demà" else now_local.date()
    local_dt = TIMEZONE_CAT.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_str.split(':')[0])))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    
    ciutats_per_selector, info_msg = (obtenir_ciutats_actives(hourly_index_sel)[0], "Anàlisi limitada a les zones de més interès.") if is_guest else (CIUTATS_CATALUNYA, None)
    ui_capcalera_selectors(ciutats_per_selector, info_msg, zona_activa="catalunya")
    
    poble_sel = st.session_state.poble_selector
    if poble_sel not in ciutats_per_selector: 
        st.session_state.poble_selector = sorted(ciutats_per_selector.keys())[0]; st.rerun()
    
    timestamp_str = f"{st.session_state.dia_selector} a les {st.session_state.hora_selector} (Hora Local)"
    lat_sel, lon_sel = ciutats_per_selector[poble_sel]['lat'], ciutats_per_selector[poble_sel]['lon']
    
    # --- LÒGICA DE CÀLCUL CENTRALITZADA ---
    data_tuple, error_msg = carregar_dades_sondeig_cat(lat_sel, lon_sel, hourly_index_sel)
    if error_msg: 
        st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
        return

    nivell_sel = 925 # Valor per defecte
    if not is_guest:
        nivells_disponibles = [1000, 950, 925, 850, 800, 700]
        index_default = nivells_disponibles.index(925) if 925 in nivells_disponibles else 0
        nivell_sel = st.selectbox(
            "Nivell d'anàlisi per a Mapes i Convergència:", 
            options=nivells_disponibles, 
            format_func=lambda x: f"{x} hPa", 
            key="level_cat_main",
            index=index_default
        )
    else:
        st.info("ℹ️ L'anàlisi de vent i convergència està fixada a **925 hPa** en el mode convidat.")

    map_data_conv, _ = carregar_dades_mapa_cat(nivell_sel, hourly_index_sel)
    
    # Aquesta línia és important per assegurar-nos que 'params_calc' existeix
    params_calc = data_tuple[1] if data_tuple else {}

    if data_tuple and map_data_conv:
        conv_value = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
        params_calc[f'CONV_{nivell_sel}hPa'] = conv_value
    
    # --- VISUALITZACIÓ EN PESTANYES (ARA AMB LÒGICA PER A IA) ---
    
    if is_guest:
        # PESTANYES PER A CONVIDATS (SENSE IA)
        tab_mapes, tab_vertical, tab_estacions = st.tabs(["Anàlisi de Mapes", "Anàlisi Vertical", "Estacions Meteorològiques"])
        with tab_mapes: 
            ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel)
        with tab_vertical: 
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel)
        with tab_estacions: 
            ui_pestanya_estacions_meteorologiques()
    else:
        # PESTANYES PER A USUARIS REGISTRATS (AMB IA)
        tab_mapes, tab_vertical, tab_ia, tab_estacions = st.tabs(["Anàlisi de Mapes", "Anàlisi Vertical", "💬 Assistent IA", "Estacions Meteorològiques"])
        with tab_mapes: 
            ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel)
        with tab_vertical: 
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel)
        with tab_ia:
            # Cridem a la nova funció de la interfície de la IA
            ui_pestanya_assistent_ia(params_calc, poble_sel)
        with tab_estacions: 
            ui_pestanya_estacions_meteorologiques()

def run_valley_halley_app():
    ui_capcalera_selectors(None, zona_activa="tornado_alley")
    
    poble_sel = st.session_state.poble_selector_usa
    dia_sel_str = st.session_state.dia_selector_usa
    hora_sel_str = st.session_state.hora_selector_usa

    day_offset = {"Avui": 0, "Demà": 1, "Demà passat": 2}[dia_sel_str]
    target_date = datetime.now(TIMEZONE_USA).date() + timedelta(days=day_offset)
    local_dt = TIMEZONE_USA.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_str.split(':')[0])))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    
    timestamp_str = f"{dia_sel_str} a les {hora_sel_str} (Central Time)"
    lat_sel, lon_sel = USA_CITIES[poble_sel]['lat'], USA_CITIES[poble_sel]['lon']
    
    # --- LÒGICA DE CÀLCUL CENTRALITZADA (ARA TAMBÉ PER A EUA) ---
    data_tuple, error_msg = carregar_dades_sondeig_usa(lat_sel, lon_sel, hourly_index_sel)
    if error_msg:
        st.error(f"No s'ha pogut carregar el sondeig per a {poble_sel}: {error_msg}")
        return

    # Selector de nivell per a mapes i convergència (model GFS)
    nivells_disponibles_gfs = [925, 850, 700, 500, 300]
    nivell_sel = st.selectbox(
        "Nivell d'anàlisi per a Mapes i Convergència:", 
        options=nivells_disponibles_gfs, 
        format_func=lambda x: f"{x} hPa", 
        key="level_usa_main"
    )

    map_data_conv, _ = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)

    params_calc = data_tuple[1] if data_tuple else {}
    if data_tuple and map_data_conv:
        conv_value = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
        params_calc[f'CONV_{nivell_sel}hPa'] = conv_value

    # --- VISUALITZACIÓ EN PESTANYES ---
    tab_mapes, tab_vertical, tab_satelit = st.tabs(["Anàlisi de Mapes", "Anàlisi Vertical", "Satèl·lit (Temps Real)"])
    with tab_mapes:
        # Passem el nivell seleccionat a la funció de mapes
        ui_pestanya_mapes_usa(hourly_index_sel, timestamp_str, nivell_sel)
    with tab_vertical:
        # Passem el nivell seleccionat a la funció vertical
        ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel)
    with tab_satelit:
        ui_pestanya_satelit_usa()
        

def ui_zone_selection():
    st.markdown("<h1 style='text-align: center;'>Zona d'Anàlisi</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Camins a les imatges locals
    path_img_cat = "catalunya.jpeg"
    path_img_usa = "tornado_alley.jpeg"

    # Comprovem si els arxius existeixen per evitar un error
    if not os.path.exists(path_img_cat) or not os.path.exists(path_img_usa):
        st.error(f"Error: No es troben les imatges. Assegura't que els arxius '{path_img_cat}' i '{path_img_usa}' estan al mateix directori que el teu script Python.")
        return

    with st.spinner('Carregant entorns geoespacials...'): 
        time.sleep(1) 

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.image(path_img_cat, caption="Tempestes catalanas tradicionals.")
            st.subheader("Catalunya")
            st.write("Anàlisi detallada d'alta resolució (Model AROME) per al territori català. Ideal per a seguiment de tempestes locals, fenòmens de costa i muntanya.")
            
            # --- LÍNIA CORREGIDA ---
            # S'ha afegit una 'key' única a aquest botó.
            if st.button("Analitzar Catalunya", use_container_width=True, type="primary", key="btn_select_catalunya"):
                st.session_state['zone_selected'] = 'catalunya'
                st.rerun()
    with col2:
        with st.container(border=True):
            st.image(path_img_usa, caption="Supercèl·lula a les planes dels Estats Units.")
            st.subheader("Tornado Alley (EUA)")
            st.write("Anàlisi a escala sinòptica (Model GFS) per al 'Corredor de Tornados' dels EUA. Perfecte per a l'estudi de sistemes de gran escala i temps sever organitzat.")
            
            # --- LÍNIA CORREGIDA ---
            # S'ha afegit una 'key' única i diferent a aquest altre botó.
            if st.button("Analitzar Tornado Alley", use_container_width=True, key="btn_select_usa"):
                st.session_state['zone_selected'] = 'valley_halley'
                st.rerun()
def main():
    # Crida la funció per amagar els estils just a l'inici
    hide_streamlit_style()
    
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'guest_mode' not in st.session_state: st.session_state['guest_mode'] = False
    if 'zone_selected' not in st.session_state: st.session_state['zone_selected'] = None

    if not st.session_state['logged_in']: show_login_page()
    elif not st.session_state['zone_selected']: ui_zone_selection()
    elif st.session_state['zone_selected'] == 'catalunya': run_catalunya_app()
    elif st.session_state['zone_selected'] == 'valley_halley': run_valley_halley_app()

def determinar_emoji_temps(params, nivell_conv):
    """
    Sistema de Diagnòstic Meteorològic Expert.
    Analitza la interacció complexa de paràmetres (termodinàmics i dinàmics)
    per determinar el tipus de núvol, el temps associat i el procés dominant.
    """
    # --- 1. Extracció i validació de tots els paràmetres clau ---
    cape = params.get('SBCAPE', 0) or 0
    li = params.get('LI', 5) or 5
    cin = params.get('SBCIN', 0) or 0
    pwat = params.get('PWAT', 0) or 0
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_1km = params.get('SRH_0-1km', 0) or 0
    lcl_hgt = params.get('LCL_Hgt', 9999) or 9999
    lfc_hgt = params.get('LFC_Hgt', 9999) or 9999
    
    conv_key = f'CONV_{nivell_conv}hPa'
    conv = params.get(conv_key, 0) or 0

    # --- 2. Lògica de Diagnòstic Avançada ---

    # == Branca A: Atmosfera amb Potencial Convectiu (CAPE > 200 J/kg i LI < 0) ==
    if cape > 200 and li < 0:
        
        # A.1: Avaluació de condicions PROHIBITIVES per a la convecció.
        # Un LFC extremadament alt o un CIN massiu fan gairebé impossible la convecció.
        if lfc_hgt > 3500 or cin < -200:
            return "🚫", "Inestabilitat Fortament Capada"

        # A.2: Avaluació del potencial de TEMPS SEVER ORGANITZAT.
        # Requereix una combinació d'alta energia i fort cisallament del vent.
        if cape > 1500 and bwd_6km > 20 and srh_1km > 100:
            return "🌪️", "Potencial de Supercèl·lula"
        if cape > 800 and bwd_6km > 18:
            return "⛈️", "Potencial de Multicèl·lules"

        # A.3: Avaluació del DISPARADOR per a convecció menys organitzada.
        # La facilitat d'iniciació depèn de la distància entre la base del núvol (LCL) i el punt d'enlairament (LFC).
        gap_lcl_lfc = lfc_hgt - lcl_hgt
        iniciacio_facil = (gap_lcl_lfc < 1000 and cin > -50)
        
        # El disparador és actiu si hi ha un fort forçament (convergència) O si l'entorn és molt favorable (iniciació fàcil).
        disparador_actiu = (conv > 4) or (conv > 1.5 and iniciacio_facil)

        if disparador_actiu:
            if cape > 500:
                return "⚡", "Tempesta Aïllada (Cb Calvus)"
            else: # cape > 200
                return "☁️", "Desenvolupament Vertical (Cu Congestus)"
        else:
            # Aquest és el clàssic "sondeig carregat" (loaded gun): molta energia però cap disparador.
            return "🌤️", "Inestabilitat Capada (latent)"

    # == Branca B: Atmosfera Estable o amb Baix Potencial Convectiu ==
    else:
        # B.1: Cel cobert i baix amb potencial de plugim (temps de "sotoportico").
        # Es caracteritza per un LCL molt baix, alta humitat (PWAT) i un perfil saturat.
        if lcl_hgt < 400 and pwat > 25:
            return "🌫️", "Boira o Estrats Baixos (St)"
        if pwat > 35 and cape < 100:
            return "🌧️", "Plugims / Ruixats (Nimbostratus)"

        # B.2: Núvols de bon temps o poc desenvolupament.
        # Depèn de l'altura de la base dels núvols (LCL).
        if lcl_hgt < 1500:
            # Base baixa, són els cúmuls de bon temps.
            return "⛅", "Cúmuls de Bon Temps (Cu humilis)"
        elif lcl_hgt < 4000:
            # Base mitjana, són núvols mitjans.
            return "🌥️", "Núvols Mitjans (Altocumulus)"
        elif lcl_hgt < 8000:
            # Base alta, són núvols alts i prims.
            return "☀️", "Núvols Alts i Prim (Cirrus)"
        else:
            # Atmosfera molt seca i estable.
            return "☀️", "Cel Serè"
    


if __name__ == "__main__":
    main()
