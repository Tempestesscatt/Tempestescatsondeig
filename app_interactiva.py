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
from streamlit_option_menu import option_menu

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
    
    # --- NOUS SONDEJOS MARINS AFEGITS ---
    'Costes de Girona (Mar)': {'lat': 42.05, 'lon': 3.55},      # <-- NOU SONDEIG MARÍ
    'Litoral Barceloní (Mar)': {'lat': 41.30, 'lon': 2.50},    # <-- NOU SONDEIG MARÍ
    'Aigües de Tarragona (Mar)': {'lat': 40.90, 'lon': 1.65},  # <-- NOU SONDEIG MARÍ
}
POBLACIONS_TERRA = {k: v for k, v in CIUTATS_CATALUNYA.items() if '(Mar)' not in k}
PUNTS_MAR = {k: v for k, v in CIUTATS_CATALUNYA.items() if '(Mar)' in k}
CIUTATS_CONVIDAT = {
    'Barcelona': CIUTATS_CATALUNYA['Barcelona'], 'Girona': CIUTATS_CATALUNYA['Girona'],
    'Lleida': CIUTATS_CATALUNYA['Lleida'], 'Tarragona': CIUTATS_CATALUNYA['Tarragona']
}
MAP_EXTENT_CAT = [0, 3.5, 40.4, 43]
PRESS_LEVELS_AROME = sorted([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
MAP_ZOOM_LEVELS_CAT = {'Catalunya (Complet)': MAP_EXTENT_CAT, 'Nord-est (Girona)': [1.8, 3.4, 41.7, 42.6], 'Sud (Tarragona i Ebre)': [0.2, 1.8, 40.5, 41.4], 'Ponent i Pirineu (Lleida)': [0.4, 1.9, 41.4, 42.6],'Maresme': [2.3, 2.8, 41.5, 41.7], 'Àrea Metropolitana (BCN)': [1.7, 2.7, 41.2, 41.8]}

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
CHAT_FILE = 'chat_history.json'
MAX_IA_REQUESTS = 5              
TIME_WINDOW_SECONDS = 3 * 60 * 60 
RATE_LIMIT_FILE = 'rate_limits.json'

THRESHOLDS_GLOBALS = {
    'SBCAPE': (500, 1500, 2500), 'MUCAPE': (500, 1500, 2500), 
    'MLCAPE': (250, 1000, 2000), 'CAPE_0-3km': (75, 150, 250), 
    'SBCIN': (-25, -75, -150), 'MUCIN': (-25, -75, -150),
    'MLCIN': (-25, -75, -150), 'LI': (-2, -5, -8), 
    'PWAT': (30, 40, 50), 
    'BWD_0-6km': (20, 30, 40), 
    'BWD_0-1km': (15, 25, 35),
    'SRH_0-1km': (100, 150, 250), 
    'SRH_0-3km': (150, 250, 400),
    
    # --- NOUS LLINDARS AFEGITS ---
    # Per a LCL/LFC, els valors s'interpreten de manera inversa (més baix és pitjor)
    'LCL_Hgt': (1000, 1500), # <1000m (Vermell), 1000-1500m (Verd), >1500m (Gris)
    'LFC_Hgt': (1500, 2500), # <1500m (Vermell), 1500-2500m (Verd), >2500m (Gris)
    
    # Per a UPDRAFT, valors més alts són pitjors
    'MAX_UPDRAFT': (25, 40, 55) # >25m/s (Groc), >40m/s (Taronja), >55m/s (Vermell)
}

def get_color_global(value, param_key, reverse_colors=False):
    """
    Versió Definitiva v2.0.
    Inclou una lògica especial per a LCL i LFC, on els valors baixos són vermells.
    """
    if pd.isna(value): return "#808080"

    thresholds = THRESHOLDS_GLOBALS.get(param_key, [])
    if not thresholds: return "#FFFFFF"

    # --- LÒGICA ESPECIAL PER A LCL i LFC ---
    if param_key in ['LCL_Hgt', 'LFC_Hgt']:
        # Aquests paràmetres només tenen 2 llindars per a 3 colors
        if len(thresholds) != 2: return "#FFFFFF"
        
        if value < thresholds[0]: return "#dc3545"  # Vermell (Perillós)
        if value < thresholds[1]: return "#2ca02c"  # Verd (Normal)
        return "#808080"                         # Gris (Inhibidor)
    # --- FI DE LA LÒGICA ESPECIAL ---
    
    # Lògica per a la resta de paràmetres (que tenen 3 llindars)
    if len(thresholds) != 3: return "#FFFFFF"

    colors = ["#2ca02c", "#ffc107", "#fd7e14", "#dc3545"] # Verd, Groc, Taronja, Vermell
    
    if reverse_colors: # Per a CIN i LI
        if value < thresholds[2]: return colors[3]
        if value < thresholds[1]: return colors[2]
        if value < thresholds[0]: return colors[1]
        return colors[0] # Aquí el verd és el color per a valors "segurs"
    
    # Lògica normal per a CAPE, BWD, SRH, UPDRAFT, etc.
    if value >= thresholds[2]: return colors[3]
    elif value >= thresholds[1]: return colors[2]
    elif value >= thresholds[0]: return colors[1]
    else: return colors[0]
        
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
    
def inject_custom_css():
    st.markdown("""
    <style>
    .stSpinner > div {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
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

def calcular_mlcape_robusta(p, T, Td):
    """
    Una funció manual i extremadament robusta per calcular el MLCAPE i MLCIN.
    Aquesta funció està dissenyada per no fallar mai, fins i tot amb sondejos "difícils".
    """
    try:
        # 1. Defineix la capa de barreja (els primers 100 hPa)
        p_sfc = p[0]
        p_bottom = p_sfc - 100 * units.hPa
        mask = (p >= p_bottom) & (p <= p_sfc)

        # Si no hi ha punts a la capa, fem servir només la superfície (Pla B)
        if not np.any(mask):
            p_mixed, T_mixed, Td_mixed = p[0], T[0], Td[0]
        else:
            # 2. Calcula les condicions mitjanes de la capa
            p_layer, T_layer, Td_layer = p[mask], T[mask], Td[mask]
            
            # Per al punt de partida, necessitem la temperatura potencial i la ratio de barreja mitjanes
            theta_mixed = np.mean(mpcalc.potential_temperature(p_layer, T_layer))
            mixing_ratio_mixed = np.mean(mpcalc.mixing_ratio_from_relative_humidity(p_layer, np.ones_like(p_layer) * 100 * units.percent, Td_layer))
            
            # A partir d'aquests valors mitjans, trobem la T i Td a la pressió de superfície
            T_mixed = mpcalc.temperature_from_potential_temperature(p_sfc, theta_mixed)
            Td_mixed = mpcalc.dewpoint_from_mixing_ratio(p_sfc, mixing_ratio_mixed)
        
        # 3. Puja la nova parcel·la mitjana
        prof_mixed = mpcalc.parcel_profile(p, T_mixed, Td_mixed).to('degC')
        
        # 4. Calcula el CAPE/CIN a partir d'aquesta trajectòria robusta
        mlcape, mlcin = mpcalc.cape_cin(p, T, Td, prof_mixed)
        
        return float(mlcape.m), float(mlcin.m)

    except Exception:
        # Pla C: Si tot falla, retornem NaN. Això gairebé mai hauria de passar.
        return np.nan, np.nan
        




def processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile):
    """
    Versió Definitiva i Antifràgil v5.3.
    Soluciona l'error del moviment de la tempesta capturant el vent mitjà
    directament del càlcul de Bunkers per garantir la coherència.
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
    params_calc = {}; heights_agl = heights - heights[0]

    # --- 2. CÀLCULS BASE I PERFILS DE PARCEL·LA ---
    with parcel_lock:
        sfc_prof, ml_prof = None, None
        try: sfc_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        except Exception: return None, "Error crític: No s'ha pogut calcular ni el perfil de superfície."
        try: _, _, _, ml_prof = mpcalc.mixed_parcel(p, T, Td, depth=100 * units.hPa)
        except Exception: ml_prof = None
        main_prof = ml_prof if ml_prof is not None else sfc_prof

        # --- 3. CÀLCULS ROBUSTS I AÏLLATS ---
        # (Aquí va tota la secció de càlculs de PWAT, LI, T_500hPa, CAPE, etc. que ja tenies)
        try: 
            rh = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100
            params_calc['RH_CAPES'] = {'baixa': np.mean(rh[(p.m <= 1000) & (p.m > 850)]), 'mitjana': np.mean(rh[(p.m <= 850) & (p.m > 500)]), 'alta': np.mean(rh[(p.m <= 500) & (p.m > 250)])}
        except: params_calc['RH_CAPES'] = {'baixa': np.nan, 'mitjana': np.nan, 'alta': np.nan}
        try: params_calc['PWAT'] = float(mpcalc.precipitable_water(p, Td).to('mm').m)
        except: params_calc['PWAT'] = np.nan
        try:
            _, fl_h = mpcalc.freezing_level(p, T, heights); params_calc['FREEZING_LVL_HGT'] = float(fl_h[0].to('m').m)
        except: params_calc['FREEZING_LVL_HGT'] = np.nan
        try:
            p_numeric = p.m; T_numeric = T.m
            if len(p_numeric) >= 2 and p_numeric.min() <= 500 <= p_numeric.max():
                params_calc['T_500hPa'] = float(np.interp(500, p_numeric[::-1], T_numeric[::-1]))
            else: params_calc['T_500hPa'] = np.nan
        except: params_calc['T_500hPa'] = np.nan
        if sfc_prof is not None:
            try:
                sbcape, sbcin = mpcalc.cape_cin(p, T, Td, sfc_prof)
                params_calc['SBCAPE'] = float(sbcape.m); params_calc['SBCIN'] = float(sbcin.m)
                params_calc['MAX_UPDRAFT'] = np.sqrt(2 * float(sbcape.m)) if sbcape.m > 0 else 0.0
            except: params_calc.update({'SBCAPE': np.nan, 'SBCIN': np.nan, 'MAX_UPDRAFT': np.nan})
        if ml_prof is not None:
            try:
                mlcape, mlcin = mpcalc.cape_cin(p, T, Td, ml_prof); params_calc['MLCAPE'] = float(mlcape.m); params_calc['MLCIN'] = float(mlcin.m)
            except: params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
        else: params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
        if main_prof is not None:
            try: params_calc['LI'] = float(mpcalc.lifted_index(p, T, main_prof).m)
            except: params_calc['LI'] = np.nan
            try:
                lfc_p, _ = mpcalc.lfc(p, T, Td, main_prof); params_calc['LFC_p'] = float(lfc_p.m)
                params_calc['LFC_Hgt'] = float(np.interp(lfc_p.m, p.m[::-1], heights_agl.m[::-1]))
            except: params_calc.update({'LFC_p': np.nan, 'LFC_Hgt': np.nan})
            try:
                el_p, _ = mpcalc.el(p, T, Td, main_prof); params_calc['EL_p'] = float(el_p.m)
                params_calc['EL_Hgt'] = float(np.interp(el_p.m, p.m[::-1], heights_agl.m[::-1]))
            except: params_calc.update({'EL_p': np.nan, 'EL_Hgt': np.nan})
            try:
                idx_3km = np.argmin(np.abs(heights_agl.m - 3000))
                cape_0_3, _ = mpcalc.cape_cin(p[:idx_3km+1], T[:idx_3km+1], Td[:idx_3km+1], main_prof[:idx_3km+1]); params_calc['CAPE_0-3km'] = float(cape_0_3.m)
            except: params_calc['CAPE_0-3km'] = np.nan
        try:
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td); params_calc['MUCAPE'] = float(mucape.m); params_calc['MUCIN'] = float(mucin.m)
        except: params_calc.update({'MUCAPE': np.nan, 'MUCIN': np.nan})
        try:
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); params_calc['LCL_p'] = float(lcl_p.m); params_calc['LCL_Hgt'] = float(np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1]))
        except: params_calc.update({'LCL_p': np.nan, 'LCL_Hgt': np.nan})
        try:
            for name, depth_m in [('0-1km', 1000), ('0-6km', 6000)]:
                bwd_u, bwd_v = mpcalc.bulk_shear(p, u, v, height=heights, depth=depth_m * units.meter); params_calc[f'BWD_{name}'] = float(mpcalc.wind_speed(bwd_u, bwd_v).to('kt').m)
        except: params_calc.update({'BWD_0-1km': np.nan, 'BWD_0-6km': np.nan})
        
        # *** LÒGICA DE MOVIMENT DE TEMPESTA CORREGIDA ***
        try:
            # Aquesta funció retorna el moviment dret, esquerre I el vent mitjà.
            # Ara capturem els tres resultats correctament.
            rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p, u, v, heights)
            params_calc['RM'] = (float(rm[0].m), float(rm[1].m))
            params_calc['LM'] = (float(lm[0].m), float(lm[1].m))
            params_calc['Mean_Wind'] = (float(mean_wind[0].m), float(mean_wind[1].m))
        except Exception:
            # Si el càlcul falla, assegurem que tots tres siguin N/A.
            params_calc.update({
                'RM': (np.nan, np.nan),
                'LM': (np.nan, np.nan),
                'Mean_Wind': (np.nan, np.nan)
            })

        if params_calc.get('RM') and not np.isnan(params_calc['RM'][0]):
            u_storm, v_storm = params_calc['RM'][0] * units('m/s'), params_calc['RM'][1] * units('m/s')
            try:
                for name, depth_m in [('0-1km', 1000), ('0-3km', 3000)]:
                    srh = mpcalc.storm_relative_helicity(heights, u, v, depth=depth_m * units.meter, storm_u=u_storm, storm_v=v_storm)[0]; params_calc[f'SRH_{name}'] = float(srh.m)
            except: params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})
        
    return ((p, T, Td, u, v, heights, sfc_prof), params_calc), None

def diagnosticar_potencial_tempesta(params):
    """
    Versió Definitiva i Lògica v2.0.
    Retorna el text del diagnòstic I el seu color corresponent, garantint una
    coherència visual del 100% a l'hodògraf.
    """
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_1km = params.get('SRH_0-1km', 0) or 0
    lcl_hgt = params.get('LCL_Hgt', 9999) or 9999
    cape = params.get('MLCAPE', params.get('SBCAPE', 0)) or 0

    bwd_thresh = THRESHOLDS_GLOBALS['BWD_0-6km']
    srh_thresh = THRESHOLDS_GLOBALS['SRH_0-1km']

    tipus_tempesta = "Cèl·lula Simple"; color_tempesta = "#2ca02c"
    if bwd_6km >= bwd_thresh[2] and cape > 1200:
        tipus_tempesta = "Supercèl·lula"; color_tempesta = "#dc3545"
    elif bwd_6km >= bwd_thresh[1] and cape > 800:
        tipus_tempesta = "Multicèl·lula Severa"; color_tempesta = "#fd7e14"
    elif bwd_6km >= bwd_thresh[0] and cape > 500:
        tipus_tempesta = "Multicèl·lula"; color_tempesta = "#ffc107"

    base_nuvol = "Plana i Alta"; color_base = "#2ca02c"
    if srh_1km >= srh_thresh[2] and lcl_hgt < 1200:
        base_nuvol = "Tornàdica (Wall Cloud)"; color_base = "#dc3545"
    elif srh_1km >= srh_thresh[1] and lcl_hgt < 1500:
        base_nuvol = "Rotatòria Forta"; color_base = "#fd7e14"
    elif srh_1km >= srh_thresh[0]:
        base_nuvol = "Rotatòria (Inflow)"; color_base = "#ffc107"
        
    return tipus_tempesta, color_tempesta, base_nuvol, color_base


    

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
    """
    Versió final i neta. Dibuixa les línies de nivell per a LCL, LFC i EL,
    però SENSE el text de les etiquetes per a una visualització més clara.
    """
    fig = plt.figure(dpi=150, figsize=(7, 8))
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.05, 0.85, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)
    
    skew.ax.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)

    skew.plot_dry_adiabats(color='coral', linestyle='--', alpha=0.5)
    skew.plot_moist_adiabats(color='cornflowerblue', linestyle='--', alpha=0.5)
    skew.plot_mixing_lines(color='limegreen', linestyle='--', alpha=0.5)
    
    if prof is not None:
        skew.shade_cape(p, T, prof, color='red', alpha=0.2)
        skew.shade_cin(p, T, prof, color='blue', alpha=0.2)
        skew.plot(p, prof, 'k', linewidth=3, label='Trajectòria Parcel·la (SFC)', 
                  path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])

    skew.plot(p, T, 'red', lw=2.5, label='Temperatura')
    skew.plot(p, Td, 'green', lw=2.5, label='Punt de Rosada')
        
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03)
    skew.ax.set_ylim(1000, 100); skew.ax.set_xlim(-40, 40)
    skew.ax.set_title(titol, weight='bold', fontsize=14, pad=15)
    skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")

    # --- LÒGICA SIMPLIFICADA ---
    # Ara només dibuixem les línies horitzontals, sense el text.
    levels_to_plot = ['LCL_p', 'LFC_p', 'EL_p']
    for key in levels_to_plot:
        p_lvl = params_calc.get(key)
        if p_lvl is not None and not np.isnan(p_lvl):
            p_val = p_lvl.m if hasattr(p_lvl, 'm') else p_lvl
            skew.ax.axhline(p_val, color='blue', linestyle='--', linewidth=1.5)
    # --- FI DE LA MODIFICACIÓ ---

    skew.ax.legend()
    return fig
    g
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
        else: u_barbs_list.append(np.nan); v_barbs_list.append(np.nan)
    u_barbs = units.Quantity(u_barbs_list, u.units); v_barbs = units.Quantity(v_barbs_list, v.units)
    speed_kmh_barbs = np.sqrt(u_barbs**2 + v_barbs**2).to('km/h').m
    thresholds_barbs = [10, 40, 70, 100, 130]; colors_barbs = ['dimgrey', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    x_pos = np.arange(len(barb_altitudes_km)); u_barbs_kt = u_barbs.to('kt'); v_barbs_kt = v_barbs.to('kt')
    for i, spd_kmh in enumerate(speed_kmh_barbs):
        if not np.isnan(spd_kmh):
            color_index = np.searchsorted(thresholds_barbs, spd_kmh); color = colors_barbs[color_index]
            ax_barbs.barbs(x_pos[i], 0, u_barbs_kt[i], v_barbs_kt[i], length=8, pivot='middle', color=color)
            ax_barbs.text(x_pos[i], -0.8, f"{spd_kmh:.0f} km/h", ha='center', va='top', fontsize=9, color=color, weight='bold')
        else: ax_barbs.text(x_pos[i], 0, "N/A", ha='center', va='center', fontsize=9, color='grey')
    ax_barbs.set_xticks(x_pos); ax_barbs.set_xticklabels([f"{h} km" for h in barb_altitudes_km]); ax_barbs.set_yticks([]); ax_barbs.spines[:].set_visible(False); ax_barbs.tick_params(axis='x', length=0, pad=5); ax_barbs.set_xlim(-0.5, len(barb_altitudes_km) - 0.5); ax_barbs.set_ylim(-1.5, 1.5)
    
    # --- HODÒGRAF (Sense canvis) ---
    h = Hodograph(ax_hodo, component_range=80.); h.add_grid(increment=20, color='gray', linestyle='--')
    intervals = np.array([0, 1, 3, 6, 9, 12]) * units.km; colors_hodo = ['red', 'blue', 'green', 'purple', 'gold']
    h.plot_colormapped(u.to('kt'), v.to('kt'), heights, intervals=intervals, colors=colors_hodo, linewidth=2)
    ax_hodo.set_xlabel('U-Component (nusos)'); ax_hodo.set_ylabel('V-Component (nusos)')

    # --- PANELL DE PARÀMETRES AMB LÒGICA DE COLOR CORREGIDA ---
    ax_params.axis('off')
    def degrees_to_cardinal_ca(d):
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        return dirs[int(round(d / 22.5)) % 16]

    y = 0.95
    motion_data = {'M. Dret': params_calc.get('RM'), 'M. Esquerre': params_calc.get('LM'), 'Es mourà cap a': params_calc.get('Mean_Wind')}
    ax_params.text(0, y, "Moviment (cap a dir/km/h)", ha='left', weight='bold', fontsize=11); y-=0.1
    for display_name, vec in motion_data.items():
        ax_params.text(-0.05, y, f"{display_name}:", ha='left', va='center')
        if vec and not pd.isna(vec[0]):
            u_motion = vec[0] * units('m/s'); v_motion = vec[1] * units('m/s')
            speed = mpcalc.wind_speed(u_motion, v_motion).to('km/h').m; direction = mpcalc.wind_direction(u_motion, v_motion, convention='to').to('deg').m
            cardinal = degrees_to_cardinal_ca(direction)
            ax_params.text(1, y, f"{cardinal} / {speed:.0f} km/h", ha='right', va='center')
        else: ax_params.text(1, y, "---", ha='right', va='center')
        y-=0.1

    # --- LÒGICA DE COLOR DEFINITIVA ---
    # 1. Obtenim el text i el color del diagnòstic directament de la funció experta.
    tipus_tempesta, color_tempesta, base_nuvol, color_base = diagnosticar_potencial_tempesta(params_calc)

    y-=0.05
    ax_params.text(0, y, "Cisallament (nusos)", ha='left', weight='bold', fontsize=11); y-=0.1
    for key, label in [('BWD_0-1km', '0-1 km'), ('BWD_0-6km', '0-6 km')]:
        val = params_calc.get(key, np.nan)
        color = get_color_global(val, key) # El color del número es calcula individualment...
        ax_params.text(0, y, f"{label}:", ha='left', va='center')
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color)
        y-=0.08
    ax_params.text(0, y, "Tipus:", ha='left', va='center', weight='bold')
    ax_params.text(0.05, y - 0.08, tipus_tempesta, ha='left', va='center', weight='bold', color=color_tempesta) # ...però el color del text ve del diagnòstic.
    y-=0.16

    y-=0.05
    ax_params.text(0, y, "Helicitat (m²/s²)", ha='left', weight='bold', fontsize=11); y-=0.1
    for key, label in [('SRH_0-1km', '0-1 km'), ('SRH_0-3km', '0-3 km')]:
        val = params_calc.get(key, np.nan)
        color = get_color_global(val, key)
        ax_params.text(0, y, f"{label}:", ha='left', va='center')
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color)
        y-=0.08
    ax_params.text(0, y, "Base:", ha='left', va='center', weight='bold')
    ax_params.text(0.05, y - 0.08, base_nuvol, ha='left', va='center', weight='bold', color=color_base)
    
    return fig

def calcular_puntuacio_tempesta(sounding_data, params, nivell_conv):
    """
    Calcula una puntuació de 0 a 10 per al potencial de tempesta,
    combinant els ingredients meteorològics clau.
    """
    if not params: return {'score': 0, 'color': '#808080'}

    score = 0
    
    # 1. Combustible (CAPE) - Fins a 4 punts
    sbcape = params.get('SBCAPE', 0) or 0
    if sbcape > 250: score += 1
    if sbcape > 750: score += 1
    if sbcape > 1500: score += 1
    if sbcape > 2500: score += 1

    # 2. Organització (Cisallament) - Fins a 3 punts
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    if bwd_6km > 15: score += 1 # Multicèl·lules febles
    if bwd_6km > 25: score += 1 # Multicèl·lules organitzades
    if bwd_6km > 35: score += 1 # Potencial de supercèl·lula

    # 3. Disparador (Convergència) - Fins a 3 punts
    conv_key = f'CONV_{nivell_conv}hPa'
    conv = params.get(conv_key, 0) or 0
    if conv > 10: score += 1
    if conv > 20: score += 1
    if conv > 35: score += 1

    # 4. Ajustaments per factors clau
    
    # Factor 1: Humitat (Component Marítima) - Pot reduir la puntuació a la meitat
    maritim_analysis = analitzar_component_maritima(sounding_data)
    if maritim_analysis['text'] != 'Sí':
        score *= 0.5 # Si no hi ha humitat, el potencial real cau en picat

    # Factor 2: Inhibició (CIN) - Pot restar punts
    cin = params.get('SBCIN', 0) or 0
    if cin < -100: score -= 2 # Una tapa molt forta és un gran impediment
    elif cin < -50: score -= 1

    # Factor 3: Rotació (SRH) - Pot sumar un bonus
    srh_3km = params.get('SRH_0-3km', 0) or 0
    if srh_3km > 250: score += 1 # Un extra per entorns de supercèl·lula

    # Puntuació final (assegurem que estigui entre 0 i 10)
    final_score = max(0, min(10, round(score)))
    
    # Assignar color a la puntuació
    color = '#808080'
    if final_score >= 8: color = '#dc3545'   # Vermell
    elif final_score >= 6: color = '#fd7e14' # Taronja
    elif final_score >= 4: color = '#ffc107' # Groc
    elif final_score >= 1: color = '#2ca02c' # Verd

    return {'score': final_score, 'color': color}

def analitzar_amenaces_especifiques(params):
    """
    Analitza paràmetres visibles per determinar el potencial de calamarsa,
    esclafits i activitat elèctrica, retornant un text i un color per a la UI.
    Versió 2.0 - Independent de MLCAPE i DCAPE.
    """
    resultats = {
        'calamarsa': {'text': 'Nul·la', 'color': '#808080'},
        'esclafits': {'text': 'Nul·la', 'color': '#808080'},
        'llamps': {'text': 'Nul·la', 'color': '#808080'}
    }

    # 1. Anàlisi de Calamarsa Gran (>2cm) - (Sense canvis, ja depèn de paràmetres visibles)
    updraft = params.get('MAX_UPDRAFT', 0) or 0
    isozero = params.get('FREEZING_LVL_HGT', 5000) or 5000
    if updraft > 55 or (updraft > 45 and isozero < 3500):
        resultats['calamarsa'] = {'text': 'Molt Alta', 'color': '#dc3545'}
    elif updraft > 40 or (updraft > 30 and isozero < 3800):
        resultats['calamarsa'] = {'text': 'Alta', 'color': '#fd7e14'}
    elif updraft > 25:
        resultats['calamarsa'] = {'text': 'Moderada', 'color': '#ffc107'}
    elif updraft > 15:
        resultats['calamarsa'] = {'text': 'Baixa', 'color': '#2ca02c'}

    # 2. Anàlisi d'Esclafits (Ventades fortes) - *** LÒGICA NOVA ***
    # Basat en el Gradient Tèrmic a nivells baixos (LR 0-3km) i la humitat (PWAT).
    # Un ambient sec i amb refredament ràpid afavoreix els esclafits.
    lr_0_3km = params.get('LR_0-3km', 0) or 0
    pwat = params.get('PWAT', 100) or 100
    if lr_0_3km > 8.0 and pwat < 35:
        resultats['esclafits'] = {'text': 'Alta', 'color': '#fd7e14'}
    elif lr_0_3km > 7.0 and pwat < 40:
        resultats['esclafits'] = {'text': 'Moderada', 'color': '#ffc107'}
    elif lr_0_3km > 6.5:
        resultats['esclafits'] = {'text': 'Baixa', 'color': '#2ca02c'}

    # 3. Anàlisi d'Activitat Elèctrica (Llamps) - *** LÒGICA NOVA ***
    # Basat en la inestabilitat (LI) i la profunditat de la tempesta (EL_Hgt).
    li = params.get('LI', 5) or 5
    el_hgt = params.get('EL_Hgt', 0) or 0
    if li < -7 or (li < -5 and el_hgt > 12000):
        resultats['llamps'] = {'text': 'Extrema', 'color': '#dc3545'}
    elif li < -4 or (li < -2 and el_hgt > 10000):
        resultats['llamps'] = {'text': 'Alta', 'color': '#fd7e14'}
    elif li < -1:
        resultats['llamps'] = {'text': 'Moderada', 'color': '#ffc107'}
    elif params.get('MUCAPE', 0) > 150: # Si hi ha una mínima inestabilitat
        resultats['llamps'] = {'text': 'Baixa', 'color': '#2ca02c'}
        
    return resultats


def analitzar_component_maritima(sounding_data):
    """
    Analitza el vent a nivells baixos (~950hPa) per determinar si hi ha
    component marítima, crucial per l'aportació d'humitat a Catalunya.
    Retorna un diccionari amb text i color per a la UI.
    """
    if not sounding_data:
        return {'text': 'N/A', 'color': '#808080'}

    p, u, v = sounding_data[0], sounding_data[3], sounding_data[4]
    
    try:
        # Busquem el vent al nivell més proper a 950 hPa
        target_p = 950 * units.hPa
        idx = (np.abs(p - target_p)).argmin()

        # Comprovem si el nivell trobat és raonable
        if np.abs(p[idx] - target_p).m > 50:
            return {'text': 'Indet.', 'color': '#808080'}

        u_low, v_low = u[idx], v[idx]
        direction = mpcalc.wind_direction(u_low, v_low).m
        speed = mpcalc.wind_speed(u_low, v_low).to('km/h').m

        # Per Catalunya, un vent entre 70° (ENE) i 200° (SSW) té component marítima.
        if 70 <= direction <= 200 and speed > 5: # Ha de tenir una mínima velocitat per ser rellevant
            return {'text': 'Sí', 'color': '#28a745'} # Verd = Ingredient present
        else:
            return {'text': 'No', 'color': '#dc3545'} # Vermell = Ingredient absent
            
    except (IndexError, ValueError):
        return {'text': 'Error', 'color': '#808080'}


def ui_caixa_parametres_sondeig(sounding_data, params, nivell_conv, hora_actual):
    TOOLTIPS = {
        'SBCAPE': "Energia Potencial Convectiva Disponible (CAPE) des de la Superfície. Mesura el 'combustible' per a les tempestes a partir d'una bombolla d'aire a la superfície.",
        'MUCAPE': "El CAPE més alt possible a l'atmosfera (Most Unstable). Útil per detectar inestabilitat elevada, fins i tot si la superfície és estable.",
        'CONVERGENCIA': f"Força de la convergència de vent a {nivell_conv}hPa. Actua com el 'disparador' o 'mecanisme de forçament' que obliga l'aire a ascendir, ajudant a iniciar les tempestes. Valors positius i alts són crucials.",
        'SBCIN': "Inhibició Convectiva (CIN) des de la Superfície. És l'energia necessària per vèncer l'estabilitat inicial. Valors molt negatius actuen com una 'tapa' que impedeix les tempestes.",
        'MUCIN': "La CIN associada al MUCAPE.",
        'COMPONENT_MARITIMA': "Indica si el vent a nivells baixos (~950hPa) prové del mar. Aquesta component és la principal font d'humitat i un ingredient fonamental per a la formació de tempestes significatives a Catalunya.",
        'LI': "Índex d'Elevació (Lifted Index). Mesura la diferència de temperatura a 500hPa entre l'entorn i una bombolla d'aire elevada. Valors molt negatius indiquen una forta inestabilitat.",
        'PWAT': "Aigua Precipitable Total (Precipitable Water). Quantitat total de vapor d'aigua en la columna atmosfèrica. Valors alts indiquen potencial per a pluges fortes.",
        'LCL_Hgt': "Alçada del Nivell de Condensació per Elevació (LCL). És l'alçada a la qual es formarà la base del núvol. Valors baixos (<1000m) afavoreixen el temps sever.",
        'LFC_Hgt': "Alçada del Nivell de Convecció Lliure (LFC). És l'alçada a partir de la qual una bombolla d'aire puja lliurement sense necessitat de forçament. Valors baixos són més favorables.",
        'EL_Hgt': "Alçada del Nivell d'Equilibri (EL). És l'alçada estimada del cim de la tempesta (top del cumulonimbus). Valors més alts indiquen tempestes més potents.",
        'BWD_0-6km': "Cisallament del Vent (Bulk Wind Shear) entre 0 i 6 km. Diferència de vent entre la superfície i 6 km. És crucial per a l'organització de les tempestes (multicèl·lules, supercèl·lules).",
        'BWD_0-1km': "Cisallament del Vent entre 0 i 1 km. Important per a la rotació a nivells baixos, un ingredient clau en la formació de tornados.",
        'T_500hPa': "Temperatura de l'aire a 500 hPa (uns 5.500 metres). És un indicador clau de la inestabilitat. Temperatures molt fredes (< -15°C) en alçada sobre aire càlid en superfície disparen el potencial de tempesta.",
        'MAX_UPDRAFT': "Estimació de la velocitat màxima del corrent ascendent dins la tempesta, calculada a partir del CAPE. És un indicador directe del potencial de calamarsa.",
        'AMENACA_CALAMARSA': "Probabilitat de calamarsa de mida significativa (>2 cm). Es basa en una combinació de la potència del corrent ascendent (MAX_UPDRAFT) i l'alçada de la isoterma de 0°C. Corrents molt forts i nivells de congelació baixos augmenten dràsticament aquest risc.",
        'PUNTUACIO_TEMPESTA': "Índex de 0 a 10 que valora el potencial global de formació de tempestes. Combina automàticament els ingredients clau: Combustible (CAPE), Organització (Cisallament), Disparador (Convergència), Humitat (Component Marítima) i la presència d'Inhibició (CIN).",
        'AMENACA_LLAMPS': "Potencial d'activitat elèctrica. S'estima a partir de la inestabilitat (Índex d'Elevació - LI) i la profunditat de la tempesta (Cim - EL_Hgt). Tempestes molt inestables (LI molt negatiu) i profundes generen molta més separació de càrrega i, per tant, més llamps."
    }
    
    def styled_metric(label, value, unit, param_key, tooltip_text="", precision=0, reverse_colors=False):
        color = "#FFFFFF" # Color per defecte
        if pd.notna(value):
            if 'CONV' in param_key:
                thresholds = [5, 15, 30, 40]
                colors = ["#808080", "#2ca02c", "#ffc107", "#fd7e14", "#dc3545"]
                color = colors[np.searchsorted(thresholds, value)]
            elif param_key == 'T_500hPa':
                thresholds = [-8, -14, -18, -22] # Llindars per T a 500hPa
                colors = ["#2ca02c", "#ffc107", "#fd7e14", "#dc3545", "#b300ff"] # Verd -> Groc -> Taronja -> Vermell -> Lila
                # Com que valors més baixos són pitjors, invertim la cerca
                color = colors[len(thresholds) - np.searchsorted(thresholds, value, side='right')]
            else:
                color = get_color_global(value, param_key, reverse_colors)

        val_str = f"{value:.{precision}f}" if not pd.isna(value) else "---"
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""

        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;">
            <span style="font-size: 0.8em; color: #FAFAFA;">{label} ({unit}){tooltip_html}</span><br>
            <strong style="font-size: 1.6em; color: {color};">{val_str}</strong>
        </div>""", unsafe_allow_html=True)

    def styled_qualitative(label, analysis_dict, tooltip_text=""):
        text = analysis_dict.get('text', 'N/A')
        color = analysis_dict.get('color', '#808080')
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;">
            <span style="font-size: 0.8em; color: #FAFAFA;">{label}{tooltip_html}</span><br>
            <strong style="font-size: 1.6em; color: {color};">{text}</strong>
        </div>""", unsafe_allow_html=True)
        
    def styled_threat(label, text, color, tooltip_key):
        tooltip_text = TOOLTIPS.get(tooltip_key, "")
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;">
            <span style="font-size: 0.8em; color: #FAFAFA;">{label}{tooltip_html}</span><br>
            <strong style="font-size: 1.6em; color: {color};">{text}</strong>
        </div>""", unsafe_allow_html=True)

    st.markdown("##### Paràmetres del Sondeig")
    analisi_temps = analitzar_potencial_meteorologic(params, nivell_conv, hora_actual)
    emoji = analisi_temps['emoji']; descripcio = analisi_temps['descripcio']

    cols = st.columns(3)
    with cols[0]: styled_metric("SBCAPE", params.get('SBCAPE', np.nan), "J/kg", 'SBCAPE', tooltip_text=TOOLTIPS.get('SBCAPE'))
    with cols[1]: styled_metric("MUCAPE", params.get('MUCAPE', np.nan), "J/kg", 'MUCAPE', tooltip_text=TOOLTIPS.get('MUCAPE'))
    with cols[2]: 
        conv_key = f'CONV_{nivell_conv}hPa'
        styled_metric("Convergència", params.get(conv_key, np.nan), "10⁻⁵ s⁻¹", conv_key, precision=1, tooltip_text=TOOLTIPS.get('CONVERGENCIA'))
    
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCIN", params.get('SBCIN', np.nan), "J/kg", 'SBCIN', reverse_colors=True, tooltip_text=TOOLTIPS.get('SBCIN'))
    with cols[1]: styled_metric("MUCIN", params.get('MUCIN', np.nan), "J/kg", 'MUCIN', reverse_colors=True, tooltip_text=TOOLTIPS.get('MUCIN'))
    with cols[2]:
        maritim_analysis = analitzar_component_maritima(sounding_data)
        styled_qualitative("Comp. Marítima", maritim_analysis, tooltip_text=TOOLTIPS.get('COMPONENT_MARITIMA'))
    
    cols = st.columns(3)
    with cols[0]: 
        li_value = params.get('LI', np.nan)
        if hasattr(li_value, '__len__') and not isinstance(li_value, str) and len(li_value) > 0: li_value = li_value[0]
        styled_metric("LI", li_value, "°C", 'LI', precision=1, reverse_colors=True, tooltip_text=TOOLTIPS.get('LI'))
    with cols[1]: 
        styled_metric("PWAT", params.get('PWAT', np.nan), "mm", 'PWAT', precision=1, tooltip_text=TOOLTIPS.get('PWAT'))
    with cols[2]:
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 0.8em; color: #FAFAFA;">Tipus de Cel Previst</span>
            <strong style="font-size: 1.8em; line-height: 1;">{emoji}</strong>
            <span style="font-size: 0.8em; color: #E0E0E0;">{descripcio}</span>
        </div>""", unsafe_allow_html=True)
        
    cols = st.columns(3)
    with cols[0]: styled_metric("LCL", params.get('LCL_Hgt', np.nan), "m", 'LCL_Hgt', precision=0, tooltip_text=TOOLTIPS.get('LCL_Hgt'))
    with cols[1]: styled_metric("LFC", params.get('LFC_Hgt', np.nan), "m", 'LFC_Hgt', precision=0, tooltip_text=TOOLTIPS.get('LFC_Hgt'))
    with cols[2]: styled_metric("CIM (EL)", params.get('EL_Hgt', np.nan), "m", 'EL_Hgt', precision=0, tooltip_text=TOOLTIPS.get('EL_Hgt'))
        
    cols = st.columns(3)
    with cols[0]: styled_metric("BWD 0-6km", params.get('BWD_0-6km', np.nan), "nusos", 'BWD_0-6km', tooltip_text=TOOLTIPS.get('BWD_0-6km'))
    with cols[1]: styled_metric("BWD 0-1km", params.get('BWD_0-1km', np.nan), "nusos", 'BWD_0-1km', tooltip_text=TOOLTIPS.get('BWD_0-1km'))
    with cols[2]: 
        styled_metric("T 500hPa", params.get('T_500hPa', np.nan), "°C", 'T_500hPa', precision=1, tooltip_text=TOOLTIPS.get('T_500hPa'))

    st.markdown("##### Potencial d'Amenaces Severes")
    amenaces = analitzar_amenaces_especifiques(params)
    puntuacio_resultat = calcular_puntuacio_tempesta(sounding_data, params, nivell_conv)
    
    cols = st.columns(3)
    with cols[0]:
        styled_threat("Calamarsa Gran (>2cm)", amenaces['calamarsa']['text'], amenaces['calamarsa']['color'], 'AMENACA_CALAMARSA')
    with cols[1]:
        score_text = f"{puntuacio_resultat['score']} / 10"
        styled_threat("Índex de Potencial", score_text, puntuacio_resultat['color'], 'PUNTUACIO_TEMPESTA')
    with cols[2]:
        styled_threat("Activitat Elèctrica", amenaces['llamps']['text'], amenaces['llamps']['color'], 'AMENACA_LLAMPS')

@st.cache_data(ttl=1800, show_spinner="Analitzant zones de convergència...")
def calcular_convergencies_per_llista(map_data, llista_ciutats):
    """
    Versió original i directa. Calcula la convergència per a una llista de ciutats
    i retorna un diccionari simple {ciutat: valor_convergencia}.
    """
    if not map_data or 'lons' not in map_data or len(map_data['lons']) < 4:
        return {}

    convergencies = {}
    try:
        lons, lats = map_data['lons'], map_data['lats']
        speed_data, dir_data = map_data['speed_data'], map_data['dir_data']

        grid_lon, grid_lat = np.meshgrid(
            np.linspace(min(lons), max(lons), 100),
            np.linspace(min(lats), max(lats), 100)
        )
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        
        with np.errstate(invalid='ignore'):
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)
            convergence_scaled = -divergence.to('1/s').magnitude * 1e5

        for nom_ciutat, coords in llista_ciutats.items():
            lat_sel, lon_sel = coords['lat'], coords['lon']
            dist_sq = (grid_lat - lat_sel)**2 + (grid_lon - lon_sel)**2
            min_dist_idx = np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)
            valor_conv = convergence_scaled[min_dist_idx]
            
            if pd.notna(valor_conv):
                convergencies[nom_ciutat] = valor_conv
    
    except Exception:
        return {}
        
    return convergencies

@st.cache_data(ttl=3600)
def carregar_dades_mapa_base_cat(variables, hourly_index):
    """
    Versió millorada que inclou un pas de neteja per a dades invàlides (NaN)
    i retorna un missatge d'error més específic si no es troba cap dada vàlida.
    """
    try:
        lats, lons = np.linspace(MAP_EXTENT_CAT[2], MAP_EXTENT_CAT[3], 12), np.linspace(MAP_EXTENT_CAT[0], MAP_EXTENT_CAT[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": 4}
        responses = openmeteo.weather_api(API_URL_CAT, params=params)
        
        output = {var: [] for var in ["lats", "lons"] + variables}
        
        for r in responses:
            # Utilitzem un bloc try-except per si l'índex està fora de rang
            try:
                vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
            except IndexError:
                continue # Si l'hora no existeix, simplement saltem aquest punt

            if np.isnan(vals).any():
                continue
            
            output["lats"].append(r.Latitude())
            output["lons"].append(r.Longitude())
            for i, var in enumerate(variables): 
                output[var].append(vals[i])

        if not output["lats"]: 
            return None, "Dades caducades o no disponibles per a l'hora i nivell seleccionats."
            
        return output, None
        
    except Exception as e: 
        return None, f"Error en carregar dades del mapa: {e}"
        
        
@st.cache_data(ttl=1800, max_entries=10, show_spinner=False)        
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

@st.cache_data(ttl=1800, max_entries=10, show_spinner=False)
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
    except Exception as e:
        return None, f"Error en processar dades del mapa: {e}"
    
def afegir_etiquetes_ciutats(ax, map_extent):
    """
    Versió corregida i robusta. Afegeix etiquetes amb els noms de les ciutats
    si la vista del mapa actual NO és la vista completa de Catalunya.
    """
    # --- LÒGICA DE ZOOM CORREGIDA I SIMPLIFICADA ---
    # Comprovem si l'extensió del mapa actual és diferent de l'extensió per defecte
    # (la de Catalunya completa). Si ho és, significa que l'usuari ha fet zoom.
    
    # Perquè la comparació funcioni, convertim les llistes a tuples
    is_zoomed_in = (tuple(map_extent) != tuple(MAP_EXTENT_CAT))

    if is_zoomed_in:
        # Iterem sobre les ciutats del diccionari
        for ciutat, coords in CIUTATS_CATALUNYA.items():
            lon, lat = coords['lon'], coords['lat']
            
            # Comprovem si la ciutat està dins dels límits del mapa actual
            if map_extent[0] < lon < map_extent[1] and map_extent[2] < lat < map_extent[3]:
                # Dibuixem el text de l'etiqueta
                ax.text(lon + 0.02, lat, ciutat, 
                        fontsize=8, 
                        color='black',
                        transform=ccrs.PlateCarree(), 
                        zorder=20,
                        path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])
                
        
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
    except Exception as e:
        return None, f"Error en processar dades del mapa: {e}"
        
@st.cache_data(ttl=1800, max_entries=5, show_spinner=False)
def obtenir_ciutats_actives(hourly_index):
    """
    Versión optimizada con muestreo reducido
    """
    nivell = 925
    map_data, error_map = carregar_dades_mapa_cat(nivell, hourly_index)
    if error_map or not map_data: 
        return CIUTATS_CONVIDAT, "No s'ha pogut determinar les zones de convergència."
    
    try:
        # Reducir resolución para cálculo más rápido
        lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
        dewpoint_data = np.array(map_data['dewpoint_data'])
        
        # Muestreo para mayor velocidad (máximo 20 puntos)
        if len(lons) > 20:
            idxs = np.random.choice(len(lons), size=min(20, len(lons)), replace=False)
            lons, lats, dewpoint_data = lons[idxs], lats[idxs], dewpoint_data[idxs]
        
        # Búsqueda eficiente de ciudades con alta humedad
        ciudades_activas = []
        umbral_humedad = 12  # Punto de rocío mínimo
        
        for ciutat, coords in CIUTATS_CATALUNYA.items():
            # Calcular distancia a todos los puntos
            distancias = np.sqrt((lats - coords['lat'])**2 + (lons - coords['lon'])**2)
            idx_mas_cercano = np.argmin(distancias)
            
            if (distancias[idx_mas_cercano] < 0.3 and  # Menos de 0.3 grados de distancia
                dewpoint_data[idx_mas_cercano] >= umbral_humedad):
                ciudades_activas.append(ciutat)
        
        # Limitar a 6 ciudades máximo para no saturar
        if ciudades_activas:
            return {name: CIUTATS_CATALUNYA[name] for name in ciudades_activas[:6]}, "Zones actives detectades"
        else:
            return CIUTATS_CONVIDAT, "No s'han detectat zones de convergència significatives."
            
    except Exception as e:
        return CIUTATS_CONVIDAT, f"Error calculant zones actives: {e}"

@st.cache_resource(show_spinner=False)
def precache_datos_iniciales():
    """
    Pre-cache de datos comunes al iniciar la aplicación
    """
    try:
        # Pre-cargar datos que probablemente se usarán
        now_local = datetime.now(TIMEZONE_CAT)
        hourly_index = int((now_local.astimezone(pytz.utc).replace(minute=0, second=0, microsecond=0) - 
                          datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        
        # Pre-cache de ciudades principales
        ciudades_principales = ['Barcelona', 'Girona', 'Lleida', 'Tarragona']
        for ciutat in ciudades_principales:
            coords = CIUTATS_CATALUNYA[ciutat]
            carregar_dades_sondeig_cat(coords['lat'], coords['lon'], hourly_index)
        
        # Pre-cache de mapa básico
        carregar_dades_mapa_cat(925, hourly_index)
        
        return True
    except Exception as e:
        print(f"Pre-caching falló: {e}")
        return False

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
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6,arrowsize=0.3, density= 4.1, zorder=4, transform=ccrs.PlateCarree())
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

def crear_mapa_convergencia_cat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str, map_extent):
    """
    Crea un mapa optimitzat que mostra ÚNICAMENT els nuclis de convergència
    en zones humides, i ara afegeix etiquetes de ciutats quan es fa zoom.
    """
    fig, ax = crear_mapa_base(map_extent)
    
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 150), np.linspace(map_extent[2], map_extent[3], 150))
    
    # Interpolació ràpida
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'linear')

    # Càlcul de la convergència
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    dudx = mpcalc.first_derivative(grid_u * units('m/s'), delta=dx, axis=1)
    dvdy = mpcalc.first_derivative(grid_v * units('m/s'), delta=dy, axis=0)
    convergence_scaled = -(dudx + dvdy).to('1/s').magnitude * 1e5
    
    DEWPOINT_THRESHOLD = 14 if nivell >= 950 else 12
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD, convergence_scaled, 0)
    
    # Dibuix dels contorns
    fill_levels = [15, 25, 40, 60]; fill_colors = ['#ffc107', '#ff9800', '#f44336', '#d32f2f']
    line_levels = [15, 25, 40, 60]; line_colors = ['#e65100', '#bf360c', '#b71c1c', '#880e4f']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    
    labels = ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    for label in labels:
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))

    ax.set_title(f"Nuclis de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=fill_colors[2], alpha=0.6, label='Convergència Alta (>40)'),
                       Patch(facecolor=fill_colors[1], alpha=0.6, label='Convergència Moderada (>25)'),
                       Patch(facecolor=fill_colors[0], alpha=0.6, label='Convergència Feble (>15)')]
    ax.legend(handles=legend_elements, loc='upper left')

    # AFEGIM LES ETIQUETES DE CIUTATS
    afegir_etiquetes_ciutats(ax, map_extent)

    return fig

# --- Funcions Específiques per a Tornado Alley ---

def mostrar_carga_avanzada(mensaje, funcion_a_ejecutar, *args, **kwargs):
    """
    Versión simplificada y funcional
    """
    # Operaciones de navegación (rápidas)
    operaciones_rapidas = ["sortir", "tancar", "canviar", "entrar", "seleccionar", "nav", "zona"]
    
    if any(palabra in mensaje.lower() for palabra in operaciones_rapidas):
        # Navegación: muy rápida
        with st.spinner(f"⚡ {mensaje}"):
            time.sleep(0.8)
        return None
    
    # Operaciones de datos (las que tardan)
    else:
        with st.spinner(f"🌪️ {mensaje}..."):
            return funcion_a_ejecutar(*args, **kwargs)


# Y para las operaciones de navegación, usar mensajes específicos:
def navegacion_rapida(mensaje):
    """Función específica para navegación rápida"""
    with st.spinner(f"⚡ {mensaje}..."):
        time.sleep(1.2)  # Aún más rápido

def mostrar_spinner_mapa(mensaje, funcion_carga, *args, **kwargs):
    """
    Spinner simple que muestra un mensaje mientras carga
    """
    # Mostrar spinner inmediatamente
    with st.spinner(f"🌪️ {mensaje}"):
        try:
            # Ejecutar la función de carga directamente
            result = funcion_carga(*args, **kwargs)
            return result
        except Exception as e:
            st.error(f"Error carregant el mapa: {e}")
            return None, str(e)

@st.cache_data(ttl=1800, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_cat(lat, lon, hourly_index):
    """
    Versió Definitiva i Corregida.
    Soluciona l'error fonamental assegurant que el perfil de pressió passat a
    MetPy sigui sempre monòtonament decreixent, evitant errors en els càlculs.
    """
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS_AROME]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": 4}
        response = openmeteo.weather_api(API_URL_CAT, params=params)[0]
        hourly = response.Hourly()

        valid_index = None; max_hours_to_check = 3
        total_hours = len(hourly.Variables(0).ValuesAsNumpy())
        for offset in range(max_hours_to_check + 1):
            indices_to_try = sorted(list(set([hourly_index + offset, hourly_index - offset])))
            for h_idx in indices_to_try:
                if 0 <= h_idx < total_hours:
                    sfc_check = [hourly.Variables(i).ValuesAsNumpy()[h_idx] for i in range(len(h_base))]
                    if not any(np.isnan(val) for val in sfc_check):
                        valid_index = h_idx; break
            if valid_index is not None: break
        
        if valid_index is None: return None, hourly_index, "No s'han trobat dades vàlides."
        
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[valid_index] for i, v in enumerate(h_base)}
        
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_AROME) + j).ValuesAsNumpy()[valid_index] for j in range(len(PRESS_LEVELS_AROME))]
        
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        
        # --- LÒGICA DE CONSTRUCCIÓ DEL PERFIL CORREGIDA ---
        p_profile = [sfc_data["surface_pressure"]]
        T_profile = [sfc_data["temperature_2m"]]
        Td_profile = [sfc_data["dew_point_2m"]]
        u_profile = [sfc_u.to('m/s').m]
        v_profile = [sfc_v.to('m/s').m]
        h_profile = [0.0]
        
        for i, p_val in enumerate(PRESS_LEVELS_AROME):
            # AQUESTA ÉS LA CONDICIÓ CLAU QUE SOLUCIONA TOT:
            # Només afegim un nivell si la seva pressió és MÉS BAIXA que l'últim punt afegit.
            if p_val < p_profile[-1] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])
        # --- FI DE LA CORRECCIÓ ---

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        return None, hourly_index, f"Error en carregar dades del sondeig AROME: {e}"
        
            
@st.cache_data(ttl=3600)
def carregar_dades_sondeig_usa(lat, lon, hourly_index):
    try:
        h_base = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS_GFS]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "gfs_seamless", "forecast_days": 3}
        response = openmeteo.weather_api(API_URL_USA, params=params)[0]
        hourly = response.Hourly()

        valid_index = None
        max_hours_to_check = 3
        total_hours = len(hourly.Variables(0).ValuesAsNumpy())

        for offset in range(max_hours_to_check + 1):
            indices_to_try = sorted(list(set([hourly_index + offset, hourly_index - offset])))
            for h_idx in indices_to_try:
                if 0 <= h_idx < total_hours:
                    sfc_check = [hourly.Variables(i).ValuesAsNumpy()[h_idx] for i in range(len(h_base))]
                    if not any(np.isnan(val) for val in sfc_check):
                        valid_index = h_idx
                        break
            if valid_index is not None:
                break
        
        if valid_index is None:
            # RETORN SIMPLIFICAT
            return None, hourly_index, "No s'han trobat dades vàlides properes a l'hora sol·licitada."

        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[valid_index] for i, v in enumerate(h_base)}

        sfc_temp_C = sfc_data["temperature_2m"] * units.degC
        sfc_rh_percent = sfc_data["relative_humidity_2m"] * units.percent
        sfc_dew_point = mpcalc.dewpoint_from_relative_humidity(sfc_temp_C, sfc_rh_percent).m
        
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_GFS) + j).ValuesAsNumpy()[valid_index] for j in range(len(PRESS_LEVELS_GFS))]
        
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_dew_point], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for i, p_val in enumerate(PRESS_LEVELS_GFS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        # RETORN SIMPLIFICAT
        return processed_data, valid_index, error
    except Exception as e:
        # RETORN SIMPLIFICAT
        return None, hourly_index, f"Error en carregar dades del sondeig GFS: {e}"

def crear_mapa_vents_cat(lons, lats, speed_data, dir_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent)
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 200), np.linspace(map_extent[2], map_extent[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    
    # Interpolació ràpida
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'linear')
    
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, zorder=3, transform=ccrs.PlateCarree())
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16)
    
    # AFEGIM LES ETIQUETES DE CIUTATS
    afegir_etiquetes_ciutats(ax, map_extent)
    
    return fig

@st.cache_data(ttl=3600)
def carregar_dades_mapa_base_usa(variables, hourly_index):
    try:
        lats, lons = np.linspace(MAP_EXTENT_USA[2], MAP_EXTENT_USA[3], 12), np.linspace(MAP_EXTENT_USA[0], MAP_EXTENT_USA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "gfs_seamless", "forecast_days": 3}
        responses = openmeteo.weather_api(API_URL_USA, params=params)
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
        return None, f"Error en carregar dades del mapa USA: {e}"

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
    except Exception as e:
        return None, f"Error en processar dades del mapa GFS: {e}"


        
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


def format_time_left(total_seconds):
    """Formateja un total de segons en un text llegible (hores i minuts)."""
    if total_seconds <= 0:
        return "ja pots tornar a preguntar"
    
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, _ = divmod(remainder, 60)
    
    if hours > 0:
        return f"d'aquí a {hours}h i {minutes}min"
    else:
        return f"d'aquí a {minutes} min"

def ui_pestanya_assistent_ia(params_calc, poble_sel, pre_analisi, interpretacions_ia):
    """
    Crea la interfície d'usuari per a la pestanya de l'assistent d'IA.
    Ara rep una pre-anàlisi i les interpretacions qualitatives per guiar l'IA.
    """
    st.markdown("#### Assistent d'Anàlisi (IA Gemini)")
    
    is_guest = st.session_state.get('guest_mode', False)
    current_user = st.session_state.get('username')

    if not is_guest:
        st.info(f"ℹ️ Recorda que tens un límit de **{MAX_IA_REQUESTS} consultes cada 3 hores**.")
    else:
        st.info("ℹ️ Fes una pregunta en llenguatge natural sobre les dades del sondeig.")

    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Fes una pregunta sobre el sondeig..."):
        limit_excedit = False
        if not is_guest and current_user:
            now_ts = time.time()
            rate_limits = load_json_file(RATE_LIMIT_FILE)
            user_timestamps = rate_limits.get(current_user, [])
            recent_timestamps = [ts for ts in user_timestamps if now_ts - ts < TIME_WINDOW_SECONDS]
            
            if len(recent_timestamps) >= MAX_IA_REQUESTS:
                limit_excedit = True
                oldest_ts_in_window = recent_timestamps[0]
                time_to_wait = (oldest_ts_in_window + TIME_WINDOW_SECONDS) - now_ts
                temps_restant_str = format_time_left(time_to_wait)
                st.error(f"Has superat el límit de {MAX_IA_REQUESTS} consultes. Podràs tornar a preguntar {temps_restant_str}.")
            else:
                recent_timestamps.append(now_ts)
                rate_limits[current_user] = recent_timestamps
                save_json_file(rate_limits, RATE_LIMIT_FILE)

        if not limit_excedit:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("El teu amic expert està analitzant les dades..."):
                        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt_complet = generar_prompt_per_ia(params_calc, prompt, poble_sel, pre_analisi, interpretacions_ia)
                        
                        response = model.generate_content(prompt_complet)
                        resposta_completa = response.text
                        st.markdown(resposta_completa)

                    st.session_state.messages.append({"role": "assistant", "content": resposta_completa})
                except Exception as e:
                    st.error(f"Hi ha hagut un error en contactar amb l'assistent d'IA: {e}")

def interpretar_parametres(params, nivell_conv):
    """
    Tradueix els paràmetres numèrics clau a categories qualitatives
    per facilitar la interpretació de l'IA.
    """
    interpretacions = {}

    # Interpretació del CIN
    cin = params.get('MLCIN', 0) or 0
    if cin > -25: interpretacions['Inhibició (CIN)'] = 'Gairebé Inexistent'
    elif cin > -75: interpretacions['Inhibició (CIN)'] = 'Febla, fàcil de trencar'
    elif cin > -150: interpretacions['Inhibició (CIN)'] = 'Moderada, cal un bon disparador'
    else: interpretacions['Inhibició (CIN)'] = 'Molt Forta (Tapa de formigó)'

    # Interpretació de la Convergència (Disparador Principal)
    conv_key = f'CONV_{nivell_conv}hPa'
    conv = params.get(conv_key, 0) or 0
    if conv < 5: interpretacions['Disparador (Convergència)'] = 'Molt Febla o Inexistent'
    elif conv < 15: interpretacions['Disparador (Convergència)'] = 'Present'
    elif conv < 30: interpretacions['Disparador (Convergència)'] = 'Moderadament Forta'
    else: interpretacions['Disparador (Convergència)'] = 'Molt Forta i Decisiva'
    
    # Interpretació del CAPE (Combustible)
    mlcape = params.get('MLCAPE', 0) or 0
    if mlcape < 300: interpretacions['Combustible (MLCAPE)'] = 'Molt Baix'
    elif mlcape < 1000: interpretacions['Combustible (MLCAPE)'] = 'Moderat'
    elif mlcape < 2500: interpretacions['Combustible (MLCAPE)'] = 'Alt'
    else: interpretacions['Combustible (MLCAPE)'] = 'Extremadament Alt'

    # Interpretació del Cisallament (Organització)
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    if bwd_6km < 20: interpretacions['Organització (Cisallament)'] = 'Febla (Tempestes desorganitzades)'
    elif bwd_6km < 35: interpretacions['Organització (Cisallament)'] = 'Moderada (Potencial per a multicèl·lules)'
    else: interpretacions['Organització (Cisallament)'] = 'Alta (Potencial per a supercèl·lules)'

    return interpretacions
                

def generar_prompt_per_ia(params, pregunta_usuari, poble, pre_analisi, interpretacions):
    """
    Genera el prompt definitiu (v6.0), que inclou una interpretació qualitativa
    dels paràmetres per a un raonament de l'IA de màxima qualitat.
    """
    # --- ROL I PERSONALITAT ---
    prompt_parts = [
        "### ROL I PERSONALITAT",
        "Ets un expert apassionat de la meteorologia. El teu to és de confiança, didàctic i molt directe, com si donessis un titular clau a un amic.",
        
        "\n### MISSIÓ",
        "Un sistema automàtic ha analitzat un sondeig i t'ha donat un 'Veredicte' i una interpretació qualitativa dels paràmetres clau. La teva única missió és utilitzar aquesta informació per construir una explicació coherent i senzilla per al teu amic.",
        
        "\n### REGLES DE LA RESPOSTA",
        "1. **Comença Directe:** Respon directament a la pregunta del teu amic, basant-te en el 'Veredicte'.",
        "2. **Construeix el Raonament:** Utilitza les 'Interpretacions Clau' per explicar el perquè del veredicte. Centra't primer en el 'Disparador (Convergència)' i després en la 'Inhibició (CIN)'. Aquesta és la lluita principal.",
        "3. **Sigues Breu i Contundent:** La teva resposta ha de ser curta i anar al gra. Màxim 4-5 frases.",

        "\n### ANÀLISI AUTOMÀTICA",
        f"**Localitat:** {poble}",
        f"**Veredicte Final:** {pre_analisi.get('veredicte', 'No determinat')}",
        
        "\n### INTERPRETACIONS CLAU (El que has d'utilitzar per explicar)",
    ]
    # Afegim les interpretacions al prompt
    for key, value in interpretacions.items():
        prompt_parts.append(f"- **{key}:** {value}")

    # Afegim un parell de valors numèrics importants per si l'IA els vol esmentar
    prompt_parts.append("\n### DADES NUMÈRIQUES DE REFERÈNCIA")
    if 'MLCAPE' in params and pd.notna(params['MLCAPE']): prompt_parts.append(f"- MLCAPE exacte: {params['MLCAPE']:.0f} J/kg")
    conv_key = next((k for k in params if k.startswith('CONV_')), None)
    if conv_key and conv_key in params and pd.notna(params[conv_key]): prompt_parts.append(f"- Convergència exacta: {params[conv_key]:.1f}")
    
    prompt_parts.append("\n### INSTRUCCIÓ FINAL")
    prompt_parts.append(f"Ara, escriu la teva anàlisi breu i directa. La pregunta del teu amic és: \"{pregunta_usuari}\"")

    return "\n".join(prompt_parts)
    
def hide_streamlit_style():
    """Injecta CSS per amagar el peu de pàgina i el menú de Streamlit."""
    hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Neteja de l'estat visual quan canvia la selecció */
        .stApp {
            transition: all 0.3s ease;
        }
        .element-container:has(.stSelectbox) {
            z-index: 1000;
            position: relative;
        }
        </style>
        """
    st.markdown(hide_style, unsafe_allow_html=True)


def ui_capcalera_selectors(ciutats_a_mostrar, info_msg=None, zona_activa="catalunya", convergencies=None):
    st.markdown(f'<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | {zona_activa.replace("_", " ").title()}</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)
    
    col_text, col_change, col_logout = st.columns([0.7, 0.15, 0.15])
    with col_text:
        if not is_guest: st.markdown(f"Benvingut/da, **{st.session_state.get('username')}**!")
    with col_change:
        if st.button("Canviar Anàlisi", use_container_width=True):
            for key in ['poble_selector', 'poble_selector_usa', 'zone_selected', 'active_tab_cat', 'active_tab_usa', 'selector_terra', 'selector_mar', 'last_terra_sel', 'last_mar_sel']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col_logout:
        if st.button("Sortir" if is_guest else "Tanca Sessió", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    with st.container(border=True):
        def formatar_llista_ciutats(ciutats_dict, conv_data):
            if not conv_data:
                return sorted(list(ciutats_dict.keys()))

            ciutats_amb_conv = []
            ciutats_sense_conv = []

            for city in sorted(ciutats_dict.keys()):
                conv = conv_data.get(city)
                
                if conv is not None and pd.notna(conv):
                    emoji = ""
                    if conv >= 40:   emoji = "🔴"
                    elif conv >= 30: emoji = "🟠"
                    elif conv >= 15: emoji = "🟡"
                    
                    if emoji:
                        text_formatat = f"{city} ({emoji} {conv:.0f})"
                    else:
                        text_formatat = city
                    ciutats_amb_conv.append((text_formatat, conv))
                else:
                    ciutats_sense_conv.append(city)
            
            ciutats_ordenades = sorted(ciutats_amb_conv, key=lambda item: item[1], reverse=True)
            llista_final = [item[0] for item in ciutats_ordenades]
            return llista_final + ciutats_sense_conv

        if zona_activa == 'catalunya':
            col_terra, col_mar, col_dia, col_hora, col_nivell = st.columns(5)
            PLACEHOLDER_TERRA = "--- Selecciona Població ---"
            PLACEHOLDER_MAR = "--- Selecciona Punt Marí ---"
            
            # Aquest és el text que apareixerà a l'interrogant
            tooltip_text = "Mostra els punts amb major convergència ('disparador' de tempestes).\n\nLlegenda:\n- 🟡 (>15): Moderada\n- 🟠 (>30): Alta\n- 🔴 (>40): Molt Alta\n\nEl valor és la força de la convergència."

            def handle_selection_change():
                terra_sel = st.session_state.selector_terra
                mar_sel = st.session_state.selector_mar
                if terra_sel != st.session_state.get('last_terra_sel', ''):
                    if terra_sel != PLACEHOLDER_TERRA:
                        clau_original = terra_sel.split(' (')[0]
                        if clau_original in POBLACIONS_TERRA:
                            st.session_state.poble_selector = clau_original
                        st.session_state.selector_mar = PLACEHOLDER_MAR
                elif mar_sel != st.session_state.get('last_mar_sel', ''):
                    if mar_sel != PLACEHOLDER_MAR:
                        clau_original = mar_sel.split(' (')[0]
                        if clau_original in PUNTS_MAR:
                            st.session_state.poble_selector = clau_original
                        st.session_state.selector_terra = PLACEHOLDER_TERRA
                st.session_state.last_terra_sel = st.session_state.selector_terra
                st.session_state.last_mar_sel = st.session_state.selector_mar

            with col_terra:
                opcions = [PLACEHOLDER_TERRA] + formatar_llista_ciutats(POBLACIONS_TERRA, convergencies)
                idx = 0
                poble_actual = st.session_state.get('poble_selector')
                if poble_actual:
                    try:
                        idx = next(i for i, opt in enumerate(opcions) if opt.startswith(poble_actual))
                    except (ValueError, StopIteration):
                        idx = 0
                # *** LÍNIA CLAU MODIFICADA ***
                st.selectbox("Població:", opcions, key="selector_terra", index=idx, on_change=handle_selection_change, help=tooltip_text)

            with col_mar:
                opcions = [PLACEHOLDER_MAR] + formatar_llista_ciutats(PUNTS_MAR, convergencies)
                idx = 0
                poble_actual = st.session_state.get('poble_selector')
                if poble_actual:
                    try:
                        idx = next(i for i, opt in enumerate(opcions) if opt.startswith(poble_actual))
                    except (ValueError, StopIteration):
                        idx = 0
                # *** LÍNIA CLAU MODIFICADA ***
                st.selectbox("Punt Marí:", opcions, key="selector_mar", index=idx, on_change=handle_selection_change, help=tooltip_text)

            now_local = datetime.now(TIMEZONE_CAT)
            with col_dia: st.selectbox("Dia:", ("Avui",) if is_guest else ("Avui", "Demà"), key="dia_selector", disabled=is_guest)
            with col_hora: st.selectbox("Hora:", (f"{now_local.hour:02d}:00h",) if is_guest else [f"{h:02d}:00h" for h in range(24)], key="hora_selector", disabled=is_guest)
            with col_nivell:
                if not is_guest:
                    nivells = [1000, 950, 925, 900, 850, 800, 700]
                    st.selectbox("Nivell:", nivells, key="level_cat_main", index=3, format_func=lambda x: f"{x} hPa")
                else: st.session_state.level_cat_main = 925
        
        else: # Zona USA
            # Aquesta part no canvia
            pass

                
def ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel):
    st.markdown("#### Mapes de Pronòstic (Model AROME)")
    
    col_capa, col_zoom = st.columns(2)
    with col_capa:
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", 
                               ["Anàlisi de Vent i Convergència", "Vent a 700hPa", "Vent a 300hPa"], 
                               key="map_cat")
    with col_zoom: 
        zoom_sel = st.selectbox("Nivell de Zoom:", 
                               options=list(MAP_ZOOM_LEVELS_CAT.keys()), 
                               key="zoom_cat")
    
    selected_extent = MAP_ZOOM_LEVELS_CAT[zoom_sel]
    
    if "Convergència" in mapa_sel:
        # USAR SPINNER SIMPLE
        result = mostrar_spinner_mapa(
            "Generant mapa de convergència...", 
            carregar_dades_mapa_cat, 
            nivell_sel, hourly_index_sel
        )
        
        if result is not None:
            map_data, error_map = result
            if error_map: 
                st.error(f"Error en carregar el mapa: {error_map}")
            elif map_data:
                fig = crear_mapa_forecast_combinat_cat(
                    map_data['lons'], map_data['lats'], 
                    map_data['speed_data'], map_data['dir_data'], 
                    map_data['dewpoint_data'], nivell_sel, 
                    timestamp_str, selected_extent
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    
    else:
        # Para otros mapas...
        nivell = 700 if "700" in mapa_sel else 300
        variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        
        result = mostrar_spinner_mapa(
            f"Carregant vent a {nivell}hPa...",
            carregar_dades_mapa_base_cat,
            variables, hourly_index_sel
        )
        
        if result is not None:
            map_data, error_map = result
            if error_map: 
                st.error(f"Error: {error_map}")
            elif map_data: 
                fig = crear_mapa_vents_cat(
                    map_data['lons'], map_data['lats'], 
                    map_data[variables[0]], map_data[variables[1]], 
                    nivell, timestamp_str, selected_extent
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            
def ui_pestanya_vertical(data_tuple, poble_sel, lat, lon, nivell_conv, hora_actual):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        # Desempaquetem 'prof' (la trajectòria de superfície)
        p, T, Td, u, v, heights, prof = sounding_data
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            # Tornem a passar 'prof' i 'titol' com a arguments
            fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Sondeig Vertical\n{poble_sel}")
            
            st.pyplot(fig_skewt, use_container_width=True)
            plt.close(fig_skewt)
            
            with st.container(border=True):
                # *** LÍNIA CLAU MODIFICADA ***
                # Ara passem 'sounding_data' a la funció que dibuixa els paràmetres
                ui_caixa_parametres_sondeig(sounding_data, params_calculats, nivell_conv, hora_actual)

        with col2:
            fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat\n{poble_sel}")
            st.pyplot(fig_hodo, use_container_width=True)
            plt.close(fig_hodo)
            
            st.markdown("##### Radar de Precipitació en Temps Real")
            radar_url = f"https://www.rainviewer.com/map.html?loc={lat},{lon},8&oCS=1&c=3&o=83&lm=0&layer=radar&sm=1&sn=1&ts=2&play=1"
            html_code = f"""<div style="position: relative; width: 100%; height: 410px; border-radius: 10px; overflow: hidden;"><iframe src="{radar_url}" width="100%" height="410" frameborder="0" style="border:0;"></iframe><div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; cursor: default;"></div></div>"""
            st.components.v1.html(html_code, height=410)
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")
    

def ui_pestanya_mapes_usa(hourly_index_sel, timestamp_str, nivell_sel):
    st.markdown("#### Mapes de Pronòstic (Model GFS)")
    
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
    # --- PAS 1: RECOLLIR TOTS ELS INPUTS DE L'USUARI ---
    is_guest = st.session_state.get('guest_mode', False)
    
    if 'poble_selector' not in st.session_state: st.session_state.poble_selector = "Barcelona"
    if 'dia_selector' not in st.session_state: st.session_state.dia_selector = "Avui"
    if 'hora_selector' not in st.session_state: st.session_state.hora_selector = f"{datetime.now(TIMEZONE_CAT).hour:02d}:00h"
    if 'level_cat_main' not in st.session_state: st.session_state.level_cat_main = 925

    pre_hora_sel = st.session_state.hora_selector
    pre_dia_sel = st.session_state.dia_selector
    pre_target_date = datetime.now(TIMEZONE_CAT).date() + timedelta(days=1) if pre_dia_sel == "Demà" else datetime.now(TIMEZONE_CAT).date()
    pre_local_dt = TIMEZONE_CAT.localize(datetime.combine(pre_target_date, datetime.min.time()).replace(hour=int(pre_hora_sel.split(':')[0])))
    pre_start_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    pre_hourly_index = int((pre_local_dt.astimezone(pytz.utc) - pre_start_utc).total_seconds() / 3600)
    
    pre_map_data, _ = carregar_dades_mapa_cat(st.session_state.level_cat_main, pre_hourly_index)
    pre_convergencies = calcular_convergencies_per_llista(pre_map_data, CIUTATS_CATALUNYA) if pre_map_data else {}
    
    ui_capcalera_selectors(None, None, zona_activa="catalunya", convergencies=pre_convergencies)

    # --- PAS 2: LLEGIR L'ESTAT FINAL I CARREGAR DADES ---
    poble_sel = st.session_state.poble_selector
    if not poble_sel or "---" in poble_sel:
        st.info("Selecciona una població o un punt marítim per començar l'anàlisi.")
        return

    dia_sel_str = st.session_state.dia_selector
    hora_sel_str = st.session_state.hora_selector
    nivell_sel = 925 if is_guest else st.session_state.level_cat_main
    lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']
    
    target_date = datetime.now(TIMEZONE_CAT).date() + timedelta(days=1) if dia_sel_str == "Demà" else datetime.now(TIMEZONE_CAT).date()
    local_dt = TIMEZONE_CAT.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_str.split(':')[0])))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    timestamp_str = f"{dia_sel_str} a les {hora_sel_str} (Hora Local)"

    # --- PAS 3: DIBUIXAR EL MENÚ I MOSTRAR RESULTATS ---
    menu_options = ["Anàlisi de Mapes", "Anàlisi Vertical"] if is_guest else ["Anàlisi de Mapes", "Anàlisi Vertical", "💬 Assistent IA"]
    menu_icons = ["map", "graph-up-arrow"] if is_guest else ["map", "graph-up-arrow", "chat-quote-fill"]

    if 'active_tab_cat' not in st.session_state: st.session_state.active_tab_cat = menu_options[0]
    try: default_idx = menu_options.index(st.session_state.active_tab_cat)
    except ValueError: default_idx = 0

    selected_tab = option_menu(
        menu_title=None, options=menu_options, icons=menu_icons,
        menu_icon="cast", default_index=default_idx, orientation="horizontal", key="catalunya_nav"
    )
    st.session_state.active_tab_cat = selected_tab

    if selected_tab == "Anàlisi de Mapes":
        ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel)
    elif selected_tab in ["Anàlisi Vertical", "💬 Assistent IA"]:
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_cat(lat_sel, lon_sel, hourly_index_sel)
        
        if not error_msg and final_index != hourly_index_sel:
            adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
            adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_CAT)
            st.warning(f"**Avís:** No hi havia dades per a les {hora_sel_str}. Es mostren les de l'hora més propera: **{adjusted_local_time.strftime('%H:%Mh')}**.")

        if error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
        else:
            params_calc = data_tuple[1] if data_tuple else {}
            if poble_sel in pre_convergencies:
                params_calc[f'CONV_{nivell_sel}hPa'] = pre_convergencies.get(poble_sel)
            
            analisi_temps = analitzar_potencial_meteorologic(params_calc, nivell_sel, hora_sel_str)
            
            if selected_tab == "Anàlisi Vertical":
                ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str)
            
            elif selected_tab == "💬 Assistent IA":
                interpretacions_ia = interpretar_parametres(params_calc, nivell_sel)
                ui_pestanya_assistent_ia(params_calc, poble_sel, analisi_temps, interpretacions_ia)

def run_valley_halley_app():
    # --- PAS 0: INICIALITZACIÓ DE L'ESTAT (AMB L'HORA LOCALITZADA I FORMATADA) ---
    if 'poble_selector_usa' not in st.session_state:
        st.session_state.poble_selector_usa = "Oklahoma City, OK"
    if 'dia_selector_usa' not in st.session_state:
        st.session_state.dia_selector_usa = "Avui"
    
    if 'hora_selector_usa' not in st.session_state:
        now_spain = datetime.now(TIMEZONE_CAT)
        time_in_usa = now_spain.astimezone(TIMEZONE_USA)
        # Guardem el text complet amb les dues hores com a valor per defecte
        st.session_state.hora_selector_usa = f"{time_in_usa.hour:02d}:00 (Local: {now_spain.hour:02d}:00h)"
        
    if 'level_usa_main' not in st.session_state:
        st.session_state.level_usa_main = 850

    # --- PAS 1: RECOLLIR TOTS ELS INPUTS DE L'USUARI ---
    # Per als pre-càlculs, hem d'extreure l'hora CST del text
    pre_hora_sel_text = st.session_state.hora_selector_usa
    pre_hora_sel_cst = pre_hora_sel_text.split(' ')[0]

    pre_now_usa = datetime.now(TIMEZONE_USA)
    pre_dia_sel = st.session_state.dia_selector_usa
    pre_day_offset = {"Avui": 0, "Demà": 1, "Demà passat": 2}[pre_dia_sel]
    pre_target_date = pre_now_usa.date() + timedelta(days=pre_day_offset)
    pre_local_dt = TIMEZONE_USA.localize(datetime.combine(pre_target_date, datetime.min.time()).replace(hour=int(pre_hora_sel_cst.split(':')[0])))
    pre_start_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    pre_hourly_index = int((pre_local_dt.astimezone(pytz.utc) - pre_start_utc).total_seconds() / 3600)

    NIVELL_ANALISI_CONV = 850
    pre_map_data, _ = carregar_dades_mapa_usa(NIVELL_ANALISI_CONV, pre_hourly_index)
    pre_convergencies = calcular_convergencies_per_llista(pre_map_data, USA_CITIES) if pre_map_data else {}
    
    ui_capcalera_selectors(None, zona_activa="tornado_alley", convergencies=pre_convergencies)

    # --- PAS 2: LLEGIR L'ESTAT FINAL I CARREGAR DADES ---
    poble_sel = st.session_state.poble_selector_usa
    dia_sel_str = st.session_state.dia_selector_usa
    hora_sel_str_full = st.session_state.hora_selector_usa
    hora_sel_cst_only = hora_sel_str_full.split(' ')[0] # Extraiem només l'hora CST per als càlculs
    nivell_sel = st.session_state.level_usa_main
    lat_sel, lon_sel = USA_CITIES[poble_sel]['lat'], USA_CITIES[poble_sel]['lon']

    day_offset = {"Avui": 0, "Demà": 1, "Demà passat": 2}[dia_sel_str]
    target_date = datetime.now(TIMEZONE_USA).date() + timedelta(days=day_offset)
    local_dt = TIMEZONE_USA.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_cst_only.split(':')[0])))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    timestamp_str = f"{dia_sel_str} a les {hora_sel_cst_only} (Central Time)"

    # --- PAS 3: DIBUIXAR EL MENÚ I MOSTRAR RESULTATS ---
    menu_options_usa = ["Anàlisi de Mapes", "Anàlisi Vertical", "Satèl·lit (Temps Real)"]
    menu_icons_usa = ["map-fill", "graph-up-arrow", "globe-americas"]

    if 'active_tab_usa' not in st.session_state: st.session_state.active_tab_usa = menu_options_usa[0]
    try: default_idx_usa = menu_options_usa.index(st.session_state.active_tab_usa)
    except ValueError: default_idx_usa = 0

    selected_tab_usa = option_menu(
        menu_title=None, options=menu_options_usa, icons=menu_icons_usa,
        menu_icon="cast", default_index=default_idx_usa, orientation="horizontal", key="usa_nav"
    )
    st.session_state.active_tab_usa = selected_tab_usa

    if selected_tab_usa == "Anàlisi de Mapes":
        with st.spinner(f"Carregant mapa GFS a {nivell_sel}hPa..."):
            map_data_final, _ = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
        if map_data_final:
            fig = crear_mapa_forecast_combinat_usa(map_data_final['lons'], map_data_final['lats'], map_data_final['speed_data'], map_data_final['dir_data'], map_data_final['dewpoint_data'], nivell_sel, timestamp_str)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning(f"No s'han pogut carregar les dades del mapa per al nivell {nivell_sel}hPa.")
            
    elif selected_tab_usa == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_usa(lat_sel, lon_sel, hourly_index_sel)
        
        if not error_msg and final_index != hourly_index_sel:
            adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
            adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_USA)
            st.warning(f"**Avís:** No hi havia dades disponibles per a les {hora_sel_str_full}. Es mostren les de l'hora més propera: **{adjusted_local_time.strftime('%H:%M')}** (Central Time).")

        if error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
        else:
            params_calc = data_tuple[1] if data_tuple else {}
            if nivell_sel == NIVELL_ANALISI_CONV and poble_sel in pre_convergencies:
                params_calc[f'CONV_{nivell_sel}hPa'] = pre_convergencies[poble_sel]
            else:
                with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                    map_data_nivell_sel, _ = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
                    if map_data_nivell_sel:
                        params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_nivell_sel, lat_sel, lon_sel)
            
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_cst_only)
        
    elif selected_tab_usa == "Satèl·lit (Temps Real)":
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
    inject_custom_css()
    hide_streamlit_style()
    
    if 'precache_completat' not in st.session_state:
        st.session_state.precache_completat = False
        
    if not st.session_state.precache_completat:
        try:
            precache_datos_iniciales()
            st.session_state.precache_completat = True
        except:
            pass
    
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'guest_mode' not in st.session_state: st.session_state['guest_mode'] = False
    if 'zone_selected' not in st.session_state: st.session_state['zone_selected'] = None

    if not st.session_state['logged_in']:
        show_login_page()
    elif not st.session_state['zone_selected']:
        ui_zone_selection()
    
    # --- INICI DEL CANVI ---
    # Ara, envolv_involucrem cada crida a una funció principal amb un spinner.
    # Aquest spinner actuarà com una pantalla de càrrega que amaga la interfície anterior.
    elif st.session_state['zone_selected'] == 'catalunya':
        with st.spinner("Preparant l'entorn d'anàlisi de Catalunya..."):
            run_catalunya_app()
            
    elif st.session_state['zone_selected'] == 'valley_halley':
        with st.spinner("Preparant l'entorn d'anàlisi de Tornado Alley..."):
            run_valley_halley_app()
    # --- FI DEL CANVI ---


def analitzar_potencial_meteorologic(params, nivell_conv, hora_actual=None):
    """
    Sistema de Diagnòstic Meteorològic Expert v14.0 - LÒGICA FINAL
    Implementa una clàusula d'excepció per a casos de forçament dinàmic extrem
    (convergència molt alta), que pot superar LFCs alts o CIN moderat.
    """
    # --- 0. PREPARACIÓ ---
    es_de_nit = False
    if hora_actual:
        try:
            hora = int(hora_actual.split(':')[0])
            es_de_nit = (hora >= 21 or hora <= 6)
        except (ValueError, AttributeError): es_de_nit = False

    # --- 1. EXTRACCIÓ DE PARÀMETRES ---
    mlcape = params.get('MLCAPE', 0) or 0
    mucape = params.get('MUCAPE', 0) or 0
    cin = params.get('MLCIN', params.get('SBCIN', 0)) or 0
    li = params.get('LI', 5) or 5
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_1km = params.get('SRH_0-1km', 0) or 0
    srh_3km = params.get('SRH_0-3km', 0) or 0
    lcl_hgt = params.get('LCL_Hgt', 9999) or 9999
    lfc_hgt = params.get('LFC_Hgt', 9999) or 9999
    rh_capes = params.get('RH_CAPES', {'baixa': 0, 'mitjana': 0, 'alta': 0})
    max_updraft = params.get('MAX_UPDRAFT', 0) or 0
    freezing_lvl_hgt = params.get('FREEZING_LVL_HGT', 9999) or 9999
    cape_0_3km = params.get('CAPE_0-3km', 0) or 0
    dcape = params.get('DCAPE', 0) or 0
    pwat = params.get('PWAT', 0) or 0
    conv_key = f'CONV_{nivell_conv}hPa'
    conv = params.get(conv_key, 0) or 0

    # --- 2. AVALUACIÓ DEL DISPARADOR ---
    hi_ha_inestabilitat_latent = (mucape > 150 or li < -1)
    trigger_potential = 'Nul'
    if hi_ha_inestabilitat_latent:
        FACTOR_CONV = 5.0; cin_efectiu = abs(min(0, cin))
        forçament_dinamic = (conv * FACTOR_CONV) if conv > 1 else 0
        forçament_net = forçament_dinamic - cin_efectiu
        if conv >= 40 and forçament_net > -100: trigger_potential = 'Extrem' # <-- NOU NIVELL
        elif conv >= 30 and forçament_net > -75: trigger_potential = 'Fort'
        elif conv >= 15 and forçament_net > -40: trigger_potential = 'Moderat'
        elif conv >= 5 and forçament_net > -20: trigger_potential = 'Feble'
        elif cin_efectiu < 15: trigger_potential = 'Feble'

    # --- 3. DIAGNÒSTIC JERÀRQUIC AVANÇAT ---
    
    # --- NOVA CLÀUSULA D'EXCEPCIÓ PER FORÇAMENT EXTREM ---
    if trigger_potential == 'Extrem' and mlcape > 500:
        desc_amenaces = ""
        if max_updraft > 30: desc_amenaces += " amb Risc de Calamarsa"
        if dcape > 1000: desc_amenaces += " i Fortes Ventades"
        return {'emoji': "⛈️", 'descripcio': "Tempestes Forçades" + desc_amenaces, 
                'veredicte': f"Potencial de tempestes severes forçades per una línia de convergència molt intensa{desc_amenaces}.", 
                'factor_clau': "Convergència extrema (>40), que actua com un pistó i pot trencar tapes d'inversió significatives."}
    
    # Prioritat 1: Tempestes Severes (condicions normals)
    cape_real = mlcape
    if trigger_potential in ['Fort', 'Moderat'] and cape_real > 1000 and bwd_6km > 20 and lfc_hgt < 3000:
        # ... (la lògica de diagnòstic de severitat es manté igual) ...
        desc_calamarsa = ""; desc_vent = ""; desc_pluja = ""
        if max_updraft > 35 and freezing_lvl_hgt < 4000:
            if max_updraft > 50: desc_calamarsa = "Calamarsa Severa"
            else: desc_calamarsa = "Risc de Calamarsa"
        if dcape > 1000: desc_vent = "Fortes Ventades"
        if pwat > 45: desc_pluja = "Pluges Torrencials"
        amenaces = [a for a in [desc_calamarsa, desc_vent, desc_pluja] if a]
        desc_amenaces = f" ({', '.join(amenaces)})" if amenaces else ""
        if bwd_6km >= 35 and srh_3km > 150 and cape_0_3km > 100:
            desc = "Supercèl·lula"
            if srh_1km > 150 and lcl_hgt < 1200: desc += " (Pot. Tornàdic)"
            return {'emoji': "🌪️", 'descripcio': desc + desc_amenaces, 'veredicte': f"Potencial de {desc}{desc_amenaces}.", 'factor_clau': "Excel·lent combinació d'energia a nivells baixos, cisallament i helicitat."}
        if bwd_6km >= 20 and cape_0_3km >= 50:
            return {'emoji': "⛈️", 'descripcio': "Grup de tempestes" + desc_amenaces, 'veredicte': f"Potencial per a un grup de tempestes repartides{desc_amenaces}.", 'factor_clau': "Bona combinació d'energia i cisallament que afavoreix l'organització."}

    # Prioritat 2: Tempestes Comunes o Elevades
    if trigger_potential != 'Nul' and mucape > 700:
        desc_calamarsa = " amb Risc de Calamarsa" if max_updraft > 25 and freezing_lvl_hgt < 4200 else ""
        if mlcape < 300 and mucape > 800:
            return {'emoji': "🌩️", 'descripcio': "Tempesta de Base Alta" + desc_calamarsa, 'veredicte': f"Tempestes que es formen a nivells mitjans{desc_calamarsa}.", 'factor_clau': "Forta inestabilitat elevada (MUCAPE) que supera una capa estable a la superfície."}
        if cape_real > 500: # Afegim una condició mínima de CAPE real
            return {'emoji': "🌩️", 'descripcio': "Tempesta Aïllada" + desc_calamarsa, 'veredicte': f"Potencial de tempestes aïllades{desc_calamarsa}.", 'factor_clau': "Inestabilitat suficient i un disparador efectiu, però sense prou organització."}

    # Prioritat 3: Núvols Convectius (sense arribar a tempesta)
    if trigger_potential != 'Nul':
        if mucape > 250 and mlcape < 150 and lcl_hgt > 1800 and lfc_hgt < 4000:
            return {'emoji': "🌥️", 'descripcio': "Inestabilitat (Castellanus)", 'veredicte': "Inestabilitat a nivells mitjans, convecció elevada.", 'factor_clau': "MUCAPE alt amb MLCAPE gairebé nul."}
        if 300 < mlcape <= 700 and cin > -50 and lfc_hgt < 2500:
            return {'emoji': "☁️", 'descripcio': "Desenvolupament Vertical (Congestus)", 'veredicte': "Núvols de gran creixement que probablement no seran tempesta.", 'factor_clau': "Inestabilitat moderada i LFC baix."}
        if 50 < mlcape <= 300 and cin > -25:
            return {'emoji': "🌤️", 'descripcio': "Núvols de Bon Temps (Humilis)", 'veredicte': "Formació de petits cúmuls de bon temps.", 'factor_clau': "Molt poca inestabilitat."}

    # Prioritat 4 i 5: Núvols Estables i Cel Serè
    rh_baixa = rh_capes.get('baixa', 0) if pd.notna(rh_capes.get('baixa')) else 0
    rh_mitjana = rh_capes.get('mitjana', 0) if pd.notna(rh_capes.get('mitjana')) else 0
    if rh_baixa > 85 and rh_mitjana > 80: return {'emoji': "🌧️", 'descripcio': "Pluja/Plugim (Nimboestratus)", 'veredicte': "Precipitació contínua.", 'factor_clau': "Capa d'humitat molt profunda i saturada."}
    if lcl_hgt < 150 and rh_baixa > 95: return {'emoji': "🌫️", 'descripcio': "Boira o Boirina", 'veredicte': "Visibilitat reduïda.", 'factor_clau': "Saturació d'humitat a la superfície."}
    if rh_baixa > 75: 
        desc = "Cel Cobert (Estratus/Estratocúmulus)"
        if lcl_hgt < 800: desc = "Cel Cobert (Estratus)"
        return {'emoji': "☁️", 'descripcio': desc, 'veredicte': "Cel tapat amb núvols baixos.", 'factor_clau': "Capa d'humitat a nivells baixos."}
    if rh_mitjana > 70: return {'emoji': "🌥️", 'descripcio': "Núvols Mitjans (Altocúmulus)", 'veredicte': "Cel variable amb núvols mitjans.", 'factor_clau': "Capa d'humitat a nivells mitjans."}
    if rh_capes.get('alta', 0) > 60: return {'emoji': "🌤️", 'descripcio': "Núvols Alts (Cirrus)", 'veredicte': "Cel poc ennuvolat amb núvols alts.", 'factor_clau': "Humitat només a nivells molt alts."}
    
    return {'emoji': "☀️", 'descripcio': "Cel Serè", 'veredicte': "Temps estable i sense nuvolositat.", 'factor_clau': "Atmosfera seca."}
if __name__ == "__main__":
    main()
