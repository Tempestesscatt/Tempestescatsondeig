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

def processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile):
    """Funció genèrica que processa dades de sondeig amb MetPy."""
    if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
    
    valid_indices = ~np.isnan(p_profile) & ~np.isnan(T_profile) & ~np.isnan(Td_profile) & ~np.isnan(u_profile) & ~np.isnan(v_profile)
    p = np.array(p_profile)[valid_indices] * units.hPa
    T = np.array(T_profile)[valid_indices] * units.degC
    Td = np.array(Td_profile)[valid_indices] * units.degC
    u = np.array(u_profile)[valid_indices] * units('m/s')
    v = np.array(v_profile)[valid_indices] * units('m/s')
    heights = np.array(h_profile)[valid_indices] * units.meter
    
    params_calc = {}; prof = None; heights_agl = heights - heights[0]
    with parcel_lock:
        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        try:
            sbcape, sbcin = mpcalc.cape_cin(p, T, Td, prof); params_calc['SBCAPE'] = sbcape.m; params_calc['SBCIN'] = sbcin.m
            params_calc['MAX_UPDRAFT'] = np.sqrt(2 * sbcape.m) if sbcape.m > 0 else 0.0
        except Exception: params_calc.update({'SBCAPE': np.nan, 'SBCIN': np.nan, 'MAX_UPDRAFT': np.nan})
        try: mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, as_pentads=False); params_calc['MUCAPE'] = mucape.m; params_calc['MUCIN'] = mucin.m
        except Exception: params_calc.update({'MUCAPE': np.nan, 'MUCIN': np.nan})
        try: mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=100 * units.hPa); params_calc['MLCAPE'] = mlcape.m; params_calc['MLCIN'] = mlcin.m
        except Exception: params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
        try: li, _ = mpcalc.lifted_index(p, T, prof); params_calc['LI'] = li.m
        except Exception: params_calc['LI'] = np.nan
        try: dcape, _ = mpcalc.dcape(p, T, Td, prof); params_calc['DCAPE'] = dcape.m
        except Exception: params_calc['DCAPE'] = np.nan
        try:
            lfc_p, _ = mpcalc.lfc(p, T, Td, prof); params_calc['LFC_p'] = lfc_p.m
            params_calc['LFC_Hgt'] = np.interp(lfc_p.m, p.m[::-1], heights_agl.m[::-1])
        except Exception: params_calc.update({'LFC_p': np.nan, 'LFC_Hgt': np.nan})
        try:
            el_p, _ = mpcalc.el(p, T, Td, prof); params_calc['EL_p'] = el_p.m
            params_calc['EL_Hgt'] = np.interp(el_p.m, p.m[::-1], heights_agl.m[::-1])
        except Exception: params_calc.update({'EL_p': np.nan, 'EL_Hgt': np.nan})
        try:
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); params_calc['LCL_p'] = lcl_p.m
            params_calc['LCL_Hgt'] = np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1])
        except Exception: params_calc.update({'LCL_p': np.nan, 'LCL_Hgt': np.nan})
        try: pwat = mpcalc.precipitable_water(p, Td); params_calc['PWAT'] = pwat.to('mm').m
        except Exception: params_calc['PWAT'] = np.nan
        try: frz_lvl, _ = mpcalc.freezing_level(p, T); params_calc['FRZG_Lvl_p'] = frz_lvl.m
        except Exception: params_calc['FRZG_Lvl_p'] = np.nan
    try:
        rm, lm, mean_wind = mpcalc.storm_motion(p, u, v, heights)
        params_calc['RM'] = (rm[0].m, rm[1].m); params_calc['LM'] = (lm[0].m, lm[1].m); params_calc['Mean_Wind'] = (mean_wind[0].m, mean_wind[1].m)
    except Exception: params_calc.update({'RM': (np.nan, np.nan), 'LM': (np.nan, np.nan), 'Mean_Wind': (np.nan, np.nan)})
    for name, depth_m in [('0-1km', 1000), ('0-6km', 6000)]:
        try:
            bwd_u, bwd_v = mpcalc.bulk_shear(p, u, v, height=heights_agl, depth=depth_m * units.m)
            params_calc[f'BWD_{name}'] = mpcalc.wind_speed(bwd_u, bwd_v).to('kt').m
        except Exception: params_calc[f'BWD_{name}'] = np.nan
    if not np.isnan(params_calc.get('RM', [(np.nan, np.nan)])[0]):
        try:
            u_storm, v_storm = params_calc['RM'] * units('m/s')
            for name, depth_m in [('0-1km', 1000), ('0-3km', 3000)]:
                srh, _, _ = mpcalc.storm_relative_helicity(heights, u, v, depth=depth_m * units.m, storm_u=u_storm, storm_v=v_storm)
                params_calc[f'SRH_{name}'] = srh.m
        except Exception: params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})
    else: params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})
    try:
        idx_3km = np.argmin(np.abs(heights_agl.m - 3000))
        cape_0_3, _ = mpcalc.cape_cin(p[:idx_3km+1], T[:idx_3km+1], Td[:idx_3km+1], prof[:idx_3km+1])
        params_calc['CAPE_0-3km'] = cape_0_3.m
    except Exception: params_calc['CAPE_0-3km'] = np.nan
    
    return ((p, T, Td, u, v, heights, prof), params_calc), None

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
        if p_lvl is not None and not np.isnan(p_lvl):
            p_val = p_lvl.m if hasattr(p_lvl, 'm') else p_lvl
            skew.ax.axhline(p_val, color='blue', linestyle='--', linewidth=1.5)
            skew.ax.text(skew.ax.get_xlim()[1] - 2, p_val, f' {name}', color='blue', ha='right', va='center', fontsize=10, weight='bold')
    skew.ax.legend(); return fig

def crear_hodograf_avancat(p, u, v, heights, params_calc, titol):
    fig = plt.figure(dpi=150, figsize=(8, 8))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.5, 6], width_ratios=[1.5, 1], hspace=0.4, wspace=0.3)
    ax_barbs = fig.add_subplot(gs[0, :]); ax_hodo = fig.add_subplot(gs[1, 0]); ax_params = fig.add_subplot(gs[1, 1])
    fig.suptitle(titol, weight='bold', fontsize=16)
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
    ax_barbs.set_xticks(x_pos); ax_barbs.set_xticklabels([f"{h} km" for h in barb_altitudes_km])
    ax_barbs.set_yticks([]); ax_barbs.spines[:].set_visible(False)
    ax_barbs.tick_params(axis='x', length=0, pad=5); ax_barbs.set_xlim(-0.5, len(barb_altitudes_km) - 0.5); ax_barbs.set_ylim(-1.5, 1.5)
    h = Hodograph(ax_hodo, component_range=80.); h.add_grid(increment=20, color='gray', linestyle='--')
    intervals = np.array([0, 1, 3, 6, 9, 12]) * units.km
    colors_hodo = ['red', 'blue', 'green', 'purple', 'gold']
    h.plot_colormapped(u.to('kt'), v.to('kt'), heights, intervals=intervals, colors=colors_hodo, linewidth=2)
    try:
        rm_vec = params_calc.get('RM')
        if rm_vec and not np.isnan(rm_vec[0]):
            u_rm_kt = (rm_vec[0] * units('m/s')).to('kt').m; v_rm_kt = (rm_vec[1] * units('m/s')).to('kt').m
            ax_hodo.plot(u_rm_kt, v_rm_kt, 'o', color='blue', markersize=8, label='Mov. Dret')
    except Exception: pass
    ax_hodo.set_xlabel('U-Component (nusos)'); ax_hodo.set_ylabel('V-Component (nusos)')
    ax_params.axis('off')
    def degrees_to_cardinal(d):
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ix = int(round(d / 22.5)); return dirs[ix % 16]
    def get_color(value, thresholds):
        if pd.isna(value): return "grey"
        colors = ["grey", "green", "#E69F00", "orange", "red", "#C71585"]; thresholds = sorted(thresholds)
        for i, threshold in enumerate(thresholds):
            if value < threshold: return colors[i]
        return colors[-1]
    THRESHOLDS = {'BWD': (10, 20, 30, 40), 'SRH': (50, 150, 250, 400), 'UPDRAFT': (15, 25, 40, 50)}
    y = 0.98; motion_data = {'MD': params_calc.get('RM'), 'ML': params_calc.get('LM'), 'VM (0-6 km)': params_calc.get('Mean_Wind')}
    ax_params.text(0, y, "Moviment (dir/km/h)", ha='left', weight='bold', fontsize=11); y-=0.1
    for display_name, vec in motion_data.items():
        if vec and not any(pd.isna(v) for v in vec):
            u_motion_ms, v_motion_ms = vec[0] * units('m/s'), vec[1] * units('m/s')
            speed_kmh = mpcalc.wind_speed(u_motion_ms, v_motion_ms).to('km/h').m
            direction_from_deg = mpcalc.wind_direction(u_motion_ms, v_motion_ms, convention='from').to('deg').m
            cardinal_dir = degrees_to_cardinal(direction_from_deg)
            ax_params.text(0, y, f"{display_name}:", ha='left', va='center'); ax_params.text(1, y, f"{direction_from_deg:.0f}° ({cardinal_dir}) / {speed_kmh:.0f}", ha='right', va='center')
        else:
            ax_params.text(0, y, f"{display_name}:", ha='left', va='center'); ax_params.text(1, y, "---", ha='right', va='center')
        y-=0.08
    y-=0.05; ax_params.text(0, y, "Cisallament (nusos)", ha='left', weight='bold', fontsize=11); y-=0.1
    for key, label in [('0-1km', '0-1 km'), ('0-6km', '0-6 km')]:
        val = params_calc.get(f'BWD_{key}', np.nan); color = get_color(val, THRESHOLDS['BWD'])
        ax_params.text(0, y, f"{label}:", ha='left', va='center'); ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color); y-=0.08
    y-=0.05; ax_params.text(0, y, "Helicitat (m²/s²)", ha='left', weight='bold', fontsize=11); y-=0.1
    for key, label in [('0-1km', '0-1 km'), ('0-3km', '0-3 km')]:
        val = params_calc.get(f'SRH_{key}', np.nan); color = get_color(val, THRESHOLDS['SRH'])
        ax_params.text(0, y, f"{label}:", ha='left', va='center'); ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color); y-=0.08
    y-=0.05; ax_params.text(0, y, "Corrent Ascendent", ha='left', weight='bold', fontsize=11); y-=0.1
    val_updraft = params_calc.get('MAX_UPDRAFT', np.nan); color_updraft = get_color(val_updraft, THRESHOLDS['UPDRAFT'])
    ax_params.text(0, y, f"Vel. Max (0-6km):", ha='left', va='center'); ax_params.text(1, y, f"{val_updraft:.1f} m/s" if not pd.isna(val_updraft) else "---", ha='right', va='center', weight='bold', color=color_updraft)
    y-=0.08; return fig

def ui_caixa_parametres_sondeig(params):
    def get_color(value, thresholds, reverse_colors=False):
        if pd.isna(value): return "#808080"
        colors = ["#808080", "#28a745", "#ffc107", "#fd7e14", "#dc3545"]
        if reverse_colors: thresholds = sorted(thresholds, reverse=True); colors = list(reversed(colors))
        else: thresholds = sorted(thresholds)
        for i, threshold in enumerate(thresholds):
            if value < threshold: return colors[i]
        return colors[-1]
    THRESHOLDS = {'SBCAPE': (100, 500, 1500, 2500), 'MUCAPE': (100, 500, 1500, 2500), 'MLCAPE': (50, 250, 1000, 2000), 'CAPE_0-3km': (25, 75, 150, 250), 'DCAPE': (200, 500, 800, 1200), 'SBCIN': (0, -25, -75, -150), 'LI': (0, -2, -5, -8), 'PWAT': (20, 30, 40, 50), 'BWD_0-6km': (10, 20, 30, 40), 'SRH_0-1km': (50, 100, 150, 250)}
    def styled_metric(label, value, unit, param_key, precision=0, reverse_colors=False):
        color = get_color(value, THRESHOLDS.get(param_key, []), reverse_colors)
        val_str = f"{value:.{precision}f}" if not pd.isna(value) else "---"
        st.markdown(f"""<div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;"><span style="font-size: 0.8em; color: #FAFAFA;">{label} ({unit})</span><br><strong style="font-size: 1.6em; color: {color};">{val_str}</strong></div>""", unsafe_allow_html=True)
    st.markdown("##### Paràmetres del Sondeig")
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCAPE", params.get('SBCAPE', np.nan), "J/kg", 'SBCAPE')
    with cols[1]: styled_metric("MUCAPE", params.get('MUCAPE', np.nan), "J/kg", 'MUCAPE')
    with cols[2]: styled_metric("MLCAPE", params.get('MLCAPE', np.nan), "J/kg", 'MLCAPE')
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCIN", params.get('SBCIN', np.nan), "J/kg", 'SBCIN', reverse_colors=True)
    with cols[1]: styled_metric("LI", params.get('LI', np.nan), "°C", 'LI', precision=1, reverse_colors=True)
    with cols[2]: styled_metric("PWAT", params.get('PWAT', np.nan), "mm", 'PWAT', precision=1)
    cols = st.columns(3)
    with cols[0]: styled_metric("LCL", params.get('LCL_Hgt', np.nan), "m", '', precision=0)
    with cols[1]: styled_metric("LFC", params.get('LFC_Hgt', np.nan), "m", '', precision=0)
    with cols[2]: styled_metric("DCAPE", params.get('DCAPE', np.nan), "J/kg", 'DCAPE')
    cols = st.columns(3)
    with cols[0]: styled_metric("BWD 0-6km", params.get('BWD_0-6km', np.nan), "nusos", 'BWD_0-6km')
    with cols[1]: styled_metric("SRH 0-1km", params.get('SRH_0-1km', np.nan), "m²/s²", 'SRH_0-1km')
    with cols[2]: styled_metric("CAPE 0-3km", params.get('CAPE_0-3km', np.nan), "J/kg", 'CAPE_0-3km')

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

def ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str):
    is_guest = st.session_state.get('guest_mode', False)
    st.markdown("#### Mapes de Pronòstic (Model AROME)")
    col_capa, col_zoom = st.columns(2)
    with col_capa:
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", ["Anàlisi de Vent i Convergència", "Vent a 700hPa", "Vent a 300hPa"], key="map_cat")
    with col_zoom: zoom_sel = st.selectbox("Nivell de Zoom:", options=list(MAP_ZOOM_LEVELS_CAT.keys()), key="zoom_cat")
    selected_extent = MAP_ZOOM_LEVELS_CAT[zoom_sel]
    
    if "Convergència" in mapa_sel:
        nivell_sel = 925 if is_guest else st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa", key="level_cat")
        if is_guest: st.info("ℹ️ L'anàlisi de vent i convergència està fixada a **925 hPa**.")
        with st.spinner("Carregant dades del mapa AROME..."): map_data, error_map = carregar_dades_mapa_cat(nivell_sel, hourly_index_sel)
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

def ui_pestanya_vertical(data_tuple, poble_sel, lat, lon):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        p, T, Td, u, v, heights, prof = sounding_data
        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Sondeig Vertical\n{poble_sel}")
            st.pyplot(fig_skewt, use_container_width=True); plt.close(fig_skewt)
            with st.container(border=True): ui_caixa_parametres_sondeig(params_calculats)
        with col2:
            fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat\n{poble_sel}")
            st.pyplot(fig_hodo, use_container_width=True); plt.close(fig_hodo)
            st.markdown("##### Radar de Precipitació en Temps Real")
            radar_url = f"https://www.rainviewer.com/map.html?loc={lat},{lon},8&oCS=1&c=3&o=83&lm=0&layer=radar&sm=1&sn=1&ts=2&play=1"
            html_code = f"""<div style="position: relative; width: 100%; height: 410px; border-radius: 10px; overflow: hidden;"><iframe src="{radar_url}" width="100%" height="410" frameborder="0" style="border:0;"></iframe><div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; cursor: default;"></div></div>"""
            st.components.v1.html(html_code, height=410)
    else: st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def ui_pestanya_mapes_usa(hourly_index_sel, timestamp_str):
    st.markdown("#### Mapes de Pronòstic (Model GFS)")
    # --- LÍNIA MODIFICADA ---
    # S'han afegit els nivells 1000, 975, 950 i 900 hPa a les opcions.
    nivell_sel = st.selectbox(
        "Nivell d'anàlisi:", 
        options=[1000, 975, 950, 925, 900, 850, 700, 500, 300], 
        format_func=lambda x: f"{x} hPa", 
        key="level_usa"
    )
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
    st.image(sat_url, caption="Imatge del satèl·lit GOES-East - CONUS (NOAA STAR)", use_container_width=True)
    st.markdown("[Font: NOAA STAR](https://www.star.nesdis.noaa.gov/GOES/index.php)")
    st.info("Aquesta imatge mostra la vista completa dels Estats Units continentals i s'actualitza cada 5-10 minuts.")

# ... (La resta de funcions es mantenen igual, com ui_pestanya_estacions_meteorologiques, ui_peu_de_pagina, etc.)
SMC_STATION_CODES = {'Amposta': 'D5', 'Balaguer': 'C9', 'Banyoles': 'UB', 'Barcelona': 'X4', 'Berga': 'C8', 'Cervera': 'CE', 'El Pont de Suert': 'C7', 'El Vendrell': 'TT', 'Falset': 'T5', 'Figueres': 'UF', 'Gandesa': 'T9', 'Girona': 'UG', 'Granollers': 'XN', 'Igualada': 'C6', 'La Bisbal d\'Empordà': 'UH', 'La Seu d\'Urgell': 'U7', 'Les Borges Blanques': 'C5', 'Lleida': 'UL', 'Manresa': 'C4', 'Mataró': 'XL', 'Moià': 'WM', 'Mollerussa': 'U4', 'Montblanc': 'T2', 'Móra d\'Ebre': 'T8', 'Olot': 'U6', 'Prats de Lluçanès': 'WP', 'Puigcerdà': 'U8', 'Reus': 'T4', 'Ripoll': 'U5', 'Sant Feliu de Llobregat': 'WZ', 'Santa Coloma de Farners': 'U1', 'Solsona': 'C3', 'Sort': 'U2', 'Tarragona': 'UT', 'Tàrrega': 'U3', 'Terrassa': 'X2', 'Tortosa': 'D4', 'Tremp': 'C2', 'Valls': 'T3', 'Vic': 'W2', 'Vielha': 'VA', 'Vilafranca del Penedès': 'X8', 'Vilanova i la Geltrú': 'XD'}

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
    data_tuple, error_msg = carregar_dades_sondeig_cat(lat_sel, lon_sel, hourly_index_sel)
    if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
    
    if is_guest:
        tab_mapes, tab_vertical, tab_estacions = st.tabs(["Anàlisi de Mapes", "Anàlisi Vertical", "Estacions Meteorològiques"])
        with tab_mapes: ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str)
        with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel)
        with tab_estacions: ui_pestanya_estacions_meteorologiques()
    else:
        # Lògica per a usuaris registrats (amb IA i Xat)
        st.warning("La funcionalitat completa per a usuaris registrats encara s'ha d'implementar en aquest refactor.")

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
    
    data_tuple, error_msg = carregar_dades_sondeig_usa(lat_sel, lon_sel, hourly_index_sel)
    if error_msg: st.error(f"No s'ha pogut carregar el sondeig per a {poble_sel}: {error_msg}")

    tab_mapes, tab_vertical, tab_satelit = st.tabs(["Anàlisi de Mapes", "Anàlisi Vertical", "Satèl·lit (Temps Real)"])
    with tab_mapes: ui_pestanya_mapes_usa(hourly_index_sel, timestamp_str)
    with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel)
    with tab_satelit: ui_pestanya_satelit_usa()

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

if __name__ == "__main__":
    main()
