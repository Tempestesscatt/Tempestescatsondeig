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
# import cartopy.crs as ccrs               # <-- DESACTIVAT
# import cartopy.feature as cfeature       # <-- DESACTIVAT
from scipy.interpolate import griddata
from datetime import datetime, timedelta, timezone
import pytz
from scipy.ndimage import label
import google.generativeai as genai
# import geopandas as gpd                  # <-- DESACTIVAT
# from shapely.geometry import Point       # <-- DESACTIVAT
from collections import Counter
import sqlite3
from streamlit_autorefresh import st_autorefresh
import asyncio
from st_oauth import st_oauth

# --- 0. CONFIGURACIÓ I CONSTANTS ---
st.set_page_config(layout="wide", page_title="Tempestes.cat | Terminal de Temps Sever")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_CONFIGURAT = True
except (KeyError, Exception):
    GEMINI_CONFIGURAT = False

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
FORECAST_DAYS = 4
API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = pytz.timezone('Europe/Madrid')
CIUTATS_CATALUNYA = {
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734}, 'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200}, 'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
DB_FILE = "users.db"

# --- FUNCIONS DE MAPES DESACTIVADES ---
# @st.cache_data(ttl=86400)
# def carregar_mapa_provincies():
#     url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/spain-provinces.geojson"
#     gdf = gpd.read_file(url)
#     return gdf[gdf['name'].isin(['Barcelona', 'Tarragona', 'Lleida', 'Girona'])]
# PROVINCIES_GDF = carregar_mapa_provincies()

# --- GESTIÓ DE LA BASE DE DADES (Per al xat) ---
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # No necessitem la taula 'users' amb el login de Google
    c.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()

def add_message_to_db(username, message):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (username, message) VALUES (?, ?)", (username, message))
    conn.commit()
    conn.close()

def get_messages_from_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE timestamp < datetime('now', '-1 hour')")
    conn.commit()
    c.execute("SELECT username, message, timestamp FROM messages ORDER BY timestamp DESC")
    messages = c.fetchall()
    conn.close()
    return messages

# --- FUNCIONS D'OBTENCIÓ DE DADES (Sondeig, etc. - Sense canvis) ---
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
        except: params_calc['SRH 0-3km'] = np.nan
        return ((p, T, Td, u, v), params_calc), None
    except Exception as e: return None, f"Error en processar dades del sondeig: {e}"

# --- FUNCIONS DE VISUALITZACIÓ (Gràfics - Sense canvis) ---
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
        if 7 <= now_local.hour < 21: url = "https://www.meteociel.fr/modeles/satanim_espagne-ne.gif"; caption = "Satèl·lit Visible (Nord-est). Font: Meteociel"
        else: url = "https://www.meteociel.fr/modeles/satanim_ir_espagne-ne.gif"; caption = "Satèl·lit Infraroig (Nord-est). Font: Meteociel"
    else: st.error("Tipus d'imatge no reconegut."); return
    try:
        response = requests.get(f"{url}?ver={int(time.time())}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code == 200: st.image(response.content, caption=caption, use_container_width=True)
        else: st.warning(f"No s'ha pogut carregar la imatge. (Codi: {response.status_code})")
    except Exception as e: st.error(f"Error de xarxa en carregar la imatge.")

# --- FUNCIONS D'IA (Sense canvis) ---
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
    # ... (la resta de la funció no canvia)
    return "#FFFFFF"

def preparar_resum_dades_per_ia(data_tuple, poble_sel, timestamp_str):
    resum_sondeig = "No hi ha dades de sondeig."
    if data_tuple:
        _, params_calculats = data_tuple
        resum_sondeig = "\n".join([f"- {key}: {value:.0f}" for key, value in params_calculats.items() if value is not None and not np.isnan(value)])
    return f"DADES:\nSondeig per a {poble_sel}:\n{resum_sondeig}\nINSTRUCCIONS:\nRespon a l'usuari."

def generar_resposta_ia_stream(historial_conversa, resum_dades, prompt_usuari):
    if not GEMINI_CONFIGURAT:
        yield "La funcionalitat d'IA no està configurada."
        return
    model = genai.GenerativeModel('gemini-1.5-flash')
    # ... (la resta de la funció no canvia)
    try:
        response = model.generate_content(resum_dades + f"\n\nPREGUNTA ACTUAL DE L'USUARI:\n'{prompt_usuari}'", stream=True)
        for chunk in response:
            yield chunk.text
    except Exception as e:
        yield f"Hi ha hagut un error contactant amb l'IA de Google: {e}"

# --- LÒGICA DE LA INTERFÍCIE D'USUARI ---
def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Eina per al pronòstic de convecció mitjançant paràmetres clau.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.selectbox("Capital de referència:", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3: st.selectbox("Hora del pronòstic (Hora Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

# --- PESTANYA DE MAPES DESACTIVADA ---
# def ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple):
#     st.warning("La funcionalitat de mapes està temporalment desactivada.")
#     # ... (Tot el codi de la funció està comentat)

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
                html_code = f"""<div style="text-align: left;"><span style="font-size: 0.8em; color: #A0A0A0;">{param}</span><br><strong style="font-size: 1.8em; color: {color};">{value_str}</strong> <span style="font-size: 1.1em; color: #A0A0A0;">{unit}</span></div>"""
                st.markdown(html_code, unsafe_allow_html=True)
        with st.expander("Què signifiquen aquests paràmetres?"):
            explanation_lines = ["- **CAPE:** Energia per a tempestes. >1000 J/kg és significatiu.", "- **CIN:** \"Tapa\" que impedeix la convecció. > -50 és una tapa forta.", "- **LFC:** Nivell on comença la convecció lliure. Com més baix, millor.", "- **Shear 0-1km:** Cisallament a nivells baixos. >15-20 nusos afavoreix la rotació i el risc de **tornados**.", "- **Shear 0-6km:** Cisallament profund. >35-40 nusos és clau per a **supercèl·lules**."]
            st.markdown("\n".join(explanation_lines))
        st.divider()
        col1, col2 = st.columns(2)
        with col1: st.pyplot(crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], f"Sondeig Vertical - {poble_sel}"))
        with col2: st.pyplot(crear_hodograf(sounding_data[3], sounding_data[4]))
    else: st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def ui_pestanya_ia(data_tuple, hourly_index_sel, poble_sel, timestamp_str):
    st.subheader("Assistent MeteoIA (amb Google Gemini)")
    st.markdown("Fes-me preguntes sobre el potencial de temps sever a partir de les dades del sondeig.")
    if not GEMINI_CONFIGURAT:
        st.error("Funcionalitat no disponible.")
        return
    if not data_tuple:
        st.warning("No hi ha dades de sondeig disponibles per analitzar.")
        return
    resum_dades = preparar_resum_dades_per_ia(data_tuple, poble_sel, timestamp_str)
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Escriu la teva pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            historial_anterior = st.session_state.messages[:-1]
            response_generator = generar_resposta_ia_stream(historial_anterior, resum_dades, prompt)
            full_response = st.write_stream(response_generator)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

def ui_pestanya_chat():
    st.subheader("Xat de la Comunitat (Missatges de l'última hora)")
    st_autorefresh(interval=15 * 1000, key="chat_refresher")
    messages = get_messages_from_db()
    chat_container = st.container(height=500, border=True)
    with chat_container:
        for username, message, timestamp in messages:
            ts_local = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).astimezone(TIMEZONE)
            with st.chat_message(name=username):
                st.markdown(f"**{username}** <span style='font-size: 0.8em; color: grey;'>- {ts_local.strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)
                st.markdown(message)
    if prompt := st.chat_input("Escriu el teu missatge..."):
        add_message_to_db(st.session_state["name"], prompt)
        st.rerun()

def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meto</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- APLICACIÓ PRINCIPAL ---
def app_principal():
    st.markdown(f"Benvingut, **{st.session_state['name']}**!")
    ui_capcalera_selectors()
    current_selection = f"{st.session_state.poble_selector}-{st.session_state.dia_selector}-{st.session_state.hora_selector}"
    if current_selection != st.session_state.get('last_selection', ''):
        st.session_state.messages = []
        st.session_state.last_selection = current_selection
    poble_sel, dia_sel, hora_sel = st.session_state.poble_selector, st.session_state.dia_selector, st.session_state.hora_selector
    hora_int = int(hora_sel.split(':')[0]); now_local = datetime.now(TIMEZONE); target_date = now_local.date()
    if dia_sel == "Demà": target_date += timedelta(days=1)
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int)); utc_dt = local_dt.astimezone(pytz.utc)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600); hourly_index_sel = max(0, time_diff_hours)
    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
    lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']
    data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
    st.markdown("---")
    global progress_placeholder
    progress_placeholder = st.empty()
    
    # MODIFICACIÓ: Eliminem la pestanya de mapes
    tab_ia, tab_vertical, tab_chat = st.tabs(["Assistent MeteoIA", "Anàlisi Vertical", "Xat de la Comunitat"])
    
    with tab_ia:
        ui_pestanya_ia(data_tuple, hourly_index_sel, poble_sel, timestamp_str)
    
    # MODIFICACIÓ: Aquest bloc està desactivat
    # with tab_mapes:
    #     ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
        
    with tab_vertical:
        ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_chat:
        ui_pestanya_chat()
    ui_peu_de_pagina()

async def main():
    # 1. Configuració de les credencials de Google des de st.secrets
    client_id = st.secrets["GOOGLE_CLIENT_ID"]
    client_secret = st.secrets["GOOGLE_CLIENT_SECRET"]
    
    # IMPORTANT: Assegura't que aquest URI coincideix amb el que has configurat a Google Cloud
    redirect_uri = "https://tempestescat.streamlit.app" 
    
    # 2. Creació del botó d'inici de sessió amb Google
    user_info = await st_oauth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        login_button_text="Inicia sessió amb Google",
        login_button_icon="fa fa-google",
        logout_button_text="Tanca la sessió",
    )
    
    # 3. Lògica de l'aplicació
    if user_info:
        st.session_state['name'] = user_info.get('name', 'Usuari desconegut')
        st.session_state['authentication_status'] = True
        app_principal()
    else:
        st.session_state['authentication_status'] = None
        st.warning("Si us plau, inicia sessió amb el teu compte de Google per continuar.")

if __name__ == "__main__":
    asyncio.run(main())
