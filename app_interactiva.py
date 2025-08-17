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
from scipy.interpolate import griddata, Rbf
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai

# --- 0. CONFIGURACIÓ I CONSTANTS ---
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_CONFIGURAT = True
except:
    GEMINI_CONFIGURAT = False

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

FORECAST_DAYS = 4
API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = pytz.timezone('Europe/Madrid')

CIUTATS_CATALUNYA = {
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734, 'emoji': '🏙️'},
    'Girona': {'lat': 41.9831, 'lon': 2.8249, 'emoji': '🏰'},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200, 'emoji': '🌾'},
    'Tarragona': {'lat': 41.1189, 'lon': 1.2445, 'emoji': '🏛️'},
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

# --- 1. FUNCIONS ORIGINALS (SENSE MODIFICAR) ---
@st.cache_data(ttl=3600)
def carregar_dades_sondeig(lat, lon, hourly_index):
    # ... (Mantenir exactament igual que al teu codi original)
    pass

@st.cache_data(ttl=3600)
def carregar_dades_mapa(variables, hourly_index):
    # ... (Mantenir igual)
    pass

@st.cache_data(ttl=3600)
def preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel):
    # ... (Mantenir igual)
    pass

@st.cache_data(ttl=3600)
def generar_resum_ia(_dades_ia, _poble_sel, _timestamp_str):
    # ... (Mantenir igual)
    pass

def crear_mapa_base():
    # ... (Mantenir igual)
    pass

def get_wind_colormap():
    # ... (Mantenir igual)
    pass

def crear_mapa_500hpa(map_data, timestamp_str):
    # ... (Mantenir igual)
    pass

def crear_mapa_vents_velocitat(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    # ... (Mantenir igual)
    pass

def crear_mapa_convergencia(lons, lats, speed_data, dir_data, nivell, lat_sel, lon_sel, nom_poble_sel, timestamp_str):
    # ... (Mantenir igual)
    pass

def crear_mapa_escalar(lons, lats, data, titol, cmap, levels, unitat, timestamp_str, extend='max'):
    # ... (Mantenir igual)
    pass

def crear_skewt(p, T, Td, u, v, titol):
    # ... (Mantenir igual)
    pass

def crear_hodograf(u, v):
    # ... (Mantenir igual)
    pass

def mostrar_imatge_temps_real(tipus):
    # ... (Mantenir igual)
    pass

# --- 2. NOVES FUNCIONS PER A LES TEVES PETICIONS ---
def generar_avis_visual(cape, shear, humitat):
    """Genera avisos visuals amb emojis i colors"""
    if cape > 2000 and shear > 15:
        return f"""
        <div style='background:#ffebee; border-left:6px solid #f44336; padding:1em; margin:1em 0; border-radius:0 8px 8px 0'>
            <h3 style='color:#d32f2f; margin:0'>🌩️ <b>PERILL!</b> Tempestes fortes</h3>
            <p style='margin:0.5em 0'>• Possible calamarsa<br>• Risc d'inundacions</p>
        </div>
        """
    elif cape > 1000:
        return f"""
        <div style='background:#fff8e1; border-left:6px solid #ffc107; padding:1em; margin:1em 0; border-radius:0 8px 8px 0'>
            <h3 style='color:#ff8f00; margin:0'>🌧️ Avis moderat</h3>
            <p style='margin:0'>Xàfecs aïllats possibles</p>
        </div>
        """
    else:
        return f"""
        <div style='background:#e8f5e9; border-left:6px solid #4caf50; padding:1em; margin:1em 0; border-radius:0 8px 8px 0'>
            <h3 style='color:#2e7d32; margin:0'>🌤️ Temps estable</h3>
            <p style='margin:0'>Sense alertes significatives</p>
        </div>
        """

def ui_preguntes_meteo(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    """Nova pestanya per fer preguntes a la IA"""
    st.subheader("❓ Fes preguntes sobre el temps")
    
    pregunta = st.text_input("Escriu la teva pregunta (ex: 'Quin temps farà demà a la tarda?')", 
                           key="pregunta_meteo")
    
    if pregunta and GEMINI_CONFIGURAT:
        with st.spinner("Analitzant..."):
            dades_ia, _ = preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel)
            if dades_ia:
                resposta = generar_resum_ia(dades_ia, poble_sel, timestamp_str)
                st.markdown(f"""
                <div style='background:#f5f5f5; border-radius:10px; padding:1em; margin-top:1em'>
                    <p style='font-size:1.1em; margin:0'>{resposta}</p>
                </div>
                """, unsafe_allow_html=True)
    elif pregunta:
        st.warning("⚠️ Configura la clau API de Google Gemini a secrets.toml")

# --- 3. INTERFÍCIE MODIFICADA ---
def ui_capcalera_selectors():
    st.markdown("""
    <style>
    .header {
        background: linear-gradient(45deg, #0061ff, #60efff);
        padding: 1.5em;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
    <div class="header">
        <h1 style="margin:0">🌦️ Terminal Meteorològic de Catalunya</h1>
        <p style="margin:0">Avisos senzills i explicacions clares</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: 
            poble = st.selectbox("Selecciona una ciutat:", sorted(CIUTATS_CATALUNYA.keys()))
        with col2: 
            dia = st.selectbox("Dia:", ["Avui", "Demà"])
        with col3: 
            hora = st.selectbox("Hora:", [f"{h:02d}:00h" for h in range(24)])
    return poble, dia, hora

def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    # ... (Mantenir la funció original però afegir al principi:)
    st.markdown(generar_avis_visual(1500, 12, 65), unsafe_allow_html=True)  # Valors simulats
    # ... (Resta del codi original)

def main():
    poble_sel, dia_sel, hora_sel = ui_capcalera_selectors()
    
    # ... (Mantenir el processament original de dades)
    
    # Modificar les pestanyes per afegir la nova funcionalitat
    tab_mapes, tab_vertical, tab_ia, tab_preguntes = st.tabs(
        ["🗺️ Mapes", "📊 Dades", "📢 Avisos", "❓ Preguntes"]
    )
    
    with tab_mapes:
        ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    with tab_vertical:
        ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia:
        st.markdown("## 📢 Avisos visuals per a avui")
        st.markdown(generar_avis_visual(1800, 14, 70), unsafe_allow_html=True)  # Exemple
    with tab_preguntes:
        ui_preguntes_meteo(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
