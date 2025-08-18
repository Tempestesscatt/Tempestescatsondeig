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
from matplotlib.path import Path
import matplotlib.patches as patches
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata, Rbf
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai
from threading import Lock

# --- 0. CONFIGURACIÓ I CONSTANTS ---

st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")

METPY_LOCK = Lock()

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
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200},
    'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)


# --- 1. FUNCIONS D'OBTENCIÓ I PROCESSAMENT DE DADES ---

def vector_to_direction(u, v):
    if np.isnan(u) or np.isnan(v): return "N/A", np.nan
    speed_ms = np.sqrt(u**2 + v**2)
    direction_deg = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    cardinal_dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return cardinal_dirs[int(round(direction_deg / 22.5))], speed_ms

@st.cache_data(ttl=3600)
def carregar_dades_sondeig(lat, lon, hourly_index):
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
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
        sfc_h = mpcalc.pressure_to_height_std(sfc_data["surface_pressure"] * units.hPa).to('meter').m
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0], [sfc_h]
        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m)
                v_profile.append(v.to('m/s').m)
                h_profile.append(p_data["H"][i])
        if len(p_profile) < 5: return None, "Perfil atmosfèric massa curt per a càlculs fiables."
        p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC; Td = np.array(Td_profile) * units.degC
        u = np.array(u_profile) * units('m/s'); v = np.array(v_profile) * units('m/s'); h = np.array(h_profile) * units.meter
        params_calc = {}
        with METPY_LOCK:
            prof = mpcalc.parcel_profile(p, T[0], Td[0])
            cape, cin = mpcalc.cape_cin(p, T, Td, prof)
            params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0
            params_calc['CIN'] = cin.to('J/kg').m
            try:
                lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); params_calc['LCL_p'] = lcl_p.to('hPa').m
                lfc_p, _ = mpcalc.lfc(p, T, Td, which='most_cape'); params_calc['LFC_p'] = lfc_p.to('hPa').m
                el_p, _ = mpcalc.el(p, T, Td); params_calc['EL_p'] = el_p.to('hPa').m
            except Exception:
                 params_calc.update({'LCL_p': np.nan, 'LFC_p': np.nan, 'EL_p': np.nan})
            s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6 * units.km)
            params_calc['Shear_0-6km'] = mpcalc.wind_speed(s_u, s_v).to('m/s').m
            try:
                h_agl = h - h[0]
                mid_level_mask = (h_agl >= 3000 * units.meter) & (h_agl <= 6000 * units.meter)
                if np.any(mid_level_mask):
                    mean_u_mid = np.mean(u[mid_level_mask]); mean_v_mid = np.mean(v[mid_level_mask])
                    params_calc['storm_motion_u'] = mean_u_mid.to('m/s').m; params_calc['storm_motion_v'] = mean_v_mid.to('m/s').m
                    storm_motion_vector = (mean_u_mid, mean_v_mid)
                else: raise ValueError("No data in 3-6km layer")
            except Exception:
                params_calc.update({'storm_motion_u': np.nan, 'storm_motion_v': np.nan}); storm_motion_vector = (np.nan*units('m/s'), np.nan*units('m/s'))
            storm_u_comp, storm_v_comp = storm_motion_vector
            _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=3 * units.km, storm_u=storm_u_comp, storm_v=storm_v_comp)
            params_calc['SRH_0-3km'] = np.sum(srh).to('m^2/s^2').m
        return ((p, T, Td, u, v, h), params_calc), None
    except Exception as e:
        return None, f"Error crític en processar dades del sondeig: {e}"

# --- [Les funcions de dades dels mapes i IA es mantenen igual] ---

# --- 2. FUNCIONS DE VISUALITZACIÓ ---

# *** NOVA FUNCIÓ PER DIBUIXAR EL PERFIL DE TEMPESTA DINS DEL SONDEIG ***
def dibuixar_perfil_inset(fig, sounding_data, params_calc):
    p, _, _, u, v, h = sounding_data
    
    # Defineix la posició i mida de l'inset [esquerra, baix, ample, alt]
    inset_ax = fig.add_axes([0.65, 0.55, 0.25, 0.35])
    
    if 'LFC_p' not in params_calc or 'EL_p' not in params_calc or np.isnan(params_calc['LFC_p']) or np.isnan(params_calc['EL_p']):
        inset_ax.text(0.5, 0.5, "Sense\nConvecció", ha='center', va='center', fontsize=12, wrap=True)
        inset_ax.set_xticks([]); inset_ax.set_yticks([])
        return
    
    h_lfc_km = np.interp(params_calc['LFC_p'], p.magnitude[::-1], h.magnitude[::-1]) / 1000
    h_el_km = np.interp(params_calc['EL_p'], p.magnitude[::-1], h.magnitude[::-1]) / 1000
    
    inset_ax.axhline(0, color='darkgreen', linewidth=4, zorder=1)
    inset_ax.text(0.5, -0.5, "Base Plana", ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
    inset_ax.set_facecolor("#a0d2f0") # Blau cel

    verts = [ (0.35, h_lfc_km), (0.65, h_lfc_km), (0.85, (h_lfc_km + h_el_km) / 2), (0.75, h_el_km),
              (0.25, h_el_km), (0.15, (h_lfc_km + h_el_km) / 2), (0.35, h_lfc_km), ]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.LINETO, Path.LINETO, Path.CURVE3, Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='white', lw=1.5, edgecolor='black', zorder=2)
    inset_ax.add_patch(patch)
    
    max_h_plot = int(h_el_km) + 2
    barb_levels_km = np.arange(0, max_h_plot, 1)
    valid_barbs = barb_levels_km * 1000 <= h.magnitude.max()
    barb_levels_km = barb_levels_km[valid_barbs]
    
    if len(barb_levels_km) > 0:
        barb_p = np.interp(barb_levels_km * 1000, h.magnitude, p.magnitude)
        barb_u = np.interp(barb_p, p.magnitude[::-1], u.to('kt').magnitude[::-1])
        barb_v = np.interp(barb_p, p.magnitude[::-1], v.to('kt').magnitude[::-1])
        inset_ax.barbs(np.full_like(barb_levels_km, 1.05), barb_levels_km, barb_u, barb_v, length=8, zorder=3)

    inset_ax.set_xlim(0, 1.3)
    inset_ax.set_ylim(0, max_h_plot)
    inset_ax.set_xticks([])
    inset_ax.set_yticks(np.arange(0, max_h_plot, 5))
    inset_ax.tick_params(axis='y', labelsize=10)

def crear_skewt(p, T, Td, u, v, titol, params_calc, sounding_data):
    fig = plt.figure(figsize=(9, 9), dpi=150) # Mida gran per a més detall
    skew = SkewT(fig, rotation=45)
    
    skew.ax.grid(True, linestyle='-', alpha=0.5)
    skew.plot(p, T, 'r', lw=2, label='Temperatura')
    skew.plot(p, Td, 'g', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03, length=6)
    skew.plot_dry_adiabats(color='brown', linestyle='--', alpha=0.5)
    skew.plot_moist_adiabats(color='blue', linestyle='--', alpha=0.5)
    skew.plot_mixing_lines(color='green', linestyle='--', alpha=0.5)

    with METPY_LOCK:
        prof = mpcalc.parcel_profile(p, T[0], Td[0])

    skew.plot(p, prof, 'k', linewidth=2, label='Trajectòria Parcel·la')
    skew.shade_cape(p, T, prof, color='red', alpha=0.3)
    skew.shade_cin(p, T, prof, color='blue', alpha=0.3)

    # Dibuixa les línies de LCL, LFC, EL sense text
    for level_name, color in [('LCL_p', 'orange'), ('LFC_p', 'darkred'), ('EL_p', 'purple')]:
        if level_name in params_calc and not np.isnan(params_calc[level_name]):
            p_level = params_calc[level_name]
            skew.ax.axhline(p_level, color=color, linestyle='--', lw=2, zorder=3)

    skew.ax.set_ylim(1000, 150)
    skew.ax.set_xlim(-30, 40)
    skew.ax.set_title(titol, weight='bold', fontsize=14)
    skew.ax.set_xlabel("Temperatura (°C)")
    skew.ax.set_ylabel("Pressió (hPa)")
    skew.ax.legend()
    
    # *** CRIDA A LA NOVA FUNCIÓ PER DIBUIXAR L'INSET ***
    dibuixar_perfil_inset(fig, sounding_data, params_calc)
    
    return fig

def crear_hodograf(u, v, params_calc):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    h = Hodograph(ax, component_range=80.)
    h.add_grid(increment=20, color='gray', linestyle='--')
    h.plot(u.to('kt'), v.to('kt'), color='red', linewidth=2)
    if 'storm_motion_u' in params_calc and not np.isnan(params_calc['storm_motion_u']):
        sm_u_kt = params_calc['storm_motion_u'] * (units('m/s')).to('kt').m
        sm_v_kt = params_calc['storm_motion_v'] * (units('m/s')).to('kt').m
        ax.arrow(0, 0, sm_u_kt, sm_v_kt, head_width=2, head_length=2, fc='darkred', ec='black', zorder=10, lw=1.5, label='Mov. Tempesta (Mitjana 3-6km)')
        direction_str, speed_ms = vector_to_direction(params_calc['storm_motion_u'], params_calc['storm_motion_v'])
        ax.text(0.03, 0.97, f"Mov. Tempesta:\n {direction_str} a {speed_ms*3.6:.0f} km/h", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_title("Hodògraf i Moviment de Tempesta", weight='bold')
    ax.set_xlabel("Component U (kt)"); ax.set_ylabel("Component V (kt)")
    ax.legend()
    return fig

# --- [La resta de funcions de visualització de mapes es mantenen igual] ---

# --- 3. LÒGICA DE LA INTERFÍCIE D'USUARI (UI) ---

def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🌪️ Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Eina per a la visualització de paràmetres clau per al pronòstic de convecció, basada en el model <b>AROME</b>.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.selectbox("Capital de referència:", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3: st.selectbox("Hora del pronòstic (Hora Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

# --- UI DE LA PESTANYA VERTICAL MODIFICADA ---
def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    with st.container(border=True):
        if data_tuple:
            sounding_data, params = data_tuple
            st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")

            # ... (la secció de mètriques es manté igual) ...
            col1, col2, col3 = st.columns(3)
            with col1:
                val = params.get('CAPE'); st.metric(label="CAPE", value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} J/kg")
            with col2:
                val = params.get('CIN'); st.metric(label="CIN", value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} J/kg")
            with col3:
                val = params.get('Shear_0-6km'); st.metric(label="Shear 0-6km", value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} m/s")
            col4, col5, col6 = st.columns(3)
            with col4:
                val = params.get('SRH_0-3km'); st.metric(label="SRH 0-3km", value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} m²/s²")
            with col5:
                val = params.get('LFC_p'); st.metric(label="LFC (Nivell Conv. Lliure)", value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} hPa")
            with col6:
                val = params.get('EL_p'); st.metric(label="EL (Nivell Equilibri)", value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} hPa")

            with st.expander("ℹ️ Què signifiquen aquests paràmetres?"):
                st.markdown(""" ... """) # (Contingut es manté)
            st.divider()

            # --- NOVA DISPOSICIÓ DELS GRÀFICS ---
            col_graf_1, col_graf_2 = st.columns([2, 1])
            with col_graf_1:
                p, T, Td, u, v, h = sounding_data
                st.pyplot(crear_skewt(p, T, Td, u, v, f"Sondeig Vertical - {poble_sel}", params, sounding_data), use_container_width=True)
            with col_graf_2:
                _, _, _, u, v, _ = sounding_data
                st.pyplot(crear_hodograf(u, v, params), use_container_width=True)
        else:
            st.warning("No hi ha dades de sondeig disponibles per a la selecció actual. Pot ser degut a dades invàlides del model.")

# --- [La resta de la UI i la funció main es mantenen pràcticament igual] ---
def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    with st.container(border=True):
        col_map_1, col_map_2 = st.columns([2.5, 1.5])
        with col_map_1:
            with st.spinner("Actualitzant anàlisi de mapes..."):
                map_options = { "CAPE (Energia)": "cape", "Flux i Convergència (Disparador)": "conv", "Anàlisi a 500hPa": "500hpa",
                                "Vent a 300hPa (Jet Stream)": "wind_300", "Vent a 700hPa": "wind_700", "Humitat a 700hPa": "rh_700" }
                mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
                map_key = map_options[mapa_sel]
                if map_key == "cape":
                    map_data, error = carregar_dades_mapa(["cape"], hourly_index_sel)
                    if map_data:
                        max_cape = np.max(map_data['cape']) if map_data['cape'] else 0
                        cape_levels = np.arange(100, max(1001, np.ceil(max_cape/250+1)*250), 250)
                        st.pyplot(crear_mapa_escalar(map_data, "cape", "CAPE", "plasma", cape_levels, "J/kg", timestamp_str), use_container_width=True)
                    elif error: st.error(f"Error en carregar el mapa: {error}")
                elif map_key == "conv":
                    nivell_sel = st.selectbox("Nivell d'anàlisi:", options=[950, 925, 850], format_func=lambda x: f"{x} hPa")
                    variables = [f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                    map_data, error = carregar_dades_mapa(variables, hourly_index_sel)
                    if map_data: st.pyplot(crear_mapa_convergencia(map_data, nivell_sel, lat_sel, lon_sel, poble_sel, timestamp_str), use_container_width=True)
                    elif error: st.error(f"Error en carregar el mapa: {error}")
                elif map_key == "500hpa":
                    variables = ["temperature_500hPa", "wind_speed_500hPa", "wind_direction_500hPa"]
                    map_data, error = carregar_dades_mapa(variables, hourly_index_sel)
                    if map_data: st.pyplot(crear_mapa_500hpa(map_data, timestamp_str), use_container_width=True)
                    elif error: st.error(f"Error en carregar el mapa: {error}")
                elif map_key in ["wind_300", "wind_700"]:
                    nivell_hpa = int(map_key.split('_')[1])
                    variables = [f"wind_speed_{nivell_hpa}hPa", f"wind_direction_{nivell_hpa}hPa"]
                    map_data, error = carregar_dades_mapa(variables, hourly_index_sel)
                    if map_data: st.pyplot(crear_mapa_vents_velocitat(map_data, nivell_hpa, timestamp_str), use_container_width=True)
                    elif error: st.error(f"Error en carregar el mapa: {error}")
                elif map_key == "rh_700":
                    map_data, error = carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
                    if map_data: st.pyplot(crear_mapa_escalar(map_data, "relative_humidity_700hPa", "Humitat Relativa a 700hPa", "Greens", np.arange(50, 101, 10), "%", timestamp_str), use_container_width=True)
                    elif error: st.error(f"Error en carregar el mapa: {error}")
        with col_map_2:
            st.subheader("Imatges en Temps Real")
            view_choice = st.radio("Selecciona la vista:", ("Satèl·lit", "Radar"), horizontal=True, label_visibility="collapsed")
            mostrar_imatge_temps_real(view_choice)

def ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    with st.container(border=True):
        st.subheader(f"Assistent d'Anàlisi per IA")
        st.info("Aquest assistent utilitza Google Gemini per interpretar les dades meteorològiques...", icon="🤖")
        if not GEMINI_CONFIGURAT: st.error("Funcionalitat no disponible. La clau API de Google no està configurada...")
        elif st.button("Generar Anàlisi d'IA", use_container_width=True, type="primary"):
            with st.spinner("L'assistent d'IA està analitzant les dades..."):
                dades_ia, error = preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel)
                if error: st.error(f"No s'ha pogut generar l'anàlisi: {error}")
                else: st.markdown(generar_resum_ia(dades_ia, poble_sel, timestamp_str))

def ui_peu_de_pagina():
    st.divider()
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

def main():
    if 'hora_selector' not in st.session_state:
        st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"
    ui_capcalera_selectors()
    poble_sel = st.session_state.poble_selector; dia_sel = st.session_state.dia_selector; hora_sel = st.session_state.hora_selector
    hora_int = int(hora_sel.split(':')[0])
    now_local = datetime.now(TIMEZONE)
    target_date = now_local.date() + timedelta(days=(1 if dia_sel == "Demà" else 0))
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int))
    start_of_forecast_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = max(0, int((local_dt.astimezone(pytz.utc) - start_of_forecast_utc).total_seconds() / 3600))
    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
    lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']
    with st.spinner('Carregant dades del sondeig inicial...'):
        data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
        if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Resum IA"])
    with tab_mapes: ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia: ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
