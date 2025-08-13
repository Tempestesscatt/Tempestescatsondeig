# -*- coding: utf-8 -*-
import streamlit as st
import openmeteo_requests
from retry_requests import retry
import requests
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.transforms as mtransforms
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc
from matplotlib.patches import Circle, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import pytz
import matplotlib.patheffects as path_effects
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# --- CONFIGURACIÓ INICIAL ---
st.set_page_config(layout="wide", page_title="Tempestes.cat")

plain_session = requests.Session()
retry_session = retry(plain_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

FORECAST_DAYS = 1
CONVERGENCIA_FORTA_THRESHOLD = -25

# --- DADES DE LOCALITATS ---
# AQUEST BLOC S'HA OBLIDAT PER BREVETAT, ENGANXA AQUÍ LA TEVA LLISTA COMPLETA
pobles_data = {
    'Barcelona': {'lat': 41.387, 'lon': 2.168}, 'Lleida': {'lat': 41.617, 'lon': 0.622},
    'Tarragona': {'lat': 41.118, 'lon': 1.245}, 'Girona': {'lat': 41.983, 'lon': 2.824},
    # ... (etc., la teva llista completa de 200+ localitats va aquí)
}

# --- FUNCIÓ DE CALLBACK I SESSION STATE ---
def actualitzar_seleccio(poble, hora):
    st.session_state.poble_selector = poble
    st.session_state.hora_selector = f"{hora:02d}:00h"
    st.session_state.avisos_expanded = False

if 'poble_selector' not in st.session_state: st.session_state.poble_selector = 'Barcelona'
if 'hora_selector' not in st.session_state: st.session_state.hora_selector = f"{datetime.now(pytz.timezone('Europe/Madrid')).hour:02d}:00h"
if 'nivell_mapa' not in st.session_state: st.session_state.nivell_mapa = 850
if 'avisos_expanded' not in st.session_state: st.session_state.avisos_expanded = True


# --- 1. LÒGICA DE CÀRREGA DE DADES I CÀLCUL ---

def parsear_sondeig_manual(text_data):
    p, T, Td, h, wspd, wdir = [], [], [], [], [], []
    lines = text_data.strip().split('\n')
    data_started = False
    
    for line in lines:
        line = line.strip()
        if not line or "Run du" in line or "locale" in line or "Iso 0°C" in line: continue
        if "Altitude" in line and "Pression" in line: data_started = True; continue
        if not data_started: continue
        parts = re.split(r'\s{2,}', line)
        if len(parts) < 6: continue
        try:
            h.append(float(parts[0].replace('m', '').replace('(Sol)', '').strip()))
            p.append(float(parts[1].replace('hPa', '').strip()))
            T.append(float(parts[2].replace('°C', '').strip()))
            Td.append(float(parts[4].replace('°C', '').strip()))
            vent_parts = parts[6].split('/')
            wdir.append(float(vent_parts[0].replace('°', '').strip()))
            wspd.append(float(vent_parts[1].replace('kt', '').strip()))
        except (ValueError, IndexError): continue
            
    if not p: return None, "No s'han pogut extreure dades vàlides. Comprova el format."
    p.reverse(); T.reverse(); Td.reverse(); h.reverse(); wspd.reverse(); wdir.reverse()
    p_units, T_units, Td_units = np.array(p)*units.hPa, np.array(T)*units.degC, np.array(Td)*units.degC
    h_units = np.array(h)*units.meter; wspd_units, wdir_units = np.array(wspd)*units.knots, np.array(wdir)*units.degrees
    u_units, v_units = mpcalc.wind_components(wspd_units, wdir_units)
    return (p_units, T_units, Td_units, u_units, v_units, h_units), None

@st.cache_data(ttl=3600)
def carregar_sondeig_per_poble(nom_poble, lat, lon):
    p_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"] for p in p_levels]
    params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "timezone": "auto", "forecast_days": FORECAST_DAYS}
    try:
        respostes = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        return respostes[0], p_levels, None
    except Exception as e: return None, None, str(e)

def obtener_dades_mapa(variable, nivell, hourly_index, forecast_days):
    lats, lons = np.linspace(40.5, 42.8, 12), np.linspace(0.2, 3.3, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats); api_vars = []
    if variable == 'temp_height': api_vars = [f"temperature_{nivell}hPa", f"geopotential_height_{nivell}hPa"]
    elif variable == 'wind': api_vars = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
    elif variable == 'dewpoint': api_vars = [f"dew_point_{nivell}hPa"]
    elif variable == 'humidity': api_vars = [f"relative_humidity_{nivell}hPa"]
    elif variable == 'temperature': api_vars = [f"temperature_{nivell}hPa"]
    else: return None, None, None, f"Variable '{variable}' no reconeguda."
    params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": api_vars, "models": "arome_seamless", "timezone": "auto", "forecast_days": forecast_days}
    try:
        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params); lats_out, lons_out, data_out = [], [], []
        for r in responses:
            hourly = r.Hourly(); values = [hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(api_vars))]
            if not any(np.isnan(v) for v in values): lats_out.append(r.Latitude()); lons_out.append(r.Longitude()); data_out.append(tuple(values) if len(values) > 1 else values[0])
        if not lats_out: return None, None, None, "No s'han rebut dades vàlides del model."
        return lats_out, lons_out, data_out, None
    except Exception as e: return None, None, None, str(e)

@st.cache_data(ttl=18000)
def calcular_convergencia_per_totes_les_localitats(_hourly_index, _nivell, _localitats_dict):
    lats_mapa, lons_mapa, data_mapa, error = obtener_dades_mapa('wind', _nivell, _hourly_index, FORECAST_DAYS)
    if error or not lats_mapa or len(lats_mapa) < 4: return {}
    speeds, dirs = zip(*data_mapa); speeds_ms = (np.array(speeds)*1000/3600)*units('m/s'); dirs_deg = np.array(dirs)*units.degrees; u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    grid_lon, grid_lat = np.linspace(min(lons_mapa), max(lons_mapa), 50), np.linspace(min(lats_mapa), max(lats_mapa), 50); X, Y = np.meshgrid(grid_lon, grid_lat); points = np.vstack((lons_mapa, lats_mapa)).T
    u_grid = griddata(points, u_comp.m, (X, Y), method='cubic'); v_grid = griddata(points, v_comp.m, (X, Y), method='cubic'); u_grid, v_grid = np.nan_to_num(u_grid), np.nan_to_num(v_grid)
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y); divergence_grid = mpcalc.divergence(u_grid*units('m/s'), v_grid*units('m/s'), dx=dx, dy=dy) * 1e5
    convergencia_per_poble = {}
    for nom_poble, coords in _localitats_dict.items():
        try:
            lon_idx, lat_idx = (np.abs(grid_lon - coords['lon'])).argmin(), (np.abs(grid_lat - coords['lat'])).argmin()
            convergencia_per_poble[nom_poble] = divergence_grid.m[lat_idx, lon_idx]
        except Exception: continue
    return convergencia_per_poble

def processar_sondeig_per_hora(sondeo, hourly_index, p_levels):
    try:
        hourly = sondeo.Hourly(); T_s, Td_s, P_s = (hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i in range(3))
        if np.isnan(P_s): return None
        s_idx, n_plvls = 3, len(p_levels); T_p, Td_p, Ws_p, Wd_p, H_p = ([hourly.Variables(s_idx + i*n_plvls + j).ValuesAsNumpy()[hourly_index] for j in range(n_plvls)] for i in range(5))
        def interpolate_sfc(sfc_val, p_sfc, p_api, d_api):
            valid_p, valid_d = [p for p, t in zip(p_api, d_api) if not np.isnan(t)], [t for t in d_api if not np.isnan(t)]
            if np.isnan(sfc_val) and len(valid_p) > 1: p_sorted, d_sorted = zip(*sorted(zip(valid_p, valid_d))); return np.interp(p_sfc, p_sorted, d_sorted)
            return sfc_val
        T_s, Td_s = interpolate_sfc(T_s, P_s, p_levels, T_p), interpolate_sfc(Td_s, P_s, p_levels, Td_p)
        if np.isnan(T_s) or np.isnan(Td_s): return None
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [P_s], [T_s], [Td_s], [0.0], [0.0], [mpcalc.pressure_to_height_std(P_s*units.hPa).m]
        for i, p_level in enumerate(p_levels):
            if p_level < P_s and not np.isnan(T_p[i]):
                p_profile.append(p_level); T_profile.append(T_p[i]); Td_profile.append(Td_p[i]); h_profile.append(H_p[i])
                u_comp, v_comp = mpcalc.wind_components(Ws_p[i]*units.knots, Wd_p[i]*units.degrees)
                u_profile.append(u_comp.to('m/s').m); v_profile.append(v_comp.to('m/s').m)
        return (np.array(p_profile)*units.hPa, np.array(T_profile)*units.degC, np.array(Td_profile)*units.degC,
                np.array(u_profile)*units.m/units.s, np.array(v_profile)*units.m/units.s, np.array(h_profile)*units.m)
    except Exception: return None

def get_next_arome_update_time():
    now_utc = datetime.now(pytz.utc); run_hours_utc = [0, 6, 12, 18]; availability_delay = timedelta(hours=4); next_update_time = None
    for run_hour in run_hours_utc:
        available_time = now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0) + availability_delay
        if available_time > now_utc: next_update_time = available_time; break
    if next_update_time is None: next_update_time = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) + availability_delay
    return f"Pròxima actualització de dades (AROME) estimada a les {next_update_time.astimezone(pytz.timezone('Europe/Madrid')).strftime('%H:%Mh')}"

def calculate_parameters(p, T, Td, u, v, h):
    params = {}; def get_val(qty, unit=None):
        try: return qty.to(unit).m if unit else qty.m
        except: return None
    params['SFC_Temp'] = {'value': get_val(T[0], 'degC'), 'units': '°C'}
    try:
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]); cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
        params['CAPE_Brut'] = {'value': get_val(cape, 'J/kg'), 'units': 'J/kg'}; params['CIN_Fre'] = {'value': get_val(cin, 'J/kg'), 'units': 'J/kg'}
        if params.get('CAPE_Brut', {}).get('value', 0) > 0:
            params['W_MAX'] = {'value': np.sqrt(2 * params['CAPE_Brut']['value']), 'units': 'm/s'}; params['CAPE_Utilitzable'] = {'value': max(0, params['CAPE_Brut']['value'] - abs(params.get('CIN_Fre', {}).get('value', 0))), 'units': 'J/kg'}
    except: pass
    try: lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); lcl_h = mpcalc.pressure_to_height_std(lcl_p); params['LCL_AGL'] = {'value': get_val(lcl_h - h[0], 'm'), 'units': 'm'}
    except: pass
    try: lfc_p, _ = mpcalc.lfc(p, T, Td); lfc_h = mpcalc.pressure_to_height_std(lfc_p); params['LFC_AGL'] = {'value': get_val(lfc_h - h[0], 'm'), 'units': 'm'}
    except: pass
    try: el_p, _ = mpcalc.el(p, T, Td); el_h = mpcalc.pressure_to_height_std(el_p); params['EL_MSL'] = {'value': get_val(el_h, 'km'), 'units': 'km'}
    except: pass
    try: s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6*units.km); params['Shear_0-6km'] = {'value': get_val(mpcalc.wind_speed(s_u, s_v), 'm/s'), 'units': 'm/s'}
    except: pass
    try: _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=1*units.km); params['SRH_0-1km'] = {'value': get_val(srh), 'units': 'm²/s²'}
    except: pass
    try: pwat = mpcalc.precipitable_water(p, Td); params['PWAT_Total'] = {'value': get_val(pwat, 'mm'), 'units': 'mm'}
    except: pass
    try:
        T_c, H_m = T.to('degC').m, h.to('m').m
        # <<< LÍNIA CORREGIDA >>>
        idx = np.where(np.diff(np.sign(T_c)))[0]
        if idx.size > 0:
            h_zero_iso_msl = np.interp(0, [T_c[idx[0]+1], T_c[idx[0]]], [H_m[idx[0]+1], H_m[idx[0]]]); params['ZeroIso_AGL'] = {'value': (h_zero_iso_msl - H_m[0]), 'units': 'm'}
    except: pass
    try:
        p_levels_interp = np.arange(p.m.min(), p.m.max(), 10)*units.hPa; T_interp = mpcalc.interpolate_1d(p, T, p_levels_interp)
        params['LapseRate_700_500'] = {'value': get_val(mpcalc.lapse_rate(p_levels_interp, T_interp, bottom=700*units.hPa, top=500*units.hPa), 'delta_degC/km'), 'units': '°C/km'}
    except: pass
    try: dcape, _ = mpcalc.dcape(p, T, Td); params['DCAPE'] = {'value': get_val(dcape, 'J/kg'), 'units': 'J/kg'}
    except: pass
    try: params['STP_cin'] = {'value': get_val(mpcalc.significant_tornado(cape=params.get('CAPE_Utilitzable',{}).get('value',0)*units('J/kg'), lcl_height=params.get('LCL_AGL',{}).get('value',9999)*units.m, storm_helicity=params.get('SRH_0-1km',{}).get('value',0)*units('m^2/s^2'), bulk_shear=params.get('Shear_0-6km',{}).get('value',0)*units('m/s'))), 'units': ''}
    except: pass
    return params

# --- 2. FUNCIONS DE VISUALITZACIÓ I FORMAT ---
def display_avis_principal(titol_avís, text_avís, color_avís, icona_personalitzada=None):
    icon_map = {"ESTABLE": "☀️", "RISC BAIX": "☁️", "PRECAUCIÓ": "⚡️", "AVÍS": "⚠️", "RISC ALT": "🌪️", "POTENCIAL SEVER": "🧐", "POTENCIAL MODERAT": "🤔", "ALERTA DE DISPARADOR": "🎯"}
    icona = icona_personalitzada if icona_personalitzada else icon_map.get(titol_avís, "ℹ️")
    st.markdown(f"""<div style="padding: 1rem; border-radius: 0.5rem; background-color: var(--secondary-background-color); border-left: 8px solid {color_avís}; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem;"><div style="font-size: 3.5em; line-height: 1;">{icona}</div><div><h3 style="color: {color_avís}; margin-top: 0; margin-bottom: 0.5rem; font-weight: bold;">{titol_avís}</h3><p style="margin-bottom: 0; color: var(--text-color);">{text_avís}</p></div></div>""", unsafe_allow_html=True)

def get_parameter_style(param_name, value):
    color = "inherit"; emoji = ""
    if value is None or not isinstance(value, (int, float, np.number)): return color, emoji
    if param_name == 'SFC_Temp':
        if value > 36: color, emoji = "#FF0000", "🔥"
        elif value > 32: color = "#FF4500"
        elif value <= 0: color, emoji = "#0000FF", "🥶"
    elif param_name == 'CIN_Fre':
        if value >= -25: color, emoji = "#32CD32", "✅"
        elif value < -150: color, emoji = "#FF4500", "⛔"
    elif 'CAPE' in param_name:
        if value > 3500: color, emoji = "#FF00FF", "💥"
        elif value > 2500: color = "#FF4500"
        elif value > 1500: color = "#FFA500"
    elif 'Shear' in param_name:
        if value > 25: color, emoji = "#FF4500", "↔️"
        elif value > 18: color = "#FFA500"
    elif 'SRH' in param_name:
        if value > 400: color, emoji = "#FF00FF", "🔄"
        elif value > 250: color = "#FF4500"
    elif 'DCAPE' in param_name:
        if value > 1200: color, emoji = "#FF4500", "💨"
        elif value > 800: color = "#FFA500"
    elif 'STP' in param_name:
        if value > 1: color, emoji = "#FF00FF", "🌪️"
        elif value > 0.5: color = "#FFA500"
    return color, emoji

def generar_avis_localitat(params, is_convergence_active):
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0); cin = params.get('CIN_Fre', {}).get('value'); shear = params.get('Shear_0-6km', {}).get('value'); srh1 = params.get('SRH_0-1km', {}).get('value'); lcl_agl = params.get('LCL_AGL', {}).get('value', 9999); lfc_agl = params.get('LFC_AGL', {}).get('value', 9999)
    dcape = params.get('DCAPE', {}).get('value', 0); stp = params.get('STP_cin', {}).get('value', 0); pwat = params.get('PWAT_Total', {}).get('value', 0)
    if cape_u < 100: return "ESTABLE", "Sense risc de tempestes significatives.", "#3CB371"
    if cin is not None and cin < -150: return "ESTABLE", f"La 'tapa' atmosfèrica (CIN de {cin:.0f} J/kg) és massa forta.", "#3CB371"
    if not is_convergence_active and lfc_agl > 3500: return "RISC BAIX", f"L'inici de la convecció (LFC a {lfc_agl:.0f} m) és massa alt.", "#4682B4"
    if shear is not None and shear > 20 and cape_u > 1500 and srh1 is not None and srh1 > 200 and lcl_agl < 1300 and stp > 0.8:
        risks = ["calamarsa grossa", "vent destructiu"];
        if lcl_agl < 1000: risks.insert(0, "tornados");
        return "RISC ALT", f"Entorn de SUPERCL·LULA TORNÀDICA. Risc de {', '.join(risks)}.", "#E60073"
    if shear is not None and shear > 18 and cape_u > 1200:
        risks = []
        if dcape > 900: risks.append("fortes ràfegues de vent")
        if pwat > 35: risks.append("pluges torrencials")
        else: risks.append("fortes pluges i calamarsa")
        return "AVÍS", f"Potencial per a tempestes SEVERES. Risc de {', '.join(risks)}.", "#FF8C00"
    if shear is not None and shear > 12 and cape_u > 500:
        return "PRECAUCIÓ", "Risc de TEMPESTES ORGANITZADES. Possibles fortes pluges i calamarsa local.", "#FFD700"
    return "RISC BAIX", "Possibles xàfecs o tempestes febles i aïllades.", "#4682B4"

def generar_analisi_detallada(params, is_convergence_active):
    def stream_text(text):
        for word in text.split(): yield word + " "; time.sleep(0.02)
        yield "\n\n"
    cape_u=params.get('CAPE_Utilitzable',{}).get('value',0); cin=params.get('CIN_Fre',{}).get('value'); shear6=params.get('Shear_0-6km',{}).get('value'); srh1=params.get('SRH_0-1km',{}).get('value'); lcl_agl=params.get('LCL_AGL',{}).get('value'); w_max=params.get('W_MAX',{}).get('value'); el_msl = params.get('EL_MSL', {}).get('value'); dcape=params.get('DCAPE',{}).get('value'); stp=params.get('STP_cin',{}).get('value'); pwat=params.get('PWAT_Total',{}).get('value')
    if cape_u is None or cape_u < 100: yield from stream_text("L'atmosfera és estable. Sense potencial per a tempestes."); return
    yield from stream_text("### Potencial Energètic")
    cape_text="feble" if cape_u<1000 else "moderada" if cape_u<2500 else "forta"
    yield from stream_text(f"**Inestabilitat (CAPE):** El valor és de **{cape_u:.0f} J/kg** (inestabilitat **{cape_text}**), amb corrents ascendents potencials de **{w_max*3.6:.0f} km/h** i un cim de núvol a **{el_msl:.1f} km**.")
    if cin is not None:
        if cin >= -25: cin_text = "La 'tapa' és molt feble. La convergència forta (si existeix) o la calor iniciaran tempestes fàcilment."
        elif -75 < cin <= -25: cin_text = "La 'tapa' és moderada. Es necessita un mecanisme de tret com la convergència forta per trencar-la."
        else: cin_text = "La 'tapa' és forta. Les tempestes són molt improbables."
        yield from stream_text(f"**Inhibició (CIN):** El valor és de **{cin:.0f} J/kg**. {cin_text}")
    yield from stream_text("### Organització i Rotació")
    if shear6 is not None: yield from stream_text(f"**Cisallament 0-6 km:** **{shear6:.1f} m/s**. Valors > 18 m/s afavoreixen supercèl·lules.")
    if srh1 is not None and srh1 > 100: yield from stream_text(f"**Helicitat 0-1 km (SRH):** **{srh1:.0f} m²/s²**. Valors > 250 m²/s², amb LCL baix, augmenten el risc de tornados.")
    yield from stream_text("### Riscos Específics")
    if stp and stp > 0.5: yield from stream_text(f"**Potencial Tornàdic (STP):** **{stp:.1f}**. Valors > 1 indiquen un entorn favorable per a supercèl·lules tornàdiques.")
    if dcape and dcape > 800: yield from stream_text(f"**Potencial de Vent Sever (DCAPE):** **{dcape:.0f} J/kg**. Potencial per a esclafits forts.")
    if pwat and pwat > 30: yield from stream_text(f"**Potencial de Precipitació Intensa:** **{pwat:.1f} mm**. Contingut d'humitat elevat.")

def display_metrics(params_dict):
    param_map = [('Temperatura','SFC_Temp'), ('CAPE Utilitzable','CAPE_Utilitzable'), ('CIN (Fre)','CIN_Fre'), ('Vel. Asc. Màx.','W_MAX'), ('Shear 0-6km','Shear_0-6km'), ('SRH 0-1km','SRH_0-1km'), ('Potencial Tornàdic','STP_cin'), ('Potencial Esclafits','DCAPE'), ('Gradient Tèrmic','LapseRate_700_500'), ('Aigua Precipitable','PWAT_Total'), ('Base núvol (AGL)','LCL_AGL'), ('Cim tempesta (MSL)','EL_MSL')]
    st.markdown("""<style>.metric-container{border:1px solid rgba(128,128,128,0.2);border-radius:10px;padding:10px;margin-bottom:10px;}</style>""", unsafe_allow_html=True)
    available_params=[(label,key) for label,key in param_map if key in params_dict and params_dict[key].get('value') is not None]
    cols=st.columns(min(4,len(available_params)))
    for i,(label,key) in enumerate(available_params):
        param=params_dict[key]; value=param['value']; units_str=param['units']; val_str=f"{value:.1f}" if isinstance(value,(float,np.floating)) else str(value); value_color,emoji=get_parameter_style(key,value); border_color=value_color if value_color!='inherit' else 'rgba(128,128,128,0.2)'
        with cols[i%4]: st.markdown(f"""<div class="metric-container" style="border-color:{border_color};"><div style="font-size:0.9em;color:gray;">{label}</div><div style="font-size:1.25em;font-weight:bold;color:{value_color};">{val_str} <span style='font-size:0.8em;color:gray;'>{units_str}</span> {emoji}</div></div>""", unsafe_allow_html=True)

def crear_mapa_base(nivell, lat_sel, lon_sel, nom_poble_sel, titol):
    fig=plt.figure(figsize=(9,9),dpi=150); ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree()); ax.set_extent([0,3.5,40.4,43],crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND,facecolor="#E0E0E0",zorder=0); ax.add_feature(cfeature.OCEAN,facecolor='#b0c4de',zorder=0); ax.add_feature(cfeature.COASTLINE,edgecolor='black',linewidth=0.5,zorder=1); ax.add_feature(cfeature.BORDERS,linestyle=':',edgecolor='black',zorder=1); ax.plot(lon_sel,lat_sel,'o',markersize=12,markerfacecolor='yellow',markeredgecolor='black',markeredgewidth=2,transform=ccrs.Geodetic(),zorder=5); ax.text(lon_sel+0.05,lat_sel+0.05,nom_poble_sel,transform=ccrs.Geodetic(),zorder=6,bbox=dict(facecolor='white',alpha=0.8,edgecolor='none',boxstyle='round,pad=0.2')); ax.set_title(f"{titol} a {nivell}hPa",weight='bold'); return fig,ax

def crear_mapa_vents(lats, lons, data, nivell, lat_sel, lon_sel, nom_poble_sel):
    speeds,dirs = zip(*data); speeds_ms = (np.array(speeds)*1000/3600)*units('m/s'); dirs_deg = np.array(dirs)*units.degrees; u_comp,v_comp = mpcalc.wind_components(speeds_ms,dirs_deg); fig,ax = crear_mapa_base(nivell, lat_sel, lon_sel, nom_poble_sel, "Flux i focus de convergència")
    grid_lon,grid_lat = np.linspace(min(lons), max(lons), 100), np.linspace(min(lats), max(lats), 100); X,Y = np.meshgrid(grid_lon,grid_lat); points = np.vstack((lons,lats)).T
    u_grid = griddata(points, u_comp.m, (X,Y), method='cubic'); v_grid = griddata(points, v_comp.m, (X,Y), method='cubic'); u_grid,v_grid = np.nan_to_num(u_grid),np.nan_to_num(v_grid)
    dx,dy = mpcalc.lat_lon_grid_deltas(X,Y); divergence = mpcalc.divergence(u_grid*units('m/s'), v_grid*units('m/s'), dx=dx, dy=dy) * 1e5
    divergence_values = np.ma.masked_where(divergence.m > -21.0, divergence.m); levels = np.linspace(-50.0, -15.0, 11)
    cont_fill = ax.contourf(X, Y, divergence_values, levels=levels, cmap='hot_r', alpha=0.5, zorder=2, transform=ccrs.PlateCarree(), extend='min'); fig.colorbar(cont_fill, ax=ax, orientation='vertical', label='Convergència (x10⁻⁵ s⁻¹)', shrink=0.7)
    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color="black", density=5, linewidth=0.6, arrowsize=0.4, zorder=4, transform=ccrs.PlateCarree()); return fig

def crear_hodograf(p, u, v, h):
    fig, ax=plt.subplots(1,1,figsize=(5,5)); hodo=Hodograph(ax,component_range=40.); hodo.add_grid(increment=10); hodoline=hodo.plot_colormapped(u.to('kt'),v.to('kt'),h.to('km'),cmap='gist_ncar'); plt.colorbar(hodoline,ax=ax,orientation='vertical',pad=0.05,shrink=0.8).set_label('Altitud (km)'); ax.set_xlabel('kt'); ax.set_ylabel('kt'); return fig

def crear_skewt(p, T, Td, u, v):
    fig = plt.figure(figsize=(7, 9)); skew = SkewT(fig, rotation=45); skew.plot(p, T, 'r', lw=2); skew.plot(p, Td, 'b', lw=2)
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), length=7, color='black'); skew.plot_dry_adiabats(color='lightcoral', ls='--', alpha=0.5); skew.plot_moist_adiabats(color='cornflowerblue', ls='--', alpha=0.5); skew.plot_mixing_lines(color='lightgreen', ls='--', alpha=0.5); skew.ax.axvline(0, color='darkturquoise', linestyle='--')
    try:
        prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', lw=2, ls='--')
        skew.shade_cape(p, T, prof, alpha=0.3, color='orange'); skew.shade_cin(p, T, prof, alpha=0.6, color='gray')
    except Exception as e:
        # Afegit per a debug en cas que el càlcul del perfil falli
        st.toast(f"No s'ha pogut calcular el perfil de la parcel·la: {e}", icon="⚠️")
        pass
    skew.ax.set_ylim(1050, 100); skew.ax.set_xlim(-40, 40); skew.ax.set_xlabel('Temperatura (°C)'); skew.ax.set_ylabel('Pressió (hPa)'); return fig

def crear_grafic_nuvol(params, H, u, v, is_convergence_active):
    lcl_agl, el_msl_km, cape = (params.get(k, {}).get('value') for k in ['LCL_AGL', 'EL_MSL', 'CAPE_Brut'])
    if lcl_agl is None or el_msl_km is None: return None
    cape = cape or 0; fig, ax = plt.subplots(figsize=(6, 9), dpi=120); ax.set_facecolor('#4F94CD'); center_x_base = 5.0
    if is_convergence_active:
        lcl_km, el_km = lcl_agl / 1000, el_msl_km; y_points = np.linspace(lcl_km, el_km, 100); cloud_width = 1.0 + np.sin(np.pi * (y_points - lcl_km) / (el_km - lcl_km)) * (1 + cape / 2000)
        for y, width in zip(y_points, cloud_width):
            tilt_offset = np.interp(y * 1000, H.m, u.m) / 15; center_x = center_x_base + tilt_offset
            for _ in range(25): ax.add_patch(Circle((center_x + (random.random() - 0.5) * width, y + (random.random() - 0.5) * 0.4), 0.2 + random.random() * 0.4, color='white', alpha=0.15, lw=0))
        top_cloud_tilt_offset = np.interp(el_km * 1000, H.m, u.m) / 15; anvil_center_x = center_x_base + top_cloud_tilt_offset; anvil_wind_spread = np.interp(el_km * 1000, H.m, u.m) / 10
        for _ in range(80): ax.add_patch(Circle((anvil_center_x + (random.random() - 0.2) * 4 + anvil_wind_spread, el_km + (random.random() - 0.5) * 0.5), 0.2 + random.random() * 0.6, color='white', alpha=0.2, lw=0))
    else: ax.text(center_x_base, 8, "Sense disparador/energia.", ha='center', va='center', color='black', fontsize=16, weight='bold')
    barb_heights_km = np.arange(1, 16, 1); u_barbs = np.interp(barb_heights_km * 1000, H.m, u.to('kt').m); v_barbs = np.interp(barb_heights_km * 1000, H.m, v.to('kt').m)
    ax.barbs(np.full_like(barb_heights_km, 9.5), barb_heights_km, u_barbs, v_barbs, length=7, color='black'); ax.set_ylim(0, 16); ax.set_xlim(0, 10); ax.set_ylabel("Altitud (km, MSL)"); ax.set_title("Visualització del Núvol"); ax.set_xticks([]); ax.grid(axis='y', linestyle='--', alpha=0.3); return fig

# --- 3. INTERFÍCIE PRINCIPAL ---
st.markdown('<h1 style="text-align: center; color: #FF4B4B;">⚡ Tempestes.cat</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Eina d\'Anàlisi i Previsió de Fenòmens Severs a Catalunya</p>', unsafe_allow_html=True)

tab_api, tab_manual = st.tabs(["📈 Dades en Viu (API)", "📋 Anàlisi Manual (TXT)"])

with tab_api:
    hourly_index_global = int(st.session_state.hora_selector.split(':')[0])
    with st.spinner("Calculant convergències..."):
        convergencies_850hpa = calcular_convergencia_per_totes_les_localitats(hourly_index_global, 850, pobles_data)
        localitats_convergencia_forta = {p for p, v in convergencies_850hpa.items() if v is not None and v < CONVERGENCIA_FORTA_THRESHOLD}
    with st.container(border=True):
        col1, col2 = st.columns([1,1], gap="large")
        with col1: st.selectbox("Hora:", [f"{h:02d}:00h" for h in range(24)], key="hora_selector")
        with col2:
            sorted_pobles = sorted(pobles_data.keys()); opciones_display = [f"⚠️ {p}" if p in localitats_convergencia_forta else p for p in sorted_pobles]
            poble_actual_net = st.session_state.poble_selector.replace("⚠️ ", "").strip(); poble_actual_display = f"⚠️ {poble_actual_net}" if poble_actual_net in localitats_convergencia_forta else poble_actual_net
            try: default_index = opciones_display.index(poble_actual_display)
            except ValueError: default_index = 0
            selected_option_raw = st.selectbox('Localitat:', options=opciones_display, index=default_index)
            st.session_state.poble_selector = selected_option_raw.replace("⚠️ ", "").strip()
    st.markdown(f'<p style="text-align: center; font-size: 0.9em; color: grey;">🕒 {get_next_arome_update_time()}</p>', unsafe_allow_html=True)
    hourly_index, poble_sel = int(st.session_state.hora_selector.split(':')[0]), st.session_state.poble_selector; lat_sel, lon_sel = pobles_data[poble_sel]['lat'], pobles_data[poble_sel]['lon']
    
    with st.spinner(f"Carregant sondeig per a {poble_sel}..."): sondeo, p_levels, error_sondeo = carregar_sondeig_per_poble(poble_sel, lat_sel, lon_sel)
    if error_sondeo: st.error(f"Error de l'API: {error_sondeo}")
    elif sondeo:
        profiles = processar_sondeig_per_hora(sondeo, hourly_index, p_levels)
        if profiles:
            p, T, Td, u, v, H = profiles; parametros = calculate_parameters(p, T, Td, u, v, H)
            is_disparador_active = poble_sel in localitats_convergencia_forta
            avis_titol, avis_text, avis_color = generar_avis_localitat(parametros, is_disparador_active); display_avis_principal(avis_titol, avis_text, avis_color)
            tab_analisi, tab_params, tab_mapes, tab_hodo, tab_sondeig, tab_nuvol = st.tabs(["🗨️ Anàlisi", "📊 Paràmetres", "🗺️ Mapes", "🧭 Hodògraf", "📍 Sondeig", "☁️ Visualització"])
            with tab_analisi: st.write_stream(generar_analisi_detallada(parametros, is_disparador_active))
            with tab_params: display_metrics(parametros)
            with tab_mapes:
                nivell_global = st.selectbox("Nivell:", [1000, 925, 850, 700, 500, 300], index=2, key="map_level_selector")
                with st.spinner("Generant mapa..."):
                    lats, lons, data, error = obtener_dades_mapa("wind", nivell_global, hourly_index, FORECAST_DAYS)
                    if error: st.error(f"Error mapa: {error}")
                    elif lats and len(lats) > 3: st.pyplot(crear_mapa_vents(lats, lons, data, nivell_global, lat_sel, lon_sel, poble_sel))
                    else: st.warning("No hi ha prou dades per generar el mapa.")
            with tab_hodo: st.pyplot(crear_hodograf(p, u, v, H))
            with tab_sondeig: st.pyplot(crear_skewt(p, T, Td, u, v))
            with tab_nuvol: st.pyplot(crear_grafic_nuvol(parametros, H, u, v, is_disparador_active))
        else: st.warning(f"No s'han pogut calcular els paràmetres per a les {hourly_index:02d}:00h.")

with tab_manual:
    st.header("Analitzador de Sondeig Manual")
    st.info("Enganxa aquí les dades d'un sondeig en format text (p. ex. de Meteociel) per a una anàlisi detallada.")
    with st.expander("Mostra'm un exemple del format de text"):
        st.code("""Mercredi 13 août 2025 17:00 locale (+3h)\n  	 Run 12Z du Mercredi 13 août 2025  \nAltitude	Pression	Température	Tw	Point de rosée	Humidité	Vent\n15310 m	125 hPa	-59.4°C	-59.4°C	-83.4°C	3 %	202 ° / 26.3 kt\n...\n689 m (Sol)	937 hPa	35.8°C	18.9°C	8.5°C	19 %	270 ° / 11.8 kt""", language="text")
    text_sondeig = st.text_area("Dades del sondeig:", height=300, label_visibility="collapsed")
    if st.button("Analitzar Dades Manuals", type="primary"):
        if not text_sondeig: st.warning("Enganxa les dades del sondeig.")
        else:
            with st.spinner("Processant dades..."): profiles, error_parse = parsear_sondeig_manual(text_sondeig)
            if error_parse: st.error(f"Error en processar: {error_parse}")
            elif profiles:
                st.success("Sondeig processat correctament!")
                p, T, Td, u, v, H = profiles; parametros = calculate_parameters(p, T, Td, u, v, H)
                avis_titol, avis_text, avis_color = generar_avis_localitat(parametros, False); display_avis_principal(avis_titol, avis_text, avis_color)
                tab_m_analisi, tab_m_params, tab_m_hodo, tab_m_sondeig, tab_m_nuvol = st.tabs(["🗨️ Anàlisi", "📊 Paràmetres", "🧭 Hodògraf", "📍 Sondeig", "☁️ Visualització"])
                with tab_m_analisi: st.write_stream(generar_analisi_detallada(parametros, False))
                with tab_m_params: display_metrics(parametros)
                with tab_m_hodo: st.pyplot(crear_hodograf(p, u, v, H))
                with tab_m_sondeig: st.pyplot(crear_skewt(p, T, Td, u, v))
                with tab_m_nuvol:
                    cape_brut = parametros.get('CAPE_Brut', {}).get('value', 0)
                    st.pyplot(crear_grafic_nuvol(parametros, H, u, v, is_convergence_active=(cape_brut > 100)))
