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
from scipy.interpolate import Rbf # Afegeix això amb els altres imports

# --- CONFIGURACIÓ INICIAL ---
st.set_page_config(layout="wide", page_title="Tempestes.cat")

plain_session = requests.Session()
retry_session = retry(plain_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

FORECAST_DAYS = 1
CONVERGENCIA_FORTA_THRESHOLD = -25

# --- DADES DE LOCALITATS (LLISTA OPTIMITZADA PER A ZONES TEMPESTUOSES) ---
pobles_data = {
    # Prepirineu i Valls Pirinenques (Focus de disparador orogràfic)
    'Berga': {'lat': 42.105, 'lon': 1.846},
    'Solsona': {'lat': 41.994, 'lon': 1.518},
    'Ripoll': {'lat': 42.201, 'lon': 2.190},        # AFEGIT: Clau al Ripollès
    
    # Catalunya Central i zones d'influència orogràfica
    'Vic': {'lat': 41.930, 'lon': 2.255},
    'Olot': {'lat': 42.181, 'lon': 2.490},         # AFEGIT: Zona Volcànica de la Garrotxa
    'Manresa': {'lat': 41.727, 'lon': 1.825},       # AFEGIT: Cor de la Catalunya Central
    
    # Ponent i Depressió Central (On les tempestes s'intensifiquen)
    'Lleida': {'lat': 41.617, 'lon': 0.620},        # AFEGIT: Capital de Ponent, focus de calor
    'Tàrrega': {'lat': 41.647, 'lon': 1.140},
    'Albi': {'lat': 41.423, 'lon': 0.995},
    
    # Interior de Tarragona (Influència del Sistema Ibèric)
    'Montblanc': {'lat': 41.376, 'lon': 1.161},
    'Gandesa': {'lat': 41.052, 'lon': 0.437},
    
    # Nord-est i zones amb influència marítima
    'Figueres': {'lat': 42.266, 'lon': 2.963},     # AFEGIT: Clau a l'Alt Empordà
    'Arbúcies': {'lat': 41.815, 'lon': 2.515},
    'Arenys de Mar': {'lat': 41.581, 'lon': 2.551},
}
# --- FUNCIÓ DE CALLBACK ---
def actualitzar_seleccio(poble, hora):
    st.session_state.poble_selector = poble
    st.session_state.hora_selector = f"{hora:02d}:00h"
    st.session_state.avisos_expanded = False

# --- INICIALITZACIÓ DEL SESSION STATE (CORREGIDA) ---
if 'poble_selector' not in st.session_state or st.session_state.poble_selector not in pobles_data:
    st.session_state.poble_selector = 'Vic' # Un valor per defecte que SÍ existeix a la llista
if 'hora_selector' not in st.session_state:
    tz = pytz.timezone('Europe/Madrid')
    st.session_state.hora_selector = f"{datetime.now(tz).hour:02d}:00h"
if 'nivell_mapa' not in st.session_state:
    st.session_state.nivell_mapa = 850
if 'avisos_expanded' not in st.session_state:
    st.session_state.avisos_expanded = True

# --- 1. LÒGICA DE CÀRREGA DE DADES I CÀLCUL ---
@st.cache_data(ttl=16000)
def carregar_sondeig_per_poble(nom_poble, lat, lon):
    p_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"] for p in p_levels]
    params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "timezone": "auto", "forecast_days": FORECAST_DAYS}
    try:
        respostes = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        return respostes[0], p_levels, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data(ttl=16000)
def obtener_dades_mapa(variable, nivell, hourly_index, forecast_days):
    lats, lons = np.linspace(40.5, 42.8, 12), np.linspace(0.2, 3.3, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    api_vars = []
    if variable == 'temp_height': api_vars = [f"temperature_{nivell}hPa", f"geopotential_height_{nivell}hPa"]
    elif variable == 'wind': api_vars = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
    elif variable == 'dewpoint': api_vars = [f"dew_point_{nivell}hPa"]
    elif variable == 'humidity': api_vars = [f"relative_humidity_{nivell}hPa"]
    elif variable == 'cape': api_vars = ["cape"]
    else: api_vars = [f"temperature_{nivell}hPa"]
    params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": api_vars, "models": "arome_seamless", "timezone": "auto", "forecast_days": forecast_days}
    try:
        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        lats_out, lons_out, data_out = [], [], []
        for r in responses:
            hourly = r.Hourly()
            values = [hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(api_vars))]
            if not any(np.isnan(v) for v in values):
                lats_out.append(r.Latitude()); lons_out.append(r.Longitude()); data_out.append(tuple(values) if len(values) > 1 else values[0])
        if not lats_out: return None, None, None, "No s'han rebut dades vàlides del model."
        return lats_out, lons_out, data_out, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data(ttl=16000)
def calcular_convergencia_per_totes_les_localitats(_hourly_index, _nivell, _localitats_dict):
    lats_mapa, lons_mapa, data_mapa, error = obtener_dades_mapa('wind', _nivell, _hourly_index, FORECAST_DAYS)
    if error or not lats_mapa or len(lats_mapa) < 4: return {}
    speeds, dirs = zip(*data_mapa); speeds_ms = (np.array(speeds) * 1000 / 3600) * units('m/s'); dirs_deg = np.array(dirs) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    grid_lon, grid_lat = np.linspace(min(lons_mapa), max(lons_mapa), 50), np.linspace(min(lats_mapa), max(lats_mapa), 50)
    X, Y = np.meshgrid(grid_lon, grid_lat); points = np.vstack((lons_mapa, lats_mapa)).T
    u_grid = griddata(points, u_comp.m, (X, Y), method='cubic'); v_grid = griddata(points, v_comp.m, (X, Y), method='cubic')
    u_grid, v_grid = np.nan_to_num(u_grid), np.nan_to_num(v_grid); dx, dy = mpcalc.lat_lon_grid_deltas(X, Y)
    divergence_grid = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5
    convergencia_per_poble = {}
    for nom_poble, coords in _localitats_dict.items():
        try:
            lon_idx, lat_idx = (np.abs(grid_lon - coords['lon'])).argmin(), (np.abs(grid_lat - coords['lat'])).argmin()
            convergencia_per_poble[nom_poble] = divergence_grid.m[lat_idx, lon_idx]
        except Exception: continue
    return convergencia_per_poble

@st.cache_data(ttl=16000)
def precalcular_potencials_del_dia(_pobles_data):
    horas_muestreadas = [0, 3, 6, 9, 12, 15, 18, 21]; potencials = {}
    avisos_a_buscar = {"PRECAUCIÓ", "AVÍS", "RISC ALT", "ALERTA DE DISPARADOR"}
    for nom_poble, coords in _pobles_data.items():
        try:
            sondeo, p_levels, _ = carregar_sondeig_per_poble(nom_poble, coords['lat'], coords['lon'])
            if sondeo:
                for hora in horas_muestreadas:
                    profiles = processar_sondeig_per_hora(sondeo, hora, p_levels)
                    if profiles:
                        parametros = calculate_parameters(*profiles)
                        if generar_avis_potencial_per_precalcul(parametros) in avisos_a_buscar:
                            potencials[nom_poble] = hora; break
            time.sleep(0.2)
        except Exception: continue
    return potencials

def generar_avis_potencial_per_precalcul(params):
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0); cin = params.get('CIN_Fre', {}).get('value')
    if cape_u > 500 and (cin is None or cin > -50): return "ALERTA DE DISPARADOR"
    shear = params.get('Shear_0-6km', {}).get('value'); srh1 = params.get('SRH_0-1km', {}).get('value'); lcl_agl = params.get('LCL_AGL', {}).get('value', 9999)
    if cape_u < 100 or (cin is not None and cin < -100): return "ESTABLE"
    cond_supercelula = shear is not None and shear > 20 and cape_u > 1500 and srh1 is not None and srh1 > 250 and lcl_agl < 1200
    cond_avis_sever = shear is not None and shear > 18 and cape_u > 1000
    cond_precaucio = shear is not None and shear > 12 and cape_u > 500
    if cond_supercelula: return "RISC ALT"
    if cond_avis_sever: return "AVÍS"
    if cond_precaucio: return "PRECAUCIÓ"
    return "RISC BAIX"

def processar_sondeig_per_hora(sondeo, hourly_index, p_levels):
    try:
        hourly = sondeo.Hourly(); T_s, Td_s, P_s = (hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i in range(3))
        if np.isnan(P_s): return None
        s_idx, n_plvls = 3, len(p_levels); T_p, Td_p, Ws_p, Wd_p, H_p = ([hourly.Variables(s_idx + i*n_plvls + j).ValuesAsNumpy()[hourly_index] for j in range(n_plvls)] for i in range(5))
        def interpolate_sfc(sfc_val, p_sfc, p_api, d_api):
            valid_p, valid_d = [p for p, t in zip(p_api, d_api) if not np.isnan(t)], [t for t in d_api if not np.isnan(t)]
            if np.isnan(sfc_val) and len(valid_p) > 1:
                p_sorted, d_sorted = zip(*sorted(zip(valid_p, valid_d))); return np.interp(p_sfc, p_sorted, d_sorted)
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
    now_utc = datetime.now(pytz.utc); run_hours_utc = [0, 6, 12, 18]; availability_delay = timedelta(hours=4)
    next_update_time = None
    for run_hour in run_hours_utc:
        available_time = now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0) + availability_delay
        if available_time > now_utc: next_update_time = available_time; break
    if next_update_time is None: next_update_time = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) + availability_delay
    return f"Pròxima actualització de dades (AROME) estimada a les {next_update_time.astimezone(pytz.timezone('Europe/Madrid')).strftime('%H:%Mh')}"

@st.cache_data(ttl=16000)
def _calculate_parameters_cached(p_m, p_units_str, T_m, T_units_str, Td_m, Td_units_str, u_m, u_units_str, v_m, v_units_str, h_m, h_units_str):
    p = p_m * units(p_units_str); T = T_m * units(T_units_str); Td = Td_m * units(Td_units_str)
    u = u_m * units(u_units_str); v = v_m * units(v_units_str); h = h_m * units(h_units_str)
    params = {}; get_val = lambda qty, unit=None: qty.to(unit).m if unit and hasattr(qty, 'is_compatible_with') and qty.is_compatible_with(unit) else qty.m
    try: params['SFC_Temp'] = {'value': get_val(T[0], 'degC'), 'units': '°C'}
    except Exception: pass
    try:
        prof = mpcalc.parcel_profile(p, T[0], Td[0]); cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params['CAPE_Brut'] = {'value': get_val(cape, 'J/kg'), 'units': 'J/kg'}; params['CIN_Fre'] = {'value': get_val(cin, 'J/kg'), 'units': 'J/kg'}
        if params.get('CAPE_Brut', {}).get('value', 0) > 0:
            params['W_MAX'] = {'value': np.sqrt(2 * params['CAPE_Brut']['value']), 'units': 'm/s'}
            params['CAPE_Utilitzable'] = {'value': max(0, params['CAPE_Brut']['value'] - abs(params['CIN_Fre']['value'])), 'units': 'J/kg'}
    except Exception: pass
    try: 
        lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); lcl_h = mpcalc.pressure_to_height_std(lcl_p)
        params['LCL_AGL'] = {'value': get_val(lcl_h - h[0], 'm'), 'units': 'm'}
    except Exception: pass
    try: 
        prof = mpcalc.parcel_profile(p, T[0], Td[0]); lfc_p, _ = mpcalc.lfc(p, T, Td, prof=prof)
        lfc_h = mpcalc.pressure_to_height_std(lfc_p); params['LFC_AGL'] = {'value': get_val(lfc_h - h[0], 'm'), 'units': 'm'}
    except Exception: pass
    try: 
        prof = mpcalc.parcel_profile(p, T[0], Td[0]); el_p, _ = mpcalc.el(p, T, Td, prof=prof)
        el_h = mpcalc.pressure_to_height_std(el_p); params['EL_MSL'] = {'value': get_val(el_h, 'km'), 'units': 'km'}
    except Exception: pass
    try: 
        s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6*units.km)
        params['Shear_0-6km'] = {'value': get_val(mpcalc.wind_speed(s_u, s_v), 'm/s'), 'units': 'm/s'}
    except Exception: pass
    try: 
        _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=1*units.km)
        params['SRH_0-1km'] = {'value': get_val(srh, 'm^2/s^2'), 'units': 'm²/s²'}
    except Exception: pass
    try: 
        pwat = mpcalc.precipitable_water(p, Td); params['PWAT_Total'] = {'value': get_val(pwat, 'mm'), 'units': 'mm'}
    except Exception: pass
    try:
        T_c, H_m = T.to('degC').m, h.to('m').m
        if (idx := np.where(np.diff(np.sign(T_c)))[0]).size > 0:
            h_zero_iso_msl = np.interp(0, [T_c[idx[0]+1], T_c[idx[0]]], [H_m[idx[0]+1], H_m[idx[0]]])
            params['ZeroIso_AGL'] = {'value': (h_zero_iso_msl - H_m[0]), 'units': 'm'}
    except Exception: pass
    try:
        p_levels_interp = np.arange(max(100, p.m.min()), p.m.max(), -10) * units.hPa; T_interp = mpcalc.interpolate_1d(p_levels_interp, T, p)
        lr = mpcalc.lapse_rate(p_levels_interp, T_interp, bottom=700*units.hPa, top=500*units.hPa)
        params['LapseRate_700_500'] = {'value': get_val(lr, 'delta_degC/km'), 'units': '°C/km'}
    except Exception: pass
    try: 
        dcape, _ = mpcalc.dcape(p, T, Td); params['DCAPE'] = {'value': get_val(dcape, 'J/kg'), 'units': 'J/kg'}
    except Exception: pass
    try:
        cape_val = params.get('CAPE_Utilitzable',{}).get('value',0) * units('J/kg'); lcl_val = params.get('LCL_AGL',{}).get('value',9999) * units.m
        srh_val = params.get('SRH_0-1km',{}).get('value',0) * units('m^2/s^2'); shear_val = params.get('Shear_0-6km',{}).get('value',0) * units('m/s')
        stp_cin = mpcalc.significant_tornado(cape=cape_val, lcl_height=lcl_val, storm_helicity=srh_val, bulk_shear=shear_val)
        params['STP_cin'] = {'value': get_val(stp_cin, dimensionless=True), 'units': ''}
    except Exception: pass
    return params

def calculate_parameters(p, T, Td, u, v, h):
    return _calculate_parameters_cached(p.m, str(p.units), T.m, str(T.units), Td.m, str(Td.units), u.m, str(u.units), v.m, str(v.units), h.m, str(h.units))

# --- 2. FUNCIONS DE VISUALITZACIÓ I FORMAT ---
def display_avis_principal(titol_avís, text_avís, color_avís, icona_personalitzada=None):
    icon_map = {"ESTABLE": "☀️", "RISC BAIX": "☁️", "PRECAUCIÓ": "⚡️", "AVÍS": "⚠️", "RISC ALT": "🌪️", "POTENCIAL SEVER": "🧐", "POTENCIAL MODERAT": "🤔", "ALERTA DE DISPARADOR": "🎯"}
    icona = icona_personalitzada if icona_personalitzada else icon_map.get(titol_avís, "ℹ️"); st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: var(--secondary-background-color); border-left: 8px solid {color_avís}; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 3.5em; line-height: 1;">{icona}</div>
        <div><h3 style="color: {color_avís}; margin-top: 0; margin-bottom: 0.5rem; font-weight: bold;">{titol_avís}</h3><p style="margin-bottom: 0; color: var(--text-color);">{text_avís}</p></div>
    </div>""", unsafe_allow_html=True)

def get_parameter_style(param_name, value):
    color = "inherit"; emoji = "";
    if value is None or not isinstance(value, (int, float, np.number)): return color, emoji
    if param_name == 'SFC_Temp':
        if value > 36: color, emoji = "#FF0000", "🔥";
        elif value > 32: color = "#FF4500";
        elif value <= 0: color, emoji = "#0000FF", "🥶"
    elif param_name == 'CIN_Fre':
        if value >= -25: color, emoji = "#32CD32", "✅";
        elif value < -150: color, emoji = "#FF4500", "⛔"
    elif 'CAPE' in param_name:
        if value > 3500: color, emoji = "#FF00FF", "💥";
        elif value > 2500: color = "#FF4500";
        elif value > 1500: color = "#FFA500"
    elif 'Shear' in param_name:
        if value > 25: color, emoji = "#FF4500", "↔️";
        elif value > 18: color = "#FFA500"
    elif 'SRH' in param_name:
        if value > 400: color, emoji = "#FF00FF", "🔄";
        elif value > 250: color = "#FF4500"
    elif 'DCAPE' in param_name:
        if value > 1200: color, emoji = "#FF4500", "💨";
        elif value > 800: color = "#FFA500"
    elif 'STP' in param_name:
        if value > 1: color, emoji = "#FF00FF", "🌪️";
        elif value > 0.5: color = "#FFA500"
    elif 'LapseRate' in param_name:
        if value > 7.0: color, emoji = "#FF4500", "🧊";
        elif value > 6.5: color = "#FFA500"
    elif 'PWAT' in param_name:
        if value > 40: color, emoji = "#0000FF", "💧";
        elif value > 30: color = "#4682B4"
    elif 'LCL' in param_name:
        if value < 1000: color, emoji = "#FFA500", "👇";
    elif 'EL' in param_name:
        if value > 14: color, emoji = "#FF4500", "🔝";
        elif value > 12: color = "#FFA500"
    return color, emoji

def generar_avis_temperatura(params):
    temp = params.get('SFC_Temp', {}).get('value')
    if temp is None: return None, None, None, None
    if temp > 36: return "AVÍS PER CALOR EXTREMA", f"Es preveu una temperatura de {temp:.1f}°C. Risc molt alt.", "#FF0000", "🥵"
    if temp < 0: return "AVÍS PER FRED INTENS", f"Es preveu una temperatura de {temp:.1f}°C. Risc de gelades fortes.", "#0000FF", "🥶"
    return None, None, None, None

def generar_avis_localitat(params, is_convergence_active):
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0); cin = params.get('CIN_Fre', {}).get('value')
    lfc_agl = params.get('LFC_AGL', {}).get('value'); shear = params.get('Shear_0-6km', {}).get('value')
    srh1 = params.get('SRH_0-1km', {}).get('value'); lcl_agl = params.get('LCL_AGL', {}).get('value', 9999)
    dcape = params.get('DCAPE', {}).get('value', 0); stp = params.get('STP_cin', {}).get('value', 0)
    lr = params.get('LapseRate_700_500', {}).get('value', 0); pwat = params.get('PWAT_Total', {}).get('value', 0)
    if cape_u < 100: return "ESTABLE", "Sense risc de tempestes significatives. L'atmosfera és estable.", "#3CB371"
    if cin is not None and cin < -150: return "ESTABLE", f"La 'tapa' atmosfèrica (CIN de {cin:.0f} J/kg) és massa forta i probablement inihibirà qualsevol convecció.", "#3CB371"
    if lfc_agl is None and not is_convergence_active: return "RISC BAIX", f"Hi ha una inestabilitat considerable (CAPE de {cape_u:.0f} J/kg), però el nivell d'inici de la convecció (LFC) és inassolible sense un forçament potent com la convergència. Les tempestes són poc probables.", "#4682B4"
    cond_supercelula = shear is not None and shear > 20 and cape_u > 1500 and srh1 is not None and srh1 > 200 and lcl_agl < 1300
    cond_avis_sever = shear is not None and shear > 18 and cape_u > 1200
    cond_precaucio = shear is not None and shear > 12 and cape_u > 500
    if cond_supercelula and stp > 0.8:
        risks = ["calamarsa grossa", "vent destructiu"];
        if lcl_agl < 1000: risks.insert(0, "tornados")
        risks_text = ", ".join(risks)
        if is_convergence_active or (lfc_agl is not None and lfc_agl < 3000): return "RISC ALT", f"Entorn de SUPERCL·LULA TORNÀDICA (STP={stp:.1f}). Disparador present. Risc de {risks_text}.", "#E60073"
        else: return "POTENCIAL SEVER", f"Potencial latent de SUPERCL·LULA TORNÀDICA (STP={stp:.1f}) a l'espera d'un disparador clar. L'entorn és perillós.", "#FFC300"
    elif cond_avis_sever:
        risks = [];
        if dcape > 900: risks.append("fortes ràfegues de vent (esclafits severs)")
        if lr is not None and lr > 6.8: risks.append("calamarsa de mida considerable")
        if pwat > 35: risks.append("pluges torrencials");
        elif not risks: risks.append("fortes pluges")
        risks_text = ", ".join(risks)
        if is_convergence_active or (lfc_agl is not None and lfc_agl < 3000): return "AVÍS", f"Potencial per a tempestes SEVERES organitzades. Disparador present. Risc de {risks_text}.", "#FF8C00"
        else: return "POTENCIAL MODERAT", f"Entorn de tempesta SEVERA latent a l'espera d'un disparador clar. Risc de {risks_text}.", "#FFD700"
    elif cond_precaucio:
        missatge = "Risc de TEMPESTES ORGANITZADES (multicèl·lules). Possibles fortes pluges i calamarsa local."
        if pwat > 35: missatge += " Atenció al risc de xàfecs torrencials."
        return "PRECAUCIÓ", missatge, "#FFD700"
    else:
        missatge = f"Hi ha inestabilitat (CAPE de {cape_u:.0f} J/kg) suficient per a xàfecs o tempestes febles i aïllades."
        if is_convergence_active: missatge += " La convergència forta pot ajudar a formar alguns nuclis."
        return "RISC BAIX", missatge, "#4682B4"

def generar_avis_convergencia(params, is_convergence_active, divergence_value):
    if not is_convergence_active: return None, None, None
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0); cin = params.get('CIN_Fre', {}).get('value')
    if cape_u > 1000 and (cin is None or cin > -50) and divergence_value is not None and divergence_value < -15:
        return "ALERTA DE DISPARADOR", f"La forta convergència de vents (valor: {divergence_value:.1f} x10⁻⁵ s⁻¹) pot actuar com a disparador. Amb un CAPE de {cape_u:.0f} J/kg i una 'tapa' (CIN) feble, hi ha un alt potencial que les tempestes s'iniciïn de manera explosiva.", "#FF4500"
    return None, None, None

def generar_analisi_detallada(params, is_convergence_active):
    def stream_text(text):
        for word in text.split(): yield word + " "; time.sleep(0.02)
        yield "\n\n"
    cape_u=params.get('CAPE_Utilitzable',{}).get('value',0); cin=params.get('CIN_Fre',{}).get('value'); shear6=params.get('Shear_0-6km',{}).get('value')
    srh1=params.get('SRH_0-1km',{}).get('value'); lcl_agl=params.get('LCL_AGL',{}).get('value'); w_max=params.get('W_MAX',{}).get('value',0)
    el_msl = params.get('EL_MSL', {}).get('value'); dcape=params.get('DCAPE',{}).get('value'); stp=params.get('STP_cin',{}).get('value')
    lr=params.get('LapseRate_700_500',{}).get('value'); pwat=params.get('PWAT_Total',{}).get('value')
    if cape_u is None or cape_u < 100: 
        yield from stream_text("### Anàlisi General\nL'atmosfera és estable. El potencial energètic per a la formació de tempestes (CAPE) és pràcticament inexistent. No s'esperen fenòmens de temps sever."); return
    yield from stream_text("### Potencial Energètic (Termodinàmica)")
    cape_text="feble" if cape_u<1000 else "moderada" if cape_u<2500 else "forta" if cape_u<4000 else "extrema"
    base_cape_text = f"**Inestabilitat (CAPE):** El valor de CAPE utilitzable és de **{cape_u:.0f} J/kg**, el que indica una inestabilitat **{cape_text}**."
    wmax_text = f"Això es tradueix en un potencial de corrents ascendents de fins a **{w_max*3.6:.0f} km/h**"
    if el_msl is not None:
        el_text = f", podent sostenir un núvol de tempesta fins a una altitud de **{el_msl:.1f} km**."; yield from stream_text(f"{base_cape_text} {wmax_text}{el_text}")
    else: yield from stream_text(f"{base_cape_text} {wmax_text}.")
    if cin is not None:
        cin_text = f"**Inhibició (CIN) i Disparador:** El valor de CIN és de **{cin:.0f} J/kg**. "
        if cin >= -25: cin_text += "Aquesta 'tapa' atmosfèrica és molt feble o inexistent. "; cin_text += "La convergència de vents detectada actuarà com un disparador molt efectiu, iniciant la convecció amb facilitat." if is_convergence_active else "Fins i tot un lleuger escalfament diürn o un petit accident orogràfic podria ser suficient per iniciar tempestes."
        elif -75 < cin <= -25: cin_text += "Aquesta 'tapa' moderada impedeix que les tempestes es formin espontàniament. "; cin_text += "Aquí, **la convergència detectada és el factor clau**: té el potencial de trencar la 'tapa' i alliberar la inestabilitat de manera explosiva." if is_convergence_active else "Sense un mecanisme de forçament clar com la convergència, es necessitarà un fort escalfament o un xoc orogràfic significatiu per disparar les tempestes."
        else: cin_text += "Aquesta 'tapa' és forta. És molt poc probable que es formin tempestes."
        yield from stream_text(cin_text)
    yield from stream_text("### Organització i Rotació (Cinemàtica)")
    if shear6 is not None:
        shear_text = "Molt feble (tempestes desorganitzades)." if shear6 < 10 else "Moderat (multicèl·lules)." if shear6 < 18 else "Fort (supercèl·lules)."; yield from stream_text(f"**Cisallament 0-6 km:** El valor és de **{shear6:.1f} m/s**. {shear_text}")
    if srh1 is not None and srh1 > 100:
        srh_text="moderat" if srh1<250 else "fort"; lcl_text = f"Amb una base de núvol baixa ({lcl_agl:.0f} m), " if lcl_agl is not None and lcl_agl < 1200 else ""
        yield from stream_text(f"**Helicitat 0-1 km (SRH):** Rotació potencial **{srh_text}** ({srh1:.0f} m²/s²). {lcl_text}Augmenta el risc de tornados.")
    yield from stream_text("### Síntesi i Riscos Específics")
    if stp and stp > 0.5: yield from stream_text(f"**Potencial Tornàdic (STP):** L'índex és de **{stp:.1f}**. Valors > 1 indiquen entorn favorable a supercèl·lules tornàdiques.")
    if dcape and dcape > 800: yield from stream_text(f"**Potencial de Vent Sever (DCAPE):** El valor de **{dcape:.0f} J/kg** suggereix potencial per a esclafits.")
    if pwat and pwat > 30: yield from stream_text(f"**Potencial de Precipitació Intensa (Aigua Precipitable):** Contingut d'humitat elevat ({pwat:.1f} mm), afavorint xàfecs intensos.")
    yield from stream_text(f"**La Clau del Pronòstic:** La interacció entre la inestabilitat **{cape_text}** i el cisallament **{('fort' if (shear6 or 0) > 18 else 'moderat' if (shear6 or 0) > 10 else 'feble')}**. La **convergència** serà el factor decisiu.")

def display_metrics(params_dict):
    param_map = [('Temperatura','SFC_Temp'), ('CAPE Utilitzable','CAPE_Utilitzable'), ('CIN (Fre)','CIN_Fre'), ('Vel. Asc. Màx.','W_MAX'),('Shear 0-6km','Shear_0-6km'), ('SRH 0-1km','SRH_0-1km'), ('Potencial Tornàdic','STP_cin'), ('Potencial Esclafits','DCAPE'),('Gradient Tèrmic','LapseRate_700_500'), ('Aigua Precipitable','PWAT_Total'), ('Base núvol (AGL)','LCL_AGL'), ('Nivell Convecció (AGL)', 'LFC_AGL'), ('Cim tempesta (MSL)','EL_MSL')]
    st.markdown("""<style>.metric-container{border:1px solid rgba(128,128,128,0.2);border-radius:10px;padding:10px;margin-bottom:10px;}</style>""", unsafe_allow_html=True); cols=st.columns(4); col_idx = 0
    for label, key in param_map:
        param = params_dict.get(key); val_str, units_str, value_color, emoji, border_color = "---", "", 'gray', '', 'rgba(128,128,128,0.2)'
        if param and param.get('value') is not None:
            value = param['value']; units_str = param.get('units', ''); val_str = f"{value:.1f}" if isinstance(value, (float, np.floating)) else str(value)
            value_color, emoji = get_parameter_style(key, value); border_color = value_color if value_color != 'inherit' else 'rgba(128,128,128,0.2)'
        with cols[col_idx % 4]:
            st.markdown(f"""<div class="metric-container" style="border-color:{border_color};"><div style="font-size:0.9em;color:gray;">{label}</div><div style="font-size:1.25em;font-weight:bold;color:{value_color};">{val_str} <span style='font-size:0.8em;color:gray;'>{units_str}</span> {emoji}</div></div>""", unsafe_allow_html=True)
        col_idx += 1

def crear_mapa_base(nivell, lat_sel, lon_sel, nom_poble_sel, titol):
    fig,ax=plt.subplots(figsize=(9,9),dpi=150,subplot_kw={'projection': ccrs.PlateCarree()}); ax.set_extent([0,3.5,40.4,43],crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,facecolor="#E0E0E0",zorder=0); ax.add_feature(cfeature.OCEAN,facecolor='#b0c4de',zorder=0)
    ax.add_feature(cfeature.COASTLINE,edgecolor='black',linewidth=0.7,zorder=5); ax.add_feature(cfeature.BORDERS,linestyle=':',edgecolor='black',zorder=5)
    ax.plot(lon_sel,lat_sel,'o',markersize=6,markerfacecolor='yellow',markeredgecolor='black',markeredgewidth=2,transform=ccrs.Geodetic(),zorder=6)
    ax.text(lon_sel+0.05,lat_sel+0.05,nom_poble_sel,transform=ccrs.Geodetic(),zorder=7,bbox=dict(facecolor='white',alpha=0.8,edgecolor='none',boxstyle='round,pad=0.2'))
    ax.set_title(f"{titol} a {nivell}hPa",weight='bold'); return fig,ax

def crear_mapa_cape(lats, lons, data, lat_sel, lon_sel, nom_poble_sel):
    titol_var = "CAPE (Energia per a tempestes)"; unitat = "J/kg"; cmap = 'plasma'; levels = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()}); ax.set_extent([0, 3.5, 40.4, 43], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0); ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.7, zorder=5); ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', zorder=5)
    ax.plot(lon_sel, lat_sel, 'o', markersize=6, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2, transform=ccrs.Geodetic(), zorder=6)
    ax.text(lon_sel + 0.05, lat_sel + 0.05, nom_poble_sel, transform=ccrs.Geodetic(), zorder=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')); ax.set_title(titol_var, weight='bold')
    grid_lon, grid_lat = np.linspace(min(lons), max(lons), 100), np.linspace(min(lats), max(lats), 100); X, Y = np.meshgrid(grid_lon, grid_lat); points = np.vstack((lons, lats)).T
    grid_data = griddata(points, data, (X, Y), method='cubic'); grid_data = np.nan_to_num(grid_data)
    cont = ax.contourf(X, Y, grid_data, cmap=cmap, levels=levels, alpha=0.7, zorder=2, transform=ccrs.PlateCarree(), extend='max')
    fig.colorbar(cont, ax=ax, orientation='vertical', label=f'{titol_var} ({unitat})', shrink=0.7); return fig


def crear_mapa_vents(lats, lons, data, nivell, lat_sel, lon_sel, nom_poble_sel):
    # 1. Preparació de dades
    speeds_kmh, dirs_deg_raw = zip(*data)
    speeds_ms = (np.array(speeds_kmh) * 1000 / 3600) * units('m/s')
    dirs_deg = np.array(dirs_deg_raw) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    
    # 2. Creació de la graella i interpolació SUAU (RBF)
    grid_lon, grid_lat = np.linspace(min(lons), max(lons), 150), np.linspace(min(lats), max(lats), 150)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    
    rbf_u = Rbf(lons, lats, u_comp.m, function='thin_plate', smooth=0)
    rbf_v = Rbf(lons, lats, v_comp.m, function='thin_plate', smooth=0)
    u_grid = rbf_u(X, Y)
    v_grid = rbf_v(X, Y)
    
    speed_grid = np.sqrt(u_grid**2 + v_grid**2) * 3.6 # Convertim a km/h

    # 3. Càlcul de la convergència
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y)
    divergence = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5
    
    max_conv = np.min(divergence.m)
    titol_mapa = f"Flux i Convergència (Màx: {max_conv:.1f})"
    
    fig, ax = crear_mapa_base(nivell, lat_sel, lon_sel, nom_poble_sel, titol_mapa)

    # --- VISUALITZACIÓ AVANÇADA ---
    
    # --- CANVI CLAU: Invertim els colors ---
    # 4. Paleta de colors: 'coolwarm'. Aquesta paleta per defecte posa el vermell
    #    als valors negatius (convergència) i el blau als positius (divergència).
    cmap_conv_div = 'coolwarm' 
    max_abs_val = 30 
    levels = np.linspace(-max_abs_val, max_abs_val, 15)
    
    cont_fill = ax.contourf(X, Y, divergence.m, levels=levels, cmap=cmap_conv_div, alpha=0.6, zorder=2, transform=ccrs.PlateCarree(), extend='both')
    fig.colorbar(cont_fill, ax=ax, orientation='vertical', label='Convergència (vermell) / Divergència (blau) (x10⁻⁵ s⁻¹)', shrink=0.7)

    # Dibuixem un contorn negre per als focus de convergència
    ax.contour(X, Y, divergence.m, 
               levels=[-35, -25, -15], 
               colors='black', 
               linewidths=[1.5, 1, 0.5],
               alpha=0.5, 
               zorder=3, 
               transform=ccrs.PlateCarree())

    # 5. Línies de flux acolorides per velocitat
    stream = ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color=speed_grid, cmap='viridis', 
                           linewidth=1, density=5.5, arrowsize=0.4, zorder=4, transform=ccrs.PlateCarree())
    
    cbar_stream = fig.colorbar(stream.lines, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.6)
    cbar_stream.set_label('Velocitat del Vent (km/h)')

    return fig

def crear_mapa_generic(lats, lons, data, nivell, lat_sel, lon_sel, nom_poble_sel, titol_var, cmap, unitat, levels):
    fig,ax=crear_mapa_base(nivell,lat_sel,lon_sel,nom_poble_sel,titol_var); grid_lon,grid_lat=np.linspace(min(lons),max(lons),100),np.linspace(min(lats),max(lats),100)
    X,Y=np.meshgrid(grid_lon,grid_lat); points=np.vstack((lons,lats)).T; grid_data=griddata(points,data,(X,Y),method='cubic'); grid_data=np.nan_to_num(grid_data)
    cont=ax.contourf(X,Y,grid_data,cmap=cmap,levels=levels,alpha=0.7,zorder=2,transform=ccrs.PlateCarree(),extend='both'); fig.colorbar(cont,ax=ax,orientation='vertical',label=f'{titol_var} ({unitat})',shrink=0.7); return fig

def crear_mapa_temp_isobares(lats, lons, data, nivell, lat_sel, lon_sel, nom_poble_sel):
    temps, heights = zip(*data); fig,ax = crear_mapa_base(nivell, lat_sel, lon_sel, nom_poble_sel, "Temperatura i Isobares")
    grid_lon,grid_lat=np.linspace(min(lons),max(lons),100),np.linspace(min(lats),max(lats),100); X,Y=np.meshgrid(grid_lon,grid_lat); points=np.vstack((lons,lats)).T
    grid_temp = griddata(points, temps, (X,Y), method='cubic'); grid_temp = np.nan_to_num(grid_temp); grid_height = griddata(points, heights, (X,Y), method='cubic'); grid_height = np.nan_to_num(grid_height)
    cont_temp = ax.contourf(X,Y,grid_temp, cmap='coolwarm', levels=20, alpha=0.7, zorder=2, transform=ccrs.PlateCarree())
    cont_height = ax.contour(X,Y,grid_height, colors='black', linewidths=0.8, alpha=0.9, zorder=3, transform=ccrs.PlateCarree()); plt.clabel(cont_height, inline=True, fontsize=9, fmt='%1.0f m')
    fig.colorbar(cont_temp, ax=ax, orientation='vertical', label='Temperatura (°C)', shrink=0.7); return fig

def crear_hodograf(p, u, v, h):
    fig, ax=plt.subplots(1,1,figsize=(5,5)); hodo=Hodograph(ax,component_range=40.); hodo.add_grid(increment=10)
    hodoline=hodo.plot_colormapped(u.to('kt'),v.to('kt'),h.to('km'),cmap='gist_ncar'); plt.colorbar(hodoline,ax=ax,orientation='vertical',pad=0.05,shrink=0.8).set_label('Altitud (km)')
    try: rm,_,_=mpcalc.bunkers_storm_motion(p,u,v,h); hodo.plot_vectors(rm[0].to('kt'),rm[1].to('kt'),color='black',label='Mov. Tempesta (RM)'); ax.legend()
    except: pass
    ax.set_xlabel('kt'); ax.set_ylabel('kt'); return fig

def crear_skewt(p, T, Td, u, v, H, params):
    fig = plt.figure(figsize=(7, 9)); skew = SkewT(fig, rotation=45); skew.plot(p, T, 'r', lw=2, label='Temperatura'); skew.plot(p, Td, 'b', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), length=7, color='black'); skew.plot_dry_adiabats(color='lightcoral', ls='--', alpha=0.5); skew.plot_moist_adiabats(color='cornflowerblue', ls='--', alpha=0.5)
    skew.plot_mixing_lines(color='lightgreen', ls='--', alpha=0.5); skew.ax.axvline(0, color='darkturquoise', linestyle='--', label='Isoterma 0°C')
    cape = params.get('CAPE_Brut', {}).get('value', 0); cin = params.get('CIN_Fre', {}).get('value', 0)
    if cape > 0 or (cin is not None and cin < 0):
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', lw=2, ls='--', label='Trajectòria Parcel·la')
            skew.shade_cape(p, T, prof, alpha=0.3, color='orange'); skew.shade_cin(p, T, prof, alpha=0.6, color='gray')
        except Exception: pass
    lcl_agl = params.get('LCL_AGL', {}).get('value'); lfc_agl = params.get('LFC_AGL', {}).get('value'); el_msl = params.get('EL_MSL', {}).get('value')
    try:
        if lcl_agl is not None: lcl_p_val = np.interp(lcl_agl + H.m[0], H.m, p.m)
        else: lcl_p_val = None
        if lfc_agl is not None: lfc_p_val = np.interp(lfc_agl + H.m[0], H.m, p.m)
        else: lfc_p_val = None
        if el_msl is not None: el_p_val = np.interp(el_msl * 1000, H.m, p.m)
        else: el_p_val = None
        if lcl_p_val and lfc_p_val and abs(lcl_p_val - lfc_p_val) < 10:
            skew.ax.axhline(lcl_p_val, color='purple', linestyle='--', label=f'LCL / LFC {lcl_p_val:.0f} hPa')
        else:
            if lcl_p_val: skew.ax.axhline(lcl_p_val, color='purple', linestyle='--', label=f'LCL {lcl_p_val:.0f} hPa')
            if lfc_p_val: skew.ax.axhline(lfc_p_val, color='darkred', linestyle='--', label=f'LFC {lfc_p_val:.0f} hPa')
        if el_p_val: skew.ax.axhline(el_p_val, color='red', linestyle='--', label=f'EL {el_p_val:.0f} hPa')
    except Exception: pass
    skew.ax.set_ylim(1050, 100); skew.ax.set_xlim(-40, 40); skew.ax.set_xlabel('Temperatura (°C)'); skew.ax.set_ylabel('Pressió (hPa)'); plt.legend(); return fig

def crear_grafic_orografia(params, zero_iso_h_agl):
    lcl_agl = params.get('LCL_AGL', {}).get('value'); lfc_agl = params.get('LFC_AGL', {}).get('value')
    if lcl_agl is None or np.isnan(lcl_agl): return None
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150); fig.patch.set_alpha(0); ax.set_xlim(0, 10); ax.set_ylabel(""); ax.tick_params(axis='y', labelleft=False, length=5, color='white'); ax.spines['left'].set_color('white'); ax.spines['bottom'].set_color('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.set_xticklabels([]); ax.set_xticks([])
    sky_gradient = np.linspace(0.6, 1.0, 256).reshape(-1, 1); sky_gradient = np.vstack((sky_gradient, sky_gradient)); ax.imshow(sky_gradient, aspect='auto', cmap='Blues_r', extent=[0, 10, 0, 14], zorder=0)
    has_lfc = lfc_agl is not None and np.isfinite(lfc_agl); peak_h_m = lfc_agl if has_lfc and lfc_agl > 1000 else (lcl_agl * 1.5 if lcl_agl > 1000 else 2000); peak_h_km = min(peak_h_m / 1000.0, 8.0)
    x_mountain = np.linspace(0, 10, 200); y_mountain = peak_h_km * (1 - np.cos(x_mountain * np.pi / 10)) / 2 * (1 - (x_mountain - 5)**2 / 25)
    ax.add_patch(Polygon(np.vstack([x_mountain, y_mountain]).T, facecolor='#6B8E23', edgecolor='#3A4D14', lw=2, zorder=2)); line_outline_effect = [path_effects.withStroke(linewidth=3.5, foreground='black')]; lcl_km = lcl_agl / 1000
    ax.axhline(lcl_km, color='white', linestyle='--', lw=2, zorder=3, path_effects=line_outline_effect); ax.text(9.9, lcl_km, f"Base del núvol (LCL): {lcl_agl:.0f} m ", color='black', backgroundcolor='white', ha='right', va='center', weight='bold')
    ax.add_patch(patches.Rectangle((0, lcl_km), 10, 0.2, facecolor='white', alpha=0.4, zorder=1))
    if has_lfc: lfc_km = lfc_agl / 1000; ax.axhline(lfc_km, color='#FFD700', linestyle='--', lw=2.5, zorder=3, path_effects=line_outline_effect); ax.text(0.1, lfc_km, f" Disparador de tempesta (LFC): {lfc_agl:.0f} m", color='black', backgroundcolor='#FFD700', ha='left', va='center', weight='bold')
    main_text_effect = [path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()]; mountain_lfc_intersect_x = np.interp(lfc_km if has_lfc else peak_h_km, y_mountain, x_mountain)
    ax.annotate("", xy=(mountain_lfc_intersect_x, lfc_km if has_lfc else peak_h_km), xytext=(1, 0.1), arrowprops=dict(arrowstyle="->", color='white', lw=2.5, linestyle=':', connectionstyle="arc3,rad=0.2"), zorder=4)
    flux_text = ax.text(1.5, 0.4, "Flux d'aire\nforçat a pujar", color='white', weight='bold', ha='center', fontsize=11); flux_text.set_path_effects(main_text_effect)
    if has_lfc:
        analysis_text = None
        if peak_h_m >= lfc_agl: ax.plot(mountain_lfc_intersect_x, lfc_km, '*', markersize=20, color='yellow', markeredgecolor='red', zorder=5); analysis_text = ax.text(5, max(peak_h_km, lfc_km) + 1, f"DISPARADOR OROGRÀFIC POTENCIAL!\nUna muntanya de {lfc_agl:.0f} m o més seria suficient.", ha='center', va='center', fontsize=14, weight='bold', color='white')
        else: ax.plot(x_mountain[np.argmax(y_mountain)], peak_h_km, 'X', markersize=15, color='red', markeredgecolor='white', zorder=5); analysis_text = ax.text(5, peak_h_km + 1, f"L'OROGRAFIA NO ÉS SUFICIENT.\nEs necessita una muntanya de {lfc_agl:.0f} m per disparar la tempesta.", ha='center', va='center', color='yellow', weight='bold', fontsize=12)
        if analysis_text: analysis_text.set_path_effects(main_text_effect)
    else: analysis_text = ax.text(5, 4, "No hi ha un LFC accessible.\nL'atmosfera és massa estable.", ha='center', va='center', color='lightblue', weight='bold', fontsize=12); analysis_text.set_path_effects(main_text_effect)
    final_ylim = max(peak_h_km, (lfc_km if has_lfc else 0)) + 2.0; ax.set_ylim(0, final_ylim)
    y_label_text = ax.text(0.2, final_ylim - 0.2, "Altitud (km)", ha='left', va='top', color='white', weight='bold', fontsize=12); y_label_text.set_path_effects(main_text_effect)
    for y_tick in ax.get_yticks():
        if y_tick > 0 and y_tick < final_ylim: tick_label = ax.text(0.15, y_tick, f'{int(y_tick)}', ha='left', va='center', color='white', weight='bold', fontsize=9); tick_label.set_path_effects(main_text_effect)
    fig.tight_layout(pad=0.5); return fig

def crear_grafic_nuvol(params, H, u, v, is_convergence_active):
    lcl_agl, el_msl_km, cape = (params.get(k, {}).get('value') for k in ['LCL_AGL', 'EL_MSL', 'CAPE_Brut'])
    if lcl_agl is None or el_msl_km is None: return None
    cape = cape or 0; fig, ax = plt.subplots(figsize=(6, 9), dpi=120); ax.set_facecolor('#4F94CD'); lcl_km = lcl_agl / 1000; el_km = el_msl_km; center_x_base = 5.0
    if is_convergence_active and cape > 100:
        y_points = np.linspace(lcl_km, el_km, 100); cloud_width = 1.0 + np.sin(np.pi * (y_points - lcl_km) / (el_km - lcl_km)) * (1 + cape / 2000)
        for y, width in zip(y_points, cloud_width):
            tilt_offset = np.interp(y * 1000, H.m, u.m) / 15; center_x = center_x_base + tilt_offset
            for _ in range(25): ax.add_patch(Circle((center_x + (random.random() - 0.5) * width, y + (random.random() - 0.5) * 0.4), 0.2 + random.random() * 0.4, color='white', alpha=0.15, lw=0))
        top_cloud_tilt_offset = np.interp(el_km * 1000, H.m, u.m) / 15; anvil_center_x = center_x_base + top_cloud_tilt_offset
        anvil_wind_spread = np.interp(el_km * 1000, H.m, u.m) / 10
        for _ in range(80): ax.add_patch(Circle((anvil_center_x + (random.random() - 0.2) * 4 + anvil_wind_spread, el_km + (random.random() - 0.5) * 0.5), 0.2 + random.random() * 0.6, color='white', alpha=0.2, lw=0))
        if cape > 2500: ax.add_patch(Circle((anvil_center_x, el_km + cape / 5000), 0.4, color='white', alpha=0.5))
    else: ax.text(center_x_base, 8, "Sense disparador o energia\nsuficient per a convecció profunda.", ha='center', va='center', color='black', fontsize=16, weight='bold', bbox=dict(facecolor='lightblue', alpha=0.7, boxstyle='round,pad=0.5'))
    barb_heights_km = np.arange(1, 16, 1); u_barbs = np.interp(barb_heights_km * 1000, H.m, u.to('kt').m); v_barbs = np.interp(barb_heights_km * 1000, H.m, v.to('kt').m)
    ax.barbs(np.full_like(barb_heights_km, 9.5), barb_heights_km, u_barbs, v_barbs, length=7, color='black')
    ax.set_ylim(0, 16); ax.set_xlim(0, 10); ax.set_ylabel("Altitud (km, MSL)"); ax.set_title("Visualització del Núvol", weight='bold'); ax.set_xticks([]); ax.grid(axis='y', linestyle='--', alpha=0.3); return fig

# --- 3. INTERFÍCIE I FLUX PRINCIPAL DE L'APLICACIÓ ---
st.markdown('<h1 style="text-align: center; color: #FF4B4B;">⚡ Tempestes.cat</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Eina d\'Anàlisi i Previsió de Fenòmens Severs a Catalunya</p>', unsafe_allow_html=True)
hourly_index_global = int(st.session_state.hora_selector.split(':')[0])
with st.spinner("Calculant convergències per a tot el territori..."):
    convergencies_850hpa = calcular_convergencia_per_totes_les_localitats(hourly_index_global, 850, pobles_data)
    localitats_convergencia_forta = {p for p, v in convergencies_850hpa.items() if v is not None and v < CONVERGENCIA_FORTA_THRESHOLD}
with st.container(border=True):
    col1, col2 = st.columns([1,1], gap="large");
    with col1: st.selectbox("Hora del pronòstic (Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")
    with col2:
        sorted_pobles = sorted(pobles_data.keys()); opciones_display = [f"⚠️ {p}" if p in localitats_convergencia_forta else p for p in sorted_pobles]
        try:
            poble_actual_net = st.session_state.poble_selector.replace("⚠️ ", "").strip(); poble_actual_display = f"⚠️ {poble_actual_net}" if poble_actual_net in localitats_convergencia_forta else poble_actual_net
            default_index = opciones_display.index(poble_actual_display)
        except (ValueError, KeyError): default_index = sorted_pobles.index(st.session_state.poble_selector) if st.session_state.poble_selector in sorted_pobles else 0
        selected_option_raw = st.selectbox('Selecciona una localitat:', options=opciones_display, index=default_index)
        st.session_state.poble_selector = selected_option_raw.replace("⚠️ ", "").strip()
st.markdown(f'<p style="text-align: center; font-size: 0.9em; color: grey;">🕒 {get_next_arome_update_time()}</p>', unsafe_allow_html=True)
hourly_index = int(st.session_state.hora_selector.split(':')[0]); poble_sel = st.session_state.poble_selector
lat_sel = pobles_data[poble_sel]['lat']; lon_sel = pobles_data[poble_sel]['lon']
with st.spinner("Analitzant potencials diaris..."): potencials_detectats_avui = precalcular_potencials_del_dia(pobles_data)
st.toast("Anàlisi del territori completat.")
with st.expander("⚡️ Avisos i Potencials Detectats Avui", expanded=st.session_state.avisos_expanded):
    if not potencials_detectats_avui: st.success("No s'ha detectat cap risc significatiu de temps sever per a les pròximes 24 hores.")
    else:
        hores_a_revisar = sorted(list(set(potencials_detectats_avui.values()))); convergencies_per_hora = {}
        for hora in hores_a_revisar:
            convergencies_per_hora[hora] = calcular_convergencia_per_totes_les_localitats(hora, 850, pobles_data); time.sleep(0.2)
        pobles_amb_disparador, pobles_amb_potencial = [], []
        for poble, hora in potencials_detectats_avui.items():
            valor_conv_poble = convergencies_per_hora.get(hora, {}).get(poble)
            if valor_conv_poble is not None and valor_conv_poble < CONVERGENCIA_FORTA_THRESHOLD: pobles_amb_disparador.append((poble, hora))
            else: pobles_amb_potencial.append((poble, hora))
        col_disparador, col_potencial = st.columns(2)
        with col_disparador:
            st.markdown("##### 🔥 Risc actiu (Amb disparador)");
            if not pobles_amb_disparador: st.write("Cap localitat amb potencial i disparador fort detectat.")
            for poble, hora in pobles_amb_disparador: st.button(f"{poble} (a les {hora:02d}:00h)", key=f"btn_disparador_{poble}", on_click=actualitzar_seleccio, args=(poble, hora))
        with col_potencial:
            st.markdown("##### 🔎 Potencial latent (Sense disparador clar)");
            if not pobles_amb_potencial: st.write("Totes les localitats amb potencial tenen un disparador fort.")
            for poble, hora in pobles_amb_potencial: st.button(f"{poble} (a les {hora:02d}:00h)", key=f"btn_potencial_{poble}", on_click=actualitzar_seleccio, args=(poble, hora))
with st.spinner(f"Carregant sondeig detallat per a {poble_sel}..."): sondeo, p_levels, error_sondeo = carregar_sondeig_per_poble(poble_sel, lat_sel, lon_sel)
if error_sondeo:
    st.error(f"L'API d'Open-Meteo ha retornat un error per a '{poble_sel}'.");
    with st.expander("Veure error tècnic"): st.code(error_sondeo)
elif sondeo:
    try:
        profiles = processar_sondeig_per_hora(sondeo, hourly_index, p_levels)
        if profiles:
            p, T, Td, u, v, H = profiles; parametros = calculate_parameters(p, T, Td, u, v, H)
            is_disparador_active = poble_sel in localitats_convergencia_forta; divergence_value_local = convergencies_850hpa.get(poble_sel)
            avis_temp_titol, avis_temp_text, avis_temp_color, avis_temp_icona = generar_avis_temperatura(parametros)
            if avis_temp_titol: display_avis_principal(avis_temp_titol, avis_temp_text, avis_temp_color, icona_personalitzada=avis_temp_icona)
            avis_conv_titol, avis_conv_text, avis_conv_color = generar_avis_convergencia(parametros, is_disparador_active, divergence_value_local)
            if avis_conv_titol: display_avis_principal(avis_conv_titol, avis_conv_text, avis_conv_color)
            avis_titol, avis_text, avis_color = generar_avis_localitat(parametros, is_disparador_active)
            display_avis_principal(avis_titol, avis_text, avis_color)
            tab_analisi, tab_params, tab_mapes, tab_hodo, tab_sondeig, tab_oro, tab_nuvol, tab_focus = st.tabs(["🗨️ Anàlisi", "📊 Paràmetres", "🗺️ Mapes", "🧭 Hodògraf","📍 Sondeig", "🏔️ Orografia", "☁️ Visualització", "🎯 Focus Convergència"])
            with tab_analisi: st.write_stream(generar_analisi_detallada(parametros, is_disparador_active))
            with tab_params: st.subheader("Paràmetres Clau"); display_metrics(parametros)
            with tab_mapes:
                st.subheader("Anàlisi de Mapes"); col1, col2 = st.columns(2)
                with col1:
                    p_levels_all = [1000, 925, 850, 700, 500, 300]; nivell_global = st.selectbox("Nivell:", p_levels_all, index=p_levels_all.index(st.session_state.nivell_mapa), key="nivell_mapa_selector")
                    st.session_state.nivell_mapa = nivell_global
                with col2:
                    map_options = {"Vents i Convergència": "wind", "CAPE (Energia)": "cape", "Temperatura i Isobares": "temp_height", "Punt de Rosada": "dewpoint", "Humitat Relativa": "humidity"}
                    selected_map_name = st.selectbox("Tipus de mapa:", map_options.keys())
                with st.spinner(f"Generant mapa de {selected_map_name.lower()}..."):
                    api_var = map_options[selected_map_name]; lats, lons, data, error = obtener_dades_mapa(api_var, nivell_global, hourly_index, FORECAST_DAYS)
                    if error: st.error(f"Error en obtenir dades del mapa: {error}")
                    elif not lats or len(lats) < 4: st.warning("No hi ha prou dades per generar el mapa.")
                    else:
                        fig = None
                        if selected_map_name == "Vents i Convergència": fig = crear_mapa_vents(lats, lons, data, nivell_global, lat_sel, lon_sel, poble_sel)
                        elif selected_map_name == "CAPE (Energia)": fig = crear_mapa_cape(lats, lons, data, lat_sel, lon_sel, poble_sel)
                        elif selected_map_name == "Temperatura i Isobares": fig = crear_mapa_temp_isobares(lats, lons, data, nivell_global, lat_sel, lon_sel, poble_sel)
                        else:
                            map_configs = {"Punt de Rosada": {"titol_var": "Punt de Rosada", "cmap": "BrBG", "unitat": "°C", "levels": np.arange(-10, 21, 2)}, "Humitat Relativa": {"titol_var": "Humitat Relativa", "cmap": "Greens", "unitat": "%", "levels": np.arange(30, 101, 5)}}
                            config = map_configs[selected_map_name]; fig = crear_mapa_generic(lats, lons, data, nivell_global, lat_sel, lon_sel, poble_sel, **config)
                        if fig: st.pyplot(fig)
            with tab_hodo: st.subheader("Hodògraf"); st.pyplot(crear_hodograf(p, u, v, H))
            with tab_sondeig: st.subheader(f"Sondeig per a {poble_sel}"); st.pyplot(crear_skewt(p, T, Td, u, v, H, parametros))
            with tab_oro: 
                st.subheader("Potencial d'Activació per Orografia"); fig_oro = crear_grafic_orografia(parametros, parametros.get('ZeroIso_AGL', {}).get('value'))
                if fig_oro: st.pyplot(fig_oro)
                else: st.info("No hi ha dades de LCL disponibles per calcular el potencial orogràfic.")
            with tab_nuvol:
                st.subheader("Visualització Conceptual del Núvol"); fig_nuvol = crear_grafic_nuvol(parametros, H, u, v, is_disparador_active)
                if fig_nuvol: st.pyplot(fig_nuvol)
                else: st.info("No hi ha dades per visualitzar l'estructura del núvol.")
            with tab_focus:
                st.subheader("Anàlisi del Disparador de Convergència");
                if is_disparador_active:
                    st.success(f"**Convergència FORTA detectada a {poble_sel}!**");
                    if divergence_value_local: st.metric("Valor de Convergència local (850hPa)", f"{divergence_value_local:.2f} x10⁻⁵ s⁻¹", help="Valors molt negatius indiquen convergència forta.")
                    st.markdown("Aquesta zona té un focus de convergència actiu que pot actuar com a **disparador** per a les tempestes.")
                else:
                    st.info(f"Cap focus de convergència significatiu detectat a {poble_sel} per a l'hora seleccionada.");
                    if divergence_value_local: st.metric("Valor de Convergència/Divergència local (850hPa)", f"{divergence_value_local:.2f} x10⁻⁵ s⁻¹")
                    st.markdown("L'absència d'un disparador clar pot dificultar la formació de tempestes.")
                st.markdown("---"); st.markdown("**Altres localitats amb convergència forta a aquesta hora:**")
                if localitats_convergencia_forta: st.markdown(f"_{', '.join(sorted(list(localitats_convergencia_forta)))}_")
                else: st.markdown("*Cap altra localitat amb avís de convergència forta per a aquesta hora.*")
        else:
            st.warning(f"No s'han pogut calcular els paràmetres per a les {hourly_index:02d}:00h. Dades invàlides.")
    except Exception as e:
        st.error(f"S'ha produït un error inesperat en processar les dades per a '{poble_sel}'."); st.exception(e)
