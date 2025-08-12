# -*- coding: utf-8 -*-
import streamlit as st
import openmeteo_requests
import requests_cache
from retry_requests import retry
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

# --- CONFIGURACIÓ INICIAL ---
st.set_page_config(layout="wide", page_title="Tempestes.cat")

cache_session = requests_cache.CachedSession('.cache', expire_after=18000)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

FORECAST_DAYS = 1

# --- DADES DE LOCALITATS ---
pobles_data = {
    'Abella de la Conca': {'lat': 42.163, 'lon': 1.092}, 'Abrera': {'lat': 41.517, 'lon': 1.901}, 'Àger': {'lat': 42.002, 'lon': 0.763},
    'Agramunt': {'lat': 41.784, 'lon': 1.096}, 'Aguilar de Segarra': {'lat': 41.737, 'lon': 1.626}, 'Agullana': {'lat': 42.395, 'lon': 2.846},
    'Aiguafreda': {'lat': 41.768, 'lon': 2.251}, 'Aiguamúrcia': {'lat': 41.332, 'lon': 1.359}, 'Aiguaviva': {'lat': 41.936, 'lon': 2.766},
    'Aitona': {'lat': 41.498, 'lon': 0.457}, 'Alàs i Cerc': {'lat': 42.358, 'lon': 1.488}, 'Albagés, L\'': {'lat': 41.428, 'lon': 0.732},
    'Albanyà': {'lat': 42.306, 'lon': 2.720}, 'Albatàrrec': {'lat': 41.564, 'lon': 0.601}, 'Albesa': {'lat': 41.751, 'lon': 0.672},
    'Albi, L\'': {'lat': 41.422, 'lon': 0.940}, 'Albinyana': {'lat': 41.242, 'lon': 1.484}, 'Albiol, L\'': {'lat': 41.258, 'lon': 1.109},
    'Albons': {'lat': 42.106, 'lon': 3.079}, 'Alcanar': {'lat': 40.544, 'lon': 0.481}, 'Alcanó': {'lat': 41.488, 'lon': 0.598},
    'Alcarràs': {'lat': 41.562, 'lon': 0.525}, 'Alcoletge': {'lat': 41.644, 'lon': 0.697}, 'Alcover': {'lat': 41.263, 'lon': 1.171},
    'Aldea, L\'': {'lat': 40.748, 'lon': 0.602}, 'Aldover': {'lat': 40.893, 'lon': 0.505}, 'Aleixar, L\'': {'lat': 41.211, 'lon': 1.060},
    'Alella': {'lat': 41.494, 'lon': 2.295}, 'Alfara de Carles': {'lat': 40.889, 'lon': 0.400}, 'Alfarràs': {'lat': 41.829, 'lon': 0.573},
    'Vilanova i la Geltrú': {'lat': 41.224, 'lon': 1.725}, 'Vilassar de Dalt': {'lat': 41.517, 'lon': 2.358}, 'Vilassar de Mar': {'lat': 41.506, 'lon': 2.392},
    'Valls': {'lat': 41.286, 'lon': 1.250}, 'Vic': {'lat': 41.930, 'lon': 2.255}, 'Vielha e Mijaran': {'lat': 42.702, 'lon': 0.796}, 'Vila-seca': {'lat': 41.111, 'lon': 1.144},
    'Badalona': {'lat': 41.450, 'lon': 2.247}, 'Balaguer': {'lat': 41.790, 'lon': 0.810}, 'Banyoles': {'lat': 42.119, 'lon': 2.766}, 'Barcelona': {'lat': 41.387, 'lon': 2.168},
    'Berga': {'lat': 42.103, 'lon': 1.845}, 'Besalú': {'lat': 42.199, 'lon': 2.698}, 'Blanes': {'lat': 41.674, 'lon': 2.793}, 'Borges Blanques, Les': {'lat': 41.522, 'lon': 0.869},
    'Cadaqués': {'lat': 42.288, 'lon': 3.277}, 'Calafell': {'lat': 41.199, 'lon': 1.567}, 'Caldes de Montbui': {'lat': 41.633, 'lon': 2.166}, 'Calella': {'lat': 41.614, 'lon': 2.664},
    'Cambrils': {'lat': 41.066, 'lon': 1.056}, 'Canet de Mar': {'lat': 41.590, 'lon': 2.580}, 'Cardedeu': {'lat': 41.640, 'lon': 2.358}, 'Cardona': {'lat': 41.914, 'lon': 1.679},
    'Castell-Platja d\'Aro': {'lat': 41.818, 'lon': 3.067}, 'Castelldefels': {'lat': 41.279, 'lon': 1.975}, 'Castellfollit de la Roca': {'lat': 42.220, 'lon': 2.551}, 'Cerdanyola del Vallès': {'lat': 41.491, 'lon': 2.141},
    'Cervera': {'lat': 41.666, 'lon': 1.272}, 'Cornellà de Llobregat': {'lat': 41.355, 'lon': 2.069}, 'Cubelles': {'lat': 41.208, 'lon': 1.674}, 'Cunit': {'lat': 41.197, 'lon': 1.635},
    'Deltebre': {'lat': 40.719, 'lon': 0.710}, 'El Masnou': {'lat': 41.481, 'lon': 2.318}, 'El Prat de Llobregat': {'lat': 41.326, 'lon': 2.095}, 'Esparreguera': {'lat': 41.536, 'lon': 1.868},
    'Esplugues de Llobregat': {'lat': 41.375, 'lon': 2.086}, 'Falset': {'lat': 41.144, 'lon': 0.819}, 'Figueres': {'lat': 42.266, 'lon': 2.962}, 'Gandesa': {'lat': 41.052, 'lon': 0.436},
    'Gavà': {'lat': 41.305, 'lon': 2.001}, 'Girona': {'lat': 41.983, 'lon': 2.824}, 'Granollers': {'lat': 41.608, 'lon': 2.289}, 'Guissona': {'lat': 41.783, 'lon': 1.288}, 'Hostalric': {'lat': 41.748, 'lon': 2.636},
    'Igualada': {'lat': 41.580, 'lon': 1.616}, 'La Garriga': {'lat': 41.683, 'lon': 2.282}, 'La Jonquera': {'lat': 42.419, 'lon': 2.875}, 'La Llagosta': {'lat': 41.516, 'lon': 2.193}, 'La Pobla de Segur': {'lat': 42.247, 'lon': 0.968},
    'La Roca del Vallès': {'lat': 41.587, 'lon': 2.327}, 'La Seu d\'Urgell': {'lat': 42.358, 'lon': 1.463}, 'L\'Ametlla de Mar': {'lat': 40.883, 'lon': 0.802}, 'L\'Ampolla': {'lat': 40.812, 'lon': 0.709}, 'L\'Escala': {'lat': 42.122, 'lon': 3.131},
    'Lleida': {'lat': 41.617, 'lon': 0.622}, 'Lliçà d\'Amunt': {'lat': 41.597, 'lon': 2.241}, 'Lloret de Mar': {'lat': 41.700, 'lon': 2.845}, 'Malgrat de Mar': {'lat': 41.645, 'lon': 2.741}, 'Manlleu': {'lat': 42.000, 'lon': 2.283},
    'Manresa': {'lat': 41.727, 'lon': 1.825}, 'Martorell': {'lat': 41.474, 'lon': 1.927}, 'Mataró': {'lat': 41.538, 'lon': 2.445}, 'Moià': {'lat': 41.810, 'lon': 2.096}, 'Molins de Rei': {'lat': 41.414, 'lon': 2.016},
    'Mollerussa': {'lat': 41.631, 'lon': 0.895}, 'Mollet del Vallès': {'lat': 41.539, 'lon': 2.213}, 'Montblanc': {'lat': 41.375, 'lon': 1.161}, 'Montcada i Reixac': {'lat': 41.485, 'lon': 2.187}, 'Montgat': {'lat': 41.464, 'lon': 2.279},
    'Monistrol de Montserrat': {'lat': 41.610, 'lon': 1.844}, 'Móra d\'Ebre': {'lat': 41.092, 'lon': 0.643}, 'Móra la Nova': {'lat': 41.106, 'lon': 0.655}, 'Olesa de Montserrat': {'lat': 41.545, 'lon': 1.894}, 'Olot': {'lat': 42.181, 'lon': 2.490},
    'Palafolls': {'lat': 41.670, 'lon': 2.753}, 'Palafrugell': {'lat': 41.918, 'lon': 3.163}, 'Palamós': {'lat': 41.846, 'lon': 3.128}, 'Palau-solità i Plegamans': {'lat': 41.583, 'lon': 2.179}, 'Parets del Vallès': {'lat': 41.573, 'lon': 2.233},
    'Piera': {'lat': 41.520, 'lon': 1.748}, 'Premià de Mar': {'lat': 41.491, 'lon': 2.359}, 'Puigcerdà': {'lat': 42.432, 'lon': 1.928}, 'Reus': {'lat': 41.155, 'lon': 1.107}, 'Ripoll': {'lat': 42.201, 'lon': 2.190},
    'Ripollet': {'lat': 41.498, 'lon': 2.158}, 'Roses': {'lat': 42.262, 'lon': 3.175}, 'Rubí': {'lat': 41.493, 'lon': 2.032}, 'Rupit i Pruit': {'lat': 42.026, 'lon': 2.465}, 'Sabadell': {'lat': 41.547, 'lon': 2.108},
    'Salou': {'lat': 41.076, 'lon': 1.140}, 'Sant Adrià de Besòs': {'lat': 41.428, 'lon': 2.219}, 'Sant Andreu de la Barca': {'lat': 41.447, 'lon': 1.979}, 'Sant Boi de Llobregat': {'lat': 41.346, 'lon': 2.041}, 'Sant Carles de la Ràpita': {'lat': 40.618, 'lon': 0.593},
    'Sant Celoni': {'lat': 41.691, 'lon': 2.491}, 'Sant Cugat del Vallès': {'lat': 41.472, 'lon': 2.085}, 'Sant Feliu de Guíxols': {'lat': 41.780, 'lon': 3.028}, 'Sant Feliu de Llobregat': {'lat': 41.381, 'lon': 2.045}, 'Sant Joan Despí': {'lat': 41.368, 'lon': 2.057},
    'Sant Just Desvern': {'lat': 41.383, 'lon': 2.072}, 'Sant Pere de Ribes': {'lat': 41.259, 'lon': 1.769}, 'Sant Pol de Mar': {'lat': 41.602, 'lon': 2.624}, 'Sant Sadurní d\'Anoia': {'lat': 41.428, 'lon': 1.785}, 'Sant Vicenç dels Horts': {'lat': 41.392, 'lon': 2.008},
    'Santa Coloma de Farners': {'lat': 41.859, 'lon': 2.668}, 'Santa Coloma de Gramenet': {'lat': 41.454, 'lon': 2.213}, 'Santa Perpètua de Mogoda': {'lat': 41.536, 'lon': 2.182}, 'Santa Susanna': {'lat': 41.636, 'lon': 2.711}, 'Sitges': {'lat': 41.235, 'lon': 1.811},
    'Solsona': {'lat': 41.992, 'lon': 1.516}, 'Sort': {'lat': 42.413, 'lon': 1.129}, 'Tàrrega': {'lat': 41.646, 'lon': 1.141}, 'Tarragona': {'lat': 41.118, 'lon': 1.245}, 'Terrassa': {'lat': 41.561, 'lon': 2.008},
    'Tordera': {'lat': 41.702, 'lon': 2.719}, 'Torelló': {'lat': 42.048, 'lon': 2.262}, 'Tortosa': {'lat': 40.812, 'lon': 0.521}, 'Tossa de Mar': {'lat': 41.720, 'lon': 2.932}, 'Tremp': {'lat': 42.166, 'lon': 0.894},
}

if not pobles_data:
    st.warning("La llista de localitats està buida. S'està utilitzant una llista de mostra.")
    pobles_data = {
        'Barcelona': {'lat': 41.387, 'lon': 2.168}, 'Girona': {'lat': 41.983, 'lon': 2.824},
        'Lleida': {'lat': 41.617, 'lon': 0.622}, 'Tarragona': {'lat': 41.118, 'lon': 1.245},
        'Puigcerdà': {'lat': 42.432, 'lon': 1.928}, 'Vielha': {'lat': 42.702, 'lon': 0.796},
        'Tortosa': {'lat': 40.812, 'lon': 0.521}
    }

# --- INICIALITZACIÓ DEL SESSION STATE ---
if 'poble_seleccionat' not in st.session_state:
    st.session_state.poble_seleccionat = next(iter(pobles_data))
if 'hora_seleccionada_str' not in st.session_state:
    try:
        tz = pytz.timezone('Europe/Madrid')
        st.session_state.hora_seleccionada_str = f"{datetime.now(tz).hour:02d}:00h"
    except:
        st.session_state.hora_seleccionada_str = "12:00h"

# --- 1. LÒGICA DE CÀRREGA DE DADES I CÀLCUL ---

def chunker(seq, size): return (seq[pos:pos + size] for pos in range(0, len(seq), size))

@st.cache_data(ttl=18000)
def carregar_dades_lot(_chunk_locations):
    p_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"] for p in p_levels]
    chunk_noms, chunk_lats, chunk_lons = zip(*_chunk_locations)
    params = { "latitude": list(chunk_lats), "longitude": list(chunk_lons), "hourly": h_base + h_press, "models": "arome_france", "timezone": "auto", "forecast_days": FORECAST_DAYS }
    try:
        respostes_chunk = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        return {chunk_noms[i]: respostes_chunk[i] for i in range(len(respostes_chunk))}, p_levels, None
    except Exception as e: return None, None, str(e)

@st.cache_data(ttl=18000)
def obtener_dades_mapa(variable, nivell, hourly_index, forecast_days):
    lats, lons = np.linspace(40.5, 42.8, 12), np.linspace(0.2, 3.3, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats); api_vars = []
    if variable == 'wind': api_vars = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
    elif variable == 'dewpoint': api_vars = [f"dew_point_{nivell}hPa"]
    elif variable == 'humidity': api_vars = [f"relative_humidity_{nivell}hPa"]
    else: return None, None, None, f"Variable '{variable}' no reconeguda."
    params = { "latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": api_vars, "models": "arome_france", "timezone": "auto", "forecast_days": forecast_days }
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
    except Exception as e: return None, None, None, str(e)

def get_next_arome_update_time():
    now_utc = datetime.now(pytz.utc)
    run_hours_utc = [0, 6, 12, 18]; availability_delay = timedelta(hours=4)
    next_update_time = None
    for run_hour in run_hours_utc:
        available_time = now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0) + availability_delay
        if available_time > now_utc: next_update_time = available_time; break
    if next_update_time is None: next_update_time = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) + availability_delay
    return f"Pròxima actualització de dades (AROME) estimada a les {next_update_time.astimezone(pytz.timezone('Europe/Madrid')).strftime('%H:%Mh')}"

def calculate_parameters(p, T, Td, u, v, h):
    params = {}
    def get_val(qty, unit=None):
        try: return qty.to(unit).m if unit else qty.m
        except: return None
    params['SFC_Temp'] = {'value': get_val(T[0], 'degC'), 'units': '°C'}
    try:
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]); cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
        params['CAPE_Brut'] = {'value': get_val(cape, 'J/kg'), 'units': 'J/kg'}
        params['CIN_Fre'] = {'value': get_val(cin, 'J/kg'), 'units': 'J/kg'}
        if params.get('CAPE_Brut', {}).get('value', 0) > 0:
            params['W_MAX'] = {'value': np.sqrt(2 * params['CAPE_Brut']['value']), 'units': 'm/s'}
            params['CAPE_Utilitzable'] = {'value': max(0, params['CAPE_Brut']['value'] - abs(params.get('CIN_Fre', {}).get('value', 0))), 'units': 'J/kg'}
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
        if (idx := np.where(np.diff(np.sign(T_c)))[0]).size > 0:
            h_zero_iso_msl = np.interp(0, [T_c[idx[0]+1], T_c[idx[0]]], [H_m[idx[0]+1], H_m[idx[0]]])
            params['ZeroIso_AGL'] = {'value': (h_zero_iso_msl - H_m[0]), 'units': 'm'}
    except: pass
    try:
        p_levels_interp = np.arange(p.m.min(), p.m.max(), 10) * units.hPa
        T_interp = mpcalc.interpolate_1d(p, T, p_levels_interp)
        lr = mpcalc.lapse_rate(p_levels_interp, T_interp, bottom=700*units.hPa, top=500*units.hPa)
        params['LapseRate_700_500'] = {'value': get_val(lr, 'delta_degC/km'), 'units': '°C/km'}
    except: pass
    try:
        dcape, _ = mpcalc.dcape(p, T, Td)
        params['DCAPE'] = {'value': get_val(dcape, 'J/kg'), 'units': 'J/kg'}
    except: pass
    try:
        stp_cin = mpcalc.significant_tornado(cape=params.get('CAPE_Utilitzable',{}).get('value',0)*units('J/kg'), lcl_height=params.get('LCL_AGL',{}).get('value',9999)*units.m, storm_helicity=params.get('SRH_0-1km',{}).get('value',0)*units('m^2/s^2'), bulk_shear=params.get('Shear_0-6km',{}).get('value',0)*units('m/s'))
        params['STP_cin'] = {'value': get_val(stp_cin), 'units': ''}
    except: pass
    return params

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

@st.cache_data(ttl=18000)
def encontrar_localitats_con_convergencia(_hourly_index, _nivell, _localitats, _threshold, _forecast_days):
    lats, lons, data, error = obtener_dades_mapa('wind', _nivell, _hourly_index, _forecast_days)
    if error or not lats or len(lats) < 4: return set()
    speeds, dirs = zip(*data); speeds_ms = (np.array(speeds) * 1000 / 3600) * units('m/s'); dirs_deg = np.array(dirs) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    grid_lon, grid_lat = np.linspace(min(lons), max(lons), 100), np.linspace(min(lats), max(lats), 100)
    X, Y = np.meshgrid(grid_lon, grid_lat); points = np.vstack((lons, lats)).T
    u_grid, v_grid = griddata(points, u_comp.m, (X, Y), method='cubic'), griddata(points, v_comp.m, (X, Y), method='cubic')
    u_grid, v_grid = np.nan_to_num(u_grid), np.nan_to_num(v_grid)
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y); divergence = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5
    localitats_en_convergencia = set()
    for nom_poble, coords in _localitats.items():
        try:
            lon_idx, lat_idx = (np.abs(grid_lon - coords['lon'])).argmin(), (np.abs(grid_lat - coords['lat'])).argmin()
            if divergence.m[lat_idx, lon_idx] < _threshold: localitats_en_convergencia.add(nom_poble)
        except: continue
    return localitats_en_convergencia

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

@st.cache_data(ttl=18000)
def precalcular_avisos_hores(_totes_les_dades, _p_levels):
    avisos_hores = {}
    avisos_a_buscar = {"PRECAUCIÓ", "AVÍS", "RISC ALT", "ALERTA DE DISPARADOR"}
    for nom_poble, sondeo in _totes_les_dades.items():
        for hora in range(24):
            profiles = processar_sondeig_per_hora(sondeo, hora, _p_levels)
            if profiles:
                parametros = calculate_parameters(*profiles)
                if generar_avis_potencial_per_precalcul(parametros) in avisos_a_buscar:
                    avisos_hores[nom_poble] = hora; break
    return dict(sorted(avisos_hores.items()))

# --- 2. FUNCIONS DE VISUALITZACIÓ I FORMAT ---

def display_avis_principal(titol_avís, text_avís, color_avís, icona_personalitzada=None):
    icon_map = {"ESTABLE": "☀️", "RISC BAIX": "☁️", "PRECAUCIÓ": "⚡️", "AVÍS": "⚠️", "RISC ALT": "🌪️", "POTENCIAL SEVER": "🧐", "POTENCIAL MODERAT": "🤔", "ALERTA DE DISPARADOR": "🎯"}
    icona = icona_personalitzada if icona_personalitzada else icon_map.get(titol_avís, "ℹ️")
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: var(--secondary-background-color); border-left: 8px solid {color_avís}; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 3.5em; line-height: 1;">{icona}</div>
        <div><h3 style="color: {color_avís}; margin-top: 0; margin-bottom: 0.5rem; font-weight: bold;">{titol_avís}</h3><p style="margin-bottom: 0; color: var(--text-color);">{text_avís}</p></div>
    </div>""", unsafe_allow_html=True)

def get_parameter_style(param_name, value):
    color = "inherit"; emoji = ""
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
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0); cin = params.get('CIN_Fre', {}).get('value'); shear = params.get('Shear_0-6km', {}).get('value'); srh1 = params.get('SRH_0-1km', {}).get('value'); lcl_agl = params.get('LCL_AGL', {}).get('value', 9999); lfc_agl = params.get('LFC_AGL', {}).get('value', 9999)
    dcape = params.get('DCAPE', {}).get('value', 0); stp = params.get('STP_cin', {}).get('value', 0); lr = params.get('LapseRate_700_500', {}).get('value', 0); pwat = params.get('PWAT_Total', {}).get('value', 0)

    if cape_u < 100: return "ESTABLE", "Sense risc de tempestes significatives. L'atmosfera és estable.", "#3CB371"
    if cin is not None and cin < -150: return "ESTABLE", f"La 'tapa' atmosfèrica (CIN de {cin:.0f} J/kg) és massa forta i probablement inihibirà qualsevol convecció.", "#3CB371"
    
    if not is_convergence_active and lfc_agl > 3500: return "RISC BAIX", f"L'inici de la convecció (LFC a {lfc_agl:.0f} m) és massa alt, fent les tempestes improbables sense un forçament potent.", "#4682B4"

    cond_supercelula = shear is not None and shear > 20 and cape_u > 1500 and srh1 is not None and srh1 > 200 and lcl_agl < 1300
    cond_avis_sever = shear is not None and shear > 18 and cape_u > 1200
    cond_precaucio = shear is not None and shear > 12 and cape_u > 500

    if cond_supercelula and stp > 0.8:
        risks = ["calamarsa grossa", "vent destructiu"]
        if lcl_agl < 1000: risks.insert(0, "tornados")
        risks_text = ", ".join(risks)
        if is_convergence_active: return "RISC ALT", f"Entorn de SUPERCL·LULA TORNÀDICA (STP={stp:.1f}). La convergència actua com a disparador. Risc de {risks_text}.", "#E60073"
        else: return "POTENCIAL SEVER", f"Potencial latent de SUPERCL·LULA TORNÀDICA (STP={stp:.1f}) per falta d'un disparador clar. L'entorn és perillós.", "#FFC300"
    
    elif cond_avis_sever:
        risks = []
        if dcape > 900: risks.append("fortes ràfegues de vent (esclafits severs)")
        if lr > 6.8: risks.append("calamarsa de mida considerable")
        if pwat > 35: risks.append("pluges torrencials i inundacions sobtades")
        elif not risks: risks.append("fortes pluges")
        risks_text = ", ".join(risks)
        if is_convergence_active: return "AVÍS", f"Potencial per a tempestes SEVERES organitzades. La convergència afavoreix el seu inici. Risc de {risks_text}.", "#FF8C00"
        else: return "POTENCIAL MODERAT", f"Entorn de tempesta SEVERA latent per falta d'un disparador clar. Risc de {risks_text} si s'activa.", "#FFD700"
    
    elif cond_precaucio:
        missatge_base = "Risc de TEMPESTES ORGANITZADES (multicèl·lules). Possibles fortes pluges i calamarsa local."
        if pwat > 35: missatge_base += " Atenció al risc de xàfecs torrencials."
        suggeriment = " La convergència detectada augmenta la probabilitat que es formin." if is_convergence_active else " L'absència de convergència clara podria limitar-ne el desenvolupament."
        return "PRECAUCIÓ", missatge_base + suggeriment, "#FFD700"
    
    else:
        missatge_base = "Possibles xàfecs o tempestes febles i aïllades."
        suggeriment = " La convergència podria ajudar a formar alguns nuclis." if is_convergence_active else ""
        return "RISC BAIX", missatge_base + suggeriment, "#4682B4"

def generar_avis_convergencia(params, is_convergence_active):
    if not is_convergence_active: return None, None, None
    cape_u=params.get('CAPE_Utilitzable',{}).get('value',0);cin=params.get('CIN_Fre',{}).get('value')
    if cape_u > 500 and (cin is None or cin > -50):
        return "ALERTA DE DISPARADOR", f"La forta convergència de vents pot actuar com a disparador. Amb un CAPE de {cape_u:.0f} J/kg i una 'tapa' (CIN) feble, hi ha un alt potencial que les tempestes s'iniciïn de manera explosiva.", "#FF4500"
    return None, None, None
    
def generar_analisi_detallada(params):
    def stream_text(text):
        for word in text.split(): yield word + " "; time.sleep(0.02)
        yield "\n\n"
    cape_u=params.get('CAPE_Utilitzable',{}).get('value',0); cin=params.get('CIN_Fre',{}).get('value'); shear6=params.get('Shear_0-6km',{}).get('value'); srh1=params.get('SRH_0-1km',{}).get('value'); lcl_agl=params.get('LCL_AGL',{}).get('value'); lfc_agl=params.get('LFC_AGL',{}).get('value'); w_max=params.get('W_MAX',{}).get('value'); el_msl = params.get('EL_MSL', {}).get('value')
    dcape=params.get('DCAPE',{}).get('value'); stp=params.get('STP_cin',{}).get('value'); lr=params.get('LapseRate_700_500',{}).get('value'); pwat=params.get('PWAT_Total',{}).get('value')

    if cape_u is None or cape_u < 100: yield from stream_text("### Anàlisi General\nL'atmosfera és estable. El potencial energètic per a la formació de tempestes (CAPE) és pràcticament inexistent. No s'esperen fenòmens de temps sever."); return
    yield from stream_text("### Potencial Energètic (Termodinàmica)")
    cape_text="feble" if cape_u<1000 else "moderada" if cape_u<2500 else "forta" if cape_u<4000 else "extrema"
    yield from stream_text(f"**Inestabilitat (CAPE):** El valor de CAPE utilitzable és de **{cape_u:.0f} J/kg**, el que indica una inestabilitat **{cape_text}**. Això es tradueix en un potencial de corrents ascendents de fins a **{w_max*3.6:.0f} km/h**, podent sostenir un núvol de tempesta fins a una altitud de **{el_msl:.1f} km**.")
    if cin is not None:
        if cin < -100: yield from stream_text(f"**Inhibició (CIN):** Hi ha una 'tapa' atmosfèrica forta de **{cin:.0f} J/kg**. Es necessitarà un mecanisme de forçament significatiu per trencar-la i alliberar la inestabilitat.")
        elif cin < -25: yield from stream_text(f"**Inhibició (CIN):** La 'tapa' de **{cin:.0f} J/kg** és moderada. Si es trenca, el desenvolupament de les tempestes podria ser explosiu.")
        else: yield from stream_text("**Inhibició (CIN):** La 'tapa' és feble o inexistent. L'energia està fàcilment disponible.")
    if lr:
        lr_text = "molt pronunciat" if lr > 7 else "moderat"
        yield from stream_text(f"**Gradient Tèrmic (700-500hPa):** El refredament amb l'altura és **{lr_text}** ({lr:.1f} °C/km), un factor que afavoreix el desenvolupament de corrents ascendents forts i, per tant, el potencial de calamarsa.")
    yield from stream_text("### Organització i Rotació (Cinemàtica)")
    if shear6 is not None:
        if shear6 < 10: shear_text = "Molt feble, afavorint tempestes desorganitzades i de curta durada (unicel·lulars)."
        elif shear6 < 18: shear_text = "Moderat, suficient per organitzar les tempestes en sistemes multicel·lulars o línies de tempestes."
        else: shear_text = "Fort, un ingredient clau per al desenvolupament de supercèl·lules amb rotació (mesociclons)."
        yield from stream_text(f"**Cisallament 0-6 km:** El valor és de **{shear6:.1f} m/s**. {shear_text}")
    if srh1 is not None and srh1 > 100:
        srh_text="moderat" if srh1<250 else "fort"
        lcl_text = f"Especialment si la base del núvol (LCL) és baixa com en aquest cas ({lcl_agl:.0f} m), " if lcl_agl < 1200 else ""
        yield from stream_text(f"**Helicitat 0-1 km (SRH):** La rotació potencial a nivells baixos és **{srh_text}** ({srh1:.0f} m²/s²). {lcl_text}Valors elevats augmenten el risc que la rotació de la tempesta arribi a terra (tornados).")
    yield from stream_text("### Síntesi i Riscos Específics")
    if stp and stp > 0.5: yield from stream_text(f"**Potencial Tornàdic (STP):** L'índex de tornado significatiu és de **{stp:.1f}**. Valors superiors a 1 indiquen un entorn molt favorable per a supercèl·lules tornàdiques si s'arriben a formar.")
    if dcape and dcape > 800: yield from stream_text(f"**Potencial de Vent Sever (DCAPE):** El valor de **{dcape:.0f} J/kg** indica un alt potencial per a la formació de corrents descendents molt forts (esclafits o 'downbursts') que poden causar danys a la superfície.")
    if pwat and pwat > 30: yield from stream_text(f"**Potencial de Precipitació Intensa (Aigua Precipitable):** El contingut d'humitat a la columna atmosfèrica és elevat ({pwat:.1f} mm), afavorint xàfecs de gran intensitat i possibles inundacions locals.")
    yield from stream_text(f"**La Clau del Pronòstic:** La clau principal avui és la interacció entre la **inestabilitat {cape_text}** i un **cisallament {('fort' if shear6>18 else 'moderat')}**. La presència (o absència) d'un mecanisme de tret com la convergència serà el factor decisiu per determinar si s'allibera aquest potencial i quin tipus de tempestes es desenvolupen.")

def display_metrics(params_dict):
    param_map = [
        ('Temperatura','SFC_Temp'), ('CAPE Utilitzable','CAPE_Utilitzable'), ('CIN (Fre)','CIN_Fre'), ('Vel. Asc. Màx.','W_MAX'),
        ('Shear 0-6km','Shear_0-6km'), ('SRH 0-1km','SRH_0-1km'), ('Potencial Tornàdic','STP_cin'), ('Potencial Esclafits','DCAPE'),
        ('Gradient Tèrmic','LapseRate_700_500'), ('Aigua Precipitable','PWAT_Total'), ('Base núvol (AGL)','LCL_AGL'), ('Cim tempesta (MSL)','EL_MSL')
    ]
    st.markdown("""<style>.metric-container{border:1px solid rgba(128,128,128,0.2);border-radius:10px;padding:10px;margin-bottom:10px;}</style>""", unsafe_allow_html=True)
    available_params=[(label,key) for label,key in param_map if key in params_dict and params_dict[key].get('value') is not None]
    cols=st.columns(min(4,len(available_params)))
    for i,(label,key) in enumerate(available_params):
        param=params_dict[key]; value=param['value']; units_str=param['units']; val_str=f"{value:.1f}" if isinstance(value,(float,np.floating)) else str(value); value_color,emoji=get_parameter_style(key,value); border_color=value_color if value_color!='inherit' else 'rgba(128,128,128,0.2)'
        with cols[i%4]: st.markdown(f"""<div class="metric-container" style="border-color:{border_color};"><div style="font-size:0.9em;color:gray;">{label}</div><div style="font-size:1.25em;font-weight:bold;color:{value_color};">{val_str} <span style='font-size:0.8em;color:gray;'>{units_str}</span> {emoji}</div></div>""", unsafe_allow_html=True)

def crear_hodograf(p, u, v, h):
    fig, ax=plt.subplots(1,1,figsize=(5,5)); hodo=Hodograph(ax,component_range=40.); hodo.add_grid(increment=10); hodoline=hodo.plot_colormapped(u,v,h.to('km'),cmap='gist_ncar'); plt.colorbar(hodoline,ax=ax,orientation='vertical',pad=0.05,shrink=0.8).set_label('Altitud (km)')
    try: rm,_,_=mpcalc.bunkers_storm_motion(p,u,v,h); hodo.plot_vectors(rm[0].to('kt'),rm[1].to('kt'),color='black',label='Mov. Tempesta (RM)')
    except: pass
    ax.set_xlabel('kt'); ax.set_ylabel('kt'); return fig

def crear_skewt(p, T, Td, u, v):
    fig = plt.figure(figsize=(7, 9)); skew = SkewT(fig, rotation=45)
    skew.plot(p, T, 'r', lw=2, label='Temperatura'); skew.plot(p, Td, 'b', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), length=7, color='black')
    skew.plot_dry_adiabats(color='lightcoral', ls='--', alpha=0.5); skew.plot_moist_adiabats(color='cornflowerblue', ls='--', alpha=0.5)
    skew.plot_mixing_lines(color='lightgreen', ls='--', alpha=0.5); skew.ax.axvline(0, color='darkturquoise', linestyle='--', label='Isoterma 0°C')
    if len(p) > 1:
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', lw=2, ls='--', label='Trajectòria de la Parcel·la')
            cape, cin = mpcalc.cape_cin(p, T, Td, prof); skew.shade_cape(p, T, prof, alpha=0.3, color='orange'); skew.shade_cin(p, T, prof, alpha=0.6, color='gray')
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); lfc_p, _ = mpcalc.lfc(p, T, Td, prof); el_p, _ = mpcalc.el(p, T, Td, prof)
            if lcl_p: skew.ax.axhline(lcl_p.m, color='purple', linestyle='--', label=f'LCL {lcl_p.m:.0f} hPa')
            if lfc_p: skew.ax.axhline(lfc_p.m, color='darkred', linestyle='--', label=f'LFC {lfc_p.m:.0f} hPa')
            if el_p: skew.ax.axhline(el_p.m, color='red', linestyle='--', label=f'EL {el_p.m:.0f} hPa')
        except: pass
    skew.ax.set_ylim(1050, 100); skew.ax.set_xlim(-40, 40); skew.ax.set_xlabel('Temperatura (°C)'); skew.ax.set_ylabel('Pressió (hPa)'); plt.legend(); return fig

def crear_mapa_base(nivell, lat_sel, lon_sel, nom_poble_sel, titol):
    fig=plt.figure(figsize=(9,9),dpi=150); ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree()); ax.set_extent([0,3.5,40.4,43],crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND,facecolor="#E0E0E0",zorder=0); ax.add_feature(cfeature.OCEAN,facecolor='#b0c4de',zorder=0); ax.add_feature(cfeature.COASTLINE,edgecolor='black',linewidth=0.5,zorder=1); ax.add_feature(cfeature.BORDERS,linestyle=':',edgecolor='black',zorder=1); ax.plot(lon_sel,lat_sel,'o',markersize=12,markerfacecolor='yellow',markeredgecolor='black',markeredgewidth=2,transform=ccrs.Geodetic(),zorder=5); ax.text(lon_sel+0.05,lat_sel+0.05,nom_poble_sel,transform=ccrs.Geodetic(),zorder=6,bbox=dict(facecolor='white',alpha=0.8,edgecolor='none',boxstyle='round,pad=0.2')); ax.set_title(f"{titol} a {nivell}hPa",weight='bold'); return fig,ax

def crear_mapa_vents(lats, lons, data, nivell, lat_sel, lon_sel, nom_poble_sel):
    speeds,dirs=zip(*data); speeds_ms=(np.array(speeds)*1000/3600)*units('m/s'); dirs_deg=np.array(dirs)*units.degrees; u_comp,v_comp=mpcalc.wind_components(speeds_ms,dirs_deg)
    fig,ax=crear_mapa_base(nivell,lat_sel,lon_sel,nom_poble_sel,"Flux i focus de convergència")
    grid_lon,grid_lat=np.linspace(min(lons),max(lons),100),np.linspace(min(lats),max(lats),100); X,Y=np.meshgrid(grid_lon,grid_lat); points=np.vstack((lons,lats)).T
    u_grid,v_grid=griddata(points,u_comp.m,(X,Y),method='cubic'),griddata(points,v_comp.m,(X,Y),method='cubic'); u_grid,v_grid=np.nan_to_num(u_grid),np.nan_to_num(v_grid)
    dx,dy=mpcalc.lat_lon_grid_deltas(X,Y); divergence=mpcalc.divergence(u_grid*units('m/s'),v_grid*units('m/s'),dx=dx,dy=dy)*1e5; divergence_values=np.ma.masked_where(divergence.m>-5.5,divergence.m)
    cont=ax.contourf(X,Y,divergence_values,levels=np.linspace(-15.0,-5.5,10),cmap='Reds_r',alpha=0.6,zorder=2,transform=ccrs.PlateCarree(),extend='min'); ax.streamplot(grid_lon,grid_lat,u_grid,v_grid,color="#000000",density=5.9,linewidth=0.5,arrowsize=0.50,zorder=4,transform=ccrs.PlateCarree()); fig.colorbar(cont,ax=ax,orientation='vertical',label='Convergència (x10⁻⁵ s⁻¹)',shrink=0.7); return fig

def crear_mapa_generic(lats, lons, data, nivell, lat_sel, lon_sel, nom_poble_sel, titol_var, cmap, unitat, levels):
    fig,ax=crear_mapa_base(nivell,lat_sel,lon_sel,nom_poble_sel,titol_var)
    grid_lon,grid_lat=np.linspace(min(lons),max(lons),100),np.linspace(min(lats),max(lats),100); X,Y=np.meshgrid(grid_lon,grid_lat); points=np.vstack((lons,lats)).T
    grid_data=griddata(points,data,(X,Y),method='cubic'); grid_data=np.nan_to_num(grid_data)
    cont=ax.contourf(X,Y,grid_data,cmap=cmap,levels=levels,alpha=0.7,zorder=2,transform=ccrs.PlateCarree(),extend='both'); fig.colorbar(cont,ax=ax,orientation='vertical',label=f'{titol_var} ({unitat})',shrink=0.7); return fig

def crear_grafic_orografia(params, zero_iso_h_agl):
    lcl_agl = params.get('LCL_AGL', {}).get('value'); lfc_agl = params.get('LFC_AGL', {}).get('value')
    if lcl_agl is None or np.isnan(lcl_agl): return None
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150); fig.patch.set_alpha(0); ax.set_xlim(0, 10); ax.set_ylabel(""); ax.tick_params(axis='y', labelleft=False, length=5, color='white'); ax.spines['left'].set_color('white'); ax.spines['bottom'].set_color('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.set_xticklabels([]); ax.set_xticks([])
    sky_gradient = np.linspace(0.6, 1.0, 256).reshape(-1, 1); sky_gradient = np.vstack((sky_gradient, sky_gradient)); ax.imshow(sky_gradient, aspect='auto', cmap='Blues_r', extent=[0, 10, 0, 14], zorder=0)
    has_lfc = lfc_agl is not None and np.isfinite(lfc_agl); peak_h_m = lfc_agl if has_lfc and lfc_agl > 1000 else (lcl_agl * 1.5 if lcl_agl > 1000 else 2000); peak_h_km = min(peak_h_m / 1000.0, 8.0)
    x_mountain = np.linspace(0, 10, 200); y_mountain = peak_h_km * (1 - np.cos(x_mountain * np.pi / 10)) / 2 * (1 - (x_mountain - 5)**2 / 25)
    ax.add_patch(Polygon(np.vstack([x_mountain, y_mountain]).T, facecolor='#6B8E23', edgecolor='#3A4D14', lw=2, zorder=2))
    line_outline_effect = [path_effects.withStroke(linewidth=3.5, foreground='black')]; lcl_km = lcl_agl / 1000
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
    lcl_agl,el_msl_km,cape=(params.get(k,{}).get('value') for k in ['LCL_AGL','EL_MSL','CAPE_Brut'])
    if lcl_agl is None or el_msl_km is None: return None
    cape=cape or 0; fig,ax=plt.subplots(figsize=(6,9),dpi=120); ax.set_facecolor('#4F94CD'); lcl_km=lcl_agl/1000; el_km=el_msl_km
    if is_convergence_active and cape > 100:
        y_points=np.linspace(lcl_km,el_km,100); cloud_width=1.0+np.sin(np.pi*(y_points-lcl_km)/(el_km-lcl_km))*(1+cape/2000)
        for y,width in zip(y_points,cloud_width):
            center_x=np.interp(y*1000,H.m,u.m)/15
            for _ in range(25): ax.add_patch(Circle((center_x+(random.random()-0.5)*width,y+(random.random()-0.5)*0.4),0.2+random.random()*0.4,color='white',alpha=0.15,lw=0))
        anvil_wind_u=np.interp(el_km*1000,H.m,u.m)/10; anvil_center_x=np.interp(el_km*1000,H.m,u.m)/15
        for _ in range(80): ax.add_patch(Circle((anvil_center_x+(random.random()-0.2)*4+anvil_wind_u,el_km+(random.random()-0.5)*0.5),0.2+random.random()*0.6,color='white',alpha=0.2,lw=0))
        if cape>2500: ax.add_patch(Circle((anvil_center_x,el_km+cape/5000),0.4,color='white',alpha=0.5))
    else: ax.text(5,8,"Sense disparador o energia\nsuficient per a convecció profunda.",ha='center',va='center',color='black',fontsize=16,weight='bold',bbox=dict(facecolor='lightblue',alpha=0.7,boxstyle='round,pad=0.5'))
    barb_heights_km=np.arange(1,16,1); u_barbs=np.interp(barb_heights_km*1000,H.m,u.to('kt').m); v_barbs=np.interp(barb_heights_km*1000,H.m,v.to('kt').m); ax.barbs(np.full_like(barb_heights_km,9.5),barb_heights_km,u_barbs,v_barbs,length=7,color='black'); ax.set_ylim(0,16); ax.set_xlim(0,10); ax.set_ylabel("Altitud (km, MSL)"); ax.set_title("Visualització del Núvol",weight='bold'); ax.set_xticks([]); ax.grid(axis='y',linestyle='--',alpha=0.3); return fig


# --- 3. INTERFÍCIE I FLUX PRINCIPAL DE L'APLICACIÓ ---
st.markdown('<h1 style="text-align: center; color: #FF4B4B;">⚡ Tempestes.cat</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Eina d\'Anàlisi i Previsió de Fenòmens Severs a Catalunya</p>', unsafe_allow_html=True)

totes_les_dades={}; p_levels=[]; error_carrega=None; locations_data=[(k,v['lat'],v['lon']) for k,v in pobles_data.items()]; chunk_size=50; chunks=list(chunker(locations_data,chunk_size)); num_chunks=len(chunks); progress_bar=st.progress(0,text="Iniciant la càrrega de dades...")
for i,chunk in enumerate(chunks):
    progress_bar.progress((i+1)/num_chunks,text=f"Carregant lot {i+1} de {num_chunks}...")
    dades_lot,p_levels_lot,error_lot=carregar_dades_lot(tuple(chunk))
    if error_lot: error_carrega=error_lot; break
    if dades_lot: totes_les_dades.update(dades_lot)
    if not p_levels and p_levels_lot: p_levels=p_levels_lot
progress_bar.empty()
if error_carrega: st.error(f"No s'ha pogut carregar la informació base. Error: {error_carrega}"); st.stop()
st.toast("Dades de sondeig carregades correctament!",icon="✅")

with st.spinner("Buscant hores amb avisos a tot el territori..."):
    avisos_hores=precalcular_avisos_hores(totes_les_dades,p_levels)

def update_from_avis_selector():
    poble_avis_format = st.session_state.avis_selector
    if " (" in poble_avis_format:
        nom_poble_real = poble_avis_format.split(" (")[0].split(" ", 1)[1]
        if nom_poble_real in avisos_hores:
            hora_avis = avisos_hores[nom_poble_real]
            st.session_state.hora_seleccionada_str = f"{hora_avis:02d}:00h"
            st.session_state.poble_seleccionat = nom_poble_real

def update_from_hora_selector():
    st.session_state.avis_selector_default = "--- Selecciona una localitat amb risc ---"

hourly_index = int(st.session_state.hora_seleccionada_str.split(':')[0])
with st.spinner("Analitzant convergències i disparadors a tot el territori..."):
    conv_threshold = -5.5
    localitats_convergencia = encontrar_localitats_con_convergencia(hourly_index, 850, pobles_data, conv_threshold, FORECAST_DAYS)

with st.container(border=True):
    col1,col2=st.columns([1,1],gap="large"); hour_options=[f"{h:02d}:00h" for h in range(24)]
    with col1:
        try: index_hora=hour_options.index(st.session_state.hora_seleccionada_str)
        except ValueError: index_hora=0
        st.selectbox("Hora del pronòstic (Local):",options=hour_options,index=index_hora, key="hora_selector", on_change=update_from_hora_selector)
    with col2:
        p_levels_all=p_levels if p_levels else [1000,925,850,700,500,300]; nivell_global=st.selectbox("Nivell d'anàlisi de mapes:",p_levels_all,index=p_levels_all.index(850) if 850 in p_levels_all else 0)
    
    opcions_avis_formatejades = []
    for poble, hora_avis in avisos_hores.items():
        if poble in localitats_convergencia:
            opcions_avis_formatejades.append(f"🔥 {poble} (Avís a les {hora_avis:02d}:00h)")
        else:
            opcions_avis_formatejades.append(f"🧐 {poble} (Potencial a les {hora_avis:02d}:00h)")
    
    opcions_avis=["--- Selecciona una localitat amb risc ---"] + sorted(list(set(opcions_avis_formatejades)))
    st.selectbox("Localitats amb risc (per a l'hora seleccionada):",options=opcions_avis,key='avis_selector',on_change=update_from_avis_selector, index=0)

st.markdown(f'<p style="text-align: center; font-size: 0.9em; color: grey;">🕒 {get_next_arome_update_time()}</p>', unsafe_allow_html=True)

hourly_index = int(st.session_state.hora_seleccionada_str.split(':')[0])
disparadors_actius = {poble for poble in avisos_hores if poble in localitats_convergencia}

def update_poble_selection():
    """
    Funció callback que s'activa en canviar la selecció del poble.
    Analitza el text del selectbox per obtenir el nom real del poble
    i l'actualitza a l'estat de la sessió. Aquesta versió és més robusta.
    """
    poble_display = st.session_state.poble_selector

    # Primer, eliminem qualsevol informació addicional entre parèntesis
    nom_net = poble_display.split(" (")[0]
    
    # Després, eliminem qualsevol emoji conegut i espais sobrants.
    # Aquest mètode és més segur que simplement reemplaçar un sol emoji.
    nom_net = nom_net.replace("⚠️", "").replace("🔥", "").replace("🧐", "").strip()

    # Finalment, actualitzem l'estat de la sessió amb el nom net
    st.session_state.poble_seleccionat = nom_net

# --- LÒGICA DE SELECCIÓ DE LOCALITAT ---

# Ordena els pobles per a una presentació coherent
sorted_pobles = sorted(pobles_data.keys())

# Crea les opcions per al menú desplegable, afegint una marca als que tenen disparador actiu
opciones_display = [f"⚠️ {p} (Disparador Actiu)" if p in disparadors_actius else p for p in sorted_pobles]

# Intenta trobar l'índex de la selecció actual per mantenir-la consistent
default_index_display = 0
try:
    # Busca la cadena que comença amb el nom del poble seleccionat
    current_selection_display = next(s for s in opciones_display if s.strip().startswith(st.session_state.poble_seleccionat))
    default_index_display = opciones_display.index(current_selection_display)
except (StopIteration, ValueError):
    # Si no es troba, simplement es deixa el valor per defecte (el primer de la llista)
    pass

st.selectbox(
    'Selecciona una localitat:',
    options=opciones_display,
    index=default_index_display,
    key='poble_selector',
    on_change=update_poble_selection,
    help="Les localitats marcades amb ⚠️ tenen un focus de convergència actiu per a l'hora seleccionada, actuant com un possible disparador."
)

# --- PROCESSAMENT I VISUALITZACIÓ DE DADES PER A LA LOCALITAT SELECCIONADA ---

poble_sel = st.session_state.poble_seleccionat
lat_sel, lon_sel = pobles_data[poble_sel]['lat'], pobles_data[poble_sel]['lon']
sondeo = totes_les_dades.get(poble_sel)

if sondeo:
    # ### CANVI ###: S'afegeix un bloc try-except general per capturar qualsevol error inesperat
    # durant el processament o la visualització, evitant que l'aplicació es bloquegi.
    try:
        data_is_valid = False
        parametros = {} # Inicialitzem el diccionari de paràmetres

        with st.spinner(f"Processant dades per a {poble_sel}..."):
            # Processa les dades del sondeig per a l'hora seleccionada
            profiles = processar_sondeig_per_hora(sondeo, hourly_index, p_levels)
            if profiles:
                p, T, Td, u, v, H = profiles
                # Calcula tots els paràmetres meteorològics
                parametros = calculate_parameters(p, T, Td, u, v, H)
                data_is_valid = True

        if data_is_valid:
            is_disparador_active = poble_sel in disparadors_actius

            # --- SECCIÓ D'AVISOS PRINCIPALS ---
            avis_temp_titol, avis_temp_text, avis_temp_color, avis_temp_icona = generar_avis_temperatura(parametros)
            if avis_temp_titol:
                display_avis_principal(avis_temp_titol, avis_temp_text, avis_temp_color, icona_personalitzada=avis_temp_icona)

            avis_conv_titol, avis_conv_text, avis_conv_color = generar_avis_convergencia(parametros, is_disparador_active)
            if avis_conv_titol:
                display_avis_principal(avis_conv_titol, avis_conv_text, avis_conv_color)

            avis_titol, avis_text, avis_color = generar_avis_localitat(parametros, is_disparador_active)
            display_avis_principal(avis_titol, avis_text, avis_color)

            # --- NAVEGACIÓ PER PESTANYES ---
            # ### CANVI ###: S'utilitza st.tabs en lloc de st.radio per una interfície més neta.
            tab_analisi, tab_params, tab_mapes, tab_hodo, tab_sondeig, tab_oro, tab_nuvol = st.tabs([
                "🗨️ Anàlisi", "📊 Paràmetres", "🗺️ Mapes", "🧭 Hodògraf",
                "📍 Sondeig", "🏔️ Orografia", "☁️ Visualització"
            ])

            with tab_analisi:
                st.write_stream(generar_analisi_detallada(parametros))

            with tab_params:
                st.subheader("Paràmetres Clau")
                display_metrics(parametros)

            with tab_mapes:
                # ### CANVI ###: S'implementa la lògica de mapes refactoritzada.
                st.subheader(f"Anàlisi de Mapes a {nivell_global}hPa")

                map_options = {
                    "Vents i Convergència": {"api_variable": "wind"},
                    "Punt de Rosada": {"api_variable": "dewpoint", "titol": "Punt de Rosada", "cmap": "BrBG", "unitat": "°C", "levels": np.arange(-10, 21, 2)},
                    "Humitat Relativa": {"api_variable": "humidity", "titol": "Humitat Relativa", "cmap": "Greens", "unitat": "%", "levels": np.arange(30, 101, 5)}
                }
                selected_map_name = st.selectbox("Selecciona la capa a visualitzar:", map_options.keys())

                with st.spinner(f"Generant mapa de {selected_map_name.lower()}..."):
                    map_config = map_options[selected_map_name]
                    api_var = map_config["api_variable"]
                    lats, lons, data, error = obtener_dades_mapa(api_var, nivell_global, hourly_index, FORECAST_DAYS)

                    if error:
                        st.error(f"Error en obtenir dades del mapa: {error}")
                    elif not lats or len(lats) < 4:
                        st.warning("No hi ha prou dades per generar el mapa.")
                    else:
                        if selected_map_name == "Vents i Convergència":
                            fig = crear_mapa_vents(lats, lons, data, nivell_global, lat_sel, lon_sel, poble_sel)
                        else:
                            fig = crear_mapa_generic(lats, lons, data, nivell_global, lat_sel, lon_sel, poble_sel,
                                                     map_config["titol"], map_config["cmap"],
                                                     map_config["unitat"], map_config["levels"])
                        st.pyplot(fig)

            with tab_hodo:
                st.subheader("Hodògraf (0-10 km)")
                st.pyplot(crear_hodograf(p, u, v, H))

            with tab_sondeig:
                st.subheader(f"Sondeig per a {poble_sel} ({datetime.now(pytz.timezone('Europe/Madrid')).strftime('%d/%m/%Y')} - {hourly_index:02d}:00h Local)")
                st.pyplot(crear_skewt(p, T, Td, u, v))

            with tab_oro:
                st.subheader("Potencial d'Activació per Orografia")
                fig_oro = crear_grafic_orografia(parametros, parametros.get('ZeroIso_AGL', {}).get('value'))
                if fig_oro:
                    st.pyplot(fig_oro)
                else:
                    st.info("No hi ha dades de LCL disponibles per calcular el potencial orogràfic.")

            with tab_nuvol:
                st.subheader("Visualització Conceptual del Núvol")
                with st.spinner("Dibuixant la possible estructura del núvol..."):
                    fig_nuvol = crear_grafic_nuvol(parametros, H, u, v, is_disparador_active)
                    if fig_nuvol:
                        st.pyplot(fig_nuvol)
                    else:
                        st.info("No hi ha dades de LCL o EL disponibles per visualitzar l'estructura del núvol.")
        else:
            st.warning(f"No s'han pogut calcular els paràmetres per a les {hourly_index:02d}:00h. Les dades del model podrien no ser vàlides per a aquesta hora. Proveu amb una altra hora o localitat.")

    except Exception as e:
        # Aquesta excepció captura qualsevol error no previst dins del bloc if sondeo:
        st.error(f"S'ha produït un error inesperat en processar les dades per a '{poble_sel}'.")
        st.exception(e) # st.exception és útil per a depurar, ja que mostra el traceback complet.

else:
    st.error(f"No s'han pogut obtenir dades per a '{poble_sel}'. Pot ser que estigui fora de la cobertura del model AROME o que hi hagi un problema amb la connexió a l'API.")
