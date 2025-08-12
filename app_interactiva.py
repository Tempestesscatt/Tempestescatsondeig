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

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

@st.cache_data(ttl=18000)
def carregar_dades_lot(_chunk_locations):
    p_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"] for p in p_levels]
    chunk_noms, chunk_lats, chunk_lons = zip(*_chunk_locations)
    params = {
        "latitude": list(chunk_lats), "longitude": list(chunk_lons), "hourly": h_base + h_press,
        "models": "arome_france", "timezone": "auto", "forecast_days": FORECAST_DAYS
    }
    intents_restants = 3
    while intents_restants > 0:
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            respostes_chunk = openmeteo.weather_api(url, params=params)
            lot_dict = {chunk_noms[i]: respostes_chunk[i] for i in range(len(respostes_chunk))}
            return lot_dict, p_levels, None
        except openmeteo_requests.exceptions.ApiError as e:
            if "Minutely API request limit exceeded" in str(e):
                intents_restants -= 1; time.sleep(61)
            else: return None, None, str(e)
        except Exception as e: return None, None, str(e)
    return None, None, "S'ha superat el límit de l'API després de diversos intents."

@st.cache_data(ttl=18000)
def obtener_dades_mapa(variable, nivell, hourly_index, forecast_days):
    lats, lons = np.linspace(40.5, 42.8, 12), np.linspace(0.2, 3.3, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    if variable == 'wind':
        api_vars = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
    elif variable == 'dewpoint':
        api_vars = [f"dew_point_{nivell}hPa"]
    elif variable == 'humidity':
        api_vars = [f"relative_humidity_{nivell}hPa"]
    else:
        return None, None, None, f"Variable '{variable}' no reconeguda."

    params = {
        "latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(),
        "hourly": api_vars, "models": "arome_france", "timezone": "auto", "forecast_days": forecast_days
    }
    
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        responses = openmeteo.weather_api(url, params=params)
        lats_out, lons_out, data_out = [], [], []
        
        for r in responses:
            hourly = r.Hourly()
            
            # --- LÍNIA CORREGIDA ---
            # En lloc d'iterar sobre el mètode, iterem sobre el rang del nombre de variables demanades.
            num_vars = len(api_vars)
            values = [hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i in range(num_vars)]
            
            if not any(np.isnan(v) for v in values):
                lats_out.append(r.Latitude())
                lons_out.append(r.Longitude())
                data_out.append(tuple(values) if len(values) > 1 else values[0])

        if not lats_out:
            return None, None, None, "No s'han rebut dades vàlides del model per aquesta zona i hora."
        return lats_out, lons_out, data_out, None
    except Exception as e:
        return None, None, None, str(e)

def get_next_arome_update_time():
    now_utc = datetime.now(pytz.utc)
    run_hours_utc = [0, 6, 12, 18]; availability_delay = timedelta(hours=4)
    next_update_time = None
    for run_hour in run_hours_utc:
        run_datetime = now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        available_time = run_datetime + availability_delay
        if available_time > now_utc:
            next_update_time = available_time; break
    if next_update_time is None:
        tomorrow = now_utc + timedelta(days=1)
        next_update_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0) + availability_delay
    local_tz = pytz.timezone('Europe/Madrid')
    next_update_local = next_update_time.astimezone(local_tz)
    return f"Pròxima actualització de dades (model AROME) estimada a les {next_update_local.strftime('%H:%Mh')}"

def calculate_parameters(p, T, Td, u, v, h):
    params = {}
    def get_val(qty, unit=None):
        try: return qty.to(unit).m if unit else qty.m
        except: return None
    params['SFC_Temp'] = {'value': get_val(T[0], 'degC'), 'units': '°C'}
    raw_cape, raw_cin = None, None
    try:
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]); cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
        raw_cape, raw_cin = get_val(cape, 'J/kg'), get_val(cin, 'J/kg')
        params.update({'CAPE_Brut': {'value': raw_cape, 'units': 'J/kg'}, 'CIN_Fre': {'value': raw_cin, 'units': 'J/kg'}})
        if raw_cape and raw_cape > 0: params['W_MAX'] = {'value': np.sqrt(2 * raw_cape), 'units': 'm/s'}
    except: pass
    if raw_cape is not None and raw_cin is not None: params['CAPE_Utilitzable'] = {'value': max(0, raw_cape - abs(raw_cin)), 'units': 'J/kg'}
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
    try: _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=3*units.km); params['SRH_0-3km'] = {'value': get_val(srh), 'units': 'm²/s²'}
    except: pass
    try: pwat = mpcalc.precipitable_water(p, Td); params['PWAT_Total'] = {'value': get_val(pwat, 'mm'), 'units': 'mm'}
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
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y)
    divergence = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5
    localitats_en_convergencia = set()
    for nom_poble, coords in _localitats.items():
        lon_idx, lat_idx = (np.abs(grid_lon - coords['lon'])).argmin(), (np.abs(grid_lat - coords['lat'])).argmin()
        if divergence.m[lat_idx, lon_idx] < _threshold: localitats_en_convergencia.add(nom_poble)
    return localitats_en_convergencia

@st.cache_data(ttl=18000)
def precalcular_disparadors_actius(_totes_les_dades, _p_levels, _hourly_index, _localitats_convergencia):
    if not _localitats_convergencia: return set()
    disparadors = set()
    for nom_poble in _localitats_convergencia:
        sondeo = _totes_les_dades.get(nom_poble)
        if sondeo:
            profiles = processar_sondeig_per_hora(sondeo, _hourly_index, _p_levels)
            if profiles:
                parametros = calculate_parameters(*profiles)
                cape_u = parametros.get('CAPE_Utilitzable', {}).get('value', 0)
                cin = parametros.get('CIN_Fre', {}).get('value')
                if cape_u > 500 and cin is not None and cin > -50: disparadors.add(nom_poble)
    return disparadors

@st.cache_data(ttl=18000)
def precalcular_avisos_hores(_totes_les_dades, _p_levels):
    avisos_hores = {}; avisos_a_buscar = {"PRECAUCIÓ", "AVÍS", "RISC ALT"}
    for nom_poble, sondeo in _totes_les_dades.items():
        for hora in range(24):
            profiles = processar_sondeig_per_hora(sondeo, hora, _p_levels)
            if profiles:
                parametros = calculate_parameters(*profiles)
                titol_avís, _, _ = generar_avis_localitat(parametros)
                if titol_avís in avisos_a_buscar:
                    avisos_hores[nom_poble] = hora; break
    return dict(sorted(avisos_hores.items()))

# --- 2. FUNCIONS DE VISUALITZACIÓ I FORMAT ---
def display_avis_principal(titol_avís, text_avís, color_avís, icona_personalitzada=None):
    icon_map = {"ESTABLE": "☀️", "RISC BAIX": "☁️", "PRECAUCIÓ": "⚡️", "AVÍS": "⚠️", "RISC ALT": "🌪️", "ALERTA DE DISPARADOR": "🎯"}
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
        if value > 36: color, emoji = "#FF0000", "🔥"
        elif value > 32: color, emoji = "#FF4500", ""
        elif value > 28: color = "#FFA500"
        elif value <= 0: color, emoji = "#0000FF", "🥶"
    elif param_name == 'CIN_Fre':
        if value >= -25: color, emoji = "#32CD32", "✅"
        elif value < -100: color, emoji = "#FF4500", "⚠️"
        elif value < -25: color = "#FFA500"
    elif 'CAPE' in param_name:
        if value > 3500: color, emoji = "#FF00FF", "⚠️"
        elif value > 2500: color, emoji = "#FF4500", "⚠️"
        elif value > 1500: color = "#FFA500"
        elif value > 500: color = "#32CD32"
    elif 'Shear' in param_name:
        if value > 25: color, emoji = "#FF4500", "⚠️"
        elif value > 18: color = "#FFA500"
        elif value > 10: color = "#32CD32"
    elif 'SRH' in param_name:
        if value > 400: color, emoji = "#FF4500", "⚠️"
        elif value > 250: color = "#FFA500"
        elif value > 100: color = "#32CD32"
    elif 'LCL' in param_name:
        if value < 1000: color = "#FFA500"
        elif value < 1500: color = "#32CD32"
    elif 'W_MAX' in param_name:
        if value > 75: color, emoji = "#FF00FF", "⚠️"
        elif value > 50: color, emoji = "#FF4500", "⚠️"
        elif value > 25: color = "#FFA500"
    return color, emoji

def generar_avis_temperatura(params):
    temp = params.get('SFC_Temp', {}).get('value')
    if temp is None: return None, None, None, None
    if temp > 36: return "AVÍS PER CALOR EXTREMA", f"Es preveu una temperatura de {temp:.1f}°C. Risc molt alt. Eviteu l'exposició al sol.", "#FF0000", "🥵"
    if temp < 0: return "AVÍS PER FRED INTENS", f"Es preveu una temperatura de {temp:.1f}°C. Risc de gelades fortes.", "#0000FF", "🥶"
    return None, None, None, None

def generar_avis_localitat(params):
    cape_u=params.get('CAPE_Utilitzable',{}).get('value',0);cin=params.get('CIN_Fre',{}).get('value');shear=params.get('Shear_0-6km',{}).get('value');srh1=params.get('SRH_0-1km',{}).get('value');lcl_agl=params.get('LCL_AGL',{}).get('value',9999);lfc_agl=params.get('LFC_AGL',{}).get('value',9999);w_max=params.get('W_MAX',{}).get('value')
    if cape_u < 100: return "ESTABLE", "Sense risc de tempestes significatives. L'atmosfera és estable.", "#3CB371"
    if cin is not None and cin < -100: return "ESTABLE", "Sense risc de tempestes. La 'tapa' atmosfèrica (CIN) és massa forta.", "#3CB371"
    if lfc_agl > 3000: return "RISC BAIX", "El nivell d'inici de la convecció (LFC) és massa alt i difícil d'assolir.", "#4682B4"
    w_max_text = ""
    if w_max:
        if w_max > 50: w_max_text = " amb corrents ascendents violents"
        elif w_max > 25: w_max_text = " amb corrents ascendents molt forts"
    if shear is not None and shear > 20 and cape_u > 1500 and srh1 is not None and srh1 > 250 and lcl_agl < 1200: return "RISC ALT", f"Condicions per a SUPERCL·LULES{w_max_text}. Potencial de tornados, calamarsa grossa i vent destructiu.", "#DC143C"
    if shear is not None and shear > 18 and cape_u > 1000: return "AVÍS", f"Potencial per a SUPERCL·LULES{w_max_text}. Risc de calamarsa grossa i fortes ratxes de vent.", "#FF8C00"
    if shear is not None and shear > 12 and cape_u > 500: return "PRECAUCIÓ", f"Risc de TEMPESTES ORGANITZADES (multicèl·lules){w_max_text}. Possibles fortes pluges i calamarsa.", "#FFD700"
    return "RISC BAIX", "Possibles xàfecs o tempestes febles i aïllades (unicel·lulars).", "#4682B4"

def generar_avis_convergencia(params, is_convergence_active):
    if not is_convergence_active: return None, None, None
    cape_u=params.get('CAPE_Utilitzable',{}).get('value',0);cin=params.get('CIN_Fre',{}).get('value')
    if cape_u > 500 and (cin is None or cin > -50):
        missatge=f"La forta convergència de vents pot actuar com a disparador. Amb un CAPE de {cape_u:.0f} J/kg i una 'tapa' (CIN) feble, hi ha un alt potencial que les tempestes s'iniciïn de manera explosiva."
        return "ALERTA DE DISPARADOR", missatge, "#FF4500"
    return None, None, None

def generar_analisi_detallada(params):
    def stream_text(text):
        for word in text.split(): yield word + " "; time.sleep(0.02)
        yield "\n\n"
    cape,cin,cape_u=(params.get(k,{}).get('value') for k in ['CAPE_Brut','CIN_Fre','CAPE_Utilitzable']);shear6,srh1=(params.get(k,{}).get('value') for k in ['Shear_0-6km','SRH_0-1km']);lcl_agl,lfc_agl=(params.get(k,{}).get('value') for k in ['LCL_AGL','LFC_AGL']);w_max=params.get('W_MAX',{}).get('value')
    yield from stream_text("### Anàlisi Termodinàmica")
    if cape is None or cape < 100: yield from stream_text("L'atmosfera és estable o quasi estable. El CAPE és pràcticament inexistent."); return
    cape_text="feble" if cape<1000 else "moderada" if cape<2500 else "forta" if cape<3500 else "extrema"; yield from stream_text(f"Tenim un CAPE de {cape:.0f} J/kg, un potencial energètic que indica inestabilitat {cape_text}.")
    if w_max: w_max_kmh=w_max*3.6;w_max_desc="molt forts" if w_max_kmh>90 else "forts";_ = "extremadament violents" if w_max_kmh>180 else w_max_desc; yield from stream_text(f"Això es tradueix en corrents ascendents {_} (~{w_max_kmh:.0f} km/h), un indicador de la potència de la tempesta.")
    if cin is not None:
        if cin < -100: yield from stream_text(f"Factor limitant: La 'tapa' d'inversió (CIN) és molt forta ({cin:.0f} J/kg).")
        elif cin < -25: yield from stream_text(f"La 'tapa' (CIN) de {cin:.0f} J/kg és considerable. Si es trenca, pot donar lloc a un desenvolupament explosiu.")
        else: yield from stream_text("La 'tapa' (CIN) és feble. L'energia està fàcilment disponible.")
    if lfc_agl is not None and lfc_agl > 3000: yield from stream_text(f"Factor limitant: El nivell d'inici de convecció (LFC) està a {lfc_agl:.0f} m, una altura molt elevada.")
    elif lcl_agl is not None and lfc_agl is not None: yield from stream_text(f"La base del núvol (LCL) se situa a {lcl_agl:.0f} m, i el nivell de tret (LFC) a {lfc_agl:.0f} m.")
    yield from stream_text("### Anàlisi Cinemàtica")
    if shear6 is not None:
        if shear6 < 10: shear_text = "Molt feble. Tempestes desorganitzades (unicel·lulars)."
        elif shear6 < 18: shear_text = "Moderat. Potencial per a sistemes multicel·lulars."
        else: shear_text = "Fort. Suficient per suportar supercèl·lules rotatòries."
        yield from stream_text(f"El cisallament 0-6 km (Shear) és de {shear6:.1f} m/s. {shear_text}")
    if srh1 is not None and srh1 > 100:
        srh_text="moderat" if srh1<250 else "fort"; lcl_risk=" Amb la base del núvol baixa, facilita que la rotació arribi a terra." if lcl_agl is not None and lcl_agl<1200 else ""; yield from stream_text(f"L'Helicitat 0-1 km (SRH) és de {srh1:.0f} m²/s², un valor {srh_text} per a la rotació a nivells baixos.{lcl_risk}")
    yield from stream_text("### Síntesi i Riscos Associats")
    if cape_u < 100 or (cin is not None and cin < -100) or (lfc_agl is not None and lfc_agl > 3000): yield from stream_text("Condicions desfavorables per a tempestes significatives.")
    else:
        if lfc_agl is not None and cin is not None and cin < -10: yield from stream_text(f"LA CLAU: Un mecanisme de tret haurà de superar la 'tapa' de {abs(cin):.0f} J/kg i assolir {lfc_agl:.0f} m (LFC) per alliberar l'energia.")
        riscos="calamarsa, fortes ratxes de vent i pluges intenses"
        if shear6 is not None and shear6 > 18 and cape_u > 1000 and srh1 is not None and srh1 > 150:
            if srh1 > 250 and lcl_agl is not None and lcl_agl < 1200: riscos += ", amb risc destacat de tornados"
            yield from stream_text(f"Entorn altament favorable per a supercèl·lules. Risc de {riscos}.")
        elif shear6 is not None and shear6 > 12 and cape_u > 500: yield from stream_text(f"Entorn òptim per a sistemes multicel·lulars. Risc de {riscos}.")
        else: yield from stream_text(f"Entorn favorable per a xàfecs o tempestes unicel·lulars. Risc de {riscos.split(',')[0]}.")

def display_metrics(params_dict):
    param_map = [('Temperatura','SFC_Temp'),('CIN (Fre)','CIN_Fre'),('CAPE (Brut)','CAPE_Brut'),('Shear 0-6km','Shear_0-6km'),('Vel. Asc. Màx.','W_MAX'),('CAPE Utilitzable','CAPE_Utilitzable'),('LCL (AGL)','LCL_AGL'),('LFC (AGL)','LFC_AGL'),('EL (MSL)','EL_MSL'),('SRH 0-1km','SRH_0-1km'),('SRH 0-3km','SRH_0-3km'),('PWAT Total','PWAT_Total')]
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
    """
    Genera un gràfic Skew-T a partir de les dades del perfil atmosfèric.

    Aquesta versió corregida assegura que l'àrea d'inhibició (CIN) es sombregi
    sempre que la parcel·la sigui més freda que l'entorn, independentment
    del valor numèric total del CIN.
    """
    fig = plt.figure(figsize=(7, 9))
    skew = SkewT(fig, rotation=45)
    
    # Dibuixa les línies principals de dades
    skew.plot(p, T, 'r', lw=2, label='Temperatura')
    skew.plot(p, Td, 'b', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), length=7, color='black')
    
    # Dibuixa les línies de referència de l'atmosfera estàndard
    skew.plot_dry_adiabats(color='lightcoral', ls='--', alpha=0.5)
    skew.plot_moist_adiabats(color='cornflowerblue', ls='--', alpha=0.5)
    skew.plot_mixing_lines(color='lightgreen', ls='--', alpha=0.5)
    skew.ax.axvline(0, color='darkturquoise', linestyle='--', label='Isoterma 0°C')

    # Aquesta comprovació assegura que tenim més d'un punt de dades per treballar
    if len(p) > 1:
        try:
            # Calcula el perfil de la parcel·la des de la superfície
            prof = mpcalc.parcel_profile(p, T[0], Td[0])
            skew.plot(p, prof, 'k', lw=2, ls='--', label='Trajectòria de la Parcel·la')
            
            # Calcula els índexs CAPE (energia) i CIN (inhibició)
            cape, cin = mpcalc.cape_cin(p, T, Td, prof)

            # --- LÒGICA DEL SOBREJAT CORREGIDA ---
            
            # 1. Ombreja l'àrea d'energia (CAPE) si n'hi ha.
            #    La funció shade_cape gestiona internament si cape.m > 0.
            skew.shade_cape(p, T, prof, alpha=0.3, color='orange')

            # 2. Ombreja l'àrea d'inhibició (CIN).
            #    S'ha eliminat la condició 'if cin.m != 0:'. Ara, la funció
            #    s'executarà sempre. Si no hi ha cap zona d'inhibició,
            #    simplement no dibuixarà res, que és el comportament desitjat.
            skew.shade_cin(p, T, prof, alpha=0.6, color='gray')

            # Calcula i dibuixa els nivells atmosfèrics clau
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0])
            lfc_p, _ = mpcalc.lfc(p, T, Td, prof)
            el_p, _ = mpcalc.el(p, T, Td, prof)
            
            # Afegeix les línies horitzontals al gràfic si els valors existeixen
            if lcl_p:
                skew.ax.axhline(lcl_p.m, color='purple', linestyle='--', label=f'LCL {lcl_p.m:.0f} hPa')
            if lfc_p:
                skew.ax.axhline(lfc_p.m, color='darkred', linestyle='--', label=f'LFC {lfc_p.m:.0f} hPa')
            if el_p:
                skew.ax.axhline(el_p.m, color='red', linestyle='--', label=f'EL {el_p.m:.0f} hPa')

        except Exception as e:
            # Si hi ha algun error en els càlculs, no fa res per evitar que l'app es trenqui
            # i podria ser útil registrar l'error per a depuració.
            # st.warning(f"No s'ha pogut calcular el perfil complet de la parcel·la: {e}")
            pass

    # Configuració final dels eixos i la llegenda
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_xlabel('Temperatura (°C)')
    skew.ax.set_ylabel('Pressió (hPa)')
    plt.legend()
    
    return fig

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
    lcl_agl = params.get('LCL_AGL', {}).get('value')
    lfc_agl = params.get('LFC_AGL', {}).get('value')

    if lcl_agl is None or np.isnan(lcl_agl):
        return None

    # --- Configuració del gràfic ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 10)
    
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelleft=False, length=5, color='white') 
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_xticks([])

    # --- 1. Gradient del Cel ---
    sky_gradient = np.linspace(0.6, 1.0, 256).reshape(-1, 1)
    sky_gradient = np.vstack((sky_gradient, sky_gradient))
    # S'ajusta l'extensió del cel al límit superior final per evitar talls
    ax.imshow(sky_gradient, aspect='auto', cmap='Blues_r', extent=[0, 10, 0, 14], zorder=0)

    # --- 2. Perfil de la Muntanya Conceptual ---
    has_lfc = lfc_agl is not None and np.isfinite(lfc_agl)
    peak_h_m = lfc_agl if has_lfc and lfc_agl > 1000 else (lcl_agl * 1.5 if lcl_agl > 1000 else 2000)
    peak_h_km = min(peak_h_m / 1000.0, 8.0)

    x_mountain = np.linspace(0, 10, 200)
    y_mountain = peak_h_km * (1 - np.cos(x_mountain * np.pi / 10)) / 2 * (1 - (x_mountain - 5)**2 / 25)
    
    mountain_poly = Polygon(np.vstack([x_mountain, y_mountain]).T, facecolor='#6B8E23', edgecolor='#3A4D14', lw=2, zorder=2)
    ax.add_patch(mountain_poly)
    
    # --- 3. Línies de LCL i LFC (ETIQUETES DINS DEL GRÀFIC) ---
    line_outline_effect = [path_effects.withStroke(linewidth=3.5, foreground='black')]
    
    lcl_km = lcl_agl / 1000
    ax.axhline(lcl_km, color='white', linestyle='--', lw=2, zorder=3, path_effects=line_outline_effect)
    # ETIQUETA MOVUDA A DINS (dreta)
    ax.text(9.9, lcl_km, f"Base del núvol (LCL): {lcl_agl:.0f} m ", color='black', backgroundcolor='white', ha='right', va='center', weight='bold')
    
    cloud_base = patches.Rectangle((0, lcl_km), 10, 0.2, facecolor='white', alpha=0.4, zorder=1)
    ax.add_patch(cloud_base)

    if has_lfc:
        lfc_km = lfc_agl / 1000
        ax.axhline(lfc_km, color='#FFD700', linestyle='--', lw=2.5, zorder=3, path_effects=line_outline_effect)
        # ETIQUETA MOVUDA A DINS (esquerra)
        ax.text(0.1, lfc_km, f" Disparador de tempesta (LFC): {lfc_agl:.0f} m", color='black', backgroundcolor='#FFD700', ha='left', va='center', weight='bold')

    # --- 4. Anotacions i Elements Visuals ---
    main_text_effect = [path_effects.Stroke(linewidth=4, foreground='black'), path_effects.Normal()]

    mountain_lfc_intersect_x = np.interp(lfc_km if has_lfc else peak_h_km, y_mountain, x_mountain)
    
    ax.annotate("", xy=(mountain_lfc_intersect_x, lfc_km if has_lfc else peak_h_km), xytext=(1, 0.1),
                arrowprops=dict(arrowstyle="->", color='white', lw=2.5, linestyle=':', connectionstyle="arc3,rad=0.2"), zorder=4)
    flux_text = ax.text(1.5, 0.4, "Flux d'aire\nforçat a pujar", color='white', weight='bold', ha='center', fontsize=11)
    flux_text.set_path_effects(main_text_effect)

    # --- TEXT D'ANÀLISI FINAL I CLARIFICAT ---
    if has_lfc:
        if peak_h_m >= lfc_agl:
            ax.plot(mountain_lfc_intersect_x, lfc_km, '*', markersize=20, color='yellow', markeredgecolor='red', zorder=5)
            analysis_text_content = f"DISPARADOR OROGRÀFIC POTENCIAL!\nUna muntanya de {lfc_agl:.0f} m o més seria suficient."
            analysis_text = ax.text(5, max(peak_h_km, lfc_km) + 1, analysis_text_content,
                                    ha='center', va='center', fontsize=14, weight='bold', color='white')
            analysis_text.set_path_effects(main_text_effect)
        else:
            ax.plot(x_mountain[np.argmax(y_mountain)], peak_h_km, 'X', markersize=15, color='red', markeredgecolor='white', zorder=5)
            analysis_text_content = f"L'OROGRAFIA NO ÉS SUFICIENT.\nEs necessita una muntanya de {lfc_agl:.0f} m per disparar la tempesta."
            analysis_text = ax.text(5, peak_h_km + 1, analysis_text_content,
                                    ha='center', va='center', color='yellow', weight='bold', fontsize=12)
            analysis_text.set_path_effects(main_text_effect)
    else:
        analysis_text_content = "No hi ha un LFC accessible.\nL'atmosfera és massa estable."
        analysis_text = ax.text(5, 4, analysis_text_content,
                                ha='center', va='center', color='lightblue', weight='bold', fontsize=12)
        analysis_text.set_path_effects(main_text_effect)


    # --- AJUST FINAL DELS LÍMITS (MÉS AJUSTAT) ---
    # S'ha reduït el padding superior de 2.5 a 2.0 per compactar el gràfic
    final_ylim = max(peak_h_km, (lfc_km if has_lfc else 0)) + 2.0
    ax.set_ylim(0, final_ylim)
    
    # --- ETIQUETES DELS EIXOS DINS DEL GRÀFIC ---
    y_label_text = ax.text(0.2, final_ylim - 0.2, "Altitud (km)",
                           ha='left', va='top', color='white', weight='bold', fontsize=12)
    y_label_text.set_path_effects(main_text_effect)

    for y_tick in ax.get_yticks():
        if y_tick > 0 and y_tick < final_ylim: # Assegurem que el tick estigui dins
            tick_label = ax.text(0.15, y_tick, f'{int(y_tick)}',
                                 ha='left', va='center',
                                 color='white', weight='bold', fontsize=9)
            tick_label.set_path_effects(main_text_effect)

    fig.tight_layout(pad=0.5)
    
    return fig

def crear_grafic_nuvol(params, H, u, v, is_convergence_active):
    lcl_agl,el_msl_km,cape,srh1=(params.get(k,{}).get('value') for k in ['LCL_AGL','EL_MSL','CAPE_Brut','SRH_0-1km'])
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
    else: ax.text(0,8,"Sense disparador o energia\nsuficient per a convecció profunda.",ha='center',va='center',color='black',fontsize=16,weight='bold',bbox=dict(facecolor='lightblue',alpha=0.7,boxstyle='round,pad=0.5'))
    barb_heights_km=np.arange(1,16,1); u_barbs=np.interp(barb_heights_km*1000,H.m,u.to('kt').m); v_barbs=np.interp(barb_heights_km*1000,H.m,v.to('kt').m); ax.barbs(np.full_like(barb_heights_km,4.5),barb_heights_km,u_barbs,v_barbs,length=7,color='black'); ax.set_ylim(0,16); ax.set_xlim(-5,5); ax.set_ylabel("Altitud (km, MSL)"); ax.set_title("Visualització del Núvol",weight='bold'); ax.set_xticks([]); ax.grid(axis='y',linestyle='--',alpha=0.3); return fig

# --- 3. INTERFÍCIE I FLUX PRINCIPAL DE L'APLICACIÓ ---
st.markdown('<h1 style="text-align: center; color: #FF4B4B;">⚡ Tempestes.cat</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Eina d\'Anàlisi i Previsió de Fenòmens Severs a Catalunya</p>', unsafe_allow_html=True)

totes_les_dades={}; p_levels=[]; error_carrega=None; locations_data=[(k,v['lat'],v['lon']) for k,v in pobles_data.items()]; chunk_size=50; chunks=list(chunker(locations_data,chunk_size)); num_chunks=len(chunks); progress_bar=st.progress(0,text="Iniciant la càrrega de dades...")
for i,chunk in enumerate(chunks):
    progress_bar.progress((i)/num_chunks,text=f"Carregant lot {i+1} de {num_chunks}...")
    dades_lot,p_levels_lot,error_lot=carregar_dades_lot(tuple(chunk))
    if error_lot: error_carrega=error_lot; break
    totes_les_dades.update(dades_lot)
    if not p_levels: p_levels=p_levels_lot
progress_bar.empty()
if error_carrega: st.error(f"No s'ha pogut carregar la informació base. L'aplicació no pot continuar. Error: {error_carrega}"); st.stop()
st.toast("Dades de sondeig carregades correctament!",icon="✅")

with st.spinner("Buscant hores amb avisos a tot el territori..."): avisos_hores=precalcular_avisos_hores(totes_les_dades,p_levels)

def update_from_avis_selector():
    poble_avis_format=st.session_state.avis_selector
    if "(Avís a les" in poble_avis_format:
        nom_poble_real=poble_avis_format.split(' (')[0]
        if nom_poble_real in avisos_hores:
            hora_avis=avisos_hores[nom_poble_real]; st.session_state.hora_seleccionada_str=f"{hora_avis:02d}:00h"; st.session_state.poble_seleccionat=nom_poble_real

with st.container(border=True):
    col1,col2=st.columns([1,1],gap="large"); hour_options=[f"{h:02d}:00h" for h in range(24)]
    with col1:
        try: index_hora=hour_options.index(st.session_state.hora_seleccionada_str)
        except ValueError: index_hora=0
        st.session_state.hora_seleccionada_str=st.selectbox("Hora del pronòstic (Local):",options=hour_options,index=index_hora)
    with col2:
        p_levels_all=p_levels if p_levels else [1000,925,850,700,500,300]; nivell_global=st.selectbox("Nivell d'anàlisi de mapes:",p_levels_all,index=p_levels_all.index(850))
    if avisos_hores:
        opcions_avis_formatejades=[f"{poble} (Avís a les {hora:02d}:00h)" for poble,hora in avisos_hores.items()]; opcions_avis=["--- 🔥 Selecciona una localitat amb avís per anar-hi directament ---"]+opcions_avis_formatejades; st.selectbox("Localitats amb previsió d'avís:",options=opcions_avis,key='avis_selector',on_change=update_from_avis_selector)

st.markdown(f'<p style="text-align: center; font-size: 0.9em; color: grey;">🕒 {get_next_arome_update_time()}</p>',unsafe_allow_html=True)
hourly_index=int(st.session_state.hora_seleccionada_str.split(':')[0])

with st.spinner("Analitzant convergències i disparadors a tot el territori..."):
    conv_threshold=-5.5; localitats_convergencia=encontrar_localitats_con_convergencia(hourly_index,nivell_global,pobles_data,conv_threshold,FORECAST_DAYS); disparadors_actius=precalcular_disparadors_actius(totes_les_dades,p_levels,hourly_index,localitats_convergencia)

def update_poble_selection():
    poble_display=st.session_state.poble_selector; st.session_state.poble_seleccionat=poble_display.replace("⚠️⛈️ ","").replace(" (disparador actiu)","")

sorted_pobles=sorted(pobles_data.keys()); opciones_display=[f"⚠️⛈️ {p} (disparador actiu)" if p in disparadors_actius else p for p in sorted_pobles]
try:
    current_selection_display=next(s for s in opciones_display if st.session_state.poble_seleccionat in s); default_index_display=opciones_display.index(current_selection_display)
except (StopIteration,ValueError): default_index_display=0
st.selectbox('Selecciona una localitat:',options=opciones_display,index=default_index_display,key='poble_selector',on_change=update_poble_selection)
poble_sel=st.session_state.poble_seleccionat; lat_sel,lon_sel=pobles_data[poble_sel]['lat'],pobles_data[poble_sel]['lon']

sondeo=totes_les_dades.get(poble_sel)
if sondeo:
    data_is_valid=False
    with st.spinner(f"Processant dades per a {poble_sel}..."):
        profiles=processar_sondeig_per_hora(sondeo,hourly_index,p_levels)
        if profiles:
            p,T,Td,u,v,H=profiles; parametros=calculate_parameters(p,T,Td,u,v,H); zero_iso_h_agl=None
            try:
                T_c,H_m=T.to('degC').m,H.to('m').m
                if (idx:=np.where(np.diff(np.sign(T_c)))[0]).size>0:
                    h_zero_iso_msl=np.interp(0,[T_c[idx[0]+1],T_c[idx[0]]],[H_m[idx[0]+1],H_m[idx[0]]]); zero_iso_h_agl=(h_zero_iso_msl-H_m[0])*units.m
            except: pass
            data_is_valid=True
    if data_is_valid:
        avis_temp_titol,avis_temp_text,avis_temp_color,avis_temp_icona=generar_avis_temperatura(parametros)
        if avis_temp_titol: display_avis_principal(avis_temp_titol,avis_temp_text,avis_temp_color,icona_personalitzada=avis_temp_icona)
        is_conv_active=poble_sel in localitats_convergencia; avis_conv_titol,avis_conv_text,avis_conv_color=generar_avis_convergencia(parametros,is_conv_active)
        if avis_conv_titol: display_avis_principal(avis_conv_titol,avis_conv_text,avis_conv_color)
        avis_titol,avis_text,avis_color=generar_avis_localitat(parametros); display_avis_principal(avis_titol,avis_text,avis_color)
        
        tab_list=["🗨️ Anàlisi en Directe","📊 Paràmetres","🗺️ Mapes","🧭Hodògraf","📍Sondeig","🏔️ Orografia","☁️ Visualització"]
        selected_tab=st.radio("Navegació:",tab_list,index=0,horizontal=True,key="main_tabs")
        
        if selected_tab == "🗨️ Anàlisi en Directe": st.write_stream(generar_analisi_detallada(parametros))
        elif selected_tab == "📊 Paràmetres": st.subheader("Paràmetres Clau"); display_metrics(parametros)
        elif selected_tab == "🗺️ Mapes":
            st.subheader(f"Anàlisi de Mapes a {nivell_global}hPa")
            map_type=st.selectbox("Selecciona la capa a visualitzar:",("Vents i Convergència","Punt de Rosada","Humitat Relativa"))
            with st.spinner(f"Generant mapa de {map_type.lower()}..."):
                if map_type == "Vents i Convergència":
                    lats,lons,data,error=obtener_dades_mapa('wind',nivell_global,hourly_index,FORECAST_DAYS)
                    if error: st.error(f"Error en obtenir dades del mapa: {error}")
                    elif lats and len(lats)>4: st.pyplot(crear_mapa_vents(lats,lons,data,nivell_global,lat_sel,lon_sel,poble_sel))
                    else: st.warning("No hi ha prou dades per generar el mapa.")
                else:
                    var_map={"Punt de Rosada":"dewpoint","Humitat Relativa":"humidity"}
                    var_details={"dewpoint":{"titol":"Punt de Rosada","cmap":"BrBG","unitat":"°C","levels":np.arange(-10,21,2)},"humidity":{"titol":"Humitat Relativa","cmap":"Greens","unitat":"%","levels":np.arange(30,101,5)}}
                    var=var_map[map_type]; details=var_details[var]; lats,lons,data,error=obtener_dades_mapa(var,nivell_global,hourly_index,FORECAST_DAYS)
                    if error: st.error(f"Error en obtenir dades del mapa: {error}")
                    elif lats:
                        fig=crear_mapa_generic(lats,lons,data,nivell_global,lat_sel,lon_sel,poble_sel,details["titol"],details["cmap"],details["unitat"],details["levels"]); st.pyplot(fig)
                    else: st.warning("No hi ha prou dades per generar el mapa.")
        elif selected_tab == "🧭Hodògraf": st.subheader("Hodògraf (0-10 km)"); st.pyplot(crear_hodograf(p,u,v,H))
        elif selected_tab == "📍Sondeig": st.subheader(f"Sondeig per a {poble_sel} ({datetime.now(pytz.timezone('Europe/Madrid')).strftime('%d/%m/%Y')} - {hourly_index:02d}:00h Local)"); st.pyplot(crear_skewt(p,T,Td,u,v))
        elif selected_tab == "🏔️ Orografia": 
            st.subheader("Potencial d'Activació per Orografia")
            fig_oro=crear_grafic_orografia(parametros,zero_iso_h_agl)
            if fig_oro:
                st.pyplot(fig_oro)
            else:
                st.info("No hi ha LCL per calcular el potencial orogràfic.")
        
        elif selected_tab == "☁️ Visualització":
            with st.spinner("Dibuixant la possible estructura del núvol..."):
                fig_nuvol = crear_grafic_nuvol(parametros, H, u, v, is_convergence_active=is_conv_active)
                if fig_nuvol:
                    st.pyplot(fig_nuvol)
                else:
                    st.info("No hi ha LCL o EL per visualitzar l'estructura del núvol.")

    else:
        st.warning(f"No s'han pogut calcular els paràmetres per a les {hourly_index:02d}:00h. Proveu amb una altra hora o localitat.")
else:
    st.error(f"No s'han pogut obtenir dades per a '{poble_sel}'. Pot ser que estigui fora de la cobertura del model AROME o que hi hagi un problema amb la connexió.")
