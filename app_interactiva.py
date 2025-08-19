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
from datetime import datetime, timedelta, timezone
import pytz
from scipy.ndimage import label
import google.generativeai as genai
import geopandas as gpd
from shapely.geometry import Point
from collections import Counter
import streamlit_authenticator as stauth
import sqlite3
from streamlit_autorefresh import st_autorefresh

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

@st.cache_data(ttl=86400)
def carregar_mapa_provincies():
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/spain-provinces.geojson"
    gdf = gpd.read_file(url)
    return gdf[gdf['name'].isin(['Barcelona', 'Tarragona', 'Lleida', 'Girona'])]
PROVINCIES_GDF = carregar_mapa_provincies()
DB_FILE = "users.db"

# --- GESTIÓ DE LA BASE DE DADES ---
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, name TEXT, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()

def get_users_from_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, name, password FROM users")
    users = c.fetchall()
    conn.close()
    credentials = {'usernames': {}}
    for user in users:
        credentials['usernames'][user[0]] = {'name': user[1], 'password': user[2]}
    return credentials

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

# --- FUNCIONS D'OBTENCIÓ DE DADES ---
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

@st.cache_data(ttl=3600)
def carregar_dades_mapa_base(variables, hourly_index):
    try:
        lats, lons = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 12), np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
        responses = openmeteo.weather_api(API_URL, params=params)
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
def carregar_dades_mapa(nivell, hourly_index):
    try:
        if nivell >= 950:
            variables = ["dew_point_2m", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base(variables, hourly_index)
            if error: return None, error
            lons, lats, speed_data, dir_data, dewpoint_data = map_data_raw['lons'], map_data_raw['lats'], map_data_raw[f"wind_speed_{nivell}hPa"], map_data_raw[f"wind_direction_{nivell}hPa"], map_data_raw['dew_point_2m']
        else:
            variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base(variables, hourly_index)
            if error: return None, error
            lons, lats, speed_data, dir_data = map_data_raw['lons'], map_data_raw['lats'], map_data_raw[f"wind_speed_{nivell}hPa"], map_data_raw[f"wind_direction_{nivell}hPa"]
            temp_data, rh_data = np.array(map_data_raw[f'temperature_{nivell}hPa']) * units.degC, np.array(map_data_raw[f'relative_humidity_{nivell}hPa']) * units.percent
            dewpoint_data = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
        grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100))
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
        grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'linear')
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)
        convergence_scaled = divergence.magnitude * -1e5
        CONV_THRESHOLD, DEW_THRESHOLD = (20, 14) if nivell >= 950 else (15, 7)
        effective_risk_mask = (convergence_scaled >= CONV_THRESHOLD) & (grid_dewpoint >= DEW_THRESHOLD)
        labels, num_features = label(effective_risk_mask)
        locations = []
        if num_features > 0:
            for i in range(1, num_features + 1):
                points = np.argwhere(labels == i); center_y, center_x = points.mean(axis=0)
                center_lon, center_lat = grid_lon[0, int(center_x)], grid_lat[int(center_y), 0]
                p = Point(center_lon, center_lat)
                for _, prov in PROVINCIES_GDF.iterrows():
                    if prov.geometry.contains(p): locations.append(prov['name']); break
        output_data = {'lons': lons, 'lats': lats, 'speed_data': speed_data, 'dir_data': dir_data, 'dewpoint_data': dewpoint_data, 'alert_locations': locations}
        return output_data, None
    except Exception as e: return None, f"Error en processar dades del mapa: {e}"

# --- FUNCIONS DE VISUALITZACIÓ ---
def crear_mapa_base():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0); ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5); PROVINCIES_GDF.plot(ax=ax, edgecolor='black', facecolor='none', alpha=0, transform=ccrs.PlateCarree())
    return fig, ax

def crear_mapa_forecast_combinat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_speed, grid_dewpoint = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    colors_wind_final = ['#FFFFFF', '#B0E0E6', '#00FFFF', '#3CB371', '#32CD32', '#ADFF2F', '#FFD700', '#F4A460', '#CD853F', '#A0522D', '#DC143C', '#8B0000', '#800080', '#FF00FF', '#FFC0CB', '#D3D3D3', '#A9A9A9']
    speed_levels_final = np.arange(0, 171, 10)
    custom_cmap = ListedColormap(colors_wind_final); norm_speed = BoundaryNorm(speed_levels_final, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=speed_levels_final[::2])
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density=4, arrowsize=0.7, zorder=4, transform=ccrs.PlateCarree())
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)
    convergence_scaled = divergence.magnitude * -1e5
    CONVERGENCE_THRESHOLD = 20
    if nivell >= 950: DEWPOINT_THRESHOLD = 14
    elif nivell >= 925: DEWPOINT_THRESHOLD = 12
    else: DEWPOINT_THRESHOLD = 7
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD, convergence_scaled, 0)
    max_convergence = np.nanmax(convergence_in_humid_areas)
    if max_convergence >= CONVERGENCE_THRESHOLD:
        level1, level2 = max_convergence * 0.60, max_convergence * 0.80
        contour_levels = [lvl for lvl in [level1, level2] if lvl >= CONVERGENCE_THRESHOLD]
        if contour_levels:
            contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=contour_levels, colors='darkred', linestyles='--', linewidths=1.5, zorder=6, transform=ccrs.PlateCarree())
            ax.clabel(contours, inline=True, fontsize=10, fmt='%1.0f')
    ax.set_title(f"Anàlisi de Vent i Nuclis de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_mapa_vents(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u, grid_v = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'), griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    colors_wind_final = ['#FFFFFF', '#B0E0E6', '#00FFFF', '#3CB371', '#32CD32', '#ADFF2F', '#FFD700', '#F4A460', '#CD853F', '#A0522D', '#DC143C', '#8B0000', '#800080', '#FF00FF', '#FFC0CB', '#D3D3D3', '#A9A9A9']
    speed_levels_final = np.arange(0, 171, 10)
    custom_cmap = ListedColormap(colors_wind_final); norm_speed = BoundaryNorm(speed_levels_final, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, arrowsize=0.6, zorder=3, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, ticks=speed_levels_final[::2])
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

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

# --- FUNCIONS PER A L'ASSISTENT D'IA ---
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
    elif param_name == 'LFC_hPa':
        if value > 900: return "#39FF14"
        if value > 800: return "#FF3131"
        return "#BC13FE"
    elif param_name == 'Shear 0-1km':
        if value < 5: return "#808080"
        if value < 15: return "#39FF14"
        if value < 25: return "#FF3131"
        return "#BC13FE"
    elif param_name == 'Shear 0-6km':
        if value < 20: return "#808080"
        if value < 35: return "#39FF14"
        if value < 50: return "#FF3131"
        return "#BC13FE"
    return "#FFFFFF"

def preparar_resum_dades_per_ia(data_tuple, map_data, nivell_mapa, poble_sel, timestamp_str):
    # [ ... EL TEU PROMPT MESTRE VA AQUÍ ... ]
    # Enganxa aquí la teva última versió de la funció `preparar_resum_dades_per_ia`
    # Per seguretat, poso una versió bàsica que no fallarà
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
    historial_formatat = []
    for missatge in historial_conversa:
        role = 'user' if missatge['role'] == 'user' else 'model'
        historial_formatat.append({'role': role, 'parts': [missatge['content']]})
    chat = model.start_chat(history=historial_formatat)
    prompt_final = resum_dades + f"\n\nPREGUNTA ACTUAL DE L'USUARI:\n'{prompt_usuari}'"
    try:
        response = chat.send_message(prompt_final, stream=True)
        for chunk in response:
            yield chunk.text
    except Exception as e:
        print(f"ERROR DETALLAT DE L'API DE GOOGLE: {e}")
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

def ui_explicacio_alertes():
    with st.expander("Què signifiquen les isòlines de convergència?"):
        text_lines = [
            "Les línies vermelles discontínues (`---`) marquen zones de **convergència d'humitat**. Són els **disparadors** potencials de tempestes.",
            "", "- **Què són?** Àrees on el vent força l'aire humit a ajuntar-se i ascendir.",
            "", "- **Com interpretar-les?** El número sobre la línia indica la seva intensitat (més alt = més fort). Valors > 20 són significatius."
        ]
        full_text = "\n".join(text_lines)
        st.markdown(full_text)

def ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple):
    col_map_1, col_map_2 = st.columns([0.7, 0.3], gap="large")
    with col_map_1:
        map_options = {"Anàlisi de Vent i Convergència": "forecast_estatic", "Vent a 700hPa": "vent_700", "Vent a 300hPa": "vent_300"}
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
        map_key = map_options[mapa_sel]
        if map_key == "forecast_estatic":
            if data_tuple and data_tuple[1]:
                cin_value, lfc_hpa = data_tuple[1].get('CIN', 0), data_tuple[1].get('LFC_hPa', np.nan)
                if cin_value < -25: st.warning(f"**AVÍS DE 'TAPA' (CIN = {cin_value:.0f} J/kg):** El sondeig mostra una forta inversió.")
                if np.isnan(lfc_hpa): st.error("**DIAGNÒSTIC LFC:** No s'ha trobat LFC. Atmosfera estable.")
                elif lfc_hpa >= 900: st.success(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció superficial. Recomanació: Anàlisi a 1000-925 hPa.")
                elif lfc_hpa >= 750: st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció baixa. Recomanació: Anàlisi a 850-800 hPa.")
                else: st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció elevada. Recomanació: Anàlisi a 700 hPa.")
            nivell_sel = st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa")
            with progress_placeholder.container():
                progress_bar = st.progress(0, text="Carregant dades del model...")
                map_data, error_map = carregar_dades_mapa(nivell_sel, hourly_index_sel)
                if not error_map:
                    progress_bar.progress(50, text="Generant visualització del mapa...")
            if error_map: 
                st.error(f"Error en carregar el mapa: {error_map}"); progress_placeholder.empty()
            elif map_data:
                st.pyplot(crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str))
                with progress_placeholder.container():
                    progress_bar.progress(100, text="Completat!")
                    time.sleep(1); progress_bar.empty()
                ui_explicacio_alertes()
        elif map_key in ["vent_700", "vent_300"]:
            nivell = 700 if map_key == "vent_700" else 300
            variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data, error_map = carregar_dades_mapa_base(variables, hourly_index_sel)
            if error_map: st.error(f"Error en carregar el mapa: {error_map}")
            elif map_data: st.pyplot(crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str))
    with col_map_2:
        st.subheader("Imatges en Temps Real")
        tab_europa, tab_ne = st.tabs(["Europa", "NE Peninsula"])
        with tab_europa: mostrar_imatge_temps_real("Satèl·lit (Europa)")
        with tab_ne: mostrar_imatge_temps_real("Satèl·lit (NE Península)")

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
    st.markdown("Fes-me preguntes sobre el potencial de temps sever combinant les dades del sondeig i del mapa.")
    nivell_mapa_ia = st.selectbox("Nivell del mapa per a l'anàlisi de l'IA:", options=[1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa", key="ia_level_selector")
    if not GEMINI_CONFIGURAT:
        st.error("Funcionalitat no disponible.")
        return
    map_data_ia, _ = carregar_dades_mapa(nivell_mapa_ia, hourly_index_sel)
    if not data_tuple and not map_data_ia:
        st.warning("No hi ha dades disponibles per analitzar.")
        return
    resum_dades = preparar_resum_dades_per_ia(data_tuple, map_data_ia, nivell_mapa_ia, poble_sel, timestamp_str)
    
    if "messages" not in st.session_state: st.session_state.messages = []
    
    # Dibuixa l'historial de missatges
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
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- APLICACIÓ PRINCIPAL ---
def app_principal():
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f"Benvingut, **{st.session_state['name']}**!")
    with col2:
        st.session_state.authenticator.logout('Tancar Sessió', 'main', key='logout')
    
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
    
    tab_ia, tab_mapes, tab_vertical, tab_chat = st.tabs(["Assistent MeteoIA", "Anàlisi de Mapes", "Anàlisi Vertical", "Xat de la Comunitat"])
    
    with tab_ia:
        ui_pestanya_ia(data_tuple, hourly_index_sel, poble_sel, timestamp_str)
    with tab_mapes: 
        ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
    with tab_vertical: 
        ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_chat:
        ui_pestanya_chat()
        
    ui_peu_de_pagina()

def main():
    setup_database()

    credentials = get_users_from_db()
    authenticator = stauth.Authenticate(credentials, "TempestesCatCookie", "abcdefg", cookie_expiry_days=30)
    st.session_state.authenticator = authenticator

    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None

    if not st.session_state["authentication_status"]:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.title("Benvingut a Tempestes.cat")
            
            choice = st.selectbox("Acció:", ["Iniciar Sessió", "Registrar-se"])

            if choice == "Iniciar Sessió":
                name, authentication_status, username = authenticator.login('main')
                if authentication_status == False: st.error('Nom d\'usuari o contrasenya incorrecta')
                elif authentication_status == None: st.warning('Si us plau, introdueix el teu usuari i contrasenya')
                
            elif choice == "Registrar-se":
                try:
                    if authenticator.register_user('Formulari de Registre', preauthorization=False):
                        new_username_data = authenticator.credentials['usernames']
                        last_user = list(new_username_data.keys())[-1]
                        last_user_data = new_username_data[last_user]
                        
                        conn = sqlite3.connect(DB_FILE)
                        c = conn.cursor()
                        hashed_password = stauth.Hasher([last_user_data['password']]).generate()[0]
                        c.execute("INSERT INTO users (username, name, password) VALUES (?, ?, ?)", 
                                  (last_user, last_user_data['name'], hashed_password))
                        conn.commit()
                        conn.close()
                        st.success('Usuari registrat correctament! Ara pots anar a "Iniciar Sessió".')
                except Exception as e:
                    st.error(e)
    
    if st.session_state["authentication_status"]:
        app_principal()

if __name__ == "__main__":
    main()
