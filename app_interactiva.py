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
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai
import io
from scipy.ndimage import label
import streamlit.components.v1 as components
from matplotlib.animation import FuncAnimation
import tempfile
import os
from matplotlib.collections import LineCollection
from scipy.interpolate import interpn

# --- 0. CONFIGURACIÓ I CONSTANTS ---

st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")

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

# --- 1. FUNCIONS D'OBTENCIÓ I PROCESSAMENT DE DADES ---

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
        p_profile, T_profile, Td_profile, u_profile, v_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0]
        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD"]):
                p_profile.append(p_val); T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m)
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
        p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC; Td = np.array(Td_profile) * units.degC
        u = np.array(u_profile) * units('m/s'); v = np.array(v_profile) * units('m/s')
        prof = mpcalc.parcel_profile(p, T[0], Td[0])
        params_calc = {}
        cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0
        params_calc['CIN'] = cin.to('J/kg').m
        try:
            p_lfc, _ = mpcalc.lfc(p, T, Td)
            params_calc['LFC_hPa'] = p_lfc.m if not np.isnan(p_lfc.m) else np.nan
        except Exception: params_calc['LFC_hPa'] = np.nan
        return ((p, T, Td, u, v), params_calc), None
    except Exception as e: return None, f"Error en processar dades del sondeig: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa(variables, hourly_index):
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
        if not output["lats"]: return None, f"No s'han rebut dades vàlides."
        return output, None
    except Exception as e: return None, f"Error en carregar dades del mapa: {e}"

# --- 2. FUNCIONS DE VISUALITZACIÓ ---

def crear_mapa_base():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0); ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5); return fig, ax

def crear_mapa_forecast_combinat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), method='cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
    if nivell >= 950: CONVERGENCE_THRESHOLD = -45; DEWPOINT_THRESHOLD_FOR_RISK = 14
    elif nivell >= 925: CONVERGENCE_THRESHOLD = -35; DEWPOINT_THRESHOLD_FOR_RISK = 12
    elif nivell >= 850: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 7
    elif nivell >= 800: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 5
    else: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 2
    colors_wind_final = ['#FFFFFF', '#B0E0E6', '#00FFFF', '#3CB371', '#32CD32', '#ADFF2F', '#FFD700', '#F4A460', '#CD853F', '#A0522D', '#DC143C', '#8B0000', '#800080', '#FF00FF', '#FFC0CB', '#D3D3D3', '#A9A9A9']
    speed_levels_final = np.arange(0, 171, 10)
    custom_cmap = ListedColormap(colors_wind_final); norm_speed = BoundaryNorm(speed_levels_final, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=speed_levels_final[::2])
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density=5, arrowsize=0.6, zorder=4)
    effective_risk_mask = (divergence.magnitude <= CONVERGENCE_THRESHOLD) & (grid_dewpoint >= DEWPOINT_THRESHOLD_FOR_RISK)
    labels, num_features = label(effective_risk_mask)
    if num_features > 0:
        for i in range(1, num_features + 1):
            points = np.argwhere(labels == i); center_y, center_x = points.mean(axis=0)
            center_lon, center_lat = grid_lon[0, int(center_x)], grid_lat[int(center_y), 0]
            warning_txt = ax.text(center_lon, center_lat, '⚠️', color='yellow', fontsize=15, ha='center', va='center', zorder=8)
            warning_txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
    ax.set_title(f"Forecast: Força del Vent + Focus de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

@st.cache_data(ttl=3600)
def obtenir_mapa_animat_cached(nivell, hourly_index, timestamp_str):
    if nivell >= 950:
        variables = ["dew_point_2m", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        map_data, _ = carregar_dades_mapa(variables, hourly_index)
        if map_data:
            dewpoint_for_calc = map_data['dew_point_2m']; speed_data = map_data[f"wind_speed_{nivell}hPa"]; dir_data = map_data[f"wind_direction_{nivell}hPa"]
    else:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        map_data, _ = carregar_dades_mapa(variables, hourly_index)
        if map_data:
            temp_data = np.array(map_data[f'temperature_{nivell}hPa']) * units.degC; rh_data = np.array(map_data[f'relative_humidity_{nivell}hPa']) * units.percent
            dewpoint_for_calc = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
            speed_data = map_data[f"wind_speed_{nivell}hPa"]; dir_data = map_data[f"wind_direction_{nivell}hPa"]
    if not map_data: return None
    gif_data = generar_gif_animat(map_data['lons'], map_data['lats'], speed_data, dir_data, dewpoint_for_calc, nivell, timestamp_str)
    return gif_data

def generar_gif_animat_optimitzat(_lons, _lats, _speed_data, _dir_data, _dewpoint_data, _nivell, _timestamp_str):
    """
    Funció 'treballadora' optimitzada: Genera un GIF animat amb línies de flux
    amb efecte de traçat, bucle perfecte i millor rendiment.
    """
    # Utilitzem la teva funció per crear el mapa base amb Cartopy
    fig, ax = crear_mapa_base()

    # --- 1. Preparació de Dades i Interpolació ---
    # Augmentem una mica la resolució per a una interpolació més suau
    grid_lon_vals = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 150)
    grid_lat_vals = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 150)
    grid_lon, grid_lat = np.meshgrid(grid_lon_vals, grid_lat_vals)
    
    points = (_lons, _lats) # Dades originals disperses
    
    # Interpolació a una graella densa amb mètode 'cubic' per a més suavitat
    grid_speed = griddata(points, _speed_data, (grid_lon, grid_lat), method='cubic')
    grid_dewpoint = griddata(points, _dewpoint_data, (grid_lon, grid_lat), method='cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(_speed_data) * units('km/h'), np.array(_dir_data) * units.degrees)
    grid_u = griddata(points, u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic', fill_value=0)
    grid_v = griddata(points, v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic', fill_value=0)

    # --- 2. Dibuix del Fons (Mapa de colors i Alertes) ---
    # Aquesta part es manté gairebé idèntica a la teva
    colors_wind_final = ['#FFFFFF', '#B0E0E6', '#00FFFF', '#3CB371', '#32CD32', '#ADFF2F', '#FFD700', '#F4A460', '#CD853F', '#A0522D', '#DC143C', '#8B0000', '#800080', '#FF00FF', '#FFC0CB', '#D3D3D3', '#A9A9A9']
    speed_levels_final = np.arange(0, 171, 10)
    custom_cmap = ListedColormap(colors_wind_final); norm_speed = BoundaryNorm(speed_levels_final, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree(), alpha=0.8)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=speed_levels_final[::2])
    cbar.set_label(f"Velocitat del Vent a {_nivell}hPa (km/h)")
    
    # Lògica de les alertes de convergència
    if _nivell >= 950: CONVERGENCE_THRESHOLD = -45; DEWPOINT_THRESHOLD_FOR_RISK = 14
    elif _nivell >= 925: CONVERGENCE_THRESHOLD = -35; DEWPOINT_THRESHOLD_FOR_RISK = 12
    elif _nivell >= 850: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 7
    else: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 5
    
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
    effective_risk_mask = (divergence.magnitude <= CONVERGENCE_THRESHOLD) & (grid_dewpoint >= DEWPOINT_THRESHOLD_FOR_RISK)
    labels, num_features = label(effective_risk_mask)
    if num_features > 0:
        for i in range(1, num_features + 1):
            points = np.argwhere(labels == i); center_y, center_x = points.mean(axis=0)
            center_lon, center_lat = grid_lon[0, int(center_x)], grid_lat[int(center_y), 0]
            warning_txt = ax.text(center_lon, center_lat, '⚠️', color='yellow', fontsize=15, ha='center', va='center', zorder=8, transform=ccrs.PlateCarree())
            warning_txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    # --- 3. Configuració de l'Animació (El nucli de la millora) ---
    GIF_DURATION_S = 10
    FPS = 25
    NUM_FRAMES = GIF_DURATION_S * FPS
    
    NUM_PARTICLES = 700
    PARTICLE_LIFESPAN = 50  # Fotogrames que una partícula 'viu' abans de reiniciar-se
    PARTICLE_SPEED_FACTOR = 0.0008 # Ajustat per a la projecció PlateCarree
    
    # Inicialització de partícules
    px = np.random.uniform(MAP_EXTENT[0], MAP_EXTENT[1], NUM_PARTICLES)
    py = np.random.uniform(MAP_EXTENT[2], MAP_EXTENT[3], NUM_PARTICLES)
    particle_age = np.random.randint(0, PARTICLE_LIFESPAN, NUM_PARTICLES)
    
    line_collection = LineCollection([], color='black', linewidth=0.9, zorder=4, transform=ccrs.PlateCarree())
    ax.add_collection(line_collection)

    # Funció d'interpolació ràpida per obtenir la velocitat en un punt qualsevol
    points_to_interp = (grid_lat_vals, grid_lon_vals)
    def get_velocity_at_points(lon, lat):
        u = interpn(points_to_interp, grid_u, (lat, lon), method='linear', bounds_error=False, fill_value=0)
        v = interpn(points_to_interp, grid_v, (lat, lon), method='linear', bounds_error=False, fill_value=0)
        return u, v

def update(frame_num):
        nonlocal px, py, particle_age
        
        # Lògica per al bucle perfecte: fade-in i fade-out
        progress = frame_num / NUM_FRAMES
        alpha = np.sin(progress * np.pi) ** 0.8 # L'exponent suavitza la transició
        
        # Moure les partícules
        px_prev, py_prev = px.copy(), py.copy()
        u, v = get_velocity_at_points(px, py)
        px += u * PARTICLE_SPEED_FACTOR
        py += v * PARTICLE_SPEED_FACTOR
        
        # Gestionar 'vida' i reinici de partícules
        particle_age += 1
        reset_mask = (px < MAP_EXTENT[0]) | (px > MAP_EXTENT[1]) | \
                     (py < MAP_EXTENT[2]) | (py > MAP_EXTENT[3]) | \
                     (particle_age > PARTICLE_LIFESPAN)
        
        num_to_reset = np.sum(reset_mask)
        if num_to_reset > 0:
            px[reset_mask] = np.random.uniform(MAP_EXTENT[0], MAP_EXTENT[1], num_to_reset)
            py[reset_mask] = np.random.uniform(MAP_EXTENT[2], MAP_EXTENT[3], num_to_reset)
            particle_age[reset_mask] = 0
            px_prev[reset_mask], py_prev[reset_mask] = px[reset_mask], py[reset_mask]

        # Actualitzar els segments de línia i la seva transparència
        segments = np.array(list(zip(zip(px_prev, py_prev), zip(px, py))))
        line_collection.set_segments(segments)
        line_collection.set_color((0, 0, 0, alpha)) # Negre amb l'alfa variable
        
        return [line_collection]

    ax.set_title(f"Forecast: Flux del Vent + Focus de Convergència a {_nivell}hPa\n{_timestamp_str}", weight='bold', fontsize=16)
    
    # --- 5. Creació, desat i retorn de l'Animació ---
    # Usem blit=True per a un rendiment òptim
    ani = FuncAnimation(fig, update, frames=NUM_FRAMES, blit=True, interval=1000/FPS)
    
    # Adaptat al teu sistema de fitxer temporal per funcionar amb st.cache_data
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        temp_filename = tmpfile.name
        ani.save(temp_filename, writer='pillow', fps=FPS, dpi=150) # Augmentem una mica la qualitat (dpi)
    
    plt.close(fig) # Molt important per alliberar memòria a Streamlit
    
    with open(temp_filename, "rb") as f:
        gif_data = f.read()
    os.remove(temp_filename)
    
    return gif_data



def crear_mapa_vents(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic'); grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    colors_wind_final = ['#FFFFFF', '#B0E0E6', '#00FFFF', '#3CB371', '#32CD32', '#ADFF2F', '#FFD700', '#F4A460', '#CD853F', '#A0522D', '#DC143C', '#8B0000', '#800080', '#FF00FF', '#FFC0CB', '#D3D3D3', '#A9A9A9']
    speed_levels_final = np.arange(0, 171, 10)
    custom_cmap = ListedColormap(colors_wind_final); norm_speed = BoundaryNorm(speed_levels_final, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2)
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, arrowsize=0.6, zorder=3)
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

# --- 3. LÒGICA DE LA INTERFÍCIE D'USUARI ---

def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🌪️ Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Eina per al pronòstic de convecció mitjançant paràmetres clau.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.selectbox("Capital de referència:", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3: st.selectbox("Hora del pronòstic (Hora Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

def ui_explicacio_alertes():
    with st.expander("📖 Què signifiquen les alertes ⚠️ que veig al mapa?"):
        st.markdown("""Cada símbol d'alerta **⚠️** assenyala un **focus de risc convectiu**. No és una predicció de tempesta garantida, sinó la detecció d'una zona on es compleix la **"recepta perfecta"** per iniciar-ne una. El nostre sistema analitza les dades del model i només marca les àrees on es donen **dues condicions clau simultàniament**:""")
        col1, col2 = st.columns(2)
        with col1: st.markdown("#### **1. El Disparador: Convergència ↗️**\nL'aire a nivells baixos està sent forçat a ascendir amb molta intensitat. És el mecanisme que \"dispara\" el moviment vertical necessari per crear un núvol de tempesta (cumulonimbus).")
        with col2: st.markdown("#### **2. El Combustible: Humitat 💧**\nAquest aire que puja no és sec; està carregat de vapor d'aigua (punt de rosada elevat). Aquesta humitat és el \"combustible\" que, en condensar-se, allibera energia i permet que el núvol creixi verticalment.")
        st.info("**En resum:** Una ⚠️ indica una zona on un potent **disparador** està actuant sobre una massa d'aire amb abundant **combustible**. Per tant, són els punts als quals cal prestar més atenció.", icon="🎯")

def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str, data_tuple):
    col_map_1, col_map_2 = st.columns([0.7, 0.3], gap="large")
    with col_map_1:
        map_options = {"Forecast: Vent ANIMAT + Convergència": "forecast_animat", "Forecast: Vent Estàtic + Convergència": "forecast_estatic", "Temperatura i Vent a 500hPa": "500hpa", "Vent a 700hPa (Streamlines)": "vent_700", "Vent a 300hPa (Streamlines)": "vent_300", "Humitat a 700hPa": "rh_700"}
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
        map_key, error_map = map_options[mapa_sel], None
        if map_key in ["forecast_animat", "forecast_estatic"]:
            cin_value = 0; lfc_hpa = np.nan
            if data_tuple and data_tuple[1]: cin_value = data_tuple[1].get('CIN', 0); lfc_hpa = data_tuple[1].get('LFC_hPa', np.nan)
            if cin_value < -25: st.warning(f"**AVÍS DE 'TAPA' (CIN = {cin_value:.0f} J/kg):** El sondeig de **{poble_sel}** mostra una forta inversió. Es necessita un forçament dinàmic potent per trencar-la.")
            if np.isnan(lfc_hpa): st.error("**DIAGNÒSTIC LFC:** No s'ha trobat LFC. L'atmosfera és estable i la convecció espontània és molt improbable.")
            elif lfc_hpa >= 900: st.success(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció de base superficial. **Recomanació: Buscar zones d'alerta ⚠️ a 1000-925 hPa.**")
            elif lfc_hpa >= 750: st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció de base baixa. **Recomanació: Buscar zones d'alerta ⚠️ a 850-800 hPa.**")
            else: st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció elevada. **Recomanació: Buscar zones d'alerta ⚠️ a 700 hPa.**")
            nivell_sel = st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850, 800, 700], format_func=lambda x: f"{x} hPa")
            if map_key == "forecast_animat":
                with st.spinner("Generant animació del flux de vent... Aquesta operació pot trigar uns segons la primera vegada."):
                    gif_data = obtenir_mapa_animat_cached(nivell=nivell_sel, hourly_index=hourly_index_sel, timestamp_str=timestamp_str)
                    if gif_data: st.image(gif_data)
                    else: st.error("No s'han pogut obtenir les dades per generar l'animació.")
                ui_explicacio_alertes()
            else:
                with st.spinner("Carregant dades del model..."):
                    if nivell_sel >= 950:
                        variables = ["dew_point_2m", f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                        map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                        if map_data: dewpoint_for_calc = map_data['dew_point_2m']; speed_data = map_data[f"wind_speed_{nivell_sel}hPa"]; dir_data = map_data[f"wind_direction_{nivell_sel}hPa"]
                    else:
                        variables = [f"temperature_{nivell_sel}hPa", f"relative_humidity_{nivell_sel}hPa", f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                        map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                        if map_data:
                            temp_data = np.array(map_data[f'temperature_{nivell_sel}hPa']) * units.degC; rh_data = np.array(map_data[f'relative_humidity_{nivell_sel}hPa']) * units.percent
                            dewpoint_for_calc = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
                            speed_data = map_data[f"wind_speed_{nivell_sel}hPa"]; dir_data = map_data[f"wind_direction_{nivell_sel}hPa"]
                if map_data: st.pyplot(crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], speed_data, dir_data, dewpoint_for_calc, nivell_sel, timestamp_str)); ui_explicacio_alertes()
        elif map_key == "500hpa":
            variables = ["temperature_500hPa", "wind_speed_500hPa", "wind_direction_500hPa"]; map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_500hpa(map_data, timestamp_str))
        elif map_key == "vent_700":
            nivell = 700; variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]; map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str))
        elif map_key == "vent_300":
            nivell = 300; variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]; map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str))
        elif map_key == "rh_700":
            map_data, error_map = carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_escalar(map_data['lons'], map_data['lats'], map_data['relative_humidity_700hPa'], "Humitat Relativa a 700hPa", "Greens", np.arange(50, 101, 5), "%", timestamp_str))
        if error_map: st.error(f"Error en carregar el mapa: {error_map}")
    with col_map_2:
        st.subheader("Imatges en Temps Real")
        tab_europa, tab_ne = st.tabs(["🇪🇺 Satèl·lit (Europa)", "🛰️ Satèl·lit (NE Península)"])
        with tab_europa: mostrar_imatge_temps_real("Satèl·lit (Europa)")
        with tab_ne: mostrar_imatge_temps_real("Satèl·lit (NE Península)")

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple; st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        cols = st.columns(4)
        for i, (param, unit) in enumerate({'CAPE': 'J/kg', 'CIN': 'J/kg', 'LFC_hPa': 'hPa'}.items()):
            val = params_calculats.get(param); cols[i].metric(label=param, value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} {unit}")
        with st.expander("ℹ️ Què signifiquen aquests paràmetres?"):
            st.markdown("- **CAPE:** Energia per a tempestes. >1000 J/kg és significatiu.\n- **CIN:** \"Tapa\" que impedeix la convecció.\n- **LFC:** Nivell on comença la convecció lliure.")
        st.divider(); col1, col2 = st.columns(2)
        with col1: st.pyplot(crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], f"Sondeig Vertical - {poble_sel}"))
        with col2: st.pyplot(crear_hodograf(sounding_data[3], sounding_data[4]))
    else: st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    st.subheader(f"Assistent d'Anàlisi per IA per a {timestamp_str}")
    if not GEMINI_CONFIGURAT: st.error("Funcionalitat no disponible. La clau API de Google no està configurada correctament."); return
    if st.button("🤖 Generar Anàlisi d'IA", use_container_width=True):
        with st.spinner("L'assistent d'IA està analitzant les dades..."):
            dades_ia, error = preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel)
            if error: st.error(f"No s'ha pogut generar l'anàlisi: {error}"); return
            st.markdown(generar_resum_ia(dades_ia, poble_sel, timestamp_str))

def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- 4. APLICACIÓ PRINCIPAL ---

def main():
    if 'poble_selector' not in st.session_state: st.session_state.poble_selector = 'Barcelona'
    if 'dia_selector' not in st.session_state: st.session_state.dia_selector = 'Avui'
    if 'hora_selector' not in st.session_state: st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"
    ui_capcalera_selectors()
    poble_sel = st.session_state.poble_selector; dia_sel = st.session_state.dia_selector; hora_sel = st.session_state.hora_selector
    hora_int = int(hora_sel.split(':')[0]); now_local = datetime.now(TIMEZONE); target_date = now_local.date()
    if dia_sel == "Demà": target_date += timedelta(days=1)
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int)); utc_dt = local_dt.astimezone(pytz.utc)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600); hourly_index_sel = max(0, time_diff_hours)
    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"; lat_sel = CIUTATS_CATALUNYA[poble_sel]['lat']; lon_sel = CIUTATS_CATALUNYA[poble_sel]['lon']
    data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Resum IA"])
    with tab_mapes: ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str, data_tuple)
    with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia: ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
