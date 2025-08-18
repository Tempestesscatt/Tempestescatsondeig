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

st.set_page_config(
    layout="wide", 
    page_title="Terminal de Temps Sever | Catalunya",
    page_icon="🌪️"
)

# Configuració d'estils
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background-color: #e0e7ff;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5;
        color: white;
    }
    .metric-container { border-radius: 10px; }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: scale(1.05);
    }
    .stSelectbox, .stRadio { background-color: white; border-radius: 8px; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

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
    'Vic': {'lat': 41.9303, 'lon': 2.2546},
    'Manresa': {'lat': 41.725, 'lon': 1.8266},
    'Reus': {'lat': 41.1548, 'lon': 1.1087},
    'Figueres': {'lat': 42.2675, 'lon': 2.9603},
    'Tortosa': {'lat': 40.8125, 'lon': 0.5216},
    'Berga': {'lat': 42.1019, 'lon': 1.8456}
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

# --- 1. FUNCIONS D'OBTENCIÓ I PROCESSAMENT DE DADES ---

@st.cache_data(ttl=3600, show_spinner="Carregant dades atmosfèriques...")
def carregar_dades_sondeig(lat, lon, hourly_index):
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS]
        params = {
            "latitude": lat, 
            "longitude": lon, 
            "hourly": h_base + h_press, 
            "models": "arome_seamless", 
            "forecast_days": FORECAST_DAYS
        }
        
        response = openmeteo.weather_api(API_URL, params=params)[0]
        hourly = response.Hourly()
        
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
        if any(np.isnan(val) for val in sfc_data.values()): 
            return None, "Dades de superfície invàlides."

        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [
                hourly.Variables(var_count + i * len(PRESS_LEVELS) + j
            ).ValuesAsNumpy()[hourly_index] for j in range(len(PRESS_LEVELS))]

        sfc_h = mpcalc.pressure_to_height_std(sfc_data["surface_pressure"] * units.hPa).to('meter').m
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = \
            [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0], [sfc_h]

        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_val = mpcalc.dewpoint_from_relative_humidity(
                    p_data["T"][i] * units.degC, 
                    p_data["RH"][i] * units.percent
                ).m
                Td_profile.append(Td_val)
                u, v = mpcalc.wind_components(
                    p_data["WS"][i] * units('km/h'), 
                    p_data["WD"][i] * units.degrees
                )
                u_profile.append(u.to('m/s').m)
                v_profile.append(v.to('m/s').m)
                h_profile.append(p_data["H"][i])
        
        if len(p_profile) < 4: 
            return None, "Perfil atmosfèric massa curt."

        p = np.array(p_profile) * units.hPa
        T = np.array(T_profile) * units.degC
        Td = np.array(Td_profile) * units.degC
        u = np.array(u_profile) * units('m/s')
        v = np.array(v_profile) * units('m/s')
        h = np.array(h_profile) * units.meter
        
        params_calc = {}
        prof = mpcalc.parcel_profile(p, T[0], Td[0])
        cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0
        params_calc['CIN'] = cin.to('J/kg').m

        s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6 * units.km)
        params_calc['Shear_0-6km'] = mpcalc.wind_speed(s_u, s_v).to('m/s').m
        
        _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=3 * units.km)
        params_calc['SRH_0-3km'] = np.sum(srh).to('m^2/s^2').m

        return ((p, T, Td, u, v, h), params_calc), None
    except Exception as e:
        return None, f"Error en processar dades del sondeig: {str(e)}"

@st.cache_data(ttl=3600, show_spinner="Generant mapes...")
def carregar_dades_mapa(variables, hourly_index):
    try:
        lats, lons = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 15), np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 15)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {
            "latitude": lat_grid.flatten().tolist(), 
            "longitude": lon_grid.flatten().tolist(), 
            "hourly": variables, 
            "models": "arome_seamless", 
            "forecast_days": FORECAST_DAYS
        }
        responses = openmeteo.weather_api(API_URL, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        
        for r in responses:
            vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
            if not any(np.isnan(v) for v in vals):
                output["lats"].append(r.Latitude())
                output["lons"].append(r.Longitude())
                for i, var in enumerate(variables): 
                    output[var].append(vals[i])
                    
        if not output["lats"]: 
            return None, "No s'han rebut dades vàlides."
        return output, None
    except Exception as e:
        return None, f"Error en carregar dades del mapa: {str(e)}"

@st.cache_data(ttl=3600, show_spinner="Preparant anàlisi d'IA...")
def preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    """Recopila dades clau i localitza el focus de convergència."""
    dades_ia = {}
    data_tuple, error_sondeig = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if not data_tuple: 
        return None, f"Falten dades del sondeig ({error_sondeig})"
    dades_ia['sondeig'] = data_tuple[1]

    variables_mapa = ["cape", "relative_humidity_700hPa", "wind_speed_925hPa", "wind_direction_925hPa"]
    map_data, error_mapa = carregar_dades_mapa(variables_mapa, hourly_index_sel)
    if not map_data: 
        return None, f"Falten dades del mapa ({error_mapa})"

    resum_mapa = {}
    if 'cape' in map_data and map_data['cape']: 
        resum_mapa['max_cape_catalunya'] = max(map_data['cape'])
    if 'relative_humidity_700hPa' in map_data and map_data['relative_humidity_700hPa']: 
        resum_mapa['max_rh700_catalunya'] = max(map_data['relative_humidity_700hPa'])
    
    # Localització del focus de convergència
    if 'wind_speed_925hPa' in map_data and map_data['wind_speed_925hPa']:
        try:
            lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
            speeds_kmh = np.array(map_data['wind_speed_925hPa']) * units('km/h')
            dirs_deg = np.array(map_data['wind_direction_925hPa']) * units.degrees
            u_comp, v_comp = mpcalc.wind_components(speeds_kmh, dirs_deg)
            
            grid_lon, grid_lat = np.meshgrid(
                np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 50), 
                np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 50)
            )
            
            grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
            grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
            
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            divergence = mpcalc.divergence(
                grid_u * units('m/s'), 
                grid_v * units('m/s'), 
                dx=dx, 
                dy=dy
            ) * 1e5
            
            resum_mapa['max_conv_925hpa'] = np.nanmin(divergence)
            
            # Localitzem les coordenades del punt de màxima convergència
            idx_min = np.nanargmin(divergence)
            idx_2d = np.unravel_index(idx_min, divergence.shape)
            resum_mapa['lat_max_conv'] = grid_lat[idx_2d]
            resum_mapa['lon_max_conv'] = grid_lon[idx_2d]
        except Exception as e:
            resum_mapa['max_conv_925hpa'] = 0
            resum_mapa['lat_max_conv'] = 0
            resum_mapa['lon_max_conv'] = 0
    
    dades_ia['mapa_resum'] = resum_mapa
    dades_ia['timestamp'] = timestamp_str
    return dades_ia, None

@st.cache_data(ttl=3600, show_spinner="Generant resum d'IA...")
def generar_resum_ia(_dades_ia, _poble_sel):
    """Genera un resum directe i identifica poblacions afectades."""
    if not GEMINI_CONFIGURAT: 
        return "Error: La clau API de Google no està configurada."
        
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    mapa = _dades_ia.get('mapa_resum', {})
    sondeig = _dades_ia.get('sondeig', {})
    timestamp = _dades_ia.get('timestamp', 'Hora desconeguda')

    prompt = f"""
    Ets un assistent de meteorologia directe i concís. La teva única tasca és analitzar les dades del model AROME per a Catalunya i generar un avís curt i clar.

    **DADES:**
    - Hora de l'anàlisi: {timestamp}
    - CAPE màxim (energia): {int(mapa.get('max_cape_catalunya', 0))} J/kg
    - Convergència màxima a 925hPa ("chispa"): {mapa.get('max_conv_925hpa', 0):.2f} (x10⁻⁵ s⁻¹)
    - Latitud del focus de convergència: {mapa.get('lat_max_conv', 0):.2f}
    - Longitud del focus de convergència: {mapa.get('lon_max_conv', 0):.2f}
    - Cisallament 0-6km (organització): {int(sondeig.get('Shear_0-6km', 0))} m/s
    - SRH 0-3km (rotació): {int(sondeig.get('SRH_0-3km', 0))} m²/s²

    **INSTRUCCIONS (MOLT IMPORTANT):**
    Vés directament al gra. Prohibides les frases llargues o explicacions tècniques complexes.

    1.  **Resumeix el Risc:** Comença amb una única frase que defineixi el nivell de risc (Baix, Moderat, Alt, Molt Alt) de tempestes.
    2.  **Anomena les Poblacions Clau:** La teva tasca més important. Utilitzant el teu coneixement geogràfic de Catalunya, identifica 3-5 ciutats o pobles importants a prop de les coordenades del "focus de convergència" que t'he proporcionat. Aquestes són les zones de màxima probabilitat de tempestes.
    3.  **Justificació Breu:** Explica en una frase per què hi ha risc, connectant la "chispa" (convergència) amb el "combustible" (CAPE).

    **FORMAT DE SORTIDA OBLIGATORI (utilitza Markdown):**
    **Resum del Risc:** [La teva frase de resum aquí]
    **Poblacions Potencialment Afectades:** [Llista de 3-5 poblacions separades per comes]
    **Justificació Tècnica (Molt Breu):** [La teva única frase d'explicació aquí]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"S'ha produït un error en contactar amb l'assistent d'IA: {str(e)}"

# --- 2. FUNCIONS DE VISUALITZACIÓ (GRÀFICS I MAPES) ---

def crear_mapa_base():
    fig, ax = plt.subplots(
        figsize=(10, 10), 
        dpi=200, 
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#F0F8FF", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#4682B4', linewidth=1.2, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='#4682B4', zorder=5)
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5, zorder=5)
    return fig, ax

def get_wind_colormap():
    colors = ['#FFFFFF', '#E0F5FF', '#B9E8FF', '#87D7F9', '#5AC7E3', '#2DB8CC', 
              '#3FC3A3', '#5ABF7A', '#75BB51', '#98D849', '#C2E240', '#EBEC38', 
              '#F5D03A', '#FDB43D', '#F7983F', '#E97F41', '#D76643', '#C44E45', 
              '#B23547', '#A22428', '#881015', '#6D002F', '#860057', '#A0007F', 
              '#B900A8', '#D300D0', '#E760E7', '#F6A9F6', '#FFFFFF', '#CCCCCC']
    levels = list(range(0, 95, 5)) + list(range(100, 211, 10))
    cmap = ListedColormap(colors, name='wind_speed_custom')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm, levels

def crear_mapa_500hpa(map_data, timestamp_str):
    fig, ax = crear_mapa_base()
    lons, lats = map_data['lons'], map_data['lats']
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), 
        np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200)
    )
    grid_temp = griddata((lons, lats), map_data['temperature_500hPa'], (grid_lon, grid_lat), method='cubic')
    temp_levels = np.arange(-30, 1, 2)
    cf = ax.contourf(
        grid_lon, grid_lat, grid_temp, 
        levels=temp_levels, cmap='coolwarm', 
        extend='min', alpha=0.7, zorder=2
    )
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label("Temperatura a 500 hPa (°C)")
    cs_temp = ax.contour(
        grid_lon, grid_lat, grid_temp, 
        levels=temp_levels, colors='gray', 
        linewidths=0.8, linestyles='--', zorder=3
    )
    ax.clabel(cs_temp, inline=True, fontsize=7, fmt='%1.0f°C')
    
    u, v = mpcalc.wind_components(
        np.array(map_data['wind_speed_500hPa']) * units('km/h'), 
        np.array(map_data['wind_direction_500hPa']) * units.degrees
    )
    ax.barbs(
        lons[::5], lats[::5], 
        u.to('kt').m[::5], v.to('kt').m[::5], 
        length=5, zorder=6, transform=ccrs.PlateCarree()
    )
    
    ax.set_title(f"Anàlisi a 500 hPa (Temperatura i Vent)\n{timestamp_str}", 
                weight='bold', fontsize=16)
    return fig

def crear_mapa_vents_velocitat(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    cmap, norm, levels = get_wind_colormap()
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), 
        np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200)
    )
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    ax.contourf(
        grid_lon, grid_lat, grid_speed, 
        levels=levels, cmap=cmap, norm=norm, 
        alpha=0.8, zorder=2, extend='max'
    )
    cs_speed = ax.contour(
        grid_lon, grid_lat, grid_speed, 
        levels=np.arange(20, 201, 20), 
        colors='gray', linestyles='--', 
        linewidths=0.8, zorder=3
    )
    ax.clabel(cs_speed, inline=True, fontsize=7, fmt='%1.0f')
    
    speeds_ms = np.array(speed_data) * units('km/h')
    dirs_deg = np.array(dir_data) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    
    rbf_u = Rbf(lons, lats, u_comp.to('m/s').m, function='thin_plate', smooth=0)
    rbf_v = Rbf(lons, lats, v_comp.to('m/s').m, function='thin_plate', smooth=0)
    u_grid = rbf_u(grid_lon, grid_lat)
    v_grid = rbf_v(grid_lon, grid_lat)
    
    ax.streamplot(
        grid_lon, grid_lat, u_grid, v_grid, 
        color='black', linewidth=0.6, 
        density=2.5, arrowsize=0.6, zorder=5
    )
    
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=ax, orientation='vertical', 
        shrink=0.7, ticks=levels[::2]
    )
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", 
                weight='bold', fontsize=16)
    return fig

def crear_mapa_convergencia(lons, lats, speed_data, dir_data, nivell, lat_sel, lon_sel, nom_poble_sel, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), 
        np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200)
    )
    
    speeds_ms = np.array(speed_data) * units('km/h')
    dirs_deg = np.array(dir_data) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    
    rbf_u = Rbf(lons, lats, u_comp.to('m/s').m, function='thin_plate', smooth=0)
    rbf_v = Rbf(lons, lats, v_comp.to('m/s').m, function='thin_plate', smooth=0)
    u_grid = rbf_u(grid_lon, grid_lat)
    v_grid = rbf_v(grid_lon, grid_lat)
    
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(
        u_grid * units('m/s'), 
        v_grid * units('m/s'), 
        dx=dx, dy=dy
    ) * 1e5
    
    levels = np.linspace(-20, 20, 15)
    cf = ax.contourf(
        grid_lon, grid_lat, divergence, 
        levels=levels, cmap='coolwarm_r', 
        alpha=0.6, zorder=2, extend='both'
    )
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label('Convergència (vermell) / Divergència (blau) [x10⁻⁵ s⁻¹]')
    
    cs_conv = ax.contour(
        grid_lon, grid_lat, divergence, 
        levels=levels, colors='black', 
        linewidths=0.7, alpha=0.2, zorder=3
    )
    ax.clabel(cs_conv, inline=True, fontsize=8, fmt='%1.0f')
    
    ax.streamplot(
        grid_lon, grid_lat, u_grid, v_grid, 
        color='black', linewidth=0.5, 
        density=5.0, arrowsize=0.5, zorder=4
    )
    
    ax.plot(
        lon_sel, lat_sel, 'o', 
        markerfacecolor='yellow', 
        markeredgecolor='black', 
        markersize=8, transform=ccrs.Geodetic(), 
        zorder=6
    )
    
    txt = ax.text(
        lon_sel + 0.05, lat_sel, nom_poble_sel, 
        transform=ccrs.Geodetic(), zorder=7, 
        fontsize=10, weight='bold'
    )
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    max_conv = np.nanmin(divergence)
    ax.set_title(f"Flux i Convergència a {nivell}hPa (Mín: {max_conv:.1f})\n{timestamp_str}", 
                weight='bold', fontsize=16)
    return fig

def crear_mapa_escalar(lons, lats, data, titol, cmap, levels, unitat, timestamp_str, extend='max'):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), 
        np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200)
    )
    grid_data = griddata((lons, lats), data, (grid_lon, grid_lat), method='cubic')
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    cf = ax.contourf(
        grid_lon, grid_lat, grid_data, 
        levels=levels, cmap=cmap, norm=norm, 
        alpha=0.8, zorder=2, extend=extend
    )
    
    contorns = ax.contour(
        grid_lon, grid_lat, grid_data, 
        levels=levels[::(len(levels)//5)], 
        colors='black', linewidths=0.7, 
        alpha=0.9, zorder=3
    )
    ax.clabel(contorns, inline=True, fontsize=8, fmt='%1.0f')
    
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=ax, orientation='vertical', 
        shrink=0.7
    )
    cbar.set_label(f"{titol} ({unitat})")
    ax.set_title(f"{titol}\n{timestamp_str}", 
                weight='bold', fontsize=16)
    return fig

def crear_skewt(p, T, Td, u, v, titol):
    fig = plt.figure(figsize=(9, 9), dpi=150)
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.8, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)
    skew.plot(p, T, 'r', lw=2, label='Temperatura')
    skew.plot(p, Td, 'g', lw=2, label='Punt de Rosada')
    
    if np.any(u.magnitude != 0) or np.any(v.magnitude != 0):
        skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03)
    
    skew.plot_dry_adiabats(color='brown', linestyle='--', alpha=0.6)
    skew.plot_moist_adiabats(color='blue', linestyle='--', alpha=0.6)
    skew.plot_mixing_lines(color='green', linestyle='--', alpha=0.6)
    
    try:
        prof = mpcalc.parcel_profile(p, T[0], Td[0])
        skew.plot(p, prof, 'k', linewidth=2, label='Trajectòria Parcel·la')
        skew.shade_cape(p, T, prof, color='red', alpha=0.3)
        skew.shade_cin(p, T, prof, color='blue', alpha=0.3)
    except Exception:
        pass
    
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_title(titol, weight='bold', fontsize=14)
    skew.ax.set_xlabel("Temperatura (°C)")
    skew.ax.set_ylabel("Pressió (hPa)")
    skew.ax.legend()
    return fig

def crear_hodograf(u, v):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    h = Hodograph(ax, component_range=60.)
    h.add_grid(increment=20, color='gray')
    
    if np.any(u.magnitude != 0) or np.any(v.magnitude != 0):
        h.plot(u.to('kt'), v.to('kt'), color='red', linewidth=2)
    
    ax.set_title("Hodògraf", weight='bold')
    return fig
    
def mostrar_imatge_temps_real(tipus):
    if tipus == "Radar":
        url = "https://www.meteociel.fr/cartes_obs/radar/lastradar_sp_ne.gif"
        caption = "Radar de precipitació. Font: Meteociel"
    else:
        now_local = datetime.now(TIMEZONE)
        if now_local.hour >= 22 or now_local.hour < 7:
            url = "https://modeles20.meteociel.fr/satellite/animsatircolmtgsp.gif"
            caption = "Satèl·lit infraroig. Font: Meteociel"
        else:
            url = "https://modeles20.meteociel.fr/satellite/latestsatviscolmtgsp.png"
            caption = "Satèl·lit visible. Font: Meteociel"
    
    try:
        response = requests.get(
            f"{url}?ver={int(time.time())}", 
            headers={'User-Agent': 'Mozilla/5.0'}, 
            timeout=10
        )
        if response.status_code == 200: 
            st.image(response.content, caption=caption, use_container_width=True)
        else: 
            st.warning(f"No s'ha pogut carregar la imatge del {tipus.lower()}. (Codi: {response.status_code})")
    except Exception as e: 
        st.error(f"Error de xarxa en carregar la imatge del {tipus.lower()}: {str(e)}")

# --- 3. LÒGICA DE LA INTERFÍCIE D'USUARI (UI) ---

def ui_capcalera_selectors():
    st.markdown(
        '<h1 style="text-align: center; color: #4f46e5; margin-bottom: 20px;">🌪️ Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.1rem; color: #555;">Eina per a la visualització de paràmetres clau per al pronòstic de convecció</p>', 
        unsafe_allow_html=True
    )
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: 
            st.selectbox(
                "**Ciutat de referència:**", 
                sorted(CIUTATS_CATALUNYA.keys()), 
                key="poble_selector"
            )
        with col2: 
            st.selectbox(
                "**Dia del pronòstic:**", 
                ("Avui", "Demà"), 
                key="dia_selector"
            )
        with col3: 
            st.selectbox(
                "**Hora del pronòstic (Hora Local):**", 
                options=[f"{h:02d}:00h" for h in range(24)], 
                key="hora_selector"
            )

def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    with st.spinner("Actualitzant anàlisi de mapes..."):
        col_map_1, col_map_2 = st.columns([2.5, 1.5])
        with col_map_1:
            map_options = {
                "CAPE (Energia Convectiva)": "cape", 
                "Flux i Convergència": "conv", 
                "Anàlisi a 500hPa": "500hpa", 
                "Vent a 300hPa": "wind_300", 
                "Vent a 700hPa": "wind_700", 
                "Humitat a 700hPa": "rh_700"
            }
            mapa_sel = st.selectbox(
                "**Selecciona la capa del mapa:**", 
                map_options.keys()
            )
            map_key = map_options[mapa_sel]
            
            if map_key == "cape":
                map_data, error_map = carregar_dades_mapa(["cape"], hourly_index_sel)
                if map_data:
                    max_cape = np.max(map_data['cape']) if map_data['cape'] else 0
                    if max_cape <= 500: 
                        cape_levels = np.arange(50, 501, 50)
                    elif max_cape <= 1500: 
                        cape_levels = np.arange(100, 1501, 100)
                    elif max_cape <= 2500: 
                        cape_levels = np.arange(250, 2501, 250)
                    else: 
                        cape_levels = np.arange(250, np.ceil(max_cape / 500) * 500 + 1, 250)
                    st.pyplot(
                        crear_mapa_escalar(
                            map_data['lons'], map_data['lats'], 
                            map_data['cape'], "CAPE", "plasma", 
                            cape_levels, "J/kg", timestamp_str
                        )
                    )
            elif map_key == "conv":
                nivell_sel = st.selectbox(
                    "**Nivell d'anàlisi:**", 
                    options=[1000, 950, 925, 850], 
                    format_func=lambda x: f"{x} hPa"
                )
                variables = [
                    f"wind_speed_{nivell_sel}hPa", 
                    f"wind_direction_{nivell_sel}hPa"
                ]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data: 
                    st.pyplot(
                        crear_mapa_convergencia(
                            map_data['lons'], map_data['lats'], 
                            map_data[variables[0]], map_data[variables[1]], 
                            nivell_sel, lat_sel, lon_sel, 
                            poble_sel, timestamp_str
                        )
                    )
            elif map_key == "500hpa":
                variables = [
                    "temperature_500hPa", 
                    "wind_speed_500hPa", 
                    "wind_direction_500hPa"
                ]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data: 
                    st.pyplot(crear_mapa_500hpa(map_data, timestamp_str))
            elif map_key in ["wind_300", "wind_700"]:
                nivell_hpa = int(map_key.split('_')[1])
                variables = [
                    f"wind_speed_{nivell_hpa}hPa", 
                    f"wind_direction_{nivell_hpa}hPa"
                ]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data: 
                    st.pyplot(
                        crear_mapa_vents_velocitat(
                            map_data['lons'], map_data['lats'], 
                            map_data[variables[0]], map_data[variables[1]], 
                            nivell_hpa, timestamp_str
                        )
                    )
            elif map_key == "rh_700":
                map_data, error_map = carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
                if map_data: 
                    st.pyplot(
                        crear_mapa_escalar(
                            map_data['lons'], map_data['lats'], 
                            map_data['relative_humidity_700hPa'], 
                            "Humitat Relativa a 700hPa", "Greens", 
                            np.arange(50, 101, 5), "%", timestamp_str
                        )
                    )
            if error_map: 
                st.error(f"Error en carregar el mapa: {error_map}")
                
        with col_map_2:
            st.subheader("📡 Imatges en Temps Real")
            view_choice = st.radio(
                "Selecciona la vista:", 
                ("Satèl·lit", "Radar"), 
                horizontal=True, 
                label_visibility="collapsed"
            )
            mostrar_imatge_temps_real(view_choice)

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        st.subheader(f"📊 Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        
        cols = st.columns(4)
        metrics = {
            'CAPE': 'J/kg',
            'CIN': 'J/kg',
            'Shear_0-6km': 'm/s',
            'SRH_0-3km': 'm²/s²'
        }
        
        for i, (param, unit) in enumerate(metrics.items()):
            val = params_calculats.get(param, 0)
            with cols[i]:
                st.metric(
                    label=param, 
                    value=f"{f'{val:.0f}' if val is not None else '---'} {unit}",
                    help={
                        'CAPE': 'Energia per a tempestes (>1000 J/kg significatiu)',
                        'CIN': '"Tapa" que impedeix la convecció',
                        'Shear_0-6km': 'Diferència de vent amb l\'altura (>15-20 m/s afavoreix l\'organització)',
                        'SRH_0-3km': 'Potencial de rotació (>150 m²/s² afavoreix supercèl·lules)'
                    }[param]
                )
        
        with st.expander("ℹ️ Explicació dels paràmetres"):
            st.markdown("""
            - **CAPE (Convective Available Potential Energy):** 
              Mesura l'energia disponible per al desenvolupament de tempestes. 
              Valors superiors a 1000 J/kg indiquen un entorn favorable per a tempestes fortes.
              
            - **CIN (Convective Inhibition):** 
              Representa l'energia que impedeix la formació de tempestes. 
              Valors negatius alts indiquen una "tapa" forta que pot evitar convecció.
              
            - **Shear 0-6km (Cisallament del vent):** 
              Diferència de velocitat i direcció del vent entre la superfície i 6 km d'altura. 
              Valors superiors a 15-20 m/s afavoreixen tempestes organitzades i duradores.
              
            - **SRH 0-3km (Helicitat Relativa a la Tempesta):** 
              Mesura el potencial de rotació en les tempestes. 
              Valors superiors a 150 m²/s² afavoreixen supercèl·lules i possible formació de tornados.
            """)
            
        st.divider()
        col1, col2 = st.columns(2)
        with col1: 
            st.pyplot(
                crear_skewt(
                    sounding_data[0], sounding_data[1], 
                    sounding_data[2], sounding_data[3], 
                    sounding_data[4], f"Sondeig Vertical - {poble_sel}"
                )
            )
        with col2: 
            st.pyplot(crear_hodograf(sounding_data[3], sounding_data[4]))
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    st.subheader(f"🤖 Assistència d'IA per a {poble_sel} - {timestamp_str}")
    
    if not GEMINI_CONFIGURAT:
        st.error("""
            **Funcionalitat no disponible**  
            La clau API de Google no està configurada correctament a `.streamlit/secrets.toml`.
        """)
        return
    
    # Generar un ID únic basat en la selecció actual
    cache_key = f"{poble_sel}_{lat_sel}_{lon_sel}_{hourly_index_sel}_{timestamp_str}"
    
    if st.button("✨ Generar Anàlisi d'IA", key=f"btn_ia_{cache_key}", use_container_width=True):
        with st.spinner("L'assistent d'IA està analitzant les dades..."):
            dades_ia, error = preparar_dades_per_ia(
                poble_sel, lat_sel, lon_sel, 
                hourly_index_sel, timestamp_str
            )
            
            if error:
                st.error(f"No s'ha pogut generar l'anàlisi: {error}")
                return
                
            resum_text = generar_resum_ia(dades_ia, poble_sel)
            
            with st.container(border=True):
                st.markdown(resum_text)

def ui_peu_de_pagina():
    st.divider()
    st.markdown("""
        <p style='text-align: center; font-size: 0.9em; color: grey;'>
            Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | 
            Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | 
            Anàlisi IA per Google Gemini
        </p>
    """, unsafe_allow_html=True)

# --- 4. APLICACIÓ PRINCIPAL ---

def main():
    # Inicialització de l'estat de la sessió
    default_hour = f"{datetime.now(TIMEZONE).hour:02d}:00h"
    if 'poble_selector' not in st.session_state: 
        st.session_state.poble_selector = 'Barcelona'
    if 'dia_selector' not in st.session_state: 
        st.session_state.dia_selector = 'Avui'
    if 'hora_selector' not in st.session_state: 
        st.session_state.hora_selector = default_hour

    ui_capcalera_selectors()

    poble_sel = st.session_state.poble_selector
    dia_sel = st.session_state.dia_selector
    hora_sel = st.session_state.hora_selector
    
    hora_int = int(hora_sel.split(':')[0])
    now_local = datetime.now(TIMEZONE)
    target_date = now_local.date()
    if dia_sel == "Demà": 
        target_date += timedelta(days=1)
    
    local_dt = TIMEZONE.localize
    datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int)
    utc_dt = local_dt.astimezone(pytz.utc)
    
    start_of_today_utc = datetime.now(pytz.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600)
    hourly_index_sel = max(0, time_diff_hours)

    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
    lat_sel = CIUTATS_CATALUNYA[poble_sel]['lat']
    lon_sel = CIUTATS_CATALUNYA[poble_sel]['lon']

    with st.spinner("Carregant dades atmosfèriques..."):
        data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
        if error_msg: 
            st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")

    tab_mapes, tab_vertical, tab_ia = st.tabs([
        "🗺️ Anàlisi de Mapes", 
        "📊 Anàlisi Vertical", 
        "🤖 Assistència d'IA"
    ])

    with tab_mapes: 
        ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    with tab_vertical: 
        ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia: 
        ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
        
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
