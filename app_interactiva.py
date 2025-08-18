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


# --- 0. CONFIGURACIÓ, CONSTANTS I ESTIL ---

st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya", page_icon="🌪️")

# INJECCIÓ DE CSS PER A L'ESTIL FUTURISTA
st.markdown("""
<style>
    /* Tema General Fosc */
    html, body, [class*="st-"] {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Títols amb efecte neó */
    h1, h2, h3 {
        color: #00BFFF; /* DeepSkyBlue */
    }
    h1 {
        text-shadow: 0 0 8px rgba(0, 191, 255, 0.7);
    }
    /* Contenidors i selectors amb vores subtils */
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-z5fcl4 {
        border: 1px solid #2A3B4C;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #161B22;
    }
    /* Botons amb estil de terminal */
    .stButton>button {
        border: 2px solid #00BFFF;
        border-radius: 8px;
        color: #00BFFF;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00BFFF;
        color: #0E1117;
        box-shadow: 0 0 15px #00BFFF;
    }
    /* Pestanyes (Tabs) amb accent magenta */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF00FF; /* Magenta */
        border-bottom: 3px solid #FF00FF;
    }
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
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734}, 'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200}, 'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

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
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = \
            [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0], [sfc_h]

        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m)
                v_profile.append(v.to('m/s').m)
                h_profile.append(p_data["H"][i])
        
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."

        p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC; Td = np.array(Td_profile) * units.degC
        u = np.array(u_profile) * units('m/s'); v = np.array(v_profile) * units('m/s'); h = np.array(h_profile) * units.meter
        
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
        return None, f"Error en processar dades del sondeig: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa(variables, hourly_index):
    try:
        # --- CORRECCIÓ CLAU ---
        # Reduïm la densitat de la graella de 12x12 a 10x10 per evitar l'error 414 (URL massa llarga)
        lats, lons = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 10), np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
        responses = openmeteo.weather_api(API_URL, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
            if not any(np.isnan(v) for v in vals):
                output["lats"].append(r.Latitude())
                output["lons"].append(r.Longitude())
                for i, var in enumerate(variables): output[var].append(vals[i])
        if not output["lats"]: return None, f"No s'han rebut dades vàlides."
        return output, None
    except Exception as e:
        return None, f"Error crític en carregar dades del mapa: {e}"

@st.cache_data(ttl=3600)
def preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel):
    dades_ia = {}
    data_tuple, error_sondeig = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if not data_tuple: return None, f"Falten dades del sondeig ({error_sondeig})"
    dades_ia['sondeig'] = data_tuple[1]

    variables_mapa = ["cape", "relative_humidity_700hPa", "wind_speed_925hPa", "wind_direction_925hPa"]
    map_data, error_mapa = carregar_dades_mapa(variables_mapa, hourly_index_sel)
    if not map_data: return None, f"Falten dades del mapa ({error_mapa})"

    resum_mapa = {
        'max_cape_catalunya': 0, 'max_rh700_catalunya': 0,
        'max_conv_925hpa': 0, 'lat_max_conv': 0, 'lon_max_conv': 0
    }
    if map_data.get('cape'): resum_mapa['max_cape_catalunya'] = max(map_data['cape'])
    if map_data.get('relative_humidity_700hPa'): resum_mapa['max_rh700_catalunya'] = max(map_data['relative_humidity_700hPa'])
    
    if map_data.get('wind_speed_925hPa'):
        try:
            lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
            speeds_kmh = np.array(map_data['wind_speed_925hPa']) * units('km/h')
            dirs_deg = np.array(map_data['wind_direction_925hPa']) * units.degrees
            u_comp, v_comp = mpcalc.wind_components(speeds_kmh, dirs_deg)
            grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 50), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 50))
            grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
            grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
            
            resum_mapa['max_conv_925hpa'] = np.nanmin(divergence)
            idx_min = np.unravel_index(np.nanargmin(divergence), divergence.shape)
            resum_mapa['lat_max_conv'] = grid_lat[idx_min]
            resum_mapa['lon_max_conv'] = grid_lon[idx_min]
        except Exception:
            pass
    
    dades_ia['mapa_resum'] = resum_mapa
    return dades_ia, None

@st.cache_data(ttl=3600)
def generar_resum_ia(_dades_ia, _poble_sel, _timestamp_str):
    if not GEMINI_CONFIGURAT: return "Error: La clau API de Google no està configurada."
    model = genai.GenerativeModel('gemini-1.5-flash')
    mapa, sondeig = _dades_ia.get('mapa_resum', {}), _dades_ia.get('sondeig', {})

    prompt = f"""
    **ROL:** Ets un Sistema d'Anàlisi Predictiu de Tormentes per a Catalunya. La teva comunicació ha de ser clara, concisa i tècnicament precisa.
    **CONTEXT:** Estàs analitzant una sortida del model AROME per al dia i hora: {_timestamp_str}.
    **DADES D'ENTRADA:**
    - **Anàlisi Regional (Catalunya):**
      - CAPE Màxim (Energia disponible): {int(mapa.get('max_cape_catalunya', 0))} J/kg
      - Convergència Màxima a 925hPa (Mecanisme de tret): {mapa.get('max_conv_925hpa', 0):.2f} (x10⁻⁵ s⁻¹)
      - Focus de Convergència (Lat/Lon): {mapa.get('lat_max_conv', 0):.2f}, {mapa.get('lon_max_conv', 0):.2f}
    - **Anàlisi Local (Punt de Referència):**
      - Cisallament 0-6km (Organització): {int(sondeig.get('Shear_0-6km', 0))} m/s
      - SRH 0-3km (Potencial de Rotació): {int(sondeig.get('SRH_0-3km', 0))} m²/s²
    **INSTRUCCIONS:**
    1.  **NIVELL DE RISC:** Avalua el risc global de temps sever (Baix, Moderat, Alt, Extrem).
    2.  **AMENACES PRINCIPALS:** Llista les amenaces meteorològiques més probables (ex: Pluja intensa, Calamarsa/Pedra, Ratxes fortes de vent).
    3.  **RESUM TÀCTIC:** En una sola frase, explica la dinàmica atmosfèrica.
    4.  **ZONES DE MÀXIMA PROBABILITAT:** Utilitzant el teu coneixement geogràfic, identifica 3-5 comarques o ciutats importants a prop del focus de convergència. Aquesta és la teva tasca més crucial.
    **FORMAT DE SORTIDA (OBLIGATORI - utilitza Markdown i emojis):**
    **🚨 Nivell de Risc:** [El teu nivell de risc aquí]
    **⚡ Amenaces Principals:**
    - [Amenaça 1]
    - [Amenaça 2]
    **🔬 Resum Tàctic:** [La teva frase d'anàlisi aquí]
    **🎯 Zones de Màxima Probabilitat:** [Llista de 3-5 comarques/poblacions]
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"S'ha produït un error en contactar amb l'assistent d'IA: {e}"


# --- 2. FUNCIONS DE VISUALITZACIÓ (GRÀFICS I MAPES) ---

TEXT_GLOW = [path_effects.withStroke(linewidth=3, foreground="black")]

def crear_mapa_base():
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#2A3B4C", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#161B22', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='white', zorder=5)
    return fig, ax

def get_wind_colormap():
    colors = ['#FFFFFF', '#E0F5FF', '#B9E8FF', '#87D7F9', '#5AC7E3', '#2DB8CC', '#3FC3A3', '#5ABF7A', '#75BB51', '#98D849', '#C2E240', '#EBEC38', '#F5D03A', '#FDB43D', '#F7983F', '#E97F41', '#D76643', '#C44E45', '#B23547', '#A22428', '#881015', '#6D002F', '#860057', '#A0007F', '#B900A8', '#D300D0', '#E760E7', '#F6A9F6', '#FFFFFF', '#CCCCCC']
    levels = list(range(0, 95, 5)) + list(range(100, 211, 10))
    cmap = ListedColormap(colors, name='wind_speed_custom')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm, levels

# RESTAURADA
def crear_mapa_500hpa(map_data, timestamp_str):
    fig, ax = crear_mapa_base()
    lons, lats = map_data['lons'], map_data['lats']
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_temp = griddata((lons, lats), map_data['temperature_500hPa'], (grid_lon, grid_lat), method='cubic')
    temp_levels = np.arange(-30, 1, 2)
    cf = ax.contourf(grid_lon, grid_lat, grid_temp, levels=temp_levels, cmap='coolwarm', extend='min', alpha=0.7, zorder=2)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7); cbar.set_label("Temperatura a 500 hPa (°C)")
    
    # AFEGIDES ISOLÍNIES DE TEMPERATURA
    cs_temp = ax.contour(grid_lon, grid_lat, grid_temp, levels=temp_levels, colors='white', linewidths=0.8, linestyles='--', alpha=0.7, zorder=3)
    ax.clabel(cs_temp, inline=True, fontsize=8, fmt='%1.0f°C')
    
    u, v = mpcalc.wind_components(np.array(map_data['wind_speed_500hPa']) * units('km/h'), np.array(map_data['wind_direction_500hPa']) * units.degrees)
    ax.barbs(lons, lats, u.to('kt').m, v.to('kt').m, length=5, zorder=6, color='yellow', transform=ccrs.PlateCarree())
    ax.set_title(f"Anàlisi a 500 hPa (Temperatura i Vent)\n{timestamp_str}", weight='bold', fontsize=16, color="#00BFFF", path_effects=TEXT_GLOW)
    return fig

# MILLORADA AMB ISOLÍNIES
def crear_mapa_vents_velocitat(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    cmap, norm, levels = get_wind_colormap()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    ax.contourf(grid_lon, grid_lat, grid_speed, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend='max')
    
    # AFEGIDES ISOLÍNIES DE VELOCITAT
    speed_iso_levels = np.arange(20, 201, 20)
    cs_speed = ax.contour(grid_lon, grid_lat, grid_speed, levels=speed_iso_levels, colors='white', linestyles='--', linewidths=0.8, alpha=0.7, zorder=3)
    ax.clabel(cs_speed, inline=True, fontsize=8, fmt='%1.0f')

    speeds_ms = np.array(speed_data) * units('km/h'); dirs_deg = np.array(dir_data) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    rbf_u = Rbf(lons, lats, u_comp.to('m/s').m, function='thin_plate', smooth=0)
    rbf_v = Rbf(lons, lats, v_comp.to('m/s').m, function='thin_plate', smooth=0)
    u_grid = rbf_u(grid_lon, grid_lat); v_grid = rbf_v(grid_lon, grid_lat)
    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color='white', linewidth=0.6, density=2.5, arrowsize=0.6, zorder=5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7, ticks=levels[::2])
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16, color="#00BFFF", path_effects=TEXT_GLOW)
    return fig

# RESTAURADA A LA VERSIÓ ORIGINAL (AMB ADAPTACIÓ ESTÈTICA)
def crear_mapa_convergencia(lons, lats, speed_data, dir_data, nivell, lat_sel, lon_sel, nom_poble_sel, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    speeds_ms=np.array(speed_data)*units('km/h'); dirs_deg=np.array(dir_data)*units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    
    u_grid = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    v_grid = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(u_grid*units('m/s'), v_grid*units('m/s'), dx=dx, dy=dy) * 1e5
    levels = np.linspace(-20, 20, 15)
    cf = ax.contourf(grid_lon, grid_lat, divergence, levels=levels, cmap='coolwarm_r', alpha=0.6, zorder=2, extend='both')
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7); cbar.set_label('Convergència (vermell) / Divergència (blau) [x10⁻⁵ s⁻¹]')
    
    # AFEGIDES ISOLÍNIES DE CONVERGÈNCIA
    cs_conv = ax.contour(grid_lon, grid_lat, divergence, levels=levels, colors='white', linewidths=0.7, alpha=0.3, zorder=3)
    ax.clabel(cs_conv, inline=True, fontsize=8, fmt='%1.0f')
    
    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color='white', linewidth=0.5, density=5.0, arrowsize=0.5, zorder=4)
    ax.plot(lon_sel, lat_sel, 'o', markerfacecolor='#FF00FF', markeredgecolor='black', markersize=10, transform=ccrs.Geodetic(), zorder=6)
    txt = ax.text(lon_sel + 0.05, lat_sel, nom_poble_sel, transform=ccrs.Geodetic(), zorder=7, fontsize=12, weight='bold', color='yellow')
    txt.set_path_effects(TEXT_GLOW)
    max_conv = np.nanmin(divergence)
    ax.set_title(f"Flux i Convergència a {nivell}hPa (Mín: {max_conv:.1f})\n{timestamp_str}", weight='bold', fontsize=16, color="#00BFFF", path_effects=TEXT_GLOW)
    return fig

# MILLORADA AMB ISOLÍNIES
def crear_mapa_escalar(lons, lats, data, titol, cmap, levels, unitat, timestamp_str, extend='max'):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_data = griddata((lons, lats), data, (grid_lon, grid_lat), method='cubic')
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    cf = ax.contourf(grid_lon, grid_lat, grid_data, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend=extend)
    
    # AFEGIDES ISOLÍNIES GENÈRIQUES
    iso_levels = levels[::(len(levels)//8 if len(levels) > 8 else 1)] # Evita massa línies
    contorns = ax.contour(grid_lon, grid_lat, grid_data, levels=iso_levels, colors='white', linewidths=0.8, alpha=0.7, zorder=3)
    ax.clabel(contorns, inline=True, fontsize=8, fmt='%1.0f')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f"{titol} ({unitat})")
    ax.set_title(f"{titol}\n{timestamp_str}", weight='bold', fontsize=16, color="#00BFFF", path_effects=TEXT_GLOW)
    return fig

def crear_skewt(p, T, Td, u, v, titol):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(9, 9), dpi=200)
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.8, 0.85))
    skew.ax.grid(True, linestyle='--', alpha=0.3)
    prof = mpcalc.parcel_profile(p, T[0], Td[0])
    skew.shade_cape(p, T, prof, color='#FF00FF', alpha=0.3)
    skew.shade_cin(p, T, prof, color='#00BFFF', alpha=0.3)
    skew.plot(p, T, 'magenta', lw=2, label='Temperatura')
    skew.plot(p, Td, 'cyan', lw=2, label='Punt de Rosada')
    skew.plot(p, prof, 'yellow', linewidth=2, label='Trajectòria Parcel·la')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03, color='white')
    skew.ax.set_ylim(1020, 150); skew.ax.set_xlim(-40, 40)
    skew.ax.set_title(titol, weight='bold', fontsize=14, color="#00BFFF", path_effects=TEXT_GLOW)
    skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")
    skew.ax.legend()
    return fig

def crear_hodograf(u, v):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    h = Hodograph(ax, component_range=60.)
    h.add_grid(increment=20, color='gray', linestyle='--')
    h.plot(u.to('kt'), v.to('kt'), color='#FF00FF', linewidth=2.5)
    ax.set_title("Hodògraf", weight='bold', color="#00BFFF", path_effects=TEXT_GLOW)
    return fig
    
def mostrar_imatge_temps_real(tipus):
    if tipus == "📡 Radar":
        url, caption = "https://www.meteociel.fr/cartes_obs/radar/lastradar_sp_ne.gif", "Radar de precipitació. Font: Meteociel"
    else:
        now_local = datetime.now(TIMEZONE)
        is_night = now_local.hour >= 22 or now_local.hour < 7
        url, caption = "https://modeles20.meteociel.fr/satellite/animsatircolmtgsp.gif", "Satèl·lit infraroig. Font: Meteociel" if is_night else "https://modeles20.meteociel.fr/satellite/latestsatviscolmtgsp.png", "Satèl·lit visible. Font: Meteociel"
    try:
        response = requests.get(f"{url}?ver={int(time.time())}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code == 200: st.image(response.content, caption=caption, use_container_width=True)
        else: st.warning(f"No s'ha pogut carregar la imatge. (Codi: {response.status_code})")
    except Exception: st.error("Error de xarxa en carregar la imatge.")

# --- 3. LÒGICA DE LA INTERFÍCIE D'USUARI (UI) ---

def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #00BFFF;">🌪️ Terminal d\'Anàlisi de Temps Sever /// Catalunya</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Plataforma avançada per a l\'anàlisi de convecció basada en el model AROME.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.selectbox("Punt de referència:", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3: st.selectbox("Hora del pronòstic (Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    with st.spinner("Processant mapes d'anàlisi..."):
        col_map_1, col_map_2 = st.columns([2.5, 1.5])
        with col_map_1:
            # RESTAURAT EL MENÚ COMPLET
            map_options = {
                "CAPE (Energia Convectiva)": "cape", 
                "Flux i Convergència": "conv", 
                "Anàlisi a 500hPa": "500hpa", 
                "Vent a 300hPa": "wind_300", 
                "Vent a 700hPa": "wind_700", 
                "Humitat a 700hPa": "rh_700"
            }
            mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
            map_key = map_options[mapa_sel]
            error_map = None
            
            if map_key == "cape":
                map_data, error_map = carregar_dades_mapa(["cape"], hourly_index_sel)
                if map_data:
                    max_cape = np.max(map_data['cape']) if map_data['cape'] else 0
                    if max_cape <= 500: cape_levels = np.arange(50, 501, 50)
                    elif max_cape <= 1500: cape_levels = np.arange(100, 1501, 100)
                    else: cape_levels = np.arange(250, np.ceil(max_cape / 250) * 250 + 1, 250)
                    st.pyplot(crear_mapa_escalar(map_data['lons'], map_data['lats'], map_data['cape'], "CAPE", "plasma", cape_levels, "J/kg", timestamp_str))
            
            elif map_key == "conv":
                nivell_sel = st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850], format_func=lambda x: f"{x} hPa")
                variables = [f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data: st.pyplot(crear_mapa_convergencia(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell_sel, lat_sel, lon_sel, poble_sel, timestamp_str))
            
            elif map_key == "500hpa": # RESTAURAT
                variables = ["temperature_500hPa", "wind_speed_500hPa", "wind_direction_500hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data: st.pyplot(crear_mapa_500hpa(map_data, timestamp_str))
            
            elif map_key in ["wind_300", "wind_700"]: # RESTAURAT
                nivell_hpa = int(map_key.split('_')[1])
                variables = [f"wind_speed_{nivell_hpa}hPa", f"wind_direction_{nivell_hpa}hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data: st.pyplot(crear_mapa_vents_velocitat(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell_hpa, timestamp_str))

            elif map_key == "rh_700":
                map_data, error_map = carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
                if map_data: st.pyplot(crear_mapa_escalar(map_data['lons'], map_data['lats'], map_data['relative_humidity_700hPa'], "Humitat Relativa a 700hPa", "Greens", np.arange(50, 101, 5), "%", timestamp_str, extend="neither"))
            
            if error_map: st.error(f"Error en carregar el mapa: {error_map}")

        with col_map_2:
            st.subheader("Visió en Temps Real")
            view_choice = st.radio("Selecciona vista:", ("🛰️ Satèl·lit", "📡 Radar"), horizontal=True, label_visibility="collapsed")
            mostrar_imatge_temps_real(view_choice)

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        cols = st.columns(4)
        metrics = {'CAPE': 'J/kg', 'CIN': 'J/kg', 'Shear_0-6km': 'm/s', 'SRH_0-3km': 'm²/s²'}
        for i, (param, unit) in enumerate(metrics.items()):
            val = params_calculats.get(param)
            cols[i].metric(label=param, value=f"{f'{val:.0f}' if val is not None else 'N/A'} {unit}")
        
        col1, col2 = st.columns([2, 1])
        with col1: st.pyplot(crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], f"Sondeig Vertical - {poble_sel}"))
        with col2: 
            st.pyplot(crear_hodograf(sounding_data[3], sounding_data[4]))
            with st.expander("ℹ️ Interpretació dels Paràmetres"):
                st.markdown("""
                - **CAPE:** Energia Convectiva. >1000 J/kg indica potencial per tempestes fortes.
                - **CIN:** "Tapa" que impedeix la convecció.
                - **Shear 0-6km:** Cisallament del vent. > 18 m/s afavoreix l'organització (supercèl·lules).
                - **SRH 0-3km:** Potencial de rotació. > 150 m²/s² suggereix alt potencial per a mesociclons.
                """)
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    st.subheader(f"Assistent d'Anàlisi Predictiva IA /// {timestamp_str}")
    if not GEMINI_CONFIGURAT:
        st.error("Funcionalitat no disponible. La clau API de Google no està configurada correctament a `.streamlit/secrets.toml`.")
        return
    if st.button("🤖 Generar Anàlisi d'IA", use_container_width=True):
        with st.spinner("L'assistent IA està processant milers de punts de dades..."):
            dades_ia, error = preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel)
            if error:
                st.error(f"No s'ha pogut generar l'anàlisi: {error}")
                return
            resum_text = generar_resum_ia(dades_ia, poble_sel, timestamp_str)
            st.markdown(resum_text)

def ui_peu_de_pagina():
    st.divider()
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- 4. APLICACIÓ PRINCIPAL ---

def main():
    if 'hora_selector' not in st.session_state: 
        st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"

    ui_capcalera_selectors()

    poble_sel = st.session_state.poble_selector
    dia_sel = st.session_state.dia_selector
    hora_sel = st.session_state.hora_selector
    
    hora_int = int(hora_sel.split(':')[0])
    now_local = datetime.now(TIMEZONE)
    target_date = now_local.date()
    if dia_sel == "Demà": target_date += timedelta(days=1)
    
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int))
    utc_dt = local_dt.astimezone(pytz.utc)
    
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600)
    hourly_index_sel = max(0, time_diff_hours)

    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
    lat_sel = CIUTATS_CATALUNYA[poble_sel]['lat']
    lon_sel = CIUTATS_CATALUNYA[poble_sel]['lon']

    data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if error_msg and 'data_tuple' not in locals(): # Mostra l'error només si les dades no es poden carregar
        st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")

    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Resum IA"])

    with tab_mapes: ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia: ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
        
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
