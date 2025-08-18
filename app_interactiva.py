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
import io
from scipy.ndimage import label
from matplotlib.patches import PathPatch
import streamlit.components.v1 as components

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
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200},
    'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
}
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 975, 950, 925, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

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
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0], [sfc_h]
        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
        p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC; Td = np.array(Td_profile) * units.degC
        u = np.array(u_profile) * units('m/s'); v = np.array(v_profile) * units('m/s'); h = np.array(h_profile) * units.meter
        params_calc = {}
        prof = mpcalc.parcel_profile(p, T[0], Td[0])
        cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0
        params_calc['CIN'] = cin.to('J/kg').m
        
        # --- NOU: Càlcul del LFC ---
        try:
            p_lfc, _ = mpcalc.lfc(p, T, Td)
            params_calc['LFC_hPa'] = p_lfc.m if not np.isnan(p_lfc.m) else np.nan
        except Exception:
            params_calc['LFC_hPa'] = np.nan # Si hi ha un error, el marquem com a no disponible

        s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6 * units.km)
        params_calc['Shear_0-6km'] = mpcalc.wind_speed(s_u, s_v).to('m/s').m
        _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=3 * units.km)
        params_calc['SRH_0-3km'] = np.sum(srh).to('m^2/s^2').m
        return ((p, T, Td, u, v, h), params_calc), None
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

@st.cache_data(ttl=3600)
def preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel):
    dades_ia = {}
    data_tuple, error_sondeig = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if not data_tuple: return None, f"Falten dades del sondeig ({error_sondeig})"
    dades_ia['sondeig'] = data_tuple[1]
    variables_mapa = ["cape", "relative_humidity_700hPa", "wind_speed_925hPa", "wind_direction_925hPa"]
    map_data, error_mapa = carregar_dades_mapa(variables_mapa, hourly_index_sel)
    if not map_data: return None, f"Falten dades del mapa ({error_mapa})"
    resum_mapa = {}
    if 'cape' in map_data and map_data['cape']: resum_mapa['max_cape_catalunya'] = max(map_data['cape'])
    if 'relative_humidity_700hPa' in map_data and map_data['relative_humidity_700hPa']: resum_mapa['max_rh700_catalunya'] = max(map_data['relative_humidity_700hPa'])
    if 'wind_speed_925hPa' in map_data and map_data['wind_speed_925hPa']:
        try:
            lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
            speeds_kmh = np.array(map_data['wind_speed_925hPa']) * units('km/h'); dirs_deg = np.array(map_data['wind_direction_925hPa']) * units.degrees
            u_comp, v_comp = mpcalc.wind_components(speeds_kmh, dirs_deg)
            grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 50), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 50))
            grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic'); grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
            resum_mapa['max_conv_925hpa'] = np.nanmin(divergence.magnitude); idx_min = np.nanargmin(divergence.magnitude)
            idx_2d = np.unravel_index(idx_min, divergence.shape)
            resum_mapa['lat_max_conv'] = grid_lat[idx_2d]; resum_mapa['lon_max_conv'] = grid_lon[idx_2d]
        except Exception: resum_mapa['max_conv_925hpa'] = 0; resum_mapa['lat_max_conv'] = 0; resum_mapa['lon_max_conv'] = 0
    dades_ia['mapa_resum'] = resum_mapa
    return dades_ia, None

@st.cache_data(ttl=3600)
def generar_resum_ia(_dades_ia, _poble_sel, _timestamp_str):
    if not GEMINI_CONFIGURAT: return "Error: La clau API de Google no està configurada."
    model = genai.GenerativeModel('gemini-1.5-flash'); mapa = _dades_ia.get('mapa_resum', {}); sondeig = _dades_ia.get('sondeig', {})
    prompt = f"""
    Ets un assistent de meteorologia directe i concís. Analitza les dades del model AROME per a Catalunya i genera un avís curt.
    **DADES:**
    - Hora: {_timestamp_str}
    - CAPE màxim: {int(mapa.get('max_cape_catalunya', 0))} J/kg
    - Convergència màxima 925hPa: {mapa.get('max_conv_925hpa', 0):.2f} (x10⁻⁵ s⁻¹)
    - Lat/Lon focus convergència: {mapa.get('lat_max_conv', 0):.2f}, {mapa.get('lon_max_conv', 0):.2f}
    - Shear 0-6km: {int(sondeig.get('Shear_0-6km', 0))} m/s
    - SRH 0-3km: {int(sondeig.get('SRH_0-3km', 0))} m²/s²
    **INSTRUCCIONS:**
    1. **Risc:** Defineix el nivell de risc (Baix, Moderat, Alt, Molt Alt).
    2. **Poblacions Clau:** Identifica 3-5 poblacions importants a prop del focus de convergència.
    3. **Justificació:** Explica breument el perquè del risc (convergència + combustible).
    **FORMAT OBLIGATORI (Markdown):**
    **Resum del Risc:** [La teva frase]
    **Poblacions Potencialment Afectades:** [Llista de poblacions]
    **Justificació Tècnica (Molt Breu):** [La teva explicació]"""
    try: return model.generate_content(prompt).text
    except Exception as e: return f"Error en contactar amb l'IA: {e}"

# --- 2. FUNCIONS DE VISUALITZACIÓ ---

def crear_mapa_base():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0); ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5); return fig, ax

def crear_mapa_forecast_combinat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    
    # Interpolar dades
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), method='cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
    
    # Llindars dinàmics amb la nova secció per a 800 hPa
    if nivell >= 950:
        CONVERGENCE_THRESHOLD = -45; DEWPOINT_THRESHOLD_FOR_RISK = 14
    elif nivell >= 925:
        CONVERGENCE_THRESHOLD = -35; DEWPOINT_THRESHOLD_FOR_RISK = 12
    elif nivell >= 850:
        CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 7
    elif nivell >= 800: # NOU: Llindars específics per a 800 hPa
        CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 5
    elif nivell >= 700:
        CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 2
    else: # Per a nivells més alts (encara que no s'utilitzin al menú)
        CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = -5

    # Escala de colors per al vent
    colors_wind_final = [
        '#FFFFFF', '#B0E0E6', '#00FFFF', '#3CB371', '#32CD32', '#ADFF2F', '#FFD700',
        '#F4A460', '#CD853F', '#A0522D', '#DC143C', '#8B0000', '#800080', '#FF00FF',
        '#FFC0CB', '#D3D3D3', '#A9A9A9'
    ]
    speed_levels_final = np.arange(0, 171, 10)
    custom_cmap = ListedColormap(colors_wind_final)
    norm_speed = BoundaryNorm(speed_levels_final, ncolors=custom_cmap.N, clip=True)
    
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=speed_levels_final[::2])
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density=5, arrowsize=0.6, zorder=4)

    # Lògica de risc
    effective_risk_mask = (divergence.magnitude <= CONVERGENCE_THRESHOLD) & (grid_dewpoint >= DEWPOINT_THRESHOLD_FOR_RISK)
    labels, num_features = label(effective_risk_mask)
    if num_features > 0:
        for i in range(1, num_features + 1):
            points = np.argwhere(labels == i)
            center_y, center_x = points.mean(axis=0)
            center_lon, center_lat = grid_lon[0, int(center_x)], grid_lat[int(center_y), 0]
            warning_txt = ax.text(center_lon, center_lat, '⚠️', color='yellow', fontsize=15, ha='center', va='center', zorder=8)
            warning_txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    ax.set_title(f"Forecast: Força del Vent + Focus de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig
    
    
def crear_mapa_500hpa(map_data, timestamp_str):
    fig, ax = crear_mapa_base(); lons, lats = map_data['lons'], map_data['lats']
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_temp = griddata((lons, lats), map_data['temperature_500hPa'], (grid_lon, grid_lat), method='cubic')
    temp_levels = np.arange(-30, 1, 2)
    cf = ax.contourf(grid_lon, grid_lat, grid_temp, levels=temp_levels, cmap='coolwarm', extend='min', alpha=0.7, zorder=2)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7); cbar.set_label("Temperatura a 500 hPa (°C)")
    cs_temp = ax.contour(grid_lon, grid_lat, grid_temp, levels=temp_levels, colors='gray', linewidths=0.8, linestyles='--', zorder=3)
    ax.clabel(cs_temp, inline=True, fontsize=7, fmt='%1.0f°C')
    u, v = mpcalc.wind_components(np.array(map_data['wind_speed_500hPa']) * units('km/h'), np.array(map_data['wind_direction_500hPa']) * units.degrees)
    ax.barbs(lons[::5], lats[::5], u.to('kt').m[::5], v.to('kt').m[::5], length=5, zorder=6, transform=ccrs.PlateCarree())
    ax.set_title(f"Anàlisi a 500 hPa (Temperatura i Vent)\n{timestamp_str}", weight='bold', fontsize=16); return fig

# NOU: Funció genèrica per crear mapes de vent a diferents nivells
def crear_mapa_vents(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))

    # Calculem i interpolem els components del vent i la velocitat
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')

    # --- CANVI CLAU: Nova escala de colors i nivells de Meteociel ---

    # 1. Definim la llista de colors (hexadecimal) extreta de la imatge de referència
    colors_meteo_vent = [
        '#FFFFFF', '#B4F0FF', '#82D3FF', '#41B4FF', '#0096FF', '#00D278', '#00B43C', 
        '#64D200', '#96F500', '#C8FF00', '#F5DC00', '#FFB400', '#FF8C00', '#FF6400', 
        '#F53C00', '#D21400', '#B40000', '#820000', '#640000', '#82004B', '#B40082', 
        '#D200B4', '#F000DC', '#FF00FF', '#FF64FF', '#FF96FF', '#FFFFFF', '#DCDCDC', 
        '#BEBEBE', '#A0A0A0'
    ]
    # Creem el mapa de colors personalitzat
    custom_cmap = ListedColormap(colors_meteo_vent)
    
    # 2. Definim els nivells de velocitat que corresponen a cada color
    speed_levels = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 
                    160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 320, 340, 
                    360, 380, 400, 420, 440]
    norm = BoundaryNorm(speed_levels, ncolors=custom_cmap.N)

    # 3. Dibuixem el fons de color amb la nova escala personalitzada
    cf = ax.contourf(grid_lon, grid_lat, grid_speed,
                     levels=speed_levels,
                     cmap=custom_cmap,
                     norm=norm,
                     zorder=2,
                     extend='both') # 'both' per si hi ha valors fora del rang

    # 4. Superposem les streamlines en negre, com abans
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v,
                  color='black',
                  linewidth=0.7,
                  density=2.5,
                  arrowsize=0.6,
                  zorder=3)

    # 5. Creem la barra de color
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7, ticks=speed_levels[::2]) # Mostrem ticks de 40 en 40
    cbar.set_label("Velocitat del Vent (km/h)")
    
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16)
    
    return fig
def crear_mapa_escalar(lons, lats, data, titol, cmap, levels, unitat, timestamp_str, extend='max'):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_data = griddata((lons, lats), data, (grid_lon, grid_lat), method='cubic'); norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    cf = ax.contourf(grid_lon, grid_lat, grid_data, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend=extend)
    contorns = ax.contour(grid_lon, grid_lat, grid_data, levels=levels[::(len(levels)//5)], colors='black', linewidths=0.7, alpha=0.9, zorder=3)
    ax.clabel(contorns, inline=True, fontsize=8, fmt='%1.0f'); cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f"{titol} ({unitat})"); ax.set_title(f"{titol}\n{timestamp_str}", weight='bold', fontsize=16); return fig

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
    
def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str, data_tuple):
    col_map_1, col_map_2 = st.columns([0.7, 0.3], gap="large")
    
    with col_map_1:
        map_options = {
            "Forecast: Vent + Convergència Efectiva": "forecast_combinat",
            "Temperatura i Vent a 500hPa": "500hpa",
            "Vent a 700hPa (Streamlines)": "vent_700",
            "Vent a 300hPa (Streamlines)": "vent_300",
            "Humitat a 700hPa": "rh_700"
        }
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
        map_key, error_map = map_options[mapa_sel], None

        if map_key == "forecast_combinat":
            
            cin_value = 0; lfc_hpa = np.nan
            if data_tuple and data_tuple[1]:
                cin_value = data_tuple[1].get('CIN', 0); lfc_hpa = data_tuple[1].get('LFC_hPa', np.nan)
            
            if cin_value < -25:
                st.warning(f"**AVÍS DE 'TAPA' (CIN = {cin_value:.0f} J/kg):** El sondeig de **{poble_sel}** mostra una forta inversió. Es necessita un forçament dinàmic potent per trencar-la.")
            
            if np.isnan(lfc_hpa):
                st.error("**DIAGNÒSTIC LFC:** No s'ha trobat LFC. L'atmosfera és estable i la convecció espontània és molt improbable.")
            elif lfc_hpa >= 900:
                st.success(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció de base superficial. **Recomanació: Buscar zones d'alerta ⚠️ a 1000-925 hPa.**")
            elif lfc_hpa >= 750:
                st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció de base baixa. **Recomanació: Buscar zones d'alerta ⚠️ a 850-800 hPa.**")
            else:
                st.info(f"**DIAGNÒSTIC LFC ({lfc_hpa:.0f} hPa):** Convecció elevada. **Recomanació: Buscar zones d'alerta ⚠️ a 700 hPa.**")

            nivell_sel = st.selectbox("Nivell d'anàlisi:", 
                                      options=[1000, 950, 925, 850, 800, 700], 
                                      format_func=lambda x: f"{x} hPa")
            
            if nivell_sel >= 950:
                variables = ["dew_point_2m", f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data:
                    dewpoint_for_calc = map_data['dew_point_2m']
                    speed_data = map_data[f"wind_speed_{nivell_sel}hPa"]
                    dir_data = map_data[f"wind_direction_{nivell_sel}hPa"]
                    st.pyplot(crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], speed_data, dir_data, dewpoint_for_calc, nivell_sel, timestamp_str))
                    ui_explicacio_alertes()
            
            else:
                variables = [f"temperature_{nivell_sel}hPa", f"relative_humidity_{nivell_sel}hPa", f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data:
                    temp_data = np.array(map_data[f'temperature_{nivell_sel}hPa']) * units.degC
                    rh_data = np.array(map_data[f'relative_humidity_{nivell_sel}hPa']) * units.percent
                    dewpoint_for_calc = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
                    speed_data = map_data[f"wind_speed_{nivell_sel}hPa"]
                    dir_data = map_data[f"wind_direction_{nivell_sel}hPa"]
                    st.pyplot(crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], speed_data, dir_data, dewpoint_for_calc, nivell_sel, timestamp_str))
                    ui_explicacio_alertes()

        elif map_key == "500hpa":
            variables = ["temperature_500hPa", "wind_speed_500hPa", "wind_direction_500hPa"]
            map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_500hpa(map_data, timestamp_str))

        elif map_key == "vent_700":
            nivell = 700
            variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str))

        elif map_key == "vent_300":
            nivell = 300
            variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str))

        elif map_key == "rh_700":
            map_data, error_map = carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
            if map_data: st.pyplot(crear_mapa_escalar(map_data['lons'], map_data['lats'], map_data['relative_humidity_700hPa'], "Humitat Relativa a 700hPa", "Greens", np.arange(50, 101, 5), "%", timestamp_str))
        
        if error_map: st.error(f"Error en carregar el mapa: {error_map}")

    with col_map_2:
        st.subheader("Imatges en Temps Real")
        # --- CANVI: Les pestanyes ara reflecteixen les dues vistes de satèl·lit ---
        tab_europa, tab_ne = st.tabs(["🇪🇺 Satèl·lit (Europa)", "🛰️ Satèl·lit (NE Península)"])

        with tab_europa:
            mostrar_imatge_temps_real("Satèl·lit (Europa)")
        
        with tab_ne:
            mostrar_imatge_temps_real("Satèl·lit (NE Península)")

def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🌪️ Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Eina per al pronòstic de convecció mitjançant paràmetres clau.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.selectbox("Capital de referència:", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3: st.selectbox("Hora del pronòstic (Hora Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

def mostrar_imatge_temps_real(tipus):
    # --- CANVI: Ara gestionem dues vistes de satèl·lit ---
    
    if tipus == "Satèl·lit (Europa)":
        # Aquesta és la nova URL del satèl·lit europeu que has triat
        url = "https://modeles20.meteociel.fr/satellite/animsatsandvisirmtgeu.gif"
        caption = "Satèl·lit Sandvitx (Visible + Infraroig). Font: Meteociel"
        
    elif tipus == "Satèl·lit (NE Península)":
        now_local = datetime.now(TIMEZONE)
        if 7 <= now_local.hour < 21:
            url = "https://modeles20.meteociel.fr/satellite/animsatviscolmtgsp.gif"
            caption = "Satèl·lit Visible (Nord-est). Font: Meteociel"
        else:
            url = "https://www.meteociel.fr/modeles/satanim_ir_espagne-ne.gif"
            caption = "Satèl·lit Infraroig (Nord-est). Font: Meteociel"
    
    else:
        st.error("Tipus d'imatge no reconegut.")
        return

    # Lògica comuna per carregar i mostrar la imatge
    try:
        response = requests.get(f"{url}?ver={int(time.time())}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code == 200: 
            st.image(response.content, caption=caption, use_container_width=True)
        else: 
            st.warning(f"No s'ha pogut carregar la imatge. (Codi: {response.status_code})")
    except Exception as e: 
        st.error(f"Error de xarxa en carregar la imatge.")

def ui_explicacio_alertes():
    """
    Crea un desplegable informatiu que explica el significat de les alertes de risc.
    """
    with st.expander("📖 Què signifiquen les alertes ⚠️ que veig al mapa?"):
        st.markdown("""
        Cada símbol d'alerta **⚠️** assenyala un **focus de risc convectiu**. No és una predicció de tempesta garantida, sinó la detecció d'una zona on es compleix la **"recepta perfecta"** per iniciar-ne una.

        El nostre sistema analitza les dades del model i només marca les àrees on es donen **dues condicions clau simultàniament**:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### **1. El Disparador: Convergència ↗️**
            L'aire a nivells baixos està sent forçat a ascendir amb molta intensitat. És el mecanisme que "dispara" el moviment vertical necessari per crear un núvol de tempesta (cumulonimbus).
            """)

        with col2:
            st.markdown("""
            #### **2. El Combustible: Humitat 💧**
            Aquest aire que puja no és sec; està carregat de vapor d'aigua (punt de rosada elevat). Aquesta humitat és el "combustible" que, en condensar-se, allibera energia i permet que el núvol creixi verticalment.
            """)
        
        st.info("**En resum:** Una ⚠️ indica una zona on un potent **disparador** està actuant sobre una massa d'aire amb abundant **combustible**. Per tant, són els punts als quals cal prestar més atenció.", icon="🎯")
            

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple; st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        cols = st.columns(4)
        for i, (param, unit) in enumerate({'CAPE': 'J/kg', 'CIN': 'J/kg', 'Shear_0-6km': 'm/s', 'SRH_0-3km': 'm²/s²'}.items()):
            val = params_calculats.get(param); cols[i].metric(label=param, value=f"{f'{val:.0f}' if val is not None else '---'} {unit}")
        with st.expander("ℹ️ Què signifiquen aquests paràmetres?"):
            st.markdown("- **CAPE:** Energia per a tempestes. >1000 J/kg és significatiu.\n- **CIN:** \"Tapa\" que impedeix la convecció.\n- **Shear 0-6km:** Diferència de vent amb l'altura. >15-20 m/s afavoreix l'organització (supercèl·lules).\n- **SRH 0-3km:** Potencial de rotació. >150 m²/s² afavoreix supercèl·lules i tornados.")
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
    
    # Carreguem les dades del sondeig aquí, una sola vegada
    data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
    
    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Resum IA"])
    
    with tab_mapes:
        # Passem les dades del sondeig (data_tuple) a la pestanya de mapes
        ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str, data_tuple)
    with tab_vertical: 
        ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia: 
        ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
        
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
