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
import re

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
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

PROVINCIES_BOUNDS = {
    "Barcelona": [1.3, 2.5, 41.1, 42.0], "Girona": [2.0, 3.3, 41.8, 42.6],
    "Lleida": [0.4, 1.9, 41.3, 42.8], "Tarragona": [0.3, 1.5, 40.8, 41.4]
}

# --- 1. FUNCIONS D'OBTENCIÓ I PROCESSAMENT DE DADES ---

@st.cache_data(ttl=3600)
def carregar_dades_sondeig(lat, lon, hourly_index):
    # Aquesta funció es manté igual, és robusta.
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
        response = openmeteo.weather_api(API_URL, params=params)[0]; hourly = response.Hourly()
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
        if any(np.isnan(val) for val in sfc_data.values()): return None, "Dades de superfície invàlides."
        p_data = {var: [hourly.Variables(len(h_base) + i * len(PRESS_LEVELS) + j).ValuesAsNumpy()[hourly_index] for j in range(len(PRESS_LEVELS))] for i, var in enumerate(["T", "RH", "WS", "WD", "H"])}
        sfc_h = mpcalc.pressure_to_height_std(sfc_data["surface_pressure"] * units.hPa).to('meter').m
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0], [sfc_h]
        for i, p_val in enumerate(PRESS_LEVELS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val); T_profile.append(p_data["T"][i]); Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
        p=np.array(p_profile)*units.hPa; T=np.array(T_profile)*units.degC; Td=np.array(Td_profile)*units.degC; u=np.array(u_profile)*units('m/s'); v=np.array(v_profile)*units('m/s'); h=np.array(h_profile)*units.meter
        params_calc = {}; prof = mpcalc.parcel_profile(p, T[0], Td[0]); cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0; params_calc['CIN'] = cin.to('J/kg').m
        s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6 * units.km); params_calc['Shear_0-6km'] = mpcalc.wind_speed(s_u, s_v).to('m/s').m
        _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=3 * units.km); params_calc['SRH_0-3km'] = np.sum(srh).to('m^2/s^2').m
        return ((p, T, Td, u, v, h), params_calc), None
    except Exception as e: return None, f"Error en processar dades del sondeig: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_24h(variables, start_hour, end_hour):
    try:
        lats, lons = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 12), np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {
            "latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(),
            "hourly": variables, "models": "arome_seamless", "forecast_days": FORECAST_DAYS,
            "start_hour": start_hour.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": end_hour.strftime("%Y-%m-%dT%H:%M")
        }
        responses = openmeteo.weather_api(API_URL, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            if not r.Hourly(): continue
            output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
            for i, var in enumerate(variables):
                vals = r.Hourly().Variables(i).ValuesAsNumpy()
                output[var].append(vals)
        if not output["lats"]: return None, "No s'han rebut dades vàlides."
        return output, None
    except Exception as e: return None, f"Error en carregar dades horàries del mapa: {e}"

# --- SECCIÓ DE LA IA TOTALMENT RECONSTRUÏDA (v7.0) ---

@st.cache_data(ttl=3600)
def generar_pronostic_diari_ia(dia_sel):
    target_date = datetime.now(TIMEZONE).date() + timedelta(days=1) if dia_sel == "Demà" else datetime.now(TIMEZONE).date()
    start_hour_utc = pytz.utc.localize(datetime.combine(target_date, datetime.min.time()))
    end_hour_utc = pytz.utc.localize(datetime.combine(target_date, datetime.max.time()))

    variables_mapa = ["cape", "wind_speed_925hPa", "wind_direction_925hPa"]
    map_data, error = carregar_dades_mapa_24h(variables_mapa, start_hour_utc, end_hour_utc)
    if error: return f"No s'han pogut carregar les dades per a l'anàlisi diària: {error}"

    dades_provincials = {}
    try:
        lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
        # Les dades ara tenen una dimensió temporal (24 hores)
        cape_24h = np.array(map_data['cape'])
        u_24h, v_24h = mpcalc.wind_components(np.array(map_data['wind_speed_925hPa'])*units('km/h'), np.array(map_data['wind_direction_925hPa'])*units.degrees)
        
        for provincia, bounds in PROVINCIES_BOUNDS.items():
            mask = (lons >= bounds[0]) & (lons <= bounds[1]) & (lats >= bounds[2]) & (lats <= bounds[3])
            if not np.any(mask):
                dades_provincials[provincia] = {"max_cape": 0, "hora_cape": 0, "max_conv": 0, "hora_conv": 0}
                continue

            # Calculem convergència per a cada hora i trobem el màxim del dia
            conv_hores = []
            for h in range(u_24h.shape[1]): # Iterem sobre les 24 hores
                prov_u, prov_v = u_24h[mask, h].to('m/s').m, v_24h[mask, h].to('m/s').m
                prov_lons, prov_lats = lons[mask], lats[mask]
                if len(prov_lons) > 3:
                    grid_lon, grid_lat = np.meshgrid(np.linspace(bounds[0], bounds[1], 10), np.linspace(bounds[2], bounds[3], 10))
                    grid_u = griddata((prov_lons, prov_lats), prov_u, (grid_lon, grid_lat), method='cubic', fill_value=0)
                    grid_v = griddata((prov_lons, prov_lats), prov_v, (grid_lon, grid_lat), method='cubic', fill_value=0)
                    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
                    divergence = mpcalc.divergence(grid_u*units('m/s'), grid_v*units('m/s'), dx=dx, dy=dy) * 1e5
                    conv_hores.append(np.nanmin(divergence))
                else:
                    conv_hores.append(0)
            
            # Trobem els valors màxims i les hores en què ocorren
            max_conv_prov = np.min(conv_hores)
            hora_max_conv = np.argmin(conv_hores)
            
            prov_cape_24h = cape_24h[mask, :]
            max_cape_prov = np.max(prov_cape_24h)
            hora_max_cape = np.argmax(np.max(prov_cape_24h, axis=0)) # Hora del màxim CAPE a la província
            
            dades_provincials[provincia] = {
                "max_cape": int(max_cape_prov), "hora_cape": int(hora_max_cape),
                "max_conv": round(max_conv_prov, 2), "hora_conv": int(hora_max_conv)
            }

    except Exception as e: return f"Error en el processament provincial 24h: {e}"

    if not GEMINI_CONFIGURAT: return "Error: La clau API de Google no està configurada."
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Ets un predictor expert del Servei Meteorològic de Catalunya. La teva tasca és redactar un butlletí de risc de tempestes per al dia {dia_sel}, analitzant l'evolució durant les 24 hores. Has de ser tècnic, precís i seguir el format al peu de la lletra.

    **CONTEXT METEOROLÒGIC CATALÀ (HAS D'UTILITZAR AQUEST CONEIXEMENT):**
    - **Topografia:** El Pirineu i Pre-litoral són zones clau d'inici de tempestes (dispar orogràfic).
    - **Marinada:** La brisa marina crea línies de convergència a la costa i al pre-litoral durant la tarda, actuant com a mecanisme de dispar.
    - **Evolució Típica:** Les tempestes solen començar a la muntanya a migdia/primera hora de la tarda i es poden desplaçar cap a la costa o l'interior al vespre.
    - **Interpretació Temporal:** La coincidència o proximitat horària entre el pic de convergència (la "chispa") i el pic de CAPE (el "combustible") és el factor més important per determinar el risc real. Un pic de CAPE de matinada sense "chispa" és irrellevant.

    **DADES PROCESSADES (VALORS MÀXIMS DEL DIA I HORA LOCAL APROXIMADA):**
    - **Barcelona:** CAPE Màx: {dades_provincials['Barcelona']['max_cape']} J/kg (a les {dades_provincials['Barcelona']['hora_cape']}h), Conv. Màx: {dades_provincials['Barcelona']['max_conv']} (a les {dades_provincials['Barcelona']['hora_conv']}h)
    - **Girona:** CAPE Màx: {dades_provincials['Girona']['max_cape']} J/kg (a les {dades_provincials['Girona']['hora_cape']}h), Conv. Màx: {dades_provincials['Girona']['max_conv']} (a les {dades_provincials['Girona']['hora_conv']}h)
    - **Lleida:** CAPE Màx: {dades_provincials['Lleida']['max_cape']} J/kg (a les {dades_provincials['Lleida']['hora_cape']}h), Conv. Màx: {dades_provincials['Lleida']['max_conv']} (a les {dades_provincials['Lleida']['hora_conv']}h)
    - **Tarragona:** CAPE Màx: {dades_provincials['Tarragona']['max_cape']} J/kg (a les {dades_provincials['Tarragona']['hora_cape']}h), Conv. Màx: {dades_provincials['Tarragona']['max_conv']} (a les {dades_provincials['Tarragona']['hora_conv']}h)

    **INSTRUCCIONS PER AL BUTLLETÍ:**
    1.  **Situació General:** Redacta un paràgraf inicial que descrigui la situació sinòptica i l'evolució general esperada per a tot el dia a Catalunya.
    2.  **Evolució Horària:** Crea una secció amb l'evolució temporal del risc (Matí, Tarda, Vespre/Nit).
    3.  **Anàlisi Provincial Detallada:** Crea una secció per a cada província.
        -   **Nivell de Risc:** Assigna un nivell (BAIX 🟢, MODERAT 🟠, ALT 🔴). Regla: Risc BAIX si el CAPE màxim és < 300 J/kg. Risc ALT si CAPE > 1000 J/kg i Convergència < -5, i les hores dels pics són pròximes (diferència <= 4h).
        -   **Anàlisi Tècnica:** Explica la teva decisió, utilitzant el context meteorològic català. Esmenta les hores clau i les comarques/zones geogràfiques de més risc (Pirineu, Pre-litoral, Plana de Lleida, etc.).
        -   **Fenòmens Esperats:** Llista els fenòmens possibles.

    **FORMAT DE SORTIDA (OBLIGATORI, USA MARKDOWN):**
    # Butlletí de Risc de Tempestes per a {dia_sel}
    (El teu paràgraf de situació general aquí)

    ---
    
    ## Evolució del Risc Durant el Dia
    - **Matí (06-12h):** [Breu descripció del risc]
    - **Tarda (12-19h):** [Descripció detallada del període de màxim risc]
    - **Vespre/Nit (19-00h):** [Descripció de com acaba la situació]

    ---

    ## Anàlisi per Províncies

    ### Barcelona
    **Risc:** [Nivell amb emoji]
    **Anàlisi Tècnica:** [Explicació detallada]
    **Fenòmens:** [Llista de fenòmens]

    ### Girona
    **Risc:** [Nivell amb emoji]
    **Anàlisi Tècnica:** [Explicació detallada]
    **Fenòmens:** [Llista de fenòmens]

    ### Lleida
    **Risc:** [Nivell amb emoji]
    **Anàlisi Tècnica:** [Explicació detallada]
    **Fenòmens:** [Llista de fenòmens]

    ### Tarragona
    **Risc:** [Nivell amb emoji]
    **Anàlisi Tècnica:** [Explicació detallada]
    **Fenòmens:** [Llista de fenòmens]
    """
    try: return model.generate_content(prompt).text.strip()
    except Exception as e: return f"Error contactant amb l'IA: {e}"

# --- 2. FUNCIONS DE VISUALITZACIÓ ---
# ... (Les teves funcions de gràfics es mantenen iguals) ...
def crear_mapa_base():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0); ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5); return fig, ax

def crear_mapa_escalar(lons, lats, data, titol, cmap, levels, unitat, timestamp_str, extend='max'):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_data = griddata((lons, lats), data, (grid_lon, grid_lat), method='cubic')
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    ax.contourf(grid_lon, grid_lat, grid_data, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend=extend)
    if len(levels) > 1:
        step = (len(levels)//5) if len(levels)>5 else 1
        contorns = ax.contour(grid_lon, grid_lat, grid_data, levels=levels[::step], colors='black', linewidths=0.7, alpha=0.9, zorder=3)
        ax.clabel(contorns, inline=True, fontsize=8, fmt='%1.0f')
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f"{titol} ({unitat})"); ax.set_title(f"{titol}\n{timestamp_str}", weight='bold', fontsize=16); return fig
    
def get_wind_colormap():
    colors=['#FFFFFF','#E0F5FF','#B9E8FF','#87D7F9','#5AC7E3','#2DB8CC','#3FC3A3','#5ABF7A','#75BB51','#98D849','#C2E240','#EBEC38','#F5D03A','#FDB43D','#F7983F','#E97F41','#D76643','#C44E45','#B23547','#A22428','#881015','#6D002F','#860057','#A0007F','#B900A8','#D300D0','#E760E7','#F6A9F6','#FFFFFF','#CCCCCC']
    levels = list(range(0, 95, 5)) + list(range(100, 211, 10)); cmap = ListedColormap(colors, name='wind_speed_custom')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True); return cmap, norm, levels

def crear_mapa_500hpa(map_data, timestamp_str):
    fig, ax = crear_mapa_base(); lons, lats = map_data['lons'], map_data['lats']
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_temp = griddata((lons, lats), map_data['temperature_500hPa'], (grid_lon, grid_lat), method='cubic'); temp_levels = np.arange(-30, 1, 2)
    ax.contourf(grid_lon, grid_lat, grid_temp, levels=temp_levels, cmap='coolwarm', extend='min', alpha=0.7, zorder=2)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=BoundaryNorm(temp_levels, ncolors=plt.get_cmap('coolwarm').N)), ax=ax, orientation='vertical', shrink=0.7); cbar.set_label("Temperatura a 500 hPa (°C)")
    cs_temp = ax.contour(grid_lon, grid_lat, grid_temp, levels=temp_levels, colors='gray', linewidths=0.8, linestyles='--', zorder=3)
    ax.clabel(cs_temp, inline=True, fontsize=7, fmt='%1.0f°C')
    u, v = mpcalc.wind_components(np.array(map_data['wind_speed_500hPa'])*units('km/h'), np.array(map_data['wind_direction_500hPa'])*units.degrees)
    ax.barbs(lons[::5], lats[::5], u.to('kt').m[::5], v.to('kt').m[::5], length=5, zorder=6, transform=ccrs.PlateCarree())
    ax.set_title(f"Anàlisi a 500 hPa\n{timestamp_str}", weight='bold', fontsize=16); return fig

def crear_mapa_vents_velocitat(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base(); cmap, norm, levels = get_wind_colormap()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    ax.contourf(grid_lon, grid_lat, grid_speed, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend='max')
    cs_speed = ax.contour(grid_lon, grid_lat, grid_speed, levels=np.arange(20, 201, 20), colors='gray', linestyles='--', linewidths=0.8, zorder=3)
    ax.clabel(cs_speed, inline=True, fontsize=7, fmt='%1.0f')
    speeds_ms=np.array(speed_data)*units('km/h'); dirs_deg=np.array(dir_data)*units.degrees; u,v=mpcalc.wind_components(speeds_ms,dirs_deg)
    grid_u=griddata((lons,lats), u.to('m/s').m, (grid_lon,grid_lat), method='cubic'); grid_v=griddata((lons,lats), v.to('m/s').m, (grid_lon,grid_lat), method='cubic')
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density=2.5, arrowsize=0.6, zorder=5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7, ticks=levels[::2])
    cbar.set_label("Velocitat del Vent (km/h)"); ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16); return fig

def crear_mapa_convergencia(lons, lats, speed_data, dir_data, nivell, lat_sel, lon_sel, nom_poble_sel, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    speeds_ms=np.array(speed_data)*units('km/h'); dirs_deg=np.array(dir_data)*units.degrees
    u,v=mpcalc.wind_components(speeds_ms, dirs_deg)
    grid_u=griddata((lons,lats),u.to('m/s').m,(grid_lon,grid_lat),method='cubic'); grid_v=griddata((lons,lats),v.to('m/s').m,(grid_lon,grid_lat),method='cubic')
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(grid_u*units('m/s'), grid_v*units('m/s'), dx=dx, dy=dy) * 1e5
    levels = np.linspace(-20, 20, 15)
    ax.contourf(grid_lon, grid_lat, divergence, levels=levels, cmap='coolwarm_r', alpha=0.6, zorder=2, extend='both')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm_r', norm=BoundaryNorm(levels, ncolors=plt.get_cmap('coolwarm_r').N)), ax=ax, orientation='vertical', shrink=0.7); cbar.set_label('Convergència (vermell) / Divergència (blau) [x10⁻⁵ s⁻¹]')
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.5, density=5.0, arrowsize=0.5, zorder=4)
    ax.plot(lon_sel, lat_sel, 'o', markerfacecolor='yellow', markeredgecolor='black', markersize=8, transform=ccrs.Geodetic(), zorder=6)
    txt = ax.text(lon_sel + 0.05, lat_sel, nom_poble_sel, transform=ccrs.Geodetic(), zorder=7, fontsize=10, weight='bold')
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')]); ax.set_title(f"Flux i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16); return fig

def crear_skewt(p, T, Td, u, v, titol):
    fig = plt.figure(figsize=(9, 9), dpi=150); skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.8, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5); skew.plot(p, T, 'r', lw=2, label='Temperatura'); skew.plot(p, Td, 'g', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03); skew.plot_dry_adiabats(color='brown', linestyle='--', alpha=0.6)
    skew.plot_moist_adiabats(color='blue', linestyle='--', alpha=0.6); skew.plot_mixing_lines(color='green', linestyle='--', alpha=0.6)
    prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', linewidth=2, label='Trajectòria Parcel·la')
    skew.shade_cape(p, T, prof, color='red', alpha=0.3); skew.shade_cin(p, T, prof, color='blue', alpha=0.3)
    skew.ax.set_ylim(1000, 100); skew.ax.set_xlim(-40, 40); skew.ax.set_title(titol, weight='bold', fontsize=14)
    skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)"); skew.ax.legend(); return fig

def crear_hodograf(u, v):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150); h = Hodograph(ax, component_range=60.)
    h.add_grid(increment=20, color='gray'); h.plot(u.to('kt'), v.to('kt'), color='red', linewidth=2)
    ax.set_title("Hodògraf", weight='bold'); return fig

def mostrar_imatge_temps_real(tipus):
    if tipus == "Radar": url, caption = "https://www.meteociel.fr/cartes_obs/radar/lastradar_sp_ne.gif", "Radar. Font: Meteociel"
    else:
        now_local = datetime.now(TIMEZONE)
        if now_local.hour >= 22 or now_local.hour < 7: url, caption = "https://modeles20.meteociel.fr/satellite/animsatircolmtgsp.gif", "Satèl·lit IR. Font: Meteociel"
        else: url, caption = "https://modeles20.meteociel.fr/satellite/latestsatviscolmtgsp.png", "Satèl·lit VIS. Font: Meteociel"
    try:
        response = requests.get(f"{url}?ver={int(time.time())}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code == 200: st.image(response.content, caption=caption, use_container_width=True)
    except: pass

# --- 3. LÒGICA DE LA INTERFÍCIE D'USUARI ---

def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🌪️ Terminal de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Capital (per al sondeig vertical):", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        c2.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        c3.selectbox("Hora (per a mapes i sondeig):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    with st.spinner("Actualitzant mapes..."):
        c1, c2 = st.columns([2.5, 1.5])
        with c1:
            opts = {"CAPE":"cape","Convergència":"conv","500hPa":"500hpa","Vent 300hPa":"wind_300","Vent 700hPa":"wind_700","Humitat 700hPa":"rh_700"}
            mapa_sel = st.selectbox("Capa del mapa:", opts.keys()); map_key, err = opts[mapa_sel], None
            if map_key == "cape":
                data, err = carregar_dades_mapa(["cape"], hourly_index_sel)
                if data:
                    max_cape = np.max(data['cape']) if data.get('cape') and len(data['cape']) > 0 else 0
                    levels = np.arange(100, (np.ceil(max_cape/250)*250)+1 if max_cape>250 else 501, 100) if max_cape > 100 else np.arange(0, 101, 20)
                    st.pyplot(crear_mapa_escalar(data['lons'], data['lats'], data['cape'], "CAPE", "plasma", levels, "J/kg", timestamp_str))
            elif map_key == "conv":
                lvl = st.selectbox("Nivell:", options=[1000, 950, 925, 850], format_func=lambda x: f"{x} hPa")
                v = [f"wind_speed_{lvl}hPa", f"wind_direction_{lvl}hPa"]; data, err=carregar_dades_mapa(v, hourly_index_sel)
                if data: st.pyplot(crear_mapa_convergencia(data['lons'],data['lats'],data[v[0]],data[v[1]],lvl,lat_sel,lon_sel,poble_sel,timestamp_str))
            elif map_key == "500hpa":
                v = ["temperature_500hPa", "wind_speed_500hPa", "wind_direction_500hPa"]; data, err=carregar_dades_mapa(v, hourly_index_sel)
                if data: st.pyplot(crear_mapa_500hpa(data, timestamp_str))
            elif map_key in ["wind_300", "wind_700"]:
                lvl = int(map_key.replace("wind_", "")); v = [f"wind_speed_{lvl}hPa", f"wind_direction_{lvl}hPa"]
                data, err=carregar_dades_mapa(v, hourly_index_sel)
                if data: st.pyplot(crear_mapa_vents_velocitat(data['lons'], data['lats'], data[v[0]], data[v[1]], lvl, timestamp_str))
            elif map_key == "rh_700":
                data, err=carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
                if data: st.pyplot(crear_mapa_escalar(data['lons'], data['lats'], data['relative_humidity_700hPa'], "Humitat Relativa 700hPa", "Greens", np.arange(50,101,5), "%", timestamp_str))
            if err: st.error(f"Error en carregar el mapa: {err}")
        with c2: st.subheader("Temps Real"); mostrar_imatge_temps_real(st.radio("Vista:", ("Satèl·lit", "Radar"), horizontal=True, label_visibility="collapsed"))

def ui_pestanya_vertical(hourly_index_sel, poble_sel, dia_sel, hora_sel):
    lat, lon = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']
    data_tuple, error_msg = carregar_dades_sondeig(lat, lon, hourly_index_sel)
    if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}"); return
    if data_tuple:
        sounding_data, params = data_tuple
        st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        cols = st.columns(4)
        for i, (p, u) in enumerate({'CAPE':'J/kg','CIN':'J/kg','Shear_0-6km':'m/s','SRH_0-3km':'m²/s²'}.items()):
            cols[i].metric(label=p, value=f"{f'{params.get(p):.0f}' if params.get(p) is not None else '---'} {u}")
        c1,c2 = st.columns(2)
        c1.pyplot(crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], f"Sondeig Vertical - {poble_sel}"))
        c2.pyplot(crear_hodograf(sounding_data[3], sounding_data[4]))

def ui_pestanya_ia(dia_sel):
    st.subheader(f"Butlletí de Risc per al dia: {dia_sel}")
    st.info("Aquest pronòstic analitza l'evolució de les 24 hores del dia per identificar els períodes i zones de màxim risc.")
    if not GEMINI_CONFIGURAT: st.error("Funcionalitat no disponible. La clau API no està configurada."); return

    if st.button("📝 Generar Pronòstic Diari", use_container_width=True):
        with st.spinner(f"Elaborant el butlletí per al dia de {dia_sel.lower()}... Aquest procés pot trigar una mica."):
            resum_text = generar_pronostic_diari_ia(dia_sel)
            if "Error" in resum_text:
                st.error(resum_text)
            else:
                st.markdown(resum_text)

def ui_peu_de_pagina():
    st.divider()
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- 4. APLICACIÓ PRINCIPAL ---

def main():
    if 'poble_selector' not in st.session_state: st.session_state.poble_selector = 'Barcelona'
    if 'dia_selector' not in st.session_state: st.session_state.dia_selector = 'Avui'
    if 'hora_selector' not in st.session_state: st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"

    ui_capcalera_selectors()
    poble_sel, dia_sel, hora_sel = st.session_state.poble_selector, st.session_state.dia_selector, st.session_state.hora_selector
    
    hora_int = int(hora_sel.split(':')[0])
    target_date = datetime.now(TIMEZONE).date() + timedelta(days=1) if dia_sel == "Demà" else datetime.now(TIMEZONE).date()
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = max(0, int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600))
    
    timestamp_str = f"{dia_sel} a les {hora_sel}"
    
    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi per Hores", "📊 Sondeig Vertical", "📝 Pronòstic Diari (IA)"])
    with tab_mapes: ui_pestanya_mapes(poble_sel, CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon'], hourly_index_sel, timestamp_str)
    with tab_vertical: ui_pestanya_vertical(hourly_index_sel, poble_sel, dia_sel, hora_sel)
    with tab_ia: ui_pestanya_ia(dia_sel)
    
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
