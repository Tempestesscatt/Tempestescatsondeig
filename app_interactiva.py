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
# --- NOU ---
import google.generativeai as genai
# --- FI NOU ---


# --- 0. CONFIGURACIÓ I CONSTANTS ---

# Configuració de la pàgina de Streamlit
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")

# --- NOU ---
# Configuració de l'API de Google Gemini
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_CONFIGURAT = True
except (KeyError, Exception):
    GEMINI_CONFIGURAT = False
# --- FI NOU ---


# Configuració de la sessió per a les peticions a l'API amb cache i reintents
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Constants de l'aplicació
FORECAST_DAYS = 4
API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = pytz.timezone('Europe/Madrid')

# Dades geogràfiques
CIUTATS_CATALUNYA = {
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200},
    'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
}
MAP_EXTENT = [0, 3.5, 40.4, 43] # [lon_min, lon_max, lat_min, lat_max]

# Nivells de pressió atmosfèrica per a l'anàlisi
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

# --- 1. FUNCIONS D'OBTENCIÓ I PROCESSAMENT DE DADES ---

@st.cache_data(ttl=3600)
def carregar_dades_sondeig(lat, lon, hourly_index):
    """Obté i processa les dades per al sondeig vertical des de l'API d'Open-Meteo."""
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": FORECAST_DAYS}
        
        response = openmeteo.weather_api(API_URL, params=params)[0]
        hourly = response.Hourly()
        
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
        if any(np.isnan(val) for val in sfc_data.values()):
            return None, "Dades de superfície invàlides o no disponibles."

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
        
        if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt per a una anàlisi fiable."

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
        return None, f"Error en processar les dades del sondeig: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa(variables, hourly_index):
    """Obté dades per a una o més variables en una graella sobre Catalunya."""
    try:
        lats, lons = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 12), np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 12)
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
            return None, f"No s'han rebut dades vàlides per a {', '.join(variables)}."
            
        return output, None
    except Exception as e:
        return None, f"Error en carregar les dades del mapa: {e}"

# --- NOU ---
@st.cache_data(ttl=3600)
def preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel):
    """Recopila i resumeix totes les dades necessàries per a l'anàlisi d'IA."""
    dades_ia = {}
    
    # 1. Dades del sondeig del punt de referència
    data_tuple, error_sondeig = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if data_tuple:
        dades_ia['sondeig'] = data_tuple[1]
    else:
        return None, error_sondeig

    # 2. Dades generals del mapa de Catalunya (valors màxims)
    variables_mapa = ["cape", "relative_humidity_700hPa"]
    map_data, error_mapa = carregar_dades_mapa(variables_mapa, hourly_index_sel)
    
    if map_data:
        resum_mapa = {}
        if 'cape' in map_data and map_data['cape']:
            resum_mapa['max_cape_catalunya'] = max(map_data['cape'])
        if 'relative_humidity_700hPa' in map_data and map_data['relative_humidity_700hPa']:
            resum_mapa['max_rh700_catalunya'] = max(map_data['relative_humidity_700hPa'])
        dades_ia['mapa_resum'] = resum_mapa
    else:
        return None, error_mapa
        
    return dades_ia, None

@st.cache_data(ttl=3600)
def generar_resum_ia(_dades_ia, _poble_sel, _timestamp_str):
    """Envia les dades a l'API de Gemini i retorna el resum."""
    if not GEMINI_CONFIGURAT:
        return "Error: La clau API de Google no està configurada. Afegeix-la al fitxer .streamlit/secrets.toml."
        
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Construcció del prompt
    prompt = f"""
    Ets un meteoròleg expert en temps sever a Catalunya. La teva tasca és analitzar les següents dades del model AROME
    i generar un butlletí de pronòstic de convecció concís i clar en català.

    **Context del Pronòstic:**
    - Data i hora d'anàlisi: {_timestamp_str}
    - Punt de referència principal: {_poble_sel}

    **Dades a Analitzar:**

    1.  **Paràmetres del Sondeig Vertical a {_poble_sel}:**
        - CAPE (Energia per convecció): {int(_dades_ia['sondeig'].get('CAPE', 0))} J/kg
        - CIN (Inhibició de la convecció): {int(_dades_ia['sondeig'].get('CIN', 0))} J/kg
        - Cisallament del vent (0-6 km): {int(_dades_ia['sondeig'].get('Shear_0-6km', 0))} m/s
        - Helicitat Relativa a la Tempesta (SRH 0-3 km): {int(_dades_ia['sondeig'].get('SRH_0-3km', 0))} m²/s²

    2.  **Valors Màxims Previstos a Catalunya (Visió general del mapa):**
        - CAPE màxima a Catalunya: {int(_dades_ia.get('mapa_resum', {}).get('max_cape_catalunya', 0))} J/kg
        - Humitat relativa màxima a 700 hPa: {int(_dades_ia.get('mapa_resum', {}).get('max_rh700_catalunya', 0))}%

    **Instruccions per al Butlletí:**

    - **Idioma:** Català.
    - **To:** Professional però fàcil d'entendre per a aficionats a la meteorologia.
    - **Estructura:**
        1.  **Sinopsi General:** Un paràgraf breu (2-3 línies) descrivint la situació general a Catalunya.
        2.  **Zones de Major Risc:** Identifica les àrees geogràfiques de Catalunya (p. ex., Pre-litoral, Pirineu oriental, Ponent, interior de Girona) amb més potencial de temps sever basant-te en la combinació de CAPE, humitat i paràmetres d'organització (cisallament, helicitat).
        3.  **Potencials Fenòmens:** Descriu breument els principals fenòmens meteorològics esperats (xàfecs intensos, calamarsa, ratxes fortes de vent, etc.). Si hi ha potencial per a estructures organitzades com supercèl·lules, esmenta-ho explícitament.
        4.  **Conclusió:** Una frase final de resum.

    - **Molt important:** No et limitis a llistar els valors. Interpreta'ls en el context de la meteorologia de Catalunya. Per exemple, un cisallament de 20 m/s és significatiu i afavoreix l'organització. Una CAPE de 1500 J/kg indica un potencial considerable de tempestes fortes.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"S'ha produït un error en contactar amb l'assistent d'IA: {e}"
# --- FI NOU ---


# --- 2. FUNCIONS DE VISUALITZACIÓ (GRÀFICS I MAPES) ---
# ... (Totes les teves funcions de visualització romanen exactament iguals) ...
def crear_mapa_base():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    return fig, ax

def get_wind_colormap():
    colors = ['#FFFFFF', '#E0F5FF', '#B9E8FF', '#87D7F9', '#5AC7E3', '#2DB8CC', '#3FC3A3', '#5ABF7A', '#75BB51', '#98D849', '#C2E240', '#EBEC38', '#F5D03A', '#FDB43D', '#F7983F', '#E97F41', '#D76643', '#C44E45', '#B23547', '#A22428', '#881015', '#6D002F', '#860057', '#A0007F', '#B900A8', '#D300D0', '#E760E7', '#F6A9F6', '#FFFFFF', '#CCCCCC']
    levels = list(range(0, 95, 5)) + list(range(100, 211, 10))
    cmap = ListedColormap(colors, name='wind_speed_custom')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm, levels

def crear_mapa_500hpa(map_data, timestamp_str):
    fig, ax = crear_mapa_base()
    lons, lats = map_data['lons'], map_data['lats']
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_temp = griddata((lons, lats), map_data['temperature_500hPa'], (grid_lon, grid_lat), method='cubic')
    temp_levels = np.arange(-30, 1, 2)
    cf = ax.contourf(grid_lon, grid_lat, grid_temp, levels=temp_levels, cmap='coolwarm', extend='min', alpha=0.7, zorder=2)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label("Temperatura a 500 hPa (°C)")
    cs_temp = ax.contour(grid_lon, grid_lat, grid_temp, levels=temp_levels, colors='gray', linewidths=0.8, linestyles='--', zorder=3)
    ax.clabel(cs_temp, inline=True, fontsize=7, fmt='%1.0f°C')
    u, v = mpcalc.wind_components(np.array(map_data['wind_speed_500hPa']) * units('km/h'), np.array(map_data['wind_direction_500hPa']) * units.degrees)
    ax.barbs(lons[::5], lats[::5], u.to('kt').m[::5], v.to('kt').m[::5], length=5, zorder=6, transform=ccrs.PlateCarree())
    ax.set_title(f"Anàlisi a 500 hPa (Temperatura i Vent)\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_mapa_vents_velocitat(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    cmap, norm, levels = get_wind_colormap()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
    ax.contourf(grid_lon, grid_lat, grid_speed, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend='max')
    speed_contour_levels = np.arange(20, 201, 20)
    cs_speed = ax.contour(grid_lon, grid_lat, grid_speed, levels=speed_contour_levels, colors='gray', linestyles='--', linewidths=0.8, zorder=3)
    ax.clabel(cs_speed, inline=True, fontsize=7, fmt='%1.0f')
    speeds_ms = np.array(speed_data) * units('km/h'); dirs_deg = np.array(dir_data) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    rbf_u = Rbf(lons, lats, u_comp.to('m/s').m, function='thin_plate', smooth=0)
    rbf_v = Rbf(lons, lats, v_comp.to('m/s').m, function='thin_plate', smooth=0)
    u_grid = rbf_u(grid_lon, grid_lat); v_grid = rbf_v(grid_lon, grid_lat)
    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color='black', linewidth=0.6, density=2.5, arrowsize=0.6, zorder=5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7, ticks=levels[::2])
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_mapa_convergencia(lons, lats, speed_data, dir_data, nivell, lat_sel, lon_sel, nom_poble_sel, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    speeds_ms=np.array(speed_data)*units('km/h'); dirs_deg=np.array(dir_data)*units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    rbf_u = Rbf(lons, lats, u_comp.to('m/s').m, function='thin_plate', smooth=0)
    rbf_v = Rbf(lons, lats, v_comp.to('m/s').m, function='thin_plate', smooth=0)
    u_grid = rbf_u(grid_lon, grid_lat); v_grid = rbf_v(grid_lon, grid_lat)
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(u_grid*units('m/s'), v_grid*units('m/s'), dx=dx, dy=dy) * 1e5
    max_abs_val = 20
    levels = np.linspace(-max_abs_val, max_abs_val, 15)
    cf = ax.contourf(grid_lon, grid_lat, divergence, levels=levels, cmap='coolwarm_r', alpha=0.6, zorder=2, extend='both')
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label('Convergència (vermell) / Divergència (blau) [x10⁻⁵ s⁻¹]')
    cs_conv = ax.contour(grid_lon, grid_lat, divergence, levels=levels, colors='black', linewidths=0.7, alpha=0.2, zorder=3)
    ax.clabel(cs_conv, inline=True, fontsize=8, fmt='%1.0f')
    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color='black', linewidth=0.5, density=5.0, arrowsize=0.5, zorder=4)
    ax.plot(lon_sel, lat_sel, 'o', markerfacecolor='yellow', markeredgecolor='black', markersize=8, transform=ccrs.Geodetic(), zorder=6)
    txt = ax.text(lon_sel + 0.05, lat_sel, nom_poble_sel, transform=ccrs.Geodetic(), zorder=7, fontsize=10, weight='bold')
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    max_conv = np.nanmin(divergence)
    ax.set_title(f"Flux i Convergència a {nivell}hPa (Mín: {max_conv:.1f})\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_mapa_escalar(lons, lats, data, titol, cmap, levels, unitat, timestamp_str, extend='max'):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_data = griddata((lons, lats), data, (grid_lon, grid_lat), method='cubic')
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    cf = ax.contourf(grid_lon, grid_lat, grid_data, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend=extend)
    contorns = ax.contour(grid_lon, grid_lat, grid_data, levels=levels[::(len(levels)//5)], colors='black', linewidths=0.7, alpha=0.9, zorder=3)
    ax.clabel(contorns, inline=True, fontsize=8, fmt='%1.0f')
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f"{titol} ({unitat})")
    ax.set_title(f"{titol}\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_skewt(p, T, Td, u, v, titol):
    fig = plt.figure(figsize=(9, 9), dpi=150)
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.8, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)
    skew.plot(p, T, 'r', lw=2, label='Temperatura')
    skew.plot(p, Td, 'g', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03)
    skew.plot_dry_adiabats(color='brown', linestyle='--', alpha=0.6)
    skew.plot_moist_adiabats(color='blue', linestyle='--', alpha=0.6)
    skew.plot_mixing_lines(color='green', linestyle='--', alpha=0.6)
    prof = mpcalc.parcel_profile(p, T[0], Td[0])
    skew.plot(p, prof, 'k', linewidth=2, label='Trajectòria Parcel·la')
    skew.shade_cape(p, T, prof, color='red', alpha=0.3)
    skew.shade_cin(p, T, prof, color='blue', alpha=0.3)
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
    h.plot(u.to('kt'), v.to('kt'), color='red', linewidth=2)
    ax.set_title("Hodògraf", weight='bold')
    return fig
    
def mostrar_imatge_temps_real(tipus):
    """Mostra imatges de satèl·lit o radar. Canvia a satèl·lit infraroig durant la nit."""
    url_a_utilitzar = ""
    caption_a_utilitzar = ""

    if tipus == "Radar":
        url_a_utilitzar = "https://www.meteociel.fr/cartes_obs/radar/lastradar_sp_ne.gif"
        caption_a_utilitzar = "Radar de precipitació (NE Península). Font: Meteociel"
    
    elif tipus == "Satèl·lit":
        now_local = datetime.now(TIMEZONE)
        # Comprova si l'hora actual està entre les 22:00 i les 06:59
        if now_local.hour >= 22 or now_local.hour < 7:
            # Període nocturn: mostra l'animació infraroja
            url_a_utilitzar = "https://modeles20.meteociel.fr/satellite/animsatircolmtgsp.gif"
            caption_a_utilitzar = "Satèl·lit infraroig (animació nocturna). Font: Meteociel"
        else:
            # Període diürn: mostra la imatge visible estàtica
            url_a_utilitzar = "https://modeles20.meteociel.fr/satellite/latestsatviscolmtgsp.png"
            caption_a_utilitzar = "Satèl·lit visible. Font: Meteociel"

    if url_a_utilitzar:
        try:
            # Afegim un paràmetre 'ver' a l'URL per evitar problemes de cache
            unique_url = f"{url_a_utilitzar}?ver={int(time.time())}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(unique_url, headers=headers, timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption=caption_a_utilitzar, use_container_width=True)
            else:
                st.warning(f"No s'ha pogut carregar la imatge del {tipus.lower()}. (Codi: {response.status_code})")
        except Exception as e:
            st.error(f"Error de xarxa en carregar la imatge del {tipus.lower()}.")


# --- 3. LÒGICA DE LA INTERFÍCIE D'USUARI (UI) ---

def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🌪️ Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Una eina per a la visualització de paràmetres meteorològics clau per al pronòstic de convecció.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Capital de referència:", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        with col2:
            st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3:
            hores_disponibles = [f"{h:02d}:00h" for h in range(24)]
            st.selectbox("Hora del pronòstic (Hora Local):", options=hores_disponibles, key="hora_selector")

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
            mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
            map_key = map_options[mapa_sel]
            error_map = None
            
            if map_key == "cape":
                map_data, error_map = carregar_dades_mapa(["cape"], hourly_index_sel)
                if map_data:
                    # Lògica per a l'escala de colors dinàmica
                    if map_data['cape']:
                        max_cape = np.max(map_data['cape'])
                    else:
                        max_cape = 0

                    if max_cape <= 500:
                        cape_levels = np.arange(50, 501, 50)  # Més detall per a valors baixos
                    elif max_cape <= 1500:
                        cape_levels = np.arange(100, 1501, 100) # Escala estàndard per a Catalunya
                    elif max_cape <= 2500:
                        cape_levels = np.arange(250, 2501, 250) # Per a esdeveniments més severs
                    else:
                        # Per a valors extrems, arrodoneix el màxim cap amunt al següent múltiple de 500
                        rounded_max = np.ceil(max_cape / 500) * 500
                        cape_levels = np.arange(250, rounded_max + 1, 250)

                    fig = crear_mapa_escalar(map_data['lons'], map_data['lats'], map_data['cape'], "CAPE", "plasma", cape_levels, "J/kg", timestamp_str, extend='max')
                    st.pyplot(fig)

            elif map_key == "conv":
                nivell_sel = st.selectbox("Nivell d'anàlisi:", options=[1000, 950, 925, 850], format_func=lambda x: f"{x} hPa")
                variables = [f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data:
                    fig = crear_mapa_convergencia(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell_sel, lat_sel, lon_sel, poble_sel, timestamp_str)
                    st.pyplot(fig)

            elif map_key == "500hpa":
                variables = ["temperature_500hPa", "wind_speed_500hPa", "wind_direction_500hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data: st.pyplot(crear_mapa_500hpa(map_data, timestamp_str))
            
            elif map_key in ["wind_300", "wind_700"]:
                nivell_hpa = int(map_key.split('_')[1])
                variables = [f"wind_speed_{nivell_hpa}hPa", f"wind_direction_{nivell_hpa}hPa"]
                map_data, error_map = carregar_dades_mapa(variables, hourly_index_sel)
                if map_data:
                    fig = crear_mapa_vents_velocitat(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell_hpa, timestamp_str)
                    st.pyplot(fig)
            
            elif map_key == "rh_700":
                map_data, error_map = carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
                if map_data:
                    fig = crear_mapa_escalar(map_data['lons'], map_data['lats'], map_data['relative_humidity_700hPa'], "Humitat Relativa a 700hPa", "Greens", np.arange(50, 101, 5), "%", timestamp_str, extend='max')
                    st.pyplot(fig)

            if error_map:
                st.error(f"Error en carregar el mapa: {error_map}")

        with col_map_2:
            st.subheader("Imatges en Temps Real")
            view_choice = st.radio("Selecciona la vista:", ("Satèl·lit", "Radar"), horizontal=True, label_visibility="collapsed")
            mostrar_imatge_temps_real(view_choice)

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        st.info("Els paràmetres i gràfics següents es calculen a partir del perfil atmosfèric per al punt i hora seleccionats.")
        cols = st.columns(4)
        metric_map = {'CAPE': 'J/kg', 'CIN': 'J/kg', 'Shear_0-6km': 'm/s', 'SRH_0-3km': 'm²/s²'}
        for i, (param, unit) in enumerate(metric_map.items()):
            val = params_calculats.get(param)
            val_str = f"{val:.0f}" if val is not None else "---"
            cols[i].metric(label=param, value=f"{val_str} {unit}")
        with st.expander("ℹ️ Què signifiquen aquests paràmetres?"):
            st.markdown("""
            - **CAPE (Convective Available Potential Energy):** Mesura l'energia disponible per a una parcel·la d'aire ascendent. Valors alts (>1000 J/kg) indiquen potencial per a tempestes fortes.
            - **CIN (Convective Inhibition):** Representa l'energia necessària per iniciar la convecció. Actua com una "tapa". Valors alts poden impedir la formació de tempestes.
            - **Shear 0-6km (Cisallament del vent):** És la diferència en el vector del vent entre la superfície i els 6 km d'altura. Valors alts (>15-20 m/s) són cruials per a l'organització de les tempestes (supercèl·lules, línies de torbonada).
            - **SRH 0-3km (Storm-Relative Helicity):** Mesura el potencial de rotació en una tempesta. Valors elevats (>150 m²/s²) afavoreixen el desenvolupament de supercèl·lules i tornados.
            
            ---

            **Nota important sobre la discrepància de CAPE:**

            És possible que el valor de CAPE calculat aquí (i l'àrea vermella del gràfic) sigui significativament més alt que el que es mostra al mapa per al mateix punt. Això passa perquè **aquest sondeig calcula el CAPE teòric màxim** a partir de les dades brutes del perfil vertical. En canvi, el **mapa mostra un valor de CAPE que el propi model AROME ha processat internament**, el qual pot ser més conservador.

            No és un error: el sondeig revela el potencial real de la massa d'aire en aquest punt, mentre que el mapa ofereix una visió més general i suavitzada.
            """)
        st.divider()
        col_sondeig_1, col_sondeig_2 = st.columns(2)
        with col_sondeig_1:
            titol_skewt = f"Sondeig Vertical (Skew-T) - {poble_sel}"
            fig_skewt = crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], titol_skewt)
            st.pyplot(fig_skewt)
        with col_sondeig_2:
            fig_hodo = crear_hodograf(sounding_data[3], sounding_data[4])
            st.pyplot(fig_hodo)
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual. Pot ser degut a dades no vàlides del model o a una petició fallida.")

# --- NOU ---
def ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    st.subheader(f"Assistent d'Anàlisi per IA per a {timestamp_str}")
    
    if not GEMINI_CONFIGURAT:
        st.error("Funcionalitat no disponible. La clau API de Google no s'ha trobat. Si us plau, configura el fitxer `.streamlit/secrets.toml`.")
        return

    st.info("Aquest assistent utilitza IA per interpretar les dades del model i generar un resum del potencial de temps sever. La seva anàlisi no substitueix el criteri d'un predictor humà.")
    
    if st.button("🤖 Generar Anàlisi d'IA", use_container_width=True):
        with st.spinner("L'assistent d'IA està analitzant les dades... Aquest procés pot trigar uns segons."):
            # 1. Preparar les dades
            dades_ia, error_dades = preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel)
            
            if error_dades:
                st.error(f"No s'ha pogut generar l'anàlisi perquè falten dades: {error_dades}")
                return
            
            # 2. Generar el resum
            resum_text = generar_resum_ia(dades_ia, poble_sel, timestamp_str)
            
            # 3. Mostrar el resultat
            st.markdown(resum_text)
# --- FI NOU ---

def ui_peu_de_pagina():
    st.divider()
    # --- MODIFICAT ---
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges en temps real via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)
    # --- FI MODIFICAT ---

# --- 4. APLICACIÓ PRINCIPAL ---

def main():
    if 'poble_selector' not in st.session_state: st.session_state.poble_selector = 'Barcelona'
    if 'dia_selector' not in st.session_state: st.session_state.dia_selector = 'Avui'
    if 'hora_selector' not in st.session_state: st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"

    ui_capcalera_selectors()

    poble_sel = st.session_state.poble_selector
    dia_sel = st.session_state.dia_selector
    hora_sel = st.session_state.hora_selector
    
    hora_int = int(hora_sel.split(':')[0])
    now_local = datetime.now(TIMEZONE)
    target_date = now_local.date()
    if dia_sel == "Demà":
        target_date += timedelta(days=1)
    
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int))
    utc_dt = local_dt.astimezone(pytz.utc)
    
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600)
    hourly_index_sel = max(0, time_diff_hours)

    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
    lat_sel = CIUTATS_CATALUNYA[poble_sel]['lat']
    lon_sel = CIUTATS_CATALUNYA[poble_sel]['lon']

    # --- MODIFICAT ---
    # Carreguem el sondeig aquí per tenir-lo disponible per a la pestanya vertical
    with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
        data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if error_msg:
        st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")

    # Creació de les pestanyes
    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Resum IA"])

    with tab_mapes:
        ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)

    with tab_vertical:
        ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
        
    with tab_ia:
        ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    # --- FI MODIFICAT ---
        
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
