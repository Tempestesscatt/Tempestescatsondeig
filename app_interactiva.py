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

st.set_page_config(layout="wide", page_title="Terminal de Temps Sever | Catalunya")

# Configuració de l'API de Gemini (si està disponible)
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_CONFIGURAT = True
except (KeyError, Exception):
    GEMINI_CONFIGURAT = False

# Configuració de la sessió de requests amb cache i reintents
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Constants de l'aplicació
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
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)


# --- 1. FUNCIONS D'OBTENCIÓ I PROCESSAMENT DE DADES ---

def vector_to_direction(u, v):
    """Converteix un vector de vent (u, v) a direcció cardinal i velocitat."""
    if np.isnan(u) or np.isnan(v):
        return "N/A", np.nan
    speed_ms = np.sqrt(u**2 + v**2)
    direction_deg = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    cardinal_dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return cardinal_dirs[int(round(direction_deg / 22.5))], speed_ms

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

        if len(p_profile) < 5: return None, "Perfil atmosfèric massa curt per a càlculs fiables."

        p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC; Td = np.array(Td_profile) * units.degC
        u = np.array(u_profile) * units('m/s'); v = np.array(v_profile) * units('m/s'); h = np.array(h_profile) * units.meter

        params_calc = {}
        prof = mpcalc.parcel_profile(p, T[0], Td[0])
        cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0
        params_calc['CIN'] = cin.to('J/kg').m

        try: # Càlcul de nivells característics
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); params_calc['LCL_p'] = lcl_p.to('hPa').m
            lfc_p, _ = mpcalc.lfc(p, T, Td, which='most_cape'); params_calc['LFC_p'] = lfc_p.to('hPa').m
            el_p, _ = mpcalc.el(p, T, Td); params_calc['EL_p'] = el_p.to('hPa').m
        except Exception:
             params_calc.update({'LCL_p': np.nan, 'LFC_p': np.nan, 'EL_p': np.nan})

        s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6 * units.km)
        params_calc['Shear_0-6km'] = mpcalc.wind_speed(s_u, s_v).to('m/s').m

        try: # Càlcul del moviment de la tempesta (Bunkers Right-Mover)
            rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p, u, v, h)
            params_calc['storm_motion_u'], params_calc['storm_motion_v'] = rm[0].to('m/s').m, rm[1].to('m/s').m
            storm_motion_vector = rm
        except Exception:
            params_calc.update({'storm_motion_u': np.nan, 'storm_motion_v': np.nan})
            storm_motion_vector = (np.nan*units('m/s'), np.nan*units('m/s'))

        # --- LÍNIA CORREGIDA ---
        storm_u_comp, storm_v_comp = storm_motion_vector
        _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=3 * units.km, storm_u=storm_u_comp, storm_v=storm_v_comp)
        params_calc['SRH_0-3km'] = np.sum(srh).to('m^2/s^2').m

        return ((p, T, Td, u, v, h), params_calc), None
    except Exception as e:
        return None, f"Error crític en processar dades del sondeig: {e}"


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
                output["lats"].append(r.Latitude())
                output["lons"].append(r.Longitude())
                for i, var in enumerate(variables): output[var].append(vals[i])
        if not output["lats"]: return None, "No s'han rebut dades vàlides del model per a aquesta hora."
        return output, None
    except Exception as e:
        return None, f"Error en carregar dades del mapa: {e}"

@st.cache_data(ttl=3600)
def preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel):
    dades_ia = {}
    data_tuple, error_sondeig = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if not data_tuple: return None, f"Falten dades del sondeig ({error_sondeig})"
    dades_ia['sondeig'] = data_tuple[1]

    variables_mapa = ["cape", "wind_speed_925hPa", "wind_direction_925hPa"]
    map_data, error_mapa = carregar_dades_mapa(variables_mapa, hourly_index_sel)
    if not map_data: return None, f"Falten dades del mapa ({error_mapa})"

    resum_mapa = {'max_cape_catalunya': max(map_data['cape']) if 'cape' in map_data and map_data['cape'] else 0}

    if 'wind_speed_925hPa' in map_data and map_data['wind_speed_925hPa']:
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
            idx_min = np.nanargmin(divergence)
            idx_2d = np.unravel_index(idx_min, divergence.shape)
            resum_mapa.update({'lat_max_conv': grid_lat[idx_2d], 'lon_max_conv': grid_lon[idx_2d]})
        except Exception:
            resum_mapa.update({'max_conv_925hpa': 0, 'lat_max_conv': 0, 'lon_max_conv': 0})

    dades_ia['mapa_resum'] = resum_mapa
    return dades_ia, None

@st.cache_data(ttl=3600)
def generar_resum_ia(_dades_ia, _poble_sel, _timestamp_str):
    if not GEMINI_CONFIGURAT: return "Error: La clau API de Google no està configurada."

    model = genai.GenerativeModel('gemini-1.5-flash')
    mapa, sondeig = _dades_ia.get('mapa_resum', {}), _dades_ia.get('sondeig', {})

    direction_str, speed_ms = vector_to_direction(sondeig.get('storm_motion_u', 0), sondeig.get('storm_motion_v', 0))
    moviment_tempesta_str = f"{direction_str} a {int(speed_ms * 3.6)} km/h" if not np.isnan(speed_ms) else "No disponible"

    prompt = f"""
    Ets un assistent de meteorologia expert, directe i concís. La teva única tasca és analitzar les dades del model AROME per a Catalunya i generar un avís curt, clar i útil.

    **DADES:**
    - Hora de l'anàlisi: {_timestamp_str}
    - CAPE màxim (energia): {int(mapa.get('max_cape_catalunya', 0))} J/kg
    - Convergència màxima a 925hPa ("disparador"): {mapa.get('max_conv_925hpa', 0):.2f} (x10⁻⁵ s⁻¹)
    - Latitud del focus de convergència: {mapa.get('lat_max_conv', 0):.2f}
    - Longitud del focus de convergència: {mapa.get('lon_max_conv', 0):.2f}
    - Cisallament 0-6km (organització): {int(sondeig.get('Shear_0-6km', 0))} m/s
    - SRH 0-3km (rotació): {int(sondeig.get('SRH_0-3km', 0))} m²/s²
    - Moviment previst de la tempesta (Bunkers RM): {moviment_tempesta_str}

    **INSTRUCCIONS (MOLT IMPORTANT):**
    Vés directament al gra. Prohibides les frases llargues o explicacions tècniques complexes.

    1.  **Resumeix el Risc:** Comença amb una única frase que defineixi el nivell de risc (Baix, Moderat, Alt, Molt Alt) de tempestes.
    2.  **Anomena les Poblacions Clau:** La teva tasca més important. Utilitzant el teu coneixement geogràfic de Catalunya, identifica 3-5 ciutats o pobles importants a prop de les coordenades del "focus de convergència". Aquest és l'origen més probable. Després, considera el "moviment previst de la tempesta" per anticipar la trajectòria i esmentar zones posteriors.
    3.  **Justificació Breu:** Explica en una frase per què hi ha risc, connectant el "disparador" (convergència) amb el "combustible" (CAPE) i l'organització (Cisallament/SRH).

    **FORMAT DE SORTIDA OBLIGATORI (utilitza Markdown):**
    **Resum del Risc:** [La teva frase de resum aquí]
    **Poblacions Potencialment Afectades:** [Llista de 3-5 poblacions separades per comes, considerant origen i trajectòria]
    **Justificació Tècnica (Molt Breu):** [La teva única frase d'explicació aquí]
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"S'ha produït un error en contactar amb l'assistent d'IA: {e}"

# --- 2. FUNCIONS DE VISUALITZACIÓ ---

def crear_mapa_base():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    return fig, ax

def crear_mapa_convergencia(map_data, nivell, lat_sel, lon_sel, nom_poble_sel, timestamp_str):
    fig, ax = crear_mapa_base()
    lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
    speed_data, dir_data = map_data[f'wind_speed_{nivell}hPa'], map_data[f'wind_direction_{nivell}hPa']
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100))
    speeds_ms=np.array(speed_data)*units('km/h'); dirs_deg=np.array(dir_data)*units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    divergence = mpcalc.divergence(grid_u*units('m/s'), grid_v*units('m/s'), dx=dx, dy=dy) * 1e5
    levels = np.linspace(-15, 15, 21)
    cf = ax.contourf(grid_lon, grid_lat, divergence, levels=levels, cmap='coolwarm_r', alpha=0.6, zorder=2, extend='both')
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7); cbar.set_label('Convergència (vermell) / Divergència (blau) [x10⁻⁵ s⁻¹]')
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density=2.5, arrowsize=0.7, zorder=4)
    ax.plot(lon_sel, lat_sel, 'o', markerfacecolor='yellow', markeredgecolor='black', markersize=8, transform=ccrs.Geodetic(), zorder=6)
    txt = ax.text(lon_sel + 0.05, lat_sel, nom_poble_sel, transform=ccrs.Geodetic(), zorder=7, fontsize=10, weight='bold')
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    max_conv = np.nanmin(divergence)
    ax.set_title(f"Flux i Convergència a {nivell}hPa (Mín: {max_conv:.1f})\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_mapa_escalar(map_data, var, titol, cmap, levels, unitat, timestamp_str, extend='max'):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    grid_data = griddata((map_data['lons'], map_data['lats']), map_data[var], (grid_lon, grid_lat), method='cubic')
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    cf = ax.contourf(grid_lon, grid_lat, grid_data, levels=levels, cmap=cmap, norm=norm, alpha=0.8, zorder=2, extend=extend)
    contorns = ax.contour(grid_lon, grid_lat, grid_data, levels=levels[::(len(levels)//5) if len(levels)>5 else 1], colors='black', linewidths=0.7, alpha=0.9, zorder=3)
    ax.clabel(contorns, inline=True, fontsize=8, fmt='%1.0f')
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f"{titol} ({unitat})")
    ax.set_title(f"{titol}\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_skewt(p, T, Td, u, v, titol, params_calc):
    fig = plt.figure(figsize=(9, 9), dpi=150)
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.8, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)
    skew.plot(p, T, 'r', lw=2, label='Temperatura'); skew.plot(p, Td, 'g', lw=2, label='Punt de Rosada')
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03, length=6)
    skew.plot_dry_adiabats(color='brown', linestyle='--', alpha=0.5); skew.plot_moist_adiabats(color='blue', linestyle='--', alpha=0.5); skew.plot_mixing_lines(color='green', linestyle='--', alpha=0.5)
    prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', linewidth=2, label='Trajectòria Parcel·la')
    skew.shade_cape(p, T, prof, color='red', alpha=0.3); skew.shade_cin(p, T, prof, color='blue', alpha=0.3)

    # Afegir nivells característics
    for level_name, color in [('LCL_p', 'orange'), ('LFC_p', 'darkred'), ('EL_p', 'purple')]:
        if level_name in params_calc and not np.isnan(params_calc[level_name]):
            p_level = params_calc[level_name]
            skew.ax.axhline(p_level, color=color, linestyle='--', lw=2)
            skew.ax.text(skew.ax.get_xlim()[1], p_level, f" {level_name.split('_')[0]}", color=color, ha='left', va='center', weight='bold', fontsize=10)

    skew.ax.set_ylim(1000, 150); skew.ax.set_xlim(-30, 40)
    skew.ax.set_title(titol, weight='bold', fontsize=14); skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")
    skew.ax.legend()
    return fig

def crear_hodograf(u, v, params_calc):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    h = Hodograph(ax, component_range=80.)
    h.add_grid(increment=20, color='gray', linestyle='--')
    h.plot(u.to('kt'), v.to('kt'), color='red', linewidth=2)

    # Dibuixar vector de moviment de la tempesta (Bunkers Right-Mover)
    if 'storm_motion_u' in params_calc and not np.isnan(params_calc['storm_motion_u']):
        sm_u_kt = params_calc['storm_motion_u'] * (units('m/s')).to('kt').m
        sm_v_kt = params_calc['storm_motion_v'] * (units('m/s')).to('kt').m
        ax.arrow(0, 0, sm_u_kt, sm_v_kt, head_width=2, head_length=2, fc='darkred', ec='black', zorder=10, lw=1.5, label='Moviment Tempesta (RM)')

        direction_str, speed_ms = vector_to_direction(params_calc['storm_motion_u'], params_calc['storm_motion_v'])
        ax.text(0.03, 0.97, f"Mov. Tempesta:\n {direction_str} a {speed_ms*3.6:.0f} km/h", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_title("Hodògraf i Moviment de Tempesta", weight='bold'); ax.set_xlabel("Component U (kt)"); ax.set_ylabel("Component V (kt)")
    ax.legend()
    return fig

def mostrar_imatge_temps_real(tipus):
    if tipus == "Radar":
        url, caption = "https://www.meteociel.fr/cartes_obs/radar/lastradar_sp_ne.gif", "Radar de precipitació. Font: Meteociel"
    else:
        now_local = datetime.now(TIMEZONE)
        url, caption = ("https://modeles20.meteociel.fr/satellite/animsatircolmtgsp.gif", "Satèl·lit infraroig. Font: Meteociel") if now_local.hour >= 22 or now_local.hour < 7 else ("https://modeles20.meteociel.fr/satellite/latestsatviscolmtgsp.png", "Satèl·lit visible. Font: Meteociel")
    try:
        response = requests.get(f"{url}?ver={int(time.time())}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if response.status_code == 200: st.image(response.content, caption=caption, use_container_width=True)
        else: st.warning(f"No s'ha pogut carregar la imatge del {tipus.lower()}. (Codi: {response.status_code})")
    except Exception: st.error(f"Error de xarxa en carregar la imatge del {tipus.lower()}.")

# --- 3. LÒGICA DE LA INTERFÍCIE D'USUARI (UI) ---

def ui_capcalera_selectors():
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🌪️ Terminal d\'Anàlisi de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Eina per a la visualització de paràmetres clau per al pronòstic de convecció, basada en el model <b>AROME</b>.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.selectbox("Capital de referència:", sorted(CIUTATS_CATALUNYA.keys()), key="poble_selector")
        with col2: st.selectbox("Dia del pronòstic:", ("Avui", "Demà"), key="dia_selector")
        with col3: st.selectbox("Hora del pronòstic (Hora Local):", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector")

def ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    with st.container(border=True):
        col_map_1, col_map_2 = st.columns([2.5, 1.5])
        with col_map_1:
            with st.spinner("Actualitzant anàlisi de mapes..."):
                map_options = {"CAPE (Energia)": "cape", "Flux i Convergència (Disparador)": "conv", "Humitat a 700hPa": "rh_700"}
                mapa_sel = st.selectbox("Selecciona la capa del mapa:", map_options.keys())
                map_key = map_options[mapa_sel]

                if map_key == "cape":
                    map_data, error = carregar_dades_mapa(["cape"], hourly_index_sel)
                    if map_data:
                        max_cape = np.max(map_data['cape']) if map_data['cape'] else 0
                        cape_levels = np.arange(100, max(1001, np.ceil(max_cape/250+1)*250), 250)
                        st.pyplot(crear_mapa_escalar(map_data, "cape", "CAPE", "plasma", cape_levels, "J/kg", timestamp_str))
                    elif error: st.error(f"Error en carregar el mapa: {error}")

                elif map_key == "conv":
                    nivell_sel = st.selectbox("Nivell d'anàlisi:", options=[950, 925, 850], format_func=lambda x: f"{x} hPa")
                    variables = [f"wind_speed_{nivell_sel}hPa", f"wind_direction_{nivell_sel}hPa"]
                    map_data, error = carregar_dades_mapa(variables, hourly_index_sel)
                    if map_data: st.pyplot(crear_mapa_convergencia(map_data, nivell_sel, lat_sel, lon_sel, poble_sel, timestamp_str))
                    elif error: st.error(f"Error en carregar el mapa: {error}")

                elif map_key == "rh_700":
                    map_data, error = carregar_dades_mapa(["relative_humidity_700hPa"], hourly_index_sel)
                    if map_data: st.pyplot(crear_mapa_escalar(map_data, "relative_humidity_700hPa", "Humitat Relativa a 700hPa", "Greens", np.arange(50, 101, 10), "%", timestamp_str))
                    elif error: st.error(f"Error en carregar el mapa: {error}")
        with col_map_2:
            st.subheader("Imatges en Temps Real")
            view_choice = st.radio("Selecciona la vista:", ("Satèl·lit", "Radar"), horizontal=True, label_visibility="collapsed")
            mostrar_imatge_temps_real(view_choice)

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    with st.container(border=True):
        if data_tuple:
            sounding_data, params = data_tuple
            st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
            cols = st.columns(4)
            for i, (param, unit) in enumerate({'CAPE': 'J/kg', 'CIN': 'J/kg', 'Shear_0-6km': 'm/s', 'SRH_0-3km': 'm²/s²'}.items()):
                val = params.get(param)
                cols[i].metric(label=param, value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} {unit}")
            with st.expander("ℹ️ Què signifiquen aquests paràmetres?"):
                st.markdown("""
                - **CAPE:** Energia per a tempestes. >1000 J/kg és significatiu.
                - **CIN:** "Tapa" que impedeix la convecció. Valors negatius petits afavoreixen l'inici.
                - **Shear 0-6km:** Cisallament (diferència de vent amb l'altura). >15-20 m/s afavoreix l'organització (supercèl·lules).
                - **SRH 0-3km:** Helicitat (potencial de rotació). >150 m²/s² afavoreix supercèl·lules i tornados.
                - **LCL, LFC, EL:** Nivells clau del sondeig que indiquen la base del núvol (LCL), on comença l'ascens lliure (LFC) i el cim de la tempesta (EL).
                """)
            st.divider()
            col1, col2 = st.columns([1.5, 1])
            with col1: st.pyplot(crear_skewt(*sounding_data, f"Sondeig Vertical - {poble_sel}", params))
            with col2: st.pyplot(crear_hodograf(sounding_data[3], sounding_data[4], params))
        else:
            st.warning("No hi ha dades de sondeig disponibles per a la selecció actual. Pot ser degut a dades invàlides del model.")

def ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str):
    with st.container(border=True):
        st.subheader(f"Assistent d'Anàlisi per IA")
        st.info("Aquest assistent utilitza Google Gemini per interpretar les dades meteorològiques i generar un resum concís del risc de temps sever, identificant les zones més probables.", icon="🤖")
        if not GEMINI_CONFIGURAT:
            st.error("Funcionalitat no disponible. La clau API de Google no està configurada correctament a `.streamlit/secrets.toml`.")
            return
        if st.button("Generar Anàlisi d'IA", use_container_width=True, type="primary"):
            with st.spinner("L'assistent d'IA està analitzant les dades... Aquesta operació pot trigar uns segons."):
                dades_ia, error = preparar_dades_per_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel)
                if error: st.error(f"No s'ha pogut generar l'anàlisi: {error}")
                else: st.markdown(generar_resum_ia(dades_ia, poble_sel, timestamp_str))

def ui_peu_de_pagina():
    st.divider()
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- 4. APLICACIÓ PRINCIPAL ---

def main():
    if 'hora_selector' not in st.session_state:
        st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"

    ui_capcalera_selectors()

    poble_sel = st.session_state.poble_selector; dia_sel = st.session_state.dia_selector; hora_sel = st.session_state.hora_selector

    hora_int = int(hora_sel.split(':')[0])
    now_local = datetime.now(TIMEZONE)
    target_date = now_local.date() + timedelta(days=(1 if dia_sel == "Demà" else 0))
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int))
    start_of_forecast_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = max(0, int((local_dt.astimezone(pytz.utc) - start_of_forecast_utc).total_seconds() / 3600))

    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
    lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']

    # Mostra un missatge d'espera mentre es carreguen les dades inicials
    with st.spinner('Carregant dades del sondeig inicial...'):
        data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
        if error_msg: 
            st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")

    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Resum IA"])

    with tab_mapes: ui_pestanya_mapes(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)
    with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia: ui_pestanya_ia(poble_sel, lat_sel, lon_sel, hourly_index_sel, timestamp_str)

    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
