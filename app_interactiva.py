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
from scipy.ndimage import label
import google.generativeai as genai

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
# SUBSTITUEIX LA TEVA VERSIÓ D'AQUESTA FUNCIÓ PER AQUESTA
# SUBSTITUEIX LA TEVA FUNCIÓ ANTIGA PER AQUESTA VERSIÓ FINAL
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
        u_profile, v_profile = [sfc_u.to('m/s').m], [sfc_v.to('m/s').m]
        h_profile = [0.0] 
        
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
        
        # Combinem i ordenem les dades per pressió descendent (altitud creixent)
        combined_data = sorted(zip(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile), key=lambda x: x[0], reverse=True)
        p_sorted, T_sorted, Td_sorted, u_sorted, v_sorted, h_sorted = [np.array(col) for col in zip(*combined_data)]
        
        # Creem els arrays amb unitats
        p = p_sorted * units.hPa
        T = T_sorted * units.degC
        Td = Td_sorted * units.degC
        u = u_sorted * units('m/s')
        v = v_sorted * units('m/s')
        heights = h_sorted * units.meter
        
        # Ara els càlculs es fan amb les dades ordenades
        prof = mpcalc.parcel_profile(p, T[0], Td[0])
        params_calc = {}
        cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        params_calc['CAPE'] = cape.to('J/kg').m if cape.magnitude > 0 else 0
        params_calc['CIN'] = cin.to('J/kg').m
        
        try:
            p_lfc, _ = mpcalc.lfc(p, T, Td)
            params_calc['LFC_hPa'] = p_lfc.m if not np.isnan(p_lfc.m) else np.nan
        except Exception: params_calc['LFC_hPa'] = np.nan
        
        # NOU MÈTODE DE CÀLCUL DE CISALLAMENT (MÉS ROBUST)
        params_calc['Shear 0-1km'] = np.nan
        params_calc['Shear 0-6km'] = np.nan
        
        try:
            # Seleccionem la capa de 0 a 1 km sobre el terra
            layer_1km = mpcalc.get_layer_agl(p, heights, depth=1000 * units.m)
            shear_1km_u, shear_1km_v = mpcalc.wind_shear(p[layer_1km], u[layer_1km], v[layer_1km])
            params_calc['Shear 0-1km'] = mpcalc.wind_speed(shear_1km_u, shear_1km_v).to('knots').m
        except (ValueError, IndexError):
            # Aquest error pot passar si el sondeig és massa curt
            pass

        try:
            # Seleccionem la capa de 0 a 6 km sobre el terra
            layer_6km = mpcalc.get_layer_agl(p, heights, depth=6000 * units.m)
            shear_6km_u, shear_6km_v = mpcalc.wind_shear(p[layer_6km], u[layer_6km], v[layer_6km])
            params_calc['Shear 0-6km'] = mpcalc.wind_speed(shear_6km_u, shear_6km_v).to('knots').m
        except (ValueError, IndexError):
            pass
            
        # Retornem les dades originals (sense ordenar) per als gràfics Skew-T, que ho gestionen internament
        p_orig = np.array(p_profile) * units.hPa
        T_orig = np.array(T_profile) * units.degC
        Td_orig = np.array(Td_profile) * units.degC
        u_orig = np.array(u_profile) * units('m/s')
        v_orig = np.array(v_profile) * units('m/s')
        
        return ((p_orig, T_orig, Td_orig, u_orig, v_orig), params_calc), None
        
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
            
            lons, lats = map_data_raw['lons'], map_data_raw['lats']
            speed_data, dir_data = map_data_raw[f"wind_speed_{nivell}hPa"], map_data_raw[f"wind_direction_{nivell}hPa"]
            dewpoint_data = map_data_raw['dew_point_2m']
        else:
            variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base(variables, hourly_index)
            if error: return None, error

            lons, lats = map_data_raw['lons'], map_data_raw['lats']
            speed_data, dir_data = map_data_raw[f"wind_speed_{nivell}hPa"], map_data_raw[f"wind_direction_{nivell}hPa"]
            temp_data = np.array(map_data_raw[f'temperature_{nivell}hPa']) * units.degC
            rh_data = np.array(map_data_raw[f'relative_humidity_{nivell}hPa']) * units.percent
            dewpoint_data = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m

        grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100))
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='linear')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='linear')
        grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), method='linear')
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
        if nivell >= 950: CONVERGENCE_THRESHOLD = -45; DEWPOINT_THRESHOLD_FOR_RISK = 14
        elif nivell >= 925: CONVERGENCE_THRESHOLD = -35; DEWPOINT_THRESHOLD_FOR_RISK = 12
        elif nivell >= 850: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 7
        else: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 5
        effective_risk_mask = (divergence.magnitude <= CONVERGENCE_THRESHOLD) & (grid_dewpoint >= DEWPOINT_THRESHOLD_FOR_RISK)
        labels, num_features = label(effective_risk_mask)

        output_data = {
            'lons': lons, 'lats': lats, 'speed_data': speed_data, 'dir_data': dir_data,
            'dewpoint_data': dewpoint_data, 'num_alertes': num_features
        }
        return output_data, None
    except Exception as e: return None, f"Error en processar dades del mapa: {e}"


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
    else: CONVERGENCE_THRESHOLD = -25; DEWPOINT_THRESHOLD_FOR_RISK = 5
    colors_wind_final = ['#FFFFFF', '#B0E0E6', '#00FFFF', '#3CB371', '#32CD32', '#ADFF2F', '#FFD700', '#F4A460', '#CD853F', '#A0522D', '#DC143C', '#8B0000', '#800080', '#FF00FF', '#FFC0CB', '#D3D3D3', '#A9A9A9']
    speed_levels_final = np.arange(0, 171, 10)
    custom_cmap = ListedColormap(colors_wind_final); norm_speed = BoundaryNorm(speed_levels_final, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=speed_levels_final[::2])
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.6, density=4, arrowsize=0.7, zorder=4, transform=ccrs.PlateCarree())
    effective_risk_mask = (divergence.magnitude <= CONVERGENCE_THRESHOLD) & (grid_dewpoint >= DEWPOINT_THRESHOLD_FOR_RISK)
    labels, num_features = label(effective_risk_mask)
    if num_features > 0:
        for i in range(1, num_features + 1):
            points = np.argwhere(labels == i); center_y, center_x = points.mean(axis=0)
            center_lon, center_lat = grid_lon[0, int(center_x)], grid_lat[int(center_y), 0]
            warning_txt = ax.text(center_lon, center_lat, '⚠️', color='yellow', fontsize=15, ha='center', va='center', zorder=8, transform=ccrs.PlateCarree())
            warning_txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
    ax.set_title(f"Anàlisi de Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def crear_mapa_vents(lons, lats, speed_data, dir_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base()
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic'); grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), method='cubic')
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


# --- 3. FUNCIONS PER A L'ASSISTENT D'IA ---
def preparar_resum_dades_per_ia(data_tuple, map_data, nivell_mapa, poble_sel, timestamp_str):
    resum_sondeig = "No s'han pogut carregar les dades del sondeig."
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        cape = params_calculats.get('CAPE', 0)
        cin = params_calculats.get('CIN', 0)
        lfc = params_calculats.get('LFC_hPa', float('nan'))
        shear_1km = params_calculats.get('Shear 0-1km', np.nan)
        shear_6km = params_calculats.get('Shear 0-6km', np.nan)
        
        resum_sondeig = f"""
    - CAPE (Energia): {cape:.0f} J/kg.
    - CIN (Inhibidor): {cin:.0f} J/kg.
    - LFC (Nivell d'inici): {'No trobat' if np.isnan(lfc) else f'{lfc:.0f} hPa'}.
    - Cisallament 0-1km (Tornados): {'No calculat' if np.isnan(shear_1km) else f'{shear_1km:.0f} nusos'}.
    - Cisallament 0-6km (Supercèl·lules): {'No calculat' if np.isnan(shear_6km) else f'{shear_6km:.0f} nusos'}."""

    resum_mapa = "No s'han pogut carregar les dades del mapa."
    if map_data:
        num_alertes = map_data.get('num_alertes', 0)
        resum_mapa = f"S'han detectat {num_alertes} focus de convergència amb humitat a {nivell_mapa}hPa, indicant zones amb potencial per iniciar tempestes." if num_alertes > 0 else f"No es detecten focus significatius de convergència amb humitat a {nivell_mapa}hPa."
    
    resum_final = f"""
    CONTEXT DE L'ANÀLISI:
    - Lloc de referència (per al sondeig): {poble_sel}
    - Data i Hora: {timestamp_str}
    DADES DEL SONDEIG VERTICAL (per a {poble_sel}):{resum_sondeig}
    DADES DEL MAPA (per a tot Catalunya a {nivell_mapa}hPa):
    - {resum_mapa}
    INSTRUCCIONS PER A L'ASSISTENT:
    Ets un meteoròleg expert en temps sever a Catalunya, anomenat MeteoIA.
    Respon les preguntes basant-te ÚNICAMENT en les dades del sondeig i del mapa. Combina la informació de les dues fonts.
    """
    return resum_final

def generar_resposta_ia(historial_conversa_text, resum_dades, prompt_usuari):
    if not GEMINI_CONFIGURAT: return "La funcionalitat d'IA no està configurada."
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt_final = resum_dades + f"\n\nHISTORIAL DE LA CONVERSA PREVI:\n{historial_conversa_text}\n\nPREGUNTA ACTUAL DE L'USUARI:\n'{prompt_usuari}'\n\nLA TEVA RESPOSTA COM A METEOIA:"
    try:
        response = model.generate_content(prompt_final)
        return response.text
    except Exception as e:
        print(f"ERROR DETALLAT DE L'API DE GOOGLE: {e}")
        return f"Hi ha hagut un error contactant amb l'IA de Google: {e}"


# --- 4. LÒGICA DE LA INTERFÍCIE D'USUARI ---
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
        st.markdown("""Cada símbol d'alerta **⚠️** assenyala un **focus de risc convectiu**. No és una predicció de tempesta garantida, sinó la detecció d'una zona on es compleix la **"recepta perfecta"** per iniciar-ne una.""")
        col1, col2 = st.columns(2)
        with col1: st.markdown("#### 1. El Disparador: Convergència ↗️\nL'aire a nivells baixos és forçat a ascendir amb intensitat.")
        with col2: st.markdown("#### 2. El Combustible: Humitat 💧\nAquest aire que puja està carregat de vapor d'aigua.")
        st.info("**En resum:** Una ⚠️ indica una zona on un potent **disparador** actua sobre una massa d'aire amb abundant **combustible**.", icon="🎯")

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
            with st.spinner("Carregant i analitzant dades del mapa..."):
                map_data, error_map = carregar_dades_mapa(nivell_sel, hourly_index_sel)
            if error_map: st.error(f"Error en carregar el mapa: {error_map}")
            elif map_data:
                st.pyplot(crear_mapa_forecast_combinat(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str))
                ui_explicacio_alertes()
        elif map_key in ["vent_700", "vent_300"]:
            nivell = 700 if map_key == "vent_700" else 300
            variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data, error_map = carregar_dades_mapa_base(variables, hourly_index_sel)
            if error_map: st.error(f"Error en carregar el mapa: {error_map}")
            elif map_data: st.pyplot(crear_mapa_vents(map_data['lons'], map_data['lats'], map_data[variables[0]], map_data[variables[1]], nivell, timestamp_str))
    with col_map_2:
        st.subheader("Imatges en Temps Real")
        tab_europa, tab_ne = st.tabs(["🇪🇺 Satèl·lit (Europa)", "🛰️ Satèl·lit (NE Península)"])
        with tab_europa: mostrar_imatge_temps_real("Satèl·lit (Europa)")
        with tab_ne: mostrar_imatge_temps_real("Satèl·lit (NE Península)")

def ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel):
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        st.subheader(f"Anàlisi Vertical per a {poble_sel} - {dia_sel} {hora_sel}")
        # MODIFICACIÓ: Afegim més mètriques
        cols = st.columns(5)
        metric_params = {'CAPE': 'J/kg', 'CIN': 'J/kg', 'LFC_hPa': 'hPa', 'Shear 0-1km': 'nusos', 'Shear 0-6km': 'nusos'}
        for i, (param, unit) in enumerate(metric_params.items()):
            val = params_calculats.get(param)
            cols[i].metric(label=param, value=f"{f'{val:.0f}' if val is not None and not np.isnan(val) else '---'} {unit}")
        
        # MODIFICACIÓ: Actualitzem l'explicació
        with st.expander("ℹ️ Què signifiquen aquests paràmetres?"):
            st.markdown("""
            - **CAPE:** Energia disponible per a les tempestes. >1000 J/kg és significatiu.
            - **CIN:** "Tapa" que impedeix la convecció. Valors molt negatius (> -50) són una tapa forta.
            - **LFC:** Nivell on comença la convecció lliure. Com més baix, més fàcil és iniciar tempestes.
            - **Shear 0-1km:** Cisallament a nivells baixos. >15-20 nusos afavoreix la rotació i el risc de **tornados**.
            - **Shear 0-6km:** Cisallament profund. >35-40 nusos és clau per a l'organització de **supercèl·lules**.
            """)
            
        st.divider()
        col1, col2 = st.columns(2)
        with col1: st.pyplot(crear_skewt(sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4], f"Sondeig Vertical - {poble_sel}"))
        with col2: st.pyplot(crear_hodograf(sounding_data[3], sounding_data[4]))
    else: st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def ui_pestanya_ia(data_tuple, hourly_index_sel, poble_sel, timestamp_str):
    st.subheader("💬 Assistent MeteoIA (amb Google Gemini)")
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Escriu la teva pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("MeteoIA està pensant..."):
                historial_text = "\n".join([f'{m["role"]}: {m["content"]}' for m in st.session_state.messages])
                response = generar_resposta_ia(historial_text, resum_dades, prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)

# --- 5. APLICACIÓ PRINCIPAL ---
def main():
    if 'poble_selector' not in st.session_state:
        st.session_state.poble_selector = 'Barcelona'
        st.session_state.dia_selector = 'Avui'
        st.session_state.hora_selector = f"{datetime.now(TIMEZONE).hour:02d}:00h"
        st.session_state.last_selection = ""
    ui_capcalera_selectors()
    current_selection = f"{st.session_state.poble_selector}-{st.session_state.dia_selector}-{st.session_state.hora_selector}"
    if current_selection != st.session_state.last_selection:
        st.session_state.messages = []
        st.session_state.last_selection = current_selection
    poble_sel = st.session_state.poble_selector
    dia_sel = st.session_state.dia_selector
    hora_sel = st.session_state.hora_selector
    hora_int = int(hora_sel.split(':')[0]); now_local = datetime.now(TIMEZONE); target_date = now_local.date()
    if dia_sel == "Demà": target_date += timedelta(days=1)
    local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_int)); utc_dt = local_dt.astimezone(pytz.utc)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    time_diff_hours = int((utc_dt - start_of_today_utc).total_seconds() / 3600); hourly_index_sel = max(0, time_diff_hours)
    timestamp_str = f"{dia_sel} a les {hora_sel} (Hora Local)"
    lat_sel = CIUTATS_CATALUNYA[poble_sel]['lat']
    lon_sel = CIUTATS_CATALUNYA[poble_sel]['lon']
    data_tuple, error_msg = carregar_dades_sondeig(lat_sel, lon_sel, hourly_index_sel)
    if error_msg: st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
    tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Assistent MeteoIA"])
    with tab_mapes: ui_pestanya_mapes(hourly_index_sel, timestamp_str, data_tuple)
    with tab_vertical: ui_pestanya_vertical(data_tuple, poble_sel, dia_sel, hora_sel)
    with tab_ia: ui_pestanya_ia(data_tuple, hourly_index_sel, poble_sel, timestamp_str)
    ui_peu_de_pagina()

if __name__ == "__main__":
    main()
