# -*- coding: utf-8 -*-
# --- 0. IMPORTS I CONFIGURACIÓ INICIAL ---
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
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

# --- CONFIGURACIÓ DE LA PÀGINA ---
st.set_page_config(
    layout="wide",
    page_title="Terminal de Temps Sever | Catalunya",
    page_icon="🌪️"
)

# --- 1. ESTIL I DISSENY (CSS PERSONALITZAT) ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* Tema General */
        html, body, [class*="st-"] {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Títols i Text */
        h1, h2, h3 {
            color: #00BFFF; /* DeepSkyBlue */
        }
        h1 {
            text-shadow: 0 0 10px #00BFFF;
        }
        /* Contenidors i Selectors */
        .st-emotion-cache-1r4qj8v, .st-emotion-cache-z5fcl4 {
            border: 1px solid #2A3B4C;
            border-radius: 10px;
            padding: 1.5rem;
            background-color: #161B22; /* Fons lleugerament més clar */
        }
        /* Botons i Tabs */
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
        /* Estil dels tabs */
        button[data-baseweb="tab"] {
            color: #C0C0C0;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #FF00FF; /* Magenta */
            border-bottom: 3px solid #FF00FF;
            box-shadow: 0 4px 10px -2px #FF00FF;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DEFINICIONS I CONSTANTS ---
@dataclass
class Config:
    """Estructura de dades per emmagatzemar la configuració de l'usuari."""
    city_name: str
    lat: float
    lon: float
    day_offset: int
    hour: int
    hourly_index: int
    timestamp_str: str

# Constants Globals
API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = pytz.timezone('Europe/Madrid')
MAP_EXTENT = [0, 3.5, 40.4, 43]
PRESS_LEVELS = sorted([1000, 950, 925, 850, 700, 600, 500, 400, 300, 250, 200], reverse=True)
CIUTATS_CATALUNYA = {
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734}, 'Girona': {'lat': 41.9831, 'lon': 2.8249},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200}, 'Tarragona': {'lat': 41.1189, 'lon': 1.2445},
}

# --- 3. CLASSE DE GESTIÓ DE DADES METEOROLÒGIQUES ---
class MeteoDataHandler:
    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    @st.cache_data(ttl=3600)
    def carregar_dades_mapa(_self, variables: List[str], hourly_index: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            lats, lons = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 15), np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 15)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            params = {
                "latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(),
                "hourly": variables, "models": "arome_seamless", "forecast_days": 4
            }
            responses = _self.openmeteo.weather_api(API_URL, params=params)
            output = {"lats": [], "lons": []}
            for var in variables: output[var] = []

            for r in responses:
                vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
                if not any(np.isnan(v) for v in vals):
                    output["lats"].append(r.Latitude())
                    output["lons"].append(r.Longitude())
                    for i, var in enumerate(variables): output[var].append(vals[i])
            
            if not output["lats"]: return None, "No s'han rebut dades vàlides del model AROME."
            return output, None
        except Exception as e:
            return None, f"Error crític en carregar dades del mapa: {e}"

    @st.cache_data(ttl=3600)
    def carregar_dades_sondeig(_self, lat: float, lon: float, hourly_index: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
            h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS]
            params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": 4}
            response = _self.openmeteo.weather_api(API_URL, params=params)[0]
            hourly = response.Hourly()

            sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i, v in enumerate(h_base)}
            if any(np.isnan(val) for val in sfc_data.values()): return None, "Dades de superfície invàlides."

            p, T, Td, u, v, h = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [0.0], [0.0], [mpcalc.pressure_to_height_std(sfc_data["surface_pressure"] * units.hPa).to('m').m]
            
            var_count = len(h_base)
            for j, p_val in enumerate(PRESS_LEVELS):
                if p_val < sfc_data["surface_pressure"]:
                    vals = [hourly.Variables(var_count + i * len(PRESS_LEVELS) + j).ValuesAsNumpy()[hourly_index] for i in range(5)]
                    if not any(np.isnan(val) for val in vals):
                        p.append(p_val); T.append(vals[0])
                        Td.append(mpcalc.dewpoint_from_relative_humidity(vals[0] * units.degC, vals[1] * units.percent).m)
                        u_comp, v_comp = mpcalc.wind_components(vals[2] * units('km/h'), vals[3] * units.degrees)
                        u.append(u_comp.to('m/s').m); v.append(v_comp.to('m/s').m); h.append(vals[4])

            if len(p) < 5: return None, "Perfil atmosfèric massa curt per a una anàlisi fiable."

            p_units, T_units, Td_units = np.array(p) * units.hPa, np.array(T) * units.degC, np.array(Td) * units.degC
            u_units, v_units, h_units = np.array(u) * units('m/s'), np.array(v) * units('m/s'), np.array(h) * units.meter
            
            prof = mpcalc.parcel_profile(p_units, T_units[0], Td_units[0])
            cape, cin = mpcalc.cape_cin(p_units, T_units, Td_units, prof)
            s_u, s_v = mpcalc.bulk_shear(p_units, u_units, v_units, height=h_units, depth=6 * units.km)
            _, srh, _ = mpcalc.storm_relative_helicity(h_units, u_units, v_units, depth=3 * units.km, storm_u=s_u, storm_v=s_v)

            resultat = {
                "perfil": (p_units, T_units, Td_units, u_units, v_units),
                "parametres": {
                    'CAPE': cape.to('J/kg').m if cape.magnitude > 0 else 0, 'CIN': cin.to('J/kg').m,
                    'Shear_0-6km': mpcalc.wind_speed(s_u, s_v).to('m/s').m,
                    'SRH_0-3km': np.sum(srh).to('m^2/s^2').m
                }
            }
            return resultat, None
        except Exception as e:
            return None, f"Error crític en processar el radiosondeig: {e}"

# --- 4. CLASSE DE VISUALITZACIÓ (GRÀFICS I MAPES) ---
class Plotter:
    def __init__(self):
        plt.style.use('dark_background')
        self.text_glow = [path_effects.withStroke(linewidth=3, foreground="black")]

    def _crear_mapa_base(self) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=250, subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#2A3B4C", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='#161B22', zorder=0)
        ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.8, zorder=5)
        ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='white', zorder=5)
        return fig, ax

    def _finalitzar_grafic(self, fig: plt.Figure, ax: plt.Axes, title: str, timestamp: str):
        ax.set_title(f"{title}\n{timestamp}", weight='bold', fontsize=16, color="#00BFFF", path_effects=self.text_glow)
        fig.tight_layout(pad=2.0)
        return fig
        
    def crear_mapa_escalar(self, map_data: Dict[str, Any], var_name: str, config: Dict[str, Any], timestamp: str) -> plt.Figure:
        fig, ax = self._crear_mapa_base()
        lons, lats, data = map_data['lons'], map_data['lats'], map_data[var_name]
        grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200))
        grid_data = griddata((lons, lats), data, (grid_lon, grid_lat), method='cubic')
        
        norm = BoundaryNorm(config['levels'], ncolors=plt.get_cmap(config['cmap']).N, clip=True)
        cf = ax.contourf(grid_lon, grid_lat, grid_data, levels=config['levels'], cmap=config['cmap'], norm=norm, alpha=0.8, zorder=2, extend=config.get('extend', 'max'))
        
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=config['cmap']), ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label(f"{config['title']} ({config['unit']})", fontsize=10)
        
        return self._finalitzar_grafic(fig, ax, config['title'], timestamp)

    def crear_mapa_convergencia(self, map_data: Dict[str, Any], nivell: int, config: Config, timestamp: str) -> plt.Figure:
        fig, ax = self._crear_mapa_base()
        lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
        speed_var, dir_var = f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"
        
        speeds_kmh = np.array(map_data[speed_var]) * units('km/h')
        dirs_deg = np.array(map_data[dir_var]) * units.degrees
        u_comp, v_comp = mpcalc.wind_components(speeds_kmh, dirs_deg)
        
        grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 100), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 100))
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
        
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
        
        levels = np.arange(-15, 16, 2)
        cf = ax.contourf(grid_lon, grid_lat, divergence, levels=levels, cmap='bwr_r', alpha=0.7, zorder=2, extend='both')
        cbar = fig.colorbar(cf, ax=ax, shrink=0.7)
        cbar.set_label('Convergència (-) / Divergència (+) [x10⁻⁵ s⁻¹]')
        
        ax.streamplot(grid_lon, grid_lat, grid_u, v_grid, color='white', linewidth=0.8, density=2.0, arrowsize=0.8, zorder=4)
        ax.plot(config.lon, config.lat, 'o', markerfacecolor='#FF00FF', markeredgecolor='black', markersize=10, transform=ccrs.Geodetic(), zorder=6)
        
        return self._finalitzar_grafic(fig, ax, f"Flux i Convergència a {nivell}hPa", timestamp)

    def crear_skewt(self, sondeig_data: Dict[str, Any], config: Config) -> plt.Figure:
        p, T, Td, u, v = sondeig_data['perfil']
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
        
        skew.ax.set_title(f"Sondeig Vertical - {config.city_name}", weight='bold', fontsize=14, color="#00BFFF", path_effects=self.text_glow)
        skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")
        skew.ax.legend()
        return fig

# --- 5. CLASSE D'ASSISTENT D'IA ---
class AIAssistant:
    def __init__(self):
        try:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.is_configured = True
        except (KeyError, Exception):
            self.is_configured = False
            self.model = None

    @st.cache_data(ttl=3600)
    def preparar_dades_per_ia(_self, data_handler: MeteoDataHandler, config: Config) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        sondeig, error_sondeig = data_handler.carregar_dades_sondeig(config.lat, config.lon, config.hourly_index)
        if error_sondeig: return None, f"Falten dades del sondeig: {error_sondeig}"

        variables_mapa = ["cape", "wind_speed_925hPa", "wind_direction_925hPa"]
        map_data, error_mapa = data_handler.carregar_dades_mapa(variables_mapa, config.hourly_index)
        if error_mapa: return None, f"Falten dades del mapa: {error_mapa}"

        # Càlcul de convergència màxima
        max_conv, lat_conv, lon_conv = 0, 0, 0
        try:
            lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
            speeds = np.array(map_data['wind_speed_925hPa']) * units('km/h')
            dirs = np.array(map_data['wind_direction_925hPa']) * units.degrees
            u_comp, v_comp = mpcalc.wind_components(speeds, dirs)
            grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 50), np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 50))
            grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
            grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), method='cubic')
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy) * 1e5
            
            max_conv = np.nanmin(divergence)
            idx_min = np.unravel_index(np.nanargmin(divergence), divergence.shape)
            lat_conv, lon_conv = grid_lat[idx_min], grid_lon[idx_min]
        except Exception:
            pass # Si falla, es queda amb 0

        dades_ia = {
            "sondeig_local": sondeig['parametres'],
            "context_regional": {
                "max_cape": max(map_data.get('cape', [0])),
                "max_conv_925hpa": max_conv,
                "focus_conv_lat": lat_conv,
                "focus_conv_lon": lon_conv,
            }
        }
        return dades_ia, None

    def generar_resum(self, dades: Dict[str, Any], timestamp: str) -> str:
        if not self.is_configured:
            return "❌ **Error de Configuració:** La clau API de Google no s'ha trobat. Afegeix-la a `.streamlit/secrets.toml`."
        
        prompt = f"""
        **ROL:** Ets un Sistema d'Anàlisi Predictiu de Tormentes per a Catalunya. La teva comunicació ha de ser clara, concisa i tècnicament precisa.

        **CONTEXT:** Estàs analitzant una sortida del model AROME per al dia i hora: {timestamp}.

        **DADES D'ENTRADA:**
        - **Anàlisi Regional (Catalunya):**
          - CAPE Màxim (Energia disponible): {int(dades['context_regional']['max_cape'])} J/kg
          - Convergència Màxima a 925hPa (Mecanisme de tret): {dades['context_regional']['max_conv_925hpa']:.2f} (x10⁻⁵ s⁻¹)
          - Focus de Convergència (Lat/Lon): {dades['context_regional']['focus_conv_lat']:.2f}, {dades['context_regional']['focus_conv_lon']:.2f}
        - **Anàlisi Local (Punt de Referència):**
          - Cisallament 0-6km (Organització): {int(dades['sondeig_local']['Shear_0-6km'])} m/s
          - SRH 0-3km (Potencial de Rotació): {int(dades['sondeig_local']['SRH_0-3km'])} m²/s²

        **INSTRUCCIONS:**
        1.  **NIVELL DE RISC:** Avalua el risc global de temps sever (Baix, Moderat, Alt, Extrem) basant-te en la combinació de tots els paràmetres.
        2.  **AMENACES PRINCIPALS:** Llista les amenaces meteorològiques més probables (ex: Pluja intensa, Calamarsa/Pedra, Ratxes fortes de vent, Tornados).
        3.  **RESUM TÀCTIC:** En una sola frase, explica la dinàmica atmosfèrica principal. Connecta el mecanisme de tret (convergència) amb l'energia (CAPE) i l'organització (cisallament).
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
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"S'ha produït un error en contactar amb l'assistent d'IA: {e}"

# --- 6. CLASSE DE LA INTERFÍCIE D'USUARI ---
class UI:
    def __init__(self):
        self.data_handler = MeteoDataHandler()
        self.plotter = Plotter()
        self.ai_assistant = AIAssistant()

    def _display_header_and_selectors(self) -> Config:
        st.markdown('<h1 style="text-align: center;">🌪️ Terminal de Temps Sever /// Catalunya</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center;">Plataforma avançada per a l\'anàlisi de convecció basada en el model AROME.</p>', unsafe_allow_html=True)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            city_name = c1.selectbox("Punt de Referència:", sorted(CIUTATS_CATALUNYA.keys()), key="city")
            day_option = c2.selectbox("Dia:", ("Avui", "Demà"), key="day")
            hour_option = c3.selectbox("Hora (Local):", [f"{h:02d}:00" for h in range(24)], key="hour")

        day_offset = 0 if day_option == "Avui" else 1
        hour = int(hour_option.split(':')[0])
        
        target_date = datetime.now(TIMEZONE).date() + timedelta(days=day_offset)
        local_dt = TIMEZONE.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hour))
        utc_dt = local_dt.astimezone(pytz.utc)
        start_of_run_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        hourly_index = max(0, int((utc_dt - start_of_run_utc).total_seconds() / 3600))
        
        timestamp_str = f"{day_option} a les {hour_option}"
        lat = CIUTATS_CATALUNYA[city_name]['lat']
        lon = CIUTATS_CATALUNYA[city_name]['lon']

        return Config(city_name, lat, lon, day_offset, hour, hourly_index, timestamp_str)

    def _display_live_images(self):
        st.subheader("Visió en Temps Real")
        view_choice = st.radio("Selecciona vista:", ("🛰️ Satèl·lit", "📡 Radar"), horizontal=True, label_visibility="collapsed")
        
        if "Satèl·lit" in view_choice:
            now_local = datetime.now(TIMEZONE)
            is_night = now_local.hour >= 21 or now_local.hour < 7
            url = "https://www.meteociel.fr/modeles/sat/ir_nw-europe2.gif" if is_night else "https://www.meteociel.fr/modeles/sat/vis_nw-europe2.gif"
            caption = "Satèl·lit Infraroig (Nit)" if is_night else "Satèl·lit Visible (Dia)"
        else:
            url = "https://www.meteociel.fr/cartes_obs/radar/lastradar_sp_ne.gif"
            caption = "Radar de Precipitació"
            
        try:
            # Afegim un paràmetre per evitar la caché del navegador
            response = requests.get(f"{url}?v={int(time.time())}", timeout=10)
            if response.status_code == 200:
                st.image(response.content, caption=f"{caption} | Font: Meteociel", use_container_width=True)
            else:
                st.warning(f"No s'ha pogut carregar la imatge. (Codi: {response.status_code})")
        except requests.exceptions.RequestException:
            st.error("Error de xarxa en carregar la imatge.")

    def _run_map_analysis_tab(self, config: Config):
        with st.spinner("Processant mapes d'anàlisi..."):
            c1, c2 = st.columns([2.5, 1.5])
            with c1:
                map_configs = {
                    "CAPE (Energia)": ("cape", {"title": "CAPE", "cmap": "plasma", "unit": "J/kg", "levels": np.arange(100, 3001, 100)}),
                    "Humitat 700hPa": ("relative_humidity_700hPa", {"title": "Humitat Relativa a 700hPa", "cmap": "Greens", "unit": "%", "levels": np.arange(60, 101, 5), "extend": "neither"}),
                }
                mapa_sel = st.selectbox("Selecciona capa:", list(map_configs.keys()) + ["Convergència a Baixos Nivells"])
                
                if mapa_sel in map_configs:
                    var, plot_config = map_configs[mapa_sel]
                    map_data, error = self.data_handler.carregar_dades_mapa([var], config.hourly_index)
                    if error: st.error(error)
                    else: st.pyplot(self.plotter.crear_mapa_escalar(map_data, var, plot_config, config.timestamp_str))
                
                elif mapa_sel == "Convergència a Baixos Nivells":
                    nivell = st.selectbox("Nivell:", [950, 925, 850], format_func=lambda x: f"{x} hPa")
                    variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
                    map_data, error = self.data_handler.carregar_dades_mapa(variables, config.hourly_index)
                    if error: st.error(error)
                    else: st.pyplot(self.plotter.crear_mapa_convergencia(map_data, nivell, config, config.timestamp_str))

            with c2:
                self._display_live_images()

    def _run_vertical_analysis_tab(self, config: Config, sondeig_data: Optional[Dict[str, Any]], error: Optional[str]):
        if error:
            st.error(f"No s'ha pogut carregar el sondeig: {error}")
            return
        if sondeig_data:
            st.subheader(f"Anàlisi Vertical Detallada per a {config.city_name}")
            params = sondeig_data['parametres']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CAPE (Energia)", f"{params['CAPE']:.0f} J/kg")
            c2.metric("CIN (Inhibició)", f"{params['CIN']:.0f} J/kg")
            c3.metric("Shear 0-6km (Organització)", f"{params['Shear_0-6km']:.1f} m/s")
            c4.metric("SRH 0-3km (Rotació)", f"{params['SRH_0-3km']:.0f} m²/s²")

            with st.expander("ℹ️ Interpretació dels Paràmetres"):
                st.markdown("""
                - **CAPE:** Energia Convectiva. >1000 J/kg indica potencial per tempestes fortes.
                - **CIN:** Energia d'Inhibició. Valors negatius alts (> -50 J/kg) poden actuar com una "tapa" que impedeix la convecció inicial.
                - **Shear 0-6km:** Cisallament del vent. > 18 m/s afavoreix l'organització de les tempestes en sistemes més severs com supercèl·lules.
                - **SRH 0-3km:** Helicitat Relativa a la Tempesta. > 150 m²/s² suggereix un alt potencial de rotació en les tempestes (mesociclons).
                """)
            
            st.pyplot(self.plotter.crear_skewt(sondeig_data, config))
        else:
            st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

    def _run_ai_summary_tab(self, config: Config):
        st.subheader("Assistent d'Anàlisi Predictiva IA")
        if not self.ai_assistant.is_configured:
            st.error("Funcionalitat no disponible. La clau API de Google no està configurada a `.streamlit/secrets.toml`.")
            return

        if st.button("🤖 Generar Anàlisi d'IA", use_container_width=True):
            with st.spinner("L'assistent IA està processant milers de punts de dades..."):
                dades_ia, error = self.ai_assistant.preparar_dades_per_ia(self.data_handler, config)
                if error:
                    st.error(f"No s'ha pogut generar l'anàlisi: {error}")
                else:
                    resum = self.ai_assistant.generar_resum(dades_ia, config.timestamp_str)
                    st.markdown(resum)

    def run(self):
        inject_custom_css()
        config = self._display_header_and_selectors()
        
        # Carreguem el sondeig un cop, ja que es pot necessitar a la pestanya vertical i a la d'IA
        sondeig_data, sondeig_error = self.data_handler.carregar_dades_sondeig(config.lat, config.lon, config.hourly_index)
        
        tab_mapes, tab_vertical, tab_ia = st.tabs(["🗺️ Anàlisi de Mapes", "📊 Anàlisi Vertical", "🤖 Resum IA"])
        with tab_mapes: self._run_map_analysis_tab(config)
        with tab_vertical: self._run_vertical_analysis_tab(config, sondeig_data, sondeig_error)
        with tab_ia: self._run_ai_summary_tab(config)

        st.divider()
        st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades del model AROME via <a href='https://open-meteo.com/'>Open-Meteo</a> | Imatges via <a href='https://www.meteociel.fr/'>Meteociel</a> | Anàlisi IA per Google Gemini.</p>", unsafe_allow_html=True)


# --- 7. PUNT D'ENTRADA DE L'APLICACIÓ ---
if __name__ == "__main__":
    app = UI()
    app.run()
