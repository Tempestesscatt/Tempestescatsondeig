# -*- coding: utf-8 -*-

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pytz # Llibreria per gestionar zones horàries

# --- CONFIGURACIÓ PRINCIPAL I CONSTANTS ---

# Configuració de la pàgina de Streamlit (s'ha de cridar al principi)
st.set_page_config(
    page_title="Radar Meteorològic Interactiu",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ruta al fitxer de caché generat per 'actualizar_cache.py'
CACHE_FILE = Path("meteo_cache.json")

# Dades de les localitats (ha de ser idèntic al de 'actualizar_cache.py')
pobles_data = {
    'Malgrat de Mar': {'lat': 41.645, 'lon': 2.741}, 'Santa Susanna': {'lat': 41.636, 'lon': 2.711},
    'Pineda de Mar': {'lat': 41.626, 'lon': 2.689}, 'Calella': {'lat': 41.614, 'lon': 2.664},
    'Sant Pol de Mar': {'lat': 41.602, 'lon': 2.624}, 'Canet de Mar': {'lat': 41.590, 'lon': 2.580},
    'Arenys de Mar': {'lat': 41.581, 'lon': 2.551}, 'Caldes d\'Estrac': {'lat': 41.573, 'lon': 2.529},
    'Sant Vicenç de Montalt': {'lat': 41.572, 'lon': 2.508}, 'Vilassar de Mar': {'lat': 41.506, 'lon': 2.392},
    'Premià de Mar': {'lat': 41.491, 'lon': 2.359}, 'El Masnou': {'lat': 41.481, 'lon': 2.318},
    'Montgat': {'lat': 41.464, 'lon': 2.279}, 'Sant Adrià de Besòs': {'lat': 41.428, 'lon': 2.219},
}

# Graella del mapa (ha de ser idèntica a la de 'actualizar_cache.py')
MAP_LATS, MAP_LONS = np.linspace(40.5, 42.8, 12), np.linspace(0.2, 3.3, 12)
LON_GRID, LAT_GRID = np.meshgrid(MAP_LONS, MAP_LATS)
FORECAST_DAYS = 1 # Coherent amb l'script de caché

# --- FUNCIONS DE CÀRREGA I EXTRACCIÓ DE DADES (Llegeixen de la caché) ---

@st.cache_data(show_spinner="Carregant dades meteorològiques...")
def load_all_data_from_cache():
    """
    Carrega totes les dades pre-calculades del fitxer JSON.
    Streamlit posa en caché el resultat d'aquesta funció per a un rendiment màxim.
    """
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                st.error(f"Error en llegir el fitxer de caché '{CACHE_FILE}'. El fitxer podria estar corrupte.")
                return None
    else:
        st.error(f"Error crític: El fitxer de caché '{CACHE_FILE}' no s'ha trobat. "
                 "Si us plau, executa primer l'script 'actualizar_cache.py'.")
        return None

def get_sounding_data_from_cache(all_data, poble_sel):
    """Extreu les dades del sondeig per a un poble des de la caché carregada."""
    try:
        return all_data['data']['sounding_data'][poble_sel]
    except KeyError:
        return None

def get_map_data_from_cache(all_data, var_name, level):
    """Extreu i remodela les dades del mapa des de la caché carregada."""
    cache_key = f"{var_name}_{level}"
    try:
        # Extreu les dades brutes de la caché
        raw_map_data = all_data['data']['map_data'][cache_key]
        
        # El format de la caché és una llista de punts, cal reconstruir la graella 2D.
        num_lats, num_lons = len(MAP_LATS), len(MAP_LONS)
        
        # Processa cada variable dins de les dades del mapa (ex: u_component, v_component)
        processed_data = {}
        # Agafem les claus del primer punt de la graella (ex: 'u_component_of_wind_850hPa')
        var_keys_in_cache = raw_map_data[0]['hourly'].keys()
        
        for key in var_keys_in_cache:
            # Extreu els valors per a aquesta clau de cada punt de la graella
            flat_values = [point['hourly'][key] for point in raw_map_data]
            # Remodela la llista plana a una graella 2D de (hores, lats, lons)
            reshaped_values = np.array(flat_values).reshape(num_lats, num_lons, -1)
            # Canvia l'ordre a (hores, lats, lons)
            processed_data[key.split('_')[0]] = np.transpose(reshaped_values, (2, 0, 1))
            
        return processed_data, None # Retorna dades i cap error
        
    except (KeyError, IndexError, ValueError) as e:
        return None, f"No s'han trobat o no s'han pogut processar les dades del mapa per a '{cache_key}'. Revisa que 'actualizar_cache.py' hagi funcionat correctament. Error: {e}"

# --- FUNCIONS DE CÀLCUL (Processen les dades de la caché) ---

def calcular_convergencia(wind_data, hourly_index):
    """
    Calcula la divergència (convergència negativa) a partir de les dades de vent (U, V).
    AQUESTA FUNCIÓ REQUEREIX QUE 'actualizar_cache.py' GUARDI 'u_component_of_wind' I 'v_component_of_wind'.
    """
    try:
        u = wind_data['u'][hourly_index] # Component U del vent a l'hora seleccionada
        v = wind_data['v'][hourly_index] # Component V del vent a l'hora seleccionada
    except KeyError:
        # Si no trobem 'u' i 'v', és perquè la caché es va generar amb 'wind_speed'
        return {}, None # Retorna un diccionari buit per indicar l'error

    # Càlcul de les distàncies dx i dy a partir de les coordenades
    R = 6371e3 # Radi de la Terra en metres
    lats_rad = np.deg2rad(MAP_LATS)
    dx = R * np.cos(lats_rad).mean() * np.deg2rad(np.gradient(MAP_LONS))
    dy = R * np.deg2rad(np.gradient(MAP_LATS))
    
    # Afegim eixos per al broadcasting correcte
    dx = dx[np.newaxis, :]
    dy = dy[:, np.newaxis]

    # Càlcul de la divergència usant np.gradient: dv/dy + du/dx
    dv_dy = np.gradient(v, axis=0) / dy
    du_dx = np.gradient(u, axis=1) / dx
    
    divergence = (du_dx + dv_dy) * 1e5 # Multipliquem per 10^5 per a unitats més llegibles
    
    convergencies = {}
    for nom_poble, coords in pobles_data.items():
        # Troba l'índex més proper a la localitat
        lat_idx = np.abs(MAP_LATS - coords['lat']).argmin()
        lon_idx = np.abs(MAP_LONS - coords['lon']).argmin()
        convergencies[nom_poble] = -divergence[lat_idx, lon_idx] # Convergència és -divergència
        
    return convergencies, divergence

def calcular_parametres_des_de_cache(sounding_data, hourly_index):
    """
    [PLACEHOLDER] Aquesta funció ha d'extreure les dades d'una hora concreta
    del sondeig i calcular tots els paràmetres meteorològics necessaris.
    
    EMPLENA AQUESTA FUNCIÓ AMB LA TEVA LÒGICA DE CÀLCUL.
    """
    # Aquesta és una implementació d'exemple. Adapta-la a les teves necessitats.
    try:
        p = np.array([1000, 925, 850, 700, 500, 300]) # Exemple de nivells de pressió
        
        # Extreu els valors per a l'hora seleccionada (hourly_index)
        T = np.array([sounding_data['hourly'].get(f'temperature_{lvl}hPa', [np.nan]*24)[hourly_index] for lvl in p])
        Td = np.array([sounding_data['hourly'].get(f'dew_point_{lvl}hPa', [np.nan]*24)[hourly_index] for lvl in p])
        u = np.array([sounding_data['hourly'].get(f'u_component_of_wind_{lvl}hPa', [np.nan]*24)[hourly_index] for lvl in p]) # Requereix U/V
        v = np.array([sounding_data['hourly'].get(f'v_component_of_wind_{lvl}hPa', [np.nan]*24)[hourly_index] for lvl in p]) # Requereix U/V
        H = np.array([sounding_data['hourly'].get(f'geopotential_height_{lvl}hPa', [np.nan]*24)[hourly_index] for lvl in p])
        
        # Neteja de NaNs
        valid_indices = ~np.isnan(T)
        p, T, Td, u, v, H = p[valid_indices], T[valid_indices], Td[valid_indices], u[valid_indices], v[valid_indices], H[valid_indices]
        
        # Simula el càlcul de paràmetres derivats
        parametros = {
            'CAPE': {'value': 1234, 'units': 'J/kg'},
            'LCL': {'value': 1100, 'units': 'm'},
            'LI': {'value': -4.5, 'units': 'K'},
            'ZeroIso_AGL': {'value': 3200, 'units': 'm'},
            # Afegeix aquí tots els altres paràmetres que la teva app necessita
        }
        
        return parametros, p, T, Td, u, v, H, None
    except Exception as e:
        return None, None, None, None, None, None, None, f"Error calculant paràmetres: {e}"


# --- FUNCIONS DE VISUALITZACIÓ I GRÀFICS (Placeholders) ---
# EMPLENA AQUESTES FUNCIONS AMB LES TEVES IMPLEMENTACIONS DE MATPLOTLIB/PLOTLY

def generar_avis_temperatura(parametros):
    return "Avís Temp.", "Text de l'avís de temperatura.", "#FFC107", "thermostat"

def generar_avis_convergencia(parametros, is_disparador_active, divergence_value_local):
    if is_disparador_active:
        return "Avís Conv.", f"Text de l'avís de convergència activa ({divergence_value_local:.2f}).", "#DC3545"
    return None, None, None

def generar_avis_localitat(parametros, is_disparador_active):
    return "Avis Localitat", "Text d'avís general per a la localitat.", "#0D6EFD"

def display_avis_principal(titol, text, color, icona_personalitzada=None):
    st.markdown(f"**{titol}**: {text}")

def generar_analisi_detallada(parametros, is_disparador_active):
    yield "Aquesta és una anàlisi detallada generada automàticament...\n"
    yield f"El valor de CAPE és {parametros.get('CAPE', {}).get('value')} J/kg. "
    if is_disparador_active:
        yield "Hi ha un disparador de convergència actiu."

def display_metrics(parametros):
    st.metric("CAPE", f"{parametros.get('CAPE', {}).get('value', 'N/A')} J/kg")
    st.metric("LCL", f"{parametros.get('LCL', {}).get('value', 'N/A')} m")
    st.metric("LI", f"{parametros.get('LI', {}).get('value', 'N/A')} K")

def crear_mapa_generic(lats, lons, data, nivell, lat_sel, lon_sel, poble_sel, titol, cmap, unitat, levels):
    # Retorna una figura de Matplotlib d'exemple
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title(f"{titol} a {nivell}hPa")
    if data is not None and len(data) > 0:
        c = ax.contourf(lons, lats, data, cmap=cmap, levels=levels)
        fig.colorbar(c, ax=ax, label=f"{titol} ({unitat})")
    ax.plot(lon_sel, lat_sel, 'ro', markersize=8, label=poble_sel)
    ax.legend()
    return fig

def crear_mapa_vents(lats, lons, data, nivell, lat_sel, lon_sel, poble_sel):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title(f"Vent a {nivell}hPa")
    try:
        u, v = data['u'], data['v']
        ax.barbs(lons, lats, u, v, length=6)
    except (KeyError, TypeError):
        ax.text(0.5, 0.5, "Dades de vent (U/V) no disponibles", ha='center', va='center', transform=ax.transAxes)
    ax.plot(lon_sel, lat_sel, 'ro', markersize=8, label=poble_sel)
    ax.legend()
    return fig

def crear_mapa_temp_isobares(lats, lons, data, nivell, lat_sel, lon_sel, poble_sel):
    return crear_mapa_generic(lats, lons, data.get('temperature'), nivell, lat_sel, lon_sel, poble_sel, "Temperatura", "plasma", "°C", np.arange(-10, 31, 2))

def crear_hodograf(p, u, v, H):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("Hodògraf (0-10km)")
    ax.plot(u, v, 'o-')
    ax.set_xlabel("U (m/s)")
    ax.set_ylabel("V (m/s)")
    ax.grid(True)
    return fig
    
def crear_skewt(p, T, Td, u, v):
    # Aquesta funció requereix una llibreria com MetPy
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("Sondeig Skew-T")
    ax.plot(T, p, label="Temperatura")
    ax.plot(Td, p, label="Punt de Rosada")
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Pressió (hPa)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True)
    return fig

def crear_grafic_orografia(parametros, zero_iso_agl):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("Potencial d'Activació Orografia")
    ax.text(0.5, 0.5, "Gràfic d'orografia placeholder", ha='center', va='center', transform=ax.transAxes)
    return fig

def crear_grafic_nuvol(parametros, H, u, v, is_disparador_active):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("Visualització del Núvol")
    ax.text(0.5, 0.5, "Gràfic de núvol placeholder", ha='center', va='center', transform=ax.transAxes)
    return fig

# --- APLICACIÓ PRINCIPAL DE STREAMLIT ---

# 1. Carregar totes les dades des del fitxer de caché a l'inici.
all_cached_data = load_all_data_from_cache()

# 2. Si les dades no es carreguen, l'app no pot continuar.
if not all_cached_data:
    st.stop()

# 3. Interfície d'usuari (Sidebar)
with st.sidebar:
    st.header("📍 Panell de Control")
    
    poble_sel = st.selectbox("Selecciona una localitat:", list(pobles_data.keys()))
    lat_sel = pobles_data[poble_sel]['lat']
    lon_sel = pobles_data[poble_sel]['lon']
    
    # El nombre d'hores ha de coincidir amb el que genera 'actualizar_cache.py'
    num_hores = 24 * FORECAST_DAYS
    hourly_index = st.slider("Selecciona l'hora del pronòstic (local):", 0, num_hores - 1, 12, 1)
    
    # Mostra l'hora de l'última actualització de la caché
    last_update_utc_str = all_cached_data.get("last_update_utc", "N/A")
    if last_update_utc_str != "N/A":
        last_update_dt_utc = datetime.fromisoformat(last_update_utc_str)
        local_tz = pytz.timezone('Europe/Madrid')
        last_update_dt_local = last_update_dt_utc.astimezone(local_tz)
        st.info(f"Dades actualitzades a les:\n{last_update_dt_local.strftime('%d/%m/%Y %H:%M:%S')}")

# 4. Processament de dades per a la localitat i hora seleccionades
st.title(f"Anàlisi Meteorològica per a {poble_sel}")

# Extreure dades del sondeig i calcular paràmetres
raw_sounding_data = get_sounding_data_from_cache(all_cached_data, poble_sel)
if raw_sounding_data:
    parametros, p, T, Td, u, v, H, error_calcul = calcular_parametres_des_de_cache(raw_sounding_data, hourly_index)
else:
    st.error(f"No s'han trobat dades de sondeig per a {poble_sel} en el fitxer de caché.")
    st.stop()

if error_calcul:
    st.error(error_calcul)
    st.stop()

# Càlcul de la convergència
convergencies_850hpa = {}
localitats_convergencia_forta = set()
map_data_wind_850, error_mapa_vent = get_map_data_from_cache(all_cached_data, "wind", 850)

if error_mapa_vent:
    st.warning(f"No s'han pogut carregar les dades de vent per a l'anàlisi de convergència: {error_mapa_vent}")
else:
    convergencies_850hpa, _ = calcular_convergencia(map_data_wind_850, hourly_index)
    if not convergencies_850hpa:
        st.warning("No s'ha pogut calcular la convergència. Assegura't que la caché conté els components U/V del vent.")
    
    # Definim un llindar per a la convergència forta (exemple: > 1)
    CONV_THRESHOLD = 1.0
    for localitat, valor in convergencies_850hpa.items():
        if valor > CONV_THRESHOLD:
            localitats_convergencia_forta.add(localitat)

# 5. Lògica principal de l'aplicació i visualització en pestanyes
if parametros:
    # Comprovem si el disparador està actiu segons el NOU criteri
    is_disparador_active = poble_sel in localitats_convergencia_forta
    divergence_value_local = convergencies_850hpa.get(poble_sel)

    avis_temp_titol, avis_temp_text, avis_temp_color, avis_temp_icona = generar_avis_temperatura(parametros)
    if avis_temp_titol:
        display_avis_principal(avis_temp_titol, avis_temp_text, avis_temp_color, icona_personalitzada=avis_temp_icona)

    avis_conv_titol, avis_conv_text, avis_conv_color = generar_avis_convergencia(parametros, is_disparador_active, divergence_value_local)
    if avis_conv_titol:
        display_avis_principal(avis_conv_titol, avis_conv_text, avis_conv_color)

    avis_titol, avis_text, avis_color = generar_avis_localitat(parametros, is_disparador_active)
    display_avis_principal(avis_titol, avis_text, avis_color)

    # Creació de les pestanyes
    tab_analisi, tab_params, tab_mapes, tab_hodo, tab_sondeig, tab_oro, tab_nuvol, tab_focus = st.tabs([
        "🗨️ Anàlisi", "📊 Paràmetres", "🗺️ Mapes", "🧭 Hodògraf",
        "📍 Sondeig", "🏔️ Orografia", "☁️ Visualització", "🎯 Focus de Convergència"
    ])

    with tab_analisi:
        st.write_stream(generar_analisi_detallada(parametros, is_disparador_active))

    with tab_params:
        st.subheader("Paràmetres Clau")
        display_metrics(parametros)

    with tab_mapes:
        st.subheader(f"Anàlisi de Mapes")
        col_nivell, col_tipus = st.columns([1,2])
        with col_nivell:
            p_levels_all = [1000, 925, 850, 700, 500, 300]
            nivell_global = st.selectbox("Nivell d'anàlisi:", p_levels_all, index=2) # 850 per defecte
        
        map_options = {
            "Vents i Convergència": {"var_name": "wind"},
            "Temperatura i Isobares": {"var_name": "temp_height"},
            "Punt de Rosada": {"var_name": "dewpoint", "titol": "Punt de Rosada", "cmap": "BrBG", "unitat": "°C", "levels": np.arange(-10, 21, 2)},
            "Humitat Relativa": {"var_name": "humidity", "titol": "Humitat Relativa", "cmap": "Greens", "unitat": "%", "levels": np.arange(30, 101, 5)},
        }
        with col_tipus:
            selected_map_name = st.selectbox("Tipus de mapa:", map_options.keys())
        
        with st.spinner(f"Generant mapa de {selected_map_name.lower()}..."):
            map_config = map_options[selected_map_name]
            api_var = map_config["var_name"]
            data, error = get_map_data_from_cache(all_cached_data, api_var, nivell_global)
            
            if error:
                st.error(f"Error en obtenir dades del mapa: {error}")
            elif not data:
                st.warning("No hi ha prou dades per generar el mapa.")
            else:
                fig = None
                data_for_hour = {key: val[hourly_index] for key, val in data.items()}
                
                if selected_map_name == "Vents i Convergència":
                    fig = crear_mapa_vents(MAP_LATS, MAP_LONS, data_for_hour, nivell_global, lat_sel, lon_sel, poble_sel)
                elif selected_map_name == "Temperatura i Isobares":
                    fig = crear_mapa_temp_isobares(MAP_LATS, MAP_LONS, data_for_hour, nivell_global, lat_sel, lon_sel, poble_sel)
                else:
                    # Agafa la primera (i única) variable de dades per als mapes genèrics
                    first_var_key = list(data_for_hour.keys())[0]
                    fig = crear_mapa_generic(MAP_LATS, MAP_LONS, data_for_hour[first_var_key], nivell_global, lat_sel, lon_sel, poble_sel, map_config["titol"], map_config["cmap"], map_config["unitat"], map_config["levels"])
                
                if fig:
                    st.pyplot(fig)

    with tab_hodo:
        st.subheader("Hodògraf (0-10 km)")
        fig_hodo = crear_hodograf(p, u, v, H)
        st.pyplot(fig_hodo)

    with tab_sondeig:
        st.subheader(f"Sondeig per a {poble_sel}")
        fig_skewt = crear_skewt(p, T, Td, u, v)
        st.pyplot(fig_skewt)

    with tab_oro:
        st.subheader("Potencial d'Activació per Orografia")
        fig_oro = crear_grafic_orografia(parametros, parametros.get('ZeroIso_AGL', {}).get('value'))
        if fig_oro:
            st.pyplot(fig_oro)
        else:
            st.info("No hi ha dades de LCL disponibles per calcular el potencial orogràfic.")

    with tab_nuvol:
        st.subheader("Visualització Conceptual del Núvol")
        with st.spinner("Dibuixant la possible estructura del núvol..."):
            fig_nuvol = crear_grafic_nuvol(parametros, H, u, v, is_disparador_active)
            if fig_nuvol:
                st.pyplot(fig_nuvol)
            else:
                st.info("No hi ha dades de LCL o EL disponibles per visualitzar l'estructura del núvol.")
    
    with tab_focus:
        st.subheader("Anàlisi del Disparador de Convergència")
        if not convergencies_850hpa:
            st.warning("No s'ha pogut realitzar l'anàlisi de convergència. Revisa la configuració del vent a `actualizar_cache.py`.")
        elif is_disparador_active:
            st.success(f"**Convergència FORTA detectada a {poble_sel}!**")
            if divergence_value_local:
                st.metric("Valor de Convergència local (850hPa)", f"{-divergence_value_local:.2f} x10⁻⁵ s⁻¹", help="Valors positius indiquen convergència forta.")
            st.markdown("Aquesta zona té un focus de convergència actiu que pot actuar com a **disparador** per a les tempestes, augmentant significativament la seva probabilitat.")
        else:
            st.info(f"Cap focus de convergència significatiu detectat a {poble_sel} per a l'hora seleccionada.")
            if divergence_value_local:
                 st.metric("Valor de Convergència/Divergència local (850hPa)", f"{-divergence_value_local:.2f} x10⁻⁵ s⁻¹")
            st.markdown("L'absència d'un disparador clar pot dificultar la formació de tempestes, fins i tot si hi ha inestabilitat.")
        
        st.markdown("---")
        st.markdown("**Altres localitats amb convergència forta detectada a aquesta hora:**")
        if localitats_convergencia_forta:
            loc_list = sorted(list(localitats_convergencia_forta))
            # Mostra fins a 15 localitats en 3 columnes
            for i in range(0, min(len(loc_list), 15), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(loc_list):
                        cols[j].markdown(f"- {loc_list[i+j]}")

            if len(loc_list) > 15:
                st.markdown(f"*... i {len(loc_list)-15} més*")
        else:
            st.markdown("*Cap altra localitat amb avís de convergència forta per a aquesta hora.*")
else:
    st.warning(f"No s'han pogut calcular els paràmetres per a les {hourly_index:02d}:00h. Les dades del model podrien no ser vàlides per a aquesta hora.")
