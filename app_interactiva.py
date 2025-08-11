import streamlit as st
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.transforms as mtransforms
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc
from matplotlib.patches import Circle, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import cartopy.io.img_tiles as cimgt
from datetime import datetime, timedelta
import pytz

# --- CONFIGURACIÓ INICIAL ---
st.set_page_config(layout="wide", page_title="Tempestes.cat")
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- DATOS DE LES LOCALITATS (LLISTA DEFINITIVA I COMPLETA) ---
pobles_data = {
    'Amposta': {'lat': 40.707, 'lon': 0.579},
    'Arbúcies': {'lat': 41.815, 'lon': 2.515},
    'Arenys de Mar': {'lat': 41.581, 'lon': 2.551},
    'Badalona': {'lat': 41.450, 'lon': 2.247},
    'Balaguer': {'lat': 41.790, 'lon': 0.810},
    'Banyoles': {'lat': 42.119, 'lon': 2.766},
    'Barcelona': {'lat': 41.387, 'lon': 2.168},
    'Berga': {'lat': 42.103, 'lon': 1.845},
    'Blanes': {'lat': 41.674, 'lon': 2.793},
    'Calafell': {'lat': 41.199, 'lon': 1.567},
    'Caldes de Montbui': {'lat': 41.633, 'lon': 2.166},
    'Calella': {'lat': 41.614, 'lon': 2.664},
    'Cambrils': {'lat': 41.066, 'lon': 1.056},
    'Canet de Mar': {'lat': 41.590, 'lon': 2.580},
    'Cardona': {'lat': 41.914, 'lon': 1.679},
    'Castell-Platja d\'Aro': {'lat': 41.818, 'lon': 3.067},
    'Castelldefels': {'lat': 41.279, 'lon': 1.975},
    'Cerdanyola del Vallès': {'lat': 41.491, 'lon': 2.141},
    'Cervera': {'lat': 41.666, 'lon': 1.272},
    'Cornellà de Llobregat': {'lat': 41.355, 'lon': 2.069},
    'Deltebre': {'lat': 40.719, 'lon': 0.710},
    'El Masnou': {'lat': 41.481, 'lon': 2.318},
    'El Pont de Suert': {'lat': 42.408, 'lon': 0.741},
    'El Prat de Llobregat': {'lat': 41.326, 'lon': 2.095},
    'El Vendrell': {'lat': 41.219, 'lon': 1.534},
    'Esplugues de Llobregat': {'lat': 41.375, 'lon': 2.086},
    'Falset': {'lat': 41.144, 'lon': 0.819},
    'Figueres': {'lat': 42.266, 'lon': 2.962},
    'Gandesa': {'lat': 41.052, 'lon': 0.436},
    'Gavà': {'lat': 41.305, 'lon': 2.001},
    'Girona': {'lat': 41.983, 'lon': 2.824},
    'Granollers': {'lat': 41.608, 'lon': 2.289},
    'Igualada': {'lat': 41.580, 'lon': 1.616},
    'L\'Ametlla de Mar': {'lat': 40.883, 'lon': 0.802},
    'L\'Escala': {'lat': 42.122, 'lon': 3.131},
    'L\'Hospitalet de Llobregat': {'lat': 41.357, 'lon': 2.102},
    'La Bisbal d\'Empordà': {'lat': 41.959, 'lon': 3.037},
    'La Jonquera': {'lat': 42.419, 'lon': 2.875},
    'La Seu d\'Urgell': {'lat': 42.358, 'lon': 1.463},
    'Les Borges Blanques': {'lat': 41.522, 'lon': 0.869},
    'Lleida': {'lat': 41.617, 'lon': 0.622},
    'Lloret de Mar': {'lat': 41.700, 'lon': 2.845},
    'Manlleu': {'lat': 42.000, 'lon': 2.283},
    'Manresa': {'lat': 41.727, 'lon': 1.825},
    'Martorell': {'lat': 41.474, 'lon': 1.927},
    'Mataró': {'lat': 41.538, 'lon': 2.445},
    'Moià': {'lat': 41.810, 'lon': 2.096},
    'Molins de Rei': {'lat': 41.414, 'lon': 2.016},
    'Mollerussa': {'lat': 41.631, 'lon': 0.895},
    'Mollet del Vallès': {'lat': 41.539, 'lon': 2.213},
    'Mont-roig del Camp': {'lat': 41.087, 'lon': 0.957},
    'Montblanc': {'lat': 41.375, 'lon': 1.161},
    'Móra d\'Ebre': {'lat': 41.092, 'lon': 0.643},
    'Olesa de Montserrat': {'lat': 41.545, 'lon': 1.894},
    'Olot': {'lat': 42.181, 'lon': 2.490},
    'Palamós': {'lat': 41.846, 'lon': 3.128},
    'Piera': {'lat': 41.520, 'lon': 1.748},
    'Premià de Mar': {'lat': 41.491, 'lon': 2.359},
    'Puigcerdà': {'lat': 42.432, 'lon': 1.928},
    'Reus': {'lat': 41.155, 'lon': 1.107},
    'Ripoll': {'lat': 42.201, 'lon': 2.190},
    'Roses': {'lat': 42.262, 'lon': 3.175},
    'Rubí': {'lat': 41.493, 'lon': 2.032},
    'Sabadell': {'lat': 41.547, 'lon': 2.108},
    'Salou': {'lat': 41.076, 'lon': 1.140},
    'Sant Adrià de Besòs': {'lat': 41.428, 'lon': 2.219},
    'Sant Boi de Llobregat': {'lat': 41.346, 'lon': 2.041},
    'Sant Carles de la Ràpita': {'lat': 40.618, 'lon': 0.593},
    'Sant Celoni': {'lat': 41.691, 'lon': 2.491},
    'Sant Cugat del Vallès': {'lat': 41.472, 'lon': 2.085},
    'Sant Feliu de Guíxols': {'lat': 41.780, 'lon': 3.028},
    'Sant Feliu de Llobregat': {'lat': 41.381, 'lon': 2.045},
    'Sant Joan Despí': {'lat': 41.368, 'lon': 2.057},
    'Santa Coloma de Farners': {'lat': 41.859, 'lon': 2.668},
    'Santa Coloma de Gramenet': {'lat': 41.454, 'lon': 2.213},
    'Santa Perpètua de Mogoda': {'lat': 41.536, 'lon': 2.182},
    'Sitges': {'lat': 41.235, 'lon': 1.811},
    'Solsona': {'lat': 41.992, 'lon': 1.516},
    'Sort': {'lat': 42.413, 'lon': 1.129},
    'Tarragona': {'lat': 41.118, 'lon': 1.245},
    'Tàrrega': {'lat': 41.646, 'lon': 1.141},
    'Terrassa': {'lat': 41.561, 'lon': 2.008},
    'Tortosa': {'lat': 40.812, 'lon': 0.521},
    'Tremp': {'lat': 42.166, 'lon': 0.894},
    'Valls': {'lat': 41.286, 'lon': 1.250},
    'Vic': {'lat': 41.930, 'lon': 2.255},
    'Vielha': {'lat': 42.702, 'lon': 0.796},
    'Vila-seca': {'lat': 41.111, 'lon': 1.144},
    'Viladecans': {'lat': 41.315, 'lon': 2.019},
    'Vilafranca del Penedès': {'lat': 41.345, 'lon': 1.698},
    'Vilanova i la Geltrú': {'lat': 41.224, 'lon': 1.725},
    'Vilassar de Mar': {'lat': 41.506, 'lon': 2.392},
}

# --- FUNCIONS ---
def get_next_arome_update_time():
    now_utc = datetime.now(pytz.utc)
    run_hours_utc = [0, 6, 12, 18]
    availability_delay = timedelta(hours=4)
    next_update_time = None
    for run_hour in run_hours_utc:
        run_datetime = now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        available_time = run_datetime + availability_delay
        if available_time > now_utc:
            next_update_time = available_time
            break
    if next_update_time is None:
        tomorrow = now_utc + timedelta(days=1)
        next_update_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0) + availability_delay
    local_tz = pytz.timezone('Europe/Madrid')
    next_update_local = next_update_time.astimezone(local_tz)
    return f"Pròxima actualització de dades (model AROME) estimada a les **{next_update_local.strftime('%H:%Mh')}**"

def get_parameter_style(param_name, value):
    color = "white"; emoji = ""
    if value is None or not isinstance(value, (int, float)): return color, emoji
    if param_name == 'CIN_Fre':
        if value >= -25: color, emoji = "#32CD32", "✅"
        elif value < -100: color, emoji = "#FF4500", "⚠️"
        elif value < -25: color, emoji = "#FFA500", ""
    elif 'CAPE' in param_name:
        if value > 3500: color, emoji = "#FF00FF", "⚠️"
        elif value > 2500: color, emoji = "#FF4500", "⚠️"
        elif value > 1500: color, emoji = "#FFA500", ""
        elif value > 500: color = "#32CD32"
    elif 'Shear' in param_name:
        if value > 25: color, emoji = "#FF4500", "⚠️"
        elif value > 18: color, emoji = "#FFA500", ""
        elif value > 10: color = "#32CD32"
    elif 'SRH' in param_name:
        if value > 400: color, emoji = "#FF4500", "⚠️"
        elif value > 250: color, emoji = "#FFA500", ""
        elif value > 100: color = "#32CD32"
    elif 'LCL' in param_name:
        if value < 1000: color = "#FFA500"
        elif value < 1500: color = "#32CD32"
    return color, emoji

def generar_avis_localitat(params):
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0)
    cin = params.get('CIN_Fre', {}).get('value')
    shear = params.get('Shear_0-6km', {}).get('value')
    srh1 = params.get('SRH_0-1km', {}).get('value')
    lcl_agl = params.get('LCL_AGL', {}).get('value', 9999)
    lfc_agl = params.get('LFC_AGL', {}).get('value', 9999)

    if cape_u < 100:
        return ("Sense risc de tempestes significatives. Atmosfera estable.", "#3CB371")
    if cin is not None and cin < -100:
        return ("Sense risc de tempestes. La 'tapa' atmosfèrica (CIN) és massa forta per permetre el seu desenvolupament.", "#3CB371")
    if lfc_agl > 3000:
        return ("Risc molt baix de tempestes. El nivell d'inici de la convecció (LFC) és massa alt i difícil d'assolir.", "#4682B4")

    if shear is not None and shear > 20 and cape_u > 1500 and srh1 is not None and srh1 > 250 and lcl_agl < 1200:
        return ("RISC ALT: Condicions favorables per a SUPERCL·LULES amb potencial de TORNADOS.", "#DC143C")
    if shear is not None and shear > 18 and cape_u > 1000:
        return ("AVÍS: Potencial per a SUPERCL·LULES. Risc de calamarsa grossa i fortes ratxes de vent.", "#FF8C00")
    if shear is not None and shear > 12 and cape_u > 500:
        return ("PRECAUCIÓ: Risc de TEMPESTES ORGANITZADES (multicèl·lules). Possibles fortes pluges i calamarsa.", "#FFD700")

    return ("Risc Baix: Possibles xàfecs o tempestes febles i aïllades (unicel·lulars).", "#4682B4")

def generar_analisi_detallada(params):
    conversa = []
    cape, cin, cape_u, pwat = (params.get(k, {}).get('value') for k in ['CAPE_Brut', 'CIN_Fre', 'CAPE_Utilitzable', 'PWAT_Total'])
    shear6, srh1, srh3 = (params.get(k, {}).get('value') for k in ['Shear_0-6km', 'SRH_0-1km', 'SRH_0-3km'])
    lcl_agl, lfc_agl, el_msl = (params.get(k, {}).get('value') for k in ['LCL_AGL', 'LFC_AGL', 'EL_MSL'])

    conversa.append("--- ANÀLISI TERMODINÀMICA ---")
    if cape is None or cape < 100:
        conversa.append("L'atmosfera és estable o quasi estable. El CAPE és pràcticament inexistent, per la qual cosa no s'espera la formació de tempestes significatives.")
        return conversa
    
    cape_text = "feble" if cape < 1000 else "moderada" if cape < 2500 else "forta" if cape < 3500 else "extrema"
    conversa.append(f"Tenim un CAPE de {cape:.0f} J/kg, un potencial energètic que indica inestabilitat {cape_text}.")

    if cin is not None:
        if cin < -100:
            conversa.append(f"⚠️ Factor limitant clau: La 'tapa' d'inversió (CIN) és molt forta ({cin:.0f} J/kg). Serà extremadament difícil que es formin tempestes, ja que es necessitaria un mecanisme de dispar molt potent per trencar-la.")
        elif cin < -25:
            conversa.append(f"La 'tapa' (CIN) de {cin:.0f} J/kg és considerable. Això pot retardar la convecció, però si es trenca, pot donar lloc a un desenvolupament explosiu. El CAPE utilitzable real és de {cape_u:.0f} J/kg.")
        else:
            conversa.append("La 'tapa' (CIN) és feble. L'energia està fàcilment disponible.")
    
    if pwat is not None:
        pwat_text = "molt seca" if pwat < 15 else "modesta, suficient per a xàfecs" if pwat < 30 else "abundant, ideal per a pluges fortes o torrencials"
        conversa.append(f"El contingut d'aigua precipitable (PWAT) és de {pwat:.1f} mm. Això indica una atmosfera {pwat_text}.")
        
    if lfc_agl is not None and lfc_agl > 3000:
         conversa.append(f"⚠️ Factor limitant clau: El nivell d'inici de convecció (LFC) està a {lfc_agl:.0f} m, una altura molt elevada. Això dificulta enormement la formació de tempestes des de la superfície.")
    elif lcl_agl is not None and lfc_agl is not None:
         conversa.append(f"La base del núvol (LCL) se situa a {lcl_agl:.0f} m, i el nivell on la convecció es dispara (LFC) a {lfc_agl:.0f} m.")

    conversa.append("--- ANÀLISI CINEMÀTICA ---")
    if shear6 is not None:
        if shear6 < 10:
            shear_text = "Molt feble. Les tempestes seran probablement desorganitzades i de cicle de vida curt (unicel·lulars)."
        elif shear6 < 18:
            shear_text = "Moderat. Hi ha potencial per a l'organització de les tempestes en sistemes multicel·lulars."
        else:
            shear_text = "Fort. Aquest cisallament és suficient per suportar el desenvolupament de supercèl·lules rotatòries."
        conversa.append(f"El cisallament del vent 0-6 km (Shear) és de {shear6:.1f} m/s. {shear_text}")

    if srh1 is not None and srh1 > 100:
        srh_text = "moderat, afavorint la rotació a nivells baixos" if srh1 < 250 else "fort, incrementant significativament el potencial de supercèl·lules i tornados"
        lcl_risk = " A més, la base del núvol és baixa, la qual cosa facilita que la rotació arribi a terra." if lcl_agl is not None and lcl_agl < 1200 else ""
        conversa.append(f"L'Helicitat Relativa a la Tempesta 0-1 km (SRH) és de {srh1:.0f} m²/s². Aquest és un valor {srh_text}.{lcl_risk}")

    conversa.append("--- SÍNTESI I RISCOS ASSOCIATS ---")
    if cape_u < 100 or (cin is not None and cin < -100) or (lfc_agl is not None and lfc_agl > 3000):
        conversa.append("Les condicions són desfavorables per a tempestes significatives degut a l'estabilitat, una tapa molt forta o un LFC inassolible.")
    elif shear6 is not None and shear6 > 18 and cape_u > 1000 and srh1 is not None and srh1 > 150:
        riscos = "calamarsa grossa, fortes ratxes de vent destructives i pluges torrencials"
        if srh1 > 250 and lcl_agl is not None and lcl_agl < 1200:
            riscos += ", amb un risc destacat de formació de tornados"
        conversa.append(f"L'entorn és altament favorable per a la formació de supercèl·lules**. El principal risc és {riscos}.")
    elif shear6 is not None and shear6 > 12 and cape_u > 500:
        conversa.append("La combinació d'energia i cisallament és òptima per a sistemes multicel·lulars organitzats. El risc principal seran les fortes pluges, calamarsa de mida mitjana i ratxes de vent fortes.")
    else:
        conversa.append("L'entorn afavoreix xàfecs o tempestes unicel·lulars. Aquestes seran generalment desorganitzades i de curta durada, tot i que poden produir localment pluja intensa i calamarsa petita.")

    return conversa

@st.cache_data
def obtener_sondeo_atmosferico(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    p_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"] for p in p_levels]
    params = {
        "latitude": lat, "longitude": lon, 
        "hourly": h_base + h_press, 
        "models": "arome_france", 
        "timezone": "auto", 
        "forecast_days": 1
    }
    try: 
        r = openmeteo.weather_api(url, params=params)
        return r[0] if r else None, p_levels
    except Exception as e: 
        st.error(f"Error a l'API d'Open-Meteo: {e}")
        return None, None

def calculate_parameters(p, T, Td, u, v, h):
    params = {}
    def get_val(qty, unit=None):
        try: return qty.to(unit).m if unit else qty.m
        except: return None
    raw_cape, raw_cin = None, None
    try:
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0])
        cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
        raw_cape = get_val(cape, 'J/kg'); raw_cin = get_val(cin, 'J/kg')
        params['CAPE_Brut'] = {'value': raw_cape, 'units': 'J/kg'}; params['CIN_Fre'] = {'value': raw_cin, 'units': 'J/kg'}
    except: pass
    if raw_cape is not None and raw_cin is not None: params['CAPE_Utilitzable'] = {'value': max(0, raw_cape - abs(raw_cin)), 'units': 'J/kg'}
    try: lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); lcl_h = mpcalc.pressure_to_height_std(lcl_p); params['LCL_AGL'] = {'value': get_val(lcl_h - h[0], 'm'), 'units': 'm'}
    except: pass
    try: lfc_p, _ = mpcalc.lfc(p, T, Td); lfc_h = mpcalc.pressure_to_height_std(lfc_p); params['LFC_AGL'] = {'value': get_val(lfc_h - h[0], 'm'), 'units': 'm'}
    except: pass
    try: el_p, _ = mpcalc.el(p, T, Td); el_h = mpcalc.pressure_to_height_std(el_p); params['EL_MSL'] = {'value': get_val(el_h, 'km'), 'units': 'km'}
    except: pass
    try: s_u, s_v = mpcalc.bulk_shear(p, u, v, height=h, depth=6*units.km); params['Shear_0-6km'] = {'value': get_val(mpcalc.wind_speed(s_u, s_v), 'm/s'), 'units': 'm/s'}
    except: pass
    try: _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=1*units.km); params['SRH_0-1km'] = {'value': get_val(srh), 'units': 'm²/s²'}
    except: pass
    try: _, srh, _ = mpcalc.storm_relative_helicity(h, u, v, depth=3*units.km); params['SRH_0-3km'] = {'value': get_val(srh), 'units': 'm²/s²'}
    except: pass
    try: pwat = mpcalc.precipitable_water(p, Td); params['PWAT_Total'] = {'value': get_val(pwat, 'mm'), 'units': 'mm'}
    except: pass
    return params

def crear_hodograf(p, u, v, h):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    hodo = Hodograph(ax, component_range=40.); hodo.add_grid(increment=10)
    hodoline = hodo.plot_colormapped(u, v, h.to('km'), cmap='gist_ncar')
    cbar = plt.colorbar(hodoline, ax=ax, orientation='vertical', pad=0.05, shrink=0.8); cbar.set_label('Altitud (km)')
    try: rm, _, _ = mpcalc.bunkers_storm_motion(p, u, v, h); hodo.plot_vectors(rm[0].to('kt'), rm[1].to('kt'), color='black', label='Mov. Tempesta (RM)')
    except: pass
    ax.set_xlabel('kt'); ax.set_ylabel('kt')
    return fig

def crear_skewt(p, T, Td, u, v):
    fig = plt.figure(figsize=(7, 9))
    skew = SkewT(fig, rotation=45)
    skew.plot(p, T, 'r', lw=2, label='T'); skew.plot(p, Td, 'b', lw=2, label='Td'); skew.plot_barbs(p, u, v, length=7, color='white')
    skew.plot_dry_adiabats(color='lightcoral', ls='--', alpha=0.5); skew.plot_moist_adiabats(color='cornflowerblue', ls='--', alpha=0.5); skew.plot_mixing_lines(color='lightgreen', ls='--', alpha=0.5)
    skew.ax.axvline(0, color='darkturquoise', linestyle='--', label='Isoterma 0°C')
    if len(p) > 1:
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', lw=2, ls='--', label='Parcela')
            wet_bulb_prof = mpcalc.wet_bulb_temperature(p, T, Td); skew.plot(p, wet_bulb_prof, color='purple', lw=1.5, label='Tª Humida')
            cape, cin = mpcalc.cape_cin(p, T, Td, prof)
            if cape.m > 0: skew.shade_cape(p, T, prof, alpha=0.4, color='khaki')
            if cin.m != 0: skew.shade_cin(p, T, prof, alpha=0.83, color='lightgray')
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); lfc_p, _ = mpcalc.lfc(p, T, Td, prof); el_p, _ = mpcalc.el(p, T, Td, prof)
            if lcl_p: skew.ax.axhline(lcl_p.m, color='purple', linestyle='--', label='LCL')
            if lfc_p: skew.ax.axhline(lfc_p.m, color='darkred', linestyle='--', label='LFC')
            if el_p: skew.ax.axhline(el_p.m, color='red', linestyle='--', label='EL')
        except: pass
    skew.ax.set_ylim(1050, 100); skew.ax.set_xlim(-50, 40); skew.ax.set_xlabel('°C'); skew.ax.set_ylabel('hPa'); plt.legend()
    return fig

def display_metrics(params_dict):
    param_map = [('CIN (Fre)', 'CIN_Fre'), ('CAPE (Brut)', 'CAPE_Brut'), ('Shear 0-6km', 'Shear_0-6km'), ('CAPE Utilitzable', 'CAPE_Utilitzable'), ('LCL (AGL)', 'LCL_AGL'), ('LFC (AGL)', 'LFC_AGL'), ('EL (MSL)', 'EL_MSL'), ('SRH 0-1km', 'SRH_0-1km'), ('SRH 0-3km', 'SRH_0-3km'), ('PWAT Total', 'PWAT_Total')]
    st.markdown("""<style>.metric-container{border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:10px;margin-bottom:10px;}</style>""", unsafe_allow_html=True)
    available_params = [ (label, key) for label, key in param_map ]
    
    cols = st.columns(min(4, len(available_params)))
    for i, (label, key) in enumerate(available_params):
        param = params_dict.get(key, {})
        value = param.get('value')
        units_str = param.get('units', '')

        if value is None or not isinstance(value, (int, float)):
            val_str = "Sense dades"
            units_display = ""
            emoji = ""
            val_color = "gray"
            border_color = "rgba(255,255,255,0.1)"
        else:
            val_str = f"{value:.1f}" if isinstance(value, float) else f"{value}"
            units_display = units_str
            border_color, emoji = get_parameter_style(key, value)
            val_color = border_color
        
        with cols[i % 4]:
            html = f"""
            <div class="metric-container" style="border-color:{border_color};">
                <div style="font-size:0.9em;color:gray;">{label}</div>
                <div style="font-size:1.25em;font-weight:bold;color:{val_color};">
                    {val_str} <span style='font-size:0.8em;color:gray;'>{units_display}</span> {emoji}
                </div>
            </div>"""
            st.markdown(html, unsafe_allow_html=True)

def crear_grafic_orografia(params, zero_iso_h_agl):
    lcl_agl = params.get('LCL_AGL', {}).get('value')
    lfc_agl = params.get('LFC_AGL', {}).get('value')
    if lcl_agl is None or np.isnan(lcl_agl): return None
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.set_yticks(np.arange(0, 10.1, 0.5)); ax.set_facecolor('#4169E1')
    sky_cmap = mcolors.LinearSegmentedColormap.from_list("sky", ["#87CEEB", "#4682B4"])
    ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect='auto', cmap=sky_cmap, origin='lower', extent=[0, 10, 0, 10])
    ax.add_patch(Circle((8, 8.5), 0.8, color='yellow', alpha=0.3, zorder=1))
    ax.add_patch(Circle((8, 8.5), 0.4, color='#FFFFE0', alpha=0.8, zorder=1))
    has_lfc = lfc_agl is not None and np.isfinite(lfc_agl)
    peak_h_m = lfc_agl if has_lfc else lcl_agl
    peak_h_km = peak_h_m / 1000.0 + 0.1
    m_verts = [(0, 0), (1.5, 0.1 * peak_h_km), (3, 0.5 * peak_h_km), (5, peak_h_km), (7, 0.4 * peak_h_km), (8.5, 0.1 * peak_h_km), (10,0)]
    mountain_path = Polygon(m_verts, color='none', zorder=5); ax.add_patch(mountain_path)
    offset = mtransforms.Affine2D().translate(5, -5)
    shadow_transform = ax.transData + offset
    shadow = patches.PathPatch(mountain_path.get_path(), facecolor='black', alpha=0.3, transform=shadow_transform, zorder=2); ax.add_patch(shadow)
    treeline_km = 1.9
    colors_veg = ['#2E4600', '#486B00', '#556B2F', '#5E412F']
    colors_rock = ['#696969', '#808080', '#A9A9A9']
    x_points, y_points = np.random.uniform(0, 10, 2000), np.random.uniform(0, peak_h_km, 2000)
    points_inside = mountain_path.get_path().contains_points(np.vstack((x_points, y_points)).T)
    patches_col = []
    for x, y in zip(x_points[points_inside], y_points[points_inside]):
        color = np.random.choice(colors_rock) if y > treeline_km else np.random.choice(colors_veg)
        patches_col.append(Circle((x, y), radius=np.random.rand() * 0.18 + 0.05, facecolor=color, alpha=0.7, edgecolor='none'))
    ax.add_collection(PatchCollection(patches_col, match_original=True, zorder=6))
    ax.add_patch(Polygon(m_verts, facecolor='none', edgecolor='black', lw=1.5, zorder=7))
    if zero_iso_h_agl is not None and peak_h_km > zero_iso_h_agl.m / 1000:
        h_snow = zero_iso_h_agl.m / 1000
        x_snow = np.linspace(0, 10, 200)
        y_mountain = np.interp(x_snow, [p[0] for p in m_verts], [p[1] for p in m_verts])
        ax.fill_between(x_snow, np.maximum(h_snow, y_mountain), peak_h_km + 1, where=y_mountain>=h_snow, facecolor='white', alpha=0.9, zorder=8)
    for _ in range(70):
        x, y = -0.5 + np.random.rand() * 11, lcl_agl/1000 + (np.random.rand()-0.5) * 0.3
        ax.add_patch(Circle((x, y), radius=0.2 + np.random.rand() * 0.6, facecolor='white', alpha=0.5, edgecolor='lightgray', lw=0.5, zorder=9))
    for i in range(40):
        x_base, height = np.random.rand() * 10, np.random.rand() * 0.2 + 0.05
        ax.add_patch(Polygon([(x_base-0.08, 0), (x_base, height), (x_base+0.08, 0)], facecolor=np.random.choice(['#004d00', '#003300']), zorder=10))
    ax.axhline(lcl_agl/1000, color='grey', linestyle='--', lw=2.5, zorder=11)
    ax.text(-0.2, lcl_agl/1000, f" LCL ({lcl_agl:.0f} m) ", color='white', backgroundcolor='black', ha='right', va='center', weight='bold', fontsize=10)
    if has_lfc:
        ax.axhline(lfc_agl/1000, color='red', linestyle='--', lw=2.5, zorder=11)
        ax.text(10.2, lfc_agl/1000, f" LFC ({lfc_agl:.0f} m) ", color='white', backgroundcolor='red', ha='left', va='center', weight='bold', fontsize=10)
        ax.text(5, lfc_agl/1000 + 0.3, f" Altura de muntanya necessària per activar tempestes: {lfc_agl:.0f} m ", color='black', bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'), ha='center', va='center', weight='bold', fontsize=14, zorder=12)
    else:
        ax.text(5, 5, "No hi ha LFC accessible.\nL'orografia no pot iniciar convecció profunda.", ha='center', va='center', color='white', fontsize=14, weight='bold', bbox=dict(facecolor='darkblue', alpha=0.8, boxstyle='round,pad=0.5'))
    ax.set_ylim(0, 10); ax.set_xlim(0, 10); ax.set_ylabel("Altitud sobre el terra (km)"); ax.set_title("Potencial d'Activació per Orografia", weight='bold', fontsize=16)
    ax.set_xticklabels([]); ax.set_xticks([]); fig.tight_layout()
    return fig

def crear_grafic_nuvol(params, H, u, v, is_convergence_active):
    lcl_agl = params.get('LCL_AGL', {}).get('value')
    lfc_agl = params.get('LFC_AGL', {}).get('value')
    el_msl_km = params.get('EL_MSL', {}).get('value')
    cape = params.get('CAPE_Brut', {}).get('value', 0)
    srh1 = params.get('SRH_0-1km', {}).get('value')
    
    if lcl_agl is None or el_msl_km is None: return None
    
    fig, ax = plt.subplots(figsize=(6, 9), dpi=120)
    ax.set_facecolor('#4F94CD'); sky_cmap = mcolors.LinearSegmentedColormap.from_list("sky", ["#4F94CD", "#B0E0E6"])
    ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect='auto', cmap=sky_cmap, origin='lower', extent=[-5, 5, 0, 16])
    ax.add_patch(Polygon([(-5, 0), (5, 0), (5, 0.5), (-5, 0.5)], color='#3A1F04'))
    
    lcl_km = lcl_agl / 1000
    el_km = el_msl_km - (H[0].m / 1000)
    
    for _ in range(70):
        x, y = -5 + random.random() * 10, lcl_km + (random.random() - 0.5) * 0.3
        ax.add_patch(Circle((x, y), 0.3 + random.random() * 0.7, color='white', alpha=0.3, lw=0))
    
    if srh1 is not None and srh1 > 250 and lcl_km < 1.2: base_txt = "Potencial de Wall Cloud i Funnels (Tornados)"
    elif srh1 is not None and srh1 > 150 and lcl_km < 1.5: base_txt = "Potencial de Bases Giratories (Mesocicló)"
    elif cape > 1500: base_txt = "Bases Turbulentes (Shelf Cloud / Arcus)"
    else: base_txt = "Base Plana (Sense organització rotacional)"
    ax.text(0, -0.5, base_txt, color='white', ha='center', weight='bold', fontsize=12)
    
    if is_convergence_active and lfc_agl is not None and np.isfinite(lfc_agl):
        lfc_km = lfc_agl / 1000
        y_points = np.linspace(lfc_km, el_km, 100)
        cloud_width = 1.0 + np.sin(np.pi * (y_points - lfc_km) / (el_km - lfc_km)) * (1 + cape/2500)
        
        for y, width in zip(y_points, cloud_width):
            center_x = np.interp(y*1000, H.m, u.m) / 15
            for _ in range(30):
                offset_x, offset_y = (random.random() - 0.5) * width, (random.random() - 0.5) * 0.4
                ax.add_patch(Circle((center_x + offset_x, y + offset_y), 0.2 + random.random() * 0.4, color='white', alpha=0.15, lw=0))
        
        anvil_wind_u = np.interp(el_km*1000, H.m, u.m) / 10
        anvil_center_x = np.interp(el_km*1000, H.m, u.m) / 15
        for _ in range(100):
            offset_x, offset_y = (random.random() - 0.2) * 4 + anvil_wind_u, (random.random() - 0.5) * 0.5
            ax.add_patch(Circle((anvil_center_x + offset_x, el_km + offset_y), 0.2 + random.random() * 0.6, color='white', alpha=0.2, lw=0))
        if cape > 2500:
            ot_height = el_km + cape/5000
            ax.add_patch(Circle((anvil_center_x, ot_height), 0.4, color='white', alpha=0.5))
    else:
        ax.text(0, 8, "Convergència insuficient\nper desenvolupar\nconvecció profunda.", ha='center', va='center', color='white', fontsize=16, weight='bold', bbox=dict(facecolor='darkblue', alpha=0.7, boxstyle='round,pad=0.5'))

    barb_heights_km = np.arange(1, 15, 1)
    u_barbs, v_barbs = (np.interp(barb_heights_km * 1000, H.m, comp.to('kt').m) for comp in (u, v))
    ax.barbs(np.full_like(barb_heights_km, 4.5), barb_heights_km, u_barbs, v_barbs, length=7, color='black')

    ax.set_ylim(0, 16); ax.set_xlim(-5, 5)
    ax.set_ylabel("Altitud sobre el terra (km)"); ax.set_title("Visualització del Núvol", weight='bold')
    ax.set_xticks([]); ax.grid(axis='y', linestyle='--', alpha=0.3)
    return fig

@st.cache_data
def obtener_dades_mapa_vents(hora, nivell):
    lats = np.linspace(40.5, 42.8, 12)
    lons = np.linspace(0.2, 3.3, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    params = {
        "latitude": lat_grid.flatten().tolist(),
        "longitude": lon_grid.flatten().tolist(),
        "hourly": [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"],
        "models": "arome_france", "timezone": "auto", "forecast_days": 1
    }
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        responses = openmeteo.weather_api(url, params=params)
        lats_out, lons_out, speeds_out, dirs_out = [], [], [], []
        for r in responses:
            hourly = r.Hourly()
            speed = hourly.Variables(0).ValuesAsNumpy()[hora]
            direction = hourly.Variables(1).ValuesAsNumpy()[hora]
            if not np.isnan(speed) and not np.isnan(direction):
                lats_out.append(r.Latitude())
                lons_out.append(r.Longitude())
                speeds_out.append(speed)
                dirs_out.append(direction)
        return lats_out, lons_out, speeds_out, dirs_out
    except:
        return None, None, None, None

def crear_mapa_vents(lats, lons, u_comp, v_comp, comarcas, nivell):
    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([0, 3.5, 40.4, 43], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', zorder=1)

    grid_lon = np.linspace(min(lons), max(lons), 100)
    grid_lat = np.linspace(min(lats), max(lats), 100)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    points = np.vstack((lons, lats)).T
    u_grid = griddata(points, u_comp.m, (X, Y), method='cubic')
    v_grid = griddata(points, v_comp.m, (X, Y), method='cubic')
    u_grid, v_grid = np.nan_to_num(u_grid), np.nan_to_num(v_grid)
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y)
    divergence = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5

    conv_threshold = -5.5
    divergence_values = divergence.m
    divergence_strong_conv = np.ma.masked_where(divergence_values > conv_threshold, divergence_values)
    
    levels = np.linspace(-15.0, conv_threshold, 10)
    
    cs = ax.contourf(X, Y, divergence_strong_conv,
                     levels=levels, cmap='Reds_r', alpha=0.6,
                     zorder=2, transform=ccrs.PlateCarree(), extend='min')

    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid,
                  color="#000000", density=5.9, linewidth=0.5,
                  arrowsize=0.50, zorder=4, transform=ccrs.PlateCarree())
        
    ax.set_title(f"Flux i focus de convergència a {nivell}hPa", weight='bold')
    return fig

@st.cache_data
def encontrar_localitats_con_convergencia(hora, nivell, localitats, threshold):
    lats, lons, speeds, dirs = obtener_dades_mapa_vents(hora, nivell)
    if not lats or len(lats) < 4: return []

    speeds_ms = (np.array(speeds) * 1000 / 3600) * units('m/s')
    dirs_deg = np.array(dirs) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    
    grid_lon = np.linspace(min(lons), max(lons), 100)
    grid_lat = np.linspace(min(lats), max(lats), 100)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    
    points = np.vstack((lons, lats)).T
    u_grid = griddata(points, u_comp.m, (X, Y), method='cubic')
    v_grid = griddata(points, v_comp.m, (X, Y), method='cubic')
    u_grid, v_grid = np.nan_to_num(u_grid), np.nan_to_num(v_grid)
    
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y)
    divergence = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5
    divergence_values = divergence.m

    localitats_en_convergencia = []
    for nom_poble, coords in localitats.items():
        lon_idx = (np.abs(grid_lon - coords['lon'])).argmin()
        lat_idx = (np.abs(grid_lat - coords['lat'])).argmin()
        valor_convergencia = divergence_values[lat_idx, lon_idx]
        
        if valor_convergencia < threshold:
            localitats_en_convergencia.append(nom_poble)
            
    return localitats_en_convergencia

# --- INTERFAZ PRINCIPAL ---
st.markdown("""
<style>
.main-title { font-size: 3.5em; font-weight: bold; text-align: center; margin-bottom: -10px; color: #FFFFFF; }
.subtitle { font-size: 1.2em; text-align: center; color: #A0A0A0; margin-bottom: 25px; }
.chat-bubble { background-color:#262D31; border-radius:15px; padding:12px 18px; margin-bottom:10px; max-width:95%; border:1px solid rgba(255,255,255,0.1); }
.avis-box { padding: 15px; border-radius: 10px; border: 2px solid; margin-bottom: 20px; text-align: center; font-size: 1.1em; font-weight: bold; }
.update-info { text-align: center; color: gray; font-style: italic; margin-top: -15px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ Tempestes.cat</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Eina d\'Anàlisi i Previsió de Fenòmens Severs a Catalunya</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    try:
        tz = pytz.timezone('Europe/Madrid')
        now_local = datetime.now(tz)
        default_hour_index = now_local.hour
    except:
        default_hour_index = 12 
        
    hour_options = [f"{h:02d}:00h" for h in range(24)]
    hora_sel_str = st.radio("Hora del pronòstic (Local):", hour_options, index=default_hour_index, horizontal=True)
    hora = int(hora_sel_str.split(':')[0])

with col2:
    p_levels_all = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    nivell_global = st.selectbox("Nivell d'anàlisi de vents:", p_levels_all, index=p_levels_all.index(850))

st.markdown(f'<p class="update-info">🕒 {get_next_arome_update_time()}</p>', unsafe_allow_html=True)

with st.spinner(f"Analitzant convergències a {nivell_global}hPa per a les {hora}:00h..."):
    conv_threshold = -5.5
    localitats_convergencia = encontrar_localitats_con_convergencia(hora, nivell_global, pobles_data, conv_threshold)

opciones_display = []
for nom_poble in sorted(pobles_data.keys()):
    if nom_poble in localitats_convergencia:
        opciones_display.append(f"📍 {nom_poble} (Convergència a les {hora}:00h)")
    else:
        opciones_display.append(nom_poble)

poble_sel_display = st.selectbox('Selecciona una localitat:', options=opciones_display)
poble_sel = poble_sel_display.replace('📍 ', '').split(' (')[0]
lat_sel, lon_sel = pobles_data[poble_sel]['lat'], pobles_data[poble_sel]['lon']

sondeo, p_levels = obtener_sondeo_atmosferico(lat_sel, lon_sel)

if sondeo:
    data_is_valid = False
    with st.spinner(f"Processant dades per a {poble_sel}..."):
        hourly = sondeo.Hourly(); T_s, Td_s, P_s = (hourly.Variables(i).ValuesAsNumpy()[hora] for i in range(3))
        if np.isnan(P_s): st.error(f"Dades de pressió superficial no disponibles per les {hora}:00h.")
        else:
            s_idx, n_plvls = 3, len(p_levels); T_p, Td_p, Ws_p, Wd_p, H_p = ([hourly.Variables(s_idx + i*n_plvls + j).ValuesAsNumpy()[hora] for j in range(n_plvls)] for i in range(5))
            def interpolate_sfc(sfc_val, p_sfc, p_api, d_api):
                if np.isnan(sfc_val):
                    valid_p = [p for p, t in zip(p_api, d_api) if not np.isnan(t)]; valid_d = [t for t in d_api if not np.isnan(t)]
                    if len(valid_p) > 1: p_sorted, d_sorted = zip(*sorted(zip(valid_p, valid_d))); return np.interp(p_sfc, p_sorted, d_sorted)
                return sfc_val
            T_s = interpolate_sfc(T_s, P_s, p_levels, T_p); Td_s = interpolate_sfc(Td_s, P_s, p_levels, Td_p)
            if not np.isnan(T_s) and not np.isnan(Td_s):
                p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [P_s], [T_s], [Td_s], [0.0], [0.0], [mpcalc.pressure_to_height_std(P_s*units.hPa).m]
                for i, p_level in enumerate(p_levels):
                    if p_level < P_s and not np.isnan(T_p[i]):
                        p_profile.append(p_level); T_profile.append(T_p[i]); Td_profile.append(Td_p[i]); h_profile.append(H_p[i])
                        u_comp, v_comp = mpcalc.wind_components(Ws_p[i]*units.knots, Wd_p[i]*units.degrees)
                        u_profile.append(u_comp.to('m/s').m); v_profile.append(v_comp.to('m/s').m)
                p = np.array(p_profile)*units.hPa; T = np.array(T_profile)*units.degC; Td = np.array(Td_profile)*units.degC
                H = np.array(h_profile)*units.m; u = np.array(u_profile)*units.m/units.s; v = np.array(v_profile)*units.m/units.s
                zero_iso_h_agl = None
                try:
                    T_c = T.to('degC').m; H_m = H.to('m').m
                    zero_cross_indices = np.where(np.diff(np.sign(T_c)))[0]
                    if zero_cross_indices.size > 0:
                        idx = zero_cross_indices[0]
                        h_zero_iso_msl = np.interp(0, [T_c[idx+1], T_c[idx]], [H_m[idx+1], H_m[idx]])
                        zero_iso_h_agl = (h_zero_iso_msl - H_m[0]) * units.m
                except Exception: pass
                parametros = calculate_parameters(p, T, Td, u, v, H); data_is_valid = True
    if data_is_valid:
        avis_text, avis_color = generar_avis_localitat(parametros)
        st.markdown(f'<div class="avis-box" style="border-color: {avis_color}; background-color: {avis_color}20;">{avis_text}</div>', unsafe_allow_html=True)

        tab_list = ["🗨️ Anàlisi Detallada", "📊 Paràmetres", "🗺️ Mapes de Vents", "🔄 Hodògraf", "🏔️ Sondeig", "🗺️ Orografia", "☁️ Visualització"]
        selected_tab = st.radio("Navegació:", tab_list, index=0, horizontal=True)
        
        if selected_tab == tab_list[0]:
            conversa = generar_analisi_detallada(parametros)
            for msg in conversa:
                st.markdown(f'<div class="chat-bubble">🧑‍🔬 {msg}</div>', unsafe_allow_html=True)
        elif selected_tab == tab_list[1]:
            st.subheader("Paràmetres Clau"); display_metrics(parametros)
        elif selected_tab == tab_list[2]:
            st.subheader(f"Vents i Convergència a {nivell_global}hPa")
            with st.spinner("Generant mapa de vents... 🌬️💨"):
                lats_map, lons_map, speeds_map, dirs_map = obtener_dades_mapa_vents(hora, nivell_global)
                if lats_map and len(lats_map) > 4:
                    speeds_ms = (np.array(speeds_map) * 1000 / 3600) * units('m/s')
                    dirs_deg = np.array(dirs_map) * units.degrees
                    u_map, v_map = mpcalc.wind_components(speeds_ms, dirs_deg)
                    fig_vents = crear_mapa_vents(lats_map, lons_map, u_map, v_map, pobles_data, nivell_global)
                    st.pyplot(fig_vents)
                else:
                    st.error("No s'han pogut obtenir les dades per al mapa de vents o no hi ha prous punts de dades per a aquest nivell i hora.")
        elif selected_tab == tab_list[3]:
            st.subheader("Hodògraf (0-10 km)"); st.pyplot(crear_hodograf(p, u, v, H))
        elif selected_tab == tab_list[4]:
            st.subheader(f"Sondeig per a {poble_sel} ({hora}:00h Local)"); st.pyplot(crear_skewt(p, T, Td, u, v))
        elif selected_tab == tab_list[5]:
            st.subheader("Potencial d'Activació per Orografia")
            fig_oro = crear_grafic_orografia(parametros, zero_iso_h_agl)
            if fig_oro: st.pyplot(fig_oro)
            else: st.info("No hi ha LCL o LFC, per tant no es pot calcular el potencial d'activació orogràfica.")
        elif selected_tab == tab_list[6]:
            with st.spinner("Dibuixant la possible estructura del núvol... ☁️⚡️"):
                st.subheader("Visualització del Núvol")
                is_conv_active = poble_sel in localitats_convergencia
                fig_nuvol = crear_grafic_nuvol(parametros, H, u, v, is_convergence_active=is_conv_active)
                if fig_nuvol: st.pyplot(fig_nuvol)
                else: st.info("No hi ha LCL o EL, per tant no es pot visualitzar l'estructura del núvol.")
    else:
        st.warning(f"No s'han pogut calcular els paràmetres per a les {hora}:00h. Pot ser que el model no tingui dades completes per a aquest punt i hora. Prova amb una altra hora o localitat.")
else:
    st.error("No s'han pogut obtenir dades. Pot ser que la localitat estigui fora de la cobertura del model AROME.")
