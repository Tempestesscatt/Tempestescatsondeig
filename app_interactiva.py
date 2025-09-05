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
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import pytz
import google.generativeai as genai
import io
from PIL import Image
import json
import hashlib
import os
import base64
import threading
import pandas as pd
import xml.etree.ElementTree as ET
from streamlit_option_menu import option_menu
from math import radians, sin, cos, sqrt, atan2, degrees
from scipy.ndimage import gaussian_filter
import uuid
from global_land_mask import globe
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap, BoundaryNorm
import imageio
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point





# --- 0. CONFIGURACIÓ I CONSTANTS ---
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever")

# --- Clients API ---
parcel_lock = threading.Lock()
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)



WEBCAM_LINKS = {
    # Catalunya (Aquests permeten 'embed')
    "Barcelona": {'type': 'embed', 'url': "https://www.youtube.com/embed/2i_o-a_I73s?autoplay=1&mute=1"},
    "Tarragona": {'type': 'embed', 'url': "https://www.youtube.com/embed/YpCY_oE852g?autoplay=1&mute=1"},

    # Tornado Alley (EUA) - Aquests NO permeten 'embed', els marquem com a 'direct'
    "Dallas, TX": {'type': 'direct', 'url': "https://www.youtube.com/watch?v=for_g-h2H6s"},
    "Wichita, KS": {'type': 'direct', 'url': "https://www.youtube.com/watch?v=iN7u2jKbDm0"},
    "Houston, TX": {'type': 'direct', 'url': "https://www.youtube.com/watch?v=SDK_m1_BVJ4"},
    "Kansas City, MO": {'type': 'direct', 'url': "https://www.youtube.com/watch?v=Ezm7jMHsx5A"},

    # Regne Unit i Irlanda (Aquests permeten 'embed')
    "Southampton": {'type': 'embed', 'url': "https://www.youtube.com/embed/QO-hO_kwwmY?autoplay=1&mute=1"},
    "Fort William": {'type': 'direct', 'url': "https://www.youtube.com/live/8miQ3QXA26Q?autoplay=1&mute=1"},
    "Dublín (Collins Ave)": {'type': 'embed', 'url': "https://www.youtube.com/embed/g1r59JJqY60?autoplay=1&mute=1"},
    "Weymouth": {'type': 'embed', 'url': "https://www.youtube.com/embed/vw6m4ORi1KI?autoplay=1&mute=1"},

    # Canadà (Aquests permeten 'embed')
    "Revelstoke, BC": {'type': 'embed', 'url': "https://www.youtube.com/embed/fIMbMz2P7Bs?autoplay=1&mute=1"},
    "Banff, AB": {'type': 'embed', 'url': "https://www.youtube.com/embed/_0wPODlF9wU?autoplay=1&mute=1"},
    "Calgary, AB": {'type': 'embed', 'url': "https://www.youtube.com/embed/MwcqP3ta6RI?autoplay=1&mute=1"},
    "Vancouver, BC": {'type': 'embed', 'url': "https://www.youtube.com/embed/-2vwOXTxbkw?autoplay=1&mute=1"},
    
    # Japó (Aquests permeten 'embed')
    "Tòquio": {'type': 'embed', 'url': "https://www.youtube.com/embed/_k-5U7IeK8g?autoplay=1&mute=1"},
    "Oshino Hakkai (Fuji)": {'type': 'embed', 'url': "https://www.youtube.com/embed/sm3xXTfDtGE?autoplay=1&mute=1"},
    "Hasaki Beach": {'type': 'embed', 'url': "https://www.youtube.com/embed/Ntz4h44KTDc?autoplay=1&mute=1"},
    "Hakodate": {'type': 'embed', 'url': "https://www.youtube.com/embed/sE1bH-zc9Pg?autoplay=1&mute=1"},

    # Alemanya – ciutats (permiten 'embed')
    "Berlín (Alexanderplatz)": {'type': 'direct', 'url': "https://www.youtube.com/watch?v=IRqboacDNFg"},
    "Hamburg (St. Michaelis)": {'type': 'direct', 'url': "https://www.youtube.com/live/mfpdquRilCk?autoplay=1&mute=1"},  
    "Múnich (Marienplatz)": {'type': 'embed', 'url': "https://www.youtube.com/embed/KxWuwC7R5kY?autoplay=1&mute=1"}, 
    "Bensersiel (Costa Nord)": {'type': 'embed', 'url': "https://www.youtube.com/embed/aYtgGjMDagw?autoplay=1&mute=1"}, 
    "Harz (Hahnenklee)": {'type': 'direct', 'url': "https://www.youtube.com/live/hM6G0VuAWtg?autoplay=1&mute=1"}, 


    "Pavullo nel Frignano": {'type': 'embed', 'url': "https://www.youtube.com/embed/xqJpFlttsf8?autoplay=1&mute=1"},
    "Castel San Pietro": {'type': 'embed', 'url': "https://www.youtube.com/embed/c2seGcq0u0o?autoplay=1&mute=1"},
    "Brescia": {'type': 'embed', 'url': "https://www.youtube.com/embed/edyIH3pVyRE?autoplay=1&mute=1"},
    "Stresa (Lago Maggiore)": {'type': 'embed', 'url': "https://www.youtube.com/embed/hc6e8Bf2-a0?autoplay=1&mute=1"},
    "Frontino (Montefeltro)": {'type': 'embed', 'url':  "https://www.youtube.com/embed/pv5PQ1EtKBE?autoplay=1&mute=1"},
    "Roma": {'type': 'embed', 'url': "https://www.youtube.com/embed/RDqrx6S2z20?autoplay=1&mute=1"},
    "Florència": {'type': 'embed', 'url': "https://www.youtube.com/embed/4eNyDCa1DBU?autoplay=1&mute=1"},
    "Massa Lubrense": {'type': 'embed', 'url': "https://www.torrecangiani.com/it/massa-lubrense-webcam/"},
    "Capo d'Orlando": {'type': 'embed', 'url': "https://www.youtube.com/embed/PEcs1ghWkaM?autoplay=1&mute=1"},
    "Ajaccio (Còrsega)": {'type': 'embed', 'url': "https://www.vision-environnement.com/livecams/webcam.php?webcam=ajaccio-panorama"},


    "Amsterdam": {'type': 'embed', 'url': "https://www.youtube.com/embed/ZnOoxCd7BGU?autoplay=1&mute=1"},
    "Volendam": {'type': 'embed', 'url': "https://www.youtube.com/embed/9UpAVPmtPtA?autoplay=1&mute=1"},
    "Zandvoort": {'type': 'embed', 'url': "https://www.youtube.com/embed/KiPmDwgTAu0?autoplay=1&mute=1"},


      # Noruega (Aquests permeten 'embed')
    "Oslo": {'type': 'embed', 'url': "https://www.youtube.com/embed/f1ZvS1Kuwhw?autoplay=1&mute=1"},
    "Bergen": {'type': 'embed', 'url': "https://www.youtube.com/embed/Z2SiE-MSfVY?autoplay=1&mute=1"},
    "Tromsø": {'type': 'embed', 'url': "https://www.youtube.com/embed/3y7_fkAzzps?autoplay=1&mute=1"},
    "Stavanger":  {'type': 'embed', 'url': "https://www.youtube.com/embed/RA6Jm7sv_F4?autoplay=1&mute=1"},




}




API_URL_NORUEGA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_NORUEGA = pytz.timezone('Europe/Oslo')
CIUTATS_NORUEGA = {
    'Oslo': {'lat': 59.9139, 'lon': 10.7522, 'sea_dir': (120, 210)},
    'Bergen': {'lat': 60.3913, 'lon': 5.3221, 'sea_dir': (200, 340)},
    'Stavanger': {'lat': 58.9700, 'lon': 5.7331, 'sea_dir': (180, 350)},
    'Tromsø': {'lat': 69.6492, 'lon': 18.9553, 'sea_dir': (0, 360)},
}
MAP_EXTENT_NORUEGA = [4, 32, 57, 71] # Extensió que cobreix des del sud fins al nord
# Els nivells de pressió del model UKMO Seamless són molt detallats
PRESS_LEVELS_NORUEGA = sorted([
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 
    375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100
], reverse=True)



# --- Constants per al Canadà Continental ---
API_URL_CANADA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_CANADA = pytz.timezone('America/Edmonton') # Canviem a Mountain Time, més representatiu
CIUTATS_CANADA = {
    'Revelstoke, BC': {'lat': 51.0024, 'lon': -118.1963, 'sea_dir': None},
    'Banff, AB': {'lat': 51.1784, 'lon': -115.5708, 'sea_dir': None},
    'Calgary, AB': {'lat': 51.0447, 'lon': -114.0719, 'sea_dir': None},
    'Vancouver, BC': {'lat': 49.2827, 'lon': -123.1207, 'sea_dir': (100, 260)},
}
MAP_EXTENT_CANADA = [-125, -110, 48, 54] # Ajustem el mapa a les noves localitats (BC i Alberta)
# Llista de nivells de pressió extremadament detallada per al model HRDPS
PRESS_LEVELS_CANADA = sorted([
    1015, 1000, 985, 970, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 
    675, 650, 625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375, 350, 325, 
    300, 275, 250, 225, 200, 175, 150, 125, 100
], reverse=True)


API_URL_UK = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_UK = pytz.timezone('Europe/London')
CIUTATS_UK = {
    'Southampton': {'lat': 50.9097, 'lon': -1.4044, 'sea_dir': (135, 225)},
    'Fort William': {'lat': 56.8167, 'lon': -5.1121, 'sea_dir': (200, 250)},
    'Dublín (Collins Ave)': {'lat': 53.3498, 'lon': -6.2603, 'sea_dir': (50, 150)},
    'Weymouth': {'lat': 50.6144, 'lon': -2.4551, 'sea_dir': (135, 225)},
}
MAP_EXTENT_UK = [-11, 2, 49, 59]
PRESS_LEVELS_UK = sorted([
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 
    375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100, 70, 50, 40, 30, 20, 10
], reverse=True)


API_URL_JAPO = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_JAPO = pytz.timezone('Asia/Tokyo')
CIUTATS_JAPO = {
    'Tòquio': {'lat': 35.6895, 'lon': 139.6917, 'sea_dir': (100, 200)},
    'Oshino Hakkai (Fuji)': {'lat': 35.4590, 'lon': 138.8340, 'sea_dir': None},
    'Hasaki Beach': {'lat': 35.7330, 'lon': 140.8440, 'sea_dir': (45, 135)},
    'Hakodate': {'lat': 41.7687, 'lon': 140.7288, 'sea_dir': (120, 270)},
}
MAP_EXTENT_JAPO = [128, 146, 30, 46] # Manté una bona cobertura
# Llista de nivells de pressió completa per al model JMA MSM
PRESS_LEVELS_JAPO = sorted([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100], reverse=True)


API_URL_HOLANDA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_HOLANDA = pytz.timezone('Europe/Amsterdam')
CIUTATS_HOLANDA = {
    # <<<--- LLISTA DE CIUTATS COMPLETAMENT ACTUALITZADA --->>>
    'Amsterdam': {'lat': 52.3779, 'lon': 4.8970, 'sea_dir': (220, 320)},
    'Volendam': {'lat': 52.4946, 'lon': 5.0718, 'sea_dir': (90, 220)},
    'Zandvoort': {'lat': 52.3725, 'lon': 4.5325, 'sea_dir': (220, 320)},
    
}
MAP_EXTENT_HOLANDA = [3.5, 7.5, 50.7, 53.7] # Ajustat per a les noves localitats
PRESS_LEVELS_HOLANDA = sorted([1000, 925, 850, 700, 500, 300], reverse=True)


API_URL_ITALIA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_ITALIA = pytz.timezone('Europe/Rome')
CIUTATS_ITALIA = {
    # <<<--- LLISTA DE CIUTATS COMPLETAMENT ACTUALITZADA --->>>
    'Pavullo nel Frignano': {'lat': 44.3315, 'lon': 10.8344, 'sea_dir': None},
    'Castel San Pietro': {'lat': 44.3989, 'lon': 11.5910, 'sea_dir': None},
    'Brescia': {'lat': 45.5388, 'lon': 10.2205, 'sea_dir': None},
    'Stresa (Lago Maggiore)': {'lat': 45.8856, 'lon': 8.5283, 'sea_dir': None},
    'Frontino (Montefeltro)': {'lat': 43.7667, 'lon': 12.3789, 'sea_dir': None},
    'Roma': {'lat': 41.9028, 'lon': 12.4964, 'sea_dir': (190, 280)},
    'Florència': {'lat': 43.7696, 'lon': 11.2558, 'sea_dir': None},
    'Massa Lubrense': {'lat': 40.6105, 'lon': 14.3467, 'sea_dir': (180, 300)},
    "Capo d'Orlando": {'lat': 38.1504, 'lon': 14.7397, 'sea_dir': (270, 360)},
    'Ajaccio (Còrsega)': {'lat': 41.9268, 'lon': 8.7369, 'sea_dir': (180, 300)},
}
MAP_EXTENT_ITALIA = [6.5, 18.5, 36.5, 47.0]
PRESS_LEVELS_ITALIA = sorted([1000, 925, 850, 700, 500, 250], reverse=True)


API_URL_ALEMANYA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_ALEMANYA = pytz.timezone('Europe/Berlin')
CIUTATS_ALEMANYA = {
    # <<<--- NOMS ACTUALITZATS: HEM CANVIAT COLÒNIA PER HARZ (HAHNENKLEE) --->>>
    'Bensersiel (Costa Nord)': {'lat': 53.676, 'lon': 7.568, 'sea_dir': (270, 360)},
    'Berlín (Alexanderplatz)': {'lat': 52.5219, 'lon': 13.4132, 'sea_dir': None},
    'Múnich (Marienplatz)': {'lat': 48.1374, 'lon': 11.5755, 'sea_dir': None},
    'Hamburg (St. Michaelis)': {'lat': 53.5484, 'lon': 9.9788, 'sea_dir': (290, 360)}, 
    'Harz (Hahnenklee)': {'lat': 51.855, 'lon': 10.339, 'sea_dir': None}, # Nova localització
}
MAP_EXTENT_ALEMANYA = [5.5, 15.5, 47.0, 55.5]
PRESS_LEVELS_ICON = sorted([1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)



# --- Constants per Catalunya ---
API_URL_CAT = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_CAT = pytz.timezone('Europe/Madrid')
CIUTATS_CATALUNYA = {
    # Capitals de província
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734, 'sea_dir': (90, 190)},
    'Girona': {'lat': 41.9831, 'lon': 2.8249, 'sea_dir': (80, 170)},
    'Lleida': {'lat': 41.6177, 'lon': 0.6200, 'sea_dir': None},
    'Tarragona': {'lat': 41.1189, 'lon': 1.2445, 'sea_dir': (110, 220)},

    # LLISTA COMPLETA I VERIFICADA
    'Agramunt': {'lat': 41.7871, 'lon': 1.0967, 'sea_dir': None},
    'Alcanar': {'lat': 40.5434, 'lon': 0.4820, 'sea_dir': (60, 160)},
    'Alella': {'lat': 41.4947, 'lon': 2.2955, 'sea_dir': (90, 180)},
    'Altafulla': {'lat': 41.1417, 'lon': 1.3750, 'sea_dir': (110, 220)},
    'Amposta': {'lat': 40.7093, 'lon': 0.5810, 'sea_dir': (70, 170)},
    'Arbúcies': {'lat': 41.8159, 'lon': 2.5152, 'sea_dir': None},
    'Arenys de Mar': {'lat': 41.5815, 'lon': 2.5504, 'sea_dir': (90, 180)},
    'Arenys de Munt': {'lat': 41.6094, 'lon': 2.5411, 'sea_dir': None},
    'Balaguer': {'lat': 41.7904, 'lon': 0.8066, 'sea_dir': None},
    'Banyoles': {'lat': 42.1197, 'lon': 2.7667, 'sea_dir': (80, 170)},
    'Begur': {'lat': 41.9542, 'lon': 3.2076, 'sea_dir': (0, 180)},
    'Bellver de Cerdanya': {'lat': 42.3705, 'lon': 1.7770, 'sea_dir': None},
    'Berga': {'lat': 42.1051, 'lon': 1.8458, 'sea_dir': None},
    'Blanes': {'lat': 41.6748, 'lon': 2.7917, 'sea_dir': (80, 180)},
    'Cabrera de Mar': {'lat': 41.5275, 'lon': 2.3958, 'sea_dir': (90, 180)},
    'Cadaqués': {'lat': 42.2888, 'lon': 3.2770, 'sea_dir': (0, 180)},
    'Calaf': {'lat': 41.7311, 'lon': 1.5126, 'sea_dir': None},
    'Caldes de Montbui': {'lat': 41.6315, 'lon': 2.1678, 'sea_dir': None},
    'Calella': {'lat': 41.6146, 'lon': 2.6653, 'sea_dir': (90, 180)},
    'Calonge': {'lat': 41.8601, 'lon': 3.0768, 'sea_dir': (80, 190)},
    'Camarasa': {'lat': 41.8753, 'lon': 0.8804, 'sea_dir': None},
    'Cambrils': {'lat': 41.0667, 'lon': 1.0500, 'sea_dir': (110, 220)},
    'Capellades': {'lat': 41.5312, 'lon': 1.6874, 'sea_dir': None},
    'Cardedeu': {'lat': 41.6403, 'lon': 2.3582, 'sea_dir': (90, 180)},
    'Cardona': {'lat': 41.9138, 'lon': 1.6806, 'sea_dir': None},
    'Cassà de la Selva': {'lat': 41.8893, 'lon': 2.8736, 'sea_dir': (80, 170)},
    'Castellbisbal': {'lat': 41.4776, 'lon': 1.9866, 'sea_dir': None},
    'Castellar del Vallès': {'lat': 41.6186, 'lon': 2.0875, 'sea_dir': None},
    'Castellfollit de la Roca': {'lat': 42.2201, 'lon': 2.5517, 'sea_dir': None},
    'Castelló d\'Empúries': {'lat': 42.2582, 'lon': 3.0725, 'sea_dir': (70, 160)},
    'Centelles': {'lat': 41.7963, 'lon': 2.2203, 'sea_dir': None},
    'Cerdanyola del Vallès': {'lat': 41.4925, 'lon': 2.1415, 'sea_dir': (100, 200)},
    'Figueres': {'lat': 42.2662, 'lon': 2.9622, 'sea_dir': (70, 160)},
    'Flaçà': {'lat': 42.0494, 'lon': 2.9559, 'sea_dir': (80, 170)},
    'Granollers': {'lat': 41.6083, 'lon': 2.2886, 'sea_dir': (90, 180)},
    'Hostalric': {'lat': 41.7479, 'lon': 2.6360, 'sea_dir': None},
    'Igualada': {'lat': 41.5791, 'lon': 1.6174, 'sea_dir': None},
    'L\'Ametlla de Mar': {'lat': 40.8824, 'lon': 0.8016, 'sea_dir': (90, 200)},
    'L\'Escala': {'lat': 42.1235, 'lon': 3.1311, 'sea_dir': (0, 160)},
    'L\'Hospitalet de Llobregat': {'lat': 41.3571, 'lon': 2.1030, 'sea_dir': (90, 190)},
    'La Bisbal d\'Empordà': {'lat': 41.9602, 'lon': 3.0378, 'sea_dir': (80, 170)},
    'La Jonquera': {'lat': 42.4194, 'lon': 2.8752, 'sea_dir': None},
    'La Pobla de Segur': {'lat': 42.2472, 'lon': 0.9678, 'sea_dir': None},
    'La Selva del Camp': {'lat': 41.2131, 'lon': 1.1384, 'sea_dir': (110, 220)},
    'La Sénia': {'lat': 40.6322, 'lon': 0.2831, 'sea_dir': None},
    'La Seu d\'Urgell': {'lat': 42.3582, 'lon': 1.4593, 'sea_dir': None},
    'Llagostera': {'lat': 41.8291, 'lon': 2.8931, 'sea_dir': (80, 180)},
    'Llançà': {'lat': 42.3625, 'lon': 3.1539, 'sea_dir': (0, 150)},
    'Lloret de Mar': {'lat': 41.7005, 'lon': 2.8450, 'sea_dir': (80, 180)},
    'Malgrat de Mar': {'lat': 41.6461, 'lon': 2.7423, 'sea_dir': (90, 180)},
    'Manlleu': {'lat': 42.0016, 'lon': 2.2844, 'sea_dir': None},
    'Manresa': {'lat': 41.7230, 'lon': 1.8268, 'sea_dir': None},
    'Mataró': {'lat': 41.5388, 'lon': 2.4449, 'sea_dir': (90, 180)},
    'Mollet del Vallès': {'lat': 41.5385, 'lon': 2.2144, 'sea_dir': (100, 200)},
    'Montblanc': {'lat': 41.3761, 'lon': 1.1610, 'sea_dir': None},
    'Montcada i Reixac': {'lat': 41.4851, 'lon': 2.1884, 'sea_dir': (100, 200)},
    'Olesa de Montserrat': {'lat': 41.5451, 'lon': 1.8955, 'sea_dir': None},
    'Olot': {'lat': 42.1818, 'lon': 2.4900, 'sea_dir': None},
    'Palamós': {'lat': 41.8465, 'lon': 3.1287, 'sea_dir': (80, 190)},
    'Pals': {'lat': 41.9688, 'lon': 3.1458, 'sea_dir': (0, 180)},
    'Pineda de Mar': {'lat': 41.6277, 'lon': 2.6908, 'sea_dir': (90, 180)},
    'Platja d\'Aro': {'lat': 41.8175, 'lon': 3.0645, 'sea_dir': (80, 190)},
    'Puigcerdà': {'lat': 42.4331, 'lon': 1.9287, 'sea_dir': None},
    'Reus': {'lat': 41.1550, 'lon': 1.1075, 'sea_dir': (120, 220)},
    'Ripoll': {'lat': 42.2013, 'lon': 2.1903, 'sea_dir': None},
    'Riudellots de la Selva': {'lat': 41.9080, 'lon': 2.8099, 'sea_dir': (80, 170)},
    'Roses': {'lat': 42.2619, 'lon': 3.1764, 'sea_dir': (90, 200)},
    'Rubí': {'lat': 41.4936, 'lon': 2.0323, 'sea_dir': (100, 200)},
    'Sabadell': {'lat': 41.5483, 'lon': 2.1075, 'sea_dir': (100, 200)},
    'Salou': {'lat': 41.0763, 'lon': 1.1417, 'sea_dir': (110, 220)},
    'Sant Cugat del Vallès': {'lat': 41.4727, 'lon': 2.0863, 'sea_dir': (100, 200)},
    'Sant Feliu de Guíxols': {'lat': 41.7801, 'lon': 3.0278, 'sea_dir': (80, 190)},
    'Sant Feliu de Llobregat': {'lat': 41.3833, 'lon': 2.0500, 'sea_dir': (100, 200)},
    'Sant Joan de les Abadesses': {'lat': 42.2355, 'lon': 2.2858, 'sea_dir': None},
    'Sant Pere de Ribes': {'lat': 41.2599, 'lon': 1.7725, 'sea_dir': (100, 220)},
    'Sant Quirze del Vallès': {'lat': 41.5303, 'lon': 2.0831, 'sea_dir': None},
    'Santa Coloma de Farners': {'lat': 41.8596, 'lon': 2.6703, 'sea_dir': None},
    'Santa Coloma de Gramenet': {'lat': 41.4550, 'lon': 2.2111, 'sea_dir': (90, 190)},
    'Santa Cristina d\'Aro': {'lat': 41.8130, 'lon': 2.9976, 'sea_dir': (80, 190)},
    'Santa Pau': {'lat': 42.1448, 'lon': 2.5695, 'sea_dir': None},
    'Santa Susanna': {'lat': 41.6366, 'lon': 2.7098, 'sea_dir': (90, 180)},
    'Sarroca de Bellera': {'lat': 42.3957, 'lon': 0.8656, 'sea_dir': None},
    'Sitges': {'lat': 41.2351, 'lon': 1.8117, 'sea_dir': (100, 220)},
    'Solsona': {'lat': 41.9942, 'lon': 1.5161, 'sea_dir': None},
    'Sort': {'lat': 42.4131, 'lon': 1.1278, 'sea_dir': None},
    'Soses': {'lat': 41.5358, 'lon': 0.5186, 'sea_dir': None},
    'Terrassa': {'lat': 41.5615, 'lon': 2.0084, 'sea_dir': (100, 200)},
    'Tortosa': {'lat': 40.8126, 'lon': 0.5211, 'sea_dir': (60, 160)},
    'Tremp': {'lat': 42.1664, 'lon': 0.8953, 'sea_dir': None},
    'Valls': {'lat': 41.2872, 'lon': 1.2505, 'sea_dir': (110, 220)},
    'Vic': {'lat': 41.9301, 'lon': 2.2545, 'sea_dir': None},
    'Vidrà': {'lat': 42.1226, 'lon': 2.3116, 'sea_dir': None},
    'Vidreres': {'lat': 41.7876, 'lon': 2.7788, 'sea_dir': (80, 180)},
    'Vielha': {'lat': 42.7027, 'lon': 0.7966, 'sea_dir': None},
    'Vilafranca del Penedès': {'lat': 41.3453, 'lon': 1.6995, 'sea_dir': (100, 200)},
    'Vilanova i la Geltrú': {'lat': 41.2241, 'lon': 1.7252, 'sea_dir': (100, 200)},
    'Viladecans': {'lat': 41.3155, 'lon': 2.0194, 'sea_dir': (100, 200)},
    'Vilassar de Dalt': {'lat': 41.5167, 'lon': 2.3583, 'sea_dir': None},
    'Vilassar de Mar': {'lat': 41.5057, 'lon': 2.3920, 'sea_dir': (90, 180)},
    
    # Punts Marins
    'Costes de Girona (Mar)':   {'lat': 42.05, 'lon': 3.30, 'sea_dir': (0, 360)},
    'Litoral Barceloní (Mar)': {'lat': 41.40, 'lon': 2.90, 'sea_dir': (0, 360)},
    'Aigües de Tarragona (Mar)': {'lat': 40.90, 'lon': 2.00, 'sea_dir': (0, 360)},
}

POBLES_MAPA_REFERENCIA = {
    # Capitals de província (es poden repetir si també vols que surtin sempre al mapa)
    "Barcelona": {'lat': 41.3851, 'lon': 2.1734}, "Girona": {'lat': 41.9831, 'lon': 2.8249},
    "Lleida": {'lat': 41.6177, 'lon': 0.6200}, "Tarragona": {'lat': 41.1189, 'lon': 1.2445},

    # Llista de pobles addicionals només per al mapa
    "Altafulla": {'lat': 41.1417, 'lon': 1.3750}, "Agramunt": {'lat': 41.7871, 'lon': 1.0967},
    "Alcanar": {'lat': 40.5434, 'lon': 0.4820}, "Alella": {'lat': 41.4947, 'lon': 2.2955},
    "Arenys de Mar": {'lat': 41.5815, 'lon': 2.5504}, "Arenys de Munt": {'lat': 41.6094, 'lon': 2.5411},
    "Balaguer": {'lat': 41.7904, 'lon': 0.8066}, "Berga": {'lat': 42.1051, 'lon': 1.8458},
    "Banyoles": {'lat': 42.1197, 'lon': 2.7667}, "Cabrera de Mar": {'lat': 41.5275, 'lon': 2.3958},
    "Caldes de Montbui": {'lat': 41.6315, 'lon': 2.1678}, "Calella": {'lat': 41.6146, 'lon': 2.6653},
    "Calaf": {'lat': 41.7311, 'lon': 1.5126}, "Camarasa": {'lat': 41.8753, 'lon': 0.8804},
    "Capellades": {'lat': 41.5312, 'lon': 1.6874}, "Cardedeu": {'lat': 41.6403, 'lon': 2.3582},
    "Cardona": {'lat': 41.9138, 'lon': 1.6806}, "Castellbisbal": {'lat': 41.4776, 'lon': 1.9866},
    "Castellar del Vallès": {'lat': 41.6186, 'lon': 2.0875}, "Castelló d'Empúries": {'lat': 42.2582, 'lon': 3.0725},
    "Centelles": {'lat': 41.7963, 'lon': 2.2203}, "Cerdanyola del Vallès": {'lat': 41.4925, 'lon': 2.1415},
    "Figueres": {'lat': 42.2662, 'lon': 2.9622}, "Flaçà": {'lat': 42.0494, 'lon': 2.9559},
    "Granollers": {'lat': 41.6083, 'lon': 2.2886}, "Igualada": {'lat': 41.5791, 'lon': 1.6174},
    "L'Ametlla de Mar": {'lat': 40.8824, 'lon': 0.8016}, "L'Escala": {'lat': 42.1235, 'lon': 3.1311},
    "L'Hospitalet de Llobregat": {'lat': 41.3571, 'lon': 2.1030}, "La Bisbal d'Empordà": {'lat': 41.9602, 'lon': 3.0378},
    "La Jonquera": {'lat': 42.4194, 'lon': 2.8752}, "La Seu d'Urgell": {'lat': 42.3582, 'lon': 1.4593},
    "La Selva del Camp": {'lat': 41.2131, 'lon': 1.1384}, "La Sénia": {'lat': 40.6322, 'lon': 0.2831},
    "Manresa": {'lat': 41.7230, 'lon': 1.8268}, "Mataró": {'lat': 41.5388, 'lon': 2.4449},
    "Mollet del Vallès": {'lat': 41.5385, 'lon': 2.2144}, "Montblanc": {'lat': 41.3761, 'lon': 1.1610},
    "Montcada i Reixac": {'lat': 41.4851, 'lon': 2.1884}, "Olot": {'lat': 42.1818, 'lon': 2.4900},
    "Olesa de Montserrat": {'lat': 41.5451, 'lon': 1.8955}, "Palamós": {'lat': 41.8465, 'lon': 3.1287},
    "Pals": {'lat': 41.9688, 'lon': 3.1458}, "Pineda de Mar": {'lat': 41.6277, 'lon': 2.6908},
    "Reus": {'lat': 41.1550, 'lon': 1.1075}, "Ripoll": {'lat': 42.2013, 'lon': 2.1903},
    "Roses": {'lat': 42.2619, 'lon': 3.1764}, "Rubí": {'lat': 41.4936, 'lon': 2.0323},
    "Sabadell": {'lat': 41.5483, 'lon': 2.1075}, "Sant Cugat del Vallès": {'lat': 41.4727, 'lon': 2.0863},
    "Sant Feliu de Guíxols": {'lat': 41.7801, 'lon': 3.0278}, "Sant Feliu de Llobregat": {'lat': 41.3833, 'lon': 2.0500},
    "Sant Joan de les Abadesses": {'lat': 42.2355, 'lon': 2.2858}, "Sant Quirze del Vallès": {'lat': 41.5303, 'lon': 2.0831},
    "Santa Coloma de Farners": {'lat': 41.8596, 'lon': 2.6703}, "Santa Coloma de Gramenet": {'lat': 41.4550, 'lon': 2.2111},
    "Sarroca de Bellera": {'lat': 42.3957, 'lon': 0.8656}, "Soses": {'lat': 41.5358, 'lon': 0.5186},
    "Solsona": {'lat': 41.9942, 'lon': 1.5161}, "Sort": {'lat': 42.4131, 'lon': 1.1278},
    "Terrassa": {'lat': 41.5615, 'lon': 2.0084}, "Tortosa": {'lat': 40.8126, 'lon': 0.5211},
    "Valls": {'lat': 41.2872, 'lon': 1.2505}, "Vic": {'lat': 41.9301, 'lon': 2.2545},
    "Vielha": {'lat': 42.7027, 'lon': 0.7966}, "Vilafranca del Penedès": {'lat': 41.3453, 'lon': 1.6995},
    "Vilanova i la Geltrú": {'lat': 41.2241, 'lon': 1.7252}, "Blanes": {'lat': 41.6748, 'lon': 2.7917},
    "Llançà": {'lat': 42.3625, 'lon': 3.1539}, "Platja d’Aro": {'lat': 41.8175, 'lon': 3.0645},
    "Sitges": {'lat': 41.2351, 'lon': 1.8117}, "Cadaqués": {'lat': 42.2888, 'lon': 3.2770},
    "Cambrils": {'lat': 41.0667, 'lon': 1.0500}, "Salou": {'lat': 41.0763, 'lon': 1.1417},
    "Vidreres": {'lat': 41.7876, 'lon': 2.7788}, "Begur": {'lat': 41.9542, 'lon': 3.2076},
    "Castellfollit de la Roca": {'lat': 42.2201, 'lon': 2.5517}, "Santa Pau": {'lat': 42.1448, 'lon': 2.5695},
    "La Pobla de Segur": {'lat': 42.2472, 'lon': 0.9678}, "Bellver de Cerdanya": {'lat': 42.3705, 'lon': 1.7770},
    "Puigcerdà": {'lat': 42.4331, 'lon': 1.9287}, "Manlleu": {'lat': 42.0016, 'lon': 2.2844},
    "Tremp": {'lat': 42.1664, 'lon': 0.8953}, "Arbúcies": {'lat': 41.8159, 'lon': 2.5152},
    "Viladecans": {'lat': 41.3155, 'lon': 2.0194}, "Vilassar de Mar": {'lat': 41.5057, 'lon': 2.3920},
    "Vilassar de Dalt": {'lat': 41.5167, 'lon': 2.3583}, "Sant Pere de Ribes": {'lat': 41.2599, 'lon': 1.7725},
    "Santa Susanna": {'lat': 41.6366, 'lon': 2.7098}, "Malgrat de Mar": {'lat': 41.6461, 'lon': 2.7423},
    "Calonge": {'lat': 41.8601, 'lon': 3.0768}, "Lloret de Mar": {'lat': 41.7005, 'lon': 2.8450},
    "Santa Cristina d'Aro": {'lat': 41.8130, 'lon': 2.9976}, "Cassà de la Selva": {'lat': 41.8893, 'lon': 2.8736},
    "Vidrà": {'lat': 42.1226, 'lon': 2.3116}, "Llagostera": {'lat': 41.8291, 'lon': 2.8931},
    "Riudellots de la Selva": {'lat': 41.9080, 'lon': 2.8099}, "Hostalric": {'lat': 41.7479, 'lon': 2.6360}
}





CIUTATS_PER_ZONA_PERSONALITZADA = {
    "Pirineu i Pre-Pirineu": {
        'Vielha': CIUTATS_CATALUNYA['Vielha'],
        'Sort': CIUTATS_CATALUNYA['Sort'],
        'Sarroca de Bellera': CIUTATS_CATALUNYA['Sarroca de Bellera'],
        'Tremp': CIUTATS_CATALUNYA['Tremp'],
        'La Pobla de Segur': CIUTATS_CATALUNYA['La Pobla de Segur'],
        'La Seu d\'Urgell': CIUTATS_CATALUNYA['La Seu d\'Urgell'],
        'Puigcerdà': CIUTATS_CATALUNYA['Puigcerdà'],
        'Bellver de Cerdanya': CIUTATS_CATALUNYA['Bellver de Cerdanya'],
        'Ripoll': CIUTATS_CATALUNYA['Ripoll'],
        'Sant Joan de les Abadesses': CIUTATS_CATALUNYA['Sant Joan de les Abadesses'],
        'Berga': CIUTATS_CATALUNYA['Berga'],
        'Solsona': CIUTATS_CATALUNYA['Solsona'],
        'Olot': CIUTATS_CATALUNYA['Olot'],
        'Santa Pau': CIUTATS_CATALUNYA['Santa Pau'],
        'Castellfollit de la Roca': CIUTATS_CATALUNYA['Castellfollit de la Roca'],
    },
    "Plana de Lleida i Ponent": {
        'Lleida': CIUTATS_CATALUNYA['Lleida'],
        'Soses': CIUTATS_CATALUNYA['Soses'],
        'Balaguer': CIUTATS_CATALUNYA['Balaguer'],
        'Agramunt': CIUTATS_CATALUNYA['Agramunt'],
        'Camarasa': CIUTATS_CATALUNYA['Camarasa'],
        'Calaf': CIUTATS_CATALUNYA['Calaf'],
    },
    "Catalunya Central": {
        'Manresa': CIUTATS_CATALUNYA['Manresa'],
        'Cardona': CIUTATS_CATALUNYA['Cardona'],
        'Igualada': CIUTATS_CATALUNYA['Igualada'],
        'Capellades': CIUTATS_CATALUNYA['Capellades'],
        'Vic': CIUTATS_CATALUNYA['Vic'],
        'Manlleu': CIUTATS_CATALUNYA['Manlleu'],
        'Centelles': CIUTATS_CATALUNYA['Centelles'],
        'Vidrà': CIUTATS_CATALUNYA['Vidrà'],
    },
    "Litoral i Prelitoral Nord (Girona)": {
        'Girona': CIUTATS_CATALUNYA['Girona'],
        'Figueres': CIUTATS_CATALUNYA['Figueres'],
        'Banyoles': CIUTATS_CATALUNYA['Banyoles'],
        'La Bisbal d\'Empordà': CIUTATS_CATALUNYA['La Bisbal d\'Empordà'],
        'Roses': CIUTATS_CATALUNYA['Roses'],
        'Cadaqués': CIUTATS_CATALUNYA['Cadaqués'],
        'Llançà': CIUTATS_CATALUNYA['Llançà'],
        'L\'Escala': CIUTATS_CATALUNYA['L\'Escala'],
        'Castelló d\'Empúries': CIUTATS_CATALUNYA['Castelló d\'Empúries'],
        'La Jonquera': CIUTATS_CATALUNYA['La Jonquera'],
        'Palamós': CIUTATS_CATALUNYA['Palamós'],
        'Platja d\'Aro': CIUTATS_CATALUNYA['Platja d\'Aro'],
        'Sant Feliu de Guíxols': CIUTATS_CATALUNYA['Sant Feliu de Guíxols'],
        'Begur': CIUTATS_CATALUNYA['Begur'],
        'Pals': CIUTATS_CATALUNYA['Pals'],
        'Calonge': CIUTATS_CATALUNYA['Calonge'],
        'Santa Cristina d\'Aro': CIUTATS_CATALUNYA['Santa Cristina d\'Aro'],
        'Blanes': CIUTATS_CATALUNYA['Blanes'],
        'Lloret de Mar': CIUTATS_CATALUNYA['Lloret de Mar'],
        'Santa Coloma de Farners': CIUTATS_CATALUNYA['Santa Coloma de Farners'],
        'Arbúcies': CIUTATS_CATALUNYA['Arbúcies'],
        'Hostalric': CIUTATS_CATALUNYA['Hostalric'],
        'Cassà de la Selva': CIUTATS_CATALUNYA['Cassà de la Selva'],
        'Llagostera': CIUTATS_CATALUNYA['Llagostera'],
        'Flaçà': CIUTATS_CATALUNYA['Flaçà'],
        'Riudellots de la Selva': CIUTATS_CATALUNYA['Riudellots de la Selva'],
        'Vidreres': CIUTATS_CATALUNYA['Vidreres'],
    },
    "Litoral i Prelitoral Central (Barcelona)": {
        'Barcelona': CIUTATS_CATALUNYA['Barcelona'],
        'L\'Hospitalet de Llobregat': CIUTATS_CATALUNYA['L\'Hospitalet de Llobregat'],
        'Santa Coloma de Gramenet': CIUTATS_CATALUNYA['Santa Coloma de Gramenet'],
        'Sabadell': CIUTATS_CATALUNYA['Sabadell'],
        'Terrassa': CIUTATS_CATALUNYA['Terrassa'],
        'Mataró': CIUTATS_CATALUNYA['Mataró'],
        'Granollers': CIUTATS_CATALUNYA['Granollers'],
        'Mollet del Vallès': CIUTATS_CATALUNYA['Mollet del Vallès'],
        'Sant Cugat del Vallès': CIUTATS_CATALUNYA['Sant Cugat del Vallès'],
        'Rubí': CIUTATS_CATALUNYA['Rubí'],
        'Viladecans': CIUTATS_CATALUNYA['Viladecans'],
        'Castellbisbal': CIUTATS_CATALUNYA['Castellbisbal'],
        'Cerdanyola del Vallès': CIUTATS_CATALUNYA['Cerdanyola del Vallès'],
        'Montcada i Reixac': CIUTATS_CATALUNYA['Montcada i Reixac'],
        'Sant Quirze del Vallès': CIUTATS_CATALUNYA['Sant Quirze del Vallès'],
        'Castellar del Vallès': CIUTATS_CATALUNYA['Castellar del Vallès'],
        'Cardedeu': CIUTATS_CATALUNYA['Cardedeu'],
        'Caldes de Montbui': CIUTATS_CATALUNYA['Caldes de Montbui'],
        'Arenys de Mar': CIUTATS_CATALUNYA['Arenys de Mar'],
        'Calella': CIUTATS_CATALUNYA['Calella'],
        'Malgrat de Mar': CIUTATS_CATALUNYA['Malgrat de Mar'],
        'Pineda de Mar': CIUTATS_CATALUNYA['Pineda de Mar'],
        'Santa Susanna': CIUTATS_CATALUNYA['Santa Susanna'],
        'Vilassar de Mar': CIUTATS_CATALUNYA['Vilassar de Mar'],
        'Alella': CIUTATS_CATALUNYA['Alella'],
        'Cabrera de Mar': CIUTATS_CATALUNYA['Cabrera de Mar'],
        'Vilanova i la Geltrú': CIUTATS_CATALUNYA['Vilanova i la Geltrú'],
        'Sitges': CIUTATS_CATALUNYA['Sitges'],
        'Vilafranca del Penedès': CIUTATS_CATALUNYA['Vilafranca del Penedès'],
        'Sant Pere de Ribes': CIUTATS_CATALUNYA['Sant Pere de Ribes'],
        'Olesa de Montserrat': CIUTATS_CATALUNYA['Olesa de Montserrat'],
        'Sant Feliu de Llobregat': CIUTATS_CATALUNYA['Sant Feliu de Llobregat'],
    },
    "Camp de Tarragona": {
        'Tarragona': CIUTATS_CATALUNYA['Tarragona'],
        'Reus': CIUTATS_CATALUNYA['Reus'],
        'Valls': CIUTATS_CATALUNYA['Valls'],
        'Salou': CIUTATS_CATALUNYA['Salou'],
        'Cambrils': CIUTATS_CATALUNYA['Cambrils'],
        'Altafulla': CIUTATS_CATALUNYA['Altafulla'],
        'La Selva del Camp': CIUTATS_CATALUNYA['La Selva del Camp'],
        'Montblanc': CIUTATS_CATALUNYA['Montblanc'],
    },
    "Terres de l'Ebre": {
        'Tortosa': CIUTATS_CATALUNYA['Tortosa'],
        'Amposta': CIUTATS_CATALUNYA['Amposta'],
        'Alcanar': CIUTATS_CATALUNYA['Alcanar'],
        'L\'Ametlla de Mar': CIUTATS_CATALUNYA['L\'Ametlla de Mar'],
        'La Sénia': CIUTATS_CATALUNYA['La Sénia'],
    },
}

POBLES_IMPORTANTS = {
    "Barcelona", "Girona", "Lleida", "Tarragona", "Altafulla", "Agramunt", "Alcanar", 
    "Alella", "Arenys de Mar", "Arenys de Munt", "Balaguer", "Berga", "Banyoles", 
    "Cabrera de Mar", "Caldes de Montbui", "Calella", "Calaf", "Camarasa", "Capellades", 
    "Cardedeu", "Cardona", "Castellbisbal", "Castellar del Vallès", "Castelló d'Empúries", 
    "Centelles", "Cerdanyola del Vallès", "Figueres", "Flaçà", "Granollers", "Igualada", 
    "L'Ametlla de Mar", "L'Escala", "L'Hospitalet de Llobregat", "La Bisbal d'Empordà", 
    "La Jonquera", "La Seu d'Urgell", "La Selva del Camp", "La Sénia", "Manresa", "Mataró", 
    "Mollet del Vallès", "Montblanc", "Montcada i Reixac", "Olot", "Olesa de Montserrat", 
    "Palamós", "Pals", "Pineda de Mar", "Reus", "Ripoll", "Roses", "Rubí", "Sabadell", 
    "Sant Cugat del Vallès", "Sant Feliu de Guíxols", "Sant Feliu de Llobregat", 
    "Sant Joan de les Abadesses", "Sant Quirze del Vallès", "Santa Coloma de Farners", 
    "Santa Coloma de Gramenet", "Sarroca de Bellera", "Soses", "Solsona", "Sort", 
    "Terrassa", "Tortosa", "Valls", "Vic", "Vielha", "Vilafranca del Penedès", 
    "Vilanova i la Geltrú", "Blanes", "Llançà", "Platja d’Aro", "Sitges", "Cadaqués", 
    "Cambrils", "Salou", "Vidreres", "Begur", "Castellfollit de la Roca", "Santa Pau", 
    "La Pobla de Segur", "Bellver de Cerdanya", "Puigcerdà", "Manlleu", "Tremp", 
    "Arbúcies", "Viladecans", "Vilassar de Mar", "Vilassar de Dalt", "Sant Pere de Ribes", 
    "Santa Susanna", "Malgrat de Mar", "Calonge", "Lloret de Mar", "Santa Cristina d'Aro", 
    "Cassà de la Selva", "Vidrà", "Llagostera", "Riudellots de la Selva", "Hostalric"
}

POBLACIONS_TERRA = {k: v for k, v in CIUTATS_CATALUNYA.items() if '(Mar)' not in k}
PUNTS_MAR = {k: v for k, v in CIUTATS_CATALUNYA.items() if '(Mar)' in k}
CIUTATS_CONVIDAT = {
    'Barcelona': CIUTATS_CATALUNYA['Barcelona'], 'Girona': CIUTATS_CATALUNYA['Girona'],
    'Lleida': CIUTATS_CATALUNYA['Lleida'], 'Tarragona': CIUTATS_CATALUNYA['Tarragona']
}
MAP_EXTENT_CAT = [0, 3.5, 40.4, 43]
PRESS_LEVELS_AROME = sorted([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)

MAP_ZOOM_LEVELS_CAT = {
    'Catalunya (Complet)': MAP_EXTENT_CAT, 

    # Capitals de província amb marge ampliat
    'Barcelona': [1.8, 2.6, 41.25, 41.65],
    'Girona': [2.5, 3.4, 41.8, 42.2],
    'Lleida': [0.3, 0.95, 41.5, 41.75],
    'Tarragona': [0.9, 1.35, 40.95, 41.3]
}

CIUTATS_PER_COMARCA = {
    "Alt Camp": {
        'Valls': {'lat': 41.2872, 'lon': 1.2505, 'sea_dir': (110, 220)},
    },
    "Alt Empordà": {
        'Cadaqués': {'lat': 42.2888, 'lon': 3.2770, 'sea_dir': (0, 180)},
        'Castelló d\'Empúries': {'lat': 42.2582, 'lon': 3.0725, 'sea_dir': (70, 160)},
        'Figueres': {'lat': 42.2662, 'lon': 2.9622, 'sea_dir': (70, 160)},
        'L\'Escala': {'lat': 42.1235, 'lon': 3.1311, 'sea_dir': (0, 160)},
        'La Jonquera': {'lat': 42.4194, 'lon': 2.8752, 'sea_dir': None},
        'Llançà': {'lat': 42.3625, 'lon': 3.1539, 'sea_dir': (0, 150)},
        'Roses': {'lat': 42.2619, 'lon': 3.1764, 'sea_dir': (90, 200)},
    },
    "Alt Penedès": {
        'Vilafranca del Penedès': {'lat': 41.3453, 'lon': 1.6995, 'sea_dir': (100, 200)},
    },
    "Alt Urgell": {
        'La Seu d\'Urgell': {'lat': 42.3582, 'lon': 1.4593, 'sea_dir': None},
    },
    "Anoia": {
        'Calaf': {'lat': 41.7311, 'lon': 1.5126, 'sea_dir': None},
        'Capellades': {'lat': 41.5312, 'lon': 1.6874, 'sea_dir': None},
        'Igualada': {'lat': 41.5791, 'lon': 1.6174, 'sea_dir': None},
    },
    "Bages": {
        'Cardona': {'lat': 41.9138, 'lon': 1.6806, 'sea_dir': None},
        'Manresa': {'lat': 41.7230, 'lon': 1.8268, 'sea_dir': None},
    },
    "Baix Camp": {
        'Cambrils': {'lat': 41.0667, 'lon': 1.0500, 'sea_dir': (110, 220)},
        'La Selva del Camp': {'lat': 41.2131, 'lon': 1.1384, 'sea_dir': (110, 220)},
        'Reus': {'lat': 41.1550, 'lon': 1.1075, 'sea_dir': (120, 220)},
    },
    "Baix Ebre": {
        'L\'Ametlla de Mar': {'lat': 40.8824, 'lon': 0.8016, 'sea_dir': (90, 200)},
        'Tortosa': {'lat': 40.8126, 'lon': 0.5211, 'sea_dir': (60, 160)},
    },
    "Baix Empordà": {
        'Begur': {'lat': 41.9542, 'lon': 3.2076, 'sea_dir': (0, 180)},
        'Calonge': {'lat': 41.8601, 'lon': 3.0768, 'sea_dir': (80, 190)},
        'La Bisbal d\'Empordà': {'lat': 41.9602, 'lon': 3.0378, 'sea_dir': (80, 170)},
        'Palamós': {'lat': 41.8465, 'lon': 3.1287, 'sea_dir': (80, 190)},
        'Pals': {'lat': 41.9688, 'lon': 3.1458, 'sea_dir': (0, 180)},
        'Platja d\'Aro': {'lat': 41.8175, 'lon': 3.0645, 'sea_dir': (80, 190)},
        'Sant Feliu de Guíxols': {'lat': 41.7801, 'lon': 3.0278, 'sea_dir': (80, 190)},
        'Santa Cristina d\'Aro': {'lat': 41.8130, 'lon': 2.9976, 'sea_dir': (80, 190)},
    },
    "Baix Llobregat": {
        'Castellbisbal': {'lat': 41.4776, 'lon': 1.9866, 'sea_dir': None},
        'L\'Hospitalet de Llobregat': {'lat': 41.3571, 'lon': 2.1030, 'sea_dir': (90, 190)},
        'Olesa de Montserrat': {'lat': 41.5451, 'lon': 1.8955, 'sea_dir': None},
        'Sant Feliu de Llobregat': {'lat': 41.3833, 'lon': 2.0500, 'sea_dir': (100, 200)},
        'Viladecans': {'lat': 41.3155, 'lon': 2.0194, 'sea_dir': (100, 200)},
    },
    "Barcelonès": {
        'Barcelona': {'lat': 41.3851, 'lon': 2.1734, 'sea_dir': (90, 190)},
        'Santa Coloma de Gramenet': {'lat': 41.4550, 'lon': 2.2111, 'sea_dir': (90, 190)},
    },
    "Berguedà": {
        'Berga': {'lat': 42.1051, 'lon': 1.8458, 'sea_dir': None},
    },
    "Cerdanya": {
        'Bellver de Cerdanya': {'lat': 42.3705, 'lon': 1.7770, 'sea_dir': None},
        'Puigcerdà': {'lat': 42.4331, 'lon': 1.9287, 'sea_dir': None},
    },
    "Conca de Barberà": {
        'Montblanc': {'lat': 41.3761, 'lon': 1.1610, 'sea_dir': None},
    },
    "Garraf": {
        'Sant Pere de Ribes': {'lat': 41.2599, 'lon': 1.7725, 'sea_dir': (100, 220)},
        'Sitges': {'lat': 41.2351, 'lon': 1.8117, 'sea_dir': (100, 220)},
        'Vilanova i la Geltrú': {'lat': 41.2241, 'lon': 1.7252, 'sea_dir': (100, 200)},
    },
    "Garrotxa": {
        'Castellfollit de la Roca': {'lat': 42.2201, 'lon': 2.5517, 'sea_dir': None},
        'Olot': {'lat': 42.1818, 'lon': 2.4900, 'sea_dir': None},
        'Santa Pau': {'lat': 42.1448, 'lon': 2.5695, 'sea_dir': None},
    },
    "Gironès": {
        'Cassà de la Selva': {'lat': 41.8893, 'lon': 2.8736, 'sea_dir': (80, 170)},
        'Flaçà': {'lat': 42.0494, 'lon': 2.9559, 'sea_dir': (80, 170)},
        'Girona': {'lat': 41.9831, 'lon': 2.8249, 'sea_dir': (80, 170)},
        'Llagostera': {'lat': 41.8291, 'lon': 2.8931, 'sea_dir': (80, 180)},
        'Riudellots de la Selva': {'lat': 41.9080, 'lon': 2.8099, 'sea_dir': (80, 170)},
    },
    "Maresme": {
        'Alella': {'lat': 41.4947, 'lon': 2.2955, 'sea_dir': (90, 180)},
        'Arenys de Mar': {'lat': 41.5815, 'lon': 2.5504, 'sea_dir': (90, 180)},
        'Arenys de Munt': {'lat': 41.6094, 'lon': 2.5411, 'sea_dir': None},
        'Cabrera de Mar': {'lat': 41.5275, 'lon': 2.3958, 'sea_dir': (90, 180)},
        'Calella': {'lat': 41.6146, 'lon': 2.6653, 'sea_dir': (90, 180)},
        'Malgrat de Mar': {'lat': 41.6461, 'lon': 2.7423, 'sea_dir': (90, 180)},
        'Mataró': {'lat': 41.5388, 'lon': 2.4449, 'sea_dir': (90, 180)},
        'Pineda de Mar': {'lat': 41.6277, 'lon': 2.6908, 'sea_dir': (90, 180)},
        'Santa Susanna': {'lat': 41.6366, 'lon': 2.7098, 'sea_dir': (90, 180)},
        'Vilassar de Dalt': {'lat': 41.5167, 'lon': 2.3583, 'sea_dir': None},
        'Vilassar de Mar': {'lat': 41.5057, 'lon': 2.3920, 'sea_dir': (90, 180)},
    },
    "Montsià": {
        'Alcanar': {'lat': 40.5434, 'lon': 0.4820, 'sea_dir': (60, 160)},
        'Amposta': {'lat': 40.7093, 'lon': 0.5810, 'sea_dir': (70, 170)},
        'La Sénia': {'lat': 40.6322, 'lon': 0.2831, 'sea_dir': None},
    },
    "Noguera": {
        'Agramunt': {'lat': 41.7871, 'lon': 1.0967, 'sea_dir': None},
        'Balaguer': {'lat': 41.7904, 'lon': 0.8066, 'sea_dir': None},
        'Camarasa': {'lat': 41.8753, 'lon': 0.8804, 'sea_dir': None},
    },
    "Osona": {
        'Centelles': {'lat': 41.7963, 'lon': 2.2203, 'sea_dir': None},
        'Manlleu': {'lat': 42.0016, 'lon': 2.2844, 'sea_dir': None},
        'Vic': {'lat': 41.9301, 'lon': 2.2545, 'sea_dir': None},
        'Vidrà': {'lat': 42.1226, 'lon': 2.3116, 'sea_dir': None},
    },
    "Pallars Jussà": {
        'La Pobla de Segur': {'lat': 42.2472, 'lon': 0.9678, 'sea_dir': None},
        'Tremp': {'lat': 42.1664, 'lon': 0.8953, 'sea_dir': None},
    },
    "Pallars Sobirà": {
        'Sarroca de Bellera': {'lat': 42.3957, 'lon': 0.8656, 'sea_dir': None},
        'Sort': {'lat': 42.4131, 'lon': 1.1278, 'sea_dir': None},
    },
    "Pla de l'Estany": {
        'Banyoles': {'lat': 42.1197, 'lon': 2.7667, 'sea_dir': (80, 170)},
    },
    "Ripollès": {
        'Ripoll': {'lat': 42.2013, 'lon': 2.1903, 'sea_dir': None},
        'Sant Joan de les Abadesses': {'lat': 42.2355, 'lon': 2.2858, 'sea_dir': None},
    },
    "Segrià": {
        'Lleida': {'lat': 41.6177, 'lon': 0.6200, 'sea_dir': None},
        'Soses': {'lat': 41.5358, 'lon': 0.5186, 'sea_dir': None},
    },
    "Selva": {
        'Arbúcies': {'lat': 41.8159, 'lon': 2.5152, 'sea_dir': None},
        'Blanes': {'lat': 41.6748, 'lon': 2.7917, 'sea_dir': (80, 180)},
        'Hostalric': {'lat': 41.7479, 'lon': 2.6360, 'sea_dir': None},
        'Lloret de Mar': {'lat': 41.7005, 'lon': 2.8450, 'sea_dir': (80, 180)},
        'Santa Coloma de Farners': {'lat': 41.8596, 'lon': 2.6703, 'sea_dir': None},
        'Vidreres': {'lat': 41.7876, 'lon': 2.7788, 'sea_dir': (80, 180)},
    },
    "Solsonès": {
        'Solsona': {'lat': 41.9942, 'lon': 1.5161, 'sea_dir': None},
    },
    "Tarragonès": {
        'Altafulla': {'lat': 41.1417, 'lon': 1.3750, 'sea_dir': (110, 220)},
        'Salou': {'lat': 41.0763, 'lon': 1.1417, 'sea_dir': (110, 220)},
        'Tarragona': {'lat': 41.1189, 'lon': 1.2445, 'sea_dir': (110, 220)},
    },
    "Val d'Aran": {
        'Vielha': {'lat': 42.7027, 'lon': 0.7966, 'sea_dir': None},
    },
    "Vallès Occidental": {
        'Castellar del Vallès': {'lat': 41.6186, 'lon': 2.0875, 'sea_dir': None},
        'Cerdanyola del Vallès': {'lat': 41.4925, 'lon': 2.1415, 'sea_dir': (100, 200)},
        'Montcada i Reixac': {'lat': 41.4851, 'lon': 2.1884, 'sea_dir': (100, 200)},
        'Rubí': {'lat': 41.4936, 'lon': 2.0323, 'sea_dir': (100, 200)},
        'Sabadell': {'lat': 41.5483, 'lon': 2.1075, 'sea_dir': (100, 200)},
        'Sant Cugat del Vallès': {'lat': 41.4727, 'lon': 2.0863, 'sea_dir': (100, 200)},
        'Sant Quirze del Vallès': {'lat': 41.5303, 'lon': 2.0831, 'sea_dir': None},
        'Terrassa': {'lat': 41.5615, 'lon': 2.0084, 'sea_dir': (100, 200)},
    },
    "Vallès Oriental": {
        'Caldes de Montbui': {'lat': 41.6315, 'lon': 2.1678, 'sea_dir': None},
        'Cardedeu': {'lat': 41.6403, 'lon': 2.3582, 'sea_dir': (90, 180)},
        'Granollers': {'lat': 41.6083, 'lon': 2.2886, 'sea_dir': (90, 180)},
        'Mollet del Vallès': {'lat': 41.5385, 'lon': 2.2144, 'sea_dir': (100, 200)},
    },
}

# Genera una llista plana a partir de la nova estructura per compatibilitat
CIUTATS_CATALUNYA = {
    ciutat: dades 
    for comarca in CIUTATS_PER_COMARCA.values() 
    for ciutat, dades in comarca.items()
}

# Afegeix els punts marins manualment
PUNTS_MAR = {
    'Costes de Girona (Mar)':   {'lat': 42.05, 'lon': 3.30, 'sea_dir': (0, 360)},
    'Litoral Barceloní (Mar)': {'lat': 41.40, 'lon': 2.90, 'sea_dir': (0, 360)},
    'Aigües de Tarragona (Mar)': {'lat': 40.90, 'lon': 2.00, 'sea_dir': (0, 360)},
}
CIUTATS_CATALUNYA.update(PUNTS_MAR)

# Defineix les llistes necessàries a partir de la principal
POBLACIONS_TERRA = {k: v for k, v in CIUTATS_CATALUNYA.items() if '(Mar)' not in k}
CIUTATS_CONVIDAT = {
    'Barcelona': CIUTATS_CATALUNYA['Barcelona'], 'Girona': CIUTATS_CATALUNYA['Girona'],
    'Lleida': CIUTATS_CATALUNYA['Lleida'], 'Tarragona': CIUTATS_CATALUNYA['Tarragona']
}

# --- Constants per Tornado Alley ---
API_URL_USA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_USA = pytz.timezone('America/Chicago')
USA_CITIES = {
    'Dallas, TX': {'lat': 32.7767, 'lon': -96.7970},
    'Wichita, KS': {'lat': 37.6872, 'lon': -97.3301},
    'Houston, TX': {'lat': 29.7604, 'lon': -95.3698},
    'Kansas City, MO': {'lat': 39.0997, 'lon': -94.5786},
}
MAP_EXTENT_USA = [-105, -85, 28, 48]
PRESS_LEVELS_HRRR = sorted([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650, 
    625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375, 350, 325, 300, 275, 
    250, 225, 200, 175, 150, 125, 100
], reverse=True)

# --- Constants Generals ---
USERS_FILE = 'users.json'
CHAT_FILE = 'chat_history.json'
MAX_IA_REQUESTS = 5              
TIME_WINDOW_SECONDS = 3 * 60 * 60 
RATE_LIMIT_FILE = 'rate_limits.json'

THRESHOLDS_GLOBALS = {
    'SBCAPE': (500, 1500, 2500), 'MUCAPE': (500, 1500, 2500), 
    'MLCAPE': (250, 1000, 2000), 'CAPE_0-3km': (75, 150, 250), 
    'SBCIN': (-25, -75, -150), 'MUCIN': (-25, -75, -150),
    'MLCIN': (-25, -75, -150), 'LI': (-2, -5, -8), 
    'PWAT': (30, 40, 50), 
    'BWD_0-6km': (20, 30, 40), 
    'BWD_0-1km': (15, 25, 35),
    'SRH_0-1km': (100, 150, 250), 
    'SRH_0-3km': (150, 250, 400),
    
    # --- NOUS LLINDARS AFEGITS ---
    # Per a LCL/LFC, els valors s'interpreten de manera inversa (més baix és pitjor)
    'LCL_Hgt': (1000, 1500), # <1000m (Vermell), 1000-1500m (Verd), >1500m (Gris)
    'LFC_Hgt': (1500, 2500), # <1500m (Vermell), 1500-2500m (Verd), >2500m (Gris)
    
    # Per a UPDRAFT, valors més alts són pitjors
    'MAX_UPDRAFT': (25, 40, 55) # >25m/s (Groc), >40m/s (Taronja), >55m/s (Vermell)
}


def graus_a_direccio_cardinal(graus):
    """Converteix un valor en graus a la seva direcció cardinal (N, NNE, NE, etc.)."""
    if pd.isna(graus):
        return "N/A"
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    index = int(round(graus / 22.5)) % 16
    return dirs[index]

def get_color_global(value, param_key, reverse_colors=False):
    """
    Versió Definitiva v2.0.
    Inclou una lògica especial per a LCL i LFC, on els valors baixos són vermells.
    """
    if pd.isna(value): return "#808080"

    thresholds = THRESHOLDS_GLOBALS.get(param_key, [])
    if not thresholds: return "#FFFFFF"

    # --- LÒGICA ESPECIAL PER A LCL i LFC ---
    if param_key in ['LCL_Hgt', 'LFC_Hgt']:
        # Aquests paràmetres només tenen 2 llindars per a 3 colors
        if len(thresholds) != 2: return "#FFFFFF"
        
        if value < thresholds[0]: return "#dc3545"  # Vermell (Perillós)
        if value < thresholds[1]: return "#2ca02c"  # Verd (Normal)
        return "#808080"                         # Gris (Inhibidor)
    # --- FI DE LA LÒGICA ESPECIAL ---
    
    # Lògica per a la resta de paràmetres (que tenen 3 llindars)
    if len(thresholds) != 3: return "#FFFFFF"

    colors = ["#2ca02c", "#ffc107", "#fd7e14", "#dc3545"] # Verd, Groc, Taronja, Vermell
    
    if reverse_colors: # Per a CIN i LI
        if value < thresholds[2]: return colors[3]
        if value < thresholds[1]: return colors[2]
        if value < thresholds[0]: return colors[1]
        return colors[0] # Aquí el verd és el color per a valors "segurs"
    
    # Lògica normal per a CAPE, BWD, SRH, UPDRAFT, etc.
    if value >= thresholds[2]: return colors[3]
    elif value >= thresholds[1]: return colors[2]
    elif value >= thresholds[0]: return colors[1]
    else: return colors[0]
        
def get_hashed_password(password): return hashlib.sha256(password.encode()).hexdigest()
def load_json_file(filename):
    if not os.path.exists(filename): return {} if 'users' in filename or 'rate' in filename else []
    try:
        with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {} if 'users' in filename or 'rate' in filename else []
def save_json_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
def load_and_clean_chat_history():
    if not os.path.exists(CHAT_FILE): return []
    try:
        with open(CHAT_FILE, 'r', encoding='utf-8') as f: history = json.load(f)
        one_hour_ago_ts = datetime.now(pytz.utc).timestamp() - 3600
        cleaned_history = [msg for msg in history if msg.get('timestamp', 0) > one_hour_ago_ts]
        if len(cleaned_history) < len(history): save_json_file(cleaned_history, CHAT_FILE)
        return cleaned_history
    except (json.JSONDecodeError, FileNotFoundError): return []
def count_unread_messages(history):
    last_seen = st.session_state.get('last_seen_timestamp', 0); current_user = st.session_state.get('username')
    return sum(1 for msg in history if msg['timestamp'] > last_seen and msg['username'] != current_user)




def generar_html_imatge_estatica(image_path, height="180px"):
    """
    Crea el codi HTML per mostrar una imatge estàtica amb un efecte de zoom
    en passar el ratolí per sobre (hover).
    """
    if not os.path.exists(image_path):
        return f"<p style='color: red;'>Imatge no trobada: {os.path.basename(image_path)}</p>"

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    
    file_extension = os.path.splitext(image_path)[1].lower().replace('.', '')
    mime_type = f"image/{file_extension}"

    # <<<--- CSS PER A L'EFECTE HOVER DIRECTAMENT A LA IMATGE ---
    # Definim l'estil directament aquí per a més simplicitat.
    html_code = f"""
    <style>
        .hover-image-container {{
            overflow: hidden; /* Molt important per a que l'efecte de zoom no se surti del quadre */
            border-radius: 10px;
            height: {height};
            margin-bottom: 10px;
        }}
        .hover-image {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease-in-out; /* L'animació suau de l'escalat */
        }}
        .hover-image-container:hover .hover-image {{
            transform: scale(1.1); /* La imatge es fa un 10% més gran */
        }}
    </style>

    <div class="hover-image-container">
        <img src="data:{mime_type};base64,{image_b64}" class="hover-image" alt="Previsualització de la zona">
    </div>
    """
    return html_code



def afegir_slideshow_de_fons():
    """
    Crea un slideshow de fons amb 5 imatges que es van alternant amb una
    transició suau. Les imatges es codifiquen en Base64 per ser incrustades
    directament a l'HTML.
    """
    # Llista de les imatges que vols utilitzar. Assegura't que existeixen!
    image_files = [
        "fons1.jpg", "fons2.jpg", "fons3.jpg", "fons4.jpg", "fons5.jpg"
    ]
    
    # Temps (en segons) que cada imatge estarà visible i la durada de la transició
    hold_time = 8  # Segons que la imatge és visible
    fade_time = 2  # Segons que dura el "cross-fade"
    
    # Càlculs per a l'animació CSS
    total_time_per_image = hold_time + fade_time
    animation_duration = len(image_files) * total_time_per_image
    fade_percentage = (fade_time / animation_duration) * 100

    # Generem les regles CSS per a cada imatge
    css_rules = ""
    for i, image_file in enumerate(image_files):
        if os.path.exists(image_file):
            with open(image_file, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            
            # Calculem el retard de l'animació per a cada imatge
            delay = i * total_time_per_image
            
            css_rules += f"""
                .slideshow-image:nth-child({i + 1}) {{
                    background-image: url("data:image/jpeg;base64,{image_b64}");
                    animation-delay: {delay}s;
                }}
            """

    slideshow_html = f"""
    <style>
    /* Estil per fer l'app transparent i que es vegi el fons */
    .stApp {{
        background-color: transparent;
    }}

    /* Contenidor principal del slideshow */
    #slideshow-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1; /* El posem al fons de tot */
    }}

    /* Estil comú per a totes les imatges del slideshow */
    .slideshow-image {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-size: cover;
        background-position: center;
        opacity: 0; /* Comencen invisibles */
        animation-name: fade;
        animation-duration: {animation_duration}s;
        animation-iteration-count: infinite; /* L'animació es repeteix indefinidament */
    }}

    /* Les regles específiques per a cada imatge (amb el seu fons i retard) */
    {css_rules}

    /* L'animació de fade in/out */
    @keyframes fade {{
        0% {{ opacity: 0; }}
        {fade_percentage}% {{ opacity: 1; }} /* Fade in */
        {(100 / len(image_files)) - fade_percentage}% {{ opacity: 1; }} /* Hold */
        {100 / len(image_files)}% {{ opacity: 0; }} /* Fade out */
        100% {{ opacity: 0; }}
    }}
    </style>
    
    <div id="slideshow-container">
        <div class="slideshow-image"></div>
        <div class="slideshow-image"></div>
        <div class="slideshow-image"></div>
        <div class="slideshow-image"></div>
        <div class="slideshow-image"></div>
    </div>
    """
    st.markdown(slideshow_html, unsafe_allow_html=True)
    
    
def inject_custom_css():
    st.markdown("""
    <style>
    /* --- ESTIL DEFINITIU I ROBUST PER A TOTS ELS SPINNERS --- */
    /* Aquesta regla s'aplica a qualsevol spinner, en qualsevol lloc de l'app */
    .stSpinner {
        position: fixed; /* Posició fixa respecte a la finestra del navegador */
        top: 0;
        left: 0;
        width: 100%;     /* Ocupa tota l'amplada */
        height: 100%;    /* Ocupa tota l'alçada */
        background-color: rgba(0, 0, 0, 0.7); /* Fons fosc semitransparent */
        z-index: 9999;   /* Assegura que estigui per sobre de tot */
        
        /* Centrat perfecte amb Flexbox */
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Estil per al contingut intern (la icona i el text) */
    .stSpinner > div {
        text-align: center;
        color: white;         /* Text en blanc per a més contrast */
        font-size: 1.2rem;    /* Mida del text una mica més gran */
        font-weight: bold;
    }
    /* --- FI DE L'ESTIL DEL SPINNER --- */
    

    /* --- ESTIL DE L'ALERTA PARPELLEJANT (ES MANTÉ) --- */
    .blinking-alert {
        animation: blink 1.5s linear infinite;
    }

    @keyframes blink {
        50% { opacity: 0.6; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    
def format_time_left(time_delta):
    total_seconds = int(time_delta.total_seconds()); hours, remainder = divmod(total_seconds, 3600); minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}min" if hours > 0 else f"{minutes} min"

def show_login_page():
    st.markdown("<h1 style='text-align: center;'>Tempestes.cat</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # <<<--- ELIMINATS ELS FORMULARIS DE LOGIN I REGISTRE --->>>
    # Ara anem directament a les opcions d'accés ràpid

    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Selecciona un mètode d'accés</p>", unsafe_allow_html=True)

    # Botó principal per entrar com a Convidat
    if st.button("Entrar a la Terminal (Accés General)", use_container_width=True, type="primary"):
        st.session_state['zone_selected'] = None
        # Definim 'guest_mode' per si en el futur hi ha funcionalitats restringides
        st.session_state.update({'guest_mode': True, 'logged_in': True})
        st.rerun()
    
    # Separador per al mode desenvolupador
    st.divider()
    st.markdown("<p style='text-align: center;'>Accés per a desenvolupadors</p>", unsafe_allow_html=True)
    
    # Formulario para modo desarrollador
    with st.form("developer_form"):
        dev_password = st.text_input("Contrasenya de desenvolupador", type="password", key="dev_pass")
        
        if st.form_submit_button("🚀 Accés Mode Desenvolupador", use_container_width=True):
            # Assegura't de tenir aquesta clau als teus secrets de Streamlit
            if dev_password == st.secrets["app_secrets"]["moderator_password"]:
                st.session_state['zone_selected'] = None
                st.session_state.update({
                    'logged_in': True, 
                    'developer_mode': True,
                    'username': 'Desenvolupador', # Assignem un nom per consistència
                    'guest_mode': False
                })
                st.success("Mode desenvolupador activat!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Contrasenya de desenvolupador incorrecta.")

def calcular_mlcape_robusta(p, T, Td):
    """
    Una funció manual i extremadament robusta per calcular el MLCAPE i MLCIN.
    Aquesta funció està dissenyada per no fallar mai, fins i tot amb sondejos "difícils".
    """
    try:
        # 1. Defineix la capa de barreja (els primers 100 hPa)
        p_sfc = p[0]
        p_bottom = p_sfc - 100 * units.hPa
        mask = (p >= p_bottom) & (p <= p_sfc)

        # Si no hi ha punts a la capa, fem servir només la superfície (Pla B)
        if not np.any(mask):
            p_mixed, T_mixed, Td_mixed = p[0], T[0], Td[0]
        else:
            # 2. Calcula les condicions mitjanes de la capa
            p_layer, T_layer, Td_layer = p[mask], T[mask], Td[mask]
            
            # Per al punt de partida, necessitem la temperatura potencial i la ratio de barreja mitjanes
            theta_mixed = np.mean(mpcalc.potential_temperature(p_layer, T_layer))
            mixing_ratio_mixed = np.mean(mpcalc.mixing_ratio_from_relative_humidity(p_layer, np.ones_like(p_layer) * 100 * units.percent, Td_layer))
            
            # A partir d'aquests valors mitjans, trobem la T i Td a la pressió de superfície
            T_mixed = mpcalc.temperature_from_potential_temperature(p_sfc, theta_mixed)
            Td_mixed = mpcalc.dewpoint_from_mixing_ratio(p_sfc, mixing_ratio_mixed)
        
        # 3. Puja la nova parcel·la mitjana
        prof_mixed = mpcalc.parcel_profile(p, T_mixed, Td_mixed).to('degC')
        
        # 4. Calcula el CAPE/CIN a partir d'aquesta trajectòria robusta
        mlcape, mlcin = mpcalc.cape_cin(p, T, Td, prof_mixed)
        
        return float(mlcape.m), float(mlcin.m)

    except Exception:
        # Pla C: Si tot falla, retornem NaN. Això gairebé mai hauria de passar.
        return np.nan, np.nan
        




def processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile):
    """
    Versió Original, Robusta i Simplificada:
    Neteja el perfil de dades eliminant qualsevol nivell on falti una dada essencial.
    Això garanteix la màxima compatibilitat amb tots els càlculs de MetPy i el dibuix.
    """
    if len(p_profile) < 4:
        return None, "Perfil atmosfèric massa curt."

    # 1. Converteix les llistes a arrays de MetPy
    p = np.array(p_profile) * units.hPa
    T = np.array(T_profile) * units.degC
    Td = np.array(Td_profile) * units.degC
    u = np.array(u_profile) * units('m/s')
    v = np.array(v_profile) * units('m/s')
    heights = np.array(h_profile) * units.meter

    # 2. Neteja de dades estricta: crea una màscara booleana
    #    Un nivell només és vàlid si Totes les dades essencials no són NaN.
    valid_mask = np.isfinite(p.m) & np.isfinite(T.m) & np.isfinite(Td.m) & np.isfinite(u.m) & np.isfinite(v.m)
    
    # 3. Aplica la màscara a tots els perfils
    p, T, Td, u, v, heights = p[valid_mask], T[valid_mask], Td[valid_mask], u[valid_mask], v[valid_mask], heights[valid_mask]

    if len(p) < 3:
        return None, "No hi ha prou dades vàlides després de la neteja."

    # 4. Ordena el perfil per pressió (de major a menor)
    sort_idx = np.argsort(p.m)[::-1]
    p, T, Td, u, v, heights = p[sort_idx], T[sort_idx], Td[sort_idx], u[sort_idx], v[sort_idx], heights[sort_idx]
    
    # A partir d'aquí, tot el codi de càlculs es manté igual,
    # ja que treballa amb dades garantides de ser netes i sincronitzades.
    params_calc = {}
    heights_agl = heights - heights[0]

    with parcel_lock:
        sfc_prof, ml_prof = None, None
        try: 
            sfc_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        except Exception: 
            return None, "Error crític: No s'ha pogut calcular ni el perfil de superfície."
        
        try: 
            _, _, _, ml_prof = mpcalc.mixed_parcel(p, T, Td, depth=100 * units.hPa)
        except Exception: 
            ml_prof = None
            
        main_prof = ml_prof if ml_prof is not None else sfc_prof

        try: 
            rh = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100
            params_calc['RH_CAPES'] = {
                'baixa': np.mean(rh[(p.m <= 1000) & (p.m > 850)]), 
                'mitjana': np.mean(rh[(p.m <= 850) & (p.m > 500)]), 
                'alta': np.mean(rh[(p.m <= 500) & (p.m > 250)])
            }
        except: 
            params_calc['RH_CAPES'] = {'baixa': np.nan, 'mitjana': np.nan, 'alta': np.nan}

        try: 
            params_calc['PWAT'] = float(mpcalc.precipitable_water(p, Td).to('mm').m)
        except: 
            params_calc['PWAT'] = np.nan

        try:
            _, fl_h = mpcalc.freezing_level(p, T, heights)
            params_calc['FREEZING_LVL_HGT'] = float(fl_h[0].to('m').m)
        except: 
            params_calc['FREEZING_LVL_HGT'] = np.nan

        try:
            p_numeric, T_numeric = p.m, T.m
            if len(p_numeric) >= 2 and p_numeric.min() <= 500 <= p_numeric.max():
                params_calc['T_500hPa'] = float(np.interp(500, p_numeric[::-1], T_numeric[::-1]))
            else: 
                params_calc['T_500hPa'] = np.nan
        except: 
            params_calc['T_500hPa'] = np.nan

        if sfc_prof is not None:
            try:
                sbcape, sbcin = mpcalc.cape_cin(p, T, Td, sfc_prof)
                params_calc['SBCAPE'] = float(sbcape.m); params_calc['SBCIN'] = float(sbcin.m)
                params_calc['MAX_UPDRAFT'] = np.sqrt(2 * float(sbcape.m)) if sbcape.m > 0 else 0.0
            except: 
                params_calc.update({'SBCAPE': np.nan, 'SBCIN': np.nan, 'MAX_UPDRAFT': np.nan})

        if ml_prof is not None:
            try:
                mlcape, mlcin = mpcalc.cape_cin(p, T, Td, ml_prof)
                params_calc['MLCAPE'] = float(mlcape.m); params_calc['MLCIN'] = float(mlcin.m)
            except: 
                params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
        else:
            params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})

        if main_prof is not None:
            try: 
                params_calc['LI'] = float(mpcalc.lifted_index(p, T, main_prof).m)
            except: 
                params_calc['LI'] = np.nan
            try:
                lfc_p, _ = mpcalc.lfc(p, T, Td, main_prof)
                params_calc['LFC_p'] = float(lfc_p.m)
                params_calc['LFC_Hgt'] = float(np.interp(lfc_p.m, p.m[::-1], heights_agl.m[::-1]))
            except: 
                params_calc.update({'LFC_p': np.nan, 'LFC_Hgt': np.nan})
            try:
                el_p, _ = mpcalc.el(p, T, Td, main_prof)
                params_calc['EL_p'] = float(el_p.m)
                params_calc['EL_Hgt'] = float(np.interp(el_p.m, p.m[::-1], heights_agl.m[::-1]))
            except: 
                params_calc.update({'EL_p': np.nan, 'EL_Hgt': np.nan})
            try:
                idx_3km = np.argmin(np.abs(heights_agl.m - 3000))
                cape_0_3, _ = mpcalc.cape_cin(p[:idx_3km+1], T[:idx_3km+1], Td[:idx_3km+1], main_prof[:idx_3km+1])
                params_calc['CAPE_0-3km'] = float(cape_0_3.m)
            except: 
                params_calc['CAPE_0-3km'] = np.nan

        try:
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td)
            params_calc['MUCAPE'] = float(mucape.m); params_calc['MUCIN'] = float(mucin.m)
        except: 
            params_calc.update({'MUCAPE': np.nan, 'MUCIN': np.nan})

        try:
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0])
            params_calc['LCL_p'] = float(lcl_p.m)
            params_calc['LCL_Hgt'] = float(np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1]))
        except: 
            params_calc.update({'LCL_p': np.nan, 'LCL_Hgt': np.nan})

        try:
            for name, depth_m in [('0-1km', 1000), ('0-6km', 6000)]:
                bwd_u, bwd_v = mpcalc.bulk_shear(p, u, v, height=heights, depth=depth_m * units.meter)
                params_calc[f'BWD_{name}'] = float(mpcalc.wind_speed(bwd_u, bwd_v).to('kt').m)
        except: 
            params_calc.update({'BWD_0-1km': np.nan, 'BWD_0-6km': np.nan})
        
        try:
            rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p, u, v, heights)
            params_calc['RM'] = (float(rm[0].m), float(rm[1].m))
            params_calc['LM'] = (float(lm[0].m), float(lm[1].m))
            params_calc['Mean_Wind'] = (float(mean_wind[0].m), float(mean_wind[1].m))
        except Exception:
            params_calc.update({'RM': (np.nan, np.nan), 'LM': (np.nan, np.nan), 'Mean_Wind': (np.nan, np.nan)})

        if params_calc.get('RM') and not np.isnan(params_calc['RM'][0]):
            u_storm, v_storm = params_calc['RM'][0] * units('m/s'), params_calc['RM'][1] * units('m/s')
            try:
                for name, depth_m in [('0-1km', 1000), ('0-3km', 3000)]:
                    srh = mpcalc.storm_relative_helicity(heights, u, v, depth=depth_m * units.meter, storm_u=u_storm, storm_v=v_storm)[0]
                    params_calc[f'SRH_{name}'] = float(srh.m)
            except: 
                params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})
        
    return ((p, T, Td, u, v, heights, sfc_prof), params_calc), None

def diagnosticar_potencial_tempesta(params):
    """
    Sistema de Diagnòstic Meteorològic Expert v10.0.
    Integra la Convergència (Disparador), LI (Inestabilitat) i EL (Profunditat)
    per a un diagnòstic complet del cicle de vida de la tempesta.
    """
    # --- 1. EXTRACCIÓ ROBUSTA DE TOTS ELS PARÀMETRES ---
    sbcape = params.get('SBCAPE', 0) or 0; mucape = params.get('MUCAPE', 0) or 0
    sbcin = params.get('SBCIN', 0) or 0; mucin = params.get('MUCIN', 0) or 0
    max_cape = max(sbcape, mucape)
    strongest_cin = min(sbcin, mucin)

    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_1km = params.get('SRH_0-1km', 0) or 0
    lcl_hgt = params.get('LCL_Hgt', 9999) or 9999
    lfc_hgt = params.get('LFC_Hgt', 9999) or 9999
    dcape = params.get('DCAPE', 0) or 0
    pwat = params.get('PWAT', 0) or 0
    
    # --- NOUS PARÀMETRES CLAU ---
    li_index = params.get('LI', 5) or 5
    el_hgt = params.get('EL_Hgt', 0) or 0
    
    # --- BLOC DE DEFENSA ANTI-ERRORS (CORRECCIÓ) ---
    conv_key = next((k for k in params if k.startswith('CONV_')), None)
    raw_conv_value = params.get(conv_key, 0)
    convergencia = raw_conv_value if isinstance(raw_conv_value, (int, float, np.number)) else 0
    # ----------------------------------------------

    # --- 2. AVALUACIÓ DEL POTENCIAL DE DISPAR (CONVERGÈNCIA vs CIN) ---
    # Aquesta és la primera batalla: pot la guspira encendre el combustible?
    potencial_dispar = False
    # Aquesta línia ara és segura
    if convergencia >= 15 and strongest_cin > -75: # Convergència moderada pot trencar una tapa feble
        potencial_dispar = True
    elif convergencia >= 30 and strongest_cin > -125: # Convergència forta pot trencar una tapa moderada
        potencial_dispar = True
    elif strongest_cin > -25: # Si gairebé no hi ha tapa, qualsevol cosa pot iniciar la tempesta
        potencial_dispar = True

    # --- BLOC DE VETO PRINCIPAL ---
    # Si no hi ha combustible O no hi ha manera d'iniciar la convecció, no hi ha tempesta.
    if max_cape < 500 or not potencial_dispar:
        tipus_tempesta = "Inhibició / Sense Dispar" if max_cape >= 500 else "Sense Energia"
        color_tempesta = "#808080"
        base_nuvol = "Atmosfera Estable"
        color_base = "#808080"
        return tipus_tempesta, color_tempesta, base_nuvol, color_base

    # --- 3. SI HI HA ENERGIA I DISPAR, CLASSIFIQUEM LA TEMPESTA ---
    tipus_tempesta = "Cèl·lula Simple"; color_tempesta = "#2ca02c"

    # Lògica d'Organització (Cisallament)
    if bwd_6km >= 35:
        # Refinem el diagnòstic de supercèl·lula amb LI i EL
        if li_index < -6 and el_hgt > 12000:
            tipus_tempesta = "Supercèl·lula (Pot. Sever)"
        else:
            tipus_tempesta = "Supercèl·lula"
        color_tempesta = "#dc3545"
    elif bwd_6km >= 20:
        if dcape > 1000:
            tipus_tempesta = "Línia Multicel·lular (Pot. Esclafits)"
            color_tempesta = "#fd7e14"
        else:
            tipus_tempesta = "Grup Multicel·lular Organitzat"
            color_tempesta = "#ffc107"
    else: # Cisallament baix
        if max_cape > 2500 and li_index < -5:
            tipus_tempesta = "Cèl·lula de Pols (Pot. Calamarsa)"
            color_tempesta = "#ffc107"
        else:
            tipus_tempesta = "Cèl·lula Simple"
            color_tempesta = "#2ca02c"

    # --- 4. LÒGICA DE LA BASE DEL NÚVOL (SENSE CANVIS) ---
    base_nuvol = "Plana i Alta"; color_base = "#2ca02c"
    
    if srh_1km >= 250 and lcl_hgt < 1000 and lfc_hgt < 1500:
        base_nuvol = "Tornàdica (Potencial Alt)"; color_base = "#dc3545"
    elif srh_1km >= 150 and lcl_hgt < 1200:
        base_nuvol = "Rotatòria Forta (Wall Cloud)"; color_base = "#fd7e14"
    elif srh_1km >= 100:
        base_nuvol = "Rotatòria (Inflow)"; color_base = "#ffc107"
        
    return tipus_tempesta, color_tempesta, base_nuvol, color_base



def debug_map_data(map_data):
    """Función para depurar los datos del mapa"""
    if not map_data:
        print("Map data is None")
        return
        
    print("Keys in map_data:", list(map_data.keys()))
    if 'lons' in map_data:
        print("Number of points:", len(map_data['lons']))
    if 'sfc_temp_data' in map_data:
        print("Temperature data sample:", map_data['sfc_temp_data'][:5])
    if 'sfc_dewpoint_data' in map_data:
        print("Dewpoint data sample:", map_data['sfc_dewpoint_data'][:5])
    

def debug_calculos(p, T, Td, u, v, heights, prof):
    """Función para depurar los cálculos problemáticos"""
    print("=== DEBUG: Cálculos problemáticos ===")
    
    # Debug LI
    try:
        li = mpcalc.lifted_index(p, T, prof)
        print(f"LI raw: {li}")
        print(f"LI type: {type(li)}")
        if hasattr(li, 'm'): print(f"LI.m: {li.m}")
        if hasattr(li, 'magnitude'): print(f"LI.magnitude: {li.magnitude}")
    except Exception as e:
        print(f"LI error: {e}")
    
    # Debug DCAPE
    try:
        dcape = mpcalc.dcape(p, T, Td)
        print(f"DCAPE raw: {dcape}")
        print(f"DCAPE type: {type(dcape)}")
        if hasattr(dcape, 'm'): print(f"DCAPE.m: {dcape.m}")
        if hasattr(dcape, 'magnitude'): print(f"DCAPE.magnitude: {dcape.magnitude}")
    except Exception as e:
        print(f"DCAPE error: {e}")
    
    # Debug SRH (necesita movimiento de tormenta)
    try:
        rm, _, _ = mpcalc.bunkers_storm_motion(p, u, v, heights)
        u_storm, v_storm = rm[0] * units('m/s'), rm[1] * units('m/s')
        srh = mpcalc.storm_relative_helicity(heights, u, v, depth=1000 * units.meter, 
                                           storm_u=u_storm, storm_v=v_storm)
        print(f"SRH raw: {srh}")
        print(f"SRH type: {type(srh)}")
        if hasattr(srh, 'm'): print(f"SRH.m: {srh.m}")
        if hasattr(srh, 'magnitude'): print(f"SRH.magnitude: {srh.magnitude}")
    except Exception as e:
        print(f"SRH error: {e}")
    
    print("=====================================")




    
def crear_mapa_base(map_extent, projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, subplot_kw={'projection': projection})
    ax.set_extent(map_extent, crs=ccrs.PlateCarree()) 
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    if projection != ccrs.PlateCarree():
        ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray', zorder=5)
    return fig, ax


def verificar_datos_entrada(p, T, Td, u, v, heights):
    """Verificar que los datos de entrada son válidos"""
    print("=== VERIFICACIÓN DE DATOS ===")
    print(f"Presión: {p.m[:5]}... (len: {len(p)})")
    print(f"Temperatura: {T.m[:5]}... (len: {len(T)})")
    print(f"Punto rocío: {Td.m[:5]}... (len: {len(Td)})")
    print(f"Alturas: {heights.m[:5]}... (len: {len(heights)})")
    
    # Verificar que tenemos datos suficientes para cálculos
    if len(p) < 10:
        print("ADVERTENCIA: Muy pocos niveles para cálculos precisos")
    
    # Verificar rango de temperaturas
    if np.max(T.m) < -20 or np.min(T.m) > 50:
        print("ADVERTENCIA: Temperaturas fuera de rango normal")
    
    print("=============================")




def crear_skewt(p, T, Td, u, v, prof, params_calc, titol, timestamp_str, zoom_capa_baixa=False):
    """
    Versió Definitiva v2.0: Soluciona el bug de l'ombra de CAPE/CIN desplaçada
    netejant les dades just abans de dibuixar l'ombrejat.
    """
    fig = plt.figure(dpi=150, figsize=(7, 8))
    
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.05, 0.85, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)

    if zoom_capa_baixa:
        pressio_superficie = p[0].m
        skew.ax.set_ylim(pressio_superficie + 5, 800)
        mask_capa_baixa = (p.m <= pressio_superficie) & (p.m >= 800)
        T_capa_baixa = T[mask_capa_baixa]; Td_capa_baixa = Td[mask_capa_baixa]
        temp_min = min(T_capa_baixa.min().m, Td_capa_baixa.min().m) - 5
        temp_max = max(T_capa_baixa.max().m, Td_capa_baixa.max().m) + 5
        skew.ax.set_xlim(temp_min, temp_max)
    else:
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 40)
        
        pressio_superficie = p[0].m
        if pressio_superficie < 995:
            colors = ["#66462F", "#799845"] 
            cmap_terreny = LinearSegmentedColormap.from_list("terreny_cmap", colors)
            gradient = np.linspace(0, 1, 256).reshape(-1, 1)
            xlims = skew.ax.get_xlim()
            skew.ax.imshow(gradient.T, aspect='auto', cmap=cmap_terreny, origin='lower', extent=(xlims[0], xlims[1], 1000, pressio_superficie), alpha=0.6, zorder=0)

    skew.ax.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    skew.plot_dry_adiabats(color='coral', linestyle='--', alpha=0.5)
    skew.plot_moist_adiabats(color='cornflowerblue', linestyle='--', alpha=0.5)
    skew.plot_mixing_lines(color='limegreen', linestyle='--', alpha=0.5)
    
    # <<<--- CORRECCIÓ DE L'OMBRA DESPLAÇADA ---
    if prof is not None:
        # 1. Creem una màscara per a trobar només els nivells on TOTES les dades
        #    necessàries per a l'ombrejat (pressió, temperatura i perfil de la parcel·la) són vàlides.
        valid_shade_mask = np.isfinite(p.m) & np.isfinite(T.m) & np.isfinite(prof.m)
        
        # 2. Creem perfils "nets" utilitzant aquesta màscara.
        p_clean = p[valid_shade_mask]
        T_clean = T[valid_shade_mask]
        prof_clean = prof[valid_shade_mask]

        # 3. Utilitzem aquestes dades netes NOMÉS per a dibuixar les ombres.
        #    Això evita que els valors 'NaN' confonguin l'algorisme de rebliment.
        skew.shade_cape(p_clean, T_clean, prof_clean, color='red', alpha=0.2)
        skew.shade_cin(p_clean, T_clean, prof_clean, color='blue', alpha=0.2)
        
        # 4. Finalment, dibuixem la línia negra de la trajectòria utilitzant les dades originals,
        #    ja que la funció 'plot' sí que sap com gestionar els forats correctament.
        skew.plot(p, prof, 'k', linewidth=3, label='Trajectòria Parcel·la (SFC)', path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])
    # <<<--- FI DE LA CORRECCIÓ ---

    skew.plot(p, T, 'red', lw=2.5, label='Temperatura')
    skew.plot(p, Td, 'green', lw=2.5, label='Punt de Rosada')
        
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03)
    
    skew.ax.set_title(f"{titol}\n{timestamp_str}", weight='bold', fontsize=14, pad=15)
    skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")

    try:
        if 'LCL_p' in params_calc and pd.notna(params_calc['LCL_p']): skew.plot_lcl_line(color='blue', linestyle='--', linewidth=1.5)
        if 'LFC_p' in params_calc and pd.notna(params_calc['LFC_p']): skew.plot_lfc_line(color='green', linestyle='--', linewidth=1.5)
        if 'EL_p' in params_calc and pd.notna(params_calc['EL_p']): skew.plot_el_line(color='red', linestyle='--', linewidth=1.5)
    except Exception as e:
        print(f"Error dibuixant línies de nivell: {e}")

    skew.ax.legend()
    return fig


def crear_hodograf_avancat(p, u, v, heights, params_calc, titol, timestamp_str):
    """
    Versió definitiva i corregida. Mou la diagnosi qualitativa a un panell
    superior, eliminant el gràfic de barbes i netejant el panell dret.
    """
    fig = plt.figure(dpi=150, figsize=(8, 8))
    # Augmentem una mica l'espai per al panell superior
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 6], width_ratios=[1.5, 1], hspace=0.4, wspace=0.3)
    ax_top_panel = fig.add_subplot(gs[0, :]); ax_hodo = fig.add_subplot(gs[1, 0]); ax_params = fig.add_subplot(gs[1, 1])
    
    fig.suptitle(f"{titol}\n{timestamp_str}", weight='bold', fontsize=16)
    
    # --- NOU PANELL SUPERIOR DE DIAGNÒSTIC (REEMPLAÇA LES BARBES DE VENT) ---
    ax_top_panel.set_title("Diagnòstic de l'Estructura de la Tempesta", fontsize=12, weight='bold', pad=15)
    ax_top_panel.axis('off') # Amaguem els eixos

    # Obtenim la diagnosi
    tipus_tempesta, color_tempesta, base_nuvol, color_base = diagnosticar_potencial_tempesta(params_calc)

    # Dibuixem les dues caixes de diagnòstic al panell superior
    # Columna 1: Tipus de Tempesta
    ax_top_panel.text(0.25, 0.55, "Tipus de Tempesta", ha='center', va='center', fontsize=10, color='gray', transform=ax_top_panel.transAxes)
    ax_top_panel.text(0.25, 0.2, tipus_tempesta, ha='center', va='center', fontsize=14, weight='bold', color=color_tempesta, transform=ax_top_panel.transAxes,
                      bbox=dict(facecolor='white', alpha=0.1, boxstyle='round,pad=0.5'))

    # Columna 2: Potencial a la Base
    ax_top_panel.text(0.75, 0.55, "Potencial a la Base", ha='center', va='center', fontsize=10, color='gray', transform=ax_top_panel.transAxes)
    ax_top_panel.text(0.75, 0.2, base_nuvol, ha='center', va='center', fontsize=14, weight='bold', color=color_base, transform=ax_top_panel.transAxes,
                      bbox=dict(facecolor='white', alpha=0.1, boxstyle='round,pad=0.5'))

    # --- HODÒGRAF (Sense canvis) ---
    h = Hodograph(ax_hodo, component_range=80.); h.add_grid(increment=20, color='gray', linestyle='--')
    intervals = np.array([0, 1, 3, 6, 9, 12]) * units.km; colors_hodo = ['red', 'blue', 'green', 'purple', 'gold']
    h.plot_colormapped(u.to('kt'), v.to('kt'), heights, intervals=intervals, colors=colors_hodo, linewidth=2)
    ax_hodo.set_xlabel('U-Component (nusos)'); ax_hodo.set_ylabel('V-Component (nusos)')

    # --- PANELL DRET (NOMÉS VALORS NUMÈRICS) ---
    ax_params.axis('off')
    def degrees_to_cardinal_ca(d):
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        return dirs[int(round(d / 22.5)) % 16]

    y = 0.98
    # Moviment
    ax_params.text(0.5, y, "Moviment (dir/ km/h)", ha='center', weight='bold', fontsize=11); y-=0.15
    motion_data = {'M. Dret': params_calc.get('RM'), 'M. Esquerre': params_calc.get('LM'), 'Direcció del sistema': params_calc.get('Mean_Wind')}
    for name, vec in motion_data.items():
        val_str = "---"
        if vec and not pd.isna(vec[0]):
            u_motion, v_motion = vec[0] * units('m/s'), vec[1] * units('m/s')
            speed = mpcalc.wind_speed(u_motion, v_motion).to('km/h').m
            direction = mpcalc.wind_direction(u_motion, v_motion, convention='to').to('deg').m
            val_str = f"{degrees_to_cardinal_ca(direction)} / {speed:.0f}"
        ax_params.text(0, y, f"{name}:", ha='left', va='center', fontsize=10)
        ax_params.text(1, y, val_str, ha='right', va='center', fontsize=10)
        y -= 0.1
    
    y -= 0.08
    # Cisallament
    ax_params.text(0.5, y, "Cisallament (nusos)", ha='center', weight='bold', fontsize=11); y-=0.15
    for key, label in [('BWD_0-1km', '0-1 km'), ('BWD_0-6km', '0-6 km')]:
        val = params_calc.get(key, np.nan)
        color = get_color_global(val, key)
        ax_params.text(0, y, f"{label}:", ha='left', va='center', fontsize=10)
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color, fontsize=12)
        y -= 0.1

    y -= 0.08
    # Helicitat
    ax_params.text(0.5, y, "Helicitat (m²/s²)", ha='center', weight='bold', fontsize=11); y-=0.15
    for key, label in [('SRH_0-1km', '0-1 km'), ('SRH_0-3km', '0-3 km')]:
        val = params_calc.get(key, np.nan)
        color = get_color_global(val, key)
        ax_params.text(0, y, f"{label}:", ha='left', va='center', fontsize=10)
        ax_params.text(1, y, f"{val:.0f}" if not pd.isna(val) else "---", ha='right', va='center', weight='bold', color=color, fontsize=12)
        y -= 0.1
    
    return fig




def calcular_puntuacio_tempesta(sounding_data, params, nivell_conv):
    """
    Versió 2.0. Elimina la dependència de la component marítima, ja que
    el nou paràmetre "Potencial de Caça" és un millor indicador global.
    """
    if not params: return {'score': 0, 'color': '#808080'}

    score = 0
    
    # 1. Combustible (CAPE) - Fins a 4 punts
    sbcape = params.get('SBCAPE', 0) or 0
    if sbcape > 250: score += 1
    if sbcape > 750: score += 1
    if sbcape > 1500: score += 1
    if sbcape > 2500: score += 1

    # 2. Organització (Cisallament) - Fins a 3 punts
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    if bwd_6km > 15: score += 1
    if bwd_6km > 25: score += 1
    if bwd_6km > 35: score += 1

    # 3. Disparador (Convergència) - Fins a 3 punts
    conv_key = f'CONV_{nivell_conv}hPa'
    raw_conv_value = params.get(conv_key, 0)
    conv = raw_conv_value if isinstance(raw_conv_value, (int, float, np.number)) else 0
    if conv > 10: score += 1
    if conv > 20: score += 1
    if conv > 35: score += 1

    # 4. Ajustaments per factors clau
    cin = params.get('SBCIN', 0) or 0
    if cin < -100: score -= 2
    elif cin < -50: score -= 1

    srh_3km = params.get('SRH_0-3km', 0) or 0
    if srh_3km > 250: score += 1

    final_score = max(0, min(10, round(score)))
    
    color = '#808080'
    if final_score >= 8: color = '#dc3545'
    elif final_score >= 6: color = '#fd7e14'
    elif final_score >= 4: color = '#ffc107'
    elif final_score >= 1: color = '#2ca02c'

    return {'score': final_score, 'color': color}



def analitzar_potencial_caca(params, nivell_conv):
    """
    Sistema de Diagnòstic v7.0 - Lògica de Recepta Completa.
    Primer comprova una checklist bàsica (MUCAPE, Conv, PWAT) i, si es compleix,
    analitza el cisallament (BWD) i l'helicitat (SRH) per refinar la recomanació.
    """
    # --- 1. Extracció de tots els paràmetres necessaris ---
    mucape = params.get('MUCAPE', 0) or 0
    pwat = params.get('PWAT', 0) or 0
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_1km = params.get('SRH_0-1km', 0) or 0
    srh_3km = params.get('SRH_0-3km', 0) or 0 # Helicitat a nivells mitjans
    
    conv_key = f'CONV_{nivell_conv}hPa'
    raw_conv_value = params.get(conv_key, 0)
    conv = raw_conv_value if isinstance(raw_conv_value, (int, float, np.number)) else 0

    # --- 2. Comprovació de la "Checklist" bàsica ---
    checklist_ok = (mucape > 1000) and (conv >= 15) and (pwat > 20)

    # --- 3. Lògica de Decisió Jeràrquica ---

    # PRIMER: Si la checklist bàsica no es compleix, és un "No" i expliquem per què.
    if not checklist_ok:
        motius_fallida = []
        if not (mucape > 700):
            motius_fallida.append(f"MUCAPE insuficient ({mucape:.0f} es baix)")
        if not (conv >= 25):
            motius_fallida.append(f"Disparador feble (Conv. {conv:.1f} tens menys del recomandat")
        if not (pwat > 20):
            motius_fallida.append(f"Humitat limitada (PWAT {pwat:.1f} falta potencial per pluges")
        motiu_final = ". ".join(motius_fallida) + "."
        return {'text': 'No', 'color': '#dc3545', 'motiu': motiu_final}

    # SI ARRIBEM AQUÍ, LA CHECKLIST BÀSICA ES COMPLEIX. Ara refinem el "Sí".
    else:
        # Cas 1: Entorn de Supercèl·lula (el millor escenari)
        if bwd_6km >= 35 and (srh_1km > 150 or srh_3km > 250):
            return {'text': 'Sí, Prioritari', 'color': '#9370db', 
                    'motiu': f'Checklist OK. Entorn clàssic de supercèl·lula (BWD: {bwd_6km:.0f} kt, SRH: {srh_1km:.0f} m²/s²).'}

        # Cas 2: Entorn de Multicèl·lula Organitzada
        elif bwd_6km >= 25:
            return {'text': 'Sí, Interessant', 'color': '#28a745', 
                    'motiu': f'Checklist OK. Bon potencial per a tempestes organitzades (BWD: {bwd_6km:.0f} kt).'}

        # Cas 3: Entorn de Tempesta d'Impuls (sense organització)
        else:
            return {'text': 'Potencial Aïllat', 'color': '#ffc107', 
                    'motiu': f'Checklist OK, però sense organització (BWD: {bwd_6km:.0f} kt). Es formaran tempestes, però probablement desorganitzades.'}
        
def analitzar_estructura_tempesta(params):
    """
    Analitza els paràmetres clau per determinar el potencial d'organització,
    la formació de mesociclons i el risc tornàdic a la base del núvol.
    Retorna un diccionari amb text i color per a la UI.
    """
    # Valors per defecte (entorn de baixa severitat)
    resultats = {
        'organitzacio': {'text': 'Febla (Cèl·lules Aïllades)', 'color': '#2ca02c'},
        'mesociclo': {'text': 'Molt Baix', 'color': '#2ca02c'},
        'tornadic': {'text': 'Nul', 'color': '#2ca02c'}
    }

    # Extreure paràmetres de manera segura
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_3km = params.get('SRH_0-3km', 0) or 0
    srh_1km = params.get('SRH_0-1km', 0) or 0
    lcl_hgt = params.get('LCL_Hgt', 9999) or 9999

    # 1. Potencial d'Organització (basat en el Cisallament 0-6 km)
    # Aquest paràmetre ens diu si la tempesta serà una simple cèl·lula, multicèl·lules o una supercèl·lula.
    if bwd_6km >= 40:
        resultats['organitzacio'] = {'text': 'Molt Alt (Pot. Supercèl·lules)', 'color': '#dc3545'}
    elif bwd_6km >= 25:
        resultats['organitzacio'] = {'text': 'Moderat (Pot. Multicèl·lules)', 'color': '#ffc107'}

    # 2. Potencial de Mesocicló (basat en l'Helicitat 0-3 km)
    # Aquest paràmetre mesura la rotació a nivells mitjans, el cor del mesocicló.
    if srh_3km >= 400:
        resultats['mesociclo'] = {'text': 'Extrem', 'color': '#dc3545'}
    elif srh_3km >= 250:
        resultats['mesociclo'] = {'text': 'Alt', 'color': '#fd7e14'}
    elif srh_3km >= 150:
        resultats['mesociclo'] = {'text': 'Moderat', 'color': '#ffc107'}

    # 3. Potencial Tornàdic a la Base (Helicitat 0-1 km + Alçada LCL)
    # Aquest és el més important per a tornados. Necessitem rotació a nivells molt baixos i una base del núvol baixa.
    if srh_1km >= 200 and lcl_hgt < 1000:
        resultats['tornadic'] = {'text': 'Alt (Entorn Favorable)', 'color': '#dc3545'}
    elif srh_1km >= 150 and lcl_hgt < 1200:
        resultats['tornadic'] = {'text': 'Moderat', 'color': '#fd7e14'}
    elif srh_1km >= 100:
        resultats['tornadic'] = {'text': 'Baix (Rotació a la base)', 'color': '#ffc107'}

    return resultats

def analitzar_amenaces_especifiques(params):
    """
    Sistema d'Anàlisi d'Amenaces v3.0.
    Ajusta el potencial teòric de calamarsa i activitat elèctrica basant-se en
    la probabilitat que la tempesta es formi (Balanç Convergència vs. CIN).
    """
    resultats = {
        'calamarsa': {'text': 'Nul·la', 'color': '#808080'},
        'esclafits': {'text': 'Nul·la', 'color': '#808080'},
        'llamps': {'text': 'Nul·la', 'color': '#808080'}
    }

    # --- 1. EXTRACCIÓ DE PARÀMETRES ---
    updraft = params.get('MAX_UPDRAFT', 0) or 0
    isozero = params.get('FREEZING_LVL_HGT', 5000) or 5000
    li = params.get('LI', 5) or 5
    el_hgt = params.get('EL_Hgt', 0) or 0
    lr_0_3km = params.get('LR_0-3km', 0) or 0
    pwat = params.get('PWAT', 100) or 100
    mucape = params.get('MUCAPE', 0) or 0

    # Paràmetres per al balanç del disparador
    conv_key = next((k for k in params if k.startswith('CONV_')), None)
    
    # --- BLOC DE DEFENSA ANTI-ERRORS (CORRECCIÓ) ---
    # S'assegura que 'convergencia' sigui sempre un número abans de qualsevol comparació.
    raw_conv_value = params.get(conv_key, 0)
    convergencia = raw_conv_value if isinstance(raw_conv_value, (int, float, np.number)) else 0
    
    cin = min(params.get('SBCIN', 0), params.get('MUCIN', 0)) or 0

    # --- 2. AVALUACIÓ DEL POTENCIAL DE DISPAR ---
    # Definim un factor de realització (de 0 a 1)
    factor_realitzacio = 0.0
    # Aquesta línia ara és segura gràcies a la comprovació anterior
    if convergencia >= 30 and cin > -100:
        factor_realitzacio = 1.0  # Disparador molt probable
    elif convergencia >= 15 and cin > -50:
        factor_realitzacio = 0.7  # Disparador probable
    elif cin > -20:
        factor_realitzacio = 0.4  # Disparador possible (sense tapa)

    # Si no hi ha CAPE, no hi ha amenaça, independentment del disparador
    if mucape < 300:
        return resultats

    # --- 3. ANÀLISI D'AMENACES AMB AJUSTAMENT ---

    # --- Calamarsa Gran (>2cm) ---
    potencial_calamarsa_teoric = 0
    if updraft > 55 or (updraft > 45 and isozero < 3500): potencial_calamarsa_teoric = 4 # Molt Alt
    elif updraft > 40 or (updraft > 30 and isozero < 3800): potencial_calamarsa_teoric = 3 # Alt
    elif updraft > 25: potencial_calamarsa_teoric = 2 # Moderat
    elif updraft > 15: potencial_calamarsa_teoric = 1 # Baix

    # Ajustem el potencial teòric amb el factor de realització
    potencial_calamarsa_real = potencial_calamarsa_teoric * factor_realitzacio
    if potencial_calamarsa_real >= 3.5:
        resultats['calamarsa'] = {'text': 'Molt Alta', 'color': '#dc3545'}
    elif potencial_calamarsa_real >= 2.5:
        resultats['calamarsa'] = {'text': 'Alta', 'color': '#fd7e14'}
    elif potencial_calamarsa_real >= 1.5:
        resultats['calamarsa'] = {'text': 'Moderada', 'color': '#ffc107'}
    elif potencial_calamarsa_real >= 0.5:
        resultats['calamarsa'] = {'text': 'Baixa', 'color': '#2ca02c'}

    # --- Activitat Elèctrica (Llamps) ---
    potencial_llamps_teoric = 0
    if li < -7 or (li < -5 and el_hgt > 12000): potencial_llamps_teoric = 4 # Extrema
    elif li < -4 or (li < -2 and el_hgt > 10000): potencial_llamps_teoric = 3 # Alta
    elif li < -1: potencial_llamps_teoric = 2 # Moderada
    elif mucape > 150: potencial_llamps_teoric = 1 # Baixa

    # Ajustem el potencial teòric
    potencial_llamps_real = potencial_llamps_teoric * factor_realitzacio
    if potencial_llamps_real >= 3.5:
        resultats['llamps'] = {'text': 'Extrema', 'color': '#dc3545'}
    elif potencial_llamps_real >= 2.5:
        resultats['llamps'] = {'text': 'Alta', 'color': '#fd7e14'}
    elif potencial_llamps_real >= 1.5:
        resultats['llamps'] = {'text': 'Moderada', 'color': '#ffc107'}
    elif potencial_llamps_real >= 0.5:
        resultats['llamps'] = {'text': 'Baixa', 'color': '#2ca02c'}

    # --- Esclafits (Aquesta amenaça es manté igual, ja que no depèn tant del CAPE) ---
    if lr_0_3km > 8.0 and pwat < 35:
        resultats['esclafits'] = {'text': 'Alta', 'color': '#fd7e14'}
    elif lr_0_3km > 7.0 and pwat < 40:
        resultats['esclafits'] = {'text': 'Moderada', 'color': '#ffc107'}
    elif lr_0_3km > 6.5:
        resultats['esclafits'] = {'text': 'Baixa', 'color': '#2ca02c'}
        
    return resultats


def analitzar_component_maritima(sounding_data, poble_sel):
    """
    Versió 2.0 - Analitza el vent EN SUPERFÍCIE.
    Determina si el vent a la superfície té component marítima,
    basant-se en la direcció del mar per a la localitat seleccionada.
    """
    # Comprovacions de seguretat inicials
    if not sounding_data:
        return {'text': 'N/A', 'color': '#808080'}
    
    city_data = CIUTATS_CATALUNYA.get(poble_sel)
    if not city_data or city_data.get('sea_dir') is None:
        # Si és una ciutat d'interior, la mètrica no aplica
        return {'text': 'N/A', 'color': '#808080'}

    sea_dir_range = city_data['sea_dir']
    
    try:
        # Extraiem els components U i V del sondeig
        u, v = sounding_data[3], sounding_data[4]
        
        # --- LÒGICA CORREGIDA: Utilitzem només el vent de superfície ---
        # L'índex [0] correspon a la superfície perquè les dades estan ordenades
        u_sfc, v_sfc = u[0], v[0]
        
        # Calculem la direcció i la velocitat a partir dels components
        direction = mpcalc.wind_direction(u_sfc, v_sfc).m
        speed = mpcalc.wind_speed(u_sfc, v_sfc).to('km/h').m

        # Funció auxiliar per comprovar si un angle està dins d'un rang
        def is_in_range(angle, range_tuple):
            start, end = range_tuple
            if start <= end:
                return start <= angle <= end
            else: # El rang creua els 0/360 graus
                return start <= angle or angle <= end

        # Comprovem si la direcció del vent prové del mar i si té una mínima força
        if is_in_range(direction, sea_dir_range) and speed > 5:
            return {'text': 'Sí', 'color': '#28a745'} # Verd = Ingredient present
        else:
            return {'text': 'No', 'color': '#dc3545'} # Vermell = Ingredient absent
            
    except (IndexError, ValueError):
        # Si hi ha algun problema amb les dades (p. ex., llistes buides)
        return {'text': 'Error', 'color': '#808080'}
    

def analitzar_regims_de_vent_cat(sounding_data, params_calc, hora_del_sondeig):
    """
    Sistema Expert v15.0. Diagnostica el règim de vent dominant i retorna
    un veredicte meteorològic complet amb un color associat.
    """
    resultat = {
        'tipus': 'Calma', 'detall': 'Vent quasi inexistent.',
        'veredicte': "Situació de calma absoluta, sense un flux d'aire definit.", 'color': '#808080'
    }
    if not sounding_data or not params_calc: return resultat
    p, T, Td, u, v = sounding_data[0], sounding_data[1], sounding_data[2], sounding_data[3], sounding_data[4]
    try:
        LLINDAR_REGIM_FORT = 20; LLINDAR_CALMA = 3
        hora_num = int(hora_del_sondeig.split(':')[0]); es_horari_diurn = 11 <= hora_num <= 21
        def es_llevant(d): return 45 <= d <= 135
        def es_vent_de_mar(d): return 45 <= d <= 200
        def es_vent_del_nord(d): return d >= 315 or d <= 45
        def es_vent_de_ponent(d): return 225 <= d <= 315
        vel_sfc = float(mpcalc.wind_speed(u[0], v[0]).to('km/h').m)
        dir_sfc = float(mpcalc.wind_direction(u[0], v[0]).m)
        dir_cardinal_sfc = graus_a_direccio_cardinal(dir_sfc)
        if vel_sfc < LLINDAR_CALMA:
            resultat['detall'] = f"Vent variable a {vel_sfc:.0f} km/h"; return resultat
        mask_alts = (p.m <= 700) & (p.m >= 300); mask_baixos = (p.m <= p[0].m) & (p.m >= 900)
        if np.count_nonzero(mask_alts) > 2 and np.count_nonzero(mask_baixos) > 2:
            vel_alts = float(mpcalc.wind_speed(np.mean(u[mask_alts]), np.mean(v[mask_alts])).to('km/h').m)
            dir_alts = float(mpcalc.wind_direction(np.mean(u[mask_alts]), np.mean(v[mask_alts])).m)
            dir_baixos = float(mpcalc.wind_direction(np.mean(u[mask_baixos]), np.mean(v[mask_baixos])).m)
            if vel_alts > 60 and es_vent_del_nord(dir_alts) and es_vent_de_mar(dir_baixos):
                resultat.update({'tipus': "Rebuf (Especial)", 'detall': f"Nortada de {vel_alts:.0f} km/h en alçada", 'veredicte': "Gran contrast. L'aire fred xoca amb l'aire humit, un mecanisme de dispar molt potent.", 'color': '#dc3545'}); return resultat
        mask_sinoptic = (p.m <= p[0].m) & (p.m >= 700)
        if np.count_nonzero(mask_sinoptic) > 3:
            vel_sinoptic = float(mpcalc.wind_speed(np.mean(u[mask_sinoptic]), np.mean(v[mask_sinoptic])).to('km/h').m)
            if vel_sinoptic > LLINDAR_REGIM_FORT:
                dir_sinoptic = float(mpcalc.wind_direction(np.mean(u[mask_sinoptic]), np.mean(v[mask_sinoptic])).m)
                dir_cardinal = graus_a_direccio_cardinal(dir_sinoptic)
                if es_llevant(dir_sinoptic):
                    resultat.update({'tipus': "Llevantada", 'detall': f"{dir_cardinal} a {vel_sinoptic:.0f} km/h", 'veredicte': "Entrada d'humitat generalitzada. Potencial de pluges extenses i/o tempestes.", 'color': '#28a745'}); return resultat
                elif es_vent_del_nord(dir_sinoptic):
                    resultat.update({'tipus': "Nortada", 'detall': f"{dir_cardinal} a {vel_sinoptic:.0f} km/h", 'veredicte': "Entrada d'aire fred i sec. Ambient ventós, baix potencial de precipitació.", 'color': '#007bff'}); return resultat
                elif es_vent_de_ponent(dir_sinoptic):
                    resultat.update({'tipus': "Ponentada", 'detall': f"{dir_cardinal} a {vel_sinoptic:.0f} km/h", 'veredicte': "Vent sec i reescalfat. Temperatures altes, humitat baixa i risc d'incendi.", 'color': '#fd7e14'}); return resultat
        if vel_sfc > LLINDAR_REGIM_FORT and es_vent_de_mar(dir_sfc) and es_horari_diurn:
             resultat.update({'tipus': "Marinada Forta", 'detall': f"{dir_cardinal_sfc} a {vel_sfc:.0f} km/h", 'veredicte': "Brisa marina que injecta humitat i pot actuar com a disparador a l'interior.", 'color': '#17a2b8'}); return resultat
        tipus = "Marinada Feble" if es_horari_diurn and es_vent_de_mar(dir_sfc) else "Terral / Vent Nocturn"
        veredicte = "Brisa marina feble, típica de calma." if tipus == "Marinada Feble" else "Vent fluix de terra o residual de mar. Sense un règim clar."
        resultat.update({'tipus': tipus, 'detall': f"{dir_cardinal_sfc} a {vel_sfc:.0f} km/h", 'veredicte': veredicte, 'color': '#808080'})
        return resultat
    except Exception:
        return {'tipus': 'Error d\'Anàlisi', 'detall': 'No s\'ha pogut determinar.', 'veredicte': "Hi ha hagut un problema analitzant el perfil de vent.", 'color': '#dc3545'}
    

def ui_caixa_parametres_sondeig(sounding_data, params, nivell_conv, hora_actual, poble_sel, avis_proximitat=None):
    TOOLTIPS = {
        'SBCAPE': "Energia Potencial Convectiva Disponible (CAPE) des de la Superfície. Mesura el 'combustible' per a les tempestes a partir d'una bombolla d'aire a la superfície.",
        'MUCAPE': "El CAPE més alt possible a l'atmosfera (Most Unstable). Útil per detectar inestabilitat elevada, fins i tot si la superfície és estable.",
        'CONVERGENCIA': f"Força de la convergència de vent a {nivell_conv}hPa. Actua com el 'disparador' o 'mecanisme de forçament' que obliga l'aire a ascendir, ajudant a iniciar les tempestes.",
        'SBCIN': "Inhibició Convectiva (CIN) des de la Superfície. És l'energia necessària per vèncer l'estabilitat inicial. Valors molt negatius actuen com una 'tapa' que impedeix les tempestes.",
        'MUCIN': "La CIN associada al MUCAPE.",
        'POTENCIAL_CACA': "Diagnòstic ràpid que sintetitza si la combinació d'energia (CAPE), disparador (Convergència) i organització (Cisallament) és favorable per a la formació de tempestes que valguin la pena observar o 'caçar'.",
        'LI': "Índex d'Elevació (Lifted Index). Mesura la diferència de temperatura a 500hPa entre l'entorn i una bombolla d'aire elevada. Valors molt negatius indiquen una forta inestabilitat.",
        'PWAT': "Aigua Precipitable Total (Precipitable Water). Quantitat total de vapor d'aigua en la columna atmosfàrica. Valors alts indiquen potencial per a pluges fortes.",
        'LCL_Hgt': "Alçada del Nivell de Condensació per Elevació (LCL). És l'alçada a la qual es formarà la base del núvol. Valors baixos (<1000m) afavoreixen el temps sever.",
        'LFC_Hgt': "Alçada del Nivell de Convecció Lliure (LFC). És l'alçada a partir de la qual una bombolla d'aire puja lliurement sense necessitat de forçament.",
        'EL_Hgt': "Alçada del Nivell d'Equilibri (EL). És l'alçada estimada del cim de la tempesta (top del cumulonimbus).",
        'BWD_0-6km': "Cisallament del Vent (Bulk Wind Shear) entre 0 i 6 km. Crucial per a l'organització de les tempestes (multicèl·lules, supercèl·lules).",
        'BWD_0-1km': "Cisallament del Vent entre 0 i 1 km. Important per a la rotació a nivells baixos (tornados).",
        'T_500hPa': "Temperatura a 500 hPa (uns 5.500 metres). Temperatures molt fredes en alçada disparen la inestabilitat.",
        'MAX_UPDRAFT': "Estimació de la velocitat màxima del corrent ascendent. Indicador directe del potencial de calamarsa.",
        'AMENACA_CALAMARSA': "Probabilitat de calamarsa de mida significativa (>2 cm). Es basa en la potència del corrent ascendent (MAX_UPDRAFT) i l'alçada de la isoterma de 0°C.",
        'PUNTUACIO_TEMPESTA': "Índex de 0 a 10 que valora el potencial global de formació de tempestes, combinant els ingredients clau.",
        'AMENACA_LLAMPS': "Potencial d'activitat elèctrica. S'estima a partir de la inestabilitat (LI) i la profunditat de la tempesta (EL_Hgt)."
    }
    
    

    def styled_metric(label, value, unit, param_key, tooltip_text="", precision=0, reverse_colors=False):
        display_label = label
        color = "#FFFFFF"
        val_str = "---"
        font_size = "1.6em"
        # Comencem amb la unitat per defecte per a tots els paràmetres
        unit_display = f"({unit})"

        is_numeric = isinstance(value, (int, float, np.number))

        if pd.notna(value) and is_numeric:
            if 'CONV' in param_key:
                if value < 0:
                    # DIVERGÈNCIA: Mostrem text, sense unitat
                    color = "#5bc0de"
                    display_label = "Dinàmica Vertical"
                    font_size = "1.4em"
                    unit_display = "" # Amaguem la unitat
                    
                    abs_val = abs(value)
                    if abs_val >= 30: val_str = "Divergència<br>Forta"
                    elif abs_val >= 15: val_str = "Divergència<br>Moderada"
                    else: val_str = "Divergència<br>Feble"
                else:
                    # CONVERGÈNCIA: Mostrem número i títol amb rang
                    display_label = "Convergència"
                    
                    # <<<--- NOU BLOC: Canviem la unitat pel rang de força --->>>
                    if value < 5: strength_label = "(Calma)"
                    elif value < 15: strength_label = "(Feble)"
                    elif value < 30: strength_label = "(Moderada)"
                    elif value < 40: strength_label = "(Forta)"
                    else: strength_label = "(Molt Forta)"
                    unit_display = strength_label # Substituïm la unitat (10⁻⁵ s⁻¹) pel text
                    # <<<--- FI DEL NOU BLOC --->>>

                    thresholds = [5, 15, 30, 40]
                    colors = ["#808080", "#2ca02c", "#ffc107", "#fd7e14", "#dc3545"]
                    color = colors[np.searchsorted(thresholds, value)]
                    val_str = f"{value:.{precision}f}"

            elif param_key == 'T_500hPa':
                thresholds = [-8, -14, -18, -22]
                colors = ["#2ca02c", "#ffc107", "#fd7e14", "#dc3545", "#b300ff"]
                color = colors[len(thresholds) - np.searchsorted(thresholds, value, side='right')]
                val_str = f"{value:.{precision}f}"
            else:
                color = get_color_global(value, param_key, reverse_colors)
                val_str = f"{value:.{precision}f}"
        
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""
        
        # Si el text ja és 'Divergència...', no cal mostrar la unitat.
        if "Divergència" in val_str:
            unit_display = ""

        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 0.8em; color: #FAFAFA;">{display_label} {unit_display}{tooltip_html}</span>
            <strong style="font-size: {font_size}; color: {color}; line-height: 1.1;">{val_str}</strong>
        </div>""", unsafe_allow_html=True)

    def styled_qualitative(label, analysis_dict, tooltip_text=""):
        # ... (aquesta funció interna no canvia)
        text = analysis_dict.get('text', 'N/A')
        color = analysis_dict.get('color', '#808080')
        motiu = analysis_dict.get('motiu', '')
        full_tooltip = f"{tooltip_text} Motiu del diagnòstic: {motiu}" if motiu else tooltip_text
        tooltip_html = f' <span title="{full_tooltip}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if full_tooltip else ""
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;">
            <span style="font-size: 0.8em; color: #FAFAFA;">{label}{tooltip_html}</span><br>
            <strong style="font-size: 1.6em; color: {color};">{text}</strong>
        </div>""", unsafe_allow_html=True)
        
    def styled_threat(label, text, color, tooltip_key):
        # ... (aquesta funció interna no canvia)
        tooltip_text = TOOLTIPS.get(tooltip_key, "")
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;">
            <span style="font-size: 0.8em; color: #FAFAFA;">{label}{tooltip_html}</span><br>
            <strong style="font-size: 1.6em; color: {color};">{text}</strong>
        </div>""", unsafe_allow_html=True)

    st.markdown("##### Paràmetres del Sondeig")
    
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCAPE", params.get('SBCAPE', np.nan), "J/kg", 'SBCAPE', tooltip_text=TOOLTIPS.get('SBCAPE'))
    with cols[1]: styled_metric("MUCAPE", params.get('MUCAPE', np.nan), "J/kg", 'MUCAPE', tooltip_text=TOOLTIPS.get('MUCAPE'))
    with cols[2]: 
        conv_key = f'CONV_{nivell_conv}hPa'
        styled_metric("Convergència", params.get(conv_key, np.nan), "10⁻⁵ s⁻¹", conv_key, precision=1, tooltip_text=TOOLTIPS.get('CONVERGENCIA'))
    
    cols = st.columns(3)
    with cols[0]: styled_metric("SBCIN", params.get('SBCIN', np.nan), "J/kg", 'SBCIN', reverse_colors=True, tooltip_text=TOOLTIPS.get('SBCIN'))
    with cols[1]: styled_metric("MUCIN", params.get('MUCIN', np.nan), "J/kg", 'MUCIN', reverse_colors=True, tooltip_text=TOOLTIPS.get('MUCIN'))
    with cols[2]:
        analisi_caca = analitzar_potencial_caca(params, nivell_conv)
        
        # --- LÒGICA D'ARREGLAMENT DEFINITIU ---
        # Si la resposta és "No", construïm un HTML personalitzat
        # que inclou el motiu directament a la caixa.
        if analisi_caca['text'] == 'No':
            motiu = analisi_caca.get('motiu', 'Motiu no especificat.')
            tooltip_text = TOOLTIPS.get('POTENCIAL_CACA')
            tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.7em; opacity: 0.7;">❓</span>'

            st.markdown(f"""
            <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 0.8em; color: #FAFAFA;">Val la pena anar-hi?{tooltip_html}</span>
                <strong style="font-size: 1.4em; color: #dc3545; line-height: 1.2;">No</strong>
                <span style="font-size: 0.7em; color: #E0E0E0; line-height: 1.1; font-style: italic;">{motiu}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Si la resposta és positiva, utilitzem la funció de sempre.
            styled_qualitative("Val la pena anar-hi?", analisi_caca, tooltip_text=TOOLTIPS.get('POTENCIAL_CACA'))

    cols = st.columns(3)
    with cols[0]: 
        li_value = params.get('LI', np.nan)
        if hasattr(li_value, '__len__') and not isinstance(li_value, str) and len(li_value) > 0: li_value = li_value[0]
        styled_metric("LI", li_value, "°C", 'LI', precision=1, reverse_colors=True, tooltip_text=TOOLTIPS.get('LI'))
    with cols[1]: 
        styled_metric("PWAT", params.get('PWAT', np.nan), "mm", 'PWAT', precision=1, tooltip_text=TOOLTIPS.get('PWAT'))
    with cols[2]:
        analisi_temps = analitzar_potencial_meteorologic(params, nivell_conv, hora_actual)
        emoji = analisi_temps['emoji']
        descripcio = analisi_temps['descripcio']

        if avis_proximitat:
            background_color = "#fd7e14"
            title_text = "⚠️ATENCIÓ: FOCUS APROP⚠️"
            main_text = "Anirá cap a tú"
            sub_text = f"Actual: {emoji} {descripcio}"
            st.markdown(f"""
            <div class="blinking-alert" style="text-align: center; padding: 5px; border-radius: 10px; background-color: {background_color}; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 0.8em; color: #FFFFFF; font-weight: bold;">{title_text}</span>
                <strong style="font-size: 1.2em; color: #FFFFFF; line-height: 1.2;">{main_text}</strong>
                <span style="font-size: 0.7em; color: #FFFFFF; opacity: 0.9;">{sub_text}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 0.8em; color: #FAFAFA;">Tipus de Cel Previst</span>
                <strong style="font-size: 1.8em; line-height: 1;">{emoji}</strong>
                <span style="font-size: 0.8em; color: #E0E0E0;">{descripcio}</span>
            </div>""", unsafe_allow_html=True)
        
    cols = st.columns(3)
    with cols[0]: styled_metric("LCL", params.get('LCL_Hgt', np.nan), "m", 'LCL_Hgt', precision=0, tooltip_text=TOOLTIPS.get('LCL_Hgt'))
    with cols[1]: styled_metric("LFC", params.get('LFC_Hgt', np.nan), "m", 'LFC_Hgt', precision=0, tooltip_text=TOOLTIPS.get('LFC_Hgt'))
    with cols[2]: styled_metric("CIM (EL)", params.get('EL_Hgt', np.nan), "m", 'EL_Hgt', precision=0, tooltip_text=TOOLTIPS.get('EL_Hgt'))
        
    cols = st.columns(3)
    with cols[0]: styled_metric("BWD 0-6km", params.get('BWD_0-6km', np.nan), "nusos", 'BWD_0-6km', tooltip_text=TOOLTIPS.get('BWD_0-6km'))
    with cols[1]: styled_metric("BWD 0-1km", params.get('BWD_0-1km', np.nan), "nusos", 'BWD_0-1km', tooltip_text=TOOLTIPS.get('BWD_0-1km'))
    with cols[2]: 
        styled_metric("T 500hPa", params.get('T_500hPa', np.nan), "°C", 'T_500hPa', precision=1, tooltip_text=TOOLTIPS.get('T_500hPa'))

    st.markdown("##### Potencial d'Amenaces Severes")
    amenaces = analitzar_amenaces_especifiques(params)
    
    puntuacio_resultat = calcular_puntuacio_tempesta(sounding_data, params, nivell_conv)
    
    cols = st.columns(3)
    with cols[0]:
        styled_threat("Calamarsa Gran (>2cm)", amenaces['calamarsa']['text'], amenaces['calamarsa']['color'], 'AMENACA_CALAMARSA')
    with cols[1]:
        score_text = f"{puntuacio_resultat['score']} / 10"
        styled_threat("Índex de Potencial", score_text, puntuacio_resultat['color'], 'PUNTUACIO_TEMPESTA')
    with cols[2]:
        styled_threat("Activitat Elèctrica", amenaces['llamps']['text'], amenaces['llamps']['color'], 'AMENACA_LLAMPS')

def analitzar_vents_locals(sounding_data, poble_sel, hora_actual_str):
    """
    Sistema de Diagnòstic v2.0: Analitza els fenòmens eòlics a diferents nivells
    i retorna una llista de diccionaris preparats per a una UI de targetes.
    """
    diagnostics = []
    if not sounding_data:
        return [{'titol': "Sense Dades", 'descripcio': "No s'ha pogut carregar el perfil de vents.", 'emoji': "❓"}]

    city_data = CIUTATS_CATALUNYA.get(poble_sel)
    p, u, v = sounding_data[0], sounding_data[3], sounding_data[4]

    def degrees_to_cardinal_ca(deg):
        if pd.isna(deg): return "Variable"
        dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSO', 'SO', 'OSO', 'O', 'ONO', 'NO', 'NNO']
        return dirs[int(round(deg / 22.5)) % 16]

    def wind_info_at_level(target_hpa):
        try:
            idx = np.argmin(np.abs(p.m - target_hpa))
            if np.abs(p.m[idx] - target_hpa) > 30: return np.nan, np.nan
            u_comp, v_comp = u[idx], v[idx]
            spd = mpcalc.wind_speed(u_comp, v_comp).to('km/h').m
            drct = mpcalc.wind_direction(u_comp, v_comp).m if spd >= 1 else np.nan
            return drct, spd
        except Exception: return np.nan, np.nan

    drct_sfc, spd_sfc = wind_info_at_level(p.m[0])
    drct_925, spd_925 = wind_info_at_level(925)
    drct_700, spd_700 = wind_info_at_level(700)
    
    hora = int(hora_actual_str.split(':')[0])
    es_diurn = 9 <= hora <= 20

    # Anàlisi de Superfície
    if spd_sfc < 3:
        diagnostics.append({'titol': "Superfície: Calma", 'descripcio': f"Vent pràcticament inexistent a la superfície.", 'emoji': "🧘"})
    elif city_data and city_data.get('sea_dir') and es_diurn and 45 <= drct_sfc <= 200:
        diagnostics.append({'titol': f"Superfície: Marinada ({graus_a_direccio_cardinal(drct_sfc)}, {spd_sfc:.0f} km/h)", 'descripcio': "Brisa humida de mar a terra. Modera la temperatura i aporta humitat.", 'emoji': "🌬️"})
    else:
        diagnostics.append({'titol': f"Superfície: Terral / Vent Local ({graus_a_direccio_cardinal(drct_sfc)}, {spd_sfc:.0f} km/h)", 'descripcio': "Flux de terra, generalment més sec i reescalfat durant el dia.", 'emoji': "🏜️"})

    # Anàlisi a 925 hPa
    if pd.notna(drct_925):
        desc_925 = "Advecció marítima. Aporta humitat i núvols baixos." if 45 <= drct_925 <= 200 else "Flux de terra/interior. Tendeix a ser més sec."
        diagnostics.append({'titol': f"925 hPa (~750m): {graus_a_direccio_cardinal(drct_925)}, {spd_925:.0f} km/h", 'descripcio': desc_925, 'emoji': "☁️"})

    # Anàlisi a 700 hPa
    if pd.notna(drct_700):
        desc_700 = "Flux de sud. Pot indicar l'aproximació d'un canvi de temps." if 135 <= drct_700 <= 225 else "Flux a nivells mitjans. Dirigeix el moviment de les tempestes."
        diagnostics.append({'titol': f"700 hPa (~3000m): {graus_a_direccio_cardinal(drct_700)}, {spd_700:.0f} km/h", 'descripcio': desc_700, 'emoji': "✈️"})
    
    # Diagnòstic de Cisallament (Diferència de vent)
    if pd.notna(drct_sfc) and pd.notna(drct_700):
        diff = abs(drct_sfc - drct_700)
        if diff > 90 and diff < 270:
             diagnostics.append({'titol': "Cisallament Direccional Present", 'descripcio': "El vent canvia de direcció amb l'alçada. Això pot afavorir la rotació de les tempestes si n'hi ha.", 'emoji': "🔄"})

    return diagnostics








def start_transition(zone_id):
    """Callback per iniciar la transició de vídeo."""
    st.session_state['zone_selected'] = zone_id
    st.session_state['show_transition_video'] = True


def degrees_to_cardinal_ca(d):
    """Converteix graus a punts cardinals en català."""
    if not isinstance(d, (int, float, np.number)) or pd.isna(d):
        return "N/A"
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return dirs[int(round(d / 22.5)) % 16]

def degrees_to_cardinal_ca(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    ix = int((deg + 22.5) // 45) % 8
    return dirs[ix]

# --- Funció que ja tens per al dial, amb correccions ---
def crear_dial_vent_animat(label, wind_dir, wind_spd):
    """
    Dial de vent elegant tipus dashboard:
    - Cada dial és independent i apunta a la seva direcció real.
    - Oscil·lació segons velocitat del vent (tremolament).
    """
    uid = str(uuid.uuid4()).replace('-', '')  # ID únic per a cada animació

    # Estat del vent
    if pd.notna(wind_spd) and wind_spd < 1:
        state = "calm"
    elif pd.notna(wind_spd) and pd.isna(wind_dir):
        state = "no_dir"
    elif pd.notna(wind_spd) and pd.notna(wind_dir):
        state = "valid"
    else:
        state = "no_data"

    # Variables segons estat
    if state == "valid":
        dir_cardinal = degrees_to_cardinal_ca(wind_dir)
        spd_text = f"{wind_spd:.0f}"
        base_angle = wind_dir
        arrow_color = "#F6F6F6A7"
    elif state == "no_dir":
        dir_cardinal = "--"
        spd_text = f"{wind_spd:.0f}"
        base_angle = 0
        arrow_color = "#AAA"
    elif state == "calm":
        dir_cardinal = "CALM"
        spd_text = "0"
        base_angle = 0
        arrow_color = "#AAA"
    else:
        dir_cardinal = "N/A"
        spd_text = "--"
        base_angle = 0
        arrow_color = "#555"

    # Oscil·lació segons velocitat
    if pd.notna(wind_spd):
        if wind_spd <= 21:
            amplitude = 3
            duration = 4

        elif wind_spd < 30:
            amplitude = 3
            duration = 2

        elif wind_spd < 40:
            amplitude = 1
            duration = 1

        elif wind_spd < 50:
            amplitude = 0.5
            duration = 0.5

        elif wind_spd < 60:
            amplitude = 0.2
            duration = 0.2

        elif wind_spd < 80:
            amplitude = 0.1
            duration = 0.1

        else:
            amplitude = 0.01
            duration = 0.01
        
 

    tremble = min(wind_spd / 2, 15) if pd.notna(wind_spd) else 0
    angle_start = base_angle - amplitude - tremble
    angle_end = base_angle + amplitude + tremble

    anim_name = f"oscillate_{uid}"

    html = f"""
    <style>
        .wind-dial-container-{uid} {{
            width: 160px; height: 200px; margin: auto;
            display: flex; flex-direction: column; align-items: center;
            font-family: 'Segoe UI', sans-serif;
        }}
        .dial-label-{uid} {{
            font-size: 1em; color: #ddd; margin-bottom: 10px;
            font-weight: 700; text-align: center; text-shadow: 0 0 5px #000;
        }}
        .dial-wrapper-{uid} {{
            position: relative; width: 120px; height: 120px; margin-bottom: 8px;
        }}
        .dial-{uid} {{
            width: 100%; height: 100%; border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #3a3c44, #1c1d22);
            border: 4px solid #666;
            box-shadow: inset 0 0 10px #00000060, 0 4px 6px #0008;
            display: flex; align-items: center; justify-content: center;
            position: relative;
        }}
        .dir-text-{uid} {{
            font-size: 1.2em; font-weight: bold; color: #fff;
            text-shadow: 0 0 5px #000;
            position: absolute; top: -20px; width: 100%; text-align: center;
        }}
        .speed-container-{uid} {{
            text-align: center; margin-top: 6px;
        }}
        .speed-text-{uid} {{
            font-size: 1.6em; font-weight: bold; color: #FFD700;
            display: block; text-shadow: 0 0 6px #000;
        }}
        .unit-text-{uid} {{
            font-size: 0.8em; color: #ccc; display: block;
        }}
        .arrow-container-{uid} {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            animation: {anim_name} {duration}s ease-in-out infinite alternate;
        }}
        .wind-arrow-{uid} {{
            width: 0; height: 0;
            border-left: 12px solid transparent;
            border-right: 12px solid transparent;
            border-bottom: 60px solid {arrow_color};
            position: absolute; top: -24px; left: 50%;
            transform: translateX(-50%) rotateX(15deg) rotateY(10deg);
            transform-origin: bottom center;
            filter: drop-shadow(0 2px 3px #0008);
        }}
        @keyframes {anim_name} {{
            from {{ transform: rotate({angle_start}deg); }}
            to {{ transform: rotate({angle_end}deg); }}
        }}
    </style>
    <div class="wind-dial-container-{uid}">
        <div class="dial-label-{uid}">{label}</div>
        <div class="dial-wrapper-{uid}">
            <div class="dial-{uid}">
                <div class="dir-text-{uid}">{dir_cardinal}</div>
            </div>
            <div class="arrow-container-{uid}">
                <div class="wind-arrow-{uid}"></div>
            </div>
        </div>
        <div class="speed-container-{uid}">
            <span class="speed-text-{uid}">{spd_text}</span>
            <span class="unit-text-{uid}">km/h</span>
        </div>
    </div>
    """
    return html


def get_wind_at_level(p_profile, u_profile, v_profile, target_level):
    """
    Busca el vent directament al perfil del sondeig per al nivell de pressió
    més proper a l'objectiu, emulant la lectura de les barbes de vent.
    És una funció robusta que evita errors d'interpolació.
    """
    try:
        if p_profile.size == 0:
            return np.nan, np.nan

        # 1. Troba l'índex del nivell de pressió més proper al nostre objectiu
        closest_idx = (np.abs(p_profile.m - target_level)).argmin()

        # 2. Comprovació de seguretat: si el nivell trobat està massa lluny, no és vàlid
        if np.abs(p_profile.m[closest_idx] - target_level) > 25: # Tolerància de 25 hPa
            return np.nan, np.nan

        # 3. Obtenim els components U i V en aquest índex exacte
        u_comp = u_profile[closest_idx]
        v_comp = v_profile[closest_idx]

        # 4. Calculem la direcció i la velocitat
        drct = mpcalc.wind_direction(u_comp, v_comp).m
        spd = mpcalc.wind_speed(u_comp, v_comp).to('km/h').m

        return drct, spd
    except Exception:
        return np.nan, np.nan

def crear_grafic_perfil_vent(p, wind_spd, wind_dir):
    """
    Crea un gràfic de Matplotlib que mostra la velocitat i direcció del vent
    amb l'altitud (pressió).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, dpi=100)
    fig.patch.set_alpha(0) # Fons transparent

    # Gràfic de Velocitat del Vent
    ax1.plot(wind_spd, p.m, color='blue', marker='o', markersize=4, linestyle='--')
    ax1.set_xlabel("Velocitat del Vent (km/h)", color='white')
    ax1.set_ylabel("Pressió (hPa)", color='white')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    
    # Gràfic de Direcció del Vent
    ax2.scatter(wind_dir, p.m, color='red', marker='x')
    ax2.set_xlabel("Direcció del Vent (°)", color='white')
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_xticklabels(['N', 'E', 'S', 'O', 'N'])
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.tick_params(axis='x', colors='white')

    # Ajustos generals
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    return fig


def ui_pestanya_analisis_vents(data_tuple, poble_sel, hora_actual_str, timestamp_str):
    """
    VERSIÓ AMB DISSENY DE TARGETES PREMIUM:
    - Mostra l'anàlisi de vents en un format de targetes visuals i modernes.
    """
    st.markdown(f"#### Anàlisi de Vents per a {poble_sel}")
    st.caption(timestamp_str)

    if not data_tuple:
        st.warning("No hi ha dades de sondeig disponibles per realitzar l'anàlisi de vents.")
        return

    diagnostics = analitzar_vents_locals(data_tuple[0], poble_sel, hora_actual_str)

    st.markdown("##### Diagnòstic de Fenòmens Eòlics")
    
    # CSS per a l'estil de les noves targetes d'informació
    st.markdown("""
    <style>
    .info-card {
        background-color: #262730; /* Fons fosc de la targeta */
        border-left: 5px solid #007bff; /* Vora esquerra de color */
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .info-icon {
        font-size: 2em; /* Icona més gran */
        margin-right: 16px;
    }
    .info-content {
        flex-grow: 1;
    }
    .info-title-line {
        display: flex;
        justify-content: space-between; /* Títol a l'esquerra, dades a la dreta */
        align-items: baseline;
    }
    .info-title {
        font-size: 1.15em;
        font-weight: bold;
        color: #f0f0f0;
    }
    .info-data {
        font-size: 1.1em;
        font-weight: bold;
        color: #ffc107; /* Dades en color groc d'accent */
        background-color: rgba(255, 193, 7, 0.1);
        padding: 2px 8px;
        border-radius: 5px;
    }
    .info-desc {
        font-size: 1em;
        color: #a0a0b0; /* Descripció en gris clar */
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Canviem el color de la vora segons la icona per a més dinamisme
    color_map = {"🧘": "#6c757d", "🌬️": "#17a2b8", "🏜️": "#fd7e14", "☁️": "#adb5bd", "✈️": "#6610f2", "🔄": "#ffc107"}

    for diag in diagnostics:
        border_color = color_map.get(diag['emoji'], "#007bff")
        st.markdown(f"""
        <div class="info-card" style="border-left-color: {border_color};">
            <div class="info-icon">{diag['emoji']}</div>
            <div class="info-content">
                <div class="info-title-line">
                    <span class="info-title">{diag['titol']}</span>
                    <span class="info-data">{diag['data']}</span>
                </div>
                <div class="info-desc">{diag['descripcio']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    st.markdown("##### Perfil de Vent per Nivells Clau")
    p, u, v = data_tuple[0][0], data_tuple[0][3], data_tuple[0][4]
    
    def get_wind_at_level(target_hpa):
        try:
            idx = np.argmin(np.abs(p.m - target_hpa))
            if np.abs(p.m[idx] - target_hpa) > 30: return np.nan, np.nan
            u_comp, v_comp = u[idx], v[idx]; spd = mpcalc.wind_speed(u_comp, v_comp).to('km/h').m
            drct = mpcalc.wind_direction(u_comp, v_comp).m if spd >= 1 else np.nan
            return drct, spd
        except Exception: return np.nan, np.nan

    dir_sfc, spd_sfc = get_wind_at_level(p.m[0])
    dir_925, spd_925 = get_wind_at_level(925)
    dir_700, spd_700 = get_wind_at_level(700)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(crear_dial_vent_animat("Superfície", dir_sfc, spd_sfc), unsafe_allow_html=True)
    with col2: st.markdown(crear_dial_vent_animat("925 hPa", dir_925, spd_925), unsafe_allow_html=True)
    with col3: st.markdown(crear_dial_vent_animat("700 hPa", dir_700, spd_700), unsafe_allow_html=True)

def ui_pestanya_vertical(data_tuple, poble_sel, lat, lon, nivell_conv, hora_actual, timestamp_str, avis_proximitat=None):
    """
    Versió Final amb Lògica de Context:
    - Comprova si la zona d'amenaça ja és la zona que s'està analitzant.
    - Si és així, mostra un botó desactivat amb un missatge informatiu.
    - Si no, mostra el botó interactiu per "viatjar" a la nova zona.
    """
    if data_tuple:
        sounding_data, params_calculats = data_tuple
        p, T, Td, u, v, heights, prof = sounding_data
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            zoom_capa_baixa = st.checkbox("🔍 Zoom a la Capa Baixa (Superfície - 800 hPa)")
            fig_skewt = crear_skewt(p, T, Td, u, v, prof, params_calculats, f"Sondeig Vertical - {poble_sel}", timestamp_str, zoom_capa_baixa=zoom_capa_baixa)
            st.pyplot(fig_skewt, use_container_width=True)
            plt.close(fig_skewt)
            with st.container(border=True):
                ui_caixa_parametres_sondeig(sounding_data, params_calculats, nivell_conv, hora_actual, poble_sel, avis_proximitat)

        with col2:
            fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat - {poble_sel}", timestamp_str)
            st.pyplot(fig_hodo, use_container_width=True)
            plt.close(fig_hodo)

            # <<-- NOU BLOC DE LÒGICA AMB COMPROVACIÓ DE CONTEXT -->>
            if avis_proximitat and isinstance(avis_proximitat, dict):
                # Sempre mostrem el missatge d'avís primer
                st.warning(f"⚠️ **AVÍS DE PROXIMITAT:** {avis_proximitat['message']}")
                
                # Comprovem si el millor punt d'anàlisi és el que ja estem veient
                if avis_proximitat['target_city'] == poble_sel:
                    # Si és així, mostrem un botó desactivat i informatiu
                    st.button("📍 Ja ets a la millor zona convergent d'anàlisi, mira si hi ha MU/SBCAPE! I poc MU/SBCIN!",
                              help="El punt d'anàlisi més proper a l'amenaça és la localitat que ja estàs consultant.",
                              use_container_width=True,
                              disabled=True)
                else:
                    # Si no, mostrem el botó interactiu de sempre
                    tooltip_text = f"Viatjar a {avis_proximitat['target_city']}, el punt d'anàlisi més proper al nucli de convergència (Força: {avis_proximitat['conv_value']:.0f})."
                    st.button("🛰️ Analitzar Zona d'Amenaça", 
                              help=tooltip_text, 
                              use_container_width=True, 
                              type="primary",
                              on_click=canviar_poble_analitzat,
                              args=(avis_proximitat['target_city'],)
                             )
            # <<-- FI DEL NOU BLOC -->>
            
            st.markdown("##### Radar de Precipitació en Temps Real")
            radar_url = f"https://www.rainviewer.com/map.html?loc={lat},{lon},10&oCS=1&c=3&o=83&lm=0&layer=radar&sm=1&sn=1&ts=2&play=1"
            html_code = f"""<div style="position: relative; width: 100%; height: 410px; border-radius: 10px; overflow: hidden;"><iframe src="{radar_url}" width="100%" height="410" frameborder="0" style="border:0;"></iframe><div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; cursor: default;"></div></div>"""
            st.components.v1.html(html_code, height=410)
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")

def debug_convergence_calculation(map_data, llista_ciutats):
    """
    Funció de depuració per imprimir l'estat dels càlculs de convergència pas a pas.
    Aquesta versió és sintàcticament correcta.
    """
    st.warning("⚠️ MODE DE DEPURACIÓ ACTIVAT. Revisa la terminal on has executat Streamlit.")
    print("\n\n" + "="*50)
    print("INICI DE LA DEPURACIÓ DE CONVERGÈNCIA")
    print("="*50)

    # --> INICI DEL BLOC TRY
    try:
        if not map_data or 'lons' not in map_data or len(map_data['lons']) < 4:
            print("[ERROR] No hi ha prou dades al `map_data` inicial.")
            print("="*50 + "\n\n")
            return {}

        print(f"[PAS 1] Dades d'entrada rebudes:")
        print(f"  - Punts de dades (lons/lats): {len(map_data['lons'])}")
        print(f"  - Claus disponibles: {list(map_data.keys())}")

        print("\n[PAS 2] Crida a la funció de càlcul real...")
        # Cridem la funció real per obtenir el resultat
        resultats = calcular_convergencies_per_llista(map_data, llista_ciutats)
        print("  - Càlcul completat sense errors.")
        
        print("\n[PAS 3] Verificant resultat per a Barcelona...")
        if 'Barcelona' in resultats:
            dades_bcn = resultats['Barcelona']
            valor_conv_bcn = dades_bcn.get('conv')
            es_humit_bcn = dades_bcn.get('es_humit')
            print(f"  - VALOR DE CONVERGÈNCIA PER A BCN: {valor_conv_bcn}")
            print(f"  - ÉS HUMIT A BCN?: {es_humit_bcn}")
        else:
            print("  - [ERROR] No s'han trobat resultats per a Barcelona.")
        
        print("="*50 + "\nFI DE LA DEPURACIÓ\n" + "="*50 + "\n\n")

        return resultats

    # --> BLOC EXCEPT CORRESPONENT I CORRECTAMENT INDENTAT
    except Exception as e:
        print(f"[ERROR CRÍTIC] Excepció durant la depuració: {e}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n\n")
        return {}
    



@st.cache_data(ttl=1800, show_spinner="Analitzant zones de convergència...")
def calcular_convergencies_per_llista(map_data, llista_ciutats):
    """
    Analitza el mapa de dades per trobar el valor MÀXIM de convergència
    en un radi proper a cada ciutat de la llista. Aquesta funció s'utilitza
    principalment per a les dades de resolució més baixa (GFS/Tornado Alley).
    """
    if not map_data or 'lons' not in map_data or len(map_data['lons']) < 4:
        return {}

    convergencies = {}
    try:
        lons, lats = map_data['lons'], map_data['lats']
        speed_data, dir_data = map_data['speed_data'], map_data['dir_data']

        # Creem una graella d'alta resolució per a l'anàlisi
        grid_lon, grid_lat = np.meshgrid(
            np.linspace(min(lons), max(lons), 200),
            np.linspace(min(lats), max(lats), 200)
        )
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
        
        with np.errstate(invalid='ignore'):
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            convergence_scaled = -mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy).to('1/s').magnitude * 1e5
            convergence_scaled[np.isnan(convergence_scaled)] = 0

        # Lògica d'anàlisi per àrea
        SEARCH_RADIUS_DEG = 0.5  # Radi de cerca més gran per a GFS

        for nom_ciutat, coords in llista_ciutats.items():
            lat_sel, lon_sel = coords['lat'], coords['lon']
            
            dist_from_city = np.sqrt((grid_lat - lat_sel)**2 + (grid_lon - lon_sel)**2)
            nearby_mask = dist_from_city <= SEARCH_RADIUS_DEG
            
            if np.any(nearby_mask):
                # Ens quedem amb el valor MÀXIM de convergència dins d'aquesta àrea.
                max_conv_in_area = np.max(convergence_scaled[nearby_mask])
                convergencies[nom_ciutat] = max_conv_in_area
            else:
                convergencies[nom_ciutat] = 0
    
    except Exception as e:
        print(f"Error crític a calcular_convergencies_per_llista: {e}")
        return {}
        
    return convergencies


@st.cache_data(ttl=1800)
def calcular_convergencia_per_llista_poblacions(hourly_index, poblacions_dict, nivell):
    """
    Calcula la convergència per a una llista de poblacions a un nivell específic.
    *** VERSIÓ CORREGIDA I DEFINITIVA: Accepta 3 paràmetres. ***
    """
    if not poblacions_dict:
        return {}

    # Utilitzem el 'nivell' que rebem com a paràmetre
    map_data, error = carregar_dades_mapa_cat(nivell, hourly_index)
    if error or not map_data:
        return {}

    try:
        lons, lats = map_data['lons'], map_data['lats']
        grid_lon, grid_lat = np.meshgrid(np.linspace(min(lons), max(lons), 100), np.linspace(min(lats), max(lats), 100))
        
        u_comp, v_comp = mpcalc.wind_components(np.array(map_data['speed_data']) * units('km/h'), np.array(map_data['dir_data']) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
        
        with np.errstate(invalid='ignore'):
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            convergence_scaled = -mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy).to('1/s').magnitude * 1e5
        
        resultats = {}
        for nom, coords in poblacions_dict.items():
            valor = griddata((grid_lon.flatten(), grid_lat.flatten()), convergence_scaled.flatten(), (coords['lon'], coords['lat']), method='cubic')
            if pd.notna(valor):
                resultats[nom] = valor
        
        return resultats
    except Exception as e:
        print(f"Error dins de calcular_convergencia_per_llista_poblacions: {e}")
        return {}



def ui_explicacio_convergencia():
    """
    Crea una secció explicativa visualment atractiva sobre els dos tipus
    principals de convergència, utilitzant un disseny de targetes.
    """
    st.divider()
    st.markdown("##### Com Interpretar els Nuclis de Convergència")

    # CSS per a l'estil de les targetes explicatives
    st.markdown("""
    <style>
    .explanation-card {
        background-color: #f0f2f6; /* Fons clar per a les targetes */
        border: 1px solid #d1d1d1;
        border-radius: 10px;
        padding: 20px;
        height: 100%; /* Assegura que les dues targetes tinguin la mateixa alçada */
        display: flex;
        flex-direction: column;
    }
    .explanation-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #1a1a2e; /* Text fosc */
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .explanation-icon {
        font-size: 1.5em;
        margin-right: 12px;
    }
    .explanation-text {
        font-size: 1em;
        color: #333; /* Text gris fosc */
        line-height: 1.6;
    }
    .explanation-text strong {
        color: #0056b3; /* Ressalta paraules clau en blau */
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="explanation-card">
            <div class="explanation-title">
                <span class="explanation-icon">💥</span>
                Convergència Frontal (Xoc)
            </div>
            <div class="explanation-text">
                Passa quan <strong>dues masses d'aire de direccions diferents xoquen</strong>. L'aire no pot anar cap als costats i es veu forçat a ascendir bruscament.
                <br><br>
                <strong>Al mapa:</strong> Busca línies on les <i>streamlines</i> (línies de vent) es troben de cara, com en un "xoc de trens". Són mecanismes de dispar molt eficients i solen generar tempestes organitzades.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="explanation-card">
            <div class="explanation-title">
                <span class="explanation-icon">⛰️</span>
                Convergència per Acumulació
            </div>
            <div class="explanation-text">
                Ocorre quan el vent es troba amb un <strong>obstacle (com una muntanya) o es desaccelera</strong>, fent que l'aire "s'amuntegui". L'única sortida per a aquesta acumulació de massa és cap amunt.
                <br><br>
                <strong>Al mapa:</strong> Busca zones on les <i>streamlines</i> s'ajunten i la velocitat del vent (color de fons) disminueix. És com un "embús a l'autopista": els cotxes s'acumulen i s'aturen.
            </div>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def carregar_dades_mapa_base_cat(variables, hourly_index):
    try:
        # --- CANVI CLAU: AUGMENTEM LA RESOLUCIÓ DE LA PETICIÓ ---
        # Passem de 12x12 a 40x40 punts. Això és crucial per a un mapa de qualitat.
        lats, lons = np.linspace(MAP_EXTENT_CAT[2], MAP_EXTENT_CAT[3], 40), np.linspace(MAP_EXTENT_CAT[0], MAP_EXTENT_CAT[1], 40)
        
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": 4}
        responses = openmeteo.weather_api(API_URL_CAT, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            try:
                vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
                if not any(np.isnan(v) for v in vals):
                    output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                    for i, var in enumerate(variables): output[var].append(vals[i])
            except IndexError:
                continue # Si l'hora no existeix per aquest punt, el saltem
                
        if not output["lats"]: return None, "No s'han rebut dades vàlides."
        return output, None
    except Exception as e: return None, f"Error en carregar dades del mapa: {e}"
        
        

@st.cache_data(ttl=3600)
def carregar_dades_mapa_cat(nivell, hourly_index):
    try:
        # Assegurem que sempre demanem la temperatura de superfície
        variables_base = ["temperature_2m", "dew_point_2m"]
        
        if nivell >= 950:
            variables = variables_base + [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base_cat(variables, hourly_index)
            if error: return None, error
            map_data_raw['dewpoint_data'] = map_data_raw.pop('dew_point_2m')
            map_data_raw['temperature_data'] = map_data_raw.pop('temperature_2m')
        else:
            variables = variables_base + [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base_cat(variables, hourly_index)
            if error: return None, error
            
            # Usem les dades del nivell per a la visualització, però guardem les de superfície per a l'anàlisi
            temp_nivell = np.array(map_data_raw.pop(f'temperature_{nivell}hPa')) * units.degC
            rh_nivell = np.array(map_data_raw.pop(f'relative_humidity_{nivell}hPa')) * units.percent
            map_data_raw['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_nivell, rh_nivell).m
            map_data_raw['temperature_data'] = map_data_raw.pop('temperature_2m') # Guardem la de superfície
            map_data_raw.pop('dew_point_2m') # La de superfície ja no la necessitem aquí

        map_data_raw['speed_data'] = map_data_raw.pop(f'wind_speed_{nivell}hPa')
        map_data_raw['dir_data'] = map_data_raw.pop(f'wind_direction_{nivell}hPa')
        return map_data_raw, None
    except Exception as e:
        return None, f"Error en processar dades del mapa: {e}"
    
    
def afegir_etiquetes_ciutats(ax, map_extent):
    """
    Versió amb etiquetes més petites per a una millor claredat visual en fer zoom.
    """
    is_zoomed_in = (tuple(map_extent) != tuple(MAP_EXTENT_CAT))

    if is_zoomed_in:
        # Itera sobre el diccionari de referència per als mapes
        for ciutat, coords in POBLES_MAPA_REFERENCIA.items():
            lon, lat = coords['lon'], coords['lat']
            
            # Comprovem si el punt de referència està dins dels límits del mapa visible
            if map_extent[0] < lon < map_extent[1] and map_extent[2] < lat < map_extent[3]:
                
                # Dibuixem el punt de referència
                ax.plot(lon, lat, 'o', color='black', markersize=1,
                        markeredgecolor='black', markeredgewidth=1.5,
                        transform=ccrs.PlateCarree(), zorder=19)

                # Dibuixem l'etiqueta de text al costat del punt
                # <<-- CANVI CLAU: Hem reduït el 'fontsize' de 8 a 6 -->>
                ax.text(lon + 0.02, lat, ciutat, 
                        fontsize= 5, # <-- AQUÍ ESTÀ EL CANVI
                        color='white',
                        transform=ccrs.PlateCarree(), 
                        zorder=2,
                        path_effects=[path_effects.withStroke(linewidth=2.5, foreground='gray')])







@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_holanda(lat, lon, hourly_index):
    """
    Versió Corregida i Robusta: Carrega dades per a Holanda sense utilitzar
    el mètode .NumberOfVariables() per ser compatible amb la resposta de l'API del model KNMI.
    """
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        press_vars = ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"]
        h_press = [f"{v}_{p}hPa" for v in press_vars for p in PRESS_LEVELS_HOLANDA]
        
        # Llista completa de totes les variables que estem demanant
        all_requested_vars = h_base + h_press
        
        params = {"latitude": lat, "longitude": lon, "hourly": all_requested_vars, "models": "knmi_harmonie_arome_europe", "forecast_days": 2}
        
        response = openmeteo.weather_api(API_URL_HOLANDA, params=params)[0]
        hourly = response.Hourly()

        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None:
            return None, hourly_index, "No s'han trobat dades vàlides properes a l'hora sol·licitada."

        # <<<--- BLOC DE LECTURA CORREGIT --->>>
        # Construïm el diccionari iterant sobre la nostra pròpia llista de variables sol·licitades.
        # Això no depèn de .NumberOfVariables() i és, per tant, compatible.
        hourly_vars = {}
        for i, var_name in enumerate(all_requested_vars):
            try:
                hourly_vars[var_name] = hourly.Variables(i).ValuesAsNumpy()
            except Exception:
                # Si una variable no es pot llegir, la marquem com a buida per evitar errors
                hourly_vars[var_name] = np.array([np.nan]) 

        # La resta de la funció pot continuar igual, ja que depèn de 'hourly_vars',
        # que ara es construeix correctament.
        sfc_data = {v: hourly_vars[v][valid_index] for v in h_base}
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for p_val in PRESS_LEVELS_HOLANDA:
            var_names_level = [f"{v}_{p_val}hPa" for v in press_vars]
            if p_val < p_profile[-1] and all(f in hourly_vars and not np.isnan(hourly_vars[f][valid_index]) for f in var_names_level):
                p_profile.append(p_val)
                T_profile.append(hourly_vars[f'temperature_{p_val}hPa'][valid_index])
                Td_profile.append(hourly_vars[f'dew_point_{p_val}hPa'][valid_index])
                u, v = mpcalc.wind_components(hourly_vars[f'wind_speed_{p_val}hPa'][valid_index] * units('km/h'), hourly_vars[f'wind_direction_{p_val}hPa'][valid_index] * units.degrees)
                u_profile.append(u.to('m/s').m)
                v_profile.append(v.to('m/s').m)
                h_profile.append(hourly_vars[f'geopotential_height_{p_val}hPa'][valid_index])

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        import traceback
        traceback.print_exc()
        return None, hourly_index, f"Error crític en carregar dades del sondeig d'Holanda: {e}"
    
@st.cache_data(ttl=3600)
def carregar_dades_mapa_holanda(nivell, hourly_index):
    try:
        variables = [f"temperature_{nivell}hPa", f"dew_point_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        lats, lons = np.linspace(MAP_EXTENT_HOLANDA[2], MAP_EXTENT_HOLANDA[3], 12), np.linspace(MAP_EXTENT_HOLANDA[0], MAP_EXTENT_HOLANDA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "knmi_harmonie_arome_europe", "forecast_days": 2}
        
        responses = openmeteo.weather_api(API_URL_HOLANDA, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        
        for r in responses:
            try:
                vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
                if not any(np.isnan(v) for v in vals):
                    output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                    for i, var in enumerate(variables): output[var].append(vals[i])
            except IndexError: continue

        if not output["lats"]: return None, "No s'han rebut dades."
        
        # El model ja ens dona dew_point, així que només reanomenem les claus
        output['dewpoint_data'] = output.pop(f'dew_point_{nivell}hPa')
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')
        # Eliminem la temperatura que no farem servir per al mapa de convergència
        del output[f'temperature_{nivell}hPa']

        return output, None
    except Exception as e: 
        return None, f"Error en carregar dades del mapa KNMI: {e}"

def crear_mapa_forecast_combinat_holanda(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base(MAP_EXTENT_HOLANDA, projection=ccrs.LambertConformal(central_longitude=5, central_latitude=52))
    if len(lons) < 4: return fig
    # La resta del codi de dibuix és idèntic al d'Alemanya o Itàlia, així que el podem reutilitzar
    # ... (Codi d'interpolació, pcolormesh, streamplot, convergència, etc.) ...
    # Per brevetat, el resultat visual serà el mateix, adaptat a les noves coordenades i dades
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_HOLANDA[0], MAP_EXTENT_HOLANDA[1], 200), np.linspace(MAP_EXTENT_HOLANDA[2], MAP_EXTENT_HOLANDA[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'); grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'); grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat); convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    convergence_in_humid_areas = np.where(grid_dewpoint >= 12, convergence_scaled, 0)
    fill_levels = [5, 10, 15, 25]; fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')
    for city, coords in CIUTATS_HOLANDA.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.1, coords['lat'] + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def ui_pestanya_mapes_holanda(hourly_index_sel, timestamp_str, nivell_sel, poble_sel):
    """
    Versió Millorada: Afegeix un spinner amb un missatge informatiu
    mentre es carreguen les dades i es genera el mapa.
    """
    st.markdown("#### Mapes de Pronòstic (Model KNMI Harmonie AROME)")

    # <<<--- SPINNER AFEGIT AQUÍ --->>>
    # Aquest bloc mostrarà el missatge mentre s'executa tot el que hi ha a dins.
    with st.spinner("Carregant mapa KNMI AROME... El primer cop pot trigar una mica."):
        map_data, error = carregar_dades_mapa_holanda(nivell_sel, hourly_index_sel)
    
        if error: 
            # Si hi ha un error, es mostrarà fora del spinner.
            st.error(f"Error en carregar el mapa: {error}")
        elif map_data:
            # Creem un títol net per al mapa
            timestamp_map_title = timestamp_str.replace(f"{poble_sel} | ", "")
            
            # Generem i mostrem la figura
            fig = crear_mapa_forecast_combinat_holanda(
                map_data['lons'], map_data['lats'], 
                map_data['speed_data'], map_data['dir_data'], 
                map_data['dewpoint_data'], nivell_sel, 
                timestamp_map_title
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else: 
            st.warning("No s'han pogut obtenir les dades per generar el mapa.")
                
@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_japo(lat, lon, hourly_index):
    """
    Versió Final i Definitiva: Utilitza el model global 'jma_gsm' i demana 'relative_humidity'
    per assegurar la màxima completesa de dades. Construeix un perfil tolerant que permet
    forats de dades en variables no essencials per dibuixar el perfil complet.
    """
    try:
        # Estratègia més segura: demanar 'relative_humidity' que sol ser més completa.
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        press_vars = ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"]
        h_press = [f"{v}_{p}hPa" for v in press_vars for p in PRESS_LEVELS_JAPO]
        all_requested_vars = h_base + h_press
        
        params = {"latitude": lat, "longitude": lon, "hourly": all_requested_vars, "models": "jma_gsm", "forecast_days": 3}
        
        response = openmeteo.weather_api(API_URL_JAPO, params=params)[0]
        hourly = response.Hourly()

        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None: 
            return None, hourly_index, "No s'han trobat dades vàlides."

        hourly_vars = {}
        for i, var_name in enumerate(all_requested_vars):
            try: hourly_vars[var_name] = hourly.Variables(i).ValuesAsNumpy()
            except Exception: hourly_vars[var_name] = np.array([np.nan] * len(hourly.Variables(0).ValuesAsNumpy()))

        sfc_data = {v: hourly_vars[v][valid_index] for v in h_base}
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        # Bucle de construcció tolerant: afegim el nivell i després les dades que trobem.
        for p_val in PRESS_LEVELS_JAPO:
            if p_val < p_profile[-1]:
                p_profile.append(p_val)
                
                temp = hourly_vars.get(f'temperature_{p_val}hPa', [np.nan])[valid_index]
                rh = hourly_vars.get(f'relative_humidity_{p_val}hPa', [np.nan])[valid_index]
                ws = hourly_vars.get(f'wind_speed_{p_val}hPa', [np.nan])[valid_index]
                wd = hourly_vars.get(f'wind_direction_{p_val}hPa', [np.nan])[valid_index]
                h = hourly_vars.get(f'geopotential_height_{p_val}hPa', [np.nan])[valid_index]
                
                T_profile.append(temp)
                h_profile.append(h)
                
                # Calculem dew_point si tenim T i RH, sinó NaN.
                if pd.notna(temp) and pd.notna(rh):
                    Td_profile.append(mpcalc.dewpoint_from_relative_humidity(temp * units.degC, rh * units.percent).m)
                else:
                    Td_profile.append(np.nan)
                
                # Calculem vent si tenim WS i WD, sinó NaN.
                if pd.notna(ws) and pd.notna(wd):
                    u, v = mpcalc.wind_components(ws * units('km/h'), wd * units.degrees)
                    u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m)
                else:
                    u_profile.append(np.nan); v_profile.append(np.nan)
        
        # Passem el perfil complet (amb possibles forats) a la funció de processament.
        # Aquesta s'encarregarà de netejar-lo abans dels càlculs i el dibuix.
        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        import traceback
        traceback.print_exc()
        return None, hourly_index, f"Error crític en carregar dades del sondeig del Japó: {e}"
    


@st.cache_data(ttl=3600)
def carregar_dades_mapa_uk(nivell, hourly_index):
    """
    Carrega les dades en una graella per al mapa del Regne Unit utilitzant el
    model d'alta resolució UKMO de 2km.
    """
    try:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        lats, lons = np.linspace(MAP_EXTENT_UK[2], MAP_EXTENT_UK[3], 12), np.linspace(MAP_EXTENT_UK[0], MAP_EXTENT_UK[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "ukmo_uk_deterministic_2km", "forecast_days": 2}
        
        responses = openmeteo.weather_api(API_URL_UK, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        
        for r in responses:
            try:
                valid_index = trobar_hora_valida_mes_propera(r.Hourly(), hourly_index, len(variables))
                if valid_index is not None:
                    vals = [r.Hourly().Variables(i).ValuesAsNumpy()[valid_index] for i in range(len(variables))]
                    if not any(np.isnan(v) for v in vals):
                        output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                        for i, var in enumerate(variables): output[var].append(vals[i])
            except IndexError: 
                continue

        if not output["lats"]: 
            return None, "No s'han rebut dades per a la graella del mapa."
        
        temp_data = np.array(output.pop(f'temperature_{nivell}hPa')) * units.degC
        rh_data = np.array(output.pop(f'relative_humidity_{nivell}hPa')) * units.percent
        output['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')

        return output, None
    except Exception as e: 
        return None, f"Error en carregar dades del mapa UKMO: {e}"
    

def crear_mapa_forecast_combinat_uk(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    """
    Crea el mapa visual de vent i convergència per al Regne Unit i Irlanda.
    """
    fig, ax = crear_mapa_base(MAP_EXTENT_UK, projection=ccrs.LambertConformal(central_longitude=-4.5, central_latitude=54))
    if len(lons) < 4: return fig

    # Interpolació de dades
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_UK[0], MAP_EXTENT_UK[1], 200), np.linspace(MAP_EXTENT_UK[2], MAP_EXTENT_UK[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    
    # Dibuix del vent (fons de color i línies de corrent)
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    # Càlcul i dibuix de la convergència
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    convergence_in_humid_areas = np.where(grid_dewpoint >= 12, convergence_scaled, 0) # Llindar de punt de rosada a 12°C per al clima atlàntic
    
    fill_levels = [5, 10, 15, 25]; fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')

    # Etiquetes de ciutats
    for city, coords in CIUTATS_UK.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.1, coords['lat'] + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig


@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_uk(lat, lon, hourly_index):
    """
    Carrega dades de sondeig per al Regne Unit utilitzant el model d'alta
    resolució UKMO de 2km, gestionant el seu gran detall vertical.
    """
    try:
        # Nota: La teva petició de prova demanava 'surface_pressure' a 'current', però la necessitem a 'hourly'.
        h_base = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        press_vars = ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height", "vertical_velocity"]
        h_press = [f"{v}_{p}hPa" for v in press_vars for p in PRESS_LEVELS_UK]
        all_requested_vars = h_base + h_press
        
        params = {"latitude": lat, "longitude": lon, "hourly": all_requested_vars, "models": "ukmo_uk_deterministic_2km", "forecast_days": 2}
        
        response = openmeteo.weather_api(API_URL_UK, params=params)[0]
        hourly = response.Hourly()

        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None: return None, hourly_index, "No s'han trobat dades vàlides."

        hourly_vars = {}
        for i, var_name in enumerate(all_requested_vars):
            try: hourly_vars[var_name] = hourly.Variables(i).ValuesAsNumpy()
            except Exception: hourly_vars[var_name] = np.array([np.nan])
        
        sfc_data = {v: hourly_vars[v][valid_index] for v in h_base}
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [mpcalc.dewpoint_from_relative_humidity(sfc_data["temperature_2m"] * units.degC, sfc_data["relative_humidity_2m"] * units.percent).m], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for p_val in PRESS_LEVELS_UK:
            if p_val < p_profile[-1]:
                p_profile.append(p_val)
                temp = hourly_vars.get(f'temperature_{p_val}hPa', [np.nan])[valid_index]
                rh = hourly_vars.get(f'relative_humidity_{p_val}hPa', [np.nan])[valid_index]
                ws = hourly_vars.get(f'wind_speed_{p_val}hPa', [np.nan])[valid_index]
                wd = hourly_vars.get(f'wind_direction_{p_val}hPa', [np.nan])[valid_index]
                h = hourly_vars.get(f'geopotential_height_{p_val}hPa', [np.nan])[valid_index]
                
                T_profile.append(temp); h_profile.append(h)
                
                if pd.notna(temp) and pd.notna(rh): Td_profile.append(mpcalc.dewpoint_from_relative_humidity(temp * units.degC, rh * units.percent).m)
                else: Td_profile.append(np.nan)
                
                if pd.notna(ws) and pd.notna(wd):
                    u, v = mpcalc.wind_components(ws * units('km/h'), wd * units.degrees)
                    u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m)
                else:
                    u_profile.append(np.nan); v_profile.append(np.nan)

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        import traceback; traceback.print_exc()
        return None, hourly_index, f"Error crític en carregar dades del sondeig del Regne Unit: {e}"
        

def ui_pestanya_satelit_japo():
    st.markdown("#### Imatge de Satèl·lit Himawari-9 (Temps Real)")
    # URL del satèl·lit geoestacionari del Japó
    sat_url = f"https://www.data.jma.go.jp/mscweb/data/himawari/img/fd/fd_P_00.jpg?{int(time.time())}"
    st.image(sat_url, caption="Imatge del satèl·lit Himawari-9 - Disc Complet (JMA)", use_container_width=True)
    st.info("Aquesta imatge del satèl·lit japonès s'actualitza cada 10 minuts.")
    st.markdown("<p style='text-align: center;'>[Font: Japan Meteorological Agency (JMA)](https://www.data.jma.go.jp/mscweb/data/himawari/index.html)</p>", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dades_mapa_base_cat(variables, hourly_index):
    """
    Versió única i correcta. Funció base per carregar dades del model AROME.
    """
    try:
        lats, lons = np.linspace(MAP_EXTENT_CAT[2], MAP_EXTENT_CAT[3], 12), np.linspace(MAP_EXTENT_CAT[0], MAP_EXTENT_CAT[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": 4}
        responses = openmeteo.weather_api(API_URL_CAT, params=params)
        
        output = {var: [] for var in ["lats", "lons"] + variables}
        
        for r in responses:
            try:
                # Agafem les dades per a l'índex horari sol·licitat
                vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
                # Només afegim el punt si TOTES les dades per a aquesta hora són vàlides
                if not any(np.isnan(v) for v in vals):
                    output["lats"].append(r.Latitude())
                    output["lons"].append(r.Longitude())
                    for i, var in enumerate(variables):
                        output[var].append(vals[i])
            except IndexError:
                # Si l'índex horari està fora de rang per a aquest punt, el saltem
                continue
                
        if not output["lats"]:
            return None, "No s'han rebut dades vàlides per a l'hora seleccionada."
            
        return output, None
    except Exception as e:
        return None, f"Error en carregar dades del mapa: {e}"




@st.cache_data(ttl=1800, max_entries=20, show_spinner=False) # TTL més curt (30min) ja que el HRDPS s'actualitza sovint
def carregar_dades_sondeig_canada(lat, lon, hourly_index):
    """
    Carrega dades de sondeig per al Canadà utilitzant el model d'alta
    resolució HRDPS (gem_hrdps_continental).
    """
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        press_vars = ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height", "vertical_velocity"]
        h_press = [f"{v}_{p}hPa" for v in press_vars for p in PRESS_LEVELS_CANADA]
        all_requested_vars = h_base + h_press
        
        params = {"latitude": lat, "longitude": lon, "hourly": all_requested_vars, "models": "gem_hrdps_continental", "forecast_days": 2}
        
        response = openmeteo.weather_api(API_URL_CANADA, params=params)[0]
        hourly = response.Hourly()

        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None: return None, hourly_index, "No s'han trobat dades vàlides."

        hourly_vars = {}
        for i, var_name in enumerate(all_requested_vars):
            try: hourly_vars[var_name] = hourly.Variables(i).ValuesAsNumpy()
            except Exception: hourly_vars[var_name] = np.array([np.nan])
        
        sfc_data = {v: hourly_vars[v][valid_index] for v in h_base}
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for p_val in PRESS_LEVELS_CANADA:
            if p_val < p_profile[-1]:
                p_profile.append(p_val)
                T_profile.append(hourly_vars.get(f'temperature_{p_val}hPa', [np.nan])[valid_index])
                Td_profile.append(hourly_vars.get(f'dew_point_{p_val}hPa', [np.nan])[valid_index])
                h_profile.append(hourly_vars.get(f'geopotential_height_{p_val}hPa', [np.nan])[valid_index])
                ws = hourly_vars.get(f'wind_speed_{p_val}hPa', [np.nan])[valid_index]
                wd = hourly_vars.get(f'wind_direction_{p_val}hPa', [np.nan])[valid_index]
                if pd.notna(ws) and pd.notna(wd):
                    u, v = mpcalc.wind_components(ws * units('km/h'), wd * units.degrees)
                    u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m)
                else:
                    u_profile.append(np.nan); v_profile.append(np.nan)

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        import traceback; traceback.print_exc()
        return None, hourly_index, f"Error crític en carregar dades del sondeig del Canadà: {e}"

@st.cache_data(ttl=1800)
def carregar_dades_mapa_canada(nivell, hourly_index):
    """
    Carrega dades de mapa per al Canadà utilitzant el model HRDPS.
    """
    try:
        variables = [f"temperature_{nivell}hPa", f"dew_point_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        lats, lons = np.linspace(MAP_EXTENT_CANADA[2], MAP_EXTENT_CANADA[3], 12), np.linspace(MAP_EXTENT_CANADA[0], MAP_EXTENT_CANADA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "gem_hrdps_continental", "forecast_days": 2}
        
        responses = openmeteo.weather_api(API_URL_CANADA, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            try:
                valid_index = trobar_hora_valida_mes_propera(r.Hourly(), hourly_index, len(variables))
                if valid_index is not None:
                    vals = [r.Hourly().Variables(i).ValuesAsNumpy()[valid_index] for i in range(len(variables))]
                    if not any(np.isnan(v) for v in vals):
                        output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                        for i, var in enumerate(variables): output[var].append(vals[i])
            except IndexError: continue

        if not output["lats"]: return None, "No s'han rebut dades per a la graella del mapa."
        
        output['dewpoint_data'] = output.pop(f'dew_point_{nivell}hPa')
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')
        del output[f'temperature_{nivell}hPa']

        return output, None
    except Exception as e:
        return None, f"Error en carregar dades del mapa HRDPS: {e}"

def crear_mapa_forecast_combinat_canada(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base(MAP_EXTENT_CANADA, projection=ccrs.LambertConformal(central_longitude=-105, central_latitude=54))
    if len(lons) < 4: return fig
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_CANADA[0], MAP_EXTENT_CANADA[1], 200), np.linspace(MAP_EXTENT_CANADA[2], MAP_EXTENT_CANADA[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'); grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'); grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat); convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    convergence_in_humid_areas = np.where(grid_dewpoint >= 5, convergence_scaled, 0) # Llindar de punt de rosada més baix per a climes més freds
    
    fill_levels = [5, 10, 15, 25]; fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')

    for city, coords in CIUTATS_CANADA.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.1, coords['lat'] + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def ui_pestanya_mapes_canada(hourly_index_sel, timestamp_str, nivell_sel, poble_sel):
    st.markdown("#### Mapes de Pronòstic (Model HRDPS)")
    with st.spinner("Carregant mapa HRDPS... El primer cop pot trigar una mica."):
        map_data, error = carregar_dades_mapa_canada(nivell_sel, hourly_index_sel)
    
        if error or not map_data:
            # <<<--- CORRECCIÓ AQUÍ --->>>
            st.error(f"Error en carregar el mapa: {error if error else 'No s''han rebut dades.'}")
        else:
            timestamp_map_title = timestamp_str.replace(f"{poble_sel} | ", "")
            fig = crear_mapa_forecast_combinat_canada(
                map_data['lons'], map_data['lats'], map_data['speed_data'],
                map_data['dir_data'], map_data['dewpoint_data'], nivell_sel,
                timestamp_map_title
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

@st.cache_data(ttl=3600)
def carregar_dades_mapa_cat(nivell, hourly_index):
    """
    Versió definitiva. Assegura que SEMPRE es demanen les dades de superfície
    (T i Td) per a l'anàlisi d'humitat, independentment del nivell seleccionat.
    """
    try:
        # Llista base de variables que SEMPRE necessitem per a l'anàlisi
        variables_base = ["temperature_2m", "dew_point_2m"]
        
        # Variables específiques del nivell seleccionat per al mapa
        variables_nivell = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        if nivell < 950:
            variables_nivell.extend([f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa"])

        # Demanem totes les variables juntes
        map_data_raw, error = carregar_dades_mapa_base_cat(variables_base + variables_nivell, hourly_index)
        if error: return None, error

        # Guardem les dades de superfície amb noms clars per a la funció de convergència
        map_data_raw['sfc_temp_data'] = map_data_raw.pop('temperature_2m')
        map_data_raw['sfc_dewpoint_data'] = map_data_raw.pop('dew_point_2m')

        # Processem les dades per a la VISUALITZACIÓ del mapa
        if nivell >= 950:
            map_data_raw['dewpoint_data'] = map_data_raw['sfc_dewpoint_data']
        else:
            temp_nivell = np.array(map_data_raw.pop(f'temperature_{nivell}hPa')) * units.degC
            rh_nivell = np.array(map_data_raw.pop(f'relative_humidity_{nivell}hPa')) * units.percent
            map_data_raw['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_nivell, rh_nivell).m

        map_data_raw['speed_data'] = map_data_raw.pop(f'wind_speed_{nivell}hPa')
        map_data_raw['dir_data'] = map_data_raw.pop(f'wind_direction_{nivell}hPa')
        
        return map_data_raw, None
    except Exception as e:
        return None, f"Error en processar dades del mapa: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_cat(nivell, hourly_index):
    try:
        if nivell >= 950:
            variables = ["dew_point_2m", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base_cat(variables, hourly_index)
            if error: return None, error
            map_data_raw['dewpoint_data'] = map_data_raw.pop('dew_point_2m')
        else:
            variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
            map_data_raw, error = carregar_dades_mapa_base_cat(variables, hourly_index)
            if error: return None, error
            temp_data = np.array(map_data_raw.pop(f'temperature_{nivell}hPa')) * units.degC
            rh_data = np.array(map_data_raw.pop(f'relative_humidity_{nivell}hPa')) * units.percent
            map_data_raw['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m

        map_data_raw['speed_data'] = map_data_raw.pop(f'wind_speed_{nivell}hPa')
        map_data_raw['dir_data'] = map_data_raw.pop(f'wind_direction_{nivell}hPa')
        return map_data_raw, None
    except Exception as e:
        return None, f"Error en processar dades del mapa: {e}"
        
@st.cache_data(ttl=1800, max_entries=5, show_spinner=False)
def obtenir_ciutats_actives(hourly_index):
    """
    Versión optimizada con muestreo reducido
    """
    nivell = 925
    map_data, error_map = carregar_dades_mapa_cat(nivell, hourly_index)
    if error_map or not map_data: 
        return CIUTATS_CONVIDAT, "No s'ha pogut determinar les zones de convergència."
    
    try:
        # Reducir resolución para cálculo más rápido
        lons, lats = np.array(map_data['lons']), np.array(map_data['lats'])
        dewpoint_data = np.array(map_data['dewpoint_data'])
        
        # Muestreo para mayor velocidad (máximo 20 puntos)
        if len(lons) > 20:
            idxs = np.random.choice(len(lons), size=min(20, len(lons)), replace=False)
            lons, lats, dewpoint_data = lons[idxs], lats[idxs], dewpoint_data[idxs]
        
        # Búsqueda eficiente de ciudades con alta humedad
        ciudades_activas = []
        umbral_humedad = 12  # Punto de rocío mínimo
        
        for ciutat, coords in CIUTATS_CATALUNYA.items():
            # Calcular distancia a todos los puntos
            distancias = np.sqrt((lats - coords['lat'])**2 + (lons - coords['lon'])**2)
            idx_mas_cercano = np.argmin(distancias)
            
            if (distancias[idx_mas_cercano] < 0.3 and  # Menos de 0.3 grados de distancia
                dewpoint_data[idx_mas_cercano] >= umbral_humedad):
                ciudades_activas.append(ciutat)
        
        # Limitar a 6 ciudades máximo para no saturar
        if ciudades_activas:
            return {name: CIUTATS_CATALUNYA[name] for name in ciudades_activas[:6]}, "Zones actives detectades"
        else:
            return CIUTATS_CONVIDAT, "No s'han detectat zones de convergència significatives."
            
    except Exception as e:
        return CIUTATS_CONVIDAT, f"Error calculant zones actives: {e}"

@st.cache_resource(show_spinner=False)
def precache_datos_iniciales():
    """
    Pre-cache de datos comunes al iniciar la aplicación
    """
    try:
        # Pre-cargar datos que probablemente se usarán
        now_local = datetime.now(TIMEZONE_CAT)
        hourly_index = int((now_local.astimezone(pytz.utc).replace(minute=0, second=0, microsecond=0) - 
                          datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        
        # Pre-cache de ciudades principales
        ciudades_principales = ['Barcelona', 'Girona', 'Lleida', 'Tarragona']
        for ciutat in ciudades_principales:
            coords = CIUTATS_CATALUNYA[ciutat]
            carregar_dades_sondeig_cat(coords['lat'], coords['lon'], hourly_index)
        
        # Pre-cache de mapa básico
        carregar_dades_mapa_cat(925, hourly_index)
        
        return True
    except Exception as e:
        print(f"Pre-caching falló: {e}")
        return False



@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_italia(lat, lon, hourly_index):
    """
    Carrega i processa dades de sondeig per a Itàlia utilitzant el model
    d'alta resolució 'italia_meteo_arpae_icon_2i'.
    Inclou la càrrega de la variable 'vertical_velocity'.
    """
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        press_vars = ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height", "vertical_velocity"]
        h_press = [f"{v}_{p}hPa" for v in press_vars for p in PRESS_LEVELS_ITALIA]
        
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "italia_meteo_arpae_icon_2i", "forecast_days": 2}
        
        response = openmeteo.weather_api(API_URL_ITALIA, params=params)[0]
        hourly = response.Hourly()

        # <<<--- LÍNIA CORREGIDA: Crida a la nova funció auxiliar --->>>
        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        
        if valid_index is None:
            return None, hourly_index, "No s'han trobat dades vàlides properes a l'hora sol·licitada."

        # La resta de la funció es manté igual
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[valid_index] for i, v in enumerate(h_base)}
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(press_vars):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_ITALIA) + j).ValuesAsNumpy()[valid_index] for j in range(len(PRESS_LEVELS_ITALIA))]

        for i, p_val in enumerate(PRESS_LEVELS_ITALIA):
            if p_val < p_profile[-1] and all(not np.isnan(p_data[v][i]) for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"]):
                p_profile.append(p_val)
                T_profile.append(p_data["temperature"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["temperature"][i] * units.degC, p_data["relative_humidity"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["wind_speed"][i] * units('km/h'), p_data["wind_direction"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["geopotential_height"][i])

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        
        if processed_data:
            params_calc = processed_data[1]
            for i, p_val in enumerate(PRESS_LEVELS_ITALIA):
                vv_value = p_data["vertical_velocity"][i]
                if pd.notna(vv_value):
                    params_calc[f'VV_{p_val}hPa'] = vv_value

        return processed_data, valid_index, error
        
    except Exception as e: 
        return None, hourly_index, f"Error en carregar dades del sondeig ICON-2I (Itàlia): {e}"



def formatar_missatge_error_api(error_msg):
    """
    Tradueix els errors de l'API a missatges més amigables per a l'usuari.
    """
    # Comprovació de seguretat per si el missatge no és un text
    if not isinstance(error_msg, str):
        return "S'ha produït un error desconegut en carregar les dades."

    # Aquesta és la clau: busquem el text específic del límit de l'API
    if "Daily API request limit exceeded" in error_msg:
        # Si el trobem, retornem el teu missatge personalitzat
        return "Estem renderitzant i optimitzant la web. Disculpin les molèsties, proveu més tard o demà."
    else:
        # Si és qualsevol altre error, el mostrem per a poder depurar-lo
        return f"S'ha produït un error inesperat: {error_msg}"
    

def trobar_hora_valida_mes_propera(hourly_response, target_index, num_base_vars, max_offset=8):
    """
    Versió Definitiva: Busca l'índex horari més proper (en qualsevol direcció)
    que tingui dades completes, buscant en una finestra més àmplia de 8 hores.
    """
    try:
        # Comprovació inicial per assegurar que la resposta de l'API és vàlida
        if hourly_response is None or hourly_response.Variables(0) is None:
            return None
        total_hours = len(hourly_response.Variables(0).ValuesAsNumpy())
    except (AttributeError, IndexError):
        return None

    # Bucle principal: comença amb offset 0 i s'expandeix cap enfora
    for offset in range(max_offset + 1):
        # Genera els índexs a comprovar: primer l'hora exacta, després -1, +1, -2, +2, etc.
        indices_to_check = [target_index] if offset == 0 else [target_index - offset, target_index + offset]
        
        for h_idx in indices_to_check:
            # Comprova que l'índex estigui dins dels límits de les dades rebudes
            if 0 <= h_idx < total_hours:
                try:
                    # Llegeix les variables base per a l'hora candidata
                    sfc_check = [hourly_response.Variables(i).ValuesAsNumpy()[h_idx] for i in range(num_base_vars)]
                    # Si cap de les variables base és 'NaN' (no és un número), hem trobat una hora vàlida
                    if not any(np.isnan(val) for val in sfc_check):
                        return h_idx  # Retorna l'índex vàlid immediatament
                except (AttributeError, IndexError):
                    # Si hi ha un problema llegint les dades per a aquest índex, el saltem i continuem
                    continue

    return None # Si no troba cap hora vàlida dins del rang de cerca, retorna None

def ui_pestanya_mapes_italia(hourly_index_sel, timestamp_str, nivell_sel):
    st.markdown("#### Mapes de Pronòstic (Model ICON 2.2km - Itàlia)")
    
    with st.spinner("Carregant mapa ICON-2I... El primer cop pot trigar una mica."):
        map_data, error = carregar_dades_mapa_italia(nivell_sel, hourly_index_sel)
    
        if error:
            st.error(f"Error en carregar el mapa: {error}")
        elif map_data:
            fig = crear_mapa_forecast_combinat_italia(
                map_data['lons'], map_data['lats'], 
                map_data['speed_data'], map_data['dir_data'], 
                map_data['dewpoint_data'], nivell_sel, 
                timestamp_str
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning("No s'han pogut obtenir les dades per generar el mapa.")

def crear_mapa_forecast_combinat_cat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str, map_extent):
    """
    VERSIÓ FINAL AMB ESCALA AJUSTADA (30-150):
    - Detecció i visualització de la convergència a partir de 30.
    - Paleta de colors i línies de contorn optimitzades per al rang 30-150.
    - Manté l'estil net, sense llegendes i amb isos negres.
    """
    # Tornem a l'estil per defecte (fons clar)
    plt.style.use('default')

    fig, ax = crear_mapa_base(map_extent)
    
    # --- 1. INTERPOLACIÓ ---
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 300), np.linspace(map_extent[2], map_extent[3], 300))
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'linear')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'linear')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    
    # --- 2. MAPA DE VELOCITAT DEL VENT ---
    colors_wind = [
        '#d2d2f0', '#b4b4e6', '#78c8c8', '#50b48c', '#32cd32', '#64ff64',
        '#ffff00', '#f5d264', '#e6b478', '#d7788c', '#ff69b4', '#9f78dc',
        '#8c64c8', '#8296d7', '#96b4d7', '#d2b4e6', '#e6dcc8', '#f5e6b4'
    ]
    speed_levels = [0, 4, 11, 18, 25, 32, 40, 47, 54, 61, 68, 76, 86, 97, 104, 130, 166, 184, 200]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree(), alpha=0.7)
    
    # --- STREAMLINES DEL VENT ---
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.5, density=6.5, arrowsize=0.3, zorder=4, transform=ccrs.PlateCarree())
    
    # --- 3. CÀLCUL I FILTRATGE DE CONVERGÈNCIA (LLINDAR A 30) ---
    with np.errstate(invalid='ignore'):
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        convergence = (-(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s')).magnitude * 1e5
        convergence[np.isnan(convergence)] = 0
        DEWPOINT_THRESHOLD = 14 if nivell >= 950 else 12
        humid_mask = grid_dewpoint >= DEWPOINT_THRESHOLD
        
        # <<-- CANVI CLAU: El llindar ara és 30 -->>
        effective_convergence = np.where((convergence >= 30) & humid_mask, convergence, 0)

    smoothed_convergence = gaussian_filter(effective_convergence, sigma=2.5)
    
    # <<-- CANVI CLAU: El filtre post-suavitzat també és 30 -->>
    smoothed_convergence[smoothed_convergence < 30] = 0
    
    # --- 4. DIBUIX DE LA CONVERGÈNCIA (NOVA ESCALA 30-150) ---
    if np.any(smoothed_convergence > 0):
        colors_conv = [
            '#5BC0DE', "#FBFF00", "#DC6D05", "#EC8383", "#F03D3D", 
            "#FF0000", "#7C7EF0", "#0408EAFF", "#000070"
        ]
        cmap_conv = LinearSegmentedColormap.from_list("conv_cmap_personalitzada", colors_conv)
        
        # <<-- CANVI CLAU: El farciment comença a 30 i acaba a 151 -->>
        fill_levels = np.arange(30, 151, 5)
        ax.contourf(grid_lon, grid_lat, smoothed_convergence,
                    levels=fill_levels, cmap=cmap_conv, alpha=0.99,
                    zorder=3, transform=ccrs.PlateCarree(), extend='max')

        # <<-- CANVI CLAU: Nous nivells per a les línies de contorn -->>
        line_levels = [30, 50, 70, 90, 120]
        contours = ax.contour(grid_lon, grid_lat, smoothed_convergence,
                              levels=line_levels, 
                              colors='black',
                              linestyles='--', linewidths=1, zorder=3,
                              transform=ccrs.PlateCarree())
        
        labels = ax.clabel(contours, inline=True, fontsize=5, fmt='%1.0f')
        for label in labels:
            label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.5))

    # Ajustos finals del títol
    ax.set_title(f"Vent i Nuclis de Convergència EFECTIVA a {nivell}hPa\n{timestamp_str}",
                 weight='bold', fontsize=16)
    afegir_etiquetes_ciutats(ax, map_extent)
    
    return fig


def forcar_regeneracio_animacio():
    """Incrementa la clau de regeneració per invalidar la memòria cau."""
    if 'regenerate_key' in st.session_state:
        st.session_state.regenerate_key += 1
    else:
        st.session_state.regenerate_key = 1




@st.cache_data(ttl=3600)
def carregar_dades_mapa_japo(nivell, hourly_index):
    """
    Versió Corregida: Redueix la densitat de la graella de punts sol·licitada
    per evitar l'error "414 Request-URI Too Long".
    """
    try:
        variables = [f"temperature_{nivell}hPa", f"dew_point_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        
        # <<<--- CANVI CLAU AQUÍ: Reduïm la graella de 15x15 a 10x10 --->>>
        lats, lons = np.linspace(MAP_EXTENT_JAPO[2], MAP_EXTENT_JAPO[3], 10), np.linspace(MAP_EXTENT_JAPO[0], MAP_EXTENT_JAPO[1], 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "jma_gsm", "forecast_days": 3}
        
        responses = openmeteo.weather_api(API_URL_JAPO, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        
        for r in responses:
            try:
                valid_index = trobar_hora_valida_mes_propera(r.Hourly(), hourly_index, len(variables))
                if valid_index is not None:
                    vals = [r.Hourly().Variables(i).ValuesAsNumpy()[valid_index] for i in range(len(variables))]
                    if not any(np.isnan(v) for v in vals):
                        output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                        for i, var in enumerate(variables): output[var].append(vals[i])
            except IndexError: 
                continue

        if not output["lats"]: 
            return None, "No s'han rebut dades per a la graella del mapa."
        
        output['dewpoint_data'] = output.pop(f'dew_point_{nivell}hPa')
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')
        del output[f'temperature_{nivell}hPa']

        return output, None
    except Exception as e: 
        return None, f"Error en carregar dades del mapa JMA GSM: {e}"
        

def crear_mapa_forecast_combinat_japo(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    """
    Versió Completa: Crea el mapa visual de vent i AFEGEIX els nuclis de convergència
    per al Japó, utilitzant les dades del model JMA GSM.
    """
    fig, ax = crear_mapa_base(MAP_EXTENT_JAPO, projection=ccrs.LambertConformal(central_longitude=138, central_latitude=36))
    
    if len(lons) < 4: 
        ax.set_title("Dades insuficients per generar el mapa")
        return fig

    # Interpolació de dades a una graella fina (sense canvis)
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_JAPO[0], MAP_EXTENT_JAPO[1], 200), np.linspace(MAP_EXTENT_JAPO[2], MAP_EXTENT_JAPO[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    
    # Dibuix de la velocitat del vent (fons de color - sense canvis)
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    custom_cmap = ListedColormap(colors_wind)
    norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    
    # Dibuix de les línies de corrent (sense canvis)
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=5.0, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    # <<<--- NOU BLOC AFEGIT: CÀLCUL I DIBUIX DE LA CONVERGÈNCIA --->>>
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    
    # Filtrem la convergència per a zones amb prou humitat (punt de rosada > 14°C)
    convergence_in_humid_areas = np.where(grid_dewpoint >= 14, convergence_scaled, 0)
    
    # Definim els nivells i colors per al dibuix de la convergència
    fill_levels = [5, 10, 15, 25]
    fill_colors = ['#ffc107', '#ff9800', '#f44336'] # Groc -> Taronja -> Vermell
    line_levels = [5, 10, 15]
    line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    # Dibuixem els contorns de color i les línies
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')
    # <<<--- FI DEL NOU BLOC ---_>>>

    # Afegir ciutats per a referència (sense canvis)
    for city, coords in CIUTATS_JAPO.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.15, coords['lat'] + 0.15, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    # <<<--- CANVI AL TÍTOL --->>>
    ax.set_title(f"Vent i Nuclis de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    
    return fig



def _dibuixar_frame_professional(frame_params):
    """Motor de renderització d'alta qualitat per a un fotograma."""
    el_hgt_km = frame_params['el_hgt_km']
    timestamp = frame_params['timestamp']
    title = frame_params['title']
    cloud_elements = frame_params['cloud_elements']
    precip_elements = frame_params.get('precip_elements', [])
    pluja_acumulada_mm = frame_params.get('pluja_acumulada_mm')
    flash_active = frame_params.get('flash_active', False)
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    fig.patch.set_facecolor('#4A7DAB')
    ax.set_facecolor('#7AB8F5')

    dist_range_km = np.linspace(-15, 15, 256)
    alt_range_km = np.linspace(0, el_hgt_km + 2, 256)
    xx, zz = np.meshgrid(dist_range_km, alt_range_km)

    sky_gradient = np.zeros((256, 1, 3))
    sky_gradient[:, 0, 0] = np.linspace(0.29, 0.48, 256)
    sky_gradient[:, 0, 1] = np.linspace(0.49, 0.72, 256)
    sky_gradient[:, 0, 2] = np.linspace(0.67, 0.96, 256)
    ax.imshow(sky_gradient, aspect='auto', origin='lower', extent=[dist_range_km[0], dist_range_km[-1], 0, el_hgt_km + 2])
    
    ground_x = np.linspace(dist_range_km[0], dist_range_km[-1], 100)
    ax.fill_between(ground_x, 0, 0.25, color='#4A7836', zorder=2)
    ax.fill_between(ground_x, 0, 0.1, color='#556B2F', zorder=2)

    cloud_density = np.zeros_like(xx)
    for cloud in cloud_elements:
        dist_sq = ((xx - cloud['x'])**2 / cloud['rx']**2) + ((zz - cloud['z'])**2 / cloud['rz']**2)
        cloud_density += np.exp(-dist_sq) * cloud['intensity']
    
    cloud_density = np.clip(cloud_density, 0, 1.2)
    levels = np.linspace(0.15, 1.2, 25)

    shadow_cmap = ListedColormap([(0.2, 0.2, 0.3, alpha) for alpha in np.linspace(0, 0.5, 25)])
    ax.contourf(xx + 0.3, zz - 0.15, cloud_density, levels=levels, cmap=shadow_cmap, zorder=3)
    
    highlight_cmap = ListedColormap([(1, 1, 0.9, alpha) for alpha in np.linspace(0, 0.8, 25)])
    ax.contourf(xx - 0.15, zz + 0.15, cloud_density, levels=levels, cmap=highlight_cmap, zorder=4)

    cloud_cmap = ListedColormap([(0.9, 0.9, 0.95, alpha) for alpha in np.linspace(0, 1, 25)])
    ax.contourf(xx, zz, cloud_density, levels=levels, cmap=cloud_cmap, zorder=5)

    if flash_active:
        flash_colors = [(1, 1, 0.8, 0.0), (1, 1, 0.9, 0.7), (1, 1, 1, 0.7)]
        flash_cmap = LinearSegmentedColormap.from_list("flash_cmap", flash_colors)
        ax.contourf(xx, zz, cloud_density, levels=levels, cmap=flash_cmap, zorder=6)

    for precip in precip_elements:
        ax.plot([precip['x'], precip['x']], [precip['z_base'], precip['z_top']], color='grey', linewidth=0.6, alpha=precip['intensity'])
    
    if pluja_acumulada_mm is not None:
        text = ax.text(dist_range_km[-1] - 0.5, 0.5, f"{pluja_acumulada_mm:.1f} mm",
                       color='white', fontsize=12, weight='bold', ha='right')
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

    ax.set_xlabel("Distància (km)")
    ax.set_ylabel("Altitud (km)")
    ax.grid(True, linestyle=':', alpha=0.3, color='black')
    ax.set_xlim(dist_range_km[0], dist_range_km[-1])
    ax.set_ylim(0, el_hgt_km + 2)
    ax.set_title(f"{title}\n{timestamp}", weight='bold', fontsize=12)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return imageio.imread(buf)

@st.cache_data(ttl=3600, show_spinner=False)
def generar_animacions_professionals(_params_tuple, timestamp_str, _regenerate_key):
    params = dict(_params_tuple)
    lcl_hgt_km = (params.get('LCL_Hgt', np.nan)) / 1000.0
    lfc_hgt_km = (params.get('LFC_Hgt', np.nan)) / 1000.0
    el_hgt_km = (params.get('EL_Hgt', np.nan)) / 1000.0
    cape = params.get('MUCAPE', 0) or 0
    shear_kts = params.get('BWD_0-6km', 0) or 0
    pwat = params.get('PWAT', 20) or 20

    # --- BLOC DE SEGURETAT PER EVITAR L'ERROR NaN ---
    if any(np.isnan(val) for val in [lcl_hgt_km, lfc_hgt_km, el_hgt_km]):
        return {'iniciacio': None, 'maduresa': None, 'dissipacio': None}
    lfc_hgt_km = max(lcl_hgt_km, lfc_hgt_km)
    # --- FI DEL BLOC DE SEGURETAT ---

    gifs = {'iniciacio': None, 'maduresa': None, 'dissipacio': None}
    max_precip_mm = np.clip((pwat - 15) / 20, 0, 1) * (20 + cape/100)

    # --- 1. Animació d'Iniciació ---
    frames_inici = []
    for i in range(20):
        cloud_elements = []
        for j in range(5):
            growth = np.sin((i / 20) * np.pi + (j * np.pi / 2.5))
            if growth > 0:
                cloud_elements.append({
                    'x': -10 + j * 5, 'z': lcl_hgt_km + 0.3 * growth,
                    'rx': 1.5 * growth, 'rz': 0.8 * growth, 'intensity': growth
                })
        frames_inici.append(_dibuixar_frame_professional({
            'el_hgt_km': el_hgt_km, 'timestamp': timestamp_str, 'title': "Fase 1: Iniciació",
            'cloud_elements': cloud_elements
        }))
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, frames_inici, format='gif', fps=5, loop=0)
    gifs['iniciacio'] = gif_buf.getvalue()

    if cape < 100 or el_hgt_km <= lfc_hgt_km: return gifs

    # --- 2. Animació de Maduresa ---
    frames_madur = []
    shear_tilt_km_per_km = shear_kts * 0.04
    for i in range(30):
        flash_active = False
        growth = np.sin((i / 30) * (np.pi / 2))
        current_top_km = lfc_hgt_km + (el_hgt_km - lfc_hgt_km) * growth
        cloud_width_km = (2.0 + cape / 800) * (growth**0.7)
        x_offset = (current_top_km - lfc_hgt_km) * shear_tilt_km_per_km * 0.5
        
        cloud_elements = []
        cloud_elements.append({'x': x_offset, 'z': (current_top_km + lfc_hgt_km)/2,
                               'rx': cloud_width_km, 'rz': (current_top_km - lfc_hgt_km)/2, 'intensity': 1.0})
        for _ in range(int(15 * growth)):
            puff_z = lfc_hgt_km + np.random.rand() * (current_top_km - lfc_hgt_km)
            puff_x = (puff_z - lfc_hgt_km) * shear_tilt_km_per_km + (np.random.rand()-0.5) * cloud_width_km
            puff_rad = (0.5 + np.random.rand()) * (1 - (puff_z/current_top_km)**2) * 2.0
            cloud_elements.append({'x': puff_x, 'z': puff_z, 'rx': puff_rad, 'rz': puff_rad*0.7, 'intensity': 0.8})
        
        if current_top_km > 6.0 and np.random.rand() < 0.2:
            flash_active = True

        pluja_acumulada_mm = max_precip_mm * growth if growth > 0.8 else None
        
        frames_madur.append(_dibuixar_frame_professional({
            'el_hgt_km': el_hgt_km, 'timestamp': timestamp_str, 'title': "Fase 2: Maduresa",
            'cloud_elements': cloud_elements, 'pluja_acumulada_mm': pluja_acumulada_mm,
            'flash_active': flash_active
        }))
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, frames_madur, format='gif', fps=7, loop=0)
    gifs['maduresa'] = gif_buf.getvalue()

    # --- 3. Animació de Dissipació ---
    frames_dissip = []
    for i in range(25):
        progress = i / 25
        
        anvil_density = 1 - progress
        anvil_width = 10 + 15 * progress
        anvil_x = (el_hgt_km - lfc_hgt_km) * shear_tilt_km_per_km + anvil_width * 0.2 * progress
        cloud_elements = [{'x': anvil_x, 'z': el_hgt_km, 'rx': anvil_width, 'rz': 0.5 * anvil_density, 'intensity': anvil_density}]

        precip_intensity = np.sin((1-progress) * np.pi) * np.clip((pwat - 15) / 25, 0, 1.2)
        precip_elements = []
        if precip_intensity > 0.1:
            rain_top = lcl_hgt_km if pwat > 20 else lcl_hgt_km * (1 - (20-pwat)/5)
            for _ in range(int(150 * precip_intensity)):
                precip_elements.append({
                    'x': anvil_x + (np.random.rand()-0.5) * anvil_width,
                    'z_top': rain_top, 'z_base': 0,
                    'intensity': 0.3 + np.random.rand() * 0.4
                })

        frames_dissip.append(_dibuixar_frame_professional({
            'el_hgt_km': el_hgt_km, 'timestamp': timestamp_str, 'title': "Fase 3: Dissipació",
            'cloud_elements': cloud_elements, 'precip_elements': precip_elements,
            'pluja_acumulada_mm': max_precip_mm
        }))
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, frames_dissip, format='gif', fps=6, loop=0)
    gifs['dissipacio'] = gif_buf.getvalue()

    return gifs






def ui_guia_tall_vertical(params, nivell_conv):
    """
    Guia d'usuari actualitzada per interpretar la nova simulació de núvol.
    """
    # (El codi d'aquesta funció no necessita canvis respecte a la versió anterior)
    st.markdown("#### 🔍 Com Interpretar la Simulació")
    
    veredicte_caca = analitzar_potencial_caca(params, nivell_conv)
    el_hgt_km = (params.get('EL_Hgt', 0) or 0) / 1000
    cape = params.get('MUCAPE', 0) or 0
    shear = params.get('BWD_0-6km', 0) or 0

    st.markdown("""
    <style>
    .guide-card { background-color: #f0f2f6; border: 1px solid #d1d1d1; border-radius: 8px; padding: 15px; margin-bottom: 12px; }
    .guide-title { font-size: 1.1em; font-weight: bold; color: #1a1a2e; display: flex; align-items: center; margin-bottom: 8px; }
    .guide-icon { font-size: 1.3em; margin-right: 10px; }
    .guide-text { font-size: 0.95em; color: #333; line-height: 1.6; }
    .guide-text strong { color: #0056b3; }
    .verdict-card { border-left: 5px solid; padding: 16px; margin-bottom: 15px; border-radius: 8px; background-color: #262730; }
    .verdict-title { font-size: 1.2em; font-weight: bold; color: white; }
    .verdict-motiu { font-size: 0.9em; color: #b0b0c8; font-style: italic; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="verdict-card" style="border-left-color: {veredicte_caca['color']};">
        <div class="verdict-title">Val la pena anar-hi? <span style="color: {veredicte_caca['color']};">{veredicte_caca['text']}</span></div>
        <div class="verdict-motiu">{veredicte_caca['motiu']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="guide-card">
        <div class="guide-title"><span class="guide-icon">📏</span>Alçada i Desenvolupament</div>
        <div class="guide-text">El núvol creix fins a un cim (top) estimat de <strong>{el_hgt_km:.1f} km</strong>. La seva rapidesa de creixement depèn del CAPE (<strong>{cape:.0f} J/kg</strong>).</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="guide-card">
        <div class="guide-title"><span class="guide-icon">📐</span>Forma i Inclinació</div>
        <div class="guide-text">La inclinació del núvol és un indicador d'<strong>organització</strong>. Està causada per un cisallament del vent de <strong>{shear:.0f} nusos</strong>. Més inclinació pot significar una tempesta més duradora i severa.</div>
    </div>
    """, unsafe_allow_html=True)


def _dibuixar_frame_tall_vertical(frame_params):
    """
    Funció auxiliar que dibuixa UN ÚNIC FOTOGRAMA de l'animació.
    Rep els paràmetres de la tempesta per a un instant de temps concret.
    """
    # Extreure paràmetres del frame actual
    dist_range = frame_params['dist_range']
    alt_range = frame_params['alt_range']
    reflectivity_grid = frame_params['reflectivity_grid']
    el_hgt_m = frame_params['el_hgt_m']
    timestamp = frame_params['timestamp']

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100) # DPI més baix per a l'animació

    colors_dbz = ['#f0f8ff', '#b0e0e6', '#87ceeb', '#4682b4', '#32cd32', '#ffff00', '#ffc800', '#ffa500', '#ff4500', '#ff0000', '#d90000', '#ff00ff']
    dbz_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75]
    cmap_dbz = ListedColormap(colors_dbz)
    norm_dbz = BoundaryNorm(dbz_levels, ncolors=cmap_dbz.N, clip=True)

    xx, zz = np.meshgrid(dist_range, alt_range)
    ax.contourf(xx, zz, reflectivity_grid, levels=dbz_levels, cmap=cmap_dbz, norm=norm_dbz, extend='max')

    terreny_x = np.linspace(-20, 20, 100)
    terreny_y = np.sin(terreny_x / 5) * 200 + 400
    ax.fill_between(terreny_x, 0, terreny_y, color='#a0785a', zorder=2)

    ax.set_xlabel("Distància (km)")
    ax.set_ylabel("Altitud (m)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, el_hgt_m + 2000)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_dbz, cmap=cmap_dbz), ax=ax, ticks=dbz_levels)
    cbar.set_label("Reflectivitat (dBZ)")
    
    # Títol dinàmic amb l'hora simulada
    ax.set_title(f"Simulació de Tall Vertical (RHI)\n{timestamp}", weight='bold', fontsize=14)
    plt.tight_layout()
    
    # Guardem el gràfic a la memòria en lloc de mostrar-lo
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)




def crear_mapa_convergencia_cat(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str, map_extent):
    """
    VERSIÓ NETA: Mostra ÚNICAMENT els nuclis de convergència (a partir de 40),
    eliminant llegendes i línies per a una visualització clara i directa.
    """
    plt.style.use('dark_background')
    fig, ax = crear_mapa_base(map_extent)
    ax.patch.set_facecolor('black')
    fig.patch.set_facecolor('black')

    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 150), np.linspace(map_extent[2], map_extent[3], 150))
    
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'linear')

    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    
    DEWPOINT_THRESHOLD = 14 if nivell >= 950 else 12
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD, convergence_scaled, 0)
    
    # Filtrem estrictament a partir de 40
    convergence_in_humid_areas[convergence_in_humid_areas < 40] = 0

    # Dibuixem només el farciment de color
    if np.any(convergence_in_humid_areas > 0):
        fill_levels = [40, 60, 80, 100]; 
        fill_colors = ['#FF9800', '#F44336', '#D32F2F', '#B71C1C']
        
        ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, 
                    levels=fill_levels, colors=fill_colors, alpha=0.65, 
                    zorder=5, transform=ccrs.PlateCarree(), extend='max')

    ax.set_title(f"Focus de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16, color='white')
    afegir_etiquetes_ciutats(ax, map_extent)

    return fig
# --- Funcions Específiques per a Tornado Alley ---



def ui_pestanya_webcams(poble_sel, zona_activa):
    """
    Versió Definitiva: Mostra un vídeo incrustat si el tipus és 'embed', un botó
    per a enllaços de tipus 'direct', o el mapa de Windy si no hi ha cap enllaç.
    """
    st.markdown(f"#### Webcams en Directe per a {poble_sel}")

    webcam_data = WEBCAM_LINKS.get(poble_sel)
    
    if webcam_data:
        link_type = webcam_data.get('type')
        url = webcam_data.get('url')

        if link_type == 'embed':
            st.info("La qualitat i disponibilitat del vídeo depenen de la font externa.")
            st.components.v1.html(
                f'<iframe width="100%" height="600" src="{url}" frameborder="0" allow="autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
                height=620
            )

        elif link_type == 'direct':
            st.warning("El propietari d'aquesta webcam no permet la inserció directa a altres pàgines.")
            st.info("Fes clic al botó de sota per obrir el vídeo en directe en una nova pestanya de YouTube.")
            
            # Creem un botó gran i visible amb HTML
            button_html = f"""
            <a href="{url}" target="_blank" rel="noopener noreferrer" style="
                display: inline-block;
                padding: 0.75rem 1.5rem;
                font-size: 1.1rem;
                font-weight: bold;
                color: white;
                background-color: #FF4B4B; /* Color vermell de YouTube */
                text-decoration: none;
                border-radius: 8px;
                text-align: center;
                margin-top: 15px;
                margin-bottom: 15px;
            ">
                🎥 Obrir la Webcam a YouTube
            </a>
            """
            st.markdown(button_html, unsafe_allow_html=True)

    else:
        # Si no hi ha cap enllaç al diccionari, mostrem el mapa de Windy com a alternativa
        st.warning(f"No s'ha configurat cap webcam específica per a **{poble_sel}**.")
        st.info("Mostrant el mapa de webcams properes de Windy.com. Pots moure't pel mapa i clicar a les icones de càmera 📷.")

        if zona_activa == 'catalunya': CIUTATS_DICT = CIUTATS_CATALUNYA
        elif zona_activa == 'valley_halley': CIUTATS_DICT = USA_CITIES
        elif zona_activa == 'alemanya': CIUTATS_DICT = CIUTATS_ALEMANYA
        elif zona_activa == 'italia': CIUTATS_DICT = CIUTATS_ITALIA
        elif zona_activa == 'holanda': CIUTATS_DICT = CIUTATS_HOLANDA
        elif zona_activa == 'japo': CIUTATS_DICT = CIUTATS_JAPO
        elif zona_activa == 'uk': CIUTATS_DICT = CIUTATS_UK
        elif zona_activa == 'canada': CIUTATS_DICT = CIUTATS_CANADA
        elif zona_activa == 'noruega': CIUTATS_DICT = CIUTATS_NORUEGA
        else: CIUTATS_DICT = {}

        coords = CIUTATS_DICT.get(poble_sel)
        if not coords:
            st.error(f"No s'han trobat les coordenades per a {poble_sel} per centrar el mapa de Windy.")
            return

        lat, lon = coords['lat'], coords['lon']
        windy_url = f"https://embed.windy.com/webcams/map/{lat}/{lon}?overlay=radar"
        
        st.components.v1.html(f'<iframe width="100%" height="500" src="{windy_url}" frameborder="0"></iframe>', height=520)

@st.cache_data(show_spinner="Carregant mapa de selecció...")
def carregar_dades_geografiques():
    """
    Carrega el teu arxiu GeoJSON personalitzat.
    """
    try:
        # --- CANVI CLAU: Utilitzem el nou arxiu ---
        file_path = "mapa_personalitzat.geojson" 
        
        if not os.path.exists(file_path):
            st.error(f"No s'ha trobat l'arxiu '{file_path}'. Assegura't que estigui a la mateixa carpeta que l'script.")
            return None

        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error en carregar l'arxiu '{file_path}'. Detall: {e}")
        return None


def on_poble_select():
    """
    Callback que s'activa quan l'usuari tria una població de la llista.
    Actualitza l'estat principal de la sessió.
    """
    poble = st.session_state.poble_selector_widget
    # Assegurem que no sigui el placeholder abans d'assignar-lo
    if poble and "---" not in poble:
        st.session_state.poble_selector = poble







    

def canviar_poble_analitzat(nom_poble):
    """
    Funció de callback per canviar la selecció del poble a l'estat de la sessió.
    Això s'executa abans de redibuixar, evitant l'error de Streamlit de
    modificar un widget que ja ha estat creat.
    """
    st.session_state.poble_selector = nom_poble

def mostrar_carga_avanzada(mensaje, funcion_a_ejecutar, *args, **kwargs):
    """
    Versión simplificada y funcional
    """
    # Operaciones de navegación (rápidas)
    operaciones_rapidas = ["sortir", "tancar", "canviar", "entrar", "seleccionar", "nav", "zona"]
    
    if any(palabra in mensaje.lower() for palabra in operaciones_rapidas):
        # Navegación: muy rápida
        with st.spinner(f"⚡ {mensaje}"):
            time.sleep(0.8)
        return None
    
    # Operaciones de datos (las que tardan)
    else:
        with st.spinner(f"🌪️ {mensaje}..."):
            return funcion_a_ejecutar(*args, **kwargs)


# Y para las operaciones de navegación, usar mensajes específicos:
def navegacion_rapida(mensaje):
    """Función específica para navegación rápida"""
    with st.spinner(f"⚡ {mensaje}..."):
        time.sleep(1.2)  # Aún más rápido

def mostrar_spinner_mapa(mensaje, funcion_carga, *args, **kwargs):
    """
    Spinner simple que muestra un mensaje mientras carga
    """
    # Mostrar spinner inmediatamente
    with st.spinner(f"🌪️ {mensaje}"):
        try:
            # Ejecutar la función de carga directamente
            result = funcion_carga(*args, **kwargs)
            return result
        except Exception as e:
            st.error(f"Error carregant el mapa: {e}")
            return None, str(e)


@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_alemanya(lat, lon, hourly_index):
    """
    Versió Definitiva v3.0: Carrega les dades d'Alemanya amb una lògica
    tolerant que no descarta els nivells superiors si falten dades parcials.
    Això soluciona el problema del tall del gràfic als 200 hPa.
    """
    try:
        h_base = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        press_vars_names = ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"]
        h_press = [f"{var}_{p}hPa" for var in press_vars_names for p in PRESS_LEVELS_ICON]
        
        params = { "latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "icon_d2", "forecast_days": 2 }
        response = openmeteo.weather_api(API_URL_ALEMANYA, params=params)[0]
        hourly = response.Hourly()

        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None:
            return None, hourly_index, "No s'han trobat dades vàlides."

        # Llegim totes les dades en un diccionari per a un accés segur
        hourly_vars = {}
        all_requested_vars = h_base + h_press
        for i, var_name in enumerate(all_requested_vars):
            try:
                hourly_vars[var_name] = hourly.Variables(i).ValuesAsNumpy()
            except Exception:
                hourly_vars[var_name] = np.array([np.nan])
        
        # Processem les dades de superfície
        sfc_data = {v: hourly_vars[v][valid_index] for v in h_base}
        sfc_dew_point = mpcalc.dewpoint_from_relative_humidity(sfc_data["temperature_2m"] * units.degC, sfc_data["relative_humidity_2m"] * units.percent).m
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        
        # Inicialitzem els perfils amb les dades de superfície
        p_profile = [sfc_data["surface_pressure"]]
        T_profile = [sfc_data["temperature_2m"]]
        Td_profile = [sfc_dew_point]
        u_profile = [sfc_u.to('m/s').m]
        v_profile = [sfc_v.to('m/s').m]
        h_profile = [0.0]

        # --- AQUÍ ESTÀ LA CORRECCIÓ CLAU ---
        # Aquest bucle ara és tolerant. Afegeix el nivell sempre, i si una dada
        # específica falta, simplement afegeix un 'NaN' en lloc de descartar tota la línia.
        for p_val in PRESS_LEVELS_ICON:
            if p_val < p_profile[-1]:
                p_profile.append(p_val)
                
                # Obtenim les dades d'aquest nivell, agafant 'NaN' si no existeixen
                temp = hourly_vars.get(f'temperature_{p_val}hPa', [np.nan])[valid_index]
                rh = hourly_vars.get(f'relative_humidity_{p_val}hPa', [np.nan])[valid_index]
                ws = hourly_vars.get(f'wind_speed_{p_val}hPa', [np.nan])[valid_index]
                wd = hourly_vars.get(f'wind_direction_{p_val}hPa', [np.nan])[valid_index]
                h = hourly_vars.get(f'geopotential_height_{p_val}hPa', [np.nan])[valid_index]
                
                # Afegim les dades als perfils (poden ser 'NaN')
                T_profile.append(temp)
                h_profile.append(h)
                
                if pd.notna(temp) and pd.notna(rh):
                    Td_profile.append(mpcalc.dewpoint_from_relative_humidity(temp * units.degC, rh * units.percent).m)
                else:
                    Td_profile.append(np.nan)
                
                if pd.notna(ws) and pd.notna(wd):
                    u, v = mpcalc.wind_components(ws * units('km/h'), wd * units.degrees)
                    u_profile.append(u.to('m/s').m)
                    v_profile.append(v.to('m/s').m)
                else:
                    u_profile.append(np.nan)
                    v_profile.append(np.nan)
        # --- FI DE LA CORRECCIÓ ---

        # Finalment, passem les dades completes (amb possibles NaNs) a la funció de processament global
        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        import traceback
        traceback.print_exc()
        return None, hourly_index, f"Error crític en carregar dades del sondeig ICON-D2: {e}"
    

@st.cache_data(ttl=3600)
def carregar_dades_mapa_alemanya(nivell, hourly_index):
    try:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        lats, lons = np.linspace(MAP_EXTENT_ALEMANYA[2], MAP_EXTENT_ALEMANYA[3], 12), np.linspace(MAP_EXTENT_ALEMANYA[0], MAP_EXTENT_ALEMANYA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "icon_d2", "forecast_days": 3}
        
        responses = openmeteo.weather_api(API_URL_ALEMANYA, params=params)
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

        # Processar dades
        temp_data = np.array(output.pop(f'temperature_{nivell}hPa')) * units.degC
        rh_data = np.array(output.pop(f'relative_humidity_{nivell}hPa')) * units.percent
        output['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')
        return output, None

    except Exception as e: 
        return None, f"Error en carregar dades del mapa ICON-D2: {e}"
    




@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_noruega(lat, lon, hourly_index):
    """
    Versió Corregida: Assegura que la funció sempre retorna 3 valors
    (data_tuple, final_index, error_msg) per evitar el ValueError.
    """
    try:
        h_base = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        press_vars = ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"]
        h_press = [f"{v}_{p}hPa" for v in press_vars for p in PRESS_LEVELS_NORUEGA]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "ukmo_seamless", "forecast_days": 2}
        
        response = openmeteo.weather_api(API_URL_NORUEGA, params=params)[0]
        hourly = response.Hourly()
        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None: 
            return None, hourly_index, "No s'han trobat dades vàlides."

        # Llegim les dades de manera robusta
        hourly_vars = {}
        all_requested_vars = h_base + h_press
        for i, var_name in enumerate(all_requested_vars):
            try:
                hourly_vars[var_name] = hourly.Variables(i).ValuesAsNumpy()
            except Exception:
                hourly_vars[var_name] = np.array([np.nan])

        sfc_data = {v: hourly_vars[v][valid_index] for v in h_base}
        sfc_dew_point = mpcalc.dewpoint_from_relative_humidity(sfc_data["temperature_2m"] * units.degC, sfc_data["relative_humidity_2m"] * units.percent).m
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_dew_point], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for p_val in PRESS_LEVELS_NORUEGA:
            if p_val < p_profile[-1]:
                p_profile.append(p_val)
                temp = hourly_vars.get(f'temperature_{p_val}hPa', [np.nan])[valid_index]
                rh = hourly_vars.get(f'relative_humidity_{p_val}hPa', [np.nan])[valid_index]
                ws = hourly_vars.get(f'wind_speed_{p_val}hPa', [np.nan])[valid_index]
                wd = hourly_vars.get(f'wind_direction_{p_val}hPa', [np.nan])[valid_index]
                h = hourly_vars.get(f'geopotential_height_{p_val}hPa', [np.nan])[valid_index]
                
                T_profile.append(temp); h_profile.append(h)
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(temp * units.degC, rh * units.percent).m if pd.notna(temp) and pd.notna(rh) else np.nan)
                if pd.notna(ws) and pd.notna(wd):
                    u, v = mpcalc.wind_components(ws * units('km/h'), wd * units.degrees)
                    u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m)
                else:
                    u_profile.append(np.nan); v_profile.append(np.nan)
        
        # <<<--- CORRECCIÓ CLAU AQUÍ ---
        # 1. Capturem el resultat de 'processar_dades_sondeig'
        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        # 2. Retornem un tuple amb els 3 valors que s'esperen
        return processed_data, valid_index, error
        # <<<--- FI DE LA CORRECCIÓ ---

    except Exception as e:
        # Aquest return ja era correcte (retornava 3 valors)
        return None, hourly_index, f"Error en carregar dades del sondeig de Noruega: {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_noruega(nivell, hourly_index):
    try:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        lats, lons = np.linspace(MAP_EXTENT_NORUEGA[2], MAP_EXTENT_NORUEGA[3], 12), np.linspace(MAP_EXTENT_NORUEGA[0], MAP_EXTENT_NORUEGA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "ukmo_seamless", "forecast_days": 2}
        
        responses = openmeteo.weather_api(API_URL_NORUEGA, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            try:
                vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
                if not any(np.isnan(v) for v in vals):
                    output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                    for i, var in enumerate(variables): output[var].append(vals[i])
            except IndexError: continue
        
        if not output["lats"]: return None, "No s'han rebut dades."
        temp_data = np.array(output.pop(f'temperature_{nivell}hPa')) * units.degC
        rh_data = np.array(output.pop(f'relative_humidity_{nivell}hPa')) * units.percent
        output['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')
        return output, None
    except Exception as e:
        return None, f"Error en carregar dades del mapa de Noruega: {e}"

def crear_mapa_forecast_combinat_noruega(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base(MAP_EXTENT_NORUEGA, projection=ccrs.LambertConformal(central_longitude=10, central_latitude=64))
    if len(lons) < 4: return fig

    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_NORUEGA[0], MAP_EXTENT_NORUEGA[1], 200), np.linspace(MAP_EXTENT_NORUEGA[2], MAP_EXTENT_NORUEGA[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat); convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    convergence_in_humid_areas = np.where(grid_dewpoint >= 5, convergence_scaled, 0) # Llindar de punt de rosada baix per clima fred
    
    fill_levels = [5, 10, 15, 25]; fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')

    for city, coords in CIUTATS_NORUEGA.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.2, coords['lat'] + 0.2, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def ui_pestanya_mapes_noruega(hourly_index_sel, timestamp_str, nivell_sel, poble_sel):
    st.markdown("#### Mapes de Pronòstic (Model UKMO Seamless)")
    with st.spinner("Carregant mapa UKMO... El primer cop pot trigar una mica."):
        map_data, error = carregar_dades_mapa_noruega(nivell_sel, hourly_index_sel)
    
        if error or not map_data:
            # <<<--- CORRECCIÓ AQUÍ --->>>
            st.error(f"Error en carregar el mapa: {error if error else 'No s''han rebut dades.'}")
        else:
            timestamp_map_title = timestamp_str.replace(f"{poble_sel} | ", "")
            fig = crear_mapa_forecast_combinat_noruega(
                map_data['lons'], map_data['lats'], map_data['speed_data'],
                map_data['dir_data'], map_data['dewpoint_data'], nivell_sel,
                timestamp_map_title
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)



def formatar_missatge_error_api(error_msg):
    """
    Tradueix els errors de l'API a missatges més amigables per a l'usuari.
    """
    if not isinstance(error_msg, str):
        return "S'ha produït un error desconegut en carregar les dades."

    if "Daily API request limit exceeded" in error_msg:
        return "Estem renderitzant i optimitzant la web. Disculpin les molèsties, proveu més tard o demà."
    else:
        return f"S'ha produït un error inesperat: {error_msg}"
    

def run_noruega_app():
    if 'poble_selector_noruega' not in st.session_state: st.session_state.poble_selector_noruega = "Oslo"
    ui_capcalera_selectors(None, zona_activa="noruega")
    poble_sel = st.session_state.poble_selector_noruega
    now_local = datetime.now(TIMEZONE_NORUEGA)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = CIUTATS_NORUEGA[poble_sel]['lat'], CIUTATS_NORUEGA[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} ({TIMEZONE_NORUEGA.zone}) / {cat_dt.strftime('%H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]

    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_noruega", default_index=0)

    if st.session_state.active_tab_noruega == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_noruega(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_NORUEGA)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_noruega(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_noruega == "Anàlisi de Mapes":
        ui_pestanya_mapes_noruega(hourly_index_sel, timestamp_str, nivell_sel, poble_sel)
    elif st.session_state.active_tab_noruega == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="noruega")


def crear_mapa_forecast_combinat_alemanya(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    fig, ax = crear_mapa_base(MAP_EXTENT_ALEMANYA, projection=ccrs.LambertConformal(central_longitude=10, central_latitude=51))
    
    if len(lons) < 4: return fig

    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_ALEMANYA[0], MAP_EXTENT_ALEMANYA[1], 200), np.linspace(MAP_EXTENT_ALEMANYA[2], MAP_EXTENT_ALEMANYA[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')

    # Dibuix de la velocitat del vent
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    
    # Línies de corrent
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    # Convergència
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat); convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    convergence_in_humid_areas = np.where(grid_dewpoint >= 14, convergence_scaled, 0)
    
    fill_levels = [5, 10, 15, 25]; fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.5, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')

    # Ciutats
    for city, coords in CIUTATS_ALEMANYA.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.1, coords['lat'] + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def ui_pestanya_satelit_europa():
    st.markdown("#### Imatge de Satèl·lit Meteosat (Temps Real)")
    # URL d'EUMETSAT per a la imatge més recent del disc complet en color natural
    sat_url = f"https://eumetview.eumetsat.int/static-images/latestImages/EUMETSAT_MSG_RGB-naturalcolor-full.png?{int(time.time())}"
    st.image(sat_url, caption="Imatge del satèl·lit Meteosat - Disc Complet (EUMETSAT)", use_container_width=True)
    st.info("Aquesta imatge del disc complet d'Europa i Àfrica s'actualitza cada 15 minuts.")
    st.markdown("<p style='text-align: center;'>[Font: EUMETSAT](https://eumetview.eumetsat.int/)</p>", unsafe_allow_html=True)



@st.cache_data(ttl=1800, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_cat(lat, lon, hourly_index):
    """
    Versió Definitiva i Corregida v2.0.
    Garanteix que el perfil de dades construït sigui sempre coherent
    i net, eliminant la font principal d'errors posteriors.
    """
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS_AROME]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": 4}
        response = openmeteo.weather_api(API_URL_CAT, params=params)[0]
        hourly = response.Hourly()

        valid_index = None; max_hours_to_check = 3
        total_hours = len(hourly.Variables(0).ValuesAsNumpy())
        for offset in range(max_hours_to_check + 1):
            indices_to_try = sorted(list(set([hourly_index + offset, hourly_index - offset])))
            for h_idx in indices_to_try:
                if 0 <= h_idx < total_hours:
                    sfc_check = [hourly.Variables(i).ValuesAsNumpy()[h_idx] for i in range(len(h_base))]
                    if not any(np.isnan(val) for val in sfc_check):
                        valid_index = h_idx; break
            if valid_index is not None: break
        
        if valid_index is None: return None, hourly_index, "No s'han trobat dades vàlides."
        
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[valid_index] for i, v in enumerate(h_base)}
        
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_AROME) + j).ValuesAsNumpy()[valid_index] for j in range(len(PRESS_LEVELS_AROME))]
        
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        
        p_profile = [sfc_data["surface_pressure"]]
        T_profile = [sfc_data["temperature_2m"]]
        Td_profile = [sfc_data["dew_point_2m"]]
        u_profile = [sfc_u.to('m/s').m]
        v_profile = [sfc_v.to('m/s').m]
        h_profile = [0.0]
        
        for i, p_val in enumerate(PRESS_LEVELS_AROME):
            if p_val < p_profile[-1] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        return None, hourly_index, f"Error en carregar dades del sondeig AROME: {e}"
    
            
@st.cache_data(ttl=1800, max_entries=20, show_spinner=False) # TTL més curt (30min) ja que el HRRR s'actualitza cada hora
def carregar_dades_sondeig_usa(lat, lon, hourly_index):
    """
    Versió Actualitzada: Carrega dades de sondeig per a EUA utilitzant el model
    d'alta resolució HRRR (gfs_hrrr), que proporciona un gran detall vertical.
    """
    try:
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        # HRRR proporciona 'dew_point' directament, la qual cosa és ideal
        press_vars = ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height", "vertical_velocity"]
        h_press = [f"{v}_{p}hPa" for v in press_vars for p in PRESS_LEVELS_HRRR]
        all_requested_vars = h_base + h_press
        
        # <<<--- CANVI CLAU: Utilitzem el model 'gfs_hrrr' --->>>
        params = {"latitude": lat, "longitude": lon, "hourly": all_requested_vars, "models": "gfs_hrrr", "forecast_days": 2} # HRRR té un pronòstic més curt
        
        response = openmeteo.weather_api(API_URL_USA, params=params)[0]
        hourly = response.Hourly()

        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None: return None, hourly_index, "No s'han trobat dades vàlides."

        hourly_vars = {}
        for i, var_name in enumerate(all_requested_vars):
            try: hourly_vars[var_name] = hourly.Variables(i).ValuesAsNumpy()
            except Exception: hourly_vars[var_name] = np.array([np.nan])
        
        sfc_data = {v: hourly_vars[v][valid_index] for v in h_base}
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for p_val in PRESS_LEVELS_HRRR:
            if p_val < p_profile[-1]:
                p_profile.append(p_val)
                T_profile.append(hourly_vars.get(f'temperature_{p_val}hPa', [np.nan])[valid_index])
                Td_profile.append(hourly_vars.get(f'dew_point_{p_val}hPa', [np.nan])[valid_index])
                h_profile.append(hourly_vars.get(f'geopotential_height_{p_val}hPa', [np.nan])[valid_index])
                ws = hourly_vars.get(f'wind_speed_{p_val}hPa', [np.nan])[valid_index]
                wd = hourly_vars.get(f'wind_direction_{p_val}hPa', [np.nan])[valid_index]
                if pd.notna(ws) and pd.notna(wd):
                    u, v = mpcalc.wind_components(ws * units('km/h'), wd * units.degrees)
                    u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m)
                else:
                    u_profile.append(np.nan); v_profile.append(np.nan)

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        import traceback; traceback.print_exc()
        return None, hourly_index, f"Error crític en carregar dades del sondeig HRRR: {e}"



@st.cache_data(ttl=1800, show_spinner="Analitzant focus de convergència a tot el territori...")
def calcular_alertes_per_comarca(hourly_index, nivell):
    """
    Versió robusta que s'adapta a qualsevol GeoJSON (amb 'nomcomar' o 'nom_zona').
    Retorna un diccionari amb el valor MÀXIM de convergència per a cada zona/comarca.
    """
    CONV_THRESHOLD = 25
    
    map_data, error = carregar_dades_mapa_cat(nivell, hourly_index)
    gdf_zones = carregar_dades_geografiques()

    if error or not map_data or gdf_zones is None or 'lons' not in map_data or len(map_data['lons']) < 4:
        return {}

    try:
        # --- LÒGICA DE DETECCIÓ AUTOMÀTICA ---
        property_name = 'nom_zona' if 'nom_zona' in gdf_zones.columns else 'nomcomar'

        lons, lats = map_data['lons'], map_data['lats']
        grid_lon, grid_lat = np.meshgrid(np.linspace(min(lons), max(lons), 150), np.linspace(min(lats), max(lats), 150))
        u_comp, v_comp = mpcalc.wind_components(np.array(map_data['speed_data']) * units('km/h'), np.array(map_data['dir_data']) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')

        with np.errstate(invalid='ignore'):
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            convergence_scaled = -mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy).to('1/s').magnitude * 1e5
        
        punts_calents_idx = np.argwhere(convergence_scaled > CONV_THRESHOLD)
        if len(punts_calents_idx) == 0: return {}
            
        punts_lats = grid_lat[punts_calents_idx[:, 0], punts_calents_idx[:, 1]]
        punts_lons = grid_lon[punts_calents_idx[:, 0], punts_calents_idx[:, 1]]
        punts_vals = convergence_scaled[punts_calents_idx[:, 0], punts_calents_idx[:, 1]]
        
        gdf_punts = gpd.GeoDataFrame({'value': punts_vals}, geometry=[Point(lon, lat) for lon, lat in zip(punts_lons, punts_lats)], crs="EPSG:4326")
        
        punts_dins_zones = gpd.sjoin(gdf_punts, gdf_zones, how="inner", predicate="within")
        
        if punts_dins_zones.empty: return {}
            
        # --- Utilitza el nom de la propietat detectada ---
        max_conv_per_zona = punts_dins_zones.groupby(property_name)['value'].max()
        return max_conv_per_zona.to_dict()
        
    except Exception as e:
        print(f"Error dins de calcular_alertes_per_comarca: {e}")
        return {}

def crear_mapa_vents_cat(lons, lats, speed_data, dir_data, nivell, timestamp_str, map_extent):
    fig, ax = crear_mapa_base(map_extent)
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 200), np.linspace(map_extent[2], map_extent[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    
    # Interpolació ràpida
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'linear')
    
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, zorder=3, transform=ccrs.PlateCarree())
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16)
    
    # AFEGIM LES ETIQUETES DE CIUTATS
    afegir_etiquetes_ciutats(ax, map_extent)
    
    return fig

@st.cache_data(ttl=3600)
def carregar_dades_mapa_base_usa(variables, hourly_index):
    try:
        lats, lons = np.linspace(MAP_EXTENT_USA[2], MAP_EXTENT_USA[3], 12), np.linspace(MAP_EXTENT_USA[0], MAP_EXTENT_USA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "gfs_seamless", "forecast_days": 3}
        responses = openmeteo.weather_api(API_URL_USA, params=params)
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
        return None, f"Error en carregar dades del mapa USA: {e}"


@st.cache_data(ttl=1800)
def carregar_dades_mapa_usa(nivell, hourly_index):
    """
    Versió Actualitzada: Carrega dades de mapa per a EUA utilitzant el model HRRR.
    """
    try:
        variables = [f"temperature_{nivell}hPa", f"dew_point_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        lats, lons = np.linspace(MAP_EXTENT_USA[2], MAP_EXTENT_USA[3], 12), np.linspace(MAP_EXTENT_USA[0], MAP_EXTENT_USA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "gfs_hrrr", "forecast_days": 2}
        
        responses = openmeteo.weather_api(API_URL_USA, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        for r in responses:
            try:
                valid_index = trobar_hora_valida_mes_propera(r.Hourly(), hourly_index, len(variables))
                if valid_index is not None:
                    vals = [r.Hourly().Variables(i).ValuesAsNumpy()[valid_index] for i in range(len(variables))]
                    if not any(np.isnan(v) for v in vals):
                        output["lats"].append(r.Latitude()); output["lons"].append(r.Longitude())
                        for i, var in enumerate(variables): output[var].append(vals[i])
            except IndexError: continue

        if not output["lats"]: return None, "No s'han rebut dades per a la graella del mapa."
        
        output['dewpoint_data'] = output.pop(f'dew_point_{nivell}hPa')
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')
        del output[f'temperature_{nivell}hPa']

        return output, None
    except Exception as e:
        return None, f"Error en carregar dades del mapa HRRR: {e}"


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distància en km entre dos punts geogràfics."""
    R = 6371  # Radi de la Terra en km
    dLat, dLon = radians(lat2 - lat1), radians(lon2 - lon1)
    lat1, lat2 = radians(lat1), radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1) * cos(lat2) * sin(dLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_bearing(lat1, lon1, lat2, lon2):
    """Calcula la direcció (bearing) des del punt 1 al punt 2."""
    dLon = radians(lon2 - lon1)
    lat1, lat2 = radians(lat1), radians(lat2)
    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    bearing = degrees(atan2(y, x))
    return (bearing + 360) % 360

def angular_difference(angle1, angle2):
    """Calcula la diferència més curta entre dos angles."""
    diff = abs(angle1 - angle2) % 360
    return diff if diff <= 180 else 360 - diff

def analitzar_amenaça_convergencia_propera(map_data, params_calc, lat_sel, lon_sel, nivell):
    """
    Versió Interactiva v3.0:
    - Si troba una amenaça, no retorna un text, sinó un diccionari amb:
      - 'message': El text de l'avís.
      - 'target_city': El nom del punt d'anàlisi més proper a l'amenaça.
      - 'conv_value': La força exacta de la convergència detectada.
    """
    if not map_data or not params_calc or 'lons' not in map_data or len(map_data['lons']) < 4:
        return None

    moviment_vector = params_calc.get('RM') or params_calc.get('Mean_Wind')
    if not moviment_vector or pd.isna(moviment_vector[0]): return None

    u_storm, v_storm = moviment_vector[0] * units('m/s'), moviment_vector[1] * units('m/s')
    storm_speed_kmh = mpcalc.wind_speed(u_storm, v_storm).to('km/h').m
    storm_dir_to = mpcalc.wind_direction(u_storm, v_storm, convention='to').m

    CONV_THRESHOLD = 30; MAX_DIST_KM = 25; MIN_STORM_SPEED_KMH = 30; ANGLE_TOLERANCE = 45

    if storm_speed_kmh < MIN_STORM_SPEED_KMH: return None

    try:
        lons, lats, speed_data, dir_data = map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data']
        grid_lon, grid_lat = np.meshgrid(np.linspace(min(lons), max(lons), 100), np.linspace(min(lats), max(lats), 100))
        
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        with np.errstate(invalid='ignore'):
            convergence = -mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy).to('1/s').magnitude * 1e5
        
        punts_forts_idx = np.argwhere(convergence > CONV_THRESHOLD)
        if len(punts_forts_idx) == 0: return None

        amenaces_potencials = []
        for idx in punts_forts_idx:
            lat_conv, lon_conv = grid_lat[idx[0], idx[1]], grid_lon[idx[0], idx[1]]
            es_punt_valid = True
            if nivell == 1000: es_punt_valid = globe.is_ocean(lat_conv, lon_conv)
            
            if es_punt_valid:
                dist = haversine_distance(lat_sel, lon_sel, lat_conv, lon_conv)
                if dist <= MAX_DIST_KM:
                    bearing_to_target = get_bearing(lat_conv, lon_conv, lat_sel, lon_sel)
                    if angular_difference(storm_dir_to, bearing_to_target) <= ANGLE_TOLERANCE:
                        # Guardem també el valor de la convergència
                        amenaces_potencials.append({'dist': dist, 'lat': lat_conv, 'lon': lon_conv, 'conv': convergence[idx[0], idx[1]]})
        
        if not amenaces_potencials: return None

        # Tria l'amenaça més propera (la que té la distància mínima)
        amenaça_principal = min(amenaces_potencials, key=lambda x: x['dist'])
        dist_final = amenaça_principal['dist']
        conv_final = amenaça_principal['conv']
        
        # <<-- NOU: Busca el punt d'anàlisi més proper a l'amenaça -->>
        ciutat_mes_propera = min(CIUTATS_CATALUNYA.keys(), 
                                key=lambda ciutat: haversine_distance(amenaça_principal['lat'], amenaça_principal['lon'], 
                                                                    CIUTATS_CATALUNYA[ciutat]['lat'], CIUTATS_CATALUNYA[ciutat]['lon']))

        bearing_from_target = get_bearing(lat_sel, lon_sel, amenaça_principal['lat'], amenaça_principal['lon'])
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        direccio_cardinal = dirs[int(round(bearing_from_target / 22.5)) % 16]

        missatge = f"S'ha detectat un nucli de forta convergència a **{dist_final:.0f} km** al **{direccio_cardinal}**. Les tempestes que es formin allà podrien desplaçar-se cap a la teva posició a uns **{storm_speed_kmh:.0f} km/h**."
        
        # <<-- NOU: Retorna un paquet d'informació complet -->>
        return {
            'message': missatge,
            'target_city': ciutat_mes_propera,
            'conv_value': conv_final
        }

    except Exception:
        return None
    

def crear_mapa_forecast_combinat_uk(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    """
    Crea el mapa visual de vent i convergència per al Regne Unit i Irlanda.
    """
    fig, ax = crear_mapa_base(MAP_EXTENT_UK, projection=ccrs.LambertConformal(central_longitude=-4.5, central_latitude=54))
    if len(lons) < 4: return fig

    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_UK[0], MAP_EXTENT_UK[1], 200), np.linspace(MAP_EXTENT_UK[2], MAP_EXTENT_UK[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic'); grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic'); grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat); convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    convergence_in_humid_areas = np.where(grid_dewpoint >= 12, convergence_scaled, 0)
    
    fill_levels = [5, 10, 15, 25]; fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')

    for city, coords in CIUTATS_UK.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.1, coords['lat'] + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

        
def crear_mapa_forecast_combinat_usa(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    # 1. Crear el mapa base amb la projecció correcta per als EUA
    fig, ax = crear_mapa_base(MAP_EXTENT_USA, projection=ccrs.LambertConformal(central_longitude=-95, central_latitude=35))
    
    # Assegurem que tenim prous dades per a la interpolació
    if len(lons) < 4:
        st.warning("No hi ha prou dades per generar un mapa interpolat.")
        ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
        return fig

    # 2. Crear una graella fina i interpolar les dades del model
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_USA[0], MAP_EXTENT_USA[1], 200), np.linspace(MAP_EXTENT_USA[2], MAP_EXTENT_USA[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')

    # 3. Dibuixar la velocitat del vent amb pcolormesh
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind)
    norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    mesh = ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(f"Velocitat del Vent a {nivell}hPa (km/h)")

    # 4. Dibuixar les línies de corrent del vent (streamplot)
    # --- LÍNIA MODIFICADA ---
    # S'ha afegit el paràmetre 'arrowsize' per controlar la mida de les fletxes.
    # Pots canviar el valor (ex: 0.8) per fer-les més petites o més grans.
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    # 5. Calcular i dibuixar la convergència
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    dudx = mpcalc.first_derivative(grid_u * units('m/s'), delta=dx, axis=1)
    dvdy = mpcalc.first_derivative(grid_v * units('m/s'), delta=dy, axis=0)
    convergence_scaled = -(dudx + dvdy).to('1/s').magnitude * 1e5
    
    DEWPOINT_THRESHOLD_USA = 16 
    convergence_in_humid_areas = np.where(grid_dewpoint >= DEWPOINT_THRESHOLD_USA, convergence_scaled, 0)
    
    fill_levels = [5, 10, 15, 25]
    fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]
    line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.5, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    labels = ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')
    for label in labels:
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
    
    # Afegir ciutats per a referència
    for city, coords in USA_CITIES.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=1, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.2, coords['lat'] + 0.2, city, fontsize=7, transform=ccrs.PlateCarree(), zorder=10,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax.set_title(f"Vent i Nuclis de convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig
# --- Seccions UI i Lògica Principal ---


def calcular_convergencia_puntual(map_data, lat_sel, lon_sel):
    """
    Versió Robusta: Utilitza interpolació 'linear' que és més estable amb
    graelles de dades poc denses i gestiona millor els valors nuls.
    """
    if not map_data or 'lons' not in map_data or len(map_data['lons']) < 4:
        return np.nan

    try:
        lons, lats = map_data['lons'], map_data['lats']
        speed_data, dir_data = map_data['speed_data'], map_data['dir_data']

        points = np.vstack((lons, lats)).T
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        
        # Interpolem directament al punt desitjat, que és més eficient
        u_interpolated = griddata(points, u_comp.to('m/s').m, (lon_sel, lat_sel), method='linear')
        v_interpolated = griddata(points, v_comp.to('m/s').m, (lon_sel, lat_sel), method='linear')
        
        # Per calcular la divergència, necessitem una petita graella al voltant del punt
        grid_lon, grid_lat = np.meshgrid(
            np.linspace(lon_sel - 0.1, lon_sel + 0.1, 3),
            np.linspace(lat_sel - 0.1, lat_sel + 0.1, 3)
        )
        
        grid_u = griddata(points, u_comp.to('m/s').m, (grid_lon, grid_lat), method='linear')
        grid_v = griddata(points, v_comp.to('m/s').m, (grid_lon, grid_lat), method='linear')

        if np.isnan(grid_u).any() or np.isnan(grid_v).any():
            return np.nan

        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)
        
        # Agafem el valor del centre de la nostra petita graella
        convergence_scaled = -divergence[1, 1].to('1/s').magnitude * 1e5
        
        return convergence_scaled if pd.notna(convergence_scaled) else np.nan

    except Exception:
        return np.nan

def hide_streamlit_style():
    """Injecta CSS per amagar el peu de pàgina i el menú de Streamlit."""
    hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
    st.markdown(hide_style, unsafe_allow_html=True)


def format_time_left(total_seconds):
    """Formateja un total de segons en un text llegible (hores i minuts)."""
    if total_seconds <= 0:
        return "ja pots tornar a preguntar"
    
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, _ = divmod(remainder, 60)
    
    if hours > 0:
        return f"d'aquí a {hours}h i {minutes}min"
    else:
        return f"d'aquí a {minutes} min"

def ui_pestanya_assistent_ia(params_calc, poble_sel, pre_analisi, interpretacions_ia, sounding_data=None):
    """
    Crea la interfície d'usuari per a la pestanya de l'assistent d'IA.
    Ara rep una pre-anàlisi i les interpretacions qualitatives per guiar l'IA.
    Incluye ahora información del hodógrafo.
    """
    st.markdown("#### Assistent d'Anàlisi (IA Gemini)")
    
    is_guest = st.session_state.get('guest_mode', False)
    current_user = st.session_state.get('username')
    is_developer = st.session_state.get('developer_mode', False)

    # Mostrar estado del modo desarrollador
    if is_developer:
        st.success("🔓 **MODO DESARROLLADOR ACTIVADO** - Preguntas ilimitadas")
    
    if not is_guest and not is_developer:
        st.info(f"ℹ️ Recorda que tens un límit de **{MAX_IA_REQUESTS} consultes cada 3 hores**.")
    elif is_guest:
        st.info("ℹ️ Fes una pregunta en llenguatge natural sobre les dades del sondeig.")

    # Formulario para activar modo desarrollador
    if not is_developer:
        with st.expander("🔧 Acceso desarrollador"):
            dev_password = st.text_input("Contraseña de desarrollador:", type="password")
            if st.button("Activar modo desarrollador"):
                if dev_password == st.secrets["app_secrets"]["moderator_password"]:
                    st.session_state.developer_mode = True
                    st.rerun()
                else:
                    st.error("Contraseña incorrecta")

    # Mostrar información del hodógrafo si está disponible
    if sounding_data:
        with st.expander("📊 Informació del hodógrafo disponible per a la IA"):
            p, T, Td, u, v, heights, prof = sounding_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                u_sfc, v_sfc = u[0], v[0]
                wind_speed_sfc = mpcalc.wind_speed(u_sfc, v_sfc).to('kt').m
                wind_dir_sfc = mpcalc.wind_direction(u_sfc, v_sfc).m
                st.metric("Vent superfície", f"{wind_speed_sfc:.1f} kt", f"{graus_a_direccio_cardinal(wind_dir_sfc)}")
            
            with col2:
                try:
                    h_500m = 500 * units.meter
                    u_500m = np.interp(h_500m, heights, u)
                    v_500m = np.interp(h_500m, heights, v)
                    wind_speed_500m = mpcalc.wind_speed(u_500m, v_500m).to('kt').m
                    wind_dir_500m = mpcalc.wind_direction(u_500m, v_500m).m
                    st.metric("Vent a 500m", f"{wind_speed_500m:.1f} kt", f"{graus_a_direccio_cardinal(wind_dir_500m)}")
                except:
                    st.metric("Vent a 500m", "N/D")
            
            with col3:
                try:
                    h_3000m = 3000 * units.meter
                    u_3000m = np.interp(h_3000m, heights, u)
                    v_3000m = np.interp(h_3000m, heights, v)
                    wind_speed_3000m = mpcalc.wind_speed(u_3000m, v_3000m).to('kt').m
                    wind_dir_3000m = mpcalc.wind_direction(u_3000m, v_3000m).m
                    st.metric("Vent a 3000m", f"{wind_speed_3000m:.1f} kt", f"{graus_a_direccio_cardinal(wind_dir_3000m)}")
                except:
                    st.metric("Vent a 3000m", "N/D")

    if "messages" not in st.session_state: 
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])

    if prompt := st.chat_input("Fes una pregunta sobre el sondeig..."):
        limit_excedit = False
        
        # Verificar límites solo si no es desarrollador
        if not is_developer and not is_guest and current_user:
            now_ts = time.time()
            rate_limits = load_json_file(RATE_LIMIT_FILE)
            user_timestamps = rate_limits.get(current_user, [])
            recent_timestamps = [ts for ts in user_timestamps if now_ts - ts < TIME_WINDOW_SECONDS]
            
            if len(recent_timestamps) >= MAX_IA_REQUESTS:
                limit_excedit = True
                oldest_ts_in_window = recent_timestamps[0]
                time_to_wait = (oldest_ts_in_window + TIME_WINDOW_SECONDS) - now_ts
                temps_restant_str = format_time_left(time_to_wait)
                st.error(f"Has superat el límit de {MAX_IA_REQUESTS} consultes. Podràs tornar a preguntar {temps_restant_str}.")
            else:
                recent_timestamps.append(now_ts)
                rate_limits[current_user] = recent_timestamps
                save_json_file(rate_limits, RATE_LIMIT_FILE)

        if not limit_excedit or is_developer:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): 
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("El teu amic expert està analitzant les dades..."):
                        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt_complet = generar_prompt_per_ia(params_calc, prompt, poble_sel, pre_analisi, interpretacions_ia, sounding_data)
                        
                        response = model.generate_content(prompt_complet)
                        resposta_completa = response.text
                        st.markdown(resposta_completa)

                    st.session_state.messages.append({"role": "assistant", "content": resposta_completa})
                except Exception as e:
                    st.error(f"Hi ha hagut un error en contactar amb l'assistent d'IA: {e}")
                    
def interpretar_parametres(params, nivell_conv):
    """
    Tradueix els paràmetres numèrics clau a categories qualitatives
    per facilitar la interpretació de l'IA.
    Utilitza MUCIN i MUCAPE (parcel·la de superfície modificada).
    """
    interpretacions = {}

    # --- Interpretació del CIN (ara MUCIN) ---
    mucin = params.get('MUCIN', 0) or 0
    if mucin > -25:
        interpretacions['Inhibició (MUCIN)'] = 'Gairebé Inexistent'
    elif mucin > -75:
        interpretacions['Inhibició (MUCIN)'] = 'Febla, fàcil de trencar'
    elif mucin > -150:
        interpretacions['Inhibició (MUCIN)'] = 'Moderada, cal un bon disparador'
    else:
        interpretacions['Inhibició (MUCIN)'] = 'Molt Forta (Tapa de formigó)'

    # --- Interpretació de la Convergència (Disparador Principal) ---
    conv_key = f'CONV_{nivell_conv}hPa'
    conv = params.get(conv_key, 0) or 0
    if conv < 5:
        interpretacions['Disparador (Convergència)'] = 'Molt Febla o Inexistent'
    elif conv < 15:
        interpretacions['Disparador (Convergència)'] = 'Present'
    elif conv < 30:
        interpretacions['Disparador (Convergència)'] = 'Moderadament Forta'
    else:
        interpretacions['Disparador (Convergència)'] = 'Molt Forta i Decisiva'
    
    # --- Interpretació del CAPE (Combustible) ara MUCAPE ---
    mucape = params.get('MUCAPE', 0) or 0
    if mucape < 300:
        interpretacions['Combustible (MUCAPE)'] = 'Molt Baix'
    elif mucape < 1000:
        interpretacions['Combustible (MUCAPE)'] = 'Moderat'
    elif mucape < 2500:
        interpretacions['Combustible (MUCAPE)'] = 'Alt'
    else:
        interpretacions['Combustible (MUCAPE)'] = 'Extremadament Alt'

    # --- Interpretació del Cisallament (Organització) ---
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    if bwd_6km < 20:
        interpretacions['Organització (Cisallament)'] = 'Febla (Tempestes desorganitzades)'
    elif bwd_6km < 35:
        interpretacions['Organització (Cisallament)'] = 'Moderada (Potencial per a multicèl·lules)'
    else:
        interpretacions['Organització (Cisallament)'] = 'Alta (Potencial per a supercèl·lules)'

    return interpretacions


    


# --- Capitals de comarca agrupades per quadrant ---
CAPITALS_QUADRANTS = {
    "E": ["Barcelona", "Mataró", "Granollers", "Sabadell", "Terrassa", "Manresa", "Girona", "Figueres", "Olot"],
    "NE": ["Girona", "Figueres", "Olot", "Ripoll", "Puigcerdà"],
    "N": ["Vic", "Ripoll", "Berga", "La Seu d’Urgell", "Puigcerdà", "Sort"],
    "NW": ["Lleida", "Balaguer", "La Seu d’Urgell", "Sort", "Tremp"],
    "W": ["Lleida", "Balaguer", "Cervera", "Tàrrega", "Mollerussa"],
    "SW": ["Tarragona", "Reus", "Falset", "Gandesa", "Tortosa"],
    "S": ["Tarragona", "Reus", "Valls", "Amposta", "Tortosa"],
    "SE": ["Barcelona", "Vilanova i la Geltrú", "Vilafranca del Penedès", "Tarragona", "Igualada"]
}


def direccio_moviment(des_de_graus):
    """
    Converteix la direcció del vent (d'on ve) en la trajectòria real (cap on va).
    """
    cap_on_va = (des_de_graus + 180) % 360
    return cap_on_va


def quadrant_capitals(cap_on_va):
    """
    Dona el quadrant cardinal i les capitals de comarca a vigilar.
    """
    if 337.5 <= cap_on_va or cap_on_va < 22.5:
        return "N", CAPITALS_QUADRANTS["N"]
    elif 22.5 <= cap_on_va < 67.5:
        return "NE", CAPITALS_QUADRANTS["NE"]
    elif 67.5 <= cap_on_va < 112.5:
        return "E", CAPITALS_QUADRANTS["E"]
    elif 112.5 <= cap_on_va < 157.5:
        return "SE", CAPITALS_QUADRANTS["SE"]
    elif 157.5 <= cap_on_va < 202.5:
        return "S", CAPITALS_QUADRANTS["S"]
    elif 202.5 <= cap_on_va < 247.5:
        return "SW", CAPITALS_QUADRANTS["SW"]
    elif 247.5 <= cap_on_va < 292.5:
        return "W", CAPITALS_QUADRANTS["W"]
    elif 292.5 <= cap_on_va < 337.5:
        return "NW", CAPITALS_QUADRANTS["NW"]
    else:
        return None, []

def graus_a_direccio_cardinal(graus):
    direccions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    index = round(graus / 22.5) % 16
    return direccions[index]

def  generar_prompt_per_ia(params, pregunta_usuari, poble, pre_analisi, interpretacions_ia, sounding_data=None, historical_context=None, user_preferences=None):
    """
    Genera un prompt potent però concís per a la IA, buscant respostes breus,
    directes i altament útils per a un meteoròleg expert.
    """

    # --- 1. Extracció i processament de paràmetres clau (mantinguts igual) ---
    mucape = params.get('MUCAPE', 0) or 0
    cape_ml = params.get('MLCAPE', 0) or 0
    cin_ml = params.get('MLCIN', 0) or 0
    li = params.get('LI', 999)

    bwd_0_6km = params.get('BWD_0-6km', 0) or 0
    srh_0_3km = params.get('SRH_0-3km', 0) or 0
    srh_0_1km = params.get('SRH_0-1km', 0) or 0

    pw = params.get('PW', 0) or 0
    lcl_height = params.get('LCL_HEIGHT', None)
    lfc_height = params.get('LFC_HEIGHT', None)
    el_height = params.get('EL_HEIGHT', None)

    temp_850hpa = params.get('TEMP_850hPa', None)
    temp_500hpa = params.get('TEMP_500hPa', None)
    geopot_500hpa = params.get('GEOPOT_500hPa', None)

    conv_key = f'CONV_{params.get("level_cat_main_conv", 925)}hPa'
    convergencia_nivell_baix = params.get(conv_key, 0) or 0

    hail_potential = params.get('HAIL_POTENTIAL', 'Baix')
    tornado_potential = params.get('TORNADO_POTENTIAL', 'Baix')
    wind_potential = params.get('WIND_POTENTIAL', 'Baix')

    temp_superficie = params.get('TEMP_SURFACE', None)
    dewp_superficie = params.get('DEWP_SURFACE', None)
    
    cin = params.get('MUCIN', 0) or 0
    eff_srh = params.get('EFF_SRH', 0) or 0
    eff_shear = params.get('EFF_SHEAR', 0) or 0
    lapse_rate_700_500 = params.get('LAPSE_RATE_700_500', 0) or 0

    direccion_sistema = "No determinable"
    flux_baix_nivell = "No determinable"
    if sounding_data:
        try:
            p, u, v, heights = sounding_data[0], sounding_data[3], sounding_data[4], sounding_data[5]
            
            mask_0_6km = (heights >= 0 * units.meter) & (heights <= 6000 * units.meter)
            mean_u_0_6km = np.mean(u[mask_0_6km])
            mean_v_0_6km = np.mean(v[mask_0_6km])
            dir_from_0_6km = mpcalc.wind_direction(mean_u_0_6km, mean_v_0_6km).m
            dir_to_0_6km = (dir_from_0_6km + 180) % 360
            direccion_sistema = f"{dir_to_0_6km:.0f}° ({graus_a_direccio_cardinal(dir_to_0_6km)})"
            
            mask_0_1km = (heights >= 0 * units.meter) & (heights <= 1000 * units.meter)
            mean_u_0_1km = np.mean(u[mask_0_1km])
            mean_v_0_1km = np.mean(v[mask_0_1km])
            dir_from_0_1km = mpcalc.wind_direction(mean_u_0_1km, mean_v_0_1km).m
            dir_to_0_1km = (dir_from_0_1km + 180) % 360
            flux_baix_nivell = f"{dir_to_0_1km:.0f}° ({graus_a_direccio_cardinal(dir_to_0_1km)})"

        except Exception as e:
            direccion_sistema = f"Error: {e}"
            flux_baix_nivell = "No determinable"

    # --- 2. Definició de l'estructura del prompt (ara més concisa) ---

    prompt_parts = []

    # ### ROL I OBJECTIU (Més resumit) ###
    prompt_parts.append(
        "**ROL**: Ets un meteoròleg expert en temps sever. "
        "**OBJECTIU**: Proporciona una anàlisi concisa i molt resumida amb to graciós, accionable i predictiva de la situació meteorològica per a l'usuari, identificant riscos i recomanacions clau."
    )

    prompt_parts.append(f"\n### ANÀLISI PER A {poble} ###")
    prompt_parts.append(f"**Pregunta de l'usuari**: {pregunta_usuari}\n")

    # ### DADES DE DIAGNÒSTIC METEOROLÒGIC (Agrupades per brevetat) ###
    prompt_parts.append("#### Condicions Clau ####")
    prompt_parts.append(f"- **Energia Convectiva (MUCAPE/MLCAPE)**: {mucape:.0f}/{cape_ml:.0f} J/kg")
    prompt_parts.append(f"- **Inhibició (MUCIN/MLCIN)**: {cin:.0f}/{cin_ml:.0f} J/kg")
    prompt_parts.append(f"- **Cisallament (0-6km/Eff.)**: {bwd_0_6km:.0f}/{eff_shear:.0f} nusos")
    prompt_parts.append(f"- **Helicitat (SRH 0-3km/0-1km/Eff.)**: {srh_0_3km:.0f}/{srh_0_1km:.0f}/{eff_srh:.0f} m²/s²")
    
    extra_data = []
    if pw > 0: extra_data.append(f"PW: {pw:.1f} mm")
    if lcl_height is not None: extra_data.append(f"LCL: {lcl_height:.0f} m")
    if lfc_height is not None: extra_data.append(f"LFC: {lfc_height:.0f} m")
    if temp_superficie is not None: extra_data.append(f"Temp/Dewp Sup.: {temp_superficie:.1f}/{dewp_superficie:.1f}°C")
    if convergencia_nivell_baix != 0: extra_data.append(f"Conv. {params.get('level_cat_main_conv', 925)}hPa: {convergencia_nivell_baix:.1f} ×10⁻⁵ s⁻¹")
    if direccion_sistema != "No determinable": extra_data.append(f"Dir. Sist.: {direccion_sistema}")
    if flux_baix_nivell != "No determinable": extra_data.append(f"Flux Baix: {flux_baix_nivell}")
    if li != 999: extra_data.append(f"LI: {li:.1f}")
    if lapse_rate_700_500 != 0: extra_data.append(f"LR 700-500: {lapse_rate_700_500:.1f}°C/km")

    if extra_data:
        prompt_parts.append(f"- **Altres indicadors**: {', '.join(extra_data)}")

    # ### INTERPRETACIÓ AUTOMÀTICA / PREDICCIÓ INICIAL ###
    prompt_parts.append("\n### Veredicte Preliminar ###")
    prompt_parts.append(f"- **General**: {pre_analisi.get('veredicte', 'No determinat')}. Tipus de núvol: {pre_analisi.get('descripcio', 'No determinat')}")
    prompt_parts.append(f"- **Potencial de Riscos**: Calamarsa: {hail_potential}. Tornado: {tornado_potential}. Vent fort: {wind_potential}.")
    
    if interpretacions_ia:
        prompt_parts.append("\n**Detalls AI Addicionals**:")
        for key, value in interpretacions_ia.items():
            prompt_parts.append(f"- {key}: {value}")

    # ### CONTEXT I PREFERÈNCIES ###
    if historical_context or user_preferences:
        prompt_parts.append("\n### Context i Interès ###")
        if historical_context:
            prompt_parts.append(f"- **Precedents**: {'; '.join(historical_context)}")
        if user_preferences:
            prompt_parts.append(f"- **Prioritat de l'usuari**: {'; '.join(user_preferences)}")

    # ### INSTRUCCIONS DETALLADES PER A LA IA (Enfocades a la concisió) ###
    prompt_parts.append("\n### INSTRUCCIONS CLAU PER A LA RESPOSTA ###")
    prompt_parts.append(
        "- **Anàlisi Experta**: Interpreta les dades de forma integrada, no les repeteixis. Explica què signifiquen en termes de risc real.\n"
        "- **Focus en Risc**: Detalla el potencial de cada fenomen sever (calamarsa, tornados, vent, pluja) per a {poble}.\n"
        "- **Evolució**: Indica moments clau i tendències. Considera escenaris alternatius si hi ha incertesa.\n"
        "- **Recomanacions**: Proporciona consells pràctics i directes. Respon a la pregunta de l'usuari de forma explícita.\n"
        "- **Concisió i Utilitat**: La resposta ha de ser breu, clara, divertida, directa i extremadament útil."
    )

    prompt_parts.append("\n### FORMAT DE RESPOSTA DESITJAT ###")
    prompt_parts.append(
        "**IDIOMA**: Català.\n"
        "**ESTRUCTURA**: Màxim 2-3 paràgrafs. Pots utilitzar llistes curtes per punts clau.\n"
        "**CONTINGUT**: 1) Diagnòstic concís, 2) Riscos i evolució, 3) Recomanacions i resposta a l'usuari.\n"
        "**TO**: Professional i directe. Evita introduccions o conclusions genèriques."
    )
    prompt_parts.append("---")
    prompt_parts.append("Comença la teva anàlisi concisa ara mateix.")

    return "\n".join(prompt_parts)





    
def hide_streamlit_style():
    """Injecta CSS per amagar el peu de pàgina i el menú de Streamlit."""
    hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Neteja de l'estat visual quan canvia la selecció */
        .stApp {
            transition: all 0.3s ease;
        }
        .element-container:has(.stSelectbox) {
            z-index: 1000;
            position: relative;
        }
        </style>
        """
    st.markdown(hide_style, unsafe_allow_html=True)



# --- BLOC 1: SUBSTITUEIX TOTES LES FUNCIONS on_... PER AQUESTES ---

def on_day_change_cat():
    """ Callback segur per al canvi de dia a Catalunya. """
    st.session_state.hora_selector = "12:00h"

def on_day_change_usa():
    """ Callback segur per al canvi de dia als EUA. """
    try:
        hora_num = int(st.session_state.hora_selector_usa.split(':')[0])
    except:
        hora_num = 12
    dia_sel_str = st.session_state.dia_selector_usa_widget
    target_date = datetime.strptime(dia_sel_str, '%d/%m/%Y').date()
    time_in_usa = TIMEZONE_USA.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_num))
    time_in_spain = time_in_usa.astimezone(TIMEZONE_CAT)
    st.session_state.hora_selector_usa = f"{time_in_usa.hour:02d}:00 (Local: {time_in_spain.hour:02d}:00h)"

def on_city_change(source_widget_key, other_widget_key, placeholder_text, city_dict):
    """
    Funció de callback genèrica i robusta per gestionar el canvi de ciutat.
    """
    selected_raw_text = st.session_state[source_widget_key]
    
    if placeholder_text in selected_raw_text:
        return # No facis res si s'ha seleccionat el placeholder

    # Troba la clau original correcta sense perill de KeyErrors
    found_key = None
    for key in city_dict.keys():
        if selected_raw_text.startswith(key):
            found_key = key
            break
            
    if found_key:
        # Si la selecció és vàlida, actualitza l'estat principal
        if 'poble_selector' in st.session_state:
             st.session_state.poble_selector = found_key
        elif 'poble_selector_usa' in st.session_state:
            st.session_state.poble_selector_usa = found_key

        # Important: Reseteja l'ALTRE selector només si existeix
        if other_widget_key and other_widget_key in st.session_state:
            # Utilitzem una clau interna per evitar disparar el callback de l'altre
            st.session_state[other_widget_key] = placeholder_text



def ui_capcalera_selectors(ciutats_a_mostrar, info_msg=None, zona_activa="catalunya", convergencies=None):
    st.markdown(f'<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | {zona_activa.replace("_", " ").title()}</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)
    
    altres_zones = {'catalunya': 'Catalunya', 'valley_halley': 'Tornado Alley', 'alemanya': 'Alemanya', 'italia': 'Itàlia', 'holanda': 'Holanda', 'japo': 'Japó', 'uk': 'Regne Unit', 'canada': 'Canadà', 'noruega': 'Noruega'}
    del altres_zones[zona_activa]
    
    col_text, col_nav, col_back, col_logout = st.columns([0.5, 0.2, 0.15, 0.15])
    
    with col_text:
        if not is_guest: st.markdown(f"Benvingut/da, **{st.session_state.get('username', 'Usuari')}**!")
    with col_nav:
        nova_zona_key = st.selectbox("Canviar a:", options=list(altres_zones.keys()), format_func=lambda k: altres_zones[k], index=None, placeholder="Anar a...")
        if nova_zona_key: st.session_state.zone_selected = nova_zona_key; st.rerun()
    with col_back:
        if st.button("⬅️ Zones", use_container_width=True, help="Tornar a la selecció de zona"):
            keys_to_clear = [k for k in st.session_state if k not in ['logged_in', 'username', 'guest_mode', 'developer_mode']]
            for key in keys_to_clear: del st.session_state[key]
            st.rerun()
    with col_logout:
        if st.button("Sortir" if is_guest else "Tanca Sessió", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    with st.container(border=True):
        # La secció de Catalunya és l'única que no es mostra aquí,
        # ja que els seus selectors globals estan a la seva pròpia funció.
        if zona_activa == 'catalunya':
            pass
        
        # Per a la resta de zones, només mostrem el selector de ciutat.
        elif zona_activa == 'valley_halley':
            st.selectbox("Ciutat:", options=sorted(list(USA_CITIES.keys())), key="poble_selector_usa")
        elif zona_activa == 'alemanya':
            st.selectbox("Ciutat:", options=sorted(list(CIUTATS_ALEMANYA.keys())), key="poble_selector_alemanya")
        elif zona_activa == 'italia':
            st.selectbox("Ciutat:", options=sorted(list(CIUTATS_ITALIA.keys())), key="poble_selector_italia")
        elif zona_activa == 'holanda':
            st.selectbox("Ciutat:", options=sorted(list(CIUTATS_HOLANDA.keys())), key="poble_selector_holanda")
        elif zona_activa == 'japo':
            st.selectbox("Ciutat:", options=sorted(list(CIUTATS_JAPO.keys())), key="poble_selector_japo")
        elif zona_activa == 'uk':
            st.selectbox("Ciutat:", options=sorted(list(CIUTATS_UK.keys())), key="poble_selector_uk")
        elif zona_activa == 'canada':
            st.selectbox("Ciutat:", options=sorted(list(CIUTATS_CANADA.keys())), key="poble_selector_canada")
        elif zona_activa == 'noruega':
            st.selectbox("Ciutat:", options=sorted(list(CIUTATS_NORUEGA.keys())), key="poble_selector_noruega")


def ui_pestanya_mapes_japo(hourly_index_sel, timestamp_str, nivell_sel, poble_sel):
    st.markdown("#### Mapes de Pronòstic (Model JMA GSM)")
    with st.spinner("Carregant mapa JMA GSM... El primer cop pot trigar una mica."):
        map_data, error = carregar_dades_mapa_japo(nivell_sel, hourly_index_sel)
    
        if error or not map_data:
            # <<<--- CORRECCIÓ AQUÍ --->>>
            st.error(f"Error en carregar el mapa: {error if error else 'No s''han rebut dades.'}")
        else:
            timestamp_map_title = timestamp_str.replace(f"{poble_sel} | ", "")
            fig = crear_mapa_forecast_combinat_japo(
                map_data['lons'], map_data['lats'], map_data['speed_data'], 
                map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, 
                timestamp_map_title
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

def ui_pestanya_mapes_uk(hourly_index_sel, timestamp_str, nivell_sel, poble_sel):
    st.markdown("#### Mapes de Pronòstic (Model UKMO 2km)")
    with st.spinner("Carregant mapa UKMO... El primer cop pot trigar una mica."):
        map_data, error = carregar_dades_mapa_uk(nivell_sel, hourly_index_sel)
    
        if error or not map_data:
            # <<<--- CORRECCIÓ AQUÍ --->>>
            st.error(f"Error en carregar el mapa: {error if error else 'No s''han rebut dades.'}")
        else:
            timestamp_map_title = timestamp_str.replace(f"{poble_sel} | ", "")
            fig = crear_mapa_forecast_combinat_uk(
                map_data['lons'], map_data['lats'], map_data['speed_data'], 
                map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, 
                timestamp_map_title
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


@st.cache_resource(ttl=1800, show_spinner=False)
def generar_mapa_cachejat_cat(hourly_index, nivell, timestamp_str, map_extent_tuple):
    """
    Funció generadora que crea i desa a la memòria cau el mapa de convergència.
    Només s'executa si els paràmetres (hora, nivell, zoom) canvien.
    """
    map_data, error = carregar_dades_mapa_cat(nivell, hourly_index)
    if error or not map_data:
        # Retorna None si no es poden carregar les dades
        return None
    
    # El tuple es converteix de nou a llista per a la funció de dibuix
    map_extent_list = list(map_extent_tuple)
    
    fig = crear_mapa_forecast_combinat_cat(
        map_data['lons'], map_data['lats'], 
        map_data['speed_data'], map_data['dir_data'], 
        map_data['dewpoint_data'], nivell, 
        timestamp_str, map_extent_list
    )
    return fig

@st.cache_resource(ttl=1800, show_spinner=False)
def generar_mapa_vents_cachejat_cat(hourly_index, nivell, timestamp_str, map_extent_tuple):
    """
    Funció generadora que crea i desa a la memòria cau els mapes de vent (700/300hPa).
    """
    variables = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
    map_data, error = carregar_dades_mapa_base_cat(variables, hourly_index)
    
    if error or not map_data:
        return None
        
    map_extent_list = list(map_extent_tuple)
    
    fig = crear_mapa_vents_cat(
        map_data['lons'], map_data['lats'], 
        map_data[variables[0]], map_data[variables[1]], 
        nivell, timestamp_str, map_extent_list
    )
    return fig


def ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel):
    st.markdown("#### Mapes de Pronòstic (Model AROME)")
    
    col_capa, col_zoom = st.columns(2)
    with col_capa:
        mapa_sel = st.selectbox("Selecciona la capa del mapa:", 
                               ["Anàlisi de Vent i Convergència", "Vent a 700hPa", "Vent a 300hPa"], 
                               key="map_cat")
    with col_zoom: 
        zoom_sel = st.selectbox("Nivell de Zoom:", 
                               options=list(MAP_ZOOM_LEVELS_CAT.keys()), 
                               key="zoom_cat")
    
    selected_extent = MAP_ZOOM_LEVELS_CAT[zoom_sel]
    
    with st.spinner(f"Carregant i generant mapa... (només la primera vegada)"):
        if "Convergència" in mapa_sel:
            fig = generar_mapa_cachejat_cat(hourly_index_sel, nivell_sel, timestamp_str, tuple(selected_extent))
            if fig is None:
                st.error(f"Error en carregar les dades per al mapa de convergència.")
            else:
                st.pyplot(fig, use_container_width=True)
        
        else:
            nivell_vent = 700 if "700" in mapa_sel else 300
            fig = generar_mapa_vents_cachejat_cat(hourly_index_sel, nivell_vent, timestamp_str, tuple(selected_extent))
            if fig is None:
                st.error(f"Error en carregar les dades per al mapa de vent a {nivell_vent}hPa.")
            else:
                st.pyplot(fig, use_container_width=True)

    # <<-- LÍNIA CLAU AFEGIDA: Crida a la funció que dibuixa l'explicació -->>
    if "Convergència" in mapa_sel:
        ui_explicacio_convergencia()
            
def ui_pestanya_analisis_vents(data_tuple, poble_sel, hora_actual_str, timestamp_str):
    """
    Versió 2.0. Substitueix el gràfic estàtic per dials de vent animats
    per a Superfície, 925 hPa i 700 hPa.
    """
    st.markdown(f"#### Anàlisi de Vents per a {poble_sel}")
    st.caption(timestamp_str)

    if not data_tuple:
        st.warning("No hi ha dades de sondeig disponibles per realitzar l'anàlisi de vents.")
        return

    sounding_data, _ = data_tuple
    
    # La secció de diagnòstic es manté igual
    diagnostics = analitzar_vents_locals(sounding_data, poble_sel, hora_actual_str)
    if len(diagnostics) == 1 and diagnostics[0]['titol'] == "Anàlisi no disponible":
        st.info(f"📍 {diagnostics[0]['descripcio']}")
        # Encara que no hi hagi anàlisi local, podem mostrar els dials de vent sinòptic
    else:
        st.markdown("##### Diagnòstic de Fenòmens Eòlics")
        for diag in diagnostics:
            with st.expander(f"{diag['emoji']} **{diag['titol']}**", expanded=True):
                st.write(diag['descripcio'])
    
    st.divider()
    st.markdown("##### Perfil de Vent per Nivells Clau")

    # --- NOVA LÒGICA PER ALS DIALS DE VENT ---
    p, u, v = sounding_data[0], sounding_data[3], sounding_data[4]
    
    # 1. Vent de Superfície (SFC)
    dir_sfc = mpcalc.wind_direction(u[0], v[0]).m
    spd_sfc = mpcalc.wind_speed(u[0], v[0]).to('km/h').m

    # 2. Vent a 925 hPa (interpolar)
    try:
        if p.m.min() <= 925:
            u_925 = np.interp(925, p.m[::-1], u.m[::-1]) * units('m/s')
            v_925 = np.interp(925, p.m[::-1], v.m[::-1]) * units('m/s')
            dir_925 = mpcalc.wind_direction(u_925, v_925).m
            spd_925 = mpcalc.wind_speed(u_925, v_925).to('km/h').m
        else: # Cas d'alta muntanya on 925hPa està sota terra
            dir_925, spd_925 = np.nan, np.nan
    except Exception:
        dir_925, spd_925 = np.nan, np.nan
        
    # 3. Vent a 700 hPa (interpolar)
    try:
        if p.m.min() <= 700:
            u_700 = np.interp(700, p.m[::-1], u.m[::-1]) * units('m/s')
            v_700 = np.interp(700, p.m[::-1], v.m[::-1]) * units('m/s')
            dir_700 = mpcalc.wind_direction(u_700, v_700).m
            spd_700 = mpcalc.wind_speed(u_700, v_700).to('km/h').m
        else:
            dir_700, spd_700 = np.nan, np.nan
    except Exception:
        dir_700, spd_700 = np.nan, np.nan

    # Dibuixem els 3 dials en columnes
    col1, col2, col3 = st.columns(3)
    with col1:
        html_sfc = crear_dial_vent_animat("Superfície", dir_sfc, spd_sfc)
        st.markdown(html_sfc, unsafe_allow_html=True)
    with col2:
        html_925 = crear_dial_vent_animat("925 hPa", dir_925, spd_925)
        st.markdown(html_925, unsafe_allow_html=True)
    with col3:
        html_700 = crear_dial_vent_animat("700 hPa", dir_700, spd_700)
        st.markdown(html_700, unsafe_allow_html=True)


def ui_analisi_regims_de_vent(analisi_resultat):
    """
    Mostra la caixa d'anàlisi de règims de vent a la interfície.
    """
    st.markdown("##### Anàlisi del Règim de Vent Dominant")
    with st.container(border=True):
        tipus_vent = analisi_resultat['tipus']
        color = analisi_resultat['color']
        veredicte = analisi_resultat['veredicte']
        detall = analisi_resultat['detall']

        # --- LÒGICA D'EMOJIS FINAL ---
        emoji = "💨"
        if "Llevant" in tipus_vent: emoji = "🌊"
        if "Marinada" in tipus_vent: emoji = "☀️"
        if "Rebuf" in tipus_vent: emoji = "🍂"
        # --- NOU BLOC PER A ADVECCIÓ ---
        if "Advecció" in tipus_vent:
            if "Humida" in tipus_vent:
                emoji = "💧" # Gota per a humida
            else:
                emoji = "🌫️" # Boira per a seca
        # --------------------------------
        if "Ponentada" in tipus_vent or "Nortada" in tipus_vent: emoji = "🌬️"
        if "Terral" in tipus_vent: emoji = "🏜️"
        if "Nocturn" in tipus_vent: emoji = "🌙"
        if "Calma" in tipus_vent: emoji = "🧘"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span style="font-size: 1.1em; color: #FAFAFA;">Règim Detectat</span><br>
                <strong style="font-size: 2.2em; color: {color};">{emoji} {tipus_vent}</strong>
            </div>""", unsafe_allow_html=True)
        with col2:
             st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span style="font-size: 1.1em; color: #FAFAFA;">Detalls del Règim</span><br>
                <strong style="font-size: 1.5em; color: #FFFFFF;">{detall}</strong>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"<p style='text-align: center; font-size: 1.1em; padding: 0 15px;'><strong>Veredicte:</strong> {veredicte}</p>", unsafe_allow_html=True)
        
def ui_pestanya_mapes_usa(hourly_index_sel, timestamp_str, nivell_sel):
    st.markdown("#### Mapes de Pronòstic (Model HRRR)")
    
    with st.spinner("Carregant mapa HRRR... El primer cop pot trigar una mica."):
        map_data, error_map = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
    
        if error_map:
            st.error(f"Error en carregar el mapa: {error_map}")
        elif map_data:
            fig = crear_mapa_forecast_combinat_usa(
                map_data['lons'], map_data['lats'], map_data['speed_data'], 
                map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, 
                timestamp_str
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        
def ui_pestanya_satelit_usa():
    st.markdown("#### Imatge de Satèl·lit GOES-East (Temps Real)")
    
    # --- LÍNIA CORREGIDA ---
    # S'ha canviat l'URL del satèl·lit de MESO (mòbil) a CONUS (fixa).
    sat_url = f"https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/GEOCOLOR/latest.jpg?{int(time.time())}"
    
    st.image(sat_url, caption="Imatge del satèl·lit GOES-East - Vista CONUS (NOAA STAR)", use_container_width=True)
    
    st.info(
        """
        Aquesta imatge mostra la vista **CONUS (Contiguous United States)**, que cobreix tots els Estats Units continentals. 
        S'actualitza cada 5-10 minuts i garanteix que sempre puguem veure la "Tornado Alley", a diferència dels sectors de mesoescala mòbils.
        """
    )
    st.markdown("<p style='text-align: center;'>[Font: NOAA STAR](https://www.star.nesdis.noaa.gov/GOES/index.php)</p>", unsafe_allow_html=True)




@st.cache_data(ttl=600)
def obtenir_dades_estacio_smc():
    try: api_key = st.secrets["SMC_API_KEY"]
    except KeyError: return None, "Falta la clau 'SMC_API_KEY' als secrets."
    url = "https://api.meteo.cat/xema/v1/observacions/mesurades/ultimes"; headers = {"X-Api-Key": api_key}
    try:
        response = requests.get(url, headers=headers, timeout=15); response.raise_for_status(); return response.json(), None
    except requests.exceptions.RequestException as e: return None, f"Error de xarxa en contactar amb l'API de l'SMC: {e}"

def ui_pestanya_estacions_meteorologiques():
    st.markdown("#### Dades en Temps Real (Xarxa d'Estacions de l'SMC)")
    if "SMC_API_KEY" not in st.secrets or not st.secrets["SMC_API_KEY"]:
        st.info("🚧 **Pestanya en Desenvolupament**\n\nAquesta secció està pendent de la validació de la clau d'accés a les dades oficials del Servei Meteorològic de Catalunya (SMC).", icon="🚧")
        return

    st.caption("Dades oficials de la Xarxa d'Estacions Meteorològiques Automàtiques (XEMA) del Servei Meteorològic de Catalunya.")
    with st.spinner("Carregant dades de la XEMA..."): dades_xema, error = obtenir_dades_estacio_smc()
    if error: st.error(error); return
    if not dades_xema: st.warning("No s'han pogut carregar les dades de les estacions de l'SMC."); return
    
    col1, col2 = st.columns([0.6, 0.4], gap="large")
    with col1:
        st.markdown("##### Mapa d'Ubicacions")
        fig, ax = crear_mapa_base(MAP_EXTENT_CAT)
        for ciutat, coords in CIUTATS_CATALUNYA.items():
            if ciutat in SMC_STATION_CODES:
                lon, lat = coords['lon'], coords['lat']
                ax.plot(lon, lat, 'o', color='darkblue', markersize=8, markeredgecolor='white', transform=ccrs.PlateCarree(), zorder=10)
                ax.text(lon + 0.03, lat, ciutat, fontsize=7, transform=ccrs.PlateCarree(), zorder=11, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col2:
        st.markdown("##### Dades de l'Estació")
        ciutat_seleccionada = st.selectbox("Selecciona una capital de comarca:", options=sorted(SMC_STATION_CODES.keys()))
        if ciutat_seleccionada:
            station_code = SMC_STATION_CODES.get(ciutat_seleccionada)
            dades_estacio = next((item for item in dades_xema if item.get("codi") == station_code), None)
            if dades_estacio:
                nom = dades_estacio.get("nom", "N/A"); data = dades_estacio.get("data", "N/A").replace("T", " ").replace("Z", "")
                variables = {var['codi']: var['valor'] for var in dades_estacio.get('variables', [])}
                st.info(f"**Estació:** {nom} | **Lectura:** {data} UTC")
                c1, c2 = st.columns(2)
                c1.metric("Temperatura", f"{variables.get(32, '--')} °C"); c2.metric("Humitat", f"{variables.get(33, '--')} %")
                st.metric("Pressió atmosfàrica", f"{variables.get(35, '--')} hPa")
                st.metric("Vent", f"{variables.get(31, '--')}° a {variables.get(30, '--')} km/h (Ràfega: {variables.get(2004, '--')} km/h)")
                st.metric("Precipitació (30 min)", f"{variables.get(34, '--')} mm")
                st.markdown(f"🔗 [Veure a la web de l'SMC](https://www.meteo.cat/observacions/xema/dades?codi={station_code})", unsafe_allow_html=True)
            else: st.error("No s'han trobat dades recents per a aquesta estació.")

def ui_peu_de_pagina():
    st.divider(); st.markdown("<p style='text-align: center; font-size: 0.9em; color: grey;'>Dades AROME/GFS via Open-Meteo | Imatges via Meteociel & NOAA | IA per Google Gemini.</p>", unsafe_allow_html=True)





def run_canada_app():
    if 'poble_selector_canada' not in st.session_state: st.session_state.poble_selector_canada = "Calgary, AB"
    ui_capcalera_selectors(None, zona_activa="canada")
    poble_sel = st.session_state.poble_selector_canada
    now_local = datetime.now(TIMEZONE_CANADA)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = CIUTATS_CANADA[poble_sel]['lat'], CIUTATS_CANADA[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} ({TIMEZONE_CANADA.zone}) / {cat_dt.strftime('%d/%m, %H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]

    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_canada", default_index=0)

    if st.session_state.active_tab_canada == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig HRDPS per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_canada(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_CANADA)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_canada(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_canada == "Anàlisi de Mapes":
        ui_pestanya_mapes_canada(hourly_index_sel, timestamp_str, nivell_sel, poble_sel)
    elif st.session_state.active_tab_canada == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="canada")

def run_catalunya_app():
    # --- PAS 1: CAPÇALERA I NAVEGACIÓ GLOBAL ---
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)

    col_text, col_change, col_logout = st.columns([0.7, 0.15, 0.15])
    with col_text:
        if not is_guest:
            st.markdown(f"Benvingut/da, **{st.session_state.get('username', 'Usuari')}**!")
    with col_change:
        if st.button("Canviar de Zona", use_container_width=True, help="Torna a la selecció de zona geogràfica"):
            st.session_state.zone_selected = None
            keys_to_clear = ['selected_area', 'poble_sel', 'active_tab_cat']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
    with col_logout:
        if st.button("Sortir" if is_guest else "Tanca Sessió", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    st.divider()

    # --- PAS 2: GESTIÓ D'ESTAT UNIFICADA ---
    if 'selected_area' not in st.session_state: st.session_state.selected_area = "--- Selecciona una zona al mapa ---"
    if 'poble_sel' not in st.session_state: st.session_state.poble_sel = "--- Selecciona una localitat ---"

    # --- PAS 3: SELECTORS GLOBALS DE TEMPS ---
    with st.container(border=True):
        col_dia, col_hora, col_nivell = st.columns(3)
        with col_dia:
            st.selectbox("Dia:", options=[(datetime.now(TIMEZONE_CAT) + timedelta(days=i)).strftime('%d/%m/%Y') for i in range(2)], key="dia_selector")
        with col_hora:
            st.selectbox("Hora:", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector", index=datetime.now(TIMEZONE_CAT).hour)
        with col_nivell:
            st.selectbox("Nivell d'Anàlisi:", options=[1000, 950, 925, 900, 850, 800, 700], key="level_cat_main", index=2, format_func=lambda x: f"{x} hPa")

    # --- PAS 4: CÀLCUL DE DADES GLOBALS ---
    dia_sel_str = st.session_state.dia_selector
    hora_sel_str = st.session_state.hora_selector
    nivell_sel = st.session_state.level_cat_main
    
    target_date = datetime.strptime(dia_sel_str, '%d/%m/%Y').date()
    local_dt = TIMEZONE_CAT.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_str.split(':')[0])))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    
    alertes_zona = calcular_alertes_per_comarca(hourly_index_sel, nivell_sel)

    # --- PAS 5: LÒGICA PRINCIPAL (SELECCIÓ O ANÀLISI) ---
    if st.session_state.poble_sel and "---" not in st.session_state.poble_sel:
        # --- VISTA D'ANÀLISI DETALLADA ---
        poble_sel = st.session_state.poble_sel
        st.success(f"### Anàlisi per a: **{poble_sel}**")
        if st.button("⬅️ Tornar al mapa de selecció"):
            st.session_state.poble_sel = "--- Selecciona una localitat ---"
            st.session_state.selected_area = "--- Selecciona una zona al mapa ---"
            if 'active_tab_cat' in st.session_state: del st.session_state['active_tab_cat']
            st.rerun()

        # ... (La resta de la lògica de la vista d'anàlisi no canvia i es manté igual) ...
        lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']
        timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} (Local)"
        menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Anàlisi de Vents", "Simulació de Núvol"]
        menu_icons = ["graph-up-arrow", "map", "wind", "cloud-upload"]
        if not is_guest:
            menu_options.append("💬 Assistent IA")
            menu_icons.append("chat-quote-fill")
        option_menu(menu_title=None, options=menu_options, icons=menu_icons, menu_icon="cast", orientation="horizontal", key="active_tab_cat", default_index=0)
        if st.session_state.active_tab_cat == "Anàlisi de Mapes":
            ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel)
        else:
            with st.spinner(f"Carregant dades del sondeig AROME per a {poble_sel}..."):
                data_tuple, final_index, error_msg = carregar_dades_sondeig_cat(lat_sel, lon_sel, hourly_index_sel)
            if final_index is not None and final_index != hourly_index_sel and not error_msg:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_CAT)
                st.warning(f"**Avís:** Dades no disponibles per a les {hora_sel_str}. Es mostren les de l'hora vàlida més propera: **{adjusted_local_time.strftime('%H:%Mh')}**.")
            if error_msg: 
                st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
            elif data_tuple:
                params_calc = data_tuple[1]
                map_data_conv, _ = carregar_dades_mapa_cat(nivell_sel, hourly_index_sel)
                if map_data_conv:
                    conv_value = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
                    if pd.notna(conv_value): params_calc[f'CONV_{nivell_sel}hPa'] = conv_value
                if st.session_state.active_tab_cat == "Anàlisi Vertical":
                    avis_proximitat = analitzar_amenaça_convergencia_propera(map_data_conv, params_calc, lat_sel, lon_sel, nivell_sel)
                    ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str, avis_proximitat)
                elif st.session_state.active_tab_cat == "Anàlisi de Vents":
                    ui_pestanya_analisis_vents(data_tuple, poble_sel, hora_sel_str, timestamp_str)
                elif st.session_state.active_tab_cat == "Simulació de Núvol":
                    st.markdown(f"#### Simulació del Cicle de Vida per a {poble_sel}")
                    st.caption(timestamp_str)
                    if 'regenerate_key' not in st.session_state: st.session_state.regenerate_key = 0
                    if st.button("🔄 Regenerar Totes les Animacions"): forcar_regeneracio_animacio()
                    with st.spinner("Generant simulacions visuals..."):
                        params_tuple = tuple(sorted(params_calc.items()))
                        gifs = generar_animacions_professionals(params_tuple, timestamp_str, st.session_state.regenerate_key)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("<h5 style='text-align: center;'>1. Iniciació</h5>", unsafe_allow_html=True)
                        if gifs['iniciacio']: st.image(gifs['iniciacio'])
                        else: st.info("Condicions estables.")
                    with col2:
                        st.markdown("<h5 style='text-align: center;'>2. Maduresa</h5>", unsafe_allow_html=True)
                        if gifs['maduresa']: st.image(gifs['maduresa'])
                        else: st.info("Sense energia per a tempesta.")
                    with col3:
                        st.markdown("<h5 style='text-align: center;'>3. Dissipació</h5>", unsafe_allow_html=True)
                        if gifs['dissipacio']: st.image(gifs['dissipacio'])
                        else: st.info("Sense fase final.")
                    st.divider()
                    ui_guia_tall_vertical(params_calc, nivell_sel)
                elif st.session_state.active_tab_cat == "💬 Assistent IA" and not is_guest:
                    analisi_temps = analitzar_potencial_meteorologic(params_calc, nivell_sel, hora_sel_str)
                    interpretacions_ia = interpretar_parametres(params_calc, nivell_sel)
                    sounding_data = data_tuple[0] if data_tuple else None
                    ui_pestanya_assistent_ia(params_calc, poble_sel, analisi_temps, interpretacions_ia, sounding_data)
    
    else: 
        # --- VISTA DE SELECCIÓ (MAPA INTERACTIU) ---
        selected_area = st.session_state.get('selected_area')
        if selected_area and "---" not in selected_area:
             st.info(f"Zona seleccionada: **{selected_area}**. Fes clic sobre el nom d'una localitat per analitzar-la.", icon="👇")
             if st.button("⬅️ Veure totes les zones"):
                 st.session_state.selected_area = "--- Selecciona una zona al mapa ---"
                 st.rerun()
        else:
             st.info("Fes clic en una zona del mapa per veure'n les localitats.", icon="👆")

        map_output = ui_mapa_display_personalitzat(list(alertes_zona.keys()))

        if map_output and map_output.get("last_object_clicked_tooltip"):
            raw_tooltip = map_output["last_object_clicked_tooltip"]
            
            if "Zona:" in raw_tooltip or "Comarca:" in raw_tooltip:
                clicked_area = raw_tooltip.split(':')[-1].strip()
                if clicked_area != st.session_state.get('selected_area'):
                    st.session_state.selected_area = clicked_area
                    st.session_state.poble_sel = "--- Selecciona una localitat ---"
                    st.rerun()
            else:
                clicked_poble = raw_tooltip.strip()
                if clicked_poble in CIUTATS_CATALUNYA:
                    st.session_state.poble_sel = clicked_poble
                    st.rerun()


def ui_mapa_display_personalitzat(zones_en_alerta):
    """
    Versió final i blindada.
    - Comprova activament si les propietats necessàries ('nomcomar' o 'nom_zona') existeixen a l'arxiu GeoJSON.
    - Si no existeixen, mostra un missatge d'error clar en lloc de trencar l'aplicació.
    - Manté tota la funcionalitat interactiva si l'arxiu és correcte.
    """
    st.markdown("#### Mapa de Situació")
    gdf = carregar_dades_geografiques()
    if gdf is None: return None

    # --- BLOC DE VALIDACIÓ DEFINITIU ---
    # Comprovem quina propietat de nom podem utilitzar
    if 'nom_zona' in gdf.columns:
        property_name = 'nom_zona'
        tooltip_alias = 'Zona:'
    elif 'nomcomar' in gdf.columns:
        property_name = 'nomcomar'
        tooltip_alias = 'Comarca:'
    else:
        # SI ARRIBEM AQUÍ, L'ARXIU GEOJSON ÉS INVÀLID PER A L'APP
        st.error(
            "**Error Crític en el Mapa:** L'arxiu GeoJSON que estàs utilitzant "
            "no conté una propietat anomenada 'nomcomar' o 'nom_zona'.\n\n"
            "**Solució:**\n"
            "1. Obre el teu arxiu GeoJSON a http://geojson.io.\n"
            "2. Clica sobre un dels polígons.\n"
            "3. Assegura't que a la taula de la dreta hi hagi una propietat amb el nom correcte (p. ex., 'nomcomar') i un valor (p. ex., 'Garraf')."
        )
        return None # Aturem l'execució per evitar l'error

    # Si el codi continua, sabem que 'property_name' és vàlid
    selected_area = st.session_state.get('selected_area')

    map_center = [41.83, 1.87]; zoom_level = 8
    if selected_area and "---" not in selected_area:
        zona_shape = gdf[gdf[property_name] == selected_area]
        if not zona_shape.empty:
            map_center = [zona_shape.geometry.centroid.y.iloc[0], zona_shape.geometry.centroid.x.iloc[0]]
            zoom_level = 10 if property_name == 'nomcomar' else 9

    m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB dark_matter", scrollWheelZoom=True)

    def style_function(feature):
        nom_feature = feature.get('properties', {}).get(property_name)
        style = {'fillColor': '#444444', 'color': '#666666', 'weight': 1, 'fillOpacity': 0.1}
        if nom_feature:
            style = {'fillColor': '#6c757d', 'color': '#adb5bd', 'weight': 1, 'fillOpacity': 0.25}
            if nom_feature in zones_en_alerta:
                style['fillColor'] = '#ffc107'; style['color'] = '#ffc107'; style['fillOpacity'] = 0.6; style['weight'] = 2
            if nom_feature == selected_area:
                style['fillColor'] = '#007bff'; style['color'] = '#ffffff'; style['weight'] = 2.5; style['fillOpacity'] = 0.55
        return style

    highlight_function = lambda x: {'color': '#ffffff', 'weight': 3, 'fillOpacity': 0.4}

    folium.GeoJson(
        gdf,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(fields=[property_name], aliases=[tooltip_alias])
    ).add_to(m)

    if selected_area and "---" not in selected_area:
        poblacions_dict = CIUTATS_PER_ZONA_PERSONALITZADA if property_name == 'nom_zona' else CIUTATS_PER_COMARCA
        poblacions_a_mostrar = poblacions_dict.get(selected_area, {})
        for nom_poble, coords in poblacions_a_mostrar.items():
            icon = folium.DivIcon(
                html=f"""<div style="font-family: sans-serif; font-size: 11px; font-weight: bold; color: white; background-color: rgba(0, 0, 0, 0.6); padding: 2px 6px; border-radius: 5px; border: 1px solid white; white-space: nowrap;">{nom_poble}</div>"""
            )
            folium.Marker(location=[coords['lat'], coords['lon']], icon=icon, tooltip=nom_poble).add_to(m)

    return st_folium(m, width="100%", height=450, returned_objects=['last_object_clicked_tooltip'])
    
                    
def run_valley_halley_app():
    if 'poble_selector_usa' not in st.session_state or st.session_state.poble_selector_usa not in USA_CITIES:
        st.session_state.poble_selector_usa = "Dallas, TX"
    ui_capcalera_selectors(None, zona_activa="valley_halley")
    poble_sel = st.session_state.poble_selector_usa
    now_local = datetime.now(TIMEZONE_USA)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = USA_CITIES[poble_sel]['lat'], USA_CITIES[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} (CST) / {cat_dt.strftime('%d/%m, %H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]
    
    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_usa", default_index=0)

    if st.session_state.active_tab_usa == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig HRRR per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_usa(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_USA)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_usa == "Anàlisi de Mapes":
        ui_pestanya_mapes_usa(hourly_index_sel, timestamp_str, nivell_sel)
    elif st.session_state.active_tab_usa == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="valley_halley")




def ui_zone_selection():
    st.markdown("<h1 style='text-align: center;'>Zona d'Anàlisi</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("🟢(tenen webcams)-🔥(Especialment recomanades) ", icon="🎞")

    def start_transition(zone_id):
        """Callback per iniciar la transició de vídeo."""
        st.session_state['zone_selected'] = zone_id
        st.session_state['show_transition_video'] = True

    paths = {
        'cat': "catalunya_preview.png", 'usa': "usa_preview.png", 'ale': "alemanya_preview.png",
        'ita': "italia_preview.png", 'hol': "holanda_preview.png", 'japo': "japo_preview.png",
        'uk': "uk_preview.png", 'can': "canada_preview.png",
        'nor': "noruega_preview.png",
        'arxiu': "arxiu_preview.png"
    }
    
    with st.spinner('Carregant entorns geoespacials...'): time.sleep(1)

    # Definim les tres files de columnes
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    row3_col1, _, _, _ = st.columns(4)


    def create_zone_button(col, path, title, key, zone_id, type="secondary"):
        with col, st.container(border=True):
            st.markdown(generar_html_imatge_estatica(path, height="160px"), unsafe_allow_html=True)
            
            display_title = title
            if zone_id == 'italia':
                display_title += " 🔥"
            elif zone_id in ['japo', 'uk', 'canada', 'valley_halley', 'alemanya', 'holanda', 'catalunya', 'noruega']:
                display_title += " 🟢"
            
            st.subheader(display_title)
            
            st.button(f"Analitzar {title}", key=key, use_container_width=True, type=type,
                      on_click=start_transition, args=(zone_id,))

    # Dibuixem els botons a les seves respectives files i columnes
    create_zone_button(row1_col1, paths['cat'], "Catalunya", "btn_cat", "catalunya", "primary")
    create_zone_button(row1_col2, paths['usa'], "Tornado Alley", "btn_usa", "valley_halley")
    create_zone_button(row1_col3, paths['ale'], "Alemanya", "btn_ale", "alemanya")
    create_zone_button(row1_col4, paths['ita'], "Itàlia", "btn_ita", "italia")
    
    create_zone_button(row2_col1, paths['hol'], "Holanda", "btn_hol", "holanda")
    create_zone_button(row2_col2, paths['japo'], "Japó", "btn_japo", "japo")
    create_zone_button(row2_col3, paths['uk'], "Regne Unit", "btn_uk", "uk")
    create_zone_button(row2_col4, paths['can'], "Canadà", "btn_can", "canada")

    create_zone_button(row3_col1, paths['nor'], "Noruega", "btn_nor", "noruega")

    # --- NOVA SECCIÓ D'ARXIUS (OCUPA TOT L'AMPLE) ---
    st.markdown("---")
    
    with st.container(border=True):
        img_col, content_col = st.columns([0.4, 0.6])

        with img_col:
            st.markdown(generar_html_imatge_estatica(paths['arxiu'], height="180px"), unsafe_allow_html=True)

        with content_col:
            st.subheader("Arxius Tempestes ⛈️")
            st.write(
                """
                Explora i analitza els **sondejos i mapes de situacions de temps sever passades**. 
                Una eina essencial per a l'estudi de casos, la comparació de patrons i l'aprenentatge.
                """
            )
            st.button("Consultar Arxius", key="btn_arxiu", use_container_width=True, type="primary",
                      on_click=start_transition, args=("arxiu_tempestes",))
            

@st.cache_data(ttl=3600)
def carregar_dades_mapa_italia(nivell, hourly_index):
    """
    Carrega les dades en una graella per al mapa d'Itàlia utilitzant el model ICON-2I.
    """
    try:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        
        # Creem una graella de punts per cobrir Itàlia
        lats, lons = np.linspace(MAP_EXTENT_ITALIA[2], MAP_EXTENT_ITALIA[3], 12), np.linspace(MAP_EXTENT_ITALIA[0], MAP_EXTENT_ITALIA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {
            "latitude": lat_grid.flatten().tolist(), 
            "longitude": lon_grid.flatten().tolist(), 
            "hourly": variables, 
            "models": "italia_meteo_arpae_icon_2i", 
            "forecast_days": 2
        }
        
        responses = openmeteo.weather_api(API_URL_ITALIA, params=params)
        output = {var: [] for var in ["lats", "lons"] + variables}
        
        # Processem la resposta de l'API
        for r in responses:
            try:
                vals = [r.Hourly().Variables(i).ValuesAsNumpy()[hourly_index] for i in range(len(variables))]
                if not any(np.isnan(v) for v in vals):
                    output["lats"].append(r.Latitude())
                    output["lons"].append(r.Longitude())
                    for i, var in enumerate(variables): 
                        output[var].append(vals[i])
            except IndexError:
                continue # Si l'hora no existeix per a aquest punt, el saltem

        if not output["lats"]: 
            return None, "No s'han rebut dades vàlides per a l'hora seleccionada."

        # Processem les dades per a la visualització
        temp_data = np.array(output.pop(f'temperature_{nivell}hPa')) * units.degC
        rh_data = np.array(output.pop(f'relative_humidity_{nivell}hPa')) * units.percent
        output['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
        output['speed_data'] = output.pop(f'wind_speed_{nivell}hPa')
        output['dir_data'] = output.pop(f'wind_direction_{nivell}hPa')
        
        return output, None

    except Exception as e: 
        return None, f"Error en carregar dades del mapa ICON-2I (Itàlia): {e}"
    


def crear_mapa_forecast_combinat_italia(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    """
    Crea el mapa visual de vent i convergència per a Itàlia.
    """
    fig, ax = crear_mapa_base(MAP_EXTENT_ITALIA, projection=ccrs.LambertConformal(central_longitude=12.5, central_latitude=42))
    
    if len(lons) < 4: 
        ax.set_title("Dades insuficients per generar el mapa")
        return fig

    # Interpolació de dades a una graella fina
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_ITALIA[0], MAP_EXTENT_ITALIA[1], 200), np.linspace(MAP_EXTENT_ITALIA[2], MAP_EXTENT_ITALIA[3], 200))
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'cubic')
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'cubic')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')

    # Dibuix de la velocitat del vent (fons de color)
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    
    # Dibuix de les línies de corrent
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.8, density=4.5, arrowsize=0.5, zorder=4, transform=ccrs.PlateCarree())
    
    # Càlcul i dibuix de la convergència
    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
    convergence_scaled = -(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s').magnitude * 1e5
    convergence_in_humid_areas = np.where(grid_dewpoint >= 14, convergence_scaled, 0) # Llindar de punt de rosada
    
    fill_levels = [5, 10, 15, 25]; fill_colors = ['#ffc107', '#ff9800', '#f44336']
    line_levels = [5, 10, 15]; line_colors = ['#e65100', '#bf360c', '#b71c1c']
    
    ax.contourf(grid_lon, grid_lat, convergence_in_humid_areas, levels=fill_levels, colors=fill_colors, alpha=0.6, zorder=5, transform=ccrs.PlateCarree())
    contours = ax.contour(grid_lon, grid_lat, convergence_in_humid_areas, levels=line_levels, colors=line_colors, linestyles='--', linewidths=1.2, zorder=6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')

    # Afegir ciutats per a referència
    for city, coords in CIUTATS_ITALIA.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='red', markersize=3, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.1, coords['lat'] + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(), zorder=10, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax.set_title(f"Vent i Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig



def run_italia_app():
    if 'poble_selector_italia' not in st.session_state: st.session_state.poble_selector_italia = "Roma"
    ui_capcalera_selectors(None, zona_activa="italia")
    poble_sel = st.session_state.poble_selector_italia
    now_local = datetime.now(TIMEZONE_ITALIA)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = CIUTATS_ITALIA[poble_sel]['lat'], CIUTATS_ITALIA[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} ({TIMEZONE_ITALIA.zone}) / {cat_dt.strftime('%H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]

    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_italia", default_index=0)

    if st.session_state.active_tab_italia == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_italia(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_ITALIA)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_italia(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_italia == "Anàlisi de Mapes":
        ui_pestanya_mapes_italia(hourly_index_sel, timestamp_str, nivell_sel)
    elif st.session_state.active_tab_italia == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="italia")


def run_alemanya_app():
    if 'poble_selector_alemanya' not in st.session_state: st.session_state.poble_selector_alemanya = "Berlín (Alexanderplatz)"
    ui_capcalera_selectors(None, zona_activa="alemanya")
    poble_sel = st.session_state.poble_selector_alemanya
    now_local = datetime.now(TIMEZONE_ALEMANYA)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = CIUTATS_ALEMANYA[poble_sel]['lat'], CIUTATS_ALEMANYA[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} ({TIMEZONE_ALEMANYA.zone}) / {cat_dt.strftime('%H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]

    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_alemanya", default_index=0)

    if st.session_state.active_tab_alemanya == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_alemanya(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_ALEMANYA)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_alemanya(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_alemanya == "Anàlisi de Mapes":
        ui_pestanya_mapes_alemanya(hourly_index_sel, timestamp_str, nivell_sel, poble_sel)
    elif st.session_state.active_tab_alemanya == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="alemanya")

# També necessitem una funció per mostrar el mapa d'Alemanya, que no existia. Afegeix-la al teu codi:
def ui_pestanya_mapes_alemanya(hourly_index_sel, timestamp_str, nivell_sel, poble_sel):
    st.markdown("#### Mapes de Pronòstic (Model ICON-D2)")
    with st.spinner("Carregant mapa ICON-D2... El primer cop pot trigar una mica."):
        map_data, error = carregar_dades_mapa_alemanya(nivell_sel, hourly_index_sel)
    
        if error or not map_data:
            # <<<--- CORRECCIÓ AQUÍ --->>>
            st.error(f"Error en carregar el mapa: {error if error else 'No s''han rebut dades.'}")
        else:
            timestamp_map_title = timestamp_str.replace(f"{poble_sel} | ", "")
            fig = crear_mapa_forecast_combinat_alemanya(
                map_data['lons'], map_data['lats'], map_data['speed_data'], 
                map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, 
                timestamp_map_title
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def run_uk_app():
    if 'poble_selector_uk' not in st.session_state: st.session_state.poble_selector_uk = "Southampton"
    ui_capcalera_selectors(None, zona_activa="uk")
    poble_sel = st.session_state.poble_selector_uk
    now_local = datetime.now(TIMEZONE_UK)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = CIUTATS_UK[poble_sel]['lat'], CIUTATS_UK[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} ({TIMEZONE_UK.zone}) / {cat_dt.strftime('%H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]
    
    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_uk", default_index=0)

    if st.session_state.active_tab_uk == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_uk(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_UK)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_uk(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_uk == "Anàlisi de Mapes":
        ui_pestanya_mapes_uk(hourly_index_sel, timestamp_str, nivell_sel, poble_sel)
    elif st.session_state.active_tab_uk == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="uk")

def run_holanda_app():
    if 'poble_selector_holanda' not in st.session_state: st.session_state.poble_selector_holanda = "Amsterdam"
    ui_capcalera_selectors(None, zona_activa="holanda")
    poble_sel = st.session_state.poble_selector_holanda
    now_local = datetime.now(TIMEZONE_HOLANDA)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = CIUTATS_HOLANDA[poble_sel]['lat'], CIUTATS_HOLANDA[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} ({TIMEZONE_HOLANDA.zone}) / {cat_dt.strftime('%H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]

    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_holanda", default_index=0)

    if st.session_state.active_tab_holanda == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_holanda(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_HOLANDA)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_holanda(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_holanda == "Anàlisi de Mapes":
        ui_pestanya_mapes_holanda(hourly_index_sel, timestamp_str, nivell_sel, poble_sel)
    elif st.session_state.active_tab_holanda == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="holanda")

def run_japo_app():
    if 'poble_selector_japo' not in st.session_state: st.session_state.poble_selector_japo = "Tòquio"
    ui_capcalera_selectors(None, zona_activa="japo")
    poble_sel = st.session_state.poble_selector_japo
    now_local = datetime.now(TIMEZONE_JAPO)
    dia_sel_str = now_local.strftime('%d/%m/%Y'); hora_sel = now_local.hour
    hora_sel_str = f"{hora_sel:02d}:00h"; nivell_sel = 925
    lat_sel, lon_sel = CIUTATS_JAPO[poble_sel]['lat'], CIUTATS_JAPO[poble_sel]['lon']
    local_dt = now_local.replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    cat_dt = local_dt.astimezone(TIMEZONE_CAT)
    timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} ({TIMEZONE_JAPO.zone}) / {cat_dt.strftime('%d/%m, %H:%Mh')} (CAT)"
    
    menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Webcams en Directe"]
    menu_icons = ["graph-up-arrow", "map-fill", "camera-video-fill"]

    option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                orientation="horizontal", key="active_tab_japo", default_index=0)

    if st.session_state.active_tab_japo == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_japo(lat_sel, lon_sel, hourly_index_sel)
        if data_tuple is None or error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
        else:
            if final_index is not None and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_JAPO)
                st.warning(f"**Avís:** Es mostren dades de les **{adjusted_local_time.strftime('%H:%Mh')}**.")
            params_calc = data_tuple[1]
            with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                map_data_conv, _ = carregar_dades_mapa_japo(nivell_sel, hourly_index_sel)
            if map_data_conv:
                params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str)
    elif st.session_state.active_tab_japo == "Anàlisi de Mapes":
        ui_pestanya_mapes_japo(hourly_index_sel, timestamp_str, nivell_sel, poble_sel)
    elif st.session_state.active_tab_japo == "Webcams en Directe":
        ui_pestanya_webcams(poble_sel, zona_activa="japo")

def main():
    inject_custom_css()
    hide_streamlit_style()

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        afegir_slideshow_de_fons()
        show_login_page()
        return

    if 'zone_selected' not in st.session_state or st.session_state.zone_selected is None:
        ui_zone_selection()
        return

    if st.session_state.zone_selected == 'catalunya': run_catalunya_app()
    elif st.session_state.zone_selected == 'valley_halley': run_valley_halley_app()
    elif st.session_state.zone_selected == 'alemanya': run_alemanya_app()
    elif st.session_state.zone_selected == 'italia': run_italia_app()
    elif st.session_state.zone_selected == 'holanda': run_holanda_app()
    elif st.session_state.zone_selected == 'japo': run_japo_app()
    elif st.session_state.zone_selected == 'uk': run_uk_app()
    elif st.session_state.zone_selected == 'canada': run_canada_app()
    elif st.session_state.zone_selected == 'noruega': run_noruega_app()
    elif st.session_state.zone_selected == 'arxiu_tempestes':
        run_arxiu_tempestes_app()



def run_arxiu_tempestes_app():
    """
    Funció principal per a la secció d'Arxius de Tempestes.
    Mostra una llista de casos d'estudi i permet visualitzar-ne els detalls.
    """
    
    # --- 1. BASE DE DADES DELS CASOS D'ESTUDI ---
    casos_notables = {
        "--- Selecciona un cas d'estudi ---": None,
        "Tempesta litoral central (SPC) (01/09/2025)": {
            "data": "1 de Set. del 2025",
            "image": "arxiu_spbcn.jpg",
            "description": """
            L'entorn atmosfèric (El Sondeig):
Comencem pel sondeig vertical de L'Hospitalet, que és el diagnòstic de l'atmosfera. El que veiem aquí és un manual de "llibre de text" per a la formació de supercèl·lules.
Inestabilitat Extrema: La taca rosa, que representa l'Energia Potencial Convectiva Disponible (CAPE), és molt àmplia i robusta. Això indica que qualsevol bombolla d'aire que aconsegueixi ascendir ho farà de manera explosiva, com un globus aerostàtic descontrolat, creant corrents ascendents molt violents.
Humitat Abundant: La línia del punt de rosada (verda) està molt a prop de la línia de temperatura (vermella) a les capes baixes. Això significa que l'aire és molt humit, la qual cosa fa baixar el nivell de condensació. Aquesta és una característica clau que afavoreix la formació de núvols de paret i tornados.
Cisallament del Vent Decisiu: Observant els vectors de vent a la dreta, veiem un canvi significatiu tant en direcció com en velocitat amb l'altura. Aquest cisallament vertical del vent és l'ingredient crucial que actua com un motor de rotació horitzontal a l'atmosfera. Els potents corrents ascendents de la tempesta inclinen aquesta rotació i la posen en un eix vertical, donant lloc a un mesocicló, l'embrió d'una supercèl·lula.
La Tempesta en Acció (El Radar):
El mapa del radar no mostra una tempesta desorganitzada, sinó un sistema convectiu altament estructurat. La distribució dels ecos, amb nuclis de reflectivitat molt alta (vermells i morats, probablement superiors a 55 dBZ), suggereix la presència de precipitació molt intensa i, amb tota seguretat, calamarsa de mida considerable. La forma i l'extensió de la tempesta són compatibles amb una o diverses cèl·lules de tipus supercel·lular incrustades dins d'un sistema més gran. Aquesta estructura és la conseqüència directa de l'entorn que hem analitzat al sondeig.
La Manifestació Visual (La Fotografia):
La imatge superior és la confirmació visual del que les dades ens estaven dient. No és només una tempesta elèctrica. La forma de la base del núvol, amb un descens molt pronunciat i una aparença de rotació (un possible wall cloud o núvol paret), és l'evidència visible del mesocicló. El llamp és un subproducte de la increïble energia vertical de la tempesta. La combinació del núvol paret i la intensa activitat elèctrica és una signatura visual clàssica d'una supercèl·lula en plena maduresa i amb un alt potencial de generar fenòmens severs a la superfície.
            """
        },
    }

    # --- 2. INTERFÍCIE D'USUARI ---
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">Arxius de Situacions de Temps Sever</h1>', unsafe_allow_html=True)
    
    if st.button("⬅️ Tornar a la Selecció de Zona"):
        st.session_state.zone_selected = None
        st.rerun()
    
    st.divider()

    selected_case = st.selectbox(
        "Selecciona un cas d'estudi per analitzar:", 
        options=list(casos_notables.keys())
    )

    # --- 3. LÒGICA PER MOSTRAR LA INFORMACIÓ ---
    if selected_case and casos_notables[selected_case]:
        case_data = casos_notables[selected_case]
        st.markdown("---")
        col_img, col_desc = st.columns([0.5, 0.5], gap="large")
        with col_img:
            try:
                st.image(case_data['image'], caption=f"Imatge de: {selected_case}", use_container_width=True)
            except FileNotFoundError:
                st.error(f"Error: No s'ha trobat la imatge '{case_data['image']}'.")
            except Exception as e:
                st.error(f"S'ha produït un error en carregar la imatge: {e}")
        with col_desc:
            st.subheader(selected_case)
            st.caption(f"**Data de l'esdeveniment:** {case_data['data']}")
            st.markdown(case_data['description'])
    else:
        st.info("Selecciona un esdeveniment de la llista superior per veure'n els detalls.", icon="👆")



def analitzar_potencial_meteorologic(params, nivell_conv, hora_actual=None):
    """
    Sistema de Diagnòstic v29.0 - Afegeix detecció de Nimbostratus.
    1. Comprova si hi ha saturació per a pluja estratiforme (Nimbostratus).
    2. Si no, comprova inhibició o falta de disparador per a convecció.
    3. Si no, comprova si l'LFC és massa alt.
    4. Si tot es supera, classifica la tempesta convectiva.
    """
    # --- 1. EXTRACCIÓ COMPLETA I ROBUSTA DE PARÀMETRES ---
    sbcape = params.get('SBCAPE', 0) or 0; mlcape = params.get('MLCAPE', 0) or 0; mucape = params.get('MUCAPE', 0) or 0
    mucin = params.get('MUCIN', 0) or 0
    lfc_hgt = params.get('LFC_Hgt', 9999) or 9999
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_3km = params.get('SRH_0-3km', 0) or 0
    pwat = params.get('PWAT', 0) or 0 # PWAT ara s'extreu aquí
    
    rh_capes = params.get('RH_CAPES', {'baixa': 0, 'mitjana': 0, 'alta': 0})
    rh_baixa = rh_capes.get('baixa', 0) if pd.notna(rh_capes.get('baixa')) else 0
    rh_mitjana = rh_capes.get('mitjana', 0) if pd.notna(rh_capes.get('mitjana')) else 0
    rh_alta = rh_capes.get('alta', 0) if pd.notna(rh_capes.get('alta')) else 0

    conv_key = f'CONV_{nivell_conv}hPa'
    conv = params.get(conv_key, 0) or 0

    # --- NOU BLOC: FILTRE PER A NIMBOSTRATUS (PRIORITARI) ---
    # Comprovem si hi ha una capa molt humida a nivells baixos i mitjans,
    # condició típica per a la pluja estratiforme contínua.
    if rh_baixa > 85 and rh_mitjana > 80:
        # Si hi ha saturació, determinem la intensitat de la pluja amb el PWAT
        if pwat >= 35: precip_desc = "Pluges fortes / Xàfecs"
        elif pwat >= 20: precip_desc = "Pluges"
        else: precip_desc = "Plujims"
        
        return {
            'emoji': "🌧️",
            'descripcio': f"Nimbostratus ({precip_desc})",
            'veredicte': "Cel cobert amb precipitació estratiforme generalitzada. Poca o nul·la activitat elèctrica.",
            'factor_clau': "Saturació a capes baixes i mitjanes."
        }
    # --- FI DEL NOU BLOC ---

    # --- 2. FILTRE PRINCIPAL: INHIBICIÓ O SENSE DISPARADOR (ara és el segon pas) ---
    if mucin < -100 or conv < 25:
        if rh_baixa > 80: return {'emoji': "☁️", 'descripcio': "Cel Cobert (Estratus)", 'veredicte': "Capa de núvols baixos sense desenvolupament vertical.", 'factor_clau': "Inhibició o falta de disparador."}
        if rh_mitjana > 70: return {'emoji': "🌥️", 'descripcio': "Núvols Mitjans (Altocúmulus)", 'veredicte': "Cel variable amb núvols a nivells mitjans.", 'factor_clau': "Inhibició o falta de disparador."}
        if rh_alta > 60: return {'emoji': "🌤️", 'descripcio': "Núvols Alts (Cirrus)", 'veredicte': "Cel poc ennuvolat amb núvols alts.", 'factor_clau': "Inhibició o falta de disparador."}
        return {'emoji': "☀️", 'descripcio': "Cel Serè", 'veredicte': "Temps estable. Les condicions no són favorables per a la convecció.", 'factor_clau': "Inhibició o falta de disparador."}

    # --- 3. FILTRE CLAU: LFC > 3000m (CONVECCIÓ DE BASE ALTA) ---
    if lfc_hgt > 3000:
        desc = "Altocúmulus Castellanus"
        veredicte = "Potencial per a convecció de base elevada. No s'esperen tempestes organitzades a la superfície."
        if rh_baixa > 70: desc = "Stratocumulus Castellanus"
        return {'emoji': "🌥️", 'descripcio': desc, 'veredicte': veredicte, 'factor_clau': f"LFC elevat ({lfc_hgt:.0f} m)."}

    # --- 4. SI ARRIBEM AQUÍ (LFC < 3000m), CLASSIFIQUEM LA TEMPESTA ---
    if mucape > 1500 and bwd_6km > 35 and srh_3km > 200:
        return {'emoji': "🌪️", 'descripcio': "Potencial de Supercèl·lula", 'veredicte': "Condicions molt favorables per a la formació de tempestes rotatòries i severes.", 'factor_clau': "Alt CAPE, fort cisallament i helicitat."}
    
    elif mucape > 1000 and bwd_6km > 25:
        return {'emoji': "⛈️", 'descripcio': "Tempestes Organitzades (Multicèl·lula)", 'veredicte': "Potencial per a la formació de grups de tempestes o línies organitzades.", 'factor_clau': "CAPE moderat-alt i cisallament suficient."}
        
    elif mucape > 800:
        return {'emoji': "🌩️", 'descripcio': "Tempesta Aïllada (Cumulonimbus)", 'veredicte': "Condicions favorables per al desenvolupament de tempestes aïllades, possiblement fortes.", 'factor_clau': "CAPE suficient per a un desenvolupament vertical complet."}

    elif mucape > 400:
        return {'emoji': "☁️", 'descripcio': "Desenvolupament Vertical (Congestus)", 'veredicte': "Potencial per a núvols de gran creixement que podrien deixar xàfecs aïllats.", 'factor_clau': "CAPE moderat, suficient per a congestus."}

    else: # MUCAPE < 400
        return {'emoji': "🌤️", 'descripcio': "Núvols de Bon Temps (Cúmuls)", 'veredicte': "Es formaran petits cúmuls de bon temps amb poc o cap desenvolupament vertical.", 'factor_clau': "CAPE baix."}



    
if __name__ == "__main__":
    main()
