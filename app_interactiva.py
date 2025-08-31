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




# --- 0. CONFIGURACIÓ I CONSTANTS ---
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever")

# --- Clients API ---
parcel_lock = threading.Lock()
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

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
    'Elk City, OK': {'lat': 35.4098, 'lon': -99.4084},    # <-- NOU PUNT AFEGIT
    'Houston, TX': {'lat': 29.7604, 'lon': -95.3698},     # <-- NOU PUNT AFEGIT
    'Kansas City, MO': {'lat': 39.0997, 'lon': -94.5786},
    'Oklahoma City, OK': {'lat': 35.4676, 'lon': -97.5164},
    'Omaha, NE': {'lat': 41.2565, 'lon': -95.9345},
    'Tulsa, OK': {'lat': 36.1540, 'lon': -95.9928},
    'Wichita, KS': {'lat': 37.6872, 'lon': -97.3301},
}
MAP_EXTENT_USA = [-105, -85, 28, 48]
PRESS_LEVELS_GFS = sorted([1000, 975, 950, 925, 900, 850, 800, 750, 700, 600, 500, 400, 300, 250, 200, 100], reverse=True)

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




def generar_html_video_animacio(video_path, height="250px"):
    """
    Crea el codi HTML per mostrar un vídeo en bucle com una animació
    dins d'un contenidor amb cantonades arrodonides.
    """
    if not os.path.exists(video_path):
        return f"<p style='color: red;'>Error: No es troba el vídeo a '{video_path}'</p>"

    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode()
    
    # Estils per al contenidor i el vídeo
    container_style = f"width: 100%; height: {height}; border-radius: 10px; overflow: hidden; margin-bottom: 10px;"
    video_style = "width: 100%; height: 100%; object-fit: cover;"

    html_code = f"""
    <div style="{container_style}">
        <video autoplay loop muted playsinline style="{video_style}">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            El teu navegador no suporta vídeos HTML5.
        </video>
    </div>
    """
    return html_code



def afegir_video_de_fons():
    """
    Llegeix un arxiu de vídeo local, el codifica en Base64 i l'injecta
    com un fons de pantalla complet per a la pàgina de login.
    """
    # Assegura't que el vídeo 'llamps.mp4' estigui a la mateixa carpeta que l'script
    video_file = 'llamps2.mp4'

    if not os.path.exists(video_file):
        # Si el vídeo no existeix, no facis res per evitar un error
        return

    with open(video_file, "rb") as video:
        video_bytes = video.read()
    
    video_b64 = base64.b64encode(video_bytes).decode()
    
    video_html = f"""
    <style>
    .stApp {{
        background-color: transparent;
    }}
    #login-bg {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        z-index: -1;
        opacity: 0.8; /* Pots ajustar l'opacitat del vídeo aquí */
    }}
    </style>
    <video autoplay loop muted id="login-bg">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)
    
    
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

    if 'view' not in st.session_state:
        st.session_state.view = 'login'

    if st.session_state.view == 'login':
        st.subheader("Inicia Sessió")
        with st.form("login_form"):
            username = st.text_input("Nom d'usuari", key="login_user")
            password = st.text_input("Contrasenya", type="password", key="login_pass")
            
            if st.form_submit_button("Entra", use_container_width=True, type="primary"):
                users = load_json_file(USERS_FILE)
                if username in users and users[username] == get_hashed_password(password):
                    # Assegurem que la selecció de zona estigui neta abans de continuar.
                    st.session_state['zone_selected'] = None
                    st.session_state.update({'logged_in': True, 'username': username, 'guest_mode': False})
                    st.rerun()
                else:
                    st.error("Nom d'usuari o contrasenya incorrectes.")
        
        if st.button("No tens un compte? Registra't aquí"):
            st.session_state.view = 'register'
            st.rerun()

    elif st.session_state.view == 'register':
        st.subheader("Crea un nou compte")
        with st.form("register_form"):
            new_username = st.text_input("Tria un nom d'usuari", key="reg_user")
            new_password = st.text_input("Tria una contrasenya", type="password", key="reg_pass")
            
            if st.form_submit_button("Registra'm", use_container_width=True):
                users = load_json_file(USERS_FILE)
                if not new_username or not new_password:
                    st.error("El nom d'usuari i la contrasenya no poden estar buits.")
                elif new_username in users:
                    st.error("Aquest nom d'usuari ja existeix.")
                elif len(new_password) < 6:
                    st.error("La contrasenya ha de tenir com a mínim 6 caràcters.")
                else:
                    users[new_username] = get_hashed_password(new_password)
                    save_json_file(users, USERS_FILE)
                    st.success("Compte creat amb èxit! Ara pots iniciar sessió.")
        
        if st.button("Ja tens un compte? Inicia sessió"):
            st.session_state.view = 'login'
            st.rerun()
    
    st.divider()
    st.markdown("<p style='text-align: center;'>O si ho prefereixes...</p>", unsafe_allow_html=True)

    # Botón para entrar como convidat
    if st.button("Entrar com a Convidat (simple i ràpid)", use_container_width=True, type="secondary"):
        st.session_state['zone_selected'] = None
        st.session_state.update({'guest_mode': True, 'logged_in': True})
        st.rerun()
    
    # Separador para el modo desarrollador
    st.divider()
    st.markdown("<p style='text-align: center;'>Accés per a desenvolupadors</p>", unsafe_allow_html=True)
    
    # Formulario para modo desarrollador
    with st.form("developer_form"):
        dev_password = st.text_input("Contrasenya de desenvolupador", type="password", key="dev_pass")
        
        if st.form_submit_button("🚀 Accés Mode Desenvolupador", use_container_width=True):
            if dev_password == st.secrets["app_secrets"]["moderator_password"]:
                st.session_state['zone_selected'] = None
                st.session_state.update({
                    'logged_in': True, 
                    'developer_mode': True,
                    'username': 'Desenvolupador',
                    'guest_mode': False
                })
                st.success("Mode desenvolupador activat! Preguntes il·limitades a la IA.")
                time.sleep(1)  # Pequeña pausa para mostrar el mensaje
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
    Versió Definitiva i Antifràgil v5.3.
    Soluciona l'error del moviment de la tempesta capturant el vent mitjà
    directament del càlcul de Bunkers per garantir la coherència.
    """
    # --- 1. PREPARACIÓ I NETEJA DE DADES ---
    if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
    p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC
    Td = np.array(Td_profile) * units.degC; u = np.array(u_profile) * units('m/s')
    v = np.array(v_profile) * units('m/s'); heights = np.array(h_profile) * units.meter
    valid_indices = ~np.isnan(p.m) & ~np.isnan(T.m) & ~np.isnan(Td.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
    p, T, Td, u, v, heights = p[valid_indices], T[valid_indices], Td[valid_indices], u[valid_indices], v[valid_indices], heights[valid_indices]
    if len(p) < 3: return None, "No hi ha prou dades vàlides."
    sort_idx = np.argsort(p.m)[::-1]
    p, T, Td, u, v, heights = p[sort_idx], T[sort_idx], Td[sort_idx], u[sort_idx], v[sort_idx], heights[sort_idx]
    params_calc = {}; heights_agl = heights - heights[0]

    # --- 2. CÀLCULS BASE I PERFILS DE PARCEL·LA ---
    with parcel_lock:
        sfc_prof, ml_prof = None, None
        try: sfc_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        except Exception: return None, "Error crític: No s'ha pogut calcular ni el perfil de superfície."
        try: _, _, _, ml_prof = mpcalc.mixed_parcel(p, T, Td, depth=100 * units.hPa)
        except Exception: ml_prof = None
        main_prof = ml_prof if ml_prof is not None else sfc_prof

        # --- 3. CÀLCULS ROBUSTS I AÏLLATS ---
        # (Aquí va tota la secció de càlculs de PWAT, LI, T_500hPa, CAPE, etc. que ja tenies)
        try: 
            rh = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100
            params_calc['RH_CAPES'] = {'baixa': np.mean(rh[(p.m <= 1000) & (p.m > 850)]), 'mitjana': np.mean(rh[(p.m <= 850) & (p.m > 500)]), 'alta': np.mean(rh[(p.m <= 500) & (p.m > 250)])}
        except: params_calc['RH_CAPES'] = {'baixa': np.nan, 'mitjana': np.nan, 'alta': np.nan}
        try: params_calc['PWAT'] = float(mpcalc.precipitable_water(p, Td).to('mm').m)
        except: params_calc['PWAT'] = np.nan
        try:
            _, fl_h = mpcalc.freezing_level(p, T, heights); params_calc['FREEZING_LVL_HGT'] = float(fl_h[0].to('m').m)
        except: params_calc['FREEZING_LVL_HGT'] = np.nan
        try:
            p_numeric = p.m; T_numeric = T.m
            if len(p_numeric) >= 2 and p_numeric.min() <= 500 <= p_numeric.max():
                params_calc['T_500hPa'] = float(np.interp(500, p_numeric[::-1], T_numeric[::-1]))
            else: params_calc['T_500hPa'] = np.nan
        except: params_calc['T_500hPa'] = np.nan
        if sfc_prof is not None:
            try:
                sbcape, sbcin = mpcalc.cape_cin(p, T, Td, sfc_prof)
                params_calc['SBCAPE'] = float(sbcape.m); params_calc['SBCIN'] = float(sbcin.m)
                params_calc['MAX_UPDRAFT'] = np.sqrt(2 * float(sbcape.m)) if sbcape.m > 0 else 0.0
            except: params_calc.update({'SBCAPE': np.nan, 'SBCIN': np.nan, 'MAX_UPDRAFT': np.nan})
        if ml_prof is not None:
            try:
                mlcape, mlcin = mpcalc.cape_cin(p, T, Td, ml_prof); params_calc['MLCAPE'] = float(mlcape.m); params_calc['MLCIN'] = float(mlcin.m)
            except: params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
        else: params_calc.update({'MLCAPE': np.nan, 'MLCIN': np.nan})
        if main_prof is not None:
            try: params_calc['LI'] = float(mpcalc.lifted_index(p, T, main_prof).m)
            except: params_calc['LI'] = np.nan
            try:
                lfc_p, _ = mpcalc.lfc(p, T, Td, main_prof); params_calc['LFC_p'] = float(lfc_p.m)
                params_calc['LFC_Hgt'] = float(np.interp(lfc_p.m, p.m[::-1], heights_agl.m[::-1]))
            except: params_calc.update({'LFC_p': np.nan, 'LFC_Hgt': np.nan})
            try:
                el_p, _ = mpcalc.el(p, T, Td, main_prof); params_calc['EL_p'] = float(el_p.m)
                params_calc['EL_Hgt'] = float(np.interp(el_p.m, p.m[::-1], heights_agl.m[::-1]))
            except: params_calc.update({'EL_p': np.nan, 'EL_Hgt': np.nan})
            try:
                idx_3km = np.argmin(np.abs(heights_agl.m - 3000))
                cape_0_3, _ = mpcalc.cape_cin(p[:idx_3km+1], T[:idx_3km+1], Td[:idx_3km+1], main_prof[:idx_3km+1]); params_calc['CAPE_0-3km'] = float(cape_0_3.m)
            except: params_calc['CAPE_0-3km'] = np.nan
        try:
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td); params_calc['MUCAPE'] = float(mucape.m); params_calc['MUCIN'] = float(mucin.m)
        except: params_calc.update({'MUCAPE': np.nan, 'MUCIN': np.nan})
        try:
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); params_calc['LCL_p'] = float(lcl_p.m); params_calc['LCL_Hgt'] = float(np.interp(lcl_p.m, p.m[::-1], heights_agl.m[::-1]))
        except: params_calc.update({'LCL_p': np.nan, 'LCL_Hgt': np.nan})
        try:
            for name, depth_m in [('0-1km', 1000), ('0-6km', 6000)]:
                bwd_u, bwd_v = mpcalc.bulk_shear(p, u, v, height=heights, depth=depth_m * units.meter); params_calc[f'BWD_{name}'] = float(mpcalc.wind_speed(bwd_u, bwd_v).to('kt').m)
        except: params_calc.update({'BWD_0-1km': np.nan, 'BWD_0-6km': np.nan})
        
        # *** LÒGICA DE MOVIMENT DE TEMPESTA CORREGIDA ***
        try:
            # Aquesta funció retorna el moviment dret, esquerre I el vent mitjà.
            # Ara capturem els tres resultats correctament.
            rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p, u, v, heights)
            params_calc['RM'] = (float(rm[0].m), float(rm[1].m))
            params_calc['LM'] = (float(lm[0].m), float(lm[1].m))
            params_calc['Mean_Wind'] = (float(mean_wind[0].m), float(mean_wind[1].m))
        except Exception:
            # Si el càlcul falla, assegurem que tots tres siguin N/A.
            params_calc.update({
                'RM': (np.nan, np.nan),
                'LM': (np.nan, np.nan),
                'Mean_Wind': (np.nan, np.nan)
            })

        if params_calc.get('RM') and not np.isnan(params_calc['RM'][0]):
            u_storm, v_storm = params_calc['RM'][0] * units('m/s'), params_calc['RM'][1] * units('m/s')
            try:
                for name, depth_m in [('0-1km', 1000), ('0-3km', 3000)]:
                    srh = mpcalc.storm_relative_helicity(heights, u, v, depth=depth_m * units.meter, storm_u=u_storm, storm_v=v_storm)[0]; params_calc[f'SRH_{name}'] = float(srh.m)
            except: params_calc.update({'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})
        
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
    Versió professional amb zoom real i proporcional. Redibuixa el gràfic
    ajustant ambdós eixos per mantenir l'aspecte correcte del Skew-T.
    """
    fig = plt.figure(dpi=150, figsize=(7, 8))
    
    # Mantenim una única configuració per a la creació del SkewT
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.05, 0.85, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)

    # --- LÒGICA DE ZOOM PROFESSIONAL I PROPORCIONAL ---
    if zoom_capa_baixa:
        # 1. Definim els límits de pressió per al zoom (eix Y)
        pressio_superficie = p[0].m
        skew.ax.set_ylim(pressio_superficie + 5, 800) # Marge petit a la superfície
        
        # 2. Calculem els límits de temperatura NOMÉS per a aquesta capa (eix X)
        # Això és el pas clau per mantenir les proporcions!
        mask_capa_baixa = (p.m <= pressio_superficie) & (p.m >= 800)
        T_capa_baixa = T[mask_capa_baixa]
        Td_capa_baixa = Td[mask_capa_baixa]
        
        # Trobem les temperatures mínima i màxima en aquesta capa i afegim un marge
        temp_min = min(T_capa_baixa.min().m, Td_capa_baixa.min().m) - 5
        temp_max = max(T_capa_baixa.max().m, Td_capa_baixa.max().m) + 5
        skew.ax.set_xlim(temp_min, temp_max)
    else:
        # Comportament normal per al gràfic complet
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 40)
        
        # Dibuixem el terreny només a la vista completa
        pressio_superficie = p[0].m
        if pressio_superficie < 995:
            colors = ["#66462F", "#799845"] 
            cmap_terreny = LinearSegmentedColormap.from_list("terreny_cmap", colors)
            gradient = np.linspace(0, 1, 256).reshape(-1, 1)
            xlims = skew.ax.get_xlim()
            skew.ax.imshow(gradient.T, aspect='auto', cmap=cmap_terreny, origin='lower', extent=(xlims[0], xlims[1], 1000, pressio_superficie), alpha=0.6, zorder=0)
    # --- FI DE LA LÒGICA DE ZOOM ---

    skew.ax.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    skew.plot_dry_adiabats(color='coral', linestyle='--', alpha=0.5)
    skew.plot_moist_adiabats(color='cornflowerblue', linestyle='--', alpha=0.5)
    skew.plot_mixing_lines(color='limegreen', linestyle='--', alpha=0.5)
    
    if prof is not None:
        skew.shade_cape(p, T, prof, color='red', alpha=0.2)
        skew.shade_cin(p, T, prof, color='blue', alpha=0.2)
        skew.plot(p, prof, 'k', linewidth=3, label='Trajectòria Parcel·la (SFC)', path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])

    skew.plot(p, T, 'red', lw=2.5, label='Temperatura')
    skew.plot(p, Td, 'green', lw=2.5, label='Punt de Rosada')
        
    skew.plot_barbs(p, u.to('kt'), v.to('kt'), y_clip_radius=0.03)
    
    skew.ax.set_title(f"{titol}\n{timestamp_str}", weight='bold', fontsize=14, pad=15)
    skew.ax.set_xlabel("Temperatura (°C)"); skew.ax.set_ylabel("Pressió (hPa)")

    levels_to_plot = ['LCL_p', 'LFC_p', 'EL_p']
    for key in levels_to_plot:
        p_lvl = params_calc.get(key)
        if p_lvl is not None and not np.isnan(p_lvl):
            p_val = p_lvl.m if hasattr(p_lvl, 'm') else p_lvl
            skew.ax.axhline(p_val, color='blue', linestyle='--', linewidth=1.5)

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
        # ... (aquesta funció interna no canvia)
        color = "#FFFFFF"
        is_numeric = isinstance(value, (int, float, np.number))
        if pd.notna(value) and is_numeric:
            if 'CONV' in param_key:
                thresholds = [5, 15, 30, 40]
                colors = ["#808080", "#2ca02c", "#ffc107", "#fd7e14", "#dc3545"]
                color = colors[np.searchsorted(thresholds, value)]
            elif param_key == 'T_500hPa':
                thresholds = [-8, -14, -18, -22]
                colors = ["#2ca02c", "#ffc107", "#fd7e14", "#dc3545", "#b300ff"]
                color = colors[len(thresholds) - np.searchsorted(thresholds, value, side='right')]
            else:
                color = get_color_global(value, param_key, reverse_colors)
        val_str = f"{value:.{precision}f}" if pd.notna(value) and is_numeric else "---"
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;">
            <span style="font-size: 0.8em; color: #FAFAFA;">{label} ({unit}){tooltip_html}</span><br>
            <strong style="font-size: 1.6em; color: {color};">{val_str}</strong>
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
    en un radi proper a cada ciutat de la llista. Això assegura que les alertes
    de la llista coincideixin amb els nuclis visualitzats al mapa.
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
        SEARCH_RADIUS_DEG = 0.15  # Radi de cerca en graus (aprox. 15-20 km)

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

# Nova funció contenidora amb memòria cau.

def generar_animacio_cachejada(_params_tuple, hora_inici_str, _regenerate_key):
    """
    Crida a la funció de creació de l'animació. El '_regenerate_key'
    força la re-execució quan l'usuari clica el botó de regenerar.
    """
    params = dict(_params_tuple)
    return crear_animacio_tall_vertical(params, hora_inici_str)



def _dibuixar_frame_tall_vertical(frame_params):
    """
    Placeholder: retorna una imatge simulada per a cada frame.
    En el teu codi real hauries de substituir per la funció que dibuixa el plot.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.imshow(frame_params['reflectivity_grid'], origin='lower', extent=[frame_params['dist_range'][0],
                frame_params['dist_range'][-1], frame_params['alt_range'][0], frame_params['alt_range'][-1]],
                cmap='jet', vmin=0, vmax=75, aspect='auto')
    ax.set_title(frame_params['timestamp'])
    plt.close(fig)
    # Guardem la figura en memòria com a imatge PIL
    from PIL import Image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(image)

def crear_animacio_tall_vertical(params, hora_inici_str):
    """
    Animació vertical del nucli de pluja, amb nucli compacte i anvil.
    """
    # --- 1. Extracció de paràmetres ---
    lcl_hgt_m = params.get('LCL_Hgt', 1000) or 1000
    el_hgt_m_max = params.get('EL_Hgt', 10000) or 10000
    cape = params.get('MUCAPE', 0) or 0
    updraft = params.get('MAX_UPDRAFT', 0) or 0
    shear = params.get('BWD_0-6km', 0) or 0

    if cape < 100 or el_hgt_m_max <= lcl_hgt_m:
        return None

    # --- 2. Paràmetres de l'animació ---
    num_frames = 24
    total_duration_min = 120
    growth_factor = np.sin(np.linspace(0, np.pi, num_frames))
    frames = []
    hora_inici = datetime.strptime(hora_inici_str.replace('h',''), '%H:%M')

    for i in range(num_frames):
        current_growth = growth_factor[i]
        current_el = lcl_hgt_m + (el_hgt_m_max - lcl_hgt_m) * current_growth
        current_max_dbz = (20 + (np.sqrt(cape) / 10) + (updraft * 0.5)) * current_growth
        current_max_dbz = min(current_max_dbz, 75)

        # Nucli més compacte
        current_width_x = (3 + cape / 250) * current_growth
        dist_range = np.linspace(-20, 20, 100)
        alt_range = np.linspace(0, el_hgt_m_max + 2000, 100)
        xx, zz = np.meshgrid(dist_range, alt_range)

        if current_el <= lcl_hgt_m: 
            continue

        # --- Càlcul de densitats ---
        tilt_km_per_km_alt = shear * 0.1
        start_x_km = -10
        z_relative = np.maximum(0, zz - lcl_hgt_m)
        x_center = start_x_km + (tilt_km_per_km_alt * (z_relative / 1000))
        core_alt_center = lcl_hgt_m + (current_el - lcl_hgt_m) * 0.3

        dist_x_sq = ((xx - x_center) / (max(1, current_width_x)))**2
        dist_z_sq = ((zz - core_alt_center) / ((current_el - lcl_hgt_m) / 4))**2

        density = np.exp(-0.5 * (dist_x_sq + dist_z_sq))

        anvil_factor = 1 + 1.5 * np.clip((zz - (el_hgt_m_max * 0.7)) / (el_hgt_m_max * 0.3), 0, 1)
        density_anvil = np.exp(-0.5 * (((xx - x_center) / (np.maximum(1, current_width_x * anvil_factor)))**2 + dist_z_sq))

        final_density = np.maximum(density, density_anvil * 0.5)

        reflectivity_grid = final_density * current_max_dbz
        reflectivity_grid[zz < lcl_hgt_m] = 0

        simulated_time = hora_inici + timedelta(minutes=i * (total_duration_min / num_frames))
        timestamp_str_frame = f"Temps Simulat: {simulated_time.strftime('%H:%Mh')} (T+{int(i * (total_duration_min / num_frames))} min)"

        frame_params = {
            'dist_range': dist_range, 'alt_range': alt_range,
            'reflectivity_grid': reflectivity_grid, 'el_hgt_m': el_hgt_m_max,
            'timestamp': timestamp_str_frame
        }
        frames.append(_dibuixar_frame_tall_vertical(frame_params))

    # --- Creació del GIF ---
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, frames, format='gif', fps=4, loop=0)
    gif_buf.seek(0)

    return gif_buf.getvalue()


def ui_guia_tall_vertical(params, nivell_conv):
    """
    Crea una guia d'usuari contextual per interpretar el tall vertical,
    ara encapçalada per un veredicte de "Val la pena anar-hi?".
    """
    st.markdown("#### 🔍 Com Interpretar la Simulació")
    
    # --- NOU: Cridem la funció d'anàlisi de "caça" ---
    veredicte_caca = analitzar_potencial_caca(params, nivell_conv)

    # Extreiem els paràmetres per a les altres targetes
    el_hgt_km = (params.get('EL_Hgt', 0) or 0) / 1000
    cape = params.get('MUCAPE', 0) or 0
    shear = params.get('BWD_0-6km', 0) or 0

    # CSS per a les targetes (ara amb una variant per al veredicte)
    st.markdown("""
    <style>
    .guide-card { background-color: #f0f2f6; border: 1px solid #d1d1d1; border-radius: 8px; padding: 15px; margin-bottom: 12px; }
    .guide-title { font-size: 1.1em; font-weight: bold; color: #1a1a2e; display: flex; align-items: center; margin-bottom: 8px; }
    .guide-icon { font-size: 1.3em; margin-right: 10px; }
    .guide-text { font-size: 0.95em; color: #333; line-height: 1.6; }
    .guide-text strong { color: #d9534f; }
    .verdict-card { border-left: 5px solid; padding: 16px; margin-bottom: 15px; border-radius: 8px; background-color: #262730; }
    .verdict-title { font-size: 1.2em; font-weight: bold; color: white; }
    .verdict-motiu { font-size: 0.9em; color: #b0b0c8; font-style: italic; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

    # --- NOU: Targeta de Veredicte Principal ---
    st.markdown(f"""
    <div class="verdict-card" style="border-left-color: {veredicte_caca['color']};">
        <div class="verdict-title">Val la pena anar-hi? <span style="color: {veredicte_caca['color']};">{veredicte_caca['text']}</span></div>
        <div class="verdict-motiu">{veredicte_caca['motiu']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Targeta 1: Alçada
    st.markdown(f"""
    <div class="guide-card">
        <div class="guide-title"><span class="guide-icon">📏</span>Alçada del Cim (Top)</div>
        <div class="guide-text">El cim de la tempesta simulada arriba fins als <strong>{el_hgt_km:.1f} km</strong>. Cims més alts solen indicar tempestes més potents.</div>
    </div>
    """, unsafe_allow_html=True)

    # Targeta 2: Colors (Reflectivitat)
    st.markdown(f"""
    <div class="guide-card">
        <div class="guide-title"><span class="guide-icon">🎨</span>Significat dels Colors</div>
        <div class="guide-text">Els colors representen la intensitat: vermells/magentes són pluja molt forta o <strong>calamarsa</strong>. Aquesta tempesta té un CAPE de <strong>{cape:.0f} J/kg</strong>.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Targeta 3: Forma (Inclinació)
    st.markdown(f"""
    <div class="guide-card">
        <div class="guide-title"><span class="guide-icon">📐</span>Forma i Inclinació</div>
        <div class="guide-text">La tempesta s'inclina a causa d'un cisallament de <strong>{shear:.0f} nusos</strong>, un signe de bona <strong>organització</strong>.</div>
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






@st.cache_data(show_spinner="Carregant mapa de selecció...")
def carregar_dades_geografiques():
    """
    Carrega i processa l'arxiu GeoJSON de les comarques per utilitzar-lo al mapa.
    Aquesta versió utilitza un camí absolut per ser compatible amb Streamlit Cloud.
    """
    try:
        # --- LÍNIES MODIFICADES ---
        # Troba el directori on s'està executant l'script actual
        script_dir = os.path.dirname(__file__)
        # Crea el camí complet i correcte a l'arxiu geojson
        file_path = os.path.join(script_dir, "comarques.geojson")
        # --------------------------

        # Carreguem el fitxer GeoJSON amb el camí complet
        gdf = gpd.read_file(file_path)
        # Assegurem que la projecció sigui la correcta per a Folium
        gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error en carregar l'arxiu 'comarques.geojson'. Assegura't que estigui a la mateixa carpeta que l'script al teu repositori de GitHub. Detall: {e}")
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



def ui_sidebar_selectors():
    """
    Crea els controls de selecció a la barra lateral. Aquest mètode és 100% fiable.
    """
    st.sidebar.header("Zona d'Anàlisi")

    # --- CALLBACKS INTERNS PER A LA LÒGICA DE SELECCIÓ ---
    def on_comarca_change():
        # Quan canvia la comarca, resetejem la selecció de poble
        st.session_state.poble_selector = None
        # També netegem el widget del selector de poble per evitar inconsistències
        if 'poble_selector_widget' in st.session_state:
            st.session_state.poble_selector_widget = "--- Selecciona una opció ---"

    def on_poble_change():
        # Quan canvia el poble, actualitzem l'estat principal
        poble = st.session_state.poble_selector_widget
        if poble and "---" not in poble:
            st.session_state.poble_selector = poble
        else:
            st.session_state.poble_selector = None

    # --- WIDGETS DE LA BARRA LATERAL ---

    # Selector de Comarca
    comarques = sorted(list(CIUTATS_PER_COMARCA.keys()))
    st.sidebar.selectbox(
        "Pas 1: Tria una comarca",
        options=["--- Selecciona una opció ---"] + comarques,
        key='selected_comarca',
        on_change=on_comarca_change
    )

    # Si s'ha triat una comarca, mostrem el selector de poblacions
    comarca_sel = st.session_state.get('selected_comarca')
    if comarca_sel and "---" not in comarca_sel:
        poblacions = sorted(list(CIUTATS_PER_COMARCA.get(comarca_sel, {}).keys()))
        st.sidebar.selectbox(
            "Pas 2: Tria una localitat",
            options=["--- Selecciona una opció ---"] + poblacions,
            key="poble_selector_widget",
            on_change=on_poble_change
        )


def ui_mapa_display():
    """
    Aquesta funció NOMÉS MOSTRA el mapa com un visor. Reacciona a les seleccions
    fetes a la barra lateral, però no és un control d'entrada.
    """
    st.markdown("#### Mapa de Situació")
    gdf = carregar_dades_geografiques()
    if gdf is None: return

    comarca_sel = st.session_state.get('selected_comarca')
    poble_sel = st.session_state.get('poble_selector')

    # Determina el centre i el zoom
    map_center = [41.83, 1.87]
    zoom_level = 8
    if comarca_sel and "---" not in comarca_sel:
        comarca_shape = gdf[gdf['nomcomar'] == comarca_sel]
        if not comarca_shape.empty:
            map_center = [comarca_shape.geometry.centroid.y.iloc[0], comarca_shape.geometry.centroid.x.iloc[0]]
            zoom_level = 10
    
    m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron", scrollWheelZoom=False)

    # Dibuixa les comarques, ressaltant la seleccionada
    def style_function(feature):
        style = {'fillColor': '#28a745', 'color': 'black', 'weight': 1, 'fillOpacity': 0.1}
        if comarca_sel and feature['properties']['nomcomar'] == comarca_sel:
            style['fillColor'] = '#FF4B4B'; style['fillOpacity'] = 0.5
        return style
    folium.GeoJson(gdf, style_function=style_function).add_to(m)

    # Si s'ha triat un poble, afegeix un marcador
    if poble_sel:
        coords = CIUTATS_CATALUNYA[poble_sel]
        folium.Marker(
            location=[coords['lat'], coords['lon']],
            tooltip=poble_sel,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    st_folium(m, width="100%", height=400, returned_objects=[])
        
        
def ui_mapa_interactiu_seleccio():
    """
    VERSIÓ SIMPLIFICADA: Dibuixa el mapa de comarques i ressalta la seleccionada.
    Retorna les dades del clic per a ser processades externament.
    """
    st.markdown("#### Pas 1: Selecciona una comarca al mapa")
    
    gdf = carregar_dades_geografiques()
    if gdf is None:
        return None, None

    selected_comarca_name = st.session_state.get('selected_comarca', None)
    
    # El mapa sempre estarà centrat a Catalunya
    m = folium.Map(location=[41.83, 1.87], zoom_start=8, tiles="CartoDB positron", scrollWheelZoom=False)

    def style_function(feature):
        # Estil per defecte per a les comarques
        style = {'fillColor': '#28a745', 'color': 'black', 'weight': 1, 'fillOpacity': 0.4}
        # Si la comarca és la que està seleccionada, canvia el seu estil per ressaltar-la
        if selected_comarca_name and feature['properties']['nomcomar'] == selected_comarca_name:
            style['fillColor'] = '#FF4B4B'
            style['fillOpacity'] = 0.7
            style['weight'] = 2.5
        return style

    # Dibuixa totes les comarques aplicant l'estil dinàmic
    folium.GeoJson(
        gdf,
        style_function=style_function,
        highlight_function=lambda x: {'weight': 3, 'color': '#FF4B4B'},
        tooltip=folium.GeoJsonTooltip(fields=['nomcomar'], aliases=['Comarca:']),
    ).add_to(m)

    # Renderitza el mapa i retorna les dades del clic
    map_data = st_folium(m, width="100%", height=450, returned_objects=['last_object_clicked_tooltip'])
    return map_data, gdf
    

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
    
            
@st.cache_data(ttl=3600)
def carregar_dades_sondeig_usa(lat, lon, hourly_index):
    try:
        h_base = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS_GFS]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "gfs_seamless", "forecast_days": 3}
        response = openmeteo.weather_api(API_URL_USA, params=params)[0]
        hourly = response.Hourly()

        valid_index = None
        max_hours_to_check = 3
        total_hours = len(hourly.Variables(0).ValuesAsNumpy())

        for offset in range(max_hours_to_check + 1):
            indices_to_try = sorted(list(set([hourly_index + offset, hourly_index - offset])))
            for h_idx in indices_to_try:
                if 0 <= h_idx < total_hours:
                    sfc_check = [hourly.Variables(i).ValuesAsNumpy()[h_idx] for i in range(len(h_base))]
                    if not any(np.isnan(val) for val in sfc_check):
                        valid_index = h_idx
                        break
            if valid_index is not None:
                break
        
        if valid_index is None:
            # RETORN SIMPLIFICAT
            return None, hourly_index, "No s'han trobat dades vàlides properes a l'hora sol·licitada."

        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[valid_index] for i, v in enumerate(h_base)}

        sfc_temp_C = sfc_data["temperature_2m"] * units.degC
        sfc_rh_percent = sfc_data["relative_humidity_2m"] * units.percent
        sfc_dew_point = mpcalc.dewpoint_from_relative_humidity(sfc_temp_C, sfc_rh_percent).m
        
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_GFS) + j).ValuesAsNumpy()[valid_index] for j in range(len(PRESS_LEVELS_GFS))]
        
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_dew_point], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]

        for i, p_val in enumerate(PRESS_LEVELS_GFS):
            if p_val < sfc_data["surface_pressure"] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        # RETORN SIMPLIFICAT
        return processed_data, valid_index, error
    except Exception as e:
        # RETORN SIMPLIFICAT
        return None, hourly_index, f"Error en carregar dades del sondeig GFS: {e}"

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

@st.cache_data(ttl=3600)
def carregar_dades_mapa_usa(nivell, hourly_index):
    try:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        map_data_raw, error = carregar_dades_mapa_base_usa(variables, hourly_index)
        if error: return None, error
        temp_data = np.array(map_data_raw.pop(f'temperature_{nivell}hPa')) * units.degC
        rh_data = np.array(map_data_raw.pop(f'relative_humidity_{nivell}hPa')) * units.percent
        map_data_raw['dewpoint_data'] = mpcalc.dewpoint_from_relative_humidity(temp_data, rh_data).m
        map_data_raw['speed_data'] = map_data_raw.pop(f'wind_speed_{nivell}hPa')
        map_data_raw['dir_data'] = map_data_raw.pop(f'wind_direction_{nivell}hPa')
        return map_data_raw, None
    except Exception as e:
        return None, f"Error en processar dades del mapa GFS: {e}"


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
    Calcula el valor de convergència per a un únic punt geogràfic.
    És una versió optimitzada de 'calcular_convergencies_per_llista' per a un sol cas.
    """
    if not map_data or 'lons' not in map_data or len(map_data['lons']) < 4:
        return np.nan

    try:
        lons, lats = map_data['lons'], map_data['lats']
        speed_data, dir_data = map_data['speed_data'], map_data['dir_data']

        # Crear una graella d'interpolació
        grid_lon, grid_lat = np.meshgrid(
            np.linspace(min(lons), max(lons), 100),
            np.linspace(min(lats), max(lats), 100)
        )

        # Calcular components U i V i interpolar-los
        u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
        grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')
        grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'cubic')

        # Calcular la convergència a tota la graella
        with np.errstate(invalid='ignore'):
            dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
            divergence = mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)
            convergence_scaled = -divergence.to('1/s').magnitude * 1e5

        # Trobar el punt més proper a les coordenades donades
        dist_sq = (grid_lat - lat_sel)**2 + (grid_lon - lon_sel)**2
        min_dist_idx = np.unravel_index(np.argmin(dist_sq, axis=None), dist_sq.shape)
        
        # Retornar el valor de convergència en aquest punt
        valor_conv = convergence_scaled[min_dist_idx]
        
        return valor_conv if pd.notna(valor_conv) else np.nan

    except Exception:
        # En cas d'error, retornar NaN per no bloquejar l'app
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
    """
    VERSIÓ AMB LLISTA PER COMARQUES:
    - Construeix el selector de localitats a partir de l'estructura CIUTATS_PER_COMARCA.
    - Manté les alertes de convergència (🟡/🟠/🔴) al costat de cada nom.
    """
    st.markdown(f'<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | {zona_activa.replace("_", " ").title()}</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)
    is_developer = st.session_state.get('developer_mode', False)
    
    # Crear columnas según el modo
    if is_developer:
        col_text, col_change, col_dev, col_logout = st.columns([0.5, 0.15, 0.15, 0.15])
    else:
        col_text, col_change, col_logout = st.columns([0.7, 0.15, 0.15])
    
    with col_text:
        if not is_guest: 
            username = st.session_state.get('username', 'Usuari')
            if is_developer:
                st.markdown(f"👨‍💻 **{username}** (Mode Desenvolupador)")
            else:
                st.markdown(f"Benvingut/da, **{username}**!")
    
    with col_change:
        if st.button("Canviar a EEUU?", use_container_width=True):
            keys_to_delete = ['poble_selector', 'poble_selector_usa', 'zone_selected', 'active_tab_cat', 'active_tab_usa', 'dia_selector', 'hora_selector', 'dia_selector_usa_widget', 'hora_selector_usa']
            for key in keys_to_delete:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
    
    if is_developer:
        with col_dev:
            if st.button("🚫 Sortir Mode Dev", use_container_width=True, type="secondary"):
                st.session_state.developer_mode = False
                st.session_state.logged_in = False
                st.rerun()
    
    with col_logout:
        if st.button("Sortir" if is_guest else "Tanca Sessió", use_container_width=True):
            for key in list(st.session_state.keys()): 
                del st.session_state[key]
            st.rerun()

    with st.container(border=True):
        st.caption("💡 Les alertes (🟡/🟠/🔴) indiquen localitats amb 'disparador' (convergència ≥ 30).")
        
        def format_city_label(city_key):
            # Formata les comarques com a títols no seleccionables
            if city_key.startswith("---"): return f"--- {city_key[3:]} ---"
            if not convergencies or city_key not in convergencies: return city_key
            
            conv = convergencies.get(city_key)
            if isinstance(conv, (int, float)):
                emoji = "🔴" if conv >= 80 else "🟠" if conv >= 50 else "🟡" if conv >= 30 else ""
                return f"{city_key} ({emoji} {conv:.0f})" if emoji else city_key
            return city_key

        if zona_activa == 'catalunya':
            col_loc, col_dia, col_hora, col_nivell = st.columns(4)
            with col_loc:
                # Construeix la llista d'opcions a partir de les comarques
                opcions_finals = []
                for comarca, ciutats_dict in sorted(CIUTATS_PER_COMARCA.items()):
                    # Afegeix la comarca com a capçalera (es marcarà al format_func)
                    opcions_finals.append(f"---{comarca.upper()}---")
                    # Afegeix les ciutats de la comarca, ordenades alfabèticament
                    opcions_finals.extend(sorted(ciutats_dict.keys()))

                # Afegeix els punts marins al final
                opcions_finals.append("---ZONES MARINES---")
                opcions_finals.extend(sorted(PUNTS_MAR.keys()))
                
                poble_actual = st.session_state.get('poble_selector', 'Barcelona')
                idx = opcions_finals.index(poble_actual) if poble_actual in opcions_finals else 0
                
                st.selectbox(
                    "Localitat:", 
                    options=opcions_finals, 
                    key="poble_selector", 
                    format_func=format_city_label, 
                    index=idx
                )

            with col_dia:
                now_local = datetime.now(TIMEZONE_CAT)
                opcions_dia = [(now_local + timedelta(days=i)).strftime('%d/%m/%Y') for i in range(2)]
                st.selectbox("Dia:", opcions_dia, key="dia_selector", disabled=is_guest)
            with col_hora:
                opcions_hora = [f"{h:02d}:00h" for h in range(24)]
                hora_actual = st.session_state.get('hora_selector', f"{datetime.now(TIMEZONE_CAT).hour:02d}:00h")
                idx = opcions_hora.index(hora_actual) if hora_actual in opcions_hora else 0
                st.selectbox("Hora:", opcions_hora, key="hora_selector", disabled=is_guest, index=idx)
            with col_nivell:
                if not is_guest:
                    nivells = [1000, 950, 925, 900, 850, 800, 700]; nivell_actual = st.session_state.get('level_cat_main', 925)
                    idx = nivells.index(nivell_actual) if nivell_actual in nivells else 2
                    st.selectbox("Nivell:", nivells, key="level_cat_main", index=idx, format_func=lambda x: f"{x} hPa")
        
        else: # Zona USA (aquesta part no canvia)
            col_loc, col_dia, col_hora, col_nivell = st.columns(4)
            with col_loc:
                actives, inactives = [], []
                if convergencies:
                    for city, conv in convergencies.items():
                        is_numeric = isinstance(conv, (int, float))
                        (actives if is_numeric and conv >= 5 else inactives).append(city)
                    actives.sort(key=lambda k: convergencies[k], reverse=True)
                else:
                    inactives = sorted(list(USA_CITIES.keys()))
                
                opcions_finals_usa = []
                if actives: opcions_finals_usa.extend(["--- FOCUS DE CONVERGÈNCIA ---"] + actives)
                if inactives: opcions_finals_usa.extend(["--- ALTRES CIUTATS ---"] + sorted(inactives))
                
                poble_actual = st.session_state.get('poble_selector_usa', "Oklahoma City, OK")
                idx = opcions_finals_usa.index(poble_actual) if poble_actual in opcions_finals_usa else 0
                st.selectbox("Ciutat:", options=opcions_finals_usa, key="poble_selector_usa", format_func=format_city_label, index=idx)
            
            with col_dia:
                now_usa = datetime.now(TIMEZONE_USA)
                opcions_dia = [(now_usa + timedelta(days=i)).strftime('%d/%m/%Y') for i in range(3)]
                st.selectbox("Dia:", opcions_dia, key="dia_selector_usa_widget", on_change=on_day_change_usa)
            
            with col_hora:
                now_spain = datetime.now(TIMEZONE_CAT)
                opcions_hora = []
                for h in range(24):
                    time_in_usa = TIMEZONE_USA.localize(datetime.now().replace(hour=h, minute=0, second=0, microsecond=0))
                    time_in_spain_equivalent = time_in_usa.astimezone(TIMEZONE_CAT)
                    opcions_hora.append(f"{h:02d}:00 (Local: {time_in_spain_equivalent.hour:02d}:00h)")
                
                hora_actual_usa = st.session_state.get('hora_selector_usa', f"{now_spain.astimezone(TIMEZONE_USA).hour:02d}:00 (Local: {now_spain.hour:02d}:00h)")
                idx = opcions_hora.index(hora_actual_usa) if hora_actual_usa in opcions_hora else 0
                st.selectbox("Hora (Central Time):", opcions_hora, key="hora_selector_usa", index=idx)
                
            with col_nivell:
                nivells_usa = [925, 850, 700, 500, 300]
                nivell_actual = st.session_state.get('level_usa_main', 850)
                idx = nivells_usa.index(nivell_actual) if nivell_actual in nivells_usa else 1
                st.selectbox("Nivell:", nivells_usa, key="level_usa_main", index=idx, format_func=lambda x: f"{x} hPa")



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
    st.markdown("#### Mapes de Pronòstic (Model GFS)")
    
    with st.spinner(f"Carregant dades del mapa GFS a {nivell_sel}hPa..."):
        map_data, error_map = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
    
    if error_map:
        st.error(f"Error en carregar el mapa: {error_map}")
    elif map_data:
        fig = crear_mapa_forecast_combinat_usa(map_data['lons'], map_data['lats'], map_data['speed_data'], map_data['dir_data'], map_data['dewpoint_data'], nivell_sel, timestamp_str)
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

# --- Lògica Principal de l'Aplicació ---

def run_catalunya_app():
    # --- PAS 1: DIBUIXAR LA INTERFÍCIE ESTÀTICA I ELS CONTROLS ---
    
    # Dibuixa els controls a la barra lateral
    ui_sidebar_selectors()

    # Dibuixa la capçalera principal
    st.markdown('<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | Catalunya</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)
    is_developer = st.session_state.get('developer_mode', False)
    # ... (la resta de la teva capçalera amb botons)
    if is_developer:
        col_text, col_change, col_dev, col_logout = st.columns([0.5, 0.15, 0.15, 0.15])
    else:
        col_text, col_change, col_logout = st.columns([0.7, 0.15, 0.15])
    with col_text:
        if not is_guest: 
            username = st.session_state.get('username', 'Usuari')
            st.markdown(f"Benvingut/da, **{username}**!")
    with col_change:
        if st.button("Canviar a EEUU?", use_container_width=True):
            st.session_state.clear(); st.session_state.logged_in = True; st.session_state.zone_selected = 'valley_halley'; st.rerun()
    if is_developer:
        with col_dev:
            if st.button("🚫 Sortir Mode Dev", use_container_width=True, type="secondary"):
                st.session_state.clear(); st.rerun()
    with col_logout:
        if st.button("Sortir" if is_guest else "Tanca Sessió", use_container_width=True):
            st.session_state.clear(); st.rerun()
    st.divider()

    # --- PAS 2: DECIDIR QUÈ MOSTRAR BASANT-SE EN L'ESTAT ---

    poble_sel = st.session_state.get('poble_selector')

    # Si encara NO s'ha seleccionat un poble, mostrem el mapa i un missatge d'ajuda
    if not poble_sel:
        ui_mapa_display() # El mapa ara és només un visor
        st.info("⬅️ Utilitza els selectors de la barra lateral per triar una comarca i una localitat.")
        return # Aturem l'execució aquí

    # Si S'HA SELECCIONAT un poble, mostrem la interfície d'anàlisi completa
    else:
        st.success(f"### Anàlisi per a: **{poble_sel}**")
        
        # Inicialització de selectors de temps i pestanyes
        # ... (Aquesta part es manté exactament igual que la tenies)
        if 'dia_selector' not in st.session_state: st.session_state.dia_selector = datetime.now(TIMEZONE_CAT).strftime('%d/%m/%Y')
        if 'hora_selector' not in st.session_state: st.session_state.hora_selector = f"{datetime.now(TIMEZONE_CAT).hour:02d}:00h"
        if 'level_cat_main' not in st.session_state: st.session_state.level_cat_main = 925
        if 'active_tab_cat' not in st.session_state: st.session_state.active_tab_cat = "Anàlisi Vertical"

        with st.container(border=True):
            col_dia, col_hora, col_nivell = st.columns(3)
            with col_dia:
                now_local = datetime.now(TIMEZONE_CAT)
                opcions_dia = [(now_local + timedelta(days=i)).strftime('%d/%m/%Y') for i in range(2)]
                st.selectbox("Dia:", opcions_dia, key="dia_selector", disabled=is_guest)
            with col_hora:
                opcions_hora = [f"{h:02d}:00h" for h in range(24)]
                hora_actual = st.session_state.get('hora_selector')
                idx = opcions_hora.index(hora_actual) if hora_actual in opcions_hora else 12
                st.selectbox("Hora:", opcions_hora, key="hora_selector", disabled=is_guest, index=idx)
            with col_nivell:
                if not is_guest:
                    nivells = [1000, 950, 925, 900, 850, 800, 700]
                    st.selectbox("Nivell (Mapes):", nivells, key="level_cat_main", index=2, format_func=lambda x: f"{x} hPa")

        # Càlculs i la resta de la lògica de les pestanyes...
        # ... (Aquesta part es manté exactament igual que la tenies)
        dia_sel_str = st.session_state.dia_selector
        hora_sel_str = st.session_state.hora_selector
        nivell_sel = st.session_state.level_cat_main if not is_guest else 925
        lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']
        
        target_date = datetime.strptime(dia_sel_str, '%d/%m/%Y').date()
        local_dt = TIMEZONE_CAT.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_str.split(':')[0])))
        start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
        timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} (Local)"

        menu_options = ["Anàlisi Vertical", "Anàlisi de Mapes", "Anàlisi de Vents", "Tall Vertical Simulat"]
        menu_icons = ["graph-up-arrow", "map", "wind", "moisture"]
        if not is_guest:
            menu_options.append("💬 Assistent IA")
            menu_icons.append("chat-quote-fill")
        
        default_idx = menu_options.index(st.session_state.active_tab_cat) if st.session_state.active_tab_cat in menu_options else 0
        selected_tab = option_menu(menu_title=None, options=menu_options, icons=menu_icons, menu_icon="cast", orientation="horizontal", default_index=default_idx, key="catalunya_nav_selector")
        st.session_state.active_tab_cat = selected_tab

        if selected_tab == "Anàlisi de Mapes":
            ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str, nivell_sel)
        else:
            with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
                data_tuple, final_index, error_msg = carregar_dades_sondeig_cat(lat_sel, lon_sel, hourly_index_sel)
            
            if not error_msg and final_index != hourly_index_sel:
                adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
                adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_CAT)
                st.warning(f"**Avís:** No hi havia dades per a les {hora_sel_str}. Es mostren les de l'hora més propera: **{adjusted_local_time.strftime('%H:%Mh')}**.")

            if error_msg: 
                st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
            else:
                params_calc = data_tuple[1] if data_tuple else {}
                map_data_conv, _ = carregar_dades_mapa_cat(nivell_sel, hourly_index_sel)
                if map_data_conv:
                    conv_value = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
                    if pd.notna(conv_value):
                        params_calc[f'CONV_{nivell_sel}hPa'] = conv_value
                
                if selected_tab == "Anàlisi Vertical":
                    avis_proximitat = analitzar_amenaça_convergencia_propera(map_data_conv, params_calc, lat_sel, lon_sel, nivell_sel)
                    ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_str, timestamp_str, avis_proximitat)
                
                elif selected_tab == "Anàlisi de Vents":
                    ui_pestanya_analisis_vents(data_tuple, poble_sel, hora_sel_str, timestamp_str)

                elif selected_tab == "Tall Vertical Simulat":
                    col_esquerra, col_dreta = st.columns([0.7, 0.3])
                    with col_esquerra:
                        st.markdown(f"#### Simulació de Cicle de Vida per a {poble_sel}")
                        st.caption(timestamp_str)
                        if 'regenerate_key' not in st.session_state:
                            st.session_state.regenerate_key = 0
                        if st.button("🔄 Regenerar Animació", help="Crea una nova versió de l'animació."):
                            st.session_state.regenerate_key += 1
                        with st.spinner("Generant animació..."):
                            params_tuple = tuple(sorted(params_calc.items()))
                            gif_bytes = generar_animacio_cachejada(params_tuple, hora_sel_str, st.session_state.regenerate_key)
                        if gif_bytes is None:
                            st.warning("Les condicions actuals (CAPE < 100) no són suficients per generar una tempesta simulada.")
                        else:
                            st.image(gif_bytes, caption="Animació del cicle de vida simulat (2 hores).")
                    with col_dreta:
                        ui_guia_tall_vertical(params_calc, nivell_sel)

                elif selected_tab == "💬 Assistent IA" and not is_guest:
                    analisi_temps = analitzar_potencial_meteorologic(params_calc, nivell_sel, hora_sel_str)
                    interpretacions_ia = interpretar_parametres(params_calc, nivell_sel)
                    sounding_data = data_tuple[0] if data_tuple else None
                    ui_pestanya_assistent_ia(params_calc, poble_sel, analisi_temps, interpretacions_ia, sounding_data)

def run_valley_halley_app():
    # --- PAS 1: INICIALITZACIÓ ROBUSTA DE L'ESTAT ---
    if 'poble_selector_usa' not in st.session_state:
        st.session_state.poble_selector_usa = "Oklahoma City, OK"
    
    # <<-- LÍNIA CLAU AFEGIDA: Inicialitzem la variable que faltava -->>
    if 'dia_selector_usa_widget' not in st.session_state:
        st.session_state.dia_selector_usa_widget = datetime.now(TIMEZONE_USA).strftime('%d/%m/%Y')
        
    if 'hora_selector_usa' not in st.session_state:
        now_spain = datetime.now(TIMEZONE_CAT)
        time_in_usa = now_spain.astimezone(TIMEZONE_USA)
        st.session_state.hora_selector_usa = f"{time_in_usa.hour:02d}:00 (Local: {now_spain.hour:02d}:00h)"
    if 'level_usa_main' not in st.session_state:
        st.session_state.level_usa_main = 850
    if 'active_tab_usa' not in st.session_state:
        st.session_state.active_tab_usa = "Anàlisi de Mapes"

    # --- PAS 2: CÀLCULS PREVIS I CAPÇALERA ---
    pre_hora_sel_text = st.session_state.hora_selector_usa
    pre_hora_sel_cst = pre_hora_sel_text.split(' ')[0]
    pre_dia_sel = st.session_state.dia_selector_usa_widget
    pre_target_date = datetime.strptime(pre_dia_sel, '%d/%m/%Y').date()
    pre_local_dt = TIMEZONE_USA.localize(datetime.combine(pre_target_date, datetime.min.time()).replace(hour=int(pre_hora_sel_cst.split(':')[0])))
    pre_start_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    pre_hourly_index = int((pre_local_dt.astimezone(pytz.utc) - pre_start_utc).total_seconds() / 3600)

    NIVELL_ANALISI_CONV = 850
    pre_map_data, _ = carregar_dades_mapa_usa(NIVELL_ANALISI_CONV, pre_hourly_index)
    pre_convergencies = calcular_convergencies_per_llista(pre_map_data, USA_CITIES) if pre_map_data else {}
    
    ui_capcalera_selectors(None, zona_activa="tornado_alley", convergencies=pre_convergencies)

    # --- PAS 3: LLEGIR ESTAT FINAL I CARREGAR DADES ---
    poble_sel = st.session_state.poble_selector_usa
    if "---" in poble_sel:
        st.info("Selecciona una ciutat de la llista per començar l'anàlisi.")
        return

    dia_sel_str = st.session_state.dia_selector_usa_widget
    hora_sel_str_full = st.session_state.hora_selector_usa
    hora_sel_cst_only = hora_sel_str_full.split(' ')[0]
    nivell_sel = st.session_state.level_usa_main
    lat_sel, lon_sel = USA_CITIES[poble_sel]['lat'], USA_CITIES[poble_sel]['lon']

    target_date = datetime.strptime(dia_sel_str, '%d/%m/%Y').date()
    local_dt = TIMEZONE_USA.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=int(hora_sel_cst_only.split(':')[0])))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    timestamp_str = f"{dia_sel_str} a les {hora_sel_cst_only} (Central Time)"

    # --- PAS 4: DIBUIXAR MENÚ I RESULTATS ---
    menu_options_usa = ["Anàlisi de Mapes", "Anàlisi Vertical", "Satèl·lit (Temps Real)"]
    menu_icons_usa = ["map-fill", "graph-up-arrow", "globe-americas"]
    default_idx_usa = menu_options_usa.index(st.session_state.active_tab_usa) if st.session_state.active_tab_usa in menu_options_usa else 0

    selected_tab_usa = option_menu(
        menu_title=None, 
        options=menu_options_usa, 
        icons=menu_icons_usa,
        menu_icon="cast", 
        orientation="horizontal", 
        default_index=default_idx_usa,
        key="usa_nav_selector"
    )
    
    st.session_state.active_tab_usa = selected_tab_usa

    if selected_tab_usa == "Anàlisi de Mapes":
        with st.spinner(f"Carregant mapa GFS a {nivell_sel}hPa..."):
            map_data_final, _ = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
        if map_data_final:
            fig = crear_mapa_forecast_combinat_usa(map_data_final['lons'], map_data_final['lats'], map_data_final['speed_data'], map_data_final['dir_data'], map_data_final['dewpoint_data'], nivell_sel, timestamp_str)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else: st.warning(f"No s'han pogut carregar les dades del mapa per al nivell {nivell_sel}hPa.")
            
    elif selected_tab_usa == "Anàlisi Vertical":
        with st.spinner(f"Carregant dades del sondeig per a {poble_sel}..."):
            data_tuple, final_index, error_msg = carregar_dades_sondeig_usa(lat_sel, lon_sel, hourly_index_sel)
        
        if not error_msg and final_index != hourly_index_sel:
            adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
            adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_USA)
            st.warning(f"**Avís:** No hi havia dades per a les {hora_sel_str_full}. Es mostren les de l'hora més propera: **{adjusted_local_time.strftime('%H:%M')}** (Central Time).")

        if error_msg:
            st.error(f"No s'ha pogut carregar el sondeig: {error_msg}")
        else:
            params_calc = data_tuple[1] if data_tuple else {}
            if nivell_sel == NIVELL_ANALISI_CONV and poble_sel in pre_convergencies:
                params_calc[f'CONV_{nivell_sel}hPa'] = pre_convergencies[poble_sel]
            else:
                with st.spinner(f"Calculant convergència a {nivell_sel}hPa..."):
                    map_data_nivell_sel, _ = carregar_dades_mapa_usa(nivell_sel, hourly_index_sel)
                    if map_data_nivell_sel:
                        params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_nivell_sel, lat_sel, lon_sel)
            
            avis_proximitat_usa = analitzar_amenaça_convergencia_propera(pre_map_data, params_calc, lat_sel, lon_sel, nivell_sel)
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, hora_sel_cst_only, timestamp_str, avis_proximitat_usa)
        
    elif selected_tab_usa == "Satèl·lit (Temps Real)":
        ui_pestanya_satelit_usa()
        
def ui_zone_selection():
    st.markdown("<h1 style='text-align: center;'>Zona d'Anàlisi</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # --- LÍNIES MODIFICADES ---
    # Camins als nous arxius de vídeo (animacions)
    path_anim_cat = "catalunya_anim.mp4"
    path_anim_usa = "tornado_alley_anim.mp4"
    # ---------------------------

    with st.spinner('Carregant entorns geoespacials...'): 
        time.sleep(1) 

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            # --- CANVI CLAU: Fem servir la nova funció per a la animació ---
            html_video_cat = generar_html_video_animacio(path_anim_cat)
            st.markdown(html_video_cat, unsafe_allow_html=True)
            # -------------------------------------------------------------

            st.subheader("Catalunya")
            st.write("Anàlisi detallada d'alta resolució (Model AROME) per al territori català. Ideal per a seguiment de tempestes locals, fenòmens de costa i muntanya.")
            
            if st.button("Analitzar Catalunya", use_container_width=True, type="primary", key="btn_select_catalunya"):
                st.session_state['zone_selected'] = 'catalunya'
                st.rerun()
    with col2:
        with st.container(border=True):
            # --- CANVI CLAU: Fem servir la nova funció per a la animació ---
            html_video_usa = generar_html_video_animacio(path_anim_usa)
            st.markdown(html_video_usa, unsafe_allow_html=True)
            # -------------------------------------------------------------
            
            st.subheader("Tornado Alley (EUA)")
            st.write("Anàlisi a escala sinòptica (Model GFS) per al 'Corredor de Tornados' dels EUA. Perfecte per a l'estudi de sistemes de gran escala i temps sever organitzat.")
            
            if st.button("Analitzar Tornado Alley", use_container_width=True, key="btn_select_usa"):
                st.session_state['zone_selected'] = 'valley_halley'
                st.rerun()
def main():
    inject_custom_css()
    hide_streamlit_style()
    
    # Inicialització d'estat a prova d'errors
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'zone_selected' not in st.session_state:
        st.session_state.zone_selected = None
    if 'developer_mode' not in st.session_state:
        st.session_state.developer_mode = False

    # --- FLUX DE NAVEGACIÓ PRINCIPAL ---

    # 1. Si l'usuari no ha iniciat sessió, mostra la pàgina de login i atura't.
    if not st.session_state.logged_in:
        afegir_video_de_fons()
        show_login_page()
        return

    # 2. Si l'usuari ha iniciat sessió però no ha triat zona, mostra la selecció i atura't.
    if st.session_state.zone_selected is None:
        ui_zone_selection()
        return

    # 3. Si s'ha triat una zona, executa l'aplicació corresponent.
    if st.session_state.zone_selected == 'catalunya':
        with st.spinner("Preparant l'entorn d'anàlisi de Catalunya..."):
            run_catalunya_app()
            
    elif st.session_state.zone_selected == 'valley_halley':
        with st.spinner("Preparant l'entorn d'anàlisi de Tornado Alley..."):
            run_valley_halley_app()

def analitzar_potencial_meteorologic(params, nivell_conv, hora_actual=None):
    """
    Sistema de Diagnòstic v28.0 - Lògica Jeràrquica amb LFC.
    1. Comprova si hi ha inhibició o falta de disparador.
    2. Comprova si l'LFC és > 3000m (convecció de base alta).
    3. Només si els filtres anteriors es superen, classifica la tempesta.
    """
    # --- 1. EXTRACCIÓ COMPLETA I ROBUSTA DE PARÀMETRES ---
    sbcape = params.get('SBCAPE', 0) or 0; mlcape = params.get('MLCAPE', 0) or 0; mucape = params.get('MUCAPE', 0) or 0
    mucin = params.get('MUCIN', 0) or 0
    lfc_hgt = params.get('LFC_Hgt', 9999) or 9999
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    srh_3km = params.get('SRH_0-3km', 0) or 0
    
    rh_capes = params.get('RH_CAPES', {'baixa': 0, 'mitjana': 0, 'alta': 0})
    rh_baixa = rh_capes.get('baixa', 0) if pd.notna(rh_capes.get('baixa')) else 0
    rh_mitjana = rh_capes.get('mitjana', 0) if pd.notna(rh_capes.get('mitjana')) else 0
    rh_alta = rh_capes.get('alta', 0) if pd.notna(rh_capes.get('alta')) else 0

    conv_key = f'CONV_{nivell_conv}hPa'
    conv = params.get(conv_key, 0) or 0

    # --- 2. FILTRE PRINCIPAL: INHIBICIÓ O SENSE DISPARADOR ---
    # Si hi ha una tapa molt forta o no hi ha un mecanisme d'ascens, no hi haurà tempesta.
    if mucin < -100 or conv < 25:
        if rh_baixa > 80: return {'emoji': "☁️", 'descripcio': "Cel Cobert (Estratus)", 'veredicte': "Capa de núvols baixos sense desenvolupament vertical.", 'factor_clau': "Inhibició o falta de disparador."}
        if rh_mitjana > 70: return {'emoji': "🌥️", 'descripcio': "Núvols Mitjans (Altocúmulus)", 'veredicte': "Cel variable amb núvols a nivells mitjans.", 'factor_clau': "Inhibició o falta de disparador."}
        if rh_alta > 60: return {'emoji': "🌤️", 'descripcio': "Núvols Alts (Cirrus)", 'veredicte': "Cel poc ennuvolat amb núvols alts.", 'factor_clau': "Inhibició o falta de disparador."}
        return {'emoji': "☀️", 'descripcio': "Cel Serè", 'veredicte': "Temps estable. Les condicions no són favorables per a la convecció.", 'factor_clau': "Inhibició o falta de disparador."}

    # --- 3. FILTRE CLAU: LFC > 3000m (CONVECCIÓ DE BASE ALTA) ---
    # Si la convecció s'inicia massa amunt, no formarà tempestes organitzades.
    if lfc_hgt > 3000:
        desc = "Altocúmulus Castellanus"
        veredicte = "Potencial per a convecció de base elevada. No s'esperen tempestes organitzades a la superfície."
        # Si hi ha molta humitat a sota, podrien ser Stratocumulus Castellanus
        if rh_baixa > 70: desc = "Stratocumulus Castellanus"
        
        return {'emoji': "🌥️", 'descripcio': desc, 'veredicte': veredicte, 'factor_clau': f"LFC elevat ({lfc_hgt:.0f} m)."}

    # --- 4. SI ARRIBEM AQUÍ (LFC < 3000m), CLASSIFIQUEM LA TEMPESTA ---
    # Utilitzem el MUCAPE (el més representatiu) per classificar la intensitat.
    
    # Cas 1: Potencial de Supercèl·lula (el més sever)
    if mucape > 1500 and bwd_6km > 35 and srh_3km > 200:
        return {'emoji': "🌪️", 'descripcio': "Potencial de Supercèl·lula", 'veredicte': "Condicions molt favorables per a la formació de tempestes rotatòries i severes.", 'factor_clau': "Alt CAPE, fort cisallament i helicitat."}
    
    # Cas 2: Potencial de Multicèl·lula Organitzada
    elif mucape > 1000 and bwd_6km > 25:
        return {'emoji': "⛈️", 'descripcio': "Tempestes Organitzades (Multicèl·lula)", 'veredicte': "Potencial per a la formació de grups de tempestes o línies organitzades.", 'factor_clau': "CAPE moderat-alt i cisallament suficient."}
        
    # Cas 3: Tempesta Aïllada (Cumulonimbus)
    elif mucape > 800:
        return {'emoji': "🌩️", 'descripcio': "Tempesta Aïllada (Cumulonimbus)", 'veredicte': "Condicions favorables per al desenvolupament de tempestes aïllades, possiblement fortes.", 'factor_clau': "CAPE suficient per a un desenvolupament vertical complet."}

    # Cas 4: Xàfecs o Desenvolupament Vertical (Congestus)
    elif mucape > 400:
        return {'emoji': "☁️", 'descripcio': "Desenvolupament Vertical (Congestus)", 'veredicte': "Potencial per a núvols de gran creixement que podrien deixar xàfecs aïllats.", 'factor_clau': "CAPE moderat, suficient per a congestus."}

    # Cas 5: Núvols de Bon Temps (Humilis/Mediocris)
    else: # MUCAPE < 400
        return {'emoji': "🌤️", 'descripcio': "Núvols de Bon Temps (Cúmuls)", 'veredicte': "Es formaran petits cúmuls de bon temps amb poc o cap desenvolupament vertical.", 'factor_clau': "CAPE baix."}



    
if __name__ == "__main__":
    main()
