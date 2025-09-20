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
from folium.plugins import HeatMap
import geojsoncontour
from matplotlib.patches import Polygon
from matplotlib.patches import Polygon, Wedge
from matplotlib.patches import Polygon, Wedge, Circle
from typing import List, Tuple, Dict, Any  
import matplotlib.lines as mlines  
import scipy.ndimage as ndi
from math import radians, sin, cos, sqrt, atan2, degrees, asin
from scipy.signal import find_peaks
from matplotlib.patches import Ellipse
from scipy.ndimage import label


# --- 0. CONFIGURACIÓ I CONSTANTS ---
st.set_page_config(layout="wide", page_title="Terminal de Temps Sever")

# --- Clients API ---
parcel_lock = threading.Lock()
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)






PICS_CATALUNYA_PER_COMARCA = {
    # COMARQUES PIRINENQUES I PRE-PIRINENQUES
    "Val d'Aran": [
        {'name': "Pica d'Estats", 'lat': 42.665, 'lon': 1.442, 'ele': 3143},
        {'name': "Montardo", 'lat': 42.651, 'lon': 0.875, 'ele': 2833},
        {'name': "Tuc de Molières", 'lat': 42.636, 'lon': 0.723, 'ele': 3010},
        {'name': "Besiberri Nord", 'lat': 42.602, 'lon': 0.817, 'ele': 3008},
        {'name': "Besiberri Sud", 'lat': 42.595, 'lon': 0.820, 'ele': 3024},
        {'name': "Tuc de Colomèrs", 'lat': 42.620, 'lon': 0.933, 'ele': 2933},
        {'name': "Punta Leberta", 'lat': 42.628, 'lon': 0.908, 'ele': 2915},
        {'name': "Tuc de Ribereta", 'lat': 42.650, 'lon': 0.850, 'ele': 2800},
        {'name': "Tuc de Marimanya", 'lat': 42.640, 'lon': 0.780, 'ele': 2730},
        {'name': "Tuc de Cap de Colomèrs", 'lat': 42.610, 'lon': 0.940, 'ele': 2885}
    ],
    "Pallars Sobirà": [
        {'name': "Pica d'Estats", 'lat': 42.665, 'lon': 1.442, 'ele': 3143},
        {'name': "Montsent de Pallars", 'lat': 42.492, 'lon': 1.050, 'ele': 2883},
        {'name': "Tuc de la Cigalera", 'lat': 42.686, 'lon': 1.018, 'ele': 2668},
        {'name': "Punta Alta", 'lat': 42.570, 'lon': 0.888, 'ele': 3014},
        {'name': "Comaloforno", 'lat': 42.585, 'lon': 0.809, 'ele': 3029},
        {'name': "Tuc de Saboredo", 'lat': 42.640, 'lon': 1.050, 'ele': 2830},
        {'name': "Tuc de Baix", 'lat': 42.670, 'lon': 1.020, 'ele': 2710},
        {'name': "Tuc de Sarraera", 'lat': 42.700, 'lon': 1.080, 'ele': 2743},
        {'name': "Tuc de Mulleres", 'lat': 42.630, 'lon': 1.120, 'ele': 3010},
        {'name': "Tuc de la Tallada", 'lat': 42.610, 'lon': 1.180, 'ele': 2910}
    ],
    "Alta Ribagorça": [
        {'name': "Comaloforno", 'lat': 42.585, 'lon': 0.809, 'ele': 3029},
        {'name': "Punta Alta", 'lat': 42.570, 'lon': 0.888, 'ele': 3014},
        {'name': "Besiberri Nord", 'lat': 42.602, 'lon': 0.817, 'ele': 3008},
        {'name': "Besiberri Sud", 'lat': 42.595, 'lon': 0.820, 'ele': 3024},
        {'name': "Tuc de Colomèrs", 'lat': 42.620, 'lon': 0.933, 'ele': 2933},
        {'name': "Montardo", 'lat': 42.651, 'lon': 0.875, 'ele': 2833},
        {'name': "Tuc de Ratera", 'lat': 42.580, 'lon': 0.950, 'ele': 2862},
        {'name': "Tuc de la Montanyeta", 'lat': 42.560, 'lon': 0.920, 'ele': 2780},
        {'name': "Punta de Passet", 'lat': 42.590, 'lon': 0.870, 'ele': 2695},
        {'name': "Tuc de Pales", 'lat': 42.540, 'lon': 0.890, 'ele': 2850}
    ],
    "Pallars Jussà": [
        {'name': "Montsec d'Ares", 'lat': 42.046, 'lon': 0.760, 'ele': 1676},
        {'name': "Gallina Pelada", 'lat': 42.174, 'lon': 1.705, 'ele': 2321},
        {'name': "Pala Pedregosa", 'lat': 42.368, 'lon': 1.042, 'ele': 2450},
        {'name': "Sant Alís", 'lat': 42.120, 'lon': 0.940, 'ele': 1920},
        {'name': "Tossal de la Torre", 'lat': 42.180, 'lon': 1.150, 'ele': 2105},
        {'name': "Tuc de la Llenga", 'lat': 42.300, 'lon': 1.100, 'ele': 2280},
        {'name': "Serra de Boumort", 'lat': 42.215, 'lon': 1.139, 'ele': 2077},
        {'name': "Tossal de la Feixa", 'lat': 42.250, 'lon': 1.080, 'ele': 1980},
        {'name': "Roca de Pena", 'lat': 42.200, 'lon': 1.200, 'ele': 1850},
        {'name': "Tuc de l'Home", 'lat': 42.280, 'lon': 1.050, 'ele': 2340}
    ],
    "Cerdanya": [
        {'name': "Puigpedrós", 'lat': 42.493, 'lon': 1.841, 'ele': 2915},
        {'name': "La Tosa", 'lat': 42.330, 'lon': 1.879, 'ele': 2536},
        {'name': "Puigmal", 'lat': 42.383, 'lon': 2.112, 'ele': 2910},
        {'name': "Tossa Plana de Lles", 'lat': 42.400, 'lon': 1.650, 'ele': 2909},
        {'name': "Bastiments", 'lat': 42.455, 'lon': 2.222, 'ele': 2881},
        {'name': "Carlit", 'lat': 42.570, 'lon': 1.930, 'ele': 2921},
        {'name': "Pic de la Vaca", 'lat': 42.420, 'lon': 1.780, 'ele': 2820},
        {'name': "Pic de Finestrelles", 'lat': 42.410, 'lon': 2.100, 'ele': 2830},
        {'name': "Tosa d'Alp", 'lat': 42.350, 'lon': 1.890, 'ele': 2536},
        {'name': "Puig de Font Negra", 'lat': 42.450, 'lon': 1.950, 'ele': 2750}
    ],
    "Alt Urgell": [
        {'name': "Vulturó", 'lat': 42.278, 'lon': 1.670, 'ele': 2648},
        {'name': "Salòria", 'lat': 42.433, 'lon': 1.403, 'ele': 2789},
        {'name': "El Boumort", 'lat': 42.215, 'lon': 1.139, 'ele': 2077},
        {'name': "Tossal de la Truita", 'lat': 42.300, 'lon': 1.500, 'ele': 2450},
        {'name': "Comabona", 'lat': 42.350, 'lon': 1.600, 'ele': 2540},
        {'name': "Tuc de la Dona", 'lat': 42.280, 'lon': 1.550, 'ele': 2380},
        {'name': "Serra del Cadí", 'lat': 42.288, 'lon': 1.849, 'ele': 2276},
        {'name': "Tossal de la Llosa", 'lat': 42.250, 'lon': 1.450, 'ele': 2200},
        {'name': "Pic de la Muga", 'lat': 42.400, 'lon': 1.350, 'ele': 2680},
        {'name': "Tuc de l'Orri", 'lat': 42.320, 'lon': 1.400, 'ele': 2420}
    ],
    "Ripollès": [
        {'name': "Puigmal", 'lat': 42.383, 'lon': 2.112, 'ele': 2910},
        {'name': "Bastiments", 'lat': 42.455, 'lon': 2.222, 'ele': 2881},
        {'name': "Taga", 'lat': 42.261, 'lon': 2.169, 'ele': 2040},
        {'name': "Costabona", 'lat': 42.400, 'lon': 2.500, 'ele': 2465},
        {'name': "Pic de les Salines", 'lat': 42.420, 'lon': 2.300, 'ele': 2730},
        {'name': "Roc Colom", 'lat': 42.380, 'lon': 2.200, 'ele': 2650},
        {'name': "Tossa de la Reina", 'lat': 42.300, 'lon': 2.250, 'ele': 2530},
        {'name': "Pic de la Fossa del Gegant", 'lat': 42.350, 'lon': 2.180, 'ele': 2780},
        {'name': "Serra de Montgrony", 'lat': 42.250, 'lon': 2.050, 'ele': 2240},
        {'name': "Tuc de la Dona", 'lat': 42.320, 'lon': 2.100, 'ele': 2610}
    ],
    "Garrotxa": [
        {'name': "Comanegra", 'lat': 42.327, 'lon': 2.583, 'ele': 1557},
        {'name': "Puigsacalm", 'lat': 42.129, 'lon': 2.441, 'ele': 1514},
        {'name': "Santa Magdalena", 'lat': 42.200, 'lon': 2.500, 'ele': 1120},
        {'name': "Finestres", 'lat': 42.150, 'lon': 2.550, 'ele': 980},
        {'name': "Tosca", 'lat': 42.180, 'lon': 2.450, 'ele': 1310},
        {'name': "Serra de Bassegoda", 'lat': 42.250, 'lon': 2.650, 'ele': 1373},
        {'name': "Mont", 'lat': 42.100, 'lon': 2.600, 'ele': 1080},
        {'name': "Turo de l'Home", 'lat': 42.120, 'lon': 2.520, 'ele': 1240},
        {'name': "Rocacorba", 'lat': 42.093, 'lon': 2.684, 'ele': 991},
        {'name': "Serra de Sant Julià", 'lat': 42.220, 'lon': 2.580, 'ele': 1050}
    ],
    "Berguedà": [
        {'name': "Pedraforca", 'lat': 42.235, 'lon': 1.713, 'ele': 2506},
        {'name': "Penyes Altes de Moixeró", 'lat': 42.288, 'lon': 1.849, 'ele': 2276},
        {'name': "Rasos de Peguera", 'lat': 42.200, 'lon': 1.900, 'ele': 2050},
        {'name': "Tosa", 'lat': 42.250, 'lon': 1.800, 'ele': 2536},
        {'name': "Serra d'Ensija", 'lat': 42.150, 'lon': 1.750, 'ele': 2320},
        {'name': "Pic de l'Àliga", 'lat': 42.180, 'lon': 1.850, 'ele': 2400},
        {'name': "Tuc de la Bòfia", 'lat': 42.220, 'lon': 1.780, 'ele': 2280},
        {'name': "Serra de Queralt", 'lat': 42.140, 'lon': 1.680, 'ele': 1200},
        {'name': "Tossal de la Trapa", 'lat': 42.200, 'lon': 1.950, 'ele': 1980},
        {'name': "Pic de Comabona", 'lat': 42.300, 'lon': 1.600, 'ele': 2540}
    ],
    "Solsonès": [
        {'name': "Pedró dels Quatre Batlles", 'lat': 42.176, 'lon': 1.503, 'ele': 2383},
        {'name': "Port del Comte", 'lat': 42.180, 'lon': 1.600, 'ele': 2330},
        {'name': "Tossal de la Truita", 'lat': 42.150, 'lon': 1.550, 'ele': 2250},
        {'name': "Serra de Busa", 'lat': 42.100, 'lon': 1.450, 'ele': 1610},
        {'name': "Tuc de la Dona", 'lat': 42.200, 'lon': 1.480, 'ele': 2150},
        {'name': "Roca de Pena", 'lat': 42.120, 'lon': 1.400, 'ele': 1850},
        {'name': "Tossal de la Llosa", 'lat': 42.250, 'lon': 1.450, 'ele': 2200},
        {'name': "Pic de la Muga", 'lat': 42.220, 'lon': 1.520, 'ele': 2100},
        {'name': "Serra de Pinós", 'lat': 41.850, 'lon': 1.550, 'ele': 900},
        {'name': "Tuc de l'Orri", 'lat': 42.180, 'lon': 1.420, 'ele': 1980}
    ],

    # COMARQUES CENTRALS I PRE-LITORALS
    "Osona": [
        {'name': "Matagalls", 'lat': 41.854, 'lon': 2.387, 'ele': 1698},
        {'name': "Bellmunt", 'lat': 42.093, 'lon': 2.274, 'ele': 1246},
        {'name': "Puigsagordi", 'lat': 41.900, 'lon': 2.300, 'ele': 1450},
        {'name': "Turó de l'Home", 'lat': 41.776, 'lon': 2.417, 'ele': 1706},
        {'name': "Les Agudes", 'lat': 41.789, 'lon': 2.434, 'ele': 1706},
        {'name': "Sant Amanç", 'lat': 41.950, 'lon': 2.200, 'ele': 1120},
        {'name': "Serra de Cabrera", 'lat': 41.850, 'lon': 2.350, 'ele': 1320},
        {'name': "Tossa de Montbui", 'lat': 41.900, 'lon': 2.150, 'ele': 980},
        {'name': "Roca Grossa", 'lat': 41.920, 'lon': 2.250, 'ele': 1050},
        {'name': "Puig de la Creu", 'lat': 41.880, 'lon': 2.280, 'ele': 1210}
    ],
    "Moianès": [
        {'name': "Puig de la Caritat", 'lat': 41.834, 'lon': 2.155, 'ele': 1009},
        {'name': "Turó de les Tres Creus", 'lat': 41.820, 'lon': 2.140, 'ele': 950},
        {'name': "Serra de l'Obac", 'lat': 41.850, 'lon': 2.100, 'ele': 890},
        {'name': "Tossal de la Baltasana", 'lat': 41.800, 'lon': 2.180, 'ele': 920},
        {'name': "Roca del Corb", 'lat': 41.830, 'lon': 2.170, 'ele': 870},
        {'name': "Puig de Sant Jordi", 'lat': 41.840, 'lon': 2.160, 'ele': 980},
        {'name': "Tossal Gros", 'lat': 41.810, 'lon': 2.150, 'ele': 840},
        {'name': "Serra de Rubió", 'lat': 41.850, 'lon': 2.050, 'ele': 780},
        {'name': "Turó del Castell", 'lat': 41.820, 'lon': 2.130, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.830, 'lon': 2.140, 'ele': 960}
    ],
    "Bages": [
        {'name': "Montserrat (Sant Jeroni)", 'lat': 41.594, 'lon': 1.837, 'ele': 1236},
        {'name': "La Mola", 'lat': 41.637, 'lon': 2.019, 'ele': 1104},
        {'name': "Sant Llorenç del Munt", 'lat': 41.648, 'lon': 2.020, 'ele': 1095},
        {'name': "Serra de Castelltallat", 'lat': 41.750, 'lon': 1.650, 'ele': 936},
        {'name': "Puig de la Creu", 'lat': 41.700, 'lon': 1.850, 'ele': 890},
        {'name': "Tossal de la Baltasana", 'lat': 41.800, 'lon': 1.750, 'ele': 920},
        {'name': "Roca del Corb", 'lat': 41.680, 'lon': 1.900, 'ele': 950},
        {'name': "Serra de Rubió", 'lat': 41.850, 'lon': 1.550, 'ele': 780},
        {'name': "Turó del Castell", 'lat': 41.620, 'lon': 1.880, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.650, 'lon': 1.950, 'ele': 1050}
    ],
    "Anoia": [
        {'name': "Montserrat", 'lat': 41.594, 'lon': 1.837, 'ele': 1236},
        {'name': "Tossa de Montbui", 'lat': 41.551, 'lon': 1.558, 'ele': 620},
        {'name': "Serra de la Llacuna", 'lat': 41.500, 'lon': 1.600, 'ele': 780},
        {'name': "Puig Agut", 'lat': 41.520, 'lon': 1.580, 'ele': 690},
        {'name': "Tossal Gros", 'lat': 41.530, 'lon': 1.550, 'ele': 650},
        {'name': "Serra de Rubió", 'lat': 41.650, 'lon': 1.550, 'ele': 780},
        {'name': "Turó del Castell", 'lat': 41.550, 'lon': 1.600, 'ele': 710},
        {'name': "Puig de la Mola", 'lat': 41.580, 'lon': 1.620, 'ele': 850},
        {'name': "Roca del Corb", 'lat': 41.560, 'lon': 1.570, 'ele': 730},
        {'name': "Tossal de la Baltasana", 'lat': 41.500, 'lon': 1.520, 'ele': 620}
    ],
    "Segarra": [
        {'name': "Tossal de la Guàrdia", 'lat': 41.748, 'lon': 1.472, 'ele': 876},
        {'name': "Serra de Rubió", 'lat': 41.650, 'lon': 1.550, 'ele': 780},
        {'name': "Tossal Gros", 'lat': 41.700, 'lon': 1.500, 'ele': 820},
        {'name': "Puig de la Creu", 'lat': 41.720, 'lon': 1.450, 'ele': 890},
        {'name': "Roca del Corb", 'lat': 41.680, 'lon': 1.480, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 41.730, 'lon': 1.460, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.710, 'lon': 1.490, 'ele': 950},
        {'name': "Tossal de la Baltasana", 'lat': 41.740, 'lon': 1.510, 'ele': 920},
        {'name': "Serra de Castelltallat", 'lat': 41.750, 'lon': 1.650, 'ele': 936},
        {'name': "Turó de les Tres Creus", 'lat': 41.760, 'lon': 1.470, 'ele': 880}
    ],
    "Urgell": [
        {'name': "Tossal de l'Àliga", 'lat': 41.530, 'lon': 1.071, 'ele': 558},
        {'name': "Serra de Rubió", 'lat': 41.650, 'lon': 1.550, 'ele': 780},
        {'name': "Tossal Gros", 'lat': 41.600, 'lon': 1.100, 'ele': 520},
        {'name': "Puig de la Creu", 'lat': 41.550, 'lon': 1.080, 'ele': 590},
        {'name': "Roca del Corb", 'lat': 41.520, 'lon': 1.090, 'ele': 530},
        {'name': "Turó del Castell", 'lat': 41.540, 'lon': 1.060, 'ele': 510},
        {'name': "Puig de la Mola", 'lat': 41.560, 'lon': 1.070, 'ele': 550},
        {'name': "Tossal de la Baltasana", 'lat': 41.580, 'lon': 1.050, 'ele': 520},
        {'name': "Serra de Castelltallat", 'lat': 41.650, 'lon': 1.200, 'ele': 600},
        {'name': "Turó de les Tres Creus", 'lat': 41.570, 'lon': 1.040, 'ele': 480}
    ],
    "Conca de Barberà": [
        {'name': "Tossal Gros de Miramar", 'lat': 41.312, 'lon': 1.229, 'ele': 867},
        {'name': "Mola d'Estat", 'lat': 41.393, 'lon': 1.018, 'ele': 1126},
        {'name': "Serra de Prades", 'lat': 41.353, 'lon': 1.034, 'ele': 1201},
        {'name': "Tossal de la Baltasana", 'lat': 41.300, 'lon': 1.200, 'ele': 920},
        {'name': "Puig de la Creu", 'lat': 41.320, 'lon': 1.180, 'ele': 890},
        {'name': "Roca del Corb", 'lat': 41.330, 'lon': 1.190, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 41.340, 'lon': 1.170, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.350, 'lon': 1.160, 'ele': 950},
        {'name': "Serra de Rubió", 'lat': 41.400, 'lon': 1.100, 'ele': 780},
        {'name': "Turó de les Tres Creus", 'lat': 41.310, 'lon': 1.210, 'ele': 880}
    ],
    "Alt Penedès": [
        {'name': "El Montmell", 'lat': 41.309, 'lon': 1.458, 'ele': 861},
        {'name': "Serra de l'Ordal", 'lat': 41.400, 'lon': 1.850, 'ele': 450},
        {'name': "Puig Aguilar", 'lat': 41.350, 'lon': 1.600, 'ele': 620},
        {'name': "Tossal Gros", 'lat': 41.320, 'lon': 1.550, 'ele': 520},
        {'name': "Roca del Corb", 'lat': 41.330, 'lon': 1.580, 'ele': 530},
        {'name': "Turó del Castell", 'lat': 41.340, 'lon': 1.570, 'ele': 510},
        {'name': "Puig de la Mola", 'lat': 41.360, 'lon': 1.560, 'ele': 550},
        {'name': "Tossal de la Baltasana", 'lat': 41.380, 'lon': 1.540, 'ele': 520},
        {'name': "Serra de Rubió", 'lat': 41.400, 'lon': 1.500, 'ele': 480},
        {'name': "Turó de les Tres Creus", 'lat': 41.370, 'lon': 1.520, 'ele': 480}
    ],

    # COMARQUES DEL LITORAL I PRE-LITORAL
    "Alt Empordà": [
        {'name': "Puig Neulós", 'lat': 42.493, 'lon': 2.846, 'ele': 1256},
        {'name': "Sant Salvador de Verdera", 'lat': 42.321, 'lon': 3.270, 'ele': 670},
        {'name': "Montperdut", 'lat': 42.450, 'lon': 2.900, 'ele': 1100},
        {'name': "Puig de les Basses", 'lat': 42.400, 'lon': 2.800, 'ele': 980},
        {'name': "Tossal Gros", 'lat': 42.380, 'lon': 2.850, 'ele': 920},
        {'name': "Roca del Corb", 'lat': 42.420, 'lon': 2.820, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 42.350, 'lon': 2.950, 'ele': 710},
        {'name': "Puig de la Mola", 'lat': 42.330, 'lon': 2.980, 'ele': 750},
        {'name': "Serra de l'Albera", 'lat': 42.480, 'lon': 2.950, 'ele': 1150},
        {'name': "Turó de les Tres Creus", 'lat': 42.370, 'lon': 2.920, 'ele': 880}
    ],
    "Gironès": [
        {'name': "Rocacorba", 'lat': 42.093, 'lon': 2.684, 'ele': 991},
        {'name': "Montigalar", 'lat': 42.000, 'lon': 2.800, 'ele': 520},
        {'name': "Puig Aguilar", 'lat': 41.950, 'lon': 2.850, 'ele': 620},
        {'name': "Tossal Gros", 'lat': 41.980, 'lon': 2.750, 'ele': 420},
        {'name': "Roca del Corb", 'lat': 42.020, 'lon': 2.700, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 42.050, 'lon': 2.650, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 42.080, 'lon': 2.600, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 42.100, 'lon': 2.550, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 42.120, 'lon': 2.500, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 42.070, 'lon': 2.680, 'ele': 480}
    ],
    "Pla de l'Estany": [
        {'name': "Puig de Sant Patllari", 'lat': 42.176, 'lon': 2.713, 'ele': 654},
        {'name': "Montigalar", 'lat': 42.150, 'lon': 2.650, 'ele': 520},
        {'name': "Puig Aguilar", 'lat': 42.130, 'lon': 2.600, 'ele': 420},
        {'name': "Tossal Gros", 'lat': 42.140, 'lon': 2.620, 'ele': 380},
        {'name': "Roca del Corb", 'lat': 42.160, 'lon': 2.680, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 42.170, 'lon': 2.700, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 42.180, 'lon': 2.720, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 42.190, 'lon': 2.740, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 42.200, 'lon': 2.760, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 42.210, 'lon': 2.780, 'ele': 480}
    ],
    "Baix Empordà": [
        {'name': "Les Gavarres (Puig d'Arques)", 'lat': 41.905, 'lon': 2.975, 'ele': 535},
        {'name': "Montigalar", 'lat': 41.850, 'lon': 3.000, 'ele': 420},
        {'name': "Puig Aguilar", 'lat': 41.900, 'lon': 3.020, 'ele': 320},
        {'name': "Tossal Gros", 'lat': 41.880, 'lon': 2.950, 'ele': 380},
        {'name': "Roca del Corb", 'lat': 41.870, 'lon': 2.920, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 41.860, 'lon': 2.900, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 41.850, 'lon': 2.880, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 41.840, 'lon': 2.860, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 41.830, 'lon': 2.840, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 41.820, 'lon': 2.820, 'ele': 480}
    ],
    "Selva": [
        {'name': "Turó de l'Home", 'lat': 41.776, 'lon': 2.417, 'ele': 1706},
        {'name': "Les Agudes", 'lat': 41.789, 'lon': 2.434, 'ele': 1706},
        {'name': "Puig de Ses Cadiretes", 'lat': 41.766, 'lon': 2.912, 'ele': 519},
        {'name': "Montseny", 'lat': 41.780, 'lon': 2.440, 'ele': 1712},
        {'name': "Matagalls", 'lat': 41.800, 'lon': 2.380, 'ele': 1698},
        {'name': "Puigsagordi", 'lat': 41.820, 'lon': 2.350, 'ele': 1450},
        {'name': "Sant Amanç", 'lat': 41.850, 'lon': 2.300, 'ele': 1120},
        {'name': "Serra de Cabrera", 'lat': 41.830, 'lon': 2.320, 'ele': 1320},
        {'name': "Tossa de Montbui", 'lat': 41.840, 'lon': 2.280, 'ele': 980},
        {'name': "Roca Grossa", 'lat': 41.860, 'lon': 2.340, 'ele': 1050}
    ],
    "Vallès Oriental": [
        {'name': "Turó de l'Home", 'lat': 41.776, 'lon': 2.417, 'ele': 1706},
        {'name': "Les Agudes", 'lat': 41.789, 'lon': 2.434, 'ele': 1706},
        {'name': "Matagalls", 'lat': 41.854, 'lon': 2.387, 'ele': 1698},
        {'name': "Puigsagordi", 'lat': 41.900, 'lon': 2.300, 'ele': 1450},
        {'name': "Sant Amanç", 'lat': 41.950, 'lon': 2.200, 'ele': 1120},
        {'name': "Serra de Cabrera", 'lat': 41.850, 'lon': 2.350, 'ele': 1320},
        {'name': "Tossa de Montbui", 'lat': 41.900, 'lon': 2.150, 'ele': 980},
        {'name': "Roca Grossa", 'lat': 41.920, 'lon': 2.250, 'ele': 1050},
        {'name': "Puig de la Creu", 'lat': 41.880, 'lon': 2.280, 'ele': 1210},
        {'name': "Montseny", 'lat': 41.780, 'lon': 2.440, 'ele': 1712}
    ],
    "Maresme": [
        {'name': "Turó d'en Vives (Montnegre)", 'lat': 41.679, 'lon': 2.584, 'ele': 760},
        {'name': "Turó del Castell", 'lat': 41.650, 'lon': 2.600, 'ele': 480},
        {'name': "Puig de la Mola", 'lat': 41.620, 'lon': 2.550, 'ele': 420},
        {'name': "Tossal Gros", 'lat': 41.630, 'lon': 2.520, 'ele': 380},
        {'name': "Roca del Corb", 'lat': 41.640, 'lon': 2.500, 'ele': 350},
        {'name': "Turó de les Tres Creus", 'lat': 41.660, 'lon': 2.580, 'ele': 520},
        {'name': "Puig Aguilar", 'lat': 41.670, 'lon': 2.560, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 41.680, 'lon': 2.540, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 41.690, 'lon': 2.520, 'ele': 280},
        {'name': "Montigalar", 'lat': 41.700, 'lon': 2.500, 'ele': 220}
    ],
    "Vallès Occidental": [
        {'name': "La Mola", 'lat': 41.637, 'lon': 2.019, 'ele': 1104},
        {'name': "Tibidabo", 'lat': 41.422, 'lon': 2.119, 'ele': 512},
        {'name': "Sant Llorenç del Munt", 'lat': 41.648, 'lon': 2.020, 'ele': 1095},
        {'name': "Montserrat", 'lat': 41.594, 'lon': 1.837, 'ele': 1236},
        {'name': "Puig de la Creu", 'lat': 41.600, 'lon': 2.100, 'ele': 890},
        {'name': "Roca del Corb", 'lat': 41.620, 'lon': 2.080, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 41.630, 'lon': 2.060, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.640, 'lon': 2.040, 'ele': 1050},
        {'name': "Tossal de la Baltasana", 'lat': 41.650, 'lon': 2.020, 'ele': 920},
        {'name': "Serra de Rubió", 'lat': 41.660, 'lon': 2.000, 'ele': 780}
    ],
    "Barcelonès": [
        {'name': "Tibidabo", 'lat': 41.422, 'lon': 2.119, 'ele': 512},
        {'name': "Turó de la Rovira", 'lat': 41.425, 'lon': 2.168, 'ele': 262},
        {'name': "Turó del Carmel", 'lat': 41.423, 'lon': 2.157, 'ele': 267},
        {'name': "Montjuïc", 'lat': 41.363, 'lon': 2.164, 'ele': 173},
        {'name': "Turó de la Peira", 'lat': 41.438, 'lon': 2.175, 'ele': 133},
        {'name': "Turó de les Tres Creus", 'lat': 41.430, 'lon': 2.140, 'ele': 280},
        {'name': "Turó del Putxet", 'lat': 41.410, 'lon': 2.143, 'ele': 177},
        {'name': "Turó de la Vilana", 'lat': 41.400, 'lon': 2.150, 'ele': 160},
        {'name': "Turó d'en Cors", 'lat': 41.415, 'lon': 2.130, 'ele': 190},
        {'name': "Turó de Monterols", 'lat': 41.408, 'lon': 2.135, 'ele': 121}
    ],
    "Baix Llobregat": [
        {'name': "Montserrat", 'lat': 41.594, 'lon': 1.837, 'ele': 1236},
        {'name': "Puig de les Agulles", 'lat': 41.346, 'lon': 1.810, 'ele': 553},
        {'name': "Sant Jeroni", 'lat': 41.594, 'lon': 1.837, 'ele': 1236},
        {'name': "La Mola", 'lat': 41.637, 'lon': 2.019, 'ele': 1104},
        {'name': "Tibidabo", 'lat': 41.422, 'lon': 2.119, 'ele': 512},
        {'name': "Puig de la Creu", 'lat': 41.380, 'lon': 1.900, 'ele': 590},
        {'name': "Roca del Corb", 'lat': 41.370, 'lon': 1.880, 'ele': 530},
        {'name': "Turó del Castell", 'lat': 41.360, 'lon': 1.860, 'ele': 510},
        {'name': "Puig de la Mola", 'lat': 41.350, 'lon': 1.840, 'ele': 550},
        {'name': "Tossal de la Baltasana", 'lat': 41.340, 'lon': 1.820, 'ele': 520}
    ],
    "Garraf": [
        {'name': "La Morella", 'lat': 41.272, 'lon': 1.839, 'ele': 594},
        {'name': "Puig Aguilar", 'lat': 41.300, 'lon': 1.800, 'ele': 420},
        {'name': "Tossal Gros", 'lat': 41.280, 'lon': 1.820, 'ele': 380},
        {'name': "Roca del Corb", 'lat': 41.290, 'lon': 1.830, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 41.270, 'lon': 1.850, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 41.260, 'lon': 1.860, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 41.250, 'lon': 1.870, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 41.240, 'lon': 1.880, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 41.230, 'lon': 1.890, 'ele': 480},
        {'name': "Montigalar", 'lat': 41.220, 'lon': 1.900, 'ele': 220}
    ],
    "Baix Penedès": [
        {'name': "Montmell-Marmellar", 'lat': 41.309, 'lon': 1.458, 'ele': 861},
        {'name': "Puig Aguilar", 'lat': 41.280, 'lon': 1.500, 'ele': 620},
        {'name': "Tossal Gros", 'lat': 41.290, 'lon': 1.520, 'ele': 520},
        {'name': "Roca del Corb", 'lat': 41.300, 'lon': 1.540, 'ele': 530},
        {'name': "Turó del Castell", 'lat': 41.310, 'lon': 1.560, 'ele': 510},
        {'name': "Puig de la Mola", 'lat': 41.320, 'lon': 1.580, 'ele': 550},
        {'name': "Tossal de la Baltasana", 'lat': 41.330, 'lon': 1.600, 'ele': 520},
        {'name': "Serra de Rubió", 'lat': 41.340, 'lon': 1.620, 'ele': 480},
        {'name': "Turó de les Tres Creus", 'lat': 41.350, 'lon': 1.640, 'ele': 480},
        {'name': "Montigalar", 'lat': 41.360, 'lon': 1.660, 'ele': 420}
    ],
    "Tarragonès": [
        {'name': "La Mola (Bonastre)", 'lat': 41.233, 'lon': 1.439, 'ele': 317},
        {'name': "Puig Aguilar", 'lat': 41.200, 'lon': 1.400, 'ele': 220},
        {'name': "Tossal Gros", 'lat': 41.210, 'lon': 1.420, 'ele': 180},
        {'name': "Roca del Corb", 'lat': 41.220, 'lon': 1.430, 'ele': 150},
        {'name': "Turó del Castell", 'lat': 41.230, 'lon': 1.440, 'ele': 110},
        {'name': "Puig de la Mola", 'lat': 41.240, 'lon': 1.450, 'ele': 250},
        {'name': "Tossal de la Baltasana", 'lat': 41.250, 'lon': 1.460, 'ele': 120},
        {'name': "Serra de les Medes", 'lat': 41.260, 'lon': 1.470, 'ele': 80},
        {'name': "Turó de les Tres Creus", 'lat': 41.270, 'lon': 1.480, 'ele': 280},
        {'name': "Montigalar", 'lat': 41.280, 'lon': 1.490, 'ele': 200}
    ],
    "Alt Camp": [
        {'name': "Tossal Gros de Miramar", 'lat': 41.312, 'lon': 1.229, 'ele': 867},
        {'name': "Puig Aguilar", 'lat': 41.300, 'lon': 1.250, 'ele': 620},
        {'name': "Tossal Gros", 'lat': 41.290, 'lon': 1.270, 'ele': 520},
        {'name': "Roca del Corb", 'lat': 41.280, 'lon': 1.290, 'ele': 530},
        {'name': "Turó del Castell", 'lat': 41.270, 'lon': 1.310, 'ele': 510},
        {'name': "Puig de la Mola", 'lat': 41.260, 'lon': 1.330, 'ele': 550},
        {'name': "Tossal de la Baltasana", 'lat': 41.250, 'lon': 1.350, 'ele': 520},
        {'name': "Serra de Rubió", 'lat': 41.240, 'lon': 1.370, 'ele': 480},
        {'name': "Turó de les Tres Creus", 'lat': 41.230, 'lon': 1.390, 'ele': 480},
        {'name': "Montigalar", 'lat': 41.220, 'lon': 1.410, 'ele': 420}
    ],
    "Baix Camp": [
        {'name': "Muntanyes de Prades (Tossal de la Baltasana)", 'lat': 41.353, 'lon': 1.034, 'ele': 1201},
        {'name': "La Mola de Colldejou", 'lat': 41.094, 'lon': 0.838, 'ele': 921},
        {'name': "Puig de la Mola", 'lat': 41.300, 'lon': 1.100, 'ele': 950},
        {'name': "Tossal Gros", 'lat': 41.320, 'lon': 1.080, 'ele': 820},
        {'name': "Roca del Corb", 'lat': 41.340, 'lon': 1.060, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 41.360, 'lon': 1.040, 'ele': 910},
        {'name': "Puig Aguilar", 'lat': 41.380, 'lon': 1.020, 'ele': 620},
        {'name': "Tossal de la Baltasana", 'lat': 41.400, 'lon': 1.000, 'ele': 920},
        {'name': "Serra de Rubió", 'lat': 41.420, 'lon': 0.980, 'ele': 780},
        {'name': "Turó de les Tres Creus", 'lat': 41.440, 'lon': 0.960, 'ele': 880}
    ],

    # PLANA DE LLEIDA I TERRES DE L'EBRE
    "Noguera": [
        {'name': "Montsec de Rúbies", 'lat': 42.053, 'lon': 1.043, 'ele': 1677},
        {'name': "Tossal Gros", 'lat': 41.900, 'lon': 1.200, 'ele': 820},
        {'name': "Roca del Corb", 'lat': 41.880, 'lon': 1.180, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 41.860, 'lon': 1.160, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.840, 'lon': 1.140, 'ele': 950},
        {'name': "Tossal de la Baltasana", 'lat': 41.820, 'lon': 1.120, 'ele': 920},
        {'name': "Serra de Rubió", 'lat': 41.800, 'lon': 1.100, 'ele': 780},
        {'name': "Turó de les Tres Creus", 'lat': 41.780, 'lon': 1.080, 'ele': 880},
        {'name': "Montigalar", 'lat': 41.760, 'lon': 1.060, 'ele': 420},
        {'name': "Puig Aguilar", 'lat': 41.740, 'lon': 1.040, 'ele': 620}
    ],
    "Segrià": [
        {'name': "Puntals dels Marquesos", 'lat': 41.455, 'lon': 0.697, 'ele': 434},
        {'name': "Tossal Gros", 'lat': 41.500, 'lon': 0.800, 'ele': 320},
        {'name': "Roca del Corb", 'lat': 41.480, 'lon': 0.750, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 41.460, 'lon': 0.700, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 41.440, 'lon': 0.650, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 41.420, 'lon': 0.600, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 41.400, 'lon': 0.550, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 41.380, 'lon': 0.500, 'ele': 480},
        {'name': "Montigalar", 'lat': 41.360, 'lon': 0.450, 'ele': 220},
        {'name': "Puig Aguilar", 'lat': 41.340, 'lon': 0.400, 'ele': 420}
    ],
    "Pla d'Urgell": [
        {'name': "Tossal Gros", 'lat': 41.600, 'lon': 0.900, 'ele': 320},
        {'name': "Roca del Corb", 'lat': 41.580, 'lon': 0.880, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 41.560, 'lon': 0.860, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 41.540, 'lon': 0.840, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 41.520, 'lon': 0.820, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 41.500, 'lon': 0.800, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 41.480, 'lon': 0.780, 'ele': 480},
        {'name': "Montigalar", 'lat': 41.460, 'lon': 0.760, 'ele': 220},
        {'name': "Puig Aguilar", 'lat': 41.440, 'lon': 0.740, 'ele': 420},
        {'name': "Tossal de la Guàrdia", 'lat': 41.420, 'lon': 0.720, 'ele': 320}
    ],
    "Garrigues": [
        {'name': "Serra de la Llena (Punta del Curull)", 'lat': 41.365, 'lon': 0.902, 'ele': 1022},
        {'name': "Tossal Gros", 'lat': 41.400, 'lon': 0.850, 'ele': 820},
        {'name': "Roca del Corb", 'lat': 41.380, 'lon': 0.800, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 41.360, 'lon': 0.750, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.340, 'lon': 0.700, 'ele': 950},
        {'name': "Tossal de la Baltasana", 'lat': 41.320, 'lon': 0.650, 'ele': 920},
        {'name': "Serra de Rubió", 'lat': 41.300, 'lon': 0.600, 'ele': 780},
        {'name': "Turó de les Tres Creus", 'lat': 41.280, 'lon': 0.550, 'ele': 880},
        {'name': "Montigalar", 'lat': 41.260, 'lon': 0.500, 'ele': 420},
        {'name': "Puig Aguilar", 'lat': 41.240, 'lon': 0.450, 'ele': 620}
    ],
    "Priorat": [
        {'name': "Roca Corbatera (Montsant)", 'lat': 41.294, 'lon': 0.828, 'ele': 1163},
        {'name': "Tossal Gros", 'lat': 41.300, 'lon': 0.800, 'ele': 820},
        {'name': "Roca del Corb", 'lat': 41.280, 'lon': 0.750, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 41.260, 'lon': 0.700, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 41.240, 'lon': 0.650, 'ele': 950},
        {'name': "Tossal de la Baltasana", 'lat': 41.220, 'lon': 0.600, 'ele': 920},
        {'name': "Serra de Rubió", 'lat': 41.200, 'lon': 0.550, 'ele': 780},
        {'name': "Turó de les Tres Creus", 'lat': 41.180, 'lon': 0.500, 'ele': 880},
        {'name': "Montigalar", 'lat': 41.160, 'lon': 0.450, 'ele': 420},
        {'name': "Puig Aguilar", 'lat': 41.140, 'lon': 0.400, 'ele': 620}
    ],
    "Ribera d'Ebre": [
        {'name': "La Figuera (Serra Major)", 'lat': 41.218, 'lon': 0.729, 'ele': 614},
        {'name': "Tossal Gros", 'lat': 41.200, 'lon': 0.700, 'ele': 520},
        {'name': "Roca del Corb", 'lat': 41.180, 'lon': 0.650, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 41.160, 'lon': 0.600, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 41.140, 'lon': 0.550, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 41.120, 'lon': 0.500, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 41.100, 'lon': 0.450, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 41.080, 'lon': 0.400, 'ele': 480},
        {'name': "Montigalar", 'lat': 41.060, 'lon': 0.350, 'ele': 220},
        {'name': "Puig Aguilar", 'lat': 41.040, 'lon': 0.300, 'ele': 420}
    ],
    "Terra Alta": [
        {'name': "Puig de l'Àliga (Cavalls)", 'lat': 41.066, 'lon': 0.548, 'ele': 714},
        {'name': "Tossal Gros", 'lat': 41.080, 'lon': 0.500, 'ele': 520},
        {'name': "Roca del Corb", 'lat': 41.060, 'lon': 0.450, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 41.040, 'lon': 0.400, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 41.020, 'lon': 0.350, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 41.000, 'lon': 0.300, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 40.980, 'lon': 0.250, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 40.960, 'lon': 0.200, 'ele': 480},
        {'name': "Montigalar", 'lat': 40.940, 'lon': 0.150, 'ele': 220},
        {'name': "Puig Aguilar", 'lat': 40.920, 'lon': 0.100, 'ele': 420}
    ],
    "Montsià": [
        {'name': "La Foradada", 'lat': 40.671, 'lon': 0.505, 'ele': 698},
        {'name': "Tossal Gros", 'lat': 40.700, 'lon': 0.450, 'ele': 520},
        {'name': "Roca del Corb", 'lat': 40.680, 'lon': 0.400, 'ele': 350},
        {'name': "Turó del Castell", 'lat': 40.660, 'lon': 0.350, 'ele': 310},
        {'name': "Puig de la Mola", 'lat': 40.640, 'lon': 0.300, 'ele': 450},
        {'name': "Tossal de la Baltasana", 'lat': 40.620, 'lon': 0.250, 'ele': 320},
        {'name': "Serra de les Medes", 'lat': 40.600, 'lon': 0.200, 'ele': 280},
        {'name': "Turó de les Tres Creus", 'lat': 40.580, 'lon': 0.150, 'ele': 480},
        {'name': "Montigalar", 'lat': 40.560, 'lon': 0.100, 'ele': 220},
        {'name': "Puig Aguilar", 'lat': 40.540, 'lon': 0.050, 'ele': 420}
    ],
    "Baix Ebre": [
        {'name': "Mont Caro", 'lat': 40.793, 'lon': 0.344, 'ele': 1441},
        {'name': "Tossal Gros", 'lat': 40.800, 'lon': 0.300, 'ele': 820},
        {'name': "Roca del Corb", 'lat': 40.780, 'lon': 0.250, 'ele': 850},
        {'name': "Turó del Castell", 'lat': 40.760, 'lon': 0.200, 'ele': 910},
        {'name': "Puig de la Mola", 'lat': 40.740, 'lon': 0.150, 'ele': 950},
        {'name': "Tossal de la Baltasana", 'lat': 40.720, 'lon': 0.100, 'ele': 920},
        {'name': "Serra de Rubió", 'lat': 40.700, 'lon': 0.050, 'ele': 780},
        {'name': "Turó de les Tres Creus", 'lat': 40.680, 'lon': 0.000, 'ele': 880},
        {'name': "Montigalar", 'lat': 40.660, 'lon': -0.050, 'ele': 420},
        {'name': "Puig Aguilar", 'lat': 40.640, 'lon': -0.100, 'ele': 620}
    ]
}


@st.cache_data
def convertir_img_a_base64(ruta_arxiu_relativa):
    """
    Versió Robusta v2.0. Converteix una imatge a Base64 utilitzant una ruta absoluta
    per a garantir que sempre trobi l'arxiu, independentment d'on s'executi l'script.
    """
    try:
        # Aquesta és la part clau: construeix la ruta completa a l'arxiu
        directori_actual = os.path.dirname(__file__)
        ruta_absoluta = os.path.join(directori_actual, ruta_arxiu_relativa)

        with open(ruta_absoluta, "rb") as f:
            contingut = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{contingut}"
    except FileNotFoundError:
        # Si falla, intenta trobar el 'fallback' de la mateixa manera robusta
        try:
            ruta_fallback = os.path.join(directori_actual, "imatges_reals/fallback.jpg")
            with open(ruta_fallback, "rb") as f:
                contingut = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{contingut}"
        except FileNotFoundError:
            # Si ni tan sols el fallback existeix, imprimeix un error útil a la consola
            print(f"ALERTA: No s'ha trobat l'arxiu d'imatge a '{ruta_absoluta}' ni el fallback.")
            return ""
    except Exception as e:
        print(f"S'ha produït un error inesperat en carregar la imatge: {e}")
        return ""
        



MAP_CONFIG = {
    # <<<--- CANVI PRINCIPAL AQUÍ: Nova paleta de colors professional per al CAPE ---
    'cape': {
        'colors': [
            '#006400', '#228B22', '#55AE3A', '#7DC83A', '#A6D839', '#D0E738',
            '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF6347', '#FF0000',
            '#DC143C', '#C71585', '#FF1493', '#FF00FF', '#DA70D6', '#EE82EE',
            '#DA70D6', '#D8BFD8', '#E6E6FA', '#FFF0F5'
        ],
        'levels': [0, 20, 40, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800, 3200, 3600, 4000, 4500, 5001],
        'cbar_ticks': [0, 400, 1000, 1600, 2400, 3200, 4000, 5000],
        'alpha': 0.75
    },
    # --- FI DEL CANVI ---
    'convergence': {
        'styles': {
            'Comuna':     {'levels': [10, 20, 30, 40], 'color': '#00FF00', 'width': 1.2},
            'Interessant':{'levels': [50, 60, 70],     'color': '#FFFF00', 'width': 1.6},
            'Molt Forta': {'levels': [80, 90, 100, 110], 'color': '#FF0000', 'width': 2.0},
            'Extrema':    {'levels': [120, 130, 140, 150], 'color': '#FF00FF', 'width': 2.4}
        },
        'sigma_filter': 2.5
    },
    'streamlines': {
        'color': 'black',
        'linewidth': 0.4,
        'density': 5.5,
        'arrowsize': 0.4
    },
    'thresholds': {
        'cape_min': 500,
        'cape_max': 6000,
        'convergence_min': 10,
        'dewpoint_low_level': 14,
        'dewpoint_mid_level': 12,
    }
}




            




    

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







# --- Constants per a l'Est de la Península Ibèrica (VERSIÓ ADAPTADA A GEOJSON DE PROVÍNCIES) ---
API_URL_EST_PENINSULA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_EST_PENINSULA = pytz.timezone('Europe/Madrid')
PRESS_LEVELS_EST_PENINSULA = sorted([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
MAP_EXTENT_EST_PENINSULA = [-4, 1, 38.5, 43.5]

CIUTATS_EST_PENINSULA = {
    'Pamplona': {'lat': 42.8125, 'lon': -1.6458, 'sea_dir': None},
    'Logroño': {'lat': 42.465, 'lon': -2.441, 'sea_dir': None},
    'Soria': {'lat': 41.7636, 'lon': -2.4676, 'sea_dir': None},
    'Zaragoza': {'lat': 41.6488, 'lon': -0.8891, 'sea_dir': None},
    'Teruel': {'lat': 40.3456, 'lon': -1.1065, 'sea_dir': None},
    'Castelló': {'lat': 39.9864, 'lon': -0.0513, 'sea_dir': (60, 180)},
    'València': {'lat': 39.4699, 'lon': -0.3763, 'sea_dir': (45, 180)},
    'Cuenca': {'lat': 40.0704, 'lon': -2.1374, 'sea_dir': None},
    'Albacete': {'lat': 38.9942, 'lon': -1.8584, 'sea_dir': None},
    # --- NOVA LÍNIA AFEGIDA ---
    'El Pobo': {'lat': 40.5408, 'lon': -0.9272, 'sea_dir': None},
}

CIUTATS_PER_ZONA_PENINSULA = {
    "Zaragoza": { 'Zaragoza': CIUTATS_EST_PENINSULA['Zaragoza'] },
    "Teruel": { 
        'Teruel': CIUTATS_EST_PENINSULA['Teruel'],
        # --- NOVA LÍNIA AFEGIDA ---
        'El Pobo': CIUTATS_EST_PENINSULA['El Pobo']
    },
    "Castellón": { 'Castelló': CIUTATS_EST_PENINSULA['Castelló'] },
    "Valencia": { 'València': CIUTATS_EST_PENINSULA['València'] },
    "Navarra": { 'Pamplona': CIUTATS_EST_PENINSULA['Pamplona'] },
    "La Rioja": { 'Logroño': CIUTATS_EST_PENINSULA['Logroño'] },
    "Soria": { 'Soria': CIUTATS_EST_PENINSULA['Soria'] },
    "Cuenca": { 'Cuenca': CIUTATS_EST_PENINSULA['Cuenca'] },
    "Albacete": { 'Albacete': CIUTATS_EST_PENINSULA['Albacete'] }
}

# Coordenades per a les etiquetes de text a cada PROVÍNCIA
CAPITALS_ZONA_PENINSULA = {
    "Zaragoza": {"nom": "Zaragoza", "lat": 41.6488, "lon": -0.8891},
    "Teruel": {"nom": "Teruel", "lat": 40.3456, "lon": -1.1065},
    "Castellón": {"nom": "Castelló", "lat": 39.9864, "lon": -0.0513},
    "Valencia": {"nom": "València", "lat": 39.4699, "lon": -0.3763},
    "Navarra": {"nom": "Pamplona", "lat": 42.8125, "lon": -1.6458},
    "La Rioja": {"nom": "Logroño", "lat": 42.465, "lon": -2.441},
    "Soria": {"nom": "Soria", "lat": 41.7636, "lon": -2.4676},
    "Cuenca": {"nom": "Cuenca", "lat": 40.0704, "lon": -2.1374},
    "Albacete": {"nom": "Albacete", "lat": 38.9942, "lon": -1.8584}
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




# --- BLOC ÚNIC I DEFINITIU DE DADES GEOGRÀFIQUES DE CATALUNYA ---
API_URL_CAT = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_CAT = pytz.timezone('Europe/Madrid')
PRESS_LEVELS_AROME = sorted([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], reverse=True)
MAP_EXTENT_CAT = [0, 3.5, 40.4, 43]

MAP_ZOOM_LEVELS_CAT = {
    'Catalunya (Complet)': MAP_EXTENT_CAT, 'Barcelona': [1.8, 2.6, 41.25, 41.65],
    'Girona': [2.5, 3.4, 41.8, 42.2], 'Lleida': [0.3, 0.95, 41.5, 41.75], 'Tarragona': [0.9, 1.35, 40.95, 41.3]
}

CIUTATS_PER_COMARCA = {
    "Alt Camp": { 'Valls': {'lat': 41.2872, 'lon': 1.2505}, 'Alcover': {'lat': 41.2642, 'lon': 1.1712}, 'El Pla de Santa Maria': {'lat': 41.3831, 'lon': 1.3000} },
    "Alt Empordà": { 'Figueres': {'lat': 42.2662, 'lon': 2.9622}, 'Roses': {'lat': 42.2619, 'lon': 3.1764}, 'La Jonquera': {'lat': 42.4194, 'lon': 2.8752}, 'Llançà': {'lat': 42.3625, 'lon': 3.1539}, 'Cadaqués': {'lat': 42.2888, 'lon': 3.2770}, 'Castelló d\'Empúries': {'lat': 42.2582, 'lon': 3.0725}, 'L\'Escala': {'lat': 42.1235, 'lon': 3.1311} },
    "Alt Penedès": { 'Vilafranca del Penedès': {'lat': 41.3453, 'lon': 1.6995}, 'Sant Sadurní d\'Anoia': {'lat': 41.4287, 'lon': 1.7850}, 'Gelida': {'lat': 41.4392, 'lon': 1.8624} },
    "Alt Urgell": { 'La Seu d\'Urgell': {'lat': 42.3582, 'lon': 1.4593}, 'Oliana': {'lat': 42.0664, 'lon': 1.3142}, 'Coll de Nargó': {'lat': 42.1751, 'lon': 1.3197} },
    "Anoia": { 'Igualada': {'lat': 41.5791, 'lon': 1.6174}, 'Calaf': {'lat': 41.7311, 'lon': 1.5126}, 'Capellades': {'lat': 41.5312, 'lon': 1.6874} },
    "Bages": { 'Manresa': {'lat': 41.7230, 'lon': 1.8268}, 'Cardona': {'lat': 41.9138, 'lon': 1.6806}, 'Súria': {'lat': 41.8322, 'lon': 1.7483} },
    "Baix Camp": { 'Reus': {'lat': 41.1550, 'lon': 1.1075}, 'Cambrils': {'lat': 41.0667, 'lon': 1.0500}, 'Mont-roig del Camp': {'lat': 41.0877, 'lon': 0.9610}, 'La Selva del Camp': {'lat': 41.2131, 'lon': 1.1384} },
    "Baix Ebre": { 'Tortosa': {'lat': 40.8126, 'lon': 0.5211}, 'L\'Ametlla de Mar': {'lat': 40.8824, 'lon': 0.8016}, 'Deltebre': {'lat': 40.7188, 'lon': 0.7099} },
    "Baix Empordà": { 'La Bisbal d\'Empordà': {'lat': 41.9602, 'lon': 3.0378}, 'Palamós': {'lat': 41.8465, 'lon': 3.1287}, 'Sant Feliu de Guíxols': {'lat': 41.7801, 'lon': 3.0278}, 'Platja d\'Aro': {'lat': 41.8175, 'lon': 3.0645}, 'Begur': {'lat': 41.9542, 'lon': 3.2076} },
    "Baix Llobregat": { 'Sant Feliu de Llobregat': {'lat': 41.3833, 'lon': 2.0500}, 'Castelldefels': {'lat': 41.2806, 'lon': 1.9750}, 'Viladecans': {'lat': 41.3155, 'lon': 2.0194}, 'Olesa de Montserrat': {'lat': 41.5451, 'lon': 1.8955} },
    "Baix Penedès": { 'El Vendrell': {'lat': 41.2195, 'lon': 1.5350}, 'Calafell': {'lat': 41.1994, 'lon': 1.5701}, 'Cunit': {'lat': 41.1982, 'lon': 1.6358} },
    "Barcelonès": { 'Barcelona': {'lat': 41.3851, 'lon': 2.1734}, 'Badalona': {'lat': 41.4503, 'lon': 2.2472}, 'Santa Coloma de Gramenet': {'lat': 41.4550, 'lon': 2.2111} },
    "Berguedà": { 'Berga': {'lat': 42.1051, 'lon': 1.8458}, 'Puig-reig': {'lat': 41.9754, 'lon': 1.8814}, 'Gironella': {'lat': 42.0368, 'lon': 1.8821} },
    "Cerdanya": { 'Puigcerdà': {'lat': 42.4331, 'lon': 1.9287}, 'Bellver de Cerdanya': {'lat': 42.3705, 'lon': 1.7770}, 'La Molina': {'lat': 42.3361, 'lon': 1.9463} },
    "Conca de Barberà": { 'Montblanc': {'lat': 41.3761, 'lon': 1.1610}, 'L\'Espluga de Francolí': {'lat': 41.3969, 'lon': 1.1039}, 'Santa Coloma de Queralt': {'lat': 41.5361, 'lon': 1.3855} },
    "Garraf": { 'Vilanova i la Geltrú': {'lat': 41.2241, 'lon': 1.7252}, 'Sitges': {'lat': 41.2351, 'lon': 1.8117}, 'Sant Pere de Ribes': {'lat': 41.2599, 'lon': 1.7725} },
    "Garrigues": { 'Les Borges Blanques': {'lat': 41.5224, 'lon': 0.8674}, 'Juneda': {'lat': 41.5515, 'lon': 0.8242}, 'Arbeca': {'lat': 41.5434, 'lon': 0.9234} },
    "Garrotxa": { 'Olot': {'lat': 42.1818, 'lon': 2.4900}, 'Besalú': {'lat': 42.2007, 'lon': 2.7001}, 'Santa Pau': {'lat': 42.1448, 'lon': 2.5695} },
    "Gironès": { 'Girona': {'lat': 41.9831, 'lon': 2.8249}, 'Cassà de la Selva': {'lat': 41.8893, 'lon': 2.8736}, 'Llagostera': {'lat': 41.8291, 'lon': 2.8931}, 'Riudellots de la Selva': {'lat': 41.9080, 'lon': 2.8099} },
    "Maresme": { 'Mataró': {'lat': 41.5388, 'lon': 2.4449}, 'Calella': {'lat': 41.6146, 'lon': 2.6653}, 'Arenys de Mar': {'lat': 41.5815, 'lon': 2.5504}, 'Vilassar de Mar': {'lat': 41.5057, 'lon': 2.3920} },
    "Moianès": { 'Moià': {'lat': 41.8105, 'lon': 2.0967}, 'Castellterçol': {'lat': 41.7533, 'lon': 2.1209}, 'L\'Estany': {'lat': 41.8653, 'lon': 2.1130} },
    "Montsià": { 'Amposta': {'lat': 40.7093, 'lon': 0.5810}, 'La Ràpita': {'lat': 40.6179, 'lon': 0.5905}, 'Alcanar': {'lat': 40.5434, 'lon': 0.4820} },
    "Noguera": { 'Balaguer': {'lat': 41.7904, 'lon': 0.8066}, 'Artesa de Segre': {'lat': 41.8950, 'lon': 1.0483}, 'Ponts': {'lat': 41.9167, 'lon': 1.1833} },
    "Osona": { 'Vic': {'lat': 41.9301, 'lon': 2.2545}, 'Manlleu': {'lat': 42.0016, 'lon': 2.2844}, 'Torelló': {'lat': 42.0494, 'lon': 2.2619} },
    "Pallars Jussà": { 'Tremp': {'lat': 42.1664, 'lon': 0.8953}, 'La Pobla de Segur': {'lat': 42.2472, 'lon': 0.9678}, 'Isona': {'lat': 42.1187, 'lon': 1.0560} },
    "Pallars Sobirà": { 'Sort': {'lat': 42.4131, 'lon': 1.1278}, 'Esterri d\'Àneu': {'lat': 42.6322, 'lon': 1.1219}, 'Llavorsí': {'lat': 42.4930, 'lon': 1.2201} },
    "Pla de l'Estany": { 'Banyoles': {'lat': 42.1197, 'lon': 2.7667}, 'Porqueres': {'lat': 42.1283, 'lon': 2.7501}, 'Cornellà del Terri': {'lat': 42.0833, 'lon': 2.8167} },
    "Pla d_Urgell": { 'Mollerussa': {'lat': 41.6315, 'lon': 0.8931}, 'Bellvís': {'lat': 41.6934, 'lon': 0.8716}, 'Linyola': {'lat': 41.7135, 'lon': 0.9080} },
    "Priorat": { 'Falset': {'lat': 41.1444, 'lon': 0.8208}, 'Cornudella de Montsant': {'lat': 41.2651, 'lon': 0.9056}, 'Porrera': {'lat': 41.1891, 'lon': 0.8540} },
    "Ribera d_Ebre": { 'Móra d\'Ebre': {'lat': 41.0945, 'lon': 0.6450}, 'Flix': {'lat': 41.2307, 'lon': 0.5501}, 'Ascó': {'lat': 41.2037, 'lon': 0.5699} },
    "Ripollès": { 'Ripoll': {'lat': 42.2013, 'lon': 2.1903}, 'Camprodon': {'lat': 42.3134, 'lon': 2.3644}, 'Sant Joan de les Abadesses': {'lat': 42.2355, 'lon': 2.2858} },
    "Segarra": { 'Cervera': {'lat': 41.6709, 'lon': 1.2721}, 'Guissona': {'lat': 41.7824, 'lon': 1.2905}, 'Torà': {'lat': 41.8124, 'lon': 1.4024} },
    "Segrià": { 'Lleida': {'lat': 41.6177, 'lon': 0.6200}, 'Alcarràs': {'lat': 41.5606, 'lon': 0.5251}, 'Aitona': {'lat': 41.4883, 'lon': 0.4578} },
    "Selva": { 'Santa Coloma de Farners': {'lat': 41.8596, 'lon': 2.6703}, 'Blanes': {'lat': 41.6748, 'lon': 2.7917}, 'Lloret de Mar': {'lat': 41.7005, 'lon': 2.8450}, 'Hostalric': {'lat': 41.7479, 'lon': 2.6360}, 'Arbúcies': {'lat': 41.8167, 'lon': 2.5167} },
    "Solsonès": { 'Solsona': {'lat': 41.9942, 'lon': 1.5161}, 'Sant Llorenç de Morunys': {'lat': 42.1374, 'lon': 1.5900}, 'Olius': {'lat': 41.9785, 'lon': 1.5323} },
    "Tarragonès": { 'Tarragona': {'lat': 41.1189, 'lon': 1.2445}, 'Salou': {'lat': 41.0763, 'lon': 1.1417}, 'Altafulla': {'lat': 41.1417, 'lon': 1.3750} },
    "Terra Alta": { 'Gandesa': {'lat': 41.0526, 'lon': 0.4337}, 'Horta de Sant Joan': {'lat': 40.9545, 'lon': 0.3160}, 'Batea': {'lat': 41.0954, 'lon': 0.3119} },
    "Urgell": { 'Tàrrega': {'lat': 41.6469, 'lon': 1.1415}, 'Agramunt': {'lat': 41.7871, 'lon': 1.0967}, 'Bellpuig': {'lat': 41.6247, 'lon': 1.0118} },
    "Val d'Aran": { 'Vielha': {'lat': 42.7027, 'lon': 0.7966}, 'Bossòst': {'lat': 42.7877, 'lon': 0.6908}, 'Les': {'lat': 42.8126, 'lon': 0.7144} },
    "Vallès Occidental": { 'Sabadell': {'lat': 41.5483, 'lon': 2.1075}, 'Terrassa': {'lat': 41.5615, 'lon': 2.0084}, 'Sant Cugat del Vallès': {'lat': 41.4727, 'lon': 2.0863}, 'Rubí': {'lat': 41.4936, 'lon': 2.0323} },
    "Vallès Oriental": { 'Granollers': {'lat': 41.6083, 'lon': 2.2886}, 'Mollet del Vallès': {'lat': 41.5385, 'lon': 2.2144}, 'Sant Celoni': {'lat': 41.6903, 'lon': 2.4908}, 'Cardedeu': {'lat': 41.6403, 'lon': 2.3582} },
    
    # --- NOVES ZONES MARÍTIMES AFEGIDES ---
    "Mar Nord (Girona)": { 'Punt Marítim Nord': {'lat': 42.10, 'lon': 3.60} },
    "Mar Central (Barcelona)": { 'Punt Marítim Central': {'lat': 41.30, 'lon': 2.80} },
    "Mar Sud (Tarragona)": { 'Punt Marítim Sud': {'lat': 40.75, 'lon': 1.50} },
}

CIUTATS_CATALUNYA = { ciutat: dades for comarca in CIUTATS_PER_COMARCA.values() for ciutat, dades in comarca.items() }
PUNTS_MAR = { 'Costes de Girona (Mar)': {'lat': 42.05, 'lon': 3.30}, 'Litoral Barceloní (Mar)': {'lat': 41.40, 'lon': 2.90}, 'Aigües de Tarragona (Mar)': {'lat': 40.90, 'lon': 2.00} }
CIUTATS_CATALUNYA.update(PUNTS_MAR)

POBLACIONS_TERRA = {k: v for k, v in CIUTATS_CATALUNYA.items() if '(Mar)' not in k}
CIUTATS_CONVIDAT = { 'Barcelona': CIUTATS_CATALUNYA['Barcelona'], 'Girona': CIUTATS_CATALUNYA['Girona'], 'Lleida': CIUTATS_CATALUNYA['Lleida'], 'Tarragona': CIUTATS_CATALUNYA['Tarragona'] }
POBLES_MAPA_REFERENCIA = {poble: {'lat': data['lat'], 'lon': data['lon']} for poble, data in POBLACIONS_TERRA.items()}

CIUTATS_PER_ZONA_PERSONALITZADA = {
    "Pirineu i Pre-Pirineu": { p: CIUTATS_CATALUNYA[p] for p in ['Vielha', 'Sort', 'Tremp', 'La Pobla de Segur', 'La Seu d\'Urgell', 'Puigcerdà', 'Bellver de Cerdanya', 'La Molina', 'Ripoll', 'Sant Joan de les Abadesses', 'Berga', 'Solsona', 'Olot', 'Santa Pau', 'Camprodon'] if p in CIUTATS_CATALUNYA },
    "Plana de Lleida i Ponent": { p: CIUTATS_CATALUNYA[p] for p in ['Lleida', 'Alcarràs', 'Balaguer', 'Agramunt', 'Artesa de Segre', 'Calaf', 'Les Borges Blanques', 'Mollerussa', 'Tàrrega', 'Cervera'] if p in CIUTATS_CATALUNYA },
    "Catalunya Central": { p: CIUTATS_CATALUNYA[p] for p in ['Manresa', 'Cardona', 'Igualada', 'Capellades', 'Vic', 'Manlleu', 'Centelles', 'Moià', 'Súria'] if p in CIUTATS_CATALUNYA },
    "Litoral i Prelitoral Nord (Girona)": { p: CIUTATS_CATALUNYA[p] for p in ['Girona', 'Figueres', 'Banyoles', 'La Bisbal d\'Empordà', 'Roses', 'Cadaqués', 'Llançà', 'L\'Escala', 'Castelló d\'Empúries', 'La Jonquera', 'Palamós', 'Platja d\'Aro', 'Sant Feliu de Guíxols', 'Begur', 'Pals', 'Blanes', 'Lloret de Mar', 'Santa Coloma de Farners'] if p in CIUTATS_CATALUNYA },
    "Litoral i Prelitoral Central (Barcelona)": { p: CIUTATS_CATALUNYA[p] for p in ['Barcelona', 'L\'Hospitalet de Llobregat', 'Badalona', 'Sabadell', 'Terrassa', 'Mataró', 'Granollers', 'Mollet del Vallès', 'Sant Cugat del Vallès', 'Rubí', 'Viladecans', 'Vilanova i la Geltrú', 'Sitges', 'Vilafranca del Penedès', 'El Vendrell', 'Calafell'] if p in CIUTATS_CATALUNYA },
    "Camp de Tarragona": { p: CIUTATS_CATALUNYA[p] for p in ['Tarragona', 'Reus', 'Valls', 'Salou', 'Cambrils', 'Altafulla', 'Montblanc', 'Falset', 'Mont-roig del Camp'] if p in CIUTATS_CATALUNYA },
    "Terres de l'Ebre": { p: CIUTATS_CATALUNYA[p] for p in ['Tortosa', 'Amposta', 'Alcanar', 'L\'Ametlla de Mar', 'Deltebre', 'La Ràpita', 'Móra d\'Ebre', 'Gandesa'] if p in CIUTATS_CATALUNYA },
}

# --- Tornado Alley ---
API_URL_USA = "https://api.open-meteo.com/v1/forecast"
TIMEZONE_USA = pytz.timezone('America/Chicago')
USA_CITIES = { 'Dallas, TX': {'lat': 32.7767, 'lon': -96.7970}, 'Wichita, KS': {'lat': 37.6872, 'lon': -97.3301}, 'Houston, TX': {'lat': 29.7604, 'lon': -95.3698}, 'Kansas City, MO': {'lat': 39.0997, 'lon': -94.5786} }
MAP_EXTENT_USA = [-105, -85, 28, 48]
PRESS_LEVELS_HRRR = sorted([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650, 625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100], reverse=True)



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
    'LCL_Hgt': (1000, 1500), 
    'LFC_Hgt': (1500, 2500),
    'MAX_UPDRAFT': (25, 40, 55),
    'DCAPE': (500, 1000, 1500),
    'LR_0-3km': (6.5, 7.5, 8.5),
    'LR_700-500hPa': (6, 7, 8),
    'K_INDEX': (20, 30, 40),
    'TOTAL_TOTALS': (44, 48, 52),
    'SHOWALTER_INDEX': (2, -1, -4),
    'EBWD': (25, 40, 55),
    'ESRH': (100, 250, 400),
    'STP_CIN': (0.5, 1, 2.5),
    'SCP': (1, 4, 8),
    'SHIP': (0.5, 1, 2),
    # --- NOU LLINDAR AFEGIT ---
    'SWEAT_INDEX': (250, 300, 400),
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
    Aquesta funció està dissenyada per no fallar mai.
    """
    try:
        p_sfc = p[0]
        p_bottom = p_sfc - 100 * units.hPa
        mask = (p >= p_bottom) & (p <= p_sfc)

        if not np.any(mask):
            p_mixed, T_mixed, Td_mixed = p[0], T[0], Td[0]
        else:
            p_layer, T_layer, Td_layer = p[mask], T[mask], Td[mask]
            theta_mixed = np.mean(mpcalc.potential_temperature(p_layer, T_layer))
            # Corregit: Utilitzar dewpoint per a la ratio de barreja
            mixing_ratio_mixed = np.mean(mpcalc.mixing_ratio_from_dewpoint(p_layer, Td_layer))
            
            T_mixed = mpcalc.temperature_from_potential_temperature(p_sfc, theta_mixed)
            Td_mixed = mpcalc.dewpoint_from_mixing_ratio(p_sfc, mixing_ratio_mixed)
        
        prof_mixed = mpcalc.parcel_profile(p, T_mixed, Td_mixed).to('degC')
        mlcape, mlcin = mpcalc.cape_cin(p, T, Td, prof_mixed)
        
        return float(mlcape.m), float(mlcin.m)
    except Exception:
        return 0.0, 0.0


# -*- coding: utf-8 -*-

def processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile):
    """
    Versió Definitiva i COMPLETA v20.0 (Càlcul K-Index Blindat).
    - **CORRECCIÓ DE BUG DEFINITIVA**: El càlcul del K-Index ara utilitza les funcions natives
      de MetPy per a buscar les dades directament als nivells de pressió requerits.
      Això soluciona el problema de '---' en perfils de muntanya o incomplets.
    """
    # --- 1. PREPARACIÓ I VALIDACIÓ DE DADES ---
    if len(p_profile) < 4: return None, "Perfil atmosfèric massa curt."
    p = np.array(p_profile) * units.hPa; T = np.array(T_profile) * units.degC
    Td = np.array(Td_profile) * units.degC; u = np.array(u_profile) * units('m/s')
    v = np.array(v_profile) * units('m/s'); heights = np.array(h_profile) * units.meter
    valid_indices = ~np.isnan(p.m) & ~np.isnan(T.m) & ~np.isnan(Td.m) & ~np.isnan(u.m) & ~np.isnan(v.m)
    p, T, Td, u, v, heights = p[valid_indices], T[valid_indices], Td[valid_indices], u[valid_indices], v[valid_indices], heights[valid_indices]
    if len(p) < 4: return None, "No hi ha prou nivells amb dades vàlides."
    sort_idx = np.argsort(p.m)[::-1]
    p, T, Td, u, v, heights = p[sort_idx], T[sort_idx], Td[sort_idx], u[sort_idx], v[sort_idx], heights[sort_idx]
    heights_agl = heights - heights[0]; params_calc = {}

    # --- 2. CÀLCUL DELS PERFILS DE BOMBOLLA ---
    sfc_prof, ml_prof, mu_prof = None, None, None
    try: sfc_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
    except Exception: return None, "Error crític: No s'ha pogut calcular el perfil de superfície."
    try: _, _, _, ml_prof = mpcalc.mixed_parcel(p, T, Td, depth=100 * units.hPa); ml_prof = ml_prof.to('degC')
    except Exception: ml_prof = sfc_prof
    try: p_mu, T_mu, Td_mu, _ = mpcalc.most_unstable_parcel(p, T, Td); mu_prof = mpcalc.parcel_profile(p, T_mu, Td_mu).to('degC')
    except Exception: mu_prof = sfc_prof
    main_prof = ml_prof if ml_prof is not None else sfc_prof

    # --- 3. CÀLCULS TERMODINÀMICS ---
    try: sbcape, sbcin = mpcalc.cape_cin(p, T, Td, sfc_prof); params_calc['SBCAPE'], params_calc['SBCIN'] = float(sbcape.m), float(sbcin.m)
    except Exception: params_calc['SBCAPE'], params_calc['SBCIN'] = 0.0, 0.0
    try: mlcape, mlcin = mpcalc.cape_cin(p, T, Td, ml_prof); params_calc['MLCAPE'], params_calc['MLCIN'] = float(mlcape.m), float(mlcin.m)
    except Exception: params_calc['MLCAPE'], params_calc['MLCIN'] = 0.0, 0.0
    try: mucape, mucin = mpcalc.cape_cin(p, T, Td, mu_prof); params_calc['MUCAPE'], params_calc['MUCIN'] = float(mucape.m), float(mucin.m)
    except Exception: params_calc['MUCAPE'], params_calc['MUCIN'] = 0.0, 0.0
    try: idx_3km = np.argmin(np.abs(heights_agl.m - 3000)); cape_0_3, _ = mpcalc.cape_cin(p[:idx_3km+1], T[:idx_3km+1], Td[:idx_3km+1], main_prof[:idx_3km+1]); params_calc['CAPE_0-3km'] = float(cape_0_3.m)
    except Exception: params_calc['CAPE_0-3km'] = 0.0
    try: params_calc['LI'] = float(mpcalc.lifted_index(p, T, main_prof)[0].m)
    except Exception: params_calc['LI'] = np.nan
    try: lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); params_calc['LCL_Hgt'] = float(mpcalc.pressure_to_height_std(lcl_p).to('m').m - mpcalc.pressure_to_height_std(p[0]).to('m').m)
    except Exception: params_calc['LCL_Hgt'] = np.nan
    try: lfc_p, _ = mpcalc.lfc(p, T, Td, main_prof); params_calc['LFC_Hgt'] = float(mpcalc.pressure_to_height_std(lfc_p).to('m').m - mpcalc.pressure_to_height_std(p[0]).to('m').m)
    except Exception: params_calc['LFC_Hgt'] = np.nan
    try: el_p, _ = mpcalc.el(p, T, Td, main_prof); params_calc['EL_Hgt'] = float(mpcalc.pressure_to_height_std(el_p).to('m').m)
    except Exception: params_calc['EL_Hgt'] = np.nan
    try: params_calc['PWAT'] = float(mpcalc.precipitable_water(p, Td).to('mm').m)
    except Exception: params_calc['PWAT'] = np.nan
    try: params_calc['T_500hPa'] = float(np.interp(500, p.m[::-1], T.m[::-1]))
    except Exception: params_calc['T_500hPa'] = np.nan
    try: p_850 = 850 * units.hPa; T_850 = np.interp(p_850.m, p.m[::-1], T.m[::-1]) * units.degC; Td_850 = np.interp(p_850.m, p.m[::-1], Td.m[::-1]) * units.degC; theta_e_850 = mpcalc.equivalent_potential_temperature(p_850, T_850, Td_850); params_calc['THETAE_850hPa'] = float(theta_e_850.to('K').m)
    except Exception: params_calc['THETAE_850hPa'] = np.nan
    try: rh = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100; params_calc['RH_CAPES'] = {'baixa': float(np.mean(rh[(p.m <= 1000) & (p.m > 850)]).m), 'mitjana': float(np.mean(rh[(p.m <= 850) & (p.m > 400)]).m), 'alta': float(np.mean(rh[(p.m <= 400) & (p.m >= 100)]).m)}
    except Exception: params_calc['RH_CAPES'] = {'baixa': np.nan, 'mitjana': np.nan, 'alta': np.nan}
    wb_profile = np.full_like(p.m, np.nan) * units.degC
    try:
        wb_profile = mpcalc.wet_bulb_temperature(p, T, Td)
        wb_profile_clean = wb_profile[~np.isnan(wb_profile)].m; heights_agl_clean = heights_agl[~np.isnan(wb_profile)].m
        if wb_profile_clean.min() < 0 < wb_profile_clean.max():
            sort_indices = np.argsort(wb_profile_clean); wbz_height = np.interp(0, wb_profile_clean[sort_indices], heights_agl_clean[sort_indices]); params_calc['WBZ_HGT'] = float(wbz_height)
        elif wb_profile_clean.max() < 0: params_calc['WBZ_HGT'] = 0.0
        else: params_calc['WBZ_HGT'] = np.nan
    except Exception: params_calc['WBZ_HGT'] = np.nan

    # <<<--- CÀLCUL DEL K-INDEX (VERSIÓ BLINDADA I DEFINITIVA) ---
    try:
        # MetPy té una funció nativa que és molt més robusta
        params_calc['K_INDEX'] = float(mpcalc.k_index(p, T, Td).m)
    except (ValueError, IndexError):
        # Aquest error salta si falten els nivells de 850, 700 o 500 hPa
        params_calc['K_INDEX'] = np.nan
    except Exception:
        params_calc['K_INDEX'] = np.nan
    # <<<--- FI DEL BLOC ---

    # --- 4. CÀLCULS CINEMÀTICS ---
    try:
        for name, depth_m in [('0-1km', 1000), ('0-6km', 6000)]: bwd_u, bwd_v = mpcalc.bulk_shear(p, u, v, height=heights, depth=depth_m * units.meter); params_calc[f'BWD_{name}'] = float(mpcalc.wind_speed(bwd_u, bwd_v).to('kt').m)
    except Exception: params_calc.update({'BWD_0-1km': np.nan, 'BWD_0-6km': np.nan})
    try:
        rm, lm, mean_wind = mpcalc.bunkers_storm_motion(p, u, v, heights)
        params_calc['RM'] = (float(rm[0].m), float(rm[1].m)); params_calc['LM'] = (float(lm[0].m), float(lm[1].m)); params_calc['Mean_Wind'] = (float(mean_wind[0].m), float(mean_wind[1].m))
        u_storm, v_storm = rm[0], rm[1]
        for name, depth_m in [('0-1km', 1000), ('0-3km', 3000)]: srh = mpcalc.storm_relative_helicity(heights, u, v, depth=depth_m * units.meter, storm_u=u_storm, storm_v=v_storm)[0]; params_calc[f'SRH_{name}'] = float(srh.m)
    except Exception: params_calc.update({'RM': (np.nan, np.nan), 'LM': (np.nan, np.nan), 'Mean_Wind': (np.nan, np.nan), 'SRH_0-1km': np.nan, 'SRH_0-3km': np.nan})
    
    # --- 5. ÍNDEXS COMPOSTOS ---
    try: params_calc['DCAPE'] = float(mpcalc.dcape(p, T, Td)[0].m)
    except Exception: params_calc['DCAPE'] = 0.0
    try:
        eff_p_bottom, eff_p_top = mpcalc.effective_inflow_layer(p, T, Td); ebwd_u, ebwd_v = mpcalc.bulk_shear(p, u, v, height=heights, bottom=eff_p_bottom, top=eff_p_top); params_calc['EBWD'] = float(mpcalc.wind_speed(ebwd_u, ebwd_v).to('kt').m); esrh = mpcalc.storm_relative_helicity(heights, u, v, bottom=mpcalc.pressure_to_height_std(eff_p_bottom), top=mpcalc.pressure_to_height_std(eff_p_top), storm_u=u_storm, storm_v=v_storm)[0]; params_calc['ESRH'] = float(esrh.m)
    except Exception: params_calc['EBWD'], params_calc['ESRH'] = np.nan, np.nan
    try: params_calc['SCP'] = float(mpcalc.supercell_composite(params_calc['MUCAPE'] * units('J/kg'), params_calc['ESRH'] * units('m^2/s^2'), params_calc['EBWD'] * units.kt).m)
    except Exception: params_calc['SCP'] = np.nan
    try: params_calc['STP_CIN'] = float(mpcalc.significant_tornado(params_calc['SBCAPE'] * units('J/kg'), params_calc['SRH_0-1km'] * units('m^2/s^2'), params_calc['BWD_0-6km'] * units.kt, params_calc['LCL_Hgt'] * units.m, params_calc['SBCIN'] * units('J/kg')).m)
    except Exception: params_calc['STP_CIN'] = np.nan
    try: params_calc['SHIP'] = float(mpcalc.significant_hail_parameter(params_calc['MUCAPE']*units('J/kg'), mpcalc.mixing_ratio_from_dewpoint(p_mu, Td_mu), params_calc['BWD_0-6km']*units('kt'), T[np.where(p.m==500)[0][0]]*units.degC, (T[0] - Td[0]).to('delta_degC')).m)
    except Exception: params_calc['SHIP'] = np.nan
    try: params_calc['SWEAT_INDEX'] = float(mpcalc.sweat_index(p, T, Td, u, v).m)
    except Exception: params_calc['SWEAT_INDEX'] = np.nan
    
    # --- 6. RETORN DE LES DADES PROCESSADES ---
    processed_tuple = (p, T, Td, u, v, heights, main_prof, wb_profile)
    return (processed_tuple, params_calc), None
    
    



def ui_parametres_addicionals_sondeig(params):
    """
    Mostra una secció expandible amb paràmetres de sondeig avançats,
    agrupats per categories per a una millor llegibilitat.
    """
    if not params:
        return

    def styled_metric_small(label, value, unit, param_key, tooltip="", precision=1, reverse=False):
        color = get_color_global(value, param_key, reverse_colors=reverse) if pd.notna(value) else "#808080"
        val_str = f"{value:.{precision}f}" if pd.notna(value) else "---"
        tooltip_html = f'<span title="{tooltip}" style="cursor: help; opacity: 0.7;"> ❓</span>' if tooltip else ""
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; border-bottom: 1px solid #333;">
            <span style="font-size: 0.9em; color: #FAFAFA;">{label}{tooltip_html}</span>
            <strong style="font-size: 1.1em; color: {color};">{val_str} <span style="font-size: 0.8em; color: #808080;">{unit}</span></strong>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔬 Anàlisi de Paràmetres Addicionals", expanded=False):
        st.markdown("##### Índexs Compostos Severs")
        col1, col2 = st.columns(2)
        with col1:
            styled_metric_small("Supercell Composite (SCP)", params.get('SCP'), "", 'SCP', tooltip="Potencial per a supercèl·lules. >1 és significatiu.")
            styled_metric_small("Significant Hail (SHIP)", params.get('SHIP'), "", 'SHIP', tooltip="Potencial per a calamarsa severa (>5cm). >1 és significatiu.")
        with col2:
            styled_metric_small("Significant Tornado (STP)", params.get('STP_CIN'), "", 'STP_CIN', tooltip="Potencial per a tornados significatius (EF2+). >1 és significatiu.")
            st.empty() 

        st.divider()
        st.markdown("##### Termodinàmica Detallada")
        col3, col4, col5 = st.columns(3)
        with col3:
            styled_metric_small("Downdraft CAPE (DCAPE)", params.get('DCAPE'), "J/kg", 'DCAPE', tooltip="Potencial per a ràfegues de vent descendents severes.")
            styled_metric_small("K Index", params.get('K_INDEX'), "", 'K_INDEX', tooltip="Potencial de tempestes per massa d'aire. >35 indica alt potencial.")
        with col4:
            styled_metric_small("Lapse Rate 0-3km", params.get('LR_0-3km'), "°C/km", 'LR_0-3km', tooltip="Refredament amb l'altura a capes baixes. >7.5°C/km és molt inestable.")
            styled_metric_small("Total Totals Index", params.get('TOTAL_TOTALS'), "", 'TOTAL_TOTALS', tooltip="Índex de severitat. >50 indica potencial per a tempestes fortes.")
        with col5:
            styled_metric_small("Lapse Rate 700-500hPa", params.get('LR_700-500hPa'), "°C/km", 'LR_700-500hPa', tooltip="Inestabilitat a nivells mitjans. >7°C/km afavoreix la calamarsa.")
            styled_metric_small("Showalter Index", params.get('SHOWALTER_INDEX'), "", 'SHOWALTER_INDEX', reverse=True, tooltip="Mesura d'inestabilitat. Valors negatius indiquen potencial de tempesta.")
            
        st.divider()
        st.markdown("##### Cinemàtica i Cisallament Avançat")
        col6, col7 = st.columns(2)
        with col6:
            styled_metric_small("Effective SRH (ESRH)", params.get('ESRH'), "m²/s²", 'ESRH', tooltip="Helicitat relativa a la tempesta a la capa efectiva. >150 m²/s² afavoreix supercèl·lules.")
            styled_metric_small("Effective Inflow Base", params.get('EFF_INFLOW_BOTTOM'), "hPa", 'EFF_INFLOW_BOTTOM', tooltip="Base de la capa d'aire que alimenta la tempesta.")
        with col7:
            styled_metric_small("Effective Shear (EBWD)", params.get('EBWD'), "nusos", 'EBWD', tooltip="Cisallament del vent a la capa efectiva. >40 nusos afavoreix supercèl·lules.")
            styled_metric_small("Effective Inflow Top", params.get('EFF_INFLOW_TOP'), "hPa", 'EFF_INFLOW_TOP', tooltip="Sostre de la capa d'aire que alimenta la tempesta.")





def get_comarca_for_poble(poble_name):
    """
    Troba la comarca OFICIAL a la qual pertany un municipi.
    Això garanteix que sempre tindrem un nom de geometria vàlid per al mapa.
    """

    for comarca, pobles in CIUTATS_PER_COMARCA.items():
        if poble_name in pobles:
            return comarca
    return None
    

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




    
def crear_mapa_base(map_extent: Tuple[float, float, float, float], projection=ccrs.PlateCarree()) -> Tuple[plt.Figure, plt.Axes]:
    """
    Crea una figura y un eje de mapa base con Cartopy (versión con tamaño y DPI ajustados).
    """
    # S'utilitzen els valors de figsize i dpi desitjats per al mapa comarcal
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120, subplot_kw={'projection': projection})
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor="#D4E6B5", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#4682B4', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=5)
    if projection != ccrs.PlateCarree():
        ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='gray', zorder=5)
    return fig, ax




def afegir_etiquetes_ciutats(ax: plt.Axes, map_extent: Tuple[float, float, float, float]):
    """
    Versió amb etiquetes més petites per a una millor claredat visual en fer zoom.
    """
    is_zoomed_in = (tuple(map_extent) != tuple(MAP_EXTENT_CAT))

    if is_zoomed_in:
        for ciutat, coords in POBLES_MAPA_REFERENCIA.items():
            lon, lat = coords['lon'], coords['lat']
            if map_extent[0] < lon < map_extent[1] and map_extent[2] < lat < map_extent[3]:
                ax.plot(lon, lat, 'o', color='black', markersize=1,
                        markeredgecolor='black', markeredgewidth=1.5,
                        transform=ccrs.PlateCarree(), zorder=19)
                ax.text(lon + 0.02, lat, ciutat, 
                        fontsize=5,
                        color='white',
                        transform=ccrs.PlateCarree(), 
                        zorder=2,
                        path_effects=[path_effects.withStroke(linewidth=2.5, foreground='gray')])


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




def crear_skewt(p, T, Td, Twb, u, v, prof, params_calc, titol, timestamp_str, zoom_capa_baixa=False):
    """
    Versió Definitiva v3.0.
    - **NOU**: Accepta i dibuixa el perfil de Temperatura de Bulb Humit (Twb) de color lila suau.
    - Manté la correcció del bug de l'ombra de CAPE/CIN.
    """
    fig = plt.figure(dpi=150, figsize=(7, 8))
    
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.05, 0.85, 0.85))
    skew.ax.grid(True, linestyle='-', alpha=0.5)

    if zoom_capa_baixa:
        pressio_superficie = p[0].m
        skew.ax.set_ylim(pressio_superficie + 5, 800)
    else:
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 40)
        
    skew.ax.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    skew.plot_dry_adiabats(color='coral', linestyle='--', alpha=0.5)
    skew.plot_moist_adiabats(color='cornflowerblue', linestyle='--', alpha=0.5)
    skew.plot_mixing_lines(color='limegreen', linestyle='--', alpha=0.5)
    
    if prof is not None:
        valid_shade_mask = np.isfinite(p.m) & np.isfinite(T.m) & np.isfinite(prof.m)
        p_clean, T_clean, prof_clean = p[valid_shade_mask], T[valid_shade_mask], prof[valid_shade_mask]
        skew.shade_cape(p_clean, T_clean, prof_clean, color='yellow', alpha=0.4)
        skew.shade_cin(p_clean, T_clean, prof_clean, color='gray', alpha=0.7)
        skew.plot(p, prof, 'k', linewidth=3, label='Trajectòria Parcel·la', path_effects=[path_effects.withStroke(linewidth=4, foreground='white')])

    skew.plot(p, T, 'red', lw=2.5, label='Temperatura')
    skew.plot(p, Td, 'blue', lw=2, label='Punt de Rosada')
    skew.plot(p, Twb, color='#C8A2C8', linestyle='-', lw=2, label='Bulb Humit (Twb)')
    
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


def analitzar_amenaces_severes(params, sounding_data, nivell_conv):
    """
    Sistema d'Anàlisi d'Amenaces v7.0 (Diagnòstic de Precipitació Detallat).
    - **CANVI RADICAL**: Avalua un ampli espectre de tipus de precipitació i mostra el
      més significatiu, des de plugims fins a calamarsa enorme.
    - Manté l'anàlisi de l'índex de potencial i l'activitat elèctrica.
    """
    # --- 1. Inicialització dels resultats per a les 3 caixes ---
    resultats = {
        'precipitacio': {'label': 'Precipitació Principal', 'text': 'Nul·la', 'color': '#808080'},
        'potencial': {'label': 'Índex de Potencial', 'text': '0 / 10', 'color': '#808080'},
        'electricitat': {'label': 'Activitat Elèctrica', 'text': 'Nul·la', 'color': '#808080'}
    }

    # --- 2. Extracció de tots els paràmetres necessaris ---
    mlcape = params.get('MLCAPE', 0) or 0
    mucape = params.get('MUCAPE', 0) or 0
    wbz_hgt = params.get('WBZ_HGT', 5000) or 5000
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    pwat = params.get('PWAT', 0) or 0
    lcl_hgt = params.get('LCL_Hgt', 9999) or 9999
    rh_baixa = params.get('RH_CAPES', {}).get('baixa', 0)
    t500 = params.get('T_500hPa', 0) or 0
    li = params.get('LI', 5) or 5
    el_hgt = params.get('EL_Hgt', 0) or 0
    conv_key = next((k for k in params if k.startswith('CONV_')), None)
    raw_conv_value = params.get(conv_key, 0)
    convergencia = raw_conv_value if isinstance(raw_conv_value, (int, float, np.number)) else 0
    cin = min(params.get('SBCIN', 0), params.get('MUCIN', 0)) or 0
    
    # --- 3. Càlcul de Potencial i Electricitat (es mantenen) ---
    puntuacio_resultat = calcular_puntuacio_tempesta(sounding_data, params, nivell_conv)
    resultats['potencial'] = {
        'label': 'Índex de Potencial',
        'text': f"{puntuacio_resultat['score']} / 10",
        'color': puntuacio_resultat['color']
    }
    
    factor_realitzacio = 0.0
    if convergencia >= 30 and cin > -100: factor_realitzacio = 1.0
    elif convergencia >= 15 and cin > -50: factor_realitzacio = 0.7
    elif cin > -20: factor_realitzacio = 0.4
    
    potencial_llamps_teoric = 0
    if li < -7 or (li < -5 and el_hgt > 12000): potencial_llamps_teoric = 4
    elif li < -4 or (li < -2 and el_hgt > 10000): potencial_llamps_teoric = 3
    elif li < -1: potencial_llamps_teoric = 2
    elif mucape > 150: potencial_llamps_teoric = 1
    potencial_llamps_real = potencial_llamps_teoric * factor_realitzacio
    if potencial_llamps_real >= 3.5: resultats['electricitat'] = {'label': 'Activitat Elèctrica', 'text': 'Extrema', 'color': '#dc3545'}
    elif potencial_llamps_real >= 2.5: resultats['electricitat'] = {'label': 'Activitat Elèctrica', 'text': 'Alta', 'color': '#fd7e14'}
    elif potencial_llamps_real >= 1.5: resultats['electricitat'] = {'label': 'Activitat Elèctrica', 'text': 'Moderada', 'color': '#ffc107'}
    elif potencial_llamps_real >= 0.5: resultats['electricitat'] = {'label': 'Activitat Elèctrica', 'text': 'Baixa', 'color': '#2ca02c'}

    # --- 4. Lògica Jeràrquica per a la Precipitació Principal ---
    # Comença per l'amenaça més severa i va baixant. El primer que es compleix és el que es mostra.
    
    if pwat < 15 and rh_baixa < 60:
        return resultats # Atmosfera massa seca, retorna els valors per defecte.

    if mucape > 3500 and bwd_6km >= 40 and (2100 <= wbz_hgt < 3000):
        resultats['precipitacio'] = {'label': 'Calamarsa Enorme (>5cm)', 'text': 'Extrem', 'color': '#9370DB'}
    elif mucape > 2000 and bwd_6km >= 35 and (1800 <= wbz_hgt < 3200):
        resultats['precipitacio'] = {'label': 'Calamarsa Gran (>2cm)', 'text': 'Molt Alta', 'color': '#dc3545'}
    elif mucape > 1500 and wbz_hgt < 3300:
        resultats['precipitacio'] = {'label': 'Calamarsa Normal', 'text': 'Alta', 'color': '#fd7e14'}
    elif pwat > 40 and mucape > 1000:
        resultats['precipitacio'] = {'label': 'Pluja Torrencial', 'text': 'Alta', 'color': '#0D6EFD'}
    elif pwat > 30 and mucape > 500:
        resultats['precipitacio'] = {'label': 'Pluja Forta', 'text': 'Moderada', 'color': '#28A745'}
    elif 50 < mlcape < 500 and t500 < -10:
        resultats['precipitacio'] = {'label': 'Calamarsa Rodona (Graupel)', 'text': 'Baixa', 'color': '#6C757D'}
    elif lcl_hgt < 400 and mlcape < 100:
        resultats['precipitacio'] = {'label': 'Plugims', 'text': 'Febla', 'color': '#adb5bd'}

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
    




MAPA_IMATGES_REALS = {
    # Tempestes i Temps Sever
    "Potencial de Supercèl·lula": "Potencial de Supercèl·lula.jpg",
    "Tempestes Organitzades": "Tempestes Organitzades.jpg",
    "Tempesta Aïllada (Molt energètica)": "Tempesta Aïllada (Molt energètica).jpg",
    "Tempesta Comuna": "Tempesta Comuna.jpg",
    "Nimbostratus (Pluja Contínua)": "Nimbostratus (Pluja Contínua).jpg",
    
    # Núvols Comuns i Altres Fenòmens
    "Cúmuls de creixement": "Cúmuls de creixement.jpg",
    "Cúmuls mediocris": "Cúmuls mediocris.jpg",
    "Cúmuls de bon temps": "Cúmuls de bon temps.jpg",
    "Estratus (Boira alta - Cel tancat)": "Estratus (Boira alta - Cel tancat).jpg", # Nom corregit
    "Fractocúmuls": "Fractocúmuls.jpg",
    "Altostratus - Altocúmulus": "Altostratus - Altocúmulus.jpg", # Nom corregit
    "Cirrus Castellanus": "Cirrus Castellanus.jpg",
    "Cirrostratus (Cel blanquinós)": "Cirrostratus (Cel blanquinós).jpg",
    "Vels de Cirrus (Molt Alts)": "Vels de Cirrus (Molt Alts).jpg",
    "Altocúmulus Lenticular": "Altocúmulus Lenticular.jpg",
    "Cel Serè": "Cel Serè.jpg",
    
    # Imatge per defecte
    "fallback": "fallback.jpg"
}

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

def ui_caixa_parametres_sondeig(sounding_data, params, nivell_conv, hora_actual, poble_sel, avis_proximitat=None):
    """
    Versió Definitiva v67.0 (Diagnòstic Estable Independent).
    """
    TOOLTIPS = {
        'MLCAPE': "Mixed-Layer CAPE: L'energia disponible per a una bombolla d'aire que representa la mitjana de les capes baixes. És l'indicador més fiable del potencial de tempesta.",
        'LI': "Lifted Index: Mesura la diferència de temperatura entre una bombolla d'aire elevada a 500 hPa i l'entorn. Valors molt negatius indiquen una forta inestabilitat.",
        'CONV_PUNTUAL': "Convergència (+): Acumulació d'aire a un nivell que força l'ascens. És el 'disparador' principal per iniciar tempestes. Divergència (-): L'aire s'escampa, afavorint l'estabilitat.",
        'CAPE_0-3km': "Energia concentrada a les capes baixes de l'atmosfera. Valors alts afavoreixen el desenvolupament de rotació (mesociclons) a les tempestes.",
        'K_INDEX': "Índex que combina temperatura i humitat a diferents nivells per estimar el potencial de tempestes per massa d'aire. Valors > 35 indiquen alt potencial.",
        'SBCIN': "Inhibició Convectiva (Tapa) des de la superfície. És l'energia que cal vèncer perquè comenci una tempesta. Valors molt negatius actuen com una tapa molt forta.",
        'PWAT': "Aigua Precipitable Total: La quantitat total de vapor d'aigua en una columna d'aire. Valors alts indiquen un alt potencial per a pluges fortes o torrencials.",
        'THETAE_850hPa': "Temperatura Potencial Equivalent a 850hPa (~1500m). Un indicador de la reserva d'energia termodinàmica (calor + humitat) a les capes baixes.",
        'LCL_Hgt': "Nivell de Condensació per Elevació: L'alçada a la qual una bombolla d'aire es refreda fins a saturar-se i formar la base del núvol.",
        'LFC_Hgt': "Nivell de Convecció Lliure: L'alçada a partir de la qual una bombolla d'aire ja és més càlida que l'entorn i accelera cap amunt sense necessitat de forçament extern.",
        'EL_Hgt': "Nivell d'Equilibri: L'alçada màxima que teòricament pot assolir el cim d'una tempesta, on la bombolla d'aire es refreda i ja no pot pujar més.",
        'BWD_0-6km': "Cisallament del Vent (Bulk Wind Difference) entre la superfície i 6 km. És el paràmetre clau per a l'organització de les tempestes. Valors > 35 nusos afavoreixen les supercèl·lules.",
        'BWD_0-1km': "Cisallament del vent a les capes molt baixes. Important per al potencial de tornados.",
        'T_500hPa': "Temperatura a 500 hPa (~5500m). Valors molt freds (< -12°C) indiquen una 'butxaca freda' que augmenta molt la inestabilitat.",
        'MUCIN': "La 'tapa' més feble de tota l'atmosfera. Si fins i tot aquest valor és molt alt, la convecció és gairebé impossible.",
        'PUNTUACIO_TEMPESTA': "Índex global de potencial de tempesta (0-10) que combina energia, cisallament i disparador.",
    }
    
    def styled_metric(label, value, unit, param_key, tooltip_text="", precision=0, reverse_colors=False):
        color = "#FFFFFF"; val_str = "---"
        is_numeric = isinstance(value, (int, float, np.number))
        if pd.notna(value) and is_numeric:
            if 'CONV' in param_key:
                conv_thresholds = [5, 15, 30, 40]; conv_colors = ["#808080", "#2ca02c", "#ffc107", "#fd7e14", "#dc3545"]
                color_idx = np.searchsorted(conv_thresholds, abs(value))
                color = conv_colors[color_idx] if value >= 0 else "#6495ED"
            else: 
                color = get_color_global(value, param_key, reverse_colors)
            val_str = f"{value:.{precision}f}"
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""
        st.markdown(f"""<div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;"><span style="font-size: 0.8em; color: #FAFAFA;">{label} ({unit}){tooltip_html}</span><strong style="font-size: 1.6em; color: {color}; line-height: 1.1;">{val_str}</strong></div>""", unsafe_allow_html=True)

    def styled_qualitative(label, text, color, tooltip_text=""):
        tooltip_html = f' <span title="{tooltip_text}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span>' if tooltip_text else ""
        st.markdown(f"""<div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;"><span style="font-size: 0.8em; color: #FAFAFA;">{label}{tooltip_html}</span><br><strong style="font-size: 1.6em; color: {color};">{text}</strong></div>""", unsafe_allow_html=True)

    analisi_temps_dict = analitzar_potencial_meteorologic(params, nivell_conv, hora_actual)

    st.markdown("##### 🌩️ Potencial de Núvols Inestables")
    analisi_inestable = analisi_temps_dict.get("inestable")
    if analisi_inestable:
        desc, veredicte = analisi_inestable.get("descripcio"), analisi_inestable.get("veredicte")
        nom_arxiu = MAPA_IMATGES_REALS.get(desc, MAPA_IMATGES_REALS["fallback"]); ruta_arxiu_imatge = os.path.join("imatges_reals", nom_arxiu)
        b64_img = convertir_img_a_base64(ruta_arxiu_imatge)
        st.markdown(f"""<div style="position: relative; width: 100%; height: 150px; border-radius: 10px; background-image: url('{b64_img}'); background-size: cover; background-position: center; display: flex; align-items: flex-end; padding: 15px; box-shadow: inset 0 -80px 60px -30px rgba(0,0,0,0.8); margin-bottom: 10px;"><div style="color: white; text-shadow: 2px 2px 5px rgba(0,0,0,0.8);"><strong style="font-size: 1.3em;">{veredicte}</strong><br><em style="font-size: 0.9em; color: #DDDDDD;">({desc})</em></div></div>""", unsafe_allow_html=True)
    else:
        st.info("L'atmosfera és estable. No s'espera formació de núvols de tempesta.", icon="✅")

    st.markdown("##### ☁️ Tipus de Nuvolositat Estable Prevista")
    analisi_estable = analisi_temps_dict.get("estable")
    if analisi_estable:
        desc, veredicte = analisi_estable.get("descripcio"), analisi_estable.get("veredicte")
        nom_arxiu = MAPA_IMATGES_REALS.get(desc, MAPA_IMATGES_REALS["fallback"]); ruta_arxiu_imatge = os.path.join("imatges_reals", nom_arxiu)
        b64_img = convertir_img_a_base64(ruta_arxiu_imatge)
        st.markdown(f"""<div style="position: relative; width: 100%; height: 150px; border-radius: 10px; background-image: url('{b64_img}'); background-size: cover; background-position: center; display: flex; align-items: flex-end; padding: 15px; box-shadow: inset 0 -80px 60px -30px rgba(0,0,0,0.8); margin-bottom: 10px;"><div style="color: white; text-shadow: 2px 2px 5px rgba(0,0,0,0.8);"><strong style="font-size: 1.3em;">{veredicte}</strong><br><em style="font-size: 0.9em; color: #DDDDDD;">({desc})</em></div></div>""", unsafe_allow_html=True)
    
    # La resta de la funció es manté igual
    st.markdown("##### ⚡ Energia i Inestabilitat")
    cols_energia = st.columns(4)
    with cols_energia[0]: styled_metric("MLCAPE", params.get('MLCAPE', np.nan), "J/kg", 'MLCAPE', tooltip_text=TOOLTIPS.get('MLCAPE'))
    with cols_energia[1]: styled_metric("LI", params.get('LI', np.nan), "°C", 'LI', tooltip_text=TOOLTIPS.get('LI'), precision=1, reverse_colors=True)
    with cols_energia[2]: styled_metric("3CAPE", params.get('CAPE_0-3km', np.nan), "J/kg", 'CAPE_0-3km', tooltip_text=TOOLTIPS.get('CAPE_0-3km'))
    with cols_energia[3]: styled_metric("K-Index", params.get('K_INDEX', np.nan), "", 'K_INDEX', tooltip_text=TOOLTIPS.get('K_INDEX'))

    st.markdown("##### 💧 Humitat i Potencial de Precipitació")
    cols_humitat = st.columns(3)
    with cols_humitat[0]: styled_metric("PWAT", params.get('PWAT', np.nan), "mm", 'PWAT', tooltip_text=TOOLTIPS.get('PWAT'), precision=1)
    with cols_humitat[1]:
        t500_val = params.get('T_500hPa', np.nan); t500_color = "#FFFFFF"
        if pd.notna(t500_val):
            if t500_val < -12: t500_color = "#0D6EFD"
            elif t500_val < -5: t500_color = "#6C757D"
            else: t500_color = "#FD7E14"
        t500_val_str = f"{t500_val:.1f}" if pd.notna(t500_val) else "---"
        st.markdown(f"""<div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;"><span style="font-size: 0.8em; color: #FAFAFA;">T 500hPa (°C) <span title="{TOOLTIPS.get('T_500hPa')}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span></span><strong style="font-size: 1.6em; color: {t500_color}; line-height: 1.1;">{t500_val_str}</strong></div>""", unsafe_allow_html=True)
    with cols_humitat[2]:
        theta_e_k = params.get('THETAE_850hPa', np.nan)
        theta_e_c = theta_e_k - 273.15 if pd.notna(theta_e_k) else np.nan
        styled_metric("Theta-E 850", theta_e_c, "°C", 'THETAE_850hPa', tooltip_text=TOOLTIPS.get('THETAE_850hPa'), precision=1)
    
    rh_capes = params.get('RH_CAPES', {}); rh_b = rh_capes.get('baixa', np.nan); rh_m = rh_capes.get('mitjana', np.nan); rh_a = rh_capes.get('alta', np.nan)
    def get_rh_color(rh_value):
        if not pd.notna(rh_value): return "#FFFFFF"
        if rh_value > 85: return "#0047AB";
        if rh_value > 70: return "#0D6EFD";
        if rh_value > 50: return "#28A745";
        if rh_value > 30: return "#FFC107";
        return "#FFDAB9"
    rh_b_str = f"{rh_b:.0f}%" if pd.notna(rh_b) else "---"; rh_m_str = f"{rh_m:.0f}%" if pd.notna(rh_m) else "---"; rh_a_str = f"{rh_a:.0f}%" if pd.notna(rh_a) else "---"
    st.markdown(f"""<div style="padding: 10px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px;"><p style="text-align:center; font-size: 0.8em; color: #FAFAFA; margin-bottom: 8px; margin-top: -5px;">Humitat Relativa (RH %)</p><div style="display: flex; justify-content: space-around; text-align: center;"><div><span style="font-size: 0.8em; color: #A0A0B0;">Baixa</span><strong style="display: block; font-size: 1.6em; color: {get_rh_color(rh_b)}; line-height: 1.1;">{rh_b_str}</strong></div><div><span style="font-size: 0.8em; color: #A0A0B0;">Mitjana</span><strong style="display: block; font-size: 1.6em; color: {get_rh_color(rh_m)}; line-height: 1.1;">{rh_m_str}</strong></div><div><span style="font-size: 0.8em; color: #A0A0B0;">Alta</span><strong style="display: block; font-size: 1.6em; color: {get_rh_color(rh_a)}; line-height: 1.1;">{rh_a_str}</strong></div></div></div>""", unsafe_allow_html=True)

    st.markdown("##### ⛔ Inhibició, Disparador i Nivells Clau")
    cols_nivells = st.columns(5)
    with cols_nivells[0]: styled_metric("SBCIN", params.get('SBCIN', np.nan), "J/kg", 'SBCIN', reverse_colors=True, tooltip_text=TOOLTIPS.get('SBCIN'))
    with cols_nivells[1]: styled_metric("MUCIN", params.get('MUCIN', np.nan), "J/kg", 'MUCIN', reverse_colors=True, tooltip_text=TOOLTIPS.get('MUCIN'))
    with cols_nivells[2]:
        conv_key = f'CONV_{nivell_conv}hPa'; conv_value = params.get(conv_key, np.nan)
        label_conv = "Convergència" if pd.isna(conv_value) or conv_value >= 0 else "Divergència"
        styled_metric(label_conv, conv_value, "×10⁻⁵ s⁻¹", "CONV_PUNTUAL", tooltip_text=TOOLTIPS.get('CONV_PUNTUAL'), precision=1)
    with cols_nivells[3]: styled_metric("LCL", params.get('LCL_Hgt', np.nan), "m", 'LCL_Hgt', precision=0, tooltip_text=TOOLTIPS.get('LCL_Hgt'))
    with cols_nivells[4]: styled_metric("LFC", params.get('LFC_Hgt', np.nan), "m", 'LFC_Hgt', precision=0, tooltip_text=TOOLTIPS.get('LFC_Hgt'))

    st.markdown("##### 💨 Cinemàtica (Vent i Cisallament)")
    cols_cinematica = st.columns(3)
    with cols_cinematica[0]: styled_metric("BWD 0-6km", params.get('BWD_0-6km', np.nan), "nusos", 'BWD_0-6km', tooltip_text=TOOLTIPS.get('BWD_0-6km'))
    with cols_cinematica[1]: styled_metric("BWD 0-1km", params.get('BWD_0-1km', np.nan), "nusos", 'BWD_0-1km', tooltip_text=TOOLTIPS.get('BWD_0-1km'))
    with cols_cinematica[2]:
        el_hgt_val = params.get('EL_Hgt', np.nan); el_color = "#FFFFFF"
        if pd.notna(el_hgt_val):
            if el_hgt_val > 12000: el_color = "#DC3545"
            elif el_hgt_val > 9000: el_color = "#FD7E14"
            elif el_hgt_val > 6000: el_color = "#FFC107"
        el_val_str = f"{el_hgt_val:.0f}" if pd.notna(el_hgt_val) else "---"
        st.markdown(f"""<div style="text-align: center; padding: 5px; border-radius: 10px; background-color: #2a2c34; margin-bottom: 10px; height: 78px; display: flex; flex-direction: column; justify-content: center;"><span style="font-size: 0.8em; color: #FAFAFA;">CIM (EL) (m) <span title="{TOOLTIPS.get('EL_Hgt')}" style="cursor: help; font-size: 0.8em; opacity: 0.7;">❓</span></span><strong style="font-size: 1.6em; color: {el_color}; line-height: 1.1;">{el_val_str}</strong></div>""", unsafe_allow_html=True)

    st.markdown("##### ⛈️ Potencial d'Amenaces Severes")
    amenaces = analitzar_amenaces_severes(params, sounding_data, nivell_conv)
    cols_amenaces = st.columns(3)
    with cols_amenaces[0]:
        precip_data = amenaces['precipitacio']; styled_qualitative(precip_data['label'], precip_data['text'], precip_data['color'])
    with cols_amenaces[1]:
        potencial_data = amenaces['potencial']; styled_qualitative(potencial_data['label'], potencial_data['text'], potencial_data['color'])
    with cols_amenaces[2]:
        electricitat_data = amenaces['electricitat']; styled_qualitative(electricitat_data['label'], electricitat_data['text'], electricitat_data['color'])

        
        
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
    Versió Final i Corregida.
    - **CORRECCIÓ DE BUG**: Passa correctament el perfil de Bulb Humit (Twb) a la funció
      crear_skewt, solucionant el TypeError.
    """

    if data_tuple:
        sounding_data, params_calculats = data_tuple
        
        # <<<--- LÍNIA CORREGIDA: Ara desempaquetem 8 valors, incloent Twb ---
        p, T, Td, u, v, heights, prof, Twb = sounding_data
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            zoom_capa_baixa = st.checkbox("🔍 Zoom a la Capa Baixa (Superfície - 800 hPa)")
            
            # <<<--- LÍNIA CORREGIDA: La crida a crear_skewt ara inclou Twb ---
            fig_skewt = crear_skewt(p, T, Td, Twb, u, v, prof, params_calculats, f"Sondeig Vertical - {poble_sel}", timestamp_str, zoom_capa_baixa=zoom_capa_baixa)
            
            st.pyplot(fig_skewt, use_container_width=True)
            plt.close(fig_skewt)
            with st.container(border=True):
                ui_caixa_parametres_sondeig(sounding_data, params_calculats, nivell_conv, hora_actual, poble_sel, avis_proximitat)

        with col2:
            fig_hodo = crear_hodograf_avancat(p, u, v, heights, params_calculats, f"Hodògraf Avançat - {poble_sel}", timestamp_str)
            st.pyplot(fig_hodo, use_container_width=True)
            plt.close(fig_hodo)

            if avis_proximitat and isinstance(avis_proximitat, dict):
                st.warning(f"⚠️ **AVÍS DE PROXIMITAT:** {avis_proximitat['message']}")
                if avis_proximitat['target_city'] == poble_sel:
                    st.button("📍 Ja ets a la millor zona convergent d'anàlisi, mira si hi ha MU/SBCAPE! I poc MU/SBCIN!",
                              help="El punt d'anàlisi més proper a l'amenaça és la localitat que ja estàs consultant.",
                              use_container_width=True,
                              disabled=True)
                else:
                    tooltip_text = f"Viatjar a {avis_proximitat['target_city']}, el punt d'anàlisi més proper al nucli de convergència (Força: {avis_proximitat['conv_value']:.0f})."
                    st.button("🛰️ Analitzar Zona d'Amenaça", 
                              help=tooltip_text, 
                              use_container_width=True, 
                              type="primary",
                              on_click=canviar_poble_analitzat,
                              args=(avis_proximitat['target_city'],)
                             )
            
            st.markdown("##### Radar de Precipitació en Temps Real")
            radar_url = f"https://www.rainviewer.com/map.html?loc={lat},{lon},10&oCS=1&c=3&o=83&lm=0&layer=radar&sm=1&sn=1&ts=2&play=1"
            html_code = f"""<div style="position: relative; width: 100%; height: 410px; border-radius: 10px; overflow: hidden;"><iframe src="{radar_url}" width="100%" height="410" frameborder="0" style="border:0;"></iframe><div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; cursor: default;"></div></div>"""
            st.components.v1.html(html_code, height=410)
            
    else:
        st.warning("No hi ha dades de sondeig disponibles per a la selecció actual.")



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
    Versió corregida per a evitar l'error 'TokenError'.
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

    # Text per a la primera targeta (Convergència Frontal)
    text_card_1 = """
    <div class="explanation-card">
        <div class="explanation-title">
            <span class="explanation-icon">💥</span>
            Convergència Frontal (Xoc)
        </div>
        <div class="explanation-text">
            Passa quan <strong>dues masses d'aire de direccions diferents xoquen</strong>. L'aire no pot anar cap als costats i es veu forçat a ascendir bruscament.
            <br><br>
            <strong>Al mapa:</strong> Busca línies on les <i>streamlines</i> (línies de vent) es troben de cara. Són mecanismes de dispar molt eficients i solen generar tempestes organitzades.
        </div>
    </div>
    """

    # Text per a la segona targeta (Convergència per Acumulació)
    text_card_2 = """
    <div class="explanation-card">
        <div class="explanation-title">
            <span class="explanation-icon">⛰️</span>
            Convergència per Acumulació
        </div>
        <div class="explanation-text">
            Ocorre quan el vent es troba amb un <strong>obstacle (com una muntanya) o es desaccelera</strong>, fent que l'aire "s'amuntegui". L'única sortida per a aquesta acumulació de massa és cap amunt.
            <br><br>
            <strong>Al mapa:</strong> Busca zones on les <i>streamlines</i> s'ajunten i la velocitat del vent disminueix. És com un "embús a l'autopista".
        </div>
    </div>
    """
    
    with col1:
        st.markdown(text_card_1, unsafe_allow_html=True)

    with col2:
        st.markdown(text_card_2, unsafe_allow_html=True)


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
    Versió millorada que carrega també el CAPE per als mapes combinats.
    """
    try:
        # AFEGIM 'cape' A LA LLISTA DE VARIABLES
        variables_base = ["temperature_2m", "dew_point_2m", "cape"]
        
        variables_nivell = [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        if nivell < 950:
            variables_nivell.extend([f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa"])

        map_data_raw, error = carregar_dades_mapa_base_cat(variables_base + variables_nivell, hourly_index)
        if error: return None, error

        # Extraiem el CAPE i el guardem
        map_data_raw['cape_data'] = map_data_raw.pop('cape')
        
        map_data_raw['sfc_temp_data'] = map_data_raw.pop('temperature_2m')
        map_data_raw['sfc_dewpoint_data'] = map_data_raw.pop('dew_point_2m')

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







def crear_mapa_forecast_combinat_cat(lons: np.ndarray, lats: np.ndarray, speed_data: np.ndarray, 
                                     dir_data: np.ndarray, dewpoint_data: np.ndarray, cape_data: np.ndarray, 
                                     nivell: int, timestamp_str: str, map_extent: List[float],
                                     cape_min_filter: int, cape_max_filter: int, convergence_min_filter: int) -> plt.Figure:
    """
    VERSIÓ 29.0 (FINAL): Afegeix una categoria "Fluixa" amb línies blanques
    puntejades i mostra els marcadors d'intensitat només a partir de la
    categoria "Comuna" per a un mapa més net i professional.
    """
    plt.style.use('default')
    fig, ax = crear_mapa_base(map_extent)
    
    # --- 1. PREPARACIÓ DE DADES ---
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 300), 
                                     np.linspace(map_extent[2], map_extent[3], 300))
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'linear')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_cape = np.nan_to_num(griddata((lons, lats), cape_data, (grid_lon, grid_lat), 'linear'))

    # --- 2. DIBUIX DEL CAPE DE FONS ---
    cfg_cape = MAP_CONFIG['cape']
    cmap_cape = ListedColormap(cfg_cape['colors'])
    norm_cape = BoundaryNorm(cfg_cape['levels'], ncolors=cmap_cape.N, clip=True)
    cape_mesh = ax.pcolormesh(grid_lon, grid_lat, grid_cape, 
                               cmap=cmap_cape, norm=norm_cape, 
                               alpha=cfg_cape['alpha'], zorder=2, transform=ccrs.PlateCarree())
    cbar_cape = fig.colorbar(cape_mesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.02, ticks=cfg_cape['cbar_ticks'])
    cbar_cape.set_label("CAPE (J/kg) - 'Combustible'")
    cbar_cape.ax.tick_params(labelsize=8)

    # --- 3. CÀLCUL I FILTRATGE DE CONVERGÈNCIA ---
    with np.errstate(invalid='ignore'):
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        convergence = (-(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s')).magnitude * 1e5
    
    dewpoint_thresh = MAP_CONFIG['thresholds']['dewpoint_low_level'] if nivell >= 950 else MAP_CONFIG['thresholds']['dewpoint_mid_level']
    humid_mask = grid_dewpoint >= dewpoint_thresh
    cape_mask = (grid_cape >= cape_min_filter) & (grid_cape <= cape_max_filter)
    effective_convergence = np.where((convergence >= convergence_min_filter) & humid_mask & cape_mask, convergence, 0)
    smoothed_convergence = gaussian_filter(effective_convergence, sigma=MAP_CONFIG['convergence']['sigma_filter'])
    smoothed_convergence[smoothed_convergence < convergence_min_filter] = 0

    # --- 4. DIBUIX DE LA CONVERGÈNCIA I ELS MARCADORS ADAPTATIUS ---
    if np.any(smoothed_convergence > 0):
        cfg_conv = MAP_CONFIG['convergence']
        
        # Dibuixem les línies de contorn
        for category_name, style in cfg_conv['styles'].items():
            if category_name == 'Fluixa': line_style = ':'
            elif category_name == 'Comuna': line_style = '--'
            else: line_style = '-'
            
            ax.contour(grid_lon, grid_lat, smoothed_convergence, 
                       levels=style['levels'], 
                       colors=style['color'] if category_name == 'Fluixa' else 'black', 
                       linewidths=style['width'], 
                       linestyles=line_style, 
                       zorder=4,
                       transform=ccrs.PlateCarree())
        
        # Dibuixem els marcadors només per a les categories rellevants
        labeled_array, num_features = ndi.label(smoothed_convergence >= 10) # Comencem a buscar a partir de 10
        for i in range(1, num_features + 1):
            blob_mask = (labeled_array == i)
            max_conv_in_blob = smoothed_convergence[blob_mask].max()
            
            # Comprovem a quina categoria pertany el focus
            category_of_blob = None
            marker_color = '#FFFFFF'
            for name, style in reversed(list(cfg_conv['styles'].items())):
                if max_conv_in_blob >= style['levels'][0]:
                    category_of_blob = name
                    marker_color = style['color']
                    break
            
            # <<<--- NOU FILTRE: Només dibuixem si la categoria NO és "Fluixa" ---
            if category_of_blob and category_of_blob != 'Fluixa':
                temp_grid = np.where(blob_mask, smoothed_convergence, 0)
                max_idx = np.unravel_index(np.argmax(temp_grid), temp_grid.shape)
                max_lon, max_lat = grid_lon[max_idx], grid_lat[max_idx]
                
                blob_area = np.sum(blob_mask)
                marker_scale = 0.6 if blob_area < 200 else (0.8 if blob_area < 800 else 1.0)
                
                ax.plot(max_lon, max_lat, 's', color=marker_color, markersize=8 * marker_scale, markeredgecolor='black', 
                        markeredgewidth=1.5 * marker_scale, transform=ccrs.PlateCarree(), zorder=13)
                line_len = 0.08 * marker_scale; line_width = 2.5 * marker_scale
                ax.plot([max_lon - line_len, max_lon - 0.015 * marker_scale], [max_lat, max_lat], color='black', linewidth=line_width, 
                        transform=ccrs.PlateCarree(), zorder=12, solid_capstyle='butt')
                ax.plot([max_lon + 0.015 * marker_scale, max_lon + line_len], [max_lat, max_lat], color='black', linewidth=line_width, 
                        transform=ccrs.PlateCarree(), zorder=12, solid_capstyle='butt')

    # --- 5. LLEGENDA, STREAMLINES, TÍTOL I ETIQUETES ---
    legend_handles = []
    for label, style in MAP_CONFIG['convergence']['styles'].items():
        if label == 'Fluixa': line_style = ':'
        elif label == 'Comuna': line_style = '--'
        else: line_style = '-'
        handle = mlines.Line2D([], [], color=style['color'] if label == 'Fluixa' else 'black', 
                               marker='s' if label != 'Fluixa' else 'None', 
                               markerfacecolor=style['color'],
                               markersize=10 if label != 'Fluixa' else 0, 
                               linestyle=line_style, 
                               linewidth=2, 
                               label=label)
        legend_handles.append(handle)
        
    ax.legend(handles=legend_handles, title="Convergència", loc='lower right', 
              fontsize=9, title_fontsize=11, frameon=True, framealpha=0.9, facecolor='white')

    cfg_stream = MAP_CONFIG['streamlines']
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color=cfg_stream['color'], 
                  linewidth=cfg_stream['linewidth'], density=cfg_stream['density'], 
                  arrowsize=cfg_stream['arrowsize'], zorder=5, transform=ccrs.PlateCarree())
    
    ax.set_title(f"CAPE (fons), Convergència (línies) i Vent a {nivell}hPa\n{timestamp_str}",
                 weight='bold', fontsize=14)
    afegir_etiquetes_ciutats(ax, map_extent)
    
    return fig

    
    

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




@st.cache_data(show_spinner="Carregant mapa de selecció de la península...")
def carregar_dades_geografiques_peninsula():
    """ Carrega el fitxer GeoJSON amb les geometries de les zones de la península. """
    try:
        gdf_zones = gpd.read_file("peninsula_zones.geojson")
        # --- LÍNIA CORREGIDA ---
        # En lloc de transformar (to_crs), definim (set_crs) el sistema de coordenades.
        gdf_zones = gdf_zones.set_crs("EPSG:4326", allow_override=True)
        return gdf_zones
    except Exception as e:
        st.error(f"Error crític: No s'ha pogut carregar l'arxiu 'peninsula_zones.geojson'. Assegura't que existeix. Detall: {e}")
        return None
    

@st.cache_data(ttl=1800, show_spinner="Analitzant focus de convergència a la península...")
def calcular_alertes_per_zona_peninsula(hourly_index, nivell):
    """
    Calcula els valors màxims de convergència per a cada zona de la península.
    (Versió amb llindar de detecció ajustat a 10)
    """
    # --- LLINDAR CORREGIT ---
    CONV_THRESHOLD = 10 # Rebaixem el llindar per detectar focus d'interès més febles.
    
    map_data, error = carregar_dades_mapa_est_peninsula(nivell, hourly_index)
    gdf_zones = carregar_dades_geografiques_peninsula()

    if error or not map_data or gdf_zones is None:
        return {}

    try:
        property_name = 'NAME_2' # Assegura't que aquest sigui el nom correcte
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
            
        max_conv_per_zona = punts_dins_zones.groupby(property_name)['value'].max()
        return max_conv_per_zona.to_dict()
        
    except Exception as e:
        print(f"Error dins de calcular_alertes_per_zona_peninsula: {e}")
        return {}

def seleccionar_poble_peninsula(nom_poble):
    """Callback per seleccionar un poble a la zona de la península."""
    st.session_state.poble_selector_est_peninsula = nom_poble
    if 'active_tab_est_peninsula_index' in st.session_state:
        st.session_state.active_tab_est_peninsula_index = 0

def tornar_a_seleccio_zona_peninsula():
    """Callback per tornar a la llista de municipis de la zona seleccionada a la península."""
    st.session_state.poble_selector_est_peninsula = "--- Selecciona una localitat ---"
    if 'active_tab_est_peninsula_index' in st.session_state:
        st.session_state.active_tab_est_peninsula_index = 0

def tornar_al_mapa_general_peninsula():
    """Callback per tornar al mapa general de la península."""
    st.session_state.poble_selector_est_peninsula = "--- Selecciona una localitat ---"
    st.session_state.selected_area_peninsula = "--- Selecciona una zona al mapa ---"
    if 'active_tab_est_peninsula_index' in st.session_state:
        st.session_state.active_tab_est_peninsula_index = 0


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
    Versió final i robusta que busca automàticament el mapa personalitzat
    i, si no el troba, utilitza el mapa de comarques per defecte.
    """
    noms_possibles = ["mapes_personalitzat.geojson", "comarques.geojson"]
    file_to_load = None
    for file in noms_possibles:
        if os.path.exists(file):
            file_to_load = file
            break

    if file_to_load is None:
        st.error(
            "**Error Crític: Mapa no trobat.**\n\n"
            "No s'ha trobat l'arxiu `mapa_personalitzat.geojson` ni `comarques.geojson` a la carpeta de l'aplicació. "
            "Assegura't que almenys un d'aquests dos arxius existeixi."
        )
        return None

    try:
        gdf = gpd.read_file(file_to_load)
        # --- LÍNIA CORREGIDA ---
        # Apliquem la mateixa solució que a la península per a més robustesa.
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return gdf
    except Exception as e:
        st.error(f"S'ha produït un error en carregar l'arxiu de mapa '{file_to_load}': {e}")
        return None





def ui_mapa_display_personalitzat(alertes_per_zona, hourly_index, show_labels):
    """
    Funció de VISUALITZACIÓ que mostra el mapa interactiu de Folium.
    """
    st.markdown("#### Mapa de Situació")
    
    selected_area_str = st.session_state.get('selected_area')
    
    alertes_tuple = tuple(sorted(alertes_per_zona.items(), key=lambda item: str(item[0])))
    
    map_data = preparar_dades_mapa_cachejat(
        alertes_tuple=alertes_tuple, 
        selected_area_str=selected_area_str, 
        show_labels=show_labels
    )
    
    if not map_data:
        st.error("No s'han pogut generar les dades per al mapa.")
        return None

    # Paràmetres base del mapa
    map_params = {
        "location": [41.83, 1.87], 
        
        # ===== LÍNIA MODIFICADA AQUÍ (VISTA GENERAL) =====
        "zoom_start": 7,  # Hem canviat de 8 a 7 per allunyar el mapa
        # ===============================================

        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; and the GIS User Community",
        "scrollWheelZoom": False,
        "dragging": False,
        "zoom_control": False,
        "doubleClickZoom": False,
        "max_bounds": [[40.4, 0.0], [42.9, 3.5]], 
        
        # ===== LÍNIA MODIFICADA AQUÍ (VISTA GENERAL) =====
        "min_zoom": 7, "max_zoom": 7 # Fixem el zoom al nou nivell 7
        # ===============================================
    }

    # Si hi ha una zona seleccionada, fem zoom i congelem el mapa
    if selected_area_str and "---" not in selected_area_str:
        gdf_temp = gpd.read_file(map_data["gdf"])
        cleaned_selected_area = selected_area_str.strip().replace('.', '')
        zona_shape = gdf_temp[gdf_temp[map_data["property_name"]].str.strip().replace('.', '') == cleaned_selected_area]
        if not zona_shape.empty:
            centroid = zona_shape.geometry.centroid.iloc[0]
            map_params.update({
                "location": [centroid.y, centroid.x], 

                # ===== LÍNIA MODIFICADA AQUÍ (VISTA COMARCA) =====
                "zoom_start": 9, # Hem canviat de 10 a 9 per allunyar una mica la vista de comarca
                # =================================================

                "max_bounds": [[zona_shape.total_bounds[1], zona_shape.total_bounds[0]], [zona_shape.total_bounds[3], zona_shape.total_bounds[2]]],

                # ===== LÍNIA MODIFICADA AQUÍ (VISTA COMARCA) =====
                "min_zoom": 9, "max_zoom": 9 # Fixem el zoom al nou nivell 9
                # =================================================
            })

    m = folium.Map(**map_params)

    def style_function(feature):
        nom_feature_raw = feature.get('properties', {}).get(map_data["property_name"])
        style = {'fillColor': '#6c757d', 'color': '#495057', 'weight': 1, 'fillOpacity': 0.25}
        if nom_feature_raw:
            nom_feature = nom_feature_raw.strip().replace('.', '')
            style = map_data["styles"].get(nom_feature, style)
            cleaned_selected_area = selected_area_str.strip().replace('.', '') if selected_area_str else ''
            if nom_feature == cleaned_selected_area:
                style.update({'fillColor': '#007bff', 'color': '#ffffff', 'weight': 3, 'fillOpacity': 0.5})
        return style

    folium.GeoJson(
        map_data["gdf"], style_function=style_function,
        highlight_function=lambda x: {'color': '#ffffff', 'weight': 3.5, 'fillOpacity': 0.5},
        tooltip=folium.GeoJsonTooltip(fields=[map_data["property_name"]], aliases=['Zona:'])
    ).add_to(m)

    for marker in map_data["markers"]:
        icon = folium.DivIcon(html=marker['icon_html'])
        folium.Marker(location=marker['location'], icon=icon, tooltip=marker['tooltip']).add_to(m)
    
    return st_folium(m, width="100%", height=650, returned_objects=['last_object_clicked_tooltip'])


    
@st.cache_data(show_spinner="Carregant geometries municipals...")
def carregar_dades_municipis():
    """
    Carrega el fitxer GeoJSON amb les geometries de tots els municipis de Catalunya.
    """
    try:
        # Aquesta línia llegeix el teu arxiu de municipis
        gdf_municipis = gpd.read_file("municipis.geojson")
        gdf_municipis = gdf_municipis.to_crs("EPSG:4326")
        # Convertim el codi de comarca a número per assegurar la compatibilitat
        gdf_municipis['comarca'] = pd.to_numeric(gdf_municipis['comarca'])
        return gdf_municipis
    except Exception as e:
        st.error(f"Error crític: No s'ha pogut carregar l'arxiu 'municipis.geojson'. Assegura't que existeix. Detall: {e}")
        return None

@st.cache_data(show_spinner="Carregant mapa de selecció...")
def carregar_dades_geografiques():
    """
    Versió final i robusta que busca automàticament el mapa personalitzat
    i, si no el troba, utilitza el mapa de comarques per defecte.
    """
    noms_possibles = ["mapes_personalitzat.geojson", "comarques.geojson"]
    file_to_load = None
    for file in noms_possibles:
        if os.path.exists(file):
            file_to_load = file
            break

    if file_to_load is None:
        st.error(
            "**Error Crític: Mapa no trobat.**\n\n"
            "No s'ha trobat l'arxiu `mapa_personalitzat.geojson` ni `comarques.geojson` a la carpeta de l'aplicació. "
            "Assegura't que almenys un d'aquests dos arxius existeixi."
        )
        return None

    try:
        gdf = gpd.read_file(file_to_load)
        # --- LÍNIA CORREGIDA ---
        # Apliquem la mateixa solució que a la península per a més robustesa.
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return gdf
    except Exception as e:
        st.error(f"S'ha produït un error en carregar l'arxiu de mapa '{file_to_load}': {e}")
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



@st.cache_data(ttl=1800, show_spinner="Analitzant focus de tempesta (CAPE + Convergència)...")
def calcular_alertes_per_comarca(hourly_index, nivell):
    """
    VERSIÓ MILLORADA: Calcula les alertes basant-se en el CAPE trobat
    al punt de MÀXIMA CONVERGÈNCIA de cada comarca.
    Retorna un diccionari: {'Comarca': {'conv': valor, 'cape': valor}}
    """
    CONV_THRESHOLD = 15
    
    map_data, error = carregar_dades_mapa_cat(nivell, hourly_index)
    gdf_zones = carregar_dades_geografiques()

    if error or not map_data or gdf_zones is None:
        return {}

    try:
        property_name = next((prop for prop in ['nom_zona', 'nom_comar', 'nomcomar'] if prop in gdf_zones.columns), 'nom_comar')
        
        lons, lats = map_data['lons'], map_data['lats']
        grid_lon, grid_lat = np.meshgrid(np.linspace(min(lons), max(lons), 150), np.linspace(min(lats), max(lats), 150))
        
        grid_cape = griddata((lons, lats), map_data['cape_data'], (grid_lon, grid_lat), 'linear')
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
        conv_vals = convergence_scaled[punts_calents_idx[:, 0], punts_calents_idx[:, 1]]
        cape_vals = grid_cape[punts_calents_idx[:, 0], punts_calents_idx[:, 1]]
        
        gdf_punts = gpd.GeoDataFrame(
            {'conv': conv_vals, 'cape': cape_vals}, 
            geometry=[Point(lon, lat) for lon, lat in zip(punts_lons, punts_lats)], 
            crs="EPSG:4326"
        )
        
        punts_dins_zones = gpd.sjoin(gdf_punts, gdf_zones, how="inner", predicate="within")
        if punts_dins_zones.empty: return {}

        alertes = {}
        for name, group in punts_dins_zones.groupby(property_name):
            max_conv_row = group.loc[group['conv'].idxmax()]
            alertes[name] = {
                'conv': max_conv_row['conv'],
                'cape': max_conv_row['cape'] if pd.notna(max_conv_row['cape']) else 0
            }
        
        return alertes
        
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



def get_emoji_for_cape(cape_value):
    """Retorna un emoji de color basat en el valor de CAPE."""
    if not isinstance(cape_value, (int, float, np.number)) or pd.isna(cape_value) or cape_value < 500:
        return "⚪"  # Gris/Blanc (Baix)
    if cape_value < 1000:
        return "🟢"  # Verd (Moderat)
    if cape_value < 2000:
        return "🟡"  # Groc (Alt)
    if cape_value < 3000:
        return "🟠"  # Taronja (Molt Alt)
    return "🔴"      # Vermell (Extrem)

def trobar_poblacions_properes_a_convergencia(smoothed_convergence, grid_lon, grid_lat, grid_cape, poblacions_dict, conv_llindar=20, cape_llindar=100):
    """
    Analitza mapes de convergència i CAPE per trobar els focus més rellevants.
    Retorna una llista de diccionaris {'poble': str, 'conv': float, 'cape': float}
    ordenada per un índex de perillositat.
    """
    focus_mask = (smoothed_convergence >= conv_llindar) & (grid_cape >= cape_llindar)
    labeled_array, num_features = label(focus_mask)
    if num_features == 0:
        return []

    focus_list = []
    
    for i in range(1, num_features + 1):
        blob_mask = (labeled_array == i)
        
        # Trobem el punt de màxima convergència dins del blob
        temp_grid = np.where(blob_mask, smoothed_convergence, 0)
        max_idx = np.unravel_index(np.argmax(temp_grid), temp_grid.shape)
        max_lon, max_lat = grid_lon[max_idx], grid_lat[max_idx]
        max_conv_in_blob = smoothed_convergence[max_idx]
        cape_at_max_conv = grid_cape[max_idx]

        poble_mes_proper = min(
            poblacions_dict.keys(),
            key=lambda poble: haversine_distance(max_lat, max_lon, poblacions_dict[poble]['lat'], poblacions_dict[poble]['lon'])
        )
        
        focus_list.append({
            "poble": poble_mes_proper,
            "conv": max_conv_in_blob,
            "cape": cape_at_max_conv
        })

    # Eliminem duplicats, conservant el focus de major convergència per a cada poble
    focus_dict = {}
    for focus in focus_list:
        poble = focus["poble"]
        if poble not in focus_dict or focus["conv"] > focus_dict[poble]["conv"]:
            focus_dict[poble] = focus

    # Ordenem la llista final per un índex de perillositat (conv * cape)
    sorted_focuses = sorted(list(focus_dict.values()), key=lambda f: f['conv'] * (f['cape'] + 1), reverse=True)
    
    return sorted_focuses



def on_focus_select():
    """
    Callback que s'activa en seleccionar un focus de tempesta.
    Extreu el nom del poble del diccionari seleccionat i actualitza l'estat.
    """
    focus_seleccionat = st.session_state.get("focus_selector_widget")
    # Comprovem que no sigui el placeholder i que sigui un diccionari
    if focus_seleccionat and isinstance(focus_seleccionat, dict):
        poble = focus_seleccionat.get("poble")
        if poble:
            st.session_state.poble_sel = poble


        
def ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str):
    """
    Gestiona la interfície de la pestanya "Anàlisi de Mapes" per a Catalunya.
    Versió 5.0 (Completa i Funcional):
    - Inclou tots els selectors (Zoom, Nivell, Focus) en un disseny de 3 columnes.
    - El selector "Focus de Tempesta" és visualment ric i coherent amb els filtres aplicats.
    - Restaura i gestiona correctament la selecció de nivell de pressió per a tots els mapes.
    """
    st.markdown("#### Mapes de Pronòstic (Model AROME)")
    
    mapa_sel = st.selectbox("Selecciona la capa del mapa:", 
                           ["Anàlisi de Vent i Convergència", "Anàlisi d'Advecció (Fronts)", "Vent a 700hPa", "Vent a 300hPa"], 
                           key="map_cat")
    
    # --- DISSENY AMB 3 COLUMNES PER ALS CONTROLS ---
    col_zoom, col_nivell, col_focus = st.columns(3)
    
    with col_zoom: 
        zoom_sel = st.selectbox("Nivell de Zoom:", options=list(MAP_ZOOM_LEVELS_CAT.keys()), key="zoom_cat")
    
    # La segona columna és dinàmica per al selector de nivell
    with col_nivell:
        if "Convergència" in mapa_sel:
            nivell_sel = st.selectbox(
                "Nivell d'Anàlisi:", 
                options=[1000, 950, 925, 900, 850, 800, 700], 
                key="level_cat_map", 
                index=2, # Manté 925hPa per defecte
                format_func=lambda x: f"{x} hPa"
            )
        elif "Advecció" in mapa_sel:
            nivell_sel = st.selectbox(
                "Nivell d'Advecció:",
                options=[1000, 925, 850, 700, 500],
                key="advection_level_selector_tab",
                format_func=lambda x: f"{x} hPa"
            )
        else: # Per als mapes de només vent, no cal selector aquí
            nivell_sel = 700 if "700" in mapa_sel else 300
            st.empty() # Deixem la columna buida per a mantenir l'alineació

    # El filtre de CAPE es mostra a sota de les columnes principals
    cape_min_seleccionat = 100 # Valor per defecte
    if "Convergència" in mapa_sel:
        FILTRES_CAPE = {"Detectar totes les convergències amb CAPE (>100)": 100, "Detectar convergències amb CAPE Significatiu (>500)": 500, "Detectar convergències amb Alt CAPE (>1000)": 1000, "Detectar convergències amb Molt de CAPE (>2000)": 2000}
        filtre_sel = st.selectbox("Filtre de CAPE per a la Convergència:", options=list(FILTRES_CAPE.keys()), key="cape_filter_cat")
        cape_min_seleccionat = FILTRES_CAPE[filtre_sel]

    # Anàlisi prèvia per a obtenir els focus de tempesta
    pobles_focus = []
    map_data_raw, error_map = carregar_dades_mapa_cat(nivell_sel, hourly_index_sel)
    
    if "Convergència" in mapa_sel and not error_map and map_data_raw:
        selected_extent = MAP_ZOOM_LEVELS_CAT.get(zoom_sel, MAP_ZOOM_LEVELS_CAT["Catalunya (Complet)"])
        grid_lon, grid_lat = np.meshgrid(np.linspace(selected_extent[0], selected_extent[1], 150), np.linspace(selected_extent[2], selected_extent[3], 150))
        
        # Interpolem totes les dades necessàries
        u_comp, v_comp = mpcalc.wind_components(np.array(map_data_raw['speed_data']) * units('km/h'), np.array(map_data_raw['dir_data']) * units.degrees)
        grid_u = griddata((map_data_raw['lons'], map_data_raw['lats']), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear'); grid_v = griddata((map_data_raw['lons'], map_data_raw['lats']), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        convergence = (-(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s')).magnitude * 1e5
        smoothed_convergence = gaussian_filter(np.nan_to_num(convergence), sigma=2.5)
        grid_cape = np.nan_to_num(griddata((map_data_raw['lons'], map_data_raw['lats']), map_data_raw['cape_data'], (grid_lon, grid_lat), 'linear'))
        
        pobles_focus = trobar_poblacions_properes_a_convergencia(smoothed_convergence, grid_lon, grid_lat, grid_cape, CIUTATS_CATALUNYA, conv_llindar=20, cape_llindar=cape_min_seleccionat)

    with col_focus:
        if "Convergència" in mapa_sel:
            st.selectbox(
                "Focus de Tempesta:", 
                options=[{"poble": "--- Viatge Ràpid ---"}] + pobles_focus, 
                key="focus_selector_widget",
                on_change=on_focus_select,
                format_func=lambda focus: f"{get_emoji_for_cape(focus.get('cape', 0))} {focus['poble']} (Conv: {focus.get('conv', 0):.0f})" if focus['poble'] != "--- Viatge Ràpid ---" else "--- Viatge Ràpid ---",
                help="Viatja directament a l'anàlisi de la població més propera a un focus de convergència que compleix els teus filtres."
            )
        else:
            st.empty() # Mantenim l'alineació
            
    selected_extent = MAP_ZOOM_LEVELS_CAT[zoom_sel]
    
    with st.spinner(f"Carregant i generant mapa..."):
        if "Convergència" in mapa_sel:
            if error_map or not map_data_raw:
                st.error(f"Error en carregar les dades per al mapa: {error_map}")
            else:
                fig = crear_mapa_forecast_combinat_cat(map_data_raw['lons'], map_data_raw['lats'], map_data_raw['speed_data'], map_data_raw['dir_data'], map_data_raw['dewpoint_data'], map_data_raw['cape_data'], nivell_sel, timestamp_str, selected_extent, cape_min_seleccionat, 6000, 1)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        
        elif "Advecció" in mapa_sel:
            map_data_adv, error_adv = carregar_dades_mapa_adveccio_cat(nivell_sel, hourly_index_sel)
            if error_adv or not map_data_adv:
                st.error(f"Error en carregar les dades d'advecció: {error_adv}")
            else:
                timestamp_str_mapa = timestamp_str.split('|')[1].strip() if '|' in timestamp_str else timestamp_str
                fig_adv = crear_mapa_adveccio_cat(map_data_adv['lons'], map_data_adv['lats'], map_data_adv['temp_data'], map_data_adv['speed_data'], map_data_adv['dir_data'], nivell_sel, timestamp_str_mapa, selected_extent)
                st.pyplot(fig_adv, use_container_width=True)
                plt.close(fig_adv)
                ui_explicacio_adveccio()
        
        else: # Mapes de només vent
            fig = generar_mapa_vents_cachejat_cat(hourly_index_sel, nivell_sel, timestamp_str, tuple(selected_extent))
            if fig is None:
                st.error(f"Error en carregar les dades per al mapa de vent a {nivell_sel}hPa.")
            else:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    if "Convergència" in mapa_sel:
        ui_explicacio_convergencia()

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







def get_comarca_for_poble(poble_name):
    """
    Troba la comarca OFICIAL a la qual pertany un municipi.
    Això garanteix que sempre tindrem un nom de geometria vàlid per al mapa.
    """
    for comarca, pobles in CIUTATS_PER_COMARCA.items():
        if poble_name in pobles:
            return comarca
    return None

def generar_icona_direccio(color, direccio_graus):
    """
    Crea una icona visual (cercle + fletxa) per a la llegenda del mapa comarcal.
    Retorna una cadena d'imatge en format Base64.
    """
    fig, ax = plt.subplots(figsize=(1, 1), dpi=72)
    fig.patch.set_alpha(0)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Dibuixa el cercle
    cercle = Circle((0.5, 0.5), 0.4, facecolor='none', edgecolor=color, linewidth=4,
                    path_effects=[path_effects.withStroke(linewidth=6, foreground='black')])
    ax.add_patch(cercle)

    # Dibuixa la fletxa de direcció
    angle_rad = np.deg2rad(90 - direccio_graus)
    ax.arrow(0.5, 0.5, 0.3 * np.cos(angle_rad), 0.3 * np.sin(angle_rad),
             head_width=0.15, head_length=0.1, fc=color, ec=color,
             length_includes_head=True, zorder=10,
             path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])

    # Converteix la figura a imatge Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()





def generar_icona_direccio(color, direccio_graus):
    """
    Versió millorada que crea icones minimalistes (cercle fi + fletxa)
    similars als de la imatge de referència.
    """
    fig, ax = plt.subplots(figsize=(1, 1), dpi=72)
    fig.patch.set_alpha(0)
    ax.set_aspect('equal')
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')

    # Dibuixa el cercle fi
    cercle = Circle((0, 0), 0.5, facecolor='none', edgecolor=color, linewidth=2)
    ax.add_patch(cercle)

    # Dibuixa la fletxa interior
    angle_rad = np.deg2rad(90 - direccio_graus)
    ax.arrow(-0.3 * np.cos(angle_rad), -0.3 * np.sin(angle_rad), # Comença des del costat oposat
             0.5 * np.cos(angle_rad), 0.5 * np.sin(angle_rad),   # Dibuixa a través del centre
             head_width=0.15, head_length=0.1, fc=color, ec=color,
             length_includes_head=True, zorder=10)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()




def crear_llegenda_direccionalitat():
    """
    Mostra una llegenda visual i explicativa completa per al mapa de focus de tempesta,
    detallant l'ombrejat de CAPE, l'ombrejat de convergència i el marcador principal.
    """
    st.markdown("""
    <style>
        .legend-box { background-color: #2a2c34; border-radius: 10px; padding: 15px; border: 1px solid #444; margin-top: 15px; }
        .legend-title { font-size: 1.1em; font-weight: bold; color: #FAFAFA; margin-bottom: 12px; }
        .legend-section { display: flex; align-items: flex-start; margin-bottom: 12px; }
        .legend-icon-container { flex-shrink: 0; margin-right: 15px; width: 30px; text-align: center; font-size: 24px; padding-top: 2px; }
        .legend-text-container { flex-grow: 1; font-size: 0.9em; color: #a0a0b0; line-height: 1.4; }
        .legend-text-container b { color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

    html_llegenda = (
        f'<div class="legend-box">'
        f'    <div class="legend-title">Com Interpretar el Mapa de Focus</div>'
        
        f'    <div class="legend-section">'
        f'        <div class="legend-icon-container" style="color: #FFC107;">⚡️</div>'
        f'        <div class="legend-text-container">'
        f'            <b>Energia (CAPE):</b> Representat per l\'<b>ombrejat de fons verd-groc-vermell</b> i les <b>isolínies blanques</b>. Indica el "combustible" disponible per a les tempestes. Valors més alts (en J/kg) afavoreixen un creixement més violent.'
        f'        </div>'
        f'    </div>'
        
        f'    <div class="legend-section">'
        f'        <div class="legend-icon-container" style="color: #6495ED;">🌀</div>'
        f'        <div class="legend-text-container">'
        f'            <b>Disparador (Convergència):</b> Són les <b>àrees ombrejades de blau a vermell/groc</b>. Indiquen zones on l\'aire es veu forçat a ascendir, actuant com la "guspira" que pot iniciar la tempesta.'
        f'        </div>'
        f'    </div>'
        
        f'    <div class="legend-section">'
        f'        <div class="legend-icon-container" style="font-size: 18px;">🎯</div>'
        f'        <div class="legend-text-container">'
        f'            <b>Focus Principal (Cercle i Fletxa):</b> Marca el punt de <b>màxima convergència</b> dins la comarca i la seva <b>trajectòria més probable</b>. És la zona amb més potencial per iniciar la convecció. El color del cercle indica la seva intensitat.'
        f'        </div>'
        f'    </div>'

        f'</div>'
    )
    st.markdown(html_llegenda, unsafe_allow_html=True)



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




@st.cache_data(ttl=1800, max_entries=20, show_spinner=False)
def carregar_dades_sondeig_est_peninsula(lat, lon, hourly_index):
    """
    Carrega dades de sondeig per a l'Est Peninsular (Model AROME).
    És una còpia funcional de la versió per a Catalunya.
    """
    try:
        # La lògica és idèntica a la de Catalunya, ja que usem el mateix model (AROME)
        h_base = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]
        h_press = [f"{v}_{p}hPa" for v in ["temperature", "relative_humidity", "wind_speed", "wind_direction", "geopotential_height"] for p in PRESS_LEVELS_EST_PENINSULA]
        params = {"latitude": lat, "longitude": lon, "hourly": h_base + h_press, "models": "arome_seamless", "forecast_days": 4}
        response = openmeteo.weather_api(API_URL_EST_PENINSULA, params=params)[0]
        hourly = response.Hourly()

        valid_index = trobar_hora_valida_mes_propera(hourly, hourly_index, len(h_base))
        if valid_index is None: return None, hourly_index, "No s'han trobat dades vàlides."
        
        sfc_data = {v: hourly.Variables(i).ValuesAsNumpy()[valid_index] for i, v in enumerate(h_base)}
        
        p_data = {}
        var_count = len(h_base)
        for i, var in enumerate(["T", "RH", "WS", "WD", "H"]):
            p_data[var] = [hourly.Variables(var_count + i * len(PRESS_LEVELS_EST_PENINSULA) + j).ValuesAsNumpy()[valid_index] for j in range(len(PRESS_LEVELS_EST_PENINSULA))]
        
        sfc_u, sfc_v = mpcalc.wind_components(sfc_data["wind_speed_10m"] * units('km/h'), sfc_data["wind_direction_10m"] * units.degrees)
        
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [sfc_data["surface_pressure"]], [sfc_data["temperature_2m"]], [sfc_data["dew_point_2m"]], [sfc_u.to('m/s').m], [sfc_v.to('m/s').m], [0.0]
        
        for i, p_val in enumerate(PRESS_LEVELS_EST_PENINSULA):
            if p_val < p_profile[-1] and all(not np.isnan(p_data[v][i]) for v in ["T", "RH", "WS", "WD", "H"]):
                p_profile.append(p_val)
                T_profile.append(p_data["T"][i])
                Td_profile.append(mpcalc.dewpoint_from_relative_humidity(p_data["T"][i] * units.degC, p_data["RH"][i] * units.percent).m)
                u, v = mpcalc.wind_components(p_data["WS"][i] * units('km/h'), p_data["WD"][i] * units.degrees)
                u_profile.append(u.to('m/s').m); v_profile.append(v.to('m/s').m); h_profile.append(p_data["H"][i])

        processed_data, error = processar_dades_sondeig(p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile)
        return processed_data, valid_index, error
        
    except Exception as e: 
        return None, hourly_index, f"Error en carregar dades del sondeig AROME (Península): {e}"

@st.cache_data(ttl=3600)
def carregar_dades_mapa_est_peninsula(nivell, hourly_index):
    """
    Carrega les dades en una graella per al mapa de l'Est Peninsular.
    """
    try:
        variables = [f"temperature_{nivell}hPa", f"relative_humidity_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        lats, lons = np.linspace(MAP_EXTENT_EST_PENINSULA[2], MAP_EXTENT_EST_PENINSULA[3], 12), np.linspace(MAP_EXTENT_EST_PENINSULA[0], MAP_EXTENT_EST_PENINSULA[1], 12)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": variables, "models": "arome_seamless", "forecast_days": 2}
        
        responses = openmeteo.weather_api(API_URL_EST_PENINSULA, params=params)
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
        return None, f"Error en carregar dades del mapa de l'Est Peninsular: {e}"

def crear_mapa_forecast_combinat_est_peninsula(lons, lats, speed_data, dir_data, dewpoint_data, nivell, timestamp_str):
    """
    Crea el mapa visual de vent i convergència per a l'Est Peninsular,
    amb un estil visual millorat i professional, similar al de Catalunya.
    """
    # Usem un fons clar per a més claredat en els detalls
    plt.style.use('default')
    fig, ax = crear_mapa_base(MAP_EXTENT_EST_PENINSULA)
    
    if len(lons) < 4: 
        ax.set_title("Dades insuficients per generar el mapa")
        return fig

    # --- 1. INTERPOLACIÓ A GRAELLA D'ALTA RESOLUCIÓ ---
    grid_lon, grid_lat = np.meshgrid(np.linspace(MAP_EXTENT_EST_PENINSULA[0], MAP_EXTENT_EST_PENINSULA[1], 300), np.linspace(MAP_EXTENT_EST_PENINSULA[2], MAP_EXTENT_EST_PENINSULA[3], 300))
    grid_dewpoint = griddata((lons, lats), dewpoint_data, (grid_lon, grid_lat), 'linear')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'linear')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    
    # --- 2. MAPA DE VELOCITAT DEL VENT (FONS) ---
    colors_wind = ['#d2d2f0', '#b4b4e6', '#78c8c8', '#50b48c', '#32cd32', '#64ff64', '#ffff00', '#f5d264', '#e6b478', '#d7788c', '#ff69b4', '#9f78dc']
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    custom_cmap = ListedColormap(colors_wind); norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree(), alpha=0.7)
    
    # --- 3. LÍNIES DE CORRENT (STREAMLINES) ---
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.5, density=6.5, arrowsize=0.3, zorder=4, transform=ccrs.PlateCarree())
    
    # --- 4. CÀLCUL, FILTRATGE I SUAVITZAT DE LA CONVERGÈNCIA ---
    with np.errstate(invalid='ignore'):
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        convergence = (-(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s')).magnitude * 1e5
        convergence[np.isnan(convergence)] = 0
        DEWPOINT_THRESHOLD = 14
        humid_mask = grid_dewpoint >= DEWPOINT_THRESHOLD
        effective_convergence = np.where((convergence >= 15) & humid_mask, convergence, 0)

    # Suavitzem els resultats per a una visualització més natural
    smoothed_convergence = gaussian_filter(effective_convergence, sigma=2.5)
    smoothed_convergence[smoothed_convergence < 15] = 0
    
    # --- 5. DIBUIX DELS FOCUS DE CONVERGÈNCIA ---
    if np.any(smoothed_convergence > 0):
        fill_levels = [15, 25, 40, 60, 80, 100]
        cmap = plt.get_cmap('plasma')
        norm = BoundaryNorm(fill_levels, ncolors=cmap.N, clip=True)

        ax.contourf(grid_lon, grid_lat, smoothed_convergence, 
                    levels=fill_levels, cmap=cmap, norm=norm, 
                    alpha=0.7, zorder=5, transform=ccrs.PlateCarree(), extend='max')

        line_levels = [20, 40, 60]
        contours = ax.contour(grid_lon, grid_lat, smoothed_convergence, 
                              levels=line_levels, colors='black', 
                              linestyles='--', linewidths=0.8, alpha=0.8, 
                              zorder=6, transform=ccrs.PlateCarree())
        
        labels = ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
        for label in labels:
            label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # --- 6. ETIQUETES DE CIUTATS ---
    for city, coords in CIUTATS_EST_PENINSULA.items():
        ax.plot(coords['lon'], coords['lat'], 'o', color='black', markersize=3, markeredgecolor='white', transform=ccrs.PlateCarree(), zorder=10)
        ax.text(coords['lon'] + 0.05, coords['lat'] + 0.05, city, fontsize=8, color='white', transform=ccrs.PlateCarree(), zorder=11,
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])

    ax.set_title(f"Vent i Nuclis de Convergència a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    return fig

def run_est_peninsula_app():
    """
    Funció principal per a l'Est Peninsular, amb la navegació dins de la vista detallada.
    """
    # --- PAS 1: GESTIÓ D'ESTAT INICIAL ---
    if 'selected_area_peninsula' not in st.session_state: st.session_state.selected_area_peninsula = "--- Selecciona una província al mapa ---"
    if 'poble_selector_est_peninsula' not in st.session_state: st.session_state.poble_selector_est_peninsula = "--- Selecciona una localitat ---"
    
    # --- PAS 2: CAPÇALERA I NAVEGACIÓ GLOBAL ---
    ui_capcalera_selectors(None, zona_activa="est_peninsula")

    # --- PAS 3: CÀLCUL DE LA DATA I HORA AMB EL SLIDER ---
    now_local = datetime.now(TIMEZONE_EST_PENINSULA)
    now_hour = now_local.hour
    
    time_options = list(range(-4, 9))
    time_labels = [f"Ara ({now_hour:02d}:00h)" if offset == 0 else f"Ara {'+' if offset > 0 else ''}{offset}h ({(now_hour + offset) % 24:02d}:00h)" for offset in time_options]
    
    selected_label = st.select_slider("Selector d'Hora:", options=time_labels, value=f"Ara ({now_hour:02d}:00h)", key="time_selector_peninsula")
    
    time_offset = time_options[time_labels.index(selected_label)]
    target_dt = (now_local + timedelta(hours=time_offset)).replace(minute=0, second=0, microsecond=0)
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((target_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)
    nivell_sel = 925

    # --- PAS 4: LÒGICA PRINCIPAL (VISTA DETALLADA O VISTA DE MAPA) ---
    if st.session_state.poble_selector_est_peninsula and "---" not in st.session_state.poble_selector_est_peninsula:
        # --- VISTA D'ANÀLISI DETALLADA D'UNA CIUTAT ---
        poble_sel = st.session_state.poble_selector_est_peninsula
        st.success(f"### Anàlisi per a: {poble_sel}")
        
        # --- NOU: Botons de navegació interns a la vista detallada ---
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            st.button("⬅️ Tornar a la Província", on_click=tornar_a_seleccio_zona_peninsula, use_container_width=True)
        with col_nav2:
            st.button("🗺️ Tornar al Mapa General", on_click=tornar_al_mapa_general_peninsula, use_container_width=True)

        lat_sel, lon_sel = CIUTATS_EST_PENINSULA[poble_sel]['lat'], CIUTATS_EST_PENINSULA[poble_sel]['lon']
        cat_dt = target_dt.astimezone(TIMEZONE_CAT)
        timestamp_str = f"{poble_sel} | {target_dt.strftime('%d/%m/%Y')} a les {target_dt.strftime('%H:%Mh')} ({TIMEZONE_EST_PENINSULA.zone}) / {cat_dt.strftime('%H:%Mh')} (CAT)"

        menu_options = ["Anàlisi Provincial", "Anàlisi Vertical", "Anàlisi de Mapes"]
        menu_icons = ["fullscreen", "graph-up-arrow", "map-fill"]
        
        active_tab = option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", 
                                orientation="horizontal", key="active_tab_est_peninsula_detail", default_index=0)
        
        # Lògica de càrrega i visualització per pestanya
        if active_tab == "Anàlisi Provincial":
            with st.spinner(f"Carregant anàlisi provincial per a les {target_dt.strftime('%H:%Mh')}..."):
                data_tuple, _, error_msg_sounding = carregar_dades_sondeig_est_peninsula(lat_sel, lon_sel, hourly_index_sel)
                map_data_conv, _ = carregar_dades_mapa_est_peninsula(nivell_sel, hourly_index_sel)
                alertes_zona = calcular_alertes_per_zona_peninsula(hourly_index_sel, nivell_sel)

            if data_tuple is None or error_msg_sounding:
                st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg_sounding)}")
            else:
                params_calc = data_tuple[1]
                if map_data_conv: params_calc[f'CONV_{nivell_sel}hPa'] = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
                provincia_actual = st.session_state.selected_area_peninsula
                valor_conv_provincial = alertes_zona.get(provincia_actual, 0)
                ui_pestanya_analisi_provincial(provincia_actual, valor_conv_provincial, poble_sel, timestamp_str, nivell_sel, map_data_conv, params_calc, target_dt.strftime('%H:%Mh'), data_tuple)

        elif active_tab == "Anàlisi Vertical":
            with st.spinner(f"Carregant dades del sondeig AROME per a {poble_sel}..."):
                data_tuple, final_index, error_msg = carregar_dades_sondeig_est_peninsula(lat_sel, lon_sel, hourly_index_sel)
            
            if data_tuple is None or error_msg:
                st.error(f"No s'ha pogut carregar el sondeig: {formatar_missatge_error_api(error_msg)}")
            else:
                params_calc = data_tuple[1]
                ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_sel, target_dt.strftime('%H:%Mh'), timestamp_str)

        elif active_tab == "Anàlisi de Mapes":
            ui_pestanya_mapes_est_peninsula(hourly_index_sel, timestamp_str, nivell_sel, poble_sel)

    else:
        # --- VISTA DE SELECCIÓ (MAPA INTERACTIU DE PROVÍNCIES) ---
        gdf_zones = carregar_dades_geografiques_peninsula()
        if gdf_zones is None: return

        st.session_state.setdefault('show_comarca_labels_peninsula', False)
        st.session_state.setdefault('alert_filter_level_peninsula', 'Tots')

        with st.container(border=True):
            st.markdown("##### Opcions de Visualització del Mapa")
            col_filter, col_labels = st.columns(2)
            with col_filter: st.selectbox("Filtrar avisos per nivell:", options=["Tots", "Moderat i superior", "Alt i superior", "Molt Alt i superior", "Només Extrems"], key="alert_filter_level_peninsula")
            with col_labels: st.toggle("Mostrar noms de les províncies amb avís", key="show_comarca_labels_peninsula")
        
        with st.spinner(f"Carregant mapa de situació per a les {target_dt.strftime('%H:%Mh')}..."):
            alertes_totals = calcular_alertes_per_zona_peninsula(hourly_index_sel, nivell_sel)
            alertes_filtrades = filtrar_alertes(alertes_totals, st.session_state.alert_filter_level_peninsula)
            map_output = ui_mapa_display_peninsula(alertes_filtrades, hourly_index_sel, show_labels=st.session_state.show_comarca_labels_peninsula)
        
        ui_llegenda_mapa_principal()

        if map_output and map_output.get("last_object_clicked_tooltip"):
            raw_tooltip = map_output["last_object_clicked_tooltip"]
            if "Provincia:" in raw_tooltip:
                clicked_area = raw_tooltip.split(':')[-1].strip()
                if clicked_area != st.session_state.get('selected_area_peninsula'):
                    st.session_state.selected_area_peninsula = clicked_area
                    st.rerun()

        selected_area = st.session_state.get('selected_area_peninsula')
        if selected_area and "---" not in selected_area:
            st.markdown(f"##### Selecciona una localitat a {selected_area}:")
            poblacions_a_mostrar = CIUTATS_PER_ZONA_PENINSULA.get(selected_area.strip(), {})
            
            if poblacions_a_mostrar:
                cols = st.columns(4)
                for i, nom_poble in enumerate(sorted(poblacions_a_mostrar.keys())):
                    with cols[i % 4]:
                        st.button(nom_poble, key=f"btn_pen_{nom_poble.replace(' ', '_')}", on_click=seleccionar_poble_peninsula, args=(nom_poble,), use_container_width=True)
            else:
                st.warning("Aquesta província no té localitats predefinides per a l'anàlisi.")
            
            if st.button("⬅️ Veure totes les províncies"):
                st.session_state.selected_area_peninsula = "--- Selecciona una província al mapa ---"
                st.rerun()
        else:
            st.info("Fes clic en una província del mapa per veure'n les localitats.", icon="👆")






def ui_pestanya_simulacio_nuvol(params_calculats, timestamp_str, poble_sel):
    """Mostra la pestanya de Simulació de Núvol amb les animacions."""
    st.markdown(f"#### Simulació del Cicle de Vida per a {poble_sel}")
    st.caption(timestamp_str)
    
    if 'regenerate_key' not in st.session_state: 
        st.session_state.regenerate_key = 0
    if st.button("🔄 Regenerar Totes les Animacions"): 
        forcar_regeneracio_animacio()
        
    with st.spinner("Generant simulacions visuals del cicle de vida..."):
        # Convertim el diccionari a un tuple per a que la funció cachejada funcioni
        params_tuple = tuple(sorted(params_calculats.items()))
        gifs = generar_animacions_professionals(params_tuple, timestamp_str, st.session_state.regenerate_key)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h5 style='text-align: center;'>1. Iniciació</h5>", unsafe_allow_html=True)
        if gifs['iniciacio']: st.image(gifs['iniciacio'])
        else: st.info("Condicions estables, no hi ha iniciació.")
    with col2:
        st.markdown("<h5 style='text-align: center;'>2. Maduresa</h5>", unsafe_allow_html=True)
        if gifs['maduresa']: st.image(gifs['maduresa'])
        else: st.info("La tempesta no arriba a la fase de maduresa.")
    with col3:
        st.markdown("<h5 style='text-align: center;'>3. Dissipació</h5>", unsafe_allow_html=True)
        if gifs['dissipacio']: st.image(gifs['dissipacio'])
        else: st.info("Sense fase de dissipació.")
        
    st.divider()
    nivell_conv_per_defecte = 925 # Usem un nivell estàndard per a la guia
    ui_guia_tall_vertical(params_calculats, nivell_conv_per_defecte)
    


def ui_pestanya_analisi_provincial(provincia, valor_conv, poble_sel, timestamp_str, nivell_sel, map_data, params_calc, hora_sel_str, data_tuple):
    """
    PESTANYA D'ANÀLISI PROVINCIAL. Utilitza el mapa de províncies de la península
    i inclou tota la lògica de visualització de convergència i direccionalitat.
    """
    st.markdown(f"#### Anàlisi de Convergència per a la Província: {provincia}")
    st.caption(timestamp_str.replace(poble_sel, provincia))

    col_mapa, col_diagnostic = st.columns([0.6, 0.4], gap="large")

    with col_mapa:
        st.markdown("##### Focus de Convergència a la Zona")
        
        with st.spinner("Generant mapa d'alta resolució de la província..."):
            gdf_provincies = carregar_dades_geografiques_peninsula()
            if gdf_provincies is None: 
                st.error("No s'ha pogut carregar el mapa de províncies.")
                return
            
            property_name = 'NAME_2'
            provincia_shape = gdf_provincies[gdf_provincies[property_name] == provincia]
            
            if provincia_shape.empty: 
                st.error(f"No s'ha trobat la geometria per a la província '{provincia}'. Revisa que el nom coincideixi amb el del fitxer GeoJSON.")
                return
            
            bounds = provincia_shape.total_bounds
            margin_lon = (bounds[2] - bounds[0]) * 0.3
            margin_lat = (bounds[3] - bounds[1]) * 0.3
            map_extent = [bounds[0] - margin_lon, bounds[2] + margin_lon, bounds[1] - margin_lat, bounds[3] + margin_lat]
            
            plt.style.use('default')
            fig, ax = crear_mapa_base(map_extent)
            ax.add_geometries(provincia_shape.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=2.5, linestyle='--', zorder=7)

            if map_data and valor_conv > 15:
                lons, lats = map_data['lons'], map_data['lats']
                grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 150), np.linspace(map_extent[2], map_extent[3], 150))
                grid_dewpoint = griddata((lons, lats), map_data['dewpoint_data'], (grid_lon, grid_lat), 'linear')
                u_comp, v_comp = mpcalc.wind_components(np.array(map_data['speed_data']) * units('km/h'), np.array(map_data['dir_data']) * units.degrees)
                grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
                grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
                
                with np.errstate(invalid='ignore'):
                    dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
                    convergence = (-(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s')).magnitude * 1e5
                    convergence[np.isnan(convergence)] = 0
                    DEWPOINT_THRESHOLD = 14 if nivell_sel >= 950 else 12
                    humid_mask = grid_dewpoint >= DEWPOINT_THRESHOLD
                    effective_convergence = np.where((convergence >= 15) & humid_mask, convergence, 0)
                
                smoothed_convergence = gaussian_filter(effective_convergence, sigma=3.5)
                smoothed_convergence[smoothed_convergence < 15] = 0

                if np.any(smoothed_convergence > 0):
                    fill_levels = [20, 30, 40, 60, 80, 100, 120]
                    cmap = plt.get_cmap('plasma')
                    norm = BoundaryNorm(fill_levels, ncolors=cmap.N, clip=True)
                    ax.contourf(grid_lon, grid_lat, smoothed_convergence, 
                                levels=fill_levels, cmap=cmap, norm=norm, 
                                alpha=0.75, zorder=3, transform=ccrs.PlateCarree(), extend='max')
                    line_levels = [30, 60, 100]
                    contours = ax.contour(grid_lon, grid_lat, smoothed_convergence, 
                                          levels=line_levels, colors='black', 
                                          linestyles='--', linewidths=0.8, alpha=0.7, 
                                          zorder=4, transform=ccrs.PlateCarree())
                    labels = ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
                    for label in labels:
                        label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

                points_df = pd.DataFrame({'lat': grid_lat.flatten(), 'lon': grid_lon.flatten(), 'conv': smoothed_convergence.flatten()})
                gdf_points = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df.lon, points_df.lat), crs="EPSG:4326")
                points_in_provincia = gpd.sjoin(gdf_points, provincia_shape.to_crs(gdf_points.crs), how="inner", predicate="within")
                
                if not points_in_provincia.empty:
                    max_conv_point = points_in_provincia.loc[points_in_provincia['conv'].idxmax()]
                    px, py = max_conv_point.geometry.x, max_conv_point.geometry.y
                    
                    if data_tuple and valor_conv >= 20:
                        if valor_conv >= 100: indicator_color = '#9370DB'
                        elif valor_conv >= 60: indicator_color = '#DC3545'
                        elif valor_conv >= 40: indicator_color = '#FD7E14'
                        else: indicator_color = '#28A745'
                        
                        path_effect = [path_effects.withStroke(linewidth=3.5, foreground='black')]
                        
                        circle = Circle((px, py), radius=0.05, facecolor='none', edgecolor=indicator_color, linewidth=2, transform=ccrs.PlateCarree(), zorder=12, path_effects=path_effect)
                        ax.add_patch(circle)
                        ax.plot(px, py, 'x', color=indicator_color, markersize=8, markeredgewidth=2, zorder=13, transform=ccrs.PlateCarree(), path_effects=path_effect)

                        try:
                            sounding_data, _ = data_tuple
                            p, u, v = sounding_data[0], sounding_data[3], sounding_data[4]
                            if p.m.min() < 500 and p.m.max() > 700:
                                u_700, v_700 = np.interp(700, p.m[::-1], u.m[::-1]), np.interp(700, p.m[::-1], v.m[::-1])
                                u_500, v_500 = np.interp(500, p.m[::-1], u.m[::-1]), np.interp(500, p.m[::-1], v.m[::-1])
                                mean_u, mean_v = (u_700 + u_500) / 2.0 * units('m/s'), (v_700 + v_500) / 2.0 * units('m/s')
                                storm_dir_to = (mpcalc.wind_direction(mean_u, mean_v).m + 180) % 360
                                
                                dir_rad = np.deg2rad(90 - storm_dir_to)
                                length = 0.25
                                end_x, end_y = px + length * np.cos(dir_rad), py + length * np.sin(dir_rad)
                                ax.plot([px, end_x], [py, end_y], color=indicator_color, linewidth=2, transform=ccrs.PlateCarree(), zorder=12, path_effects=path_effect)
                                
                                num_barbs = 3; barb_length = 0.04
                                barb_angle_rad = dir_rad + np.pi / 2
                                for i in range(1, num_barbs + 1):
                                    pos = 0.4 + (i * 0.2)
                                    barb_cx, barb_cy = px + length * pos * np.cos(dir_rad), py + length * pos * np.sin(dir_rad)
                                    barb_sx, barb_sy = barb_cx - barb_length / 2 * np.cos(barb_angle_rad), barb_cy - barb_length / 2 * np.sin(barb_angle_rad)
                                    barb_ex, barb_ey = barb_cx + barb_length / 2 * np.cos(barb_angle_rad), barb_cy + barb_length / 2 * np.sin(barb_angle_rad)
                                    ax.plot([barb_sx, barb_ex], [barb_sy, barb_ey], color=indicator_color, linewidth=2, transform=ccrs.PlateCarree(), zorder=12, path_effects=path_effect)
                        except Exception: pass
            
            poble_coords = CIUTATS_EST_PENINSULA.get(poble_sel)
            if poble_coords:
                lon_poble, lat_poble = poble_coords['lon'], poble_coords['lat']
                ax.text(lon_poble, lat_poble, '( Tú )\n▼', transform=ccrs.PlateCarree(),
                        fontsize=10, fontweight='bold', color='black',
                        ha='center', va='bottom', zorder=14,
                        path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])

            ax.set_title(f"Focus de Convergència a {provincia}", weight='bold', fontsize=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_diagnostic:
        st.markdown("##### Diagnòstic de la Zona")
        if valor_conv >= 100:
            nivell_alerta, color_alerta, emoji, descripcio = "Extrem", "#9370DB", "🔥", f"S'ha detectat un focus de convergència excepcionalment fort a la província, amb un valor màxim de {valor_conv:.0f}. Aquesta és una senyal inequívoca per a la formació de temps sever organitzat i potencialment perillós."
        elif valor_conv >= 60:
            nivell_alerta, color_alerta, emoji, descripcio = "Molt Alt", "#DC3545", "🔴", f"S'ha detectat un focus de convergència extremadament fort a la província, amb un valor màxim de {valor_conv:.0f}. Aquesta és una senyal molt clara per a la formació imminent de tempestes, possiblement severes i organitzades."
        elif valor_conv >= 40:
            nivell_alerta, color_alerta, emoji, descripcio = "Alt", "#FD7E14", "🟠", f"Hi ha un focus de convergència forta a la província, amb un valor màxim de {valor_conv:.0f}. Aquest és un disparador molt eficient i és molt probable que es desenvolupin tempestes a la zona."
        elif valor_conv >= 20:
            nivell_alerta, color_alerta, emoji, descripcio = "Moderat", "#28A745", "🟢", f"S'observa una zona de convergència moderada a la província, amb un valor màxim de {valor_conv:.0f}. Aquesta condició pot ser suficient per iniciar tempestes si l'atmosfera és inestable."
        else:
            nivell_alerta, color_alerta, emoji, descripcio = "Baix", "#6c757d", "⚪", f"No es detecten focus de convergència significatius (Valor: {valor_conv:.0f}). El forçament dinàmic per iniciar tempestes és feble o inexistent."

        st.markdown(f"""
        <div style="text-align: center; padding: 12px; background-color: #2a2c34; border-radius: 10px; border: 1px solid #444;">
             <span style="font-size: 1.2em; color: #FAFAFA;">{emoji} Potencial de Dispar: <strong style="color:{color_alerta}">{nivell_alerta}</strong></span>
             <p style="font-size:0.95em; color:#a0a0b0; margin-top:10px; text-align: left;">{descripcio}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Validació Atmosfèrica")
        if not params_calc:
            st.warning("No hi ha dades de sondeig disponibles per a la validació.")
        else:
            mucin = params_calc.get('MUCIN', 0) or 0
            mucape = params_calc.get('MUCAPE', 0) or 0
            
            vered_titol, vered_color, vered_emoji, vered_desc = "", "", "", ""
            if mucin < -75:
                vered_titol, vered_color, vered_emoji = "Inhibida", "#DC3545", "👎"
                vered_desc = f"Tot i la convergència, hi ha una inhibició (CIN) molt forta de **{mucin:.0f} J/kg** que actua com una 'tapa', dificultant o impedint el desenvolupament de tempestes."
            elif mucape < 250:
                vered_titol, vered_color, vered_emoji = "Sense Energia", "#FD7E14", "🤔"
                vered_desc = f"El disparador existeix, però l'atmosfera té molt poc 'combustible' (CAPE), amb només **{mucape:.0f} J/kg**. Les tempestes, si es formen, seran febles."
            else:
                vered_titol, vered_color, vered_emoji = "Efectiva", "#28A745", "👍"
                vered_desc = f"Les condicions són favorables! La convergència troba una atmosfera amb prou energia (**{mucape:.0f} J/kg**) i una inhibició baixa (**{mucin:.0f} J/kg**) per a desenvolupar tempestes."

            st.markdown(f"""
            <div style="text-align: center; padding: 12px; background-color: #2a2c34; border-radius: 10px; border: 1px solid #444;">
                 <span style="font-size: 1.1em; color: #FAFAFA;">{vered_emoji} Veredicte: Convergència <strong style="color:{vered_color}">{vered_titol}</strong></span>
                 <p style="font-size:0.9em; color:#a0a0b0; margin-top:10px; text-align: left;">{vered_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Aquesta validació es basa en el sondeig vertical de {poble_sel}.")
        
        crear_llegenda_direccionalitat()



def on_manual_poble_select():
    """
    Callback que s'activa en seleccionar una localitat del menú de cerca directa.
    Actualitza la clau de session_state correcta per a la zona activa.
    """
    zona_activa = st.session_state.get('zone_selected')
    poble_seleccionat = st.session_state.get("manual_selector_widget")

    if not zona_activa or not poble_seleccionat or "---" in poble_seleccionat:
        return

    # Diccionari que mapeja la zona activa amb la seva clau de session_state
    ZONE_SESSION_KEYS = {
        'catalunya': 'poble_sel',
        'est_peninsula': 'poble_selector_est_peninsula',
        'valley_halley': 'poble_selector_usa',
        'alemanya': 'poble_selector_alemanya',
        'italia': 'poble_selector_italia',
        'holanda': 'poble_selector_holanda',
        'japo': 'poble_selector_japo',
        'uk': 'poble_selector_uk',
        'canada': 'poble_selector_canada',
        'noruega': 'poble_selector_noruega'
    }
    
    session_key = ZONE_SESSION_KEYS.get(zona_activa)
    if session_key:
        st.session_state[session_key] = poble_seleccionat


def ui_capcalera_selectors(ciutats_a_mostrar, info_msg=None, zona_activa="catalunya", convergencies=None):
    """
    Mostra la capçalera de l'aplicació amb els selectors de navegació.
    Versió 2.0:
    - Afegeix un selector de "Cerca Directa" per a navegar a qualsevol localitat
      de la zona activa de manera instantània.
    - Reorganitza les columnes per a una millor distribució.
    """
    st.markdown(f'<h1 style="text-align: center; color: #FF4B4B;">Terminal de Temps Sever | {zona_activa.replace("_", " ").title()}</h1>', unsafe_allow_html=True)
    is_guest = st.session_state.get('guest_mode', False)
    
    altres_zones = {
        'catalunya': 'Catalunya', 'valley_halley': 'Tornado Alley', 'alemanya': 'Alemanya', 
        'italia': 'Itàlia', 'holanda': 'Holanda', 'japo': 'Japó', 'uk': 'Regne Unit', 
        'canada': 'Canadà', 'noruega': 'Noruega', 'est_peninsula': 'Est Península'
    }
    if zona_activa in altres_zones:
        del altres_zones[zona_activa]
    
    # --- NOU DISSENY DE COLUMNES ---
    col_text, col_manual_select, col_zone_change, col_actions = st.columns([0.35, 0.3, 0.2, 0.15])
    
    with col_text:
        if not is_guest: st.markdown(f"Benvingut/da, **{st.session_state.get('username', 'Usuari')}**!")

    # --- NOU SELECTOR DE CERCA DIRECTA ---
    with col_manual_select:
        ZONE_CITIES_MAP = {
            'catalunya': CIUTATS_CATALUNYA, 'est_peninsula': CIUTATS_EST_PENINSULA,
            'valley_halley': USA_CITIES, 'alemanya': CIUTATS_ALEMANYA, 'italia': CIUTATS_ITALIA,
            'holanda': CIUTATS_HOLANDA, 'japo': CIUTATS_JAPO, 'uk': CIUTATS_UK,
            'canada': CIUTATS_CANADA, 'noruega': CIUTATS_NORUEGA
        }
        
        ciutats_zona_actual = ZONE_CITIES_MAP.get(zona_activa, {})
        if ciutats_zona_actual:
            opcions = ["--- Anar a... ---"] + sorted(list(ciutats_zona_actual.keys()))
            st.selectbox(
                "Cerca Directa:", 
                options=opcions, 
                key="manual_selector_widget", 
                on_change=on_manual_poble_select,
                help="Selecciona qualsevol localitat per anar directament a la seva anàlisi."
            )

    with col_zone_change:
        nova_zona_key = st.selectbox("Canviar a:", options=list(altres_zones.keys()), format_func=lambda k: altres_zones[k], index=None, placeholder="Altres zones...")
        if nova_zona_key:
            st.session_state.zone_selected = nova_zona_key
            st.rerun()
            
    with col_actions:
        # Usem subcolumnes per als botons
        col_back, col_logout = st.columns(2)
        with col_back:
            if st.button("⬅️", use_container_width=True, help="Tornar a la selecció de zona"):
                keys_to_clear = [k for k in st.session_state if k not in ['logged_in', 'username', 'guest_mode', 'developer_mode']]
                for key in keys_to_clear: del st.session_state[key]
                st.rerun()
        with col_logout:
            if st.button("🚪", use_container_width=True, help="Sortir / Tancar Sessió"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()

    st.divider()

    # --- BLOC DE SELECTORS RESTAURAT ---
    # Aquesta part torna a crear els selectors de ciutats per a les zones que no tenen mapa interactiu.
    if zona_activa not in ['catalunya', 'est_peninsula']:
        with st.container(border=True):
            if zona_activa == 'valley_halley':
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


@st.cache_data(ttl=600, show_spinner="Preparant dades del mapa de situació...")
def preparar_dades_mapa_cachejat(alertes_tuple, selected_area_str, show_labels):
    """
    Funció CACHEADA que prepara les dades per al mapa de Folium, amb etiquetes
    simplificades i millor posicionament.
    """
    alertes_per_zona = dict(alertes_tuple)
    gdf = carregar_dades_geografiques()
    if gdf is None: return None
    property_name = next((prop for prop in ['nom_zona', 'nom_comar', 'nomcomar'] if prop in gdf.columns), 'nom_comar')

    def get_color_from_cape(cape_value):
        if not isinstance(cape_value, (int, float, np.number)) or pd.isna(cape_value) or cape_value < 250:
            return '#6c757d', '#FFFFFF'
        if cape_value < 750: return '#28A745', '#FFFFFF'
        if cape_value < 1500: return '#FFFF00', '#000000'
        if cape_value < 2500: return '#FD7E14', '#FFFFFF'
        if cape_value < 3500: return '#DC3545', '#FFFFFF'
        return '#9370DB', '#FFFFFF'

    styles_dict = {}
    for feature in gdf.iterfeatures():
        nom_feature_raw = feature.get('properties', {}).get(property_name)
        if nom_feature_raw and isinstance(nom_feature_raw, str):
            nom_feature = nom_feature_raw.strip().replace('.', '')
            alert_data = alertes_per_zona.get(nom_feature)
            cape_val = alert_data['cape'] if alert_data else 0
            alert_color, _ = get_color_from_cape(cape_val)
            styles_dict[nom_feature] = {
                'fillColor': alert_color, 'color': alert_color,
                'fillOpacity': 0.60 if alert_data else 0.25,
                'weight': 2.5 if alert_data else 1
            }

    markers_data = []
    if show_labels:
        for zona, data in alertes_per_zona.items():
            capital_info = CAPITALS_COMARCA.get(zona)
            if capital_info:
                cape_val = data['cape']; conv_val = data['conv']
                bg_color, text_color = get_color_from_cape(cape_val)
                # <<<--- ETIQUETA MODIFICADA: Sense nom de comarca i més ample ---
                icon_html = f"""<div style="background-color: {bg_color}; color: {text_color}; padding: 5px 10px; border-radius: 8px; border: 2px solid {text_color}; font-family: sans-serif; font-size: 11px; font-weight: bold; text-align: center; box-shadow: 3px 3px 5px rgba(0,0,0,0.5); min-width: 120px;">⚡ {cape_val:.0f} J/kg | 🌀 {conv_val:.0f}</div>"""
                markers_data.append({
                    'location': [capital_info['lat'], capital_info['lon']],
                    'icon_html': icon_html, 'tooltip': f"Comarca: {zona}"
                })

    return {"gdf": gdf.to_json(), "property_name": property_name, "styles": styles_dict, "markers": markers_data}



@st.cache_resource(ttl=1800, show_spinner=False)
def generar_mapa_cachejat_cat(hourly_index, nivell, timestamp_str, map_extent_tuple, cape_min_filter, cape_max_filter, convergence_min_filter):
    """
    Funció generadora que crea i desa a la memòria cau el mapa de convergència.
    Ara accepta els paràmetres de filtre.
    """
    map_data, error = carregar_dades_mapa_cat(nivell, hourly_index)
    if error or not map_data:
        return None
    
    map_extent_list = list(map_extent_tuple)
    
    fig = crear_mapa_forecast_combinat_cat(
        map_data['lons'], map_data['lats'], 
        map_data['speed_data'], map_data['dir_data'], 
        map_data['dewpoint_data'],
        map_data['cape_data'],
        nivell, 
        timestamp_str, 
        map_extent_list,
        cape_min_filter,
        cape_max_filter,
        convergence_min_filter
    )
    return fig


def crear_mapa_vents_cat(lons, lats, speed_data, dir_data, nivell, timestamp_str, map_extent):
    """
    Crea un mapa que mostra la velocitat del vent (color de fons) i la direcció (línies).
    """
    fig, ax = crear_mapa_base(map_extent)
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 200), np.linspace(map_extent[2], map_extent[3], 200))
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    
    # Interpolació ràpida
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_speed = griddata((lons, lats), speed_data, (grid_lon, grid_lat), 'linear')
    
    colors_wind = ['#d1d1f0', '#6495ed', '#add8e6', '#90ee90', '#32cd32', '#adff2f', '#f0e68c', '#d2b48c', '#bc8f8f', '#cd5c5c', '#c71585', '#9370db']
    speed_levels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    custom_cmap = ListedColormap(colors_wind)
    norm_speed = BoundaryNorm(speed_levels, ncolors=custom_cmap.N, clip=True)
    
    ax.pcolormesh(grid_lon, grid_lat, grid_speed, cmap=custom_cmap, norm=norm_speed, zorder=2, transform=ccrs.PlateCarree())
    ax.streamplot(grid_lon, grid_lat, grid_u, grid_v, color='black', linewidth=0.7, density=2.5, zorder=3, transform=ccrs.PlateCarree())
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm_speed, cmap=custom_cmap), ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label("Velocitat del Vent (km/h)")
    ax.set_title(f"Vent a {nivell} hPa\n{timestamp_str}", weight='bold', fontsize=16)
    
    afegir_etiquetes_ciutats(ax, map_extent)
    
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




@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dades_mapa_adveccio_cat(nivell, hourly_index):
    """
    Funció dedicada a carregar les dades necessàries per al mapa d'advecció
    al nivell de pressió especificat (p. ex., 850, 700, 500 hPa).
    """
    try:
        # El nivell ja no està fixat, sinó que ve del paràmetre de la funció
        variables = [f"temperature_{nivell}hPa", f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"]
        map_data_raw, error = carregar_dades_mapa_base_cat(variables, hourly_index)

        if error:
            return None, error

        # Reanomenem les claus de forma dinàmica segons el nivell
        map_data_raw['temp_data'] = map_data_raw.pop(f'temperature_{nivell}hPa')
        map_data_raw['speed_data'] = map_data_raw.pop(f'wind_speed_{nivell}hPa')
        map_data_raw['dir_data'] = map_data_raw.pop(f'wind_direction_{nivell}hPa')
        return map_data_raw, None

    except Exception as e:
        return None, f"Error en processar dades del mapa d'advecció: {e}"




    
def ui_explicacio_adveccio():
    """
    Crea una secció explicativa sobre com interpretar el mapa d'advecció.
    """
    st.markdown("---")
    st.markdown("##### Com Interpretar el Mapa d'Advecció")
    st.markdown("""
    <style>
    .explanation-card { background-color: #f0f2f6; border: 1px solid #d1d1d1; border-radius: 10px; padding: 20px; height: 100%; display: flex; flex-direction: column; }
    .explanation-title { font-size: 1.3em; font-weight: bold; color: #1a1a2e; margin-bottom: 10px; display: flex; align-items: center; }
    .explanation-icon { font-size: 1.5em; margin-right: 12px; }
    .explanation-text { font-size: 1em; color: #333; line-height: 1.6; }
    .explanation-text strong { color: #0056b3; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="explanation-card">
            <div class="explanation-title"><span class="explanation-icon" style="color:red;">🌡️⬆️</span>Advecció Càlida (Zones Vermelles)</div>
            <div class="explanation-text">
                Indica que el vent està transportant <strong>aire més càlid</strong> cap a la zona. Aquest procés força l'aire a ascendir lentament sobre l'aire més fred que hi ha a sota.
                <br><br>
                <strong>Efectes típics:</strong> Formació de núvols estratiformes (capes de núvols com Nimbostratus o Altostratus) i potencial per a <strong>pluges febles però contínues i generalitzades</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="explanation-card">
            <div class="explanation-title"><span class="explanation-icon" style="color:blue;">❄️⬇️</span>Advecció Freda (Zones Blaves)</div>
            <div class="explanation-text">
                Indica que el vent està transportant <strong>aire més fred</strong>. L'aire fred, en ser més dens, tendeix a ficar-se per sota de l'aire més càlid, desestabilitzant l'atmosfera.
                <br><br>
                <strong>Efectes típics:</strong> Assecament de l'atmosfera a nivells mitjans, però pot actuar com un <strong>mecanisme de dispar</strong> per a la convecció si hi ha humitat a nivells baixos, generant ruixats o tempestes.
            </div>
        </div>
        """, unsafe_allow_html=True)
        



def get_peak_names(points_to_check: List[Dict[str, float]]) -> Dict[str, any]:
    """
    Consulta l'API de GeoNames per a obtenir el nom dels cims més propers.
    Versió 3.0 (Doble Búsqueda):
    - ESTRATÈGIA DOBLE: Primer intenta buscar a Wikipedia. Si falla, fa una
      segona cerca a OpenStreetMap (OSM) per a màxima probabilitat d'èxit.
    - GESTIÓ D'ERRORS MILLORADA: Detecta si s'ha excedit el límit de peticions de l'API.
    """
    try:
        username = st.secrets["GEONAMES_USERNAME"]
    except KeyError:
        return {"error": "La clau 'GEONAMES_USERNAME' no està configurada als secrets. No es poden identificar els cims."}

    peak_names = []
    session = requests.Session() # Usem una sessió per a més eficiència

    for point in points_to_check:
        lat, lon = point['lat'], point['lon']
        peak_found = False
        
        # --- Primer Intent: Wikipedia (noms més nets) ---
        try:
            url_wiki = f"http://api.geonames.org/findNearbyWikipediaJSON?lat={lat}&lng={lon}&radius=10&maxRows=1&featureClass=T&lang=ca&username={username}"
            response = session.get(url_wiki, timeout=5)
            response.raise_for_status()
            data = response.json()
            if 'geonames' in data and data['geonames']:
                peak = data['geonames'][0]
                is_duplicate = any(p['name'] == peak.get('title') for p in peak_names)
                if not is_duplicate:
                    peak_names.append({"name": peak.get('title'), "lat": float(peak.get('lat')), "lon": float(peak.get('lon')), "elevation": float(peak.get('elevation', 0))})
                    peak_found = True
            elif "status" in data: # Comprovem si hi ha un missatge d'error de l'API
                if "limit" in data["status"]["message"]:
                    return {"error": "S'ha excedit el límit de peticions a l'API de GeoNames. Prova-ho més tard."}
        except Exception:
            pass # Si falla, simplement ho intentem amb OSM

        # --- Segon Intent: OpenStreetMap (més dades, fallback) ---
        if not peak_found:
            try:
                # 'T.PK' és el codi per a 'peak' a OSM
                url_osm = f"http://api.geonames.org/findNearbyPOIsOSMJSON?lat={lat}&lng={lon}&radius=10&maxRows=1&featureCode=T.PK&username={username}"
                response = session.get(url_osm, timeout=5)
                response.raise_for_status()
                data = response.json()
                if 'poi' in data and data['poi']:
                    peak = data['poi'][0]
                    is_duplicate = any(p['name'] == peak.get('name') for p in peak_names)
                    if not is_duplicate:
                        # L'elevació no ve a l'API d'OSM, així que la demanem a part
                        elev_resp = session.get(f"http://api.geonames.org/gtopo30JSON?lat={peak.get('lat')}&lng={peak.get('lon')}&username={username}").json()
                        elevation = elev_resp.get('gtopo30', 0)
                        peak_names.append({"name": peak.get('name'), "lat": float(peak.get('lat')), "lon": float(peak.get('lon')), "elevation": float(elevation)})
            except Exception:
                continue # Si aquest també falla, passem al següent pic
            
    return {"peaks": peak_names}




def dibuixar_fronts_aproximats(ax, grid_lon, grid_lat, grid_u, grid_v, advection_data):
    """
    Versió 7.0 (ALTA FIDELITAT): Dibuixa els fronts amb línies fines i símbols
    petits i detallats per a un acabat visual extremadament professional i subtil.
    """
    try:
        # --- PARÀMETRES DE CONFIGURACIÓ VISUAL ---
        LLINDAR_FRONT_FRED = -1.5
        LLINDAR_FRONT_CALID = 1.5
        MIN_FRONT_LENGTH = 20
        MAX_FRONTS_TO_DRAW = 2
        OFFSET_FACTOR = 0.08 # Un desplaçament més petit per a més precisió
        
        # Efecte visual per a línies extremadament netes
        path_effect_front = [path_effects.withStroke(linewidth=3.0, foreground='black')]

        # --- ANÀLISI DEL FRONT FRED ---
        cs_fred = ax.contour(grid_lon, grid_lat, advection_data, levels=[LLINDAR_FRONT_FRED],
                             colors='none', transform=ccrs.PlateCarree())
        
        segments_freds = [seg for seg in cs_fred.allsegs[0] if len(seg) > MIN_FRONT_LENGTH]
        segments_freds.sort(key=len, reverse=True)

        for seg in segments_freds[:MAX_FRONTS_TO_DRAW]:
            center_lon, center_lat = np.mean(seg[:, 0]), np.mean(seg[:, 1])
            wind_u = griddata((grid_lon.flatten(), grid_lat.flatten()), grid_u.flatten(), (center_lon, center_lat), method='nearest')
            wind_v = griddata((grid_lon.flatten(), grid_lat.flatten()), grid_v.flatten(), (center_lon, center_lat), method='nearest')

            wind_magnitude = np.sqrt(wind_u**2 + wind_v**2)
            if wind_magnitude > 0:
                offset_lon = (wind_u / wind_magnitude) * OFFSET_FACTOR
                offset_lat = (wind_v / wind_magnitude) * OFFSET_FACTOR
            else:
                offset_lon, offset_lat = 0, 0
            
            path_desplacada = seg + np.array([offset_lon, offset_lat])
            
            # Línia base molt més fina
            ax.plot(path_desplacada[:, 0], path_desplacada[:, 1], color='#0077BE', linewidth=1.8,
                    transform=ccrs.PlateCarree(), zorder=5, path_effects=path_effect_front)

            # Símbols (triangles) molt més petits i delicats
            for i in range(5, len(path_desplacada) - 5, 15): # Més espaiat per a un look més net
                p1, p2 = path_desplacada[i], path_desplacada[i+1]
                mid_point = (p1 + p2) / 2
                angle_linea_rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                angle_vent_rad = np.arctan2(wind_v, wind_u)
                
                triangle_size = 0.040 # <<<--- Mida dràsticament reduïda
                
                p_tip = mid_point + triangle_size * np.array([np.cos(angle_vent_rad), np.sin(angle_vent_rad)])
                base_half_width = triangle_size * 0.8
                p_base1 = mid_point - base_half_width * np.array([np.cos(angle_linea_rad), np.sin(angle_linea_rad)])
                p_base2 = mid_point + base_half_width * np.array([np.cos(angle_linea_rad), np.sin(angle_linea_rad)])

                triangle = Polygon([p_base1, p_tip, p_base2], facecolor='#0077BE', edgecolor='black', linewidth=0.5,
                                   transform=ccrs.PlateCarree(), zorder=6)
                ax.add_patch(triangle)

        # --- ANÀLISI DEL FRONT CÀLID (amb el mateix refinament) ---
        cs_calid = ax.contour(grid_lon, grid_lat, advection_data, levels=[LLINDAR_FRONT_CALID],
                              colors='none', transform=ccrs.PlateCarree())
        
        segments_calids = [seg for seg in cs_calid.allsegs[0] if len(seg) > MIN_FRONT_LENGTH]
        segments_calids.sort(key=len, reverse=True)

        for seg in segments_calids[:MAX_FRONTS_TO_DRAW]:
            center_lon, center_lat = np.mean(seg[:, 0]), np.mean(seg[:, 1])
            wind_u = griddata((grid_lon.flatten(), grid_lat.flatten()), grid_u.flatten(), (center_lon, center_lat), method='nearest')
            wind_v = griddata((grid_lon.flatten(), grid_lat.flatten()), grid_v.flatten(), (center_lon, center_lat), method='nearest')

            wind_magnitude = np.sqrt(wind_u**2 + wind_v**2)
            if wind_magnitude > 0:
                offset_lon = (wind_u / wind_magnitude) * OFFSET_FACTOR
                offset_lat = (wind_v / wind_magnitude) * OFFSET_FACTOR
            else:
                offset_lon, offset_lat = 0, 0

            path_desplacada = seg + np.array([offset_lon, offset_lat])
            
            ax.plot(path_desplacada[:, 0], path_desplacada[:, 1], color='#D81E05', linewidth=1.8,
                    transform=ccrs.PlateCarree(), zorder=5, path_effects=path_effect_front)

            for i in range(5, len(path_desplacada) - 5, 18): # Encara més espaiat
                p1, p2 = path_desplacada[i], path_desplacada[i+1]
                mid_point = (p1 + p2) / 2
                angle_vent_deg = np.rad2deg(np.arctan2(wind_v, wind_u))
                
                semicircle = Wedge(center=mid_point, r=0.035, # <<<--- Mida dràsticament reduïda
                                   theta1=angle_vent_deg - 90, 
                                   theta2=angle_vent_deg + 90,
                                   facecolor='#D81E05', edgecolor='black', linewidth=0.5,
                                   transform=ccrs.PlateCarree(), zorder=6)
                ax.add_patch(semicircle)

    except Exception as e:
        print(f"No s'han pogut dibuixar els fronts (versió d'alta fidelitat): {e}")
        

def crear_mapa_adveccio_cat(lons, lats, temp_data, speed_data, dir_data, nivell, timestamp_str, map_extent):
    """
    Crea un mapa d'advecció tèrmica amb renderitzat d'alta qualitat i
    ARA AFEGEIX UNA REPRESENTACIÓ VISUAL DELS FRONTS.
    """
    plt.style.use('default')
    fig, ax = crear_mapa_base(map_extent)

    # ... (tot el codi d'interpolació i càlcul de l'advecció es manté exactament igual) ...
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 300), np.linspace(map_extent[2], map_extent[3], 300))
    grid_temp = griddata((lons, lats), temp_data, (grid_lon, grid_lat), 'linear')
    u_comp, v_comp = mpcalc.wind_components(np.array(speed_data) * units('km/h'), np.array(dir_data) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')

    with np.errstate(invalid='ignore'):
        num_lats, num_lons = grid_lat.shape
        lon1_dx, lon2_dx = grid_lon[num_lats // 2, 0], grid_lon[num_lats // 2, -1]
        lat_mid_dx = grid_lat[num_lats // 2, 0]
        dist_x_km = haversine_distance(lat_mid_dx, lon1_dx, lat_mid_dx, lon2_dx)
        dx = (dist_x_km * 1000) / (num_lons - 1)
        lat1_dy, lat2_dy = grid_lat[0, num_lons // 2], grid_lat[-1, num_lons // 2]
        lon_mid_dy = grid_lon[0, num_lons // 2]
        dist_y_km = haversine_distance(lat1_dy, lon_mid_dy, lat2_dy, lon_mid_dy)
        dy = (dist_y_km * 1000) / (num_lats - 1)
        grad_temp_y, grad_temp_x = np.gradient(grid_temp, dy, dx)
        advection_calc = - ((grid_u * grad_temp_x) + (grid_v * grad_temp_y))
        advection_c_per_hour = advection_calc * 3600
        advection_c_per_hour[np.isnan(advection_c_per_hour)] = 0
        
    smoothed_advection = gaussian_filter(advection_c_per_hour, sigma=2.5)

    # ... (el codi per dibuixar el fons de color de l'advecció i les isotermes es manté igual) ...
    fill_levels_adv = np.arange(-3.0, 3.1, 0.25)
    cmap_adv = plt.get_cmap('bwr')
    norm_adv = BoundaryNorm(fill_levels_adv, ncolors=cmap_adv.N, clip=True)
    im = ax.contourf(grid_lon, grid_lat, smoothed_advection, 
                     levels=fill_levels_adv, cmap=cmap_adv, norm=norm_adv,
                     alpha=0.7, zorder=2, transform=ccrs.PlateCarree(), extend='both')
    
    iso_levels = np.arange(int(np.nanmin(grid_temp)) - 2, int(np.nanmax(grid_temp)) + 2, 2)
    contours_temp = ax.contour(grid_lon, grid_lat, grid_temp, levels=iso_levels, colors='black',
                               linestyles='--', linewidths=0.8, zorder=4, transform=ccrs.PlateCarree())
    ax.clabel(contours_temp, inline=True, fontsize=8, fmt='%1.0f°')

    # <<<--- LÍNIA AFEGIDA AQUÍ ---
    # Després de dibuixar l'advecció, cridem la nova funció per superposar els fronts
    dibuixar_fronts_aproximats(ax, grid_lon, grid_lat, grid_u, grid_v, smoothed_advection)
    # <<<--------------------------

    # ... (la resta de la funció per a la barra de color, títols, etc., es manté igual) ...
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label(f"Advecció Tèrmica a {nivell}hPa (°C / hora)")
    ax.set_title(f"Advecció Tèrmica i Fronts a {nivell}hPa\n{timestamp_str}", weight='bold', fontsize=16)
    afegir_etiquetes_ciutats(ax, map_extent)

    return fig

def ui_explicacio_adveccio():
    """
    Crea una secció explicativa sobre com interpretar el mapa d'advecció.
    """
    st.markdown("---")
    st.markdown("##### Com Interpretar el Mapa d'Advecció")
    st.markdown("""
    <style>
    .explanation-card { background-color: #f0f2f6; border: 1px solid #d1d1d1; border-radius: 10px; padding: 20px; height: 100%; display: flex; flex-direction: column; }
    .explanation-title { font-size: 1.3em; font-weight: bold; color: #1a1a2e; margin-bottom: 10px; display: flex; align-items: center; }
    .explanation-icon { font-size: 1.5em; margin-right: 12px; }
    .explanation-text { font-size: 1em; color: #333; line-height: 1.6; }
    .explanation-text strong { color: #0056b3; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="explanation-card">
            <div class="explanation-title"><span class="explanation-icon" style="color:red;">🌡️⬆️</span>Advecció Càlida (Zones Vermelles)</div>
            <div class="explanation-text">
                Indica que el vent està transportant <strong>aire més càlid</strong> cap a la zona. Aquest procés força l'aire a ascendir lentament sobre l'aire més fred que hi ha a sota.
                <br><br>
                <strong>Efectes típics:</strong> Formació de núvols estratiformes (capes de núvols com Nimbostratus o Altostratus) i potencial per a <strong>pluges febles però contínues i generalitzades</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="explanation-card">
            <div class="explanation-title"><span class="explanation-icon" style="color:blue;">❄️⬇️</span>Advecció Freda (Zones Blaves)</div>
            <div class="explanation-text">
                Indica que el vent està transportant <strong>aire més fred</strong>. L'aire fred, en ser més dens, tendeix a ficar-se per sota de l'aire més càlid, desestabilitzant l'atmosfera.
                <br><br>
                <strong>Efectes típics:</strong> Assecament de l'atmosfera a nivells mitjans, però pot actuar com un <strong>mecanisme de dispar</strong> per a la convecció si hi ha humitat a nivells baixos, generant ruixats o tempestes.
            </div>
        </div>
        """, unsafe_allow_html=True)




def seleccionar_poble(nom_poble):
    """Callback segur per a seleccionar un poble i canviar a la vista detallada."""
    st.session_state.poble_sel = nom_poble

def tornar_a_seleccio_comarca():
    """Callback segur per a tornar a la llista de municipis de la comarca actual."""
    st.session_state.poble_sel = "--- Selecciona una localitat ---"

def tornar_al_mapa_general():
    """Callback segur per a tornar a la vista principal del mapa de Catalunya."""
    st.session_state.poble_sel = "--- Selecciona una localitat ---"
    st.session_state.selected_area = "--- Selecciona una zona al mapa ---"



def generar_bulleti_automatic_catalunya(alertes_totals, hora_str):
    """
    Analitza totes les alertes comarcals i genera un butlletí de risc estructurat,
    seguint els llindars de CAPE especificats per l'expert. VERSIÓ SENSE XIFRES DE CAPE.
    """
    if not alertes_totals:
        max_cape = 0
    else:
        max_cape = max(data.get('cape', 0) for data in alertes_totals.values() if data.get('cape') is not None)

    # --- 1. Classificació de les comarques ---
    zones_per_risc = {
        "RISC VERMELL (CAPE > 3000)": [], "RISC TARONJA (CAPE > 2500)": [],
        "RISC GROC FORT (CAPE > 2000)": [], "RISC GROC (CAPE > 1500)": [],
        "RISC MODERAT (CAPE > 1000)": [], "SEGUIMENT (CAPE > 500)": []
    }
    comarca_max_cape = ""
    cape_valor_maxim = 0
    if alertes_totals:
        for comarca, data in alertes_totals.items():
            cape = data.get('cape', 0)
            if cape > cape_valor_maxim:
                cape_valor_maxim = cape
                comarca_max_cape = comarca
            
            if cape >= 3000: zones_per_risc["RISC VERMELL (CAPE > 3000)"].append(comarca)
            elif cape >= 2500: zones_per_risc["RISC TARONJA (CAPE > 2500)"].append(comarca)
            elif cape >= 2000: zones_per_risc["RISC GROC FORT (CAPE > 2000)"].append(comarca)
            elif cape >= 1500: zones_per_risc["RISC GROC (CAPE > 1500)"].append(comarca)
            elif cape >= 1000: zones_per_risc["RISC MODERAT (CAPE > 1000)"].append(comarca)
            elif cape >= 500: zones_per_risc["SEGUIMENT (CAPE > 500)"].append(comarca)

    # --- 2. Definició del Nivell de Risc General i Textos associats (SENSE XIFRES) ---
    if max_cape >= 3000:
        nivell_risc = {"text": "Molt Alt (ALERTA VERMELLA)", "color": "#B71C1C"}; titol = "Alerta Màxima per Temps Violent"
        resum_general = f"Situació extremadament perillosa. L'energia disponible a l'atmosfera és **explosiva**, molt favorable per a la formació de supercèl·lules i altres fenòmens de temps violent, especialment a la zona de **{comarca_max_cape}**."
        fenomens = ["Tempestes amb potencial de formar-se supercèl·lules.", "Calamarsa de grans dimensions (>4 cm) i/o pedra.", "Ratxes de vent huracanades (esclafits > 120 km/h).", "Alt risc de tornados i aiguats torrencials."]
        recomanacio = "ALERTA MÀXIMA: Busqueu refugi IMMEDIATAMENT. No sortiu a l'exterior sota cap concepte. Risc extrem per a la vida i els béns."
    elif max_cape >= 2500:
        nivell_risc = {"text": "Alt (ALERTA TARONJA)", "color": "#DC3545"}; titol = "Alerta per Tempestes Molt Fortes i Organitzades"
        resum_general = f"Situació d'alt risc. L'atmosfera conté **energia suficient per a generar tempestes severes i organitzades**. El focus de major perill se situa a la zona de **{comarca_max_cape}**."
        fenomens = ["Tempestes organitzades (sistemes multicel·lulars).", "Calamarsa o pedra (>2 cm).", "Fortes ratxes de vent (>90 km/h).", "Xàfecs d'intensitat torrencial (>40 l/m² en 1h)."]
        recomanacio = "PRECAUCIÓ EXTREMA: Eviteu qualsevol desplaçament. Assegureu objectes a l'exterior. Allunyeu-vos de rieres i zones inundables."
    elif max_cape >= 2000:
        nivell_risc = {"text": "Alt (AVÍS GROC FORT)", "color": "#FD7E14"}; titol = "Avís per Tempestes Fortes amb Fenòmens Severs"
        resum_general = f"L'atmosfera estarà **molt inestable**, permetent el desenvolupament de tempestes fortes amb capacitat de generar fenòmens severs de manera localitzada. La zona de **{comarca_max_cape}** presenta el major potencial."
        fenomens = ["Tempestes fortes, localment severes.", "Probabilitat alta de calamarsa.", "Ratxes de vent fortes (>70 km/h).", "Xàfecs intensos."]
        recomanacio = "ATENCIÓ: Suspeneu activitats a l'aire lliure a les zones de risc. Molta precaució a la carretera. Risc de caiguda de branques."
    elif max_cape >= 1500:
        nivell_risc = {"text": "Moderat (AVÍS GROC)", "color": "#FFC107"}; titol = "Avís per Tempestes amb Possibilitat de Calamarsa"
        resum_general = f"S'esperen xàfecs i tempestes localment fortes. L'energia disponible és **notable** i suficient per a la formació de calamarsa, especialment a la rodalia de **{comarca_max_cape}**."
        fenomens = ["Tempestes localment fortes.", "Possible calamarsa o pedra petita.", "Ratxes de vent moderades a fortes.", "Activitat elèctrica abundant."]
        recomanacio = "PRUDÈNCIA: Estigueu atents a l'evolució del temps. Les activitats a l'aire lliure poden veure's afectades. Protegiu vehicles del possible impacte de calamarsa."
    elif max_cape >= 1000:
        nivell_risc = {"text": "Moderat", "color": "#28a745"}; titol = "Preavís per Xàfecs i Tronades Intenses"
        resum_general = "La inestabilitat serà **considerable**, donant lloc a xàfecs i tempestes d'intensitat moderada a forta, sobretot a la tarda."
        fenomens = ["Xàfecs forts i tronades.", "Activitat elèctrica.", "Possibles ratxes de vent fortes puntuals."]
        recomanacio = "SEGUIMENT: Consulteu les previsions periòdicament. No es requereixen mesures especials, però sí estar informat."
    elif max_cape >= 500:
        nivell_risc = {"text": "Baix", "color": "#6495ED"}; titol = "Seguiment per Possibles Xàfecs"
        resum_general = "L'atmosfera presenta un grau d'inestabilitat **baix**. Es podrien formar alguns xàfecs dispersos, generalment de curta durada i poca intensitat."
        fenomens = ["Ruixats i xàfecs dispersos.", "Alguna tronada aïllada."]
        recomanacio = "Situació sense risc destacable. No cal prendre precaucions especials."
    elif max_cape >= 20:
        nivell_risc = {"text": "Molt Baix", "color": "#adb5bd"}; titol = "Possibles Cúmuls Convectius"
        resum_general = "L'atmosfera presenta una **energia mínima**, suficient només per a la formació de núvols de tipus cúmul amb un desenvolupament vertical molt limitat."
        fenomens = ["Cúmuls de bon temps (cumulus humilis).", "Possibles gotellades o ruixats molt febles i aïllats."]
        recomanacio = "Temps estable. No es requereixen precaucions."
    else: # max_cape < 20
        nivell_risc = {"text": "Sense Risc", "color": "#6c757d"}; titol = "Situació Plenament Estable"
        resum_general = "L'energia a l'atmosfera és **pràcticament inexistent**, impedint el desenvolupament de qualsevol tipus de convecció."
        fenomens = ["Cel poc ennuvolat o serè."]; recomanacio = "No es requereixen precaucions."
        
    return {
        "nivell_risc": nivell_risc, "titol": titol, "resum_general": resum_general,
        "zones_afectades": {k: v for k, v in zones_per_risc.items() if v},
        "fenomens_previstos": fenomens, "recomanacio": recomanacio
    }
    


def ui_bulleti_automatic(bulleti_data):
    """
    Mostra el butlletí generat amb un format d'alerta complet i professional.
    VERSIÓ FINAL SENSE CRONOLOGIA.
    """
    st.markdown("---")
    with st.container(border=True):
        st.markdown("##### 📢 Butlletí d'Avisos per a Catalunya")
        
        st.markdown(f"""
        <div style="padding: 10px; background-color: {bulleti_data['nivell_risc']['color']}; border-radius: 5px; text-align: center; margin-bottom: 15px;">
            <h4 style="color: white; margin: 0; text-shadow: 1px 1px 2px black;">Nivell de Risc General: {bulleti_data['nivell_risc']['text']}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader(bulleti_data['titol'])
        
        # Ajustem les columnes per donar més espai al resum
        col1, col2 = st.columns([0.6, 0.4], gap="large") 
        with col1:
            st.markdown("###### 📝 **Resum de la Situació General**")
            st.markdown(bulleti_data['resum_general'])

        with col2:
            st.markdown("###### ⛈️ **Fenòmens Previstos**")
            for fenomen in bulleti_data['fenomens_previstos']:
                st.markdown(f"- {fenomen}")

        if bulleti_data['zones_afectades']:
            st.markdown("---")
            st.markdown("###### 🗺️ **Comarques Afectades per Nivell de Risc**")
            
            nivells_amb_dades = list(bulleti_data['zones_afectades'].keys())
            if nivells_amb_dades:
                num_cols = min(len(nivells_amb_dades), 3)
                cols = st.columns(num_cols)
                
                for i, nivell in enumerate(nivells_amb_dades):
                    with cols[i % num_cols]:
                        if "VERMELL" in nivell: color = "#B71C1C"
                        elif "TARONJA" in nivell: color = "#DC3545"
                        elif "GROC FORT" in nivell: color = "#FD7E14"
                        elif "GROC" in nivell: color = "#FFC107"
                        elif "MODERAT" in nivell: color = "#28a745"
                        else: color = "#6495ED"
                        
                        st.markdown(f"<p style='color:{color}; font-weight:bold; font-size:0.9em;'>{nivell}</p>", unsafe_allow_html=True)
                        llista_comarques = ", ".join(sorted(bulleti_data['zones_afectades'][nivell]))
                        st.markdown(f"<p style='font-size:0.85em;'>{llista_comarques}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("###### ⚠️ **Recomanacions a la Població**")
        st.warning(bulleti_data['recomanacio'])





@st.cache_data(ttl=1800, show_spinner="Analitzant les condicions marítimes...")
def carregar_dades_maritimes(hourly_index):
    """
    Obté les dades clau (CAPE) per als tres punts marítims definits.
    Retorna un diccionari amb les dades per a cada punt.
    """
    punts_maritims = {
        "Punt Marítim Nord": CIUTATS_CATALUNYA["Punt Marítim Nord"],
        "Punt Marítim Central": CIUTATS_CATALUNYA["Punt Marítim Central"],
        "Punt Marítim Sud": CIUTATS_CATALUNYA["Punt Marítim Sud"]
    }
    
    dades_resultat = {}
    
    for nom, coords in punts_maritims.items():
        data_tuple, _, _ = carregar_dades_sondeig_cat(coords['lat'], coords['lon'], hourly_index)
        if data_tuple:
            params = data_tuple[1]
            dades_resultat[nom] = {
                "cape": params.get('MLCAPE', 0),
                "lat": coords['lat'],
                "lon": coords['lon']
            }
        else:
            dades_resultat[nom] = {"cape": 0, "lat": coords['lat'], "lon": coords['lon']}
            
    return dades_resultat





def ui_vista_maritima(hourly_index):
    """
    Mostra la interfície d'anàlisi per a les zones marítimes amb el mapa de fons clar
    i completament bloquejat.
    """
    st.markdown("#### Mapa de Situació Marítima")
    
    dades_mar = carregar_dades_maritimes(hourly_index)
    
    # ===== PARÀMETRES DEL MAPA MODIFICATS AQUÍ =====
    map_params = {
        "location": [41.4, 2.5], 
        "zoom_start": 8,
        
        # 1. Canviem el fons del mapa a un de clar
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; and the GIS User Community",
        
        # 2. Bloquegem completament la interactivitat
        "scrollWheelZoom": False,
        "dragging": False,
        "zoom_control": False,
        "doubleClickZoom": False,
        "min_zoom": 8, # Fixem el zoom
        "max_zoom": 8  # Fixem el zoom
    }
    # ===============================================
    
    m = folium.Map(**map_params)
    
    # Afegim les comarques com a referència visual de fons
    gdf = carregar_dades_geografiques()
    if gdf is not None:
        folium.GeoJson(
            gdf,
            style_function=lambda x: {'color': '#555', 'weight': 1, 'fillOpacity': 0.1, 'fillColor': '#EEE'},
        ).add_to(m)

    # Afegim marcadors per a cada punt marítim
    for nom, data in dades_mar.items():
        color, _ = get_color_from_cape(data['cape'])
        icon_html = f"""
        <div style="background-color: {color}; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; border: 2px solid white; box-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🌊
        </div>
        """
        icon = folium.DivIcon(html=icon_html)
        folium.Marker(
            location=[data['lat'], data['lon']],
            icon=icon,
            tooltip=f"{nom} | CAPE: {int(data['cape'])} J/kg"
        ).add_to(m)

    st_folium(m, width="100%", height=450)
    
    st.markdown("---")
    st.markdown("##### Selecciona un punt per a una anàlisi detallada:")
    
    cols = st.columns(3)
    with cols[0]:
        st.button(f"🌊 Analitzar Mar Nord (Girona)", key="btn_mar_nord", use_container_width=True,
                  on_click=seleccionar_poble, args=("Punt Marítim Nord",))
    with cols[1]:
        st.button(f"🌊 Analitzar Mar Central (Bcn)", key="btn_mar_central", use_container_width=True,
                  on_click=seleccionar_poble, args=("Punt Marítim Central",))
    with cols[2]:
        st.button(f"🌊 Analitzar Mar Sud (Tgn)", key="btn_mar_sud", use_container_width=True,
                  on_click=seleccionar_poble, args=("Punt Marítim Sud",))
                  


def run_catalunya_app():
    """
    Versió Final amb selector de vista Terra/Mar i correcció del bug de crida a la funció de mapes.
    """
    # --- PAS 1: CAPÇALERA I INICIALITZACIÓ D'ESTAT ---
    ui_capcalera_selectors(None, zona_activa="catalunya")
    if 'selected_area' not in st.session_state: st.session_state.selected_area = "--- Selecciona una zona al mapa ---"
    if 'poble_sel' not in st.session_state: st.session_state.poble_sel = "--- Selecciona una localitat ---"
    
    # --- PAS 2: SELECTORS GLOBALS I CÀLCUL DE TEMPS ---
    with st.container(border=True):
        col_dia, col_hora = st.columns(2) # Eliminem la columna de nivell d'aquí
        with col_dia:
            dies_disponibles = [(datetime.now(TIMEZONE_CAT) + timedelta(days=i)).strftime('%d/%m/%Y') for i in range(2)]
            dia_sel_str = st.selectbox("Dia:", options=dies_disponibles, key="dia_selector")
        with col_hora:
            hora_sel_str = st.selectbox("Hora:", options=[f"{h:02d}:00h" for h in range(24)], key="hora_selector", index=datetime.now(TIMEZONE_CAT).hour)
    
    st.caption("ℹ️ Dades del model AROME 2.5km. L'aplicació es refresca cada 10 minuts.")
    target_date = datetime.strptime(dia_sel_str, '%d/%m/%Y').date()
    hora_num = int(hora_sel_str.split(':')[0])
    local_dt = TIMEZONE_CAT.localize(datetime.combine(target_date, datetime.min.time()).replace(hour=hora_num))
    start_of_today_utc = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    hourly_index_sel = int((local_dt.astimezone(pytz.utc) - start_of_today_utc).total_seconds() / 3600)

    # --- PAS 3: LÒGICA PRINCIPAL (VISTA DETALLADA O VISTA DE MAPA) ---
    if st.session_state.poble_sel and "---" not in st.session_state.poble_sel:
        # --- VISTA D'ANÀLISI DETALLADA D'UNA LOCALITAT (TERRA O MAR) ---
        poble_sel = st.session_state.poble_sel
        st.success(f"### Anàlisi per a: {poble_sel}")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1: st.button("⬅️ Tornar a la Vista Anterior", on_click=tornar_a_seleccio_comarca, use_container_width=True)
        with col_nav2: st.button("🗺️ Tornar al Mapa General", on_click=tornar_al_mapa_general, use_container_width=True)
        timestamp_str = f"{poble_sel} | {dia_sel_str} a les {hora_sel_str} (Local)"
        
        # El nivell per a l'anàlisi vertical el definim aquí
        nivell_analisi_vertical = 925
        
        with st.spinner(f"Carregant dades del sondeig i mapa per a {poble_sel}..."):
            lat_sel, lon_sel = CIUTATS_CATALUNYA[poble_sel]['lat'], CIUTATS_CATALUNYA[poble_sel]['lon']
            data_tuple, final_index, error_msg_sounding = carregar_dades_sondeig_cat(lat_sel, lon_sel, hourly_index_sel)
            map_data_conv, error_msg_map = carregar_dades_mapa_cat(nivell_analisi_vertical, hourly_index_sel)

        if error_msg_sounding or not data_tuple:
            st.error(f"No s'ha pogut carregar el sondeig: {error_msg_sounding if error_msg_sounding else 'Dades no disponibles.'}")
            return
            
        params_calculats = data_tuple[1]
        
        if not error_msg_map and map_data_conv:
            conv_puntual = calcular_convergencia_puntual(map_data_conv, lat_sel, lon_sel)
            if pd.notna(conv_puntual):
                params_calculats[f'CONV_{nivell_analisi_vertical}hPa'] = conv_puntual

        if final_index is not None and final_index != hourly_index_sel:
            adjusted_utc = start_of_today_utc + timedelta(hours=final_index)
            adjusted_local_time = adjusted_utc.astimezone(TIMEZONE_CAT)
            st.warning(f"Avís: Dades no disponibles per a les {hora_sel_str}. Es mostren les de l'hora vàlida més propera: {adjusted_local_time.strftime('%H:%Mh')}.")
        
        menu_options = ["Anàlisi Comarcal", "Anàlisi Vertical", "Anàlisi Orogràfica", "Anàlisi de Mapes", "Simulació de Núvol"]
        menu_icons = ["fullscreen", "graph-up-arrow", "bar-chart-line", "map", "cloud-upload"]
        active_tab = option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", orientation="horizontal", key=f'option_menu_{poble_sel}')
        
        if active_tab == "Anàlisi Comarcal":
            with st.spinner("Carregant anàlisi comarcal completa..."):
                alertes_totals = calcular_alertes_per_comarca(hourly_index_sel, nivell_analisi_vertical)
            comarca_actual = get_comarca_for_poble(poble_sel)
            if comarca_actual:
                valor_conv_comarcal = alertes_totals.get(comarca_actual, {}).get('conv', 0)
                ui_pestanya_analisi_comarcal(comarca_actual, valor_conv_comarcal, poble_sel, timestamp_str, nivell_analisi_vertical, map_data_conv, params_calculats, hora_sel_str, data_tuple, alertes_totals)
        
        elif active_tab == "Anàlisi Vertical":
            ui_pestanya_vertical(data_tuple, poble_sel, lat_sel, lon_sel, nivell_analisi_vertical, hora_sel_str, timestamp_str)
        
        elif active_tab == "Anàlisi Orogràfica":
            ui_pestanya_orografia(data_tuple, poble_sel, timestamp_str, params_calculats)

        elif active_tab == "Anàlisi de Mapes":
            # --- LÍNIA CORREGIDA ---
            # La crida ara només passa els dos arguments que la funció espera.
            ui_pestanya_mapes_cat(hourly_index_sel, timestamp_str)
            # --- FI DE LA CORRECCIÓ ---

        elif active_tab == "Simulació de Núvol":
            ui_pestanya_simulacio_nuvol(params_calculats, timestamp_str, poble_sel)
    else: 
        # ... (la resta de la teva funció run_catalunya_app per a la vista general es manté igual) ...
        # Aquesta part ja estava correcta.
        st.session_state.setdefault('analysis_view', 'Terra')
        st.session_state.setdefault('show_comarca_labels', False)
        st.session_state.setdefault('alert_filter_level_cape', 'Energia Baixa i superior')

        with st.container(border=True):
            st.markdown("##### Opcions de Visualització del Mapa")
            
            num_cols = 3 if st.session_state.get('analysis_view', 'Terra') == 'Terra' else 1
            cols = st.columns(num_cols)
            
            with cols[0]:
                st.selectbox("Tipus d'Anàlisi:", options=["Terra", "Mar"], key="analysis_view")
            
            if st.session_state.analysis_view == 'Terra':
                with cols[1]:
                    st.selectbox("Filtrar per nivell d'energia (CAPE):", options=["Tots", "Energia Baixa i superior", "Energia Moderada i superior", "Energia Alta i superior", "Només Extrems"], key="alert_filter_level_cape")
                with cols[2]:
                    st.toggle("Mostrar detalls de les zones actives", key="show_comarca_labels")
        
        if st.session_state.analysis_view == 'Terra':
            # Nivell per defecte per al mapa general
            nivell_mapa_general = 925 
            with st.spinner("Analitzant focus de convergència a tot Catalunya..."):
                alertes_totals = calcular_alertes_per_comarca(hourly_index_sel, nivell_mapa_general)
            
            LLINDARS_CAPE = {"Tots": 0, "Energia Baixa i superior": 500, "Energia Moderada i superior": 1000, "Energia Alta i superior": 2000, "Només Extrems": 3500}
            llindar_cape_sel = LLINDARS_CAPE.get(st.session_state.alert_filter_level_cape, 500)
            alertes_filtrades = {zona: data for zona, data in alertes_totals.items() if data['cape'] >= llindar_cape_sel}
            
            with st.spinner("Dibuixant mapa interactiu..."):
                map_output = ui_mapa_display_personalitzat(alertes_per_zona=alertes_filtrades, hourly_index=hourly_index_sel, show_labels=st.session_state.show_comarca_labels)
            
            ui_llegenda_mapa_principal()
            
            bulleti_data = generar_bulleti_automatic_catalunya(alertes_totals, hora_sel_str)
            ui_bulleti_automatic(bulleti_data)
            
            if map_output and map_output.get("last_object_clicked_tooltip"):
                raw_tooltip = map_output["last_object_clicked_tooltip"]
                if "Comarca:" in raw_tooltip or "Zona:" in raw_tooltip:
                    clicked_area = raw_tooltip.split(':')[-1].strip().replace('.', '')
                    if clicked_area != st.session_state.get('selected_area'):
                        st.session_state.selected_area = clicked_area
                        st.rerun()
            
            selected_area = st.session_state.get('selected_area')
            if selected_area and "---" not in selected_area:
                st.markdown(f"##### Selecciona una localitat a {selected_area}:")
                gdf = carregar_dades_geografiques()
                property_name = next((prop for prop in ['nom_zona', 'nom_comar', 'nomcomar'] if prop in gdf.columns), 'nom_comar')
                poblacions_dict = CIUTATS_PER_ZONA_PERSONALITZADA if property_name == 'nom_zona' else CIUTATS_PER_COMARCA
                poblacions_a_mostrar = poblacions_dict.get(selected_area.strip().replace('.', ''), {})
                if poblacions_a_mostrar:
                    cols_pobles = st.columns(4)
                    for i, nom_poble in enumerate(sorted(poblacions_a_mostrar.keys())):
                        with cols_pobles[i % 4]:
                            st.button(nom_poble, key=f"btn_{nom_poble.replace(' ', '_')}", on_click=seleccionar_poble, args=(nom_poble,), use_container_width=True)
                else:
                    st.warning("Aquesta zona no té localitats predefinides per a l'anàlisi.")
                if st.button("⬅️ Veure totes les zones"):
                    st.session_state.selected_area = "--- Selecciona una zona al mapa ---"
                    st.rerun()
            else:
                st.info("Fes clic en una zona del mapa per veure'n les localitats.", icon="👆")
        
        else:
            ui_vista_maritima(hourly_index_sel)









def analitzar_orografia(poble_sel, data_tuple):
    """
    Algoritme v18.2 (Correcció AttributeError).
    - Soluciona el bug crític que passava en intentar accedir a l'atribut '.ms' inexistent
      en les variables de vent. S'ha corregit per a utilitzar l'atribut correcte '.m'.
    """
    if not data_tuple: return {"error": "Falten dades del sondeig."}
    sounding_data, params_calc = data_tuple
    p, T, Td, u, v, heights, _, _ = sounding_data
    
    poble_coords = CIUTATS_CATALUNYA[poble_sel]
    
    try:
        if poble_coords.get('lat', 0) > 42 and np.min(heights.m) > 700:
            altura_superficie = heights.m[0]; mask_capa_baixa = (heights.m >= altura_superficie) & (heights.m <= altura_superficie + 1500)
            u_dom, v_dom = (np.mean(u[mask_capa_baixa]), np.mean(v[mask_capa_baixa])) if np.any(mask_capa_baixa) else (u[0], v[0])
        else:
            u_dom, v_dom = (u[0], v[0]) if p.m.min() > 850 else (np.interp(850, p.m[::-1], u.m[::-1]) * units('m/s'), np.interp(850, p.m[::-1], v.m[::-1]) * units('m/s'))
        wind_dir_from_real = mpcalc.wind_direction(u_dom, v_dom).m
        wind_spd_kmh = mpcalc.wind_speed(u_dom, v_dom).to('km/h').m
    except Exception: return {"error": "No s'ha pogut determinar el vent dominant."}
    
    BEARING_FIXE_PER_AL_TALL = 14.0
    start_point_bearing = (BEARING_FIXE_PER_AL_TALL + 180) % 360
    lat_inici, lon_inici = punt_desti(poble_coords['lat'], poble_coords['lon'], start_point_bearing, 40)
    profile_data, error = get_elevation_profile(lat_inici, lon_inici, BEARING_FIXE_PER_AL_TALL, 80, 100)
    if error: return {"error": error}
    
    elevations = np.array(profile_data['elevations']); distances = np.array(profile_data['distances'])
    dist_al_poble = [haversine_distance(poble_coords['lat'], poble_coords['lon'], lat, lon) for lat, lon in zip(profile_data['lats'], profile_data['lons'])]
    idx_poble = np.argmin(dist_al_poble); elev_poble_precisa = elevations[idx_poble]
    
    punts_dinteres_etiquetes = []
    comarca = get_comarca_for_poble(poble_sel)
    cims_de_la_comarca = PICS_CATALUNYA_PER_COMARCA.get(comarca, [])
    cims_propers = [pic for pic in cims_de_la_comarca if np.min([haversine_distance(pic['lat'], pic['lon'], lat_t, lon_t) for lat_t, lon_t in zip(profile_data['lats'], profile_data['lons'])]) < 10]
    for cim in cims_propers:
        dist_al_cim = [haversine_distance(cim['lat'], cim['lon'], lat_t, lon_t) for lat_t, lon_t in zip(profile_data['lats'], profile_data['lons'])]
        idx_cim_proper = np.argmin(dist_al_cim)
        indices_pics_locals, _ = find_peaks(elevations[max(0, idx_cim_proper-10):idx_cim_proper+10], prominence=100)
        if any(p + max(0, idx_cim_proper-10) == idx_cim_proper for p in indices_pics_locals):
            punts_dinteres_etiquetes.append({"name": cim['name'], "ele_oficial": cim['ele'], "idx": idx_cim_proper})

    if wind_spd_kmh < 15: posicio, diagnostico, detalls = "Indeterminat (Vent Feble)", "Efecte Orogràfic Menyspreable", "El vent és massa feble per a interactuar de manera significativa amb el terreny."
    else:
        idx_inici_analisi = max(0, idx_poble - 20)
        if idx_poble > idx_inici_analisi + 5:
            dist_tram = distances[idx_inici_analisi:idx_poble]; elev_tram = elevations[idx_inici_analisi:idx_poble]; pendent = np.polyfit(dist_tram * 1000, elev_tram, 1)[0]
            if abs(pendent) < 0.01: posicio, diagnostico, detalls = "Plana / Vall Oberta", "Sense Efecte Orogràfic Dominant", "El flux de vent travessa un terreny majoritàriament pla."
            elif pendent > 0.03: posicio, diagnostico, detalls = "Sobrevent", "Ascens Orogràfic Forçat", "El flux d'aire impacta contra un pendent ascendent, forçant l'ascens, refredament i possible condensació."
            elif pendent < -0.03: posicio, diagnostico, detalls = "Sotavent", "Subsidència i Efecte Foehn", "L'aire descendeix, comprimint-se i reescalfant-se, la qual cosa dissipa la nuvolositat i asseca l'ambient."
            else: posicio, diagnostico, detalls = "Flux Paral·lel / Terreny Ondulat", "Efectes Locals Menors", "El vent es mou sobre un terreny amb ondulacions suaus sense un forçament a gran escala."
        else: posicio, diagnostico, detalls = "Plana / Vall Oberta", "Sense Efecte Orogràfic Dominant", "La localitat es troba en una zona oberta."
        
    rh_profile = (mpcalc.relative_humidity_from_dewpoint(T, Td) * 100).m
    wind_speed_profile = mpcalc.wind_speed(u, v).to('km/h').m; temp_profile = T.m
    theta_e_profile = mpcalc.equivalent_potential_temperature(p, T, Td).to('K').m
    parcel_profile = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC').m
    
    # --- LÍNIA CORREGIDA ---
    # S'ha canviat u.ms per u.m i v.ms per v.m
    sondeig_complet_per_grafic = (heights.m, u.m, v.m, rh_profile, temp_profile, wind_speed_profile, theta_e_profile, parcel_profile)
    # --- FI DE LA CORRECCIÓ ---
    
    cloud_layers = analitzar_formacio_nuvols(sounding_data, params_calc)

    return {
        "transect_distances": distances, "transect_elevations": elevations, "poble_sel": poble_sel, 
        "poble_dist": distances[idx_poble], "poble_elev": elev_poble_precisa,
        "wind_dir_from": wind_dir_from_real, "wind_spd_kmh": wind_spd_kmh, "posicio": posicio, 
        "diagnostico": diagnostico, "detalls": detalls, "punts_dinteres": punts_dinteres_etiquetes,
        "sondeig_perfil_complet": sondeig_complet_per_grafic, "cloud_layers": cloud_layers,
        "bearing_fixe": BEARING_FIXE_PER_AL_TALL
    }

def analitzar_formacio_nuvols(sounding_data, params_calc):
    """
    Analitza el perfil atmosfèric per identificar i classificar les capes de núvols.
    Distingeix entre núvols estratiformes i convectius basant-se en la humitat,
    l'estabilitat i l'energia disponible (CAPE).
    """
    if not sounding_data or not params_calc:
        return []

    p, T, Td, heights, _, _, _, _ = sounding_data
    cloud_layers = []
    
    # Paràmetres per a la detecció
    RH_THRESHOLD = 85  # Humitat relativa (%) mínima per considerar un núvol
    MIN_LAYER_THICKNESS_M = 150 # Gruix mínim en metres per a una capa estratiforme

    # --- Pas 1: Detecció de núvols convectius (tempestes) ---
    cape = params_calc.get('MLCAPE', 0)
    lcl_hgt = params_calc.get('LCL_Hgt')
    el_hgt = params_calc.get('EL_Hgt')

    convective_range = (-1, -1) # Rang d'altituds ocupat per la convecció
    if cape > 100 and lcl_hgt is not None and el_hgt is not None and el_hgt > lcl_hgt:
        cloud_layers.append({
            "type": "Convectiu",
            "base_m": lcl_hgt,
            "top_m": el_hgt,
            "density": np.clip(cape / 2000, 0.2, 1.0) # La densitat/opacitat depèn del CAPE
        })
        convective_range = (lcl_hgt, el_hgt)

    # --- Pas 2: Detecció de núvols estratiformes per capes ---
    height_grid = np.arange(0, 12000, 50) # Creem una graella vertical d'alta resolució
    rh_on_grid = np.interp(height_grid, heights.m, (mpcalc.relative_humidity_from_dewpoint(T, Td) * 100).m)
    
    is_in_cloud = False
    cloud_base = 0
    for i, h in enumerate(height_grid):
        # Si estem dins del rang d'una tempesta ja detectada, no busquem núvols estratiformes
        if convective_range[0] <= h <= convective_range[1]:
            if is_in_cloud: # Si veníem d'una capa estratiforme, la tanquem just abans d'entrar a la zona convectiva
                cloud_top = height_grid[i-1]
                if cloud_top - cloud_base >= MIN_LAYER_THICKNESS_M:
                    cloud_layers.append({"type": "Estratiforme", "base_m": cloud_base, "top_m": cloud_top, "density": 0.6})
                is_in_cloud = False
            continue

        # Inici d'una nova capa de núvols
        if rh_on_grid[i] >= RH_THRESHOLD and not is_in_cloud:
            is_in_cloud = True
            cloud_base = h
        # Final d'una capa de núvols
        elif rh_on_grid[i] < RH_THRESHOLD and is_in_cloud:
            is_in_cloud = False
            cloud_top = height_grid[i-1]
            if cloud_top - cloud_base >= MIN_LAYER_THICKNESS_M:
                cloud_layers.append({"type": "Estratiforme", "base_m": cloud_base, "top_m": cloud_top, "density": 0.6})

    # Si un núvol estratiforme continua fins al final de la graella
    if is_in_cloud:
        cloud_top = height_grid[-1]
        if cloud_top - cloud_base >= MIN_LAYER_THICKNESS_M:
            cloud_layers.append({"type": "Estratiforme", "base_m": cloud_base, "top_m": cloud_top, "density": 0.6})

    return cloud_layers


def crear_grafic_perfil_orografic(analisi, params_calc, layer_to_show, max_alt_m, show_barbs=True):
    """
    Crea una secció transversal atmosfèrica amb interacció realista i capes de dades avançades.
    Versió 18.1 (Final i Completa):
    - Dibuixa un flux d'aire amb ondulació subtil i realista sobre l'orografia.
    - Utilitza una font de dades d'elevació única i precisa per al terreny i totes les etiquetes.
    - Inclou totes les capes de visualització (Vent, Humitat, Temperatura, Theta-E, CAPE, Núvols).
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#E6F2FF')

    dist_total_km = analisi['transect_distances'][-1]
    dist_centrat = analisi['transect_distances'] - (dist_total_km / 2)
    elev = analisi['transect_elevations']

    # --- Configuració de Paletes de Colors i Nivells ---
    levels_theta_e = list(range(15, 75, 5)); cmap_theta_e = plt.get_cmap('viridis'); norm_theta_e = BoundaryNorm(levels_theta_e, ncolors=cmap_theta_e.N, clip=True)
    levels_buoyancy = [-8, -6, -4, -2, -0.5, 0.5, 2, 4, 6, 8, 10, 12]; cmap_buoyancy = plt.get_cmap('bwr'); norm_buoyancy = BoundaryNorm(levels_buoyancy, ncolors=cmap_buoyancy.N, clip=True)
    colors_humitat = ['#f0e68c', '#90ee90', '#4682b4', '#191970']; levels_humitat = [0, 30, 60, 80, 101]; cmap_humitat = ListedColormap(colors_humitat); norm_humitat = BoundaryNorm(levels_humitat, ncolors=cmap_humitat.N, clip=True)
    colors_vent = ['#d3d3d3', '#add8e6', '#48d1cc', '#90ee90', '#32cd32', '#6b8e23', '#f0e68c', '#d2b48c', '#bc8f8f', '#ffb6c1', '#da70d6', '#9932cc', '#8a2be2']; levels_vent = [0, 11, 25, 40, 54, 68, 86, 104, 131]; cmap_vent = ListedColormap(colors_vent); norm_vent = BoundaryNorm(levels_vent, ncolors=cmap_vent.N, clip=True)
    colors_temp = ['#8a2be2', '#0000ff', '#1e90ff', '#00ffff', '#32cd32', '#ffff00', '#ffa500', '#ff0000', '#dc143c', '#ff00ff']; levels_temp = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]; cmap_temp = ListedColormap(colors_temp); norm_temp = BoundaryNorm(levels_temp, ncolors=cmap_temp.N, clip=True)
    
    heights_m, u_ms, v_ms, rh_profile, temp_profile, wind_speed_profile, theta_e_profile, parcel_temp_profile = analisi['sondeig_perfil_complet']
    
    x_grid = dist_centrat; y_grid = np.linspace(0, max_alt_m, 100)
    xx, zz_asl = np.meshgrid(x_grid, y_grid)
    
    # --- Algoritme de Deformació Subtil ---
    max_displacement_factor = 0.4; decay_scale_height = 3000.0
    displacement_decay = np.exp(-zz_asl / decay_scale_height)
    vertical_displacement = np.outer(np.ones(len(y_grid)), elev) * max_displacement_factor * displacement_decay
    zz_deformat = zz_asl + vertical_displacement
    
    data_grid, label, im = None, "", None
    if layer_to_show != "Núvols":
        if layer_to_show == "Humitat":
            profile_1d = np.interp(zz_deformat, heights_m, rh_profile); cmap, norm, levels, label = cmap_humitat, norm_humitat, levels_humitat, "Humitat Relativa (%)"
        elif layer_to_show == "Temperatura":
            profile_1d = np.interp(zz_deformat, heights_m, temp_profile); cmap, norm, levels, label = cmap_temp, norm_temp, levels_temp, "Temperatura (°C)"
        elif layer_to_show == "Vent":
            profile_1d = np.interp(zz_deformat, heights_m, wind_speed_profile); cmap, norm, levels, label = cmap_vent, norm_vent, levels_vent, "Velocitat del Vent (km/h)"
        elif layer_to_show == "Theta-E":
            profile_1d = np.interp(zz_deformat, heights_m, theta_e_profile - 273.15); cmap, norm, levels, label = cmap_theta_e, norm_theta_e, levels_theta_e, "Theta-E (°C)"
        elif layer_to_show == "CAPE (Flotabilitat)":
            parcel_temp_interp = np.interp(zz_deformat, heights_m, parcel_temp_profile); env_temp_interp = np.interp(zz_deformat, heights_m, temp_profile)
            profile_1d = parcel_temp_interp - env_temp_interp; cmap, norm, levels, label = cmap_buoyancy, norm_buoyancy, levels_buoyancy, "Flotabilitat (°C)"
        
        masked_data = np.where(zz_asl > elev, profile_1d, np.nan)
        im = ax.contourf(xx, zz_asl, masked_data, levels=levels, cmap=cmap, norm=norm, extend='both', zorder=1)
        contours = ax.contour(xx, zz_asl, masked_data, levels=levels[1:-1:2], colors='white', linewidths=0.6, alpha=0.8, zorder=2)
        ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')
        fig.colorbar(im, ax=ax, label=label, pad=0.02, ticks=levels[::2])
        if layer_to_show == "CAPE (Flotabilitat)":
            buoyancy_contour = ax.contour(xx, zz_asl, masked_data, levels=[2], colors=['#00FF00'], linewidths=[2.0], linestyles=['--'], zorder=4)
            clabels = ax.clabel(buoyancy_contour, inline=True, fontsize=9, fmt='+%1.0f°C Accel.')
            for label_obj in clabels:
                label_obj.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')]); label_obj.set_color('white')

    if layer_to_show == "Núvols":
        sky_gradient = np.linspace(0.8, 0.4, 100).reshape(-1, 1)
        ax.imshow(sky_gradient, aspect='auto', cmap='Blues_r', origin='lower', extent=[dist_centrat.min(), dist_centrat.max(), 0, max_alt_m], zorder=0)
        rh_grid = np.interp(zz_deformat, heights_m, rh_profile)
        cloud_density = np.clip((rh_grid - 80) / 19.9, 0, 1)
        cape = params_calc.get('MLCAPE', 0)
        if cape > 100:
            lcl = params_calc.get('LCL_Hgt', 9999); el = params_calc.get('EL_Hgt', 0)
            if el > lcl:
                pendent_1d = np.gradient(elev, dist_centrat * 1000); upslope_mask_1d = pendent_1d > 0.03
                if np.any(upslope_mask_1d):
                    convective_boost_profile = np.zeros_like(y_grid); convective_layer_mask = (y_grid >= lcl) & (y_grid <= el)
                    mid_convective_layer = (lcl + el) / 2; sigma_z = (el - lcl) * 0.4
                    convective_boost_profile[convective_layer_mask] = np.exp(-((y_grid[convective_layer_mask] - mid_convective_layer)**2 / (2 * sigma_z**2)))
                    convective_boost_2d = np.outer(convective_boost_profile, upslope_mask_1d)
                    convective_intensity = np.clip(cape / 1200, 0.5, 1.5)
                    cloud_density += convective_boost_2d * convective_intensity
        masked_density = np.where(zz_asl > elev, np.clip(cloud_density, 0, 1.2), 0)
        cloud_levels = np.linspace(0.1, 1.2, 11)
        cloud_cmap = LinearSegmentedColormap.from_list("cloud_cmap", [(1, 1, 1, 0), (1, 1, 1, 0.95)])
        ax.contourf(xx, zz_asl, masked_density, levels=cloud_levels, cmap=cloud_cmap, zorder=2)
        ax.set_title(f"Secció Transversal Atmosfèrica - Nuvolositat (HR > 85%)")

    ax.fill_between(dist_centrat, 0, elev, color='black', zorder=3)
    if np.min(elev) <= 5:
        x_wave = np.linspace(dist_centrat.min(), dist_centrat.max(), 200); y_wave = np.sin(x_wave * 0.5) * 5 + 5
        ax.fill_between(x_wave, -100, y_wave, where=y_wave > 0, color='#6495ED', alpha=0.6, zorder=2)

    is_convective = params_calc.get('MLCAPE', 0) > 400
    level_hgt = params_calc.get('LFC_Hgt') if is_convective else params_calc.get('LCL_Hgt')
    level_label = "LFC" if is_convective else "LCL"
    if level_hgt is not None and level_hgt < max_alt_m:
        ax.axhline(y=level_hgt, color='white', linestyle=':', linewidth=2, label=f"{level_label}: {level_hgt:.0f} m", zorder=4, path_effects=[path_effects.withStroke(linewidth=3.5, foreground='black')])

    if show_barbs:
        barb_x_upper = np.linspace(dist_centrat.min() + 5, dist_centrat.max() - 5, 7); barb_y_upper = np.arange(1000, max_alt_m, 500)
        barb_xx, barb_zz = np.meshgrid(barb_x_upper, barb_y_upper); terrain_at_barbs_upper = np.interp(barb_xx.flatten(), dist_centrat, elev)
        valid_mask_upper = barb_zz.flatten() > terrain_at_barbs_upper
        barb_u = np.interp(barb_zz.flatten(), heights_m, u_ms) * 1.94384; barb_v = np.interp(barb_zz.flatten(), heights_m, v_ms) * 1.94384
        ax.barbs(barb_xx.flatten()[valid_mask_upper], barb_zz.flatten()[valid_mask_upper], barb_u[valid_mask_upper], barb_v[valid_mask_upper], length=6, zorder=5, color='white', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        barb_x_surface = np.linspace(dist_centrat.min() + 2, dist_centrat.max() - 2, 10)
        surface_elev_at_barbs = np.interp(barb_x_surface, dist_centrat, elev); barb_y_surface = surface_elev_at_barbs + 250
        barb_u_surface = np.interp(barb_y_surface, heights_m, u_ms) * 1.94384; barb_v_surface = np.interp(barb_y_surface, heights_m, v_ms) * 1.94384
        mask = barb_y_surface < max_alt_m
        ax.barbs(barb_x_surface[mask], barb_y_surface[mask], barb_u_surface[mask], barb_v_surface[mask], length=6, zorder=5, color='#F0E68C', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        for x, y_terrain, y_barb in zip(barb_x_surface[mask], surface_elev_at_barbs[mask], barb_y_surface[mask]):
            ax.plot([x, x], [y_terrain, y_barb], color='#F0E68C', lw=0.6, linestyle='--', zorder=4)

    poble_dist_centrat = analisi['poble_dist'] - (dist_total_km / 2)
    ax.plot(poble_dist_centrat, analisi['poble_elev'], 'o', color='red', markersize=8, label=f"{analisi['poble_sel']} ({analisi['poble_elev']:.0f} m)", zorder=10, markeredgecolor='white')
    ax.axvline(x=poble_dist_centrat, color='red', linestyle='--', linewidth=1, zorder=1)

    for punt in analisi.get("punts_dinteres", []):
        x_pos = dist_centrat[punt['idx']]; y_pos_terreny = elev[punt['idx']]; altitud_oficial = punt['ele_oficial']
        ax.annotate(f"{punt['name']}\n({altitud_oficial:.0f} m)", xy=(x_pos, y_pos_terreny), xytext=(x_pos, y_pos_terreny + max_alt_m * 0.08),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4), ha='center', va='bottom', fontsize=8, zorder=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

    ax.set_xlabel(f"Distància (km) | Vent → ({analisi['bearing_fixe']:.0f}°)")
    ax.set_ylabel("Elevació (m)")
    if layer_to_show != "Núvols":
        ax.set_title("Secció Transversal Atmosfèrica")
    ax.grid(True, linestyle=':', alpha=0.5, color='black', zorder=0)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(bottom=0, top=max_alt_m)
    ax.set_xlim(dist_centrat.min(), dist_centrat.max())
    ax.invert_xaxis()
    plt.tight_layout()
    return fig
    
    
def ui_pestanya_orografia(data_tuple, poble_sel, timestamp_str, params_calc):
    """
    Mostra la interfície completa per a la pestanya d'Anàlisi d'Interacció Vent-Orografia,
    ara amb totes les noves opcions de visualització.
    """
    st.markdown(f"#### Anàlisi d'Interacció Vent-Orografia per a {poble_sel}")
    st.caption(timestamp_str)

    with st.spinner("Analitzant el flux d'aire i la formació de núvols sobre el terreny..."):
        analisi_orografica = analitzar_orografia(poble_sel, data_tuple)

    if "error" in analisi_orografica:
        st.error(f"No s'ha pogut realitzar l'anàlisi: {analisi_orografica['error']}")
        return
        
    with st.container(border=True):
        col_layer, col_height, col_barbs = st.columns(3)
        with col_layer:
            # --- LÍNIA MODIFICADA: AFEGIM LES NOVES OPCIONS ---
            layer_sel = st.selectbox(
                "Capa de dades a visualitzar:", 
                options=["Vent", "Humitat", "Temperatura", "Theta-E", "CAPE (Flotabilitat)", "Núvols"], 
                key="orog_layer_selector",
                help="Selecciona la variable atmosfèrica que vols visualitzar sobre el perfil del terreny."
            )
        with col_height:
            max_alt_sel = st.slider("Altura màxima del perfil (m):", 
                                    min_value=1000, max_value=12000, value=4000, step=100, 
                                    key="orog_height_slider",
                                    help="Ajusta l'eix vertical del gràfic per enfocar-te en diferents nivells de l'atmosfera.")
        with col_barbs:
            st.write("") 
            show_barbs_sel = st.toggle("Mostrar Barbes de Vent", value=True, 
                                       key="show_barbs_toggle",
                                       help="Activa o desactiva la visualització dels vectors de vent (barbes) sobre el gràfic.")

    # La resta de la funció (dibuix del gràfic i diagnòstic) es manté igual
    col1, col2 = st.columns([0.65, 0.35], gap="large")
    with col1:
        st.markdown("##### Perfil del Terreny i Flux Atmosfèric")
        if "transect_distances" in analisi_orografica:
            fig = crear_grafic_perfil_orografic(analisi_orografica, params_calc, layer_sel, max_alt_sel, show_barbs=show_barbs_sel)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    
    with col2:
        st.markdown("##### Diagnòstic Orogràfic")
        posicio = analisi_orografica.get("posicio", "Indeterminat")
        diagnostico = analisi_orografica.get("diagnostico", "Sense diagnòstic clar.")
        detalls = analisi_orografica.get("detalls", "Les condicions no permeten un diagnòstic detallat.")
        
        if posicio == "Sobrevent": color, emoji = "#28a745", "🔼"
        elif posicio == "Sotavent": color, emoji = "#fd7e14", "🔽"
        elif posicio in ["Exposat al Cim / Carena", "Flux Paral·lel"]: color, emoji = "#007bff", "💨"
        else: color, emoji = "#6c757d", "↔️"
            
        st.markdown(f"""
        <div style="padding: 12px; background-color: #2a2c34; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 15px;">
             <span style="font-size: 1.2em; color: #FAFAFA;">{emoji} Posició Relativa: <strong style="color:{color}">{posicio}</strong></span>
             <h6 style="color: white; margin-top: 10px; margin-bottom: 5px;">Diagnòstic: {diagnostico}</h6>
             <p style="font-size:0.95em; color:#a0a0b0; text-align: left;">{detalls}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("###### Detalls del Flux:")
        st.metric(
            label="Direcció del Vent Dominant", 
            value=f"{analisi_orografica['wind_dir_from']:.0f}° ({graus_a_direccio_cardinal(analisi_orografica['wind_dir_from'])})"
        )
        st.metric(
            label="Velocitat del Vent Dominant", 
            value=f"{analisi_orografica['wind_spd_kmh']:.0f} km/h"
        )
        st.caption("Vent dominant calculat de manera adaptativa segons l'entorn.")
        

@st.cache_data(ttl=86400, show_spinner="Obtenint perfil del terreny...")
def get_elevation_profile(lat_start, lon_start, bearing_deg, distance_km=80, num_points=100):
    """
    Obté un perfil d'elevació al llarg d'un transecte des d'un punt inicial.
    Retorna les coordenades dels punts, les seves distàncies i les seves elevacions.
    """
    R = 6371.0
    lats, lons, dists = [], [], []
    lat_start_rad, lon_start_rad, bearing_rad = map(radians, [lat_start, lon_start, bearing_deg])

    for i in range(num_points):
        d = i * (distance_km / (num_points - 1))
        dists.append(d)
        lat_rad = asin(sin(lat_start_rad) * cos(d / R) + cos(lat_start_rad) * sin(d / R) * cos(bearing_rad))
        lon_rad = lon_start_rad + atan2(sin(bearing_rad) * sin(d / R) * cos(lat_start_rad), cos(d / R) - sin(lat_start_rad) * sin(lat_rad))
        lats.append(degrees(lat_rad)); lons.append(degrees(lon_rad))
        
    try:
        response = requests.get("https://api.open-meteo.com/v1/elevation", params={"latitude": lats, "longitude": lons}, timeout=10)
        response.raise_for_status()
        elevations = response.json().get('elevation', [0] * num_points)
        return {"distances": dists, "elevations": elevations, "lats": lats, "lons": lons}, None
    except Exception as e:
        return None, f"Error obtenint dades d'elevació: {e}"


def punt_desti(lat, lon, bearing, distance_km):
    """
    Calcula les coordenades d'un punt de destinació donat un punt de partida,
    una direcció (bearing) i una distància. Funció auxiliar per a l'anàlisi orogràfica.
    """
    R = 6371.0
    lat, lon, bearing = map(radians, [lat, lon, bearing])
    lat2 = asin(sin(lat) * cos(distance_km / R) + cos(lat) * sin(distance_km / R) * cos(bearing))
    lon2 = lon + atan2(sin(bearing) * sin(distance_km / R) * cos(lat), cos(distance_km / R) - sin(lat) * sin(lat2))
    return degrees(lat2), degrees(lon2)


def _is_wind_onshore(wind_dir_from, sea_dir_range):
    """Funció auxiliar per a comprovar si el vent ve del mar."""
    if sea_dir_range is None:
        return False
    start, end = sea_dir_range
    # Cas normal (ex: 90 a 180)
    if start <= end:
        return start <= wind_dir_from <= end
    # Cas on el rang creua els 360 graus (ex: 330 a 45)
    else:
        return start <= wind_dir_from or wind_dir_from <= end





def punt_desti(lat, lon, bearing, distance_km):
    """
    Calcula les coordenades d'un punt de destinació donat un punt de partida,
    una direcció (bearing) i una distància. Funció auxiliar per a l'anàlisi orogràfica.
    """
    R = 6371.0
    lat, lon, bearing = map(radians, [lat, lon, bearing])
    lat2 = asin(sin(lat) * cos(distance_km / R) + cos(lat) * sin(distance_km / R) * cos(bearing))
    lon2 = lon + atan2(sin(bearing) * sin(distance_km / R) * cos(lat), cos(distance_km / R) - sin(lat) * sin(lat2))
    return degrees(lat2), degrees(lon2)







def _is_wind_onshore(wind_dir_from, sea_dir_range):
    """Funció auxiliar per a comprovar si el vent ve del mar."""
    if sea_dir_range is None:
        return False
    start, end = sea_dir_range
    # Cas normal (ex: 90 a 180)
    if start <= end:
        return start <= wind_dir_from <= end
    # Cas on el rang creua els 360 graus (ex: 330 a 45)
    else:
        return start <= wind_dir_from or wind_dir_from <= end
        


    


def get_color_from_cape(cape_value):
    """
    Retorna un color i un color de text basat en un valor de CAPE,
    seguint els llindars d'alerta.
    """
    if not isinstance(cape_value, (int, float, np.number)) or pd.isna(cape_value) or cape_value < 500:
        return '#6c757d', '#FFFFFF'  # Gris (Baix / Sense Risc)
    if cape_value < 1000:
        return '#28A745', '#FFFFFF'  # Verd (Moderat)
    if cape_value < 1500:
        return '#FFC107', '#000000'  # Groc (Groc)
    if cape_value < 2000:
        return '#FD7E14', '#FFFFFF'  # Taronja (Groc Fort)
    if cape_value < 3000:
        return '#DC3545', '#FFFFFF'  # Vermell (Taronja)
    return '#9370DB', '#FFFFFF'      # Lila/Violeta (Vermell)
    

def forcar_regeneracio_animacio():
    """Incrementa la clau de regeneració per invalidar la memòria cau."""
    if 'regenerate_key' in st.session_state:
        st.session_state.regenerate_key += 1
    else:
        st.session_state.regenerate_key = 1
        
def ui_mapa_display_peninsula(alertes_per_zona, hourly_index, show_labels):
    """
    Funció de VISUALITZACIÓ específica per al mapa de l'Est Peninsular.
    (Versió Final: Només mostra les províncies analitzades, la resta són invisibles)
    """
    st.markdown("#### Mapa de Situació")
    
    selected_area_str = st.session_state.get('selected_area_peninsula')

    alertes_tuple = tuple(sorted((k, float(v)) for k, v in alertes_per_zona.items()))
    
    map_data = preparar_dades_mapa_peninsula_cachejat(
        alertes_tuple, 
        selected_area_str, 
        show_labels
    )
    
    if not map_data:
        return None

    map_params = {
        "location": [40.8, -1.0], "zoom_start": 7,
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; and the GIS User Community",
        "scrollWheelZoom": True, "dragging": True, "zoom_control": True, "doubleClickZoom": True,
        "max_bounds": [[38.0, -4.5], [43.5, 1.5]], "min_zoom": 7, "max_zoom": 12
    }

    m = folium.Map(**map_params)

    def style_function(feature):
        nom_feature_raw = feature.get('properties', {}).get(map_data["property_name"])
        
        # --- LÒGICA MILLORADA PER A LA VISIBILITAT ---
        # 1. Comprovem si la província del GeoJSON és una de les que analitzem
        if nom_feature_raw and nom_feature_raw.strip() in CIUTATS_PER_ZONA_PENINSULA:
            # 2. Si ho és, apliquem tota la lògica de colors habitual
            nom_feature = nom_feature_raw.strip().replace('.', '')
            style = map_data["styles"].get(nom_feature, {'fillColor': '#6c757d', 'color': '#495057', 'weight': 1, 'fillOpacity': 0.25})
            
            cleaned_selected_area = selected_area_str.strip().replace('.', '') if selected_area_str else ''
            
            if nom_feature == cleaned_selected_area:
                style.update({'fillColor': '#FFC107', 'color': '#000000', 'weight': 3, 'fillOpacity': 0.6})
            
            return style
        else:
            # 3. Si NO és una de les nostres províncies, la fem completament invisible.
            return {'fillOpacity': 0, 'weight': 0, 'color': 'transparent'}
        # --- FI DE LA LÒGICA MILLORADA ---

    folium.GeoJson(
        map_data["gdf"], style_function=style_function,
        highlight_function=lambda x: {'color': '#ffffff', 'weight': 3.5, 'fillOpacity': 0.5},
        tooltip=folium.GeoJsonTooltip(fields=[map_data["property_name"]], aliases=['Provincia:'])
    ).add_to(m)

    for marker in map_data["markers"]:
        icon = folium.DivIcon(html=marker['icon_html'])
        folium.Marker(location=marker['location'], icon=icon, tooltip=marker['tooltip']).add_to(m)
    
    return st_folium(m, width="100%", height=450, returned_objects=['last_object_clicked_tooltip'])




def format_slider_label(offset, now_hour):
    """ Formata l'etiqueta del slider de temps per ser més intuïtiva. """
    if offset == 0:
        return f"Ara ({now_hour:02d}:00h)"
    
    target_hour = (now_hour + offset) % 24
    sign = "+" if offset > 0 else ""
    return f"Ara {sign}{offset}h ({target_hour:02d}:00h)"


@st.cache_data(ttl=600, show_spinner="Preparant dades del mapa de la península...")
def preparar_dades_mapa_peninsula_cachejat(alertes_tuple, selected_area_str, show_labels):
    """
    Funció CACHEADA per a la península, amb nova lògica de color per a focus febles (a partir de 10).
    """
    alertes_per_zona = dict(alertes_tuple)
    
    gdf = carregar_dades_geografiques_peninsula()
    if gdf is None: 
        return None

    property_name = 'NAME_2'
    if property_name not in gdf.columns:
        st.error(f"Error de configuració del mapa: L'arxiu 'peninsula_zones.geojson' no conté la columna de propietats esperada ('{property_name}').")
        st.warning("Les columnes que s'han trobat són:", icon="ℹ️")
        st.code(f"{list(gdf.columns)}")
        st.info(f"Si us plau, modifica la variable 'property_name' a la funció 'preparar_dades_mapa_peninsula_cachejat' amb el nom correcte de la columna que conté els noms de les províncies.")
        return None

    # --- FUNCIÓ DE COLOR MODIFICADA ---
    def get_color_from_convergence(value):
        if not isinstance(value, (int, float)): return '#4a4a4a', '#FFFFFF' # Color per defecte
        if value >= 100: return '#9370DB', '#FFFFFF'  # Extrem
        if value >= 60: return '#DC3545', '#FFFFFF'   # Molt Alt
        if value >= 40: return '#FD7E14', '#FFFFFF'   # Alt
        if value >= 20: return '#28A745', '#FFFFFF'   # Moderat
        if value >= 10: return '#6495ED', '#FFFFFF'   # Blau clar per a focus d'interès (10-19)
        return '#4a4a4a', '#FFFFFF'
    # --- FI DE LA MODIFICACIÓ ---

    styles_dict = {}
    for feature in gdf.iterfeatures():
        nom_feature_raw = feature.get('properties', {}).get(property_name)
        if nom_feature_raw and isinstance(nom_feature_raw, str):
            nom_feature = nom_feature_raw.strip().replace('.', '')
            conv_value = alertes_per_zona.get(nom_feature)
            alert_color, _ = get_color_from_convergence(conv_value)
            
            fill_opacity = 0.55 if conv_value and conv_value >= 10 else 0.25
            
            styles_dict[nom_feature] = {
                'fillColor': alert_color, 'color': alert_color,
                'fillOpacity': fill_opacity,
                'weight': 2.5 if conv_value and conv_value >= 10 else 1
            }

    markers_data = []
    if show_labels:
        for zona, conv_value in alertes_per_zona.items():
            if conv_value >= 10: # Només mostrem etiqueta si hi ha focus d'interès
                capital_info = CAPITALS_ZONA_PENINSULA.get(zona)
                if capital_info:
                    bg_color, text_color = get_color_from_convergence(conv_value)
                    icon_html = f"""<div style="position: relative; background-color: {bg_color}; color: {text_color}; padding: 6px 12px; border-radius: 8px; border: 2px solid {text_color}; font-family: sans-serif; font-size: 11px; font-weight: bold; text-align: center; min-width: 80px; box-shadow: 3px 3px 5px rgba(0,0,0,0.5); transform: translate(-50%, -100%);"><div style="position: absolute; bottom: -10px; left: 50%; transform: translateX(-50%); width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 8px solid {bg_color};"></div><div style="position: absolute; bottom: -13.5px; left: 50%; transform: translateX(-50%); width: 0; height: 0; border-left: 10px solid transparent; border-right: 10px solid transparent; border-top: 10px solid {text_color}; z-index: -1;"></div>{zona}: {conv_value:.0f}</div>"""
                    markers_data.append({
                        'location': [capital_info['lat'], capital_info['lon']],
                        'icon_html': icon_html,
                        'tooltip': f"Provincia: {zona}"
                    })

    return {
        "gdf": gdf.to_json(),
        "property_name": property_name,
        "styles": styles_dict,
        "markers": markers_data
    }


def seleccionar_poble(nom_poble):
    """Callback que s'activa en clicar un poble. Actualitza l'estat directament."""
    st.session_state.poble_sel = nom_poble
    # Reseteja la pestanya per començar sempre per "Anàlisi Comarcal"
    if 'active_tab_cat_index' in st.session_state:
        st.session_state.active_tab_cat_index = 0

def tornar_a_seleccio_comarca():
    """Callback per tornar a la vista de selecció de municipis de la comarca actual."""
    st.session_state.poble_sel = "--- Selecciona una localitat ---"
    # Reseteja la pestanya activa per evitar inconsistències visuals
    if 'active_tab_cat_index' in st.session_state:
        st.session_state.active_tab_cat_index = 0

def tornar_al_mapa_general():
    """Callback per tornar a la vista principal del mapa de Catalunya."""
    st.session_state.poble_sel = "--- Selecciona una localitat ---"
    st.session_state.selected_area = "--- Selecciona una zona al mapa ---"
    if 'active_tab_cat_index' in st.session_state:
        st.session_state.active_tab_cat_index = 0

def generar_icona_direccio(color, direccio_graus):
    """
    Crea una icona visual (cercle + fletxa) per a la llegenda del mapa comarcal.
    Retorna una cadena d'imatge en format Base64.
    """
    fig, ax = plt.subplots(figsize=(1, 1), dpi=72)
    fig.patch.set_alpha(0)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Dibuixa el cercle
    cercle = Circle((0.5, 0.5), 0.4, facecolor='none', edgecolor=color, linewidth=4,
                    path_effects=[path_effects.withStroke(linewidth=6, foreground='black')])
    ax.add_patch(cercle)

    # Dibuixa la fletxa de direcció
    angle_rad = np.deg2rad(90 - direccio_graus)
    ax.arrow(0.5, 0.5, 0.3 * np.cos(angle_rad), 0.3 * np.sin(angle_rad),
             head_width=0.15, head_length=0.1, fc=color, ec=color,
             length_includes_head=True, zorder=10,
             path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])

    # Converteix la figura a imatge Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()









def generar_bulleti_inteligent(params_calc, poble_sel, valor_conv, cape_mapa):
    """
    Algoritme intel·ligent v8.0 (Validació Creuada): Utilitza el CAPE del focus
    de la tempesta (mapa) com a valor principal i el SBCAPE del sondeig local
    per a validació, oferint un diagnòstic de màxima fiabilitat.
    """
    # --- 1. Extracció de paràmetres ---
    # Dades del SONDEIG LOCAL (per a validació i paràmetres secundaris)
    sbcape_sondeig = params_calc.get('SBCAPE', 0) or 0
    mucin = params_calc.get('MUCIN', 0) or 0
    bwd_6km = params_calc.get('BWD_0-6km', 0) or 0
    srh_1km = params_calc.get('SRH_0-1km', 0) or 0
    lcl_hgt = params_calc.get('LCL_Hgt', 9999) or 9999
    lfc_hgt = params_calc.get('LFC_Hgt', 9999) or 9999
    dcape = params_calc.get('DCAPE', 0) or 0
    
    # Dades del FOCUS DE LA TEMPESTA (mapa)
    conv_mapa = valor_conv
    cape_final = cape_mapa  # <-- EL CAPE DEL FOCUS ÉS EL VALOR PRINCIPAL

    # --- 2. Anàlisi del Balanç Energètic i Condicions de Veto ---
    cost_energetic = abs(mucin) + (lfc_hgt / 150)
    força_disparador = conv_mapa * 1.5
    index_iniciacio = força_disparador - cost_energetic
    
    if cape_final < 300:
        return {"nivell_risc": {"text": "Nul", "color": "#6c757d"}, "titol": "Situació Estable", "resum": f"L'atmosfera al focus de la tempesta no té prou energia (CAPE Focus: {cape_final:.0f} J/kg) per a la formació de tempestes.", "fenomens_previstos": []}
    
    if index_iniciacio < 0:
        return {"nivell_risc": {"text": "Nul", "color": "#6c757d"}, "titol": "Potencial Latent (Falta Disparador)", "resum": f"Tot i que hi ha energia ({cape_final:.0f} J/kg), el disparador (Conv. {conv_mapa:.0f}) no té prou força per vèncer la inhibició atmosfèrica (CIN de {mucin:.0f} i LFC a {lfc_hgt:.0f}m). Les tempestes no s'iniciaran.", "fenomens_previstos": []}

    # --- 3. Classificació Jeràrquica del Potencial de Tempesta (Basat en el CAPE del FOCUS) ---
    fenomens = []
    
    if cape_final >= 1500 and bwd_6km >= 35 and srh_1km >= 150:
        nivell_risc = {"text": "Extrem", "color": "#9370DB"}; titol = "Potencial de Supercèl·lules"
        resum = f"La combinació d'energia explosiva al focus ({cape_final:.0f} J/kg) i una forta cizalladura ({bwd_6km:.0f} nusos) és molt favorable per a la formació de supercèl·lules."
        fenomens.extend(["Calamarsa gran (> 2cm)", "Fortes ratxes de vent (> 90 km/h)"])
        if lcl_hgt < 1200: fenomens.append("Possibilitat de tornados")
    
    elif cape_final >= 800 and bwd_6km >= 25:
        nivell_risc = {"text": "Alt", "color": "#DC3545"}; titol = "Tempestes Organitzades"
        resum = f"L'energia al focus ({cape_final:.0f} J/kg) i una cizalladura considerable ({bwd_6km:.0f} nusos) permetran que les tempestes s'organitzin en sistemes multicel·lulars."
        fenomens.append("Calamarsa o pedra")
        if dcape > 1000: fenomens.append("Esclafits o ratxes de vent molt fortes")
        else: fenomens.append("Fortes ratxes de vent")

    elif cape_final >= 1000 and bwd_6km < 20:
        nivell_risc = {"text": "Moderat", "color": "#FD7E14"}; titol = "Tempestes d'Impuls Aïllades"
        resum = f"Hi ha molta energia al focus ({cape_final:.0f} J/kg) però poca organització. Es poden formar tempestes puntuals però molt intenses."
        fenomens.extend(["Xàfecs localment torrencials", "Possible calamarsa petita", "Ratxes de vent fortes sota la tempesta"])

    else: # Per a CAPE > 300 però que no compleix les condicions superiors
        nivell_risc = {"text": "Baix", "color": "#28A745"}; titol = "Xàfecs i Tronades"
        resum = f"Les condicions són suficients per al desenvolupament de xàfecs i algunes tempestes, generalment de caràcter dispers."
        fenomens.extend(["Ruixats localment moderats", "Activitat elèctrica aïllada"])
    
    if cape_final > 800 and "Activitat elèctrica" not in "".join(fenomens):
        fenomens.insert(0, "Activitat elèctrica freqüent")

    # --- 4. Validació Creuada al Resum Final ---
    discrepancia = abs(sbcape_sondeig - cape_mapa)
    if discrepancia > 500:
        resum += f" Atenció: Hi ha una notable diferència entre l'energia del sondeig local (SBCAPE: {sbcape_sondeig:.0f}) i la del focus real de la tempesta (CAPE: {cape_mapa:.0f})."
    else:
        resum += " El sondeig local és representatiu de l'entorn de la tempesta."
        
    return {"nivell_risc": nivell_risc, "titol": titol, "resum": resum, "fenomens_previstos": fenomens}


def viatjar_a_comarca(nom_comarca):
    """
    Callback per canviar l'anàlisi a una nova comarca directament.
    """
    st.session_state.selected_area = nom_comarca
    pobles_en_comarca = CIUTATS_PER_COMARCA.get(nom_comarca, {})
    if pobles_en_comarca:
        primer_poble = list(pobles_en_comarca.keys())[0]
        st.session_state.poble_sel = primer_poble
        

def ui_portal_viatges_rapids(alertes_totals, comarca_actual):
    """Mostra un panell amb enllaços ràpids a altres comarques amb alertes actives."""
    LLINDAR_CAPE_INTERES = 500
    LLINDAR_CONV_INTERES = 15
    zones_interessants = {
        zona: data for zona, data in alertes_totals.items()
        if data.get('cape', 0) >= LLINDAR_CAPE_INTERES and \
           data.get('conv', 0) >= LLINDAR_CONV_INTERES
    }
    zones_ordenades = sorted(zones_interessants.items(), key=lambda item: item[1]['cape'], reverse=True)
    
    with st.container(border=True):
        st.markdown("<h5 style='text-align: center;'>🚀 Portal de Viatges Ràpids</h5>", unsafe_allow_html=True)
        if not zones_ordenades or (len(zones_ordenades) == 1 and zones_ordenades[0][0] == comarca_actual):
            st.info("No hi ha altres focus de tempesta significatius actius en aquest moment.")
        else:
            st.caption("Viatja directament a altres comarques amb potencial de tempesta:")
            cols = st.columns(2)
            for i, (zona, data) in enumerate(zones_ordenades[:4]):
                with cols[i % 2]:
                    if zona == comarca_actual:
                        st.button(f"{zona} (Estàs aquí)", disabled=True, use_container_width=True, key=f"portal_btn_{zona}")
                    else:
                        st.button(f"C:{data['cape']:.0f} | V:{data['conv']:.0f} - {zona}", 
                                  on_click=viatjar_a_comarca, args=(zona,),
                                  use_container_width=True, key=f"portal_btn_{zona}")
                                  
                  

def ui_pestanya_analisi_comarcal(comarca, valor_conv, poble_sel, timestamp_str, nivell_sel, map_data, params_calc, hora_sel_str, data_tuple, alertes_totals):
    """
    PESTANYA D'ANÀLISI COMARCAL, versió final amb ombrejat + isolínies ADAPTATIVES de CAPE i convergència.
    """
    st.markdown(f"#### Anàlisi de Convergència i CAPE per a la Comarca: {comarca}")
    st.caption(timestamp_str.replace(poble_sel, comarca))

    map_display_data = None
    bulleti_data = None
    
    with st.spinner("Analitzant focus de convergència i trajectòries..."):
        if params_calc:
            cape_del_focus = alertes_totals.get(comarca, {}).get('cape', 0)
            bulleti_data = generar_bulleti_inteligent(params_calc, poble_sel, valor_conv, cape_del_focus)
        
        gdf_comarques = carregar_dades_geografiques()
        if gdf_comarques is None:
            st.error("No s'ha pogut carregar el mapa de comarques."); return

        property_name = next((prop for prop in ['nom_zona', 'nom_comar', 'nomcomar'] if prop in gdf_comarques.columns), 'nom_comar')
        comarca_shape = gdf_comarques[gdf_comarques[property_name] == comarca]
        poble_coords = CIUTATS_CATALUNYA.get(poble_sel)

        if not comarca_shape.empty and map_data and 'cape_data' in map_data:
            map_display_data = _preparar_dades_mapa_comarcal(map_data, comarca_shape, nivell_sel, data_tuple, comarca, poble_coords)

    col_mapa, col_diagnostic = st.columns([0.6, 0.4], gap="large")

    with col_mapa:
        st.markdown("##### Focus de Convergència i Energia (CAPE)")
        
        if map_display_data:
            fig, ax = crear_mapa_base(map_display_data["map_extent"])
            ax.add_geometries(comarca_shape.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=2.5, linestyle='--', zorder=7)
            
            grid_cape = map_display_data["grid_cape"]
            if np.nanmax(grid_cape) > 20:
                # --- LÒGICA D'ISOLÍNIES ADAPTATIVES ---
                max_cape_in_view = np.nanmax(grid_cape)
                if max_cape_in_view < 500:
                    cape_line_levels = [20, 100, 250, 400]
                elif max_cape_in_view < 1500:
                    cape_line_levels = [250, 500, 750, 1000, 1250]
                elif max_cape_in_view < 3000:
                    cape_line_levels = [500, 1000, 1500, 2000, 2500]
                else:
                    cape_line_levels = [1000, 2000, 3000, 4000, 5000, 6000]
                # --- FI DE LA LÒGICA ADAPTATIVA ---

                # Dibuix de l'ombrejat (sempre amb la gamma completa per a consistència de color)
                full_cape_levels = [20, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
                cape_colors = ['#ADD8E6', '#90EE90', '#32CD32', '#ADFF2F', '#FFFF00', '#FFA500', 
                               '#FF4500', '#FF0000', '#DC143C', '#FF00FF', '#9932CC', '#8A2BE2']
                custom_cape_cmap = ListedColormap(cape_colors)
                norm_cape = BoundaryNorm(full_cape_levels, ncolors=custom_cape_cmap.N, clip=True)
                ax.contourf(map_display_data["grid_lon"], map_display_data["grid_lat"], grid_cape, levels=full_cape_levels, cmap=custom_cape_cmap, norm=norm_cape, alpha=0.35, zorder=2, transform=ccrs.PlateCarree(), extend='max')
                
                # Dibuix de les isolínies amb els nivells adaptatius
                cape_line_contours = ax.contour(map_display_data["grid_lon"], map_display_data["grid_lat"], grid_cape, levels=cape_line_levels, colors='white', linewidths=0.9, alpha=0.8, linestyles='solid', zorder=5, transform=ccrs.PlateCarree())
                cape_labels = ax.clabel(cape_line_contours, inline=True, fontsize=9, fmt='%1.0f')
                for label in cape_labels:
                    label.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

            # Dibuix de la convergència i marcadors (sense canvis)
            if map_display_data["max_conv_point"] is not None:
                smoothed_convergence = map_display_data["smoothed_convergence"]
                fill_levels_conv = [10, 20, 30, 40, 60, 80, 100, 120]
                cmap_conv = plt.get_cmap('plasma'); norm_conv = BoundaryNorm(fill_levels_conv, ncolors=cmap_conv.N, clip=True)
                ax.contourf(map_display_data["grid_lon"], map_display_data["grid_lat"], smoothed_convergence, levels=fill_levels_conv, cmap=cmap_conv, norm=norm_conv, alpha=0.7, zorder=3, transform=ccrs.PlateCarree(), extend='max')
                
                px, py = map_display_data["max_conv_point"].geometry.x, map_display_data["max_conv_point"].geometry.y
                path_effect = [path_effects.withStroke(linewidth=3.5, foreground='black')]
                if bulleti_data and bulleti_data['nivell_risc']['text'] == "Nul":
                    circle = Circle((px, py), radius=0.05, facecolor='none', edgecolor='grey', linewidth=2, transform=ccrs.PlateCarree(), zorder=12, path_effects=path_effect, linestyle='--')
                    ax.add_patch(circle); ax.plot(px, py, 'x', color='grey', markersize=8, markeredgewidth=2, zorder=13, transform=ccrs.PlateCarree(), path_effects=path_effect)
                else:
                    if valor_conv >= 100: indicator_color = '#9370DB';
                    elif valor_conv >= 60: indicator_color = '#DC3545';
                    elif valor_conv >= 40: indicator_color = '#FD7E14';
                    elif valor_conv >= 20: indicator_color = '#28A745';
                    else: indicator_color = '#6495ED';
                    circle = Circle((px, py), radius=0.05, facecolor='none', edgecolor=indicator_color, linewidth=2, transform=ccrs.PlateCarree(), zorder=12, path_effects=path_effect)
                    ax.add_patch(circle)
                    ax.plot(px, py, 'x', color=indicator_color, markersize=8, markeredgewidth=2, zorder=13, transform=ccrs.PlateCarree(), path_effects=path_effect)
                    if map_display_data["storm_dir_to"] is not None:
                        dir_rad = np.deg2rad(90 - map_display_data["storm_dir_to"]); length = 0.25
                        end_x, end_y = px + length * np.cos(dir_rad), py + length * np.sin(dir_rad)
                        ax.plot([px, end_x], [py, end_y], color=indicator_color, linewidth=2, transform=ccrs.PlateCarree(), zorder=12, path_effects=path_effect)
            
            if poble_coords:
                ax.text(poble_coords['lon'], poble_coords['lat'], '( Tú )\n▼', transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', color='black', ha='center', va='bottom', zorder=14, path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])
            
            ax.set_title(f"Focus de Tempesta a {comarca}", weight='bold', fontsize=12)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.info("No hi ha dades suficients per generar el mapa detallat de la comarca.")

    with col_diagnostic:
        if bulleti_data and map_display_data and map_display_data.get("max_conv_point") is not None:
            ui_bulleti_inteligent(bulleti_data)
            
            distance_km = map_display_data.get("distance_km")
            is_threat = map_display_data.get("is_threat", False)
            
            if distance_km is not None:
                if distance_km <= 5:
                    amenaça_titol, amenaça_color, amenaça_emoji, amenaça_text = "A sobre!", "#DC3545", "⚠️", f"El focus principal és a menys de 5 km. La tempesta es formarà pràcticament sobre la teva posició."
                elif is_threat:
                    amenaça_titol, amenaça_color, amenaça_emoji, amenaça_text = "S'apropa!", "#FD7E14", "🎯", f"El focus principal a {distance_km:.0f} km es desplaça en la teva direcció. La tempesta podria arribar en les properes hores."
                else:
                    amenaça_titol, amenaça_color, amenaça_emoji, amenaça_text = "Fora de Risc", "#28A745", "✅", f"El focus a {distance_km:.0f} km no és una amenaça directa i estàs fora de la seva àrea d'influència."
                
                st.markdown(f"""
                <div style="padding: 12px; background-color: #2a2c34; border-radius: 10px; border: 1px solid #444; margin-top:10px;">
                     <span style="font-size: 1.2em; color: #FAFAFA;">{amenaça_emoji} Amenaça Directa: <strong style="color:{amenaça_color}">{amenaça_titol}</strong></span>
                     <p style="font-size:0.95em; color:#a0a0b0; margin-top:10px; text-align: left;">{amenaça_text}</p>
                </div>
                """, unsafe_allow_html=True)

            st.caption(f"Aquesta anàlisi es basa en el sondeig de {poble_sel}.")
            crear_llegenda_direccionalitat()
            ui_portal_viatges_rapids(alertes_totals, comarca)
        else:
            st.info("No s'han detectat focus de convergència significatius a la comarca per a l'hora seleccionada.")




@st.cache_data(ttl=600, show_spinner=False)
def _preparar_dades_mapa_comarcal(map_data, _comarca_shape, nivell_sel, _data_tuple, comarca_name, poble_coords):
    """
    Funció interna i cachejada per al processament pesat de dades geoespacials.
    Retorna un diccionari amb totes les dades llestes per a ser dibuixades.
    Els arguments amb '_' s'ignoren en el càlcul de la memòria cau.
    """
    bounds = _comarca_shape.total_bounds
    margin_lon = (bounds[2] - bounds[0]) * 0.3
    margin_lat = (bounds[3] - bounds[1]) * 0.3
    map_extent = [bounds[0] - margin_lon, bounds[2] + margin_lon, bounds[1] - margin_lat, bounds[3] + margin_lat]

    lons, lats = map_data['lons'], map_data['lats']
    grid_lon, grid_lat = np.meshgrid(np.linspace(map_extent[0], map_extent[1], 150), np.linspace(map_extent[2], map_extent[3], 150))
    
    # Interpolar totes les dades necessàries
    grid_dewpoint = griddata((lons, lats), map_data['dewpoint_data'], (grid_lon, grid_lat), 'linear')
    grid_cape = griddata((lons, lats), map_data['cape_data'], (grid_lon, grid_lat), 'linear')
    u_comp, v_comp = mpcalc.wind_components(np.array(map_data['speed_data']) * units('km/h'), np.array(map_data['dir_data']) * units.degrees)
    grid_u = griddata((lons, lats), u_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    grid_v = griddata((lons, lats), v_comp.to('m/s').m, (grid_lon, grid_lat), 'linear')
    
    # Càlcul de convergència
    with np.errstate(invalid='ignore'):
        dx, dy = mpcalc.lat_lon_grid_deltas(grid_lon, grid_lat)
        convergence = (-(mpcalc.divergence(grid_u * units('m/s'), grid_v * units('m/s'), dx=dx, dy=dy)).to('1/s')).magnitude * 1e5
        convergence[np.isnan(convergence)] = 0
        DEWPOINT_THRESHOLD = 14 if nivell_sel >= 950 else 12
        humid_mask = grid_dewpoint >= DEWPOINT_THRESHOLD
        effective_convergence = np.where((convergence >= 10) & humid_mask, convergence, 0)
    
    smoothed_convergence = gaussian_filter(effective_convergence, sigma=5.5)
    smoothed_convergence[smoothed_convergence < 10] = 0

    # Trobar el focus principal i la seva trajectòria
    points_df = pd.DataFrame({'lat': grid_lat.flatten(), 'lon': grid_lon.flatten(), 'conv': smoothed_convergence.flatten()})
    gdf_points = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df.lon, points_df.lat), crs="EPSG:4326")
    points_in_comarca = gpd.sjoin(gdf_points, _comarca_shape.to_crs(gdf_points.crs), how="inner", predicate="within")
    
    max_conv_point = None
    storm_dir_to = None
    distance_km = None
    is_threat = False

    if not points_in_comarca.empty and points_in_comarca['conv'].max() > 10:
        max_conv_point = points_in_comarca.loc[points_in_comarca['conv'].idxmax()]
        if _data_tuple:
            sounding_data, _ = _data_tuple
            p, u, v = sounding_data[0], sounding_data[3], sounding_data[4]
            if p.m.min() < 500 and p.m.max() > 700:
                u_700, v_700 = np.interp(700, p.m[::-1], u.m[::-1]), np.interp(700, p.m[::-1], v.m[::-1])
                u_500, v_500 = np.interp(500, p.m[::-1], u.m[::-1]), np.interp(500, p.m[::-1], v.m[::-1])
                mean_u, mean_v = (u_700 + u_500) / 2.0 * units('m/s'), (v_700 + v_500) / 2.0 * units('m/s')
                storm_dir_to = (mpcalc.wind_direction(mean_u, mean_v).m + 180) % 360
                
                if poble_coords and storm_dir_to is not None:
                    user_lat, user_lon = poble_coords['lat'], poble_coords['lon']
                    px, py = max_conv_point.geometry.x, max_conv_point.geometry.y
                    distance_km = haversine_distance(user_lat, user_lon, py, px)
                    bearing_to_user = get_bearing(py, px, user_lat, user_lon)
                    is_threat = angular_difference(storm_dir_to, bearing_to_user) <= 45

    return {
        "map_extent": map_extent, "grid_lon": grid_lon, "grid_lat": grid_lat,
        "grid_cape": grid_cape, "smoothed_convergence": smoothed_convergence,
        "max_conv_point": max_conv_point, "storm_dir_to": storm_dir_to,
        "distance_km": distance_km, "is_threat": is_threat
    }
        
        


def ui_bulleti_inteligent(bulleti_data):
    """
    Mostra el butlletí generat per l'algoritme.
    """
    st.markdown("##### Butlletí d'Alertes per a la Zona")
    st.markdown(f"""
    <div style="padding: 12px; background-color: #2a2c34; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;">
         <span style="font-size: 1.2em; color: #FAFAFA;">Nivell de Risc: <strong style="color:{bulleti_data['nivell_risc']['color']}">{bulleti_data['nivell_risc']['text']}</strong></span>
         <h6 style="color: white; margin-top: 10px; margin-bottom: 5px;">{bulleti_data['titol']}</h6>
         <p style="font-size:0.95em; color:#a0a0b0; text-align: left;">{bulleti_data['resum']}</p>
    """, unsafe_allow_html=True)
    if bulleti_data['fenomens_previstos']:
        st.markdown("<b style='color: white;'>Fenòmens previstos:</b>", unsafe_allow_html=True)
        for fenomen in bulleti_data['fenomens_previstos']:
            st.markdown(f"- <span style='font-size:0.95em; color:#a0a0b0;'>{fenomen}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)




            
def seleccionar_poble(nom_poble):
    """Callback que s'activa en clicar un poble. Actualitza l'estat directament."""
    st.session_state.poble_sel = nom_poble
    # Reseteja la pestanya per començar sempre per "Anàlisi Comarcal"
    if 'active_tab_cat_index' in st.session_state:
        st.session_state.active_tab_cat_index = 0




def filtrar_alertes(alertes_totals, nivell_seleccionat):
    """
    Filtra el diccionari d'alertes per mostrar només les que superen un llindar.
    """
    LLINDARS = {
        "Tots": 0,
        "Moderat i superior": 20,
        "Alt i superior": 40,
        "Molt Alt i superior": 60,
        "Només Extrems": 100
    }
    llindar_valor = LLINDARS.get(nivell_seleccionat, 0)
    
    if llindar_valor == 0:
        return alertes_totals
    
    return {zona: valor for zona, valor in alertes_totals.items() if valor >= llindar_valor}
    


def ui_llegenda_mapa_principal():
    """Mostra una llegenda millorada que explica la nova lògica del mapa."""
    st.markdown("""
    <style>
        .legend-container-main { background-color: #262730; border-radius: 8px; padding: 18px; margin-top: 15px; border: 1px solid #444; }
        .legend-title-main { font-size: 1.2em; font-weight: bold; color: #FAFAFA; margin-bottom: 8px; }
        .legend-subtitle-main { font-size: 0.95em; color: #a0a0b0; margin-bottom: 18px; }
        .legend-gradient-bar { height: 15px; border-radius: 7px; background: linear-gradient(to right, #28A745, #FFFF00, #FD7E14, #DC3545, #9370DB); margin-bottom: 5px; border: 1px solid #555; }
        .legend-labels { display: flex; justify-content: space-between; font-size: 0.8em; color: #a0a0b0; padding: 0 5px; }
    </style>
    """, unsafe_allow_html=True)
    html_llegenda = (
        '<div class="legend-container-main">'
        '    <div class="legend-title-main">Com Interpretar el Mapa de Situació</div>'
        '    <div class="legend-subtitle-main">El color indica l\'<b>energia (CAPE)</b> trobada al <b>focus de convergència (Disparador)</b> més potent de la zona:</div>'
        '    <div class="legend-gradient-bar"></div>'
        '    <div class="legend-labels">'
        '        <span>500</span>'
        '        <span>1000</span>'
        '        <span>2000</span>'
        '        <span>3000+ J/kg</span>'
        '    </div>'
        '</div>'
    )
    st.markdown(html_llegenda, unsafe_allow_html=True)
    
    
    
# --- DICCIONARI DE CAPITALS (Necessari per a les coordenades) ---
CAPITALS_COMARCA = {
    "Alt Camp": {"nom": "Valls", "lat": 41.2872, "lon": 1.2505},
    "Alt Empordà": {"nom": "Figueres", "lat": 42.2662, "lon": 2.9622},
    "Alt Penedès": {"nom": "Vilafranca del Penedès", "lat": 41.3453, "lon": 1.6995},
    "Alt Urgell": {"nom": "La Seu d'Urgell", "lat": 42.3582, "lon": 1.4593},
    "Anoia": {"nom": "Igualada", "lat": 41.5791, "lon": 1.6174},
    "Bages": {"nom": "Manresa", "lat": 41.7230, "lon": 1.8268},
    "Baix Camp": {"nom": "Reus", "lat": 41.1550, "lon": 1.1075},
    "Baix Ebre": {"nom": "Tortosa", "lat": 40.8126, "lon": 0.5211},
    "Baix Empordà": {"nom": "La Bisbal d'Empordà", "lat": 41.9602, "lon": 3.0378},
    "Baix Llobregat": {"nom": "Sant Feliu de Llobregat", "lat": 41.3833, "lon": 2.0500},
    "Barcelonès": {"nom": "Barcelona", "lat": 41.3851, "lon": 2.1734},
    "Berguedà": {"nom": "Berga", "lat": 42.1051, "lon": 1.8458},
    "Cerdanya": {"nom": "Puigcerdà", "lat": 42.4331, "lon": 1.9287},
    "Conca de Barberà": {"nom": "Montblanc", "lat": 41.3761, "lon": 1.1610},
    "Garraf": {"nom": "Vilanova i la Geltrú", "lat": 41.2241, "lon": 1.7252},
    "Garrigues": {"nom": "Les Borges Blanques", "lat": 41.5224, "lon": 0.8674},
    "Garrotxa": {"nom": "Olot", "lat": 42.1818, "lon": 2.4900},
    "Gironès": {"nom": "Girona", "lat": 41.9831, "lon": 2.8249},
    "Maresme": {"nom": "Mataró", "lat": 41.5388, "lon": 2.4449},
    "Montsià": {"nom": "Amposta", "lat": 40.7093, "lon": 0.5810},
    "Noguera": {"nom": "Balaguer", "lat": 41.7904, "lon": 0.8066},
    "Osona": {"nom": "Vic", "lat": 41.9301, "lon": 2.2545},
    "Pallars Jussà": {"nom": "Tremp", "lat": 42.1664, "lon": 0.8953},
    "Pallars Sobirà": {"nom": "Sort", "lat": 42.4131, "lon": 1.1278},
    "Pla de l'Estany": {"nom": "Banyoles", "lat": 42.1197, "lon": 2.7667},
    "Pla d_Urgell": {"nom": "Mollerussa", "lat": 41.6315, "lon": 0.8931},
    "Priorat": {"nom": "Falset", "lat": 41.1444, "lon": 0.8208},
    "Ribera d_Ebre": {"nom": "Móra d'Ebre", "lat": 41.0945, "lon": 0.6450},
    "Ripollès": {"nom": "Ripoll", "lat": 42.2013, "lon": 2.1903},
    "Segarra": {"nom": "Cervera", "lat": 41.6709, "lon": 1.2721},
    "Segrià": {"nom": "Lleida", "lat": 41.6177, "lon": 0.6200},
    "Selva": {"nom": "Santa Coloma de Farners", "lat": 41.8596, "lon": 2.6703},
    "Solsonès": {"nom": "Solsona", "lat": 41.9942, "lon": 1.5161},
    "Tarragonès": {"nom": "Tarragona", "lat": 41.1189, "lon": 1.2445},
    "Terra Alta": {"nom": "Gandesa", "lat": 41.0526, "lon": 0.4337},
    "Urgell": {"nom": "Tàrrega", "lat": 41.6469, "lon": 1.1415},
    "Val d'Aran": {"nom": "Vielha", "lat": 42.7027, "lon": 0.7966},
    "Vallès Occidental": {"nom": "Sabadell", "lat": 41.5483, "lon": 2.1075},
    "Vallès Oriental": {"nom": "Granollers", "lat": 41.6083, "lon": 2.2886}
}






@st.cache_data(ttl=600, show_spinner="Generant mapa de situació...")
def generar_mapa_folium_catalunya(alertes_per_zona, selected_area_str):
    """
    Funció CACHEADA que fa el treball pesat: carrega les geometries i
    construeix l'objecte del mapa Folium. Retorna l'objecte 'm'.
    """
    gdf = carregar_dades_geografiques()
    if gdf is None: return None

    property_name = next((prop for prop in ['nom_zona', 'nom_comar', 'nomcomar'] if prop in gdf.columns), None)
    if not property_name:
        # No podem usar st.error dins d'una funció cachejada, així que imprimim i retornem None.
        print("Error Crític en el Mapa: L'arxiu GeoJSON no conté una propietat de nom vàlida.")
        return None
    tooltip_alias = 'Comarca:'

    # Paràmetres del mapa
    map_params = {
        "location": [41.83, 1.87], "zoom_start": 8,
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; and the GIS User Community",
        "scrollWheelZoom": True, "dragging": True, "zoom_control": True, "doubleClickZoom": True,
        "max_bounds": [[40.4, 0.0], [42.9, 3.5]], "min_zoom": 8, "max_zoom": 12
    }

    # Lògica per congelar el mapa si hi ha una zona seleccionada
    if selected_area_str and "---" not in selected_area_str:
        cleaned_selected_area = selected_area_str.strip().replace('.', '')
        zona_shape = gdf[gdf[property_name].str.strip().str.replace('.', '') == cleaned_selected_area]
        if not zona_shape.empty:
            centroid = zona_shape.geometry.centroid.iloc[0]
            map_params.update({
                "location": [centroid.y, centroid.x], "zoom_start": 10,
                "scrollWheelZoom": False, "dragging": False, "zoom_control": False, "doubleClickZoom": False,
                "max_bounds": [[zona_shape.total_bounds[1], zona_shape.total_bounds[0]], [zona_shape.total_bounds[3], zona_shape.total_bounds[2]]]
            })

    m = folium.Map(**map_params)

    # Funcions d'estil (exactament les mateixes que tenies)
    def get_color_from_convergence(value):
        if not isinstance(value, (int, float)): return '#6c757d', '#FFFFFF'
        if value >= 100: return '#9370DB', '#FFFFFF'
        if value >= 60: return '#DC3545', '#FFFFFF'
        if value >= 40: return '#FD7E14', '#FFFFFF'
        if value >= 20: return '#28A745', '#FFFFFF'
        return '#6c757d', '#FFFFFF'

    def style_function(feature):
        style = {'fillColor': '#6c757d', 'color': '#495057', 'weight': 1, 'fillOpacity': 0.25}
        nom_feature_raw = feature.get('properties', {}).get(property_name)
        if nom_feature_raw and isinstance(nom_feature_raw, str):
            nom_feature = nom_feature_raw.strip().replace('.', '')
            conv_value = alertes_per_zona.get(nom_feature)
            if conv_value:
                alert_color, _ = get_color_from_convergence(conv_value)
                style.update({'fillColor': alert_color, 'color': alert_color, 'fillOpacity': 0.55, 'weight': 2.5})
            cleaned_selected_area = selected_area_str.strip().replace('.', '') if selected_area_str else ''
            if nom_feature == cleaned_selected_area:
                style.update({'fillColor': '#007bff', 'color': '#ffffff', 'weight': 3, 'fillOpacity': 0.5})
        return style

    highlight_function = lambda x: {'color': '#ffffff', 'weight': 3.5, 'fillOpacity': 0.5}

    folium.GeoJson(
        gdf, style_function=style_function, highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(fields=[property_name], aliases=[tooltip_alias])
    ).add_to(m)

    # Afegir les etiquetes de text
    for zona, conv_value in alertes_per_zona.items():
        capital_info = CAPITALS_COMARCA.get(zona)
        if capital_info:
            bg_color, text_color = get_color_from_convergence(conv_value)
            icon_html = f"""<div style="... font-size: 11px; ...">{zona}: {conv_value:.0f}</div>""" # El teu HTML aquí
            icon = folium.DivIcon(html=icon_html)
            folium.Marker(location=[capital_info['lat'], capital_info['lon']], icon=icon, tooltip=f"Comarca: {zona}").add_to(m)
    
    return m
    




def tornar_a_seleccio_comarca():
    """Callback per tornar a la vista de selecció de municipis de la comarca actual."""
    st.session_state.poble_sel = "--- Selecciona una localitat ---"
    # Reseteja la pestanya activa per evitar inconsistències visuals
    if 'active_tab_cat_index' in st.session_state:
        st.session_state.active_tab_cat_index = 0

def tornar_al_mapa_general():
    """Callback per tornar a la vista principal del mapa de Catalunya."""
    st.session_state.poble_sel = "--- Selecciona una localitat ---"
    st.session_state.selected_area = "--- Selecciona una zona al mapa ---"
    if 'active_tab_cat_index' in st.session_state:
        st.session_state.active_tab_cat_index = 0
        
    
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


def ui_pestanya_mapes_est_peninsula(hourly_index_sel, timestamp_str, nivell_sel, poble_sel):
    """
    Mostra la interfície de la pestanya d'Anàlisi de Mapes per a l'Est Peninsular.
    """
    st.markdown("#### Mapes de Pronòstic (Model AROME)")
    with st.spinner("Carregant mapa AROME per a la península..."):
        map_data, error = carregar_dades_mapa_est_peninsula(nivell_sel, hourly_index_sel)
    
        if error or not map_data:
            st.error(f"Error en carregar el mapa: {error if error else 'No s''han rebut dades.'}")
        else:
            timestamp_map_title = timestamp_str.replace(f"{poble_sel} | ", "")
            fig = crear_mapa_forecast_combinat_est_peninsula(
                map_data['lons'], map_data['lats'], map_data['speed_data'],
                map_data['dir_data'], map_data['dewpoint_data'], nivell_sel,
                timestamp_map_title
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)




def inject_custom_css():
    st.markdown("""
    <style>
    /* ... (El teu altre CSS de spinner i blinking es manté igual) ... */

    /* --- NOU: ESTIL PER A LA VORA DAURADA ANIMADA (VERSIÓ HTML) --- */
    @property --angle {
        syntax: '<angle>';
        initial-value: 0deg;
        inherits: false;
    }

    .animated-gold-wrapper {
        --angle: 0deg;
        border: 3px solid;
        border-image: conic-gradient(from var(--angle), #DAA520, #FFD700, #F0E68C, #FFD700, #DAA520) 1;
        animation: rotate-border 4s linear infinite;
        border-radius: 12px;
        padding: 1.2rem;
        background-color: rgba(38, 39, 48, 0.5);
    }

    @keyframes rotate-border {
        to { --angle: 360deg; }
    }
    /* --- FI DEL NOU ESTIL --- */
    </style>
    """, unsafe_allow_html=True)
    

def ui_zone_selection():
    st.markdown("<h1 style='text-align: center;'>Zona d'Anàlisi</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("🟢(tenen webcams)-🔥(Especialment recomanades) ", icon="🎞")

    def start_transition(zone_id):
        st.session_state['zone_selected'] = zone_id
        st.session_state['show_transition_video'] = True

    paths = {
        'cat': "catalunya_preview.png", 'usa': "usa_preview.png", 'ale': "alemanya_preview.png",
        'ita': "italia_preview.png", 'hol': "holanda_preview.png", 'japo': "japo_preview.png",
        'uk': "uk_preview.png", 'can': "canada_preview.png",
        'nor': "noruega_preview.png",
        'arxiu': "arxiu_preview.png",
        'peninsula': "peninsula_preview.png"
    }
    
    with st.spinner('Carregant entorns geoespacials...'): time.sleep(1)

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)

    # --- FUNCIÓ create_zone_button MODIFICADA (VERSIÓ COMPATIBLE) ---
    def create_zone_button(col, path, title, key, zone_id, type="secondary", height="160px", animated_border=False):
        with col:
            # Si té vora animada, creem un embolcall HTML
            if animated_border:
                st.markdown('<div class="animated-gold-wrapper">', unsafe_allow_html=True)

            # El contenidor de Streamlit va a dins (o sol si no és animat)
            with st.container(border=not animated_border):
                st.markdown(generar_html_imatge_estatica(path, height=height), unsafe_allow_html=True)
                
                display_title = title
                if zone_id == 'italia': display_title += " 🔥"
                elif zone_id in ['japo', 'uk', 'canada', 'valley_halley', 'alemanya', 'holanda', 'catalunya', 'noruega']: display_title += " 🟢"
                
                st.subheader(display_title)
                
                st.button(f"Analitzar {title}", key=key, use_container_width=True, type=type,
                          on_click=start_transition, args=(zone_id,))
            
            # Tanquem l'embolcall HTML si l'hem obert
            if animated_border:
                st.markdown('</div>', unsafe_allow_html=True)
    # --- FI DE LA MODIFICACIÓ ---

    # Dibuixem els botons amb el nou paràmetre
    create_zone_button(row1_col1, paths['cat'], "Catalunya", "btn_cat", "catalunya", "primary", height="200px", animated_border=True)
    create_zone_button(row1_col2, paths['peninsula'], "Est Península", "btn_peninsula", "est_peninsula", "primary", height="200px", animated_border=True)

    create_zone_button(row2_col1, paths['usa'], "Tornado Alley", "btn_usa", "valley_halley")
    create_zone_button(row2_col2, paths['ale'], "Alemanya", "btn_ale", "alemanya")
    create_zone_button(row2_col3, paths['ita'], "Itàlia", "btn_ita", "italia")
    create_zone_button(row2_col4, paths['hol'], "Holanda", "btn_hol", "holanda")
    
    create_zone_button(row3_col1, paths['japo'], "Japó", "btn_japo", "japo")
    create_zone_button(row3_col2, paths['uk'], "Regne Unit", "btn_uk", "uk")
    create_zone_button(row3_col3, paths['can'], "Canadà", "btn_can", "canada")
    create_zone_button(row3_col4, paths['nor'], "Noruega", "btn_nor", "noruega")

    # Secció d'Arxius
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
    # <-- AFEGEIX AQUESTA LÍNIA -->
    elif st.session_state.zone_selected == 'est_peninsula': run_est_peninsula_app()
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



def analitzar_precipitacio_no_severa(params):
    """
    Sistema expert per a diagnosticar el tipus de precipitació estable o de convecció dèbil.
    Utilitza una combinació de PWAT, CAPE_0-3km, LCL i humitat per capes per a donar
    un diagnòstic extremadament detallat, des de "4 gotes" fins a "risc d'inundacions".
    """
    rh_baixa = params.get('RH_CAPES', {}).get('baixa', 0)
    rh_mitjana = params.get('RH_CAPES', {}).get('mitjana', 0)
    pwat = params.get('PWAT', 0)
    cape_0_3km = params.get('CAPE_0-3km', 0)
    lcl_hgt = params.get('LCL_Hgt', 9999)
    max_cape = max(params.get('MLCAPE', 0), params.get('MUCAPE', 0))

    # --- JERARQUIA DE RISC (DE MÉS A MENYS INTENS) ---

    # 1. Condicions d'aiguats extrems i inundacions
    if pwat > 50 and cape_0_3km > 200 and lcl_hgt < 800 and rh_baixa > 90 and rh_mitjana > 85:
        return {'descripcio': "Petacs d'aigua molt forts", 'veredicte': "Potencial extrem per a aiguats torrencials de curta durada. Hi ha risc d'inundacions sobtades."}
    if pwat > 40 and cape_0_3km > 150 and lcl_hgt < 1000 and rh_baixa > 88:
        return {'descripcio': "Petac d'aigua (aiguat local)", 'veredicte': "Condicions molt favorables per a xàfecs d'intensitat torrencial i gran acumulació en poca estona."}

    # 2. Condicions de xàfecs convectius forts
    if pwat > 35 and cape_0_3km > 100 and rh_baixa > 85:
        return {'descripcio': "Xàfecs agressius", 'veredicte': "Es preveuen xàfecs de gran intensitat, possiblement acompanyats de calamarsa petita i ratxes de vent fortes."}
    if pwat > 30 and cape_0_3km > 75 and rh_baixa > 80:
        return {'descripcio': "Xàfecs forts", 'veredicte': "Formació de xàfecs localment forts que poden deixar acumulacions importants."}

    # 3. Condicions de ruixats (convecció dèbil)
    if pwat > 25 and cape_0_3km > 50 and rh_baixa > 75:
        return {'descripcio': "Ruixats dispersos", 'veredicte': "L'atmosfera té prou humitat i inestabilitat per a generar ruixats de distribució irregular."}
    
    # 4. Condicions de precipitació estable (sense inestabilitat a baixos nivells)
    if cape_0_3km < 40:
        if rh_baixa >= 90 and rh_mitjana >= 80 and max_cape < 300:
             return {'descripcio': "Nimbostratus (Pluja Contínua)", 'veredicte': "Saturació profunda i estable, favorable a pluges extenses, persistents i d'intensitat feble a moderada."}
        if rh_baixa > 90 and lcl_hgt < 400:
            return {'descripcio': "Plugims o ruixadet local", 'veredicte': "Capa de núvols baixos molt saturada i enganxada a terra, produint precipitació molt fina però persistent."}
        if rh_baixa > 85 and lcl_hgt < 600:
            return {'descripcio': "Espurnes o 4 gotes", 'veredicte': "La base dels núvols és prou humida per a deixar escapar algunes gotes, però sense acumulació."}

    # Si no es compleix cap condició de precipitació, no retornem res
    return None



def analitzar_potencial_meteorologic(params, nivell_conv, hora_actual=None):
    """
    Sistema de Diagnòstic Expert v79.0 (Classificació Convectiva/Estable Correcta).
    - La lògica de diagnòstic de precipitació no severa s'integra a la funció principal.
    - Els diagnòstics de 'xàfecs' i 'ruixats', que depenen de la inestabilitat (CAPE_0-3km),
      ara s'assignen correctament a la categoria 'inestable' i s'associen a núvols
      de tipus 'Cúmul Congestus'.
    - La categoria 'estable' només conté diagnòstics de precipitació estratiforme (plugims, pluja contínua)
      o altres tipus de núvols sense desenvolupament vertical significatiu.
    """
    
    # --- 1. Extracció de Paràmetres ---
    mlcape = params.get('MLCAPE', 0) or 0; mucape = params.get('MUCAPE', 0) or 0; max_cape = max(mlcape, mucape)
    cin = min(params.get('SBCIN', 0), params.get('MUCIN', 0)) or 0
    lfc_hgt = params.get('LFC_Hgt', 9999) or 9999; lcl_hgt = params.get('LCL_Hgt', 9999) or 9999
    bwd_6km = params.get('BWD_0-6km', 0) or 0
    rh_capes = params.get('RH_CAPES', {}); rh_baixa = rh_capes.get('baixa', 0) if pd.notna(rh_capes.get('baixa')) else 0
    rh_mitjana = rh_capes.get('mitjana', 0) if pd.notna(rh_capes.get('mitjana')) else 0; rh_alta = rh_capes.get('alta', 0) if pd.notna(rh_capes.get('alta')) else 0
    conv_key = f'CONV_{nivell_conv}hPa'; conv = params.get(conv_key, 0) or 0
    pwat = params.get('PWAT', 0)
    cape_0_3km = params.get('CAPE_0-3km', 0)

    resultat = {"inestable": None, "estable": None}
    
    # --- BLOC 1: ANÀLISI DEL POTENCIAL INESTABLE (DE MÉS A MENYS SEVER) ---
    condicions_de_dispar = (cin > -100 and lfc_hgt < 2200 and conv > 10)
    
    # 1.1 Tempestes severes
    if max_cape > 1500 and bwd_6km >= 35 and condicions_de_dispar:
        resultat["inestable"] = {'descripcio': "Potencial de Supercèl·lula", 'veredicte': "Entorn explosiu per a tempestes severes organitzades amb rotació."}
    elif max_cape > 1000 and bwd_6km >= 25 and condicions_de_dispar:
        resultat["inestable"] = {'descripcio': "Tempestes Organitzades", 'veredicte': "Potencial per a la formació de sistemes multicel·lulars o línies de tempesta."}
    elif max_cape > 2500 and bwd_6km < 20 and condicions_de_dispar:
        resultat["inestable"] = {'descripcio': "Tempesta Aïllada (Molt energètica)", 'veredicte': "Molt de 'combustible' però poca organització. Potencial per a calamarsa i esclafits."}
    elif max_cape > 800 and condicions_de_dispar:
        resultat["inestable"] = {'descripcio': "Tempesta Comuna", 'veredicte': "Energia suficient per a tempestes amb forta pluja i activitat el·lèctrica."}
    elif max_cape > 2000 and cin < -125:
         resultat["inestable"] = {'descripcio': "Potencial Explosiu (Pistola Carregada)", 'veredicte': "Molta energia atrapada sota una forta 'tapa'. Si un disparador la trenca, les tempestes seran violentes."}
    
    # 1.2 Precipitació convectiva no severa (Xàfecs i Ruixats)
    elif pwat > 35 and cape_0_3km > 100 and rh_baixa > 85:
        resultat["inestable"] = {'descripcio': "Cúmul Congestus (Xàfecs agressius)", 'veredicte': "Es preveuen xàfecs de gran intensitat, possiblement acompanyats de calamarsa petita i ratxes de vent fortes."}
    elif pwat > 30 and cape_0_3km > 75 and rh_baixa > 80:
        resultat["inestable"] = {'descripcio': "Cúmul Congestus (Xàfecs forts)", 'veredicte': "Formació de xàfecs localment forts que poden deixar acumulacions importants."}
    elif pwat > 25 and cape_0_3km > 50 and rh_baixa > 75:
        resultat["inestable"] = {'descripcio': "Cúmul Congestus (Ruixats dispersos)", 'veredicte': "L'atmosfera té prou humitat i inestabilitat per a generar ruixats de distribució irregular."}
    
    # 1.3 Altres núvols de tipus convectiu
    elif max_cape > 250 and lfc_hgt < 3000 and rh_baixa > 70:
        resultat["inestable"] = {'descripcio': "Cúmuls de creixement", 'veredicte': "Inici de convecció amb creixement vertical. Precursors de possibles tempestes."}
    elif rh_alta >= 60 and (params.get('LI', 5) < 0 or params.get('T_500hPa', 0) < -15):
        resultat["inestable"] = {'descripcio': "Cirrus Castellanus", 'veredicte': "Núvols alts amb petites 'torres', indiquen inestabilitat en nivells superiors."}
    elif 60 <= rh_baixa < 75 and max_cape > 150:
         resultat["inestable"] = {'descripcio': "Cúmuls mediocris", 'veredicte': "Núvols baixos amb cert desenvolupament vertical, però sense arribar a ser tempestes."}
    elif 40 <= rh_baixa < 60 and max_cape > 50:
        resultat["inestable"] = {'descripcio': "Cúmuls de bon temps", 'veredicte': "Humitat suficient per a formar petits cúmuls dispersos sense desenvolupament."}

    # --- BLOC 2: ANÀLISI INDEPENDENT DE LA NUVOLOSITAT ESTABLE ---
    # Aquesta anàlisi s'executa sempre i només diagnostica núvols estratiformes.
    
    if rh_baixa >= 85 and rh_mitjana >= 80 and max_cape < 300:
        resultat["estable"] = {'descripcio': "Nimbostratus (Pluja Contínua)", 'veredicte': "Saturació profunda i estable, favorable a pluges extenses, persistents i d'intensitat feble a moderada."}
    elif rh_baixa > 90 and lcl_hgt < 400 and cape_0_3km < 40:
        resultat["estable"] = {'descripcio': "Estratus (Plugims)", 'veredicte': "Capa de núvols baixos molt saturada i enganxada a terra, produint precipitació molt fina però persistent."}
    elif rh_baixa > 85 and lcl_hgt < 600 and cape_0_3km < 40:
         resultat["estable"] = {'descripcio': "Estratus (4 gotes)", 'veredicte': "La base dels núvols és prou humida per a deixar escapar algunes gotes, però sense acumulació."}
    elif rh_baixa >= 75 and rh_mitjana < 60:
        resultat["estable"] = {'descripcio': "Estratocúmuls (Cel trencat)", 'veredicte': "Bancs de núvols baixos amb zones clares, típic d'inversions de temperatura."}
    elif rh_mitjana >= 70 and rh_baixa < 70:
        if bwd_6km > 40:
             resultat["estable"] = {'descripcio': "Altocúmulus Lenticular", 'veredicte': "Núvols en forma de 'plat volador', indiquen vent fort i turbulència en alçada."}
        else:
            resultat["estable"] = {'descripcio': "Altostratus - Altocúmulus", 'veredicte': "Cel cobert o parcialment cobert per una capa de núvols a nivells mitjans."}
    elif rh_alta >= 60 and rh_mitjana < 50:
        resultat["estable"] = {'descripcio': "Cirrostratus (Cel blanquinós)", 'veredicte': "Presència de núvols alts de tipus cirrus que poden produir un halo solar/lunar."}
    
    # --- BLOC 3: CONDICIÓ PER DEFECTE SI NO S'HA TROBAT NUVOLOSITAT ESTABLE ---
    if resultat["estable"] is None:
        if rh_baixa < 40 and rh_mitjana < 40 and rh_alta < 40:
             resultat["estable"] = {'descripcio': "Cel Serè", 'veredicte': "Atmosfera seca a tots els nivells. Absència de nuvolositat."}
        elif rh_baixa < 60 and rh_mitjana < 50:
             resultat["estable"] = {'descripcio': "Cel amb núvols fragmentats", 'veredicte': "Cel majoritàriament serè amb possibles fractostratus o cúmuls molt aïllats."}
        else:
             resultat["estable"] = {'descripcio': "Sense nuvolositat estable significativa", 'veredicte': "Les condicions no afavoreixen la formació de capes de núvols estables."}
             
    return resultat

    
if __name__ == "__main__":
    main()



