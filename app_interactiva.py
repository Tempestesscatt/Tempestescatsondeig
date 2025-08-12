# -*- coding: utf-8 -*-
import streamlit as st
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
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
from datetime import datetime, timedelta
import pytz

# --- CONFIGURACIÓ INICIAL ---
st.set_page_config(layout="wide", page_title="Tempestes.cat")

cache_session = requests_cache.CachedSession('.cache', expire_after=18000) # 5 hores
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

FORECAST_DAYS = 1

pobles_data = {
    'Abella de la Conca': {'lat': 42.163, 'lon': 1.092},
    'Abrera': {'lat': 41.517, 'lon': 1.901},
    'Àger': {'lat': 42.002, 'lon': 0.763},
    'Agramunt': {'lat': 41.784, 'lon': 1.096},
    'Aguilar de Segarra': {'lat': 41.737, 'lon': 1.626},
    'Agullana': {'lat': 42.395, 'lon': 2.846},
    'Aiguafreda': {'lat': 41.768, 'lon': 2.251},
    'Aiguamúrcia': {'lat': 41.332, 'lon': 1.359},
    'Aiguaviva': {'lat': 41.936, 'lon': 2.766},
    'Aitona': {'lat': 41.498, 'lon': 0.457},
    'Alàs i Cerc': {'lat': 42.358, 'lon': 1.488},
    'Albagés, L\'': {'lat': 41.428, 'lon': 0.732},
    'Albanyà': {'lat': 42.306, 'lon': 2.720},
    'Albatàrrec': {'lat': 41.564, 'lon': 0.601},
    'Albesa': {'lat': 41.751, 'lon': 0.672},
    'Albi, L\'': {'lat': 41.422, 'lon': 0.940},
    'Albinyana': {'lat': 41.242, 'lon': 1.484},
    'Albiol, L\'': {'lat': 41.258, 'lon': 1.109},
    'Albons': {'lat': 42.106, 'lon': 3.079},
    'Alcanar': {'lat': 40.544, 'lon': 0.481},
    'Alcanó': {'lat': 41.488, 'lon': 0.598},
    'Alcarràs': {'lat': 41.562, 'lon': 0.525},
    'Alcoletge': {'lat': 41.644, 'lon': 0.697},
    'Alcover': {'lat': 41.263, 'lon': 1.171},
    'Aldea, L\'': {'lat': 40.748, 'lon': 0.602},
    'Aldover': {'lat': 40.893, 'lon': 0.505},
    'Aleixar, L\'': {'lat': 41.211, 'lon': 1.060},
    'Alella': {'lat': 41.494, 'lon': 2.295},
    'Alfara de Carles': {'lat': 40.889, 'lon': 0.400},
    'Alfarràs': {'lat': 41.829, 'lon': 0.573},
    'Alfés': {'lat': 41.531, 'lon': 0.613},
    'Alforja': {'lat': 41.211, 'lon': 0.972},
    'Alguaire': {'lat': 41.720, 'lon': 0.577},
    'Alió': {'lat': 41.311, 'lon': 1.298},
    'Almacelles': {'lat': 41.728, 'lon': 0.425},
    'Almatret': {'lat': 41.309, 'lon': 0.422},
    'Almenar': {'lat': 41.794, 'lon': 0.567},
    'Almoster': {'lat': 41.196, 'lon': 1.131},
    'Alòs de Balaguer': {'lat': 41.905, 'lon': 0.957},
    'Alp': {'lat': 42.373, 'lon': 1.887},
    'Alpens': {'lat': 42.122, 'lon': 2.105},
    'Alpicat': {'lat': 41.659, 'lon': 0.559},
    'Alt Àneu': {'lat': 42.639, 'lon': 1.121},
    'Amer': {'lat': 42.008, 'lon': 2.602},
    'Ametlla de Mar, L\'': {'lat': 40.883, 'lon': 0.802},
    'Ametlla del Vallès, L\'': {'lat': 41.670, 'lon': 2.261},
    'Ampolla, L\'': {'lat': 40.812, 'lon': 0.709},
    'Amposta': {'lat': 40.707, 'lon': 0.579},
    'Anglès': {'lat': 41.956, 'lon': 2.639},
    'Anglesola': {'lat': 41.685, 'lon': 1.082},
    'Arbeca': {'lat': 41.541, 'lon': 0.923},
    'Arbolí': {'lat': 41.229, 'lon': 0.961},
    'Arbúcies': {'lat': 41.815, 'lon': 2.515},
    'Arenys de Mar': {'lat': 41.581, 'lon': 2.551},
    'Arenys de Munt': {'lat': 41.611, 'lon': 2.540},
    'Argençola': {'lat': 41.621, 'lon': 1.442},
    'Argentona': {'lat': 41.554, 'lon': 2.401},
    'Armentera, L\'': {'lat': 42.152, 'lon': 3.076},
    'Arnes': {'lat': 40.912, 'lon': 0.315},
    'Arres': {'lat': 42.753, 'lon': 0.704},
    'Arsèguel': {'lat': 42.360, 'lon': 1.544},
    'Artesa de Lleida': {'lat': 41.551, 'lon': 0.708},
    'Artesa de Segre': {'lat': 41.895, 'lon': 1.045},
    'Artés': {'lat': 41.798, 'lon': 1.956},
    'Ascó': {'lat': 41.181, 'lon': 0.569},
    'Avellanes i Santa Linya, Les': {'lat': 41.928, 'lon': 0.771},
    'Avinyó': {'lat': 41.860, 'lon': 1.968},
    'Avinyonet de Puigventós': {'lat': 42.238, 'lon': 2.915},
    'Avinyonet del Penedès': {'lat': 41.365, 'lon': 1.772},
    'Avià': {'lat': 42.083, 'lon': 1.834},
    'Badalona': {'lat': 41.450, 'lon': 2.247},
    'Badia del Vallès': {'lat': 41.507, 'lon': 2.150},
    'Bagà': {'lat': 42.253, 'lon': 1.861},
    'Baix Pallars': {'lat': 42.327, 'lon': 1.066},
    'Balaguer': {'lat': 41.790, 'lon': 0.810},
    'Balenyà': {'lat': 41.802, 'lon': 2.230},
    'Balsareny': {'lat': 41.863, 'lon': 1.876},
    'Banyoles': {'lat': 42.119, 'lon': 2.766},
    'Barberà de la Conca': {'lat': 41.412, 'lon': 1.178},
    'Barberà del Vallès': {'lat': 41.518, 'lon': 2.124},
    'Barcelona': {'lat': 41.387, 'lon': 2.168},
    'Bàscara': {'lat': 42.161, 'lon': 2.910},
    'Batea': {'lat': 41.096, 'lon': 0.310},
    'Bausen': {'lat': 42.846, 'lon': 0.706},
    'Begues': {'lat': 41.332, 'lon': 1.921},
    'Begur': {'lat': 41.954, 'lon': 3.204},
    'Belianes': {'lat': 41.583, 'lon': 1.014},
    'Bell-lloc d\'Urgell': {'lat': 41.631, 'lon': 0.775},
    'Bellaguarda': {'lat': 41.346, 'lon': 0.738},
    'Bellcaire d\'Empordà': {'lat': 42.073, 'lon': 3.104},
    'Bellcaire d\'Urgell': {'lat': 41.769, 'lon': 0.916},
    'Bellmunt d\'Urgell': {'lat': 41.737, 'lon': 0.970},
    'Bellmunt del Priorat': {'lat': 41.155, 'lon': 0.769},
    'Bellprat': {'lat': 41.503, 'lon': 1.458},
    'Bellpuig': {'lat': 41.624, 'lon': 1.011},
    'Bellver de Cerdanya': {'lat': 42.370, 'lon': 1.776},
    'Bellvei': {'lat': 41.233, 'lon': 1.579},
    'Bellvís': {'lat': 41.688, 'lon': 0.871},
    'Benavent de Segrià': {'lat': 41.721, 'lon': 0.638},
    'Benifallet': {'lat': 40.975, 'lon': 0.520},
    'Benissanet': {'lat': 41.066, 'lon': 0.637},
    'Berga': {'lat': 42.103, 'lon': 1.845},
    'Besalú': {'lat': 42.199, 'lon': 2.698},
    'Bescanó': {'lat': 41.969, 'lon': 2.748},
    'Beuda': {'lat': 42.228, 'lon': 2.684},
    'Bigues i Riells': {'lat': 41.666, 'lon': 2.221},
    'Biosca': {'lat': 41.846, 'lon': 1.349},
    'Biure': {'lat': 42.327, 'lon': 2.871},
    'Blancafort': {'lat': 41.439, 'lon': 1.159},
    'Blanes': {'lat': 41.674, 'lon': 2.793},
    'Boadella i les Escaules': {'lat': 42.355, 'lon': 2.859},
    'Bolvir': {'lat': 42.417, 'lon': 1.884},
    'Bonastre': {'lat': 41.229, 'lon': 1.442},
    'Bòrredà': {'lat': 42.138, 'lon': 1.996},
    'Borges Blanques, Les': {'lat': 41.522, 'lon': 0.869},
    'Borges del Camp, Les': {'lat': 41.181, 'lon': 1.042},
    'Bossòst': {'lat': 42.784, 'lon': 0.691},
    'Bot': {'lat': 41.009, 'lon': 0.386},
    'Botarell': {'lat': 41.160, 'lon': 1.034},
    'Bràfim': {'lat': 41.277, 'lon': 1.346},
    'Breda': {'lat': 41.748, 'lon': 2.557},
    'Bruc, El': {'lat': 41.580, 'lon': 1.779},
    'Brull, El': {'lat': 41.817, 'lon': 2.306},
    'Brunyola': {'lat': 41.895, 'lon': 2.678},
    'Cabanabona': {'lat': 41.848, 'lon': 1.189},
    'Cabanelles': {'lat': 42.229, 'lon': 2.822},
    'Cabanes': {'lat': 42.307, 'lon': 2.977},
    'Cabanyes, Les': {'lat': 41.365, 'lon': 1.685},
    'Cabra del Camp': {'lat': 41.389, 'lon': 1.258},
    'Cabrera d\'Anoia': {'lat': 41.458, 'lon': 1.705},
    'Cabrera de Mar': {'lat': 41.527, 'lon': 2.396},
    'Cabrils': {'lat': 41.529, 'lon': 2.370},
    'Cadaqués': {'lat': 42.288, 'lon': 3.277},
    'Calaf': {'lat': 41.731, 'lon': 1.512},
    'Calafell': {'lat': 41.199, 'lon': 1.567},
    'Calders': {'lat': 41.794, 'lon': 2.016},
    'Caldes de Malavella': {'lat': 41.838, 'lon': 2.812},
    'Caldes de Montbui': {'lat': 41.633, 'lon': 2.166},
    'Caldes d\'Estrac': {'lat': 41.572, 'lon': 2.528},
    'Calella': {'lat': 41.614, 'lon': 2.664},
    'Calldetenes': {'lat': 41.921, 'lon': 2.278},
    'Callús': {'lat': 41.785, 'lon': 1.785},
    'Calonge': {'lat': 41.859, 'lon': 3.078},
    'Camarasa': {'lat': 41.876, 'lon': 0.878},
    'Cambrils': {'lat': 41.066, 'lon': 1.056},
    'Camós': {'lat': 42.102, 'lon': 2.778},
    'Campdevànol': {'lat': 42.224, 'lon': 2.167},
    'Campelles': {'lat': 42.289, 'lon': 2.138},
    'Campins': {'lat': 41.718, 'lon': 2.463},
    'Campllong': {'lat': 41.912, 'lon': 2.821},
    'Camprodon': {'lat': 42.312, 'lon': 2.364},
    'Canejan': {'lat': 42.827, 'lon': 0.738},
    'Canet d\'Adri': {'lat': 42.028, 'lon': 2.760},
    'Canet de Mar': {'lat': 41.590, 'lon': 2.580},
    'Canovelles': {'lat': 41.620, 'lon': 2.296},
    'Cànoves i Samalús': {'lat': 41.696, 'lon': 2.351},
    'Cantallops': {'lat': 42.439, 'lon': 2.923},
    'Canyelles': {'lat': 41.295, 'lon': 1.722},
    'Capafonts': {'lat': 41.295, 'lon': 1.026},
    'Capçanes': {'lat': 41.100, 'lon': 0.781},
    'Capellades': {'lat': 41.531, 'lon': 1.687},
    'Capmany': {'lat': 42.373, 'lon': 2.921},
    'Cardedeu': {'lat': 41.640, 'lon': 2.358},
    'Cardona': {'lat': 41.914, 'lon': 1.679},
    'Carme': {'lat': 41.513, 'lon': 1.621},
    'Caseres': {'lat': 41.040, 'lon': 0.252},
    'Casserres': {'lat': 42.012, 'lon': 1.844},
    'Castell de l\'Areny': {'lat': 42.186, 'lon': 2.051},
    'Castell de Mur': {'lat': 42.122, 'lon': 0.857},
    'Castellar de la Ribera': {'lat': 42.023, 'lon': 1.439},
    'Castellar de n\'Hug': {'lat': 42.285, 'lon': 2.017},
    'Castellar del Riu': {'lat': 42.112, 'lon': 1.758},
    'Castellar del Vallès': {'lat': 41.616, 'lon': 2.087},
    'Castellbell i el Vilar': {'lat': 41.637, 'lon': 1.849},
    'Castellbisbal': {'lat': 41.481, 'lon': 1.993},
    'Castellcir': {'lat': 41.764, 'lon': 2.155},
    'Castelldans': {'lat': 41.506, 'lon': 0.771},
    'Castell-Platja d\'Aro': {'lat': 41.818, 'lon': 3.067},
    'Castelldefels': {'lat': 41.279, 'lon': 1.975},
    'Castellet i la Gornal': {'lat': 41.260, 'lon': 1.636},
    'Castellfollit de la Roca': {'lat': 42.220, 'lon': 2.551},
    'Castellfollit de Riubregós': {'lat': 41.777, 'lon': 1.411},
    'Castellfollit del Boix': {'lat': 41.673, 'lon': 1.680},
    'Castellgalí': {'lat': 41.681, 'lon': 1.842},
    'Castellnou de Bages': {'lat': 41.801, 'lon': 1.826},
    'Castelló d\'Empúries': {'lat': 42.257, 'lon': 3.074},
    'Castelló de Farfanya': {'lat': 41.828, 'lon': 0.725},
    'Castellolí': {'lat': 41.609, 'lon': 1.677},
    'Castellserà': {'lat': 41.716, 'lon': 0.985},
    'Castellterçol': {'lat': 41.753, 'lon': 2.120},
    'Castellví de la Marca': {'lat': 41.321, 'lon': 1.635},
    'Castellví de Rosanes': {'lat': 41.449, 'lon': 1.932},
    'Catllar, El': {'lat': 41.171, 'lon': 1.321},
    'Cava': {'lat': 42.317, 'lon': 1.626},
    'Celrà': {'lat': 42.021, 'lon': 2.877},
    'Cercs': {'lat': 42.148, 'lon': 1.860},
    'Cerdanyola del Vallès': {'lat': 41.491, 'lon': 2.141},
    'Cervelló': {'lat': 41.401, 'lon': 1.961},
    'Cervera': {'lat': 41.666, 'lon': 1.272},
    'Cervià de les Garrigues': {'lat': 41.439, 'lon': 0.793},
    'Cervià de Ter': {'lat': 42.072, 'lon': 2.911},
    'Ciutadilla': {'lat': 41.564, 'lon': 1.144},
    'Clariana de Cardener': {'lat': 41.954, 'lon': 1.616},
    'Cogul, El': {'lat': 41.465, 'lon': 0.697},
    'Colera': {'lat': 42.404, 'lon': 3.153},
    'Coll de Nargó': {'lat': 42.172, 'lon': 1.319},
    'Collbató': {'lat': 41.572, 'lon': 1.828},
    'Colldejou': {'lat': 41.111, 'lon': 0.826},
    'Collsuspina': {'lat': 41.831, 'lon': 2.177},
    'Colomers': {'lat': 42.083, 'lon': 2.986},
    'Coma i la Pedra, La': {'lat': 42.176, 'lon': 1.595},
    'Conca de Dalt': {'lat': 42.238, 'lon': 0.985},
    'Conesa': {'lat': 41.516, 'lon': 1.293},
    'Constantí': {'lat': 41.157, 'lon': 1.214},
    'Copons': {'lat': 41.685, 'lon': 1.492},
    'Corbera de Llobregat': {'lat': 41.419, 'lon': 1.929},
    'Corbera d\'Ebre': {'lat': 41.080, 'lon': 0.479},
    'Corbins': {'lat': 41.693, 'lon': 0.669},
    'Cornellà de Llobregat': {'lat': 41.355, 'lon': 2.069},
    'Cornellà del Terri': {'lat': 42.062, 'lon': 2.822},
    'Cornudella de Montsant': {'lat': 41.263, 'lon': 0.906},
    'Creixell': {'lat': 41.168, 'lon': 1.442},
    'Crespià': {'lat': 42.162, 'lon': 2.802},
    'Cruïlles, Monells i Sant Sadurní de l\'Heura': {'lat': 41.957, 'lon': 3.004},
    'Cubelles': {'lat': 41.208, 'lon': 1.674},
    'Cubells': {'lat': 41.830, 'lon': 0.963},
    'Cunit': {'lat': 41.197, 'lon': 1.635},
    'Darnius': {'lat': 42.368, 'lon': 2.829},
    'Das': {'lat': 42.365, 'lon': 1.870},
    'Deltebre': {'lat': 40.719, 'lon': 0.710},
    'Dosrius': {'lat': 41.595, 'lon': 2.405},
    'Duesaigües': {'lat': 41.144, 'lon': 0.923},
    'Escala, L\'': {'lat': 42.122, 'lon': 3.131},
    'Esparreguera': {'lat': 41.536, 'lon': 1.868},
    'Espinelves': {'lat': 41.869, 'lon': 2.417},
    'Espluga Calba, L\'': {'lat': 41.491, 'lon': 0.978},
    'Espluga de Francolí, L\'': {'lat': 41.396, 'lon': 1.102},
    'Esplugues de Llobregat': {'lat': 41.375, 'lon': 2.086},
    'Espolla': {'lat': 42.392, 'lon': 3.000},
    'Esponellà': {'lat': 42.127, 'lon': 2.793},
    'Espot': {'lat': 42.576, 'lon': 1.086},
    'Estamariu': {'lat': 42.378, 'lon': 1.503},
    'Estaràs': {'lat': 41.696, 'lon': 1.341},
    'Esterri d\'Àneu': {'lat': 42.631, 'lon': 1.123},
    'Esterri de Cardós': {'lat': 42.593, 'lon': 1.258},
    'Falset': {'lat': 41.144, 'lon': 0.819},
    'Far d\'Empordà, El': {'lat': 42.235, 'lon': 2.992},
    'Farrera': {'lat': 42.544, 'lon': 1.269},
    'Fatarella, La': {'lat': 41.171, 'lon': 0.490},
    'Febró, La': {'lat': 41.261, 'lon': 0.985},
    'Figaró-Montmany': {'lat': 41.722, 'lon': 2.277},
    'Fígols': {'lat': 42.172, 'lon': 1.841},
    'Fígols i Alinyà': {'lat': 42.215, 'lon': 1.411},
    'Figuera, La': {'lat': 41.229, 'lon': 0.728},
    'Figueres': {'lat': 42.266, 'lon': 2.962},
    'Figuerola del Camp': {'lat': 41.373, 'lon': 1.233},
    'Flaçà': {'lat': 42.049, 'lon': 2.956},
    'Flix': {'lat': 41.229, 'lon': 0.551},
    'Floresta, La': {'lat': 41.512, 'lon': 0.893},
    'Fogars de la Selva': {'lat': 41.737, 'lon': 2.680},
    'Fogars de Montclús': {'lat': 41.742, 'lon': 2.451},
    'Foixà': {'lat': 42.041, 'lon': 3.003},
    'Folgueroles': {'lat': 41.939, 'lon': 2.312},
    'Fondarella': {'lat': 41.621, 'lon': 0.852},
    'Fonollosa': {'lat': 41.758, 'lon': 1.688},
    'Font-rubí': {'lat': 41.413, 'lon': 1.652},
    'Fontals dels Alforins': {'lat': 41.442, 'lon': 0.930},
    'Fontanals de Cerdanya': {'lat': 42.404, 'lon': 1.912},
    'Fontanilles': {'lat': 42.016, 'lon': 3.109},
    'Fontcoberta': {'lat': 42.152, 'lon': 2.779},
    'Foradada': {'lat': 41.868, 'lon': 0.942},
    'Forallac': {'lat': 41.956, 'lon': 3.056},
    'Forès': {'lat': 41.503, 'lon': 1.237},
    'Fornells de la Selva': {'lat': 41.936, 'lon': 2.812},
    'Fortià': {'lat': 42.234, 'lon': 3.033},
    'Freginals': {'lat': 40.671, 'lon': 0.505},
    'Fulleda': {'lat': 41.458, 'lon': 0.970},
    'Gaià': {'lat': 41.879, 'lon': 1.910},
    'Gàlgema': {'lat': 41.776, 'lon': 2.091},
    'Gallifa': {'lat': 41.682, 'lon': 2.115},
    'Gandesa': {'lat': 41.052, 'lon': 0.436},
    'Garcia': {'lat': 41.139, 'lon': 0.655},
    'Garidells, Els': {'lat': 41.241, 'lon': 1.285},
    'Garriga, La': {'lat': 41.683, 'lon': 2.282},
    'Garrigàs': {'lat': 42.178, 'lon': 2.946},
    'Garrigoles': {'lat': 42.118, 'lon': 3.045},
    'Garriguella': {'lat': 42.330, 'lon': 3.052},
    'Gavà': {'lat': 41.305, 'lon': 2.001},
    'Gavet de la Conca': {'lat': 42.100, 'lon': 0.979},
    'Gelida': {'lat': 41.440, 'lon': 1.863},
    'Ger': {'lat': 42.417, 'lon': 1.846},
    'Gimenells i el Pla de la Font': {'lat': 41.666, 'lon': 0.366},
    'Ginestar': {'lat': 41.077, 'lon': 0.635},
    'Girona': {'lat': 41.983, 'lon': 2.824},
    'Gironella': {'lat': 42.036, 'lon': 1.882},
    'Gisclareny': {'lat': 42.247, 'lon': 1.782},
    'Godall': {'lat': 40.638, 'lon': 0.473},
    'Golmés': {'lat': 41.625, 'lon': 0.925},
    'Gombrèn': {'lat': 42.247, 'lon': 2.089},
    'Gósol': {'lat': 42.234, 'lon': 1.660},
    'Granada, La': {'lat': 41.378, 'lon': 1.721},
    'Granadella, La': {'lat': 41.358, 'lon': 0.669},
    'Granera': {'lat': 41.720, 'lon': 2.059},
    'Granja d\'Escarp, La': {'lat': 41.423, 'lon': 0.395},
    'Granollers': {'lat': 41.608, 'lon': 2.289},
    'Granyanella': {'lat': 41.660, 'lon': 1.218},
    'Granyena de les Garrigues': {'lat': 41.408, 'lon': 0.655},
    'Granyena de Segarra': {'lat': 41.628, 'lon': 1.240},
    'Gratallops': {'lat': 41.192, 'lon': 0.771},
    'Gualba': {'lat': 41.739, 'lon': 2.508},
    'Gualta': {'lat': 42.028, 'lon': 3.104},
    'Guardiola de Berguedà': {'lat': 42.228, 'lon': 1.880},
    'Guardiola de Font-rubí': {'lat': 41.405, 'lon': 1.636},
    'Guils de Cerdanya': {'lat': 42.456, 'lon': 1.890},
    'Guimerà': {'lat': 41.565, 'lon': 1.186},
    'Guingueta d\'Àneu, La': {'lat': 42.603, 'lon': 1.127},
    'Guissona': {'lat': 41.783, 'lon': 1.288},
    'Guixers': {'lat': 42.128, 'lon': 1.642},
    'Gurb': {'lat': 41.956, 'lon': 2.246},
    'Horta de Sant Joan': {'lat': 40.954, 'lon': 0.315},
    'Hostalets de Pierola, Els': {'lat': 41.536, 'lon': 1.776},
    'Hostalric': {'lat': 41.748, 'lon': 2.636},
    'Igualada': {'lat': 41.580, 'lon': 1.616},
    'Isona i Conca Dellà': {'lat': 42.118, 'lon': 1.050},
    'Isòvol': {'lat': 42.399, 'lon': 1.834},
    'Ivars de Noguera': {'lat': 41.838, 'lon': 0.605},
    'Ivars d\'Urgell': {'lat': 41.666, 'lon': 0.952},
    'Ivorra': {'lat': 41.771, 'lon': 1.393},
    'Jafre': {'lat': 42.071, 'lon': 3.016},
    'Jonquera, La': {'lat': 42.419, 'lon': 2.875},
    'Jorba': {'lat': 41.611, 'lon': 1.543},
    'Josa i Tuixén': {'lat': 42.261, 'lon': 1.589},
    'Juià': {'lat': 42.012, 'lon': 2.905},
    'Juncosa': {'lat': 41.380, 'lon': 0.771},
    'Juneda': {'lat': 41.552, 'lon': 0.835},
    'Les': {'lat': 42.812, 'lon': 0.714},
    'Linyola': {'lat': 41.714, 'lon': 0.906},
    'Llacuna, La': {'lat': 41.472, 'lon': 1.535},
    'Lladó': {'lat': 42.246, 'lon': 2.813},
    'Lladorre': {'lat': 42.632, 'lon': 1.319},
    'Lladurs': {'lat': 42.062, 'lon': 1.516},
    'Llagosta, La': {'lat': 41.516, 'lon': 2.193},
    'Llagostera': {'lat': 41.828, 'lon': 2.892},
    'Llambilles': {'lat': 41.916, 'lon': 2.836},
    'Llanars': {'lat': 42.327, 'lon': 2.339},
    'Llançà': {'lat': 42.364, 'lon': 3.153},
    'Llardecans': {'lat': 41.378, 'lon': 0.584},
    'Llavorsí': {'lat': 42.493, 'lon': 1.258},
    'Lleida': {'lat': 41.617, 'lon': 0.622},
    'Llers': {'lat': 42.298, 'lon': 2.911},
    'Lles de Cerdanya': {'lat': 42.392, 'lon': 1.687},
    'Lliçà d\'Amunt': {'lat': 41.597, 'lon': 2.241},
    'Lliçà de Vall': {'lat': 41.564, 'lon': 2.246},
    'Llimiana': {'lat': 42.072, 'lon': 0.916},
    'Lloar, El': {'lat': 41.171, 'lon': 0.751},
    'Llobera': {'lat': 41.970, 'lon': 1.487},
    'Llorac': {'lat': 41.529, 'lon': 1.341},
    'Llorenç del Penedès': {'lat': 41.277, 'lon': 1.543},
    'Lloret de Mar': {'lat': 41.700, 'lon': 2.845},
    'Llosses, Les': {'lat': 42.152, 'lon': 2.131},
    'Maçanet de Cabrenys': {'lat': 42.389, 'lon': 2.753},
    'Maçanet de la Selva': {'lat': 41.782, 'lon': 2.730},
    'Madremanya': {'lat': 41.972, 'lon': 2.955},
    'Maià de Montcal': {'lat': 42.203, 'lon': 2.741},
    'Maials': {'lat': 41.376, 'lon': 0.505},
    'Maldà': {'lat': 41.554, 'lon': 1.034},
    'Malgrat de Mar': {'lat': 41.645, 'lon': 2.741},
    'Malla': {'lat': 41.870, 'lon': 2.243},
    'Manlleu': {'lat': 42.000, 'lon': 2.283},
    'Manresa': {'lat': 41.727, 'lon': 1.825},
    'Marçà': {'lat': 41.127, 'lon': 0.799},
    'Margalef': {'lat': 41.288, 'lon': 0.754},
    'Marganell': {'lat': 41.637, 'lon': 1.798},
    'Martorell': {'lat': 41.474, 'lon': 1.927},
    'Martorelles': {'lat': 41.528, 'lon': 2.235},
    'Mas de Barberans': {'lat': 40.759, 'lon': 0.360},
    'Masarac': {'lat': 42.345, 'lon': 2.960},
    'Masdenverge': {'lat': 40.697, 'lon': 0.551},
    'Masies de Roda, Les': {'lat': 42.008, 'lon': 2.308},
    'Masies de Voltregà, Les': {'lat': 42.022, 'lon': 2.251},
    'Masllorenç': {'lat': 41.280, 'lon': 1.411},
    'Masnou, El': {'lat': 41.481, 'lon': 2.318},
    'Masó, La': {'lat': 41.246, 'lon': 1.228},
    'Maspujols': {'lat': 41.191, 'lon': 1.096},
    'Masquefa': {'lat': 41.503, 'lon': 1.821},
    'Massalcoreig': {'lat': 41.458, 'lon': 0.362},
    'Massanes': {'lat': 41.776, 'lon': 2.653},
    'Massoteres': {'lat': 41.796, 'lon': 1.309},
    'Matadepera': {'lat': 41.604, 'lon': 2.023},
    'Mataró': {'lat': 41.538, 'lon': 2.445},
    'Mediona': {'lat': 41.472, 'lon': 1.611},
    'Menàrguens': {'lat': 41.730, 'lon': 0.751},
    'Meranges': {'lat': 42.457, 'lon': 1.787},
    'Mieres': {'lat': 42.128, 'lon': 2.637},
    'Milà, El': {'lat': 41.284, 'lon': 1.226},
    'Miravet': {'lat': 41.031, 'lon': 0.596},
    'Moià': {'lat': 41.810, 'lon': 2.096},
    'Moja': {'lat': 41.353, 'lon': 1.713},
    'Molins de Rei': {'lat': 41.414, 'lon': 2.016},
    'Mollerussa': {'lat': 41.631, 'lon': 0.895},
    'Mollet de Peralada': {'lat': 42.348, 'lon': 2.997},
    'Mollet del Vallès': {'lat': 41.539, 'lon': 2.213},
    'Molló': {'lat': 42.349, 'lon': 2.404},
    'Molsosa, La': {'lat': 41.780, 'lon': 1.545},
    'Monistrol de Calders': {'lat': 41.745, 'lon': 2.013},
    'Monistrol de Montserrat': {'lat': 41.610, 'lon': 1.844},
    'Mont-ral': {'lat': 41.288, 'lon': 1.098},
    'Mont-roig del Camp': {'lat': 41.087, 'lon': 0.957},
    'Montagut i Oix': {'lat': 42.233, 'lon': 2.593},
    'Montblanc': {'lat': 41.375, 'lon': 1.161},
    'Montbrió del Camp': {'lat': 41.121, 'lon': 1.011},
    'Montcada i Reixac': {'lat': 41.485, 'lon': 2.187},
    'Montclar': {'lat': 42.028, 'lon': 1.761},
    'Montellà i Martinet': {'lat': 42.361, 'lon': 1.691},
    'Montesquiu': {'lat': 42.111, 'lon': 2.211},
    'Montferrer i Castellbò': {'lat': 42.352, 'lon': 1.428},
    'Montferri': {'lat': 41.253, 'lon': 1.365},
    'Montgai': {'lat': 41.776, 'lon': 0.963},
    'Montgat': {'lat': 41.464, 'lon': 2.279},
    'Montmajor': {'lat': 42.022, 'lon': 1.700},
    'Montmaneu': {'lat': 41.638, 'lon': 1.420},
    'Montmell, El': {'lat': 41.325, 'lon': 1.466},
    'Montmeló': {'lat': 41.552, 'lon': 2.249},
    'Móra d\'Ebre': {'lat': 41.092, 'lon': 0.643},
    'Móra la Nova': {'lat': 41.106, 'lon': 0.655},
    'Morell, El': {'lat': 41.189, 'lon': 1.206},
    'Morera de Montsant, La': {'lat': 41.261, 'lon': 0.843},
    'Muntanyola': {'lat': 41.867, 'lon': 2.160},
    'Mura': {'lat': 41.698, 'lon': 1.977},
    'Navarcles': {'lat': 41.751, 'lon': 1.905},
    'Navàs': {'lat': 41.900, 'lon': 1.879},
    'Navata': {'lat': 42.228, 'lon': 2.859},
    'Naut Aran': {'lat': 42.695, 'lon': 0.887},
    'Nou de Berguedà, La': {'lat': 42.187, 'lon': 1.928},
    'Nou de Gaià, La': {'lat': 41.171, 'lon': 1.371},
    'Nulles': {'lat': 41.249, 'lon': 1.298},
    'Odèn': {'lat': 42.137, 'lon': 1.444},
    'Òdena': {'lat': 41.603, 'lon': 1.636},
    'Ogassa': {'lat': 42.273, 'lon': 2.261},
    'Olesa de Bonesvalls': {'lat': 41.352, 'lon': 1.831},
    'Olesa de Montserrat': {'lat': 41.545, 'lon': 1.894},
    'Oliana': {'lat': 42.066, 'lon': 1.314},
    'Oliola': {'lat': 41.838, 'lon': 1.157},
    'Olius': {'lat': 41.986, 'lon': 1.547},
    'Olivella': {'lat': 41.315, 'lon': 1.821},
    'Olost': {'lat': 41.986, 'lon': 2.095},
    'Olot': {'lat': 42.181, 'lon': 2.490},
    'Olvan': {'lat': 42.061, 'lon': 1.890},
    'Omellons, Els': {'lat': 41.517, 'lon': 0.902},
    'Omells de na Gaia, Els': {'lat': 41.528, 'lon': 1.077},
    'Ordis': {'lat': 42.203, 'lon': 2.909},
    'Orís': {'lat': 42.057, 'lon': 2.241},
    'Oristà': {'lat': 41.928, 'lon': 2.059},
    'Orpí': {'lat': 41.526, 'lon': 1.597},
    'Òrrius': {'lat': 41.549, 'lon': 2.355},
    'Orís': {'lat': 42.057, 'lon': 2.241},
    'Oristà': {'lat': 41.928, 'lon': 2.059},
    'Orpí': {'lat': 41.526, 'lon': 1.597},
    'Òrrius': {'lat': 41.549, 'lon': 2.355},
    'Os de Balaguer': {'lat': 41.872, 'lon': 0.722},
    'Osor': {'lat': 41.942, 'lon': 2.556},
    'Ossó de Sió': {'lat': 41.745, 'lon': 1.134},
    'Pacs del Penedès': {'lat': 41.354, 'lon': 1.670},
    'Palafolls': {'lat': 41.670, 'lon': 2.753},
    'Palafrugell': {'lat': 41.918, 'lon': 3.163},
    'Palamós': {'lat': 41.846, 'lon': 3.128},
    'Palau d\'Anglesola, El': {'lat': 41.649, 'lon': 0.884},
    'Palau de Santa Eulàlia': {'lat': 42.164, 'lon': 2.966},
    'Palau-sator': {'lat': 41.990, 'lon': 3.121},
    'Palau-solità i Plegamans': {'lat': 41.583, 'lon': 2.179},
    'Pallaresa, La': {'lat': 41.171, 'lon': 1.240},
    'Palma d\'Ebre, La': {'lat': 41.282, 'lon': 0.655},
    'Papiol, El': {'lat': 41.442, 'lon': 2.011},
    'Pardines': {'lat': 42.312, 'lon': 2.213},
    'Parets del Vallès': {'lat': 41.573, 'lon': 2.233},
    'Parlavà': {'lat': 42.018, 'lon': 3.029},
    'Passanant i Belltall': {'lat': 41.536, 'lon': 1.218},
    'Pau': {'lat': 42.314, 'lon': 3.121},
    'Paüls': {'lat': 40.916, 'lon': 0.395},
    'Pedret i Marzà': {'lat': 42.290, 'lon': 3.078},
    'Penelles': {'lat': 41.758, 'lon': 0.963},
    'Pera, La': {'lat': 42.022, 'lon': 2.973},
    'Perafita': {'lat': 42.046, 'lon': 2.110},
    'Perafort': {'lat': 41.189, 'lon': 1.259},
    'Peralada': {'lat': 42.308, 'lon': 3.010},
    'Peramola': {'lat': 42.062, 'lon': 1.267},
    'Perelló, El': {'lat': 40.873, 'lon': 0.716},
    'Piera': {'lat': 41.520, 'lon': 1.748},
    'Piles, Les': {'lat': 41.493, 'lon': 1.285},
    'Pinell de Brai, El': {'lat': 41.012, 'lon': 0.513},
    'Pinell de Solsonès': {'lat': 41.921, 'lon': 1.459},
    'Pinós': {'lat': 41.841, 'lon': 1.543},
    'Pira': {'lat': 41.422, 'lon': 1.196},
    'Pla de Santa Maria, El': {'lat': 41.353, 'lon': 1.300},
    'Pla del Penedès, El': {'lat': 41.401, 'lon': 1.725},
    'Plans de Sió, Els': {'lat': 41.691, 'lon': 1.194},
    'Pobla de Cérvoles, La': {'lat': 41.366, 'lon': 0.914},
    'Pobla de Claramunt, La': {'lat': 41.551, 'lon': 1.674},
    'Pobla de Farnals, La': {'lat': 41.451, 'lon': 0.316},
    'Pobla de Lillet, La': {'lat': 42.245, 'lon': 1.975},
    'Pobla de Mafumet, La': {'lat': 41.181, 'lon': 1.222},
    'Pobla de Massaluca, La': {'lat': 41.221, 'lon': 0.306},
    'Pobla de Montornès, La': {'lat': 41.192, 'lon': 1.417},
    'Pobla de Segur, La': {'lat': 42.247, 'lon': 0.968},
    'Poboleda': {'lat': 41.232, 'lon': 0.852},
    'Poal, El': {'lat': 41.683, 'lon': 0.835},
    'Pollença': {'lat': 39.877, 'lon': 3.013},
    'Polinyà': {'lat': 41.542, 'lon': 2.158},
    'Pont de Bar, El': {'lat': 42.365, 'lon': 1.611},
    'Pont de Molins': {'lat': 42.327, 'lon': 2.929},
    'Pont de Suert, El': {'lat': 42.408, 'lon': 0.741},
    'Pont de Vilomara i Rocafort, El': {'lat': 41.695, 'lon': 1.874},
    'Pontils': {'lat': 41.456, 'lon': 1.385},
    'Pontons': {'lat': 41.416, 'lon': 1.516},
    'Ponts': {'lat': 41.916, 'lon': 1.190},
    'Porqueres': {'lat': 42.126, 'lon': 2.748},
    'Porrera': {'lat': 41.188, 'lon': 0.853},
    'Port de la Selva, El': {'lat': 42.333, 'lon': 3.203},
    'Portbou': {'lat': 42.427, 'lon': 3.161},
    'Portella, La': {'lat': 41.758, 'lon': 0.655},
    'Pradell de la Teixeta': {'lat': 41.157, 'lon': 0.865},
    'Prades': {'lat': 41.311, 'lon': 0.990},
    'Prat de Comte': {'lat': 40.979, 'lon': 0.407},
    'Prat de Llobregat, El': {'lat': 41.326, 'lon': 2.095},
    'Prats de Lluçanès': {'lat': 42.010, 'lon': 2.031},
    'Prats i Sansor': {'lat': 42.378, 'lon': 1.831},
    'Preixana': {'lat': 41.603, 'lon': 1.050},
    'Preixens': {'lat': 41.802, 'lon': 0.985},
    'Premià de Dalt': {'lat': 41.512, 'lon': 2.348},
    'Premià de Mar': {'lat': 41.491, 'lon': 2.359},
    'Preses, Les': {'lat': 42.158, 'lon': 2.470},
    'Prullans': {'lat': 42.383, 'lon': 1.745},
    'Puig-reig': {'lat': 41.974, 'lon': 1.882},
    'Puigcerdà': {'lat': 42.432, 'lon': 1.928},
    'Puigdàlber': {'lat': 41.383, 'lon': 1.706},
    'Puiggròs': {'lat': 41.536, 'lon': 0.893},
    'Puigpelat': {'lat': 41.267, 'lon': 1.258},
    'Puigverd d\'Agramunt': {'lat': 41.761, 'lon': 1.106},
    'Puigverd de Lleida': {'lat': 41.564, 'lon': 0.741},
    'Pujalt': {'lat': 41.716, 'lon': 1.428},
    'Pallejà': {'lat': 41.411, 'lon': 1.996},
    'Querol': {'lat': 41.422, 'lon': 1.393},
    'Queralbs': {'lat': 42.350, 'lon': 2.164},
    'Quintana, La': {'lat': 41.989, 'lon': 2.183},
    'Rabós': {'lat': 42.373, 'lon': 3.057},
    'Rajadell': {'lat': 41.714, 'lon': 1.704},
    'Rasquera': {'lat': 40.998, 'lon': 0.597},
    'Regencós': {'lat': 41.939, 'lon': 3.166},
    'Rellinars': {'lat': 41.642, 'lon': 1.914},
    'Reus': {'lat': 41.155, 'lon': 1.107},
    'Rialp': {'lat': 42.445, 'lon': 1.136},
    'Ribera d\'Ondara': {'lat': 41.636, 'lon': 1.285},
    'Ribera d\'Urgellet': {'lat': 42.316, 'lon': 1.429},
    'Ribes de Freser': {'lat': 42.306, 'lon': 2.169},
    'Riells i Viabrea': {'lat': 41.796, 'lon': 2.531},
    'Ridaura': {'lat': 42.176, 'lon': 2.417},
    'Riera de Gaià, La': {'lat': 41.177, 'lon': 1.358},
    'Ripoll': {'lat': 42.201, 'lon': 2.190},
    'Ripollet': {'lat': 41.498, 'lon': 2.158},
    'Riumors': {'lat': 42.219, 'lon': 3.042},
    'Riudarenes': {'lat': 41.821, 'lon': 2.716},
    'Riudaura': {'lat': 42.176, 'lon': 2.417},
    'Riudecanyes': {'lat': 41.134, 'lon': 0.938},
    'Riudecols': {'lat': 41.164, 'lon': 1.047},
    'Riudellots de la Selva': {'lat': 41.897, 'lon': 2.805},
    'Riudoms': {'lat': 41.144, 'lon': 1.050},
    'Roca del Vallès, La': {'lat': 41.587, 'lon': 2.327},
    'Rocafort de Queralt': {'lat': 41.464, 'lon': 1.229},
    'Rocafort i Vilumara': {'lat': 41.688, 'lon': 1.865},
    'Rocallaura': {'lat': 41.493, 'lon': 1.102},
    'Roda de Berà': {'lat': 41.185, 'lon': 1.458},
    'Roda de Ter': {'lat': 41.983, 'lon': 2.309},
    'Rodonyà': {'lat': 41.294, 'lon': 1.401},
    'Roquetes': {'lat': 40.819, 'lon': 0.493},
    'Roses': {'lat': 42.262, 'lon': 3.175},
    'Rosselló': {'lat': 41.696, 'lon': 0.596},
    'Rourell, El': {'lat': 41.222, 'lon': 1.258},
    'Rubí': {'lat': 41.493, 'lon': 2.032},
    'Rubió': {'lat': 41.670, 'lon': 1.543},
    'Rupit i Pruit': {'lat': 42.026, 'lon': 2.465},
    'Sabadell': {'lat': 41.547, 'lon': 2.108},
    'Sagàs': {'lat': 42.035, 'lon': 1.961},
    'Salàs de Pallars': {'lat': 42.213, 'lon': 0.923},
    'Sallent': {'lat': 41.823, 'lon': 1.896},
    'Salomó': {'lat': 41.229, 'lon': 1.378},
    'Salou': {'lat': 41.076, 'lon': 1.140},
    'Sanaüja': {'lat': 41.871, 'lon': 1.309},
    'Sant Adrià de Besòs': {'lat': 41.428, 'lon': 2.219},
    'Sant Agustí de Lluçanès': {'lat': 42.083, 'lon': 2.131},
    'Sant Andreu de la Barca': {'lat': 41.447, 'lon': 1.979},
    'Sant Andreu de Llavaneres': {'lat': 41.571, 'lon': 2.482},
    'Sant Andreu Salou': {'lat': 41.874, 'lon': 2.829},
    'Sant Aniol de Finestres': {'lat': 42.084, 'lon': 2.585},
    'Sant Antoni de Vilamajor': {'lat': 41.675, 'lon': 2.404},
    'Sant Bartomeu del Grau': {'lat': 41.984, 'lon': 2.167},
    'Sant Boi de Llobregat': {'lat': 41.346, 'lon': 2.041},
    'Sant Boi de Lluçanès': {'lat': 42.082, 'lon': 2.155},
    'Sant Carles de la Ràpita': {'lat': 40.618, 'lon': 0.593},
    'Sant Cebrià de Vallalta': {'lat': 41.628, 'lon': 2.607},
    'Sant Celoni': {'lat': 41.691, 'lon': 2.491},
    'Sant Climent de Llobregat': {'lat': 41.344, 'lon': 2.008},
    'Sant Climent Sescebes': {'lat': 42.369, 'lon': 2.981},
    'Sant Cugat del Vallès': {'lat': 41.472, 'lon': 2.085},
    'Sant Cugat Sesgarrigues': {'lat': 41.365, 'lon': 1.748},
    'Sant Esteve de la Sarga': {'lat': 42.052, 'lon': 0.741},
    'Sant Esteve de Palautordera': {'lat': 41.706, 'lon': 2.433},
    'Sant Esteve Sesrovires': {'lat': 41.495, 'lon': 1.876},
    'Sant Feliu de Buixalleu': {'lat': 41.791, 'lon': 2.585},
    'Sant Feliu de Codines': {'lat': 41.688, 'lon': 2.164},
    'Sant Feliu de Guíxols': {'lat': 41.780, 'lon': 3.028},
    'Sant Feliu de Llobregat': {'lat': 41.381, 'lon': 2.045},
    'Sant Feliu de Pallerols': {'lat': 42.079, 'lon': 2.508},
    'Sant Feliu Sasserra': {'lat': 41.954, 'lon': 2.020},
    'Sant Ferriol': {'lat': 42.176, 'lon': 2.661},
    'Sant Fost de Campsentelles': {'lat': 41.503, 'lon': 2.222},
    'Sant Fruitós de Bages': {'lat': 41.748, 'lon': 1.861},
    'Sant Gregori': {'lat': 42.001, 'lon': 2.768},
    'Sant Guim de Freixenet': {'lat': 41.654, 'lon': 1.424},
    'Sant Guim de la Plana': {'lat': 41.719, 'lon': 1.309},
    'Sant Hilari Sacalm': {'lat': 41.879, 'lon': 2.508},
    'Sant Hipòlit de Voltregà': {'lat': 42.015, 'lon': 2.239},
    'Sant Iscle de Vallalta': {'lat': 41.624, 'lon': 2.570},
    'Sant Jaume de Frontanyà': {'lat': 42.188, 'lon': 2.029},
    'Sant Jaume de Llierca': {'lat': 42.206, 'lon': 2.610},
    'Sant Jaume dels Domenys': {'lat': 41.298, 'lon': 1.517},
    'Sant Jaume d\'Enveja': {'lat': 40.713, 'lon': 0.725},
    'Sant Joan de les Abadesses': {'lat': 42.233, 'lon': 2.285},
    'Sant Joan de Mollet': {'lat': 42.030, 'lon': 2.923},
    'Sant Joan de Vilatorrada': {'lat': 41.743, 'lon': 1.802},
    'Sant Joan Despí': {'lat': 41.368, 'lon': 2.057},
    'Sant Joan les Fonts': {'lat': 42.214, 'lon': 2.511},
    'Sant Jordi Desvalls': {'lat': 42.062, 'lon': 2.946},
    'Sant Julià de Cerdanyola': {'lat': 42.211, 'lon': 1.884},
    'Sant Julià de Ramis': {'lat': 42.026, 'lon': 2.852},
    'Sant Julià de Vilatorta': {'lat': 41.919, 'lon': 2.321},
    'Sant Julià del Llor i Bonmatí': {'lat': 41.968, 'lon': 2.685},
    'Sant Just Desvern': {'lat': 41.383, 'lon': 2.072},
    'Sant Llorenç de la Muga': {'lat': 42.321, 'lon': 2.788},
    'Sant Llorenç de Morunys': {'lat': 42.137, 'lon': 1.591},
    'Sant Llorenç d\'Hortons': {'lat': 41.479, 'lon': 1.838},
    'Sant Llorenç Savall': {'lat': 41.678, 'lon': 2.059},
    'Sant Martí de Llémena': {'lat': 42.046, 'lon': 2.679},
    'Sant Martí de Riucorb': {'lat': 41.581, 'lon': 1.050},
    'Sant Martí de Tous': {'lat': 41.564, 'lon': 1.523},
    'Sant Martí d\'Albars': {'lat': 42.021, 'lon': 2.075},
    'Sant Martí Sarroca': {'lat': 41.385, 'lon': 1.611},
    'Sant Martí Sesgueioles': {'lat': 41.705, 'lon': 1.492},
    'Sant Mateu de Bages': {'lat': 41.821, 'lon': 1.741},
    'Sant Miquel de Campmajor': {'lat': 42.128, 'lon': 2.688},
    'Sant Miquel de Fluvià': {'lat': 42.170, 'lon': 2.993},
    'Sant Mori': {'lat': 42.155, 'lon': 2.992},
    'Sant Pau de Segúries': {'lat': 42.264, 'lon': 2.364},
    'Sant Pere de Ribes': {'lat': 41.259, 'lon': 1.769},
    'Sant Pere de Riudebitlles': {'lat': 41.442, 'lon': 1.708},
    'Sant Pere de Torelló': {'lat': 42.074, 'lon': 2.299},
    'Sant Pere de Vilamajor': {'lat': 41.685, 'lon': 2.387},
    'Sant Pere Pescador': {'lat': 42.189, 'lon': 3.084},
    'Sant Pere Sallavinera': {'lat': 41.748, 'lon': 1.571},
    'Sant Pol de Mar': {'lat': 41.602, 'lon': 2.624},
    'Sant Quintí de Mediona': {'lat': 41.458, 'lon': 1.662},
    'Sant Quirze de Besora': {'lat': 42.102, 'lon': 2.222},
    'Sant Quirze del Vallès': {'lat': 41.531, 'lon': 2.081},
    'Sant Quirze Safaja': {'lat': 41.729, 'lon': 2.133},
    'Sant Sadurní d\'Anoia': {'lat': 41.428, 'lon': 1.785},
    'Sant Sadurní d\'Osormort': {'lat': 41.899, 'lon': 2.378},
    'Sant Salvador de Guardiola': {'lat': 41.677, 'lon': 1.765},
    'Sant Vicenç de Castellet': {'lat': 41.671, 'lon': 1.865},
    'Sant Vicenç de Montalt': {'lat': 41.583, 'lon': 2.508},
    'Sant Vicenç de Torelló': {'lat': 42.046, 'lon': 2.262},
    'Sant Vicenç dels Horts': {'lat': 41.392, 'lon': 2.008},
    'Santa Bàrbara': {'lat': 40.715, 'lon': 0.495},
    'Santa Cecília de Voltregà': {'lat': 42.006, 'lon': 2.213},
    'Santa Coloma de Cervelló': {'lat': 41.371, 'lon': 2.023},
    'Santa Coloma de Farners': {'lat': 41.859, 'lon': 2.668},
    'Santa Coloma de Gramenet': {'lat': 41.454, 'lon': 2.213},
    'Santa Coloma de Queralt': {'lat': 41.536, 'lon': 1.385},
    'Santa Cristina d\'Aro': {'lat': 41.815, 'lon': 2.998},
    'Santa Eugènia de Berga': {'lat': 41.905, 'lon': 2.278},
    'Santa Eulàlia de Riuprimer': {'lat': 41.884, 'lon': 2.210},
    'Santa Eulàlia de Ronçana': {'lat': 41.644, 'lon': 2.222},
    'Santa Fe del Penedès': {'lat': 41.372, 'lon': 1.733},
    'Santa Llogaia d\'Àlguema': {'lat': 42.235, 'lon': 2.951},
    'Santa Margarida de Montbui': {'lat': 41.571, 'lon': 1.597},
    'Santa Margarida i els Monjos': {'lat': 41.311, 'lon': 1.666},
    'Santa Maria de Besora': {'lat': 42.128, 'lon': 2.257},
    'Santa Maria de Corcó': {'lat': 42.052, 'lon': 2.368},
    'Santa Maria de Martorelles': {'lat': 41.520, 'lon': 2.260},
    'Santa Maria de Merlès': {'lat': 42.001, 'lon': 1.986},
    'Santa Maria de Miralles': {'lat': 41.493, 'lon': 1.494},
    'Santa Maria de Palautordera': {'lat': 41.696, 'lon': 2.443},
    'Santa Maria d\'Oló': {'lat': 41.874, 'lon': 2.034},
    'Santa Oliva': {'lat': 41.246, 'lon': 1.551},
    'Santa Pau': {'lat': 42.145, 'lon': 2.569},
    'Santa Perpètua de Mogoda': {'lat': 41.536, 'lon': 2.182},
    'Santa Susanna': {'lat': 41.636, 'lon': 2.711},
    'Santpedor': {'lat': 41.782, 'lon': 1.865},
    'Sarral': {'lat': 41.446, 'lon': 1.248},
    'Sarrià de Ter': {'lat': 42.009, 'lon': 2.822},
    'Sarroca de Bellera': {'lat': 42.373, 'lon': 0.865},
    'Sarroca de Lleida': {'lat': 41.488, 'lon': 0.609},
    'Saus, Camallera i Llampaies': {'lat': 42.146, 'lon': 2.969},
    'Savallà del Comtat': {'lat': 41.538, 'lon': 1.258},
    'Secuita, La': {'lat': 41.196, 'lon': 1.295},
    'Sedó': {'lat': 41.657, 'lon': 1.233},
    'Segarra': {'lat': 41.7, 'lon': 1.3},
    'Segur de Calafell': {'lat': 41.190, 'lon': 1.596},
    'Selva de Mar, La': {'lat': 42.327, 'lon': 3.187},
    'Selva del Camp, La': {'lat': 41.213, 'lon': 1.138},
    'Senan': {'lat': 41.411, 'lon': 1.077},
    'Senterada': {'lat': 42.326, 'lon': 0.941},
    'Sentiu de Sió, La': {'lat': 41.794, 'lon': 0.863},
    'Sentmenat': {'lat': 41.608, 'lon': 2.136},
    'Seo de Urgel, La': {'lat': 42.358, 'lon': 1.463},
    'Serinyà': {'lat': 42.170, 'lon': 2.744},
    'Seròs': {'lat': 41.472, 'lon': 0.407},
    'Setcases': {'lat': 42.373, 'lon': 2.298},
    'Seva': {'lat': 41.815, 'lon': 2.285},
    'Sidamon': {'lat': 41.631, 'lon': 0.825},
    'Sils': {'lat': 41.808, 'lon': 2.743},
    'Sitges': {'lat': 41.235, 'lon': 1.811},
    'Siurana': {'lat': 42.196, 'lon': 3.012},
    'Sobremunt': {'lat': 42.030, 'lon': 2.164},
    'Soleràs, El': {'lat': 41.423, 'lon': 0.725},
    'Solivella': {'lat': 41.458, 'lon': 1.185},
    'Solsona': {'lat': 41.992, 'lon': 1.516},
    'Sora': {'lat': 42.086, 'lon': 2.179},
    'Soriguera': {'lat': 42.388, 'lon': 1.185},
    'Sort': {'lat': 42.413, 'lon': 1.129},
    'Soses': {'lat': 41.536, 'lon': 0.491},
    'Subirats': {'lat': 41.385, 'lon': 1.791},
    'Sudanell': {'lat': 41.579, 'lon': 0.551},
    'Sumacàrcer': {'lat': 39.066, 'lon': -0.569},
    'Sunyer': {'lat': 41.536, 'lon': 0.584},
    'Súria': {'lat': 41.831, 'lon': 1.750},
    'Tagamanent': {'lat': 41.777, 'lon': 2.291},
    'Talamanca': {'lat': 41.722, 'lon': 1.982},
    'Talarn': {'lat': 42.181, 'lon': 0.903},
    'Tallada d\'Empordà, La': {'lat': 42.073, 'lon': 3.055},
    'Tarragona': {'lat': 41.118, 'lon': 1.245},
    'Taradell': {'lat': 41.874, 'lon': 2.285},
    'Tàrrega': {'lat': 41.646, 'lon': 1.141},
    'Tarrés': {'lat': 41.433, 'lon': 1.019},
    'Tavertet': {'lat': 42.000, 'lon': 2.417},
    'Tavèrnoles': {'lat': 41.951, 'lon': 2.338},
    'Teià': {'lat': 41.503, 'lon': 2.321},
    'Térmens': {'lat': 41.718, 'lon': 0.762},
    'Terrades': {'lat': 42.316, 'lon': 2.846},
    'Terrassa': {'lat': 41.561, 'lon': 2.008},
    'Tiana': {'lat': 41.479, 'lon': 2.268},
    'Tírvia': {'lat': 42.502, 'lon': 1.238},
    'Tivenys': {'lat': 40.871, 'lon': 0.516},
    'Tivissa': {'lat': 41.042, 'lon': 0.732},
    'Tona': {'lat': 41.849, 'lon': 2.228},
    'Tordera': {'lat': 41.702, 'lon': 2.719},
    'Torelló': {'lat': 42.048, 'lon': 2.262},
    'Torms, Els': {'lat': 41.388, 'lon': 0.725},
    'Tornabous': {'lat': 41.696, 'lon': 1.050},
    'Torre de Cabdella, La': {'lat': 42.420, 'lon': 0.982},
    'Torre de Claramunt, La': {'lat': 41.517, 'lon': 1.666},
    'Torre de Fontaubella, La': {'lat': 41.147, 'lon': 0.846},
    'Torrebesses': {'lat': 41.428, 'lon': 0.589},
    'Torredembarra': {'lat': 41.145, 'lon': 1.398},
    'Torrefarrera': {'lat': 41.681, 'lon': 0.609},
    'Torrefeta i Florejacs': {'lat': 41.751, 'lon': 1.258},
    'Torregrossa': {'lat': 41.583, 'lon': 0.852},
    'Torrelameu': {'lat': 41.711, 'lon': 0.708},
    'Torrelavit': {'lat': 41.429, 'lon': 1.748},
    'Torrelles de Foix': {'lat': 41.391, 'lon': 1.573},
    'Torrelles de Llobregat': {'lat': 41.365, 'lon': 1.986},
    'Torres de Segre': {'lat': 41.536, 'lon': 0.518},
    'Torroella de Fluvià': {'lat': 42.179, 'lon': 3.050},
    'Torroella de Montgrí': {'lat': 42.043, 'lon': 3.128},
    'Torroja del Priorat': {'lat': 41.206, 'lon': 0.809},
    'Tortellà': {'lat': 42.238, 'lon': 2.636},
    'Tortosa': {'lat': 40.812, 'lon': 0.521},
    'Toses': {'lat': 42.327, 'lon': 2.015},
    'Tossa de Mar': {'lat': 41.720, 'lon': 2.932},
    'Tremp': {'lat': 42.166, 'lon': 0.894},
    'Ullà': {'lat': 42.051, 'lon': 3.111},
    'Ulldecona': {'lat': 40.598, 'lon': 0.449},
    'Ulldemolins': {'lat': 41.321, 'lon': 0.871},
    'Ullastrell': {'lat': 41.535, 'lon': 1.954},
    'Ullastret': {'lat': 42.000, 'lon': 3.067},
    'Urús': {'lat': 42.348, 'lon': 1.838},
    'Vacarisses': {'lat': 41.597, 'lon': 1.921},
    'Vall d\'Alcalà, La': {'lat': 38.800, 'lon': -0.216},
    'Vall de Bianya, La': {'lat': 42.222, 'lon': 2.417},
    'Vall de Boí, La': {'lat': 42.504, 'lon': 0.802},
    'Vall de Cardós': {'lat': 42.565, 'lon': 1.229},
    'Vall-de-roures': {'lat': 40.828, 'lon': 0.055},
    'Vall-llobrega': {'lat': 41.871, 'lon': 3.136},
    'Vallbona de les Monges': {'lat': 41.525, 'lon': 1.088},
    'Vallcebre': {'lat': 42.206, 'lon': 1.821},
    'Vallclara': {'lat': 41.387, 'lon': 1.034},
    'Vallfogona de Balaguer': {'lat': 41.737, 'lon': 0.816},
    'Vallfogona de Ripollès': {'lat': 42.188, 'lon': 2.302},
    'Vallfogona de Riucorb': {'lat': 41.564, 'lon': 1.233},
    'Vallgorguina': {'lat': 41.644, 'lon': 2.511},
    'Vallirana': {'lat': 41.385, 'lon': 1.933},
    'Vallmoll': {'lat': 41.229, 'lon': 1.250},
    'Valls': {'lat': 41.286, 'lon': 1.250},
    'Vandellòs i l\'Hospitalet de l\'Infant': {'lat': 40.997, 'lon': 0.822},
    'Valls d\'Aguilar, Les': {'lat': 42.317, 'lon': 1.383},
    'Valls de Valira, Les': {'lat': 42.399, 'lon': 1.455},
    'Vansa i Fórnols, La': {'lat': 42.246, 'lon': 1.503},
    'Veià': {'lat': 41.229, 'lon': 1.579},
    'Vespella de Gaià': {'lat': 41.192, 'lon': 1.358},
    'Verdú': {'lat': 41.611, 'lon': 1.041},
    'Verges': {'lat': 42.062, 'lon': 3.047},
    'Vic': {'lat': 41.930, 'lon': 2.255},
    'Vidrà': {'lat': 42.124, 'lon': 2.311},
    'Vidreres': {'lat': 41.787, 'lon': 2.778},
    'Vielha e Mijaran': {'lat': 42.702, 'lon': 0.796},
    'Vila-sacra': {'lat': 42.247, 'lon': 3.011},
    'Vila-sana': {'lat': 41.650, 'lon': 0.932},
    'Vila-seca': {'lat': 41.111, 'lon': 1.144},
    'Vilabertran': {'lat': 42.285, 'lon': 2.981},
    'Vilabella': {'lat': 41.258, 'lon': 1.321},
    'Vilablareix': {'lat': 41.956, 'lon': 2.793},
    'Vilada': {'lat': 42.138, 'lon': 1.931},
    'Viladamat': {'lat': 42.133, 'lon': 3.072},
    'Viladasens': {'lat': 42.106, 'lon': 2.915},
    'Viladecans': {'lat': 41.315, 'lon': 2.019},
    'Viladecavalls': {'lat': 41.558, 'lon': 1.966},
    'Viladrau': {'lat': 41.849, 'lon': 2.387},
    'Vilafant': {'lat': 42.251, 'lon': 2.936},
    'Vilafranca del Penedès': {'lat': 41.345, 'lon': 1.698},
    'Vilajuïga': {'lat': 42.327, 'lon': 3.093},
    'Vilalba dels Arcs': {'lat': 41.121, 'lon': 0.413},
    'Vilalba Sasserra': {'lat': 41.655, 'lon': 2.433},
    'Vilaller': {'lat': 42.477, 'lon': 0.716},
    'Vilallonga de Ter': {'lat': 42.333, 'lon': 2.311},
    'Vilallonga del Camp': {'lat': 41.213, 'lon': 1.206},
    'Vilamacolum': {'lat': 42.203, 'lon': 3.064},
    'Vilamalla': {'lat': 42.228, 'lon': 2.977},
    'Vilamaniscle': {'lat': 42.378, 'lon': 3.071},
    'Vilamòs': {'lat': 42.766, 'lon': 0.725},
    'Vilanant': {'lat': 42.261, 'lon': 2.888},
    'Vilanova de Bellpuig': {'lat': 41.611, 'lon': 0.963},
    'Vilanova de l\'Aguda': {'lat': 41.905, 'lon': 1.255},
    'Vilanova de la Barca': {'lat': 41.670, 'lon': 0.726},
    'Vilanova de Meià': {'lat': 41.996, 'lon': 1.021},
    'Vilanova de Prades': {'lat': 41.353, 'lon': 0.957},
    'Vilanova de Sau': {'lat': 41.947, 'lon': 2.385},
    'Vilanova de Segrià': {'lat': 41.714, 'lon': 0.621},
    'Vilanova del Camí': {'lat': 41.569, 'lon': 1.636},
    'Vilanova del Vallès': {'lat': 41.564, 'lon': 2.298},
    'Vilanova d\'Escornalbou': {'lat': 41.116, 'lon': 0.923},
    'Vilanova i la Geltrú': {'lat': 41.224, 'lon': 1.725},
    'Vilaplana': {'lat': 41.229, 'lon': 1.034},
    'Vilasantar': {'lat': 43.033, 'lon': -8.116},
    'Vilassar de Dalt': {'lat': 41.517, 'lon': 2.358},
    'Vilassar de Mar': {'lat': 41.506, 'lon': 2.392},
    'Vilaür': {'lat': 42.148, 'lon': 2.978},
    'Vilaverd': {'lat': 41.332, 'lon': 1.178},
    'Vilella Alta, La': {'lat': 41.203, 'lon': 0.771},
    'Vilella Baixa, La': {'lat': 41.200, 'lon': 0.751},
    'Vilobí d\'Onyar': {'lat': 41.884, 'lon': 2.761},
    'Vilobí del Penedès': {'lat': 41.385, 'lon': 1.666},
    'Viloví de la Selva': {'lat': 41.834, 'lon': 2.753},
    'Vimbodí i Poblet': {'lat': 41.398, 'lon': 1.047},
    'Vinebre': {'lat': 41.196, 'lon': 0.589},
    'Vinyols i els Arcs': {'lat': 41.118, 'lon': 1.041},
}




if not pobles_data:
    st.warning("La llista de localitats està buida. S'està utilitzant una llista de mostra.")
    pobles_data = {
        'Barcelona': {'lat': 41.387, 'lon': 2.168}, 'Girona': {'lat': 41.983, 'lon': 2.824},
        'Lleida': {'lat': 41.617, 'lon': 0.622}, 'Tarragona': {'lat': 41.118, 'lon': 1.245},
        'Puigcerdà': {'lat': 42.432, 'lon': 1.928}, 'Vielha': {'lat': 42.702, 'lon': 0.796},
        'Tortosa': {'lat': 40.812, 'lon': 0.521}
    }

# --- INICIALITZACIÓ DEL SESSION STATE ---
if 'poble_seleccionat' not in st.session_state:
    st.session_state.poble_seleccionat = next(iter(pobles_data))
if 'hora_seleccionada_str' not in st.session_state:
    try: tz = pytz.timezone('Europe/Madrid'); st.session_state.hora_seleccionada_str = f"{datetime.now(tz).hour:02d}:00h"
    except: st.session_state.hora_seleccionada_str = "12:00h"

# --- 1. LÒGICA DE CÀRREGA DE DADES ---
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

@st.cache_data(ttl=18000)
def carregar_dades_lot(_chunk_locations):
    p_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"] for p in p_levels]
    chunk_noms, chunk_lats, chunk_lons = zip(*_chunk_locations)
    params = { "latitude": list(chunk_lats), "longitude": list(chunk_lons), "hourly": h_base + h_press, "models": "arome_france", "timezone": "auto", "forecast_days": FORECAST_DAYS }
    intents_restants = 3
    while intents_restants > 0:
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            respostes_chunk = openmeteo.weather_api(url, params=params)
            lot_dict = {chunk_noms[i]: respostes_chunk[i] for i in range(len(respostes_chunk))}
            return lot_dict, p_levels, None
        except openmeteo_requests.exceptions.ApiError as e:
            if "Minutely API request limit exceeded" in str(e):
                intents_restants -= 1; time.sleep(61)
            else: return None, None, str(e)
        except Exception as e: return None, None, str(e)
    return None, None, "S'ha superat el límit de l'API després de diversos intents."

@st.cache_data(ttl=18000)
def obtener_dades_mapa_vents(hourly_index, nivell, forecast_days):
    lats, lons = np.linspace(40.5, 42.8, 12), np.linspace(0.2, 3.3, 12)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    params = {"latitude": lat_grid.flatten().tolist(), "longitude": lon_grid.flatten().tolist(), "hourly": [f"wind_speed_{nivell}hPa", f"wind_direction_{nivell}hPa"], "models": "arome_france", "timezone": "auto", "forecast_days": forecast_days}
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        responses = openmeteo.weather_api(url, params=params)
        lats_out, lons_out, speeds_out, dirs_out = [], [], [], []
        for r in responses:
            hourly = r.Hourly(); speed, direction = hourly.Variables(0).ValuesAsNumpy()[hourly_index], hourly.Variables(1).ValuesAsNumpy()[hourly_index]
            if not np.isnan(speed) and not np.isnan(direction):
                lats_out.append(r.Latitude()); lons_out.append(r.Longitude()); speeds_out.append(speed); dirs_out.append(direction)
        return lats_out, lons_out, speeds_out, dirs_out, None
    except Exception as e:
        return None, None, None, None, str(e)

# --- 2. TOTES LES FUNCIONS DE CÀLCUL I VISUALITZACIÓ ---
def get_next_arome_update_time():
    now_utc = datetime.now(pytz.utc)
    run_hours_utc = [0, 6, 12, 18]; availability_delay = timedelta(hours=4)
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
    return f"Pròxima actualització de dades (model AROME) estimada a les {next_update_local.strftime('%H:%Mh')}"

def display_avis_principal(titol_avís, text_avís, color_avís, icona_personalitzada=None):
    icon_map = { "ESTABLE": "☀️", "RISC BAIX": "☁️", "PRECAUCIÓ": "⚡️", "AVÍS": "⚠️", "RISC ALT": "🌪️", "ALERTA DE DISPARADOR": "🎯" }
    icona = icona_personalitzada if icona_personalitzada else icon_map.get(titol_avís, "ℹ️")
    st.markdown(f"""
    <style>
    .avis-container-pretty {{
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: var(--secondary-background-color); /* ADAPTATIU */
        border-left: 8px solid {color_avís};
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    .avis-icon-pretty {{ font-size: 3.5em; line-height: 1; }}
    .avis-text-pretty h3 {{ color: {color_avís}; margin-top: 0; margin-bottom: 0.5rem; font-weight: bold; }}
    .avis-text-pretty p {{ margin-bottom: 0; color: var(--text-color); }} /* ADAPTATIU */
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="avis-container-pretty">
        <div class="avis-icon-pretty">{icona}</div>
        <div class="avis-text-pretty">
            <h3>{titol_avís}</h3>
            <p>{text_avís}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_parameter_style(param_name, value):
    """
    Retorna un color i un emoji per a un paràmetre donat.
    Versió corregida per ser compatible amb temes clars (light mode).
    """
    # CORRECCIÓ 1: El color per defecte ara és 'inherit' per adaptar-se al tema
    color = "inherit"; emoji = ""
    
    if value is None or not isinstance(value, (int, float, np.number)):
        return color, emoji
    
    if param_name == 'SFC_Temp':
        if value > 36: color, emoji = "#FF0000", "🔥"
        elif value > 32: color, emoji = "#FF4500", ""
        elif value > 28: color = "#FFA500"
        elif value > 5:  color = "inherit" # S'adaptarà al tema
        elif value > 0:  color = "#87CEEB"
        elif value <= 0: color, emoji = "#0000FF", "🥶"
        
    elif param_name == 'CIN_Fre':
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
    elif 'W_MAX' in param_name:
        if value > 75: color, emoji = "#FF00FF", "⚠️"
        elif value > 50: color, emoji = "#FF4500", "⚠️"
        elif value > 25: color = "#FFA500"
        
    return color, emoji

def generar_avis_temperatura(params):
    temp = params.get('SFC_Temp', {}).get('value')
    if temp is None: return None, None, None, None
    if temp > 36: return "AVÍS PER CALOR EXTREMA", f"Es preveu una temperatura de {temp:.1f}°C. Risc molt alt per a la salut. Eviteu l'exposició al sol i manteniu-vos hidratats.", "#FF0000", "🥵"
    if temp < 0: return "AVÍS PER FRED INTENS", f"Es preveu una temperatura de {temp:.1f}°C. Risc de gelades fortes. Protegiu-vos del fred.", "#0000FF", "🥶"
    return None, None, None, None

def generar_avis_localitat(params):
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0)
    cin = params.get('CIN_Fre', {}).get('value')
    shear = params.get('Shear_0-6km', {}).get('value')
    srh1 = params.get('SRH_0-1km', {}).get('value')
    lcl_agl = params.get('LCL_AGL', {}).get('value', 9999)
    lfc_agl = params.get('LFC_AGL', {}).get('value', 9999)
    w_max = params.get('W_MAX', {}).get('value')
    if cape_u < 100: return "ESTABLE", "Sense risc de tempestes significatives. L'atmosfera és estable.", "#3CB371"
    if cin is not None and cin < -100: return "ESTABLE", "Sense risc de tempestes. La 'tapa' atmosfèrica (CIN) és massa forta.", "#3CB371"
    if lfc_agl > 3000: return "RISC BAIX", "El nivell d'inici de la convecció (LFC) és massa alt i difícil d'assolir.", "#4682B4"
    w_max_text = ""
    if w_max:
        if w_max > 50: w_max_text = " amb corrents ascendents violents"
        elif w_max > 25: w_max_text = " amb corrents ascendents molt forts"
    if shear is not None and shear > 20 and cape_u > 1500 and srh1 is not None and srh1 > 250 and lcl_agl < 1200:
        return "RISC ALT", f"Condicions per a SUPERCL·LULES{w_max_text}. Potencial de tornados, calamarsa grossa i vent destructiu.", "#DC143C"
    if shear is not None and shear > 18 and cape_u > 1000:
        return "AVÍS", f"Potencial per a SUPERCL·LULES{w_max_text}. Risc de calamarsa grossa i fortes ratxes de vent.", "#FF8C00"
    if shear is not None and shear > 12 and cape_u > 500:
        return "PRECAUCIÓ", f"Risc de TEMPESTES ORGANITZADES (multicèl·lules){w_max_text}. Possibles fortes pluges i calamarsa.", "#FFD700"
    return "RISC BAIX", "Possibles xàfecs o tempestes febles i aïllades (unicel·lulars).", "#4682B4"

def generar_avis_convergencia(params, is_convergence_active):
    if not is_convergence_active: return None, None, None
    cape_u = params.get('CAPE_Utilitzable', {}).get('value', 0)
    cin = params.get('CIN_Fre', {}).get('value')
    if cape_u > 500 and (cin is None or cin > -50):
        missatge = f"La forta convergència de vents pot actuar com a disparador. Amb un CAPE de {cape_u:.0f} J/kg i una 'tapa' (CIN) feble, hi ha un alt potencial que les tempestes s'iniciïn de manera explosiva."
        return "ALERTA DE DISPARADOR", missatge, "#FF4500"
    return None, None, None

def generar_analisi_detallada(params):
    """
    Genera una anàlisi detallada en format de text que s'emet paraula a paraula.
    Versió completa i correcta.
    """
    def stream_text(text):
        for word in text.split():
            yield word + " "
            time.sleep(0.02)
        yield "\n\n"

    cape, cin, cape_u = (params.get(k, {}).get('value') for k in ['CAPE_Brut', 'CIN_Fre', 'CAPE_Utilitzable'])
    shear6, srh1 = (params.get(k, {}).get('value') for k in ['Shear_0-6km', 'SRH_0-1km'])
    lcl_agl, lfc_agl = (params.get(k, {}).get('value') for k in ['LCL_AGL', 'LFC_AGL'])
    w_max = params.get('W_MAX', {}).get('value')

    yield from stream_text("### Anàlisi Termodinàmica")

    if cape is None or cape < 100:
        yield from stream_text("L'atmosfera és estable o quasi estable. El CAPE és pràcticament inexistent.")
        return # Important: Aquesta funció acaba aquí si no hi ha inestabilitat

    cape_text = "feble" if cape < 1000 else "moderada" if cape < 2500 else "forta" if cape < 3500 else "extrema"
    yield from stream_text(f"Tenim un CAPE de {cape:.0f} J/kg, un potencial energètic que indica inestabilitat {cape_text}.")
    
    if w_max:
        w_max_kmh = w_max * 3.6
        w_max_desc = "molt forts" if w_max_kmh > 90 else "forts"
        if w_max_kmh > 180: w_max_desc = "extremadament violents"
        yield from stream_text(f"Això es tradueix en corrents ascendents {w_max_desc} (~{w_max_kmh:.0f} km/h), un indicador de la potència de la tempesta i la seva capacitat per generar calamarsa grossa.")

    if cin is not None:
        if cin < -100:
            yield from stream_text(f"Factor limitant: La 'tapa' d'inversió (CIN) és molt forta ({cin:.0f} J/kg).")
        elif cin < -25:
            yield from stream_text(f"La 'tapa' (CIN) de {cin:.0f} J/kg és considerable. Si es trenca, pot donar lloc a un desenvolupament explosiu.")
        else:
            yield from stream_text("La 'tapa' (CIN) és feble. L'energia està fàcilment disponible.")

    if lfc_agl is not None and lfc_agl > 3000:
        yield from stream_text(f"Factor limitant: El nivell d'inici de convecció (LFC) està a {lfc_agl:.0f} m, una altura molt elevada.")
    elif lcl_agl is not None and lfc_agl is not None:
        yield from stream_text(f"La base del núvol (LCL) se situa a {lcl_agl:.0f} m, i el nivell de tret (LFC) a {lfc_agl:.0f} m.")

    yield from stream_text("### Anàlisi Cinemàtica")

    if shear6 is not None:
        if shear6 < 10:
            shear_text = "Molt feble. Tempestes desorganitzades (unicel·lulars)."
        elif shear6 < 18:
            shear_text = "Moderat. Potencial per a sistemes multicel·lulars."
        else:
            shear_text = "Fort. Suficient per suportar supercèl·lules rotatòries."
        yield from stream_text(f"El cisallament 0-6 km (Shear) és de {shear6:.1f} m/s. {shear_text}")

    if srh1 is not None and srh1 > 100:
        srh_text = "moderat" if srh1 < 250 else "fort"
        lcl_risk = " Amb la base del núvol baixa, facilita que la rotació arribi a terra." if lcl_agl is not None and lcl_agl < 1200 else ""
        yield from stream_text(f"L'Helicitat 0-1 km (SRH) és de {srh1:.0f} m²/s², un valor {srh_text} per a la rotació a nivells baixos.{lcl_risk}")

    yield from stream_text("### Síntesi i Riscos Associats")

    if cape_u < 100 or (cin is not None and cin < -100) or (lfc_agl is not None and lfc_agl > 3000):
        yield from stream_text("Condicions desfavorables per a tempestes significatives.")
    else:
        if lfc_agl is not None and cin is not None and cin < -10:
            yield from stream_text(f"LA CLAU: Un mecanisme de tret haurà de superar la 'tapa' de {abs(cin):.0f} J/kg i assolir {lfc_agl:.0f} m (LFC) per alliberar l'energia.")
        
        if shear6 is not None and shear6 > 18 and cape_u > 1000 and srh1 is not None and srh1 > 150:
            riscos = "calamarsa grossa, fortes ratxes de vent i pluges torrencials"
            if srh1 > 250 and lcl_agl is not None and lcl_agl < 1200:
                riscos += ", amb risc destacat de tornados"
            yield from stream_text(f"Entorn altament favorable per a supercèl·lules. Risc de {riscos}.")
        elif shear6 is not None and shear6 > 12 and cape_u > 500:
            yield from stream_text("Entorn òptim per a sistemes multicel·lulars. Risc de fortes pluges, calamarsa i ratxes de vent.")
        else:
            yield from stream_text("Entorn favorable per a xàfecs o tempestes unicel·lulars.")

def calculate_parameters(p, T, Td, u, v, h):
    params = {}
    def get_val(qty, unit=None):
        try: return qty.to(unit).m if unit else qty.m
        except: return None
    params['SFC_Temp'] = {'value': get_val(T[0], 'degC'), 'units': '°C'}
    raw_cape, raw_cin = None, None
    try:
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]); cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof)
        raw_cape, raw_cin = get_val(cape, 'J/kg'), get_val(cin, 'J/kg')
        params.update({'CAPE_Brut': {'value': raw_cape, 'units': 'J/kg'}, 'CIN_Fre': {'value': raw_cin, 'units': 'J/kg'}})
        if raw_cape and raw_cape > 0: params['W_MAX'] = {'value': np.sqrt(2 * raw_cape), 'units': 'm/s'}
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

def processar_sondeig_per_hora(sondeo, hourly_index, p_levels):
    try:
        hourly = sondeo.Hourly(); T_s, Td_s, P_s = (hourly.Variables(i).ValuesAsNumpy()[hourly_index] for i in range(3))
        if np.isnan(P_s): return None
        s_idx, n_plvls = 3, len(p_levels); T_p, Td_p, Ws_p, Wd_p, H_p = ([hourly.Variables(s_idx + i*n_plvls + j).ValuesAsNumpy()[hourly_index] for j in range(n_plvls)] for i in range(5))
        def interpolate_sfc(sfc_val, p_sfc, p_api, d_api):
            valid_p, valid_d = [p for p, t in zip(p_api, d_api) if not np.isnan(t)], [t for t in d_api if not np.isnan(t)]
            if np.isnan(sfc_val) and len(valid_p) > 1:
                p_sorted, d_sorted = zip(*sorted(zip(valid_p, valid_d))); return np.interp(p_sfc, p_sorted, d_sorted)
            return sfc_val
        T_s, Td_s = interpolate_sfc(T_s, P_s, p_levels, T_p), interpolate_sfc(Td_s, P_s, p_levels, Td_p)
        if np.isnan(T_s) or np.isnan(Td_s): return None
        p_profile, T_profile, Td_profile, u_profile, v_profile, h_profile = [P_s], [T_s], [Td_s], [0.0], [0.0], [mpcalc.pressure_to_height_std(P_s*units.hPa).m]
        for i, p_level in enumerate(p_levels):
            if p_level < P_s and not np.isnan(T_p[i]):
                p_profile.append(p_level); T_profile.append(T_p[i]); Td_profile.append(Td_p[i]); h_profile.append(H_p[i])
                u_comp, v_comp = mpcalc.wind_components(Ws_p[i]*units.knots, Wd_p[i]*units.degrees)
                u_profile.append(u_comp.to('m/s').m); v_profile.append(v_comp.to('m/s').m)
        return (np.array(p_profile)*units.hPa, np.array(T_profile)*units.degC, np.array(Td_profile)*units.degC,
                np.array(u_profile)*units.m/units.s, np.array(v_profile)*units.m/units.s, np.array(h_profile)*units.m)
    except Exception: return None

@st.cache_data(ttl=18000)
def encontrar_localitats_con_convergencia(_hourly_index, _nivell, _localitats, _threshold, _forecast_days):
    lats, lons, speeds, dirs, error = obtener_dades_mapa_vents(_hourly_index, _nivell, _forecast_days)
    if error or not lats or len(lats) < 4: return set()
    speeds_ms = (np.array(speeds) * 1000 / 3600) * units('m/s'); dirs_deg = np.array(dirs) * units.degrees
    u_comp, v_comp = mpcalc.wind_components(speeds_ms, dirs_deg)
    grid_lon, grid_lat = np.linspace(min(lons), max(lons), 100), np.linspace(min(lats), max(lats), 100)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    points = np.vstack((lons, lats)).T
    u_grid, v_grid = griddata(points, u_comp.m, (X, Y), method='cubic'), griddata(points, v_comp.m, (X, Y), method='cubic')
    u_grid, v_grid = np.nan_to_num(u_grid), np.nan_to_num(v_grid)
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y)
    divergence = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5
    localitats_en_convergencia = set()
    for nom_poble, coords in _localitats.items():
        lon_idx, lat_idx = (np.abs(grid_lon - coords['lon'])).argmin(), (np.abs(grid_lat - coords['lat'])).argmin()
        if divergence.m[lat_idx, lon_idx] < _threshold:
            localitats_en_convergencia.add(nom_poble)
    return localitats_en_convergencia

@st.cache_data(ttl=18000)
def precalcular_disparadors_actius(_totes_les_dades, _p_levels, _hourly_index, _localitats_convergencia):
    if not _localitats_convergencia: return set()
    disparadors = set()
    for nom_poble in _localitats_convergencia:
        sondeo = _totes_les_dades.get(nom_poble)
        if sondeo:
            profiles = processar_sondeig_per_hora(sondeo, _hourly_index, _p_levels)
            if profiles:
                parametros = calculate_parameters(*profiles)
                cape_u = parametros.get('CAPE_Utilitzable', {}).get('value', 0)
                cin = parametros.get('CIN_Fre', {}).get('value')
                if cape_u > 500 and cin is not None and cin > -50:
                    disparadors.add(nom_poble)
    return disparadors

@st.cache_data(ttl=18000)
def precalcular_avisos_hores(_totes_les_dades, _p_levels):
    avisos_hores = {}
    avisos_a_buscar = {"PRECAUCIÓ", "AVÍS", "RISC ALT"}
    for nom_poble, sondeo in _totes_les_dades.items():
        for hora in range(24):
            profiles = processar_sondeig_per_hora(sondeo, hora, _p_levels)
            if profiles:
                parametros = calculate_parameters(*profiles)
                titol_avís, _, _ = generar_avis_localitat(parametros)
                if titol_avís in avisos_a_buscar:
                    avisos_hores[nom_poble] = hora
                    break
    return dict(sorted(avisos_hores.items()))

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
    skew.plot(p, T, 'r', lw=2, label='T'); skew.plot(p, Td, 'b', lw=2, label='Td'); skew.plot_barbs(p, u, v, length=7, color='black') # ADAPTATIU
    skew.plot_dry_adiabats(color='lightcoral', ls='--', alpha=0.5); skew.plot_moist_adiabats(color='cornflowerblue', ls='--', alpha=0.5); skew.plot_mixing_lines(color='lightgreen', ls='--', alpha=0.5)
    skew.ax.axvline(0, color='darkturquoise', linestyle='--', label='Isoterma 0°C')
    if len(p) > 1:
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]); skew.plot(p, prof, 'k', lw=2, ls='--', label='Parcela')
            wet_bulb_prof = mpcalc.wet_bulb_temperature(p, T, Td); skew.plot(p, wet_bulb_prof, color='purple', lw=1.5, label='Tª Humida')
            cape, cin = mpcalc.cape_cin(p, T, Td, prof)
            if cape.m > 0: skew.shade_cape(p, T, prof, alpha=0.6, color='khaki')
            if cin.m != 0: skew.shade_cin(p, T, prof, alpha=1.83, color='gray')
            lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0]); lfc_p, _ = mpcalc.lfc(p, T, Td, prof); el_p, _ = mpcalc.el(p, T, Td, prof)
            if lcl_p: skew.ax.axhline(lcl_p.m, color='purple', linestyle='--', label='LCL')
            if lfc_p: skew.ax.axhline(lfc_p.m, color='darkred', linestyle='--', label='LFC')
            if el_p: skew.ax.axhline(el_p.m, color='red', linestyle='--', label='EL')
        except: pass
    skew.ax.set_ylim(1050, 100); skew.ax.set_xlim(-50, 40); skew.ax.set_xlabel('°C'); skew.ax.set_ylabel('hPa'); plt.legend()
    return fig

def display_metrics(params_dict):
    """
    Mostra els paràmetres clau amb format net i colors adaptatius.
    """
    param_map = [
        ('Temperatura', 'SFC_Temp'), ('CIN (Fre)', 'CIN_Fre'), ('CAPE (Brut)', 'CAPE_Brut'), 
        ('Shear 0-6km', 'Shear_0-6km'), ('Vel. Asc. Màx.', 'W_MAX'), ('CAPE Utilitzable', 'CAPE_Utilitzable'), 
        ('LCL (AGL)', 'LCL_AGL'), ('LFC (AGL)', 'LFC_AGL'), ('EL (MSL)', 'EL_MSL'), 
        ('SRH 0-1km', 'SRH_0-1km'), ('SRH 0-3km', 'SRH_0-3km'), ('PWAT Total', 'PWAT_Total')
    ]
    
    # CORRECCIÓ 2: La vora per defecte ara és d'un gris neutre compatible amb ambdós temes
    st.markdown("""<style>.metric-container{border:1px solid rgba(128,128,128,0.2);border-radius:10px;padding:10px;margin-bottom:10px;}</style>""", unsafe_allow_html=True)
    
    available_params = [(label, key) for label, key in param_map if key in params_dict and params_dict[key].get('value') is not None]
    cols = st.columns(min(4, len(available_params)))
    
    for i, (label, key) in enumerate(available_params):
        param = params_dict[key]; value = param['value']; units_str = param['units']
        
        if isinstance(value, (float, np.floating)):
            val_str = f"{value:.1f}"
        else:
            val_str = str(value)
        
        value_color, emoji = get_parameter_style(key, value)
        
        # CORRECCIÓ 3: Si el color és 'inherit', la vora també utilitza el color gris per defecte
        border_color = value_color if value_color != 'inherit' else 'rgba(128,128,128,0.2)'
        
        with cols[i % 4]:
            html = f"""
            <div class="metric-container" style="border-color:{border_color};">
                <div style="font-size:0.9em;color:gray;">{label}</div>
                <div style="font-size:1.25em;font-weight:bold;color:{value_color};">
                    {val_str} <span style='font-size:0.8em;color:gray;'>{units_str}</span> {emoji}
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

def crear_mapa_vents(lats, lons, u_comp, v_comp, nivell, lat_sel, lon_sel, nom_poble_sel):
    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([0, 3.5, 40.4, 43], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#E0E0E0", zorder=0); ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, zorder=1); ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', zorder=1)
    grid_lon, grid_lat = np.linspace(min(lons), max(lons), 100), np.linspace(min(lats), max(lats), 100)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    points = np.vstack((lons, lats)).T
    u_grid, v_grid = griddata(points, u_comp.m, (X, Y), method='cubic'), griddata(points, v_comp.m, (X, Y), method='cubic')
    u_grid, v_grid = np.nan_to_num(u_grid), np.nan_to_num(v_grid)
    dx, dy = mpcalc.lat_lon_grid_deltas(X, Y)
    divergence = mpcalc.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx, dy=dy) * 1e5
    divergence_values = np.ma.masked_where(divergence.m > -5.5, divergence.m)
    ax.contourf(X, Y, divergence_values, levels=np.linspace(-15.0, -5.5, 10), cmap='Reds_r', alpha=0.6, zorder=2, transform=ccrs.PlateCarree(), extend='min')
    ax.streamplot(grid_lon, grid_lat, u_grid, v_grid, color="#000000", density=5.9, linewidth=0.5, arrowsize=0.50, zorder=4, transform=ccrs.PlateCarree())
    ax.plot(lon_sel, lat_sel, 'o', markersize=12, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=2, transform=ccrs.Geodetic(), zorder=5)
    ax.text(lon_sel + 0.05, lat_sel + 0.05, nom_poble_sel, transform=ccrs.Geodetic(), zorder=6, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.set_title(f"Flux i focus de convergència a {nivell}hPa", weight='bold')
    return fig

def crear_grafic_orografia(params, zero_iso_h_agl):
    lcl_agl = params.get('LCL_AGL', {}).get('value')
    lfc_agl = params.get('LFC_AGL', {}).get('value')
    if lcl_agl is None or np.isnan(lcl_agl): return None
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.set_yticks(np.arange(0, 10.1, 0.5)); ax.set_facecolor('#4169E1')
    sky_cmap = mcolors.LinearSegmentedColormap.from_list("sky", ["#87CEEB", "#4682B4"])
    ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect='auto', cmap=sky_cmap, origin='lower', extent=[0, 10, 0, 10])
    ax.add_patch(Circle((8, 8.5), 0.8, color='yellow', alpha=0.3, zorder=1)); ax.add_patch(Circle((8, 8.5), 0.4, color='#FFFFE0', alpha=0.8, zorder=1))
    has_lfc = lfc_agl is not None and np.isfinite(lfc_agl)
    peak_h_m = lfc_agl if has_lfc else lcl_agl; peak_h_km = peak_h_m / 1000.0 + 0.1
    m_verts = [(0, 0), (1.5, 0.1 * peak_h_km), (3, 0.5 * peak_h_km), (5, peak_h_km), (7, 0.4 * peak_h_km), (8.5, 0.1 * peak_h_km), (10,0)]
    mountain_path = Polygon(m_verts, color='none', zorder=5); ax.add_patch(mountain_path)
    offset = mtransforms.Affine2D().translate(5, -5); shadow_transform = ax.transData + offset
    shadow = patches.PathPatch(mountain_path.get_path(), facecolor='black', alpha=0.3, transform=shadow_transform, zorder=2); ax.add_patch(shadow)
    x_points, y_points = np.random.uniform(0, 10, 2000), np.random.uniform(0, peak_h_km, 2000)
    points_inside = mountain_path.get_path().contains_points(np.vstack((x_points, y_points)).T)
    patches_col = [Circle((x, y), radius=np.random.rand() * 0.18 + 0.05, facecolor=np.random.choice(['#696969', '#808080', '#A9A9A9']) if y > 1.9 else np.random.choice(['#2E4600', '#486B00', '#556B2F', '#5E412F']), alpha=0.7, edgecolor='none') for x, y in zip(x_points[points_inside], y_points[points_inside])]
    ax.add_collection(PatchCollection(patches_col, match_original=True, zorder=6))
    ax.add_patch(Polygon(m_verts, facecolor='none', edgecolor='black', lw=1.5, zorder=7))
    if zero_iso_h_agl is not None and peak_h_km > zero_iso_h_agl.m / 1000:
        h_snow = zero_iso_h_agl.m / 1000; x_snow = np.linspace(0, 10, 200); y_mountain = np.interp(x_snow, [p[0] for p in m_verts], [p[1] for p in m_verts])
        ax.fill_between(x_snow, np.maximum(h_snow, y_mountain), peak_h_km + 1, where=y_mountain>=h_snow, facecolor='white', alpha=0.9, zorder=8)
    for _ in range(40):
        x_base, height = np.random.rand() * 10, np.random.rand() * 0.2 + 0.05
        ax.add_patch(Polygon([(x_base-0.08, 0), (x_base, height), (x_base+0.08, 0)], facecolor=np.random.choice(['#004d00', '#003300']), zorder=10))
    ax.axhline(lcl_agl/1000, color='grey', linestyle='--', lw=2.5, zorder=11)
    ax.text(-0.2, lcl_agl/1000, f" LCL ({lcl_agl:.0f} m) ", color='white', backgroundcolor='black', ha='right', va='center', weight='bold', fontsize=10)
    if has_lfc:
        ax.axhline(lfc_agl/1000, color='red', linestyle='--', lw=2.5, zorder=11)
        ax.text(10.2, lfc_agl/1000, f" LFC ({lfc_agl:.0f} m) ", color='white', backgroundcolor='red', ha='left', va='center', weight='bold', fontsize=10)
        ax.text(5, lfc_agl/1000 + 0.3, f" Altura de muntanya necessària per activar tempestes: {lfc_agl:.0f} m ", color='black', bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'), ha='center', va='center', weight='bold', fontsize=14, zorder=12)
    else: ax.text(5, 5, "No hi ha LFC accessible.\nL'orografia no pot iniciar convecció profunda.", ha='center', va='center', color='black', fontsize=14, weight='bold', bbox=dict(facecolor='lightblue', alpha=0.8, boxstyle='round,pad=0.5'))
    ax.set_ylim(0, 10); ax.set_xlim(0, 10); ax.set_ylabel("Altitud (km)"); ax.set_title("Potencial d'Activació per Orografia", weight='bold', fontsize=16)
    ax.set_xticklabels([]); ax.set_xticks([]); fig.tight_layout()
    return fig

def crear_grafic_nuvol(params, H, u, v, is_convergence_active):
    lcl_agl, lfc_agl, el_msl_km, cape, srh1 = (params.get(k, {}).get('value') for k in ['LCL_AGL', 'LFC_AGL', 'EL_MSL', 'CAPE_Brut', 'SRH_0-1km'])
    if lcl_agl is None or el_msl_km is None: return None
    cape = cape or 0
    fig, ax = plt.subplots(figsize=(6, 9), dpi=120)
    ax.set_facecolor('#4F94CD'); sky_cmap = mcolors.LinearSegmentedColormap.from_list("sky", ["#4F94CD", "#B0E0E6"])
    ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect='auto', cmap=sky_cmap, origin='lower', extent=[-5, 5, 0, 16])
    ax.add_patch(Polygon([(-5, 0), (5, 0), (5, 0.5), (-5, 0.5)], color='#3A1F04'))
    lcl_km = lcl_agl / 1000; el_km = el_msl_km - (H[0].m / 1000)
    if srh1 is not None and srh1 > 250 and lcl_km < 1.2: base_txt = "Potencial de Wall Cloud i Funnels (Tornados)"
    elif srh1 is not None and srh1 > 150 and lcl_km < 1.5: base_txt = "Potencial de Bases Giratories (Mesocicló)"
    elif cape > 1500: base_txt = "Bases Turbulentes (Shelf Cloud / Arcus)"
    else: base_txt = "Base Plana"
    ax.text(0, -0.5, base_txt, color='black', ha='center', weight='bold', fontsize=12) # ADAPTATIU
    if is_convergence_active and lfc_agl is not None and np.isfinite(lfc_agl) and cape > 100:
        lfc_km = lfc_agl / 1000; y_points = np.linspace(lfc_km, el_km, 100)
        cloud_width = 1.0 + np.sin(np.pi * (y_points - lfc_km) / (el_km - lfc_km)) * (1 + cape/2000)
        for y, width in zip(y_points, cloud_width):
            center_x = np.interp(y*1000, H.m, u.m) / 15
            for _ in range(30): ax.add_patch(Circle((center_x + (random.random() - 0.5) * width, y + (random.random() - 0.5) * 0.4), 0.2 + random.random() * 0.4, color='white', alpha=0.15, lw=0))
        anvil_wind_u = np.interp(el_km*1000, H.m, u.m) / 10; anvil_center_x = np.interp(el_km*1000, H.m, u.m) / 15
        for _ in range(100): ax.add_patch(Circle((anvil_center_x + (random.random() - 0.2) * 4 + anvil_wind_u, el_km + (random.random() - 0.5) * 0.5), 0.2 + random.random() * 0.6, color='white', alpha=0.2, lw=0))
        if cape > 2500: ax.add_patch(Circle((anvil_center_x, el_km + cape/5000), 0.4, color='white', alpha=0.5))
    else: ax.text(0, 8, "Sense disparador o energia\nsuficient per a convecció profunda.", ha='center', va='center', color='black', fontsize=16, weight='bold', bbox=dict(facecolor='lightblue', alpha=0.7, boxstyle='round,pad=0.5')) # ADAPTATIU
    barb_heights_km = np.arange(1, 15, 1); u_barbs, v_barbs = (np.interp(barb_heights_km * 1000, H.m, comp.to('kt').m) for comp in (u, v))
    ax.barbs(np.full_like(barb_heights_km, 4.5), barb_heights_km, u_barbs, v_barbs, length=7, color='black')
    ax.set_ylim(0, 16); ax.set_xlim(-5, 5); ax.set_ylabel("Altitud (km)"); ax.set_title("Visualització del Núvol", weight='bold'); ax.set_xticks([]); ax.grid(axis='y', linestyle='--', alpha=0.3)
    return fig

# --- 3. INTERFAZ I FLUX PRINCIPAL ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # El teu CSS aquí
st.markdown('<p class="main-title">⚡ Tempestes.cat</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Eina d\'Anàlisi i Previsió de Fenòmens Severs a Catalunya</p>', unsafe_allow_html=True)

totes_les_dades = {}; p_levels = []; error_carrega = None
locations_data = [(k, v['lat'], v['lon']) for k, v in pobles_data.items()]
chunk_size = 50; chunks = list(chunker(locations_data, chunk_size)); num_chunks = len(chunks)
progress_bar = st.progress(0, text="Iniciant la càrrega de dades...")
for i, chunk in enumerate(chunks):
    progress_bar.progress((i) / num_chunks, text=f"Carregant lot {i+1} de {num_chunks}...")
    dades_lot, p_levels_lot, error_lot = carregar_dades_lot(tuple(chunk))
    if error_lot: error_carrega = error_lot; break
    totes_les_dades.update(dades_lot)
    if not p_levels: p_levels = p_levels_lot
progress_bar.empty()
if error_carrega:
    st.error(f"No s'ha pogut carregar la informació base. L'aplicació no pot continuar. Error: {error_carrega}")
    st.stop()
st.toast("Dades de sondeig carregades correctament!", icon="✅")

with st.spinner("Buscant hores amb avisos a tot el territori..."):
    avisos_hores = precalcular_avisos_hores(totes_les_dades, p_levels)

def update_from_avis_selector():
    poble_avis = st.session_state.avis_selector
    if poble_avis in avisos_hores:
        hora_avis = avisos_hores[poble_avis]
        st.session_state.hora_seleccionada_str = f"{hora_avis:02d}:00h"
        st.session_state.poble_seleccionat = poble_avis

with st.container(border=True):
    col1, col2 = st.columns([1, 1], gap="large")
    hour_options = [f"{h:02d}:00h" for h in range(24)]
    with col1:
        index_hora = hour_options.index(st.session_state.hora_seleccionada_str)
        hora_sel_str = st.selectbox("Hora del pronòstic (Local):", options=hour_options, index=index_hora, key="hora_selector")
        st.session_state.hora_seleccionada_str = hora_sel_str
    with col2:
        p_levels_all = p_levels if p_levels else [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
        nivell_global = st.selectbox("Nivell d'anàlisi de vents:", p_levels_all, index=p_levels_all.index(850))
    if avisos_hores:
        opcions_avis = ["--- 🔥 Selecciona una localitat amb avís per anar-hi directament ---"] + list(avisos_hores.keys())
        st.selectbox("Localitats amb previsió d'avís:", options=opcions_avis, key='avis_selector', on_change=update_from_avis_selector)

st.markdown(f'<p class="update-info">🕒 {get_next_arome_update_time()}</p>', unsafe_allow_html=True)
hourly_index = int(st.session_state.hora_seleccionada_str.split(':')[0])

with st.spinner("Analitzant tot el territori..."):
    conv_threshold = -5.5
    localitats_convergencia = encontrar_localitats_con_convergencia(hourly_index, nivell_global, pobles_data, conv_threshold, FORECAST_DAYS)
    disparadors_actius = precalcular_disparadors_actius(totes_les_dades, p_levels, hourly_index, localitats_convergencia)

def update_poble_selection():
    poble_display = st.session_state.poble_selector
    st.session_state.poble_seleccionat = poble_display.replace("⚠️⛈️ ", "").replace(" (disparador actiu)", "")

sorted_pobles = sorted(pobles_data.keys())
opciones_display = [f"⚠️⛈️ {p} (disparador actiu)" if p in disparadors_actius else p for p in sorted_pobles]
try:
    current_selection_display = next(s for s in opciones_display if st.session_state.poble_seleccionat in s)
    default_index_display = opciones_display.index(current_selection_display)
except (StopIteration, ValueError): default_index_display = 0
poble_sel_display = st.selectbox('Selecciona una localitat:', options=opciones_display, index=default_index_display, key='poble_selector', on_change=update_poble_selection)
poble_sel = st.session_state.poble_seleccionat
lat_sel, lon_sel = pobles_data[poble_sel]['lat'], pobles_data[poble_sel]['lon']

sondeo = totes_les_dades.get(poble_sel)

if sondeo:
    data_is_valid = False
    with st.spinner(f"Processant dades per a {poble_sel}..."):
        profiles = processar_sondeig_per_hora(sondeo, hourly_index, p_levels)
        if profiles:
            p, T, Td, u, v, H = profiles; parametros = calculate_parameters(p, T, Td, u, v, H)
            zero_iso_h_agl = None
            try:
                T_c, H_m = T.to('degC').m, H.to('m').m
                if (idx := np.where(np.diff(np.sign(T_c)))[0]).size > 0:
                    h_zero_iso_msl = np.interp(0, [T_c[idx[0]+1], T_c[idx[0]]], [H_m[idx[0]+1], H_m[idx[0]]])
                    zero_iso_h_agl = (h_zero_iso_msl - H_m[0]) * units.m
            except Exception: pass
            data_is_valid = True
            
    if data_is_valid:
        avis_temp_titol, avis_temp_text, avis_temp_color, avis_temp_icona = generar_avis_temperatura(parametros)
        if avis_temp_titol:
            display_avis_principal(avis_temp_titol, avis_temp_text, avis_temp_color, icona_personalitzada=avis_temp_icona)
        
        is_conv_active = poble_sel in localitats_convergencia
        avis_conv_titol, avis_conv_text, avis_conv_color = generar_avis_convergencia(parametros, is_conv_active)
        if avis_conv_titol: display_avis_principal(avis_conv_titol, avis_conv_text, avis_conv_color)
        
        avis_titol, avis_text, avis_color = generar_avis_localitat(parametros)
        display_avis_principal(avis_titol, avis_text, avis_color)
        
        tab_list = ["🗨️ Anàlisi en Directe", "📊 Paràmetres", "🗺️ Mapa de Vents", "🧭Hodògraf", "📍Sondeig", "🏔️ Orografia", "☁️ Visualització"]
        selected_tab = st.radio("Navegació:", tab_list, index=0, horizontal=True, key="main_tabs")
        
        if selected_tab == "🗨️ Anàlisi en Directe": st.write_stream(generar_analisi_detallada(parametros))
        elif selected_tab == "📊 Paràmetres": st.subheader("Paràmetres Clau"); display_metrics(parametros)
        elif selected_tab == "🗺️ Mapa de Vents":
            st.subheader(f"Vents i Convergència a {nivell_global}hPa")
            with st.spinner("Generant mapa de vents..."):
                lats_map, lons_map, speeds_map, dirs_map, error_mapa = obtener_dades_mapa_vents(hourly_index, nivell_global, FORECAST_DAYS)
                if error_mapa: st.error(f"No s'han pogut obtenir les dades per al mapa de vents. Error: {error_mapa}")
                elif lats_map and len(lats_map) > 4:
                    speeds_ms = (np.array(speeds_map) * 1000 / 3600) * units('m/s'); dirs_deg = np.array(dirs_map) * units.degrees
                    u_map, v_map = mpcalc.wind_components(speeds_ms, dirs_deg); st.pyplot(crear_mapa_vents(lats_map, lons_map, u_map, v_map, nivell_global, lat_sel, lon_sel, poble_sel))
                else: st.error("No s'han pogut obtenir les dades per al mapa de vents (raó desconeguda).")
        elif selected_tab == "🧭Hodògraf": st.subheader("Hodògraf (0-10 km)"); st.pyplot(crear_hodograf(p, u, v, H))
        elif selected_tab == "📍Sondeig":
            st.subheader(f"Sondeig per a {poble_sel} ({datetime.now(pytz.timezone('Europe/Madrid')).strftime('%d/%m/%Y')} - {hourly_index:02d}:00h Local)")
            st.pyplot(crear_skewt(p, T, Td, u, v))
        elif selected_tab == "🏔️ Orografia":
            st.subheader("Potencial d'Activació per Orografia")
            if (fig_oro := crear_grafic_orografia(parametros, zero_iso_h_agl)): st.pyplot(fig_oro)
            else: st.info("No hi ha LCL per calcular el potencial orogràfic.")
        elif selected_tab == "☁️ Visualització":
            with st.spinner("Dibuixant la possible estructura del núvol..."):
                if (fig_nuvol := crear_grafic_nuvol(parametros, H, u, v, is_convergence_active=is_conv_active)): st.pyplot(fig_nuvol)
                else: st.info("No hi ha LCL o EL per visualitzar l'estructura del núvol.")
    else:
        st.warning(f"No s'han pogut calcular els paràmetres per a les {hourly_index:02d}:00h. Proveu amb una altra hora o localitat.")
else:
    st.error(f"No s'han pogut obtenir dades per a '{poble_sel}'. Pot ser que estigui fora de la cobertura del model AROME o que hi hagi un problema amb la connexió.")
