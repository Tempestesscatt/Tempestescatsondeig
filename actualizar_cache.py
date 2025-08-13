# -*- coding: utf-8 -*-
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import openmeteo_requests
from retry_requests import retry
import requests
import numpy as np

CACHE_FILE = Path(__file__).parent / "meteo_cache.json"
MAP_LATS, MAP_LONS = np.linspace(40.5, 42.8, 12), np.linspace(0.2, 3.3, 12)
LON_GRID, LAT_GRID = np.meshgrid(MAP_LONS, MAP_LATS)

pobles_data = {


    # Maresme i Costa de Barcelona
    'Malgrat de Mar': {'lat': 41.645, 'lon': 2.741}, 'Santa Susanna': {'lat': 41.636, 'lon': 2.711},
    'Pineda de Mar': {'lat': 41.626, 'lon': 2.689}, 'Calella': {'lat': 41.614, 'lon': 2.664},
    'Sant Pol de Mar': {'lat': 41.602, 'lon': 2.624}, 'Canet de Mar': {'lat': 41.590, 'lon': 2.580},
    'Arenys de Mar': {'lat': 41.581, 'lon': 2.551}, 'Caldes d\'Estrac': {'lat': 41.573, 'lon': 2.529},
    'Sant Vicenç de Montalt': {'lat': 41.572, 'lon': 2.508}, 'Vilassar de Mar': {'lat': 41.506, 'lon': 2.392},
    'Premià de Mar': {'lat': 41.491, 'lon': 2.359}, 'El Masnou': {'lat': 41.481, 'lon': 2.318},
    'Montgat': {'lat': 41.464, 'lon': 2.279}, 'Sant Adrià de Besòs': {'lat': 41.428, 'lon': 2.219},

   
}

FORECAST_DAYS = 1
MODELS = "arome_seamless"
MAP_LEVELS = [1000, 925, 850, 700, 500, 300]
MAP_VARIABLES = {
    "temp_height": ["temperature", "geopotential_height"],
    "wind": ["wind_speed", "wind_direction"],
    "dewpoint": ["dew_point"],
    "humidity": ["relative_humidity"]
}
SOUNDING_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]

def fetch_map_data(client, api_vars, forecast_days):
    """Obté dades per a la graella de mapes."""
    params = {
        "latitude": LAT_GRID.flatten().tolist(),
        "longitude": LON_GRID.flatten().tolist(),
        "hourly": api_vars,
        "models": MODELS,
        "timezone": "auto",
        "forecast_days": forecast_days
    }
    responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    
    processed_responses = []
    for r in responses:
        processed_responses.append({
            "latitude": r.Latitude(),
            "longitude": r.Longitude(),
            "hourly": {
                var.Variable().Name(): var.ValuesAsNumpy().tolist() 
                for var in r.Hourly().Variables()
            }
        })
    return processed_responses

def fetch_sounding_data(client, lat, lon, forecast_days):
    """Obté dades de sondeig per a una localitat."""
    h_base = ["temperature_2m", "dew_point_2m", "surface_pressure"]
    h_press = [f"{v}_{p}hPa" for v in ["temperature", "dew_point", "wind_speed", "wind_direction", "geopotential_height"] for p in SOUNDING_LEVELS]
    
    params = {
        "latitude": lat, "longitude": lon, "hourly": h_base + h_press,
        "models": MODELS, "timezone": "auto", "forecast_days": forecast_days
    }
    responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    
    r = responses[0]
    return {
        "latitude": r.Latitude(),
        "longitude": r.Longitude(),
        "hourly": {
            var.Variable().Name(): var.ValuesAsNumpy().tolist()
            for var in r.Hourly().Variables()
        }
    }

def main():
    """Funció principal que executa l'actualització."""
    print(f"Iniciant l'actualització de la caché a les {datetime.now()}")
    start_time = time.time()
    
    plain_session = requests.Session()
    retry_session = retry(plain_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    cache_data = {
        "map_data": {},
        "sounding_data": {}
    }

    print("Descarregant dades per als mapes...")
    for var_name, api_vars_list in MAP_VARIABLES.items():
        for level in MAP_LEVELS:
            api_vars_with_level = [f"{v}_{level}hPa" for v in api_vars_list]
            cache_key = f"{var_name}_{level}"
            try:
                print(f"  - Obtenint {cache_key}...")
                cache_data["map_data"][cache_key] = fetch_map_data(openmeteo, api_vars_with_level, FORECAST_DAYS)
            except Exception as e:
                print(f"    ERROR en obtenir {cache_key}: {e}")

    print("\nDescarregant sondeigs per a cada localitat...")
    for nom_poble, coords in pobles_data.items():
        try:
            print(f"  - Obtenint sondeig per a {nom_poble}...")
            cache_data["sounding_data"][nom_poble] = fetch_sounding_data(openmeteo, coords['lat'], coords['lon'], FORECAST_DAYS)
        except Exception as e:
            print(f"    ERROR en obtenir sondeig per a {nom_poble}: {e}")

    final_data = {
        "last_update_utc": datetime.now(timezone.utc).isoformat(),
        "data": cache_data
    }
    
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(final_data, f)
        print(f"\nCaché guardada correctament a '{CACHE_FILE}'")
    except Exception as e:
        print(f"\nError crític al guardar el fitxer de caché: {e}")
        
    end_time = time.time()
    print(f"Procés completat en {end_time - start_time:.2f} segons.")

if __name__ == "__main__":
    main()
