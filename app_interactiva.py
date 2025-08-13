import openmeteo_requests
import requests
from retry_requests import retry

print("--- INICIANT PROVA DE CONNEXIÓ DIRECTA ---")

try:
    # 1. Configuració del client (sense memòria cau)
    print("Pas 1: Creant client de xarxa...")
    plain_session = requests.Session()
    retry_session = retry(plain_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    print("Pas 1: Client creat amb èxit.")

    # 2. Paràmetres per a una única trucada
    params = {
        "latitude": 41.387,
        "longitude": 2.168,
        "hourly": "temperature_2m",
        "models": "arome_seamless",
        "forecast_days": 1
    }
    print("Pas 2: Paràmetres definits.")

    # 3. La trucada a l'API
    print("Pas 3: REALITZANT LA TRUCADA A L'API... (Aquest és el punt crític)")
    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    print("Pas 4: TRUCADA COMPLETADA AMB ÈXIT!")

    # 4. Processar la resposta
    response = responses[0]
    print(f"Coordenades rebudes: {response.Latitude()}°N {response.Longitude()}°E")
    
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    print("Dades de temperatura rebudes correctament.")
    print("--- PROVA FINALITZADA AMB ÈXIT! LA CONNEXIÓ FUNCIONA. ---")

except Exception as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! HA FALLAT LA PROVA DE CONNEXIÓ !!!")
    print(f"!!! ERROR: {e}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\nDiagnòstic: El problema és de xarxa. El teu entorn no pot contactar amb l'API.")
