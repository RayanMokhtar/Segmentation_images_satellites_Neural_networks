import requests

def get_weather_visualcrossing(api_key, lat, lon, date):
    """
    Récupère les données météorologiques historiques pour une localisation et une date donnée
    via l'API Visual Crossing.

    Args:
        api_key (str): Votre clé API Visual Crossing.
        lat (float): Latitude de la région.
        lon (float): Longitude de la région.
        date (str): Date au format 'YYYY-MM-DD'.

    Returns:
        dict: Données météorologiques ou un message d'erreur.
    """
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    url = f"{base_url}{lat},{lon}/{date}/{date}"
    params = {
        "unitGroup": "metric",  # Unités métriques (Celsius, mm, etc.)
        "key": api_key,
        "include": "obs"  # Inclure les observations historiques
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Lèvera une exception pour les codes d'erreur HTTP 4xx/5xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP: {http_err} - {response.text}")
        return {"error": response.text, "status_code": response.status_code}
    except requests.exceptions.RequestException as req_err:
        print(f"Erreur de requête: {req_err}")
        return {"error": str(req_err)}

if __name__ == "__main__":
    # Exemple d'utilisation
    # REMPLACEZ CECI PAR VOTRE VRAIE CLÉ API VISUAL CROSSING
    api_key = "PUTUNBTHW5R3Q9K2WUW6MPSD6"
    
    latitude = -19.138148  # Latitude exemple
    longitude = 146.851468  # Longitude exemple
    date_str = "2019-01-18"  # Date souhaitée

    print(f"Récupération des données météo pour Lat: {latitude}, Lon: {longitude}, Date: {date_str}")
    weather_data = get_weather_visualcrossing(api_key, latitude, longitude, date_str)

    if "error" in weather_data:
        print("Erreur lors de la récupération des données :")
        if "status_code" in weather_data:
            print(f"  Status Code: {weather_data['status_code']}")
        print(f"  Message: {weather_data['error']}")
    else:
        print("Données météo récupérées avec succès :")
        # Afficher quelques informations clés pour vérifier
        if weather_data.get('days'):
            print(f"  Température max: {weather_data['days'][0].get('tempmax')}")
            print(f"  Température min: {weather_data['days'][0].get('tempmin')}")
            print(f"  Conditions: {weather_data['days'][0].get('conditions')}")
        else:
            print("  Format de données inattendu.")
        # print("Données météo complètes:", weather_data) # Décommentez pour voir toutes les données
