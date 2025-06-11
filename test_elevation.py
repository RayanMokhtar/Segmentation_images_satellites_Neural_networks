import requests

def get_elevation_data(lat, lon):
    """
    Récupère les données d'élévation via l'API Open Topo Data.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        float: Élévation en mètres ou None en cas d'erreur
    """
    print(f"Récupération de l'élévation pour lat={lat}, lon={lon}...")
    
    # URL de base de l'API OpenTopoData
    elevation_api_base_url = "https://api.opentopodata.org/v1/srtm30m"
    
    # Construction de l'URL avec les paramètres
    url = f"{elevation_api_base_url}?locations={lat},{lon}"
    
    try:
        # Envoi de la requête GET
        response = requests.get(url)
        response.raise_for_status()  # Lever une exception en cas d'erreur HTTP
        
        # Analyse de la réponse JSON
        data = response.json()
        elevation = data['results'][0]['elevation']
        
        # Vérifier si l'élévation est null/None (cas des coordonnées en mer)
        if elevation is None:
            print(f"Élévation nulle (probablement des coordonnées maritimes). Utilisation de 0 mètre.")
            return 0
        else:
            print(f"Élévation récupérée: {elevation} mètres")
            return elevation
            
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP lors de la récupération de l'élévation: {http_err}")
        return None
    except (requests.exceptions.RequestException, KeyError, IndexError) as err:
        print(f"Erreur lors de la récupération de l'élévation: {err}")
        return None

# Exemple d'utilisation avec les coordonnées spécifiées
if __name__ == "__main__":
    latitude = 41.383
    longitude = 2.183
    elevation = get_elevation_data(latitude, longitude)
    
    if elevation is not None:
        print(f"L'élévation aux coordonnées ({latitude}, {longitude}) est de {elevation} mètres.")
    else:
        print(f"Impossible de récupérer l'élévation pour les coordonnées ({latitude}, {longitude}).")