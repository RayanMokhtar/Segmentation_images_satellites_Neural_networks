import requests
import pandas as pd
from standardize_fixed import standardize_features_fixed

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

if __name__ == "__main__":
    # 1. Récupérer l'élévation
    latitude = 41.383
    longitude = 2.183
    elevation = get_elevation_data(latitude, longitude)
    
    if elevation is None:
        print(f"Impossible de récupérer l'élévation pour les coordonnées ({latitude}, {longitude}).")
        exit(1)
    
    # 2. Créer un DataFrame simple avec quelques données fictives de météo
    data = {
        'temp': [25.5, 26.2, 24.8],
        'humidity': [65, 70, 62],
        'precipprob': [10, 20, 5],
        'elevation': [elevation, elevation, elevation]  # Même élévation pour toutes les lignes
    }
    df = pd.DataFrame(data)
    
    print("\nDataFrame original:")
    print(df)
    
    # 3. Standardiser avec la méthode normale (qui normalise l'élévation)
    from sklearn.preprocessing import StandardScaler
    numeric_features = ['temp', 'humidity', 'precipprob', 'elevation']
    df_normal = df.copy()
    scaler = StandardScaler()
    df_normal[numeric_features] = scaler.fit_transform(df_normal[numeric_features])
    
    print("\nDataFrame après standardisation normale (élévation normalisée):")
    print(df_normal)
    
    # 4. Standardiser avec notre fonction modifiée (qui préserve l'élévation)
    df_fixed = df.copy()
    df_fixed = standardize_features_fixed(df_fixed, numeric_features)
    
    print("\nDataFrame après standardisation fixée (élévation préservée):")
    print(df_fixed)
    
    # 5. Comparer les deux résultats
    print("\nComparaison des résultats:")
    print(f"Élévation originale: {elevation} mètres")
    print(f"Élévation après standardisation normale: {df_normal['elevation'].values[0]}")
    print(f"Élévation après standardisation fixée: {df_fixed['elevation'].values[0]}")
