from sentinelhub import (
    SHConfig, BBox, CRS, bbox_to_dimensions,
    SentinelHubRequest, DataCollection, MimeType, MosaickingOrder
)
import requests
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime
import json
import math 
from .configuration import CLIENT_ID , CLIENT_SECRET 

def get_acquisition_date(bbox_coords, start_date, end_date, client_id, client_secret):
    """
    Récupère la date d'acquisition exacte des images Sentinel-1 via l'API Catalog.
    
    Args:
        bbox_coords: Coordonnées de la zone d'intérêt [minX, minY, maxX, maxY]
        start_date: Date de début au format ISO (YYYY-MM-DD)
        end_date: Date de fin au format ISO (YYYY-MM-DD)
        client_id: Identifiant client pour l'authentification OAuth
        client_secret: Secret client pour l'authentification OAuth
        
    Returns:
        Une liste de dates d'acquisition ou None si erreur
    """
    try:
        # 1. Obtenir un token OAuth2
        token_url = "https://services.sentinel-hub.com/oauth/token"
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        token = token_response.json()["access_token"]
        
        # 2. Préparer la requête de recherche STAC
        catalog_url = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Formater les dates au format ISO avec heure
        start_datetime = f"{start_date}T00:00:00Z"
        end_datetime = f"{end_date}T23:59:59Z"
        
        payload = {
            "bbox": bbox_coords,
            "datetime": f"{start_datetime}/{end_datetime}",
            "collections": ["sentinel-1-grd"],
            "limit": 5,  # On limite à 5 résultats pour éviter trop de données
            "query": {
                "sar:instrument_mode": {"eq": "IW"}  # Mode Interferometric Wide Swath
            }
        }
        
        # 3. Envoyer la requête et analyser les résultats
        print("🔍 Recherche des métadonnées d'acquisition précises...")
        response = requests.post(catalog_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # 4. Extraire les dates d'acquisition
        results = response.json()
        
        # Sauvegarder la réponse complète pour débogage et analyse
        with open("catalog_response.json", "w") as f:
            json.dump(results, f, indent=2)
            
        features = results.get("features", [])
        acquisition_dates = []
        
        print(f"\n📋 RÉSULTATS DE RECHERCHE: {len(features)} images trouvées")
        for i, feature in enumerate(features):
            properties = feature.get("properties", {})
            acquisition_dt = properties.get("datetime")
            platform = properties.get("platform", "inconnu")
            orbit_state = properties.get("sat:orbit_state", "inconnu")
            
            # Extraire d'autres métadonnées utiles
            cloud_cover = properties.get("eo:cloud_cover", "N/A")
            instrument = properties.get("instrument", "SAR")
            
            print(f"\n🛰️ Image {i+1}:")
            print(f"  📅 Date d'acquisition: {acquisition_dt}")
            print(f"  🛰️ Plateforme: {platform}")
            print(f"  🧭 Direction orbitale: {orbit_state.upper()}")
            
            acquisition_dates.append({
                "datetime": acquisition_dt,
                "platform": platform,
                "orbit_state": orbit_state,
                "cloud_cover": cloud_cover,
                "instrument": instrument
            })
            
        return acquisition_dates
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération des dates d'acquisition: {e}")
        return None


def get_image_satellitaire(
    client_id: str,
    client_secret: str,
    bbox_coords: list[float],
    start_date: str,
    end_date: str,
    orbit_direction: str = "DESCENDING",
    resolution: int = 20,
    is_visualised: bool = False,
    save_path: str = None
) -> np.ndarray:

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret

    data_folder = os.path.join(os.getcwd(), "sentinelhub_data")
    os.makedirs(data_folder, exist_ok=True)
    config.sh_data_folder = data_folder

    # Récupérer d'abord les dates précises d'acquisition
    acquisition_info = get_acquisition_date(bbox_coords, start_date, end_date, client_id, client_secret)
    
    # Créer le dictionnaire de métadonnées avec plus de précision
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    metadata_dict = {
        "acquisition": {
            "start_date": start_date,
            "end_date": end_date,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "orbit_direction": orbit_direction
        },
        "processing": {
            "resolution": resolution,
            "software": "FloodAI Processor",
            "version": "1.0.0",
            "processed_at": datetime.now().isoformat()
        },
        "geographic": {
            "bbox": bbox_coords,
            "crs": "EPSG:4326",
            "dimensions": {"width": size[0], "height": size[1]}
        },
        "sensor": {
            "satellite": "Sentinel-1",
            "instrument": "SAR C-Band",
            "polarisations": ["VV", "VH"]
        }
    }
    
    # Ajouter les dates précises d'acquisition si disponibles
    if acquisition_info and len(acquisition_info) > 0:
        metadata_dict["acquisition"]["precise_dates"] = acquisition_info
        metadata_dict["acquisition"]["source_date"] = acquisition_info[0]["datetime"]
    try:
        response_metadata = request.download_list[0].get_header('rhh')
        if response_metadata:
            if 'date' in response_metadata:
                metadata_dict["acquisition"]["source_date"] = response_metadata['date']
            else : 
                metadata_dict["acquisition"]["source_date"] = "Non disponible"
            metadata_dict["sentinel_hub"] = response_metadata
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération des métadonnées HTTP: {e}")
    
    print("\n📋 MÉTADONNÉES DE L'IMAGE:")
    print(f"  🛰️  Satellite: Sentinel-1 (SAR)")
    print(f"  📅 Période d'acquisition: {start_date} à {end_date}")
    if "source_date" in metadata_dict["acquisition"]:
        print(f"  📅 Date exacte: {metadata_dict['acquisition']['source_date']}")
    print(f"  🧭 Direction orbitale: {orbit_direction}")
    print(f"  📐 Résolution: {resolution}m")
    print(f"  🗺️  Zone: [{bbox_coords[0]:.4f}, {bbox_coords[1]:.4f}, {bbox_coords[2]:.4f}, {bbox_coords[3]:.4f}]")
    print(f"  📏 Dimensions: {size[0]}×{size[1]} pixels")
    

    evalscript = """
        //VERSION=3
        function setup() {
        return {
            input: [{
                bands: ["VV", "VH"]
                // La ligne units: "dB" a été supprimée
            }],
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        };
        }
        function evaluatePixel(sample) {
        return [sample.VV, sample.VH];
        }
        """

    # 4️⃣ Construction de la requête
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=(start_date, end_date),
                mosaicking_order=MosaickingOrder.MOST_RECENT,
                other_args={"orbitDirection": orbit_direction}
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=config
    )

    # 5️⃣ Télécharger les données

    print("📡 Téléchargement de l'image Sentinel-1...")
    request.data_folder = data_folder

    response_data = request.get_data(save_data=True)

    # Format de réponse: tableau NumPy avec dimensions [height, width, bands]
    sar_image = response_data[0]  
    
    # 6️⃣ Extraire la date depuis les métadonnées
    try:
        # Extraire les métadonnées des entêtes HTTP
        metadata = request.download_list[0].get_header('rhh')
        if metadata and 'date' in metadata:
            print(f"📅 Date de l'image: {metadata['date']}")
        else:
            print("⚠️ Date de l'image non disponible dans les métadonnées")
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération des métadonnées: {e}")
    
    # 7️⃣ Afficher des informations sur les données
    print(f"✅ Image téléchargée: {sar_image.shape}")
    print(f"📊 VV - min={sar_image[:,:,0].min():.4f}, max={sar_image[:,:,0].max():.4f}, moyenne={sar_image[:,:,0].mean():.4f}")
    print(f"📊 VH - min={sar_image[:,:,1].min():.4f}, max={sar_image[:,:,1].max():.4f}, moyenne={sar_image[:,:,1].mean():.4f}")

    # 8️⃣ Visualiser les données si demandé
    if is_visualised:
        plt.figure(figsize=(15, 8))
        
        # Afficher VV (linéaire)
        plt.subplot(1, 2, 1)
        plt.imshow(sar_image[:, :, 0], cmap="gray")
        plt.colorbar(label='Backscatter (linear)')
        plt.title("Sentinel-1 SAR - VV")
        plt.axis("off")
        
        # Afficher VH (linéaire)
        plt.subplot(1, 2, 2)
        plt.imshow(sar_image[:, :, 1], cmap="gray")
        plt.colorbar(label='Backscatter (linear)')
        plt.title("Sentinel-1 SAR - VH")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    # 9️⃣ Sauvegarder en TIFF si un chemin est fourni
    if save_path:
        try:
            # Créer le dossier sortie s'il n'existe pas
            output_dir = os.path.join(os.getcwd(), "sortie")
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculer la transformation géographique
            transform = from_bounds(
                bbox_coords[0], bbox_coords[1],  # min X, min Y
                bbox_coords[2], bbox_coords[3],  # max X, max Y
                sar_image.shape[1], sar_image.shape[0]  # width, height
            )
            
            # Préparer les métadonnées communes
            metadata_base = {
                'driver': 'GTiff',
                'height': sar_image.shape[0],
                'width': sar_image.shape[1],
                'dtype': rasterio.float32,
                'crs': '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                'transform': transform,
            }
            
            # Sauvegarder le fichier combiné si demandé
            full_save_path = os.path.join(output_dir, os.path.basename(save_path))
            with rasterio.open(full_save_path, 'w', **{**metadata_base, 'count': 2}) as dst:
                dst.write(sar_image[:,:,0], 1)  # VV
                dst.write(sar_image[:,:,1], 2)  # VH
                dst.set_band_description(1, "VV polarization")
                dst.set_band_description(2, "VH polarization")
            print(f"✅ Image combinée sauvegardée: {full_save_path}")
            
            # Sauvegarder VV séparément
            base_filename = os.path.splitext(os.path.basename(save_path))[0]
            vv_path = os.path.join(output_dir, f"{base_filename}_VV.tiff")
            with rasterio.open(vv_path, 'w', **{**metadata_base, 'count': 1}) as dst:
                dst.write(sar_image[:,:,0], 1)  # VV uniquement
                dst.set_band_description(1, "VV polarization")
            print(f"✅ Image VV sauvegardée: {vv_path}")
            
            # Sauvegarder VH séparément
            vh_path = os.path.join(output_dir, f"{base_filename}_VH.tiff")
            with rasterio.open(vh_path, 'w', **{**metadata_base, 'count': 1}) as dst:
                dst.write(sar_image[:,:,1], 1)  # VH uniquement
                dst.set_band_description(1, "VH polarization")
            print(f"✅ Image VH sauvegardée: {vh_path}")
            
        except ImportError:
            print("⚠️ La bibliothèque 'rasterio' est nécessaire pour sauvegarder en TIFF.")
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde: {e}")

    return sar_image



def read_satellite_image(input_data: np.ndarray, is_visualised: bool = False) -> np.ndarray:
    """
    Traite un tableau NumPy contenant des données SAR avec les deux bandes VV et VH.
    
    Args:
        input_data: Tableau NumPy [hauteur, largeur, 2] contenant les bandes VV et VH
        is_visualised: Si True, affiche les visualisations des deux bandes
        
    Returns:
        Le même tableau NumPy avec les deux bandes
    """
    # Vérifier que l'entrée est bien un tableau NumPy
    if not isinstance(input_data, np.ndarray):
        print("⚠️ L'entrée doit être un tableau NumPy")
        return None
        
    # Vérifier la structure du tableau (hauteur, largeur, 2 bandes)
    if len(input_data.shape) != 3 or input_data.shape[2] < 2:
        print(f"⚠️ Format de tableau incorrect. Attendu: [hauteur, largeur, 2+]. Reçu: {input_data.shape}")
        return None
    
    # Extraire les deux bandes
    vv_band = input_data[:,:,0]
    vh_band = input_data[:,:,1]
            
    print("\n📋 ANALYSE DES DONNÉES SAR:")
    print(f"📏 Dimensions: {input_data.shape}")
    print(f"📊 VV - min={vv_band.min():.4f}, max={vv_band.max():.4f}, moyenne={vv_band.mean():.4f}, écart-type={vv_band.std():.4f}")
    print(f"📊 VH - min={vh_band.min():.4f}, max={vh_band.max():.4f}, moyenne={vh_band.mean():.4f}, écart-type={vh_band.std():.4f}")
    
    # Calculer le ratio VH/VV (utile pour l'analyse)
    valid_mask = (vv_band > 0)
    ratio = np.zeros_like(vv_band)
    ratio[valid_mask] = vh_band[valid_mask] / vv_band[valid_mask]
    
    print(f"📊 Ratio VH/VV - min={np.nanmin(ratio):.4f}, max={np.nanmax(ratio):.4f}, moyenne={np.nanmean(ratio):.4f}")
    
    # Visualisation des bandes si demandée
    if is_visualised:
        plt.figure(figsize=(18, 12))
        
        # Améliorer le contraste pour la visualisation
        def enhance_contrast(img, p_low=2, p_high=98):
            p_min = np.percentile(img[img > 0], p_low) if np.sum(img > 0) > 0 else 0
            p_max = np.percentile(img[img > 0], p_high) if np.sum(img > 0) > 0 else 1
            return np.clip((img - p_min) / (p_max - p_min + 1e-10), 0, 1)
            
        # Visualisation des bandes individuelles
        plt.subplot(2, 2, 1)
        vv_enhanced = enhance_contrast(vv_band)
        plt.imshow(vv_enhanced, cmap='gray')
        plt.colorbar(label='Rétrodiffusion normalisée')
        plt.title("VV - Standard", fontsize=12)
        plt.axis("off")
        
        plt.subplot(2, 2, 2)
        vh_enhanced = enhance_contrast(vh_band)
        plt.imshow(vh_enhanced, cmap='gray')
        plt.colorbar(label='Rétrodiffusion normalisée')
        plt.title("VH - Standard", fontsize=12)
        plt.axis("off")
        
        # Visualisations améliorées
        plt.subplot(2, 2, 3)
        plt.imshow(vv_enhanced, cmap='inferno')
        plt.colorbar(label='Rétrodiffusion normalisée')
        plt.title("VV - Enhanced (Inferno)", fontsize=12)
        plt.axis("off")
        
        plt.subplot(2, 2, 4)
        plt.imshow(vh_enhanced, cmap='viridis')
        plt.colorbar(label='Rétrodiffusion normalisée')
        plt.title("VH - Enhanced (Viridis)", fontsize=12)
        plt.axis("off")
        
        plt.suptitle("Analyse SAR - Polarisations VV et VH", fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Afficher également une composition colorée RGB
        plt.figure(figsize=(12, 10))
        
        # Créer une composition colorée: R=VV, G=VH, B=VV/VH
        rgb = np.zeros((vv_band.shape[0], vv_band.shape[1], 3))
        rgb[:,:,0] = enhance_contrast(vv_band)
        rgb[:,:,1] = enhance_contrast(vh_band)
        rgb[:,:,2] = enhance_contrast(ratio, p_low=5, p_high=95)
        
        plt.imshow(rgb)
        plt.title("Composition colorée (R=VV, G=VH, B=Ratio VH/VV)", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        # Histogrammes des valeurs
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(vv_band.flatten(), bins=100, alpha=0.7, color='red')
        plt.title("Histogramme VV", fontsize=12)
        plt.xlabel("Valeur")
        plt.ylabel("Fréquence")
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(vh_band.flatten(), bins=100, alpha=0.7, color='green')
        plt.title("Histogramme VH", fontsize=12)
        plt.xlabel("Valeur")
        plt.ylabel("Fréquence")
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 3, 3)
        valid_ratios = ratio[~np.isnan(ratio) & ~np.isinf(ratio) & (ratio < 5)]  # Filtrer les valeurs aberrantes
        plt.hist(valid_ratios.flatten(), bins=100, alpha=0.7, color='blue')
        plt.title("Histogramme Ratio VH/VV", fontsize=12)
        plt.xlabel("Valeur")
        plt.ylabel("Fréquence")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return input_data

def get_bbox_from_long_lat(longitude: float, latitude: float, size_km: float = 10.0, aspect_ratio: float = 1.0) -> list:
    """
    Génère une bounding box (rectangle) autour des coordonnées de longitude et latitude spécifiées.
    
    Args:
        longitude: Longitude du centre (en degrés décimaux)
        latitude: Latitude du centre (en degrés décimaux)
        size_km: Taille approximative de la bbox en kilomètres (côté le plus long)
        aspect_ratio: Rapport largeur/hauteur du rectangle (1.0 = carré)
        
    Returns:
        Liste [minX, minY, maxX, maxY] définissant la bounding box
    """
    # Constantes de conversion
    # À l'équateur, 1 degré = ~111 km
    km_per_degree_lat = 111.0
    # La distance en longitude varie avec la latitude
    km_per_degree_long = 111.0 * math.cos(math.radians(abs(latitude)))
    
    # Calculer les dimensions en degrés
    if aspect_ratio >= 1.0:
        # Rectangle plus large que haut
        half_width_km = size_km / 2.0
        half_height_km = size_km / (2.0 * aspect_ratio)
    else:
        # Rectangle plus haut que large
        half_width_km = size_km * aspect_ratio / 2.0
        half_height_km = size_km / 2.0
    
    # Conversion en degrés
    half_width_deg = half_width_km / km_per_degree_long
    half_height_deg = half_height_km / km_per_degree_lat
    
    # Créer la bounding box [minX, minY, maxX, maxY]
    bbox = [
        longitude - half_width_deg,   # minX (ouest)
        latitude - half_height_deg,   # minY (sud)
        longitude + half_width_deg,   # maxX (est)
        latitude + half_height_deg    # maxY (nord)
    ]
    
    # Arrondir à 6 décimales (précision ~10cm)
    bbox = [round(coord, 6) for coord in bbox]
    
    return bbox

# Améliorer le contraste pour la visualisation
def enhance_sar_image(img, p_low=2, p_high=98):
    p_min = np.percentile(img, p_low)
    p_max = np.percentile(img, p_high)
    img_enhanced = np.clip((img - p_min) / (p_max - p_min), 0, 1)
    return img_enhanced
    


# # Centre historique de Rome (5km autour du Colisée)
# ROME_HISTORIC_BBOX = get_bbox_from_long_lat(2.2945, 48.8584, size_km=5)

# print(f"  📌 Zone: Centre historique de Rome")
# print(f"  🗺️ Coordonnées: {ROME_HISTORIC_BBOX}")
# print(f"  📏 Dimensions: ~5km × 5km")

# # Télécharger l'image SAR de Rome
# image = get_image_satellitaire(
#     CLIENT_ID, CLIENT_SECRET,
#     bbox_coords=ROME_HISTORIC_BBOX,
#     start_date="2025-01-30",
#     end_date="2025-02-07",
#     orbit_direction="DESCENDING",
#     resolution=10,
#     is_visualised=False,  # Visualisation activée
#     save_path="sentinel1_rome.tiff"
# )

# # Charger et visualiser les bandes VV et VH séparément
# image_lue = read_satellite_image(image, is_visualised=False)

# vv = image[:,:,0]
# vh = image[:,:,1]
# print(f"📊 Analyse des bandes VV et VH:")
# print(f"  📏 Dimensions: {vv.shape}")
# print(f" 📏 Dimensions: {vh.shape}")