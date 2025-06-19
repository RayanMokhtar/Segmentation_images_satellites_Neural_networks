import json
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import numpy as np
import cv2
import os
import redis
import hashlib
import traceback
from .services import get_image_satellitaire , get_bbox_from_long_lat
from .configuration import CLIENT_ID , CLIENT_SECRET
from .artificial_intelligence import FloodDetectionCNN , preprocess_sar_image_in_memory
from .cache import FloodPredictionCache
from .fake_prediction import FloodPredictionCache

MODEL_PATH = "C:/Users/mokht/Desktop/PDS/flood_dataset/resultats/best_model.pth"


SEUIL_FAIBLE=20
SEUIL_MOYEN=50
SEUIL_ELEVÉ=70
CACHE = FloodPredictionCache()

def load_model(model_path):
    """
    Charge un modèle PyTorch depuis le chemin spécifié
    
    Args:
        model_path (str): Chemin vers le fichier de poids du modèle (.pth)
        
    Returns:
        torch.nn.Module: Le modèle chargé en mode évaluation
    """
    try:

        model_path = MODEL_PATH
        model = FloodDetectionCNN(
            num_classes=2,      # Inondé ou non inondé
            input_channels=2,   # VH et VV (2 canaux)
            dropout_rate=0.5
        )
        
        # 2. Charger les poids du modèle (state_dict)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 3. Appliquer les poids au modèle
        model.load_state_dict(state_dict)
        
        # 4. Mettre en mode évaluation
        model.eval()
        
        print(f"Modèle chargé avec succès depuis {model_path}")
        return model
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        traceback.print_exc()
        raise


def predict_flood(longitude, latitude, size_km=5, start_date="2025-01-30", end_date="2025-02-07"):
    """
    Fonction simplifiée pour prédire les inondations à partir de coordonnées
    """
    try:
        print(f"Analyse de la zone: {longitude}, {latitude} (rayon {size_km}km)")
        
        cached_prediction = CACHE.get_prediction(longitude, latitude, size_km, start_date, end_date)
        if cached_prediction:
            print("Utilisation du résultat en cache")
            return cached_prediction
        
        print("🔄 Aucune donnée en cache, traitement en cours...")
        results_dir = "resultats_image"
        os.makedirs(results_dir, exist_ok=True)

        # 1. Récupérer les coordonnées de la bbox
        bbox_coords = get_bbox_from_long_lat(longitude, latitude, size_km)
        print(f"Coordonnées bbox: {bbox_coords}")
        
        # 2. Récupérer l'image satellitaire
        save_path = f"flood_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff"
        print(f"Récupération de l'image satellitaire...")
        
        image = get_image_satellitaire(
            CLIENT_ID, CLIENT_SECRET,
            bbox_coords=bbox_coords,
            start_date=start_date,
            end_date=end_date,
            orbit_direction="DESCENDING",
            resolution=10,
            is_visualised=False,
            save_path=save_path
        )
        if image is None or np.isnan(image).any():
            print("⚠️ Image non disponible ou contenant des valeurs NaN")
            return {
                'success': False, 
                'error': "Pas d'image disponible pour cette zone ou cette période"
            }
        
        # 3. Prétraiter l'image pour le modèle
        print("Prétraitement de l'image...")
        image_tensor = preprocess_sar_image_in_memory(
            image, 
            output_dir="flood_prediction/temp",
            visualize=False
        )
        
        # Vérifier que le tenseur existe
        if image_tensor is None:
            raise ValueError("Le prétraitement de l'image a échoué - image_tensor est None")
        
        print(f"Forme du tenseur d'image: {image_tensor.shape}")
        
        # Extraire VV et VH - CORRIGÉ pour le format [canaux, hauteur, largeur]
        vv_band = image_tensor[0, :, :].numpy()  # Premier canal (VV)
        vh_band = image_tensor[1, :, :].numpy()  # Second canal (VH)
        # Ajouter la dimension batch
        image_tensor = image_tensor.unsqueeze(0)
        
        model_path = MODEL_PATH
        
        # 4. Charger le modèle
        print(f"Chargement du modèle: {model_path}")
        model = load_model(model_path)
    
        # 5. Faire la prédiction
        print("Prédiction en cours...")
        model.eval()  # Mode évaluation
        
        with torch.no_grad():
            outputs = model(image_tensor)
            # Convertir en probabilités avec sigmoid
            flood_probability = torch.sigmoid(outputs[:, 1]).item()

            # Obtenir la classe prédite (0=non inondé, 1=inondé)
            _, predicted = torch.max(outputs, 1)
            
            # Calculer les métriques
            flood_class = predicted.item()  # 0 ou 1 pour toute l'image
            flood_percentage = flood_probability * 100  # Probabilité d'inondation
            confidence = max(flood_probability, 1 - flood_probability) * 100  # Confiance (0-100%)



        # 6. Créer une visualisation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(results_dir, f"flood_prediction_{timestamp}.png")
        
        plt.figure(figsize=(12, 12))
        
        # Visualiser l'image VV originale
        plt.subplot(2, 1, 1)
        plt.imshow(vv_band, cmap='gray')
        plt.title("Image SAR originale (VV)")
        plt.colorbar()
        plt.axis('off')
        
        # CORRECTION : Visualiser l'image avec une couleur selon la prédiction
        plt.subplot(2, 1, 2)
        # Créer une superposition de couleur selon la prédiction
        if flood_class == 1:  # Si classé comme inondé
            # Afficher l'image avec une teinte rouge pour indiquer l'inondation
            plt.imshow(vv_band, cmap='gray')
            plt.imshow(np.ones_like(vv_band), cmap='autumn', alpha=0.5)
            title = f"ZONE INONDÉE (confiance: {flood_percentage:.1f}%)"
        else:
            # Afficher l'image avec une teinte bleue pour indiquer l'absence d'inondation
            plt.imshow(vv_band, cmap='gray')
            plt.imshow(np.zeros_like(vv_band), cmap='autumn', alpha=0.3)
            title = f"ZONE NON INONDÉE (confiance: {100-flood_percentage:.1f}%)"
        
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        
        # 7. Déterminer le niveau de risque
        if flood_percentage < SEUIL_FAIBLE :
            risk_level = "Faible"
        elif flood_percentage < SEUIL_MOYEN:
            risk_level = "Modéré"
        elif flood_percentage < SEUIL_ELEVÉ:
            risk_level = "Élevé"
        else:
            risk_level = "Très élevé"
        
        # 8. Préparer et retourner les résultats
        result = {
            'success': True,
            'coordinates': {
                'longitude': longitude,
                'latitude': latitude,
                'bbox': bbox_coords
            },
            'prediction': {
                'flood_percentage': round(flood_percentage, 2),
                'confidence': round(confidence, 2),
                'risk_level': risk_level,
                'is_flooded': bool(flood_class)
            },
            'files': {
                'input_image': save_path,
                'output_visualization': output_path
            }
        }
        
        print(f"Analyse terminée! Résultat: {risk_level} risque d'inondation")
        print(f"Pourcentage de zone inondée: {flood_percentage:.2f}%")
        print(f"Confiance: {confidence:.2f}%")
        CACHE.save_prediction(longitude, latitude, size_km, start_date, end_date, result)

        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


""" à ajouter logique lstm """




def predict_flood_from_image(image_path):
    """
    Fonction pour prédire les inondations directement à partir d'une image téléchargée
    Version modifiée pour utiliser rasterio avec les images TIF
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        
    Returns:
        dict: Résultat de la prédiction avec visualisation
    """
    try:
        print(f"Analyse de l'image: {image_path}")
        
        # Créer le dossier de résultats dans le répertoire media pour accessibilité web
        from django.conf import settings
        results_dir = os.path.join(settings.MEDIA_ROOT, "resultats_image")
        os.makedirs(results_dir, exist_ok=True)

        # 1. Charger l'image (détection automatique TIF vs autres formats)
        print("Chargement de l'image...")
        file_extension = os.path.splitext(image_path)[1].lower()
        is_tif = file_extension in ['.tif', '.tiff']
        
        if is_tif:
            # Utiliser rasterio pour les fichiers TIF
            import rasterio
            print("Chargement avec rasterio...")
            
            with rasterio.open(image_path) as src:
                image_data = src.read()
                
                # Extraire le canal d'affichage (même logique que convertir_tif_avec_rasterio)
                if len(image_data.shape) == 3 and image_data.shape[0] > 1:
                    # Format rasterio: (bands, height, width)
                    display_band = image_data[0, :, :]
                    print(f"Utilisation de la première bande, forme: {display_band.shape}")
                    
                    # Pour le preprocessing, créer une image 3D (H, W, C)
                    # Prendre les 2 premières bandes si disponibles (VV, VH)
                    if image_data.shape[0] >= 2:
                        image = np.stack([image_data[0], image_data[1]], axis=-1)  # (H, W, 2)
                    else:
                        # Dupliquer la bande unique
                        image = np.stack([image_data[0], image_data[0]], axis=-1)  # (H, W, 2)
                else:
                    # Une seule bande
                    display_band = image_data.squeeze()
                    print(f"Bande unique, forme après squeeze: {display_band.shape}")
                    
                    # Créer une image 3D en dupliquant la bande
                    image = np.stack([display_band, display_band], axis=-1)  # (H, W, 2)
                
                # Traitement des valeurs NoData
                if hasattr(src, 'nodata') and src.nodata is not None:
                    image = np.where(image == src.nodata, np.nan, image)
                    display_band = np.where(display_band == src.nodata, np.nan, display_band)
                
                # Filtrer les valeurs invalides
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                display_band = np.nan_to_num(display_band, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Utiliser PIL pour les autres formats
            from PIL import Image as PILImage
            print("Chargement avec PIL...")
            
            pil_image = PILImage.open(image_path)
            image = np.array(pil_image)
            
            # Vérifier la dimension de l'image et convertir si nécessaire
            print(f"Forme de l'image originale: {image.shape}")
            
            if len(image.shape) == 2:
                # Image en niveaux de gris (H, W) -> (H, W, 2)
                display_band = image
                image = np.stack([image, image], axis=-1)
                print(f"Image convertie en 2 canaux. Nouvelle forme: {image.shape}")
            else:
                # Image couleur, prendre le premier canal pour l'affichage
                display_band = image[:,:,0] if image.shape[2] > 0 else image.squeeze()
                
                # S'assurer qu'on a exactement 2 canaux pour le modèle
                if image.shape[2] >= 2:
                    image = image[:,:,:2]  # Prendre les 2 premiers canaux
                else:
                    # Dupliquer le canal unique
                    image = np.stack([image.squeeze(), image.squeeze()], axis=-1)
        
        # Vérification finale
        if image is None or np.isnan(image).any() or image.size == 0:
            print("⚠️ Image non disponible ou contenant des valeurs NaN")
            return {
                'success': False, 
                'error': "Image non valide ou contenant des valeurs NaN"
            }
        
        print(f"Image chargée avec succès. Forme finale: {image.shape}")
        print(f"Display band forme: {display_band.shape}")
        
        # 2. Prétraiter l'image pour le modèle
        print("Prétraitement de l'image...")
        image_tensor = preprocess_sar_image_in_memory(
            image, 
            output_dir=os.path.join(settings.MEDIA_ROOT, "flood_prediction/temp"),
            visualize=False
        )
        
        # Vérifier que le tenseur existe
        if image_tensor is None:
            raise ValueError("Le prétraitement de l'image a échoué - image_tensor est None")
        
        print(f"Forme du tenseur d'image: {image_tensor.shape}")
        
        # Ajouter la dimension batch
        image_tensor = image_tensor.unsqueeze(0)
        
        # 3. Charger le modèle
        model_path = MODEL_PATH
        print(f"Chargement du modèle: {model_path}")
        model = load_model(model_path)
    
        # 4. Faire la prédiction
        print("Prédiction en cours...")
        model.eval()
        
        with torch.no_grad():
            outputs = model(image_tensor)
            # Convertir en probabilités avec sigmoid
            flood_probability = torch.sigmoid(outputs[:, 1]).item()

            # Obtenir la classe prédite (0=non inondé, 1=inondé)
            _, predicted = torch.max(outputs, 1)
            
            # Calculer les métriques
            flood_class = predicted.item()
            flood_percentage = flood_probability * 100
            confidence = max(flood_probability, 1 - flood_probability) * 100

        # 5. Créer une visualisation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"flood_prediction_{timestamp}.png"
        output_path = os.path.join(results_dir, output_filename)
        
        plt.figure(figsize=(12, 12))
        
        # Visualiser l'image originale
        plt.subplot(2, 1, 1)
        plt.imshow(display_band, cmap='gray')
        plt.title("Image originale")
        plt.colorbar()
        plt.axis('off')
        
        # Visualiser l'image avec une couleur selon la prédiction
        plt.subplot(2, 1, 2)
        if flood_class == 1:  # Si classé comme inondé
            plt.imshow(display_band, cmap='gray')
            plt.imshow(np.ones_like(display_band), cmap='autumn', alpha=0.5)
            title = f"ZONE INONDÉE (confiance: {flood_percentage:.1f}%)"
        else:
            plt.imshow(display_band, cmap='gray')
            plt.imshow(np.zeros_like(display_band), cmap='autumn', alpha=0.3)
            title = f"ZONE NON INONDÉE (confiance: {100-flood_percentage:.1f}%)"
        
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()  # Fermer la figure pour libérer la mémoire
        
        # 6. Déterminer le niveau de risque
        if flood_percentage < SEUIL_FAIBLE:
            risk_level = "Faible"
        elif flood_percentage < SEUIL_MOYEN:
            risk_level = "Modéré"
        elif flood_percentage < SEUIL_ELEVÉ:
            risk_level = "Élevé"
        else:
            risk_level = "Très élevé"
        
        # 7. Préparer et retourner les résultats
        result = {
            'success': True,
            'prediction': {
                'flood_percentage': round(flood_percentage, 2),
                'confidence': round(confidence, 2),
                'risk_level': risk_level,
                'is_flooded': bool(flood_class)
            },
            'files': {
                'input_image': image_path,
                'output_visualization': output_path,
                'output_filename': output_filename
            }
        }
        
        print(f"Analyse terminée! Résultat: {risk_level} risque d'inondation")
        print(f"Pourcentage de zone inondée: {flood_percentage:.2f}%")
        print(f"Confiance: {confidence:.2f}%")

        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
def convertir_tif_avec_rasterio(tif_path, png_path):
    """
    Convertit une image TIF SAR en PNG en utilisant rasterio avec améliorations visuelles
    
    Args:
        tif_path (str): Chemin de l'image TIF SAR
        png_path (str): Chemin où sauvegarder le PNG
        
    Returns:
        str: Chemin du fichier PNG créé ou None en cas d'erreur
    """
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from matplotlib import colors
    
    try:
        print(f"Chargement avec rasterio: {tif_path}")
        
        # 1. Ouvrir l'image avec rasterio
        with rasterio.open(tif_path) as src:
            # Lire toutes les bandes
            image_data = src.read()
            print(f"Forme des données rasterio: {image_data.shape}")
            print(f"Nombre de bandes: {src.count}")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"Type de données: {image_data.dtype}")
            
            # 2. Extraire le canal d'affichage
            if len(image_data.shape) == 3 and image_data.shape[0] > 1:
                display_band = image_data[0, :, :]
                print(f"Utilisation de la première bande, forme: {display_band.shape}")
            else:
                display_band = image_data.squeeze()
                print(f"Bande unique, forme après squeeze: {display_band.shape}")
        
        # 3. Traitement avancé des valeurs pour améliorer la clarté
        # Remplacer les valeurs NoData
        if hasattr(src, 'nodata') and src.nodata is not None:
            display_band = np.where(display_band == src.nodata, np.nan, display_band)
        
        # Filtrer les valeurs invalides
        display_band = np.nan_to_num(display_band, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Amélioration du contraste et de la clarté
        valid_pixels = display_band[display_band > 0]  # Exclure les pixels à 0 (background)
        
        if len(valid_pixels) > 0:
            # Utiliser les percentiles pour un meilleur contraste
            p1, p99 = np.percentile(valid_pixels, (1, 99))
            p5, p95 = np.percentile(valid_pixels, (5, 95))
            
            print(f"Statistiques des pixels valides:")
            print(f"  Min: {np.min(valid_pixels):.3f}, Max: {np.max(valid_pixels):.3f}")
            print(f"  Percentiles 1-99: {p1:.3f} - {p99:.3f}")
            print(f"  Percentiles 5-95: {p5:.3f} - {p95:.3f}")
            
            # Normalisation pour améliorer le contraste
            display_normalized = np.clip(display_band, p1, p99)
            
            # Application d'une correction gamma pour améliorer la visibilité
            gamma = 0.8  # Valeur < 1 pour éclaircir l'image
            display_gamma = np.power(display_normalized / np.max(display_normalized), gamma)
            
            vmin, vmax = p5, p95  # Utiliser p5-p95 pour l'affichage
        else:
            display_normalized = display_band
            display_gamma = display_band
            vmin, vmax = None, None
        
        # 4. Créer une visualisation améliorée
        plt.figure(figsize=(14, 12), dpi=120)  # Augmenter la résolution
        
        # Image originale avec amélioration du contraste
        plt.subplot(2, 1, 1)
        im1 = plt.imshow(display_normalized, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar(im1, label='Intensité normalisée', fraction=0.046, pad=0.04)
        plt.title("Image SAR originale (VV) - Contraste amélioré", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Image avec correction gamma et superposition
        plt.subplot(2, 1, 2)
        # Utiliser une colormap qui améliore les détails
        im2 = plt.imshow(display_gamma, cmap='viridis', alpha=0.9)
        
        # Superposition avec transparence réduite pour ne pas masquer les détails
        overlay = plt.imshow(np.zeros_like(display_band), cmap='autumn', alpha=0.2)
        
        plt.colorbar(im2, label='Intensité avec correction gamma', fraction=0.046, pad=0.04)
        plt.title("Image SAR - Vue d'analyse améliorée", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Ajouter des informations sur l'amélioration
        info_text = f"Améliorations appliquées:\n• Normalisation percentiles 1-99%\n• Correction gamma γ={gamma}\n• Contraste optimisé"
        plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 5. Sauvegarder avec une meilleure qualité
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Image convertie avec rasterio (version améliorée): {png_path}")
        return png_path
        
    except Exception as e:
        print(f"❌ Erreur avec rasterio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



def convertir_tif_sar_detaille(tif_path, png_path, is_flood=False, flood_percentage=None):
    """
    Version détaillée avec rasterio qui reproduit exactement predict_flood
    
    Args:
        tif_path (str): Chemin de l'image TIF SAR
        png_path (str): Chemin où sauvegarder le PNG
        is_flood (bool): Indique si c'est une zone inondée
        flood_percentage (float): Pourcentage d'inondation
        
    Returns:
        str: Chemin du fichier PNG créé
    """
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    try:
        print(f"Traitement détaillé avec rasterio: {tif_path}")
        
        # 1. Charger avec rasterio
        with rasterio.open(tif_path) as src:
            # Lire la première bande (VV)
            vv_band = src.read(1)
            print(f"Bande VV chargée, forme: {vv_band.shape}, type: {vv_band.dtype}")
            
            # Informations sur l'image
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Bounds: {src.bounds}")
        
        # 2. Traitement des données (comme dans predict_flood)
        # Gérer les valeurs NoData
        vv_band = np.nan_to_num(vv_band, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Créer la visualisation (reproduction exacte de predict_flood)
        plt.figure(figsize=(12, 12))
        
        # Visualiser l'image VV originale
        plt.subplot(2, 1, 1)
        plt.imshow(vv_band, cmap='gray')
        plt.title("Image SAR originale (VV)")
        plt.colorbar()
        plt.axis('off')
        
        # Visualiser l'image avec une couleur selon la prédiction
        plt.subplot(2, 1, 2)
        # Créer une superposition de couleur selon la prédiction
        if is_flood:  # Si classé comme inondé
            # Afficher l'image avec une teinte rouge pour indiquer l'inondation
            plt.imshow(vv_band, cmap='gray')
            plt.imshow(np.ones_like(vv_band), cmap='autumn', alpha=0.5)
            title = f"ZONE INONDÉE"
            if flood_percentage is not None:
                title += f" (confiance: {flood_percentage:.1f}%)"
        else:
            # Afficher l'image avec une teinte bleue pour indiquer l'absence d'inondation
            plt.imshow(vv_band, cmap='gray')
            plt.imshow(np.zeros_like(vv_band), cmap='autumn', alpha=0.3)
            title = f"ZONE NON INONDÉE"
            if flood_percentage is not None:
                title += f" (confiance: {(100-flood_percentage):.1f}%)"
        
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        
        print(f"✅ Visualisation détaillée sauvegardée: {png_path}")
        return png_path
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    


