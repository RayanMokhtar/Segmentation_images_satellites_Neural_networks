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

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_model.pth')


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