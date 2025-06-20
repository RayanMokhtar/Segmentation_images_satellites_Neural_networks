#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predire_regions.py

Script qui récupère toutes les régions auxquelles des utilisateurs sont abonnés,
effectue des prédictions CNN-LSTM pour les 3 prochains jours et enregistre 
les résultats dans la table HistoriquePrediction.
"""

import os
import sys
import django
import logging
import traceback
from datetime import datetime, timedelta

# Configuration du logging plus détaillé
logging.basicConfig(
    filename="predictions_regions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ajouter également les logs à la console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

print("=== DÉMARRAGE DU SCRIPT DE PRÉDICTION RÉGIONS ===")
print(f"Date actuelle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Répertoire de travail: {os.getcwd()}")

# Configuration de l'environnement Django
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'floodAI.settings')

print("Initialisation de Django...")
django.setup()
print("Django initialisé avec succès!")

# Après avoir configuré l'environnement Django, on peut importer les modèles
from flood_prediction.models import AbonnementVille, HistoriquePrediction
from flood_prediction.views import predict_flood  # Fonction pour la prédiction CNN
from flood_prediction.lstm_prediction import predict_flood_lstm  # Fonction pour la prédiction LSTM
from django.db import IntegrityError
from django.db.models import Count
import json

def get_regions_with_subscribers():
    """
    Récupère toutes les régions uniques auxquelles au moins un utilisateur est abonné
    """
    print("\n--- RÉCUPÉRATION DES RÉGIONS AVEC ABONNÉS ---")
    try:
        # Récupère toutes les régions uniques avec le nombre d'abonnés
        regions = (AbonnementVille.objects
                .values('ville', 'latitude', 'longitude')
                .annotate(num_subscribers=Count('user'))
                .filter(num_subscribers__gt=0)
                .order_by('ville'))
        
        regions_list = list(regions)
        print(f"Nombre de régions trouvées: {len(regions_list)}")
        
        # Afficher les premières régions pour vérification
        if regions_list:
            print("Premières régions trouvées:")
            for i, r in enumerate(regions_list[:5]):
                print(f"  {i+1}. {r['ville']} ({r['latitude']}, {r['longitude']}) - {r['num_subscribers']} abonnés")
            if len(regions_list) > 5:
                print(f"  ... et {len(regions_list) - 5} autres régions")
        else:
            print("ATTENTION: Aucune région avec abonnés n'a été trouvée!")
            
        return regions_list
    except Exception as e:
        print(f"ERREUR lors de la récupération des régions: {e}")
        logging.error(f"Erreur lors de la récupération des régions: {e}")
        logging.error(traceback.format_exc())
        return []

def predict_region_flooding(region, horizon=3):
    """
    Effectue des prédictions d'inondation pour une région donnée sur plusieurs jours
    en utilisant les modèles CNN et LSTM.
    
    Args:
        region: Dictionnaire contenant les informations sur la région
        horizon: Nombre de jours pour lesquels effectuer des prédictions
        
    Returns:
        Une liste de prédictions pour les jours demandés
    """
    print(f"\n--- PRÉDICTION POUR LA RÉGION: {region['ville']} ---")
    predictions = []
    lat = region['latitude']
    lng = region['longitude']
    
    try:
        print(f"Coordonnées: {lat}, {lng}")
        today = datetime.now().date()
        
        # 1. Prédiction CNN pour aujourd'hui (Jour J)
        print("Exécution de la prédiction CNN (jour J)...")
        cnn_result = predict_flood(
            longitude=lng,
            latitude=lat,
            size_km=5,
            start_date=(today - timedelta(days=10)).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d')
        )
        
        if not cnn_result.get('success', False):
            print(f"ERREUR lors de la prédiction CNN: {cnn_result.get('error', 'Erreur inconnue')}")
            logging.error(f"Erreur lors de la prédiction CNN pour {region['ville']}: {cnn_result.get('error', 'Erreur inconnue')}")
        else:
            print("Prédiction CNN réussie!")
            
            # Extraire les résultats CNN
            cnn_proba = float(cnn_result.get('flood_probability', 0)) * 100
            cnn_risk = cnn_result.get('risk_level', 'faible')
            cnn_is_flooded = cnn_risk != 'faible'
            
            print(f"CNN - Probabilité: {cnn_proba:.2f}%, Risque: {cnn_risk}, Inondation: {'Oui' if cnn_is_flooded else 'Non'}")
            
            # Ajouter la prédiction pour aujourd'hui (J)
            predictions.append({
                'region': region['ville'],
                'lat': lat,
                'lng': lng,
                'date': today,
                'probabilite': round(cnn_proba, 2),
                'niveau_risque': cnn_risk,
                'inondation_prevue': cnn_is_flooded
            })
        
        # 2. Prédiction LSTM pour les jours suivants (J+1, J+2)
        print(f"Exécution de la prédiction LSTM pour les {horizon-1} prochains jours...")
        lstm_result = predict_flood_lstm(
            lat=lat,
            lon=lng,
            target_date=today.strftime('%Y-%m-%d'),
            horizon=horizon-1,  # Déjà traité J avec CNN
            use_cnn_labels=True  # Utiliser les labels CNN pour améliorer les prédictions
        )
        
        if lstm_result.get('success', False):
            print("Prédiction LSTM réussie!")
            forecast_days = lstm_result.get('forecast_days', [])
            
            # Traiter chaque jour de prévision (J+1, J+2, etc.)
            for i, day_forecast in enumerate(forecast_days, 1):
                # Extraire les données du jour
                day_date = today + timedelta(days=i)
                proba = float(day_forecast.get('probability', 0))
                is_flooded = day_forecast.get('is_flooded', False)
                risk_level = day_forecast.get('risk_level', 'faible')
                
                print(f"LSTM J+{i} ({day_date}) - Probabilité: {proba:.2f}%, "
                      f"Risque: {risk_level}, Inondation: {'Oui' if is_flooded else 'Non'}")
                
                # Ajouter la prédiction
                predictions.append({
                    'region': region['ville'],
                    'lat': lat,
                    'lng': lng,
                    'date': day_date,
                    'probabilite': round(proba, 2),
                    'niveau_risque': risk_level,
                    'inondation_prevue': is_flooded
                })
        else:
            print(f"ERREUR lors de la prédiction LSTM: {lstm_result.get('error', 'Erreur inconnue')}")
            logging.error(f"Erreur lors de la prédiction LSTM pour {region['ville']}: {lstm_result.get('error', 'Erreur inconnue')}")
            
    except Exception as e:
        print(f"ERREUR CRITIQUE lors de la prédiction pour {region['ville']}: {e}")
        logging.error(f"Erreur lors de la prédiction pour {region['ville']}: {e}")
        logging.error(traceback.format_exc())
    
    print(f"Nombre de prédictions générées pour {region['ville']}: {len(predictions)}")
    return predictions

def save_predictions_to_db(predictions):
    """
    Enregistre les prédictions dans la base de données
    
    Args:
        predictions: Liste de dictionnaires contenant les prédictions
        
    Returns:
        Le nombre de prédictions enregistrées avec succès
    """
    print(f"\n--- ENREGISTREMENT DE {len(predictions)} PRÉDICTIONS DANS LA BASE DE DONNÉES ---")
    saved_count = 0
    skipped_count = 0
    
    for i, pred in enumerate(predictions, 1):
        try:
            # Création ou mise à jour de l'entrée
            hist_pred, created = HistoriquePrediction.objects.update_or_create(
                region=pred['region'],
                date_prediction=pred['date'],
                defaults={
                    'latitude': pred['lat'],
                    'longitude': pred['lng'],
                    'probabilite': pred['probabilite'],
                    'niveau_risque': pred['niveau_risque'],
                    'inondation_prevue': pred['inondation_prevue'],
                    'modele_utilise': 'CNN-LSTM'
                }
            )
            
            status = "Créée" if created else "Mise à jour"
            print(f"Prédiction {i}/{len(predictions)}: {status} - {pred['region']} - {pred['date']} - {pred['probabilite']}% - {pred['niveau_risque']}")
            
            if created:
                logging.info(f"Nouvelle prédiction créée: {hist_pred}")
            else:
                logging.info(f"Prédiction mise à jour: {hist_pred}")
                
            saved_count += 1
            
        except IntegrityError:
            print(f"ATTENTION: Prédiction déjà existante pour {pred['region']} à la date {pred['date']}")
            logging.warning(f"Prédiction déjà existante pour {pred['region']} à la date {pred['date']}")
            skipped_count += 1
        except Exception as e:
            print(f"ERREUR lors de l'enregistrement de la prédiction {i}/{len(predictions)}: {e}")
            logging.error(f"Erreur lors de l'enregistrement de la prédiction: {e}")
            logging.error(traceback.format_exc())
    
    print(f"\nRésultat de l'enregistrement: {saved_count} prédictions sauvegardées, {skipped_count} ignorées")
    return saved_count, skipped_count

def main():
    """Fonction principale du script"""
    start_time = datetime.now()
    print("\n=== DÉMARRAGE DU TRAITEMENT PRINCIPAL ===")
    print(f"Heure de début: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Démarrage du script de prédiction automatique pour les régions avec abonnés")
    
    try:
        # Récupérer toutes les régions avec des abonnés
        regions = get_regions_with_subscribers()
        
        if not regions:
            print("ERREUR: Aucune région trouvée avec des abonnés. Fin du script.")
            logging.warning("Aucune région trouvée avec des abonnés.")
            return
        
        # Liste pour stocker toutes les prédictions
        all_predictions = []
        
        # Pour chaque région, effectuer des prédictions
        for i, region in enumerate(regions, 1):
            print(f"\n=== TRAITEMENT DE LA RÉGION {i}/{len(regions)} : {region['ville']} ===")
            logging.info(f"Traitement de la région {i}/{len(regions)}: {region['ville']}")
            
            region_predictions = predict_region_flooding(region)
            
            if region_predictions:
                all_predictions.extend(region_predictions)
                print(f"Prédictions pour {region['ville']} ajoutées au lot (total: {len(all_predictions)})")
            else:
                print(f"Aucune prédiction générée pour {region['ville']}")
        
        # Enregistrer toutes les prédictions dans la base de données
        if all_predictions:
            print(f"\n=== ENREGISTREMENT DES PRÉDICTIONS ===")
            print(f"Nombre total de prédictions à enregistrer: {len(all_predictions)}")
            saved_count, skipped_count = save_predictions_to_db(all_predictions)
            
            print(f"\n=== RÉSUMÉ FINAL ===")
            print(f"Régions traitées: {len(regions)}")
            print(f"Prédictions générées: {len(all_predictions)}")
            print(f"Prédictions enregistrées: {saved_count}")
            print(f"Prédictions ignorées: {skipped_count}")
            
            logging.info(f"Script terminé - {saved_count} prédictions enregistrées, {skipped_count} prédictions ignorées")
        else:
            print("ATTENTION: Aucune prédiction n'a pu être générée!")
            logging.warning("Aucune prédiction n'a pu être générée!")
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nTemps d'exécution: {duration}")
        logging.info(f"Temps d'exécution: {duration}")
        print("\n=== SCRIPT TERMINÉ AVEC SUCCÈS ===")
        
    except Exception as e:
        print(f"\n!!! ERREUR CRITIQUE LORS DE L'EXÉCUTION DU SCRIPT !!!")
        print(f"Nature de l'erreur: {e}")
        print("Trace complète de l'erreur:")
        traceback.print_exc()
        
        logging.error(f"Erreur générale lors de l'exécution du script: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
