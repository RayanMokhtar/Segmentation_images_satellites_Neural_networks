#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demonstration.py

Script de démonstration pour la prédiction d'inondations en utilisant trois API :
1. Visual Crossing Weather API pour les données météorologiques
2. Open Topo Data API pour l'élévation
3. OpenEPI Soil Type API pour le type de sol

Utilisation:
    python demonstration.py --lat [LATITUDE] --lon [LONGITUDE] --date [DATE]

Exemple:
    python demonstration.py --lat -19.138148 --lon 146.851468 --date 2019-03-15 --plot
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import requests
import httpx
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Paramètres du modèle LSTM (identiques à l'entraînement) ---
L = 7       # Taille de la fenêtre historique (jours)
H = 3       # Horizon de prédiction (jours après la fenêtre)

# --- Configuration des API ---
WEATHER_API_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
ELEVATION_API_BASE_URL = "https://api.opentopodata.org/v1/srtm30m"
SOIL_API_BASE_URL = "https://api.openepi.io/soil/type"
# Clé API injectée directement
WEATHER_API_KEY = "PUTUNBTHW5R3Q9K2WUW6MPSD6"

# --- Liste des caractéristiques utilisées par le modèle ---
NUMERIC_FEATURES = [
    'latitude_centroid', 'longitude_centroid',
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
    'dew', 'humidity', 'precipprob', 'precipcover',
    'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility',
    'elevation'
]

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Prédiction d'inondations à partir de coordonnées et d'une date")
    parser.add_argument("--lat", type=float, required=True, help="Latitude (ex: -19.138148)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (ex: 146.851468)")
    parser.add_argument("--date", type=str, required=True, help="Date au format YYYY-MM-DD (ex: 2019-03-15)")
    parser.add_argument("--api-key", type=str, required=False, help="Clé API Visual Crossing Weather (optionnelle)")
    parser.add_argument("--model", type=str, default="best_model.keras", help="Chemin vers le modèle LSTM entraîné (.keras ou .h5)")
    parser.add_argument("--output", type=str, default="prediction_results.csv", help="Chemin pour sauvegarder les résultats")
    parser.add_argument("--plot", action="store_true", help="Générer un graphique des prédictions")
    return parser.parse_args()

def get_weather_data(lat, lon, start_date, end_date, api_key=None):
    """
    Récupère les données météorologiques pour une période donnée via l'API Visual Crossing.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Date de début au format YYYY-MM-DD
        end_date (str): Date de fin au format YYYY-MM-DD
        api_key (str, optional): Clé API Visual Crossing. Si None, utilise la clé par défaut.
        
    Returns:
        dict: Données météorologiques ou un message d'erreur
    """
    print(f"Récupération des données météo pour lat={lat}, lon={lon} du {start_date} au {end_date}...")
    
    # Utiliser la clé API fournie ou celle par défaut
    key_to_use = api_key if api_key else WEATHER_API_KEY
    
    url = f"{WEATHER_API_BASE_URL}{lat},{lon}/{start_date}/{end_date}"
    params = {
        "unitGroup": "metric",
        "key": key_to_use,
        "include": "days",
        "contentType": "json"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Données météo récupérées avec succès: {len(data['days'])} jours")
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP: {http_err} - {response.text}")
        return {"error": response.text, "status_code": response.status_code}
    except requests.exceptions.RequestException as req_err:
        print(f"Erreur de requête: {req_err}")
        return {"error": str(req_err)}

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
    url = f"{ELEVATION_API_BASE_URL}?locations={lat},{lon}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
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

def get_soil_type(lat, lon):
    """
    Récupère le type de sol via l'API OpenEPI.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        str: Type de sol ou None en cas d'erreur
    """
    print(f"Récupération du type de sol pour lat={lat}, lon={lon}...")
    
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(SOIL_API_BASE_URL, params={'lat': lat, 'lon': lon})
        
        if response.status_code == 200:
            props = response.json().get('properties', {})
            soil_type = props.get('most_probable_soil_type')
            print(f"Type de sol récupéré: {soil_type}")
            return soil_type
        else:
            print(f"Erreur HTTP {response.status_code} lors de la récupération du type de sol")
            return None
    except Exception as e:
        print(f"Erreur lors de la récupération du type de sol: {e}")
        return None

def prepare_data_for_prediction(weather_data, elevation, soil_type, lat, lon):
    """
    Prépare les données pour la prédiction.
    
    Args:
        weather_data (dict): Données météorologiques
        elevation (float): Élévation
        soil_type (str): Type de sol
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        pd.DataFrame: DataFrame contenant les données préparées
    """
    print("Préparation des données pour la prédiction...")
    
    # Créer un DataFrame à partir des données météorologiques
    rows = []
    for day in weather_data['days']:
        row = {
            'date': day['datetime'],
            'latitude_centroid': lat,
            'longitude_centroid': lon,
            'tempmax': day['tempmax'],
            'tempmin': day['tempmin'],
            'temp': day['temp'],
            'feelslikemax': day['feelslikemax'],
            'feelslikemin': day['feelslikemin'],
            'feelslike': day['feelslike'],
            'dew': day['dew'],
            'humidity': day['humidity'],
            'precipprob': day.get('precipprob', 0),
            'precipcover': day.get('precipcover', 0),
            'windspeed': day['windspeed'],
            'winddir': day['winddir'],
            'pressure': day['pressure'],
            'cloudcover': day['cloudcover'],
            'visibility': day.get('visibility', 10),  # valeur par défaut si non disponible
            'elevation': elevation,
            'soil_type': soil_type,
            'label': 0  # valeur par défaut (pour la phase de construction des données)
        }
        rows.append(row)
      # Créer le DataFrame
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
      # Liste de tous les types de sol possibles dans le dataset d'entraînement
    soil_types = ['Acrisols', 'Arenosols', 'Calcisols', 'Cambisols', 'Ferralsols', 
                 'Fluvisols', 'Gleysols', 'Lixisols', 'Luvisols', 'Nitisols', 
                 'No_information', 'Planosols', 'Regosols', 'Solonchaks', 'Vertisols']
    
    # Ajouter l'encodage one-hot pour le type de sol avec toutes les colonnes attendues
    df = pd.get_dummies(df, columns=['soil_type'], prefix='soil')
    
    # S'assurer que toutes les colonnes soil_* sont présentes
    for soil in soil_types:
        col_name = f'soil_{soil}'
        if col_name not in df.columns:
            df[col_name] = 0
    
    # S'assurer que toutes les colonnes soil_* sont du même type (numérique)
    for col in df.columns:
        if col.startswith('soil_'):
            df[col] = df[col].astype(float)
    
    print(f"Données préparées: {len(df)} jours, {df.columns.size} caractéristiques")
    return df

def standardize_features(df, numeric_features):
    """
    Standardise les caractéristiques numériques.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        numeric_features (list): Liste des caractéristiques numériques à standardiser
        
    Returns:
        pd.DataFrame: DataFrame avec les caractéristiques standardisées
    """
    print("Standardisation des caractéristiques numériques...")
    
    # Filtrer les colonnes numériques existantes
    existing_numeric_features = [f for f in numeric_features if f in df.columns]
    
    # Remplacer les valeurs NaN par la moyenne de chaque colonne
    for col in existing_numeric_features:
        if df[col].isna().any():
            print(f"Remplacement des valeurs NaN dans la colonne '{col}' par la moyenne")
            df[col].fillna(df[col].mean() if not pd.isna(df[col].mean()) else 0, inplace=True)
    
    # Appliquer la standardisation
    scaler = StandardScaler()
    df[existing_numeric_features] = scaler.fit_transform(df[existing_numeric_features])
    
    return df

def prepare_sequence_for_lstm(df, sequence_length=L):
    """
    Prépare la séquence pour le modèle LSTM.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        sequence_length (int): Longueur de la séquence (fenêtre historique)
        
    Returns:
        tuple: (X, dates) où X est la séquence formatée pour le LSTM et dates sont les dates correspondantes
    """
    print(f"Préparation de la séquence pour le modèle LSTM (longueur={sequence_length})...")
    
    # S'assurer qu'il y a suffisamment de données
    if len(df) < sequence_length:
        print(f"ERREUR: Pas assez de données pour former une séquence (besoin de {sequence_length} jours, mais n'a que {len(df)} jours)")
        return None, None
    
    # Utiliser les L derniers jours comme fenêtre historique
    seq_data = df.iloc[-sequence_length:]
    feature_cols = [col for col in seq_data.columns if col not in ['date', 'label']]
    
    # Extraire les caractéristiques
    X = seq_data[feature_cols].values
    
    # Convertir en format 3D attendu par le LSTM: [batch_size, sequence_length, n_features]
    X = X.reshape(1, sequence_length, -1).astype(np.float32)
    
    # Dates pour l'affichage des résultats
    dates = seq_data['date'].tolist()
    
    print(f"Séquence préparée avec forme: {X.shape}")
    return X, dates

def load_lstm_model(model_path):
    """
    Charge le modèle LSTM entraîné.
    
    Args:
        model_path (str): Chemin vers le fichier du modèle
        
    Returns:
        tf.keras.Model: Modèle LSTM chargé
    """
    print(f"Chargement du modèle LSTM depuis {model_path}...")
    
    try:
        model = load_model(model_path)
        print(f"Modèle chargé avec succès. Forme d'entrée attendue: {model.input_shape}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        
        # Tenter de trouver d'autres modèles
        model_files = [f for f in os.listdir() if f.endswith(('.keras', '.h5'))]
        if model_files:
            print(f"Tentative avec d'autres modèles disponibles: {model_files}")
            try:
                model = load_model(model_files[0])
                print(f"Modèle alternatif chargé avec succès: {model_files[0]}")
                return model
            except Exception as e2:
                print(f"Échec du chargement du modèle alternatif: {e2}")
        
        print("Aucun modèle valide n'a pu être chargé.")
        return None

def predict_flood_risk(model, X, prediction_dates, input_dates, horizon=H):
    """
    Prédit le risque d'inondation pour les prochains jours.
    
    Args:
        model (tf.keras.Model): Modèle LSTM chargé
        X (numpy.ndarray): Données d'entrée pour le modèle
        prediction_dates (list): Liste des dates pour les prédictions
        input_dates (list): Liste des dates d'entrée
        horizon (int): Horizon de prédiction (jours après la fenêtre)
        
    Returns:
        pd.DataFrame: DataFrame avec les prédictions
    """
    print(f"Prédiction du risque d'inondation pour un horizon de {horizon} jours...")
    
    # Vérifier les dimensions de X
    expected_features = model.input_shape[-1]
    if X.shape[2] != expected_features:
        print(f"ATTENTION: Le nombre de caractéristiques ({X.shape[2]}) ne correspond pas à ce qu'attend le modèle ({expected_features})")
        print("Adaptation des dimensions...")
        
        # Si trop de caractéristiques, tronquer
        if X.shape[2] > expected_features:
            X = X[:, :, :expected_features]
            print(f"Caractéristiques tronquées à {X.shape[2]}")
        # Si pas assez de caractéristiques, compléter avec des zéros
        elif X.shape[2] < expected_features:
            padding = np.zeros((X.shape[0], X.shape[1], expected_features - X.shape[2]))
            X = np.concatenate([X, padding], axis=2)
            print(f"Caractéristiques complétées avec des zéros à {X.shape[2]}")
    
    # Faire la prédiction
    try:
        y_prob = model.predict(X, verbose=0)[0, 0]
        y_pred = 1 if y_prob >= 0.5 else 0
        
        # Calculer la date de prédiction (H jours après la dernière date d'entrée)
        last_date = pd.to_datetime(input_dates[-1])
        prediction_date = last_date + timedelta(days=horizon)
        
        # Créer un DataFrame avec les résultats
        results = pd.DataFrame({
            'date': [prediction_date],
            'probabilite': [float(y_prob)],
            'prediction': [int(y_pred)]
        })
        
        print("\n=== Résultat de la prédiction ===")
        print(f"Date de prédiction: {prediction_date.strftime('%Y-%m-%d')}")
        print(f"Probabilité d'inondation: {y_prob:.3f}")
        print(f"Prédiction: {'INONDATION' if y_pred == 1 else 'PAS D INONDATION'}")
        
        return results
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None

def plot_prediction_results(input_data, prediction_result, horizon=H):
    """
    Génère un graphique des données d'entrée et de la prédiction.
    
    Args:
        input_data (pd.DataFrame): Données d'entrée
        prediction_result (pd.DataFrame): Résultat de la prédiction
        horizon (int): Horizon de prédiction
    """
    print("Génération du graphique des résultats...")
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Sous-graphique pour les données météo
        plt.subplot(3, 1, 1)
        plt.plot(input_data['date'], input_data['temp'], 'b-', label='Température')
        plt.plot(input_data['date'], input_data['humidity'], 'g-', label='Humidité')
        plt.title('Données météorologiques (entrée du modèle)')
        plt.ylabel('Valeur')
        plt.legend()
        plt.grid(True)
        
        # Sous-graphique pour les précipitations
        plt.subplot(3, 1, 2)
        plt.bar(input_data['date'], input_data['precipprob'], width=0.8, alpha=0.6, label='Probabilité précip.')
        plt.bar(input_data['date'], input_data['precipcover'], width=0.4, alpha=0.6, label='Couverture précip.')
        plt.title('Précipitations (entrée du modèle)')
        plt.ylabel('Valeur (%)')
        plt.legend()
        plt.grid(True)
        
        # Sous-graphique pour la prédiction
        plt.subplot(3, 1, 3)
        
        # Construire l'axe des dates pour la prédiction
        last_date = input_data['date'].iloc[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        # Marquer la zone de prédiction
        plt.axvspan(last_date, prediction_dates[-1], alpha=0.2, color='yellow', label='Zone de prédiction')
          # Tracer la probabilité d'inondation pour la date prédite
        if not prediction_result.empty:
            pred_date = prediction_result['date'].iloc[0]
            pred_prob = prediction_result['probabilite'].iloc[0]
            plt.scatter([pred_date], [pred_prob], s=100, c='red' if pred_prob >= 0.5 else 'green', 
                      marker='o', label=f"Prédiction: {'INONDATION' if pred_prob >= 0.5 else 'PAS D INONDATION'}")
            
            # Ajouter une ligne de seuil
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Seuil (0.5)')
        
        plt.title('Prédiction du risque d\'inondation')
        plt.ylabel('Probabilité')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # Formatage des dates sur l'axe x
        plt.gcf().autofmt_xdate()
        
        # Ajouter un titre global
        plt.suptitle('Analyse et prédiction du risque d\'inondation', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder et afficher
        plt.savefig('prediction_graph.png')
        print("Graphique sauvegardé sous 'prediction_graph.png'")
        plt.show()
    except Exception as e:
        print(f"Erreur lors de la génération du graphique: {e}")

def main():
    """Fonction principale"""
    # Analyser les arguments
    args = parse_arguments()
    
    # Convertir la date en objet datetime
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Format de date invalide: {args.date}. Utilisez le format YYYY-MM-DD.")
        return 1
    
    # Calculer les dates pour la fenêtre historique (L jours jusqu'à la date cible)
    start_date = (target_date - timedelta(days=L-1)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")
      # 1. Récupérer les données météorologiques
    weather_data = get_weather_data(args.lat, args.lon, start_date, end_date, args.api_key)
    if "error" in weather_data:
        print("Échec de la récupération des données météorologiques. Arrêt du programme.")
        return 1
    
    # 2. Récupérer l'élévation
    elevation = get_elevation_data(args.lat, args.lon)
    if elevation is None:
        print("Avertissement: Impossible de récupérer l'élévation. Utilisation de 0 comme valeur par défaut.")
        elevation = 0
    
    # 3. Récupérer le type de sol
    soil_type = get_soil_type(args.lat, args.lon)
    if soil_type is None:
        print("Avertissement: Impossible de récupérer le type de sol. Utilisation de 'Unknown' comme valeur par défaut.")
        soil_type = "Unknown"
    
    # 4. Préparer les données pour la prédiction
    df = prepare_data_for_prediction(weather_data, elevation, soil_type, args.lat, args.lon)
    
    # 5. Standardiser les caractéristiques numériques
    df = standardize_features(df, NUMERIC_FEATURES)
      # 6. Préparer la séquence pour le LSTM
    X, input_dates = prepare_sequence_for_lstm(df)
    if X is None:
        print("Échec de la préparation de la séquence LSTM. Arrêt du programme.")
        return 1
    
    # Afficher la structure des données d'entrée pour vérification
    print("\n=== Structure des données d'entrée du modèle ===")
    feature_cols = [col for col in df.columns if col not in ['date', 'label']]
    print(f"Nombre total de caractéristiques: {len(feature_cols)}")
    print("Catégories de caractéristiques:")
    print(f"- Météo et localisation: {[f for f in feature_cols if not f.startswith('soil_')]}")
    print(f"- Types de sol: {[f for f in feature_cols if f.startswith('soil_')]}")    
    print("\nExemple d'une ligne de données (premier jour de la séquence):")
    sample_day = df.iloc[-X.shape[1]].copy()  # Premier jour de la séquence
    for col in feature_cols:
        if not pd.isna(sample_day[col]):
            # S'assurer que toutes les valeurs sont affichées de manière cohérente (comme des nombres)
            if isinstance(sample_day[col], bool):
                print(f"- {col}: {int(sample_day[col])}")
            elif isinstance(sample_day[col], float):
                print(f"- {col}: {sample_day[col]:.4f}")
            else:
                print(f"- {col}: {sample_day[col]}")
    
    # Afficher également les valeurs brutes (non standardisées) des coordonnées
    print("\nCoordonnées brutes (avant standardisation):")
    print(f"- Latitude: {args.lat}")
    print(f"- Longitude: {args.lon}")
    
    # 7. Charger le modèle LSTM
    model = load_lstm_model(args.model)
    if model is None:
        print("Échec du chargement du modèle LSTM. Arrêt du programme.")
        return 1
    
    # 8. Faire la prédiction
    prediction_dates = [pd.to_datetime(end_date) + timedelta(days=i+1) for i in range(H)]
    prediction_result = predict_flood_risk(model, X, prediction_dates, input_dates)
    if prediction_result is None:
        print("Échec de la prédiction. Arrêt du programme.")
        return 1
    
    # 9. Sauvegarder les résultats
    prediction_result.to_csv(args.output, index=False)
    print(f"Résultats de prédiction sauvegardés dans {args.output}")
    
    # 10. Tracer le graphique si demandé
    if args.plot:
        plot_prediction_results(df, prediction_result)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
