#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demonstration_incremental.py

Version modifiée du script de démonstration pour la prédiction d'inondations
permettant de réaliser des prédictions incrémentales pour J+1, J+2 et J+3
en utilisant une approche où chaque prédiction est utilisée comme entrée
pour la prédiction suivante.

Utilisation:
    python demonstration_incremental.py --lat [LATITUDE] --lon [LONGITUDE] --date [DATE]

Exemple:
    python demonstration_incremental.py --lat -19.138148 --lon 146.851468 --date 2019-03-15 --plot
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
import copy

# --- Paramètres du modèle LSTM modifiés pour l'approche incrémentale ---
L = 7       # Taille de la fenêtre historique (jours)
H = 1       # Horizon de prédiction réduit à 1 jour pour l'approche incrémentale

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
    parser = argparse.ArgumentParser(description="Prédiction incrémentale d'inondations à partir de coordonnées et d'une date")
    parser.add_argument("--lat", type=float, required=True, help="Latitude (ex: -19.138148)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (ex: 146.851468)")
    parser.add_argument("--date", type=str, required=True, help="Date au format YYYY-MM-DD (ex: 2019-03-15)")
    parser.add_argument("--api-key", type=str, required=False, help="Clé API Visual Crossing Weather (optionnelle)")
    parser.add_argument("--model", type=str, default="best_model.keras", help="Chemin vers le modèle LSTM entraîné (.keras ou .h5)")
    parser.add_argument("--output", type=str, default="prediction_results_incremental.csv", help="Chemin pour sauvegarder les résultats")
    parser.add_argument("--plot", action="store_true", help="Générer un graphique des prédictions")
    parser.add_argument("--horizon", type=int, default=3, help="Nombre de jours à prédire (par défaut: 3)")
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
    return X, dates, feature_cols

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

def predict_flood_risk_incremental(model, X, weather_api_data, df, feature_cols, input_dates, 
                                 lat, lon, elevation, soil_type, total_horizon=3):
    """
    Prédit le risque d'inondation de manière incrémentale pour plusieurs jours à venir.
    
    Args:
        model (tf.keras.Model): Modèle LSTM chargé
        X (numpy.ndarray): Données d'entrée pour le modèle (séquence initiale)
        weather_api_data (dict): Données météo obtenues de l'API
        df (pd.DataFrame): DataFrame initial avec les données d'entrée
        feature_cols (list): Liste des colonnes de caractéristiques
        input_dates (list): Liste des dates d'entrée
        lat, lon (float): Coordonnées géographiques
        elevation (float): Élévation du terrain
        soil_type (str): Type de sol
        total_horizon (int): Nombre total de jours à prédire
        
    Returns:
        pd.DataFrame: DataFrame avec les prédictions incrémentales
    """
    print(f"Prédiction incrémentale du risque d'inondation pour les {total_horizon} prochains jours...")
    
    # Créer un DataFrame pour stocker les résultats
    results = pd.DataFrame(columns=['date', 'probabilite', 'prediction'])
    
    # Faire une copie du DataFrame d'entrée et de la séquence X
    current_df = df.copy()
    current_X = X.copy()
    current_dates = input_dates.copy()
    
    # Récupérer les données météo futures si disponibles dans l'API
    future_weather_days = {}
    last_date = pd.to_datetime(input_dates[-1])
    
    for i in range(1, total_horizon + 1):
        future_date = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for day in weather_api_data.get('days', []):
            if day['datetime'] == future_date:
                future_weather_days[future_date] = day
                break
    
    # Pour chaque jour à prédire
    for day in range(1, total_horizon + 1):
        # Vérifier les dimensions de X
        expected_features = model.input_shape[-1]
        if current_X.shape[2] != expected_features:
            print(f"ATTENTION: Le nombre de caractéristiques ({current_X.shape[2]}) ne correspond pas à ce qu'attend le modèle ({expected_features})")
            print("Adaptation des dimensions...")
            
            # Si trop de caractéristiques, tronquer
            if current_X.shape[2] > expected_features:
                current_X = current_X[:, :, :expected_features]
                print(f"Caractéristiques tronquées à {current_X.shape[2]}")
            # Si pas assez de caractéristiques, compléter avec des zéros
            elif current_X.shape[2] < expected_features:
                padding = np.zeros((current_X.shape[0], current_X.shape[1], expected_features - current_X.shape[2]))
                current_X = np.concatenate([current_X, padding], axis=2)
                print(f"Caractéristiques complétées avec des zéros à {current_X.shape[2]}")
        
        # Faire la prédiction pour le jour actuel
        try:
            y_prob = model.predict(current_X, verbose=0)[0, 0]
            y_pred = 1 if y_prob >= 0.5 else 0
            
            # Calculer la date de prédiction (1 jour après la dernière date de la séquence)
            last_date = pd.to_datetime(current_dates[-1])
            prediction_date = last_date + timedelta(days=1)
            
            # Ajouter le résultat au DataFrame
            new_result = pd.DataFrame({
                'date': [prediction_date],
                'probabilite': [float(y_prob)],
                'prediction': [int(y_pred)]
            })
            results = pd.concat([results, new_result], ignore_index=True)
            
            print(f"\n=== Prédiction pour {prediction_date.strftime('%Y-%m-%d')} (J+{day}) ===")
            print(f"Probabilité d'inondation: {y_prob:.3f}")
            print(f"Prédiction: {'INONDATION' if y_pred == 1 else 'PAS D INONDATION'}")
            
            # Si ce n'est pas le dernier jour à prédire, mettre à jour la séquence
            if day < total_horizon:
                # Créer une nouvelle ligne pour le jour prédit
                new_row = {}
                
                # Si nous avons des données météo futures disponibles, les utiliser
                prediction_date_str = prediction_date.strftime("%Y-%m-%d")
                if prediction_date_str in future_weather_days:
                    future_day = future_weather_days[prediction_date_str]
                    print(f"Utilisation des données météo API pour {prediction_date_str}")
                    
                    # Remplir avec les données météo de l'API
                    new_row = {
                        'date': prediction_date,
                        'latitude_centroid': lat,
                        'longitude_centroid': lon,
                        'tempmax': future_day['tempmax'],
                        'tempmin': future_day['tempmin'],
                        'temp': future_day['temp'],
                        'feelslikemax': future_day['feelslikemax'],
                        'feelslikemin': future_day['feelslikemin'],
                        'feelslike': future_day['feelslike'],
                        'dew': future_day['dew'],
                        'humidity': future_day['humidity'],
                        'precipprob': future_day.get('precipprob', 0),
                        'precipcover': future_day.get('precipcover', 0),
                        'windspeed': future_day['windspeed'],
                        'winddir': future_day['winddir'],
                        'pressure': future_day['pressure'],
                        'cloudcover': future_day['cloudcover'],
                        'visibility': future_day.get('visibility', 10),
                        'elevation': elevation,
                        'label': y_pred  # Utiliser la prédiction comme label
                    }
                else:
                    # Sinon, utiliser les données du dernier jour comme approximation
                    print(f"Données météo API non disponibles pour {prediction_date_str}, utilisation du dernier jour")
                    last_row = current_df.iloc[-1].copy()
                    
                    # Copier toutes les valeurs sauf date et label
                    for col in current_df.columns:
                        if col not in ['date', 'label']:
                            new_row[col] = last_row[col]
                    
                    # Mettre à jour date et label
                    new_row['date'] = prediction_date
                    new_row['label'] = y_pred
                
                # Ajouter la nouvelle ligne au DataFrame
                new_df_row = pd.DataFrame([new_row])
                
                # Traiter les colonnes one-hot pour le type de sol
                if 'soil_type' in new_row:
                    # Si soil_type est encore présent, faire l'encodage one-hot
                    soil_val = new_row['soil_type']
                    new_df_row = pd.get_dummies(new_df_row, columns=['soil_type'], prefix='soil')
                    
                    # S'assurer que toutes les colonnes soil_* sont présentes
                    for col in current_df.columns:
                        if col.startswith('soil_') and col not in new_df_row.columns:
                            new_df_row[col] = 0
                    
                    # Si la colonne pour ce type de sol existe, la mettre à 1
                    soil_col = f'soil_{soil_val}'
                    if soil_col in new_df_row.columns:
                        new_df_row[soil_col] = 1
                else:
                    # Copier les valeurs one-hot du sol du dernier jour
                    for col in current_df.columns:
                        if col.startswith('soil_') and col not in new_df_row.columns:
                            new_df_row[col] = current_df.iloc[-1][col]
                
                # Standardiser les nouvelles données (uniquement les caractéristiques numériques)
                numeric_cols = [c for c in NUMERIC_FEATURES if c in new_df_row.columns]
                if numeric_cols:
                    # Créer un scaler temporaire ajusté aux données actuelles
                    temp_scaler = StandardScaler()
                    temp_scaler.fit(current_df[numeric_cols])
                    new_df_row[numeric_cols] = temp_scaler.transform(new_df_row[numeric_cols])
                
                # Ajouter la nouvelle ligne au DataFrame
                current_df = pd.concat([current_df, new_df_row], ignore_index=True)
                
                # Mettre à jour la séquence en supprimant le premier jour et en ajoutant le nouveau
                current_df_sorted = current_df.sort_values('date').reset_index(drop=True)
                seq_data = current_df_sorted.iloc[-L:]
                
                # Mettre à jour les dates
                current_dates = seq_data['date'].tolist()
                
                # Extraire les caractéristiques
                feature_cols = [col for col in seq_data.columns if col not in ['date', 'label']]
                X_new = seq_data[feature_cols].values
                
                # Mettre à jour la séquence
                current_X = X_new.reshape(1, L, -1).astype(np.float32)
                
                print(f"Séquence mise à jour avec le nouveau jour. Nouvelle fenêtre: {current_dates[0].strftime('%Y-%m-%d')} à {current_dates[-1].strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"Erreur lors de la prédiction pour le jour {day}: {e}")
            break
    
    return results

def plot_prediction_results_incremental(input_data, prediction_results):
    """
    Génère un graphique des données d'entrée et des prédictions incrémentales.
    
    Args:
        input_data (pd.DataFrame): Données d'entrée
        prediction_results (pd.DataFrame): Résultats des prédictions
    """
    print("Génération du graphique des résultats incrémentaux...")
    
    try:
        plt.figure(figsize=(14, 10))
        
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
        
        # Sous-graphique pour les prédictions
        plt.subplot(3, 1, 3)
        
        # Marquer la dernière date d'entrée
        last_input_date = input_data['date'].iloc[-1]
        
        # Tracer les probabilités d'inondation pour les dates prédites
        if not prediction_results.empty:
            # Créer une plage de dates pour l'affichage
            all_dates = pd.date_range(start=input_data['date'].iloc[0], end=prediction_results['date'].iloc[-1])
            
            # Marquer la zone de prédiction
            plt.axvspan(last_input_date, prediction_results['date'].iloc[-1], alpha=0.2, color='yellow', label='Zone de prédiction')
            
            # Tracer les points de prédiction
            for i, row in prediction_results.iterrows():
                color = 'red' if row['prediction'] == 1 else 'green'
                plt.scatter([row['date']], [row['probabilite']], s=100, c=color, marker='o')
            
            # Relier les points de prédiction
            plt.plot(prediction_results['date'], prediction_results['probabilite'], 'b--', alpha=0.7)            # Ajouter des annotations pour chaque prédiction
            for i, row in prediction_results.iterrows():
                day_num = i + 1
                term = 'INONDATION' if row['prediction'] == 1 else "PAS D'INONDATION"
                label = f"J+{day_num}: {term}"
                plt.annotate(label, 
                             xy=(row['date'], row['probabilite']),
                             xytext=(10, 10 if i % 2 == 0 else -25),
                             textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
            
            # Ajouter une ligne de seuil
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Seuil (0.5)')
        
        plt.title('Prédictions incrémentales du risque d\'inondation')
        plt.ylabel('Probabilité')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # Formatage des dates sur l'axe x
        plt.gcf().autofmt_xdate()
        
        # Ajouter un titre global
        plt.suptitle('Analyse et prédiction incrémentale du risque d\'inondation', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder et afficher
        plt.savefig('prediction_incremental_graph.png')
        print("Graphique sauvegardé sous 'prediction_incremental_graph.png'")
        plt.show()
    except Exception as e:
        print(f"Erreur lors de la génération du graphique: {e}")

def main():
    """Fonction principale"""
    # Analyser les arguments
    args = parse_arguments()
    
    # Nombre de jours à prédire
    total_horizon = args.horizon
    
    # Convertir la date en objet datetime
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Format de date invalide: {args.date}. Utilisez le format YYYY-MM-DD.")
        return 1
    
    # Calculer les dates pour la fenêtre historique (L jours jusqu'à la date cible)
    start_date = (target_date - timedelta(days=L-1)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")
    
    # 1. Récupérer les données météorologiques (incluant quelques jours futurs si possible)
    future_end_date = (target_date + timedelta(days=total_horizon)).strftime("%Y-%m-%d")
    weather_data = get_weather_data(args.lat, args.lon, start_date, future_end_date, args.api_key)
    if "error" in weather_data:
        print("Échec de la récupération des données météorologiques. Arrêt du programme.")
        # Essayer de récupérer uniquement les données historiques
        print("Tentative de récupération des données historiques uniquement...")
        weather_data = get_weather_data(args.lat, args.lon, start_date, end_date, args.api_key)
        if "error" in weather_data:
            print("Échec de la récupération des données historiques. Arrêt du programme.")
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
    
    # 4. Préparer les données pour la prédiction (uniquement les données historiques)
    # Filtrer les données météo jusqu'à la date cible
    historical_weather_data = copy.deepcopy(weather_data)
    historical_weather_data['days'] = [day for day in weather_data['days'] 
                                     if datetime.strptime(day['datetime'], "%Y-%m-%d") <= target_date]
    
    df = prepare_data_for_prediction(historical_weather_data, elevation, soil_type, args.lat, args.lon)
    
    # 5. Standardiser les caractéristiques numériques
    df = standardize_features(df, NUMERIC_FEATURES)
    
    # 6. Préparer la séquence pour le LSTM
    X, input_dates, feature_cols = prepare_sequence_for_lstm(df)
    if X is None:
        print("Échec de la préparation de la séquence LSTM. Arrêt du programme.")
        return 1
    
    # 7. Charger le modèle LSTM
    model = load_lstm_model(args.model)
    if model is None:
        print("Échec du chargement du modèle LSTM. Arrêt du programme.")
        return 1
    
    # 8. Faire les prédictions incrémentales
    prediction_results = predict_flood_risk_incremental(
        model, X, weather_data, df, feature_cols, input_dates, 
        args.lat, args.lon, elevation, soil_type, total_horizon
    )
    
    if prediction_results is None or prediction_results.empty:
        print("Échec des prédictions incrémentales. Arrêt du programme.")
        return 1
    
    # 9. Sauvegarder les résultats
    prediction_results.to_csv(args.output, index=False)
    print(f"Résultats de prédiction incrémentale sauvegardés dans {args.output}")
    
    # 10. Tracer le graphique si demandé
    if args.plot:
        plot_prediction_results_incremental(df, prediction_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
