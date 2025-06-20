#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lstm_prediction.py

Module pour la prédiction d'inondations basé sur le modèle LSTM avec labels.
S'inspire de demonstration_incremental_with_labels.py.
"""
import sys, math, re
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import requests, joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from django.conf import settings

# --- Constantes alignées sur l'entraînement avec labels ---
L = 7  # Taille de la fenêtre de prédiction
DEFAULT_H = 3  # Horizon de prédiction par défaut
WEATHER_API = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
ELEV_API = "https://api.opentopodata.org/v1/srtm30m"
SOIL_API = "https://api.openepi.io/soil/type"
NOMINATIM_API = "https://nominatim.openstreetmap.org/reverse"
WEATHER_KEY = "PUTUNBTHW5R3Q9K2WUW6MPSD6"

#api requête MER35CT7Y4ZQUTCQSYC2SEADA

# --- Listes de features (RESTAURÉES pour correspondre à l'entraînement) ---
TEMP_FEATS = [
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
    'dew', 'humidity', 'precipprob', 'precipcover',
    'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility'
]
STATIC_NUM = ['elevation', 'historique_region']
CAT_FEAT = 'soil_type'
CYCLE_FEATS = ['day_of_year_sin', 'day_of_year_cos']
ROLL_BASE = ['precipprob', 'humidity', 'precipcover']
ROLL_FEATS = [f'rolling_{c}_mean' for c in ROLL_BASE] + [f'rolling_{c}_std' for c in ROLL_BASE]

# --- Chemins vers les fichiers du modèle avec labels ---
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'best_model_with_labels_attention.keras')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', 'scaler_train_with_labels_attention.pkl')
ENCODER_PATH = os.path.join(settings.BASE_DIR, 'models', 'encoder_train_with_labels_attention.pkl')
FEATS_PATH = os.path.join(settings.BASE_DIR, 'models', 'feats_list_with_labels_attention.pkl')
EMDAT_FILE = os.path.join(settings.BASE_DIR, 'static', 'public_emdat_custom_request_2025-05-22_1a78f1da-122a-41fb-9eac-038db183ca0a(1).csv')

# --- Chemins vers les fichiers du modèle SANS labels (autonome) ---
MODEL_PATH_STANDALONE = os.path.join(settings.BASE_DIR, 'models_sans_label', 'best_model_high_recall.keras')
SCALER_PATH_STANDALONE = os.path.join(settings.BASE_DIR, 'models_sans_label', 'scaler_train_high_recall.pkl')
ENCODER_PATH_STANDALONE = os.path.join(settings.BASE_DIR, 'models_sans_label', 'encoder_train_high_recall.pkl')
FEATS_PATH_STANDALONE = os.path.join(settings.BASE_DIR, 'models_sans_label', 'feats_list_high_recall.pkl')


def get_city_from_coords_nominatim(lat, lon):
    """
    Récupère le nom de la ville à partir des coordonnées GPS en utilisant l'API Nominatim
    """
    headers = {'User-Agent': 'FloodPredictionScript/1.0'}
    params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10, 'accept-language': 'fr'}
    try:
        resp = requests.get(NOMINATIM_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        address = data.get('address', {})
        city = address.get('city', address.get('town', address.get('village')))
        return city
    except requests.exceptions.RequestException as e:
        print(f"Erreur API Nominatim: {e}")
        return None


def get_flood_history_from_emdat(city_name, floods_df=None):
    """
    Obtient l'historique des inondations pour une ville à partir du fichier EM-DAT
    """
    if not city_name or pd.isna(city_name):
        return 0
        
    # Si le DataFrame n'est pas fourni et que le fichier existe, on le charge
    if floods_df is None:
        try:
            if os.path.exists(EMDAT_FILE):
                floods_df = pd.read_csv(EMDAT_FILE, sep=';', encoding='utf-8')
                # Filtrer pour ne garder que les inondations
                flood_types = ['Flood', 'Riverine flood', 'Flash flood', 'Coastal flood', 'Flood (General)']
                # Vérifier si la colonne 'Disaster Type' existe
                if 'Disaster Type' in floods_df.columns:
                    floods_df = floods_df[floods_df['Disaster Type'].str.contains('|'.join(flood_types), na=False)].copy()
                # Si non, essayer avec 'Disaster Subtype'
                elif 'Disaster Subtype' in floods_df.columns:
                    floods_df = floods_df[floods_df['Disaster Subtype'].str.contains('|'.join(flood_types), na=False)].copy()
                else:
                    print("Colonnes 'Disaster Type' ou 'Disaster Subtype' non trouvées dans le fichier EMDAT")
            else:
                return 0  # Si pas de fichier, on renvoie 0 comme historique
        except Exception as e:
            print(f"Erreur chargement EM-DAT: {e}")
            return 0
            
    if floods_df is None or floods_df.empty:
        return 0
        
    city_pattern = r'(?i)\b' + re.escape(city_name) + r'\b'
    
    # Tenter de trouver des correspondances dans différentes colonnes
    total_matches = 0
    
    # Vérifier dans 'Location'
    if 'Location' in floods_df.columns:
        matches = floods_df['Location'].str.contains(city_pattern, na=False, regex=True)
        total_matches += int(matches.sum())
      # Vérifier dans 'Event Name'
    if 'Event Name' in floods_df.columns and total_matches == 0:
        matches = floods_df['Event Name'].str.contains(city_pattern, na=False, regex=True)
        total_matches += int(matches.sum())
    
    # Si aucune correspondance directe, essayer de trouver des inondations dans le pays
    # (Cela nécessiterait de connaître le pays de la ville, que nous n'avons pas directement)
    # Nous pourrions implémenter cette logique si nécessaire
    
    return total_matches
    
    # Si aucune correspondance directe, essayer de trouver des inondations dans le pays
    # (Cela nécessiterait de connaître le pays de la ville, que nous n'avons pas directement)
    # Nous pourrions implémenter cette logique si nécessaire
    
    return total_matches


def get_weather(lat, lon, start, end, key=None):
    """
    Récupère les données météo pour une période et des coordonnées données
    """
    k = key or WEATHER_KEY
    r = requests.get(f"{WEATHER_API}{lat},{lon}/{start}/{end}", 
                     params={'unitGroup': 'metric', 'key': k, 'include': 'days', 'contentType': 'json'})
    r.raise_for_status()
    return r.json()


def get_elevation(lat, lon):
    """
    Récupère l'élévation pour des coordonnées données
    """
    try:
        r = requests.get(f"{ELEV_API}?locations={lat},{lon}")
        r.raise_for_status()
        e = r.json()['results'][0]['elevation']
        return 0 if e is None else e
    except Exception as e:
        print(f"Erreur récupération élévation: {e}")
        return 0


def get_soil(lat, lon):
    """
    Récupère le type de sol pour des coordonnées données
    """
    try:
        r = requests.get(SOIL_API, params={'lat': lat, 'lon': lon}, timeout=10)
        if r.status_code == 200:
            return r.json().get('properties', {}).get('most_probable_soil_type', 'Unknown')
    except Exception as e:
        print(f"Erreur récupération type de sol: {e}")
    return 'Unknown'


def build_raw_df(wx, elev, soil, flood_history):
    """
    Construit un DataFrame à partir des données météo et des caractéristiques statiques
    """
    rows = []
    for day_key, day in wx['days'].items():
        row = {'date': day['datetime']}
        # Remplir les features météo en gérant les None
        for f in TEMP_FEATS:
            row[f] = day.get(f, 0)
        row['elevation'] = elev
        row['soil_type'] = soil
        row['historique_region'] = flood_history
        rows.append(row)
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def derive_features(df):
    """
    Dérive les caractéristiques cycliques et de moyenne mobile
    """
    # Encodage cyclique du jour de l'année
    doy = df['date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * doy / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * doy / 365)
    
    # Caractéristiques de moyenne mobile
    for c in ROLL_BASE:
        df[f'rolling_{c}_mean'] = df[c].rolling(L, min_periods=1).mean()
        df[f'rolling_{c}_std'] = df[c].rolling(L, min_periods=1).std().fillna(0)
    df[ROLL_FEATS] = df[ROLL_FEATS].bfill()
    
    return df


def apply_transforms(df, scaler, encoder, feats_list):
    """
    Applique les transformations de standardisation et encodage
    """
    df_transformed = df.copy()
    num_feats = [f for f in feats_list if f in scaler.feature_names_in_]
    df_transformed[num_feats] = scaler.transform(df_transformed[num_feats])
    cat_arr = encoder.transform(df_transformed[[CAT_FEAT]])
    cat_cols = [f"{CAT_FEAT}_{v}" for v in encoder.categories_[0]]
    df_transformed[cat_cols] = cat_arr
    return df_transformed


def make_sequence(df, feats):
    """
    Crée une séquence pour l'entrée du modèle LSTM
    """
    arr = df.iloc[-L:][feats].values.astype(np.float32)
    return arr.reshape(1, L, -1)


def load_model_or_exit(path):
    """
    Charge le modèle TensorFlow ou renvoie une erreur
    """
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"Modèle chargé: {path}, input_shape={m.input_shape}")
        return m
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
        raise e


def predict_incremental_with_labels(model, df_initial, scaler, encoder, feats_list, horizon):
    """
    Effectue la prédiction incrémentale avec mise à jour des labels
    """
    df_cur = df_initial.copy()
    results = []

    for h in range(1, horizon + 1):
        print(f"\n--- Prédiction pour J+{h} ---")
        
        df_processed = derive_features(df_cur)
        df_transformed = apply_transforms(df_processed, scaler, encoder, feats_list)
        X_cur = make_sequence(df_transformed, feats_list)

        print("Fenêtre d'entrée (7 derniers jours) pour la prédiction (valeurs brutes) :")
        display_cols = ['date', 'tempmax', 'tempmin', 'temp', 'precipprob', 'precipcover']
        if 'label' in df_processed.columns:
            display_cols.append('label')
        print(df_processed.tail(L)[display_cols].to_string(index=False))

        # Prédiction
        prob = float(model.predict(X_cur)[0, 0])
        pred_label = int(prob >= 0.5)
        pred_date = df_cur['date'].iloc[-1] + timedelta(days=1)
        
        # Récupérer les données météo pour le jour prédit
        ds_str = pred_date.strftime("%Y-%m-%d")
        day_weather = {}
        if 'days' in wx and ds_str in wx['days']:
            day_data = wx['days'][ds_str]
            day_weather = {
                'temp': day_data.get('temp'),
                'precip': day_data.get('precip', 0),
                'humidity': day_data.get('humidity')
            }
        
        results.append({'date': pred_date, 'prob': prob, 'pred': pred_label, 'weather': day_weather})

        if h < horizon:
            ds_str = pred_date.strftime("%Y-%m-%d")
            
            # Création d'une nouvelle ligne pour le jour suivant
            if ds_str in wx.get('days', {}):
                day_data = wx['days'][ds_str]
                new_row_dict = {'date': pred_date}
                for f in TEMP_FEATS:
                    new_row_dict[f] = day_data.get(f, 0)
                new_row_dict['elevation'] = df_cur['elevation'].iloc[0]
                new_row_dict['soil_type'] = df_cur['soil_type'].iloc[0]
                new_row_dict['historique_region'] = df_cur['historique_region'].iloc[0]
                new_row_dict['label'] = pred_label
            else:
                # Si nous n'avons pas de données météo pour le jour suivant, 
                # utiliser les dernières valeurs
                new_row_dict = df_cur.iloc[-1].to_dict()
                new_row_dict['date'] = pred_date
                new_row_dict['label'] = pred_label

            # Ajouter la nouvelle ligne au DataFrame
            df_cur = pd.concat([df_cur, pd.DataFrame([new_row_dict])], ignore_index=True)

    return pd.DataFrame(results)


def predict_incremental_without_labels(model, df_initial, scaler, encoder, feats_list, horizon):
    """
    Effectue la prédiction incrémentale sans dépendance aux labels CNN.
    Cette fonction suit la même approche que celle utilisée dans demonstration_incremental.py.
    
    Args:
        model: Le modèle TensorFlow à utiliser pour les prédictions
        df_initial: DataFrame initial avec les données météorologiques et statiques
        scaler: StandardScaler pour normaliser les features numériques
        encoder: OneHotEncoder pour encoder les features catégorielles
        feats_list: Liste des features à utiliser pour les prédictions
        horizon: Nombre de jours pour lesquels faire des prédictions
    
    Returns:
        pd.DataFrame: DataFrame contenant les prédictions pour chaque jour
    """
    df_cur = df_initial.copy()
    results = []
    
    # Prétraiter les données initiales
    df_processed = derive_features(df_cur)
    df_transformed = apply_transforms(df_processed, scaler, encoder, feats_list)
    X_cur = make_sequence(df_transformed, feats_list)
    
    for h in range(1, horizon + 1):
        print(f"\n--- Prédiction pour J+{h} (modèle autonome) ---")
        
        print("Fenêtre d'entrée (7 derniers jours) pour la prédiction (valeurs brutes) :")
        display_cols = ['date', 'tempmax', 'tempmin', 'temp', 'precipprob', 'precipcover', 'humidity']
        print(df_processed.tail(L)[display_cols].to_string(index=False))
        
        # Prédiction
        prob = float(model.predict(X_cur)[0, 0])
        pred_label = int(prob >= 0.5)
        pred_date = df_cur['date'].iloc[-1] + timedelta(days=1)
        
        # Récupérer les données météo pour le jour prédit
        ds_str = pred_date.strftime("%Y-%m-%d")
        day_weather = {}
        if 'days' in wx and ds_str in wx['days']:
            day_data = wx['days'][ds_str]
            day_weather = {
                'temp': day_data.get('temp'),
                'precip': day_data.get('precip', 0),
                'humidity': day_data.get('humidity')
            }
            
        results.append({'date': pred_date, 'prob': prob, 'pred': pred_label, 'weather': day_weather})
        
        if h < horizon:
            ds_str = pred_date.strftime("%Y-%m-%d")
            
            # Création d'une nouvelle ligne pour le jour suivant
            if ds_str in wx.get('days', {}):
                day_data = wx['days'][ds_str]
                new_row_dict = {'date': pred_date}
                for f in TEMP_FEATS:
                    new_row_dict[f] = day_data.get(f, 0)
                new_row_dict['elevation'] = df_cur['elevation'].iloc[0]
                new_row_dict['soil_type'] = df_cur['soil_type'].iloc[0]
                new_row_dict['historique_region'] = df_cur['historique_region'].iloc[0]
            else:
                # Si nous n'avons pas de données météo pour le jour suivant,
                # utiliser les dernières valeurs
                new_row_dict = df_cur.iloc[-1].to_dict()
                new_row_dict['date'] = pred_date
            
            # Ajouter la nouvelle ligne au DataFrame
            df_cur = pd.concat([df_cur, pd.DataFrame([new_row_dict])], ignore_index=True)
            
            # Prétraiter pour la prochaine prédiction
            df_processed = derive_features(df_cur)
            df_transformed = apply_transforms(df_processed, scaler, encoder, feats_list)
            X_cur = make_sequence(df_transformed, feats_list)
    
    return pd.DataFrame(results)


def generate_prediction_plot(df_raw, preds, output_dir=None):
    """
    Génère un graphique de prédiction
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer les probabilités d'inondation
    ax.plot(preds['date'], preds['prob'] * 100, 'r-o', label='Probabilité d\'inondation (%)')
    
    # Ajouter des lignes horizontales pour les seuils
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Seuil de décision (50%)')
    
    # Colorier l'arrière-plan pour indiquer les zones de risque
    ax.axhspan(0, 25, color='green', alpha=0.1, label='Risque faible')
    ax.axhspan(25, 50, color='yellow', alpha=0.1, label='Risque modéré')
    ax.axhspan(50, 75, color='orange', alpha=0.1, label='Risque élevé')
    ax.axhspan(75, 100, color='red', alpha=0.1, label='Risque très élevé')
    
    # Marquer les prédictions positives
    for i, row in preds.iterrows():
        if row['pred'] == 1:
            ax.scatter(row['date'], row['prob'] * 100, color='red', s=100, marker='X')
    
    # Formater le graphique
    ax.set_xlabel('Date')
    ax.set_ylabel('Probabilité d\'inondation (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Prédiction du risque d\'inondation')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Formater les dates sur l'axe x
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarder si un répertoire est spécifié
    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'flood_prediction_{timestamp}.png')
        plt.savefig(output_path)
        return output_path
    
    # Convertir le graphique en base64 pour l'affichage web
    from io import BytesIO
    import base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(image_png).decode('utf-8')


def predict_flood_lstm(lat, lon, target_date, horizon=DEFAULT_H, use_cnn_labels=False):
    """
    Fonction principale pour la prédiction d'inondation LSTM
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        target_date (str): Date cible au format YYYY-MM-DD
        horizon (int): Horizon de prédiction en jours
        use_cnn_labels (bool): Utiliser les labels CNN pour améliorer la prédiction
        
    Returns:
        dict: Résultats de la prédiction
    """
    try:        # Conversion de la date cible
        try:
            tgt_date = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            return {'success': False, 'error': "Format de date invalide. Utilisez YYYY-MM-DD."}
            
        # Chargement des données historiques d'inondation
        floods_df = None
        if os.path.exists(EMDAT_FILE):
            try:
                floods_df = pd.read_csv(EMDAT_FILE, sep=';', encoding='utf-8')
                # Filtrer pour ne garder que les inondations
                flood_types = ['Flood', 'Riverine flood', 'Flash flood', 'Coastal flood', 'Flood (General)']
                floods_df = floods_df[floods_df['Disaster Type'].str.contains('|'.join(flood_types), na=False)].copy()
                print(f"Fichier EMDAT chargé: {len(floods_df)} événements d'inondation trouvés")
            except Exception as e:
                print(f"Erreur chargement EM-DAT: {e}")
        else:
            print(f"Fichier EMDAT non trouvé: {EMDAT_FILE}")
        
        # Récupération du nom de la ville
        print(f"Détermination de la ville pour ({lat}, {lon})...")
        city_name = get_city_from_coords_nominatim(lat, lon)
        if not city_name:
            return {'success': False, 'error': "Impossible de déterminer la ville pour ces coordonnées."}
        
        # Récupération de l'historique des inondations
        flood_history = get_flood_history_from_emdat(city_name, floods_df)
        print(f"Historique inondations pour '{city_name}': {flood_history}")
        
        # Calcul des dates pour la requête météo
        start_date = (tgt_date - timedelta(days=L-1)).strftime("%Y-%m-%d")
        end_date_h = (tgt_date + timedelta(days=horizon)).strftime("%Y-%m-%d")
        
        # Récupération des données météo
        print("Récupération des données météo...")
        global wx
        wx = get_weather(lat, lon, start_date, end_date_h)
        wx['days'] = {d['datetime']: d for d in wx['days']}
        
        # Récupération des données d'élévation et de sol
        elev = get_elevation(lat, lon)
        soil = get_soil(lat, lon)
        
        # Construction du DataFrame brut
        df_raw = build_raw_df(wx, elev, soil, flood_history)
        df_hist = df_raw[df_raw['date'] <= tgt_date].reset_index(drop=True)
        
        # CHOIX DU MODÈLE EN FONCTION DU PARAMÈTRE use_cnn_labels
        if use_cnn_labels:
            print("Utilisation du modèle LSTM avec labels CNN")
            # Hypothèse initiale : les 7 jours d'historique n'avaient pas d'inondation
            df_hist['label'] = 0
            
            # Vérification de la taille de l'historique
            if len(df_hist) < L:
                return {'success': False, 'error': f"Pas assez de données historiques ({len(df_hist)} jours)."}
            
            # Chargement des fichiers de preprocessing pour le modèle avec labels
            try:
                scaler = joblib.load(SCALER_PATH)
                encoder = joblib.load(ENCODER_PATH)
                feats_list = joblib.load(FEATS_PATH)
            except FileNotFoundError as e:
                return {'success': False, 'error': f"Fichier de preprocessing non trouvé: {e}"}
            
            # Chargement du modèle avec labels
            try:
                model = load_model_or_exit(MODEL_PATH)
            except Exception as e:
                return {'success': False, 'error': f"Erreur de chargement du modèle: {e}"}
            
            # Prédiction avec le modèle utilisant les labels CNN
            preds = predict_incremental_with_labels(model, df_hist, scaler, encoder, feats_list, horizon)
            model_type = 'LSTM avec CNN'
        else:
            print("Utilisation du modèle LSTM autonome (sans labels CNN)")
            # Vérification de la taille de l'historique
            if len(df_hist) < L:
                return {'success': False, 'error': f"Pas assez de données historiques ({len(df_hist)} jours)."}
                
            # Chargement des fichiers de preprocessing pour le modèle autonome
            try:
                scaler = joblib.load(SCALER_PATH_STANDALONE)
                encoder = joblib.load(ENCODER_PATH_STANDALONE)
                feats_list = joblib.load(FEATS_PATH_STANDALONE)
            except FileNotFoundError as e:
                return {'success': False, 'error': f"Fichier de preprocessing du modèle autonome non trouvé: {e}"}
                
            # Chargement du modèle autonome
            try:
                model = load_model_or_exit(MODEL_PATH_STANDALONE)
            except Exception as e:
                return {'success': False, 'error': f"Erreur de chargement du modèle autonome: {e}"}
                
            # Prédiction avec le modèle autonome (sans labels CNN)
            preds = predict_incremental_without_labels(model, df_hist, scaler, encoder, feats_list, horizon)
            model_type = 'LSTM standard'
          # Nous ne générons plus le graphique de prédiction
        plot_img = None
          # Détermination du niveau de risque pour la première prédiction
        first_prob = preds.iloc[0]['prob'] * 100
        if first_prob < 25:
            risk_level = "faible"
            color = "green"
        elif first_prob < 50:
            risk_level = "modéré"
            color = "yellow"
        elif first_prob < 75:
            risk_level = "élevé" 
            color = "orange"
        else:
            risk_level = "très élevé"
            color = "red"
        
        # Correction de couleur pour s'assurer qu'un risque élevé soit bien affiché en orange/rouge
        if risk_level == "élevé" and color != "orange":
            color = "orange"
        if first_prob >= 60:  # Forcer le rouge pour les probabilités vraiment élevées
            color = "red"
        
        # Construction du résultat
        result = {
            'success': True,
            'prediction_done': True,
            'prediction_value': round(first_prob),
            'risk_level': risk_level,
            'color': color,
            'date_prediction': tgt_date.strftime('%d/%m/%Y'),
            'latitude': lat,
            'longitude': lon,
            'model_type': 'LSTM avec CNN' if use_cnn_labels else 'LSTM standard',
            'city_name': city_name,
            'forecast_days': [
                {
                    'date': row['date'].strftime('%d/%m/%Y'),
                    'probability': round(row['prob'] * 100, 2),
                    'is_flooded': bool(row['pred']),
                    'risk_level': 'élevé' if row['prob'] >= 0.5 else 'faible',
                    'weather': row.get('weather', {})
                }
                for _, row in preds.iterrows()
            ],
            'plot_base64': plot_img if isinstance(plot_img, str) else None,
            'plot_path': plot_img if not isinstance(plot_img, str) else None
        }
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
