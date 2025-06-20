#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demonstration_incremental.py

Script de démonstration adapté au modèle robuste.
À partir de coordonnées et d'une date, ce script :
1. Détermine la ville correspondante via l'API Nominatim.
2. Interroge le fichier local EM-DAT pour obtenir l'historique des inondations de cette ville.
3. Récupère les données météo, l'altitude et le type de sol via des APIs.
4. Prépare les données en appliquant la même chaîne de traitement que l'entraînement.
5. Prédit les risques d'inondation de manière incrémentale sur un horizon de H jours.
6. Affiche la fenêtre d'entrée du modèle avant chaque prédiction.
"""
import sys, argparse, copy, math, re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests, httpx, joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Constantes alignées sur l'entraînement robuste ---
L = 7               # Fenêtre historique (WINDOW_SIZE)
DEFAULT_H = 3       # Horizon de prédiction par défaut
WEATHER_API = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
ELEV_API    = "https://api.opentopodata.org/v1/srtm30m"
SOIL_API    = "https://api.openepi.io/soil/type"
NOMINATIM_API = "https://nominatim.openstreetmap.org/reverse"
WEATHER_KEY = "PUTUNBTHW5R3Q9K2WUW6MPSD6" # Remplacez par votre clé si nécessaire

# --- Fichier de données EMDAT ---
EMDAT_FILE = r"d:\\dataset\\SEN12FLOOD (1)\\public_emdat_custom_request_2025-05-22_1a78f1da-122a-41fb-9eac-038db183ca0a(1).csv"

# --- Listes de features identiques à l'entraînement ---
TEMP_FEATS   = [
    'tempmax','tempmin','temp','feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precipprob','precipcover',
    'windspeed','winddir','pressure','cloudcover','visibility'
]
STATIC_NUM   = ['elevation', 'historique_region']
CAT_FEAT     = 'soil_type'
CYCLE_FEATS  = ['day_of_year_sin','day_of_year_cos']
ROLL_BASE    = ['precipprob','humidity','precipcover']
ROLL_FEATS   = [f'rolling_{c}_mean' for c in ROLL_BASE] + [f'rolling_{c}_std' for c in ROLL_BASE]

# --- Chemins vers les fichiers du modèle robuste ---
MODEL_PATH_DEFAULT    = 'best_model_high_recall.keras'
SCALER_PATH   = 'scaler_train_high_recall.pkl'
ENCODER_PATH  = 'encoder_train_high_recall.pkl'
FEATS_PATH    = 'feats_list_high_recall.pkl'

def get_city_from_coords_nominatim(lat, lon):
    """
    Utilise l'API Nominatim pour obtenir le nom de la ville à partir des coordonnées.
    """
    headers = {
        'User-Agent': 'FloodPredictionScript/1.0'
    }
    params = {
        'lat': lat,
        'lon': lon,
        'format': 'json',
        'zoom': 10,  # Niveau de zoom pour obtenir la ville
        'accept-language': 'en'
    }
    try:
        resp = requests.get(NOMINATIM_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        address = data.get('address', {})
        # Cherche la ville, le bourg ou le village dans les détails de l'adresse
        city = address.get('city', address.get('town', address.get('village')))
        return city
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'appel à l'API Nominatim: {e}")
        return None

def get_flood_history_from_emdat(city_name, floods_df):
    """
    Compte les inondations pour une ville donnée à partir du dataframe EM-DAT pré-filtré.
    """
    if not city_name or pd.isna(city_name):
        return 0
    
    city_pattern = r'(?i)\b' + re.escape(city_name) + r'\b'
    matches = floods_df['Location'].str.contains(city_pattern, na=False, regex=True)
    return int(matches.sum())

def get_weather(lat, lon, start, end, key=None):
    k = key or WEATHER_KEY
    r = requests.get(
        f"{WEATHER_API}{lat},{lon}/{start}/{end}",
        params={'unitGroup':'metric','key':k,'include':'days','contentType':'json'}
    )
    r.raise_for_status()
    return r.json()

def get_elevation(lat, lon):
    r = requests.get(f"{ELEV_API}?locations={lat},{lon}")
    r.raise_for_status()
    e = r.json()['results'][0]['elevation']
    return 0 if e is None else e

def get_soil(lat, lon):
    try:
        with httpx.Client(timeout=10) as c:
            r = c.get(SOIL_API, params={'lat':lat,'lon':lon})
        if r.status_code==200:
            return r.json().get('properties',{}).get('most_probable_soil_type','Unknown')
    except:
        pass
    return 'Unknown'

def build_raw_df(wx, elev, soil, flood_history):
    rows=[]
    for day in wx['days']:
        row = {'date': day['datetime']}
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
    doy = df['date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2*np.pi*doy/365)
    df['day_of_year_cos'] = np.cos(2*np.pi*doy/365)
    for c in ROLL_BASE:
        df[f'rolling_{c}_mean'] = df[c].rolling(L, min_periods=1).mean()
        df[f'rolling_{c}_std']  = df[c].rolling(L, min_periods=1).std().fillna(0)
    df[ROLL_FEATS] = df[ROLL_FEATS].bfill()
    return df

def apply_transforms(df, scaler, encoder, feats_list):
    df_transformed = df.copy()
    num_feats  = [f for f in feats_list if f in scaler.feature_names_in_]
    df_transformed[num_feats] = scaler.transform(df_transformed[num_feats])
    
    cat_arr    = encoder.transform(df_transformed[[CAT_FEAT]])
    cat_cols   = [f"{CAT_FEAT}_{v}" for v in encoder.categories_[0]]
    df_transformed[cat_cols] = cat_arr
    return df_transformed

def make_sequence(df, feats):
    arr = df.iloc[-L:][feats].values.astype(np.float32)
    return arr.reshape(1, L, -1)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--lat',     type=float, required=True, help="Latitude du point à prédire")
    p.add_argument('--lon',     type=float, required=True, help="Longitude du point à prédire")
    p.add_argument('--date',    type=str,   required=True, help="Date de départ de la prédiction (YYYY-MM-DD)")
    p.add_argument('--horizon', type=int,   default=DEFAULT_H, help="Nombre de jours à prédire dans le futur")
    p.add_argument('--api-key', type=str,   default=None, help="Clé API pour VisualCrossing Weather")
    p.add_argument('--model',   type=str,   default=MODEL_PATH_DEFAULT, help="Chemin vers le modèle .keras")
    p.add_argument('--plot',    action='store_true', help="Afficher un graphique des résultats")
    return p.parse_args()

def load_model_or_exit(path):
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"Modèle chargé: {path}, input_shape={m.input_shape}")
        return m
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        sys.exit(1)

def predict_incremental(model, X0, df_raw, scaler, encoder, feats_list, horizon):
    df_cur = df_raw.copy()
    X_cur = X0.copy()
    
    future_weather = {d['datetime']: d for d in wx['days']}
    results = []

    for h in range(1, horizon + 1):
        print(f"\n--- Prédiction pour J+{h} ---")
        print("Fenêtre d'entrée (7 derniers jours) :\n", df_cur.tail(L)[['date', 'temp', 'humidity', 'precipprob', 'historique_region']])
        
        prob = float(model.predict(X_cur)[0, 0])
        pred_label = int(prob >= 0.5)
        pred_date = df_cur['date'].iloc[-1] + timedelta(days=1)
        results.append({'date': pred_date, 'prob': prob, 'pred': pred_label})

        if h < horizon:
            ds_str = pred_date.strftime("%Y-%m-%d")
            
            if ds_str in future_weather:
                day_data = future_weather[ds_str]
                new_row_dict = {'date': pred_date}
                for f in TEMP_FEATS:
                    new_row_dict[f] = day_data.get(f, 0)
                new_row_dict['elevation'] = df_cur['elevation'].iloc[0]
                new_row_dict['soil_type'] = df_cur['soil_type'].iloc[0]
                new_row_dict['historique_region'] = df_cur['historique_region'].iloc[0]
            else:
                new_row_dict = df_cur.iloc[-1].to_dict()
                new_row_dict['date'] = pred_date

            df_cur = pd.concat([df_cur, pd.DataFrame([new_row_dict])], ignore_index=True)
            df_processed = derive_features(df_cur)
            df_transformed = apply_transforms(df_processed, scaler, encoder, feats_list)
            X_cur = make_sequence(df_transformed, feats_list)

    return pd.DataFrame(results)

def plot_results(df_raw, preds):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(12,10), gridspec_kw={'height_ratios':[2,1,1]})
    
    ax1.plot(df_raw['date'], df_raw['temp'], 'b-', label='Température (°C)')
    ax1.plot(df_raw['date'], df_raw['humidity'], 'g-', label='Humidité (%)')
    ax1.plot(df_raw['date'], df_raw['precipprob'], 'c--', label='Prob. Précip (%)')
    ax1.axvline(preds['date'].iloc[0] - timedelta(days=1), color='r', linestyle='--', label='Début prédiction')
    ax1.legend(); ax1.set_title('Météo Historique & Prévisions'); ax1.grid(True)

    ax2.plot(preds['date'], preds['prob'], 'ro-', label='Probabilité prédite')
    ax2.axhline(0.5, color='k', linestyle='--', label='Seuil de décision (0.5)')
    ax2.set_ylim(0,1); ax2.set_title('Probabilité d’inondation'); ax2.grid(True); ax2.legend()

    sf = ['historique_region','elevation']
    vals = [df_raw[f].iloc[0] for f in sf]
    ax3.bar(sf, vals, color=['crimson','navy']); ax3.set_title('Facteurs Statiques')
    for i,v in enumerate(vals):
        ax3.text(i, v, str(round(v, 2)), ha='center', va='bottom')
        
    plt.tight_layout(); plt.show()

def main():
    args = parse_args()
    try:
        tgt_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print("Date invalide. Le format doit être YYYY-MM-DD."); sys.exit(1)

    # 1. Chargement du fichier EM-DAT et filtrage des inondations
    try:
        emdat_df = pd.read_csv(EMDAT_FILE, sep=';', encoding='utf-8')
        floods_df = emdat_df[emdat_df['Disaster Type'].str.contains('Flood', na=False)].copy()
    except Exception as e:
        print(f"Erreur lors du chargement ou du filtrage du fichier EM-DAT: {e}"); sys.exit(1)

    # 2. Détermination de la ville via Nominatim (NOUVELLE LOGIQUE)
    print(f"Détermination de la ville pour les coordonnées ({args.lat}, {args.lon}) via Nominatim...")
    city_name = get_city_from_coords_nominatim(args.lat, args.lon)

    if not city_name:
        print("Erreur: Impossible de déterminer la ville à partir des coordonnées. Arrêt du script.")
        sys.exit(1)
    else:
        print(f"-> Ville déterminée : '{city_name}'")

    # 3. Récupération de l'historique des inondations
    print("Récupération de l'historique des inondations depuis le fichier EM-DAT...")
    flood_history = get_flood_history_from_emdat(city_name, floods_df)
    print(f"-> Historique pour la ville '{city_name}': {flood_history} inondations trouvées.")

    # 4. Récupération des autres données externes
    start_date = (tgt_date - timedelta(days=L-1)).strftime("%Y-%m-%d")
    end_date_h = (tgt_date + timedelta(days=args.horizon)).strftime("%Y-%m-%d")

    print("Récupération des données météo...")
    global wx; wx = get_weather(args.lat, args.lon, start_date, end_date_h, args.api_key)
    
    print("Récupération de l'altitude et du type de sol...")
    elev = get_elevation(args.lat, args.lon)
    soil = get_soil(args.lat, args.lon)

    # 5. Construction et prétraitement du dataframe initial
    df_raw = build_raw_df(wx, elev, soil, flood_history)
    df_with_features = derive_features(df_raw)
    
    df_hist = df_with_features[df_with_features['date'] <= tgt_date].reset_index(drop=True)
    if len(df_hist) < L:
        print(f"Pas assez de données historiques ({len(df_hist)} jours) pour former une fenêtre de taille {L}."); sys.exit(1)

    # 6. Chargement des objets de preprocessing et du modèle
    try:
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        feats_list = joblib.load(FEATS_PATH)
    except FileNotFoundError as e:
        print(f"Erreur: Fichier de preprocessing non trouvé: {e}. Lancez le script d'entraînement d'abord.")
        sys.exit(1)

    # 7. Transformation des données et création de la première séquence
    df_hist_transformed = apply_transforms(df_hist, scaler, encoder, feats_list)
    X0 = make_sequence(df_hist_transformed, feats_list)
    
    model = load_model_or_exit(args.model)

    # 8. Lancement de la prédiction incrémentale
    df_raw_hist = df_raw[df_raw['date'] <= tgt_date]
    preds = predict_incremental(model, X0, df_raw_hist, scaler, encoder, feats_list, args.horizon)

    print("\n=== Prédictions Finales ===")
    print(preds.to_string(index=False))

    if args.plot:
        plot_results(df_raw, preds)

if __name__ == "__main__":
    main()