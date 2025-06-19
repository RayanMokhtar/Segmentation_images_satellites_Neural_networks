#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demonstration_incremental_with_labels.py

Script de démonstration pour le modèle qui utilise les labels historiques.
"""
import sys, argparse, copy, math, re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests, httpx, joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Constantes alignées sur l'entraînement avec labels ---
L = 7
DEFAULT_H = 3
WEATHER_API = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
ELEV_API    = "https://api.opentopodata.org/v1/srtm30m"
SOIL_API    = "https://api.openepi.io/soil/type"
NOMINATIM_API = "https://nominatim.openstreetmap.org/reverse"
WEATHER_KEY = "MER35CT7Y4ZQUTCQSYC2SEADA"
# clé de secours : PUTUNBTHW5R3Q9K2WUW6MPSD6
# clé de secours : MER35CT7Y4ZQUTCQSYC2SEADA
EMDAT_FILE = r"d:\dataset\SEN12FLOOD (1)\public_emdat_custom_request_2025-05-22_1a78f1da-122a-41fb-9eac-038db183ca0a(1).csv"

# --- Listes de features (RESTAURÉES pour correspondre à l'entraînement) ---
TEMP_FEATS   = [
    'tempmax','tempmin','temp','feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precipprob','precipcover', # Restauré
    'windspeed','winddir','pressure','cloudcover','visibility'
]
STATIC_NUM   = ['elevation', 'historique_region']
CAT_FEAT     = 'soil_type'
CYCLE_FEATS  = ['day_of_year_sin','day_of_year_cos']
ROLL_BASE    = ['precipprob','humidity','precipcover'] # Restauré
ROLL_FEATS   = [f'rolling_{c}_mean' for c in ROLL_BASE] + [f'rolling_{c}_std' for c in ROLL_BASE]

# --- Chemins vers les fichiers du modèle avec labels ---
MODEL_PATH_DEFAULT    = 'best_model_with_labels_attention.keras'
SCALER_PATH   = 'scaler_train_with_labels_attention.pkl'
ENCODER_PATH  = 'encoder_train_with_labels_attention.pkl'
FEATS_PATH    = 'feats_list_with_labels_attention.pkl'

def get_city_from_coords_nominatim(lat, lon):
    headers = {'User-Agent': 'FloodPredictionScript/1.0'}
    params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10, 'accept-language': 'en'}
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

def get_flood_history_from_emdat(city_name, floods_df):
    if not city_name or pd.isna(city_name):
        return 0
    city_pattern = r'(?i)\b' + re.escape(city_name) + r'\b'
    matches = floods_df['Location'].str.contains(city_pattern, na=False, regex=True)
    return int(matches.sum())

def get_weather(lat, lon, start, end, key=None):
    k = key or WEATHER_KEY
    r = requests.get(f"{WEATHER_API}{lat},{lon}/{start}/{end}", params={'unitGroup':'metric','key':k,'include':'days','contentType':'json'})
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
    except: pass
    return 'Unknown'

def build_raw_df(wx, elev, soil, flood_history):
    rows=[]
    for day in wx['days'].values():
        row = {'date': day['datetime']}
        # RESTAURATION: On remplit les features météo en gérant les None
        for f in TEMP_FEATS:
            row[f] = day.get(f, 0) # Utilise .get(f, 0) pour éviter les erreurs
        row['elevation'] = elev
        row['soil_type'] = soil
        row['historique_region'] = flood_history
        rows.append(row)
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def derive_features(df):
    # RESTAURATION: Logique originale
    doy = df['date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2*np.pi*doy/365)
    df['day_of_year_cos'] = np.cos(2*np.pi*doy/365)
    for c in ROLL_BASE: # Utilise la ROLL_BASE restaurée
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
    p.add_argument('--lat', type=float, required=True)
    p.add_argument('--lon', type=float, required=True)
    p.add_argument('--date', type=str, required=True, help="YYYY-MM-DD")
    p.add_argument('--horizon', type=int, default=DEFAULT_H)
    p.add_argument('--model', type=str, default=MODEL_PATH_DEFAULT)
    p.add_argument('--plot', action='store_true')
    return p.parse_args()

def load_model_or_exit(path):
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"Modèle chargé: {path}, input_shape={m.input_shape}")
        return m
    except Exception as e:
        print(f"Erreur chargement modèle: {e}"); sys.exit(1)

def predict_incremental_with_labels(model, df_initial, scaler, encoder, feats_list, horizon):
    df_cur = df_initial.copy()
    results = []

    for h in range(1, horizon + 1):
        print(f"\n--- Prédiction pour J+{h} ---")
        
        df_processed = derive_features(df_cur)
        df_transformed = apply_transforms(df_processed, scaler, encoder, feats_list)
        X_cur = make_sequence(df_transformed, feats_list)

        print("Fenêtre d'entrée (7 derniers jours) pour la prédiction (valeurs brutes) :")
        # RESTAURATION: Affiche les bonnes colonnes
        display_cols = ['date', 'tempmax', 'tempmin', 'temp', 'precipprob', 'precipcover']
        if 'label' in df_processed.columns:
            display_cols.append('label')
        print(df_processed.tail(L)[display_cols].to_string(index=False))

        # Prédiction
        prob = float(model.predict(X_cur)[0, 0])
        pred_label = int(prob >= 0.5)
        pred_date = df_cur['date'].iloc[-1] + timedelta(days=1)
        results.append({'date': pred_date, 'prob': prob, 'pred': pred_label})

        if h < horizon:
            ds_str = pred_date.strftime("%Y-%m-%d")
            
            if ds_str in wx['days']:
                day_data = wx['days'][ds_str]
                new_row_dict = {'date': pred_date}
                # RESTAURATION: Logique originale pour la nouvelle ligne
                for f in TEMP_FEATS:
                    new_row_dict[f] = day_data.get(f, 0)
                new_row_dict['elevation'] = df_cur['elevation'].iloc[0]
                new_row_dict['soil_type'] = df_cur['soil_type'].iloc[0]
                new_row_dict['historique_region'] = df_cur['historique_region'].iloc[0]
                new_row_dict['label'] = pred_label
            else:
                new_row_dict = df_cur.iloc[-1].to_dict()
                new_row_dict['date'] = pred_date
                new_row_dict['label'] = pred_label

            df_cur = pd.concat([df_cur, pd.DataFrame([new_row_dict])], ignore_index=True)

    return pd.DataFrame(results)

def plot_results(df_raw, preds):
    # ... (code de plot inchangé) ...
    pass

def main():
    args = parse_args()
    try:
        tgt_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print("Date invalide. Format YYYY-MM-DD."); sys.exit(1)

    try:
        emdat_df = pd.read_csv(EMDAT_FILE, sep=';', encoding='utf-8')
        floods_df = emdat_df[emdat_df['Disaster Type'].str.contains('Flood', na=False)].copy()
    except Exception as e:
        print(f"Erreur chargement EM-DAT: {e}"); sys.exit(1)

    print(f"Détermination de la ville pour ({args.lat}, {args.lon})...")
    city_name = get_city_from_coords_nominatim(args.lat, args.lon)
    if not city_name:
        print("Erreur: Impossible de déterminer la ville."); sys.exit(1)
    print(f"-> Ville: '{city_name}'")

    flood_history = get_flood_history_from_emdat(city_name, floods_df)
    print(f"-> Historique inondations pour '{city_name}': {flood_history}")

    start_date = (tgt_date - timedelta(days=L-1)).strftime("%Y-%m-%d")
    end_date_h = (tgt_date + timedelta(days=args.horizon)).strftime("%Y-%m-%d")

    print("Récupération des données météo...")
    global wx; wx = get_weather(args.lat, args.lon, start_date, end_date_h)
    wx['days'] = {d['datetime']: d for d in wx['days']}

    elev = get_elevation(args.lat, args.lon)
    soil = get_soil(args.lat, args.lon)

    df_raw = build_raw_df(wx, elev, soil, flood_history)
    df_hist = df_raw[df_raw['date'] <= tgt_date].reset_index(drop=True)
    
    # HYPOTHÈSE INITIALE : les 7 jours d'historique n'avaient pas d'inondation
    df_hist['label'] = 0

    if len(df_hist) < L:
        print(f"Pas assez de données historiques ({len(df_hist)} jours)."); sys.exit(1)

    try:
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        feats_list = joblib.load(FEATS_PATH)
    except FileNotFoundError as e:
        print(f"Erreur: Fichier de preprocessing non trouvé: {e}"); sys.exit(1)

    model = load_model_or_exit(args.model)

    preds = predict_incremental_with_labels(model, df_hist, scaler, encoder, feats_list, args.horizon)

    print("\n=== Prédictions Finales ===")
    print(preds.to_string(index=False))

    if args.plot:
        plot_results(df_raw, preds)

if __name__ == "__main__":
    main()