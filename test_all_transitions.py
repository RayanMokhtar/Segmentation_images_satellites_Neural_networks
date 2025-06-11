# -*- coding: utf-8 -*-
"""
test_transitions_notebook.py

Script d'évaluation des transitions dans le jeu de test :
1. Charger directement test_dataset.csv (données de test déjà exportées)
2. Prétraiter les données (imputation, one-hot soil_type, standardisation)
3. Pour chaque région de test :
     - détecter la première transition 0→1 du label
     - construire la séquence des L jours historiques précédant la transition (moins H)
     - prédire la probabilité et la classe
4. Rassembler toutes les prédictions et calculer TP, FN, FP, TN, accuracy, precision, recall, F1, AUC ROC et un rapport détaillé.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

# --- 1. Paramètres identiques au notebook ---
L      = 7       # taille de la fenêtre historique (jours)
H      = 3       # horizon de prédiction (jours après la fenêtre)
TARGET = 'label'

# --- 2. Définition des features utiles ---
# Météo / topo numériques + soil_type + label + date + region
numeric_feats = [
    'latitude_centroid','longitude_centroid',
    'tempmax','tempmin','temp','feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precipprob','precipcover',
    'windspeed','winddir','pressure','cloudcover','visibility',
    'elevation'
]
cat_feats = ['soil_type']
all_feats = numeric_feats + cat_feats

# --- 3. Chargement du jeu de test directement depuis test_dataset.csv ---
print("Chargement du jeu de test...")
try:
    # Vérifier les colonnes disponibles dans le fichier
    df_peek = pd.read_csv('./test_dataset_enriched.csv', nrows=1)
    print(f"Colonnes disponibles: {df_peek.columns.tolist()}")
    
    # Vérifier si soil_type est déjà encodé en one-hot
    soil_cols = [col for col in df_peek.columns if col.startswith('soil_')]
    if soil_cols:
        print(f"Colonnes de type de sol trouvées (one-hot): {soil_cols}")
        cat_feats = []  # soil_type déjà encodé
    else:
        print("Colonne soil_type trouvée (non encodée)")
        cat_feats = ['soil_type']
    
    # Adapter la liste des colonnes à charger
    all_feats = numeric_feats + cat_feats
    usecols = ['chemin_directory', 'date'] + all_feats + soil_cols + [TARGET]
    
    # Chargement du jeu de test
    df_test = pd.read_csv(
        './test_dataset.csv',
        usecols=usecols,
        parse_dates=['date'],
        dayfirst=True
    )
    print(f"Jeu de test chargé: {len(df_test)} lignes")
    
    # S'assurer que la colonne date est au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df_test['date']):
        print("Conversion de la colonne 'date' au format datetime...")
        df_test['date'] = pd.to_datetime(df_test['date'])
    print(f"Type de la colonne date: {df_test['date'].dtype}")
    
except Exception as e:
    print(f"Erreur lors du chargement du fichier test_dataset.csv: {str(e)}")
    print("Tentative de chargement sans filtrage de colonnes...")
    df_test = pd.read_csv(
        './test_dataset.csv',
        parse_dates=['date'],
        dayfirst=True
    )
    print(f"Jeu de test chargé sans filtrage: {len(df_test)} lignes")
    
    # S'assurer que la colonne date est au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df_test['date']):
        print("Conversion de la colonne 'date' au format datetime...")
        df_test['date'] = pd.to_datetime(df_test['date'])
    print(f"Type de la colonne date: {df_test['date'].dtype}")

# --- 4. Imputation des NaN sur les variables numériques ---
print("Imputation des valeurs manquantes...")
for col in numeric_feats:
    if col in df_test.columns:
        median_val = df_test[col].median()
        df_test[col] = df_test[col].fillna(median_val)
        print(f"Colonne {col}: {df_test[col].isna().sum()} valeurs manquantes après imputation")

# --- 5. One-hot encoding du type de sol (si nécessaire) ---
if 'soil_type' in df_test.columns:
    print("Application du one-hot encoding pour soil_type...")
    df_test = pd.get_dummies(df_test, columns=['soil_type'], prefix='soil')
    soil_cols = [col for col in df_test.columns if col.startswith('soil_')]
    print(f"Nouvelles colonnes créées: {soil_cols}")

# --- 6. Standardisation des variables numériques continues ---
print("Normalisation des variables numériques...")
available_numeric_feats = [f for f in numeric_feats if f in df_test.columns]
if available_numeric_feats:
    scaler = StandardScaler().fit(df_test[available_numeric_feats])
    df_test[available_numeric_feats] = scaler.transform(df_test[available_numeric_feats])
    print(f"Variables normalisées: {available_numeric_feats}")
else:
    print("AVERTISSEMENT: Aucune variable numérique à normaliser")

# --- 7. Tri par région puis date, réindexation ---
df_test = df_test.sort_values(['chemin_directory','date']).reset_index(drop=True)

# --- 8. Chargement du modèle entraîné (.keras) ---
try:
    model = load_model('./modele_inondation_complete.keras')
    print("Modèle chargé avec succès: best_model.keras")
except Exception as e:
    print(f"Erreur lors du chargement du modèle best_model.keras: {str(e)}")
    # Chercher d'autres modèles disponibles
    import os
    model_files = [f for f in os.listdir() if f.endswith('.keras') or f.endswith('.h5')]
    if model_files:
        print(f"Autres modèles disponibles: {model_files}")
        try:
            model = load_model(model_files[0])
            print(f"Modèle alternatif chargé: {model_files[0]}")
        except Exception as e2:
            print(f"Erreur lors du chargement du modèle alternatif: {str(e2)}")
            raise Exception("Impossible de charger un modèle valide")
    else:
        raise Exception("Aucun modèle trouvé dans le répertoire")

# --- 10. Parcours des régions de test et détection des transitions 0→1 ---
y_trues = []
y_preds = []
y_probs = []

for region, grp in df_test.groupby('chemin_directory'):
    grp = grp.reset_index(drop=True)
    # 10a. Cherche le premier indice i tel que label passe de 0→1
    idx_trans = None
    for i in range(1, len(grp)):
        if grp.loc[i-1, TARGET] == 0 and grp.loc[i, TARGET] == 1:
            idx_trans = i
            break
    if idx_trans is None:
        continue  # pas de transition dans cette région

    # 10b. Calcul des bornes de la séquence historique
    date_trans = pd.to_datetime(grp.loc[idx_trans, 'date'])
    print(f"  Région {region}: date de transition: {date_trans}, type: {type(date_trans)}")
    end_date   = date_trans - pd.Timedelta(days=H)
    start_date = end_date - pd.Timedelta(days=L-1)    # 10c. Extraction de la séquence des L jours
    seq = grp[(grp['date'] >= start_date) & (grp['date'] <= end_date)]
    if len(seq) != L:
        print(f"  Région {region}: séquence incomplète ({len(seq)}/{L} jours), ignorée")
        continue  # insufficient data, skip

    # 10d. Préparation de l'entrée pour le modèle
    feature_cols = [col for col in seq.columns if col not in ['chemin_directory', 'date', TARGET]]
    X = seq[feature_cols].values
    
    # Vérifier la forme attendue par le modèle
    expected_features = model.input_shape[-1]
    if X.shape[1] != expected_features:
        print(f"  ATTENTION: Le nombre de caractéristiques ({X.shape[1]}) ne correspond pas à ce qu'attend le modèle ({expected_features})")
        print(f"  Caractéristiques disponibles: {feature_cols}")
        print(f"  Adaptation des dimensions...")
        
        # Si trop de caractéristiques, tronquer
        if X.shape[1] > expected_features:
            X = X[:, :expected_features]
            print(f"  Caractéristiques tronquées à {X.shape[1]}")
        # Si pas assez de caractéristiques, compléter avec des zéros
        elif X.shape[1] < expected_features:
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.hstack((X, padding))
            print(f"  Caractéristiques complétées avec des zéros à {X.shape[1]}")
    
    X = X.reshape(1, L, -1).astype(np.float32)
    print(f"  Région {region}: forme de X après reshape: {X.shape}")

    # 10e. Prédiction
    prob = float(model.predict(X, verbose=0)[0,0])
    pred = 1 if prob >= 0.5 else 0

    # 10f. Stockage
    y_trues.append(1)   # on teste toujours une transition vers 1
    y_preds.append(pred)
    y_probs.append(prob)

# --- 11. Calcul des métriques globales ---
# Matrice de confusion
tn, fp, fn, tp = confusion_matrix(y_trues, y_preds, labels=[0,1]).ravel()

accuracy  = accuracy_score(y_trues, y_preds)
precision = precision_score(y_trues, y_preds, zero_division=0)
recall    = recall_score(y_trues, y_preds, zero_division=0)
f1        = f1_score(y_trues, y_preds, zero_division=0)
# AUC ROC
try:
    auc = roc_auc_score(y_trues, y_probs)
except ValueError:
    auc = float('nan')

# --- 12. Affichage des résultats ---
print("\n=== Résultats sur le jeu de test ===")
print(f"Régions testées       : {len(y_trues)}")
print(f"TP (vrais positifs)   : {tp}")
print(f"FN (faux négatifs)    : {fn}")
print(f"FP (faux positifs)    : {fp}")
print(f"TN (vrais négatifs)   : {tn}\n")

print(f"Accuracy  = {accuracy:.3f}")
print(f"Precision = {precision:.3f}")
print(f"Recall    = {recall:.3f}")
print(f"F1-score  = {f1:.3f}")
print(f"AUC ROC   = {auc if not np.isnan(auc) else 'n/a'}\n")

print("--- Rapport détaillé ---")
print(classification_report(y_trues, y_preds, digits=3))
