# -*- coding: utf-8 -*-
"""
test_transition.py

1. Charger un modèle LSTM entraîné (.keras)
2. Détecter dans le dataset une première transition 0→1
3. Construire la séquence des L jours historiques avant la transition (moins H)
4. Faire la prédiction et vérifier si elle correspond au vrai label
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- 1. Paramètres identiques à l'entraînement ---
L = 7       # taille de la fenêtre historique
H = 3       # horizon de prédiction
TARGET = 'label'

# --- 2. Liste des features à charger (pas de geometry ni d'objets) ---
features = [
    'latitude_centroid','longitude_centroid',
    'tempmax','tempmin','temp','feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precipprob','precipcover',
    'windspeed','winddir','pressure','cloudcover','visibility',
    'elevation','soil_type'
]

# --- 3. Chargement du CSV en ne gardant que les colonnes utiles ---
df = pd.read_csv(
    './dataset_prepared.csv',
    usecols=['chemin_directory','date'] + features + [TARGET],
    parse_dates=['date'],
    dayfirst=True
)

# --- 4. Imputation des NaN sur les seules variables numériques ---
numeric_feats = [c for c in features if c != 'soil_type']
for col in numeric_feats:
    df[col] = df[col].fillna(df[col].median())

# --- 5. One-hot encoding du type de sol ---
df = pd.get_dummies(df, columns=['soil_type'], prefix='soil')

# --- 6. Normalisation des variables numériques continues ---
num_cols = [c for c in numeric_feats]
scaler = StandardScaler().fit(df[num_cols])
df[num_cols] = scaler.transform(df[num_cols])

# --- 7. Tri par région puis date ---
df.sort_values(['chemin_directory','date'], inplace=True)

# --- 8. Recherche d’une transition 0→1 ---
transition = None
for region, grp in df.groupby('chemin_directory'):
    grp = grp.reset_index(drop=True)
    for i in range(1, len(grp)):
        if grp.loc[i-1, TARGET] == 0 and grp.loc[i, TARGET] == 1:
            transition = (region, grp.loc[i, 'date'])
            break
    if transition:
        break

if transition is None:
    raise RuntimeError("Aucune transition 0→1 trouvée dans le dataset.")

region, t_date = transition
print(f"→ Région : {region}")
print(f"→ Date de transition 0→1 : {t_date.date()}")

# --- 9. Construction de la séquence d’entrée ---
end_date   = t_date - pd.Timedelta(days=H)
start_date = end_date - pd.Timedelta(days=L-1)
print(f"→ Séquence requise de {start_date.date()} à {end_date.date()}")

df_reg = df[df['chemin_directory'] == region]
seq_df = df_reg[(df_reg['date'] >= start_date) & (df_reg['date'] <= end_date)]
seq_df = seq_df.sort_values('date')

if len(seq_df) != L:
    raise RuntimeError(f"Séquence incomplète : {len(seq_df)} lignes (attendu {L}).")

# --- 10. Préparation de l’entrée pour le modèle ---
X_input = seq_df.drop(columns=['chemin_directory','date', TARGET]).values
X_input = X_input.reshape(1, L, -1).astype(np.float32)

# Label réel au jour de transition
y_true = int(df_reg.loc[df_reg['date'] == t_date, TARGET].values[0])

# --- 11. Chargement du modèle et prédiction ---
model = load_model('best_model.keras')
prob  = float(model.predict(X_input)[0,0])
y_pred = 1 if prob >= 0.5 else 0

# --- 12. Affichage du résultat ---
print("\nRésultat de la prédiction :")
print(f"  Probabilité inondation = {prob:.3f}")
print(f"  Classe prédite         = {y_pred}")
print(f"  Label réel             = {y_true}")

if y_pred == y_true:
    print("✅ Le modèle a correctement prévu la transition 0→1.")
else:
    print("❌ Le modèle a manqué la transition 0→1.")
