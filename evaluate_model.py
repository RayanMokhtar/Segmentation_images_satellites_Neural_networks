# -*- coding: utf-8 -*-
"""
test_all_transitions_full_eval.py

Pour chaque région du dataset :
  1. Détecter la première transition 0→1 de 'label'
  2. Construire la séquence historique des L jours
     précédant cette transition (moins l’horizon H)
  3. Faire une prédiction (probabilité + classe)
Rassembler toutes les prédictions et calculer :
  - TP, FN, FP, TN
  - accuracy, precision, recall, F1-score, AUC ROC (si possible)
  - rapport détaillé
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

# 1. Paramètres (les mêmes que pour l'entraînement et le test unitaire)
L = 7       # longueur de la fenêtre historique
H = 3       # horizon de prédiction (jours après la fenêtre)
TARGET = 'label'

# 2. Colonnes météo / topo
features = [
    'latitude_centroid','longitude_centroid',
    'tempmax','tempmin','temp','feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precipprob','precipcover',
    'windspeed','winddir','pressure','cloudcover','visibility',
    'elevation','soil_type'
]

# 3. Chargement du dataset préparé
df = pd.read_csv(
    './test_sequences_evaluation.csv',
    usecols=['chemin_directory','date'] + features + [TARGET],
    parse_dates=['date'], dayfirst=True
)

# 4. Imputation des NaN sur variables numériques uniquement
numeric_feats = [f for f in features if f != 'soil_type']
for col in numeric_feats:
    df[col].fillna(df[col].median(), inplace=True)

# 5. One-hot encoding du type de sol
df = pd.get_dummies(df, columns=['soil_type'], prefix='soil')

# 6. Normalisation des variables numériques continues
scaler = StandardScaler().fit(df[numeric_feats])
df[numeric_feats] = scaler.transform(df[numeric_feats])

# 7. Tri par région puis date
df.sort_values(['chemin_directory','date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 8. Chargement du modèle entraîné
model = load_model('best_model.keras')

# 9. Pour chaque région, détecter et prédire la première transition 0→1
y_trues = []
y_preds = []
y_probs = []

for region, grp in df.groupby('chemin_directory'):
    grp = grp.reset_index(drop=True)
    # recherche du premier indice où label passe 0→1
    idx = None
    for i in range(1, len(grp)):
        if grp.loc[i-1, TARGET] == 0 and grp.loc[i, TARGET] == 1:
            idx = i
            break
    if idx is None:
        continue  # pas de transition dans cette région

    # dates de la fenêtre historique
    date_trans = grp.loc[idx, 'date']
    end_date   = date_trans - pd.Timedelta(days=H)
    start_date = end_date - pd.Timedelta(days=L-1)

    # extraction de la séquence
    seq = grp[(grp['date'] >= start_date) & (grp['date'] <= end_date)]
    if len(seq) != L:
        continue  # séquence incomplète : on skip

    # préparation de l'entrée
    X = seq.drop(columns=['chemin_directory','date',TARGET]).values
    X = X.reshape(1, L, -1).astype(np.float32)

    # prédiction
    prob = float(model.predict(X)[0,0])
    pred = 1 if prob >= 0.5 else 0

    # stockage
    y_trues.append(1)     # par construction, c'est toujours la classe 1
    y_preds.append(pred)
    y_probs.append(prob)

# 10. Calcul de la confusion et des métriques
# Note : y_trues ne contient que des 1, donc FP=TN=0 par design
TP = sum(1 for t,p in zip(y_trues, y_preds) if t==1 and p==1)
FN = sum(1 for t,p in zip(y_trues, y_preds) if t==1 and p==0)
FP = sum(1 for t,p in zip(y_trues, y_preds) if t==0 and p==1)
TN = sum(1 for t,p in zip(y_trues, y_preds) if t==0 and p==0)

total = len(y_trues)
accuracy  = (TP + TN) / total if total>0 else np.nan
precision = TP / (TP + FP) if (TP + FP)>0 else np.nan
recall    = TP / (TP + FN) if (TP + FN)>0 else np.nan
f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall)>0 else np.nan

# AUC ROC uniquement si y_trues contient au moins 2 classes
try:
    auc = roc_auc_score(y_trues, y_probs)
except ValueError:
    auc = np.nan

# 11. Affichage des résultats
print("\n=== Évaluation sur toutes les premières transitions 0→1 ===")
print(f"Régions testées                : {total}")
print(f"TP (vrais positifs)            : {TP}")
print(f"FN (faux négatifs)             : {FN}")
print(f"FP (faux positifs)             : {FP}")
print(f"TN (vrais négatifs)            : {TN}\n")

print(f"Accuracy  : {accuracy:.3f}")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1-score  : {f1:.3f}")
print(f"AUC ROC   : {auc if not np.isnan(auc) else 'n/a'}\n")

print("--- Rapport détaillé ---")
# classification_report gérera l'absence de classe 0 ou 1
print(classification_report(y_trues, y_preds, digits=3))
