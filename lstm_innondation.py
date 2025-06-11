import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from keras.models import Sequential, load_model
from keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import AUC
from keras import regularizers
from sklearn.utils import class_weight
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 2. Chargement et analyse du dataset
# Chargement du dataset complet
# df = pd.read_csv('dataset_complet.csv', sep=';') # OLD LINE
df = pd.read_csv('dataset_prepared.csv', sep=',') # MODIFIED: Nom de fichier et séparateur
if 'Unnamed: 0' in df.columns: # AJOUTÉ: Gestion de la colonne d'index potentielle
    df = df.drop(columns=['Unnamed: 0'])
# ...existing code...
# print(df.head())
# print('\\nInformations générales :')
# print(df.info())
# print('\\nStatistiques descriptives :')
# print(df.describe())
print("Aperçu du dataset:") 
print(df.head())
print('\\nInformations générales :\\n') 
df.info() 
print('\\nStatistiques descriptives :\\n') 
print(df.describe(include='all'))

## 3. Filtrage des colonnes pertinentes
# print('Colonnes disponibles :', df.columns.tolist()) 
# Sélection des features météo et statiques
# weather_cols = ['tempmax', 'tempmin', 'precip', 'humidity', 'windspeed', 'pressure', 'cloudcover'] # OLD
# static_cols = ['latitude_centroid', 'longitude_centroid'] # OLD
# target_col = 'label'
# # longueur de la séquence et horizon de prédiction
# SEQ_LEN = 10  # nombre de pas temporels en entrée # OLD
# HORIZON = 7   # prédiction à HORIZON jours # OLD
# features = weather_cols + static_cols # OLD
# df_sel = df[['chemin_directory', 'date'] + weather_cols + static_cols + [target_col]].copy() # OLD
# df_sel['date'] = pd.to_datetime(df_sel['date'], dayfirst=True) # OLD
# df_sel.sort_values(['chemin_directory','date'], inplace=True) # OLD
# print('Colonnes disponibles :', df.columns.tolist()) # OLD, repeated
# print(df_sel.head()) # OLD
print('\\nColonnes disponibles après chargement :', df.columns.tolist())

# MODIFIÉ: Définition des colonnes à utiliser et à exclure
# Colonnes à exclure explicitement de l'entraînement direct
# (certaines comme 'date' et 'chemin_directory' sont utilisées pour le traitement)
cols_to_exclude_from_features = [
    'Unnamed: 0', 'geometry', 'tile_number', 'id', 'location_id', 
    'image_dir', 'date_satellites_S1', 'nouveau_chemin', 
    'latitude_centroid', 'longitude_centroid', # Exclues des features directes
    'source', 'timezone', 'folder'
]

# Colonnes météorologiques à utiliser (restreintes)
weather_cols = [col for col in ['temp', 'feelslike', 'dew', 'humidity', 'precipprob', 'precipcover', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility']
               if col in df.columns]

# Colonnes statiques à utiliser (uniquement l'élévation comme demandé)
static_cols = ['elevation']
static_cols = [col for col in static_cols if col in df.columns]

# Colonne catégorielle à utiliser
categorical_cols = ['soil_type']
categorical_cols = [col for col in categorical_cols if col in df.columns]

target_col = 'label' 

SEQ_LEN = 6  
# Horizon de prédiction en jours (au lieu de 7, maintenant 3 jours)
HORIZON = 3

# Colonnes à conserver initialement pour le traitement
# Inclut celles nécessaires pour le groupement/tri, les features, et la target.
initial_cols_to_keep = ['chemin_directory', 'date'] + weather_cols + static_cols + categorical_cols + [target_col]
# Filtrer pour ne garder que les colonnes existantes dans le DataFrame
initial_cols_to_keep = [col for col in initial_cols_to_keep if col in df.columns]
df_processed = df[initial_cols_to_keep].copy()

print(f"\\nColonnes conservées pour le traitement initial: {df_processed.columns.tolist()}")

df_processed['date'] = pd.to_datetime(df_processed['date'], format='%d/%m/%Y') 
df_processed.sort_values(['chemin_directory', 'date'], inplace=True)

# AJOUTÉ: Encodage One-Hot pour 'soil_type'
if 'soil_type' in df_processed.columns:
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, prefix='soil', dummy_na=False) # dummy_na=False pour ne pas créer de colonne pour NaN
    soil_one_hot_cols = [col for col in df_processed.columns if col.startswith('soil_')]
    print(f"\\nColonnes après encodage One-Hot de 'soil_type': {soil_one_hot_cols}")
else:
    soil_one_hot_cols = []
    print("\\nLa colonne 'soil_type' n'est pas présente, pas d'encodage One-Hot.")

# Imputation des données manquantes
numerical_features_to_impute = weather_cols + static_cols 
print(f"\\nImputation des valeurs manquantes pour les colonnes numériques: {numerical_features_to_impute}")
for col in numerical_features_to_impute: 
    if col in df_processed.columns: 
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        # S'assurer que le groupement ne se fait pas sur une colonne qui n'existe plus si elle a été enlevée
        if 'chemin_directory' in df_processed.columns:
            df_processed[col] = df_processed.groupby('chemin_directory')[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both').bfill().ffill()
            )
        else: # Au cas où chemin_directory aurait été retiré par erreur plus tôt
            df_processed[col] = df_processed[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

        if df_processed[col].isnull().any():
            print(f"Attention: NaN restants dans {col} après interpolation, remplissage par la moyenne globale.")
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    else:
        print(f"Attention: La colonne {col} n'est pas présente dans df_processed pour l'imputation.")

# MODIFIÉ: Définition des colonnes pour les séquences X
# Inclut les features météo, statiques (elevation), soil_type (one-hot) et le label des jours précédents
feature_columns_for_X = weather_cols + static_cols + soil_one_hot_cols
# S'assurer que toutes les colonnes existent dans df_processed
feature_columns_for_X = [col for col in feature_columns_for_X if col in df_processed.columns]
print(f"\\nColonnes utilisées pour construire les séquences X: {feature_columns_for_X}")

# AJOUTÉ: Vérification des NaN après imputation
print("\\nNaN restants avant création des séquences (uniquement les colonnes pour X):")
if feature_columns_for_X: # Vérifier que la liste n'est pas vide
    print(df_processed[feature_columns_for_X].isnull().sum().sort_values(ascending=False))
    # Optionnel mais recommandé: supprimer les lignes où des NaN persistent dans les features ou la target
    df_processed.dropna(subset=feature_columns_for_X + [target_col], inplace=True)
    print(f"Forme de df_processed après dropna: {df_processed.shape}")
else:
    print("Aucune colonne de feature sélectionnée pour X, vérifiez les étapes précédentes.")


# Construction des fenêtres glissantes (pas=1) pour chaque région
# def build_xy_windows(df_sel): # OLD FUNCTION
#     X, y = [], []
#     for r, grp in df_sel.groupby('chemin_directory'):
#         arr = grp.sort_values('date')[features].values
#         lbl = grp.sort_values('date')[target_col].values
#         for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
#             X.append(arr[i:i+SEQ_LEN])
#             y.append(lbl[i+SEQ_LEN+HORIZON-1])
#     return np.array(X), np.array(y)
# X_all, y_all = build_xy_windows(df_sel) # OLD CALL
# print('Total sequences :', X_all.shape[0]) # OLD

# Fonction pour construire les séquences X et y avec horizon
def build_sequences(df_data, feature_cols_for_x_step, target_label_col, seq_length, horizon):
    X_list, y_list, y_prev_list = [], [], []
    for _, group in df_data.groupby('chemin_directory'):
        features_values = group[feature_cols_for_x_step].values
        target_values = group[target_label_col].values
        if len(features_values) >= seq_length + horizon:
            for i in range(len(features_values) - seq_length - horizon + 1):
                X_list.append(features_values[i : i + seq_length])
                y_list.append(target_values[i + seq_length + horizon - 1])
                y_prev_list.append(target_values[i + seq_length + horizon - 2])
    return np.array(X_list), np.array(y_list), np.array(y_prev_list)

print(f"\nCréation des séquences avec SEQ_LEN={SEQ_LEN} et HORIZON={HORIZON}...")
# Unpack sequences avec labels précédents
X_all, y_all, y_prev_all = build_sequences(df_processed, feature_columns_for_X, target_col, SEQ_LEN, HORIZON)

print('Total des séquences créées :', X_all.shape[0]) # MODIFIED PRINT (from old)
if X_all.shape[0] == 0: # AJOUTÉ: Vérification si des séquences sont créées
    print("Aucune séquence n'a pu être créée. Vérifiez les données et SEQ_LEN.")
    exit()
print('Shape de X_all :', X_all.shape) # AJOUTÉ
print('Shape de y_all :', y_all.shape) # AJOUTÉ
print('Shape de y_prev_all :', y_prev_all.shape) # AJOUTÉ


# 3) Split temporel (60/20/20)
n = len(X_all)
idx1 = int(n * 0.6)
idx2 = int(n * 0.8)
X_train, y_train = X_all[:idx1], y_all[:idx1]
X_val,   y_val   = X_all[idx1:idx2], y_all[idx1:idx2]
X_test,  y_test  = X_all[idx2:], y_all[idx2:]
# Labels du jour précédent pour évaluation
y_prev_test = y_prev_all[idx2:]
print('Train/Val/Test shapes :', X_train.shape, X_val.shape, X_test.shape)

# AJOUTÉ: Application de SMOTE sur l'ensemble d'entraînement
print("\nApplication de SMOTEENN sur l'ensemble d'entraînement...")
print('Distribution y_train avant SMOTEENN:', Counter(y_train))
smote_enn = SMOTEENN(random_state=42)
# Remodeler X_train pour SMOTEENN: (nombre_samples, SEQ_LEN * nombre_features)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_reshaped, y_train)
# Remodeler à forme originale: (samples, SEQ_LEN, features)
X_train_res = X_train_res.reshape(X_train_res.shape[0], SEQ_LEN, -1)
print('Shape de X_train après SMOTEENN :', X_train_res.shape)
print('Distribution y_train après SMOTEENN:', Counter(y_train_res))

# 4) Afficher distribution des classes
from collections import Counter
print('Distribution y_train :', Counter(y_train)) # Ceci affichera la distribution avant SMOTE, ce qui est bien pour comparer

# 5) Normaliser sur X_train_smote seulement
scaler = MinMaxScaler()
# Ajuster le scaler sur les données d'entraînement après SMOTE
flat_smote = X_train_res.reshape(-1, X_train_res.shape[2])
scaler.fit(flat_smote)
# Sauvegarde du scaler entraîné pour inférence ultérieure
joblib.dump(scaler, 'min_max_scaler_v3.gz')

def scale_set(X_set_to_scale, scaler_fitted): # Nom de la fonction et arguments modifiés pour plus de clarté
    return scaler_fitted.transform(X_set_to_scale.reshape(-1, X_set_to_scale.shape[2])).reshape(X_set_to_scale.shape)

X_train = scale_set(X_train_res, scaler) # X_train est maintenant X_train_smote normalisé
X_val   = scale_set(X_val, scaler)
X_test  = scale_set(X_test, scaler)

# 6) Calcul du class_weight - Moins critique si SMOTE est utilisé, mais peut être conservé pour info
# from sklearn.utils import class_weight # Assurer l'import # Déjà importé
# cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# class_weight_dict = dict(zip(np.unique(y_train), cw))
# print('Class weights :', class_weight_dict)

## 6. Construction de l'architecture LSTM
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQ_LEN, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['AUC', 'accuracy']
) 
model.summary()

## 7. Entraînement du modèle
# MODIFIÉ: Paramètres EarlyStopping et entraînement
early = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
history = model.fit(
    X_train, y_train_res, # Utiliser y_train_smote ici
    validation_data=(X_val, y_val),
    epochs=50, # Augmentation des époques
    batch_size=32,
    # class_weight={0:1.0,1:3.0}, # RETIRÉ: SMOTE gère le déséquilibre
    callbacks=[early, reduce_lr]
)

## 8. Évaluation des performances du modèle
# y_pred = (model.predict(X_test).max(axis=1) > 0.5).astype(int) # OLD
# print(classification_report(y_test, y_pred)) # OLD
# print('AUC :', roc_auc_score(y_test, model.predict(X_test).max(axis=1))) # OLD
# print('Accuracy :', np.mean(y_pred == y_test)) # OLD

y_pred_proba = model.predict(X_test) # NOUVELLE logique de prédiction
# Recherche du meilleur seuil sur le jeu de validation
print("\nRecherche du meilleur seuil sur validation...")
val_proba = model.predict(X_val)
best_thr, best_f1 = 0.5, 0
for thr in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_val, (val_proba > thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print(f"Seuil optimal: {best_thr:.2f} avec F1={best_f1:.4f}")
# Application du seuil optimal
y_pred = (y_pred_proba > best_thr).astype(int)

print(classification_report(y_test, y_pred)) # Inchangé mais contexte de y_pred change
print('AUC :', roc_auc_score(y_test, y_pred_proba)) # MODIFIÉ: utilise y_pred_proba
print('Accuracy :', accuracy_score(y_test, y_pred)) # MODIFIÉ: utilise accuracy_score
cm = confusion_matrix(y_test, y_pred) # AJOUTÉ
print('Matrice de confusion :\\n', cm) # AJOUTÉ

# Visualisation de l'évolution de la perte et de la précision
plt.figure(figsize=(12, 6)) # MODIFIÉ: Taille
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss') # OLD
plt.plot(history.history['val_loss'], label='Validation Loss') # MODIFIÉ: Label
# plt.title('Loss') # OLD
plt.title('Évolution de la Perte') # MODIFIÉ: Titre
# plt.xlabel('Epochs') # OLD
plt.xlabel('Époques') # MODIFIÉ: Label
# plt.ylabel('Loss') # OLD
plt.ylabel('Perte') # MODIFIÉ: Label
plt.legend()
plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Train Accuracy') # OLD
# plt.plot(history.history['val_accuracy'], label='Val Accuracy') # OLD
# plt.title('Accuracy') # OLD
# plt.xlabel('Epochs') # OLD
# plt.ylabel('Accuracy') # OLD
# plt.legend()
# NOUVELLE logique pour afficher la bonne métrique (accuracy ou AUC)
metric_to_plot = ''
if 'accuracy' in history.history:
    metric_to_plot = 'accuracy'
elif 'auc' in history.history: # Gérer les variations de nom pour AUC (ex: auc_1)
    auc_keys = [key for key in history.history.keys() if 'auc' in key.lower() and 'val' not in key.lower()]
    if auc_keys:
        metric_to_plot = auc_keys[0]

if metric_to_plot:
    val_metric_to_plot = f'val_{metric_to_plot}'
    if metric_to_plot in history.history and val_metric_to_plot in history.history:
        plt.plot(history.history[metric_to_plot], label=f'Train {metric_to_plot.capitalize()}')
        plt.plot(history.history[val_metric_to_plot], label=f'Validation {metric_to_plot.capitalize()}')
        plt.title(f'Évolution de {metric_to_plot.capitalize()}')
        plt.xlabel('Époques')
        plt.ylabel(metric_to_plot.capitalize())
        plt.legend()
    else:
        print(f"Clés de métrique '{metric_to_plot}' ou '{val_metric_to_plot}' non trouvées dans history.history.")
else:
    print("Métrique ('accuracy' ou 'auc') non trouvée dans history.history pour le graphique.")

plt.tight_layout()
plt.show()

## 9. Enregistrement du modèle pour une utilisation ultérieure
# model.save('lstm_inondation.h5') # OLD
# model.save('lstm_inondation_v2.h5') # MODIFIÉ: Nom de modèle
model.save('lstm_inondation_v3.h5') # MODIFIÉ: Nom de modèle v3
# print('Modèle enregistré : lstm_inondation.h5') # OLD
# print('Modèle enregistré : lstm_inondation_v2.h5') # MODIFIÉ: Message
print('Modèle enregistré : lstm_inondation_v3.h5') # MODIFIÉ: Message v3


# ## 10. Test avec un exemple spécifique et Évaluation des transitions 0 -> 1 sur X_test
print("\n## 10. Test et Évaluation des transitions 0 -> 1 sur X_test")

# --- Chargement du modèle et vérification du scaler ---
model_path = 'lstm_inondation_v3.h5'
try:
    model = load_model(model_path)
    print(f"Modèle '{model_path}' chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle '{model_path}': {e}")
    print("Assurez-vous que le modèle a été correctement sauvegardé et que le chemin est correct.")
    exit() # Quitter si le modèle ne peut pas être chargé

# Vérification de l'existence et de l'état de l'objet 'scaler'
# Ce scaler doit être celui qui a été 'fit' sur les données d'entraînement du modèle v3 (X_train_smote).
if 'scaler' not in globals(): # Vérifier si scaler est dans la portée globale
    print("ERREUR: L'objet 'scaler' n'est pas défini dans la portée globale.")
    print("Si le scaler a été sauvegardé séparément (ex: avec joblib), chargez-le ici.")
    # Exemple: import joblib; scaler = joblib.load('min_max_scaler_v3.gz')
    exit()
# Vérification basique si le scaler a été fitté (pour MinMaxScaler)
if not hasattr(scaler, 'data_min_') or not hasattr(scaler, 'data_max_'):
     print("ATTENTION: Le scaler ne semble pas avoir été ajusté ('fit') correctement ou est d'un type inattendu.")
     print("Les résultats de la mise à l'échelle pourraient être incorrects.")
     # exit() # Optionnel: arrêter si le scaler n'est pas bon

# --- Évaluation généralisée des transitions 0 -> 1 sur le jeu de test (X_test) ---
print("\n## Évaluation spécifique des transitions 0 -> 1 sur le jeu de test (X_test)")

# Vérifier la disponibilité de X_test, y_test et y_prev_test
if 'X_test' not in globals() or 'y_test' not in globals() or 'y_prev_test' not in globals():
    print("ERREUR: X_test, y_test ou y_prev_test manquant. Exécutez tout le pipeline.")
    exit()
if X_test.shape[0] == 0:
    print("ERREUR: X_test est vide.")
    exit()

# Prédiction et évaluation des transitions 0->1
print(f"Analyse de {len(y_test)} échantillons pour transitions 0->1 (prev=0)...")
y_pred_proba_test = model.predict(X_test)
thr = globals().get('best_thr', 0.5)
y_pred_label_test = (y_pred_proba_test > thr).astype(int).flatten()
TP = FP = FN = 0
for actual, prev, pred in zip(y_test, y_prev_test, y_pred_label_test):
    if prev == 0:
        if actual == 1:
            if pred == 1: TP += 1
            else: FN += 1
        else:
            if pred == 1: FP += 1
total0 = sum(y_prev_test == 0)
print(f"\n--- Transitions 0->1 (jours précédents=0) : {total0} cas ---")
print(f"TP={TP}, FP={FP}, FN={FN}")
precision_t = TP/(TP+FP) if (TP+FP)>0 else 0
recall_t = TP/(TP+FN) if (TP+FN)>0 else 0
f1_t = 2*precision_t*recall_t/(precision_t+recall_t) if (precision_t+recall_t)>0 else 0
print(f"Précision transitions: {precision_t:.4f}, Rappel: {recall_t:.4f}, F1: {f1_t:.4f}")

print("\nFin du script d'entraînement et d'évaluation.")