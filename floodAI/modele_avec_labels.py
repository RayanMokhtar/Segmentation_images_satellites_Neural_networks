#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
modele_avec_labels.py

Modèle LSTM qui utilise les labels historiques (statut d'inondation des jours précédents)
comme une caractéristique d'entrée supplémentaire pour améliorer la prédiction.
"""
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, GRU, # Ajout de GRU
    BatchNormalization, GaussianNoise, SpatialDropout1D,
    Attention, GlobalAveragePooling1D
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit

# --- HYPERPARAMÈTRES (AJUSTÉS POUR LA ROBUSTESSE) ---
WINDOW_SIZE = 7
STRIDE      = 1
BATCH_SIZE  = 32   # Gradient plus stable
EPOCHS      = 50
PATIENCE    = 40

# --- Caractéristiques météo brutes (retour à la version originale) ---
TEMP_FEATS = [
    'tempmax','tempmin','temp',
    'feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precipprob','precipcover',
    'windspeed','winddir','pressure','cloudcover','visibility'
]

# --- Attributs statiques ---
STATIC_NUM = ['elevation', 'historique_region']

# --- Type de sol (catégorie) ---
CAT_FEAT = 'soil_type'

# --- Encodage cyclique du jour ---
CYCLE_FEATS = ['day_of_year_sin','day_of_year_cos']

# --- Rolling features sur 7 jours (retour à la version originale) ---
ROLL_BASE  = ['precipprob','humidity','precipcover']
ROLL_FEATS = [f'rolling_{c}_mean' for c in ROLL_BASE] + \
             [f'rolling_{c}_std'  for c in ROLL_BASE]

# --- Label comme feature ---
LABEL_FEAT = ['label']

# --- Chemins (mis à jour pour le nouveau modèle avec attention) ---
CSV_PATH      = 'dataset_enriched.csv'
MODEL_PATH    = 'best_model_with_labels_attention.keras'
SCALER_PATH   = 'scaler_train_with_labels_attention.pkl'
ENCODER_PATH  = 'encoder_train_with_labels_attention.pkl'
FEATS_PATH    = 'feats_list_with_labels_attention.pkl'

# Focal loss pour se concentrer sur les exemples difficiles
def focal_loss(alpha=0.6, gamma=2.0): # Alpha rééquilibré
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.math.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss

def preprocess(df: pd.DataFrame):
    # 1) Dates & tri
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['chemin_directory','date'])

    # NOTE: La création de la feature 'precip' a été supprimée.

    # 2) Encodage cyclique
    doy = df['date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2*np.pi*doy/365)
    df['day_of_year_cos'] = np.cos(2*np.pi*doy/365)

    # 3) Rolling features
    for c in ROLL_BASE:
        grp = df.groupby('chemin_directory')[c]
        df[f'rolling_{c}_mean'] = grp.transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).mean())
        df[f'rolling_{c}_std'] = grp.transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).std().fillna(0))
    df[ROLL_FEATS] = df[ROLL_FEATS].bfill()

    # 4) Standardisation des features continues
    cont_feats = TEMP_FEATS + STATIC_NUM + CYCLE_FEATS + ROLL_FEATS
    for col in cont_feats:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    scaler = StandardScaler()
    df[cont_feats] = scaler.fit_transform(df[cont_feats])

    # 5) One-hot encoding de 'soil_type'
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    soil_arr = enc.fit_transform(df[[CAT_FEAT]])
    soil_cols = [f"{CAT_FEAT}_{v}" for v in enc.categories_[0]]
    df[soil_cols] = soil_arr

    # 6) Liste finale des features
    feats = cont_feats + soil_cols + LABEL_FEAT
    return df, feats, scaler, enc

def make_sequences(df: pd.DataFrame, feats, window=WINDOW_SIZE, stride=STRIDE):
    X, y = [], []
    for _, g in df.groupby('chemin_directory'):
        g = g.sort_values('date')
        # 'data' contient maintenant toutes les features, y compris le label historique
        data = g[feats].values
        # 'labels' est toujours la colonne 'label' seule, pour la cible
        labels = g['label'].values
        for i in range(0, len(g) - window, stride):
            X.append(data[i:i+window])
            y.append(labels[i+window]) # La cible est le label du jour suivant la fenêtre
    return np.array(X), np.array(y)

def build_model(time_steps, n_feats):
    """
    Construit une architecture récurrente robuste avec Bi-LSTM, Bi-GRU, Attention
    et une tête de classification améliorée pour une meilleure performance.
    Le nombre de features (n_feats) inclut le label historique.
    """
    inp = Input(shape=(time_steps, n_feats), name='input_seq')

    # Régularisation en entrée
    x = GaussianNoise(0.05)(inp)
    x = SpatialDropout1D(0.25)(x)

    # Couche Bi-LSTM principale pour capturer les dépendances temporelles complexes
    x = Bidirectional(LSTM(80, return_sequences=True,
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                  recurrent_dropout=0.25))(x)
    x = BatchNormalization()(x)

    # NOUVEAU: Couche Bi-GRU pour une capture efficace des motifs temporels
    x = Bidirectional(GRU(40, return_sequences=True,
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                recurrent_dropout=0.25))(x)
    x = BatchNormalization()(x)

    # Mécanisme d'Attention pour pondérer les pas de temps les plus pertinents
    attn_out = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn_out)

    # Tête de classification améliorée
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout augmenté pour réduire les faux positifs
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inp, out)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Taux d'apprentissage plus fin
        loss=focal_loss(),
        metrics=[
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def train():
    # Chargement & prétraitement
    df = pd.read_csv(CSV_PATH)
    df, feats, scaler, enc = preprocess(df)

    # Sauvegarde des objets de prétraitement
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(enc,    ENCODER_PATH)
    joblib.dump(feats,  FEATS_PATH)

    # Construction des séquences
    X, y = make_sequences(df, feats)
    if X.shape[0] == 0:
        print("Aucune séquence n'a pu être créée.")
        return
        
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"Nombre de features en entrée (incluant le label) : {X.shape[2]}")

    # Cross-validation temporelle
    tss = TimeSeriesSplit(n_splits=5) # Augmentation du nombre de splits
    best_recall = 0.0

    for fold, (train_idx, val_idx) in enumerate(tss.split(X), start=1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        print(f"\n--- Fold {fold} ---")
        print(f"Train samples: {len(X_tr)}, Validation samples: {len(X_va)}")

        model = build_model(WINDOW_SIZE, X.shape[2])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_precision', mode='max', # NOUVEAU: On surveille la précision
                patience=PATIENCE, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_PATH, monitor='val_precision', # NOUVEAU: On sauvegarde le meilleur modèle en précision
                mode='max', save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2,
                patience=7, min_lr=1e-6, verbose=1
            )
        ]

        # Pondération ajustée pour pénaliser les faux positifs
        class_weight = {0: 5.0, 1:12.0}

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2
        )

        if 'val_recall' in history.history and history.history['val_recall']:
            fold_recall = max(history.history['val_recall'])
            print(f"Fold {fold} — Best validation recall: {fold_recall:.4f}")
            if fold_recall > best_recall:
                best_recall = fold_recall
        else:
            print(f"Fold {fold} — No validation recall recorded.")

    print(f"\nMeilleur recall en validation sur tous les folds : {best_recall:.4f}")

if __name__ == '__main__':
    train()
