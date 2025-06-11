#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
comparaison_modeles.py

Script pour comparer les performances des modèles avec et sans historique d'inondation
sur le même jeu de test. Ce script:
1. Charge les deux modèles (enrichi et original)
2. Adapte le jeu de test pour chaque modèle
3. Évalue les deux modèles avec les mêmes métriques
4. Génère des visualisations comparatives
5. Identifie les cas où l'ajout de l'historique des inondations a changé la prédiction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
import os
import sys

# Paramètres globaux
L = 7  # taille de la fenêtre historique (jours)
H = 3  # horizon de prédiction (jours)
TARGET = 'label'

def load_and_preprocess_data(dataset_path, include_history=True):
    """
    Charge et prétraite le jeu de données de test
    
    Args:
        dataset_path: Chemin vers le fichier CSV de test
        include_history: Si True, inclut historique_region, sinon l'exclut
        
    Returns:
        DataFrame prétraité
    """
    print(f"Chargement du jeu de données: {dataset_path}")
    df = pd.read_csv(dataset_path, parse_dates=['date'], dayfirst=True)
    
    # Sélection des colonnes pertinentes
    numeric_feats = [
        'tempmax', 'tempmin', 'temp',
        'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipprob',
        'precipcover', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility',
        'elevation'
    ]
    
    # Ajout de historique_region si nécessaire
    if include_history and 'historique_region' in df.columns:
        numeric_feats.append('historique_region')
        print("La variable 'historique_region' est incluse dans le modèle.")
    else:
        if not include_history:
            print("La variable 'historique_region' est délibérément exclue.")
        else:
            print("ATTENTION: La variable 'historique_region' n'est pas disponible dans le jeu de données.")
    
    # Exclusion des coordonnées lat/long
    if 'latitude_centroid' in df.columns or 'longitude_centroid' in df.columns:
        print("Les coordonnées latitude/longitude sont exclues des deux modèles.")
    
    # Imputation des valeurs manquantes
    for col in numeric_feats:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            df[col].fillna(med, inplace=True)
    
    # One-hot encoding de soil_type
    if 'soil_type' in df.columns:
        df = pd.get_dummies(df, columns=['soil_type'], prefix='soil')
    
    # Standardisation des variables numériques
    scaler = StandardScaler()
    df[numeric_feats] = scaler.fit_transform(df[numeric_feats])
    
    # Tri par région et date
    df = df.sort_values(['chemin_directory', 'date'])
    
    return df

def create_sequences(df, L, H, exclude_cols):
    """
    Crée des séquences temporelles pour le modèle LSTM
    
    Args:
        df: DataFrame prétraité
        L: Longueur de la séquence d'entrée
        H: Horizon de prédiction
        exclude_cols: Colonnes à exclure des features
        
    Returns:
        X: Séquences d'entrée
        y: Labels correspondants
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    Xs, ys = [], []
    for region, grp in df.groupby('chemin_directory'):
        data = grp.reset_index(drop=True)
        for i in range(len(data) - L - H + 1):
            Xs.append(data.iloc[i : i+L][feature_cols].values)
            ys.append(data.iloc[i+L-1 + H][TARGET])
    
    X = np.array(Xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    
    return X, y, feature_cols

def evaluate_model(model, X, y_true, model_name):
    """
    Évalue un modèle et renvoie les métriques
    
    Args:
        model: Modèle Keras chargé
        X: Données d'entrée
        y_true: Labels réels
        model_name: Nom du modèle pour l'affichage
        
    Returns:
        results: Dictionnaire des métriques
        y_pred: Prédictions binaires
        y_prob: Probabilités prédites
    """
    print(f"\nÉvaluation du modèle: {model_name}")
    print(f"Forme des données d'entrée: {X.shape}")
    
    y_prob = model.predict(X).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calcul des métriques
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Spécificité et VPN
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    results = {
        'Modèle': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'AUC': auc,
        'Spécificité': specificity,
        'VPN': npv,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }
    
    # Affichage des résultats
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Spécificité: {specificity:.3f}")
    print(f"VPN: {npv:.3f}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    return results, y_pred, y_prob

def plot_comparison(results_original, results_enriched):
    """
    Génère des visualisations comparatives des deux modèles
    
    Args:
        results_original: Résultats du modèle original
        results_enriched: Résultats du modèle enrichi
    """
    # Comparaison des métriques principales
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'Spécificité', 'VPN']
    values_original = [results_original[m] for m in metrics]
    values_enriched = [results_enriched[m] for m in metrics]
    
    plt.figure(figsize=(12, 8))
    x = range(len(metrics))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], values_original, width, label='Modèle Original')
    plt.bar([i + width/2 for i in x], values_enriched, width, label='Modèle Enrichi')
    
    plt.xlabel('Métrique')
    plt.ylabel('Valeur')
    plt.title('Comparaison des modèles avec et sans historique d\'inondation')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Ajout des valeurs sur les barres
    for i, v in enumerate(values_original):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(values_enriched):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('comparaison_metriques.png', dpi=300)
    plt.show()
    
    # Comparaison des matrices de confusion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Matrice de confusion du modèle original
    cm_orig = np.array([[results_original['TN'], results_original['FP']], 
                        [results_original['FN'], results_original['TP']]])
    
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Pas d\'inondation', 'Inondation'],
                yticklabels=['Pas d\'inondation', 'Inondation'])
    ax1.set_title('Matrice de confusion - Modèle Original')
    ax1.set_xlabel('Prédit')
    ax1.set_ylabel('Réel')
    
    # Matrice de confusion du modèle enrichi
    cm_enr = np.array([[results_enriched['TN'], results_enriched['FP']], 
                      [results_enriched['FN'], results_enriched['TP']]])
    
    sns.heatmap(cm_enr, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Pas d\'inondation', 'Inondation'],
                yticklabels=['Pas d\'inondation', 'Inondation'])
    ax2.set_title('Matrice de confusion - Modèle Enrichi')
    ax2.set_xlabel('Prédit')
    ax2.set_ylabel('Réel')
    
    plt.tight_layout()
    plt.savefig('comparaison_matrices.png', dpi=300)
    plt.show()

def analyze_differences(y_true, y_pred_orig, y_pred_enr, y_prob_orig, y_prob_enr):
    """
    Analyse les cas où les prédictions diffèrent entre les deux modèles
    
    Args:
        y_true: Labels réels
        y_pred_orig: Prédictions du modèle original
        y_pred_enr: Prédictions du modèle enrichi
        y_prob_orig: Probabilités du modèle original
        y_prob_enr: Probabilités du modèle enrichi
    """
    # Identifier les indices où les prédictions diffèrent
    diff_indices = np.where(y_pred_orig != y_pred_enr)[0]
    
    # Cas améliorés: mauvais (orig) -> bon (enrichi)
    improved = np.where((y_pred_orig != y_true) & (y_pred_enr == y_true))[0]
    
    # Cas détériorés: bon (orig) -> mauvais (enrichi)
    deteriorated = np.where((y_pred_orig == y_true) & (y_pred_enr != y_true))[0]
    
    print(f"\nAnalyse des différences entre les modèles:")
    print(f"Nombre total de prédictions: {len(y_true)}")
    print(f"Nombre de cas avec prédictions différentes: {len(diff_indices)} ({len(diff_indices)/len(y_true)*100:.2f}%)")
    print(f"Cas améliorés (mauvais -> bon): {len(improved)} ({len(improved)/len(y_true)*100:.2f}%)")
    print(f"Cas détériorés (bon -> mauvais): {len(deteriorated)} ({len(deteriorated)/len(y_true)*100:.2f}%)")
    
    # Écart moyen des probabilités
    prob_diff = np.abs(y_prob_enr - y_prob_orig)
    avg_prob_diff = np.mean(prob_diff)
    print(f"Écart moyen des probabilités: {avg_prob_diff:.4f}")
    
    # Différence moyenne dans les cas améliorés vs détériorés
    if len(improved) > 0:
        avg_diff_improved = np.mean(np.abs(y_prob_enr[improved] - y_prob_orig[improved]))
        print(f"Écart moyen pour les cas améliorés: {avg_diff_improved:.4f}")
    
    if len(deteriorated) > 0:
        avg_diff_deteriorated = np.mean(np.abs(y_prob_enr[deteriorated] - y_prob_orig[deteriorated]))
        print(f"Écart moyen pour les cas détériorés: {avg_diff_deteriorated:.4f}")
    
    # Visualisation de l'écart des probabilités
    plt.figure(figsize=(10, 6))
    plt.hist(prob_diff, bins=20, alpha=0.7)
    plt.axvline(avg_prob_diff, color='r', linestyle='--', 
                label=f'Écart moyen: {avg_prob_diff:.4f}')
    plt.xlabel('Écart absolu des probabilités')
    plt.ylabel('Nombre de cas')
    plt.title('Distribution des écarts de probabilité entre les modèles')
    plt.legend()
    plt.grid(True)
    plt.savefig('ecarts_probabilites.png', dpi=300)
    plt.show()
    
    # Création d'un tableau des cas les plus significatifs
    if len(diff_indices) > 0:
        # Sélectionner jusqu'à 10 cas avec les plus grands écarts
        n_cases = min(10, len(diff_indices))
        largest_diff_idx = diff_indices[np.argsort(prob_diff[diff_indices])[-n_cases:]]
        
        diff_data = []
        for idx in largest_diff_idx:
            diff_data.append({
                'Index': idx,
                'Label réel': int(y_true[idx]),
                'Prédiction original': int(y_pred_orig[idx]),
                'Prédiction enrichi': int(y_pred_enr[idx]),
                'Probabilité original': y_prob_orig[idx],
                'Probabilité enrichi': y_prob_enr[idx],
                'Écart': abs(y_prob_enr[idx] - y_prob_orig[idx]),
                'Résultat': 'Amélioré' if ((y_pred_orig[idx] != y_true[idx]) and 
                                          (y_pred_enr[idx] == y_true[idx])) else
                            'Détérioré' if ((y_pred_orig[idx] == y_true[idx]) and 
                                           (y_pred_enr[idx] != y_true[idx])) else
                            'Différent'
            })
        
        diff_df = pd.DataFrame(diff_data)
        print("\nCas avec les plus grands écarts de probabilité:")
        print(diff_df)
        
        # Enregistrement dans un fichier CSV
        diff_df.to_csv('cas_differents.csv', index=False)
        print("Résultats enregistrés dans 'cas_differents.csv'")

def main():
    """Fonction principale"""
    print("=== Comparaison des modèles avec et sans historique d'inondation ===\n")
    
    # 1. Chargement des modèles
    try:
        model_original = load_model('modele_inondation_complete.keras')
        print("Modèle original chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle original: {e}")
        print("Tentative avec le fichier best_model.keras...")
        try:
            model_original = load_model('best_model.keras')
            print("Modèle original chargé depuis best_model.keras")
        except:
            print("Impossible de charger le modèle original. Arrêt du programme.")
            sys.exit(1)
    
    try:
        model_enriched = load_model('modele_inondation_enriched.keras')
        print("Modèle enrichi chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle enrichi: {e}")
        print("Tentative avec le fichier best_model_enriched.keras...")
        try:
            model_enriched = load_model('best_model_enriched.keras')
            print("Modèle enrichi chargé depuis best_model_enriched.keras")
        except:
            print("Impossible de charger le modèle enrichi. Arrêt du programme.")
            sys.exit(1)
    
    # 2. Chargement et prétraitement des données
    # Pour le modèle original (sans historique)
    df_original = load_and_preprocess_data('test_dataset.csv', include_history=False)
    
    # Pour le modèle enrichi (avec historique)
    try:
        df_enriched = load_and_preprocess_data('test_dataset_enriched.csv', include_history=True)
    except:
        print("Fichier test_dataset_enriched.csv non trouvé. Utilisation du fichier test_dataset.csv...")
        print("ATTENTION: Les résultats peuvent être biaisés si le dataset n'inclut pas l'historique des inondations.")
        df_enriched = load_and_preprocess_data('test_dataset.csv', include_history=True)
    
    # 3. Création des séquences
    exclude_cols = ['chemin_directory', 'date', TARGET]
    X_orig, y_orig, feature_cols_orig = create_sequences(df_original, L, H, exclude_cols)
    X_enr, y_enr, feature_cols_enr = create_sequences(df_enriched, L, H, exclude_cols)
    
    # Vérification que les labels sont identiques
    if not np.array_equal(y_orig, y_enr):
        print("ATTENTION: Les labels diffèrent entre les deux jeux de données.")
        print("Pour une comparaison équitable, ils devraient être identiques.")
    
    # 4. Évaluation des modèles
    results_orig, y_pred_orig, y_prob_orig = evaluate_model(
        model_original, X_orig, y_orig, "Modèle Original (sans historique)"
    )
    
    results_enr, y_pred_enr, y_prob_enr = evaluate_model(
        model_enriched, X_enr, y_enr, "Modèle Enrichi (avec historique)"
    )
    
    # 5. Visualisations comparatives
    plot_comparison(results_orig, results_enr)
    
    # 6. Analyse des différences
    analyze_differences(y_enr, y_pred_orig, y_pred_enr, y_prob_orig, y_prob_enr)
    
    print("\n=== Fin de la comparaison ===")
    print("Les visualisations ont été enregistrées dans le répertoire courant.")

if __name__ == "__main__":
    main()
