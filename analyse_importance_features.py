#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyse_importance_features.py

Script pour analyser l'importance des features dans le modèle LSTM enrichi en utilisant
différentes techniques:
1. Importance par permutation: mesure l'impact sur les performances du modèle 
   lorsqu'une feature est remplacée par des valeurs aléatoires
2. Analyse comparative des différentes features, y compris historique_region
3. Visualisation des résultats sous différentes formes

Cette analyse complète l'analyse PCA réalisée dans le notebook LSTM_final_enriched.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Paramètres globaux
L = 7       # taille de la fenêtre historique (jours)
H = 3       # horizon de prédiction (jours)
TARGET = 'label'

def load_test_data(test_file='test_dataset_enriched.csv'):
    """
    Charge et prétraite le jeu de test enrichi
    
    Returns:
        X_test: Séquences d'entrée prétraitées
        y_test: Labels correspondants
        feature_cols: Noms des features utilisées
    """
    print(f"Chargement du jeu de test: {test_file}")
    
    try:
        df = pd.read_csv(test_file, parse_dates=['date'], dayfirst=True)
    except FileNotFoundError:
        print(f"Fichier {test_file} non trouvé, tentative avec test_dataset.csv...")
        df = pd.read_csv('test_dataset.csv', parse_dates=['date'], dayfirst=True)
        if 'historique_region' not in df.columns:
            print("Ajout d'une colonne historique_region factice (valeurs à zéro)")
            df['historique_region'] = 0
    
    # Vérification des features disponibles
    print(f"Colonnes disponibles: {df.columns.tolist()}")
    print(f"Nombre de lignes: {len(df)}")
    
    # Sélection des features pertinentes
    features = [
        'tempmax', 'tempmin', 'temp',
        'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipprob',
        'precipcover', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility',
        'elevation', 'soil_type', 'historique_region'
    ]
    
    # Vérification de la présence de historique_region
    if 'historique_region' not in df.columns:
        print("ATTENTION: La colonne 'historique_region' n'est pas présente.")
        features.remove('historique_region')
    
    # Sélection des colonnes pertinentes
    required_cols = ['chemin_directory', 'date'] + features + [TARGET]
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]
    
    # Imputation des valeurs manquantes
    numeric_features = [f for f in features if f != 'soil_type']
    for col in numeric_features:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # One-hot encoding de soil_type
    if 'soil_type' in df.columns:
        df = pd.get_dummies(df, columns=['soil_type'], prefix='soil')
    
    # Standardisation des variables numériques
    numeric_cols = [c for c in df.columns if c not in 
                    ['chemin_directory', 'date', TARGET] and not c.startswith('soil_')]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Tri par région et date
    df = df.sort_values(['chemin_directory', 'date'])
    
    # Création des séquences
    exclude_cols = ['chemin_directory', 'date', TARGET]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Création des séquences temporelles
    X, y = [], []
    for region, grp in df.groupby('chemin_directory'):
        data = grp.reset_index(drop=True)
        for i in range(len(data) - L - H + 1):
            X.append(data.iloc[i:i+L][feature_cols].values)
            y.append(data.iloc[i+L-1+H][TARGET])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Données prétraitées: X shape={X.shape}, y shape={y.shape}")
    return X, y, feature_cols

def load_model_and_weights(model_file='modele_inondation_enriched.keras'):
    """
    Charge le modèle LSTM préentraîné
    
    Returns:
        model: Modèle Keras chargé
    """
    try:
        print(f"Chargement du modèle: {model_file}")
        model = load_model(model_file)
        print("Modèle chargé avec succès")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {model_file}: {e}")
        alternatives = [f for f in os.listdir() if f.endswith('.keras') or f.endswith('.h5')]
        if alternatives:
            print(f"Tentative avec d'autres modèles disponibles: {alternatives}")
            for alt in alternatives:
                try:
                    model = load_model(alt)
                    print(f"Modèle alternatif chargé: {alt}")
                    return model
                except:
                    continue
        raise Exception("Impossible de charger un modèle valide")

def evaluate_baseline(model, X, y):
    """
    Évalue les performances de base du modèle
    
    Returns:
        baseline_metrics: Dictionnaire des métriques de base
    """
    print("Évaluation des performances de base...")
    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    baseline_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_prob)
    }
    
    print(f"Performances de base: Accuracy={baseline_metrics['accuracy']:.3f}, "
          f"F1={baseline_metrics['f1']:.3f}, AUC={baseline_metrics['auc']:.3f}")
    
    return baseline_metrics, y_pred, y_prob

def permutation_importance(model, X, y, feature_cols, baseline_metrics, n_repeats=5):
    """
    Calcule l'importance des features par permutation
    
    Args:
        model: Modèle Keras
        X: Données d'entrée
        y: Labels
        feature_cols: Noms des features
        baseline_metrics: Métriques de référence
        n_repeats: Nombre de répétitions pour chaque feature
        
    Returns:
        importance_df: DataFrame avec les importances calculées
    """
    print(f"\nCalcul de l'importance par permutation (n_repeats={n_repeats})...")
    
    # Initialisation des résultats
    importances = {'feature': [], 'metric': [], 'importance': [], 'std': []}
    metrics = ['accuracy', 'f1', 'auc']
    
    # Parcours des features
    start_time = time.time()
    n_features = X.shape[2]
    
    for feat_idx in range(n_features):
        feat_name = feature_cols[feat_idx]
        print(f"Traitement de la feature {feat_idx+1}/{n_features}: {feat_name}")
        
        # Pour chaque métrique
        for metric in metrics:
            # Répéter n_repeats fois pour la stabilité
            importance_values = []
            
            for _ in range(n_repeats):
                # Copier les données
                X_permuted = X.copy()
                
                # Permuter la feature pour toutes les séquences et tous les pas de temps
                for seq_idx in range(X_permuted.shape[0]):
                    # Permuter les valeurs de cette feature pour cette séquence
                    np.random.shuffle(X_permuted[seq_idx, :, feat_idx])
                
                # Évaluer avec la feature permutée
                y_prob_perm = model.predict(X_permuted, verbose=0).ravel()
                y_pred_perm = (y_prob_perm >= 0.5).astype(int)
                
                # Calculer la métrique
                if metric == 'accuracy':
                    score_permuted = accuracy_score(y, y_pred_perm)
                elif metric == 'f1':
                    score_permuted = f1_score(y, y_pred_perm)
                else:  # auc
                    score_permuted = roc_auc_score(y, y_prob_perm)
                
                # L'importance est la dégradation de la performance
                importance = baseline_metrics[metric] - score_permuted
                importance_values.append(importance)
            
            # Stocker les résultats
            importances['feature'].append(feat_name)
            importances['metric'].append(metric)
            importances['importance'].append(np.mean(importance_values))
            importances['std'].append(np.std(importance_values))
    
    elapsed_time = time.time() - start_time
    print(f"Calcul terminé en {elapsed_time:.1f} secondes")
    
    # Créer un DataFrame avec les résultats
    importance_df = pd.DataFrame(importances)
    
    # Normaliser l'importance pour chaque métrique
    for metric in metrics:
        metric_data = importance_df[importance_df['metric'] == metric]
        max_importance = metric_data['importance'].max()
        if max_importance > 0:
            # Normaliser pour avoir une somme à 100%
            sum_importance = metric_data['importance'].sum()
            importance_df.loc[importance_df['metric'] == metric, 'importance_pct'] = \
                importance_df.loc[importance_df['metric'] == metric, 'importance'] / sum_importance * 100
    
    return importance_df

def plot_feature_importance(importance_df, top_n=15):
    """
    Génère des visualisations de l'importance des features
    
    Args:
        importance_df: DataFrame avec les importances calculées
        top_n: Nombre de features à afficher
    """
    metrics = importance_df['metric'].unique()
    
    # 1. Visualisation par métrique
    for metric in metrics:
        metric_data = importance_df[importance_df['metric'] == metric].copy()
        metric_data = metric_data.sort_values('importance', ascending=False)
        
        # Limiter aux top_n features
        if len(metric_data) > top_n:
            metric_data = metric_data.head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=metric_data)
        plt.title(f'Importance des features par permutation - Métrique: {metric}')
        plt.xlabel('Diminution de performance')
        plt.ylabel('Feature')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig(f'importance_{metric}.png', dpi=300)
        plt.show()
    
    # 2. Visualisation composite (moyenne des métriques)
    avg_importance = importance_df.groupby('feature')['importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('importance', ascending=False)
    
    # Limiter aux top_n features
    if len(avg_importance) > top_n:
        avg_importance = avg_importance.head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=avg_importance)
    plt.title('Importance moyenne des features (toutes métriques confondues)')
    plt.xlabel('Diminution moyenne de performance')
    plt.ylabel('Feature')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('importance_moyenne.png', dpi=300)
    plt.show()
    
    # 3. Heatmap des importances par métrique
    pivot_data = importance_df.pivot(index='feature', columns='metric', values='importance')
    pivot_data = pivot_data.reindex(avg_importance['feature'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title('Importance des features par métrique')
    plt.tight_layout()
    plt.savefig('importance_heatmap.png', dpi=300)
    plt.show()
    
    # 4. Visualisation spécifique de historique_region
    if 'historique_region' in importance_df['feature'].values:
        hist_importance = importance_df[importance_df['feature'] == 'historique_region']
        other_features = importance_df[importance_df['feature'] != 'historique_region']
        
        for metric in metrics:
            hist_value = hist_importance[hist_importance['metric'] == metric]['importance'].values[0]
            metric_data = other_features[other_features['metric'] == metric].copy()
            metric_data = metric_data.sort_values('importance', ascending=False)
            
            # Classement de historique_region
            rank = sum(metric_data['importance'] > hist_value) + 1
            total = len(metric_data) + 1
            
            print(f"Pour la métrique {metric}, historique_region se classe {rank}/{total}")
            
            # Visualisation comparative
            plt.figure(figsize=(10, 6))
            plt.bar(0, hist_value, width=0.4, label='historique_region')
            plt.bar(range(1, 6), metric_data['importance'].head(5), width=0.4, 
                    label='Top 5 autres features')
            plt.xticks(range(6), ['historique_region'] + metric_data['feature'].head(5).tolist())
            plt.title(f'Comparaison de l\'importance - Métrique: {metric}')
            plt.ylabel('Diminution de performance')
            plt.legend()
            plt.grid(True, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'comparaison_historique_{metric}.png', dpi=300)
            plt.show()

def main():
    """Fonction principale"""
    print("=== Analyse de l'importance des features pour le modèle LSTM enrichi ===\n")
    
    # 1. Chargement des données
    X, y, feature_cols = load_test_data()
    
    # 2. Chargement du modèle
    model = load_model_and_weights()
    
    # 3. Évaluation des performances de base
    baseline_metrics, y_pred, y_prob = evaluate_baseline(model, X, y)
    
    # 4. Calcul de l'importance par permutation
    importance_df = permutation_importance(model, X, y, feature_cols, baseline_metrics)
    
    # 5. Visualisation des résultats
    plot_feature_importance(importance_df)
    
    # 6. Sauvegarde des résultats
    importance_df.to_csv('feature_importance_permutation.csv', index=False)
    print("\nRésultats enregistrés dans 'feature_importance_permutation.csv'")
    
    print("\n=== Analyse terminée ===")
    print("Les visualisations ont été enregistrées dans le répertoire courant.")

if __name__ == "__main__":
    main()
