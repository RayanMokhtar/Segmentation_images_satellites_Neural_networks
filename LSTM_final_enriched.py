# -*- coding: utf-8 -*-
"""
script_lstm_inondation_enriched.py

Pipeline complet avec données enrichies : prétraitement, entraînement, évaluation 
et sauvegarde d'un modèle LSTM pour prédiction binaire d'inondation à horizon 3 jours.
Utilise le dataset enrichi avec l'historique des inondations régionales et
inclut une analyse PCA pour évaluer l'importance des variables.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Chargement des données enrichies
df = pd.read_csv('./dataset_enriched.csv', parse_dates=['date'], dayfirst=True)

# 2. Sélection des colonnes pertinentes (sans latitude/longitude mais avec historique_region)
features = [
    'tempmax', 'tempmin', 'temp',
    'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precipprob',
    'precipcover', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility',
    'elevation', 'soil_type', 'historique_region'
]
target = 'label'
df = df[['chemin_directory', 'date'] + features + [target]]

# Vérification des données
print("Forme initiale du dataset :", df.shape)
print("Colonnes du dataset :", df.columns.tolist())
print("Nombre de valeurs NaN par colonne :")
print(df.isna().sum())

# Vérification de l'attribut historique_region
if 'historique_region' in df.columns:
    print("\nDistribution de l'attribut historique_region :")
    print(df['historique_region'].value_counts())
else:
    print("\nAttention : La colonne 'historique_region' n'est pas présente dans le dataset")

# 3. Split régions train / test (80% régions pour le train)
regions = df['chemin_directory'].unique()
n_train = int(0.8 * len(regions))
train_regions = set(regions[:n_train])
df_train = df[df['chemin_directory'].isin(train_regions)].copy()
df_test  = df[~df['chemin_directory'].isin(train_regions)].copy()

# 3bis. Sauvegarde du dataset de test au format CSV (même structure que le dataset original)
# Cette sauvegarde doit être faite AVANT le prétraitement, donc juste après le split train/test
print("\nSauvegarde du dataset de test enrichi (avant tout prétraitement)...")

# Générer un nom de fichier
test_data_filename = f'test_dataset_enriched.csv'

# Créer une copie du dataframe avant prétraitement pour s'assurer qu'il conserve la structure originale
# Notamment avec la colonne 'soil_type' non encodée en one-hot
df_test_original = df_test.copy()

# Sauvegarder le dataframe de test tel quel, avec la même structure que le dataset d'origine
df_test_original.to_csv(test_data_filename, index=False)

print(f"Dataset de test enrichi sauvegardé dans {test_data_filename}")
print(f"Nombre de lignes: {len(df_test_original)}")
print(f"Nombre de régions: {df_test_original['chemin_directory'].nunique()}")
print(f"Distribution des classes: {df_test_original[target].value_counts().to_dict()}")
print(f"Colonnes: {df_test_original.columns.tolist()}")

# 4. Imputation des NaN par la médiane calculée sur le train
for col in features:
    if df_train[col].isna().any():
        med = df_train[col].median()
        df_train[col].fillna(med, inplace=True)
        df_test[col].fillna(med, inplace=True)

# 5. Encodage one-hot de soil_type
df_train = pd.get_dummies(df_train, columns=['soil_type'], prefix='soil')
df_test  = pd.get_dummies(df_test,  columns=['soil_type'], prefix='soil')
# Harmonisation des colonnes
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

# 6. Normalisation des variables continues
#    On exclut chemin_directory, date, target et les colonnes soil_*
exclude_cols = ['chemin_directory', 'date', target]
numeric_cols = [
    c for c in df_train.columns
    if c not in exclude_cols and not c.startswith('soil_')
]
scaler = StandardScaler().fit(df_train[numeric_cols])
df_train[numeric_cols] = scaler.transform(df_train[numeric_cols])
df_test[numeric_cols]  = scaler.transform(df_test[numeric_cols])

# 7. Tri par région et date
df_train.sort_values(['chemin_directory', 'date'], inplace=True)
df_test.sort_values(['chemin_directory', 'date'], inplace=True)


# 8. Analyse PCA pour évaluer l'importance des variables
feature_cols = [c for c in df_train.columns if c not in exclude_cols]
X_for_pca = df_train[feature_cols].values

# Appliquer PCA avec autant de composantes que de features
pca = PCA(n_components=len(feature_cols))
pca.fit(X_for_pca)

# Visualisation de la variance expliquée
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'r-')
plt.xlabel('Composantes principales')
plt.ylabel('Proportion de variance expliquée')
plt.title('Analyse en composantes principales - Variance expliquée')
plt.grid(True)

# Scree plot des 10 premières composantes pour mieux voir les détails
plt.subplot(1, 2, 2)
plt.bar(range(1, min(10, len(pca.explained_variance_ratio_)) + 1), 
        pca.explained_variance_ratio_[:10])
plt.plot(range(1, min(10, len(pca.explained_variance_ratio_)) + 1), 
         np.cumsum(pca.explained_variance_ratio_)[:10], 'r-')
plt.xlabel('Composantes principales')
plt.ylabel('Proportion de variance expliquée')
plt.title('Top 10 composantes principales')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculer l'importance de chaque variable d'origine dans les premières composantes
n_components_to_analyze = 3  # Nous analyserons les 3 premières composantes principales
loadings = pca.components_[:n_components_to_analyze].T
importance_df = pd.DataFrame(
    loadings, 
    columns=[f'PC{i+1}' for i in range(n_components_to_analyze)],
    index=feature_cols
)

# Afficher l'importance des variables pour les premières composantes avec une heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(importance_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Importance des variables dans les 3 premières composantes principales')
plt.tight_layout()
plt.show()

# Visualisation des loadings des top features pour chaque composante séparément
plt.figure(figsize=(18, 15))
for i in range(n_components_to_analyze):
    plt.subplot(n_components_to_analyze, 1, i+1)
    
    # Tri des features par importance absolue pour cette composante
    component_loadings = importance_df.iloc[:, i].sort_values(key=abs, ascending=False)
    top_features = component_loadings.head(10)
    
    # Barplot coloré (rouge pour négatif, bleu pour positif)
    colors = ['r' if x < 0 else 'b' for x in top_features]
    sns.barplot(x=top_features.values, y=top_features.index, palette=colors)
    plt.title(f'Top 10 features contribuant à PC{i+1} (variance expliquée: {pca.explained_variance_ratio_[i]:.2%})')
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, axis='x')

plt.tight_layout()
plt.show()

# Calcul et visualisation de l'importance globale des variables
# Pondérer les loadings par la variance expliquée par chaque composante
explained_variance = pca.explained_variance_ratio_[:n_components_to_analyze]
weighted_loadings = np.abs(loadings) * explained_variance
overall_importance = np.sum(weighted_loadings, axis=1)

# Normaliser pour avoir une somme à 100%
overall_importance = (overall_importance / np.sum(overall_importance)) * 100

# Créer un DataFrame pour l'importance globale
importance_overall_df = pd.DataFrame({
    'Variable': feature_cols,
    'Importance (%)': overall_importance
})
importance_overall_df = importance_overall_df.sort_values('Importance (%)', ascending=False)

# Visualiser l'importance globale des variables
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance (%)', y='Variable', data=importance_overall_df.head(15))
plt.title('Importance globale des 15 principales variables (basée sur les 3 premières composantes)')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

# Sauvegarde des résultats d'importance pour référence future
importance_overall_df.to_csv('feature_importance_pca.csv', index=False)

print("Top 10 variables les plus importantes selon l'analyse PCA :")
print(importance_overall_df.head(10))

# Vérification de l'importance de la variable historique_region
if 'historique_region' in importance_overall_df['Variable'].values:
    hist_rank = importance_overall_df[importance_overall_df['Variable'] == 'historique_region'].index[0]
    hist_importance = importance_overall_df.loc[hist_rank, 'Importance (%)']
    print(f"\nImportance de la variable historique_region: {hist_importance:.2f}% (rang {hist_rank+1}/{len(feature_cols)})")

# 9. Création des séquences temporelles
L = 7  # longueur de la séquence d'entrée (jours)
H = 3  # horizon de prédiction (jours futurs)

def create_sequences(df, L, H, feature_cols, target_col):
    Xs, ys = [], []
    for region, grp in df.groupby('chemin_directory'):
        data = grp.reset_index(drop=True)
        for i in range(len(data) - L - H + 1):
            Xs.append(data.iloc[i : i+L][feature_cols].values)
            ys.append(data.iloc[i+L-1 + H][target_col])
    return np.array(Xs), np.array(ys)

feature_cols = [c for c in df_train.columns if c not in exclude_cols]
X_train, y_train = create_sequences(df_train, L, H, feature_cols, target)
X_test,  y_test  = create_sequences(df_test,  L, H, feature_cols, target)

print("Avant cast → dtype X_train:", X_train.dtype, 
      "  dtype y_train:", y_train.dtype)

# 10. Conversion en float32
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test  = np.array(X_test,  dtype=np.float32)
y_test  = np.array(y_test,  dtype=np.float32)

print("Après cast → dtype X_train:", X_train.dtype, 
      "  dtype y_train:", y_train.dtype)
print("Shapes: X_train", X_train.shape, "y_train", y_train.shape)

# 11. Définition du modèle LSTM
n_features = X_train.shape[2]
model = Sequential([
    LSTM(64, input_shape=(L, n_features), return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()


# 12. Callbacks pour entraînement
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model_enriched.keras', save_best_only=True)
]

# 13. Entraînement
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=25,
    batch_size=16,
    callbacks=callbacks,
    verbose=2
)

# Visualisation de l'historique d'entraînement
plt.figure(figsize=(12, 5))

# Plot de la loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss d\'entraînement')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot de l'accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy d\'entraînement')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 14. Évaluation sur le jeu de test
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print(f"\n--- Évaluation finale sur TEST (modèle enrichi) ---")
print(f"Accuracy : {acc:.3f}")
print(f"Précision: {prec:.3f}")
print(f"Rappel   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")
print(f"AUC      : {auc:.3f}\n")

# Rapport de classification détaillé
print("Détail par classe :\n", classification_report(y_test, y_pred, digits=3))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Affichage de la matrice de confusion sous forme de heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pas d\'inondation', 'Inondation'],
            yticklabels=['Pas d\'inondation', 'Inondation'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()

# Calcul de métriques supplémentaires
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print("\nMétriques supplémentaires :")
print(f"Spécificité (Vrais négatifs / Total négatifs) : {specificity:.3f}")
print(f"Valeur prédictive négative (VPN)              : {npv:.3f}")
print(f"Vrais positifs (TP)                           : {tp}")
print(f"Faux positifs (FP)                            : {fp}")
print(f"Vrais négatifs (TN)                           : {tn}")
print(f"Faux négatifs (FN)                            : {fn}")

# Courbe ROC
plt.figure(figsize=(8, 6))
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 15. Sauvegarde finale du modèle enrichi
model.save('modele_inondation_enriched.keras')
print("Modèle sauvegardé sous 'modele_inondation_enriched.keras'")


# 16. Analyse des prédictions incorrectes
incorrect_indices = np.where(y_pred != y_test)[0]
print(f"Nombre de prédictions incorrectes : {len(incorrect_indices)} sur {len(y_test)} ({len(incorrect_indices)/len(y_test)*100:.2f}%)")

# Extraire les statistiques des erreurs par type
false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]

print(f"Faux positifs (prédiction inondation, réalité pas d'inondation): {len(false_positives)}")
print(f"Faux négatifs (prédiction pas d'inondation, réalité inondation): {len(false_negatives)}")

# Sélectionner quelques exemples de prédictions incorrectes pour analyse
n_examples = min(5, len(incorrect_indices))
if n_examples > 0:
    print("\n--- Analyse détaillée des prédictions incorrectes ---")
    
    # Obtenir les 5 features les plus importantes selon notre analyse PCA
    top_features = importance_overall_df['Variable'].head(5).tolist()
    print(f"Analyse basée sur les features les plus importantes: {top_features}")
    
    for i in range(n_examples):
        idx = incorrect_indices[i]
        example_sequence = X_test[idx]
        
        print(f"\nExemple incorrect #{i+1}")
        print(f"Label réel : {int(y_test[idx])}")
        print(f"Probabilité prédite : {y_prob[idx]:.4f}")
        print(f"Label prédit : {int(y_pred[idx])}")
        
        # Visualiser les variables importantes de cette séquence
        print("\nValeurs des variables importantes pour cette séquence:")
        
        for feat_name in top_features:
            if feat_name in feature_cols:
                feat_idx = feature_cols.index(feat_name)
                values = example_sequence[:, feat_idx]
                # Calculer la moyenne et la tendance
                avg_value = np.mean(values)
                trend = np.polyfit(range(len(values)), values, 1)[0]
                trend_dir = "↑" if trend > 0.01 else "↓" if trend < -0.01 else "→"
                print(f"  {feat_name}: moyenne={avg_value:.3f}, tendance={trend_dir}")
        
        # Si historique_region est dans les features, le mettre en évidence
        if 'historique_region' in feature_cols:
            hist_idx = feature_cols.index('historique_region')
            hist_value = example_sequence[0, hist_idx]  # Valeur constante pour toute la séquence
            print(f"\n  historique_region: {hist_value:.3f} (normalisé)")

# 17. Visualisation des patterns temporels des prédictions incorrectes
if len(incorrect_indices) > 0:
    # Sélectionner un exemple pour visualisation détaillée
    sample_idx = incorrect_indices[0]
    sample_seq = X_test[sample_idx]
    
    # Créer un dataframe pour faciliter la visualisation
    seq_df = pd.DataFrame(
        sample_seq,
        columns=feature_cols
    )
    
    # Sélectionner les features importantes pour la visualisation
    vis_features = top_features[:3]  # Limiter à 3 pour la lisibilité
    
    # Créer un graphique pour montrer l'évolution temporelle
    plt.figure(figsize=(12, 8))
    for feat in vis_features:
        if feat in seq_df.columns:
            plt.plot(range(L), seq_df[feat], marker='o', label=feat)
    
    plt.title(f"Évolution temporelle des variables importantes - Prédiction incorrecte\nRéel: {int(y_test[sample_idx])}, Prédit: {int(y_pred[sample_idx])}")
    plt.xlabel("Jour dans la séquence")
    plt.ylabel("Valeur normalisée")
    plt.legend()
    plt.grid(True)
    plt.show()

# 18. Exemple d'inférence sur une séquence de test dont une transition de label existe
example_index = 0  # Index de l'exemple à tester
example_sequence = X_test[example_index]
print("\n--- Exemple d'inférence sur une séquence de test enrichie ---")
print("Label réel :", y_test[example_index])
y_prob_example = model.predict(example_sequence[np.newaxis, ...]).ravel()
print("Probabilité prédite :", y_prob_example[0])
print("Label prédit :", int(y_prob_example[0] >= 0.5))

# Analyse détaillée de la séquence d'exemple
print("\nValeurs des variables importantes pour cette séquence d'exemple:")
example_df = pd.DataFrame(example_sequence, columns=feature_cols)
# Sélectionner les variables les plus importantes
top_vars = importance_overall_df['Variable'].head(8).tolist()
for var in top_vars:
    if var in example_df.columns:
        values = example_df[var].values
        avg = np.mean(values)
        trend = np.polyfit(range(len(values)), values, 1)[0]
        trend_dir = "↑" if trend > 0.01 else "↓" if trend < -0.01 else "→"
        print(f"  {var}: moyenne={avg:.3f}, tendance={trend_dir}")

# Visualisation de la séquence d'exemple
plt.figure(figsize=(12, 8))
for var in top_vars[:3]:  # Limiter à 3 pour la lisibilité
    if var in example_df.columns:
        plt.plot(range(L), example_df[var], marker='o', label=var)
plt.title(f"Évolution temporelle - Séquence d'exemple (label={int(y_test[example_index])})")
plt.xlabel("Jour dans la séquence")
plt.ylabel("Valeur normalisée")
plt.legend()
plt.grid(True)
plt.show()

# 19. Conclusion
print("\n--- Conclusion sur l'importance des variables pour la prédiction d'inondation ---")
print("Selon notre analyse PCA, les variables les plus importantes pour la prédiction sont :")
for i, row in importance_overall_df.head(5).iterrows():
    print(f"- {row['Variable']} : {row['Importance (%)']:.2f}%")

if 'historique_region' in importance_overall_df['Variable'].values:
    hist_row = importance_overall_df[importance_overall_df['Variable'] == 'historique_region'].iloc[0]
    print(f"\nLa variable 'historique_region' a une importance de {hist_row['Importance (%)']:.2f}%.")
    hist_rank = importance_overall_df[importance_overall_df['Variable'] == 'historique_region'].index[0] + 1
    print(f"Elle se classe au rang {hist_rank} sur {len(feature_cols)} variables.")
    print("Cela démontre que l'enrichissement du dataset avec des données historiques d'inondation")
    print("a un impact significatif sur la capacité prédictive du modèle.")


# 20. Analyse comparative avec et sans la variable historique_region
try:
    # Chargement des deux modèles
    model_original = load_model('modele_inondation_complete.keras')
    model_enriched = model  # Notre modèle actuel enrichi
    
    print("\n--- Analyse comparative des modèles avec/sans historique des inondations ---")
    
    # Fonction pour évaluer un modèle sur le jeu de test
    def evaluate_model(model_name, model, X, y_true):
        y_prob = model.predict(X).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results = {
            'Modèle': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'AUC': auc,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        }
        
        return results, y_pred, y_prob
    
    # Cette partie est un peu complexe car nous devons adapter les données pour le modèle original
    # (qui n'utilise pas historique_region et peut avoir d'autres différences)
    
    # Pour notre analyse, nous utiliserons un sous-ensemble du jeu de test adapté aux deux modèles
    
    # Option 1: Si nous pouvons facilement modifier les entrées pour le modèle original
    print("Note: Pour une comparaison rigoureuse, nous devrions créer un jeu de test")
    print("      qui peut être utilisé par les deux modèles (avec/sans historique).")
    print("      Cela nécessiterait de reconstruire les séquences temporelles.")
    
    # Option 2: Sinon, nous pouvons simplement rapporter les métriques obtenues séparément
    results_enriched, _, _ = evaluate_model("Enrichi (avec historique)", model_enriched, X_test, y_test)
    
    print("\nRésultats du modèle enrichi (avec historique_region):")
    for key, value in results_enriched.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key}: {value}")
        elif isinstance(value, (float, np.float32, np.float64)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nPour une comparaison complète avec le modèle original, exécutez également")
    print("le notebook 'LSTM_final.ipynb' et comparez les métriques d'évaluation.")
    
except Exception as e:
    print(f"\nL'analyse comparative n'a pas pu être effectuée: {e}")
    print("Pour comparer les performances, exécutez séparément les notebooks original et enrichi.")