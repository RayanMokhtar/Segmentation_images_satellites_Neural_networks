# Module de Prédiction LSTM pour FloodAI

Ce module implémente la prédiction d'inondations basée sur un modèle LSTM (Long Short-Term Memory) avec attention qui utilise des données météorologiques et géographiques.

## Fonctionnalités

- Prédiction du risque d'inondation pour une localisation spécifique (latitude, longitude)
- Horizon de prédiction configurable (par défaut: 5 jours)
- Deux variantes du modèle : LSTM standard et LSTM avec labels CNN
- Visualisation des résultats avec graphiques de prévision
- Intégration des données historiques d'inondations via la base EM-DAT

## Utilisation

1. Accédez à la page LSTM via le menu de navigation
2. Sélectionnez le modèle à utiliser : LSTM standard ou LSTM avec labels CNN
3. Entrez la date de prédiction souhaitée
4. Entrez les coordonnées géographiques (latitude, longitude)
5. Cliquez sur "Générer la prédiction"

## Architecture technique

Le module est basé sur l'architecture suivante :
- Un modèle LSTM bidirectionnel avec mécanisme d'attention
- Standardisation des caractéristiques numériques et encodage one-hot des catégories
- Caractéristiques cycliques pour capturer les variations saisonnières
- Utilisation de données météorologiques externes via l'API Visual Crossing
- Intégration des données d'élévation via OpenTopoData
- Classification du type de sol via OpenEPI

## Fichiers requis

Les fichiers suivants doivent être présents dans le dossier `models/` :
- `best_model_with_labels_attention.keras` : Modèle LSTM entraîné
- `scaler_train_with_labels_attention.pkl` : Standardisateur des caractéristiques numériques
- `encoder_train_with_labels_attention.pkl` : Encodeur one-hot des caractéristiques catégorielles
- `feats_list_with_labels_attention.pkl` : Liste des caractéristiques utilisées par le modèle

Le fichier suivant doit être présent dans le dossier `static/` :
- `public_emdat_custom_request_2025-05-22_1a78f1da-122a-41fb-9eac-038db183ca0a(1).csv` : Base de données historiques d'inondations (EM-DAT)

## Dépendances

Le module requiert les bibliothèques Python suivantes :
- tensorflow
- numpy
- pandas
- matplotlib
- joblib
- scikit-learn
- requests

## Crédits

- Modèle LSTM avec attention basé sur le travail de [demonstration_incremental_with_labels.py]
- Architecture du réseau inspirée de [modele_avec_labels.py]
