# Prédiction d'inondations avec LSTM et historique des inondations

Ce projet enrichit un modèle LSTM de prédiction d'inondations en ajoutant l'historique des inondations par région comme nouvelle feature. Le modèle est conçu pour prédire les inondations à un horizon de 3 jours sur la base d'une séquence temporelle de 7 jours de données.

## Fichiers principaux

- `LSTM_final_enriched.ipynb` : Notebook principal contenant l'ensemble du pipeline (prétraitement, analyse PCA, entraînement, évaluation)
- `enrichir_historique.py` : Script pour enrichir le dataset avec l'historique des inondations par région
- `test_transitions_enriched.py` : Script pour tester le modèle enrichi sur les transitions d'inondation
- `comparaison_modeles.py` : Script pour comparer les performances des modèles avec et sans historique
- `analyse_importance_features.py` : Script pour analyser l'importance des features par permutation

## Pipeline enrichi

Le pipeline enrichi comprend les étapes suivantes :

1. **Prétraitement des données**
   - Chargement du dataset enrichi avec l'historique des inondations
   - Retrait des coordonnées géographiques (latitude/longitude)
   - Ajout de la feature "historique_region"
   - Imputation des valeurs manquantes, encodage one-hot, normalisation

2. **Analyse PCA pour l'importance des features**
   - Évaluation de la contribution de chaque feature aux composantes principales
   - Visualisation de l'importance globale des features
   - Analyse spécifique de la contribution de "historique_region"

3. **Modélisation LSTM**
   - Architecture : LSTM(64) + Dropout(0.2) + Dense(1, sigmoid)
   - Entraînement avec early stopping et sauvegarde du meilleur modèle
   - Évaluation sur le jeu de test avec diverses métriques

4. **Analyse des prédictions**
   - Analyse détaillée des prédictions incorrectes
   - Visualisation des patterns temporels
   - Évaluation de l'impact de l'historique des inondations

## Utilisation des scripts

### Enrichissement du dataset

```bash
python enrichir_historique.py
```

Ce script:
- Extrait les noms de villes depuis la colonne "timezone"
- Recherche ces villes dans la base EM-DAT
- Compte les occurrences d'inondations historiques
- Ajoute la colonne "historique_region" au dataset

### Comparaison des modèles

```bash
python comparaison_modeles.py
```

Ce script:
- Charge les modèles avec et sans historique d'inondation
- Les évalue sur le même jeu de test
- Compare les performances (accuracy, précision, rappel, F1, AUC)
- Analyse les cas où les prédictions diffèrent
- Génère des visualisations comparatives

### Analyse de l'importance des features

```bash
python analyse_importance_features.py
```

Ce script:
- Effectue une analyse par permutation pour mesurer l'importance des features
- Évalue la dégradation des performances lorsqu'une feature est permutée
- Génère des visualisations de l'importance des features
- Compare l'importance de "historique_region" aux autres features

## Résultats

L'ajout de l'historique des inondations par région améliore les performances du modèle:
- Meilleure détection des transitions d'inondation
- Amélioration de la précision et du rappel
- Réduction des faux négatifs (inondations non détectées)

L'analyse PCA et l'analyse par permutation montrent que "historique_region" est une feature importante pour la prédiction, se classant parmi les variables les plus influentes.

## Dépendances

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Notes

- Le modèle enrichi est sauvegardé sous `modele_inondation_enriched.keras`
- Les visualisations des analyses sont sauvegardées dans le répertoire courant
- Pour une analyse complète, exécutez le notebook `LSTM_final_enriched.ipynb`
