# README: Démonstration de prédiction d'inondations

Ce document explique comment utiliser le script `demonstration.py` pour prédire le risque d'inondation à partir de coordonnées géographiques et d'une date.

## Prérequis

1. Python 3.8+ installé
2. Bibliothèques requises :
   - numpy
   - pandas
   - matplotlib
   - tensorflow
   - sklearn
   - requests
   - httpx

Vous pouvez installer les dépendances avec la commande :
```
pip install numpy pandas matplotlib tensorflow scikit-learn requests httpx
```

3. Une clé API pour Visual Crossing Weather (gratuite avec limitations) :
   - Inscrivez-vous sur [Visual Crossing](https://www.visualcrossing.com/)
   - Obtenez une clé API gratuite

## Utilisation

### Syntaxe de base

```
python demonstration.py --lat [LATITUDE] --lon [LONGITUDE] --date [DATE] --api-key [CLE_API_VISUAL_CROSSING]
```

### Paramètres obligatoires

- `--lat` : Latitude de la région d'intérêt (format décimal, ex: -19.138148)
- `--lon` : Longitude de la région d'intérêt (format décimal, ex: 146.851468)
- `--date` : Date à analyser au format YYYY-MM-DD (ex: 2019-03-15)
- `--api-key` : Votre clé API Visual Crossing Weather

### Paramètres optionnels

- `--model` : Chemin vers le modèle LSTM entraîné (.keras ou .h5, par défaut: best_model.keras)
- `--output` : Chemin pour sauvegarder les résultats (par défaut: prediction_results.csv)
- `--plot` : Générer un graphique des prédictions (ajoutez ce flag sans valeur)

### Exemple

```
python demonstration.py --lat -19.138148 --lon 146.851468 --date 2019-03-15 --api-key PUTUNBTHW5R3Q9K2WUW6MPSD6 --plot
```

## Fonctionnement

Le script effectue les étapes suivantes :

1. Récupération des données météorologiques pour une fenêtre de 7 jours jusqu'à la date spécifiée (Visual Crossing Weather API)
2. Récupération de l'élévation pour les coordonnées spécifiées (Open Topo Data API)
3. Récupération du type de sol pour les coordonnées spécifiées (OpenEPI Soil Type API)
4. Préparation des données au format attendu par le modèle LSTM
5. Standardisation des caractéristiques numériques
6. Chargement du modèle LSTM préalablement entraîné
7. Prédiction du risque d'inondation pour 3 jours après la fenêtre d'observation
8. Affichage et sauvegarde des résultats

## Sorties

- Console : Affichage détaillé des étapes et du résultat de prédiction
- Fichier CSV : Sauvegarde du résultat de prédiction (date, probabilité, classe prédite)
- Graphique : Visualisation des données d'entrée et de la prédiction (si `--plot` est spécifié)

## Notes importantes

- Le script utilise une fenêtre historique de 7 jours (L=7) pour prédire le risque d'inondation 3 jours (H=3) après la fin de cette fenêtre
- Les API ont des limites d'utilisation quotidiennes pour les comptes gratuits
- Le modèle attend un certain format de données. En cas d'incompatibilité, le script tentera d'adapter les dimensions
- Pour de meilleurs résultats, assurez-vous que le modèle a été entraîné sur des données similaires à celles de la région que vous analysez
