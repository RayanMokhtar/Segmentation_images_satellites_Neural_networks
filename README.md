# Segmentation_images_satellites_Neural_networks
Segmentation d'images satellites & Prédiction temporelle sur 3 jours 🚀

Yo les devs ! Bienvenue sur le projet Segmentation_images_satellites_Neural_networks : on segmente des images SAR avec un CNN, on prédit leur évolution sur 3 jours avec un LSTM, et on balance le tout sur une carte interactive Django. Let’s go !

🎯 Objectifs clés

Segmentation CNN : extraire des labels (eau, végétation, bâti…) à partir d’images SAR.

Prédiction LSTM : modéliser la suite temporelle (t+1, t+2, t+3 jours).

Interface web Django : visualiser en temps réel les prédictions sur une carte interactive (Leaflet).

🛠 Stack & Pré-requis

Python 3.8+

TensorFlow ou PyTorch (ton choix)

Django 4.x

GDAL, rasterio, numpy, pandas

Leaflet.js + django-leaflet

PostgreSQL + PostGIS (optionnel pour data volumineuse)

🚀 Installation rapide

Clone ce repo :

git clone https://github.com/ton-orga/Segmentation_images_satellites_Neural_networks.git
cd Segmentation_images_satellites_Neural_networks

Virtualenv & dépendances :

python -m venv venv
source venv/bin/activate   # sous Linux/Mac
venv\\Scripts\\activate  # sous Windows
pip install -r requirements.txt

Variables d’environnement : copie .env.example ➡️ .env et ajuste :

DJANGO_SECRET_KEY=TaClefUltraSecrète
DATA_PATH=/chemin/vers/tes/images_SAR

🔧 Prétraitement des données

Chargement des images SAR (.tif)

Denoising / Speckle filtering (ex. filtre Lee)

Normalisation pixel-wise

Découpage en patches (e.g. 256×256)

Création de la séquence temporelle (t-2, t-1, t)

Script dédié : scripts/preprocess_data.py

🧠 Architecture des modèles

1) Segmentation CNN

Backbone : U-Net-like ou ResNet encoder

Output : masque binaire multi-classe

Loss : Dice + Cross-Entropy

2) Prédiction LSTM

Input : features extraites par le CNN à chaque t

Module : LSTM (hidden_size=512, 2 layers)

Output : features t+1, t+2, t+3

Décodage : upsampling → masque prédit pour chaque horizon

Entrainer les deux séparément ou en joint training selon tes préférences !

🎓 Entraînement & Évaluation

Lancer la segmentation :

python train_segmentation.py --epochs 50 --batch-size 16

Générer les embeddings temporels :

python extract_features.py --checkpoint best_cnn.pth

Lancer le LSTM :

python train_lstm.py --epochs 30 --batch-size 8

Metrics : IoU, Accuracy, RMSE temporel

Logs : TensorBoard (tensorboard --logdir logs/)

🌐 Carte interactive Django

Migration base :

python manage.py migrate

Importer prédictions (script CSV/GeoJSON) :

python scripts/load_predictions.py --file preds.geojson

Lancer le serveur :

python manage.py runserver

🔍 Ouvre http://localhost:8000 et check la map : zoom, couches, timeline 3 jours.