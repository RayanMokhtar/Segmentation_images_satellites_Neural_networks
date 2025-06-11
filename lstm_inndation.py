# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from collections import Counter
from keras.losses import CategoricalFocalLoss
from keras.layers import Conv1D

# ... (autres importations et code précédent)

# 3) Split temporel (60/20/20)
n = len(X_all)
idx1 = int(n * 0.6)
idx2 = int(n * 0.8)
X_train, y_train = X_all[:idx1], y_all[:idx1]
X_val, y_val = X_all[idx1:idx2], y_all[idx1:idx2]
X_test, y_test = X_all[idx2:], y_all[idx2:]

# Labels du jour précédent pour chaque split
y_prev_train = y_prev_all[:idx1]
y_prev_val = y_prev_all[idx1:idx2]
y_prev_test = y_prev_all[idx2:]

print('Train/Val/Test shapes :', X_train.shape, X_val.shape, X_test.shape)

# Utilisation de class_weight sur y_train pour gérer le déséquilibre
print("\nDistribution y_train avant entraînement:", Counter(y_train))
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), cw))
print('Class weights:', class_weight_dict)

# Aucun sur-échantillonnage : X_train_res = X_train, y_train_res = y_train
X_train_res, y_train_res = X_train, y_train

# 5) Normaliser sur X_train_smote seulement
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_res.reshape(-1, X_train_res.shape[-1])).reshape(X_train_res.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# ... (suite du code pour l'entraînement du modèle, en utilisant CategoricalFocalLoss comme fonction de perte)