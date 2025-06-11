#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demonstration_fixed.py

Version modifiée du script de démonstration qui préserve la valeur d'élévation originale
lors de la standardisation des caractéristiques.

Cette version utilise standardize_features_fixed au lieu de standardize_features
pour éviter que l'élévation ne soit normalisée à 0.

Utilisation:
    python demonstration_fixed.py --lat [LATITUDE] --lon [LONGITUDE] --date [DATE]

Exemple:
    python demonstration_fixed.py --lat 41.383 --lon 2.183 --date 2023-06-10
"""

# Importer les modules du script original
from demonstration import *

# Importer la fonction modifiée de standardisation
from standardize_fixed import standardize_features_fixed

# Remplacer la fonction standardize_features par notre version modifiée
standardize_features = standardize_features_fixed

# Si ce script est exécuté directement
if __name__ == "__main__":
    main()
