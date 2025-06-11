#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour enrichir le fichier dataset_prepared.csv avec des données historiques d'inondations
issues du fichier EM-DAT.

Ce script:
1. Extrait le nom de la ville à partir de la colonne "timezone" (deuxième partie après "/")
2. Recherche dans le jeu de données EM-DAT les emplacements correspondants à l'aide d'expressions régulières
3. Compte les occurrences où "Disaster Type" est "Flood" pour chaque emplacement
4. Ajoute une nouvelle colonne "historique_region" au fichier dataset_prepared.csv avec le nombre 
   d'inondations historiques pour chaque région
"""

import pandas as pd
import re
import os

def main():
    # Chemins des fichiers
    emdat_file = r"d:\dataset\SEN12FLOOD (1)\public_emdat_custom_request_2025-05-22_1a78f1da-122a-41fb-9eac-038db183ca0a(1).csv"
    dataset_file = r"d:\dataset\SEN12FLOOD (1)\dataset_prepared.csv"
    output_file = r"d:\dataset\SEN12FLOOD (1)\dataset_enriched.csv"
    
    print("Chargement des fichiers...")
    
    # Chargement du fichier EM-DAT (avec séparateur point-virgule)
    emdat_df = pd.read_csv(emdat_file, sep=';', encoding='utf-8')
    
    # Chargement du fichier dataset_prepared
    dataset_df = pd.read_csv(dataset_file, encoding='utf-8')
    
    print(f"Fichier EM-DAT chargé avec {len(emdat_df)} lignes")
    print(f"Fichier dataset_prepared chargé avec {len(dataset_df)} lignes")
    
    # Filtrer le jeu de données EM-DAT pour ne garder que les inondations
    floods_df = emdat_df[emdat_df['Disaster Type'].str.contains('Flood', na=False)]
    print(f"Nombre d'événements d'inondation dans EM-DAT: {len(floods_df)}")
    
    # Extraire les noms de villes à partir de la colonne timezone
    dataset_df['city'] = dataset_df['timezone'].apply(lambda x: x.split('/')[1] if pd.notna(x) and '/' in x else None)
    
    # Créer un dictionnaire pour stocker le nombre d'inondations par ville
    flood_counts = {}
    
    print("Comptage des inondations historiques par région...")
    
    # Pour chaque ville unique dans le dataset
    unique_cities = dataset_df['city'].dropna().unique()
    
    for city in unique_cities:
        # Créer un modèle regex pour rechercher la ville dans les emplacements
        # Utilisation de \b pour les limites de mots et (?i) pour ignorer la casse
        city_pattern = r'(?i)\b' + re.escape(city) + r'\b'
        
        # Compter les occurrences d'inondations où l'emplacement correspond au modèle
        count = 0
        
        # Vérifier dans la colonne Location
        if 'Location' in floods_df.columns:
            location_matches = floods_df['Location'].str.contains(city_pattern, regex=True, na=False)
            count += location_matches.sum()
        
        # Vérifier aussi dans la colonne Country au cas où
        country_matches = floods_df['Country'].str.contains(city_pattern, regex=True, na=False)
        count += country_matches.sum()
        
        flood_counts[city] = count
        
        # Afficher des informations de progression
        if count > 0:
            print(f"Ville: {city}, Nombre d'inondations historiques: {count}")
    
    # Ajouter la colonne historique_region au dataset
    dataset_df['historique_region'] = dataset_df['city'].map(flood_counts).fillna(0).astype(int)
    
    # Sauvegarder le dataset enrichi
    dataset_df.to_csv(output_file, index=False)
    print(f"Dataset enrichi sauvegardé sous {output_file}")
    
    # Afficher quelques statistiques
    total_with_history = len(dataset_df[dataset_df['historique_region'] > 0])
    print(f"Nombre de lignes avec données historiques d'inondation: {total_with_history}")
    print(f"Nombre maximum d'inondations historiques pour une région: {dataset_df['historique_region'].max()}")

if __name__ == "__main__":
    main()