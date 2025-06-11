import pandas as pd
import numpy as np
import logging
import sys
from datetime import timedelta
from tqdm import tqdm
import requests
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enrichissement_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# --- Fonctions pour l'interpolation des données météorologiques ---
def interpolate_weather_data(merged_df, weather_cols):
    """
    Interpole les données météorologiques entre les dates connues.
    
    Arguments:
        merged_df: DataFrame fusionné avec toutes les dates dans la plage
        weather_cols: Liste des colonnes météorologiques à interpoler
    """
    # Pour chaque colonne météorologique, effectuer une interpolation linéaire
    for col in weather_cols:
        if col in merged_df.columns:
            try:
                merged_df[col] = merged_df[col].interpolate(method='linear')
            except Exception as e:
                logger.warning(f"Impossible d'interpoler la colonne {col}: {e}")
    
    return merged_df

def fill_static_columns(merged_df, static_cols):
    """
    Remplit les colonnes statiques en utilisant des méthodes de propagation avant/arrière.
    Évite l'utilisation directe de fillna(method='ffill/bfill') qui est déprécié.
    
    Arguments:
        merged_df: DataFrame avec les données fusionnées
        static_cols: Liste des colonnes statiques à remplir
    """
    for col in static_cols:
        if col in merged_df.columns:
            # Utilisation de bfill/ffill comme méthode distincte plutôt que comme paramètre de fillna
            merged_df[col] = merged_df[col].ffill().bfill()
    
    return merged_df

def set_source_for_interpolated_values(df):
    """
    Marque les valeurs interpolées avec 'interpolation' comme source.
    
    Arguments:
        df: DataFrame contenant les données
    """
    if 'source' in df.columns:
        # Marquer comme 'interpolation' les lignes avec données météo mais sans source
        mask = df['source'].isna() & df['tempmax'].notna()
        df.loc[mask, 'source'] = 'interpolation'
    else:
        # Créer la colonne source si elle n'existe pas
        temp_mask = df['tempmax'].notna()  # Vérifier si des données météo existent
        df['source'] = np.where(temp_mask, 'interpolation', np.nan)
    
    return df

# --- Configuration ---
INPUT_CSV = './dataset_final - Copie.csv'
OUTPUT_CSV = './dataset_complet.csv'
DELIMITER = ';'

def enrich_timeseries_data():
    """Ré-écriture complète de la fonction pour regrouper par chemin_directory et combler les dates manquantes."""
    # Lecture du CSV
    try:
        df = pd.read_csv(INPUT_CSV, delimiter=DELIMITER)
    except FileNotFoundError:
        logger.error(f"ERREUR : Le fichier {INPUT_CSV} n'a pas été trouvé.")
        return
    except Exception as e:
        logger.error(f"ERREUR lors de la lecture du CSV : {e}")
        return

    # Conversion de la colonne date en datetime (format jour/mois/année)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    # Séparation des colonnes numériques et non numériques (chaînes de caractères)
    # Exclure explicitement 'label' afin de ne pas l’interpoler mais la propager
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'label']
    string_cols = [col for col in df.columns if col not in numeric_cols + ['date']]

    enriched_groups = []
    # Regroupement par time series via chemin_directory
    for chemin, group in df.groupby('chemin_directory'):
        group = group.sort_values('date').reset_index(drop=True)
        rows = []
        # Itération sur chaque paire consécutive
        for i in range(len(group) - 1):
            prev = group.iloc[i]
            nxt = group.iloc[i + 1]
            rows.append(prev)
            diff_days = (nxt['date'] - prev['date']).days
            # Création des lignes manquantes si écart > 1 jour
            if diff_days > 1:
                for delta in range(1, diff_days):
                    d = prev['date'] + timedelta(days=delta)
                    new_row = prev.copy()
                    new_row['date'] = d
                    # Moyenne pour les colonnes numériques
                    for col in numeric_cols:
                        new_row[col] = (prev[col] + nxt[col]) / 2
                    # Propagation de la valeur précédente pour les chaînes et labels
                    for col in string_cols:
                        new_row[col] = prev[col]
                    rows.append(new_row)
        # Ajout de la dernière observation
        if not group.empty:
            rows.append(group.iloc[-1])
        enriched_groups.append(pd.DataFrame(rows))

    # Fusion de tous les groupes enrichis
    df_enriched = pd.concat(enriched_groups, ignore_index=True)
    # Reformatage de la date au format original
    df_enriched['date'] = df_enriched['date'].dt.strftime('%d/%m/%Y')

    # --- Enrichissement des altitudes via l'API OpenTopoData ---
    # Aucun key nécessaire pour l'API publique
    coords = df_enriched[['latitude_centroid','longitude_centroid']].drop_duplicates()
    elev_map = {}
    # Fonction pour chunker la liste
    def chunks(lst, n=100):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    coord_list = [(row['latitude_centroid'], row['longitude_centroid']) for _, row in coords.iterrows()]
    for chunk in chunks(coord_list, 100):
        # construire le paramètre locations
        locs = '|'.join(f"{lat},{lon}" for lat,lon in chunk)
        url = f"https://api.opentopodata.org/v1/srtm90m?locations={locs}"
        resp = requests.get(url)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            for (lat,lon), res in zip(chunk, results):
                elev_map[(lat,lon)] = res.get('elevation')
        elif resp.status_code == 429:
            # trop de requêtes, attendre 1 seconde puis retry
            time.sleep(1)
            resp = requests.get(url)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                for (lat,lon), res in zip(chunk, results):
                    elev_map[(lat,lon)] = res.get('elevation')
        else:
            resp.raise_for_status()
        # respecter max 1 call/sec
        time.sleep(1)
    # appliquer les altitudes
    df_enriched['elevation'] = df_enriched.apply(
        lambda r: elev_map.get((r['latitude_centroid'], r['longitude_centroid'])), axis=1)

    # Sauvegarde du fichier enrichi
    try:
        df_enriched.to_csv(OUTPUT_CSV, sep=DELIMITER, index=False)
        logger.info(f"Fichier enrichi sauvegardé : {OUTPUT_CSV}")
    except Exception as e:
        logger.error(f"ERREUR lors de la sauvegarde du CSV : {e}")
        return

if __name__ == "__main__":
    enrich_timeseries_data()
