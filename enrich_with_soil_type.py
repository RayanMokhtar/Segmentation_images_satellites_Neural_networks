import pandas as pd
import httpx
import time
import logging
import sys

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enrich_soil_type_log.txt", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_CSV = 'dataset_complet.csv'
OUTPUT_CSV = 'dataset_with_soil_type.csv'
DELIMITER = ';'
# Endpoint soil type
BASE_URL = 'https://api.openepi.io/soil/type'
TOP_K = 3  # récupère les top K probabilités

RATE_LIMIT_SECONDS = 1

def fetch_most_probable_soil(lat, lon, retries=3, backoff=1):
    """
    Récupère le type de sol le plus probable pour une coordonnée.
    """
    for i in range(retries):
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(BASE_URL, params={'lat': lat, 'lon': lon})
            if resp.status_code == 200:
                props = resp.json().get('properties', {})
                return props.get('most_probable_soil_type')
            elif resp.status_code == 429:
                time.sleep(backoff)
                continue
            else:
                logger.error(f"HTTP {resp.status_code} for {lat},{lon}")
                return None
        except Exception as e:
            logger.error(f"Error fetching soil for {lat},{lon}: {e}")
            time.sleep(backoff)
    logger.error(f"Failed after {retries} retries for {lat},{lon}")
    return None

def main():
    df = pd.read_csv(INPUT_CSV, sep=DELIMITER, encoding='utf-8')
    # on récupère un point représentatif par région (chemin_directory)
    regions = df.groupby('chemin_directory')[['latitude_centroid','longitude_centroid']].first()
    logger.info(f"Nombre de régions à traiter : {len(regions)}")

    # 2) Récupération du soil_type le plus probable par région
    region_soil = {}
    for region, row in regions.iterrows():
        lat, lon = row['latitude_centroid'], row['longitude_centroid']
        soil = fetch_most_probable_soil(lat, lon)
        region_soil[region] = soil
        logger.info(f"Région {region} -> soil_type: {soil}")
        time.sleep(RATE_LIMIT_SECONDS)

    # appliquer le même soil_type à chaque ligne selon sa région
    df['soil_type'] = df['chemin_directory'].map(region_soil)

    df.to_csv(OUTPUT_CSV, sep=DELIMITER, index=False, encoding='utf-8')
    logger.info(f"Dataset enrichi sauvegardé : {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
