import schedule
import time
import subprocess
import logging
import os
import sys

# Configuration du logging
logging.basicConfig(
    filename="cron_job.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def job(script_path):
    try:
        # Vérifie si le script existe
        if not os.path.exists(script_path):
            logging.error(f"Le script {script_path} n'existe pas.")
            return

        # Exécute le script Python donné
        subprocess.run([sys.executable, script_path], check=True)
        logging.info(f"Script {script_path} exécuté avec succès.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l'exécution du script {script_path}: {e}")
    except Exception as e:
        logging.error(f"Une erreur inattendue s'est produite: {e}")

def cleanup_logs():
    try:
        # Supprime les anciens logs si le fichier dépasse une taille limite (ex: 5 Mo)
        log_file = "cron_job.log"
        max_size = 5 * 1024 * 1024  # 5 Mo
        if os.path.exists(log_file) and os.path.getsize(log_file) > max_size:
            with open(log_file, "w") as f:
                f.truncate(0)
            logging.info("Fichier de log nettoyé.")
    except Exception as e:
        logging.error(f"Erreur lors du nettoyage des logs: {e}")

# Planifie l'exécution du script de prédiction des risques pour les régions abonnées
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predire_regions.py")
schedule.every().day.at("02:45").do(job, script_path=script_path)
logging.info(f"Script de prédiction des régions planifié pour 2h du matin: {script_path}")

# Planifie le nettoyage des logs chaque jour à 23h
schedule.every().day.at("23:00").do(cleanup_logs)

# Boucle infinie pour vérifier et exécuter les tâches planifiées
while True:
    schedule.run_pending()
    time.sleep(1)