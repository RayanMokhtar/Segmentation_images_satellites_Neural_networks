import redis
import json
import os
import hashlib
from datetime import datetime, timedelta

class FloodPredictionCache:
    """
    Classe gérant le cache Redis pour stocker les prédictions d'inondation
    """
    
    def __init__(self, host='localhost', port=6379, db=0, ttl_days=30):
        """
        Initialise la connexion Redis
            host: Hôte Redis (par défaut: localhost)
            port: Port Redis (par défaut: 6379)
            db: Base de données Redis (par défaut: 0)
            ttl_days: Durée de vie des entrées en cache en jours (par défaut: 30)
        """
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl_seconds = ttl_days * 24 * 60 * 60  # Conversion en secondes 30 jours le ttl 
        
        # Essayer de se connecter pour vérifier que Redis est disponible
        try:
            self.redis.ping()
            print("✅ Connexion au cache Redis établie avec succès")
        except redis.exceptions.ConnectionError:
            print("⚠️ Impossible de se connecter à Redis. Le cache ne sera pas utilisé.")
            self.redis = None
    
    def generate_cache_key(self, longitude, latitude, size_km, start_date, end_date):
        """
        Génère une clé de cache unique basée sur les paramètres de requête
        
        Returns:
            str: Clé de cache unique
        """
        # Créer une chaîne représentant les paramètres
        params_str = f"{longitude}_{latitude}_{size_km}_{start_date}_{end_date}"
        
        #hacher => binaire => hexadécimal Paramètres: "68.8574_27.7052_20_2022-09-22_2022-09-30"
        key = f"flood:prediction:{hashlib.md5(params_str.encode()).hexdigest()}"
        return key
    
    def get_prediction(self, longitude, latitude, size_km, start_date, end_date):
        """
        Récupère une prédiction depuis le cache
        
        Returns:
            dict: Résultat de la prédiction ou None si non trouvé
        """
        if not self.redis:
            return None
            
        #générer hachage de la clé 
        key = self.generate_cache_key(longitude , latitude , size_km , start_date , end_date)
        # Récupérer les données du cache
        cached_data = self.redis.get(key)     
        if cached_data:
            # Décompter le nombre d'accès au cache
            self.redis.hincrby("flood:stats", "cache_hits", 1)
            
            # Charger la prédiction depuis la chaîne JSON
            prediction = json.loads(cached_data)

            return prediction #directement on charge la prédiction et on la retourne
        else:
            # Décompter les échecs de cache => métriques à analyser 
            self.redis.hincrby("flood:stats", "cache_misses", 1)
            return None
    
    def save_prediction(self, longitude, latitude, size_km, start_date, end_date, prediction ):
        """
        Enregistre une prédiction dans le cache prediction 
        """
        if not self.redis:
            return
            
        key = self.generate_cache_key(longitude, latitude, size_km, start_date, end_date)
        
        # Enregistrer l'heure actuelle dans la prédiction
        prediction['cache_metadata'] = {
            'cached_at': datetime.now().isoformat(),
            'longitude': longitude,
            'latitude': latitude,
            'size_km': size_km,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Stocker la prédiction sous forme de chaîne JSON
        self.redis.set(key, json.dumps(prediction))
        
        # Définir l'expiration
        self.redis.expire(key, self.ttl_seconds)
        
        # Mettre à jour les statistiques
        self.redis.hincrby("flood:stats", "total_predictions", 1)
        self.redis.hincrby("flood:stats", "total_cached", 1)
        
        print(f"✅ Prédiction mise en cache pour {longitude}, {latitude} (expire dans {self.ttl_seconds//86400} jours)")
    
    def get_stats(self):
        """
        Récupère les statistiques d'utilisation du cache
        
        Returns:
            dict: Statistiques du cache
        """
        if not self.redis:
            return {"status": "non connecté"}
            
        stats = self.redis.hgetall("flood:stats")
        
        # Convertir les valeurs en entiers
        for key in stats:
            stats[key] = int(stats[key])
        
        # Calculer le taux de succès du cache
        hits = stats.get("cache_hits", 0)
        misses = stats.get("cache_misses", 0)
        total = hits + misses
        
        if total > 0:
            stats["cache_hit_rate"] = round((hits / total) * 100, 2)
        else:
            stats["cache_hit_rate"] = 0
            
        # Ajouter le nombre d'entrées actuelles dans le cache
        stats["current_cache_entries"] = self.redis.keys("flood:prediction:*")
        
        return stats
    
    def clear_cache(self):
        """
        Vide complètement le cache des prédictions
        
        Returns:
            int: Nombre d'entrées supprimées
        """
        if not self.redis:
            return 0
            
        # Récupérer toutes les clés de prédiction
        keys = self.redis.keys("flood:prediction:*")
        
        # Supprimer toutes les clés
        if keys:
            return self.redis.delete(*keys)
        return 0
    

