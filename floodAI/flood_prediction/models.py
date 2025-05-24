from django.db import models
from django.contrib.auth.models import User

# Modèle pour les alertes
class Alerte(models.Model):
    titre = models.CharField(max_length=100)
    message = models.TextField()
    date_creation = models.DateTimeField(auto_now_add=True)
    ville = models.CharField(max_length=100)
    niveau = models.IntegerField(default=1)  # 1: faible, 2: moyen, 3: élevé
    
    def __str__(self):
        return f"{self.titre} - {self.ville} (Niveau {self.niveau})"
    class Meta: 
        db_table = 'flood_prediction_alerte'
