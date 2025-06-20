from django.db import models
from django.contrib.auth.models import User
import uuid
from datetime import timedelta
from django.utils import timezone

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

# Modèle pour les abonnements aux villes
class AbonnementVille(models.Model):
    user = models.ForeignKey(User, related_name='abonnements', on_delete=models.CASCADE)
    ville = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    date_abonnement = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['user', 'ville']
        verbose_name = "Abonnement à une ville"
        verbose_name_plural = "Abonnements aux villes"

    def __str__(self):
        return f"{self.user.username} - {self.ville}"

# Modèle pour la vérification d'email
class EmailVerification(models.Model):
    email = models.EmailField(unique=True)
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    username = models.CharField(max_length=150)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    password = models.CharField(max_length=128)
    adresse = models.CharField(max_length=255, blank=True, null=True)
    ville = models.CharField(max_length=100, blank=True, null=True)
    code_postal = models.CharField(max_length=10, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Vérification pour {self.email}"
    
    @property
    def is_expired(self):
        expiration_time = timedelta(hours=24)
        return (timezone.now() - self.created_at) > expiration_time

# Modèle pour les profils utilisateurs avec adresse
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    adresse = models.CharField(max_length=255, blank=True, null=True)
    ville = models.CharField(max_length=100, blank=True, null=True)
    code_postal = models.CharField(max_length=10, blank=True, null=True)
    
    def __str__(self):
        return f"Profil de {self.user.username}"

# Modèle pour la réinitialisation de mot de passe
class PasswordResetToken(models.Model):
    email = models.EmailField()
    token = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    used = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Réinitialisation pour {self.email}"
    
    @property
    def is_expired(self):
        expiration_time = timedelta(hours=24)
        return (timezone.now() - self.created_at) > expiration_time

# Modèle pour l'historique des prédictions
class HistoriquePrediction(models.Model):
    region = models.CharField(max_length=100, verbose_name="Nom de la région")
    latitude = models.FloatField()
    longitude = models.FloatField()
    date_prediction = models.DateField(verbose_name="Date de la prédiction")
    date_execution = models.DateTimeField(auto_now_add=True, verbose_name="Date d'exécution")
    probabilite = models.FloatField(verbose_name="Probabilité d'inondation (%)")
    niveau_risque = models.CharField(max_length=20, verbose_name="Niveau de risque")
    inondation_prevue = models.BooleanField(default=False, verbose_name="Inondation prévue")
    modele_utilise = models.CharField(max_length=50, default="CNN-LSTM", verbose_name="Modèle utilisé")
    
    class Meta:
        verbose_name = "Historique de prédiction"
        verbose_name_plural = "Historique des prédictions"
        ordering = ['-date_execution', 'region']
        # Une prédiction unique par région et date de prédiction
        unique_together = ['region', 'date_prediction']
        
    def __str__(self):
        return f"{self.region} - {self.date_prediction} - {self.niveau_risque} ({self.probabilite:.2f}%)"
