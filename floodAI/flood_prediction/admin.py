from django.contrib import admin
from .models import Alerte

# Configuration de l'interface d'administration pour Alerte
class AlerteAdmin(admin.ModelAdmin):
    list_display = ('titre', 'ville', 'niveau', 'date_creation')
    list_filter = ('niveau', 'ville')
    search_fields = ('titre', 'message', 'ville')

# Enregistrement des modèles dans l'administration
admin.site.register(Alerte, AlerteAdmin)
