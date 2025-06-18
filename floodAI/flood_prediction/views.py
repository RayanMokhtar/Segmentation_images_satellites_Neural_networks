from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Alerte, AbonnementVille
from django.db.models import Q
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django import forms
from django.http import JsonResponse
from django.core.mail import send_mail
from django.conf import settings
import json
import requests
import random
from datetime import datetime, date

# Formulaire d'inscription personnalisé
class InscriptionForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, label='Prénom')
    last_name = forms.CharField(max_length=30, required=True, label='Nom')
    email = forms.EmailField(max_length=254, required=True, label='Email')

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
        
        return user

# Vue principale de la page d'accueil
def home(request):
    context = {
        'page_title': 'Prévisions des inondations',
        'default_location': {
            'lat': 46.603354,
            'lng': 1.888334
        }    }
    return render(request, 'home.html', context)
    
    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'})

def register(request):
    if request.method == 'POST':
        form = InscriptionForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            login(request, user)
            messages.success(request, f'Compte créé avec succès! Bienvenue {user.first_name}!')
            return redirect('home')
    else:
        form = InscriptionForm()
    
    return render(request, 'register.html', {'form': form})

# Vue de connexion
def login_page(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Bienvenue {user.first_name}!')
                next_url = request.GET.get('next', 'home')
                return redirect(next_url)
        else:
            messages.error(request, 'Identifiants incorrects. Veuillez réessayer.')
    else:
        form = AuthenticationForm()
    
    return render(request, 'login.html', {'form': form})

# Déconnexion
def logout_view(request):
    logout(request)
    messages.success(request, 'Vous avez été déconnecté avec succès.')
    return redirect('home')

# Vue pour récupérer les données météo
def get_weather_data(request):
    lat = request.GET.get('lat')
    lng = request.GET.get('lng')
    if not lat or not lng:
        return JsonResponse({'error': 'Latitude et longitude sont requis'}, status=400)
    
    try:
        weather_api_url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m&windspeed_unit=kmh&timezone=auto'
        response = requests.get(weather_api_url)
        response.raise_for_status()
        return JsonResponse(response.json())
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

# Vue pour récupérer les informations de localisation
def get_location_info(request):
    lat = request.GET.get('lat')
    lng = request.GET.get('lng')
    if not lat or not lng:
        return JsonResponse({'error': 'Latitude et longitude sont requis'}, status=400)
    
    try:
        nominatim_api_url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=10&addressdetails=1'
        headers = {
            'Accept-Language': 'fr',
            'User-Agent': 'FloodAI/1.0 (contact@example.com)'
        }
        response = requests.get(nominatim_api_url, headers=headers)
        response.raise_for_status()
        return JsonResponse(response.json())
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

# Vue pour vérifier si l'utilisateur est abonné à une ville
@login_required
def check_subscription_status(request):
    city_name = request.GET.get('city_name')
    if not city_name:
        return JsonResponse({'success': False, 'error': 'Nom de ville requis'}, status=400)
    
    is_subscribed = AbonnementVille.objects.filter(user=request.user, ville=city_name).exists()
    
    return JsonResponse({
        'success': True,
        'is_subscribed': is_subscribed
    })

# Vue pour s'abonner/désabonner à une ville
@login_required
def subscribe_city(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            city_name = data.get('city_name')
            lat = data.get('lat')
            lng = data.get('lng')
            action = data.get('action', 'subscribe')  # 'subscribe' ou 'unsubscribe'
            
            if not all([city_name, lat, lng]):
                return JsonResponse({'success': False, 'error': 'Données incomplètes'}, status=400)
            
            # Désabonnement
            if action == 'unsubscribe':
                abonnement = AbonnementVille.objects.filter(user=request.user, ville=city_name)
                if abonnement.exists():
                    abonnement.delete()
                    return JsonResponse({
                        'success': True, 
                        'message': f'Désabonnement de {city_name} effectué avec succès',
                        'is_subscribed': False
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': f'Vous n\'êtes pas abonné à {city_name}'
                    })
            
            # Abonnement (par défaut)
            else:
                # Vérifier si l'abonnement existe déjà
                existing_subscription = AbonnementVille.objects.filter(user=request.user, ville=city_name).exists()
                if existing_subscription:
                    return JsonResponse({
                        'success': False,
                        'error': f'Vous êtes déjà abonné à {city_name}',
                        'is_subscribed': True
                    })
                
                # Créer un nouvel abonnement
                abonnement = AbonnementVille.objects.create(
                    user=request.user,
                    ville=city_name,
                    latitude=lat,
                    longitude=lng
                )
                
                return JsonResponse({
                    'success': True, 
                    'message': f'Abonnement à {city_name} créé avec succès',
                    'is_subscribed': True
                })
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'}, status=405)

# Vue pour supprimer un abonnement à une ville
@login_required
def unsubscribe_city(request, abonnement_id):
    try:
        abonnement = get_object_or_404(AbonnementVille, id=abonnement_id, user=request.user)
        ville_name = abonnement.ville
        abonnement.delete()
        messages.success(request, f'Abonnement à {ville_name} supprimé avec succès.')
    except Exception as e:
        messages.error(request, f'Erreur lors de la suppression : {str(e)}')
    
    return redirect('profil')

# Vue profil (protégée par login_required)
@login_required
def profil(request):
    user = request.user
    
    # Récupérer les abonnements de l'utilisateur
    abonnements = AbonnementVille.objects.filter(user=user).order_by('ville')
    
    # Récupérer quelques alertes génériques pour la démo
    alertes_recentes = Alerte.objects.all().order_by('-date_creation')[:3]
    alertes = [f"{alerte.titre} : {alerte.message}" for alerte in alertes_recentes]
    
    # Si pas d'alertes, ajouter quelques exemples
    if not alertes:
        alertes = [
            "Aucune alerte en cours.",
            "Vous recevrez des notifications en cas de risque d'inondation."
        ]
    
    infos = {
        'nom': user.last_name or '',
        'prenom': user.first_name or '',
        'mail': user.email or '',
        'adresse': '',  # Plus utilisé avec le modèle User standard
        'ville': ''     # Plus utilisé avec le modèle User standard
    }
    
    return render(request, 'profil.html', {
        'infos': infos, 
        'abonnements': abonnements,
        'alertes': alertes
    })

# Vue pour la page de segmentation d'images
def segmentation(request):
    context = {
        'model_type': request.POST.get('model_type', 'CNN')
    }
    
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        
        # Sauvegarde temporaire de l'image
        import os
        from django.conf import settings
        import tempfile
        
        # Créer un dossier pour les images temporaires s'il n'existe pas
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_images')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Sauvegarder l'image temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=temp_dir) as temp_file:
            for chunk in uploaded_image.chunks():
                temp_file.write(chunk)
            temp_image_path = temp_file.name
        
        # Chemins relatifs pour les templates
        relative_path = os.path.relpath(temp_image_path, settings.MEDIA_ROOT)
        original_image_url = os.path.join(settings.MEDIA_URL, relative_path)
        
        # Ici, vous pourriez appeler votre modèle de segmentation (CNN ou U-Net)
        # Pour l'instant, nous allons simplement simuler un résultat
        # Dans une implémentation réelle, vous appelleriez votre modèle ici
        
        # Simulation d'une image résultante (même image pour le moment)
        segmented_image_url = original_image_url
        
        context.update({
            'original_image': original_image_url,
            'segmented_image': segmented_image_url,
            'prediction_done': True
        })
    
    return render(request, 'segmentation.html', context)

# Vue pour la page de prédiction LSTM
def lstm(request):
    import random
    from datetime import datetime
    
    context = {
        'model_type': request.POST.get('model_type', 'LSTM standard')
    }
    
    if request.method == 'POST':
        # Récupération des données du formulaire
        date_prediction = request.POST.get('date_prediction', '')
        latitude = request.POST.get('latitude', '')
        longitude = request.POST.get('longitude', '')
        model_type = request.POST.get('model_type', 'LSTM standard')
        
        # Validation basique
        if date_prediction and latitude and longitude:
            # Génération d'un résultat de prédiction aléatoire (0-100%)
            prediction_value = random.randint(0, 100)
            
            # Déterminer la couleur du résultat et le message
            if prediction_value > 50:
                risk_level = "élevé"
                color = "red"
            elif prediction_value == 50:
                risk_level = "modéré"
                color = "yellow"
            else:
                risk_level = "faible"
                color = "green"
                
            # Formatage de la date pour l'affichage
            try:
                # Essayer de parser la date au format YYYY-MM-DD
                date_obj = datetime.strptime(date_prediction, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d/%m/%Y')
            except ValueError:
                # Si erreur, utiliser la valeur telle quelle
                formatted_date = date_prediction
                
            # Construction du contexte pour le template
            context.update({
                'prediction_done': True,
                'prediction_value': prediction_value,
                'risk_level': risk_level,
                'color': color,
                'date_prediction': formatted_date,
                'latitude': latitude,
                'longitude': longitude,
                'model_type': model_type
            })
    
    return render(request, 'lstm.html', context)

# Fonction pour envoyer un email de prédiction
@login_required
def send_prediction_email(request):
    user = request.user
    email = user.email
    
    # Débogage pour voir les paramètres d'email
    from django.conf import settings
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Tentative d'envoi d'email à {email}")
    logger.error(f"Configuration: {settings.EMAIL_HOST}, {settings.EMAIL_PORT}, {settings.EMAIL_HOST_USER}")
    
    if not email:
        messages.error(request, "Vous n'avez pas d'adresse email configurée dans votre profil.")
        return redirect('profil')
    
    # Générer une prédiction aléatoire pour démonstration
    prediction_value = random.randint(0, 100)
    
    # Déterminer le niveau de risque basé sur la valeur de prédiction
    if prediction_value > 70:
        risk_level = "élevé"
    elif prediction_value > 30:
        risk_level = "modéré"
    else:
        risk_level = "faible"
    
    # Date du jour pour la prédiction
    today = date.today().strftime('%d/%m/%Y')
      # Contenu de l'email
    subject = f"Prédiction de risque d'inondation - {today}"
    message = f"""
Bonjour {user.first_name} {user.last_name},

Voici votre prédiction de risque d'inondation du {today}.

Risque d'inondation : {prediction_value}% - Niveau {risk_level}

Cette prédiction est basée sur les données météorologiques actuelles et l'analyse de notre modèle IA.

Nous vous recommandons de rester vigilant et de suivre les consignes de sécurité en cas d'alerte météorologique.

Cordialement,
L'équipe FireFloodAI
    """
    
    try:
        # Utiliser le DEFAULT_FROM_EMAIL des settings pour l'expéditeur
        from django.conf import settings
        sender_email = settings.DEFAULT_FROM_EMAIL
        
        # Envoyer l'email
        send_mail(
            subject,
            message,
            sender_email,  # Expéditeur configuré dans settings.py
            [email],  # Destinataire
            fail_silently=False,
        )
        messages.success(request, f"Un email de prédiction a été envoyé à votre adresse {email}.")
        
        # Log de succès
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Email envoyé avec succès à {email}")
        
    except Exception as e:
        # Enregistrer l'erreur détaillée
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Erreur lors de l'envoi de l'email à {email}: {str(e)}")
        logger.error(traceback.format_exc())
        
        messages.error(request, f"Erreur lors de l'envoi de l'email: {str(e)}")
    
    return redirect('profil')
