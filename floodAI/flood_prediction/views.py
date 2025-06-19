from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Alerte, AbonnementVille, PasswordResetToken
from django.db.models import Q
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm, SetPasswordForm
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
    adresse = forms.CharField(max_length=255, required=True, label='Adresse')
    ville = forms.CharField(max_length=100, required=True, label='Ville')
    code_postal = forms.CharField(max_length=10, required=True, label='Code postal')

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'adresse', 'ville', 'code_postal', 'password1', 'password2')
    
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
        if form.is_valid():            # Ne pas créer l'utilisateur tout de suite
            user_data = {
                'username': form.cleaned_data['username'],
                'email': form.cleaned_data['email'],
                'first_name': form.cleaned_data['first_name'],
                'last_name': form.cleaned_data['last_name'],
                'password': form.cleaned_data['password1'],
                'adresse': form.cleaned_data['adresse'],
                'ville': form.cleaned_data['ville'],
                'code_postal': form.cleaned_data['code_postal'],
            }
            
            # Vérifier si l'email existe déjà
            if User.objects.filter(email=user_data['email']).exists():
                messages.error(request, 'Un compte avec cet email existe déjà.')
                return render(request, 'register.html', {'form': form})
            
            # Vérifier si une vérification est déjà en cours pour cet email
            from .models import EmailVerification
            existing_verification = EmailVerification.objects.filter(email=user_data['email'])
            if existing_verification.exists():
                # Supprimer les anciennes vérifications pour cet email
                existing_verification.delete()
              # Créer une entrée dans la table de vérification
            verification = EmailVerification.objects.create(
                email=user_data['email'],
                username=user_data['username'],
                first_name=user_data['first_name'],
                last_name=user_data['last_name'],
                password=user_data['password'],
                adresse=user_data['adresse'],
                ville=user_data['ville'],
                code_postal=user_data['code_postal'],
            )
              # Envoyer un email de vérification
            verification_url = request.build_absolute_uri(f'/verify-email/{verification.token}/')
            subject = 'Vérification de votre adresse email - FloodAI'
            message = f"""
Bonjour {user_data['first_name']},

Merci de vous être inscrit sur FloodAI. Pour activer votre compte, veuillez cliquer sur le lien ci-dessous:

{verification_url}

Ce lien expire dans 24 heures.

L'équipe FloodAI
"""
            from_email = settings.DEFAULT_FROM_EMAIL
            recipient_list = [user_data['email']]
            send_mail(subject, message, from_email, recipient_list, fail_silently=False)
            
            # Rediriger vers la page de vérification d'email
            return render(request, 'email_verification_sent.html', {'email': user_data['email']})
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
    
    # Récupérer les villes auxquelles l'utilisateur est abonné
    villes_abonnees = abonnements.values_list('ville', flat=True)
    
    # Récupérer uniquement les alertes pour les villes auxquelles l'utilisateur est abonné
    alertes_recentes = Alerte.objects.filter(ville__in=villes_abonnees).order_by('-date_creation')[:5]
    alertes = [f"{alerte.titre} ({alerte.date_creation.strftime('%d/%m/%Y')}) : {alerte.message}" for alerte in alertes_recentes]
    
    # Si pas d'alertes, ajouter un message d'information
    if not alertes:
        alertes = [
            "Aucune alerte en cours pour vos villes abonnées.",
            "Vous recevrez des notifications en cas de risque d'inondation dans les villes que vous suivez."
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

# Fonction pour envoyer un email à tous les abonnés d'une ville (accès staff seulement)
@login_required
def send_city_alert(request):
    # Vérifier que l'utilisateur est membre du staff
    if not request.user.is_staff:
        messages.error(request, "Vous n'avez pas les autorisations nécessaires pour effectuer cette action.")
        return redirect('profil')
    
    # Vérifier que c'est une requête POST
    if request.method != 'POST':
        return redirect('profil')
    
    # Récupérer les données du formulaire
    city_name = request.POST.get('city_name', '').strip()
    alert_message = request.POST.get('alert_message', '').strip()
    
    if not city_name or not alert_message:
        messages.error(request, "Le nom de la ville et le message d'alerte sont requis.")
        return redirect('profil')
    
    # Récupérer tous les abonnements pour cette ville
    abonnements = AbonnementVille.objects.filter(ville__iexact=city_name)
    
    if not abonnements.exists():
        messages.warning(request, f"Aucun utilisateur n'est abonné à {city_name}.")
        return redirect('profil')
    
    # La date du jour pour l'alerte
    today_date = date.today()
    today_formatted = today_date.strftime('%d/%m/%Y')
    
    # Calculer un niveau de risque aléatoire
    # On pourrait le remplacer par une vraie logique basée sur des données météo
    risk_level = random.randint(1, 3)  # 1 = faible, 2 = moyen, 3 = élevé
    risk_text = "faible" if risk_level == 1 else "moyen" if risk_level == 2 else "élevé"
    
    # Formater le titre en fonction du niveau de risque
    alert_title = f"Alerte niveau {risk_level} - {city_name}"
    
    # Préparer l'email
    subject = f"ALERTE INONDATION ({risk_text}) - {city_name} - {today_formatted}"
    
    # Compter le nombre d'emails envoyés
    emails_sent = 0
    emails_failed = 0
    
    # Enregistrer l'alerte dans la base de données
    alerte = Alerte.objects.create(
        titre=alert_title,
        message=alert_message,
        ville=city_name,
        niveau=risk_level,
        date_creation=today_date
    )
    
    # Pour chaque abonnement, envoyer un email à l'utilisateur associé
    for abonnement in abonnements:
        user = abonnement.user
        
        if not user.email:
            # Ignorer les utilisateurs sans email
            emails_failed += 1
            continue
        
        # Message personnalisé avec le nom de l'utilisateur et les coordonnées GPS
        message = f"""
Bonjour {user.first_name} {user.last_name},

ALERTE MÉTÉOROLOGIQUE POUR {city_name.upper()} - NIVEAU DE RISQUE: {risk_text.upper()}

{alert_message}

Cette alerte est générée le {today_formatted} pour les coordonnées:
Latitude: {abonnement.latitude:.6f}
Longitude: {abonnement.longitude:.6f}

Recommandations:
{get_recommendations_by_risk_level(risk_level)}

Prenez les précautions nécessaires et suivez les consignes de sécurité locales.

Vous recevez cet email car vous êtes abonné aux alertes pour {city_name}.
Pour vous désabonner, connectez-vous à votre profil sur FireFloodAI.

Cordialement,
L'équipe FireFloodAI
        """
        
        try:
            # Envoyer l'email
            from django.conf import settings
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                fail_silently=False,
            )
            emails_sent += 1
        except Exception as e:
            # Log l'erreur
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Erreur d'envoi d'email à {user.email}: {str(e)}")
            emails_failed += 1
    
    # Message de confirmation
    if emails_sent > 0:
        messages.success(request, f"Alerte envoyée à {emails_sent} abonné(s) de {city_name}. Échecs: {emails_failed}")
    else:
        messages.error(request, f"Impossible d'envoyer les alertes. {emails_failed} échecs.")
    
    return redirect('profil')

# Fonction utilitaire pour obtenir des recommandations en fonction du niveau de risque
def get_recommendations_by_risk_level(risk_level):
    if risk_level == 1:  # Faible
        return """
- Restez informé des prévisions météorologiques.
- Assurez-vous que vos gouttières et systèmes d'évacuation d'eau sont dégagés.
- Vérifiez que vous avez une trousse d'urgence à jour.
"""
    elif risk_level == 2:  # Moyen
        return """
- Mettez en hauteur les objets de valeur et les documents importants.
- Préparez-vous à couper l'électricité si nécessaire.
- Évitez de vous déplacer dans les zones basses.
- Ayez un moyen de communication d'urgence chargé et fonctionnel.
"""
    else:  # Élevé (3)
        return """
- RESTEZ À L'ABRI et ne vous déplacez pas sauf instructions contraires des autorités.
- Si vous êtes à l'extérieur, cherchez un terrain élevé.
- Ne traversez JAMAIS une zone inondée à pied ou en voiture.
- Préparez-vous à évacuer si les autorités le demandent.
- Gardez votre trousse d'urgence à portée de main.
- Coupez le gaz et l'électricité si vous êtes dans une zone inondée.
"""

# Vue pour vérifier l'email
def verify_email(request, token):
    try:
        from .models import EmailVerification, UserProfile
        verification = get_object_or_404(EmailVerification, token=token, is_verified=False)
        
        # Vérifier si le token n'a pas expiré
        if verification.is_expired:
            messages.error(request, 'Le lien de vérification a expiré. Veuillez vous inscrire à nouveau.')
            verification.delete()
            return redirect('register')
        
        # Vérifier si l'utilisateur existe déjà avec cet email
        if User.objects.filter(email=verification.email).exists():
            messages.error(request, 'Un compte avec cet email existe déjà.')
            verification.delete()
            return redirect('login')
        
        # Créer l'utilisateur
        user = User.objects.create_user(
            username=verification.username,
            email=verification.email,
            password=verification.password,
            first_name=verification.first_name,
            last_name=verification.last_name,
        )
        
        # Créer le profil utilisateur avec les informations d'adresse
        UserProfile.objects.create(
            user=user,
            adresse=verification.adresse,
            ville=verification.ville,
            code_postal=verification.code_postal,
        )
        
        # Marquer la vérification comme effectuée
        verification.is_verified = True
        verification.save()
        
        # Connecter l'utilisateur
        login(request, user)
        messages.success(request, f'Votre compte a été activé avec succès! Bienvenue {user.first_name}!')
        return redirect('home')
    
    except Exception as e:
        messages.error(request, f"Erreur lors de la vérification de l'email: {str(e)}")
        return redirect('register')

# Vue de mot de passe oublié
def forgot_password(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        
        # Vérifier si l'email existe
        if not User.objects.filter(email=email).exists():
            messages.error(request, 'Aucun compte associé à cet email.')
            return render(request, 'forgot_password.html')
        
        # Supprimer les anciens tokens pour cet email
        PasswordResetToken.objects.filter(email=email).delete()
        
        # Créer un nouveau token
        reset_token = PasswordResetToken.objects.create(email=email)
        
        # Envoyer l'email
        reset_url = request.build_absolute_uri(f'/reset-password/{reset_token.token}/')
        subject = 'Réinitialisation de votre mot de passe - FloodAI'
        message = f"""
Bonjour,

Vous avez demandé une réinitialisation de mot de passe. Veuillez cliquer sur le lien ci-dessous pour créer un nouveau mot de passe:

{reset_url}

Ce lien expire dans 24 heures.

Si vous n'avez pas demandé cette réinitialisation, veuillez ignorer cet email.

L'équipe FloodAI
"""
        from_email = settings.DEFAULT_FROM_EMAIL
        recipient_list = [email]
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)
        
        messages.success(request, 'Un email de réinitialisation a été envoyé à votre adresse.')
        return redirect('login')
    
    return render(request, 'forgot_password.html')

# Vue pour réinitialiser le mot de passe avec le token
def reset_password(request, token):
    # Récupérer le token
    try:
        reset_token = PasswordResetToken.objects.get(token=token)
        
        # Vérifier si le token est expiré ou utilisé
        if reset_token.is_expired or reset_token.used:
            messages.error(request, 'Ce lien de réinitialisation est expiré ou a déjà été utilisé.')
            return redirect('forgot_password')
        
        # Récupérer l'utilisateur
        try:
            user = User.objects.get(email=reset_token.email)
        except User.DoesNotExist:
            messages.error(request, 'Aucun utilisateur associé à cet email.')
            return redirect('forgot_password')
        
        if request.method == 'POST':
            form = SetPasswordForm(user, request.POST)
            if form.is_valid():
                form.save()
                
                # Marquer le token comme utilisé
                reset_token.used = True
                reset_token.save()
                
                messages.success(request, 'Votre mot de passe a été réinitialisé avec succès. Vous pouvez maintenant vous connecter.')
                return redirect('login')
        else:
            form = SetPasswordForm(user)
        
        return render(request, 'reset_password.html', {'form': form})
    
    except PasswordResetToken.DoesNotExist:
        messages.error(request, 'Lien de réinitialisation invalide.')
        return redirect('forgot_password')
