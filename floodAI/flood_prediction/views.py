from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Alerte
from django.db.models import Q
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django import forms
from django.http import JsonResponse
import json

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
        }
    }
    return render(request, 'home.html', context)

# Vue pour gérer les abonnements aux régions

# Vue pour s'abonner à une région (fonction simplifiée)
@login_required
def subscribe_region(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            region_name = data.get('region_name', 'Région inconnue')
            
            # Fonctionnalité simplifiée - juste retourner un message de succès
            return JsonResponse({'success': True, 'message': f'Abonnement à la région {region_name} enregistré'})
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
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

# Vue profil (protégée par login_required)
@login_required
def profil(request):
    user = request.user
    
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
    
    return render(request, 'profil.html', {'infos': infos, 'alertes': alertes})
