// Initialisation de la carte
let map = null;
let selectedMarker = null;

// Fonction pour formater la date en français
function formatDate(date) {
    const options = { weekday: 'long', day: 'numeric', month: 'long' };
    return date.toLocaleDateString('fr-FR', options);
}

// Fonction pour récupérer les données météo via la vue Django
async function fetchWeatherData(lat, lng) {
    try {
        const response = await fetch(`/api/weather/?lat=${lat}&lng=${lng}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Erreur lors de la récupération des données météo:', error);
        return null;
    }
}

// Fonction pour récupérer les informations de lieu via la vue Django
async function fetchLocationInfo(lat, lng) {
    try {
        const response = await fetch(`/api/location/?lat=${lat}&lng=${lng}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Erreur lors de la récupération des informations de lieu:', error);
        return null;
    }
}

// Fonction d'initialisation de la carte
function initMap() {
    if (map !== null) {
        map.remove(); // Supprime la carte existante si elle existe
    }

    // Création de la carte
    map = L.map('map', {
        zoomControl: false,
        attributionControl: true,
        maxBounds: [[-90, -180], [90, 180]], // Limites de la carte
        maxBoundsViscosity: 1.0, // Force avec laquelle la carte reste dans les limites
        minZoom: 2, // Zoom minimum pour éviter la duplication
        worldCopyJump: false // Désactive la duplication de la carte
    }).setView([46.603354, 1.888334], 6);

    // Ajout du fond de carte OpenStreetMap
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
        noWrap: true, // Empêche la duplication horizontale des tuiles
        bounds: [[-90, -180], [90, 180]] // Limites des tuiles
    }).addTo(map);

    // Ajout du contrôle de zoom personnalisé
    const zoomControl = L.control.zoom({
        position: 'topleft',
        zoomInText: '+',
        zoomOutText: '−'
    }).addTo(map);

    // Configuration du contrôle de localisation
    const locateControl = L.control.locate({
        position: 'topleft',
        icon: 'fa fa-location-crosshairs',
        iconLoading: 'fa fa-spinner fa-spin',
        strings: {
            title: "Ma position"
        },
        locateOptions: {
            enableHighAccuracy: true,
            maxZoom: 15
        },
        flyTo: true,
        showCompass: false,
        showPopup: false,
        markerClass: L.Marker,
        markerStyle: {
            icon: L.divIcon({
                className: 'location-marker',
                html: '<img src="/static/images/location-marker.png" alt="Location marker">',
                iconSize: [32, 32],
                iconAnchor: [16, 32]
            })
        }
    }).addTo(map);

    // Gestionnaire d'événements pour le clic sur la carte
    map.on('click', function(e) {
        createMarker(e.latlng.lat, e.latlng.lng);
    });

    // Localiser l'utilisateur au chargement
    if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(function(position) {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            
            // Centrer la carte sur la position de l'utilisateur
            map.setView([lat, lng], 13);
            
            // Créer le marqueur à la position de l'utilisateur
            createMarker(lat, lng);
        }, function(error) {
            console.log("Erreur de géolocalisation:", error);
        });
    }
}

// Fonction pour créer un marqueur
async function createMarker(lat, lng) {
    // Supprime l'ancien marqueur s'il existe
    if (selectedMarker) {
        map.removeLayer(selectedMarker);
    }

    // Crée un nouveau marqueur avec une icône PNG
    selectedMarker = L.marker([lat, lng], {
        icon: L.divIcon({
            className: 'location-marker',
            html: '<img src="/static/images/location-marker.png" alt="Location marker">',
            iconSize: [32, 32],
            iconAnchor: [16, 32]
        })
    }).addTo(map);

    // Créer ou récupérer le conteneur pour les infos de localisation et afficher un indicateur de chargement
    let locationInfoDiv = document.getElementById('location-info');
    if (!locationInfoDiv) {
        const sidebar = document.querySelector('.sidebar');
        locationInfoDiv = document.createElement('div');
        locationInfoDiv.id = 'location-info';
        sidebar.insertBefore(locationInfoDiv, sidebar.firstChild);
    }
    
    locationInfoDiv.innerHTML = `
        <div class="location-header">
            <h3>Lieu sélectionné</h3>
            <div class="loading-location">
                <span class="spinner"></span> Recherche de la localité...
            </div>
        </div>
    `;

    // Met à jour les prévisions météo immédiatement
    updatePredictions(lat, lng);
    
    // Récupérer les informations de localisation via Nominatim en parallèle
    fetchLocationInfo(lat, lng)
        .then(locationInfo => {
            if (locationInfo) {
                updateLocationInfo(locationInfo);
            }
        })
        .catch(error => {
            console.error("Erreur lors de la récupération des données de localisation:", error);
            // Afficher un message simple en cas d'échec
            locationInfoDiv.innerHTML = `
                <div class="location-header">
                    <h3>Lieu sélectionné</h3>
                    <div class="location-name">Coordonnées: ${lat.toFixed(4)}, ${lng.toFixed(4)}</div>
                </div>
            `;
        });
}

// Styles pour les zones de risque
const riskStyles = {
    high: {
        color: '#d93025',
        fillColor: '#d93025',
        fillOpacity: 0.2,
        weight: 2
    },
    medium: {
        color: '#f29900',
        fillColor: '#f29900',
        fillOpacity: 0.2,
        weight: 2
    },
    low: {
        color: '#188038',
        fillColor: '#188038',
        fillOpacity: 0.2,
        weight: 2
    }
};

// Fonction pour mettre à jour les prévisions
async function updatePredictions(lat, lng) {
    const weatherData = await fetchWeatherData(lat, lng);
    
    if (!weatherData) {
        console.error('Impossible de récupérer les données météo');
        return;
    }

    const riskLevels = document.querySelector('.risk-levels');
    riskLevels.innerHTML = '';

    const hourlyData = weatherData.hourly;
    const now = new Date();
    const currentHour = now.getHours();
    const forecasts = [];

    // Créer 4 prévisions (aujourd'hui et 3 jours suivants)
    for (let i = 0; i < 4; i++) {
        const date = new Date(now);
        date.setDate(date.getDate() + i);
        
        // Utiliser la même heure pour chaque jour
        const hourIndex = i * 24 + currentHour;
        
        forecasts.push({
            date: date,
            temp: hourlyData.temperature_2m[hourIndex],
            precip: hourlyData.precipitation[hourIndex],
            humidity: hourlyData.relative_humidity_2m[hourIndex],
            windSpeed: hourlyData.wind_speed_10m[hourIndex]
        });
    }

    forecasts.forEach(forecast => {
        let riskLevel = 'Faible';
        if (forecast.precip > 50 || forecast.humidity > 80) {
            riskLevel = 'Élevé';
        } else if (forecast.precip > 25 || forecast.humidity > 60) {
            riskLevel = 'Moyen';
        }

        const riskClass = riskLevel.toLowerCase() === 'élevé' ? 'high' : 
                         riskLevel.toLowerCase() === 'moyen' ? 'medium' : 'low';

        const predHtml = `
            <div class="risk-item ${riskClass}">
                <div class="risk-date">${formatDate(forecast.date)}</div>
                <div class="risk-info">
                    <div class="risk-level">Risque d'inondation : ${riskLevel}</div>
                    <div class="weather-info">
                        <div>Température : ${forecast.temp.toFixed(1)}°C</div>
                        <div>Précipitations : ${forecast.precip.toFixed(1)}mm</div>
                        <div>Humidité : ${forecast.humidity}%</div>
                        <div>Vent : ${forecast.windSpeed.toFixed(1)} km/h</div>
                    </div>
                </div>
            </div>
        `;
        riskLevels.innerHTML += predHtml;
    });
}

// Fonction pour afficher les informations de localisation dans la barre latérale
async function updateLocationInfo(locationData) {
    // Créer ou récupérer le conteneur pour les infos de localisation
    let locationInfoDiv = document.getElementById('location-info');
    if (!locationInfoDiv) {
        const sidebar = document.querySelector('.sidebar');
        locationInfoDiv = document.createElement('div');
        locationInfoDiv.id = 'location-info';
        sidebar.insertBefore(locationInfoDiv, sidebar.firstChild);
    }
    
    // Extraire les informations pertinentes
    const address = locationData.address || {};
    let locationName = '';
    
    // Priorité des informations de localisation simplifiée pour plus de rapidité
    for (const key of ['city', 'town', 'village']) {
        if (address[key]) {
            locationName = address[key];
            break;
        }
    }
    
    if (!locationName && locationData.display_name) {
        // Si on ne trouve pas de ville/village, utiliser le premier segment du display_name
        const displayParts = locationData.display_name.split(',');
        locationName = displayParts[0].trim();
    }
    
    if (!locationName) {
        locationName = 'Lieu sélectionné';
    }
    
    // Afficher temporairement les infos de base pendant la vérification d'abonnement
    locationInfoDiv.innerHTML = `
        <div class="location-header">
            <h3>Lieu sélectionné</h3>
            <div class="location-info-row">
                <div class="location-name">${locationName}</div>
                <div class="subscription-loading"><span class="spinner"></span></div>
            </div>
        </div>
    `;
    
    // Vérifier si l'utilisateur est connecté (présence du nom de profil)
    const isLoggedIn = document.querySelector('.profile-name');
    
    let isSubscribed = false;
    
    // Si l'utilisateur est connecté, vérifier s'il est abonné à cette ville
    if (isLoggedIn) {
        try {
            const response = await fetch(`/api/check-subscription/?city_name=${encodeURIComponent(locationName)}`);
            const data = await response.json();
            isSubscribed = data.success && data.is_subscribed;
        } catch (error) {
            console.error('Erreur lors de la vérification de l\'abonnement:', error);
        }
    }
    
    // Construire le HTML pour afficher les informations de localisation
    let locationHtml = `
        <div class="location-header">
            <h3>Lieu sélectionné</h3>
            <div class="location-info-row">
                <div class="location-name">${locationName}</div>
    `;
    
    // Ajouter le bouton approprié selon l'état de connexion et d'abonnement
    if (isLoggedIn) {
        if (isSubscribed) {
            locationHtml += `
                <button class="subscribe-btn subscribed" type="button" onclick="toggleCitySubscription('${locationName}', 'unsubscribe')">
                    <i class="fas fa-bell-slash"></i> Désabonner
                </button>
            `;
        } else {
            locationHtml += `
                <button class="subscribe-btn" type="button" onclick="toggleCitySubscription('${locationName}', 'subscribe')">
                    <i class="fas fa-bell"></i> S'abonner
                </button>
            `;
        }
    } else {
        locationHtml += `
            <button class="login-required-btn" type="button" onclick="window.location.href='/login/'">
                <i class="fas fa-sign-in-alt"></i> Se connecter
            </button>
        `;
    }
    
    locationHtml += `
            </div>
        </div>
    `;
    
    // Afficher les informations
    locationInfoDiv.innerHTML = locationHtml;
}

// Fonction pour basculer l'état de l'abonnement (s'abonner ou se désabonner)
async function toggleCitySubscription(cityName, action) {
    if (!selectedMarker) return;
    
    const lat = selectedMarker.getLatLng().lat;
    const lng = selectedMarker.getLatLng().lng;
    
    try {
        const response = await fetch('/api/subscribe-city/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({
                city_name: cityName,
                lat: lat,
                lng: lng,
                action: action
            })
        });
        
        const data = await response.json();
        if (data.success) {
            // Afficher une notification de succès
            showNotification(data.message, 'success');
            
            // Mettre à jour le bouton selon l'état de l'abonnement
            const subscribeBtn = document.querySelector('.subscribe-btn');
            if (subscribeBtn) {
                if (action === 'subscribe') {
                    subscribeBtn.innerHTML = '<i class="fas fa-bell-slash"></i> Désabonner';
                    subscribeBtn.classList.add('subscribed');
                    subscribeBtn.onclick = () => toggleCitySubscription(cityName, 'unsubscribe');
                } else {
                    subscribeBtn.innerHTML = '<i class="fas fa-bell"></i> S\'abonner';
                    subscribeBtn.classList.remove('subscribed');
                    subscribeBtn.onclick = () => toggleCitySubscription(cityName, 'subscribe');
                }
            }
        } else {
            showNotification('Erreur: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Erreur lors de la modification de l\'abonnement:', error);
        showNotification('Erreur de connexion', 'error');
    }
}

// Fonction pour obtenir le token CSRF
function getCsrfToken() {
    const cookieValue = document.cookie
        .split('; ')
        .find(row => row.startsWith('csrftoken='))
        ?.split('=')[1];
    return cookieValue || '';
}

// Fonction pour afficher une notification
function showNotification(message, type) {
    const notificationArea = document.getElementById('notifications-container') || createNotificationArea();
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button class="close-btn">×</button>
    `;
    
    notificationArea.appendChild(notification);
    
    // Supprimer après 5 secondes
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 500);
    }, 5000);
    
    // Supprimer au clic sur le bouton de fermeture
    notification.querySelector('.close-btn').addEventListener('click', () => {
        notification.remove();
    });
}

// Fonction pour créer la zone de notification si elle n'existe pas
function createNotificationArea() {
    const area = document.createElement('div');
    area.id = 'notifications-container';
    area.className = 'notifications-container';
    document.body.appendChild(area);
    return area;
}

// Initialisation de la carte au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    initMap();
});