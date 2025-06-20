// Initialisation de la carte
let map = null;
let selectedMarker = null;
let selectedLocation = null;

// Fonction pour formater la date en français
function formatDate(date) {
    const options = { weekday: 'long', day: 'numeric', month: 'long' };
    return date.toLocaleDateString('fr-FR', options);
}

// Récupérer le CSRF token
function getCsrfToken() {
    return document.cookie
        .split('; ')
        .find(row => row.startsWith('csrftoken='))
        ?.split('=')[1] || '';
}

// Appels API
async function fetchWeatherData(lat, lng) {
    try {
        const res = await fetch(`/api/weather/?lat=${lat}&lng=${lng}`);
        return await res.json();
    } catch (err) {
        console.error('Erreur météo :', err);
        return null;
    }
}

async function fetchCombinedPredictions(lat, lng) {
    try {
        const formData = new FormData();
        formData.append('input_data', JSON.stringify({ latitude: lat, longitude: lng }));
        const res = await fetch('/api/combined-prediction/', {
            method: 'POST',
            headers: { 'X-CSRFToken': getCsrfToken() },
            body: formData
        });
        return await res.json();
    } catch (err) {
        console.error('Erreur prédictions combinées :', err);
        return null;
    }
}

async function fetchLocationInfo(lat, lng) {
    try {
        const res = await fetch(`/api/location/?lat=${lat}&lng=${lng}`);
        return await res.json();
    } catch (err) {
        console.error('Erreur location :', err);
        return null;
    }
}

// Initialisation de Leaflet
function initMap() {
    if (map) return;

    map = L.map('map', {
        zoomControl: false,
        attributionControl: true,
        maxBounds: [[-90, -180], [90, 180]],
        maxBoundsViscosity: 1.0,
        minZoom: 2,
        worldCopyJump: false
    }).setView([46.603354, 1.888334], 6);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
        noWrap: true,
        bounds: [[-90, -180], [90, 180]]
    }).addTo(map);

    L.control.zoom({ position: 'topleft', zoomInText: '+', zoomOutText: '−' }).addTo(map);

    L.control.locate({
        position: 'topleft',
        icon: 'fa fa-location-crosshairs',
        flyTo: true,
        markerStyle: {
            icon: L.divIcon({
                className: 'location-marker',
                html: '<img src="/static/images/location-marker.png" alt="Location marker">',
                iconSize: [32, 32],
                iconAnchor: [16, 32]
            })
        }
    }).addTo(map);

    map.on('click', e => selectLocation(e.latlng.lat, e.latlng.lng));

    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(pos => {
            const { latitude: lat, longitude: lng } = pos.coords;
            map.setView([lat, lng], 13);
            selectLocation(lat, lng);
        });
    }

    document.getElementById('predict-button')
        .addEventListener('click', () => predictRisks());
}

// Sélection d'un point sur la carte
async function selectLocation(lat, lng) {
    // supprime ancien marqueur
    if (selectedMarker) map.removeLayer(selectedMarker);

    // masque guidance
    const ug = document.getElementById('user-guidance');
    if (ug) ug.style.display = 'none';

    // nouveau marker
    selectedMarker = L.marker([lat, lng], {
        icon: L.divIcon({
            className: 'location-marker',
            html: '<img src="/static/images/location-marker.png" alt="Location marker">',
            iconSize: [32, 32],
            iconAnchor: [16, 32]
        })
    }).addTo(map);

    selectedLocation = { lat, lng };

    // container location-info
    let locDiv = document.getElementById('location-info');
    if (!locDiv) {
        const sidebar = document.querySelector('.sidebar');
        locDiv = document.createElement('div');
        locDiv.id = 'location-info';
        sidebar.insertBefore(locDiv, sidebar.firstChild);
    }
    locDiv.innerHTML = `
        <div class="location-header">
            <h3>Lieu sélectionné</h3>
            <div class="loading-location">
                <span class="spinner"></span> Recherche de la localité…
            </div>
        </div>`;

    // afficher le bouton de prédiction
    const btnWrapper = document.getElementById('prediction-button-container');
    if (btnWrapper) btnWrapper.style.display = 'block';

    // Utiliser l'élément risk-levels existant dans le HTML
    const riskLevels = document.querySelector('.sidebar .risk-levels');
    if (riskLevels) {
        riskLevels.innerHTML = '';
        riskLevels.style.display = 'none';
    }

    // récupérer et afficher le nom de la ville
    try {
        const info = await fetchLocationInfo(lat, lng);
        if (info) updateLocationInfo(info);
    } catch {
        locDiv.innerHTML = `
          <div class="location-header">
            <h3>Lieu sélectionné</h3>
            <div class="location-name">${lat.toFixed(4)}, ${lng.toFixed(4)}</div>
          </div>`;
    }
}

// Quand on clique sur "Prédire les risques"
async function predictRisks() {
    if (!selectedLocation) {
        showNotification('Veuillez d\'abord sélectionner un lieu.', 'error');
        return;
    }
    document.getElementById('prediction-button-container').style.display = 'none';
    document.getElementById('prediction-loading').style.display = 'block';

    try {
        const weather = await fetchWeatherData(selectedLocation.lat, selectedLocation.lng);
        if (!weather) throw new Error();
        await updatePredictions(selectedLocation.lat, selectedLocation.lng);
    } catch {
        showNotification('Erreur lors du calcul des prédictions.', 'error');
    } finally {
        document.getElementById('prediction-loading').style.display = 'none';
        document.getElementById('prediction-button-container').style.display = 'block';
    }
}

// Met à jour le HTML des prédictions
async function updatePredictions(lat, lng) {
    const riskLevels = document.querySelector('.sidebar .risk-levels');
    if (!riskLevels) return console.error('.risk-levels introuvable');

    // on l'affiche là où il est
    riskLevels.style.display = 'block';
    riskLevels.innerHTML = '<div class="loading-predictions"><span class="spinner"></span> Chargement des prédictions…</div>';

    // essai combiné CNN+LSTM
    try {
        const comb = await fetchCombinedPredictions(lat, lng);
        if (comb?.prediction) {
            const p = comb.prediction;
            riskLevels.innerHTML = '';

            // CNN (J)
            if (p.cnn_prediction) {
                const c = p.cnn_prediction;
                let cls = 'low', lvl = c.risk_level || 'Faible';
                if (lvl.toLowerCase().includes('élevé')) { cls = 'high'; lvl = 'Élevé'; }
                else if (lvl.toLowerCase().includes('modéré')) { cls = 'medium'; lvl = 'Modéré'; }
                
                const weather_html = c.weather ? `
                    <div>Temp: ${c.weather.temp.toFixed(1)}°C</div>
                    <div>Précip: ${c.weather.precip.toFixed(1)}mm</div>
                    <div>Humidité: ${c.weather.humidity.toFixed(0)}%</div>
                    <div>Probabilité d'inondation : ${c.flood_percentage}</div>
                ` : '';

                riskLevels.innerHTML += `
                  <div class="risk-item ${cls}">
                    <div class="risk-date">${c.date} (aujourd'hui)</div>
                    <div class="risk-info">
                      <div class="risk-level">Risque d'inondation : ${lvl}</div>
                      <div class="weather-info">
                        <div>Probabilité : ${c.flood_percentage}%</div>
                        ${weather_html}
                        <div>Modèle : CNN</div>
                      </div>
                    </div>
                  </div>`;
            }

            // LSTM (J+1, J+2, J+3…)
            if (p.lstm_predictions?.length) {
                p.lstm_predictions.forEach(l => {
                    let cls = 'low', lvl = l.risk_level || 'Faible';
                    if (lvl.toLowerCase().includes('élevé')) { cls = 'high'; lvl = 'Élevé'; }
                    else if (lvl.toLowerCase().includes('modéré')) { cls = 'medium'; lvl = 'Modéré'; }
                    
                    const weather_html = l.weather ? `
                        <div>Temp: ${l.weather.temp.toFixed(1)}°C</div>
                        <div>Précip: ${l.weather.precip.toFixed(1)}mm</div>
                        <div>Humidité: ${l.weather.humidity.toFixed(0)}%</div>
                    ` : '';

                    riskLevels.innerHTML += `
                      <div class="risk-item ${cls}">
                        <div class="risk-date">${l.date}</div>
                        <div class="risk-info">
                          <div class="risk-level">Risque : ${lvl}</div>
                          <div class="weather-info">
                            <div>Probabilité : ${l.probability}%</div>
                            <div>Inondation : ${l.is_flooded ? 'Oui' : 'Non'}</div>
                            ${weather_html}
                            <div>Modèle : LSTM</div>
                          </div>
                        </div>
                      </div>`;
                });
            }

            // visuels CNN & plot LSTM (supprimés à la demande de l'utilisateur)
            /* if (p.visualizations?.output_image) {
                const img = document.createElement('img');
                img.src = p.visualizations.output_image;
                img.alt = 'Visu CNN';
                img.className = 'prediction-image';
                riskLevels.parentNode.insertBefore(img, riskLevels);
            }
            if (p.plot_base64) {
                const plot = document.createElement('img');
                plot.src = p.plot_base64;
                plot.alt = 'Graphique LSTM';
                plot.className = 'prediction-plot';
                riskLevels.parentNode.appendChild(plot);
            } */
            return;
        }
    } catch {
        console.warn('Prédictions combinées KO, fallback météo');
    }

    // fallback météo simple sur 4 jours
    const weather = await fetchWeatherData(lat, lng);
    const now = new Date(), h = now.getHours();
    const hourly = weather.hourly;
    const forecasts = [];
    for (let i = 0; i < 4; i++) {
        const d = new Date(now);
        d.setDate(d.getDate() + i);
        const idx = i * 24 + h;
        forecasts.push({
            date: d,
            temp: hourly.temperature_2m[idx],
            precip: hourly.precipitation[idx],
            humidity: hourly.relative_humidity_2m[idx],
            windSpeed: hourly.wind_speed_10m[idx]
        });
    }

    riskLevels.innerHTML = '';
    forecasts.forEach(f => {
        let lvl = 'Faible';
        if (f.precip > 50 || f.humidity > 80) lvl = 'Élevé';
        else if (f.precip > 25 || f.humidity > 60) lvl = 'Moyen';
        const cls = lvl === 'Élevé' ? 'high' : lvl === 'Moyen' ? 'medium' : 'low';
        riskLevels.innerHTML += `
          <div class="risk-item ${cls}">
            <div class="risk-date">${formatDate(f.date)}</div>
            <div class="risk-info">
              <div class="risk-level">Risque : ${lvl}</div>
              <div class="weather-info">
                <div>Temp : ${f.temp.toFixed(1)}°C</div>
                <div>Précip : ${f.precip.toFixed(1)}mm</div>
                <div>Humidité : ${f.humidity}%</div>
                <div>Vent : ${f.windSpeed.toFixed(1)} km/h</div>
              </div>
            </div>
          </div>`;
    });
}

// Mise à jour des infos de lieu / abonnement
async function updateLocationInfo(locationData) {
    const address = locationData.address || {};
    let name = address.city || address.town || address.village 
               || locationData.display_name.split(',')[0] || 'Lieu sélectionné';

    const locDiv = document.getElementById('location-info');
    const logged = !!document.querySelector('.profile-name');

    let subscribed = false;
    if (logged) {
        try {
            const res = await fetch(`/api/check-subscription/?city_name=${encodeURIComponent(name)}`);
            const json = await res.json();
            subscribed = json.success && json.is_subscribed;
        } catch {}
    }

    const btn = logged
      ? subscribed
        ? `<button class="subscribe-btn subscribed" onclick="toggleCitySubscription('${name}','unsubscribe')">
             <i class="fas fa-bell-slash"></i> Désabonner
           </button>`
        : `<button class="subscribe-btn" onclick="toggleCitySubscription('${name}','subscribe')">
             <i class="fas fa-bell"></i> S'abonner
           </button>`
      : `<button class="login-required-btn" onclick="window.location.href='/login/'">
           <i class="fas fa-sign-in-alt"></i> Se connecter
         </button>`;

    locDiv.innerHTML = `
      <div class="location-header">
        <h3>Lieu sélectionné</h3>
        <div class="location-info-row">
          <div class="location-name">${name}</div>
          ${btn}
        </div>
      </div>`;
}

// Toggle abonnement
async function toggleCitySubscription(cityName, action) {
    if (!selectedMarker) return;
    const { lat, lng } = selectedMarker.getLatLng();
    try {
        const res = await fetch('/api/subscribe-city/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({ city_name: cityName, lat, lng, action })
        });
        const json = await res.json();
        showNotification(json.message || json.error, json.success ? 'success' : 'error');
        updateLocationInfo({ address: {}, display_name: cityName });
    } catch {
        showNotification('Erreur de connexion', 'error');
    }
}

// Notifications
function showNotification(msg, type) {
    let area = document.getElementById('notifications-container');
    if (!area) {
        area = document.createElement('div');
        area.id = 'notifications-container';
        document.body.appendChild(area);
    }
    const n = document.createElement('div');
    n.className = `notification ${type}`;
    n.innerHTML = `<span>${msg}</span><button class="close-btn">×</button>`;
    area.appendChild(n);
    setTimeout(() => { n.classList.add('fade-out'); setTimeout(() => n.remove(), 500); }, 5000);
    n.querySelector('.close-btn').onclick = () => n.remove();
}

// Démarrage
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    console.log('Carte initialisée');
});