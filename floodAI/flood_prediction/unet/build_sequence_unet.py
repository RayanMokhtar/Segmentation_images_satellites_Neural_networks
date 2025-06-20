import torch
import numpy as np
import rasterio
import requests
import time
from scipy.interpolate import griddata
from skimage.transform import resize
import os 
import requests
import numpy as np
import torch
import rasterio
from datetime import date, datetime
import xarray as xr
import tempfile

def obtenir_tenseur_precipitation(south, north, west, east, api_key, target_shape=(512, 512), 
                                 grid_size=5, save_tif=True , is_normalized=True):
    """
    Télécharge les données de précipitations et les convertit en tenseur PyTorch
    
    Args:
        south, north, west, east: Coordonnées de la zone d'intérêt
        api_key: Clé API OpenWeatherMap
        target_shape: Taille cible du tenseur (hauteur, largeur)
        grid_size: Nombre de points par côté dans la grille d'échantillonnage
        save_tif: Sauvegarder aussi en format GeoTIFF
        
    Returns:
        torch.Tensor: Tenseur de précipitations de forme [1, hauteur, largeur]
    """
    print(f"Récupération des données de précipitations pour la zone: {south}°S, {north}°N, {west}°E, {east}°E...")
    
    # 1. Créer une grille de points à échantillonner
    lat_points = np.linspace(south, north, grid_size)
    lon_points = np.linspace(west, east, grid_size)
    
    # Conteneur pour les données de précipitation
    precipitation_data = []
    coords = []
    
    # 2. Collecte des données de précipitations
    for lat in lat_points:
        for lon in lon_points:
            # Requête vers l'API OpenWeatherMap avec forecast pour avoir les précipitations
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extraire les précipitations des prochaines 24h (moyenne)
                precipitation_sum = 0
                count = 0
                print(f"Récupération des données pour le point ({lat:.4f}, {lon:.4f})...")
                # Parcourir les prévisions (généralement par tranches de 3h)
                for forecast in data['list'][:8]:  # 8 x 3h = 24h
                    # Récupérer les précipitations (en mm) s'il y en a
                    if 'rain' in forecast:
                        if '3h' in forecast['rain']:
                            precipitation_sum += forecast['rain']['3h']
                        elif '1h' in forecast['rain']:
                            precipitation_sum += forecast['rain']['1h']
                    count += 1
                
                # Calculer la moyenne ou utiliser 0 si pas de données
                avg_precipitation = precipitation_sum / count if count > 0 else 0
                precipitation_data.append(avg_precipitation)
                coords.append((lon, lat))
                
                print(f"Point ({lat:.4f}, {lon:.4f}): Précipitation {avg_precipitation:.2f} mm")
            else:
                print(f"Erreur API pour le point ({lat}, {lon}): {response.status_code}")
                # Valeur par défaut
                precipitation_data.append(0)
                coords.append((lon, lat))
                
    # 3. Conversion en array numpy
    precipitation_array = np.array(precipitation_data)
    coords_array = np.array(coords)
    
    # 4. Créer une grille régulière pour l'interpolation
    grid_lon, grid_lat = np.mgrid[west:east:complex(0, grid_size), 
                                 south:north:complex(0, grid_size)]
    
    grid_points = np.column_stack((grid_lon.flatten(), grid_lat.flatten()))
    precipitation_grid = griddata(coords_array, precipitation_array, grid_points, method='cubic')
    precipitation_grid = precipitation_grid.reshape(grid_lon.shape)
    
    precipitation_grid = np.maximum(precipitation_grid, 0)
    
    # 6. Redimensionner à la taille cible
    precipitation_resized = resize(precipitation_grid, target_shape, mode='edge', preserve_range=True)
    
    # 7. Convertir en tenseur PyTorch [1, H, W]
    precipitation_tensor = torch.tensor(precipitation_resized).float().unsqueeze(0)
    
    print(f"Tenseur de précipitations créé avec succès - forme: {precipitation_tensor.shape}, "
          f"min: {precipitation_tensor.min().item():.4f}, max: {precipitation_tensor.max().item():.4f}")
    
    if save_tif:
        output_file = f"precipitation_{grid_size}x{grid_size}.tif"
        
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=grid_size,
            width=grid_size,
            count=1,
            dtype=rasterio.float32,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=rasterio.transform.from_bounds(west, south, east, north, grid_size, grid_size),
        ) as dst:
            dst.write(precipitation_grid.astype(np.float32), 1)
        
        print(f"Données de précipitations sauvegardées en GeoTIFF: {output_file}")
    
    # Normaliser pour le modèle
    if is_normalized:
        precipitation_normalized = precipitation_tensor / 50.0  # Normaliser avec une échelle de 0-50mm
        precipitation_normalized = torch.clamp(precipitation_normalized, 0.0, 1.0)
    else :
        precipitation_normalized = precipitation_tensor
        print(f"Tenseur de précipitations pas normalisé - min: {precipitation_normalized.min().item():.4f}, "
        f"max: {precipitation_normalized.max().item():.4f}")
    return precipitation_normalized



def coordonnees_vers_tenseur(south, north, west, east, 
                            dem_type="SRTM", api_key=None, 
                            target_shape=(512, 512),
                            clip_min=0.0, clip_max=2000.0,
                            output_dir="./data"):
    """
    Fonction unifiée qui télécharge une image GeoTIFF à partir de coordonnées
    et la convertit directement en tenseur PyTorch.
    
    Args:
        south, north, west, east: Coordonnées de la zone d'intérêt
        dem_type: Type de données à télécharger ("SRTM", "ASTER", "COP30", etc.)
        api_key: Clé API pour le service de données (OpenTopography, etc.)
        target_shape: Dimensions cible du tenseur (hauteur, largeur)
        clip_min, clip_max: Valeurs pour l'écrêtage des données
        output_dir: Répertoire où sauvegarder les fichiers téléchargés
        
    Returns:
        torch.Tensor: Tenseur de forme [1, hauteur, largeur]
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Téléchargement des données
    # Construire le nom de fichier basé sur les coordonnées
    region_name = f"{south:.2f}_{north:.2f}_{west:.2f}_{east:.2f}"
    tif_filename = os.path.join(output_dir, f"{dem_type.lower()}_{region_name}.tif")
    
    # Vérifier si le fichier existe déjà
    
    print(f"Téléchargement des données {dem_type} pour la région {region_name}...")
    
    # Déterminer le service à utiliser selon le type de données
    if dem_type.upper() == "SRTM":
        # OpenTopography API pour SRTM
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "SRTMGL1",
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff"
        }
        
        # Ajouter la clé API si fournie
        if api_key:
            params["API_Key"] = api_key
        
        # Télécharger les données
        response = requests.get(url, params=params, stream=True)
        
        if response.status_code == 200:
            with open(tif_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Données téléchargées: {tif_filename}")
        else:
            print(f"Erreur lors du téléchargement: {response.status_code}")
            print(response.text)
            raise RuntimeError(f"Échec du téléchargement: {response.text}")
    else:
        raise ValueError(f"Type de données non supporté: {dem_type}")
    
    
    # 2. Chargement et traitement de l'image TIFF
    try:
        with rasterio.open(tif_filename) as src:
            data = src.read(1).astype(np.float32)
            print(f"Image chargée: forme={data.shape}, min={data.min():.2f}, max={data.max():.2f}")
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {e}")
        raise
    
    # 3. Prétraitement: écrêtage des valeurs
    data = np.clip(data, clip_min, clip_max)
    
    # 4. Redimensionnement à la taille cible
    if target_shape and data.shape != target_shape:
        data_resized = resize(data, target_shape, mode='constant', anti_aliasing=True, preserve_range=True)
        print(f"Image redimensionnée à {target_shape}")
    else:
        data_resized = data
    
    # 5. Normalisation (division par la valeur max)
    data_normalized = data_resized / clip_max
    
    # 6. Conversion en tenseur PyTorch [1, H, W]
    tensor = torch.tensor(data_normalized, dtype=torch.float32).unsqueeze(0)
    print(f"Tenseur créé: forme={tensor.shape}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
    
    print(f"Préparation du tenseur DEM terminée: {tensor}")
    return tensor

def prepare_dem_tensor(dem_tif_path, target_shape=None, pad_to_shape=None, clip_min=0.0, clip_max=2000.0):
    """
    Prépare un fichier DEM pour l'utiliser comme canal d'entrée du modèle U-Net
    
    Args:
        dem_tif_path (str): Chemin vers le fichier DEM GeoTIFF
        target_shape (tuple): Shape cible (hauteur, largeur) pour redimensionner
        pad_to_shape (tuple): Shape cible avec padding (hauteur, largeur)
        clip_min (float): Valeur minimale pour l'écrêtage
        clip_max (float): Valeur maximale pour l'écrêtage
        
    Returns:
        numpy.ndarray: Le DEM prétraité et normalisé prêt pour l'inférence
    """
    print(f"Préparation DEM: {dem_tif_path}")
    
    # 1. Charger le DEM
    try:
        with rasterio.open(dem_tif_path) as dem_src:
            dem_data = dem_src.read(1).astype(np.float32)
            print(f"DEM chargé: shape={dem_data.shape}, min={dem_data.min():.2f}, max={dem_data.max():.2f}")
    except Exception as e:
        print(f"Erreur lors du chargement du DEM: {e}")
        print("Utilisation d'un DEM substitut (zéros)")
        if target_shape:
            dem_data = np.zeros(target_shape, dtype=np.float32)
        else:
            raise ValueError("Si le DEM ne peut pas être chargé, target_shape doit être spécifié")
    
    # 2. Prétraitement: écrêtage et normalisation
    dem_data = np.clip(dem_data, clip_min, clip_max)
    dem_normalized = dem_data 
    
    # 3. Redimensionnement si demandé
    if target_shape and dem_data.shape != target_shape:
        from skimage.transform import resize
        dem_normalized = resize(dem_normalized, target_shape, 
                                mode='constant', anti_aliasing=True, 
                                preserve_range=True)
        print(f"DEM redimensionné à {target_shape}")
    
    # 4. Padding si nécessaire
    if pad_to_shape:
        h, w = dem_normalized.shape
        pad_h = pad_to_shape[0] - h
        pad_w = pad_to_shape[1] - w
        
        if pad_h > 0 or pad_w > 0:
            pad_h = max(0, pad_h)
            pad_w = max(0, pad_w)
            dem_normalized = np.pad(dem_normalized, ((0, pad_h), (0, pad_w)), mode='constant')
            print(f"DEM padded à {dem_normalized.shape}")
    
    print(f"DEM préparé: shape={dem_normalized.shape}, min={dem_normalized.min():.4f}, max={dem_normalized.max():.4f}")
    
    tenseur_dem = torch.tensor(dem_normalized, dtype=torch.float32).unsqueeze(0)  # Ajouter une dimension pour le batch

    return tenseur_dem




import os
import tempfile
import gzip
import shutil
import requests
import numpy as np
import torch
from datetime import datetime, date
import rasterio
from skimage.transform import resize

# ─── 1. Génération de données synthétiques en fallback ────────────────────────
def generer_donnees_synthetiques(shape, mode="realiste", date_obj=None):
    """
    Génère un array de précipitations synthétique si toutes les sources échouent.
    """
    h, w = shape
    if mode == "uniforme":
        return np.full(shape, 0.5, dtype=np.float32)
    # mode 'realiste': quelques cells intenses + fond exponentiel
    arr = np.random.exponential(0.2, size=shape).astype(np.float32)
    if date_obj:
        mois = date_obj.month
        # plus de pluie en été (hém. nord)
        i_max = 20.0 if 5 <= mois <= 9 else 5.0 if mois in [12,1,2] else 10.0
    else:
        i_max = 10.0
    # ajouter 1–3 cellules de forte précipitation
    for _ in range(np.random.randint(1,4)):
        cy, cx = np.random.randint(0,h), np.random.randint(0,w)
        rayon = np.random.randint(w//10, w//3)
        intens = np.random.uniform(2.0, i_max)
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = np.exp(-(x*x + y*y)/(2*(rayon/2)**2))
        arr += mask * intens
    return arr

# ─── 2. CHIRPS (fichiers .tif.gz) ───────────────────────────────────────────────
def essayer_chirps(year, month, day, west, south, east, north, verbose=True):
    """
    Tente de télécharger CHIRPS-2.0 (.tif.gz), décompresse et extrait la bbox.
    """
    pad = 0.05  # élargir la bbox pour éviter les bords
    west  -= pad; east  += pad
    south -= pad; north += pad
    date_str = f"{year}.{month:02d}.{day:02d}"
    # les deux chemins possibles : final ou prelim
    variants = ["prelim/global_daily",
                "global_daily"]
    for var in variants:
        url = (f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/{var}/tifs/p05/"
               f"{year}/chirps-v2.0.{date_str}.tif.gz")
        if verbose: print(f"🌐 CHIRPS try: {url}")
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code != 200:
                if verbose: print(f"  → {r.status_code} {r.reason}")
                continue
            # écrire le .gz
            tmp_gz = os.path.join(tempfile.gettempdir(), "chirps.tif.gz")
            with open(tmp_gz, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            # décompresser
            tmp_tif = tmp_gz[:-3]
            with gzip.open(tmp_gz, "rb") as f_in, open(tmp_tif, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            # extraire la fenêtre
            with rasterio.open(tmp_tif) as src:
                window = rasterio.windows.from_bounds(west, south, east, north, src.transform)
                arr = src.read(1, window=window)
            os.remove(tmp_gz)
            os.remove(tmp_tif)
            if verbose: print("✅ CHIRPS loaded")
            print(f" CHIRPS ARRAY c'est  → {arr.shape} shape, min: {arr.max()}, array :  {arr}")
            # convertir en mm/jour
            return arr
        except Exception as e:
            if verbose: print(f"  ⚠️ CHIRPS error: {e}")
    if verbose: print("❌ CHIRPS all attempts failed")
    return None

# ─── 3. NASA POWER (API daily point) ───────────────────────────────────────────
def essayer_nasa_power(year, month, day, west, south, east, north, verbose=True):
    """
    Tente la NASA POWER API en point (centre de la bbox).
    """
    lon = (west + east) / 2
    lat = (south + north) / 2
    dd = f"{year}{month:02d}{day:02d}"
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOT&community=AG&longitude={lon}&latitude={lat}"
        f"&start={dd}&end={dd}&format=JSON"
    )
    if verbose: print(f"🌐 NASA POWER try: {url}")
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        print(f"données nasa : {data}")
        val = data["properties"]["parameter"]["PRECTOTCORR"].get(dd)
        if val is None:
            if verbose: print("  → no PRECTOT for this date")
            return None
        # fabriquer un array uniforme (rés ~5km)
        h = max(10, int(abs(north - south) * 20))
        w = max(10, int(abs(east - west) * 20))
        arr = np.full((h, w), float(val), dtype=np.float32)
        if verbose: print(f"✅ NASA POWER: {val} mm")
        return arr
    except Exception as e:
        if verbose: print(f"  ⚠️ NASA POWER error: {e}")
        print("❌ NASA POWER failed")
    return None

# ─── 4. WorldClim (moyennes mensuelles) ────────────────────────────────────────
def essayer_worldclim(month, west, south, east, north, verbose=True):
    """
    Tente de télécharger WorldClim v2.1 précip mensuelles et extrait la bbox.
    """
    url = (f"https://geodata.ucdavis.edu/climate/worldclim/2.1/base/"
           f"wc2.1_2.5m_prec/wc2.1_2.5m_prec_{month:02d}.tif")
    if verbose: print(f"🌐 WorldClim try: {url}")
    try:
        tmp = os.path.join(tempfile.gettempdir(), f"wc{month:02d}.tif")
        r = requests.get(url, stream=True, timeout=20)
        if r.status_code != 200:
            if verbose: print(f"  → {r.status_code} {r.reason}")
            return None
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        with rasterio.open(tmp) as src:
            win = rasterio.windows.from_bounds(west, south, east, north, src.transform)
            arr = src.read(1, window=win)
        os.remove(tmp)
        # convertir en mm/jour
        days = [31,28,31,30,31,30,31,31,30,31,30,31][month-1]
        arr = arr.astype(np.float32) / days
        if verbose: print("✅ WorldClim loaded")
        return arr
    except Exception as e:
        if verbose: print(f"  ⚠️ WorldClim error: {e}")
        print("❌ WorldClim failed")
    return None

# ─── 5. Fonction principale ───────────────────────────────────────────────────
def obtenir_precipitation_historique(south, north, west, east,
                                     date="2000-01-01",
                                     target_shape=(512, 512),
                                     save_tif=True,
                                     is_normalized=True,
                                     verbose=True):
    """
    Récupère et renvoie un torch.Tensor de précipitations [0–1].
    Cascade : CHIRPS → NASA POWER → WorldClim → Synthétique.
    """
    # parse date
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        if verbose: print(f"⚠️ Date invalide: {date}")
        dt = datetime.now()
    Y, M, D = dt.year, dt.month, dt.day

    # cache
    cache_dir = os.path.join("data", "precip_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{Y}{M:02d}{D:02d}_{west:.2f}_{east:.2f}_{south:.2f}_{north:.2f}"
    cache_file = os.path.join(cache_dir, f"precip_{key}.npy")

    # essai des sources
    arr = None
    for fn in (essayer_chirps,essayer_nasa_power, lambda *a,**k: essayer_worldclim(M, west, south, east, north, verbose)):
        arr = fn(Y, M, D, west, south, east, north, verbose) if fn != essayer_worldclim else fn
        if arr is not None and arr.size>0:
            break
    if arr is None:
        if verbose: print("💡 Fallback synthétique")
        arr = generer_donnees_synthetiques(target_shape, mode="realiste", date_obj=dt)
    np.save(cache_file, arr)
    if verbose: print(f"💾 Saved cache: {cache_file}")

    # redimensionner
    if arr.shape != target_shape:
        if verbose: print(f"📏 Resize {arr.shape} → {target_shape}")
        arr = resize(arr, target_shape, mode='edge', preserve_range=True)

    # to tensor
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    # normalisation 0–1 par 50 mm/j
    if verbose: print(f"📊 Tensor shape chirps: {tensor.shape}, min: {tensor.min().item()}, max: {tensor.max().item()}")
    if is_normalized:
        tensor = torch.clamp(tensor / 50.0, 0.0, 1.0)
        if verbose: print("📊 Normalisation /50 mm, clamp 0–1")

    # sauvegarde GeoTIFF
    if save_tif:
        out = f"precip_{Y}{M:02d}{D:02d}.tif"
        with rasterio.open(
            out, 'w', driver='GTiff',
            height=target_shape[0], width=target_shape[1],
            count=1, dtype=rasterio.float32,
            crs='+proj=longlat +datum=WGS84',
            transform=rasterio.transform.from_bounds(west, south, east, north,
                                                     target_shape[1], target_shape[0])
        ) as dst:
            dst.write(arr.astype(np.float32), 1)
        if verbose: print(f"💾 GeoTIFF sauvegardé: {out}")

    return tensor
