import os
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models import Unet
from build_sequence_unet import coordonnees_vers_tenseur , obtenir_tenseur_precipitation , obtenir_precipitation_historique
from skimage.transform import resize

# ========= CONFIG =========

VV_PATH = "C:/Users/mokht/Desktop/PDS/flood_dataset/SEN12FLOOD/0167/S1A_IW_GRDH_1SDV_20190314T030815_20190314T030840_026328_02F196_7D7C_corrected_VV.tif"
VH_PATH = "C:/Users/mokht/Desktop/PDS/flood_dataset/SEN12FLOOD/0167/S1B_IW_GRDH_1SDV_20190314T160711_20190314T160736_015352_01CBEE_7F77_corrected_VV.tif"
CHECKPOINT = 'C:/Users/mokht/Desktop/PDS/Pytorch-UNet-Flood-Segmentation/models/model_20230411_161404_31'
CHECKPOINT2='C:/Users/mokht/Desktop/PDS/Pytorch-UNet-Flood-Segmentation/models/model_20230412_015857_48'



# Chemin pour l'image VH
SEN12FLOOD_VH_PATH = 'C:/Users/mokht/Desktop/PDS/flood_dataset/sen12flood/0294/S1A_IW_GRDH_1SDV_20190517T022015_20190517T022040_027261_0312E1_AD95_corrected_VH.tif'
SEN12FLOOD_VV_PATH = 'C:/Users/mokht/Desktop/PDS/flood_dataset/sen12flood/0294/S1A_IW_GRDH_1SDV_20190517T022015_20190517T022040_027261_0312E1_AD95_corrected_VV.tif'
OUTPUT_PATH = './mask_sar.tif'
THRESHOLD = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
API_PRECIPITATION="9133d48914f19b5f639a2aa60dac3041"
API_DEM="13ab4c018a6ab1b00e771d400d424345"
# ==========================


def explorer_pixels(image_vv, carte_proba, masque_binaire, threshold):
    """Visualisation interactive des valeurs de pixels"""
    plt.figure(figsize=(12, 8))
    
    # Afficher l'image VV normalisée comme fond
    plt.imshow(image_vv, cmap='gray')
    
    # Superposer la carte de probabilités en semi-transparent
    overlay = plt.imshow(carte_proba, alpha=0.7, cmap='plasma')
    plt.colorbar(overlay, label='Probabilité d\'eau')
    
    # Tracer le contour du masque binaire en blanc
    plt.contour(masque_binaire, levels=[0.5], colors=['white'], linewidths=1)
    
    plt.title(f"Carte de probabilités d'inondation (seuil={threshold})")
    
    # Fonction pour afficher les valeurs au clic
    def on_click(event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < carte_proba.shape[1] and 0 <= y < carte_proba.shape[0]:
                # Récupérer les valeurs au point cliqué
                prob = carte_proba[y, x]
                vv_val = image_vv[y, x]
                classe = "EAU" if prob > threshold else "NON-EAU"
                
                print(f"Position: ({x}, {y})")
                print(f"Probabilité d'eau: {prob:.4f} ({prob*100:.2f}%)")
                print(f"Valeur SAR (normalisée): {vv_val:.4f}")
                print(f"Classification: {classe}")
                print("-" * 30)
    
    # Connecter la fonction au clic de souris
    plt.connect('button_press_event', on_click)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyser_regions(carte_proba, threshold):
    """Analyse statistique des valeurs de probabilité par région"""
    # Créer un masque binaire
    masque = (carte_proba > threshold).astype(int)
    
    # Statistiques pour les pixels classés comme "eau"
    pixels_eau = carte_proba[masque == 1]
    stats_eau = {
        "Nombre de pixels": len(pixels_eau),
        "Probabilité minimale": pixels_eau.min() if len(pixels_eau) > 0 else "N/A",
        "Probabilité maximale": pixels_eau.max() if len(pixels_eau) > 0 else "N/A",
        "Probabilité moyenne": pixels_eau.mean() if len(pixels_eau) > 0 else "N/A"
    }
    
    # Statistiques pour les pixels classés comme "non-eau"
    pixels_non_eau = carte_proba[masque == 0]
    stats_non_eau = {
        "Nombre de pixels": len(pixels_non_eau),
        "Probabilité minimale": pixels_non_eau.min() if len(pixels_non_eau) > 0 else "N/A",
        "Probabilité maximale": pixels_non_eau.max() if len(pixels_non_eau) > 0 else "N/A",
        "Probabilité moyenne": pixels_non_eau.mean() if len(pixels_non_eau) > 0 else "N/A"
    }
    
    # Afficher les résultats
    print("\n=== ANALYSE DES RÉGIONS SEGMENTÉES ===")
    print(f"Seuil utilisé: {threshold}")
    
    print("\nRÉGION CLASSÉE COMME EAU:")
    for k, v in stats_eau.items():
        print(f"  {k}: {v if isinstance(v, str) else v:.4f}")
    
    print("\nRÉGION CLASSÉE COMME NON-EAU:")
    for k, v in stats_non_eau.items():
        print(f"  {k}: {v if isinstance(v, str) else v:.4f}")
    
    # Distribution des probabilités
    plt.figure(figsize=(10, 6))
    
    # Histogramme séparé pour eau et non-eau
    if len(pixels_eau) > 0:
        plt.hist(pixels_eau, bins=20, alpha=0.7, label='Eau', color='blue')
    if len(pixels_non_eau) > 0:
        plt.hist(pixels_non_eau, bins=20, alpha=0.7, label='Non-eau', color='orange')
    
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Seuil ({threshold})')
    plt.xlabel('Probabilité')
    plt.ylabel('Nombre de pixels')
    plt.title('Distribution des probabilités par classe')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def creer_tenseur_precipitation_extreme(target_shape, intensite=1.0, modele='uniforme'):
    """
    Crée un tenseur de précipitation synthétique avec des valeurs élevées
    
    Args:
        target_shape: Forme du tenseur (hauteur, largeur)
        intensite: Facteur d'intensité (0.0-1.0) où 1.0 = maximum
        modele: Type de distribution ('uniforme', 'gradient', 'zones')
        
    Returns:
        torch.Tensor: Tenseur de précipitations normalisé
    """
    h, w = target_shape
    
    if modele == 'uniforme':
        # Précipitation uniforme sur toute l'image
        precipitation = np.ones((h, w), dtype=np.float32) * intensite
    
    elif modele == 'gradient':
        # Gradient de précipitation (plus fort d'un côté)
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        precipitation = (X + Y) / 2 * intensite
    
    elif modele == 'zones':
        # Zones de précipitation avec différentes intensités
        precipitation = np.zeros((h, w), dtype=np.float32)
        
        # Créer 3-5 zones de précipitation intense
        for _ in range(4):
            # Position aléatoire du centre
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            # Taille du cercle (rayon)
            radius = np.random.randint(h//8, h//3)
            # Intensité de cette zone
            intensity = np.random.uniform(0.7, 1.0) * intensite
            
            # Créer un masque circulaire
            y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
            mask = x*x + y*y <= radius*radius
            
            # Appliquer le masque
            precipitation[mask] = max(precipitation[mask].max(), intensity)
        
        # Ajouter un léger bruit de fond
        precipitation += np.random.uniform(0.0, 0.1, size=(h, w)) * intensite
        precipitation = np.clip(precipitation, 0.0, 1.0)
    
    else:
        raise ValueError(f"Modèle de précipitation '{modele}' non reconnu")
    
    # Normaliser et convertir en tenseur PyTorch
    tensor = torch.tensor(precipitation).float().unsqueeze(0)
    print(f"Tenseur de précipitation créé: forme={tensor.shape}, min={tensor.min().item():.2f}, max={tensor.max().item():.2f}")
    
    return tensor


def segmenter_image_sar(vv_path, vh_path, modele_path, output_path="./data_dem", threshold=0.5 , date_precipitation="2023-10-01"):
    """
    Effectue la segmentation d'une image SAR avec un modèle UNet pré-entraîné
    et utilise des données réelles de DEM et précipitation
    """
    print(f"🔄 Segmentation de l'image SAR...")

    
    # 1. Charger les images SAR et extraire les coordonnées géographiques
    with rasterio.open(vv_path) as src:
        vv = src.read(1).astype(np.float32)
        profile = src.profile
        # Extraire les coordonnées géographiques (bounds) de l'image
        bounds = src.bounds
        print(f"les bounds de l'image: {bounds}")
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
        print(f"Coordonnées de l'image: {south:.6f}°S, {north:.6f}°N, {west:.6f}°E, {east:.6f}°E")
    
    with rasterio.open(vh_path) as src:
        vh = src.read(1).astype(np.float32)
    
    # 2. Prétraitement des images SAR
    vv = np.clip(vv, -50, 25)
    vh = np.clip(vh, -50, 25)
    vv_norm = (vv - (-50)) / (25 - (-50))
    vh_norm = (vh - (-50)) / (25 - (-50))
    
    h, w = vv.shape
    target_shape = (h, w)
    
    # 3. Récupérer les données DEM avec coordonnees_vers_tenseur
    try:
        print("🏔️ Récupération des données d'élévation (DEM)...")
        dem_tensor = coordonnees_vers_tenseur(
            south=south, 
            north=north, 
            west=west, 
            east=east, 
            dem_type="SRTM", 
            api_key=API_DEM, 
            target_shape=target_shape,
            clip_min=0.0, 
            clip_max=2000.0,
            output_dir="./data/dem"
        )
        
        # Le tenseur est déjà normalisé par la fonction coordonnees_vers_tenseur
        ele_norm = dem_tensor.squeeze().numpy()
        print(f"DEM obtenu avec succès: shape={ele_norm.shape}, min={ele_norm.min():.4f}, max={ele_norm.max():.4f}")
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération du DEM: {e}")
        print("Utilisation d'un DEM substitut (zéros)")
        dem_substitute = np.zeros((h, w), dtype=np.float32)
        ele_norm = dem_substitute / 2000.0
    
    # 4. Récupérer les données de précipitation avec obtenir_tenseur_precipitation
    try:
        # Vous devez avoir une clé API OpenWeatherMap pour cette fonction
        print("☔ Récupération des données de précipitation...")
        
        precip_tensor = obtenir_precipitation_historique(
                south=south,
                north=north,
                west=west,
                east=east,
                date=date_precipitation,
                target_shape=target_shape,
                save_tif=True,
                is_normalized=True
        )
        
        # La fonction retourne déjà un tenseur normalisé
        wat_norm = precip_tensor.squeeze().numpy()
        print(f"Données de précipitation obtenues avec succès: shape={wat_norm.shape}, min={wat_norm.min():.4f}, max={wat_norm.max():.4f}")
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération des précipitations: {e}")
        print("Utilisation d'un substitut de précipitation (zéros)")
        pwat_substitute = np.zeros((h, w), dtype=np.float32)
        wat_norm = pwat_substitute / 100.0

# Calculer la taille cible (multiple de 16 le plus proche)
    h_target = ((h + 15) // 16) * 16
    w_target = ((w + 15) // 16) * 16

    print(f"Redimensionnement: {h}×{w} → {h_target}×{w_target}")

    # Redimensionner au lieu de padder
    # Créer des précipitations élevées comme tableau NumPy
    
    vv_resized = resize(vv_norm, (h_target, w_target), mode='edge', preserve_range=True)
    vh_resized = resize(vh_norm, (h_target, w_target), mode='edge', preserve_range=True)
    dem_resized = resize(ele_norm, (h_target, w_target), mode='edge', preserve_range=True)
    pwat_resized = resize(wat_norm, (h_target, w_target), mode='edge', preserve_range=True)

    # Empiler les canaux
    img = np.stack([vv_resized, vh_resized, vv_resized, vh_resized], axis=0)
    tensor = torch.from_numpy(img).unsqueeze(0).float()
    
    print(f"Dimensions de l'image après padding: {tensor.shape}")
    
    # Le reste de la fonction reste inchangé
    # 6. Charger le modèle UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Utilisation du device: {device}")
    
    model = Unet(in_channels=4, out_channels=1)
    model.to(device)
    
    # Charger les poids
    state = torch.load(modele_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    # 7. Faire l'inférence avec gestion des dimensions
    tensor = tensor.to(device)
    with torch.no_grad():
        print("🧠 Inférence en cours...")
        prediction = model(tensor).squeeze().cpu().numpy()

    prediction = prediction[:h, :w]
    
    # 8. Binariser avec le seuil
    mask_binary = (prediction > threshold).astype(np.uint8)

    # Ajouter après création du masque dans segmenter_image_sar:
    print(f"Valeurs uniques dans le masque: {np.unique(mask_binary)}")
    print(f"Pourcentage de pixels inondés: {100 * np.sum(mask_binary == 1) / mask_binary.size:.2f}%")

    print(f"Dimensions du masque binaire: {mask_binary.shape}")
    # 9. Sauvegarder le masque
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask_binary, 1)

    print(f"VV stats: min={vv.min():.2f}, max={vv.max():.2f}, mean={vv.mean():.2f}")
    print(f"VH stats: min={vh.min():.2f}, max={vh.max():.2f}, mean={vh.mean():.2f}")
    print(f"DEM stats: min={ele_norm.min():.4f}, max={ele_norm.max():.4f}")
    print(f"Precipitation stats: min={wat_norm.min():.4f}, max={wat_norm.max():.4f}")
    print(f"💾 Segmentation terminée - Masque sauvegardé: {output_path}")
    # Add inside segmenter_image_sar after prediction
    print(f"Prediction stats: min={prediction.min():.6f}, max={prediction.max():.6f}, mean={prediction.mean():.6f}")
    plt.hist(prediction.flatten(), bins=50)
    plt.title("Distribution of flood probabilities")
    plt.savefig("probabilities_histogram.png")
    plt.close()
    return output_path, prediction

def custom_forward(model, x):
    """Forward pass personnalisé pour gérer les erreurs de dimensions"""
    from torch import cat, sigmoid
    
    # Partie encodeur
    enc_conv_1 = model.enc_conv_01(x)
    enc_conv_2 = model.enc_conv_02(model.down_sample_01(enc_conv_1))
    enc_conv_3 = model.enc_conv_03(model.down_sample_02(enc_conv_2))
    enc_conv_4 = model.enc_conv_04(model.down_sample_03(enc_conv_3))
    base_block = model.base(model.down_sample_04(enc_conv_4))
    
    # Partie décodeur avec ajustement des dimensions
    dec_conv_4 = model.up_sample_04(base_block)
    if dec_conv_4.shape[2:] != enc_conv_4.shape[2:]:
        dec_conv_4 = F.interpolate(dec_conv_4, size=enc_conv_4.shape[2:], mode='bilinear', align_corners=False)
    dec_conv_4 = cat((enc_conv_4, dec_conv_4), dim=1)
    dec_conv_4 = model.dec_conv_04(dec_conv_4)
    
    dec_conv_3 = model.up_sample_03(dec_conv_4)
    if dec_conv_3.shape[2:] != enc_conv_3.shape[2:]:
        dec_conv_3 = F.interpolate(dec_conv_3, size=enc_conv_3.shape[2:], mode='bilinear', align_corners=False)
    dec_conv_3 = cat((enc_conv_3, dec_conv_3), dim=1)
    dec_conv_3 = model.dec_conv_03(dec_conv_3)
    
    dec_conv_2 = model.up_sample_02(dec_conv_3)
    if dec_conv_2.shape[2:] != enc_conv_2.shape[2:]:
        dec_conv_2 = F.interpolate(dec_conv_2, size=enc_conv_2.shape[2:], mode='bilinear', align_corners=False)
    dec_conv_2 = cat((enc_conv_2, dec_conv_2), dim=1)
    dec_conv_2 = model.dec_conv_02(dec_conv_2)
    
    dec_conv_1 = model.up_sample_01(dec_conv_2)
    if dec_conv_1.shape[2:] != enc_conv_1.shape[2:]:
        dec_conv_1 = F.interpolate(dec_conv_1, size=enc_conv_1.shape[2:], mode='bilinear', align_corners=False)
    dec_conv_1 = cat((enc_conv_1, dec_conv_1), dim=1)
    dec_conv_1 = model.dec_conv_01(dec_conv_1)
    
    return sigmoid(model.final_conv(dec_conv_1))

def visualiser_tif(tif_path, save_png=False, clip_percentile=True):
    """
    Visualise une image TIFF avec rendu amélioré
    
    Args:
        tif_path: Chemin vers l'image TIFF à visualiser
        save_png: Booléen, sauvegarder l'image en PNG ou non
        clip_percentile: Si True, utilise une méthode robuste pour l'étirement de contraste
    """
    print(f"🔍 Visualisation du fichier: {tif_path}")
    
    # Charger l'image TIFF
    with rasterio.open(tif_path) as src:
        band = src.read(1)
        print(f"Dimensions: {band.shape}, Type: {band.dtype}, Min: {band.min()}, Max: {band.max()}")
    
    # Traitement spécifique pour les images SAR
    if band.min() < -10 and ("VV" in tif_path or "VH" in tif_path):
        print("Image SAR détectée - Amélioration du contraste")
        
        # La "plage" est l'intervalle de valeurs utilisé pour l'affichage
        # On utilise les percentiles pour éliminer les valeurs extrêmes
        if clip_percentile:
            vmin, vmax = np.percentile(band[band > -100], [2, 98])
            print(f"Plage utilisée (2-98%): [{vmin:.2f}, {vmax:.2f}] dB")
        else:
            vmin, vmax = -25, 0
            print(f"Plage standard SAR: [{vmin:.2f}, {vmax:.2f}] dB")
        
        # Amélioration du contraste en limitant les valeurs et normalisant
        band_display = np.clip(band, vmin, vmax)
        band_display = (band_display - vmin) / (vmax - vmin)
        
        plt.figure(figsize=(12, 10))
        img = plt.imshow(band_display, cmap='viridis')
        plt.colorbar(img, label='Intensité SAR (dB normalisée)')
        plt.title(f"Image SAR: {os.path.basename(tif_path)}\nPlage: [{vmin:.2f}, {vmax:.2f}] dB")
        
    # Pour les masques binaires (0-1)
    elif np.array_equal(np.unique(band), [0, 1]) or np.array_equal(np.unique(band), [0, 1, 255]):
        print("Masque binaire détecté")
        unique_values = np.unique(band)
        percent_one = 100 * np.sum(band > 0) / band.size
        
        plt.figure(figsize=(12, 10))
        img = plt.imshow(band, cmap='Blues')
        plt.colorbar(img, label='Classe')
        plt.title(f"Masque de segmentation: {os.path.basename(tif_path)}\n"
                 f"Valeurs: {unique_values}, {percent_one:.2f}% de pixels positifs")
        
    # Autres images (DEM, précipitations, etc.)
    else:
        print("Autre type d'image détecté")
        if clip_percentile:
            # La "plage" pour les autres images = intervalle de valeurs pour l'affichage
            vmin, vmax = np.percentile(band[np.isfinite(band)], [2, 98])
            print(f"Plage utilisée (2-98%): [{vmin:.2f}, {vmax:.2f}]")
        else:
            vmin, vmax = band.min(), band.max()
            print(f"Plage complète: [{vmin:.2f}, {vmax:.2f}]")
        
        band_display = np.clip(band, vmin, vmax)
        band_display = (band_display - vmin) / (vmax - vmin)
        
        plt.figure(figsize=(12, 10))
        img = plt.imshow(band_display, cmap='viridis')
        plt.colorbar(img, label='Valeur normalisée')
        plt.title(f"Image: {os.path.basename(tif_path)}\nPlage: [{vmin:.2f}, {vmax:.2f}]")
    
    plt.axis('off')
    
    # Sauvegarder en PNG si demandé
    if save_png:
        png_path = tif_path.replace('.tif', '_visualisation.png')
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"📷 Image sauvegardée: {png_path}")
    
    plt.show()
    return band

def visualiser_resultat_segmentation(vv_path, masque_path, proba_map=None, output_png=None, threshold=0.5):
    """
    Visualise l'image SAR originale et sa segmentation (avant/après)
    
    Args:
        vv_path: Chemin de l'image SAR VV
        masque_path: Chemin du masque de segmentation
        proba_map: Carte de probabilités brute (sortie du modèle avant seuil)
        output_png: Chemin de sortie pour l'image
        threshold: Seuil utilisé pour la binarisation (pour affichage)
    """
    print("📊 Visualisation des résultats...")
    
    # 1. Charger les images et vérifier les dimensions
    with rasterio.open(vv_path) as src:
        vv = src.read(1).astype(np.float32)
        print(f"Image VV chargée: forme={vv.shape}")
    
    with rasterio.open(masque_path) as src:
        mask = src.read(1)
        print(f"Masque chargé: forme={mask.shape}")
        # Vérifier le contenu du masque
        unique_values = np.unique(mask)
        print(f"Valeurs uniques dans le masque: {unique_values}")
    
    # 2. Vérifier la correspondance des dimensions
    if vv.shape != mask.shape:
        print(f"⚠️ ATTENTION: Les dimensions ne correspondent pas! VV:{vv.shape}, masque:{mask.shape}")
    
    # 3. Normalisation du masque si nécessaire (pour le cas où il y a des valeurs 255)
    if 255 in unique_values:
        print("Normalisation du masque (255 → 1)")
        mask = (mask > 0).astype(np.uint8)
    
    # 4. Prétraitement pour visualisation (améliorer le contraste)
    p_min, p_max = np.percentile(vv[vv > -100], [2, 98])
    print(f"Plage de normalisation SAR: [{p_min:.2f}, {p_max:.2f}]")
    vv_display = np.clip(vv, p_min, p_max)
    vv_norm = (vv_display - vv_display.min()) / (vv_display.max() - vv_display.min())
    
    # 5. Calcul du pourcentage de pixels inondés
    pourcentage_inondation = 100 * np.sum(mask == 1) / mask.size
    print(f"Pourcentage de pixels inondés: {pourcentage_inondation:.2f}%")
    
    # 6. Configuration de la figure
    n_plots = 3 if proba_map is not None else 2
    plt.figure(figsize=(6*n_plots, 6))
    
    # 7. Image originale (première sous-figure)
    plt.subplot(1, n_plots, 1)
    plt.imshow(vv_norm, cmap='viridis')
    plt.colorbar(label='Intensité SAR normalisée')
    plt.title("Image SAR originale (VV)", fontsize=14)
    plt.axis('off')
    
    # 8. Carte de probabilités (sous-figure du milieu, si disponible)
    if proba_map is not None:
        plt.subplot(1, n_plots, 2)
        plt.imshow(proba_map, cmap='plasma')
        plt.colorbar(label='Probabilité d\'eau')
        
        # Tracer le contour correspondant au seuil
        plt.contour(proba_map > threshold, levels=[0.5], colors=['white'], linewidths=1)
        
        plt.title(f"Probabilités d'inondation\nSeuil: {threshold}", fontsize=14)
        plt.axis('off')
        
        # Petit histogramme inséré pour montrer la distribution des probabilités
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(plt.gca(), width="40%", height="30%", loc='lower right')
        axins.hist(proba_map.flatten(), bins=20, color='skyblue', alpha=0.7)
        axins.axvline(x=threshold, color='red', linestyle='--')
        axins.set_xticks([0, threshold, 1])
        axins.set_yticks([])
        axins.set_title('Distribution')
    
    # 9. Résultat de segmentation (dernière sous-figure)
    plt.subplot(1, n_plots, n_plots)
    
    # 9a. Afficher l'image originale en niveau de gris
    plt.imshow(vv_norm, cmap='gray')
    
    # 9b. Superposer le masque en bleu semi-transparent
    if np.sum(mask) > 0:  # Vérifier que le masque contient des pixels positifs
        masked = np.ma.masked_where(mask == 0, mask)
        plt.imshow(masked, cmap='cool', alpha=0.6)
        txt_color = 'black'
    else:
        txt_color = 'red'  # Rouge si aucun pixel inondé
        
    plt.title(f"Segmentation: {pourcentage_inondation:.2f}% inondé", 
             fontsize=14, color=txt_color)
    plt.axis('off')
    
    # 10. Finaliser la mise en page
    plt.tight_layout()
    
    # 11. Sauvegarder l'image
    if output_png is None:
        output_png = os.path.join(os.path.dirname(vv_path), 'resultat_visualisation.png')
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✅ Visualisation sauvegardée: {output_png}")
    
    plt.show()
    return output_png

if __name__ == "__main__":
    # 1. Segmenter l'image
    threshold = 0.0075 # Utiliser le premier seuil par défaut
    print("\n" + "="*50)
    print("ÉTAPE 1: SEGMENTATION DE L'IMAGE SAR")
    print("="*50)
    print(f"🔍 Utilisation du seuil: {threshold}")
    masque_path, proba_map = segmenter_image_sar(
        vv_path=VV_PATH,
        vh_path=VH_PATH,
        modele_path=CHECKPOINT2,
        output_path=OUTPUT_PATH,
        threshold=threshold,
        date_precipitation="2019-03-14"
    )
    
    # 2. Visualiser l'image VV
    print("\n" + "="*50)
    print("ÉTAPE 2: VISUALISATION DE L'IMAGE VV")
    print("="*50)
    visualiser_tif(VV_PATH, save_png=True)
    
    # 3. Visualiser le masque
    print("\n" + "="*50)
    print("ÉTAPE 3: VISUALISATION DU MASQUE")
    print("="*50)
    visualiser_tif(masque_path, save_png=True)
    
    # 4. Visualiser le résultat avant/après
    print("\n" + "="*50)
    print("ÉTAPE 4: VISUALISATION DES RÉSULTATS")
    print("="*50)
    visualiser_resultat_segmentation(VV_PATH, masque_path, proba_map, threshold=threshold)

    print("\n✨ Traitement terminé! ✨")

