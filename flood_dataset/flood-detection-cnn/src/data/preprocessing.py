import numpy as np
import cv2
import rasterio
from scipy import ndimage
import random
import torch
from skimage import exposure, restoration
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm


"""remarque sur indices : 

L’eau : Elle absorbe fortement dans le NIR (donc réflectance très faible, par exemple 0,05) mais réfléchit modérément dans le vert (disons 0,3).

"""

def normalize_image(image, mean=0.5, std=0.5):
    """
    Normalize the image using the given mean and standard deviation.
    """
    return (image - mean) / std

def resize_image(image, size=(256, 256)):
    """
    Resize the image to the specified size.
    """
    return cv2.resize(image, size)

def preprocess_image(image, img_size=(256, 256), scaling_value=1.0):
    """
    Preprocess the image by scaling, resizing, and normalizing.
    """
    # Apply scaling
    image = image / scaling_value
    
    # Clip values to avoid extreme values
    image = np.clip(image, 0, 5)
    
    # Resize the image
    image = resize_image(image, size=img_size)
    
    # Normalize the image
    image = normalize_image(image)
    
    return image

def preprocess_images(image_paths, img_size=(256, 256), scaling_value=1.0):
    """
    Preprocess a list of images given their file paths.
    """
    processed_images = []
    for path in image_paths:
        with rasterio.open(path) as src:
            band = src.read(1)
            processed_image = preprocess_image(band, img_size=img_size, scaling_value=scaling_value)
            processed_images.append(processed_image)
    
    return np.array(processed_images)

# Fonctions de prétraitement avancées

def apply_log_transform(image : np.ndarray):
    """
    Applique une transformation logarithmique à l'image SAR
    
    Args:
        image (np.ndarray): Image à transformer
        epsilon (float): Petite valeur pour éviter log(0)
        
    Returns:
        np.ndarray: Image transformée en logarithme
    """

    epsilon = 1e-10
    image_positive = np.maximum(image, epsilon)
    
    log_image = np.log(image_positive)
    
    return log_image

def apply_inverse_log_transform(log_image : np.ndarray ) -> np.ndarray :
    """
    Applique la transformation inverse du logarithme
    
    Args:
        log_image (np.ndarray): Image en logarithme
        
    Returns:
        np.ndarray: Image transformée en exponentielle
    """
    return np.exp(log_image)

def apply_residual_despeckling(image : np.ndarray , log_transform : bool = True, filter_type='lee', window_size=3) -> np.ndarray :
    """
    Applique une approche résiduelle pour le despeckling

    Args:
        image (np.ndarray): Image SAR originale
        log_transform (bool): Appliquer transformation logarithmique
        filter_type (str): Type de filtre pour l'estimation du bruit
        window_size (int): Taille de la fenêtre pour le filtre
        
    Returns:
        np.ndarray: Image avec bruit speckle réduit
    """
    #Appliquer la transformation logarithmique si demandé
    if log_transform:
        img_log = apply_log_transform(image)
    else:
        img_log = image.copy()
    
    # 2. Appliquer un filtre pour estimer l'image sans bruit
    if filter_type == 'lee':
        # Implémentation du filtre de Lee pour estimer l'image sans bruit
        mean = ndimage.uniform_filter(img_log, size=window_size)
        variance = ndimage.uniform_filter(img_log**2, size=window_size) - mean**2
        # Ajouter un petit epsilon pour éviter la division par zéro
        epsilon = 1e-10
        img_variance = np.mean(variance)
        filtered_log = mean + (variance / (variance + img_variance + epsilon)) * (img_log - mean)
    else:
        # Utiliser un filtre gaussien par défaut
        filtered_log = ndimage.gaussian_filter(img_log, sigma=window_size/3)
    
    #3 calcul bruit résiduel 
    noise_residual = img_log - filtered_log
    
    # 4. Soustraire une version réduite du bruit résiduel
    # (on garde une partie du bruit pour éviter le sur-lissage)
    denoised_log = img_log - 0.9 * noise_residual
    
    #5Appliquer la transformation inverse si nécessaire
    if log_transform:
        denoised_image = apply_inverse_log_transform(denoised_log)
    else:
        denoised_image = denoised_log
    
    return denoised_image

def apply_speckle_filter(image, filter_type='lee', window_size=5):
    """
    Applique un filtre de réduction du bruit speckle à l'image SAR
    
    Args:
        image (np.ndarray): Image à filtrer
        filter_type (str): Type de filtre ('lee', 'kuan', 'frost', 'median')
        window_size (int): Taille de la fenêtre de filtrage
        
    Returns:
        np.ndarray: Image filtrée
    """
    if filter_type == 'median':
        return ndimage.median_filter(image, size=window_size)
    elif filter_type == 'lee':
        # Implémentation simplifié du filtre de Lee
        mean = ndimage.uniform_filter(image, size=window_size)
        variance = ndimage.uniform_filter(image**2, size=window_size) - mean**2
        # Ajouter un petit epsilon pour éviter la division par zéro
        epsilon = 1e-10
        img_variance = np.mean(variance)
        filtered = mean + (variance / (variance + img_variance + epsilon)) * (image - mean)
        return filtered
    elif filter_type == 'bilateral':
        # Normaliser l'image pour le filtre bilatéral
        norm_img = (image - np.min(image)) / (np.max(image) - np.min(image))
        norm_img = (norm_img * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(norm_img, window_size, 75, 75)
        # Revenir à l'échelle originale
        filtered = filtered.astype(np.float32) / 255
        filtered = filtered * (np.max(image) - np.min(image)) + np.min(image)
        return filtered
    else:
        # Par défaut utiliser un filtre gaussien
        return ndimage.gaussian_filter(image, sigma=window_size/3)

def enhance_contrast(image, method='histogram_equalization'):
    """
    Améliore le contraste de l'image SAR
    
    Args:
        image (np.ndarray): Image à améliorer
        method (str): Méthode d'amélioration ('histogram_equalization', 'clahe', 'adaptive_gamma')
        
    Returns:
        np.ndarray: Image avec contraste amélioré
    """
    # Normaliser l'image pour le traitement
    p2, p98 = np.percentile(image, (2, 98))
    img_norm = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    if method == 'histogram_equalization':
        return exposure.equalize_hist(img_norm)
    elif method == 'clahe':
        return exposure.equalize_adapthist(img_norm, clip_limit=0.03)
    elif method == 'adaptive_gamma':
        return exposure.adjust_gamma(img_norm, gamma=0.8)
    else:
        return img_norm

# Fonctions de visualisation

def visualize_pixel_matrix(image, title="Matrice de pixels", roi_size=10):
    """
    Visualise une région d'intérêt de la matrice de pixels de l'image
    
    Args:
        image (np.ndarray): Image à analyser
        title (str): Titre de la visualisation
        roi_size (int): Taille de la région d'intérêt
    """
    # Sélectionner une petite région au centre de l'image
    h, w = image.shape
    center_h, center_w = h // 2, w // 2
    start_h, start_w = center_h - roi_size // 2, center_w - roi_size // 2
    
    # Extraire la sous-région
    roi = image[start_h:start_h+roi_size, start_w:start_w+roi_size]
    
    # Créer une figure pour afficher la matrice et ses valeurs
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Afficher la matrice comme une image
    im = ax.imshow(roi, cmap='viridis')
    plt.colorbar(im, ax=ax, label="Valeur du pixel")
    
    # Ajouter les valeurs dans les cellules
    for i in range(roi_size):
        for j in range(roi_size):
            value = roi[i, j]
            text_color = 'white' if value > np.mean(roi) else 'black'
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)
    
    ax.set_title(f"{title} - Matrice {roi_size}x{roi_size} au centre")
    plt.tight_layout()
    plt.show()

def visualize_log_transform_effect(image_path, scaling_value=50.0):
    """
    Visualise l'effet de la transformation logarithmique sur une image SAR
    
    Args:
        image_path (str): Chemin de l'image SAR
        scaling_value (float): Valeur d'échelle pour l'image
    """
    # Charger l'image
    with rasterio.open(image_path) as src:
        raw_image = src.read(1)
    
    # Appliquer la mise à l'échelle et le clipping
    scaled_image = raw_image / scaling_value
    clipped_image = np.clip(scaled_image, 0, 5)
    
    # Appliquer la transformation logarithmique
    log_image = apply_log_transform(clipped_image)
    
    # Créer la figure avec 3 colonnes
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
    
    # Première ligne: images
    ax1 = plt.subplot(gs[0, 0])
    im1 = ax1.imshow(clipped_image, cmap='viridis')
    ax1.set_title("Image originale (après mise à l'échelle)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = plt.subplot(gs[0, 1])
    im2 = ax2.imshow(log_image, cmap='viridis')
    ax2.set_title("Image après transformation logarithmique")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Appliquer le despeckling avec approche résiduelle
    despecked_image = apply_residual_despeckling(clipped_image)
    
    ax3 = plt.subplot(gs[0, 2])
    im3 = ax3.imshow(despecked_image, cmap='viridis')
    ax3.set_title("Image après despeckling résiduel")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Deuxième ligne: histogrammes
    ax4 = plt.subplot(gs[1, 0])
    ax4.hist(clipped_image.flatten(), bins=100, alpha=0.7)
    ax4.set_title("Histogramme de l'image originale")
    ax4.set_xlabel("Valeur de pixel")
    ax4.set_ylabel("Fréquence")
    
    ax5 = plt.subplot(gs[1, 1])
    ax5.hist(log_image.flatten(), bins=100, alpha=0.7)
    ax5.set_title("Histogramme après transformation log")
    ax5.set_xlabel("Valeur de pixel (log)")
    
    ax6 = plt.subplot(gs[1, 2])
    ax6.hist(despecked_image.flatten(), bins=100, alpha=0.7)
    ax6.set_title("Histogramme après despeckling")
    ax6.set_xlabel("Valeur de pixel")
    
    # Troisième ligne: visualisation des valeurs
    # Sélectionner une petite région d'intérêt pour montrer les valeurs des pixels
    h, w = clipped_image.shape
    roi_h, roi_w = h // 2, w // 2
    roi_size = 5
    roi_orig = clipped_image[roi_h:roi_h+roi_size, roi_w:roi_w+roi_size]
    roi_log = log_image[roi_h:roi_h+roi_size, roi_w:roi_w+roi_size]
    roi_desp = despecked_image[roi_h:roi_h+roi_size, roi_w:roi_w+roi_size]
    
    # Afficher les valeurs dans une petite matrice
    ax7 = plt.subplot(gs[2, 0])
    im7 = ax7.imshow(roi_orig, cmap='viridis')
    ax7.set_title(f"Matrice {roi_size}x{roi_size} au centre (orig)")
    for i in range(roi_size):
        for j in range(roi_size):
            ax7.text(j, i, f"{roi_orig[i, j]:.2f}", ha="center", va="center", 
                    color="white" if roi_orig[i, j] > np.mean(roi_orig) else "black")
    
    ax8 = plt.subplot(gs[2, 1])
    im8 = ax8.imshow(roi_log, cmap='viridis')
    ax8.set_title(f"Matrice {roi_size}x{roi_size} (log)")
    for i in range(roi_size):
        for j in range(roi_size):
            ax8.text(j, i, f"{roi_log[i, j]:.2f}", ha="center", va="center",
                   color="white" if roi_log[i, j] > np.mean(roi_log) else "black")
    
    ax9 = plt.subplot(gs[2, 2])
    im9 = ax9.imshow(roi_desp, cmap='viridis')
    ax9.set_title(f"Matrice {roi_size}x{roi_size} (despecked)")
    for i in range(roi_size):
        for j in range(roi_size):
            ax9.text(j, i, f"{roi_desp[i, j]:.2f}", ha="center", va="center",
                   color="white" if roi_desp[i, j] > np.mean(roi_desp) else "black")
    
    plt.tight_layout()
    plt.show()
    
    # Afficher également une analyse de l'effet du log sur les variations relatives
    print("\nAnalyse de l'effet de la transformation logarithmique:")
    
    # Calculer les rapports entre pixels adjacents
    h, w = clipped_image.shape
    ratios_orig = []
    ratios_log = []
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            central = clipped_image[i, j]
            neighbors = [
                clipped_image[i-1, j], 
                clipped_image[i+1, j],
                clipped_image[i, j-1],
                clipped_image[i, j+1]
            ]
            
            for neighbor in neighbors:
                if central > 0 and neighbor > 0:
                    ratios_orig.append(max(central, neighbor) / min(central, neighbor))
            
            # Même chose pour l'image log
            central_log = log_image[i, j]
            neighbors_log = [
                log_image[i-1, j], 
                log_image[i+1, j],
                log_image[i, j-1],
                log_image[i, j+1]
            ]
            
            for neighbor_log in neighbors_log:
                ratios_log.append(abs(central_log - neighbor_log))
    
    print(f"Rapport moyen entre pixels adjacents (original): {np.mean(ratios_orig):.2f}")
    print(f"Différence moyenne entre pixels adjacents (log): {np.mean(ratios_log):.2f}")
    print(f"Max rapport (original): {np.max(ratios_orig):.2f}")
    print(f"Max différence (log): {np.max(ratios_log):.2f}")
    
    # Afficher la distribution des rapports
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(ratios_orig, bins=50, alpha=0.7, range=(1, 10))
    plt.title("Distribution des rapports entre pixels adjacents\n(image originale)")
    plt.xlabel("Rapport (max/min)")
    plt.ylabel("Fréquence")
    
    plt.subplot(1, 2, 2)
    plt.hist(ratios_log, bins=50, alpha=0.7)
    plt.title("Distribution des différences entre pixels adjacents\n(image log)")
    plt.xlabel("Différence absolue")
    plt.ylabel("Fréquence")
    
    plt.tight_layout()
    plt.show()

def compare_advanced_preprocessing_methods(image_path, scaling_value=50.0):
    """
    Compare différentes méthodes de prétraitement avancées sur une image SAR
    
    Args:
        image_path (str): Chemin de l'image SAR
        scaling_value (float): Valeur d'échelle pour l'image
    """
    # Charger l'image
    with rasterio.open(image_path) as src:
        raw_image = src.read(1)
    
    # Appliquer la mise à l'échelle et le clipping
    scaled_image = raw_image / scaling_value
    clipped_image = np.clip(scaled_image, 0, 5)
    
    # 1. Transformation logarithmique
    log_image = apply_log_transform(clipped_image)
    
    # 2. Filtre de Lee standard
    lee_filtered = apply_speckle_filter(clipped_image, filter_type='lee')
    
    # 3. Filtre de Lee en logarithme puis exponentielle
    log_lee_filtered = apply_speckle_filter(log_image, filter_type='lee')
    log_lee_exp = apply_inverse_log_transform(log_lee_filtered)
    
    # 4. Approche résiduelle pour le despeckling
    residual_despecked = apply_residual_despeckling(clipped_image)
    
    # 5. Amélioration du contraste par CLAHE
    clahe_enhanced = enhance_contrast(lee_filtered, method='clahe')
    
    # 6. Pipeline complet: log -> residual despeckling -> CLAHE
    full_pipeline = enhance_contrast(residual_despecked, method='clahe')
    
    # Créer une figure pour afficher toutes les images
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Première ligne
    axes[0, 0].imshow(clipped_image, cmap='viridis')
    axes[0, 0].set_title("Image originale (mise à l'échelle)")
    
    axes[0, 1].imshow(log_image, cmap='viridis')
    axes[0, 1].set_title("Transformation logarithmique")
    
    axes[0, 2].imshow(lee_filtered, cmap='viridis')
    axes[0, 2].set_title("Filtre de Lee standard")
    
    # Deuxième ligne
    axes[1, 0].imshow(log_lee_exp, cmap='viridis')
    axes[1, 0].set_title("Filtre de Lee en log-domaine")
    
    axes[1, 1].imshow(residual_despecked, cmap='viridis')
    axes[1, 1].set_title("Despeckling résiduel")
    
    axes[1, 2].imshow(full_pipeline, cmap='viridis')
    axes[1, 2].set_title("Pipeline complet")
    
    # Désactiver les axes
    for ax in axes.flatten():
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Afficher également une analyse quantitative
    print("\nAnalyse quantitative des méthodes de prétraitement:")
    
    # Calculer les statistiques pour chaque image
    images = {
        "Original": clipped_image,
        "Log transform": log_image,
        "Lee filter": lee_filtered,
        "Log-Lee filter": log_lee_exp,
        "Residual despeckling": residual_despecked,
        "Full pipeline": full_pipeline
    }
    
    # Tableau des statistiques
    stats = {}
    for name, img in images.items():
        stats[name] = {
            "Min": np.min(img),
            "Max": np.max(img),
            "Mean": np.mean(img),
            "Std": np.std(img),
            "Dynamic range": np.max(img) - np.min(img)
        }
    
    # Afficher les statistiques
    for name, metrics in stats.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

def advanced_sar_preprocessing(image, advanced_options=None):
    """
    Pipeline de prétraitement avancé pour les images SAR
    
    Args:
        image (np.ndarray): Image SAR brute
        advanced_options (dict): Options de prétraitement
            - speckle_filter (bool): Appliquer un filtre anti-speckle
            - filter_type (str): Type de filtre ('lee', 'median', etc.)
            - enhance_contrast (bool): Améliorer le contraste
            - contrast_method (str): Méthode d'amélioration de contraste
            - preserve_ratio (bool): Préserver le ratio d'aspect
            - log_transform (bool): Appliquer transformation logarithmique
            - residual_despeckling (bool): Utiliser l'approche résiduelle
            
    Returns:
        np.ndarray: Image prétraitée
    """
    if advanced_options is None:
        advanced_options = {}
    
    # Valeurs par défaut
    options = {
        'speckle_filter': True,
        'filter_type': 'lee',
        'enhance_contrast': True,
        'contrast_method': 'clahe',
        'preserve_ratio': True,
        'target_size': (256, 256),
        'scaling_value': 50.0,  # VH scaling par défaut
        'log_transform': True,  # Nouveau paramètre
        'residual_despeckling': True  # Nouveau paramètre
    }
    
    # Mettre à jour avec les options fournies
    options.update(advanced_options)
    
    # Pipeline de prétraitement
    processed = image.copy()
    
    # 1. Mise à l'échelle
    processed = processed / options['scaling_value']
    
    # 2. Clip des valeurs extrêmes
    processed = np.clip(processed, 0, 5)
    
    # 3. Transformation logarithmique
    if options['log_transform']:
        processed = apply_log_transform(processed)
    
    # 4. Réduction du bruit speckle
    if options['residual_despeckling']:
        if options['log_transform']:
            # Si log déjà appliqué, pas besoin de le réappliquer
            processed = apply_residual_despeckling(processed, log_transform=False)
        else:
            processed = apply_residual_despeckling(processed)
    elif options['speckle_filter']:
        processed = apply_speckle_filter(processed, 
                                        filter_type=options['filter_type'])
    
    # 5. Transformation inverse du logarithme si nécessaire
    if options['log_transform'] and not (options['enhance_contrast'] or options['preserve_ratio']):
        processed = apply_inverse_log_transform(processed)
    
    # 6. Amélioration du contraste
    if options['enhance_contrast']:
        processed = enhance_contrast(processed, 
                                   method=options['contrast_method'])
    
    # 7. Transformation inverse du logarithme après amélioration du contraste
    if options['log_transform'] and options['enhance_contrast']:
        processed = apply_inverse_log_transform(processed)
    
    # 8. Redimensionnement
    if options['preserve_ratio']:
        processed = preserve_aspect_ratio_resize(processed, 
                                              target_size=options['target_size'])
    else:
        processed = resize_image(processed, size=options['target_size'])
    
    # 9. Normalisation finale
    processed = normalize_image(processed)
    
    return processed

def preserve_aspect_ratio_resize(image, target_size=(256, 256), pad_value=0):
    """
    Redimensionne l'image en préservant le ratio d'aspect et ajoute du padding si nécessaire
    
    Args:
        image (np.ndarray): Image à redimensionner
        target_size (tuple): Taille cible (hauteur, largeur)
        pad_value (float): Valeur utilisée pour le padding
        
    Returns:
        np.ndarray: Image redimensionnée avec padding
    """
    h, w = image.shape
    target_h, target_w = target_size
    
    # Calculer le ratio pour garder les proportions
    ratio = min(target_h / h, target_w / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # Redimensionner l'image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Créer une image vide (padding) de la taille cible
    padded = np.full((target_h, target_w), pad_value, dtype=resized.dtype)
    
    # Calculer la position pour centrer l'image redimensionnée
    h_offset = (target_h - new_h) // 2
    w_offset = (target_w - new_w) // 2
    
    # Placer l'image redimensionnée sur l'image vide
    padded[h_offset:h_offset+new_h, w_offset:w_offset+new_w] = resized
    
    return padded


def analyser_pixels_sar(chemin_image, scaling_value=50.0, roi_size=10, show_full_stats=True):

    """
    Analyse détaillée des valeurs de pixels d'une image SAR
    
    Args:
        chemin_image (str): Chemin vers l'image SAR
        scaling_value (float): Valeur de mise à l'échelle (50.0 pour VH, 100.0 pour VV)
        roi_size (int): Taille de la région d'intérêt pour l'affichage des valeurs de pixels
        show_full_stats (bool): Afficher les statistiques complètes et l'histogramme
    """
  
    
    # Charger l'image
    with rasterio.open(chemin_image) as src:
        # Lire les métadonnées
        height = src.height
        width = src.width
        
        # Lire l'image complète
        image = src.read(1)
        
        # Appliquer la mise à l'échelle standard
        scaled_image = image / scaling_value
        
        print(f"Taille de l'image: {height}×{width} pixels")
        print(f"Valeur minimale: {np.min(scaled_image):.8f}")
        print(f"Valeur maximale: {np.max(scaled_image):.8f}")
        print(f"Valeur moyenne: {np.mean(scaled_image):.8f}")
        print(f"Écart-type: {np.std(scaled_image):.8f}")
        
        # Extraire une petite région au centre de l'image
        center_h, center_w = height // 2, width // 2
        start_h, start_w = center_h - roi_size // 2, center_w - roi_size // 2
        
        # Essayer d'abord de trouver une région avec des valeurs variées
        # Échantillonner plusieurs régions et choisir celle avec le plus grand écart-type
        regions = []
        for i in range(5):  # Essayer 5 régions différentes
            h_offset = np.random.randint(0, height - roi_size)
            w_offset = np.random.randint(0, width - roi_size)
            region = scaled_image[h_offset:h_offset+roi_size, w_offset:w_offset+roi_size]
            regions.append((region, h_offset, w_offset, np.std(region)))
        
        # Trier par écart-type décroissant
        regions.sort(key=lambda x: x[3], reverse=True)
        best_region, h_offset, w_offset, _ = regions[0]
        
        # Afficher la région sélectionnée et ses valeurs
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Afficher l'image complète avec un rectangle pour indiquer la région
        im0 = axes[0].imshow(scaled_image, cmap='viridis')
        rect = plt.Rectangle((w_offset, h_offset), roi_size, roi_size, 
                           edgecolor='red', facecolor='none', linewidth=2)
        axes[0].add_patch(rect)
        axes[0].set_title("Image complète")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Afficher les valeurs de pixels dans la région sélectionnée
        im1 = axes[1].imshow(best_region, cmap='viridis')
        axes[1].set_title(f"Valeurs des pixels (région {roi_size}×{roi_size})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Ajouter les valeurs numériques
        for i in range(roi_size):
            for j in range(roi_size):
                value = best_region[i, j]
                # Format d'affichage selon la grandeur des valeurs
                if value < 0.001:
                    value_str = f"{value:.2e}"  # Notation scientifique pour les très petites valeurs
                else:
                    value_str = f"{value:.4f}"
                
                # Couleur du texte (blanc sur fond foncé, noir sur fond clair)
                text_color = 'white' if value < np.mean(best_region) else 'black'
                axes[1].text(j, i, value_str, ha='center', va='center', 
                          color=text_color, fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        if show_full_stats:
            # Créer une figure pour les statistiques détaillées
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Histogramme standard
            axes[0, 0].hist(scaled_image.flatten(), bins=100, color='skyblue', edgecolor='black')
            axes[0, 0].set_title("Histogramme des valeurs de pixels")
            axes[0, 0].set_xlabel("Valeur de pixel")
            axes[0, 0].set_ylabel("Fréquence")
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Histogramme logarithmique pour mieux voir les faibles valeurs
            axes[0, 1].hist(scaled_image.flatten(), bins=100, color='skyblue', edgecolor='black', log=True)
            axes[0, 1].set_title("Histogramme logarithmique (fréquence)")
            axes[0, 1].set_xlabel("Valeur de pixel")
            axes[0, 1].set_ylabel("Fréquence (log)")
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Afficher la distribution cumulée
            hist, bin_edges = np.histogram(scaled_image.flatten(), bins=100)
            cdf = np.cumsum(hist) / np.sum(hist)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            axes[1, 0].plot(bin_centers, cdf, 'b-', linewidth=2)
            axes[1, 0].set_title("Distribution cumulée")
            axes[1, 0].set_xlabel("Valeur de pixel")
            axes[1, 0].set_ylabel("Probabilité cumulée")
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Analyse par catégorie de terrain
            # Définir les seuils d'après vos remarques
            water_mask = (scaled_image <= 0.0005)
            bare_soil_mask = (scaled_image > 0.0005) & (scaled_image <= 0.01)
            vegetation_mask = (scaled_image > 0.01) & (scaled_image <= 0.1)
            structures_mask = (scaled_image > 0.1)
            
            # Calculer les pourcentages
            total_pixels = scaled_image.size
            water_percent = np.sum(water_mask) / total_pixels * 100
            bare_soil_percent = np.sum(bare_soil_mask) / total_pixels * 100
            vegetation_percent = np.sum(vegetation_mask) / total_pixels * 100
            structures_percent = np.sum(structures_mask) / total_pixels * 100
            
            # Graphique en camembert
            categories = ['Eau (≤0.0005)', 'Sol nu (0.0005-0.01)', 
                       'Végétation (0.01-0.1)', 'Structures (>0.1)']
            sizes = [water_percent, bare_soil_percent, vegetation_percent, structures_percent]
            colors = ['blue', 'brown', 'green', 'gray']
            
            axes[1, 1].pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%',
                        startangle=90, shadow=True)
            axes[1, 1].set_title("Répartition estimée des types de terrain")
            
            plt.tight_layout()
            plt.show()
            
            # Afficher des statistiques par catégorie
            print("\nStatistiques par catégorie estimée de terrain:")
            print(f"Eau (≤0.0005): {water_percent:.2f}% des pixels")
            if water_percent > 0:
                print(f"  Moyenne: {np.mean(scaled_image[water_mask]):.8f}")
                print(f"  Écart-type: {np.std(scaled_image[water_mask]):.8f}")
            
            print(f"Sol nu (0.0005-0.01): {bare_soil_percent:.2f}% des pixels")
            if bare_soil_percent > 0:
                print(f"  Moyenne: {np.mean(scaled_image[bare_soil_mask]):.8f}")
                print(f"  Écart-type: {np.std(scaled_image[bare_soil_mask]):.8f}")
            
            print(f"Végétation (0.01-0.1): {vegetation_percent:.2f}% des pixels")
            if vegetation_percent > 0:
                print(f"  Moyenne: {np.mean(scaled_image[vegetation_mask]):.8f}")
                print(f"  Écart-type: {np.std(scaled_image[vegetation_mask]):.8f}")
            
            print(f"Structures (>0.1): {structures_percent:.2f}% des pixels")
            if structures_percent > 0:
                print(f"  Moyenne: {np.mean(scaled_image[structures_mask]):.8f}")
                print(f"  Écart-type: {np.std(scaled_image[structures_mask]):.8f}")




def preprocessing_pipeline_image(image, preprocessing_params=None, visualize_steps=False):
    """
    Pipeline complet de prétraitement pour images SAR avant entrée dans un CNN
    
    Args:
        image (np.ndarray): Image SAR brute (matrice 2D)
        preprocessing_params (dict): Paramètres de prétraitement
        visualize_steps (bool): Afficher les étapes intermédiaires du prétraitement
        
    Returns:
        np.ndarray: Image prétraitée prête pour l'entrée dans un CNN
    """
    # Paramètres par défaut optimisés pour la détection d'inondation
    default_params = {
        # Étape 1: Mise à l'échelle
        'apply_scaling': True,
        'scaling_value': 50.0,  # 50.0 pour VH, 100.0 pour VV
        'clip_values': True,
        'clip_min': 0.0,
        'clip_max': 5.0,
        
        # Étape 2: Transformation logarithmique
        'apply_log_transform': True,
        
        # Étape 3: Réduction du bruit speckle
        'apply_despeckling': True,
        'despeckling_method': 'residual',  # 'residual', 'lee', 'median', 'bilateral', 'frost', 'kuan'
        'despeckling_window_size': 5,
        'residual_strength': 0.9,  # Force de la soustraction résiduelle (0-1)
        
        # Étape 4: Amélioration du contraste
        'apply_contrast_enhancement': True,
        'contrast_method': 'clahe',  # 'histogram_equalization', 'clahe', 'adaptive_gamma'
        'clahe_clip_limit': 0.03,
        'clahe_tile_size': 8,
        'gamma_value': 0.8,
        
        # Étape 5: Expansion dynamique (rescale intensity)
        'apply_dynamic_expansion': True,
        'expansion_percentiles': (2, 98),  # Percentiles pour l'expansion
        
        # Étape 6: Redimensionnement
        'apply_resize': True,
        'target_size': (256, 256),
        'preserve_aspect_ratio': True,
        'pad_value': 0.0,
        
        # Étape 7: Normalisation finale
        'apply_normalization': True,
        'normalization_method': 'standardization',  # 'minmax', 'standardization', 'custom'
        'normalization_mean': 0.5,
        'normalization_std': 0.5,
        'normalization_min': 0.0,
        'normalization_max': 1.0
    }
    
    # Mise à jour des paramètres par défaut avec ceux fournis
    params = default_params.copy()
    if preprocessing_params:
        params.update(preprocessing_params)
    
    # Initialiser la liste d'images intermédiaires pour la visualisation
    steps_images = []
    steps_names = []
    
    # Stocker l'image originale
    original_image = image.copy()
    steps_images.append(original_image)
    steps_names.append("Image originale")
    
    # Étape 1: Mise à l'échelle et écrêtage
    processed = image.copy()
    
    if params['apply_scaling']:
        processed = processed / params['scaling_value']
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après scaling (÷{params['scaling_value']})")
    
    if params['clip_values']:
        processed = np.clip(processed, params['clip_min'], params['clip_max'])
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après clipping ({params['clip_min']}-{params['clip_max']})")
    
    # Étape 2: Transformation logarithmique
    if params['apply_log_transform']:
        # Éviter log(0)
        epsilon = 1e-10
        processed = np.log(np.maximum(processed, epsilon))
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append("Après transformation logarithmique")
    
    # Étape 3: Réduction du bruit speckle
    if params['apply_despeckling']:
        window_size = params['despeckling_window_size']
        
        if params['despeckling_method'] == 'residual':
            # Approche résiduelle pour le despeckling
            # Appliquer un filtre pour estimer l'image sans bruit
            mean = ndimage.uniform_filter(processed, size=window_size)
            variance = ndimage.uniform_filter(processed**2, size=window_size) - mean**2
            epsilon = 1e-10
            img_variance = np.mean(variance)
            filtered = mean + (variance / (variance + img_variance + epsilon)) * (processed - mean)
            
            # Calculer le bruit résiduel
            noise_residual = processed - filtered
            
            # Soustraire une version réduite du bruit résiduel
            processed = processed - params['residual_strength'] * noise_residual
            
        elif params['despeckling_method'] == 'lee':
            # Filtre de Lee
            mean = ndimage.uniform_filter(processed, size=window_size)
            variance = ndimage.uniform_filter(processed**2, size=window_size) - mean**2
            epsilon = 1e-10
            img_variance = np.mean(variance)
            processed = mean + (variance / (variance + img_variance + epsilon)) * (processed - mean)
            
        elif params['despeckling_method'] == 'median':
            # Filtre médian
            processed = ndimage.median_filter(processed, size=window_size)
            
        elif params['despeckling_method'] == 'bilateral':
            # Filtre bilatéral (pour préserver les bords)
            # Normaliser temporairement pour le filtre bilatéral
            norm_img = (processed - np.min(processed)) / (np.max(processed) - np.min(processed) + 1e-10)
            norm_img = (norm_img * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(norm_img, window_size, 75, 75)
            # Revenir à l'échelle originale
            min_val, max_val = np.min(processed), np.max(processed)
            processed = (filtered.astype(np.float32) / 255) * (max_val - min_val) + min_val
            
        elif params['despeckling_method'] == 'frost':
            # Version simplifiée du filtre de Frost
            sigma = ndimage.generic_filter(processed, np.std, size=window_size)
            weight = np.exp(-4 * sigma / (np.mean(sigma) + 1e-10))
            processed = ndimage.gaussian_filter(processed, sigma=window_size/4 * weight)
            
        elif params['despeckling_method'] == 'kuan':
            # Version simplifiée du filtre de Kuan
            mean = ndimage.uniform_filter(processed, size=window_size)
            variance = ndimage.uniform_filter(processed**2, size=window_size) - mean**2
            ci = np.sqrt(variance) / (mean + 1e-10)  # Coefficient de variation
            cu = 0.25  # Coefficient de variation du bruit (typique pour SAR)
            w = (1 - cu**2/ci**2) / (1 + cu**2)
            w = np.clip(w, 0, 1)
            processed = w * processed + (1 - w) * mean
            
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après despeckling ({params['despeckling_method']})")
    
    # Étape 4: Transformation inverse du logarithme (si applicable)
    if params['apply_log_transform'] and not (params['apply_contrast_enhancement'] or params['apply_dynamic_expansion']):
        processed = np.exp(processed)
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append("Après transformation exponentielle")
    
    # Étape 5: Amélioration du contraste
    if params['apply_contrast_enhancement']:
        if params['contrast_method'] == 'histogram_equalization':
            # Égalisation d'histogramme globale
            processed = exposure.equalize_hist(processed)
            
        elif params['contrast_method'] == 'clahe':
            # CLAHE - Contrast Limited Adaptive Histogram Equalization
            processed = exposure.equalize_adapthist(
                processed, 
                clip_limit=params['clahe_clip_limit'],
                kernel_size=params['clahe_tile_size']
            )
            
        elif params['contrast_method'] == 'adaptive_gamma':
            # Correction gamma adaptative
            processed = exposure.adjust_gamma(processed, gamma=params['gamma_value'])
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après amélioration du contraste ({params['contrast_method']})")
    
    # Transformation inverse du logarithme après amélioration du contraste
    if params['apply_log_transform'] and (params['apply_contrast_enhancement'] or params['apply_dynamic_expansion']):
        processed = np.exp(processed)
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append("Après transformation exponentielle")
    
    # Étape 6: Expansion dynamique
    if params['apply_dynamic_expansion']:
        p_low, p_high = params['expansion_percentiles']
        p2, p98 = np.percentile(processed, (p_low, p_high))
        processed = exposure.rescale_intensity(processed, in_range=(p2, p98))
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après expansion dynamique (p{p_low}-p{p_high})")
    
    # Étape 7: Redimensionnement
    if params['apply_resize']:
        if params['preserve_aspect_ratio']:
            # Redimensionnement avec préservation du ratio d'aspect
            h, w = processed.shape
            target_h, target_w = params['target_size']
            
            # Calculer le ratio pour garder les proportions
            ratio = min(target_h / h, target_w / w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            
            # Redimensionner l'image
            resized = cv2.resize(processed, (new_w, new_h))
            
            # Créer une image vide (padding) de la taille cible
            padded = np.full(params['target_size'], params['pad_value'], dtype=resized.dtype)
            
            # Calculer la position pour centrer l'image redimensionnée
            h_offset = (target_h - new_h) // 2
            w_offset = (target_w - new_w) // 2
            
            # Placer l'image redimensionnée sur l'image vide
            padded[h_offset:h_offset+new_h, w_offset:w_offset+new_w] = resized
            processed = padded
        else:
            # Redimensionnement simple
            processed = cv2.resize(processed, params['target_size'])
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après redimensionnement ({params['target_size'][0]}×{params['target_size'][1]})")
    
    # Étape 8: Normalisation finale
    if params['apply_normalization']:
        if params['normalization_method'] == 'minmax':
            # Normalisation Min-Max
            min_val, max_val = np.min(processed), np.max(processed)
            if max_val > min_val:
                processed = (processed - min_val) / (max_val - min_val)
                processed = processed * (params['normalization_max'] - params['normalization_min']) + params['normalization_min']
                
        elif params['normalization_method'] == 'standardization':
            # Standardisation (Z-score)
            mean = params['normalization_mean']
            std = params['normalization_std']
            processed = (processed - mean) / std
            
        elif params['normalization_method'] == 'custom':
            # Normalisation personnalisée
            processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-10)
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après normalisation ({params['normalization_method']})")
    
    # Visualiser les étapes si demandé
    if visualize_steps:
        n_steps = len(steps_images)
        n_cols = min(3, n_steps)
        n_rows = (n_steps + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (img, name) in enumerate(zip(steps_images, steps_names)):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Afficher l'image
            im = ax.imshow(img, cmap='viridis')
            ax.set_title(name)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        
        # Désactiver les axes inutilisés
        for i in range(n_steps, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Afficher les statistiques avant/après
        print("\nStatistiques de l'image:")
        print(f"{'Étape':<30} {'Min':<10} {'Max':<10} {'Moyenne':<10} {'Écart-type':<10}")
        print("-" * 70)
        
        for i, (img, name) in enumerate(zip(steps_images, steps_names)):
            print(f"{name:<30} {np.min(img):<10.5f} {np.max(img):<10.5f} {np.mean(img):<10.5f} {np.std(img):<10.5f}")
    
    return processed


def preprocessing_pipeline_image(chemin_image : str , preprocessing_params : dict =None, visualize_steps : bool =False, detect_polarization : bool =True):
    """
    Pipeline complet de prétraitement pour images SAR avant entrée dans un CNN
    
    Args:
        chemin_image (str): Chemin vers l'image SAR
        preprocessing_params (dict): Paramètres de prétraitement
        visualize_steps (bool): Afficher les étapes intermédiaires du prétraitement
        detect_polarization (bool): Détecter automatiquement la polarisation (VH/VV)
        
    Returns:
        np.ndarray: Image prétraitée prête pour l'entrée dans un CNN
    """
    # Charger l'image avec rasterio
    with rasterio.open(chemin_image) as src:
        # Lire les métadonnées
        height = src.height
        width = src.width
        
        # Lire l'image
        image = src.read(1)
        
        # Détection de polarisation automatique
        if detect_polarization:
            # Vérifier si 'vh' ou 'vv' est dans le nom du fichier (insensible à la casse)
            filename = chemin_image.lower()
            if 'vh' in filename:
                default_scaling = 50.0
                polarization = 'VH'
            elif 'vv' in filename:
                default_scaling = 100.0
                polarization = 'VV'
            else:
                # Valeur par défaut si impossible à déterminer
                default_scaling = 50.0
                polarization = 'Indéterminée'
            
            print(f"Polarisation détectée: {polarization} (scaling {default_scaling})")
        else:
            default_scaling = 50.0
    
    # Paramètres par défaut optimisés pour la détection d'inondation
    default_params = {
        # Étape 1: Mise à l'échelle
        'apply_scaling': True,
        'scaling_value': default_scaling,  # Détecté automatiquement ou 50.0 par défaut
        'clip_values': True,
        'clip_min': 0.0,
        'clip_max': 5.0,
        
        # Étape 2: Transformation logarithmique
        'apply_log_transform': True,
        
        # Étape 3: Réduction du bruit speckle
        'apply_despeckling': True,
        'despeckling_method': 'residual',  # 'residual', 'lee', 'median', 'bilateral', 'frost', 'kuan'
        'despeckling_window_size': 5,
        'residual_strength': 0.9,  # Force de la soustraction résiduelle (0-1)
        
        # Étape 4: Amélioration du contraste
        'apply_contrast_enhancement': True,
        'contrast_method': 'clahe',  # 'histogram_equalization', 'clahe', 'adaptive_gamma'
        'clahe_clip_limit': 0.03,
        'clahe_tile_size': 8,
        'gamma_value': 0.8,
        
        # Étape 5: Expansion dynamique (rescale intensity)
        'apply_dynamic_expansion': True,
        'expansion_percentiles': (2, 98),  # Percentiles pour l'expansion
        
        # Étape 6: Redimensionnement
        'apply_resize': True,
        'target_size': (256, 256),
        'preserve_aspect_ratio': True,
        'pad_value': 0.0,
        
        # Étape 7: Normalisation finale
        'apply_normalization': True,
        'normalization_method': 'standardization',  # 'minmax', 'standardization', 'custom'
        'normalization_mean': 0.5,
        'normalization_std': 0.5,
        'normalization_min': 0.0,
        'normalization_max': 1.0
    }
    
    # Mise à jour des paramètres par défaut avec ceux fournis
    params = default_params.copy()
    if preprocessing_params:
        params.update(preprocessing_params)
    
    # Afficher les informations de base sur l'image
    print(f"Taille de l'image: {height}×{width} pixels")
    print(f"Valeurs brutes - Min: {np.min(image):.8f}, Max: {np.max(image):.8f}, Moy: {np.mean(image):.8f}")
    
    # Initialiser la liste d'images intermédiaires pour la visualisation
    steps_images = []
    steps_names = []
    
    # Stocker l'image originale
    original_image = image.copy()
    steps_images.append(original_image)
    steps_names.append("Image originale")
    
    # Étape 1: Mise à l'échelle et écrêtage
    processed = image.copy()
    
    if params['apply_scaling']:
        processed = processed / params['scaling_value']
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après scaling (÷{params['scaling_value']})")
    
    if params['clip_values']:
        processed = np.clip(processed, params['clip_min'], params['clip_max'])
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après clipping ({params['clip_min']}-{params['clip_max']})")
    
    # Étape 2: Transformation logarithmique
    if params['apply_log_transform']:
        # Éviter log(0)
        epsilon = 1e-10
        processed = np.log(np.maximum(processed, epsilon))
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append("Après transformation logarithmique")
    
    # Étape 3: Réduction du bruit speckle
    if params['apply_despeckling']:
        window_size = params['despeckling_window_size']
        
        if params['despeckling_method'] == 'residual':
            # Approche résiduelle pour le despeckling
            # Appliquer un filtre pour estimer l'image sans bruit
            mean = ndimage.uniform_filter(processed, size=window_size)
            variance = ndimage.uniform_filter(processed**2, size=window_size) - mean**2
            epsilon = 1e-10
            img_variance = np.mean(variance)
            filtered = mean + (variance / (variance + img_variance + epsilon)) * (processed - mean)
            
            # Calculer le bruit résiduel
            noise_residual = processed - filtered
            
            # Soustraire une version réduite du bruit résiduel
            processed = processed - params['residual_strength'] * noise_residual
            
        elif params['despeckling_method'] == 'lee':
            # Filtre de Lee
            mean = ndimage.uniform_filter(processed, size=window_size)
            variance = ndimage.uniform_filter(processed**2, size=window_size) - mean**2
            epsilon = 1e-10
            img_variance = np.mean(variance)
            processed = mean + (variance / (variance + img_variance + epsilon)) * (processed - mean)
            
        elif params['despeckling_method'] == 'median':
            # Filtre médian
            processed = ndimage.median_filter(processed, size=window_size)
            
        elif params['despeckling_method'] == 'bilateral':
            # Filtre bilatéral (pour préserver les bords)
            # Normaliser temporairement pour le filtre bilatéral
            norm_img = (processed - np.min(processed)) / (np.max(processed) - np.min(processed) + 1e-10)
            norm_img = (norm_img * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(norm_img, window_size, 75, 75)
            # Revenir à l'échelle originale
            min_val, max_val = np.min(processed), np.max(processed)
            processed = (filtered.astype(np.float32) / 255) * (max_val - min_val) + min_val
            
        elif params['despeckling_method'] == 'frost':
            # Version simplifiée du filtre de Frost
            sigma = ndimage.generic_filter(processed, np.std, size=window_size)
            weight = np.exp(-4 * sigma / (np.mean(sigma) + 1e-10))
            processed = ndimage.gaussian_filter(processed, sigma=window_size/4 * weight)
            
        elif params['despeckling_method'] == 'kuan':
            # Version simplifiée du filtre de Kuan
            mean = ndimage.uniform_filter(processed, size=window_size)
            variance = ndimage.uniform_filter(processed**2, size=window_size) - mean**2
            ci = np.sqrt(variance) / (mean + 1e-10)  # Coefficient de variation
            cu = 0.25  # Coefficient de variation du bruit (typique pour SAR)
            w = (1 - cu**2/ci**2) / (1 + cu**2)
            w = np.clip(w, 0, 1)
            processed = w * processed + (1 - w) * mean
            
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après despeckling ({params['despeckling_method']})")
    
    # Étape 4: Transformation inverse du logarithme (si applicable)
    if params['apply_log_transform'] and not (params['apply_contrast_enhancement'] or params['apply_dynamic_expansion']):
        processed = np.exp(processed)
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append("Après transformation exponentielle")
    
    # Étape 5: Amélioration du contraste
    if params['apply_contrast_enhancement']:
        if params['contrast_method'] == 'histogram_equalization':
            # Égalisation d'histogramme globale
            processed = exposure.equalize_hist(processed)
            
        elif params['contrast_method'] == 'clahe':
            # CLAHE - Contrast Limited Adaptive Histogram Equalization
            processed = exposure.equalize_adapthist(
                processed, 
                clip_limit=params['clahe_clip_limit'],
                kernel_size=params['clahe_tile_size']
            )
            
        elif params['contrast_method'] == 'adaptive_gamma':
            # Correction gamma adaptative
            processed = exposure.adjust_gamma(processed, gamma=params['gamma_value'])
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après amélioration du contraste ({params['contrast_method']})")
    
    # Transformation inverse du logarithme après amélioration du contraste
    if params['apply_log_transform'] and (params['apply_contrast_enhancement'] or params['apply_dynamic_expansion']):
        processed = np.exp(processed)
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append("Après transformation exponentielle")
    
    # Étape 6: Expansion dynamique
    if params['apply_dynamic_expansion']:
        p_low, p_high = params['expansion_percentiles']
        p2, p98 = np.percentile(processed, (p_low, p_high))
        processed = exposure.rescale_intensity(processed, in_range=(p2, p98))
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après expansion dynamique (p{p_low}-p{p_high})")
    
    # Étape 7: Redimensionnement
    if params['apply_resize']:
        if params['preserve_aspect_ratio']:
            # Redimensionnement avec préservation du ratio d'aspect
            h, w = processed.shape
            target_h, target_w = params['target_size']
            
            # Calculer le ratio pour garder les proportions
            ratio = min(target_h / h, target_w / w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            
            # Redimensionner l'image
            resized = cv2.resize(processed, (new_w, new_h))
            
            # Créer une image vide (padding) de la taille cible
            padded = np.full(params['target_size'], params['pad_value'], dtype=resized.dtype)
            
            # Calculer la position pour centrer l'image redimensionnée
            h_offset = (target_h - new_h) // 2
            w_offset = (target_w - new_w) // 2
            
            # Placer l'image redimensionnée sur l'image vide
            padded[h_offset:h_offset+new_h, w_offset:w_offset+new_w] = resized
            processed = padded
        else:
            # Redimensionnement simple
            processed = cv2.resize(processed, params['target_size'])
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après redimensionnement ({params['target_size'][0]}×{params['target_size'][1]})")
    
    # Étape 8: Normalisation finale
    if params['apply_normalization']:
        if params['normalization_method'] == 'minmax':
            # Normalisation Min-Max
            min_val, max_val = np.min(processed), np.max(processed)
            if max_val > min_val:
                processed = (processed - min_val) / (max_val - min_val)
                processed = processed * (params['normalization_max'] - params['normalization_min']) + params['normalization_min']
                
        elif params['normalization_method'] == 'standardization':
            # Standardisation (Z-score)
            mean = params['normalization_mean']
            std = params['normalization_std']
            processed = (processed - mean) / std
            
        elif params['normalization_method'] == 'custom':
            # Normalisation personnalisée
            processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-10)
        
        if visualize_steps:
            steps_images.append(processed.copy())
            steps_names.append(f"Après normalisation ({params['normalization_method']})")
    
    # Visualiser les étapes si demandé
    if visualize_steps:
        n_steps = len(steps_images)
        n_cols = min(3, n_steps)
        n_rows = (n_steps + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (img, name) in enumerate(zip(steps_images, steps_names)):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Afficher l'image
            im = ax.imshow(img, cmap='viridis')
            ax.set_title(name)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        
        # Désactiver les axes inutilisés
        for i in range(n_steps, n_rows * n_cols):
            if i < n_rows * n_cols:  # S'assurer qu'on ne dépasse pas les indices valides
                row, col = i // n_cols, i % n_cols
                axes[row, col].axis('off')
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Afficher les statistiques avant/après
        print("\nStatistiques de l'image:")
        print(f"{'Étape':<30} {'Min':<10} {'Max':<10} {'Moyenne':<10} {'Écart-type':<10}")
        print("-" * 70)
        
        for i, (img, name) in enumerate(zip(steps_images, steps_names)):
            print(f"{name:<30} {np.min(img):<10.5f} {np.max(img):<10.5f} {np.mean(img):<10.5f} {np.std(img):<10.5f}")
    
    return processed