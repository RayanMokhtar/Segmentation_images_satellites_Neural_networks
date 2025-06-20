import openeo
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image, ImageEnhance
from typing import Literal, Optional
import glob
import rasterio
from scipy import ndimage
import cv2
from scipy.ndimage import zoom


def get_image_satellitaire(
    lng: float, 
    lat: float, 
    region: Optional[str] = None, 
    is_telecharge: bool = True,
    format: Literal["brut", "traité"] = "traité",
    is_visualised: bool = False,
    date_debut: Optional[str] = None,
    date_fin: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Télécharge et traite une image SAR Sentinel-1 avec contrôle total
    
    Paramètres:
    -----------
    lng : float
        Longitude du centre de la région
    lat : float
        Latitude du centre de la région
    region : str, optionnel
        Nom de la région (ex: "Paris", "Nice", "Lyon")
        Si None, génère automatiquement: "region_lat_lng"
    is_telecharge : bool
        True = Télécharge depuis Copernicus
        False = Utilise une image existante dans ./sar_images/
    format : Literal["brut", "traité"]
        "brut" = Retourne l'image TIFF brute sans traitement
        "traité" = Applique le pipeline de traitement SAR
    is_visualised : bool
        True = Affiche les visualisations détaillées
        False = Pas de visualisation
        
    Retourne:
    ---------
    str : Chemin vers l'image finale ou None si échec
    
    Exemples:
    ---------
    # Image brute téléchargée
    path = get_image_satellitaire(2.3522, 48.8566, "Paris", True, "brut", False)
    
    # Image traitée avec visualisation
    path = get_image_satellitaire(2.3522, 48.8566, "Paris", True, "traité", True)
    
    # Traitement d'une image existante
    path = get_image_satellitaire(2.3522, 48.8566, "Paris", False, "traité", True)
    """
    
    print(f"🛰️ GESTION IMAGE SATELLITE COPERNICUS")
    print("="*60)
    
    # 📍 ÉTAPE 1: Configuration de la région
    if region is None:
        region = f"region_{lat:.3f}_{lng:.3f}"
    
    print(f"📍 Région: {region}")
    print(f"📍 Coordonnées: {lat:.4f}, {lng:.4f}")
    print(f"📥 Téléchargement: {'✅ OUI' if is_telecharge else '❌ NON (utilise existant)'}")
    print(f"🎨 Format: {'📄 BRUT' if format == 'brut' else '⚙️ TRAITÉ'}")
    print(f"📊 Visualisation: {'✅ ACTIVÉE' if is_visualised else '❌ DÉSACTIVÉE'}")
    
    # 📁 ÉTAPE 2: Préparer les dossiers
    output_dir = "./sar_images"
    os.makedirs(output_dir, exist_ok=True)
    
    
    print(f"\n📡 TÉLÉCHARGEMENT DEPUIS COPERNICUS")
    print("-"*40)
    tiff_file = telecharger_sar_copernicus(lng, lat, region, output_dir , date_debut, date_fin)
    if not tiff_file:
        print("❌ Téléchargement échoué")
        return None
        
    # 🎯 ÉTAPE 4: Gestion du format
    if format == "brut":
        print(f"\n📄 FORMAT BRUT SÉLECTIONNÉ")
        print("-"*30)
        print(f"✅ Retour image brute: {os.path.basename(tiff_file)}")
        
        # Visualisation optionnelle même pour le brut
        if is_visualised:
            visualiser_image_brute(tiff_file, region)
    
        try:
            with rasterio.open(tiff_file) as src:
                # Lire toutes les bandes
                data = src.read()  # Shape: (bands, height, width)
                print(f"✅ Image brute chargée: {data.shape} (bands, H, W)")
                print(f"📊 Statistiques:")
                for i in range(data.shape[0]):
                    band = data[i]
                    print(f"   Bande {i+1}: min={band.min():.3f}, max={band.max():.3f}, moy={band.mean():.3f}")
                
                return data
                
        except Exception as e:
            print(f"❌ Erreur lecture image brute: {e}")
            return None    
   
    elif format == "traité":
        print(f"\n⚙️ FORMAT SEN12FLOOD SÉLECTIONNÉ")
        print("-"*30)

        if is_visualised:
            # Faire la visualisation
            visualiser_brut_et_traite(tiff_file, output_dir, region)
            # Puis charger et retourner l'image traitée
            sen12flood_file = adapt_to_senflood12(tiff_file, output_dir, region, False)
            if sen12flood_file:
                with rasterio.open(sen12flood_file) as src:
                    data = src.read()  # shape (2, 512, 512)
                    print(f"✅ Image traitée chargée: {data.shape}")
                    return data
            else:
                print("❌ Échec de traitement avec visualisation")
                return None
        else:
            # Traitement sans visualisation
            sen12flood_file = adapt_to_senflood12(tiff_file, output_dir, region, False)
            if sen12flood_file:
                with rasterio.open(sen12flood_file) as src:
                    data = src.read()  # shape (2, 512, 512)
                    print(f"✅ Image SEN12FLOOD chargée: {data.shape}")
                    return data
            else:
                print("⚠️ Conversion SEN12FLOOD échouée, retour image brute")
                # Essayer de charger l'image brute
                try:
                    with rasterio.open(tiff_file) as src:
                        return src.read()
                except:
                    return None
            

def telecharger_sar_copernicus(lng: float, lat: float, region: str, output_dir: str, date_debut: Optional[str] = None, 
    date_fin: Optional[str] = None) -> Optional[str]:
    """
    Télécharge une image SAR Sentinel-1
    Remarque : si pas de date_debut ou date_fin, on prend le jour actuel 
    """
    try:
        # 🔧 Zone de téléchargement
        bbox_size = 0.05  # ~10km x 10km
        ouest = lng - bbox_size
        est = lng + bbox_size
        sud = lat - bbox_size
        nord = lat + bbox_size
        
        print(f"📦 Zone: {ouest:.4f}, {sud:.4f}, {est:.4f}, {nord:.4f}")
        
        # 🔗 Connexion OpenEO
        print("🔗 Connexion OpenEO...")
        connection = openeo.connect("https://earthengine.openeo.org")
        print("✅ Connexion établie")
        
        # 🔐 Authentification
        print("🔐 Authentification...")
        connection.authenticate_oidc(
            client_id="693079969436-06hpof40l5qmh60ieh0cg5jnulj2ctis.apps.googleusercontent.com",
            client_secret="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"
        )
        print("✅ Authentification réussie")
        
        # 📅 Période - VÉRIFICATION DES DATES FUTURES
        today = datetime.now()
        
        # Convertir les dates en objets datetime pour comparaison
        date_debut_dt = datetime.strptime(date_debut, '%Y-%m-%d') if date_debut else None
        date_fin_dt = datetime.strptime(date_fin, '%Y-%m-%d') if date_fin else None
        
        # Vérifier si les dates sont dans le futur
        if date_debut_dt and date_debut_dt > today:
            print(f"⚠️ Date de début {date_debut} est dans le futur!")
            # Utiliser 30 jours avant aujourd'hui
            date_debut = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            print(f"⚠️ Utilisation de {date_debut} à la place")
        
        if date_fin_dt and date_fin_dt > today:
            print(f"⚠️ Date de fin {date_fin} est dans le futur!")
            # Utiliser aujourd'hui
            date_fin = today.strftime('%Y-%m-%d')
            print(f"⚠️ Utilisation de {date_fin} à la place")
        
        # Définir les dates finales
        if date_fin is None:
            target_date = today.strftime('%Y-%m-%d')
        else:
            target_date = date_fin
        
        if date_debut is None:
            start_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            start_date = date_debut

        # S'assurer que les dates ne sont pas identiques (causes erreur API)
        if start_date == target_date:
            start_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"⚠️ Dates identiques, ajusté début à {start_date}")
            
        print(f"📅 Période recherchée: {start_date} à {target_date}")

        # 🛰️ Collection Sentinel-1
        print("📡 Chargement Sentinel-1...")
        
        datacube_s1 = connection.load_collection(
            "COPERNICUS/S1_GRD",
            spatial_extent={"west": ouest, "south": sud, "east": est, "north": nord},
            temporal_extent=[start_date, target_date]
        ) 
        print("✅ Collection chargée")
        metadata = datacube_s1.metadata
        print(f"📊 Métadonnées: {metadata}")
        # 💾 Téléchargement
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tiff_file = f"{output_dir}/{region}_sentinel_{start_date}_{target_date}_{timestamp}.tif"
        
        print(f"📥 Téléchargement: {os.path.basename(tiff_file)}")
        datacube_s1.download(tiff_file, format="GTiff")
        
        # ✅ Vérification
        if os.path.exists(tiff_file):
            size_mb = os.path.getsize(tiff_file) / (1024*1024)
            print(f"✅ Téléchargement réussi ({size_mb:.1f} MB)")
            return tiff_file
        else:
            print("❌ Fichier non créé")
            return None
            
    except Exception as e:
        print(f"❌ Erreur téléchargement: {e}")
        return None


def visualiser_brut_et_traite(tiff_file: str, output_dir: str, region: str):
    """
    Visualise l'image brute ET le traitement complet en une seule vue
    """
    try:
        print("🎬 VISUALISATION COMPLÈTE: BRUT → TRAITÉ")
        print("="*50)
        
        # 1. Charger l'image brute
        print("📖 Lecture image brute...")
        with rasterio.open(tiff_file) as src:
            print(f"📊 Image brute: {src.width}x{src.height}, {src.count} bandes")
            
            # Extraire VV et VH
            vv_brut = src.read(1)
            vh_brut = src.read(2) if src.count >= 2 else vv_brut.copy()
        
        # 2. Traitement SEN12FLOOD (récupérer toutes les étapes)
        print("⚙️ Traitement SEN12FLOOD...")
        
        # Conversion dB
        def to_db(data):
            if data.max() > 10:
                return 10 * np.log10(np.maximum(data, 1e-10))
            return data
        
        vv_db = to_db(vv_brut)
        vh_db = to_db(vh_brut)
        
        # Normalisation SEN12FLOOD
        db_min, db_max = -25.0, 5.0
        def normalize_exact(data_db):
            clipped = np.clip(data_db, db_min, db_max)
            return (clipped - db_min) / (db_max - db_min)
        
        vv_norm = normalize_exact(vv_db)
        vh_norm = normalize_exact(vh_db)
        
        # Redimensionnement 512x512
        h, w = vv_norm.shape
        zoom_h, zoom_w = 512/h, 512/w
        
        vv_final = zoom(vv_norm, (zoom_h, zoom_w), order=1)
        vh_final = zoom(vh_norm, (zoom_h, zoom_w), order=1)
        
        # 3. VISUALISATION COMPLÈTE
        print("📊 Création visualisation complète...")
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'🛰️ PIPELINE COMPLET: BRUT → SEN12FLOOD - {region}', 
                     fontsize=18, fontweight='bold')
        
        # Créer grille 3×4
        gs = plt.GridSpec(3, 4, wspace=0.3, hspace=0.4)
        
        # === LIGNE 1: IMAGES BRUTES ===
        # VV Brut
        ax1 = plt.subplot(gs[0, 0])
        im1 = ax1.imshow(vv_brut, cmap='gray')
        ax1.set_title('📄 VV BRUT\n(Valeurs originales)', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # VH Brut  
        ax2 = plt.subplot(gs[0, 1])
        im2 = ax2.imshow(vh_brut, cmap='gray')
        ax2.set_title('📄 VH BRUT\n(Valeurs originales)', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # === LIGNE 2: CONVERSION dB ===
        # VV dB
        ax3 = plt.subplot(gs[1, 0])
        im3 = ax3.imshow(vv_db, cmap='gray', vmin=-25, vmax=5)
        ax3.set_title('🔄 VV dB\n(-25 à +5 dB)', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # VH dB
        ax4 = plt.subplot(gs[1, 1])
        im4 = ax4.imshow(vh_db, cmap='gray', vmin=-25, vmax=5)
        ax4.set_title('🔄 VH dB\n(-25 à +5 dB)', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        # === LIGNE 3: FORMAT FINAL SEN12FLOOD ===
        # VV Final
        ax5 = plt.subplot(gs[2, 0])
        im5 = ax5.imshow(vv_final, cmap='gray', vmin=0, vmax=1)
        ax5.set_title('✅ VV FINAL\n512×512, [0-1]', fontweight='bold', color='green')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, shrink=0.8)
        
        # VH Final
        ax6 = plt.subplot(gs[2, 1])
        im6 = ax6.imshow(vh_final, cmap='gray', vmin=0, vmax=1)
        ax6.set_title('✅ VH FINAL\n512×512, [0-1]', fontweight='bold', color='green')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, shrink=0.8)
        
        # === COLONNE 3: HISTOGRAMMES ===
        # Histo VV
        ax7 = plt.subplot(gs[0, 2])
        ax7.hist(vv_brut.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        ax7.set_title('📊 Histogramme VV Brut', fontweight='bold')
        ax7.set_xlabel('Valeurs')
        ax7.grid(True, alpha=0.3)
        
        ax8 = plt.subplot(gs[1, 2])
        ax8.hist(vv_db.flatten(), bins=50, alpha=0.7, color='orange', density=True)
        ax8.set_title('📊 Histogramme VV dB', fontweight='bold')
        ax8.set_xlabel('dB')
        ax8.grid(True, alpha=0.3)
        
        ax9 = plt.subplot(gs[2, 2])
        ax9.hist(vv_final.flatten(), bins=50, alpha=0.7, color='green', density=True)
        ax9.set_title('📊 Histogramme VV Final', fontweight='bold')
        ax9.set_xlabel('[0-1]')
        ax9.grid(True, alpha=0.3)
        
        # === COLONNE 4: STATISTIQUES ===
        ax10 = plt.subplot(gs[:, 3])
        ax10.axis('off')
        
        # Calculer statistiques
        stats_text = f"""
            🔢 STATISTIQUES DÉTAILLÉES

            📄 IMAGES BRUTES:
            VV: Min={vv_brut.min():.3f}, Max={vv_brut.max():.3f}
                Moy={vv_brut.mean():.3f}, Std={vv_brut.std():.3f}
            VH: Min={vh_brut.min():.3f}, Max={vh_brut.max():.3f}
                Moy={vh_brut.mean():.3f}, Std={vh_brut.std():.3f}

            🔄 CONVERSION dB:
            VV: Min={vv_db.min():.1f}, Max={vv_db.max():.1f}
                Moy={vv_db.mean():.1f}, Std={vv_db.std():.1f}
            VH: Min={vh_db.min():.1f}, Max={vh_db.max():.1f}
                Moy={vh_db.mean():.1f}, Std={vh_db.std():.1f}

            ✅ FORMAT SEN12FLOOD:
            VV: Min={vv_final.min():.3f}, Max={vv_final.max():.3f}
                Moy={vv_final.mean():.3f}, Std={vv_final.std():.3f}
            VH: Min={vh_final.min():.3f}, Max={vh_final.max():.3f}
                Moy={vh_final.mean():.3f}, Std={vh_final.std():.3f}

            📐 TRANSFORMATIONS:
            • Taille originale: {h}×{w}
            • Taille finale: 512×512
            • Zoom: {zoom_h:.2f}×{zoom_w:.2f}
            • Normalisation: [-25,+5] dB → [0,1]
            • Format: float32, 2 bandes
        """
        
        ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Ajout informations en bas
        plt.figtext(0.5, 0.02, 
                   "🎯 PIPELINE: Image brute SAR → Conversion dB → Normalisation SEN12FLOOD → Redimensionnement 512×512",
                   ha='center', fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))
        
        print("📊 Affichage de la visualisation complète...")
        plt.show()
        
        # 4. Sauvegarder le fichier SEN12FLOOD
        print("💾 Sauvegarde format SEN12FLOOD...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sen12_filename = f"{region}_sen12flood_{timestamp}.tif"
        sen12_path = os.path.join(output_dir, sen12_filename)
        
        # Assembler et sauvegarder
        sen12_data = np.stack([vv_final, vh_final], axis=0).astype(np.float32)
        
        profile = {
            'driver': 'GTiff',
            'height': 512,
            'width': 512,
            'count': 2,
            'dtype': 'float32',
            'compress': 'lzw'
        }
        
        with rasterio.open(sen12_path, 'w', **profile) as dst:
            dst.write(sen12_data)
        
        print(f"✅ Fichier SEN12FLOOD créé: {os.path.basename(sen12_path)}")
        
    except Exception as e:
        print(f"❌ Erreur visualisation complète: {e}")
        import traceback
        traceback.print_exc()

def visualiser_image_brute(tiff_file: str, region: str):
    """
    Visualise une image TIFF brute
    """
    try:
        
        print("📊 Visualisation image brute...")
        
        with rasterio.open(tiff_file) as src:
            # Lire toutes les bandes
            bands_data = []
            for i in range(src.count):
                band = src.read(i+1)
                bands_data.append(band)
            
            print(f"📊 Image: {src.width}x{src.height}, {src.count} bandes")
            
            # Créer visualisation
            fig, axes = plt.subplots(1, min(len(bands_data), 3), figsize=(15, 5))
            if len(bands_data) == 1:
                axes = [axes]
            
            fig.suptitle(f'📄 Image Brute - {region}', fontsize=14, fontweight='bold')
            
            for i, (band, ax) in enumerate(zip(bands_data[:3], axes)):
                im = ax.imshow(band, cmap='gray')
                ax.set_title(f'Bande {i+1}\nMin: {band.min():.3f}, Max: {band.max():.3f}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"❌ Erreur visualisation brute: {e}")


def adapt_to_senflood12(tiff_file: str, output_dir: str, region: str, is_visualised: bool) -> Optional[str]:
    """
    Adapte l'image satellite brute au format exact SEN12FLOOD:
    - 2 bandes: VV, VH
    - Taille: 512x512 pixels
    - Type: float32 normalisé [0,1]
    - Normalisation: (dB + 25) / 30
    """
    try:
        print("🎯 CONVERSION FORMAT SEN12FLOOD...")
        
        # Variables pour visualisation
        plot_data = {} if is_visualised else None
        
        # 1. Lire l'image brute
        with rasterio.open(tiff_file) as src:
            print(f"📊 Image brute: {src.width}x{src.height}, {src.count} bandes")
            
            # Extraire VV et VH (2 premières bandes)
            vv = src.read(1)
            vh = src.read(2) if src.count >= 2 else vv.copy()  # Si une seule bande, dupliquer
            
            if plot_data:
                plot_data['brut_vv'] = vv.copy()
                plot_data['brut_vh'] = vh.copy()
        
        # 2. Conversion en dB (si nécessaire)
        print("🔄 Conversion dB...")
        
        def to_db(data):
            if data.max() > 10:  # Valeurs linéaires
                return 10 * np.log10(np.maximum(data, 1e-10))
            return data  # Déjà en dB
        
        vv_db = to_db(vv)
        vh_db = to_db(vh)
        
        if plot_data:
            plot_data['db_vv'] = vv_db.copy()
            plot_data['db_vh'] = vh_db.copy()
        
        # 3. Normalisation SEN12FLOOD exacte
        print("📐 Normalisation [0,1]...")
        db_min, db_max = -25.0, 5.0  # Plage SEN12FLOOD
        
        def normalize_exact(data_db):
            clipped = np.clip(data_db, db_min, db_max)
            return (clipped - db_min) / (db_max - db_min)
        
        vv_norm = normalize_exact(vv_db)
        vh_norm = normalize_exact(vh_db)
        
        if plot_data:
            plot_data['norm_vv'] = vv_norm.copy()
            plot_data['norm_vh'] = vh_norm.copy()
        
        # 4. Redimensionnement à 512x512
        print("🔄 Redimensionnement 512x512...")
        h, w = vv_norm.shape
        zoom_h, zoom_w = 512/h, 512/w
        
        vv_512 = zoom(vv_norm, (zoom_h, zoom_w), order=1)
        vh_512 = zoom(vh_norm, (zoom_h, zoom_w), order=1)
        
        if plot_data:
            plot_data['final_vv'] = vv_512.copy()
            plot_data['final_vh'] = vh_512.copy()
        
        # 5. Sauvegarde en format SEN12FLOOD
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sen12_filename = f"{region}_sen12flood_{timestamp}.tif"
        sen12_path = os.path.join(output_dir, sen12_filename)
        
        # Assembler les deux bandes
        sen12_data = np.stack([vv_512, vh_512], axis=0).astype(np.float32)
        
        # Créer GeoTIFF avec profil SEN12FLOOD
        profile = {
            'driver': 'GTiff',
            'height': 512,
            'width': 512,
            'count': 2,
            'dtype': 'float32',
            'compress': 'lzw'
        }
        
        with rasterio.open(sen12_path, 'w', **profile) as dst:
            dst.write(sen12_data)
        
        print(f"✅ Format SEN12FLOOD créé: {os.path.basename(sen12_path)}")
        print(f"📊 Format: 2 bandes, 512x512, float32 [0,1]")
        
        # Visualisation comparative si demandée
        if is_visualised and plot_data:
            afficher_comparaison_sen12flood(plot_data, region, sen12_path)
        
        return sen12_path
        
    except Exception as e:
        print(f"❌ Erreur conversion SEN12FLOOD: {e}")
        return None

def afficher_comparaison_sen12flood(plot_data: dict, region: str, sen12_path: str):
    """
    Affiche une comparaison entre l'image originale et le format SEN12FLOOD
    """
    try:
        print("📊 Début visualisation comparative...")
        print(f"🔍 Vérification des données: {len(plot_data)} éléments")
        
        # Vérifier que toutes les clés nécessaires sont présentes
        required_keys = ['brut_vv', 'brut_vh', 'db_vv', 'db_vh', 'final_vv', 'final_vh']
        missing = [k for k in required_keys if k not in plot_data]
        
        if missing:
            print(f"⚠️ Clés manquantes: {missing}")
            print("⚠️ La visualisation pourrait être incomplète")
        
        # Configuration de la figure
        plt.figure(figsize=(16, 10))
        plt.suptitle(f'🛰️ Format SEN12FLOOD - {region}', fontsize=16, fontweight='bold')
        
        # Créer une grille 2×3 pour VV et VH
        grid = plt.GridSpec(2, 3, wspace=0.2, hspace=0.3)
        
        # Première ligne: VV
        ax1 = plt.subplot(grid[0, 0])
        ax1.imshow(plot_data['brut_vv'], cmap='gray')
        ax1.set_title('VV brut (valeurs originales)')
        ax1.axis('off')
        
        ax2 = plt.subplot(grid[0, 1])
        ax2.imshow(plot_data['db_vv'], cmap='gray', vmin=-25, vmax=5)
        ax2.set_title('VV (dB, -25 à +5)')
        ax2.axis('off')
        
        ax3 = plt.subplot(grid[0, 2])
        im3 = ax3.imshow(plot_data['final_vv'], cmap='gray', vmin=0, vmax=1)
        ax3.set_title('VV normalisé [0-1]')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, orientation='vertical', shrink=0.7)
        
        # Deuxième ligne: VH
        ax4 = plt.subplot(grid[1, 0])
        ax4.imshow(plot_data['brut_vh'], cmap='gray')
        ax4.set_title('VH brut (valeurs originales)')
        ax4.axis('off')
        
        ax5 = plt.subplot(grid[1, 1])
        ax5.imshow(plot_data['db_vh'], cmap='gray', vmin=-25, vmax=5)
        ax5.set_title('VH (dB, -25 à +5)')
        ax5.axis('off')
        
        ax6 = plt.subplot(grid[1, 2])
        im6 = ax6.imshow(plot_data['final_vh'], cmap='gray', vmin=0, vmax=1)
        ax6.set_title('VH normalisé [0-1]')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, orientation='vertical', shrink=0.7)
        
        # Ajout d'informations sur le format SEN12FLOOD
        plt.figtext(0.02, 0.01, 
                   "FORMAT SEN12FLOOD: 2 bandes (VV+VH), 512×512 pixels, float32 [0,1]",
                   fontsize=12, fontweight='bold')
        
        print("📊 Affichage de la visualisation...")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        
    except Exception as e:
        print(f"❌ Erreur visualisation: {e}")
        import traceback
        traceback.print_exc()

def afficher_image_satellite(image_data: np.ndarray, region: str = "Région", titre: Optional[str] = None):
    """
    Affiche l'image satellite retournée par get_image_satellitaire
    
    Paramètres:
    -----------
    image_data : np.ndarray
        Tableau NumPy retourné par get_image_satellitaire
        - Format brut: shape (bands, height, width)
        - Format traité: shape (2, 512, 512) - SEN12FLOOD
    region : str
        Nom de la région pour le titre
    titre : str, optionnel
        Titre personnalisé à afficher
    """
    if image_data is None:
        print("❌ Aucune image à afficher")
        return
    
    # Détecter le type d'image
    is_sen12flood = (image_data.shape[0] == 2 and image_data.shape[1:] == (512, 512))
    
    # Configurer le titre
    if titre is None:
        if is_sen12flood:
            titre = f"🛰️ Image SEN12FLOOD - {region} - Format traité [0,1]"
        else:
            titre = f"🛰️ Image Satellite - {region} - Format brut"
    
    # Configuration de la figure
    if is_sen12flood:
        # Pour SEN12FLOOD (VV, VH)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(titre, fontsize=16, fontweight='bold')
        
        # VV
        vv = image_data[0]
        im1 = axes[0].imshow(vv, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Bande VV [0-1]\nMin: {vv.min():.4f}, Max: {vv.max():.4f}, Moy: {vv.mean():.4f}')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # VH
        vh = image_data[1]
        im2 = axes[1].imshow(vh, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Bande VH [0-1]\nMin: {vh.min():.4f}, Max: {vh.max():.4f}, Moy: {vh.mean():.4f}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Composite VV/VH (rouge=VV, vert=VH)
        composite = np.zeros((image_data.shape[1], image_data.shape[2], 3))
        composite[:,:,0] = vv  # VV en rouge
        composite[:,:,1] = vh  # VH en vert
        im3 = axes[2].imshow(composite)
        axes[2].set_title('Composite VV/VH\nRouge=VV, Vert=VH')
        axes[2].axis('off')
        
        # Informations
        plt.figtext(0.5, 0.01, 
                   "FORMAT SEN12FLOOD: 2 bandes (VV+VH), 512×512 pixels, float32 [0,1]",
                   ha='center', fontsize=12, fontweight='bold')
        
    else:
        # Pour image brute (nombre variable de bandes)
        n_bands = min(3, image_data.shape[0])  # Max 3 bandes
        fig, axes = plt.subplots(1, n_bands, figsize=(6*n_bands, 6))
        fig.suptitle(titre, fontsize=16, fontweight='bold')
        
        # Si une seule bande, axes n'est pas un tableau
        if n_bands == 1:
            axes = [axes]
        
        # Afficher chaque bande
        for i in range(n_bands):
            band = image_data[i]
            im = axes[i].imshow(band, cmap='gray')
            axes[i].set_title(f'Bande {i+1}\nMin: {band.min():.2f}, Max: {band.max():.2f}, Moy: {band.mean():.2f}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Informations
        plt.figtext(0.5, 0.01, 
                   f"FORMAT BRUT: {image_data.shape[0]} bandes, {image_data.shape[1]}×{image_data.shape[2]} pixels",
                   ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Afficher quelques statistiques supplémentaires
    print("\n📊 STATISTIQUES DÉTAILLÉES DE L'IMAGE:")
    print(f"🔹 Shape: {image_data.shape}")
    print(f"🔹 Type: {image_data.dtype}")
    print(f"🔹 Min global: {image_data.min():.4f}")
    print(f"🔹 Max global: {image_data.max():.4f}")
    print(f"🔹 Moyenne globale: {image_data.mean():.4f}")
    print(f"🔹 Écart-type global: {image_data.std():.4f}")

#date p1/06/2025 ai = "2025-06-01"
data = get_image_satellitaire(2.3522, 48.8566, "Paris", True, "brut", is_visualised=False, date_debut="2025-06-01", date_fin="2025-06-02")
print("Image data:", data)
afficher_image_satellite(data, "Lyon")

