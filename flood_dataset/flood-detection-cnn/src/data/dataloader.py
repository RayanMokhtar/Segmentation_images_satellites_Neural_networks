import os
import ast
import numpy as np
import pandas as pd
import rasterio  
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from .preprocessing import preprocessing_pipeline_image

#commentaire carte : 4974 1012 3975 5903  , 08/26 , 219 

CHEMIN_CSV = "c:/Users/mokht/Desktop/PDS/flood_dataset/satellite_S1_new.csv"
CHEMIN_DATASET = "c:/Users/mokht/Desktop/PDS/flood_dataset/SEN12FLOOD"
FICHIER_SAUVEGARDE_DATASET = "sar_dataset.pkl"

class SARDataset(Dataset):
    VH_SCALING = 50.0
    VV_SCALING = 100.0
    """Dataset Sentinel-1 SAR pour la détection d'inondations"""
    def __init__(self, csv_path=CHEMIN_CSV, 
                 base_dir=None, img_size=(256, 256), transform=None, limit=None):
        """
        Initialise le dataset
        
        Args:
            csv_path (str): Chemin vers le fichier CSV
            base_dir (str): Répertoire de base pour les chemins relatifs d'images
            img_size (tuple): Taille cible pour les images
            transform (callable): Transformations à appliquer
            limit (int): Limite le nombre d'échantillons
        """
        # Charger le CSV
        self.df = pd.read_csv(csv_path)
        print(f"Fichier CSV chargé avec succès: {len(self.df)} lignes")
        
        if limit:
            self.df = self.df.head(limit)
        
        # Trouver le dossier sen12flood si base_dir n'est pas spécifié
        if base_dir is None:
            base_dir = self._find_sen12flood_dir()
        
        self.base_dir = base_dir
        self.img_size = img_size 
        self.transform = transform
        
        # Précharger les chemins
        self.image_paths = []
        self.labels = []
        
        print("Préparation des chemins d'images...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                # Liste de chemins avec différentes polarisations
                paths_str = row['nouveau_chemin']
                if isinstance(paths_str, str):
                    try:
                        file_paths = ast.literal_eval(paths_str)
                    except:
                        file_paths = paths_str.strip("[]'").replace("'", "").split(", ")
                
                # Vérifier qu'il y a au moins 2 chemins (VH et VV)
                if len(file_paths) < 2:
                    continue
                
                # Créer les chemins absolus
                vh_path = os.path.join(base_dir, file_paths[0]) if base_dir else file_paths[0]
                vv_path = os.path.join(base_dir, file_paths[1]) if base_dir else file_paths[1]
                
                # Vérifier l'existence du premier fichier (pour les premiers seulement)
                if idx < 5 and not os.path.exists(vh_path):
                    print(f"Attention: {vh_path} n'existe pas")
                    continue
                
                self.image_paths.append((vh_path, vv_path))
                self.labels.append(row['label'])
                
            except Exception as e:
                if idx < 5:
                    print(f"Erreur à l'index {idx}: {str(e)}")
                continue
        
        print(f"Dataset préparé avec {len(self.image_paths)} échantillons valides")
    
    def _find_sen12flood_dir(self):
        """Cherche le dossier sen12flood dans l'arborescence"""
        # Obtenir le chemin du fichier script actuel
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Remonter jusqu'à 4 niveaux pour trouver sen12flood
        current_dir = script_dir
        for _ in range(4):
            # Vérifier si sen12flood existe à ce niveau
            test_path = os.path.join(os.path.dirname(current_dir), "SEN")
            if os.path.exists(test_path) and os.path.isdir(test_path):
                base_dir = os.path.dirname(current_dir)
                print(f"Dossier sen12flood trouvé à: {test_path}")
                return base_dir
            
            # Remonter d'un niveau
            current_dir = os.path.dirname(current_dir)
        
        # Si toujours pas trouvé, essayer un emplacement spécifique
        specific_path = "C:/Users/mokht/Desktop/PDS/flood_dataset/SEN12FLOOD"
        if os.path.exists(specific_path) and os.path.isdir(specific_path):
            base_dir = "C:/Users/mokht/Desktop/PDS/flood_dataset/"
            print(f"Dossier sen12flood trouvé à l'emplacement spécifique: {specific_path}")
            return base_dir
        
        print("ATTENTION: Impossible de trouver automatiquement le dossier sen12flood")
        return None
    
    def __len__(self):
        """Retourne le nombre d'échantillons dans le dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Charge et prétraite une paire d'images (VH, VV) et son label
        
        Args:
            idx (int): Index de l'échantillon à charger
            
        Returns:
            tuple: (image, label) où image est un tensor de forme (2, height, width)
        """
        vh_path, vv_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Charger et prétraiter les images
        try:
            vh_img = preprocessing_pipeline_image(vh_path)
            vv_img = preprocessing_pipeline_image(vv_path)
            
            print("vh_img.shape", vh_img.shape)
            print("vv_img.shape", vv_img.shape)
            # Empiler les deux canaux

            #remarque comment ça marche 
            """np.stack prend une séquence d'arrays et les combine le long d'un nouvel axe:

            axis=0 (par défaut): Crée un nouvel axe au début → [n_arrays, dim1, dim2, ...]
            axis=1: Crée un nouvel axe après la première dimension → [dim1, n_arrays, dim2, ...]
            axis=2: Crée un nouvel axe après la deuxième dimension → [dim1, dim2, n_arrays, ...]
            """
            img = np.stack([vh_img, vv_img], axis=0)
            
            print("shape img", img.shape)
            # Convertir en tensor PyTorch
            img_tensor = torch.from_numpy(img.astype(np.float32))
            
            print("img_tensor.shape", img_tensor.shape)
            # Appliquer d'autres transformations si spécifiées
            if self.transform:
                img_tensor = self.transform(img_tensor)
            
            return img_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            # En cas d'erreur, retourner un échantillon vide avec un message de log
            print(f"Erreur lors du chargement de l'image {idx}: {str(e)}")
            # Créer une image vide de la bonne taille
            empty_img = torch.zeros((2, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return empty_img, torch.tensor(-1, dtype=torch.long)  # -1 indique un échantillon invalide
    
    def load_and_preprocess_image_sar(self, filepath, scaling_value=1.0):
        """
        Charge et prétraite une image SAR
        
        Args:
            filepath (str): Chemin vers le fichier image
            scaling_value (float): Valeur pour normaliser l'image
        
        Returns:
            np.ndarray: Image prétraitée
        """
        with rasterio.open(filepath) as src:
            band = src.read(1)
            
            # Normalisation (les images SAR ont souvent des valeurs très variables)
            band = band / scaling_value
            
            # Limiter les valeurs pour éviter les valeurs extrêmes
            band = np.clip(band, 0, 5)
            
            # Redimensionner l'image
            band = cv2.resize(band, self.img_size)
            
            return band
    
    def save(self, output_file="sar_dataset.pkl"):
        """
        Sauvegarde le dataset pour un chargement rapide ultérieur
        
        Args:
            output_file (str): Chemin du fichier de sortie
        """
        print(f"Sauvegarde du dataset de {len(self)} échantillons...")
        with open(output_file, 'wb') as f:
            pickle.dump({
                'image_paths': self.image_paths,
                'labels': self.labels,
                'img_size': self.img_size,
                'base_dir': self.base_dir
            }, f)
        print(f"Dataset sauvegardé dans {output_file}")
    
    @classmethod
    def load(cls, file_path="sar_dataset.pkl", transform=None):
        """
        Charge un dataset sauvegardé précédemment
        
        Args:
        x        file_path (str): Chemin du fichier
            transform (callable): Transformations à appliquer
            
        Returns:
            SARDataset: Le dataset chargé
        """
        print(f"Chargement du dataset depuis {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Créer un dataset vide
        dataset = cls.__new__(cls)
        
        # Initialiser les attributs
        dataset.image_paths = data['image_paths']
        dataset.labels = data['labels']
        dataset.img_size = data['img_size']
        dataset.base_dir = data['base_dir']
        dataset.transform = transform
        dataset.df = None  # Pas besoin du DataFrame original
        
        print(f"Dataset chargé: {len(dataset)} échantillons")
        return dataset
    
    @classmethod
    def create_dataloader(cls , csv_path=CHEMIN_CSV, 
                         base_dir=None, img_size=(256, 256), limit=None, 
                         batch_size=32, num_workers=4, shuffle=True,
                         use_saved_dataset=False, saved_dataset_path=FICHIER_SAUVEGARDE_DATASET,
                         save_after_loading=False):
        #cls correspond ici à la Classe dataset , qu'on trouve régulièrement dans les @classmethod
        """
        Crée un DataLoader pour les images SAR
        
        Args:
            csv_path (str): Chemin vers le fichier CSV
            base_dir (str): Répertoire de base pour les chemins relatifs d'images
            img_size (tuple): Taille cible pour les images
            limit (int): Limite le nombre d'échantillons
            batch_size (int): Taille des lots pour le DataLoader
            num_workers (int): Nombre de processus pour charger les données
            shuffle (bool): Mélanger ou non les données
            use_saved_dataset (bool): Utiliser un dataset sauvegardé
            saved_dataset_path (str): Chemin du dataset sauvegardé
            save_after_loading (bool): Sauvegarder le dataset après chargement
            
        Returns:
            tuple: (data_loader, dataset) 
        """
        # Charger ou créer le dataset
        if use_saved_dataset and os.path.exists(saved_dataset_path):
            dataset = cls.load(saved_dataset_path)
        else:
            dataset = cls(csv_path, base_dir, img_size, limit=limit)
            if save_after_loading:
                dataset.save(saved_dataset_path)
        
        # Créer le DataLoader
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return data_loader, dataset
    


    