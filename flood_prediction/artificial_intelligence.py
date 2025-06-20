import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import time
import logging
import cv2
from datetime import datetime

class FloodDetectionCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, dropout_rate=0.5):
        """
        Initialise le CNN pour la détection d'inondations
        
        Args:
            num_classes (int): Nombre de classes (2 pour inondé/non-inondé)
            input_channels (int): Nombre de canaux d'entrée (2 pour VH et VV)
            dropout_rate (float): Taux de dropout pour la régularisation
        """
        super(FloodDetectionCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # Premier bloc: 2 -> 32 -> 16x16
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            
            # Deuxième bloc: 32 -> 64 -> 8x8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            
            # Troisième bloc: 64 -> 128 -> 4x4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128)
        )
        
        # Calculer la taille de sortie après les convolutions (dépend de la taille d'entrée)
        # Pour une entrée 256x256, après 3 max pooling: 32x32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialisation des poids (Xavier/Glorot)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids du réseau avec l'initialisation de Xavier/Glorot"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Passe avant à travers le réseau
        
        Args:
            x (Tensor): Batch d'images d'entrée de forme [batch_size, channels, height, width]
            
        Returns:
            Tensor: Logits de sortie de forme [batch_size, num_classes]
        """
        # Extraction des caractéristiques
        features = self.feature_extractor(x)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def train_model(self, train_loader, val_loader, num_epochs=30, 
                   learning_rate=0.001, weight_decay=1e-5, 
                   checkpoint_dir='checkpoints', early_stopping_patience=5):
        """
        Entraîne le modèle CNN
        
        Args:
            train_loader (DataLoader): DataLoader pour les données d'entraînement
            val_loader (DataLoader): DataLoader pour les données de validation
            num_epochsrow (int): Nombre d'époques d'entraînement
            learning_rate (float): Taux d'apprentissage pour l'optimiseur
            weight_decay (float): Régularisation L2
            checkpoint_dir (str): Répertoire pour sauvegarder les checkpoints
            early_stopping_patience (int): Nombre d'époques sans amélioration avant d'arrêter
            
        Returns:
            dict: Historique d'entraînement (pertes et métriques)
        """
        # Créer le répertoire de checkpoints s'il n'existe pas
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Définir l'appareil d'exécution (GPU si disponible)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Définir l'optimiseur et la fonction de perte
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  #TODO à changer to BinaryCrossEntropy si binaire

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        # Pour l'early stopping
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_epoch = 0
        
        print(f"Entraînement sur {device} pendant {num_epochs} époques")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Mode entraînement
            self.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # Barre de progression pour l'entraînement
            train_pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs} [Train]")
            
            for inputs, targets in train_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Remise à zéro des gradients
                optimizer.zero_grad()
                
                # Passe avant
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Rétropropagation
                loss.backward()
                optimizer.step()
                
                # Suivi des statistiques
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
                
                # Mise à jour de la barre de progression
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': correct_train / total_train
                })
            
            # Calculer la perte et l'exactitude moyennes pour cette époque
            train_loss = train_loss / total_train
            train_acc = correct_train / total_train
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Mode évaluation
            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            # Désactiver le calcul de gradient pour l'évaluation
            with torch.no_grad():
                # Barre de progression pour la validation
                val_pbar = tqdm(val_loader, desc=f"Époque {epoch+1}/{num_epochs} [Val]")
                
                for inputs, targets in val_pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Passe avant
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Suivi des statistiques
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()
                    
                    # Mise à jour de la barre de progression
                    val_pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': correct_val / total_val
                    })
            
            # Calculer la perte et l'exactitude moyennes pour cette époque
            val_loss = val_loss / total_val
            val_acc = correct_val / total_val
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Afficher un résumé de l'époque
            epoch_time = time.time() - epoch_start
            print(f"Époque {epoch+1}/{num_epochs} - {epoch_time:.1f}s - "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Sauvegarder le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                early_stop_counter = 0
                
                # Sauvegarder le checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"Meilleur modèle sauvegardé à l'époque {epoch+1}")
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= early_stopping_patience:
                print(f"Early stopping après {epoch+1} époques sans amélioration")
                break
        
        # Fin de l'entraînement
        total_time = time.time() - start_time
        print(f"Entraînement terminé en {total_time:.1f}s")
        print(f"Meilleur modèle à l'époque {best_epoch} avec une perte de validation de {best_val_loss:.4f}")
        
        # Restaurer le meilleur modèle
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        return history
    
    def predict(self, image_tensor, device=None):
        """
        Prédit la classe d'une image
        
        Args:
            image_tensor (Tensor): Tensor d'image de forme [channels, height, width]
            device (torch.device): Appareil à utiliser (par défaut: celui du modèle)
            
        Returns:
            tuple: (classe prédite, probabilités)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Mode évaluation
        self.eval()
        
        # Ajouter une dimension de batch si nécessaire
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Déplacer le tensor vers l'appareil approprié
        image_tensor = image_tensor.to(device)
        
        # Désactiver le calcul de gradient pour l'inférence
        with torch.no_grad():
            # Passe avant
            output = self(image_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Obtenir la classe prédite
            _, predicted_class = torch.max(output, 1)
        
        return predicted_class.item(), probabilities.cpu().numpy()
    
    def evaluate(self, test_loader, device=None):
        """
        Évalue le modèle sur un ensemble de test
        
        Args:
            test_loader (DataLoader): DataLoader pour les données de test
            device (torch.device): Appareil à utiliser (par défaut: celui du modèle)
            
        Returns:
            dict: Métriques d'évaluation
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Mode évaluation
        self.eval()
        
        # Pour le suivi des prédictions
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        # Désactiver le calcul de gradient pour l'évaluation
        with torch.no_grad():
            # Barre de progression
            test_pbar = tqdm(test_loader, desc="Évaluation")
            
            for inputs, targets in test_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Passe avant
                outputs = self(inputs)
                probabilities = F.softmax(outputs, dim=1)
                
                # Obtenir les prédictions
                _, predictions = torch.max(outputs, 1)
                
                # Stocker les résultats
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convertir en tableaux numpy
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculer les métriques
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Afficher les résultats
        print("\nRésultats de l'évaluation:")
        print(f"Exactitude: {accuracy:.4f}")
        print(f"Précision: {precision:.4f}")
        print(f"Rappel: {recall:.4f}")
        print(f"Score F1: {f1:.4f}")
        print("\nMatrice de confusion:")
        print(conf_matrix)
        
        # Retourner les métriques
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def visualize_training_history(self, history):
        """
        Visualise l'historique d'entraînement
        
        Args:
            history (dict): Historique d'entraînement (retourné par train_model)
        """
        plt.figure(figsize=(12, 5))
        
        # Graphique de la perte
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Entraînement')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Évolution de la perte')
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique de l'exactitude
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Entraînement')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Évolution de l\'exactitude')
        plt.xlabel('Époque')
        plt.ylabel('Exactitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def visualize_results(self, metrics, class_names=['Non inondé', 'Inondé']):
        """
        Visualise les résultats de l'évaluation
        
        Args:
            metrics (dict): Métriques d'évaluation (retournées par evaluate)
            class_names (list): Noms des classes
        """
        # Extraire les métriques
        conf_matrix = metrics['confusion_matrix']
        
        # Matrice de confusion
        plt.figure(figsize=(10, 8))
        
        # Normalisation pour les pourcentages
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        plt.imshow(conf_matrix_percent, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matrice de confusion (normalisée)')
        plt.colorbar(label='Pourcentage')
        
        # Étiquettes
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Annotations
        thresh = conf_matrix_percent.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, f"{conf_matrix[i, j]} ({conf_matrix_percent[i, j]*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if conf_matrix_percent[i, j] > thresh else "black")
        
        plt.ylabel('Vérité terrain')
        plt.xlabel('Prédiction')
        plt.tight_layout()
        plt.show()
        
        # Métriques par classe
        precision, recall, f1, _ = precision_recall_fscore_support(metrics['targets'], metrics['predictions'])
        
        # Afficher les métriques par classe sous forme de tableau
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(class_names))-0.2, precision, width=0.2, label='Précision')
        plt.bar(np.arange(len(class_names)), recall, width=0.2, label='Rappel')
        plt.bar(np.arange(len(class_names))+0.2, f1, width=0.2, label='F1-Score')
        plt.xticks(np.arange(len(class_names)), class_names)
        plt.ylabel('Score')
        plt.title('Métriques par classe')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def save(self, path):
        """
        Sauvegarde le modèle
        
        Args:
            path (str): Chemin du fichier de sauvegarde
        """
        torch.save(self.state_dict(), path)
        print(f"Modèle sauvegardé à {path}")
    
    @classmethod
    def load(cls, path, **kwargs):
        """
        Charge un modèle sauvegardé
        
        Args:
            path (str): Chemin du fichier de sauvegarde
            **kwargs: Arguments supplémentaires pour l'initialisation du modèle
            
        Returns:
            FloodDetectionCNN: Modèle chargé
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Modèle chargé depuis {path}")
        return model
    

def preprocess_sar_image_in_memory(image, output_dir=None, visualize=True):
    """
    Prétraite une image SAR déjà chargée en mémoire pour la détection d'inondation
    
    Args:
        image (np.ndarray): Image SAR à 2 canaux (VV, VH) de dimensions [height, width, channels]
        output_dir (str): Répertoire pour les visualisations (optionnel)
        visualize (bool): Activer les visualisations du prétraitement
        
    Returns:
        torch.Tensor: Tensor au format [C, H, W] prêt pour le modèle
    """
    print("\n=== PRÉTRAITEMENT DE L'IMAGE SAR EN MÉMOIRE ===")
    
    # Créer le répertoire de sortie si nécessaire
    if visualize and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. VÉRIFICATION DE L'IMAGE
    if image is None:
        raise ValueError("Image non valide (None)")
        
    # Vérifier le format de l'image
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Format d'image non supporté: {type(image)}, numpy.ndarray requis")
    
    height, width, num_channels = image.shape
    print(f"✓ Image en mémoire: {width}×{height} pixels, {num_channels} canal(aux)")
    
    # 2. SÉPARATION DES CANAUX
    if num_channels >= 2:
        # Extraire les canaux VV et VH
        vv_band = image[:, :, 0]  # Premier canal = VV
        vh_band = image[:, :, 1]  # Second canal = VH
        print("✓ Canaux VV et VH extraits")
    else:
        # Si un seul canal, utiliser pour VV et simuler VH
        vv_band = image[:, :, 0]
        vh_band = vv_band * 0.6  # Approximation de VH
        print("⚠️ Un seul canal détecté - VH simulé à partir de VV")
    
    # 3. STATISTIQUES INITIALES
    vv_stats = {
        'min': float(np.nanmin(vv_band)),
        'max': float(np.nanmax(vv_band)),
        'mean': float(np.nanmean(vv_band)),
        'std': float(np.nanstd(vv_band))
    }
    
    vh_stats = {
        'min': float(np.nanmin(vh_band)),
        'max': float(np.nanmax(vh_band)),
        'mean': float(np.nanmean(vh_band)),
        'std': float(np.nanstd(vh_band))
    }
    
    print(f"VV brut - Min: {vv_stats['min']:.4f}, Max: {vv_stats['max']:.4f}, Moy: {vv_stats['mean']:.4f}")
    print(f"VH brut - Min: {vh_stats['min']:.4f}, Max: {vh_stats['max']:.4f}, Moy: {vh_stats['mean']:.4f}")
    
    # 4. PRÉTRAITEMENT VV
    print("\n=== PRÉTRAITEMENT DU CANAL VV ===")
    # 4.1 Correction des valeurs NaN/Inf
    vv_processed = np.nan_to_num(vv_band, nan=0.0, posinf=vv_stats['max'], neginf=vv_stats['min'])
    
    # 4.2 Mise à l'échelle (facteur standard pour VV)
    vv_scaling = 100.0
    vv_scaled = vv_processed / vv_scaling
    print(f"✓ VV - Mise à l'échelle appliquée (÷{vv_scaling})")
    
    # 4.3 Clipping des valeurs extrêmes
    vv_clip_min, vv_clip_max = 0.0, 5.0
    vv_clipped = np.clip(vv_scaled, vv_clip_min, vv_clip_max)
    print(f"✓ VV - Clipping appliqué: [{vv_clip_min}, {vv_clip_max}]")
    
    # 4.4 Redimensionnement
    target_size = (256, 256)
    vv_resized = cv2.resize(vv_clipped, target_size)
    
    # 4.5 Normalisation logarithmique pour VV
    vv_log = np.log10(vv_resized + 1e-5)  # +epsilon pour éviter log(0)
    
    # Normaliser entre 0 et 1
    vv_min = np.min(vv_log)
    vv_max = np.max(vv_log)
    
    if vv_max > vv_min:
        vv_normalized = (vv_log - vv_min) / (vv_max - vv_min)
    else:
        vv_normalized = np.zeros_like(vv_log)
    
    # Convertir de [0,1] à [-1,1]
    vv_final = 2 * vv_normalized - 1
    vv_final = np.clip(vv_final, -1, 1)
    
    print(f"✓ VV - Normalisé - Min: {np.min(vv_final):.4f}, Max: {np.max(vv_final):.4f}")
    
    # 5. PRÉTRAITEMENT VH
    print("\n=== PRÉTRAITEMENT DU CANAL VH ===")
    # 5.1 Correction des valeurs NaN/Inf
    vh_processed = np.nan_to_num(vh_band, nan=0.0, posinf=vh_stats['max'], neginf=vh_stats['min'])
    
    # 5.2 Mise à l'échelle (facteur standard pour VH)
    vh_scaling = 50.0
    vh_scaled = vh_processed / vh_scaling
    print(f"✓ VH - Mise à l'échelle appliquée (÷{vh_scaling})")
    
    # 5.3 Clipping des valeurs extrêmes
    vh_clip_min, vh_clip_max = 0.0, 5.0
    vh_clipped = np.clip(vh_scaled, vh_clip_min, vh_clip_max)
    print(f"✓ VH - Clipping appliqué: [{vh_clip_min}, {vh_clip_max}]")
    
    # 5.4 Redimensionnement
    vh_resized = cv2.resize(vh_clipped, target_size)
    
    # 5.5 Normalisation logarithmique pour VH
    vh_log = np.log10(vh_resized + 1e-5)
    
    # Normaliser entre 0 et 1
    vh_min = np.min(vh_log)
    vh_max = np.max(vh_log)
    
    if vh_max > vh_min:
        vh_normalized = (vh_log - vh_min) / (vh_max - vh_min)
    else:
        vh_normalized = np.zeros_like(vh_log)
    
    # Convertir de [0,1] à [-1,1]
    vh_final = 2 * vh_normalized - 1
    vh_final = np.clip(vh_final, -1, 1)
    
    print(f"✓ VH - Normalisé - Min: {np.min(vh_final):.4f}, Max: {np.max(vh_final):.4f}")
    
    # 6. CRÉATION DU TENSEUR POUR LE MODÈLE
    tensor_image = torch.stack([
        torch.from_numpy(vv_final).float(),
        torch.from_numpy(vh_final).float()
    ])
    
    print(f"✓ Tenseur créé: {tensor_image.shape}")
    
    # 7. VISUALISATIONS
    if visualize:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(16, 10))
        
        # Première ligne: Images originales
        plt.subplot(2, 3, 1)
        plt.imshow(vv_band, cmap='gray')
        plt.title("VV Original")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(vh_band, cmap='gray')
        plt.title("VH Original")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        # Deuxième ligne: Images après scaling et clipping
        plt.subplot(2, 3, 4)
        plt.imshow(vv_clipped, cmap='gray')
        plt.title("VV après scaling/clip")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(vh_clipped, cmap='gray')
        plt.title("VH après scaling/clip")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        # Images finales normalisées
        plt.subplot(2, 3, 3)
        plt.imshow(vv_final, cmap='coolwarm')
        plt.title("VV Normalisé [-1,1]")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(vh_final, cmap='coolwarm')
        plt.title("VH Normalisé [-1,1]")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder la visualisation
        if output_dir:
            viz_path = os.path.join(output_dir, f"preprocessing_viz_{timestamp}.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualisation sauvegardée: {viz_path}")
    
        # 8. TEST DE FILTRAGE DE SPECKLE
        # Option pour ajouter le filtrage speckle si nécessaire
    apply_speckle = False
    if apply_speckle:
        # Utiliser un filtre médian 3x3 pour réduire le bruit speckle
        from scipy import ndimage
        vv_final_filtered = ndimage.median_filter(vv_final, size=3)
        vh_final_filtered = ndimage.median_filter(vh_final, size=3)
        
        # Recréer le tenseur
        tensor_image = torch.stack([
            torch.from_numpy(vv_final_filtered).float(),
            torch.from_numpy(vh_final_filtered).float()
        ])
        print("✓ Filtrage speckle appliqué")
    
    print("\n✅ Prétraitement terminé avec succès!")
    return tensor_image

# def get_combined_prediction(lat, lon):
#     """
#     Exécute les modèles CNN et LSTM pour obtenir une prédiction combinée.
#     """
#     # ... (le reste de la fonction reste inchangé)


# # --- NOUVELLE FONCTION ---
# def get_cnn_predictions_for_period(lat, lon, start_date, end_date):
#     """
#     Exécute le modèle CNN pour chaque jour d'une période donnée pour obtenir des labels.

#     Args:
#         lat (float): Latitude.
#         lon (float): Longitude.
#         start_date (datetime): Date de début.
#         end_date (datetime): Date de fin.

#     Returns:
#         list: Une liste de prédictions (0 ou 1) pour chaque jour de la période.
#     """
#     from .sar_downloader import download_sar_images
#     from .artificial_intelligence import get_cnn_prediction
#     from datetime import timedelta
#     import numpy as np

#     num_days = (end_date - start_date).days + 1
    
#     try:
#         # Le downloader peut retourner des images pour une plage.
#         # On va mapper les images aux dates.
#         image_paths, dates = download_sar_images(lat, lon, start_date, end_date)
        
#         if not image_paths:
#             print("Aucune image SAR trouvée pour la période, retourne des labels nuls.")
#             return [0] * num_days

#         # Créer un dictionnaire de prédictions par date
#         date_to_prediction = {}
#         for img_path, img_date in zip(image_paths, dates):
#             # Utiliser la fonction de prédiction CNN existante
#             # Note: get_cnn_prediction est supposée retourner un dict avec 'cnn_risk', 'cnn_proba'
#             cnn_result = get_cnn_prediction(lat, lon, image_path=img_path)
            
#             # Convertir le risque en label binaire (0 ou 1)
#             # On considère 'faible' comme 0, et tout le reste comme 1.
#             is_flood = 1 if cnn_result.get('cnn_risk', 'faible') != 'faible' else 0
#             date_to_prediction[img_date.strftime('%Y-%m-%d')] = is_flood

#         # Construire la liste finale de labels pour toute la période
#         all_labels = []
#         current_date = start_date
#         while current_date <= end_date:
#             date_str = current_date.strftime('%Y-%m-%d')
#             # Utiliser la prédiction si disponible, sinon 0 (pas d'inondation)
#             all_labels.append(date_to_prediction.get(date_str, 0))
#             current_date += timedelta(days=1)
            
#         return all_labels

#     except Exception as e:
#         print(f"Erreur durant la prédiction CNN pour la période : {e}. Retourne des labels nuls.")
#         return [0] * num_days