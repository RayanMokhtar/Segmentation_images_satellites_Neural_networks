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
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignorer les pixels non annotés
        
        # Pour le suivi de l'entraînement
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


# Fonction utilitaire pour configurer l'entraînement complet
def train_flood_detection_model(train_loader, val_loader, test_loader=None, 
                                model_params=None, training_params=None):
    """
    Fonction utilitaire pour entraîner un modèle de détection d'inondation
    
    Args:
        train_loader (DataLoader): DataLoader pour les données d'entraînement
        val_loader (DataLoader): DataLoader pour les données de validation
        test_loader (DataLoader): DataLoader pour les données de test
        model_params (dict): Paramètres pour l'initialisation du modèle
        training_params (dict): Paramètres pour l'entraînement
        
    Returns:
        tuple: (model, history, metrics)
    """
    # Paramètres par défaut
    if model_params is None:
        model_params = {}
    
    if training_params is None:
        training_params = {}
    
    # Créer le modèle
    model = FloodDetectionCNN(**model_params)
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Entraîner le modèle
    history = model.train_model(train_loader, val_loader, **training_params)
    
    # Évaluer le modèle si des données de test sont fournies
    metrics = None
    if test_loader is not None:
        metrics = model.evaluate(test_loader)
        
        # Visualiser les résultats
        model.visualize_results(metrics)
    
    # Visualiser l'historique d'entraînement
    model.visualize_training_history(history)
    
    return model, history, metrics