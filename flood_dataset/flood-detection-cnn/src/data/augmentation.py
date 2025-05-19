import torch
import numpy as np
import random
from torchvision import transforms
from scipy import ndimage
from PIL import Image

def get_sar_augmentation_transforms(prob=0.5):
    """
    Retourne un ensemble de transformations d'augmentation optimisées pour les images SAR
    
    Args:
        prob (float): Probabilité d'appliquer chaque transformation
        
    Returns:
        Compose: Pipeline de transformations PyTorch
    """
    return transforms.Compose([
        # Transformations géométriques (ne modifient pas les valeurs)
        transforms.RandomHorizontalFlip(p=prob),         # Retournement horizontal - utile car l'orientation n'affecte pas les propriétés radar
        transforms.RandomVerticalFlip(p=prob),           # Retournement vertical - idem
        SARRandomRotation(degrees=30, p=prob),           # Rotation aléatoire - simule différentes orientations d'acquisition
        SARRandomCrop(size=(224, 224), p=prob),          # Découpage aléatoire - focus sur différentes parties de l'image
        
        # Transformations d'intensité (spécifiques aux images SAR)
        SARRandomNoiseInjection(std_range=(0.01, 0.05), p=prob),  # Ajout de bruit speckle simulé
        SARRandomIntensityShift(shift_range=(-0.1, 0.1), p=prob),  # Variation mineure d'intensité
        SARRandomContrastAdjustment(gamma_range=(0.8, 1.2), p=prob),  # Ajustement du contraste
        
        # Attention: PAS de ColorJitter car inapproprié pour des images SAR monochromes
        
        # Normalisation finale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Une seule valeur car les images SAR sont monochromes
    ])

# Classes personnalisées pour les transformations spécifiques aux images SAR

class SARRandomRotation:
    """Rotation aléatoire préservant les caractéristiques SAR"""
    def __init__(self, degrees, p=0.5):
        self.degrees = degrees
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            if isinstance(img, torch.Tensor):
                img = img.numpy()
                rotated = ndimage.rotate(img, angle, reshape=False, mode='nearest')
                return torch.from_numpy(rotated)
            else:  # PIL Image
                return img.rotate(angle, resample=Image.NEAREST)
        return img

class SARRandomCrop:
    """Découpage aléatoire optimisé pour les images SAR"""
    def __init__(self, size, p=0.5):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, torch.Tensor):
                _, h, w = img.shape
            else:  # PIL Image
                w, h = img.size
                
            th, tw = self.size
            if h == th and w == tw:
                return img
                
            i = random.randint(0, h - th) if h > th else 0
            j = random.randint(0, w - tw) if w > tw else 0
            
            if isinstance(img, torch.Tensor):
                return img[:, i:i+th, j:j+tw]
            else:  # PIL Image
                return img.crop((j, i, j+tw, i+th))
        return img

class SARRandomNoiseInjection:
    """Ajoute du bruit speckle supplémentaire (multiplicatif)"""
    def __init__(self, std_range=(0.01, 0.05), p=0.5):
        self.std_min, self.std_max = std_range
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(img, torch.Tensor):
                img_np = img.numpy()
                std = random.uniform(self.std_min, self.std_max)
                noise = np.random.normal(1, std, img_np.shape)  # Bruit multiplicatif
                noisy_img = img_np * noise
                return torch.from_numpy(noisy_img)
            else:  # PIL Image
                img_np = np.array(img).astype(np.float32)
                std = random.uniform(self.std_min, self.std_max)
                noise = np.random.normal(1, std, img_np.shape)
                noisy_img = img_np * noise
                noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_img)
        return img

class SARRandomIntensityShift:
    """Ajoute un léger décalage d'intensité, simulant des variations de calibration"""
    def __init__(self, shift_range=(-0.1, 0.1), p=0.5):
        self.shift_min, self.shift_max = shift_range
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            shift = random.uniform(self.shift_min, self.shift_max)
            if isinstance(img, torch.Tensor):
                return img + shift
            else:  # PIL Image
                img_np = np.array(img).astype(np.float32)
                img_np = img_np + shift * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                return Image.fromarray(img_np)
        return img

class SARRandomContrastAdjustment:
    """Ajuste le contraste via une transformation gamma"""
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        self.gamma_min, self.gamma_max = gamma_range
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(self.gamma_min, self.gamma_max)
            if isinstance(img, torch.Tensor):
                # Normaliser entre 0-1 pour appliquer gamma
                min_val, max_val = torch.min(img), torch.max(img)
                img_norm = (img - min_val) / (max_val - min_val + 1e-8)
                img_gamma = img_norm ** gamma
                # Restaurer à l'échelle originale
                return img_gamma * (max_val - min_val) + min_val
            else:  # PIL Image
                img_np = np.array(img).astype(np.float32) / 255.0
                img_gamma = np.power(img_np, gamma)
                img_gamma = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(img_gamma)
        return img

def augment_sar_dataset(images, labels, n_augmentations=3):
    """
    Augmente un ensemble d'images SAR avec leurs étiquettes
    
    Args:
        images (list): Liste d'images SAR (numpy arrays ou tensors)
        labels (list): Liste d'étiquettes correspondantes
        n_augmentations (int): Nombre d'images augmentées à créer pour chaque image originale
        
    Returns:
        tuple: (augmented_images, augmented_labels)
    """
    augmentation_transforms = get_sar_augmentation_transforms()
    augmented_images = []
    augmented_labels = []
    
    # Ajouter les images originales
    augmented_images.extend(images)
    augmented_labels.extend(labels)
    
    # Créer des versions augmentées
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        
        # Convertir en PIL Image si nécessaire
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:  # Image monochrome
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
            else:  # Image multicanal
                pil_img = Image.fromarray((img[0] * 255).astype(np.uint8))  # Premier canal seulement
        else:
            pil_img = img
            
        # Créer plusieurs versions augmentées
        for _ in range(n_augmentations):
            aug_img = augmentation_transforms(pil_img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    return augmented_images, augmented_labels

def visualize_augmentations(image, n_samples=5):
    """
    Visualise différentes versions augmentées d'une même image
    
    Args:
        image: Image source (PIL, numpy array ou tensor)
        n_samples: Nombre de versions augmentées à générer
    """
    import matplotlib.pyplot as plt
    
    # Convertir en PIL Image si nécessaire
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Image monochrome
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
        else:  # Image multicanal
            pil_img = Image.fromarray((image[0] * 255).astype(np.uint8))
    else:
        pil_img = image
    
    # Créer des transformations avec 100% de probabilité pour la visualisation
    augmentation_transforms = get_sar_augmentation_transforms(prob=1.0)
    
    # Créer la figure
    plt.figure(figsize=(15, 4))
    
    # Afficher l'image originale
    plt.subplot(1, n_samples+1, 1)
    plt.imshow(pil_img, cmap='viridis')
    plt.title("Original")
    plt.axis('off')
    
    # Générer et afficher des versions augmentées
    for i in range(n_samples):
        # Appliquer les transformations
        aug_img = augmentation_transforms(pil_img)
        
        # Convertir le tenseur en array pour l'affichage
        if isinstance(aug_img, torch.Tensor):
            aug_img = aug_img.squeeze().numpy()
            
        # Afficher l'image augmentée
        plt.subplot(1, n_samples+1, i+2)
        plt.imshow(aug_img, cmap='viridis')
        plt.title(f"Augmentation {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()