import matplotlib.pyplot as plt
import numpy as np
import rasterio

def plot_predictions(images, true_labels, predicted_labels, num_samples=5):
    """
    Plot the images along with their true and predicted labels.
    
    Parameters:
    - images: A batch of images to visualize.
    - true_labels: The true labels corresponding to the images.
    - predicted_labels: The predicted labels from the model.
    - num_samples: Number of samples to display.
    """
    

    indices = np.random.choice(range(len(images)), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(images[idx].numpy().transpose(1, 2, 0), cmap='gray')
        axes[i, 0].set_title(f"True: {'Flood' if true_labels[idx] == 1 else 'No Flood'}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(images[idx].numpy().transpose(1, 2, 0), cmap='gray')
        axes[i, 1].set_title(f"Pred: {'Flood' if predicted_labels[idx] == 1 else 'No Flood'}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_ndfi(ndfi_values, threshold=0.5):
    """
    Visualize the Normalized Difference Flood Index (NDFI) values.
    
    Parameters:
    - ndfi_values: A 2D array of NDFI values.
    - threshold: The threshold for flood detection.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.imshow(ndfi_values, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDFI Value')
    plt.title('Normalized Difference Flood Index (NDFI)')
    plt.contour(ndfi_values, levels=[threshold], colors='blue', linewidths=2)
    plt.title('NDFI with Flood Detection Threshold')
    plt.axis('off')
    plt.show()

def obtenir_taille_originale(chemin_image):
    """Affiche la taille originale d'une image raster"""
    with rasterio.open(chemin_image) as src:
        height = src.height
        width = src.width
        return height, width