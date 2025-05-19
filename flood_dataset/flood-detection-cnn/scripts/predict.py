import torch
import numpy as np
import cv2
import os
from torchvision import transforms
from src.models.cnn import CNN  # Import your CNN model
from src.indices.ndfi import compute_ndfi  # Import NDFI computation function
from src.utils.visualization import visualize_prediction  # Import visualization function

def load_image(image_path, img_size=(256, 256)):
    """
    Load and preprocess a single image for prediction.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    return image

def predict(model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make a prediction on a single image.
    """
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    image = load_image(image_path)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        flood_probability = probabilities[0, 1].item()  # Probability of class 1 (flood)
    
    return {
        'predicted_class': predicted_class,  # 0: No Flood, 1: Flood
        'flood_probability': flood_probability,
        'prediction': 'Flood' if predicted_class == 1 else 'No Flood'
    }

if __name__ == "__main__":
    model = CNN()  # Initialize your model
    model.load_state_dict(torch.load('path_to_your_model_weights.pth'))  # Load trained model weights
    model.eval()
    
    image_path = 'path_to_your_image.jpg'  # Path to the image you want to predict
    result = predict(model, image_path)
    
    print(f"Prediction: {result['prediction']}, Flood Probability: {result['flood_probability']:.2f}")