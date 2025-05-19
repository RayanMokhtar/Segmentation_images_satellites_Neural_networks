def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the trained model from the specified path.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def preprocess_image(image_path, img_size=(256, 256)):
    """
    Load and preprocess a single image for prediction.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.clip(image, 0, 1)  # Ensure values are within [0, 1]
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    return image

def predict_flood(model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make a prediction on a single image.
    """
    model.to(device)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)  # Add batch dimension
    
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

def main(image_path, model_path):
    """
    Main function to run the prediction.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    
    result = predict_flood(model, image_path, device)
    print(f"Prediction: {result['prediction']}, Flood Probability: {result['flood_probability']:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Flood Detection Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to the image for prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    
    args = parser.parse_args()
    main(args.image, args.model)