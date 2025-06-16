import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from models import Unet  # Updated to use the repository's Unet class

# Load the pre-trained model
MODEL_PATH = "models/model_20230412_015857_48"  # Update with the correct path to the trained model
model = Unet(in_channels=4, out_channels=1)  # Adjusted for 4 input channels
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted normalization for 3-channel images
])

# Function to perform segmentation
def segment_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Convert to RGB for 3 channels
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Add a dummy channel to match the model's expected input
    input_tensor = torch.cat([input_tensor, torch.zeros_like(input_tensor[:, :1, :, :])], dim=1)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        segmentation = torch.sigmoid(output).squeeze().numpy()

    return segmentation

# Function to visualize the segmentation
def visualize_segmentation(image_path, segmentation):
    original_image = Image.open(image_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Flooded Regions")
    plt.imshow(segmentation, cmap="jet")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    # Example usage
    input_image_path = "./data/S1B_IW_GRDH_1SDV_20190209T181740_20190209T181805_014873_01BC28_D12A_corrected_VH.tif"  # Update with the correct path
    segmentation_result = segment_image(input_image_path)
    visualize_segmentation(input_image_path, segmentation_result)
