import os
import yaml
import torch
import numpy as np
from src.data.dataset import Dataset
from src.data.augmentation import augment_data
from src.data.preprocessing import preprocess_images
from src.models.cnn import CNN
from src.models.resnet import ResNet
from src.models.unet import UNet
from src.training.trainer import Trainer
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_training_history

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configurations
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    dataset = Dataset(training_config['data_path'])
    train_loader, val_loader = dataset.get_data_loaders(batch_size=training_config['batch_size'])

    # Initialize model
    if model_config['architecture'] == 'cnn':
        model = CNN(**model_config['params']).to(device)
    elif model_config['architecture'] == 'resnet':
        model = ResNet(**model_config['params']).to(device)
    elif model_config['architecture'] == 'unet':
        model = UNet(**model_config['params']).to(device)
    else:
        raise ValueError("Unsupported model architecture")

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, training_config)

    # Train the model
    history = trainer.train()

    # Evaluate the model
    metrics = calculate_metrics(trainer.model, val_loader, device)
    print(f"Validation Metrics: {metrics}")

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()