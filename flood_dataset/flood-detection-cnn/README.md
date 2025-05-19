# Flood Detection CNN Project

This project implements a convolutional neural network (CNN) architecture for flood detection using Synthetic Aperture Radar (SAR) images. The model integrates advanced image processing techniques and computes the Normalized Difference Flood Index (NDFI) to enhance flood detection accuracy.

## Project Structure

```
flood-detection-cnn
├── src
│   ├── data
│   │   ├── dataset.py        # Handles dataset loading and management
│   │   ├── augmentation.py    # Contains data augmentation techniques
│   │   └── preprocessing.py    # Includes image preprocessing functions
│   ├── models
│   │   ├── cnn.py             # Defines the CNN architecture
│   │   ├── resnet.py          # Implements the ResNet architecture
│   │   └── unet.py            # Defines the U-Net architecture for segmentation
│   ├── indices
│   │   └── ndfi.py            # Functions to compute the Normalized Difference Flood Index
│   ├── utils
│   │   ├── visualization.py    # Functions for visualizing predictions and data samples
│   │   └── metrics.py         # Functions to calculate evaluation metrics
│   ├── training
│   │   ├── trainer.py         # Manages the training loop
│   │   └── callbacks.py       # Defines training callbacks
│   └── inference
│       └── predict.py         # Functions for making predictions on new data
├── notebooks
│   ├── exploratory_analysis.ipynb  # Exploratory data analysis
│   ├── model_training.ipynb        # Model training notebook
│   └── results_visualization.ipynb  # Visualizes results and metrics
├── configs
│   ├── model_config.yaml          # Model architecture configuration
│   └── training_config.yaml       # Training process configuration
├── scripts
│   ├── train.py                   # Entry point for training the model
│   ├── evaluate.py                # Evaluates the trained model
│   └── predict.py                 # Makes predictions on new images
├── requirements.txt               # Lists project dependencies
├── setup.py                       # Project packaging and dependency management
└── README.md                      # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd flood-detection-cnn
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your SAR images in the appropriate directory and update the dataset paths in `src/data/dataset.py`.

2. **Training the Model**: Use the `scripts/train.py` script to start training the model. You can adjust hyperparameters in `configs/training_config.yaml`.

3. **Evaluating the Model**: After training, evaluate the model using `scripts/evaluate.py` to see the performance metrics.

4. **Making Predictions**: Use `scripts/predict.py` to make predictions on new SAR images.

5. **Exploratory Analysis**: Utilize the Jupyter notebooks in the `notebooks` directory for exploratory data analysis and results visualization.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.