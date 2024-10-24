# IEEE GRSS Hackathon - Data Driven AI in Remote Sensing
> Project by Solar Hawks – Skibidi

## Project Overview
This project implements a deep learning solution for remote sensing data analysis using PyTorch. The system is designed to detect and segment burn scars from satellite imagery using a transformer-based architecture.

## Initial Setup

### Prerequisites
- AWS Account with appropriate permissions
- Python environment with required packages
- Git

### Environment Setup
1. Access AWS Login Portal using assigned team credentials
2. Create a new private JupyterLab space
3. Clone the repository:
```bash
git clone https://github.com/NASA-IMPACT/rsds-hackathon-24.git
```
4. Install system dependencies:
```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
```
5. Install Python requirements:
```bash
pip install -r requirements.txt
```

## Model Architecture
The project implements a Transformer-based segmentation model with the following key components:

- **Patch Embedding**: Converts input images into embedded patches
- **Multi-Head Attention**: Processes spatial relationships in the data
- **Transformer Blocks**: Deep learning layers for feature extraction
- **Segmentation Head**: Final layer for pixel-wise classification

## Training Configuration Evolution

### Stage 1
- Mixed-precision training (16-bit floating point)
- Increased early stopping patience (5 epochs)
- Increased max epochs to 100
- Batch size increased to 16
- Enabled checkpointing

### Stage 2
- Enhanced logging with version tracking
- Increased early stopping patience to 10 epochs
- Gradient clipping (1.0)
- Changed to UperNetDecoder
- Implemented dice_ce loss function
- OneCycleLR scheduler

### Stage 3
- Mixed-precision training optimization
- Reduced early stopping patience to 3
- Optimized checkpointing
- Updated data augmentation pipeline
- Modified optimizer settings
- Implemented ReduceLROnPlateau scheduler

### Final Configuration
- Batch size: 4
- Max epochs: 100
- Simple augmentations (RandomCrop 224x224, HorizontalFlip)
- FCNDecoder with 256 decoder channels
- Dice loss function
- Adam optimizer with small learning rate (1.3e-5)
- ReduceLROnPlateau scheduler

## Key Features
- Custom BurnScarDataset implementation
- Real-time metrics tracking
- Advanced data augmentation pipeline
- Comprehensive logging system
- Training curve visualization
- Multi-metric evaluation (IoU, F1-Score, Accuracy)

## Performance Metrics
The model's performance is evaluated using:
- Intersection over Union (IoU)
- F1 Score
- Accuracy
- Loss metrics for both training and validation

## File Structure
```
project/
├── training_terratorch.ipynb
├── requirements.txt
└── src/
    ├── dataset.py
    ├── model.py
    └── training.py
```

## Usage
1. Prepare the environment as described in setup
2. Open `training_terratorch.ipynb` in JupyterLab
3. Follow the notebook instructions for training
4. Monitor progress using Weights & Biases and TensorBoard

## Dependencies
- PyTorch
- rasterio
- pandas
- numpy
- matplotlib
- tqdm
- torchmetrics
- einops

## Logging and Monitoring
- Comprehensive logging system with both file and console output
- Real-time metric tracking
- Training curve visualization every 5 epochs
- Model checkpointing based on validation IoU

## Contributing
For contributions, please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments
- SRM Institute of Science and Technology
- IEEE GRSS
- NASA-IMPACT