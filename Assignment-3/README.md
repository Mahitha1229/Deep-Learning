# Deep Learning Training Pipeline with PyTorch

This repository provides a modular training pipeline for image classification using **PyTorch** and **Torchvision**.  
It supports multiple datasets (MNIST, CIFAR10) and models (VGG16, AlexNet, ResNet18/50, MobileNetV2), integrates **Focal Loss**, **early stopping**, and includes **t-SNE visualization** for feature clustering.

---

##  Features
- **Dataset Loader**: Easily switch between MNIST and CIFAR10 with preprocessing and augmentation.
- **Custom Loss Function**: Implements **Focal Loss** to handle class imbalance.
- **Model Zoo**: Supports VGG16, AlexNet, ResNet18, ResNet50, and MobileNetV2.
- **Training Loop**: Includes accuracy tracking, tqdm progress bar, and early stopping.
- **Evaluation**: Computes test accuracy and loss.
- **Visualization**: t-SNE feature clustering for model outputs.

---

## Stucture 
```
├── data/                # Dataset will be downloaded here automatically
├── main.py              # Training + evaluation + visualization script
└── README.md            # Project documentation
```

---

## Usage

1. Run Training
   ```
   dataset = "CIFAR10"   # Options: "MNIST", "CIFAR10"
   model_name = "VGG"    # Options: "VGG", "AlexNet", "ResNet18", "ResNet50", "MobileNet"

   ```
2. Early Stopping
   The best model is automatically saved as : ```best_model.pth```

3. t-SNE Visualization
   - After training, the script generates a 2D scatter plot of features using t-SNE:
   - Each point represents an image.
   - Colors correspond to class labels.
   - Helps visualize feature clustering.
  
  ---

## Execution 

Execute on Google Colab

