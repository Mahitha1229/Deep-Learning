## Comparative Analysis of Different CNN Architectures

# Overview

This assignment implements and compares different Convolutional Neural Network (CNN) architectures using PyTorch.

The goal is to analyze how different architectures, loss functions, and optimization strategies affect classification performance.

Datasets used:
MNIST
CIFAR10

The experiments study:
Network depth
Loss functions
Optimizers
Feature representations

# Technologies Used

Python
PyTorch
Torchvision
NumPy
Matplotlib
Scikit-learn
tqdm

# Dataset Preprocessing

Datasets are loaded using Torchvision.

Preprocessing steps include:
Image normalization
Image resizing
Data augmentation

# Data Augmentation Techniques

RandomHorizontalFlip
RandomRotation
ColorJitter

# MNIST Processing

Since MNIST images are grayscale, they are converted to 3-channel images
so that they can be used with CNN architectures originally designed for RGB images.

# Part 1 — CNN Architecture Comparison

Different CNN architectures are trained on the CIFAR-10 dataset.

Models Tested:

LeNet5
AlexNet
MobileNet
ResNet50
ResNet101
EfficientNet
VGG16
InceptionV3

# Evaluation Metrics

The following metrics are recorded during experiments:

Number of parameters
Training time
Best test accuracy

# Example Result Format

Model           Parameters      Time(s)      Test Accuracy
----------------------------------------------------------
LeNet           XXXXX           XX           XX%
AlexNet         XXXXX           XX           XX%
MobileNet       XXXXX           XX           XX%
ResNet50        XXXXX           XX           XX%
EfficientNet    XXXXX           XX           XX%

# Observations from Architecture Comparison

Deeper architectures generally achieve higher accuracy.

Lightweight models such as MobileNet provide faster training with reduced computational cost.

EfficientNet provides a good balance between performance and efficiency.

# Part 2 — Loss Function and Optimization Analysis

This section analyzes how different loss functions and optimizers affect CNN performance.

# Experiment 1 — Architecture Performance with Focal Loss

Datasets Used:
MNIST
CIFAR10

Models Tested:
AlexNet
ResNet50
EfficientNet
VGG

Loss Function:
Focal Loss

Optimizer:
Adam

Purpose:
Evaluate how different CNN architectures perform when trained with Focal Loss.

# Experiment 2 — Loss Function Comparison

Model Used:
VGG

Dataset:
CIFAR10

Optimizer:
Adam

Loss Functions Compared:

BCE (Binary Cross Entropy)
ArcFace Loss

Purpose:
Study how different loss functions influence classification accuracy.

# Experiment 3 — Optimizer Comparison

Model:
VGG

Dataset:
CIFAR10

Loss Function:
BCE Loss

Optimizers Compared:

SGD
RMSprop

Optimizers Compared:

SGD
RMSprop

Loss Functions Compared:

BCE
Focal Loss
ArcFace

Better clustering indicates stronger learned feature representations.

# Findings

Deeper CNN architectures achieve higher classification accuracy.

Lightweight architectures provide faster computation but slightly lower accuracy.

Loss functions significantly influence training performance.

SGD optimizer shows strong convergence for CNN models.

t-SNE visualization helps understand how neural networks cluster different classes.

This project demonstrates how CNN architecture, loss functions, and optimization strategies impact image classification performance.

- Architecture depth improves model accuracy
- Training strategies affect convergence
- Loss functions influence feature learning
- Visualization techniques help interpret model behavior
