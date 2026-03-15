## Handling Imbalanced Datasets in Image Classification

# Project Overview

This project focuses on handling imbalanced datasets in deep learning-based image classification.

The objective is to analyze how different techniques improve model performance when training data is highly imbalanced.

The implementation includes:

Handling dataset imbalance
Training CNN models
Using Focal Loss
Applying class weighting
Data augmentation for minority classes
Feature visualization using t-SNE and PCA
Error analysis and model comparison

# Datasets Used

CIFAR-10

Flowers Recognition Dataset

# Technologies Used

Python
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn

# Installation

Install the required packages:

pip install imbalanced-learn scikit-learn matplotlib seaborn

# Running the Project

Run the main pipeline script or notebook.

The pipeline automatically performs:

Dataset loading
Data preprocessing
Imbalance handling
Model training
Model evaluation
Feature visualization
Error analysis
Model comparison

# Dataset Loading

Datasets are loaded using TensorFlow and custom dataset loaders.

Two dataset options are implemented:

CIFAR-10 dataset
Flowers image dataset (or simulated flowers dataset)

Images are resized to:

32 × 32 × 3

# Creating Imbalanced Dataset

The project intentionally creates imbalanced datasets.

Example imbalance ratio for CIFAR-10:

5000
2997
1796
1077
645
387
232
139
83
50

# Class Imbalance Handling Techniques

The following techniques are implemented:

Class Weights
Random Oversampling
Random Undersampling
Minority Class Data Augmentation

# Data Augmentation for Minority Classes

Image augmentation is applied to increase minority class samples.

Techniques used:

Rotation
Horizontal Flip
Zoom
Width Shift
Height Shift

# CNN Architectures Used

Custom CNN
EfficientNetB0
ResNet50

# Custom CNN Architecture

The custom CNN architecture contains:

Convolution Layers
Batch Normalization
ReLU Activation
Max Pooling
Dropout Regularization
Fully Connected Layers
Softmax Output Layer

# Loss Functions Used

Categorical Cross Entropy

Focal Loss

# Focal Loss

Focal Loss is designed to address class imbalance by focusing more on hard-to-classify samples.

Gamma parameter used:

gamma = 2.0

# Optimizers Used

Adam

SGD

AdamW

RMSprop

# Model Training

Training configuration:

Epochs: 15

Batch Size: 64

Callbacks used:

EarlyStopping
ReduceLROnPlateau

# Evaluation Metrics

Accuracy

Balanced Accuracy

Macro F1 Score

Weighted F1 Score

Confusion Matrix

Classification Report

# Feature Visualization

Feature embeddings are extracted from intermediate CNN layers.

Dimensionality reduction methods used:

t-SNE

PCA

# Error Analysis

The project includes detailed error analysis:

Misclassified samples detection

Per-class error rates

Top confusion patterns

Prediction confidence analysis

# Visualization

Several visualizations are generated:

Training accuracy and loss curves

Confusion matrices

Feature space projections

Class distribution plots

Confidence analysis histograms

# Experimental Pipeline

Step 1:
Load dataset

Step 2:
Create imbalanced dataset

Step 3:
Apply imbalance handling techniques

Step 4:
Train CNN models

Step 5:
Evaluate models

Step 6:
Visualize feature space

Step 7:
Perform error analysis

Step 8:
Compare model performances

# Comparative Model Analysis

The project compares different models using:

Accuracy

Balanced Accuracy

Macro F1 Score

Weighted F1 Score

# Output Files Generated

Training history plots

Confusion matrices

Feature space visualization plots

Confidence analysis plots

Model comparison CSV files

# Final Results

Results are saved in CSV files:

cifar10_model_comparison.csv

flowers_model_comparison.csv

# Conclusion

This project demonstrates how deep learning models can be improved for imbalanced datasets.

Key observations:

Class imbalance significantly affects classification accuracy.

Focal Loss improves learning for minority classes.

Data augmentation helps increase minority class representation.

Balanced accuracy provides better evaluation for imbalanced datasets.

Feature visualization helps understand learned representations.

