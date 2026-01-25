# Digit Recognition using CNN and MLP (MNIST)

This repository contains a Jupyter Notebook project focused on building and refining deep learning models for **digit recognition** using the popular **MNIST dataset**.

This notebook has how different **activation functions**, **optimization algorithms**, and **regularization techniques** affect the performance of neural networks.

---

## Objective

The objective of this assignment is to design and improve a **Convolutional Neural Network (CNN)** and a **Multi-Layer Perceptron (MLP)** to accurately classify handwritten digits (0–9).

The notebook includes experiments that demonstrate why certain activation functions and optimizers lead to better convergence and higher accuracy.

---

## File Included

- **Lab2_12_1_26.ipynb**  
  Complete notebook containing all experiments, training runs, plots, and accuracy comparisons.

---

## Dataset Used

- **MNIST Dataset**
  - 60,000 training images
  - 10,000 testing images
  - Grayscale images of size **28 × 28**
  - 10 digit classes (0–9)

---

## Model Architectures

### Base CNN Architecture

- Input: 28×28 grayscale images  
- Conv2D Layer 1: 32 filters, 3×3 kernel, ReLU  
- Conv2D Layer 2: 64 filters, 3×3 kernel, ReLU  
- MaxPooling: 2×2  
- Dropout: 0.25  
- Dense Layer: 128 neurons, ReLU  
- Output Layer: 10 neurons, Softmax  

---

### Base MLP Architecture

- Flatten Layer (784 input features)
- Dense Layer (256 neurons)
- Batch Normalization
- ReLU Activation
- Dense Layer (128 neurons)
- Batch Normalization
- ReLU Activation
- Output Layer (10 neurons, Softmax)

---

## Tasks Performed

The notebook contains three major experiment categories:

---

## Task 1: Activation Function Challenge

Training CNN using:

- Sigmoid  
- Tanh  
- ReLU  

Observations include:

- Sigmoid suffers from vanishing gradients  
- Tanh converges faster than sigmoid  
- ReLU achieves the best performance and fastest convergence  

---

## Task 2: Optimizer Showdown

Using the best activation function (ReLU), the model is trained with:

- SGD (Stochastic Gradient Descent)
- SGD + Momentum
- Adam Optimizer

Results show Adam converges faster and reaches higher accuracy.

---

## Task 3: Regularization Scenarios

The notebook compares performance under these conditions:

1. Without BatchNorm and Dropout  
2. Without BatchNorm, Dropout = 0.1  
3. With BatchNorm, Dropout = 0.25  

This demonstrates how regularization reduces overfitting and improves generalization.

---

## Outputs Included

The notebook generates:

- Accuracy comparison table  
- Training and validation loss curves  
- Training and validation accuracy curves  
- Final test accuracy for each experiment  

Example results table:

| Experiment | Activation | Optimizer | Epochs | Final Accuracy |
|----------|------------|----------|--------|--------------|
| CNN-1    | ReLU       | Adam     | 10     | ~99%         |
| MLP-1    | ReLU       | SGD      | 20     | ~97%         |
| MLP-2    | ReLU       | Adam     | 15     | ~98%         |

---

## Execution

Execute on Google Colab
