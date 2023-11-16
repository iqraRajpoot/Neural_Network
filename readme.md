# Neural Network for Handwritten Digit Recognition (PyTorch)

This repository contains a convolutional neural network (CNN) implemented in PyTorch for the task of handwritten digit recognition using the MNIST dataset.

## Overview

The neural network architecture consists of two convolutional layers (`conv1` and `conv2`) with ReLU activations, max-pooling, and dropout. This is followed by three fully connected layers (`fc1`, `fc2`, `fc3`) for classification. The network is trained on the MNIST dataset, and training progress is visualized.

## Prerequisites

Make sure you have the following dependencies installed:

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

Install dependencies using:

```bash
pip install torch torchvision numpy matplotlib

```


## Results

Training progress, including loss and accuracy over iterations, is visualized and saved in the `training-loss.png` and `training-acc.png` files. The model parameters are saved after each epoch in files named `epoch-XXXX.pth`.

## Prediction

A random image from the validation dataset is selected, displayed, and its label is predicted using the trained model.
