# Week 5: Image Classification using Convolutional Neural Networks (CNN)

This repository contains the complete implementation and documentation for the Week 5 Deep Learning task, demonstrating an end-to-end multi-class image classification pipeline built with TensorFlow and Keras.

---

## Overview

The project classifies clothing categories from the **Fashion-MNIST** dataset. To optimize performance and bypass bandwidth and compute constraints, a sampled subset of 3,000 training images and 1,000 test images was used. The custom 2-block CNN architecture incorporates Batch Normalization and progressive Dropout (25% to 40%) to eliminate overfitting and deliver robust performance within 1 minute of CPU training.

---

## Repository Structure


├── main.py                             # Training and evaluation script
├── generate_report.py                  # Script to build the Word report (.docx)
├── Week_05_Deep_Learning_Report.docx   # Comprehensive project report (.docx)
└── README.md                           # Documentation

---

Key Features
--------------------------
Lightweight Dataset: Efficient ~30 MB footprint using $28 \times 28$ grayscale apparel images.
Network Architecture: 2-Block CNN with $3 \times 3$ convolutions, ReLU activations, and Max Pooling.Regularization: Batch Normalization and progressive Dropout ($0.25 \rightarrow 0.40$) for variance control.
Optimization: Adam optimizer with EarlyStopping and ReduceLROnPlateau callbacks.

Output
---------------
Final Test Accuracy: 89.50%
Final Test Loss: 0.3206
