# Disaster-Risk-Monitoring-Using-Satellite-Imagery

**Developed by NVIDIA**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)

## Overview

Floods, wildfires, hurricanes, and other natural disasters can have devastating impacts on communities and infrastructure. This project leverages satellite imagery combined with advanced deep learning segmentation techniques to monitor disaster risks in near real-time. The system automates the detection and delineation of disaster-affected areas, providing actionable insights for emergency responders and decision-makers.

Key aspects of the project include:
- **Automated Image Processing:** Pre-processes satellite images for consistency and optimal model performance.
- **Deep Learning Segmentation:** Uses state-of-the-art neural network architectures (e.g., U-Net with ResNet backbone) to segment affected areas.
- **Scalable Deployment:** Utilizes NVIDIA Triton Inference Server for real-time inference and low-latency deployment.
- **Real-World Validation:** Demonstrated using UNOSAT flood event imagery to ensure practical utility.

---

## Features

- **Data Pre-Processing:**  
  - Normalization of pixel values to standardize inputs.
  - Color space conversion (RGB to BGR) to match model requirements.
  - Data augmentation to improve model generalization.
  - Reshaping of data into required tensor formats.

- **Deep Learning Segmentation Model:**  
  - A U-Net-based architecture optimized for pixel-level segmentation.
  - Utilizes a ResNet backbone for efficient feature extraction.
  - Configurable input dimensions, loss functions (e.g., cross-entropy, Dice loss), and optimizer settings.

- **Efficient Training Pipeline:**  
  - Supports rapid prototyping with adjustable batch sizes and epochs.
  - Integrated monitoring metrics such as IoU and Dice Score for performance evaluation.

- **Scalable Deployment:**  
  - Deployed using NVIDIA Triton Inference Server.
  - Model repository configuration enables efficient real-time inference.
  - Supports simultaneous processing of multiple images for large-scale applications.

- **Real-World Case Study:**  
  - Validated on UNOSAT flood event imagery.
  - Provides quantitative evaluation using established metrics.

---

## Project Structure

```plaintext
Disaster-Risk-Monitoring-Using-Satellite-Imagery/
├── data/
│   ├── images/
│   │   ├── train/         # Training satellite images
│   │   ├── val/           # Validation satellite images
│   │   └── test/          # Test satellite images
│   └── masks/
│       ├── train/         # Segmentation masks for training
│       ├── val/           # Segmentation masks for validation
│       └── test/          # Segmentation masks for testing
├── notebooks/
│   ├── 00_introduction.ipynb
│   ├── 01_disaster_risk_monitoring_systems_and_data_pre-processing.ipynb
│   ├── 02_efficient_model_training.ipynb
│   ├── 03_model_deployment_for_inference.ipynb
│   └── 04_unosat_flood_event_case_study.ipynb
├── models/
│   └── flood_segmentation_model/
│       ├── config.pbtxt # Triton model configuration file
│       └── (model files) # Trained model files
├── README.md
└── requirements.txt
