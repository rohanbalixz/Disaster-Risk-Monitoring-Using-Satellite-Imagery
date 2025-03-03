# Disaster-Risk-Monitoring-Using-Satellite-Imagery

**Developed by NVIDIA**

## Overview

Disasters such as floods, wildfires, and hurricanes pose significant threats to lives and infrastructure. This project leverages satellite imagery and advanced deep learning segmentation techniques to monitor disaster risks in near real-time. By automating the detection of affected regions, our system provides timely insights to support emergency responders and decision-makers. Built with NVIDIA technologies—including the TAO Toolkit and Triton Inference Server—this solution represents a robust, scalable approach to disaster risk monitoring.

## Features

- **Data Pre-Processing:**  
  Standardizes satellite imagery through normalization, color conversion, and augmentation to enhance model performance.
  
- **Deep Learning Segmentation Model:**  
  Utilizes a U-Net-based architecture with a ResNet backbone to accurately segment disaster-affected regions from satellite images.

- **Efficient Training Pipeline:**  
  Implements combined loss functions (cross-entropy and Dice loss) and optimized hyperparameters for fast and stable model convergence.

- **Scalable Deployment:**  
  Deployed using NVIDIA Triton Inference Server to achieve low-latency, real-time inference suitable for emergency applications.

- **Real-World Validation:**  
  Validated on actual disaster events (e.g., flood events with UNOSAT imagery) to ensure practical utility and accuracy.

## Project Structure

Disaster-Risk-Monitoring-Using-Satellite-Imagery/ ├── data/ │ ├── images/ │ │ ├── train/ │ │ ├── val/ │ │ └── test/ │ └── masks/ │ ├── train/ │ ├── val/ │ └── test/ ├── notebooks/ │ ├── 00_introduction.ipynb │ ├── 01_disaster_risk_monitoring_systems_and_data_pre-processing.ipynb │ ├── 02_efficient_model_training.ipynb │ ├── 03_model_deployment_for_inference.ipynb │ └── 04_unosat_flood_event_case_study.ipynb ├── models/ │ └── flood_segmentation_model/ │ ├── config.pbtxt │ └── (model files) ├── README.md └── requirements.txt
