# THYROX - Thyroid Detection, Classification and Lymph Detection

## Overview
THYROX is an AI-based system for automatically detecting thyroid nodules and suspicious lymph nodes from ultrasound images, classifying nodules as benign or malignant, and providing risk scores to assist radiologists.

## Tech Stack
- **Python** - Core implementation language
- **PyTorch** - Deep learning framework for CNN models
- **YOLOv5** - Object detection for nodule/lymph node localization
- **ResNet-50** - Transfer learning for benign/malignant classification
- **XGBoost** - Ensemble classifier for final prediction
- **OpenCV** - Image preprocessing and visualization
- **scikit-image** - Geometric feature extraction
- **Flask** - Web API for deployment

## Project Structure