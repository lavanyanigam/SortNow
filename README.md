# SortNow - Your Smart Waste Classification Assistant

A custom YOLO based object detection model to classify waste into 6 categories for better garbage segregation and environmental impact.

## About The Project

SortNow helps identify and classify different types of waste in real-time using a custom-built YOLO architecture. The goal is to make waste sorting easier and more accurate, ultimately reducing contamination in recycling streams.

## Classes Detected:
- Biodegradable waste
- Cardboard
- Glass
- Metal
- Paper  
- Plastic

## Dataset

- **Training images:** 7,260
- **Validation images:** 3,114
- **Format:** YOLO format (normalized bounding boxes)
- **Image size:** 448×448

## Model Architecture

Built a simplified 5 layer CNN backbone inspired by YOLO:
- 4 MaxPooling stages reducing 448×448 to 14×14 grid
- 512 channels in final conv layers
- Grid size: 14×14 
- 2 bounding boxes per cell 
- Output: 14×14×16 predictions

## Training Details

- **Framework:** PyTorch
- **Optimizer:** Adam 
- **Learning rate:** 1e-4 
- **Batch size:** 16
- **Epochs:** 100 
- **Hardware:** Kaggle GPU P100

Trained without augmentations

## Results

Early stopping triggered at epoch 64
Best model at epoch 54 with validation loss: 176.3113
Final training loss: 65.5796
Final validation loss: 177.0820

![Training History](yolo_training_plot.png)



## Development History

### Single Object Detection (Initial Attempt)
- Built basic CNN for detecting one waste item per image
- Limitations: Couldn't handle multiple objects in frame

### Multi-Object Detection (Final Version)
- Implemented full YOLO architecture with 14×14 grid
- Supports multiple waste items per image with bounding boxes
