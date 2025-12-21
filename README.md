# ZOD-SAM

This project provides tools for processing and visualizing the Zenseact Open Dataset (ZOD) using camera and LiDAR data. It includes functionality to convert ZOD bounding box annotations into semantic segmentation masks using Meta's Segment Anything Model (SAM).

## Overview

The Zenseact Open Dataset (ZOD) provides valuable multi-modal data for autonomous driving but lacks dense semantic segmentation annotations, limiting its use for pixel-level perception tasks. We introduce a preprocessing pipeline using the Segment Anything Model (SAM) to convert ZOD's 2D bounding box annotations into dense pixel-level segmentation masks, enabling semantic segmentation training on this dataset for the first time. Due to the imperfect nature of automated mask generation, only 36% of frames passed manual quality control and were included in the final dataset. We present a comprehensive comparison between transformer-based Camera-LiDAR Fusion Transformers (CLFT) and CNN-based DeepLabV3+ architectures for multi-modal semantic segmentation on ZOD across RGB, LiDAR, and fusion modalities under diverse weather conditions. Furthermore, we investigate model specialization techniques to address class imbalance, developing separate modules optimized for large-scale objects (vehicles) and small-scale vulnerable road users (pedestrians, cyclists, traffic signs). The specialized models significantly improve detection of underrepresented safety-critical classes while maintaining overall segmentation accuracy, providing practical insights for deploying multi-modal perception systems in autonomous vehicles. To enable reproducible research, we release the complete open-source implementation of our processing pipeline.

This codebase accompanies the paper "SAM-Enhanced Semantic Segmentation on ZOD: Specialized Models for Vulnerable Road Users".

## Setup Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# ZOD Data full download
```bash
zod download \
  --url="https://www.dropbox.com/scl/fo/q81qqpiqygaeys7mppgoe/AFuqa-QrSkGzHmnkhhpvbBE?dl=0&e=3&rlkey=ocr9n0gq3u083zj8sn1yo1ak6" \
  --output-dir="/media/tom/ml/zod-data" \
  --subset=frames \
  --version=full \
  --images \
  --lidar \
  --annotations \
  --no-radar \
  --no-oxts \
  --infos \
  --no-vehicle-data \
  --no-dnat \
  --num-scans-before=1 \
  --num-scans-after=1 \
  --no-confirm \
  --no-extract
```

## Data Processing Pipeline

This project includes a complete pipeline for processing ZOD data into CLFT (Camera-LiDAR Fusion Transformer) format. Run the scripts in the following order:

### 1. Semantic Segmentation Generation
- **`generate_sam.py`**: Converts ZOD bounding box annotations into semantic segmentation masks using Meta's Segment Anything Model (SAM). Processes camera images to create detailed object masks for vehicles, pedestrians, signs, and cyclists.

### 2. LiDAR Data Processing
- **`generate_lidar_pickle.py`**: Creates CLFT-format LiDAR pickle files containing 3D point clouds with camera projection information for camera-LiDAR fusion training.
- **`generate_lidar_png.py`**: Generates 3D geometric LiDAR projections as 3-channel PNG images, creating visual representations of LiDAR point distributions.

### 3. Annotation Generation
- **`generate_camera_only_annotation.py`**: Produces camera-only segmentation annotations with minimal ignore regions, optimized for camera-based training.
- **`generate_lidar_only_annotation.py`**: Creates LiDAR-native segmentation annotations using SAM guidance, designed for LiDAR-focused model training.

### 4. Dataset Analysis and Splitting
- **`generate_balanced_splits.py`**: Analyzes processed frames and creates balanced train/validation splits ensuring class representation and pixel count equilibrium across weather conditions (day_fair, day_rain, night_fair, night_rain, snow).

## Usage Example

After setting up the environment and extracting ZOD data:

```bash
# Generate semantic segmentation masks
python generate_sam.py

# Process LiDAR data
python generate_lidar_pickle.py
python generate_lidar_png.py

# Create annotations for different modalities
python generate_camera_only_annotation.py
python generate_lidar_only_annotation.py

# Analyze and create balanced splits
python generate_balanced_splits.py
```
