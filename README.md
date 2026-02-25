# ZOD-SAM Generator

This repository is the SAM-based annotation generation codebase accompanying the paper:

> SAM-Enhanced Segmentation on Road Datasets: Balancing Critical Classes in Autonomous Driving
> Toomas Tahves, Mauro Bellone, Junyi Gu, Raivo Sell
> AIM 2026

## Overview

The Zenseact Open Dataset (ZOD) provides rich multi-sensor data from Northern European environments — including challenging rain, snow, and low-light conditions — but includes only 2D bounding-box annotations for traffic participants, lacking the dense pixel-level labels required for semantic segmentation research.

This pipeline converts ZOD bounding boxes into dense segmentation masks using Meta's Segment Anything Model (SAM). Over 100,000 frames were processed; 6,400 were manually inspected and 2,300 high-quality frames (36% acceptance rate) were selected to form the curated pilot dataset. This enables the first dense multi-modal object segmentation benchmark on ZOD, focusing on dynamic traffic participants (vehicles, pedestrians, cyclists) and critical infrastructure (traffic signs).

## Related Repositories

| Resource | Link |
|---|---|
| Multi-modal fusion training framework | https://github.com/taltech-av/paper-aim2026-fusion-trainer |
| Processed datasets (ZOD subset + Iseauto) | https://app.visin.eu/datasets |
| Training logs and visualization dashboards | https://app.visin.eu/projects/sam-zod |

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

## ZOD Data Download
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

### 1. SAM Mask Generation
- **`generate_sam.py`**: Converts ZOD bounding box annotations into dense pixel-level segmentation masks using SAM ViT-H. The pipeline applies class-specific size thresholds (vehicles >30 px, signs/pedestrians/cyclists >15 px), aspect-ratio constraints (<8:1), and IoU-based deduplication (IoU >0.3) before running SAM inference in batches of 16 on 1024 px images. Conflicts between overlapping masks are resolved with a class-priority scheme: pedestrians and cyclists > signs > vehicles. Outputs 768 px dense masks for vehicles, pedestrians, cyclists, and traffic signs.

### 2. LiDAR Data Processing
- **`generate_lidar_pickle.py`**: Creates CLFT-format LiDAR pickle files containing 3D point clouds with camera projection information for camera-LiDAR fusion training.
- **`generate_lidar_png.py`**: Generates 3D geometric LiDAR projections as 3-channel PNG images, creating visual representations of LiDAR point distributions.

### 3. Annotation Generation
- **`generate_camera_only_annotation.py`**: Produces camera-only segmentation annotations from SAM-derived dense masks. Applies minimal ignore regions (narrow ~1% image-height border strip) and removes very small components (<25 px) to improve label reliability while preserving the full camera field of view for supervision.
- **`generate_lidar_only_annotation.py`**: Creates LiDAR-native segmentation annotations by projecting SAM masks onto actual LiDAR returns. Applies distance filtering (up to 90 m) and strict geometric alignment without dilation, preserving the true sensor sampling pattern for LiDAR-only model training.

### 4. Dataset Splitting
- **`generate_splits.py`**: Analyzes processed frames and creates balanced train/validation/test splits (50%/25%/25%) ensuring class representation and pixel count equilibrium across weather conditions (day_fair, day_rain, night_fair, night_rain, snow).

## Usage Example

After setting up the environment and extracting ZOD data:

```bash
# Generate SAM segmentation masks from ZOD bounding boxes
python generate_sam.py

# Process LiDAR data into CLFT-compatible formats
python generate_lidar_pickle.py
python generate_lidar_png.py

# Create modality-specific annotations
python generate_camera_only_annotation.py
python generate_lidar_only_annotation.py

# Generate balanced train/val/test splits
python generate_splits.py
```
