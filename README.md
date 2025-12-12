# ZOD CLFT

This project provides tools for processing and visualizing the Zenseact Open Dataset (ZOD) using camera and LiDAR data. It includes functionality to convert ZOD bounding box annotations into semantic segmentation masks using Meta's Segment Anything Model (SAM).

## Prerequisites

- Docker
- ZOD dataset files (mini or full version)

## Setup and Installation
### Start the Application
CPU-only (lightweight):
```
docker compose run --rm --service-ports python-lite bash
```

GPU-enabled (CUDA):
```  
docker compose run --rm --service-ports python-cuda bash
```

### Data Preparation
Copy the unpacked ZOD data files to the `/data` folder:
- `drives_mini.tar.gz`
- `frames_mini.tar.gz`
- `sequences_mini.tar.gz`

Run command to extract data:
```
sh scripts/extract_zod_data.sh
```

### Start jupyter
```
scripts/jupyter_startup.sh
```

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

# Full download
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
- **`generate_fusion_annotation.py`**: Generates fusion annotations combining camera and LiDAR data for multi-modal training scenarios.

### 4. Dataset Analysis and Splitting
- **`generate_balanced_splits.py`**: Analyzes processed frames and creates balanced train/validation splits ensuring class representation and pixel count equilibrium across weather conditions (day_fair, day_rain, night_fair, night_rain, snow).

## Additional Resources

- **`notebooks/`**: Jupyter notebooks for data exploration, visualization, and analysis including weather/time analysis and class distribution studies.
- **`paper/`**: Research paper materials including performance comparisons (CLFT vs DeepLab), analysis scripts, and visualization diagrams.
- **`scripts/`**: Utility scripts for data extraction (`extract_zod_data.sh`) and Jupyter environment setup (`jupyter_startup.sh`).

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
python generate_fusion_annotation.py

# Analyze and create balanced splits
python generate_balanced_splits.py
```

## Key Files and Models

- **`good_framest.txt`**: Pre-filtered list of high-quality frames for processing (2300+ frames)
- **`models/`**: Pre-trained models including:
  - SAM models (ViT-B, ViT-L, ViT-H) for semantic segmentation
  - YOLO models (YOLOv8, YOLO11) for object detection
- **`requirements.txt`**: Python dependencies for the processing pipeline
- **`compose.yml`**: Docker Compose configuration for CPU/GPU environments
