# ZOD CLFT

This project provides tools for processing and visualizing the Zenseact Open Dataset (ZOD) using camera and LiDAR data.

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

## Quick Analysis Commands

Run weather and time analysis:
```bash
python3 analyze_time_weather_combinations.py --data_path /media/tom/ml/zod-data/single_frames --workers 8
```

Run object class distribution analysis:
```bash
python3 analyze_class_distributions.py --data_path /media/tom/ml/zod-data/single_frames --workers 8
```

## Full Dataset Conversion

Camera only:
```bash
source venv/bin/activate
python convert_zod_full_dataset.py \
  --dataset-root /media/tom/ml/zod-data \
  --version full \
  --output-root /media/tom/ml/projects/clft-zod/output_clft_full \
  --components camera \
  --workers 20 \
  --batch-size 256 \
  --progress-log /media/tom/ml/projects/clft-zod/output_clft_full/progress_camera.log \
  --resume-progress
```

Lidar only:
```bash
source venv/bin/activate
python convert_zod_full_dataset.py \
  --dataset-root /media/tom/ml/zod-data \
  --version full \
  --output-root /media/tom/ml/projects/clft-zod/output_clft_full \
  --components lidar \
  --workers 20 \
  --batch-size 256 \
  --progress-log /media/tom/ml/projects/clft-zod/output_clft_full/progress_lidar.log \
  --resume-progress
```

LiDAR overlays only (reuse existing camera PNGs and LiDAR pickles):
```bash
source venv/bin/activate
python convert_zod_full_dataset.py \
  --dataset-root /media/tom/ml/zod-data \
  --version full \
  --output-root /media/tom/ml/projects/clft-zod/output_clft_full \
  --lidar-overlay-only \
  --workers 20 \
  --batch-size 256 \
  --progress-log /media/tom/ml/projects/clft-zod/output_clft_full/progress_lidar_overlay.log \
  --resume-progress
```

This mode expects matching `camera/frame_<id>.png` and `lidar/frame_<id>.pkl` files inside the output directory. Missing artefacts are logged and skipped without deleting existing files.

SAM only
```bash
source venv/bin/activate
python convert_zod_full_dataset.py \
  --dataset-root /media/tom/ml/zod-data \
  --version full \
  --output-root /media/tom/ml/projects/clft-zod/output_clft_full \
  --sam-only \
  --workers 18 \
  --sam-model-type vit_h \
  --batch-size 256 \
  --progress-log /media/tom/ml/projects/clft-zod/output_clft_full/progress_sam.log \
  --resume-progress
```
Speed tip: add `--sam-model-type vit_b` to switch to the lighter ViT-B backbone. When the default checkpoint path is used, the converter will automatically look for `models/sam_vit_b_01ec64.pth` and download it if missing.

SAM overlays only (reuse existing camera + SAM mask artefacts):
```bash
source venv/bin/activate
python convert_zod_full_dataset.py \
  --dataset-root /media/tom/ml/zod-data \
  --version full \
  --output-root /media/tom/ml/projects/clft-zod/output_clft_full \
  --sam-overlay-only \
  --workers 18 \
  --batch-size 256 \
  --progress-log /media/tom/ml/projects/clft-zod/output_clft_full/progress_sam_overlay.log \
  --resume-progress
```
Requires matching `camera/frame_<id>.png` and `annotation/frame_<id>.png` files in the output directory. Overlays are written to `visualizations/frame_<id>_sam_overlay.png` without rerunning SAM.

