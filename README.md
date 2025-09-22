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
