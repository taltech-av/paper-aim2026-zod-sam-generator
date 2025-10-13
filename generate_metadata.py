#!/usr/bin/env python3
"""
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
OUTPUT_DIR = Path("output_clft_v2")
ANNOTATION_DIR = OUTPUT_DIR / "annotation"
METADATA_DIR = OUTPUT_DIR / "metadata"
DATASET_ROOT = Path("/media/tom/ml/zod-data")

# Class names (updated for merged SAM + LiDAR annotations)
CLASS_NAMES = {
    0: "background",
    1: "vehicle",
    2: "sign",
    3: "cyclist",
    4: "pedestrian",
    5: "ignore"  # LiDAR-only regions
}

print("🔧 Generating metadata for segmentation frames...")
print(f"📁 Annotation dir: {ANNOTATION_DIR}")
print(f"📁 Metadata dir: {METADATA_DIR}")
print(f"📁 ZOD dataset root: {DATASET_ROOT}")

# Create metadata directory
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Find all annotation files
annotation_files = sorted(list(ANNOTATION_DIR.glob("frame_*.png")))
print(f"\n📊 Found {len(annotation_files)} segmentation files")

if len(annotation_files) == 0:
    print("❌ No annotation files found!")
    exit(1)

# Statistics for splits
split_counts = defaultdict(int)
failed_metadata = []

print(f"\n🔄 Processing {len(annotation_files)} frames...")

for anno_file in tqdm(annotation_files, desc="Generating metadata"):
    # Extract frame ID from filename: frame_022228.png -> 022228
    frame_id = anno_file.stem.replace("frame_", "")
    
    try:
        # Load annotation mask
        mask = cv2.imread(str(anno_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            failed_metadata.append(frame_id)
            continue
        
        # Count pixels per class
        unique_classes, counts = np.unique(mask, return_counts=True)
        pixel_counts = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}
        
        # Load ZOD metadata from JSON file
        metadata_path = DATASET_ROOT / "single_frames" / frame_id / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                zod_metadata = json.load(f)
            
            # Extract weather from scraped_weather field
            # Examples: "partly-cloudy-day", "cloudy", "rain", "clear-night", etc.
            scraped_weather = str(zod_metadata.get("scraped_weather", "")).lower()
            
            # Determine if rain or fair based on weather description
            precipitation_keywords = ["rain", "snow", "sleet", "hail", "storm", "drizzle"]
            has_precipitation = any(keyword in scraped_weather for keyword in precipitation_keywords)
            weather = "rain" if has_precipitation else "fair"
            
            # Extract time of day
            # Field: "time_of_day" with values like "day", "night", "dawn", "dusk"
            time_of_day = str(zod_metadata.get("time_of_day", "day")).lower()
            
            # Map to day/night (treat dawn/dusk as day)
            if "night" in time_of_day:
                timeofday = "night"
            else:
                timeofday = "day"
            
            # Store original scraped weather for reference
            scraped_weather_orig = zod_metadata.get("scraped_weather", "unknown")
            
        else:
            # Fallback if metadata file doesn't exist
            weather = "fair"
            timeofday = "day"
            scraped_weather_orig = "unknown"
        
        # Combine into split
        split = f"{timeofday}_{weather}"
        split_counts[split] += 1
        
        # Calculate statistics
        total_pixels = sum(pixel_counts.values())
        foreground_pixels = sum(v for k, v in pixel_counts.items() if k > 0)
        
        # Create metadata
        metadata = {
            "frame_id": frame_id,
            "split": split,
            "weather": weather,
            "timeofday": timeofday,
            "scraped_weather": scraped_weather_orig,
            "image_size": {
                "width": mask.shape[1],
                "height": mask.shape[0]
            },
            "pixel_counts": pixel_counts,
            "pixel_counts_named": {
                CLASS_NAMES.get(k, f"class_{k}"): v 
                for k, v in pixel_counts.items()
            },
            "statistics": {
                "total_pixels": total_pixels,
                "foreground_pixels": foreground_pixels,
                "background_pixels": pixel_counts.get(0, 0),
                "foreground_percentage": float(foreground_pixels / total_pixels * 100) if total_pixels > 0 else 0,
                "num_classes": len([k for k in pixel_counts.keys() if k > 0])
            },
            "class_percentages": {
                CLASS_NAMES.get(k, f"class_{k}"): float(v / foreground_pixels * 100) if foreground_pixels > 0 else 0
                for k, v in pixel_counts.items() if k > 0
            }
        }
        
        # Save individual metadata file
        metadata_file = METADATA_DIR / f"frame_{frame_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        failed_metadata.append(frame_id)
        print(f"\n❌ Error processing {frame_id}: {e}")
        continue

print(f"\n✅ Generated metadata for {len(annotation_files) - len(failed_metadata)} frames")

if failed_metadata:
    print(f"⚠️  Failed: {len(failed_metadata)} frames")

# Print split statistics
print(f"\n📊 Dataset Splits:")
print(f"{'Split':<15} {'Count':>10} {'Percentage':>12}")
print("-" * 40)
total = sum(split_counts.values())
for split in sorted(split_counts.keys()):
    count = split_counts[split]
    pct = (count / total * 100) if total > 0 else 0
    print(f"{split:<15} {count:>10,} {pct:>11.1f}%")
print("-" * 40)
print(f"{'TOTAL':<15} {total:>10,} {100.0:>11.1f}%")

# Save split summary
split_summary = {
    "total_frames": total,
    "splits": {
        split: {
            "count": count,
            "percentage": float(count / total * 100) if total > 0 else 0
        }
        for split, count in split_counts.items()
    }
}

summary_file = OUTPUT_DIR / "split_summary.json"
with open(summary_file, 'w') as f:
    json.dump(split_summary, f, indent=2)

print(f"\n💾 Split summary saved to: {summary_file}")

# Calculate segmentation statistics (total pixels per class across dataset)
print(f"\n📊 Calculating segmentation statistics...")
class_totals = defaultdict(int)
total_pixels_all_frames = 0

for metadata_file in METADATA_DIR.glob("frame_*.json"):
    if metadata_file.name == "split_summary.json":
        continue
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        pixel_counts = metadata.get("pixel_counts", {})
        
        for class_id, count in pixel_counts.items():
            class_totals[int(class_id)] += count
            total_pixels_all_frames += count

# Print segmentation statistics
print(f"\n📈 Segmentation Statistics (Total Pixels per Class):")
print(f"{'Class':<12} {'Name':<12} {'Pixels':>15} {'Percentage':>12}")
print("-" * 55)
for class_id in sorted(class_totals.keys()):
    class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
    pixels = class_totals[class_id]
    percentage = (pixels / total_pixels_all_frames * 100) if total_pixels_all_frames > 0 else 0
    print(f"{class_id:<12} {class_name:<12} {pixels:>15,} {percentage:>11.1f}%")
print("-" * 55)
print(f"{'TOTAL':<12} {'':<12} {total_pixels_all_frames:>15,} {100.0:>11.1f}%")

# Save segmentation statistics
segmentation_stats = {
    "total_pixels": total_pixels_all_frames,
    "classes": {
        class_id: {
            "name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "pixels": class_totals[class_id],
            "percentage": float(class_totals[class_id] / total_pixels_all_frames * 100) if total_pixels_all_frames > 0 else 0
        }
        for class_id in sorted(class_totals.keys())
    }
}

stats_file = OUTPUT_DIR / "segmentation_statistics.json"
with open(stats_file, 'w') as f:
    json.dump(segmentation_stats, f, indent=2)

print(f"\n💾 Segmentation statistics saved to: {stats_file}")

# Generate split files (lists of frame IDs per split)
print(f"\n📝 Generating dataset split files...")

# Collect all frames by split
frames_by_split = defaultdict(list)
all_frames = []

for metadata_file in METADATA_DIR.glob("frame_*.json"):
    if metadata_file.name == "split_summary.json":
        continue
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        split = metadata.get("split", "day_fair")
        frame_id = metadata.get("frame_id")
        if frame_id:
            frames_by_split[split].append(frame_id)
            all_frames.append(frame_id)

# Sort all frames
all_frames = sorted(all_frames)
for split in frames_by_split:
    frames_by_split[split] = sorted(frames_by_split[split])

# Calculate split sizes
total_frames = len(all_frames)
train_size = 13000
valid_size = 4400
test_size = total_frames - train_size - valid_size

print(f"\n📊 Dataset split plan:")
print(f"   Total frames: {total_frames:,}")
print(f"   Train: {train_size:,} ({train_size/total_frames*100:.1f}%)")
print(f"   Validation (early_stop): {valid_size:,} ({valid_size/total_frames*100:.1f}%)")
print(f"   Test: {test_size:,} ({test_size/total_frames*100:.1f}%)")

# Score frames by object diversity and quality
print(f"\n🎯 Scoring frames by object diversity...")
frame_scores = []

for frame_id in tqdm(all_frames, desc="Scoring frames"):
    metadata_file = METADATA_DIR / f"frame_{frame_id}.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
        pixel_counts = metadata.get("pixel_counts", {})
        stats = metadata.get("statistics", {})
        
        # Count pixels for important classes (prioritize rare objects)
        pedestrian_px = pixel_counts.get(4, 0)  # Pedestrian (class 4)
        cyclist_px = pixel_counts.get(3, 0)     # Cyclist (class 3)
        sign_px = pixel_counts.get(2, 0)        # Sign (class 2)
        vehicle_px = pixel_counts.get(1, 0)     # Vehicle (class 1)
        ignore_px = pixel_counts.get(5, 0)      # Ignore (class 5) - LiDAR only
        
        num_classes = stats.get("num_classes", 0)
        fg_percentage = stats.get("foreground_percentage", 0)
        
        # Score: prioritize rare objects (pedestrians, cyclists, signs)
        # Weight: Pedestrian=5x, Cyclist=4x, Sign=3x, Vehicle=1x
        # Ignore class gets minimal weight since it's LiDAR-only regions
        score = (
            (pedestrian_px > 0) * 5000 +  # Has pedestrian (bonus)
            (cyclist_px > 0) * 4000 +      # Has cyclist (bonus)
            (sign_px > 0) * 3000 +         # Has sign (bonus)
            pedestrian_px * 0.1 +          # Pedestrian pixel count
            cyclist_px * 0.08 +            # Cyclist pixel count
            sign_px * 0.05 +               # Sign pixel count
            vehicle_px * 0.01 +            # Vehicle pixel count (common, lower weight)
            ignore_px * 0.005 +            # Ignore regions (minimal weight)
            num_classes * 1000 +           # Diversity bonus
            fg_percentage * 10             # Foreground coverage
        )
        
        frame_scores.append((frame_id, score, metadata.get("split", "day_fair")))

# Sort by score (highest first) for quality-based selection
frame_scores.sort(key=lambda x: -x[1])

print(f"✅ Scored {len(frame_scores)} frames")
print(f"\nTop 5 frames by diversity score:")
for i, (fid, score, split) in enumerate(frame_scores[:5], 1):
    print(f"   {i}. frame_{fid}: score={score:.0f}, split={split}")

# Split into train/valid/test with quality distribution
# Train: Top quality frames for learning
# Valid: High quality frames for validation
# Test: Representative sample across all conditions

train_frames = [fid for fid, score, split in frame_scores[:train_size]]
valid_frames = [fid for fid, score, split in frame_scores[train_size:train_size + valid_size]]
test_frames = [fid for fid, score, split in frame_scores[train_size + valid_size:]]

print(f"\n✅ Selected frames:")
print(f"   Train: Top {len(train_frames):,} quality frames")
print(f"   Valid: Next {len(valid_frames):,} quality frames")
print(f"   Test: Remaining {len(test_frames):,} frames")

# Convert to sets for fast lookup
train_set = set(train_frames)
valid_set = set(valid_frames)
test_set = set(test_frames)

# Separate test frames by category
test_by_category = {
    'day_fair': [],
    'day_rain': [],
    'night_fair': [],
    'night_rain': []
}

for frame_id in test_frames:
    metadata_file = METADATA_DIR / f"frame_{frame_id}.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        split = metadata.get("split", "day_fair")
        test_by_category[split].append(frame_id)

# Format: camera/frame_XXXXXX.png
def format_path(frame_id):
    return f"camera/frame_{frame_id}.png"

# Save all.txt
all_file = OUTPUT_DIR / "all.txt"
with open(all_file, 'w') as f:
    for frame_id in sorted(all_frames):
        f.write(f"{format_path(frame_id)}\n")
print(f"\n✅ all.txt: {len(all_frames):,} frames")

# Save train_all.txt
train_file = OUTPUT_DIR / "train_all.txt"
with open(train_file, 'w') as f:
    for frame_id in sorted(train_frames):
        f.write(f"{format_path(frame_id)}\n")
print(f"✅ train_all.txt: {len(train_frames):,} frames")

# Save early_stop_valid.txt
valid_file = OUTPUT_DIR / "early_stop_valid.txt"
with open(valid_file, 'w') as f:
    for frame_id in sorted(valid_frames):
        f.write(f"{format_path(frame_id)}\n")
print(f"✅ early_stop_valid.txt: {len(valid_frames):,} frames")

# Save test split files
test_files = {
    'test_day_fair.txt': 'day_fair',
    'test_day_rain.txt': 'day_rain',
    'test_night_fair.txt': 'night_fair',
    'test_night_rain.txt': 'night_rain'
}

for filename, category in test_files.items():
    test_file = OUTPUT_DIR / filename
    category_frames = sorted(test_by_category[category])
    with open(test_file, 'w') as f:
        for frame_id in category_frames:
            f.write(f"{format_path(frame_id)}\n")
    print(f"✅ {filename}: {len(category_frames):,} frames")

# Print summary
print(f"\n📊 Test set breakdown:")
for category in sorted(test_by_category.keys()):
    count = len(test_by_category[category])
    pct = (count / len(test_frames) * 100) if test_frames else 0
    print(f"   {category:<15}: {count:>6,} frames ({pct:>5.1f}% of test)")

print(f"\n🎉 Metadata generation complete!")
print(f"\n📁 Output files:")
print(f"   {METADATA_DIR}/ - Individual frame metadata (JSON)")
print(f"   {OUTPUT_DIR}/all.txt - All frames ({len(all_frames):,})")
print(f"   {OUTPUT_DIR}/train_all.txt - Training set ({len(train_frames):,})")
print(f"   {OUTPUT_DIR}/early_stop_valid.txt - Validation set ({len(valid_frames):,})")
print(f"   {OUTPUT_DIR}/test_day_fair.txt - Test day+fair ({len(test_by_category['day_fair']):,})")
print(f"   {OUTPUT_DIR}/test_day_rain.txt - Test day+rain ({len(test_by_category['day_rain']):,})")
print(f"   {OUTPUT_DIR}/test_night_fair.txt - Test night+fair ({len(test_by_category['night_fair']):,})")
print(f"   {OUTPUT_DIR}/test_night_rain.txt - Test night+rain ({len(test_by_category['night_rain']):,})")
