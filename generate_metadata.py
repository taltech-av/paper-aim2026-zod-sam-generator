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
    1: "ignore",      # LiDAR-only regions
    2: "vehicle",     # From SAM
    3: "sign",        # From SAM
    4: "cyclist",     # From SAM
    5: "pedestrian"   # From SAM
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

# Calculate comprehensive dataset statistics
print(f"\n📊 Calculating comprehensive dataset statistics...")
class_totals = defaultdict(int)
total_pixels_all_frames = 0

# Additional statistics
frame_counts_per_class = defaultdict(int)  # How many frames have each class
pixel_sums_per_class = defaultdict(list)   # Pixel counts per frame for each class
class_cooccurrence = defaultdict(lambda: defaultdict(int))  # Which classes appear together
quality_metrics = {
    'frames_with_multiple_classes': 0,
    'frames_with_rare_objects': 0,  # Pedestrians or cyclists
    'total_foreground_pixels': 0,
    'avg_classes_per_frame': 0,
    'max_classes_in_frame': 0
}

all_frame_data = []

for metadata_file in METADATA_DIR.glob("frame_*.json"):
    if metadata_file.name == "split_summary.json":
        continue

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        pixel_counts = metadata.get("pixel_counts", {})
        stats = metadata.get("statistics", {})
        split = metadata.get("split", "day_fair")
        frame_id = metadata.get("frame_id")

        # Basic pixel counting
        frame_pixels = 0
        classes_in_frame = []

        for class_id, count in pixel_counts.items():
            class_id = int(class_id)
            class_totals[class_id] += count
            total_pixels_all_frames += count
            frame_pixels += count

            if count > 0:
                classes_in_frame.append(class_id)
                frame_counts_per_class[class_id] += 1
                pixel_sums_per_class[class_id].append(count)

        # Class co-occurrence
        for i, class_a in enumerate(classes_in_frame):
            for class_b in classes_in_frame[i+1:]:
                class_cooccurrence[min(class_a, class_b)][max(class_a, class_b)] += 1

        # Quality metrics
        num_classes = len(classes_in_frame)
        if num_classes > 1:
            quality_metrics['frames_with_multiple_classes'] += 1
        if any(cls in [3, 4] for cls in classes_in_frame):  # Cyclist or pedestrian
            quality_metrics['frames_with_rare_objects'] += 1

        fg_pixels = sum(pixel_counts.get(str(cls), 0) for cls in [1, 2, 3, 4, 5])  # Exclude background
        quality_metrics['total_foreground_pixels'] += fg_pixels
        quality_metrics['avg_classes_per_frame'] += num_classes
        quality_metrics['max_classes_in_frame'] = max(quality_metrics['max_classes_in_frame'], num_classes)

        all_frame_data.append({
            'frame_id': frame_id,
            'split': split,
            'classes_present': classes_in_frame,
            'num_classes': num_classes,
            'foreground_pixels': fg_pixels,
            'total_pixels': frame_pixels
        })

# Calculate averages
total_frames = len(all_frame_data)
if total_frames > 0:
    quality_metrics['avg_classes_per_frame'] /= total_frames

# Calculate class averages and statistics
class_statistics = {}
for class_id in sorted(class_totals.keys()):
    pixels = pixel_sums_per_class[class_id]
    class_statistics[class_id] = {
        "name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "total_pixels": class_totals[class_id],
        "percentage": float(class_totals[class_id] / total_pixels_all_frames * 100) if total_pixels_all_frames > 0 else 0,
        "frames_with_class": frame_counts_per_class[class_id],
        "frames_percentage": float(frame_counts_per_class[class_id] / total_frames * 100) if total_frames > 0 else 0,
        "avg_pixels_per_frame": float(sum(pixels) / len(pixels)) if pixels else 0,
        "min_pixels_in_frame": min(pixels) if pixels else 0,
        "max_pixels_in_frame": max(pixels) if pixels else 0,
        "median_pixels_in_frame": float(sorted(pixels)[len(pixels)//2]) if pixels else 0
    }

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

# Create comprehensive dataset info
dataset_info = {
    "dataset_overview": {
        "name": "ZOD CLFT Dataset",
        "version": "v2",
        "total_frames": total_frames,
        "total_pixels": total_pixels_all_frames,
        "image_resolution": "768x768 (resized from 3848x2168)",
        "classes": CLASS_NAMES,
        "generation_date": "2025-10-13"
    },

    "split_statistics": split_summary,

    "segmentation_statistics": {
        "total_pixels": total_pixels_all_frames,
        "classes": class_statistics
    },

    "quality_metrics": {
        "frames_with_multiple_classes": {
            "count": quality_metrics['frames_with_multiple_classes'],
            "percentage": float(quality_metrics['frames_with_multiple_classes'] / total_frames * 100) if total_frames > 0 else 0
        },
        "frames_with_rare_objects": {
            "count": quality_metrics['frames_with_rare_objects'],
            "percentage": float(quality_metrics['frames_with_rare_objects'] / total_frames * 100) if total_frames > 0 else 0
        },
        "average_classes_per_frame": quality_metrics['avg_classes_per_frame'],
        "max_classes_in_single_frame": quality_metrics['max_classes_in_frame'],
        "total_foreground_pixels": quality_metrics['total_foreground_pixels'],
        "foreground_percentage": float(quality_metrics['total_foreground_pixels'] / total_pixels_all_frames * 100) if total_pixels_all_frames > 0 else 0
    },

    "class_cooccurrence": {
        f"{min_cls}_{max_cls}": {
            "classes": [CLASS_NAMES.get(min_cls, f"class_{min_cls}"), CLASS_NAMES.get(max_cls, f"class_{max_cls}")],
            "frame_count": count,
            "percentage": float(count / total_frames * 100) if total_frames > 0 else 0
        }
        for (min_cls, max_cooccurrences) in class_cooccurrence.items()
        for max_cls, count in max_cooccurrences.items()
    },

    "dataset_splits": {
        "train_frames": len(train_frames),
        "validation_frames": len(valid_frames),
        "test_frames": len(test_frames),
        "test_breakdown": {
            category: {
                "count": len(frames),
                "percentage": float(len(frames) / len(test_frames) * 100) if test_frames else 0
            }
            for category, frames in test_by_category.items()
        }
    },

    "data_characteristics": {
        "most_common_classes": sorted(
            [(class_id, stats["frames_with_class"]) for class_id, stats in class_statistics.items()],
            key=lambda x: x[1], reverse=True
        )[:5],
        "rarest_classes": sorted(
            [(class_id, stats["frames_with_class"]) for class_id, stats in class_statistics.items()],
            key=lambda x: x[1]
        )[:5],
        "largest_objects_avg": sorted(
            [(class_id, stats["avg_pixels_per_frame"]) for class_id, stats in class_statistics.items()],
            key=lambda x: x[1], reverse=True
        )[:5],
        "smallest_objects_avg": sorted(
            [(class_id, stats["avg_pixels_per_frame"]) for class_id, stats in class_statistics.items()],
            key=lambda x: x[1]
        )[:5]
    }
}

# Save comprehensive dataset info
dataset_file = OUTPUT_DIR / "dataset_info.json"
with open(dataset_file, 'w') as f:
    json.dump(dataset_info, f, indent=2)

print(f"\n💾 Comprehensive dataset info saved to: {dataset_file}")

# Print enhanced statistics
print(f"\n📈 Enhanced Dataset Statistics:")
print(f"{'='*80}")
print(f"Total Frames: {total_frames:,}")
print(f"Total Pixels: {total_pixels_all_frames:,}")
print(f"Average Classes per Frame: {quality_metrics['avg_classes_per_frame']:.2f}")
print(f"Frames with Multiple Classes: {quality_metrics['frames_with_multiple_classes']:,} ({quality_metrics['frames_with_multiple_classes']/total_frames*100:.1f}%)")
print(f"Frames with Rare Objects: {quality_metrics['frames_with_rare_objects']:,} ({quality_metrics['frames_with_rare_objects']/total_frames*100:.1f}%)")
print(f"Foreground Pixel Percentage: {quality_metrics['total_foreground_pixels']/total_pixels_all_frames*100:.1f}%")

print(f"\n📊 Class Presence (Frames containing each class):")
print(f"{'Class':<12} {'Name':<12} {'Frames':>10} {'Frame %':>8} {'Avg Px':>10}")
print("-" * 60)
for class_id in sorted(class_statistics.keys()):
    stats = class_statistics[class_id]
    print(f"{class_id:<12} {stats['name']:<12} {stats['frames_with_class']:>10,} {stats['frames_percentage']:>7.1f}% {stats['avg_pixels_per_frame']:>10,.0f}")
print("-" * 60)

print(f"\n🎯 Top Co-occurring Class Pairs:")
cooccurrences = [(k, v) for k, v in dataset_info['class_cooccurrence'].items()]
cooccurrences.sort(key=lambda x: x[1]['frame_count'], reverse=True)
for pair_key, data in cooccurrences[:5]:
    cls_names = data['classes']
    print(f"   {cls_names[0]} + {cls_names[1]}: {data['frame_count']:,} frames ({data['percentage']:.1f}%)")

print(f"\n🏆 Most Common Classes (by frame presence):")
for class_id, count in dataset_info['data_characteristics']['most_common_classes'][:3]:
    name = CLASS_NAMES.get(class_id, f"class_{class_id}")
    pct = class_statistics[class_id]['frames_percentage']
    print(f"   {name}: {count:,} frames ({pct:.1f}%)")

print(f"\n⚠️  Missing Classes:")
missing_classes = [cls for cls in [1] if cls not in class_statistics]  # Vehicle class
if missing_classes:
    for cls in missing_classes:
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        print(f"   {name} (class {cls}): 0 frames, 0 pixels")
else:
    print("   None - all classes present!")

print(f"{'='*80}")

print(f"\n🎉 Metadata generation complete!")
print(f"\n📁 Output files:")
print(f"   {METADATA_DIR}/ - Individual frame metadata (JSON)")
print(f"   {OUTPUT_DIR}/dataset_info.json - Comprehensive dataset statistics")
print(f"   {OUTPUT_DIR}/all.txt - All frames ({len(all_frames):,})")
print(f"   {OUTPUT_DIR}/train_all.txt - Training set ({len(train_frames):,})")
print(f"   {OUTPUT_DIR}/early_stop_valid.txt - Validation set ({len(valid_frames):,})")
print(f"   {OUTPUT_DIR}/test_day_fair.txt - Test day+fair ({len(test_by_category['day_fair']):,})")
print(f"   {OUTPUT_DIR}/test_day_rain.txt - Test day+rain ({len(test_by_category['day_rain']):,})")
print(f"   {OUTPUT_DIR}/test_night_fair.txt - Test night+fair ({len(test_by_category['night_fair']):,})")
print(f"   {OUTPUT_DIR}/test_night_rain.txt - Test night+rain ({len(test_by_category['night_rain']):,})")
