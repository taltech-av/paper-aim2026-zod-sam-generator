#!/usr/bin/env python3
"""
Simplified Pixel Counting Analysis for CLFT-ZOD Dataset

Counts pixels per class in annotations, aggregates statistics by weather condition,
and selects optimal frames for training, validation, and testing.

Features:
- Pixel counting per class (background, ignore, vehicle, sign, cyclist, pedestrian)
- Weather condition analysis (day_fair, day_rain, night_fair, night_rain)
- Class presence statistics across conditions
- Quality-based frame selection for training splits
- Comprehensive reporting and split file generation

Usage: python generate_analysis.py
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import argparse

# Paths
OUTPUT_DIR = Path("output")
CAMERA_DIR = OUTPUT_DIR / "camera"
LIDAR_PNG_DIR = OUTPUT_DIR / "lidar_png"
CAMERA_ANNOTATION_DIR = OUTPUT_DIR / "annotation_camera_only"
LIDAR_ANNOTATION_DIR = OUTPUT_DIR / "annotation_lidar_only"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
SPLITS_DIR = OUTPUT_DIR / "splits"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations" / "selected_frames"
DATASET_ROOT = Path("/media/tom/ml/zod-data")

# Weather conditions
CONDITIONS = ['day_fair', 'day_rain', 'night_fair', 'night_rain']

def count_pixels_simple(annotation_path):
    """Simple pixel counting using numpy unique"""
    try:
        mask = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        unique_classes, counts = np.unique(mask, return_counts=True)
        pixel_counts = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}

        # Basic statistics
        total_pixels = sum(pixel_counts.values())
        foreground_pixels = sum(v for k, v in pixel_counts.items() if k > 0)
        classes_present = [cls for cls in pixel_counts.keys() if cls > 0]

        return {
            'pixel_counts': pixel_counts,
            'total_pixels': total_pixels,
            'foreground_pixels': foreground_pixels,
            'classes_present': classes_present,
            'num_classes': len(classes_present),
            'foreground_percentage': float(foreground_pixels / total_pixels * 100) if total_pixels > 0 else 0
        }
    except Exception as e:
        print(f"Error counting pixels for {annotation_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Simple pixel counting analysis for weather-based frame selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analyzes annotations to count pixels per class and aggregates by weather conditions.
Generates comprehensive statistics and selects frames for training splits.

Weather Conditions:
- day_fair: Clear daytime conditions
- day_rain: Rainy daytime conditions
- night_fair: Clear nighttime conditions
- night_rain: Rainy nighttime conditions

Outputs:
- Pixel counts per class per weather condition
- Class presence statistics
- Weather-specific train/val/test splits
- Comprehensive analysis report

Usage: python generate_analysis.py
        """
    )
    args = parser.parse_args()

    print("=" * 60)
    print("📊 Simple Pixel Counting Analysis")
    print("=" * 60)

    # Create directories
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # Find all available files
    print("\n🔍 Scanning for available files...")

    # Get all frame IDs from different sources
    camera_frames = set()
    lidar_png_frames = set()
    camera_annotation_frames = set()
    lidar_annotation_frames = set()

    # Scan camera images
    if CAMERA_DIR.exists():
        camera_files = list(CAMERA_DIR.glob("frame_*.png"))
        camera_frames = {f.stem.replace("frame_", "") for f in camera_files}
        print(f"📷 Found {len(camera_frames):,} camera images")

    # Scan camera annotations
    if CAMERA_ANNOTATION_DIR.exists():
        camera_annotation_files = list(CAMERA_ANNOTATION_DIR.glob("frame_*.png"))
        camera_annotation_frames = {f.stem.replace("frame_", "") for f in camera_annotation_files}
        print(f"📋 Found {len(camera_annotation_frames):,} camera annotations")

    # Scan LiDAR annotations
    if LIDAR_ANNOTATION_DIR.exists():
        lidar_annotation_files = list(LIDAR_ANNOTATION_DIR.glob("frame_*.png"))
        lidar_annotation_frames = {f.stem.replace("frame_", "") for f in lidar_annotation_files}
        print(f"📋 Found {len(lidar_annotation_frames):,} LiDAR annotations")

    # Find frames with complete data (camera + annotations)
    complete_frames = camera_frames & camera_annotation_frames & lidar_annotation_frames

    print(f"\n✅ Complete frames: {len(complete_frames):,}")

    if len(complete_frames) == 0:
        print("❌ No frames with complete data found!")
        return

    # Initialize data structures
    condition_data = {condition: [] for condition in CONDITIONS}
    condition_stats = {condition: defaultdict(int) for condition in CONDITIONS}

    print(f"\n🔄 Analyzing {len(complete_frames)} frames...")

    # Process each complete frame
    for frame_id in tqdm(sorted(complete_frames), desc="Processing frames"):
        try:
            # File paths
            camera_annotation_path = CAMERA_ANNOTATION_DIR / f"frame_{frame_id}.png"
            lidar_annotation_path = LIDAR_ANNOTATION_DIR / f"frame_{frame_id}.png"

            # Get weather/time info from ZOD metadata
            weather = "fair"
            timeofday = "day"

            try:
                metadata_path = DATASET_ROOT / "single_frames" / frame_id / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        zod_metadata = json.load(f)

                    # Extract weather
                    scraped_weather = str(zod_metadata.get("scraped_weather", "")).lower()
                    precipitation_keywords = ["rain", "snow", "sleet", "hail", "storm", "drizzle"]
                    has_precipitation = any(keyword in scraped_weather for keyword in precipitation_keywords)
                    weather = "rain" if has_precipitation else "fair"

                    # Extract time of day
                    time_of_day = str(zod_metadata.get("time_of_day", "day")).lower()
                    timeofday = "night" if "night" in time_of_day else "day"

            except Exception as e:
                pass  # Skip metadata errors

            condition = f"{timeofday}_{weather}"

            # Skip if not one of our target conditions
            if condition not in CONDITIONS:
                continue

            # Count pixels in annotations
            camera_stats = count_pixels_simple(camera_annotation_path)
            lidar_stats = count_pixels_simple(lidar_annotation_path)

            if camera_stats is None or lidar_stats is None:
                continue

            # Simple quality score based on class presence and pixel counts
            has_vehicle = 2 in camera_stats['classes_present']
            has_sign = 3 in camera_stats['classes_present']
            has_cyclist = 4 in camera_stats['classes_present']
            has_pedestrian = 5 in camera_stats['classes_present']

            # Basic quality score (0-100)
            quality_score = 0
            if has_vehicle:
                quality_score += 40
            if has_sign:
                quality_score += 20
            if has_cyclist:
                quality_score += 20
            if has_pedestrian:
                quality_score += 20

            # Bonus for multiple classes
            quality_score += min(camera_stats['num_classes'] * 10, 30)

            # Penalty for too much/too little foreground
            fg_pct = camera_stats['foreground_percentage']
            if not (5 <= fg_pct <= 50):
                quality_score *= 0.7

            frame_info = {
                'frame_id': frame_id,
                'condition': condition,
                'quality_score': quality_score,
                'camera_stats': camera_stats,
                'lidar_stats': lidar_stats,
                'has_vehicle': has_vehicle,
                'has_sign': has_sign,
                'has_cyclist': has_cyclist,
                'has_pedestrian': has_pedestrian,
            }

            condition_data[condition].append(frame_info)

            # Update condition statistics
            for class_id, count in camera_stats['pixel_counts'].items():
                condition_stats[condition][f'camera_class_{class_id}'] += count

            for class_id, count in lidar_stats['pixel_counts'].items():
                condition_stats[condition][f'lidar_class_{class_id}'] += count

        except Exception as e:
            print(f"❌ Error processing {frame_id}: {e}")
            continue

    print("\n✅ Analysis complete!")

    # Calculate comprehensive statistics per condition
    total_frames_by_condition = {condition: len(condition_data[condition]) for condition in CONDITIONS}
    total_all_frames = sum(total_frames_by_condition.values())

    # Create condition distribution with percentages as array
    condition_distribution_array = []
    for condition in CONDITIONS:
        count = total_frames_by_condition[condition]
        percentage = (count / total_all_frames * 100) if total_all_frames > 0 else 0
        condition_distribution_array.append({
            "condition": condition,
            "frame_count": count,
            "percentage": round(percentage, 2)
        })

    print("\n📊 Condition Distribution:")
    print("=" * 60)
    print(f"{'Condition':<15} {'Frames':>8} {'Percentage':>10}")
    print("-" * 35)
    for condition in CONDITIONS:
        count = total_frames_by_condition[condition]
        pct = (count / total_all_frames * 100) if total_all_frames > 0 else 0
        print(f"{condition:<15} {count:>8,} {pct:>9.1f}%")

    # Calculate pixel counts per class per condition
    print("\n📊 Pixel Counts per Class per Weather Condition:")
    print("=" * 80)

    class_names = {
        0: "background",
        1: "ignore",
        2: "vehicle",
        3: "sign",
        4: "cyclist",
        5: "pedestrian"
    }

    # Aggregate pixel counts per condition
    condition_pixel_stats = {condition: defaultdict(int) for condition in CONDITIONS}

    for condition in CONDITIONS:
        frames = condition_data[condition]
        for frame in frames:
            # Camera pixel counts
            for class_id, count in frame['camera_stats']['pixel_counts'].items():
                condition_pixel_stats[condition][f'camera_class_{class_id}'] += count
            # LiDAR pixel counts
            for class_id, count in frame['lidar_stats']['pixel_counts'].items():
                condition_pixel_stats[condition][f'lidar_class_{class_id}'] += count

    # Display pixel statistics
    print(f"{'Condition':<12} {'Class':<12} {'Camera Pixels':>15} {'LiDAR Pixels':>15} {'Total Pixels':>15} {'Percentage':>10}")
    print("-" * 95)

    for condition in CONDITIONS:
        print(f"\n{condition}:")
        total_condition_pixels = 0

        for class_id in sorted(class_names.keys()):
            camera_pixels = condition_pixel_stats[condition].get(f'camera_class_{class_id}', 0)
            lidar_pixels = condition_pixel_stats[condition].get(f'lidar_class_{class_id}', 0)
            total_pixels = camera_pixels + lidar_pixels
            total_condition_pixels += total_pixels

        # Second pass to calculate percentages
        for class_id in sorted(class_names.keys()):
            camera_pixels = condition_pixel_stats[condition].get(f'camera_class_{class_id}', 0)
            lidar_pixels = condition_pixel_stats[condition].get(f'lidar_class_{class_id}', 0)
            total_pixels = camera_pixels + lidar_pixels

            if total_pixels > 0:  # Only show classes with pixels
                percentage = (total_pixels / total_condition_pixels * 100) if total_condition_pixels > 0 else 0
                print(f"{'':<12} {class_names[class_id]:<12} {camera_pixels:>15,} {lidar_pixels:>15,} {total_pixels:>15,} {percentage:>9.1f}%")

        print(f"{'':<12} {'TOTAL':<12} {'-':>15} {'-':>15} {total_condition_pixels:>15,} {'100.0%':>10}")

    # Class presence statistics
    print("\n📊 Class Presence Statistics:")
    print("=" * 60)

    for condition in CONDITIONS:
        frames = condition_data[condition]
        if not frames:
            continue

        print(f"\n{condition} ({len(frames)} frames):")

        # Count frames with each class
        class_presence = defaultdict(int)
        for frame in frames:
            for class_id in [2, 3, 4, 5]:  # Only count meaningful classes
                if class_id in frame['camera_stats']['classes_present']:
                    class_presence[class_id] += 1

        for class_id, count in class_presence.items():
            pct = (count / len(frames)) * 100
            print(f"  {class_names[class_id]:<12}: {count:>5} frames ({pct:>5.1f}%)")

    # Frame selection - simple approach
    print("\n🎯 Selecting frames for training...")

    TARGET_TRAINING_FRAMES = 15000
    TARGET_VALIDATION_FRAMES = 2500
    MAX_TESTING_FRAMES_TOTAL = 2000  # Limit total testing frames across all conditions
    VISUALIZATION_FRAMES_PER_CONDITION = 5

    # Collect all frames, sorted by quality score
    all_frames = []
    for condition in CONDITIONS:
        for frame in condition_data[condition]:
            frame['condition'] = condition
            all_frames.append(frame)

    all_frames.sort(key=lambda x: -x['quality_score'])

    # Select frames
    training_frames = all_frames[:TARGET_TRAINING_FRAMES]
    training_frame_ids = {f['frame_id'] for f in training_frames}

    remaining_frames = all_frames[TARGET_TRAINING_FRAMES:]
    validation_frames = remaining_frames[:TARGET_VALIDATION_FRAMES]
    validation_frame_ids = {f['frame_id'] for f in validation_frames}

    # Testing frames - limited to MAX_TESTING_FRAMES_TOTAL total
    testing_frames = {condition: [] for condition in CONDITIONS}
    excluded_frame_ids = training_frame_ids | validation_frame_ids

    # First, collect all available testing frames per condition
    available_testing_frames = []
    for condition in CONDITIONS:
        remaining_condition_frames = [f for f in condition_data[condition] if f['frame_id'] not in excluded_frame_ids]
        remaining_condition_frames.sort(key=lambda x: -x['quality_score'])
        for frame in remaining_condition_frames:
            available_testing_frames.append((condition, frame))

    # Select top MAX_TESTING_FRAMES_TOTAL testing frames
    selected_testing_frames = available_testing_frames[:MAX_TESTING_FRAMES_TOTAL]

    # Group by condition
    for condition, frame in selected_testing_frames:
        testing_frames[condition].append(frame['frame_id'])

    # Visualization frames (5 per condition from remaining frames)
    print("\n🎨 Selecting visualization frames...")
    visualization_frames = {condition: [] for condition in CONDITIONS}

    for condition in CONDITIONS:
        # Get all frames for this condition, sorted by quality score
        condition_frames = sorted(condition_data[condition], key=lambda x: -x['quality_score'])

        # Select top 5 for visualization (or all available if less than 5)
        num_viz = min(VISUALIZATION_FRAMES_PER_CONDITION, len(condition_frames))
        visualization_frames[condition] = [f['frame_id'] for f in condition_frames[:num_viz]]

        print(f"✅ {condition}: {len(visualization_frames[condition])} visualization frames")

    print(f"✅ Training: {len(training_frames)} frames")
    print(f"✅ Validation: {len(validation_frames)} frames")
    print(f"✅ Testing: {sum(len(testing_frames[c]) for c in CONDITIONS)} frames")
    print(f"✅ Visualization: {sum(len(visualization_frames[c]) for c in CONDITIONS)} frames")

    # Generate split files
    print("\n📝 Generating split files...")

    def format_path(frame_id, modality="camera"):
        if modality == "camera":
            return f"camera/frame_{frame_id}.png"
        elif modality == "camera_annotation":
            return f"annotation_camera_only/frame_{frame_id}.png"
        elif modality == "lidar_annotation":
            return f"annotation_lidar_only/frame_{frame_id}.png"
        return f"camera/frame_{frame_id}.png"

    # Training split
    train_file = SPLITS_DIR / "train.txt"
    with open(train_file, 'w') as f:
        for frame_id in sorted(training_frame_ids):
            f.write(f"{format_path(frame_id)}\n")

    # Validation split
    val_file = SPLITS_DIR / "validation.txt"
    with open(val_file, 'w') as f:
        for frame_id in sorted(validation_frame_ids):
            f.write(f"{format_path(frame_id)}\n")

    # Testing splits per condition
    for condition in CONDITIONS:
        if testing_frames[condition]:
            test_file = SPLITS_DIR / f"{condition}_test.txt"
            with open(test_file, 'w') as f:
                for frame_id in sorted(testing_frames[condition]):
                    f.write(f"{format_path(frame_id)}\n")

    # Visualization splits per condition
    for condition in CONDITIONS:
        if visualization_frames[condition]:
            viz_file = SPLITS_DIR / f"{condition}_visualization.txt"
            with open(viz_file, 'w') as f:
                for frame_id in sorted(visualization_frames[condition]):
                    f.write(f"{format_path(frame_id)}\n")

    # All frames (train + validation + test)
    all_selected_frames = training_frame_ids | validation_frame_ids
    for condition in CONDITIONS:
        all_selected_frames.update(testing_frames[condition])

    all_file = SPLITS_DIR / "all.txt"
    with open(all_file, 'w') as f:
        for frame_id in sorted(all_selected_frames):
            f.write(f"{format_path(frame_id)}\n")

    # Count split files created
    splits_created = [
        f"train.txt: {len(training_frame_ids)} frames",
        f"validation.txt: {len(validation_frame_ids)} frames",
        f"all.txt: {len(all_selected_frames)} frames"
    ]

    for condition in CONDITIONS:
        if testing_frames[condition]:
            splits_created.append(f"{condition}_test.txt: {len(testing_frames[condition])} testing frames")
        if visualization_frames[condition]:
            splits_created.append(f"{condition}_visualization.txt: {len(visualization_frames[condition])} visualization frames")

    # Save comprehensive analysis report
    # Calculate pixel statistics with class names and percentages for JSON using arrays
    pixel_stats_arrays = {}
    for condition in CONDITIONS:
        pixel_stats_arrays[condition] = {
            "camera": [],
            "lidar": []
        }
        total_condition_pixels = sum(condition_pixel_stats[condition].values())

        # Initialize arrays for all classes (0-5)
        for class_idx in range(6):
            camera_key = f'camera_class_{class_idx}'
            lidar_key = f'lidar_class_{class_idx}'

            camera_pixels = condition_pixel_stats[condition].get(camera_key, 0)
            lidar_pixels = condition_pixel_stats[condition].get(lidar_key, 0)

            class_name = class_names.get(class_idx, f"unknown_{class_idx}")

            # Camera array entry
            camera_percentage = (camera_pixels / total_condition_pixels * 100) if total_condition_pixels > 0 else 0
            pixel_stats_arrays[condition]["camera"].append({
                "class_name": class_name,
                "pixel_count": camera_pixels,
                "percentage": round(camera_percentage, 2)
            })

            # LiDAR array entry
            lidar_percentage = (lidar_pixels / total_condition_pixels * 100) if total_condition_pixels > 0 else 0
            pixel_stats_arrays[condition]["lidar"].append({
                "class_name": class_name,
                "pixel_count": lidar_pixels,
                "percentage": round(lidar_percentage, 2)
            })

    analysis_report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "analysis_type": "Simple Pixel Counting Analysis",
        "total_frames_analyzed": total_all_frames,
        "conditions_analyzed": CONDITIONS,
        "condition_distribution": condition_distribution_array,
        "class_mapping": class_names,
        "pixel_statistics": pixel_stats_arrays,
        "frame_selection": {
            "target_training_frames": TARGET_TRAINING_FRAMES,
            "target_validation_frames": TARGET_VALIDATION_FRAMES,
            "max_testing_frames_total": MAX_TESTING_FRAMES_TOTAL,
            "visualization_frames_per_condition": VISUALIZATION_FRAMES_PER_CONDITION,
            "training_frames": len(training_frame_ids),
            "validation_frames": len(validation_frame_ids),
            "testing_frames": [
                {"condition": condition, "frame_count": len(testing_frames[condition])}
                for condition in CONDITIONS
            ],
            "visualization_frames": [
                {"condition": condition, "frame_count": len(visualization_frames[condition])}
                for condition in CONDITIONS
            ],
            "summary": {
                "total_training_frames": len(training_frame_ids),
                "total_validation_frames": len(validation_frame_ids),
                "total_testing_frames": sum(len(testing_frames[c]) for c in CONDITIONS),
                "total_visualization_frames": sum(len(visualization_frames[c]) for c in CONDITIONS),
                "total_selected_frames": len(all_selected_frames)
            }
        }
    }

    analysis_file = ANALYSIS_DIR / "pixel_count_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_report, f, indent=2)

    print("\n💾 Analysis complete!")
    print(f"📁 Results saved to: {ANALYSIS_DIR}")
    print(f"📁 Split files saved to: {SPLITS_DIR}")
    print(f"📊 Total frames analyzed: {total_all_frames}")
    print(f"🎯 Training frames selected: {len(training_frame_ids)}")
    print(f"✅ Validation frames selected: {len(validation_frame_ids)}")
    print(f"🧪 Testing frames selected: {sum(len(testing_frames[c]) for c in CONDITIONS)} (max {MAX_TESTING_FRAMES_TOTAL} total)")
    print(f"📸 Visualization frames selected: {sum(len(visualization_frames[c]) for c in CONDITIONS)} ({VISUALIZATION_FRAMES_PER_CONDITION} per condition)")
    print(f"📋 All selected frames: {len(all_selected_frames)} (saved to all.txt)")

    print("\n✅ Simple pixel counting analysis complete!")

if __name__ == "__main__":
    main()