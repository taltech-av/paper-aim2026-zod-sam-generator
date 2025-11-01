#!/usr/bin/env python3
"""
Fast Analysis for CLFT-ZOD Dataset

Optimized for speed while providing essential analysis:
- Weather condition counting per frame
- Fast frame selection for train/val/test splits
- Image mean/std calculation for training
- Visualization frame selection

Usage: python generate_analysis_fast.py
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import argparse
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Paths
OUTPUT_DIR = Path("/media/tom/ml/zod_temp")
CAMERA_DIR = OUTPUT_DIR / "camera"
LIDAR_PNG_DIR = OUTPUT_DIR / "lidar_png"
CAMERA_ANNOTATION_DIR = OUTPUT_DIR / "annotation_camera_only"
LIDAR_ANNOTATION_DIR = OUTPUT_DIR / "annotation_lidar_only"
SPLITS_DIR = OUTPUT_DIR / "splits"
DATASET_ROOT = Path("/media/tom/ml/zod-data")

# Weather conditions
CONDITIONS = ['day_fair', 'day_rain', 'night_fair', 'night_rain']

def get_weather_from_metadata(frame_id):
    """Extract weather and time from ZOD metadata"""
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

            condition = f"{timeofday}_{weather}"
            return condition if condition in CONDITIONS else None
    except Exception:
        pass
    return None

def analyze_frame_fast(frame_id):
    """Fast analysis of a single frame - count pixels per class"""
    try:
        camera_annotation_path = CAMERA_ANNOTATION_DIR / f"frame_{frame_id}.png"

        # Count pixels per class
        mask = cv2.imread(str(camera_annotation_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        unique_classes, counts = np.unique(mask, return_counts=True)
        pixel_counts = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}

        # Get classes present (excluding background class 0)
        classes_present = [int(cls) for cls in unique_classes if cls > 0]

        # Calculate total pixels and object pixels
        total_pixels = sum(pixel_counts.values())
        object_pixels = sum(v for k, v in pixel_counts.items() if k > 0)  # All non-background pixels

        # Simple quality score based on class diversity and object pixels
        quality_score = len(classes_present) * 20  # 20 points per class present

        # Bonus for having vehicles (most important)
        if 2 in classes_present:
            quality_score += 30

        # Bonus for more object pixels (up to 50 points)
        if total_pixels > 0:
            object_percentage = object_pixels / total_pixels
            quality_score += min(object_percentage * 100, 50)

        return {
            'frame_id': frame_id,
            'classes_present': classes_present,
            'pixel_counts': pixel_counts,
            'total_pixels': total_pixels,
            'object_pixels': object_pixels,
            'object_percentage': (object_pixels / total_pixels * 100) if total_pixels > 0 else 0,
            'quality_score': quality_score,
            'num_classes': len(classes_present)
        }
    except Exception:
        return None

def calculate_class_weights(total_pixel_stats, class_order, class_names):
    """Calculate recommended class weights using inverse frequency weighting for CrossEntropyLoss

    Args:
        total_pixel_stats: Dict of class_id -> pixel_count
        class_order: List of class IDs in desired order
        class_names: Dict of class_id -> class_name

    Returns:
        Dict with recommended weights for PyTorch CrossEntropyLoss
    """
    # Object classes we care about: vehicle(2), sign(3), cyclist(4), pedestrian(5)
    object_class_ids = [2, 3, 4, 5]
    eps = 1e-6  # small floor to avoid division by zero

    # Get pixel counts for all classes and objects
    all_pixels = np.array([total_pixel_stats.get(cid, 0) + eps for cid in class_order], dtype=np.float64)
    obj_pixels = np.array([total_pixel_stats.get(cid, 0) + eps for cid in object_class_ids], dtype=np.float64)

    # Calculate inverse frequencies (higher weight for rarer classes)
    inv_freq_all = 1.0 / all_pixels
    inv_freq_obj = 1.0 / obj_pixels

    # Scale the weights so background has weight ~0.1 and rare classes have reasonable weights
    # This prevents extremely large weights while maintaining the inverse frequency relationship
    background_weight = 0.1
    inv_freq_all_scaled = inv_freq_all * (background_weight / inv_freq_all[0])  # Scale so background = 0.1

    # Set ignore class (index 1) weight to 0.0 since it's masked out during training
    inv_freq_all_scaled[1] = 0.0

    # For objects only, scale similarly
    obj_background_equivalent = inv_freq_obj[0]  # vehicle as reference
    inv_freq_obj_scaled = inv_freq_obj * (background_weight / obj_background_equivalent)

    return {
        "recommended_weights": {
            "all_classes": {class_names[cid]: float(round(w, 6)) for cid, w in zip(class_order, inv_freq_all_scaled)},
            "all_classes_array": [float(round(w, 6)) for w in inv_freq_all_scaled],
            "objects_only": {class_names[cid]: float(round(w, 6)) for cid, w in zip(object_class_ids, inv_freq_obj_scaled)},
            "objects_only_array": [float(round(w, 6)) for w in inv_freq_obj_scaled]
        },
        "note": "Weights use inverse frequency scaling for CrossEntropyLoss. Background scaled to ~0.1, ignore set to 0.0. Higher weights for rarer classes."
    }
def calculate_image_statistics_fast(frame_ids, sample_size):
    """Fast image statistics calculation using sampling"""
    if sample_size >= len(frame_ids):
        sampled_ids = list(frame_ids)
    else:
        sampled_ids = random.sample(list(frame_ids), sample_size)

    print(f"\n📊 Calculating image statistics from {len(sampled_ids)} samples...")

    # Accumulators for camera images (RGB)
    camera_pixel_sum = np.zeros(3, dtype=np.float64)
    camera_pixel_sq_sum = np.zeros(3, dtype=np.float64)
    camera_pixel_count = 0

    # Accumulators for LiDAR images (3-channel geometric)
    lidar_pixel_sum = np.zeros(3, dtype=np.float64)
    lidar_pixel_sq_sum = np.zeros(3, dtype=np.float64)
    lidar_pixel_count = 0

    for frame_id in tqdm(sampled_ids, desc="Computing statistics"):
        # Camera image
        camera_path = CAMERA_DIR / f"frame_{frame_id}.png"
        if camera_path.exists():
            img = cv2.imread(str(camera_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_norm = img_rgb.astype(np.float64) / 255.0

                camera_pixel_sum += img_norm.sum(axis=(0, 1))
                camera_pixel_sq_sum += (img_norm ** 2).sum(axis=(0, 1))
                camera_pixel_count += img_norm.shape[0] * img_norm.shape[1]

        # LiDAR image
        lidar_path = LIDAR_PNG_DIR / f"frame_{frame_id}.png"
        if lidar_path.exists():
            img = cv2.imread(str(lidar_path), cv2.IMREAD_UNCHANGED)
            if img is not None and len(img.shape) == 3:
                img_norm = img.astype(np.float64) / 255.0
                mask = np.any(img > 0, axis=2)
                if np.any(mask):
                    lidar_pixel_sum += img_norm[mask].sum(axis=0)
                    lidar_pixel_sq_sum += (img_norm[mask] ** 2).sum(axis=0)
                    lidar_pixel_count += mask.sum()

    # Calculate statistics
    camera_mean = camera_pixel_sum / camera_pixel_count if camera_pixel_count > 0 else np.zeros(3)
    camera_std = np.sqrt(np.maximum(0, camera_pixel_sq_sum / camera_pixel_count - camera_mean ** 2)) if camera_pixel_count > 0 else np.ones(3)

    lidar_mean = lidar_pixel_sum / lidar_pixel_count if lidar_pixel_count > 0 else np.zeros(3)
    lidar_std = np.sqrt(np.maximum(0, lidar_pixel_sq_sum / lidar_pixel_count - lidar_mean ** 2)) if lidar_pixel_count > 0 else np.ones(3)

    return {
        'camera': {
            'mean': camera_mean.tolist(),
            'std': camera_std.tolist(),
            'sample_size': len(sampled_ids)
        },
        'lidar': {
            'mean': lidar_mean.tolist(),
            'std': lidar_std.tolist(),
            'sample_size': len(sampled_ids)
        }
    }

def setup_analysis():
    """Setup analysis environment and find available frames"""
    parser = argparse.ArgumentParser(description="Fast analysis for CLFT-ZOD dataset")
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--stats-samples', type=int, default=1000, help='Number of samples for image statistics')
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Fast CLFT-ZOD Analysis")
    print("=" * 60)

    # Create directories
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all available frames
    print("\n🔍 Scanning for available frames...")

    camera_annotation_frames = set()
    if CAMERA_ANNOTATION_DIR.exists():
        camera_annotation_files = list(CAMERA_ANNOTATION_DIR.glob("frame_*.png"))
        camera_annotation_frames = {f.stem.replace("frame_", "") for f in camera_annotation_files}

    lidar_annotation_frames = set()
    if LIDAR_ANNOTATION_DIR.exists():
        lidar_annotation_files = list(LIDAR_ANNOTATION_DIR.glob("frame_*.png"))
        lidar_annotation_frames = {f.stem.replace("frame_", "") for f in lidar_annotation_files}

    # Frames with complete annotations
    complete_frames = camera_annotation_frames & lidar_annotation_frames
    print(f"✅ Found {len(complete_frames):,} frames with complete annotations")

    if len(complete_frames) == 0:
        print("❌ No frames with complete data found!")
        return None

    return args, complete_frames

def analyze_weather_conditions(complete_frames):
    """Analyze weather conditions from metadata"""
    print("\n🌤️ Analyzing weather conditions...")

    weather_counts = defaultdict(int)
    frame_weather_map = {}

    for frame_id in tqdm(sorted(complete_frames), desc="Reading metadata"):
        weather = get_weather_from_metadata(frame_id)
        if weather:
            weather_counts[weather] += 1
            frame_weather_map[frame_id] = weather

    # Display weather analysis
    print("\n📊 Weather Condition Analysis:")
    print("=" * 40)
    total_analyzed = sum(weather_counts.values())
    for condition in CONDITIONS:
        count = weather_counts[condition]
        pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0
        print(f"{condition:<12}: {count:>6,} frames ({pct:>5.1f}%)")

    return weather_counts, frame_weather_map, total_analyzed

def analyze_frames_quality(complete_frames, frame_weather_map, workers):
    """Analyze frames for quality and pixel counts"""
    print("\n🔍 Analyzing frames for quality...")

    valid_frames = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(analyze_frame_fast, frame_id) for frame_id in sorted(complete_frames)]
        for future in tqdm(futures, desc="Processing frames"):
            result = future.result()
            if result and result['frame_id'] in frame_weather_map:
                result['condition'] = frame_weather_map[result['frame_id']]
                valid_frames.append(result)

    print(f"✅ Analyzed {len(valid_frames):,} frames with valid data")
    return valid_frames

def select_frames(valid_frames):
    """Select frames for training, validation, testing, and visualization"""
    print("\n🎯 Selecting frames for splits...")

    # Sort frames by quality score
    valid_frames.sort(key=lambda x: -x['quality_score'])

    # Target sizes
    TARGET_TRAIN = 15000
    TARGET_VAL = 1000
    MAX_TEST_PER_CONDITION = 2000
    VISUALIZATION_TOTAL = 20  # Total visualization frames

    # Select training frames (best quality, balanced across conditions if possible)
    training_frames = []
    condition_counts = defaultdict(int)

    for frame in valid_frames:
        if len(training_frames) >= TARGET_TRAIN:
            break
        # Try to balance conditions roughly
        if condition_counts[frame['condition']] < TARGET_TRAIN // len(CONDITIONS) + 1000:
            training_frames.append(frame)
            condition_counts[frame['condition']] += 1

    training_frame_ids = {f['frame_id'] for f in training_frames}

    # Select validation frames from remaining
    remaining_frames = [f for f in valid_frames if f['frame_id'] not in training_frame_ids]
    validation_frames = remaining_frames[:TARGET_VAL]
    validation_frame_ids = {f['frame_id'] for f in validation_frames}

    # Select test frames per condition
    test_frames = {condition: [] for condition in CONDITIONS}
    excluded_ids = training_frame_ids | validation_frame_ids

    for condition in CONDITIONS:
        condition_frames = [f for f in remaining_frames[TARGET_VAL:] if f['condition'] == condition and f['frame_id'] not in excluded_ids]
        condition_frames.sort(key=lambda x: -x['quality_score'])
        test_frames[condition] = [f['frame_id'] for f in condition_frames[:MAX_TEST_PER_CONDITION]]

    # Select visualization frames (best quality across all conditions)
    all_remaining = [f for f in valid_frames if f['frame_id'] not in (training_frame_ids | validation_frame_ids)]
    all_remaining.sort(key=lambda x: -x['quality_score'])
    visualization_frame_ids = [f['frame_id'] for f in all_remaining[:VISUALIZATION_TOTAL]]

    print(f"✅ Training: {len(training_frames)} frames")
    print(f"✅ Validation: {len(validation_frames)} frames")
    print(f"✅ Testing: {sum(len(test_frames[c]) for c in CONDITIONS)} frames")
    print(f"✅ Visualization: {len(visualization_frame_ids)} frames")

    return training_frames, training_frame_ids, validation_frames, validation_frame_ids, test_frames, visualization_frame_ids

def calculate_pixel_statistics(valid_frames):
    """Calculate and display pixel statistics"""
    print("\n📊 Calculating pixel statistics...")

    # Class mapping in desired order
    class_order = [0, 1, 2, 3, 4, 5]  # background, ignore, vehicle, sign, cyclist, pedestrian
    class_names = {
        0: "background",
        1: "ignore",
        2: "vehicle",
        3: "sign",
        4: "cyclist",
        5: "pedestrian"
    }

    # Aggregate pixel counts by condition
    condition_pixel_stats = {condition: defaultdict(int) for condition in CONDITIONS}
    total_pixel_stats = defaultdict(int)

    for frame in valid_frames:
        condition = frame['condition']
        pixel_counts = frame['pixel_counts']

        # Aggregate by condition
        for class_id, count in pixel_counts.items():
            condition_pixel_stats[condition][class_id] += count
            total_pixel_stats[class_id] += count

    # Create pixel statistics arrays
    pixel_stats_arrays = {}
    for condition in CONDITIONS:
        pixel_stats_arrays[condition] = []
        total_condition_pixels = sum(condition_pixel_stats[condition].values())

        for class_id in class_order:
            pixels = condition_pixel_stats[condition][class_id]
            percentage = (pixels / total_condition_pixels * 100) if total_condition_pixels > 0 else 0
            pixel_stats_arrays[condition].append({
                "class_id": class_id,
                "class_name": class_names[class_id],
                "pixel_count": pixels,
                "percentage": round(percentage, 2)
            })

    # Overall pixel statistics array
    total_pixel_array = []
    total_all_pixels = sum(total_pixel_stats.values())

    for class_id in class_order:
        pixels = total_pixel_stats[class_id]
        percentage = (pixels / total_all_pixels * 100) if total_all_pixels > 0 else 0
        total_pixel_array.append({
            "class_id": class_id,
            "class_name": class_names[class_id],
            "pixel_count": pixels,
            "percentage": round(percentage, 2)
        })

    # Display pixel statistics
    print("\n📊 Pixel Counts by Weather Condition:")
    print("=" * 80)
    print(f"{'Condition':<12} {'Class':<12} {'Pixels':>15} {'Percentage':>10}")
    print("-" * 50)

    for condition in CONDITIONS:
        total_condition_pixels = sum(condition_pixel_stats[condition].values())
        if total_condition_pixels > 0:
            print(f"\n{condition}:")
            for class_id in class_order:
                pixels = condition_pixel_stats[condition][class_id]
                percentage = (pixels / total_condition_pixels * 100)
                class_name = class_names[class_id]
                print(f"{'':<12} {class_name:<12} {pixels:>15,} {percentage:>9.1f}%")
            print(f"{'':<12} {'TOTAL':<12} {total_condition_pixels:>15,} {'100.0%':>10}")

    # Overall pixel statistics
    print(f"\n📊 Overall Pixel Statistics (All {len(valid_frames)} frames):")
    print("=" * 60)
    for item in total_pixel_array:
        print(f"{item['class_name']:<12}: {item['pixel_count']:>15,} pixels ({item['percentage']:>5.1f}%)")

    print(f"{'TOTAL':<12}: {total_all_pixels:>15,} pixels")

    return total_pixel_stats, total_pixel_array, pixel_stats_arrays, class_order, class_names

def generate_split_files(training_frame_ids, validation_frame_ids, test_frames, visualization_frame_ids):
    """Generate all split files"""
    print("\n📝 Generating split files...")

    def format_path(frame_id):
        return f"camera/frame_{frame_id}.png"

    # Training
    with open(SPLITS_DIR / "train.txt", 'w') as f:
        for frame_id in sorted(training_frame_ids):
            f.write(f"{format_path(frame_id)}\n")

    # Validation
    with open(SPLITS_DIR / "validation.txt", 'w') as f:
        for frame_id in sorted(validation_frame_ids):
            f.write(f"{format_path(frame_id)}\n")

    # Test files (4 files, one per condition)
    for condition in CONDITIONS:
        if test_frames[condition]:
            with open(SPLITS_DIR / f"test_{condition}.txt", 'w') as f:
                for frame_id in sorted(test_frames[condition]):
                    f.write(f"{format_path(frame_id)}\n")

    # Visualization
    with open(SPLITS_DIR / "visualization.txt", 'w') as f:
        for frame_id in sorted(visualization_frame_ids):
            f.write(f"{format_path(frame_id)}\n")

    # All frames (train + validation + test + visualization)
    all_selected_frames = training_frame_ids | validation_frame_ids
    for condition in CONDITIONS:
        all_selected_frames.update(test_frames[condition])
    all_selected_frames.update(visualization_frame_ids)

    with open(SPLITS_DIR / "all.txt", 'w') as f:
        for frame_id in sorted(all_selected_frames):
            f.write(f"{format_path(frame_id)}\n")

    return all_selected_frames

def compute_and_print_weights(total_pixel_stats, class_order, class_names):
    """Compute and print recommended weights"""
    # Compute recommended weights using the separate function
    recommended_weights = calculate_class_weights(total_pixel_stats, class_order, class_names)

    # Extract the recommended weights
    weights = recommended_weights["recommended_weights"]
    all_classes_array = weights["all_classes_array"]
    objects_array = weights["objects_only_array"]

    # Print recommended weights to console (clean and simple)
    print("\n🔧 Recommended Class Weights for CrossEntropyLoss:")
    print("=" * 60)
    print("Class order: [background(0), ignore(1), vehicle(2), sign(3), cyclist(4), pedestrian(5)]")
    print("Method: Inverse frequency weighting (higher weights for rarer classes)")
    print("Note: ignore(1) has weight 0.0 since it's masked out during training")
    print()
    print("📋 All Classes Array (copy this to your PyTorch config):")
    print(f"   {all_classes_array}")
    print()
    print("📋 Objects Only Array [vehicle, sign, cyclist, pedestrian]:")
    print(f"   {objects_array}")
    print()
    print("📋 Individual Class Breakdown:")
    for i, (cid, w) in enumerate(zip(class_order, all_classes_array)):
        status = "(ignored in loss)" if cid == 1 else ""
        print(f"   {i}: {class_names[cid]:<12} {w:.6f} {status}")
    print()
    print("💡 Usage: torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))")

    return recommended_weights

def save_analysis_report(weather_counts, total_analyzed, total_pixel_array, pixel_stats_arrays,
                        recommended_weights, training_frames, validation_frames, test_frames,
                        visualization_frame_ids, all_selected_frames, image_stats):
    """Save comprehensive analysis report to JSON"""
    analysis_report = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "Fast CLFT-ZOD Analysis",
        "weather_analysis": {
            "total_frames_analyzed": total_analyzed,
            "condition_counts": dict(weather_counts),
            "condition_percentages": {
                condition: round(weather_counts[condition] / total_analyzed * 100, 2) if total_analyzed > 0 else 0
                for condition in CONDITIONS
            }
        },
        "pixel_statistics": {
            "total": total_pixel_array,
            **pixel_stats_arrays
        },
        "recommended_weights": recommended_weights,
        "frame_selection": {
            "training_frames": len(training_frames),
            "validation_frames": len(validation_frames),
            "testing_frames": {condition: len(test_frames[condition]) for condition in CONDITIONS},
            "visualization_frames": len(visualization_frame_ids),
            "total_selected_frames": len(all_selected_frames)
        },
        "image_statistics": image_stats
    }

    with open(SPLITS_DIR / "fast_analysis_report.json", 'w') as f:
        json.dump(analysis_report, f, indent=2)

def print_final_summary(total_analyzed, training_frames, validation_frames, test_frames,
                       visualization_frame_ids, all_selected_frames, image_stats):
    """Print final analysis summary"""
    print("\n💾 Analysis complete!")
    print(f"📁 Files saved to: {SPLITS_DIR}")
    print(f"📊 Weather analysis: {total_analyzed} frames analyzed")
    print(f"🎯 Training: {len(training_frames)} frames")
    print(f"✅ Validation: {len(validation_frames)} frames")
    print(f"🧪 Testing: {sum(len(test_frames[c]) for c in CONDITIONS)} frames across {len(CONDITIONS)} conditions")
    print(f"📸 Visualization: {len(visualization_frame_ids)} frames")
    print(f"📋 All selected frames: {len(all_selected_frames)} (saved to all.txt)")

    print("\n📊 Image Normalization Values:")
    print("Camera (RGB):")
    print(f"  Mean: {[round(v, 6) for v in image_stats['camera']['mean']]}")
    print(f"  Std:  {[round(v, 6) for v in image_stats['camera']['std']]}")
    print("\nLiDAR (X, Y, Z):")
    print(f"  Mean: {[round(v, 6) for v in image_stats['lidar']['mean']]}")
    print(f"  Std:  {[round(v, 6) for v in image_stats['lidar']['std']]}")

    print("\n✅ Fast analysis complete!")

def main():
    # Setup
    setup_result = setup_analysis()
    if setup_result is None:
        return
    args, complete_frames = setup_result

    # Weather analysis
    weather_counts, frame_weather_map, total_analyzed = analyze_weather_conditions(complete_frames)

    # Frame quality analysis
    valid_frames = analyze_frames_quality(complete_frames, frame_weather_map, args.workers)

    # Frame selection
    training_frames, training_frame_ids, validation_frames, validation_frame_ids, test_frames, visualization_frame_ids = select_frames(valid_frames)

    # Pixel statistics
    total_pixel_stats, total_pixel_array, pixel_stats_arrays, class_order, class_names = calculate_pixel_statistics(valid_frames)

    # Image statistics
    image_stats = calculate_image_statistics_fast(training_frame_ids, args.stats_samples)

    # Generate split files
    all_selected_frames = generate_split_files(training_frame_ids, validation_frame_ids, test_frames, visualization_frame_ids)

    # Compute and print weights
    recommended_weights = compute_and_print_weights(total_pixel_stats, class_order, class_names)

    # Save analysis report
    save_analysis_report(weather_counts, total_analyzed, total_pixel_array, pixel_stats_arrays,
                        recommended_weights, training_frames, validation_frames, test_frames,
                        visualization_frame_ids, all_selected_frames, image_stats)

    # Final summary
    print_final_summary(total_analyzed, training_frames, validation_frames, test_frames,
                       visualization_frame_ids, all_selected_frames, image_stats)

if __name__ == "__main__":
    main()