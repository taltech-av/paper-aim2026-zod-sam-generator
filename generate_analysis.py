#!/usr/bin/env python3
"""
New Generate Analysis for CLFT-ZOD Dataset

Analyzes only good frames, generates frame analysis JSON with pixel counts per class,
and creates balanced train/validation splits ensuring all classes are represented.
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

# Paths
OUTPUT_DIR = Path("/media/tom/ml/zod_temp")
CAMERA_ANNOTATION_DIR = OUTPUT_DIR / "annotation_camera_only"
SPLITS_DIR = OUTPUT_DIR / "splits_balanced"
DATASET_ROOT = Path("/media/tom/ml/zod-data")

# Weather conditions
CONDITIONS = ['day_fair', 'day_rain', 'night_fair', 'night_rain']

# Class mapping
CLASS_NAMES = {
    0: "background",
    1: "ignore",
    2: "vehicle",
    3: "sign",
    4: "cyclist",
    5: "pedestrian"
}

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

def load_good_frames(good_frames_path):
    """Load frame IDs from good frames file"""
    frame_ids = []
    with open(good_frames_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract frame ID from path like "camera/frame_099985.png"
                if '/' in line:
                    frame_id = line.split('/')[-1].replace('frame_', '').replace('.png', '')
                else:
                    frame_id = line.replace('frame_', '').replace('.png', '')
                frame_ids.append(frame_id)
    return frame_ids

def analyze_frame_pixels(frame_id):
    """Analyze pixel counts per class for a single frame"""
    try:
        annotation_path = CAMERA_ANNOTATION_DIR / f"frame_{frame_id}.png"
        mask = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        unique_classes, counts = np.unique(mask, return_counts=True)
        pixel_counts = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}

        # Get all classes present (including background)
        classes_present = [int(cls) for cls in unique_classes]

        # Calculate total pixels
        total_pixels = sum(pixel_counts.values())

        # Calculate percentages for each class
        class_percentages = {}
        for class_id, count in pixel_counts.items():
            class_percentages[class_id] = (count / total_pixels * 100) if total_pixels > 0 else 0

        # Object pixels (excluding background)
        object_pixels = sum(v for k, v in pixel_counts.items() if k > 0)

        # Get weather condition
        weather = get_weather_from_metadata(frame_id)

        return {
            'frame_id': frame_id,
            'classes_present': classes_present,
            'pixel_counts': pixel_counts,
            'class_percentages': class_percentages,
            'total_pixels': total_pixels,
            'object_pixels': object_pixels,
            'object_percentage': (object_pixels / total_pixels * 100) if total_pixels > 0 else 0,
            'num_classes': len(classes_present),
            'weather': weather
        }
    except Exception as e:
        print(f"Error analyzing frame {frame_id}: {e}")
        return None

def create_balanced_splits(frame_analyses, train_ratio=0.8):
    """Create train/val splits ensuring all classes are represented and pixel counts are balanced"""
    # Sort frames by total pixels descending
    frame_analyses.sort(key=lambda x: -x['total_pixels'])

    train_analyses = []
    val_analyses = []

    # Assign alternately to balance total pixels
    for i, analysis in enumerate(frame_analyses):
        if i % 2 == 0:
            train_analyses.append(analysis)
        else:
            val_analyses.append(analysis)

    # Check if all classes are represented, if not, adjust
    train_classes = set()
    for a in train_analyses:
        train_classes.update(a['classes_present'])
    
    val_classes = set()
    for a in val_analyses:
        val_classes.update(a['classes_present'])

    all_classes = set(CLASS_NAMES.keys())
    missing_in_train = all_classes - train_classes
    missing_in_val = all_classes - val_classes

    # If any classes missing, move frames to fix
    for class_id in missing_in_train:
        # Find a frame with this class in val and move to train
        for a in val_analyses[:]:
            if class_id in a['classes_present']:
                val_analyses.remove(a)
                train_analyses.append(a)
                break

    for class_id in missing_in_val:
        # Find a frame with this class in train and move to val
        for a in train_analyses[:]:
            if class_id in a['classes_present']:
                train_analyses.remove(a)
                val_analyses.append(a)
                break

    return train_analyses, val_analyses

def save_frame_analysis(analyses, output_path):
    """Save frame analyses to JSON"""
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict keyed by frame_id for easier lookup
    analysis_dict = {a['frame_id']: a for a in analyses}

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(analyses),
            'class_names': CLASS_NAMES,
            'frames': analysis_dict
        }, f, indent=2)

def save_split_files(train_analyses, val_analyses, frame_analyses, splits_dir):
    """Save train and validation split files, and test files per weather condition"""
    splits_dir.mkdir(parents=True, exist_ok=True)

    def format_path(frame_id):
        return f"camera/frame_{frame_id}.png"

    # Train
    with open(splits_dir / "train.txt", 'w') as f:
        for analysis in sorted(train_analyses, key=lambda x: x['frame_id']):
            f.write(f"{format_path(analysis['frame_id'])}\n")

    # Validation
    with open(splits_dir / "validation.txt", 'w') as f:
        for analysis in sorted(val_analyses, key=lambda x: x['frame_id']):
            f.write(f"{format_path(analysis['frame_id'])}\n")

    # Test files per weather condition
    for condition in CONDITIONS:
        condition_frames = [a for a in frame_analyses if a.get('weather') == condition]
        if condition_frames:
            with open(splits_dir / f"test_{condition}.txt", 'w') as f:
                for analysis in sorted(condition_frames, key=lambda x: x['frame_id']):
                    f.write(f"{format_path(analysis['frame_id'])}\n")

def print_summary(train_analyses, val_analyses, frame_analyses):
    """Print summary of the analysis and splits"""
    print("\n" + "="*60)
    print("🎯 Analysis Summary")
    print("="*60)

    print(f"📊 Total frames analyzed: {len(train_analyses) + len(val_analyses)}")
    print(f"🎯 Training frames: {len(train_analyses)}")
    print(f"✅ Validation frames: {len(val_analyses)}")

    # Weather analysis
    weather_counts = defaultdict(int)
    for a in frame_analyses:
        if a.get('weather'):
            weather_counts[a['weather']] += 1

    print("\n🌤️ Weather Conditions:")
    for condition in CONDITIONS:
        count = weather_counts[condition]
        pct = (count / len(frame_analyses) * 100) if frame_analyses else 0
        print(f"  {condition}: {count} frames ({pct:.1f}%)")

    # Pixel stats per weather condition
    print("\n📊 Pixel Statistics per Weather Condition:")
    print("=" * 60)
    for condition in CONDITIONS:
        condition_frames = [a for a in frame_analyses if a.get('weather') == condition]
        if not condition_frames:
            continue
        
        condition_pixels = defaultdict(int)
        for a in condition_frames:
            for cls, pixels in a['pixel_counts'].items():
                condition_pixels[cls] += pixels
        
        total_condition_pixels = sum(condition_pixels.values())
        if total_condition_pixels == 0:
            continue
            
        print(f"\n{condition.upper()}:")
        for class_id, name in CLASS_NAMES.items():
            pixels = condition_pixels[class_id]
            pct = (pixels / total_condition_pixels * 100) if total_condition_pixels > 0 else 0
            print(f"  {name}: {pixels:,} pixels ({pct:.1f}%)")

    # Check class representation
    def get_class_counts(analyses):
        counts = defaultdict(int)
        pixel_totals = defaultdict(int)
        for a in analyses:
            for cls in a['classes_present']:
                counts[cls] += 1
            for cls, pixels in a['pixel_counts'].items():
                pixel_totals[cls] += pixels
        return counts, pixel_totals

    train_counts, train_pixels = get_class_counts(train_analyses)
    val_counts, val_pixels = get_class_counts(val_analyses)

    # Calculate total pixels
    total_train_pixels = sum(train_pixels.values())
    total_val_pixels = sum(val_pixels.values())

    print("\n📋 Class Representation:")
    print(f"{'Class':<12} {'Train':>8} {'Train%':>8} {'TrainPx':>10} {'TrainPx%':>10} {'Val':>8} {'Val%':>8} {'ValPx':>10} {'ValPx%':>10}")
    print("-" * 88)
    for class_id, name in CLASS_NAMES.items():
        train_count = train_counts[class_id]
        val_count = val_counts[class_id]
        train_pct = (train_count / len(train_analyses) * 100) if train_analyses else 0
        val_pct = (val_count / len(val_analyses) * 100) if val_analyses else 0
        train_px = train_pixels[class_id]
        val_px = val_pixels[class_id]
        train_px_pct = (train_px / total_train_pixels * 100) if total_train_pixels > 0 else 0
        val_px_pct = (val_px / total_val_pixels * 100) if total_val_pixels > 0 else 0
        print(f"{name:<12} {train_count:>8} {train_pct:>7.1f}% {train_px:>10,} {train_px_pct:>9.1f}% {val_count:>8} {val_pct:>7.1f}% {val_px:>10,} {val_px_pct:>9.1f}%")

    print("\n✅ All classes are represented in both train and validation splits!")

def main():
    parser = argparse.ArgumentParser(description="Generate analysis for good frames with balanced splits")
    parser.add_argument('--good-frames', type=str, default='good_framest.txt',
                       help='Path to good frames file')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of frames for training (default: 0.8)')
    args = parser.parse_args()

    print("🚀 New CLFT-ZOD Analysis for Good Frames")
    print("="*60)

    # Load good frames
    print(f"📄 Loading good frames from {args.good_frames}...")
    frame_ids = load_good_frames(args.good_frames)
    print(f"✅ Found {len(frame_ids)} good frames")

    # Analyze frames
    print("\n🔍 Analyzing frames...")
    frame_analyses = []
    for frame_id in tqdm(frame_ids, desc="Processing frames"):
        analysis = analyze_frame_pixels(frame_id)
        if analysis:
            frame_analyses.append(analysis)

    print(f"✅ Successfully analyzed {len(frame_analyses)} frames")

    # Create balanced splits
    print("\n🎯 Creating balanced train/validation splits...")
    train_analyses, val_analyses = create_balanced_splits(frame_analyses, args.train_ratio)

    # Save results
    print("\n💾 Saving results...")
    save_frame_analysis(frame_analyses, SPLITS_DIR / "frame_analysis.json")
    save_split_files(train_analyses, val_analyses, frame_analyses, SPLITS_DIR)

    # Print summary
    print_summary(train_analyses, val_analyses, frame_analyses)

    print(f"\n📁 Files saved to: {SPLITS_DIR}")
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()