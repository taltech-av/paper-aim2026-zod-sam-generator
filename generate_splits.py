#!/usr/bin/env python3
"""
CLFT-ZOD Dataset Analysis and Balanced Split Generation

This script performs comprehensive analysis of the CLFT-ZOD dataset by:
1. Analyzing pixel counts per class for pre-filtered "good" frames
2. Creating balanced train/validation splits ensuring all classes are represented
3. Generating weather-based test splits for robust evaluation
4. Providing detailed statistics and visualization frame selection

The analysis ensures balanced splits by alternating frame assignment and correcting
for class representation, while maintaining pixel count balance between splits.

REQUIRES: Run generate_camera_only_annotation.py first to create annotations!
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


# ===== CONFIGURATION =====
# Dataset paths and directories
OUTPUT_DIR = Path("/media/tom/ml/zod_temp")
CAMERA_ANNOTATION_DIR = OUTPUT_DIR / "annotation_camera_only"  # Source of annotations
SPLITS_DIR = OUTPUT_DIR / "splits_balanced"  # Output directory for splits
DATASET_ROOT = Path("/media/tom/ml/zod-data")  # ZOD dataset root

# Weather conditions for analysis and test splits
CONDITIONS = ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']

# Class mapping for CLFT annotations
CLASS_NAMES = {
    0: "background",  # Trainable negative samples
    1: "ignore",      # Loss-masked regions
    2: "vehicle",     # Cars, trucks, etc.
    3: "sign",        # Traffic signs
    4: "cyclist",     # Bicycles, motorcycles
    5: "pedestrian"   # People
}


def get_weather_from_metadata(frame_id):
    """
    Extract weather and time conditions from ZOD frame metadata.

    Parses the ZOD metadata JSON to determine weather conditions and time of day,
    which are used for creating weather-specific test splits and analysis.

    Args:
        frame_id: ZOD frame identifier string

    Returns:
        str or None: Weather condition string (e.g., 'day_fair', 'night_rain', 'snow')
                     or None if metadata unavailable or parsing fails
    """
    try:
        # Load ZOD metadata for this frame
        metadata_path = DATASET_ROOT / "single_frames" / frame_id / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            zod_metadata = json.load(f)

        # ===== WEATHER EXTRACTION =====
        # Parse scraped weather information from metadata
        scraped_weather = str(zod_metadata.get("scraped_weather", "")).lower()

        # Classify weather conditions based on keywords
        if "snow" in scraped_weather:
            weather = "snow"
        elif any(keyword in scraped_weather for keyword in ["rain", "sleet", "hail", "storm", "drizzle"]):
            weather = "rain"
        else:
            weather = "fair"  # Default to fair weather

        # ===== TIME OF DAY EXTRACTION =====
        # Determine if frame was captured during day or night
        time_of_day = str(zod_metadata.get("time_of_day", "day")).lower()
        timeofday = "night" if "night" in time_of_day else "day"

        # ===== CONDITION FORMATION =====
        # Combine time and weather into condition string
        if weather == "snow":
            condition = "snow"  # Snow overrides time (always snowy)
        else:
            condition = f"{timeofday}_{weather}"

        # Validate condition is in expected set
        return condition if condition in CONDITIONS else None

    except Exception:
        # Silently handle metadata parsing errors
        return None


def load_good_frames(good_frames_path):
    """
    Load frame IDs from the good frames file.

    Parses the good frames file to extract frame identifiers. The file contains
    paths like "camera/frame_099985.png" from which frame IDs are extracted.

    Args:
        good_frames_path: Path to file containing list of good frame paths

    Returns:
        list: List of frame ID strings (e.g., ['099985', '100001', ...])
    """
    frame_ids = []

    with open(good_frames_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract frame ID from path format: "camera/frame_XXXXXX.png"
                if '/' in line:
                    frame_id = line.split('/')[-1].replace('frame_', '').replace('.png', '')
                else:
                    # Handle simple frame ID format
                    frame_id = line.replace('frame_', '').replace('.png', '')
                frame_ids.append(frame_id)

    return frame_ids


def analyze_frame_pixels(frame_id):
    """
    Analyze pixel counts and class distribution for a single frame.

    Performs comprehensive pixel-level analysis of camera annotations including:
    - Per-class pixel counts and percentages
    - Class presence detection
    - Weather condition extraction
    - Object pixel statistics

    Args:
        frame_id: Frame identifier string

    Returns:
        dict or None: Analysis dictionary containing:
            - frame_id: Original frame identifier
            - classes_present: List of class IDs present in frame
            - pixel_counts: Dict of class_id -> pixel_count
            - class_percentages: Dict of class_id -> percentage
            - total_pixels: Total pixels in frame
            - object_pixels: Pixels belonging to objects (excluding background)
            - object_percentage: Percentage of object pixels
            - num_classes: Number of classes present
            - weather: Weather condition string or None
        Returns None if analysis fails
    """
    try:
        # Load camera annotation mask
        annotation_path = CAMERA_ANNOTATION_DIR / f"frame_{frame_id}.png"
        mask = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # ===== PIXEL COUNT ANALYSIS =====
        # Get unique classes and their pixel counts
        unique_classes, counts = np.unique(mask, return_counts=True)
        pixel_counts = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}

        # Identify which classes are present in this frame
        classes_present = [int(cls) for cls in unique_classes]

        # Calculate total pixels in frame
        total_pixels = sum(pixel_counts.values())

        # ===== PERCENTAGE CALCULATIONS =====
        # Calculate percentage for each class
        class_percentages = {}
        for class_id, count in pixel_counts.items():
            class_percentages[class_id] = (count / total_pixels * 100) if total_pixels > 0 else 0

        # ===== OBJECT STATISTICS =====
        # Calculate object pixels (excluding background class 0)
        object_pixels = sum(v for k, v in pixel_counts.items() if k > 0)
        object_percentage = (object_pixels / total_pixels * 100) if total_pixels > 0 else 0

        # ===== WEATHER EXTRACTION =====
        # Get weather condition from ZOD metadata
        weather = get_weather_from_metadata(frame_id)

        # ===== RETURN COMPREHENSIVE ANALYSIS =====
        return {
            'frame_id': frame_id,
            'classes_present': classes_present,
            'pixel_counts': pixel_counts,
            'class_percentages': class_percentages,
            'total_pixels': total_pixels,
            'object_pixels': object_pixels,
            'object_percentage': object_percentage,
            'num_classes': len(classes_present),
            'weather': weather
        }

    except Exception as e:
        print(f"Error analyzing frame {frame_id}: {e}")
        return None


def create_balanced_splits_per_weather(frame_analyses):
    """
    Create balanced train/validation/test splits per weather condition ensuring class representation and pixel balance.

    For each weather condition, splits frames into:
    - Train: 50%
    - Validation: 25%
    - Test: 25%

    Uses stratified splitting based on the dominant class (class with most pixels) to balance class frequencies.

    Args:
        frame_analyses: List of frame analysis dictionaries

    Returns:
        tuple: (train_analyses, val_analyses, test_analyses) - Lists of analysis dicts for each split
    """
    train_analyses = []
    val_analyses = []
    test_analyses = []

    # Group frames by weather condition
    weather_groups = defaultdict(list)
    for analysis in frame_analyses:
        weather = analysis.get('weather')
        if weather:
            weather_groups[weather].append(analysis)

    for condition, frames in weather_groups.items():
        if not frames:
            continue

        # Group frames by dominant class
        class_groups = defaultdict(list)
        for frame in frames:
            if frame['pixel_counts']:
                dominant_class = max(frame['pixel_counts'], key=frame['pixel_counts'].get)
                class_groups[dominant_class].append(frame)
            else:
                class_groups[0].append(frame)  # Default to background

        # For each class group, split proportionally
        for cls, group_frames in class_groups.items():
            random.shuffle(group_frames)  # Shuffle for randomness
            n_frames = len(group_frames)
            n_train = int(n_frames * 0.5)
            n_val = int(n_frames * 0.25)
            n_test = n_frames - n_train - n_val

            train_analyses.extend(group_frames[:n_train])
            val_analyses.extend(group_frames[n_train:n_train + n_val])
            test_analyses.extend(group_frames[n_train + n_val:])

    return train_analyses, val_analyses, test_analyses


def save_frame_analysis(analyses, output_path):
    """
    Save comprehensive frame analyses to JSON format.

    Creates a structured JSON file containing all frame analyses with metadata,
    class information, and timestamp for reproducibility.

    Args:
        analyses: List of frame analysis dictionaries
        output_path: Path where JSON file should be saved
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary keyed by frame_id for easier lookup
    analysis_dict = {a['frame_id']: a for a in analyses}

    # Save with metadata and class information
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(analyses),
            'class_names': CLASS_NAMES,
            'frames': analysis_dict
        }, f, indent=2)


def save_split_files(train_analyses, val_analyses, test_analyses, frame_analyses, splits_dir):
    """
    Save train/validation/test split files.

    Creates split files:
    - train.txt: Training frame paths
    - validation.txt: Validation frame paths
    - test.txt: Test frame paths
    - test_{condition}.txt: Test frames for each weather condition

    Args:
        train_analyses: List of training frame analyses
        val_analyses: List of validation frame analyses
        test_analyses: List of test frame analyses
        frame_analyses: Complete list of all frame analyses
        splits_dir: Directory where split files should be saved
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Helper function to format frame paths
    def format_path(frame_id):
        return f"camera/frame_{frame_id}.png"

    # ===== SAVE TRAINING SPLIT =====
    with open(splits_dir / "train.txt", 'w') as f:
        for analysis in sorted(train_analyses, key=lambda x: x['frame_id']):
            f.write(f"{format_path(analysis['frame_id'])}\n")

    # ===== SAVE VALIDATION SPLIT =====
    with open(splits_dir / "validation.txt", 'w') as f:
        for analysis in sorted(val_analyses, key=lambda x: x['frame_id']):
            f.write(f"{format_path(analysis['frame_id'])}\n")

    # ===== SAVE TEST SPLIT =====
    with open(splits_dir / "test.txt", 'w') as f:
        for analysis in sorted(test_analyses, key=lambda x: x['frame_id']):
            f.write(f"{format_path(analysis['frame_id'])}\n")

    # ===== SAVE WEATHER-BASED TEST SPLITS =====
    # Create separate test files for each weather condition
    for condition in CONDITIONS:
        condition_frames = [a for a in test_analyses if a.get('weather') == condition]
        if condition_frames:
            with open(splits_dir / f"test_{condition}.txt", 'w') as f:
                for analysis in sorted(condition_frames, key=lambda x: x['frame_id']):
                    f.write(f"{format_path(analysis['frame_id'])}\n")


def print_summary(train_analyses, val_analyses, test_analyses, frame_analyses):
    """
    Print comprehensive summary of dataset analysis and splits.

    Displays detailed statistics including:
    - Frame counts and weather distribution
    - Pixel statistics per weather condition
    - Class representation in train/validation/test splits
    - Balance verification

    Args:
        train_analyses: List of training frame analyses
        val_analyses: List of validation frame analyses
        test_analyses: List of test frame analyses
        frame_analyses: Complete list of all frame analyses
    """
    print("\n" + "="*60)
    print("🎯 Dataset Analysis Summary")
    print("="*60)

    print(f"📊 Total frames analyzed: {len(frame_analyses)}")
    print(f"🎯 Training frames: {len(train_analyses)} ({len(train_analyses)/len(frame_analyses)*100:.1f}%)")
    print(f"✅ Validation frames: {len(val_analyses)} ({len(val_analyses)/len(frame_analyses)*100:.1f}%)")
    print(f"🧪 Test frames: {len(test_analyses)} ({len(test_analyses)/len(frame_analyses)*100:.1f}%)")

    # ===== WEATHER DISTRIBUTION ANALYSIS =====
    weather_counts = defaultdict(int)
    for a in frame_analyses:
        if a.get('weather'):
            weather_counts[a['weather']] += 1

    print("\n🌤️ Weather Conditions Distribution:")
    for condition in CONDITIONS:
        count = weather_counts[condition]
        pct = (count / len(frame_analyses) * 100) if frame_analyses else 0
        print(f"  {condition}: {count} frames ({pct:.1f}%)")

    # ===== SPLIT DISTRIBUTION PER WEATHER =====
    print("\n📊 Split Distribution per Weather Condition:")
    for condition in CONDITIONS:
        condition_frames = [a for a in frame_analyses if a.get('weather') == condition]
        train_cond = [a for a in train_analyses if a.get('weather') == condition]
        val_cond = [a for a in val_analyses if a.get('weather') == condition]
        test_cond = [a for a in test_analyses if a.get('weather') == condition]
        if condition_frames:
            total = len(condition_frames)
            print(f"  {condition}: Train {len(train_cond)} ({len(train_cond)/total*100:.1f}%), Val {len(val_cond)} ({len(val_cond)/total*100:.1f}%), Test {len(test_cond)} ({len(test_cond)/total*100:.1f}%)")

    # ===== PIXEL STATISTICS PER WEATHER CONDITION =====
    print("\n📊 Pixel Statistics per Weather Condition:")
    print("=" * 60)
    for condition in CONDITIONS:
        condition_frames = [a for a in frame_analyses if a.get('weather') == condition]
        if not condition_frames:
            continue

        # Aggregate pixel counts across all frames in this condition
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

    # ===== CLASS REPRESENTATION VERIFICATION =====
    def get_class_counts(analyses):
        """Helper function to count class occurrences and pixel totals"""
        counts = defaultdict(int)      # Frame counts per class
        pixel_totals = defaultdict(int) # Total pixels per class
        for a in analyses:
            for cls in a['classes_present']:
                counts[cls] += 1
            for cls, pixels in a['pixel_counts'].items():
                pixel_totals[cls] += pixels
        return counts, pixel_totals

    train_counts, train_pixels = get_class_counts(train_analyses)
    val_counts, val_pixels = get_class_counts(val_analyses)
    test_counts, test_pixels = get_class_counts(test_analyses)

    # Calculate total pixels for percentage calculations
    total_train_pixels = sum(train_pixels.values())
    total_val_pixels = sum(val_pixels.values())
    total_test_pixels = sum(test_pixels.values())

    # ===== CLASS REPRESENTATION TABLE =====
    print("\n📋 Class Representation in Train/Validation/Test Splits:")
    header = f"{'Class':<12} {'Train':>8} {'Train%':>8} {'TrainPx':>10} {'TrainPx%':>10} {'Val':>8} {'Val%':>8} {'ValPx':>10} {'ValPx%':>10} {'Test':>8} {'Test%':>8} {'TestPx':>10} {'TestPx%':>10}"
    print(header)
    print("-" * len(header))

    for class_id, name in CLASS_NAMES.items():
        train_count = train_counts[class_id]
        val_count = val_counts[class_id]
        test_count = test_counts[class_id]
        train_pct = (train_count / len(train_analyses) * 100) if train_analyses else 0
        val_pct = (val_count / len(val_analyses) * 100) if val_analyses else 0
        test_pct = (test_count / len(test_analyses) * 100) if test_analyses else 0
        train_px = train_pixels[class_id]
        val_px = val_pixels[class_id]
        test_px = test_pixels[class_id]
        train_px_pct = (train_px / total_train_pixels * 100) if total_train_pixels > 0 else 0
        val_px_pct = (val_px / total_val_pixels * 100) if total_val_pixels > 0 else 0
        test_px_pct = (test_px / total_test_pixels * 100) if total_test_pixels > 0 else 0

        print(f"{name:<12} {train_count:>8} {train_pct:>7.1f}% {train_px:>10,} {train_px_pct:>9.1f}% {val_count:>8} {val_pct:>7.1f}% {val_px:>10,} {val_px_pct:>9.1f}% {test_count:>8} {test_pct:>7.1f}% {test_px:>10,} {test_px_pct:>9.1f}%")

    print("\n✅ Splits created with balanced class frequencies per weather condition!")


def print_detailed_class_analysis(train_analyses, val_analyses, test_analyses, frame_analyses, output_file):
    """
    Print detailed class distribution analysis for each split to a file.

    Shows frame counts and pixel distributions for train, validation, and test splits,
    plus weather-specific test splits.

    Args:
        train_analyses: List of training frame analyses
        val_analyses: List of validation frame analyses
        test_analyses: List of test frame analyses
        frame_analyses: Complete list of all frame analyses
        output_file: Path to output file for the analysis
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("📊 DETAILED CLASS DISTRIBUTION ANALYSIS\n")
        f.write("="*80 + "\n")

        def analyze_split_class_distribution(frames, split_name):
            f.write(f"\n📊 {split_name.upper()} CLASS DISTRIBUTION:\n")
            f.write("=" * 60 + "\n")

            total_frames = len(frames)
            f.write(f"Total frames: {total_frames}\n")

            # Class counts (frames containing each class)
            class_frame_counts = defaultdict(int)
            # Pixel totals per class
            class_pixel_totals = defaultdict(int)
            # Total pixels across all frames
            total_pixels = 0

            for analysis in frames:
                frame_id = analysis['frame_id']
                for cls in analysis['classes_present']:
                    class_frame_counts[cls] += 1
                for cls_str, pixels in analysis['pixel_counts'].items():
                    cls = int(cls_str)
                    class_pixel_totals[cls] += pixels
                total_pixels += analysis['total_pixels']

            # Sort classes by frame count
            sorted_classes = sorted(class_frame_counts.keys())

            f.write("\nFrame counts (frames containing each class):\n")
            f.write("Class | Frames | Percentage\n")
            f.write("-" * 30 + "\n")
            for cls in sorted_classes:
                count = class_frame_counts[cls]
                pct = (count / total_frames) * 100
                f.write(f"{cls:>5} | {count:>6} | {pct:>9.1f}%\n")

            f.write(f"\nPixel distribution (total pixels: {total_pixels:,}):\n")
            f.write("Class | Pixels | Percentage\n")
            f.write("-" * 30 + "\n")
            for cls in sorted_classes:
                pixels = class_pixel_totals[cls]
                pct = (pixels / total_pixels) * 100 if total_pixels > 0 else 0
                f.write(f"{cls:>5} | {pixels:>10,} | {pct:>9.1f}%\n")

        # Analyze main splits
        analyze_split_class_distribution(train_analyses, "train")
        analyze_split_class_distribution(val_analyses, "validation")
        analyze_split_class_distribution(test_analyses, "test")

        # Analyze weather-specific test splits
        f.write("\n" + "="*80 + "\n")
        f.write("🌤️ WEATHER-SPECIFIC TEST SPLIT DISTRIBUTIONS\n")
        f.write("="*80 + "\n")

        weather_groups = defaultdict(list)
        for analysis in test_analyses:
            weather = analysis.get('weather')
            if weather:
                weather_groups[weather].append(analysis)

        for condition in CONDITIONS:
            if condition in weather_groups:
                condition_name = condition.replace('_', ' ').title()
                analyze_split_class_distribution(weather_groups[condition], f"Test {condition_name}")

    print(f"📊 Detailed class analysis saved to: {output_file}")


def main():
    """
    Main function orchestrating the complete dataset analysis pipeline.

    Command-line interface for running the analysis with configurable parameters.
    """
    parser = argparse.ArgumentParser(description="Generate analysis for good frames with balanced splits")
    parser.add_argument('--good-frames', type=str, default='data/all.txt',
                       help='Path to good frames file')
    args = parser.parse_args()

    print("🚀 CLFT-ZOD Dataset Analysis and Balanced Split Generation")
    print("="*60)

    # ===== LOAD GOOD FRAMES =====
    print(f"📄 Loading good frames from {args.good_frames}...")
    frame_ids = load_good_frames(args.good_frames)
    print(f"✅ Found {len(frame_ids)} good frames")

    # ===== ANALYZE ALL FRAMES =====
    print("\n🔍 Analyzing frames...")
    frame_analyses = []
    for frame_id in tqdm(frame_ids, desc="Processing frames"):
        analysis = analyze_frame_pixels(frame_id)
        if analysis:
            frame_analyses.append(analysis)

    print(f"✅ Successfully analyzed {len(frame_analyses)} frames")

    # ===== SELECT VISUALIZATION FRAMES =====
    # Select best 2 frames per weather condition for visualization (total 10)
    print("\n🎨 Selecting best frames for visualization (2 per weather condition)...")
    selected_frames = []
    for condition in CONDITIONS:
        condition_frames = [a for a in frame_analyses if a.get('weather') == condition]
        if condition_frames:
            # Sort by object_pixels descending (frames with most objects first)
            condition_frames.sort(key=lambda x: -x['object_pixels'])
            # Take top 2 frames per condition
            selected_frames.extend(condition_frames[:2])
            print(f"  {condition}: selected {len(condition_frames[:2])} frames")

    print(f"✅ Selected {len(selected_frames)} frames for visualization")

    # Save selected frames to visualization.txt for easy access
    with open(SPLITS_DIR / "visualization.txt", 'w') as f:
        for analysis in sorted(selected_frames, key=lambda x: x['frame_id']):
            f.write(f"camera/frame_{analysis['frame_id']}.png\n")

    # ===== CREATE BALANCED SPLITS =====
    print("\n🎯 Creating balanced train/validation/test splits per weather condition...")
    train_analyses, val_analyses, test_analyses = create_balanced_splits_per_weather(frame_analyses)

    # ===== SAVE ALL RESULTS =====
    print("\n💾 Saving results...")
    save_frame_analysis(frame_analyses, SPLITS_DIR / "frame_analysis.json")
    save_split_files(train_analyses, val_analyses, test_analyses, frame_analyses, SPLITS_DIR)

    # ===== PRINT COMPREHENSIVE SUMMARY =====
    print_summary(train_analyses, val_analyses, test_analyses, frame_analyses)

    # ===== PRINT DETAILED CLASS ANALYSIS =====
    print_detailed_class_analysis(train_analyses, val_analyses, test_analyses, frame_analyses, SPLITS_DIR / "detailed_class_analysis.txt")

    print(f"\n📁 Files saved to: {SPLITS_DIR}")
    print("✅ Dataset analysis and split generation complete!")


if __name__ == "__main__":
    main()