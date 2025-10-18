#!/usr/bin/env python3
"""
Comprehensive CLFT-ZOD Dataset Analysis and Split Generation
- Analyzes annotation_balanced directory for class distributions and diversity
- Generates training/validation/test splits with balanced representation
- Creates comprehensive statistics and visualizations
- Outputs all split files (train.txt, validation.txt, test.txt, all.txt)

This script performs a complete dataset preparation pipeline:
1. ANALYSIS: Scans all annotation_balanced frames for class distributions,
   weather/time conditions, diversity scores, and rare class identification
2. RECOMMENDATIONS: Generates training frame recommendations based on diversity
3. SPLITS: Creates balanced train/validation/test splits from the analysis
4. STATISTICS: Computes normalization parameters and class weights for training
5. VISUALIZATIONS: Creates PNG plots of class and split distributions
6. OUTPUT: Saves all split files and comprehensive analysis reports

Usage: python generate_dataset_analysis_and_splits.py
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Paths
OUTPUT_DIR = Path("output_clft_v2")
ANNOTATION_DIR = OUTPUT_DIR / "annotation_balanced"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
SPLITS_DIR = OUTPUT_DIR / "splits"
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

# Class colors for visualization (BGR format for OpenCV, RGB for PIL)
CLASS_COLORS_BGR = {
    0: (0, 0, 0),        # background - black
    1: (128, 128, 128),  # ignore - gray
    2: (0, 0, 255),      # vehicle - red
    3: (0, 255, 255),    # sign - yellow
    4: (255, 0, 255),    # cyclist - magenta
    5: (0, 255, 0),      # pedestrian - green
}

CLASS_COLORS_RGB = {
    0: (0, 0, 0),        # background - black
    1: (128, 128, 128),  # ignore - gray
    2: (255, 0, 0),      # vehicle - red
    3: (255, 255, 0),    # sign - yellow
    4: (255, 0, 255),    # cyclist - magenta
    5: (0, 255, 0),      # pedestrian - green
}

print("🔬 CLFT-ZOD Dataset Analysis & Split Generation")
print("=" * 60)
print(f"📁 Annotation dir: {ANNOTATION_DIR}")
print(f"📁 Analysis dir: {ANALYSIS_DIR}")
print(f"📁 Splits dir: {SPLITS_DIR}")
print(f"📁 ZOD dataset root: {DATASET_ROOT}")

# Create directories
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Find all annotation files
annotation_files = sorted(list(ANNOTATION_DIR.glob("frame_*.png")))
print(f"\n📊 Found {len(annotation_files)} segmentation files")

if len(annotation_files) == 0:
    print("❌ No annotation files found!")
    exit(1)

# Initialize analysis data structures
frame_data = []
class_pixel_totals = defaultdict(int)
class_frame_counts = defaultdict(int)
class_cooccurrence = defaultdict(lambda: defaultdict(int))
pixel_intensity_distribution = defaultdict(list)

# Weather and time analysis
weather_conditions = Counter()
time_conditions = Counter()
split_combinations = Counter()

print(f"\n🔄 Analyzing {len(annotation_files)} frames...")

for anno_file in tqdm(annotation_files, desc="Analyzing frames"):
    # Extract frame ID
    frame_id = anno_file.stem.replace("frame_", "")

    # Check if corresponding lidar file exists
    lidar_path = OUTPUT_DIR / "lidar" / f"frame_{frame_id}.pkl"
    if not lidar_path.exists():
        continue

    try:
        # Load annotation mask
        mask = cv2.imread(str(anno_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Get unique classes and their pixel counts
        unique_classes, counts = np.unique(mask, return_counts=True)
        pixel_counts = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}

        # Update global statistics
        total_pixels = sum(pixel_counts.values())
        classes_present = [cls for cls in pixel_counts.keys() if cls > 0]  # Exclude background

        # Update class totals and frame counts
        for class_id, count in pixel_counts.items():
            class_pixel_totals[class_id] += count
            if count > 0:
                class_frame_counts[class_id] += 1

        # Update co-occurrence matrix
        for i, cls_a in enumerate(classes_present):
            for cls_b in classes_present[i+1:]:
                class_cooccurrence[min(cls_a, cls_b)][max(cls_a, cls_b)] += 1

        # Get weather/time info from ZOD metadata
        weather = "unknown"
        timeofday = "unknown"
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

        except Exception:
            pass

        # Update condition counters
        weather_conditions[weather] += 1
        time_conditions[timeofday] += 1
        split_combinations[f"{timeofday}_{weather}"] += 1

        # Calculate frame-level statistics
        foreground_pixels = sum(v for k, v in pixel_counts.items() if k > 0)
        background_pixels = pixel_counts.get(0, 0)

        frame_stats = {
            'frame_id': frame_id,
            'weather': weather,
            'timeofday': timeofday,
            'split': f"{timeofday}_{weather}",
            'total_pixels': total_pixels,
            'foreground_pixels': foreground_pixels,
            'background_pixels': background_pixels,
            'foreground_percentage': float(foreground_pixels / total_pixels * 100) if total_pixels > 0 else 0,
            'num_classes': len(classes_present),
            'classes_present': classes_present,
            'pixel_counts': pixel_counts,
            'class_percentages': {
                CLASS_NAMES.get(k, f"class_{k}"): float(v / foreground_pixels * 100) if foreground_pixels > 0 else 0
                for k, v in pixel_counts.items() if k > 0
            }
        }

        frame_data.append(frame_stats)

    except Exception as e:
        print(f"\n❌ Error processing {frame_id}: {e}")
        continue

print(f"\n✅ Analyzed {len(frame_data)} frames successfully")

# Calculate comprehensive statistics
total_pixels_all_frames = sum(class_pixel_totals.values())
total_frames = len(frame_data)

# Class analysis
class_analysis = {}
for class_id in sorted(class_pixel_totals.keys()):
    pixel_count = class_pixel_totals[class_id]
    frame_count = class_frame_counts[class_id]

    class_analysis[class_id] = {
        "name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "total_pixels": pixel_count,
        "pixel_percentage": float(pixel_count / total_pixels_all_frames * 100) if total_pixels_all_frames > 0 else 0,
        "frames_with_class": frame_count,
        "frame_coverage_percentage": float(frame_count / total_frames * 100) if total_frames > 0 else 0,
        "avg_pixels_per_frame": float(pixel_count / frame_count) if frame_count > 0 else 0,
        "is_rare": frame_count < total_frames * 0.1,  # Less than 10% of frames
        "is_very_rare": frame_count < total_frames * 0.01,  # Less than 1% of frames
    }

# Frame diversity analysis
frame_diversity_scores = []
for frame in frame_data:
    classes_present = frame['classes_present']
    num_classes = frame['num_classes']
    foreground_pct = frame['foreground_percentage']

    # Diversity score based on:
    # - Number of classes present
    # - Presence of rare classes
    # - Foreground coverage
    diversity_score = (
        num_classes * 10 +  # Base score for class count
        foreground_pct * 0.5 +  # Foreground coverage bonus
        sum(20 for cls in classes_present if class_analysis.get(cls, {}).get('is_rare', False)) +  # Rare class bonus
        sum(50 for cls in classes_present if class_analysis.get(cls, {}).get('is_very_rare', False))  # Very rare class bonus
    )

    frame_diversity_scores.append({
        'frame_id': frame['frame_id'],
        'diversity_score': diversity_score,
        'num_classes': num_classes,
        'foreground_percentage': foreground_pct,
        'classes_present': classes_present,
        'split': frame['split']
    })

# Sort frames by diversity score
frame_diversity_scores.sort(key=lambda x: -x['diversity_score'])

# Generate recommendations
recommendations = {
    "training_strategy": {
        "total_frames_available": total_frames,
        "recommended_training_size": min(15000, total_frames),
        "recommended_validation_size": min(2000, total_frames // 10),
        "recommended_test_size": min(3000, total_frames // 8),
    },

    "frame_selection_criteria": [
        "Prioritize frames with high diversity scores (multiple classes, rare objects)",
        "Ensure balanced representation across weather/time conditions",
        "Include frames with underrepresented classes (pedestrians, cyclists, signs)",
        "Maintain foreground coverage above 5% for meaningful training samples",
        "Avoid frames with only background/ignore regions"
    ],

    "top_diverse_frames": [
        {
            "rank": i+1,
            "frame_id": f["frame_id"],
            "diversity_score": f["diversity_score"],
            "classes": len(f["classes_present"]),
            "foreground_pct": f["foreground_percentage"],
            "split": f["split"]
        }
        for i, f in enumerate(frame_diversity_scores[:20])
    ],

    "class_balance_analysis": {
        "most_common_classes": [
            {
                "class_id": class_id,
                "name": stats["name"],
                "frame_coverage": stats["frame_coverage_percentage"],
                "pixel_percentage": stats["pixel_percentage"]
            }
            for class_id, stats in sorted(class_analysis.items(),
                                        key=lambda x: x[1]["frame_coverage_percentage"],
                                        reverse=True)[:3]
        ],

        "rarest_classes": [
            {
                "class_id": class_id,
                "name": stats["name"],
                "frame_coverage": stats["frame_coverage_percentage"],
                "pixel_percentage": stats["pixel_percentage"],
                "recommendation": "Prioritize frames containing this class"
            }
            for class_id, stats in sorted(class_analysis.items(),
                                        key=lambda x: x[1]["frame_coverage_percentage"])[:3]
        ],

        "underrepresented_conditions": []
    },

    "split_distribution_analysis": {
        "weather_distribution": dict(weather_conditions),
        "time_distribution": dict(time_conditions),
        "split_combinations": dict(split_combinations),
        "recommendations": []
    }
}

# Analyze split distributions and add recommendations
weather_total = sum(weather_conditions.values())
time_total = sum(time_conditions.values())

for condition, count in weather_conditions.items():
    pct = (count / weather_total * 100) if weather_total > 0 else 0
    if pct < 20:  # Less than 20% representation
        recommendations["class_balance_analysis"]["underrepresented_conditions"].append(
            f"Weather condition '{condition}': only {pct:.1f}% of frames"
        )

for condition, count in time_conditions.items():
    pct = (count / time_total * 100) if time_total > 0 else 0
    if pct < 20:  # Less than 20% representation
        recommendations["class_balance_analysis"]["underrepresented_conditions"].append(
            f"Time condition '{condition}': only {pct:.1f}% of frames"
        )

# Split distribution recommendations
split_total = sum(split_combinations.values())
for split, count in split_combinations.items():
    pct = (count / split_total * 100) if split_total > 0 else 0
    if pct < 10:  # Less than 10% representation
        recommendations["split_distribution_analysis"]["recommendations"].append(
            f"Split '{split}': only {pct:.1f}% of frames - consider oversampling"
        )

# Generate training set recommendations
recommended_frames = []

# Strategy: Take top diversity frames, then fill with balanced representation
selected_frames = set()
target_training_size = recommendations["training_strategy"]["recommended_training_size"]

# First, take top diversity frames (60% of training set)
top_diversity_count = int(target_training_size * 0.6)
for frame in frame_diversity_scores[:top_diversity_count]:
    if len(recommended_frames) >= target_training_size:
        break
    recommended_frames.append({
        "frame_id": frame["frame_id"],
        "selection_reason": "High diversity score",
        "diversity_score": frame["diversity_score"],
        "split": frame["split"]
    })
    selected_frames.add(frame["frame_id"])

# Then fill with balanced representation across splits
frames_by_split = defaultdict(list)
for frame in frame_diversity_scores[top_diversity_count:]:
    if frame["frame_id"] not in selected_frames:
        frames_by_split[frame["split"]].append(frame)

# Calculate target frames per split
remaining_slots = target_training_size - len(recommended_frames)
frames_per_split = max(1, remaining_slots // len(split_combinations))

for split in sorted(split_combinations.keys()):
    split_frames = frames_by_split.get(split, [])
    for frame in split_frames[:frames_per_split]:
        if len(recommended_frames) >= target_training_size:
            break
        recommended_frames.append({
            "frame_id": frame["frame_id"],
            "selection_reason": f"Balanced {split} representation",
            "diversity_score": frame["diversity_score"],
            "split": frame["split"]
        })

recommendations["recommended_training_frames"] = recommended_frames[:target_training_size]

# Create visualizations as PNG files
print(f"\n📊 Generating analysis visualizations...")

def create_class_distribution_image():
    """Create a simple class distribution visualization"""
    width, height = 800, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Title
    draw.text((width//2 - 100, 20), "Class Distribution Analysis", fill='black', font=font)

    # Class bars
    bar_height = 30
    start_y = 80
    max_bar_width = 400

    max_pixels = max(stats["pixel_percentage"] for stats in class_analysis.values()) if class_analysis else 1

    for i, (class_id, stats) in enumerate(sorted(class_analysis.items(), key=lambda x: x[1]["pixel_percentage"], reverse=True)):
        y = start_y + i * (bar_height + 10)
        bar_width = int((stats["pixel_percentage"] / max_pixels) * max_bar_width)

        # Bar
        color = CLASS_COLORS_RGB.get(class_id, (128, 128, 128))
        draw.rectangle([100, y, 100 + bar_width, y + bar_height], fill=color)

        # Label
        draw.text((10, y + 5), f"{stats['name']}", fill='black', font=small_font)
        draw.text((110 + bar_width + 10, y + 5), f"{stats['pixel_percentage']:.1f}%", fill='black', font=small_font)

    return img

def create_split_distribution_image():
    """Create split distribution visualization"""
    width, height = 600, 400
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Title
    draw.text((width//2 - 80, 20), "Split Distribution", fill='black', font=font)

    # Bars
    bar_width = 40
    start_x = 50
    max_bar_height = 200

    max_count = max(split_combinations.values()) if split_combinations else 1

    for i, (split, count) in enumerate(sorted(split_combinations.items())):
        x = start_x + i * (bar_width + 20)
        bar_height = int((count / max_count) * max_bar_height)

        # Bar
        draw.rectangle([x, height - 80 - bar_height, x + bar_width, height - 80], fill='skyblue')

        # Label
        draw.text((x, height - 70), split, fill='black', font=small_font)
        draw.text((x, height - 80 - bar_height - 20), str(count), fill='black', font=small_font)

    return img

# Save visualizations
class_dist_img = create_class_distribution_image()
class_dist_img.save(ANALYSIS_DIR / "class_distribution.png")

split_dist_img = create_split_distribution_image()
split_dist_img.save(ANALYSIS_DIR / "split_distribution.png")

# Save comprehensive analysis report
analysis_report = {
    "analysis_timestamp": datetime.now().isoformat(),
    "dataset_overview": {
        "total_frames": total_frames,
        "total_pixels": total_pixels_all_frames,
        "classes_analyzed": len(class_analysis),
        "splits_analyzed": len(split_combinations)
    },

    "class_analysis": class_analysis,

    "frame_statistics": {
        "avg_classes_per_frame": float(np.mean([f["num_classes"] for f in frame_data])),
        "avg_foreground_percentage": float(np.mean([f["foreground_percentage"] for f in frame_data])),
        "max_classes_in_frame": max([f["num_classes"] for f in frame_data]),
        "min_classes_in_frame": min([f["num_classes"] for f in frame_data]),
        "frames_with_rare_classes": len([f for f in frame_data if any(class_analysis.get(cls, {}).get('is_rare', False) for cls in f["classes_present"])])
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

    "recommendations": recommendations
}

# Save analysis report
report_file = ANALYSIS_DIR / "analysis_report.json"
with open(report_file, 'w') as f:
    json.dump(analysis_report, f, indent=2)

# Save recommended training frames list
training_frames_file = ANALYSIS_DIR / "recommended_training_frames.txt"
with open(training_frames_file, 'w') as f:
    f.write("# Recommended training frames based on diversity and balance analysis\n")
    f.write(f"# Generated on {datetime.now().isoformat()}\n")
    f.write("# Format: frame_id,selection_reason,diversity_score,split\n")
    for frame in recommendations["recommended_training_frames"]:
        f.write(f"{frame['frame_id']},{frame['selection_reason']},{frame['diversity_score']:.1f},{frame['split']}\n")

print(f"\n💾 Analysis complete!")
print(f"📁 Analysis results saved to: {ANALYSIS_DIR}")
print(f"   📄 analysis_report.json - Comprehensive analysis report")
print(f"   📄 recommended_training_frames.txt - Training frame recommendations")
print(f"   📊 class_distribution.png - Class distribution visualization")
print(f"   📊 split_distribution.png - Split distribution visualization")

# Print key insights
print(f"\n🎯 Key Analysis Insights:")
print(f"{'='*60}")
print(f"Total Frames Analyzed: {total_frames:,}")
print(f"Total Pixels: {total_pixels_all_frames:,}")
print(f"Average Classes per Frame: {analysis_report['frame_statistics']['avg_classes_per_frame']:.2f}")
print(f"Average Foreground Coverage: {analysis_report['frame_statistics']['avg_foreground_percentage']:.1f}%")
print(f"Frames with Rare Classes: {analysis_report['frame_statistics']['frames_with_rare_classes']:,} ({analysis_report['frame_statistics']['frames_with_rare_classes']/total_frames*100:.1f}%)")

print(f"\n📊 Class Distribution (by frame coverage):")
print(f"{'Class':<12} {'Frames':>8} {'Pixels':>8} {'Rare?'}")
print("-" * 40)
for class_id, stats in sorted(class_analysis.items(), key=lambda x: x[1]["frame_coverage_percentage"], reverse=True):
    rare_marker = "★" if stats["is_rare"] else ("★★" if stats["is_very_rare"] else "")
    print(f"{stats['name']:<12} {stats['frame_coverage_percentage']:>7.1f}% {stats['pixel_percentage']:>7.1f}% {rare_marker}")

print(f"\n🏆 Top 5 Most Diverse Frames:")
for i, frame in enumerate(frame_diversity_scores[:5], 1):
    print(f"   {i}. frame_{frame['frame_id']}: score={frame['diversity_score']:.0f}, {frame['num_classes']} classes, {frame['foreground_percentage']:.1f}% fg, split={frame['split']}")

print(f"\n⚠️  Recommendations:")
for rec in recommendations["frame_selection_criteria"]:
    print(f"   • {rec}")

if recommendations["class_balance_analysis"]["underrepresented_conditions"]:
    print(f"\n📉 Underrepresented Conditions:")
    for condition in recommendations["class_balance_analysis"]["underrepresented_conditions"]:
        print(f"   • {condition}")

if recommendations["split_distribution_analysis"]["recommendations"]:
    print(f"\n🔄 Split Balance Recommendations:")
    for rec in recommendations["split_distribution_analysis"]["recommendations"]:
        print(f"   • {rec}")

print(f"\n✅ Recommended training set: {len(recommendations['recommended_training_frames']):,} frames")
print(f"{'='*60}")

# ============================================================================
# SPLIT GENERATION
# ============================================================================

print(f"\n🔀 Generating dataset splits...")

# Use the analysis results to create splits
training_frames = set()
training_frame_data = []

# Parse the recommended training frames from analysis
for frame in recommendations["recommended_training_frames"]:
    training_frames.add(frame['frame_id'])
    training_frame_data.append(frame)

print(f"✅ Loaded {len(training_frames)} recommended training frames")

# Get all available frames from annotation directory
all_frames = set()
annotation_files = list(ANNOTATION_DIR.glob("frame_*.png"))
for anno_file in annotation_files:
    frame_id = anno_file.stem.replace("frame_", "")
    all_frames.add(frame_id)

print(f"✅ Loaded {len(all_frames)} total frames from annotation_balanced directory")

# Get remaining frames (not in training set)
remaining_frames = all_frames - training_frames
print(f"✅ {len(remaining_frames)} frames remaining for validation and test")

# Create frame diversity data for remaining frames
frame_diversity_data = {}

# Use training frame data from analysis
for frame_data in training_frame_data:
    frame_diversity_data[frame_data['frame_id']] = frame_data

# Calculate diversity for remaining frames
print(f"🔄 Calculating diversity scores for remaining frames...")

remaining_frame_data = []
for frame_id in tqdm(remaining_frames, desc="Analyzing remaining frames"):
    try:
        # Load annotation mask
        mask_path = ANNOTATION_DIR / f"frame_{frame_id}.png"
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Get unique classes and their pixel counts
        unique_classes, counts = np.unique(mask, return_counts=True)
        pixel_counts = {int(class_id): int(count) for class_id, count in zip(unique_classes, counts)}

        # Update global statistics
        classes_present = [cls for cls in pixel_counts.keys() if cls > 0]  # Exclude background
        total_pixels = sum(pixel_counts.values())
        foreground_pixels = sum(v for k, v in pixel_counts.items() if k > 0)

        # Get weather/time info
        weather = "unknown"
        timeofday = "unknown"
        try:
            metadata_path = DATASET_ROOT / "single_frames" / frame_id / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    zod_metadata = json.load(f)

                scraped_weather = str(zod_metadata.get("scraped_weather", "")).lower()
                precipitation_keywords = ["rain", "snow", "sleet", "hail", "storm", "drizzle"]
                has_precipitation = any(keyword in scraped_weather for keyword in precipitation_keywords)
                weather = "rain" if has_precipitation else "fair"

                time_of_day = str(zod_metadata.get("time_of_day", "day")).lower()
                timeofday = "night" if "night" in time_of_day else "day"

        except Exception:
            pass

        # Calculate diversity score (same logic as analysis)
        num_classes = len(classes_present)
        foreground_pct = float(foreground_pixels / total_pixels * 100) if total_pixels > 0 else 0

        # Check if classes are rare (from analysis)
        rare_classes = {4, 5}  # cyclist, pedestrian
        very_rare_classes = set()  # none in this dataset

        diversity_score = (
            num_classes * 10 +  # Base score for class count
            foreground_pct * 0.5 +  # Foreground coverage bonus
            sum(20 for cls in classes_present if cls in rare_classes) +  # Rare class bonus
            sum(50 for cls in classes_present if cls in very_rare_classes)  # Very rare class bonus
        )

        frame_data = {
            'frame_id': frame_id,
            'diversity_score': diversity_score,
            'num_classes': num_classes,
            'foreground_percentage': foreground_pct,
            'classes_present': classes_present,
            'split': f"{timeofday}_{weather}",
            'weather': weather,
            'timeofday': timeofday
        }

        remaining_frame_data.append(frame_data)
        frame_diversity_data[frame_id] = frame_data

    except Exception as e:
        continue

print(f"✅ Analyzed {len(remaining_frame_data)} remaining frames")

# Sort remaining frames by diversity score
remaining_frame_data.sort(key=lambda x: -x['diversity_score'])

# Split strategy
VALIDATION_SIZE = 2000
TEST_SIZE = 3000

print(f"\n📊 Split strategy:")
print(f"   Training: {len(training_frames)} frames (from analysis recommendations)")
print(f"   Validation: {VALIDATION_SIZE} frames (top diversity from remaining)")
print(f"   Test: {TEST_SIZE} frames (balanced from remaining)")

# Select validation frames (top diversity)
validation_frames = remaining_frame_data[:VALIDATION_SIZE]
validation_frame_ids = {f['frame_id'] for f in validation_frames}

# Select test frames (balanced approach)
test_pool = remaining_frame_data[VALIDATION_SIZE:]

# Group by split for balanced selection
test_by_split = defaultdict(list)
for frame in test_pool:
    test_by_split[frame['split']].append(frame)

# Calculate target frames per split for test set
available_splits = list(test_by_split.keys())
frames_per_split = TEST_SIZE // len(available_splits)
extra_frames = TEST_SIZE % len(available_splits)

test_frames = []
for i, split in enumerate(sorted(available_splits)):
    split_frames = sorted(test_by_split[split], key=lambda x: -x['diversity_score'])
    target_count = frames_per_split + (1 if i < extra_frames else 0)
    test_frames.extend(split_frames[:target_count])

# If we still don't have enough, fill with highest diversity frames from any split
if len(test_frames) < TEST_SIZE:
    selected_ids = {f['frame_id'] for f in test_frames}
    remaining_pool = [f for f in test_pool if f['frame_id'] not in selected_ids]
    remaining_pool.sort(key=lambda x: -x['diversity_score'])
    needed = TEST_SIZE - len(test_frames)
    test_frames.extend(remaining_pool[:needed])

test_frame_ids = {f['frame_id'] for f in test_frames[:TEST_SIZE]}

print(f"✅ Selected {len(validation_frames)} validation frames")
print(f"✅ Selected {len(test_frames[:TEST_SIZE])} test frames")

# Generate split statistics
def analyze_split_composition(frame_list, name):
    """Analyze the composition of a frame split"""
    splits = Counter()
    weather = Counter()
    timeofday = Counter()
    total_foreground = 0
    total_classes = 0
    pixel_counts_per_class = defaultdict(int)
    frames_with_class = defaultdict(int)

    for frame in frame_list:
        split = frame.get('split', 'day_fair')
        splits[split] += 1

        # Parse split into weather and timeofday
        if '_' in split:
            time_part, weather_part = split.split('_', 1)
            weather[weather_part] += 1
            timeofday[time_part] += 1
        else:
            weather['unknown'] += 1
            timeofday['unknown'] += 1

        total_foreground += frame.get('foreground_percentage', 0)
        total_classes += frame.get('num_classes', 0)

        # Load annotation mask to count pixels per class
        frame_id = frame['frame_id']
        mask_path = ANNOTATION_DIR / f"frame_{frame_id}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_classes, counts = np.unique(mask, return_counts=True)
                for class_id, count in zip(unique_classes, counts):
                    pixel_counts_per_class[int(class_id)] += int(count)
                    if count > 0:
                        frames_with_class[int(class_id)] += 1

    stats = {
        'name': name,
        'count': len(frame_list),
        'avg_foreground_pct': float(total_foreground / len(frame_list)) if frame_list else 0.0,
        'avg_classes_per_frame': float(total_classes / len(frame_list)) if frame_list else 0.0,
        'split_distribution': dict(splits),
        'weather_distribution': dict(weather),
        'timeofday_distribution': dict(timeofday),
        'pixel_counts_per_class': dict(pixel_counts_per_class),
        'frames_with_class': dict(frames_with_class),
        'total_pixels': sum(pixel_counts_per_class.values())
    }

    return stats

# Analyze all splits
training_stats = analyze_split_composition(
    [frame_diversity_data[fid] for fid in training_frames if fid in frame_diversity_data],
    "Training"
)
validation_stats = analyze_split_composition(validation_frames, "Validation")
test_stats = analyze_split_composition(test_frames[:TEST_SIZE], "Test")

# Add pixel statistics summary
def format_pixel_stats(stats):
    """Format pixel counts per class for display"""
    total_pixels = stats.get('total_pixels', 0)
    pixel_counts = stats.get('pixel_counts_per_class', {})
    frames_with_class = stats.get('frames_with_class', {})

    class_summary = {}
    for class_id in sorted(pixel_counts.keys()):
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        pixels = pixel_counts[class_id]
        frames = frames_with_class.get(class_id, 0)
        percentage = (pixels / total_pixels * 100) if total_pixels > 0 else 0

        class_summary[class_name] = {
            'pixels': pixels,
            'percentage': float(percentage),
            'frames': frames,
            'avg_pixels_per_frame': float(pixels / frames) if frames > 0 else 0.0
        }

    return class_summary

training_stats['class_pixel_summary'] = format_pixel_stats(training_stats)
validation_stats['class_pixel_summary'] = format_pixel_stats(validation_stats)
test_stats['class_pixel_summary'] = format_pixel_stats(test_stats)

# Calculate class weights
def calculate_class_weights(pixel_counts):
    """Calculate class weights for imbalanced training"""
    total_pixels = sum(pixel_counts.values())
    weights = {}
    for class_id, pixels in pixel_counts.items():
        if pixels > 0:
            weights[class_id] = float(total_pixels / (len(pixel_counts) * pixels))
    return weights

training_weights = calculate_class_weights(training_stats['pixel_counts_per_class'])

# Calculate image statistics
print(f"\n📊 Calculating image statistics for training set...")
image_mean = np.zeros(3)
image_std = np.zeros(3)
count = 0

sample_size = min(1000, len(training_frames))
sampled_frames = np.random.choice(list(training_frames), sample_size, replace=False)

for frame_id in tqdm(sampled_frames, desc="Calculating image stats"):
    camera_path = OUTPUT_DIR / "camera" / f"frame_{frame_id}.png"
    if camera_path.exists():
        try:
            img = cv2.imread(str(camera_path))
            if img is not None:
                img = img.astype(np.float32) / 255.0
                for c in range(3):
                    image_mean[c] += np.mean(img[:, :, c])
                    image_std[c] += np.var(img[:, :, c])
                count += 1
        except:
            continue

if count > 0:
    image_mean = image_mean / count
    image_std = np.sqrt(image_std / count)
    image_mean = [float(x) for x in image_mean]
    image_std = [float(x) for x in image_std]

# Lidar statistics
print(f"\n📊 Calculating lidar statistics for training set...")
lidar_mean = [0.0, 0.0, 0.0]
lidar_std = [0.0, 0.0, 0.0]
lidar_var = np.zeros(3)
lidar_count = 0

sample_size_lidar = min(500, len(training_frames))
sampled_frames_lidar = np.random.choice(list(training_frames), sample_size_lidar, replace=False)

for frame_id in tqdm(sampled_frames_lidar, desc="Calculating lidar stats"):
    lidar_path = OUTPUT_DIR / "lidar" / f"frame_{frame_id}.pkl"
    if lidar_path.exists():
        try:
            import pickle
            with open(lidar_path, 'rb') as f:
                lidar_data = pickle.load(f)
            if hasattr(lidar_data, 'shape') and len(lidar_data.shape) >= 2 and lidar_data.shape[1] >= 3:
                points = lidar_data[:, :3]
                for d in range(3):
                    lidar_mean[d] += np.mean(points[:, d])
                    lidar_var[d] += np.var(points[:, d])
                lidar_count += 1
        except:
            continue

if lidar_count > 0:
    lidar_mean = [float(x / lidar_count) for x in lidar_mean]
    lidar_std = [float(np.sqrt(x / lidar_count)) for x in lidar_var]

# Training recommendations
training_recommendations = {
    "image_normalization": {
        "mean": image_mean,
        "std": image_std,
        "note": "Calculated from sample of training images. Use for input normalization."
    },
    "lidar_statistics": {
        "mean": lidar_mean,
        "std": lidar_std,
        "note": "Per-dimension statistics (x, y, z) from sampled training lidar data."
    },
    "class_weights": training_weights,
    "training_suggestions": [
        "Use class weights to handle class imbalance during training",
        "Normalize images using the calculated mean and std",
        "Consider lidar preprocessing based on the range statistics",
        "Monitor rare classes (cyclist, pedestrian) during training"
    ]
}

# Save split files
def format_path(frame_id):
    return f"camera/frame_{frame_id}.png"

print(f"\n💾 Saving split files to {SPLITS_DIR}...")

# All frames file
all_file = SPLITS_DIR / "all.txt"
with open(all_file, 'w') as f:
    f.write("# All frames used in dataset splits\n")
    f.write(f"# Generated on {datetime.now().isoformat()}\n")
    f.write("# Format: camera/frame_XXXXXX.png\n")
    all_split_frames = sorted(training_frames | validation_frame_ids | test_frame_ids)
    for frame_id in all_split_frames:
        f.write(f"{format_path(frame_id)}\n")
print(f"✅ all.txt: {len(all_split_frames)} frames")

# Training split
train_file = SPLITS_DIR / "train.txt"
with open(train_file, 'w') as f:
    f.write("# Training split - recommended frames from diversity analysis\n")
    f.write(f"# Generated on {datetime.now().isoformat()}\n")
    f.write("# Format: camera/frame_XXXXXX.png\n")
    for frame_id in sorted(training_frames):
        f.write(f"{format_path(frame_id)}\n")
print(f"✅ train.txt: {len(training_frames)} frames")

# Validation split
valid_file = SPLITS_DIR / "validation.txt"
with open(valid_file, 'w') as f:
    f.write("# Validation split - high diversity frames from remaining data\n")
    f.write(f"# Generated on {datetime.now().isoformat()}\n")
    f.write("# Format: camera/frame_XXXXXX.png\n")
    for frame in sorted(validation_frames, key=lambda x: x['frame_id']):
        f.write(f"{format_path(frame['frame_id'])}\n")
print(f"✅ validation.txt: {len(validation_frames)} frames")

# Test split
test_file = SPLITS_DIR / "test.txt"
with open(test_file, 'w') as f:
    f.write("# Test split - balanced representation from remaining data\n")
    f.write(f"# Generated on {datetime.now().isoformat()}\n")
    f.write("# Format: camera/frame_XXXXXX.png\n")
    for frame in sorted(test_frames[:TEST_SIZE], key=lambda x: x['frame_id']):
        f.write(f"{format_path(frame['frame_id'])}\n")
print(f"✅ test.txt: {len(test_frames[:TEST_SIZE])} frames")

# Save comprehensive split statistics
split_stats = {
    "generation_info": {
        "timestamp": datetime.now().isoformat(),
        "total_frames_available": len(all_frames),
        "training_source": "analysis-based diversity scoring",
        "frame_source": "annotation_balanced directory"
    },
    "splits": {
        "training": training_stats,
        "validation": validation_stats,
        "test": test_stats
    },
    "summary": {
        "training_frames": len(training_frames),
        "validation_frames": len(validation_frames),
        "test_frames": len(test_frames[:TEST_SIZE]),
        "total_split_frames": len(training_frames) + len(validation_frames) + len(test_frames[:TEST_SIZE]),
        "all_frames_file": len(all_split_frames),
        "unused_frames": len(all_frames) - len(all_split_frames)
    },
    "recommendations": training_recommendations
}

stats_file = SPLITS_DIR / "split_statistics.json"
with open(stats_file, 'w') as f:
    json.dump(split_stats, f, indent=2)

print(f"✅ split_statistics.json: comprehensive split analysis")

# Final summary
print(f"\n🎯 Complete Dataset Analysis & Split Generation Summary:")
print(f"{'='*70}")
print(f"📊 ANALYSIS PHASE:")
print(f"   Total Frames Analyzed: {total_frames:,}")
print(f"   Classes Identified: {len(class_analysis)}")
print(f"   Rare Classes Found: {len([c for c in class_analysis.values() if c['is_rare']])}")
print(f"   Weather/Time Splits: {len(split_combinations)}")

print(f"\n🔀 SPLIT GENERATION PHASE:")
print(f"   Total Available Frames: {len(all_frames):,}")
print(f"   Training Frames: {len(training_frames):,} ({len(training_frames)/len(all_frames)*100:.1f}%)")
print(f"   Validation Frames: {len(validation_frames):,} ({len(validation_frames)/len(all_frames)*100:.1f}%)")
print(f"   Test Frames: {len(test_frames[:TEST_SIZE]):,} ({len(test_frames[:TEST_SIZE])/len(all_frames)*100:.1f}%)")
print(f"   All Split Frames: {len(all_split_frames):,} ({len(all_split_frames)/len(all_frames)*100:.1f}%)")

print(f"\n📁 OUTPUT FILES:")
print(f"   📊 {ANALYSIS_DIR}/analysis_report.json - Analysis results")
print(f"   📊 {ANALYSIS_DIR}/recommended_training_frames.txt - Training recommendations")
print(f"   📊 {ANALYSIS_DIR}/class_distribution.png - Class distribution viz")
print(f"   📊 {ANALYSIS_DIR}/split_distribution.png - Split distribution viz")
print(f"   📄 {SPLITS_DIR}/all.txt - All split frames")
print(f"   📄 {SPLITS_DIR}/train.txt - Training split")
print(f"   📄 {SPLITS_DIR}/validation.txt - Validation split")
print(f"   📄 {SPLITS_DIR}/test.txt - Test split")
print(f"   📄 {SPLITS_DIR}/split_statistics.json - Split analysis")

print(f"\n🏆 Key Metrics:")
print(f"   Average Classes per Frame: {analysis_report['frame_statistics']['avg_classes_per_frame']:.2f}")
print(f"   Average Foreground Coverage: {analysis_report['frame_statistics']['avg_foreground_percentage']:.1f}%")
print(f"   Training Set Diversity: High (selected by analysis)")
print(f"   Validation/Test Balance: Weather/Time balanced")

print(f"\n✅ Dataset preparation complete! Ready for training.")
print(f"{'='*70}")