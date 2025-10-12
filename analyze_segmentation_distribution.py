#!/usr/bin/env python3
"""
Analyze object class distributions in completed segmentation masks
"""

import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import json

OUTPUT_DIR = Path("output_clft_v2")
ANNOTATION_DIR = OUTPUT_DIR / "annotation"

# Class mapping
CLASS_NAMES = {
    0: "Background",
    1: "Lane",
    2: "Vehicle", 
    3: "Sign",
    4: "Cyclist",
    5: "Pedestrian"
}

print(f"🔍 Analyzing segmentations in: {ANNOTATION_DIR}")

if not ANNOTATION_DIR.exists():
    print(f"❌ Directory not found: {ANNOTATION_DIR}")
    exit(1)

# Find all annotation files
annotation_files = list(ANNOTATION_DIR.glob("frame_*.png"))
print(f"📁 Found {len(annotation_files)} segmentation masks")

if len(annotation_files) == 0:
    print("❌ No segmentation files found!")
    exit(1)

# Stats collection
class_pixel_counts = defaultdict(int)
class_frame_counts = defaultdict(int)
frames_analyzed = 0

print(f"\n📊 Analyzing {len(annotation_files)} frames...")

for idx, anno_file in enumerate(annotation_files):
    # Load annotation mask
    mask = cv2.imread(str(anno_file), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        continue
    
    frames_analyzed += 1
    
    # Count pixels per class
    unique_classes, counts = np.unique(mask, return_counts=True)
    
    for class_id, count in zip(unique_classes, counts):
        class_pixel_counts[int(class_id)] += int(count)
        if class_id > 0:  # Don't count background for frame presence
            class_frame_counts[int(class_id)] += 1
    
    # Progress
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(annotation_files)} frames...")

print(f"\n✅ Analyzed {frames_analyzed} frames")

# Calculate statistics
total_pixels = sum(class_pixel_counts.values())
total_foreground = sum(v for k, v in class_pixel_counts.items() if k > 0)

print(f"\n" + "="*60)
print(f"📊 SEGMENTATION STATISTICS")
print(f"="*60)

print(f"\n📏 Pixel Distribution:")
print(f"{'Class':<15} {'Pixels':>15} {'% of Total':>12} {'% of FG':>12} {'Frames':>10}")
print("-" * 70)

for class_id in sorted(class_pixel_counts.keys()):
    pixels = class_pixel_counts[class_id]
    pct_total = (pixels / total_pixels * 100) if total_pixels > 0 else 0
    pct_fg = (pixels / total_foreground * 100) if total_foreground > 0 and class_id > 0 else 0
    frames = class_frame_counts[class_id]
    
    class_name = CLASS_NAMES.get(class_id, f"Unknown-{class_id}")
    
    if class_id == 0:
        print(f"{class_name:<15} {pixels:>15,} {pct_total:>11.2f}% {'-':>12} {'-':>10}")
    else:
        print(f"{class_name:<15} {pixels:>15,} {pct_total:>11.2f}% {pct_fg:>11.2f}% {frames:>10,}")

print("-" * 70)
print(f"{'TOTAL':<15} {total_pixels:>15,} {100.0:>11.2f}%")
print(f"{'Foreground':<15} {total_foreground:>15,} {total_foreground/total_pixels*100:>11.2f}%")

# Object presence statistics
print(f"\n📈 Object Presence in Frames:")
print(f"{'Class':<15} {'Frames':>10} {'% of Frames':>12} {'Avg per Frame':>15}")
print("-" * 60)

for class_id in sorted([k for k in class_pixel_counts.keys() if k > 0]):
    frames = class_frame_counts[class_id]
    pct_frames = (frames / frames_analyzed * 100) if frames_analyzed > 0 else 0
    avg_pixels = class_pixel_counts[class_id] / frames if frames > 0 else 0
    
    class_name = CLASS_NAMES.get(class_id, f"Unknown-{class_id}")
    print(f"{class_name:<15} {frames:>10,} {pct_frames:>11.2f}% {avg_pixels:>14,.0f} px")

# Save results
results = {
    "frames_analyzed": frames_analyzed,
    "total_pixels": int(total_pixels),
    "total_foreground": int(total_foreground),
    "class_pixel_counts": {CLASS_NAMES.get(k, str(k)): int(v) for k, v in class_pixel_counts.items()},
    "class_frame_counts": {CLASS_NAMES.get(k, str(k)): int(v) for k, v in class_frame_counts.items()},
    "class_percentages": {
        CLASS_NAMES.get(k, str(k)): float(v / total_foreground * 100) if total_foreground > 0 and k > 0 else 0
        for k, v in class_pixel_counts.items() if k > 0
    }
}

output_json = OUTPUT_DIR / "segmentation_statistics.json"
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_json}")

# Summary insights
print(f"\n💡 Key Insights:")

# Find dominant class
fg_classes = {k: v for k, v in class_pixel_counts.items() if k > 0}
if fg_classes:
    dominant_class = max(fg_classes, key=fg_classes.get)
    dominant_name = CLASS_NAMES.get(dominant_class, str(dominant_class))
    dominant_pct = fg_classes[dominant_class] / total_foreground * 100
    print(f"  • Most common: {dominant_name} ({dominant_pct:.1f}% of foreground)")

# Check coverage
if 2 in class_frame_counts:  # Vehicles
    vehicle_coverage = class_frame_counts[2] / frames_analyzed * 100
    print(f"  • Vehicles appear in {vehicle_coverage:.1f}% of frames")

if 5 in class_frame_counts:  # Pedestrians
    ped_coverage = class_frame_counts[5] / frames_analyzed * 100
    print(f"  • Pedestrians appear in {ped_coverage:.1f}% of frames")

# Check balance
fg_pcts = [v / total_foreground * 100 for k, v in class_pixel_counts.items() if k > 0]
if len(fg_pcts) > 1:
    max_pct = max(fg_pcts)
    min_pct = min(fg_pcts)
    ratio = max_pct / min_pct if min_pct > 0 else float('inf')
    print(f"  • Class imbalance ratio: {ratio:.1f}:1 (max/min)")

print()
