#!/usr/bin/env python3
"""
Analyze Waymo Dataset and Generate LaTeX Table

Analyzes the Waymo dataset splits (train, validation, test) by weather conditions
and generates a formatted LaTeX table with pixel counts and percentages.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse

# Paths
WAYMO_DATASET_ROOT = Path("/media/tom/ml/projects/clft-zod-training/waymo_dataset")
SPLITS_DIR = WAYMO_DATASET_ROOT / "splits_clft"

# Waymo class mapping
CLASS_NAMES = {
    0: "ignore",
    1: "vehicle",
    2: "pedestrian",
    3: "sign",
    4: "cyclist",
    5: "background"
}

def determine_condition_from_path(path):
    """Determine weather condition from file path"""
    # Path format: labeled/day/not_rain/camera/...
    # or: labeled/night/rain/camera/...
    
    parts = path.split('/')
    if len(parts) < 3:
        return 'unknown'
    
    time = parts[1]  # 'day' or 'night'
    weather_part = parts[2]  # 'not_rain' or 'rain'
    
    # Map to condition names
    if weather_part == 'not_rain':
        weather = 'fair'
    elif weather_part == 'rain':
        weather = 'rain'
    else:
        return 'unknown'
    
    if time not in ['day', 'night']:
        return 'unknown'
    
    return f"{time}_{weather}"

def load_frames_from_file(split_file):
    """Load frame paths and conditions from a specific split file"""
    frames_by_condition = defaultdict(list)
    
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                condition = determine_condition_from_path(line)
                frames_by_condition[condition].append(line)
    
    return frames_by_condition

def get_annotation_path(frame_path):
    """Convert camera path to annotation_gray path"""
    # Replace camera with annotation_gray
    anno_path = frame_path.replace('/camera/', '/annotation_gray/')
    return WAYMO_DATASET_ROOT / anno_path

def analyze_frames_pixel_distribution(frame_paths, split_name, condition):
    """Analyze pixel counts for a set of frames"""
    total_pixel_counts = defaultdict(int)
    class_representation = defaultdict(int)
    total_frames_analyzed = 0

    for frame_path in tqdm(frame_paths, desc=f"  {condition}", unit="frame", leave=False):
        # Frame path is already relative, just need to convert camera to annotation
        anno_path = frame_path.replace('/camera/', '/annotation/')
        annotation_path = WAYMO_DATASET_ROOT / anno_path
        
        if not annotation_path.exists():
            continue

        mask = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        unique_classes, counts = np.unique(mask, return_counts=True)
        
        # Update totals
        for class_id, count in zip(unique_classes, counts):
            total_pixel_counts[int(class_id)] += int(count)
            class_representation[int(class_id)] += 1

        total_frames_analyzed += 1

    return {
        'total_frames': total_frames_analyzed,
        'pixel_counts': dict(total_pixel_counts),
        'class_representation': dict(class_representation)
    }

def format_number(num):
    """Format large numbers in millions with one decimal"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def generate_latex_table(results, output_file='waymo_table.tex'):
    """Generate LaTeX table from analysis results"""
    
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Waymo Dataset Statistics by Split and Weather Condition}")
    lines.append(r"\begin{tabular}{@{}l l r r r r r @{}}")
    lines.append(r"\toprule")
    lines.append(r"Split & Weather & Samples & Background & Vehicle & Vuln. Users & Sign \\")
    lines.append(r" & Condition & & \multicolumn{4}{c}{\scriptsize (pixels / \% of total)} \\")
    lines.append(r"\midrule")
    
    # Order of splits and conditions
    split_order = ['train', 'validation', 'test']
    split_display = {'train': 'Train', 'validation': 'Validation', 'test': 'Test'}
    condition_order = ['day_fair', 'day_rain', 'night_fair', 'night_rain']
    condition_display = {
        'day_fair': 'Day Fair',
        'day_rain': 'Day Rain',
        'night_fair': 'Night Fair',
        'night_rain': 'Night Rain'
    }
    
    for split_idx, split_name in enumerate(split_order):
        if split_name not in results:
            continue
            
        split_data = results[split_name]
        
        # Count conditions present
        num_conditions = len([c for c in condition_order if c in split_data])
        
        # Add split rows
        for cond_idx, condition in enumerate(condition_order):
            if condition not in split_data:
                continue
                
            data = split_data[condition]
            total_pixels = sum(data['pixel_counts'].values())
            
            if total_pixels == 0:
                continue
            
            # Calculate values
            samples = data['total_frames']
            background = data['pixel_counts'].get(5, 0) + data['pixel_counts'].get(0, 0)  # Combine background + ignore
            vehicle = data['pixel_counts'].get(1, 0)
            vuln_users = data['pixel_counts'].get(2, 0) + data['pixel_counts'].get(4, 0)  # pedestrian + cyclist
            sign = data['pixel_counts'].get(3, 0)
            
            # Percentages
            bg_pct = (background / total_pixels * 100) if total_pixels > 0 else 0
            veh_pct = (vehicle / total_pixels * 100) if total_pixels > 0 else 0
            vuln_pct = (vuln_users / total_pixels * 100) if total_pixels > 0 else 0
            sign_pct = (sign / total_pixels * 100) if total_pixels > 0 else 0
            
            # Format row
            if cond_idx == 0:
                row = f"\\multirow{{{num_conditions + 1}}}{{*}}{{{split_display[split_name]}}}"
            else:
                row = ""
            
            row += f" & {condition_display[condition]}"
            row += f" & {samples:,}"
            row += f" & {format_number(background)} / {bg_pct:.1f}\\%"
            row += f" & {format_number(vehicle)} / {veh_pct:.1f}\\%"
            row += f" & {format_number(vuln_users)} / {vuln_pct:.1f}\\%"
            row += f" & {format_number(sign)} / {sign_pct:.1f}\\%"
            row += " \\\\"
            
            lines.append(row)
        
        # Add total row for this split
        total_samples = sum(d['total_frames'] for d in split_data.values())
        total_pixels_all = sum(sum(d['pixel_counts'].values()) for d in split_data.values())
        
        total_bg = sum(d['pixel_counts'].get(5, 0) + d['pixel_counts'].get(0, 0) for d in split_data.values())
        total_veh = sum(d['pixel_counts'].get(1, 0) for d in split_data.values())
        total_vuln = sum(d['pixel_counts'].get(2, 0) + d['pixel_counts'].get(4, 0) for d in split_data.values())
        total_sign = sum(d['pixel_counts'].get(3, 0) for d in split_data.values())
        
        total_row = " & \\textbf{Total}"
        total_row += f" & \\textbf{{{total_samples:,}}}"
        total_row += f" & \\textbf{{{format_number(total_bg)} / {total_bg/total_pixels_all*100:.1f}\\%}}"
        total_row += f" & \\textbf{{{format_number(total_veh)} / {total_veh/total_pixels_all*100:.1f}\\%}}"
        total_row += f" & \\textbf{{{format_number(total_vuln)} / {total_vuln/total_pixels_all*100:.1f}\\%}}"
        total_row += f" & \\textbf{{{format_number(total_sign)} / {total_sign/total_pixels_all*100:.1f}\\%}}"
        total_row += " \\\\"
        
        lines.append(total_row)
        
        # Add midrule between splits
        if split_idx < len(split_order) - 1:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:waymo_stats}")
    lines.append(r"\end{table*}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nLaTeX table saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Waymo dataset and generate LaTeX table')
    parser.add_argument('--output', type=str, default='waymo_table.tex',
                       help='Output LaTeX file')
    args = parser.parse_args()
    
    # Define split files
    # Train and validation are single files with mixed conditions
    # Test is split into separate files per condition
    splits = {
        'train': SPLITS_DIR / 'train_all.txt',
        'validation': SPLITS_DIR / 'early_stop_valid.txt',
        'test': {
            'day_fair': SPLITS_DIR / 'test_day_fair.txt',
            'day_rain': SPLITS_DIR / 'test_day_rain.txt',
            'night_fair': SPLITS_DIR / 'test_night_fair.txt',
            'night_rain': SPLITS_DIR / 'test_night_rain.txt',
        }
    }
    
    results = {}
    
    # Process train and validation (single files with mixed conditions)
    for split_name in ['train', 'validation']:
        split_file = splits[split_name]
        if not split_file.exists():
            continue
            
        print(f"\nAnalyzing {split_name} split...")
        frames_by_condition = load_frames_from_file(split_file)
        
        results[split_name] = {}
        
        for condition in ['day_fair', 'day_rain', 'night_fair', 'night_rain']:
            if condition not in frames_by_condition:
                continue
            
            frame_paths = frames_by_condition[condition]
            if not frame_paths:
                continue
            
            analysis = analyze_frames_pixel_distribution(frame_paths, split_name, condition)
            results[split_name][condition] = analysis
            
            print(f"  {condition}: {analysis['total_frames']} frames")
    
    # Process test split (separate files per condition)
    print(f"\nAnalyzing test split...")
    results['test'] = {}
    
    for condition in ['day_fair', 'day_rain', 'night_fair', 'night_rain']:
        split_file = splits['test'][condition]
        
        if not split_file.exists():
            continue
        
        # Load frame paths from file
        with open(split_file, 'r') as f:
            frame_paths = [line.strip() for line in f if line.strip()]
        
        if not frame_paths:
            continue
        
        # Analyze this condition
        analysis = analyze_frames_pixel_distribution(frame_paths, 'test', condition)
        results['test'][condition] = analysis
        
        print(f"  {condition}: {analysis['total_frames']} frames")
    
    # Generate LaTeX table
    generate_latex_table(results, args.output)
    
    print(f"\n✓ Analysis complete! LaTeX table saved to {args.output}")

if __name__ == "__main__":
    main()
