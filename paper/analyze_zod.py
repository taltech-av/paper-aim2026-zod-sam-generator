#!/usr/bin/env python3
"""
Analyze ZOD dataset and generate LaTeX table with statistics
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
DATASET_DIR = Path("/media/tom/ml/projects/clft-zod-training/zod_dataset")
ZOD_METADATA_ROOT = Path("/media/tom/ml/zod-data/single_frames")

# Class mapping (ZOD uses different class IDs than Waymo)
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
        metadata_path = ZOD_METADATA_ROOT / frame_id / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                zod_metadata = json.load(f)

            # Extract weather
            scraped_weather = str(zod_metadata.get("scraped_weather", "")).lower()
            if "snow" in scraped_weather:
                weather = "snow"
            elif any(keyword in scraped_weather for keyword in ["rain", "sleet", "hail", "storm", "drizzle"]):
                weather = "rain"
            else:
                weather = "fair"

            # Extract time of day
            time_of_day = str(zod_metadata.get("time_of_day", "day")).lower()
            timeofday = "night" if "night" in time_of_day else "day"

            if weather == "snow":
                condition = "snow"
            else:
                condition = f"{timeofday}_{weather}"
            return condition
    except Exception as e:
        print(f"Warning: Could not read metadata for {frame_id}: {e}")
    return None

def load_frames_from_file(split_file):
    """Load frame paths from split file and group by weather condition"""
    frames_by_condition = defaultdict(list)
    
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract frame ID from path like "camera/frame_099985.png"
            frame_id = line.split('/')[-1].replace('frame_', '').replace('.png', '')
            
            # Get condition from metadata
            condition = get_weather_from_metadata(frame_id)
            if condition:
                frames_by_condition[condition].append(line)
    
    return frames_by_condition

def analyze_frames_pixel_distribution(frame_paths, split_name, condition):
    """Analyze pixel distribution across all frames for a given condition"""
    
    pixel_counts = defaultdict(int)
    total_frames = len(frame_paths)
    
    print(f"    Analyzing {len(frame_paths)} frames for {condition}...")
    
    for frame_path in tqdm(frame_paths, desc=f"  {split_name}/{condition}", leave=False):
        # Convert camera path to annotation path
        # camera/frame_099985.png -> annotation_fusion/frame_099985.png
        annotation_path = DATASET_DIR / frame_path.replace('camera/', 'annotation_fusion/')
        
        if not annotation_path.exists():
            print(f"    Warning: Annotation not found: {annotation_path}")
            continue
        
        # Load annotation mask
        mask = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"    Warning: Could not load: {annotation_path}")
            continue
        
        # Count pixels for each class
        unique, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique, counts):
            pixel_counts[int(class_id)] += int(count)
    
    return {
        'total_frames': total_frames,
        'pixel_counts': dict(pixel_counts)
    }

def format_number(num):
    """Format large numbers with K/M suffix"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def generate_latex_table(results, output_file):
    """Generate LaTeX table from analysis results"""
    
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{ZOD Dataset Statistics by Split and Weather Condition}")
    lines.append(r"\begin{tabular}{@{}l l r r r r r @{}}")
    lines.append(r"\toprule")
    lines.append(r"Split & Weather & Samples & Background & Vehicle & Vuln. Users & Sign \\")
    lines.append(r" & Condition & & \multicolumn{4}{c}{\scriptsize (pixels / \% of total)} \\")
    lines.append(r"\midrule")
    
    # Order of splits and conditions
    split_order = ['train', 'validation', 'test']
    split_display = {'train': 'Train', 'validation': 'Validation', 'test': 'Test'}
    condition_order = ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']
    condition_display = {
        'day_fair': 'Day Fair',
        'day_rain': 'Day Rain',
        'night_fair': 'Night Fair',
        'night_rain': 'Night Rain',
        'snow': 'Snow'
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
            
            # Calculate values (note: ZOD class mapping)
            samples = data['total_frames']
            background = data['pixel_counts'].get(0, 0) + data['pixel_counts'].get(1, 0)  # Combine background + ignore
            vehicle = data['pixel_counts'].get(2, 0)
            vuln_users = data['pixel_counts'].get(5, 0) + data['pixel_counts'].get(4, 0)  # pedestrian + cyclist
            sign = data['pixel_counts'].get(3, 0)
            
            # Percentages
            bg_pct = (background / total_pixels * 100) if total_pixels > 0 else 0
            vh_pct = (vehicle / total_pixels * 100) if total_pixels > 0 else 0
            vu_pct = (vuln_users / total_pixels * 100) if total_pixels > 0 else 0
            sg_pct = (sign / total_pixels * 100) if total_pixels > 0 else 0
            
            # First row of this split gets the split name
            if cond_idx == 0:
                split_label = f"\\multirow{{{num_conditions+1}}}{{*}}{{{split_display[split_name]}}}"
            else:
                split_label = ""
            
            # Format row
            lines.append(
                f"{split_label} & {condition_display[condition]} & "
                f"{samples:,} & "
                f"{format_number(background)} / {bg_pct:.1f}\\% & "
                f"{format_number(vehicle)} / {vh_pct:.1f}\\% & "
                f"{format_number(vuln_users)} / {vu_pct:.1f}\\% & "
                f"{format_number(sign)} / {sg_pct:.1f}\\% \\\\"
            )
        
        # Add total row for this split
        total_samples = sum(data['total_frames'] for data in split_data.values())
        total_bg = sum(data['pixel_counts'].get(0, 0) + data['pixel_counts'].get(1, 0) for data in split_data.values())
        total_vh = sum(data['pixel_counts'].get(2, 0) for data in split_data.values())
        total_vu = sum(data['pixel_counts'].get(5, 0) + data['pixel_counts'].get(4, 0) for data in split_data.values())
        total_sg = sum(data['pixel_counts'].get(3, 0) for data in split_data.values())
        total_all = total_bg + total_vh + total_vu + total_sg
        
        lines.append(
            f" & \\textbf{{Total}} & "
            f"\\textbf{{{total_samples:,}}} & "
            f"\\textbf{{{format_number(total_bg)} / {(total_bg/total_all*100):.1f}\\%}} & "
            f"\\textbf{{{format_number(total_vh)} / {(total_vh/total_all*100):.1f}\\%}} & "
            f"\\textbf{{{format_number(total_vu)} / {(total_vu/total_all*100):.1f}\\%}} & "
            f"\\textbf{{{format_number(total_sg)} / {(total_sg/total_all*100):.1f}\\%}} \\\\"
        )
        
        # Add midrule between splits
        if split_idx < len(split_order) - 1:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:zod_stats}")
    lines.append(r"\end{table*}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nLaTeX table saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ZOD dataset and generate LaTeX table')
    parser.add_argument('--output', type=str, default='zod_table.tex',
                       help='Output LaTeX file')
    args = parser.parse_args()
    
    # Define split files (all are single files with mixed conditions)
    splits = {
        'train': DATASET_DIR / 'train.txt',
        'validation': DATASET_DIR / 'validation.txt',
        'test': {
            'day_fair': DATASET_DIR / 'test_day_fair.txt',
            'day_rain': DATASET_DIR / 'test_day_rain.txt',
            'night_fair': DATASET_DIR / 'test_night_fair.txt',
            'night_rain': DATASET_DIR / 'test_night_rain.txt',
            'snow': DATASET_DIR / 'test_snow.txt',
        }
    }
    
    results = {}
    
    # Process train and validation (single files with mixed conditions)
    for split_name in ['train', 'validation']:
        split_file = splits[split_name]
        if not split_file.exists():
            print(f"Warning: Split file not found: {split_file}")
            continue
            
        print(f"\nAnalyzing {split_name} split...")
        frames_by_condition = load_frames_from_file(split_file)
        
        results[split_name] = {}
        
        for condition in ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']:
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
    
    for condition in ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']:
        split_file = splits['test'][condition]
        
        if not split_file.exists():
            print(f"  Warning: Test file not found: {split_file}")
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
