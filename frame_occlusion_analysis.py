#!/usr/bin/env python3
"""Analyze frame occlusion levels and filter frames based on heavy occlusion threshold.

This script processes metadata JSON files from the CLFT dataset conversion and filters
out frames where heavy + very heavy occlusion exceeds a specified threshold (default 10%).
The remaining frame IDs are written to a text file for further processing.

Usage:
    python frame_occlusion_analysis.py \
        --metadata-dir /path/to/output_clft_full/metadata \
        --output-dir zod_dataset \
        --threshold 10.0 \
        --test-percentage 15.0 \
        --min-test-size 100 \
        --max-test-size 2000
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def analyze_frame_occlusion(metadata_file: Path, threshold: float) -> bool:
    """Analyze a single frame's occlusion levels.

    Args:
        metadata_file: Path to the metadata JSON file
        threshold: Maximum allowed percentage for heavy + very heavy occlusion

    Returns:
        True if frame passes the threshold (should be kept), False otherwise
    """
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Get occlusion percentages
        occlusion_percentages = metadata.get('occlusion_percentages', {})

        # Calculate heavy + very heavy occlusion percentage
        heavy_pct = occlusion_percentages.get('heavy', 0.0)
        veryheavy_pct = occlusion_percentages.get('veryheavy', 0.0)
        total_heavy_occlusion = heavy_pct + veryheavy_pct

        # Check if it exceeds threshold
        if total_heavy_occlusion > threshold:
            return False  # Filter out this frame

        return True  # Keep this frame

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logging.warning(f"Error processing {metadata_file}: {e}")
        return False  # Filter out on error


def get_frame_category(metadata_file: Path) -> str:
    """Get the category of a frame based on weather and time of day.

    Args:
        metadata_file: Path to the metadata JSON file

    Returns:
        Category string: 'day_clear', 'day_rain', 'night_clear', 'night_rain', or 'other'
    """
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        weather_condition = metadata.get('weather_condition', '').lower()
        is_daylight = metadata.get('is_daylight', False)

        # Categorize weather
        if weather_condition in ['clear', 'partly']:
            weather = 'clear'
        elif weather_condition in ['rain', 'snow']:
            weather = 'rain'
        else:
            weather = 'other'

        # Categorize time
        time = 'day' if is_daylight else 'night'

        if weather == 'other':
            return 'other'
        else:
            return f"{time}_{weather}"

    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return 'other'


def copy_training_data(output_dir: Path, frame_ids: List[str]) -> None:
    """Copy training data files for the specified frame IDs from output_clft_full to output_dir.

    Args:
        output_dir: Destination directory (zod_dataset)
        frame_ids: List of frame IDs to copy (e.g., ['000000', '000001', ...])
    """
    import shutil

    source_dir = Path('output_clft_full')
    folders_to_copy = ['annotation', 'annotation_rgb', 'camera', 'lidar', 'metadata', 'visualizations']

    # File extensions for each folder
    file_extensions = {
        'annotation': '.png',
        'annotation_rgb': '.png',
        'camera': '.png',
        'lidar': '.pkl',
        'metadata': '.json',
        'visualizations': '_sam_overlay.png'
    }

    total_files_copied = 0

    for folder in folders_to_copy:
        source_folder = source_dir / folder
        dest_folder = output_dir / folder

        if not source_folder.exists():
            logging.warning(f"Source folder {source_folder} does not exist, skipping")
            continue

        # Create destination folder
        dest_folder.mkdir(parents=True, exist_ok=True)

        folder_files_copied = 0

        for frame_id in frame_ids:
            # Construct source filename
            if folder == 'visualizations':
                source_filename = f"frame_{frame_id}_sam_overlay.png"
            else:
                source_filename = f"frame_{frame_id}{file_extensions[folder]}"

            source_file = source_folder / source_filename
            dest_file = dest_folder / source_filename

            # Copy file if it exists
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                folder_files_copied += 1
            else:
                logging.warning(f"Source file {source_file} does not exist")

        logging.info(f"Copied {folder_files_copied} files to {dest_folder}")
        total_files_copied += folder_files_copied

    logging.info(f"Total files copied: {total_files_copied}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--metadata-dir',
        type=Path,
        default=Path('output_clft_full/metadata'),
        help='Directory containing metadata JSON files (default: output_clft_full/metadata)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='zod_dataset',
        help='Output directory for results (default: zod_dataset)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Maximum allowed percentage for heavy + very heavy occlusion (default: 10.0)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--test-percentage',
        type=float,
        default=15.0,
        help='Percentage of frames from each weather/time category to use for testing (default: 15.0)'
    )
    parser.add_argument(
        '--min-test-size',
        type=int,
        default=100,
        help='Minimum test set size for any category (default: 100)'
    )
    parser.add_argument(
        '--max-test-size',
        type=int,
        default=2000,
        help='Maximum test set size for any category (default: 2000)'
    )
    parser.add_argument(
        '--copy-training-data',
        action='store_true',
        help='Copy training data files from output_clft_full to zod_dataset for frames in all.txt'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Validate input directory
    if not args.metadata_dir.exists():
        logging.error(f"Metadata directory does not exist: {args.metadata_dir}")
        return 1

    if not args.metadata_dir.is_dir():
        logging.error(f"Path is not a directory: {args.metadata_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all metadata JSON files
    metadata_files = list(args.metadata_dir.glob('*.json'))
    if not metadata_files:
        logging.error(f"No JSON files found in {args.metadata_dir}")
        return 1

    logging.info(f"Found {len(metadata_files)} metadata files to process")

    # Process each frame
    filtered_frames: List[str] = []
    frame_categories: Dict[str, List[str]] = defaultdict(list)
    total_frames = len(metadata_files)
    filtered_out = 0

    for metadata_file in metadata_files:
        # Extract frame ID from filename (remove 'frame_' prefix and '.json' suffix)
        frame_id = metadata_file.stem.replace('frame_', '')

        if analyze_frame_occlusion(metadata_file, args.threshold):
            filtered_frames.append(frame_id)
            # Get category for this frame
            category = get_frame_category(metadata_file)
            frame_categories[category].append(frame_id)
        else:
            filtered_out += 1

        if (len(filtered_frames) + filtered_out) % 100 == 0:
            logging.info(f"Processed {len(filtered_frames) + filtered_out}/{total_frames} frames")

    # Sort frame IDs for consistent output
    filtered_frames.sort()
    for category in frame_categories:
        frame_categories[category].sort()

    # Create all.txt with all camera frame paths
    all_file = args.output_dir / 'all.txt'
    with open(all_file, 'w', encoding='utf-8') as f:
        for frame_id in filtered_frames:
            camera_path = f"camera/frame_{frame_id}.png"
            f.write(f"{camera_path}\n")

    # Split frames by category to avoid overlap between train/early_stop and test sets
    train_frames = []
    early_stop_frames = []
    test_frames = {
        'test_day_fair.txt': [],
        'test_day_rain.txt': [],
        'test_night_fair.txt': [],
        'test_night_rain.txt': []
    }

    # Define target sizes for train and early_stop (total should be around 17k)
    target_train_size = 13000
    target_early_stop_size = 4000

    # Process each category and split into train/early_stop/test
    category_mapping = {
        'day_clear': 'test_day_fair.txt',
        'day_rain': 'test_day_rain.txt',
        'night_clear': 'test_night_fair.txt',
        'night_rain': 'test_night_rain.txt'
    }

    for category, frames in frame_categories.items():
        if category == 'other':
            continue

        category_size = len(frames)
        test_filename = category_mapping.get(category)

        if test_filename:
            # Calculate test set size as percentage of category, clamped between min and max
            test_size = int(category_size * args.test_percentage / 100.0)
            test_size = max(args.min_test_size, min(test_size, args.max_test_size))
            test_size = min(test_size, category_size)  # Don't exceed available frames

            # Split category: test frames come from the end, train/early_stop from the beginning
            test_frames[test_filename] = frames[-test_size:]  # Last N frames for test
            remaining_frames = frames[:-test_size]  # All but last N frames

            # Add to train and early_stop from remaining frames
            train_frames.extend(remaining_frames)
        else:
            # For any uncategorized frames, add to train
            train_frames.extend(frames)

    # Now split train_frames into train and early_stop
    # Shuffle for better distribution, but keep it deterministic with seed
    random.seed(42)  # For reproducible results
    random.shuffle(train_frames)

    # Take target sizes, but don't exceed available frames
    actual_train_size = min(target_train_size, len(train_frames))
    actual_early_stop_size = min(target_early_stop_size, len(train_frames) - actual_train_size)

    final_train_frames = train_frames[:actual_train_size]
    final_early_stop_frames = train_frames[actual_train_size:actual_train_size + actual_early_stop_size]

    # Sort for consistent output
    final_train_frames.sort()
    final_early_stop_frames.sort()

    # Create train.txt
    train_file = args.output_dir / 'train.txt'
    with open(train_file, 'w', encoding='utf-8') as f:
        for frame_id in final_train_frames:
            camera_path = f"camera/frame_{frame_id}.png"
            f.write(f"{camera_path}\n")

    # Create early_stop.txt
    early_stop_file = args.output_dir / 'early_stop.txt'
    with open(early_stop_file, 'w', encoding='utf-8') as f:
        for frame_id in final_early_stop_frames:
            camera_path = f"camera/frame_{frame_id}.png"
            f.write(f"{camera_path}\n")

    # Create test files
    test_sizes = {}
    for filename, frames in test_frames.items():
        test_sizes[filename] = len(frames)
        test_file = args.output_dir / filename
        with open(test_file, 'w', encoding='utf-8') as f:
            for frame_id in frames:
                camera_path = f"camera/frame_{frame_id}.png"
                f.write(f"{camera_path}\n")

    # Print summary to console
    kept_percentage = (len(filtered_frames) / total_frames) * 100 if total_frames > 0 else 0

    print("=== Frame Occlusion Analysis Results ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames kept: {len(filtered_frames)} ({kept_percentage:.1f}%)")
    print(f"Frames filtered out: {filtered_out}")
    print(f"Threshold used: {args.threshold}% heavy + very heavy occlusion")
    print(f"Test percentage: {args.test_percentage}% (min: {args.min_test_size}, max: {args.max_test_size})")
    print()
    print("Files created in zod_dataset/:")
    print(f"  all.txt: {len(filtered_frames)} frames")
    print(f"  train.txt: {len(final_train_frames)} frames")
    print(f"  early_stop.txt: {len(final_early_stop_frames)} frames")
    print(f"  test_day_fair.txt: {test_sizes['test_day_fair.txt']} frames")
    print(f"  test_day_rain.txt: {test_sizes['test_day_rain.txt']} frames")
    print(f"  test_night_fair.txt: {test_sizes['test_night_fair.txt']} frames")
    print(f"  test_night_rain.txt: {test_sizes['test_night_rain.txt']} frames")
    print()
    print("Weather/Time distribution:")
    for category, frames in frame_categories.items():
        if category != 'other':
            print(f"  {category}: {len(frames)} frames")

    logging.info(f"Successfully created dataset files in {args.output_dir}")

    # Copy training data files if requested
    if args.copy_training_data:
        logging.info("Copying training data files...")
        copy_training_data(args.output_dir, filtered_frames)

    return 0


if __name__ == '__main__':
    exit(main())
