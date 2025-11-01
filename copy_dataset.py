#!/usr/bin/env python3
"""
Copy Dataset Files for Training

Copies files listed in all.txt from multiple directories to create a complete
training dataset in /media/tom/ml/zod_dataset.

Copies from:
- annotation_camera_only/ -> annotation_camera_only/
- annotation_lidar_only/ -> annotation_lidar_only/
- annotation_fusion/ -> annotation_fusion/
- camera/ -> camera/
- lidar_png/ -> lidar_png/
- splits/ -> splits/

Usage: python copy_dataset.py
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
SOURCE_BASE = Path("/media/tom/ml/zod_temp")
SPLIT_FILE = SOURCE_BASE / "splits" / "all.txt"
DEST_BASE = Path("/media/tom/ml/zod_dataset")

# Source directories
SOURCE_DIRS = {
    "annotation_camera_only": SOURCE_BASE / "annotation_camera_only",
    "annotation_lidar_only": SOURCE_BASE / "annotation_lidar_only",
    "annotation_fusion": SOURCE_BASE / "annotation_fusion",
    "camera": SOURCE_BASE / "camera",
    "lidar_png": SOURCE_BASE / "lidar_png",
    "splits": SOURCE_BASE / "splits"
}

def extract_frame_id(path_str):
    """Extract frame ID from path like 'camera/frame_000000.png'"""
    # Split by '/' and take the last part
    filename = path_str.split('/')[-1]
    # Remove 'frame_' prefix and '.png' suffix
    frame_id = filename.replace('frame_', '').replace('.png', '')
    return frame_id

def copy_splits_directory(dry_run=False):
    """Copy the entire splits directory"""
    src_splits = SOURCE_BASE / "splits"
    dest_splits = DEST_BASE

    if not src_splits.exists():
        print(f"⚠️  Warning: Splits directory not found: {src_splits}")
        return 0

    if not dry_run:
        # Copy entire directory
        if dest_splits.exists():
            shutil.rmtree(dest_splits)
        shutil.copytree(src_splits, dest_splits)

    # Count files in splits directory
    split_files = list(src_splits.glob("*"))
    return len(split_files)

def copy_frame_files(frame_id, dry_run=False):
    """Copy all files for a given frame ID"""
    files_copied = 0

    for modality, source_dir in SOURCE_DIRS.items():
        if modality == "splits":
            # Skip splits directory - we'll copy it entirely later
            continue

        # Source file
        src_file = source_dir / f"frame_{frame_id}.png"

        # Destination file
        dest_dir = DEST_BASE / modality
        dest_file = dest_dir / f"frame_{frame_id}.png"

        if src_file.exists():
            if not dry_run:
                # Create destination directory if it doesn't exist
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Copy file
                shutil.copy2(src_file, dest_file)
            files_copied += 1
        else:
            print(f"⚠️  Warning: {src_file} not found")

    return files_copied

def main():
    print("📋 Dataset Copy Script")
    print("=" * 50)
    print(f"📁 Source: {SOURCE_BASE}")
    print(f"📄 Split file: {SPLIT_FILE}")
    print(f"📁 Destination: {DEST_BASE}")

    # Check if split file exists
    if not SPLIT_FILE.exists():
        print(f"❌ Split file not found: {SPLIT_FILE}")
        return

    # Read all frame paths
    print(f"\n📖 Reading {SPLIT_FILE}...")
    with open(SPLIT_FILE, 'r') as f:
        frame_paths = [line.strip() for line in f if line.strip()]

    print(f"📊 Found {len(frame_paths):,} frame entries")

    # Extract unique frame IDs
    frame_ids = set()
    for path in frame_paths:
        frame_id = extract_frame_id(path)
        frame_ids.add(frame_id)

    print(f"🎯 Unique frame IDs: {len(frame_ids):,}")

    # Create destination directories
    print("\n📁 Creating destination directories...")
    for modality in SOURCE_DIRS.keys():
        dest_dir = DEST_BASE / modality
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dest_dir}")

    # Count total files to copy (excluding splits directory from per-frame count)
    total_png_files_expected = len(frame_ids) * (len(SOURCE_DIRS) - 1)  # -1 for splits
    print(f"\n📊 Total PNG files to copy: {total_png_files_expected:,}")

    # Copy PNG files
    print("\n🚀 Copying PNG files...")
    copied_files = 0
    missing_files = 0

    for frame_id in tqdm(sorted(frame_ids), desc="Copying frames"):
        files_for_frame = copy_frame_files(frame_id)
        copied_files += files_for_frame

        if files_for_frame < (len(SOURCE_DIRS) - 1):  # -1 for splits
            missing_files += ((len(SOURCE_DIRS) - 1) - files_for_frame)

    # Copy splits directory
    print("\n📋 Copying splits directory...")
    splits_files_copied = copy_splits_directory()
    print(f"📊 Split files copied: {splits_files_copied}")

    total_files_copied = copied_files + splits_files_copied

    # Summary
    print("\n✅ Copy complete!")
    print(f"📊 PNG files copied: {copied_files:,}")
    print(f"📊 Split files copied: {splits_files_copied:,}")
    print(f"📊 Files missing: {missing_files:,}")
    print(f"📊 PNG files expected: {total_png_files_expected:,}")

    if missing_files > 0:
        print(f"⚠️  {missing_files} PNG files were missing from source directories")

    # Verify destination
    print("\n🔍 Verifying destination...")
    dest_files = 0
    for modality in SOURCE_DIRS.keys():
        dest_dir = DEST_BASE / modality
        if dest_dir.exists():
            if modality == "splits":
                # Count all files in splits directory
                files_in_dir = list(dest_dir.glob("*"))
            else:
                # Count PNG files for other directories
                files_in_dir = list(dest_dir.glob("*.png"))
            dest_files += len(files_in_dir)
            print(f"  {modality}: {len(files_in_dir):,} files")

    print(f"\n📊 Total files in destination: {dest_files:,}")

    if dest_files == total_files_copied:
        print("✅ All files copied successfully!")
    else:
        print(f"⚠️  Mismatch: {dest_files} files found vs {total_files_copied} expected")

    print(f"\n📁 Dataset ready at: {DEST_BASE}")
    print(f"📊 Total files copied: {total_files_copied:,}")
    print("Ready for training! 🚀")

if __name__ == "__main__":
    main()