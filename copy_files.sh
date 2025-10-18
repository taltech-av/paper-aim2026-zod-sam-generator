#!/bin/bash

# Script to copy annotation and lidar files listed in output_clft_v2/all.txt to /media/tom/ml/projects/clft-zod-training/zod_dataset

SOURCE_DIR="output_clft_v2"
DEST_DIR="/media/tom/ml/projects/clft-zod-training/zod_dataset"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Read the file line by line and copy each file
while IFS= read -r file_path; do
    # Skip empty lines
    if [[ -z "$file_path" ]]; then
        continue
    fi
    
    # Copy the camera file
    camera_path="$file_path"
    source_file="$SOURCE_DIR/$camera_path"
    dest_file="$DEST_DIR/$camera_path"
    mkdir -p "$(dirname "$dest_file")"
    if [[ -f "$source_file" ]]; then
        cp "$source_file" "$dest_file"
        echo "Copied: $camera_path"
    else
        echo "Warning: Source file not found: $source_file"
    fi
    
    # Copy the corresponding annotation file
    annotation_path="${file_path/camera/annotation}"
    source_file="$SOURCE_DIR/$annotation_path"
    dest_file="$DEST_DIR/$annotation_path"
    mkdir -p "$(dirname "$dest_file")"
    if [[ -f "$source_file" ]]; then
        cp "$source_file" "$dest_file"
        echo "Copied: $annotation_path"
    else
        echo "Warning: Source file not found: $source_file"
    fi
    
    # Copy the corresponding lidar file
    lidar_path="${file_path/camera/lidar}"
    lidar_path="${lidar_path%.png}.pkl"
    source_file="$SOURCE_DIR/$lidar_path"
    dest_file="$DEST_DIR/$lidar_path"
    mkdir -p "$(dirname "$dest_file")"
    if [[ -f "$source_file" ]]; then
        cp "$source_file" "$dest_file"
        echo "Copied: $lidar_path"
    else
        echo "Warning: Source file not found: $source_file"
    fi
    
done < "$SOURCE_DIR/splits/all.txt"

echo "Copy operation completed."