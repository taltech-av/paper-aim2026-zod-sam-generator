#!/usr/bin/env python3
"""
Calculate LiDAR statistics (mean and std) from ZOD dataset projections.

This script processes all available frames in the ZOD dataset, extracts
LiDAR point cloud data, and computes global statistics for z-coordinates
(lidar_mean_zod and lidar_std_zod).
"""

import numpy as np
from zod import ZodFrames
from typing import List, Tuple
import os

def initialize_zod_frames(dataset_root: str = "./data", version: str = "mini") -> ZodFrames:
    """
    Initialize ZodFrames object with specified dataset root and version.

    Args:
        dataset_root: Path to the ZOD dataset root directory
        version: Dataset version to use ("mini" or "full")

    Returns:
        Initialized ZodFrames object
    """
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)
    print(f"Initialized ZOD dataset from: {dataset_root}")
    print(f"Version: {version}")
    print(f"Total frames available: {len(zod_frames)}")
    return zod_frames

def get_all_frame_ids(zod_frames: ZodFrames) -> List[str]:
    """
    Get all available frame IDs from training and validation splits.

    Args:
        zod_frames: ZodFrames object

    Returns:
        List of all frame IDs (train + validation)
    """
    train_frame_ids = list(zod_frames.get_split("train"))
    val_frame_ids = list(zod_frames.get_split("val"))
    all_frame_ids = train_frame_ids + val_frame_ids

    print(f"Found {len(all_frame_ids)} total frames ({len(train_frame_ids)} train + {len(val_frame_ids)} val)")
    return all_frame_ids

def calculate_lidar_statistics(zod_frames: ZodFrames, frame_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and standard deviation of LiDAR coordinates (x, y, z) across all frames.

    Args:
        zod_frames: ZodFrames object
        frame_ids: List of frame IDs to process

    Returns:
        Tuple of (mean_xyz, std_xyz) as numpy arrays with shape (3,)
    """
    all_points = []

    print(f"Processing {len(frame_ids)} frames for LiDAR statistics...")

    for i, frame_id in enumerate(frame_ids):
        try:
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(frame_ids)}: {frame_id}")

            # Get the frame object
            zod_frame = zod_frames[frame_id]

            # Get image timestamp for LiDAR aggregation
            image_timestamp = zod_frame.info.keyframe_time.timestamp()

            # Get aggregated LiDAR point cloud
            aggregated_lidar = zod_frame.get_aggregated_lidar(
                num_before=1,
                num_after=1,
                timestamp=image_timestamp
            )

            # Extract coordinates from LiDAR points
            if hasattr(aggregated_lidar, 'points'):
                # If it's a point cloud object with .points attribute
                points = aggregated_lidar.points[:, :3]  # [x, y, z]
            elif isinstance(aggregated_lidar, np.ndarray):
                # If it's a numpy array
                if aggregated_lidar.shape[1] >= 3:
                    points = aggregated_lidar[:, :3]  # [x, y, z]
                else:
                    print(f"  Warning: Unexpected LiDAR array shape {aggregated_lidar.shape} for frame {frame_id}")
                    continue
            else:
                print(f"  Warning: Unexpected LiDAR data type for frame {frame_id}")
                continue

            # Filter out invalid points (NaN or infinite values)
            valid_mask = np.isfinite(points).all(axis=1)
            valid_points = points[valid_mask]

            if len(valid_points) > 0:
                all_points.append(valid_points)

        except Exception as e:
            print(f"  Error processing frame {frame_id}: {e}")
            continue

    if not all_points:
        raise ValueError("No valid LiDAR points found across all frames")

    # Concatenate all valid points
    all_points_array = np.concatenate(all_points, axis=0)

    # Calculate per-dimension statistics
    lidar_mean_zod = np.mean(all_points_array, axis=0)  # [mean_x, mean_y, mean_z]
    lidar_std_zod = np.std(all_points_array, axis=0)    # [std_x, std_y, std_z]

    print("\nLiDAR Statistics Summary:")
    print(f"  Total points processed: {len(all_points_array):,}")
    print(f"  lidar_mean_zod: [{lidar_mean_zod[0]:.6f}, {lidar_mean_zod[1]:.6f}, {lidar_mean_zod[2]:.6f}]")
    print(f"  lidar_std_zod: [{lidar_std_zod[0]:.6f}, {lidar_std_zod[1]:.6f}, {lidar_std_zod[2]:.6f}]")
    print(f"  X-range: [{np.min(all_points_array[:, 0]):.3f}, {np.max(all_points_array[:, 0]):.3f}]")
    print(f"  Y-range: [{np.min(all_points_array[:, 1]):.3f}, {np.max(all_points_array[:, 1]):.3f}]")
    print(f"  Z-range: [{np.min(all_points_array[:, 2]):.3f}, {np.max(all_points_array[:, 2]):.3f}]")

    return lidar_mean_zod, lidar_std_zod

def main():
    """
    Main function to calculate LiDAR statistics from ZOD dataset.
    """
    # Configuration
    dataset_root = "./data"
    version = "mini"

    print("=== LiDAR Statistics Calculator ===")

    # Initialize ZOD dataset
    zod_frames = initialize_zod_frames(dataset_root, version)

    # Get all frame IDs
    all_frame_ids = get_all_frame_ids(zod_frames)

    # Calculate LiDAR statistics
    try:
        lidar_mean_zod, lidar_std_zod = calculate_lidar_statistics(zod_frames, all_frame_ids)

        print("\n=== RESULTS ===")
        print(f"\"lidar_mean_zod\": [{lidar_mean_zod[0]:.8f}, {lidar_mean_zod[1]:.8f}, {lidar_mean_zod[2]:.8f}],")
        print(f"\"lidar_std_zod\": [{lidar_std_zod[0]:.8f}, {lidar_std_zod[1]:.8f}, {lidar_std_zod[2]:.8f}]")

        # Save results to file
        output_file = "/workspace/lidar_statistics.txt"
        with open(output_file, 'w') as f:
            f.write(f"\"lidar_mean_zod\": [{lidar_mean_zod[0]:.8f}, {lidar_mean_zod[1]:.8f}, {lidar_mean_zod[2]:.8f}],\n")
            f.write(f"\"lidar_std_zod\": [{lidar_std_zod[0]:.8f}, {lidar_std_zod[1]:.8f}, {lidar_std_zod[2]:.8f}]\n")
            f.write(f"Dataset: {dataset_root}\n")
            f.write(f"Version: {version}\n")
            f.write(f"Frames processed: {len(all_frame_ids)}\n")

        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return

    print("=== Calculation Complete ===")

if __name__ == "__main__":
    main()
