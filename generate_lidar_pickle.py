#!/usr/bin/env python3
"""
Generate LiDAR pickle files for CLFT format from ZOD dataset.

This script processes ZOD frames to create CLFT-compliant pickle files containing:
- 3D LiDAR points in camera coordinate systems
- Class/instance labels (currently zeros)
- Camera pixel coordinates for each point

The output format is used for training CLFT (Camera-LiDAR Fusion Transformer) models.

Usage:
    python generate_lidar_pickle.py
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import os

from zod import ZodFrames
from zod.constants import Camera, Lidar
from zod.data_classes.geometry import Pose
from zod.utils.geometry import transform_points, project_3d_to_2d_kannala, get_points_in_camera_fov


class LiDARDataGenerator:
    """
    Generates CLFT-format LiDAR pickle files from ZOD dataset frames.
    
    This class handles:
    - Loading ZOD dataset frames
    - Projecting 3D LiDAR points to camera coordinate systems
    - Filtering points by field of view and image bounds
    - Creating CLFT-compliant data structures
    - Saving pickle files with progress tracking
    """
    
    def __init__(self, zod_root, output_root, frames_list, verbose=False):
        """
        Initialize LiDAR data generator.
        
        Args:
            zod_root: Path to ZOD dataset root directory
            output_root: Path to output directory for CLFT data
            frames_list: Path to text file containing frame IDs to process
            verbose: Enable detailed progress logging
        """
        self.zod_root = Path(zod_root)
        self.output_root = Path(output_root)
        self.frames_list = Path(frames_list)
        self.verbose = verbose
        
        # Create output directory for LiDAR data
        self.lidar_dir = self.output_root / "lidar"
        self.lidar_dir.mkdir(parents=True, exist_ok=True)
        
        # Camera mapping for CLFT format (ZOD camera names to CLFT camera IDs)
        # Currently only Camera.FRONT is available in ZOD
        self.camera_mapping = {
            Camera.FRONT: 1,
        }
        
        print("Initializing ZOD dataset...")
        self.zod_frames = ZodFrames(dataset_root=self.zod_root, version="full")
        print("✓ ZOD dataset initialized")
        
        # Load list of frames to process
        self._load_frame_list()
    
    def project_lidar_to_camera(self, points_3d, calibration, camera_enum):
        """
        Project 3D LiDAR points to 2D camera pixel coordinates.
        
        Args:
            points_3d: (N, 3) array of 3D points in LiDAR coordinate system
            calibration: ZOD calibration object with camera and LiDAR parameters
            camera_enum: Camera identifier (e.g., Camera.FRONT)
            
        Returns:
            tuple: (points_2d, points_3d_valid) where points_2d are pixel coordinates
                   and points_3d_valid are the corresponding 3D points in camera frame.
                   Returns (None, None) if no valid points.
        """
        try:
            # Get coordinate transformations from calibration
            t_refframe_to_lidar = calibration.lidars[Lidar.VELODYNE].extrinsics
            t_refframe_to_camera = calibration.cameras[camera_enum].extrinsics
            
            # Calculate transformation from LiDAR coordinate system to camera coordinate system
            t_camera_from_refframe = t_refframe_to_camera.inverse
            t_lidar_to_camera = Pose(t_camera_from_refframe.transform @ t_refframe_to_lidar.transform)
            
            # Transform 3D points from LiDAR to camera coordinate system
            camera_data = transform_points(points_3d, t_lidar_to_camera.transform)
            
            # Filter points with positive depth (in front of camera)
            positive_depth = camera_data[:, 2] > 0
            camera_data = camera_data[positive_depth]
            
            if len(camera_data) == 0:
                return None, None
            
            # Filter points within camera field of view
            camera_calib = calibration.cameras[camera_enum]
            camera_data_fov, fov_mask = get_points_in_camera_fov(
                camera_calib.field_of_view, camera_data
            )
            
            if len(camera_data_fov) == 0:
                return None, None
            
            # Project 3D points to 2D pixel coordinates using Kannala-Brandt model
            xy_array = project_3d_to_2d_kannala(
                camera_data_fov,
                camera_calib.intrinsics[..., :3],  # Camera intrinsic parameters
                camera_calib.distortion,           # Lens distortion parameters
            )
            
            # Filter points within image bounds (with margin to avoid edge artifacts)
            image_width = int(camera_calib.image_dimensions[0])
            image_height = int(camera_calib.image_dimensions[1])
            margin = 10  # Pixel margin from image edges
            
            valid_mask = (
                (xy_array[:, 0] >= margin) & 
                (xy_array[:, 0] < image_width - margin) &
                (xy_array[:, 1] >= margin) & 
                (xy_array[:, 1] < image_height - margin)
            )
            
            # Return valid 2D pixel coordinates and corresponding 3D points
            valid_points_2d = xy_array[valid_mask]
            valid_points_3d = camera_data_fov[valid_mask]
            
            return valid_points_2d, valid_points_3d
            
        except Exception as e:
            if self.verbose:
                print(f"    Error in projection: {e}")
            return None, None
    
    def create_clft_pkl(self, frame_id):
        """
        Create CLFT-format pickle data for a single frame.
        
        Args:
            frame_id: Frame identifier string
            
        Returns:
            dict: CLFT data structure with 3d_points, class_instance, and camera_coordinates
                  Returns None if processing fails
        """
        try:
            zod_frame = self.zod_frames[frame_id]
            
            # Get aggregated LiDAR points from multiple sweeps around the keyframe timestamp
            image_timestamp = zod_frame.info.keyframe_time.timestamp()
            aggregated_lidar = zod_frame.get_aggregated_lidar(
                num_before=1,  # Include 1 sweep before keyframe
                num_after=1,   # Include 1 sweep after keyframe
                timestamp=image_timestamp
            )
            
            if aggregated_lidar is None:
                return None
            
            # Extract point cloud data (handle different possible data structures)
            if hasattr(aggregated_lidar, 'points'):
                points = aggregated_lidar.points
            elif hasattr(aggregated_lidar, 'data'):
                points = aggregated_lidar.data
            elif isinstance(aggregated_lidar, np.ndarray):
                points = aggregated_lidar
            else:
                return None
            
            # Validate point cloud format
            if not (isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] >= 3):
                return None
            
            # Extract XYZ coordinates (ignore intensity/reflectivity if present)
            points_3d = points[:, :3].astype(np.float32)
            
            calibration = zod_frame.calibration
            
            # Collect projections from all available cameras
            all_3d_points = []
            all_camera_coords = []
            all_class_instances = []
            
            for camera_enum, camera_id in self.camera_mapping.items():
                # Project LiDAR points to this camera's view
                points_2d, points_3d_valid = self.project_lidar_to_camera(
                    points_3d, calibration, camera_enum
                )
                
                if points_2d is not None and len(points_2d) > 0:
                    # Create camera coordinate array: [camera_id, x_pixel, y_pixel]
                    x_pixels = np.clip(np.round(points_2d[:, 0]), 0, 65535).astype(np.uint16)
                    y_pixels = np.clip(np.round(points_2d[:, 1]), 0, 65535).astype(np.uint16)
                    
                    camera_coords = np.column_stack([
                        np.full(len(points_2d), camera_id, dtype=np.uint16),  # Camera ID
                        x_pixels,  # X pixel coordinate
                        y_pixels   # Y pixel coordinate
                    ])
                    
                    # Create class_instance array (zeros for now - no semantic labels in ZOD)
                    class_instance = np.zeros((len(points_2d), 2), dtype=np.uint8)
                    
                    # Accumulate data from all cameras
                    all_3d_points.append(points_3d_valid.astype(np.float32))
                    all_camera_coords.append(camera_coords)
                    all_class_instances.append(class_instance)
            
            # Check if we have any valid projections
            if not all_3d_points:
                return None
            
            # Combine projections from all cameras into single arrays
            combined_3d_points = np.vstack(all_3d_points)
            combined_camera_coords = np.vstack(all_camera_coords)
            combined_class_instances = np.vstack(all_class_instances)
            
            # Create CLFT-compliant data structure
            clft_data = {
                '3d_points': combined_3d_points,              # (N, 3) float32 - XYZ in camera frame
                'class_instance': combined_class_instances,   # (N, 2) uint8 - class/instance IDs (zeros)
                'camera_coordinates': combined_camera_coords  # (N, 3) uint16 - [cam_id, x, y]
            }
            
            return clft_data
            
        except Exception as e:
            if self.verbose:
                print(f"  Error creating pkl for {frame_id}: {e}")
            return None
    
    def process_all_frames(self):
        """Process all frames in the frame list and generate pickle files."""
        print(f"\nProcessing {len(self.frame_ids):,} frames...")
        print(f"Output: {self.lidar_dir}")
        
        success_count = 0
        error_count = 0
        skip_count = 0
        
        # Process each frame with progress bar
        for frame_id in tqdm(self.frame_ids, desc="Generating LiDAR pkl files"):
            # Skip if output file already exists
            pkl_path = self.lidar_dir / f"frame_{frame_id}.pkl"
            
            if pkl_path.exists():
                skip_count += 1
                continue
            
            try:
                # Generate CLFT data for this frame
                clft_data = self.create_clft_pkl(frame_id)
                
                if clft_data is None:
                    error_count += 1
                    continue
                
                # Save data as pickle file
                with open(pkl_path, 'wb') as f:
                    pickle.dump(clft_data, f)
                
                success_count += 1
                
            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"\n  Error processing {frame_id}: {e}")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("LIDAR PKL GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully processed: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        print(f"\nOutput: {self.lidar_dir}")
        print("\nPKL format:")
        print("  - 3d_points: (N, 3) float32 - XYZ coordinates in camera frame")
        print("  - class_instance: (N, 2) uint8 - class/instance IDs (currently zeros)")
        print("  - camera_coordinates: (N, 3) uint16 - [camera_id, x_pixel, y_pixel]")


def main():
    """Main entry point for LiDAR pickle generation."""
    # Configuration - adjust paths as needed
    ZOD_ROOT = Path("/media/tom/ml/zod-data")      # ZOD dataset root directory
    OUTPUT_ROOT = Path("/media/tom/ml/zod_temp")   # Output directory for CLFT data
    FRAMES_LIST = Path("/media/tom/ml/projects/clft-zod/frames_to_process.txt")  # Frame list file
    
    print("="*60)
    print("LiDAR PKL Generator for CLFT")
    print("="*60)
    
    # Initialize generator with configuration
    generator = LiDARDataGenerator(
        zod_root=ZOD_ROOT,
        output_root=OUTPUT_ROOT,
        frames_list=FRAMES_LIST,
        verbose=False  # Set to True for detailed error messages
    )
    
    # Process all frames and generate pickle files
    generator.process_all_frames()


if __name__ == "__main__":
    main()
