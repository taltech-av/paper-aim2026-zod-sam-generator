#!/usr/bin/env python3
"""
Generate LiDAR pkl files for CLFT format.
Reads frames from frames_to_process.txt and creates pkl files with 3D points, 
class_instance, and camera_coordinates.

Usage:
    python generate_lidar_data.py
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
    def __init__(self, zod_root, output_root, frames_list, verbose=False):
        """Initialize LiDAR data generator
        
        Args:
            zod_root: Path to ZOD dataset root
            output_root: Path to output directory (output_clft_v2)
            frames_list: Path to frames_to_process.txt
            verbose: Print detailed progress
        """
        self.zod_root = Path(zod_root)
        self.output_root = Path(output_root)
        self.frames_list = Path(frames_list)
        self.verbose = verbose
        
        # Output directory
        self.lidar_dir = self.output_root / "lidar"
        self.lidar_dir.mkdir(parents=True, exist_ok=True)
        
        # Camera mapping for CLFT format (camera_name -> camera_id)
        # ZOD only provides Camera.FRONT
        self.camera_mapping = {
            Camera.FRONT: 1,
        }
        
        print("Initializing ZOD dataset...")
        self.zod_frames = ZodFrames(dataset_root=self.zod_root, version="full")
        print(f"✓ ZOD initialized")
        
        # Load frames to process
        try:
            with open(self.frames_list) as f:
                self.frame_ids = [line.strip() for line in f if line.strip()]
            print(f"✓ Loaded {len(self.frame_ids):,} frames from {self.frames_list}")
        except FileNotFoundError:
            # Check for environment variable override
            env_frames = os.environ.get("FRAMES_LIST")
            if env_frames:
                try:
                    env_frames_path = Path(env_frames)
                    with open(env_frames_path) as f:
                        self.frame_ids = [line.strip() for line in f if line.strip()]
                    print(f"✓ Loaded {len(self.frame_ids):,} frames from environment override: {env_frames_path}")
                except Exception as e:
                    print(f"❌ Error loading frames from environment: {e}")
                    self.frame_ids = []
            else:
                self.frame_ids = sorted(self.zod_frames.get_all_ids())
                print(f"✓ Frames file not found, processing all {len(self.frame_ids):,} frames from ZOD dataset")
    
    def project_lidar_to_camera(self, points_3d, calibration, camera_enum):
        """Project LiDAR points to camera using ZOD's projection method"""
        try:
            # Get transformations
            t_refframe_to_lidar = calibration.lidars[Lidar.VELODYNE].extrinsics
            t_refframe_to_camera = calibration.cameras[camera_enum].extrinsics
            
            # Calculate transformation from LiDAR to camera
            t_camera_from_refframe = t_refframe_to_camera.inverse
            t_lidar_to_camera = Pose(t_camera_from_refframe.transform @ t_refframe_to_lidar.transform)
            
            # Transform points to camera frame
            camera_data = transform_points(points_3d, t_lidar_to_camera.transform)
            
            # Filter points with positive depth
            positive_depth = camera_data[:, 2] > 0
            camera_data = camera_data[positive_depth]
            
            if len(camera_data) == 0:
                return None, None
            
            # Filter by FOV
            camera_calib = calibration.cameras[camera_enum]
            camera_data_fov, fov_mask = get_points_in_camera_fov(
                camera_calib.field_of_view, camera_data
            )
            
            if len(camera_data_fov) == 0:
                return None, None
            
            # Project to 2D pixel coordinates
            xy_array = project_3d_to_2d_kannala(
                camera_data_fov,
                camera_calib.intrinsics[..., :3],
                camera_calib.distortion,
            )
            
            # Filter within image bounds (with margin)
            image_width = int(camera_calib.image_dimensions[0])
            image_height = int(camera_calib.image_dimensions[1])
            margin = 10
            
            valid_mask = (
                (xy_array[:, 0] >= margin) & 
                (xy_array[:, 0] < image_width - margin) &
                (xy_array[:, 1] >= margin) & 
                (xy_array[:, 1] < image_height - margin)
            )
            
            valid_points_2d = xy_array[valid_mask]
            valid_points_3d = camera_data_fov[valid_mask]
            
            return valid_points_2d, valid_points_3d
            
        except Exception as e:
            if self.verbose:
                print(f"    Error in projection: {e}")
            return None, None
    
    def create_clft_pkl(self, frame_id):
        """Create CLFT-format pkl data with 3D points and camera projections"""
        try:
            zod_frame = self.zod_frames[frame_id]
            
            # Get aggregated LiDAR points and raw lidar data
            image_timestamp = zod_frame.info.keyframe_time.timestamp()
            aggregated_lidar = zod_frame.get_aggregated_lidar(
                num_before=1, 
                num_after=1, 
                timestamp=image_timestamp
            )
            
            if aggregated_lidar is None:
                return None
            
            # Extract points from LidarData object
            if hasattr(aggregated_lidar, 'points'):
                points = aggregated_lidar.points
            elif hasattr(aggregated_lidar, 'data'):
                points = aggregated_lidar.data
            elif isinstance(aggregated_lidar, np.ndarray):
                points = aggregated_lidar
            else:
                return None
            
            if not (isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] >= 3):
                return None
            
            points_3d = points[:, :3].astype(np.float32)
            
            calibration = zod_frame.calibration
            
            # Collect projections from all cameras
            all_3d_points = []
            all_camera_coords = []
            all_class_instances = []
            
            for camera_enum, camera_id in self.camera_mapping.items():
                points_2d, points_3d_valid = self.project_lidar_to_camera(
                    points_3d, calibration, camera_enum
                )
                
                if points_2d is not None and len(points_2d) > 0:
                    # Create camera coordinates [camera_id, x_pixel, y_pixel]
                    x_pixels = np.clip(np.round(points_2d[:, 0]), 0, 65535).astype(np.uint16)
                    y_pixels = np.clip(np.round(points_2d[:, 1]), 0, 65535).astype(np.uint16)
                    
                    camera_coords = np.column_stack([
                        np.full(len(points_2d), camera_id, dtype=np.uint16),
                        x_pixels,
                        y_pixels
                    ])
                    
                    # Create class_instance array (zeros for now)
                    class_instance = np.zeros((len(points_2d), 2), dtype=np.uint8)
                    
                    all_3d_points.append(points_3d_valid.astype(np.float32))
                    all_camera_coords.append(camera_coords)
                    all_class_instances.append(class_instance)
            
            if not all_3d_points:
                return None
            
            # Combine all camera projections
            combined_3d_points = np.vstack(all_3d_points)
            combined_camera_coords = np.vstack(all_camera_coords)
            combined_class_instances = np.vstack(all_class_instances)
            
            # Create CLFT-compliant data structure
            clft_data = {
                '3d_points': combined_3d_points,              # (N, 3) float32
                'class_instance': combined_class_instances,   # (N, 2) uint8
                'camera_coordinates': combined_camera_coords  # (N, 3) uint16
            }
            
            return clft_data
            
        except Exception as e:
            if self.verbose:
                print(f"  Error creating pkl for {frame_id}: {e}")
            return None
    
    def process_all_frames(self):
        """Process all frames sequentially and create pkl files"""
        print(f"\nProcessing {len(self.frame_ids):,} frames...")
        print(f"Output: {self.lidar_dir}")
        
        success_count = 0
        error_count = 0
        skip_count = 0
        
        for frame_id in tqdm(self.frame_ids, desc="Generating LiDAR pkl files"):
            # Check if already exists
            pkl_path = self.lidar_dir / f"frame_{frame_id}.pkl"
            
            if pkl_path.exists():
                skip_count += 1
                continue
            
            try:
                # Create pkl data
                clft_data = self.create_clft_pkl(frame_id)
                
                if clft_data is None:
                    error_count += 1
                    continue
                
                # Save pkl
                with open(pkl_path, 'wb') as f:
                    pickle.dump(clft_data, f)
                
                success_count += 1
                
            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"\n  Error processing {frame_id}: {e}")
        
        print(f"\n{'='*60}")
        print(f"LIDAR PKL GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully processed: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        print(f"\nOutput: {self.lidar_dir}")
        print(f"\nPKL format:")
        print(f"  - 3d_points: (N, 3) float32 - XYZ coordinates")
        print(f"  - class_instance: (N, 2) uint8 - class/instance IDs")
        print(f"  - camera_coordinates: (N, 3) uint16 - [cam_id, x, y]")


def main():
    # Configuration
    ZOD_ROOT = Path("/media/tom/ml/zod-data")
    OUTPUT_ROOT = Path("/media/tom/ml/zod_temp")
    FRAMES_LIST = Path("/media/tom/ml/projects/clft-zod/frames_to_process.txt")
    
    print("="*60)
    print("LiDAR PKL Generator")
    print("="*60)
    
    generator = LiDARDataGenerator(
        zod_root=ZOD_ROOT,
        output_root=OUTPUT_ROOT,
        frames_list=FRAMES_LIST,
        verbose=False
    )
    
    generator.process_all_frames()


if __name__ == "__main__":
    main()
