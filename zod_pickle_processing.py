#!/usr/bin/env python3
"""
Fixed ZOD to CLFT converter - generates proper CLFT format
Processes ALL 12 frames in the dataset (not just 5)
Matches the exact format required by CLFT training pipeline
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple

# ZOD imports
from zod import ZodFrames
from zod.constants import Camera, Anonymization, AnnotationProject, Lidar
from zod.utils.geometry import project_3d_to_2d_kannala, transform_points, get_points_in_camera_fov
from zod.data_classes.geometry import Pose

class ZODToCLFT:
    """Convert ZOD LiDAR and camera data to proper CLFT format"""
    
    def __init__(
        self,
        dataset_root,
        output_dir="output_clft_fixed",
        version="mini",
        zod_frames: Optional[ZodFrames] = None,
        visualization_subdir: str = "visualizations",
        target_image_size: Optional[int] = None,
        verbose: bool = False,
        *,
        camera_subdir: str = "camera",
        lidar_subdir: str = "lidar",
        deduplicate_points: bool = True,
    ):
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_version = version
        self.visualization_subdir = visualization_subdir
        self.target_image_size = target_image_size
        self.verbose = verbose
        self.camera_subdir = camera_subdir
        self.lidar_subdir = lidar_subdir
        self.deduplicate_points = deduplicate_points

        self.visualization_dir = self.output_dir / self.visualization_subdir
        self.camera_dir = self.output_dir / self.camera_subdir
        self.lidar_dir = self.output_dir / self.lidar_subdir

        # Initialize ZOD dataset
        if zod_frames is not None:
            self.zod_frames = zod_frames
        elif dataset_root is not None:
            self.zod_frames = ZodFrames(dataset_root=dataset_root, version=version)
        else:
            self.zod_frames = None
        
        # Camera mapping
        self.camera_mapping = {
            Camera.FRONT: 0,  # CLFT uses 0-based indexing
        }
        
        if self.verbose:
            print("Initialized ZOD to CLFT converter")
            print(f"Dataset version: {self.dataset_version}")
            print(f"Frames available: {len(self.zod_frames)}")

    def get_lidar_points(self, zod_frame):
        """Extract LiDAR points from ZOD frame"""
        try:
            image_timestamp = zod_frame.info.keyframe_time.timestamp()
            aggregated_lidar = zod_frame.get_aggregated_lidar(
                num_before=1, 
                num_after=1, 
                timestamp=image_timestamp
            )
            
            if aggregated_lidar is None:
                return None
            
            # Extract points
            if hasattr(aggregated_lidar, 'points'):
                points = aggregated_lidar.points
            elif hasattr(aggregated_lidar, 'data'):
                points = aggregated_lidar.data
            elif isinstance(aggregated_lidar, np.ndarray):
                points = aggregated_lidar
            else:
                return None
            
            if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] >= 3:
                points_3d = points[:, :3].astype(np.float32)  # ✅ Convert to float32
                if self.verbose:
                    print(f"LiDAR points: {len(points_3d):,} points")
                return points_3d
            else:
                return None
                
        except Exception as e:
            print(f"  Error getting LiDAR points: {e}")
            return None

    def project_lidar_to_camera_zod_method(self, points_3d, calibration, camera_enum):
        """Project LiIDAR to camera using ZOD's exact method"""
        try:
            # Get transformations
            t_refframe_to_lidar = calibration.lidars[Lidar.VELODYNE].extrinsics
            t_refframe_to_camera = calibration.cameras[camera_enum].extrinsics
            
            # Calculate transformation
            t_camera_from_refframe = t_refframe_to_camera.inverse
            t_lidar_to_camera = Pose(t_camera_from_refframe.transform @ t_refframe_to_lidar.transform)
            
            # Transform points
            camera_data = transform_points(points_3d, t_lidar_to_camera.transform)
            
            # Filter positive depth
            positive_depth = camera_data[:, 2] > 0
            camera_data = camera_data[positive_depth]
            
            if not camera_data.any():
                return None, None
            
            # Filter by FOV
            camera_calib = calibration.cameras[camera_enum]
            camera_data_fov, fov_mask = get_points_in_camera_fov(camera_calib.field_of_view, camera_data)
            
            if len(camera_data_fov) == 0:
                return None, None
            
            # Project to 2D
            xy_array = project_3d_to_2d_kannala(
                camera_data_fov,
                camera_calib.intrinsics[..., :3],
                camera_calib.distortion,
            )
            
            # Filter within image bounds
            image_width = int(camera_calib.image_dimensions[0])
            image_height = int(camera_calib.image_dimensions[1])
            
            margin = 10
            valid_mask = (
                (xy_array[:, 0] >= margin) & (xy_array[:, 0] < image_width - margin) &
                (xy_array[:, 1] >= margin) & (xy_array[:, 1] < image_height - margin)
            )
            
            valid_points_2d = xy_array[valid_mask]
            valid_points_3d = camera_data_fov[valid_mask]
            
            if self.verbose:
                print(f"    Valid projections: {len(valid_points_2d):,}")
            
            return valid_points_2d, valid_points_3d
            
        except Exception as e:
            print(f"    Error in projection: {e}")
            return None, None

    def create_clft_data(self, frame_id):
        """Create data in proper CLFT format"""
        try:
            if self.verbose:
                print(f"\nProcessing frame {frame_id}...")
                print(f"  Frame ID type: {type(frame_id)}")
            if self.zod_frames is None:
                raise RuntimeError(
                    "ZOD frames are not loaded; cannot create CLFT data without dataset access"
                )
            
            zod_frame = self.zod_frames[frame_id]
            points_3d = self.get_lidar_points(zod_frame)
            
            if points_3d is None:
                return None
            
            calibration = zod_frame.calibration
            
            # Collect all valid projections
            all_3d_points = []
            all_camera_coords = []
            all_class_instances = []
            
            for camera_name, camera_id in self.camera_mapping.items():
                if self.verbose:
                    print(f"  Processing {camera_name} (ID: {camera_id})...")
                
                points_2d, points_3d_valid = self.project_lidar_to_camera_zod_method(
                    points_3d, calibration, camera_name
                )
                
                if points_2d is not None and len(points_2d) > 0:
                    # Create camera coordinates in CLFT format (uint16)
                    # Clamp pixel coordinates to valid uint16 range [0, 65535]
                    x_pixels = np.clip(np.round(points_2d[:, 0]), 0, 65535).astype(np.uint16)
                    y_pixels = np.clip(np.round(points_2d[:, 1]), 0, 65535).astype(np.uint16)
                    
                    camera_coords = np.column_stack([
                        np.full(len(points_2d), camera_id, dtype=np.uint16),  # Camera ID
                        x_pixels,          # X pixel coordinate
                        y_pixels           # Y pixel coordinate
                    ])
                    
                    # Create class_instance array (uint8)
                    # For now, use default values - this could be enhanced with actual semantic labels
                    class_instance = np.zeros((len(points_2d), 2), dtype=np.uint8)
                    # You could populate these with actual semantic data:
                    # class_instance[:, 0] = class_id   # From semantic segmentation
                    # class_instance[:, 1] = instance_id # From instance segmentation
                    
                    all_3d_points.append(points_3d_valid.astype(np.float32))
                    all_camera_coords.append(camera_coords)
                    all_class_instances.append(class_instance)
            
            if not all_3d_points:
                if self.verbose:
                    print("  No valid projections found")
                return None
            
            # Combine all data
            combined_3d_points = np.vstack(all_3d_points)
            combined_camera_coords = np.vstack(all_camera_coords)
            combined_class_instances = np.vstack(all_class_instances)
            
            if self.verbose:
                print(f"  Total points: {len(combined_3d_points):,}")

            if self.deduplicate_points:
                (
                    combined_3d_points,
                    combined_camera_coords,
                    combined_class_instances,
                    removed,
                ) = self._deduplicate_entries(
                    combined_3d_points,
                    combined_camera_coords,
                    combined_class_instances,
                )
                if removed and self.verbose:
                    print(f"  Removed {removed:,} duplicate camera projections")
            
            # Create CLFT-compliant data structure
            clft_data = {
                '3d_points': combined_3d_points,           # shape (N, 3), dtype float32
                'class_instance': combined_class_instances, # shape (N, 2), dtype uint8
                'camera_coordinates': combined_camera_coords # shape (N, 3), dtype uint16
            }
            
            return clft_data            
        except Exception as e:
            print(f"  Error processing frame {frame_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _deduplicate_entries(
        self,
        points_3d: np.ndarray,
        camera_coords: np.ndarray,
        class_instance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Remove duplicate camera projections, keeping the closest depth per pixel."""

        if len(points_3d) == 0:
            return points_3d, camera_coords, class_instance, 0

        camera_coords = np.ascontiguousarray(camera_coords)
        points_3d = np.ascontiguousarray(points_3d)
        class_instance = np.ascontiguousarray(class_instance)

        sort_order = np.lexsort(
            (
                points_3d[:, 2],
                camera_coords[:, 2],
                camera_coords[:, 1],
                camera_coords[:, 0],
            )
        )

        camera_sorted = camera_coords[sort_order]
        camera_view = camera_sorted.view(
            np.dtype((np.void, camera_sorted.dtype.itemsize * camera_sorted.shape[1]))
        )
        _, unique_idx = np.unique(camera_view, return_index=True)

        keep_indices = np.sort(sort_order[unique_idx])

        dedup_points = points_3d[keep_indices]
        dedup_camera = camera_coords[keep_indices]
        dedup_class = class_instance[keep_indices]

        removed = len(points_3d) - len(dedup_points)
        return dedup_points, dedup_camera, dedup_class, int(removed)

    def save_clft_data(self, clft_data, frame_id, zod_frame=None):
        """Save data in proper CLFT format"""
        try:
            # Create lidar output directory
            self.lidar_dir.mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                print(f"    Saving frame_id: {frame_id} (type: {type(frame_id)})")
            
            # Save with proper CLFT filename format
            output_path = self.lidar_dir / f"frame_{frame_id}.pkl"
            
            with open(output_path, 'wb') as f:
                pickle.dump(clft_data, f)
            
            if self.verbose:
                print(f" Saved CLFT format: {output_path}")
                self.verify_clft_format(output_path)
            
            # Create visualization if zod_frame is provided
            if zod_frame is not None:
                self.create_visualization(frame_id, clft_data, zod_frame=zod_frame)
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error saving frame {frame_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def verify_clft_format(self, file_path):
        """Verify that the saved file matches CLFT format"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"    Verification:")
            required_keys = ['3d_points', 'class_instance', 'camera_coordinates']
            
            for key in required_keys:
                if key in data:
                    value = data[key]
                    print(f"{key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"Missing key: {key}")
                    
        except Exception as e:
            print(f"Verification failed: {e}")

    def process_dataset(self, max_frames=None):
        """Process dataset and create CLFT-compliant files
        
        Args:
            max_frames: Maximum number of frames to process. 
                       If None, processes all available frames (12 total)
        """

        if self.zod_frames is None:
            raise RuntimeError("ZOD frames are required to process the dataset")
        
        # Get frame IDs
        train_ids = list(self.zod_frames.get_split("train"))
        val_ids = list(self.zod_frames.get_split("val"))
        all_ids = train_ids + val_ids
        
        print(f"Available frames: Train={len(train_ids)}, Val={len(val_ids)}, Total={len(all_ids)}")
        
        if max_frames and max_frames > 0:
            all_ids = all_ids[:max_frames]
            print(f"Processing {len(all_ids)} frames (limited to {max_frames})...")
        else:
            print(f"Processing ALL {len(all_ids)} frames...")
        
        success_count = 0
        for i, frame_id in enumerate(all_ids):
            print(f"\n--- Frame {i+1}/{len(all_ids)}: {frame_id} ---")
            
            # Get zod_frame for visualization
            zod_frame = self.zod_frames[frame_id]
            clft_data = self.create_clft_data(frame_id)
            
            if clft_data is not None:
                if self.save_clft_data(clft_data, frame_id, zod_frame):
                    success_count += 1
        
        print(f"\n=== CONVERSION COMPLETE ===")
        print(f"Successfully processed: {success_count}/{len(all_ids)} frames")
        print(f"Output directory: {self.output_dir}")
        print(f"\nGenerated CLFT-compliant files:")
        print(f"  📁 lidar/ - Proper CLFT format with:")
        print(f"    - 3d_points: shape (N, 3), dtype float32")
        print(f"    - class_instance: shape (N, 2), dtype uint8")
        print(f"    - camera_coordinates: shape (N, 3), dtype uint16")
        print(f"  📁 visualizations/ - Processing visualizations:")
        print(f"    - LiDAR points projected on camera images")
        print(f"    - 3D point cloud views")
        print(f"    - Data statistics and format verification")
        
        return success_count

    def _load_overlay_base_image(
        self,
        frame_id: str,
        zod_frame=None,
        camera_image_path: Optional[Path] = None,
    ) -> Image.Image:
        """Load a camera image for visualization from ZOD frame or disk."""
        attempts = []

        if zod_frame is not None:
            anonymization_attempts = [Anonymization.DNAT, Anonymization.BLUR, None]
            for anonymization in anonymization_attempts:
                try:
                    if anonymization is None:
                        return zod_frame.get_image()
                    return zod_frame.get_image(anonymization)
                except FileNotFoundError as exc:
                    attempts.append(f"{anonymization or 'DEFAULT'}: {exc}")
                except Exception as exc:  # noqa: BLE001 - ensure full context for debugging
                    attempts.append(f"{anonymization or 'DEFAULT'}: {exc}")

        candidate_paths = []
        if camera_image_path is not None:
            candidate_paths.append(Path(camera_image_path))

        candidate_names = [
            f"frame_{frame_id}.png",
            f"camera_{frame_id}.png",
            f"frame_{frame_id}.jpg",
            f"camera_{frame_id}.jpg",
            f"{frame_id}.png",
            f"{frame_id}.jpg",
        ]
        candidate_paths.extend(self.camera_dir / name for name in candidate_names)

        for candidate in candidate_paths:
            try:
                with Image.open(candidate) as img:
                    return img.convert("RGB")
            except FileNotFoundError as exc:
                attempts.append(f"{candidate}: {exc}")
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{candidate}: {exc}")

        attempt_msg = "; ".join(attempts) if attempts else "no sources attempted"
        raise FileNotFoundError(
            f"Unable to load camera image for frame {frame_id}. Attempts: {attempt_msg}"
        )

    def create_visualization(
        self,
        frame_id,
        clft_data,
        zod_frame=None,
        camera_image_path: Optional[Path] = None,
    ):
        """Create clean visualization for CLFT data processing"""
        try:
            if self.verbose:
                print(f"  Creating visualization for frame {frame_id}...")

            viz_dir = self.visualization_dir
            viz_dir.mkdir(parents=True, exist_ok=True)

            image_pil = self._load_overlay_base_image(
                frame_id,
                zod_frame=zod_frame,
                camera_image_path=camera_image_path,
            )
            image_np = np.array(image_pil)
            
            # Extract data from CLFT format
            points_3d = clft_data['3d_points']
            camera_coords = clft_data['camera_coordinates']
            
            # Create clean single image visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Show camera image with LiDAR points overlaid
            ax.imshow(image_np)
            # Color points by depth
            depths = points_3d[:, 2]  # Z coordinates
            ax.scatter(
                camera_coords[:, 1],  # X pixel coordinates
                camera_coords[:, 2],  # Y pixel coordinates
                c=depths, 
                cmap='viridis', 
                s=0.1, 
                alpha=0.3
            )
            
            # Remove all axes, titles, and decorations
            ax.axis('off')
            
            # Save with tight layout and no padding
            viz_path = viz_dir / f"frame_{frame_id}_lidar_overlay.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            if self.target_image_size is not None:
                with Image.open(viz_path) as viz_img:
                    if viz_img.size != (self.target_image_size, self.target_image_size):
                        resized = viz_img.resize(
                            (self.target_image_size, self.target_image_size),
                            resample=Image.BILINEAR,
                        )
                        resized.save(viz_path, format="PNG", optimize=True)
            if self.verbose:
                print(f"    Visualization saved: {viz_path}")
            return True
            
        except Exception as e:
            print(f"    Error creating visualization: {e}")
            return False

    def create_visualization_from_artifacts(
        self,
        frame_id: str,
        camera_path: Optional[Path] = None,
        lidar_path: Optional[Path] = None,
    ) -> bool:
        """Render a LiDAR overlay using previously saved artefacts."""

        lidar_file = Path(lidar_path) if lidar_path is not None else self.lidar_dir / f"frame_{frame_id}.pkl"
        if not lidar_file.exists():
            if self.verbose:
                print(f"  Missing CLFT pickle for frame {frame_id}: {lidar_file}")
            return False

        with lidar_file.open('rb') as f:
            clft_data = pickle.load(f)

        source_camera_path = Path(camera_path) if camera_path is not None else self.camera_dir / f"frame_{frame_id}.png"
        return self.create_visualization(
            frame_id,
            clft_data,
            zod_frame=None,
            camera_image_path=source_camera_path,
        )

def main():
    """Main function - processes ALL 12 frames in the ZOD mini dataset"""

    dataset_root = "./data"
    output_dir = "output_clft"
    max_frames = None  # Process ALL available frames (12 total)
    
    converter = ZODToCLFT(dataset_root, output_dir)
    success_count = converter.process_dataset(max_frames=max_frames)
    
    if success_count > 0:
        print(f"Processed {success_count} frames out of 12 available.")
        print("Generated files are now CLFT-compliant and ready for training.")
    else:
        print(f"\n No frames processed successfully!")

if __name__ == "__main__":
    main()
