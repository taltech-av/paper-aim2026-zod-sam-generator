#!/usr/bin/env python3
"""
Generate enhanced annotations by merging SAM camera annotations with LiDAR data.

Logic:
1. Keep all SAM camera annotations (Vehicle, Sign, Cyclist, Pedestrian)
2. Add "Ignore" class (5) where LiDAR points exist but no camera annotation
3. Background (0) remains where neither camera nor LiDAR detects anything

Classes:
    0: Background
    1: Vehicle
    2: Sign  
    3: Cyclist
    4: Pedestrian
    5: Ignore (LiDAR-only regions)
"""

import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import argparse


class AnnotationMerger:
    def __init__(self, output_root, frames_list):
        """Initialize annotation merger
        
        Args:
            output_root: Path to output_clft_v2 directory
            frames_list: Path to frames_to_process.txt
        """
        self.output_root = Path(output_root)
        self.frames_list = Path(frames_list)
        
        # Input directories
        self.camera_dir = self.output_root / "camera"
        self.sam_annotation_dir = self.output_root / "annotation_sam"  # Renamed from annotation
        self.lidar_dir = self.output_root / "lidar"
        
        # Output directory
        self.merged_annotation_dir = self.output_root / "annotation"  # Renamed from annotation_merged
        self.merged_annotation_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping
        self.CLASSES = {
            0: 'background',
            1: 'vehicle',
            2: 'sign',
            3: 'cyclist',
            4: 'pedestrian',
            5: 'ignore',  # LiDAR-only regions
        }
        
        # Load frames to process
        with open(self.frames_list) as f:
            self.frame_ids = [line.strip() for line in f if line.strip()]
        
        # Load already processed frames to skip them
        self.processed_frames_file = self.output_root / "processed_merged_frames.txt"
        self.processed_frames = set()
        if self.processed_frames_file.exists():
            with open(self.processed_frames_file) as f:
                self.processed_frames = set(line.strip() for line in f if line.strip())
        
        # Filter out already processed frames
        original_count = len(self.frame_ids)
        self.frame_ids = [fid for fid in self.frame_ids if fid not in self.processed_frames]
        
        print(f"✓ Loaded {original_count:,} frames to process")
        print(f"✓ Already processed: {len(self.processed_frames):,}")
        print(f"✓ Remaining to process: {len(self.frame_ids):,}")
    
    def load_sam_annotation(self, frame_id):
        """Load SAM camera annotation"""
        anno_path = self.sam_annotation_dir / f"frame_{frame_id}.png"
        if not anno_path.exists():
            return None
        
        annotation = cv2.imread(str(anno_path), cv2.IMREAD_GRAYSCALE)
        return annotation
    
    def load_lidar_pkl(self, frame_id, target_shape=None, return_distances=False):
        """Load LiDAR pkl file and extract front camera projections
        
        Args:
            frame_id: Frame identifier
            target_shape: (height, width) to scale LiDAR points to match resized images
            return_distances: If True, also return distances for each point
        
        Returns:
            If return_distances=False: (N, 2) array of [x, y] pixel coordinates
            If return_distances=True: tuple of ((N, 2) coordinates, (N,) distances)
        """
        pkl_path = self.lidar_dir / f"frame_{frame_id}.pkl"
        if not pkl_path.exists():
            return None if not return_distances else (None, None)
        
        try:
            with open(pkl_path, 'rb') as f:
                clft_data = pickle.load(f)
            
            # Extract camera coordinates and filter for front camera (ID=0)
            camera_coords = clft_data['camera_coordinates']
            front_mask = camera_coords[:, 0] == 0
            
            if not np.any(front_mask):
                return None if not return_distances else (None, None)
            
            # Get pixel coordinates for front camera (in original ZOD resolution)
            front_coords = camera_coords[front_mask]
            x_coords = front_coords[:, 1].astype(np.float32)  # X pixel
            y_coords = front_coords[:, 2].astype(np.float32)  # Y pixel
            
            lidar_points = np.column_stack([x_coords, y_coords])
            
            # Get distances if requested
            distances = None
            if return_distances:
                # Extract 3D points for distance calculation
                points_3d = clft_data['3d_points'][front_mask]
                distances = np.sqrt(np.sum(points_3d[:, :3]**2, axis=1))  # Euclidean distance
            
            # Scale to target shape if provided
            if target_shape is not None and len(lidar_points) > 0:
                # ZOD original resolution for front camera is 3848x2168
                orig_width = 3848
                orig_height = 2168
                
                # Calculate scale factors
                target_height, target_width = target_shape
                scale_x = target_width / orig_width
                scale_y = target_height / orig_height
                
                # Scale coordinates
                lidar_points[:, 0] *= scale_x
                lidar_points[:, 1] *= scale_y
            
            if return_distances:
                return lidar_points, distances
            return lidar_points
            
        except Exception as e:
            print(f"  Error loading LiDAR pkl for {frame_id}: {e}")
            return None if not return_distances else (None, None)
    
    def get_distance_color(self, distance, min_dist=0, max_dist=80):
        """Get color for a distance value using cyan-to-blue gradient
        
        Args:
            distance: Distance in meters
            min_dist: Minimum distance for color mapping
            max_dist: Maximum distance for color mapping
        
        Returns:
            (B, G, R) color tuple for OpenCV
        """
        # Clamp distance to range
        distance = np.clip(distance, min_dist, max_dist)
        
        # Normalize to 0-1
        normalized = (distance - min_dist) / (max_dist - min_dist)
        
        # Cyan (close) to Blue (far) gradient
        # Close: Cyan (0, 255, 255) - BGR format
        # Far: Dark Blue (139, 0, 0) - BGR format
        r = int(255 * (1 - normalized))
        g = int(255 * (1 - normalized * 0.6))  # Gradual decrease
        b = int(139 + 116 * (1 - normalized))  # 139 to 255
        
        return (b, g, r)
    
    def create_lidar_mask(self, lidar_points, shape, radius=3):
        """Create a binary mask from LiDAR points
        
        Args:
            lidar_points: (N, 2) array of [x, y] pixel coordinates
            shape: (height, width) of the output mask
            radius: Radius around each LiDAR point to mark as LiDAR-covered
        
        Returns:
            Binary mask where 1 = LiDAR coverage, 0 = no LiDAR
        """
        mask = np.zeros(shape, dtype=np.uint8)
        
        if lidar_points is None or len(lidar_points) == 0:
            return mask
        
        # Vectorized approach for speed
        lidar_points = lidar_points.astype(np.int32)
        
        # Filter points within bounds
        valid_mask = (
            (lidar_points[:, 0] >= 0) & (lidar_points[:, 0] < shape[1]) &
            (lidar_points[:, 1] >= 0) & (lidar_points[:, 1] < shape[0])
        )
        valid_points = lidar_points[valid_mask]
        
        # Draw circles for each point
        for point in valid_points:
            x, y = point[0], point[1]
            cv2.circle(mask, (x, y), radius, 1, -1)
        
        return mask
    
    def merge_annotations(self, frame_id):
        """Merge SAM annotation with LiDAR data
        
        Logic:
        1. Start with SAM camera annotation (classes 1-4)
        2. Create LiDAR mask with scaled coordinates
        3. Where LiDAR exists but annotation is background (0), set to ignore (5)
        4. Keep all other SAM annotations unchanged
        """
        try:
            # Load SAM annotation
            sam_annotation = self.load_sam_annotation(frame_id)
            if sam_annotation is None:
                return None
            
            # Load LiDAR points with scaling to match annotation size
            lidar_points = self.load_lidar_pkl(frame_id, target_shape=sam_annotation.shape)
            
            # Start with SAM annotation
            merged_annotation = sam_annotation.copy()
            
            # If we have LiDAR data, add ignore class
            if lidar_points is not None:
                # Create LiDAR coverage mask
                lidar_mask = self.create_lidar_mask(
                    lidar_points, 
                    sam_annotation.shape,
                    radius=3  # 3-pixel radius around each LiDAR point
                )
                
                # Where LiDAR exists (1) AND annotation is background (0), set to ignore (5)
                ignore_regions = (lidar_mask == 1) & (sam_annotation == 0)
                merged_annotation[ignore_regions] = 5
            
            return merged_annotation
            
        except Exception as e:
            print(f"  Error merging annotations for {frame_id}: {e}")
            return None
    
    def process_all_frames(self, create_vis=False):
        """Process all frames and create merged annotations
        
        Args:
            create_vis: Whether to create visualizations for ALL frames
        """
        print(f"\nMerging {len(self.frame_ids):,} frames...")
        print(f"Input: {self.sam_annotation_dir}")
        print(f"       {self.lidar_dir}")
        print(f"Output: {self.merged_annotation_dir}")
        if create_vis:
            print(f"Visualizations: ALL frames")
        
        success_count = 0
        error_count = 0
        skip_count = 0
        vis_count = 0
        
        # Prepare visualization directory if needed
        if create_vis:
            vis_dir = self.output_root / "visualizations" / "annotation"  # Updated to match new folder name
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Color map for visualization
            color_map = {
                0: np.array([0, 0, 0], dtype=np.uint8),          # Background - Black
                1: np.array([255, 0, 0], dtype=np.uint8),        # Vehicle - Red
                2: np.array([255, 255, 0], dtype=np.uint8),      # Sign - Yellow
                3: np.array([255, 0, 255], dtype=np.uint8),      # Cyclist - Magenta
                4: np.array([0, 255, 0], dtype=np.uint8),        # Pedestrian - Green
                5: np.array([128, 128, 128], dtype=np.uint8),    # Ignore - Gray
            }
        
        for idx, frame_id in enumerate(tqdm(self.frame_ids, desc="Merging annotations")):
            # Check if already exists
            output_path = self.merged_annotation_dir / f"frame_{frame_id}.png"
            if output_path.exists():
                skip_count += 1
                
                # Still create visualization if requested and doesn't exist
                if create_vis:
                    vis_path = vis_dir / f"frame_{frame_id}.png"
                    if not vis_path.exists():
                        try:
                            merged_annotation = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                            camera_path = self.camera_dir / f"frame_{frame_id}.png"
                            if camera_path.exists() and merged_annotation is not None:
                                camera_img = cv2.imread(str(camera_path))
                                if camera_img is not None:
                                    h, w = merged_annotation.shape
                                    colored = np.zeros((h, w, 3), dtype=np.uint8)
                                    for class_id, color in color_map.items():
                                        colored[merged_annotation == class_id] = color
                                    overlay = cv2.addWeighted(camera_img, 0.6, colored, 0.4, 0)
                                    
                                    # Overlay LiDAR points with distance-based colors (semi-transparent)
                                    lidar_data = self.load_lidar_pkl(frame_id, target_shape=(h, w), return_distances=True)
                                    if lidar_data is not None and lidar_data[0] is not None:
                                        lidar_points, distances = lidar_data
                                        
                                        # Create a separate layer for LiDAR points
                                        lidar_layer = overlay.copy()
                                        
                                        # Draw each LiDAR point with distance-based color (smaller circles)
                                        for point, dist in zip(lidar_points, distances):
                                            x, y = int(point[0]), int(point[1])
                                            if 0 <= x < w and 0 <= y < h:
                                                color = self.get_distance_color(dist)
                                                cv2.circle(lidar_layer, (x, y), 1, color, -1)  # Smaller: radius=1
                                        
                                        # Blend LiDAR points with 30% opacity so objects show through
                                        overlay = cv2.addWeighted(overlay, 0.7, lidar_layer, 0.3, 0)
                                    
                                    cv2.imwrite(str(vis_path), overlay)
                                    vis_count += 1
                        except Exception:
                            pass
                continue
            
            # Merge annotations
            merged_annotation = self.merge_annotations(frame_id)
            
            if merged_annotation is None:
                error_count += 1
                continue
            
            # Save merged annotation
            cv2.imwrite(str(output_path), merged_annotation)
            success_count += 1
            
            # Mark frame as processed
            self.processed_frames.add(frame_id)
            with open(self.processed_frames_file, 'a') as f:
                f.write(f"{frame_id}\n")
            
            # Create visualization if requested
            if create_vis:
                try:
                    # Load camera image
                    camera_path = self.camera_dir / f"frame_{frame_id}.png"
                    if camera_path.exists():
                        camera_img = cv2.imread(str(camera_path))
                        if camera_img is not None:
                            # Create colored annotation
                            h, w = merged_annotation.shape
                            colored = np.zeros((h, w, 3), dtype=np.uint8)
                            
                            for class_id, color in color_map.items():
                                colored[merged_annotation == class_id] = color
                            
                            # Create overlay
                            overlay = cv2.addWeighted(camera_img, 0.6, colored, 0.4, 0)
                            
                            # Overlay LiDAR points with distance-based colors (semi-transparent)
                            lidar_data = self.load_lidar_pkl(frame_id, target_shape=(h, w), return_distances=True)
                            if lidar_data is not None and lidar_data[0] is not None:
                                lidar_points, distances = lidar_data
                                
                                # Create a separate layer for LiDAR points
                                lidar_layer = overlay.copy()
                                
                                # Draw each LiDAR point with distance-based color (smaller circles)
                                for point, dist in zip(lidar_points, distances):
                                    x, y = int(point[0]), int(point[1])
                                    if 0 <= x < w and 0 <= y < h:
                                        color = self.get_distance_color(dist)
                                        cv2.circle(lidar_layer, (x, y), 1, color, -1)  # Smaller: radius=1
                                
                                # Blend LiDAR points with 30% opacity so objects show through
                                overlay = cv2.addWeighted(overlay, 0.7, lidar_layer, 0.3, 0)
                            
                            # Add label with class counts
                            class_counts = {class_id: np.sum(merged_annotation == class_id) 
                                          for class_id in range(6)}
                            label_text = f"Frame {frame_id}"
                            cv2.putText(overlay, label_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                            # Add class legend
                            y_offset = 60
                            for class_id in [1, 2, 3, 4, 5]:
                                if class_counts[class_id] > 0:
                                    color = color_map[class_id].tolist()
                                    class_name = self.CLASSES[class_id]
                                    count = class_counts[class_id]
                                    text = f"{class_name}: {count:,} px"
                                    cv2.putText(overlay, text, (10, y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    y_offset += 25
                            
                            # Add LiDAR distance legend
                            if lidar_data is not None and lidar_data[0] is not None:
                                y_offset += 10
                                cv2.putText(overlay, "LiDAR Distance:", (10, y_offset),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                y_offset += 20
                                
                                # Show color gradient samples
                                for dist in [5, 20, 40, 60, 80]:
                                    color = self.get_distance_color(dist)
                                    text = f"  {dist}m"
                                    cv2.putText(overlay, text, (10, y_offset),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                                    y_offset += 18
                            
                            # Save visualization
                            vis_path = vis_dir / f"merged_{frame_id}.png"
                            cv2.imwrite(str(vis_path), overlay)
                            vis_count += 1
                except Exception as e:
                    pass  # Skip visualization errors
        
        print(f"\n{'='*60}")
        print(f"ANNOTATION MERGING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully merged: {success_count:,}")
        print(f"⊙ Skipped (existing): {skip_count:,}")
        print(f"✗ Errors: {error_count:,}")
        if create_vis:
            print(f"📊 Visualizations created: {vis_count:,}")
        print(f"\nOutput:")
        print(f"  📁 {self.merged_annotation_dir}/ - Merged annotations")
        if create_vis:
            print(f"  📁 {vis_dir}/ - Visualizations")
        print(f"\nClass mapping:")
        for class_id, class_name in self.CLASSES.items():
            print(f"  {class_id}: {class_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge SAM annotations with LiDAR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Class mapping:
  0: Background
  1: Vehicle (camera annotation)
  2: Sign (camera annotation)
  3: Cyclist (camera annotation)
  4: Pedestrian (camera annotation)
  5: Ignore (LiDAR exists, no camera annotation)

Logic:
  - Keep all SAM camera annotations (classes 1-4)
  - Add "Ignore" class (5) where LiDAR points exist but no camera annotation
  - Background (0) remains where neither camera nor LiDAR detects anything
        """
    )
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations for ALL frames (alongside merged annotations)')
    args = parser.parse_args()
    
    
    # Configuration
    OUTPUT_ROOT = Path("/media/tom/ml/projects/clft-zod/output_clft_v2")
    FRAMES_LIST = Path("/media/tom/ml/projects/clft-zod/frames_to_process.txt")
    
    print("="*60)
    print("SAM + LiDAR Annotation Merger")
    print("="*60)
    
    merger = AnnotationMerger(
        output_root=OUTPUT_ROOT,
        frames_list=FRAMES_LIST
    )
    
    # Process all frames with optional visualization
    merger.process_all_frames(create_vis=args.visualize)


if __name__ == "__main__":
    main()
